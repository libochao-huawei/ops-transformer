// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

#include <algorithm>
#include <cstring>
#include <vector>
#include <torch/extension.h>
#include <acl/acl_rt.h>
#include "aclnn_common.h"

namespace op_api {
using npu_utils = at_npu::native::NpuUtils;
const int DIM_ONE = 1;
const int DIM_TWO = 2;
const int DIM_THREE = 3;

namespace {
constexpr int64_t ATTN_FFN_STATUS_TABLE_SLOTS = 4; // 32B: [0]=completed round, [1..3] reserved

#define ACL_RT_CHECK(expr) \
    do { \
        aclError _s = (expr); \
        TORCH_CHECK(_s == ACL_SUCCESS, #expr " failed, ret=", static_cast<int64_t>(_s)); \
    } while (0)
} // namespace

// Host-visible send-completion status table backing the async tensor keepalive of
// attention_to_ffn: the kernel publishes the highest fully-completed round into slot 0 of this
// registered (device-mapped) memory, and the Python side polls it without any device
// synchronization to release keepalive tensor references late. Mirrors HostPinnedCounter in
// elastic_buffer.cpp. Layout contract with the kernel (ReportSendCompletion):
//   slot 0 (int64): completed round id, monotonically non-decreasing, -1 until the second
//                   kernel launch confirms the first round;
//   slot 1..3     : mirror slot 0 (only slot 0 is defined to the reader).
class AttentionToFfnHostStatusTable {
public:
    AttentionToFfnHostStatusTable()
    {
        constexpr int64_t bytes = ATTN_FFN_STATUS_TABLE_SLOTS * static_cast<int64_t>(sizeof(int64_t));
        ACL_RT_CHECK(aclrtMallocHost(&hostPtr_, bytes));
        // The destructor does not run when the constructor throws (pybind11 surfaces the error to
        // Python and a retry would accumulate leaks), so each step rolls back everything the
        // previous steps acquired before re-raising.
        aclError ret = aclrtHostRegisterV2(hostPtr_, bytes, ACL_HOST_REG_MAPPED);
        if (ret != ACL_SUCCESS) {
            (void)aclrtFreeHost(hostPtr_);
            hostPtr_ = nullptr;
            TORCH_CHECK(false, "aclrtHostRegisterV2 failed, ret=", static_cast<int64_t>(ret));
        }
        ret = aclrtHostGetDevicePointer(hostPtr_, &devPtr_, 0);
        if (ret != ACL_SUCCESS) {
            (void)aclrtHostUnregister(hostPtr_);
            (void)aclrtFreeHost(hostPtr_);
            hostPtr_ = nullptr;
            devPtr_ = nullptr;
            TORCH_CHECK(false, "aclrtHostGetDevicePointer failed, ret=", static_cast<int64_t>(ret));
        }
        Reset();
    }

    ~AttentionToFfnHostStatusTable()
    {
        if (hostPtr_ == nullptr) {
            return;
        }
        try {
            ACL_RT_CHECK(aclrtSynchronizeDevice());
            ACL_RT_CHECK(aclrtHostUnregister(hostPtr_));
            ACL_RT_CHECK(aclrtFreeHost(hostPtr_));
        } catch (const std::exception& e) {
            ASCEND_LOGE("AttentionToFfnHostStatusTable destroy failed: %s", e.what());
        }
        hostPtr_ = nullptr;
        devPtr_ = nullptr;
    }

    AttentionToFfnHostStatusTable(const AttentionToFfnHostStatusTable&) = delete;
    AttentionToFfnHostStatusTable& operator=(const AttentionToFfnHostStatusTable&) = delete;

    void Reset()
    {
        auto* slots = reinterpret_cast<volatile int64_t*>(hostPtr_);
        slots[0] = -1; // no completed round yet
        for (int64_t i = 1; i < ATTN_FFN_STATUS_TABLE_SLOTS; ++i) {
            slots[i] = 0;
        }
    }

    int64_t CompletedRound() const
    {
        return *reinterpret_cast<volatile int64_t*>(hostPtr_);
    }

    uintptr_t DevicePtr() const
    {
        return reinterpret_cast<uintptr_t>(devPtr_);
    }

private:
    void* hostPtr_ = nullptr;
    void* devPtr_ = nullptr;
};

/**
 * @brief Wrapper for attention_to_ffn. The op has no host-visible output tensors;
 *        data is sent to peer ranks via the HCCL window. We therefore return an empty
 *        tensor so PyTorch's dispatcher contract (at least one output) is satisfied.
 */
void NpuAttentionToFfn(const at::Tensor& context, const at::Tensor& x, const at::Tensor& sessionId,
                       const at::Tensor& microBatchId, const at::Tensor& layerId, const at::Tensor& expertIds,
                       const at::Tensor& expertRankTable, const c10::optional<at::Tensor>& scalesOptional,
                       const c10::optional<at::Tensor>& activeMaskOptional, std::string group, int64_t worldSize,
                       const std::vector<int64_t>& ffnTokenInfoTableShape,
                       const std::vector<int64_t>& ffnTokenDataShape,
                       const std::vector<int64_t>& attnTokenInfoTableShape, int64_t moeExpertNum, int64_t quantMode,
                       int64_t syncFlag, int64_t ffnStartRankId, int64_t cclBufferSize)
{
    TORCH_CHECK((x.dim() == DIM_THREE), "The x should be 3D, current dim is: ", x.dim());
    TORCH_CHECK((sessionId.dim() == DIM_ONE) && (microBatchId.dim() == DIM_ONE) && (layerId.dim() == DIM_ONE),
                "sessionId, microBatchId, layerId should be 1D.");
    TORCH_CHECK((expertIds.dim() == DIM_THREE), "The expertIds should be 3D, current dim is: ", expertIds.dim());
    TORCH_CHECK((expertRankTable.dim() == DIM_THREE),
                "The expertRankTable should be 3D, current dim is: ", expertRankTable.dim());
    TORCH_CHECK((worldSize > 0), "The world_size should be greater than 0, current is: ", worldSize);
    TORCH_CHECK(!group.empty(), "group should not be empty.");
    TORCH_CHECK((ffnTokenInfoTableShape.size() == 3),
                "ffn_token_info_table_shape should be 3D, current size is: ", ffnTokenInfoTableShape.size());
    TORCH_CHECK((ffnTokenDataShape.size() == 5),
                "ffn_token_data_shape should be 5D, current size is: ", ffnTokenDataShape.size());
    TORCH_CHECK((attnTokenInfoTableShape.size() == 3),
                "attn_token_info_table_shape should be 3D, current size is: ", attnTokenInfoTableShape.size());
    TORCH_CHECK((moeExpertNum > 0), "The moe_expert_num should be greater than 0, current is: ", moeExpertNum);
    TORCH_CHECK(((x.scalar_type() == at::kBFloat16) || (x.scalar_type() == at::kHalf)),
                "dtype of x should be bfloat16 or float16, but got ", x.scalar_type());
    TORCH_CHECK((sessionId.scalar_type() == at::kInt), "dtype of sessionId should be int32.");
    TORCH_CHECK((microBatchId.scalar_type() == at::kInt), "dtype of microBatchId should be int32.");
    TORCH_CHECK((layerId.scalar_type() == at::kInt), "dtype of layerId should be int32.");
    TORCH_CHECK((expertIds.scalar_type() == at::kInt), "dtype of expertIds should be int32.");
    TORCH_CHECK((expertRankTable.scalar_type() == at::kInt), "dtype of expertRankTable should be int32.");

    // Set DeviceGuard on x (the first input tensor that lives on NPU)
    auto localDevice = c10::Device(x.device());
    const c10::OptionalDeviceGuard deviceGuard(localDevice);

    std::string groupStr = std::string(group);
    char* groupPtr = const_cast<char*>(groupStr.c_str());

    at::IntArrayRef ffnTokenInfoTableShapeRef(ffnTokenInfoTableShape);
    at::IntArrayRef ffnTokenDataShapeRef(ffnTokenDataShape);
    at::IntArrayRef attnTokenInfoTableShapeRef(attnTokenInfoTableShape);

    ACLNN_CMD(aclnnAttentionToFfnV2, context, x, sessionId, microBatchId, layerId, expertIds, expertRankTable,
              scalesOptional, activeMaskOptional, groupPtr, worldSize, ffnTokenInfoTableShapeRef, ffnTokenDataShapeRef,
              attnTokenInfoTableShapeRef, moeExpertNum, quantMode, syncFlag, ffnStartRankId, cclBufferSize);
}

namespace {
constexpr int64_t MB_ALIGN = 2LL * 1024LL * 1024LL; // 2MB
constexpr int64_t NO_QUANT_MODE = 0;
constexpr int64_t BYTES_PER_FP16 = 2;
constexpr int64_t BYTES_PER_INT32 = 4;
constexpr int64_t BYTES_PER_QUANT_ELEM = 1; // INT8/FP8 = 1 byte, FP4-packed HS already accounts for packing
constexpr int64_t WIN_REGION_ALIGN = 512;   // window region start alignment (matches kernel WIN_ALIGN)

int64_t CeilAlign(int64_t val, int64_t align)
{
    return (val + align - 1) / align * align;
}
} // namespace

int64_t GetAttentionToFfnCclBufferSize(const std::vector<int64_t>& ffnTokenInfoTableShape,
                                       const std::vector<int64_t>& ffnTokenDataShape, int64_t quantMode)
{
    TORCH_CHECK(ffnTokenInfoTableShape.size() == 3, "ffn_token_info_table_shape should have 3 elements, but got ",
                ffnTokenInfoTableShape.size());
    TORCH_CHECK(ffnTokenDataShape.size() == 5, "ffn_token_data_shape should have 5 elements, but got ",
                ffnTokenDataShape.size());

    int64_t attentionWorkerNum = ffnTokenDataShape[0];
    int64_t microBatchNum = ffnTokenDataShape[1];
    int64_t bs = ffnTokenDataShape[2];
    int64_t kPlusShared = ffnTokenDataShape[3];
    int64_t hs = ffnTokenDataShape[4];
    int64_t infoTableLastDim = ffnTokenInfoTableShape[2];

    int64_t tokenInfoSize = attentionWorkerNum * microBatchNum * infoTableLastDim * BYTES_PER_INT32;
    int64_t bytesPerElem = (quantMode == NO_QUANT_MODE) ? BYTES_PER_FP16 : BYTES_PER_QUANT_ELEM;
    int64_t tokenDataSize = attentionWorkerNum * microBatchNum * bs * kPlusShared * hs * bytesPerElem;
    int64_t needed = CeilAlign(tokenInfoSize, WIN_REGION_ALIGN) + CeilAlign(tokenDataSize, WIN_REGION_ALIGN);
    return CeilAlign(needed, MB_ALIGN);
}

// Bind the C++ function to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_attention_to_ffn", &NpuAttentionToFfn, "npu_attention_to_ffn");
    m.def("get_attention_to_ffn_ccl_buffer_size", &GetAttentionToFfnCclBufferSize,
          "get_attention_to_ffn_ccl_buffer_size");
    pybind11::class_<AttentionToFfnHostStatusTable>(m, "AttentionToFfnHostStatusTable")
        .def(pybind11::init<>())
        .def("reset", &AttentionToFfnHostStatusTable::Reset)
        .def("completed_round", &AttentionToFfnHostStatusTable::CompletedRound)
        .def("device_ptr", &AttentionToFfnHostStatusTable::DevicePtr);
}
} // namespace op_api
