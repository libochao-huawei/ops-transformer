/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_recurrent_kda.cpp
 * \brief RecurrentKda two-phase ACLNN example with BF16 inputs and FP32 state.
 */

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_recurrent_kda.h"

#define CHECK_RET(cond, returnExpr) \
    do { \
        if (!(cond)) { \
            returnExpr; \
        } \
    } while (0)

namespace {
constexpr int64_t BATCH = 1;
constexpr int64_t SEQ_LEN = 2;
constexpr int64_t KEY_HEADS = 2;
constexpr int64_t VALUE_HEADS = 4;
constexpr int64_t KEY_DIM = 128;
constexpr int64_t VALUE_DIM = 128;
constexpr int64_t PRINT_COUNT = 5;
constexpr float TOLERANCE = 1.0e-6F;

struct TensorResource {
    void* deviceAddr = nullptr;
    aclTensor* tensor = nullptr;
};

struct Resources {
    int32_t deviceId = 0;
    bool aclInitialized = false;
    bool deviceInitialized = false;
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    void* workspaceAddr = nullptr;
    std::vector<TensorResource> tensors;

    ~Resources()
    {
        if (stream != nullptr) {
            (void)aclrtSynchronizeStream(stream);
        }
        for (const auto& tensor : tensors) {
            if (tensor.tensor != nullptr) {
                (void)aclDestroyTensor(tensor.tensor);
            }
            if (tensor.deviceAddr != nullptr) {
                (void)aclrtFree(tensor.deviceAddr);
            }
        }
        if (workspaceAddr != nullptr) {
            (void)aclrtFree(workspaceAddr);
        }
        if (stream != nullptr) {
            (void)aclrtDestroyStream(stream);
        }
        if (context != nullptr) {
            (void)aclrtDestroyContext(context);
        }
        if (deviceInitialized) {
            (void)aclrtResetDevice(deviceId);
        }
        if (aclInitialized) {
            (void)aclFinalize();
        }
    }
};

bool CheckAcl(aclError ret, const char* operation)
{
    if (ret == ACL_SUCCESS) {
        return true;
    }
    const char* message = aclGetRecentErrMsg();
    std::printf("%s failed. ERROR: %d. %s\n", operation, ret, message == nullptr ? "" : message);
    return false;
}

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (auto dim : shape) {
        size *= dim;
    }
    return size;
}

// The example values 0, 1/16 and 1 are represented exactly by BF16.
uint16_t FloatToBfloat16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16U);
}

float Bfloat16ToFloat(uint16_t value)
{
    uint32_t bits = static_cast<uint32_t>(value) << 16U;
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

bool Init(Resources& resources)
{
    CHECK_RET(CheckAcl(aclInit(nullptr), "aclInit"), return false);
    resources.aclInitialized = true;
    CHECK_RET(CheckAcl(aclrtSetDevice(resources.deviceId), "aclrtSetDevice"), return false);
    resources.deviceInitialized = true;
    CHECK_RET(CheckAcl(aclrtCreateContext(&resources.context, resources.deviceId), "aclrtCreateContext"), return false);
    CHECK_RET(CheckAcl(aclrtSetCurrentContext(resources.context), "aclrtSetCurrentContext"), return false);
    CHECK_RET(CheckAcl(aclrtCreateStream(&resources.stream), "aclrtCreateStream"), return false);
    return true;
}

template <typename T>
bool CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, aclDataType dataType,
                     Resources& resources, TensorResource& result)
{
    CHECK_RET(static_cast<int64_t>(hostData.size()) == GetShapeSize(shape), return false);
    resources.tensors.emplace_back();
    auto& tensor = resources.tensors.back();
    const size_t bytes = hostData.size() * sizeof(T);
    CHECK_RET(CheckAcl(aclrtMalloc(&tensor.deviceAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc"), return false);
    CHECK_RET(CheckAcl(aclrtMemcpy(tensor.deviceAddr, bytes, hostData.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE),
                       "aclrtMemcpy H2D"),
              return false);
    std::vector<int64_t> strides(shape.size(), 1);
    for (size_t i = shape.size() - 1; i > 0; --i) {
        strides[i - 1] = strides[i] * shape[i];
    }
    tensor.tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND,
                                    shape.data(), shape.size(), tensor.deviceAddr);
    CHECK_RET(tensor.tensor != nullptr, std::printf("aclCreateTensor failed.\n"); return false);
    result = tensor;
    return true;
}

template <typename T>
bool CopyToHost(const TensorResource& tensor, std::vector<T>& result)
{
    const size_t bytes = result.size() * sizeof(T);
    return CheckAcl(aclrtMemcpy(result.data(), bytes, tensor.deviceAddr, bytes, ACL_MEMCPY_DEVICE_TO_HOST),
                    "aclrtMemcpy D2H");
}

bool CheckResults(const TensorResource& out, const TensorResource& finalState, const TensorResource& initialState)
{
    const int64_t valuesPerToken = VALUE_HEADS * VALUE_DIM;
    const int64_t stateSize = BATCH * VALUE_HEADS * VALUE_DIM * KEY_DIM;
    std::vector<uint16_t> outData(BATCH * SEQ_LEN * valuesPerToken);
    std::vector<float> finalStateData(stateSize);
    std::vector<float> initialStateData(stateSize);
    CHECK_RET(CopyToHost(out, outData), return false);
    CHECK_RET(CopyToHost(finalState, finalStateData), return false);
    CHECK_RET(CopyToHost(initialState, initialStateData), return false);

    // S0=0, k=q=1/16, v=1, gate=0, beta=1/2, scale=1.
    // S1=1/32, S2=7/128; o1=1/4, o2=7/16 for every head and value element.
    const float expectedOut[SEQ_LEN] = {0.25F, 0.4375F};
    const float expectedState = 0.0546875F;
    for (size_t i = 0; i < outData.size(); ++i) {
        const float actual = Bfloat16ToFloat(outData[i]);
        const int64_t token = (i / valuesPerToken) % SEQ_LEN;
        if (!std::isfinite(actual) || std::fabs(actual - expectedOut[token]) > TOLERANCE) {
            std::printf("out[%zu] mismatch: actual=%f, expected=%f\n", i, actual, expectedOut[token]);
            return false;
        }
    }
    for (size_t i = 0; i < finalStateData.size(); ++i) {
        if (!std::isfinite(finalStateData[i]) || !std::isfinite(initialStateData[i]) ||
            std::fabs(finalStateData[i] - expectedState) > TOLERANCE ||
            std::fabs(initialStateData[i] - expectedState) > TOLERANCE) {
            std::printf("state[%zu] mismatch: final=%f, inplace=%f, expected=%f\n", i, finalStateData[i],
                        initialStateData[i], expectedState);
            return false;
        }
    }
    for (int64_t token = 0; token < SEQ_LEN; ++token) {
        for (int64_t i = 0; i < PRINT_COUNT; ++i) {
            std::printf("out token[%ld] result[%ld] is: %f\n", token, i,
                        Bfloat16ToFloat(outData[token * valuesPerToken + i]));
        }
    }
    for (int64_t i = 0; i < PRINT_COUNT; ++i) {
        std::printf("finalState result[%ld] is: %.7f\n", i, finalStateData[i]);
    }
    std::printf("RecurrentKda example passed: out, finalState and inplace state match expected values.\n");
    return true;
}

bool RunExample(Resources& resources)
{
    // BSND: q/k [B,T,Hk,K], value/out [B,T,Hv,V], gate [B,T,Hv,K], beta [B,T,Hv].
    // stateVFirst=true: state [B,Hv,V,K]. Hv/Hk=2 demonstrates shared query/key heads.
    const std::vector<int64_t> qkShape = {BATCH, SEQ_LEN, KEY_HEADS, KEY_DIM};
    const std::vector<int64_t> valueShape = {BATCH, SEQ_LEN, VALUE_HEADS, VALUE_DIM};
    const std::vector<int64_t> gateShape = {BATCH, SEQ_LEN, VALUE_HEADS, KEY_DIM};
    const std::vector<int64_t> betaShape = {BATCH, SEQ_LEN, VALUE_HEADS};
    const std::vector<int64_t> stateShape = {BATCH, VALUE_HEADS, VALUE_DIM, KEY_DIM};
    const std::vector<uint16_t> qkData(GetShapeSize(qkShape), FloatToBfloat16(0.0625F));
    const std::vector<uint16_t> valueData(GetShapeSize(valueShape), FloatToBfloat16(1.0F));
    const std::vector<float> gateData(GetShapeSize(gateShape), 0.0F);
    const std::vector<float> betaData(GetShapeSize(betaShape), 0.5F);
    const std::vector<float> stateData(GetShapeSize(stateShape), 0.0F);
    const std::vector<uint16_t> outData(GetShapeSize(valueShape), FloatToBfloat16(0.0F));
    TensorResource query, key, value, gate, beta, initialState, out, finalState;
    CHECK_RET(CreateAclTensor(qkData, qkShape, ACL_BF16, resources, query), return false);
    CHECK_RET(CreateAclTensor(qkData, qkShape, ACL_BF16, resources, key), return false);
    CHECK_RET(CreateAclTensor(valueData, valueShape, ACL_BF16, resources, value), return false);
    CHECK_RET(CreateAclTensor(gateData, gateShape, ACL_FLOAT, resources, gate), return false);
    CHECK_RET(CreateAclTensor(betaData, betaShape, ACL_FLOAT, resources, beta), return false);
    CHECK_RET(CreateAclTensor(stateData, stateShape, ACL_FLOAT, resources, initialState), return false);
    CHECK_RET(CreateAclTensor(outData, valueShape, ACL_BF16, resources, out), return false);
    CHECK_RET(CreateAclTensor(stateData, stateShape, ACL_FLOAT, resources, finalState), return false);

    // 1. First phase: validate inputs and obtain the workspace size and executor.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const char layout[] = "BSND";
    const double scale = 1.0;
    const bool outputFinalState = true;
    const bool inplaceFinalState = true;
    const bool useQkL2normInKernel = false;
    const bool useGateInKernel = false;
    const bool useBetaSigmoidInKernel = false;
    const bool allowNegEigval = false;
    const bool safeGate = false;
    const double lowerBound = -5.0;
    const bool stateVFirst = true;
    auto ret = aclnnRecurrentKdaGetWorkspaceSize(
        query.tensor, key.tensor, value.tensor, gate.tensor, beta.tensor, initialState.tensor, nullptr, nullptr,
        nullptr, nullptr, nullptr, layout, scale, outputFinalState, inplaceFinalState, useQkL2normInKernel,
        useGateInKernel, useBetaSigmoidInKernel, allowNegEigval, safeGate, lowerBound, stateVFirst, out.tensor,
        finalState.tensor, &workspaceSize, &executor);
    CHECK_RET(CheckAcl(ret, "aclnnRecurrentKdaGetWorkspaceSize"), return false);
    if (workspaceSize > 0) {
        CHECK_RET(CheckAcl(aclrtMalloc(&resources.workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST),
                           "allocate workspace"),
                  return false);
    }

    // 2. Second phase: launch the operator and wait for completion before reading results.
    ret = aclnnRecurrentKda(resources.workspaceAddr, workspaceSize, executor, resources.stream);
    CHECK_RET(CheckAcl(ret, "aclnnRecurrentKda"), return false);
    CHECK_RET(CheckAcl(aclrtSynchronizeStream(resources.stream), "aclrtSynchronizeStream"), return false);
    return CheckResults(out, finalState, initialState);
}
} // namespace

int main(int argc, char* argv[])
{
    Resources resources;
    if (argc > 2) {
        std::printf("Usage: %s [deviceId]\n", argv[0]);
        return 1;
    }
    if (argc == 2) {
        char* end = nullptr;
        errno = 0;
        const long deviceId = std::strtol(argv[1], &end, 10);
        if (errno != 0 || end == argv[1] || *end != '\0' || deviceId < 0 ||
            deviceId > std::numeric_limits<int32_t>::max()) {
            std::printf("Invalid deviceId: %s\n", argv[1]);
            return 1;
        }
        resources.deviceId = static_cast<int32_t>(deviceId);
    }
    CHECK_RET(Init(resources), return 1);
    CHECK_RET(RunExample(resources), return 1);
    return 0;
}
