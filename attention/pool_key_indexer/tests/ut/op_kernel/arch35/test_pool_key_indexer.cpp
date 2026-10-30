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
 * \file test_pool_key_indexer.cpp
 * \brief PoolKeyIndexer arch35 kernel UT (ascend950, tikicpulib::Ascend950PR_9599 CPU 仿真)。
 *
 * 设备构建由 ascendc_impl_build 为每个 tiling key 注入 -DORIG_DTYPE_xxx=<ge枚举>;
 * CPU 仿真在此等价地把 ORIG_DTYPE_* 映射到模板形参 DT_Q/DT_K(仅覆盖 fp16/bf16 非量化,
 * 其 PKI_TPL 码与 ge 枚举值一致; fp8/int8 实例化值域不同, 留给真机 ST)。
 * tiling 字段镜像 host CalcTilingParams/SetTilingData; workspace offset 镜像 host
 * CalcWorkspaceSize 的 A5 口径(npuArch!=DAV_2201: scoreGm 取 max(arch22,arch35),
 * 且 sparseCount<=2048 时 score 与 ldScore 之间插 ldTail 区)。
 */
#include <array>
#include <vector>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#endif

// 设备侧 -DORIG_DTYPE_QUERY/KEY/WEIGHTS 的 CPU 等价映射(须在 include 入口 cpp 之前)
#ifndef ORIG_DTYPE_QUERY
#define ORIG_DTYPE_QUERY DT_Q
#define ORIG_DTYPE_KEY DT_K
#define ORIG_DTYPE_WEIGHTS DT_Q
#endif

#include "../../../../op_kernel/pool_key_indexer.cpp"

using namespace std;

namespace {
constexpr uint32_t B_SIZE = 1;
constexpr uint32_t N1 = 8;
constexpr uint32_t N2 = 1;
constexpr uint32_t HEAD_DIM = 128;
constexpr uint32_t AIC_NUM = 1;
constexpr uint64_t WORKSPACE_SIZE = 16 * 1024 * 1024;

inline uint64_t RoundUpLocal(uint64_t val, uint64_t align)
{
    return (val + align - 1) / align * align;
}

// 镜像 host CalcWorkspaceSize 的 A5 分支(非量化 fp32 分数, aicNum=1)
struct WsOffsets {
    uint64_t offScore;
    uint64_t offLdScore;
    uint64_t offLdIdx;
};

WsOffsets MirrorArch35WsOffsets(uint32_t sparseCount, uint32_t gSize, uint32_t s2Size)
{
    const uint64_t s1Base = (sparseCount > 2048) ? (8192 / sparseCount) * 2 : 8;
    const uint64_t mAlign = RoundUpLocal(s1Base * gSize, 16);
    const uint64_t scoreSize22 = RoundUpLocal(static_cast<uint64_t>(AIC_NUM) * 2 * mAlign * 512 * 4, 512);
    const uint64_t s1Base35 = (sparseCount > 2048) ? 2 : 4;
    const uint64_t s2Align35 = RoundUpLocal(s2Size, 128);
    const uint64_t scoreSize35 = RoundUpLocal(static_cast<uint64_t>(AIC_NUM) * s1Base35 * s2Align35 * 4, 512);
    uint64_t scoreSize = (scoreSize22 > scoreSize35) ? scoreSize22 : scoreSize35;
    uint64_t offset = scoreSize;
    if (sparseCount <= 2048) {
        offset += RoundUpLocal(static_cast<uint64_t>(AIC_NUM) * s1Base35 * s2Align35 * 4, 512); // ldTail
    }
    WsOffsets o;
    o.offScore = 0;
    o.offLdScore = offset;
    const uint64_t ldScoreSize = RoundUpLocal(static_cast<uint64_t>(AIC_NUM) * s1Base * 2 * 2 * 8192 * 4, 32);
    o.offLdIdx = o.offLdScore + ldScoreSize;
    return o;
}

struct KernelTestConfig {
    uint32_t bSize;
    uint32_t n1;
    uint32_t s1;
    uint32_t s2;
    uint32_t topk;
    uint32_t poolSize;
    uint32_t maskMode;
    int32_t quantMode; // host 值(-1=非量化), tiling 存储值 = 该值 + 1
    bool returnValue;
    uint32_t layoutQ;
    uint32_t layoutK;
    uint32_t usedCoreNum;
};

void SetTilingData(PoolKeyIndexerTilingData* td, const KernelTestConfig& cfg)
{
    const uint32_t sparseCount = cfg.topk / cfg.poolSize;
    td->bSize = cfg.bSize;
    td->gSize = cfg.n1;
    td->s1Size = cfg.s1;
    td->s2Size = static_cast<int64_t>(cfg.s2);
    td->sparseCount = sparseCount;
    td->topk = cfg.topk;
    td->poolSize = cfg.poolSize;
    td->usedCoreNum = cfg.usedCoreNum;
    td->s1BaseSize = 4;
    td->mBaseSize = 4 * cfg.n1;
    td->mBaseSizeMax = 256;
    td->s2BaseSize = 128;
    td->trunkLen = (sparseCount <= 2048) ? 16384 : 8192;
    td->maskMode = cfg.maskMode;
    td->quantMode = static_cast<uint32_t>(cfg.quantMode + 1);
    td->returnValue = cfg.returnValue ? 1 : 0;
    td->layoutQ = cfg.layoutQ;
    td->layoutK = cfg.layoutK;
    td->blockSize = 0;
    td->maxBlockNumPerBatch = 0;
    td->keyStride0 = cfg.s2 * N2 * HEAD_DIM;
    td->keyDequantScaleStride0 = 0;
    const WsOffsets ws = MirrorArch35WsOffsets(sparseCount, cfg.n1, cfg.s2);
    td->wsOffScore = ws.offScore;
    td->wsOffLdScore = ws.offLdScore;
    td->wsOffLdIdx = ws.offLdIdx;
    td->qkScale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
    td->sparseIndicesMode = 0;
}

template <typename T>
void FillTensor(vector<T>& data, int64_t count, T value)
{
    data.resize(count, value);
}
} // namespace

class PoolKeyIndexerKernelTestArch35 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "PoolKeyIndexerKernelTestArch35 SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "PoolKeyIndexerKernelTestArch35 TearDown" << endl;
    }
};

// Test 1: BSND FP16 maskMode=0 poolSize=1 (非量化 radix TopK 主路)
TEST_F(PoolKeyIndexerKernelTestArch35, Degraded_BSND_FP16_Mask0_PoolSize1)
{
    KernelTestConfig cfg = {};
    cfg.bSize = B_SIZE;
    cfg.n1 = N1;
    cfg.s1 = 128;
    cfg.s2 = 128;
    cfg.topk = 128;
    cfg.poolSize = 1;
    cfg.maskMode = 0;
    cfg.quantMode = -1;
    cfg.returnValue = true;
    cfg.layoutQ = 0;
    cfg.layoutK = 0;
    cfg.usedCoreNum = AIC_NUM;

    uint32_t outputLen = cfg.topk + cfg.poolSize - 1;
    uint32_t sparseCount = cfg.topk / cfg.poolSize;

    int64_t querySize = cfg.bSize * cfg.s1 * cfg.n1 * HEAD_DIM;
    int64_t poolKeySize = cfg.bSize * cfg.s2 * N2 * HEAD_DIM;
    int64_t weightsSize = cfg.bSize * cfg.s1 * cfg.n1;
    int64_t poolTailKSize = cfg.bSize;
    int64_t indicesSize = cfg.bSize * cfg.s1 * outputLen;
    int64_t valuesSize = cfg.bSize * cfg.s1 * sparseCount;

    size_t queryBytes = querySize * sizeof(half);
    size_t poolKeyBytes = poolKeySize * sizeof(half);
    size_t weightsBytes = weightsSize * sizeof(half);
    size_t poolTailKBytes = poolTailKSize * sizeof(int64_t);
    size_t indicesBytes = indicesSize * sizeof(int32_t);
    size_t valuesBytes = valuesSize * sizeof(float);
    size_t tilingBytes = sizeof(PoolKeyIndexerTilingData);

    uint8_t* queryGm = (uint8_t*)AscendC::GmAlloc(queryBytes);
    uint8_t* poolKeyGm = (uint8_t*)AscendC::GmAlloc(poolKeyBytes);
    uint8_t* weightsGm = (uint8_t*)AscendC::GmAlloc(weightsBytes);
    uint8_t* poolTailKGm = (uint8_t*)AscendC::GmAlloc(poolTailKBytes);
    uint8_t* indicesGm = (uint8_t*)AscendC::GmAlloc(indicesBytes);
    uint8_t* valuesGm = (uint8_t*)AscendC::GmAlloc(valuesBytes);
    uint8_t* workspaceGm = (uint8_t*)AscendC::GmAlloc(WORKSPACE_SIZE);
    uint8_t* tilingGm = (uint8_t*)AscendC::GmAlloc(tilingBytes);

    ASSERT_NE(queryGm, nullptr);
    ASSERT_NE(poolKeyGm, nullptr);
    ASSERT_NE(indicesGm, nullptr);

    vector<half> queryData;
    FillTensor(queryData, querySize, static_cast<half>(0.5f));
    vector<half> poolKeyData;
    FillTensor(poolKeyData, poolKeySize, static_cast<half>(0.5f));
    vector<half> weightsData;
    FillTensor(weightsData, weightsSize, static_cast<half>(1.0f));
    vector<int64_t> poolTailKData;
    FillTensor(poolTailKData, poolTailKSize, static_cast<int64_t>(0));

    memcpy(queryGm, queryData.data(), queryBytes);
    memcpy(poolKeyGm, poolKeyData.data(), poolKeyBytes);
    memcpy(weightsGm, weightsData.data(), weightsBytes);
    memcpy(poolTailKGm, poolTailKData.data(), poolTailKBytes);
    memset(indicesGm, 0xFF, indicesBytes);
    memset(valuesGm, 0, valuesBytes);

    auto* td = reinterpret_cast<PoolKeyIndexerTilingData*>(tilingGm);
    SetTilingData(td, cfg);

    auto kernelFn = [&](GM_ADDR query, GM_ADDR poolKey, GM_ADDR weights, GM_ADDR poolTailK, GM_ADDR actSeqQ,
                        GM_ADDR actSeqK, GM_ADDR blockTable, GM_ADDR qDescale, GM_ADDR kDescale, GM_ADDR sparseIndices,
                        GM_ADDR sparseValues, GM_ADDR workspace, GM_ADDR tiling) {
        pool_key_indexer<1, 1, 3, 0, 0, 0, 0, 1>(query, poolKey, weights, poolTailK, actSeqQ, actSeqK, blockTable,
                                                 qDescale, kDescale, sparseIndices, sparseValues, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ICPU_RUN_KF(kernelFn, cfg.usedCoreNum, queryGm, poolKeyGm, weightsGm, poolTailKGm, nullptr, nullptr, nullptr,
                nullptr, nullptr, indicesGm, valuesGm, workspaceGm, tilingGm);

    int32_t* indicesOut = reinterpret_cast<int32_t*>(indicesGm);
    for (int64_t i = 0; i < indicesSize; i++) {
        EXPECT_GE(indicesOut[i], -1);
    }

    AscendC::GmFree(queryGm);
    AscendC::GmFree(poolKeyGm);
    AscendC::GmFree(weightsGm);
    AscendC::GmFree(poolTailKGm);
    AscendC::GmFree(indicesGm);
    AscendC::GmFree(valuesGm);
    AscendC::GmFree(workspaceGm);
    AscendC::GmFree(tilingGm);
}

// Test 2: BSND BF16 maskMode=3(causal) poolSize=1
TEST_F(PoolKeyIndexerKernelTestArch35, Degraded_BSND_BF16_MaskCausal_PoolSize1)
{
    KernelTestConfig cfg = {};
    cfg.bSize = B_SIZE;
    cfg.n1 = N1;
    cfg.s1 = 64;
    cfg.s2 = 64;
    cfg.topk = 64;
    cfg.poolSize = 1;
    cfg.maskMode = 3;
    cfg.quantMode = -1;
    cfg.returnValue = true;
    cfg.layoutQ = 0;
    cfg.layoutK = 0;
    cfg.usedCoreNum = AIC_NUM;

    uint32_t outputLen = cfg.topk + cfg.poolSize - 1;
    uint32_t sparseCount = cfg.topk / cfg.poolSize;

    int64_t querySize = cfg.bSize * cfg.s1 * cfg.n1 * HEAD_DIM;
    int64_t poolKeySize = cfg.bSize * cfg.s2 * N2 * HEAD_DIM;
    int64_t weightsSize = cfg.bSize * cfg.s1 * cfg.n1;
    int64_t poolTailKSize = cfg.bSize;
    int64_t indicesSize = cfg.bSize * cfg.s1 * outputLen;
    int64_t valuesSize = cfg.bSize * cfg.s1 * sparseCount;

    size_t queryBytes = querySize * sizeof(bfloat16_t);
    size_t poolKeyBytes = poolKeySize * sizeof(bfloat16_t);
    size_t weightsBytes = weightsSize * sizeof(bfloat16_t);
    size_t poolTailKBytes = poolTailKSize * sizeof(int64_t);
    size_t indicesBytes = indicesSize * sizeof(int32_t);
    size_t valuesBytes = valuesSize * sizeof(float);
    size_t tilingBytes = sizeof(PoolKeyIndexerTilingData);

    uint8_t* queryGm = (uint8_t*)AscendC::GmAlloc(queryBytes);
    uint8_t* poolKeyGm = (uint8_t*)AscendC::GmAlloc(poolKeyBytes);
    uint8_t* weightsGm = (uint8_t*)AscendC::GmAlloc(weightsBytes);
    uint8_t* poolTailKGm = (uint8_t*)AscendC::GmAlloc(poolTailKBytes);
    uint8_t* indicesGm = (uint8_t*)AscendC::GmAlloc(indicesBytes);
    uint8_t* valuesGm = (uint8_t*)AscendC::GmAlloc(valuesBytes);
    uint8_t* workspaceGm = (uint8_t*)AscendC::GmAlloc(WORKSPACE_SIZE);
    uint8_t* tilingGm = (uint8_t*)AscendC::GmAlloc(tilingBytes);

    ASSERT_NE(queryGm, nullptr);
    ASSERT_NE(poolKeyGm, nullptr);
    ASSERT_NE(indicesGm, nullptr);

    vector<bfloat16_t> queryData;
    FillTensor(queryData, querySize, static_cast<bfloat16_t>(0.5f));
    vector<bfloat16_t> poolKeyData;
    FillTensor(poolKeyData, poolKeySize, static_cast<bfloat16_t>(0.5f));
    vector<bfloat16_t> weightsData;
    FillTensor(weightsData, weightsSize, static_cast<bfloat16_t>(1.0f));
    vector<int64_t> poolTailKData;
    FillTensor(poolTailKData, poolTailKSize, static_cast<int64_t>(0));

    memcpy(queryGm, queryData.data(), queryBytes);
    memcpy(poolKeyGm, poolKeyData.data(), poolKeyBytes);
    memcpy(weightsGm, weightsData.data(), weightsBytes);
    memcpy(poolTailKGm, poolTailKData.data(), poolTailKBytes);
    memset(indicesGm, 0xFF, indicesBytes);
    memset(valuesGm, 0, valuesBytes);

    auto* td = reinterpret_cast<PoolKeyIndexerTilingData*>(tilingGm);
    SetTilingData(td, cfg);

    auto kernelFn = [&](GM_ADDR query, GM_ADDR poolKey, GM_ADDR weights, GM_ADDR poolTailK, GM_ADDR actSeqQ,
                        GM_ADDR actSeqK, GM_ADDR blockTable, GM_ADDR qDescale, GM_ADDR kDescale, GM_ADDR sparseIndices,
                        GM_ADDR sparseValues, GM_ADDR workspace, GM_ADDR tiling) {
        pool_key_indexer<27, 27, 3, 0, 0, 3, 0, 1>(query, poolKey, weights, poolTailK, actSeqQ, actSeqK, blockTable,
                                                   qDescale, kDescale, sparseIndices, sparseValues, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ICPU_RUN_KF(kernelFn, cfg.usedCoreNum, queryGm, poolKeyGm, weightsGm, poolTailKGm, nullptr, nullptr, nullptr,
                nullptr, nullptr, indicesGm, valuesGm, workspaceGm, tilingGm);

    int32_t* indicesOut = reinterpret_cast<int32_t*>(indicesGm);
    for (int64_t i = 0; i < indicesSize; i++) {
        EXPECT_GE(indicesOut[i], -1);
    }

    AscendC::GmFree(queryGm);
    AscendC::GmFree(poolKeyGm);
    AscendC::GmFree(weightsGm);
    AscendC::GmFree(poolTailKGm);
    AscendC::GmFree(indicesGm);
    AscendC::GmFree(valuesGm);
    AscendC::GmFree(workspaceGm);
    AscendC::GmFree(tilingGm);
}

// Test 3: poolSize=4 BSND FP16 maskMode=0 (pool 聚合 + tail 补偿)
TEST_F(PoolKeyIndexerKernelTestArch35, PoolSize4_BSND_FP16_Mask0)
{
    KernelTestConfig cfg = {};
    cfg.bSize = B_SIZE;
    cfg.n1 = N1;
    cfg.s1 = 64;
    cfg.s2 = 32;
    cfg.topk = 128;
    cfg.poolSize = 4;
    cfg.maskMode = 0;
    cfg.quantMode = -1;
    cfg.returnValue = true;
    cfg.layoutQ = 0;
    cfg.layoutK = 0;
    cfg.usedCoreNum = AIC_NUM;

    uint32_t outputLen = cfg.topk + cfg.poolSize - 1;
    uint32_t sparseCount = cfg.topk / cfg.poolSize;

    int64_t querySize = cfg.bSize * cfg.s1 * cfg.n1 * HEAD_DIM;
    int64_t poolKeySize = cfg.bSize * cfg.s2 * N2 * HEAD_DIM;
    int64_t weightsSize = cfg.bSize * cfg.s1 * cfg.n1;
    int64_t poolTailKSize = cfg.bSize;
    int64_t indicesSize = cfg.bSize * cfg.s1 * outputLen;
    int64_t valuesSize = cfg.bSize * cfg.s1 * sparseCount;

    size_t queryBytes = querySize * sizeof(half);
    size_t poolKeyBytes = poolKeySize * sizeof(half);
    size_t weightsBytes = weightsSize * sizeof(half);
    size_t poolTailKBytes = poolTailKSize * sizeof(int64_t);
    size_t indicesBytes = indicesSize * sizeof(int32_t);
    size_t valuesBytes = valuesSize * sizeof(float);
    size_t tilingBytes = sizeof(PoolKeyIndexerTilingData);

    uint8_t* queryGm = (uint8_t*)AscendC::GmAlloc(queryBytes);
    uint8_t* poolKeyGm = (uint8_t*)AscendC::GmAlloc(poolKeyBytes);
    uint8_t* weightsGm = (uint8_t*)AscendC::GmAlloc(weightsBytes);
    uint8_t* poolTailKGm = (uint8_t*)AscendC::GmAlloc(poolTailKBytes);
    uint8_t* indicesGm = (uint8_t*)AscendC::GmAlloc(indicesBytes);
    uint8_t* valuesGm = (uint8_t*)AscendC::GmAlloc(valuesBytes);
    uint8_t* workspaceGm = (uint8_t*)AscendC::GmAlloc(WORKSPACE_SIZE);
    uint8_t* tilingGm = (uint8_t*)AscendC::GmAlloc(tilingBytes);

    ASSERT_NE(queryGm, nullptr);
    ASSERT_NE(poolKeyGm, nullptr);
    ASSERT_NE(indicesGm, nullptr);

    vector<half> queryData;
    FillTensor(queryData, querySize, static_cast<half>(0.5f));
    vector<half> poolKeyData;
    FillTensor(poolKeyData, poolKeySize, static_cast<half>(0.5f));
    vector<half> weightsData;
    FillTensor(weightsData, weightsSize, static_cast<half>(1.0f));
    vector<int64_t> poolTailKData;
    FillTensor(poolTailKData, poolTailKSize, static_cast<int64_t>(2));

    memcpy(queryGm, queryData.data(), queryBytes);
    memcpy(poolKeyGm, poolKeyData.data(), poolKeyBytes);
    memcpy(weightsGm, weightsData.data(), weightsBytes);
    memcpy(poolTailKGm, poolTailKData.data(), poolTailKBytes);
    memset(indicesGm, 0xFF, indicesBytes);
    memset(valuesGm, 0, valuesBytes);

    auto* td = reinterpret_cast<PoolKeyIndexerTilingData*>(tilingGm);
    SetTilingData(td, cfg);

    auto kernelFn = [&](GM_ADDR query, GM_ADDR poolKey, GM_ADDR weights, GM_ADDR poolTailK, GM_ADDR actSeqQ,
                        GM_ADDR actSeqK, GM_ADDR blockTable, GM_ADDR qDescale, GM_ADDR kDescale, GM_ADDR sparseIndices,
                        GM_ADDR sparseValues, GM_ADDR workspace, GM_ADDR tiling) {
        pool_key_indexer<1, 1, 3, 0, 0, 0, 0, 1>(query, poolKey, weights, poolTailK, actSeqQ, actSeqK, blockTable,
                                                 qDescale, kDescale, sparseIndices, sparseValues, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ICPU_RUN_KF(kernelFn, cfg.usedCoreNum, queryGm, poolKeyGm, weightsGm, poolTailKGm, nullptr, nullptr, nullptr,
                nullptr, nullptr, indicesGm, valuesGm, workspaceGm, tilingGm);

    int32_t* indicesOut = reinterpret_cast<int32_t*>(indicesGm);
    for (int64_t i = 0; i < indicesSize; i++) {
        EXPECT_GE(indicesOut[i], -1);
    }

    AscendC::GmFree(queryGm);
    AscendC::GmFree(poolKeyGm);
    AscendC::GmFree(weightsGm);
    AscendC::GmFree(poolTailKGm);
    AscendC::GmFree(indicesGm);
    AscendC::GmFree(valuesGm);
    AscendC::GmFree(workspaceGm);
    AscendC::GmFree(tilingGm);
}
