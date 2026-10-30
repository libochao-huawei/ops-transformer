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
 * \brief PoolKeyIndexer arch22 kernel UT (ascend910b/ascend910_93, tikicpulib CPU 仿真)。
 *
 * pool_key_indexer.cpp 的入口是纯模板(设备侧由 tiling key 元数据实例化), CPU 编译其独立
 * TU 不产生任何实例化符号, 故在本测试 TU 内直接 include 该 cpp 并以显式模板实参调用,
 * 与框架另行编译的 KernelFile 目标文件不冲突。
 * tiling 字段按 host 侧 CalcTilingParams/SetTilingData/CalcWorkspaceSize 的产物逐值手填
 * (aicNum=1、非量化小 shape: s1BaseSize=4, mBaseSize=s1BaseSize*gSize, mBaseSizeMax=256,
 * s2BaseSize=128, sparseCount<=2048 时 trunkLen=16384, quantMode 存储值=host值+1=0,
 * sparseIndicesMode=0)。workspace 三区 offset 镜像 host 的 arch22 分支(PKI_WS_ARCH_BRANCH=1 时
 * DAV_2201 不并入 arch35 项)。
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

#include "../../../op_kernel/pool_key_indexer.cpp"

using namespace std;

namespace {
constexpr uint32_t B_SIZE = 1;
constexpr uint32_t N1 = 8; // gSize(query heads)
constexpr uint32_t N2 = 1; // kv heads
constexpr uint32_t HEAD_DIM = 128;
constexpr uint32_t AIC_NUM = 1;
constexpr uint64_t WORKSPACE_SIZE = 16 * 1024 * 1024;

inline uint64_t RoundUpLocal(uint64_t val, uint64_t align)
{
    return (val + align - 1) / align * align;
}

// 镜像 host CalcWorkspaceSize 的 arch22 口径(非量化, aicNum=1):
// | mm1ResGm(双缓冲 score, 512B 对齐) | (A5 才有 ldTail) | ldScore(32B 对齐) | ldIdx |
struct WsOffsets {
    uint64_t offScore;
    uint64_t offLdScore;
    uint64_t offLdIdx;
};

WsOffsets MirrorArch22WsOffsets(uint32_t sparseCount, uint32_t gSize)
{
    const uint64_t s1Base = (sparseCount > 2048) ? (8192 / sparseCount) * 2 : 8;
    const uint64_t mAlign = RoundUpLocal(s1Base * gSize, 16);
    const uint64_t scoreElem = 4; // 非量化 fixpipe 输出 fp32
    const uint64_t scoreSize = RoundUpLocal(AIC_NUM * 2 * mAlign * 512 * scoreElem, 512);
    const uint64_t ldScoreSize = RoundUpLocal(static_cast<uint64_t>(AIC_NUM) * s1Base * 2 * 2 * 8192 * 4, 32);
    WsOffsets o;
    o.offScore = 0;
    o.offLdScore = scoreSize;
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
    uint32_t tilingKey;
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
    td->s1BaseSize = 4; // PKI_S1_BASE_SIZE (sparseCount <= 6144)
    td->mBaseSize = 4 * cfg.n1;
    td->mBaseSizeMax = 256; // PKI_M_BASE_SIZE
    td->s2BaseSize = 128;   // PKI_S2_BASE_SIZE
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
    const WsOffsets ws = MirrorArch22WsOffsets(sparseCount, cfg.n1);
    td->wsOffScore = ws.offScore;
    td->wsOffLdScore = ws.offLdScore;
    td->wsOffLdIdx = ws.offLdIdx;
    td->qkScale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
    td->sparseIndicesMode = 0; // V1 legacy
}

template <typename T>
void FillTensor(vector<T>& data, int64_t count, T value)
{
    data.resize(count, value);
}
} // namespace

class PoolKeyIndexerKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "PoolKeyIndexerKernelTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "PoolKeyIndexerKernelTest TearDown" << endl;
    }
};

// Test 1: Degraded BSND FP16 maskMode=0 poolSize=1
// query(1,128,8,128) poolKey(1,128,1,128) weights(1,128,8) poolTailK=[0]
// topk=128 poolSize=1 -> sparseCount=128; indices(1,128,128) values(1,128,128)
TEST_F(PoolKeyIndexerKernelTest, Degraded_BSND_FP16_Mask0_PoolSize1)
{
    KernelTestConfig cfg = {};
    cfg.bSize = B_SIZE;
    cfg.n1 = N1;
    cfg.s1 = 128;
    cfg.s2 = 128;
    cfg.topk = 128;
    cfg.poolSize = 1;
    cfg.maskMode = 0;   // defaultMask
    cfg.quantMode = -1; // no quantization
    cfg.returnValue = true;
    cfg.layoutQ = 0; // BSND
    cfg.layoutK = 0; // BSND
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
    ASSERT_NE(weightsGm, nullptr);
    ASSERT_NE(poolTailKGm, nullptr);
    ASSERT_NE(indicesGm, nullptr);
    ASSERT_NE(valuesGm, nullptr);
    ASSERT_NE(workspaceGm, nullptr);
    ASSERT_NE(tilingGm, nullptr);

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

    // sanity: kernel 跑完不崩, 索引值域合法(>= -1)
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

// Test 2: Degraded BSND BF16 maskMode=3(causal) poolSize=1
// query(1,64,8,128) poolKey(1,64,1,128) weights(1,64,8) poolTailK=[0]
TEST_F(PoolKeyIndexerKernelTest, Degraded_BSND_BF16_MaskCausal_PoolSize1)
{
    KernelTestConfig cfg = {};
    cfg.bSize = B_SIZE;
    cfg.n1 = N1;
    cfg.s1 = 64;
    cfg.s2 = 64;
    cfg.topk = 64;
    cfg.poolSize = 1;
    cfg.maskMode = 3; // causal
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

// Test 3: poolSize>1 BSND FP16 maskMode=0
// query(1,64,8,128) poolKey(1,32,1,128) weights(1,64,8) poolTailK=[2]
// topk=128 poolSize=4 -> sparseCount=32; indices(1,64,131) values(1,64,32)
TEST_F(PoolKeyIndexerKernelTest, PoolSize4_BSND_FP16_Mask0)
{
    KernelTestConfig cfg = {};
    cfg.bSize = B_SIZE;
    cfg.n1 = N1;
    cfg.s1 = 64;
    cfg.s2 = 32; // 32 pools
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
    size_t valuesBytes = valuesSize * float(4);
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
