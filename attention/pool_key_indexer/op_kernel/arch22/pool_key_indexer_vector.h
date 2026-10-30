/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file pool_key_indexer_vector.h
 * \brief
 */
#ifndef POOL_KEY_INDEXER_VECTOR_H
#define POOL_KEY_INDEXER_VECTOR_H

#include "pool_key_indexer_vector.h"
#include "kernel_operator.h"

namespace PkiServiceVec {
using namespace AscendC;

constexpr int32_t NEG_INF = 0xFF800000;
constexpr int32_t INVALID_INDEX = -1;
constexpr uint8_t VEC_REPEAT_MAX = 255;
constexpr uint8_t B32_VEC_ELM_NUM = 64;
constexpr uint8_t B32_BLOCK_ALIGN_NUM = 8;
constexpr uint8_t B32_VEC_REPEAT_STRIDE = 8;
constexpr uint64_t VEC_REPEAT_BYTES = 256;
constexpr int32_t CONST_TWO = 2;
constexpr int64_t VALUE_AND_INDEX_NUM = 2;
constexpr int64_t BLOCK_BYTES = 32;
constexpr int64_t MRG_QUE_0 = 0;
constexpr int64_t MRG_QUE_1 = 1;
constexpr int64_t MRG_QUE_2 = 2;
constexpr int64_t MRG_QUE_3 = 3;
constexpr int64_t MRG_BLOCK_2 = 2;
constexpr int64_t MRG_BLOCK_3 = 3;
constexpr int64_t MRG_BLOCK_4 = 4;

template <typename T>
__aicore__ inline void CopyIn(LocalTensor<float>& mmOutUb, LocalTensor<T>& weightsUb, GlobalTensor<float>& mMoutGm,
                              GlobalTensor<T>& weightScaleGm, int64_t MMout_gmoffset, int64_t weights_gmoffset,
                              int64_t groupInner, int64_t s2Inner, int64_t mmUbStride)
{
    // 将MMout_gmoffset copy到UB上
    AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams dataCopymMoutParams;
    dataCopymMoutParams.blockCount = groupInner;
    dataCopymMoutParams.blockLen = s2Inner * sizeof(float);
    dataCopymMoutParams.srcStride = 0;
    dataCopymMoutParams.dstStride = mmUbStride;
    dataCopymMoutParams.rsv = 0;
    AscendC::DataCopyPad(mmOutUb, mMoutGm[MMout_gmoffset], dataCopymMoutParams, padParams);

    // 将weights_gmoffset copy到UB
    AscendC::DataCopyPadExtParams<T> padTParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams dataCopyweightParams;
    dataCopyweightParams.blockCount = 1;
    dataCopyweightParams.blockLen = groupInner * sizeof(T);
    dataCopyweightParams.srcStride = 0;
    dataCopyweightParams.dstStride = 0;
    dataCopyweightParams.rsv = 0;
    AscendC::DataCopyPad(weightsUb, weightScaleGm[weights_gmoffset], dataCopyweightParams, padTParams);
}

/**
  仅搬 mm1 结果的 CopyIn 变体: weights 已由调用方整块预取(见 service_vector 的
  weightsRowBuf_), 搬运参数与 CopyIn 的 mm1 部分一致。
 */
__aicore__ inline void CopyInMm1(LocalTensor<float>& mmOutUb, GlobalTensor<float>& mMoutGm, int64_t MMout_gmoffset,
                                 int64_t groupInner, int64_t s2Inner, int64_t mmUbStride)
{
    // 将MMout_gmoffset copy到UB上
    AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams dataCopymMoutParams;
    dataCopymMoutParams.blockCount = groupInner;
    dataCopymMoutParams.blockLen = s2Inner * sizeof(float);
    dataCopymMoutParams.srcStride = 0;
    dataCopymMoutParams.dstStride = mmUbStride;
    dataCopymMoutParams.rsv = 0;
    AscendC::DataCopyPad(mmOutUb, mMoutGm[MMout_gmoffset], dataCopymMoutParams, padParams);
}

template <typename T>
__aicore__ inline void CopyOut(const GlobalTensor<T>& dstGm, const LocalTensor<T>& srcUb, int64_t copyCount)
{
    AscendC::DataCopyParams dataCopyOutyParams;
    dataCopyOutyParams.blockCount = 1;
    dataCopyOutyParams.blockLen = copyCount * sizeof(T);
    dataCopyOutyParams.srcStride = 0;
    dataCopyOutyParams.dstStride = 0;
    AscendC::DataCopyPad(dstGm, srcUb, dataCopyOutyParams);
}

// int8 路径搬入: score 为 fixpipe DEQF16 写出的 fp16, weights/q_descale 为 FP16。
// score 直搬 float 计算区的 half 视图"行后半"(行距 2*ubRowHalf half): half 源起点比
// float 目的起点高 ubRowHalf*sizeof(half) 字节, 后续原地扩展 Cast 逐元素写恒落后于读,
// 无需独立 half 暂存区(UB 预算零追加); weights 与 q_descale 同 pattern 同偏移(去 D 轴逐维相等)。
// s2Inner 为 GM 行距(actualSingleProcessSInnerSizeAlign), ubRowHalf 为 UB 行宽(s2BaseSize)。
template <typename T>
__aicore__ inline void CopyInInt8(LocalTensor<float>& mmOutUb, LocalTensor<T>& weightsUb, LocalTensor<T>& qScaleUb,
                                  GlobalTensor<half>& mMoutGm, GlobalTensor<T>& weightScaleGm,
                                  GlobalTensor<T>& qScaleGm, int64_t MMout_gmoffset, int64_t weights_gmoffset,
                                  int64_t groupInner, int64_t s2Inner, int64_t ubRowHalf, int64_t mmUbStride)
{
    // score(fp16) 搬入 float 计算区行后半: 目的行距 = 2*ubRowHalf*2B, blockLen = s2Inner*2B
    // dstStride 单位 32B
    AscendC::DataCopyPadExtParams<half> padParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams dataCopymMoutParams;
    dataCopymMoutParams.blockCount = groupInner;
    dataCopymMoutParams.blockLen = s2Inner * sizeof(half);
    dataCopymMoutParams.srcStride = 0;
    dataCopymMoutParams.dstStride = mmUbStride;
    dataCopymMoutParams.rsv = 0;
    LocalTensor<half> mmOutHalfUb = mmOutUb.template ReinterpretCast<half>()[ubRowHalf];
    AscendC::DataCopyPad(mmOutHalfUb, mMoutGm[MMout_gmoffset], dataCopymMoutParams, padParams);

    // weights 与 q_descale 同偏移各搬一行(半精度, 落在段尾 float 区头部 half 视图)
    AscendC::DataCopyPadExtParams<T> padTParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams dataCopyweightParams;
    dataCopyweightParams.blockCount = 1;
    dataCopyweightParams.blockLen = groupInner * sizeof(T);
    dataCopyweightParams.srcStride = 0;
    dataCopyweightParams.dstStride = 0;
    dataCopyweightParams.rsv = 0;
    AscendC::DataCopyPad(weightsUb, weightScaleGm[weights_gmoffset], dataCopyweightParams, padTParams);
    AscendC::DataCopyPad(qScaleUb, qScaleGm[weights_gmoffset], dataCopyweightParams, padTParams);
}

/**
  int8 路径**仅搬 score** 的 CopyInInt8 变体(预取路径): weights/q_descale 已由
  调用方在 base block 之前一次整块预取(见 service_vector 的 weightsRowBuf_/
  qScaleRowBuf_), 行内不再逐 outerG 下发两条 32B 微拷贝。score 段的搬运参数与
  CopyInInt8 的前半段**完全一致**(同 blockCount/blockLen/srcStride/dstStride/落点),
  故搬入 UB 的位型不变。
 */
__aicore__ inline void CopyInScoreInt8(LocalTensor<float>& mmOutUb, GlobalTensor<half>& mMoutGm, int64_t MMout_gmoffset,
                                       int64_t groupInner, int64_t s2Inner, int64_t ubRowHalf, int64_t mmUbStride)
{
    // score(fp16) 搬入 float 计算区行后半: 目的行距 = 2*ubRowHalf*2B, blockLen = s2Inner*2B
    // dstStride 单位 32B
    AscendC::DataCopyPadExtParams<half> padParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams dataCopymMoutParams;
    dataCopymMoutParams.blockCount = groupInner;
    dataCopymMoutParams.blockLen = s2Inner * sizeof(half);
    dataCopymMoutParams.srcStride = 0;
    dataCopymMoutParams.dstStride = mmUbStride;
    dataCopymMoutParams.rsv = 0;
    LocalTensor<half> mmOutHalfUb = mmOutUb.ReinterpretCast<half>()[ubRowHalf];
    AscendC::DataCopyPad(mmOutHalfUb, mMoutGm[MMout_gmoffset], dataCopymMoutParams, padParams);
}

// int8 路径加权: wq = w ⊗ dq 在 FP16 域预乘(half 硬件乘=正确舍入, 与 golden
// 逐位一致), 再 Cast fp32 广播加权; 与非量化 DoScale 的差异仅此预乘, 其余同构。
// score 的 half→float 原地扩展 Cast 由调用方先行完成(逐行, 行内源/目的无重叠风险)。
// wqTUb = wq 预乘的**落盘区**, weightsTUb = weights 源:
//   - 块级预取路径: 两者不同 buffer(wqTUb=wqRowBuf_), 预取区全程只读
//     ⇒ 同一 base block 内后续行/后续 outerG 读到的 weights 恒为 GM 原值;
//   - 回退路径: 调用方传同一 tensor ⇒ Mul 退化为原地写, 逐字节一致。
// 两个操作数的取值与 count 均不因落盘区改变 ⇒ wq 位型不变。
template <typename T>
__aicore__ inline void DoScaleInt8(const LocalTensor<float>& reduceCacheBuf, LocalTensor<float>& mmOutUb,
                                   LocalTensor<float>& weightsUb, LocalTensor<T>& wqTUb, LocalTensor<T>& weightsTUb,
                                   const LocalTensor<T>& qScaleTUb, LocalTensor<float>& tmpBuff, int64_t groupInner,
                                   int64_t s2Inner, int32_t outerGidx)
{
    // wq 预乘: half 域乘(同宽无重叠), 结果写 wqTUb(落盘区)
    AscendC::Mul(wqTUb, weightsTUb, qScaleTUb, groupInner);
    AscendC::PipeBarrier<PIPE_V>();

    // wq cast fp32: 原地扩展(half 源在 float 目的之后, 逐元素写落后于读, 与 DoScale 同模式)
    AscendC::Cast(weightsUb, wqTUb, RoundMode::CAST_NONE, groupInner);
    AscendC::PipeBarrier<PIPE_V>();

    // weight broadcast: [groupInner, 1] -> [groupInner, 8]
    AscendC::Brcb(tmpBuff, weightsUb, PkiCommon::CeilDiv(groupInner, static_cast<int64_t>(B32_BLOCK_ALIGN_NUM)),
                  {1, B32_VEC_REPEAT_STRIDE});
    AscendC::PipeBarrier<PIPE_V>();

    // do scale: [groupInner, 8] * [groupInner, s2Inner]
    uint64_t countPerRepeat = VEC_REPEAT_BYTES / sizeof(float);
    uint64_t repeatTimes = s2Inner / countPerRepeat;
    for (int32_t i = 0; i < groupInner; i++) {
        if (outerGidx == 0) {
            AscendC::Mul(reduceCacheBuf[i * s2Inner], mmOutUb[i * s2Inner], tmpBuff[i * B32_BLOCK_ALIGN_NUM],
                         countPerRepeat, repeatTimes, {1, 1, 0, B32_VEC_REPEAT_STRIDE, B32_VEC_REPEAT_STRIDE, 0});
        } else {
            AscendC::Mul(mmOutUb[i * s2Inner], mmOutUb[i * s2Inner], tmpBuff[i * B32_BLOCK_ALIGN_NUM], countPerRepeat,
                         repeatTimes, {1, 1, 0, B32_VEC_REPEAT_STRIDE, B32_VEC_REPEAT_STRIDE, 0});
        }
    }

    if (outerGidx != 0) {
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(reduceCacheBuf, mmOutUb, reduceCacheBuf, groupInner * s2Inner);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void DoScale(const LocalTensor<float>& reduceCacheBuf, LocalTensor<float>& mmOutUb,
                               LocalTensor<float>& weightsUb, LocalTensor<T>& weightsTUb, LocalTensor<float>& tmpBuff,
                               int64_t groupInner, int64_t s2Inner, int32_t outerGidx)
{
    // cast bfloat16_t to float
    if constexpr (!IsSameType<T, float>::value) {
        AscendC::Cast(weightsUb, weightsTUb, RoundMode::CAST_NONE, groupInner);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // weight broadcast: [groupInner, 1] -> [groupInner, 8]
    AscendC::Brcb(tmpBuff, weightsUb, PkiCommon::CeilDiv(groupInner, static_cast<int64_t>(B32_BLOCK_ALIGN_NUM)),
                  {1, B32_VEC_REPEAT_STRIDE});
    AscendC::PipeBarrier<PIPE_V>();

    // do scale: [groupInner, 8] * [groupInner, s2Inner]
    uint64_t countPerRepeat = VEC_REPEAT_BYTES / sizeof(float);
    uint64_t repeatTimes = s2Inner / countPerRepeat;
    for (int32_t i = 0; i < groupInner; i++) {
        if (outerGidx == 0) {
            AscendC::Mul(reduceCacheBuf[i * s2Inner], mmOutUb[i * s2Inner], tmpBuff[i * B32_BLOCK_ALIGN_NUM],
                         countPerRepeat, repeatTimes, {1, 1, 0, B32_VEC_REPEAT_STRIDE, B32_VEC_REPEAT_STRIDE, 0});
        } else {
            AscendC::Mul(mmOutUb[i * s2Inner], mmOutUb[i * s2Inner], tmpBuff[i * B32_BLOCK_ALIGN_NUM], countPerRepeat,
                         repeatTimes, {1, 1, 0, B32_VEC_REPEAT_STRIDE, B32_VEC_REPEAT_STRIDE, 0});
        }
    }

    if (outerGidx != 0) {
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(reduceCacheBuf, mmOutUb, reduceCacheBuf, groupInner * s2Inner);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline uint64_t FindNearestPower2(uint64_t value)
{
    if (value <= CONST_TWO) {
        return value;
    } else {
        const uint64_t pow = 63 - AscendC::ScalarCountLeadingZero(
                                      value); // 返回前导0的个数，对于64位整数，最大有效位位置 = 63 - 前导0个数
        return (1 << pow);
    }
}

// dstTensor 需要初始化0
__aicore__ inline void DoReduce(const LocalTensor<float>& srcTensor, LocalTensor<float>& dstTensor, int32_t rNum,
                                int32_t aNum)
{
    if (rNum == 1) {
        AscendC::Adds<float>(dstTensor, srcTensor, 0, aNum);
        AscendC::PipeBarrier<PIPE_V>();
        return;
    }

    uint32_t dichotomizeAddPow = FindNearestPower2(rNum);
    uint32_t dichotomizeAddDiffSize = rNum - dichotomizeAddPow;
    if (dichotomizeAddDiffSize != 0) {
        AscendC::Add(srcTensor, srcTensor, srcTensor[dichotomizeAddPow * aNum], dichotomizeAddDiffSize * aNum);
        AscendC::PipeBarrier<PIPE_V>();
    }
    int32_t nowRows = dichotomizeAddPow;
    while (nowRows > CONST_TWO) {
        nowRows = nowRows / CONST_TWO;
        AscendC::Add(srcTensor, srcTensor, srcTensor[nowRows * aNum], nowRows * aNum);
        AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::Add(dstTensor, srcTensor, srcTensor[aNum], aNum);
    AscendC::PipeBarrier<PIPE_V>();
}

/**
  src: 传入的初始化空间
  eleNum: 需要初始化的元素个数需为64整数倍，元素将被初始化为交错排布的-inf，-1
 */
__aicore__ inline void InitSortOutBuf(const LocalTensor<float>& src, int64_t eleNum)
{
    uint64_t mask1[2] = {0x5555555555555555, 0};
    uint64_t mask0[2] = {0xaaaaaaaaaaaaaaaa, 0};
    int64_t repeatNum = eleNum / B32_VEC_ELM_NUM;
    int64_t forLoop = repeatNum / VEC_REPEAT_MAX;
    int64_t forRemain = repeatNum % VEC_REPEAT_MAX;
    for (int i = 0; i < forLoop; i++) {
        AscendC::Duplicate(src.template ReinterpretCast<int32_t>(), NEG_INF, mask1, VEC_REPEAT_MAX, 1,
                           B32_VEC_REPEAT_STRIDE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(src.template ReinterpretCast<int32_t>(), INVALID_INDEX, mask0, VEC_REPEAT_MAX, 1,
                           B32_VEC_REPEAT_STRIDE);
    }
    if (forRemain > 0) {
        AscendC::Duplicate(src.template ReinterpretCast<int32_t>()[forLoop * VEC_REPEAT_MAX * B32_VEC_ELM_NUM], NEG_INF,
                           mask1, forRemain, 1, B32_VEC_REPEAT_STRIDE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(src.template ReinterpretCast<int32_t>()[forLoop * VEC_REPEAT_MAX * B32_VEC_ELM_NUM],
                           INVALID_INDEX, mask0, forRemain, 1, B32_VEC_REPEAT_STRIDE);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

/**
  src: logits和索引，前logitsNum为logits，后logitsNum为索引
  tmp: 计算使用到的临时空间，大小与src一致
  logitsNum: 排序的元素个数, 暂只支持[128,256,384,512,1024,2048]
 */
__aicore__ inline void SortAll(LocalTensor<float>& src, LocalTensor<float>& tmp, int64_t logitsNum)
{
    int64_t sort32Repeats = logitsNum / BLOCK_BYTES;
    AscendC::Sort32(tmp, src, src[logitsNum].ReinterpretCast<uint32_t>(), sort32Repeats);
    AscendC::PipeBarrier<PIPE_V>();

    int64_t mrgGroups = sort32Repeats;
    int64_t mrgElements = BLOCK_BYTES;
    int64_t i = 0;
    AscendC::LocalTensor<float> srcTensor;
    AscendC::LocalTensor<float> dstTensor;
    while (true) {
        if (i % CONST_TWO == 0) {
            srcTensor = tmp;
            dstTensor = src;
        } else {
            srcTensor = src;
            dstTensor = tmp;
        }
        AscendC::MrgSort4Info params;
        params.elementLengths[0] = mrgElements;
        params.elementLengths[MRG_QUE_1] = mrgElements;
        params.elementLengths[MRG_QUE_2] = mrgElements;
        params.elementLengths[MRG_QUE_3] = mrgElements;
        params.ifExhaustedSuspension = false;
        params.validBit = 0b1111;

        AscendC::MrgSortSrcList<float> srcList;
        srcList.src1 = srcTensor[0];
        srcList.src2 = srcTensor[MRG_QUE_1 * VALUE_AND_INDEX_NUM * mrgElements];
        srcList.src3 = srcTensor[MRG_QUE_2 * VALUE_AND_INDEX_NUM * mrgElements];
        srcList.src4 = srcTensor[MRG_QUE_3 * VALUE_AND_INDEX_NUM * mrgElements];
        if (mrgGroups <= MRG_BLOCK_4) {
            params.repeatTimes = 1;
            if (mrgGroups == 1) {
                break;
            } else if (mrgGroups == MRG_BLOCK_2) {
                params.validBit = 0b0011;
            } else if (mrgGroups == MRG_BLOCK_3) {
                params.validBit = 0b0111;
            } else if (mrgGroups == MRG_BLOCK_4) {
                params.validBit = 0b1111;
            }
            AscendC::MrgSort<float>(dstTensor, srcList, params);
            i += 1;
            AscendC::PipeBarrier<PIPE_V>();
            break;
        } else {
            params.repeatTimes = mrgGroups / MRG_BLOCK_4;
            AscendC::MrgSort<float>(dstTensor, srcList, params);
            i += 1;
            mrgElements = mrgElements * MRG_BLOCK_4;
            mrgGroups = mrgGroups / MRG_BLOCK_4;
        }
        AscendC::PipeBarrier<PIPE_V>();
    }
    if (i % CONST_TWO == 0) {
        AscendC::DataCopy(src, tmp, logitsNum * VALUE_AND_INDEX_NUM);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

/**
  dst: 输出全排序的结果，排布方式为value，index
  srcValue：输入的待排序浮点数
  srcIndex：浮点数的索引
  tmp: 计算使用到的临时空间，大小为srcValue+srcIndex
  logitsNum: 排序的元素个数
 */
__aicore__ inline void SortAll(LocalTensor<float>& dst, LocalTensor<float>& srcValue, LocalTensor<uint32_t>& srcIndex,
                               LocalTensor<float>& tmpTensor, int64_t logitsNum)
{
    int64_t sort32Repeats = logitsNum / BLOCK_BYTES;
    AscendC::Sort<float, true>(dst, srcValue, srcIndex, tmpTensor, sort32Repeats);
    AscendC::PipeBarrier<PIPE_V>();
}

/**
  mrgDst: 合并进的Tensor
  mrgSrc: 待合并的Tensor
  tmpTensor：空间为mrgDst+mrgSrc
 */
__aicore__ inline void MergeSort(const LocalTensor<float>& mrgDst, int32_t mrgDstNum, LocalTensor<float>& mrgSrc,
                                 int32_t mrgSrcNum, LocalTensor<float>& tmpTensor)
{
    if (mrgDstNum <= 3072) { // 3072: threshold of data size for different processing strategy
        AscendC::MrgSort4Info params;
        // src 顺序: 累加器(mrgDst)必须在 src1, 新块(mrgSrc)在 src2, 否则输出不完整污染累加器
        params.elementLengths[MRG_QUE_0] = mrgDstNum;
        params.elementLengths[MRG_QUE_1] = mrgSrcNum;
        params.ifExhaustedSuspension = false;
        params.validBit = 0b0011;
        params.repeatTimes = 1;

        AscendC::MrgSortSrcList<float> srcList;
        srcList.src1 = mrgDst;
        srcList.src2 = mrgSrc;

        AscendC::MrgSort<float>(tmpTensor, srcList, params);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopy(mrgDst, tmpTensor, mrgDstNum * VALUE_AND_INDEX_NUM);
        AscendC::PipeBarrier<PIPE_V>();
    } else {
        // 4 路归并: 累加器 mrgDst 精确三等分(各段>=1, 完整覆盖 [0, mrgDstNum)) + 新块 mrgSrc,
        // 避免 1024 整除切分时尾部被静默丢弃(输出混入无效索引)或出现零长度队列。
        int64_t len1 = (mrgDstNum + 2) / 3;
        int64_t len2 = (mrgDstNum - len1 + 1) / 2;
        int64_t len3 = mrgDstNum - len1 - len2;

        AscendC::MrgSort4Info params;
        params.elementLengths[MRG_QUE_0] = len1;
        params.elementLengths[MRG_QUE_1] = len2;
        params.elementLengths[MRG_QUE_2] = len3;
        params.elementLengths[MRG_QUE_3] = mrgSrcNum;

        params.ifExhaustedSuspension = false;
        params.validBit = 0b1111;
        params.repeatTimes = 1;

        AscendC::MrgSortSrcList<float> srcList;
        srcList.src1 = mrgDst[0];
        srcList.src2 = mrgDst[len1 * VALUE_AND_INDEX_NUM];
        srcList.src3 = mrgDst[(len1 + len2) * VALUE_AND_INDEX_NUM];
        srcList.src4 = mrgSrc;

        AscendC::MrgSort<float>(tmpTensor, srcList, params);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopy(mrgDst, tmpTensor, mrgDstNum * VALUE_AND_INDEX_NUM);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

/**
  MergeSort 的无回拷乒乓版: 归并(src1=accCur 累加器, src2=mrgSrc 新块, 顺序与
  回拷版一致保证稳定保序)全量输出(mrgDstNum+mrgSrcNum 对)写 accNxt, 由调用方
  翻转 parity 跟踪累加器位置, 省去 DataCopy 回拷与其后的 PipeBarrier。
  约束: accNxt 容量 >= (mrgDstNum+mrgSrcNum)*2 float(累加器行 spill 区或 sortPPBuf_);
  accCur/accNxt/mrgSrc 两两地址不重叠(MrgSort 不支持 src/dst 重叠); mrgDstNum<=3072
  (与回拷版小分量分支同口径, 调用侧已限定非 over2K, virTopK<=2048)。
  MrgSort 流式输出的前 mrgDstNum 对与回拷版截断后的内容逐位一致。
 */
__aicore__ inline void MergeSortPP(const LocalTensor<float>& accCur, LocalTensor<float>& accNxt, int32_t mrgDstNum,
                                   LocalTensor<float>& mrgSrc, int32_t mrgSrcNum)
{
    AscendC::MrgSort4Info params;
    // src 顺序: 累加器(accCur)必须在 src1, 新块(mrgSrc)在 src2, 否则输出不完整污染累加器
    params.elementLengths[MRG_QUE_0] = mrgDstNum;
    params.elementLengths[MRG_QUE_1] = mrgSrcNum;
    params.ifExhaustedSuspension = false;
    params.validBit = 0b0011;
    params.repeatTimes = 1;

    AscendC::MrgSortSrcList<float> srcList;
    srcList.src1 = accCur;
    srcList.src2 = mrgSrc;

    AscendC::MrgSort<float>(accNxt, srcList, params);
    AscendC::PipeBarrier<PIPE_V>();
}

/**
 * @brief 合并基础块函数
 * @param dst 归并后的输出, 大小为blockNum * basicBlockSize * 2 * sizeof(float)
 * @param src 基本块输入
 * @param blockNum 基本块的数量
 * @param basicBlockSize 基础块的大小
 * @return 无
 */
__aicore__ inline void MrgBasicBlock(const LocalTensor<float>& dst, const LocalTensor<float>& src, int64_t blockNum,
                                     int64_t basicBlockSize)
{
    // 初始化合并排序参数
    AscendC::MrgSort4Info params;
    params.elementLengths[MRG_QUE_0] = basicBlockSize;
    params.elementLengths[MRG_QUE_1] = basicBlockSize;
    params.elementLengths[MRG_QUE_2] = basicBlockSize;
    params.elementLengths[MRG_QUE_3] = basicBlockSize;
    params.ifExhaustedSuspension = false;
    // 根据块的数量设置有效位
    if (blockNum == MRG_BLOCK_2) {
        params.validBit = 0b0011;
    } else if (blockNum == MRG_BLOCK_3) {
        params.validBit = 0b0111;
    } else if (blockNum == MRG_BLOCK_4) {
        params.validBit = 0b1111;
    } else {
        AscendC::DataCopy(dst, src, basicBlockSize * VALUE_AND_INDEX_NUM);
        return;
    }
    // 初始化源列表
    AscendC::MrgSortSrcList<float> srcList;
    srcList.src1 = src[0];
    srcList.src2 = src[basicBlockSize * VALUE_AND_INDEX_NUM * MRG_QUE_1];
    srcList.src3 = src[basicBlockSize * VALUE_AND_INDEX_NUM * MRG_QUE_2];
    srcList.src4 = src[basicBlockSize * VALUE_AND_INDEX_NUM * MRG_QUE_3];
    // 执行合并排序
    AscendC::MrgSort<float>(dst, srcList, params);
}

/**
 * @brief 从两个队列中选择topk
 * @param dst 已经归并好的topk数据
 * @param needsMerging 需要合并的有序数据
 * @param tmp 临时空间
 * @param topk topk的元素个数
 * @param mergSize 待合并的元素个数
 * @return 无
 */
template <bool needMrg = true>
__aicore__ inline void SparseTopK(const LocalTensor<float>& dst, const LocalTensor<float>& needsMerging,
                                  const LocalTensor<float>& tmp, int64_t topk, int64_t mergSize)
{
    // 如果不需要合并，则直接复制数据
    if (!needMrg) {
        AscendC::DataCopy(dst, needsMerging, mergSize * VALUE_AND_INDEX_NUM);
        return;
    }
    // 初始化合并排序参数
    AscendC::MrgSort4Info params;
    params.elementLengths[0] = topk;
    params.elementLengths[1] = mergSize;
    params.ifExhaustedSuspension = (topk == mergSize);
    params.validBit = 0b0011;
    // 初始化源列表
    AscendC::MrgSortSrcList<float> srcList;
    srcList.src1 = dst;
    srcList.src2 = needsMerging;
    // 执行合并排序
    AscendC::MrgSort<float>(tmp, srcList, params);
    AscendC::PipeBarrier<PIPE_V>();
    // 将结果复制到目标张量
    AscendC::DataCopy(dst, tmp, topk * VALUE_AND_INDEX_NUM);
}

__aicore__ inline void ExtractIndex(const LocalTensor<uint32_t>& idxULocal, const LocalTensor<uint32_t>& sortLocal,
                                    int64_t extractNum)
{
    AscendC::GatherMaskParams gatherMaskParams;
    gatherMaskParams.repeatTimes = Ceil(extractNum * sizeof(float) * VALUE_AND_INDEX_NUM, VEC_REPEAT_BYTES);
    gatherMaskParams.src0BlockStride = 1;
    gatherMaskParams.src0RepeatStride = B32_VEC_REPEAT_STRIDE;
    gatherMaskParams.src1RepeatStride = 0;
    uint64_t rsvdCnt = 0;    // 用于保存筛选后保留下来的元素个数
    uint8_t src1Pattern = 2; // 固定模式2,表示筛选出奇数索引的数
    AscendC::GatherMask(idxULocal, sortLocal, src1Pattern, false, static_cast<uint32_t>(0), gatherMaskParams, rsvdCnt);
    AscendC::PipeBarrier<PIPE_V>();
}

template <HardEvent event>
__aicore__ inline void SetWaitFlag(HardEvent evt)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    AscendC::SetFlag<event>(eventId);
    AscendC::WaitFlag<event>(eventId);
}

} // namespace PkiServiceVec
#endif // POOL_KEY_INDEXER_VECTOR_H
