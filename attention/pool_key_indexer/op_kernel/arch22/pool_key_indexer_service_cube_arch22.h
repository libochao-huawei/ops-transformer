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
 * \file pool_key_indexer_service_cube_arch22.h
 * \brief use 5 buffer for matmul l1, better pipeline
 */
#ifndef POOL_KEY_INDEXER_SERVICE_CUBE_ARCH22_H
#define POOL_KEY_INDEXER_SERVICE_CUBE_ARCH22_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../pool_key_indexer_common.h"

namespace PkiKernel {
using namespace PkiCommon;

// int8 fixpipe 常量 CSR 写外提开关。1=把 Fixp() int8 分支里两条
// **全 kernel 常量**的 CSR 写(SetFixpipeNz2ndFlag(1,1,1) → set_nd_para、
// SetFixpipePreQuantFlag(0x3a800000) → set_quant_pre)外提到 AllocEventID() 一次;
// 0=每次 Fixp 重写(基线; 每 base block 8 次)。
// 数值 bit-exact(依据 CANN 实现级核对):
//   - 两个 CSR 都是 AIC 侧粘滞寄存器(dav_c220/kernel_operator_fixpipe_impl.h:67-84),
//     写一次即保持到被覆写;
//   - 本 kernel 内 nd_para / quant_pre 的**唯一写者就是这两行**: 全仓 dav_c220 只有
//     三个入口(SetFixpipeNz2ndFlagImpl / SetFixpipePreQuantFlagImpl、set_spr 的
//     SetNdParaImpl/SetQuantPreImpl、cube_others 的 SetNdPARA/SetQuantPRE), 后两组
//     本算子从不调用; 实际出数的 DataCopyL0C2GMImpl(data_copy_impl.h:1095+)只把
//     quantPre/nz2ndEn/reluPre/unitFlag 作 copy_matrix_cc_to_gm 的**指令字段**传入,
//     全程不改写这两个 CSR(会内部 SetNdParaImpl 的是 Fixpipe<> API 路径的
//     FixpipeL0C2GMImpl:256-262, arch22 PKI 不用它, arch35 才用);
//   - SetMMLayoutTransform 写的是 set_ctrl 的 MM_LAYOUT_MODE_BIT(kernel_reg.h:190),
//     Mmad/LoadData 写的是 set_fmatrix / set_fmatrix_b / set_padding
//     (dav_c220/kernel_operator_mm_impl.h:480/499/515)⇒ 三者与 nd_para/quant_pre
//     是**不同 CSR**, 块间不存在复位者;
//   - 外提点取 AllocEventID() 的 SetMMLayoutTransform(true) **之后**(保守次序),
//     且 AllocEventID 在 ProcessMain 首部、早于任何 Fixp。
// **只改 int8 分支**。bf16 分支的 SetFixpipeNz2ndFlag 原样不动 ⇒
//      bf16 的指令序列与 CSR 写次数完全不变, 平凡满足"bf16 不劣化/不受影响";
//      外提处用 if constexpr (LIT::isInt8) 包住, bf16 实例化不产生 quant_pre 写
//      (bf16 本就不写 quant_pre)。
// ⚠ 失效前提: 若将来把 arch22 的出数改成 Fixpipe<> API, 该 API 内部会
//   SetNdParaImpl(ndPara) 覆写 nd_para ⇒ 本外提即失效, 必须同时关掉本开关。
#ifndef PKI_FIXPIPE_CSR_HOIST
#define PKI_FIXPIPE_CSR_HOIST 1
#endif

template <typename LIT>
class PoolKeyIndexerServiceCube {
public:
    using Q_T = typename LIT::queryType;
    using K_T = typename LIT::keyType;
    // fixpipe 输出 dtype: int8 走 DEQF16 反量化直写 GM fp16, 其余 fp32
    using MM1_OUT_T = typename std::conditional<LIT::isInt8, half, float>::type;

    __aicore__ inline PoolKeyIndexerServiceCube(){};
    __aicore__ inline void InitBuffers(TPipe* pipe);
    __aicore__ inline void InitMm1GlobalTensor(const GlobalTensor<int32_t>& blkTableGm, const GlobalTensor<K_T>& keyGm,
                                               const GlobalTensor<Q_T>& queryGm,
                                               const GlobalTensor<MM1_OUT_T>& mm1ResGm);
    __aicore__ inline void InitParams(const ConstInfo& constInfo);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void ComputeMm1(const PkiCommon::RunInfo& runInfo);

    static constexpr IsResetLoad3dConfig LOAD3DV2_CONFIG = {true, true}; // isSetFMatrix isSetPadding;
    static constexpr uint64_t KEY_BUF_NUM = 3;
    static constexpr uint64_t QUERY_BUF_NUM = 2;
    static constexpr uint64_t L0_BUF_NUM = 2;

    static constexpr uint32_t KEY_MTE1_MTE2_EVENT = EVENT_ID2;
    static constexpr uint32_t QUERY_MTE1_MTE2_EVENT = EVENT_ID5; // KEY_MTE1_MTE2_EVENT + KEY_BUF_NUM;
    static constexpr uint32_t M_MTE1_EVENT = EVENT_ID3;

    static constexpr uint32_t MTE2_MTE1_EVENT = EVENT_ID2;
    static constexpr uint32_t MTE1_M_EVENT = EVENT_ID2;

    static constexpr uint64_t M_BASIC_BLOCK = 256;
    static constexpr uint64_t D_BASIC_BLOCK = 128;
    static constexpr uint64_t S2_BASIC_BLOCK = 256;

    static constexpr uint64_t M_BASIC_BLOCK_L0 = 128;
    static constexpr uint64_t D_BASIC_BLOCK_L0 = 128;
    static constexpr uint64_t S2_BASIC_BLOCK_L0 = 128;

    // NZ 布局行偏移乘子: fp16 每 16 元素(32B)一行块, int8 为 32 元素(32B);
    // int8 每行 32 元素(32B), dstNzC0Stride 的 16 行对齐保持不变
    static constexpr uint64_t FP16_BLOCK_CUBE = 16;
    static constexpr uint64_t S8_BLOCK_CUBE = 32;
    static constexpr uint64_t QK_BLOCK_CUBE = (sizeof(K_T) == 1) ? S8_BLOCK_CUBE : FP16_BLOCK_CUBE;

    static constexpr uint64_t QUERY_BUFFER_OFFSET = M_BASIC_BLOCK * D_BASIC_BLOCK;
    static constexpr uint64_t KEY_BUFFER_OFFSET = S2_BASIC_BLOCK * D_BASIC_BLOCK;
    static constexpr uint64_t L0AB_BUFFER_OFFSET = M_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0;
    static constexpr uint64_t L0C_BUFFER_OFFSET = M_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0;

protected:
    __aicore__ inline void Fixp(uint64_t s1gGmOffset, uint64_t s2GmOffset, uint64_t s1gL0RealSize,
                                uint64_t s2L0RealSize, const PkiCommon::RunInfo& runInfo);
    __aicore__ inline void ComuteL0c(uint64_t s1gL0RealSize, uint64_t s2L0RealSize, const PkiCommon::RunInfo& runInfo);
    __aicore__ inline void LoadKeyToL0b(uint64_t s2L1Offset, uint64_t s2L1RealSize, uint64_t s2L0RealSize,
                                        const PkiCommon::RunInfo& runInfo);
    __aicore__ inline void LoadQueryToL0a(uint64_t s1gL1Offset, uint64_t s1gL0Offset, uint64_t s1gL1RealSize,
                                          uint64_t s1gL0RealSize, const PkiCommon::RunInfo& runInfo);
    __aicore__ inline void QueryNd2Nz(uint64_t s1gL1RealSize, uint64_t s1gL1Offset, const PkiCommon::RunInfo& runInfo);
    __aicore__ inline void KeyNd2Nz(uint64_t s2L1RealSize, uint64_t s2GmOffset, const PkiCommon::RunInfo& runInfo);
    __aicore__ inline void KeyNd2NzForPA(uint64_t s2L1RealSize, uint64_t s2GmOffset, const PkiCommon::RunInfo& runInfo);
    GlobalTensor<int32_t> blkTableGm_;
    GlobalTensor<K_T> keyGm_;
    GlobalTensor<Q_T> queryGm_;
    GlobalTensor<MM1_OUT_T> mm1ResGm_;

    TBuf<TPosition::A1> bufQL1_;
    LocalTensor<Q_T> queryL1_;
    TBuf<TPosition::B1> bufKeyL1_;
    LocalTensor<K_T> keyL1_;

    TBuf<TPosition::A2> bufQL0_;
    LocalTensor<Q_T> queryL0_;
    TBuf<TPosition::B2> bufKeyL0_;
    LocalTensor<K_T> keyL0_;

    TBuf<TPosition::CO1> bufL0C_;
    LocalTensor<float> cL0_;

    uint64_t keyL1BufIdx_ = 0;
    uint64_t queryL1Mte2BufIdx_ = 0;
    uint64_t queryL1Mte1BufIdx_ = 0;
    uint64_t l0BufIdx_ = 0;

    ConstInfo constInfo_;

private:
    static constexpr bool PAGE_ATTENTION = LIT::pageAttention;
};

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceCube<LIT>::InitParams(const ConstInfo& constInfo)
{
    constInfo_ = constInfo;
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceCube<LIT>::InitBuffers(TPipe* pipe)
{
    pipe->InitBuffer(bufQL1_, QUERY_BUF_NUM * M_BASIC_BLOCK * D_BASIC_BLOCK * sizeof(Q_T));
    queryL1_ = bufQL1_.Get<Q_T>();
    pipe->InitBuffer(bufKeyL1_, KEY_BUF_NUM * S2_BASIC_BLOCK * D_BASIC_BLOCK * sizeof(K_T));
    keyL1_ = bufKeyL1_.Get<K_T>();

    pipe->InitBuffer(bufQL0_, L0_BUF_NUM * M_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0 * sizeof(Q_T));
    queryL0_ = bufQL0_.Get<Q_T>();
    pipe->InitBuffer(bufKeyL0_, L0_BUF_NUM * D_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0 * sizeof(K_T));
    keyL0_ = bufKeyL0_.Get<K_T>();

    pipe->InitBuffer(bufL0C_, L0_BUF_NUM * M_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0 * sizeof(float));
    cL0_ = bufL0C_.Get<float>();
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceCube<LIT>::InitMm1GlobalTensor(const GlobalTensor<int32_t>& blkTableGm,
                                                                           const GlobalTensor<K_T>& keyGm,
                                                                           const GlobalTensor<Q_T>& queryGm,
                                                                           const GlobalTensor<MM1_OUT_T>& mm1ResGm)
{
    blkTableGm_ = blkTableGm;
    keyGm_ = keyGm;
    queryGm_ = queryGm;
    mm1ResGm_ = mm1ResGm;
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceCube<LIT>::ComputeMm1(const PkiCommon::RunInfo& runInfo)
{
    uint64_t s2GmBaseOffset = runInfo.s2Idx * constInfo_.s2BaseSize;
    uint64_t s1gProcessSize = runInfo.actMBaseSize;
    uint64_t s2ProcessSize = runInfo.actualSingleProcessSInnerSize;
    for (uint64_t s2GmOffset = 0; s2GmOffset < s2ProcessSize; s2GmOffset += S2_BASIC_BLOCK) {
        WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
        uint64_t s2L1RealSize =
            s2GmOffset + S2_BASIC_BLOCK > s2ProcessSize ? s2ProcessSize - s2GmOffset : S2_BASIC_BLOCK;
        if (PAGE_ATTENTION) {
            KeyNd2NzForPA(s2L1RealSize, s2GmBaseOffset + s2GmOffset, runInfo);
        } else {
            KeyNd2Nz(s2L1RealSize, s2GmOffset, runInfo);
        }

        SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
        WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
        // s1gProcessSize当前必定不会超过2倍的s1g basic block
        for (uint64_t s1gGmOffset = 0; s1gGmOffset < s1gProcessSize; s1gGmOffset += M_BASIC_BLOCK) {
            uint64_t s1gL1RealSize =
                s1gGmOffset + M_BASIC_BLOCK > s1gProcessSize ? s1gProcessSize - s1gGmOffset : M_BASIC_BLOCK;
            if (runInfo.isFirstS2InnerLoop && s2GmOffset == 0) {
                queryL1Mte2BufIdx_++;
                queryL1Mte1BufIdx_ = queryL1Mte2BufIdx_;
                WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + queryL1Mte2BufIdx_ % QUERY_BUF_NUM);
                QueryNd2Nz(s1gL1RealSize, s1gGmOffset, runInfo);
                SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
                WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
            } else {
                queryL1Mte1BufIdx_ =
                    queryL1Mte2BufIdx_ - (CeilDiv(s1gProcessSize, M_BASIC_BLOCK) - 1 - (s1gGmOffset > 0));
            }
            for (uint64_t s2L1Offset = 0; s2L1Offset < s2L1RealSize; s2L1Offset += S2_BASIC_BLOCK_L0) {
                uint64_t s2L0RealSize =
                    s2L1Offset + S2_BASIC_BLOCK_L0 > s2L1RealSize ? s2L1RealSize - s2L1Offset : S2_BASIC_BLOCK_L0;
                for (uint64_t s1gL1Offset = 0; s1gL1Offset < s1gL1RealSize; s1gL1Offset += M_BASIC_BLOCK_L0) {
                    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);
                    uint64_t s1gL0RealSize =
                        s1gL1Offset + M_BASIC_BLOCK_L0 > s1gL1RealSize ? s1gL1RealSize - s1gL1Offset : M_BASIC_BLOCK_L0;
                    LoadQueryToL0a(s1gGmOffset, s1gL1Offset, s1gL1RealSize, s1gL0RealSize, runInfo);
                    LoadKeyToL0b(s2L1Offset, s2L1RealSize, s2L0RealSize, runInfo);

                    SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
                    WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);

                    ComuteL0c(s1gL0RealSize, s2L0RealSize, runInfo);

                    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);

                    Fixp(s1gGmOffset + s1gL1Offset, s2GmOffset + s2L1Offset, s1gL0RealSize, s2L0RealSize, runInfo);
                    l0BufIdx_++;
                }
            }
            if (s2GmOffset + S2_BASIC_BLOCK >= s2ProcessSize && runInfo.isLastS2InnerLoop) {
                SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + queryL1Mte1BufIdx_ % QUERY_BUF_NUM);
            }
        }

        SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
        keyL1BufIdx_++;
    }
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceCube<LIT>::KeyNd2Nz(uint64_t s2L1RealSize, uint64_t s2GmOffset,
                                                                const PkiCommon::RunInfo& runInfo)
{
    // 非PA key (B,S2_pool,N2,D)/(T2_pool,N2,D): S2 维本身即池, 每池一行, 行距 = headDim
    uint64_t s2L1Offset = 0;
    while (s2L1Offset < s2L1RealSize) {
        uint64_t keyGmOffset = runInfo.tensorKeyOffset + (s2GmOffset + s2L1Offset) * constInfo_.headDim;
        // 搬运按照S2_BASIC_BLOCK_L0*D_BASIC_BLOCK_L0的方式在l1上排布, 方便后续mte1
        // 根据s2的offset判断当前属于前一个L0分型还是后一个L0分型，暂时只支持两个分型
        uint64_t s2Mte2Size = (s2L1RealSize <= S2_BASIC_BLOCK_L0 || s2L1Offset >= S2_BASIC_BLOCK_L0) ?
                                  s2L1RealSize - s2L1Offset :
                                  S2_BASIC_BLOCK_L0 - s2L1Offset;

        Nd2NzParams nd2nzPara;
        nd2nzPara.ndNum = 1;
        nd2nzPara.nValue = s2Mte2Size; // 行数(池数)
        nd2nzPara.dValue = constInfo_.headDim;
        nd2nzPara.srcDValue = constInfo_.headDim;
        nd2nzPara.dstNzC0Stride = s2L1Offset >= S2_BASIC_BLOCK_L0 ?
                                      CeilAlign(s2L1RealSize - S2_BASIC_BLOCK_L0, (uint64_t)BLOCK_CUBE) :
                                      (s2L1RealSize > S2_BASIC_BLOCK_L0 ?
                                           S2_BASIC_BLOCK_L0 :
                                           CeilAlign(s2L1RealSize, (uint64_t)BLOCK_CUBE)); // 对齐到16 单位block
        nd2nzPara.dstNzNStride = 1;
        nd2nzPara.srcNdMatrixStride = 0;
        nd2nzPara.dstNzMatrixStride = 0;
        // NZ 行块偏移乘子随 dtype: int8 每行块 32 元素(S8), fp16 为 16
        DataCopy(keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * KEY_BUFFER_OFFSET +
                        (s2L1Offset >= S2_BASIC_BLOCK_L0 ?
                             S2_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0 + (s2L1Offset - S2_BASIC_BLOCK_L0) * QK_BLOCK_CUBE :
                             s2L1Offset * QK_BLOCK_CUBE)],
                 keyGm_[keyGmOffset], nd2nzPara);

        s2L1Offset += s2Mte2Size;
    }
}

// blkNum, blkSize, N2, D
template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceCube<LIT>::KeyNd2NzForPA(uint64_t s2L1RealSize, uint64_t s2GmOffset,
                                                                     const PkiCommon::RunInfo& runInfo)
{
    // PA key (blockNum, blockSize, N2, D): 块内每行 = 1 个池, srcDValue=headDim
    // 连续行; actS2SizeOrig 的 token 语义仅用于 causal 可见性换算, 与搬运无关
    uint64_t s2L1Offset = 0;
    while (s2L1Offset < s2L1RealSize) {
        uint64_t s2BlkId = (s2L1Offset + s2GmOffset) / constInfo_.kCacheBlockSize;
        uint64_t s2BlkOffset = (s2L1Offset + s2GmOffset) % constInfo_.kCacheBlockSize;
        uint64_t keyGmOffset =
            blkTableGm_.GetValue(runInfo.bIdx * constInfo_.maxBlockNumPerBatch + s2BlkId) * constInfo_.keyStride0 +
            s2BlkOffset * constInfo_.headDim;
        // 搬运按照S2_BASIC_BLOCK_L0*D_BASIC_BLOCK_L0的方式在l1上排布, 方便后续mte1
        // 根据s2的offset判断当前属于前一个L0分型还是后一个L0分型，暂时只支持两个分型
        uint64_t s2Mte2Size = (s2L1RealSize <= S2_BASIC_BLOCK_L0 || s2L1Offset >= S2_BASIC_BLOCK_L0) ?
                                  s2L1RealSize - s2L1Offset :
                                  S2_BASIC_BLOCK_L0 - s2L1Offset;
        s2Mte2Size = s2BlkOffset + s2Mte2Size >= constInfo_.kCacheBlockSize ? constInfo_.kCacheBlockSize - s2BlkOffset :
                                                                              s2Mte2Size;
        Nd2NzParams nd2nzPara;
        nd2nzPara.ndNum = 1;
        nd2nzPara.nValue = s2Mte2Size; // 行数(池数)
        nd2nzPara.dValue = constInfo_.headDim;
        nd2nzPara.srcDValue = constInfo_.headDim;
        nd2nzPara.dstNzC0Stride = s2L1Offset >= S2_BASIC_BLOCK_L0 ?
                                      CeilAlign(s2L1RealSize - S2_BASIC_BLOCK_L0, (uint64_t)BLOCK_CUBE) :
                                      (s2L1RealSize > S2_BASIC_BLOCK_L0 ?
                                           S2_BASIC_BLOCK_L0 :
                                           CeilAlign(s2L1RealSize, (uint64_t)BLOCK_CUBE)); // 对齐到16 单位block
        nd2nzPara.dstNzNStride = 1;
        nd2nzPara.srcNdMatrixStride = 0;
        nd2nzPara.dstNzMatrixStride = 0;
        // NZ 行块偏移乘子随 dtype: int8 每行块 32 元素(S8), fp16 为 16
        DataCopy(keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * KEY_BUFFER_OFFSET +
                        (s2L1Offset >= S2_BASIC_BLOCK_L0 ?
                             S2_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0 + (s2L1Offset - S2_BASIC_BLOCK_L0) * QK_BLOCK_CUBE :
                             s2L1Offset * QK_BLOCK_CUBE)],
                 keyGm_[keyGmOffset], nd2nzPara);

        s2L1Offset += s2Mte2Size;
    }
}

// batch, s1, n2, g, d
template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceCube<LIT>::QueryNd2Nz(uint64_t s1gL1RealSize, uint64_t s1gGmOffset,
                                                                  const PkiCommon::RunInfo& runInfo)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = s1gL1RealSize; // 行数
    nd2nzPara.dValue = constInfo_.headDim;
    nd2nzPara.srcDValue = constInfo_.headDim;
    nd2nzPara.dstNzC0Stride = CeilAlign(s1gL1RealSize, (uint64_t)BLOCK_CUBE); // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    // 默认一块buf最多放两份
    DataCopy(queryL1_[(queryL1Mte2BufIdx_ % QUERY_BUF_NUM) * QUERY_BUFFER_OFFSET],
             queryGm_[runInfo.tensorQueryOffset + s1gGmOffset * constInfo_.headDim], nd2nzPara);
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceCube<LIT>::LoadQueryToL0a(uint64_t s1gGmOffset, uint64_t s1gL1Offset,
                                                                      uint64_t s1gL1RealSize, uint64_t s1gL0RealSize,
                                                                      const PkiCommon::RunInfo& runInfo)
{
    LoadData3DParamsV2<Q_T> loadData3DParams;
    // SetFmatrixParams
    loadData3DParams.l1H = CeilDiv(s1gL1RealSize, BLOCK_CUBE); // Hin=M1=8
    loadData3DParams.l1W = BLOCK_CUBE;                         // Win=M0
    loadData3DParams.channelSize = constInfo_.headDim;         // Cin=K

    loadData3DParams.padList[0] = 0;
    loadData3DParams.padList[1] = 0;
    loadData3DParams.padList[2] = 0;
    loadData3DParams.padList[3] = 255; // 尾部数据不影响滑窗的结果

    // SetLoadToA0Params
    loadData3DParams.mExtension = CeilAlign(s1gL0RealSize, BLOCK_CUBE); // M height维度目的
    loadData3DParams.kExtension = constInfo_.headDim;                   // K width维度目的
    loadData3DParams.mStartPt = s1gL1Offset;
    loadData3DParams.kStartPt = 0;
    loadData3DParams.strideW = 1;
    loadData3DParams.strideH = 1;
    loadData3DParams.filterW = 1;
    loadData3DParams.filterSizeW = (1 >> 8) & 255;
    loadData3DParams.filterH = 1;
    loadData3DParams.filterSizeH = (1 >> 8) & 255;
    loadData3DParams.dilationFilterW = 1;
    loadData3DParams.dilationFilterH = 1;
    loadData3DParams.enTranspose = 0;
    loadData3DParams.fMatrixCtrl = 0;

    LoadData<Q_T, LOAD3DV2_CONFIG>(queryL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET],
                                   queryL1_[(queryL1Mte1BufIdx_ % QUERY_BUF_NUM) * QUERY_BUFFER_OFFSET],
                                   loadData3DParams);
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceCube<LIT>::LoadKeyToL0b(uint64_t s2L1Offset, uint64_t s2L1RealSize,
                                                                    uint64_t s2L0RealSize,
                                                                    const PkiCommon::RunInfo& runInfo)
{
    uint64_t keyL1Offset = s2L1Offset >= S2_BASIC_BLOCK_L0 ? S2_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0 : 0;
    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    // K 向因子按 dtype 块宽: int8 128/32=4(repeatTimes 折半), fp16 128/16=8
    loadData2DParams.repeatTimes = CeilDiv(s2L0RealSize, BLOCK_CUBE) * CeilDiv(constInfo_.headDim, QK_BLOCK_CUBE);
    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = false;
    LoadData(keyL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET],
             keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * KEY_BUFFER_OFFSET + keyL1Offset], loadData2DParams);
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceCube<LIT>::ComuteL0c(uint64_t s1gL0RealSize, uint64_t s2L0RealSize,
                                                                 const PkiCommon::RunInfo& runInfo)
{
    MmadParams mmadParams;
    mmadParams.m = CeilAlign(s1gL0RealSize, BLOCK_CUBE);
    mmadParams.n = s2L0RealSize;
    mmadParams.k = constInfo_.headDim;
    mmadParams.cmatrixInitVal = true;
    mmadParams.cmatrixSource = false;
    mmadParams.unitFlag = 0b11;
    if constexpr (LIT::isInt8) {
        // int8×int8 → int32 精确累加(|q̂k̂| ≤ 128·127² < 2^31), cL0_ 按位型重解释
        Mmad(cL0_.ReinterpretCast<int32_t>()[(l0BufIdx_ % L0_BUF_NUM) * L0C_BUFFER_OFFSET],
             queryL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET],
             keyL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET], mmadParams);
    } else {
        Mmad(cL0_[(l0BufIdx_ % L0_BUF_NUM) * L0C_BUFFER_OFFSET],
             queryL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET],
             keyL0_[(l0BufIdx_ % L0_BUF_NUM) * L0AB_BUFFER_OFFSET], mmadParams);
    }
    if ((mmadParams.m / 16) * (mmadParams.n / 16) < 10) {
        PipeBarrier<PIPE_M>();
    }
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceCube<LIT>::Fixp(uint64_t s1gGmOffset, uint64_t s2GmOffset,
                                                            uint64_t s1gL0RealSize, uint64_t s2L0RealSize,
                                                            const PkiCommon::RunInfo& runInfo)
{
    if constexpr (LIT::isInt8) {
        // int8: L0C int32 --DEQF16(×2^-10)+reluPre--> GM fp16
        // quantPre/reluPre/SetFixpipePreQuantFlag 参数组; nz2nd 直写 GM 沿用
        // 非量化路径; unitFlag 取 GM nz2nd 输出路径的 0b11(与 quantPre 正交,
        // 若出现错数优先复核此项)
        AscendC::DataCopyCO12DstParams intriParams;
        intriParams.mSize = CeilAlign(s1gL0RealSize, BLOCK_CUBE);
        intriParams.nSize = s2L0RealSize; // 与非量化同口径, 硬件按 nSize 精确写列
        intriParams.dstStride = runInfo.actualSingleProcessSInnerSizeAlign;
        intriParams.srcStride = CeilAlign(s1gL0RealSize, BLOCK_CUBE);
        intriParams.quantPre = QuantMode_t::DEQF16; // int32 → fp16 反量化
        intriParams.channelSplit = 0;
        intriParams.nz2ndEn = true;
        intriParams.unitFlag = 0b11;
        intriParams.reluPre = 1; // DEQF16 舍入后 ReLU(= golden clamp 顺序)
#if PKI_FIXPIPE_CSR_HOIST
        // 两条常量 CSR 写已外提到 AllocEventID()(见该处注释与开关处的
        // 唯一写者论证)。指令字段(quantPre/nz2ndEn/reluPre/unitFlag/channelSplit)
        // 随 intriParams 每次照传。
#else
        AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
        AscendC::SetFixpipePreQuantFlag(0x3a800000); // FP32 位型的 2^-10
#endif
        AscendC::DataCopy(mm1ResGm_[(runInfo.loop % 2) * constInfo_.mBaseSizeAlign * constInfo_.s2BaseSize +
                                    s1gGmOffset * intriParams.dstStride + s2GmOffset],
                          cL0_.ReinterpretCast<int32_t>()[(l0BufIdx_ % L0_BUF_NUM) * L0C_BUFFER_OFFSET], intriParams);
    } else {
        AscendC::DataCopyCO12DstParams intriParams;
        intriParams.mSize = CeilAlign(s1gL0RealSize, BLOCK_CUBE);
        intriParams.nSize = s2L0RealSize;
        intriParams.dstStride = runInfo.actualSingleProcessSInnerSizeAlign;
        intriParams.srcStride = CeilAlign(s1gL0RealSize, BLOCK_CUBE);
        // set mode according to dtype
        intriParams.quantPre = QuantMode_t::NoQuant;
        intriParams.nz2ndEn = true;
        intriParams.unitFlag = 0b11; // 3 unitflag
        intriParams.reluPre = 1;
        AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
        AscendC::DataCopy(mm1ResGm_[(runInfo.loop % 2) * constInfo_.mBaseSizeAlign * constInfo_.s2BaseSize +
                                    s1gGmOffset * intriParams.dstStride + s2GmOffset],
                          cL0_[(l0BufIdx_ % L0_BUF_NUM) * L0C_BUFFER_OFFSET], intriParams);
    }
}
template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceCube<LIT>::AllocEventID()
{
    SetMMLayoutTransform(true);
#if PKI_FIXPIPE_CSR_HOIST
    // int8 fixpipe 的两条常量 CSR 写外提到此处, 每 kernel 一次(基线是每次 Fixp
    // 重写一次)。次序: 必须在 SetMMLayoutTransform(true) 之后(二者为不同 CSR ——
    // set_ctrl vs set_nd_para/set_quant_pre —— 仍取保守次序)。
    // 只覆盖 int8: bf16 分支的 SetFixpipeNz2ndFlag 保持原位原样, 且 bf16
    // 实例化在此不产生任何 quant_pre CSR 写。
    if constexpr (LIT::isInt8) {
        AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
        AscendC::SetFixpipePreQuantFlag(0x3a800000); // FP32 位型的 2^-10
    }
#endif
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 0);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 1);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 2);

    SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 0);
    SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 1);

    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 0);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 1);
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceCube<LIT>::FreeEventID()
{
    SetMMLayoutTransform(false);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 0);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 1);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 2);

    WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 0);
    WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 1);

    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 0);
    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 1);
}
} // namespace PkiKernel
#endif
