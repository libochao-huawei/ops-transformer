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
 * \file qmm_mx_block_mmad_fragment.h
 * \brief
 */

#pragma once

// 低版本 devkit(如 9.2.0) 的编译环境不含 hifloat4x2_t/AscendC::hif4_scale，
// 以 devkit 版本守卫 HIF4 kernel 代码（CANN >= 9.3 才启用，与 op_host 侧
// ENABLE_ALL_GATHER_MATMUL_DT_HIFLOAT4 守卫配套、阈值一致）
#if ASC_DEVKIT_MAJOR > 9 || (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR >= 3)
#define APACE_HIF4_TYPE_SUPPORTED 1
#else
#define APACE_HIF4_TYPE_SUPPORTED 0
#endif

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_mmad.h"
#include "tensor_api/tensor.h"
#include "blaze/gemm/tile/tile_trait.h"
#include "blaze/gemm/tile/pad_mx_kl1.h"
#include "apace/basic/fragment_tensor/fragment_tensor.h"

namespace Blaze {
namespace Gemm {
namespace Block {

template <uint64_t A_FULL_LOAD_MODE, bool ATOMIC_ADD, class AType_, class LayoutA_, class BType_, class LayoutB_,
          class CType_, class LayoutC_, class BiasType_, class LayoutBias_>
class QmmMxBlockMmadFragment;

// 用于区分 FragmentTensor 与普通连续 tensor，从而在 GM2L1 拷贝时分流：
//   - FragmentTensor → FragmentSliceCopy（跨 GM 段拼装）
//   - 普通 tensor   → asc::te::copy（单段连续）
template <typename T>
struct IsFragmentTensor {
    static constexpr bool value = false;
};
template <uint32_t Dims, uint32_t MaxFragments, typename LayoutFactory, typename ElementType>
struct IsFragmentTensor<Apace::Basic::FragmentTensor<Dims, MaxFragments, LayoutFactory, ElementType>> {
    static constexpr bool value = true;
};

#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510 || __NPU_ARCH__ == 9201))

template <uint64_t A_FULL_LOAD_MODE, bool ATOMIC_ADD, class AType_, class LayoutA_, class BType_, class LayoutB_,
          class CType_, class LayoutC_, class BiasType_, class LayoutBias_>
class QmmMxBlockMmadFragment {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using BiasType = BiasType_;
    using DispatchPolicy = MatmulWithScaleMx<A_FULL_LOAD_MODE, ATOMIC_ADD>;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    static constexpr bool weightNz = IsWeightNz<LayoutB>::value;
    static constexpr bool transA = IsTrans<LayoutA>::value;
    static constexpr bool transB = IsTrans<LayoutB>::value;

#if APACE_HIF4_TYPE_SUPPORTED
    static constexpr bool IS_HIF4 = AscendC::IsSameType<AType, hifloat4x2_t>::value;
    using ScaleType = AscendC::Std::conditional_t<IS_HIF4, AscendC::hif4_scale, AscendC::fp8_e8m0_t>;
#else
    static constexpr bool IS_HIF4 = false;
    using ScaleType = AscendC::fp8_e8m0_t;
#endif
    using ByteScaleType = AscendC::fp8_e8m0_t;

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        GM_ADDR scaleAGmAddr{nullptr};
        GM_ADDR scaleBGmAddr{nullptr};
    };

    struct L1Params {
        uint64_t kL1;
        uint64_t scaleKL1;
        uint64_t l1BufNum;
    };

    __aicore__ inline QmmMxBlockMmadFragment()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_3);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(INPUT_BUFFER_FLAG_0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(INPUT_BUFFER_FLAG_1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1_FLAG_0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1_FLAG_1);
        AscendC::SetMMLayoutTransform(true); // true means column first when fixpipe_l0c2out
    }

    __aicore__ inline ~QmmMxBlockMmadFragment()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(INPUT_BUFFER_FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(INPUT_BUFFER_FLAG_1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1_FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1_FLAG_1);
        AscendC::SetMMLayoutTransform(false); // false means row first when fixpipe_l0c2out
    }

    __aicore__ inline void Init(const ProblemShape& problemShape, const BlockShape& l0TileShape,
                                const L1Params& l1Params, bool isBias, bool dbL0C, uint64_t splitKNum = 1)
    {
        k_ = asc::te::get<IDX_K_IDX>(problemShape);
        kL1_ = l1Params.kL1;
        scaleKL1_ = l1Params.scaleKL1;
        splitKNum_ = splitKNum;
        isBias_ = isBias;
        l1BufNum_ = l1Params.l1BufNum;
        enableL0cPingPong_ = dbL0C;
        uint64_t baseM = asc::te::get<IDX_M_IDX>(l0TileShape);
        baseN_ = asc::te::get<IDX_N_IDX>(l0TileShape);
        baseK_ = asc::te::get<IDX_K_IDX>(l0TileShape);
        constexpr uint64_t sizeShift = (IS_HIF4 || IsFp4<AType>()) ? 1 : 0;
        uint64_t l1HalfBufNum = l1BufNum_ >> 1;
        uint64_t bL1OneBuffer = (baseN_ * kL1_) >> sizeShift;
        uint64_t scaleBL1OneBuffer =
            baseN_ * Blaze::Gemm::CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE) * SCALE_BYTES_PER_GROUP;
        uint64_t biasL1OneBuffer = isBias_ ? baseN_ * sizeof(BiasType) : 0;
        if constexpr (DispatchPolicy::FULL_LOAD_MODE == NONE_FULL_LOAD_MODE) {
            InitNoneFullLoadL1BufferOffsets(baseM, l1HalfBufNum, bL1OneBuffer, scaleBL1OneBuffer);
        } else {
            InitAFullLoadL1BufferOffsets(baseM, l1HalfBufNum, bL1OneBuffer, scaleBL1OneBuffer, biasL1OneBuffer);
        }
        kL1Iter_ = CeilDiv(k_, kL1_);
    }

    template <typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB, typename TensorBias,
              typename TensorC>
    __aicore__ inline void operator()(TensorA const& gmA, TensorB const& gmB, TensorScaleA const& gmScaleA,
                                      TensorScaleB const& gmScaleB, TensorBias const& gmBias, TensorC const& gmC,
                                      BlockShape const& singleShape, uint64_t splitKIdx = 0)
    {
        TileL1L0Param tileL1L0Param;
        tileL1L0Param.curM = asc::te::get<IDX_M_TILEIDX>(singleShape);
        tileL1L0Param.curN = asc::te::get<IDX_N_TILEIDX>(singleShape);
        uint64_t l0cOffset = (l0cPingPong_ & 1) * HALF_L0C_SIZE;
        auto layoutL0C = asc::te::frame_layout_format<asc::te::nz_layout_ptn, AscendC::Std::Int<C0_SIZE_L0C>>{}(
            tileL1L0Param.curM, tileL1L0Param.curN);
        auto tensorL0C =
            asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0c, float>(l0cOffset), layoutL0C);
        const uint64_t scaleKIter = scaleKL1_ / kL1_;
        uint64_t scaleKIterIdx = 0;
        const uint64_t scaleKOffsetStride = Blaze::Gemm::CeilDiv(kL1_, MXFP_DIVISOR_SIZE) * GM_SCALE_C0;
        for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
            ProcessL1Iteration(gmA, gmB, gmScaleA, gmScaleB, gmBias, tensorL0C, tileL1L0Param, iter0,
                               scaleKOffsetStride, scaleKIter, scaleKIterIdx, splitKIdx);
        }
        if (splitKIdx == splitKNum_ - 1) {
            CopyL0CToOut(gmC, tensorL0C);
            if (enableL0cPingPong_) {
                l0cPingPong_++;
            }
        }
    }

private:
    constexpr static uint64_t HALF_L0_SIZE = AscendC::TOTAL_L0A_SIZE / DOUBLE_BUFFER_COUNT;
    constexpr static uint64_t HALF_L0C_SIZE = AscendC::TOTAL_L0C_SIZE / DOUBLE_BUFFER_COUNT;
    constexpr static uint64_t C0_SIZE = (IS_HIF4 || IsFp4<AType>()) ? C0_SIZE_B4 : C0_SIZE_B8;

    // 两个 scale 域不可混用（hif4 精度问题根因）：GM 域（ScaleType/GM_SCALE_C0）按真实 scale 元素数
    // 计算 offset/length；L1/L0 域（ByteScaleType/SCALE_BYTES_PER_GROUP）恒为 1 字节载体，用于
    // 分形与 L1 缓冲预算。SCALE_FRAME_C0 为 Te 分形 C0。
    constexpr static uint64_t GM_SCALE_C0 = IS_HIF4 ? 1 : 2;
    constexpr static uint64_t SCALE_BYTES_PER_GROUP = GM_SCALE_C0 * sizeof(ScaleType);
    constexpr static uint64_t SCALE_FRAME_C0 = 2;
    // 等价于上游 AscendC::Te::GetScaleDivisor<T>()（该接口 devkit 未提供，本地按同一契约以常量表达）：
    // divisor = 数据分形字节数 / 配套 scale 字节数，fp8/fp4 (2B/64K) -> 16，hif4 (4B/64K) -> 8；
    // hif4 取值为本地推导，上板需与 GetScaleDivisor 实际返回值核对。
    template <typename T>
    __aicore__ constexpr static uint64_t ScaleBytesPerGroupOf()
    {
#if APACE_HIF4_TYPE_SUPPORTED
        return AscendC::IsSameType<T, hifloat4x2_t>::value ? 4U : 2U;
#else
        return 2U;
#endif
    }
    template <typename T>
    __aicore__ constexpr static uint64_t ScaleL0OffsetDivisor()
    {
        return 32U / ScaleBytesPerGroupOf<T>();
    }
    constexpr static uint64_t SCALE_BUFFER_NUM = 2;

    using MakeLayoutAL1 =
        AscendC::Std::conditional_t<transA,
                                    asc::te::frame_layout_format<asc::te::zn_layout_ptn, AscendC::Std::Int<C0_SIZE>>,
                                    asc::te::frame_layout_format<asc::te::nz_layout_ptn, AscendC::Std::Int<C0_SIZE>>>;
    using MakeLayoutBL1 =
        AscendC::Std::conditional_t<transB,
                                    asc::te::frame_layout_format<asc::te::zn_layout_ptn, AscendC::Std::Int<C0_SIZE>>,
                                    asc::te::frame_layout_format<asc::te::nz_layout_ptn, AscendC::Std::Int<C0_SIZE>>>;

    struct TileL1L0Param {
        uint64_t curM = 0;
        uint64_t curN = 0;
        uint64_t curGmKL1 = 0;
        uint64_t curPadKL1 = 0; // pad to 64 align
    };

    template <typename TensorScaleA, typename TensorScaleB>
    struct ScalePair {
        TensorScaleA scaleA;
        TensorScaleB scaleB;
    };

    // GM → L1 分段/连续 统一入口：FragmentTensor 走 FragmentSliceCopy，普通 tensor 走单段 asc::te::copy。
    template <typename CopyHandleT, typename TensorDstT, typename GmSrcT>
    __aicore__ inline void CopyGmInL1(CopyHandleT& handle, TensorDstT& dst, GmSrcT const& src)
    {
        if constexpr (IsFragmentTensor<GmSrcT>::value) {
            Apace::Basic::FragmentSliceCopy<false>(handle, dst, src);
        } else {
            asc::te::copy(handle, dst, src);
        }
    }

    template <typename TensorScaleA, typename TensorScaleB>
    __aicore__ inline auto CopyScalesInL1(TensorScaleA const& gmScaleA, TensorScaleB const& gmScaleB,
                                          const TileL1L0Param& tileL1L0Param, uint64_t scaleL1BufId, uint64_t kL1Offset,
                                          uint64_t scaleGmOffset, bool needCopyScale)
    {
        const uint64_t scaleKL1Len = Blaze::Gemm::CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE) * SCALE_BYTES_PER_GROUP;
        auto layoutScaleBL1 = asc::te::make_frame_layout<asc::te::nn_layout_ptn, AscendC::Std::Int<SCALE_FRAME_C0>>(
            scaleKL1Len, tileL1L0Param.curN);
        auto tensorScaleBL1 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, ByteScaleType>(l1BufferScaleBOffset_[scaleL1BufId]),
            layoutScaleBL1);
        if constexpr (DispatchPolicy::FULL_LOAD_MODE == NONE_FULL_LOAD_MODE) {
            return CopyPartialScalesInL1(gmScaleA, gmScaleB, tensorScaleBL1, tileL1L0Param, scaleL1BufId, kL1Offset,
                                         scaleGmOffset, scaleKL1Len, needCopyScale);
        } else {
            return CopyFullLoadScalesInL1(gmScaleA, gmScaleB, tensorScaleBL1, tileL1L0Param, scaleL1BufId, kL1Offset,
                                          scaleGmOffset, needCopyScale);
        }
    }

    template <typename TensorScaleA, typename TensorScaleB, typename TensorScaleBL1>
    __aicore__ inline auto CopyPartialScalesInL1(TensorScaleA const& gmScaleA, TensorScaleB const& gmScaleB,
                                                 TensorScaleBL1& tensorScaleBL1, const TileL1L0Param& tileL1L0Param,
                                                 uint64_t scaleL1BufId, uint64_t kL1Offset, uint64_t scaleGmOffset,
                                                 uint64_t scaleKL1Len, bool needCopyScale)
    {
        auto layoutScaleAL1 = asc::te::make_frame_layout<asc::te::zz_layout_ptn, AscendC::Std::Int<SCALE_FRAME_C0>>(
            tileL1L0Param.curM, scaleKL1Len);
        auto tensorScaleAL1 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, ByteScaleType>(l1BufferScaleAOffset_[scaleL1BufId]),
            layoutScaleAL1);
        if (needCopyScale) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0 + scaleL1BufId);
            const uint64_t curScaleKL1 = Min(scaleKL1_, k_ - kL1Offset);
            uint64_t curScaleKLen = Blaze::Gemm::CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE) * GM_SCALE_C0;
            auto blockScaleA = gmScaleA.slice(asc::te::make_coord(0, scaleGmOffset),
                                              asc::te::make_shape(tileL1L0Param.curM, curScaleKLen));
            auto CopyScaleGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
            CopyGmInL1(CopyScaleGM2L1, tensorScaleAL1, blockScaleA);
            CopyScaleBInL1(gmScaleB, tensorScaleBL1, tileL1L0Param, scaleGmOffset, curScaleKLen);
        }
        return ScalePair<decltype(tensorScaleAL1), decltype(tensorScaleBL1)>{tensorScaleAL1, tensorScaleBL1};
    }

    template <typename TensorScaleA, typename TensorScaleB, typename TensorScaleBL1>
    __aicore__ inline auto CopyFullLoadScalesInL1(TensorScaleA const& gmScaleA, TensorScaleB const& gmScaleB,
                                                  TensorScaleBL1& tensorScaleBL1, const TileL1L0Param& tileL1L0Param,
                                                  uint64_t scaleL1BufId, uint64_t kL1Offset, uint64_t scaleGmOffset,
                                                  bool needCopyScale)
    {
        auto layoutScaleAL1 = asc::te::make_frame_layout<asc::te::zz_layout_ptn, AscendC::Std::Int<SCALE_FRAME_C0>>(
            tileL1L0Param.curM, Blaze::Gemm::CeilDiv(k_, MXFP_DIVISOR_SIZE) * SCALE_BYTES_PER_GROUP);
        auto tensorScaleAL1 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, ByteScaleType>(l1BufferScaleAOffset_[0]), layoutScaleAL1);
        if (needCopyScale) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0 + scaleL1BufId);
            const uint64_t curScaleKL1 = Min(scaleKL1_, k_ - kL1Offset);
            uint64_t curScaleKLen = Blaze::Gemm::CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE) * GM_SCALE_C0;
            CopyScaleBInL1(gmScaleB, tensorScaleBL1, tileL1L0Param, scaleGmOffset, curScaleKLen);
        }
        if (abL1LoopCnt_ == 0) {
            auto CopyScaleGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
            CopyGmInL1(CopyScaleGM2L1, tensorScaleAL1, gmScaleA);
        }
        return ScalePair<decltype(tensorScaleAL1), decltype(tensorScaleBL1)>{tensorScaleAL1, tensorScaleBL1};
    }

    template <typename TensorScaleB, typename TensorScaleBL1>
    __aicore__ inline void CopyScaleBInL1(TensorScaleB const& gmScaleB, TensorScaleBL1& tensorScaleBL1,
                                          const TileL1L0Param& tileL1L0Param, uint64_t scaleGmOffset,
                                          uint64_t curScaleKLen)
    {
        auto CopyScaleGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
        auto gmBlockScaleB = gmScaleB.slice(asc::te::make_coord(scaleGmOffset, 0),
                                            asc::te::make_shape(curScaleKLen, tileL1L0Param.curN));
        asc::te::copy(CopyScaleGM2L1, tensorScaleBL1, gmBlockScaleB);
    }

    template <typename TensorA>
    __aicore__ inline auto CopyAInL1(TensorA const& gmA, const TileL1L0Param& tileL1L0Param, uint64_t l1BufId,
                                     uint64_t iter0)
    {
        auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
        if constexpr (DispatchPolicy::FULL_LOAD_MODE == NONE_FULL_LOAD_MODE) {
            auto layoutAL1 = MakeLayoutAL1{}(tileL1L0Param.curM, tileL1L0Param.curPadKL1);
            auto tensorAL1 = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l1, AType>(l1BufferAOffset_[l1BufId]), layoutAL1);
            auto blockA = gmA.slice(asc::te::make_coord(0, iter0 * kL1_),
                                    asc::te::make_shape(tileL1L0Param.curM, tileL1L0Param.curGmKL1));
            Apace::Basic::PadMxKAL1Zero(tensorAL1, tileL1L0Param.curGmKL1);
            CopyGmInL1(copyGM2L1, tensorAL1, blockA);
            return tensorAL1;
        } else {
            auto layoutAL1 = MakeLayoutAL1{}(tileL1L0Param.curM, Blaze::Gemm::CeilAlign(k_, MXFP_DIVISOR_SIZE));
            auto tensorTotalAL1 = asc::te::make_tensor(
                asc::te::make_mem_ptr<asc::te::location::l1, AType>(l1BufferAOffset_[0]), layoutAL1);
            auto tensorAL1 = tensorTotalAL1.slice(asc::te::make_coord(0, iter0 * kL1_),
                                                  asc::te::make_shape(tileL1L0Param.curM, tileL1L0Param.curPadKL1));
            if (abL1LoopCnt_ < kL1Iter_) {
                auto blockA = gmA.slice(asc::te::make_coord(0, iter0 * kL1_),
                                        asc::te::make_shape(tileL1L0Param.curM, tileL1L0Param.curGmKL1));
                Apace::Basic::PadMxKAL1Zero(tensorAL1, tileL1L0Param.curGmKL1);
                CopyGmInL1(copyGM2L1, tensorAL1, blockA);
            }
            return tensorAL1;
        }
    }

    template <typename TensorBias>
    __aicore__ inline auto CopyBiasInL1(TensorBias const& gmBias, const TileL1L0Param& tileL1L0Param,
                                        uint64_t scaleL1BufId, uint64_t iter0, uint64_t splitKIdx)
    {
        auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
        auto layoutBiasL1 = asc::te::frame_layout_format<asc::te::nd_ext_layout_ptn>{}(
            1UL, Blaze::Gemm::CeilAlign(tileL1L0Param.curN, BLOCK_CUBE));
        auto tensorBiasL1 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, BiasType>(l1BufferBiasOffset_[scaleL1BufId]), layoutBiasL1);
        if (isBias_ && iter0 == 0 && splitKIdx == 0) {
            asc::te::copy(copyGM2L1, tensorBiasL1, gmBias);
        }
        return tensorBiasL1;
    }

    template <typename TensorB>
    __aicore__ inline auto CopyBInL1(TensorB const& gmB, const TileL1L0Param& tileL1L0Param, uint64_t l1BufId,
                                     uint64_t iter0)
    {
        auto copyGM2L1 = asc::te::make_copy(asc::te::copy_gm_to_l1{});
        auto layoutBL1 = MakeLayoutBL1{}(tileL1L0Param.curPadKL1, tileL1L0Param.curN);
        auto tensorBL1 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l1, BType>(l1BufferBOffset_[l1BufId]), layoutBL1);
        auto gmBlockB = gmB.slice(asc::te::make_coord(iter0 * kL1_, 0),
                                  asc::te::make_shape(tileL1L0Param.curGmKL1, tileL1L0Param.curN));
        Blaze::Gemm::Tile::PadMxKBL1::PadZero(tensorBL1, gmBlockB);
        asc::te::copy(copyGM2L1, tensorBL1, gmBlockB);
        return tensorBL1;
    }

    template <typename TensorScaleAL1, typename TensorScaleBL1, typename TensorAL1, typename TensorBL1,
              typename TensorBiasL1, typename TensorL0C>
    __aicore__ inline void Iterate(const TileL1L0Param& tileL1L0Param, uint64_t iter0, uint64_t biasBtOffset,
                                   TensorScaleAL1& tensorScaleAL1, TensorScaleBL1& tensorScaleBL1, TensorAL1& tensorAL1,
                                   TensorBL1& tensorBL1, TensorBiasL1& tensorBiasL1, TensorL0C& tensorL0C,
                                   uint64_t splitKIdx)
    {
        // 从scaleKL1中切出kL1_对应的部分
        auto scaleKL1Len = Blaze::Gemm::CeilDiv(kL1_, MXFP_DIVISOR_SIZE) * SCALE_BYTES_PER_GROUP;
        auto coordScaleKL1 = (iter0 % (scaleKL1_ / kL1_)) * scaleKL1Len;
        auto tensorBlockScaleBL1 = tensorScaleBL1.slice(asc::te::make_coord(coordScaleKL1, 0),
                                                        asc::te::make_shape(scaleKL1Len, tileL1L0Param.curN));
        if constexpr (DispatchPolicy::FULL_LOAD_MODE != NONE_FULL_LOAD_MODE) {
            coordScaleKL1 = iter0 * scaleKL1Len;
        }
        auto tensorBlockScaleAL1 = tensorScaleAL1.slice(asc::te::make_coord(0, coordScaleKL1),
                                                        asc::te::make_shape(tileL1L0Param.curM, scaleKL1Len));

        auto layoutBt = asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(
            1UL, Blaze::Gemm::CeilAlign(tileL1L0Param.curN, BLOCK_CUBE));
        auto tensorBt =
            asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::bias, float>(biasBtOffset), layoutBt);

        uint64_t kL0Iter = Blaze::Gemm::CeilDiv(tileL1L0Param.curGmKL1, baseK_);
        for (uint16_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
            ProcessL0Iteration(tileL1L0Param, iter0, iter1, kL0Iter, splitKIdx, tensorBlockScaleAL1,
                               tensorBlockScaleBL1, tensorAL1, tensorBL1, tensorBiasL1, tensorL0C, tensorBt);
        }
    }

    template <typename TensorScaleAL1, typename TensorScaleBL1, typename TensorAL1, typename TensorBL1,
              typename TensorBiasL1, typename TensorL0C, typename TensorBT>
    __aicore__ inline void ProcessL0Iteration(const TileL1L0Param& tileL1L0Param, uint64_t iter0, uint16_t iter1,
                                              uint64_t kL0Iter, uint64_t splitKIdx, TensorScaleAL1& tensorBlockScaleAL1,
                                              TensorScaleBL1& tensorBlockScaleBL1, TensorAL1& tensorAL1,
                                              TensorBL1& tensorBL1, TensorBiasL1& tensorBiasL1, TensorL0C& tensorL0C,
                                              TensorBT& tensorBt)
    {
        auto curKL0 =
            (iter1 * baseK_ + baseK_ > tileL1L0Param.curPadKL1) ? (tileL1L0Param.curPadKL1 - iter1 * baseK_) : baseK_;
        // Load data to L0 and open DB
        uint64_t l0Offset = HALF_L0_SIZE * (l0PingPong_ & 0x1);
        uint16_t mte1WaitMFlag = (l0PingPong_ & 0x1) + M_MTE1_FLAG_0;
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(mte1WaitMFlag);

        // A, ScaleA L1->L0
        auto tensorAL0 = CopyAInL0(tensorAL1, tensorBlockScaleAL1, tileL1L0Param, iter1, curKL0, l0Offset);

        // bias L1->BT
        if (NeedBias(iter0, iter1, splitKIdx)) {
            auto CopyL12BT = asc::te::make_copy(asc::te::copy_l1_to_biastable{});
            asc::te::copy(CopyL12BT, tensorBt, tensorBiasL1);
        }

        // B, scaleB L1->L0
        auto tensorBL0 = CopyBInL0(tensorBL1, tensorBlockScaleBL1, tileL1L0Param, iter1, curKL0, l0Offset);

        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);

        Mmad(tileL1L0Param, iter0, iter1, kL0Iter, curKL0, tensorL0C, tensorAL0, tensorBL0, tensorBt, splitKIdx);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(mte1WaitMFlag);
        l0PingPong_++;
    }

    template <typename TensorAL1, typename TensorScaleAL1>
    __aicore__ inline auto CopyAInL0(TensorAL1& tensorAL1, TensorScaleAL1& tensorBlockScaleAL1,
                                     const TileL1L0Param& tileL1L0Param, uint64_t iter1, uint64_t curKL0,
                                     uint64_t l0Offset)
    {
        // A, ScaleA L1->L0
        auto layoutAL0 =
            asc::te::make_frame_layout<asc::te::nz_layout_ptn, AscendC::Std::Int<C0_SIZE>>(tileL1L0Param.curM, curKL0);
        auto tensorAL0 =
            asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0a, AType>(l0Offset), layoutAL0);
        auto tensorBlockAL1 =
            tensorAL1.slice(asc::te::make_coord(0, iter1 * baseK_), asc::te::make_shape(tileL1L0Param.curM, curKL0));
        auto CopyL12L0A = asc::te::make_copy(asc::te::copy_l1_to_l0a{});
        asc::te::copy(CopyL12L0A, tensorAL0, tensorBlockAL1);

        auto CopyL12L0ScaleA = asc::te::make_copy(asc::te::copy_l1_to_l0scalea{});
        auto layoutScaleAL0 = asc::te::make_frame_layout<asc::te::zz_layout_ptn, AscendC::Std::Int<SCALE_FRAME_C0>>(
            tileL1L0Param.curM, Blaze::Gemm::CeilDiv(curKL0, MXFP_DIVISOR_SIZE) * SCALE_BYTES_PER_GROUP);
        auto tensorScaleAL0 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l0scalea, ByteScaleType>(l0Offset / ScaleL0OffsetDivisor<AType>()),
            layoutScaleAL0);
        asc::te::copy(
            CopyL12L0ScaleA, tensorScaleAL0,
            tensorBlockScaleAL1.slice(
                asc::te::make_coord(0, iter1 * Blaze::Gemm::CeilDiv(baseK_, MXFP_DIVISOR_SIZE) * SCALE_BYTES_PER_GROUP),
                asc::te::make_shape(tileL1L0Param.curM,
                                    Blaze::Gemm::CeilDiv(curKL0, MXFP_DIVISOR_SIZE) * SCALE_BYTES_PER_GROUP)));
        return tensorAL0;
    }

    template <typename TensorBL1, typename TensorScaleBL1>
    __aicore__ inline auto CopyBInL0(TensorBL1& tensorBL1, TensorScaleBL1& tensorBlockScaleBL1,
                                     const TileL1L0Param& tileL1L0Param, uint64_t iter1, uint64_t curKL0,
                                     uint64_t l0Offset)
    {
        // B, scaleB L1->L0
        auto layoutBL0 =
            asc::te::make_frame_layout<asc::te::zn_layout_ptn, AscendC::Std::Int<C0_SIZE>>(curKL0, tileL1L0Param.curN);
        auto tensorBL0 =
            asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::l0b, BType>(l0Offset), layoutBL0);
        auto tensorBlockBL1 =
            tensorBL1.slice(asc::te::make_coord(iter1 * baseK_, 0), asc::te::make_shape(curKL0, tileL1L0Param.curN));
        auto CopyL12L0B = asc::te::make_copy(asc::te::copy_l1_to_l0b{});
        asc::te::copy(CopyL12L0B, tensorBL0, tensorBlockBL1);

        auto CopyL12L0ScaleB = asc::te::make_copy(asc::te::copy_l1_to_l0scaleb{});
        auto layoutScaleBL0 = asc::te::make_frame_layout<asc::te::nn_layout_ptn, AscendC::Std::Int<SCALE_FRAME_C0>>(
            Blaze::Gemm::CeilDiv(curKL0, MXFP_DIVISOR_SIZE) * SCALE_BYTES_PER_GROUP, tileL1L0Param.curN);
        auto tensorScaleBL0 = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::l0scaleb, ByteScaleType>(l0Offset / ScaleL0OffsetDivisor<BType>()),
            layoutScaleBL0);
        asc::te::copy(
            CopyL12L0ScaleB, tensorScaleBL0,
            tensorBlockScaleBL1.slice(
                asc::te::make_coord(iter1 * Blaze::Gemm::CeilDiv(baseK_, MXFP_DIVISOR_SIZE) * SCALE_BYTES_PER_GROUP, 0),
                asc::te::make_shape(Blaze::Gemm::CeilDiv(curKL0, MXFP_DIVISOR_SIZE) * SCALE_BYTES_PER_GROUP,
                                    tileL1L0Param.curN)));
        return tensorBL0;
    }

    template <typename TensorA, typename TensorB, typename TensorScaleA, typename TensorScaleB, typename TensorBias,
              typename TensorL0C>
    __aicore__ inline void ProcessL1Iteration(TensorA const& gmA, TensorB const& gmB, TensorScaleA const& gmScaleA,
                                              TensorScaleB const& gmScaleB, TensorBias const& gmBias,
                                              TensorL0C& tensorL0C, TileL1L0Param& tileL1L0Param, uint64_t iter0,
                                              uint64_t scaleKOffsetStride, uint64_t scaleKIter, uint64_t& scaleKIterIdx,
                                              uint64_t splitKIdx)
    {
        uint64_t l1BufId = abL1LoopCnt_ & (l1BufNum_ - 1);
        uint64_t scaleL1BufId = scaleLoopCnt_ & 1;
        const uint64_t kL1Offset = iter0 * kL1_;
        const uint64_t scaleGmOffset = iter0 * scaleKOffsetStride;

        // scaleA, scaleB GM->L1
        auto scalePair = CopyScalesInL1(gmScaleA, gmScaleB, tileL1L0Param, scaleL1BufId, kL1Offset, scaleGmOffset,
                                        scaleKIterIdx == 0);
        auto& tensorScaleAL1 = scalePair.scaleA;
        auto& tensorScaleBL1 = scalePair.scaleB;

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
        tileL1L0Param.curGmKL1 = (iter0 + 1 == kL1Iter_) ? (k_ - kL1Offset) : kL1_;
        tileL1L0Param.curPadKL1 = Blaze::Gemm::CeilAlign(tileL1L0Param.curGmKL1, MXFP_DIVISOR_SIZE); // pad to 64 align

        // A GM->L1
        auto tensorAL1 = CopyAInL1(gmA, tileL1L0Param, l1BufId, iter0);

        // bias GM->L1
        auto tensorBiasL1 = CopyBiasInL1(gmBias, tileL1L0Param, scaleL1BufId, iter0, splitKIdx);

        // B GM->L1; 先slice再copy
        auto tensorBL1 = CopyBInL1(gmB, tileL1L0Param, l1BufId, iter0);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);

        const uint64_t biasBtOffset = baseN_ * scaleL1BufId * sizeof(float);
        Iterate(tileL1L0Param, iter0, biasBtOffset, tensorScaleAL1, tensorScaleBL1, tensorAL1, tensorBL1, tensorBiasL1,
                tensorL0C, splitKIdx);

        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
        if (scaleKIterIdx + 1 == scaleKIter || iter0 + 1 == kL1Iter_) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0 + scaleL1BufId);
            scaleLoopCnt_++;
            scaleKIterIdx = 0;
        } else {
            scaleKIterIdx++;
        }
        abL1LoopCnt_++;
    }

    template <typename TensorC, typename TensorL0C>
    __aicore__ inline void CopyL0CToOut(TensorC const& gmC, TensorL0C& tensorL0C)
    {
        auto copyL0C2GM = asc::te::make_copy(asc::te::copy_l0c_to_gm{});
        FragmentSliceCopy<true>(copyL0C2GM.with(asc::te::l0c_to_gm_params(asc::te::unit_flag_mode::enable_update)),
                                tensorL0C, gmC);
    }

private:
    __aicore__ inline void InitNoneFullLoadL1BufferOffsets(uint64_t baseM, uint64_t l1HalfBufNum, uint64_t bL1OneBuffer,
                                                           uint64_t scaleBL1OneBuffer)
    {
        constexpr uint64_t sizeShift = (IS_HIF4 || IsFp4<AType>()) ? 1 : 0;
        const uint64_t halfL1Size = AscendC::TOTAL_L1_SIZE >> 1;
        uint64_t aL1OneBuffer = (baseM * Blaze::Gemm::CeilAlign(kL1_, MXFP_DIVISOR_SIZE)) >> sizeShift;
        uint64_t scaleAL1OneBuffer = baseM * Blaze::Gemm::CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE) * SCALE_BYTES_PER_GROUP;
        for (int32_t bufferId = 0; bufferId < l1BufNum_; bufferId++) {
            // 2 buffer: L1 space is : A0|B0|AScale0|BScale0|bias0|...|A1|B1|AScale1|BScale1|bias1|...
            // 4 buffer: L1 space is : A0A2|B0B2|AScale0|BScale0|bias0|...|A1A3|B1B3|AScale1|BScale1|bias1|...
            uint64_t l1Offset = halfL1Size * (bufferId & 1);
            l1BufferAOffset_[bufferId] = l1Offset + aL1OneBuffer * (bufferId >> 1);
            l1BufferBOffset_[bufferId] = l1Offset + aL1OneBuffer * l1HalfBufNum + bL1OneBuffer * (bufferId >> 1);
        }
        const uint64_t scaleABaseOffset = bL1OneBuffer * l1HalfBufNum;
        for (int32_t bufferId = 0; bufferId < SCALE_BUFFER_NUM; bufferId++) {
            l1BufferScaleAOffset_[bufferId] = l1BufferBOffset_[bufferId] + scaleABaseOffset;
            l1BufferScaleBOffset_[bufferId] = l1BufferScaleAOffset_[bufferId] + scaleAL1OneBuffer;
            l1BufferBiasOffset_[bufferId] = l1BufferScaleBOffset_[bufferId] + scaleBL1OneBuffer;
        }
    }

    __aicore__ inline void InitAFullLoadL1BufferOffsets(uint64_t baseM, uint64_t l1HalfBufNum, uint64_t bL1OneBuffer,
                                                        uint64_t scaleBL1OneBuffer, uint64_t biasL1OneBuffer)
    {
        constexpr uint64_t sizeShift = (IS_HIF4 || IsFp4<AType>()) ? 1 : 0;
        const uint64_t halfL1Size = AscendC::TOTAL_L1_SIZE >> 1;
        uint64_t mAlign = Blaze::Gemm::CeilAlign(baseM, transA ? C0_SIZE : BLOCK_CUBE);
        uint64_t kAlign = Blaze::Gemm::CeilAlign(k_, MXFP_DIVISOR_SIZE);
        uint64_t aL1OneBuffer = (mAlign * kAlign) >> sizeShift;
        uint64_t scaleAL1OneBuffer = baseM * Blaze::Gemm::CeilDiv(k_, MXFP_DIVISOR_SIZE) * SCALE_BYTES_PER_GROUP;
        // 2 buffer: L1 space is : B0|BScale0|bias0|A|AScale|...|B1|BScale1|bias1|
        // 4 buffer: L1 space is : B0B2|BScale0|bias0|A|AScale|...|B1B3|BScale1|bias1|...
        l1BufferAOffset_[0] = bL1OneBuffer * l1HalfBufNum + scaleBL1OneBuffer + biasL1OneBuffer;
        l1BufferScaleAOffset_[0] = l1BufferAOffset_[0] + aL1OneBuffer;
        uint64_t b1Offset = l1BufferScaleAOffset_[0] + scaleAL1OneBuffer >= halfL1Size ?
                                l1BufferScaleAOffset_[0] + scaleAL1OneBuffer :
                                halfL1Size;
        for (int32_t bufferId = 0; bufferId < l1BufNum_; bufferId++) {
            l1BufferBOffset_[bufferId] = b1Offset * (bufferId & 1) + bL1OneBuffer * (bufferId >> 1);
        }
        const uint64_t scaleBBaseOffset = bL1OneBuffer * l1HalfBufNum;
        for (int32_t bufferId = 0; bufferId < SCALE_BUFFER_NUM; bufferId++) {
            l1BufferScaleBOffset_[bufferId] = l1BufferBOffset_[bufferId] + scaleBBaseOffset;
            l1BufferBiasOffset_[bufferId] = l1BufferScaleBOffset_[bufferId] + scaleBL1OneBuffer;
        }
    }

    __aicore__ inline bool NeedBias(uint64_t kIter0, uint64_t kIter1, uint64_t kIter2)
    {
        return isBias_ && kIter0 == 0 && kIter1 == 0 && kIter2 == 0;
    }

    template <typename TensorL0CType, typename TensorAL0, typename TensorBL0, typename TensorBT>
    __aicore__ inline void Mmad(const TileL1L0Param& tileL1L0Param, uint64_t iter0, uint64_t iter1, uint64_t kL0Iter,
                                uint64_t curKL0, TensorL0CType& tensorL0C, TensorAL0 const& tensorAL0,
                                TensorBL0 const& tensorBL0, TensorBT const& tensorBt, uint64_t splitKIdx)
    {
        asc::te::mmad_params params;
        params.m = static_cast<uint16_t>(tileL1L0Param.curM);
        params.k = static_cast<uint16_t>(Blaze::Gemm::CeilAlign(curKL0, MXFP_DIVISOR_SIZE));
        params.n = static_cast<uint16_t>(tileL1L0Param.curN);
        params.unit_flag = (iter0 + 1 == kL1Iter_ && iter1 + 1 == kL0Iter && (splitKIdx == splitKNum_ - 1)) ?
                               asc::te::unit_flag_mode::enable_update :
                               asc::te::unit_flag_mode::enable_keep;
        params.init_with_zero = (iter0 == 0 && iter1 == 0 && splitKIdx == 0 && !isBias_);
        if (NeedBias(iter0, iter1, splitKIdx)) {
            asc::te::mmad(
                asc::te::mmad_atom<asc::te::mmad_traits<asc::te::mmad_operation, Blaze::Gemm::Tile::MmadTraitMX>>{}
                    .with(params),
                tensorL0C, tensorAL0, tensorBL0, tensorBt);
        } else {
            asc::te::mmad(
                asc::te::mmad_atom<asc::te::mmad_traits<asc::te::mmad_operation, Blaze::Gemm::Tile::MmadTraitMX>>{}
                    .with(params),
                tensorL0C, tensorAL0, tensorBL0);
        }
    }

    uint64_t k_;
    uint64_t l1BufNum_{1};
    uint64_t kL1Iter_{0};
    uint64_t kL1_{1};
    uint64_t scaleKL1_{1};
    uint64_t baseN_{16};
    uint64_t baseK_{16};
    bool isBias_{false};
    uint64_t abL1LoopCnt_{0};
    uint64_t scaleLoopCnt_{0};
    uint64_t l0PingPong_{0};
    uint64_t l0cPingPong_{0};
    uint64_t splitKNum_{1};
    bool enableL0cPingPong_{false};

    uint64_t l1BufferAOffset_[4] = {0UL};      // default 4 buffer
    uint64_t l1BufferBOffset_[4] = {0UL};      // default 4 buffer
    uint64_t l1BufferScaleAOffset_[2] = {0UL}; // default 2 buffer
    uint64_t l1BufferScaleBOffset_[2] = {0UL}; // default 2 buffer
    uint64_t l1BufferBiasOffset_[2] = {0UL};   // default 2 buffer
};
#endif
} // namespace Block
} // namespace Gemm
} // namespace Blaze
