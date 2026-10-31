/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file all_gather_qbmm_mx_kernel.h
 * \brief AllGather Prefill 专用 QMM 内核 — 基于 FragmentTensor
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

#include "kernel_basic_intf.h"
#include "kernel_tiling/kernel_tiling.h"

#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#define WINDOW_LEN 1L // 调度器窗口设置为1，非侵入式修改
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#undef WINDOW_LEN
#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "include/tensor_api/tensor.h"
#include "apace/block/mmad/qmm_mx_block_mmad_fragment.h"
#include "apace/basic/fragment_tensor/fragment_tensor.h"
#include "apace/basic/fragment_tensor/fragment_tensor_api.h"
#include "apace/basic/buffer/buffer_channel.h"
#include "../../../utils/op_state_dump.h"

namespace Apace {

using namespace AscendC;

#define ALL_GATHER_QBMM_MX_KERNEL_CLASS_TEM_PARAMS \
    template <class BlockMmadFrag, class BlockScheduler, class BufferSyncManager, bool enableBufferReuse>

#define ALL_GATHER_QBMM_MX_KERNEL_FUNC_TEM_PARAMS BlockMmadFrag, BlockScheduler, BufferSyncManager, enableBufferReuse

enum RegionTag {
    HEAD,
    MAIN,
    TAIL
};

/**
 * @brief AllGather QMM 内核
 * 通信同步经 BufferSyncManager 策略类注入（组合模式）：基类持 bufferSyncMgr_ 成员对象，
 * 调用点直接 bufferSyncMgr_.Acquire(flagId)，编译期由模板参数绑定具体策略。
 * enableBufferReuse=true（urma）：win 区 remote 紧凑 + data 槽回绕复用，A/scaleA 连续读，
 *   消费后 Release 通知写端可覆盖；false（hcomm）：保持 rank-major 全量 frag 读。
 */
template <class BlockMmadFrag, class BlockScheduler, class BufferSyncManager, bool enableBufferReuse>
class AllGatherQbmmMxKernel {
public:
    __aicore__ inline AllGatherQbmmMxKernel() {}
    __aicore__ inline ~AllGatherQbmmMxKernel() {}

    // ---- Component types ----
    using FragL1Params = typename BlockMmadFrag::L1Params;
    using BlockMmadParams = typename BlockMmadFrag::Params;
    using AType = typename BlockMmadFrag::AType;
    using BType = typename BlockMmadFrag::BType;
    using CType = typename BlockMmadFrag::CType;
    using BiasType = typename BlockMmadFrag::BiasType;
    using BlockSchedulerParams = typename BlockScheduler::Params;
    using BlockShape = typename BlockMmadFrag::BlockShape;
    using BlockCoord = asc::te::coord<int64_t, int64_t, int64_t, int64_t>;
    using LayoutA = typename BlockMmadFrag::LayoutA;
    using LayoutB = typename BlockMmadFrag::LayoutB;
    using LayoutC = typename BlockMmadFrag::LayoutC;
    using ProblemShape = typename BlockMmadFrag::ProblemShape;
    static constexpr bool transB = BlockMmadFrag::transB;

#if APACE_HIF4_TYPE_SUPPORTED
    static constexpr bool IS_HIF4 = AscendC::IsSameType<AType, hifloat4x2_t>::value;
    using ScaleType = AscendC::Std::conditional_t<IS_HIF4, AscendC::hif4_scale, AscendC::fp8_e8m0_t>;
#else
    static constexpr bool IS_HIF4 = false;
    using ScaleType = AscendC::fp8_e8m0_t;
#endif

    // ---- Compile-time constants ----
    static constexpr bool weightNz = BlockMmadFrag::weightNz;
    static constexpr int64_t C0_SIZE =
        (IS_HIF4 || Blaze::Gemm::IsFp4<AType>()) ? Blaze::Gemm::C0_SIZE_B4 : Blaze::Gemm::C0_SIZE_B8;
    static constexpr int64_t kCacheLineAlignMask = (IS_HIF4 || Blaze::Gemm::IsFp4<AType>()) ? 0xffUL : 0x7fUL;
    static constexpr int32_t kScaleC0 = IS_HIF4 ? 1 : 2;
    static constexpr uint32_t DIMS_NUM = 2UL;

    // ---- Layout factories ----
    using MakeLayoutA = asc::te::frame_layout_format<LayoutA, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutScaleA = asc::te::frame_layout_format<asc::te::scalea_nd_layout_ptn, AscendC::Std::Int<kScaleC0>>;
    using MakeLayoutB = asc::te::frame_layout_format<LayoutB, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutScaleB = AscendC::Std::conditional_t<
        transB, asc::te::frame_layout_format<asc::te::scaleb_dn_layout_ptn, AscendC::Std::Int<kScaleC0>>,
        asc::te::frame_layout_format<asc::te::scaleb_nd_layout_ptn, AscendC::Std::Int<kScaleC0>>>;
    using MakeLayoutC = asc::te::frame_layout_format<LayoutC, AscendC::Std::Int<asc::te::c0_element<CType>>>;

    // ---- FragmentTensor types ----
    using FragTensorA = Apace::Basic::FragmentTensor<DIMS_NUM, Apace::Basic::MAX_FRAGMENT_COUNT, MakeLayoutA, AType>;
    using FragScaleA =
        Apace::Basic::FragmentTensor<DIMS_NUM, Apace::Basic::MAX_FRAGMENT_COUNT, MakeLayoutScaleA, ScaleType>;
    using FragTensorC = Apace::Basic::FragmentTensor<DIMS_NUM, Apace::Basic::MAX_FRAGMENT_COUNT, MakeLayoutC, CType>;

    /**
     * @brief Tiling 配置（对标 blaze QBMMTiling）
     */
    struct QBMMTiling {
        uint32_t baseM{0};
        uint32_t baseN{0};
        uint32_t baseK{0};
        uint32_t dbL0C{0};
    };

    /**
     * @brief Fragment 相关参数
     */
    struct FragmentParams {
        uint32_t tileCnt{0};
        uint32_t tileM{0};
        uint32_t tailCnt{0};
        uint32_t tailM{0};
        uint32_t paddedTailM{0};
        uint32_t commTurn{0};
        uint64_t headRows{0};
        uint32_t rankId{0};
        uint32_t rankSize{0};
        uint32_t mPerRank{0}; // M per rank
        uint64_t k{0};
        uint64_t n{0};
        uint64_t scaleKLen{0};
    };

    /**
     * @brief 顶层参数结构
     */
    struct Params {
        FragmentParams fragParams;
        BlockMmadParams mmadParams;
        BlockSchedulerParams schParams;
        FragL1Params l1Params;
        QBMMTiling qbmmParams;

        // Win buffer base addresses (computed by communication layer)
        GM_ADDR winDataBase{nullptr};
        GM_ADDR winScaleBase{nullptr};

        // Computed dimensions
        uint64_t dataBytesPerMRow{0};
        uint64_t scaleBytesPerMRow{0};
        uint64_t cBytesPerM{0};
    };

    __aicore__ inline void Run(const Params& params, Mc2Kernel::OpStateDump& opStateDump);
    __aicore__ inline void operator()(const Params& params, Mc2Kernel::OpStateDump& opStateDump)
    {
        Run(params, opStateDump);
    }
    __aicore__ inline BufferSyncManager& GetBufferSyncMgr()
    {
        return bufferSyncMgr_;
    }

    // channel 由 fusions 层统一构造并注入（对齐 a2amm 模板）：slotNum/slotSize 以注入的 channel 为单一来源
    __aicore__ inline void SetChannels(Apace::Basic::BufferChannel* dataChannel,
                                       Apace::Basic::BufferChannel* scaleChannel)
    {
        dataChannel_ = dataChannel;
        scaleChannel_ = scaleChannel;
    }

private:
    __aicore__ inline void Init(const Params& params);
    __aicore__ inline void Process(const Params& params, const ProblemShape& problemShape, BlockScheduler& bs,
                                   BlockMmadFrag& mmadFrag, Mc2Kernel::OpStateDump& opStateDump);

    // 读端仅经 GetNextSlot 推进游标，data 槽回绕复用、scale 槽恒全量。
    // 循环内：等待 tile 依赖的 round；reuse 按 0-based round（dependTileIdx-1），更新 slot 偏移。
    __aicore__ inline void WaitDependTiles(int32_t dependTileIdx, int32_t commTurnI, int32_t& readyTileIdx,
                                           uint64_t& dataSlotOffset, uint64_t& scaleSlotOffset,
                                           Mc2Kernel::OpStateDump& opStateDump);
    // 收尾兜底：排空剩余 round 的 flag；不写偏移、不推进 scale 游标。
    __aicore__ inline void DrainWaitTiles(int32_t commTurnI, int32_t& readyTileIdx,
                                          Mc2Kernel::OpStateDump& opStateDump);

    // ---- L2 cache optimization ----
    template <typename TensorB, typename TensorScaleB>
    __aicore__ inline void SetL2Cache(const ProblemShape& problemShape, int64_t baseM, int64_t baseN, TensorB& gmB,
                                      TensorScaleB& gmScaleB);

    // ---- FragmentTensor builders ----
    __aicore__ inline void BuildFragmentTensors(const Params& params);
    __aicore__ inline void EnsureWinRankBasesReady(const Params& params);
    __aicore__ inline void BuildMainFragment(const Params& params, uint32_t roundIdx);
    __aicore__ inline void BuildTailFragment(const Params& params);
    __aicore__ inline void UpdateMainRoundAddrs(const Params& params, uint32_t roundIdx);
    __aicore__ inline Apace::Basic::FragmentParam<DIMS_NUM> MakeFragParam(uint64_t fragSize, uint64_t realFragSize,
                                                                          uint32_t fragCnt, uint64_t shape1) const;

    // ---- Tile context resolution ----
    struct TileCtx {
        uint32_t dependTileIdx;
        uint32_t roundIdx;
        RegionTag region;
        int64_t regionMPos;
        const FragTensorA* fragA;
        const FragScaleA* fragScaleA;
        const FragTensorC* fragC;
    };
    __aicore__ inline TileCtx ResolveTileCtx(int64_t mPos, int64_t headMainRows, int64_t mainRoundRows,
                                             int64_t mainSectionRows, uint32_t commTurn) const;

    // ---- Per-tile 计算 ----
    template <typename GmTensorA, typename GmTensorScaleA, typename GmTensorB, typename GmTensorScaleB,
              typename GmTensorBias>
    __aicore__ inline void ProcessOneBlock(BlockMmadFrag& mmadFrag, const GmTensorA& gmA,
                                           const GmTensorScaleA& gmScaleA, const GmTensorB& gmB,
                                           const GmTensorScaleB& gmScaleB, const GmTensorBias& gmBias,
                                           const TileCtx& ctx, const BlockShape& singleShape, int64_t nPos,
                                           int64_t baseM, int64_t baseN, int64_t K, int64_t scaleKLen);

    // ---- GM addresses ----
    __gm__ AType* aGmAddr_{nullptr};
    __gm__ ScaleType* scaleAGmAddr_{nullptr};
    __gm__ BType* bGmAddr_{nullptr};
    __gm__ ScaleType* scaleBGmAddr_{nullptr};
    __gm__ CType* cGmAddr_{nullptr};
    __gm__ float* biasGmAddr_{nullptr};

    // ---- FragmentTensor state ----
    static constexpr uint32_t MAX_FRAG = Apace::Basic::MAX_FRAGMENT_COUNT;
    struct FragStreams {
        FragTensorA a{};
        FragScaleA scaleA{};
        FragTensorC c{};
        GM_ADDR addrListA[MAX_FRAG]{};
        GM_ADDR addrListScaleA[MAX_FRAG]{};
        GM_ADDR addrListC[MAX_FRAG]{};
    };
    FragStreams headFrag_{};
    FragStreams curMainFrag_{};
    FragStreams tailFrag_{};

    static constexpr uint32_t MAX_RANK_SIZE = 16U; // 与 tiling rank_size 上限一致
    GM_ADDR cFragAddrs_[MAX_RANK_SIZE]{};
    GM_ADDR winDataRankBase_[MAX_RANK_SIZE]{};
    GM_ADDR winScaleRankBase_[MAX_RANK_SIZE]{};

    BufferSyncManager bufferSyncMgr_;
    Apace::Basic::BufferChannel* dataChannel_{nullptr};
    Apace::Basic::BufferChannel* scaleChannel_{nullptr};
    uint32_t curMainRoundIdx_{0xFFFFFFFF};
    bool winBasesReady_{false};
    bool tailBuilt_{false};
    uint64_t curRoundDataOff_{};
    uint64_t curRoundScaleOff_{};
    uint64_t curRoundCOff_{};

    // ---- Tile strides ----
    uint64_t tileMDataStride_{};
    uint64_t tileMScaleStride_{};
    uint64_t tileMCStride_{};
};

ALL_GATHER_QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void AllGatherQbmmMxKernel<ALL_GATHER_QBMM_MX_KERNEL_FUNC_TEM_PARAMS>::Init(const Params& params)
{
    if ASCEND_IS_AIV {
        return;
    }
    aGmAddr_ = reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr);
    bGmAddr_ = reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr);
    cGmAddr_ = reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr);
    scaleAGmAddr_ = reinterpret_cast<__gm__ ScaleType*>(params.mmadParams.scaleAGmAddr);
    scaleBGmAddr_ = reinterpret_cast<__gm__ ScaleType*>(params.mmadParams.scaleBGmAddr);
    biasGmAddr_ = reinterpret_cast<__gm__ float*>(params.mmadParams.biasGmAddr);
    tileMDataStride_ = static_cast<uint64_t>(params.fragParams.tileM) * params.dataBytesPerMRow;
    tileMScaleStride_ = static_cast<uint64_t>(params.fragParams.tileM) * params.scaleBytesPerMRow;
    tileMCStride_ = static_cast<uint64_t>(params.fragParams.tileM) * params.cBytesPerM;
}

ALL_GATHER_QBMM_MX_KERNEL_CLASS_TEM_PARAMS
template <typename TensorB, typename TensorScaleB>
__aicore__ inline void AllGatherQbmmMxKernel<ALL_GATHER_QBMM_MX_KERNEL_FUNC_TEM_PARAMS>::SetL2Cache(
    const ProblemShape& problemShape, int64_t baseM, int64_t baseN, TensorB& gmB, TensorScaleB& gmScaleB)
{
    const bool fullMBlock = (baseM >= asc::te::get<Blaze::Gemm::MNK_M>(problemShape));

    // B L2 hint: 对齐 128B 时关闭 L2 cache 以 streaming
    if constexpr (weightNz) {
        gmB.set_l2_cache_hint(fullMBlock ? asc::te::cache_mode::disable : asc::te::cache_mode::normal);
    } else {
        if constexpr (transB) {
            // DN: fast dim = K
            const bool bAlignForL2Stream = (asc::te::get<Blaze::Gemm::MNK_K>(problemShape) & kCacheLineAlignMask) == 0;
            gmB.set_l2_cache_hint((fullMBlock && bAlignForL2Stream) ? asc::te::cache_mode::disable :
                                                                      asc::te::cache_mode::normal);
        } else {
            // ND: fast dim = N
            const bool bAlignForL2Stream =
                (asc::te::get<Blaze::Gemm::MNK_N>(problemShape) & kCacheLineAlignMask) == 0 &&
                (baseN & kCacheLineAlignMask) == 0;
            gmB.set_l2_cache_hint((fullMBlock && bAlignForL2Stream) ? asc::te::cache_mode::disable :
                                                                      asc::te::cache_mode::normal);
        }
    }

    // ScaleB L2 hint
    if constexpr (transB) {
        // ScaleBDN: 连续段为 scaleK 方向
        const int64_t scaleKRowBytes = Blaze::Gemm::CeilDiv(asc::te::get<Blaze::Gemm::MNK_K>(problemShape),
                                                            static_cast<int64_t>(Blaze::Gemm::MXFP_DIVISOR_SIZE)) *
                                       kScaleC0 * sizeof(ScaleType);
        const bool scaleAlignForL2Stream = (scaleKRowBytes & kCacheLineAlignMask) == 0;
        gmScaleB.set_l2_cache_hint((fullMBlock && scaleAlignForL2Stream) ? asc::te::cache_mode::disable :
                                                                           asc::te::cache_mode::normal);
    } else {
        // ScaleBND: 连续段为 N 方向
        const int64_t scaleNStrideBytes = asc::te::get<Blaze::Gemm::MNK_N>(problemShape) * kScaleC0 * sizeof(ScaleType);
        const int64_t scaleBaseNStrideBytes = baseN * kScaleC0 * sizeof(ScaleType);
        const bool scaleAlignForL2Stream =
            (scaleNStrideBytes & kCacheLineAlignMask) == 0 && (scaleBaseNStrideBytes & kCacheLineAlignMask) == 0;
        gmScaleB.set_l2_cache_hint((fullMBlock && scaleAlignForL2Stream) ? asc::te::cache_mode::disable :
                                                                           asc::te::cache_mode::normal);
    }
}

ALL_GATHER_QBMM_MX_KERNEL_CLASS_TEM_PARAMS
template <typename GmTensorA, typename GmTensorScaleA, typename GmTensorB, typename GmTensorScaleB,
          typename GmTensorBias>
__aicore__ inline void AllGatherQbmmMxKernel<ALL_GATHER_QBMM_MX_KERNEL_FUNC_TEM_PARAMS>::ProcessOneBlock(
    BlockMmadFrag& mmadFrag, const GmTensorA& gmA, const GmTensorScaleA& gmScaleA, const GmTensorB& gmB,
    const GmTensorScaleB& gmScaleB, const GmTensorBias& gmBias, const TileCtx& ctx, const BlockShape& singleShape,
    int64_t nPos, int64_t baseM, int64_t baseN, int64_t K, int64_t scaleKLen)
{
    auto gmBlockB = gmB.slice(asc::te::make_coord(0L, nPos), asc::te::make_shape(K, baseN));
    auto gmBlockScaleB = gmScaleB.slice(asc::te::make_coord(0L, nPos), asc::te::make_shape(scaleKLen, baseN));
    auto gmBlockBias = gmBias.slice(asc::te::make_coord(0L, nPos), asc::te::make_shape(1L, baseN));
    auto blockA = gmA.slice(asc::te::make_coord(ctx.regionMPos, 0L), asc::te::make_shape(baseM, K));
    auto blockScaleA = gmScaleA.slice(asc::te::make_coord(ctx.regionMPos, 0L), asc::te::make_shape(baseM, scaleKLen));
    auto blockC = ctx.fragC->slice(asc::te::make_coord(ctx.regionMPos, nPos), asc::te::make_shape(baseM, baseN));
    mmadFrag(blockA, gmBlockB, blockScaleA, gmBlockScaleB, gmBlockBias, blockC, singleShape);
}

ALL_GATHER_QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void AllGatherQbmmMxKernel<ALL_GATHER_QBMM_MX_KERNEL_FUNC_TEM_PARAMS>::Run(
    const Params& params, Mc2Kernel::OpStateDump& opStateDump)
{
    Init(params);

    const auto& fragParams = params.fragParams;
    const auto& qbmmParams = params.qbmmParams;
    const int64_t K = static_cast<int64_t>(fragParams.k);
    const int64_t N = static_cast<int64_t>(fragParams.n);

    // 总 logical M = head + main 段 + tail。
    bool hasTail = (fragParams.tailCnt > 0 && fragParams.tailM > 0);
    const int64_t headMainRows = static_cast<int64_t>(fragParams.headRows);
    const int64_t mainRoundRows =
        static_cast<int64_t>(fragParams.rankSize - 1) * static_cast<int64_t>(fragParams.tileM);
    const int64_t mainSectionRows = static_cast<int64_t>(fragParams.tileCnt) * mainRoundRows;
    const int64_t tailRows =
        hasTail ? static_cast<int64_t>(fragParams.rankSize) * static_cast<int64_t>(fragParams.paddedTailM) : 0;
    const int64_t totalLogicalM = headMainRows + mainSectionRows + tailRows;

    ProblemShape problemShape{totalLogicalM, N, K, 1};
    BlockScheduler sch(problemShape, params.schParams);
    const BlockShape l0TileShape{static_cast<int64_t>(qbmmParams.baseM), static_cast<int64_t>(qbmmParams.baseN),
                                 static_cast<int64_t>(qbmmParams.baseK), 0L};
    BlockMmadFrag mmadFrag;
    bool isBias = biasGmAddr_ != nullptr;
    mmadFrag.Init(problemShape, l0TileShape, params.l1Params, isBias, qbmmParams.dbL0C > 1);

    // 延迟构建 fragment tensor（head 先建，main/tail 在 Process 中按需建）
    BuildFragmentTensors(params);

    Process(params, problemShape, sch, mmadFrag, opStateDump);
}

ALL_GATHER_QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void AllGatherQbmmMxKernel<ALL_GATHER_QBMM_MX_KERNEL_FUNC_TEM_PARAMS>::Process(
    const Params& params, const ProblemShape& problemShape, BlockScheduler& sch, BlockMmadFrag& mmadFrag,
    Mc2Kernel::OpStateDump& opStateDump)
{
    const auto& fragParams = params.fragParams;
    const int64_t K = static_cast<int64_t>(fragParams.k);
    const int64_t N = static_cast<int64_t>(fragParams.n);
    const int64_t scaleKLen = static_cast<int64_t>(fragParams.scaleKLen);

    const int64_t headMainRows = static_cast<int64_t>(fragParams.headRows);
    const int64_t mainRoundRows =
        static_cast<int64_t>(fragParams.rankSize - 1) * static_cast<int64_t>(fragParams.tileM);
    const int64_t mainSectionRows = static_cast<int64_t>(fragParams.tileCnt) * mainRoundRows;

    // B / scaleB / bias 全局共享 tensor
    auto gmB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(bGmAddr_), MakeLayoutB{}(K, N));
    auto gmScaleB = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(scaleBGmAddr_),
                                         MakeLayoutScaleB{}(scaleKLen, N));
    auto gmBias = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(biasGmAddr_),
                                       asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn>(1L, N));

    const auto& mTailTile = params.schParams.mTailTile;
    const auto& nTailTile = params.schParams.nTailTile;
    // 尾块拆分：需核数(尾块数×M拆分×N拆分)≤总核数才拆分；mTailTile=nTailTile=1时等价no-op。
    if ((sch.GetEndBlockIdx() + 1) * mTailTile * nTailTile <= AscendC::GetBlockNum()) {
        sch.UpdateTailTile(mTailTile, nTailTile);
    }

    // 预触发 HEAD（flagId=0）：HEAD 读本地 GM 不经通信；hcomm 策略下为 no-op。
    bufferSyncMgr_.Acquire(0);
    // dependTileIdx=0 为本 rank 数据，无需等待通信。
    opStateDump.DoDump(DUMP_FIELD_WAIT);

    const int32_t commTurnI = static_cast<int32_t>(fragParams.commTurn);
    // 复用路径：0-based round 序号（初始 -1 表示尚未 Acquire 任何 round）；
    // 非复用路径：保持 1-based dependTileIdx 语义（与 hcomm 策略的 handle 等待兼容）。
    int32_t readyTileIdx = enableBufferReuse ? -1 : 0;
    uint64_t dataSlotOffset = 0;
    uint64_t scaleSlotOffset = 0;

    BlockCoord blockIdx;
    int64_t mPos = 0L, nPos = 0L;

    while (sch.GetTileIdx(blockIdx)) {
        auto singleShape = sch.template GetBlockShape<Blaze::Gemm::QuantMode::MX_PERGROUP_MODE,
                                                      Blaze::Gemm::QuantMode::MX_PERGROUP_MODE, weightNz>(blockIdx);
        int64_t baseM = asc::te::get<0>(singleShape);
        int64_t baseN = asc::te::get<1>(singleShape);
        if (baseM <= 0 || baseN <= 0) {
            break;
        }
        opStateDump.DoDump(DUMP_FIELD_TURN_INC);
        sch.GetTileCoord(blockIdx, mPos, nPos);

        auto ctx = ResolveTileCtx(mPos, headMainRows, mainRoundRows, mainSectionRows, fragParams.commTurn);

        WaitDependTiles(static_cast<int32_t>(ctx.dependTileIdx), commTurnI, readyTileIdx, dataSlotOffset,
                        scaleSlotOffset, opStateDump);

        // Per-tile L2 cache hint
        SetL2Cache(problemShape, baseM, baseN, gmB, gmScaleB);

        // MAIN/TAIL 的 fragment tensor 延迟到首次命中时构建/更新地址。
        if (ctx.region == MAIN) {
            if (curMainRoundIdx_ == 0xFFFFFFFF) {
                BuildMainFragment(params, ctx.roundIdx);
            } else if (curMainRoundIdx_ != ctx.roundIdx) {
                UpdateMainRoundAddrs(params, ctx.roundIdx);
            }
        } else if (ctx.region == TAIL && !tailBuilt_) {
            BuildTailFragment(params);
            tailBuilt_ = true;
        }

        if constexpr (enableBufferReuse) {
            if (ctx.region == MAIN) {
                // MAIN 连续读：当前 slot 内 (rankSize-1) 个 remote 段按 remoteIdx 紧凑排布，
                // regionMPos 即 slot 内行偏移；data 槽回绕复用，scale 槽全量不回绕。
                auto gmA =
                    asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ AType*>(
                                             reinterpret_cast<__gm__ char*>(params.winDataBase) + dataSlotOffset)),
                                         MakeLayoutA{}(mainRoundRows, K));
                auto gmScaleA = asc::te::make_tensor(
                    asc::te::make_mem_ptr<asc::te::location::gm>(reinterpret_cast<__gm__ ScaleType*>(
                        reinterpret_cast<__gm__ char*>(params.winScaleBase) + scaleSlotOffset)),
                    MakeLayoutScaleA{}(mainRoundRows, scaleKLen));
                ProcessOneBlock(mmadFrag, gmA, gmScaleA, gmB, gmScaleB, gmBias, ctx, singleShape, nPos, baseM, baseN, K,
                                scaleKLen);
                continue;
            }
        }
        ProcessOneBlock(mmadFrag, *ctx.fragA, *ctx.fragScaleA, gmB, gmScaleB, gmBias, ctx, singleShape, nPos, baseM,
                        baseN, K, scaleKLen);
    }

    // 收尾兜底：确保所有 round 均已 Acquire；复用路径还需 Release（尾核不释放会导致写端覆盖前死等）。
    DrainWaitTiles(commTurnI, readyTileIdx, opStateDump);
}

ALL_GATHER_QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void AllGatherQbmmMxKernel<ALL_GATHER_QBMM_MX_KERNEL_FUNC_TEM_PARAMS>::WaitDependTiles(
    int32_t dependTileIdx, int32_t commTurnI, int32_t& readyTileIdx, uint64_t& dataSlotOffset,
    uint64_t& scaleSlotOffset, Mc2Kernel::OpStateDump& opStateDump)
{
    if constexpr (enableBufferReuse) {
        // wait + release merged：release 已消费 round，写端 round>=dataSlotNum 覆盖同槽前等待该 flag；
        // 最后 dataSlotNum 个 round 的槽位不会再被覆盖，无需 release。
        // reuse 内部按 0-based round 等待（dependTileIdx-1）；dataSlotNum 取自 dataChannel_（reuse 下必已注入）。
        const int32_t dataSlotNumI = static_cast<int32_t>(dataChannel_->GetSlotNum());
        while (readyTileIdx < dependTileIdx - 1) {
            if (readyTileIdx >= 0 && readyTileIdx < commTurnI - dataSlotNumI) {
                bufferSyncMgr_.Release(dataChannel_->GetRoundSlotIdx(static_cast<uint32_t>(readyTileIdx)) + 1);
            }
            readyTileIdx++;
            auto dataSlot = dataChannel_->GetNextSlot();
            bufferSyncMgr_.Acquire(dataSlot.slotIdx + 1); // +1 避开 HEAD 预触发的 flagId=0
            dataSlotOffset = dataSlot.offset;
            scaleSlotOffset = scaleChannel_->GetNextSlot().offset;
            opStateDump.DoDump(DUMP_FIELD_WAIT);
        }
    } else {
        // 非复用路径：保持 1-based dependTileIdx 语义（与 hcomm 策略的 handle 等待兼容）。
        while (readyTileIdx < dependTileIdx) {
            readyTileIdx++;
            bufferSyncMgr_.Acquire(static_cast<uint32_t>(readyTileIdx));
            opStateDump.DoDump(DUMP_FIELD_WAIT);
        }
    }
}

ALL_GATHER_QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void AllGatherQbmmMxKernel<ALL_GATHER_QBMM_MX_KERNEL_FUNC_TEM_PARAMS>::DrainWaitTiles(
    int32_t commTurnI, int32_t& readyTileIdx, Mc2Kernel::OpStateDump& opStateDump)
{
    if constexpr (enableBufferReuse) {
        const int32_t dataSlotNumI = static_cast<int32_t>(dataChannel_->GetSlotNum());
        while (readyTileIdx < commTurnI - 1) {
            if (readyTileIdx >= 0 && readyTileIdx < commTurnI - dataSlotNumI) {
                bufferSyncMgr_.Release(dataChannel_->GetRoundSlotIdx(static_cast<uint32_t>(readyTileIdx)) + 1);
            }
            readyTileIdx++;
            auto dataSlot = dataChannel_->GetNextSlot();
            bufferSyncMgr_.Acquire(dataSlot.slotIdx + 1);
            opStateDump.DoDump(DUMP_FIELD_WAIT);
        }
    } else {
        while (readyTileIdx < commTurnI) {
            readyTileIdx++;
            bufferSyncMgr_.Acquire(static_cast<uint32_t>(readyTileIdx));
            opStateDump.DoDump(DUMP_FIELD_WAIT);
        }
    }
}

ALL_GATHER_QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline Apace::Basic::FragmentParam<
    AllGatherQbmmMxKernel<ALL_GATHER_QBMM_MX_KERNEL_FUNC_TEM_PARAMS>::DIMS_NUM>
AllGatherQbmmMxKernel<ALL_GATHER_QBMM_MX_KERNEL_FUNC_TEM_PARAMS>::MakeFragParam(uint64_t fragSize,
                                                                                uint64_t realFragSize, uint32_t fragCnt,
                                                                                uint64_t shape1) const
{
    Apace::Basic::FragmentParam<DIMS_NUM> param{};
    param.assembleAxis = 0;
    param.assembledShape[0] = fragSize * fragCnt;
    param.assembledShape[1] = shape1;
    param.fragmentSize = fragSize;
    param.realFragmentSize = realFragSize;
    param.fragmentCnt = fragCnt;
    return param;
}

ALL_GATHER_QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void AllGatherQbmmMxKernel<ALL_GATHER_QBMM_MX_KERNEL_FUNC_TEM_PARAMS>::BuildFragmentTensors(
    const Params& params)
{
    const auto& fragParams = params.fragParams;

    for (uint32_t r = 0; r < fragParams.rankSize; ++r) {
        cFragAddrs_[r] = params.mmadParams.cGmAddr + static_cast<uint64_t>(r) * fragParams.mPerRank * params.cBytesPerM;
    }

    // HEAD：自身数据，单 fragment tensor。
    headFrag_.addrListA[0] = params.mmadParams.aGmAddr;
    headFrag_.addrListScaleA[0] = params.mmadParams.scaleAGmAddr;
    headFrag_.addrListC[0] = cFragAddrs_[fragParams.rankId];
    headFrag_.a = Apace::Basic::MakeFragmentTensor<DIMS_NUM, MAX_FRAG, MakeLayoutA, AType>(
        MakeFragParam(fragParams.headRows, fragParams.headRows, 1, fragParams.k), headFrag_.addrListA);
    headFrag_.scaleA = Apace::Basic::MakeFragmentTensor<DIMS_NUM, MAX_FRAG, MakeLayoutScaleA, ScaleType>(
        MakeFragParam(fragParams.headRows, fragParams.headRows, 1, fragParams.scaleKLen), headFrag_.addrListScaleA);
    headFrag_.c = Apace::Basic::MakeFragmentTensor<DIMS_NUM, MAX_FRAG, MakeLayoutC, CType>(
        MakeFragParam(fragParams.headRows, fragParams.headRows, 1, fragParams.n), headFrag_.addrListC);
    // main / tail 的 fragment tensor 延迟到首次使用时构建，构建开销被 comm wait 掩盖。
}

ALL_GATHER_QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void AllGatherQbmmMxKernel<ALL_GATHER_QBMM_MX_KERNEL_FUNC_TEM_PARAMS>::EnsureWinRankBasesReady(
    const Params& params)
{
    if (winBasesReady_) {
        return;
    }
    const auto& fragParams = params.fragParams;
    for (uint32_t r = 0; r < fragParams.rankSize; ++r) {
        winDataRankBase_[r] =
            params.winDataBase + static_cast<uint64_t>(r) * fragParams.mPerRank * params.dataBytesPerMRow;
        winScaleRankBase_[r] =
            params.winScaleBase + static_cast<uint64_t>(r) * fragParams.mPerRank * params.scaleBytesPerMRow;
    }
    winBasesReady_ = true;
}

ALL_GATHER_QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void AllGatherQbmmMxKernel<ALL_GATHER_QBMM_MX_KERNEL_FUNC_TEM_PARAMS>::BuildMainFragment(
    const Params& params, uint32_t roundIdx)
{
    if constexpr (!enableBufferReuse) {
        EnsureWinRankBasesReady(params);
    }
    const auto& fragParams = params.fragParams;

    if constexpr (!enableBufferReuse) {
        curRoundDataOff_ = static_cast<uint64_t>(roundIdx) * tileMDataStride_;
        curRoundScaleOff_ = static_cast<uint64_t>(roundIdx) * tileMScaleStride_;
    }
    curRoundCOff_ = static_cast<uint64_t>(roundIdx) * tileMCStride_;

    uint32_t fragIdx = 0;
    for (uint32_t r = 0; r < fragParams.rankSize; ++r) {
        if (r == fragParams.rankId) {
            continue;
        }
        if constexpr (!enableBufferReuse) {
            curMainFrag_.addrListA[fragIdx] = winDataRankBase_[r] + curRoundDataOff_;
            curMainFrag_.addrListScaleA[fragIdx] = winScaleRankBase_[r] + curRoundScaleOff_;
        }
        curMainFrag_.addrListC[fragIdx] = cFragAddrs_[r] + curRoundCOff_;
        fragIdx++;
    }
    if constexpr (!enableBufferReuse) {
        curMainFrag_.a = Apace::Basic::MakeFragmentTensor<DIMS_NUM, MAX_FRAG, MakeLayoutA, AType>(
            MakeFragParam(fragParams.tileM, fragParams.tileM, fragParams.rankSize - 1, fragParams.k),
            curMainFrag_.addrListA);
        curMainFrag_.scaleA = Apace::Basic::MakeFragmentTensor<DIMS_NUM, MAX_FRAG, MakeLayoutScaleA, ScaleType>(
            MakeFragParam(fragParams.tileM, fragParams.tileM, fragParams.rankSize - 1, fragParams.scaleKLen),
            curMainFrag_.addrListScaleA);
    }
    curMainFrag_.c = Apace::Basic::MakeFragmentTensor<DIMS_NUM, MAX_FRAG, MakeLayoutC, CType>(
        MakeFragParam(fragParams.tileM, fragParams.tileM, fragParams.rankSize - 1, fragParams.n),
        curMainFrag_.addrListC);
    curMainRoundIdx_ = roundIdx;
}

ALL_GATHER_QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void AllGatherQbmmMxKernel<ALL_GATHER_QBMM_MX_KERNEL_FUNC_TEM_PARAMS>::BuildTailFragment(
    const Params& params)
{
    if constexpr (!enableBufferReuse) {
        EnsureWinRankBasesReady(params);
    }
    const auto& fragParams = params.fragParams;

    uint64_t tailRowOff = fragParams.headRows;
    uint64_t tailDataOff = tailRowOff * params.dataBytesPerMRow;
    uint64_t tailScaleOff = tailRowOff * params.scaleBytesPerMRow;
    uint64_t tailCOff = tailRowOff * params.cBytesPerM;

    // TAIL 为最后一个 round（序号 = tileCnt），不走 channel 游标，slot 偏移手工计算，
    // 须与读端 channel 在该 round 的状态一致：data 用 tileCnt % dataChannel_->GetSlotNum() 回绕，scale 全量不回绕。
    uint64_t dataSlotOff = 0;
    uint64_t scaleSlotOff = 0;
    if constexpr (enableBufferReuse) {
        dataSlotOff = (static_cast<uint64_t>(fragParams.tileCnt) % dataChannel_->GetSlotNum()) *
                      (static_cast<uint64_t>(fragParams.rankSize - 1) * tileMDataStride_);
        scaleSlotOff = static_cast<uint64_t>(fragParams.tileCnt) *
                       (static_cast<uint64_t>(fragParams.rankSize - 1) * tileMScaleStride_);
    }

    for (uint32_t r = 0; r < fragParams.rankSize; ++r) {
        if (r == fragParams.rankId) {
            tailFrag_.addrListA[r] = params.mmadParams.aGmAddr + tailDataOff;
            tailFrag_.addrListScaleA[r] = params.mmadParams.scaleAGmAddr + tailScaleOff;
        } else if constexpr (enableBufferReuse) {
            // remote 紧凑：remoteIdx 为跳过本卡后的升序序号，与写端 PUT 的槽内布局一致
            uint32_t remoteIdx = (r > fragParams.rankId) ? (r - 1) : r;
            tailFrag_.addrListA[r] =
                params.winDataBase + dataSlotOff + static_cast<uint64_t>(remoteIdx) * tileMDataStride_;
            tailFrag_.addrListScaleA[r] =
                params.winScaleBase + scaleSlotOff + static_cast<uint64_t>(remoteIdx) * tileMScaleStride_;
        } else {
            tailFrag_.addrListA[r] = winDataRankBase_[r] + tailDataOff;
            tailFrag_.addrListScaleA[r] = winScaleRankBase_[r] + tailScaleOff;
        }
        tailFrag_.addrListC[r] = cFragAddrs_[r] + tailCOff;
    }
    tailFrag_.a = Apace::Basic::MakeFragmentTensor<DIMS_NUM, MAX_FRAG, MakeLayoutA, AType>(
        MakeFragParam(fragParams.paddedTailM, fragParams.tailM, fragParams.rankSize, fragParams.k),
        tailFrag_.addrListA);
    tailFrag_.scaleA = Apace::Basic::MakeFragmentTensor<DIMS_NUM, MAX_FRAG, MakeLayoutScaleA, ScaleType>(
        MakeFragParam(fragParams.paddedTailM, fragParams.tailM, fragParams.rankSize, fragParams.scaleKLen),
        tailFrag_.addrListScaleA);
    tailFrag_.c = Apace::Basic::MakeFragmentTensor<DIMS_NUM, MAX_FRAG, MakeLayoutC, CType>(
        MakeFragParam(fragParams.paddedTailM, fragParams.tailM, fragParams.rankSize, fragParams.n),
        tailFrag_.addrListC);
}

ALL_GATHER_QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void AllGatherQbmmMxKernel<ALL_GATHER_QBMM_MX_KERNEL_FUNC_TEM_PARAMS>::UpdateMainRoundAddrs(
    const Params& params, uint32_t roundIdx)
{
    const auto& fragParams = params.fragParams;
    uint32_t delta = roundIdx - curMainRoundIdx_;
    if constexpr (!enableBufferReuse) {
        curRoundDataOff_ += static_cast<uint64_t>(delta) * tileMDataStride_;
        curRoundScaleOff_ += static_cast<uint64_t>(delta) * tileMScaleStride_;
    }
    curRoundCOff_ += static_cast<uint64_t>(delta) * tileMCStride_;

    uint32_t fragIdx = 0;
    for (uint32_t r = 0; r < fragParams.rankSize; ++r) {
        if (r == fragParams.rankId) {
            continue;
        }
        if constexpr (!enableBufferReuse) {
            curMainFrag_.addrListA[fragIdx] = winDataRankBase_[r] + curRoundDataOff_;
            curMainFrag_.addrListScaleA[fragIdx] = winScaleRankBase_[r] + curRoundScaleOff_;
        }
        curMainFrag_.addrListC[fragIdx] = cFragAddrs_[r] + curRoundCOff_;
        fragIdx++;
    }
    if constexpr (!enableBufferReuse) {
        curMainFrag_.a.UpdateAddrList(curMainFrag_.addrListA);
        curMainFrag_.scaleA.UpdateAddrList(curMainFrag_.addrListScaleA);
    }
    curMainFrag_.c.UpdateAddrList(curMainFrag_.addrListC);
    curMainRoundIdx_ = roundIdx;
}

ALL_GATHER_QBMM_MX_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline typename AllGatherQbmmMxKernel<ALL_GATHER_QBMM_MX_KERNEL_FUNC_TEM_PARAMS>::TileCtx
AllGatherQbmmMxKernel<ALL_GATHER_QBMM_MX_KERNEL_FUNC_TEM_PARAMS>::ResolveTileCtx(int64_t mPos, int64_t headMainRows,
                                                                                 int64_t mainRoundRows,
                                                                                 int64_t mainSectionRows,
                                                                                 uint32_t commTurn) const
{
    TileCtx ctx{};
    if (mPos < headMainRows) {
        ctx.dependTileIdx = 0;
        ctx.region = HEAD;
        ctx.regionMPos = mPos;
        ctx.fragA = &headFrag_.a;
        ctx.fragScaleA = &headFrag_.scaleA;
        ctx.fragC = &headFrag_.c;
    } else if (mPos < headMainRows + mainSectionRows) {
        int64_t relPos = mPos - headMainRows;
        ctx.roundIdx = static_cast<uint32_t>(relPos / mainRoundRows);
        ctx.dependTileIdx = ctx.roundIdx + 1;
        ctx.region = MAIN;
        ctx.regionMPos = relPos % mainRoundRows;
        ctx.fragA = &curMainFrag_.a;
        ctx.fragScaleA = &curMainFrag_.scaleA;
        ctx.fragC = &curMainFrag_.c;
    } else {
        ctx.dependTileIdx = commTurn;
        ctx.region = TAIL;
        ctx.regionMPos = mPos - headMainRows - mainSectionRows;
        ctx.fragA = &tailFrag_.a;
        ctx.fragScaleA = &tailFrag_.scaleA;
        ctx.fragC = &tailFrag_.c;
    }
    return ctx;
}

} // namespace Apace
