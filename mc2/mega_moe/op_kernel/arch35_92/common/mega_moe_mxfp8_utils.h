/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_MXFP8_UTILS_H
#define MEGA_MOE_MXFP8_UTILS_H

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/reg_compute/kernel_reg_compute_intf.h"
#endif
#include "mega_moe_constants.h"
#include "mega_moe_utils.h"
#include "../../../../common/op_kernel/quantize_functions.h"
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "c_api/asc_simd.h"
#endif

namespace MegaMoeImpl {

using namespace AscendC;

namespace Mxfp8 {

// 将一行 BF16 token 量化为带 padding 的 MXFP8 data + scale 记录。
template <uint8_t QuantMode, typename ExpandXType>
__aicore__ inline void QuantMxFp8(__ubuf__ ExpandXType* outLocal, __ubuf__ ExpandXType* inLocal,
                                  __ubuf__ float* floatTemp, int32_t processLen)
{
    uint32_t mxScaleNum = Ops::Base::CeilAlign(
        Ops::Base::CeilDiv(static_cast<uint32_t>(processLen), static_cast<uint32_t>(ALIGN_32)), 2U);
    using Fp8Type = typename std::conditional<QuantMode == MXFP8_E4M3_COMM_QUANT, fp8_e4m3fn_t, fp8_e5m2_t>::type;
    __ubuf__ Fp8Type* quantData = reinterpret_cast<__ubuf__ Fp8Type*>(outLocal);
    __ubuf__ ExpandXType* srcAddr = (__ubuf__ ExpandXType*)inLocal;
    __ubuf__ uint16_t* maxExpAddr = (__ubuf__ uint16_t*)floatTemp;
    __ubuf__ uint16_t* halfScaleLocalAddr =
        (__ubuf__ uint16_t*)(floatTemp + (Ops::Base::CeilAlign(mxScaleNum, static_cast<uint32_t>(ALIGN_32))));
    __ubuf__ int8_t* outLocalAddr = (__ubuf__ int8_t*)quantData;
    uint32_t tokenStorageElementCount =
        Ops::Base::CeilAlign(static_cast<uint32_t>(processLen), static_cast<uint32_t>(ALIGN_256));
    __ubuf__ uint16_t* mxScaleLocalAddr = (__ubuf__ uint16_t*)(quantData + (tokenStorageElementCount));
    Quant::ComputeMaxExp(srcAddr, maxExpAddr, static_cast<uint32_t>(processLen));
    Quant::ComputeScale<Fp8Type>(maxExpAddr, mxScaleLocalAddr, halfScaleLocalAddr, mxScaleNum);
    Quant::ComputeFp8Data<ExpandXType, Fp8Type, AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
        srcAddr, halfScaleLocalAddr, outLocalAddr, static_cast<uint32_t>(processLen));
}

#if __NPU_ARCH__ != 3510
// 将一条 MXFP8 token 记录反量化为 FP32，供 Unpermute 累加。
__simd_callee__ inline void ConvertMxScaleToBf16(__ubuf__ fp8_e8m0_t* srcPtr0, __ubuf__ bfloat16_t* dyScaleBf16Ptr,
                                                 uint32_t scaleLen, uint32_t bf16RepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<fp8_e8m0_t> vSrcReg;
    AscendC::Reg::RegTensor<bfloat16_t> vDstReg;
    static constexpr AscendC::Reg::CastTrait FP82BF16CastTraitZero = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
    AscendC::Reg::MaskReg maskReg;
    for (uint16_t i = 0; i < repeatTimes; i++) {
        maskReg = AscendC::Reg::UpdateMask<bfloat16_t>(scaleLen);
        Reg::DataCopy<fp8_e8m0_t, Reg::LoadDist::DIST_UNPACK_B8>(vSrcReg, srcPtr0 + i * bf16RepeatSize);
        Reg::Cast<bfloat16_t, fp8_e8m0_t, FP82BF16CastTraitZero>(vDstReg, vSrcReg, maskReg);
        Reg::DataCopy<bfloat16_t, Reg::StoreDist::DIST_INTLV_B16>(dyScaleBf16Ptr + i * bf16RepeatSize * 2, vDstReg,
                                                                  vDstReg, maskReg);
    }
    Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
}

__simd_callee__ inline void ConvertMxScaleToFp32(__ubuf__ bfloat16_t* dyScaleBf16Ptr, __ubuf__ float* dyScaleFp32Ptr,
                                                 uint32_t quantCount, uint32_t fp32RepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<bfloat16_t> dyScaleBf16Reg;
    AscendC::Reg::RegTensor<float> dyScaleFp32Reg;
    static constexpr AscendC::Reg::CastTrait FP162FP32CastTraitZero = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
    AscendC::Reg::MaskReg maskReg1;
    for (uint16_t i = 0; i < repeatTimes; i++) {
        maskReg1 = AscendC::Reg::UpdateMask<float>(quantCount);
        Reg::DataCopy<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(dyScaleBf16Reg, dyScaleBf16Ptr + i * fp32RepeatSize);
        Reg::Cast<float, bfloat16_t, FP162FP32CastTraitZero>(dyScaleFp32Reg, dyScaleBf16Reg, maskReg1);
        Reg::DataCopy<float, Reg::StoreDist::DIST_INTLV_B32>(dyScaleFp32Ptr + i * fp32RepeatSize * 2, dyScaleFp32Reg,
                                                             dyScaleFp32Reg, maskReg1);
    }
    Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
}

template <typename T>
__simd_callee__ inline void DequantizeMxFp8Token(__ubuf__ T* tokenPtr0, __ubuf__ float* dyScaleFp32Ptr,
                                                 __ubuf__ float* sumDstPtr, uint32_t tokenLen, uint32_t fp32RepeatSize,
                                                 uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> tokenSrcReg;
    AscendC::Reg::RegTensor<float> tokenFp32SrcReg;
    AscendC::Reg::RegTensor<float> dyScaleFp32Reg;
    AscendC::Reg::RegTensor<float> sumDstReg;
    AscendC::Reg::RegTensor<float> sumLocalDstReg;
    static constexpr AscendC::Reg::CastTrait FP82BF16CastTraitZero = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
    AscendC::Reg::MaskReg maskReg2;
    for (uint16_t i = 0; i < repeatTimes; i++) {
        maskReg2 = AscendC::Reg::UpdateMask<float>(tokenLen);
        Reg::DataCopy<float, Reg::LoadDist::DIST_E2B_B32>(dyScaleFp32Reg, dyScaleFp32Ptr + i * 8);
        Reg::DataCopy<T, Reg::LoadDist::DIST_UNPACK4_B8>(tokenSrcReg, tokenPtr0 + i * fp32RepeatSize);
        Reg::Cast<float, T, FP82BF16CastTraitZero>(tokenFp32SrcReg, tokenSrcReg, maskReg2);
        Reg::Mul(sumLocalDstReg, dyScaleFp32Reg, tokenFp32SrcReg, maskReg2);
        Reg::DataCopy(sumDstPtr + i * fp32RepeatSize, sumLocalDstReg, maskReg2);
    }
}

template <typename T, typename XType>
__simd_vf__ inline void DeQuantMxFp8Vf(__ubuf__ XType* inAddr, __ubuf__ float* sumDstPtr,
                                       __ubuf__ bfloat16_t* dyScaleBf16Ptr, __ubuf__ float* dyScaleFp32Ptr,
                                       uint32_t scaleLen, uint32_t tokenLen)
{
    __ubuf__ T* tokenPtr0 = (__ubuf__ T*)inAddr;
    uint32_t tokenStorageElementCount = ((tokenLen + ALIGN_256 - 1U) / ALIGN_256 * ALIGN_256) / 2U;
    __ubuf__ fp8_e8m0_t* srcPtr0 = (__ubuf__ fp8_e8m0_t*)(inAddr + tokenStorageElementCount);
    constexpr uint32_t bf16RepeatSize = AscendC::VECTOR_REG_WIDTH / sizeof(bfloat16_t);
    constexpr uint32_t fp32RepeatSize = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    uint16_t bf16RepeatTimes = (scaleLen + bf16RepeatSize - 1U) / bf16RepeatSize;
    uint32_t quantCount = scaleLen * 2U;
    uint16_t scaleFp32RepeatTimes = (quantCount + fp32RepeatSize - 1U) / fp32RepeatSize;
    uint16_t tokenFp32RepeatTimes = (tokenLen + fp32RepeatSize - 1U) / fp32RepeatSize;
    ConvertMxScaleToBf16(srcPtr0, dyScaleBf16Ptr, scaleLen, bf16RepeatSize, bf16RepeatTimes);
    ConvertMxScaleToFp32(dyScaleBf16Ptr, dyScaleFp32Ptr, quantCount, fp32RepeatSize, scaleFp32RepeatTimes);
    DequantizeMxFp8Token(tokenPtr0, dyScaleFp32Ptr, sumDstPtr, tokenLen, fp32RepeatSize, tokenFp32RepeatTimes);
}

#else
__simd_callee__ inline void ConvertMxScaleToBf16(__ubuf__ fp8_e8m0_t* srcPtr, __ubuf__ bfloat16_t* dyScaleBf16Ptr,
                                                 uint32_t scaleLen, uint32_t bf16RepeatSize, uint16_t repeatTimes)
{
    auto* scalePtr = reinterpret_cast<__ubuf__ uint8_t*>(srcPtr);
    auto* dstPtr = reinterpret_cast<__ubuf__ uint16_t*>(dyScaleBf16Ptr);
    vector_uint8_t scaleReg;
    vector_uint16_t bf16Bits;
    vector_bool nanMask;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        vector_bool mask = asc_update_mask_b16(scaleLen);
        asc_loadalign_unpack_postupdate(scaleReg, scalePtr, bf16RepeatSize);
        // UNPACK_B8 already zero-extends each scale byte to a uint16_t lane.
        asc_shiftleft_scalar(bf16Bits, reinterpret_cast<vector_uint16_t&>(scaleReg), int16_t{7}, mask);
        asc_eq_scalar(nanMask, bf16Bits, uint16_t{0x7f80}, mask);
        vdup(bf16Bits, uint16_t{0x7fc0}, nanMask, MODE_MERGING);
        asc_storealign_intlv(dstPtr, bf16Bits, bf16Bits);
        dstPtr += bf16RepeatSize * 2U;
    }
    asc_mem_bar(VST_VLD);
}

__simd_callee__ inline void ConvertMxScaleToFp32(__ubuf__ bfloat16_t* dyScaleBf16Ptr, __ubuf__ float* dyScaleFp32Ptr,
                                                 uint32_t quantCount, uint32_t fp32RepeatSize, uint16_t repeatTimes)
{
    vector_bfloat16_t bf16Reg;
    vector_float fp32Reg;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        vector_bool mask = asc_update_mask_b32(quantCount);
        asc_loadalign_unpack_postupdate(bf16Reg, dyScaleBf16Ptr, fp32RepeatSize);
        asc_bfloat162float(fp32Reg, bf16Reg, mask, ASC_POSITION_EVEN);
        asc_storealign_intlv(reinterpret_cast<__ubuf__ uint32_t*>(dyScaleFp32Ptr),
                             reinterpret_cast<vector_uint32_t&>(fp32Reg), reinterpret_cast<vector_uint32_t&>(fp32Reg));
        dyScaleFp32Ptr += fp32RepeatSize * 2U;
    }
    asc_mem_bar(VST_VLD);
}

template <typename T>
__simd_callee__ inline void DequantizeMxFp8Token(__ubuf__ T* tokenPtr, __ubuf__ float* dyScaleFp32Ptr,
                                                 __ubuf__ float* sumDstPtr, uint32_t tokenLen, uint32_t fp32RepeatSize,
                                                 uint16_t repeatTimes)
{
    static_assert(IsSameType<T, fp8_e4m3fn_t>::value || IsSameType<T, fp8_e5m2_t>::value);
    using TokenReg =
        typename std::conditional<IsSameType<T, fp8_e4m3fn_t>::value, vector_fp8_e4m3fn_t, vector_fp8_e5m2_t>::type;
    TokenReg tokenReg;
    vector_float valueReg;
    vector_float scaleReg;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        vector_bool mask = asc_update_mask_b32(tokenLen);
        asc_loadalign_brc_elem2datablock_postupdate(scaleReg, dyScaleFp32Ptr, 8);
        asc_loadalign_unpack4_postupdate(tokenReg, tokenPtr, fp32RepeatSize);
        if constexpr (IsSameType<T, fp8_e4m3fn_t>::value) {
            asc_e4m32float(valueReg, tokenReg, mask, ASC_DISPERSE_FIRST_QUARTER);
        } else {
            asc_e5m22float(valueReg, tokenReg, mask, ASC_DISPERSE_FIRST_QUARTER);
        }
        asc_mul(valueReg, scaleReg, valueReg, mask);
        asc_storealign_postupdate(sumDstPtr, valueReg, fp32RepeatSize, mask);
    }
}

template <typename T, typename XType>
__simd_vf__ inline void DeQuantMxFp8Vf(__ubuf__ XType* inAddr, __ubuf__ float* sumDstPtr,
                                       __ubuf__ bfloat16_t* dyScaleBf16Ptr, __ubuf__ float* dyScaleFp32Ptr,
                                       uint32_t scaleLen, uint32_t tokenLen)
{
    auto* tokenPtr = reinterpret_cast<__ubuf__ T*>(inAddr);
    uint32_t tokenStorageElementCount = ((tokenLen + ALIGN_256 - 1U) / ALIGN_256 * ALIGN_256) / 2U;
    auto* scalePtr = reinterpret_cast<__ubuf__ fp8_e8m0_t*>(inAddr + tokenStorageElementCount);
    constexpr uint32_t bf16RepeatSize = AscendC::VECTOR_REG_WIDTH / sizeof(bfloat16_t);
    constexpr uint32_t fp32RepeatSize = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    uint16_t bf16RepeatTimes = (scaleLen + bf16RepeatSize - 1U) / bf16RepeatSize;
    uint32_t quantCount = scaleLen * 2U;
    uint16_t scaleFp32RepeatTimes = (quantCount + fp32RepeatSize - 1U) / fp32RepeatSize;
    uint16_t tokenFp32RepeatTimes = (tokenLen + fp32RepeatSize - 1U) / fp32RepeatSize;
    ConvertMxScaleToBf16(scalePtr, dyScaleBf16Ptr, scaleLen, bf16RepeatSize, bf16RepeatTimes);
    ConvertMxScaleToFp32(dyScaleBf16Ptr, dyScaleFp32Ptr, quantCount, fp32RepeatSize, scaleFp32RepeatTimes);
    DequantizeMxFp8Token(tokenPtr, dyScaleFp32Ptr, sumDstPtr, tokenLen, fp32RepeatSize, tokenFp32RepeatTimes);
}
#endif

template <typename T, typename XType>
__aicore__ inline void DeQuantMxFp8(__ubuf__ XType* inLocal, __ubuf__ float* sumTensor,
                                    __ubuf__ bfloat16_t* scaleBf16Tensor, __ubuf__ float* scaleFP32Tensor,
                                    uint32_t scaleLen, uint32_t tokenLen)
{
    __ubuf__ XType* inAddr = (__ubuf__ XType*)inLocal;
    __ubuf__ bfloat16_t* dyScaleBf16Ptr = (__ubuf__ bfloat16_t*)scaleBf16Tensor;
    __ubuf__ float* dyScaleFp32Ptr = (__ubuf__ float*)scaleFP32Tensor;
    __ubuf__ float* sumDstPtr = (__ubuf__ float*)sumTensor;
    CallVf<DeQuantMxFp8Vf<T, XType>>(inAddr, sumDstPtr, dyScaleBf16Ptr, dyScaleFp32Ptr, scaleLen, tokenLen);
}

} // namespace Mxfp8
} // namespace MegaMoeImpl

#endif // MEGA_MOE_MXFP8_UTILS_H
