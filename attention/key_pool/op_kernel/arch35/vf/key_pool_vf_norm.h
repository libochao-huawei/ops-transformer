/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KEY_POOL_VF_NORM_H
#define KEY_POOL_VF_NORM_H
#include "kernel_tensor.h"
namespace KeyPoolNormVector {
using namespace AscendC;
using namespace AscendC::Reg;
constexpr uint32_t HEAD_DIM = 128U;
constexpr float HEAD_DIM_FLOAT = 128.0f;
constexpr uint32_t VECTOR_PART_COUNT = 2U;
constexpr uint32_t HALF_HEAD_DIM = HEAD_DIM / VECTOR_PART_COUNT;
constexpr uint32_t BETA_OFFSET = HEAD_DIM;
constexpr float INVERSE_HEAD_DIM = 1.0f / HEAD_DIM_FLOAT;

// params holds gamma[HEAD_DIM] followed by beta[HEAD_DIM]. Each vector part
// processes half of a row so all four parameter segments fit in registers.
__simd_vf__ inline void NormalizeRows(__ubuf__ float* data, __ubuf__ float* params, uint32_t rows, uint32_t stride,
                                      float eps)
{
    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> gamma0, gamma1, beta0, beta1;
    LoadAlign(gamma0, params);
    LoadAlign(gamma1, params + HALF_HEAD_DIM);
    LoadAlign(beta0, params + BETA_OFFSET);
    LoadAlign(beta1, params + BETA_OFFSET + HALF_HEAD_DIM);
    for (uint32_t row = 0; row < rows; ++row) {
        RegTensor<float> x0, x1, scaled0, scaled1, sum0, sum1, mean;
        RegTensor<float> sq0, sq1, variance, stddev, out0, out1;
        LoadAlign(x0, data + row * stride);
        LoadAlign(x1, data + row * stride + HALF_HEAD_DIM);
        // Scale each element by 1 / HEAD_DIM before reducing so summing both
        // partial rows produces the full-row mean directly.
        Muls(scaled0, x0, INVERSE_HEAD_DIM, mask);
        Muls(scaled1, x1, INVERSE_HEAD_DIM, mask);
        Add(scaled0, scaled0, scaled1, mask);
        Reduce<ReduceType::SUM, float, float>(sum0, scaled0, mask);
        Duplicate(mean, sum0, mask);
        Sub(x0, x0, mean, mask);
        Sub(x1, x1, mean, mask);
        Mul(sq0, x0, x0, mask);
        Mul(sq1, x1, x1, mask);
        // Apply the same full-row scale to both squared partial rows.
        Muls(sq0, sq0, INVERSE_HEAD_DIM, mask);
        Muls(sq1, sq1, INVERSE_HEAD_DIM, mask);
        Add(sq0, sq0, sq1, mask);
        Reduce<ReduceType::SUM, float, float>(sum1, sq0, mask);
        Duplicate(variance, sum1, mask);
        Adds(variance, variance, eps, mask);
        Sqrt(stddev, variance, mask);
        Div(out0, x0, stddev, mask);
        Div(out1, x1, stddev, mask);
        Mul(out0, out0, gamma0, mask);
        Mul(out1, out1, gamma1, mask);
        Add(out0, out0, beta0, mask);
        Add(out1, out1, beta1, mask);
        StoreAlign(data + row * stride, out0, mask);
        StoreAlign(data + row * stride + HALF_HEAD_DIM, out1, mask);
    }
}
} // namespace KeyPoolNormVector
namespace KeyPool {
__aicore__ inline void KeyPoolLayerNormRowsVectorized(const LocalTensor<float>& data, const LocalTensor<float>& scratch,
                                                      const GlobalTensor<float>& gamma, const GlobalTensor<float>& beta,
                                                      float eps, uint32_t rows, uint32_t stride, uint32_t dim)
{
    if (dim != KeyPoolNormVector::HEAD_DIM || rows == 0U) {
        KeyPoolLayerNormRowsInplace(data, scratch, gamma, beta, eps, rows, stride, dim);
        return;
    }
    DataCopy(scratch, gamma, KeyPoolNormVector::HEAD_DIM);
    DataCopy(scratch[KeyPoolNormVector::HEAD_DIM], beta, KeyPoolNormVector::HEAD_DIM);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    // NormalizeRows overwrites data in place and consumes the contiguous
    // gamma/beta buffer prepared above.
    KeyPoolNormVector::NormalizeRows((__ubuf__ float*)data.GetPhyAddr(), (__ubuf__ float*)scratch.GetPhyAddr(), rows,
                                     stride, eps);
    PipeBarrier<PIPE_V>();
}
} // namespace KeyPool
#endif
