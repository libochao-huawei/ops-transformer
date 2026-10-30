/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KEY_POOL_LAYER_NORM_H
#define KEY_POOL_LAYER_NORM_H

namespace KeyPool {

__aicore__ inline void KeyPoolLayerNorm(const LocalTensor<float>& data, const LocalTensor<float>& in,
                                        const LocalTensor<float>& out, const LocalTensor<float>& mean,
                                        const LocalTensor<float>& gamma, const LocalTensor<float>& beta, float eps,
                                        uint32_t count)
{
    const float reciprocal = 1.0f / static_cast<float>(static_cast<int64_t>(count));

    Duplicate(mean, reciprocal, count);
    PipeBarrier<PIPE_V>();
    Mul(in, data, mean, count);
    PipeBarrier<PIPE_V>();
    ReduceSum(in, in, in, count);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    float meanValue = in.GetValue(0);
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    Duplicate(mean, meanValue, count);
    PipeBarrier<PIPE_V>();

    Sub(data, data, mean, count);
    PipeBarrier<PIPE_V>();
    Mul(out, data, data, count);
    PipeBarrier<PIPE_V>();
    Muls(out, out, reciprocal, count);
    PipeBarrier<PIPE_V>();
    ReduceSum(out, out, out, count);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    float variance = out.GetValue(0);
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    Duplicate(out, variance, count);
    PipeBarrier<PIPE_V>();
    Adds(out, out, eps, count);
    PipeBarrier<PIPE_V>();
    Sqrt(out, out, count);
    PipeBarrier<PIPE_V>();

    Div(data, data, out, count);
    PipeBarrier<PIPE_V>();
    Mul(data, data, gamma, count);
    PipeBarrier<PIPE_V>();
    Add(data, data, beta, count);
    PipeBarrier<PIPE_V>();
}

// Use vector broadcast to avoid scalar reduction readback while retaining the
// FP32 reduction order and the existing five-vector scratch layout.
#if (__CCE_AICORE__ == 220)
__aicore__ inline void KeyPoolLayerNormBroadcast(const LocalTensor<float>& data, const LocalTensor<float>& input,
                                                 const LocalTensor<float>& output, const LocalTensor<float>& broadcast,
                                                 const LocalTensor<float>& gamma, const LocalTensor<float>& beta,
                                                 float eps)
{
    constexpr uint32_t HEAD_DIM = 128U;
    constexpr float HEAD_DIM_FLOAT = 128.0f;
    constexpr uint32_t VECTOR_PART_COUNT = 2U;
    constexpr uint32_t HALF_HEAD_DIM = HEAD_DIM / VECTOR_PART_COUNT;
    constexpr uint32_t BLOCK_ELEMENT_COUNT = 8U;
    constexpr uint8_t SCALAR_REPEAT_COUNT = 1U;
    constexpr uint8_t BINARY_REPEAT_COUNT = static_cast<uint8_t>(VECTOR_PART_COUNT);
    constexpr uint8_t BLOCK_STRIDE = 1U;
    constexpr uint8_t REPEAT_STRIDE = static_cast<uint8_t>(BLOCK_ELEMENT_COUNT);
    constexpr uint8_t BROADCAST_STRIDE = 0U;
    constexpr float INVERSE_HEAD_DIM = 1.0f / HEAD_DIM_FLOAT;
    Duplicate(broadcast, INVERSE_HEAD_DIM, HEAD_DIM);
    PipeBarrier<PIPE_V>();
    Mul(input, data, broadcast, HEAD_DIM);
    PipeBarrier<PIPE_V>();
    ReduceSum(input, input, input, HEAD_DIM);
    PipeBarrier<PIPE_V>();
    // Brcb copies the reduced scalar into consecutive 32-byte FP32 blocks.
    // Sub and Div consume those blocks as a broadcast operand for both halves.
    Brcb(broadcast, input, SCALAR_REPEAT_COUNT, {BLOCK_STRIDE, REPEAT_STRIDE});
    PipeBarrier<PIPE_V>();
    // BLOCK_STRIDE advances one FP32 block within a half row, REPEAT_STRIDE
    // advances one full half row, and BROADCAST_STRIDE keeps the source fixed.
    Sub(data, data, broadcast, HALF_HEAD_DIM, BINARY_REPEAT_COUNT,
        {BLOCK_STRIDE, BLOCK_STRIDE, BROADCAST_STRIDE, REPEAT_STRIDE, REPEAT_STRIDE, BROADCAST_STRIDE});
    PipeBarrier<PIPE_V>();
    Mul(output, data, data, HEAD_DIM);
    PipeBarrier<PIPE_V>();
    Muls(output, output, INVERSE_HEAD_DIM, HEAD_DIM);
    PipeBarrier<PIPE_V>();
    ReduceSum(output, output, output, HEAD_DIM);
    PipeBarrier<PIPE_V>();
    Brcb(broadcast, output, SCALAR_REPEAT_COUNT, {BLOCK_STRIDE, REPEAT_STRIDE});
    PipeBarrier<PIPE_V>();
    Adds(broadcast, broadcast, eps, BLOCK_ELEMENT_COUNT);
    PipeBarrier<PIPE_V>();
    Sqrt(broadcast, broadcast, BLOCK_ELEMENT_COUNT);
    PipeBarrier<PIPE_V>();
    Div(data, data, broadcast, HALF_HEAD_DIM, BINARY_REPEAT_COUNT,
        {BLOCK_STRIDE, BLOCK_STRIDE, BROADCAST_STRIDE, REPEAT_STRIDE, REPEAT_STRIDE, BROADCAST_STRIDE});
    PipeBarrier<PIPE_V>();
    Mul(data, data, gamma, HEAD_DIM);
    PipeBarrier<PIPE_V>();
    Add(data, data, beta, HEAD_DIM);
    PipeBarrier<PIPE_V>();
}
#endif

// Normalize complete current K rows before they enter the pooling path. The
// caller provides a scratch tensor with room for input/output and three
// vectors: mean, gamma, and beta.
__aicore__ inline void KeyPoolLayerNormRowsInplace(const LocalTensor<float>& data, const LocalTensor<float>& scratch,
                                                   const GlobalTensor<float>& normWeight,
                                                   const GlobalTensor<float>& normBias, float eps, uint32_t rowCount,
                                                   uint32_t rowStride, uint32_t headDim)
{
    if (rowCount == 0 || rowStride < headDim) {
        return;
    }
    LocalTensor<float> input = scratch;
    LocalTensor<float> output = input[headDim];
    LocalTensor<float> mean = output[headDim];
    LocalTensor<float> gamma = mean[headDim];
    LocalTensor<float> beta = gamma[headDim];
    DataCopy(gamma, normWeight, headDim);
    DataCopy(beta, normBias, headDim);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
#if (__CCE_AICORE__ == 220)
    constexpr uint32_t HEAD_DIM = 128U;
    if (headDim == HEAD_DIM) {
        for (uint32_t row = 0; row < rowCount; ++row) {
            KeyPoolLayerNormBroadcast(data[static_cast<uint64_t>(row) * rowStride], input, output, mean, gamma, beta,
                                      eps);
        }
        return;
    }
#endif
    for (uint32_t row = 0; row < rowCount; ++row) {
        KeyPoolLayerNorm(data[static_cast<uint64_t>(row) * rowStride], input, output, mean, gamma, beta, eps, headDim);
    }
}

__aicore__ inline void ApplyKeyPoolRotaryPlaceholder(const LocalTensor<float>&, uint32_t)
{
    // RoPE inputs are rejected by the Host in this version. Keep the transform
    // boundary explicit so a later RoPE implementation is inserted between
    // normalization and cache writeback without changing the Pool stage.
}

} // namespace KeyPool

#endif // KEY_POOL_LAYER_NORM_H
