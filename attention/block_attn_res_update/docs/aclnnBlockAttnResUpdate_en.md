# aclnnBlockAttnResUpdate

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T08:07:06.396Z -->

## Applicable Products

<!-- npu="950" id1 -->

- <term>Ascend 950PR&950DT products</term>: Supported

<!-- end id1 -->

<!-- npu="A3" id2 -->

- <term>Atlas A3 products</term>: Not supported

<!-- end id2 -->

<!-- npu="910b" id3 -->

- <term>Atlas A2 products</term>: Not supported

<!-- end id3 -->

<!-- npu="310b" id4 -->

- <term>Atlas 200I/500 A2 inference products</term>: Not supported

<!-- end id4 -->

<!-- npu="310p" id5 -->

- <term>Atlas inference products</term>: Not supported

<!-- end id5 -->

<!-- npu="910" id6 -->

- <term>Atlas training products</term>: Not supported

<!-- end id6 -->

## Function

- **Description**:

  Phase two of the two-phase historical residual attention computation for Attention Residuals: accumulates `delta` into `partialBlockRef` in place and computes the RMSNorm score of the updated residual; then computes the output `h` with the softmax statistics `numerator`, `logitMax`, and `expSum` (corresponding to O, M, and L, respectively) returned by the first-phase API `block_attn_res_prepare`, and writes the updated residual back to `partialBlockRef` in place.
- **Formula**:

  Let `T` denote the number of tokens, and `D` denote the hidden size. The main computation process is as follows:

  1. Update the current `partialBlockRef`:

     $$
     p[t,d] = partialBlockRef[t,d] + \operatorname{FP32}(delta[t,d])
     $$
  2. Calculate the RMSNorm denominator and the current score:

     $$
     r[t] = \sqrt{\frac{1}{D}\sum_{d=0}^{D-1}p[t,d]^2 + eps}
     $$

     $$
     score[t] = \frac{\sum_{d=0}^{D-1}p[t,d] \times pseudoQuery[d]}{r[t]}
     $$
  3. Merge the current score with the historical online softmax state:

     $$
     m[t] = \max(logitMax[t], score[t])
     $$

     $$
     alpha[t] = \exp(logitMax[t] - m[t]), \qquad beta[t] = \exp(score[t] - m[t])
     $$

     $$
     ell[t] = expSum[t] \times alpha[t] + beta[t]
     $$

     $$
     updatedNumerator[t,d] = numerator[t,d] \times alpha[t] + p[t,d] \times beta[t]
     $$
  4. Generate the output:

     $$
     partialBlockRef[t,d] \leftarrow p[t,d]
     $$

     $$
     h[t,d] = \operatorname{BF16}\left(\frac{updatedNumerator[t,d]}{ell[t]}\right)
     $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must call
`aclnnBlockAttnResUpdateGetWorkspaceSize` first to obtain the workspace size and executor, and then call
`aclnnBlockAttnResUpdate` to perform the computation.

```cpp
aclnnStatus aclnnBlockAttnResUpdateGetWorkspaceSize(
    aclTensor       *partialBlockRef,
    const aclTensor *delta,
    const aclTensor *pseudoQuery,
    const aclTensor *numerator,
    const aclTensor *logitMax,
    const aclTensor *expSum,
    double           eps,
    aclTensor       *h,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnBlockAttnResUpdate(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnBlockAttnResUpdateGetWorkspaceSize

- **Parameters**

  <table style="table-layout: fixed; width: 1554px"><colgroup>
  <col style="width: 248px">
  <col style="width: 121px">
  <col style="width: 210px">
  <col style="width: 327px">
  <col style="width: 160px">
  <col style="width: 115px">
  <col style="width: 138px">
  <col style="width: 145px">
  </colgroup>
    <thead>
      <tr>
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Instruction</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Dimension (shape)</th>
        <th>Non-contiguous Tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>partialBlockRef(aclTensor*)</td>
        <td>Input/Output</td>
        <td>The accumulated partialBlockRef.</td>
        <td>          <ul>
            <li>Required.</li>
            <li>The data type supports FLOAT.</li>
            <li>Both the original format and storage format only support ND.</li>
            <li>Must be a contiguous tensor.</li>
            <li><code>T >= 0</code>, <code>0 <= D <= 8192</code>.</li>
            <li>Supports empty tensors.</li>
          </ul>
        </td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[T, D]</td>
        <td>×</td>
      </tr>
      <tr>
        <td>delta(const aclTensor*)</td>
        <td>Input</td>
        <td>The delta to be accumulated.</td>
        <td>          <ul>
            <li>Required.</li>
            <li>Data type: BFLOAT16.</li>
            <li>Both the original format and storage format only support ND.</li>
            <li>Must be a contiguous tensor.</li>
            <li>The shape must be consistent with partialBlockRef.</li>
            <li>Supports empty tensor.</li>
          </ul>
        </td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>[T, D]</td>
        <td>×</td>
      </tr>
      <tr>
        <td>pseudoQuery(const aclTensor*)</td>
        <td>Input</td>
        <td>The pseudoQuery used to compute the logit of the current partial block.</td>
        <td>          <ul>
            <li>Required.</li>
            <li>Data type: FLOAT.</li>
            <li>Both the original format and storage format only support ND.</li>
            <li>Must be a contiguous tensor.</li>
            <li>The length must be equal to the D of partialBlockRef.</li>
            <li>Supports empty tensor.</li>
          </ul>
        </td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[D]</td>
        <td>×</td>
      </tr>
      <tr>
        <td>numerator(const aclTensor*)</td>
        <td>Input</td>
        <td>The historical online softmax weighted sum output by block_attn_res_prepare.</td>
        <td>          <ul>
            <li>Required.</li>
            <li>Supports the FLOAT data type.</li>
            <li>Both the original format and storage format only support ND.</li>
            <li>The tensor must be contiguous.</li>
            <li>The shape must be consistent with partialBlockRef.</li>
            <li>Supports empty tensors.</li>
          </ul>
        </td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[T, D]</td>
        <td>×</td>
      </tr>
      <tr>
        <td>logitMax(const aclTensor*)</td>
        <td>Input</td>
        <td>Historical maximum logit output by block_attn_res_prepare.</td>
        <td>          <ul>
            <li>Required.</li>
            <li>Data type: FLOAT.</li>
            <li>Both the original format and storage format only support ND.</li>
            <li>Must be a contiguous tensor.</li>
            <li>Length must be equal to T of partialBlockRef.</li>
            <li>Supports empty tensor.</li>
          </ul>
        </td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[T]</td>
        <td>×</td>
      </tr>
      <tr>
        <td>expSum(const aclTensor*)</td>
        <td>Input</td>
        <td>Historical softmax denominator accumulated values output by block_attn_res_prepare.</td>
        <td>          <ul>
            <li>Required.</li>
            <li>The data type supports FLOAT.</li>
            <li>Both the original format and storage format only support ND.</li>
            <li>Must be a contiguous tensor.</li>
            <li>Length must be equal to T of partialBlockRef.</li>
            <li>Supports empty tensors.</li>
          </ul>
        </td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[T]</td>
        <td>×</td>
      </tr>
      <tr>
        <td>eps(double)</td>
        <td>Input</td>
        <td>RMSNorm numerical stability term.</td>
        <td>          <ul>
            <li>Required.</li>
            <li>Must be a finite positive number.</li>
          </ul>
        </td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>h(aclTensor*)</td>
        <td>Output</td>
        <td>Result of the current layer.</td>
        <td>          <ul>
            <li>Required.</li>
            <li>Supported data type: BFLOAT16.</li>
            <li>Both the original format and storage format only support ND.</li>
            <li>Must be a contiguous tensor.</li>
            <li>shape must be consistent with partialBlockRef.</li>
            <li>Returns an empty tensor if the input is an empty tensor.</li>
          </ul>
        </td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>[T, D]</td>
        <td>×</td>
      </tr>
      <tr>
        <td>workspaceSize(uint64_t*)</td>
        <td>Output</td>
        <td>Returns the size of the workspace to be requested on the Device.</td>
        <td>          <ul>
            <li>Required.</li>
          </ul>
        </td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor(aclOpExecutor**)</td>
        <td>Output</td>
        <td>Returns the executor containing the operator computation flow.</td>
        <td>          <ul>
            <li>Required.</li>
          </ul>
        </td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
  </tbody></table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <div style="overflow-x: auto;">
  <table style="table-layout: fixed; width: 1100px">
    <colgroup>
      <col style="width: 250px">
      <col style="width: 130px">
      <col style="width: 720px">
    </colgroup>
    <thead>
      <tr>
        <th>Return Value</th>
        <th>Error Code</th>
        <th>Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td><code>partialBlockRef</code>, <code>delta</code>, <code>pseudoQuery</code>, <code>numerator</code>, <code>logitMax</code>, <code>expSum</code>, <code>h</code>, <code>workspaceSize</code>, or <code>executor</code> is null.</td>
      </tr>
      <tr>
        <td class="merged-cell" rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
        <td class="merged-cell" rowspan="5">161002</td>
        <td>The dtype, original or storage format, dimension, or shape relationship of the input or output does not meet the constraints.</td>
      </tr>
      <tr>
        <td><code>T < 0</code>, or <code>D</code> is not in the range <code>[0, 8192]</code>.</td>
      </tr>
      <tr>
        <td>In non-empty cases, <code>T * D</code> exceeds the range representable by <code>int64_t</code>.</td>
      </tr>
      <tr>
        <td>Any input or output Tensor is a non-contiguous Tensor.</td>
      </tr>
      <tr>
        <td><code>eps</code> is not a finite positive number.</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_RUNTIME_ERROR</td>
        <td>361001</td>
        <td>The product model is not in the supported range.</td>
      </tr>
    </tbody>
  </table>
  </div>

## aclnnBlockAttnResUpdate

- **Parameters**

  <table style="table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
  </colgroup>
  <thead>
    <tr>
    <th>Parameter</th>
    <th>Input/Output</th>
    <th>Description</th>
    </tr></thead>
  <tbody>
  <tr>
    <td>workspace</td>
    <td>Input</td>
    <td>Workspace memory address applied for on the Device side.</td>
  </tr>
  <tr>
    <td>workspaceSize</td>
    <td>Input</td>
    <td>Workspace size applied for on the Device side, obtained by calling the first-phase API aclnnBlockAttnResUpdateGetWorkspaceSize.</td>
  </tr>
  <tr>
    <td>executor</td>
    <td>Input</td>
    <td>Executor of the op, containing the operator computation flow.</td>
  </tr>
  <tr>
    <td>stream</td>
    <td>Input</td>
    <td>Specifies the Stream for task execution.</td>
  </tr>
  </tbody></table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnBlockAttnResUpdate defaults to a deterministic implementation.
- All Tensor inputs support empty Tensors, but the shape relationships between them must still be satisfied. A Tensor cannot be set to empty on its own.
- In non-empty cases, `T * D` must not exceed the range representable by `int64_t`.

## Example

The following sample code is for reference only. For details about the compilation and execution process, see
[Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md). The complete sample source code is in
[test_aclnn_block_attn_res_update.cpp](../examples/arch35/test_aclnn_block_attn_res_update.cpp).

```cpp
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_block_attn_res_update.h"

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define CHECK_FREE_RET(cond, return_expr)                                                                              \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            Finalize(deviceId, stream);                                                                                \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

constexpr size_t FIRST_ELEMENT_INDEX = 0UL;

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

uint16_t FloatToBfloat16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t roundingBias = 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<uint16_t>((bits + roundingBias) >> 16U);
}

float Bfloat16ToFloat(uint16_t value)
{
    const uint32_t bits = static_cast<uint32_t>(value) << 16U;
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclFormat formatType, aclTensor **tensor)
{
    const size_t size = static_cast<size_t>(GetShapeSize(shape)) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, formatType, shape.data(),
                              shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed - returned nullptr\n");
              return ACL_ERROR_FAILURE);
    return ACL_SUCCESS;
}

int RunBlockAttnResUpdate(int32_t deviceId, aclrtStream &stream)
{
    // 1. Initialize the Device and Stream.
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output data.
    constexpr int64_t tokenNum = 2;
    constexpr int64_t hiddenSize = 64;
    const std::vector<int64_t> matrixShape = {tokenNum, hiddenSize};
    const std::vector<int64_t> queryShape = {hiddenSize};
    const std::vector<int64_t> statsShape = {tokenNum};

    std::vector<float> partialBlockRefHostData(GetShapeSize(matrixShape), 0.25F);
    std::vector<uint16_t> deltaHostData(GetShapeSize(matrixShape), FloatToBfloat16(0.125F));
    std::vector<float> pseudoQueryHostData(GetShapeSize(queryShape), 1.0F / static_cast<float>(hiddenSize));
    std::vector<float> numeratorHostData(GetShapeSize(matrixShape), 0.5F);
    std::vector<float> logitMaxHostData(GetShapeSize(statsShape), 0.0F);
    std::vector<float> expSumHostData(GetShapeSize(statsShape), 1.0F);
    std::vector<uint16_t> hHostData(GetShapeSize(matrixShape), 0U);

    void *partialBlockRefDeviceAddr = nullptr;
    void *deltaDeviceAddr = nullptr;
    void *pseudoQueryDeviceAddr = nullptr;
    void *numeratorDeviceAddr = nullptr;
    void *logitMaxDeviceAddr = nullptr;
    void *expSumDeviceAddr = nullptr;
    void *hDeviceAddr = nullptr;

    aclTensor *partialBlockRef = nullptr;
    aclTensor *delta = nullptr;
    aclTensor *pseudoQuery = nullptr;
    aclTensor *numerator = nullptr;
    aclTensor *logitMax = nullptr;
    aclTensor *expSum = nullptr;
    aclTensor *h = nullptr;

    // 3. Create the input and output aclTensors.
    ret = CreateAclTensor<float>(partialBlockRefHostData, matrixShape, &partialBlockRefDeviceAddr,
                                 aclDataType::ACL_FLOAT, aclFormat::ACL_FORMAT_ND, &partialBlockRef);
    std::unique_ptr<void, aclError (*)(void *)> partialBlockRefDeviceAddrPtr(partialBlockRefDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> partialBlockRefTensorPtr(partialBlockRef,
                                                                                           aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<uint16_t>(deltaHostData, matrixShape, &deltaDeviceAddr, aclDataType::ACL_BF16,
                                    aclFormat::ACL_FORMAT_ND, &delta);
    std::unique_ptr<void, aclError (*)(void *)> deltaDeviceAddrPtr(deltaDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> deltaTensorPtr(delta, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<float>(pseudoQueryHostData, queryShape, &pseudoQueryDeviceAddr, aclDataType::ACL_FLOAT,
                                 aclFormat::ACL_FORMAT_ND, &pseudoQuery);
    std::unique_ptr<void, aclError (*)(void *)> pseudoQueryDeviceAddrPtr(pseudoQueryDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> pseudoQueryTensorPtr(pseudoQuery, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<float>(numeratorHostData, matrixShape, &numeratorDeviceAddr, aclDataType::ACL_FLOAT,
                                 aclFormat::ACL_FORMAT_ND, &numerator);
    std::unique_ptr<void, aclError (*)(void *)> numeratorDeviceAddrPtr(numeratorDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> numeratorTensorPtr(numerator, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<float>(logitMaxHostData, statsShape, &logitMaxDeviceAddr, aclDataType::ACL_FLOAT,
                                 aclFormat::ACL_FORMAT_ND, &logitMax);
    std::unique_ptr<void, aclError (*)(void *)> logitMaxDeviceAddrPtr(logitMaxDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> logitMaxTensorPtr(logitMax, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<float>(expSumHostData, statsShape, &expSumDeviceAddr, aclDataType::ACL_FLOAT,
                                 aclFormat::ACL_FORMAT_ND, &expSum);
    std::unique_ptr<void, aclError (*)(void *)> expSumDeviceAddrPtr(expSumDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> expSumTensorPtr(expSum, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<uint16_t>(hHostData, matrixShape, &hDeviceAddr, aclDataType::ACL_BF16,
                                    aclFormat::ACL_FORMAT_ND, &h);
    std::unique_ptr<void, aclError (*)(void *)> hDeviceAddrPtr(hDeviceAddr, aclrtFree);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> hTensorPtr(h, aclDestroyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    constexpr double eps = 1.0e-6;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;

    // 4. Call the first-phase API to obtain the workspace size and executor.
    ret = aclnnBlockAttnResUpdateGetWorkspaceSize(partialBlockRef, delta, pseudoQuery, numerator, logitMax, expSum,
                                                   eps, h, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnBlockAttnResUpdateGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        // Allocate the device-side workspace based on the size returned by the first-phase API.
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    // 5. Call the second-phase API to execute the operator.
    ret = aclnnBlockAttnResUpdate(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBlockAttnResUpdate failed. ERROR: %d\n", ret); return ret);

    // 6. Synchronously wait for task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 7. Copy the in-place update result and output result back to the host side.
    const size_t partialBlockRefSize = partialBlockRefHostData.size() * sizeof(float);
    ret = aclrtMemcpy(partialBlockRefHostData.data(), partialBlockRefSize, partialBlockRefDeviceAddr,
                      partialBlockRefSize,
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy partialBlockRef to host failed. ERROR: %d\n", ret); return ret);

    const size_t hSize = hHostData.size() * sizeof(uint16_t);
    ret = aclrtMemcpy(hHostData.data(), hSize, hDeviceAddr, hSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy h to host failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("partialBlockRef[0] after in-place update: %.6f\n", partialBlockRefHostData[FIRST_ELEMENT_INDEX]);
    LOG_PRINT("h[0]: %.6f\n", Bfloat16ToFloat(hHostData[FIRST_ELEMENT_INDEX]));
    LOG_PRINT("block_attn_res_update example execute success.\n");
    return ACL_SUCCESS;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = RunBlockAttnResUpdate(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("RunBlockAttnResUpdate failed. ERROR: %d\n", ret); return ret);

    // 8. Release Device and Stream resources.
    Finalize(deviceId, stream);
    return ACL_SUCCESS;
}
```
