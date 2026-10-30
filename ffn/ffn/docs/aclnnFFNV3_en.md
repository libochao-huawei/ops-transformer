# aclnnFFNV3

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T11:22:32.321Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Not supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: supports
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: supports
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: This FFN operator provides the computation capabilities of MoeFFN and FFN. When there is no expert group (expertTokens is empty), it is FFN; when there is an expert group, it is MoeFFN. Both are collectively referred to as FFN and belong to the MoE structure. MoE (Mixture-of-Experts) is a technique for training models with trillions of parameters. MoE decomposes the predictive modeling task into several subtasks, trains an expert model on each subtask, and develops a gating model that assigns one or more experts based on the input data, ultimately combining the computation results of multiple experts as the prediction result. The Mixture-of-Experts model assigns input data to the most relevant one or more experts and combines the computation results of all involved experts to determine the final result.

  Compared with the [FFNV2](aclnnFFNV2_en.md) API, **this API changes expertTokens from an array to a Tensor input.**

  Compared with the [FFN](aclnnFFN_en.md) API, **this API adds support for the expertTokens index input, distinguished by tokensIndexFlag. expertTokens is changed from an array to a Tensor input.**
- Formula:

  - **Non-quantized scenario:**

    $$
    y=activation(x * W1 + b1) * W2 + b2
    $$

  - **Quantization scenario:**

    $$
    y=((activation((x * W1 + b1) * deqScale1) * scale + offset) * W2 + b2) * deqScale2
    $$

  - **Pseudo-Quantized Scenario:**

    $$
    y=activation(x * ((W1 + antiquantOffset1) * antiquantScale1) + b1) * ((W2 + antiquantOffset2) * antiquantScale2) + b2
    $$

  **NOTE**

  Whether FFN delivers performance gains in scenarios without experts or with a single expert must be determined based on actual test results. When the vector latency of small operators corresponding to the FFN structure in the entire network exceeds 30 μs and accounts for more than 10% of the FFN structure, you can try using this fusion operator. If actual test results show performance degradation, do not use it.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call the "aclnnFFNV3GetWorkspaceSize" API to obtain the workspace size required for calculation and the executor that includes the operator calculation process, and then call the "aclnnFFNV3" API to execute the calculation.

```Cpp
aclnnStatus aclnnFFNV3GetWorkspaceSize(
  const aclTensor* x,
  const aclTensor* weight1,
  const aclTensor* weight2,
  const aclTensor* expertTokensOptional,
  const aclTensor* bias1Optional,
  const aclTensor* bias2Optional,
  const aclTensor* scaleOptional,
  const aclTensor* offsetOptional,
  const aclTensor* deqScale1Optional,
  const aclTensor* deqScale2Optional,
  const aclTensor* antiquantScale1Optional,
  const aclTensor* antiquantScale2Optional,
  const aclTensor* antiquantOffset1Optional,
  const aclTensor* antiquantOffset2Optional,
  const char*      activation,
  int64_t          innerPrecise,
  bool             tokensIndexFlag,
  const aclTensor* y,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnFFNV3(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnFFNV3GetWorkspaceSize

**NOTE:** Description of the variables involved in the parameter descriptions below
  M indicates the number of tokens, corresponding to BS in the transformer (B: Batch, indicating the batch size of input samples.
  S: Seq-Length, indicating the sequence length of input samples).
  K1 indicates the input channel count of the first matmul, corresponding to H in the transformer (Head-Size, indicating the hidden layer size).
  N1 indicates the output channel count of the first matmul.
  K2 indicates the input channel count of the second matmul.
  N2 indicates the output channel count of the second matmul, corresponding to H in the transformer.
  <term>Atlas A2 products</term>: E indicates the number of experts when experts are present; G indicates the number of groups of antiquantOffset and antiquantScale in the pseudo-quantized per-group scenario.

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1645px">
  <colgroup> <!-- When adjusting column widths with the tool, the total column width must not exceed 1550 -->
  <col style="width: 180px"> <!-- Parameter: adjust the column width automatically. Principle: the content must not wrap to a new line -->
  <col style="width: 120px"> <!-- Input/Output: fixed column width -->
  <col style="width: 280px"> <!-- Description column width: adjust the column width automatically -->
  <col style="width: 300px"> <!-- Instruction column width: adjust the column width automatically -->
  <col style="width: 300px"> <!-- Data type column width: adjust the column width automatically -->
  <col style="width: 120px"> <!-- Data format: adjust the column width automatically -->
  <col style="width: 300px"> <!-- Dimension (shape): adjust the column width automatically. Reduce the column width appropriately when the content is minimal -->
  <col style="width: 145px"> <!-- Non-contiguous tensor: adjust the column width automatically. Do not modify this value -->
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
  </tr>
  </thead>
  <tbody>
  <tr>
  <td>x (aclTensor*)</td>
  <td>Input</td>
  <td>Calculation input, the input x in the formula.</td>
  <td>-</td>
  <td>FLOAT16, BFLOAT16, INT8</td>
  <td>ND</td>
  <td>-</td>
  <td>√</td>
  </tr>
  <tr>
  <td>weight1 (aclTensor*)</td>
  <td>Input</td>
  <td>Expert weight data, W1 in the formula.</td>
  <td>-</td>
  <td>FLOAT16, BFLOAT16, INT8, INT4</td>
  <td>ND</td>
  <td>-</td>
  <td>√</td>
  </tr>
  <tr>
  <td>weight2 (aclTensor*)</td>
  <td>Input</td>
  <td>Expert weight data, W2 in the formula.</td>
  <td>-</td>
  <td>FLOAT16, BFLOAT16, INT8, INT4</td>
  <td>ND</td>
  <td>-</td>
  <td>√</td>
  </tr>
  <tr>
  <td>expertTokensOptional (aclTensor*)</td>
  <td>Optional input</td>
  <td>Token count for each expert.</td>
  <td>-</td>
  <td>INT64</td>
  <td>ND</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>bias1Optional (aclTensor*)</td>
  <td>Optional input</td>
  <td>Weight data correction value, which is b1 in the formula.</td>
  <td>-</td>
  <td>FLOAT16, FLOAT32, INT32</td>
  <td>ND</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>bias2Optional(aclTensor*)</td>
  <td>Optional input</td>
  <td>Weight data correction value, b2 in the formula.</td>
  <td>-</td>
  <td>FLOAT16, FLOAT32, INT32</td>
  <td>ND</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>scaleOptional(aclTensor*)</td>
  <td>Optional input</td>
  <td>Quantization parameter, quantization scale factor.</td>
  <td>-</td>
  <td>FLOAT32</td>
  <td>ND</td>
  <td>-</td>
  <td>√</td>
  </tr>
  <tr>
  <td>offsetOptional (aclTensor*)</td>
  <td>Optional input</td>
  <td>Quantization parameter, quantization offset.</td>
  <td>-</td>
  <td>FLOAT32</td>
  <td>ND</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>deqScale1Optional(aclTensor*)</td>
  <td>Optional input</td>
  <td>Quantization parameter, dequantization scaling coefficient of the first matmul.</td>
  <td>-</td>
  <td>UINT64, INT64, FLOAT32, BFLOAT16</td>
  <td>ND</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>deqScale2Optional (aclTensor*)</td>
  <td>Optional input</td>
  <td>Quantization parameter, dequantization scale of the second matmul.</td>
  <td>-</td>
  <td>UINT64, INT64, FLOAT32, BFLOAT16</td>
  <td>ND</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>antiquantScale1Optional (aclTensor*)</td>
  <td>Optional input</td>
  <td>Pseudo-quantization parameter, scaling coefficient of the first matmul.</td>
  <td>-</td>
  <td>FLOAT16, BFLOAT16</td>
  <td>ND</td>
  <td>-</td>
  <td>√</td>
  </tr>
  <tr>
  <td>antiquantScale2Optional (aclTensor*)</td>
  <td>Optional input</td>
  <td>Pseudo-quantization parameter, the scaling factor of the second matmul.</td>
  <td>-</td>
  <td>FLOAT16, BFLOAT16</td>
  <td>ND</td>
  <td>-</td>
  <td>√</td>
  </tr>
  <tr>
  <td>antiquantOffset1Optional (aclTensor*)</td>
  <td>Optional input</td>
  <td>Pseudo-quantization parameter, the offset of the first matmul.</td>
  <td>-</td>
  <td>FLOAT16, BFLOAT16</td>
  <td>ND</td>
  <td>-</td>
  <td>√</td>
  </tr>
  <tr>
  <td>antiquantOffset2Optional (aclTensor*)</td>
  <td>Optional input</td>
  <td>Pseudo-quantization parameter, which is the offset of the second matmul.</td>
  <td>-</td>
  <td>FLOAT16, BFLOAT16</td>
  <td>ND</td>
  <td>-</td>
  <td>√</td>
  </tr>
  <tr>
  <td>activation (char*)</td>
  <td>Input</td>
  <td>Indicates the activation function used, corresponding to activation in the formula.</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>innerPrecise (int64_t)</td>
  <td>Optional input</td>
  <td>Indicates a choice between high precision and high performance.</td>
  <td>  <ul>
  <li>When innerPrecise is 0, high-precision mode is enabled. In non-quantization scenarios where all mandatory parameters are FLOAT16, the input and output of the activation layer inside the operator are both calculated using the FLOAT32 data type.</li>
  <li>When innerPrecise is 1, high-performance mode is enabled.</li>
  </ul>
  </td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>tokensIndexFlag (bool)</td>
  <td>Optional input</td>
  <td>Indicates whether expertTokens is an index value.</td>
  <td>  <ul>
  <li>If tokensIndexFlag is true, indicates that expertTokens is an index value.</li>
  <li>If tokensIndexFlag is false, indicates that expertTokens is the token count of each expert.</li>
  </ul>
  </td>
  <td>bool</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>y (aclTensor*)</td>
  <td>Output</td>
  <td>Output y in the formula.</td>
  <td>-</td>
  <td>FLOAT16, BFLOAT16</td>
  <td>ND</td>
  <td>-</td>
  <td>√</td>
  </tr>
  <tr>
  <td>workspaceSize (uint64_t*)</td>
  <td>Output</td>
  <td>Returns the size of the workspace that needs to be applied for on the Device side.</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>executor (aclOpExecutor**)</td>
  <td>Output</td>
  <td>Returns the op executor, which includes calculation process of the operator.</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  </tr>
  </tbody>
  </table>

  <!-- npu="910b" id7 -->
  - <term>Atlas A2 products</term>:
    - x supports FLOAT16, BFLOAT16, and INT8, and supports input with a minimum of 2 dimensions [M, K1] and a maximum of 8 dimensions.
    - weight1 supports FLOAT16, BFLOAT16, INT8, and INT4. The dimension is [E, K1, N1] when experts are present and [K1, N1] when no experts are present.
    - weight2 supports FLOAT16, BFLOAT16, INT8, and INT4. The dimension is [E, K2, N2] when experts are present and [K2, N2] when no experts are present.
    - expertTokensOptional supports INT64. If it is not empty, the maximum supported length is 256, and it is a 1D vector.
    - bias1Optional supports FLOAT16, FLOAT32, and INT32. The dimension is [E, N1] when experts are present and [N1] when no experts are present.
    - bias2Optional supports FLOAT16, FLOAT32, and INT32. The dimension is [E, N2] when experts are present and [N2] when no experts are present.
    - scaleOptional supports FLOAT32: in the per-tensor case, the input is a 1D vector whether experts are present or not, and the number of input elements is [E] when experts are present and [1] when no experts are present; in the per-channel case, the input is a 2D vector when experts are present and a 1D vector when no experts are present, and the number of input elements is [E, N1] when experts are present and [N1] when no experts are present.
    - offsetOptional supports FLOAT32. It is a 1D vector, and the number of input elements is [E] when experts are present and [1] when no experts are present.
    - deqScale1Optional supports UINT64, INT64, FLOAT32, and BFLOAT16. The input is [E, N1] when experts are present and [N1] when no experts are present.
    - deqScale2Optional supports UINT64, INT64, FLOAT32, and BFLOAT16. The input dimensions are [E, N2]/[N2] respectively when experts are present/when no experts are present.
    - antiquantScale1Optional supports FLOAT16 and BFLOAT16. For per-channel, the input dimensions are [E, N1]/[N1] respectively when experts are present/when no experts are present; for per-group, they are [E, G, N1]/[G, N1] respectively when experts are present/when no experts are present.
    - antiquantScale2Optional supports FLOAT16 and BFLOAT16. For per-channel, the input dimensions are [E, N2]/[N2] respectively when experts are present/when no experts are present; for per-group, they are [E, G, N2]/[G, N2] respectively when experts are present/when no experts are present.
    - antiquantOffset1Optional supports FLOAT16 and BFLOAT16. For per-channel, the input dimensions are [E, N1]/[N1] respectively when experts are present/when no experts are present; for per-group, they are [E, G, N1]/[G, N1] respectively when experts are present/when no experts are present.
    - antiquantOffset2Optional supports FLOAT16 and BFLOAT16. For per-channel, the input dimensions are [E, N2]/[N2] respectively when experts are present/when no experts are present; for per-group, they are [E, G, N2]/[G, N2] respectively when experts are present/when no experts are present.
    - activation supports fastgelu/gelu/relu/silu and geglu/swiglu/reglu.
    - innerPrecise takes effect only when all mandatory parameters are FLOAT16 in the non-quantized field. In other use cases, no distinction is made between high precision and high performance.
    - y supports FLOAT16 and BFLOAT16. The output dimension is the same as that of x.
  <!-- end id7 -->
  <!-- npu="310p" id8 -->
  - <term>Atlas inference products</term>:
    - x supports FLOAT16, and its dimension is 2D [M, K1].
    - weight1 supports FLOAT16 and the dimension is 2D [K1, N1].
    - weight2 supports FLOAT16 and the dimension is 2D [K2, N2].
    - expertTokensOptional only supports passing a null pointer.
    - bias1Optional supports FLOAT16, and the input dimension is 1D [N1].
    - bias2Optional supports FLOAT16, and the input dimension is 1D [N2].
    - scaleOptional, offsetOptional, deqScale1Optional, deqScale2Optional, antiquantScale1Optional, antiquantScale2Optional, antiquantOffset1Optional, and antiquantOffset2Optional only support passing null pointers.
    - activation supports fastgelu/gelu/relu/silu.
    - innerPrecise only supports passing 1.
    - y supports FLOAT16, and the output dimension is the same as that of x.
  <!-- end id8 -->

- **Return value**

  Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation. If the following error codes occur, the corresponding causes are:

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 294px">
  <col style="width: 134px">
  <col style="width: 722px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>A required input, output, or attribute parameter is passed as a null pointer. 161001 is returned.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data types or data formats of x, weight1, weight2, activation, expertTokensOptional, bias1Optional, bias2Optional, and y are not within the supported range.</td>
    </tr>
  </tbody>
  </table>

## aclnnFFNV3

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 167px">
  <col style="width: 134px">
  <col style="width: 848px">
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
      <td>Workspace memory address applied by the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size applied by the Device side, obtained by the first-phase API aclnnFFNV3GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which includes calculation flows of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- deterministic computation:
  - aclnnFFNV3 defaults to a non-deterministic implementation. You can enable deterministic computation through aclrtSetSysParamOpt.
- In all scenarios, K1 = N2, K1 < 65536, K2 < 65536, and the M-axis after 32-byte alignment must be less than the maximum value of INT32.

<!-- npu="910b" id9 -->
- <term>Atlas A2 products</term>:
  - Scenarios involving the BFLOAT16 data type are supported only on Atlas 800I A2 inference products.
  - With expert, the total number of expert data must be consistent with M of x.
  - When the activation layer is geglu/swiglu/reglu, only the FLOAT16 high-performance scenario without expert grouping is supported (the FLOAT16 scenario refers to the scenario where all required parameters of the aclTensor type have the FLOAT16 data type), and N1=2*K2.
  - When the activation layer is gelu/fastgelu/relu/silu, the FLOAT16 high-precision and high-performance scenarios, BFLOAT16 scenario, quantization scenario, and pseudo-quantization scenario with or without expert grouping are supported, and N1=K2.
  - Quantization parameters and pseudo-quantization parameters cannot be input in the non-quantization scenario, pseudo-quantization parameters cannot be input in the quantization scenario, and quantization parameters cannot be input in the pseudo-quantization scenario.
  - Data types in the quantization scenario: x is INT8, weight is INT8, biasOptional is INT32, scaleOptional is FLOAT32, offsetOptional is FLOAT32. The data types of other parameters fall into two cases depending on y:
    - When y is FLOAT16, deqScaleOptional supports the following data types: UINT64, INT64, FLOAT32.
    - When y is BFLOAT16, deqScaleOptional supports the following data type: BFLOAT16.
    - The data types of deqScale1Optional and deqScale2Optional must be consistent.
  - Data types in the quantization scenario that supports the per-channel mode of scale: x is INT8, weight is INT8, biasOptional is INT32, scaleOptional is FLOAT32, offsetOptional is FLOAT32. The data types of other parameters fall into two cases depending on y:
    - When y is FLOAT16, deqScaleOptional supports the following data types: UINT64 and INT64.
    - When y is BFLOAT16, deqScaleOptional supports the following data type: BFLOAT16.
    - The data types of deqScale1Optional and deqScale2Optional must be consistent.
  - The pseudo-quantization scenario supports two different parameter types:
    - y is FLOAT16, x is FLOAT16, biasOptional is FLOAT16, antiquantScaleOptional is FLOAT16, antiquantOffsetOptional is FLOAT16, and weight supports the INT8 and INT4 data types.
    - y is BFLOAT16, x is BFLOAT16, biasOptional is FLOAT32, antiquantScaleOptional is BFLOAT16, antiquantOffsetOptional is BFLOAT16, and weight supports INT8 and INT4 data types.
  - When the data type of weight1/weight2 is INT4, the last dimension of its shape must be an even number.
  - In the pseudo-quantization scenario, under per-group mode, the number of groups G in antiquantScale1Optional and antiquantOffset1Optional must be divisible by K1, and the number of groups G in antiquantScale2Optional and antiquantOffset2Optional must be divisible by K2.
  - In the BFLOAT16 non-quantization scenario, the innerPrecise parameter can only be set to 0. In the FLOAT16 non-quantization scenario, it can be set to 0 or 1. In the quantization or pseudo-quantization scenario, both 0 and 1 can be configured, but the configuration does not take effect.
  - When tokensIndexFlag is true and there are experts (expertTokens is not empty), the values in expertTokens must satisfy the following condition: if i and j are both valid array indices in expertTokens and j is greater than i, then the value of the j-th element in expertTokens must be greater than or equal to the value of the i-th element in expertTokens.
<!-- end id9 -->

<!-- npu="310p" id10 -->
- <term>Atlas inference products</term>:
  - Only the without-expert scenario is supported.
  - N1=K2 must be satisfied.
<!-- end id10 -->

## Example

This fused operator API supports only the aclnn single-operator call mode. The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_ffn_v3.h"
#include "aclnn/opdev/fp16_t.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // Boilerplate, resource initialization
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * aclDataTypeSize(dataType);
  // Call aclrtMalloc to allocate memory on the device side.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy host-side data to the device-side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor API to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) device/stream initialization. For details, see the list of public AscendCL APIs.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as required.
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs. Customize the construction based on the API.
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  std::vector<int64_t> weight1Shape = {2, 2};
  std::vector<int64_t> weight2Shape = {2, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* weight1DeviceAddr = nullptr;
  void* weight2DeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclTensor* weight1 = nullptr;
  aclTensor* weight2 = nullptr;
  std::vector<op::fp16_t> selfHostData = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  std::vector<op::fp16_t> outHostData = {0, 0, 0, 0};
  std::vector<op::fp16_t> weight1HostData = {0.1, 0.2, 0.3, 0.4};
  std::vector<op::fp16_t> weight2HostData = {0.4, 0.3, 0.2, 0.1};
  // Create self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT16, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create weight1 aclTensor.
  ret = CreateAclTensor(weight1HostData, weight1Shape, &weight1DeviceAddr, aclDataType::ACL_FLOAT16, &weight1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create weight2 aclTensor.
  ret = CreateAclTensor(weight2HostData, weight2Shape, &weight2DeviceAddr, aclDataType::ACL_FLOAT16, &weight2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnFFNV3 API call example
  LOG_PRINT("test aclnnFFNV3\n");

  // 3. Call the CANN operator library API, and replace it with the actual API name.
  // Call the first-phase API of aclnnFFNV3.
  ret = aclnnFFNV3GetWorkspaceSize(self, weight1, weight2, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,   "relu", 1, false, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFFNV3GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnFFNV3.
  ret = aclnnFFNV3(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFFNV3 failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronously wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output values and copy the result on the device memory to the host. Modify it according to the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<op::fp16_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    std::cout << "index: " << i << ": " << static_cast<float>(resultData[i]) << std::endl;
  }

  // 6. Release aclTensor. Modify it according to the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(out);
  aclDestroyTensor(weight1);
  aclDestroyTensor(weight2);

  // 7. Release device resources.
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(weight1DeviceAddr);
  aclrtFree(weight2DeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
