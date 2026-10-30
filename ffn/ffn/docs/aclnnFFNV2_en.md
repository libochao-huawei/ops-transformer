# aclnnFFNV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T11:13:33.790Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Not supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: This FFN operator provides the computation capabilities of MoeFFN and FFN. When there is no expert group (expertTokens is empty), it is FFN; when there is an expert group, it is MoeFFN. Both are collectively referred to as FFN and belong to the MoE structure. MoE (Mixture-of-Experts) is a technique for training models with trillions of parameters. MoE decomposes the predictive modeling task into several subtasks, trains an expert model on each subtask, and develops a gating model that assigns one or more experts based on the input data, ultimately combining the computation results of multiple experts as the prediction result. The Mixture-of-Experts model assigns input data to the most relevant one or more experts and combines the computation results of all involved experts to determine the final result.

  Compared with the [FFN](aclnnFFN_en.md) API, **this API additionally supports the expertTokens index input, which is distinguished by tokensIndexFlag.**
- Formula:

  - **Non-quantized scenario:**

    $$
    y=activation(x * W1 + b1) * W2 + b2
    $$

  - **Quantization use case:**

    $$
    y=((activation((x * W1 + b1) * deqScale1) * scale + offset) * W2 + b2) * deqScale2
    $$

  - **Pseudo-Quantized Scenario:**

    $$
    y=activation(x * ((W1 + antiquantOffset1) * antiquantScale1) + b1) * ((W2 + antiquantOffset2) * antiquantScale2) + b2
    $$

  **NOTE:**
  Whether FFN delivers a performance gain in the no-expert or single-expert use case depends on actual test results. Try this fusion operator if the vector time of the small operators corresponding to the FFN structure in the entire network exceeds 30us and accounts for more than 10% of the FFN structure time. Do not use it if actual tests show performance degradation.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnFFNV2GetWorkspaceSize" API first to obtain the workspace size required for computation and the executor that contains the computation process of the operator, and then call the "aclnnFFNV2" API to perform the computation.

```Cpp
aclnnStatus aclnnFFNV2GetWorkspaceSize(
  const aclTensor*   x,
  const aclTensor*   weight1,
  const aclTensor*   weight2,
  const aclIntArray* expertTokens,
  const aclTensor*   bias1,
  const aclTensor*   bias2,
  const aclTensor*   scale,
  const aclTensor*   offset,
  const aclTensor*   deqScale1,
  const aclTensor*   deqScale2,
  const aclTensor*   antiquantScale1,
  const aclTensor*   antiquantScale2,
  const aclTensor*   antiquantOffset1,
  const aclTensor*   antiquantOffset2,
  const char*        activation,
  int64_t            innerPrecise,
  bool               tokensIndexFlag,
  const aclTensor*   y,
  uint64_t*          workspaceSize,
  aclOpExecutor**    executor)
```

```Cpp
aclnnStatus aclnnFFNV2(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnFFNV2GetWorkspaceSize

Description of common variables in the Parameters section:
M indicates the number of tokens, corresponding to BS in the transformer (B: Batch, indicating the batch size of input samples. S: Seq-Length, indicating the sequence length of input samples).
K1 indicates the input channel count of the first matmul, corresponding to H in the transformer (Head-Size, indicating the hidden layer size).
N1 indicates the output channel count of the first matmul.
K2 indicates the input channel count of the second matmul.
N2 indicates the output channel count of the second matmul, corresponding to H in the transformer.
<term>Atlas A2 products</term>: E indicates the number of experts when experts are present; G indicates the number of groups of antiquantOffset and antiquantScale in the pseudo-quantization per-group scenario.

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1640px">
  <colgroup> <!-- When adjusting column widths using the tool, ensure the total width of columns does not exceed 1550 -->
  <col style="width: 180px"> <!-- Parameter: adjust the column width as needed; principle: content must be displayed on a single line -->
  <col style="width: 120px"> <!-- Input/Output: fixed column width -->
  <col style="width: 280px"> <!-- Description column width: adjust as needed -->
  <col style="width: 290px"> <!-- Instruction column width: adjust as needed -->
  <col style="width: 290px"> <!-- Data type column width: adjust as needed -->
  <col style="width: 120px"> <!-- Data format: adjust the column width as needed -->
  <col style="width: 320px"> <!-- Dimension (shape): adjust the column width as needed; reduce the width appropriately when the content is short -->
  <col style="width: 140px"> <!-- Non-contiguous tensor: adjust the column width as needed; do not modify this value -->
  </colgroup>
  <thead>
  <tr>
  <th>Parameter</th>
  <th>Input/Output</th>
  <th>Description</th>
  <th>Instruction</th>
  <th>Data Type</th>
  <th>Data Format</th>
  <th>Shape</th>
  <th>Non-contiguous Tensor</th>
  </tr>
  </thead>
  <tbody>
  <tr>
  <td>x (aclTensor*)</td>
  <td>Input</td>
  <td>Calculation input, which is x in the formula.</td>
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
  <td>expertTokens (aclIntArray*)</td>
  <td>Optional input</td>
  <td>Token count of each expert.</td>
  <td>-</td>
  <td>INT64</td>
  <td>ND</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>bias1 (aclTensor*)</td>
  <td>Optional input</td>
  <td>Bias correction value for weight data, corresponding to b1 in the formula.</td>
  <td>-</td>
  <td>FLOAT16, FLOAT32, INT32</td>
  <td>ND</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>bias2 (aclTensor*)</td>
  <td>Optional input</td>
  <td>Weight data correction value, which is b2 in the formula.</td>
  <td>-</td>
  <td>FLOAT16, FLOAT32, INT32</td>
  <td>ND</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>scale (aclTensor*)</td>
  <td>Optional input</td>
  <td>Quantization parameter, the quantization scaling factor.</td>
  <td>-</td>
  <td>FLOAT32</td>
  <td>ND</td>
  <td>-</td>
  <td>√</td>
  </tr>
  <tr>
  <td>offset (aclTensor*)</td>
  <td>Optional input</td>
  <td>Quantization parameter, the quantization offset.</td>
  <td>-</td>
  <td>FLOAT32</td>
  <td>ND</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>deqScale1 (aclTensor*)</td>
  <td>Optional input</td>
  <td>Quantization parameter. Dequantization scale coefficient for the first matmul.</td>
  <td>-</td>
  <td>UINT64, INT64, FLOAT32, BFLOAT16</td>
  <td>ND</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>deqScale2 (aclTensor*)</td>
  <td>Optional input</td>
  <td>Quantization parameter, the dequantization scale factor for the second matmul.</td>
  <td>-</td>
  <td>UINT64, INT64, FLOAT32, BFLOAT16</td>
  <td>ND</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>antiquantScale1(aclTensor*)</td>
  <td>Optional input</td>
  <td>Pseudo-quantization parameter, scaling coefficient of the first matmul.</td>
  <td>-</td>
  <td>FLOAT16, BFLOAT16</td>
  <td>ND</td>
  <td>-</td>
  <td>√</td>
  </tr>
  <tr>
  <td>antiquantScale2(aclTensor*)</td>
  <td>Optional input</td>
  <td>Pseudo-quantization parameter, scale factor of the second matmul.</td>
  <td>-</td>
  <td>FLOAT16, BFLOAT16</td>
  <td>ND</td>
  <td>-</td>
  <td>√</td>
  </tr>
  <tr>
  <td>antiquantOffset1 (aclTensor*)</td>
  <td>Optional input</td>
  <td>Pseudo-quantization parameter, offset of the first matmul.</td>
  <td>-</td>
  <td>FLOAT16, BFLOAT16</td>
  <td>ND</td>
  <td>-</td>
  <td>√</td>
  </tr>
  <tr>
  <td>antiquantOffset2 (aclTensor*)</td>
  <td>Optional input</td>
  <td>Pseudo-quantization parameter, the offset of the second matmul.</td>
  <td>-</td>
  <td>FLOAT16, BFLOAT16</td>
  <td>ND</td>
  <td>-</td>
  <td>√</td>
  </tr>
  <tr>
  <td>activation (char*)</td>
  <td>Input</td>
  <td>Represents the activation function used, i.e., activation in the formula.</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>innerPrecise (int64_t)</td>
  <td>Optional input</td>
  <td>Specifies high precision or high performance.</td>
  <td>  <ul>
  <li>When innerPrecise is 0, high precision mode is enabled. In the non-quantized field, when all mandatory parameters are FLOAT16, the operator internally uses the FLOAT32 data type to compute both the input and output of the activation layer.</li>
  <li>When innerPrecise is 1, high performance mode is enabled.</li>
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
  <li>If tokensIndexFlag is true, expertTokens is an index value.</li>
  <li>If tokensIndexFlag is false, expertTokens is the token count of each expert.</li>
  </ul>
  </td>
  <td>BOOL</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>y (aclTensor*)</td>
  <td>Output</td>
  <td>The output y in the formula.</td>
  <td>-</td>
  <td>FLOAT16, BFLOAT16</td>
  <td>ND</td>
  <td>-</td>
  <td>√</td>
  </tr>
  <tr>
  <td>workspaceSize (uint64_t*)</td>
  <td>Output</td>
  <td>Returns the size of the workspace to be allocated on the Device.</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  </tr>
  <tr>
  <td>executor (aclOpExecutor**)</td>
  <td>Output</td>
  <td>Returns the op executor that contains the operator computation flow.</td>
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
    - x supports FLOAT16, BFLOAT16, and INT8. The supported input has a minimum of 2 dimensions [M, K1] and a maximum of 8 dimensions.
    - weight1 supports FLOAT16, BFLOAT16, INT8, and INT4. The dimension is [E, K1, N1] when experts are present and [K1, N1] when experts are absent.
    - weight2 supports FLOAT16, BFLOAT16, INT8, and INT4. The dimension is [E, K2, N2] when experts are present and [K2, N2] when experts are absent.
    - expertTokens supports INT64. If not empty, the maximum supported length is 256.
    - bias1 supports FLOAT16, FLOAT32, and INT32. The dimension is [E, N1] when experts are present and [N1] when experts are absent.
    - bias2 supports FLOAT16, FLOAT32, and INT32. The dimension is [E, N2] when experts are present and [N2] when experts are absent.
    - scale supports FLOAT32: For per-tensor, the input is always a one-dimensional vector when experts are present/absent, and the number of input elements is [E]/[1] when experts are present/absent, respectively; for per-channel, the input is a two-dimensional vector/one-dimensional vector when experts are present/absent, and the number of input elements is [E, N1]/[N1] when experts are present/absent, respectively.
    - offset supports FLOAT32 and is a one-dimensional vector. The number of input elements is [E]/[1] when experts are present/absent, respectively.
    - deqScale1 supports UINT64, INT64, FLOAT32, and BFLOAT16. The input is [E, N1]/[N1] when experts are present/absent, respectively.
    - deqScale2 supports UINT64, INT64, FLOAT32, and BFLOAT16. The input is [E, N2] when experts are present and [N2] when experts are absent.
    - antiquantScale1 supports FLOAT16 and BFLOAT16. In per-channel mode, the input is [E, N1] when experts are present and [N1] when experts are absent. In per-group mode, the input is [E, G, N1] when experts are present and [G, N1] when experts are absent.
    - antiquantScale2 supports FLOAT16 and BFLOAT16. In per-channel mode, the input is [E, N2] when experts are present and [N2] when experts are absent. In per-group mode, the input is [E, G, N2] when experts are present and [G, N2] when experts are absent.
    - antiquantOffset1 supports FLOAT16 and BFLOAT16. In per-channel mode, the input is [E, N1] when experts are present and [N1] when experts are absent. In per-group mode, the input is [E, G, N1] when experts are present and [G, N1] when experts are absent.
    - antiquantOffset2 supports FLOAT16 and BFLOAT16. In per-channel mode, the input is [E, N2] when experts are present and [N2] when experts are absent. In per-group mode, the input is [E, G, N2] when experts are present and [G, N2] when experts are absent.
    - activation supports fastgelu/gelu/relu/silu as well as geglu/swiglu/reglu.
    - innerPrecise takes effect only in the non-quantized field when all mandatory parameters are FLOAT16. In other use cases, there is no distinction between high precision and high performance.
    - y supports FLOAT16 and BFLOAT16. The output dimension is the same as that of x.
  <!-- end id7 -->
  <!-- npu="310p" id8 -->
  - <term>Atlas inference products</term>:
    - x supports FLOAT16. The dimension is two-dimensional [M, K1].
    - weight1 supports FLOAT16, and the dimension is 2D [K1, N1].
    - weight2 supports FLOAT16, and the dimension is 2D [K2, N2].
    - expertTokens supports only null pointers.
    - bias1 supports FLOAT16, and the supported input dimension is 1D [N1].
    - bias2 supports FLOAT16, and the supported input dimension is 1D [N2].
    - scale, offset, deqScale1, deqScale2, antiquantScale1, antiquantScale2, antiquantOffset1, and antiquantOffset2 support only null pointers.
    - activation supports fastgelu/gelu/relu/silu.
    - innerPrecise supports only the value 1.
    - y supports FLOAT16, and the output dimension is the same as that of x.
  <!-- end id8 -->

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation. If the following error codes occur, the corresponding causes are:

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 287px">
  <col style="width: 119px">
  <col style="width: 743px">
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
      <td>If the input parameter is a required input, output, or required attribute and is a null pointer, 161001 is returned.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data types and data formats of x, weight1, weight2, activation, expertTokens, bias1, bias2, and y are out of the supported range.</td>
    </tr>
  </tbody>
  </table>

## aclnnFFNV2

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
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
      <td>Workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device, obtained by calling the first-phase API aclnnFFNV2GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the operator, which contains the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnFFNV2 defaults to a non-deterministic implementation. You can enable deterministic computation through aclrtSetSysParamOpt.
- In all scenarios, K1 = N2, K1 < 65536, K2 < 65536, and the M-axis after 32-byte alignment must be less than the maximum value of INT32.

<!-- npu="910b" id9 -->
- <term>Atlas A2 products</term>:
  - With expert, the total number of expert data must be consistent with M of x.
  - When the activation layer is geglu/swiglu/reglu, only the FLOAT16 high-performance scenario without expert grouping is supported (FLOAT16 scenario refers to the scenario where all required parameters of the aclTensor type are of the FLOAT16 data type), and N1=2\*K2.
  - When the activation layer is gelu/fastgelu/relu/silu, the FLOAT16 high-precision and high-performance scenarios, BFLOAT16 scenario, quantization scenario, and pseudo-quantization scenario with or without expert grouping are supported, and N1=K2.
  - Quantization parameters and pseudo-quantization parameters cannot be input in the non-quantization scenario, pseudo-quantization parameters cannot be input in the quantization scenario, and quantization parameters cannot be input in the pseudo-quantization scenario.
  - Quantization scenario parameter types: x is INT8, weight is INT8, bias is INT32, scale is FLOAT32, offset is FLOAT32. The types of other parameters fall into two cases depending on y:
    - When y is FLOAT16, deqScale supports the following data types: UINT64, INT64, FLOAT32.
    - y is BFLOAT16, and deqScale supports the following data type: BFLOAT16.
    - deqScale1 and deqScale2 must have the same data type.
  - In the quantization scenario, the per-channel mode parameter types for scale are: x is INT8, weight is INT8, bias is INT32, scale is FLOAT32, offset is FLOAT32. The types of the remaining parameters fall into two cases depending on y:
    - When y is FLOAT16, deqScale supports the following data types: UINT64, INT64.
    - y is BFLOAT16, and deqScale supports the following data type: BFLOAT16.
    - deqScale1 and deqScale2 must have the same data type.
  - The pseudo-quantization scenario supports two different parameter types:
    - y is FLOAT16, x is FLOAT16, bias is FLOAT16, antiquantScale is FLOAT16, antiquantOffset is FLOAT16, and weight supports data types INT8 and INT4.
    - y is BFLOAT16, x is BFLOAT16, bias is FLOAT32, antiquantScale is BFLOAT16, antiquantOffset is BFLOAT16, and weight supports data types INT8 and INT4.
  - When the data type of weight1/weight2 is INT4, the last dimension of its shape must be an even number.
  - In the pseudo-quantization scenario, under per-group mode, the group count G in antiquantScale1 and antiquantOffset1 must be divisible by K1, and the group count G in antiquantScale2 and antiquantOffset2 must be divisible by K2.
  - In the BFLOAT16 non-quantization scenario, the innerPrecise parameter can only be set to 0. In the FLOAT16 non-quantization scenario, it can be set to 0 or 1. In the quantization or pseudo-quantization scenario, both 0 and 1 can be configured, but the configuration does not take effect.
  - When tokensIndexFlag is true and there are experts (expertTokens is not empty), the values in expertTokens must satisfy the following condition: if i and j are both valid array indices in expertTokens and j is greater than i, then the value of the j-th element in expertTokens must be greater than or equal to the value of the i-th element in expertTokens.

<!-- end id9 -->
<!-- npu="310p" id10 -->
- <term>Atlas inference products</term>:
  - Only the without-expert scenario is supported.
  - N1=K2 must be satisfied.
<!-- end id10 -->

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_ffn_v2.h"
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
  // Boilerplate: resource initialization.
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
  // Call aclrtMalloc to allocate device-side memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy data from the host side to device-side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize the device/stream. For details, see the public AscendCL API list.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as needed.
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs. Customize the construction based on the API.
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
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT16, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the weight1 aclTensor.
  ret = CreateAclTensor(weight1HostData, weight1Shape, &weight1DeviceAddr, aclDataType::ACL_FLOAT16, &weight1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the weight2 aclTensor.
  ret = CreateAclTensor(weight2HostData, weight2Shape, &weight2DeviceAddr, aclDataType::ACL_FLOAT16, &weight2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Example of calling the aclnnFFNV2 API
  LOG_PRINT("test aclnnFFNV2\n");

  // 3. Call the CANN operator library API, and change it to the specific API name.
  // Call the first-phase API of aclnnFFNV2.
  ret = aclnnFFNV2GetWorkspaceSize(self, weight1, weight2, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,    "relu", 1, false, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFFNV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnFFNV2.
  ret = aclnnFFNV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFFNV2 failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize and wait for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the value of the output, copy the result in the device memory to the host, and modify it according to the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<op::fp16_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    std::cout << "index: " << i << ": " << static_cast<float>(resultData[i]) << std::endl;
  }

  // 6. Release aclTensor, and modify it according to the specific API definition.
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
