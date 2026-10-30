# aclnnFFN

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T11:13:01.199Z -->

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
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: This FFN operator provides the computation capabilities of MoeFFN and FFN. When there is no expert group (expertTokens is empty), it is FFN; when there is an expert group, it is MoeFFN. Both are collectively referred to as FFN and belong to the MoE structure. MoE (Mixture-of-Experts) is a technique for training models with trillions of parameters. MoE decomposes the predictive modeling task into several subtasks, trains an expert model on each subtask, and develops a gating model that assigns one or more experts based on the input data, ultimately combining the computation results of multiple experts as the prediction result. The Mixture-of-Experts model assigns input data to the most relevant one or more experts and combines the computation results of all involved experts to determine the final result.
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

**NOTE:**
Whether FFN provides performance gains without experts or with a single expert depends on actual test results. You can try this fused operator if the vector time of the small operators corresponding to the FFN structure in the entire network exceeds 30 us and accounts for more than 10% of the FFN structure time. Do not use the operator if actual tests show performance degradation.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnFFNGetWorkspaceSize" first to obtain the workspace size required for computation and the executor containing the operator computation process. Then call "aclnnFFN" to perform the computation.

```Cpp
aclnnStatus aclnnFFNGetWorkspaceSize(
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
  const aclTensor*   y,
  uint64_t*          workspaceSize,
  aclOpExecutor**    executor)
```

```Cpp
aclnnStatus aclnnFFN(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnFFNGetWorkspaceSize

**NOTE:**
M indicates the number of tokens, corresponding to BS in the transformer (B: Batch, indicating the batch size of input samples; S: Seq-Length, indicating the sequence length of input samples);
K1 indicates the number of input channels of the first matmul, corresponding to H in the transformer (Head-Size, indicating the hidden layer size);
N1 indicates the number of output channels of the first matmul; K2 indicates the number of input channels of the second matmul;
N2 indicates the number of output channels of the second matmul, corresponding to H in the transformer;
E indicates the number of experts in the scenario with experts.
G indicates the number of groups of antiquantOffset and antiquantScale in the pseudo-quantization per-group scenario.

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1650px">
  <colgroup>
  <col style="width: 200px"> <!-- Parameter -->
  <col style="width: 120px"> <!-- Input/Output -->
  <col style="width: 280px">  <!-- Description -->
  <col style="width: 300px">  <!-- Instruction -->
  <col style="width: 250px">  <!-- Data Type -->
  <col style="width: 120px">  <!-- Data Format -->
  <col style="width: 240px"> <!-- Dimension (shape) -->
  <col style="width: 140px">  <!-- Non-contiguous Tensor -->
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
      <td>Computation input, corresponding to x in the formula.</td>
      <td>        <ul>
          <li>Empty tensors are not supported.</li>
          <li>No Optional suffix; must not pass a null pointer.</li>
        </ul>
      </td>
      <td>FLOAT16, BFLOAT16, INT8</td>
      <td>ND</td>
      <td>2-8 dimensions [M, K1]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight1 (aclTensor*)</td>
      <td>Input</td>
      <td>Expert weight data, W1 in the formula.</td>
      <td>        <ul>
          <li>Empty tensors are not supported.</li>
          <li>No Optional suffix; the pointer must not be null.</li>
        </ul>
      </td>
      <td>FLOAT16, BFLOAT16, INT8, INT4</td>
      <td>ND</td>
      <td>with experts [E,K1,N1]<br>without experts [K1,N1]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight2 (aclTensor*)</td>
      <td>Input</td>
      <td>Expert weight data, W2 in the formula.</td>
      <td>        <ul>
          <li>Empty tensors are not supported.</li>
          <li>No Optional suffix; the pointer must not be null.</li>
        </ul>
      </td>
      <td>FLOAT16, BFLOAT16, INT8, INT4</td>
      <td>ND</td>
      <td>With experts [E,K2,N2]<br>Without experts [K2,N2]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>expertTokens (aclIntArray*)</td>
      <td>Optional input</td>
      <td>Token count of each expert.</td>
      <td>        <ul>
          <li>Supports passing a null pointer (empty Tensor).</li>
          <li>No Optional suffix; no token count constraint when passing a null pointer.</li>
          <li>Maximum length is 256 when not null.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>ND</td>
      <td>1D, maximum length 256.</td>
      <td>-</td>
    </tr>
    <tr>
      <td>bias1 (aclTensor*)</td>
      <td>Optional input</td>
      <td>Weight correction value, b1 in the formula.</td>
      <td>        <ul>
          <li>Supports empty tensors. A null pointer can be passed.</li>
          <li>Without the Optional suffix, no bias constraint applies when a null pointer is passed.</li>
        </ul>
      </td>
      <td>FLOAT16, FLOAT32, INT32</td>
      <td>ND</td>
      <td>with experts[E,N1]<br>without experts[N1]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>bias2 (aclTensor*)</td>
      <td>Optional input</td>
      <td>Weight correction value, which is b2 in the formula.</td>
      <td>        <ul>
          <li>Supports null tensor. Null pointer can be passed.</li>
          <li>No Optional suffix. No bias constraint when null pointer is passed.</li>
        </ul>
      </td>
      <td>FLOAT16, FLOAT32, INT32</td>
      <td>ND</td>
      <td>With experts [E, N2]<br>Without experts [N2]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale (aclTensor*)</td>
      <td>Optional input</td>
      <td>Quantization parameter, quantization scale factor.</td>
      <td>        <ul>
          <li>Null tensor supported. A nullable pointer can be passed.</li>
          <li>No Optional suffix. Passing a null pointer imposes no scale constraint.</li>
        </ul>
      </td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>per-tensor 1D, with experts [E], without experts [1]<br>per-channel with experts [E,N1], without experts [N1]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>offset (aclTensor*)</td>
      <td>Optional input</td>
      <td>Quantization parameter, quantization offset.</td>
      <td>        <ul>
          <li>Null tensor supported. A nullable pointer can be passed.</li>
          <li>No Optional suffix. Passing a null pointer imposes no offset constraint.</li>
        </ul>
      </td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1 dimension, with experts [E]<br>without experts [1]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>deqScale1 (aclTensor*)</td>
      <td>Optional input</td>
      <td>Quantization parameter, the dequantization scale coefficient of the first matmul.</td>
      <td>        <ul>
          <li>Empty tensor supported. A null pointer can be passed.</li>
          <li>No Optional suffix. No dequantization constraint applies when a null pointer is passed.</li>
        </ul>
      </td>
      <td>UINT64, INT64, FLOAT32, BFLOAT16</td>
      <td>ND</td>
      <td>with experts[E,N1]<br>without experts[N1]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>deqScale2 (aclTensor*)</td>
      <td>Optional input</td>
      <td>Quantization parameter, the dequantization scale of the second matmul.</td>
      <td>        <ul>
          <li>Empty Tensor supported, nullable pointer allowed.</li>
          <li>Without the Optional suffix, passing a null pointer imposes no dequantization constraints.</li>
        </ul>
      </td>
      <td>UINT64, INT64, FLOAT32, BFLOAT16</td>
      <td>ND</td>
      <td>with experts[E,N2]<br>without experts[N2]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>antiquantScale1(aclTensor*)</td>
      <td>Optional input</td>
      <td>Fake-quantization parameter. Scaling coefficient of the first matmul.</td>
      <td>        <ul>
          <li>Null tensor supported. Nullable pointer.</li>
          <li>No Optional suffix. Passing a null pointer means no fake quantization constraint.</li>
        </ul>
      </td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>per-channel with experts [E,N1], without experts [N1]<br>per-group with experts [E,G,N1], without experts [G,N1]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>antiquantScale2(aclTensor*)</td>
      <td>Optional input</td>
      <td>Fake-quantization parameter, scaling factor of the second matmul.</td>
      <td>        <ul>
          <li>Empty tensor supported. Nullable pointer.</li>
          <li>No Optional suffix. Passing a null pointer means no fake quantization constraint.</li>
        </ul>
      </td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>per-channel with experts [E,N2], without experts [N2]<br>per-group with experts [E,G,N2], without experts [G,N2]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>antiquantOffset1 (aclTensor*)</td>
      <td>Optional input</td>
      <td>Fake-quantization parameter, offset of the first matmul.</td>
      <td>        <ul>
          <li>Empty tensor supported. Nullable pointer.</li>
          <li>No Optional suffix. Passing a null pointer means no fake quantization constraint.</li>
        </ul>
      </td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>per-channel with experts [E,N1], without experts [N1]<br>per-group with experts [E,G,N1], without experts [G,N1]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>antiquantOffset2 (aclTensor*)</td>
      <td>Optional input</td>
      <td>Fake-quantization parameter. Offset of the second matmul.</td>
      <td>        <ul>
          <li>Null tensor supported, nullable pointer.</li>
          <li>Without the Optional suffix, passing a null pointer means no fake quantization constraints.</li>
        </ul>
      </td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>per-channel: [E,N2] with experts, [N2] without experts<br>per-group: [E,G,N2] with experts, [G,N2] without experts</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activation (char*)</td>
      <td>Input</td>
      <td>Activation function used, corresponding to activation in the formula.</td>
      <td>        <ul>
          <li>Empty tensors are not supported; the pointer must not be null.</li>
          <li>No Optional suffix; a valid value must be passed.</li>
          <li>Supported values: fastgelu/gelu/relu/silu/geglu/swiglu/reglu.</li>
        </ul>
      </td>
      <td>CHAR</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>innerPrecise (int64_t)</td>
      <td>Optional input</td>
      <td>High precision or high performance selection.</td>
      <td>        <ul>
          <li>No empty tensor concept. Default values can be passed.</li>
          <li>No Optional suffix. Valid only for FLOAT16.</li>
          <li>0=high precision (FLOAT32 computation), 1=high performance.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y (aclTensor*)</td>
      <td>Output</td>
      <td>Computed output, the output y in the formula.</td>
      <td>        <ul>
          <li>Empty tensors are not supported.</li>
          <li>No Optional suffix, must not be a null pointer.</li>
        </ul>
      </td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>Same dimensions as x</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be requested on the Device.</td>
      <td>        <ul>
          <li>No empty tensor concept, must not be a null pointer.</li>
          <li>No Optional suffix, returns a non-negative integer.</li>
        </ul>
      </td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor containing the operator computation process.</td>
      <td>        <ul>
          <li>No empty tensor concept; must not be null.</li>
          <li>No Optional suffix. Resources of the returned executor must be released.</li>
        </ul>
      </td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API validates input parameters. If the following error codes occur, the corresponding causes are:

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
      <td>If the parameter is a required input, output, or required attribute and is a null pointer, 161001 is returned.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data types and data formats of x, weight1, weight2, activation, expertTokens, bias1, bias2, and y are not within the supported range.</td>
    </tr>
  </tbody>
  </table>

## aclnnFFN

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
      <td>Workspace size allocated on the Device, obtained by the first-phase API aclnnFFNGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnFFN defaults to a non-deterministic implementation. You can enable deterministic computation through aclrtSetSysParamOpt.
- With expert, the total number of expert data must be consistent with M of x.
- When the activation layer is geglu/swiglu/reglu, only the FLOAT16 high-performance scenario without expert grouping is supported (FLOAT16 scenario refers to the scenario where all required parameters of the aclTensor type are of the FLOAT16 data type), and N1=2\*K2.
- When the activation layer is gelu/fastgelu/relu/silu, the FLOAT16 high-precision and high-performance scenarios, BFLOAT16 scenario, quantization scenario, and pseudo-quantization scenario with or without expert grouping are supported, and N1=K2.
- In all scenarios, K1 = N2, K1 < 65536, K2 < 65536, and the M-axis after 32-byte alignment must be less than the maximum value of INT32.
- Quantization parameters and pseudo-quantization parameters cannot be input in the non-quantization scenario, pseudo-quantization parameters cannot be input in the quantization scenario, and quantization parameters cannot be input in the pseudo-quantization scenario.
- Quantization scenario parameter types: x is INT8, weight is INT8, bias is INT32, scale is FLOAT32, offset is FLOAT32. The types of other parameters fall into two cases depending on y:
  - When y is FLOAT16, the supported data types of deqScale are: UINT64, INT64, FLOAT32.
  - y is BFLOAT16, and the supported data type of deqScale is BFLOAT16.
  - deqScale1 and deqScale2 must have the same data type.
- In the quantization scenario, the per-channel mode parameter types for scale are: x is INT8, weight is INT8, bias is INT32, scale is FLOAT32, offset is FLOAT32. The types of the remaining parameters fall into two cases depending on y:
  - y is FLOAT16, and the supported data type of deqScale is UINT64 or INT64.
  - y is BFLOAT16, and the supported data type of deqScale is BFLOAT16.
  - deqScale1 and deqScale2 must have the same data type.
- The pseudo-quantization scenario supports two different parameter types:
  - y is FLOAT16, x is FLOAT16, bias is FLOAT16, antiquantScale is FLOAT16, antiquantOffset is FLOAT16, and weight supports data types INT8 and INT4.
  - y is BFLOAT16, x is BFLOAT16, bias is FLOAT32, antiquantScale is BFLOAT16, antiquantOffset is BFLOAT16, and weight supports data types INT8 and INT4.
- When the data type of weight1/weight2 is INT4, the last dimension of its shape must be an even number.
- In pseudo-quantization scenario, under per-group, K1 in antiQuantScale1 and antiQuantOffset1 must be divisible by the number of groups G, and K2 in antiQuantScale2 and antiQuantOffset2 must be divisible by the number of groups G.
- In pseudo-quantization scenario, under per-group, currently only the scenario where weight is of INT4 data type is supported.
- In the BFLOAT16 non-quantization scenario, the innerPrecise parameter can only be set to 0. In the FLOAT16 non-quantization scenario, it can be set to 0 or 1. In the quantization or pseudo-quantization scenario, both 0 and 1 can be configured, but the configuration does not take effect.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_ffn.h"
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
  // Boilerplate code, resource initialization.
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
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy data from the host to the device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of the contiguous tensor.
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
  // 1. (boilerplate) Initialize the device/stream. For details, see the AscendCL API list.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as needed.
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output as required by the API.
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

  // Example of calling the aclnnFFN API.
  LOG_PRINT("test aclnnFFN\n");

  // 3. Call the CANN operator library API. Replace Api with the actual API name.
  // Call the first-phase API of aclnnFFN.
  ret = aclnnFFNGetWorkspaceSize(self, weight1, weight2, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, "relu", 1, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFFNGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnFFN.
  ret = aclnnFFN(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFFN failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from device memory to host memory. Modify it according to the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<op::fp16_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    std::cout << "index: " << i << ": " << static_cast<float>(resultData[i]) << std::endl;
  }

  // 6. Release the aclTensor. Modify it according to the specific API definition.
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
