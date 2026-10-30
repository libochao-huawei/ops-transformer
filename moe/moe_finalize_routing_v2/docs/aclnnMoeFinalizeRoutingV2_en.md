# aclnnMoeFinalizeRoutingV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:00:54.446Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
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

- **Description**: In MoE (Mixture of Experts) computation, performs the final processing to merge the outputs of the MoE (Mixture of Experts) FFN (Feed-Forward Network).
- **Formula**:

  $$
  expertid=expertIdx[i,k]
  $$

  $$
  out(i,j)=x1_{i,j}+x2_{i,j}+\sum_{k=0}^{K}(scales_{i,k}*(expandedX_{expandedRowIdx_{i+k*num\_rows},j}+bias_{expertid,j}))
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnMoeFinalizeRoutingV2GetWorkspaceSize" first to obtain the workspace size required for computation and the executor containing the operator computation process, and then call "aclnnMoeFinalizeRoutingV2" to execute the computation.

```c++
aclnnStatus aclnnMoeFinalizeRoutingV2GetWorkspaceSize(
    const aclTensor *expandedX,
    const aclTensor *expandedRowIdx,
    const aclTensor *x1Optional,
    const aclTensor *x2Optional,
    const aclTensor *biasOptional,
    const aclTensor *scalesOptional,
    const aclTensor *expertIdxOptional,
    int64_t          dropPadMode,
    const aclTensor *out,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```c++
aclnnStatus aclnnMoeFinalizeRoutingV2(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnMoeFinalizeRoutingV2GetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1522px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 315px">
  <col style="width: 231px">
  <col style="width: 210px">
  <col style="width: 121px">
  <col style="width: 210px">
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
    <td>expandedX(aclTensor*)</td>
    <td>Input</td>
    <td>expandedX in the formula, the FFN output of MoE.</td>
    <td>-</td>
    <td>FLOAT16, BFLOAT16, FLOAT32</td>
    <td>ND</td>
    <td>drop less use case: (NUM_ROWS * K, H),<br>drop pad use case: (E, C, H).</td>
    <td>√</td>
  </tr>
  <tr>
    <td>expandedRowIdx(aclTensor*)</td>
    <td>Input</td>
    <td>expandedRowIdx in the formula.</td>
    <td>-</td>
    <td>INT32</td>
    <td>ND</td>
    <td>(NUM_ROWS * K)</td>
    <td>√</td>
  </tr>
  <tr>
    <td>x1Optional(aclTensor*)</td>
    <td>Input</td>
    <td>x1 in the formula, indicating the first shared expert.</td>
    <td>-</td>
    <td>Same as expandedX.</td>
    <td>ND</td>
    <td>Same as out.</td>
    <td>√</td>
  </tr>
  <tr>
    <td>x2Optional(aclTensor*)</td>
    <td>Input</td>
    <td>x2 in the formula. Indicates the second shared expert.</td>
    <td>-</td>
    <td>Same as expandedX.</td>
    <td>ND</td>
    <td>Same as out.</td>
    <td>√</td>
  </tr>
  <tr>
    <td>biasOptional(aclTensor*)</td>
    <td>Input</td>
    <td>bias in the formula, indicating the bias value.</td>
    <td>-</td>
    <td>Same as expandedX.</td>
    <td>ND</td>
    <td>(E, H)</td>
    <td>√</td>
  </tr>
  <tr>
    <td>scalesOptional(aclTensor*)</td>
    <td>Input</td>
    <td>scales in the formula.</td>
    <td>-</td>
    <td>FLOAT16, BFLOAT16, FLOAT32</td>
    <td>ND</td>
    <td>(NUM_ROWS, K)</td>
    <td>√</td>
  </tr>
  <tr>
    <td>expertIdxOptional(aclTensor*)</td>
    <td>Input</td>
    <td>expertIdx in the formula.</td>
    <td>The value range of values in the tensor is [0, E-1].</td>
    <td>INT32</td>
    <td>ND</td>
    <td>(NUM_ROWS, K)</td>
    <td>√</td>
  </tr>
  <tr>
    <td>dropPadMode(int64_t)</td>
    <td>Input</td>
    <td>Specifies whether the drop mode is supported and the arrangement of expandedRowIdx.</td>
    <td>The value range is [0, 3].</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>out(aclTensor*)</td>
    <td>Output</td>
    <td>Output in the formula.</td>
    <td>-</td>
    <td>Same as expandedX.</td>
    <td>ND</td>
    <td>(NUM_ROWS, H)</td>
    <td>×</td>
  </tr>
  <tr>
    <td>workspaceSize(uint64_t)</td>
    <td>Output</td>
    <td>Returns the workspace size to be requested on the Device.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>executor(aclOpExecutor**)</td>
    <td>Output</td>
    <td>Returns the op executor, which contains the computation process of the operator.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  </tbody></table>

  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
    - expandedX must be a 2D or 3D tensor. Supported data types are FLOAT16, BFLOAT16, and FLOAT32. Both drop-less and drop-pad scenarios are supported.
    - scalesOptional: In mixed-precision mode, when expandedX is BFLOAT16, scalesOptional can be FLOAT32; in non-mixed-precision mode, the data type must be the same as that of expandedX.
  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT products</term>:
    - expandedX must be a 2D or 3D tensor. Supported data types are FLOAT16, BFLOAT16, and FLOAT32. Both drop-less and drop-pad scenarios are supported.
    - The data type of scalesOptional can differ from that of expandedX.
  <!-- end id8 -->
  <!-- npu="310p" id9 -->
  - <term>Atlas inference products</term>:
    - expandedX must be a 2D tensor. Supported data types are FLOAT16 and FLOAT32. The last axis H must be 32-aligned.
    - x1Optional, x2Optional, biasOptional, and expertIdxOptional only support passing nullptr.
    - Only dropPadMode set to 2 is supported.
    - The data type of scalesOptional supports FLOAT16 and FLOAT32, and must be consistent with expandedX.

  <!-- end id9 -->

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 253px">
  <col style="width: 140px">
  <col style="width: 762px">
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
    <td>A required input, output, or attribute is a null pointer.</td>
    </tr>
    <tr>
    <td>ACLNN_ERR_PARAM_INVALID</td>
    <td>161002</td>
    <td>The data type, data format, or dimension of an input or output is not within the supported range.</td>
    </tr>
    <tr>
    <td rowspan="2">ACLNN_ERR_INNER_NULLPTR</td>
    <td rowspan="2">561103</td>
    <td>The shapes of multiple input tensors do not match.</td>
    </tr>
    <tr>
    <td>The shapes of the input attributes and input tensors do not match.</td>
    </tr>
  </tbody></table>

## aclnnMoeFinalizeRoutingV2

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1179px"> <colgroup>
  <col style="width: 169px">
  <col style="width: 130px">
  <col style="width: 880px">
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
    <td>Workspace memory address allocated on the device.</td>
  </tr>
  <tr>
    <td>workspaceSize</td>
    <td>Input</td>
    <td>Size of the workspace allocated on the device, obtained by the first-phase API <code>aclnnMoeFinalizeRoutingV2GetWorkspaceSize</code>.</td>
  </tr>
  <tr>
    <td>executor</td>
    <td>Input</td>
    <td>op executor, containing the operator computation process.</td>
  </tr>
  <tr>
    <td>stream</td>
    <td>Input</td>
    <td>Specifies the Stream for executing tasks.</td>
  </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

1. Deterministic computation:
    - aclnnMoeFinalizeRoutingV2 defaults to a deterministic implementation.

2. NUM\_ROWS: Indicates the number of rows;
    - K: Indicates that K experts are selected from the total E experts;
    - H: Indicates the hidden size, i.e., the sequence length of each token, which is the number of columns.
    - E: Indicates the expert num, i.e., the number of experts. E must be greater than or equal to K.
    - C: Indicates the expert capacity, i.e., the capacity threshold for the number of tokens an expert can process.

3. expandedRowIdx: When the dropPadMode parameter value is 0 or 2, the value range of the tensor is [0, NUM_ROWS * K - 1]; when the dropPadMode parameter value is 1 or 3, the value range of the tensor is [-1, E \* C - 1].

4. When the x1Optional parameter is not input, the x2Optional parameter cannot be input either.

5. When scalesOptional does not exist, K is 1.

6. When biasOptional exists, expertIdxOptional must also exist.

7. The values and meanings of dropPadMode are as follows:
    - 0: drop less use case. expandedRowIdx is in **column** arrangement (corresponding to the output format of [aclnnMoeInitRouting](../../moe_init_routing/docs/aclnnMoeInitRouting_en.md)).
    - 1: drop pad use case. expandedRowIdx is in **column** arrangement (corresponding to the output format of [aclnnMoeInitRouting](../../moe_init_routing/docs/aclnnMoeInitRouting_en.md)).
    - 2: drop less use case, expandedRowIdx uses **row** arrangement (corresponding to the output format of [aclnnMoeInitRoutingV2](../../moe_init_routing_v2/docs/aclnnMoeInitRoutingV2_en.md)).
    - 3: drop pad use case, expandedRowIdx uses **row** arrangement (corresponding to the output format of [aclnnMoeInitRoutingV2](../../moe_init_routing_v2/docs/aclnnMoeInitRoutingV2_en.md)).

8. Empty tensors and scalar tensors are not supported as inputs.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_finalize_routing_v2.h"
#include <iostream>
#include <vector>
#include <cstdio>


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
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}
int Init(int32_t deviceId, aclrtStream* stream) {
    // Boilerplate: resource initialization
    auto  ret = aclInit(nullptr);
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
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate device-side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host-side data to device-side memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    // Calculate the strides of the contiguous tensor.
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
  // 1. (Boilerplate) Device/stream initialization. For details, see the acl API manual.
  // Fill in deviceId according to the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle check as needed.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs. Perform custom construction according to the API.
  std::vector<int64_t> expandedXShape = {3 * 2, 4};
  std::vector<int64_t> x1Shape = {3, 4};
  std::vector<int64_t> x2OptionalShape = {3, 4};
  std::vector<int64_t> biasShape = {2, 4};
  std::vector<int64_t> scalesShape = {3, 2};
  std::vector<int64_t> expandedExpertIdxShape = {3, 2};
  std::vector<int64_t> expandedRowIdxShape = {3 * 2};
  std::vector<int64_t> outShape = {3, 4};
  void* expandedXAddr = nullptr;
  void* x1Addr = nullptr;
  void* x2OptionalAddr = nullptr;
  void* biasAddr = nullptr;
  void* scalesDeviceAddr = nullptr;
  void* expandedExpertIdxAddr = nullptr;
  void* expandedRowIdxAddr = nullptr;
  void* outDeviceAddr = nullptr;

  aclTensor* expandedX = nullptr;
  aclTensor* x1 = nullptr;
  aclTensor* x2Optional = nullptr;
  aclTensor* bias = nullptr;
  aclTensor* scales = nullptr;
  aclTensor* expandedExpertIdx = nullptr;
  aclTensor* expandedRowIdx = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> expandedXHostData = {0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1,
                                                     0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1};
  std::vector<float> x1HostData = {0.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 10.2, 11.2};
  std::vector<float> x2OptionalHostData = {0.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 10.2, 11.2};
  std::vector<float> biasHostData = {0.2, 0.4, 0.2, 0.4, 0.2, 0.4, 0.2, 0.4};
  std::vector<float> scalesHostData = {1.3, 1.6, 1.2, 1.8, 1.2, 2.3};
  std::vector<int32_t> expandedExpertIdxHostData = {0, 1, 0, 1, 0, 1};
  std::vector<int32_t> expandedRowIdxHostData = {2, 1, 4, 3, 0, 5};
  std::vector<float> outHostData(12, 0.0f);
  int64_t dropPadMode = 0;
  // Create expandedX aclTensor.
  ret = CreateAclTensor(expandedXHostData, expandedXShape, &expandedXAddr,
                        aclDataType::ACL_FLOAT, &expandedX);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create x1 aclTensor.
  ret = CreateAclTensor(x1HostData, x1Shape, &x1Addr, aclDataType::ACL_FLOAT, &x1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create x2Optional aclTensor.
  ret = CreateAclTensor(x2OptionalHostData, x2OptionalShape, &x2OptionalAddr, aclDataType::ACL_FLOAT, &x2Optional);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create bias aclTensor.
  ret = CreateAclTensor(biasHostData, biasShape, &biasAddr, aclDataType::ACL_FLOAT, &bias);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create scales aclTensor.
  ret = CreateAclTensor(scalesHostData, scalesShape, &scalesDeviceAddr, aclDataType::ACL_FLOAT, &scales);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create expandedExpertIdx aclTensor.
  ret = CreateAclTensor(expandedExpertIdxHostData, expandedExpertIdxShape, &expandedExpertIdxAddr,
                        aclDataType::ACL_INT32, &expandedExpertIdx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create expandedRowIdx aclTensor.
  ret = CreateAclTensor(expandedRowIdxHostData, expandedRowIdxShape, &expandedRowIdxAddr,
                        aclDataType::ACL_INT32, &expandedRowIdx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the Out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, and change it to the specific operator API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnMoeFinalizeRoutingV2.
  ret = aclnnMoeFinalizeRoutingV2GetWorkspaceSize(expandedX, expandedRowIdx, x1, x2Optional, bias, scales,
                                                  expandedExpertIdx, dropPadMode, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeFinalizeRoutingV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnMoeFinalizeRoutingV2.
  ret = aclnnMoeFinalizeRoutingV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeFinalizeRoutingV2 failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output values, and copy the results from the device side to the host side. Modify it based on the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0.0f);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                    outDeviceAddr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
      LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release the aclTensor and aclScalar. Modify it based on the specific API definition.
  aclDestroyTensor(expandedX);
  aclDestroyTensor(x1);
  aclDestroyTensor(x2Optional);
  aclDestroyTensor(bias);
  aclDestroyTensor(scales);
  aclDestroyTensor(expandedExpertIdx);
  aclDestroyTensor(expandedRowIdx);
  aclDestroyTensor(out);

  // 7. Release the device resources. Modify it based on the specific API definition.
  aclrtFree(expandedXAddr);
  aclrtFree(x1Addr);
  aclrtFree(x2OptionalAddr);
  aclrtFree(biasAddr);
  aclrtFree(scalesDeviceAddr);
  aclrtFree(expandedExpertIdxAddr);
  aclrtFree(expandedRowIdxAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
