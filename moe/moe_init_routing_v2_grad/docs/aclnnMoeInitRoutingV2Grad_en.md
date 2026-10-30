# aclnnMoeInitRoutingV2Grad

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:16:13.982Z -->

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
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- **Description**: Backward propagation of [aclnnMoeInitRoutingV2](../../moe_init_routing_v2/docs/aclnnMoeInitRoutingV2_en.md), which performs the weighted sum of tokens.
- **Formula**:

    $$
    gradX_i=\sum_{t=0}^{topK}gradExpandedX[expandedRowIdx[i * topK + t]]
    $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnMoeInitRoutingV2GradGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor containing the operator computation process, and then call the "aclnnMoeInitRoutingV2Grad" API to perform the computation.

```cpp
aclnnStatus aclnnMoeInitRoutingV2GradGetWorkspaceSize(
    const aclTensor  *gradExpandedX,
    const aclTensor  *expandedRowIdx,
    int64_t           topK,
    int64_t           dropPadMode,
    int64_t           activeNum,
    const aclTensor  *out,
    uint64_t         *workspaceSize,
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnMoeInitRoutingV2Grad(
    void            *workspace,
    uint64_t         workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream      stream)
```

## aclnnMoeInitRoutingV2GradGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 300px">
    <col style="width: 550px">
    <col style="width: 212px">
    <col style="width: 100px">
    <col style="width: 190px">
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
      <th>Non-Contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>gradExpandedX</td>
      <td>Input</td>
      <td>Indicates the target tensor after routing.</td>
      <td>Must be a 2D/3D tensor. The 2D shape is [B*S*K, H] in the Dropless scenario or [A, H] in the Active scenario. The 3D shape is [E, C, H] in the Drop/Pad scenario.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2 or 3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>expandedRowIdx</td>
      <td>Input</td>
      <td>Indicates the indices of tokens sorted by expert order.</td>
      <td><ul><li>Empty tensors are supported.</li><li>Must be a 1D tensor with shape [B*S*K].</li><li>Element values are in the range [-1, E*C) in Drop/Pad scenarios and [0, B*S*K) in other scenarios, and are unique except for -1.</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>topK</td>
      <td>Input</td>
      <td>Value of topK.</td>
      <td>Must be greater than 0, and the size of dimension 0 of expandedRowIdx must be divisible by topK.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dropPadMode</td>
      <td>Input</td>
      <td>Indicates whether it is a Drop/Pad scenario.</td>
      <td>The value is 0 or 1.<ul><li>0: Indicates the Dropless scenario.</li><li>1: Indicates the Drop/Pad scenario.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activeNum</td>
      <td>Input</td>
      <td>Indicates whether the scenario is an Active scenario.</td>
      <td>Value range: greater than or equal to 0. This parameter takes effect when dropPadMode is 0. 0 indicates a non-Active scenario, and a value greater than 0 indicates an Active scenario. In the Active scenario, the size of axis 0 of gradExpandedX must be equal to activeNum.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Indicates the backward output of Routing.</td>
      <td><ul><li>Empty tensors are supported.</li><li>Requires a 2D tensor with shape [B*S, H].</li><li>The data type must be the same as that of input gradExpandedX.</li></ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  Shape symbol description:
  - B: batch size
  - S: number of tokens.
  - H: hidden size, i.e., the sequence length of each token.
  - K: topK, i.e., the number of experts that process each token.
  - A: activeNum value.
  - E: number of experts.
  - C: expert capacity, which indicates the capacity threshold for the number of tokens an expert can process.

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:
  <table style="undefined;table-layout: fixed; width: 1180px">
    <colgroup>
      <col style="width: 250px">
      <col style="width: 130px">
      <col style="width: 800px">
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
        <td>The input or output tensor is a null pointer.</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td>The data type of the input or output is not within the supported range.</td>
      </tr>
      <tr>
        <td rowspan="8">ACLNN_ERR_INNER_NULLPTR</td>
        <td rowspan="8">561103</td>
        <td>The value of the dropPadMode attribute is not 0 or 1.</td>
      </tr>
      <tr>
        <td>topK is less than or equal to 0.</td>
      </tr>
      <tr>
        <td>activeNum is less than 0.</td>
      </tr>
      <tr>
        <td>gradExpandedX is not 2D/3D, or gradExpandedX is not 3D when dropPadMode is 1.</td>
      </tr>
      <tr>
        <td>When both dropPadMode and activeNum are 0, the size of dimension 0 of gradExpandedX is not equal to that of expandedRowIdx.</td>
      </tr>
      <tr>
        <td>When dropPadMode is 0 and activeNum is greater than 0, the size of dimension 0 of gradExpandedX is not equal to activeNum.</td>
      </tr>
      <tr>
        <td>The size of the trailing dimension of out is not equal to that of gradExpandedX.</td>
      </tr>
      <tr>
        <td>The size of dimension 0 of out is not equal to the size of dimension 0 of expandedRowIdx divided by topK.</td>
      </tr>
    </tbody>
  </table>

## aclnnMoeInitRoutingV2Grad

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1180px">
    <colgroup>
      <col style="width: 250px">
      <col style="width: 130px">
      <col style="width: 800px">
    </colgroup>
    <thead>
      <tr>
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>workspace</td>
        <td>Input</td>
        <td>Workspace memory address allocated on the device.</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Workspace size allocated on the device, obtained by calling the first-phase API <code>aclnnMoeInitRoutingV2GradGetWorkspaceSize</code>.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Op executor containing the operator computation process.</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Specifies the stream for executing tasks.</td>
      </tr>
    </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnMoeInitRoutingV2Grad defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_init_routing_v2_grad.h"
#include <iostream>
#include <vector>

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
    // Boilerplate for resource initialization.
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
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate device memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host data to device memory.
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
    // 1. Boilerplate: device/stream initialization. For details, see the ACL API list.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as required.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs according to the API definition.
    std::vector<int64_t> gradExpandedXShape = {4, 2};
    std::vector<int64_t> expandedRowIdxShape = {4};
    std::vector<int64_t> gradXShape = {2, 2};
    void* gradExpandedXDeviceAddr = nullptr;
    void* expandedRowIdxDeviceAddr = nullptr;
    void* gradXDeviceAddr = nullptr;
    aclTensor* gradExpandedX = nullptr;
    aclTensor* expandedRowIdx = nullptr;
    aclScalar* k = nullptr;
    aclScalar* dropPadMode = nullptr;
    aclScalar* activeNum = nullptr;
    aclTensor* expertIdx = nullptr;
    aclTensor* out = nullptr;
    std::vector<float> gradExpandedXHostData = {0.1, 0.1, 0.3, 0.3, 0.2, 0.2, 0.4, 0.4};
    std::vector<int32_t> expandedRowIdxHostData = {2, 0, 1, 3};
    std::vector<float> gradXOutHostData = {0, 0, 0, 0};
    int32_t kValue = 2;
    int32_t dropPadModeValue = 0;
    int32_t activeNumValue = 0;

    // Create the input aclTensor.
    ret = CreateAclTensor(gradExpandedXHostData, gradExpandedXShape, &gradExpandedXDeviceAddr, aclDataType::ACL_FLOAT, &gradExpandedX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandedRowIdxHostData, expandedRowIdxShape, &expandedRowIdxDeviceAddr, aclDataType::ACL_INT32, &expandedRowIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    k = aclCreateScalar(&kValue, aclDataType::ACL_INT32);
    CHECK_RET(k != nullptr, return ret);
    dropPadMode = aclCreateScalar(&dropPadModeValue, aclDataType::ACL_INT32);
    CHECK_RET(dropPadMode != nullptr, return ret);
    activeNum = aclCreateScalar(&activeNumValue, aclDataType::ACL_INT32);
    CHECK_RET(activeNum != nullptr, return ret);
    // Create the output aclTensor.
    ret = CreateAclTensor(gradXOutHostData, gradXShape, &gradXDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Change it to the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnMoeInitRoutingV2Grad
    ret = aclnnMoeInitRoutingV2GradGetWorkspaceSize(gradExpandedX, expandedRowIdx, kValue, dropPadModeValue, activeNumValue, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeInitRoutingV2GradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnMoeInitRoutingV2Grad
    ret = aclnnMoeInitRoutingV2Grad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeInitRoutingV2Grad failed. ERROR: %d\n", ret); return ret);

    // 4. Boilerplate. Synchronously wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output values and copy the results on the device to the host. Modify it according to the specific API definition.
    auto gradXSize = GetShapeSize(gradXShape);
    std::vector<float> gradXData(gradXSize, 0);
    ret = aclrtMemcpy(gradXData.data(), gradXData.size() * sizeof(gradXData[0]), gradXDeviceAddr, gradXSize * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < gradXSize; i++) {
        LOG_PRINT("gradXData[%ld] is: %f\n", i, gradXData[i]);
    }

    // 6. Release aclTensor and aclScalar. Modify it according to the specific API definition.
    aclDestroyTensor(gradExpandedX);
    aclDestroyTensor(expandedRowIdx);
    aclDestroyScalar(k);
    aclDestroyScalar(dropPadMode);
    aclDestroyScalar(activeNum);
    aclDestroyTensor(out);

    // 7. Release device resources. Modify it according to the specific API definition.
    aclrtFree(gradExpandedXDeviceAddr);
    aclrtFree(expandedRowIdxDeviceAddr);
    aclrtFree(gradXDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
