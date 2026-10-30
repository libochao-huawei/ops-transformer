# aclnnMoeFinalizeRoutingV2Grad

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:04:44.503Z -->

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

- Description: Backward propagation of aclnnMoeFinalizeRoutingV2.
- Formula:

  $$
  i : 0 \sim R * K - 1
  $$

  $$
  j : 0 \sim H
  $$

  (1) scalesOptional is a null pointer:

  $$
  gradExpandedXOut[expandedRowIdx[i]][j] = gradY[i / K][j]
  $$

  (2) scalesOptional is a non-null pointer and biasOptional is a null pointer:

  $$
  gradExpandedXOut[expandedRowIdx[i]][j] = gradY[i / K][j] * scalesOptional[i / K][i \% K]
  $$

  $$
  gradScalesOut[i] = sum(expandedXOptional[expandedRowIdx[i]][j] * gradY[i / K][j])
  $$

  (3) scalesOptional is a non-null pointer and biasOptional is a non-null pointer:

  $$
  gradExpandedXOut[expandedRowIdx[i]][j] = gradY[i / K][j] * scalesOptional[i / K][i \% K]
  $$

  $$
  gradScalesOut[i] = sum((expandedXOptional[expandedRowIdx[i]][j] + biasOptional[expertIdxOptional[i]][j]) * gradY[i / K][j])
  $$

  Where R represents batch * sequence, H represents hidden, and K represents topK.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnMoeFinalizeRoutingV2GradGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor that contains the operator computation flow. Then call the "aclnnMoeFinalizeRoutingV2Grad" API to execute the computation.

```c++
aclnnStatus aclnnMoeFinalizeRoutingV2GradGetWorkspaceSize(
  const aclTensor *gradY,
  const aclTensor *expandedRowIdx,
  const aclTensor *expandedXOptional,
  const aclTensor *scalesOptional,
  const aclTensor *expertIdxOptional,
  const aclTensor *biasOptional,
  int64_t          dropPadMode,
  int64_t          activeNum,
  int64_t          expertNum,
  int64_t          expertCapacity,
  const aclTensor *gradExpandedXOut,
  const aclTensor *gradScalesOut,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor);
```

```c++
aclnnStatus aclnnMoeFinalizeRoutingV2Grad(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream);
```

## aclnnMoeFinalizeRoutingV2GradGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1656px"><colgroup>
  <col style="width: 226px">
  <col style="width: 124px">
  <col style="width: 288px">
  <col style="width: 288px">
  <col style="width: 193px">
  <col style="width: 193px">
  <col style="width: 193px">
  <col style="width: 151px">
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
      <td>gradY (aclTensor)</td>
      <td>Input</td>
      <td>Indicates the derivative of the forward output y of MoeFinalizeRoutingV2.</td>
      <td>-</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>(R, H)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>expandedRowIdx (aclTensor)</td>
      <td>Input</td>
      <td>Indicates the index of tokens sorted by expert order.</td>
      <td>If scalesOptional is a null pointer, K must be 1. If dropPadMode is 0, the value range is [0, R * K - 1] and no duplicate indices are allowed; if dropPadMode is 1, the value range is [-1, expertNum * expertCapacity - 1] and no duplicate indices are allowed except -1.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(R * K)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>expandedXOptional (aclTensor)</td>
      <td>Input</td>
      <td>Indicates the features expanded based on expertIdx.</td>
      <td>If scalesOptional is a non-null pointer, this parameter cannot be a null pointer either.</td>
      <td>Consistent with gradY.</td>
      <td>ND</td>
      <td>When dropPadMode is 0, if activeNum is greater than 0 and less than R * K, the shape is (activeNum, H); otherwise, (R * K, H).<br>When dropPadMode is 1: (expertNum, expertCapacity, H)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scalesOptional (aclTensor)</td>
      <td>Input</td>
      <td>Indicates the scaling applied to features.</td>
      <td>-</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>(R, K)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>expertIdxOptional (aclTensor)</td>
      <td>Input</td>
      <td>Indicates the index of the expert that processes each feature.</td>
      <td>If biasOptional is a non-null pointer, expertIdxOptional cannot be a null pointer. Value range: [0, E - 1], E &gt;= 1. Duplicate indexes are allowed. E indicates the number of experts.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(R, K)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>biasOptional (aclTensor)</td>
      <td>Input</td>
      <td>Indicates the offset applied to features.</td>
      <td>-</td>
      <td>Consistent with gradY.</td>
      <td>ND</td>
      <td>(E, H)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dropPadMode (int64_t)</td>
      <td>Input</td>
      <td>Indicates different use cases.</td>
      <td>The value can be 0 or 1. 0 indicates the dropless use case, where expertNum and expertCapacity are not verified; 1 indicates the drop use case, where expertNum and expertCapacity must be verified, and values processed by each expert that exceed or fall short of expertCapacity are handled accordingly.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activeNum (int64_t)</td>
      <td>Input</td>
      <td>Indicates the maximum number of output rows of gradExpandedXOut.</td>
      <td>When dropPadMode is 0, this parameter takes effect only when activeNum is greater than 0 and less than R * K; when dropPadMode is 1, this parameter does not take effect.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertNum (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of experts.</td>
      <td>If dropPadMode is 0, this parameter does not take effect. If dropPadMode is 1 and biasOptional is a non-null pointer, expertNum must be equal to E. If biasOptional is a null pointer, expertNum must be greater than 0. Otherwise, an error is reported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertCapacity (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of rows each expert can process.</td>
      <td>When dropPadMode is 0, this parameter does not take effect. When dropPadMode is 1, expertCapacity must be greater than 0; otherwise, an error is reported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradExpandedXOut (aclTensor)</td>
      <td>Output</td>
      <td>Derivative of expandedX, which is the forward input of MoeFinalizeRoutingV2.</td>
      <td>-</td>
      <td>Consistent with gradY.</td>
      <td>ND</td>
      <td>Consistent with expandedXOptional.</td>
      <td>×</td>
    </tr>
    <tr>
      <td>gradScalesOut (aclTensor)</td>
      <td>Output</td>
      <td>Derivative of scales, which is the forward input of MoeFinalizeRoutingV2.</td>
      <td>This output is valid only when scalesOptional is not a null pointer.</td>
      <td>Consistent with scalesOptional.</td>
      <td>ND</td>
      <td>(R, K)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t)</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
      - The data type of scalesOptional must be the same as that of gradY.
  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT products</term>:
      - The data type of scalesOptional can be different from that of gradY.

  <!-- end id8 -->

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 330px">
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
    <td> ACLNN_ERR_PARAM_NULLPTR </td>
    <td> 161001 </td>
    <td>The tensors of required inputs and outputs are null pointers.</td>
    </tr>
    <tr>
    <td> ACLNN_ERR_PARAM_INVALID </td>
    <td> 161002 </td>
    <td>The data types and formats of the input and output are not within the supported range.</td>
    </tr>
    <tr>
    <td> ACLNN_ERR_INNER_NULLPTR </td>
    <td> 561103 </td>
    <td>The shapes and values of the input and output do not meet the requirements described in Parameters.</td>
    </tr>
  </tbody></table>

## aclnnMoeFinalizeRoutingV2Grad

- **Parameters**
  <table style="undefined;table-layout: fixed; width: 1148px"><colgroup>
  <col style="width: 170px">
  <col style="width: 134px">
  <col style="width: 844px">
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
      <td>Workspace size allocated on the device, obtained by calling the first-phase API aclnnMoeFinalizeRoutingV2GradGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnMoeFinalizeRoutingV2Grad defaults to a deterministic implementation.

## Example

The following is sample code for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_finalize_routing_v2_grad.h"
#include <cstdio>

#define CHECK_RET(cond, return_expr)                                           \
  do {                                                                         \
    if (!(cond)) {                                                             \
      return_expr;                                                             \
    }                                                                          \
  } while (0)

#define LOG_PRINT(message, ...)                                                \
  do {                                                                         \
    printf(message, ##__VA_ARGS__);                                            \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                         size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream *stream) {
  // Boilerplate, resource initialization.
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host data to the device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of the contiguous tensors.
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
  // 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
  // Set deviceId to the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Customize the construction based on the API.
  std::vector<int64_t> gradYShape = {2, 2};
  std::vector<int64_t> expandedRowIdxShape = {4};
  std::vector<int64_t> expandedXShape = {4, 2};
  std::vector<int64_t> scalesShape = {2, 2};
  std::vector<int64_t> expertIdxShape = {2, 2};
  std::vector<int64_t> biasShape = {2, 2};
  std::vector<int64_t> gradExpandedXShape = {4, 2};
  std::vector<int64_t> gradScalesShape = {2, 2};
  void* gradYDeviceAddr = nullptr;
  void* expandedRowIdxDeviceAddr = nullptr;
  void* expandedXDeviceAddr = nullptr;
  void* scalesDeviceAddr = nullptr;
  void* expertIdxDeviceAddr = nullptr;
  void* biasDeviceAddr = nullptr;
  void* gradExpandedXDeviceAddr = nullptr;
  void* gradScalesDeviceAddr = nullptr;

  aclTensor* gradY = nullptr;
  aclTensor* expandedRowIdx = nullptr;
  aclTensor* expandedX = nullptr;
  aclTensor* scales = nullptr;
  aclTensor* expertIdx = nullptr;
  aclTensor* bias = nullptr;
  int64_t dropPadMode = 0;
  int64_t activeNum = 0;
  int64_t expertNum = 0;
  int64_t expertCapacity = 0;
  aclTensor* gradExpandedX = nullptr;
  aclTensor* gradScales = nullptr;

  std::vector<float> gradYHostData = {0.3816, 0.3939, 0.8474, 0.1652};
  std::vector<int> expandedRowIdxHostData = {1, 3, 0, 2};
  std::vector<float> expandedXHostData = {0.6049, 0.3315, 0.4954, 0.3284, 0.7060, 0.4359, 0.6514, 0.9476};
  std::vector<float> scalesHostData = {0.4708, 0.0656, 0.9652, 0.9512};
  std::vector<int> expertIdxHostData = {0, 1, 0, 1};
  std::vector<float> biasHostData = {0.6452, 0.1981, 0.4159, 0.9575};
  std::vector<float> gradExpandedXHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> gradScalesHostData = {0, 0, 0, 0};

  ret = CreateAclTensor(gradYHostData, gradYShape, &gradYDeviceAddr, aclDataType::ACL_FLOAT, &gradY);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(expandedRowIdxHostData, expandedRowIdxShape, &expandedRowIdxDeviceAddr, aclDataType::ACL_INT32,
                        &expandedRowIdx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(expandedXHostData, expandedXShape, &expandedXDeviceAddr, aclDataType::ACL_FLOAT, &expandedX);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(scalesHostData, scalesShape, &scalesDeviceAddr, aclDataType::ACL_FLOAT, &scales);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(expertIdxHostData, expertIdxShape, &expertIdxDeviceAddr, aclDataType::ACL_INT32, &expertIdx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gradExpandedXHostData, gradExpandedXShape, &gradExpandedXDeviceAddr, aclDataType::ACL_FLOAT,
                        &gradExpandedX);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gradScalesHostData, gradScalesShape, &gradScalesDeviceAddr, aclDataType::ACL_FLOAT, &gradScales);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace the API name with the actual one.
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;

  // Call the first-phase API of aclnnMoeFinalizeRoutingV2Grad.
  ret = aclnnMoeFinalizeRoutingV2GradGetWorkspaceSize(gradY, expandedRowIdx, expandedX, scales, expertIdx, bias,
                                                      dropPadMode, activeNum, expertNum, expertCapacity, gradExpandedX,gradScales, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeFinalizeRoutingV2GradGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);

  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnMoeFinalizeRoutingV2Grad
  ret = aclnnMoeFinalizeRoutingV2Grad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeFinalizeRoutingV2Grad failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronously wait for the task execution to complete
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output values, copy the results from device memory to host memory, and modify the code based on the specific API definition
  LOG_PRINT("gradExpandedX result is: \n");
  PrintOutResult(gradExpandedXShape, &gradExpandedXDeviceAddr);
  LOG_PRINT("gradScales result is: \n");
  PrintOutResult(gradScalesShape, &gradScalesDeviceAddr);

  // 6. Release aclTensor and aclScalar, and modify the code based on the specific API definition
  aclDestroyTensor(gradY);
  aclDestroyTensor(expandedRowIdx);
  aclDestroyTensor(expandedX);
  aclDestroyTensor(scales);
  aclDestroyTensor(expertIdx);
  aclDestroyTensor(bias);
  aclDestroyTensor(gradExpandedX);
  aclDestroyTensor(gradScales);

  // 7. Release device resources
  aclrtFree(gradYDeviceAddr);
  aclrtFree(expandedRowIdxDeviceAddr);
  aclrtFree(expandedXDeviceAddr);
  aclrtFree(scalesDeviceAddr);
  aclrtFree(expertIdxDeviceAddr);
  aclrtFree(biasDeviceAddr);
  aclrtFree(gradExpandedXDeviceAddr);
  aclrtFree(gradScalesDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
