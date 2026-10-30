# aclnnMoeInitRouting

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:16:07.956Z -->

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

- **Description**: MoE routing computation. Performs routing processing based on the computation result of [aclnnMoeGatingTopKSoftmax](../../moe_gating_top_k_softmax/docs/aclnnMoeGatingTopKSoftmax_en.md).
- **Formula**:
  expertIdx with the input shape [NUM_ROWS, K] is flattened into one row for sorting, where NUM_ROWS is the number of input tokens and K is the number of experts selected by each token. sortedRowIdx is the sorted result of rowIdx.

  $$
  expandedExpertIdxOut,sortedRowIdx=keyValueSort(expertIdx,rowIdx)
  $$

  $$
  expandedRowIdxOut[sortedRowIdx[i]]=i
  $$

  $$
  expandedXOut[i]=x[sortedRowIdx[i]\%NUM_ROWS]
  $$

## Function Prototype

Each operator is split into a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnMoeInitRoutingGetWorkspaceSize" API first to obtain the workspace size required for the computation and the executor that contains the computation process of the operator. Then call the "aclnnMoeInitRouting" API to execute the computation.

```cpp
aclnnStatus aclnnMoeInitRoutingGetWorkspaceSize(
    const aclTensor  *x,
    const aclTensor  *rowIdx,
    const aclTensor  *expertIdx,
    int64_t           activeNum,
    const aclTensor  *expandedXOut,
    const aclTensor  *expandedRowIdxOut,
    const aclTensor  *expandedExpertIdxOut,
    uint64_t         *workspaceSize,
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnMoeInitRouting(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    aclrtStream       stream)
```

## aclnnMoeInitRoutingGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1543px"><colgroup>
    <col style="width: 200px">
    <col style="width: 120px">
    <col style="width: 300px">
    <col style="width: 232px">
    <col style="width: 219px">
    <col style="width: 121px">
    <col style="width: 200px">
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
      <td>x</td>
      <td>Input</td>
      <td>MOE input, which is the token feature input.</td>
      <td>-</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>shape (NUM_ROWS, H)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rowIdx</td>
      <td>Input</td>
      <td>Indicates the original row position corresponding to each position.</td>
      <td>rowIdx values start from 0 and increase along dimension 1.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>Shape must be the same as that of expertIdx.</td>
      <td>√</td>
    </tr>
    <tr>
      <td>expertIdx</td>
      <td>Input</td>
      <td>K processing experts corresponding to each row of features.</td>
      <td>-</td>
      <td>INT32</td>
      <td>ND</td>
      <td>The shape is (NUM_ROWS, K).</td>
      <td>√</td>
    </tr>
    <tr>
      <td>activeNum</td>
      <td>Input</td>
      <td>Indicates the total maximum number of rows to process, which must be greater than or equal to 0. Only this number of rows in expandedXOut is valid.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expandedXOut</td>
      <td>Output</td>
      <td>Features expanded based on expertIdx.</td>
      <td>-</td>
      <td>Same as x</td>
      <td>ND</td>
      <td>shape: (min(NUM_ROWS, activeNum) * K, H)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>expandedRowIdxOut</td>
      <td>Output</td>
      <td>Mapping between expandedX and x.</td>
      <td>-</td>
      <td>INT32</td>
      <td>ND</td>
      <td>Shape is (NUM_ROWS*K, ).</td>
      <td>x</td>
    </tr>
    <tr>
      <td>expandedExpertIdxOut</td>
      <td>Output</td>
      <td>Sorted result of expertIdx.</td>
      <td>-</td>
      <td>INT32</td>
      <td>ND</td>
      <td>shape is (NUM_ROWS*K, )</td>
      <td>x</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    </tbody>
  </table>

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
        <th>Return value</th>
        <th>Error code</th>
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
        <td>The data types of the input and output are not within the supported range.</td>
      </tr>
      <tr>
        <td rowspan="5">ACLNN_ERR_INNER_TILING_ERROR</td>
        <td rowspan="5">561002</td>
        <td>The dimension of the x shape is not 2.</td>
      </tr>
      <tr>
        <td>The shape of rowIdx is not 2-dimensional, or the shape of rowIdx is not equal to that of expertIdx.</td>
      </tr>
      <tr>
        <td>The value of activeNum is less than 0.</td>
      </tr>
      <tr>
        <td>The shape of expandedXOut is not (min(num_rows, activeNum) * K, H).</td>
      </tr>
      <tr>
        <td>The shape of expandedRowIdxOut is not equal to that of expandedExpertIdxOut, and is not (num_rows * K, ).</td>
      </tr>
    </tbody>
  </table>

## aclnnMoeInitRouting

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1179px">
  <colgroup>
  <col style="width: 169px">
  <col style="width: 130px">
  <col style="width: 880px">
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
        <td>Workspace memory address allocated on the Device.</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Workspace size allocated on the Device, obtained by calling the first-phase API <code>aclnnMoeInitRoutingGetWorkspaceSize</code>.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>op executor that contains the operator computation flow.</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream for executing tasks.</td>
      </tr>
    </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnMoeInitRouting defaults to a deterministic implementation.
- The values of elements in expertIdx must be greater than -2\*\*24 and not greater than 2\*\*24; otherwise, precision issues may occur.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_init_routing.h"
#include <iostream>
#include <cstdio>
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
    // Call aclrtMemcpy to copy data from the host to device memory.
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
    // 1. Boilerplate for device/stream initialization. For details, see the acl external API list.
    // Set deviceId based on the actual device in use.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as needed.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the input and output based on the API definition.
    std::vector<int64_t> xShape = {3, 4};
    std::vector<int64_t> idxShape = {3, 2};
    std::vector<int64_t> expandedXOutShape = {6, 4};
    std::vector<int64_t> idxOutShape = {6};
    void* xDeviceAddr = nullptr;
    void* rowIdxDeviceAddr = nullptr;
    void* expertIdxDeviceAddr = nullptr;
    void* expandedXOutDeviceAddr = nullptr;
    void* expandedRowIdxOutDeviceAddr = nullptr;
    void* expandedExpertIdxOutDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* rowIdx = nullptr;
    aclTensor* expertIdx = nullptr;
    int64_t activeNum = 3;
    aclTensor* expandedXOut = nullptr;
    aclTensor* expandedRowIdxOut = nullptr;
    aclTensor* expandedExpertIdxOut = nullptr;
    std::vector<float> xHostData = {0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3};
    std::vector<int> expertIdxHostData = {1, 2, 0, 1, 0, 2};
    std::vector<int> rowIdxHostData = {0, 3, 1, 4, 2, 5};
    std::vector<float> expandedXOutHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int> expandedRowIdxOutHostData = {0, 0, 0, 0, 0, 0};
    std::vector<int> expandedExpertIdxOutHostData = {0, 0, 0, 0, 0, 0};
    // Create the self aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(rowIdxHostData, idxShape, &rowIdxDeviceAddr, aclDataType::ACL_INT32, &rowIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertIdxHostData, idxShape, &expertIdxDeviceAddr, aclDataType::ACL_INT32, &expertIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the out aclTensor.
    ret = CreateAclTensor(expandedXOutHostData, expandedXOutShape, &expandedXOutDeviceAddr, aclDataType::ACL_FLOAT, &expandedXOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandedRowIdxOutHostData, idxOutShape, &expandedRowIdxOutDeviceAddr, aclDataType::ACL_INT32, &expandedRowIdxOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandedExpertIdxOutHostData, idxOutShape, &expandedExpertIdxOutDeviceAddr, aclDataType::ACL_INT32, &expandedExpertIdxOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. Call the CANN operator library API. Modify it to the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnMoeInitRouting.
    ret = aclnnMoeInitRoutingGetWorkspaceSize(x, rowIdx, expertIdx, activeNum, expandedXOut, expandedRowIdxOut, expandedExpertIdxOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeInitRoutingGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Request device memory based on the workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnMoeInitRouting.
    ret = aclnnMoeInitRouting(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeInitRouting failed. ERROR: %d\n", ret); return ret);
    // 4. Boilerplate: wait synchronously for the task to finish.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value, and copy the result from device memory to host memory. Modify it based on the actual API definition.
    auto expandedXSize = GetShapeSize(expandedXOutShape);
    std::vector<float> expandedXData(expandedXSize, 0);
    ret = aclrtMemcpy(expandedXData.data(), expandedXData.size() * sizeof(expandedXData[0]), expandedXOutDeviceAddr, expandedXSize * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expandedXSize; i++) {
        LOG_PRINT("expandedXData[%ld] is: %f\n", i, expandedXData[i]);
    }
    auto expandedRowIdxSize = GetShapeSize(idxOutShape);
    std::vector<int> expandedRowIdxData(expandedRowIdxSize, 0);
    ret = aclrtMemcpy(expandedRowIdxData.data(), expandedRowIdxData.size() * sizeof(expandedRowIdxData[0]), expandedRowIdxOutDeviceAddr, expandedRowIdxSize * sizeof(int32_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expandedRowIdxSize; i++) {
        LOG_PRINT("expandedRowIdxData[%ld] is: %d\n", i, expandedRowIdxData[i]);
    }
    auto expandedExpertIdxSize = GetShapeSize(idxOutShape);
    std::vector<int> expandedExpertIdxData(expandedExpertIdxSize, 0);
    ret = aclrtMemcpy(expandedExpertIdxData.data(), expandedExpertIdxData.size() * sizeof(expandedExpertIdxData[0]), expandedExpertIdxOutDeviceAddr, expandedExpertIdxSize * sizeof(int32_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expandedExpertIdxSize; i++) {
        LOG_PRINT("expandedExpertIdxData[%ld] is: %d\n", i, expandedExpertIdxData[i]);
    }
    // 6. Release the aclTensor and aclScalar. Modify it based on the actual API definition.
    aclDestroyTensor(x);
    aclDestroyTensor(rowIdx);
    aclDestroyTensor(expertIdx);
    aclDestroyTensor(expandedXOut);
    aclDestroyTensor(expandedRowIdxOut);
    aclDestroyTensor(expandedExpertIdxOut);

    // 7. Release the device resources. Modify it based on the actual API definition.
    aclrtFree(xDeviceAddr);
    aclrtFree(rowIdxDeviceAddr);
    aclrtFree(expertIdxDeviceAddr);
    aclrtFree(expandedXOutDeviceAddr);
    aclrtFree(expandedRowIdxOutDeviceAddr);
    aclrtFree(expandedExpertIdxOutDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
