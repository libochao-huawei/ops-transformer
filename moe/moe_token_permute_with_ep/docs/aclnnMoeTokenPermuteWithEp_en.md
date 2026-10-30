# aclnnMoeTokenPermuteWithEp

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:31:07.927Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Not supported
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

- **Description**: The MoE permute operation. It broadcasts tokens and optional probs based on indices, sorts them, and slices them according to the range specified in rangeOptional.
- **Formula**:
  - When paddedMode is `false`, the formula is as follows. topK indicates the number of experts selected for each token. If Indices is 2-dimensional, topK equals the size of the last dimension of Indices. If Indices is 1-dimensional, topK is 1.

    $$
    sortedIndicesFirst=argSort(\text{flatten}(Indices))
    $$

    $$
    sortedIndicesOut=argSort(sortedIndicesFirst)
    $$

    When rangeOptional[0] <= sortedIndicesOut[i] < rangeOptional[1]:

    $$
    permuteTokensOut[sortedIndicesOut[i] - rangeOptional[0]]=tokens[i//topK]
    $$

  - When paddedMode is `true` (not supported yet):

    $$
    permuteTokensOut[i]=tokens[Indices[i]]
    $$

    $$
    sortedIndicesOut=Indices
    $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call the "aclnnMoeTokenPermuteWithEpGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor containing the operator computation flow, and then call the "aclnnMoeTokenPermuteWithEp" API to execute the computation.

```cpp
aclnnStatus aclnnMoeTokenPermuteWithEpGetWorkspaceSize(
    const aclTensor    *tokens,
    const aclTensor    *indices,
    const aclTensor    *probsOptional,
    const aclIntArray  *rangeOptional,
    int64_t             numOutTokens,
    bool                paddedMode,
    const aclTensor    *permuteTokensOut,
    const aclTensor    *sortedIndicesOut,
    const aclTensor    *permuteProbsOut,
    uint64_t           *workspaceSize,
    aclOpExecutor     **executor)
```

```cpp
aclnnStatus aclnnMoeTokenPermuteWithEp(
    void            *workspace,
    uint64_t         workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream      stream)
```

## aclnnMoeTokenPermuteWithEpGetWorkspaceSize

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
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>tokens</td>
      <td>Input</td>
      <td>Represents the input tokens in permute.</td>
      <td><ul><li>Empty tensors are supported.</li><li>Must be a 2D tensor with shape (num_tokens, hidden_size).</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Input</td>
      <td>Represents the expert indices of the input tokens.</td>
      <td><ul><li>Supports empty tensors.</li><li>Requires a 1D or 2D shape.</li><li>When paddedMode is false, indicates the topK expert indices for processing each input token, with a shape of (num_tokens, topK_num) or (num_tokens).</li><li>When paddedMode is true, indicates the token indices selected by each expert (not supported yet).</li><li>The number of elements must be less than 16777215, and the values must be greater than or equal to 0 and less than 16777215.</li></ul></td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>1 or 2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>probsOptional</td>
      <td>Input</td>
      <td>Indicates the expert probabilities corresponding to the input tokens.</td>
      <td><ul><li>Supports empty tensors.</li><li>Corresponds to the permuteProbsOut output. If an empty tensor is passed in, permuteProbsOut is not output.</li><li>The shape must be the same as that of indices.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>1 or 2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rangeOptional</td>
      <td>Input</td>
      <td>Valid range of EP split.</td>
      <td><ul><li>Empty is supported.</li><li>The size is 2.</li><li>If empty, probsOptional and permuteTokensOut are ignored, and the execution logic falls back to aclnnMoeTokenPermute.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numOutTokens</td>
      <td>Input</td>
      <td>Number of valid output tokens.</td>
      <td>Value range: any integer; 0 indicates that no tokens are deleted; a value greater than 0 slices the tokens sorted by expert according to numOutTokens and retains the first numOutTokens tokens; a value less than 0 is processed as a negative slice index.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>paddedMode</td>
      <td>Input</td>
      <td>Indicates whether padded mode is used.</td>
      <td>The value can be false or true.<ul><li>false: non-padded mode, where indices are sorted.</li><li>true: padded mode, where indices have been padded to represent the token indexes selected by each expert. In this case, indices are not sorted (not supported yet).</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>permuteTokensOut</td>
      <td>Output</td>
      <td>Tokens that are expanded and sorted according to indices.</td>
      <td><ul><li>Supports empty tensors.</li><li>Required to be a 2D tensor with shape (rangeOptional[1] - rangeOptional[0], hidden_size).</li><li>The data type is the same as that of tokens.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sortedIndicesOut</td>
      <td>Output</td>
      <td>The mapping between permuteTokensOut and tokens.</td>
      <td><ul><li>Supports empty tensors.</li><li>Required to be a 1D tensor with shape (num_tokens * topK_num).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>permuteProbsOut</td>
      <td>Output</td>
      <td>Probs expanded and sorted based on indices.</td>
      <td><ul><li>Empty tensors are supported.</li><li>Must be a 1D tensor with shape (rangeOptional[1] - rangeOptional[0]).</li><li>Data type is the same as tokens.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be requested on the Device.</td>
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

- **Return value**

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
        <td>The data type or data format of the input or output is out of the supported range.</td>
      </tr>
      <tr>
        <td rowspan="3">ACLNN_ERR_INNER_TILING_ERROR</td>
        <td rowspan="3">561002</td>
        <td>The shape dimension of tokens is not 2.</td>
      </tr>
      <tr>
        <td>The shape of indices is not 1D or 2D, or the first dimension of indices is not equal to the first dimension of tokens when paddedMode is false.</td>
      </tr>
      <tr>
        <td>paddedMode is true (not supported yet).</td>
      </tr>
    </tbody>
  </table>

## aclnnMoeTokenPermuteWithEp

- **Parameters:**

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
      <td>Workspace size allocated on the device, obtained by calling the first-phase API aclnnMoeTokenPermuteWithEpGetWorkspaceSize.</td>
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

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- deterministic computation:
  - aclnnMoeTokenPermuteWithEp defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_token_permute_with_ep.h"
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
    // Boilerplate code for resource initialization.
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}
template <typename T>
int CreateAclIntArray(const std::vector<T>& hostData, void** deviceAddr, aclIntArray** intArray) {
  auto size = GetShapeSize(hostData) * sizeof(T);
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy the data on the host to the memory on the device.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Call aclCreateIntArray to create an aclIntArray.
  *intArray = aclCreateIntArray(hostData.data(), hostData.size());
  return 0;
}
template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate device memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy data from the host to the device memory.
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
    // 1. Boilerplate for device/stream initialization. For details, see the list of ACL APIs.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as needed.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the input and output based on the API definition.
    std::vector<int64_t> xShape = {3, 4};
    std::vector<int64_t> idxShape = {3, 2};
    std::vector<int64_t> probsShape = {3, 2};
    std::vector<int64_t> expandedXOutShape = {4, 4};
    std::vector<int64_t> idxOutShape = {6};
    std::vector<int64_t> expandedProbsOutShape = {4};

    void* xDeviceAddr = nullptr;
    void* indicesDeviceAddr = nullptr;
    void* probsDeviceAddr = nullptr;
    void* expandedXOutDeviceAddr = nullptr;
    void* sortedIndicesOutDeviceAddr = nullptr;
    void* expandedProbsOutDeviceAddr = nullptr;
    void* rangeDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* indices = nullptr;
    aclTensor* probs = nullptr;
    aclIntArray* range = nullptr;
    int64_t numTokenOut = 6;
    bool padMode = false;

    aclTensor* expandedXOut = nullptr;
    aclTensor* sortedIndicesOut = nullptr;
    aclTensor* expandedProbsOut = nullptr;
    std::vector<float> xHostData = {0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3};
    std::vector<int> indicesHostData = {1, 2, 3, 1, 2, 3};
    std::vector<float> probsHostData = {0.5, 0.3, 0.4, 0.2, 0.5, 0.4};
    std::vector<float> expandedXOutHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int> sortedIndicesOutHostData = {0, 0, 0, 0, 0, 0};
    std::vector<float> expandedProbsOutHostData = {0, 0, 0, 0};
    std::vector<int64_t> rangeHostData = {1, 5};
    // Create the self aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(indicesHostData, idxShape, &indicesDeviceAddr, aclDataType::ACL_INT32, &indices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(probsHostData, probsShape, &probsDeviceAddr, aclDataType::ACL_BF16, &probs);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the out aclTensor.
    ret = CreateAclTensor(expandedXOutHostData, expandedXOutShape, &expandedXOutDeviceAddr, aclDataType::ACL_BF16, &expandedXOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sortedIndicesOutHostData, idxOutShape, &sortedIndicesOutDeviceAddr, aclDataType::ACL_INT32, &sortedIndicesOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandedProbsOutHostData, expandedProbsOutShape, &expandedProbsOutDeviceAddr, aclDataType::ACL_BF16, &expandedProbsOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the related attr.
    ret = CreateAclIntArray(rangeHostData, &rangeDeviceAddr, &range);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. Call the CANN operator library API. Change it to the specific API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnMoeTokenPermuteWithEp.
    ret = aclnnMoeTokenPermuteWithEpGetWorkspaceSize(x, indices, probs, range, numTokenOut, padMode, expandedXOut, sortedIndicesOut, expandedProbsOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeTokenPermuteWithEpGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnMoeTokenPermuteWithEp.
    ret = aclnnMoeTokenPermuteWithEp(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeTokenPermuteWithEp failed. ERROR: %d\n", ret); return ret);
    // 4. Boilerplate: synchronously wait for the task execution to finish.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from the device side to the host side. Change it according to the specific API definition.
    auto expandedXSize = GetShapeSize(expandedXOutShape);
    std::vector<float> expandedXData(expandedXSize, 0);
    ret = aclrtMemcpy(expandedXData.data(), expandedXData.size() * sizeof(expandedXData[0]), expandedXOutDeviceAddr, expandedXSize * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expandedXSize; i++) {
        LOG_PRINT("expandedXData[%ld] is: %f\n", i, expandedXData[i]);
    }
    auto sortedIndicesSize = GetShapeSize(idxOutShape);
    std::vector<int> sortedIndicesData(sortedIndicesSize, 0);
    ret = aclrtMemcpy(sortedIndicesData.data(), sortedIndicesData.size() * sizeof(sortedIndicesData[0]), sortedIndicesOutDeviceAddr, sortedIndicesSize * sizeof(int32_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < sortedIndicesSize; i++) {
        LOG_PRINT("sortedIndicesData[%ld] is: %d\n", i, sortedIndicesData[i]);
    }
    auto expandedProbsSize = GetShapeSize(expandedProbsOutShape);
    std::vector<float> expandedProbsData(expandedProbsSize, 0);
    ret = aclrtMemcpy(expandedProbsData.data(), expandedProbsData.size() * sizeof(expandedProbsData[0]), expandedProbsOutDeviceAddr, expandedProbsSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expandedProbsSize; i++) {
        LOG_PRINT("expandedProbsData[%ld] is: %f\n", i, expandedProbsData[i]);
    }
    // 6. Release aclTensor and aclScalar. Change it according to the specific API definition.
    aclDestroyTensor(x);
    aclDestroyTensor(indices);
    aclDestroyTensor(probs);
    aclDestroyTensor(expandedXOut);
    aclDestroyTensor(sortedIndicesOut);
    aclDestroyTensor(expandedProbsOut);

    // 7. Release device resources. Change it according to the specific API definition.
    aclrtFree(xDeviceAddr);
    aclrtFree(indicesDeviceAddr);
    aclrtFree(probsDeviceAddr);
    aclrtFree(expandedXOutDeviceAddr);
    aclrtFree(sortedIndicesOutDeviceAddr);
    aclrtFree(expandedProbsOutDeviceAddr);
    aclrtFree(rangeDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
