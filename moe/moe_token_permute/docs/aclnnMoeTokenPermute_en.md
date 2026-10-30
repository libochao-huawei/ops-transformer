# aclnnMoeTokenPermute

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:24:15.433Z -->

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

- **Description**: MoE permute computation, which broadcasts and sorts tokens based on the indices.

- **Formula**:
    - When paddedMode is `false`, the formula is as follows. topK refers to the number of experts selected by a token. When Indices is 2-dimensional, topK equals the size of the last dimension of Indices. When Indices is 1-dimensional, topK equals 1:

      $$
      sortedIndicesFirst=argSort(\text{flatten}(Indices))
      $$

      $$
      sortedIndicesOut=argSort(sortedIndicesFirst)
      $$

      $$
      permuteTokensOut[sortedIndicesOut[i]]=tokens[i//topK]
      $$

    - When paddedMode is `true` (not supported yet):

      $$
      permuteTokensOut[i]=tokens[indices[i]]
      $$

      $$
      sortedIndicesOut=indices
      $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnMoeTokenPermuteGetWorkspaceSize" first to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call "aclnnMoeTokenPermute" to perform the computation.

```cpp
aclnnStatus aclnnMoeTokenPermuteGetWorkspaceSize(
    const aclTensor  *tokens,
    const aclTensor  *indices,
    int64_t           numOutTokens,
    bool              paddedMode,
    const aclTensor  *permuteTokensOut,
    const aclTensor  *sortedIndicesOut,
    uint64_t         *workspaceSize,
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnMoeTokenPermute(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    aclrtStream       stream)
```

## aclnnMoeTokenPermuteGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 265px">
    <col style="width: 120px">
    <col style="width: 223px">
    <col style="width: 391px">
    <col style="width: 181px">
    <col style="width: 111px">
    <col style="width: 126px">
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
      <td>Input token features.</td>
      <td><ul><li>Supports empty tensors.</li><li>Must be a tensor with at least 2 dimensions, and the first dimension size must be num_tokens.</li></ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT32, INT8</td>
      <td>ND</td>
      <td>≥2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Input</td>
      <td>Input indices.</td>
      <td><ul><li>Supports empty tensors.</li><li>Requires a 2D or 1D shape.</li><li>When paddedMode is false, indicates the topK processed expert indexes corresponding to each input token, with shape (num_tokens, topK) or (num_tokens).</li><li>When paddedMode is true, indicates the token indexes selected by each expert (not supported yet).</li><li>The number of elements is less than 16777215, and each value is greater than or equal to 0 and less than 16777215.</li></ul></td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>1 or 2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>numOutTokens</td>
      <td>Input</td>
      <td>Number of valid output tokens.</td>
      <td>The value can be any integer. 0 indicates no token is removed. A value greater than 0 slices the tokens sorted by expert according to numOutTokens and retains the first numOutTokens tokens. A value less than 0 is processed as a negative slicing index.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>paddedMode</td>
      <td>Input</td>
      <td>Indicates whether padded mode is used.</td>
      <td>The value can be false or true.<ul><li>false: non-padded mode, indices are sorted.</li><li>true: padded mode, indices are padded to represent the token indexes selected by each expert. In this case, indices are not sorted (not supported yet).</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>permuteTokensOut</td>
      <td>Output</td>
      <td>Tokens expanded and sorted according to indices.</td>
      <td><ul><li>Supports empty tensor.</li><li>Requires a tensor with two or more dimensions. The first dimension size is min(num_tokens * topK, numOutTokens).</li><li>The product of the sizes of all dimensions except the first is the same as that of tokens.</li><li>The data type is the same as tokens.</li></ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT32, INT8</td>
      <td>ND</td>
      <td>≥2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sortedIndicesOut</td>
      <td>Output</td>
      <td>Mapping relationship between permuteTokensOut and tokens.</td>
      <td><ul><li>Supports empty tensor.</li><li>Requires a 1D tensor with shape (num_tokens*topK).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the size of the workspace that needs to be requested on the Device.</td>
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
        <td>The data types of the input and output are not within the supported range.</td>
      </tr>
      <tr>
        <td rowspan="3">ACLNN_ERR_INNER_TILING_ERROR</td>
        <td rowspan="3">561002</td>
        <td>The shape dimension of tokens is less than 2.</td>
      </tr>
      <tr>
        <td>The shape of indices is not 1D or 2D, or the first dimension of indices is not equal to the first dimension of tokens when paddedMode is false.</td>
      </tr>
      <tr>
        <td>paddedMode is true.</td>
      </tr>
    </tbody>
  </table>

## aclnnMoeTokenPermute

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
        <td>Workspace memory address requested on the Device side.</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Workspace size requested on the Device side, obtained by the first-phase API <code>aclnnMoeTokenPermuteGetWorkspaceSize</code>.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>op executor, which contains the operator computation flow.</td>
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

- When this API is called, `indices` must be 1D or 2D. Otherwise, the API directly returns a parameter error.
- Deterministic computation:
  - aclnnMoeTokenPermute defaults to a deterministic implementation.
<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT products</term>: Only this product supports `tokens` and `permuteTokensOut` of the INT8 data type. They are processed in a non-quantized manner, and their data types must be the same. For INT8, `tokens` must be 2D. Tensors with 3 or more dimensions are not supported.
<!-- end id7 -->
<!-- npu="950" id8 -->
- <term>Ascend 950PR&950DT products</term>: When the data type of `tokens` is INT8, `indices` indicates the expert ID. The value range is `[0, 10240)`. The maximum value is `10239`, and `10240` is not supported.
<!-- end id8 -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_token_permute.h"
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
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
    // Call the aclCreateTensor API to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}
int main() {
    // 1. Boilerplate code for device/stream initialization. For details, see the list of ACL APIs.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as needed.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the input and output based on the API definition
    std::vector<int64_t> xShape = {3, 4};
    std::vector<int64_t> idxShape = {3, 2};
    std::vector<int64_t> expandedXOutShape = {6, 4};
    std::vector<int64_t> idxOutShape = {6};
    void* xDeviceAddr = nullptr;
    void* indicesDeviceAddr = nullptr;
    void* expandedXOutDeviceAddr = nullptr;
    void* sortedIndicesOutDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* indices = nullptr;
    int64_t numTokenOut = 0;
    bool padMode = false;

    aclTensor* expandedXOut = nullptr;
    aclTensor* sortedIndicesOut = nullptr;
    std::vector<float> xHostData = {0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3};
    std::vector<int> indicesHostData = {1, 2, 0, 1, 0, 2};
    std::vector<float> expandedXOutHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int> sortedIndicesOutHostData = {0, 0, 0, 0, 0, 0};
    // Create the self aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(indicesHostData, idxShape, &indicesDeviceAddr, aclDataType::ACL_INT32, &indices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the out aclTensor.
    ret = CreateAclTensor(expandedXOutHostData, expandedXOutShape, &expandedXOutDeviceAddr, aclDataType::ACL_FLOAT, &expandedXOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sortedIndicesOutHostData, idxOutShape, &sortedIndicesOutDeviceAddr, aclDataType::ACL_INT32, &sortedIndicesOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. Call the CANN operator library API. Change it to the specific API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnMoeTokenPermute.
    ret = aclnnMoeTokenPermuteGetWorkspaceSize(x, indices, numTokenOut, padMode, expandedXOut,
                                               sortedIndicesOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeTokenPermuteGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnMoeTokenPermute.
    ret = aclnnMoeTokenPermute(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeTokenPermute failed. ERROR: %d\n", ret); return ret);
    // 4. Boilerplate. Synchronously wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value. Copy the result from device memory to the host. Modify the code based on the specific API definition.
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
    // 6. Release the aclTensor and aclScalar. Modify the code based on the specific API definition.
    aclDestroyTensor(x);
    aclDestroyTensor(indices);
    aclDestroyTensor(expandedXOut);
    aclDestroyTensor(sortedIndicesOut);

    // 7. Release device resources. Modify them according to the specific API definition.
    aclrtFree(xDeviceAddr);
    aclrtFree(indicesDeviceAddr);
    aclrtFree(expandedXOutDeviceAddr);
    aclrtFree(sortedIndicesOutDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
