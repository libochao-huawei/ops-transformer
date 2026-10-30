# aclnnMoeGatingTopKSoftmax

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:13:52.541Z -->

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

- Description: In MoE computation, performs Softmax calculation on the output of x and then takes the topK operation. yOut is the topK result of softmax; expertIdxOut is the index result of the topK values, i.e., the corresponding expert number; rowIdxOut is the column-wise result with the same shape as expertIdxOut, indicating the original row position corresponding to each position. If the finished flag of the corresponding row in expertIdxOut is True, the value is directly filled with the num\_expert value (i.e., the size of the last axis of x).
- Formula:

  $$
  softmaxOut=softmax(x,axis=-1)
  $$

  $$
  yOut,expertIdxOut=topK(softmaxOut,k=k)
  $$

  $$
  rowIdxRange=arange(expertIdxOut.shape[0]*expertIdxOut.shape[1])
  $$

  $$
  rowIdxOut=rowIdxRange.reshape([expertIdxOut.shape[1],expertIdxOut.shape[0]]).transpose(1,0)
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnMoeGatingTopKSoftmaxGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor containing the operator computation process. Then call the "aclnnMoeGatingTopKSoftmax" API to execute the computation.

```c++
aclnnStatus aclnnMoeGatingTopKSoftmaxGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *finishedOptional,
    int64_t          k,
    const aclTensor *yOut,
    const aclTensor *expertIdxOut,
    const aclTensor *rowIdxOut,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```c++
aclnnStatus aclnnMoeGatingTopKSoftmax(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnMoeGatingTopKSoftmaxGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 187px">
  <col style="width: 121px">
  <col style="width: 287px">
  <col style="width: 387px">
  <col style="width: 187px">
  <col style="width: 187px">
  <col style="width: 187px">
  <col style="width: 146px">
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
    <td>x in the formula, the input to be calculated.</td>
    <td>Must be a 2D/3D tensor. The size of each dimension must not exceed the maximum value of int32 (2147483647).</td>
    <td>FLOAT16, BFLOAT16, FLOAT32</td>
    <td>ND</td>
    <td>2-3</td>
    <td>√</td>
  </tr>
  <tr>
    <td>finishedOptional</td>
    <td>Input</td>
    <td>finished in the formula, indicating whether the row participates in the computation.</td>
    <td>The shape is x_shape[:-1].</td>
    <td>BOOL</td>
    <td>ND</td>
    <td>1-2</td>
    <td>√</td>
  </tr>
  <tr>
    <td>k</td>
    <td>Input</td>
    <td>k value of topK, indicating the number of experts.</td>
    <td>0 <= k <= the size of axis -1 of x, and k <= 1024.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>yOut</td>
    <td>Output</td>
    <td>topK values after softmax on x.</td>
    <td>Axes of its shape other than -1 must be consistent with the corresponding axes of x in size, and its -1 axis must have the same size as k</td>
    <td>Must be consistent with input x.</td>
    <td>ND</td>
    <td>2-3</td>
    <td>x</td>
  </tr>
  <tr>
    <td>expertIdxOut</td>
    <td>Output</td>
    <td>Indices of the topK values after softmax is performed on x, that is, expert indices.</td>
    <td>Shape must be consistent with that of yOut.</td>
    <td>INT32</td>
    <td>ND</td>
    <td>2-3</td>
    <td>x</td>
  </tr>
  <tr>
    <td>rowIdxOut</td>
    <td>Output</td>
    <td>rowIdxOut in the formula.</td>
    <td>The shape must be consistent with that of yOut.</td>
    <td>INT32</td>
    <td>ND</td>
    <td>2-3</td>
    <td>x</td>
  </tr>
  <tr>
    <td>workspaceSize</td>
    <td>Output</td>
    <td>Returns the workspace size required on the Device side.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>executor</td>
    <td>Output</td>
    <td>Returns the op executor containing the operator computation flow.</td>
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
    <td> ACLNN_ERR_PARAM_NULLPTR </td>
    <td> 161001 </td>
    <td>A required input, required output, or required attribute is a null pointer.</td>
    </tr>
    <tr>
    <td> ACLNN_ERR_PARAM_INVALID </td>
    <td> 161002 </td>
    <td>The data type or data format of an input or output is out of the supported range.</td>
    </tr>
    <tr>
    <td rowspan="4"> ACLNN_ERR_INNER_TILING_ERROR </td>
    <td rowspan="4"> 561002 </td>
    <td>The shape information of multiple input tensors does not match.</td>
    </tr>
    <tr>
    <td>The shape information of input attributes does not match that of input tensors.</td>
    </tr>
    <tr>
    <td>The value of k is less than 0 or greater than the size of the last axis.</td>
    </tr>
    <tr>
    <td>The value of k is greater than 1024.</td>
    </tr>
  </tbody></table>

## aclnnMoeGatingTopKSoftmax

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
      <td>Workspace memory address allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device side, obtained by the first-phase API aclnnMoeGatingTopKSoftmaxGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor. It contains the operator computation flow.</td>
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

- deterministic computation:
  - aclnnMoeGatingTopKSoftmax defaults to a deterministic implementation.

<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT products</term>:
  - The size of the last dimension of x (i.e., the number of experts) ranges from [1, 2048].

<!-- end id7 -->
<!-- npu="310p" id8 -->
- <term>Atlas inference products</term>:
  - The input x shape is [token_num, expert_num]. expert_num must be aligned to 16, that is, expert_num must be a multiple of 16 and cannot exceed 256.
  - Only FLOAT16 is supported for input x.
  - finishedOptional is not supported as input. Only None can be used.
  - rowIdxOut is not supported as output. The output of rowIdxOut is all 0.

<!-- end id8 -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_gating_top_k_softmax.h"
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
  // 1. (boilerplate) device/stream initialization, refer to the acl API manual
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as required
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs. Custom construction is required based on the API.
  std::vector<int64_t> inputShape = {3, 4};
  std::vector<int64_t> outShape = {3, 2};
  std::vector<int64_t> expertIdOutShape = {3, 2};
  std::vector<int64_t> rowOutShape = {3, 2};

  void* inputAddr = nullptr;
  void* outAddr = nullptr;
  void* expertIdOutAddr = nullptr;
  void* rowOutAddr = nullptr;

  aclTensor* input = nullptr;
  aclTensor* finished = nullptr;
  aclTensor* out = nullptr;
  aclTensor* expertIdOut = nullptr;
  aclTensor* rowOut = nullptr;

  std::vector<float> inputHostData = {0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1};
  std::vector<float> outHostData = {0.1, 1.1, 2.1, 3.1, 4.1, 5.1};
  std::vector<int32_t> expertIdOutHostData = {1, 1, 1, 1, 1, 1};
  std::vector<int32_t> rowOutHostData = {1, 1, 1, 1, 1, 1};

  // Create the expandedPermutedRows aclTensor
  ret = CreateAclTensor(inputHostData, inputShape, &inputAddr, aclDataType::ACL_FLOAT, &input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the expertForSourceRow aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the expandedSrcToDstRow aclTensor
  ret = CreateAclTensor(expertIdOutHostData, expertIdOutShape, &expertIdOutAddr, aclDataType::ACL_INT32, &expertIdOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the Out aclTensor
  ret = CreateAclTensor(rowOutHostData, rowOutShape, &rowOutAddr, aclDataType::ACL_INT32, &rowOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Change it to the specific operator API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnMoeGatingTopKSoftmax
  ret = aclnnMoeGatingTopKSoftmaxGetWorkspaceSize(input, finished, 2, out, expertIdOut, rowOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeGatingTopKSoftmaxGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnMoeGatingTopKSoftmax.
  ret = aclnnMoeGatingTopKSoftmax(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeGatingTopKSoftmax failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronously wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result on the device memory to the host. Modify according to the actual API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0.0f);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                    outAddr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify according to the actual API definition.
  aclDestroyTensor(input);
  aclDestroyTensor(out);
  aclDestroyTensor(expertIdOut);
  aclDestroyTensor(rowOut);

  // 7. Release device resources. Modify according to the actual API definition.
  aclrtFree(inputAddr);
  aclrtFree(outAddr);
  aclrtFree(expertIdOutAddr);
  aclrtFree(rowOutAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
