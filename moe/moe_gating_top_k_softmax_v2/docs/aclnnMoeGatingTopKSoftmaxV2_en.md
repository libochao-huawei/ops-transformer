# aclnnMoeGatingTopKSoftmaxV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:10:44.912Z -->

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

- Description: In MoE computation, if renorm=0, Softmax is first computed on the output of x, and then the topK operation is performed; if renorm=1, the topK operation is first performed on the output of x, and then Softmax is computed. yOut: when renorm=0, the topK values obtained after Softmax is computed on x; when renorm=1, the result obtained by computing topK on x first and then computing Softmax. expertIdxOut is the index result of the topK values, that is, the corresponding expert index; if finished of the corresponding row is True, the expert index is directly filled with num\_expert (that is, the size of the last axis of x).
- Formula:

1. renorm = 0,

    $$
    softmaxResultOutOptional=softmax(x,axis=-1)
    $$

    $$
    yOut,expertIdxOut=topK(softmaxResultOutOptional,k=k)
    $$

2. renorm = 1

    $$
    topkOut,expertIdxOut=topK(x, k=k)
    $$

    $$
    yOut = softmax(topkOut,axis=-1)
    $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First, call "aclnnMoeGatingTopKSoftmaxV2GetWorkspaceSize" to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, call "aclnnMoeGatingTopKSoftmaxV2" to perform the computation.

```c++
aclnnStatus aclnnMoeGatingTopKSoftmaxV2GetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *finishedOptional,
    int64_t          k,
    int64_t          renorm,
    bool             outputSoftmaxResultFlag,
    const aclTensor *yOut,
    const aclTensor *expertIdxOut,
    const aclTensor *softmaxResultOutOptional,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```c++
aclnnStatus aclnnMoeGatingTopKSoftmaxV2(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnMoeGatingTopKSoftmaxV2GetWorkspaceSize

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
    <td>x in the formula, the input to be computed.</td>
    <td>The input must be a 2D/3D tensor, and the size of each dimension must not exceed the maximum value of int32 (2147483647).</td>
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
    <td>k value of topK, which is the number of experts.</td>
    <td>0 <= k <= size of axis -1 of x, and k is not greater than 1024.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>renorm</td>
    <td>Input</td>
    <td>Order of Softmax and TopK.</td>
    <td>Only 0 and 1 are supported. 0 indicates that Softmax is computed before TopK; 1 indicates the reverse order.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>outputSoftmaxResultFlag</td>
    <td>Input</td>
    <td>Indicates whether to output the softmax result.</td>
    <td>When renorm=0, true indicates outputting the Softmax result and false indicates not outputting it. When renorm=1, this parameter does not take effect and the Softmax result is not output.</td>
    <td>BOOL</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>yOut</td>
    <td>Output</td>
    <td>TopK values obtained by applying softmax to x.</td>
    <td>The size of each axis of its shape other than the -1 axis must be consistent with the corresponding axis of x, and the size of its -1 axis must be the same as k.</td>
    <td>Must be consistent with input x.</td>
    <td>ND</td>
    <td>2-3</td>
    <td>x</td>
  </tr>
  <tr>
    <td>expertIdxOut</td>
    <td>Output</td>
    <td>expertIdxOut in the formula, the index result of the topK values, that is, the corresponding expert numbers.</td>
    <td>The shape must be consistent with yOut.</td>
    <td>INT32</td>
    <td>ND</td>
    <td>2-3</td>
    <td>x</td>
  </tr>
  <tr>
    <td>softmaxResultOutOptional</td>
    <td>Output</td>
    <td>The Softmax result during computation.</td>
    <td>The shape must be consistent with x.</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>2-3</td>
    <td>x</td>
  </tr>
  <tr>
    <td>workspaceSize</td>
    <td>Output</td>
    <td>Returns the size of the workspace to be allocated on the Device side.</td>
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
    <td>The data type or data format of inputs and outputs is out of the supported range.</td>
    </tr>
    <tr>
    <td rowspan="8"> ACLNN_ERR_INNER_TILING_ERROR </td>
    <td rowspan="8">561002</td>
    <td>The shape information of multiple input tensors does not match.</td>
    </tr>
    <tr>
    <td>The shape information of the input attribute and input tensor does not match.</td>
    </tr>
    <tr>
    <td>The shape dimension of x is not 2 or 3.</td>
    </tr>
    <tr>
    <td>The shapes of x and finishedOptional do not match.</td>
    </tr>
    <tr>
    <td>The value of k is less than 0 or greater than the size of the -1 axis.</td>
    </tr>
    <tr>
    <td>The value of k is greater than 1024.</td>
    </tr>
    <tr>
    <td>The value of renorm is not 0 or 1.</td>
    </tr>
    <tr>
    <td>The data type of softmaxResultOptional is not in the supported range.</td>
    </tr>
  </tbody></table>

## aclnnMoeGatingTopKSoftmaxV2

- **Parameters:**
  <table>
    <thead>
      <tr><th>Parameter</th><th>Input/Output</th><th>Description</th></tr>
    </thead>
    <tbody>
      <tr><td>workspace</td><td>Input</td><td>Workspace memory address allocated on the Device side.</td></tr>
      <tr><td>workspaceSize</td><td>Input</td><td>Workspace size allocated on the Device side. Obtained by the first-phase API aclnnMoeGatingTopKSoftmaxV2GetWorkspaceSize.</td></tr>
      <tr><td>executor</td><td>Input</td><td> Op executor that contains the operator computation flow. </td></tr>
      <tr><td>stream</td><td>Input</td><td> Stream for task execution. </td></tr>
    </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- deterministic computation:
  - aclnnMoeGatingTopKSoftmaxV2 defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_gating_top_k_softmax_v2.h"
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
  // Boilerplate, resource initialization
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
  // 1. (Boilerplate) Initialize the device/stream. For details, see the list of AscendCL external APIs.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as required.
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. The construction must be customized based on the API.
  std::vector<int64_t> inputShape = {3, 4};
  std::vector<int64_t> outShape = {3, 2};
  std::vector<int64_t> expertIdOutShape = {3, 2};
  std::vector<int64_t> softmaxResultOutOptionalShape = {3, 4};

  void* inputAddr = nullptr;
  void* outAddr = nullptr;
  void* expertIdOutAddr = nullptr;
  void* softmaxResultOutOptionalAddr = nullptr;

  aclTensor* input = nullptr;
  aclTensor* out = nullptr;
  aclTensor* expertIdOut = nullptr;
  aclTensor* softmaxResultOutOptional = nullptr;

  std::vector<float> inputHostData = {0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1};
  std::vector<float> outHostData = {0.1, 1.1, 2.1, 3.1, 4.1, 5.1};
  std::vector<int32_t> expertIdOutHostData = {1, 1, 1, 1, 1, 1};
  std::vector<float> softmaxResultOutOptionalHostData = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  // Create the expandedPermutedRows aclTensor.
  ret = CreateAclTensor(inputHostData, inputShape, &inputAddr, aclDataType::ACL_FLOAT, &input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the expertForSourceRow aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the expandedSrcToDstRow aclTensor.
  ret = CreateAclTensor(expertIdOutHostData, expertIdOutShape, &expertIdOutAddr, aclDataType::ACL_INT32, &expertIdOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the softmaxResultOutOptional aclTensor.
  ret = CreateAclTensor(softmaxResultOutOptionalHostData, softmaxResultOutOptionalShape, &softmaxResultOutOptionalAddr, aclDataType::ACL_FLOAT, &softmaxResultOutOptional);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the specific operator API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnMoeGatingTopKSoftmaxV2.
  ret = aclnnMoeGatingTopKSoftmaxV2GetWorkspaceSize(input, nullptr, 2, 0, true, out, expertIdOut, softmaxResultOutOptional, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeGatingTopKSoftmaxV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnMoeGatingTopKSoftmaxV2.
  ret = aclnnMoeGatingTopKSoftmaxV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeGatingTopKSoftmaxV2 failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait synchronously for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from device memory to the Host. Modify it based on the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0.0f);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                    outAddr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release the aclTensor and aclScalar. Modify it based on the specific API definition.
  aclDestroyTensor(input);
  aclDestroyTensor(out);
  aclDestroyTensor(expertIdOut);
  aclDestroyTensor(softmaxResultOutOptional);

  // 7. Release the device resources. Modify it based on the specific API definition.
  aclrtFree(inputAddr);
  aclrtFree(outAddr);
  aclrtFree(expertIdOutAddr);
  aclrtFree(softmaxResultOutOptionalAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
