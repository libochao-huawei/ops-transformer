# aclnnMoeTokenPermuteGrad

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:29:06.523Z -->

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

## Description

- **Description**: backward propagation computation of [aclnnMoeTokenPermute](../../moe_token_permute/docs/aclnnMoeTokenPermute_en.md).
- **Formula**:

  $$
  inputGrad = permutedOutputGrad.indexSelect(0, sortedIndices)
  $$

  $$
  inputGrad = inputGrad.reshape(-1, numTopk, hiddenSize)
  $$

  $$
  inputGrad = inputGrad.sum(dim = 1)
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnMoeTokenPermuteGradGetWorkspaceSize" API to obtain the input parameters, and calculate the required workspace size and the executor that contains the operator computation process based on the computation flow. Then call the "aclnnMoeTokenPermuteGrad" API to perform the computation.

```c++
aclnnStatus aclnnMoeTokenPermuteGradGetWorkspaceSize(
    const aclTensor *permutedOutputGrad,
    const aclTensor *sortedIndices,
    int64_t          numTopk,
    bool             paddedMode,
    aclTensor       *out,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```c++
aclnnStatus aclnnMoeTokenPermuteGrad(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnMoeTokenPermuteGradGetWorkspaceSize

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
      <td>permutedOutputGrad</td>
      <td>Input</td>
      <td>Gradient of the forward output permutedTokens.</td>
      <td>Required to be a 2D tensor, where tokens_num is the number of tokens and topK_num is the value of numTopk.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>[tokens_num * topK_num, hidden_size]</td>
      <td>√</td>
  </tr>
  <tr>
      <td>sortedIndices</td>
      <td>Input</td>
      <td>Sorted index values.</td>
      <td>Value range: (0, tokens_num * topK_num - 1), with no duplicate indexes.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>tokens_num * topK_num</td>
      <td>√</td>
  </tr>
  <tr>
      <td>numTopk</td>
      <td>Input</td>
      <td>Number of selected experts.</td>
      <td>numTopk <= 512.</td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
  </tr>
  <tr>
      <td>paddedMode</td>
      <td>Input</td>
      <td>Whether to enable paddedMode mode.</td>
      <td>true indicates that paddedMode is enabled, false indicates that paddedMode is disabled. Currently, only false is supported.</td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  <tr>
      <td>out</td>
      <td>Output</td>
      <td>Gradient of input tokens.</td>
      <td>-</td>
      <td>BFLOAT16, FLOAT16, FLOAT32.</td>
      <td>ND</td>
      <td>(tokens_num, hidden_size)</td>
      <td>√</td>
  </tr>
  <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor that contains the operator computation flow.</td>
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

  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
      <col style="width: 267px">
      <col style="width: 124px">
      <col style="width: 775px">
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
          <td>The passed mandatory input, mandatory output, or mandatory attribute is a null pointer.</td>
          </tr>
          <tr>
          <td> ACLNN_ERR_PARAM_INVALID </td>
          <td> 161002 </td>
          <td>The data type and data format of the input and output are out of the supported range.</td>
          </tr>
      </tbody>
  </table>

## aclnnMoeTokenPermuteGrad

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
      <td>Workspace size allocated on the Device side, obtained by the first-phase API aclnnMoeTokenPermuteGradGetWorkspaceSize.</td>
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

- Deterministic computation:
  - aclnnMoeTokenPermuteGrad defaults to a deterministic implementation.

<!-- npu="A3,910b" id7 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: numTopk <= 512.
<!-- end id7 -->
<!-- npu="950" id8 -->
- <term>Ascend 950PR&950DT products</term>:
  When this API is called, the framework internally calls [aclnnMoeInitRoutingV2Grad](../../moe_init_routing_v2_grad/docs/aclnnMoeInitRoutingV2Grad_en.md). If a parameter error message is displayed, see the following parameter mapping:
  - The permutedOutputGrad input equivalence is the gradExpandedX input of the aclnnMoeInitRoutingV2Grad API.
  - The sortedIndices input equivalence is the expandedRowIdx input of the aclnnMoeInitRoutingV2Grad API.
  - The numTopk input equivalence is the topK input of the aclnnMoeInitRoutingV2Grad API.
  - The paddedMode input equivalence is the dropPadMode input of the aclnnMoeInitRoutingV2Grad API.
  - The out output is equivalent to the out output of the aclnnMoeInitRoutingV2Grad API.

<!-- end id8 -->

## Example

The following sample code is for reference only. For details about how to compile and run the code, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp

#include "acl/acl.h"
#include "aclnnop/aclnn_moe_token_permute_grad.h"
#include <iostream>
#include <vector>
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
  auto ret = aclrtMemcpy(
      resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
      return );
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream *stream) {
  // Boilerplate: resource initialization
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
            return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData,
                    const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate device-side memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
            return ret);
  // Call aclrtMemcpy to copy host-side data to device-side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size,
                    ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
            return ret);

  // Calculate the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor API to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
                            strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API documentation.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
            return ret);

  // 2. Construct the inputs and outputs. The construction must be customized based on the API interface.

  int64_t num_topk = 2;
  std::vector<float> permuted_output_grad_Data = {1, 2, 3, 4};
  std::vector<int64_t> permuted_output_grad_Shape = {2, 2};
  void *permuted_output_grad_Addr = nullptr;
  aclTensor *permuted_output_grad = nullptr;

  ret = CreateAclTensor(permuted_output_grad_Data, permuted_output_grad_Shape,
                        &permuted_output_grad_Addr, aclDataType::ACL_FLOAT,
                        &permuted_output_grad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int> sortedIndicesData = {0, 1};
  std::vector<int64_t> sortedIndicesShape = {2};
  void *sortedIndicesAddr = nullptr;
  aclTensor *sortedIndices = nullptr;

  ret = CreateAclTensor(sortedIndicesData, sortedIndicesShape, &sortedIndicesAddr,
                      aclDataType::ACL_INT32, &sortedIndices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<float> outData = {0, 0};
  std::vector<int64_t> outShape = {1, 2};
  void *outAddr = nullptr;
  aclTensor *out = nullptr;

  ret = CreateAclTensor(outData, outShape, &outAddr, aclDataType::ACL_FLOAT,
                        &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;

  // Call the first-phase API of aclnnMoeTokenPermuteGrad.
  ret = aclnnMoeTokenPermuteGradGetWorkspaceSize(permuted_output_grad, sortedIndices,
                                                 num_topk, false,
                                                 out, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnMoeTokenPermuteGradGetWorkspaceSize failed. ERROR: %d\n",
                ret);
      return ret);

  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  // Call the second-phase API of aclnnMoeTokenPermuteGrad.
  ret = aclnnMoeTokenPermuteGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnMoeTokenPermuteGrad failed. ERROR: %d\n", ret);
            return ret);

  // 4. (boilerplate) Synchronize and wait for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5. Obtain the output value, and copy the result on device memory to host memory. Modify this step based on the specific API definition.
  PrintOutResult(outShape, &outAddr);

  // 6. Release aclTensor and aclScalar. Modify this step based on the specific API definition.
  aclDestroyTensor(permuted_output_grad);
  aclDestroyTensor(sortedIndices);
  aclDestroyTensor(out);

  // 7. Release device resources.
  aclrtFree(permuted_output_grad_Addr);
  aclrtFree(sortedIndicesAddr);
  aclrtFree(outAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
