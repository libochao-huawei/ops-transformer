# aclnnMoeGatingTopKBackward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:08:01.445Z -->

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

- Description: Computes backward gradients for MoE (Mixture of Experts) gating Top-K selection. This operator is the backward operator of aclnnMoeGatingTopK. It computes the gradient (gradX) of the input score matrix based on the normalized scores (xNorm), upstream gradients (gradY), and expert indices (expertIdx) output by the forward operator. The sigmoid mode (normType=1) is supported.
- Formula (sigmoid mode, normType=1):

  1. Scale gradients

      $$
      gradYScaled_{ip} = routedScalingFactor \cdot gradY_{ip}
      $$

  2. Backward propagation of forward renorm

      $$
      wPrime_{ip} = xNorm_{i,\ expertIdx_{ip}}
      $$

      $$
      D_i = \sum_{p} wPrime_{ip} + eps
      $$

      $$
      w_{ip} = \frac{wPrime_{ip}}{D_i}
      $$

      $$
      beta_i = \sum_{p} w_{ip} \cdot gradYScaled_{ip}
      $$

      $$
      gradWPrime_{ip} = \frac{gradYScaled_{ip} - beta_i}{D_i}
      $$

  3. Scatter to the full dimension

      $$
      gradNormX_{ij} = \sum_{p:\ expertIdx_{ip}=j} gradWPrime_{ip}
      $$

  4. Sigmoid backward propagation

      $$
      gradX_{ij} = xNorm_{ij} \cdot (1 - xNorm_{ij}) \cdot gradNormX_{ij}
      $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnMoeGatingTopKBackwardGetWorkspaceSize" first to obtain the workspace size required for computation and the executor that contains the computation process of the operator, and then call "aclnnMoeGatingTopKBackward" to perform the computation.

```cpp
aclnnStatus aclnnMoeGatingTopKBackwardGetWorkspaceSize(
  const aclTensor  *xNorm,
  const aclTensor  *gradY,
  const aclTensor  *expertIdx,
  int64_t           renorm,
  int64_t           normType,
  double            routedScalingFactor,
  double            eps,
  const aclTensor  *out,
  uint64_t         *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnMoeGatingTopKBackward(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)
```

## aclnnMoeGatingTopKBackwardGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
   <col style="width: 230px">
  <col style="width: 120px">
  <col style="width: 230px">
  <col style="width: 330px">
  <col style="width: 210px">
  <col style="width: 100px">
  <col style="width: 140px">
  <col style="width: 140px">
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
      <td>xNorm (aclTensor*)</td>
      <td>Input</td>
      <td>Input for computation, corresponding to xNorm in the formula.</td>
      <td><ul><li>Does not support empty tensors.</li><li>Must be a 2D tensor whose dimension is [M,N].</li><li>The last dimension (number of experts N) must be greater than or equal to 2 and less than or equal to 2048.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradY (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the upstream gradient of forward operator output yOut, corresponding to gradY in the formula.</td>
      <td><ul><li>Does not support empty tensors.</li><li>A 2D tensor is required. The dimension is [M,K], where K is greater than or equal to 1 and less than or equal to N.</li></ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>expertIdx(aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the top-k expert indices output by the forward operator, corresponding to expertIdx in the formula.</td>
      <td><ul><li>Does not support empty tensors.</li><li>The shape must be consistent with gradY. The dimension is [M,K].</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>renorm (int64_t)</td>
      <td>Input</td>
      <td>Indicates the renorm flag of the forward operator in softmax mode.</td>
      <td>0: no renorm; 1: renorm is required; reserved parameter. Currently, only the sigmoid mode is supported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>normType (int64_t)</td>
      <td>Input</td>
      <td>Indicates the norm function type.</td>
      <td>1 indicates the Sigmoid function, and 0 indicates the Softmax function. Currently, only 1 is supported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>routedScalingFactor (double)</td>
      <td>Input</td>
      <td>Indicates the routed_scaling_factor coefficient used by the forward operator, corresponding to routedScalingFactor in the formula.</td>
      <td></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps (double)</td>
      <td>Input</td>
      <td>Indicates the constant used in forward computation to prevent division by zero, corresponding to eps in the formula.</td>
      <td></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the gradient of the forward operator input parameter x, corresponding to gradX in the formula.</td>
      <td><ul><li>Does not support empty tensors.</li><li>The data type must be consistent with that of gradY.</li><li>The shape must be consistent with that of xNorm, and the dimension is [M,N].</li></ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which includes the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Return Value**

  aclnnStatus: Returned status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
  <col style="width: 300px">
  <col style="width: 150px">
  <col style="width: 650px">
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
      <td>xNorm, gradY, expertIdx, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5"> ACLNN_ERR_PARAM_INVALID </td>
      <td rowspan="5"> 161002 </td>
      <td>The data type of xNorm, gradY, or expertIdx is not within the supported range.</td>
    </tr>
    <tr>
      <td>The data type of out is inconsistent with that of xNorm.</td>
    </tr>
    <tr>
      <td>The shape dimension of xNorm, gradY, or expertIdx is not within the supported range.</td>
    </tr>
    <tr>
      <td>The shape of gradY is inconsistent with that of expertIdx.</td>
    </tr>
    <tr>
      <td>The shapes of out and xNorm are inconsistent.</td>
    </tr>
  </tbody></table>

## aclnnMoeGatingTopKBackward

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
   <col style="width: 200px">
   <col style="width: 130px">
   <col style="width: 770px">
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
      <td>Workspace size allocated on the device, obtained by the first-phase API aclnnMoeGatingTopKBackwardGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream that executes tasks.</td>
    </tr>
  </tbody></table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnMoeGatingTopKBackward defaults to a deterministic implementation.

<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT products</term>:
  - The size of the last dimension of xNorm (the number of experts N) ranges from [2, 2048].
  - The size of the last dimension of gradY (K) ranges from [1, N].

<!-- end id7 -->

## Example

The following sample code is for reference only. For details about how to compile and run, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_gating_top_k_backward.h"

#define CHECK_RET(cond, return_expr)                                                                                  \
    do {                                                                                                              \
        if (!(cond)) {                                                                                                \
            return_expr;                                                                                              \
        }                                                                                                             \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                       \
    do {                                                                                                              \
        printf(message, ##__VA_ARGS__);                                                                               \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
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
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1.Initialize device/stream.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2.Construct inputs and outputs.
    // xNorm: [M, N] = [4, 8], float32, sigmoid output values are in (0,1)
    std::vector<int64_t> xNormShape = {4, 8};
    std::vector<float> xNormHostData = {0.5f, 0.7f, 0.3f, 0.8f, 0.2f, 0.6f, 0.9f, 0.4f, 0.6f, 0.4f, 0.8f,
                                        0.1f, 0.7f, 0.5f, 0.3f, 0.9f, 0.2f, 0.9f, 0.5f, 0.6f, 0.8f, 0.3f,
                                        0.7f, 0.4f, 0.8f, 0.3f, 0.6f, 0.5f, 0.4f, 0.7f, 0.2f, 0.9f};
    // gradY: [M, K] = [4, 2], float32
    std::vector<int64_t> gradYShape = {4, 2};
    std::vector<float> gradYHostData = {1.0f, 0.5f, 0.8f, 0.3f, 0.6f, 0.9f, 0.4f, 0.7f};
    // expertIdx: [M, K] = [4, 2], int32
    std::vector<int64_t> expertIdxShape = {4, 2};
    std::vector<int32_t> expertIdxHostData = {3, 6, 2, 7, 1, 4, 0, 7};
    // gradX: [M, N] = [4, 8], float32
    std::vector<int64_t> gradXShape = {4, 8};
    std::vector<float> gradXHostData(32, 0);

    void *xNormDeviceAddr = nullptr;
    void *gradYDeviceAddr = nullptr;
    void *expertIdxDeviceAddr = nullptr;
    void *gradXDeviceAddr = nullptr;
    aclTensor *xNorm = nullptr;
    aclTensor *gradY = nullptr;
    aclTensor *expertIdx = nullptr;
    aclTensor *gradX = nullptr;

    ret = CreateAclTensor(xNormHostData, xNormShape, &xNormDeviceAddr, aclDataType::ACL_FLOAT, &xNorm);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradYHostData, gradYShape, &gradYDeviceAddr, aclDataType::ACL_FLOAT, &gradY);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertIdxHostData, expertIdxShape, &expertIdxDeviceAddr, aclDataType::ACL_INT32, &expertIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradXHostData, gradXShape, &gradXDeviceAddr, aclDataType::ACL_FLOAT, &gradX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3.Call aclnnMoeGatingTopKBackward.
    int64_t renorm = 0;
    int64_t normType = 1; // sigmoid mode
    double routedScalingFactor = 2.5;
    double eps = 1e-20;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;

    ret = aclnnMoeGatingTopKBackwardGetWorkspaceSize(xNorm, gradY, expertIdx, renorm, normType, routedScalingFactor,
                                                     eps, gradX, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeGatingTopKBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }

    ret = aclnnMoeGatingTopKBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeGatingTopKBackward failed. ERROR: %d\n", ret); return ret);

    // 4.Wait for task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5.Obtain the output values.
    auto size = GetShapeSize(gradXShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradXDeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradX[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release aclTensor.
    aclDestroyTensor(xNorm);
    aclDestroyTensor(gradY);
    aclDestroyTensor(expertIdx);
    aclDestroyTensor(gradX);

    // 7. Release device resources.
    aclrtFree(xNormDeviceAddr);
    aclrtFree(gradYDeviceAddr);
    aclrtFree(expertIdxDeviceAddr);
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
