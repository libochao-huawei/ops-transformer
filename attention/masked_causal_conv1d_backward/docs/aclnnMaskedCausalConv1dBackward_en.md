# aclnnMaskedCausalConv1dBackward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T11:12:52.206Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
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

- Description: Performs backward gradient computation for 1D grouped convolution operations on tokens in the hidden layer.

- Formula:

  Assume that the convolution input x, the gradient of the convolution output gradY, and the gradient of the convolution input gradX have a shape of [S, B, H], weight has a shape of [W, H], i and j denote the indices of the S/B axes respectively, and k is the index within the convolution window W. The computation is expressed as:

  $$
  gradYMasked[i,j] = mask[j,i] * gradY[i,j]
  $$

  $$
  gradX[i,j] = \sum_{k=0}^{W-1} gradYMasked[i+k,j] * weight[W-1-k]
  $$

  $$
  gradWeight[k] = \sum_{j=0}^{B-1}\sum_{i=0}^{S-1} gradYMasked[i+W-1-k,j] * x[i,j]
  $$

  where invalid positions are padded with 0; currently only 3 is supported for W; the operation on the H axis is elementwise and is not reflected in the preceding formulas.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must call "aclnnMaskedCausalConv1dBackwardGetWorkspaceSize" first to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call "aclnnMaskedCausalConv1dBackward" to execute the computation.

```Cpp
aclnnStatus aclnnMaskedCausalConv1dBackwardGetWorkspaceSize(
  const aclTensor   *gradY,
  const aclTensor   *x,
  const aclTensor   *weight,
  const aclTensor   *maskOptional,
  const aclTensor   *gradXOut,
  const aclTensor   *gradWeightOut,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnMaskedCausalConv1dBackward(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  aclrtStream       stream)
```

## aclnnMaskedCausalConv1dBackwardGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1565px">
    <colgroup>
      <col style="width: 160px">
      <col style="width: 135px">
      <col style="width: 326px">
      <col style="width: 246px">
      <col style="width: 275px">
      <col style="width: 101px">
      <col style="width: 190px">
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
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>gradY (aclTensor*)</td>
        <td>Input</td>
        <td>Represents the gradient of grouped convolution output y, corresponding to gradY in the formula.</td>
        <td>          <ul>
            <li>Empty tensor not supported.</li>
            <li>The shape is [S, B, H].</li>
          </ul>
        </td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x (aclTensor*)</td>
        <td>Input</td>
        <td>Represents the grouped convolution input, corresponding to x in the formula.</td>
        <td>          <ul>
            <li>Empty tensors are not supported.</li>
            <li>The shape must be consistent with gradY.</li>
          </ul>
        </td>
        <td>Data type consistent with gradY</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weight (aclTensor*)</td>
        <td>Input</td>
        <td>Convolution weight, which is the weight in the corresponding formula.</td>
        <td>          <ul>
            <li>Empty tensors are not supported.</li>
            <li>The shape must be [W, H].</li>
          </ul>
        </td>
        <td>Data type consistent with gradY</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>maskOptional (aclTensor*)</td>
        <td>Optional input</td>
        <td>Represents the output mask of the convolution operation, corresponding to mask in the formula.</td>
        <td>          <ul>
            <li>Empty tensors are not supported.</li>
            <li>The shape is [B, S].</li>
            <li>The default value is nullptr.</li>
          </ul>
        </td>
        <td>BOOL</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>gradXOut (aclTensor*)</td>
        <td>Output</td>
        <td>Represents the gradient of the grouped convolution input x, corresponding to gradX in the formula.</td>
        <td>          <ul>
            <li>Empty tensors are not supported.</li>
            <li>The shape must be consistent with gradY.</li>
          </ul>
        </td>
        <td>Data type consistent with gradY</td>
        <td>ND</td>
        <td>3</td>
        <td>x</td>
      </tr>
      <tr>
        <td>gradWeightOut (aclTensor*)</td>
        <td>Output</td>
        <td>Gradient of the grouped convolution input weight, i.e., gradWeight in the corresponding formula.</td>
        <td>          <ul>
            <li>Empty tensors are not supported.</li>
            <li>The shape is consistent with that of weight.</li>
          </ul>
        </td>
        <td>Data type consistent with gradY</td>
        <td>ND</td>
        <td>2</td>
        <td>x</td>
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
        <td>Returns the op executor, which contains the operator computation flow.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody>
  </table>

- **Return value**

  Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).
  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input parameter is a required input, output, or required attribute and is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data type or data format of gradY, x, weight, maskOptional, gradXOut, or gradWeightOut is not in the supported range.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>The shape of the input or output tensors is not in the supported range.</td>
    </tr>
  </tbody>
  </table>

## aclnnMaskedCausalConv1dBackward

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 598px"><colgroup>
  <col style="width: 144px">
  <col style="width: 125px">
  <col style="width: 700px">
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
      <td>Workspace memory address allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device side, obtained by the first-phase API aclnnMaskedCausalConv1dBackwardGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor that contains the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- When using this API with PyTorch, ensure that the versions of CANN-related packages and PyTorch-related packages are compatible.
- Deterministic computation:
  - aclnnMaskedCausalConv1dBackward defaults to a deterministic implementation.
- Input restrictions:
  - The shape data range constraints for inputs and outputs are as follows:
    - B (Batchsize): value range 1-32.
    - B *S (Batchsize* SeqLength): value range 1-512K.
    - H (hiddenSize): value range 384-24576, and it must be a multiple of 64.
    - W: only 3 is supported currently.
- The operator input parameters and intermediate computation results do not exceed the value range of the corresponding runtime data types (float16/bfloat16).
- Operator inputs do not support ±inf or nan values.

## Example

The following is sample code for calling the aclnn single operator API, for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_masked_causal_conv1d_backward.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

void PrintOutResult(std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<int8_t> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // Boilerplate for resource initialization
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate device memory.
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
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. Boilerplate: device/stream initialization. For details, see the acl API.
    // Fill in deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output according to the API definition.
    int64_t S = 1;
    int64_t B = 2;
    int64_t H = 192 * 2;
    int64_t W = 3;

    std::vector<int64_t> gradYShape = {S, B, H};
    std::vector<int64_t> xShape = {S, B, H};
    std::vector<int64_t> weightShape = {W, H};
    std::vector<int64_t> maskShape = {B, S};
    std::vector<int64_t> gradXShape = {S, B, H};
    std::vector<int64_t> gradWeightShape = {W, H};

    std::vector<op::fp16_t> gradYHostData(S * B * H, 0);
    std::vector<op::fp16_t> xHostData(S * B * H, 0);
    std::vector<op::fp16_t> weightHostData(W * H, 0);
    std::vector<uint8_t> maskHostData(B * S, 0);
    std::vector<op::fp16_t> gradXHostData(S * B * H, 0);
    std::vector<op::fp16_t> gradWeightHostData(W * H, 0);

    void* gradYDeviceAddr = nullptr;
    void* xDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* maskDeviceAddr = nullptr;
    void* gradXDeviceAddr = nullptr;
    void* gradWeightDeviceAddr = nullptr;

    aclTensor* gradY = nullptr;
    aclTensor* x = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* mask = nullptr;
    aclTensor* gradX = nullptr;
    aclTensor* gradWeight = nullptr;

    ret = CreateAclTensor(gradYHostData, gradYShape, &gradYDeviceAddr, aclDataType::ACL_FLOAT16, &gradY);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(maskHostData, maskShape, &maskDeviceAddr, aclDataType::ACL_BOOL, &mask);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradXHostData, gradXShape, &gradXDeviceAddr, aclDataType::ACL_FLOAT16, &gradX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradWeightHostData, gradWeightShape, &gradWeightDeviceAddr, aclDataType::ACL_FLOAT16, &gradWeight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Change it to the specific API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // Call the first-phase API of aclnnMaskedCausalConv1dBackward.
    ret = aclnnMaskedCausalConv1dBackwardGetWorkspaceSize(
        gradY, x, weight, mask, gradX, gradWeight, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaskedCausalConv1dBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // Call the second-phase API of aclnnMaskedCausalConv1dBackward.
    ret = aclnnMaskedCausalConv1dBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaskedCausalConv1dBackward failed. ERROR: %d\n", ret); return ret);

    // 4. Boilerplate: synchronously wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result on the device to the host. Modify the code based on the specific API definition.
    PrintOutResult(gradXShape, &gradXDeviceAddr);
    PrintOutResult(gradWeightShape, &gradWeightDeviceAddr);

    // 6. Release aclTensor and aclScalar. Modify the code based on the specific API definition.
    aclDestroyTensor(gradY);
    aclDestroyTensor(x);
    aclDestroyTensor(weight);
    aclDestroyTensor(mask);
    aclDestroyTensor(gradX);
    aclDestroyTensor(gradWeight);

    // 7. Release device resources.
    aclrtFree(gradYDeviceAddr);
    aclrtFree(xDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(maskDeviceAddr);
    aclrtFree(gradXDeviceAddr);
    aclrtFree(gradWeightDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
