# aclnnMoeInitRoutingV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:17:05.574Z -->

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

- **Description**: This operator corresponds to the **Routing computation** in MoE (Mixture of Experts). It takes the computation result of the [aclnnMoeGatingTopKSoftmax](../../moe_gating_top_k_softmax/docs/aclnnMoeGatingTopKSoftmax_en.md) operator as input, and outputs results such as the Routing matrix expandedXOut for subsequent computation. This API includes the following function changes compared with [aclnnMoeInitRouting](../../moe_init_routing/docs/aclnnMoeInitRouting_en.md). Select an appropriate API based on the actual requirements:

  - Adds the Drop mode. In this mode, the output aligns the number of Tokens to be processed by each expert to expertCapacity. Tokens exceeding expertCapacity are dropped, and insufficient Tokens are padded with 0.
  - Adds the optional output expertTokensCountOrCumsumOut in Dropless mode, which outputs the cumulative number of tokens (Cumsum) to be processed by each expert, or the number of tokens (Count) to be processed by each expert.
  - Adds the optional output expertTokensBeforeCapacityOut in Drop mode, which outputs the number of tokens to be processed by each expert before the Drop operation.
  - Deletes the rowIdx input.

  **NOTE:**
  Routing computation is a step in MoE models. A MoE model mainly consists of a group of expert models and a gating model. During computation, the input data is first processed by the Gating Network (which includes the MoeGatingTopKSoftmax operator) to obtain the k experts with the highest weights for each data element. The result is then input to the MoeInitRouting operator to generate the Routing matrix. Each expert in the model then processes the data it should handle based on the Routing matrix and generates the corresponding output. The outputs of all experts are finally weighted and summed to form the final prediction result.
- **Formula**:

  1) Flatten the input expertIdx with shape [numRows, k] or [numRows] into one row and sort it, obtaining the sorted result sortedExpertIdx and the corresponding indices sortedRowIdx, where numRows is the number of tokens, k indicates the k experts with the highest weights corresponding to each token data element, and k is set to 1 when expertIdx is 1-dimensional:

  $$
  sortedExpertIdx, sortedRowIdx=keyValueSort(\text{flatten}(expertIdx))
  $$

  2) Perform position mapping using sortedRowIdx to obtain expandedRowIdxOut:

  $$
  expandedRowIdxOut[sortedRowIdx[i]]=i
  $$

  3) Arrange tokens by expert order according to sortedRowIdx. When dropPadMode is 1, perform count alignment so that the number of tokens to be processed by each expert is aligned to expertCapacity. Tokens exceeding expertCapacity are dropped, and the insufficient ones are padded with 0. Obtain expandedXOut:

  $$
  expandedXOut[expandedRowIdxOut[i]]=x[i//k]
  $$

  4.Calculate the histogram result of each expert in sortedExpertIdx, then perform Cumsum to obtain expertTokensCountOrCumsumOut:

  $$
  expertTokensCountOrCumsumOut[i]=Cumsum(Histogram(sortedExpertIdx))
  $$

  5.Calculate the histogram result of each expert in sortedExpertIdx to obtain expertTokensBeforeCapacityOut:

  $$
  expertTokensBeforeCapacityOut[i]=Histogram(sortedExpertIdx)
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnMoeInitRoutingV2GetWorkspaceSize" API first to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the "aclnnMoeInitRoutingV2" API to perform the computation.

```cpp
aclnnStatus aclnnMoeInitRoutingV2GetWorkspaceSize(
    const aclTensor  *x,
    const aclTensor  *expertIdx,
    int64_t           activeNum,
    int64_t           expertCapacity,
    int64_t           expertNum,
    int64_t           dropPadMode,
    int64_t           expertTokensCountOrCumsumFlag,
    bool              expertTokensBeforeCapacityFlag,
    const aclTensor  *expandedXOut,
    const aclTensor  *expandedRowIdxOut,
    const aclTensor  *expertTokensCountOrCumsumOut,
    const aclTensor  *expertTokensBeforeCapacityOut,
    uint64_t         *workspaceSize,
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnMoeInitRoutingV2(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    aclrtStream       stream)
```

## aclnnMoeInitRoutingV2GetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1562px"><colgroup>
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
      <td>x</td>
      <td>Input</td>
      <td>MoE input, i.e., token feature input.</td>
      <td><ul><li>Supports empty tensor.</li><li>Must be a 2D tensor with shape [numRows, h], where numRows indicates the number of tokens and h indicates the length of each token.</li></ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>expertIdx</td>
      <td>Input</td>
      <td>Indices of the k experts for each token.</td>
      <td><ul><li>Supports empty tensors.</li><li>In the Drop/Pad scenario, or in the non-Drop/Pad scenario where expertTokensCountOrCumsumOut needs to be output, the value range must be [0, expertNum - 1]. In other scenarios, the value must be greater than or equal to 0.</li></ul></td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>1 or 2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>activeNum</td>
      <td>Input</td>
      <td>Specifies whether the Active scenario is used.</td>
      <td>This attribute takes effect in the non-Drop/Pad scenario where dropPadMode is 0. The value range is greater than or equal to 0. A value of 0 indicates the Dropless scenario, and a value greater than 0 indicates the Active scenario. The total number of tokens processed by all experts must not exceed activeNum.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertCapacity</td>
      <td>Input</td>
      <td>Represents the number of tokens that each expert can process.</td>
      <td>Value range: greater than or equal to 0. In the Drop/Pad scenario, the value range is (0, numRows]. In this case, each expert drops tokens that exceed capacity and pads all-zero tokens when the number of tokens is less than the capacity threshold. Other scenarios do not use this attribute value.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertNum</td>
      <td>Input</td>
      <td>Total number of experts.</td>
      <td>Value range: greater than or equal to 0. In the Drop/Pad scenario or when expertTokensCountOrCumsumFlag is greater than 0 and expertTokensCountOrCumsumOut needs to be output, expertNum must be greater than 0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dropPadMode</td>
      <td>Input</td>
      <td>Indicates whether the Drop/Pad scenario is used.</td>
      <td>Value is 0 or 1.<ul><li>0: non-Drop/Pad scenario. In this scenario, expertCapacity is not validated.</li><li>1: Drop/Pad scenario. expertNum and expertCapacity need to be validated. In this mode, the output performs count alignment so that the number of tokens to be processed by each expert is aligned to expertCapacity. Tokens exceeding expertCapacity are dropped, and the insufficient part is padded with 0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertTokensCountOrCumsumFlag</td>
      <td>Input</td>
      <td>Controls whether to output expertTokensCountOrCumsumOut.</td>
      <td>The value is 0, 1, or 2.<ul><li>0: expertTokensCountOrCumsumOut is not output.</li><li>1: The output value is the cumulative number of tokens processed by each expert.</li><li>2: The output value is the number of tokens processed by each expert.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expertTokensBeforeCapacityFlag</td>
      <td>Input</td>
      <td>Controls whether to output expertTokensBeforeCapacityOut.</td>
      <td>The value is false or true.<ul><li>false: expertTokensBeforeCapacityOut is not output.</li><li>true: expertTokensBeforeCapacityOut is output, and the value is the number of tokens processed by each expert before drop.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expandedXOut</td>
      <td>Output</td>
      <td>Features expanded based on expertIdx.</td>
      <td><ul><li>Supports empty tensor.</li><li>The data type must be the same as that of x.</li><li>In Dropless/Active scenarios, this parameter must be a 2D tensor. The shape is [numRows * k, h] in the Dropless scenario and [min(activeNum, numRows * k), h] in the Active scenario.<br>In Drop/Pad scenarios, this parameter must be a 3D tensor with the shape [expertNum, expertCapacity, h].</li></ul>
      </td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2 or 3</td>
      <td>×</td>
    </tr>
    <tr>
      <td>expandedRowIdxOut</td>
      <td>Output</td>
      <td>Index mapping between expandedXOut and x.</td>
      <td><ul><li>Supports empty tensors.</li><li>Must be a 1D tensor with shape [numRows*k].</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>expertTokensCountOrCumsumOut</td>
      <td>Output</td>
      <td>Outputs the statistics and cumulative value of the number of tokens processed by each expert.</td>
      <td><ul><li>Supports empty tensor.</li><li>Whether to output is controlled by the expertTokensCountOrCumsumFlag parameter. The value is output only in scenarios other than Drop/Pad. It must be a 1D tensor with shape [expertNum].</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>expertTokensBeforeCapacityOut</td>
      <td>Output</td>
      <td>Outputs the statistics of the number of tokens processed by each expert before drop.</td>
      <td><ul><li>Supports empty tensor</li><li>Output is controlled by the expertTokensBeforeCapacityFlag parameter. This value is output only in the Drop/Pad scenario and must be a 1D tensor with shape [expertNum].</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size that the user needs to allocate on the Device.</td>
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

  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: The data type of the input expertIdx supports INT32. It must be a 2D shape [numRows, k].
  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT products</term>: The data type of the input expertIdx supports INT32 and INT64. It must be a 2D shape [numRows, k] or a 1D shape [numRows]. A 1D shape indicates k=1.
  <!-- end id8 -->
  <!-- npu="310p" id9 -->
  - |<term>Atlas inference products</term>: The data type of the input expertIdx supports INT32. It must be a 2D shape [numRows, k]. Only 0 is supported for dropPadMode. The trailing axis size of input x must be divisible by 32.

  <!-- end id9 -->

- **Return value**

  `aclnnStatus`: Return status code. For details, see <a href="../../../docs/en/context/aclnn_return_code.md">aclnn return codes</a>.

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
        <td>The computation inputs and required computation outputs are null pointers.</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td>The data types and formats of the computation inputs and outputs are not within the supported range.</td>
      </tr>
      <tr>
        <td rowspan="5">ACLNN_ERR_INNER_NULLPTR</td>
        <td rowspan="5">561103</td>
        <td>When output of expertTokensCountOrCumsumOut is required, expertNum is 0.</td>
      </tr>
      <tr>
        <td>The shape dimensions of x and expertIdx are not 2, and their first dimensions are not equal. On |<term>Atlas inference products</term>, the size of the trailing axis of input x cannot be evenly divided by 32</td>
      </tr>
      <tr>
        <td>The value of activeNum, expertNum, or expertCapacity is less than 0.</td>
      </tr>
      <tr>
        <td>dropPadMode, expertTokensCountOrCumsumFlag, or expertTokensBeforeCapacityFlag is out of the value range.</td>
      </tr>
      <tr>
        <td>When dropPadMode is 1, expertCapacity and expertNum are 0.</td>
      </tr>
    </tbody>
  </table>

## aclnnMoeInitRoutingV2

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1179px"> <colgroup>
  <col style="width: 169px">
  <col style="width: 130px">
  <col style="width: 880px">
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
    <td>Workspace memory address requested on the Device.</td>
  </tr>
  <tr>
    <td>workspaceSize</td>
    <td>Input</td>
    <td>Workspace size requested on the Device, obtained by the first-phase API <code>aclnnMoeInitRoutingV2GetWorkspaceSize</code>.</td>
  </tr>
  <tr>
    <td>executor</td>
    <td>Input</td>
    <td>Op executor that contains the operator computation process.</td>
  </tr>
  <tr>
    <td>stream</td>
    <td>Input</td>
    <td>Stream for task execution.</td>
  </tr>
  </tbody></table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnMoeInitRoutingV2 defaults to a deterministic implementation.

## Example

The following is sample code for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_init_routing_v2.h"
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
    // Boilerplate: resource initialization.
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
    // Call aclrtMemcpy to copy data on the host to memory on the device.
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
    // 1. Boilerplate: device/stream initialization. For details, see the ACL external API list.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as needed.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the inputs and outputs according to the API definition.
    std::vector<int64_t> xShape = {3, 4};
    std::vector<int64_t> idxShape = {3, 2};
    std::vector<int64_t> expandedXOutShape = {3, 2, 4};
    std::vector<int64_t> idxOutShape = {6};
    std::vector<int64_t> expertTokenOutShape = {3};
    void* xDeviceAddr = nullptr;
    void* expertIdxDeviceAddr = nullptr;
    void* expandedXOutDeviceAddr = nullptr;
    void* expandedRowIdxOutDeviceAddr = nullptr;
    void* expertTokenBeforeCapacityOutDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* expertIdx = nullptr;
    int64_t activeNum = 0;
    int64_t expertCapacity = 2;
    int64_t expertNum = 3;
    int64_t dropPadMode = 1;
    int64_t expertTokensCountOrCumsumFlag = 0;
    bool expertTokensBeforeCapacityFlag = true;
    aclTensor* expandedXOut = nullptr;
    aclTensor* expandedRowIdxOut = nullptr;
    aclTensor* expertTokensBeforeCapacityOut = nullptr;
    std::vector<float> xHostData = {0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3};
    std::vector<int> expertIdxHostData = {1, 2, 0, 1, 0, 2};
    std::vector<float> expandedXOutHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int> expandedRowIdxOutHostData = {0, 0, 0, 0, 0, 0};
    std::vector<int> expertTokensBeforeCapacityOutHostData = {0, 0, 0};
    // Create the self aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertIdxHostData, idxShape, &expertIdxDeviceAddr, aclDataType::ACL_INT32, &expertIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the out aclTensor.
    ret = CreateAclTensor(expandedXOutHostData, expandedXOutShape, &expandedXOutDeviceAddr, aclDataType::ACL_FLOAT, &expandedXOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandedRowIdxOutHostData, idxOutShape, &expandedRowIdxOutDeviceAddr, aclDataType::ACL_INT32, &expandedRowIdxOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertTokensBeforeCapacityOutHostData, expertTokenOutShape, &expertTokenBeforeCapacityOutDeviceAddr, aclDataType::ACL_INT32, &expertTokensBeforeCapacityOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. Call the CANN operator library API. Replace it with the specific API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnMoeInitRoutingV2.
    ret = aclnnMoeInitRoutingV2GetWorkspaceSize(x, expertIdx, activeNum, expertCapacity, expertNum, dropPadMode, expertTokensCountOrCumsumFlag, expertTokensBeforeCapacityFlag, expandedXOut, expandedRowIdxOut, nullptr, expertTokensBeforeCapacityOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeInitRoutingV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnMoeInitRoutingV2.
    ret = aclnnMoeInitRoutingV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeInitRoutingV2 failed. ERROR: %d\n", ret); return ret);
    // 4. Boilerplate: wait synchronously for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value, and copy the result from device memory to host memory. Modify it according to the interface definition of the specific API.
    auto expandedXOutSize = GetShapeSize(expandedXOutShape);
    std::vector<float> expandedXOutData(expandedXOutSize, 0);
    ret = aclrtMemcpy(expandedXOutData.data(), expandedXOutData.size() * sizeof(expandedXOutData[0]), expandedXOutDeviceAddr, expandedXOutSize * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expandedXOutSize; i++) {
        LOG_PRINT("expandedXOutData[%ld] is: %f\n", i, expandedXOutData[i]);
    }
    auto expandedRowIdxOutSize = GetShapeSize(idxOutShape);
    std::vector<int> expandedRowIdxOutData(expandedRowIdxOutSize, 0);
    ret = aclrtMemcpy(expandedRowIdxOutData.data(), expandedRowIdxOutData.size() * sizeof(expandedRowIdxOutData[0]), expandedRowIdxOutDeviceAddr, expandedRowIdxOutSize * sizeof(int32_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expandedRowIdxOutSize; i++) {
        LOG_PRINT("expandedRowIdxOutData[%ld] is: %d\n", i, expandedRowIdxOutData[i]);
    }
    auto expertTokensBeforeCapacityOutSize = GetShapeSize(expertTokenOutShape);
    std::vector<int> expertTokenIdxOutData(expertTokensBeforeCapacityOutSize, 0);
    ret = aclrtMemcpy(expertTokenIdxOutData.data(), expertTokenIdxOutData.size() * sizeof(expertTokenIdxOutData[0]), expertTokenBeforeCapacityOutDeviceAddr, expertTokensBeforeCapacityOutSize * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expertTokensBeforeCapacityOutSize; i++) {
        LOG_PRINT("expertTokenIdxOutData[%ld] is: %d\n", i, expertTokenIdxOutData[i]);
    }
    // 6. Release aclTensor and aclScalar. Modify it according to the interface definition of the specific API.
    aclDestroyTensor(x);
    aclDestroyTensor(expertIdx);
    aclDestroyTensor(expandedXOut);
    aclDestroyTensor(expandedRowIdxOut);
    aclDestroyTensor(expertTokensBeforeCapacityOut);

    // 7. Release device resources. Modify it according to the interface definition of the specific API.
    aclrtFree(xDeviceAddr);
    aclrtFree(expertIdxDeviceAddr);
    aclrtFree(expandedXOutDeviceAddr);
    aclrtFree(expandedRowIdxOutDeviceAddr);
    aclrtFree(expertTokenBeforeCapacityOutDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
