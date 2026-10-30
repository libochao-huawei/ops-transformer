# aclnnAttentionUpdate

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T08:03:07.805Z -->

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

- Description: Updates the intermediate results lse and localOut output by the PA operator in each SP domain into global results.
- Formula:

$$
lse_{max} = \text{max}lse_i
$$

$$
lse = \sum_i \text{exp}(lse_i - lse_{max})
$$

$$
lse_m = lse_{max} + \text{log}(lse)
$$

$$
O = \sum_i O_i \cdot \text{exp}(lse_i - lse_m)
$$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnAttentionUpdateGetWorkspaceSize" API first to obtain the workspace size required for the computation and the executor that contains the operator computation flow. Then call the "aclnnAttentionUpdate" API to perform the computation.

```c++
aclnnStatus aclnnAttentionUpdateGetWorkspaceSize(
   const aclTensorList  *lse,
   const aclTensorList  *localOut,
   int64_t              updateType,
   aclTensor            *out,
   aclTensor            *lseOut,
   uint64_t             *workspaceSize,
   aclOpExecutor        **executor)
```

```c++
aclnnStatus aclnnAttentionUpdate(
   void          *workspace,
   uint64_t       workspaceSize,
   aclOpExecutor *executor,
   aclrtStream    stream)
```

## aclnnAttentionUpdateGetWorkspaceSize

- **Parameters**

  <table style="undefined; table-layout: fixed; width: 1567px">
    <colgroup>
      <col style="width: 170px"> <!-- Parameter -->
      <col style="width: 120px"> <!-- Input/Output -->
      <col style="width: 300px"> <!-- Description -->
      <col style="width: 330px"> <!-- Instruction -->
      <col style="width: 212px"> <!-- Data Type -->
      <col style="width: 100px"> <!-- Data Format -->
      <col style="width: 190px"> <!-- Dimensions (shape) -->
      <col style="width: 145px"> <!-- Non-contiguous Tensor -->
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
        <td>lse</td>
        <td>Input</td>
        <td>Local lse of each SP domain.</td>
        <td>The length of tensorList is sp.</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>[batch * seqLen * headNum]</td>
        <td>x</td>
      </tr>
      <tr>
        <td>localOut</td>
        <td>Input</td>
        <td>Local attentionout of each SP domain.</td>
        <td>The length of tensorList is sp.</td>
        <td>FLOAT32, FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>[batch * seqLen * headNum, headDim]</td>
        <td>x</td>
      </tr>
      <tr>
        <td>updateType</td>
        <td>Input</td>
        <td>Controls whether lseOut is output.</td>
        <td>0 and 1, indicating that lseOut is not output or is output, respectively.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out</td>
        <td>Output</td>
        <td>Output tensor.</td>
        <td>          -
        </td>
        <td>Same as localOut</td>
        <td>ND</td>
        <td>[batch * seqLen * headNum, headDim]</td>
        <td>x</td>
      </tr>
      <tr>
        <td>lseOut</td>
        <td>Optional output</td>
        <td>Serves as the optional output of lse_m.</td>
        <td>If lseOut is not output, pass in nullptr.</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>[batch * seqLen * headNum]</td>
        <td>x</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the workspace size to be allocated on the Device.</td>
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
    </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <div style="overflow-x: auto;">
    <table style="table-layout: fixed; width: 1100px">
      <colgroup>
        <col style="width: 250px">
        <col style="width: 130px">
        <col style="width: 720px">
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
          <td>lse, localOut, or out is a null pointer.</td>
        </tr>
        <tr>
          <!-- Add the merged-cell class to merged cells to keep consistency with the template structure -->
          <td class="merged-cell" rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
          <td class="merged-cell" rowspan="5">161002</td>
          <td>The data type/data format of lse, localOut, or out is not within the supported range.</td>
        </tr>
        <tr>
          <td>updateType or sp is out of the value range.</td>
        </tr>
        <tr>
          <td>The shape of lse, localOut, or out does not meet the constraints.</td>
        </tr>
        <tr>
          <td>When updateType is 0, lseOut is not nullptr.</td>
        </tr>
        <tr>
          <td>When updateType is 1, lseOut is nullptr.</td>
        </tr>
      </tbody>
    </table>
  </div>

## aclnnAttentionUpdate

- **Parameters**

  <table><thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Workspace memory address allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the device. The value is obtained by the first-phase API aclnnAttentionUpdateGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, containing the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnAttentionUpdate defaults to a deterministic implementation.
- Value range of the sequence parallelism degree sp:

  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: [1, 128].
  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT products</term>: [1, 16].
  <!-- end id8 -->

- headDim ranges from 8 to 512 and is a multiple of 8.
- Empty tensors are supported.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_attention_update.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
  do {                                     \
      if (!(cond)) {                       \
          Finalize(deviceId, stream);      \
          return_expr;                     \
      }                                    \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // Boilerplate for initialization.
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
  // Call aclrtMalloc to allocate device-side memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host-side data to the device-side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the stride of a contiguous tensor.
  std::vector<int64_t> stride(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    stride[i] = shape[i + 1] * stride[i + 1];
  }

  // Call the aclCreateTensor API to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, stride.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
}

aclnnStatus aclnnAttentionUpdateTest(int32_t deviceId, aclrtStream& stream) {
  auto ret = Init(deviceId, &stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs. Customize the construction based on the API.

  std::vector<int64_t> lseShape = {256};
  std::vector<int64_t> localOutShape = {256, 128};
  std::vector<int64_t> outShape = {256, 128};

  int64_t updateType = 0;

  void* lseDeviceAddr[2] = {nullptr, nullptr};
  void* localOutDeviceAddr[2] = {nullptr, nullptr};
  void* outDeviceAddr = nullptr;

  std::vector<aclTensor*> lse = {nullptr, nullptr};
  std::vector<aclTensor*> localOut = {nullptr, nullptr};
  aclTensor* out = nullptr;

  std::vector<float> lse1HostData(GetShapeSize(lseShape), 1);
  std::vector<float> lse2HostData(GetShapeSize(lseShape), 1);
  std::vector<float> localOut1HostData(GetShapeSize(localOutShape), 1);
  std::vector<float> localOut2HostData(GetShapeSize(localOutShape), 1);
  std::vector<float> outHostData(GetShapeSize(outShape), 0);

  // Create the lse aclTensorList.
  ret = CreateAclTensor(lse1HostData, lseShape, &(lseDeviceAddr[0]), aclDataType::ACL_FLOAT, &(lse[0]));
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> lse1TensorPtr(lse[0], aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> lse1DeviceAddrPtr(lseDeviceAddr[0], aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(lse2HostData, lseShape, &(lseDeviceAddr[1]), aclDataType::ACL_FLOAT, &(lse[1]));
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> lse2TensorPtr(lse[1], aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> lse2DeviceAddrPtr(lseDeviceAddr[1], aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  aclTensorList *lseList = aclCreateTensorList(lse.data(), lse.size());

  // Create the localOut aclTensorList.
  ret = CreateAclTensor(localOut1HostData, localOutShape, &(localOutDeviceAddr[0]), aclDataType::ACL_FLOAT, &(localOut[0]));
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> localOut1TensorPtr(localOut[0], aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> localOut1DeviceAddrPtr(localOutDeviceAddr[0], aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(localOut2HostData, localOutShape, &(localOutDeviceAddr[1]), aclDataType::ACL_FLOAT, &(localOut[1]));
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> localOut2TensorPtr(localOut[1], aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> localOut2DeviceAddrPtr(localOutDeviceAddr[1], aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  aclTensorList *localOutList = aclCreateTensorList(localOut.data(), localOut.size());

  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace Api with the actual API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnAttentionUpdate.
  ret = aclnnAttentionUpdateGetWorkspaceSize(lseList,
                                       localOutList,
                                       updateType,
                                       out,
                                       nullptr,
                                       &workspaceSize,
                                       &executor);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAttentionUpdateGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Request device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    workspaceAddrPtr.reset(workspaceAddr);
  }
  // Call the second-phase API of aclnnAttentionUpdate.
  ret = aclnnAttentionUpdate(workspaceAddr, workspaceSize, executor, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAttentionUpdate failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Get the output value. Copy the result from device memory to the host side. Modify the code based on the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> outData(size, 0);
  ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(outData[0]), outDeviceAddr,
                    size * sizeof(outData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("out result[%ld] is: %f\n", i, outData[i]);
  }

  return ACL_SUCCESS;
}

int main() {
  // 1. (boilerplate) Initialize the device/stream. For details, see the list of external APIs.
  // Fill in deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = aclnnAttentionUpdateTest(deviceId, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAttentionUpdateTest failed. ERROR: %d\n", ret); return ret);

  Finalize(deviceId, stream);
  return 0;
}
```
