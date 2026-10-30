# aclnnSwigluGatedMlp

## Applicable Products

<!-- npu="950" id6 -->
- <term>Ascend 950PR&950DT products</term>: Not Supported
<!-- end id6 -->
<!-- npu="A3" id1 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id2 -->
<!-- npu="310b" id3 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not Supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas inference products</term>: Not Supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- <term>Atlas training products</term>: Not Supported
<!-- end id5 -->

For details about product forms, see [Ascend Product Form Description](https://www.hiascend.com/document/redirect/CannCommunityProductForm).

## Function

- Operator function: Performs fused SwiGLU gated MLP computation. This operator fuses the first matrix multiplication, SwiGLU activation, and the second matrix multiplication into a two-phase aclnn API.
- Formula:

  $$
  gate\_up = x * gate\_up\_weight
  $$

  $$
  hidden = swiglu(gate\_up)
  $$

  $$
  y = hidden * down\_weight
  $$

  Here, `gate_up` is split equally into two parts along the last dimension, and the SwiGLU activation is computed as:

  $$
  swiglu(gate\_up)=silu(gate) * up
  $$

  `gate` represents the first half of `gate_up`, and `up` represents the second half of `gate_up`.

## Implementation Principle

SwigluGatedMlp consists mainly of two MatMul operations and one SwiGLU activation. The computation proceeds in three steps:

1. `gate_up = MatMul(x, gate_up_weight)`: Performs the first matrix multiplication. If the shape of `x` is `[..., hiddenSize]`, it is internally treated as a 2D matrix `[rows, hiddenSize]`, where `rows` is the product of all dimensions of `x` except the last one.
2. `hidden = SwiGLU(gate_up)`: Splits `gate_up` equally into two parts along the last dimension, computes `silu(gate) * up`, and outputs the intermediate result `hidden`.
3. `y = MatMul(hidden, down_weight)`: Performs the second matrix multiplication and reshapes the result to the shape of the user output `y`.

The following code example shows the correspondence between the small operator and the SwigluGatedMlp fused operator:

```python
# Small operator
gate_up = MatMul(x, gate_up_weight)
gate, up = Split(gate_up, split_num=2, axis=-1)
hidden = Silu(gate) * up
y = MatMul(hidden, down_weight)

# Fused operator
y = SwigluGatedMlp(x, gate_up_weight, down_weight, cube_math_type)
```

## Operator Execution APIs

Each operator uses a [two-phase API](../../../../docs/en/context/two_phase_api.md). You must first call the "aclnnSwigluGatedMlpGetWorkspaceSize" API to obtain the workspace size required for computation and the executor containing the operator computation flow, and then call the "aclnnSwigluGatedMlp" API to execute the computation.

- `aclnnStatus aclnnSwigluGatedMlpGetWorkspaceSize(const aclTensor* x, const aclTensor* gateUpWeight, const aclTensor* downWeight, int64_t cubeMathType, aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnSwigluGatedMlp(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

**Note**:

- The operator execution API hides the internal implementation logic of the operator and the differences between NPU generations. Developers do not need to compile the operator, enabling simplified operator invocation.
- If developers do not use the operator execution API to invoke the operator, they can also define an operator description file based on Ascend IR, compile it using the ATC tool to obtain the operator om file, and then load the model file to execute the operator. For details, see the single operator call > single operator model execution section in the [Application Development Guide](https://hiascend.com/en/document/redirect/CannCommunityCppOpcall).

### aclnnSwigluGatedMlpGetWorkspaceSize

- **Parameters:**

  - x (aclTensor\*, computation input): Required parameter. An aclTensor on the Device side, corresponding to input `x` in the formula. Data types supported are FLOAT16, FLOAT, BFLOAT16. The [data format](../../../../docs/en/context/data_format.md) supports ND. The shape is `[..., hiddenSize]`, and the number of dimensions must be greater than or equal to 2.
  - gateUpWeight (aclTensor\*, computation input): Required parameter. An aclTensor on the Device side, representing the weight of the first MatMul, i.e., `gate_up_weight` in the formula. The data type must be the same as `x`. Supports FLOAT16, FLOAT, BFLOAT16. The [data format](../../../../docs/en/context/data_format.md) supports ND. The shape is `[hiddenSize, 2 * intermediateSize]`.
  - downWeight (aclTensor\*, computation input): Required parameter. An aclTensor on the Device side, representing the weight of the second MatMul, i.e., `down_weight` in the formula. The data type must be the same as `x`. Supports FLOAT16, FLOAT, BFLOAT16. The [data format](../../../../docs/en/context/data_format.md) supports ND. The shape is `[intermediateSize, outSize]`.
  - cubeMathType (int64\_t, computation input): Host-side attribute value indicating the MatMul computation mode. Currently supports values 0 and 1. The default value is 1.
  - y (aclTensor\*, computation output): Required parameter. An aclTensor on the Device side, corresponding to output `y` in the formula. The data type must be the same as `x`. Supports FLOAT16, FLOAT, BFLOAT16. The [data format](../../../../docs/en/context/data_format.md) supports ND. The shape is `[..., outSize]`.
  - workspaceSize (uint64\_t\*, output): Returns the workspace size that the user needs to allocate on the Device side.
  - executor (aclOpExecutor\*\*, output): Returns the op executor, which contains the operator computation flow.

  > **Note:**
  > `hiddenSize` indicates the size of the last dimension of input `x`; `intermediateSize` indicates the size of the last dimension of the intermediate result after SwiGLU activation; `outSize` indicates the size of the last dimension of output `y`. The second dimension of `gateUpWeight` must equal `2 * intermediateSize`, and the first dimension of `downWeight` must equal `intermediateSize`.

- **Return Value:**

  Returns the aclnnStatus code. For details, see [aclnn Return Codes](../../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs parameter validation. If the following error codes are returned, the corresponding causes are:

  - Returns 161001 (ACLNN_ERR_PARAM_NULLPTR): If a required input, output, or output parameter is a null pointer, 161001 is returned.
  - Returns 161002 (ACLNN_ERR_PARAM_INVALID): If the data type, shape relationship of the input or output, or the value of `cubeMathType` is outside the supported range.

### aclnnSwigluGatedMlp

- **Parameters:**

  - workspace (void\*, input): The workspace memory address allocated on the Device side.
  - workspaceSize (uint64\_t, input): The workspace size allocated on the Device side, obtained from the first-phase API aclnnSwigluGatedMlpGetWorkspaceSize.
  - executor (aclOpExecutor\*, input): The op executor, which contains the operator computation flow.
  - stream (aclrtStream, input): Specifies the Stream for executing the task.

- **Return Value:**

  Returns the aclnnStatus code. For details, see [aclnn Return Codes](../../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnSwigluGatedMlp uses a deterministic implementation by default. The deterministic computation configuration does not affect the operator execution result.
- `x`, `gateUpWeight`, `downWeight`, and `y` do not support null pointers.
- The data types of `x`, `gateUpWeight`, `downWeight`, and `y` must be consistent. Currently supports FLOAT16, FLOAT, and BFLOAT16.
- The number of dimensions of `x` must be greater than or equal to 2. `gateUpWeight` and `downWeight` must be 2D.
- The 0th dimension of `gateUpWeight` must equal the last dimension of `x`.
- The 1st dimension of `gateUpWeight` must be a positive number and divisible by 2.
- The 0th dimension of `downWeight` must equal half of the 1st dimension of `gateUpWeight`.
- The number of dimensions of `y` must match that of `x`. All dimensions of `y` except the last must match those of `x`. The last dimension of `y` must equal the 1st dimension of `downWeight`.
- `cubeMathType` currently only supports values 0 and 1.

## Operator Prototype

```c++
REG_OP(SwigluGatedMlp)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(gate_up_weight, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(down_weight, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .ATTR(cube_math_type, Int, 1)
    .OP_END_FACTORY_REG(SwigluGatedMlp)
```

For parameter explanations, see **Operator Execution APIs**.

## Example

The following call example code is provided for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <cstdio>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_swiglu_gated_mlp.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    std::printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto dim : shape) {
    shapeSize *= dim;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // Fixed code for resource initialization
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
  auto size = GetShapeSize(shape) * aclDataTypeSize(dataType);
  // Call aclrtMalloc to allocate device-side memory
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy host-side data to device-side memory
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute strides for a contiguous tensor
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call aclCreateTensor API to create an aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Fixed code) Device/Stream initialization. Refer to the AscendCL external API list.
  // Fill in the deviceId based on your actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check according to your needs
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs. Customize according to the API interface definition.
  std::vector<int64_t> xShape = {2, 3};
  std::vector<int64_t> gateUpWeightShape = {3, 8};
  std::vector<int64_t> downWeightShape = {4, 2};
  std::vector<int64_t> yShape = {2, 2};

  void* xDeviceAddr = nullptr;
  void* gateUpWeightDeviceAddr = nullptr;
  void* downWeightDeviceAddr = nullptr;
  void* yDeviceAddr = nullptr;

  aclTensor* x = nullptr;
  aclTensor* gateUpWeight = nullptr;
  aclTensor* downWeight = nullptr;
  aclTensor* y = nullptr;

  std::vector<float> xHostData = {
    0.1F, -0.2F, 0.3F,
    0.4F, 0.5F, -0.6F
  };
  std::vector<float> gateUpWeightHostData = {
    0.1F, 0.2F, 0.3F, 0.4F, -0.1F, -0.2F, 0.5F, 0.6F,
    0.2F, 0.1F, -0.4F, 0.3F, 0.7F, -0.3F, 0.2F, 0.1F,
    -0.5F, 0.4F, 0.2F, -0.1F, 0.3F, 0.8F, -0.2F, 0.5F
  };
  std::vector<float> downWeightHostData = {
    0.5F, -0.1F,
    0.2F, 0.4F,
    -0.3F, 0.6F,
    0.7F, 0.1F
  };
  std::vector<float> yHostData(GetShapeSize(yShape), 0.0F);

  // Create x aclTensor
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create gateUpWeight aclTensor
  ret = CreateAclTensor(gateUpWeightHostData, gateUpWeightShape, &gateUpWeightDeviceAddr,
                        aclDataType::ACL_FLOAT, &gateUpWeight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create downWeight aclTensor
  ret = CreateAclTensor(downWeightHostData, downWeightShape, &downWeightDeviceAddr, aclDataType::ACL_FLOAT,
                        &downWeight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create y aclTensor
  ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnSwigluGatedMlp API call example
  LOG_PRINT("test aclnnSwigluGatedMlp\n");

  // 3. Call the CANN operator library API. Replace with the specific API name.
  // Call the first-phase API of aclnnSwigluGatedMlp
  int64_t cubeMathType = 1;
  ret = aclnnSwigluGatedMlpGetWorkspaceSize(x, gateUpWeight, downWeight, cubeMathType, y, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnSwigluGatedMlpGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspace size calculated by the first-phase API
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnSwigluGatedMlp
  ret = aclnnSwigluGatedMlp(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluGatedMlp failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed code) Wait synchronously for task completion
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Get the output values. Copy the result from device-side memory to host side.
  // Modify according to the specific API interface definition.
  auto outputSize = GetShapeSize(yShape);
  std::vector<float> resultData(outputSize, 0.0F);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr,
                    outputSize * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < outputSize; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor. Modify according to the specific API interface definition.
  aclDestroyTensor(x);
  aclDestroyTensor(gateUpWeight);
  aclDestroyTensor(downWeight);
  aclDestroyTensor(y);

  // 7. Release device resources
  aclrtFree(xDeviceAddr);
  aclrtFree(gateUpWeightDeviceAddr);
  aclrtFree(downWeightDeviceAddr);
  aclrtFree(yDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
