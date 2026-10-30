# aclnnInplaceWeightQuantMatmulAllReduceAddRmsNorm

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T13:30:17.450Z -->

**NOTE**

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Not supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
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

**NOTE**: When using this API, ensure that the driver firmware package and the CANN package are both of the matching 8.0.RC2 version or a matching later version. Otherwise, errors such as BUS ERROR will occur.

## Function

- **Description**: Performs the in-place mm + all_reduce + add + rms_norm computation.
- **Formula**:

  $$
  mm\_out = AllReduce(x1 @ (x2*antiquantScale + antiquantOffset) + bias)
  $$

  $$
  y = mm\_out + residual
  $$

  $$
  normOut = \frac{y}{RMS(y)} * gamma, RMS(y) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} y_{i}^{2} + epsilon}
  $$

## Function Prototype

- aclnnWeightQuantMatmulAllReduceAddRmsNorm and aclnnInplaceWeightQuantMatmulAllReduceAddRmsNorm implement the same function. The differences in usage are as follows. Select the appropriate operator based on your actual scenario.

  - aclnnWeightQuantMatmulAllReduceAddRmsNorm: Creates two output tensors, normOut and y, to store the computation results.
  - aclnnInplaceWeightQuantMatmulAllReduceAddRmsNorm: Creates one output tensor normOut. The result that would be stored in the newly created output tensor y in the non-in-place scenario is directly stored in the memory of the input tensor residual.

- Each operator uses a two-phase API. Call "aclnnInplaceWeightQuantMatmulAllReduceAddRmsNormGetWorkspaceSize" first to obtain the workspace size required for computation and the executor that contains the operator computation flow. Then call "aclnnInplaceWeightQuantMatmulAllReduceAddRmsNorm" to execute the computation.

```cpp
aclnnStatus aclnnInplaceWeightQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *bias,
    const aclTensor *antiquantScale,
    const aclTensor *antiquantOffset,
    const aclTensor *residual,
    const aclTensor *gamma,
    double           epsilon,
    const char      *group,
    const char      *reduceOp,
    int64_t          commTurn,
    int64_t          streamMode,
    int64_t          antiquantGroupSize,
    const aclTensor *normOut,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnInplaceWeightQuantMatmulAllReduceAddRmsNorm(
    void              *workspace,
    uint64_t           workspaceSize,
    aclOpExecutor     *executor,
    const aclrtStream  stream)
```

## aclnnInplaceWeightQuantMatmulAllReduceAddRmsNormGetWorkspaceSize

- **Parameters:**
    <table style="undefined;table-layout: fixed; width: 1567px"><colgroup>
      <col style="width: 170px">
      <col style="width: 120px">
      <col style="width: 300px">
      <col style="width: 330px">
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
          <td>x1</td>
          <td>Input/Output</td>
          <td>Left matrix of the MatMul computation, which is x1 in the formula. It is also the output, which is y in the formula.</td>
          <td><ul><li>Empty tensors are supported.</li><li>The current version supports only 2D or 3D input.</li></ul></td>
          <td>BFLOAT16, FLOAT16</td>
          <td>ND</td>
          <td>2-3</td>
          <td>×</td>
        </tr>
        <tr>
          <td>x2</td>
          <td>Input</td>
          <td>Right matrix of the MatMul computation, which is x2 in the formula.</td>
          <td><ul><li>Supports empty tensors.</li><li>The current version supports only 2D inputs, in transposed/non-transposed scenarios.</li><li>Supports non-contiguous tensors in transposed scenarios.</li></ul></td>
          <td>INT8, INT4</td>
          <td>ND</td>
          <td>2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>bias</td>
          <td>Input</td>
          <td>Bias in the formula.</td>
          <td><ul><li>Supports scenarios where a null pointer is passed in.</li><li>The current version supports only 1D inputs.</li></ul></td>
          <td>BFLOAT16, FLOAT16</td>
          <td>ND</td>
          <td>1</td>
          <td>√</td>
        </tr>
        <tr>
          <td>antiquantScale</td>
          <td>Input</td>
          <td>antiquantScale in the formula.</td>
          <td>The shape is (1) for the pertensor scenario; (n)/(1,n) for the PerChannel scenario, where n is the size of the last dimension of x2; (ceil(k,antiquantGroupSize),n) for the pergroup scenario.</td>
          <td>BFLOAT16, FLOAT16</td>
          <td>ND</td>
          <td>1-2</td>
          <td>×</td>
        </tr>
        <tr>
          <td>antiquantOffset</td>
          <td>Input</td>
          <td>The offset parameter for the pseudo-quantization computation on x2, corresponding to antiquantOffset in the formula.</td>
          <td>Optional, can be empty. If non-empty, the shape must be consistent with antiquantScale.</td>
          <td>BFLOAT16, FLOAT16</td>
          <td>ND</td>
          <td>1-2</td>
          <td>×</td>
        </tr>
        <tr>
          <td>residual</td>
          <td>Input</td>
          <td>Residual input of the AddRmsNorm fused operator, that is, residual in the formula.</td>
          <td>The current version supports only 3D input.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>3</td>
          <td>×</td>
        </tr>
        <tr>
          <td>gamma</td>
          <td>Input</td>
          <td>RmsNorm computation input of the AddRmsNorm fused operator, that is, gamma in the formula.</td>
          <td>Only 1D input is supported in the current version.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>1</td>
          <td>×</td>
        </tr>
        <tr>
          <td>epsilon</td>
          <td>Input</td>
          <td>Host-side double-precision value used to prevent division-by-zero errors. It is epsilon in the formula.</td>
          <td>The value of epsilon must be within the value range (0,1).</td>
          <td>double</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>group</td>
          <td>Input</td>
          <td>Host-side string that identifies the communicator. It is the communicator name.</td>
          <td>Obtained through the Hccl API "extern HcclResult HcclGetCommName(HcclComm comm, char* commName);". commName is the group.</td>
          <td>string</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>reduceOp</td>
          <td>Input</td>
          <td>String on the host that identifies the operation type, indicating the reduce operation type.</td>
          <td>Currently, only "sum" is supported.</td>
          <td>string</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>commTurn</td>
          <td>Input</td>
          <td>Host-side integer. Number of data slices for communication, that is, total data volume/single communication volume.</td>
          <td>The current version supports only the value 0.</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>streamMode</td>
          <td>Input</td>
          <td>Host-side integer. Enumeration of the stream mode.</td>
          <td>Currently, only enumerated value 1 is supported.</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>antiquantGroupSize</td>
          <td>Input</td>
          <td>groupSize input for dequantization of x2 in pergroup fake-quantization mode.</td>
          <td>Pass 0 when pergroup is not supported. When supported, the value range is [32, min(k-1,INT_MAX)] and the value must be a multiple of 32. The value range of k is consistent with that of the mm API.</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>normOut</td>
          <td>Output</td>
          <td>Result of mm + all_reduce + add + rms_norm, that is, normOut in the formula.</td>
          <td><ul><li>Empty tensors are not supported.</li><li>The data type is consistent with the residual input.</li></ul></td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>2</td>
          <td>√</td>
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
          <td>Returns the op executor containing the operator computation process.</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
      </tbody>
    </table>

- **Return value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API performs input parameter validation and returns an error in the following scenarios:

    <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
    <col style="width: 250px">
    <col style="width: 130px">
    <col style="width: 650px">
    </colgroup>
    <thead>
    <tr>
        <th>Return Value</th>
        <th>Error Code</th>
        <th>Description</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>x1, x2, residual, antiquantScale, or output is a null pointer.</td>
    </tr>
    <tr>
        <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="2">161002</td>
        <td>The shape of x1, x2, residual, or output does not meet the constraints.</td>
    </tr>
    <tr>
        <td>streamMode is out of the valid range.</td>
    </tr>
    </tbody>
    </table>

## aclnnInplaceWeightQuantMatmulAllReduceAddRmsNorm

- **Parameters:**
    <table style="undefined;table-layout: fixed; width: 1312px"><colgroup>
    <col style="width: 158px">
    <col style="width: 120px">
    <col style="width: 750px">
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
        <td>Workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Workspace size allocated on the Device, obtained by the first-phase API aclnnInplaceWeightQuantMatmulAllReduceAddRmsNormGetWorkspaceSize.</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Op executor that contains the operator computation flow.</td>
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
  - aclnnInplaceWeightQuantMatmulAllReduceAddRmsNorm defaults to a non-deterministic implementation. Deterministic computation can be enabled by setting the `HCCL_DETERMINISTIC` environment variable to true.

- The scenarios are the same as those of the fused operator aclnnWeightQuantMatmulAllReduce: MC2 is not enabled in incremental scenarios and is enabled in full scenarios.
- The input x1 can be two-dimensional or three-dimensional, with its shape being (b, s, k) or (s, k). x2 must be two-dimensional, with its shape being (k, n). The axes must meet the input requirements of the mm operator, with the k axes being equal. The range of b\*s and s is [1, 2147483647], and the range of k and n is [1, 65535]. If bias is not null, bias is one-dimensional with its shape being (n).
- The input residual must be three-dimensional with shape (b, s, n). When x1 is two-dimensional, (b*s) of residual equals s of x1. The input gamma must be one-dimensional with shape (n).
- antiquantScale has shape (1) in the per-tensor scenario, shape (1,n)/(n) in the per-channel scenario, and shape (ceil(k,antiquantGroupSize),n) in the per-group scenario. If antiquantOffset is not null, its shape is the same as that of antiquantScale.
- The output y and normOut have the same dimension and data type as residual. If bias is not null, its shape size equals the size of the last dimension of normOut.
- The data type of x2 must be int8 or int4. The data types of x1, bias, residual, gamma, y, and normOut as computation inputs must be consistent.
- Only the transposed/non-transposed scenarios of the x2 matrix are supported. The x1 matrix supports the non-transposed scenario.
- The antiquantGroupSize value falls within the range [32, min(k-1, INT_MAX)] and is a multiple of 32.
- Supports 1, 2, 4, or 8 cards, and only supports HCCS all-mesh networking.
- Supports empty tensors where (b*s) and n are 0, but does not support empty tensors where k is 0.

<!-- npu="910b" id7 -->
- <term>Atlas A2 products</term>: A communication-computation fused MC2 operator in a model only supports the same communication domain.

<!-- end id7 -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include <thread>
#include "hccl/hccl.h"
#include "aclnnop/aclnn_inplace_weight_quant_matmul_all_reduce_add_rms_norm.h"

int ndev = 8;

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

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
    int64_t shapeSize = 1;
    for (auto i: shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

template<typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate device memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host data to device memory.
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

struct Args {
    uint32_t rankId;
    HcclComm hcclComm;
    aclrtStream stream;
    aclrtContext context;
};

int launchOneThreadweightQuantmatmulAllReduceAddRmsNorm(Args &args) {
    int ret;
    ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
    char hcom_name[128];
    ret = HcclGetCommName(args.hcclComm, hcom_name);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d \n", ret); return -1);
    LOG_PRINT("[INFO] rank %d hcom: %s stream: %p, context : %p\n", args.rankId, hcom_name, args.stream,
            args.context);

    std::vector<int64_t> x1Shape = {32, 64};
    std::vector<int64_t> x2Shape = {64, 128};
    std::vector<int64_t> biasShape = {128};
    std::vector<int64_t> antiquantScaleShape = {128};
    std::vector<int64_t> antiquantOffsetShape = {128};
    std::vector<int64_t> residualShape = {1, 32, 128};
    std::vector<int64_t> gammaShape = {128};
    std::vector<int64_t> yShape = {1, 32, 128};
    std::vector<int64_t> normOutShape = {1, 32, 128};
    void *x1DeviceAddr = nullptr;
    void *x2DeviceAddr = nullptr;
    void *biasDeviceAddr = nullptr;
    void *antiquantScaleDeviceAddr = nullptr;
    void *antiquantOffsetDeviceAddr = nullptr;
    void *residualDeviceAddr = nullptr;
    void *gammaDeviceAddr = nullptr;
    void *yDeviceAddr = nullptr;
    void *normOutDeviceAddr = nullptr;
    aclTensor *x1 = nullptr;
    aclTensor *x2 = nullptr;
    aclTensor *bias = nullptr;
    aclTensor *antiquantScale = nullptr;
    aclTensor *antiquantOffset = nullptr;
    aclTensor *residual = nullptr;
    aclTensor *gamma = nullptr;
    aclTensor *y = nullptr;
    aclTensor *normOut = nullptr;

    int64_t commTurn = 0;
    int64_t streamMode = 1;
    double  epsilon = 0.000001;
    int64_t antiquantGroupSize = 0;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    void *workspaceAddr = nullptr;

    long long x1ShapeSize = GetShapeSize(x1Shape);
    long long x2ShapeSize = GetShapeSize(x2Shape);
    long long biasShapeSize = GetShapeSize(biasShape);
    long long antiquantScaleShapeSize = GetShapeSize(antiquantScaleShape);
    long long antiquantOffsetShapeSize = GetShapeSize(antiquantOffsetShape);
    long long residualShapeSize = GetShapeSize(residualShape);
    long long gammaShapeSize = GetShapeSize(gammaShape);
    long long yShapeSize = GetShapeSize(yShape);
    long long normOutShapeSize = GetShapeSize(normOutShape);
    std::vector<int16_t> x1HostData(x1ShapeSize, 1);
    std::vector<int8_t> x2HostData(x2ShapeSize, 1);
    std::vector<int16_t> biasHostData(biasShapeSize, 1);
    std::vector<int16_t> antiquantScaleHostData(antiquantScaleShapeSize, 1);
    std::vector<int16_t> antiquantOffsetHostData(antiquantOffsetShapeSize, 1);
    std::vector<int16_t> residualHostData(residualShapeSize, 1);
    std::vector<int16_t> gammaHostData(gammaShapeSize, 1);
    std::vector<int16_t> yHostData(yShapeSize, 0);
    std::vector<int16_t> normOutHostData(normOutShapeSize, 0);
    // Create a tensor.
    ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT16, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(antiquantScaleHostData, antiquantScaleShape, &antiquantScaleDeviceAddr,
                        aclDataType::ACL_FLOAT16, &antiquantScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(antiquantOffsetHostData, antiquantOffsetShape, &antiquantOffsetDeviceAddr,
                        aclDataType::ACL_FLOAT16, &antiquantOffset);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(residualHostData, residualShape, &residualDeviceAddr, aclDataType::ACL_FLOAT16, &residual);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT16, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT16, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(normOutHostData, normOutShape, &normOutDeviceAddr, aclDataType::ACL_FLOAT16, &normOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Example of calling aclnnInplaceWeightQuantMatmulAllReduceAddRmsNorm.
    // Call the first-phase API.
    ret = aclnnInplaceWeightQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(x1, x2, bias, antiquantScale, antiquantOffset,
                                                        residual, gamma, epsilon, hcom_name,
                                                        "sum", commTurn, streamMode, antiquantGroupSize, normOut,
                                                        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnWeightQuantMatmulAllReduceAddRmsNormGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API.
    ret = aclnnInplaceWeightQuantMatmulAllReduceAddRmsNorm(workspaceAddr, workspaceSize, executor, args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantMatmulAllReduceAddRmsNorm failed. ERROR: %d\n", ret); return ret);
    // (Boilerplate) Synchronously wait for the task to complete.
    ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("device%d aclnnWeightQuantMatmulAllReduceAddRmsNorm execute success \n", args.rankId);
    // Release device resources. Modify it according to the specific API definition.

    if (x1 != nullptr) {
        aclDestroyTensor(x1);
    }
    if (x2 != nullptr) {
        aclDestroyTensor(x2);
    }
    if (bias != nullptr) {
        aclDestroyTensor(bias);
    }
    if (antiquantScale != nullptr) {
        aclDestroyTensor(antiquantScale);
    }
    if (antiquantOffset != nullptr) {
        aclDestroyTensor(antiquantOffset);
    }
    if (residual != nullptr) {
        aclDestroyTensor(residual);
    }
    if (gamma != nullptr) {
        aclDestroyTensor(gamma);
    }
    if (y != nullptr) {
        aclDestroyTensor(y);
    }
    if (normOut != nullptr) {
        aclDestroyTensor(normOut);
    }
    if (x1DeviceAddr != nullptr) {
        aclrtFree(x1DeviceAddr);
    }
    if (x2DeviceAddr != nullptr) {
        aclrtFree(x2DeviceAddr);
    }
    if (biasDeviceAddr != nullptr) {
        aclrtFree(biasDeviceAddr);
    }
    if (antiquantScaleDeviceAddr != nullptr) {
        aclrtFree(antiquantScaleDeviceAddr);
    }
    if (antiquantOffsetDeviceAddr != nullptr) {
        aclrtFree(antiquantOffsetDeviceAddr);
    }
    if (residualDeviceAddr != nullptr) {
        aclrtFree(residualDeviceAddr);
    }
    if (gammaDeviceAddr != nullptr) {
        aclrtFree(gammaDeviceAddr);
    }
    if (yDeviceAddr != nullptr) {
        aclrtFree(yDeviceAddr);
    }
    if (normOutDeviceAddr != nullptr) {
        aclrtFree(normOutDeviceAddr);
    }
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(args.stream);
    HcclCommDestroy(args.hcclComm);
    aclrtDestroyContext(args.context);
    aclrtResetDevice(args.rankId);
    return 0;
}

int main(int argc, char *argv[]) {
    int ret;
    int32_t devices[ndev];
    for (int i = 0; i < ndev; i++) {
        devices[i] = i;
    }
    HcclComm comms[128];
    ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    // Initialize the communicator.
    for (int i = 0; i < ndev; i++) {
        ret = aclrtSetDevice(devices[i]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    }
    ret = HcclCommInitAll(ndev, devices, comms);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("HcclCommInitAll failed. ERROR: %d\n", ret); return ret);
    Args args[ndev];
    aclrtStream stream[ndev];
    aclrtContext context[ndev];
    for (uint32_t rankId = 0; rankId < ndev; rankId++) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
        ret = aclrtCreateContext(&context[rankId], rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
        ret = aclrtCreateStream(&stream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    }
    // Start multiple threads.
    std::vector<std::unique_ptr<std::thread>> threads(ndev);
    for (uint32_t rankId = 0; rankId < ndev; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].hcclComm = comms[rankId];
        args[rankId].stream = stream[rankId];
        args[rankId].context = context[rankId];
        threads[rankId].reset(
                new(std::nothrow) std::thread(&launchOneThreadweightQuantmatmulAllReduceAddRmsNorm, std::ref(args[rankId])));
    }
    for (uint32_t rankId = 0; rankId < ndev; rankId++) {
        threads[rankId]->join();
    }
    aclFinalize();
    return 0;
}
```
