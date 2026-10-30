# aclnnInplaceQuantMatmulAllReduceAddRmsNorm

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T13:29:38.785Z -->

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

- Description: Performs in-place mm + all_reduce + add + rms_norm computation.
- Formula:

  $$
  mm_out = allReduce(dequantScale * (x1_{int8}@x2_{int8} + bias_{int32}))
  $$

  $$
  y = mm_out + residual
  $$

  $$
  normOut = \frac{y}{RMS(y)} * gamma, RMS(y) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} y_{i}^{2} + epsilon}
  $$

## Function Prototype

- aclnnQuantMatmulAllReduceAddRmsNorm and aclnnInplaceQuantMatmulAllReduceAddRmsNorm implement the same functionality. The differences in usage are as follows. Select the appropriate operator based on your actual scenario.

  - aclnnQuantMatmulAllReduceAddRmsNorm: Two output tensors, normOut and y, need to be created to store the computation results.
  - aclnnInplaceQuantMatmulAllReduceAddRmsNorm: One output tensor normOut needs to be created. In non-in-place scenarios, the result stored in the newly created output tensor y is directly stored into the memory of the input tensor residual.

- Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnInplaceQuantMatmulAllReduceAddRmsNormGetWorkspaceSize" API to obtain the workspace size required for computation and the executor containing the operator computation flow, and then call the "aclnnInplaceQuantMatmulAllReduceAddRmsNorm" API to perform the computation.

```cpp
aclnnStatus aclnnInplaceQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *bias,
    const aclTensor *dequantScale,
    const aclTensor *residual,
    const aclTensor *gamma,
    double           epsilon,
    const char      *group,
    const char      *reduceOp,
    int64_t          commTurn,
    int64_t          streamMode,
    const aclTensor *normOut,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnInplaceQuantMatmulAllReduceAddRmsNorm(
    void              *workspace,
    uint64_t           workspaceSize,
    aclOpExecutor     *executor,
    const aclrtStream  stream)
```

## aclnnInplaceQuantMatmulAllReduceAddRmsNormGetWorkspaceSize

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
          <td>x1 (aclTensor)</td>
          <td>Input</td>
          <td>Left matrix for the MatMul computation, that is, x1 in the formula, and also the output, that is, y in the formula.</td>
          <td><ul><li>Supports empty tensors.</li><li>Must be of the same data type as x2.</li><li>The current version supports only two-dimensional or three-dimensional inputs.</li></ul></td>
          <td>INT8</td>
          <td>ND</td>
          <td>2-3</td>
          <td>×</td>
        </tr>
        <tr>
          <td>x2 (aclTensor)</td>
          <td>Input</td>
          <td>Right matrix for the MatMul computation, that is, x2 in the formula.</td>
          <td><ul><li>Supports empty tensors.</li><li>Consistent with the data type of x1.</li><li>The current version supports only two-dimensional input, for both transposed and non-transposed use cases.</li><li>Supports non-contiguous tensors in the transposed use case.</li></ul></td>
          <td>INT8</td>
          <td>ND</td>
          <td>2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>bias (aclTensor)</td>
          <td>Input</td>
          <td>bias in the formula.</td>
          <td><ul><li>Supports a null pointer.</li><li>The current version supports only one-dimensional input.</li></ul></td>
          <td>INT32</td>
          <td>ND</td>
          <td>1</td>
          <td>√</td>
        </tr>
        <tr>
          <td>dequantScale (aclTensor)</td>
          <td>Input</td>
          <td>Dequantization scale after the MatMul computation, that is, dequantScale in the formula.</td>
          <td>The shape is (1) in the pertensor use case, and (n) or (1, n) in the perchannel use case.</td>
          <td>UINT64, INT64, BFLOAT16</td>
          <td>ND</td>
          <td>1-2</td>
          <td>×</td>
        </tr>
        <tr>
          <td>residual (aclTensor)</td>
          <td>Input</td>
          <td>Residual input of the AddRmsNorm fused operator, corresponding to residual in the formula.</td>
          <td>In the inplace use case, residual serves as the output address of y. Only three-dimensional input is supported in the current version.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>3</td>
          <td>×</td>
        </tr>
        <tr>
          <td>gamma (aclTensor)</td>
          <td>Input</td>
          <td>RmsNorm calculation input of the AddRmsNorm fused operator, which is gamma in the formula.</td>
          <td>Only 1-D input is supported in the current version.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>1</td>
          <td>×</td>
        </tr>
        <tr>
          <td>epsilon (double)</td>
          <td>Input</td>
          <td>Double precision on the Host, used to prevent division by 0, which is epsilon in the formula.</td>
          <td>The value of epsilon must be in the range (0,1).</td>
          <td>Double</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>group (char)</td>
          <td>Input</td>
          <td>String on the host that identifies the communicator, that is, the communicator name.</td>
          <td>Obtained through the Hccl API "extern HcclResult HcclGetCommName(HcclComm comm, char* commName);", where commName is the group.</td>
          <td>String</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>reduceOp (char)</td>
          <td>Input</td>
          <td>String on the host that identifies the operation type. The value is a reduce operation type.</td>
          <td>Currently, only "sum" is supported.</td>
          <td>String</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>commTurn (int64_t)</td>
          <td>Input</td>
          <td>Number of communication data splits, that is, total data volume/single communication volume.</td>
          <td>Only 0 is supported in the current version.</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>streamMode (int64_t)</td>
          <td>Input</td>
          <td>Enumeration of the stream mode.</td>
          <td>Currently, only enumerated value 1 is supported.</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>normOut (aclTensor)</td>
          <td>Output</td>
          <td>Result of mm + all_reduce + add + rms_norm, that is, normOut in the formula.</td>
          <td><ul><li>Empty tensors are not supported.</li><li>The data type is the same as that of the residual input.</li></ul></td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>workspaceSize (uint64_t)</td>
          <td>Output</td>
          <td>Returns the workspace size to be requested on the Device side.</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>executor (aclOpExecutor)</td>
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
        <td>x1, x2, residual, gamma, or dequantScale is a null pointer.</td>
    </tr>
    <tr>
        <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="3">161002</td>
        <td>The shapes of x1, x2, and residual do not meet the constraint requirements.</td>
    </tr>
    <tr>
        <td>streamMode is not within the valid range.</td>
    </tr>
    <tr>
        <td>The size k of the last dimension of x1 is 0.</td>
    </tr>
    </tbody>
    </table>

## aclnnInplaceQuantMatmulAllReduceAddRmsNorm

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
        <td>Address of the workspace memory requested on the Device.</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Size of the workspace requested on the Device, obtained by calling the first-phase API aclnnInplaceQuantMatmulAllReduceAddRmsNormGetWorkspaceSize.</td>
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
    </tbody></table>

- **Return value**

    Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnInplaceQuantMatmulAllReduceAddRmsNorm defaults to a non-deterministic implementation. Deterministic computation can be enabled by setting the `HCCL_DETERMINISTIC` environment variable to true.

- The use case is the same as that of the fused operator aclnnQuantMatmulAllReduce: MC2 is not enabled for the incremental use case and is enabled for the full use case.
- The input x1 can be two-dimensional or three-dimensional, with its shape being (b, s, k) or (m, k). x2 must be two-dimensional, with its shape being (k, n). The axes must meet the input requirements of the mm operator, with the k axes being equal. If bias is not empty, bias is one-dimensional with its shape being (n).
- The size of m must not exceed 2147483647. The last dimension size of x1 and x2 must not exceed 65535. The last dimension of x1 refers to k, and the last dimension of x2 refers to k when transposed or n when not transposed.
- The input residual must be three-dimensional with the shape (b, s, n). When x1 is two-dimensional, (b*s) of residual equals m of x1. The input gamma has the shape (n).
- The output y and normOut have the same dimensions and data type as residual, with the shape (b, s, n).
- If the output residual type is FLOAT16, the type of dequantScale is UINT64 or INT64. If the output residual type is BFLOAT16, the type of dequantScale is BFLOAT16.
- The data type of x1 and x2 is INT8. The data type of bias is INT32. The data types of residual, gamma, y, and normOut as computation inputs must be consistent.
- Only the transposed/non-transposed scenarios of the x2 matrix are supported. The x1 matrix supports the non-transposed scenario.
- Supports 1, 2, 4, or 8 cards, and only supports HCCS all-mesh networking.
- Supports empty tensors where (b*s) and n are 0, but does not support empty tensors where k is 0.

<!-- npu="910b" id7 -->
- <term>Atlas A2 products</term>: A communication-computation fused MC2 operator in a model only supports the same communication domain.

<!-- end id7 -->

## Example

The following sample code is for reference only. For details about the compilation and execution procedure, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include <thread>
#include <cstdio>
#include "hccl/hccl.h"
#include "aclnnop/aclnn_inplace_quant_matmul_all_reduce_add_rms_norm.h"

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
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host data to the device memory.
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

int launchOneThreadQuantMatmulAllReduceAddRmsNorm(Args &args) {
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
    std::vector<int64_t> dequantScaleShape = {128};
    std::vector<int64_t> residualShape = {1, 32, 128};
    std::vector<int64_t> gammaShape = {128};
    std::vector<int64_t> yShape = {1, 32, 128};
    std::vector<int64_t> normOutShape = {1, 32, 128};
    void *x1DeviceAddr = nullptr;
    void *x2DeviceAddr = nullptr;
    void *biasDeviceAddr = nullptr;
    void *dequantScaleDeviceAddr = nullptr;
    void *residualDeviceAddr = nullptr;
    void *gammaDeviceAddr = nullptr;
    void *yDeviceAddr = nullptr;
    void *normOutDeviceAddr = nullptr;
    aclTensor *x1 = nullptr;
    aclTensor *x2 = nullptr;
    aclTensor *bias = nullptr;
    aclTensor *dequantScale = nullptr;
    aclTensor *residual = nullptr;
    aclTensor *gamma = nullptr;
    aclTensor *y = nullptr;
    aclTensor *normOut = nullptr;

    int64_t commTurn = 0;
    int64_t streamMode = 1;
    double  epsilon = 0.000001;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    void *workspaceAddr = nullptr;

    long long x1ShapeSize = GetShapeSize(x1Shape);
    long long x2ShapeSize = GetShapeSize(x2Shape);
    long long biasShapeSize = GetShapeSize(biasShape);
    long long dequantScaleShapeSize = GetShapeSize(dequantScaleShape);
    long long residualShapeSize = GetShapeSize(residualShape);
    long long gammaShapeSize = GetShapeSize(gammaShape);
    long long yShapeSize = GetShapeSize(yShape);
    long long normOutShapeSize = GetShapeSize(normOutShape);

    std::vector<int8_t> x1HostData(x1ShapeSize, 1);
    std::vector<int8_t> x2HostData(x2ShapeSize, 1);
    std::vector<int32_t> biasHostData(biasShapeSize, 1);
    std::vector<uint64_t> dequantScaleHostData(dequantScaleShapeSize, 1);
    std::vector<int16_t> residualHostData(residualShapeSize, 1);
    std::vector<int16_t> gammaHostData(gammaShapeSize, 1);
    std::vector<int16_t> yHostData(yShapeSize, 0);
    std::vector<int16_t> normOutHostData(normOutShapeSize, 0);
    // Create tensors.
    ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_INT32, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dequantScaleHostData, dequantScaleShape, &dequantScaleDeviceAddr,
                        aclDataType::ACL_UINT64, &dequantScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(residualHostData, residualShape, &residualDeviceAddr, aclDataType::ACL_FLOAT16, &residual);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT16, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT16, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(normOutHostData, normOutShape, &normOutDeviceAddr, aclDataType::ACL_FLOAT16, &normOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Example of calling the aclnnInplaceQuantMatmulAllReduceAddRmsNorm API.
    // Call the first-phase API.
    ret = aclnnInplaceQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(x1, x2, bias, dequantScale, residual, gamma, epsilon,
                                        hcom_name, "sum", commTurn, streamMode, normOut,
                                        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnInplaceQuantMatmulAllReduceAddRmsNormGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API.
    ret = aclnnInplaceQuantMatmulAllReduceAddRmsNorm(workspaceAddr, workspaceSize, executor, args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceQuantMatmulAllReduceAddRmsNorm failed. ERROR: %d\n", ret); return ret);
    // (Boilerplate) Synchronize and wait for the task execution to complete.
    ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("device%d aclnnInplaceQuantMatmulAllReduceAddRmsNorm execute success \n", args.rankId);
    // Release device resources. Modify the code based on the specific API definition.

    if (x1 != nullptr) {
        aclDestroyTensor(x1);
    }
    if (x2 != nullptr) {
        aclDestroyTensor(x2);
    }
    if (bias != nullptr) {
        aclDestroyTensor(bias);
    }
    if (dequantScale != nullptr) {
        aclDestroyTensor(dequantScale);
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
    if (dequantScaleDeviceAddr != nullptr) {
        aclrtFree(dequantScaleDeviceAddr);
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
                new(std::nothrow) std::thread(&launchOneThreadQuantMatmulAllReduceAddRmsNorm, std::ref(args[rankId])));
    }
    for (uint32_t rankId = 0; rankId < ndev; rankId++) {
        threads[rankId]->join();
    }
    aclFinalize();
    return 0;
}
```
