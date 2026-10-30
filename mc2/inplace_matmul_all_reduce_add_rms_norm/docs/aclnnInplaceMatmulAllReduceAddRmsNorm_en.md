# aclnnInplaceMatmulAllReduceAddRmsNorm

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T13:28:01.748Z -->

**NOTE: This API will be deprecated in later versions. Use aclnnMatmulAllReduce and aclnnAddRmsNorm instead.**

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
    mm\_out = all_reduce(x1 @ x2 + bias)
    $$

    $$
    y = mm\_out + residual
    $$

    $$
    normOut = \frac{y}{RMS(y)} * gamma, RMS(y) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} y_{i}^{2} + epsilon}
    $$

## Function Prototype

- aclnnMatmulAllReduceAddRmsNorm and aclnnInplaceMatmulAllReduceAddRmsNorm implement the same function. The differences in usage are as follows. Select the appropriate operator based on your actual scenario.

- aclnnMatmulAllReduceAddRmsNorm: Creates two output tensors, normOut and y, to store the computation results.
- aclnnInplaceMatmulAllReduceAddRmsNorm: Creates one output tensor, normOut. The result that would be stored in the newly created output tensor y in the non-inplace scenario is directly stored in the memory of the input tensor residual.

Each operator is a two-phase API. Call the "aclnnInplaceMatmulAllReduceAddRmsNormGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor that contains the computation process of the operator. Then call the "aclnnInplaceMatmulAllReduceAddRmsNorm" API to perform computation.

```cpp
aclnnStatus aclnnInplaceMatmulAllReduceAddRmsNormGetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *bias,
    const aclTensor *residual,
    const aclTensor *gamma,
    double          epsilon,
    const char*     group,
    const char      *reduceOp,
    int64_t         commTurn,
    int64_t         streamMode,
    const aclTensor *normOut,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnInplaceMatmulAllReduceAddRmsNorm(
    void                *workspace,
    uint64_t            workspaceSize,
    aclOpExecutor       *executor,
    const aclrtStream   stream)
```

## aclnnInplaceMatmulAllReduceAddRmsNormGetWorkspaceSize

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
          <td>Input</td>
          <td>Left matrix of the MatMul computation, i.e., x1 in the formula. It is also the output, i.e., y in the formula.</td>
          <td><ul><li>Empty tensor is supported.</li><li>The data type must be consistent with x2.</li><li>The current version supports only two-dimensional or three-dimensional input.</li></ul></td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>2-3</td>
          <td>×</td>
        </tr>
        <tr>
          <td>x2</td>
          <td>Input</td>
          <td>Right matrix of the MatMul computation, i.e., x2 in the formula.</td>
          <td><ul><li>Empty tensors are supported.</li><li>Matches the data type of x1.</li><li>The current version supports only two-dimensional input, and supports transposed and non-transposed use cases.</li><li>Non-contiguous tensors are supported when the last two axes are transposed.</li></ul></td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>bias</td>
          <td>Input</td>
          <td>The bias in the formula.</td>
          <td><ul><li>Passing a null pointer is supported.</li><li>The current version supports only one-dimensional input.</li></ul></td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>1</td>
          <td>√</td>
        </tr>
        <tr>
          <td>residual</td>
          <td>Input</td>
          <td>Residual input of the AddRmsNorm fused operator, corresponding to residual in the formula.</td>
          <td>In the inplace use case, residual is used as the output address of y. The current version supports only three-dimensional input.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>3</td>
          <td>×</td>
        </tr>
        <tr>
          <td>gamma</td>
          <td>Input</td>
          <td>RmsNorm computation input of the AddRmsNorm fused operator, that is, gamma in the formula.</td>
          <td>The current version supports only one-dimensional input.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>1</td>
          <td>×</td>
        </tr>
        <tr>
          <td>epsilon</td>
          <td>Input</td>
          <td>Used to prevent division by zero, that is, the epsilon in the formula.</td>
          <td>The value of epsilon is in the range (0,1).</td>
          <td>double</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>group</td>
          <td>Input</td>
          <td>Host-side string identifying the communicator, that is, the communicator name.</td>
          <td>Obtained through the Hccl-provided interface "extern HcclResult HcclGetCommName(HcclComm comm, char* commName);", where commName is the group.</td>
          <td>string</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>reduceOp</td>
          <td>Input</td>
          <td>Host-side string that identifies the operation type, which is the reduce operation type.</td>
          <td>Currently, only the input "sum" is supported.</td>
          <td>string</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>commTurn</td>
          <td>Input</td>
          <td>Number of communication data splits, that is, total data volume/communication volume per time.</td>
          <td>The current version supports only input 0.</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>streamMode</td>
          <td>Input</td>
          <td>Enumeration of stream mode.</td>
          <td>Currently, only enum value 1 is supported.</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>normOut</td>
          <td>Output</td>
          <td>Result of mm + all_reduce + add + rms_norm, which is normOut in the formula.</td>
          <td><ul><li>Empty tensors are not supported.</li><li>The data type is the same as the residual input.</li></ul></td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>workspaceSize</td>
          <td>Output</td>
          <td>Returns the size of the workspace to be allocated on the device.</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>executor</td>
          <td>Output</td>
          <td>Returns the op executor, which contains the operator computation process.</td>
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
        <td>x1, x2, residual, or gamma is a null pointer.</td>
    </tr>
    <tr>
        <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="2">161002</td>
        <td>The shapes of x1, x2, and residual do not meet the constraints.</td>
    </tr>
    <tr>
        <td>streamMode is out of the valid range.</td>
    </tr>
    </tbody>
    </table>

## aclnnInplaceMatmulAllReduceAddRmsNorm

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
        <td>Workspace memory address allocated on the Device side.</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Workspace size allocated on the Device side, obtained by the first-phase API aclnnInplaceMatmulAllReduceAddRmsNormGetWorkspaceSize.</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>Input</td>
        <td>op executor, which contains the operator computation flow.</td>
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
  - aclnnInplaceMatmulAllReduceAddRmsNorm defaults to a non-deterministic implementation. Deterministic computation can be enabled by setting the `HCCL_DETERMINISTIC` environment variable to true.

- The use cases are the same as those of the fused operator aclnnMatmulAllReduce: MC2 is disabled in the incremental use case and enabled in the full use case.
- The input x1 can be two-dimensional or three-dimensional, with its shape being (b, s, k) or (m, k). x2 must be two-dimensional, with its shape being (k, n). The axes must meet the input requirements of the mm operator, with the k axes being equal. If bias is not empty, bias is one-dimensional with its shape being (n).
- The values of b*s, m, k, and n must not exceed 2147483647 (INT32_MAX).
- The input residual must be three-dimensional with the shape (b, s, n). When x1 is two-dimensional, (b*s) of residual equals m of x1. The input gamma must be one-dimensional with the shape (n).
- The output y and normOut have the same shape and data type as residual, with the shape (b, s, n).
- The data types of x1, x2, bias, residual, gamma, y, and normOut computation inputs must be consistent.
- Only the transposed/non-transposed scenarios of the x2 matrix are supported. The x1 matrix supports the non-transposed scenario.
- Supports 1, 2, 4, or 8 cards, and only supports HCCS all-mesh networking.
- Supports empty tensors where (b*s) and n are 0, but does not support empty tensors where k is 0.

<!-- npu="910b" id7 -->
- <term>Atlas A2 products</term>: A communication-computation fused MC2 operator in a model only supports the same communication domain.

<!-- end id7 -->

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

<!-- npu="910b" id8 -->
- <term>Atlas A2 products</term>:

<!-- end id8 -->

```Cpp
#include <iostream>
#include <vector>
#include <thread>
#include "hccl/hccl.h"
#include "aclnnop/aclnn_inplace_matmul_all_reduce_add_rms_norm.h"

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
    // Call aclrtMemcpy to copy data from the host to the device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    // Call aclCreateTensor to create an aclTensor.
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

int launchOneThreadMatmulAllReduceAddRmsNorm(Args &args) {
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
    std::vector<int64_t> residualShape = {1, 32, 128};
    std::vector<int64_t> gammaShape = {128};
    std::vector<int64_t> yShape = {1, 32, 128};
    std::vector<int64_t> normOutShape = {1, 32, 128};
    void *x1DeviceAddr = nullptr;
    void *x2DeviceAddr = nullptr;
    void *biasDeviceAddr = nullptr;
    void *residualDeviceAddr = nullptr;
    void *gammaDeviceAddr = nullptr;
    void *yDeviceAddr = nullptr;
    void *normOutDeviceAddr = nullptr;
    aclTensor *x1 = nullptr;
    aclTensor *x2 = nullptr;
    aclTensor *bias = nullptr;
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
    long long residualShapeSize = GetShapeSize(residualShape);
    long long gammaShapeSize = GetShapeSize(gammaShape);
    long long yShapeSize = GetShapeSize(yShape);
    long long normOutShapeSize = GetShapeSize(normOutShape);
    std::vector<int16_t> x1HostData(x1ShapeSize, 1);
    std::vector<int16_t> x2HostData(x2ShapeSize, 1);
    std::vector<int16_t> biasHostData(biasShapeSize, 1);
    std::vector<int16_t> residualHostData(residualShapeSize, 1);
    std::vector<int16_t> gammaHostData(gammaShapeSize, 1);
    std::vector<int16_t> yHostData(yShapeSize, 0);
    std::vector<int16_t> normOutHostData(normOutShapeSize, 0);
    // Create tensors.
    ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT16, &x2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT16, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(residualHostData, residualShape, &residualDeviceAddr, aclDataType::ACL_FLOAT16, &residual);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT16, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT16, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(normOutHostData, normOutShape, &normOutDeviceAddr, aclDataType::ACL_FLOAT16, &normOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Example of calling the aclnnInplaceMatmulAllReduceAddRmsNorm API
    // Call the first-phase API.
    ret = aclnnInplaceMatmulAllReduceAddRmsNormGetWorkspaceSize(x1, x2, bias, residual, gamma, epsilon, hcom_name, "sum", commTurn, streamMode,
                                            normOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnInplaceMatmulAllReduceAddRmsNormGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize computed by the first-phase API.
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API.
    ret = aclnnInplaceMatmulAllReduceAddRmsNorm(workspaceAddr, workspaceSize, executor, args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceMatmulAllReduceAddRmsNorm failed. ERROR: %d\n", ret); return ret);
    // (boilerplate) Wait synchronously for the task execution to complete.
    ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("device%d aclnnInplaceMatmulAllReduceAddRmsNorm execute success \n", args.rankId);
    // Release device resources. Modify it according to the definition of the specific API.

    if (x1 != nullptr) {
        aclDestroyTensor(x1);
    }
    if (x2 != nullptr) {
        aclDestroyTensor(x2);
    }
    if (bias != nullptr) {
        aclDestroyTensor(bias);
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
    // This sample is implemented based on Atlas A2 and must run on Atlas A2.
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
        threads[rankId].reset(new(std::nothrow) std::thread(&launchOneThreadMatmulAllReduceAddRmsNorm, std::ref(args[rankId])));
    }
    for (uint32_t rankId = 0; rankId < ndev; rankId++) {
        threads[rankId]->join();
    }
    aclFinalize();
    return 0;
}
```
