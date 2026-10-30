# aclnnMatmulAllReduceV3

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T13:33:20.343Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
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

Note: When using this API, ensure that both the driver firmware package and the CANN package are of the matching 8.0.RC2 version or a matching later version. Otherwise, errors such as Bus Error will occur.

## Function

- **Description**: Fuses MatMul computation with AllReduce communication. This API is compatible with the functions supported by `aclnnMatmulAllReduceV2` and adds the communication engine parameter commMode, allowing you to select a communication engine.
- **Formula**:

    $$
    output = AllReduce(x1 @ x2 + bias + x3)
    $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call `aclnnMatmulAllReduceV3GetWorkspaceSize` first to obtain the workspace size required for computation and the executor containing the operator computation process. Then call `aclnnMatmulAllReduceV3` to execute the computation.

```cpp
aclnnStatus aclnnMatmulAllReduceV3GetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *bias,
    const aclTensor *x3,
    const char      *group,
    const char      *reduceOp,
    const char      *commMode,
    int64_t          commTurn,
    int64_t          streamMode,
    const aclTensor *output,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnMatmulAllReduceV3(
    void              *workspace,
    uint64_t           workspaceSize,
    aclOpExecutor     *executor,
    const aclrtStream  stream)
```

## aclnnMatmulAllReduceV3GetWorkspaceSize

- **Parameters**
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
          <td>Left matrix for the MatMul computation, that is, x1 in the formula.</td>
          <td><ul><li>The current version supports only two-dimensional or three-dimensional input.</li><li>The non-transposition use case is supported.</li></ul></td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>2-3</td>
          <td>×</td>
        </tr>
        <tr>
          <td>x2</td>
          <td>Input</td>
          <td>Right matrix for the MatMul computation, that is, x2 in the formula.</td>
          <td><ul><li>The Current Version supports only 2D inputs.</li><li>Transpose/non-transpose use cases are supported.</li><li>Under the ND format, non-contiguous tensors with the last two axes transposed are supported. Other non-contiguous tensors are Not Supported.</li></ul></td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>2</td>
          <td>×</td>
        </tr>
        <tr>
          <td>x3</td>
          <td>Input</td>
          <td>Add computation after the MatMul computation, that is, x3 in the formula.</td>
          <td>The shape is the same as the shape after the MatMul computation.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>bias</td>
          <td>Input</td>
          <td>The bias offset in the formula.</td>
          <td>The Current Version supports only one-dimensional input.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>0-1</td>
          <td>√</td>
        </tr>
        <tr>
          <td>group</td>
          <td>Input</td>
          <td>Communicator name.</td>
          <td>Obtained through the Hccl API "extern HcclResult HcclGetCommName(HcclComm comm, char* commName);", where commName is the group.</td>
          <td>String</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>reduceOp</td>
          <td>Input</td>
          <td>Reduce operation type.</td>
          <td>Current Version supports only input "sum".</td>
          <td>String</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>commMode</td>
          <td>Input</td>
          <td>Communication Engine parameter.</td>
          <td>Specifies the communication engine. Input restrictions vary with the chip model. For a brief introduction to the communication engine and input constraints, see <a href="#constraints">Constraints</a>.</td>
          <td>String</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>commTurn</td>
          <td>Input</td>
          <td>Number of splits of communication data, that is, total data volume/data volume per communication.</td>
          <td>Only 0 is supported in the Current Version.</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>streamMode</td>
          <td>Input</td>
          <td>Enum of AscendCL stream modes.</td>
          <td>Only enum value 1 is supported in the current version.</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>Output</td>
          <td>Output</td>
          <td>Result of MatMul computation and AllReduce communication, that is, output in the formula.</td>
          <td>output has the same dimensions as x1.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>2-3</td>
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
          <td>Returns the op executor, which contains the operator computation flow.</td>
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

    <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
    <col style="width: 282px">
    <col style="width: 120px">
    <col style="width: 747px">
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
        <td>The input x1, x2, or output is a null pointer.</td>
    </tr>
    <tr>
        <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="3">161002</td>
        <td>The data type of x1, x2, bias, x3, or output is out of the supported range.</td>
    </tr>
    <tr>
        <td>reduceOp or streamMode is out of the valid range.</td>
    </tr>
    <tr>
        <td>The shape of x1, x2, bias, x3, or output does not meet the constraints.</td>
    </tr>
    </tbody>
    </table>

## aclnnMatmulAllReduceV3

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 168px">
    <col style="width: 128px">
    <col style="width: 854px">
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
        <td>Workspace memory address allocated on the device.</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Workspace size allocated on the device, obtained by calling the first-phase API <code>aclnnMatmulAllReduceV3GetWorkspaceSize</code>.</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Op executor that contains the computation process of the operator.</td>
    </tr>
    <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream that executes the task.</td>
    </tr>
    </tbody></table>

- **Return Value**

    Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Communication Engine commMode support:

  <!-- npu="910b" id7 -->
  - <term>Atlas A2 products</term>: Currently, specifying a Communication Engine is not supported. commMode allows only "ai_cpu" as the input, which uses the AICPU Communication Engine.
  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT products</term>: Currently, the supported Communication Engines are AICPU and CCU. commMode allows "ai_cpu" or "ccu" as the input. CCU supports only UB domain interconnection within a single node, and AICPU supports UB domain interconnection across nodes.
  <!-- end id8 -->
  - Within the same communication link, only one type of communication engine can be selected.
  - Brief introduction to the AICPU and CCU communication engines:
    - AICPU: Does not occupy computation cores and provides high communication efficiency, but incurs high static communication overhead. It is not suitable for communication tasks with small data volumes, but is suitable for high-bandwidth tasks with large data volumes.
    - CCU: Reduces memory access bandwidth usage and computation core occupation, but supports only a limited number of communicators due to on-chip resource constraints. Suitable for high-bandwidth, low-latency communication tasks.
    - For more information about communication engines, see [Communication Engine](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910beta1/programug/commopdev/hcclopdev_000005.html).
- Deterministic computation:

  <!-- npu="910b" id9 -->
  - <term>Atlas A2 products</term>: `aclnnMatmulAllReduceV3` defaults to a non-deterministic implementation. Set the `HCCL_DETERMINISTIC` environment variable to true to enable deterministic computation.
  <!-- end id9 -->
  <!-- npu="950" id10 -->
  - <term>Ascend 950PR&950DT products</term>: `aclnnMatmulAllReduceV3` defaults to a deterministic implementation.
  <!-- end id10 -->

- MC2 is disabled in incremental scenarios and enabled in full scenarios.
- The input x1 can be two-dimensional or three-dimensional, with its shape being (b, s, k) or (m, k). x2 must be two-dimensional, with its shape being (k, n). The axes must meet the MM operator input requirements, with the k-axis being equal. If bias is not null, its shape is (n).
- The values of b*s, m, k, and n must not exceed 2147483647 (INT32_MAX).
- When the shape of input x1 is (b, s, k), the shape of output is (b, s, n); when the shape of input x1 is (m, k), the shape of output is (m, n).
- The data types of x1, x2, and bias computation inputs must be consistent with the data type of the output computation result. The passed x1, x2, x3, or output must not be null pointers.
- Only HCCS all-mesh networking is supported.

    <!-- npu="910b" id11 -->
    - <term>Atlas A2 products</term>: Supports 1, 2, 4, or 8 cards.
    <!-- end id11 -->
    <!-- npu="950" id12 -->
    - <term>Ascend 950PR&950DT products</term>: Supports 1, 2, 4, 8, 16, 32, or 64 cards.
    <!-- end id12 -->

<!-- npu="910b" id13 -->
- <term>Atlas A2 products</term>: A communication-computation fused MC2 operator in a model only supports the same communication domain.
<!-- end id13 -->
- Null tensor support:
  - Supports the scenario where k is 0. The output is bias + x3.
  - Supports the scenario where bs/m/n is 0. In this case, the output passed in should also be an empty tensor, and the computation does not enter the kernel but returns directly.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

NOTE: This sample code calls some HCCL collective communication library APIs, including HcclGetCommName, HcclCommInitAll, and HcclCommDestroy. For details, see [*HCCL API (C)*](https://hiascend.com/en/document/redirect/CannCommunityHcclCppApi).

<!-- npu="950,910b" id14 -->
- <term>Atlas A2 products</term>, <term>Ascend 950PR&950DT products</term>:

    ```Cpp
    #include <iostream>
    #include <vector>
    #include <thread>
    #include "hccl/hccl.h"
    #include "aclnn/opdev/fp16_t.h"
    #include "aclnnop/aclnn_matmul_all_reduce_v3.h"

    int ndev = 2;

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
        // Call aclrtMalloc to allocate device memory
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
        // Call aclrtMemcpy to copy host data to device memory
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
        // Calculate the strides of the contiguous tensor
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = shape.size() - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
        // Call aclCreateTensor to create an aclTensor
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

    int launchOneThreadMatmulAllReduce(Args &args) {
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
        std::vector<int64_t> x3Shape = {32, 128};
        std::vector<int64_t> outShape = {32, 128};
        void *x1DeviceAddr = nullptr;
        void *x2DeviceAddr = nullptr;
        void *biasDeviceAddr = nullptr;
        void *x3DeviceAddr = nullptr;
        void *outDeviceAddr = nullptr;
        aclTensor *x1 = nullptr;
        aclTensor *x2 = nullptr;
        aclTensor *bias = nullptr;
        aclTensor *x3 = nullptr;
        aclTensor *out = nullptr;

        int64_t commTurn = 0;
        int64_t streamMode = 1;
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;
        void *workspaceAddr = nullptr;

        long long x1ShapeSize = GetShapeSize(x1Shape);
        long long x2ShapeSize = GetShapeSize(x2Shape);
        long long biasShapeSize = GetShapeSize(biasShape);
        long long x3ShapeSize = GetShapeSize(x3Shape);
        long long outShapeSize = GetShapeSize(outShape);
        std::vector<op::fp16_t> x1HostData(x1ShapeSize, 1);
        std::vector<op::fp16_t> x2HostData(x2ShapeSize, 1);
        std::vector<op::fp16_t> biasHostData(biasShapeSize, 1);
        std::vector<op::fp16_t> x3HostData(x3ShapeSize, 1);
        std::vector<op::fp16_t> outHostData(outShapeSize, 0);
        // Create a tensor
        ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT16, &x2);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT16, &bias);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x3HostData, x3Shape, &x3DeviceAddr, aclDataType::ACL_FLOAT16, &x3);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Call the first-phase API
        ret = aclnnMatmulAllReduceV3GetWorkspaceSize(x1, x2, bias, x3, hcom_name, "sum", "", commTurn, streamMode, out, &workspaceSize, &executor);

        CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclnnMatmulAllReduceV3GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
        // Allocate device memory based on the workspaceSize calculated by the first-phase API
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        // Call the second-phase API
        ret = aclnnMatmulAllReduceV3(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulAllReduceV3 failed. ERROR: %d\n", ret); return ret);
        // (Boilerplate) Synchronously wait for the task to complete
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
        LOG_PRINT("device%d aclnnMatmulAllReduceV3 execute success \n", args.rankId);
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
        if (x3 != nullptr) {
            aclDestroyTensor(x3);
        }
        if (out != nullptr) {
            aclDestroyTensor(out);
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
        if (x3DeviceAddr != nullptr) {
            aclrtFree(x3DeviceAddr);
        }
        if (outDeviceAddr != nullptr) {
            aclrtFree(outDeviceAddr);
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
        // Start multithreading.
        std::vector<std::unique_ptr<std::thread>> threads(ndev);
        for (uint32_t rankId = 0; rankId < ndev; rankId++) {
            args[rankId].rankId = rankId;
            args[rankId].hcclComm = comms[rankId];
            args[rankId].stream = stream[rankId];
            args[rankId].context = context[rankId];
            threads[rankId].reset(new(std::nothrow) std::thread(&launchOneThreadMatmulAllReduce, std::ref(args[rankId])));
        }
        for (uint32_t rankId = 0; rankId < ndev; rankId++) {
            threads[rankId]->join();
        }
        aclFinalize();
        return 0;
    }
    ```

<!-- end id14 -->
