# aclnnBatchMatMulReduceScatterAlltoAll

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T13:08:46.044Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Not supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
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

- Description: BatchMatMulReduceScatterAllToAll is a fused operator that implements BatchMatMul computation in parallel with ReduceScatter and AllToAll collective communication.

- Formula: The overall computation flow is: BatchMatMul computation --> transpose (required when yShardType equals 0) --> ReduceScatter collective communication --> Add --> AllToAll collective communication. The computation logic is as follows, where y is the output.

$$
temp1 = BatchMatMul(x,weight)
$$

$$
temp2 = ReduceScatter(temp1)
$$

$$
temp3 = Add(temp2, bias)
$$

$$
y = AllToAll(temp3)
$$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnBatchMatMulReduceScatterAlltoAllGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor that contains the operator's computation process, and then call the "aclnnBatchMatMulReduceScatterAlltoAll" API to perform the computation.

```cpp
aclnnStatus aclnnBatchMatMulReduceScatterAlltoAllGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* weight,
    const aclTensor* biasOptional,
    const char*      groupEp,
    const char*      groupTp,
    int64_t          epWorldSize,
    int64_t          tpWorldSize,
    int64_t          yShardType,
    aclTensor*       out,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnBatchMatMulReduceScatterAlltoAll(
    void*           workspace,
    uint64_t        workspaceSize,
    aclOpExecutor*  executor,
    aclrtStream     stream)
```

## aclnnBatchMatMulReduceScatterAlltoAllGetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1600px"> <colgroup>
    <col style="width: 170px">
    <col style="width: 140px">
    <col style="width: 310px">
    <col style="width: 350px">
    <col style="width: 150px">
    <col style="width: 120px">
    <col style="width: 150px">
    <col style="width: 190px">
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
    <td>x(aclTensor*)</td>
    <td>Input</td>
    <td>Left matrix for the BatchMatMul computation.</td>
    <td>Must have 3 dimensions.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>3</td>
    <td>×</td>
    </tr>
    <tr>
    <td>weight(aclTensor*)</td>
    <td>Input</td>
    <td>Right matrix for the BatchMatMul computation.</td>
    <td><ul><li>The data type must be the same as that of x.</li><li>Must be 3-dimensional.</li></ul></td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>3</td>
    <td>√ (only for the transpose use case)</td>
    </tr>
    <tr>
    <td>biasOptional(aclTensor*)</td>
    <td>Input</td>
    <td>Bias of the Add computation.</td>
    <td><ul><li>The Add operation must be performed after the ReduceScatter collective communication.</li><li>If x is FLOAT16, biasOptional must be FLOAT16; if x is BFLOAT16, biasOptional must be FLOAT32.</li><li>Supports two or three dimensions. A null pointer is allowed.</li></ul></td>
    <td>FLOAT16, FLOAT32</td>
    <td>ND</td>
    <td>2 or 3</td>
    <td>×</td>
    </tr>
    <tr>
    <td>groupEp(char*)</td>
    <td>Input</td>
    <td>Communicator name for expert parallel.</td>
    <td>String length must be greater than 0 and less than 128.</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>groupTp(char*)</td>
    <td>Input</td>
    <td>Name of the communicator for tensor parallelism.</td>
    <td>The string length must be greater than 0 and less than 128.</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>epWorldSize(int64_t)</td>
    <td>Input</td>
    <td>ep communicator size</td>
    <td>Value range: 2, 4, 8, 16, 32.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>tpWorldSize (int64_t)</td>
    <td>Input</td>
    <td>tp communicator size</td>
    <td>Value range: 2, 4, 8, 16, 32.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>yShardType (int64_t)</td>
    <td>Input</td>
    <td>Integer</td>
    <td><ul><li>0: ReduceScatter is performed on the H dimension (the second dimension of the BatchMatMul output, which has 3 dimensions, indexed 0, 1, and 2) by tp.</li><li>1: ReduceScatter is performed on the C dimension (the first dimension of the BatchMatMul output) by tp.</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>out(aclTensor*)</td>
    <td>Output</td>
    <td>The result of batch_matmul computation + reduce_scatter computation + all_to_all communication.</td>
    <td><ul><li>Same data type as input x.</li><li>Must be 3 dimensions.</li></ul></td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>3</td>
    <td>×</td>
    </tr>
    <tr>
    <td>workspaceSize (uint64_t)</td>
    <td>Output</td>
    <td>Returns the workspace size.</td>
    <td>-</td>
    <td>UINT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor(aclOpExecutor**)</td>
    <td>Output</td>
    <td>Returns the op executor that contains the computation flow of the operator.</td>
    <td>-</td>
    <td>aclOpExecutor*</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    </tbody></table>

- **Return Value**

    Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

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
    <td>1. x, weight, groupEp, groupTp, or out is a null pointer.</td>
    </tr>
    <tr>
    <td class="merged-cell" rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
    <td class="merged-cell" rowspan="5">161002</td>
    <td>1. The string length of groupEp or groupTp is invalid.</td>
    </tr>
    <tr>
    <td>2. The input data type is not supported.</td>
    </tr>
    <tr>
    <td>3. The attribute value is invalid.</td>
    </tr>
    <tr>
    <td>4. The aclTensor dimension is invalid.</td>
    </tr>
    <tr>
    <td>5. The aclTensor shape is invalid.</td>
    </tr>
    </tbody></table>

## aclnnBatchMatMulReduceScatterAlltoAll

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
    <td>Address of the workspace memory allocated on the device.</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>Input</td>
    <td>Size of the workspace memory allocated on the device, obtained by calling the first-phase API <code>aclnnBatchMatMulReduceScatterAlltoAllGetWorkspaceSize</code>.</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>Input</td>
    <td>Executor of the op, including the operator computation flow.</td>
    </tr>
    <tr>
    <td>stream</td>
    <td>Input</td>
    <td>Specifies the stream for task execution.</td>
    </tr>
    </tbody></table>

- **Return Value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnBatchMatMulReduceScatterAlltoAll defaults to a deterministic implementation.

  - Due to the requirements of collective communication and BatchMatMul computation, the input and output shapes must satisfy the following mathematical relationships: (where ep = epWorldSize, tp = tpWorldSize)
  - ReduceScatter scenario along the H axis, i.e., the scenario where yShardType is 0:

      - x: (E/ep, ep*C, M/tp)
      - weight: (E/ep, M/tp, H)
      - biasOptional: When the pointer is non-null, (E/ep, 1, H/tp) when three-dimensional, or (E/ep, H/tp) when two-dimensional.
      - y: (E, C, H/tp)

  - For the ReduceScatter scenario along the C axis, i.e., when yShardType is 1:
     - x: (E/ep, ep*tp*C/tp, M/tp)
     - weight: (E/ep, M/tp, H)
     - biasOptional: If not a null pointer, (E/ep, 1, H) for three dimensions, or (E/ep, H) for two dimensions.
     - y: (E, C/tp, H)

- Data relationship description:
  - For example, if x.size(0) is equal to E/tp and y.size(0) is equal to E, it means that y.size(0) = ep*x.size(0), and y.size(0) is an integer multiple of ep; other relationships are similar.
  - The value range of E is [2, 512], and E is an integer multiple of ep.
  - The value range of H is [1, 65535]. When yShardType is 0, H is an integer multiple of tp.
  - The value range of M/tp is [1, 65535].
  - The value range of E/ep is [1, 32].
  - Both ep and tp only support 2, 4, 8, 16, and 32.
  - The groupEp and groupTp names cannot be the same.
  - C is greater than 0, with the upper limit being the operator's device memory limit. When yShardType is 1, C is an integer multiple of tp.
  - The communication-computation fusion operator does not support concurrent calls, and different communication-computation fusion operators also do not support concurrent calls.
  - Cross-super-node operation is not supported. Only intra-super-node operation is supported.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

<!-- npu="A3" id7 -->
- <term>Atlas A3 products</term>  :

    ```Cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <vector>
    #include "acl/acl.h"
    #include "hccl/hccl.h"
    #include "aclnnop/aclnn_batch_matmul_reduce_scatter_all_to_all.h"

    #define CHECK_RET(cond, return_expr) \
        do {                             \
            if (!(cond)) {               \
                return_expr;             \
            }                            \
        } while (0)

    #define LOG_PRINT(message, ...)         \
        do {                                \
            printf(message, ##__VA_ARGS__); \
        } while(0)

    constexpr int EP_WORLD_SIZE = 4;
    constexpr int TP_WORLD_SIZE = 2;
    constexpr int DEV_NUM = EP_WORLD_SIZE * TP_WORLD_SIZE;

    int64_t GetShapeSize(const std::vector<int64_t> &shape)
    {
        int64_t shape_size = 1;
        for (auto i : shape) {
            shape_size *= i;
        }
        return shape_size;
    }

    template<typename T>
    int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
        aclDataType dataType, aclTensor **tensor)
    {
        auto size = GetShapeSize(shape) * sizeof(T);
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret);
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = shape.size() - 2; i >= 0; i--) {
            strides[i] = shape[i +1] * strides[i + 1];
        }
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
            shape.data(), shape.size(), *deviceAddr);
        return 0;
    }

    struct Args {
        int rankId;
        HcclComm hcclEpComm;
        HcclComm hcclTpComm;
        aclrtStream stream;
        aclrtContext context;
    };

    int LaunchOneThreadBatchMMRSAlltoAll(Args &args)
    {
        int ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret: %d\n", ret); return ret);
        char hcomEpName[128] = {0};
        ret = HcclGetCommName(args.hcclEpComm, hcomEpName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetEpCommName failed. ret: %d\n", ret); return -1);
        char hcomTpName[128] = {0};
        ret = HcclGetCommName(args.hcclTpComm, hcomTpName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetTpCommName failed. ret: %d\n", ret); return -1);
        LOG_PRINT("[INFO] rank = %d, hcomEpName = %s, hcomTpName = %s, stream = %p, context = %p\n", args.rankId,
            hcomEpName, hcomTpName, args.stream, args.context);

        int64_t E = 4 * EP_WORLD_SIZE;
        int64_t C = 6 * TP_WORLD_SIZE;
        int64_t H = 2 * TP_WORLD_SIZE;
        int64_t M = 6 * TP_WORLD_SIZE;
        int64_t xShardType = 1;

        std::vector<int64_t> xShape;
        std::vector<int64_t> weightShape;
        std::vector<int64_t> biasShape;
        std::vector<int64_t> yOutShape;

        if (xShardType == 1) {
            xShape = {E / EP_WORLD_SIZE, EP_WORLD_SIZE * TP_WORLD_SIZE * C / TP_WORLD_SIZE, M / TP_WORLD_SIZE};
            weightShape = {E / EP_WORLD_SIZE, M / TP_WORLD_SIZE, H};
            biasShape = {E / EP_WORLD_SIZE, 1, H};
            yOutShape = {E, C / TP_WORLD_SIZE, H};
        } else if (xShardType == 0) {
            xShape = {E / EP_WORLD_SIZE, EP_WORLD_SIZE * C, M / TP_WORLD_SIZE};
            weightShape = {E / EP_WORLD_SIZE, M / TP_WORLD_SIZE, H};
            biasShape = {E / EP_WORLD_SIZE, 1, H / TP_WORLD_SIZE};
            yOutShape = {E, C, H / TP_WORLD_SIZE};
        } else {
            LOG_PRINT("[ERROR] unsupported xShardType = %ld.\n", xShardType);
            return -1;
        }

        printf("x_shape: %d %d %d\n", xShape[0], xShape[1], xShape[2]);
        printf("weight_shape: %d %d %d\n", weightShape[0], weightShape[1], weightShape[2]);
        printf("bias_shape: %d %d %d\n", biasShape[0], biasShape[1], biasShape[2]);
        printf("y_shape: %d %d %d\n", yOutShape[0], yOutShape[1], yOutShape[2]);

        void *xDeviceAddr = nullptr;
        void *weightDeviceAddr = nullptr;
        void *biasDeviceAddr = nullptr;
        void *yOutDeviceAddr = nullptr;
        aclTensor *x = nullptr;
        aclTensor *weight = nullptr;
        aclTensor *bias = nullptr;
        aclTensor *yOut = nullptr;

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        void *workspaceAddr = nullptr;

        long long xShapeSize = GetShapeSize(xShape);
        long long weightShapeSize = GetShapeSize(weightShape);
        long long biasShapeSize = GetShapeSize(biasShape);
        long long yOutShapeSize = GetShapeSize(yOutShape);

        std::vector<int16_t> xHostData(xShapeSize, 1);
        std::vector<int16_t> weightHostData(weightShapeSize, 2);
        std::vector<int16_t> biasHostData(biasShapeSize, 3);
        std::vector<int16_t> y1OutHostData(yOutShapeSize, 0);

        ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(y1OutHostData, yOutShape, &yOutDeviceAddr, aclDataType::ACL_FLOAT16, &yOut);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        // Call the first-phase API.
        ret = aclnnBatchMatMulReduceScatterAlltoAllGetWorkspaceSize(x, weight, bias, hcomEpName, hcomTpName, EP_WORLD_SIZE,
            TP_WORLD_SIZE, xShardType, yOut, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnBatchMatMulReduceScatterAlltoAllGetWorkspaceSize failed. ret = %d \n", ret); return ret);
        // Allocate device memory based on the workspaceSize calculated by the first-phase API.
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }
        // Call the second-phase API.
        ret = aclnnBatchMatMulReduceScatterAlltoAll(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnBatchMatMulReduceScatterAlltoAll failed. ret = %d \n", ret);
            return ret);
        //(boilerplate) Synchronize and wait for the task execution to complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
            return ret);
        LOG_PRINT("[INFO] device_%d aclnnBatchMatMulReduceScatterAlltoAll execute successfully.\n", args.rankId);
        HcclCommDestroy(args.hcclEpComm);
        HcclCommDestroy(args.hcclTpComm);
        // Release device resources. Modify based on the definition of the specific API.
        if (x != nullptr) {
            aclDestroyTensor(x);
        }
        if (weight != nullptr) {
            aclDestroyTensor(weight);
        }
        if (bias != nullptr) {
            aclDestroyTensor(bias);
        }
        if (yOut != nullptr) {
            aclDestroyTensor(yOut);
        }
        if (xDeviceAddr != nullptr) {
            aclrtFree(xDeviceAddr);
        }
        if (weightDeviceAddr != nullptr) {
            aclrtFree(weightDeviceAddr);
        }
        if (biasDeviceAddr != nullptr) {
            aclrtFree(biasDeviceAddr);
        }
        if (yOutDeviceAddr != nullptr) {
            aclrtFree(yOutDeviceAddr);
        }
        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }
        ret = aclrtDestroyStream(args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtDestroyStream failed. ret = %d \n", ret); return ret);
        ret = aclrtResetDevice(args.rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtResetDevice failed. ret = %d \n", ret); return ret);
        return 0;
    }

    int main(int argc, char *argv[])
    {
        // This sample is implemented based on Atlas A3 and must run on Atlas A3.
        int ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d \n", ret); return ret);
        aclrtStream stream[DEV_NUM];
        aclrtContext context[DEV_NUM];
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            ret = aclrtSetDevice(rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);
            ret = aclrtCreateContext(&context[rankId], rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d \n", ret); return ret);
            ret = aclrtCreateStream(&stream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d \n", ret); return ret);
        }

        int32_t devicesEp[DEV_NUM];
        int32_t devicesTp[DEV_NUM];

        // Initialize the ep domain. ep = 4  {0,2,4,6} {1,3,5,7}
        HcclComm commsEp[DEV_NUM];
        for (int i = 0; i < TP_WORLD_SIZE; i++) {
            for (int j =0; j < EP_WORLD_SIZE; j++) {
                devicesEp[j + i * EP_WORLD_SIZE] = i + j * TP_WORLD_SIZE;
            }
            ret = HcclCommInitAll(EP_WORLD_SIZE, &devicesEp[i * EP_WORLD_SIZE], &commsEp[i * EP_WORLD_SIZE]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll ep world %d failed. ret = %d \n", i, ret);
                return ret);
        }

        // Initialize the tp domain. tp = 4  {0,1},{2,3},{4,5},{6,7}
        HcclComm commsTp[DEV_NUM];
        for (int i = 0; i < EP_WORLD_SIZE; i++) {
            for (int j =0; j < TP_WORLD_SIZE; j++) {
                devicesTp[j + i * TP_WORLD_SIZE] = j + i * TP_WORLD_SIZE;
            }
            ret = HcclCommInitAll(TP_WORLD_SIZE, &devicesTp[i * TP_WORLD_SIZE], &commsTp[i * TP_WORLD_SIZE]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll tp world %d failed. ret = %d \n", i, ret);
                return ret);
        }

        Args args[DEV_NUM];
        // Start multiple threads.
        std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            args[rankId].rankId = rankId;
            args[rankId].hcclEpComm = commsEp[rankId % TP_WORLD_SIZE * EP_WORLD_SIZE + rankId / TP_WORLD_SIZE];
            args[rankId].hcclTpComm = commsTp[rankId];
            args[rankId].stream = stream[rankId];
            args[rankId].context = context[rankId];
            threads[rankId].reset(new std::thread(&LaunchOneThreadBatchMMRSAlltoAll, std::ref(args[rankId])));
        }
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            threads[rankId]->join();
        }
        aclFinalize();
        return 0;
    }
    ```

<!-- end id7 -->
