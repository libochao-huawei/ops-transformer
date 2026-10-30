# aclnnFFNToAttention

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T13:24:44.940Z -->

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

Sends data on FFN nodes to Attention nodes.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must call the "aclnnFFNToAttentionGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor that contains the computation flow, and then call the "aclnnFFNToAttention" API to execute the computation.

```cpp
aclnnStatus aclnnFFNToAttentionGetWorkspaceSize(
    const aclTensor   *x,
    const aclTensor   *sessionIds,
    const aclTensor   *microBatchIds,
    const aclTensor   *tokenIds,
    const aclTensor   *expertOffsets,
    const aclTensor   *actualTokenNum,
    const aclTensor   *attnRankTableOptional,
    const char        *group,
    int64_t            worldSize,
    const aclIntArray *tokenInfoTableShape,
    const aclIntArray *tokenDataShape,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor)
```

```cpp
aclnnStatus aclnnFFNToAttention(
    void           *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor  *executor,
    aclrtStream     stream)
```

## aclnnFFNToAttentionGetWorkspaceSize

- **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1484px"><colgroup>
    <col style="width: 186px">
    <col style="width: 123px">
    <col style="width: 283px">
    <col style="width: 295px">
    <col style="width: 181px">
    <col style="width: 122px">
    <col style="width: 147px">
    <col style="width: 147px">
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
      <td>x</td>
      <td>Input</td>
      <td>Token data sent by the current device.</td>
      <td>Shape is <code>(Y, H)</code>.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sessionIds</td>
      <td>Input</td>
      <td>Attention Worker node index of each token.</td>
      <td>Shape is <code>(Y, )</code>, value range is [0, attnRankNum-1].</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>microBatchIds</td>
      <td>Input</td>
      <td>microBatch index of each token.</td>
      <td>Shape is <code>(Y, )</code>, value range is [0, MicroBatchNum-1].</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>tokenIds</td>
      <td>Input</td>
      <td>Token index of each token in the microBatch.</td>
      <td>Shape is <code>(Y, )</code>, and the value range is [0, Bs-1].</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>expertOffsets</td>
      <td>Input</td>
      <td>Index of each token in PerTokenExpertNum of tokenInfoTableShape.</td>
      <td>Shape is <code>(Y, )</code>, value range is [0, ExpertNumPerToken-1].</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>actualTokenNum</td>
      <td>Input</td>
      <td>Actual total number of tokens sent by the current device, 1D tensor.</td>
      <td>Shape: <code>(1, )</code>.</td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>attnRankTableOptional</td>
      <td>Optional input</td>
      <td>Maps the device ID of each Attention Worker.</td>
      <td>Attention Worker must be deployed consecutively starting from device 0; if a null pointer is passed, the default policy applies: the ID of each device is used as the ID of the corresponding Attention Worker, with a value range of [0, attnRankNum-1].</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>group</td>
      <td>Input</td>
      <td>Communicator name (expert parallel).</td>
      <td>String length: [1, 128).</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>worldSize</td>
      <td>Input</td>
      <td>Communicator size.</td>
      <td>The value range of worldSize is [2, 768].</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>tokenInfoTableShape</td>
      <td>Input</td>
      <td>Size of the token information list.</td>
      <td>Includes the microBatch size (MicroBatchNum), BatchSize (Bs), and the number of experts per token (ExpertNumPerToken).</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>tokenDataShape</td>
      <td>Input</td>
      <td>Token data list size.</td>
      <td>Includes the microBatch size (MicroBatchNum), BatchSize (Bs), number of experts per token (ExpertNumPerToken), and token and scale lengths (HS).</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Size of the workspace to be requested on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Op executor containing the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>×</td>
    </tr>
  </tbody></table>

- **Return value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API performs input parameter validation and returns an error in the following scenarios:

    <table style="undefined;table-layout: fixed; width: 1166px">
    <colgroup>
    <col style="width: 166px">
    <col style="width: 100px">
    <col style="width: 900px">
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
    <td>A required input or output tensor parameter is a null pointer.</td>
    </tr>
    <tr>
    <td>ACLNN_ERR_PARAM_INVALID</td>
    <td>161002</td>
    <td>The data types of inputs and outputs are out of the supported range.</td>
    </tr>
    </tbody>
    </table>

## aclnnFFNToAttention

- **Parameters:**

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
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>workspace</td>
    <td>Input</td>
    <td>Workspace memory address requested on the Device side.</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>Input</td>
    <td>Size of the workspace requested on the Device side, obtained by the first-phase API <code>aclnnFFNToAttentionGetWorkspaceSize</code>.</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>Input</td>
    <td>Executor of the op, containing the operator computation flow.</td>
    </tr>
    <tr>
    <td>stream</td>
    <td>Input</td>
    <td>Stream for executing tasks.</td>
    </tr>
    </tbody>
    </table>

- **Return value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- **Deterministic constraints**:
  - aclnnFFNToAttention defaults to a deterministic implementation

- **Parameter Consistency Constraints**:
  - The `group`, `worldSize`, `tokenInfoTableShape`, and `tokenDataShape` parameters and the value of `HCCL_BUFFSIZE` must be consistent across all devices.

- **Product-Specific Constraints**:

  <!-- npu="A3" id7 -->
  - <term>Atlas A3 products</term>: In this use case, each device contains dual DIEs (also referred to as dies). Therefore, the "local device" in Parameters refers to a single DIE.

  <!-- end id7 -->

- **Shape Variable Constraints**:

  | Variable | Definition and Value Range |
  | :----------- | :------------------------------------------------------------------------------------- |
  | Y | Maximum number of tokens to be dispatched by the local device. |
  | Bs | Number of tokens sent on each Attention node.<ul><li><term>Atlas A3 products</term>: <code>0 < Bs ≤ 512 </code>.</li></ul> |
  | H (hidden size) | Hidden size.<ul><li><term>Atlas A3 products</term>: <code>1024 ≤  H ≤ 8192 </code>.</li></ul> |
  | HS (hidden and scale size) | Hidden and scale size.<ul><li><term>Atlas A3 products</term>: <code>1152 ≤  HS ≤ 8320 </code>.</li></ul>|
  | MicroBatchNum | microBatch size. Currently, only <code>MicroBatchNum = 1</code> is supported. |
  | ExpertNumPerToken | Number of Experts corresponding to each token, <code>ExpertNumPerToken = K + sharedExpertNum</code>. |
  | K | Indicates selection of topK experts. Value range: <code>0 < K ≤ 16 </code>. |
  | ffnRankNum | Indicates selection of ffnRankNum devices as FFnWorkers. Value range: <code>0 < ffnRankNum < worldSize </code>. |
  | attnRankNum | Indicates selection of attnRankNum devices as AttnWorkers. Value range: <code>0 < attnRankNum < worldSize </code>. |
  | sharedExpertNum | Number of shared experts (a shared expert can be deployed in replica on multiple ffnRank devices). Current value range: [0, 4]. |

- **Communication Domain Usage Constraints**:
  - Other operators are not allowed in the communicator of the FFNToAttention operator.

## Example

- File preparation:

  1. Create the FFNtoAttentionDemo directory. Create the aclnnFFNtoAttentionDemo.cpp and FFNtoAttention.sh files under FFNtoAttentionDemo as instructed below, and modify them with reference to the following code.

  2. Install the CANN package and compile and run FFNtoAttentionDemo as instructed below.

- FFNtoAttention.sh build script

    ```bash
    #!/bin/bash
    cann_path="/path/to/cann_env" # Change the path of the CANN package environment.
    g++ "aclnnFFNtoAttentionDemo.cpp" -o FFNtoAttentionDemo -I"$cann_path/latest/include/" -I"$cann_path/latest/include/aclnnop/" \
                        -L="$cann_path/latest/lib64/" -lascendcl -lnnopbase -lopapi_math -lop_common -lpthread -lhccl
    ```

- Compile and run:

    ```bash
    # Source the CANN environment.
    source /path/to/cann_env/latest/bin/setenv.bash

    # Compile aclnnFFNtoAttentionDemo.cpp.
    bash FFNtoAttention.sh

    ./FFNtoAttentionDemo
    ```

- The sample code is as follows, for reference only.

    ```Cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <vector>
    #include <unordered_set>
    #include "acl/acl.h"
    #include "hccl/hccl.h"
    #include "aclnnop/aclnn_ffn_to_attention.h"

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

    struct Args {
        uint32_t rankId;
        HcclComm hcclComm;
        aclrtStream FFN2AttentionStream;
        aclrtContext context;
    };

    constexpr uint32_t WORLD_SIZE = 16;
    constexpr uint32_t ATTN_NUM = 8;
    constexpr uint32_t DEV_NUM = WORLD_SIZE;

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

    int LaunchOneProcessFFN2Attention(Args &args)
    {
        int ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed, ret %d\n", ret); return ret);

        char hcomName[128] = {0};
        ret = HcclGetCommName(args.hcclComm, hcomName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed, ret %d\n", ret); return -1);
        LOG_PRINT("[INFO] rank = %d, hcomName = %s, FFN2AttentionStream = %p, \
                    context = %p\n", args.rankId, hcomName, args.FFN2AttentionStream, \
                    args.context);

        int64_t micro_batch_num = 1;
        int64_t Y = 8;
        int64_t H = 7168;
        int64_t K = 7;
        int64_t attention_worker_num = ATTN_NUM;
        int64_t sharedExpertNum = 1;
        int64_t expert_num_per_token = K + sharedExpertNum;
        int64_t Token_info_shape[] = {micro_batch_num, Y, expert_num_per_token};
        int64_t Token_data_shape[] = {micro_batch_num, Y, expert_num_per_token, H};

        void *xDeviceAddr = nullptr;
        void *sessionIdsDeviceAddr = nullptr;
        void *microBatchIdsDeviceAddr = nullptr;
        void *tokenIdsDeviceAddr = nullptr;
        void *expertOffsetsDeviceAddr = nullptr;
        void *actualTokenNumDeviceAddr = nullptr;
        void *attnRankTableDeviceAddr = nullptr;

        aclTensor *x = nullptr;
        aclTensor *sessionIds = nullptr;
        aclTensor *microBatchIds = nullptr;
        aclTensor *tokenIds = nullptr;
        aclTensor *expertOffsets = nullptr;
        aclTensor *actualTokenNum = nullptr;
        aclTensor *attnRankTable = nullptr;
        aclIntArray *tokenInfoTableShape = aclCreateIntArray(Token_info_shape, 3);
        aclIntArray *tokenDataShape = aclCreateIntArray(Token_data_shape, 4);


        // Define the dimensions of each variable in the current use case.
        std::vector<int64_t> xShape{Y, H};
        std::vector<int64_t> sessionIdsShape{Y};
        std::vector<int64_t> microBatchIdsShape{Y};
        std::vector<int64_t> tokenIdsShape{Y};
        std::vector<int64_t> expertOffsetsShape{Y};
        std::vector<int64_t> actualTokenNumShape{1};
        std::vector<int64_t> attnRankTableShape{attention_worker_num};


        int64_t xShapeSize = GetShapeSize(xShape);
        int64_t sessionIdsShapeSize = GetShapeSize(sessionIdsShape);
        int64_t microBatchIdsShapeSize = GetShapeSize(microBatchIdsShape);
        int64_t tokenIdsShapeSize = GetShapeSize(tokenIdsShape);
        int64_t expertOffsetsShapeSize = GetShapeSize(expertOffsetsShape);
        int64_t actualTokenNumShapeSize = GetShapeSize(actualTokenNumShape);
        int64_t attnRankTableShapeSize = GetShapeSize(attnRankTableShape);


        std::vector<int16_t> xHostData(xShapeSize, 1);
        std::vector<int32_t> sessionIdsHostData(sessionIdsShapeSize, 0);
        std::vector<int32_t> microBatchIdsHostData(microBatchIdsShapeSize, 0);
        std::vector<int32_t> tokenIdsHostData(tokenIdsShapeSize, 0);
        std::vector<int32_t> expertOffsetsHostData(expertOffsetsShapeSize, 0);
        std::vector<int64_t> actualTokenNumHostData(actualTokenNumShapeSize, 8);
        std::vector<int32_t> attnRankTableHostData(attnRankTableShapeSize);
        for (int32_t i = 0; i < Y; i++) {
            sessionIdsHostData[i] = i % attention_worker_num;
            tokenIdsHostData[i] = i % Y;
            expertOffsetsHostData[i] = i % expert_num_per_token;
        }
        for (int32_t i = 0; i < attention_worker_num; i++) {
            attnRankTableHostData[i] = static_cast<int32_t>(i);
        }


        ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(sessionIdsHostData, sessionIdsShape, &sessionIdsDeviceAddr, aclDataType::ACL_INT32, &sessionIds);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(microBatchIdsHostData, microBatchIdsShape, &microBatchIdsDeviceAddr, aclDataType::ACL_INT32, &microBatchIds);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(tokenIdsHostData, tokenIdsShape, &tokenIdsDeviceAddr, aclDataType::ACL_INT32, &tokenIds);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expertOffsetsHostData, expertOffsetsShape, &expertOffsetsDeviceAddr, aclDataType::ACL_INT32, &expertOffsets);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(actualTokenNumHostData, actualTokenNumShape, &actualTokenNumDeviceAddr,  aclDataType::ACL_INT64, &actualTokenNum);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(attnRankTableHostData, attnRankTableShape, &attnRankTableDeviceAddr, aclDataType::ACL_INT32, &attnRankTable);
        CHECK_RET(ret == ACL_SUCCESS, return ret);


        uint64_t FFN2AttentionWorkspaceSize = 0;
        aclOpExecutor *FFN2AttentionExecutor = nullptr;
        void *FFN2AttentionWorkspaceAddr = nullptr;

        /**************************************** Call FFN2Attention. ********************************************/
        // Call the first-phase API.
        ret = aclnnFFNToAttentionGetWorkspaceSize(x, sessionIds, microBatchIds, tokenIds,
                                                            expertOffsets, actualTokenNum, attnRankTable,
                                                            hcomName, WORLD_SIZE, tokenInfoTableShape, tokenDataShape,
                                                            &FFN2AttentionWorkspaceSize, &FFN2AttentionExecutor);
        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnFFNToAttentionGetWorkspaceSize failed. ret = %d \n", ret); return ret);
        // Allocate device memory based on the workspaceSize calculated by the first-phase API.
        if (FFN2AttentionWorkspaceSize > 0) {
            ret = aclrtMalloc(&FFN2AttentionWorkspaceAddr, FFN2AttentionWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }
        // Call the second-phase API.
        ret = aclnnFFNToAttention(FFN2AttentionWorkspaceAddr, FFN2AttentionWorkspaceSize, FFN2AttentionExecutor, args.FFN2AttentionStream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnFFNToAttention failed. ret = %d \n", ret);
            return ret);
        // (Boilerplate) Wait for the task execution to complete.
        if (args.rankId >= ATTN_NUM) {
            ret = aclrtSynchronizeStreamWithTimeout(args.FFN2AttentionStream, 10000);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
            return ret);
            LOG_PRINT("[INFO] device_%d FFNToAttention execute successfully.\n", args.rankId);
        } else {
            std::this_thread::sleep_for(std::chrono::seconds(10));
            LOG_PRINT("[INFO] device_%d is AttentionWorker, sleeping 10 seconds...\n", args.rankId);
        }


        // Release device resources.
        if (FFN2AttentionWorkspaceSize > 0) {
            aclrtFree(FFN2AttentionWorkspaceAddr);
        }
        if (x != nullptr) {
            aclDestroyTensor(x);
        }
        if (sessionIds != nullptr) {
            aclDestroyTensor(sessionIds);
        }
        if (microBatchIds != nullptr) {
            aclDestroyTensor(microBatchIds);
        }
        if (tokenIds != nullptr) {
            aclDestroyTensor(tokenIds);
        }

        if (expertOffsets != nullptr) {
            aclDestroyTensor(expertOffsets);
        }
        if (actualTokenNum != nullptr) {
            aclDestroyTensor(actualTokenNum);
        }
        if (attnRankTable != nullptr) {
            aclDestroyTensor(attnRankTable);
        }
        if (tokenInfoTableShape != nullptr) {
            aclDestroyIntArray(tokenInfoTableShape);
        }
        if (tokenDataShape != nullptr) {
            aclDestroyIntArray(tokenDataShape);
        }


        if (xDeviceAddr != nullptr) {
            aclrtFree(xDeviceAddr);
        }
        if (sessionIdsDeviceAddr != nullptr) {
            aclrtFree(sessionIdsDeviceAddr);
        }
        if (microBatchIdsDeviceAddr != nullptr) {
            aclrtFree(microBatchIdsDeviceAddr);
        }
        if (tokenIdsDeviceAddr != nullptr) {
            aclrtFree(tokenIdsDeviceAddr);
        }
        if (expertOffsetsDeviceAddr != nullptr) {
            aclrtFree(expertOffsetsDeviceAddr);
        }
        if (actualTokenNumDeviceAddr != nullptr) {
            aclrtFree(actualTokenNumDeviceAddr);
        }
        if (attnRankTableDeviceAddr != nullptr) {
            aclrtFree(attnRankTableDeviceAddr);
        }

        HcclCommDestroy(args.hcclComm);
        aclrtDestroyStream(args.FFN2AttentionStream);
        aclrtDestroyContext(args.context);
        aclrtResetDevice(args.rankId);

        return 0;
    }

    int main(int argc, char *argv[])
    {
        int ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed, ret = %d\n", ret); return ret);

        aclrtStream FFN2AttentionStream[DEV_NUM];
        aclrtContext context[DEV_NUM];
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            ret = aclrtSetDevice(rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed, ret = %d\n", ret); return ret);
            ret = aclrtCreateContext(&context[rankId], rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed, ret = %d\n", ret); return ret);
            ret = aclrtCreateStream(&FFN2AttentionStream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed, ret = %d\n", ret); return ret);
        }

        int32_t devices[WORLD_SIZE];
        for (int32_t deviceId = 0; deviceId < WORLD_SIZE; deviceId++) {
            devices[deviceId] = deviceId ;
        }

        HcclComm comms[WORLD_SIZE];
        ret = HcclCommInitAll(WORLD_SIZE, devices, comms);
        CHECK_RET(ret == ACL_SUCCESS,
                    LOG_PRINT("[ERROR] HcclCommInitAll failed, ret %d\n", ret); return ret);


        Args args[DEV_NUM];
        std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            args[rankId].rankId = rankId;
            args[rankId].hcclComm = comms[rankId];
            args[rankId].FFN2AttentionStream = FFN2AttentionStream[rankId];
            args[rankId].context = context[rankId];
            threads[rankId].reset(new(std::nothrow) std::thread(&LaunchOneProcessFFN2Attention, std::ref(args[rankId])));
        }

        for(uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            threads[rankId]->join();
        }

        aclFinalize();
        LOG_PRINT("[INFO] aclFinalize success\n");

        return 0;
    }
    ```
