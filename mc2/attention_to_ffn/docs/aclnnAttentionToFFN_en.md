# aclnnAttentionToFFN

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T13:09:05.139Z -->

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

Sends token data on the Attention node to the FFN node.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnAttentionToFFNGetWorkspaceSize" first to obtain the input parameters and calculate the required workspace size based on the computation process, and then call "aclnnAttentionToFFN" to perform the computation.

```c++
aclnnStatus aclnnAttentionToFFNGetWorkspaceSize(
    const aclTensor    *x,
    const aclTensor    *sessionId,
    const aclTensor    *microBatchId,
    const aclTensor    *layerId,
    const aclTensor    *expertIds,
    const aclTensor    *expertRankTable,
    const aclTensor    *scalesOptional,
    const aclTensor    *activeMaskOptional,
    const char         *group,
    int64_t             worldSize,
    const aclIntArray  *ffnTokenInfoTableShape,
    const aclIntArray  *ffnTokenDataShape,
    const aclIntArray  *attnTokenInfoTableShape,
    int64_t             moeExpertNum,
    int64_t             quantMode,
    int64_t             syncFlag,
    int64_t             ffnStartRankId,
    uint64_t           *workspaceSize,
    aclOpExecutor     **executor)
```

```c++
aclnnStatus aclnnAttentionToFFN(
    void            *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream     stream)
```

## aclnnAttentionToFFNGetWorkspaceSize

- **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
      <col style="width: 170px">
      <col style="width: 120px">
      <col style="width: 300px">
      <col style="width: 300px">
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
        <td>x</td>
        <td>Input</td>
        <td>Token data sent by this device.</td>
        <td>Shape is (X, Bs, H).</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>sessionId</td>
        <td>Input</td>
        <td>Indicates the ID of the current Attention Worker node.</td>
        <td>Shape is (X, ), and the value range is [0, attentionWorkerNum).</td>
        <td>INT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>microBatchId</td>
        <td>Input</td>
        <td>Indicates the microBatch ID.</td>
        <td>Shape is (X, ).</td>
        <td>INT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>layerId</td>
        <td>Input</td>
        <td>Indicates the ID of the current model layer.</td>
        <td>The shape is (X, ).</td>
        <td>INT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>expertIds</td>
        <td>Input</td>
        <td>Indicates the topK expert indices of each token in each micro batch group.</td>
        <td>Shape is (X, Bs, K), value range is [0, moeExpertNum).</td>
        <td>INT32</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>expertRankTable</td>
        <td>Input</td>
        <td>Mapping table from expert ID to expert deployment on FFN devices in each micro batch group (the caller must ensure that the values are correct).</td>
        <td>Shape (L, moeExpertNum + sharedExpertNum, M).</td>
        <td>INT32</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>scalesOptional</td>
        <td>Input</td>
        <td>Quantization smoothing parameter of each expert.</td>
        <td>In non-quantization use cases, a null pointer must be passed. In dynamic quantization, valid data or a null pointer can be passed. The shape is (L, moeExpertNum + sharedExpertNum, H).</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>activeMaskOptional</td>
        <td>Input</td>
        <td>Indicates whether tokens participate in communication. Valid data or a null pointer can be passed.</td>
        <td>If a null pointer is passed, all tokens participate in communication by default. If a value is passed, the shape is (X, Bs). true indicates that the token participates in communication, and true must be placed before false.</td>
        <td>BOOL</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>group</td>
        <td>Input</td>
        <td>Communicator name (expert parallel).</td>
        <td>String length range: [1, 128).</td>
        <td>STRING</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>worldSize</td>
        <td>Input</td>
        <td>Indicates the communicator size.</td>
        <td>Value range: [2, 768].</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>ffnTokenInfoTableShape</td>
        <td>Input</td>
        <td>Indicates the list of shape sizes of the token information table on the FFN node.</td>
        <td>Length of 3, includes the number of Attention nodes, the size of microBatchSize, and the size of the shape of the sending status information corresponding to each token.</td>
        <td>INT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>ffnTokenDataShape</td>
        <td>Input</td>
        <td>Indicates the list of shape sizes of the token data table on the FFN node.</td>
        <td>Length of 5, including the number of Attention nodes, the size of microBatchSize, the size of batchSize, the number of experts each token needs to send to (including shared experts), and the length of a single token.</td>
        <td>INT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>attnTokenInfoTableShape</td>
        <td>Input</td>
        <td>Indicates the list of shape sizes of the token information table on the Attention node.</td>
        <td>Length of 3, including the size of microBatchSize, the size of batchSize, and the number of experts each token needs to send to (including shared experts).</td>
        <td>INT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>moeExpertNum</td>
        <td>Input</td>
        <td>Number of MoE experts.</td>
        <td>Value range: (0, 1024].</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantMode</td>
        <td>Input</td>
        <td>Indicates the quantization mode.</td>
        <td>Only 0 (non-quantization) and 2 (dynamic quantization) are supported.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>syncFlag</td>
        <td>Input</td>
        <td>Indicates the FFN node synchronization mode.</td>
        <td>Only 0 (synchronous) and 1 (asynchronous) are supported.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>ffnStartRankId</td>
        <td>Input</td>
        <td>Indicates the start ID of the FFN node.</td>
        <td>The value range is [0, worldSize).</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the size of the workspace to be requested on the Device side.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Returns the op executor that contains the operator computation process.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>

- **Return value**

    Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API performs input parameter validation and returns an error in the following scenarios:

    <table style="undefined;table-layout: fixed; width: 1150px"> <colgroup>
    <col style="width: 280px">
    <col style="width: 100px">
    <col style="width: 900px">
      </colgroup><thead>
      <tr>
        <th>Return Value</th>
        <th>Error Code</th>
        <th>Description</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>The mandatory input and output Tensor parameter is a null pointer.</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td>The data type of the input and output is not supported.</td>
      </tr>
      <tr>
        <td rowspan="2">ACLNN_ERR_INNER_TILING_ERROR</td>
        <td rowspan="2">561002</td>
        <td>The shape of the input and output is not supported.</td>
      </tr>
      <tr>
        <td>The parameter value is not supported.</td>
      </tr>
    </tbody>
    </table>

## aclnnAttentionToFFN

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1150px"> <colgroup>
    <col style="width: 150px">
    <col style="width: 100px">
    <col style="width: 900px">
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
        <td>Size of the workspace allocated on the Device side, obtained by the first-phase API <code>aclnnAttentionToFFNGetWorkspaceSize</code>.</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Op executor, which includes the operator computing process.</td>
    </tr>
    <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Specifies the Stream for executing tasks.</td>
    </tr>
    </tbody></table>

- **Return value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- **Deterministic constraints**:
  - `aclnnAttentionToFFN` defaults to a deterministic implementation.

- **Parameter Consistency Constraints**:
  - The `group`, `worldSize`, `moeExpertNum`, `ffnTokenInfoTableShape`, `ffnTokenDataShape`, `ffnStartRankId` parameters and the `HCCL_BUFFSIZE` value used during the API call must be consistent on all devices, consistent across different layers of the network, and consistent with the corresponding parameters of the separated use case operator series.

- **Product-Specific Constraints**:

  <!-- npu="A3" id7 -->
  - <term>Atlas A3 products</term>: In this use case, a single device contains dual DIE (simply referred to as "die" or "bare die"), so "the current device" in the parameter description refers to a single DIE.

  <!-- end id7 -->

- **Shape Variable Constraints**:

  | Variable | Definition and value range |
  | :--- | :--- |
  | X | Indicates the micro batch sequence size (number of token groups). Only <code>X = 1</code> is supported in the current version. |
  | H (hidden size) | Indicates the hidden layer size, value range <code>[1024, 8192]</code>. |
  | Bs | Indicates the batch sequence size (number of tokens finally output on the current device), value range <code>0 < Bs ≤ 512</code>. |
  | K | Indicates that topK experts are selected, value range <code>0 < K ≤ 16</code>, and <code>0 < K ≤ moeExpertNum</code>. |
  | L | Indicates the number of model layers. Only <code>L = 1</code> is supported in the current version. |
  | M | Indicates the length of the last dimension of expertRankTable, specifically the length of the deployment information list of the expert with the maximum number of deployments on FFN nodes. Value range: <code>1 < M ≤ FFNWorkerNum * 2 + 1</code>. |
  | moeExpertNum | Indicates the number of MoE experts, value range <code>(0, 1024]</code>. |
  | sharedExpertNum | Indicates the number of shared experts (a shared expert can be replicated and deployed on multiple ffnRank devices), value range <code>[0, 4]</code>. |
  | moeExpertRankNum | Indicates the number of FFN nodes on which MoE experts are deployed, value range <code>0 < moeExpertRankNum < FFNWorkerNum</code>. |
  | sharedExpertRankNum | Indicates the number of FFN nodes on which shared experts are deployed, value range <code>0 < sharedExpertRankNum ≤ FFNWorkerNum</code>. |
  | FFNWorkerNum | Indicates the number of FFN nodes, value range <code>0 < FFNWorkerNum < worldSize</code>, and satisfies <code>FFNWorkerNum = moeExpertRankNum + sharedExpertRankNum</code>. |
  | AttentionWorkerNum | Indicates the number of Attention nodes, value range <code>0 < AttentionWorkerNum < worldSize</code>.

- **Environment Variable Constraints**:
  - **HCCL_BUFFSIZE**: Before calling this API, check whether the value of the HCCL_BUFFSIZE environment variable is reasonable. This environment variable indicates the memory size occupied by a single communication domain, in MB. The default value is 200 MB when not configured.

- **Communication Domain Usage Constraints**:
  - No other operators are allowed in the communicator of the AttentionToFFN operator.

## Example

File preparation:

1.Create the AttentionToFFNDemo directory. Create the aclnnAttentionToFFNDemo.cpp and AttentionToFFN.sh files under AttentionToFFNDemo as instructed below, and modify them with reference to the following code.

2.Install the CANN package, and compile and run AttentionToFFNDemo as instructed below.

AttentionToFFN.sh build script

```bash
#!/bin/bash
cann_path="/path/to/cann_env" # Change the path of the CANN package environment.
g++ "aclnnAttentionToFFNDemo.cpp" -o AttentionToFFNDemo -I"$cann_path/latest/include/" -I"$cann_path/latest/include/aclnnop/" \
                    -L="$cann_path/latest/lib64/" -lascendcl -lnnopbase -lopapi_math -lop_common -lpthread -lhccl
```

Compile and run:

```bash
# Source the CANN environment.
source /path/to/cann_env/latest/bin/setenv.bash

# Compile aclnnAttentionToFFNDemo.cpp.
bash AttentionToFFN.sh

./AttentionToFFNDemo
```

The sample code is as follows, for reference only.

```c++
#include <thread>
#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_attention_to_ffn.h"

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
    aclrtStream attentionToFFNStream;
    aclrtContext context;
};

constexpr uint32_t WORLD_SIZE = 16;
constexpr uint32_t FFN_WORKER_NUM = 5;
constexpr uint32_t ATTENTION_WORKER_NUM = WORLD_SIZE - FFN_WORKER_NUM;

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
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int LaunchOneProcessAttentionToFFN(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed, ret %d\n", ret); return ret);

    char hcomName[128] = {0};
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed, ret %d\n", ret); return -1);
    LOG_PRINT("[INFO] rank = %d, hcomName = %s, attentionToFFNStream = %p, context = %p\n",
              args.rankId, hcomName, args.attentionToFFNStream, args.context);

    int64_t X = 1;
    int64_t L = 1;
    int64_t Bs = 8;
    int64_t H = 7168;
    int64_t K = 4;
    int64_t sharedExpertNum = 1;
    int64_t sharedExpertRankNum = 1;
    int64_t moeExpertNum = 8;
    int64_t expertNumPerToken = K + sharedExpertNum;
    int64_t M = 2 *(FFN_WORKER_NUM - sharedExpertNum * sharedExpertRankNum) + 1;
    int64_t quantMode = 0;
    int64_t syncFlag = 0;
    int64_t ffnStartRankId = 0;
    int64_t ffnTokenInfoTableShapeData[] = {ATTENTION_WORKER_NUM , X, 2 + Bs * expertNumPerToken};
    int64_t ffnTokenDataShapeData[] = {ATTENTION_WORKER_NUM, X, Bs, expertNumPerToken, H};
    int64_t attnTokenInfoTableShapeData[] = {X, Bs, expertNumPerToken};

    /* Construct device-side input and output variables based on the current use case. */
    // Declare device-side input and output variables.
    void *xDeviceAddr = nullptr;
    void *sessionIdDeviceAddr = nullptr;
    void *microBatchIdDeviceAddr = nullptr;
    void *layerIdDeviceAddr = nullptr;
    void *expertIdsDeviceAddr = nullptr;
    void *expertRankTableDeviceAddr = nullptr;
    void *scalesDeviceAddr = nullptr;

    aclTensor *x = nullptr;
    aclTensor *sessionId = nullptr;
    aclTensor *microBatchId = nullptr;
    aclTensor *layerId = nullptr;
    aclTensor *expertIds = nullptr;
    aclTensor *expertRankTable = nullptr;
    aclTensor *scales = nullptr;
    aclIntArray *ffnTokenInfoTableShape = aclCreateIntArray(ffnTokenInfoTableShapeData, 3);
    aclIntArray *ffnTokenDataShape = aclCreateIntArray(ffnTokenDataShapeData, 5);
    aclIntArray *attnTokenInfoTableShape = aclCreateIntArray(attnTokenInfoTableShapeData, 3);

    // Define dimensions of each variable for the current use case.
    std::vector<int64_t> xShape{X, Bs, H};
    std::vector<int64_t> sessionIdShape{X};
    std::vector<int64_t> microBatchIdShape{X};
    std::vector<int64_t> layerIdShape{X};
    std::vector<int64_t> expertIdsShape{X, Bs, K};
    std::vector<int64_t> expertRankTableShape{L, moeExpertNum + sharedExpertNum, M};
    std::vector<int64_t> scalesShape{L, moeExpertNum + sharedExpertNum, H};

    int64_t xShapeSize = GetShapeSize(xShape);
    int64_t sessionIdShapeSize = GetShapeSize(sessionIdShape);
    int64_t microBatchIdShapeSize = GetShapeSize(microBatchIdShape);
    int64_t layerIdShapeSize = GetShapeSize(layerIdShape);
    int64_t expertIdsShapeSize = GetShapeSize(expertIdsShape);
    int64_t expertRankTableShapeSize = GetShapeSize(expertRankTableShape);
    int64_t scalesShapeSize = GetShapeSize(scalesShape);

    // Construct host-side variables.
    std::vector<int16_t> xHostData(xShapeSize, 1);
    std::vector<int16_t> sessionIdHostData(sessionIdShapeSize, args.rankId - FFN_WORKER_NUM);
    std::vector<int16_t> microBatchIdHostData(microBatchIdShapeSize, 0);
    std::vector<int16_t> layerIdHostData(layerIdShapeSize, 0);
    std::vector<int32_t> expertIdsHostData;
    for (int32_t micro_batch_id = 0; micro_batch_id < expertIdsShape[0]; micro_batch_id++) {
        for (int32_t token_id = 0; token_id < expertIdsShape[1]; token_id++) {
            for (int32_t k_id = 0; k_id < expertIdsShape[2]; k_id++) {
                expertIdsHostData.push_back(k_id);
            }
        }
    }

    std::vector<int32_t> expertRankTableHostData = {4, 2, 4, 3, 7, 1, 3, 2, 5, 2, 2, 5, 1, 2, 0, 0, 0, 0,
                                                    3, 2, 5, 0, 0, 3, 7, 0, 0, 4, 1, 3, 0, 1, 2, 4, 3, 7,
                                                    4, 0, 0, 3, 6, 1, 3, 2, 5, 3, 3, 7, 2, 4, 1, 2, 0, 0,
                                                    2, 2, 5, 0, 0, 0, 0, 0, 0, 3, 3, 6, 2, 5, 3, 7, 0, 0,
                                                    1, 4, 8, 0, 0, 0, 0, 0, 0};

    std::vector<float> scalesHostData(scalesShapeSize, 0.1);

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sessionIdHostData, sessionIdShape, &sessionIdDeviceAddr, aclDataType::ACL_INT32, &sessionId);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(microBatchIdHostData, microBatchIdShape, &microBatchIdDeviceAddr, aclDataType::ACL_INT32, &microBatchId);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(layerIdHostData, layerIdShape, &layerIdDeviceAddr, aclDataType::ACL_INT32, &layerId);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertIdsHostData, expertIdsShape, &expertIdsDeviceAddr, aclDataType::ACL_INT32, &expertIds);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertRankTableHostData, expertRankTableShape, &expertRankTableDeviceAddr, aclDataType::ACL_INT32, &expertRankTable);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(scalesHostData, scalesShape, &scalesDeviceAddr, aclDataType::ACL_FLOAT, &scales);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t attentionToFFNWorkspaceSize = 0;
    aclOpExecutor *attentionToFFNExecutor = nullptr;
    void *attentionToFFNWorkspaceAddr = nullptr;

    /**************************************** Call AttentionToFFN ********************************************/
    // Call the first-phase API.
    ret = aclnnAttentionToFFNGetWorkspaceSize(x, sessionId, microBatchId, layerId, expertIds, expertRankTable, (quantMode > 0 ? scales : nullptr),
                                              nullptr, hcomName, WORLD_SIZE, ffnTokenInfoTableShape, ffnTokenDataShape, attnTokenInfoTableShape,
                                              moeExpertNum, quantMode, syncFlag, ffnStartRankId, &attentionToFFNWorkspaceSize, &attentionToFFNExecutor);

    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] aclnnAttentionToFFNGetWorkspaceSize failed. ret = %d \n", ret); return ret);

    // Request device memory based on the workspaceSize calculated by the first-phase API.
    if (attentionToFFNWorkspaceSize > 0) {
        ret = aclrtMalloc(&attentionToFFNWorkspaceAddr, attentionToFFNWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
    }

    if (args.rankId < FFN_WORKER_NUM) {  // FFN Worker
        // Wait for the Attention Worker task to complete.
        LOG_PRINT("[INFO] device_%d is FFN worker, skipping aclnnAttentionToFFN execute.\n", args.rankId);
        std::this_thread::sleep_for(std::chrono::seconds(30));
    } else {    // Attention Worker
        // Call the second-phase API.
        ret = aclnnAttentionToFFN(attentionToFFNWorkspaceAddr, attentionToFFNWorkspaceSize,
                                    attentionToFFNExecutor, args.attentionToFFNStream);

        // (Boilerplate) Synchronize and wait for the task to complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.attentionToFFNStream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnAttentionToFFN failed. ret = %d \n", ret);  \
            return ret);

        LOG_PRINT("[INFO] device_%d aclnnAttentionToFFN execute successfully.\n", args.rankId);
    }

    // Release device resources.
    if (attentionToFFNWorkspaceSize > 0) {
        aclrtFree(attentionToFFNWorkspaceAddr);
    }

    if (x != nullptr) {
        aclDestroyTensor(x);
    }
    if (sessionId != nullptr) {
        aclDestroyTensor(sessionId);
    }
    if (microBatchId != nullptr) {
        aclDestroyTensor(microBatchId);
    }
    if (layerId != nullptr) {
        aclDestroyTensor(layerId);
    }
    if (expertIds != nullptr) {
        aclDestroyTensor(expertIds);
    }
    if (expertRankTable != nullptr) {
        aclDestroyTensor(expertRankTable);
    }
    if (scales != nullptr) {
        aclDestroyTensor(scales);
    }
    if (ffnTokenInfoTableShape != nullptr) {
        aclDestroyIntArray(ffnTokenInfoTableShape);
    }
    if (ffnTokenDataShape != nullptr) {
        aclDestroyIntArray(ffnTokenDataShape);
    }
    if (attnTokenInfoTableShape != nullptr) {
        aclDestroyIntArray(attnTokenInfoTableShape);
    }

    if (xDeviceAddr != nullptr) {
        aclrtFree(xDeviceAddr);
    }
    if (sessionIdDeviceAddr != nullptr) {
        aclrtFree(sessionIdDeviceAddr);
    }
    if (microBatchIdDeviceAddr != nullptr) {
        aclrtFree(microBatchIdDeviceAddr);
    }
    if (layerIdDeviceAddr != nullptr) {
        aclrtFree(layerIdDeviceAddr);
    }
    if (expertIdsDeviceAddr != nullptr) {
        aclrtFree(expertIdsDeviceAddr);
    }
    if (expertRankTableDeviceAddr != nullptr) {
        aclrtFree(expertRankTableDeviceAddr);
    }
    if (scalesDeviceAddr != nullptr) {
        aclrtFree(scalesDeviceAddr);
    }

    HcclCommDestroy(args.hcclComm);
    aclrtDestroyStream(args.attentionToFFNStream);
    aclrtDestroyContext(args.context);
    aclrtResetDevice(args.rankId);

    return 0;
}

int main(int argc, char *argv[])
{
    // This sample is implemented based on Atlas A3 and must run on Atlas A3.
    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed, ret = %d\n", ret); return ret);

    aclrtStream attentionToFFNStream[WORLD_SIZE];
    aclrtContext context[WORLD_SIZE];

    for (uint32_t rankId = 0; rankId < WORLD_SIZE; rankId++) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed, ret = %d\n", ret); return ret);
        ret = aclrtCreateContext(&context[rankId], rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed, ret = %d\n", ret); return ret);
        ret = aclrtCreateStream(&attentionToFFNStream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed, ret = %d\n", ret); return ret);
    }

    int32_t devices[WORLD_SIZE];
    for (int32_t id = 0; id < WORLD_SIZE; id++) {
        devices[id] = id;
    }

    HcclComm comms[WORLD_SIZE];
    ret = HcclCommInitAll(WORLD_SIZE, devices, comms);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[ERROR] HcclCommInitAll failed, ret %d\n", ret); return ret);

    Args args[WORLD_SIZE];
    std::vector<std::unique_ptr<std::thread>> threads(WORLD_SIZE);
    for (uint32_t rankId = 0; rankId < WORLD_SIZE; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].hcclComm = comms[rankId];
        args[rankId].attentionToFFNStream = attentionToFFNStream[rankId];
        args[rankId].context = context[rankId];
        threads[rankId].reset(new(std::nothrow) std::thread(&LaunchOneProcessAttentionToFFN, std::ref(args[rankId])));
    }

    for(uint32_t rankId = 0; rankId < WORLD_SIZE; rankId++) {
        threads[rankId]->join();
    }

    aclFinalize();
    LOG_PRINT("[INFO] aclFinalize success\n");

    return 0;
}
```
