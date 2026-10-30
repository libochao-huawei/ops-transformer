# aclnnDenseLightningIndexerSoftmaxLseV2Metadata

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T08:47:12.965Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
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

- Description: This operator is an AICPU operator and the preceding operator of aclnnDenseLightningIndexerSoftmaxLseV2. Based on the input shape, layout, mask, and compression ratio information of the aclnnDenseLightningIndexerSoftmaxLseV2 operator, it computes and outputs the core-partitioned metadata. The output can be used as the metadataOptional input of the aclnnDenseLightningIndexerSoftmaxLseV2 operator, reducing host array access in the tiling phase of the main operator.

  **This operator is not recommended for standalone use. Use it with the aclnnDenseLightningIndexerSoftmaxLseV2 operator to form a complete workflow.**
    1. Receive the shape information of the main operator, including batchSize, maxSeqLenQ, maxSeqLenK, numHeadsQ, numHeadsK, headDim, layout, and mask information.
    2. Calculate the total number of seqs involved in load balancing based on layout and the optional input seqUsedQOptional, and evenly split tasks onto available AIC cores in a strided manner.
    3. After the metadata is output, it is subsequently used as the metadataOptional input of the aclnnDenseLightningIndexerSoftmaxLseV2 operator.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call "aclnnDenseLightningIndexerSoftmaxLseV2MetadataGetWorkspaceSize" to obtain the workspace size, and then call "aclnnDenseLightningIndexerSoftmaxLseV2Metadata" to perform computation.

``` cpp
aclnnStatus aclnnDenseLightningIndexerSoftmaxLseV2MetadataGetWorkspaceSize(
    const aclTensor   *cuSeqLensQOptional,
    const aclTensor   *cuSeqLensKOptional,
    const aclTensor   *seqUsedQOptional,
    const aclTensor   *seqUsedKOptional,
    const aclTensor   *cmpResidualKOptional,
    int64_t            batchSize,
    int64_t            maxSeqLenQ,
    int64_t            maxSeqLenK,
    int64_t            numHeadsQ,
    int64_t            numHeadsK,
    int64_t            headDim,
    char              *layoutQ,
    char              *layoutK,
    int64_t            maskMode,
    int64_t            cmpRatio,
    const aclTensor   *metadata,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor);
```

``` cpp
aclnnStatus aclnnDenseLightningIndexerSoftmaxLseV2Metadata(
    void           *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor  *executor,
    aclrtStream     stream);
```

## aclnnDenseLightningIndexerSoftmaxLseV2MetadataGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1600px"><colgroup>
  <col style="width: 150px">
  <col style="width: 100px">
  <col style="width: 350px">
  <col style="width: 150px">
  <col style="width: 70px">
  <col style="width: 70px">
  <col style="width: 190px">
  <col style="width: 80px">
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
      <td>cuSeqLensQOptional (const aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the cumulative sequence length of queries in different batches.<br>Mandatory in TND use cases, and the batch can be derived from the shape of this input parameter.<br>The first value is fixed to 0.</td>
      <td>Empty tensors are supported.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1 dimension, shape (B+1,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>cuSeqLensKOptional (const aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the cumulative sequence length of keys in different batches.<br>Mandatory in TND use cases.<br>The first value is fixed to 0.</td>
      <td>Supports empty tensors.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D, shape (B+1,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>seqUsedQOptional (const aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the sequence length of queries actually involved in computation for different batches.</td>
      <td>Supports empty tensors. When this parameter exists, the total number of sequences is calculated by accumulating the sequences actually used in each batch.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D, shape (B,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>seqUsedKOptional (const aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the actual sequence length of keys that participate in computation in each batch.</td>
      <td>Supports empty tensors. If this parameter exists, the value of seqused_k is not greater than the actual seqlen_k of each batch.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D, shape (B,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>cmpResidualKOptional (const aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the residual related to cmpRatio for the sequence length of keys in different batches.</td>
      <td>Empty tensors are supported. This parameter is required when maskMode=3 and cmpRatio>1, and each value must be less than cmpRatio.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1-D, shape (B,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>batchSize (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of batches.</td>
      <td>Supports non-negative values. In the TND use case, it can be set to 0 and derived via cuSeqLensQOptional.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqLenQ (int64_t)</td>
      <td>Input</td>
      <td>Indicates the maximum sequence length of query.</td>
      <td>Non-negative values are supported. In BSND use cases, the value must be a positive number.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqLenK (int64_t)</td>
      <td>Input</td>
      <td>Indicates the maximum sequence length of the key.</td>
      <td>Non-negative values are supported. In BSND use cases, the value must be a positive number.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numHeadsQ (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of query heads.</td>
      <td>Must be a positive number, value range [1, 128], and can be divided by numHeadsK.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numHeadsK (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of key heads.</td>
      <td>Must be a positive number. Currently, only 1 is supported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>headDim (int64_t)</td>
      <td>Input</td>
      <td>Indicates the head dimension of q/k.</td>
      <td>Must be a positive number. Currently, only 128 is supported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQ (char*)</td>
      <td>Input</td>
      <td>Indicates the layout format of query.</td>
      <td>Supports BSND/TND. BSND is used if a null pointer is passed.</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutK (char*)</td>
      <td>Input</td>
      <td>Indicates the layout format on the key side.</td>
      <td>Supports BSND/TND. BSND is used if a null pointer is passed. Must be consistent with layoutQ.</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maskMode (int64_t)</td>
      <td>Input</td>
      <td>Indicates the sparse mask mode.</td>
      <td>0: No mask<br>3: Causal</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpRatio (int64_t)</td>
      <td>Input</td>
      <td>Indicates the compression ratio of the key.</td>
      <td>Value range [1, 128].</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metadata (const aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the output of the load balancing result.</td>
      <td>The output is used as the metadataOptional input of aclnnDenseLightningIndexerSoftmaxLseV2.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1-D, shape fixed to (64,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize(uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be requested on the Device side.</td>
      <td>The current implementation returns 0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returned op executor, which contains the operator computing process.</td>
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

    <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
    <col style="width: 300px">
    <col style="width: 150px">
    <col style="width: 550px">
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
      <td>A null pointer exists in metadata, workspaceSize, or executor.</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>The data type of metadata is not within the supported range.</td>
    </tr>
    <tr>
      <td>numHeadsQ, numHeadsK, or headDim is not within the supported range.</td>
    </tr>
    <tr>
      <td>layoutQ is inconsistent with layoutK, or not within the supported range.</td>
    </tr>
    <tr>
      <td>batchSize, maxSeqLenQ, or maxSeqLenK does not meet the value constraint.</td>
    </tr>
    <tr>
      <td>maskMode is not within the supported range.</td>
    </tr>
    <tr>
      <td>cmpRatio is not within the supported range.</td>
    </tr>
    </tbody></table>

## aclnnDenseLightningIndexerSoftmaxLseV2Metadata

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
        </tr></thead>
    <tbody>
        <tr>
        <td>workspace</td>
        <td>Input</td>
        <td>Workspace memory address applied for on the Device side.</td>
        </tr>
        <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Size of the workspace applied for on the Device side, obtained via the first-phase API aclnnDenseLightningIndexerSoftmaxLseV2MetadataGetWorkspaceSize.</td>
        </tr>
        <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Op executor that contains the operator computation flow.</td>
        </tr>
        <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream for executing the task.</td>
        </tr>
    </tbody>
    </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

  - aclnnDenseLightningIndexerSoftmaxLseV2Metadata defaults to a deterministic implementation. The deterministic computation configuration does not change its output rules.

  - BSND use case
    - batchSize, maxSeqLenQ, and maxSeqLenK are required to obtain shape information.
  - TND use case
    - cuSeqLensQOptional and cuSeqLensKOptional are mandatory parameters for obtaining correct shape information.
    - When batchSize is 0, batch is derived from the shape of cuSeqLensQOptional.

<details>
<summary><a id="Mask"></a>Mask</summary>
    &nbsp;&nbsp;<table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
        <col style="width: 165px">
        <col style="width: 625px">
        <col style="width: 360px">
        </colgroup>
        <thead>
            <tr>
                <th>mask_mode</th>
                <th>Description</th>
                <th>Remarks</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>0</td>
            <td>No mask.</td>
            <td>Supported</td>
        </tr>
        <tr>
            <td>3</td>
            <td>Mask in rightDownCausal mode, corresponding to the lower-triangular use case with the right vertex as the partition point.</td>
            <td>Supported</td>
        </tr>
        </tbody>
    </table>
</details>

<details>
<summary><a id="special-constraints"></a>Special Constraints</summary>

  - Batch Value Rules
    - If batchSize is greater than 0, batchSize takes precedence.
    - If batchSize is less than or equal to 0 and layoutQ is TND, batch is derived from the shape of cuSeqLensQOptional.
    - If batchSize is less than or equal to 0 and layoutQ is BSND, an error is reported.
  - Seqlen Value Rules
    - In TND use cases, calculate the actual q/k length of each batch via cuSeqLensQOptional and cuSeqLensKOptional.
    - In BSND use cases, obtain the q/k length via maxSeqLenQ and maxSeqLenK.
  - layout constraints
    - layoutQ must be BSND or TND.
    - layoutK supports BSND and TND, and must be consistent with layoutQ.
  - head constraints
    - numHeadsQ: range [1, 128], and must be divisible by numHeadsK.
    - numHeadsK currently supports only 1.
    - headDim currently supports only 128.
  - load balancing constraints
    - The value range of cmpRatio is [1, 128].
    - Currently, maskMode supports only 0 and 3.
  - cmp_residual_k constraints
    - If maskMode=3 and cmpRatio>1, cmpResidualKOptional must be passed in.
    - Each value of cmpResidualKOptional must be less than cmpRatio.
  - seqused constraints
    - Each value of seqUsedQOptional must not exceed the actual seqlen_q of each batch.
    - Each value of seqUsedKOptional must not exceed the actual seqlen_k of each batch.

</details>

<details>
<summary><a id="Metadata"></a>Metadata Output Layout</summary>

  The metadata output is an INT32 Tensor, and the shape is currently fixed to (64,). The field layout is as follows.

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 180px">
    <col style="width: 150px">
    <col style="width: 820px">
    </colgroup>
    <thead>
        <tr>
        <th>Field</th>
        <th>Index</th>
        <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <td>forecore_num</td>
        <td>0</td>
        <td>Number of fore cores, that is, the number of cores that process more tasks. If totalNum cannot be evenly divided by aicCoreNum, forecore_num = totalNum % aicCoreNum, and each fore core processes ceil(totalNum / aicCoreNum) tasks. If totalNum can be evenly divided by aicCoreNum, forecore_num = aicCoreNum.</td>
        </tr>
        <tr>
        <td>tail_core_num</td>
        <td>1</td>
        <td>Number of tail cores, that is, the number of cores that process fewer tasks. tail_core_num = aicCoreNum - forecore_num, and each tail core processes floor(totalNum / aicCoreNum) tasks. The value is 0 if totalNum can be evenly divided by aicCoreNum.</td>
        </tr>
        <tr>
        <td>b_s1_per_core</td>
        <td>2</td>
        <td>Number of B*S1 tasks processed by each preceding core, that is, ceil(totalNum / aicCoreNum).</td>
        </tr>
        <tr>
        <td>b_s1_per_tail_core</td>
        <td>3</td>
        <td>Number of B*S1 tasks processed by each tail core, that is, floor(totalNum / aicCoreNum). Set to 0 when totalNum can be evenly divided by aicCoreNum.</td>
        </tr>
    </tbody>
  </table>

  Differentiated block partitioning is used for preceding cores and tail cores. Let totalNum be the total number of seqs involved in load balancing and N be aicCoreNum. When totalNum can be evenly divided by N, forecore_num=N, tail_core_num=0, and each core processes totalNum/N tasks. When it cannot be evenly divided, let base=floor(totalNum/N) and rem=totalNum%N. Then forecore_num=rem, each preceding core processes base+1 tasks, tail_core_num=N-rem, and each tail core processes base tasks. For example, when totalNum=16 and N=36, 16<36, forecore_num=16, tail_core_num=0, and b_s1_per_core=1. The first 16 cores each process 1 task, and the remaining cores are idle.

</details>

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

``` cpp
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <utility>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_dense_lightning_indexer_softmax_lse_v2_metadata.h"

#define CHECK_LOG_RET(cond, ret_val, fmt, ...)      \
    do {                                            \
        if (!(cond)) {                              \
            printf(fmt "\n", ##__VA_ARGS__);        \
            return (ret_val);                       \
        }                                           \
    } while (0)

constexpr uint32_t DLI_METADATA_SIZE = 64;

struct DenseLISoftmaxLseV2MetaData {
    int32_t forecore_num;
    int32_t tail_core_num;
    int32_t b_s1_per_core;
    int32_t b_s1_per_tail_core;
};

struct ScopeGuard
{
    explicit ScopeGuard(std::function<void()> onExitScope) : m_exitFunc(std::move(onExitScope)),
        m_isDismissed(false) {}
    ScopeGuard(const ScopeGuard&) = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;

    ~ScopeGuard()
    {
        if (!m_isDismissed) {
            m_exitFunc();
        }
    }

    void Dismiss()
    {
        m_isDismissed = true;
    }

    std::function<void()> m_exitFunc;
    bool m_isDismissed;
};

struct Tensor {
    void *hostAddr { nullptr };
    void *deviceAddr { nullptr };
    aclTensor *data { nullptr };
};

struct ArgScenario {
    bool hasCuSeq { true };
};

struct ArgContext {
    Tensor cuSeqLensQOptional {};
    Tensor cuSeqLensKOptional {};
    Tensor seqUsedQOptional {};
    Tensor seqUsedKOptional {};
    Tensor cmpResidualKOptional {};
    Tensor metadata {};
    int64_t batchSize { 0 };
    int64_t maxSeqLenQ { 0 };
    int64_t maxSeqLenK { 0 };
    int64_t numHeadsQ { 8 };
    int64_t numHeadsK { 1 };
    int64_t headDim { 128 };
    char *layoutQ { nullptr };
    char *layoutK { nullptr };
    int64_t maskMode { 3 };
    int64_t cmpRatio { 4 };
};

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

aclnnStatus Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclInit failed. ERROR: %d", ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtSetDevice failed. ERROR: %d", ret);
    ret = aclrtCreateStream(stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtCreateStream failed. ERROR: %d", ret);
    return ACL_SUCCESS;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

aclnnStatus CreateTensor(aclDataType dataType, const std::vector<int64_t> &shape, Tensor &tensor)
{
    auto size = GetShapeSize(shape) * aclDataTypeSize(dataType);
    auto ret = aclrtMallocHost(&(tensor.hostAddr), size);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMallocHost failed. ERROR: %d", ret);
    memset(tensor.hostAddr, 0, size);

    ret = aclrtMalloc(&(tensor.deviceAddr), size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMalloc failed. ERROR: %d", ret);
    tensor.data = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND,
        shape.data(), shape.size(), tensor.deviceAddr);

    ret = aclrtMemcpy(tensor.deviceAddr, size, tensor.hostAddr, size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMemcpy failed. ERROR: %d", ret);
    return ACL_SUCCESS;
}

void SetInt32TensorData(Tensor &tensor, const std::vector<int32_t> &hostData)
{
    auto size = hostData.size() * sizeof(int32_t);
    memcpy(tensor.hostAddr, hostData.data(), size);
    aclrtMemcpy(tensor.deviceAddr, size, tensor.hostAddr, size, ACL_MEMCPY_HOST_TO_DEVICE);
}

void DestroyTensor(Tensor &tensor)
{
    if (tensor.data != nullptr) {
        aclDestroyTensor(tensor.data);
        tensor.data = nullptr;
    }
    if (tensor.deviceAddr != nullptr) {
        aclrtFree(tensor.deviceAddr);
        tensor.deviceAddr = nullptr;
    }
    if (tensor.hostAddr != nullptr) {
        aclrtFreeHost(tensor.hostAddr);
        tensor.hostAddr = nullptr;
    }
}

void DestroyArgs(ArgContext &context)
{
    DestroyTensor(context.metadata);
    DestroyTensor(context.cuSeqLensQOptional);
    DestroyTensor(context.cuSeqLensKOptional);
    DestroyTensor(context.seqUsedQOptional);
    DestroyTensor(context.seqUsedKOptional);
    DestroyTensor(context.cmpResidualKOptional);

    if (context.layoutQ != nullptr) {
        free(context.layoutQ);
        context.layoutQ = nullptr;
    }
    if (context.layoutK != nullptr) {
        free(context.layoutK);
        context.layoutK = nullptr;
    }
}

aclnnStatus CreateArgs(const ArgScenario &scenario, ArgContext &context)
{
    ScopeGuard argsGuard([&] { DestroyArgs(context); });
    aclnnStatus ret;

    int64_t batchSize = 1;
    context.maxSeqLenQ = 16;
    context.maxSeqLenK = 4;
    context.layoutQ = (char *)malloc(sizeof(char) * 16);
    context.layoutK = (char *)malloc(sizeof(char) * 16);
    strcpy(context.layoutQ, scenario.hasCuSeq ? "TND" : "BSND");
    strcpy(context.layoutK, scenario.hasCuSeq ? "TND" : "BSND");

    ret = CreateTensor(aclDataType::ACL_INT32, { DLI_METADATA_SIZE }, context.metadata);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create metadata failed. Error: %d", ret);

    if (scenario.hasCuSeq) {
        ret = CreateTensor(aclDataType::ACL_INT32, { batchSize + 1 }, context.cuSeqLensQOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cuSeqLensQOptional failed. Error: %d", ret);
        ret = CreateTensor(aclDataType::ACL_INT32, { batchSize + 1 }, context.cuSeqLensKOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cuSeqLensKOptional failed. Error: %d", ret);
        SetInt32TensorData(context.cuSeqLensQOptional, { 0, static_cast<int32_t>(context.maxSeqLenQ) });
        SetInt32TensorData(context.cuSeqLensKOptional, { 0, static_cast<int32_t>(context.maxSeqLenK) });
        context.batchSize = 0;
    } else {
        context.batchSize = batchSize;
    }

    // cmp_residual_k is required when maskMode=3 and cmpRatio>1
    if (context.maskMode == 3 && context.cmpRatio > 1) {
        ret = CreateTensor(aclDataType::ACL_INT32, { batchSize }, context.cmpResidualKOptional);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create cmpResidualKOptional failed. Error: %d", ret);
        SetInt32TensorData(context.cmpResidualKOptional, std::vector<int32_t>(batchSize, 0));
    }

    argsGuard.Dismiss();
    return ACL_SUCCESS;
}

void PrintMetadata(const DenseLISoftmaxLseV2MetaData &metadata)
{
    printf("forecore_num      : %d\n", metadata.forecore_num);
    printf("tail_core_num     : %d\n", metadata.tail_core_num);
    printf("b_s1_per_core     : %d\n", metadata.b_s1_per_core);
    printf("b_s1_per_tail_core: %d\n", metadata.b_s1_per_tail_core);
}

int main() {
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Init acl failed. ERROR: %d", ret);
    ScopeGuard sysGuard([&] { Finalize(deviceId, stream); });

    ArgScenario scenario {};
    scenario.hasCuSeq = true;
    ArgContext context {};
    ret = CreateArgs(scenario, context);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "Create input arguments failed. ERROR: %d", ret);
    ScopeGuard argsGuard([&] { DestroyArgs(context); });

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;
    ret = aclnnDenseLightningIndexerSoftmaxLseV2MetadataGetWorkspaceSize(
        context.cuSeqLensQOptional.data, context.cuSeqLensKOptional.data, context.seqUsedQOptional.data,
        context.seqUsedKOptional.data, context.cmpResidualKOptional.data, context.batchSize, context.maxSeqLenQ,
        context.maxSeqLenK, context.numHeadsQ, context.numHeadsK, context.headDim, context.layoutQ,
        context.layoutK, context.maskMode, context.cmpRatio, context.metadata.data, &workspaceSize, &executor);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret,
        "aclnnDenseLightningIndexerSoftmaxLseV2MetadataGetWorkspaceSize failed. ERROR: %d", ret);

    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "allocate workspace failed. ERROR: %d", ret);
    }
    ScopeGuard workspaceGuard([&] {
        if (workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
            workspaceAddr = nullptr;
        }
    });

    ret = aclnnDenseLightningIndexerSoftmaxLseV2Metadata(workspaceAddr, workspaceSize, executor, stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret,
        "aclnnDenseLightningIndexerSoftmaxLseV2Metadata failed. ERROR: %d", ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtSynchronizeStream failed. ERROR: %d", ret);

    DenseLISoftmaxLseV2MetaData result {};
    ret = aclrtMemcpy(&result, sizeof(result), context.metadata.deviceAddr, sizeof(result), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMemcpy failed. ERROR: %d", ret);
    PrintMetadata(result);
    printf("pass\n");

    return 0;
}
```

## Troubleshooting

- For common execution issues or exception errors of the AI CPU operator Kernel, see "[Fault Cases>Operator Execution Issues>AI CPU Operator Kernel Execution Error](https://www.hiascend.com/document/detail/en/canncommercial/latest/maintenref/troubleshooting/troubleshooting_0151.html)" in *Fault Handling*.
