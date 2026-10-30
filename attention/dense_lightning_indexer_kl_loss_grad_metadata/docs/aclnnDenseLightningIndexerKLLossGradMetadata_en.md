# aclnnDenseLightningIndexerKLLossGradMetadata

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T08:37:39.049Z -->

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

- Description: This operator is an AICPU operator and the preceding operator of aclnnDenseLightningIndexerKLLossGrad. Based on the input shape, layout, mask, and compression ratio information of the aclnnDenseLightningIndexerKLLossGrad operator, this operator computes and outputs the core-splitting metadata. The output can be used as the metadataOptional input of the aclnnDenseLightningIndexerKLLossGrad operator, reducing host array access in the tiling phase of the main operator.

  **This operator is not recommended for standalone use. It is recommended to use it together with the aclnnDenseLightningIndexerKLLossGrad operator to form a complete workflow.**
    1. Receives the shape information of the main operator, including batchSize, maxSeqLenQ, maxSeqLenK, numHeadsQ, numHeadsK, headDim, layout, and mask information.
    2. Estimate the load based on the valid sequence length of each query, and evenly split the tasks with merged B/S1 axes across the available AIC cores.
    3. After the metadata is output, it is used later as the metadataOptional input of the aclnnDenseLightningIndexerKLLossGrad operator.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call "aclnnDenseLightningIndexerKLLossGradMetadataGetWorkspaceSize" to obtain the workspace size, and then call "aclnnDenseLightningIndexerKLLossGradMetadata" to perform the computation.

```cpp
aclnnStatus aclnnDenseLightningIndexerKLLossGradMetadataGetWorkspaceSize(
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
    aclOpExecutor    **executor)
```

```cpp
aclnnStatus aclnnDenseLightningIndexerKLLossGradMetadata(
    void           *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor  *executor,
    aclrtStream     stream)
```

## aclnnDenseLightningIndexerKLLossGradMetadataGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 350px">
  <col style="width: 250px">
  <col style="width: 100px">
  <col style="width: 100px">
  <col style="width: 100px">
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
      <td>Indicates the cumulative sequence length of query in different batches.</td>
      <td><ul><li>Supports empty tensor</li><li>Required in TND use cases. The batch can be derived from the shape of this input parameter.</li><li>The first value is fixed to 0.</li><li>The shape is fixed to (B+1, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D, shape (B+1,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>cuSeqLensKOptional (const aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the cumulative sequence length of key in different batches.</td>
      <td><ul><li>Supports empty tensor.</li><li>Mandatory in TND use cases.</li><li>The first value is fixed to 0.</li><li>The shape is fixed to (B+1, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1D, shape (B+1,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>seqUsedQOptional (const aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the actual sequence length of queries participating in computation in different batches.</td>
      <td><ul><li>Supports empty tensor.</li><li>The shape is fixed to (B, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1-dim, shape (B,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>seqUsedKOptional (const aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the sequence length of keys actually involved in the computation in different batches.</td>
      <td><ul><li>Supports empty Tensor.</li><li>The shape is fixed to (B, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1-dim, shape (B,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>cmpResidualKOptional(const aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the residual related to cmpRatio of the key sequence length in different batches.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The shape is fixed to (B, ).</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1-D, shape (B,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>batchSize(int64_t)</td>
      <td>Input</td>
      <td>Indicates the batch quantity.</td>
      <td><ul><li>Supports non-negative values. In TND use cases, this can be set to 0 and derived through cuSeqLensQOptional.</li><li>The recommended value is 0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqLenQ (int64_t)</td>
      <td>Input</td>
      <td>Indicates the maximum sequence length of query.</td>
      <td><ul><li>Supports non-negative numbers. In BSND layout, it must be positive.</li><li>The recommended value is 0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqLenK (int64_t)</td>
      <td>Input</td>
      <td>Indicates the maximum sequence length of the key.</td>
      <td><ul><li>Supports non-negative numbers. In BSND layout, it must be positive.</li><li>The recommended value is 0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numHeadsQ (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of query heads.</td>
      <td>Must be positive and divisible by numHeadsK. The currently supported range is [1, 128].</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numHeadsK (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of key heads.</td>
      <td>Must be positive. Currently only supports 1.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>headDim (int64_t)</td>
      <td>Input</td>
      <td>Indicates the head dimension of q/k.</td>
      <td>Must be positive. Currently only supports 128.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQ (char*)</td>
      <td>Input</td>
      <td>Indicates the layout on the query side.</td>
      <td><ul><li>Supports BSND and TND.</li><li>The recommended value is BSND.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutK (char*)</td>
      <td>Input</td>
      <td>Indicates the layout on the key side.</td>
      <td><ul><li>Supports BSND and TND.</li><li>The recommended value is BSND.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maskMode (int64_t)</td>
      <td>Input</td>
      <td>Indicates the mask mode.</td>
      <td><ul><li>0: No mask.</li><li>3: mask in rightDownCausal mode, corresponding to the lower triangular use case where the right vertex is the dividing point.</li><li>The recommended value is 0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpRatio (int64_t)</td>
      <td>Input</td>
      <td>Indicates the compression ratio of key.</td>
      <td><ul><li>range [1, 128]</li><li>The recommended value is 1, indicating no compression.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metadata (const aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the load balancing result output.</td>
      <td>The output is used as the metadataOptional input of aclnnDenseLightningIndexerKLLossGrad.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1-D, shape fixed to (64,)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be applied on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator calculation process.</td>
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
      <td>ACLNN_ERR_INNER_NULLPTR</td>
      <td>561103</td>
      <td>workspaceSize or executor is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>The data type or dimension of cuSeqLensQOptional, cuSeqLensKOptional, seqUsedQOptional, seqUsedKOptional, or cmpResidualKOptional is not within the supported range.</td>
    </tr>
    <tr>
      <td>metadata is a null pointer, or its data type, dimension, or shape is not within the supported range.</td>
    </tr>
    <tr>
      <td>numHeadsQ, numHeadsK, or headDim is not within the supported range.</td>
    </tr>
    <tr>
      <td>layoutQ or layoutK is a null pointer, or is not within the supported range.</td>
    </tr>
    <tr>
      <td>batchSize, maxSeqLenQ, maxSeqLenK, maskMode, or cmpRatio does not meet the value constraints.</td>
    </tr>
    <tr>
      <td>In the TND use case, the corresponding cuSeqLensQOptional or cuSeqLensKOptional is not passed in; or cmpResidualKOptional is not passed in when maskMode=3 and cmpRatio is not 1.</td>
    </tr>
    <tr>
      <td>The batch quantities derived from query, key, or cmpResidualKOptional are inconsistent.</td>
    </tr>
    </tbody></table>

## aclnnDenseLightningIndexerKLLossGradMetadata

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
    <col style="width: 180px">
    <col style="width: 120px">
    <col style="width: 700px">
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
        <td>Workspace memory address applied on the Device side.</td>
        </tr>
        <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Workspace size applied on the Device side, obtained through the first-phase API aclnnDenseLightningIndexerKLLossGradMetadataGetWorkspaceSize.</td>
        </tr>
        <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Op executor that includes the operator computation process.</td>
        </tr>
        <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream used to execute tasks.</td>
        </tr>
    </tbody>
    </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation: aclnnDenseLightningIndexerKLLossGradMetadata defaults to a deterministic implementation.
- BSND use case
  - batchSize, maxSeqLenQ, and maxSeqLenK are required to obtain the shape information.
- TND use case
  - Pass the cuSeqLensQOptional and cuSeqLensKOptional parameters to obtain correct shape information.
  - When batchSize is 0, derive the batch through the shape of cuSeqLensQOptional.

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
            <td>Mask in rightDownCausal mode, corresponding to the lower triangular case where the right vertex is the dividing point.</td>
            <td>Supported</td>
        </tr>
        </tbody>
    </table>
</details>

<details>
<summary><a id="special-constraints"></a>Special Constraints</summary>

  - Batch value rules
    - If batchSize is greater than 0, batchSize takes precedence.
    - If batchSize is less than or equal to 0 and layoutQ is TND, derive the batch through the shape of cuSeqLensQOptional.
    - If batchSize is less than or equal to 0 and layoutQ is BSND, an error is reported.
  - Seqlen value rules
    - In TND use cases, calculate the actual q/k lengths of each batch through cuSeqLensQOptional and cuSeqLensKOptional.
    - In BSND use cases, obtain the q/k length through maxSeqLenQ and maxSeqLenK.
  - layout constraints
    - layoutQ must be BSND or TND.
    - layoutK supports BSND and TND. It is recommended to keep it consistent with layoutQ.
  - head constraints
    - numHeadsQ, numHeadsK, and headDim must be positive.
    - numHeadsQ must be divisible by numHeadsK.
  - dense constraints
    - cmpRatio: range [1, 128].
    - maskMode currently only supports 0 and 3.

</details>

<details>
<summary><a id="Metadata"></a>Metadata Output Layout</summary>

  The metadata output is an INT32 Tensor with the shape fixed to (64,). The field layout is as follows.

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 180px">
    <col style="width: 150px">
    <col style="width: 820px">
    </colgroup>
    <thead>
        <tr>
        <th>Field</th>
        <th>index</th>
        <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <td>totalSize</td>
        <td>0</td>
        <td>Total number of task rows after B/S1 axes are combined.</td>
        </tr>
        <tr>
        <td>formerCoreSize</td>
        <td>1</td>
        <td>Core-splitting granularity in the upper-aligned use case.</td>
        </tr>
        <tr>
        <td>remainCoreSize</td>
        <td>2</td>
        <td>Core-splitting granularity in the lower-aligned use case.</td>
        </tr>
        <tr>
        <td>remainCoreNum</td>
        <td>3</td>
        <td>Quantity of cube cores processing remainCoreSize.</td>
        </tr>
        <tr>
        <td>usedCoreNum</td>
        <td>4</td>
        <td>Quantity of cube cores actually used by the main kernel.</td>
        </tr>
        <tr>
        <td>maxSeqK</td>
        <td>5</td>
        <td>Maximum sequence length of key.</td>
        </tr>
        <tr>
        <td>reserved</td>
        <td>6-63</td>
        <td>Reserved field, currently set to 0.</td>
        </tr>
    </tbody>
  </table>

</details>

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <utility>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_dense_lightning_indexer_kl_loss_grad_metadata.h"

#define CHECK_LOG_RET(cond, ret_val, fmt, ...)      \
    do {                                            \
        if (!(cond)) {                              \
            printf(fmt "\n", ##__VA_ARGS__);        \
            return (ret_val);                       \
        }                                           \
    } while (0)

constexpr uint32_t DLI_METADATA_MAX_CORE_NUM = 25;
constexpr uint32_t DLI_METADATA_HEADER_SIZE = 8;
constexpr uint32_t DLI_METADATA_SIZE = 64;

struct DliGradKLLossMetaData {
    int32_t coreNum;
    int32_t totalSize;
    int32_t splitFactorSize;
    int32_t reserved[DLI_METADATA_HEADER_SIZE - 3];
    int32_t bS1Index[DLI_METADATA_MAX_CORE_NUM];
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
    int64_t maskMode { 0 };
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

    argsGuard.Dismiss();
    return ACL_SUCCESS;
}

void PrintMetadata(const DliGradKLLossMetaData &metadata)
{
    printf("coreNum          : %d\n", metadata.coreNum);
    printf("totalSize        : %d\n", metadata.totalSize);
    printf("splitFactorSize  : %d\n", metadata.splitFactorSize);
    for (uint32_t i = 0; i < std::min<uint32_t>(metadata.coreNum, DLI_METADATA_MAX_CORE_NUM); ++i) {
        printf("bS1Index[%u]      : %d\n", i, metadata.bS1Index[i]);
    }
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
    ret = aclnnDenseLightningIndexerKLLossGradMetadataGetWorkspaceSize(
        context.cuSeqLensQOptional.data, context.cuSeqLensKOptional.data, context.seqUsedQOptional.data,
        context.seqUsedKOptional.data, context.cmpResidualKOptional.data, context.batchSize, context.maxSeqLenQ,
        context.maxSeqLenK, context.numHeadsQ, context.numHeadsK, context.headDim, context.layoutQ,
        context.layoutK, context.maskMode, context.cmpRatio, context.metadata.data, &workspaceSize, &executor);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret,
        "aclnnDenseLightningIndexerKLLossGradMetadataGetWorkspaceSize failed. ERROR: %d", ret);

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

    ret = aclnnDenseLightningIndexerKLLossGradMetadata(workspaceAddr, workspaceSize, executor, stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret,
        "aclnnDenseLightningIndexerKLLossGradMetadata failed. ERROR: %d", ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtSynchronizeStream failed. ERROR: %d", ret);

    DliGradKLLossMetaData result {};
    ret = aclrtMemcpy(&result, sizeof(result), context.metadata.deviceAddr, sizeof(result), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_LOG_RET(ret == ACL_SUCCESS, ret, "aclrtMemcpy failed. ERROR: %d", ret);
    PrintMetadata(result);

    return 0;
}
```

## Troubleshooting

- For common execution issues or exception errors of the AI CPU operator kernel, see "[Fault Cases > Operator Execution Issues > AI CPU Operator Kernel Execution Error](https://www.hiascend.com/document/detail/en/canncommercial/latest/maintenref/troubleshooting/troubleshooting_0151.html)" in *Troubleshooting* for fault locating methods.
