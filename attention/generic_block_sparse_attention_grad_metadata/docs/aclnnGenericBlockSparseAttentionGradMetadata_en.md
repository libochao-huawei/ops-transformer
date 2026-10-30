# aclnnGenericBlockSparseAttentionGradMetadata

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T10:13:11.037Z -->

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

- Description: aclnnGenericBlockSparseAttentionGradMetadata performs partitioning and load balancing for sparse attention based on information such as sparseBlockIdx, sparseBlockCount, and seqlen. It is a preceding AICPU operator of aclnnGenericBlockSparseAttentionGrad. It expands the `(b, n2, j, g)` task list in the order of B → N2 → J → G and performs greedy load balancing across AIC cores at the `[baseM, baseN] = [128, 128]` basic block granularity, outputting metadata for the main Grad operator to consume.
- This operator is not recommended for standalone use. It is recommended to use it with aclnnGenericBlockSparseAttentionGrad to form a complete workflow.

$$
\text{metaSize} = 80 + B \times N1 \times J \times 4
$$

where `J = ceilDiv(maxKvSeqlen, blockShapeY)`.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnGenericBlockSparseAttentionGradMetadataGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call the "aclnnGenericBlockSparseAttentionGradMetadata" API to perform the computation.

```c++
aclnnStatus aclnnGenericBlockSparseAttentionGradMetadataGetWorkspaceSize(
    const aclTensor *sparseBlockIdx,
    const aclTensor *sparseBlockCount,
    const aclTensor *cuSeqLengthsQOptional,
    const aclTensor *cuSeqLengthsKvOptional,
    const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKvOptional,
    int64_t maxQSeqlen,
    int64_t maxKvSeqlen,
    int64_t numQHeads,
    int64_t numKvHeads,
    int64_t headDim,
    const aclIntArray *blockShape,
    int64_t isPackedGQA,
    char *layoutQ,
    char *layoutKv,
    int64_t maskType,
    int64_t softmaxPrecision,
    int64_t winLeft,
    int64_t winRight,
    aclTensor *metadata,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
```

```c++
aclnnStatus aclnnGenericBlockSparseAttentionGradMetadata(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
```

## aclnnGenericBlockSparseAttentionGradMetadataGetWorkspaceSize

- **Parameters**

  <table style="undefined; table-layout: fixed; width: 1567px">
    <colgroup>
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
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>sparseBlockIdx</td>
        <td>Input</td>
        <td>Array of sparse block indices, specifying the Q block/token indices selected for each KV block.</td>
        <td>The Q sparse pattern is the same for each KVHead in the same group (isPackedGQA=1). Empty tensors are not supported.</td>
        <td>INT32</td>
        <td>ND</td>
        <td>TND: [B, N2, ceilDiv(maxS2, blockShapeY), maxS1]; BNSD/BSND: [B, N2, ceilDiv(S2, blockShapeY), S1]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>sparseBlockCount</td>
        <td>Input</td>
        <td>Specifies the number of Q blocks actually selected for each KV block.</td>
        <td>Empty tensors are not supported.</td>
        <td>INT32</td>
        <td>ND</td>
        <td>[B, N2, ceilDiv(maxS2, blockShapeY)] or [B, N2, ceilDiv(S2, blockShapeY)]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>cuSeqLengthsQOptional</td>
        <td>Optional input</td>
        <td>Prefix sum of query sequence lengths for each batch.</td>
        <td>Required when layoutQ is "TND"; set to nullptr when layoutQ is "BNSD" or "BSND".</td>
        <td>INT64</td>
        <td>ND</td>
        <td>[B+1]</td>
        <td>-</td>
      </tr>
      <tr>
        <td>cuSeqLengthsKvOptional</td>
        <td>Optional input</td>
        <td>Prefix sum of key/value sequence lengths for each batch.</td>
        <td>Must be configured when layoutKv is "TND"; set to nullptr when layoutKv is "BNSD" or "BSND".</td>
        <td>INT64</td>
        <td>ND</td>
        <td>[B+1]</td>
        <td>-</td>
      </tr>
      <tr>
        <td>sequsedQOptional</td>
        <td>Optional input</td>
        <td>Actual sequence length of query in each batch. Takes effect only when layoutQ is "TND"; nullptr must be passed when layoutQ is "BNSD" or "BSND", in which case the actual length is the same as maxQSeqlen or the S dimension of query.</td>
        <td>Length B.</td>
        <td>INT32</td>
        <td>ND</td>
        <td>[B]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>sequsedKvOptional</td>
        <td>Optional Input</td>
        <td>Actual sequence length of kv in each batch. Takes effect only when layoutKv is "TND"; when it is "BNSD" or "BSND", nullptr must be passed, and the actual length is obtained from maxKvSeqlen (which must be the same as the S dimension of key/value).</td>
        <td>The length is B.</td>
        <td>INT32</td>
        <td>ND</td>
        <td>[B]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>maxQSeqlen</td>
        <td>Input</td>
        <td>Maximum value of qSeqlen across all batches.</td>
        <td>Must be ≥ 0. In BNSD/BSND use cases, must be consistent with the S dimension of query.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>maxKvSeqlen</td>
        <td>Input</td>
        <td>Maximum kvSeqlen across all batches.</td>
        <td>Must be ≥ 0. Used to calculate J = ceilDiv(maxKvSeqlen, blockShapeY).</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>numQHeads</td>
        <td>Input</td>
        <td>Number of query heads (N1).</td>
        <td>Must be in the range of [1, 128] and divisible by numKvHeads.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>numKvHeads</td>
        <td>Input</td>
        <td>Number of key/value heads (N2).</td>
        <td>Must be in the range [1, 128] and consistent with the N2 dimension of sparseBlockIdx.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>headDim</td>
        <td>Input</td>
        <td>Embed dimension of query/key/value.</td>
        <td>Currently fixed to 128.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>blockShape</td>
        <td>Input</td>
        <td>Sparse block shape array.</td>
        <td>Contains two elements [blockShapeX, blockShapeY]. Currently only 1 is supported for blockShapeX; blockShapeY must be ≥128 and a multiple of 64.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>isPackedGQA</td>
        <td>Input</td>
        <td>Whether qHeads in the same group share the same sparse pattern.</td>
        <td>Currently only 1 is supported.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>layoutQ</td>
        <td>Input</td>
        <td>Data layout format of the input query.</td>
        <td>Currently, "TND", "BNSD", and "BSND" are supported.</td>
        <td>STRING</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>layoutKv</td>
        <td>Input</td>
        <td>Data layout format of input key and value.</td>
        <td>Currently "TND", "BNSD", and "BSND" are supported. This parameter must be the same as layoutQ.</td>
        <td>STRING</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>maskType</td>
        <td>Input</td>
        <td>Mask type in the attention computation.</td>
        <td>Currently, only 1 (CAUSAL) is supported.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>softmaxPrecision</td>
        <td>Input</td>
        <td>Precision level used for the Softmax computation.</td>
        <td>Currently, only 0 is supported.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>winLeft</td>
        <td>Input</td>
        <td>Number of tokens included forward in the sliding window.</td>
        <td>Must be -1 when disabled.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>winRight</td>
        <td>Input</td>
        <td>Number of tokens included backward by the sliding window.</td>
        <td>Must be -1 when not enabled.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>metadata</td>
        <td>Output</td>
        <td>Partitioning information of sparse attention.</td>
        <td>Empty tensors are not supported. metaSize ≥ 80 + B×N1×J×4.</td>
        <td>INT32</td>
        <td>ND</td>
        <td>[metaSize]</td>
        <td>×</td>
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

  <div style="overflow-x: auto;">
    <table style="table-layout: fixed; width: 1100px">
      <colgroup>
        <col style="width: 250px">
        <col style="width: 130px">
        <col style="width: 720px">
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
          <td>ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
          <td>561101</td>
          <td>Failed to create aclOpExecutor.</td>
        </tr>
        <tr>
          <td>ACLNN_ERR_INNER_NULLPTR</td>
          <td>561103</td>
          <td>workspaceSize or executor is a null pointer.</td>
        </tr>
        <tr>
          <td class="merged-cell" rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
          <td class="merged-cell" rowspan="3">161002</td>
          <td>sparseBlockIdx/sparseBlockCount/metadata is null or shape/dtype is invalid.</td>
        </tr>
        <tr>
          <td>cuSeqLengths is not provided when layout is TND; headDim, blockShape, isPackedGQA, and window parameters are out of the supported range.</td>
        </tr>
        <tr>
          <td>The J dimension of sparseBlockIdx is inconsistent with ceilDiv(maxKvSeqlen, blockShapeY); the metadata length is insufficient; B×N1×J exceeds the task count limit (1M).</td>
        </tr>
      </tbody>
    </table>
  </div>

## aclnnGenericBlockSparseAttentionGradMetadata

- **Parameters**

  <table><thead>
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
      <td>Size of the workspace memory allocated on the device, obtained by calling the first-phase API aclnnGenericBlockSparseAttentionGradMetadataGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, containing the operator computation flow.</td>
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

- Deterministic computation:
  - The output of this operator is deterministic.
- Must be used together with [aclnnGenericBlockSparseAttentionGrad](../../generic_block_sparse_attention_grad/docs/aclnnGenericBlockSparseAttentionGrad_en.md). This operator must be successfully executed before the main operator is called.
- HeadDim is fixed to 128; numQHeads/numKvHeads must be in the range [1, 128], and numQHeads % numKvHeads == 0.
- blockShape currently supports only [1, 128]; isPackedGQA currently supports only 1; maskType currently supports only 1.
- layoutQ and layoutKv must be the same, with the value "TND"/"BNSD"/"BSND"; cuSeqLengthsQOptional/cuSeqLengthsKvOptional are mandatory when the layout is TND.
- sequsedQOptional/sequsedKvOptional take effect only for TND; nullptr must be passed for BNSD/BSND, and the actual sequence length is taken from maxQSeqlen/maxKvSeqlen (which must be the same as the S dimension of Q/K).
- winLeft and winRight must be -1 when not enabled.
- The last dimension maxS1 of sparseBlockIdx must be ≥ maxQSeqlen; J = ceilDiv(maxKvSeqlen, blockShapeY) must be the same as the 3rd dimension of sparseBlockIdx.
- The length of metadata must satisfy shape[0] ≥ 80 + B × numQHeads × J × 4; the upper bound of the number of tasks is B × numQHeads × J ≤ 1048576.

## Example

The following sample code is for reference only. For details about how to compile and run the code, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_generic_block_sparse_attention_grad_metadata.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t n = 1;
    for (auto v : shape) {
        n *= v;
    }
    return n;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    const int64_t B = 1, N1 = 1, N2 = 1, S1 = 128, S2 = 128, D = 128;
    const int64_t blockY = 128;
    const int64_t J = (S2 + blockY - 1) / blockY;
    const int64_t metaSize = 80 + B * N1 * J * 4;

    std::vector<int64_t> idxShape = {B, N2, J, S1};
    std::vector<int64_t> cntShape = {B, N2, J};
    std::vector<int64_t> metaShape = {metaSize};

    std::vector<int32_t> idxHost(static_cast<size_t>(GetShapeSize(idxShape)), -1);
    std::vector<int32_t> cntHost(static_cast<size_t>(GetShapeSize(cntShape)), 0);
    std::vector<int32_t> metaHost(static_cast<size_t>(metaSize), 0);
    for (int64_t q = 0; q < S1; ++q) {
        idxHost[static_cast<size_t>(q)] = static_cast<int32_t>(q);
    }
    cntHost[0] = static_cast<int32_t>(S1);

    void *idxAddr = nullptr, *cntAddr = nullptr, *metaAddr = nullptr;
    aclTensor *idx = nullptr, *cnt = nullptr, *metadata = nullptr;
    ret = CreateAclTensor(idxHost, idxShape, &idxAddr, aclDataType::ACL_INT32, &idx);
    CHECK_RET(ret == ACL_SUCCESS, Finalize(deviceId, stream); return ret);
    ret = CreateAclTensor(cntHost, cntShape, &cntAddr, aclDataType::ACL_INT32, &cnt);
    CHECK_RET(ret == ACL_SUCCESS, Finalize(deviceId, stream); return ret);
    ret = CreateAclTensor(metaHost, metaShape, &metaAddr, aclDataType::ACL_INT32, &metadata);
    CHECK_RET(ret == ACL_SUCCESS, Finalize(deviceId, stream); return ret);

    const int64_t blockShapeData[] = {1, blockY};
    aclIntArray *blockShape = aclCreateIntArray(blockShapeData, 2);
    char layout[] = "BNSD";

    uint64_t wsSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnGenericBlockSparseAttentionGradMetadataGetWorkspaceSize(
        idx, cnt, nullptr, nullptr, nullptr, nullptr, S1, S2, N1, N2, D, blockShape, 1, layout, layout, 1, 0, -1, -1,
        metadata, &wsSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, Finalize(deviceId, stream); return ret);

    void *ws = nullptr;
    if (wsSize > 0) {
        ret = aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, Finalize(deviceId, stream); return ret);
    }
    ret = aclnnGenericBlockSparseAttentionGradMetadata(ws, wsSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, Finalize(deviceId, stream); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, Finalize(deviceId, stream); return ret);

    LOG_PRINT("GenericBlockSparseAttentionGradMetadata finished successfully.\n");

    aclDestroyIntArray(blockShape);
    aclDestroyTensor(idx);
    aclDestroyTensor(cnt);
    aclDestroyTensor(metadata);
    aclrtFree(idxAddr);
    aclrtFree(cntAddr);
    aclrtFree(metaAddr);
    if (wsSize > 0) {
        aclrtFree(ws);
    }
    Finalize(deviceId, stream);
    return 0;
}
```
