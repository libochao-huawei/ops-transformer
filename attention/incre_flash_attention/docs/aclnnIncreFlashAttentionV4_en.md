# aclnnIncreFlashAttentionV4

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T10:33:51.412Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Not supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: Compatible with the [aclnnIncreFlashAttentionV3](aclnnIncreFlashAttentionV3_en.md) API, with the **kv left Padding feature** added on this basis.

  For auto-regressive language models, as new tokens are generated, the inference input length keeps increasing. **Incremental inference** is implemented on top of the original full inference. The S axis of the query is fixed to 1. The key and value, after KV Cache, superimpose the previously inferred state information. The actual length of the S axis for each batch may differ, and the input data is fixed-length data after padding.

  Compared with the FlashAttention operator in the full-attention scenario ([PromptFlashAttention](https://gitcode.com/cann/ops-transformer/blob/9.1.0/attention/prompt_flash_attention/README.md)), the incremental inference process is not completely equivalent to normal full-attention inference, but the precision of incremental inference does not exhibit significant degradation.

  **NOTE**

  KV Cache is a common technique for optimizing the inference performance of large models. During sampling, the Transformer model uses the given prompt/context as the initial input for inference (which can be processed in parallel), and then generates additional tokens one by one to further refine the generated sequence (reflecting the autoregressive nature of the model). During the sampling process, the Transformer performs self-attention operations, which requires extracting key-value (KV) vectors for each item in the current sequence (whether prompt/context or generated tokens). These vectors are stored in a matrix, commonly referred to as the KV cache.

- Formula:

  Self-attention leverages the relationships within the input sample itself to construct an attention model. The principle is as follows: assuming an input sample sequence $x$ of length $n$, where each element of $x$ is a $d$-dimensional vector, each $d$-dimensional vector can be regarded as a token embedding. This sequence is transformed through three weight matrices to obtain three matrices of dimension $n*d$.

    The calculation formula of self-attention is generally defined as follows, where $Q$, $K$, and $V$ are important attribute elements of the input sample, obtained through spatial transformation of the input sample, and can be unified into a single feature space.

  $$
     Attention(Q,K,V)=Score(Q,K)V
  $$

    In this operator, the Score function uses the Softmax function, and the self-attention calculation formula is:

  $$
  Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
  $$

    Where the product of $Q$ and $K^T$ represents the attention of input $x$. To prevent this value from becoming too large, it is typically divided by the square root of $d$ for scaling, and each row is normalized by softmax. After multiplying with $V$, an $n*d$ matrix is obtained.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnIncreFlashAttentionV4GetWorkspaceSize" API first to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the "aclnnIncreFlashAttentionV4" API to perform the computation.

```cpp
aclnnStatus aclnnIncreFlashAttentionV4GetWorkspaceSize(
    const aclTensor     *query,
    const aclTensorList *key,
    const aclTensorList *value,
    const aclTensor     *pseShift,
    const aclTensor     *attenMask,
    const aclIntArray   *actualSeqLengths,
    const aclTensor     *dequantScale1,
    const aclTensor     *quantScale1,
    const aclTensor     *dequantScale2,
    const aclTensor     *quantScale2,
    const aclTensor     *quantOffset2,
    const aclTensor     *antiquantScale,
    const aclTensor     *antiquantOffset,
    const aclTensor     *blocktable,
    const aclTensor     *kvPaddingSize,
    int64_t              numHeads,
    double               scaleValue,
    char                *inputLayout,
    int64_t              numKeyValueHeads,
    int64_t              blockSize,
    int64_t              innerPrecise,
    const aclTensor     *attentionOut,
    uint64_t            *workspaceSize,
    aclOpExecutor       **executor)
```

```cpp
aclnnStatus aclnnIncreFlashAttentionV4(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnIncreFlashAttentionV4GetWorkspaceSize

- **Parameters**

  <div style="overflow-x: auto;">
  <table style="undefined;table-layout: fixed; width: 1567px">
  <colgroup>
   <col style="width: 190px">
   <col style="width: 120px">
   <col style="width: 300px">
   <col style="width: 330px">
   <col style="width: 212px">
   <col style="width: 100px">
   <col style="width: 170px">
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
        <td>query</td>
        <td>Input</td>
        <td>Input Q in the formula.</td>
         <td><ul><li>The shapes of query and attentionOut must be exactly the same.</li><li>Only the case where the S axis of query is equal to 1 is supported.</li></ul></td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td><ul><li>(B, N, S, D)</li><li>(B, S, N, D)</li><li>(B, S, H)</li></ul></td>
        <td>×</td>
      </tr>
      <tr>
        <td>key</td>
        <td>Input</td>
        <td>Input K in the formula.</td>
        <td>The shapes of the corresponding tensors in key and value must be exactly the same.</td>
        <td>FLOAT16, BFLOAT16, INT8</td>
        <td>ND</td>
        <td><ul><li>(B, N, S, D)</li><li>(B, S, N, D)</li><li>(B, S, H)</li></ul></td>
        <td>×</td>
      </tr>
      <tr>
        <td>value</td>
        <td>Input</td>
        <td>Input V in the formula.</td>
        <td>The shapes of the corresponding tensors in key and value must be exactly the same.</td>
        <td>FLOAT16, BFLOAT16, INT8</td>
        <td>ND</td>
        <td><ul><li>(B, N, S, D)</li><li>(B, S, N, D)</li><li>(B, S, H)</li></ul></td>
        <td>×</td>
      </tr>
      <tr>
        <td>pseShift</td>
        <td>Input</td>
        <td>Position encoding.</td>
        <td><ul><li>Empty tensors are supported.</li><li>The pseShift data type must be consistent with the query data type.</li></ul></td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td><ul><li>(B, N, 1, S)</li><li>(1, N, 1, S)</li></ul></td>
        <td>×</td>
      </tr>
      <tr>
        <td>attenMask</td>
        <td>Input</td>
        <td>attention mask matrix.</td>
        <td><ul><li>Supports empty tensors.</li><li>When the data type of attenMask is INT8 or UINT8, values in the tensor must be 0 or 1.</li></ul></td>
        <td>BOOL, INT8, UINT8</td>
        <td>ND</td>
        <td><ul><li>(B, S)</li><li>(B, 1, S)</li><li>(B, 1, 1, S)</li></ul></td>
        <td>×</td>
      </tr>
      <tr>
        <td>actualSeqLengths</td>
        <td>Input</td>
        <td>Actual length of the S dimension of key and value.</td>
        <td>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</td>
        <td>INT64</td>
        <td>ND</td>
        <td>(B)</td>
        <td>-</td>
      </tr>
      <tr>
        <td>dequantScale1</td>
        <td>Input</td>
        <td>Dequantization factor after BMM1.</td>
        <td><ul><li>Supports empty tensors.</li><li>Supports per-tensor (scalar).</li></ul></td>
        <td>UINT64, FLOAT32</td>
        <td>ND</td>
        <td>(1)</td>
        <td>×</td>
      </tr>
       <tr>
        <td>quantScale1</td>
        <td>Input</td>
        <td>Quantization factor before BMM2.</td>
        <td><ul><li>Supports empty tensors.</li><li>Supports per-tensor (scalar).</li></ul></td>
        <td>FLOAT32, BFLOAT16</td>
        <td>ND</td>
        <td>(1)</td>
        <td>×</td>
      </tr>
      <tr>
        <td>dequantScale2</td>
        <td>Input</td>
        <td>Quantization factor for dequantization after BMM2.</td>
        <td><ul><li>Supports empty tensors.</li><li>Supports per-tensor (scalar).</li></ul></td>
        <td>UINT64, FLOAT32</td>
        <td>ND</td>
        <td>(1)</td>
        <td>×</td>
      </tr>
      <tr>
        <td>quantScale2</td>
        <td>Input</td>
        <td>Quantization factor for output quantization.</td>
        <td><ul><li>Supports empty tensors.</li><li>Supports per-tensor and per-channel.</li></ul></td>
        <td>FLOAT32, BFLOAT16</td>
        <td>ND</td>
        <td><ul><li>(1)</li><li>(D * N), where N is numKeyValueHeads</li></ul></td>
        <td>×</td>
      </tr>
      <tr>
        <td>quantOffset2</td>
        <td>Input</td>
        <td>Quantization offset for output quantization.</td>
        <td><ul><li>Supports empty tensors.</li><li>Supports per-tensor and per-channel.</li></ul></td>
        <td>FLOAT32, BFLOAT16</td>
        <td>ND</td>
        <td><ul><li>(1)</li><li>(D * N), wherein N is numKeyValueHeads</li></ul></td>
        <td>×</td>
      </tr>
      <tr>
        <td>antiquantScale</td>
        <td>Input</td>
        <td>Quantization factor.</td>
        <td><ul><li>Empty tensors are supported.</li><li>Supports per-tensor and per-channel (list), determined by shape. In the BNSD scenario, the shape is (2, N, 1, D); in the BSH scenario, the shape is (2, H); in the BSND scenario, the shape is (2, N, D).</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td><ul><li>(2, N, 1, D)</li><li>(2, H)</li><li>(2, N, D)</li><li>(2)</li></ul></td>
        <td>×</td>
      </tr>
        <tr>
        <td>antiquantOffset</td>
        <td>Input</td>
        <td>Quantization factor for output quantization.</td>
        <td><ul><li>Empty tensors are supported.</li><li>Supports per-tensor and per-channel (list), determined by shape. In the BNSD scenario, the shape is (2, N, 1, D); in the BSH scenario, the shape is (2, H); in the BSND scenario, the shape is (2, N, D).</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td><ul><li>(2, N, 1, D)</li><li>(2, H)</li><li>(2, N, D)</li><li>(2)</li></ul></td>
        <td>×</td>
      </tr>
      <tr>
        <td>blocktable</td>
        <td>Input</td>
        <td>Block mapping table used for KV storage in page attention.</td>
        <td><ul><li>Supports empty tensors.</li><li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>INT32</td>
        <td>ND</td>
        <td>(B, maxBlockNumPerSeq)</td>
        <td>×</td>
      </tr>
        <tr>
        <td>kvPaddingSize</td>
        <td>Input</td>
        <td>KV left padding scenario.</td>
        <td>Supports empty tensors.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>×</td>
      </tr>
      <tr>
        <td>numHeads</td>
        <td>Input</td>
        <td>Head count of query.</td>
        <td>numHeads must be a multiple of numKeyValueHeads.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>scaleValue</td>
        <td>Input</td>
        <td>Reciprocal of the square root of d in the formula.</td>
        <td>-</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>inputLayout</td>
        <td>Input</td>
        <td>Identifies the data layout format of input query, key, and value.</td>
        <td>Currently supports BSH, BNSD, and BSND. If the user does not specifically specify, BSH is recommended.</td>
        <td>STRING</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>numKeyValueHeads</td>
        <td>Input</td>
        <td>Count of heads in key and value.</td>
        <td><ul><li>Used to support the GQA (Grouped-Query Attention) scenario. 0 indicates that the value is equal to the count of heads in query.</li><li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
        <tr>
        <td>blockSize</td>
        <td>Input</td>
        <td>Maximum token count in each block of the KV storage in page attention.</td>
        <td><ul><li>It is recommended to pass 0 if not specifically specified.</li><li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
         <tr>
        <td>innerPrecise</td>
        <td>Input</td>
        <td>Indicates the high-precision/high-performance option.</td>
        <td>0 indicates high precision, and 1 indicates high performance. If not specifically specified, 1 is recommended.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>attentionOut</td>
        <td>Output</td>
        <td>Output in the formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16, INT8</td>
        <td>ND</td>
        <td><ul><li>(B, N, S, D)</li><li>(B, S, N, D)</li><li>(B, S, H)</li></ul></td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the workspace size that the user needs to allocate on the Device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Returns the op executor containing the operator computation flow.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>
    </div>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 280px">
  <col style="width: 119px">
  <col style="width: 751px">
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
      <td>If a passed-in parameter is a mandatory input, output, or mandatory attribute and is a null pointer, 161001 is returned.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data types or data formats of query, key, value, actualSeqLengths, pseShift, attenMask, and attentionOut are out of the supported range.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>An exception occurs when the API calls the npu runtime API for memory operations.</td>
    </tr>
  </tbody>
  </table>

## aclnnIncreFlashAttentionV4

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
      <td>Workspace memory address requested on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size requested on the Device side, obtained by the first-phase API aclnnIncreFlashAttentionV4GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnIncreFlashAttentionV4 defaults to a deterministic implementation.

<!-- npu="910b" id7 -->
- <term>Atlas A2 products</term>:
  - Supports B-axis less than or equal to 65536, N-axis less than or equal to 256, and D-axis less than or equal to 512.
  - The query data type supports FLOAT16 and BFLOAT16, and the attentionOut, key, and value data types support FLOAT16, INT8, and BFLOAT16.
  - The dequantScale1 and dequantScale2 data types support UINT64 and FLOAT32.
  - The quantScale1, quantScale2, and quantOffset2 data types support FLOAT32.
  - The numKeyValueHeads data type supports INT64.
<!-- end id7 -->
<!-- npu="310p" id8 -->
- <term>Atlas inference products</term>:
  - Supports B-axis less than or equal to 256, N-axis less than or equal to 256, and D-axis less than or equal to 512.
  - Supports key and value S-axis less than or equal to 65536.
  - query, key, value, and attentionOut Only Supports FLOAT16 data type.
  - dequantScale1, quantScale1, dequantScale2, quantScale2, and quantOffset2 Only Supports nullptr.
  - numKeyValueHeads Only Supports the value 0.
<!-- end id8 -->
- In non-contiguous scenarios, the number of tensors in the tensorlist of the key and value parameters is equal to B of query (due to tensorlist limitations, B must be less than or equal to 256 in non-contiguous scenarios). The shape must be fully consistent except for S, and batch can only be 1.
- The data layout format of query, key, and value supports interpretation from multiple dimensions, where B (Batch) indicates the input sample batch size, S (Seq-Length) indicates the input sample sequence length, H (Head-Size) indicates the hidden layer size, N (Head-Num) indicates the number of heads, D (Head-Dim) indicates the minimum unit dimension of the hidden layer, and D = H/N is satisfied.
- The N in the query parameter is equal to the numHeads value, the N of key and value is equal to the numKeyValueHeads value, and numHeads is a multiple of numKeyValueHeads.
- The scenario where query, key, and value inputs are all INT8 is not supported.
- In the scenario where query, key, and value inputs are FLOAT16 and the output is INT8: the input parameter quantScale2 is required, quantOffset2 is optional, and the dequantScale1, quantScale1, and dequantScale2 parameters cannot be passed (i.e., they must be nullptr).
- Constraints on antiquantScale and antiquantOffset parameters:
  - Per-channel mode: The shape of both parameters can be (2, N, 1, D), (2, N, D), or (2, H), where N is numKeyValueHeads. The data type of the parameters is the same as the query data type.
  - Per-tensor mode: The shape of both parameters is (2), and the data type is the same as the query data type.
- Input parameters quantScale2 and quantOffset2 support the per-tensor/per-channel formats and the FLOAT32/BFLOAT16 data types. If quantOffset2 is passed in, its type and shape information must be consistent with quantScale2. When the input is BFLOAT16, both FLOAT32 and BFLOAT16 are supported; otherwise, only FLOAT32 is supported. For the per-channel format, when the output layout is BSH, the product of all dimensions of quantScale2 must be equal to H; for other layouts, the product must be equal to N*D. (When the output layout is BSH, it is recommended to pass the quantScale2 shape [1,1,H] or [H]; when the output is BNSD, it is recommended to pass [1,N,1,D] or [N,D]; when the output is BSND, it is recommended to pass [1,1,N,D] or [N,D]).
- Page attention scenario:
  - The prerequisite for enabling page attention is that blockTable exists and is valid, and key and value are arranged in a contiguous memory region based on the indices in blockTable. The key and value dtype supports FLOAT16/BFLOAT16/INT8. In this scenario, the inputLayout parameter of key and value is invalid.
  - blockSize is a custom parameter. Its value affects the performance of page attention. When page attention is enabled, blockSize must be set to a non-zero value and cannot exceed 512. If key and value are of the FLOAT16/BFLOAT16 type, they must be aligned to 16; if they are of the INT8 type, they must be aligned to 32. 128 is recommended. Generally, page attention improves throughput but causes performance degradation.
  - In the page attention scenario, when the inputLayout of query is BNSD, the KV cache layout supports two formats: (blocknum, blocksize, H) and (blocknum, KV_N, blocksize, D). When the inputLayout of query is BSH or BSND, the KV cache layout supports only one format: (blocknum, blocksize, H). blocknum must not be less than the sum of the number of blocks for each batch calculated based on actualSeqLengthsKv and blockSize. In addition, the shapes of key and value must be consistent.
  - In the page attention scenario, the performance of the KV cache layout (blocknum, KV_N, blocksize, D) is generally better than that of the KV cache layout (blocknum, blocksize, H). It is recommended to prioritize the (blocknum, KV_N, blocksize, D) format.
  - When page attention is enabled, if the input kv cache layout is (blocknum, blocksize, H) and numKvHeads * headDim exceeds 64k, an error is reported due to hardware instruction constraints. To resolve this issue, enable GQA (reduce numKvHeads) or change the kv cache layout to (blocknum, numKvHeads, blocksize, D).
  - In the page attention scenario, the input actualSeqLengths must be passed.
  - In the page attention scenario, blockTable must be two-dimensional, with the first dimension length equal to B and the second dimension length not less than maxBlockNumPerSeq (maxBlockNumPerSeq is the number of blocks corresponding to the maximum actualSeqLengthsKv in each batch).
  - When page attention is enabled, the input S must be greater than or equal to maxBlockNumPerSeq * blockSize in the following scenarios.
  - Attention mask is enabled, for example, the mask shape is (B, 1, 1, S).
  - pseShift is enabled, for example, the pseShift shape is (B, N, 1, S).
  - kvPaddingSize is enabled, for example, the kvPaddingSize shape is (1). S must be greater than or equal to the sum of kvPaddingSize and the maximum actualSeqLengthsKv of each Batch.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "securec.h"
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_incre_flash_attention_v4.h"

using namespace std;

namespace {

#define CHECK_RET(cond) ((cond) ? true :(false))

#define LOG_PRINT(message, ...)     \
  do {                              \
    (void)printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  auto ret = aclInit(nullptr);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
    return ret;
  }
  ret = aclrtSetDevice(deviceId);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
    return ret;
  }
  ret = aclrtCreateStream(stream);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
    return ret;
  }
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
    return ret;
  }

  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
    return ret;
  }

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

struct TensorResources {
    void* queryDeviceAddr = nullptr;
    void* keyDeviceAddr = nullptr;
    void* valueDeviceAddr = nullptr;
    void* attenDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* queryTensor = nullptr;
    aclTensor* keyTensor = nullptr;
    aclTensor* valueTensor = nullptr;
    aclTensor* attenTensor = nullptr;
    aclTensor* outTensor = nullptr;
    aclTensorList* tensorKeyList = nullptr;
    aclTensorList* tensorValueList = nullptr;
    aclIntArray* actualSeqLengths = nullptr;
};

int InitializeTensors(TensorResources& resources) {
    std::vector<int64_t> queryShape = {1, 2, 1, 16};
    std::vector<int64_t> keyShape = {1, 2, 2, 16};
    std::vector<int64_t> valueShape = {1, 2, 2, 16};
    std::vector<int64_t> attenShape = {1, 1, 1, 2};
    std::vector<int64_t> outShape = {1, 2, 1, 16};

    int64_t queryShapeSize = GetShapeSize(queryShape);
    int64_t keyShapeSize = GetShapeSize(keyShape);
    int64_t valueShapeSize = GetShapeSize(valueShape);
    int64_t attenShapeSize = GetShapeSize(attenShape);
    int64_t outShapeSize = GetShapeSize(outShape);

    std::vector<op::fp16_t> queryHostData(queryShapeSize, 1);
    std::vector<op::fp16_t> keyHostData(keyShapeSize, 1);
    std::vector<op::fp16_t> valueHostData(valueShapeSize, 1);
    std::vector<int8_t> attenHostData(attenShapeSize, 1);
    std::vector<op::fp16_t> outHostData(outShapeSize, 1);

    int ret = CreateAclTensor(queryHostData, queryShape, &resources.queryDeviceAddr,
                             aclDataType::ACL_FLOAT16, &resources.queryTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(keyHostData, keyShape, &resources.keyDeviceAddr,
                         aclDataType::ACL_FLOAT16, &resources.keyTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(valueHostData, valueShape, &resources.valueDeviceAddr,
                         aclDataType::ACL_FLOAT16, &resources.valueTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(attenHostData, attenShape, &resources.attenDeviceAddr,
                         aclDataType::ACL_INT8, &resources.attenTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(outHostData, outShape, &resources.outDeviceAddr,
                         aclDataType::ACL_FLOAT16, &resources.outTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    int kvTensorNum = 1;
    aclTensor* tensorsOfKey[] = {resources.keyTensor};
    resources.tensorKeyList = aclCreateTensorList(tensorsOfKey, kvTensorNum);

    aclTensor* tensorsOfValue[] = {resources.valueTensor};
    resources.tensorValueList = aclCreateTensorList(tensorsOfValue, kvTensorNum);

    std::vector<int64_t> actualSeqlenVector = {2};
    resources.actualSeqLengths = aclCreateIntArray(actualSeqlenVector.data(),
                                                  actualSeqlenVector.size());

    return ACL_SUCCESS;
}

int ExecuteIncreFlashAttention(TensorResources& resources, aclrtStream stream,
                              void** workspaceAddr, uint64_t* workspaceSize) {
    int64_t numHeads = 2;
    int64_t numKeyValueHeads = numHeads;
    int64_t blockSize = 1;
    int64_t innerPrecise = 1;
    double scaleValue = 1 / sqrt(2);
    constexpr const char LAYER_OUT_STR[] = "BNSD";
    constexpr size_t LAYER_OUT_LEN = sizeof(LAYER_OUT_STR);
    char layerOut[LAYER_OUT_LEN];
    memcpy(layerOut, LAYER_OUT_STR, LAYER_OUT_LEN);

    aclOpExecutor* executor;
    int ret = aclnnIncreFlashAttentionV4GetWorkspaceSize(
        resources.queryTensor, resources.tensorKeyList, resources.tensorValueList,
        nullptr, resources.attenTensor, resources.actualSeqLengths, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        numHeads, scaleValue, layerOut, numKeyValueHeads, blockSize, innerPrecise,
        resources.outTensor, workspaceSize, &executor);

    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnIncreFlashAttentionV4GetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    if (*workspaceSize > 0ULL) {
        ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    ret = aclnnIncreFlashAttentionV4(*workspaceAddr, *workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnIncreFlashAttentionV4 failed. ERROR: %d\n", ret);
        return ret;
    }

    return ACL_SUCCESS;
}

int ProcessResults(TensorResources& resources, const std::vector<int64_t>& outShape) {
    auto size = GetShapeSize(outShape);
    std::vector<op::fp16_t> resultData(size, 0);

    int ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         resources.outDeviceAddr, size * sizeof(resultData[0]),
                         ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        return ret;
    }

    for (int64_t i = 0; i < size; i++) {
        std::cout << "index: " << i << ": " << static_cast<float>(resultData[i]) << std::endl;
    }

    return ACL_SUCCESS;
}

void CleanupResources(TensorResources& resources, void* workspaceAddr,
                     aclrtStream stream, int32_t deviceId) {
    if (resources.queryTensor) {
      aclDestroyTensor(resources.queryTensor);
    }
    if (resources.keyTensor) {
      aclDestroyTensor(resources.keyTensor);
    }
    if (resources.valueTensor) {
      aclDestroyTensor(resources.valueTensor);
    }
    if (resources.attenTensor) {
      aclDestroyTensor(resources.attenTensor);
    }
    if (resources.outTensor) {
      aclDestroyTensor(resources.outTensor);
    }
    if (resources.actualSeqLengths) {
      aclDestroyIntArray(resources.actualSeqLengths);
    }

    if (resources.queryDeviceAddr) {
      aclrtFree(resources.queryDeviceAddr);
    }
    if (resources.keyDeviceAddr) {
      aclrtFree(resources.keyDeviceAddr);
    }
    if (resources.valueDeviceAddr) {
      aclrtFree(resources.valueDeviceAddr);
    }
    if (resources.attenDeviceAddr) {
      aclrtFree(resources.attenDeviceAddr);
    }
    if (resources.outDeviceAddr) {
      aclrtFree(resources.outDeviceAddr);
    }

    if (workspaceAddr) {
      aclrtFree(workspaceAddr);
    }
    if (stream) {
      aclrtDestroyStream(stream);
    }

    aclrtResetDevice(deviceId);
    aclFinalize();
}

} // namespace

int main() {
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    TensorResources resources = {};
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    std::vector<int64_t> outShape = {1, 2, 1, 16};
    int ret = ACL_SUCCESS;

    // 1. Initialize device and stream
    ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    // 2. Initialize tensors
    ret = InitializeTensors(resources);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 3. Execute the operation
    ret = ExecuteIncreFlashAttention(resources, stream, &workspaceAddr, &workspaceSize);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 4. Synchronize stream
    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 5. Process results
    ret = ProcessResults(resources, outShape);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 6. Cleanup resources
    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return 0;
}
```
