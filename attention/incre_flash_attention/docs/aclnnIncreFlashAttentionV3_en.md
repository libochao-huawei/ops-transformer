# aclnnIncreFlashAttentionV3

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T10:26:24.203Z -->

**Notice: This API will be deprecated in a later version. Use the latest API [aclnnIncreFlashAttentionV4](./aclnnIncreFlashAttentionV4_en.md) instead.**

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
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: Compatible with the [aclnnIncreFlashAttentionV2](aclnnIncreFlashAttentionV2_en.md) API. **Adds positional encoding, page attention, and KV cache dequantization features** on top of it.

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

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnIncreFlashAttentionV3GetWorkspaceSize" API first to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the "aclnnIncreFlashAttentionV3" API to execute the computation.

```cpp
aclnnStatus aclnnIncreFlashAttentionV3GetWorkspaceSize(
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
aclnnStatus aclnnIncreFlashAttentionV3(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnIncreFlashAttentionV3GetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1567px"><colgroup>
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
        <td>The shapes of query and attentionOut must be exactly the same.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td><ul><li>(B, N, S, D)</li><li>(B, S, N, D)</li><li>(B, S, H)</li></ul></td>
        <td>×</td>
      </tr>
      <tr>
        <td>key</td>
        <td>Input</td>
        <td>Input K in the formula.</td>
        <td>The shapes of the corresponding tensors in key and value must be identical.</td>
        <td>FLOAT16, BFLOAT16, INT8</td>
        <td>ND</td>
        <td><ul><li>(B, N, S, D)</li><li>(B, S, N, D)</li><li>(B, S, H)</li></ul></td>
        <td>×</td>
      </tr>
      <tr>
        <td>value</td>
        <td>Input</td>
        <td>Input V in the formula.</td>
        <td>The shapes of the corresponding tensors in key and value must be identical.</td>
        <td>FLOAT16, BFLOAT16, INT8</td>
        <td>ND</td>
        <td><ul><li>(B, N, S, D)</li><li>(B, S, N, D)</li><li>(B, S, H)</li></ul></td>
        <td>×</td>
      </tr>
      <tr>
        <td>pseShift</td>
        <td>Input</td>
        <td>Position encoding.</td>
        <td><ul><li>Supports empty tensors.</li><li>The data type of pseShift must be the same as that of query.</li></ul></td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td><ul><li>(B, N, 1, S)</li><li>(1, N, 1, S)</li></ul></td>
        <td>×</td>
      </tr>
      <tr>
        <td>attenMask</td>
        <td>Input</td>
        <td>Attention mask matrix.</td>
        <td><ul><li>Empty tensors are supported.</li><li>When the data type of attenMask is INT8 or UINT8, the tensor values must be 0 or 1.</li></ul></td>
        <td>BOOL, INT8, UINT8</td>
        <td>ND</td>
        <td><ul><li>(B, S)</li><li>(B, 1, S)</li><li>(B, 1, 1, S)</li></ul></td>
        <td>×</td>
      </tr>
      <tr>
        <td>actualSeqLengths</td>
        <td>Input</td>
        <td>Actual length of key and value on the S axis.</td>
        <td>See <a href="#constraints">Constraints</a> for comprehensive constraints.</td>
        <td>INT64</td>
        <td>ND</td>
        <td>(B)</td>
        <td>-</td>
      </tr>
      <tr>
        <td>dequantScale1</td>
        <td>Input</td>
        <td>Quantization factor for dequantization after BMM1.</td>
        <td><ul><li>Empty tensors are supported.</li><li>Per-tensor (scalar) is supported.</li></ul></td>
        <td>UINT64, FLOAT32</td>
        <td>ND</td>
        <td>(1)</td>
        <td>×</td>
      </tr>
      <tr>
        <td>quantScale1</td>
        <td>Input</td>
        <td>Quantization factor for quantization before BMM2.</td>
        <td><ul><li>Empty tensors are supported.</li><li>Per-tensor (scalar) is supported.</li></ul></td>
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
        <td><ul><li>Empty tensors are supported.</li><li>per-tensor and per-channel are supported.</li></ul></td>
        <td>FLOAT32, BFLOAT16</td>
        <td>ND</td>
        <td><ul><li>(1)</li><li>(D * N), where N is numKeyValueHeads</li></ul></td>
        <td>×</td>
      </tr>
      <tr>
        <td>quantOffset2</td>
        <td>Input</td>
        <td>Quantization offset for output quantization.</td>
        <td><ul><li>Supports empty tensor.</li><li>Supports per-tensor, per-channel.</li></ul></td>
        <td>FLOAT32, BFLOAT16</td>
        <td>ND</td>
        <td><ul><li>(1)</li><li>(D * N), where N is numKeyValueHeads</li></ul></td>
        <td>×</td>
      </tr>
      <tr>
        <td>antiquantScale</td>
        <td>Input</td>
        <td>Pseudo-quantization factor.</td>
        <td><ul><li>Supports empty tensors.</li><li>Supports per-channel (list), determined by shape: (2, N, 1, D) in BNSD scenario, (2, H) in BSH scenario, and (2, N, D) in BSND scenario.</li></ul></td>
        <td>FLOAT16, BFLOAT16</td>
        <td>-</td>
        <td><ul><li>(2, N, 1, D)</li><li>(2, H)</li><li>(2, N, D)</li><li>(2)</li></ul></td>
        <td>×</td>
      </tr>
        <tr>
        <td>antiquantOffset</td>
        <td>Input</td>
        <td>Pseudo-quantization factor.</td>
        <td><ul><li>Supports empty tensors.</li><li>Supports per-channel (list), determined by shape: (2, N, 1, D) in BNSD scenario, (2, H) in BSH scenario, and (2, N, D) in BSND scenario.</li></ul></td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td><ul><li>(2, N, 1, D)</li><li>(2, H)</li><li>(2, N, D)</li><li>(2)</li></ul></td>
        <td>×</td>
      </tr>
      <tr>
        <td>blocktable</td>
        <td>Input</td>
        <td>Block mapping table used for KV storage in the page attention use case.</td>
        <td><ul><li>Supports empty tensors.</li><li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>INT32</td>
        <td>ND</td>
        <td>(B, maxBlockNumPerSeq)</td>
        <td>×</td>
      </tr>
      <tr>
        <td>numHeads</td>
        <td>Input</td>
        <td>Number of heads of query.</td>
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
        <td>Currently supports BSH, BNSD, and BSND. If the user does not specify, "BSH" is recommended.</td>
        <td>STRING</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>numKeyValueHeads</td>
        <td>Input</td>
        <td>Number of heads in key and value.</td>
        <td><ul><li>Used to support the GQA (Grouped-Query Attention) use case. If the value is 0, the number of heads is equal to that of query.</li><li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
        <tr>
        <td>blockSize</td>
        <td>Input</td>
        <td>Maximum number of tokens in each KV storage block in page attention.</td>
        <td><ul><li>If not specifically specified, it is recommended to pass 0.</li><li>For Comprehensive Constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
        <tr>
        <td>innerPrecise</td>
        <td>Input</td>
        <td>Represents the high-precision/high-performance selection.</td>
        <td>0 indicates high precision and 1 indicates high performance. Passing 1 is recommended if not specified.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>attentionOut</td>
        <td>Output</td>
        <td>Output of the formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16, INT8</td>
        <td>ND</td>
        <td><ul><li>(B, N, S, D)</li><li>(B, S, N, D)</li><li>(B, S, H)</li></ul></td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the workspace size that users need to allocate on the Device.</td>
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
    </tbody></table>

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
      <td>If a required input, output, or required attribute is a null pointer, 161001 is returned.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data types or data formats of query, key, value, pseShift, attenMask, and attentionOut are out of the supported range.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>An exception occurs when the API calls the npu runtime API for memory allocation.</td>
    </tr>
  </tbody>
  </table>

## aclnnIncreFlashAttentionV3

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
      <td>Workspace memory address requested on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size requested on the Device. Obtained by the first-phase API aclnnIncreFlashAttentionV3GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor containing the operator computation flow.</td>
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
  - aclnnIncreFlashAttentionV3 defaults to a deterministic implementation.

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
- The N in the query parameter is equal to the numHeads value, the N of key and value is equal to the numKeyValueHeads value, and numHeads is a multiple of numKeyValueHeads.
- Only the S axis of the query equal to 1 is supported.
- When the data type of attenMask is INT8 or UINT8, the values in its tensor must be 0 or 1.
- The scenario where query, key, and value inputs are all INT8 is not supported.
- In the scenario where query, key, and value inputs are FLOAT16 and the output is INT8: the input parameter quantScale2 is required, quantOffset2 is optional, and the dequantScale1, quantScale1, and dequantScale2 parameters cannot be passed (i.e., they must be nullptr).

- Constraints on antiquantScale and antiquantOffset parameters:
  - Per-channel mode: The shape of both parameters can be (2, N, 1, D), (2, N, D), or (2, H), where N is numKeyValueHeads. The data type of the parameters is the same as the query data type.
  - Per-tensor mode: The shape of both parameters is (2), and the data type is the same as the query data type.

- Input parameters quantScale2 and quantOffset2 support the per-tensor/per-channel formats and the FLOAT32/BFLOAT16 data types. If quantOffset2 is passed in, ensure that its type and shape are consistent with quantScale2. When the input is BFLOAT16, both FLOAT32 and BFLOAT16 are supported. Otherwise, only FLOAT32 is supported. For the per-channel format, when the output layout is BSH, the product of all dimensions of quantScale2 must be equal to H. For other layouts, the product must be equal to N*D. (When the output layout is BSH, the shape of quantScale2 is recommended to be [1,1,H] or [H]; when the output is BNSD, [1,N,1,D] or [N,D] is recommended; when the output is BSND, [1,1,N,D] or [N,D] is recommended).

- Page attention scenario:
  - Enabling page attention requires that blockTable exists and is valid, and that key and value are arranged in a contiguous memory block according to the indexes in blockTable. The key and value dtypes can be FLOAT16/BFLOAT16/INT8. In this use case, the inputLayout parameter of key and value is invalid.
  - blockSize is a user-defined parameter, and its value affects the performance of page attention. In the page attention use case, blockSize must be a non-zero value and must not exceed 512. When key and value are of the type FLOAT16/BFLOAT16, the value must be aligned to 16. When key and value are of the type INT8, the value must be aligned to 32. 128 is recommended. Generally, page attention improves throughput but degrades performance.
  - In the page attention scenario, when the inputLayout of query is BNSD, the KV cache layout supports two formats: (blocknum, blocksize, H) and (blocknum, KV_N, blocksize, D). When the inputLayout of query is BSH or BSND, the KV cache layout supports only one format: (blocknum, blocksize, H). blocknum must not be less than the sum of the number of blocks for each batch calculated based on actualSeqLengthsKv and blockSize. In addition, the shapes of key and value must be consistent.
  - In the page attention scenario, the performance of the KV cache layout (blocknum, KV_N, blocksize, D) is generally better than that of the KV cache layout (blocknum, blocksize, H). It is recommended to prioritize the (blocknum, KV_N, blocksize, D) format.
  - In the page attention use case, if the input kv cache layout is (blocknum, blocksize, H) and numKeyValueHeads * headDim exceeds 64k, an error is reported due to hardware instruction constraints. Enable GQA (reduce numKeyValueHeads) or change the kv cache layout to (blocknum, numKeyValueHeads, blocksize, D).
  - In the page attention scenario, the input actualSeqLengths must be passed.
  - In the page attention scenario, blockTable must be two-dimensional, with the first dimension length equal to B and the second dimension length not less than maxBlockNumPerSeq (maxBlockNumPerSeq is the number of blocks corresponding to the maximum actualSeqLengthsKv in each batch).
  - In the page attention use case, S in the following use cases must be greater than or equal to maxBlockNumPerSeq * blockSize.
  - Attention mask is enabled. For example, the mask shape is (B, 1, 1, S).
  - pseShift is enabled. For example, the pseShift shape is (B, N, 1, S).

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <iostream>
#include <vector>
#include <math.h>
#include <cstring>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_incre_flash_attention_v3.h"

using namespace std;

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

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
      shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // Boilerplate: resource initialization.
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                  aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate device memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host-side data to device-side memory.
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

int main() {
  // 1. (Boilerplate) Initialize device/stream. For details, see the acl API guide.
  // Fill in deviceId according to the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs. Perform custom construction based on the API.
  int32_t batchSize = 1;
  int32_t numHeads = 2;
  int32_t headDims = 16;
  int32_t keyNumHeads = 2;
  int32_t sequenceLengthKV = 16;
  std::vector<int64_t> queryShape = {batchSize, numHeads, 1, headDims}; // BNSD
  std::vector<int64_t> keyShape = {batchSize, keyNumHeads, sequenceLengthKV, headDims}; // BNSD
  std::vector<int64_t> valueShape = {batchSize, keyNumHeads, sequenceLengthKV, headDims}; // BNSD
  std::vector<int64_t> attenShape = {batchSize, 1, 1, sequenceLengthKV}; // B11S
  std::vector<int64_t> outShape = {batchSize, numHeads, 1, headDims}; // BNSD
  void *queryDeviceAddr = nullptr;
  void *keyDeviceAddr = nullptr;
  void *valueDeviceAddr = nullptr;
  void *attenDeviceAddr = nullptr;
  void *outDeviceAddr = nullptr;
  aclTensor *queryTensor = nullptr;
  aclTensor *keyTensor = nullptr;
  aclTensor *valueTensor = nullptr;
  aclTensor *attenTensor = nullptr;
  aclTensor *outTensor = nullptr;
  std::vector<float> queryHostData(batchSize * numHeads * headDims, 1.0f);
  std::vector<float> keyHostData(batchSize * keyNumHeads * sequenceLengthKV * headDims, 1.0f);
  std::vector<float> valueHostData(batchSize * keyNumHeads * sequenceLengthKV * headDims, 1.0f);
  std::vector<int8_t> attenHostData(batchSize * sequenceLengthKV, 0);
  std::vector<float> outHostData(batchSize * numHeads * headDims, 1.0f);

  // Create the query aclTensor.
  ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the key aclTensor.
  ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &keyTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  int kvTensorNum = 1;
  aclTensor *tensorsOfKey[kvTensorNum];
  tensorsOfKey[0] = keyTensor;
  auto tensorKeyList = aclCreateTensorList(tensorsOfKey, kvTensorNum);
  // Create the value aclTensor.
  ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &valueTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclTensor *tensorsOfValue[kvTensorNum];
  tensorsOfValue[0] = valueTensor;
  auto tensorValueList = aclCreateTensorList(tensorsOfValue, kvTensorNum);
  // Create the atten aclTensor.
  ret = CreateAclTensor(attenHostData, attenShape, &attenDeviceAddr, aclDataType::ACL_INT8, &attenTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &outTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> actualSeqlenVector = {sequenceLengthKV};
  auto actualSeqLengths = aclCreateIntArray(actualSeqlenVector.data(), actualSeqlenVector.size());

  int64_t numKeyValueHeads = numHeads;
  int64_t blockSize = 1;
  int64_t innerPrecise = 1;
  double scaleValue = 1 / sqrt(headDims); // 1/sqrt(d)
  string sLayerOut = "BNSD";
  char layerOut[sLayerOut.length()+1];
  strcpy(layerOut, sLayerOut.c_str());
  // 3. Call the CANN operator library API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API.
  ret = aclnnIncreFlashAttentionV3GetWorkspaceSize(queryTensor, tensorKeyList, tensorValueList, nullptr, attenTensor, actualSeqLengths, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                   nullptr, nullptr, nullptr, numHeads, scaleValue, layerOut, numKeyValueHeads, blockSize, innerPrecise, outTensor, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIncreFlashAttentionV3GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API.
  ret = aclnnIncreFlashAttentionV3(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIncreFlashAttentionV3 failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronously wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output values and copy the results from the device side to the host side. Modify this part based on the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<op::fp16_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
      std::cout << "index: " << i << ": " << static_cast<float>(resultData[i]) << std::endl;
  }

  // 6. Release resources.
  aclDestroyTensor(queryTensor);
  aclDestroyTensor(keyTensor);
  aclDestroyTensor(valueTensor);
  aclDestroyTensor(attenTensor);
  aclDestroyTensor(outTensor);
  aclDestroyIntArray(actualSeqLengths);
  aclrtFree(queryDeviceAddr);
  aclrtFree(keyDeviceAddr);
  aclrtFree(valueDeviceAddr);
  aclrtFree(attenDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
