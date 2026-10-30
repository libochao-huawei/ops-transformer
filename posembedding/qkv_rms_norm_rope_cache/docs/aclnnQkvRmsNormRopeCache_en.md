# aclnnQkvRmsNormRopeCache

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:57:13.245Z -->

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
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: Inputs a fused QKV tensor, splits it into Q, K, and V tensors via SplitVD, performs RmsNorm, ApplyRotaryPosEmb, Quant, and Scatter fusion operations, and outputs qOut, kCache, vCache, qBeforeQuant (optional), kBeforeQuant (optional), and vBeforeQuant (optional).
- The scenarios currently supported by this API are as follows:

  |Scenario Type|Overview|
  |:---|:---|
  |<ul><li>cacheMode is PA_NZ</li><li>Q is non-quantized</li><li>K and V support non-quantized, symmetric quantization, and asymmetric quantization</li><li>qBeforeQuant/kBeforeQuant/vBeforeQuant are not output</li></ul>|The qkv shape is [$B_{qkv}$ *$S_{qkv}$, $N_{qkv}$* $D_{qkv}$]. q, k, and v have exactly the same D dimension. Main computation process and corresponding outputs:<br><ul><li>qkv through SplitVD->q, k, v</li><li>q through RmsNorm, RoPE->qOut</li><li>k through RmsNorm, RoPE, Quant(optional), Scatter->kCache</li><li>v through Quant(optional), Scatter->vCache</li></ul>|

- Formula:

  (1) SplitVD:

  In the following formula, $N_q$, $N_k$, and $N_v$ denote the numbers of attention heads of the q, k, and v components, respectively. The following conditions must be met:

  $$
  \begin{cases}
  N_k = N_v \\
  N_{qkv} = N_k + N_v + N_q \\
  D_{qkv} = D_q = D_k = D_v
  \end{cases}
  $$

  $$
  \begin{aligned}
  q &= qkv[..., [:N_q] * D_{qkv}] \\
  k &= qkv[..., [N_q:-N_v] * D_{qkv}] \\
  v &= qkv[..., [-N_v:] * D_{qkv}]
  \end{aligned}
  $$

  (2) RmsNorm:

  Here, x and y represent the input tensor and output tensor of RmsNorm, respectively. The normalization is performed along the last dimension (feature dimension). This computation rule applies to both q and k components.

  $$
  squareX = x * x
  $$

  $$
  meanSquareX = squareX.mean(dim = -1, keepdim = True)
  $$

  $$
  rms = \sqrt{meanSquareX + epsilon}
  $$

  $$
  y = (x / rms) * gamma
  $$

  (3) RoPE (Half-and-Half):

  Here, y refers to the output result after completing the RmsNorm computation.

  $$
  y1 = y[\ldots, :d/2]
  $$

  $$
  y2 = y[\ldots, d/2:]
  $$

  $$
  y\_RoPE = torch.cat((-y2, y1), dim = -1)
  $$

  $$
  y\_embed = (y * cos) + y\_RoPE * sin
  $$

  (4) Quant:

  Non-quantized:

  $$
  kQuant = kRoPE \\
  vQuant = v
  $$

  Symmetric quantization:

  $$
  kQuant = kRoPE / kScale \\
  vQuant = v / vScale
  $$

  Asymmetric quantization part:

  $$
  kQuant = kRoPE / kScale + kOffset \\
  vQuant = v / vScale + vOffset
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnQkvRmsNormRopeCacheGetWorkspaceSize" API to obtain the input parameters and calculate the required workspace size based on the flow, and then call the "aclnnQkvRmsNormRopeCache" API to perform the computation.

```Cpp
aclnnStatus aclnnQkvRmsNormRopeCacheGetWorkspaceSize(
  const aclTensor   *qkv,
  const aclTensor   *qGamma,
  const aclTensor   *kGamma,
  const aclTensor   *cos,
  const aclTensor   *sin,
  const aclTensor   *index,
  aclTensor         *qOut,
  aclTensor         *kCache,
  aclTensor         *vCache,
  const aclTensor   *kScaleOptional,
  const aclTensor   *vScaleOptional,
  const aclTensor   *kOffsetOptional,
  const aclTensor   *vOffsetOptional,
  const aclIntArray *qkvSize,
  const aclIntArray *headNums,
  double             epsilon,
  char              *cacheModeOptional,
  const aclTensor   *qOutBeforeQuant,
  const aclTensor   *kOutBeforeQuant,
  const aclTensor   *vOutBeforeQuant,
  uint64_t          *workspaceSize,
  aclOpExecutor     **executor)
```

```Cpp
aclnnStatus aclnnQkvRmsNormRopeCache(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream      stream)
```

## aclnnQkvRmsNormRopeCacheGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1503px"><colgroup>
  <col style="width: 146px">
  <col style="width: 120px">
  <col style="width: 271px">
  <col style="width: 392px">
  <col style="width: 228px">
  <col style="width: 101px">
  <col style="width: 100px">
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
      <td>qkv</td>
      <td>Input</td>
      <td>Input data used for splitting q, k, and v, corresponding to qkv in the formula.</td>
      <td><ul><li>Does not support null tensors.</li><li>shape: [B<sub>qkv</sub> * S<sub>qkv</sub>, N<sub>qkv</sub> * D<sub>qkv</sub>].</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>qGamma</td>
      <td>Input</td>
      <td>Input data used for the rms_norm computation of q, corresponding to gamma in the formula.</td>
      <td><ul><li>Does not support null Tensor.</li><li>Same data type as the input qkv.</li><li>The shape is [D<sub>qkv</sub>].</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kGamma</td>
      <td>Input</td>
      <td>Input data used for the rms_norm computation of k, corresponding to gamma in the formula.</td>
      <td><ul><li>Does not support null Tensor.</li><li>Same data type as the input qkv.</li><li>The shape is [D<sub>qkv</sub>].</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cos</td>
      <td>Input</td>
      <td>Input data used for rope computation. Applies cosine transformation to the input tensor, corresponding to cos in the formula.</td>
      <td><ul><li>Does not support null tensors.</li><li>Same data type as the input qkv.</li><li>Shape is [B<sub>qkv</sub> * S<sub>qkv</sub>, 1 * D_rope], where D_rope = D<sub>qkv</sub> and (D<sub>qkv</sub> * number of bytes occupied by the qkv data type) must be divisible by 32.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sin</td>
      <td>Input</td>
      <td>Input data used for the rope computation. Performs sine transformation on the input tensor. The corresponding computation in the formula is sin.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The data type and format must remain consistent with those of the input cos.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>index</td>
      <td>Input</td>
      <td>Used for specifying the index position for writing to the cache, corresponding to the index in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>Same data type as input qkv.</li><li>The shape is [B<sub>qkv</sub> * S<sub>qkv</sub>]. The value of index cannot be duplicated, and the value range is [-1, BlockNum * BlockSize].</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>qOut</td>
      <td>Input \ output</td>
      <td>The cache to be applied for, with input and output being in-place copy.</td>
      <td><ul><li>Does not support null Tensor.</li><li>Same data type as the input qkv.</li><li>The shape is [B<sub>qkv</sub> * S<sub>qkv</sub>, N<sub>q</sub> * D<sub>qkv</sub>].</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kCache</td>
      <td>input \ output</td>
      <td>The cache applied for. The input and output share the same address (in-place copy).</td>
      <td><ul><li>Does not support null Tensor.</li><li>Same data type as the input qkv (k non-quantized), or INT8 (k quantized).</li><li>The shape is [BlockNum, N<sub>k</sub> * D<sub>qkv</sub> // 16, BlockSize, 16] (k non-quantized), or [BlockNum, N<sub>k</sub> * D<sub>qkv</sub> // 32, BlockSize, 32] (k quantized).</li></ul></td>
      <td>FLOAT16, BFLOAT16, INT8</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vCache</td>
      <td>input \/output</td>
      <td>Cache to allocate. The input and output use the same address (in-place copy).</td>
      <td><ul><li>Does not support null tensors.</li><li>The data type must be the same as the input qkv (v non-quantized), or INT8 (v quantization).</li><li>The shape is [BlockNum, N<sub>k</sub> * D<sub>qkv</sub> // 16, BlockSize, 16] (v non-quantized), or [BlockNum, N<sub>k</sub> * D<sub>qkv</sub> // 32, BlockSize, 32] (v quantization).</li></ul></td>
      <td>FLOAT16, BFLOAT16, INT8</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kScaleOptional</td>
      <td>Optional input</td>
      <td>Required when the data type of kCache is INT8. Corresponds to kScale in the formula.</td>
      <td>The shape is [N<sub>k</sub>, D<sub>qkv</sub>].</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vScaleOptional</td>
      <td>Optional input</td>
      <td>Required when the data type of vCache is INT8. Corresponds to vScale in the formula.</td>
      <td>The shape is [N<sub>v</sub>, D<sub>qkv</sub>].</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kOffsetOptional</td>
      <td>Optional input</td>
      <td>Corresponds to kOffset in the formula.</td>
      <td><ul><li>This input parameter is required when the data type of kCache is INT8, the corresponding kScaleOptional input exists, and asymmetric quantization is used.</li><li>The shape is [N<sub>k</sub>, D<sub>qkv</sub>].</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vOffsetOptional</td>
      <td>Optional input</td>
      <td>Corresponds to vOffset in the formula.</td>
      <td><ul><li>This input parameter is required when the data type of vCache is INT8, the corresponding vScaleOptional input exists, and asymmetric quantization is used.</li><li>The shape is [N<sub>v</sub>, D<sub>qkv</sub>].</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>qkvSize</td>
      <td>Input</td>
      <td>Passed in the [B<sub>qkv</sub>, S<sub>qkv</sub>, N<sub>qkv</sub>, D<sub>qkv</sub>] order. Provides the sizes of dimensions B, S, N, and D of the qkv matrix input parameter.</td>
      <td><ul><li>Requires N<sub>qkv</sub> = N<sub>q</sub> + N<sub>k</sub> + N<sub>v</sub>.</li><li>The number of elements must be exactly 4, and each element must be a valid non-zero positive integer.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>headNums</td>
      <td>Input</td>
      <td>Passed in the order of [N<sub>q</sub>, N<sub>k</sub>, N<sub>v</sub>], providing the specific sizes of the N dimension into which qkv components are split in the input parameter qkv matrix.</td>
      <td>[N<sub>q</sub>, N<sub>k</sub>, N<sub>v</sub>] must be the actual original values and must not be reduced.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Input</td>
      <td>Prevents division by zero in RmsNorm computation. Corresponds to epsilon in the formula.</td>
      <td>Recommended value: 1e-6</td>
      <td>FLOAT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cacheModeOptional</td>
      <td>Input</td>
      <td>Indicates the cache format.</td>
      <td>The recommended value is "PA_NZ".</td>
      <td>CHAR*</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>qOutBeforeQuant</td>
      <td>Output</td>
      <td>Indicates the data to be written to qOut.</td>
      <td><ul><li>The typing and data type must match the input parameter qkv and remain consistent.</li><li>The shape is [B<sub>qkv</sub> * S<sub>qkv</sub>, N<sub>q</sub> * D<sub>qkv</sub>].</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kOutBeforeQuant</td>
      <td>Output</td>
      <td>Indicates that the data to be written to vCache is the intermediate computation result before quantization and Scatter.</td>
      <td><ul><li>Typing and data type must match the input parameter qkv.</li><li>The shape is [B<sub>qkv</sub> * S<sub>qkv</sub>, N<sub>q</sub> * D<sub>qkv</sub>].</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vOutBeforeQuant</td>
      <td>Output</td>
      <td>Indicates that the data will be written to vCache. It is the intermediate computation result before quantization and Scatter.</td>
      <td><ul><li>Typing and data type must remain consistent with the input parameter qkv.</li><li>The shape is [B<sub>qkv</sub> * S<sub>qkv</sub>, N<sub>q</sub> * D<sub>qkv</sub>].</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size that needs to be applied for on the Device side.</td>
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
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:
  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 253px">
  <col style="width: 140px">
  <col style="width: 762px">
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
      <td rowspan="1">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="1">161001</td>
      <td>The input qkv, qGamma, kGamma, cos, sin, or other parameters are null pointers.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type of the input or output is not within the supported range.</td>
    </tr>
    <tr>
      <td>The parameter dimension of the input or output is not within the supported range.</td>
    </tr>
    <tr>
      <td>dim is not within the specified value range.</td>
    </tr>
  </tbody></table>

## aclnnQkvRmsNormRopeCache

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
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
      <td>Workspace size applied on the Device side, obtained by the first-phase API aclnnQkvRmsNormRopeCacheGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return value. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnQkvRmsNormRopeCache defaults to a deterministic implementation.

- Input shape constraints:
    - B<sub>qkv</sub> is the batch_size of the input qkv, and S<sub>qkv</sub> is the sequence length of the input qkv, whose size is determined by qkvSize.
    - N<sub>qkv</sub> is the head number of the input qkv. D<sub>qkv</sub> is the head dim of the input qkv, which currently supports only 128. D<sub>q</sub>, D<sub>k</sub>, and D<sub>v</sub> are the head dims of q, k, and v respectively, requiring D<sub>qkv</sub> = D<sub>q</sub> = D<sub>k</sub> = D<sub>v</sub>. D<sub>qkv</sub> must satisfy that (D<sub>qkv</sub> * byte size of the qkv data type) is divisible by 32.
    - According to the rope rules, D<sub>k</sub> and D<sub>q</sub> must be even numbers. If cacheMode is PA_NZ, D<sub>k</sub> and D<sub>q</sub> must be 32-byte aligned; BlockSize must be 32-byte aligned.
    - For the 32B alignment cases described above, the alignment value is determined by the data type of the cache. Taking BlockSize as an example: if the cache data type is int8, BlockSize % 32 = 0 must be satisfied; if the cache data type is float16, BlockSize % 16 = 0 must be satisfied; if the dtypes of the kCache and vCache parameters are inconsistent, BlockSize must satisfy both BlockSize % 32 = 0 and BlockSize % 16 = 0.
    - BlockNum is the number of memory blocks written to the cache, and its size is determined by the user input scenario. It must satisfy BlockNum >= Ceil(S<sub>qkv</sub> / BlockSize) * B<sub>qkv</sub>.
    - requireMemory indicates the size of the space required to store data. The following requirement must be met: requireMemory >= (B<sub>qkv</sub> *S<sub>qkv</sub>* N<sub>qkv</sub> *D<sub>qkv</sub> + 2* D<sub>qkv</sub> + 2 *B<sub>qkv</sub>* S<sub>qkv</sub> *D<sub>qkv</sub> + B<sub>qkv</sub>* S<sub>qkv</sub> *N<sub>q</sub>* D<sub>qkv</sub> + BlockNum *BlockSize* N<sub>v</sub> *D<sub>qkv</sub> + BlockNum* BlockSize *N<sub>k</sub>* D<sub>qkv</sub>) *sizeof(FLOAT16) + B<sub>qkv</sub>* S<sub>qkv</sub> *sizeof(INT64) + (2* N<sub>k</sub> *D<sub>qkv</sub> + 2* N<sub>v</sub>) * sizeof(FLOAT). If the calculated requireMemory size exceeds the total GM space of the current AI processor, this API is not supported.
- Other constraints:
    - For index, the value range of index values must be [-1, BlockNum * BlockSize). Values must not be duplicated. When index is -1, it indicates skipping the update.
    - kScaleOptional and vScaleOptional represent scaling factors for symmetric quantization, so their values cannot be 0 if passed.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_qkv_rms_norm_rope_cache.h"

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<int8_t> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    // Boilerplate code for resource initialization.
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
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy data from the host to the device.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1. Boilerplate code for device/stream initialization. Refer to the acl API.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output as defined by the API.
    std::vector<int64_t> qkvShape = {48, 2304};
    std::vector<int64_t> qGammaShape = {128};
    std::vector<int64_t> kGammaShape = {128};
    std::vector<int64_t> cosShape = {48, 128};
    std::vector<int64_t> sinShape = {48, 128};
    std::vector<int64_t> indexShape = {48};
    std::vector<int64_t> qOutShape = {48, 2048};
    std::vector<int64_t> kCacheShape = {16, 4, 128, 32};
    std::vector<int64_t> vCacheShape = {16, 4, 128, 32};
    std::vector<int64_t> kScaleShape = {1, 128};
    std::vector<int64_t> vScaleShape = {1, 128};
    std::vector<int64_t> qkv_size_list = {16, 3, 18, 128};
    std::vector<int64_t> head_nums_list = {16, 1, 1};

    std::vector<int16_t> qkvHostData(48 * 2304, 0);
    std::vector<int16_t> qGammaHostData(128, 0);
    std::vector<int16_t> kGammaHostData(128, 0);
    std::vector<int16_t> cosHostData(48 * 128, 0);
    std::vector<int16_t> sinHostData(48 * 128, 0);
    std::vector<int64_t> indexHostData(48, 0);
    std::vector<int16_t> qOutHostData(48 * 2048, 0);
    std::vector<int16_t> kCacheHostData(16 * 4 * 128 * 32, 0);
    std::vector<int16_t> vCacheHostData(16 * 4 * 128 * 32, 0);
    std::vector<float> kScaleHostData(1 * 128, 0.0f);
    std::vector<float> vScaleHostData(1 * 128, 0.0f);

    void *qkvDeviceAddr = nullptr;
    void *qGammaDeviceAddr = nullptr;
    void *kGammaDeviceAddr = nullptr;
    void *cosDeviceAddr = nullptr;
    void *sinDeviceAddr = nullptr;
    void *indexDeviceAddr = nullptr;
    void *qOutDeviceAddr = nullptr;
    void *kCacheDeviceAddr = nullptr;
    void *vCacheDeviceAddr = nullptr;
    void *kScaleDeviceAddr = nullptr;
    void *vScaleDeviceAddr = nullptr;

    aclTensor *qkv = nullptr;
    aclTensor *qGamma = nullptr;
    aclTensor *kGamma = nullptr;
    aclTensor *cos = nullptr;
    aclTensor *sin = nullptr;
    aclTensor *index = nullptr;
    aclTensor *qOut = nullptr;
    aclTensor *kCache = nullptr;
    aclTensor *vCache = nullptr;
    aclTensor *kScale = nullptr;
    aclTensor *vScale = nullptr;

    aclIntArray *qkv_size = aclCreateIntArray(qkv_size_list.data(), qkv_size_list.size());
    aclIntArray *head_nums = aclCreateIntArray(head_nums_list.data(), head_nums_list.size());

    double epsilon = 1e-6;
    char *cacheMode = "PA_NZ";

    ret = CreateAclTensor(qkvHostData, qkvShape, &qkvDeviceAddr, aclDataType::ACL_FLOAT16, &qkv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(qGammaHostData, qGammaShape, &qGammaDeviceAddr, aclDataType::ACL_FLOAT16, &qGamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kGammaHostData, kGammaShape, &kGammaDeviceAddr, aclDataType::ACL_FLOAT16, &kGamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cosHostData, cosShape, &cosDeviceAddr, aclDataType::ACL_FLOAT16, &cos);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sinHostData, sinShape, &sinDeviceAddr, aclDataType::ACL_FLOAT16, &sin);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(indexHostData, indexShape, &indexDeviceAddr, aclDataType::ACL_INT64, &index);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(qOutHostData, qOutShape, &qOutDeviceAddr, aclDataType::ACL_FLOAT16, &qOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kCacheHostData, kCacheShape, &kCacheDeviceAddr, aclDataType::ACL_INT8, &kCache);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(vCacheHostData, vCacheShape, &vCacheDeviceAddr, aclDataType::ACL_INT8, &vCache);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kScaleHostData, kScaleShape, &kScaleDeviceAddr, aclDataType::ACL_FLOAT, &kScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(vScaleHostData, vScaleShape, &vScaleDeviceAddr, aclDataType::ACL_FLOAT, &vScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Change it to the specific API.
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;

    // Call the first-phase API of aclnnQkvRmsNormRopeCache
    ret = aclnnQkvRmsNormRopeCacheGetWorkspaceSize(qkv, qGamma, kGamma, cos, sin, index, qOut, kCache, vCache, kScale,
                                                   vScale, nullptr, nullptr, qkv_size, head_nums, epsilon, cacheMode,
                                                   nullptr, nullptr, nullptr, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQkvRmsNormRopeCacheGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // Call the second-phase API of aclnnQkvRmsNormRopeCache.
    ret = aclnnQkvRmsNormRopeCache(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQkvRmsNormRopeCache failed. ERROR: %d\n", ret); return ret);

    // 4. Boilerplate: synchronize and wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value, copy the result from the device memory to the host, and modify the code based on the specific API definition.
    PrintOutResult(kCacheShape, &kCacheDeviceAddr);
    PrintOutResult(vCacheShape, &vCacheDeviceAddr);

    // 6. Release the aclTensor, and modify the code based on the specific API definition.
    aclDestroyTensor(qkv);
    aclDestroyTensor(qGamma);
    aclDestroyTensor(kGamma);
    aclDestroyTensor(cos);
    aclDestroyTensor(sin);
    aclDestroyTensor(index);
    aclDestroyTensor(qOut);
    aclDestroyTensor(kCache);
    aclDestroyTensor(vCache);
    aclDestroyTensor(kScale);
    aclDestroyTensor(vScale);

    // 7. Release the device resource.
    aclrtFree(qkvDeviceAddr);
    aclrtFree(qGammaDeviceAddr);
    aclrtFree(kGammaDeviceAddr);
    aclrtFree(cosDeviceAddr);
    aclrtFree(sinDeviceAddr);
    aclrtFree(indexDeviceAddr);
    aclrtFree(qOutDeviceAddr);
    aclrtFree(kCacheDeviceAddr);
    aclrtFree(vCacheDeviceAddr);
    aclrtFree(kScaleDeviceAddr);
    aclrtFree(vScaleDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
