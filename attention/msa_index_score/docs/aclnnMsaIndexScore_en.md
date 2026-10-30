# aclnnMsaIndexScore

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T11:53:15.939Z -->

## Applicable Products

<!-- npu="910b" id1 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id3 -->

## Function

- Description: Computes the block score in the Index Branch of the MSA (MiniMax Sparse Attention) module. For each query token and each KV sparse block, performs the "matmul+maxpool" operation on $Q_{idx}$ and $K_{idx}$ (optional int8 dequantization) of all causally visible tokens in the block to obtain the per-block importance score (score), which is used as the input of the subsequent TopK in the Index Branch. The same API supports both Prefill and Decode.

- **Formula**:

    - Non-quantization scenario:

    $$
    score = Maxpool[ Q_{idx}@K_{idx}^{T} ]
    $$

    - int8 quantization scenario:

    $$
    score = Maxpool[ scale \cdot Q_{idx}@K_{idx}^{T}  ]
    $$

    Complete formula (including causal mask and local_mask):

    $$
    score = Maxpool[(scale \cdot) Q_{idx}@K_{idx}^{T} + atten\_mask] + local\_mask
    $$

    $local\_mask$ is generated from `startLoc`, `initBlocks`, and `localBlocks`: logic blocks $[0, initBlocks)$ are written with $1\mathrm{e}30$; the window $[max(0, startLoc+1-localBlocks), startLoc]$ is written with $1\mathrm{e}29$ (overwriting the init at the same positions). When both are 0, $local\_mask$ is disabled.

- **Parameters**:

    > - B (Batch Size) indicates the input batch size
    > - S (Sequence Length) indicates the sequence length, where $S1$ is for the query side and $S2$ is for the key side
    > - T indicates the accumulated sum of sequence lengths of all batches, where $T1$ is for the query side and $T2$ is for the key side
    > - N (Head Num) indicates the number of heads, where $N1$ is for the query side and $N2$ is for the key side
    > - D (Head Dim) indicates the dimension of a single attention head;
    > - In the PageAttention scenario, $block\_num$ is the total number of physical blocks, $block\_size$ is the number of tokens in each block, $maxBlockNumPerSeq$ is the maximum number of logic blocks per batch (usually $\ge\lceil S2/block\_size\rceil$), and $M_b=\lceil S2/block\_size\rceil$ is the total number of logic blocks

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the `aclnnMsaIndexScoreGetWorkspaceSize` API to obtain the input parameters and calculate the required workspace size, and then call the `aclnnMsaIndexScore` API to perform the computation.

```cpp
aclnnStatus aclnnMsaIndexScoreGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *blockTableOptional,
    const aclTensor *scaleOptional,
    const aclTensor *attenMaskOptional,
    const aclTensor *actualSeqQlenOptional,
    const aclTensor *actualSeqKlenOptional,
    const aclTensor *startLoc,
    char            *layoutKeyOptional,
    int64_t          sparseMode,
    int64_t          initBlocks,
    int64_t          localBlocks,
    const aclTensor *score,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor);
aclnnStatus aclnnMsaIndexScore(
    void           *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor  *executor,
    aclrtStream     stream);
```

## aclnnMsaIndexScoreGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1600px"><colgroup>
  <col style="width: 200px">
  <col style="width: 100px">
  <col style="width: 280px">
  <col style="width: 420px">
  <col style="width: 280px">
  <col style="width: 100px">
  <col style="width: 220px">
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>query</td>
      <td>Input</td>
      <td>$Q_{idx}$ in the formula.</td>
      <td>Supported shape: TND.</td>
      <td>BFLOAT16, FLOAT16, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>3 ($[T1, N1, D]$)</td>
    </tr>
    <tr>
      <td>key</td>
      <td>Input</td>
      <td>$K_{idx}$ in the formula.</td>
      <td>Supported shapes: TND, BNBD, BBND. On A2/A3 and Ascend 950PR&950DT products, the PA key allows dim0 (physical page) to be non-contiguous; non-leading axes with size>1 must be contiguous; otherwise, the tiling rejects it. TND does not allow non-contiguous layouts.</td>
      <td>BFLOAT16, FLOAT16, INT8, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>3 ($[T2, N2, D]$) or 4 ($[block\_num, N2, block\_size, D]$, $[block\_num, block\_size, N2, D]$)</td>
    </tr>
    <tr>
      <td>blockTableOptional</td>
      <td>Input</td>
      <td>Represents the block mapping table passed in, using PageAttention storage.</td>
      <td>In the PageAttention use case, blockTableOptional must be 2D; the length of the second dimension must not be less than $maxBlockNumPerSeq$.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>2 ($[B, S2/block\_size]$)</td>
    </tr>
    <tr>
      <td>scaleOptional</td>
      <td>Input</td>
      <td>The dequantization coefficient $scale$ in the formula.</td>
      <td>For the non-quantized field, pass in <code>nullptr</code>; required in the quantization scenario. PA: BNB/BBN; TND: $[T2, N2]$ ($[T2]$ is allowed when N2=1).</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>3 ($[block\_num, N2, block\_size]$, $[block\_num, block\_size, N2]$) or 2 ($[T2, N2]$)</td>
    </tr>
    <tr>
      <td>attenMaskOptional</td>
      <td>Input</td>
      <td>The mask that controls causal visibility.</td>
      <td>Used only when sparseMode=3, serving as the base mask to control causal visibility. A value of 1 represents this bit is not involved in computation, and 0 represents this bit is involved in computation.</td>
      <td>INT8</td>
      <td>ND</td>
      <td>2 ($[2048, 2048]$)</td>
    </tr>
    <tr>
      <td>actualSeqQlenOptional</td>
      <td>Input</td>
      <td>Number of valid Query tokens in each Batch.</td>
      <td>When TND is passed in, this parameter is required and must be monotonically non-decreasing (prefix sum).</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1 ($[B+1]$)</td>
    </tr>
    <tr>
      <td>actualSeqKlenOptional</td>
      <td>Input</td>
      <td>Number of valid Key tokens in each batch.</td>
      <td>Required when key is TND, monotonically non-decreasing (prefix sum, $[B+1]$); in the PageAttention scenario, the visible $S2$ of each request ($[B]$).</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1 ($[B]$ or $[B+1]$)</td>
    </tr>
    <tr>
      <td>startLoc</td>
      <td>Input</td>
      <td>Index of the logical block where the current query resides (not the token prefix).</td>
      <td>Generates $local\_mask$ together with <code>initBlocks</code> / <code>localBlocks</code>.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1 ($[B]$)</td>
    </tr>
    <tr>
      <td>layoutKeyOptional</td>
      <td>Input</td>
      <td>Data layout of key.</td>
      <td>Value: <code>"TND"</code> / <code>"BBND"</code> / <code>"BNBD"</code>. Defaults to <code>"BBND"</code> if not passed or set to an empty string. Must match the actual shape of <code>key</code> and cannot be inferred from dimensions alone.</td>
      <td>CHAR*</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseMode</td>
      <td>Input</td>
      <td>Represents the sparse mode.</td>
      <td>If the value is 0, it represents the defaultMask mode. If the value is 3, it represents the mask of the rightDownCausal mode, corresponding to the lower-triangle use case divided by the right vertex.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>initBlocks</td>
      <td>Input</td>
      <td>The number of head blocks forcibly selected by $local\_mask$.</td>
      <td>Writes high scores $1\mathrm{e}30$ to logic blocks $[0, initBlocks)$. Optional, default $0$; must be $\ge 0$ and $\le maxBlockNumPerSeq$.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>localBlocks</td>
      <td>Input</td>
      <td>The local window length forcibly selected by $local\_mask$.</td>
      <td>The window is $[max(0, startLoc+1-localBlocks), startLoc]$; writes high scores $1\mathrm{e}29$ (overriding the init at the same position). Optional, default $1$ (aligned with MiniMax HF); set to $0$ when aligned with Triton raw score.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>score</td>
      <td>Output</td>
      <td>The $score$ in the formula.</td>
      <td>Per-block importance scores; the last dimension is the number of aligned logical blocks.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>3 ($[N1, T1, RoundUp(maxBlockNumPerSeq, 16)]$)</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size that needs to be applied by the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation process.</td>
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
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_SUCCESS</td>
      <td>0</td>
      <td>Success.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>A required input or output parameter is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data type, data format, dimension, or value does not meet the constraints.</td>
    </tr>
  </tbody>
  </table>

## aclnnMsaIndexScore

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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Workspace memory address applied by the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size applied by the Device side, obtained by the first-phase API aclnnMsaIndexScoreGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor for the op, containing the operator computation process.</td>
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

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_SUCCESS</td>
      <td>0</td>
      <td>Execution succeeded.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>Invalid parameter.</td>
    </tr>
  </tbody>
  </table>

## Constraints

- The current $block\_size$ value uses 128.
- `layoutKeyOptional` must explicitly specify the key layout: `"BBND"` ($[block\_num, block\_size, N2, D]$), `"BNBD"` ($[block\_num, N2, block\_size, D]$), or `"TND"` ($[T2, N2, D]$). Defaults to `"BBND"` if not passed.
- In the PageAttention scenario (`layoutKey` is `"BBND"` / `"BNBD"`), `blockTableOptional` must be passed in. In the TND key scenario, `blockTableOptional` must not be passed in, and `actualSeqKlenOptional` is the $[B+1]$ prefix sum.
- In the non-quantized field, the dtype of `key` is the same as that of `query` (BFLOAT16 / FLOAT16; Ascend 950PR&950DT products also supports HIFLOAT8 / FLOAT8_E5M2 / FLOAT8_E4M3FN), and `scaleOptional` must be `nullptr`. The quantization scenario supports only INT8 (fp16 query). `scaleOptional` is required, and its dtype is FLOAT: For PA, $[block\_num, N2, block\_size]$ or $[block\_num, block\_size, N2]$; for TND, $[T2, N2]$. The three FP8 types are supported only on 950, and query and key must be of the same type.
- `sparseMode` currently supports only 0 and 3:
    - When it is 0, it represents the defaultMask mode, and `attenMaskOptional` is set to `nullptr`;
    - When it is 3, it represents the rightDownCausal mode, `attenMaskOptional` must be passed in, with a shape of $[2048, 2048]$. A value of 1 represents this bit is not involved in the computation, and 0 represents this bit is involved in the computation.
- `initBlocks` and `localBlocks` must be $\ge 0$ and must not exceed the number of logical blocks (for PA, the second dimension of `blockTableOptional`; for TND, the aligned width of the last dimension of score). If both are 0, $local\_mask$ is skipped.
- `q_len` / `kv_len` can be 0 (including the whole batch). The corresponding request skips QK; the score for empty KV is filled with $-inf$; when the whole batch has $T1=0$, `SetBlockDim(1)` is called (aligned with FIA PR 9246).
- A2/A3 and Ascend 950PR&950DT products: MIX is launched based on the estimated number of M-tasks (for a single batch, $\mathrm{CeilDiv}(T1\cdot N1,128)$; for multiple batches, the upper bound of packed+$B$ is taken and truncated to the number of AICs), and short decodes do not fill idle cores. Ascend 950PR&950DT products splits `kvChunks` along S when the M-tasks cannot fill the AICs and the visible KV stile is $>1$; for a wide `blockTable`, cores are estimated only by visible pages to avoid idle spinning for short KVs.
- PageAttention `key` (BBND/BNBD) allows the first axis to be non-contiguous (`key | gap | key | ...`). Tiling reads the dim0 element stride through `GetInputStride` and writes it into `strideKvBlock`. Non-first axes (from dim1 to the last dimension with size>1) must be equal to the compact stride; otherwise, unsupported/invalid input is returned. TND `key` cannot be non-contiguous. `scale` still follows the compact layout of logical pages.
- The second dimension of the PageAttention `blockTable` can be greater than the actual number of logical KV blocks; the last dimension of score is $\mathrm{RoundUp}(width, 16)$. In the Ascend 950PR&950DT products C2UB path, a last dimension with more than 256 columns is flushed in 256-column sliding windows, and the remaining positions are filled with $-inf$.
- Precision self-verification matrix: 40 fp16/bf16/int8 cases (including PA key dim0 stride, wide `blockTable`, short decode, and q4-kv275) + 10 FP8 cases. For 950, the last line of the pass criteria is `[PASS]: 50/50 cases passed`. A2/A3 skip FP8 and expect 40/40 (skipped 10). The tolerance for fp16/bf16/int8 is $1\mathrm{e}{-3}$, and that for FP8 is $2\mathrm{e}{-2}$.
- The Ascend 950PR&950DT products kernel implementation is located in `op_kernel/arch35/`: currently, the 8-page S workspace (non-quantized fp16 / int8 fp32) is shared with A2; Cube natively supports three FP8 formats. The default soc of `--run_example` is 910b; for 950, explicitly specify `--soc=ascend950`.

## Example

Sample code is as follows (BBND PageAttention). TND: `layoutKeyOptional="TND"`, `key` is $[T2, N2, D]$, pass `nullptr` to `blockTableOptional`, and `actualSeqKlenOptional` is the prefix sum of $[B+1]$. BNBD: `layoutKeyOptional="BNBD"`, `key` is $[block\_num, N2, block\_size, D]$. For accuracy self-verification involving TND/BNBD, see [test_aclnn_msa_index_score.cpp](../examples/test_aclnn_msa_index_score.cpp).

```Cpp
#include <iostream>
#include <vector>
#include <cstdint>
#include "acl/acl.h"
#include "aclnnop/aclnn_msa_index_score.h"

using namespace std;

namespace {

#define CHECK_RET(cond) ((cond) ? true : (false))

#define LOG_PRINT(message, ...)         \
  do {                                  \
    (void)printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int64_t RoundUp(int64_t value, int64_t align) {
  return (value + align - 1) / align * align;
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
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

struct TensorResources {
  void* queryDeviceAddr = nullptr;
  void* keyDeviceAddr = nullptr;
  void* blockTableDeviceAddr = nullptr;
  void* attenMaskDeviceAddr = nullptr;
  void* actualSeqQlenDeviceAddr = nullptr;
  void* actualSeqKlenDeviceAddr = nullptr;
  void* startLocDeviceAddr = nullptr;
  void* scoreDeviceAddr = nullptr;

  aclTensor* queryTensor = nullptr;
  aclTensor* keyTensor = nullptr;
  aclTensor* blockTableTensor = nullptr;
  aclTensor* attenMaskTensor = nullptr;
  aclTensor* actualSeqQlenTensor = nullptr;
  aclTensor* actualSeqKlenTensor = nullptr;
  aclTensor* startLocTensor = nullptr;
  aclTensor* scoreTensor = nullptr;
};

int InitializeTensors(TensorResources& resources) {
  // TND query + PA_BBND key, sparseMode=3
  constexpr int64_t B = 1;
  constexpr int64_t T1 = 2;
  constexpr int64_t N1 = 2;
  constexpr int64_t N2 = 1;
  constexpr int64_t D = 128;
  constexpr int64_t S2 = 256;
  constexpr int64_t blockSize = 128;
  constexpr int64_t blockNum = 2;
  constexpr int64_t maxBlockNumPerSeq = S2 / blockSize;  // 2
  const int64_t scoreStride = RoundUp(maxBlockNumPerSeq, 16);

  std::vector<int64_t> queryShape = {T1, N1, D};
  std::vector<int64_t> keyShape = {blockNum, blockSize, N2, D};           // BBND
  std::vector<int64_t> blockTableShape = {B, maxBlockNumPerSeq};
  std::vector<int64_t> attenMaskShape = {2048, 2048};
  std::vector<int64_t> actualSeqQlenShape = {B + 1};
  std::vector<int64_t> actualSeqKlenShape = {B};
  std::vector<int64_t> startLocShape = {B};
  std::vector<int64_t> scoreShape = {N1, T1, scoreStride};

  std::vector<uint16_t> queryHostData(GetShapeSize(queryShape), 0x3C00);  // fp16 1.0
  std::vector<uint16_t> keyHostData(GetShapeSize(keyShape), 0x3C00);
  std::vector<int32_t> blockTableHostData = {0, 1};
  std::vector<int8_t> attenMaskHostData(GetShapeSize(attenMaskShape), 0);  // 0: participates in the computation
  std::vector<int32_t> actualSeqQlenHostData = {0, static_cast<int32_t>(T1)};
  std::vector<int32_t> actualSeqKlenHostData = {static_cast<int32_t>(S2)};
  std::vector<int32_t> startLocHostData = {1};  // Index of the logic block where the current query is located
  std::vector<float> scoreHostData(GetShapeSize(scoreShape), 0.0f);

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
  ret = CreateAclTensor(blockTableHostData, blockTableShape, &resources.blockTableDeviceAddr,
                        aclDataType::ACL_INT32, &resources.blockTableTensor);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    return ret;
  }
  ret = CreateAclTensor(attenMaskHostData, attenMaskShape, &resources.attenMaskDeviceAddr,
                        aclDataType::ACL_INT8, &resources.attenMaskTensor);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    return ret;
  }
  ret = CreateAclTensor(actualSeqQlenHostData, actualSeqQlenShape,
                        &resources.actualSeqQlenDeviceAddr, aclDataType::ACL_INT32,
                        &resources.actualSeqQlenTensor);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    return ret;
  }
  ret = CreateAclTensor(actualSeqKlenHostData, actualSeqKlenShape,
                        &resources.actualSeqKlenDeviceAddr, aclDataType::ACL_INT32,
                        &resources.actualSeqKlenTensor);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    return ret;
  }
  ret = CreateAclTensor(startLocHostData, startLocShape, &resources.startLocDeviceAddr,
                        aclDataType::ACL_INT32, &resources.startLocTensor);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    return ret;
  }
  ret = CreateAclTensor(scoreHostData, scoreShape, &resources.scoreDeviceAddr,
                        aclDataType::ACL_FLOAT, &resources.scoreTensor);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    return ret;
  }
  return ACL_SUCCESS;
}

int ExecuteMsaIndexScore(TensorResources& resources, aclrtStream stream,
                         void** workspaceAddr, uint64_t* workspaceSize) {
  int64_t sparseMode = 3;
  int64_t initBlocks = 0;
  int64_t localBlocks = 1;
  char layoutKey[] = "BBND";
  aclOpExecutor* executor = nullptr;

  // Non-quantized: pass nullptr to scaleOptional
  int ret = aclnnMsaIndexScoreGetWorkspaceSize(
      resources.queryTensor,
      resources.keyTensor,
      resources.blockTableTensor,
      nullptr,  // scaleOptional
      resources.attenMaskTensor,
      resources.actualSeqQlenTensor,
      resources.actualSeqKlenTensor,
      resources.startLocTensor,
      layoutKey,
      sparseMode,
      initBlocks,
      localBlocks,
      resources.scoreTensor,
      workspaceSize,
      &executor);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclnnMsaIndexScoreGetWorkspaceSize failed. ERROR: %d\n", ret);
    return ret;
  }

  if (*workspaceSize > 0ULL) {
    ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
      return ret;
    }
  }

  ret = aclnnMsaIndexScore(*workspaceAddr, *workspaceSize, executor, stream);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclnnMsaIndexScore failed. ERROR: %d\n", ret);
    return ret;
  }
  return ACL_SUCCESS;
}

int PrintScoreOutResult(const std::vector<int64_t>& shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0.0f);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
    return ret;
  }
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("score result[%ld] is: %f\n", i, resultData[i]);
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
  if (resources.blockTableTensor) {
    aclDestroyTensor(resources.blockTableTensor);
  }
  if (resources.attenMaskTensor) {
    aclDestroyTensor(resources.attenMaskTensor);
  }
  if (resources.actualSeqQlenTensor) {
    aclDestroyTensor(resources.actualSeqQlenTensor);
  }
  if (resources.actualSeqKlenTensor) {
    aclDestroyTensor(resources.actualSeqKlenTensor);
  }
  if (resources.startLocTensor) {
    aclDestroyTensor(resources.startLocTensor);
  }
  if (resources.scoreTensor) {
    aclDestroyTensor(resources.scoreTensor);
  }

  if (resources.queryDeviceAddr) {
    aclrtFree(resources.queryDeviceAddr);
  }
  if (resources.keyDeviceAddr) {
    aclrtFree(resources.keyDeviceAddr);
  }
  if (resources.blockTableDeviceAddr) {
    aclrtFree(resources.blockTableDeviceAddr);
  }
  if (resources.attenMaskDeviceAddr) {
    aclrtFree(resources.attenMaskDeviceAddr);
  }
  if (resources.actualSeqQlenDeviceAddr) {
    aclrtFree(resources.actualSeqQlenDeviceAddr);
  }
  if (resources.actualSeqKlenDeviceAddr) {
    aclrtFree(resources.actualSeqKlenDeviceAddr);
  }
  if (resources.startLocDeviceAddr) {
    aclrtFree(resources.startLocDeviceAddr);
  }
  if (resources.scoreDeviceAddr) {
    aclrtFree(resources.scoreDeviceAddr);
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

}  // namespace

int main() {
  int32_t deviceId = 0;
  aclrtStream stream = nullptr;
  TensorResources resources = {};
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
  constexpr int64_t T1 = 2;
  constexpr int64_t N1 = 2;
  constexpr int64_t maxBlockNumPerSeq = 2;
  std::vector<int64_t> scoreShape = {N1, T1, RoundUp(maxBlockNumPerSeq, 16)};
  int ret = ACL_SUCCESS;

  ret = Init(deviceId, &stream);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
    return ret;
  }

  ret = InitializeTensors(resources);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("InitializeTensors failed. ERROR: %d\n", ret);
    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return ret;
  }

  ret = ExecuteMsaIndexScore(resources, stream, &workspaceAddr, &workspaceSize);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("ExecuteMsaIndexScore failed. ERROR: %d\n", ret);
    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return ret;
  }

  ret = aclrtSynchronizeStream(stream);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return ret;
  }

  PrintScoreOutResult(scoreShape, &resources.scoreDeviceAddr);

  CleanupResources(resources, workspaceAddr, stream, deviceId);
  return 0;
}
```

The differences between TND / BNBD and the preceding BBND example are as follows (for complete accuracy test cases, see `L1-bnbd*`, `L1-tnd*`, and `L0-tnd-tiny` in [test_aclnn_msa_index_score.cpp](../examples/test_aclnn_msa_index_score.cpp)).

```Cpp
// BNBD PageAttention: layoutKey="BNBD", key is [block_num, N2, block_size, D]
char layoutKeyBnbd[] = "BNBD";
std::vector<int64_t> keyShapeBnbd = {blockNum, N2, blockSize, D};

// TND packed key: layoutKey="TND", do not pass blockTableOptional, actualSeqKlenOptional is the prefix sum of [B+1]
char layoutKeyTnd[] = "TND";
constexpr int64_t T2 = 256;
std::vector<int64_t> keyShapeTnd = {T2, N2, D};
std::vector<int64_t> actualSeqKlenShapeTnd = {B + 1};  // For example, {0, T2}
aclTensor* blockTableTnd = nullptr;
```
