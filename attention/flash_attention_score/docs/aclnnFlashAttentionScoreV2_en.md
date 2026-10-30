# aclnnFlashAttentionScoreV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T08:46:49.682Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
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

- Description: In training scenarios, uses the FlashAttention algorithm to implement self-attention computation. **Compared with the [aclnnFlashAttentionScore](./aclnnFlashAttentionScore_en.md) API, this API adds the pseType parameter**:

  - When pseType=1, the implementation is the same as [aclnnFlashAttentionScore](./aclnnFlashAttentionScore_en.md).
  - When pseType is set to other values, mul is performed first, followed by add.

- Formula:

  The forward calculation formula of attention is as follows:

  - When pseType = 1, the formula is the same as that of [aclnnFlashAttentionScore](./aclnnFlashAttentionScore_en.md).

  - For other values of pseType, the formula is as follows:

    $$
    attention\_out=Dropout(Softmax(Mask(scale*(query*key^T) + pse),atten\_mask),keep\_prob)*value
    $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must call the "aclnnFlashAttentionScoreV2GetWorkspaceSize" API to obtain the workspace size required for the computation and the executor that contains the computation process, and then call the "aclnnFlashAttentionScoreV2" API to execute the computation.

```c++
aclnnStatus aclnnFlashAttentionScoreV2GetWorkspaceSize(
  const aclTensor   *query,
  const aclTensor   *key,
  const aclTensor   *value,
  const aclTensor   *realShiftOptional,
  const aclTensor   *dropMaskOptional,
  const aclTensor   *paddingMaskOptional,
  const aclTensor   *attenMaskOptional,
  const aclIntArray *prefixOptional,
  const aclIntArray *qStartIdxOptional,
  const aclIntArray *kvStartIdxOptional,
  double             scaleValue,
  double             keepProb,
  int64_t            preTokens,
  int64_t            nextTokens,
  int64_t            headNum,
  char              *inputLayout,
  int64_t            innerPrecise,
  int64_t            sparseMode,
  int64_t            pseType,
  const aclTensor   *softmaxMaxOut,
  const aclTensor   *softmaxSumOut,
  const aclTensor   *softmaxOutOut,
  const aclTensor   *attentionOutOut,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnFlashAttentionScoreV2(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  const aclrtStream stream)
```

## aclnnFlashAttentionScoreV2GetWorkspaceSize

- **Parameters**
  <table style="undefined;table-layout: fixed; width: 1452px"><colgroup>
    <col style="width: 174px">
    <col style="width: 121px">
    <col style="width: 253px">
    <col style="width: 262px">
    <col style="width: 213px">
    <col style="width: 115px">
    <col style="width: 169px">
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
        <td>Query in the formula.</td>
        <td>The data type is the same as that of key/value.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>[BNSD], [BSND], [BSH], [SBH]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>key</td>
        <td>Input</td>
        <td>Key in the formula.</td>
        <td>The data type is the same as that of query/value.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>[BNSD], [BSND], [BSH], [SBH]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>value</td>
        <td>Input</td>
        <td>The value in the Formula.</td>
        <td>The data type is the same as that of query/key.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>[BNSD], [BSND], [BSH], [SBH]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>realShiftOptional</td>
        <td>Optional input</td>
        <td>pse in the formula.</td>
        <td>The data type is the same as that of query. This parameter must be used together with pseType.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>[B,N,Sq,Skv], [B,N,1,Skv], [1,N,Sq,Skv], [B,N,1024,Skv], [1,N,1024,Skv], [B,N], [N]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dropMaskOptional</td>
        <td>Optional input</td>
        <td>Dropout in the formula.</td>
        <td>-</td>
        <td>UINT8</td>
        <td>ND</td>
        <td>0, 1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>paddingMaskOptional</td>
        <td>Optional input</td>
        <td>Reserved parameter, currently unused.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>attenMaskOptional</td>
        <td>Optional input</td>
        <td>atten_mask in the formula.</td>
        <td>A value of 1 indicates that the bit does not participate in the computation, and a value of 0 indicates that the bit participates in the computation.</td>
        <td>BOOL, UINT8</td>
        <td>ND</td>
        <td>[B,N,Sq,Skv], [B,1,Sq,Skv], [1,1,Sq,Skv], [Sq,Skv] </td>
        <td>√</td>
      </tr>
      <tr>
        <td>prefixOptional</td>
        <td>Optional input</td>
        <td>Indicates the value of N for each Batch in the prefix sparse computation scenario.</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>0, 1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>qStartIdxOptional</td>
        <td>Optional input</td>
        <td>Indicates the start index of the query sequence of the current block in the global sequence in the long sequence scenario.</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>0, 1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>kvStartIdxOptional</td>
        <td>Optional input</td>
        <td>Indicates the start index of the key and value sequences of the current chunk in the global sequence in the KV chunking use case.</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>0, 1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>scaleValue</td>
        <td>Optional input</td>
        <td>The scale in the formula, indicating the scaling factor.</td>
        <td>-</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>keepProb</td>
        <td>Optional input</td>
        <td>Indicates the proportion of 1 in dropMaskOptional.</td>
        <td>The value range is (0, 1].</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>preTokens</td>
        <td>Optional input</td>
        <td>Used for sparse computation. Indicates the left boundary of the sliding window.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>nextTokens</td>
        <td>Optional input</td>
        <td>Used for sparse computation. Indicates the right boundary of the sliding window.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>headNum</td>
        <td>Input</td>
        <td>Indicates the number of heads on a single device, that is, the length of the N axis of input query.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>inputLayout</td>
        <td>Input</td>
        <td>Indicates the data layout format of the input query, key, and value.</td>
        <td>Supports BSH, SBH, BSND, and BNSD.</td>
        <td>String</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>innerPrecise</td>
        <td>Optional input</td>
        <td>Used to improve precision.</td>
        <td>The default value 0 is recommended.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>sparseMode</td>
        <td>Optional input</td>
        <td>Indicates the sparse mode.</td>
        <td>The supported values are 0, 1, 2, 3, 4, 5, and 6.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>pseType</td>
        <td>Optional input</td>
        <td>Controls the computation order of mul and add. The value can be 0, 1, 2, or 3.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>softmaxMaxOut</td>
        <td>Output</td>
        <td>Intermediate Max result of the Softmax calculation, used for backward computation.</td>
        <td>-</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[B,N,Sq,8]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>softmaxSumOut</td>
        <td>Output</td>
        <td>Intermediate Sum result of the Softmax computation, used for backward computation.</td>
        <td>-</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>[B,N,Sq,8]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>softmaxOutOut</td>
        <td>Output</td>
        <td>Reserved parameter, not used currently.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>attentionOutOut</td>
        <td>Output</td>
        <td>Final output of the formula.</td>
        <td>The data type and shape are the same as those of query.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>[BNSD], [BSND], [BSH], [SBH]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the size of the workspace to be allocated on the device.</td>
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

  <table style="undefined;table-layout: fixed;width: 1202px"><colgroup>
  <col style="width: 262px">
  <col style="width: 121px">
  <col style="width: 819px">
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
      <td>The input parameter is a required input, output, or required attribute, and is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>The data types of query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, softmaxMaxOut, softmaxSumOut, softmaxOutOut, and attentionOutOut are not in the supported range.</td>
    </tr>
    <tr>
      <td>The data formats of query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, softmaxMaxOut, softmaxSumOut, softmaxOutOut, and attentionOutOut are not in the supported range.</td>
    </tr>
  </tbody>
  </table>

## aclnnFlashAttentionScoreV2

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1154px"><colgroup>
  <col style="width: 153px">
  <col style="width: 121px">
  <col style="width: 880px">
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
      <td>The workspace memory address allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>The size of the workspace allocated on the Device side, obtained by calling the first-phase API aclnnFlashAttentionScoreV2GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the computation flow of the operator.</td>
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
  - aclnnFlashAttentionScoreV2 defaults to a deterministic implementation.
- When using this API with PyTorch, ensure that the versions of CANN-related packages and PyTorch-related packages are compatible.
- Constraints on the query, key, and value inputs:
  - B: The batchsize must be equal.
  - D: The head dimensions must satisfy (qD == kD && kD >= vD).
  - The inputLayout must be consistent.
- The shapes of the input key and value must be consistent except for D.
- Constraints on data shapes, using BSND and BNSD of inputLayout as examples (in BSH and SBH, H = N * D), where:
    - B: The value range is 1 to 2M. When prefixOptional is present, B supports a maximum of 2K.
    - N: ranges from 1 to 256.
    - S: ranges from 1 to 1M.
    - D: ranges from 1 to 768.
- The data layout format of query, key, and value supports interpretation from multiple dimensions, where B (Batch) indicates the input sample batch size, S (Seq-Length) indicates the input sample sequence length, H (Head-Size) indicates the hidden layer size, N (Head-Num) indicates the number of heads, D (Head-Dim) indicates the minimum unit dimension of the hidden layer, and D = H/N is satisfied.
- realShiftOptional: If Sq is greater than 1024, Sq and Skv are of the same length in each batch, and the mask is a lower-triangular mask with sparseMode of 0, 2, or 3, you can enable alibi position encoding compression. In this case, you only need to input the last 1024 rows of the original PSE for memory optimization, that is, alibi_compress = ori_pse[:, :, -1024:, :]. The details are as follows:
  - When the parameter differs for each batch, the shape is BNHSkv (H = 1024).
  - When the parameter is the same for each batch, the shape is 1NHSkv (H = 1024).
  - If pseType is 2 or 3, the data type must be FLOAT32, and the supported shapes are [B,N] or [N].
  - If this parameter is not enabled, pass nullptr to realShiftOptional and 1 to pseType.
- innerPrecise: Currently, 0 and 1 are reserved configuration values, and 2 enables invalid row computation. Invalid row computation prevents precision loss caused by full-row masks during computation, but degrades performance. If the operator detects an invalid row use case, it automatically enables invalid row computation, for example, when sparseMode is 3 and Sq > Skv.
- Meaning of each pseType value

    | pseType     | Meaning                       |      Note   |
    | ----------- | --------------------------------- | ----------|
    | 0           | Externally input pse: mul first, then add | - |
    | 1           | Externally input pse: add first, then mul | Consistent with the implementation of [FlashAttentionScore](./aclnnFlashAttentionScore_en.md). |
    | 2           | Internally generated pse: mul first, then add | - |
    | 3           | Internally generated pse: mul first, then add, then sqrt | - |

- When pseType is 2 or 3, only equal-length Sq and Skv are supported.
- Constraints on sparseMode are as follows:
  - When all attenMaskOptional shapes are smaller than 2048 and identical, it is recommended to use the default mode to reduce memory usage.
  - When sparseMode is set to 1, 2, 3, or 5, the user-configured preTokens and nextTokens do not take effect.
  - When configured as 0 or 4, ensure that attenMaskOptional is consistent with the ranges of preTokens and nextTokens.
  - If not explicitly specified, it is recommended to pass 0.
  - For details about different sparse modes, see [sparseMode Introduction](../../../docs/en/context/sparse_mode_introduction.md).
- In some scenarios, if the computation workload is too large, the operator execution may time out (aicore error with errorStr: timeout or trap error). In this case, axis splitting is recommended. Note: The computation workload here is affected by parameters such as B, S, N, and D. Larger values result in a heavier workload.
- In the band scenario, preTokens and nextTokens must have an intersection.
- In the prefixOptional sparse computation scenario (sparseMode = 5 or sparseMode = 6), when Sq > Skv, the N value range of prefix is [0, Skv]; when Sq <= Skv, the N value range of prefix is [Skv - Sq, Skv].
- When Sq in realShiftOptional is greater than 1024 and the shape is BNHS or 1NHS, Sq and Skv must have the same length.

## Example

The following sample code is for reference only. For details about compiling and running, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```C++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_flash_attention_score.h"

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

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // Boilerplate: initialize resources.
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
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy data from the host to the device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output as required by the API.
  std::vector<int64_t> qShape = {256, 1, 128};
  std::vector<int64_t> kShape = {256, 1, 128};
  std::vector<int64_t> vShape = {256, 1, 128};
  std::vector<int64_t> attenmaskShape = {256, 256};

  std::vector<int64_t> attentionOutShape = {256, 1, 128};
  std::vector<int64_t> softmaxMaxShape = {1, 1, 256, 8};
  std::vector<int64_t> softmaxSumShape = {1, 1, 256, 8};

  void* qDeviceAddr = nullptr;
  void* kDeviceAddr = nullptr;
  void* vDeviceAddr = nullptr;
  void* attenmaskDeviceAddr = nullptr;
  void* attentionOutDeviceAddr = nullptr;
  void* softmaxMaxDeviceAddr = nullptr;
  void* softmaxSumDeviceAddr = nullptr;

  aclTensor* q = nullptr;
  aclTensor* k = nullptr;
  aclTensor* v = nullptr;
  aclTensor* pse = nullptr;
  aclTensor* dropMask = nullptr;
  aclTensor* padding = nullptr;
  aclTensor* attenmask = nullptr;
  aclTensor* attentionOut = nullptr;
  aclTensor* softmaxMax = nullptr;
  aclTensor* softmaxSum = nullptr;
  aclTensor* softmaxOut = nullptr;

  std::vector<float> qHostData(32768, 1);
  std::vector<float> kHostData(32768, 1);
  std::vector<float> vHostData(32768, 1);
  std::vector<uint8_t> attenmaskHostData(65536, 0);
  std::vector<float> attentionOutHostData(32768, 0);
  std::vector<float> softmaxMaxHostData(2048, 3.0);
  std::vector<float> softmaxSumHostData(2048, 3.0);

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attenmaskHostData, attenmaskShape, &attenmaskDeviceAddr, aclDataType::ACL_UINT8, &attenmask);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attentionOutHostData, attentionOutShape, &attentionOutDeviceAddr, aclDataType::ACL_FLOAT, &attentionOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> prefixOp = {0};
  std::vector<int64_t> qStartIdxOp = {0};
  std::vector<int64_t> kvStartIdxOp = {0};
  aclIntArray *prefix = aclCreateIntArray(prefixOp.data(), 1);
  aclIntArray *qStartIdx = aclCreateIntArray(qStartIdxOp.data(), 1);
  aclIntArray *kvStartIdx = aclCreateIntArray(kvStartIdxOp.data(), 1);
  double scaleValue = 0.088388;
  double keepProb = 1;
  int64_t preTokens = 65536;
  int64_t nextTokens = 65536;
  int64_t headNum = 1;
  int64_t innerPrecise = 0;
  int64_t sparseMode = 0;
  int64_t pseType = 1;
  char layOut[5] = {'S', 'B', 'H', 0};

  // 3. Call the CANN operator library API. Replace the name with the actual API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnFlashAttentionScoreV2.
  ret = aclnnFlashAttentionScoreV2GetWorkspaceSize(
            q, k, v, pse, dropMask, padding, attenmask, prefix, qStartIdx, kvStartIdx, scaleValue,
            keepProb, preTokens, nextTokens, headNum, layOut, innerPrecise,
            sparseMode, pseType, softmaxMax, softmaxSum, softmaxOut, attentionOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionScoreV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnFlashAttentionScoreV2.
  ret = aclnnFlashAttentionScoreV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionScoreV2 failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the specific API definition.
  PrintOutResult(attentionOutShape, &attentionOutDeviceAddr);
  PrintOutResult(softmaxMaxShape, &softmaxMaxDeviceAddr);
  PrintOutResult(softmaxSumShape, &softmaxSumDeviceAddr);

  // 6. Release aclTensor and aclScalar. Modify the code based on the specific API definition.
  aclDestroyTensor(q);
  aclDestroyTensor(k);
  aclDestroyTensor(v);
  aclDestroyTensor(attenmask);
  aclDestroyTensor(attentionOut);
  aclDestroyTensor(softmaxMax);
  aclDestroyTensor(softmaxSum);

  // 7. Release device resources.
  aclrtFree(qDeviceAddr);
  aclrtFree(kDeviceAddr);
  aclrtFree(vDeviceAddr);
  aclrtFree(attenmaskDeviceAddr);
  aclrtFree(attentionOutDeviceAddr);
  aclrtFree(softmaxMaxDeviceAddr);
  aclrtFree(softmaxSumDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}

```
