# aclnnPromptFlashAttention

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T12:17:19.736Z -->

**Notice: This API will be deprecated in a later version. Use the latest API [aclnnPromptFlashAttentionV3](./aclnnPromptFlashAttentionV3_en.md) instead.**

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

- Description: FlashAttention operator for full inference use cases.

- Formula:

    Self-attention builds an attention model using the relationships within the input samples themselves. The principle is as follows: assume there is an input sample sequence $x$ of length $n$, where each element of $x$ is a $d$-dimensional vector. Each $d$-dimensional vector can be regarded as a token embedding. After passing such a sequence through three weight matrices, three matrices of dimension $n*d$ are obtained.

    The calculation formula of self-attention is generally defined as follows, where $Q, K, V$ are important attribute elements of the input samples, obtained through spatial transformation of the input samples and can be unified into one feature space. "Attention" in the formula and the operator name is short for "self-attention".
    $$
     Attention(Q,K,V)=Score(Q,K)V
    $$

    In this operator, the Score function uses the Softmax function, and the self-attention calculation formula is:

    $$
    Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
    $$

    Where: the product of $Q$ and $K^T$ represents the attention of input $x$. To prevent this value from becoming too large, it is usually scaled by dividing by the square root of $d$, and softmax normalization is performed on each row. Multiplying with $V$ yields a n\*d matrix.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnPromptFlashAttentionGetWorkspaceSize" first to obtain the workspace size required for the computation and the executor containing the operator computation process, and then call "aclnnPromptFlashAttention" to execute the computation.

```cpp
aclnnStatus aclnnPromptFlashAttentionGetWorkspaceSize(
    const aclTensor   *query,
    const aclTensor   *key,
    const aclTensor   *value,
    const aclTensor   *pseShift,
    const aclTensor   *attenMask,
    const aclIntArray *actualSeqLengths,
    int64_t            numHeads,
    double             scaleValue,
    int64_t            preTokens,
    int64_t            nextTokens,
    char             *inputLayout,
    int64_t           numKeyValueHeads,
    const aclTensor  *attentionOut,
    uint64_t         *workspaceSize,
    aclOpExecutor    **executor)
```

```cpp
aclnnStatus aclnnPromptFlashAttention(
    void              *workspace,
    uint64_t           workspaceSize,
    aclOpExecutor     *executor,
    const aclrtStream  stream)
```

## aclnnPromptFlashAttentionGetWorkspaceSize

- **Parameters**

  <div style="overflow-x: auto;">
   <table style="undefined;table-layout: fixed; width: 1567px"><colgroup>
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
  </tr></thead>
  <tbody>
  <tr>
    <td>query</td>
    <td>Input</td>
    <td>The input Q in the formula.</td>
    <td>Keep consistent with the data types of key and value.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
  </tr>
  <tr>
    <td>key</td>
    <td>Input</td>
    <td>Input K in the formula.</td>
    <td>Keep the data type consistent with that of query and value.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
  </tr>
  <tr>
    <td>value</td>
    <td>Input</td>
    <td>Input V in the formula.</td>
    <td>Keep the data type consistent with that of query and key.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>×</td>
  </tr>
  <tr>
    <td>pseShift</td>
    <td>Input</td>
    <td>Position encoding.</td>
    <td>Reserved parameter, currently unused. This parameter is forcibly set to nullptr.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>4</td>
    <td>×</td>
  </tr>
  <tr>
    <td>attenMask</td>
    <td>Input</td>
    <td>Mask matrix.</td>
    <td><ul><li>Pass nullptr if this function is not used.</li></ul>
        <ul><li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
    <td>BOOL, INT8, UINT8</td>
    <td>ND</td>
    <td>2-4</td>
    <td>×</td>
  </tr>
  <tr>
    <td>actualSeqLengths</td>
    <td>Input</td>
    <td>Valid sequence lengths of query in different batches.</td>
    <td><ul><li>Pass nullptr if sequence lengths are not specified.</li></ul>
        <ul><li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
    <td>INT64</td>
    <td>ND</td>
    <td>1</td>
    <td>-</td>
  </tr>
  <tr>
    <td>numHeads</td>
    <td>Input</td>
    <td>Number of heads of query.</td>
    <td>Restrictions: In BNSD/NSD use cases, the value must be the same as the N-axis shape value of query in shape.</td>
    <td>INT64</td>
    <td>ND</td>
    <td>1</td>
    <td>-</td>
  </tr>
  <tr>
    <td>scaleValue</td>
    <td>Input</td>
    <td>The reciprocal of the square root of d in the formula.</td>
    <td><ul><li>The data type of this parameter and that of query must comply with the data type deduction rules. </li></ul>
        <ul><li>1.0 is recommended if not specified. </li></ul></td>
    <td>DOUBLE</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
  </tr>
  <tr>
    <td>preTokens</td>
    <td>Input</td>
    <td>Indicates that attention is computed in association with the preceding tokens.</td>
    <td><ul><li>If not specifically specified, 2147483647 is recommended.</li></ul>
       <ul><li>Negative values are supported.</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
  </tr>
  <tr>
    <td>nextTokens</td>
    <td>Input</td>
    <td>Indicates the number of subsequent tokens that attention computation needs to be associated with.</td>
    <td><ul><li>If not specified, 0 is recommended.</li></ul>
       <ul><li>Negative values are supported.</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
  </tr>
  <tr>
    <td>inputLayout</td>
    <td>Input</td>
    <td>Indicates the data layout format of the input query, key, and value.</td>
    <td><ul><li>If not specified, "BSH" is recommended.</li></ul>
        <ul><li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>numKeyValueHeads</td>
    <td>Input</td>
    <td>Number of heads in key and value.</td>
    <td><ul><li>It is recommended to pass in 0 if not specifically specified, indicating that key/value and query have the same number of heads.</li></ul>
        <ul><li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>attentionOut</td>
    <td>Output</td>
    <td>The output in the formula.</td>
    <td>-</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>3-4</td>
    <td>-</td>
  </tr>
  <tr>
    <td>workspaceSize</td>
    <td>Output</td>
    <td>Returns the workspace size that the user needs to allocate on the Device.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
  </tr>
  <tr>
    <td>executor</td>
    <td>Output</td>
    <td>Returns the op executor, which contains the operator calculation flow.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>1</td>
    <td>-</td>
  </tr>
  </tbody></table>
  </div>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation. If the following error codes occur, the corresponding causes are:

  <table style="undefined;table-layout: fixed; width: 1152px"><colgroup>
  <col style="width: 302px">
  <col style="width: 119px">
  <col style="width: 731px">
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
      <td>Returns 161001 if a required input, output, or required attribute is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data types and data formats of query, key, value, pseShift, attenMask, and attentionOut are out of the supported range.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>An exception occurs when the API calls the npu runtime API for memory.</td>
    </tr>
  </tbody>
  </table>

## aclnnPromptFlashAttention

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1143px"><colgroup>
  <col style="width: 178px">
  <col style="width: 134px">
  <col style="width: 831px">
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
      <td>Pointer to the workspace memory requested on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace memory requested on the Device, obtained by calling the first-phase API aclnnPromptFlashAttentionGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, which contains the computation flow of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnPromptFlashAttention defaults to a deterministic implementation.
- When using this API with PyTorch, ensure that the versions of CANN-related packages and PyTorch-related packages are compatible.
- Handling of empty input parameters: The operator internally checks whether query is empty and returns directly if it is empty. If query is not an empty tensor but key and value are empty tensors (that is, S2 is 0), attentionOut is filled with all zeros. If attentionOut is an empty tensor, the AscendCLNN framework processes it. For other input parameters marked as "can be nullptr" in the preceding parameter descriptions, null pointers are not processed.
- The data layout format of query, key, and value supports interpretation from multiple dimensions, where B (Batch) indicates the input sample batch size, S (Seq-Length) indicates the input sample sequence length, H (Head-Size) indicates the hidden layer size, N (Head-Num) indicates the number of heads, D (Head-Dim) indicates the minimum unit dimension of the hidden layer, and D = H/N is satisfied.

- The function usage restrictions for query, key, and value inputs are as follows:

  - Input shape constraints:

    <!-- npu="910b" id24 -->
    - <term>Atlas A2 products</term>:

      - The B axis must be less than or equal to 65536 (64k). If the input type includes INT8 and the D axis is not aligned to 32, or the input type is FLOAT16 or BFLOAT16 and the D axis is not aligned to 16, the B axis supports only up to 128.

      - The N axis must be less than or equal to 256.

      - S must be less than or equal to 20971520 (20M). In some long sequence scenarios, if the computation load is too large, the PFA operator may time out (aicore error type, with errorStr: timeout or trap error). In such scenarios, S-axis splitting is recommended. Note: The computation load is affected by B, S, N, D, etc. Larger values result in greater computation load. Typical long sequence scenarios that may cause timeouts (i.e., where the product of B, S, N, and D is large) include but are not limited to:
        <table style="undefined;table-layout: fixed; width: 600px"><colgroup>
        <col style="width: 100px">
        <col style="width: 100px">
        <col style="width: 200px">
        <col style="width: 100px">
        <col style="width: 100px">
        <col style="width: 200px">
        </colgroup><thead>
        <tr>
        <th>B</th>
        <th>Q_N</th>
        <th>Q_S</th>
        <th>D</th>
        <th>KV_N</th>
        <th>KV_S</th>
        </tr></thead>
        <tbody>
        <tr>
        <td>1</td>
        <td>20</td>
        <td>2097152</td>
        <td>256</td>
        <td>1</td>
        <td>2097152</td>
        </tr>
        <tr>
        <td>1</td>
        <td>2</td>
        <td>20971520</td>
        <td>256</td>
        <td>2</td>
        <td>20971520</td>
        </tr>
        <tr>
        <td>20</td>
        <td>1</td>
        <td>2097152</td>
        <td>256</td>
        <td>1</td>
        <td>2097152</td>
        </tr>
        <tr>
        <td>1</td>
        <td>10</td>
        <td>2097152</td>
        <td>512</td>
        <td>1</td>
        <td>2097152</td>
        </tr>
        </tbody>
        </table>
    <!-- end id24 -->
      - The D axis must be less than or equal to 512. When inputLayout is BSH or BSND, N*D must be less than 65535.
    <!-- npu="910b" id7 -->
    - <term>Atlas A2 products</term>: Comprehensive restrictions on query, key, and value inputs in TND scenarios:
        - T is less than or equal to 65536;
        - N is 8/16/32/64/128, and Q_N, K_N, and V_N are the same;
        - Q_D and K_D are 192, and V_D is 128/192;
        - Only the BFLOAT16 data type is supported;
        - The sparse mode supports only sparse=0 without mask or sparse=3 with mask;
        - When sparse=3, actualSeqLengths of each batch must be less than actualSeqLengthsKv.
    <!-- end id7 -->
    - <term>Atlas inference products</term>:
        - When inputLayout is BSH, the B axis must be less than or equal to 300; in other cases, the B axis must be less than or equal to 128.
        - The N axis must be less than or equal to 256.
        - The S axis must be less than or equal to 65535 (64k). When Q_S or KV_S is not 128-aligned, or when Q_S and KV_S have unequal lengths, configuring atten_mask is not supported.
        - Supports D axis less than or equal to 512.
  - Input data type restrictions:

    <!-- npu="910b" id8 -->
    - <term>Atlas A2 products</term>: Supports the FLOAT16 and BFLOAT16 data types.
    <!-- end id8 -->

    <!-- npu="310p" id16 -->
    - <term>Atlas inference products</term>: Supports only the FLOAT16 data type.
    <!-- end id16 -->
- pseShift function usage restrictions as follows:
  - Input data type restrictions:

    <!-- npu="910b" id9 -->
    - <term>Atlas A2 products</term>: Supports the FLOAT16 and BFLOAT16 data types.
    <!-- end id9 -->
    <!-- npu="310p" id17 -->
    - <term>Atlas inference products</term>: Supports only nullptr.
    <!-- end id17 -->
- Restrictions on the attenMask function are as follows:
  - Input shape restrictions: Typically, the recommended input shapes are Q_S, KV_S; B, Q_S, KV_S; 1, Q_S, KV_S; B, 1, Q_S, KV_S; and 1, 1, Q_S, KV_S, where Q_S is the S of the query shape, and KV_S is the S of the key and value shapes.
  - Input data type restrictions:

    <!-- npu="910b" id10 -->
    - <term>Atlas A2 products</term>: Supports the BOOL, INT8, and UINT8 data types.
    <!-- end id10 -->
    <!-- npu="310p" id18 -->
    - <term>Atlas inference products</term>: Supports only the BOOL data type.
    <!-- end id18 -->
  - Other restrictions: If the KV_S of attenMask is not 32-aligned, pad it to 32 alignment to improve performance, and fill the extra elements with 1.
- For the actualSeqLengths input, function usage restrictions are as follows:
  - Input value range restriction: The valid Sequence Length of each batch in this input should not be greater than the Sequence Length of the corresponding batch in query.
  - Input data type restrictions:

    <!-- npu="910b" id11 -->
    - <term>Atlas A2 products</term>: The data type supports INT64.
    <!-- end id11 -->
    <!-- npu="310p" id19 -->
    - <term>Atlas inference products</term>: The data type supports INT64.
    <!-- end id19 -->
- The preTokens input has the following function usage restrictions:
  - Input data type restrictions:

    <!-- npu="910b" id12 -->
    - <term>Atlas A2 products</term>: The data type supports INT64.
    <!-- end id12 -->
    <!-- npu="310p" id20 -->
    - <term>Atlas inference products</term>: Only the value 2147483647 is supported.
    <!-- end id20 -->
- The nextTokens input has the following function usage restrictions:
  - Input data type restrictions:

    <!-- npu="910b" id13 -->
    - <term>Atlas A2 products</term>: The data type supports INT64.
    <!-- end id13 -->
    <!-- npu="310p" id21 -->
    - <term>Atlas inference products</term>: Only the values 0 and 2147483647 are supported.
    <!-- end id21 -->
- The inputLayout input has the following function usage restrictions:
  - Input data type restrictions:
    - Currently, BSH, BSND, BNSD, and BNSD_BSND are supported (when the input is BNSD, the output format is BSND). It is recommended to pass "BSH" when the user does not specify a format.
- Restrictions on the numKeyValueHeads input and function usage are as follows:
  - Input attribute restrictions: numHeads must be divisible by numKeyValueHeads, and in BSND, BNSD, and BNSD_BSND use cases, it must be the same as the N-axis shape value of key/value in shape. Otherwise, an error is reported.
  - Input data type restrictions:

    <!-- npu="910b" id14 -->
    - <term>Atlas A2 products</term>: The data type supports INT64.
    <!-- end id14 -->
    <!-- npu="310p" id22 -->
    - <term>Atlas inference products</term>: Only the value 0 is supported.
    <!-- end id22 -->
- The attentionOut output has the following function usage restrictions:
  - Shape restrictions: When inputLayout is BNSD_BSND, the shape of the input query is BNSD and the output shape is BSND; in other cases, the shape of this input parameter must be consistent with the shape of the input query.
  - Type restrictions:

    <!-- npu="910b" id15 -->
    - <term>Atlas A2 products</term>: The data types support FLOAT16 and BFLOAT16.
    <!-- end id15 -->
    <!-- npu="310p" id23 -->
    - <term>Atlas inference products</term>: Supports FLOAT16 only.
    <!-- end id23 -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <iostream>
#include <vector>
#include <math.h>
#include <cstring>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_prompt_flash_attention.h"

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
  // Boilerplate for resource initialization.
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
  // 1. (Boilerplate) Initialize the device/stream. For details, see the public AscendCL APIs.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Customize the construction as required by the API.
  int32_t batchSize = 1;
  int32_t numHeads = 2;
  int32_t sequenceLengthQ = 1;
  int32_t headDims = 16;
  int32_t keyNumHeads = 2;
  int32_t sequenceLengthKV = 16;
  std::vector<int64_t> queryShape = {batchSize, numHeads, sequenceLengthQ, headDims}; // BNSD
  std::vector<int64_t> keyShape = {batchSize, keyNumHeads, sequenceLengthKV, headDims}; // BNSD
  std::vector<int64_t> valueShape = {batchSize, keyNumHeads, sequenceLengthKV, headDims}; // BNSD
  std::vector<int64_t> attenShape = {batchSize, 1, 1, sequenceLengthKV}; // B11S
  std::vector<int64_t> outShape = {batchSize, numHeads, sequenceLengthQ, headDims}; // BNSD
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
  std::vector<op::fp16_t> queryHostData(batchSize * numHeads * sequenceLengthQ * headDims, 1.0f);
  std::vector<op::fp16_t> keyHostData(batchSize * keyNumHeads * sequenceLengthKV * headDims, 1.0f);
  std::vector<op::fp16_t> valueHostData(batchSize * keyNumHeads * sequenceLengthKV * headDims, 1.0f);
  std::vector<int8_t> attenHostData(batchSize * sequenceLengthKV, 0);
  std::vector<op::fp16_t> outHostData(batchSize * numHeads * sequenceLengthQ * headDims, 1.0f);

  // Create the query aclTensor.
  ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the key aclTensor.
  ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &keyTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the value aclTensor.
  ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &valueTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the atten aclTensor.
  ret = CreateAclTensor(attenHostData, attenShape, &attenDeviceAddr, aclDataType::ACL_BOOL, &attenTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &outTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> actualSeqlenVector = {sequenceLengthKV};
  auto actualSeqLengths = aclCreateIntArray(actualSeqlenVector.data(), actualSeqlenVector.size());

  int64_t numKeyValueHeads = numHeads;
  double scaleValue = 1 / sqrt(headDims); // 1/sqrt(d)
  int64_t preTokens = 2147483647;
  int64_t nextTokens = 0;
  string sLayerOut = "BNSD";
  char layerOut[sLayerOut.length()+1];
  strcpy(layerOut, sLayerOut.c_str());
  // 3. Call the CANN operator library API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API.
  ret = aclnnPromptFlashAttentionGetWorkspaceSize(queryTensor, keyTensor, valueTensor, nullptr, nullptr, nullptr, numHeads, scaleValue,
                                                  preTokens, nextTokens, layerOut, numKeyValueHeads, outTensor, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPromptFlashAttentionGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API.
  ret = aclnnPromptFlashAttention(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPromptFlashAttention failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host memory. Modify it based on the specific API definition.
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
