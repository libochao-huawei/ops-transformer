# aclnnPromptFlashAttentionV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T12:31:15.693Z -->

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

- Description: FlashAttention operator for full-sequence inference. Compared with [aclnnPromptFlashAttention](aclnnPromptFlashAttention_en.md), this API additionally supports sparse optimization, actualSeqLengthsKv optimization, and int8 quantization.
- Formula:

    Self-attention builds an attention model using the relationships within the input samples themselves. The principle is as follows: assume there is an input sample sequence $x$ of length $n$, where each element of $x$ is a $d$-dimensional vector. Each $d$-dimensional vector can be regarded as a token embedding. After passing such a sequence through three weight matrices, three matrices of dimension $n*d$ are obtained.

    The calculation formula of self-attention is generally defined as follows, where $Q$, $K$, and $V$ are important property elements of the input sample, obtained through spatial transformation of the input sample and can be unified into a feature space. "Attention" in the formula and operator name is an abbreviation of "self-attention".

    $$
     Attention(Q,K,V)=Score(Q,K)V
    $$

    In this operator, the Score function uses the Softmax function, and the self-attention calculation formula is:

    $$
    Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
    $$

    Where: the product of $Q$ and $K^T$ represents the attention of the input $x$. To prevent this value from becoming too large, it is usually divided by the square root of $d$ for scaling, and each row is normalized by softmax. After multiplying with $V$, an $n*d$ matrix is obtained.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnPromptFlashAttentionV2GetWorkspaceSize" API to obtain the workspace size required for computation and the executor that includes the operator computation process, and then call the "aclnnPromptFlashAttentionV2" API to perform the computation.

```cpp
aclnnStatus aclnnPromptFlashAttentionV2GetWorkspaceSize(
    const aclTensor   *query,
    const aclTensor   *key,
    const aclTensor   *value,
    const aclTensor   *pseShift,
    const aclTensor   *attenMask,
    const aclIntArray *actualSeqLengths,
    const aclIntArray *actualSeqLengthsKv,
    const aclTensor   *deqScale1,
    const aclTensor   *quantScale1,
    const aclTensor   *deqScale2,
    const aclTensor   *quantScale2,
    const aclTensor   *quantOffset2,
    int64_t            numHeads,
    double             scaleValue,
    int64_t            preTokens,
    int64_t            nextTokens,
    char              *inputLayout,
    int64_t            numKeyValueHeads,
    int64_t            sparseMode,
    const aclTensor   *attentionOut,
    uint64_t          *workspaceSize,
    aclOpExecutor     **executor)
```

```cpp
aclnnStatus aclnnPromptFlashAttentionV2(
     void              *workspace,
     uint64_t           workspaceSize,
     aclOpExecutor     *executor,
     const aclrtStream  stream)
```

## aclnnPromptFlashAttentionV2GetWorkspaceSize

- **Parameters**

    <div style="overflow-x: auto;">
    <table style="undefined;table-layout: fixed; width: 1577px"><colgroup>
    <col style="width: 180px">
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
        <th>data type</th>
        <th>data format</th>
        <th>Dimension (shape)</th>
        <th>non-contiguous tensor</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>query</td>
        <td>Input</td>
        <td>Input Q in the formula.</td>
        <td>Keep consistent with the data type of key and value.</td>
        <td>FLOAT16, BFLOAT16, INT8</td>
        <td>ND</td>
        <td>3-4</td>
        <td>×</td>
    </tr>
    <tr>
        <td>key</td>
        <td>Input</td>
        <td>Input K in the formula.</td>
        <td>Keep consistent with the data types of query and value.</td>
        <td>FLOAT16, BFLOAT16, INT8</td>
        <td>ND</td>
        <td>3-4</td>
        <td>×</td>
    </tr>
    <tr>
        <td>value</td>
        <td>Input</td>
        <td>Input V in the formula.</td>
        <td>Keep consistent with the data types of query and key.</td>
        <td>FLOAT16, BFLOAT16, INT8</td>
        <td>ND</td>
        <td>3-4</td>
        <td>×</td>
    </tr>
    <tr>
        <td>pseShift</td>
        <td>Input</td>
        <td>Position encoding.</td>
        <td>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</td>
        <td>FLOAT16, BFLOAT16, nullptr</td>
        <td>ND</td>
        <td>4</td>
        <td>×</td>
    </tr>
    <tr>
        <td>attenMask</td>
        <td>Input</td>
        <td>Mask matrix</td>
        <td><ul><li>Do not use this function. Pass in nullptr.</li></ul>
            <ul><li>For Comprehensive Constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>BOOL, INT8, UINT8</td>
        <td>ND</td>
        <td>2-4</td>
        <td>×</td>
    </tr>
    <tr>
        <td>actualSeqLengths</td>
        <td>Input</td>
        <td>Effective sequence length of query in different batches.</td>
        <td><ul><li>Pass in nullptr if no sequence length is specified.</li></ul>
            <ul><li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
    </tr>
    <tr>
        <td>actualSeqLengthsKv</td>
        <td>Input</td>
        <td>Valid sequence length of key/value in different batches.</td>
        <td><ul><li>Pass in nullptr if the sequence length is not specified.</li></ul>
            <ul><li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
    </tr>
    <tr>
        <td>deqScale1</td>
        <td>Input</td>
        <td>Dequantization factor after BMM1.</td>
        <td><ul><li>Supports per-tensor.</li></ul>
            <ul><li>Pass in nullptr when this function is not used.</li></ul></td>
        <td>UINT64, FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
    </tr>
    <tr>
        <td>quantScale1</td>
        <td>Input</td>
        <td>Quantization factor before BMM2.</td>
        <td><ul><li>Supports per-tensor. </li></ul>
            <ul><li>Pass in nullptr when you do not use this function.</li></ul></td>
        <td>FLOAT32, nullptr</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
    </tr>
    <tr>
        <td>deqScale2</td>
        <td>Input</td>
        <td>Dequantization factor after BMM2.</td>
        <td><ul><li>Supports per-tensor. </li></ul>
            <ul><li>Pass in nullptr when this function is not used.</li></ul></td>
        <td>UINT64, FLOAT32, nullptr</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
    </tr>
    <tr>
        <td>quantScale2</td>
        <td>Input</td>
        <td>Quantization factor of the output.</td>
        <td><ul><li>Supports per-tensor and per-channel. </li></ul>
            <ul><li>Pass in nullptr when this function is not used.</li></ul></td>
        <td>FLOAT32, BFLOAT16, nullptr</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
    </tr>
    <tr>
        <td>quantOffset2</td>
        <td>Input</td>
        <td>Quantization offset of the output.</td>
        <td><ul><li>Supports per-tensor and per-channel. </li></ul>
            <ul><li>Pass in nullptr when this function is not used.</li></ul></td>
        <td>FLOAT32, BFLOAT16, nullptr</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
    </tr>
    <tr>
        <td>numHeads</td>
        <td>Input</td>
        <td>Head count of query.</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
    </tr>
    <tr>
        <td>scaleValue</td>
        <td>Input</td>
        <td>Reciprocal of the square root of d in the formula.</td>
        <td><ul><li>The data type and the data type of query must satisfy the data type deduction rules. </li></ul>
            <ul><li>If the user does not specifically specify it, it is recommended to pass in 1.0. </li></ul></td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
    </tr>
    <tr>
        <td>preTokens</td>
        <td>Input</td>
        <td>Indicates the number of preceding tokens that the attention computation needs to be associated with.</td>
        <td>If not specifically specified, it is recommended to pass in 2147483647.</td>
        <td>INT64</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
    </tr>
    <tr>
        <td>nextTokens</td>
        <td>Input</td>
        <td>Indicates the count of subsequent tokens that the attention computation needs to be associated with.</td>
        <td>If not specifically specified, it is recommended to pass in 0.</td>
        <td>INT64</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
    </tr>
    <tr>
        <td>inputLayout</td>
        <td>Input</td>
        <td>Indicates the data layout format of the input query, key, and value.</td>
        <td><ul><li>If not specifically specified, it is recommended to pass in "BSH".</li></ul>
            <ul><li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>STRING</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>numKeyValueHeads</td>
        <td>Input</td>
        <td>Head count in key and value.</td>
        <td><ul><li>If the user does not specifically specify it, it is recommended to pass in 0, indicating that the head count of key/value is equal to that of query.</li></ul>
            <ul><li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>sparseMode</td>
        <td>Input</td>
        <td>Sparse mode.</td>
        <td>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</td>
        <td>INT64</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
    </tr>
    <tr>
        <td>attentionOut</td>
        <td>Output</td>
        <td>The output in the formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16, INT8</td>
        <td>ND</td>
        <td>3-4</td>
        <td>-</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the workspace size that the user needs to allocate on the Device side.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Returns the op executor, which includes the calculation flow of the operator.</td>
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
      <td>Returns 161001 if a passed-in parameter is a required input, output, or required attribute and is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data types and data formats of query, key, value, pseShift, attenMask, and attentionOut are not within the supported range.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>An exception occurs when the API calls the npu runtime API for memory.</td>
    </tr>
  </tbody>
  </table>

## aclnnPromptFlashAttentionV2

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
      <td>Workspace memory address applied on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size applied on the Device side, obtained by the first-phase API aclnnPromptFlashAttentionV2GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which includes the calculation flow of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream that executes the task.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnPromptFlashAttentionV2 defaults to a deterministic implementation.
- When using this API with PyTorch, ensure that the versions of CANN-related packages and PyTorch-related packages are compatible.

- Handling of null input parameters: The operator internally checks whether the query parameter is null. If it is null, the operator returns directly. If the query parameter is a non-null tensor and the key and value parameters are null tensors (i.e., S2 is 0), attentionOut is filled with all zeros. When attentionOut is a null tensor, the AscendCLNN framework handles it. For other input parameters marked as "nullable" in the parameter descriptions above, no processing is performed when they are null pointers.

- The function usage restrictions for query, key, and value inputs are as follows:

  <!-- npu="910b" id32 -->
  - <term>Atlas A2 products</term>:

    - The B-axis must be less than or equal to 65536 (64k). When the input type includes INT8 and the D-axis is not 32-aligned, or when the input type is FLOAT16 or BFLOAT16 and the D-axis is not 16-aligned, the B-axis only supports up to 128.
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
    - The D axis must be less than or equal to 512. When inputLayout is BSH or BSND, N*D must be less than 65535.
  <!-- end id32 -->

  <!-- npu="310p" id31 -->
  - <term>Atlas inference products</term>:
      - When inputLayout is BSH, the B axis must be less than or equal to 300; in other cases, the B axis must be less than or equal to 128.
      - The N axis must be less than or equal to 256.
      - The S axis must be less than or equal to 65535 (64k). When Q_S or KV_S is not 128-aligned, or when Q_S and KV_S have unequal lengths, configuring atten_mask is not supported.
      - Supports D axis less than or equal to 512.
  <!-- end id31 -->

  - Input data type restrictions:

    <!-- npu="910b" id7 -->
    - <term>Atlas A2 products</term>: Supported data types are FLOAT16, BFLOAT16, and INT8
    <!-- end id7 -->
    <!-- npu="310p" id19 -->
    - <term>Atlas inference products</term>: only FLOAT16 is supported
    <!-- end id19 -->

- pseShift function usage restrictions as follows:

  - Reserved parameter, not yet in use. This is a device-side aclTensor whose data type must satisfy the data type deduction rules with the query data type. Currently, this parameter is forcibly set to nullptr.
  - Input data type restrictions:

    <!-- npu="910b" id8 -->
    - <term>Atlas A2 products</term>: Supported data types are FLOAT16 and BFLOAT16
    <!-- end id8 -->
    <!-- npu="310p" id20 -->
    - <term>Atlas inference products</term>: only nullptr is supported
    <!-- end id20 -->

- Restrictions on the attenMask function are as follows:

  - Input shape restrictions: If this function is not used, nullptr can be passed in. It is generally recommended that the shape be Q_S,KV_S; B,Q_S,KV_S; 1,Q_S,KV_S; B,1,Q_S,KV_S; 1,1,Q_S,KV_S, where Q_S is S in the query shape, and KV_S is S in the key and value shapes. For scenarios where KV_S of attenMask is not 32-aligned, it is recommended to pad to 32-alignment for better performance, with the excess portion filled with 1.
  - Input data type restrictions:

    <!-- npu="910b" id9 -->
    - <term>Atlas A2 products</term>: Supported data types are BOOL, INT8, and UINT8.
    <!-- end id9 -->
    <!-- npu="310p" id21 -->
    - <term>Atlas inference products</term>: only BOOL is supported.
    <!-- end id21 -->
  - When the data type of attenMask is INT8 or UINT8, the values in its tensor must be 0 or 1.

- actualSeqLengths and actualSeqLengthsKv inputs have the following function usage restrictions:

  - Input value range restrictions:
    - For actualSeqLengths, if the sequence length is not specified, nullptr can be passed, indicating that the valid sequence length is the same as the S length in the query shape. Note that the valid sequence length of each batch in this parameter must not exceed the sequence length of the corresponding batch in the query.
    - For actualSeqLengthsKv, if the sequence length is not specified, nullptr can be passed, indicating that the valid sequence length is the same as the S length in the key/value shape. Note that the valid sequence length of each batch in this parameter must not exceed the sequence length of the corresponding batch in the key/value.
  - Input attribute restrictions: The following rules apply to the input length of seqlen: When the input length is 1, all batches use the same seqlen; when the input length is greater than or equal to the number of batches, the first Batch values of seqlen are used; other input lengths are not supported.
  - Input data type restrictions:

    <!-- npu="910b" id10 -->
    - <term>Atlas A2 products</term>: Supported data type is INT64.
    <!-- end id10 -->
    <!-- npu="310p" id22 -->
    - <term>Atlas inference products</term>: Supported data type is INT64.
    <!-- end id22 -->

- deqScale1 and deqScale2 inputs, function usage restrictions as follows:

  - Input data type restrictions:

    <!-- npu="910b" id11 -->
    - <term>Atlas A2 products</term>: Supported data types are UINT64 and FLOAT32.
    <!-- end id11 -->
    <!-- npu="310p" id23 -->
    - <term>Atlas inference products</term>: only nullptr is supported.
    <!-- end id23 -->

- quantScale1 input, function usage restrictions as follows:

  - Input data type restrictions:

    <!-- npu="910b" id12 -->
    - <term>Atlas A2 products</term>: Supports the data type FLOAT32.
    <!-- end id12 -->
    <!-- npu="310p" id24 -->
    - <term>Atlas inference products</term>: Only supports nullptr.
    <!-- end id24 -->

- The restrictions on the use of the quantScale2 and quantOffset2 inputs are as follows:

  - Input data type restrictions:

    <!-- npu="910b" id13 -->
    - <term>Atlas A2 products</term>: Supports the data types FLOAT32 and BFLOAT16.
    <!-- end id13 -->
    <!-- npu="310p" id25 -->
    - <term>Atlas inference products</term>: Only supports nullptr.
    <!-- end id25 -->

- The preTokens input has the following function usage restrictions:

  - Input data type restrictions:

    <!-- npu="910b" id14 -->
    - <term>Atlas A2 products</term>: Supports the data type INT64.
    <!-- end id14 -->
    <!-- npu="310p" id26 -->
    - <term>Atlas inference products</term>: Only supports the value 2147483647.
    <!-- end id26 -->
- The nextTokens input has the following function usage restrictions:

  - Input data type restrictions:

    <!-- npu="910b" id15 -->
    - <term>Atlas A2 products</term>: Supports the data type INT64.
    <!-- end id15 -->
    <!-- npu="310p" id27 -->
    - <term>Atlas inference products</term>: Only supports values 0 and 2147483647.
    <!-- end id27 -->
- The inputLayout input has the following function usage restrictions:

  - Input data type restrictions:
    - Currently, BSH, BSND, BNSD, and BNSD_BSND are supported (when the input is BNSD, the output format is BSND). It is recommended to pass "BSH" when the user does not specify a format.
    - Description: The data layout formats of query, key, and value can be interpreted from multiple dimensions, where B (Batch) indicates the input sample batch size, S (Seq-Length) indicates the input sample sequence length, H (Head-Size) indicates the hidden layer size, N (Head-Num) indicates the number of heads, D (Head-Dim) indicates the minimum unit dimension of the hidden layer and satisfies D = H/N, and T indicates the cumulative sum of the sequence lengths of all batch input samples.

- Restrictions on the numKeyValueHeads input and function usage are as follows:

  - An int on the host side, representing the number of heads in key and value, used to support GQA (Grouped-Query Attention) scenarios. It is recommended to pass 0 when the user does not specify a value, indicating that the number of heads in key/value and query are equal. Constraints: numHeads must be divisible by numKeyValueHeads, and in BSND, BNSD, and BNSD_BSND scenarios, the value must match the N-axis shape value of key/value in the shape; otherwise, an error is reported.
  - Input data type restrictions:

    <!-- npu="910b" id16 -->
    - <term>Atlas A2 products</term>: Supports the data type INT64.
    <!-- end id16 -->
    <!-- npu="310p" id28 -->
    - <term>Atlas inference products</term>: Only supports the value 0.
    <!-- end id28 -->

- sparseMode input, function usage restrictions as follows:

  <!-- npu="910b" id17 -->
  - <term>Atlas A2 products</term>:
    - When sparseMode is 0, it represents the defaultMask mode. If attenmask is not passed, no mask operation is performed, and preTokens and nextTokens are ignored (internally assigned to INT_MAX). If attenmask is passed, the complete attenmask matrix (S1 * S2) must be provided, indicating that the portion between preTokens and nextTokens needs to be computed.
    - When sparseMode is 1, it represents allMask, and the complete attenmask matrix (S1 * S2) must be passed in.
    - When sparseMode is 2, it represents the leftUpCausal mode mask, and the optimized attenmask matrix (2048*2048) must be passed in.
    - When sparseMode is 3, it represents the rightDownCausal mode mask, corresponding to the lower triangular scenario partitioned by the right vertex, and the optimized attenmask matrix (2048*2048) must be passed in.
    - When sparseMode is 4, it represents the band mode mask, and the optimized attenmask matrix (2048*2048) must be passed in.
    - When sparseMode is 5, 6, 7, or 8, they represent prefix, global, dilated, and block_local respectively, and **none of them are currently supported**. If the user does not specify a value, 0 is recommended.

  <!-- end id17 -->
  <!-- npu="310p" id29 -->
  - <term>Atlas inference products</term>: Only supports the value 0
  <!-- end id29 -->

- The attentionOut output has the following function usage restrictions:

  - Shape restrictions: When inputLayout is BNSD_BSND, the shape of the input query is BNSD and the output shape is BSND; in other cases, the shape of this input parameter must be consistent with the shape of the input query.
  - Type restrictions:

    <!-- npu="910b" id18 -->
    - <term>Atlas A2 products</term>: The supported data types are FLOAT16, BFLOAT16, and INT8.
    <!-- end id18 -->
    <!-- npu="310p" id30 -->
    - <term>Atlas inference products</term>: Only supports FLOAT16.
    <!-- end id30 -->

- Other constraints:
  - Comprehensive constraints on the number of int8 quantization-related input parameters and the input/output data formats:
    - When both input and output are INT8: The input parameters deqScale1, quantScale1, deqScale2, and quantScale2 must all be present. quantOffset2 is optional and defaults to 0 if not passed.
    - For the scenario where the input is INT8 and the output is FLOAT16: the input parameters deqScale1, quantScale1, and deqScale2 must be passed in at the same time. If the input parameter quantOffset2 or quantScale2 exists (that is, not nullptr), an error is reported and the API returns.
    - For the scenario where the input is FLOAT16 or BFLOAT16 and the output is INT8: the input parameter quantScale2 must be passed in, and quantOffset2 is optional and defaults to 0 if not passed in. If the input parameter deqScale1, quantScale1, or deqScale2 exists (that is, not nullptr), an error is reported and the API returns.
    - The input parameters quantScale2 and quantOffset2 support the per-tensor/per-channel formats and the FLOAT32/BFLOAT16 data types. If quantOffset2 is passed in, its type and shape information must be consistent with those of quantScale2. When the input is BFLOAT16, both FLOAT32 and BFLOAT16 are supported; otherwise, only FLOAT32 is supported. In the per-channel format, if the output layout is BSH, the product of all dimensions of quantScale2 must be equal to H; for other layouts, the product must be equal to N*D. (When the output layout is BSH, it is recommended to pass in a shape of [1,1,H] or [H] for quantScale2; when the output is BNSD, it is recommended to pass in [1,N,1,D] or [N,D]; when the output is BSND, it is recommended to pass in [1,1,N,D] or [N,D])
    - When the output is int8 and quantScale2 and quantOffset2 are per-channel, left padding, Ring Attention, or non-32Byte-aligned D is not supported yet.
    - When the output is INT8, the scenario where sparse is band and preTokens/nextTokens are negative is not supported.
  - When the output is INT8, the input parameter quantOffset2 is passed in with a non-null pointer and a non-empty tensor value, and sparseMode, preTokens, and nextTokens meet the following conditions, some rows of the matrix do not participate in the computation, causing errors in the computation result. This scenario is intercepted (solution: if you want this scenario not to be intercepted, perform post-quantization outside the PFA API instead of enabling it inside the PFA API):
    - sparseMode = 0: if attenMask is a non-null pointer, the interception condition is met when actualSeqLengths - actualSeqLengthsKV - preTokens > 0 or nextTokens < 0 for each batch.
    - For sparseMode = 1 or 2, the interception condition is never satisfied.
    - When sparseMode = 3, if each batch satisfies actualSeqLengthsKV - actualSeqLengths < 0, the interception condition is met.
    - For sparseMode = 4, the interception condition is satisfied when preTokens < 0 or nextTokens + actualSeqLengthsKV - actualSeqLengths < 0 for each batch.

## Example

The following sample code is for reference only. For details about how to compile and run the code, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <iostream>
#include <vector>
#include <math.h>
#include <cstring>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_prompt_flash_attention_v2.h"

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
  // Fixed format, AscendCL initialization
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
  auto size = GetShapeSize(shape) * aclDataTypeSize(dataType);
  // Call aclrtMalloc to request device side memory
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host side data to device side memory
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of continuous tensor
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor interface to create aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Fixed format) Device/stream initialization, refer to AscendCL external interface list
  // Fill in the deviceId based on your actual device
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. To construct input and output, it is necessary to customize the construction according to the API interface
  std::vector<int64_t> queryShape = {1, 2, 1, 16}; // BNSD
  std::vector<int64_t> keyShape = {1, 2, 2, 16}; // BNSD
  std::vector<int64_t> valueShape = {1, 2, 2, 16}; // BNSD
  std::vector<int64_t> attenShape = {1, 1, 1, 2}; // B 1 S1 S2
  std::vector<int64_t> outShape = {1, 2, 1, 16}; // BNSD
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
  int64_t queryShapeSize = GetShapeSize(queryShape); // BNSD
  int64_t keyShapeSize = GetShapeSize(keyShape); // BNSD
  int64_t valueShapeSize = GetShapeSize(valueShape); // BNSD
  int64_t attenShapeSize = GetShapeSize(attenShape); // B 1 S1 S2
  int64_t outShapeSize = GetShapeSize(outShape); // BNSD
  std::vector<op::fp16_t> queryHostData(queryShapeSize, 1);
  std::vector<op::fp16_t> keyHostData(keyShapeSize, 1);
  std::vector<op::fp16_t> valueHostData(valueShapeSize, 1);
  std::vector<int8_t> attenHostData(attenShapeSize, 1);
  std::vector<op::fp16_t> outHostData(outShapeSize, 1);

  // Create query aclTensor
  ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create key aclTensor
  ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &keyTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create value aclTensor
  ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &valueTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create atten aclTensor
  ret = CreateAclTensor(attenHostData, attenShape, &attenDeviceAddr, aclDataType::ACL_BOOL, &attenTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &outTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> actualSeqlenVector = {2};
  auto actualSeqLengths = aclCreateIntArray(actualSeqlenVector.data(), actualSeqlenVector.size());
  int64_t numHeads=2; // N
  int64_t numKeyValueHeads = numHeads;
  double scaleValue= 1 / sqrt(2); // 1/sqrt(d)
  int64_t preTokens = 2147483647;
  int64_t nextTokens = 0;
  char layerOut[] = "BNSD";
  int64_t sparseMode = 0;
  // 3. Call the CANN operator library API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first interface
  ret = aclnnPromptFlashAttentionV2GetWorkspaceSize(queryTensor, keyTensor, valueTensor, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    numHeads, scaleValue, preTokens, nextTokens, layerOut, numKeyValueHeads, sparseMode, outTensor, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPromptFlashAttentionV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated from the first interface
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second interface
  ret = aclnnPromptFlashAttentionV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPromptFlashAttentionV2 failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed format) Synchronize and wait for the completion of task execution
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value, copy the result from the device side memory to the host side, and modify it according to the specific API interface definition
  auto size = GetShapeSize(outShape);
  std::vector<op::fp16_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

  // 6. Release resources
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
