# aclnnPromptFlashAttentionV3

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T12:27:06.035Z -->

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

- Description: FlashAttention operator for the prompt inference task. Supports sparse optimization, actualSeqLengthsKv optimization, int8 quantization, and the innerPrecise parameter (for selecting the high-precision or high-performance mode).

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

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnPromptFlashAttentionV3GetWorkspaceSize" API to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the "aclnnPromptFlashAttentionV3" API to execute the computation.

```cpp
aclnnStatus aclnnPromptFlashAttentionV3GetWorkspaceSize(
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
    int64_t            innerPrecise,
    const aclTensor   *attentionOut,
    uint64_t          *workspaceSize,
    aclOpExecutor     **executor)
```

```cpp
aclnnStatus aclnnPromptFlashAttentionV3(
     void              *workspace,
     uint64_t           workspaceSize,
     aclOpExecutor     *executor,
     const aclrtStream  stream)
```

## aclnnPromptFlashAttentionV3GetWorkspaceSize

- **Parameters**

    <div style="overflow-x: auto;">
    <table style="undefined;table-layout: fixed; width: 1567px"><colgroup>
     <col style="width: 180px">
     <col style="width: 120px">
     <col style="width: 300px">
     <col style="width: 330px">
     <col style="width: 252px">
     <col style="width: 100px">
     <col style="width: 140px">
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
        <td>Keep data type consistent with key and value.</td>
        <td>FLOAT16, BFLOAT16, INT8</td>
        <td>ND</td>
        <td>3-4</td>
        <td>×</td>
      </tr>
      <tr>
        <td>key</td>
        <td>Input</td>
        <td>Input K in the formula.</td>
        <td>Keep Consistent With the data types of query and value.</td>
        <td>FLOAT16, BFLOAT16, INT8</td>
        <td>ND</td>
        <td>3-4</td>
        <td>×</td>
      </tr>
      <tr>
        <td>value</td>
        <td>Input</td>
        <td>Input V in the formula.</td>
        <td>Keep Consistent With the data types of query and key.</td>
        <td>FLOAT16, BFLOAT16, INT8</td>
        <td>ND</td>
        <td>3-4</td>
        <td>×</td>
      </tr>
      <tr>
        <td>pseShift</td>
        <td>Input</td>
        <td>Position encoding.</td>
        <td><ul><li>Can be set to nullptr if this feature is not used.</li></ul>
            <ul><li>See <a href="#constraints">Constraints</a> for comprehensive constraints.</li></ul></td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>4</td>
        <td>×</td>
      </tr>
      <tr>
        <td>attenMask</td>
        <td>Input</td>
        <td>Mask matrix.</td>
        <td><ul><li>Input nullptr if this function is not used.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>BOOL, INT8, UINT8</td>
        <td>ND</td>
        <td>2-4</td>
        <td>×</td>
      </tr>
      <tr>
        <td>actualSeqLengths</td>
        <td>Input</td>
        <td>Effective sequence length of the query in each batch.</td>
        <td><ul><li>Input nullptr if the sequence length is not specified.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actualSeqLengthsKv</td>
        <td>Input</td>
        <td>Effective sequence length of key/value in different batches.</td>
        <td><ul><li>If the sequence length is not specified, nullptr can be passed.</li></ul>
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
              <ul><li>If this feature is not used, nullptr can be passed.</li></ul></td>
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
            <ul><li>Set to nullptr if not used.</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>deqScale2</td>
        <td>Input</td>
        <td>Dequantization factor after BMM2.</td>
        <td><ul><li>Supports per-tensor. </li></ul>
            <ul><li>Set to nullptr if this feature is not used.</li></ul></td>
        <td>UINT64, FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantScale2</td>
        <td>Input</td>
        <td>Quantization factor of the output.</td>
        <td><ul><li>Supports per-tensor, per-channel. </li></ul>
            <ul><li>Input nullptr if this function is not used.</li></ul></td>
        <td>FLOAT32, BFLOAT16</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantOffset2</td>
        <td>Input</td>
        <td>Quantization offset of the output.</td>
        <td><ul><li>Supports per-tensor and per-channel. </li></ul>
            <ul><li>Input nullptr when this feature is not used.</li></ul></td>
        <td>FLOAT32, BFLOAT16</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>numHeads</td>
        <td>Input</td>
        <td>Number of query heads.</td>
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
        <td><ul><li>The data type must comply with the data type deduction rule with the data type of query. </li></ul>
            <ul><li>1.0 is recommended if not specifically specified by the user. </li></ul></td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>preTokens</td>
        <td>Input</td>
        <td>Indicates the number of preceding Tokens that attention needs to compute with.</td>
        <td>If not specifically specified, the recommended input is 2147483647.</td>
        <td>INT64</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>nextTokens</td>
        <td>Input</td>
        <td>Indicates the number of subsequent tokens involved in attention computation.</td>
        <td>If not specified, the recommended input is 0.</td>
        <td>INT64</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>inputLayout</td>
        <td>Input</td>
        <td>Indicates the data layout format of input query, key, and value.</td>
        <td><ul><li>"BSH" is the recommended input if not specifically specified.</li></ul>
            <ul><li>For Comprehensive Constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>STRING</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>numKeyValueHeads</td>
        <td>Input</td>
        <td>Number of heads in key and value.</td>
        <td><ul><li>0 is the recommended input if not specified by the user, indicating that the number of heads in key/value is equal to that of query.</li></ul>
            <ul><li>For Comprehensive Constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
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
        <td>innerPrecise</td>
        <td>Input</td>
        <td>Indicates high-precision or high-performance selection.</td>
        <td>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</td>
        <td>INT8</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>attentionOut</td>
        <td>Output</td>
        <td>Output in the formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16, INT8</td>
        <td>ND</td>
        <td>3-4</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the workspace size to be allocated on the Device by the user.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Returns the op executor, which contains the operator computation process.</td>
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

  The first-phase API validates input parameters. If the following error codes occur, the corresponding causes are:

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
      <td>If the input parameter is a required input, output, or required attribute and is a null pointer, 161001 is returned.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data types and data formats of query, key, value, pseShift, attenMask, and attentionOut are not in the supported range.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>An exception occurs when the API calls the npu runtime API for memory operations.</td>
    </tr>
  </tbody>
  </table>

## aclnnPromptFlashAttentionV3

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
      <td>Workspace memory address allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the Device side, obtained by calling the first-phase API aclnnPromptFlashAttentionV3GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream that executes tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnPromptFlashAttentionV3 defaults to a deterministic implementation.
- When using this API with PyTorch, ensure that the versions of CANN-related packages and PyTorch-related packages are compatible.

- Handling of null input parameters: The operator internally checks whether the query parameter is null. If it is null, the operator returns directly. If the query parameter is a non-null tensor and the key and value parameters are null tensors (i.e., S2 is 0), attentionOut is filled with all zeros. When attentionOut is a null tensor, the AscendCLNN framework handles it. For other input parameters marked as "nullable" in the parameter descriptions above, no processing is performed when they are null pointers.

- The query, key, and value data layouts can be interpreted from multiple dimensions, where B (Batch) indicates the input sample batch size, S (Seq-Length) indicates the input sample sequence length, H (Head-Size) indicates the hidden layer size, N (Head-Num) indicates the number of heads, D (Head-Dim) indicates the smallest unit size of the hidden layer with D=H/N, and T indicates the accumulated sequence length of input samples across all batches.

- The function usage restrictions for query, key, and value inputs are as follows:

  <!-- npu="910b" id7 -->
  - <term>Atlas A2 products</term>:

    - Supports B axis less than or equal to 65536 (64k). If the input type includes INT8 and the D axis is not 32-aligned, or the input type is FLOAT16 or BFLOAT16 and the D axis is not 16-aligned, the B axis only supports up to 128.

    - Supports N axis less than or equal to 256.

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

  <!-- end id7 -->
  <!-- npu="910b" id8 -->
  - <term>Atlas A2 products</term>: Comprehensive restrictions on query, key, and value inputs in TND scenarios:
      - T is less than or equal to 65536.
      - N is equal to 8/16/32/64/128, and Q_N, K_N, and V_N are equal.
      - Q_D and K_D are equal to 192, and V_D is equal to 128/192.
      - The data type only supports BFLOAT16.
      - The sparse mode only supports sparse=0 without mask input, or sparse=3 with mask input.
      - When sparse=3, actualSeqLengths must be less than actualSeqLengthsKv for each individual batch.

  <!-- end id8 -->
  - <term>Atlas inference products</term>:
      - When inputLayout is BSH, the B axis is less than or equal to 300; otherwise, the B axis is less than or equal to 128. The N axis is less than or equal to 256. The S axis is less than or equal to 65535 (64k). If Q_S or KV_S is not 128-aligned and Q_S and KV_S are of different lengths, configuring atten_mask is not supported. The D axis is less than or equal to 512.

  <!-- npu="910b" id9 -->
  - <term>Atlas A2 products</term>: The supported data types are FLOAT16, BFLOAT16, and INT8.
  <!-- end id9 -->
  - <term>Atlas inference products</term>: Only the FLOAT16 data type is supported.

- pseShift function usage restrictions as follows:

  - The input shape must be (B, N, Q_S, KV_S) or (1, N, Q_S, KV_S), where Q_S is the S in the shape of query and KV_S is the S in the shapes of key and value.
  - Q_S must be greater than or equal to the S length of query, and KV_S must be greater than or equal to the S length of key.
  - When the KV_S of pseShift is not 32-aligned, it is recommended to pad it to 32 bytes to improve performance. The values of the extra part are not restricted. If this feature is not used, nullptr can be input.

  <!-- npu="910b" id10 -->
  - <term>Atlas A2 products</term>: The supported data types are FLOAT16 and BFLOAT16. When pseShift is of the FLOAT16 type, query must be of the FLOAT16 or INT8 type. When pseShift is of the BFLOAT16 type, query must be of the BFLOAT16 type. If query, key, and value are of the FLOAT16 type and pseShift is present, the high-precision mode is used by default, and the corresponding restrictions are inherited from those of the high-precision mode.
  <!-- end id10 -->
  - <term>Atlas inference products</term>: Only nullptr is supported.

- Restrictions on the attenMask function are as follows:

  - When the KV_S of attenMask is not 32-aligned, it is recommended to pad it to 32-byte alignment to improve performance. The extra part is padded with 1.
  - Typically, the recommended input shapes are Q_S, KV_S; B, Q_S, KV_S; 1, Q_S, KV_S; B, 1, Q_S, KV_S; and 1, 1, Q_S, KV_S, where Q_S is the S in the shape of query and KV_S is the S in the shapes of key and value.

  <!-- npu="910b" id11 -->
  - <term>Atlas A2 products</term>: The data types BOOL, INT8, and UINT8 are supported.
  <!-- end id11 -->
  - <term>Atlas inference products</term>: Only BOOL is supported.

- actualSeqLengths and actualSeqLengthsKv inputs have the following function usage restrictions:

  - For actualSeqLengths, if the sequence length is not specified, nullptr can be passed, indicating that the valid sequence length is the same as the S length in the query shape. Note that the valid sequence length of each batch in this parameter must not exceed the sequence length of the corresponding batch in the query.
  - For actualSeqLengthsKv, if the sequence length is not specified, nullptr can be passed, indicating that the valid sequence length is the same as the S length in the key/value shape. Note that the valid sequence length of each batch in this parameter must not exceed the sequence length of the corresponding batch in the key/value.
  - The input length of seqlen is subject to the following rules: If the input length is 1, all batches use the same seqlen; if the input length is greater than or equal to the number of batches, the first batch values of seqlen are used; input of other lengths is not supported.
  - When the inputLayout of query is TND, this input is required, and the number of elements in this input is used as the batch value. Each element value indicates the sum of the Sequence Length values of the current batch and all preceding batches. Therefore, the value of an element must be greater than or equal to that of the preceding element, and negative values are not allowed.

  <!-- npu="910b" id12 -->
  - <term>Atlas A2 products</term>: The data type INT64 is supported, and the TND format is supported.
  <!-- end id12 -->
  - <term>Atlas inference products</term>: The data type INT64 is supported.

- deqScale1 and deqScale2 inputs, function usage restrictions as follows:

  <!-- npu="910b" id14 -->
  - <term>Atlas A2 products</term>: The data types UINT64 and FLOAT32 are supported.
  <!-- end id14 -->
  - <term>Atlas inference products</term>: Only nullptr is supported.

- quantScale1 input, function usage restrictions as follows:

  <!-- npu="910b" id15 -->
  - <term>Atlas A2 products</term>: The data type FLOAT32 is supported.
  <!-- end id15 -->
  - <term>Atlas inference products</term>: Only nullptr is supported.

- The restrictions on the use of the quantScale2 and quantOffset2 inputs are as follows:

  <!-- npu="910b" id16 -->
  - <term>Atlas A2 products</term>: The supported data types are FLOAT32 and BFLOAT16.
  <!-- end id16 -->
  - <term>Atlas inference products</term>: Only Supports nullptr.

- The preTokens input has the following function usage restrictions:

  <!-- npu="910b" id17 -->
  - <term>Atlas A2 products</term>: The supported data type is INT64.
  <!-- end id17 -->
  - <term>Atlas inference products</term>: Only Supports Taking the value 2147483647.

- The nextTokens input has the following function usage restrictions:

  <!-- npu="910b" id18 -->
  - <term>Atlas A2 products</term>: The supported data type is INT64.
  <!-- end id18 -->
  - <term>Atlas inference products</term>: Only Supports Taking the values 0 and 2147483647.

- The inputLayout input has the following function usage restrictions:

  - Currently, BSH, BSND, BNSD, and BNSD_BSND are supported (when the input is BNSD, the output format is BSND). It is recommended to pass "BSH" when the user does not specify a format.

  <!-- npu="910b" id19 -->
  - <term>Atlas A2 products</term>: In addition to the preceding formats, TND is also supported (pse, full quantization, and post quantization are not supported).

  <!-- end id19 -->

- Restrictions on the numKeyValueHeads input and function usage are as follows:

  - numHeads must be divisible by numKeyValueHeads, and in the BSND, BNSD, and BNSD_BSND use cases, it must be the same as the N-axis shape value of key/value in shape; otherwise, an error is reported.

  <!-- npu="910b" id20 -->
  - <term>Atlas A2 products</term>: The supported data type is INT64.
  <!-- end id20 -->
  - <term>Atlas inference products</term>: Only Supports Taking the value 0.

- sparseMode input, function usage restrictions as follows:

  <!-- npu="910b" id21 -->
  - <term>Atlas A2 products</term>:
    - When sparseMode is 0, it represents the defaultMask mode. If attenmask is not passed, no mask operation is performed, and preTokens and nextTokens are ignored (internally assigned to INT_MAX). If attenmask is passed, the complete attenmask matrix (S1 * S2) must be provided, indicating that the portion between preTokens and nextTokens needs to be computed.
    - When sparseMode is 1, it represents allMask, and the complete attenmask matrix (S1 * S2) must be passed in.
    - When sparseMode is 2, it represents the leftUpCausal mode mask, and the optimized attenmask matrix (2048*2048) must be passed in.
    - When sparseMode is 3, it represents the rightDownCausal mode mask, corresponding to the lower triangular scenario partitioned by the right vertex, and the optimized attenmask matrix (2048*2048) must be passed in.
    - When sparseMode is 4, it represents the band mode mask, and the optimized attenmask matrix (2048*2048) must be passed in.
    - When sparseMode is 5, 6, 7, or 8, they represent prefix, global, dilated, and block_local respectively, and **none of them are currently supported**. If the user does not specify a value, 0 is recommended.
  <!-- end id21 -->
  <!-- npu="910b" id22 -->
  - <term>Atlas A2 products</term>: When inputLayout is TND, sparseMode only supports the values 0 and 3.
  <!-- end id22 -->
  - <term>Atlas inference products</term>: Only the value 0 is supported.

- Feature usage restrictions for the innerPrecise input are as follows:

  - There are four modes: 0, 1, 2, and 3. They occupy two bits in total. Bit 0 indicates the selection between high precision and high performance, and bit 1 indicates whether row invalidation correction is performed.
    <table style="undefined;table-layout: fixed; width: 600px"><colgroup>
    <col style="width: 200px">
    <col style="width: 200px">
      <col style="width: 200px">
    </colgroup><thead>
    <tr>
    <th>innerPrecise</th>
    <th>Mode</th>
    <th>Invalid Row Correction</th>
    </tr></thead>
    <tbody>
    <tr>
    <td>0</td>
    <td>High Precision</td>
    <td>×</td>
    </tr>
    <tr>
    <td>1</td>
    <td>High Performance</td>
    <td>×</td>
    </tr>
    <tr>
    <td>2</td>
    <td>High Precision</td>
    <td>√</td>
    </tr>
    <tr>
    <td>3</td>
    <td>High Performance</td>
    <td>√</td>
    </tr>
    </tbody>
    </table>

  - When Q_S>1, sparse_mode is 0 or 1, and a user-defined mask is input, enabling row invalid correction is recommended.

  - BFLOAT16 and INT8 do not distinguish between High Precision and High Performance. The invalid row correction takes effect on FLOAT16, BFLOAT16, and INT8. Currently, 0 and 1 are reserved configuration values. When the "mask portion that participates in the computation" contains an entire row of all 1s, precision loss may occur. In this case, set this parameter to 2 or 3 to enable the invalid row function to improve precision. However, this configuration degrades performance. If the operator can identify invalid row use cases, invalid row computation is automatically enabled. For example, sparse_mode is 3 and Sq > Skv.

- The attentionOut output has the following function usage restrictions:

  - When inputLayout is BNSD_BSND, the shape of the input query is BNSD and the output shape is BSND. In other cases, the shape of this parameter must keep consistent with the shape of the input query.

  <!-- npu="910b" id23 -->
  - <term>Atlas A2 products</term>: The supported data types are FLOAT16, BFLOAT16, and INT8.
  <!-- end id23 -->
  - <term>Atlas inference products</term>: Only FLOAT16 is supported.

- Comprehensive constraints on the number of int8 quantization-related input parameters and the input/output data formats:

  - When both input and output are INT8: The input parameters deqScale1, quantScale1, deqScale2, and quantScale2 must all be present. quantOffset2 is optional and defaults to 0 if not passed.
  - Use case where the input is INT8 and the output is FLOAT16: input parameters deqScale1, quantScale1, and deqScale2 must be input at the same time. If input parameter quantOffset2 or quantScale2 exists (that is, not nullptr), an error is reported and returned.
  - Use case where the input is FLOAT16 or BFLOAT16 and the output is INT8: the quantScale2 parameter is required, and quantOffset2 is optional, defaulting to 0 if not passed. If the parameters deqScale1, quantScale1, or deqScale2 exist (that is, are not nullptr), an error is reported and returned.
  - The parameters quantScale2 and quantOffset2 support two formats (per-tensor/per-channel) and two data types (FLOAT32/BFLOAT16). If quantOffset2 is passed, its type and shape information must be consistent with those of quantScale2. When the input is BFLOAT16, both FLOAT32 and BFLOAT16 are supported; otherwise, only FLOAT32 is supported. In per-channel format, when the output layout is BSH, the product of all dimensions of quantScale2 must equal H; for other layouts, the product must equal N*D. (It is recommended to pass a quantScale2 shape of [1,1,H] or [H] when the output layout is BSH, [1,N,1,D] or [N,D] when the output is BNSD, and [1,1,N,D] or [N,D] when the output is BSND)
  - When the output is int8 and quantScale2 and quantOffset2 are per-channel, left padding, Ring Attention, or the use case where D is not 32-byte aligned is not supported currently.
  - When the output is INT8, the scenario where sparse is band and preTokens/nextTokens are negative is not supported.

- When the output is INT8, the quantOffset2 parameter is passed with a non-null pointer and a non-empty tensor value, and sparseMode, preTokens, and nextTokens satisfy the following conditions, some rows of the matrix do not participate in computation, resulting in computation result errors. This use case is intercepted (solution: if this use case should not be intercepted, perform post-quantization outside the PFA API and do not enable it inside the PFA API):
    - sparseMode = 0. If attenMask is not a null pointer and actualSeqLengths - actualSeqLengthsKV - preTokens > 0 or nextTokens < 0 for each batch, the interception condition is satisfied.
    - For sparseMode = 1 or 2, the interception condition is never satisfied.
    - When sparseMode = 3, if each batch satisfies actualSeqLengthsKV - actualSeqLengths < 0, the interception condition is met.
    - For sparseMode = 4, the interception condition is satisfied when preTokens < 0 or nextTokens + actualSeqLengthsKV - actualSeqLengths < 0 for each batch.

## Example

The following sample code is for reference only. For details about the compilation and execution procedure, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_prompt_flash_attention_v3.h"
#include "securec.h"

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
                         aclDataType::ACL_BOOL, &resources.attenTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(outHostData, outShape, &resources.outDeviceAddr,
                         aclDataType::ACL_FLOAT16, &resources.outTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    std::vector<int64_t> actualSeqlenVector = {2};
    resources.actualSeqLengths = aclCreateIntArray(actualSeqlenVector.data(),
                                                  actualSeqlenVector.size());

    return ACL_SUCCESS;
}

int ExecutePromptFlashAttention(TensorResources& resources, aclrtStream stream,
                               void** workspaceAddr, uint64_t* workspaceSize) {
    int64_t numHeads = 2;
    int64_t numKeyValueHeads = numHeads;
    double scaleValue = 1 / sqrt(2);
    int64_t preTokens = 2147483647;
    int64_t nextTokens = 0;
    int64_t sparseMode = 0;
    int64_t innerPrecise = 1;

    constexpr const char LAYER_OUT_STR[] = "BNSD";
    constexpr size_t LAYER_OUT_LEN = sizeof(LAYER_OUT_STR);
    char layerOut[LAYER_OUT_LEN];
    memcpy(layerOut, LAYER_OUT_STR, LAYER_OUT_LEN);

    aclOpExecutor* executor;
    int ret = aclnnPromptFlashAttentionV3GetWorkspaceSize(
        resources.queryTensor, resources.keyTensor, resources.valueTensor,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        numHeads, scaleValue, preTokens, nextTokens, layerOut, numKeyValueHeads,
        sparseMode, innerPrecise, resources.outTensor, workspaceSize, &executor);

    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnPromptFlashAttentionV3GetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    if (*workspaceSize > 0ULL) {
        ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    ret = aclnnPromptFlashAttentionV3(*workspaceAddr, *workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnPromptFlashAttentionV3 failed. ERROR: %d\n", ret);
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
    ret = ExecutePromptFlashAttention(resources, stream, &workspaceAddr, &workspaceSize);
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
