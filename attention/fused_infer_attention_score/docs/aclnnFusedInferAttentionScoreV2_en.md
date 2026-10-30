# aclnnFusedInferAttentionScoreV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T09:42:32.700Z -->

**NOTE: This API will be deprecated in subsequent versions. Use the latest API [aclnnFusedInferAttentionScoreV5](./aclnnFusedInferAttentionScoreV5_en.md) instead.**

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

- Description: A FlashAttention operator suitable for both decode and prefill scenarios. It supports the prefill computation scenario (PromptFlashAttention) and the decode computation scenario (IncreFlashAttention).

  Compared with FusedInferAttentionScore, this API adds the keyAntiquantScaleOptional, keyAntiquantOffsetOptional, valueAntiquantScaleOptional, valueAntiquantOffsetOptional, keySharedPrefixOptional, valueSharedPrefixOptional, actualSharedPrefixLenOptional, keyAntiquantMode, and valueAntiquantMode parameters.

  **NOTE**

  KV Cache specific to the decode scenario: KV Cache is a common technique for optimizing inference performance in large models. During sampling, the Transformer model performs inference using the given prompt/context as the initial input (which can be processed in parallel), and then generates additional tokens one by one to further refine the generated sequence (reflecting the autoregressive nature of the model). During the sampling process, the Transformer performs self-attention operations, which requires extracting key-value (KV) vectors for each item in the current sequence (whether prompt/context or generated tokens). These vectors are stored in a matrix, commonly referred to as the KV cache.
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

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnFusedInferAttentionScoreV2GetWorkspaceSize" API first to obtain input parameters and calculate the workspace size required for the computation process, and then call the "aclnnFusedInferAttentionScoreV2" API to perform the computation.

```cpp
aclnnStatus aclnnFusedInferAttentionScoreV2GetWorkspaceSize(
    const aclTensor     *query,
    const aclTensorList *key,
    const aclTensorList *value,
    const aclTensor     *pseShiftOptional,
    const aclTensor     *attenMaskOptional,
    const aclIntArray   *actualSeqLengthsOptional,
    const aclIntArray   *actualSeqLengthsKvOptional,
    const aclTensor     *deqScale1Optional,
    const aclTensor     *quantScale1Optional,
    const aclTensor     *deqScale2Optional,
    const aclTensor     *quantScale2Optional,
    const aclTensor     *quantOffset2Optional,
    const aclTensor     *antiquantScaleOptional,
    const aclTensor     *antiquantOffsetOptional,
    const aclTensor     *blockTableOptional,
    const aclTensor     *queryPaddingSizeOptional,
    const aclTensor     *kvPaddingSizeOptional,
    const aclTensor     *keyAntiquantScaleOptional,
    const aclTensor     *keyAntiquantOffsetOptional,
    const aclTensor     *valueAntiquantScaleOptional,
    const aclTensor     *valueAntiquantOffsetOptional,
    const aclTensor     *keySharedPrefixOptional,
    const aclTensor     *valueSharedPrefixOptional,
    const aclIntArray   *actualSharedPrefixLenOptional,
    int64_t              numHeads,
    double               scaleValue,
    int64_t              preTokens,
    int64_t              nextTokens,
    char                *inputLayout,
    int64_t              numKeyValueHeads,
    int64_t              sparseMode,
    int64_t              innerPrecise,
    int64_t              blockSize,
    int64_t              antiquantMode,
    bool                 softmaxLseFlag,
    int64_t              keyAntiquantMode,
    int64_t              valueAntiquantMode,
    const aclTensor     *attentionOut,
    const aclTensor     *softmaxLse,
    uint64_t            *workspaceSize,
    aclOpExecutor       **executor)
```

```cpp
aclnnStatus aclnnFusedInferAttentionScoreV2(
     void                *workspace,
     uint64_t             workspaceSize,
     aclOpExecutor       *executor,
     const aclrtStream    stream)
```

## aclnnFusedInferAttentionScoreV2GetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1589px"><colgroup>
    <col style="width: 269px">
    <col style="width: 132px">
    <col style="width: 252px">
    <col style="width: 341px">
    <col style="width: 185px">
    <col style="width: 119px">
    <col style="width: 146px">
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
        <td>-</td>
        <td>FLOAT16, BFLOAT16, INT8</td>
        <td>ND</td>
        <td>3-4</td>
        <td>×</td>
      </tr>
      <tr>
        <td>key</td>
        <td>Input</td>
        <td>Input K in the formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16, INT8, INT4(INT32), HIFLOAT8, FLOAT8_E4M3FN, FLOAT4_E2M1</td>
        <td>ND</td>
        <td>3-4</td>
        <td>×</td>
      </tr>
      <tr>
        <td>value</td>
        <td>Input</td>
        <td>Input V in the formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16, INT8, INT4(INT32), HIFLOAT8, FLOAT8_E4M3FN, FLOAT4_E2M1</td>
        <td>ND</td>
        <td>3-4</td>
        <td>×</td>
      </tr>
      <tr>
        <td>pseShiftOptional</td>
        <td>Input</td>
        <td>Position encoding.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>Pass nullptr if this feature is not used.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>4</td>
        <td>×</td>
      </tr>
      <tr>
        <td>attenMaskOptional</td>
        <td>Input</td>
        <td>Mask matrix.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>Pass nullptr if this feature is not used.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>BOOL, INT8, UINT8</td>
        <td>ND</td>
        <td>2-4</td>
        <td>×</td>
      </tr>
      <tr>
        <td>actualSeqLengthsOptional</td>
        <td>Input</td>
        <td>Valid sequence length of query in each batch.</td>
        <td><ul><li>If the sequence length is not specified, pass in nullptr.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actualSeqLengthsKvOptional</td>
        <td>Input</td>
        <td>Valid sequence length of key/value in each batch.</td>
        <td><ul><li>If the sequence length is not specified, pass nullptr, indicating the Same Length as the S dimension of the key/value shape.</li>
            <li>Limits: The valid Sequence Length of each batch in this parameter must not be greater than the Sequence Length of the corresponding batch in key/value. If seqlenKv is of length 1, each Batch uses the same seqlenKv. If the length is Greater Than or Equal To Batch, the first Batch values of seqlenKv are used. Other lengths are not supported.</li>
            <li>For Comprehensive Constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>deqScale1Optional</td>
        <td>Input</td>
        <td>Dequantization factor following BMM1.</td>
          <td><ul><li>Empty tensors are not supported.</li>
          <li>per-tensor is supported.</li>
              <li>Pass nullptr if this function is not used.</li>
              <li>For Comprehensive Constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>UINT64, FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantScale1Optional</td>
        <td>Input</td>
        <td>Quantization factor before BMM2.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>Supports per-tensor. </li>
            <li>Pass nullptr if this feature is not used.</li>
            <li>For Comprehensive Constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>deqScale2Optional</td>
        <td>Input</td>
        <td>Dequantization factor after BMM2.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>Per-tensor is supported. </li>
           <li>Pass nullptr if this feature is not used.</li>
            <li>See <a href="#constraints">Constraints</a> for comprehensive constraints.</li></ul></td>
        <td>UINT64, FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantScale2Optional</td>
        <td>Input</td>
        <td>Quantization factor of the output.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>Supports per-tensor and per-channel. </li>
            <li>Pass nullptr if this function is not used.</li>
             <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT32, BFLOAT16</td>
        <td>ND</td>
        <td>1-4</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantOffset2Optional</td>
        <td>Input</td>
        <td>Quantization offset of the output.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>Supports per-tensor and per-channel. </li>
            <li>Pass nullptr if this feature is not used.</li>
             <li>For Comprehensive Constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT32, BFLOAT16</td>
        <td>ND</td>
        <td>1-4</td>
        <td>-</td>
      </tr>
      <tr>
        <td>antiquantScaleOptional</td>
        <td>Input</td>
        <td>Pseudo-quantization factor.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>Supports per-tensor, per-channel, and per-token. </li>
            <li>Pass nullptr if this feature is not used.</li>
            <li>For Comprehensive Constraints, see <a href="#constraints">Constraints</a>.</li>
            <li>The KV pseudo-quantization parameter separation mode is recommended.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>1-4</td>
        <td>-</td>
      </tr>
        <tr>
      <td>antiquantOffsetOptional</td>
        <td>Input</td>
        <td>Pseudo-quantization offset.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>Per-tensor, per-channel, and per-token are supported. </li>
            <li>When used, the shape must keep consistent with antiquantScaleOptional.</li>
            <li>If this function is not used, nullptr can be passed.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li>
            <li>The KV pseudo-quantization parameter separation mode is recommended.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>1-4</td>
        <td>-</td>
      </tr>
    <tr>
       <td>blockTableOptional</td>
        <td>Input</td>
        <td>Block mapping table used for KV storage in PageAttention.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>Set to nullptr if this feature is not used.</li></ul></td>
        <td>INT32</td>
        <td>ND</td>
        <td>2</td>
        <td>-</td>
      </tr>
      <tr>
       <td>queryPaddingSizeOptional</td>
        <td>Input</td>
        <td>Indicates whether the data in each batch of Query is right-aligned and the number of right-aligned items.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>Only Q_S greater than 1 is supported. This parameter does not take effect in other use cases.</li>
            <li>If this feature is not used, pass nullptr.</li></ul></td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
       <td>kvPaddingSizeOptional</td>
        <td>Input</td>
        <td>Indicates whether the data in each batch of key/value is right-aligned and the number of right-aligned items.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>Pass nullptr if this function is not used.</li></ul></td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
       <td>keyAntiquantScaleOptional</td>
        <td>Input</td>
        <td>Indicates the dequantization factor of key when the KV pseudo-quantization parameters are separated.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>Pass nullptr if this function is not used.</li>
            <li>For Comprehensive Constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT32, FLOAT8_E8M0</td>
        <td>ND</td>
        <td>1-4</td>
        <td>-</td>
      </tr>
        <tr>
       <td>keyAntiquantOffsetOptional</td>
        <td>Input</td>
        <td>Indicates the dequantization offset of key when kv Pseudo-Quantization Parameters are separated.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>If used, the shape must keep consistent with that of keyAntiquantScaleOptional.</li>
            <li>Pass nullptr if this function is not used.</li>
            <li>For Comprehensive Constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>1-4</td>
        <td>-</td>
      </tr>
         <tr>
       <td>valueAntiquantScaleOptional</td>
        <td>Input</td>
        <td>Dequantization factor of value when KV Pseudo-Quantization Parameters are separated.</td>
        <td><ul>
            <li>Empty tensors are not supported.</li>
            <li>nullptr can be passed in when this function is not used.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT32, FLOAT8_E8M0</td>
        <td>ND</td>
        <td>1-4</td>
        <td>-</td>
      </tr>
         <tr>
       <td>valueAntiquantOffsetOptional</td>
        <td>Input</td>
        <td>Indicates the dequantization offset of value when the KV Pseudo-Quantization Parameters are separated.</td>
        <td><ul>
            <li>Empty tensors are not supported.</li>
            <li>If used, the shape must keep consistent with valueAntiquantScaleOptional.</li>
            <li>Pass in nullptr if this function is not used.</li>
            <li>For Comprehensive Constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>1-4</td>
        <td>-</td>
      </tr>
       <tr>
       <td>keySharedPrefixOptional</td>
        <td>Input</td>
        <td>Parameter of the system prefix part of Key in the attention structure.</td>
        <td><ul>
            <li>Empty tensors are not supported.</li>
            <li>Pass nullptr if this feature is not used.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT16, BFLOAT16, INT8</td>
        <td>ND</td>
        <td>3-4</td>
        <td>x</td>
      </tr>
       <tr>
       <td>valueSharedPrefixOptional</td>
        <td>Input</td>
        <td>Input of the system prefix part of Value in the attention structure.</td>
        <td><ul>
            <li>Empty tensors are not supported.</li>
            <li>Pass nullptr if this feature is not used.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT16, BFLOAT16, INT8</td>
        <td>ND</td>
        <td>3-4</td>
        <td>x</td>
    </tr>
      <tr>
        <td>actualSharedPrefixLenOptional</td>
        <td>Input</td>
        <td>The valid Sequence Length of keySharedPrefix/valueSharedPrefix.</td>
        <td><ul><li>If this function is not used, nullptr can be passed in, indicating the same length as the s of keySharedPrefix/valueSharedPrefix.</li>
         <li>The valid Sequence Length in this parameter must not be greater than the Sequence Length in keySharedPrefix/valueSharedPrefix.</li>
         <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>numHeads</td>
        <td>Input</td>
        <td>Number of heads in query.</td>
        <td>In BNSD, BSND, BNSD_BSND, and TND use cases, this value must be the same as the value of the N axis of query in the shape. Otherwise, an execution error occurs.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>scaleValue</td>
        <td>Input</td>
        <td>Reciprocal of the square root of d in the formula.</td>
        <td><ul><li>The data type must comply with the data type deduction rules for query. </li>
            <li>1.0 is recommended if not specifically specified by the user. </li></ul></td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>preTokens</td>
        <td>Input</td>
        <td>Used for sparse computation, indicating the number of preceding tokens that attention needs to compute correlation with.</td>
          <td><ul><li>It is recommended to pass in 2147483647 if not specifically specified.</li>
              <li>This parameter is invalid when Q_S is 1.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>nextTokens</td>
        <td>Input</td>
        <td>Indicates that attention needs to be associated with the computation of the next few Tokens.</td>
        <td><ul><li>It is recommended to pass in 2147483647 if not specifically specified.</li>
            <li>This parameter is invalid when Q_S is 1.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>inputLayout</td>
        <td>Input</td>
        <td>Specifies the data layout format of the input query, key, and value.</td>
        <td><ul><li>If not specifically specified, "BSH" is recommended.</li></ul>
            <ul><li>For comprehensive constraints, see <a href="#constraints">Constraints</a></li></ul></td>
        <td>STRING</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>numKeyValueHeads</td>
        <td>Input</td>
        <td>Number of heads in key and value.</td>
        <td><ul><li>If not specifically specified, it is recommended to pass 0, indicating that key/value and query have the same number of heads.</li></ul>
            <ul><li>For Comprehensive Constraints, see <a href="#constraints">Constraints</a></li></ul></td>
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
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>innerPrecise</td>
        <td>Input</td>
        <td>Indicates the high-precision or high-performance option.</td>
        <td>See <a href="#constraints">Constraints</a> for comprehensive constraints.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>blockSize</td>
        <td>Input</td>
        <td>Maximum number of tokens in each block of the KV storage in PageAttention.</td>
        <td>See <a href="#constraints">Constraints</a> for comprehensive constraints.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>antiquantMode</td>
        <td>Input</td>
        <td>Pseudo-quantization mode.</td>
          <td><ul><li>If 0 is passed, per-channel is used (per-channel includes per-tensor).</li>
              <li>If 1 is passed, per-token is used.</li>
              <li>It is recommended to pass 0 if the mode is not specified.</li>
              <li>If Q_S is 1, a value other than 0 or 1 causes an execution exception. This parameter is invalid when Q_S is greater than or equal to 2.</li>
              <li>It is recommended to use the KV pseudo-quantization parameter separation mode.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>softmaxLseFlag</td>
        <td>Input</td>
        <td>Whether to output softmax_lse.</td>
          <td><ul><li>Supports outer slicing along the S dimension (an additional output).</li>
              <li>false is recommended if not specifically specified by the user.</li></ul></td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>keyAntiquantMode</td>
        <td>Input</td>
        <td>Pseudo-quantization mode of key.</td>
        <td><ul>
            <li>It is recommended to pass 0 if not specifically specified.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li>
            </ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>valueAntiquantMode</td>
        <td>Input</td>
        <td>Pseudo-quantization mode of value.</td>
          <td><ul><li>Except for the use case where keyAntiquantMode is 0 and valueAntiquantMode is 1, this parameter must be kept consistent with keyAntiquantMode.</li>
              <li>If the user does not explicitly specify this parameter, 0 is recommended.</li>
               <li>For Comprehensive Constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>attentionOut</td>
        <td>Output</td>
        <td>The output in the formula.</td>
        <td>The D dimension of this parameter is kept consistent with the D of value, and the other dimensions are kept consistent with the shape of query.</td>
        <td>FLOAT16, BFLOAT16, INT8</td>
        <td>ND</td>
        <td>3-4</td>
        <td>-</td>
      </tr>
      <tr>
        <td>softmaxLse</td>
        <td>Output</td>
        <td>Result of query multiplied by key in the ring attention algorithm.</td>
        <td>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>4</td>
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
      <th>Return Values</th>
      <th>Error Codes</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input query, key, value, or attentionOut is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data types and data formats of query, key, value, pseShift, attenMask, and attentionOut are out of the supported range.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>An exception occurs when the API calls the npu runtime API for memory operations.</td>
    </tr>
  </tbody>
  </table>

## aclnnFusedInferAttentionScoreV2

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
      <td>Workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device, obtained by the first-phase API aclnnFusedInferAttentionScoreV2GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, containing the operator computation flow.</td>
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
  - aclnnFusedInferAttentionScoreV2 defaults to a deterministic implementation.
- When using this API with PyTorch, ensure that the versions of CANN-related packages and PyTorch-related packages are compatible.

- Handling of null input parameters: The operator internally checks whether the query parameter is null. If it is null, the operator returns directly. If the query parameter is a non-null tensor and the key and value parameters are null tensors (i.e., S2 is 0), attentionOut is filled with all zeros. When attentionOut is a null tensor, the AscendCLNN framework handles it. For other input parameters marked as "nullable" in the parameter descriptions above, no processing is performed when they are null pointers.

- The shapes of the tensors corresponding to key and value must be exactly the same. In the non-contiguous use case, the batch in the tensorlists of key and value can only be 1, the number of tensors must be equal to B of query, and N and D must be the same. Due to tensorlist limitations, B cannot be greater than 256 in the non-contiguous use case.

- Usage restrictions for the key and value parameters:

  <!-- npu="910b" id44 -->
  - <term>Atlas A2 products</term>: The supported data types are FLOAT16, BFLOAT16, INT8, and INT4 (INT32).
  <!-- end id44 -->
  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>: The supported data types are FLOAT16, BFLOAT16, INT8, HIFLOAT8, FLOAT8_E4M3FN, INT4 (INT32), and FLOAT4_E2M1.

  <!-- end id7 -->

- Usage restrictions for pseShiftOptional:

  - If Q_S is not 1, when pseShiftOptional is of the FLOAT16 type, query must be of the FLOAT16 or INT8 type; when pseShiftOptional is of the BFLOAT16 type, query must be of the BFLOAT16 type. The input shape must be (B,N,Q_S,KV_S) or (1,N,Q_S,KV_S), where Q_S is the S in the shape of query and KV_S is the S in the shapes of key and value. For the use case where the KV_S of pseShiftOptional is not 32-byte aligned, it is recommended to pad to 32 bytes to improve performance, and the padding values of the redundant part are not restricted.
  - If Q_S is 1, when pseShiftOptional is of the FLOAT16 type, query must be of the FLOAT16 type; when pseShiftOptional is of the BFLOAT16 type, query must be of the BFLOAT16 type. The input shape must be (B,N,1,KV_S) or (1,N,1,KV_S), where KV_S is the S in the shape of key and value. If the KV_S of pseShiftOptional is not 32-byte aligned, padding to 32 bytes is recommended to improve performance. The padded values in the extra part are not restricted.

- attenMaskOptional: Masks the QK result to indicate whether to compute the correlation between tokens.

  - If Q_S is not 1, the recommended input shapes are (Q_S,KV_S); (B,Q_S,KV_S); (1,Q_S,KV_S); (B,1,Q_S,KV_S); (1,1,Q_S,KV_S).
  - When Q_S is 1, the recommended shape input is (B,KV_S); (B,1,KV_S); (B,1,1,KV_S).
  - Q_S is the S in the shape of query, and KV_S is the S in the shape of key and value. However, if Q_S or KV_S is not aligned to 16 or 32, it can be rounded up to the aligned S.

- When the data type of attenMask is INT8 or UINT8, the values in its tensor must be 0 or 1.

- actualSeqLengthsOptional: an aclIntArray on the host side, representing the effective sequence length of query in different batches. The data type supports INT64. If seqlen is not specified, nullptr can be passed in, indicating that the length is the same as S in the shape of query. Constraint: The effective sequence length of each batch in this input parameter must not be greater than the sequence length of the corresponding batch in query. This parameter is invalid when Q_S is 1. When the input length of seqlen is 1, the same seqlen is used for each batch. When the input length is greater than or equal to the number of batches, the first Batch elements of seqlen are used. Other lengths are not supported.

- actualSeqLengthsKvOptional: an aclIntArray on the host side. nullptr can be passed in, representing the effective sequence length of key/value in different batches. The data type supports INT64. If seqlen is not specified, nullptr can be passed in, indicating that the length is the same as S in the shape of key/value. Constraint: The effective sequence length of each batch in this input parameter must not be greater than the sequence length of the corresponding batch in key/value. When the input length of seqlenKv is 1, the same seqlenKv is used for each batch. When the input length is greater than or equal to the number of batches, the first Batch elements of seqlenKv are used. Other lengths are not supported.

- Comprehensive constraints on the number of int8 quantization-related input parameters and the input/output data formats:

  - When both input and output are INT8: The input parameters deqScale1, quantScale1, deqScale2, and quantScale2 must all be present. quantOffset2 is optional and defaults to 0 if not passed.
  - For the use case where the input is INT8 and the output is FLOAT16: parameters deqScale1, quantScale1, and deqScale2 must exist at the same time. If parameter quantOffset2 or quantScale2 exists (that is, is not nullptr), an error is reported and returned.
  - For the use case where the input is FLOAT16 or BFLOAT16 and the output is INT8: parameter quantScale2 must exist, and quantOffset2 is optional and defaults to 0 if not passed. If parameter deqScale1, quantScale1, or deqScale2 exists (that is, is not nullptr), an error is reported and returned.
  - Parameters quantScale2 and quantOffset2 support the per-tensor/per-channel formats and the FLOAT32/BFLOAT16 data types. If quantOffset2 is passed, its type and shape information must be the same as those of quantScale2. When the input is BFLOAT16, both FLOAT32 and BFLOAT16 are supported; otherwise, only FLOAT32 is supported. In per-channel format, if the output layout is BSH, the product of all dimensions of quantScale2 must be H; for other layouts, the product must be N*D. (If the output layout is BSH, the recommended quantScale2 shape is [1,1,H] or [H]; if the output layout is BNSD, the recommended shape is [1,N,1,D] or [N,D]; if the output layout is BSND, the recommended shape is [1,1,N,D] or [N,D]).
  - inputLayout supports only BSH, BNSD, BSND, and BNSD_BSND.

- Constraints on pseudo-quantization parameters antiquantScale and antiquantOffset:

  - Only the pseudo-quantization scenario where kv_dtype is int8 is supported.
  - Per-channel mode: The shape of the two parameters can be \(2, N, 1, D\), \(2, N, D\), or \(2, H\), where N is numKeyValueHeads. The parameter data type is the same as the query data type, and antiquantMode is set to 0.
  - per-tensor mode: the shapes of the two parameters are both (2), the data type is the same as the query data type, and antiquantMode is set to 0.
  - per-token mode: the shapes of the two parameters are both \(2, B, S\), the data type is fixed to FLOAT32, and antiquantMode is set to 1.
  - In asymmetric quantization mode, both antiquantScale and antiquantOffset parameters must be present.
  - In symmetric quantization mode, antiquantOffset can be null (i.e., nullptr). When the antiquantOffset parameter is null, symmetric quantization is performed; otherwise, asymmetric quantization is performed.

  <!-- npu="910b" id8 -->
  - <term>Atlas A2 products</term>: When Q_S equals 1, the data type supports FLOAT16, BFLOAT16, and FLOAT32. When Q_S is greater than or equal to 2, only FLOAT16 is supported.
  <!-- end id8 -->
  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT products</term>: The data type supports FLOAT16, BFLOAT16, and FLOAT32.

  <!-- end id9 -->

- keyAntiquantScaleOptional: An aclTensor on the Device side. The data format supports ND. During KV pseudo-quantization parameter separation, it represents the dequantization factor of key. If this function is not used, nullptr can be passed in.

  <!-- npu="910b" id10 -->
  - <term>Atlas A2 products</term>: The data type supports FLOAT16, BFLOAT16, and FLOAT32. When Q_S is greater than or equal to 2, only per-token and per-channel modes are supported. When Q_S equals 1, only per-tensor, per-channel, per-token, per-tensor stacked with per-head, per-token stacked with per-head, per-token stacked with page attention mode management of scale/offset, and per-token stacked with per-head and page attention mode management of scale/offset are supported.
  <!-- end id10 -->
  <!-- npu="950" id11 -->
  - <term>Ascend 950PR&950DT products</term>: The data type supports FLOAT16, BFLOAT16, FLOAT32, and FLOAT8_E8M0. It supports per-tensor, per-channel, per-token, per-tensor stacked with per-head, per-token stacked with per-head, per-token stacked with page attention mode management of scale/offset, per-token stacked with per-head and page attention mode management of scale/offset, and per-token-group.

  <!-- end id11 -->

- keyAntiquantOffsetOptional: A device-side aclTensor. The supported data types are FLOAT16, BFLOAT16, and FLOAT32. The supported data format is ND. During KV pseudo-quantization parameter separation, it represents the dequantization offset of key. If this function is used, its data type and shape must be consistent with keyAntiquantScaleOptional. If this function is not used, nullptr can be passed in.

  <!-- npu="910b" id12 -->
  - <term>Atlas A2 products</term>: When Q_S is greater than or equal to 2, only per-token and per-channel modes are supported. When Q_S equals 1, only per-tensor, per-channel, per-token, per-tensor stacked with per-head, per-token stacked with per-head, per-token stacked with page attention mode management of scale/offset, and per-token stacked with per-head and page attention mode management of scale/offset are supported.
  <!-- end id12 -->
  <!-- npu="950" id13 -->
  - <term>Ascend 950PR&950DT products</term>: Supports per-tensor, per-channel, per-token, per-tensor stacked with per-head, per-token stacked with per-head, per-token stacked with page attention mode management of scale/offset, and per-token stacked with per-head with page attention mode management of scale/offset.

  <!-- end id13 -->

- valueAntiquantScaleOptional: Device-side aclTensor. The data format supports ND. During KV pseudo-quantization parameter separation, it represents the dequantization factor of value. If this function is not used, nullptr can be passed in.

  <!-- npu="910b" id14 -->
  - <term>Atlas A2 products</term>: The data type supports FLOAT16, BFLOAT16, and FLOAT32. When Q_S is greater than or equal to 2, only per-token and per-channel modes are supported. When Q_S equals 1, only per-tensor, per-channel, per-token, per-tensor stacked with per-head, per-token stacked with per-head, per-token stacked with page attention mode management of scale/offset, and per-token stacked with per-head and page attention mode management of scale/offset are supported.
  <!-- end id14 -->
  <!-- npu="950" id15 -->
  - <term>Ascend 950PR&950DT products</term>: The data type supports FLOAT16, BFLOAT16, FLOAT32, and FLOAT8_E8M0. It supports per-tensor, per-channel, per-token, per-tensor stacked with per-head, per-token stacked with per-head, per-token stacked with page attention mode management of scale/offset, per-token stacked with per-head and page attention mode management of scale/offset, and per-token-group.

  <!-- end id15 -->

- valueAntiquantOffsetOptional: Device-side aclTensor. The data type supports FLOAT16, BFLOAT16, and FLOAT32. The data format supports ND. During KV pseudo-quantization parameter separation, it represents the dequantization offset of value. If this function is used, its data type and shape must be kept consistent with valueAntiquantScaleOptional. If this function is not used, nullptr can be passed in.

  <!-- npu="910b" id16 -->
  - <term>Atlas A2 products</term>: When Q_S is greater than or equal to 2, only per-token and per-channel modes are supported. When Q_S equals 1, only per-tensor, per-channel, per-token, per-tensor stacked with per-head, per-token stacked with per-head, per-token stacked with page attention mode management of scale/offset, and per-token stacked with per-head and page attention mode management of scale/offset are supported.
  <!-- end id16 -->
  <!-- npu="950" id17 -->
  - <term>Ascend 950PR&950DT products</term>: Supports per-tensor, per-channel, per-token, per-tensor stacked with per-head, per-token stacked with per-head, per-token stacked with page attention mode management of scale/offset, and per-token stacked with per-head with page attention mode management of scale/offset.

  <!-- end id17 -->

- actualSharedPrefixLenOptional: A Host-side aclIntArray. Nullptr can be passed, representing the valid sequence length of keySharedPrefix/valueSharedPrefix. The data type supports INT64. If seqlen is not specified, nullptr can be passed, indicating that the length is the same as that of keySharedPrefix/valueSharedPrefix. Constraint: The valid sequence length in this input parameter must not be greater than the sequence length in keySharedPrefix/valueSharedPrefix.

- inputLayout: Specifies the data layout format of the input query, key, and value. Currently supports BSH, BSND, BNSD, BNSD_BSND (when the input is BNSD, the output format is BSND, only supported when Q_S is greater than 1), and TND. If not explicitly specified by the user, it is recommended to pass "BSH".

  - Description: The data layout format of query, key, and value can be interpreted from multiple dimensions, where B (Batch) indicates the batch size of input samples, S (Seq-Length) indicates the sequence length of input samples, H (Hidden-Size) indicates the hidden layer size, N (Head-Num) indicates the number of heads, and D (Head-Dim) indicates the smallest unit dimension of the hidden layer, satisfying D = H/N.

- numKeyValueHeads usage constraints: numHeads must be divisible by numKeyValueHeads. In BSND, BNSD, BNSD_BSND, and TND scenarios, it must also be the same as the N-axis shape value of key/value in the shape; otherwise, an execution exception occurs.

- The usage restrictions of sparseMode are as follows:

  <div style="overflow-x: auto;">
    <table style="table-layout: fixed; width: 1210px">
        <colgroup>
            <col style="width: 150px">
            <col style="width: 210px">
            <col style="width: 850px">
        </colgroup>
        <thead>
            <tr>
                <th>sparseMode</th>
                <th>Mode</th>
                <th>Description</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>0</td>
                <td>defaultMask mode</td>
                <td>                    <ul style="margin: 0; padding-left: 20px;">
                        <li>If attenmask is not passed: no mask operation is performed, and preTokens and nextTokens are ignored (internally set to INT_MAX);</li>
                        <li>If attenmask is passed: the complete attenmask matrix (S1 * S2) must be passed, indicating that the part between preTokens and nextTokens needs to be calculated.</li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td>1</td>
                <td>allMask mode</td>
                <td>The complete attenmask matrix (S1 * S2) must be passed.</td>
            </tr>
            <tr>
                <td>2</td>
                <td>leftUpCausal mode</td>
                <td>The optimized attenmask matrix (2048*2048) must be passed in.</td>
            </tr>
            <tr>
                <td>3</td>
                <td>rightDownCausal mode</td>
                <td>Corresponds to the lower triangular use case partitioned by the right vertex. The optimized attenmask matrix (2048*2048) must be passed in.</td>
            </tr>
            <tr>
                <td>4</td>
                <td>band mode</td>
                <td>The optimized attenmask matrix (2048*2048) must be passed in.</td>
            </tr>
            <tr>
                <td>5</td>
                <td>prefix mode</td>
                <td>This mode is not supported currently. It is recommended to pass in 0 if the user does not specify it.</td>
            </tr>
            <tr>
                <td>6</td>
                <td>global mode</td>
                <td>This mode is not supported currently. It is recommended to pass in 0 if not specifically specified.</td>
            </tr>
            <tr>
                <td>7</td>
                <td>dilated mode</td>
                <td>This mode is not supported currently. It is recommended to pass in 0 if not specifically specified.</td>
            </tr>
            <tr>
                <td>8</td>
                <td>block_local mode</td>
                <td>This mode is not supported currently. It is recommended to pass in 0 if not specifically specified.</td>
            </tr>
            <tr>
                <td colspan="3" style="text-align: left; ">                    <strong>Special constraint:</strong> When Q_S is 1, the sparseMode parameter is invalid.
                </td>
            </tr>
          </tbody>
      </table>
  </div>

- The usage restrictions of innerPrecise are as follows:

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
    <td>High precision</td>
    <td>×</td>
    </tr>
    <tr>
    <td>1</td>
    <td>High performance</td>
    <td>×</td>
    </tr>
    <tr>
    <td>2</td>
    <td>High precision</td>
    <td>√</td>
    </tr>
    <tr>
    <td>3</td>
    <td>High performance</td>
    <td>√</td>
    </tr>
    </tbody>
    </table>

  - Note: BFLOAT16 and INT8 do not distinguish between high precision and high performance. Row invalid correction takes effect for FLOAT16, BFLOAT16, and INT8. Currently, 0 and 1 are reserved configuration values. When a row in the mask part involved in the computation is fully set to 1s, precision loss may occur. In this case, set this parameter to 2 or 3 to enable the row invalidation function and improve precision. However, this configuration causes performance degradation. If the operator can identify use cases with invalid rows, it automatically enables invalid row computation, for example, when sparse_mode is 3 and Sq > Skv.

- keyAntiquantMode usage restrictions are as follows:

  <div style="overflow-x: auto;">
    <table style="table-layout: fixed; width: 830px">
        <colgroup>
            <col style="width: 230px">
            <col style="width: 600px">
        </colgroup>
        <thead>
            <tr>
                <th>keyAntiquantMode</th>
                <th>Description</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>0</td>
                <td>Represents the per-channel mode (per-channel includes per-tensor).</td>
            </tr>
            <tr>
                <td>1</td>
                <td>Represents the per-token mode.</td>
            </tr>
            <tr>
                <td>2</td>
                <td>Represents the per-tensor stacked with per-head mode.</td>
            </tr>
            <tr>
                <td>3</td>
                <td>Represents the per-token stacked with per-head mode.</td>
            </tr>
            <tr>
                <td>4</td>
                <td>Represents the per-token stacked usage with the scale/offset mode managed by page attention mode.</td>
            </tr>
            <tr>
                <td>5</td>
                <td>Represents per-token stacked with per-head, with scale/offset managed by page attention mode.</td>
            </tr>
            <tr>
                <td>6</td>
                <td>Represents the per-token-group mode.</td>
            </tr>
            <tr>
                <td colspan="2" style="text-align: left; ">                    <strong>Special constraints:</strong> Keep consistent with valueAntiquantMode, except when keyAntiquantMode is 0 and valueAntiquantMode is 1.
                </td>
            </tr>
          </tbody>
      </table>
  </div>

  <!-- npu="910b" id18 -->
  - <term>Atlas A2 products</term>: When Q_S equals 1, passing values other than 0, 1, 2, 3, 4, or 5 causes abnormal execution. When Q_S is greater than or equal to 2, only 0 and 1 are supported; other values cause abnormal execution.
  <!-- end id18 -->
  <!-- npu="950" id19 -->
  - Ascend 950PR&950DT products: Passing values other than 0, 1, 2, 3, 4, 5, and 6 causes abnormal execution.

  <!-- end id19 -->

- The usage restrictions of valueAntiquantMode are as follows:

  - Keep consistent with keyAntiquantMode, except when keyAntiquantMode is 0 and valueAntiquantMode is 1.

  <!-- npu="910b" id20 -->
  - <term>Atlas A2 products</term>: When Q_S equals 1, passing values other than 0, 1, 2, 3, 4, or 5 causes abnormal execution. When Q_S is greater than or equal to 2, only 0 and 1 are supported; other values cause abnormal execution.
  <!-- end id20 -->
  <!-- npu="950" id21 -->
  - Ascend 950PR&950DT products: Passing values other than 0, 1, 2, 3, 4, 5, and 6 causes abnormal execution.

  <!-- end id21 -->

- softmaxLse usage constraints are as follows:

  - In the ring attention algorithm, softmax_max is obtained by first taking the max of the result of query multiplied by key. Then, softmax_max is subtracted from the result of query multiplied by key, and exp is taken. Next, the sum is computed to obtain softmax_sum. Finally, log is taken on softmax_sum, and softmax_max is added to obtain the result.
  - When softmaxLseFlag is True, the shape must be [B, N, Q_S, 1]. Data with inf indicates invalid data.
  - When softmaxLseFlag is False, if the softmaxLse tensor passed in is not empty, the tensor data is directly returned. If nullptr is passed for softmaxLse, a tensor of shape {1} with all zeros is returned.

- **When Q_S is greater than 1**:

  - The function usage restrictions for query, key, and value inputs are as follows:

    - B-axis restrictions
      - Supports B-axis less than or equal to 65536.
      - In non-contiguous cases, the batch of tensors in the key and value tensorlists can only be 1, and the number of tensors must be equal to the B of query. N and D must be the same. Due to tensorlist limitations, B cannot be greater than 256 in non-contiguous cases.

      <!-- npu="910b" id22 -->
      - <term>Atlas A2 products</term>:
        - If the input type is INT8 and the D-axis is not 32-byte-aligned, the maximum supported value of the B-axis is 128. If the input type is FLOAT16 or BFLOAT16 and the D-axis is not 16-byte-aligned, the B-axis is also only supported up to 128.

      <!-- end id22 -->

    - The N axis must be less than or equal to 256, and the D axis must be less than or equal to 512. When inputLayout is BSH or BSND, it is recommended that N*D be less than 65535.

    - S must be less than or equal to 20971520 (20M). In some long sequence scenarios, if the computation load is too large, the PFA operator may time out (aicore error type, with errorStr: timeout or trap error). In such scenarios, S-axis splitting is recommended. Note: The computation load is affected by B, S, N, D, etc. Larger values result in greater computation load. Typical long sequence scenarios that may cause timeouts (i.e., where the product of B, S, N, and D is large) include but are not limited to:

      <div style="overflow-x: auto;">
      <table style="undefined;table-layout: fixed; width: 900px"><colgroup>
      <col style="width: 130px">
      <col style="width: 130px">
      <col style="width: 230px">
      <col style="width: 130px">
      <col style="width: 130px">
      <col style="width: 180px">
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
      </div>

    - D-axis constraints:

      <!-- npu="910b" id23 -->
      - <term>Atlas A2 products</term>:
        - When the type of query, key, value, or attentionOut includes INT8, the D-axis must be 32-aligned. When the type of query, key, value, or attentionOut includes INT4, the D-axis must be 64-aligned. When all types are FLOAT16 or BFLOAT16, the D-axis must be 16-aligned.

      <!-- end id23 -->
      <!-- npu="950" id24 -->
      - Ascend 950PR&950DT products:

        - Non-quantization scenario: The types of query, key, and value are all FLOAT16 or BFLOAT16, and D-axis 1–512 is fully supported.
        - Full-quantization scenario: The types of query, key, and value are all INT8, and D-axis 1–512 is fully supported.
        - Pseudo-quantization scenario: The type of query is FLOAT16 or BFLOAT16, and the types of key and value are INT8/HIFLOAT8/FLOAT8_E4M3FN/FLOAT4_E2M1/INT4(INT32). When the types of key and value are FLOAT4_E2M1/INT4(INT32), the D axis of query and the D axis of key and value only support 64-byte alignment (INT32 only supports 8-byte alignment for the D axis of key and value).

      <!-- end id24 -->

  - The sparseMode parameter currently only supports values of 0, 1, 2, 3, and 4. An error is reported when other values are used.

    - When sparseMode = 0, if attenMask is a null pointer or attenMask is passed in a left-padding scenario, the input parameters preTokens and nextTokens are ignored.
    - When sparseMode = 2, 3, or 4, the shape of attenMask must be S,S or 1,S,S or 1,1,S,S, where the value of S must be fixed to 2048, and the user must ensure that the passed attenMask is lower triangular. If attenMask is nullptr or the passed shape is incorrect, an error is reported.
    - In scenarios where sparseMode = 1, 2, or 3, the input parameters preTokens and nextTokens are ignored and assigned values according to relevant rules.
  - The composite parameter scenario for kvCache dequantization supports only the case where query is FLOAT16, dequantizing key and value of INT8 type to FLOAT16. The product range of the datarange of the input parameter key/value and the datarange of the input parameter antiQuantScale falls within (-1, 1). The high-performance mode can guarantee precision; otherwise, the high-precision mode must be enabled to guarantee precision.
  - Page attention scenario:

    - To enable page attention, blockTable must exist and be valid, and key and value are arranged in contiguous memory according to the indexes in blockTable. In this use case, the inputLayout parameter of key and value does not take effect. blockTable is filled with blockids. The validity of blockids is not verified currently, and you need to ensure it by yourself.

      <!-- npu="910b" id25 -->
      - <term>Atlas A2 products</term>: key and value with the FLOAT16/BFLOAT16 data type are supported.
      <!-- end id25 -->
      <!-- npu="950" id26 -->
      - <term>Ascend 950PR&950DT products</term>: Supports key and value dtype of FLOAT16/BFLOAT16/INT8/HIFLOAT8/FLOAT8_E4M3FN/FLOAT4_E2M1/INT4(INT32).
      <!-- end id26 -->

    - blockSize is a user-defined parameter whose value affects the performance of page attention. When page attention is enabled, the minimum value of blockSize is 128, the maximum value is 512, and blockSize must be a multiple of 128. Generally, page attention improves throughput but decreases performance.
    - In page attention use cases, if the input kv cache layout is BnBsH (blocknum, blocksize, H) and KV_N * D exceeds 65535, the execution is intercepted and an error is reported due to hardware instruction constraints. To resolve this issue, enable GQA (to reduce KV_N) or change the kv cache layout to BnNBsD (blocknum, KV_N, blocksize, D). When the inputLayout of query is BNSD, the kv cache layout supports both BnBsH and BnNBsD. When the inputLayout of query is BSH or BSND, the kv cache layout supports only BnBsH. blocknum cannot be less than the sum of the number of blocks per batch calculated based on actualSeqLengthsKv and blockSize. In addition, the shapes of key and value must be the same.
    - page attention pseudo-quantization use case

      <!-- npu="910b" id27 -->
      - <term>Atlas A2 products</term>: Supports query as FLOAT16/BFLOAT16, and supports key and value as INT8.
      <!-- end id27 -->
      <!-- npu="950" id28 -->
      - <term>Ascend 950PR&950DT products</term>: Supports query dtype of FLOAT16/BFLOAT16 and key and value dtype of INT8/HIFLOAT8/FLOAT8_E4M3FN/FLOAT4_E2M1/INT4(INT32).
      <!-- end id28 -->

    - page attention full quantization use case

      <!-- npu="910b" id29 -->
      - <term>Atlas A2 products</term>: Not supported query dtype as INT8.
      <!-- end id29 -->
      <!-- npu="950" id30 -->
      - <term>Ascend 950PR&950DT products</term>: Both query and KV cache in INT8 are supported.
      <!-- end id30 -->

    - Page attention does not support the tensorlist scenario or the left-padding scenario.
    - In the page attention scenario, actualSeqLengthsKv must be passed.
    - In page attention scenarios, blockTable must be two-dimensional. The first dimension length must equal B, and the second dimension length cannot be less than maxBlockNumPerSeq (maxBlockNumPerSeq is the number of blocks corresponding to the maximum actualSeqLengthsKv across different batches).
    - When page attention is enabled, the input KV_S in the following use cases must be greater than or equal to maxBlockNumPerSeq * blockSize
      - When attenMask is passed in, for example, the mask shape is (B, 1, Q_S, KV_S)
      - When pseShift is passed in, for example, the pseShift shape is (B, N, Q_S, KV_S)
      - Pseudo-quantization per-token mode enabled: the shapes of the input parameters antiquantScale and antiquantOffset are both (2,B,S)
      - Pseudo-quantization per-token stacked with per-head mode enabled: the shapes of the input parameters antiquantScale and antiquantOffset are both (B,N,S), the data type is fixed to FLOAT32, and this mode is supported when the data types of key and value are INT8 or INT4 (INT32).
      - Per-token-group mode enabled: the shape of antiquantScale is \(1, B, N, S, D/32\), the data type is fixed to FLOAT8_E8M0, and antiquantOffset is not supported. This mode is supported when the data types of key and value are FLOAT4_E2M1.
  - Query left-padding scenario:

    - The formula for the query copy start position is: Q_S - queryPaddingSize - actualSeqLengths. The formula for the query copy end position is: Q_S - queryPaddingSize. The query copy start position must not be less than 0, and the end position must not be greater than Q_S. Otherwise, the result will not meet expectations.
    - If kvPaddingSize is less than 0, it is set to 0.
    - Must be enabled together with the actualSeqLengths parameter. Otherwise, the query right-padding use case applies by default.
    - PageAttention is not supported and cannot be enabled together with the blocktable parameter.

    <!-- npu="910b" id31 -->
    - <term>Atlas A2 products</term>: The scenario where Q is BF16/FP16 and KV is INT4 is not supported.
    <!-- end id31 -->

  - kv left-padding scenario:

    - The copy start point of key and value is calculated as: KV_S - kvPaddingSize - actualSeqLengthsKv. The copy end point of key and value is calculated as: KV_S - kvPaddingSize. The copy start point of key and value must not be less than 0, and the end point must not be greater than KV_S. Otherwise, the result will not meet expectations.
    - If kvPaddingSize is less than 0, it is set to 0.
    - Must be enabled together with the actualSeqLengthsKv parameter. Otherwise, the KV right-padding use case applies by default.
    - PageAttention is not supported and cannot be enabled together with the blocktable parameter.

    <!-- npu="910b" id32 -->
    - <term>Atlas A2 products</term>: The scenario where Q is BF16/FP16 and KV is INT4 is not supported.
    <!-- end id32 -->

  - When the output is int8 and quantScale2 and quantOffset2 are per-channel, the left-padding use case, Ring Attention, or the use case where D is not 32-byte aligned is not yet supported.
  - When the output is INT8, the scenario where sparse is band and preTokens/nextTokens are negative is not supported.
  - When the output is INT8, if quantOffset2 is passed in as a non-null pointer and a non-null tensor value, and sparseMode, preTokens, and nextTokens meet the following conditions, some rows in the matrix will not participate in the computation, resulting in computation errors. This use case is intercepted (Solution: if you want this use case not to be intercepted, perform post-quantization outside the FIA API instead of inside the FIA API):

    - sparseMode = 0: if attenMask is a non-null pointer, and actualSeqLengths - actualSeqLengthsKV - actualSharedPrefixLen - preTokens > 0 or nextTokens < 0 for each batch, the interception condition is met
    - sparseMode = 1 or 2: the interception condition is never met
    - sparseMode = 3, if each batch satisfies actualSeqLengthsKV + actualSharedPrefixLen - actualSeqLengths < 0, the interception condition is met.
    - sparseMode = 4: if preTokens < 0 or nextTokens + actualSeqLengthsKV + actualSharedPrefixLen - actualSeqLengths < 0 for each batch, the interception condition is met

  - pseShift function usage restrictions as follows:

    - The pseShift function is supported when the query data type is FLOAT16, BFLOAT16, or INT8.
    - When the query data type is FLOAT16 and pseShift is present, the high-precision mode is forcibly used, and the corresponding restrictions are inherited from those of the high-precision mode.
    - Q_S must be greater than or equal to the S length of query, and KV_S must be greater than or equal to the S length of key. In the prefix scenario, KV_S must be greater than or equal to the sum of actualSharedPrefixLen and the S length of key.

    <!-- npu="950" id33 -->
    - <term>Ascend 950PR&950DT products</term>: Non-quantized and fully quantized scenarios: No alignment restrictions.
    <!-- end id33 -->

  - Prefix-related parameter constraints:

    - keySharedPrefix and valueSharedPrefix must either both be null or both be non-null.
    - When both keySharedPrefix and valueSharedPrefix are non-null, keySharedPrefix, valueSharedPrefix, key, and value must have the same number of dimensions and keep their dtypes consistent.
    - When both keySharedPrefix and valueSharedPrefix are non-null, the first dimension (batch) of keySharedPrefix must be 1; in BNSD and BSND layouts, the N and D axes must be consistent with key; in BSH layout, H must be consistent with key. The same applies to valueSharedPrefix. S of keySharedPrefix and valueSharedPrefix must be equal.
    - When actualSharedPrefixLen is present, its shape must be [1], and its value cannot exceed S of keySharedPrefix and valueSharedPrefix.
    - The sum of S of the shared prefix and S of key or value must satisfy the original S constraints of key or value.
    - prefix does not support PageAttention scenarios, left padding scenarios, or tensorlist scenarios.
    - In prefix scenarios, when sparse is 0 or 1, if attenmask is passed, S2 must be greater than or equal to the sum of actualSharedPrefixLen and the S length of key.
    - In prefix scenarios, the case where all qkv inputs are int8 is not supported.
  - KV pseudo-quantization parameter separation:

    <!-- npu="910b" id34 -->
    - <term>Atlas A2 products</term>:
      - keyAntiquantMode and valueAntiquantMode must be kept consistent
      - keyAntiquantScale and valueAntiquantScale are either both null or both non-null; keyAntiquantOffset and valueAntiquantOffset are either both null or both non-null
      - If both KeyAntiquantScale and valueAntiquantScale are non-null, their shapes must be kept consistent; if both keyAntiquantOffset and valueAntiquantOffset are non-null, their shapes must be kept consistent
      - Only per-token and per-channel modes are supported. In per-token mode, the shapes of both parameters must be \(B, S\) and the data type is fixed to FLOAT32. In per-channel mode, the shapes of both parameters must be \(N, D\), \(N, 1, D\) or \(H\) and the data type is fixed to BF16.
      - If both pseudo-quantization parameters and KV-decoupled quantization parameters are passed in, the KV-decoupled quantization parameters take precedence.
      - In scenarios where keyAntiquantScale and valueAntiquantScale are non-null, the s of query must be less than or equal to 16.
      - In scenarios where keyAntiquantScale and valueAntiquantScale are non-null, the dtype of query must be BFLOAT16, the dtype of key and value must be INT8, and the dtype of the output must be BFLOAT16.
      - In scenarios where keyAntiquantScale and valueAntiquantScale are non-null, the tensorlist, left padding, and page attention features are not supported.
    <!-- end id34 -->
    <!-- npu="950" id35 -->
    - <term>Ascend 950PR&950DT products</term>:
      - Except for the use case where keyAntiquantMode is 0 and valueAntiquantMode is 1, keyAntiquantMode and valueAntiquantMode must keep consistent.
      - keyAntiquantScale and valueAntiquantScale must be both empty or both non-empty; keyAntiquantOffset and valueAntiquantOffset must be both empty or both non-empty.
      - When KeyAntiquantScale and valueAntiquantScale are both non-empty, their shapes must keep consistent except for the use case where keyAntiquantMode is 0 and valueAntiquantMode is 1; when keyAntiquantOffset and valueAntiquantOffset are both non-empty, their shapes must keep consistent except for the use case where keyAntiquantMode is 0 and valueAntiquantMode is 1.
      - Nine modes are supported: per-channel, per-tensor, per-token, per-tensor stacked with per-head, per-token stacked with per-head, per-token using page attention mode to manage scale/offset, per-token stacked with per-head using page attention mode to manage scale/offset, key supporting per-channel stacked with value supporting per-token, and per-token-group. In the following, N denotes numKeyValueHeads.
        - per-channel mode: The shapes of the two parameters can be (1, N, 1, D), (1, N, D), (1, H), (N, 1, D), (N, D), or (H). The parameter data type is the same as the query data type. This mode is supported when the key and value data types are INT8, INT4 (INT32), HIFLOAT8, or FLOAT8_E4M3FN. When the key and value data types are HIFLOAT8 or FLOAT8_E4M3FN, antiQuantOffset is not supported.
        - per-tensor mode: The shape of both parameters is (1). The data type is the same as the query data type. This mode is supported when the key and value data types are INT8.
        - per-token mode: The shapes of the two parameters can be (1, B, S) or (B, S), and the data type is fixed to FLOAT32. This mode is supported when the key and value data types are INT8 or INT4(INT32).
        - per-tensor plus per-head mode: The shapes of the two parameters are both (N), with the data type the same as the query data type. Supported when the key and value data types are INT8.
        - key per-channel plus value per-token mode: For key per-channel, the shapes of the two parameters can be (1, N, 1, D), (1, N, D), (1, H), (N, 1, D), (N, D), or (H), and the parameter data type is the same as the query data type. For value per-token, the shapes of the two parameters are both (1, B, S) and the data type is fixed to FLOAT32. Supported when the key and value data types are INT8 or INT4 (INT32). When the key and value data types are INT8, the query and output support only FLOAT16.
        - per-token-group mode: The shape of antiquantScale is (1, B, N, S, D/32), and the data type is fixed to FLOAT8_E8M0. antiquantOffset is not supported. This mode is supported when the data types of key and value are FLOAT4_E2M1.
        - per-token combined with per-head mode: The shapes of both parameters are (B, N, S), and the data type is fixed to FLOAT32. Supported when the key and value data types are INT8 or INT4(INT32).
        - per-token mode using page attention to manage scale/offset mode: The shapes of both parameters are (blocknum, blocksize), and the data type is fixed to FLOAT32. Supported when the key and value data types are INT8.
        - per-token superimposed with per-head mode using page attention to manage scale/offset mode: The shape of both parameters is (blocknum, N, blocksize), the data type is fixed to FLOAT32, and it is supported when the data type of key and value is INT8.
      - When pseudo-quantization parameters and KV-separated quantization parameters are passed in at the same time, the KV-separated quantization parameters take precedence.
      - The INT4 (INT32) pseudo-quantization scenario supports only KV pseudo-quantization parameter separation, specifically including:
        - per-channel mode;
        - per-token mode;
        - per-token plus per-head mode;
        - key supports per-channel plus value supports per-token mode.
      - Some pseudo-quantization scenarios do not support post-quantization
        - <term>Atlas A2 products</term>: Post-quantization is not supported in INT4 (INT32) pseudo-quantization scenarios.
        - <term>Ascend 950PR&950DT products</term>: Post-quantization is not supported in INT4 (INT32) and FLOAT4_E2M1 pseudo-quantization scenarios.

    <!-- end id35 -->

- **When Q_S equals 1**:

  - The function usage restrictions for query, key, and value inputs are as follows:
    - The B-axis must be less than or equal to 65536, the N-axis must be less than or equal to 256, and the D-axis must be less than or equal to 512.
    - The scenario where query, key, and value input types are all INT8 is not supported yet.
    - In INT4 (INT32) pseudo-quantization scenarios, the aclnn single-operator call supports KV INT4 input or INT4 concatenated into INT32 input. (It is recommended to use dynamicQuant to generate data in INT4 format, because dynamicQuant treats one INT32 as containing eight INT4 values.)
    - In the INT4 (INT32) pseudo-quantization scenario, if KV INT4 is packed into INT32 input, then N, D, or H of KV is one-eighth of the actual value (same for prefix).
    - key and value have restrictions on the D axis under specific data types.

      <!-- npu="910b" id36 -->
      - <term>Atlas A2 products</term>: When the key and value input type is INT4 (INT32), the D axis must be 64-aligned (INT32 only supports D 8-aligned).
      <!-- end id36 -->
      <!-- npu="950" id37 -->
      - <term>Ascend 950PR&950DT products</term>: When the key and value input type is FLOAT4_E2M1/INT4 (INT32), the D axis of query and the D axis of key and value must be 64-aligned (INT32 only supports D 8-aligned for key and value).
      <!-- end id37 -->

  - Page attention scenario:
    - The necessary condition for enabling page attention is that blocktable exists and is valid, and key and value are arranged in contiguous memory according to the indexes in blocktable. In this use case, the inputLayout parameters of key and value are invalid.
    - blockSize is a user-defined parameter, and its value affects the performance of page attention. In the use case where page attention is enabled, blockSize must be a non-zero value and cannot exceed 512. Generally, page attention improves throughput but reduces performance.
    - In the page attention scenario, when the inputLayout of query is BNSD, the KV cache layout supports two formats: BnBsH (blocknum, blocksize, H) and BnNBsD (blocknum, KV_N, blocksize, D). When the inputLayout of query is BSH or BSND, the KV cache layout only supports one format: BnBsH. blocknum cannot be less than the sum of the number of blocks in each batch calculated based on actualSeqLengthsKv and blockSize. The shapes of key and value must be consistent.
    - In the page attention scenario, the performance of the KV cache layout BnNBsD is generally better than that of BnBsH. It is recommended to prefer the BnNBsD format.
    - In the use case where page attention is enabled, if the input kv cache layout is BnBsH and numKvHeads * headDim exceeds 64k, the input is intercepted with an error due to hardware instruction constraints. This can be resolved by enabling GQA (reducing numKvHeads) or changing the kv cache layout to BnNBsD.
    - Page attention does not support the tensorlist scenario or the left-padding scenario.
    - In the page attention scenario, actualSeqLengthsKv must be passed.
    - In the page attention scenario, blockTable must be two-dimensional, with the first dimension length equal to B and the second dimension length not less than maxBlockNumPerSeq (maxBlockNumPerSeq is the number of blocks corresponding to the maximum actualSeqLengthsKv in each batch).
    - In the use case where page attention is enabled, the input S must be greater than or equal to maxBlockNumPerSeq * blockSize in the following use cases.
      - Attention mask is enabled, for example, the mask shape is \(B, 1, 1, S\).
      - pseShift is enabled, e.g. the shape of pseShift is \(B, N, 1, S\).
      - Pseudo-quantization per-token mode is enabled: the shapes of the input parameters antiquantScale and antiquantOffset are both \(2, B, S\).
      - Per-token stacked with per-head mode is enabled: the shapes of the two parameters are both \(B, N, S\), the data type is fixed to FLOAT32, and it is supported when the data types of key and value are INT8 or INT4\(INT32\).
      - Per-token-group mode is enabled: the shape of antiquantScale is \(1, B, N, S, D/32\),, the data type is fixed to FLOAT8_E8M0, and antiquantOffset is not supported. It is supported when the data types of key and value are FLOAT4_E2M1.
  - kv left-padding scenario:

    <!-- npu="910b" id38 -->
    - <term>Atlas A2 products</term>: The scenario where Q is BF16/FP16 and KV is INT4(INT32) is not supported.
    <!-- end id38 -->
    <!-- npu="950" id39 -->
    - <term>Ascend 950PR&950DT products</term>: The scenario where Q is BF16/FP16 and KV is INT4(INT32) is supported, and there is no restriction on QKV data types.
    <!-- end id39 -->
    - The formula for the kvCache copy start point is: KV_S - kvPaddingSize - actualSeqLengthsKv. The formula for the kvCache copy end point is: KV_S - kvPaddingSize. If the kvCache copy start point or end point is less than 0, the returned data result is all zeros.
    - If kvPaddingSize is less than 0, it is set to 0.
    - It must be enabled together with the actualSeqLengthsKv parameter; otherwise, the kv right-padding use case applies by default.
    - PageAttention and tensorlist are not supported; otherwise, the default is the kv right-padding scenario.
    - When enabled together with the attenMask parameter, ensure that attenMask is semantically correct, that is, it can correctly mask invalid data. Otherwise, precision issues will be introduced.
  - pseShift function usage restrictions as follows:
    - The data type of pseShift must be consistent with that of query.
  - KV pseudo-quantization parameter separation:
    - Except for the use case where keyAntiquantMode is 0 and valueAntiquantMode is 1, keyAntiquantMode and valueAntiquantMode must be kept consistent
    - keyAntiquantScale and valueAntiquantScale are either both empty or both non-empty; keyAntiquantOffset and valueAntiquantOffset are either both empty or both non-empty
    - When both KeyAntiquantScale and valueAntiquantScale are non-empty, except for the use case where keyAntiquantMode is 0 and valueAntiquantMode is 1, their shapes must be kept consistent; when both keyAntiquantOffset and valueAntiquantOffset are non-empty, except for the use case where keyAntiquantMode is 0 and valueAntiquantMode is 1, their shapes must be kept consistent

    <!-- npu="910b" id40 -->
    - <term>Atlas A2 products</term>: Supports eight modes: per-channel, per-tensor, per-token, per-tensor plus per-head, per-token plus per-head, per-token with page attention mode for scale/offset management, per-token plus per-head with page attention mode for scale/offset management, and key per-channel plus value per-token. In the following, N denotes numKeyValueHeads.
      - Per-channel mode: The shapes of the two parameters can be \(1, N, 1, D\), \(1, N, D\), or \(1, H\). The parameter data type is the same as the query data type. Supported when the key and value data types are INT8 or INT4 (INT32).
      - Per-tensor mode: The shapes of both parameters are \(1\). The data type is the same as the query data type. Supported when the key and value data types are INT8.
      - Per-token mode: The shapes of both parameters are \(1, B, S\). The data type is fixed to FLOAT32. Supported when the key and value data types are INT8 or INT4 (INT32).
      - Per-tensor plus per-head mode: The shapes of both parameters are \(N\). The data type is the same as the query data type. Supported when the key and value data types are INT8.
      - key per-channel and value per-token mode: For key per-channel, the shapes of the two parameters can be \(1, N, 1, D\), \(1, N, D\), or \(1, H\), and the parameter data type is the same as the query data type. For value per-token, the shapes of the two parameters are both \(1, B, S\) and the data type is fixed to FLOAT32. This mode is supported when the key and value data types are INT8 or INT4(INT32). When the key and value data types are INT8, only FLOAT16 is supported for the query and attentionOut data types.
    <!-- end id40 -->
    <!-- npu="950" id41 -->
    - <term>Ascend 950PR&950DT products</term>: Supports nine modes: per-channel, per-tensor, per-token, per-tensor with per-head, per-token with per-head, per-token using page attention mode to manage scale/offset, per-token with per-head using page attention mode to manage scale/offset, key per-channel with value per-token, and per-token-group. In the following, N denotes numKeyValueHeads.
      - per-channel mode: The shapes of the two parameters can be (1, N, 1, D), (1, N, D), or (1, H). The parameter data type is the same as the query data type. This mode is supported when the key and value data types are INT8, INT4(INT32), HIFLOAT8, or FLOAT8_E4M3FN. When the key and value data types are HIFLOAT8 or FLOAT8_E4M3FN, carrying antiQuantOffset is not supported.
      - per-tensor mode: The shapes of the two parameters are both (1), and the data type is the same as the query data type. This mode is supported when the key and value data types are INT8 or INT4(INT32).
      - per-token mode: The shapes of the two parameters can be (1, B, S) or (B, S), and the data type is fixed to FLOAT32. This mode is supported when the key and value data types are INT8 or INT4(INT32).
      - per-tensor combined with per-head mode: The shapes of both parameters are (N), and the data type is the same as the query data type. Supported when the key and value data types are INT8 or INT4(INT32).
      - key supports per-channel combined with value supports per-token mode: For key supporting per-channel, the shapes of the two parameters can be (1, N, 1, D), (1, N, D), or (1, H), and the parameter data type is the same as the query data type. For value supporting per-token, the shapes of both parameters are (1, B, S) and the data type is fixed to FLOAT32. Supported when the key and value data types are INT8 or INT4(INT32). When the key and value data types are INT8, the query and output support only FLOAT16.
      - Per-token-group mode: the shape of antiquantScale is (1, B, N, S, D/32), the data type is fixed to FLOAT8_E8M0, and antiquantOffset is not supported. It is supported when the data types of key and value are FLOAT4_E2M1.
      - per-token combined with per-head mode: The shapes of both parameters are (B, N, S), and the data type is fixed to FLOAT32. Supported when the key and value data types are INT8 or INT4(INT32).
      - per-token mode using page attention to manage scale/offset mode: The shapes of both parameters are (blocknum, blocksize), and the data type is fixed to FLOAT32. Supported when the key and value data types are INT8.
      - per-token superimposed with per-head mode using page attention to manage scale/offset mode: The shape of both parameters is (blocknum, N, blocksize), the data type is fixed to FLOAT32, and it is supported when the data type of key and value is INT8.
    <!-- end id41 -->
    - When both pseudo-quantization parameters and KV-separated quantization parameters are passed in, the KV-separated quantization parameters take precedence.
    - The INT4 (INT32) pseudo-quantization scenario supports only KV pseudo-quantization parameter separation, specifically including:
      - per-tensor mode;
      - per-channel mode;
      - per-token mode;
      - per-tensor plus per-head mode;
      - per-token plus per-head mode;
      - key supports per-channel plus value supports per-token mode.
    - Some pseudo-quantization scenarios do not support post-quantization

      <!-- npu="910b" id42 -->
      - <term>Atlas A2 products</term>: Post-quantization is not supported in INT4 (INT32) pseudo-quantization scenarios.
      <!-- end id42 -->
      <!-- npu="950" id43 -->
      - <term>Ascend 950PR&950DT products</term>: Post-quantization is not supported in INT4 (INT32) and FLOAT4_E2M1 pseudo-quantization scenarios.
      <!-- end id43 -->

  - Prefix-related parameter constraints:
    - keySharedPrefix and valueSharedPrefix must either both be null or both be non-null.
    - When both keySharedPrefix and valueSharedPrefix are non-null, keySharedPrefix, valueSharedPrefix, key, and value must have the same number of dimensions and keep their dtypes consistent.
    - When both keySharedPrefix and valueSharedPrefix are not empty, the first dimension (batch) of keySharedPrefix's shape must be 1. In BNSD and BSND layouts, the N and D axes must be consistent with those of key. In BSH layout, H must be consistent with that of key. The same applies to valueSharedPrefix. The S of keySharedPrefix and valueSharedPrefix must be equal.
    - When actualSharedPrefixLen exists, its shape must be [1], and its value cannot be greater than the S of keySharedPrefix and valueSharedPrefix.
    - The result of adding the S of the common prefix to the S of key or value must satisfy the original S constraints of key or value.

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <iostream>
#include <vector>
#include <math.h>
#include <cstring>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_fused_infer_attention_score_v2.h"

using namespace std;

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
  int64_t shapeSize = 1;
  for (auto i : shape) {
      shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
  // Boilerplate, AscendCL initialization.
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void** deviceAddr,
                  aclDataType dataType, aclTensor** tensor)
{
  auto size = GetShapeSize(shape) * aclDataTypeSize(dataType);
  // Call aclrtMalloc to allocate device-side memory.
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

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main()
{
  // 1. (Boilerplate) Device/stream initialization. See the AscendCL external API list.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Custom construction is required based on the API.
  int32_t batchSize = 1;
  int32_t numHeads = 2;
  int32_t numKeyValueHeads = 2;
  int32_t sequenceLengthQ = 1;
  int32_t sequenceLengthKV = 512;
  int32_t headDims = 128;
  std::vector<int64_t> queryShape = {batchSize, numHeads, sequenceLengthQ, headDims};           // BNSD
  std::vector<int64_t> keyShape = {batchSize, numKeyValueHeads, sequenceLengthKV, headDims};    // BNSD
  std::vector<int64_t> valueShape = {batchSize, numKeyValueHeads, sequenceLengthKV, headDims};  // BNSD
  std::vector<int64_t> attenMaskShape = {batchSize, 1, sequenceLengthQ, sequenceLengthKV};      // B 1 S1 S2
  std::vector<int64_t> outShape = {batchSize, numHeads, sequenceLengthQ, headDims};             // BNSD
  void* queryDeviceAddr = nullptr;
  void* keyDeviceAddr = nullptr;
  void* valueDeviceAddr = nullptr;
  void* attenMaskDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* queryTensor = nullptr;
  aclTensor* keyTensor = nullptr;
  aclTensor* valueTensor = nullptr;
  aclTensor* attenMaskTensor = nullptr;
  aclTensor* outTensor = nullptr;
  int64_t queryShapeSize = GetShapeSize(queryShape);          // BNSD
  int64_t keyShapeSize = GetShapeSize(keyShape);              // BNSD
  int64_t valueShapeSize = GetShapeSize(valueShape);          // BNSD
  int64_t attenMaskShapeSize = GetShapeSize(attenMaskShape);  // B 1 S1 S2
  int64_t outShapeSize = GetShapeSize(outShape);              // BNSD
  std::vector<op::fp16_t> queryHostData(queryShapeSize, 1.0);
  std::vector<op::fp16_t> keyHostData(keyShapeSize, 1.0);
  std::vector<op::fp16_t> valueHostData(valueShapeSize, 1.0);
  std::vector<int8_t> attenMaskHostData(attenMaskShapeSize, 0);
  std::vector<op::fp16_t> outHostData(outShapeSize, 1.0);

  // Create the query aclTensor.
  ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the key aclTensor.
  ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &keyTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  int kvTensorNum = 1;
  aclTensor* tensorsOfKey[kvTensorNum];
  tensorsOfKey[0] = keyTensor;
  auto tensorKeyList = aclCreateTensorList(tensorsOfKey, kvTensorNum);
  // Create the value aclTensor.
  ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &valueTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclTensor* tensorsOfValue[kvTensorNum];
  tensorsOfValue[0] = valueTensor;
  auto tensorValueList = aclCreateTensorList(tensorsOfValue, kvTensorNum);
  // Create the attenMask aclTensor.
  ret = CreateAclTensor(attenMaskHostData, attenMaskShape, &attenMaskDeviceAddr, aclDataType::ACL_INT8, &attenMaskTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &outTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> actualSeqlenVector = {sequenceLengthKV};
  auto actualSeqLengths = aclCreateIntArray(actualSeqlenVector.data(), actualSeqlenVector.size());

  double scaleValue = 1 / sqrt(headDims); // 1 / sqrt(d)
  int64_t preTokens = 2147483647;
  int64_t nextTokens = 2147483647;
  string sLayerOut = "BNSD";
  char layerOut[sLayerOut.length()+1];
  strcpy(layerOut, sLayerOut.c_str());
  int64_t sparseMode = 0;
  int64_t innerPrecise = 0;
  int blockSize = 0;
  int antiquantMode = 0;
  bool softmaxLseFlag = false;
  int keyAntiquantMode = 0;
  int valueAntiquantMode = 0;

  // 3. Call the CANN operator library API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API.
  ret = aclnnFusedInferAttentionScoreV2GetWorkspaceSize(
      queryTensor, tensorKeyList, tensorValueList, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, numHeads, scaleValue, preTokens, nextTokens, layerOut, numKeyValueHeads, sparseMode,
      innerPrecise, blockSize, antiquantMode, softmaxLseFlag, keyAntiquantMode, valueAntiquantMode, outTensor,
      nullptr, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedInferAttentionScoreV2GetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API.
  ret = aclnnFusedInferAttentionScoreV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedInferAttentionScoreV2 failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize and wait for task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value, copy the result on the device to the host, and modify the code based on the specific API definition.
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
  aclDestroyTensor(attenMaskTensor);
  aclDestroyTensor(outTensor);
  aclDestroyIntArray(actualSeqLengths);
  aclrtFree(queryDeviceAddr);
  aclrtFree(keyDeviceAddr);
  aclrtFree(valueDeviceAddr);
  aclrtFree(attenMaskDeviceAddr);
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
