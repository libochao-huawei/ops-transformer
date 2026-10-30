# aclnnFusedInferAttentionScoreV3

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T09:49:46.057Z -->

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

- Description: A FlashAttention operator adapted to decode & prefill scenarios, supporting both the prefill computation scenario (PromptFlashAttention) and the decode computation scenario (IncreFlashAttention). Compared with FusedInferAttentionScoreV2, this API adds the queryRopeOptional, keyRopeOptional, and keyRopeAntiquantScaleOptional parameters.

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

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnFusedInferAttentionScoreV3GetWorkspaceSize" API to obtain the input parameters and calculate the workspace size required for the computation process, and then call the "aclnnFusedInferAttentionScoreV3" API to execute the computation.

```cpp
aclnnStatus aclnnFusedInferAttentionScoreV3GetWorkspaceSize(
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
    const aclTensor     *queryRopeOptional,
    const aclTensor     *keyRopeOptional,
    const aclTensor     *keyRopeAntiquantScaleOptional,
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
aclnnStatus aclnnFusedInferAttentionScoreV3(
     void                *workspace,
     uint64_t             workspaceSize,
     aclOpExecutor       *executor,
     const aclrtStream    stream)
```

## aclnnFusedInferAttentionScoreV3GetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1625px"><colgroup>
    <col style="width: 247px">
    <col style="width: 132px">
    <col style="width: 232px">
    <col style="width: 293px">
    <col style="width: 185px">
    <col style="width: 119px">
    <col style="width: 272px">
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
        <td>FLOAT16, BFLOAT16, INT8, HIFLOAT8, FLOAT8_E4M3FN</td>
        <td>ND</td>
        <td>See the inputLayout parameter.</td>
        <td>×</td>
      </tr>
      <tr>
        <td>key</td>
        <td>Input</td>
        <td>Input K in the formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16, INT8, HIFLOAT8, FLOAT8_E4M3FN, INT4(INT32), FLOAT4_E2M1</td>
        <td>ND</td>
        <td>See the inputLayout parameter</td>
        <td>×</td>
      </tr>
      <tr>
        <td>value</td>
        <td>Input</td>
        <td>Input V in the formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16, INT8, HIFLOAT8, FLOAT8_E4M3FN, INT4(INT32), FLOAT4_E2M1</td>
        <td>ND</td>
        <td>See the inputLayout parameter.</td>
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
        <td>Recommended input shapes: (B,Q_N,Q_S,KV_S), (1,Q_N,Q_S,KV_S).</td>
        <td>×</td>
      </tr>
      <tr>
        <td>attenMaskOptional</td>
        <td>Input</td>
        <td>Mask matrix.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>Pass nullptr if this function is not used.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>BOOL, INT8, UINT8</td>
        <td>ND</td>
        <td>            <ul>
                <li>When sparseMode = 2, 3, or 4, the shape of attenMaskOptional must be (2048,2048), (1,2048,2048), or (1,1,2048,2048).</li>
                <li>For other sparseMode values, if Q_S is not 1, the recommended shapes are (Q_S,KV_S); (B,Q_S,KV_S); (1,Q_S,KV_S); (B,1,Q_S,KV_S); (1,1,Q_S,KV_S).</li>
                <li>For other sparseMode values, if Q_S is 1, the recommended shapes are (B,KV_S); (B,1,KV_S); (B,1,1,KV_S).</li>
            </ul>
            </td>
        <td>×</td>
      </tr>
      <tr>
        <td>actualSeqLengthsOptional</td>
        <td>Input</td>
        <td>Effective sequence length of query in each batch.</td>
        <td><ul><li>Pass nullptr if the sequence length is not specified.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actualSeqLengthsKvOptional</td>
        <td>Input</td>
        <td>Effective sequence length of key/value in each batch.</td>
        <td><ul><li>If the sequence length is not specified, pass nullptr.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
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
          <li>Per-tensor is supported.</li>
              <li>Pass nullptr if this function is not used.</li>
              <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>UINT64, FLOAT32</td>
        <td>ND</td>
        <td>See <a href="#INT8">int8 Quantization-Related Comprehensive Constraints on the Number of Input Parameters and Input/Output Data Formats</a>.</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantScale1Optional</td>
        <td>Input</td>
        <td>Quantization Factor before BMM2.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>Supports per-tensor. </li>
            <li>Pass nullptr if this feature is not used.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>See <a href="#INT8">int8 Quantization-Related Comprehensive Constraints on the Number of Input Parameters and Input/Output Data Formats</a>.</td>
        <td>-</td>
      </tr>
      <tr>
        <td>deqScale2Optional</td>
        <td>Input</td>
        <td>Dequantization factor after BMM2.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>per-tensor mode is supported. </li>
           <li>Pass nullptr if this feature is not used.</li>
            <li>See <a href="#constraints">Constraints</a> for comprehensive constraints.</li></ul></td>
        <td>UINT64, FLOAT32</td>
        <td>ND</td>
        <td>See <a href="#INT8">int8 Quantization-Related Comprehensive Constraints on the Number of Input Parameters and Input/Output Data Formats</a>.</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantScale2Optional</td>
        <td>Input</td>
        <td>Quantization factor of the output.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>Supports per-tensor and per-channel. </li>
            <li>Can be set to nullptr when this function is not used.</li>
             <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT32, BFLOAT16</td>
        <td>ND</td>
        <td>per-channel format: When the output layout is BSH, the product of all dimensions of quantScale2 must equal H; for other layouts, the product must equal N*D. (When the output layout is BSH, the recommended shape of quantScale2 is [1,1,H] or [H]; for BNSD output, [1,N,1,D] or [N,D] is recommended; for BSND output, [1,1,N,D] or [N,D] is recommended; for TND output, [1,N,D] or [N,D] is recommended.)</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantOffset2Optional</td>
        <td>Input</td>
        <td>Quantization offset of the output.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>Per-tensor and per-channel are supported. </li>
            <li>Pass nullptr if this function is not used.</li>
             <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT32, BFLOAT16</td>
        <td>ND</td>
        <td>Same as quantScale2Optional.</td>
        <td>-</td>
      </tr>
      <tr>
        <td>antiquantScaleOptional</td>
        <td>Input</td>
        <td>Pseudo-quantization factor.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>Per-tensor, per-channel, and per-token are supported. </li>
            <li>Pass nullptr if this function is not used.</li>
            <li>The KV pseudo-quantization parameter separation mode is recommended.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>See <a href="#AntiQuant">antiquantScale and antiquantOffset pseudo-quantization parameter constraints</a></td>
        <td>-</td>
      </tr>
        <tr>
      <td>antiquantOffsetOptional</td>
        <td>Input</td>
        <td>Pseudo-quantization offset.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>Supports per-tensor, per-channel, and per-token. </li>
            <li>The shape must be the same as that of antiquantScaleOptional when this parameter is used.</li>
            <li>nullptr can be passed if this parameter is not used.</li>
            <li>The KV pseudo-quantization parameter separation mode is recommended.</li>
            <li>For comprehensive constraints, see <a href="#Constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>Same as antiquantScaleOptional</td>
        <td>-</td>
      </tr>
    <tr>
       <td>blockTableOptional</td>
        <td>Input</td>
        <td>Block mapping table used for KV storage in PageAttention.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>Pass nullptr if this feature is not used.</li></ul></td>
        <td>INT32</td>
        <td>ND</td>
        <td>The first dimension length must be equal to B, and the second dimension length cannot be less than maxBlockNumPerSeq (maxBlockNumPerSeq is the number of blocks corresponding to the maximum actualSeqLengthsKv among different batches).</td>
        <td>-</td>
      </tr>
      <tr>
       <td>queryPaddingSizeOptional</td>
        <td>Input</td>
        <td>Indicates whether the data of each batch in Query is right-aligned, and the number of right-aligned values.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>Only Q_S greater than 1 is supported. This parameter is invalid in other use cases.</li>
            <li>If this function is not used, pass nullptr.</li></ul></td>
        <td>INT64</td>
        <td>ND</td>
        <td>(1)</td>
        <td>-</td>
      </tr>
      <tr>
       <td>kvPaddingSizeOptional</td>
        <td>Input</td>
        <td>Indicates whether the data of each batch in key/value is right-aligned, and the number of right-aligned values.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>Pass in nullptr if this feature is not used.</li></ul></td>
        <td>INT64</td>
        <td>ND</td>
        <td>(1)</td>
        <td>-</td>
      </tr>
      <tr>
       <td>keyAntiquantScaleOptional</td>
        <td>Input</td>
        <td>Indicates the dequantization factor of key when kv pseudo-quantization parameters are separated.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>Pass in nullptr if this feature is not used.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT32, FLOAT8_E8M0</td>
        <td>ND</td>
        <td>See <a href="#constraints">Constraints</a></td>
        <td>-</td>
      </tr>
        <tr>
       <td>keyAntiquantOffsetOptional</td>
        <td>Input</td>
        <td>Indicates the dequantization offset of key when the KV pseudo-quantization parameters are separated.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>When used, the shape must be the same as that of keyAntiquantScaleOptional.</li>
            <li>Pass nullptr when this function is not used.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>See <a href="#constraints">Constraints</a>.</td>
        <td>-</td>
      </tr>
         <tr>
       <td>valueAntiquantScaleOptional</td>
        <td>Input</td>
        <td>Indicates the dequantization factor of value when the KV pseudo-quantization parameters are separated.</td>
        <td><ul><li>Empty tensors are not supported.</li>
            <li>nullptr can be passed in when this function is not used.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT32, FLOAT8_E8M0</td>
        <td>ND</td>
        <td>See <a href="#constraints">Constraints</a>.</td>
        <td>-</td>
      </tr>
         <tr>
       <td>valueAntiquantOffsetOptional</td>
        <td>Input</td>
        <td>When kv pseudo-quantization parameters are separated, this parameter indicates the Dequantization factor of value.</td>
        <td><ul><li>Empty tensors are not supported.</li>
        <li>When used, the shape must be the same as that of valueAntiquantScaleOptional.</li>
            <li>When this function is not used, nullptr can be passed in.</li>
            <li>For Comprehensive Constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>See <a href="#constraints">Constraints</a>.</td>
        <td>-</td>
      </tr>
       <tr>
       <td>keySharedPrefixOptional</td>
        <td>Input</td>
        <td>Parameters of the system prefix part of Key in the attention structure.</td>
        <td><ul><li>Empty tensors are not supported.</li>
            <li>You can pass in nullptr when this function is not used.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT16, BFLOAT16, INT8</td>
        <td>ND</td>
        <td>            <ul>
                <li>When input_layout is BSH, the shape is (1, prefix_S, H=KV_N*KV_D).</li>
                <li>When input_layout is BSND, the shape is (1, prefix_S, KV_N, KV_D).</li>
                <li>When input_layout is BNSD or BNSD_BSND, the shape is (1, KV_N, prefix_S, KV_D).</li>
            </ul>
        </td>
        <td>×</td>
      </tr>
       <tr>
       <td>valueSharedPrefixOptional</td>
        <td>Input</td>
        <td>Input of the system prefix part of Value in the attention structure.</td>
        <td><ul><li>Empty tensors are not supported.</li>
            <li>You can pass in nullptr when this function is not used.</li>
            <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT16, BFLOAT16, INT8</td>
        <td>ND</td>
        <td>            <ul>
                <li>If input_layout is BSH, the shape is (1, prefix_S, H=KV_N*KV_D).</li>
                <li>If input_layout is BSND, the shape is (1, prefix_S, KV_N, KV_D).</li>
                <li>If input_layout is BNSD or BNSD_BSND, the shape is (1, KV_N, prefix_S, KV_D).</li>
            </ul>
        </td>
        <td>×</td>
    </tr>
      <tr>
        <td>actualSharedPrefixLenOptional</td>
        <td>Input</td>
        <td>Valid Sequence Length of keySharedPrefix/valueSharedPrefix.</td>
        <td><ul><li>If this function is not used, nullptr can be passed in, indicating that the length is the same as the s length of keySharedPrefix/valueSharedPrefix.</li>
         <li>The valid Sequence Length in this input parameter must not be greater than the Sequence Length in keySharedPrefix/valueSharedPrefix.</li>
         <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>(1)</td>
        <td>-</td>
      </tr>
      <tr>
        <td>queryRopeOptional</td>
        <td>Input</td>
        <td>Rope information of query in the MLA structure.</td>
        <td><ul><li>Empty tensors are not supported.</li>
                <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>d in the shape of queryRope is 64, and remaining dimensions are the same as those of query</td>
        <td>×</td>
      </tr>
      <tr>
        <td>keyRopeOptional</td>
        <td>Input</td>
        <td>rope information of the key in the MLA structure.</td>
        <td><ul><li>Empty tensors are not supported.</li>
                <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>The value of d in the shape of keyRope is 64, and the remaining dimensions are the same as those of key.</td>
        <td>×</td>
      </tr>
       <tr>
        <td>keyRopeAntiquantScaleOptional</td>
        <td>Input</td>
        <td>Dequantization factor of the rope information of the key in the MLA structure.</td>
        <td><ul><li>Empty tensors are not supported.</li>
                <li>Reserved parameter, which does not take effect in the current version. Pass in nullptr.</li></ul></td>
        <td>FLOAT16, BFLOAT16</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>numHeads</td>
        <td>Input</td>
        <td>Number of heads in query.</td>
        <td>In BNSD, BSND, BNSD_BSND, and TND use cases, this value must be the same as the N-axis shape value of query in shape. Otherwise, an execution error occurs.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>scaleValue</td>
        <td>Input</td>
        <td>Reciprocal of the square root of d in the formula.</td>
        <td><ul><li>The data type and the data type of query must comply with the data type deduction rules. </li>
            <li>1.0 is recommended if not specifically specified. </li></ul></td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>preTokens</td>
        <td>Input</td>
        <td>Used for sparse computation. Indicates that attention needs to be associated with the preceding several tokens.</td>
          <td><ul><li>If not specified, 2147483647 is recommended.</li>
              <li>This parameter is invalid when Q_S is 1.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>nextTokens</td>
        <td>Input</td>
        <td>Indicates the number of subsequent tokens that attention computation needs to be associated with.</td>
        <td><ul><li>If not specifically specified, 2147483647 is recommended.</li>
            <li>This parameter does not take effect when Q_S is 1.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>inputLayout</td>
        <td>Input</td>
        <td>Indicates the data layout format of the input query, key, and value.</td>
        <td><ul><li>Currently, BSH, BSND, BNSD, BNSD_BSND (when the input is BNSD, the output format is BSND, and only Q_S greater than 1 is supported), BSH_NBSD, BSND_NBSD, BNSD_NBSD (when the output format is NBSD, only Q_S greater than 1 and less than or equal to 16 is supported), TND, TND_NTD, and NTD_TND are supported. For comprehensive constraints on TND-related use cases, see <a href="#constraints">Constraints</a>. If not specifically specified, "BSH" is recommended.</li></ul>
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
        <td><ul><li>If not specifically specified, 0 is recommended, indicating that key/value and query have the same number of heads.</li></ul>
            <ul><li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>sparseMode</td>
        <td>Input</td>
        <td>The sparse mode.</td>
        <td>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>innerPrecise</td>
        <td>Input</td>
        <td>Specifies whether to select high precision or high performance.</td>
        <td>For Comprehensive Constraints, see <a href="#constraints">Constraints</a>.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>blockSize</td>
        <td>Input</td>
        <td>Maximum number of tokens in each KV storage block in PageAttention.</td>
        <td>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>antiquantMode</td>
        <td>Input</td>
        <td>Pseudo quantization mode.</td>
          <td><ul><li>Passing 0 indicates per-channel (per-channel includes per-tensor).</li>
              <li>Passing 1 indicates per-token.</li>
              <li>Passing 0 is recommended if not specifically specified.</li>
              <li>If Q_S is 1, passing a value other than 0 or 1 causes an execution exception. This parameter is invalid if Q_S is greater than or equal to 2.</li>
              <li>The KV pseudo-quantization parameter separation mode is recommended.</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>softmaxLseFlag</td>
        <td>Input</td>
        <td>Whether to output softmax_lse.</td>
          <td><ul><li>Supports external splitting along the S axis (adding output).</li>
              <li>false is recommended if not specifically specified by the user.</li></ul></td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>keyAntiquantMode</td>
        <td>Input</td>
        <td>Pseudo quantization mode of key.</td>
        <td><ul>
            <li>If not specifically specified, it is recommended to pass in 0.</li>
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
        <td>The pseudo quantization mode of value.</td>
          <td><ul><li>This value must be consistent with keyAntiquantMode, except when keyAntiquantMode is 0 and valueAntiquantMode is 1.</li>
              <li>It is recommended to pass in 0 if not specifically specified.</li>
               <li>For comprehensive constraints, see <a href="#constraints">Constraints</a>.</li></ul></td>
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
        <td>FLOAT16, BFLOAT16, INT8</td>
        <td>ND</td>
        <td>The D dimension of this input parameter is the same as the D dimension of value, and the remaining dimensions must be the same as the shape of the input parameter query.</td>
        <td>-</td>
      </tr>
      <tr>
        <td>softmaxLse</td>
        <td>Output</td>
        <td>Result of query multiplied by key computed by the ring attention algorithm.</td>
        <td>For Comprehensive Constraints, see <a href="#constraints">Constraints</a>.</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>If softmaxLseFlag is True, the shape must be [B, N, Q_S, 1] in general. If inputLayout is TND/NTD_TND, the shape must be [T, N, 1].</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the workspace size to be requested by the user on the Device.</td>
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
      <td>An exception occurs when the API calls the npu runtime interface for memory operations.</td>
    </tr>
  </tbody>
  </table>

## aclnnFusedInferAttentionScoreV3

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
      <td>The workspace memory address allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>The workspace size allocated on the device, obtained by the first-phase API aclnnFusedInferAttentionScoreV3GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnFusedInferAttentionScoreV3 defaults to a deterministic implementation.
- When using this API with PyTorch, ensure that the versions of CANN-related packages and PyTorch-related packages are compatible.

- Handling of null input parameters: The operator internally checks whether the query parameter is null. If it is null, the operator returns directly. If the query parameter is a non-null tensor and the key and value parameters are null tensors (i.e., S2 is 0), attentionOut is filled with all zeros. When attentionOut is a null tensor, the AscendCLNN framework handles it. For other input parameters marked as "nullable" in the parameter descriptions above, no processing is performed when they are null pointers.

- The shapes of the corresponding tensors in parameters key and value must be identical. In non-continuous use cases, the batch in the tensorlist of key and value can only be 1, the number equals the B of query, and N and D must be equal. Due to the tensorlist limitation, B cannot be greater than 256 in non-continuous use cases.

- When the data type of attenMask is INT8 or UINT8, the values in its tensor must be 0 or 1.

- <a id="INT8"></a>Comprehensive constraints on the number of INT8 quantization-related input parameters and the input/output data formats:

  - When both input and output are INT8: The input parameters deqScale1, quantScale1, deqScale2, and quantScale2 must all be present. quantOffset2 is optional and defaults to 0 if not passed.
  - For input INT8 and output FLOAT16: input parameters deqScale1, quantScale1, and deqScale2 must exist at the same time. If input parameter quantOffset2 or quantScale2 exists (that is, it is not nullptr), an error is reported and returned.
  - For the use case where the input is FLOAT16 or BFLOAT16 and the output is INT8: the input parameter quantScale2 is required, and quantOffset2 is optional (defaults to 0 if not passed). If any of the input parameters deqScale1, quantScale1, or deqScale2 exists (that is, is not nullptr), an error is reported and returned.
  - The input parameters quantScale2 and quantOffset2 support two formats, per-tensor and per-channel, and two data types, FLOAT32 and BFLOAT16. If quantOffset2 is passed, ensure that its type and shape information are the same as those of quantScale2. When the input is BFLOAT16, both FLOAT32 and BFLOAT16 are supported; otherwise, only FLOAT32 is supported. In per-channel format, when the output layout is BSH, the product of all dimensions of quantScale2 must be H. For other layouts, the product must be N*D. (If the output layout is BSH, it is recommended to set the quantScale2 shape to [1,1,H] or [H]. For BNSD output, [1,N,1,D] or [N,D] is recommended. For BSND output, [1,1,N,D] or [N,D] is recommended.)
  - inputLayout supports only BSH, BNSD, BSND, and BNSD_BSND.
- <a id="AntiQuant"></a>Constraints on pseudo-quantization parameters antiquantScale and antiquantOffset:

  - Only the pseudo-quantization scenario where kv_dtype is int8 is supported.
  - Per-channel mode: The shape of the two parameters can be \(2, N, 1, D\), \(2, N, D\), or \(2, H\), where N is numKeyValueHeads. The parameter data type is the same as the query data type, and antiquantMode is set to 0.
  - per-tensor mode: the shapes of the two parameters are both (2), the data type is the same as that of query, and antiquantMode is set to 0.
  - per-token mode: the shapes of the two parameters are both \(2, B, S\), the data type is fixed to FLOAT32, and antiquantMode is set to 1.
  - In asymmetric quantization mode, both antiquantScale and antiquantOffset parameters must be present.
  - In symmetric quantization mode, antiquantOffset can be null (i.e., nullptr). When the antiquantOffset parameter is null, symmetric quantization is performed; otherwise, asymmetric quantization is performed.

- Comprehensive constraints for query, key, and value inputs in TND, TND_NTD, and NTD_TND scenarios:

  - actualSeqLengths and actualSeqLengthsKv must be passed, and the number of elements in these input parameters is used as the Batch value. The value of each element in these input parameters represents the sum of the sequence lengths of the current batch and all previous batches. Therefore, the value of a subsequent element must be greater than or equal to the value of the preceding element.

  <!-- npu="910b" id53 -->
  - <term>Atlas A2 products</term>:
    - The sparse mode only supports sparse=0 with attenMask set to nullptr, or sparse=3 with attenMask not set to nullptr, or sparse=4 with attenMask not set to nullptr.
    - When d of query equals 512:
      - Supports TND and TND_NTD.
      - Page attention must be enabled. In this case, the length of actualSeqLengthsKv equals the batch value of key/value, representing the actual length of each batch, and the value is not greater than KV_S.
      - Supports s of 1 to 16 for each batch of query.
      - Requires n of query to be 32, 64, or 128, and n of key and value to be 1.
      - Requires queryRope and keyRope to be non-null, and d of queryRope and keyRope to be 64.
      - Not supported: left padding, tensorlist, pse, prefix, pseudo-quantization, full quantization, post-quantization.
      - In the NTD_TND scenario, enabling SoftMaxLse is not supported.
    - When d of query is not equal to 512:
      - When queryRope and keyRope are null: in the TND scenario, Q_D, K_D, and V_D must be equal and less than or equal to 256, or Q_D and K_D must be equal to 192 with V_D equal to 128 or 192; in the NTD_TND scenario, Q_D and K_D must be equal to 128 or 192, and V_D must be equal to 128. When queryRope and keyRope are not null, Q_D, K_D, and V_D must be equal to 128.
      - TND and NTD_TND are supported.
      - In TND scenarios, only FLOAT16 and BFLOAT16 data types are supported. In NTD_TND scenarios, only BFLOAT16 data type is supported.
      - In TND scenarios, when the head ratio is GQA/MQA (that is, the numHeads and numKeyValueHeads parameters must be fully passed, numHeads is an integer multiple of numKeyValueHeads, and the two are not equal), the following constraints apply:
        - When the data type is FLOAT16 or BFLOAT16, sparse=0 with attenMask set to nullptr is supported, or sparse=3 with an optimized attenMask is supported:
             - When Q_D, K_D, and V_D are equal and less than or equal to 256, or Q_D and K_D are equal to 192 and V_D is equal to 128 or 192, sparse=4 with an optimized attenMask is supported, requiring preTokens>=-actualSeqLengths, nextTokens>=-actualSeqLengthsKv, and preTokens+nextTokens>=0.
        - Only innerPrecise=0 is supported, that is, the high-precision mode without row invalidation.
        - Page attention is supported. The kv cache layout format supports BnBsH (blocknum, blocksize, H), where H is not greater than 65535, and blockSize supports values ≤ 128 and must be 16-aligned.
      - In the TND scenario, when the head ratio is MHA, the following constraints apply:
        - When the data type is FLOAT16 or BFLOAT16, sparse=0 with attenMask set to nullptr is supported, or sparse=3 or 4 with an optimized attenMask passed in is supported.
        - When the data type is FLOAT16, innerPrecise=0 and innerPrecise=1 are supported.
        - When the data type is BFLOAT16, only innerPrecise=0 is supported.
        - Page attention is supported. The kv cache layout format supports BnBsH (blocknum, blocksize, H), where H is not greater than 65535, and blockSize supports values ≤ 128 and must be 16-aligned.
      - In the NTD_TND scenario, page attention is not supported.
      - When sparse = 3, actualSeqLengths < actualSeqLengthsKv is required for each batch individually.
      - Left padding, tensorlist, pse, page attention, prefix, pseudo-quantization, full quantization, and post-quantization are not supported.
      - The number of elements in actualSeqLengths and actualSeqLengthsKv must not be greater than 4096.
  <!-- end id53 -->
  <!-- npu="950" id7 -->
  - Ascend 950PR&950DT products:
    - TND is supported.
    - Left padding, tensorlist, pseType=0, and prefix are not supported.

  <!-- end id7 -->

- When queryRope and keyRope are input, it is an MLA scenario, and the parameter constraints are as follows:

  - The data type and data format of queryRope must be consistent with query.
  - The data type and data format of keyRope are consistent with those of key.
  - queryRope and keyRope must be configured together or not configured at all. Configuring only one of them is not supported.
  - When queryRope and keyRope are input, only the following features are supported:
    - dtype: FP16, BF16;
    - The d of query only supports 512/128;

    <!-- npu="910b" id8 -->
    - <term>Atlas A2 products</term>:
      - When Q_S is greater than 1 (i.e., MTP), the actualSeqLengths parameter is supported only when inputLayout is TND, and is not supported for other layouts.
      - When d of query equals 512:
        - When queryRope is configured, the s of query must be 1-16, the d in the queryRope shape must be 64, and the remaining dimensions must be the same as those of query;
        - When keyRope is configured, n of key must be 1, d must be 512, d in the shape of keyRope must be 64, and the remaining dimensions must be consistent with key.
        - sparse: When Q_S equals 1, only sparse=0 is supported and attenMask must be nullptr. When Q_S is greater than 1, only sparse=3 is supported and mask must be passed.
        - key&value&keyRope supports ND and NZ input. When the input is in NZ format, the input format is [blockNum, N, D/16, blockSize, 16].
        - inputLayout: BSH, BSND, BNSD, BNSD_NBSD, BSND_NBSD, BSH_NBSD, TND, TND_NTD. NZ input does not support BNSD or BNSD_NBSD.
        - Page attention must be enabled: blockSize supports 16 and 128. NZ input does not support configuring 16.
        - Not supported: left padding, tensorlist, pse, prefix, pseudo-quantization, full quantization, post-quantization.
      - When d of query equals 128:
        - inputLayout: TND, NTD_TND;
        - When queryRope is configured, d in the shape of queryRope must be 64, and the remaining dimensions must be consistent with query;
        - When keyRope is configured, d in the shape of keyRope must be 64, and the remaining dimensions must be consistent with key;
        - Other constraints remain consistent with the comprehensive restrictions in TND and NTD_TND scenarios;
        - Left padding, tensorlist, pse, page attention, prefix, pseudo-quantization, full quantization, and post-training quantization are not supported.
    <!-- end id8 -->
    <!-- npu="950" id9 -->
    - Ascend 950PR&950DT products:
      - When d of query equals 512:
        - When queryRope is configured, s of query must be 1-16 and d must be 512. In the queryRope shape, b, n, and s must be the same as those of query, and d must be 64;
        - When keyRope is configured, n of key must be 1, d must be 512, and in the shape of keyRope, b, n, and s must be consistent with key, and d must be 64.
        - sparse: Supports sparse=0, sparse=3 with mask passed, and sparse=4 with mask passed.
        - key&value supports ND input.
        - inputLayout: BSH, BSND, BNSD, TND.
        - Supports actualSeqLengths and actualSeqLengthsKv parameters. When Q_S is greater than 1 (i.e., MTP) and the normal part of key&value reuses the same data, the actualSeqLengths parameter is supported only when inputLayout is TND, and is not supported for other layouts.
        - pse is not supported.
      - When d of query equals 128:
        - When queryRope is configured, b, n, and s in the shape of queryRope must be consistent with query, and d must be 64.
        - When keyRope is configured, b, n, and s in the shape of keyRope must be consistent with key, and d must be 64.
        - inputLayout: BSH, BSND, BNSD, BNSD_BSND, TND;
        - Page attention, prefix, pseudo-quantization, full quantization, and post-training quantization are not supported.
        - When kv is a tensorlist, b in the shape of keyRope must be consistent with the tensorlist length, n and s must be equal to n and s of each tensor in the tensorlist, and d is 64.
        - pse is not supported.

    <!-- end id9 -->

- numKeyValueHeads usage constraints: numHeads must be divisible by numKeyValueHeads. In BSND, BNSD, BNSD_BSND, and TND scenarios, it must also be the same as the N-axis shape value of key/value in the shape; otherwise, an execution exception occurs.

  <!-- npu="950" id10 -->
  - <term>Ascend 950PR&950DT products</term>:
    - In pseudo quantization and full quantization use cases, the ratio of numHeads to numKeyValueHeads cannot exceed 64. In non-quantized MLA decode use cases, the ratio of numHeads to numKeyValueHeads is unlimited. In non-quantized and MLA prefill use cases, a ratio of numHeads to numKeyValueHeads greater than 64 is supported only when the D axis is 64 or 128; it is not supported for other D axis values.

  <!-- end id10 -->

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
              <td>                  <ul style="margin: 0; padding-left: 20px;">
                      <li>If attenmask is not passed in: no mask operation is performed, and preTokens and nextTokens are ignored (internally set to INT_MAX);</li>
                      <li>If attenmask is passed in: the complete attenmask matrix (S1 * S2) must be passed in, indicating that the part between preTokens and nextTokens needs to be computed; preTokens > -actualSeqLengthsKv and preTokens + nextTokens >= 0 are required. In the prefix use case, the prefix length must be added to actualSeqLengthsKv.</li>
                  </ul>
              </td>
          </tr>
          <tr>
              <td>1</td>
              <td>allMask mode</td>
              <td>The complete attenmask matrix (S1 * S2) must be passed in.</td>
          </tr>
          <tr>
              <td>2</td>
              <td>leftUpCausal mode</td>
              <td>The optimized attenmask matrix (2048*2048) must be passed in.</td>
          </tr>
          <tr>
              <td>3</td>
              <td>rightDownCausal mode</td>
              <td>Corresponds to the lower-triangle use case divided by the right vertex. Pass in the optimized attenmask matrix (2048*2048).</td>
          </tr>
          <tr>
              <td>4</td>
              <td>band mode</td>
              <td>                <ul style="margin: 0; padding-left: 20px;">
                  <li>Pass in the optimized attenmask matrix (2048*2048);</li>
                  <li>preTokens > -actualSeqLengths, nextTokens > -actualSeqLengthsKv, and preTokens + nextTokens >= 0 are required. In the prefix use case, the prefix length must be added to actualSeqLengthsKv.</li></ul>
              </td>
          </tr>
          <tr>
              <td>5</td>
              <td>prefix mode</td>
              <td>This mode is not supported currently. It is recommended to pass 0 if the user does not specifically specify this parameter.</td>
          </tr>
          <tr>
              <td>6</td>
              <td>global mode</td>
              <td>This mode is not supported currently. It is recommended to pass 0 if the user does not specifically specify it.</td>
          </tr>
          <tr>
              <td>7</td>
              <td>dilated mode</td>
              <td>This mode is not supported currently. It is recommended to pass 0 if the user does not specifically specify it.</td>
          </tr>
          <tr>
              <td>8</td>
              <td>block_local mode</td>
              <td>This mode is not supported currently. It is recommended to pass 0 if the user does not specifically specify it.</td>
          </tr>
          <tr>
              <td colspan="3" style="text-align: left; ">                  <strong>Special constraints:</strong> When Q_S is 1 and no rope input is used, the sparseMode parameter does not take effect.
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

  - NOTE: BFLOAT16 and INT8 do not distinguish between high precision and high performance. The invalid row correction takes effect for FLOAT16, BFLOAT16, and INT8. Currently, 0 and 1 are reserved configuration values. If the mask part involved in computation contains an entire row of all 1s, precision loss may occur. In this case, you can set this parameter to 2 or 3 to enable the invalid row function to improve precision, but this setting degrades performance. If the operator can detect invalid rows, the invalid row computation is automatically enabled, for example, when sparse_mode is 3 and Sq > Skv.

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
                <td>Represents the per-tensor plus per-head mode.</td>
            </tr>
            <tr>
                <td>3</td>
                <td>Represents the per-token plus per-head mode.</td>
            </tr>
            <tr>
                <td>4</td>
                <td>Represents the per-token plus the mode where scale/offset is managed using page attention.</td>
            </tr>
            <tr>
                <td>5</td>
                <td>Represents per-token combined with per-head, with scale/offset managed in page attention mode.</td>
            </tr>
            <tr>
                <td>6</td>
                <td>Represents the per-token-group mode.</td>
            </tr>
            <tr>
                <td colspan="2" style="text-align: left; ">                    <strong>Special constraint:</strong> Except for the use case where keyAntiquantMode is 0 and valueAntiquantMode is 1, this parameter must be the same as valueAntiquantMode.</td>
            </tr>
          </tbody>
        </table>
    </div>

  <!-- npu="910b" id11 -->
  - <term>Atlas A2 products</term>: When Q_S equals 1, passing values other than 0, 1, 2, 3, 4, or 5 causes abnormal execution. When Q_S is greater than or equal to 2, only 0 and 1 are supported; other values cause abnormal execution.
  <!-- end id11 -->
  <!-- npu="950" id12 -->
  - <term>Ascend 950PR&950DT products</term>: Passing values other than 0, 1, 2, 3, 4, 5, and 6 causes abnormal execution.

  <!-- end id12 -->

- The usage restrictions of valueAntiquantMode are as follows:

  - Except for the use case where keyAntiquantMode is 0 and valueAntiquantMode is 1, this parameter must be the same as keyAntiquantMode.

  <!-- npu="910b" id13 -->
  - <term>Atlas A2 products</term>: When Q_S equals 1, passing values other than 0, 1, 2, 3, 4, or 5 causes abnormal execution. When Q_S is greater than or equal to 2, only 0 and 1 are supported; other values cause abnormal execution.
  <!-- end id13 -->
  <!-- npu="950" id14 -->
  - <term>Ascend 950PR&950DT products</term>: Passing values other than 0, 1, 2, 3, 4, 5, and 6 causes abnormal execution.

  <!-- end id14 -->

- softmaxLse usage constraints are as follows:

  - In the ring attention algorithm, max is taken over the result of query multiplied by key to obtain softmax_max. Then exp is applied to the result of query multiplied by key minus softmax_max, and the sum is computed to obtain softmax_sum. Finally, log is taken of softmax_sum, and softmax_max is added to obtain the result.
  - When softmaxLseFlag is True, the shape must be [B,N,Q_S,1] in general. When inputLayout is TND/NTD_TND, the shape must be [T,N,1].
  - When softmaxLseFlag is False, if the softmaxLse tensor passed in is not empty, the tensor data is directly returned. If nullptr is passed for softmaxLse, a tensor of shape {1} with all zeros is returned.

- **When Q_S is greater than 1**:

  - The function usage restrictions for query, key, and value inputs are as follows:

    - B-axis restrictions
      - Supports B-axis less than or equal to 65536.
      - In the non-contiguous use case, the batch in the tensorlists of key and value must be 1, the number of tensors must equal B of query, and N and D must be the same. Due to the tensorlist limitation, B cannot be greater than 256 in the non-contiguous use case.

      <!-- npu="910b" id15 -->
      - <term>Atlas A2 products</term>:
        - If the input type is INT8 and the D-axis is not 32-byte-aligned, the maximum supported value of the B-axis is 128. If the input type is FLOAT16 or BFLOAT16 and the D-axis is not 16-byte-aligned, the B-axis is also only supported up to 128.

      <!-- end id15 -->

    - N-axis restrictions

      <!-- npu="950" id16 -->
      - <term>Ascend 950PR&950DT products</term>:
        - In GQA non-quantization scenarios, the N axis supports values greater than 256. In pseudo-quantization and full-quantization scenarios, the N axis must be less than or equal to 256.
      <!-- end id16 -->
      <!-- npu="910b" id17 -->
      - <term>Atlas A2 products</term>: The N axis must be less than or equal to 256.

      <!-- end id17 -->

    - The D axis supports values less than or equal to 512. When inputLayout is BSH or BSND, it is recommended that N*D be less than 65535.
    - S supports values less than or equal to 20971520 (20M). In some long sequence scenarios, if the computation workload is too large, the operator execution may time out (aicore error, with errorStr: timeout or trap error). In such scenarios, S splitting is recommended. Note: The computation workload here is affected by B, S, N, D, etc. Larger values result in greater computation workload. Typical long sequence scenarios that may cause timeouts (i.e., where the product of B, S, N, and D is large) include but are not limited to:

      <div style="overflow-x: auto;">
      <table style="undefined;table-layout: fixed; width: 930px"><colgroup>
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

    <!-- npu="910b" id18 -->
    - D-axis constraints: <term>Atlas A2 products</term>:
      - When the type of query, key, value, or attentionOut includes INT8, the D-axis must be 32-aligned. When the type of query, key, value, or attentionOut includes INT4, the D-axis must be 64-aligned. When all types are FLOAT16 or BFLOAT16, the D-axis must be 16-aligned.

    <!-- end id18 -->
    <!-- npu="950" id19 -->
    - <term>Ascend 950PR&950DT products</term>:

      - Non-quantization scenario: The types of query, key, and value are all FLOAT16 or BFLOAT16, and D-axis 1–512 is fully supported.
      - Full-quantization scenario: The types of query, key, and value are all INT8, and D-axis 1–512 is fully supported.
      - Pseudo-quantization scenario: The type of query is FLOAT16 or BFLOAT16, and the types of key and value are INT8/HIFLOAT8/FLOAT8_E4M3FN/FLOAT4_E2M1/INT4(INT32). When the types of key and value are FLOAT4_E2M1/INT4(INT32), the D-axis of query and the D-axis of key and value only support 64-aligned (under INT32, the D-axis of key and value is 8-aligned).

    <!-- end id19 -->

  - The actualSeqLengths input parameter must be a non-negative number when passed.

    <!-- npu="910b" id20 -->
    - <term>Atlas A2 products</term>: The valid sequence length of each batch in this input parameter must not exceed the sequence length of the corresponding batch in query. When the passed length of seqlen is 1, each batch uses the same seqlen. When the passed length is greater than or equal to the number of batches, the first batch count of seqlen values are used. Other lengths are not supported. This applies when the inputLayout of query is TND/NTD_TND.
    <!-- end id20 -->
    <!-- npu="950" id21 -->
    - <term>Ascend 950PR&950DT products</term>: When inputLayout differs, its meaning and interception conditions vary: When inputLayout is not TND, this input parameter is optional, with a length of 1 or greater than or equal to the batch value of query. The values in this parameter represent the actual length of each batch and should not exceed Q_S. When inputLayout is TND, this input parameter must be passed. The b-th value represents the cumulative length of the S axis of the first b batches, and the values should be arranged in ascending order (greater than or equal to the previous value). The length of this input parameter represents the total number of batches.
    <!-- end id21 -->

  - The actualSeqLengthsKv input parameter should be non-negative when passed.

    <!-- npu="910b" id22 -->
    - <term>Atlas A2 products</term>: The valid Sequence Length of each batch in this input parameter should not be greater than the Sequence Length of the corresponding batch in key/value. When the passed length of seqlenKv is 1, each batch uses the same seqlenKv. When the passed length is greater than or equal to the number of batches, the first batch count of seqlenKv is used. Other lengths are not supported. This applies when the inputLayout of key/value is TND/NTD_TND.
    <!-- end id22 -->
    <!-- npu="950" id23 -->
    - <term>Ascend 950PR&950DT products</term>: When inputLayout differs, its meaning and interception conditions vary: When inputLayout is not TND, this input parameter is optional, with a length of 1 or greater than or equal to the batch value of key/value. The values in this parameter represent the actual length of each batch and should not exceed KV_S. When inputLayout is TND, this input parameter must be passed. In non-PA scenarios, the b-th value represents the cumulative length of the S axis of the first b batches, and the values should be arranged in ascending order (greater than or equal to the previous value). The length of this input parameter represents the total number of batches. In PA scenarios, its length equals the batch value of key/value, representing the actual length of each batch, and the value does not exceed KV_S.
    <!-- end id23 -->

  - The sparseMode parameter currently only supports values of 0, 1, 2, 3, and 4. An error is reported when other values are used.

    - When sparseMode = 0, if attenMask is a null pointer or attenMask is passed in a left-padding scenario, the input parameters preTokens and nextTokens are ignored.
    - When sparseMode = 2, 3, or 4, the shape of attenMask must be S,S or 1,S,S or 1,1,S,S, where the value of S must be fixed to 2048, and the user must ensure that the passed attenMask is lower triangular. If attenMask is nullptr or the passed shape is incorrect, an error is reported.
    - In scenarios where sparseMode = 1, 2, or 3, the input parameters preTokens and nextTokens are ignored and assigned values according to relevant rules.
  - The composite parameter scenario for kvCache dequantization supports only the case where query is FLOAT16, dequantizing key and value of INT8 type to FLOAT16. The product range of the datarange of the input parameter key/value and the datarange of the input parameter antiQuantScale falls within (-1, 1). The high-performance mode can guarantee precision; otherwise, the high-precision mode must be enabled to guarantee precision.
  - Page attention scenario:

    - Enabling page attention requires blockTable to exist and be valid, and key and value to be laid out in a contiguous memory region based on the indexes in blockTable. In this use case, the inputLayout parameter of key and value does not take effect. blockTable is filled with blockids. The validity of blockid is not verified currently and must be ensured by users.
    - blockSize is a user-defined parameter whose value affects the performance of page attention. When page attention is enabled, blockSize ranges from 128 to 512 and must be a multiple of 128. Generally, page attention improves throughput but degrades performance.
    - In page attention use cases, if the input kv cache layout is BnBsH (blocknum, blocksize, H) and KV_N * D exceeds 65535, an error is reported due to hardware instruction constraints. To resolve this issue, enable GQA (to reduce KV_N) or change the kv cache layout to BnNBsD (blocknum, KV_N, blocksize, D). If the inputLayout of query is BNSD or TND, the kv cache layout supports two formats: BnBsH and BnNBsD. If the inputLayout of query is BSH or BSND, the kv cache layout supports only the BnBsH format. blocknum cannot be less than the sum of the number of blocks per batch calculated based on actualSeqLengthsKv and blockSize. In addition, the shapes of key and value must be consistent.
    - page attention pseudo quantization use case

      <!-- npu="910b" id24 -->
      - <term>Atlas A2 products</term>: Supports query as FLOAT16/BFLOAT16, and supports key and value as INT8.
      <!-- end id24 -->
      <!-- npu="950" id25 -->
      - <term>Ascend 950PR&950DT products</term>: Supports key and value dtype of FLOAT16/BFLOAT16/INT8/HIFLOAT8/FLOAT8_E4M3FN/FLOAT4_E2M1/INT4(INT32).
      <!-- end id25 -->

    - page attention full quantization use case

      <!-- npu="910b" id26 -->
      - <term>Atlas A2 products</term>: Not supported query dtype as INT8.
      <!-- end id26 -->
      <!-- npu="950" id27 -->
      - <term>Ascend 950PR&950DT products</term>: Supports query dtype as INT8.
      <!-- end id27 -->

    - Page attention does not support the tensorlist scenario or the left-padding scenario.
    - In the page attention scenario, actualSeqLengthsKv must be passed.
    - In page attention scenarios, blockTable must be two-dimensional. The first dimension length must equal B, and the second dimension length cannot be less than maxBlockNumPerSeq (maxBlockNumPerSeq is the number of blocks corresponding to the maximum actualSeqLengthsKv across different batches).
    - When page attention is enabled, KV_S of the input must be greater than or equal to maxBlockNumPerSeq * blockSize in the following use cases:
      - When attenMask is passed in, for example, the mask shape is (B, 1, Q_S, KV_S)
      - When pseShift is passed in, for example, the pseShift shape is (B, N, Q_S, KV_S)
  - Query left-padding scenario:

    - The formula for the query copy start position is: Q_S - queryPaddingSize - actualSeqLengths. The formula for the query copy end position is: Q_S - queryPaddingSize. The query copy start position must not be less than 0, and the end position must not be greater than Q_S. Otherwise, the result will not meet expectations.
    - If kvPaddingSize is less than 0, it is set to 0.
    - Must be used together with the actualSeqLengths parameter. Otherwise, the query right-padding use case is used by default.
    - PageAttention is not supported and cannot be used together with the blocktable parameter.

    <!-- npu="910b" id28 -->
    - <term>Atlas A2 products</term>: The scenario where Q is BF16/FP16 and KV is INT4 is not supported.
    <!-- end id28 -->

  - kv left-padding scenario:

    - The copy start point of key and value is calculated as: KV_S - kvPaddingSize - actualSeqLengthsKv. The copy end point of key and value is calculated as: KV_S - kvPaddingSize. The copy start point of key and value must not be less than 0, and the end point must not be greater than KV_S. Otherwise, the result will not meet expectations.
    - If kvPaddingSize is less than 0, it is set to 0.
    - Must be used together with the actualSeqLengthsKv parameter. Otherwise, the KV right-padding use case is used by default.
    - PageAttention is not supported and cannot be used together with the blocktable parameter.

    <!-- npu="910b" id29 -->
    - <term>Atlas A2 products</term>: The scenario where Q is BF16/FP16 and KV is INT4 is not supported.
    <!-- end id29 -->

  - When the output is int8 and quantScale2 and quantOffset2 are per-channel, left padding, Ring Attention, or the use case where D is not 32-byte aligned is currently not supported.
  - When the output is INT8, the scenario where sparse is band and preTokens/nextTokens are negative is not supported.
  - When the output is INT8, if a non-null pointer and a non-empty tensor are passed for quantOffset2, and sparseMode, preTokens, and nextTokens meet the following conditions, some rows of the matrix do not participate in the computation, resulting in calculation errors. In this use case, the call is intercepted (Solution: if this use case is not to be intercepted, perform post quantization outside the FIA API instead of enabling it inside the FIA API):

    - sparseMode = 0: if attenMask is a non-null pointer and actualSeqLengths - actualSeqLengthsKV - actualSharedPrefixLen - preTokens > 0 or nextTokens < 0 for each batch, the interception condition is met
    - sparseMode = 1 or 2: the interception condition is never met
    - sparseMode = 3, if each batch satisfies actualSeqLengthsKV + actualSharedPrefixLen - actualSeqLengths < 0, the interception condition is met.
    - sparseMode = 4: If preTokens < 0 or nextTokens + actualSeqLengthsKV + actualSharedPrefixLen - actualSeqLengths of each batch < 0, the interception condition is met.
  - pseShift function usage restrictions as follows:

    - The pseShift function is supported when the query data type is FLOAT16, BFLOAT16, or INT8.
    - When the query data type is FLOAT16 and pseShift is present, the high-precision mode is forcibly used, and the corresponding restrictions are inherited from those of the high-precision mode.
    - Q_S must be greater than or equal to the S length of query, and KV_S must be greater than or equal to the S length of key. In the prefix scenario, KV_S must be greater than or equal to the sum of actualSharedPrefixLen and the S length of key.

    <!-- npu="950" id30 -->
    - Ascend 950PR&950DT products: In non-quantized and full quantization scenarios, there is no alignment restriction.
    <!-- end id30 -->

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

    <!-- npu="910b" id31 -->
    - <term>Atlas A2 products</term>:
      - keyAntiquantMode and valueAntiquantMode must be the same.
      - keyAntiquantScale and valueAntiquantScale must be either both empty or both non-empty; keyAntiquantOffset and valueAntiquantOffset must be either both empty or both non-empty.
      - If keyAntiquantScale and valueAntiquantScale are both non-empty, their shapes must be the same; if keyAntiquantOffset and valueAntiquantOffset are both non-empty, their shapes must be the same.
      - Only per-token and per-channel modes are supported. In per-token mode, the shapes of both parameters must be \(B, S\) and the data type is fixed to FLOAT32. In per-channel mode, the shapes of both parameters must be \(N, D\), \(N, 1, D\) or \(H\) and the data type is fixed to BF16.
      - If both pseudo-quantization parameters and KV separate quantization parameters are passed in, the KV separate quantization parameters take precedence.
      - In scenarios where keyAntiquantScale and valueAntiquantScale are non-null, the s of query must be less than or equal to 16.
      - In scenarios where keyAntiquantScale and valueAntiquantScale are non-null, the dtype of query must be BFLOAT16, the dtype of key and value must be INT8, and the dtype of the output must be BFLOAT16.
      - In scenarios where keyAntiquantScale and valueAntiquantScale are non-null, the tensorlist, left padding, and page attention features are not supported.
    <!-- end id31 -->
    <!-- npu="950" id32 -->
    - <term>Ascend 950PR&950DT products</term>:
      - keyAntiquantMode and valueAntiquantMode must be the same, except when keyAntiquantMode is 0 and valueAntiquantMode is 1.
      - keyAntiquantScale and valueAntiquantScale must be either both empty or both non-empty; keyAntiquantOffset and valueAntiquantOffset must be either both empty or both non-empty.
      - If keyAntiquantScale and valueAntiquantScale are both non-empty, their shapes must be the same except when keyAntiquantMode is 0 and valueAntiquantMode is 1; if keyAntiquantOffset and valueAntiquantOffset are both non-empty, their shapes must be the same except when keyAntiquantMode is 0 and valueAntiquantMode is 1.
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
      - When both pseudo-quantization parameters and KV separate quantization parameters are passed in, the KV separate quantization parameters take precedence.
      - The INT4 (INT32) pseudo-quantization scenario supports only KV pseudo-quantization parameter separation, specifically including:
        - per-channel mode;
        - per-token mode;
        - per-token plus per-head mode;
        - key supports per-channel plus value supports per-token mode.
      - Some pseudo-quantization scenarios do not support post-quantization
        - <term>Atlas A2 products</term>: Post-quantization is not supported in INT4 (INT32) pseudo-quantization scenarios.
        - <term>Ascend 950PR&950DT products</term>: Post-quantization is not supported in INT4 (INT32) and FLOAT4_E2M1 pseudo-quantization scenarios.

    <!-- end id32 -->

- **When Q_S equals 1**:

  - The function usage restrictions for query, key, and value inputs are as follows:
    - The B-axis must be less than or equal to 65536, and the D-axis must be less than or equal to 512.
    - N-axis restrictions

      <!-- npu="950" id33 -->
      - <term>Ascend 950PR&950DT products</term>:
        - In GQA non-quantization scenarios, the N axis supports values greater than 256. In pseudo-quantization and full-quantization scenarios, the N axis must be less than or equal to 256.
      <!-- end id33 -->
      <!-- npu="910b" id34 -->
      - <term>Atlas A2 products</term>: The N axis must be less than or equal to 256.
      <!-- end id34 -->

    - The scenario where query, key, and value input types are all INT8 is not supported yet.
    - In INT4 (INT32) pseudo-quantization scenarios, the aclnn single-operator call supports KV INT4 input or INT4 concatenated into INT32 input. (It is recommended to use dynamicQuant to generate data in INT4 format, because dynamicQuant treats one INT32 as containing eight INT4 values.)
    - In the INT4 (INT32) pseudo-quantization scenario, if KV INT4 is packed into INT32 input, then N, D, or H of KV is one-eighth of the actual value (same for prefix).
    - key and value have restrictions on the D axis under specific data types.

      <!-- npu="910b" id35 -->
      - <term>Atlas A2 products</term>: When the key and value input type is INT4 (INT32), the D axis must be 64-aligned (INT32 only supports D 8-aligned).
      <!-- end id35 -->
      <!-- npu="950" id36 -->
      - <term>Ascend 950PR&950DT products</term>: When the key and value input type is FLOAT4_E2M1/INT4 (INT32), the D axis of query and the D axis of key and value must be 64-aligned (INT32 only supports D 8-aligned for key and value).
      <!-- end id36 -->

  - The actualSeqLengths input parameter must be a non-negative number when passed.

    <!-- npu="910b" id37 -->
    - <term>Atlas A2 products</term>: When the inputLayout of query is not TND and Q_S is 1, this parameter is invalid. When the inputLayout of query is TND/TND_NTD.
    <!-- end id37 -->
    <!-- npu="950" id38 -->
    - <term>Ascend 950PR&950DT products</term>: When inputLayout differs, its meaning and interception conditions vary: When inputLayout is not TND, this input parameter is optional, with a length of 1 or greater than or equal to the batch value of query. The values in this parameter represent the actual length of each batch and should not exceed Q_S. When inputLayout is TND, this input parameter must be passed. The b-th value represents the cumulative length of the S axis of the first b batches, and the values should be arranged in ascending order (greater than or equal to the previous value). The length of this input parameter represents the total number of batches.
    <!-- end id38 -->

  - The actualSeqLengthsKv input parameter should be non-negative when passed.

    <!-- npu="910b" id39 -->
    - <term>Atlas A2 products</term>: The effective Sequence Length of each batch in this input parameter should not exceed the Sequence Length of the corresponding batch in key/value. When the passed length of seqlenKv is 1, each batch uses the same seqlenKv. When the passed length is greater than or equal to the number of batches, the first Batch number of seqlenKv is used. Other lengths are not supported. When the inputLayout of key/value is TND/TND_NTD.
    <!-- end id39 -->
    <!-- npu="950" id40 -->
    - <term>Ascend 950PR&950DT products</term>: When inputLayout differs, its meaning and interception conditions vary: When inputLayout is not TND, this input parameter is optional, with a length of 1 or greater than or equal to the batch value of key/value. The values in this parameter represent the actual length of each batch and should not exceed KV_S. When inputLayout is TND, this input parameter must be passed. In non-PA scenarios, the b-th value represents the cumulative length of the S axis of the first b batches, and the values should be arranged in ascending order (greater than or equal to the previous value). The length of this input parameter represents the total number of batches. In PA scenarios, its length equals the batch value of key/value, representing the actual length of each batch, and the value does not exceed KV_S.
    <!-- end id40 -->

  - Page attention scenario:
    - Page attention can be enabled only when blocktable exists and is valid, and key and value are arranged in a contiguous memory block according to the indexes in blocktable. In this use case, the inputLayout parameters of key and value are invalid.

      <!-- npu="910b" id41 -->
      - <term>Atlas A2 products</term>: Supports key and value dtype of FLOAT16/BFLOAT16/INT8.
      <!-- end id41 -->
      <!-- npu="950" id42 -->
      - <term>Ascend 950PR&950DT products</term>: Supports key and value dtype of FLOAT16/BFLOAT16/INT8/HIFLOAT8/FLOAT8_E4M3FN/FLOAT4_E2M1/INT4(INT32).
      <!-- end id42 -->

    - blockSize is a user-defined parameter whose value affects the performance of page attention. When page attention is enabled, a non-zero value must be passed to blockSize, and the maximum value of blockSize is 512. Generally, page attention improves throughput but causes performance degradation.

      <!-- npu="910b" id43 -->
      - <term>Atlas A2 products</term>: When the input types of key and value are FLOAT16/BFLOAT16, the value must be aligned to 16; when the input types of key and value are INT8, the value must be aligned to 32, and 128 is recommended.
      <!-- end id43 -->
      <!-- npu="950" id44 -->
      - <term>Ascend 950PR&950DT products</term>: When the input types of key and value are FLOAT16/BFLOAT16, the value must be aligned to 16; when the input types of key and value are INT8/HIFLOAT8/FLOAT8_E4M3FN, the value must be aligned to 32; when the input types of key and value are FLOAT4_E2M1/INT4(INT32), the value must be aligned to 64.
      <!-- end id44 -->

    - In the page attention scenario, when the inputLayout of query is BNSD or TND, the KV cache layout supports two formats: BnBsH (blocknum, blocksize, H) and BnNBsD (blocknum, KV_N, blocksize, D). When the inputLayout of query is BSH or BSND, the KV cache layout supports only the BnBsH format. blocknum cannot be less than the sum of the number of blocks for each batch calculated based on actualSeqLengthsKv and blockSize. In addition, the shapes of key and value must be consistent.
    - In the page attention scenario, the performance of the KV cache layout BnNBsD is generally better than that of BnBsH. It is recommended to prefer the BnNBsD format.
    - When page attention is enabled, if the input kv cache layout is BnBsH and numKvHeads * headDim exceeds 64k, the operation is intercepted and an error is reported due to hardware instruction constraints. To resolve this issue, enable GQA (reduce numKvHeads) or change the kv cache layout to BnNBsD.
    - Page attention does not support the tensorlist scenario or the left-padding scenario.

        <!-- npu="910b" id45 -->
        - <term>Atlas A2 products</term>: The scenario where Q is BF16/FP16 and KV is INT4(INT32) is not supported.
        <!-- end id45 -->
        <!-- npu="950" id46 -->
        - <term>Ascend 950PR&950DT products</term>: The scenario where Q is BF16/FP16 and KV is INT4 (INT32) is supported.
        <!-- end id46 -->

    - In the page attention scenario, actualSeqLengthsKv must be passed.
    - In the page attention scenario, blockTable must be two-dimensional, with the first dimension length equal to B and the second dimension length not less than maxBlockNumPerSeq (maxBlockNumPerSeq is the number of blocks corresponding to the maximum actualSeqLengthsKv in each batch).
    - When page attention is enabled, input S must be greater than or equal to the second dimension of blockTable multiplied by blockSize in the following use cases.
      - Attention mask is enabled, for example, the mask shape is \(B, 1, 1, S\).
      - pseShift is enabled, for example, with a shape of \(B, N, 1, S\).
      - The pseudo quantization per-token mode is enabled: the shapes of the input parameters antiquantScale and antiquantOffset are both \(2, B, S\).
      - The per-token plus per-head mode is enabled: the shapes of the two parameters are both \(B, N, S\), the data type is fixed to FLOAT32, and this mode is supported when the data types of key and value are INT8 and INT4\(INT32\).
      - The per-token-group mode is enabled: the shape of antiquantScale is \(1, B, N, S, D/32\),, the data type is fixed to FLOAT8_E8M0, and antiquantOffset is not supported. It is supported when the data types of key and value are FLOAT4_E2M1.
  - kv left-padding scenario:

    <!-- npu="910b" id47 -->
    - <term>Atlas A2 products</term>: The scenario where Q is BF16/FP16 and KV is INT4(INT32) is not supported.
    <!-- end id47 -->
    <!-- npu="950" id48 -->
    - <term>Ascend 950PR&950DT products</term>: The scenario where Q is BF16/FP16 and KV is INT4(INT32) is supported, and there is no restriction on QKV data types.
    <!-- end id48 -->
    - The formula for the kvCache copy start point is: KV_S - kvPaddingSize - actualSeqLengthsKv. The formula for the kvCache copy end point is: KV_S - kvPaddingSize. If the kvCache copy start point or end point is less than 0, the returned data result is all zeros.
    - If kvPaddingSize is less than 0, it is set to 0.
    - It must be enabled together with the actualSeqLengthsKv parameter. Otherwise, the KV right padding use case is used by default.
    - PageAttention and tensorlist are not supported; otherwise, the default is the kv right-padding scenario.
    - When it is enabled together with the attenMask parameter, ensure that attenMask is correctly defined, that is, it can correctly hide invalid data. Otherwise, precision issues are introduced.
  - pseShift function usage restrictions as follows:
    - The data type of pseShift must be consistent with that of query.
  - KV pseudo-quantization parameter separation:
    - Except for the use case where keyAntiquantMode is 0 and valueAntiquantMode is 1, keyAntiquantMode and valueAntiquantMode must be the same
    - keyAntiquantScale and valueAntiquantScale are either both null or both non-null; keyAntiquantOffset and valueAntiquantOffset are either both null or both non-null
    - When keyAntiquantScale and valueAntiquantScale are both non-null, their shapes must be the same, except for the use case where keyAntiquantMode is 0 and valueAntiquantMode is 1; when keyAntiquantOffset and valueAntiquantOffset are both non-null, their shapes must be the same, except for the use case where keyAntiquantMode is 0 and valueAntiquantMode is 1

    <!-- npu="910b" id49 -->
    - <term>Atlas A2 products</term>: Supports eight modes: per-channel, per-tensor, per-token, per-tensor plus per-head, per-token plus per-head, per-token with page attention mode for scale/offset management, per-token plus per-head with page attention mode for scale/offset management, and key per-channel plus value per-token. In the following, N denotes numKeyValueHeads.
      - Per-channel mode: The shapes of the two parameters can be \(1, N, 1, D\), \(1, N, D\), or \(1, H\). The parameter data type is the same as the query data type. Supported when the key and value data types are INT8 or INT4 (INT32).
      - Per-tensor mode: The shapes of both parameters are \(1\). The data type is the same as the query data type. Supported when the key and value data types are INT8.
      - Per-token mode: The shapes of both parameters are \(1, B, S\). The data type is fixed to FLOAT32. Supported when the key and value data types are INT8 or INT4 (INT32).
      - Per-tensor plus per-head mode: The shapes of both parameters are \(N\). The data type is the same as the query data type. Supported when the key and value data types are INT8.
      - key per-channel and value per-token mode: For key per-channel, the shapes of the two parameters can be \(1, N, 1, D\), \(1, N, D\), or \(1, H\), and the parameter data type is the same as the query data type. For value per-token, the shapes of the two parameters are both \(1, B, S\) and the data type is fixed to FLOAT32. This mode is supported when the key and value data types are INT8 or INT4(INT32). When the key and value data types are INT8, only FLOAT16 is supported for the query and attentionOut data types.
    <!-- end id49 -->
    <!-- npu="950" id50 -->
    - <term>Ascend 950PR&950DT products</term>: Supports nine modes: per-channel, per-tensor, per-token, per-tensor with per-head, per-token with per-head, per-token using page attention mode to manage scale/offset, per-token with per-head using page attention mode to manage scale/offset, key per-channel with value per-token, and per-token-group. In the following, N denotes numKeyValueHeads.
      - per-channel mode: The shapes of the two parameters can be (1, N, 1, D), (1, N, D), or (1, H). The parameter data type is the same as the query data type. This mode is supported when the key and value data types are INT8, INT4(INT32), HIFLOAT8, or FLOAT8_E4M3FN. When the key and value data types are HIFLOAT8 or FLOAT8_E4M3FN, carrying antiQuantOffset is not supported.
      - per-tensor mode: The shapes of the two parameters are both (1), and the data type is the same as the query data type. This mode is supported when the key and value data types are INT8 or INT4(INT32).
      - per-token mode: The shapes of the two parameters can be (1, B, S) or (B, S), and the data type is fixed to FLOAT32. This mode is supported when the key and value data types are INT8 or INT4(INT32).
        - per-tensor combined with per-head mode: The shapes of both parameters are (N), and the data type is the same as the query data type. Supported when the key and value data types are INT8 or INT4(INT32).
      - key supports per-channel combined with value supports per-token mode: For key supporting per-channel, the shapes of the two parameters can be (1, N, 1, D), (1, N, D), or (1, H), and the parameter data type is the same as the query data type. For value supporting per-token, the shapes of both parameters are (1, B, S) and the data type is fixed to FLOAT32. Supported when the key and value data types are INT8 or INT4(INT32). When the key and value data types are INT8, the query and output support only FLOAT16.
      - per-token-group mode: the shape of antiquantScale is (1, B, N, S, D/32), the data type is fixed to FLOAT8_E8M0, and antiquantOffset is not supported. It is supported when the data types of key and value are FLOAT4_E2M1.
      - per-token combined with per-head mode: The shapes of both parameters are (B, N, S), and the data type is fixed to FLOAT32. Supported when the key and value data types are INT8 or INT4(INT32).
      - per-token mode using page attention to manage scale/offset mode: The shapes of both parameters are (blocknum, blocksize), and the data type is fixed to FLOAT32. Supported when the key and value data types are INT8.
      - per-token superimposed with per-head mode using page attention to manage scale/offset mode: The shape of both parameters is (blocknum, N, blocksize), the data type is fixed to FLOAT32, and it is supported when the data type of key and value is INT8.
      - When pseudo-quantization parameters and KV-separated quantization parameters are passed in at the same time, the KV-separated quantization parameters take precedence.
    <!-- end id50 -->
    - The INT4 (INT32) pseudo-quantization scenario supports only KV pseudo-quantization parameter separation, specifically including:
      - per-tensor mode;
      - per-channel mode;
      - per-token mode;
      - per-tensor plus per-head mode;
      - per-token plus per-head mode;
      - key supports per-channel plus value supports per-token mode.
    - Some pseudo-quantization scenarios do not support post-quantization

      <!-- npu="910b" id51 -->
      - <term>Atlas A2 products</term>: Post-quantization is not supported in INT4 (INT32) pseudo-quantization scenarios.
      <!-- end id51 -->
      <!-- npu="950" id52 -->
      - <term>Ascend 950PR&950DT products</term>: Post-quantization is not supported in INT4 (INT32) and FLOAT4_E2M1 pseudo-quantization scenarios.
      <!-- end id52 -->

  - Prefix-related parameter constraints:
    - keySharedPrefix and valueSharedPrefix must either both be null or both be non-null.
    - When both keySharedPrefix and valueSharedPrefix are non-null, keySharedPrefix, valueSharedPrefix, key, and value must have the same number of dimensions and keep their dtypes consistent.
    - When both keySharedPrefix and valueSharedPrefix are not empty, the first dimension (batch) of keySharedPrefix's shape must be 1. In BNSD and BSND layouts, the N and D axes must be consistent with those of key. In BSH layout, H must be consistent with that of key. The same applies to valueSharedPrefix. The S of keySharedPrefix and valueSharedPrefix must be equal.
    - When actualSharedPrefixLen exists, its shape must be [1], and its value cannot be greater than the S of keySharedPrefix and valueSharedPrefix.
    - The result of adding the S of the common prefix to the S of key or value must satisfy the original S constraints of key or value.

## Example

The sample code is for reference only. For details about the compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <iostream>
#include <vector>
#include <math.h>
#include <cstring>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_fused_infer_attention_score_v3.h"

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
  // Boilerplate: AscendCL initialization.
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
  // Call aclrtMalloc to request device-side memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host-side data to device-side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate strides of the contiguous tensor.
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
  // 1. (Boilerplate) Device/stream initialization. Refer to the list of AscendCL external APIs.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs. Customize the construction based on the API.
  int32_t batchSize = 1;
  int32_t numHeads = 2;
  int32_t sequenceLengthQ = 1;
  int32_t headDims = 16;
  int32_t numKeyValueHeads = 2;
  int32_t sequenceLengthKV = 16;
  std::vector<int64_t> queryShape = {batchSize, numHeads, sequenceLengthQ, headDims};           // BNSD
  std::vector<int64_t> keyShape = {batchSize, numKeyValueHeads, sequenceLengthKV, headDims};    // BNSD
  std::vector<int64_t> valueShape = {batchSize, numKeyValueHeads, sequenceLengthKV, headDims};  // BNSD
  std::vector<int64_t> attenMaskShape = {batchSize, 1, sequenceLengthQ, sequenceLengthKV};      // B 1 S1 S2
  std::vector<int64_t> outShape = {batchSize, numHeads, sequenceLengthQ, headDims};             // BNSD
  void *queryDeviceAddr = nullptr;
  void *keyDeviceAddr = nullptr;
  void *valueDeviceAddr = nullptr;
  void *attenMaskDeviceAddr = nullptr;
  void *outDeviceAddr = nullptr;
  aclTensor *queryTensor = nullptr;
  aclTensor *keyTensor = nullptr;
  aclTensor *valueTensor = nullptr;
  aclTensor *attenMaskTensor = nullptr;
  aclTensor *outTensor = nullptr;
  int64_t queryShapeSize = GetShapeSize(queryShape);          // BNSD
  int64_t keyShapeSize = GetShapeSize(keyShape);              // BNSD
  int64_t valueShapeSize = GetShapeSize(valueShape);          // BNSD
  int64_t attenMaskShapeSize = GetShapeSize(attenMaskShape);  // B 1 S1 S2
  int64_t outShapeSize = GetShapeSize(outShape);              // BNSD
  std::vector<op::fp16_t> queryHostData(queryShapeSize, 1.0f);
  std::vector<op::fp16_t> keyHostData(keyShapeSize, 1.0f);
  std::vector<op::fp16_t> valueHostData(valueShapeSize, 1.0f);
  std::vector<int8_t> attenMaskHostData(attenMaskShapeSize, 0);
  std::vector<op::fp16_t> outHostData(outShapeSize, 1.0f);

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
  // Create the attenMask aclTensor.
  ret = CreateAclTensor(attenMaskHostData, attenMaskShape, &attenMaskDeviceAddr, aclDataType::ACL_BOOL, &attenMaskTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &outTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> actualSeqlenVector = {sequenceLengthKV};
  auto actualSeqLengths = aclCreateIntArray(actualSeqlenVector.data(), actualSeqlenVector.size());

  double scaleValue = 1 / sqrt(headDims); // 1/sqrt(d)
  int64_t preTokens = 2147483647;
  int64_t nextTokens = 2147483647;
  string sLayerOut = "BNSD";
  char layerOut[sLayerOut.length()+1];
  strcpy(layerOut, sLayerOut.c_str());
  int64_t sparseMode = 0;
  int64_t innerPrecise = 1;
  int blockSize = 0;
  int antiquantMode = 0;
  bool softmaxLseFlag = false;
  int keyAntiquantMode = 0;
  int valueAntiquantMode = 0;
  // 3. Call the CANN operator library API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API.
  ret = aclnnFusedInferAttentionScoreV3GetWorkspaceSize(queryTensor, tensorKeyList, tensorValueList,  nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, numHeads, scaleValue, preTokens, nextTokens, layerOut, numKeyValueHeads, sparseMode, innerPrecise, blockSize, antiquantMode, softmaxLseFlag, keyAntiquantMode, valueAntiquantMode, outTensor, nullptr, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedInferAttentionScoreV3GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API.
  ret = aclnnFusedInferAttentionScoreV3(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedInferAttentionScoreV3 failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value, copy the result from the device memory to the host memory, and modify the code according to the specific API definition.
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
