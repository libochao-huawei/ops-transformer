# aclnnUndGenQkvRmsNormRopeCache

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T16:11:14.467Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
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

- Description: QKV backbone fusion pre-processing operator for multimodal large model inference. The API concatenates the QKV inputs of the undecoded phase and the generated phase into a single output sequence through indirect addressing with `catIndicesOptional`, splits Q/K/V for each token along the head dimension, performs per-head RMSNorm and MRoPE (multimodal rotary position embedding) on Q and K, and leaves V out of normalization and rotation. Q is returned as an independent output, and K and V are written into the paged KV Cache by `slotMapping` (in-place update of `kCacheRef`/`vCacheRef`). The RMSNorm weight is selected between two sets of weights depending on whether the source token falls on the undecoded phase or the generated phase. The computation is performed in float32 throughout, and both the output and the KV Cache are BF16.

- Formula:

  The complete computation process can be decomposed into the following basic computation units.

  Step 1: Based on the indirect addressing relationship of `catIndicesOptional`, retrieve the source data row corresponding to each output token from the QKV input of the undecoded phase or the generated phase:

  $$
  X_t = \begin{cases} X_{\text{und}}[c_t] & \text{ if } c_t < L_{\text{und}} \\ X_{\text{gen}}[c_t - L_{\text{und}}] & \text{ if } c_t \geq L_{\text{und}} \end{cases}
  $$

  Description: `catIndicesOptional` provides the source token index c<sub>t</sub> of the t-th output token, based on which the corresponding QKV data row is retrieved from the understanding-phase input or generation-phase input. The row is split along the head dimension into three segments: Q (H<sub>q</sub> heads), K (H<sub>k</sub> heads), and V (H<sub>v</sub> heads). The RMSNorm weights used subsequently are selected under the same condition, choosing either the und segment (`undWeightsQ`/`undWeightsK`) or the gen segment (`genWeightsQOptional`/`genWeightsKOptional`).

  Step 2: Perform RMSNorm normalization on the D-dimensional vector of each Q or K head:

  $$
  \hat{x}_h = \frac{x_h}{\sqrt{\frac{1}{D} \sum_{i=0}^{D-1} x_{h,i}^2 + \epsilon}} \odot \gamma_h
  $$

  Description: Computes the root mean square value of the D-dimensional vector of a single head, divides the original vector by this value, and then performs element-wise multiplication with the corresponding RMSNorm weight γ. All Q heads of the same token share one Q weight, and all K heads share one K weight. V is not involved in this step.

  Step 3: Apply MRoPE rotary position embedding to the normalized Q and K, splitting the D-dimensional vector into the lower half and the upper half (the front and back halves) for rotation. For a lower-half index l (0 ≤ l < D/2), let p<sub>l</sub> = p[α(l), t], cos<sub>l</sub> = C[p<sub>l</sub>, l], and sin<sub>l</sub> = C[p<sub>l</sub>, l + D/2]:

  $$
  \tilde{x}_{\text{lo},l} = \hat{x}_{\text{lo},l} \cdot \cos_l - \hat{x}_{\text{hi},l} \cdot \sin_l
  $$

  $$
  \tilde{x}_{\text{hi},l} = \hat{x}_{\text{hi},l} \cdot \cos_l + \hat{x}_{\text{lo},l} \cdot \sin_l
  $$

  Description: The first D/2 columns of `cosSinCache` are cos values and the last D/2 columns are sin values. For each lower-half position l, the axis mapping function α(l) determines which axis of `positions` to take the position value from, and the position value is then used as the row index to look up `cosSinCache`. When `mropeSection` is null, all positions are mapped to the temporal axis, degenerating to standard RoPE. All Q heads and K heads of the same token share the same cos/sin values. V is not involved in this step.

  Step 4: Convert the rotated Q and K to BF16 and write them out; directly pass V through and write it to the KV Cache. Let s<sub>t</sub> = s[t]:

  $$
  Q_{\text{out}}[t] = \mathrm{Cast}_{\text{bf16}}(\tilde{q}_t), \quad K_{\text{cache}}\left[\left\lfloor s_t / B_s \right\rfloor,\ s_t \bmod B_s\right] = \mathrm{Cast}_{\text{bf16}}(\tilde{k}_t), \quad V_{\text{cache}}\left[\left\lfloor s_t / B_s \right\rfloor,\ s_t \bmod B_s\right] = v_t
  $$

  Description: Q is written to a separate output tensor in output token order; K is written to the paged KV Cache at the slots specified by `slotMapping`; V is written directly to the paged KV Cache at the slots specified by `slotMapping`, without normalization or rotation. Slot number s<sub>t</sub> is pre-computed by the caller and split into a block number and a row number within the block to locate the corresponding row in the cache. Cache positions not hit by `slotMapping` keep the original values passed in by the caller.

  where X<sub>und</sub> indicates the parameter `undQkv`, X<sub>gen</sub> indicates the parameter `genQkvOptional`, γ<sub>h</sub> indicates the parameter `undWeightsQ`, `undWeightsK`, `genWeightsQOptional`, or `genWeightsKOptional` (the corresponding weight is selected based on whether the source token is from the understanding phase or the generation phase, and whether the current head is a Q head or a K head), C indicates the parameter `cosSinCache`, K<sub>cache</sub> indicates the parameter `kCacheRef`, V<sub>cache</sub> indicates the parameter `vCacheRef`, s indicates the parameter `slotMapping`, p indicates the parameter `positions`, c indicates the parameter `catIndicesOptional`, H<sub>q</sub> indicates the parameter `numHeadsQ`, H<sub>k</sub> indicates the parameter `numHeadsK`, H<sub>v</sub> indicates the parameter `numHeadsV`, ε indicates the parameter `normEps`, m indicates the parameter `mropeSection`, Q<sub>out</sub> indicates the parameter `qOut`, B<sub>s</sub> indicates the second dimension of `kCacheRef` (blockSize), D indicates the head dimension (fixed to 128), and T indicates the total number of tokens (equal to the sum of the first dimension of `undQkv` and that of `genQkvOptional`).

  - L<sub>und</sub>: Number of tokens in the understanding phase, that is, the first dimension of `undQkv`. Non-interface parameter.
  - α(l): MRoPE axis mapping function, which maps each low-half position l (0 ≤ l < D/2) to one of the three axes: temporal (0), height (1), and width (2). The rule is as follows:

    $$
    \alpha(l) = \begin{cases} 1 & l \bmod 3 = 1 \ \text{ and }\ l < 3 m_1 \\ 2 & l \bmod 3 = 2 \ \text{ and }\ l < 3 m_2 \\ 0 & \text{ otherwise} \end{cases}
    $$

    This rule reads only the 2nd and 3rd components (m<sub>1</sub>, m<sub>2</sub>) of `mropeSection`; the 1st component m<sub>0</sub> does not participate in the computation, and the temporal axis serves as the fallback axis that "takes all the rest". Therefore, `mropeSection=[16,16,16]` actually yields temporal/height/width = 32/16/16, and the axis mappings of `[64,16,16]` and `[0,16,16]` are identical element-wise. When `mropeSection` is empty, α(l) ≡ 0, degrading to standard RoPE. Non-interface parameter.

  - cos<sub>l</sub>, sin<sub>l</sub>: Rotation angle components obtained by selecting an axis based on α(l) and looking up `cosSinCache` with the corresponding position value in `positions` as the row number. Non-interface parameter.
  - x<sub>h</sub>: D-dimensional input vector of a single head, split from X<sub>t</sub> by head. Non-interface parameter.
  - x̂<sub>h</sub>: Intermediate result after RMSNorm normalization. Non-interface parameter.
  - x̂<sub>lo</sub> and x̂<sub>hi</sub>: lower half (first D/2 elements) and upper half (last D/2 elements) of the normalized vector, non-interface parameters.
  - x̃<sub>lo</sub> and x̃<sub>hi</sub>: lower-half and upper-half results after MRoPE rotation, non-interface parameters.
  - q̃<sub>t</sub> and k̃<sub>t</sub>: Q and K rotation results of the t-th token, non-interface parameters.
  - v<sub>t</sub>: V data of the t-th token, passed through directly without any conversion, non-interface parameter.
  - Cast<sub>bf16</sub>: data type conversion operation that converts float32 intermediate results to BF16, non-interface parameter.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnUndGenQkvRmsNormRopeCacheGetWorkspaceSize" API first to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call the "aclnnUndGenQkvRmsNormRopeCache" API to execute the computation.

```cpp
aclnnStatus aclnnUndGenQkvRmsNormRopeCacheGetWorkspaceSize(
  const aclTensor   *undQkv,
  const aclTensor   *undWeightsQ,
  const aclTensor   *undWeightsK,
  const aclTensor   *cosSinCache,
  aclTensor         *kCacheRef,
  aclTensor         *vCacheRef,
  const aclTensor   *slotMapping,
  const aclTensor   *positions,
  const aclTensor   *genQkvOptional,
  const aclTensor   *genWeightsQOptional,
  const aclTensor   *genWeightsKOptional,
  const aclTensor   *catIndicesOptional,
  int64_t            numHeadsQ,
  int64_t            numHeadsK,
  int64_t            numHeadsV,
  double             normEps,
  const aclIntArray *mropeSection,
  aclTensor         *qOut,
  uint64_t          *workspaceSize,
  aclOpExecutor     **executor)
```

```cpp
aclnnStatus aclnnUndGenQkvRmsNormRopeCache(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnUndGenQkvRmsNormRopeCacheGetWorkspaceSize

- Mandatory

  <div style="overflow-x: auto;">
    <table style="table-layout: fixed; width: 1500px;"><colgroup>
      <col style="width: 180px;">
      <col style="width: 120px;">
      <col style="width: 300px;">
      <col style="width: 350px;">
      <col style="width: 250px;">
      <col style="width: 100px;">
      <col style="width: 100px;">
      <col style="width: 100px;">
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
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>undQkv(const aclTensor*)</td>
        <td>Input</td>
        <td>X<sub>und</sub> in the formula, which indicates the QKV input tensor in the understanding phase.</td>
        <td><ul><li>Does not support null pointer or null Tensor; <code>und_len</code> must be a positive number.</li><li>The second dimension N must equal <code>numHeadsQ + numHeadsK + numHeadsV</code>, and the tensor is split into Q/K/V in order along this dimension.</li></ul></td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td><code>[und_len,N,D]</code></td>
        <td>√</td>
      </tr>
      <tr>
        <td>undWeightsQ(const aclTensor*)</td>
        <td>Input</td>
        <td>γ<sub>h</sub> in the formula, indicating the RMSNorm weight of Q in the understanding phase.</td>
        <td><ul><li>Does not support null pointer or null tensor.</li><li>When the source token falls on the understanding segment, all Q heads of the token share this weight.</li></ul></td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td><code>[D]</code></td>
        <td>√</td>
      </tr>
      <tr>
        <td>undWeightsK(const aclTensor*)</td>
        <td>Input</td>
        <td>γ<sub>h</sub> in the formula, indicating the RMSNorm weight of K in the understanding phase.</td>
        <td><ul><li>Does not support null pointer or null tensor.</li><li>When the source token falls on the understanding segment, all K heads of the token share this weight.</li></ul></td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td><code>[D]</code></td>
        <td>√</td>
      </tr>
      <tr>
        <td>cosSinCache (const aclTensor*)</td>
        <td>Input</td>
        <td>C in the formula, indicating the cos/sin cache table for rotary position embedding.</td>
        <td><ul><li>Does not support null pointers or null Tensors; <code>max_pos</code> must be a positive number.</li><li>The first <code>D/2</code> columns are cos, and the last <code>D/2</code> columns are sin.</li><li>The first dimension must cover the maximum position value in <code>positions</code>.</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td><code>[max_pos,D]</code></td>
        <td>√</td>
      </tr>
      <tr>
        <td>kCacheRef (aclTensor*)</td>
        <td>Input/Output</td>
        <td>K<sub>cache</sub> in the formula, indicates the paged KV Cache buffer of K.</td>
        <td><ul><li>Does not support null pointer or null Tensor.</li><li>Pre-allocated by the caller; the API performs in-place update based on the passed-in Tensor.</li><li>Must be a memory-contiguous Tensor; the API directly returns failure if the Tensor is non-contiguous.</li><li>The capacity must satisfy <code>Bn * Bs >= T</code>.</li></ul></td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td><code>[Bn,Bs,Hk,D]</code></td>
        <td>×</td>
      </tr>
      <tr>
        <td>vCacheRef (aclTensor*)</td>
        <td>Input/Output</td>
        <td>V<sub>cache</sub> in the formula, indicating the paged KV Cache buffer of V.</td>
        <td><ul><li>Does not support null pointer or null tensor.</li><li>Pre-allocated by the caller. The API performs in-place update based on the passed-in tensor.</li><li>Must be stored contiguously in memory; otherwise, the API directly returns failure.</li><li><code>Bn</code> and <code>Bs</code> must be consistent with <code>kCacheRef</code>.</li></ul></td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td><code>[Bn,Bs,Hv,D]</code></td>
        <td>×</td>
      </tr>
      <tr>
        <td>slotMapping (const aclTensor*)</td>
        <td>Input</td>
        <td>s in the formula, indicates the slot number for writing each output token into the KV Cache.</td>
        <td><ul><li>Does not support null pointer or null tensor.</li><li>The slot number is pre-computed by the caller: <code>slot = blockIdx * Bs + rowIdx</code>.</li><li>The value range must be <code>[0, Bn*Bs-1]</code>, and slot numbers must be unique; for duplicate slots, the write order and result are undefined on multiple cores.</li></ul></td>
        <td>INT64</td>
        <td>ND</td>
        <td><code>[T]</code></td>
        <td>√</td>
      </tr>
      <tr>
        <td>positions (const aclTensor*)</td>
        <td>Input</td>
        <td>p in the formula, indicates the MRoPE three-axis position table, where the three rows are the temporal axis, height axis, and width axis in sequence.</td>
        <td><ul><li>Does not support null pointer or null tensor.</li><li>The first dimension is fixed to 3.</li><li>The value must fall on <code>[0, max_pos-1]</code>.</li></ul></td>
        <td>INT64</td>
        <td>ND</td>
        <td><code>[3,T]</code></td>
        <td>√</td>
      </tr>
      <tr>
        <td>genQkvOptional (const aclTensor*)</td>
        <td>Optional input</td>
        <td>X<sub>gen</sub> in the formula, indicates the QKV input tensor of the generation phase.</td>
        <td><ul><li>In the current version, it must be passed in and <code>gen_len</code> must be a positive number. Passing in nullptr (pure prefill) is not supported yet.</li><li>The N and D dimensions must be consistent with those of <code>undQkv</code>.</li></ul></td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td><code>[gen_len,N,D]</code></td>
        <td>√</td>
      </tr>
      <tr>
        <td>genWeightsQOptional (const aclTensor*)</td>
        <td>Optional input</td>
        <td>Indicates γ<sub>h</sub> in the formula, the RMSNorm weight of Q in the generation phase.</td>
        <td><ul><li>In the current version, this parameter must be passed in; passing nullptr is not supported yet.</li><li>When <code>genQkvOptional</code> is passed in, this parameter must be passed in together with <code>genWeightsKOptional</code>.</li></ul></td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td><code>[D]</code></td>
        <td>√</td>
      </tr>
      <tr>
        <td>genWeightsKOptional(const aclTensor*)</td>
        <td>Optional input</td>
        <td>γ<sub>h</sub> in the formula, which indicates the RMSNorm weight of K in the generation phase.</td>
        <td><ul><li>In the current version, it must be passed in. Passing in nullptr is not supported yet.</li><li>When <code>genQkvOptional</code> is passed in, it must be passed in together with <code>genWeightsQOptional</code>.</li></ul></td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td><code>[D]</code></td>
        <td>√</td>
      </tr>
      <tr>
        <td>catIndicesOptional(const aclTensor*)</td>
        <td>Optional input</td>
        <td>c in the formula indicates the mapping table from output tokens to source tokens.</td>
        <td><ul><li>Must be passed in for the current version. Passing nullptr (identity mapping for a single sequence) is not supported yet.</li><li>The value should fall on <code>[0, T-1]</code>; values less than <code>und_len</code> fall on the understanding phase, and the others fall on the generation phase.</li></ul></td>
        <td>INT64</td>
        <td>ND</td>
        <td><code>[T]</code></td>
        <td>√</td>
      </tr>
      <tr>
        <td>numHeadsQ (int64_t)</td>
        <td>Input</td>
        <td>H<sub>q</sub> in the formula indicates the number of Q heads.</td>
        <td><ul><li>Must be a positive number.</li><li><code>(numHeadsQ, numHeadsK, numHeadsV)</code> currently only supports two combinations: <code>(8,1,1)</code> and <code>(16,2,2)</code>.</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>numHeadsK (int64_t)</td>
        <td>Input</td>
        <td>H<sub>k</sub> in the formula, indicates the number of K heads.</td>
        <td><ul><li>Must be a positive number and must equal the third dimension of <code>kCacheRef</code>.</li><li>Currently only two combinations of <code>(numHeadsQ, numHeadsK, numHeadsV)</code> are supported: <code>(8,1,1)</code> and <code>(16,2,2)</code>.</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>numHeadsV (int64_t)</td>
        <td>Input</td>
        <td>H<sub>v</sub> in the formula, indicating the number of heads of V.</td>
        <td><ul><li>Must be a positive number and must equal the third dimension of <code>vCacheRef</code>.</li><li><code>(numHeadsQ, numHeadsK, numHeadsV)</code> currently only supports two combinations: <code>(8,1,1)</code> and <code>(16,2,2)</code>.</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>normEps (double)</td>
        <td>Input</td>
        <td>ε in the formula, which indicates the parameter for preventing division by zero in RMSNorm.</td>
        <td><ul><li>Must be a positive number. <code>1e-6</code> is recommended.</li><li>Participates in the computation as FLOAT inside the API.</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>mropeSection (const aclIntArray*)</td>
        <td>Input</td>
        <td>m in the formula indicates the MRoPE three-axis segmentation and determines the axis mapping function α(l).</td>
        <td><ul><li>A null pointer or null array can be passed in. In this case, it is processed as <code>[D/2,0,0]</code> and degenerates to standard RoPE.</li><li>If not empty, the length must be 3, each element must be non-negative, and the sum of the three elements must not exceed <code>D/2</code>.</li><li>Only the 2nd and 3rd elements participate in the axis mapping, and the 1st element does not participate in the computation. For details, see the definition of α(l) in "Function".</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>Length: 3</td>
        <td>-</td>
      </tr>
      <tr>
        <td>qOut (aclTensor*)</td>
        <td>Output</td>
        <td>Q<sub>out</sub> in the formula indicates the output Q tensor after normalization and rotation.</td>
        <td><ul><li>Does not support a null pointer or null tensor.</li><li>Pre-allocated by the caller. The shape must be <code>[T, numHeadsQ, D]</code>.</li></ul></td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td><code>[T,Hq,D]</code></td>
        <td>×</td>
      </tr>
      <tr>
        <td>workspaceSize (uint64_t*)</td>
        <td>Output</td>
        <td>Returns the workspace size that needs to be applied by the Device side.</td>
        <td>Does not support null pointers.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor(aclOpExecutor**)</td>
        <td>Output</td>
        <td>Returns the op executor containing the operator computation flow.</td>
        <td>Does not support null pointers.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>
  </div>

- Return Value

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:
  <table style="table-layout: fixed; width: 1000px;"><colgroup>
    <col style="width: 300px;">
    <col style="width: 150px;">
    <col style="width: 550px;">
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
      <td rowspan="2" style="white-space: nowrap">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="2">161001</td>
      <td>Any of <code>undQkv</code>, <code>undWeightsQ</code>, <code>undWeightsK</code>, <code>cosSinCache</code>, <code>kCacheRef</code>, <code>vCacheRef</code>, <code>slotMapping</code>, <code>positions</code>, <code>qOut</code>, <code>workspaceSize</code>, or <code>executor</code> is a null pointer.</td>
    </tr>
    <tr>
      <td><code>genQkvOptional</code> is provided, but <code>genWeightsQOptional</code> or <code>genWeightsKOptional</code> is a null pointer.</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td><code>kCacheRef</code> or <code>vCacheRef</code> is a non-contiguous tensor.</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
      <td>561101</td>
      <td>Failed to create <code>aclOpExecutor</code>, or the Contiguous processing of the input fails.</td>
    </tr>
  </tbody>
  </table>

  > The complete validation of data types, shapes, and attribute values is implemented in a unified manner at the tiling layer (both the aclnn single-operator path and the graph mode eventually go through tiling, avoiding drift caused by implementing the same constraint repeatedly in multiple places). The tradeoff is that such invalid inputs do not trigger errors in the first-phase API; instead, tiling returns failure during the execution of the second-phase API `aclnnUndGenQkvRmsNormRopeCache`. See host-side logs for the specific cause.

## aclnnUndGenQkvRmsNormRopeCache

- Mandatory

  <table style="table-layout: fixed; width: 1000px"><colgroup>
    <col style="width: 180px">
    <col style="width: 120px">
    <col style="width: 700px">
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
      <td>Address of the workspace memory applied on the Device side. If the <code>workspaceSize</code> returned by the first-phase API is 0, <code>nullptr</code> can be passed in.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace memory applied on the Device side, obtained by the first-phase API aclnnUndGenQkvRmsNormRopeCacheGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, containing the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- Return Value

  aclnnStatus: Returned status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation: aclnnUndGenQkvRmsNormRopeCache defaults to a deterministic implementation.

<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT products</term>:

  - **General Constraints**
    <a id="general-constraints"></a>
    - The computation uses float32 throughout, and the KV Cache is stored in BF16. FP8/INT8 quantization is not supported.
    - headDim is fixed to 128.
    - `(numHeadsQ, numHeadsK, numHeadsV)` supports only two combinations: `(8,1,1)` and `(16,2,2)`.
    - numHeadsQ + numHeadsK + numHeadsV must equal the second dimension N of `undQkv`.
    - und_len must be a positive number; T = und_len + gen_len must be a positive number.
    - No artificial upper limit is set for T. The actual upper limit is determined by the KV Cache capacity, requiring blockNum × blockSize ≥ T. On-device verification has been performed up to T = 64K.
    - No limit is set for blockSize: cache is forced to be contiguous BBND, and `[Bn, Bs, N, D]` is equivalent to `[Bn*Bs, N, D]` after flattening. The slot number is directly used as the flattened row number. Bs is not involved in address computation, multi-core partitioning, or UB tiling. On-device verification has covered Bs ∈ {16, 64, 100, 128, 256, 512}, including non-powers of two.
    - The Bn and Bs of `vCacheRef` must be consistent with those of `kCacheRef`.
    - The third dimension of `kCacheRef` must equal numHeadsK, and the fourth dimension must equal headDim (128).
    - The third dimension of `vCacheRef` must equal numHeadsV, and the fourth dimension must equal headDim (128).
    - `kCacheRef` and `vCacheRef` must be memory-contiguous tensors. Non-contiguous tensors (including the BNBD physical layout) are not supported.
    - The N and D dimensions of `genQkvOptional` must be consistent with those of `undQkv`.
    - `undWeightsQ`, `undWeightsK`, `genWeightsQOptional`, and `genWeightsKOptional` must all be 1D `[D]` tensors, where D = headDim = 128.
    - `cosSinCache` must be a 2D `[max_pos, D]` tensor, where max_pos is a positive number and D = headDim = 128.
    - `slotMapping` must be a 1D `[T]` tensor; `catIndicesOptional` must be a 1D `[T]` tensor; `positions` must be a 2D `[3, T]` tensor, with the first dimension fixed to 3.
    - `qOut` must be a 3D `[T, numHeadsQ, D]` tensor.
    - `normEps` must be a positive number.
    - If `mropeSection` is a null pointer or null array, it is treated as `[D/2, 0, 0]`; if not empty, its length must be 3, each element must be non-negative, and the sum of the three elements must not exceed D/2.
    - **In the current version, all four parameters `genQkvOptional`, `genWeightsQOptional`, `genWeightsKOptional`, and `catIndicesOptional` must be passed in, and gen_len must be a positive number**. The degenerate paths corresponding to passing in nullptr (pure prefill, single-sequence identity mapping) are not supported yet; calling the second-phase API returns failure. This restriction will be lifted in later versions.
  - **Input value range constraints** (runtime data that cannot be verified on the host side; guaranteed by the caller)
    - The value of `slotMapping` must fall on `[0, Bn*Bs-1]` and all values must be distinct. If duplicate slots occur, multiple cores write to the same cache line, and both the write order and the final result are indeterminate.
    - The value of `positions` must fall on `[0, max_pos-1]`; otherwise, `cosSinCache` is accessed out of bounds.
    - The value of `catIndicesOptional` must fall on `[0, T-1]`.

<!-- end id7 -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
/*!
 * \file test_aclnn_und_gen_qkv_rms_norm_rope_cache.cpp
 * \brief aclnnUndGenQkvRmsNormRopeCache two-phase API call example (including slot_mapping pre-computation)
 */

#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_und_gen_qkv_rms_norm_rope_cache.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)     \
    do {                            \
        printf(message, ##__VA_ARGS__); \
    } while (0)

namespace {
constexpr int64_t HEAD_DIM = 128;
constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t NUM_HEADS_Q = 8;
constexpr int64_t NUM_HEADS_K = 1;
constexpr int64_t NUM_HEADS_V = 1;
constexpr int64_t UND_LEN = 5;
constexpr int64_t GEN_LEN = 3;
constexpr int64_t MAX_POS = 32;
constexpr int64_t MROPE_AXIS_NUM = 3;
constexpr float NORM_EPS = 1e-6f;

// bf16 is carried by uint16_t: take the high 16 bits of float32 with round-to-nearest
uint16_t FloatToBf16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    uint32_t rounded = bits + 0x7FFFU + ((bits >> 16) & 1U);
    return static_cast<uint16_t>(rounded >> 16);
}

float Bf16ToFloat(uint16_t value)
{
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
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
                    aclDataType dataType, aclTensor** tensor, aclFormat format = ACL_FORMAT_ND)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format, shape.data(),
                              shape.size(), *deviceAddr);
    return 0;
}
} // namespace

int main()
{
    // 1. Initialize the device and stream
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    const int64_t total = UND_LEN + GEN_LEN;
    const int64_t numHead = NUM_HEADS_Q + NUM_HEADS_K + NUM_HEADS_V;
    const int64_t blockNum = (total + BLOCK_SIZE - 1) / BLOCK_SIZE + 1;

    // 2. Construct the input shape
    std::vector<int64_t> undQkvShape = {UND_LEN, numHead, HEAD_DIM};
    std::vector<int64_t> genQkvShape = {GEN_LEN, numHead, HEAD_DIM};
    std::vector<int64_t> weightShape = {HEAD_DIM};
    std::vector<int64_t> cosSinShape = {MAX_POS, HEAD_DIM};
    std::vector<int64_t> kCacheShape = {blockNum, BLOCK_SIZE, NUM_HEADS_K, HEAD_DIM};
    std::vector<int64_t> vCacheShape = {blockNum, BLOCK_SIZE, NUM_HEADS_V, HEAD_DIM};
    std::vector<int64_t> slotMappingShape = {total};
    std::vector<int64_t> positionsShape = {MROPE_AXIS_NUM, total};
    std::vector<int64_t> catIndicesShape = {total};
    std::vector<int64_t> qShape = {total, NUM_HEADS_Q, HEAD_DIM};

    // 3. Create host data (bf16 is carried by uint16_t)
    // Fill QKV with non-zero data that varies slowly with position, to prevent an all-zero input from degrading the sample into a trivial run that always equals 0
    std::vector<uint16_t> undQkvHost(GetShapeSize(undQkvShape));
    for (size_t i = 0; i < undQkvHost.size(); ++i) {
        undQkvHost[i] = FloatToBf16(0.05f * static_cast<float>(i % 17) - 0.4f);
    }
    std::vector<uint16_t> genQkvHost(GetShapeSize(genQkvShape));
    for (size_t i = 0; i < genQkvHost.size(); ++i) {
        genQkvHost[i] = FloatToBf16(0.05f * static_cast<float>(i % 13) - 0.3f);
    }
    // RMSNorm weight: set to 1.0 for the und segment and 0.5 for the gen segment, to easily check whether the weights of the two segments are correctly selected by catIndicesOptional
    std::vector<uint16_t> undWeightsQHost(HEAD_DIM, FloatToBf16(1.0f));
    std::vector<uint16_t> undWeightsKHost(HEAD_DIM, FloatToBf16(1.0f));
    std::vector<uint16_t> genWeightsQHost(HEAD_DIM, FloatToBf16(0.5f));
    std::vector<uint16_t> genWeightsKHost(HEAD_DIM, FloatToBf16(0.5f));
    // cos/sin cache table: the first D/2 columns are cos and the last D/2 columns are sin, generated using the theta of standard RoPE.
    std::vector<float> cosSinHost(GetShapeSize(cosSinShape));
    for (int64_t pos = 0; pos < MAX_POS; ++pos) {
        for (int64_t i = 0; i < HEAD_DIM / 2; ++i) {
            float freq = 1.0f / std::pow(10000.0f, 2.0f * static_cast<float>(i) / static_cast<float>(HEAD_DIM));
            float angle = static_cast<float>(pos) * freq;
            cosSinHost[pos * HEAD_DIM + i] = std::cos(angle);
            cosSinHost[pos * HEAD_DIM + HEAD_DIM / 2 + i] = std::sin(angle);
        }
    }
    // The KV Cache is pre-allocated by the caller and updated in place. Rows not hit by slot_mapping retain the passed-in values.
    std::vector<uint16_t> kCacheHost(GetShapeSize(kCacheShape), 0);
    std::vector<uint16_t> vCacheHost(GetShapeSize(vCacheShape), 0);
    std::vector<uint16_t> qHost(GetShapeSize(qShape), 0);

    // slot_mapping is pre-computed by the caller: slot = blockIdx * blockSize + rowIdx.
    std::vector<int64_t> slotMappingHost(total);
    for (int64_t t = 0; t < total; ++t) {
        int64_t blockIdx = t / BLOCK_SIZE;
        int64_t rowIdx = t % BLOCK_SIZE;
        slotMappingHost[t] = blockIdx * BLOCK_SIZE + rowIdx;
    }
    // positions: three-axis positions
    std::vector<int64_t> positionsHost(MROPE_AXIS_NUM * total);
    for (int64_t axis = 0; axis < MROPE_AXIS_NUM; ++axis) {
        for (int64_t t = 0; t < total; ++t) {
            positionsHost[axis * total + t] = t % MAX_POS;
        }
    }
    // cat_indices: und/gen interleaved, out_t -> src_t
    std::vector<int64_t> catIndicesHost = {0, 5, 1, 6, 2, 7, 3, 4};

    // 4. Create aclTensor.
    void *undQkvDev = nullptr, *genQkvDev = nullptr, *undWqDev = nullptr, *undWkDev = nullptr;
    void *genWqDev = nullptr, *genWkDev = nullptr, *cosSinDev = nullptr, *kCacheDev = nullptr;
    void *vCacheDev = nullptr, *slotMappingDev = nullptr, *positionsDev = nullptr, *catIndicesDev = nullptr;
    void* qDev = nullptr;
    aclTensor *undQkv = nullptr, *genQkvOptional = nullptr, *undWq = nullptr, *undWk = nullptr, *genWq = nullptr;
    aclTensor *genWk = nullptr, *cosSin = nullptr, *kCacheRef = nullptr, *vCacheRef = nullptr, *slotMapping = nullptr;
    aclTensor *positions = nullptr, *catIndicesOptional = nullptr, *qOut = nullptr;

    ret = CreateAclTensor(undQkvHost, undQkvShape, &undQkvDev, ACL_BF16, &undQkv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(undWeightsQHost, weightShape, &undWqDev, ACL_BF16, &undWq);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(undWeightsKHost, weightShape, &undWkDev, ACL_BF16, &undWk);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cosSinHost, cosSinShape, &cosSinDev, ACL_FLOAT, &cosSin);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kCacheHost, kCacheShape, &kCacheDev, ACL_BF16, &kCacheRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(vCacheHost, vCacheShape, &vCacheDev, ACL_BF16, &vCacheRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(slotMappingHost, slotMappingShape, &slotMappingDev, ACL_INT64, &slotMapping);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(positionsHost, positionsShape, &positionsDev, ACL_INT64, &positions);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(genQkvHost, genQkvShape, &genQkvDev, ACL_BF16, &genQkvOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(genWeightsQHost, weightShape, &genWqDev, ACL_BF16, &genWq);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(genWeightsKHost, weightShape, &genWkDev, ACL_BF16, &genWk);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(catIndicesHost, catIndicesShape, &catIndicesDev, ACL_INT64, &catIndicesOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(qHost, qShape, &qDev, ACL_BF16, &qOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // mrope_section: three-axis segments
    std::vector<int64_t> mropeSectionData = {16, 16, 16};
    aclIntArray* mropeSection = aclCreateIntArray(mropeSectionData.data(), mropeSectionData.size());

    // 5. First-phase API: calculate the workspace size.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnUndGenQkvRmsNormRopeCacheGetWorkspaceSize(
        undQkv, undWq, undWk, cosSin, kCacheRef, vCacheRef, slotMapping, positions, genQkvOptional, genWq, genWk, catIndicesOptional,
        NUM_HEADS_Q, NUM_HEADS_K, NUM_HEADS_V, NORM_EPS, mropeSection, qOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnUndGenQkvRmsNormRopeCacheGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 6. Second-phase API: execute the computation.
    ret = aclnnUndGenQkvRmsNormRopeCache(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUndGenQkvRmsNormRopeCache failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 7. Copy back results: qOut is read directly; the corresponding rows of k_cache/v_cache are located by slot_mapping.
    auto qSize = GetShapeSize(qShape);
    std::vector<uint16_t> qResult(qSize, 0);
    ret = aclrtMemcpy(qResult.data(), qResult.size() * sizeof(qResult[0]), qDev, qSize * sizeof(uint16_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy qOut result failed. ERROR: %d\n", ret); return ret);

    auto kCacheSize = GetShapeSize(kCacheShape);
    std::vector<uint16_t> kCacheResult(kCacheSize, 0);
    ret = aclrtMemcpy(kCacheResult.data(), kCacheResult.size() * sizeof(kCacheResult[0]), kCacheDev,
                      kCacheSize * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy k_cache result failed. ERROR: %d\n", ret); return ret);

    // Print the first several results. k_cache uses slot_mapping[0] to locate the written row.
    constexpr int64_t PRINT_NUM = 8;
    for (int64_t i = 0; i < PRINT_NUM; ++i) {
        LOG_PRINT("qOut[%ld] = %f\n", i, Bf16ToFloat(qResult[i]));
    }
    int64_t kCacheRowOffset = slotMappingHost[0] * NUM_HEADS_K * HEAD_DIM;
    for (int64_t i = 0; i < PRINT_NUM; ++i) {
        LOG_PRINT("kCache[slot %ld][%ld] = %f\n", slotMappingHost[0], i,
                  Bf16ToFloat(kCacheResult[kCacheRowOffset + i]));
    }
    LOG_PRINT("run aclnnUndGenQkvRmsNormRopeCache success, qOut size = %ld, k_cache size = %ld\n", qSize, kCacheSize);

    // 8. Release resources.
    aclDestroyTensor(undQkv);
    aclDestroyTensor(undWq);
    aclDestroyTensor(undWk);
    aclDestroyTensor(cosSin);
    aclDestroyTensor(kCacheRef);
    aclDestroyTensor(vCacheRef);
    aclDestroyTensor(slotMapping);
    aclDestroyTensor(positions);
    aclDestroyTensor(genQkvOptional);
    aclDestroyTensor(genWq);
    aclDestroyTensor(genWk);
    aclDestroyTensor(catIndicesOptional);
    aclDestroyTensor(qOut);
    aclDestroyIntArray(mropeSection);

    aclrtFree(undQkvDev);
    aclrtFree(undWqDev);
    aclrtFree(undWkDev);
    aclrtFree(cosSinDev);
    aclrtFree(kCacheDev);
    aclrtFree(vCacheDev);
    aclrtFree(slotMappingDev);
    aclrtFree(positionsDev);
    aclrtFree(genQkvDev);
    aclrtFree(genWqDev);
    aclrtFree(genWkDev);
    aclrtFree(catIndicesDev);
    aclrtFree(qDev);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}

```
