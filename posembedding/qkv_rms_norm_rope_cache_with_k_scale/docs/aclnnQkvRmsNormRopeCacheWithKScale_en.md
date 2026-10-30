# aclnnQkvRmsNormRopeCacheWithKScale

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T16:02:44.051Z -->

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

- Description: In a single call, `aclnnQkvRmsNormRopeCacheWithKScale` performs fused QKV splitting, Q/K RMSNorm and position encoding, executes Q/K matrix multiplication or quantization based on the scenario, and updates the PagedAttention KV Cache. Based on the scenario, the Q branch writes to `qOut` and `qScaleOptional`; based on `slotMapping`, the K and V branches update `kCacheRef`, `vCacheRef`, and `kScaleCacheRef` in place. Currently, three complete semantic scenarios are supported:
  - RoPE: Q/K perform `RMSNorm -> RoPE -> rotation -> PerTokenPerHead quantization`; Q/K data is FP8 E4M3FN and scale is FP32;
  - M-RoPE: Q/K perform `RMSNorm -> M-RoPE -> rotation`; Q is directly output as BF16, and K performs PerTokenPerHead INT8 quantization;
  - M-RoPE MX: Q/K performs `RMSNorm -> M-RoPE -> Dynamic MX Quant` and does not execute `rotationOptional`. Each D32 block is quantized to FP8 E4M3FN using the FLOAT8_E8M0 power-of-two scale of cuBLAS MX FP8.
  The V branch does not perform RMSNorm, position encoding, or matrix multiplication. In the M-RoPE MX use case, `vScaleOptional[Nv,D]` is used for per-head-per-channel static quantization, and the output is FP8 E4M3FN.

- Formula:

  The following formulas are described based on the logical `[T,N,D]` layout. `layoutQkv` and `layoutQOut` only determine the physical order of the T and N axes in the tensor and do not change the element-wise computation. $T$ indicates the total number of tokens, $N_q/N_k/N_v$ indicates the number of Q/K/V heads, and $D=128$ indicates the head dimension.

  The computation consists of QKV splitting, normalization, position encoding, use-case-specific matrix multiplication/quantization, and cache write-back.

  **1. Q/K/V Splitting and Q/K RMSNorm**

  Denote the logical view of `qkv` as $\boldsymbol X$, then

  $$
  \begin{aligned}
  \boldsymbol X
  &=[\,\boldsymbol Q\mid\boldsymbol K\mid\boldsymbol V\,]_{N},\\
  \boldsymbol Q&\in\mathbb R^{T\times N_q\times D},\qquad
  \boldsymbol K\in\mathbb R^{T\times N_k\times D},\qquad
  \boldsymbol V\in\mathbb R^{T\times N_v\times D}.
  \end{aligned}
  $$

  The split boundaries along the logical head axis are

  $$
  \boldsymbol X_{u,n,:}=
  \begin{cases}
  \boldsymbol Q_{u,n,:}, & 0\leq n<N_q,\\
  \boldsymbol K_{u,n-N_q,:}, & N_q\leq n<N_q+N_k,\\
  \boldsymbol V_{u,n-N_q-N_k,:}, & N_q+N_k\leq n<N_q+N_k+N_v.
  \end{cases}
  $$

  For $\boldsymbol A\in\{\boldsymbol Q,\boldsymbol K\}$, let $\boldsymbol\gamma^{Q}$ and $\boldsymbol\gamma^{K}$ correspond to `qGamma` and `kGamma` respectively. The RMSNorm result is

  $$
  Y^{A}_{u,n,d}
  =
  \frac{A_{u,n,d}\gamma^{A}_{d}}
  {\sqrt{\epsilon+\dfrac{1}{D}\displaystyle\sum_{j=0}^{D-1}A_{u,n,j}^{2}}},
  \qquad 0\leq d<D.
  $$

  **2. Position Parameter Parsing and cos/sin Selection**

  **RoPE Position Input**

  For batch $b$, the current call contains $L_b$ tokens. For the $i$-th token in the batch, its global token index $u$ and RoPE position $p_u$ are

  $$
  \begin{aligned}
  L_b
  &=\mathrm{queryStartLocOptional}_{b+1}
    -\mathrm{queryStartLocOptional}_{b},\\
  u
  &=\mathrm{queryStartLocOptional}_{b}+i,\qquad 0\leq i<L_b,\\
  p_u
  &=\mathrm{seqLensOptional}_{b}-L_b+i.
  \end{aligned}
  $$

  Let $D_{\mathrm{half}}=D/2$. Then, the cos and sin used by token $u$ are

  $$
  c_{u,\ell}
  =\mathrm{cosSin}_{p_u,\ell},
  \qquad
  s_{u,\ell}
  =\mathrm{cosSin}_{p_u,D_{\mathrm{half}}+\ell},
  \qquad 0\leq\ell<D_{\mathrm{half}}.
  $$

  **M-RoPE Position Input**

  Each row of `mropePositionOptional` corresponds to a token, and its three columns sequentially specify the `cosSin` row indices for the T/H/W channels. The column mapping is defined as

  $$
  \iota(\mathrm T)=0,\qquad
  \iota(\mathrm H)=1,\qquad
  \iota(\mathrm W)=2,
  $$

  as well as the three raw position encodings

  $$
  R_{a,u,d}
  =
  \mathrm{cosSin}_{
    \mathrm{mropePositionOptional}_{u,\iota(a)},d},
  \qquad
  a\in\{\mathrm T,\mathrm H,\mathrm W\}.
  $$

  Denote `mropeSectionOptional=[t,h,w]` as

  $$
  \boldsymbol{s}
  =(s_{\mathrm T},s_{\mathrm H},s_{\mathrm W})
  =(t,h,w).
  $$

  $s_{\mathrm T}$ does not participate in lane source selection and is not subject to an independent lane capacity limit check. It must still be non-negative and is counted in the three-term sum check. The T path is the default source for the half-lane, and H/W replace the corresponding lanes within their respective coverage ranges. For $0\leq\ell<D_{\mathrm{half}}$, define

  $$
  r(\ell)=\left\lfloor\frac{\ell}{3}\right\rfloor,\qquad
  \rho(\ell)=\ell\bmod 3,
  $$

  $$
  \sigma(\ell)=
  \begin{cases}
  \mathrm H, & \rho(\ell)=1\ \land\ r(\ell)<s_{\mathrm H},\\
  \mathrm W, & \rho(\ell)=2\ \land\ r(\ell)<s_{\mathrm W},\\
  \mathrm T, & \text{ otherwise}.
  \end{cases}
  $$

  The final cos and sin used by M-RoPE are

  $$
  c_{u,\ell}=R_{\sigma(\ell),u,\ell},
  \qquad
  s_{u,\ell}=R_{\sigma(\ell),u,D_{\mathrm{half}}+\ell}.
  $$

  M-RoPE only changes how cos/sin values are obtained. The head dimension of the Q/K output remains $D$ and is not extended to $3D$.

  **3. Q/K Position Encoding**

  For $\boldsymbol A\in\{\boldsymbol Q,\boldsymbol K\}$, the RMSNorm result $\boldsymbol Y^A$ is split into the front and back halves along the head dimension. The half-split position encoding is

  $$
  \begin{aligned}
  Z^{A,\mathrm{low}}_{u,n,\ell}
  &=Y^{A,\mathrm{low}}_{u,n,\ell}c_{u,\ell}
    -Y^{A,\mathrm{high}}_{u,n,\ell}s_{u,\ell},\\
  Z^{A,\mathrm{high}}_{u,n,\ell}
  &=Y^{A,\mathrm{high}}_{u,n,\ell}c_{u,\ell}
    +Y^{A,\mathrm{low}}_{u,n,\ell}s_{u,\ell}.
  \end{aligned}
  $$

  **4. Q/K Shared Matrix Multiplication in RoPE and M-RoPE Scenarios**

  In RoPE and M-RoPE, let $\boldsymbol W$ denote `rotationOptional`. After the position encoding result is converted to BF16, it is right-multiplied by the shared matrix to obtain the FP32 results $\boldsymbol H^Q$ and $\boldsymbol H^K$:

  $$
  \boldsymbol H^A_{u,n,:}
  =
  \operatorname{BF16}\!\left(\boldsymbol Z^A_{u,n,:}\right)
  \boldsymbol W,
  \qquad A\in\{Q,K\}.
  $$

  In M-RoPE MX, `rotationOptional` must be empty. The M-RoPE result $\boldsymbol Z^Q/\boldsymbol Z^K$ skips this matrix multiplication and directly enters MX quantization.

  **5. Q/K Output and Dynamic Quantization**

  For $\boldsymbol H^A$ obtained through the shared matrix multiplication in RoPE and M-RoPE scenarios, the output data type and whether to perform PerTokenPerHead dynamic quantization are determined by "Constraints". When dynamic quantization is not performed, the result is directly converted to the output data type specified in the constraints $\tau_A$:

  $$
  O^A_{u,n,d}=\operatorname{cast}_{\tau_A}\!\left(H^A_{u,n,d}\right),
  \qquad A\in\{Q,K\}.
  $$

  For branches that perform PerTokenPerHead dynamic quantization, let $M_{\tau_A}$ denote the forward quantization upper bound corresponding to the target type. The per-token and per-head quantization formula is:

  $$
  M_{\tau_A}=
  \begin{cases}
  448, & \tau_A=\mathrm{FP8\ E4M3FN},\\
  127, & \tau_A=\mathrm{INT8},
  \end{cases}
  $$

  $$
  \begin{aligned}
  m^A_{u,n}
  &=\max_{0\leq d<D}\left|H^A_{u,n,d}\right|,\\
  \alpha^A_{u,n}
  &=\frac{m^A_{u,n}}{M_{\tau_A}},\\
  \widehat H^A_{u,n,d}
  &=\operatorname{cast}_{\tau_A}
    \!\left(\frac{H^A_{u,n,d}}{\alpha^A_{u,n}}\right).
  \end{aligned}
  $$

  Wherein `cast` indicates conversion to the target type; when the target type is INT8, the conversion includes rounding and saturation to the range $[-127,127]$. When PerTokenPerHead quantization is enabled, $\boldsymbol\alpha^Q$ of the Q branch is written to `qScaleOptional` and $\widehat{\boldsymbol H}^Q$ is written to `qOut`; $\boldsymbol\alpha^K$ of the K branch is written to `kScaleCacheRef` and $\widehat{\boldsymbol H}^K$ is written to `kCacheRef`. Branches without dynamic quantization produce only direct conversion results and no scale.

  M-RoPE MX performs cuBLAS MX FP8 quantization independently on each D32 block of $\boldsymbol Z^Q/\boldsymbol Z^K$. For a block $\boldsymbol x_b$ with finite inputs, compute in FP32:

  $$
  \begin{aligned}
  a_b&=\max_i|x_{b,i}|,\\
  e_b&=\max\left(-127,\left\lceil\log_2\frac{a_b}{448}\right\rceil\right),\\
  s_b&=2^{e_b},\\
  q_{b,i}&=\operatorname{cast}^{\mathrm{rint,sat}}_{\mathrm{FP8\ E4M3FN}}\left(\frac{x_{b,i}}{s_b}\right).
  \end{aligned}
  $$

  When $a_b=0$, $e_b$ is set to $-127$. $s_b$ is stored as FLOAT8_E8M0, and the raw encoding of a finite scale is $e_b+127$; if a block contains Inf or NaN, the raw encoding of the scale is `0xFF`. This definition corresponds to `scaleAlg=1` of `DynamicMxQuantV3`. The operator name and numeric properties are not the sole specification of the semantics of this API.

  Let $B=\lceil D/32\rceil$. Effective scales are stored along the last axis in the linear order of D32 blocks:

  $$
  \mathrm{scale}[...,b],
  \qquad 0\leq b<B.
  $$

  The current implementation still fixes $D=128$, so each Q/K head produces 4 FLOAT8_E8M0 scales; this does not mean that arbitrary D is already supported.

  **6. V Scaling, Quantization, and Cache Write Back**

  Broadcast `vScaleOptional` to the logical shape of V based on its input shape, denoted as $\boldsymbol\beta$. The V branch uniformly applies element-wise multiplication by this scaling factor before the FP8 E4M3FN conversion:

  $$
  \widehat V_{u,n,d}
  =
  \operatorname{cast}_{\mathrm{FP8\ E4M3FN}}
  \!\left(V_{u,n,d}\beta_{n,d}\right).
  $$

  For token $u$, let $z_u=\mathrm{slotMapping}_u$, where $B_{\mathrm{cache}}$ indicates the `BlockSize` of the Cache. Then the write-back block and the offset within the block are

  $$
  b_u=\left\lfloor\frac{z_u}{B_{\mathrm{cache}}}\right\rfloor,
  \qquad
  o_u=z_u\bmod B_{\mathrm{cache}}.
  $$

  Only the following positions are updated:

  $$
  \begin{aligned}
  \mathrm{kCacheRef}'_{b_u,n,o_u,d}
  &=\widehat H^K_{u,n,d},\\
  \mathrm{vCacheRef}'_{b_u,n,o_u,d}
  &=\widehat V_{u,n,d},\\
  \mathrm{kScaleCacheRef}'_{b_u,n,o_u,\ldots}
  &=\mathrm{kScale}_{u,n,\ldots}.
  \end{aligned}
  $$

  Wherein the primed tensors indicate the in-place update results after the API is executed; Cache positions not selected by `slotMapping` remain unchanged.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnQkvRmsNormRopeCacheWithKScaleGetWorkspaceSize" first to obtain the input parameters and calculate the required workspace size based on the computation flow, and then call "aclnnQkvRmsNormRopeCacheWithKScale" to execute the computation.

```cpp
aclnnStatus aclnnQkvRmsNormRopeCacheWithKScaleGetWorkspaceSize(
    const aclTensor   *qkv,
    const aclTensor   *qGamma,
    const aclTensor   *kGamma,
    const aclTensor   *cosSin,
    const aclTensor   *slotMapping,
    aclTensor         *kCacheRef,
    aclTensor         *vCacheRef,
    aclTensor         *kScaleCacheRef,
    const aclTensor   *queryStartLocOptional,
    const aclTensor   *seqLensOptional,
    const aclTensor   *rotationOptional,
    const aclTensor   *vScaleOptional,
    const aclTensor   *mropePositionOptional,
    const aclIntArray *headNums,
    const char        *layoutQkv,
    const char        *layoutQOut,
    float              epsilon,
    const aclIntArray *mropeSectionOptional,
    const char        *qQuantMode,
    const char        *kQuantMode,
    aclTensor         *qOut,
    aclTensor         *qScaleOptional,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor);
```

```cpp
aclnnStatus aclnnQkvRmsNormRopeCacheWithKScale(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream);
```

## aclnnQkvRmsNormRopeCacheWithKScaleGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1312px"><colgroup>
  <col style="width: 158px">
  <col style="width: 120px">
  <col style="width: 333px">
  <col style="width: 137px">
  <col style="width: 212px">
  <col style="width: 100px">
  <col style="width: 107px">
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="white-space: nowrap">qkv (const aclTensor*)</td>
      <td>Input</td>
      <td>Fused Q/K/V input, corresponding to <code>qkv</code> in the formula.</td>
      <td><ul><li>Null pointer or empty tensor is not supported.</li></ul></td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td><code>[T,Nq+Nk+Nv,D]</code> or <code>[Nq+Nk+Nv,T,D]</code></td>
      <td>×</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">qGamma (const aclTensor*)</td>
      <td>Input</td>
      <td>RMSNorm weight of the Q branch, corresponding to <code>gamma</code> in the Q branch formula.</td>
      <td><ul><li>Null pointers or empty tensors are not supported.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td><code>[D]</code></td>
      <td>×</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">kGamma (const aclTensor*)</td>
      <td>Input</td>
      <td>RMSNorm weight of the K branch, corresponding to <code>gamma</code> in the formula of the K branch.</td>
      <td><ul><li>Null pointers or empty tensors are not supported.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td><code>[D]</code></td>
      <td>×</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">cosSin(const aclTensor*)</td>
      <td>Input</td>
      <td>RoPE/M-RoPE position encoding table, corresponding to <code>cosSin</code>, <code>cos</code>, and <code>sin</code> in the formula.</td>
      <td><ul><li>Null pointers or empty tensors are not supported.</li><li>The first <code>D/2</code> columns are cos, and the last <code>D/2</code> columns are sin.</li><li>The first dimension must cover the RoPE positions accessed in this call.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td><code>[MaxSeqLen,D]</code></td>
      <td>×</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">slotMapping (const aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the slot index for each token to write to the cache, corresponding to <code>slotMapping</code> in the formula.</td>
      <td><ul><li>Null pointer or empty tensor is not supported.</li><li>The value range should be <code>[0,BlockNum*BlockSize-1]</code>.</li><li>If multiple tokens write to the same slot in a single call, the final write order and result are undefined.</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td><code>[T]</code></td>
      <td>×</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">kCacheRef (aclTensor*)</td>
      <td>Input/Output</td>
      <td>KCache write back tensor, corresponding to <code>kCacheRef</code> in the formula.</td>
      <td><ul><li>Null pointer or empty tensor is not supported.</li><li>In-place update is performed based on the input tensor.</li><li>For the data type and its correspondence with the K quantization result, see "Constraints".</li><li>Non-contiguous tensor is supported, subject to the stride limit in "Constraints".</li></ul></td>
      <td>FLOAT8_E4M3FN, INT8<br>RoPE/M-RoPE MX: FLOAT8_E4M3FN<br>M-RoPE: INT8</td>
      <td>ND</td>
      <td><code>[BlockNum,Nk,BlockSize,D]</code></td>
      <td>√</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">vCacheRef (aclTensor*)</td>
      <td>Input/Output</td>
      <td>VCache write back tensor, corresponding to <code>vCacheRef</code> in the formula.</td>
      <td><ul><li>Null pointers and empty tensors are not supported.</li><li>The API performs an in-place update based on the passed-in tensor.</li><li>Non-contiguous tensors are supported, provided that the stride limits in "Constraints" are met.</li></ul></td>
      <td>FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td><code>[BlockNum,Nv,BlockSize,D]</code></td>
      <td>√</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">kScaleCacheRef (aclTensor*)</td>
      <td>Input/Output</td>
      <td>Tensor for writing back the K dynamic quantization scale cache, corresponding to <code>kScaleCacheRef</code> in the formula.</td>
      <td><ul><li>Null pointers and empty tensors are not supported.</li><li>The API performs an in-place update based on the passed-in tensor.</li><li>Non-contiguous tensors are supported, provided that the stride limits in "Constraints" are met.</li></ul></td>
      <td>RoPE/M-RoPE: FLOAT<br>M-RoPE MX: FLOAT8_E8M0</td>
      <td>ND</td>
      <td>RoPE/M-RoPE: <code>[BlockNum,Nk,BlockSize,1]</code><br>M-RoPE MX: <code>[BlockNum,Nk,BlockSize,ceil(D/32)]</code></td>
      <td>√</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">queryStartLocOptional (const aclTensor*)</td>
      <td>Optional Input</td>
      <td>Prefix sum of the number of tokens in each batch within the current call, corresponding to <code>queryStartLoc</code> in the formula.</td>
      <td><ul><li>When passed, null pointer or empty tensor is not supported; for whether it is passed and its combination with other position inputs, see "Constraints".</li><li>The length must be greater than or equal to 2.</li><li><code>queryStartLocOptional[0]</code> must be 0, and <code>queryStartLocOptional[Batch]</code> must be <code>T</code>.</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td><code>[Batch+1]</code></td>
      <td>×</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">seqLensOptional (const aclTensor*)</td>
      <td>Optional Input</td>
      <td>Actual sequence length of each batch after the current token is appended, corresponding to <code>seqLens</code> in the formula.</td>
      <td><ul><li>When passed, null pointer or empty tensor is not supported. See "Constraints" for whether it is passed and its combination with other position inputs.</li><li>The length must be equal to <code>queryStartLocOptional.shape[0]-1</code>.</li><li>The number of current tokens and the actual sequence length of each batch must not exceed <code>MaxSeqLen</code>. For specific restrictions, see "Constraints".</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td><code>[Batch]</code></td>
      <td>×</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">rotationOptional (const aclTensor*)</td>
      <td>Optional Input</td>
      <td>Q/K shared matrix multiplication weights for RoPE/M-RoPE.</td>
      <td><ul><li>For RoPE/M-RoPE, a valid tensor must be passed.</li><li>For M-RoPE MX, a null pointer must be passed and matrix multiplication is not performed.</li></ul></td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td><code>[D,D]</code></td>
      <td>×</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">vScaleOptional (const aclTensor*)</td>
      <td>Optional Input</td>
      <td>Quantization scale factor for the V branch, corresponding to <code>vScaleOptional</code> in the formula.</td>
      <td><ul><li>Valid Tensors must be passed for all three scenarios.</li><li>M-RoPE MX performs static quantization on V per head and per channel.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>RoPE: <code>[Nv]</code><br>M-RoPE/M-RoPE MX: <code>[Nv,D]</code></td>
      <td>×</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">mropePositionOptional (const aclTensor*)</td>
      <td>Optional Input</td>
      <td>T/H/W position indices for the M-RoPE scenario. Each token occupies one row, and the three columns correspond to T, H, and W respectively.</td>
      <td><ul><li>Null pointers or empty Tensors are not supported when passed in; see "Constraints" for whether to pass it and its combination with other position inputs.</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td><code>[T,3]</code></td>
      <td>×</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">headNums (const aclIntArray*)</td>
      <td>Input</td>
      <td>Array of the numbers of Q/K/V heads, mapped in order to <code>Nq</code>, <code>Nk</code>, and <code>Nv</code> in the formula.</td>
      <td><ul><li>Null pointers are not supported.</li><li>Must contain 3 positive integers.</li><li>All three scenarios require <code>Nq&lt;=64</code>, <code>Nq=8*Nk</code>, and <code>Nk=Nv</code>.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>Length 3</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">layoutQkv (const char*)</td>
      <td>Input</td>
      <td>N/T axis layout identifier of <code>qkv</code>, corresponding to the axes of <code>T</code>, <code>Nq</code>, <code>Nk</code>, and <code>Nv</code> in the formula.</td>
      <td><ul><li>Default value is <code>"TND"</code>.</li><li>Passing a null pointer or null string applies the default value.</li><li>Case-sensitive. Only supports <code>"TND"</code> and <code>"NTD"</code>.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">layoutQOut (const char*)</td>
      <td>Input</td>
      <td>Layout identifier of the N/T axes of <code>qOut</code> and <code>qScaleOptional</code>.</td>
      <td><ul><li>Default value is <code>"NTD"</code>.</li><li>If a null pointer or empty string is passed, the default value is used.</li><li>Case-sensitive. Only <code>"TND"</code> and <code>"NTD"</code> are supported.</li><li>Currently, <code>layoutQkv="NTD"</code> and <code>layoutQOut="TND"</code> are not supported.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">epsilon (float)</td>
      <td>Input</td>
      <td>RMSNorm parameter to prevent division by zero, corresponding to <code>epsilon</code> in the formula.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">mropeSectionOptional (const aclIntArray*)</td>
      <td>Optional Input</td>
      <td>T/H/W lane capacity parameter for the M-RoPE scenario. The semantic order is <code>(T,H,W)</code>, passed as <code>[t,h,w]</code>.</td>
      <td><ul><li>For whether it is passed and its combination with <code>mropePositionOptional</code>, see "Constraints".</li><li>An empty array is equivalent to not passing it. If not empty, the length must be 3. For the value range and sum constraints, see "Constraints".</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>Length of 3 (<code>[t,h,w]</code>)</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">qQuantMode (const char*)</td>
      <td>Input</td>
      <td>Quantization mode of the Q branch.</td>
      <td><ul><li>When a null pointer or empty string is passed, the default value <code>"PerTokenPerHead"</code> is used.</li><li>RoPE uses <code>"PerTokenPerHead"</code>, M-RoPE uses <code>"NoQuant"</code>, and M-RoPE MX uses <code>"Mx"</code>.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">kQuantMode (const char*)</td>
      <td>Input</td>
      <td>Quantization algorithm mode of the K branch.</td>
      <td><ul><li>This parameter is directly received by <code>aclnnQkvRmsNormRopeCacheWithKScaleGetWorkspaceSize</code>.</li><li>When a null pointer or null string is passed, the default value <code>"PerTokenPerHead"</code> is applied.</li><li><code>"PerTokenPerHead"</code> selects PerTokenPerHead quantization, and <code>"Mx"</code> selects M-RoPE MX quantization.</li><li>This parameter does not select the output dtype; the storage type of K is still determined by the <code>kCacheRef</code> dtype and the scenario contract.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">qOut (aclTensor*)</td>
      <td>Output</td>
      <td>Q branch output. FP8 E4M3FN for RoPE and M-RoPE MX; BF16 for M-RoPE.</td>
      <td><ul><li>Null pointer or empty tensor is not supported.</li><li>The API directly reads the data type of <code>qOut</code>; the dtype attribute is not passed in separately.</li><li>See "Constraints" for the output types and their combinations with quantization modes and position inputs.</li></ul></td>
      <td>FLOAT8_E4M3FN, BFLOAT16</td>
      <td>ND</td>
      <td>RoPE: <code>[T,Nq,D]</code> or <code>[Nq,T,D]</code><br>M-RoPE/M-RoPE MX: <code>[T,Nq,D]</code></td>
      <td>×</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">qScaleOptional(aclTensor*)</td>
      <td>Optional Output</td>
      <td>Dynamic quantization scale of the Q branch. One FP32 scale per token/head for RoPE; one FLOAT8_E8M0 scale per D32 block for M-RoPE MX; M-RoPE does not produce a public scale.</td>
      <td><ul><li>RoPE and M-RoPE MX require a valid tensor.</li><li>M-RoPE requires a null pointer.</li></ul></td>
      <td>FLOAT, FLOAT8_E8M0</td>
      <td>ND</td>
      <td>RoPE: <code>[T,Nq]</code> or <code>[Nq,T]</code><br>M-RoPE MX: <code>[T,Nq,ceil(D/32)]</code><br>M-RoPE: None</td>
      <td>×</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be allocated on the Device.</td>
      <td>Null pointers are not supported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation flow.</td>
      <td>Null pointer not supported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API reports an error in the following use cases:

  <table style="undefined;table-layout: fixed; width: 1134px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
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
      <td style="white-space: nowrap">ACLNN_ERR_PARAM_NULLPTR</td>
      <td style="white-space: nowrap">161001</td>
      <td><code>qkv</code>, <code>qGamma</code>, <code>kGamma</code>, <code>cosSin</code>, <code>slotMapping</code>, <code>kCacheRef</code>, <code>vCacheRef</code>, <code>kScaleCacheRef</code>, <code>vScaleOptional</code>, <code>headNums</code>, <code>qOut</code>, <code>workspaceSize</code>, or <code>executor</code> is a null pointer, or <code>queryStartLocOptional</code>, <code>seqLensOptional</code>, <code>rotationOptional</code>, or <code>qScaleOptional</code>, which must be non-null in supported use cases, is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="6" style="white-space: nowrap">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6" style="white-space: nowrap">161002</td>
      <td>Tensor is empty.</td>
    </tr>
    <tr>
      <td>The tensor data type is not in the supported range.</td>
    </tr>
    <tr>
      <td>The tensor data format is a private format or does not meet the ND format requirements.</td>
    </tr>
    <tr>
      <td>Tensor shape, <code>headNums</code>, <code>layoutQkv</code>, or <code>layoutQOut</code> does not meet the API constraints.</td>
    </tr>
    <tr>
      <td>The stride of the non-contiguous tensors <code>kCacheRef</code>, <code>vCacheRef</code>, or <code>kScaleCacheRef</code> does not meet the constraints.</td>
    </tr>
    <tr>
      <td><code>mropePositionOptional</code> and non-empty <code>mropeSectionOptional</code> are not both absent or both present, the length or per-item value range of <code>mropeSectionOptional</code> does not meet the constraints, or the combination of position parameters, <code>rotationOptional</code>, <code>qQuantMode</code>, <code>kQuantMode</code>, output/cache dtype, and scale shape does not match the selected scenario.</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
      <td style="white-space: nowrap">561101</td>
      <td>Failed to create <code>aclOpExecutor</code>.</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">ACLNN_ERR_INNER_NULLPTR</td>
      <td style="white-space: nowrap">561103</td>
      <td>Input Contiguous processing, Cache view creation, operator task graph construction, or graph construction output check failed.</td>
    </tr>
  </tbody></table>

## aclnnQkvRmsNormRopeCacheWithKScale

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 200px">
  <col style="width: 130px">
  <col style="width: 770px">
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
      <td>Workspace memory address allocated on the device. If <code>workspaceSize</code> returned by the first-phase API is 0, <code>nullptr</code> can be passed.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the device, obtained by calling the first-phase API <code>aclnnQkvRmsNormRopeCacheWithKScaleGetWorkspaceSize</code>.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator computing process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream used to execute tasks.</td>
    </tr>
  </tbody></table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Supported scenarios:

  | Scenario | Position Input | rotation | Q Branch | K Branch | V Branch | Supported N/T Layouts |
  |:---|:---|:---|:---|:---|:---|:---|
  | RoPE | Pass `queryStartLocOptional/seqLensOptional`; do not pass `mropePositionOptional`; `mropeSectionOptional` is not passed or is empty. | Must pass BF16 `[D,D]`. | `qQuantMode="PerTokenPerHead"`; `qOut` is FP8 E4M3FN; `qScaleOptional` is FP32 rank2. | `kQuantMode="PerTokenPerHead"`; K is FP8 E4M3FN; `kScaleCacheRef` is FP32 `[BlockNum,Nk,BlockSize,1]`. | `vScaleOptional=[Nv]`, output is FP8 E4M3FN. | `NTD -> NTD`, `TND -> TND`, `TND -> NTD` |
  | M-RoPE | Do not pass `queryStartLocOptional/seqLensOptional`; pass `mropePositionOptional[T,3]` and `mropeSectionOptional=[t,h,w]`. | Must pass BF16 `[D,D]`. | `qQuantMode="NoQuant"`; `qOut` is BF16; `qScaleOptional=nullptr`. | `kQuantMode="PerTokenPerHead"`; K is INT8; `kScaleCacheRef` is FP32 `[BlockNum,Nk,BlockSize,1]`. | `vScaleOptional=[Nv,D]`, output is FP8 E4M3FN. | Only `TND -> TND` |
  | M-RoPE MX | Same as M-RoPE. | Must be `nullptr`. | `qQuantMode="Mx"`; `qOut` is FP8 E4M3FN `[T,Nq,D]`; `qScaleOptional` is FLOAT8_E8M0 `[T,Nq,ceil(D/32)]`. | `kQuantMode="Mx"`; K is FP8 E4M3FN; `kScaleCacheRef` is FLOAT8_E8M0 `[BlockNum,Nk,BlockSize,ceil(D/32)]`. | `vScaleOptional=[Nv,D]`, per-head-per-channel static quantization to FP8 E4M3FN. | Only `TND -> TND` |

  This API determines the three scenarios based on the position input and Q/K quantization modes. An empty `mropeSectionOptional` is equivalent to not passing it. Each group of position inputs must be present or absent as a whole; the two groups cannot both be present or both be absent. Other cross combinations are not supported. The data type of `qOut` is determined by the tensor constructed by the caller, and the storage type of K is determined by the `kCacheRef` dtype. `kQuantMode` only distinguishes between the PerTokenPerHead and Mx algorithms for K, and is not a dtype selector.

  All three scenarios require `D=128`, `0<Nq<=64`, `Nq=8*Nk`, and `Nk=Nv`. The three Cache Tensors are input-output aliases. Only the slots specified by `slotMapping` are updated, and other positions retain their original values.

- Deterministic computation: aclnnQkvRmsNormRopeCacheWithKScale defaults to a deterministic implementation.
- Input shape constraints:
  - Only `D=128` is supported. All three scenarios require `1<=T<=262144`.
  - `headNums=[Nq,Nk,Nv]` must satisfy `0<Nq<=64`, `Nq=8*Nk`, and `Nk=Nv`.
  - The shape of `vScaleOptional` must be `[Nv]` in the RoPE scenario and `[Nv,D]` in the M-RoPE and M-RoPE MX scenarios.
  - `layoutQkv` controls the N/T layout of `qkv`, and the default value is `"TND"`; `layoutQOut` controls the N/T layout of `qOut` and `qScaleOptional`, and the default value is `"NTD"`:
    - RoPE with `layoutQkv="TND"` and `layoutQOut="TND"`: `qkv=[T,Nq+Nk+Nv,D]`, `qOut=[T,Nq,D]`, `qScaleOptional=[T,Nq]`.
    - RoPE with `layoutQkv="TND"` and `layoutQOut="NTD"`: `qkv=[T,Nq+Nk+Nv,D]`, `qOut=[Nq,T,D]`, `qScaleOptional=[Nq,T]`.
    - RoPE with `layoutQkv="NTD"` and `layoutQOut="NTD"`: `qkv=[Nq+Nk+Nv,T,D]`, `qOut=[Nq,T,D]`, `qScaleOptional=[Nq,T]`.
    - M-RoPE and M-RoPE MX only support `TND -> TND`. For M-RoPE MX, `qScaleOptional` is fixed to `[T,Nq,ceil(D/32)]`.
  - The `BlockNum` and `BlockSize` of `kCacheRef`, `vCacheRef`, and `kScaleCacheRef` must be the same.
  - `kCacheRef` and `vCacheRef` are both 4-dimensional with positive strides. The last-dimension stride is 1, and the strides of the head and token dimensions are no less than `D=128`. The first three dimensions of `kCacheRef` and `vCacheRef` must have the same strides. `kScaleCacheRef` for RoPE/M-RoPE is 4-dimensional with positive strides and a last-dimension stride of 1. For M-RoPE MX, `kScaleCacheRef` is likewise 4-dimensional, with the last axis of `ceil(D/32)` contiguous and a stride of 1.
  - The logical shape of `mropePositionOptional` for M-RoPE is fixed to `[T,3]`, where the position indexes `P[u,0]`, `P[u,1]`, and `P[u,2]` correspond to the T/H/W coordinates of token `u` respectively.
- Input value range restrictions:
  - In the RoPE scenario, `queryStartLocOptional` must be monotonically non-decreasing. Let `L_b=queryStartLocOptional[b+1]-queryStartLocOptional[b]` be the number of tokens of the `b`-th batch in this call. Then `0 <= L_b <= MaxSeqLen` and `L_b <= seqLensOptional[b] <= MaxSeqLen` must hold, where `MaxSeqLen=cosSin.shape[0]`. If the actual sequence length of a non-empty batch exceeds the table capacity, the position encoding accesses positions outside the table, and the behavior is undefined.
  - `slotMapping` must be a one-dimensional INT32 Tensor with shape `[T]`, and each element must be within `[0,BlockNum*BlockSize-1]`. For RoPE and M-RoPE, if duplicate slots exist, the final write order and results are undefined. For the slot uniqueness requirement of M-RoPE MX, see below.
  - In the two M-RoPE scenarios, each position index of `mropePositionOptional` must satisfy `0 <= value < MaxSeqLen`.
  - All three scenarios require `1<=T<=262144`; M-RoPE MX additionally requires that the slots in `slotMapping` be unique within this call.
  - In the M-RoPE scenario, `mropeSectionOptional=[t,h,w]` must have a length of 3. Let $\boldsymbol{s}=(s_{\mathrm T},s_{\mathrm H},s_{\mathrm W})$, where $i=0,1,2$ corresponds to T/H/W respectively. Then

    $$
    s_{\mathrm T}\geq 0,\qquad
    0\leq s_i\leq C_i=\left\lfloor\frac{D/2+2-i}{3}\right\rfloor\ (i\in\{1,2\}),
    \qquad s_{\mathrm T}+s_{\mathrm H}+s_{\mathrm W}\leq D/2.
    $$

    For the current `D=128`, the upper limits of H/W are both 21; $s_{\mathrm T}$ has no independent lane limit, and the sum of the three terms must not exceed 64.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT products</term>:

  For the M-RoPE MX call, create tensors with the dtype/shape listed in the parameter table: `qOut` is `ACL_FLOAT8_E4M3FN`; `qScaleOptional` is `ACL_FLOAT8_E8M0` with shape `[T,Nq,ceil(D/32)]`; `kCacheRef` is `ACL_FLOAT8_E4M3FN`; `kScaleCacheRef` is `ACL_FLOAT8_E8M0` with shape `[BlockNum,Nk,BlockSize,ceil(D/32)]`; `vScaleOptional` is `ACL_FLOAT` with shape `[Nv,D]`. The core call is as follows (pass `nullptr` to `queryStartLocOptional`, `seqLensOptional`, and `rotationOptional`):

  ```c++
  aclnnStatus status = aclnnQkvRmsNormRopeCacheWithKScaleGetWorkspaceSize(
      qkv, qGamma, kGamma, cosSin, slotMapping, kCacheRef, vCacheRef,
      kScaleCacheRef, nullptr, nullptr, nullptr, vScaleOptional, mropePosition,
      headNums, "TND", "TND", 1e-6f, mropeSection,
      "Mx", "Mx", qOut, qScaleOptional, &workspaceSize, &executor);
  ```

  `mropePosition` is `ACL_INT32 [T,3]`, and `mropeSection` is an `aclIntArray` of length 3, such as `{22,12,10}`. Then call the execution API as required by the two-phase API.

  ```c++
  #include <cstdint>
  #include <cstdio>
  #include <cstring>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_qkv_rms_norm_rope_cache_with_k_scale.h"

  #define CHECK_RET(cond, return_expr) \
      do {                             \
          if (!(cond)) {               \
              return_expr;             \
          }                            \
      } while (0)

  #define LOG_PRINT(message, ...)         \
      do {                                \
          printf(message, ##__VA_ARGS__); \
      } while (0)

  struct TensorResource {
      aclTensor *tensor = nullptr;
      void *deviceAddr = nullptr;
  };

  struct AclResource {
      int32_t deviceId = 0;
      aclrtStream stream = nullptr;
      bool aclInited = false;
      bool deviceSet = false;
      std::vector<TensorResource *> tensors;
      aclIntArray *headNums = nullptr;
      aclOpExecutor *executor = nullptr;
      void *workspaceAddr = nullptr;
  };

  int64_t GetShapeSize(const std::vector<int64_t> &shape)
  {
      int64_t shapeSize = 1;
      for (auto dim : shape) {
          shapeSize *= dim;
      }
      return shapeSize;
  }

  std::vector<int64_t> GetContiguousStrides(const std::vector<int64_t> &shape)
  {
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }
      return strides;
  }

  uint16_t FloatToBf16(float value)
  {
      uint32_t bits = 0;
      std::memcpy(&bits, &value, sizeof(bits));
      return static_cast<uint16_t>(bits >> 16);
  }

  int Init(int32_t deviceId, AclResource &resource)
  {
      auto ret = aclInit(nullptr);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
      resource.aclInited = true;

      ret = aclrtSetDevice(deviceId);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
      resource.deviceId = deviceId;
      resource.deviceSet = true;

      ret = aclrtCreateStream(&resource.stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
      return ACL_SUCCESS;
  }

  void FreeResource(AclResource &resource)
  {
      for (auto *tensorResource : resource.tensors) {
          if (tensorResource != nullptr && tensorResource->tensor != nullptr) {
              aclDestroyTensor(tensorResource->tensor);
              tensorResource->tensor = nullptr;
          }
      }
      if (resource.headNums != nullptr) {
          aclDestroyIntArray(resource.headNums);
          resource.headNums = nullptr;
      }
      for (auto *tensorResource : resource.tensors) {
          if (tensorResource != nullptr && tensorResource->deviceAddr != nullptr) {
              aclrtFree(tensorResource->deviceAddr);
              tensorResource->deviceAddr = nullptr;
          }
      }
      if (resource.workspaceAddr != nullptr) {
          aclrtFree(resource.workspaceAddr);
          resource.workspaceAddr = nullptr;
      }
      if (resource.stream != nullptr) {
          aclrtDestroyStream(resource.stream);
          resource.stream = nullptr;
      }
      if (resource.deviceSet) {
          aclrtResetDevice(resource.deviceId);
          resource.deviceSet = false;
      }
      if (resource.aclInited) {
          aclFinalize();
          resource.aclInited = false;
      }
  }

  int ReturnAfterCleanup(int ret, AclResource &resource)
  {
      FreeResource(resource);
      return ret;
  }

  template <typename T>
  int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, aclDataType dataType,
                      TensorResource &resource)
  {
      const size_t size = static_cast<size_t>(GetShapeSize(shape)) * sizeof(T);
      auto ret = aclrtMalloc(&resource.deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

      ret = aclrtMemcpy(resource.deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      std::vector<int64_t> strides = GetContiguousStrides(shape);
      resource.tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                                        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), resource.deviceAddr);
      CHECK_RET(resource.tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_INVALID_PARAM);
      return ACL_SUCCESS;
  }

  int main()
  {
      AclResource resource;
      auto ret = Init(0, resource);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
                return ReturnAfterCleanup(ret, resource));

      constexpr int64_t T = 4;
      constexpr int64_t Nq = 16;
      constexpr int64_t Nk = 2;
      constexpr int64_t Nv = 2;
      constexpr int64_t D = 128;
      constexpr int64_t Batch = 1;
      constexpr int64_t MaxSeqLen = 16;
      constexpr int64_t BlockNum = 1;
      constexpr int64_t BlockSize = 16;

      std::vector<int64_t> qkvShape = {T, Nq + Nk + Nv, D};
      std::vector<int64_t> qGammaShape = {D};
      std::vector<int64_t> kGammaShape = {D};
      std::vector<int64_t> cosSinShape = {MaxSeqLen, D};
      std::vector<int64_t> slotMappingShape = {T};
      std::vector<int64_t> kCacheShape = {BlockNum, Nk, BlockSize, D};
      std::vector<int64_t> vCacheShape = {BlockNum, Nv, BlockSize, D};
      std::vector<int64_t> kScaleCacheShape = {BlockNum, Nk, BlockSize, 1};
      std::vector<int64_t> queryStartLocShape = {Batch + 1};
      std::vector<int64_t> seqLensShape = {Batch};
      std::vector<int64_t> rotationShape = {D, D};
      std::vector<int64_t> vScaleShape = {Nv};
      std::vector<int64_t> qOutShape = {T, Nq, D};
      std::vector<int64_t> qScaleShape = {T, Nq};

      std::vector<uint16_t> qkvHostData(GetShapeSize(qkvShape), FloatToBf16(0.125f));
      std::vector<float> qGammaHostData(GetShapeSize(qGammaShape), 1.0f);
      std::vector<float> kGammaHostData(GetShapeSize(kGammaShape), 1.0f);
      std::vector<float> cosSinHostData(GetShapeSize(cosSinShape), 0.0f);
      for (int64_t row = 0; row < MaxSeqLen; ++row) {
          for (int64_t col = 0; col < D / 2; ++col) {
              cosSinHostData[row * D + col] = 1.0f;
          }
      }
      std::vector<int32_t> slotMappingHostData = {0, 1, 2, 3};
      std::vector<uint8_t> kCacheHostData(GetShapeSize(kCacheShape), 0);
      std::vector<uint8_t> vCacheHostData(GetShapeSize(vCacheShape), 0);
      std::vector<float> kScaleCacheHostData(GetShapeSize(kScaleCacheShape), 0.0f);
      std::vector<int32_t> queryStartLocHostData = {0, T};
      std::vector<int32_t> seqLensHostData = {T};
      std::vector<uint16_t> rotationHostData(GetShapeSize(rotationShape), FloatToBf16(0.0f));
      for (int64_t i = 0; i < D; ++i) {
          rotationHostData[i * D + i] = FloatToBf16(1.0f);
      }
      std::vector<float> vScaleHostData(GetShapeSize(vScaleShape), 1.0f);
      std::vector<uint8_t> qOutHostData(GetShapeSize(qOutShape), 0);
      std::vector<float> qScaleHostData(GetShapeSize(qScaleShape), 0.0f);

      TensorResource qkv;
      TensorResource qGamma;
      TensorResource kGamma;
      TensorResource cosSin;
      TensorResource slotMapping;
      TensorResource kCache;
      TensorResource vCache;
      TensorResource kScaleCache;
      TensorResource queryStartLoc;
      TensorResource seqLens;
      TensorResource rotation;
      TensorResource vScale;
      TensorResource qOut;
      TensorResource qScale;
      resource.tensors = {&qkv,         &qGamma,       &kGamma, &cosSin, &slotMapping, &kCache, &vCache,
                          &kScaleCache, &queryStartLoc, &seqLens, &rotation, &vScale,  &qOut,   &qScale};

      ret = CreateAclTensor(qkvHostData, qkvShape, ACL_BF16, qkv);
      CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
      ret = CreateAclTensor(qGammaHostData, qGammaShape, ACL_FLOAT, qGamma);
      CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
      ret = CreateAclTensor(kGammaHostData, kGammaShape, ACL_FLOAT, kGamma);
      CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
      ret = CreateAclTensor(cosSinHostData, cosSinShape, ACL_FLOAT, cosSin);
      CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
      ret = CreateAclTensor(slotMappingHostData, slotMappingShape, ACL_INT32, slotMapping);
      CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
      ret = CreateAclTensor(kCacheHostData, kCacheShape, ACL_FLOAT8_E4M3FN, kCache);
      CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
      ret = CreateAclTensor(vCacheHostData, vCacheShape, ACL_FLOAT8_E4M3FN, vCache);
      CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
      ret = CreateAclTensor(kScaleCacheHostData, kScaleCacheShape, ACL_FLOAT, kScaleCache);
      CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
      ret = CreateAclTensor(queryStartLocHostData, queryStartLocShape, ACL_INT32, queryStartLoc);
      CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
      ret = CreateAclTensor(seqLensHostData, seqLensShape, ACL_INT32, seqLens);
      CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
      ret = CreateAclTensor(rotationHostData, rotationShape, ACL_BF16, rotation);
      CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
      ret = CreateAclTensor(vScaleHostData, vScaleShape, ACL_FLOAT, vScale);
      CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
      ret = CreateAclTensor(qOutHostData, qOutShape, ACL_FLOAT8_E4M3FN, qOut);
      CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));
      ret = CreateAclTensor(qScaleHostData, qScaleShape, ACL_FLOAT, qScale);
      CHECK_RET(ret == ACL_SUCCESS, return ReturnAfterCleanup(ret, resource));

      std::vector<int64_t> headNumsVec = {Nq, Nk, Nv};
      resource.headNums = aclCreateIntArray(headNumsVec.data(), headNumsVec.size());
      CHECK_RET(resource.headNums != nullptr,
                LOG_PRINT("aclCreateIntArray failed.\n"); return ReturnAfterCleanup(ACL_ERROR_INVALID_PARAM, resource));

      const char *layoutQkv = "TND";
      const char *layoutQOut = "TND";
      float epsilon = 1e-6f;
      uint64_t workspaceSize = 0;
      aclnnStatus status = aclnnQkvRmsNormRopeCacheWithKScaleGetWorkspaceSize(
          qkv.tensor, qGamma.tensor, kGamma.tensor, cosSin.tensor, slotMapping.tensor, kCache.tensor, vCache.tensor,
          kScaleCache.tensor, queryStartLoc.tensor, seqLens.tensor, rotation.tensor, vScale.tensor, nullptr,
          resource.headNums, layoutQkv, layoutQOut, epsilon, nullptr, "PerTokenPerHead", "PerTokenPerHead",
          qOut.tensor, qScale.tensor, &workspaceSize, &resource.executor);
      CHECK_RET(status == ACL_SUCCESS,
                LOG_PRINT("aclnnQkvRmsNormRopeCacheWithKScaleGetWorkspaceSize failed. ERROR: %d\n", status);
                return ReturnAfterCleanup(static_cast<int>(status), resource));

      if (workspaceSize > 0) {
          ret = aclrtMalloc(&resource.workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS,
                    LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                    return ReturnAfterCleanup(ret, resource));
      }

      status = aclnnQkvRmsNormRopeCacheWithKScale(resource.workspaceAddr, workspaceSize, resource.executor,
                                                  resource.stream);
      CHECK_RET(status == ACL_SUCCESS, LOG_PRINT("aclnnQkvRmsNormRopeCacheWithKScale failed. ERROR: %d\n", status);
                return ReturnAfterCleanup(static_cast<int>(status), resource));

      ret = aclrtSynchronizeStream(resource.stream);
      CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
                return ReturnAfterCleanup(ret, resource));
      LOG_PRINT("aclnnQkvRmsNormRopeCacheWithKScale execute success.\n");
      FreeResource(resource);
      return 0;
  }
  ```

<!-- end id7 -->
