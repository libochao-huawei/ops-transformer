# qkv_rms_norm_rope_cache_with_k_scale

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:59:57.530Z -->

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

- **Description**:

  This document describes two torch_extension APIs, `qkv_rms_norm_rope_cache_with_k_scale_` and `qkv_rms_norm_rope_cache_with_k_scale`. Both encapsulate `aclnnQkvRmsNormRopeCacheWithKScale` for Q/K/V preprocessing and PagedAttention KV Cache update in LLM inference use cases. The APIs support the following three complete use cases:
  - RoPE: Generates standard RoPE positions using `query_start_loc/seq_lens`; Q/K perform `RMSNorm -> RoPE -> rotation -> PerTokenPerHead quantization`, with data in FP8 E4M3FN and scale in FP32.
  - M-RoPE: Generates interleaved T/H/W positions using `mrope_position/mrope_section`; Q/K perform `RMSNorm -> M-RoPE -> rotation`. Q returns BF16 with `q_scale=None`, and K performs PerTokenPerHead INT8 quantization.
  - M-RoPE MX: Q/K perform `RMSNorm -> M-RoPE -> Dynamic MX Quant`, without performing `rotation`; each D32 block is quantized to FP8 E4M3FN using the power-of-two FLOAT8_E8M0 scale of cuBLAS MX FP8. V is statically quantized per-head-per-channel by `v_scale[Nv,D]` and written to the FP8 Cache.
  - `qkv_rms_norm_rope_cache_with_k_scale_`: updates the caller-provided `k_cache`, `v_cache`, and `k_scale_cache` in place, and returns `q_out` and `q_scale`.
  - `qkv_rms_norm_rope_cache_with_k_scale`: copies the three cache tensors internally, performs the update on the copies, and returns `q_out`, `q_scale`, and the three updated cache tensors; the cache tensors passed in by the caller remain unchanged.

- **Formula**:

  Split Q, K, and V from `qkv` by `head_nums=[Nq,Nk,Nv]`:

  $$
  q, k, v = split(qkv, [Nq, Nk, Nv])
  $$

  The Q and K branches apply RMSNorm using `q_gamma` and `k_gamma` respectively:

  $$
  y = \frac{x}{\sqrt{mean(x^2) + epsilon}} * gamma
  $$

  The Q/K branches perform RoPE, while the V branch does not:

  $$
  y_{rope} = concat(y_{low} * cos - y_{high} * sin,\ y_{high} * cos + y_{low} * sin)
  $$

  In RoPE and M-RoPE, the Q/K branches share the `rotation` matrix:

  $$
  q_{rot} = q_{rope} @ rotation,\quad k_{rot} = k_{rope} @ rotation
  $$

  M-RoPE MX requires `rotation=None`. The M-RoPE result directly enters MX quantization.

  For each D32 block $x_b$ of finite input, take $a_b=\max_i|x_{b,i}|$ in FP32, and set
  $e_b=\max(-127,\lceil\log_2(a_b/448)\rceil)$ and $s_b=2^{e_b}$, then compute
  $q_{b,i}=\operatorname{cast}^{\mathrm{rint,sat}}_{\mathrm{FP8\ E4M3FN}}(x_{b,i}/s_b)$.
  $s_b$ is stored as FLOAT8_E8M0. The raw encoding of a finite scale is $e_b+127$. When $a_b=0$, $e_b=-127$ is used. If the block contains Inf or NaN, the raw encoding of the scale is `0xFF`. This rule follows the cuBLAS MX FP8 scale semantics (corresponding to `scaleAlg=1` of `DynamicMxQuantV3`).

- Position in the RoPE use case:

  The RoPE position of the `i`-th token in the `b`-th batch is determined by `query_start_loc` and `seq_lens`:

  $$
  position = seq\_lens[b] - (query\_start\_loc[b + 1] - query\_start\_loc[b]) + i
  $$

  `cos_sin[..., :D/2]` is cos, and `cos_sin[..., D/2:]` is sin.

- cos/sin interleaving in the M-RoPE use case:

  Let $C=\mathrm{cos\_sin}$, $P=\mathrm{mrope\_position}$, and $D_{\mathrm{half}}=D/2$, and define
  $\iota(\mathrm T)=0,\ \iota(\mathrm H)=1,\ \iota(\mathrm W)=2$.
  For token $u$, axis $a\in\{\mathrm T,\mathrm H,\mathrm W\}$, and column $d$, the three raw positional encodings are defined as follows.
  The logical shape of `mrope_position` is $[T,3]$. Each row corresponds to one token, and the three columns correspond to the T, H, and W axes, respectively.

  $$
  R_{a,u,d}=C_{P_{u,\iota(a)},d},\qquad 0\leq u<T,\quad 0\leq d<D.
  $$

  Therefore, the three columns of `mrope_position` provide the T, H, and W position paths respectively, with each row corresponding to one token, rather than concatenating the three outputs first.
  Denote `mrope_section=[t,h,w]` as
  $\boldsymbol{s}=(s_{\mathrm T},s_{\mathrm H},s_{\mathrm W})=(t,h,w)$;
  where $s_{\mathrm T}$ does not participate in lane source selection and is not subject to an independent lane capacity upper limit check; it must still be non-negative and participates in the check on the sum of the three terms. For
  $0\leq\ell<D_{\mathrm{half}}$, define

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

  The valid cos/sin values are defined column-wise as

  $$
  \widehat c_{u,\ell}=R_{\sigma(\ell),u,\ell},\qquad
  \widehat s_{u,\ell}=R_{\sigma(\ell),u,D_{\mathrm{half}}+\ell}.
  $$

  where lanes not covered by H/W fall back to the T path in the same column; substitute
  $\widehat c$ and $\widehat s$ into the preceding RoPE rotation formula. The output dimension remains
  $D$ and is not expanded to $3D$.

- Q/K outputs of RoPE and M-RoPE with PerTokenPerHead dynamic quantization:

  Let $X^Q=q_{rot}$, $X^K=k_{rot}$, and the target type be
  $\tau_A\in\{\mathrm{FP8\ E4M3FN},\mathrm{INT8}\}$. The two target types differ only in the positive quantization upper limit:

  $$
  M_{\tau_A}=
  \begin{cases}
  448, & \tau_A=\mathrm{FP8\ E4M3FN},\\
  127, & \tau_A=\mathrm{INT8}.
  \end{cases}
  $$

  For $A\in\{Q,K\}$, token $u$, and head $n$, when dynamic quantization is enabled, the following unified computation is performed:

  $$
  \begin{aligned}
  m^A_{u,n}&=\max_{0\leq d<D}\left|X^A_{u,n,d}\right|,\\
  \alpha^A_{u,n}&=\frac{m^A_{u,n}}{M_{\tau_A}},\\
  \widehat X^A_{u,n,d}
  &=\operatorname{cast}_{\tau_A}
    \!\left(\frac{X^A_{u,n,d}}{\alpha^A_{u,n}}\right).
  \end{aligned}
  $$

  `cast` performs rounding and saturation according to the target data type. The INT8 target range is $[-127,127]$. When PerTokenPerHead dynamic quantization is not enabled, the output is

  $$
  O^A_{u,n,d}=\operatorname{cast}_{\tau_A}\!\left(X^A_{u,n,d}\right),
  $$

  and no scale is generated. The target data types and quantization switches for each use case are listed in "Constraints".

- M-RoPE MX quantization:

  For Q/K after M-RoPE, the preceding cuBLAS MX FP8 power-of-two scale quantization is performed independently on each D32 block, generating FP8 E4M3FN data and FLOAT8_E8M0 scale. Valid scales are linearly stored along the last axis of length `ceil(D/32)` per D32 block. The current implementation still fixes `D=128`; therefore, each head produces 4 scales. The symbol D in the formula does not mean that any D is supported.

  The V branch is uniformly multiplied by `v_scale` before FP8 E4M3FN conversion; `v_scale` is broadcast from its input shape to the logical shape of V:

  $$
  v_{fp8} = cast_{FP8\ E4M3FN}(v * v\_scale)
  $$

  The Cache write-back position is determined by `slot_mapping`:

  $$
  blockId = slot\_mapping[t] / BlockSize,\quad blockOffset = slot\_mapping[t]\ \%\ BlockSize
  $$

## Function Prototype

`qkv_rms_norm_rope_cache_with_k_scale_` is the in-place cache update API; `qkv_rms_norm_rope_cache_with_k_scale` is the functional variant. The two APIs have identical input parameters.

```python
cann_ops_transformer.qkv_rms_norm_rope_cache_with_k_scale_(
    qkv,
    q_gamma,
    k_gamma,
    cos_sin,
    slot_mapping,
    k_cache,
    v_cache,
    k_scale_cache,
    query_start_loc,
    seq_lens,
    head_nums,
    *,
    rotation=None,
    v_scale=None,
    layout_qkv="TND",
    layout_q_out="NTD",
    epsilon=0.000001,
    mrope_position=None,
    mrope_section=None,
    q_quant_mode="PerTokenPerHead",
    k_quant_mode="PerTokenPerHead",
    q_out_dtype=torch.float8_e4m3fn,
) -> (Tensor, Optional[Tensor])
```

```python
cann_ops_transformer.qkv_rms_norm_rope_cache_with_k_scale(
    qkv,
    q_gamma,
    k_gamma,
    cos_sin,
    slot_mapping,
    k_cache,
    v_cache,
    k_scale_cache,
    query_start_loc,
    seq_lens,
    head_nums,
    *,
    rotation=None,
    v_scale=None,
    layout_qkv="TND",
    layout_q_out="NTD",
    epsilon=0.000001,
    mrope_position=None,
    mrope_section=None,
    q_quant_mode="PerTokenPerHead",
    k_quant_mode="PerTokenPerHead",
    q_out_dtype=torch.float8_e4m3fn,
) -> (Tensor, Optional[Tensor], Tensor, Tensor, Tensor)
```

`query_start_loc` and `seq_lens` are Optional positional parameters that cannot be omitted: pass a valid tensor for RoPE, and explicitly pass `None` for both M-RoPE use cases. `k_quant_mode` is a keyword-only parameter and defaults to `"PerTokenPerHead"`. The underlying ACLNN C function of the same name includes this parameter. OPP, ACLNN callers, and the `cann_ops_transformer` extension must be built and installed as a set with consistent versions. Do not mix binaries with different ABIs.

## Mandatory

The two APIs below have identical input parameters.

| Name | Type | Type | Description | Data Type | Dimension (Shape) |
|---|---|---|---|---|---|
| qkv | Tensor | Required | Fused Q/K/V input. When `layout_qkv="TND"`, the shape is `[T,Nq+Nk+Nv,D]`; when `layout_qkv="NTD"`, the shape is `[Nq+Nk+Nv,T,D]`. | torch.bfloat16 | 3 |
| q_gamma | Tensor | Required | RMSNorm weight for the Q branch. | torch.float32 | `[D]` |
| k_gamma | Tensor | Required | RMSNorm weight for the K branch. | torch.float32 | `[D]` |
| cos_sin | Tensor | Required | RoPE positional encoding table. The first `D/2` columns are cos, and the last `D/2` columns are sin. The first dimension must cover all positions accessed in the current call. | torch.float32 | `[MaxSeqLen,D]` |
| slot_mapping | Tensor | Required | Index of the cache slot to which each token is written. Each element must be in `[0,BlockNum*BlockSize-1]`. | torch.int32 | `[T]` |
| k_cache | Tensor | Required | K Cache. The in-place interface updates this tensor directly; the functional interface updates a copy and returns it. FP8 E4M3FN for RoPE and M-RoPE MX; INT8 for M-RoPE. | torch.float8_e4m3fn or torch.int8 | `[BlockNum,Nk,BlockSize,D]` |
| v_cache | Tensor | Required | V Cache. The in-place interface updates this tensor directly; the functional interface updates a copy and returns it. | torch.float8_e4m3fn | `[BlockNum,Nv,BlockSize,D]` |
| k_scale_cache | Tensor | Required | Scale cache for K dynamic quantization. FP32 `[BlockNum,Nk,BlockSize,1]` for RoPE/M-RoPE; FLOAT8_E8M0 `[BlockNum,Nk,BlockSize,ceil(D/32)]` for M-RoPE MX. The in-place interface updates this tensor; the functional interface updates a copy and returns it. | torch.float32 or torch.float8_e8m0fnu | 4 |
| query_start_loc | Optional[Tensor] | required (positional parameter) | One of the position sources for RoPE. A valid tensor must be passed together with `seq_lens`; `None` must be explicitly passed for both M-RoPE use cases. | torch.int32 or None | `[Batch+1]` or None |
| seq_lens | Optional[Tensor] | required (positional parameter) | One of the position sources for RoPE. A valid tensor must be passed together with `query_start_loc`; `None` must be explicitly passed for both M-RoPE use cases. | torch.int32 or None | `[Batch]` or None |
| head_nums | List[int] | Required | Array of the Q/K/V head counts. Must be passed as `[Nq,Nk,Nv]`. | int | Length 3 |
| rotation | Optional[Tensor] | Optional | Shared Q/K matmul weight for RoPE/M-RoPE. Must be BF16 `[D,D]`; must be `None` for M-RoPE MX. | torch.bfloat16 or None | `[D,D]` or None |
| v_scale | Optional[Tensor] | Optional | Quantization scaling factor for the V branch. A valid tensor must be passed in all three use cases. M-RoPE MX quantizes V per head and per channel with static quantization. | torch.float32 | RoPE: `[Nv]`; M-RoPE/M-RoPE MX: `[Nv,D]` |
| layout_qkv | Optional[str] | Optional | N/T axis layout identifier of `qkv`. Default value is `"TND"`. If `None` or an empty string is passed, the default value is used. Case sensitive. Only `"TND"` and `"NTD"` are supported. | str | - |
| layout_q_out | Optional[str] | Optional | N/T axis layout identifier of `q_out` and `q_scale` (if present). Default value is `"NTD"`. If `None` or an empty string is passed, the default value is used. Case sensitive. Only `"TND"` and `"NTD"` are supported. | str | - |
| epsilon | float | Optional | RMSNorm parameter for preventing division by zero. Default value is `1e-6`. | float | - |
| mrope_position | Optional[Tensor] | Optional | M-RoPE position indices. Each row corresponds to one token, and the three columns are the T, H, and W positions in order. Must be `None` for RoPE. | torch.int32 | `[T,3]` |
| mrope_section | Optional[List[int]] | Optional | T/H/W section parameters `[t,h,w]` of M-RoPE. Must be `None` or an empty list for RoPE. | int | Length 0 or 3 |
| q_quant_mode | str | Optional | Quantization mode of the Q branch. Default value is `"PerTokenPerHead"`. Supports `"PerTokenPerHead"`, `"NoQuant"`, and `"Mx"`. | str | - |
| k_quant_mode | str | Optional | Quantization algorithm of the K branch. Default value is `"PerTokenPerHead"`; supports `"PerTokenPerHead"` and `"Mx"`. It does not select a dtype. The K storage type is still determined by the `k_cache` dtype and the use case contract. | str | - |
| q_out_dtype | torch.dtype | Optional | Data type of `q_out`. `torch.float8_e4m3fn` for RoPE/M-RoPE MX; `torch.bfloat16` for M-RoPE. | torch.dtype | - |

## Return Values

### qkv_rms_norm_rope_cache_with_k_scale_

| Name | Type | Type | Description | Data Type | Dimension (Shape) |
|---|---|---|---|---|---|
| q_out | Tensor | Required | Q branch output. For RoPE, FP8 E4M3FN, with shape `[T,Nq,D]` or `[Nq,T,D]` depending on `layout_q_out`; for M-RoPE, BF16 `[T,Nq,D]`; for M-RoPE MX, FP8 E4M3FN `[T,Nq,D]`. | torch.float8_e4m3fn or torch.bfloat16 | 3D |
| q_scale | Optional[Tensor] | Optional | For RoPE, FP32 rank 2; for M-RoPE, `None`; for M-RoPE MX, FLOAT8_E8M0 `[T,Nq,ceil(D/32)]`. | torch.float32, torch.float8_e8m0fnu, or None | 2D, 3D, or None |

### qkv_rms_norm_rope_cache_with_k_scale

| Name | Type | Type | Description | Data Type | Dimension (Shape) |
|---|---|---|---|---|---|
| q_out | Tensor | Required | Output of the Q branch. The shape and dtype rules are the same as those of the in-place interface. | torch.float8_e4m3fn or torch.bfloat16 | 3D |
| q_scale | Optional[Tensor] | Optional | Same as the in-place interface: FP32 for RoPE, None for M-RoPE, and FLOAT8_E8M0 rank 3 for M-RoPE MX. | torch.float32, torch.float8_e8m0fnu, or None | 2D, 3D, or None |
| k_cache_out | Tensor | Required | Updated K Cache. Its shape and dtype are the same as those of the input `k_cache`. | torch.float8_e4m3fn or torch.int8 | `[BlockNum,Nk,BlockSize,D]` |
| v_cache_out | Tensor | Required | Updated V Cache. Its shape and dtype are the same as those of the input `v_cache`. | torch.float8_e4m3fn | `[BlockNum,Nv,BlockSize,D]` |
| k_scale_cache_out | Tensor | Required | Updated scale cache of K dynamic quantization. Its shape and dtype are the same as those of the input `k_scale_cache`. | torch.float32 or torch.float8_e8m0fnu | 4D |

## Constraints

- This API supports invocation in single-operator mode for inference use cases. TorchAir graph mode invocation is currently not supported.
- The parameter combinations for the three use cases are as follows. Other cross combinations are not supported:

  | Condition | RoPE | M-RoPE | M-RoPE MX |
  |---|---|---|---|
  | query_start_loc / seq_lens | Both are valid tensors. | Both are None. | Both are None. |
  | mrope_position / mrope_section | The former is None, and the latter is None or an empty list. | The former exists, and the latter is a non-empty list of three items. | The former exists, and the latter is a non-empty list of three items. |
  | rotation | BF16 `[D,D]` | BF16 `[D,D]` | None |
  | q_quant_mode | `"PerTokenPerHead"` | `"NoQuant"` | `"Mx"` |
  | k_quant_mode | `"PerTokenPerHead"` | `"PerTokenPerHead"` | `"Mx"` |
  | q_out / q_scale | FP8 E4M3FN; FP32 rank2 | BF16; None | FP8 E4M3FN `[T,Nq,D]`; FLOAT8_E8M0 `[T,Nq,ceil(D/32)]` |
  | k_cache / k_scale_cache | FP8 E4M3FN; FP32 `[BlockNum,Nk,BlockSize,1]` | INT8; FP32 `[BlockNum,Nk,BlockSize,1]` | FP8 E4M3FN; FLOAT8_E8M0 `[BlockNum,Nk,BlockSize,ceil(D/32)]` |
  | v_cache / v_scale | FP8 E4M3FN; `[Nv]` | FP8 E4M3FN; `[Nv,D]` | FP8 E4M3FN; `[Nv,D]` per-head-per-channel static quantization |
  | layout_qkv -> layout_q_out | NTD->NTD, TND->TND, TND->NTD | TND->TND only | TND->TND only |

- Only `D=128` is supported. `head_nums=[Nq,Nk,Nv]` must meet `0<Nq<=64`, `Nq=8*Nk`, and `Nk=Nv`.
- In the M-RoPE use case, the three items of `mrope_section=[t,h,w]` must be non-negative. The range of both H and W is `[0,21]`. T has no independent lane upper limit, and `t+h+w<=64`. The shape of `mrope_position` must be `[T,3]`, and each position index must satisfy `0 <= value < MaxSeqLen`.
- `k_cache`, `v_cache`, and `k_scale_cache` must have the same `BlockNum` and `BlockSize`. `k_cache` and `v_cache` are both 4D with positive strides. The stride of the last dimension is 1, and the strides of the first three dimensions must be the same. The `k_scale_cache` of RoPE/M-RoPE is 4D with positive strides. The `k_scale_cache` of M-RoPE MX is also 4D. The last axis `ceil(D/32)` is contiguous and its stride is 1.
- In the RoPE use case, `query_start_loc[0]` should be 0, `query_start_loc[-1]` should equal `T`, and `query_start_loc` must be monotonically non-decreasing. The length of `seq_lens` should equal `query_start_loc.shape[0]-1`. Let the number of tokens of the `b`-th batch in this call be `L_b=query_start_loc[b+1]-query_start_loc[b]`. Then `0 <= L_b <= MaxSeqLen` and `L_b <= seq_lens[b] <= MaxSeqLen` must be met, where `MaxSeqLen=cos_sin.shape[0]`. If the actual sequence length of a non-empty batch exceeds the table capacity, the positional encoding accesses positions outside the table, and the behavior is undefined.
- `slot_mapping` must be a 1D torch.int32 tensor with shape `[T]`. Each element must be within `[0,BlockNum*BlockSize-1]`. Slots must be unique within the same call. For RoPE and M-RoPE, if duplicate slots exist, the final write order and results are undefined.
- `1<=T<=262144`.

## Deterministic Computation

Defaults to a deterministic implementation.

## Instruction

- In-place call for the RoPE use case:

  All optional parameters have default values.

  ```python
  import torch
  import torch_npu
  import cann_ops_transformer

  torch_npu.npu.set_device(0)

  T, Nq, Nk, Nv, D = 4, 16, 2, 2, 128
  block_num, block_size = 1, 16
  head_nums = [Nq, Nk, Nv]

  qkv = torch.randn(T, Nq + Nk + Nv, D, device="npu", dtype=torch.bfloat16)
  q_gamma = torch.ones(D, device="npu", dtype=torch.float32)
  k_gamma = torch.ones(D, device="npu", dtype=torch.float32)
  cos_sin = torch.zeros(16, D, device="npu", dtype=torch.float32)
  cos_sin[:, : D // 2] = 1.0
  slot_mapping = torch.arange(T, device="npu", dtype=torch.int32)
  k_cache = torch.empty(block_num, Nk, block_size, D, device="npu", dtype=torch.float8_e4m3fn)
  v_cache = torch.empty(block_num, Nv, block_size, D, device="npu", dtype=torch.float8_e4m3fn)
  k_scale_cache = torch.empty(block_num, Nk, block_size, 1, device="npu", dtype=torch.float32)
  query_start_loc = torch.tensor([0, T], device="npu", dtype=torch.int32)
  seq_lens = torch.tensor([T], device="npu", dtype=torch.int32)
  rotation = torch.eye(D, device="npu", dtype=torch.bfloat16)
  v_scale = torch.ones(Nv, device="npu", dtype=torch.float32)

  q_out, q_scale = cann_ops_transformer.qkv_rms_norm_rope_cache_with_k_scale_(
      qkv,
      q_gamma,
      k_gamma,
      cos_sin,
      slot_mapping,
      k_cache,
      v_cache,
      k_scale_cache,
      query_start_loc,
      seq_lens,
      head_nums,
      rotation=rotation,
      v_scale=v_scale,
  )

  print(q_out.shape, q_out.dtype, q_scale.shape, q_scale.dtype)
  ```

- Call for the M-RoPE use case:

  For M-RoPE, explicitly pass `None` at the fixed positions of `query_start_loc` and `seq_lens`, and pass the complete M-RoPE contract in the keyword-only parameters.

  ```python
  import torch
  import torch_npu
  import cann_ops_transformer

  torch_npu.npu.set_device(0)

  T, Nq, Nk, Nv, D = 5, 16, 2, 2, 128
  block_num, block_size = 1, 16
  head_nums = [Nq, Nk, Nv]

  qkv = torch.randn(T, Nq + Nk + Nv, D, device="npu", dtype=torch.bfloat16)
  q_gamma = torch.ones(D, device="npu", dtype=torch.float32)
  k_gamma = torch.ones(D, device="npu", dtype=torch.float32)
  cos_sin = torch.randn(32, D, device="npu", dtype=torch.float32)
  slot_mapping = torch.arange(T, device="npu", dtype=torch.int32)
  k_cache = torch.empty(block_num, Nk, block_size, D, device="npu", dtype=torch.int8)
  v_cache = torch.empty(block_num, Nv, block_size, D, device="npu", dtype=torch.float8_e4m3fn)
  k_scale_cache = torch.empty(block_num, Nk, block_size, 1, device="npu", dtype=torch.float32)
  rotation = torch.eye(D, device="npu", dtype=torch.bfloat16)
  v_scale = torch.ones(Nv, D, device="npu", dtype=torch.float32)
  mrope_position = torch.tensor(
      [[0, 1, 2], [3, 5, 8], [7, 4, 3], [7, 9, 6], [2, 6, 1]],
      device="npu",
      dtype=torch.int32,
  )

  q_out, q_scale, k_cache_out, v_cache_out, k_scale_cache_out = (
      cann_ops_transformer.qkv_rms_norm_rope_cache_with_k_scale(
          qkv,
          q_gamma,
          k_gamma,
          cos_sin,
          slot_mapping,
          k_cache,
          v_cache,
          k_scale_cache,
          None,
          None,
          head_nums,
          rotation=rotation,
          v_scale=v_scale,
          layout_qkv="TND",
          layout_q_out="TND",
          mrope_position=mrope_position,
          mrope_section=[22, 12, 10],
          q_quant_mode="NoQuant",
          q_out_dtype=torch.bfloat16,
      )
  )

  assert q_scale is None
  print(q_out.shape, q_out.dtype, k_cache_out.dtype)
  ```

- Call for the M-RoPE MX use case:

  For M-RoPE MX, do not pass `rotation`; both Q and K use `Mx`. Q/K data is FP8 E4M3FN, and the scale of each D32 block is `torch.float8_e8m0fnu`.

  ```python
  import torch
  import torch_npu
  import cann_ops_transformer

  torch_npu.npu.set_device(0)

  T, Nq, Nk, Nv, D = 5, 16, 2, 2, 128
  block_num, block_size = 1, 16
  head_nums = [Nq, Nk, Nv]

  qkv = torch.randn(T, Nq + Nk + Nv, D, device="npu", dtype=torch.bfloat16)
  q_gamma = torch.ones(D, device="npu", dtype=torch.float32)
  k_gamma = torch.ones(D, device="npu", dtype=torch.float32)
  cos_sin = torch.randn(32, D, device="npu", dtype=torch.float32)
  slot_mapping = torch.arange(T, device="npu", dtype=torch.int32)
  k_cache = torch.empty(block_num, Nk, block_size, D, device="npu", dtype=torch.float8_e4m3fn)
  v_cache = torch.empty(block_num, Nv, block_size, D, device="npu", dtype=torch.float8_e4m3fn)
  k_scale_cache = torch.empty(
      block_num, Nk, block_size, D // 32, device="npu", dtype=torch.float8_e8m0fnu
  )
  v_scale = torch.ones(Nv, D, device="npu", dtype=torch.float32)
  mrope_position = torch.tensor(
      [[0, 1, 2], [3, 5, 8], [7, 4, 3], [7, 9, 6], [2, 6, 1]],
      device="npu",
      dtype=torch.int32,
  )

  q_out, q_scale = cann_ops_transformer.qkv_rms_norm_rope_cache_with_k_scale_(
      qkv,
      q_gamma,
      k_gamma,
      cos_sin,
      slot_mapping,
      k_cache,
      v_cache,
      k_scale_cache,
      None,
      None,
      head_nums,
      v_scale=v_scale,
      layout_qkv="TND",
      layout_q_out="TND",
      mrope_position=mrope_position,
      mrope_section=[22, 12, 10],
      q_quant_mode="Mx",
      k_quant_mode="Mx",
      q_out_dtype=torch.float8_e4m3fn,
  )

  print(q_out.shape, q_out.dtype, q_scale.shape, q_scale.dtype)
  ```

- TorchAir graph mode call:

  TorchAir graph mode calls are not supported currently.
