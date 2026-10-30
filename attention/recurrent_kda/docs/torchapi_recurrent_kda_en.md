# recurrent_kda

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T13:02:34.728Z -->

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

- Description: aclnnRecurrentKda is the recurrent implementation of the linear attention forward computation API based on the Gated Delta Rule, targeting decode / MTP (Multi-Token Prediction) short-sequence use cases. This API advances the recurrent state $S_t$ token by token, supports variable-length input (with `cuSeqlensOptional`), MTP slot reuse (with `ssmStateIndicesOptional` and `numAcceptedTokensOptional`), and optionally performs numerical stabilization in the kernel, such as interpreting `gate` as the raw gate, applying L2 normalization to q/k, and applying sigmoid to beta.
- Formula:

  Assume that the internal state matrix participates in the computation with the shape `[V_dim, K_dim]`. When `useQkL2normInKernel=true`, the operator first applies L2 normalization to $q_t$ and $k_t$ of each token; then it multiplies only $q_t$ by the scaling factor `scale` to obtain $\bar q_t$ and $\bar k_t$.

  When `useGateInKernel=false`, the input `gate` is already the step log gate, that is, $\ell_t=gate_t$. When `useGateInKernel=true`, let $x_t=gate_t+dtBias$ (no bias is added if `dtBiasOptional` is not passed) and $a=\exp(A_{\log})$. Then:

  $$
  \ell_t =
  \begin{cases}
  lowerBound\cdot\sigma(a x_t), & safeGate=true \\
  -a\cdot\operatorname{softplus}(x_t), & safeGate=false
  \end{cases},
  \qquad
  d_t=\exp(\ell_t)
  $$

  The valid values of beta are:

  $$
  \bar\beta_t =
  \begin{cases}
  \beta_t, & useBetaSigmoidInKernel=false \\
  \sigma(\beta_t), & useBetaSigmoidInKernel=true,\ allowNegEigval=false \\
  2\sigma(\beta_t), & useBetaSigmoidInKernel=true,\ allowNegEigval=true
  \end{cases}
  $$

  The recurrent update for each token is:

  $$
  \begin{aligned}
  \bar S_t &= S_{t-1}\odot d_t \\
  \Delta_t &= \bar\beta_t\left(v_t-\bar S_t\bar k_t\right) \\
  S_t &= \bar S_t+\Delta_t\bar k_t^T \\
  o_t &= S_t\bar q_t
  \end{aligned}
  $$

  where $S_{t-1},S_t \in R^{V_{dim}\times K_{dim}}$, $q_t,k_t\in R^{K_{dim}}$, $v_t,o_t\in R^{V_{dim}}$. $d_t$ is broadcast along the K dimension of the state matrix. Multiple value heads can share the same query/key head, and $H_v$ must be divisible by $H_k$.

- Symbol Description

  | Symbol      | Meaning                                                                        |
  | ----------- | ---------------------------------------------------------------------------------- |
  | B           | Batch Size, the batch size of input samples.                                       |
  | T           | Token Capacity, B*T for BSND, and the first input dimension for TND; the number of valid tokens can be less than T. |
  | H_k         | Number of Query/Key heads.                                                         |
  | H_v         | Number of Value/Gate heads (H_v must be divisible by H_k).                         |
  | K_dim       | Dimension of each Key head.                                                        |
  | V_dim       | Dimension of each Value head.                                                      |
  | seq_num     | Number of logical sequences (B for fixed length, and the length of `cuSeqlensOptional` minus 1 for variable length). |
  | scale       | Scale factor, typically K_dim**-0.5.                                               |
  | gate        | Gate tensor. When `useGateInKernel=true`, it is the raw gate input, which is converted to the step log gate $\ell_t$ inside the operator; when `useGateInKernel=false`, it is the step log gate pre-computed externally. |
  | $d_t$       | Recurrent decay coefficient, $d_t=\exp(\ell_t)$, broadcast along the K dimension of the state matrix. |
  | A_log       | Raw gate branch parameter. The operator uses $\exp(A_{\log})$ in the gate computation. |
  | dtBias      | Raw gate bias.                                                                     |
  | beta        | Delta update coefficient.                                                          |
  | $S_t$       | Recurrent state matrix. Stored as `[state_capacity, H_v, V_dim, K_dim]` in the `stateVFirst=True` layout, and as `[state_capacity, H_v, K_dim, V_dim]` in the `stateVFirst=False` layout. |
  | $\bar S_t$  | Decayed state, $\bar S_t=S_{t-1}\odot d_t$                                        |
  | $\Delta_t$  | Delta update amount, $\Delta_t=\bar\beta_t(v_t-\bar S_t\bar k_t)$                    |

## Function Prototype

```python
cann_ops_transformer.recurrent_kda(
    q,
    k,
    v,
    g,
    beta,
    initial_state,
    *,
    cu_seqlens=None,
    ssm_state_indices=None,
    A_log=None,
    dt_bias=None,
    num_accepted_tokens=None,
    layout="BSND",
    scale=None,
    output_final_state=False,
    inplace_final_state=True,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    safe_gate=False,
    lower_bound=None,
    state_v_first=False,
) -> Tuple[Tensor, Optional[Tensor]]
```

## Parameters

| Name | Type | Required/Optional | Description | data type | Dimension (Shape) |
|--------|----------|-----------|------|----------|-------------|
| q | Tensor | Required | Query input. | bfloat16 | BSND: [B,T,H,K]; TND: [T,H,K] |
| k | Tensor | Required | Key input, with the same shape as q. | bfloat16 | Same as q |
| v | Tensor | Required | Value input. | bfloat16 | BSND: [B,T,HV,V]; TND: [T,HV,V] |
| g | Tensor | Required | Gate input. | float16, bfloat16, float32 | BSND: [B,T,HV,K]; TND: [T,HV,K] |
| beta | Tensor | Required | Delta update coefficient. | float16, bfloat16, float32 | BSND: [B,T,HV]; TND: [T,HV] |
| initial_state | Tensor | Required | Initial recurrent state; updated in inplace mode. | bfloat16, float32 | state_v_first=False: [state_capacity,HV,K,V]; when True: [state_capacity,HV,V,K] |
| cu_seqlens | Tensor | Optional | Cumulative lengths of variable-length sequences. | int32, int64 | [seq_num+1] |
| ssm_state_indices | Tensor | Optional | State slot index corresponding to each logical sequence. | int32, int64 | [seq_num] |
| A_log | Tensor | Optional | A_log used in the kernel gating mode. | float32 | [HV] |
| dt_bias | Tensor | Optional | dt bias used in the kernel gating mode. | float32 | [HV*K] |
| num_accepted_tokens | Tensor | Optional | Number of accepted tokens per sequence in speculative decoding. | int32, int64 | [seq_num] |
| layout | str | Optional | Input layout, BSND or TND. Default: BSND. | - | - |
| scale | float | Optional | QK scale factor; if None, K**-0.5 is used. | - | - |
| output_final_state | bool | Optional | Whether to return the valid final state. Default: False. | - | - |
| inplace_final_state | bool | Optional | Whether to write the final state back to initial_state in place. Default: True. | - | - |
| use_qk_l2norm_in_kernel | bool | Optional | Whether to perform Q/K L2 normalization in the kernel. Default: False. | - | - |
| use_gate_in_kernel | bool | Optional | Whether to process gating in the kernel. Default: False. | - | - |
| use_beta_sigmoid_in_kernel | bool | Optional | Whether to apply sigmoid to beta in the kernel. Default: False. | - | - |
| allow_neg_eigval | bool | Optional | Whether to allow the negative eigenvalue path. Default: False. | - | - |
| safe_gate | bool | Optional | Whether to enable safe gating. Default: False. | - | - |
| lower_bound | float | Optional | Lower bound of safe gating; if None, -5.0 is used. | - | - |
| state_v_first | bool | Optional | Whether the last two dimensions of the state are in [V,K] order. Default: False. | - | - |

## Return Values

- The first Tensor is the attention output, with the same shape as v and a dtype of bfloat16.
- The second return value is the final state. None is returned when output_final_state=False; when True, its shape and dtype are the same as those of initial_state. When inplace_final_state=True, this return value corresponds to the updated initial_state.

## Constraints

- Only K=128 and V=128 or 256 are supported.
- H and HV must be positive, and HV must be divisible by H.
- Under the TND layout, q, k, v, and g are 3-dimensional and beta is 2-dimensional; under the BSND layout, q, k, v, and g are 4-dimensional and beta is 3-dimensional.
- When safe_gate=True, use_gate_in_kernel=True must also be set, and the value range of lower_bound is [-5, 0).
- When ssm_state_indices is not provided, the state_capacity of initial_state must be equal to the number of logical sequences; when it is provided, state slots are selected by the indices.
- output_final_state and inplace_final_state are independent switches; the second return value contains a valid final state only when output_final_state=True.

## Usage Instructions

- Calling in single-operator mode

    ```python
    import torch
    import torch_npu
    from cann_ops_transformer.ops import recurrent_kda

    torch_npu.npu.set_device(0)

    B, S, H, HV, K, V = 1, 64, 96, 96, 128, 128
    q = torch.randn(B, S, H, K, device="npu", dtype=torch.bfloat16)
    k = torch.randn(B, S, H, K, device="npu", dtype=torch.bfloat16)
    v = torch.randn(B, S, HV, V, device="npu", dtype=torch.bfloat16)
    g = torch.randn(B, S, HV, K, device="npu", dtype=torch.float32)
    beta = torch.randn(B, S, HV, device="npu", dtype=torch.float32)
    initial_state = torch.randn(
        B,
        HV,
        K,
        V,
        device="npu",
        dtype=torch.float32,
    )

    attn_out, final_state = recurrent_kda(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        layout="BSND",
        output_final_state=True,
    )
    ```
