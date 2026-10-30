# und_gen_qkv_rms_norm_rope_cache

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T16:06:14.659Z -->

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

  This API wraps `aclnnUndGenQkvRmsNormRopeCache` and is used for Q/K/V preprocessing and PagedAttention KV Cache update in multimodal LLM inference. It concatenates the Q/K/V inputs of the understanding phase (undecoded) and the generation phase (generated) into one output sequence via indirect addressing with `cat_indices`, splits Q/K/V along the head dimension token by token, applies per-head RMSNorm and MRoPE (multimodal rotary position embedding) to Q and K, and leaves V unnormalized and unrotated. Q is returned as the output, and K and V are written to the paged KV Cache based on `slot_mapping`. The RMSNorm weight is selected from two sets of weights depending on whether the source token falls in the understanding phase or the generation phase. All computations are performed in float32, and both the return value and the KV Cache are bfloat16.

- **In-place update**:

  `k_cache` and `v_cache` are input parameters updated in place (corresponding to `Tensor(a!)` and `Tensor(b!)` in the schema). After the call, read these two tensors directly to obtain the updated results. The API does not return them separately.

## Function Prototype

```python
cann_ops_transformer.und_gen_qkv_rms_norm_rope_cache(
    und_qkv,
    und_weights_q,
    und_weights_k,
    cos_sin_cache,
    k_cache,
    v_cache,
    slot_mapping,
    positions,
    gen_qkv=None,
    gen_weights_q=None,
    gen_weights_k=None,
    cat_indices=None,
    *,
    num_heads_q=8,
    num_heads_k=1,
    num_heads_v=1,
    norm_eps=1e-6,
    mrope_section=(),
)
```

## Mandatory

Let `T = und_len + gen_len` be the total number of output tokens, `N = Hq + Hk + Hv`, `D` the head dimension (fixed at 128), and `Bn`/`Bs` the number of pages and rows per page in the KV Cache.

| Name | Type | Type | Description | Data Type | Dimension (Shape) |
|---|---|---|---|---|---|
| und_qkv | Tensor | Required | Fused Q/K/V input of the understanding phase, with heads arranged as `[Hq,Hk,Hv]` along the head dimension. | torch.bfloat16 | `[und_len,N,D]` |
| und_weights_q | Tensor | Required | RMSNorm weight of the Q branch in the understanding phase. | torch.bfloat16 | `[D]` |
| und_weights_k | Tensor | Required | RMSNorm weight of the K branch in the understanding phase. | torch.bfloat16 | `[D]` |
| cos_sin_cache | Tensor | Required | Position encoding table, with the first `D/2` columns for cos and the last `D/2` columns for sin. | torch.float32 | `[max_pos,D]` |
| k_cache | Tensor | Required | K Cache, updated in place. Memory must be continuous. | torch.bfloat16 | `[Bn,Bs,Hk,D]` |
| v_cache | Tensor | Required | V Cache, updated in place. Memory must be continuous. | torch.bfloat16 | `[Bn,Bs,Hv,D]` |
| slot_mapping | Tensor | Required | Slot index for writing each output token to the cache, where `slot = block_idx * Bs + row_idx`. | torch.int64 | `[T]` |
| positions | Tensor | Required | T/H/W positional indices of MRoPE, with one axis per row. | torch.int64 | `[3,T]` |
| gen_qkv | Optional[Tensor] | Optional | Fused Q/K/V input of the generation phase. The N and D dimensions must be consistent with those of `und_qkv`. In the current version, this parameter must be passed and `gen_len` must be a positive number. Passing `None` (prefill only) is not supported yet. | torch.bfloat16 | `[gen_len,N,D]` |
| gen_weights_q | Optional[Tensor] | Optional | RMSNorm weight of the Q branch in the generation phase. In the current version, this parameter must be passed and must be paired with `gen_weights_k`. | torch.bfloat16 | `[D]` |
| gen_weights_k | Optional[Tensor] | Optional | RMSNorm weight of the K branch in the generation phase. In the current version, this parameter must be passed and must be paired with `gen_weights_q`. | torch.bfloat16 | `[D]` |
| cat_indices | Optional[Tensor] | Optional | Mapping from output tokens to source tokens `out_t -> src_t`. Values smaller than `und_len` indicate the understanding segment; otherwise, the generation segment. In the current version, this parameter must be passed. Passing `None` (identity mapping for a single sequence) is not supported yet. | torch.int64 | `[T]` |
| num_heads_q | int | Optional | Number of Q heads `Hq`. The default value is `8`. | int | - |
| num_heads_k | int | Optional | Number of K heads `Hk`. The default value is `1`. | int | - |
| num_heads_v | int | Optional | Number of V heads `Hv`. The default value is `1`. | int | - |
| norm_eps | float | Optional | RMSNorm parameter to prevent division by zero. The default value is `1e-6`, which must be a positive number. | float | - |
| mrope_section | List[int] | Optional | T/H/W section parameters `[t,h,w]` of MRoPE. The default value is `()`. Passing an empty list falls back to standard RoPE (the three axes share the same source). | int | 0 or 3 |

## Return Values

`q`: Tensor, Q output after concatenating the understanding phase and generation phase. The data type is torch.bfloat16, and the shape is `[T,Hq,D]`.

`k_cache` and `v_cache` are updated in place and are not included in the return values. After the call, read the input tensors directly.

## Constraints

- This API supports single-operator mode calls in inference use cases. A TorchAir graph mode converter is also provided in `ops/graph_convert/`.
- Only `D=128` is supported; `(num_heads_q, num_heads_k, num_heads_v)` supports only the `(8,1,1)` and `(16,2,2)` combinations, and the sum of the three values must equal the N dimension of `und_qkv`.
- The current version requires that all four parameters `gen_qkv`, `gen_weights_q`, `gen_weights_k`, and `cat_indices` must be passed, and `gen_len` must be positive. The degraded path where `None` is passed in (pure prefill, single-sequence identity mapping) is not supported yet, and the call returns a failure. Support will be provided in a later version.
- `Bs` has no restriction. It is not involved in addressing or splitting. The only constraint is the capacity `Bn * Bs >= T`.
- `T` has no upper limit and only needs to be positive. The actual upper limit is determined by the KV Cache capacity `Bn * Bs >= T`. `T = 64K` has been verified on board.
- The three items of `mrope_section` must be non-negative, and their sum must not exceed `D/2`.
- `k_cache` and `v_cache` must be continuous in memory, and their `Bn` and `Bs` must be consistent.
- The following values are runtime data and cannot be verified on the Host side. The caller must guarantee them: the value of `slot_mapping` must fall in `[0, Bn*Bs)` and be unique, the value of `positions` must fall in `[0, max_pos)`, and the value of `cat_indices` must fall in `[0, T)`.

## Deterministic Computation

Deterministic computation is supported by default, provided that the values of `slot_mapping` are mutually distinct. When multiple tokens in a single call write to the same slot, the write order and results are undefined across cores.

## Calling Instructions

- Single-operator mode call:

  ```python
  import torch
  import torch_npu
  import cann_ops_transformer

  torch_npu.npu.set_device(0)

  D, Hq, Hk, Hv = 128, 8, 1, 1
  N = Hq + Hk + Hv
  und_len, gen_len = 5, 3
  total = und_len + gen_len
  block_num, block_size = 2, 128

  und_qkv = torch.randn(und_len, N, D, dtype=torch.bfloat16).npu()
  gen_qkv = torch.randn(gen_len, N, D, dtype=torch.bfloat16).npu()
  w = [torch.randn(D, dtype=torch.bfloat16).npu() for _ in range(4)]
  cos_sin_cache = torch.randn(32, D, dtype=torch.float32).npu()
  k_cache = torch.zeros(block_num, block_size, Hk, D, dtype=torch.bfloat16).npu()
  v_cache = torch.zeros(block_num, block_size, Hv, D, dtype=torch.bfloat16).npu()
  slot_mapping = torch.arange(total, dtype=torch.int64).npu()
  positions = torch.randint(0, 32, (3, total), dtype=torch.int64).npu()
  cat_indices = torch.tensor([0, 5, 1, 6, 2, 7, 3, 4], dtype=torch.int64).npu()

  q = cann_ops_transformer.und_gen_qkv_rms_norm_rope_cache(
      und_qkv, w[0], w[1], cos_sin_cache, k_cache, v_cache, slot_mapping, positions,
      gen_qkv, w[2], w[3], cat_indices,
      num_heads_q=Hq, num_heads_k=Hk, num_heads_v=Hv,
      norm_eps=1e-6, mrope_section=[16, 16, 16])
  print(q.shape, k_cache.shape)
  ```
