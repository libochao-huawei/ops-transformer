# dense_lightning_indexer_kl_loss_grad / dense_lightning_indexer_kl_loss_grad_metadata

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T08:37:00.763Z -->

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

- Description:

    - `dense_lightning_indexer_kl_loss_grad`: Computes the backward outputs in the Lightning Indexer KL Loss training use case. This API takes `q`, `k`, and `w` of the Lightning Indexer branch, as well as the target distributions `attn_softmax_l1_norm` and `softmax_lse` pre-computed by the main Attention branch, and outputs `dq`, `dk`, `dw`, and `softmax_out` of the Lightning Indexer branch.
    - `dense_lightning_indexer_kl_loss_grad_metadata`: Generates the kernel splitting metadata before the main operator. The output must be passed as the `metadata` input of `dense_lightning_indexer_kl_loss_grad`.

- Formula:

    Indexer logits can be expressed as:

   $$
   S_{t,:}=q_{t,:}@K^{T}
   $$

   $$
   I_{t,:}=W_{t,:}@\mathrm{ReLU}(S_{t,:})
   $$

   Here, $q$ and $K$ correspond to q and k of this API respectively, and $W$ corresponds to w of this API. The softmax output of the Indexer branch is:

   $$
   y_{t,:}=\operatorname{Softmax}(I_{t,:})
   $$

   This API writes $y$ to softmaxOut. The target distribution $p$ is provided by the attnSoftmaxL1Norm input. It is equivalent to the result obtained internally in the legacy kernel by performing head sum and L1 normalization on the main attention score. If KL Loss is computed subsequently, its form is the same as in the legacy version:

   $$
   L(I){=}\sum_tD_{KL}(p_{t,:}||\operatorname{Softmax}(I_{t,:}))
   $$

   $$
   D_{KL}(a||b){=}\sum_ia_i\mathrm{log}{\left(\frac{a_i}{b_i}\right)}
   $$

   The gradient expression of the loss can be obtained by derivation:

   $$
   dI_{t,:}=\operatorname{Softmax}(I_{t,:})-p_{t,:}
   $$

   The gradients of the w, q, and k matrices can be computed using the chain rule:

   $$
   dW_{t,:}=dI_{t,:}\text{@}\left(\mathrm{ReLU}(S_{t,:})\right)^{T}
   $$

   $$
   dq_{t,:}=dS_{t,:}@K
   $$

   $$
   dK=\sum_t\left(dS_{t,:}\right)^{T}@q_{t,:}
   $$

   dK is accumulated directly to the corresponding key positions. No scatter-add is required.

> [!NOTE]
>
> `cmp_residual_k` is an optional input of both `dense_lightning_indexer_kl_loss_grad` and `dense_lightning_indexer_kl_loss_grad_metadata`. When the KV is compressed and `mask_mode=3`, this parameter is used to restore the key length before compression: `pre_compress_k_len = compressed_k_len * cmp_ratio + cmp_residual_k[b]`, thereby determining the causal valid range.

## Function Prototype

```python
from cann_ops_transformer.ops import dense_lightning_indexer_kl_loss_grad_metadata

dense_lightning_indexer_kl_loss_grad_metadata(
    num_heads_q,
    num_heads_k,
    head_dim,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    batch_size=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    layout_q=None,
    layout_k=None,
    mask_mode=None,
    cmp_ratio=None
) -> Tensor
```

```python
from cann_ops_transformer.ops import dense_lightning_indexer_kl_loss_grad

dense_lightning_indexer_kl_loss_grad(
    q,
    k,
    w,
    attn_softmax_l1_norm,
    softmax_lse,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    metadata=None,
    layout_q="TND",
    layout_k="TND",
    mask_mode=3,
    cmp_ratio=1
) -> (Tensor, Tensor, Tensor, Tensor)
```

## Parameters

### dense_lightning_indexer_kl_loss_grad

| Name | Type | Input/Output | Description | Data Type | Dimension (shape) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| q | Tensor | required input | Query input of the Lightning Indexer branch. D is fixed to 128. | `float16`, `bfloat16` | `[B, S1, N1, D]` when `layout_q="BSND"`; `[T1, N1, D]` when `layout_q="TND"`. |
| k | Tensor | required input | Key input of the Lightning Indexer branch. N2 supports only 1. D is fixed to 128. | Same as `q` | `[B, S2, N2, D]` when `layout_k="BSND"`; `[T2, N2, D]` when `layout_k="TND"`. |
| w | Tensor | required input | Head weights of the Indexer logits. | `float32` | `[B, S1, N1]` when `layout_q="BSND"`; `[T1, N1]` when `layout_q="TND"`. |
| attn_softmax_l1_norm | Tensor | required input | Target distribution `p` precomputed by the main Attention branch. | `float32` | `[B, S1, N2, S2]` when `layout_k="BSND"`; `[T1, N2, MAX_SEQ_K]` when `layout_k="TND"`. |
| softmax_lse | Tensor | required input | Intermediate output of the main Attention computation. | `float32` | `[B, N2, S1]` when `layout_q="BSND"`; `[N2, T1]` when `layout_q="TND"`. |
| cu_seqlens_q | Tensor | optional input | Prefix sum of query sequence lengths in TND use cases. The first element is 0. | `int32` | `[B + 1]` |
| cu_seqlens_k | Tensor | optional input | Prefix sum of key sequence lengths in TND use cases. The first element is 0. | `int32` | `[B + 1]` |
| seqused_q | Tensor | optional input | Actual query length used in each batch. | `int32` | `[B]` |
| seqused_k | Tensor | optional input | Actual key length used in each batch. | `int32` | `[B]` |
| cmp_residual_k | Tensor | optional input | Residual length in compressed key use cases. When `mask_mode=3` and `cmp_ratio!=1`, it is used to restore the key length before compression. | `int32` | `[B]` |
| metadata | Tensor | optional input | Task splitting result generated by `dense_lightning_indexer_kl_loss_grad_metadata`. Passing this parameter is recommended. | `int32` | `[64]` |
| layout_q | str | optional attribute | Layout of `q`. | `str` | Supports `"BSND"` and `"TND"`. The default value is `"BSND"`. |
| layout_k | str | optional attribute | Layout of `k`. | `str` | Supports `"BSND"` and `"TND"`. The default value is `"BSND"`. |
| mask_mode | int | optional attribute | Mask mode. | `int` | Currently, `0` and `3` are supported. The default value is `0`. |
| cmp_ratio | int | optional attribute | Key compression ratio. | `int` | Value range `[1, 128]`. The default value is `1`. |

### dense_lightning_indexer_kl_loss_grad_metadata

| Name | Type | Type | Description | Data Type | Dimension (Shape) |
|--------|----------|-----------|------|----------|-------------|
| num_heads_q | int | Required | Number of heads of `q`, that is, `N1` of `q` in the main API. Currently, [1, 128] is supported. | int32 | - |
| num_heads_k | int | Required | Number of heads of `k`, that is, `N2` of `k` in the main API. Currently, only 1 is supported. | int32 | - |
| head_dim | int | Required | Last dimension `D` of `q` and `k`. Currently, only 128 is supported. | int32 | - |
| cu_seqlens_q | Tensor | Optional | Prefix sum of query sequence lengths in TND use cases. The first element is 0. The data format is ND, and non-contiguous tensors are supported. | int32 | (B+1, ) |
| cu_seqlens_k | Tensor | Optional | Prefix sum of key sequence lengths in TND use cases. The first element is 0. The data format is ND, and non-contiguous tensors are supported. | int32 | (B+1, ) |
| seqused_q | Tensor | Optional | Indicates the length of query actually used in each batch. The data format is ND, and non-contiguous tensors are supported. | int32 | (B, ) |
| seqused_k | Tensor | Optional | Indicates the length of key actually used in each batch. The data format is ND, and non-contiguous tensors are supported. | int32 | (B, ) |
| cmp_residual_k | Tensor | Optional | Residual length in compressed key use cases. The data format is ND, and non-contiguous tensors are supported. | int32 | (B, ) |
| batch_size | int | Optional | Batch size. It is recommended to explicitly pass this parameter in BSND use cases. In TND use cases, it can also be derived from `cu_seqlens_q`. The default value is 0. | int32 | - |
| max_seqlen_q | int | Optional | Maximum query sequence length in a single batch. The default value is 0. | int32 | - |
| max_seqlen_k | int | Optional | Maximum compressed key sequence length in a single batch. The default value is 0. | int32 | - |
| layout_q | str | Optional | Layout on the `q` side. `"BSND"` and `"TND"` are supported. The default value is `"BSND"`. | string | - |
| layout_k | str | Optional | Layout on the `k` side. `"BSND"` and `"TND"` are supported. The default value is `"BSND"`. | string | - |
| mask_mode | int | Optional | Mask mode. 0 indicates No mask, and 3 indicates the rightDownCausal mode. The default value is 0. | int32 | - |
| cmp_ratio | int | Optional | Key compression ratio. Currently, [1, 128] is supported. The default value is 1. | int32 | - |

## Return Values

### dense_lightning_indexer_kl_loss_grad

| Name | Type | Type | Description | data type | Dimension (Shape) |
|--------|----------|-----------|------|----------|-------------|
| dq | Tensor | Required | Gradient of `q`. | Same as `q` | Same as `q` |
| dk | Tensor | Required | Gradient of `k`. | Same as `k` | Same as `k` |
| dw | Tensor | Required | Gradient of `w`. | float32 | Same as `w` |
| softmax_out | Tensor | Required | Softmax output of the Lightning Indexer branch. | float32 | Same as `attn_softmax_l1_norm` |

### dense_lightning_indexer_kl_loss_grad_metadata

| Name | Type | Type | Description | Data Type | Dimension (Shape) |
|--------|----------|-----------|------|----------|-------------|
| metadata | Tensor | Required | Indexes of the Batch, Head, and Q/K block partition of the Attention computation task on each AI Core. The data format is ND. Non-contiguous tensors are not supported. | int32 | shape (64, ) |

## Constraints

- This API supports training use cases.
- This API supports the single-operator mode and aclgraph mode.
- `q` and `k` must have the same data type. `float16` and `bfloat16` are supported.
- The data types of `w`, `attn_softmax_l1_norm`, `dw`, and `softmax_out` should be `float32`.
- The data types of `cu_seqlens_q`, `cu_seqlens_k`, `seqused_q`, `seqused_k`, `cmp_residual_k`, and `metadata` should be `int32`.
- `layout_q` and `layout_k` support `BSND` and `TND`.
- If `layout_q="TND"`, pass `cu_seqlens_q`. If `layout_k="TND"`, pass `cu_seqlens_k`.
- Currently, only 1 is supported for `num_heads_k`.
- In TND use cases, the last dimension of `attn_softmax_l1_norm` is max_seqlen_k, the longest seqlens_k across all batches.
- Currently, `mask_mode` supports `0` and `3`.
- The value range of `cmp_ratio` is `[1, 128]`.
- When the key is compressed and `mask_mode=3`, the key length before compression is calculated as `compressed_k_len * cmp_ratio + cmp_residual_k[b]`. In the CP splitting use case, when each CP shard is called separately, pass the query length of the current shard to `q`, pass the compressed key prefix length corresponding to the shard to `k`, and pass the corresponding `cmp_residual_k` according to this formula.
- **Specification constraints**:

    | Specification Item | Value | Description |
    | :--- | :--- | :--- |
    | B | - | Supports generalization. |
    | S1 | - | Supports generalization. |
    | S2 | - | Supports generalization. |
    | N1 | 1-128 | - |
    | N2 | 1 | Currently, only N2=1 is supported. |
    | D | 128 | The last dimensions of q and k must be the same. |
    | cmp_ratio | 1-128 | - |
    | mask_mode | 0 and 3 | - |

## Deterministic Computation

- Deterministic computation is not supported by default. It can be enabled through the PyTorch switch (torch.use_deterministic_algorithms).

## Example

- Call in single-operator mode:

```python
import torch
import torch_npu
from cann_ops_transformer.ops import dense_lightning_indexer_kl_loss_grad
from cann_ops_transformer.ops import dense_lightning_indexer_kl_loss_grad_metadata

torch_npu.npu.set_device(0)
device = torch.device("npu:0")

q_len = 32
compressed_k_len = 128
cmp_ratio = 4
cmp_residual_k = 0
batch_size = 1
num_heads_q = 64
num_heads_k = 1
head_dim = 128
layout = "TND"
mask_mode = 3
dtype = torch.bfloat16

q = torch.randn(q_len, num_heads_q, head_dim, dtype=dtype, device=device)
k = torch.randn(compressed_k_len, num_heads_k, head_dim, dtype=dtype, device=device)
w = torch.randn(q_len, num_heads_q, dtype=torch.float32, device=device) * (0.1 / 6.0)
softmax_lse = torch.randn((num_heads_k, q_len), dtype=torch.float32, device=device)
attn_l1_cpu = torch.zeros((q_len, 1, compressed_k_len), dtype=torch.float32)

attn_softmax_l1_norm = attn_l1_cpu.to(device)

cu_seqlens_q = torch.tensor([0, q_len], dtype=torch.int32, device=device)
cu_seqlens_k = torch.tensor([0, compressed_k_len], dtype=torch.int32, device=device)
cmp_residual_k_tensor = torch.tensor([cmp_residual_k], dtype=torch.int32, device=device)

metadata = dense_lightning_indexer_kl_loss_grad_metadata(
    num_heads_q,
    num_heads_k,
    head_dim,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    cmp_residual_k=cmp_residual_k_tensor,
    batch_size=batch_size,
    max_seqlen_q=q_len,
    max_seqlen_k=compressed_k_len,
    layout_q=layout,
    layout_k=layout,
    mask_mode=mask_mode,
    cmp_ratio=cmp_ratio,
)

dq, dk, dw, softmax_out = dense_lightning_indexer_kl_loss_grad(
    q,
    k,
    w,
    attn_softmax_l1_norm,
    softmax_lse,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    cmp_residual_k=cmp_residual_k_tensor,
    metadata=metadata,
    layout_q=layout,
    layout_k=layout,
    mask_mode=mask_mode,
    cmp_ratio=cmp_ratio,
)

torch.npu.synchronize()
print(dq.shape, dk.shape, dw.shape, softmax_out.shape)
```
