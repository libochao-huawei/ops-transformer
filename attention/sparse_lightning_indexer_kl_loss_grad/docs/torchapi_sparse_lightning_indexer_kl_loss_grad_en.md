# sparse_lightning_indexer_kl_loss_grad

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T13:35:15.093Z -->

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
<!-- npu="310p" id8 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id8 -->
<!-- npu="910" id9 -->
- <term>Atlas training products</term>: Not supported
<!-- end id9 -->

## Function

- **Description**:

  - `sparse_lightning_indexer_kl_loss_grad`: Computes the backward output for the Lightning Indexer KL Loss training use case. This API takes `q`, `k`, `w`, and `sparse_indices` of the Lightning Indexer branch, and the target distribution `attn_softmax_l1_norm` precomputed by the main Attention branch as inputs, and outputs `dq`, `dk`, `dw`, and `softmax_out` of the Lightning Indexer branch.
  - `sparse_lightning_indexer_kl_loss_grad_metadata`: Used to generate kernel splitting metadata before the main operator. The output must be passed as the `metadata` input of `sparse_lightning_indexer_kl_loss_grad`.

- **Formula**:

    For each query token, after selecting the top-k keys based on `sparse_indices`, the Indexer logits can be expressed as:

    $$
    S_{t,:}=q_{t,:}@K_{\operatorname{topk}(t),:}^{T}
    $$

    $$
    I_{t,:}=W_{t,:}@\mathrm{ReLU}(S_{t,:})
    $$

    The softmax output of the Indexer branch is:

    $$
    y_{t,:}=\operatorname{Softmax}(I_{t,:})
    $$

    The target distribution `p` is provided by the `attn_softmax_l1_norm` input. The main operator writes `y` to `softmax_out`. The gradient used for backpropagation in the current KL Loss backward is:

    $$
    p_{\mathrm{reduce}}=\operatorname{ReduceSum}(p,\operatorname{axis}=-1,\operatorname{keepdim}=true)
    $$

    $$
    dI_{t,:}=y_{t,:} * p_{\mathrm{reduce}} - p_{t,:}
    $$

    Then, the following is computed using the chain rule:

    $$
    dW_{t,:}=dI_{t,:}\text{@}\left(\mathrm{ReLU}(S_{t,:})\right)^{T}
    $$

    $$
    dq_{t,:}=dS_{t,:}@K_{\operatorname{topk}(t),:}
    $$

    $$
    dK_{\operatorname{topk}(t),:}=\left(dS_{t,:}\right)^{T}@q_{t,:}
    $$

    When `dk` is written back, scatter-add is performed at the key positions indicated by `sparse_indices`. Invalid top-k positions are excluded from the computation.

> [!NOTE]
>
> `cmp_residual_k` is an optional input of both `sparse_lightning_indexer_kl_loss_grad` and `sparse_lightning_indexer_kl_loss_grad_metadata`. When compressed KV is used and `mask_mode=3`, this parameter is used to restore the key length before compression: `pre_compress_k_len = compressed_k_len * cmp_ratio + cmp_residual_k[b]`, so as to determine the valid causal range.

## Function Prototype

```python
cann_ops_transformer.sparse_lightning_indexer_kl_loss_grad_metadata(
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
    topk=None,
    layout_q=None,
    layout_k=None,
    mask_mode=None,
    cmp_ratio=None
) -> Tensor
```

```python
cann_ops_transformer.sparse_lightning_indexer_kl_loss_grad(
    q,
    k,
    w,
    sparse_indices,
    attn_softmax_l1_norm,
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

### sparse_lightning_indexer_kl_loss_grad

| Name | Input/Output | Description | Data Type | Dimension |
| :--- | :--- | :--- | :--- | :--- |
| q | required input | Query input of the Lightning Indexer branch. D: fixed to 128. | `float16`, `bfloat16` | `[B, S1, N1, D]` when `layout_q="BSND"`; `[T1, N1, D]` when `layout_q="TND"`. |
| k | required input | Key input of the Lightning Indexer branch. N2 supports only 1. D: fixed to 128. | Same as `q` | `[B, S2, N2, D]` when `layout_k="BSND"`; `[T2, N2, D]` when `layout_k="TND"`. |
| w | required input | Head weight of the Indexer logits. | `float32` | `[B, S1, N1]` when `layout_q="BSND"`; `[T1, N1]` when `layout_q="TND"`. |
| sparse_indices | required input | Top-k key indices corresponding to each query. Fill valid positions with key indices and invalid positions with `-1`. | `int32` | `[B, S1, N2, K]` when `layout_q="BSND"`; `[T1, N2, K]` when `layout_q="TND"`. |
| attn_softmax_l1_norm | required input | Target distribution `p` precomputed by the main Attention branch. It is recommended to set invalid top-k positions to 0. | `float32` | Same as `sparse_indices`. |
| cu_seqlens_q | optional input | Prefix sum of query sequence lengths in the TND use case. The first element is 0. | `int32` | `[B + 1]`. |
| cu_seqlens_k | optional input | Prefix sum of key sequence lengths in the TND use case. The first element is 0. | `int32` | `[B + 1]`. |
| seqused_q | optional input | Query length actually used in each batch. | `int32` | `[B]`. |
| seqused_k | optional input | Key length actually used in each batch. | `int32` | `[B]`. |
| cmp_residual_k | optional input | Residual length in the compressed key use case. Used to restore the key length before compression when `mask_mode=3` and `cmp_ratio!=1`. | `int32` | `[B]`. |
| metadata | optional input | Task splitting result generated by `sparse_lightning_indexer_kl_loss_grad_metadata`. Passing this parameter is recommended. | `int32` | `[64]`. |
| layout_q | optional attribute | Layout of `q`. | `str` | Supports `"BSND"` and `"TND"`. Default value: `"TND"`. |
| layout_k | optional attribute | Layout of `k`. | `str` | Supports `"BSND"` and `"TND"`. Default value: `"TND"`. |
| mask_mode | optional attribute | Sparse mask mode. | `int` | Currently supports `0` and `3`. Default value: `3`. |
| cmp_ratio | optional attribute | Key compression ratio. | `int` | Value range: `[1, 128]`. Default value: `1`. |

<!-- npu="A3,910b" id5 -->
- <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: The `seqused_q` and `seqused_k` parameters are not supported yet.
<!-- end id5 -->

### sparse_lightning_indexer_kl_loss_grad_metadata

| Name | Type | Type | Description | Data Type | Dimension (Shape) |
|--------|----------|-----------|------|----------|-------------|
| num_heads_q | int | Required | Number of heads of `q`, that is, `N1` of `q` in the main API. Currently, [1, 128] is supported. | int32 | - |
| num_heads_k | int | Required | Number of heads of `k`, that is, `N2` of `k` in the main API. Currently, only 1 is supported. | int32 | - |
| head_dim | int | Required | Last dimension `D` of `q` and `k`. Currently, only 128 is supported. | int32 | - |
| cu_seqlens_q | Tensor | Optional | Prefix sum of query sequence lengths in the TND use case. The first element is 0. The data format is ND. Non-contiguous tensors are supported. | int32 | (B+1, ) |
| cu_seqlens_k | Tensor | Optional | Prefix sum of key sequence lengths in the TND use case. The first element is 0. The data format is ND. Non-contiguous tensors are supported. | int32 | (B+1, ) |
| seqused_q | Tensor | Optional | Query length actually used in each batch. The data format is ND. Non-contiguous tensors are supported. | int32 | (B, ) |
| seqused_k | Tensor | Optional | Key length actually used in each batch. The data format is ND. Non-contiguous tensors are supported. | int32 | (B, ) |
| cmp_residual_k | Tensor | Optional | Residual length in the compressed key use case. The data format is ND. Non-contiguous tensors are supported. | int32 | (B, ) |
| batch_size | int | Optional | Batch size. Set this parameter explicitly in the BSND use case. In the TND use case, it can also be deduced from `cu_seqlens_q`. The default value is 0. | int32 | - |
| max_seqlen_q | int | Optional | Maximum query sequence length in a single batch. The default value is 0. | int32 | - |
| max_seqlen_k | int | Optional | Maximum key sequence length after compression in a single batch. The default value is 0. | int32 | - |
| topk | int | Optional | top-k size, that is, the last dimension `K` of `sparse_indices`. Currently, [1, 2048], 4096, and 8192 are supported. | int32 | - |
| layout_q | str | Optional | Layout on the `q` side. Valid values are `"BSND"` and `"TND"`. The default value is `"BSND"`. | string | - |
| layout_k | str | Optional | Layout on the `k` side. Valid values are `"BSND"` and `"TND"`. The default value is `"BSND"`. | string | - |
| mask_mode | int | Optional | sparse mask mode. 0 indicates No mask, and 3 indicates the rightDownCausal mode. The default value is 0. | int32 | - |
| cmp_ratio | int | Optional | key compression ratio. Currently, [1, 128] is supported. The default value is 1. | int32 | - |

<!-- npu="950" id6 -->
- <term>Ascend 950PR&950DT products</term>: `topk` supports only [1, 2048].
<!-- end id6 -->
<!-- npu="A3,910b" id7 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: `seqused_q`, `seqused_k`, and `cmp_residual_k` are not supported currently. `num_heads_q` supports only 8/16/32/64, and `topk` supports only 512/1024/2048/4096/8192.
<!-- end id7 -->

## Return Values

### sparse_lightning_indexer_kl_loss_grad

- **dq**: Gradient of `q`, with the same shape and data type as `q`.
- **dk**: Gradient of `k`, with the same shape and data type as `k`.
- **dw**: Gradient of `w`, with the same shape as `w` and data type `float32`.
- **softmax_out**: Softmax output of the Lightning Indexer branch, with the same shape as `attn_softmax_l1_norm` and data type `float32`.

### sparse_lightning_indexer_kl_loss_grad_metadata

| Name | Type | Type | Description | Data Type | Dimension (Shape) |
|--------|----------|-----------|------|----------|-------------|
| metadata | Tensor | Required | Index of the Batch, Head, and Q/K block partitions of the Attention computation task on each AI Core. The data format is ND. Non-contiguous tensors are not supported. | int32 | The shape is (64, ) |

## Constraints

- This API supports training use cases.
- This API supports the single-operator mode and the aclgraph mode.
- The data types of `q` and `k` must be the same. `float16` and `bfloat16` are supported.
- The data types of `w`, `attn_softmax_l1_norm`, `dw`, and `softmax_out` must be `float32`.
- The data types of `sparse_indices`, `cu_seqlens_q`, `cu_seqlens_k`, `seqused_q`, `seqused_k`, `cmp_residual_k`, and `metadata` must be `int32`.
- `layout_q` and `layout_k` support `BSND` and `TND`.
- If `layout_q="TND"`, `cu_seqlens_q` must be passed in; if `layout_k="TND"`, `cu_seqlens_k` must be passed in.
- Currently, only 1 is supported for `num_heads_k`.
- Valid positions in `sparse_indices` must be within the key sequence range of the current batch. Invalid positions are padded with `-1`.
- Invalid top-k positions in `attn_softmax_l1_norm` are recommended to be set to 0 and kept consistent with the valid positions in `sparse_indices`.
- `mask_mode` currently supports `0` and `3`.
- The value range of `cmp_ratio` is `[1, 128]`.
- With compressed keys and `mask_mode=3`, the key length before compression is computed as `compressed_k_len * cmp_ratio + cmp_residual_k[b]`. In the CP splitting use case, when each CP shard is called separately, pass the query length of the current shard to `q`, pass the compressed key prefix length corresponding to the shard to `k`, and pass the corresponding `cmp_residual_k` per the formula.

- **Specification Constraints**:

    | Specification Item | Specification | Specification Description |
    | :--- | :--- | :--- |
    | N2 | 1 | Currently, only N2=1 is supported. |
    | D | 128 | The last dimension of q and k must be the same. |
    | cmp_ratio | 1-128 | - |
    | mask_mode | 0, 3 | - |

  <!-- npu="A3,910b" id10 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:

    - B: 1-256 supported.
    - S1, S2: 1-8192 supported for S1, and 1-524288 supported for S2.
    - N1: 8, 16, 32, 64 supported.
    - K: 1-8192 supported.
  <!-- end id10 -->

  <!-- npu="950" id11 -->
  - <term>Ascend 950PR&950DT products</term>:

    - B: B>0.
    - S1, S2: S1>0, S2>0.
    - N1: 1-128 supported.
    - K: 1-2048 supported.
  <!-- end id11 -->

## Deterministic Computation

Deterministic computation is supported by default.

## Example

### TND Single-Batch, Compressed Key Use Case

```python
import torch
import torch_npu
from cann_ops_transformer.ops import sparse_lightning_indexer_kl_loss_grad
from cann_ops_transformer.ops import sparse_lightning_indexer_kl_loss_grad_metadata


def valid_k_count(local_q_idx, q_len, compressed_k_len, cmp_ratio, cmp_residual_k):
    pre_compress_k_len = compressed_k_len * cmp_ratio + cmp_residual_k
    numerator = pre_compress_k_len - q_len + local_q_idx + 1
    return numerator // cmp_ratio if numerator >= 0 else -((-numerator) // cmp_ratio)


def make_sparse_indices_and_target(q_len, compressed_k_len, topk, cmp_ratio, cmp_residual_k):
    sparse_indices = torch.full((q_len, 1, topk), -1, dtype=torch.int32)
    attn_softmax_l1_norm = torch.zeros((q_len, 1, topk), dtype=torch.float32)

    for q_idx in range(q_len):
        real_k = valid_k_count(q_idx, q_len, compressed_k_len, cmp_ratio, cmp_residual_k)
        real_k = max(0, min(real_k, compressed_k_len, topk))
        if real_k == 0:
            continue

        sparse_indices[q_idx, 0, :real_k] = torch.randperm(real_k, dtype=torch.int32)
        attn_softmax_l1_norm[q_idx, 0, :real_k] = 1.0 / real_k

    return sparse_indices, attn_softmax_l1_norm


torch_npu.npu.set_device(0)
device = torch.device("npu:0")

q_len = 512
compressed_k_len = 128
cmp_ratio = 4
cmp_residual_k = 0
batch_size = 1
num_heads_q = 64
num_heads_k = 1
head_dim = 128
topk = 512
layout = "TND"
mask_mode = 3
dtype = torch.bfloat16

q = torch.randn(q_len, num_heads_q, head_dim, dtype=dtype, device=device)
k = torch.randn(compressed_k_len, num_heads_k, head_dim, dtype=dtype, device=device)
w = torch.randn(q_len, num_heads_q, dtype=torch.float32, device=device) * (0.1 / 6.0)

sparse_indices_cpu, attn_l1_cpu = make_sparse_indices_and_target(
    q_len, compressed_k_len, topk, cmp_ratio, cmp_residual_k
)
sparse_indices = sparse_indices_cpu.to(device)
attn_softmax_l1_norm = attn_l1_cpu.to(device)

cu_seqlens_q = torch.tensor([0, q_len], dtype=torch.int32, device=device)
cu_seqlens_k = torch.tensor([0, compressed_k_len], dtype=torch.int32, device=device)
cmp_residual_k_tensor = torch.tensor([cmp_residual_k], dtype=torch.int32, device=device)

metadata = sparse_lightning_indexer_kl_loss_grad_metadata(
    num_heads_q,
    num_heads_k,
    head_dim,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    cmp_residual_k=cmp_residual_k_tensor,
    batch_size=batch_size,
    max_seqlen_q=q_len,
    max_seqlen_k=compressed_k_len,
    topk=topk,
    layout_q=layout,
    layout_k=layout,
    mask_mode=mask_mode,
    cmp_ratio=cmp_ratio,
)

dq, dk, dw, softmax_out = sparse_lightning_indexer_kl_loss_grad(
    q,
    k,
    w,
    sparse_indices,
    attn_softmax_l1_norm,
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
