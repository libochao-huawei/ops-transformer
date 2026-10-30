# dense_lightning_indexer_softmax_lse / dense_lightning_indexer_softmax_lse_metadata

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T08:42:22.068Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: supports
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

- Function:

    - `dense_lightning_indexer_softmax_lse`: Computes the Softmax Log-Sum-Exp (LSE) values corresponding to the Lightning Indexer for Dense scoring. The main computation process is as follows:

        1. Performs matrix multiplication between the input Query tensor `q` and the Key tensor `k` to obtain the relevance score matrix between each Query token and all Key tokens.
        2. After filtering out negative correlation signals via the ReLU activation function, multiply the result by the weighting coefficient `w` element-wise.
        3. Sum along the Query Head dimension (N1) to merge the weighted scores of all heads.
        4. Perform the Softmax Log-Sum-Exp operation along the Key sequence dimension (S2) and output the LSE value of each Query Token.

        This operator is commonly used together with the Dense scoring mode of `lightning_indexer` / `quant_lightning_indexer` for the subsequent Softmax normalization.

    - `dense_lightning_indexer_softmax_lse_metadata`: Used to generate core splitting metadata before the main operator. The output must be passed in as the `metadata` input of `dense_lightning_indexer_softmax_lse`.

- Formula:

    1. Calculate the correlation score matrix:

    $$
    Score = Q_{index} @ K_{index}^T
    $$

    2. ReLU activation and weighting:

    $$
    Weighted = ReLU(Score) \odot W
    $$

    3. Sum along the N1 dimension:

    $$
    X = \sum_{N1} Weighted
    $$

    4. Softmax LSE over the S2 dimension:

    $$
    LSE = \max_{S2}(X) + \log\left(\sum_{S2}\exp(X - \max_{S2}(X))\right)
    $$

    where $Q_{index} \in \mathbb{R}^{B \times S1 \times N1 \times D}$, $K_{index} \in \mathbb{R}^{B \times S2 \times N2 \times D}$, $W \in \mathbb{R}^{B \times S1 \times N1}$, and the output $LSE \in \mathbb{R}^{B \times N2 \times S1}$.

> [!NOTE]
>
> `cmp_residual_k` is an optional input of both `dense_lightning_indexer_softmax_lse` and `dense_lightning_indexer_softmax_lse_metadata`. When KV is compressed and `mask_mode=3`, this parameter is used to restore the key length before compression: `pre_compress_k_len = compressed_k_len * cmp_ratio + cmp_residual_k[b]`, thereby determining the valid causal range.

## Function Prototype

```python
from cann_ops_transformer.ops import dense_lightning_indexer_softmax_lse_metadata

dense_lightning_indexer_softmax_lse_metadata(
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
from cann_ops_transformer.ops import dense_lightning_indexer_softmax_lse

dense_lightning_indexer_softmax_lse(
    q,
    k,
    w,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    metadata=None,
    layout_q="BSND",
    layout_k="BSND",
    mask_mode=0,
    cmp_ratio=1
) -> Tensor
```

## Parameters

>**NOTE:**<br>
>
>- Dimension meanings of the q, k, and w parameters: B (Batch Size) indicates the batch size of input samples, S1 indicates the sequence length of the q input samples, S2 indicates the sequence length of the k input samples, N1 indicates the number of heads of q, N2 indicates the number of heads of k, and D (Head Dim) indicates the dimension of the attention head. D in parameter q is equal to D in parameter k, and the value currently only supports 128.

### dense_lightning_indexer_softmax_lse

| Name | Type | Input/Output | Description | Data Type | Dimension |
| :--- | :--- | :--- | :--- | :--- | :--- |
| q | Tensor | Required input | Input $Q_{index}$ in the formula. Supports empty tensor. The data format is ND. | `float16`, `bfloat16` | `[B, S1, N1, D]` when `layout_q="BSND"`; `[T1, N1, D]` when `layout_q="TND"`. |
| k | Tensor | Required input | Input $K_{index}$ in the formula. Supports empty tensor. The data format is ND. N2 supports only 1. D is fixed to 128. | Same as `q` | `[B, S2, N2, D]` when `layout_k="BSND"`; `[T2, N2, D]` when `layout_k="TND"`. |
| w | Tensor | Required input | Weighting coefficient $W$ in the formula. Supports empty tensor. The data format is ND. | `float32` | `[B, S1, N1]` when `layout_q="BSND"`; `[T1, N1]` when `layout_q="TND"`. |
| cu_seqlens_q | Tensor | Optional input | Accumulation of the numbers of valid tokens of q in the current batch and preceding batches. Must be passed in only when `layout_q="TND"`. The first value is fixed to 0. The data format is ND. | `int32` | `[B+1]`. |
| cu_seqlens_k | Tensor | Optional input | Accumulation of the numbers of valid tokens of k in the current batch and preceding batches. Must be passed in only when `layout_k="TND"`. The first value is fixed to 0. The data format is ND. | `int32` | `[B+1]`. |
| seqused_q | Tensor | Optional input | Actual used length of q in different batches. The data format is ND. | `int32` | `[B]`. |
| seqused_k | Tensor | Optional input | Actual used length of k in different batches. The data format is ND. | `int32` | `[B]`. |
| cmp_residual_k | Tensor | Optional input | Indicates the remainder of the number of tokens of k before compression divided by cmp_ratio. Must be passed in when `mask_mode=3` and `cmp_ratio≠1`. The data format is ND. | `int32` | `[B]`. |
| metadata | Tensor | Optional input | Core splitting information generated by `dense_lightning_indexer_softmax_lse_metadata`, including the number of cores used, block size, and the start point of the data processed by each core. The data format is ND. | `int32` | `[1024]`. |
| layout_q | str | Optional attribute | Indicates the data layout format of input q. Supports `BSND` and `TND`. The default value is `BSND`. | `str` | - |
| layout_k | str | Optional attribute | Indicates the data layout format of input k. Supports `BSND` and `TND`. The default value is `BSND`. `layout_q` and `layout_k` must be the same. | `str` | - |
| mask_mode | int | Optional attribute | Indicates the mask mode. `0` represents the defaultMask mode, and `3` represents the rightDownCausal mode. The default value is `0`. | `int` | - |
| cmp_ratio | int | Optional attribute | Indicates the compression ratio of k. Value range: `[1, 128]`. The default value is `1`, indicating no compression. | `int` | - |

### dense_lightning_indexer_softmax_lse_metadata

| Name | Type | Type | Description | Data Type | Dimension (Shape) |
|--------|----------|-----------|------|----------|-------------|
| num_heads_q | int | Required | Number of heads of `q`, which is `N1` of `q` in the main API. Currently supports [1, 128]. | int32 | - |
| num_heads_k | int | Required | Number of heads of `k`, which is `N2` of `k` in the main API. Currently only supports 1. | int32 | - |
| head_dim | int | Required | Last dimension `D` of `q` and `k`. Currently only supports 128. | int32 | - |
| cu_seqlens_q | Tensor | Optional | Prefix-sum sequence length of queries in the TND use case. The first element is 0. The data format is ND, and it supports non-contiguous tensors. | int32 | (B+1, ) |
| cu_seqlens_k | Tensor | Optional | Prefix-sum sequence length of keys in the TND use case. The first element is 0. The data format is ND, and it supports non-contiguous tensors. | int32 | (B+1, ) |
| seqused_q | Tensor | Optional | Indicates the actual query length used in each batch. The data format is ND, and it supports non-contiguous tensors. | int32 | (B, ) |
| seqused_k | Tensor | Optional | Indicates the actual key length used in each batch. The data format is ND, and it supports non-contiguous tensors. | int32 | (B, ) |
| cmp_residual_k | Tensor | Optional | Residual length in the compressed key use case. The data format is ND, and it supports non-contiguous tensors. | int32 | (B, ) |
| batch_size | int | Optional | Batch size. It is recommended to explicitly pass in this parameter in BSND use cases. In TND use cases, it can also be deduced from `cu_seqlens_q`. The default value is 0. | int32 | - |
| max_seqlen_q | int | Optional | Maximum query sequence length in a single batch. The default value is 0. | int32 | - |
| max_seqlen_k | int | Optional | Maximum compressed key sequence length in a single batch. The default value is 0. | int32 | - |
| layout_q | str | Optional | Layout of `q`. Supports `"BSND"` and `"TND"`. The default value is `"BSND"`. | string | - |
| layout_k | str | Optional | Layout of `k`. Supports `"BSND"` and `"TND"`. The default value is `"BSND"`. | string | - |
| mask_mode | int | Optional | Sparse mask mode. 0 indicates No mask, and 3 indicates the rightDownCausal mode. The default value is 0. | int32 | - |
| cmp_ratio | int | Optional | Key compression ratio. Currently supports [1, 128]. The default value is 1. | int32 | - |

## Return Values

### dense_lightning_indexer_softmax_lse

| Name | Type | Type | Description | Data Type | Dimension (Shape) |
|--------|----------|-----------|------|----------|-------------|
| softmax_lse | Tensor | Required | Softmax Log-Sum-Exp calculation result. | float32 | `[B, N2, S1]` when `layout_q="BSND"`; `[N2, T1]` when `layout_q="TND"`. |

### dense_lightning_indexer_softmax_lse_metadata

| Name | Type | Type | Description | Data Type | Dimension (Shape) |
|--------|----------|-----------|------|----------|-------------|
| metadata | Tensor | Required | Indexes of the Batch, Head, and Q and K chunks of the Attention computation task for each AIcore. The data format is ND. Non-contiguous tensors are not supported. | int32 | The shape is (1024, ) |

## Constraints

- This API can be used in both inference and training use cases.
- This API can be called in single-operator mode and aclgraph graph mode.
- All input tensors support empty tensors by default.
- The data types of `q` and `k` must be the same, and `float16` and `bfloat16` are supported.
- The data types of `w` and `softmax_lse` should be `float32`.
- B (Batch) indicates the batch size of input samples and supports generalization.
- S1 and S2 indicate the sequence length and support generalization. For ultra-long sequence use cases, excessive computation may cause NPU memory overflow.
- N1 of parameter q supports 1-128, and N2 of k is fixed to 1.
- Dimension D is fixed to 128.
- `layout_q` and `layout_k` support `BSND` and `TND`, and the two must be the same.
- When `layout_q="TND"`, `cu_seqlens_q` must be passed in; when `layout_k="TND"`, `cu_seqlens_k` must be passed in.
- `mask_mode` currently supports `0` and `3`. For details about the meaning of `mask_mode`, see [sparse_mode Introduction](../../../docs/en/context/sparse_mode_introduction.md).
- The value range of `cmp_ratio` is `[1, 128]`.
- The values of `cu_seqlens_q` and `cu_seqlens_k` must be the accumulation of the numbers of valid tokens in the current batch and preceding batches. The value of each element must be greater than or equal to that of the previous element.
- The values of `seqused_q` and `seqused_k` must not exceed the actual sequence length of each batch.
  - In BSND use cases: seqused_q ≤ S1 and seqused_k ≤ S2.
  - In TND use cases: seqused_q ≤ cu_seqlens_q[i+1] - cu_seqlens_q[i] and seqused_k ≤ cu_seqlens_k[i+1] - cu_seqlens_k[i].
- Parameter `cmp_residual_k` must satisfy cmp_residual_k[i] < cmp_ratio.
- When cmp_ratio > 1 and mask_mode = 3, cmp_residual_k must be passed in; otherwise, it is not required.
- x1 is INT8, x2 is INT8, x1Scale is FLOAT32, and x2Scale is FLOAT32.
  - The seqused_q, seqused_k, and metadata functions are currently not supported. It is not recommended to pass in these parameters.
- <term>Ascend 950PR&950DT products</term>:
  - When layout_q is BSND, passing in cu_seqlens_q is not supported; when layout_k is BSND, passing in cu_seqlens_k is not supported.
  - When layout_q is TND, cu_seqlens_q must be passed in. If seqused_q is also passed in, ensure that the query length of each batch specified by seqused_q does not exceed the query length of each batch calculated based on cu_seqlens_q. When the seqused_q of a batch is smaller than the actual length, the TND Padding function is enabled and the excess part of the batch output is filled with invalid values.
  - When cmp_ratio > 1 and mask_mode = 3, cmp_residual_k must be passed in.

## Deterministic Computation

- Defaults to a deterministic implementation.

## Example

- Call in single-operator mode:

  ```python
  import torch
  import torch_npu
  import numpy as np
  from cann_ops_transformer.ops import dense_lightning_indexer_softmax_lse
  from cann_ops_transformer.ops import dense_lightning_indexer_softmax_lse_metadata

  B = 2
  S1 = 4096
  S2 = 1024
  N1 = 32
  N2 = 1
  D = 128

  torch.npu.set_device(0)
  device = torch.device("npu:0")

  # Construct the input.
  q = torch.randn(B, S1, N1, D, dtype=torch.float16, device=device)
  k = torch.randn(B, S2, N2, D, dtype=torch.float16, device=device)
  w = torch.randn(B, S1, N1, dtype=torch.float32, device=device)

  # Generate metadata.
  metadata = dense_lightning_indexer_softmax_lse_metadata(
      num_heads_q=N1,
      num_heads_k=N2,
      head_dim=D,
      batch_size=B,
      max_seqlen_q=S1,
      max_seqlen_k=S2,
      layout_q="BSND",
      layout_k="BSND",
      mask_mode=0,
      cmp_ratio=1
  )

  # Execute the operator.
  lse = dense_lightning_indexer_softmax_lse(
      q,
      k,
      w,
      metadata=metadata,
      layout_q="BSND",
      layout_k="BSND",
      mask_mode=0,
      cmp_ratio=1
  )
  print(f"lse shape: {lse.shape}")  # (B, 1, S1)
  ```

- Call in aclgraph graph mode:

  ```python
  import torch
  import torch_npu
  import numpy as np
  import torchair
  from cann_ops_transformer.ops import dense_lightning_indexer_softmax_lse
  from cann_ops_transformer.ops import dense_lightning_indexer_softmax_lse_metadata

  B = 2
  S1 = 4096
  S2 = 1024
  N1 = 32
  N2 = 1
  D = 128

  torch.npu.set_device(0)
  device = torch.device("npu:0")

  q = torch.randn(B, S1, N1, D, dtype=torch.float16, device=device)
  k = torch.randn(B, S2, N2, D, dtype=torch.float16, device=device)
  w = torch.randn(B, S1, N1, dtype=torch.float32, device=device)

  class DenseLightningIndexerLSENetwork(torch.nn.Module):
      def __init__(self):
          super(DenseLightningIndexerLSENetwork, self).__init__()

      def forward(self, q, k, w):
          metadata = torch.ops.cann_ops_transformer.dense_lightning_indexer_softmax_lse_metadata(
              num_heads_q=N1,
              num_heads_k=N2,
              head_dim=D,
              batch_size=B,
              max_seqlen_q=S1,
              max_seqlen_k=S2,
              layout_q="BSND",
              layout_k="BSND",
              mask_mode=0,
              cmp_ratio=1
          )
          return torch.ops.cann_ops_transformer.dense_lightning_indexer_softmax_lse(
              q,
              k,
              w,
              metadata=metadata,
              layout_q="BSND",
              layout_k="BSND",
              mask_mode=0,
              cmp_ratio=1
          )

  from torchair.configs.compiler_config import CompilerConfig
  config = CompilerConfig()
  config.mode = "reduce-overhead"
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  torch._dynamo.reset()
  npu_mode = torch.compile(
      DenseLightningIndexerLSENetwork(), fullgraph=True, backend=npu_backend, dynamic=False
  )
  lse = npu_mode(q, k, w)
  print(f"lse shape: {lse.shape}")
  ```
