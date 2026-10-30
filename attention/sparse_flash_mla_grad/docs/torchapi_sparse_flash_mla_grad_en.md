# sparse_flash_mla_grad

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T13:27:40.121Z -->

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

  - `sparse_flash_mla_grad`: Computes the backward output of attention in the `SparseFlashMla` training use case. Supports Sliding Window Attention, Compressed Attention, and Sparse Compressed Attention.
  - `sparse_flash_mla_grad_metadata`: Generates a task list. The list contains the start and end points of the Attention computation task on each AI Core, including Batch, Head, and Q and K chunk indices, for use by the subsequent `sparse_flash_mla_grad` operator.

- **Formula**:

    Stage 1: Select between input ori_kv and cmp_kv based on different cmp_ratio use cases

  - When cmp_ratio = 1 (SWA):

    $$
    selectedKv\text{ }=\text{ }orikv
    $$

  - When cmp_ratio = 4 (SCFA):

    $$
    selectedKv\text{ }=concat(oriKv, \text{ }Gather \left( cmpkv,topkIndices \left[ i \left]  \left)) ,\text{ }0\text{ } < =i < \text{ }selectBlockCount\right. \right. \right. \right.
    $$

  - else (CFA):

    $$
    selectedKv\text{ }=concat(oriKv, \text{ }cmpkv)
    $$

    Stage 2: Compute P, dP, dS

    $$
    P = SimpleSoftmax(Mask(Q \text{ }@\text{ } selectedKv^{{T}} \cdot \text{ } scale), lse)
    $$

    $$
    dP = dO \text{ }@\text{ } selectedKv^{{T}}
    $$

    $$
    dS = P \times (dP\text{ } -\text{ } SoftmaxGrad(dO, O))
    $$

    Stage 3: Compute dQ, dKV, dSinks

    $$
    dQ = dS \text{ } @ \text{ } selectedKv \text{ }  \cdot \text{ } scale
    $$

    $$
    dKV = dS^{{T}} \text{ } @ \text{ } Q \text{ } \cdot \text{ } scale + P^{{T}}  @ \text{ } dO
    $$

    $$
    dSinks = ReduceSum(-P \text{ }\times\text{ } dP \text{ }\times\text{ } SimpleSoftmax(sinks, lse), dim=-1)
    $$

## Function Prototype

```python
cann_ops_transformer.sparse_flash_mla_grad_metadata(
    num_heads_q,
    num_heads_kv,
    head_dim,
    *,
    cu_seqlens_q=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    cmp_residual_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    batch_size=None,
    max_seqlen_q=None,
    max_seqlen_ori_kv=None,
    max_seqlen_cmp_kv=None,
    ori_topk=None,
    cmp_topk=None,
    cmp_ratio=None,
    ori_mask_mode=0,
    cmp_mask_mode=0,
    ori_win_left=-1,
    ori_win_right=-1,
    layout_q="BSND",
    layout_kv="BSND",
    has_ori_kv=True,
    has_cmp_kv=True
) -> Tensor
```

```python
cann_ops_transformer.sparse_flash_mla_grad(
    q,
    dout,
    attn_out,
    softmax_lse,
    *,
    ori_kv=None,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    cu_seqlens_q=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    cmp_residual_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    sinks=None, metadata=None,
    softmax_scale=None,
    cmp_ratio=None,
    ori_mask_mode=0,
    cmp_mask_mode=0,
    ori_win_left=-1,
    ori_win_right=-1,
    layout_q="BSND",
    layout_kv="BSND"
) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)
```

## Parameters

### sparse_flash_mla_grad

- **q** (`Tensor`): Required parameter, corresponding to $Q$ in the formula. When `layout_q`="BSND", the shape is `[B, S1, N1, D]`; when `layout_q`="TND", the shape is `[T1, N1, D]`. B: supports generalization; S1: supports generalization; D: 512; T1: B × S1. Support Non-contiguous, the data format supports ND, and the data types support `bfloat16` and `float16`.

- **dout** (`Tensor`): Required parameter, gradient of the attention forward output matrix, corresponding to $dO$ in the formula. The data type and shape keep consistent with q. Support Non-contiguous, the data format supports ND, and the data types support `bfloat16` and `float16`.

- **attn_out** (`Tensor`): Required parameter, attention forward output matrix, corresponding to $O$ in the formula. The data type and shape keep consistent with q. Support Non-contiguous, the data format supports ND, and the data types support `bfloat16` and `float16`.

- **softmax_lse** (`Tensor`): Required parameter, output lse of the attention forward computation. When `layout_q`="BSND", the shape is `[B, N2, S1, G]`; when `layout_q`="TND", the shape is `[N2, T1, G]`. B: keep consistent with the B of q; N2: 1; S1: keep consistent with the S1 of q; G: N1/N2; T1: B × S1. Support Non-contiguous, the data format supports ND, and the data types support `float32`.

- **ori_kv** (`Tensor`): Optional parameter, corresponding to $oriKv$ in the formula. When `layout_kv`="BSND", the shape is `[B, S2, N2, D]`; when `layout_kv`="TND", the shape is `[T2, N2, D]`. B: keep consistent with the B of q; S2: supports generalization; N2: 1; D: 512; T2: B × S2. Support Non-contiguous, the data format supports ND, and the data types support `bfloat16` and `float16`.

- **cmp_kv** (`Tensor`): Optional parameter, corresponding to $cmpkv$ in the formula. When `layout_kv`="BSND", the shape is `[B, S3, N2, D]`; when `layout_kv`="TND", the shape is `[T3, N2, D]`. B: keep consistent with the B of q; S3: supports generalization; N2: 1; D: 512; T3: B × S3. When None is passed, computation is performed in the SWA use case. Support Non-contiguous, the data format supports ND, and the data types support `bfloat16` and `float16`.

- **ori_sparse_indices** (`Tensor`): Optional parameter, topk indices corresponding to the oriKv part. When `layout_q`="BSND", the shape is `[B, S1, N2, K1]`; when `layout_q`="TND", the shape is `[T1, N2, K1]`. B: keep consistent with the B of q; S1: keep consistent with the S1 of q; N2: 1; K1: supports generalization; T1: B × S1. Support Non-contiguous, the data format supports ND, and the data types support `int32`.

- **cmp_sparse_indices** (`Tensor`): Optional parameter, corresponding to $topkIndices$ in the formula. When `layout_q`="BSND", the shape is `[B, S1, N2, K2]`; when `layout_q`="TND", the shape is `[T1, N2, K2]`. B: keep consistent with the B of q; S1: keep consistent with the S1 of q; N2: 1; K2: supports generalization; T1: B × S1. If cmp_kv is not None, computation is performed in the SCFA use case when cmp_sparse_indices is not None, and in the CFA use case when it is None; if cmp_kv is None, cmp_sparse_indices can only be None, and computation is performed in the SWA use case. Support Non-contiguous, the data format supports ND, and the data types support `int32`.

- **cu_seqlens_q** (`Tensor`): Optional parameter, in the form of the accumulated sum of the number of valid tokens of q in each batch. Must pass when `layout_q`="TND". The shape is `[B+1]`, and the accumulated sum keeps consistent with T1. Support Non-contiguous, the data format supports ND, and the data types support `int32`.

- **cu_seqlens_ori_kv** (`Tensor`): Optional parameter, the accumulated sum of the number of valid ori_kv tokens in each Batch. Must pass when `layout_kv`="TND". Shape is `[B+1]`, and the accumulated sum must be consistent with T2. Supports non-contiguous tensors, ND data format, and `int32` data type.

- **cu_seqlens_cmp_kv** (`Tensor`): Optional parameter, the accumulated sum of the number of valid cmp_kv tokens in each Batch. Must pass when `layout_kv`="TND". Shape is `[B+1]`, and the accumulated sum must be consistent with T3. Supports non-contiguous tensors, ND data format, and `int32` data type.

- **seqused_q** (`Tensor`): Optional parameter, represents the number of query tokens that actually participate in computation in different batches. Shape is `[B]`. Supports non-contiguous tensors, ND data format, and `int32` data type.

- **seqused_ori_kv** (`Tensor`): Optional parameter, represents the number of ori_kv tokens that actually participate in computation in different batches. Shape is `[B]`. Supports non-contiguous tensors, ND data format, and `int32` data type.

- **seqused_cmp_kv** (`Tensor`): Optional parameter, represents the number of cmp_kv tokens that actually participate in computation in different batches. Shape is `[B]`. Supports non-contiguous tensors, ND data format, and `int32` data type.

- **cmp_residual_kv** (`Tensor`): Optional parameter, represents the remainder of S2 // cmpRatio for each batch. Shape is `[B]`. Must pass when cmp_kv is not empty and cmp_mask_mode=3. Supports non-contiguous tensors, ND data format, and `int32` data type.

- **ori_topk_length** (`Tensor`): Optional parameter, indicates the actually optional topk length of ori_kv corresponding to each row of query. Shape is `[B, S1, N2]` (BSND) or `[T1, N2]` (TND). Must pass when ori_mask_mode=0 and ori_sparse_indices is not None, and the value must be accurate. Supports non-contiguous tensors, ND data format, and `int32` data type.

- **cmp_topk_length** (`Tensor`): Optional parameter, indicates the actually optional topk length of cmp_kv corresponding to each row of query. Shape is `[B, S1, N2]` (BSND) or `[T1, N2]` (TND). Must pass when cmp_mask_mode=0 and cmp_sparse_indices is not None, and the value must be accurate. Supports non-contiguous tensors, ND data format, and `int32` data type.

- **sinks** (`Tensor`): Optional parameter, attention sink tensor. Shape is `[N1]`. Supports non-contiguous tensors, ND data format, and `float32` data type.

- **metadata** (`Tensor`): Optional parameter, represents the output result of the aicpu operator with tiling sinking. Supports non-contiguous tensors, ND data format, and `int32` data type.

- **softmax_scale** (`double`): Optional parameter. Represents the scaling factor. Data type `double` is supported. Default value: 1.0 / sqrt(D).

- **cmp_ratio** (`int`): Optional parameter. Represents the compression ratio. Value range: 1-128. Data type `int` is supported. Default value: 1.

- **ori_mask_mode** (`int`): Optional parameter. Mask mode for the computation between q and ori_kv. Mode 0: no mask operation; mode 3: rightDownCausal; mode 4: band (sliding window, starting from the bottom-right corner). Data type `int` is supported.

- **cmp_mask_mode** (`int`): Optional parameter. Mask mode for the computation between q and cmp_kv. Mode 0: no mask operation; mode 3: rightDownCausal. Data type `int` is supported.

- **ori_win_left** (`int`): Optional parameter. Number of past tokens computed by q in the computation between q and ori_kv. Currently only supports the value 127. Data type `int` is supported.

- **ori_win_right** (`int`): Optional parameter. Number of future tokens computed by q in the computation between q and ori_kv. Currently only supports the value 0. Data type `int` is supported.

- **layout_q** (`str`): Optional parameter. Data layout format of q. Supports "BSND" and "TND". Data type `str` is supported.

- **layout_kv** (`str`): Optional parameter. Data layout format of ori_kv and cmp_kv. Supports "BSND" and "TND". Currently must keep consistent with layout_q. Data type `str` is supported.

<!-- npu="A3,910b" id5 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: The seqused_q, seqused_ori_kv, seqused_cmp_kv, ori_topk_length, cmp_topk_lengt, and metadata parameters are not supported yet.
<!-- end id5 -->

### sparse_flash_mla_grad_metadata

| Name | Type | Required/Optional | Description | Data Type | Dimension (Shape) |
|--------|----------|-----------|------|----------|-------------|
| num_heads_q | int | Required | Indicates the number of heads of $Q$ in the formula (that is, N1). Currently, the supported value range is [1, 128]. | int32 | - |
| num_heads_kv | int | Required | Indicates the number of heads of $oriKv$ or $cmpkv$ in the formula (that is, N2). Currently, only 1 is supported. | int32 | - |
| head_dim | int | Required | Indicates the head dimension (that is, D). Currently, only 512 is supported. | int32 | - |
| cu_seqlens_q | Tensor | Optional | Represents the accumulated sum form of the number of valid q tokens in each batch. This parameter must pass when `layout_q` is TND. Supports non-contiguous tensors. Supports the ND data format. The accumulated sum keeps consistent with T1. | int32 | (B+1, ) |
| cu_seqlens_ori_kv | Tensor | Optional | Represents the accumulated sum form of the number of valid ori_kv tokens in each batch. This parameter must pass when `layout_kv` is TND. Supports non-contiguous tensors. Supports the ND data format. The accumulated sum keeps consistent with T2. | int32 | (B+1, ) |
| cu_seqlens_cmp_kv | Tensor | Optional | Represents the accumulated sum form of the number of valid cmp_kv tokens in each batch. This parameter must pass when `layout_kv` is TND. Supports non-contiguous tensors. Supports the ND data format. The accumulated sum keeps consistent with T3. | int32 | (B+1, ) |
| seqused_q | Tensor | Optional | Represents the number of tokens of query that actually participate in the operation in different batches. | int32 | (B, ) |
| seqused_ori_kv | Tensor | Optional | Represents the number of tokens of ori_kv that actually participate in the operation in different batches. | int32 | (B, ) |
| seqused_cmp_kv | Tensor | Optional | Represents the number of tokens of cmp_kv that actually participate in the operation in different batches. | int32 | (B, ) |
| cmp_residual_kv | Tensor | Optional | Represents the remainder after actual ori_s2 // cmpRatio in each batch. Supports non-contiguous tensors. Supports the ND data format. Must pass when cmp_kv is not empty and cmp_mask_mode=3. | int32 | (B, ) |
| ori_topk_length | Tensor | Optional | Indicates the actually optional topk length of ori_kv corresponding to each row of query. | int32 | (B, S1, N2) or (T1, N2) |
| cmp_topk_length | Tensor | Optional | Indicates the actually optional topk length of cmp_kv corresponding to each row of query. | int32 | (B, S1, N2) or (T1, N2) |
| batch_size | int | Optional | Indicates the batch size of the input samples (that is, B). Default value is 0. | int32 | - |
| max_seqlen_q | int | Optional | Indicates the maximum sequence length of the input q in the TND use case. Default value is 0. | int32 | - |
| max_seqlen_ori_kv | int | Optional | Indicates the maximum sequence length of the input ori_kv in the TND use case. Default value is 0. | int32 | - |
| max_seqlen_cmp_kv | int | Optional | Indicates the maximum sequence length of the input cmp_kv in the TND use case. Default value is 0. | int32 | - |
| ori_topk | int | Optional | Indicates the topk length of ori_kv. Default value is 0. | int32 | - |
| cmp_topk | int | Optional | Indicates the topk length of cmp_kv. Default value is 0. | int32 | - |
| cmp_ratio | int | Optional | Represents the compression ratio. Value range: 1-128. Default value is 1. | int32 | - |
| ori_mask_mode | int | Optional | Indicates the mask mode for computation between q and ori_kv. 0 indicates No mask, 3 indicates rightDownCausal mode, 4 indicates sliding window mode. Default value is 0. | int32 | - |
| cmp_mask_mode | int | Optional | Indicates the mask mode for computation between q and cmp_kv. 0 indicates No mask, 3 indicates rightDownCausal mode. Default value is 0. | int32 | - |
| ori_win_left | int | Optional | Indicates the number of past tokens computed by q in the computation between q and ori_kv. -1 indicates infinity. Default value is -1. | int32 | - |
| ori_win_right | int | Optional | Indicates the number of future tokens computed by q in the computation between q and ori_kv. -1 indicates infinity. Default value is -1. | int32 | - |
| layout_q | str | Optional | Indicates the data layout format of q. Supports "BSND" and "TND". Default value is BSND. | string | - |
| layout_kv | str | Optional | Indicates the data layout format of ori_kv and cmp_kv. Supports "BSND" and "TND". Currently must keep consistent with layout_q. Default value is BSND. | string | - |
| has_ori_kv | bool | Optional | Indicates whether ori_kv is passed in. Default value is true. | bool | - |
| has_cmp_kv | bool | Optional | Indicates whether cmp_kv is passed in. Default value is true. | bool | - |

<!-- npu="A3,910b" id6 -->
- <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: seqused_q, seqused_ori_kv, seqused_cmp_kv, ori_topk_length, and cmp_topk_length are not supported. ori_mask_mode only supports 4, cmp_mask_mode only supports 3, ori_win_left only supports 127, and ori_win_right only supports 0.
<!-- end id6 -->

## Return Values

### sparse_flash_mla_grad

- **dq** (`Tensor`): Corresponds to $dQ$ in the formula. Supports non-contiguous. The data format supports ND. The data type and shape keep consistent with input q.

- **dori_kv** (`Tensor`): Optional output. Indicates the gradient of input ori_kv. Supports non-contiguous. The data format supports ND. The data type and shape keep consistent with input ori_kv.

- **dcmp_kv** (`Tensor`): Optional output. Indicates the gradient of input cmp_kv. Supports non-contiguous. The data format supports ND. The data type and shape keep consistent with input cmp_kv. If cmp_kv is None, dcmp_kv is also None.

- **dsinks** (`Tensor`): Optional output. Indicates the gradient of input sinks. Supports non-contiguous. The data format supports ND. The data type supports `float32`. The shape keeps consistent with input sinks.

- **ori_softmax_l1norm** (`Tensor`): Optional output. Indicates the L1Norm result of the softmax computed from q and ori_kv. The formula is reduceG(softmax)/G. The data type is `float32`. The shape is `[B,S1,N2,K1]` when `layout_q` is BSND, and `[T1,N2,K1]` when `layout_q` is TND. This output is not empty when ori_sparse_indices is not None, and is None in other use cases.

- **cmp_softmax_l1norm** (`Tensor`): Optional output. Indicates the L1Norm result of the softmax computed from q and cmp_kv. The formula is reduceG(softmax)/G. This output is not empty when cmp_sparse_indices is not None, and is None in other use cases.

### sparse_flash_mla_grad_metadata

| Name | Type | Required/Optional | Description | Data Type | Dimension (Shape) |
|--------|----------|-----------|------|----------|-------------|
| metadata | Tensor | Required | Indices of the Batch, Head, and Q/K blocks of FlashAttention computation tasks on each cube core, and indices of FlashDecode reduction tasks on each vector core. The data format is ND. Non-contiguous tensors are not supported. | int32 | (1024, ) |

## Constraints

- This API can be used in training use cases.
- This API supports the single-operator mode and aclgraph mode.
- The data types of q, dout, attn_out, ori_kv, and cmp_kv must be kept consistent.
- Handling when the input is empty: the API returns directly if q is an empty tensor.
- The usage of cmp_kv and cmp_sparse_indices in each use case is described as follows:
  - SWA use case: requires cmp_kv == None && cmp_sparse_indices == None
  - SCFA use case: requires cmp_kv != None && cmp_sparse_indices != None
  - CFA use case: requires cmp_kv != None && cmp_sparse_indices == None

- **Mask Mode Support**:

    | Mode | Meaning | Note |
    | :--- | :--- | :--- |
    | 0 | No mask operation | Supported |
    | 3 | rightDownCausal | Supported |
    | 4 | band (sliding window, starting from the bottom-right corner) | Supported by oriMaskMode |

- **Specification Constraints**:

    | Specification Item | Specification | Specification Description |
    | :--- | :--- | :--- |
    | B | Support Generalization | - |
    | S1 and S2 | Support Generalization | Supports S1 and S2 with different lengths. |
    | N1 | 1-128 | - |
    | N2 | 1 | Currently only supports N2=1. |
    | D | 512 | The last dimension of q, ori_kv, and cmp_kv must keep consistent. |
    | layout_q/kv | BSND / TND, must be the same | - |
    | cmp_ratio | 1-128 | - |

  - Support situation for passing None to `ori_kv`/`cmp_kv`:
    <!-- npu="A3,910b" id7 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: Not supported.
    <!-- end id7 -->
    <!-- npu="950" id10 -->
    - <term>Ascend 950PR&950DT products</term>: Supported.
    <!-- end id10 -->
  - Support situation for `ori_sparse_indices`:
    <!-- npu="A3,910b" id11 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: Not supported.
    <!-- end id11 -->
    <!-- npu="950" id12 -->
    - <term>Ascend 950PR&950DT products</term>: Supported.
    <!-- end id12 -->
  - Support situation of `seqused_q`:
    <!-- npu="A3,910b" id13 -->
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: Not supported (None must be passed).
    <!-- end id13 -->
    <!-- npu="950" id14 -->
    - <term>Ascend 950PR&950DT products</term>: Supported.
    <!-- end id14 -->
  - Support situation of `ori_topk_length`/`cmp_topk_length`:
    <!-- npu="A3,910b" id15 -->
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: Not supported (None must be passed).
    <!-- end id15 -->
    <!-- npu="950" id16 -->
    - <term>Ascend 950PR&950DT products</term>: Supported.
    <!-- end id16 -->
  - Support situation of passing None to `sinks`:
    <!-- npu="A3,910b" id17 -->
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: Not supported.
    <!-- end id17 -->
    <!-- npu="950" id18 -->
    - <term>Ascend 950PR&950DT products</term>: Supported.
    <!-- end id18 -->
  - Support situation of `metadata`:
    <!-- npu="A3,910b" id19 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: Only supports passing None.
    <!-- end id19 -->
    <!-- npu="950" id20 -->
    - <term>Ascend 950PR&950DT products</term>: Must be passed.
    <!-- end id20 -->
  - Support situation of `ori_mask_mode`:
    <!-- npu="A3,910b" id21 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: Only supports mode 4.
    <!-- end id21 -->
    <!-- npu="950" id22 -->
    - <term>Ascend 950PR&950DT products</term>: Supports modes 0, 3, and 4.
    <!-- end id22 -->
  - Support situation of `cmp_mask_mode`:
    <!-- npu="A3,910b" id23 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: Only supports mode 3.
    <!-- end id23 -->
    <!-- npu="950" id24 -->
    - <term>Ascend 950PR&950DT products</term>: Supports modes 0 and 3.
    <!-- end id24 -->

## Deterministic Computation

Deterministic computation is supported by default.

## Example

- Single-operator mode call

    ```python
    import math
    import random
    import numpy as np

    import torch
    import torch_npu
    import cann_ops_transformer

    S1 = 256
    S2 = 1024
    cmp_ratio = 4
    actual_seq_q = [S1]
    actual_seq_ori_kv = [S2]
    actual_seq_cmp_kv = [S2 // cmp_ratio]
    cmp_residual = [i % cmp_ratio for i in actual_seq_ori_kv]

    T1 = sum(actual_seq_q)
    T2 = sum(actual_seq_ori_kv)
    T3 = sum(actual_seq_cmp_kv)

    B = 1
    N1 = 64
    N2 = 1
    D = 512
    K = 512
    scaleValue = 1 / math.sqrt(D)
    dtype = torch.float16
    input_layout = "TND"

    q_shape = tuple((T1, N1, D))
    ori_kv_shape = tuple((T2, N2, D))
    cmp_kv_shape = tuple((T3, N2, D))
    out_shape = tuple((T1, N1, D))
    lse_shape = tuple((N2, T1, N1 // N2))

    cu_seq_qlen = [0] + [sum(actual_seq_q[:x+1]) for x in range(len(actual_seq_q))]
    cu_seq_ori_kvlen = [0] + [sum(actual_seq_ori_kv[:x+1]) for x in range(len(actual_seq_ori_kv))]
    cu_seq_cmp_kvlen = [0] + [sum(actual_seq_cmp_kv[:x+1]) for x in range(len(actual_seq_cmp_kv))]

    q = (torch.rand(q_shape).to(dtype)) * 2
    ori_kv = (torch.rand(ori_kv_shape).to(dtype)) * 2
    cmp_kv = (torch.rand(cmp_kv_shape).to(dtype)) * 2 if cmp_ratio != 1 else None
    dy = (torch.rand(out_shape).to(dtype)) * 2
    out = (torch.rand(out_shape).to(dtype)) * 2 # Use the forward output attn_out in actual use cases.
    sinks = (torch.rand(N1).to(torch.float32)) * (128)
    lse = (torch.rand(lse_shape).to(torch.float32)) # Use the forward output lse in actual use cases.

    if cmp_ratio == 4:
        cmp_sparse_indices = torch.ones([T1, N2, K], dtype=torch.int32) * -1
        accum_t = 0
        for i in range(B):
            cur_s1 = actual_seq_q[i]
            cur_ori_s2 = actual_seq_cmp_kv[i] * cmp_ratio + cmp_residual[i]
            delta_s = cur_ori_s2 - cur_s1
            for s1_idx in range(cur_s1):
                threshold = delta_s + s1_idx + 1 if delta_s + s1_idx + 1 >= 0 else 0
                max_cmp_s2 = threshold // cmp_ratio
                cur_s2_idxs = np.arange(max_cmp_s2)
                if max_cmp_s2 < K:
                    cmp_sparse_indices[accum_t, :, :max_cmp_s2] = torch.tensor(np.random.choice(cur_s2_idxs, size=max_cmp_s2, replace=False))
                else:
                    cmp_sparse_indices[accum_t, :, :] = torch.tensor(np.random.choice(cur_s2_idxs, size=K, replace=False))
                accum_t += 1
    else:
        cmp_sparse_indices = None

    cmp_kv_npu = cmp_kv.npu() if cmp_kv != None else None
    cmp_sparse_indices_npu = cmp_sparse_indices.npu() if cmp_sparse_indices != None else None
    cu_seq_qlen = None if input_layout != "TND" else torch.tensor(cu_seq_qlen).to(torch.int32).npu()
    cu_seq_ori_kvlen = None if input_layout != "TND" else torch.tensor(cu_seq_ori_kvlen).to(torch.int32).npu()
    cu_seq_cmp_kvlen = None if input_layout != "TND" else torch.tensor(cu_seq_cmp_kvlen).to(torch.int32).npu()
    cmp_residual_kv = torch.tensor(cmp_residual).to(torch.int32).npu() if cmp_residual != None else None

    metadata = cann_ops_transformer.ops.sparse_flash_mla_grad_metadata(N1, N2, D)
    npu_out = cann_ops_transformer.ops.sparse_flash_mla_grad(
            q.npu(), dy.npu(), out.npu(), lse.npu(),
            ori_kv=ori_kv.npu(), cmp_kv=cmp_kv_npu,
            cmp_sparse_indices=cmp_sparse_indices_npu,
            cu_seqlens_q=cu_seq_qlen,
            cu_seqlens_ori_kv=cu_seq_ori_kvlen,
            cu_seqlens_cmp_kv=cu_seq_cmp_kvlen,
            cmp_residual_kv=cmp_residual_kv,
            sinks=sinks.npu(),
            softmax_scale=scaleValue,
            cmp_ratio=cmp_ratio,
            layout_q=input_layout,
            layout_kv=input_layout,
            ori_win_left=127,
            ori_win_right=0,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            metadata=metadata
        )
    ```
