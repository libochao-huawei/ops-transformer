# flash\_attn\_grad

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T09:18:46.944Z -->

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

  `flash_attn_grad` is a `cann_ops_transformer` extension API based on `TorchNPU`. It computes the backward result of Flash Attention. Based on the forward computation results `softmax_lse` and `attn_out` and the upstream gradient `dout`, the API computes the gradients `dq`, `dk`, and `dv` of Query, Key, and Value.

  Before calling this API, call `flash_attn_metadata` to generate the `metadata` required for backward computation. When generating the metadata, set `is_grad_enabled=True`, and the shape, layout, mask, and sequence length related parameters of the two APIs must be consistent.

- **Formula**:

  The forward computation is:

  $$
  S=scale\cdot QK^T
  $$

  $$
  P_{ij}=\exp(S_{ij}-softmax\_lse_i)
  $$

  $$
  Y=PV
  $$

  The backward computation is:

  $$
  dV=P^TdY
  $$

  $$
  dP=dYV^T
  $$

  $$
  sfmg=rowsum(dY\odot Y)
  $$

  $$
  dS=P\odot(dP-sfmg)
  $$

  $$
  dQ=scale\cdot(dS\cdot K)
  $$

  $$
  dK=scale\cdot(dS^T\cdot Q)
  $$

  Here, $Q$, $K$, and $V$ correspond to inputs `q`, `k`, and `v` respectively, $Y$ corresponds to `attn_out`, and $dY$ corresponds to `dout`. When `softmax_scale` is not 0, $scale$ is set to `softmax_scale`; when it is 0, $scale=1/\sqrt{D}$.

> [!NOTE]
>
> B indicates the batch size. S1 indicates the sequence length of Query. S2 indicates the sequence length of Key/Value. N1 indicates the number of Query heads. N2 indicates the number of Key/Value heads. D indicates the head dim of Query/Key. Dv indicates the head dim of Value/output. T1 indicates the cumulative sum of Query sequence lengths across batches. T2 indicates the cumulative sum of Key/Value sequence lengths across batches. In GQA use cases, N1 is an integer multiple of N2.

## Function Prototype

```python
cann_ops_transformer.flash_attn_grad(
    q,
    k,
    v,
    dout,
    attn_out,
    softmax_lse,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    sinks=None,
    attn_mask=None,
    metadata=None,
    softmax_scale=0.0,
    mask_mode=0,
    win_left=-1,
    win_right=-1,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    layout_q="BSND",
    layout_kv="BSND",
    layout_out="BSND"
) -> (Tensor, Tensor, Tensor)
```

The currently registered PyTorch schema does not use the `*` separator. Therefore, parameters from `cu_seqlens_q` onward can be passed in either by position or by keyword. It is recommended to pass in optional parameters by keyword.

## Enumerations

`mask_mode` supports passing in an `IntEnum` enumeration or the corresponding int value in the Python API. The enumeration is defined in `cann_ops_transformer.ops.attention.flash_attn_grad`:

### mask_mode Enumeration

| Enumeration Name | Value | Meaning |
| :--- | :---: | :--- |
| `NO_MASK` | 0 | Full computation mode (default value). |
| `CAUSAL` | 3 | Causal mode. |
| `SLIDING_WINDOW` | 4 | Sliding Window mode. |

> [!NOTE]
>
> The enumeration is `IntEnum` and can be passed directly to the underlying operator as an int. The API also supports passing in the string corresponding to the enumeration name (case-insensitive) or the int value.

## Parameters

| Name | Type | Required/Optional | Description | Data Type | Data Format | Dimension |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| q | Tensor | Required | Q in the formula. | bfloat16/float16 | ND | <ul><li>BSND: (B, S1, N1, D)</li><li>BNSD: (B, N1, S1, D)</li><li>TND: (T1, N1, D)</li></ul> |
| k | Tensor | Required | K in the formula. | bfloat16/float16 | ND | <ul><li>BSND: (B, S2, N2, D)</li><li>BNSD: (B, N2, S2, D)</li><li>TND: (T2, N2, D)</li></ul> |
| v | Tensor | Required | V in the formula. | bfloat16/float16 | ND | <ul><li>BSND: (B, S2, N2, Dv)</li><li>BNSD: (B, N2, S2, Dv)</li><li>TND: (T2, N2, Dv)</li></ul> |
| dout | Tensor | Required | dY in the formula, the upstream gradient of the forward output. | bfloat16/float16 | ND | <ul><li>BSND: (B, S1, N1, Dv)</li><li>BNSD: (B, N1, S1, Dv)</li><li>TND: (T1, N1, Dv)</li></ul> |
| attn_out | Tensor | Required | Y in the formula, i.e., the attention output returned by the forward API. | bfloat16/float16 | ND | Same shape as `dout` |
| softmax_lse | Tensor | Required | Log-sum-exp result returned by the forward API when `return_softmax_lse=True`. | float32 | ND | <ul><li>BSND/BNSD: (B, N1, S1)</li><li>TND: (N1, T1)</li></ul> |
| cu_seqlens_q | Tensor | Optional | Cumulative sequence lengths of Q under the TND layout. The first element must be 0, and the last element is equal to T1. Must be passed when `layout_q` is TND; not supported when `layout_q` is not TND. Default value is None. | int32 | ND | (B+1,) |
| cu_seqlens_kv | Tensor | Optional | Cumulative sequence lengths of KV under the TND layout. The first element must be 0, and the last element is equal to T2. Must be passed when `layout_kv` is TND; not supported when `layout_kv` is not TND. Default value is None. | int32 | ND | (B+1,) |
| seqused_q | Tensor | Optional | Actual sequence lengths of Q used by each batch. Default value is None. | int32 | ND | (B,) |
| seqused_kv | Tensor | Optional | Actual sequence lengths of KV used by each batch. Default value is None. | int32 | ND | (B,) |
| sinks | Tensor | Optional | Sink parameter, used to improve the numerical stability of self-attention computation. Default value is None. | float32 | ND | (N1,) |
| attn_mask | Tensor | Optional | Mask matrix. Default value is None. | int8 | ND | (2048, 2048) |
| metadata | Tensor | Optional | FAG task splitting data generated by `flash_attn_metadata`. Optional in the schema, but must be passed in actual calls. | int32 | ND | Shape dynamically computed based on batch size and N2 |
| softmax_scale | float | Optional | Softmax scaling coefficient. Default value is 0.0, indicating use of $1/\sqrt{D}$ | float32 | - | - |
| mask_mode | int/MaskMode | Optional | Mask mode. Supports passing in the enumeration or the corresponding int value. For the enumeration definition, see "mask_mode Enumeration". Default value is 0. | int32 | - | - |
| win_left | int | Optional | Left window of the window mask. The value must be greater than or equal to -1, and -1 indicates positive infinity. Default value is -1. | int32 | - | - |
| win_right | int | Optional | Right window of the window mask. The value must be greater than or equal to -1, and -1 indicates positive infinity. Default value is -1. | int32 | - | - |
| max_seqlen_q | int | Optional | Maximum sequence length of Q. Must be greater than or equal to -1. In BSND/BNSD use cases, keep the default value -1, and S1 is inferred from the input shape. | int32 | - | - |
| max_seqlen_kv | int | Optional | Maximum sequence length of KV. Must be greater than or equal to -1. In BSND/BNSD use cases, keep the default value -1, and S2 is inferred from the input shape. | int32 | - | - |
| layout_q | string | Optional | Layout of q. Currently supports BSND, BNSD, and TND. Default value is "BSND". | string | - | - |
| layout_kv | string | Optional | Layout of k and v. Must be the same as `layout_q`. Default value is "BSND". | string | - | - |
| layout_out | string | Optional | Layout of `dout` and `attn_out`. Must be the same as `layout_q`. Default value is "BSND". | string | - | - |

> [!NOTE]
>
> `q`, `k`, `v`, `dout`, and `attn_out`, as well as the outputs `dq`, `dk`, and `dv`, support float16 and bfloat16. The data types must be consistent.

## Return Value Description

| Name | Type | Required/Optional | Description | Data Type | Data Format | Dimension |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| dq | Tensor | Required | dQ in the formula, gradient of Query. | bfloat16/float16 | ND | Same shape as `q` |
| dk | Tensor | Required | dK in the formula, gradient of Key. | bfloat16/float16 | ND | Same shape as `k` |
| dv | Tensor | Required | dV in the formula, gradient of Value. | bfloat16/float16 | ND | Same shape as `v` |

## Constraints

- The API supports the following combinations:
  - The data type is float16 or bfloat16.
  - `layout_q`, `layout_kv`, and `layout_out` support BSND, BNSD, and TND, and must be the same.
  - Dense Attention with `mask_mode=0`, `attn_mask=None`, `win_left=-1`, and `win_right=-1`.
  - Under the TND layout, `cu_seqlens_q` and `cu_seqlens_kv` must be passed in; passing them in is not supported under non-TND layouts.
- The data types of `q`, `k`, `v`, `dout`, and `attn_out` must be consistent.
- B, S1, S2, N1, N2, D, and Dv must be positive, and the value range of B is (0, 65536).
- N1 must be divisible by N2. MHA (N1=N2) and GQA (N1>N2) are supported.
- The head dim of both Query and Key is D; the head dim of Value, `dout`, and `attn_out` is Dv, satisfying `0 < Dv <= D <= 192`.
- In a joint call with `flash_attn` in the current repository, the forward API additionally requires D=Dv and D to be 64, 128, or 256. Considering the constraint of this API that D does not exceed 192, the joint call currently supports D=Dv=64 or 128.
- `softmax_lse` must be float32, with shape (B, N1, S1) for BSND/BNSD cases or (N1, T1) for TND cases.
- The data format of all inputs is ND. The operator has `AutoContiguous` registered. When a non-contiguous tensor is passed in, the framework converts it to a contiguous tensor.
- `metadata` must be generated by `flash_attn_metadata` with `is_grad_enabled=True` set. When generating the metadata and calling this API, the parameters related to N1, N2, D, B, S1, S2, layout, and mask must be consistent; otherwise, the behavior is undefined.
- The metadata generated with `is_grad_enabled=True` contains both forward and backward task splitting data. The forward `flash_attn` and the backward `flash_attn_grad` can use the same metadata, so no separate generation is required.
- `softmax_lse` and `attn_out` must come from a forward call with the same configuration as this backward computation. In particular, `softmax_scale` and layout must be consistent.
- Currently, only the single-operator mode is supported.

## Related APIs

Before calling `flash_attn_grad`, generate the backward task splitting data using `flash_attn_metadata`.

```python
cann_ops_transformer.flash_attn_metadata(
    num_heads_q,
    num_heads_kv,
    head_dim,
    *,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    batch_size=None,
    max_seqlen_q=None,
    max_seqlen_kv=None,
    mask_mode=None,
    win_left=None,
    win_right=None,
    layout_q=None,
    layout_kv=None,
    layout_out=None,
    is_grad_enabled=False
) -> Tensor
```

The parameters directly related to the backward API are as follows:

| Name | Type | Required/Optional | Description | Data Type | Data Format | Dimension |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| num_heads_q | int | Required | Number of Query heads (N1) | int32 | - | - |
| num_heads_kv | int | Required | Number of Key/Value heads (N2) | int32 | - | - |
| head_dim | int | Required | Head dim of Query/Key (D) | int32 | - | - |
| cu_seqlens_q | Tensor | Optional | Cumulative sequence length of Q under the TND layout. Must be passed when `layout_q` is TND; not supported when `layout_q` is not TND. The default value is None | int32 | ND | (B+1,) |
| cu_seqlens_kv | Tensor | Optional | Cumulative sequence length of KV under the TND layout. Must be passed when `layout_kv` is TND; not supported when `layout_kv` is not TND. The default value is None | int32 | ND | (B+1,) |
| seqused_q | Tensor | Optional | Actual sequence length of Q used in each batch. The default value is None | int32 | ND | (B,) |
| seqused_kv | Tensor | Optional | Actual sequence length of KV used in each batch. The default value is None | int32 | ND | (B,) |
| batch_size | int | Optional | Batch size. The actual B must be passed in BSND/BNSD scenarios. Value range: (0, 65536) | int32 | - | - |
| max_seqlen_q | int | Optional | Maximum sequence length of Q. The actual S1 must be passed in BSND/BNSD scenarios and must be greater than 0 | int32 | - | - |
| max_seqlen_kv | int | Optional | Maximum sequence length of KV. The actual S2 must be passed in BSND/BNSD scenarios and must be greater than 0 | int32 | - | - |
| mask_mode | int/MaskMode | Optional | Must be consistent with `flash_attn_grad` | int32 | - | - |
| win_left | int | Optional | Must be consistent with `flash_attn_grad` | int32 | - | - |
| win_right | int | Optional | Must be consistent with `flash_attn_grad` | int32 | - | - |
| layout_q | string | Optional | Must be consistent with `flash_attn_grad` | string | - | - |
| layout_kv | string | Optional | Must be consistent with `flash_attn_grad` | string | - | - |
| layout_out | string | Optional | Must be consistent with `flash_attn_grad` | string | - | - |
| is_grad_enabled | bool | Optional | Whether to generate the metadata required by the backward operator. Must be set to True before calling `flash_attn_grad`. The default value is False | bool | - | - |

The returned `metadata` is a 1D tensor of int32 and ND format. The length is dynamically calculated based on the batch size and N2.

## Example

- Joint call example of `flash_attn_metadata`, `flash_attn`, and `flash_attn_grad` (BSND)

  `flash_attn_metadata` with `is_grad_enabled=True` generates task splitting data for both forward and backward propagation. The metadata can be used by both the forward and backward operators, so there is no need to generate them separately.

    ```python
    import math
    import torch
    import torch_npu
    import cann_ops_transformer

    torch_npu.npu.set_device(0)

    dtype = torch.float16
    B = 2
    S1 = 128
    S2 = 128
    N1 = 8
    N2 = 2
    D = 128
    Dv = 128
    scale = 1.0 / math.sqrt(D)

    q = torch.randn(B, S1, N1, D, dtype=dtype, device="npu")
    k = torch.randn(B, S2, N2, D, dtype=dtype, device="npu")
    v = torch.randn(B, S2, N2, Dv, dtype=dtype, device="npu")

    metadata = cann_ops_transformer.flash_attn_metadata(
        N1,
        N2,
        D,
        batch_size=B,
        max_seqlen_q=S1,
        max_seqlen_kv=S2,
        mask_mode=0,
        win_left=-1,
        win_right=-1,
        layout_q="BSND",
        layout_kv="BSND",
        layout_out="BSND",
        is_grad_enabled=True,
    )

    attn_out, softmax_lse = cann_ops_transformer.flash_attn(
        q,
        k,
        v,
        metadata=metadata,
        softmax_scale=scale,
        mask_mode=0,
        win_left=-1,
        win_right=-1,
        max_seqlen_q=S1,
        max_seqlen_kv=S2,
        layout_q="BSND",
        layout_kv="BSND",
        layout_out="BSND",
        return_softmax_lse=True,
    )
    dout = torch.randn_like(attn_out)

    dq, dk, dv = cann_ops_transformer.flash_attn_grad(
        q,
        k,
        v,
        dout,
        attn_out,
        softmax_lse,
        metadata=metadata,
        softmax_scale=scale,
        mask_mode=0,
        win_left=-1,
        win_right=-1,
        max_seqlen_q=S1,
        max_seqlen_kv=S2,
        layout_q="BSND",
        layout_kv="BSND",
        layout_out="BSND",
    )
    torch_npu.npu.synchronize()

    assert dq.shape == q.shape
    assert dk.shape == k.shape
    assert dv.shape == v.shape
    ```
