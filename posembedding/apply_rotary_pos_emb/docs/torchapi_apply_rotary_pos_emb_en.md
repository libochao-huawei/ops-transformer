# apply_rotary_pos_emb

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:40:22.268Z -->

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
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- **Description**:

    To improve network performance, the rotary position embedding of query and key is fused into a single kernel call. The API returns the query and key output tensors after rotary position embedding, and the input tensors are not modified.

- **Formula**:

    When `rotary_mode` is `"half"`, `query` and `key` are split into two equal parts along the last dimension, and then rotary position embedding is performed:

    $$query\_q1 = query[..., : query.shape[-1] // 2]$$

    $$query\_q2 = query[..., query.shape[-1] // 2 :]$$

    $$query\_rotate = cat((-query\_q2, query\_q1), dim=-1)$$

    $$q\_embed = (query \cdot cos) + query\_rotate \cdot sin$$

    `key` is computed in the same way as `query`, producing $k\_embed$. When `rotary_mode` is `"interleave"`, adjacent elements form a pair and are rotated together. When `rotary_mode` is `"quarter"`, the last dimension is divided into four equal parts, and elements are paired and rotated.

- **NOTE**

  - When `layout` is `"TND"`, the input is a 3D tensor. For other `layout` values, the input is a 4D tensor. B (Batch) denotes the batch size, S (Seq-Length) the sequence length, N (Head-Num) the number of heads, D (Head-Dim) the hidden dimension size of each head, and T the merged axis of B and S.
  - `"BSH"` and `"BSND"` share the same underlying layout and are processed according to the dimension semantics of `"BSND"`.
  - When `rotary_mode` is `"half"`, this interface supports automatic differentiation and automatically calls [apply_rotary_pos_emb_grad](../../apply_rotary_pos_emb_grad/docs/torchapi_apply_rotary_pos_emb_grad_en.md) for backward propagation.

## Function Prototype

```python
cann_ops_transformer.apply_rotary_pos_emb(query, key, cos, sin, layout="BSND", rotary_mode="half") -> (Tensor, Tensor)
```

## Mandatory

| Name | Type | Type | Description | Data Type | Dimension (Shape) |
|--------|----------|-----------|------|----------|-------------|
| query | Tensor | Required | The first tensor for performing rotary position embedding. | bfloat16, float16, float32 | 3D when `layout` is "TND"; 4D for other `layout` values |
| key | Tensor | Required | The second tensor for performing rotary position embedding. | same as query | same as query |
| cos | Tensor | Required | Cosine tensor for rotary position embedding. The N dimension must be equal to 1. | same as query | same as query |
| sin | Tensor | Required | Sine tensor for rotary position embedding. The shape must be consistent with cos. | same as query | same as cos |
| layout | str | Optional | Layout format of input tensors. Supports "BSND", "BSH", "SBND", "BNSD", and "TND". Defaults to "BSND". | - | - |
| rotary_mode | str | Optional | Rotary encoding mode. Supports "half", "quarter", and "interleave". Defaults to "half". | - | - |

## Return Values

- **query_out** (`Tensor`): Output tensor of query after rotary position embedding. It is a newly allocated tensor, and its shape and data type are consistent with the input `query`. The input `query` is not modified.
- **key_out** (`Tensor`): Output tensor of key after rotary position embedding. It is a newly allocated tensor, and its shape and data type are consistent with the input `key`. The input `key` is not modified.

<!-- npu="950" id7 -->
> **Automatic differentiation note**: Supported only on <term>Ascend 950PR&950DT products</term>. When any of the inputs `query`, `key`, `cos`, and `sin` has `requires_grad=True` and `rotary_mode="half"`, this interface supports automatic differentiation. When `.backward()` is executed on the loss, `apply_rotary_pos_emb_grad` is automatically called to compute the four gradients of `query`, `key`, `cos`, and `sin`. Automatic differentiation is not supported when `rotary_mode` is `"quarter"` or `"interleave"`.
<!-- end id7 -->

## Constraints

- This interface supports use in both inference and training use cases.
- This interface supports calls in single-operator mode and graph mode.
- Empty tensors are not supported.
- The input tensors `query`, `key`, `cos`, and `sin` must have the same data type.
- The shapes of `cos` and `sin` must be the same, and the N dimension must be equal to 1.
- When `rotary_mode` is "half" or "interleave", the last dimension (D) of the input shape must be divisible by 2. When `rotary_mode` is "quarter", the last dimension (D) of the input shape must be divisible by 4.
<!-- npu="A3,910b,310p" id8 -->
- <term>Atlas inference products</term>, <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
  - `layout` only supports 4D Tensors of "BSND" and 3D Tensors of "TND".
  - When `layout` is "BSND", the first 2 dimensions (B, S) of the input shapes of `query`, `key`, `cos`, and `sin` must be equal; when `layout` is "TND", the first dimension (T) must be equal.
  - The last dimension (D) of the input shapes of `query` and `key` must be equal to 128 or 64, and the last dimension (D) of the input shapes of `cos` and `sin` must be equal to it.
<!-- end id8 -->
<!-- npu="950" id9 -->
- <term>Ascend 950PR&950DT products</term>:
  - In training use cases, automatic differentiation only supports `rotary_mode="half"`.
  - `layout` supports 4D Tensors of "BSND", "SBND", and "BNSD", and 3D Tensors of "TND".
  - For any `layout`, all dimensions of `query` and `key` except N must be the same.
  - The last dimension (D) of the input shapes of `query` and `key` must be equal and less than or equal to 1024; the last dimension (D) of the input shapes of `cos` and `sin` must be equal to and less than or equal to the last dimension (D) of the input shapes of `query` and `key`.
  - When `layout` is "BSND", the B dimension of `cos` and `sin` can be equal to 1 or consistent with the B dimension of `query`.
<!-- end id9 -->
<!-- npu="310p" id10 -->
- <term>Atlas inference products</term>: `bfloat16` is not supported.
<!-- end id10 -->

## Deterministic Computation

Defaults to a deterministic implementation.

## Example

- Call in single-operator mode

    ```python
    import torch
    import torch_npu
    from cann_ops_transformer.ops import apply_rotary_pos_emb

    torch_npu.npu.set_device(0)

    B = 1
    S = 64
    N = 8
    D = 128

    query = torch.randn(B, S, N, D, device="npu", dtype=torch.float16)
    key = torch.randn(B, S, N, D, device="npu", dtype=torch.float16)
    cos = torch.randn(B, S, 1, D, device="npu", dtype=torch.float16)
    sin = torch.randn(B, S, 1, D, device="npu", dtype=torch.float16)

    query_out, key_out = apply_rotary_pos_emb(
        query,
        key,
        cos,
        sin,
        layout="BSND",
        rotary_mode="half",
    )

    print(f"Output query shape: {query_out.shape}")
    print(f"Output key shape: {key_out.shape}")
    ```
<!-- npu="950" id11 -->
- Call in <term>Ascend 950PR&950DT products</term> training mode (automatic differentiation)

    ```python
    import torch
    import torch_npu
    from cann_ops_transformer.ops import apply_rotary_pos_emb

    torch_npu.npu.set_device(0)

    B, S, N, D = 1, 64, 8, 128

    query = torch.randn(B, S, N, D, device="npu", dtype=torch.float16, requires_grad=True)
    key = torch.randn(B, S, N, D, device="npu", dtype=torch.float16, requires_grad=True)
    cos = torch.randn(B, S, 1, D, device="npu", dtype=torch.float16, requires_grad=True)
    sin = torch.randn(B, S, 1, D, device="npu", dtype=torch.float16, requires_grad=True)

    # Forward: returns q_embed and k_embed, and automatically tracks the computation graph.
    query_out, key_out = apply_rotary_pos_emb(
        query,
        key,
        cos,
        sin,
        layout="BSND",
        rotary_mode="half",  # automatic differentiation only supports the half mode.
    )

    loss = query_out.sum() + key_out.sum()
    loss.backward()  # automatically calls apply_rotary_pos_emb_grad.

    print(query.grad.shape)  # gradient of query
    print(key.grad.shape)    # gradient of key
    print(cos.grad.shape)    # gradient of cos
    print(sin.grad.shape)    # gradient of sin
    ```
<!-- end id11 -->

- Graph mode call

    ```python
    import torch
    import torch_npu
    import torchair
    from cann_ops_transformer.ops import apply_rotary_pos_emb

    torch_npu.npu.set_device(0)

    B, S, N, D = 1, 64, 8, 128

    class ApplyRotaryPosEmbModel(torch.nn.Module):
        def forward(self, query, key, cos, sin):
            return apply_rotary_pos_emb(query, key, cos, sin, layout="BSND", rotary_mode="half")

    model = ApplyRotaryPosEmbModel().npu()
    npu_backend = torchair.get_npu_backend()
    model = torch.compile(model, backend=npu_backend, dynamic=False)

    query = torch.randn(B, S, N, D, device="npu", dtype=torch.float16)
    key = torch.randn(B, S, N, D, device="npu", dtype=torch.float16)
    cos = torch.randn(B, S, 1, D, device="npu", dtype=torch.float16)
    sin = torch.randn(B, S, 1, D, device="npu", dtype=torch.float16)

    query_out, key_out = model(query, key, cos, sin)
    ```
