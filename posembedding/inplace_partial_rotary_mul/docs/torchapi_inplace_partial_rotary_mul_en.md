# inplace_partial_rotary_mul

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:44:01.388Z -->

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

- **Description**:

    Executes inplace computation of single-path rotary position encoding and directly modifies the input tensor `x` without generating a new output tensor. This interface supports specifying a partial range on the last dimension of the input tensor through the `partial_slice` parameter. Rotary position encoding is executed only on the data in this range, and values at other positions remain unchanged.

- **Formula**:

    In interleave mode (`rotary_mode` is `"interleave"`), when `partial_slice=[start, end]`, the rotated partial tensor is:

    $$x_{slice} = x[..., start:end]$$

    The computation process is as follows:

    $$x_1 = x_{slice}[..., ::2]$$

    $$x_2 = x_{slice}[..., 1::2]$$

    $$x_{rotate} = \text{cat}(-x_2, x_1)$$

    $$x_{out} = x_{slice} \cdot \cos + x_{rotate} \cdot \sin$$

    Finally, $x_{out}$ is written back to `x[..., start:end]` in place. Here, $x$ indicates the parameter `x`, $\cos$ indicates the parameter `r1`, and $\sin$ indicates the parameter `r2`. When `start` equals `end`, rotary position encoding is not executed and `x` remains unchanged. The backward operator `inplace_partial_rotary_mul_backward` also supports use cases with empty tensors and zero-length slices (a no-op is executed. For details, see [Constraints](#constraints)).

- **NOTE**

  - The input `x` uses the BSND dimension format, where B (Batch) indicates the batch size, S (Seq-Length) indicates the sequence length, N (Head-Num) indicates the number of heads, and D (Head-Dim) indicates the hidden dimension size of each head.
  - `partial_slice` applies to the last dimension (D) of the input `x`, and the value range is the left-closed and right-open interval `[start, end)`. In the Python API, the default value of `partial_slice` is `None`, which is internally processed as `[0, 0]`.

## Function Prototype

```python
cann_ops_transformer.inplace_partial_rotary_mul(x, r1, r2, *, rotary_mode="interleave", partial_slice=None) -> None
```

## Mandatory

| Name | Type | Type | Description | Data Type | Dimension (Shape) |
|--------|----------|-----------|------|----------|-------------|
| x | Tensor | Required | Tensor on which rotary position encoding is executed, corresponding to x in the formula. In Inplace mode, the computation result is directly written back to this tensor. | bfloat16, float16, float32 | (B, S, N, D) |
| r1 | Tensor | Required | Position encoding tensor, corresponding to the cos component in the formula. | bfloat16, float16, float32 | 4D, must satisfy broadcasting with x |
| r2 | Tensor | Required | Position encoding tensor, corresponding to the sin component in the formula. The data type of `r2` must be the same as that of `r1`. | bfloat16, float16, float32 | 4D, must satisfy broadcasting with x |
| rotary_mode | str | Optional | Rotary mode. Currently, only "interleave" is supported. The default value is "interleave". | - | - |
| partial_slice | List[int] | Optional | Slice range [start, end) for partial rotation, which applies to the last dimension (D) of x. The default value is None, which is internally processed as [0, 0]. | - | - |

## Return Values

This API has no return value (`None`). The computation result is written in place to the input tensor `x`. After the computation, the shape and data type of `x` remain unchanged. Data outside the range specified by `partial_slice` retains its original values.

> **Automatic differentiation**: When `x.requires_grad` is True, calling `.backward()` on the loss automatically triggers `inplace_partial_rotary_mul_backward`, without manually calling the backward operator. The gradients of `r1` (cos) and `r2` (sin) are **not computed** and are always None. The backward operator `inplace_partial_rotary_mul_backward` supports empty tensors and a slice length of zero (executes no-op), so automatic differentiation works as expected.

## Constraints

- This API supports training and inference use cases.
- This API supports calls in single-operator mode and graph mode.
- Non-contiguous tensors are not supported.
- The last dimension D of `x` must not exceed 1024, and D must be a multiple of 2.
- `partial_slice` must contain two integers that satisfy `start >= 0`, `end >= 0`, `end <= D`, and `end - start >= 0`.
- The slice length of `partial_slice` (that is, `end - start`) must be a multiple of 2. When `end` equals `start` (the slice length is zero), both forward and backward computation execute no-op and return directly.
- The last dimension of `r1` and `r2` must be the same and must equal the slice length of `partial_slice` (that is, `end - start`).
- The shapes of `r1` and `r2` must satisfy the broadcasting relationship with `x[..., start:end]`. The following constraints apply:
  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>: Currently, the shapes of `r1` and `r2` support only BSND, B1ND, B11D, and 111D layouts.
  <!-- end id7 -->
  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: Currently, the shapes of `r1` and `r2` support only BS1D and B11D layouts.
  <!-- end id8 -->
- Each dimension value of `x` must be greater than 0. If `partial_slice` is not an empty slice, the dimension values of `r1` and `r2` that participate in the computation must be greater than 0.
- **Automatic differentiation constraint**: Only the gradient of `x` is computed. The gradients of `r1` and `r2` are not computed and are always None. Because the input and output of the operator are at the same address, `x` cannot be a leaf tensor with `requires_grad=True`.

## Deterministic Computation

Defaults to a deterministic implementation.

## Calling Instructions

- Single-operator mode call

    ```python
    import torch
    import torch_npu
    from cann_ops_transformer.ops import inplace_partial_rotary_mul

    torch_npu.npu.set_device(0)

    B = 2
    S = 32
    N = 8
    D = 128
    slice_start = 0
    slice_end = 64

    x = torch.randn(B, S, N, D, device="npu", dtype=torch.float16)
    r1 = torch.randn(B, S, 1, slice_end - slice_start, device="npu", dtype=torch.float16)
    r2 = torch.randn(B, S, 1, slice_end - slice_start, device="npu", dtype=torch.float16)

    inplace_partial_rotary_mul(
        x,
        r1,
        r2,
        rotary_mode="interleave",
        partial_slice=[slice_start, slice_end],
    )
    ```

- Training mode call (automatic differentiation)

    ```python
    import torch
    import torch_npu
    from cann_ops_transformer.ops import inplace_partial_rotary_mul

    torch_npu.npu.set_device(0)

    B, S, N, D = 2, 32, 8, 128
    slice_start, slice_end = 0, 64

    x = torch.randn(B, S, N, D, device="npu", dtype=torch.float16, requires_grad=True)
    r1 = torch.randn(B, S, 1, slice_end - slice_start, device="npu", dtype=torch.float16)
    r2 = torch.randn(B, S, 1, slice_end - slice_start, device="npu", dtype=torch.float16)
    y = x * 1.0
    y.retain_grad()

    # Forward: automatically tracks the computation graph (y is modified in-place; no return value needs to be received)
    inplace_partial_rotary_mul(
        y, r1, r2,
        rotary_mode="interleave",
        partial_slice=[slice_start, slice_end],
    )

    # Continue the forward computation.
    loss = y.sum()
    loss.backward()  # Automatically calls inplace_partial_rotary_mul_backward.

    print(y.grad.shape)
    print(x.grad.shape)
    # r1.grad and r2.grad are always None (cos/sin do not compute gradients).
    ```

- Graph mode call

    ```python
    import torch
    import torch_npu
    import torchair
    from cann_ops_transformer.ops import inplace_partial_rotary_mul

    torch_npu.npu.set_device(0)

    B = 2
    S = 32
    N = 8
    D = 128
    slice_start = 0
    slice_end = 64

    class InplacePartialRotaryMulModel(torch.nn.Module):
        def forward(self, x, r1, r2):
            inplace_partial_rotary_mul(
                x,
                r1,
                r2,
                rotary_mode="interleave",
                partial_slice=[slice_start, slice_end],
            )
            return x

    model = InplacePartialRotaryMulModel().npu()
    npu_backend = torchair.get_npu_backend()
    model = torch.compile(model, backend=npu_backend, dynamic=False)

    x = torch.randn(B, S, N, D, device="npu", dtype=torch.float16)
    r1 = torch.randn(B, S, 1, slice_end - slice_start, device="npu", dtype=torch.float16)
    r2 = torch.randn(B, S, 1, slice_end - slice_start, device="npu", dtype=torch.float16)

    output = model(x, r1, r2)
    ```
