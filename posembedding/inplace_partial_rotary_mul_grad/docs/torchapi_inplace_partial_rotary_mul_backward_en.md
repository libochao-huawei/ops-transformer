# inplace_partial_rotary_mul_backward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:45:27.455Z -->

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

    Performs the backward computation of the partial rotary position embedding `inplace_partial_rotary_mul`. This API computes the rotary position embedding gradient for the slice `[start, end)` region along the D dimension of the input `grad_output`, and the computation result is written back to `grad_output` inplace. Elements outside the slice remain unchanged.

- **Formula**:

    In interleave mode (`rotary_mode` is `"interleave"`), assume `partial_slice=[start, end]`. The partial tensor to be computed is:

    $$dy' = grad\_output[..., start:end]$$

    $$cos' = r1[..., start:end]$$

    $$sin' = r2[..., start:end]$$

    Let:

    $$dy1', dy2' = dy'[..., ::2], dy'[..., 1::2]$$

    $$cos1', cos2' = cos'[..., ::2], cos'[..., 1::2]$$

    $$sin1', sin2' = sin'[..., ::2], sin'[..., 1::2]$$

    Then the gradient is computed as:

    $$dx' = stack((cos1' \cdot dy1' + sin2' \cdot dy2', \ cos2' \cdot dy2' - sin1' \cdot dy1'), \ dim=-1).reshape(dy'.shape)$$

    The result of $dx'$ is written back inplace to the `[start, end)` range of `grad_output`.

- **Note**:

  - Currently, this operator only implements the `rotary_mode="interleave"` mode. The half (0), quarter (2), and interleave-half (3) modes are not yet supported.
  - The input `grad_output` uses the BSND format, where B (Batch) is the batch size, S (Seq-Length) is the sequence length, N (Head-Num) is the number of heads, and D (Head-Dim) is the hidden dimension size of each head.
  - `partial_slice` acts on the last dimension (D) of the input `grad_output`, with a value range in the left-closed, right-open interval `[start, end)`.

## Function Prototype

```python
cann_ops_transformer.inplace_partial_rotary_mul_backward(grad_output, r1, r2, *, rotary_mode="interleave", partial_slice=None) -> None
```

## Mandatory

| Name | Type | Type | Description | Data type | Dimension (Shape) |
|--------|----------|-----------|------|----------|-------------|
| grad_output | Tensor | Required | Gradient of the forward output y, corresponding to the gradient of `x` of the forward operator `inplace_partial_rotary_mul`. In inplace mode, the computation result is directly written back to this tensor, becoming the gradient `dx` of the forward input `x`. | bfloat16, float16, float32 | (B, S, N, D) |
| r1 | Tensor | Required | Position encoding tensor, corresponding to the cos component in the forward computation. | bfloat16, float16, float32 | 4D, must satisfy the broadcast relationship with grad_output |
| r2 | Tensor | Required | Position encoding tensor, corresponding to the sin component in the forward computation. The data type of `r2` must be the same as that of `r1`. | bfloat16, float16, float32 | 4D, must satisfy the broadcast relationship with grad_output |
| rotary_mode | str | Optional | Rotation mode. Currently only `"interleave"` is supported. The default value is `"interleave"`. | - | - |
| partial_slice | List[int] | Optional | Slice range `[start, end)` for partial rotation, acting on the last dimension (D) of `grad_output`. The default value is `None`. Internally, the API processes it as `[0, 0]`. | - | - |

## Return Values

This API has no return value. The computation result is directly written back to the input tensor `grad_output`. The shape and data type of `grad_output` remain unchanged after computation. Data outside the range specified by `partial_slice` keeps its original value, and data within the specified range is replaced with the gradient `dx` of the forward input `x`.

## Constraints

- This API can be called in single-operator mode or TorchAir graph mode.
- This operator supports only contiguous tensors; non-contiguous tensors are not supported.
- **The current version of this operator supports only interleave mode (`rotary_mode="interleave"`)**. The half (0), quarter (2), and interleave-half (3) modes are not yet implemented.
- **Empty input support**: When the slice length is 0 (start equals end in `partial_slice`, for example `[0, 0]`), or when any of `grad_output`, `r1`, and `r2` is an empty tensor (with a dimension of size 0), the operator performs a no-op operation and returns directly without computing the rotary position embedding gradient.
- The size of the last dimension D of `grad_output` must not exceed 1024.
- `partial_slice` must contain two integers, satisfying `start >= 0`, `end >= 0`, `end <= D`, and `end - start >= 0`.
- When the slice length (that is, `end - start`) is greater than 0, the slice length must be a multiple of 2.
- When the slice length (that is, `end - start`) is greater than 0, the last dimensions of `r1` and `r2` must have the same size, and the size must equal the slice length.
- The shapes of `r1` and `r2` must satisfy the broadcasting relationship with `grad_output[..., start:end]`, and the shapes of `r1` and `r2` currently support only BSND, B1ND, B11D, and 111D layouts.
- When the slice length (i.e., `end - start`) is greater than 0, each dimension value of `grad_output`, `r1`, and `r2` must be greater than 0.

## Deterministic Computation

Defaults to a deterministic implementation.

## Related APIs

This operator is the backward operator of [inplace_partial_rotary_mul](../../inplace_partial_rotary_mul/docs/torchapi_inplace_partial_rotary_mul_en.md). The parameters `r1`, `r2`, `rotary_mode`, and `partial_slice` of the two operators must be consistent.

> **NOTE**: When `x.requires_grad` of the forward `inplace_partial_rotary_mul` is True, `loss.backward()` automatically triggers this operator, which does not need to be manually called. The manual call path is retained only for use cases where explicit gradient control is needed.

## Usage Instructions

- Single-operator mode call

    ```python
    import torch
    import torch_npu
    from cann_ops_transformer.ops import inplace_partial_rotary_mul
    from cann_ops_transformer.ops import inplace_partial_rotary_mul_backward

    torch_npu.npu.set_device(0)

    B = 2
    S = 32
    N = 8
    D = 128
    slice_start = 0
    slice_end = 64

    # Forward: x is modified inplace
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

    # Backward: grad_output is modified inplace to dx
    grad_output = torch.randn_like(x)
    inplace_partial_rotary_mul_backward(
        grad_output,
        r1,
        r2,
        rotary_mode="interleave",
        partial_slice=[slice_start, slice_end],
    )
    ```

- Graph mode call

    ```python
    import torch
    import torch_npu
    import torchair
    from cann_ops_transformer.ops import inplace_partial_rotary_mul_backward

    torch_npu.npu.set_device(0)

    B = 2
    S = 32
    N = 8
    D = 128
    slice_start = 0
    slice_end = 64

    class InplacePartialRotaryMulBackwardModel(torch.nn.Module):
        def forward(self, grad_output, r1, r2):
            inplace_partial_rotary_mul_backward(
                grad_output,
                r1,
                r2,
                rotary_mode="interleave",
                partial_slice=[slice_start, slice_end],
            )
            return grad_output

    model = InplacePartialRotaryMulBackwardModel().npu()
    npu_backend = torchair.get_npu_backend()
    model = torch.compile(model, backend=npu_backend, dynamic=False)

    grad_output = torch.randn(B, S, N, D, device="npu", dtype=torch.float16)
    r1 = torch.randn(B, S, 1, slice_end - slice_start, device="npu", dtype=torch.float16)
    r2 = torch.randn(B, S, 1, slice_end - slice_start, device="npu", dtype=torch.float16)

    output = model(grad_output, r1, r2)
    ```
