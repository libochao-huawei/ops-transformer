# moe_token_permute

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:30:04.190Z -->

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

`moe_token_permute` is a `cann_ops_transformer` extension API based on `torch_npu`. This API expands the input `tokens` based on `indices`, sorts them by expert index, and returns the sorted tokens and their mapping to the original tokens.

<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT products</term>: This API also supports quantizing the output to MXFP8 or MXFP4 during permute and returns the corresponding per-token block scale.
<!-- end id7 -->
<!-- npu="A3,910b" id8 -->
- <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: quantization mode is not supported; execution follows the non-quantized path.
<!-- end id8 -->

## Function Prototype

```python
cann_ops_transformer.ops.moe_token_permute(
    tokens,
    indices,
    num_out_tokens=None,
    padded_mode=False,
    quant_mode=-1,
) -> (Tensor, Tensor, Tensor)
```

## Mandatory

| Name | Type | Type | Description | Data Type | Dimension (Shape) |
| --- | --- | --- | --- | --- | --- |
| `tokens` | Tensor | Required | Input token features. The shape is `[N, H]`. In quantization mode, only float16 and bfloat16 are supported. | Non-quantization: float16, bfloat16, float32; Ascend 950PR&950DT products non-quantization additionally supports int8; quantization: float16, bfloat16 | `[N, H]` |
| `indices` | Tensor | Required | Expert indices of tokens. Can be a one-dimensional index, or indicate that each token corresponds to K experts. The number of elements is `F`. | int32, int64 | `[N]` or `[N, K]` |
| `num_out_tokens` | int | Optional | Controls the number of valid output rows of `permuted_tokens`. `None` or `0` indicates that all rows are retained; a positive value indicates that at most the specified number of rows are retained; a negative value indicates that the corresponding number of rows are deleted from the tail of the full result. Default value is `None`. | int64 | - |
| `padded_mode` | bool | Optional | Whether to use padded mode. Currently only `False` is supported. Default value is `False`. | bool | - |
| `quant_mode` | int | Optional | Quantization mode: `-1` indicates no quantization; `2` indicates MXFP8 E5M2; `3` indicates MXFP8 E4M3FN; `9` indicates MXFP4 E2M1. Default value is `-1`. | int64 | - |

<!-- npu="A3,910b" id9 -->
- <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: `quant_mode` supports only -1.
<!-- end id9 -->
<!-- npu="950" id10 -->
- <term>Ascend 950PR&950DT products</term>: `quant_mode` supports -1/2/3/9. If the value is 9, the hidden dimension `H` must be an even number. If the value is 2/3/9, autograd is not supported.
<!-- end id10 -->

## Output Values

The API returns three tensors:

```text
permuted_tokens, sorted_indices, expanded_scale
```

Definition:

```text
F = indices.numel()
H = tokens.shape[1]

num_out_tokens is None or num_out_tokens == 0: M = F
num_out_tokens > 0:                              M = min(num_out_tokens, F)
num_out_tokens < 0:                              M = max(F + num_out_tokens, 0)
```

The outputs in each quantization mode are as follows:

| `quant_mode` | `permuted_tokens` | `sorted_indices` | `expanded_scale` |
| --- | --- | --- | --- |
| `-1` | shape `[M, H]`, dtype the same as `tokens` | shape `[F]`, dtype int32 | empty tensor with shape `[0]`, dtype float32 |
| `2` | shape `[M, H]`, dtype float8_e5m2 | shape `[F]`, dtype int32 | shape `[M, AlignUp(CeilDiv(H, 32), 2)]`, dtype float8_e8m0 |
| `3` | shape `[M, H]`, dtype float8_e4m3fn | shape `[F]`, dtype int32 | shape `[M, AlignUp(CeilDiv(H, 32), 2)]`, dtype float8_e8m0 |
| `9` | PyTorch physical shape `[M, H / 2]`, dtype uint8 | shape `[F]`, dtype int32 | PyTorch physical shape `[M, CeilDiv(H, 64), 2]`, dtype uint8 |

Where:

```text
CeilDiv(a, b)  = (a + b - 1) // b
AlignUp(a, b)  = CeilDiv(a, b) * b
```

- `permuted_tokens`: tokens expanded based on `indices` and sorted by expert index; only the first dimensions of this output and `expanded_scale` are affected by `num_out_tokens`.
- `sorted_indices`: row mapping between `permuted_tokens` and the original `tokens`; the length is always `indices.numel()`.
- `expanded_scale`: per-token block scale corresponding to the quantized output; an empty tensor is returned in non-quantized mode.

> [!NOTE]
>
> MXFP4 data is currently physically stored in a uint8 tensor. When the aclnn API is called, `permuted_tokens` is interpreted as `ACL_FLOAT4_E2M1`, and `expanded_scale` is interpreted as `ACL_FLOAT8_e8m0`. Therefore, uint8 in the table indicates the physical storage type on the PyTorch side, not the logical type of the quantized data.

## Constraints

- The number of elements in `indices` must be less than `16777215`, and each element value must be greater than or equal to `0` and less than `16777215`.
- For MXFP8, every 32 quantized values share one E8M0 scale, and the number of scales is aligned to 2.
- For MXFP4, two 4-bit values are packed into one uint8. Therefore, the PyTorch physical hidden dimension of `permuted_tokens` is `H / 2`.

## Deterministic Computation

Defaults to a deterministic implementation.

## Example

### Non-quantized Call

```python
import torch
import torch_npu
from cann_ops_transformer.ops import moe_token_permute

tokens = torch.randn(4, 128, dtype=torch.float16, device="npu")
indices = torch.tensor(
    [[1, 0], [0, 1], [1, 0], [0, 1]],
    dtype=torch.int32,
    device="npu",
)

permuted_tokens, sorted_indices, expanded_scale = moe_token_permute(
    tokens,
    indices,
    num_out_tokens=6,
)

print(permuted_tokens.shape)  # torch.Size([6, 128])
print(sorted_indices.shape)   # torch.Size([8])
print(expanded_scale.shape)   # torch.Size([0])
```

<!-- npu="950" id11 -->
### Ascend 950PR&950DT products MXFP8 Call

```python
permuted_tokens, sorted_indices, expanded_scale = moe_token_permute(
    tokens,
    indices,
    quant_mode=2,
)

print(permuted_tokens.shape)  # torch.Size([8, 128])
print(expanded_scale.shape)   # torch.Size([8, 4])
```
<!-- end id11 -->
