# moe_finalize_routing

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:04:00.756Z -->

## Applicable Products

- <term>Ascend 950PR&950DT products</term>: Supported
- <term>Atlas A3 products</term>: Supported
- <term>Atlas A2 products</term>: Supported
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
- <term>Atlas inference products</term>: Supported
- <term>Atlas training products</term>: Not supported

## Function

`moe_finalize_routing` is an extension API of `cann_ops_transformer` based on `torch_npu`. This API is used in the final phase of MoE (Mixture of Experts) computation. It merges the FFN outputs of experts by routing weights and restores the original token sequence.

Formula:

$$
\begin{aligned} &\mathrm{expertId} = \mathrm{expertIdx}[i,k] \\ &\text{if } \mathrm{expertId} \in \mathrm{zero\_expert\_range}:\\&\quad \text{skip}; \\&\text{elif } \mathrm{expertId} \in \mathrm{copy\_expert\_range}: \\ &\quad x = x[i] \\ &\quad \mathrm{out}(i, j) = x1_{i, j} + x2_{i, j} + \sum_{k=1}^{K}\mathrm{scales}_{i,k} \cdot \left(x + \mathrm{bias}_{\mathrm{expertId},j}\right) \\ &\text{elif } \mathrm{expertId} \in \mathrm{constant\_expert\_range}: \\&\quad x = \alpha_1 \cdot x[i] + \alpha_2 \cdot v \\ &\quad \mathrm{out}(i, j) = x1_{i, j} + x2_{i, j} + \sum_{k=1}^{K}\mathrm{scales}_{i,k} \cdot \left(x + \mathrm{bias}_{\mathrm{expertId},j}\right) \\ &\text{else}: \\ &\quad \mathrm{out}(i, j) = x1_{i, j} + x2_{i, j} + \sum_{k=0}^{K}\mathrm{scales}_{i,k} \cdot\big(\mathrm{expandedX}_{\mathrm{expandedRowIdx}_{i\cdot K+k},j} + \mathrm{bias}_{\mathrm{expertId}_{i\cdot K+k},j}\big) \end{aligned}
$$

## Function Prototype

```python
cann_ops_transformer.ops.moe_finalize_routing(
    expanded_x,
    expanded_row_idx,
    x1=None,
    x2=None,
    bias=None,
    scales=None,
    expert_idx=None,
    x=None,
    alpha1=None,
    alpha2=None,
    v=None,
    drop_pad_mode=0,
    zero_expert_range=None,
    copy_expert_range=None,
    constant_expert_range=None,
    k=1,
) -> Tensor
```

## Mandatory

| Name | Type | Type | Description | Data Type | Dimension (Shape) |
| --- | --- | --- | --- | --- | --- |
| `expanded_x` | Tensor | Required | Output of the MoE FFN, that is, expandedX in the formula. In the drop less use case, the shape is `(NUM_ROWS * K, H)`; in the drop pad use case, the shape is `(E, C, H)`. | FLOAT16, BFLOAT16, FLOAT32 | `(NUM_ROWS * K, H)` or `(E, C, H)` |
| `expanded_row_idx` | Tensor | Required | Token index mapping, that is, expandedRowIdx in the formula. If drop_pad_mode is 0 or 2, the value range is `[0, NUM_ROWS * K - 1]`; if drop_pad_mode is 1 or 3, the value range is `[-1, E * C - 1]`, where -1 indicates that the token is dropped. | INT32 | `(NUM_ROWS * K,)` |
| `x1` | Tensor | Optional | Output of the first shared expert, that is, x1 in the formula. The shape is the same as that of the output. | Same as expanded_x | `(NUM_ROWS, H)` |
| `x2` | Tensor | Optional | Output of the second shared expert, that is, x2 in the formula. The shape is the same as that of the output. If x1 is not passed in, x2 cannot be passed in either. | Same as expanded_x | `(NUM_ROWS, H)` |
| `bias` | Tensor | Optional | Bias. If bias exists, expert_idx must also exist. | Same as expanded_x | `(E, H)` |
| `scales` | Tensor | Optional | Routing weight coefficients. | FLOAT16, BFLOAT16, FLOAT32 | `(NUM_ROWS, K)` |
| `expert_idx` | Tensor | Optional | Expert index, that is, expertIdx in the formula. It must be passed in together with bias. The value range is `[0, E-1]`. | INT32 | `(NUM_ROWS, K)` |
| `x` | Tensor | Optional | Input of the copy expert and constant expert use cases. | Same as expanded_x | `(NUM_ROWS, H)` |
| `alpha1` | Tensor | Optional | First scaling coefficient of the constant expert. | Same as expanded_x | `(constant_expert_range_num, H)` |
| `alpha2` | Tensor | Optional | Second scaling coefficient of the constant expert. | Same as expanded_x | `(constant_expert_range_num, H)` |
| `v` | Tensor | Optional | Offset vector of the constant expert. | Same as expanded_x | `(constant_expert_range_num, H)` |
| `drop_pad_mode` | int | Optional | Drop/pad mode and arrangement of expandedRowIdx. `0`: drop less, column arrangement; `1`: drop pad, column arrangement; `2`: drop less, row arrangement; `3`: drop pad, row arrangement. Default value: 0. | int64 | - |
| `zero_expert_range` | list[int] | Optional | Zero expert range `[start, end)`, left-closed and right-open. Outputs of experts in this range are skipped. | list[int64] | length is 2 |
| `copy_expert_range` | list[int] | Optional | Copy expert range `[start, end)`, left-closed and right-open. `x[i]` is used to replace expandedX in this range. | list[int64] | length is 2 |
| `constant_expert_range` | list[int] | Optional | Constant expert range `[start, end)`, left-closed and right-open. `alpha1 * x[i] + alpha2 * v` is used to replace expandedX in this range. | list[int64] | length is 2 |
| `k` | int | Optional | Number of top-K experts selected for each token. If scales is not empty and k is explicitly passed in, k must be the same as the second dimension of scales. Default value: 1. | int64 | - |

## Return Values

Returns a Tensor with shape `(NUM_ROWS, H)` and the same data type as `expanded_x`.

## Constraints

- `expanded_x` must be a 2D or 3D Tensor.
- `expanded_row_idx` must be a 1D Tensor. If `k` is explicitly passed, the number of elements must be divisible by `k`.
- The value range of `drop_pad_mode` is `[0, 3]`.
- If `x1` is not passed, `x2` must not be passed.
- When `bias` is present, `expert_idx` must also be present.
- If `scales` is not empty and `k` is explicitly passed, `k` must be consistent with the second dimension of `scales`.
- If `scales` is empty, `k` must be passed.
- The three ranges `zero_expert_range`, `copy_expert_range`, and `constant_expert_range` must not overlap.
- **Automatic backward (autograd)**: Automatic backward is supported when `requires_grad` of differentiable inputs such as `expanded_x` and `scales` is True. The backward operator is [moe\_finalize\_routing\_grad](../../moe_finalize_routing_v2_grad/docs/torchapi_moe_finalize_routing_grad_en.md) (which encapsulates `aclnnMoeFinalizeRoutingV2Grad`). The automatic backward constraints are as follows:
  - Supported only when the forward propagation degrades to the aclnnMoeFinalizeRoutingV2 use case, that is, the aclnnMoeFinalizeRoutingV4-specific inputs `x`, `alpha1`, `alpha2`, and `v` are not passed, and `zero_expert_range`, `copy_expert_range`, and `constant_expert_range` are None or invalid (for example, `[-1, -1]`). If any of the preceding aclnnMoeFinalizeRoutingV4-specific features is used, calling automatic backward throws `NotImplementedError`.
  - `drop_pad_mode` supports only 0 or 1 (column permutation mode), and does not support 2 or 3 (row permutation mode).
  - Only `grad_expanded_x` and `grad_scales` are computed; the gradients of `x1`, `x2`, and `bias` are not computed (`None` is returned). To compute the gradients of `x1`/`x2`, use external residual addition instead of passing them as forward inputs.
  - The forward `expanded_row_idx` uses the `(K, R)` layout, and the backward operator uses the `(R, K)` layout. In automatic backward, the framework performs the transpose automatically.
  - **Product support differences**: `aclnnMoeFinalizeRoutingV2Grad` is not supported on <term>Atlas inference products</term>, although the forward `moe_finalize_routing` is supported on this product. Therefore, `moe_finalize_routing` does not support automatic backward on <term>Atlas inference products</term>.

## Example

```python
import torch
import torch_npu
from cann_ops_transformer import moe_finalize_routing

NUM_ROWS = 4
K = 2
H = 8
E = 8
CE_NUM = 2  # constant_expert_range_num = end - start

expanded_x = torch.randn(NUM_ROWS * K, H, dtype=torch.float32, device="npu")
row_idx = torch.arange(NUM_ROWS * K, dtype=torch.int32, device="npu")
scales = torch.randn(NUM_ROWS, K, dtype=torch.float32, device="npu")
x1 = torch.randn(NUM_ROWS, H, dtype=torch.float32, device="npu")
x2 = torch.randn(NUM_ROWS, H, dtype=torch.float32, device="npu")
bias = torch.randn(E, H, dtype=torch.float32, device="npu")
expert_idx = torch.tensor(
    [[0, 1], [2, 3], [4, 5], [6, 7]], dtype=torch.int32, device="npu"
)
x = torch.randn(NUM_ROWS, H, dtype=torch.float32, device="npu")
alpha1 = torch.randn(CE_NUM, H, dtype=torch.float32, device="npu")
alpha2 = torch.randn(CE_NUM, H, dtype=torch.float32, device="npu")
v = torch.randn(CE_NUM, H, dtype=torch.float32, device="npu")

out = moe_finalize_routing(
    expanded_x,
    row_idx,
    x1=x1,
    x2=x2,
    bias=bias,
    scales=scales,
    expert_idx=expert_idx,
    x=x,
    alpha1=alpha1,
    alpha2=alpha2,
    v=v,
    drop_pad_mode=0,
    zero_expert_range=[0, 2],
    copy_expert_range=[2, 4],
    constant_expert_range=[6, 8],
    k=K,
)
print(out.shape)  # torch.Size([4, 8])
```

- Automatic backward call (training use case):

  ```python
  import torch
  import torch_npu
  from cann_ops_transformer.ops import moe_finalize_routing

  torch_npu.npu.set_device(0)

  NUM_ROWS = 4
  K = 2
  H = 8

  # Autograd is automatically enabled when expanded_x.requires_grad=True
  expanded_x = torch.randn(NUM_ROWS * K, H, dtype=torch.float32, device="npu",
                           requires_grad=True)
  row_idx = torch.arange(NUM_ROWS * K, dtype=torch.int32, device="npu")
  scales = torch.randn(NUM_ROWS, K, dtype=torch.float32, device="npu",
                       requires_grad=True)

  # Forward: do not pass the aclnnMoeFinalizeRoutingV4-specific inputs (x/alpha1/alpha2/v) and do not set zero_expert_range
  y = moe_finalize_routing(expanded_x, row_idx, scales=scales, k=K)

  # Backward: moe_finalize_routing_grad is triggered automatically
  loss = y.sum()
  loss.backward()
  print(expanded_x.grad.shape)  # torch.Size([8, 8])
  print(scales.grad.shape)      # torch.Size([4, 2])
  ```

## Deterministic Computation

Defaults to a deterministic implementation.
