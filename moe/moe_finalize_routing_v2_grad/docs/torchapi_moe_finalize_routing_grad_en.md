# moe_finalize_routing_grad

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:05:23.506Z -->

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

- Description: `moe_finalize_routing_grad` is the backward computation of `moe_finalize_routing` and encapsulates the `aclnnMoeFinalizeRoutingV2Grad` API. Based on the forward output `expanded_row_idx`, this API maps `grad_y` (that is, the gradient of the forward output `y`) back to the gradient `grad_expanded_x` of `expanded_x` and the gradient `grad_scales` of `scales` according to the routing weight `scales`.

- Formula:

  Let `i` be the flattened index in the range `[0, R * K)`, where `R` is the number of token rows and `K` is the number of top-K experts:

  - When `scales` is not passed (K must be 1):

    $$
    grad\_expanded\_x[expanded\_row\_idx[i]][j] = grad\_y[i / K][j]
    $$

  - When `scales` is passed and `bias` is not passed:

    $$
    grad\_expanded\_x[expanded\_row\_idx[i]][j] = grad\_y[i / K][j] \cdot scales[i / K][i \% K]
    $$

    $$
    grad\_scales[i] = \sum_{j} expanded\_x[expanded\_row\_idx[i]][j] \cdot grad\_y[i / K][j]
    $$

  - When both `scales` and `bias` are passed:

    $$
    grad\_expanded\_x[expanded\_row\_idx[i]][j] = grad\_y[i / K][j] \cdot scales[i / K][i \% K]
    $$

    $$
    grad\_scales[i] = \sum_{j} (expanded\_x[expanded\_row\_idx[i]][j] + bias[expert\_idx[i]][j]) \cdot grad\_y[i / K][j]
    $$

- Description:

  - This backward computation only computes `grad_expanded_x` and `grad_scales`, not the gradients of `x1`, `x2`, or `bias` (the gradients return `None` in automatic backward propagation).
  - The layout of `expanded_row_idx` differs from that of the forward computation: the forward computation uses the `(K, R)` layout (`idx[k * R + row]`), while this backward computation uses the `(R, K)` layout (`idx[row * K + k]`). When called through automatic backward propagation, the framework automatically transposes the layout. During manual invocation, ensure the layout is correct.

## Function Prototype

```python
cann_ops_transformer.moe_finalize_routing_grad(
    grad_y: torch.Tensor,
    expanded_row_idx: torch.Tensor,
    expanded_x: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
    expert_idx: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    drop_pad_mode: int = 0,
    active_num: int = 0,
    expert_num: int = 0,
    expert_capacity: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]
```

## Mandatory

|  Name   | Type |  Type |    Description    |     Data Type    |   Dimension (Shape)   |
|---------|----------|----------|---------------|---------------|------------|
|  grad_y  |  Tensor |  Required  |  Gradient of the forward output `y`.  |  float16, bfloat16, float32   | `(R, H)` |
|  expanded_row_idx  |  Tensor |  Required  |  Row index mapping, in the `(R, K)` layout. Note that it is different from the `(K, R)` layout in the forward computation. In the Dropless use case, values are in the range `[0, R * K - 1]` and are not repeated; in the DropPad use case, values are in the range `[-1, E * C - 1]` and are not repeated except -1 (-1 indicates dropping). K must be 1 when `scales` is not passed.  |  int32   | `(R * K,)` |
|  expanded_x  |  Tensor |  Optional  |  Forward input `expanded_x`, used to compute `grad_scales`. Must be passed when `scales` is passed. The data type must be consistent with grad_y.  |  consistent with grad_y   | `(NUM_ROWS * K, H)` or `(E, C, H)` |
|  scales  |  Tensor |  Optional  |  Forward input `scales`, i.e., the routing weight coefficients. `expanded_x` must be passed when `scales` is passed.  |  float16, bfloat16, float32   | `(R, K)` |
|  expert_idx  |  Tensor |  Optional  |  Forward input `expert_idx`, i.e., the expert index. Must be passed when `bias` is passed. The value range is `[0, E-1]`.  |  int32   | `(R, K)` |
|  bias  |  Tensor |  Optional  |  Forward input `bias`, i.e., the bias.  |  consistent with grad_y   | `(E, H)` |
|  drop_pad_mode  |  int |  Optional  |  Drop/pad mode, which must be consistent with that of the forward computation. `0`: drop less; `1`: drop pad. The default value is 0.  |  -   | - |
|  active_num  |  int |  Optional  |  Maximum number of output rows of `grad_expanded_x` in the Dropless use case. It takes effect only when the value is greater than 0 and less than `R * K`; it has no effect in the DropPad use case. The default value is 0.  |  -   | - |
|  expert_num  |  int |  Optional  |  Number of experts. It has no effect in the Dropless use case. In the DropPad use case, it must be greater than 0, and when `bias` is passed, it must be equal to the size of dimension 0 of `bias` (E). The default value is 0.  |  -   | - |
|  expert_capacity  |  int |  Optional  |  Number of tokens that each expert can process. Must be passed and greater than 0 when `drop_pad_mode=1`. The default value is 0.  |  -   | - |

## Return Values

|  Name   | Type |  Type |    Description    |     Data Type    |   Dimension (Shape)   |
|---------|----------|----------|---------------|---------------|------------|
|  grad_expanded_x  |  Tensor |  Required  |  Gradient of `expanded_x`. In the Dropless scenario, the shape is `(dim0, H)`, where `dim0` is `expanded_row_idx.numel()` (when `active_num=0`) or `min(active_num, expanded_row_idx.numel())` (when `active_num > 0`); in the DropPad scenario, the shape is `(expert_num, expert_capacity, H)`. The data type is consistent with `grad_y`.  |  consistent with grad_y   | `(dim0, H)` or `(E, C, H)` |
|  grad_scales  |  Tensor |  Required  |  Gradient of `scales`. The shape is `(R, K)`, where `K` is the second dimension of `scales` (when `scales` is passed) or 1 (when `scales` is not passed). Data type: consistent with `scales` when `scales` is passed; otherwise consistent with `grad_y`.  |  float16, bfloat16, float32   | `(R, K)` |

## Constraints

- This API supports use in the training scenario.
- This API supports the single-operator mode and graph mode.
- `grad_y` must be a 2D tensor.
- `expanded_row_idx` must be a 1D tensor.
- `drop_pad_mode` supports only 0 or 1; 2 or 3 (row permutation mode) is not supported.
- When `bias` is passed, `expert_idx` must also be passed.
- When `scales` is passed, `expanded_x` must also be passed.
- When `drop_pad_mode=1`, `expert_num` must be greater than 0 (equal to E, the size of dimension 0 of `bias`, when `bias` is passed), and `expert_capacity` must be greater than 0.
- The data types of `expanded_x` and `bias` must be consistent with `grad_y`. The data type of `scales` must be consistent with `grad_y` on <term>Atlas A2 products</term> and <term>Atlas A3 products</term>, but can be inconsistent with `grad_y` on <term>Ascend 950PR&950DT products</term>.
- This backward computation supports only the regular expert use case. It does not support the features specific to aclnnMoeFinalizeRoutingV4 (forward inputs `x`, `alpha1`, `alpha2`, and `v`, and valid `zero_expert_range`, `copy_expert_range`, and `constant_expert_range`). If these features are used in the forward computation, the automatic backward invocation throws `NotImplementedError`.
- This operator is the backward computation of [moe\_finalize\_routing](../../moe_finalize_routing_v2/docs/torchapi_moe_finalize_routing_en.md). All parameters must be the same as those in the forward call.
- `expanded_row_idx` layout difference: the `expanded_row_idx` of the forward `moe_finalize_routing` uses the `(K, R)` layout (when `drop_pad_mode` is 0 or 1), while this backward computation uses the `(R, K)` layout. In automatic backward propagation, the framework performs the transposition automatically. For manual invocation, provide the correct layout.

## Deterministic Computation

Defaults to a deterministic implementation.

## Related APIs

This operator is the backward computation of [moe\_finalize\_routing](../../moe_finalize_routing_v2/docs/torchapi_moe_finalize_routing_en.md).

> **NOTE**: When `requires_grad` of differentiable inputs such as `expanded_x` and `scales` in the forward `moe_finalize_routing` is True, and aclnnMoeFinalizeRoutingV4-specific features (`x`, `alpha1`, `alpha2`, `v`, `zero_expert_range`, etc.) are not used, `loss.backward()` automatically triggers this operator without manual invocation. The manual invocation path is retained only in the scenario where explicit gradient control is required. Note: gradients of `x1`, `x2`, and `bias` are not computed. If gradients of `x1`/`x2` are required, use external residual addition instead of passing them as forward inputs.

## Example

- Single-operator mode call:

  ```python
  import torch
  import torch_npu
  from cann_ops_transformer.ops import moe_finalize_routing, moe_finalize_routing_grad

  torch_npu.npu.set_device(0)

  NUM_ROWS = 2
  K = 2
  H = 8
  E = 4

  expanded_x = torch.randn(NUM_ROWS * K, H, dtype=torch.float32, device="npu")
  # The forward expanded_row_idx is in (K, R) layout: idx[k*R + row]
  forward_row_idx = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device="npu")
  scales = torch.randn(NUM_ROWS, K, dtype=torch.float32, device="npu")
  expert_idx = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device="npu")
  bias = torch.randn(E, H, dtype=torch.float32, device="npu")

  # Forward pass
  y = moe_finalize_routing(expanded_x, forward_row_idx, bias=bias,
                           scales=scales, expert_idx=expert_idx, k=K)

  # Manual invocation of backward: transpose expanded_row_idx from the (K, R) layout to the (R, K) layout.
  grad_row_idx = forward_row_idx.reshape(K, NUM_ROWS).t().contiguous().reshape(-1)
  grad_y = torch.randn_like(y)
  grad_expanded_x, grad_scales = moe_finalize_routing_grad(
      grad_y, grad_row_idx, expanded_x=expanded_x, scales=scales,
      expert_idx=expert_idx, bias=bias, drop_pad_mode=0,
  )
  print(grad_expanded_x.shape)  # torch.Size([4, 8])
  print(grad_scales.shape)      # torch.Size([2, 2])
  ```

- Automatic backward call:

  ```python
  import torch
  import torch_npu
  from cann_ops_transformer.ops import moe_finalize_routing

  torch_npu.npu.set_device(0)

  NUM_ROWS = 4
  K = 2
  H = 8

  expanded_x = torch.randn(NUM_ROWS * K, H, dtype=torch.float32, device="npu",
                           requires_grad=True)
  row_idx = torch.arange(NUM_ROWS * K, dtype=torch.int32, device="npu")
  scales = torch.randn(NUM_ROWS, K, dtype=torch.float32, device="npu",
                       requires_grad=True)

  # Forward pass (autograd is automatically enabled when expanded_x.requires_grad=True)
  y = moe_finalize_routing(expanded_x, row_idx, scales=scales, k=K)

  # Backward pass automatically triggers moe_finalize_routing_grad
  loss = y.sum()
  loss.backward()
  print(expanded_x.grad.shape)  # torch.Size([8, 8])
  print(scales.grad.shape)      # torch.Size([4, 2])
  ```
