# moe_init_routing_grad

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:18:54.355Z -->

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

- Description: `moe_init_routing_grad` is the backward computation of `moe_init_routing` and encapsulates the `aclnnMoeInitRoutingV2Grad` API. Based on `expanded_row_idx` output by the forward `moe_init_routing`, this API maps `grad_expanded_x` (the gradient of `expanded_x`) row by row back to `grad_x`, the gradient of the original input `x`.

- Formula:

  $$
  grad\_x[i] = \sum_{t=0}^{K-1} grad\_expanded\_x[expanded\_row\_idx[i \cdot K + t]]
  $$

  In the formula, `i` is in the range `[0, NUM_ROWS)`, `K` is `top_k`, and `expanded_row_idx` is the output of the forward `moe_init_routing`. In the DropPad scenario, entries in `expanded_row_idx` with a value of -1 are excluded from the accumulation.

## Function Prototype

```python
cann_ops_transformer.moe_init_routing_grad(
    grad_expanded_x: torch.Tensor,
    expanded_row_idx: torch.Tensor,
    top_k: int,
    drop_pad_mode: int = 0,
    active_num: int = 0,
) -> torch.Tensor
```

## Mandatory

| Name | Type | Type | Description | Data Type | Dimension (Shape) |
|---------|----------|----------|---------------|---------------|------------|
| grad_expanded_x | Tensor | Required | Gradient of the forward output `expanded_x`. A 2D tensor in the Dropless scenario, with shape `(active_num, H)` or `(NUM_ROWS * K, H)`; a 3D tensor in the DropPad scenario, with shape `(expert_num, expert_capacity, H)`. | float16, bfloat16, float32 | ND |
| expanded_row_idx | Tensor | Required | Output `expanded_row_idx` of the forward `moe_init_routing`, indicating the row mapping between `expanded_x` and `x`. A 1D tensor with shape `(NUM_ROWS * K,)`. | int32 | ND |
| top_k | int | Required | Dimension size K of the second dimension of the forward input `expert_idx`, that is, the number of top-K experts selected for each token. The value must be greater than 0. | - | - |
| drop_pad_mode | int | Optional | Default value 0, indicating the drop/pad mode, which must be consistent with the forward computation. 0 indicates the Dropless scenario; 1 indicates the DropPad scenario. | - | - |
| active_num | int | Optional | Default value 0, indicating the `active_num` parameter value of the forward `moe_init_routing`. The value must be greater than or equal to 0 and takes effect when `drop_pad_mode=0`: 0 indicates the non-Active scenario, and a value greater than 0 indicates the Active scenario (in this case, the dimension size of dimension 0 of `grad_expanded_x` must be equal to `active_num`). | - | - |

## Return Values

|  Name   | Type |  Type |    Description    |     Data Type    |   Dimension (Shape)   |
|---------|----------|----------|---------------|---------------|------------|
|  grad_x  |  Tensor |  Required  |  Gradient of input `x`. The shape is `(NUM_ROWS, H)`, where `NUM_ROWS = expanded_row_idx.numel() // top_k`, and `H` is the last dimension of `grad_expanded_x` (the third dimension in the DropPad scenario). The data type is the same as that of `grad_expanded_x`.  |  Same as grad_expanded_x   | ND |

## Constraints

- This API supports training scenarios.
- This API supports both the single-operator mode and the graph mode.
- `grad_expanded_x` must be a 2D tensor in the Dropless scenario and a 3D tensor in the DropPad scenario.
- `expanded_row_idx` must be a 1D tensor, and its length must be divisible by `top_k`.
- `drop_pad_mode` supports only the values 0 and 1.
- `top_k` must be greater than 0.
- `active_num` must be greater than or equal to 0.
- In the Dropless scenario where `active_num=0`, the dimension size of dimension 0 of `grad_expanded_x` must be the same as the length of `expanded_row_idx`. In the Dropless scenario where `active_num>0`, the dimension size of dimension 0 of `grad_expanded_x` must be equal to `active_num`.
- The last dimension of `grad_expanded_x` (the third dimension in the DropPad scenario) must be consistent with the second dimension of output `grad_x`.
- This operator is the backward computation of [moe\_init\_routing](../../moe_init_routing_v4/docs/torchapi_moe_init_routing_en.md). `expanded_row_idx`, `top_k`, `drop_pad_mode`, and `active_num` must be consistent with those in the forward call.
- This backward computation only supports the non-quantization scenario (forward `quant_mode=-1`) and does not support features specific to aclnnMoeInitRoutingV4 (`scale`, `offset`, `topk_weight`, `x_dtype`, and non-zero `row_idx_type`). If the forward pass uses these features, the automatic backward call throws `NotImplementedError`.

## Deterministic Computation

Defaults to a deterministic implementation.

## Related APIs

This operator is the backward computation of [moe\_init\_routing](../../moe_init_routing_v4/docs/torchapi_moe_init_routing_en.md).

> **NOTE**: When `x.requires_grad` is True in the forward `moe_init_routing` and no aclnnMoeInitRoutingV4-specific features (quantization, `scale`, `offset`, `topk_weight`, etc.) are used, `loss.backward()` automatically triggers this operator. No manual call is required. Keep the manual call path only in scenarios that require explicit gradient control.

## Example

- Single-operator mode call:

  ```python
  import torch
  import torch_npu
  from cann_ops_transformer.ops import moe_init_routing, moe_init_routing_grad

  torch_npu.npu.set_device(0)

  n = 4
  h = 8
  k = 2
  expert_num = 8
  drop_pad_mode = 0
  expert_tokens_num_type = 1
  expert_tokens_num_flag = True
  quant_mode = -1
  active_expert_range = [0, 4]
  row_idx_type = 0

  x = torch.randn((n, h), dtype=torch.float32, device="npu")
  expert_idx = torch.randint(0, expert_num, (n, k), dtype=torch.int32, device="npu")

  # Forward
  expanded_x, expanded_row_idx, expert_tokens_count, expanded_scale, expanded_topk_weight = \
      moe_init_routing(x, expert_idx, active_num=-1, expert_capacity=-1,
                       expert_num=expert_num, drop_pad_mode=drop_pad_mode,
                       expert_tokens_num_type=expert_tokens_num_type,
                       expert_tokens_num_flag=expert_tokens_num_flag,
                       quant_mode=quant_mode, active_expert_range=active_expert_range,
                       row_idx_type=row_idx_type)

  # Manually call backward.
  grad_expanded_x = torch.randn_like(expanded_x)
  grad_x = moe_init_routing_grad(grad_expanded_x, expanded_row_idx, top_k=k,
                                 drop_pad_mode=drop_pad_mode, active_num=0)
  print(grad_x.shape)  # torch.Size([4, 8])
  ```

- Automatic backward call:

  ```python
  import torch
  import torch_npu
  from cann_ops_transformer.ops import moe_init_routing

  torch_npu.npu.set_device(0)

  n = 4
  h = 8
  k = 2
  expert_num = 8

  x = torch.randn((n, h), dtype=torch.float32, device="npu", requires_grad=True)
  expert_idx = torch.randint(0, expert_num, (n, k), dtype=torch.int32, device="npu")

  # Forward (autograd is automatically enabled when x.requires_grad=True)
  expanded_x, expanded_row_idx, expert_tokens_count, expanded_scale, expanded_topk_weight = \
      moe_init_routing(x, expert_idx, active_num=-1, expert_capacity=-1,
                       expert_num=expert_num, drop_pad_mode=0,
                       expert_tokens_num_type=1, expert_tokens_num_flag=True,
                       quant_mode=-1, active_expert_range=[0, 4], row_idx_type=0)

  # Backward automatically triggers moe_init_routing_grad
  loss = expanded_x.sum()
  loss.backward()
  print(x.grad.shape)  # torch.Size([4, 8])
  ```
