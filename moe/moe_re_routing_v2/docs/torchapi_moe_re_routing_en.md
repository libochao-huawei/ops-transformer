# moe_re_routing

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:21:46.253Z -->

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

- Description: In a MoE network, after the tokens to be computed are obtained from other devices through AlltoAll, rearranges the tokens in expert order. Compared with `torch_npu.npu_moe_re_routing`, this operator adds the optional input `expert_topk_weight` and the optional output `permute_topk_weight`, and supports rearranging topkWeight in expert order, so that `permute_topk_weight` and `permute_tokens` are in one-to-one correspondence. `expert_topk_weight` and `permute_topk_weight` must be passed in together or omitted together.

- Formula:

  $$SrcOffset = \sum_{i=0}^{cur_rank} \left( \sum_{j=0}^{cur_expert} expert_token_num_per_rank(i,j) \right)$$$$

  $$DstOffset = \sum_{j=0}^{cur_expert} \left( \sum_{i=0}^{cur_rank} expert_token_num_per_rank(i,j) \right)$$$$

  $$permute_tokens[DstOffset + k] = tokens[SrcOffset + k]$$

  $$permute_per_token_scales[DstOffset + k] = per_token_scales[SrcOffset + k]$$

  $$permute_topk_weight[DstOffset + k] = expert_topk_weight[SrcOffset + k]$$

  - SrcOffset indicates the source offset of the token to be moved, calculated based on the value of `expert_token_num_per_rank`.
  - DstOffset indicates the destination offset of the token to be moved.
  - cur_rank is the vertical axis index of `expert_token_num_per_rank`, indicating the device where the token originally resides.
  - cur_expert is the horizontal axis index of `expert_token_num_per_rank`, indicating that the token is computed by expert cur_expert on the device.
  - k indicates the offset of the k-th token under the current expert (0 ≤ k < currTokenNum).
  - topkWeight has a one-to-one correspondence with tokens. The copy offsets are identical to those of tokens, directly reusing SrcOffset and DstOffset of tokens.

## Function Prototype

```python
cann_ops_transformer.moe_re_routing(
    tokens: torch.Tensor,
    expert_token_num_per_rank: torch.Tensor,
    *,
    per_token_scales: Optional[torch.Tensor] = None,
    expert_topk_weight: Optional[torch.Tensor] = None,
    expert_token_num_type: int = 1,
    idx_type: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
```

## Mandatory

> [!NOTE]
> Variables used in the tensor shape:
>
> - A: indicates the number of tokens. The value must meet Sum(expert_token_num_per_rank)=A.
> - H: indicates the token length. The value must meet 0 < H < 16384.
> - N: indicates the number of devices. No restriction on the value.
> - E: indicates the number of experts on each device. No restriction on the value.

|  Name   | Type |  Type |    Description    |     Data Type    |   Dimension (Shape)   |
|---------|----------|----------|---------------|---------------|------------|
|  tokens  |  Tensor |  Required  |  Indicates the tokens to be re-arranged. Must be 2D with shape [A, H].  |   float16, bfloat16, int8, float8_e5m2, float8_e4m3fn, hifloat8, float4_e2m1, float4_e1m2   |  2  |
|  expert_token_num_per_rank  |  Tensor |  Required  |  2D matrix. Element [i, j] indicates the number of tokens obtained from device i and processed by expert j on the current device. Must be 2D with shape [N, E]. The value must be greater than 0.  |  int32, int64   | 2  |
|  per_token_scales  |  Tensor |  Optional  |  Indicates the scale corresponding to each token, which needs to be re-arranged along with the tokens. The default value is None. If not input, scale is not used and the values in output `permute_per_token_scales` are meaningless. Supports 1D shape [A] (`float32`), 2D shape [A, S] (`float32`), and 3D shape [A, K/64, 2] (`float8_e8m0`, used for FP8 quantized tokens).  |  float32, float8_e8m0   | 1-3  |
|  expert_topk_weight  |  Tensor |  Optional  |  Indicates the topk weight value corresponding to each token, which needs to be re-arranged along with the tokens. The default value is None. If not input, `permute_topk_weight` is not output. Must be 2D with shape [A, 1]. Must be passed in or not passed in together with `permute_topk_weight`.  |  float32   | 2  |
|  expert_token_num_type  |  int |  Optional  |  Indicates the mode of output `expert_token_num`. The default value is 1. 0 indicates the cumsum mode, and 1 indicates the count mode. Currently, only 1 is supported.  |  -  | - |
|  idx_type  |  int |  Optional  |  Indicates the index type of output `permute_token_idx`. The default value is 0. 0 indicates the gather index, and 1 indicates the scatter index.  |  -  | - |

## Return Values

|  Name   | Type |  Type |    Description    |     Data Type    |   Dimension (Shape)   |
|---------|----------|----------|---------------|---------------|------------|
|  permute_tokens  |  Tensor |  Required  |  Indicates the re-arranged tokens. The shape is [A, H] and the data type is the same as that of `tokens`.  |  Same as tokens  | 2  |
|  permute_per_token_scales  |  Tensor |  Required  |  Indicates the re-arranged `per_token_scales`. If `per_token_scales` is input, the shape and data type are the same as those of `per_token_scales`. If `per_token_scales` is not input, the shape is [A], the data type is `float32`, and the output is meaningless.  |  float32, float8_e8m0  | 1-3  |
|  permute_token_idx  |  Tensor |  Required  |  Indicates each token's index in the original arrangement. The shape is [A].  |  int32   | 1  |
|  expert_token_num  |  Tensor |  Required  |  Indicates the number of tokens processed by each expert. The shape is [E] and the data type is the same as that of `expert_token_num_per_rank`.  |  int32, int64   | 1  |
|  permute_topk_weight  |  Tensor |  Optional  |  Indicates the re-arranged `expert_topk_weight`, which corresponds one-to-one with `permute_tokens`. If `expert_topk_weight` is input, this output is required, with the shape [A, 1]. If `expert_topk_weight` is not input, the output is an empty tensor with the shape (0,).  |  float32   | 2  |

## Constraints

- This API supports use in inference scenarios.
- This API supports calls in single-operator mode and TorchAir graph mode.
- Currently, `expert_token_num_type` supports only 1 (count mode).

## Deterministic Computation

Defaults to a deterministic implementation.

## Example

- Single-operator mode call:

  ```python
  import torch
  import torch_npu
  import cann_ops_transformer

  tokens_num = 16384
  tokens_length = 7168
  rank_num = 16
  expert_num = 16

  tokens = torch.randint(low=-10, high=20, size=(tokens_num, tokens_length), dtype=torch.int8).npu()
  expert_token_num_per_rank = torch.ones(rank_num, expert_num, dtype=torch.int32).npu()
  per_token_scales = torch.randn(tokens_num, dtype=torch.float32).npu()

  # If expert_topk_weight is not passed in, permute_topk_weight is an empty tensor.
  permute_tokens, permute_per_token_scales, permute_token_idx, expert_token_num, permute_topk_weight = \
      cann_ops_transformer.moe_re_routing(tokens, expert_token_num_per_rank,
                                           per_token_scales=per_token_scales,
                                           expert_token_num_type=1, idx_type=0)

  # If expert_topk_weight is passed in, permute_topk_weight is valid data.
  expert_topk_weight = torch.randn(tokens_num, 1, dtype=torch.float32).npu()
  permute_tokens, permute_per_token_scales, permute_token_idx, expert_token_num, permute_topk_weight = \
      cann_ops_transformer.moe_re_routing(tokens, expert_token_num_per_rank,
                                           per_token_scales=per_token_scales,
                                           expert_topk_weight=expert_topk_weight,
                                           expert_token_num_type=1, idx_type=0)
  ```

- TorchAir graph mode call:

  ```python
  import torch
  import torch.nn as nn
  import torch_npu
  import torchair as tng
  from torchair.configs.compiler_config import CompilerConfig
  import cann_ops_transformer

  config = CompilerConfig()
  config.experimental_config.keep_inference_input_mutations = True
  npu_backend = tng.get_npu_backend(compiler_config=config)

  class MoeReRoutingModel(nn.Module):
      def __init__(self):
          super().__init__()

      def forward(self, tokens, expert_token_num_per_rank, *,
                   per_token_scales=None, expert_topk_weight=None,
                   expert_token_num_type=1, idx_type=0):
          return cann_ops_transformer.moe_re_routing(tokens, expert_token_num_per_rank,
                                                      per_token_scales=per_token_scales,
                                                      expert_topk_weight=expert_topk_weight,
                                                      expert_token_num_type=expert_token_num_type,
                                                      idx_type=idx_type)

  def main():
      tokens_num = 16384
      tokens_length = 7168
      rank_num = 16
      expert_num = 16

      tokens = torch.randint(low=-10, high=20, size=(tokens_num, tokens_length), dtype=torch.int8).npu()
      expert_token_num_per_rank = torch.ones(rank_num, expert_num, dtype=torch.int32).npu()
      per_token_scales = torch.randn(tokens_num, dtype=torch.float32).npu()
      expert_topk_weight = torch.randn(tokens_num, 1, dtype=torch.float32).npu()

      model = MoeReRoutingModel().npu()
      model = torch.compile(model, backend=npu_backend, dynamic=False)
      permute_tokens, permute_per_token_scales, permute_token_idx, expert_token_num, permute_topk_weight = \
          model(tokens, expert_token_num_per_rank, per_token_scales=per_token_scales,
                expert_topk_weight=expert_topk_weight, expert_token_num_type=1, idx_type=0)

  if __name__ == '__main__':
      main()
  ```
