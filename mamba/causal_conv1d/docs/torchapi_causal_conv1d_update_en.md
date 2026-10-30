# causal_conv1d_update

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T12:43:03.329Z -->

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

    Causal 1D convolution state update (decode/update), encapsulating aclnnCausalConv1dUpdate. Uses cached data to pad the head of each sequence along the sequence dimension, ensuring that the output depends on both current and historical inputs. Optionally applies SiLU activation after the convolution. After the computation, updates part of the current sequence data to the cache. Supports features such as cache index (cacheIndices) and speculative decoding (numAcceptedTokens).

- **Formula**:

    At each time step $t$, the convolution output $y_t$ is computed based on the current input $x_t$, the convolution weight $w$, and the historical state:

    $$
    y_t = \text{Activation}\left(\sum_{j=0}^{W-1} w_j \cdot x_{t-j} + b\right)
    $$

    where $W$ is the kernel width (2, 3, and 4 are supported), $w_j$ is the convolution weight, $b$ is the bias, and $\text{Activation}$ is the activation function (SiLU or no activation).

- **NOTE**

  - The operator also maintains the convolution state `conv_state` to cache historical inputs during incremental inference, enabling efficient state updates.
  - The mode is automatically inferred from the input shape: decode/update mode when x is 3D and seq_len == 1; 2D variable-length input is supported only when num_accepted_tokens (speculative decoding) is provided.

## Function Prototype

```python
cann_ops_transformer.causal_conv1d_update(
    x, conv_state, weight,
    bias=None, activation="silu",
    conv_state_indices=None,
    num_accepted_tokens=None,
    query_start_loc=None,
    max_query_len=-1, null_block_id=0,
    block_idx_last_scheduled_token=None,
    initial_state_idx=None,
) -> Tensor
```

## Mandatory

| Name | Type | Type | Description | Data Type | Dimension (Shape) |
|--------|----------|-----------|------|----------|-------------|
| x | Tensor | Required | Input sequence. | float16, bfloat16 | [batch, 1, dim] (fixed batch, 3D). 2D [cu_seq_len, dim] is supported only when num_accepted_tokens (speculative decoding) is provided. |
| conv_state | Tensor | Required | Convolution state cache, updated in place after computation. | same as x | [num_cache_lines, state_len, dim], state_len ≥ kW-1 |
| weight | Tensor | Required | Convolution weight. | same as x | [kW, dim], kW∈{2, 3, 4} |
| bias | Tensor | Optional | Convolution bias. Default None means no bias is used. | same as x | [dim] |
| activation | str | Optional | Activation function type, "silu" or "none". Default value is "silu". | - | - |
| conv_state_indices | Tensor | Optional | Cache index. Specifies the index of the cache state corresponding to each sequence in conv_state. Default None uses identity mapping. | int32 | [batch] |
| num_accepted_tokens | Tensor | Optional | Number of accepted tokens per batch in speculative decoding. Supported only when kW=4. Default None. | int32 | [batch] |
| query_start_loc | Tensor | Optional | Start position index of variable-length sequences. Default None. If provided, the batch size is determined by query_start_loc.size(0)-1 instead of being inferred from the shape of x. | int32 | [batch+1] |
| max_query_len | int | Optional | Maximum query length. -1 means no limit. Default value is -1. | - | - |
| null_block_id | int | Optional | ID marking an invalid cache slot. If conv_state_indices[i]==null_block_id, the sequence is skipped and the output is zero-filled. Default value is 0. | - | - |
| block_idx_last_scheduled_token | Tensor | Optional | Block index of the last scheduled token. | int32 | [batch] |
| initial_state_idx | Tensor | Optional | Initial state index. | int32 | [batch] |

## Return Values

Returns the convolution output Tensor y. The shape is the same as x, and the dtype is the same as x.

## Constraints

- This API is supported only in inference use cases.
- This API supports both single-operator mode and graph mode calls.
- Non-contiguous tensors are not supported.
- kW of `weight` supports only 2, 3, and 4.
- For `conv_state`, state_len must be ≥ kW-1, and num_cache_lines must be ≥ batch (when conv_state_indices is not provided).
- `query_start_loc`[0] must be 0, `query_start_loc`[-1] must be equal to cu_seq_len, and the values must be non-decreasing.
- Values of `conv_state_indices` must be in [0, num_cache_lines), or equal to null_block_id to skip the sequence.
- `num_accepted_tokens` is supported only when kW=4. The value must be in [0, state_len - (kW - 1)], where state_len is conv_state.shape[1]. Both 0 and 1 correspond to offset=0 (reading from the start of conv_state). Values beyond this range cause out-of-bounds access to conv_state.
- The operator input and intermediate computation results do not exceed the value range of the corresponding data type (float16/bfloat16).
- Operator input with ±inf or nan is not supported.

## Calling Instructions

- Calling in single-operator mode

    ```python
    import torch
    import torch_npu
    from cann_ops_transformer.ops import causal_conv1d_update

    torch_npu.npu.set_device(0)

    B = 4
    D = 512
    kW = 2

    x = torch.randn(B, 1, D, device="npu", dtype=torch.float16)
    weight = torch.randn(kW, D, device="npu", dtype=torch.float16)
    conv_state = torch.zeros(B, kW - 1, D, device="npu", dtype=torch.float16)

    y = causal_conv1d_update(
        x, conv_state, weight,
        activation="silu",
    )
    ```

- Calling in graph mode

    The operator is automatically converted to a GE graph operator through `torch.compile(backend="atc")` or `torchair`. No additional configuration is required.

    ```python
    import torch
    import torch_npu
    import torchair
    from cann_ops_transformer.ops import causal_conv1d_update

    torch_npu.npu.set_device(0)

    B = 4
    D = 512
    kW = 2

    class CausalConv1dUpdateModel(torch.nn.Module):
        def forward(self, x, conv_state, weight):
            y = causal_conv1d_update(
                x, conv_state, weight,
                activation="silu",
            )
            return y

    model = CausalConv1dUpdateModel().npu()
    npu_backend = torchair.get_npu_backend()
    model = torch.compile(model, backend=npu_backend, dynamic=False)

    x = torch.randn(B, 1, D, device="npu", dtype=torch.float16)
    weight = torch.randn(kW, D, device="npu", dtype=torch.float16)
    conv_state = torch.zeros(B, kW - 1, D, device="npu", dtype=torch.float16)

    output = model(x, conv_state, weight)
    ```
