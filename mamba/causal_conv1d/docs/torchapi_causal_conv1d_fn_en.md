# causal_conv1d_fn

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T12:42:16.000Z -->

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

    Causal 1D convolution forward computation (prefill/chunk-prefill), encapsulating aclnnCausalConv1dFn. Along the sequence dimension, cache data (length: kernel width minus 1) is used to pad the head of each sequence, ensuring that the output depends on the current and historical inputs. SiLU activation is optionally applied after the convolution. After the computation, part of the current sequence data is updated to the cache. Features such as cache index (cacheIndices) and initial state mode (initialStateMode) are supported.

- **Formula**:

    At each time step $t$, the convolution output $y_t$ is computed based on the current input $x_t$, the convolution weight $w$, and the historical state:

    $$
    y_t = \text{Activation}\left(\sum_{j=0}^{W-1} w_j \cdot x_{t-j} + b\right)
    $$

    where $W$ is the convolution kernel width (2, 3, and 4 are supported), $w_j$ is the convolution weight, $b$ is the bias, and $\text{Activation}$ is the activation function (SiLU or no activation).

- **NOTE**

  - The operator also maintains the convolution state `conv_states` to cache historical inputs during incremental inference, enabling efficient state updates.
  - The mode is automatically inferred from the input shape: prefill mode when x is 3D and seq_len > 1.

## Function Prototype

```python
cann_ops_transformer.causal_conv1d_fn(
    x, weight, bias, conv_states, query_start_loc,
    *,
    cache_indices=None, has_initial_state=None,
    activation="silu", pad_slot_id=-1, null_block_id=0,
    block_idx_first_scheduled_token=None,
    block_idx_last_scheduled_token=None,
    initial_state_idx=None, num_computed_tokens=None,
    block_size_to_align=0,
) -> Tensor
```

## Mandatory

| Name | Type | Type | Description | Data Type | Dimension (Shape) |
|--------|----------|-----------|------|----------|-------------|
| x | Tensor | Required | Input sequence. | float16, bfloat16 | [batch, seq_len, dim] (fixed batch) or [cu_seq_len, dim] (variable length) |
| weight | Tensor | Required | Convolution weight. | Same as x | [kW, dim], kW∈{2, 3, 4} |
| bias | Tensor | Required | Convolution bias. None indicates that no bias is used. | Same as x | [dim] |
| conv_states | Tensor | Required | Convolution state cache, updated in place after computation. | Same as x | [num_cache_lines, state_len, dim], state_len ≥ kW-1 |
| query_start_loc | Tensor | Required | Start index of variable-length sequences. Required for variable-length use cases; an empty Tensor can be passed for fixed-batch use cases. | int32 | [batch+1] |
| cache_indices | Tensor | Optional | Cache indices, specifying the index of the cache state in conv_states for each sequence. The default value is None, which uses the identity mapping. | int32 | [batch] |
| has_initial_state | Tensor | Optional | Initial state flag. 1 = use cached history, 0 = zero initialization (cold start). The default value is None, which means zero initialization for all sequences. | int32 | [batch] |
| activation | str | Optional | Activation function type, "silu" or "none". The default value is "silu". | - | - |
| pad_slot_id | int | Optional | Padding slot ID. The default value is -1. | - | - |
| null_block_id | int | Optional | ID of an invalid cache slot. If cache_indices[i]==null_block_id, the sequence is skipped and the output is filled with zeros. The default value is 0. | - | - |
| block_idx_first_scheduled_token | Tensor | Optional | Index of the first scheduled token block. | int32 | [batch] |
| block_idx_last_scheduled_token | Tensor | Optional | Index of the last scheduled token block. | int32 | [batch] |
| initial_state_idx | Tensor | Optional | Initial state index. | int32 | [batch] |
| num_computed_tokens | Tensor | Optional | Number of computed tokens for each sequence. | int32 | [batch] |
| block_size_to_align | int | Optional | Block size for cache state alignment. The default value is 0. | - | - |

## Return Values

Returns the convolution output tensor y, with the same shape as x and the same data type as x.

## Constraints

- This API is supported only in inference use cases.
- This API supports calls in single-operator mode and graph mode.
- Non-contiguous tensors are not supported.
- kW of `weight` supports only 2, 3, and 4.
- state_len of `conv_states` must be ≥ kW-1, and num_cache_lines must be ≥ batch (when cache_indices is not provided).
- `query_start_loc`[0] must be 0, `query_start_loc`[-1] must be equal to cu_seq_len, and the values must be non-decreasing.
- Values of `cache_indices` must be in [0, num_cache_lines), or equal to null_block_id, indicating that the sequence is skipped.
- Operator inputs and intermediate computation results do not exceed the value range of the corresponding data type (float16/bfloat16).
- Operator inputs containing ±inf or nan are not supported.

## API Call

- Call in single-operator mode

    ```python
    import torch
    import torch_npu
    from cann_ops_transformer.ops import causal_conv1d_fn

    torch_npu.npu.set_device(0)

    B = 2
    S = 16
    D = 512
    kW = 4

    x = torch.randn(B, S, D, device="npu", dtype=torch.float16)
    weight = torch.randn(kW, D, device="npu", dtype=torch.float16)
    bias = torch.randn(D, device="npu", dtype=torch.float16)
    conv_states = torch.zeros(B, kW - 1, D, device="npu", dtype=torch.float16)
    query_start_loc = torch.tensor([0, S, 2 * S], device="npu", dtype=torch.int32)

    y = causal_conv1d_fn(
        x, weight, bias, conv_states, query_start_loc,
        activation="silu",
    )
    ```

- Call in graph mode

    The operator is automatically converted to a GE graph operator through `torch.compile(backend="atc")` or `torchair`. No additional configuration is required.

    ```python
    import torch
    import torch_npu
    import torchair
    from cann_ops_transformer.ops import causal_conv1d_fn

    torch_npu.npu.set_device(0)

    B = 2
    S = 16
    D = 512
    kW = 4

    class CausalConv1dFnModel(torch.nn.Module):
        def forward(self, x, weight, bias, conv_states, query_start_loc):
            y = causal_conv1d_fn(
                x, weight, bias, conv_states, query_start_loc,
                activation="silu",
            )
            return y

    model = CausalConv1dFnModel().npu()
    npu_backend = torchair.get_npu_backend()
    model = torch.compile(model, backend=npu_backend, dynamic=False)

    x = torch.randn(B, S, D, device="npu", dtype=torch.float16)
    weight = torch.randn(kW, D, device="npu", dtype=torch.float16)
    bias = torch.randn(D, device="npu", dtype=torch.float16)
    conv_states = torch.zeros(B, kW - 1, D, device="npu", dtype=torch.float16)
    query_start_loc = torch.tensor([0, S, 2 * S], device="npu", dtype=torch.int32)

    output = model(x, weight, bias, conv_states, query_start_loc)
    ```
