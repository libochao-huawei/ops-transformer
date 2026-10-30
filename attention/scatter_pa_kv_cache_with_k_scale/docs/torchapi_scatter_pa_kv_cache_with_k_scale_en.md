# scatter\_pa\_kv\_cache\_with\_k\_scale

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T13:02:23.899Z -->

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
<!-- npu="310p" id8 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id8 -->
<!-- npu="910" id9 -->
- <term>Atlas training products</term>: Not supported
<!-- end id9 -->

## Function

- **Description**:

  `scatter_pa_kv_cache_with_k_scale` is a `cann_ops_transformer` extension API based on `torch_npu`. It calls the `ScatterPaKvCacheWithKScale` operator to update the KV Cache of FP8-format key/value and the corresponding key_scale in PagedAttention use cases.

- **Formula**:

  For each token (i ∈ [0, num_tokens)) and each head (j ∈ [0, num_head)):

  $$
  block\_idx = \lfloor slot\_mapping[i] / block\_size \rfloor
  $$

  $$
  block\_offset = slot\_mapping[i] \bmod block\_size
  $$

  $$
  key\_cache[block\_idx][j][block\_offset][:] = key[i][j][:]
  $$

  $$
  value\_cache[block\_idx][j][block\_offset][:] = value[i][j][:]
  $$

  $$
  key\_scale\_cache[block\_idx][j][block\_offset][0] = key\_scale[i][j]
  $$

  Where:

  - num_tokens = batch * seq_len
  - block_idx: Index of the block to which slot_mapping is mapped
  - block_offset: Offset within a block

> [!NOTE]
>
> B (Batch) indicates the batch size of input samples, S (Seq-Length) indicates the sequence length of input samples, N (Head-Num) indicates the number of attention heads, and D (Head-Dim) indicates head_dim, the minimum unit size of the hidden layer. num_tokens = B × S.
> num_blocks indicates the total number of KV cache blocks, block_size indicates the number of tokens in each block, and num_slots = num_blocks × block_size indicates the total number of tokens that the cache can hold.

## Function Prototype

```python
cann_ops_transformer.scatter_pa_kv_cache_with_k_scale(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    key_scale,
    key_scale_cache,
    *,
    cache_layout='BNBD'
) -> None
```

## Parameters

| Name | Type | Required/Optional | Description | Data Type | Dimension (Shape) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| key | Tensor | Required | Key values to be updated, namely the keys of multiple tokens in the current step. Empty tensors are not supported. | float8_e5m2, float8_e4m3fn | (num_tokens, num_head, k_head_size) |
| value | Tensor | Required | Value values to be updated, namely the values of multiple tokens in the current step. Empty tensors are not supported. | float8_e5m2, float8_e4m3fn | (num_tokens, num_head, v_head_size) |
| key_cache | Tensor | Required | Key cache that needs update, namely the key cache of the current layer. Empty tensors are not supported. | Same as key | (num_blocks, num_head, block_size, k_head_size) |
| value_cache | Tensor | Required | Value cache that needs update, namely the value cache of the current layer. Empty tensors are not supported. | Same as value | (num_blocks, num_head, block_size, v_head_size) |
| slot_mapping | Tensor | Required | Storage offset of each token key or value in the cache. Empty tensors are not supported. | int32, int64 | (num_tokens,) |
| key_scale | Tensor | Required | Key scale values to be updated, namely the key scale of multiple tokens in the current step. The trailing axis can be non-contiguous. Empty tensors are not supported. | float | (num_tokens, num_head) |
| key_scale_cache | Tensor | Required | Key scale cache that needs update, namely the key scale cache of the current layer. The last dimension is 1, and the trailing axis must be contiguous. Empty tensors are not supported. | float | (num_blocks, num_head, block_size, 1) |
| cache_layout | str | Optional | Indicates the memory layout of key_cache and value_cache. If "BNBD" is passed, the format is [num_blocks, num_head, block_size, head_size]. Default value: "BNBD". | STRING | - |

## Return Values

No return value. This API updates data in place. After the call, `key_cache`, `value_cache`, and `key_scale_cache` are updated in place, so there is no need to receive a return value.

## Constraints

- Declaration: The slot_mapping parameter is a tensor. The operator cannot obtain the actual values of the tensor in the Tiling phase, so the tiling side does not validate the values. You must ensure the correctness. If an invalid value is passed in, undefined behavior will occur (such as precision issues or program crashes caused by invalid memory access).
- This API supports use in inference scenarios.
- This API can be called in single-operator mode and graph mode (torchair).
- The data types of key, value, key_cache, and value_cache must be the same and must be float8_e5m2 or float8_e4m3fn.
- The data type of key_scale and key_scale_cache must be float.
- The shapes of the first two dimensions of key and value must be the same.
- The value range of slot_mapping is \[0, num_blocks\*block_size-1\]. The elements in slot_mapping are guaranteed to be unique. Correctness is not guaranteed if duplicates exist.
- key_scale is a two-dimensional tensor with the shape \[num_tokens, num_head\]. The last dimension can be non-contiguous.
- key_scale_cache is a four-dimensional tensor with the shape \[num_blocks, num_head, block_size, 1\]. The last dimension must be 1, and the last dimension must be contiguous.
- num_tokens indicates the number of tokens to be updated to the cache, and num_tokens = batch * seq_len.
- num_blocks indicates the total number of KV cache blocks, and block_size indicates the number of tokens in each block.
- num_head indicates the number of attention heads, and k_head_size and v_head_size indicate the head dimensions of key and value respectively.

## Deterministic Computation

Defaults to a deterministic implementation.

## Example

- Single-operator mode call

    ```python
    import torch
    import torch_npu
    import cann_ops_transformer

    torch_npu.npu.set_device(0)

    # Shape definition
    num_tokens = 4       # Number of tokens to write this time
    num_head = 8         # Number of attention heads
    k_head_size = 128    # Head dimension of key
    v_head_size = 128    # Head dimension of value
    num_blocks = 2       # Total number of KV cache blocks
    block_size = 16      # Number of tokens in each block

    # FP8 dtype (both float8_e5m2 and float8_e4m3fn are supported; e4m3fn is used as an example here)
    kv_dtype = torch.float8_e4m3fn

    # Construct inputs: key/value are the new data to be written
    key = torch.randn(num_tokens, num_head, k_head_size, dtype=torch.float32, device="npu").to(kv_dtype)
    value = torch.randn(num_tokens, num_head, v_head_size, dtype=torch.float32, device="npu").to(kv_dtype)

    # Construct the KV cache (target of the inplace update, initialized to 0 for easy verification)
    key_cache = torch.zeros(num_blocks, num_head, block_size, k_head_size, dtype=kv_dtype, device="npu")
    value_cache = torch.zeros(num_blocks, num_head, block_size, v_head_size, dtype=kv_dtype, device="npu")

    # Construct slot_mapping: the offset of each token in the cache, value range [0, num_blocks*block_size-1]
    slot_mapping = torch.tensor([0, 1, 16, 17], dtype=torch.int32, device="npu")

    # Construct key_scale and its cache (FP8 dequantization scale per token-head)
    key_scale = torch.randn(num_tokens, num_head, dtype=torch.float32, device="npu")
    key_scale_cache = torch.zeros(num_blocks, num_head, block_size, 1, dtype=torch.float32, device="npu")

    # Call the operator to write key/value/key_scale into the cache based on slot_mapping (in-place update, no return value)
    cann_ops_transformer.scatter_pa_kv_cache_with_k_scale(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        key_scale,
        key_scale_cache,
        cache_layout='BNBD',
    )

    torch_npu.npu.synchronize()
    print(key_cache.shape, key_cache.dtype)
    print(value_cache.shape, value_cache.dtype)
    print(key_scale_cache.shape, key_scale_cache.dtype)
    ```

- Graph mode (torchair) call:

    ```python
    import torch
    import torch_npu
    import torch.nn as nn
    import torchair
    from torchair.configs.compiler_config import CompilerConfig
    import cann_ops_transformer

    torch_npu.npu.set_device(0)

    # Shape definition
    num_tokens = 4       # Number of tokens to be written
    num_head = 8         # Number of attention heads
    k_head_size = 128    # Key head dimension
    v_head_size = 128    # value head dimension
    num_blocks = 2       # Total number of KV cache blocks
    block_size = 16      # Number of tokens per block

    # FP8 dtype (both float8_e5m2 and float8_e4m3fn are supported; e4m3fn is used as an example here)
    kv_dtype = torch.float8_e4m3fn

    # Construct inputs: key/value are the new data to be written
    key = torch.randn(num_tokens, num_head, k_head_size, dtype=torch.float32, device="npu").to(kv_dtype)
    value = torch.randn(num_tokens, num_head, v_head_size, dtype=torch.float32, device="npu").to(kv_dtype)

    # Construct the KV cache (target of the inplace update, initialized to 0 for easy verification)
    key_cache = torch.zeros(num_blocks, num_head, block_size, k_head_size, dtype=kv_dtype, device="npu")
    value_cache = torch.zeros(num_blocks, num_head, block_size, v_head_size, dtype=kv_dtype, device="npu")

    # Construct slot_mapping: offset of each token in the cache, value range [0, num_blocks*block_size-1]
    slot_mapping = torch.tensor([0, 1, 16, 17], dtype=torch.int32, device="npu")

    # Construct key_scale and its cache (per-token-head FP8 dequantization scale)
    key_scale = torch.randn(num_tokens, num_head, dtype=torch.float32, device="npu")
    key_scale_cache = torch.zeros(num_blocks, num_head, block_size, 1, dtype=torch.float32, device="npu")

    class ScatterPaKvCacheWithKScaleNetwork(nn.Module):
        def __init__(self):
            super(ScatterPaKvCacheWithKScaleNetwork, self).__init__()
        @torch._dynamo.disable
        def forward(self, key, value, key_cache, value_cache, slot_mapping, key_scale, key_scale_cache,
                    cache_layout='BNBD'):
            torch.ops.cann_ops_transformer.scatter_pa_kv_cache_with_k_scale(
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
                key_scale,
                key_scale_cache,
                cache_layout=cache_layout,
            )
            return key_cache, value_cache, key_scale_cache

    config = CompilerConfig()
    config.mode = "reduce-overhead"
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    torch._dynamo.reset()
    npu_mode = torch.compile(ScatterPaKvCacheWithKScaleNetwork(), backend=npu_backend, dynamic=False)
    key_cache, value_cache, key_scale_cache = npu_mode(
        key, value, key_cache, value_cache, slot_mapping, key_scale, key_scale_cache, cache_layout='BNBD')

    print(key_cache.shape, key_cache.dtype)
    print(value_cache.shape, value_cache.dtype)
    print(key_scale_cache.shape, key_scale_cache.dtype)
    ```
