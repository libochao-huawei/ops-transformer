# block_attention_residuals

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
- <term>Atlas 200I/500 A2 inference products</term>: Not Supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not Supported
<!-- end id6 -->

## Function

- **Description**: Concatenates `partial_block` and `block_res` along the block dimension, then performs RMS normalization, projection scoring with `norm_weight ⊙ proj_weight`, Softmax weighted fusion, and outputs `hidden_states`.

- **Formula**:

    $$
    \operatorname{block\_attention\_residuals}
    \left(partial\_block, block\_res, proj\_weight, norm\_weight; valid\_block\_num, norm\_eps\right)
    \longrightarrow hidden\_states.
    $$

    Let the number of tokens be $T$, the number of blocks be $N$, the hidden size be $H$, and the logical number of rows after concatenation be $B = N + 1$. The row order is consistent with the golden result: the first $N$ rows come from `block_res`, and the last row comes from `partial_block`.

    $$
    v_{t,i,h} =
    \begin{cases}
    block\_res_{t,i,h}, & 0 \le i < N \\
    partial\_block_{t,h}, & i = N
    \end{cases}
    $$

    $$
    variance_{t,i} = \frac{1}{H}\sum_{h=0}^{H-1} v_{t,i,h}^{2}
    $$

    $$
    inv\_rms_{t,i} = \frac{1}{\sqrt{variance_{t,i} + norm\_eps}}
    $$

    $$
    k_{t,i,h} = v_{t,i,h} \cdot inv\_rms_{t,i}
    $$

    $$
    score\_weight_{h} = norm\_weight_{h} \cdot proj\_weight_{0,h}
    $$

    $$
    s_{t,i} = \sum_{h=0}^{H-1} k_{t,i,h} \cdot score\_weight_{h}
    $$

    $$
    probs_{t,i} = \frac{e^{s_{t,i}}}{\sum_{j=0}^{N} e^{s_{t,j}}}
    $$

    $$
    hidden\_states_{t,h} = \sum_{i=0}^{N} probs_{t,i} \cdot v_{t,i,h}
    $$

    The weighted fusion uses the **original** $v$ (not the RMS-normalized $k$). The Torch API currently returns only $hidden\_states$.

    Where:

    - $partial\_block \in \mathbb{R}^{T \times H}$ represents the input `partial_block`, used as the $N$th row value after concatenation.
    - $block\_res \in \mathbb{R}^{T \times N \times H}$ represents the input `block_res`, used as the first $N$ rows value after concatenation.
    - $proj\_weight$ represents the input `proj_weight`, with shape $[1, H]$; $proj\_weight_{0,h}$ indicates its $h$th element.
    - $norm\_weight \in \mathbb{R}^{H}$ represents the input `norm_weight`.
    - $valid\_block\_num$ represents the input `valid_block_num`. It is an optional reserved attribute. The default value is `-1`. Currently only `-1` (meaning use `N`) or `N` (`block_res.shape[1]`) is supported. It does not participate in computation.
    - $norm\_eps$ represents the input `norm_eps`, which is the numerical stability term used in computing the RMS normalization factor.
    - $t \in [0, T)$, $i \in [0, N]$, $h \in [0, H)$ represent the indices of token, logical row, and hidden dimension respectively.
    - $v_{t,i,h}$ represents the concatenated value. $inv\_rms_{t,i}$ represents the per-row RMS normalization coefficient. $score\_weight_{h}$ represents the element-wise product of `norm_weight` and `proj_weight`.
    - $s_{t,i}$ represents the projection score. $probs_{t,i}$ represents the Softmax probability. $hidden\_states_{t,h}$ represents the weighted fusion output.
    - $\sum$, $\exp$ represent summation and natural exponential operation respectively, and $\mathbb{R}$ represents the real number domain.

## Prototype

```python
cann_ops_transformer.block_attention_residuals(
    partial_block,
    block_res,
    proj_weight,
    norm_weight,
    valid_block_num=-1,
    norm_eps=1.0e-6,
) -> Tensor
```

## Parameters

| Parameter Name | Parameter Type | Optional/Required | Description                                                                                        | Data Type                  | Shape    |
| -------------- | -------------- | ----------------- | -------------------------------------------------------------------------------------------------- | -------------------------- | -------- |
| partial_block  | Tensor         | Required          | The $N$th row value after concatenation, corresponding to $partial\_block$ in the formula. Data format is ND. | float16, bfloat16, float32 | [T, H]   |
| block_res      | Tensor         | Required          | The first $N$ rows value, corresponding to $block\_res$ in the formula. Data format is ND. dtype must be the same as `partial_block`. | Same as partial_block      | [T, N, H] |
| proj_weight    | Tensor         | Required          | Projection weight, together with `norm_weight` forming $score\_weight$. Data format is ND. dtype must be the same as `partial_block`. | Same as partial_block      | [1, H]   |
| norm_weight    | Tensor         | Required          | RMS scaling weight. Data format is ND. dtype must be the same as `partial_block`.                   | Same as partial_block      | [H]      |
| valid_block_num| int            | Optional          | Reserved attribute. Default value is `-1`. Currently only `-1` (meaning use `N`) or `N` (`block_res.shape[1]`) is supported. Does not participate in computation. | int                        | -        |
| norm_eps       | float          | Optional          | Numerical stability term for RMS normalization. Must be a finite positive number. Default value is `1.0e-6`. | float                      | -        |

## Return Value Description

Returns `hidden_states`.

| Parameter Name | Parameter Type | Optional/Required | Description                                                                               | Data Type             | Shape |
| -------------- | -------------- | ----------------- | ----------------------------------------------------------------------------------------- | --------------------- | ----- |
| hidden_states  | Tensor         | Required          | Weighted fusion result, corresponding to $hidden\_states$ in the formula. Data format is ND. | Same as partial_block | [T, H] |

## Constraints

- This API supports single operator mode. TorchAir graph mode is not supported yet.
- The dtype of the main path must be consistent. Supports float16 / bfloat16 / float32.
- Shape requirements: `T >= 0`, `H >= 0`, `1 <= N <= 100`. When `T == 0` or `H == 0`, returns an empty output with the corresponding shape.
- `block_res.shape[0]` and `block_res.shape[2]` must equal $T$ and $H$ of `partial_block` respectively. The last dimension of `proj_weight` and the length of `norm_weight` must equal $H$.
- `valid_block_num` is an optional reserved attribute. Default value is `-1`. Currently only `-1` (meaning use `N`) or `N` (`block_res.shape[1]`) is supported. Does not participate in computation.
- `norm_eps` must be a finite positive number.
- If `partial_block` or `block_res` contains `NaN` or `Inf`, or overflow occurs during square, multiplication, or cumulative addition, the result may contain `NaN` or `Inf`.

## Deterministic Computation/Batch Consistency

- Deterministic computation: Deterministic computation is supported by default.

- Batch consistency: Batch-consistent implementation by default.

## Example

- Single operator mode call:

    ```python
    import torch
    import torch_npu
    import cann_ops_transformer

    torch_npu.npu.set_device(0)

    partial_block = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0]],
        dtype=torch.bfloat16,
        device="npu",
    )
    block_res = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 0.0, -1.0, 1.0],
            ]
        ],
        dtype=torch.bfloat16,
        device="npu",
    )
    proj_weight = torch.tensor([[0.5, 0.25, 0.25, 0.0]], dtype=torch.bfloat16, device="npu")
    norm_weight = torch.ones(4, dtype=torch.bfloat16, device="npu")

    # valid_block_num / norm_eps are optional parameters with default values of -1 and 1e-6 respectively.
    # Only hidden_states is returned.
    hidden_states = cann_ops_transformer.block_attention_residuals(
        partial_block,
        block_res,
        proj_weight,
        norm_weight,
    )
    torch_npu.npu.synchronize()

    print("hidden_states:", hidden_states)
    ```

- TorchAir graph mode call: Not supported yet.
