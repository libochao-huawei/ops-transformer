# lightning\_indexer\_kl\_loss

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T11:01:44.242Z -->

## Applicable Products

| Product | Supported |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | × |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

- Description

  - **teacher side** (target_score): Raw unnormalized main attention scores from the teacher side for the compressed segment, with sum ≠ 1. Use `clamp_min` to prevent NaN caused by log(0) when target_score=0.
  - **student side** (index_probs): Probability distribution after indexer softmax. Use `+eps` to preserve gradients when index_probs→0.

- Formula:

  $$
  y = \text{target\_score}, \quad Y = \text{index\_probs}
  $$

  $$
  P = \frac{y}{\text{sum}(y, \text{dim}=-1, \text{keepdim=True}) + \varepsilon}
  $$

  $$
  \log\_P = \log(\text{clamp\_min}(\tilde{y}, \varepsilon))
  $$

  $$
  \log\_Y = \log(Y + \varepsilon)
  $$

  $$
  \text{loss} = \sum((\log\_P - \log\_Y) \cdot \text{weight})
  $$

  Here, $\varepsilon$ is the `eps` parameter, with a default value of $10^{-9}$.

  The selection of weight is controlled by `weight_type`:

  - `'logits'` (default): weight = y, i.e., the raw unnormalized score
  - `'probs'`: weight = P, i.e., the normalized probability

## Function Prototype

```python
from cann_ops_transformer import lightning_indexer_kl_loss

lightning_indexer_kl_loss(
    target_score,
    index_probs,
    eps=1e-9,
    weight_type='logits'
) -> Tensor
```

## Parameters

| Name | Input/Output | Description | Data Type | Dimension |
| :--- | :--- | :--- | :--- | :--- |
| target_score | Required input | Raw unnormalized main attention scores from the teacher side for the compressed segment, with sum != 1. | `float16`, `bfloat16`, `float32` | `(B, S, K)` or `(T, K)` |
| index_probs | Required input | Probability distribution after indexer softmax on the student side. | Consistent with `target_score` | Consistent with `target_score` |
| eps | Optional attribute | Numerical stability constant, with a default value of 1e-9. | `float` | - |
| weight_type | Optional attribute | Outer weight selection. `'logits'` (default) uses the raw target_score as the outer weight; `'probs'` uses the normalized probability P = target_score / sum(target_score) as the outer weight. | `str` | - |

## Return Values

| Output | Description | Data Type | Dimension |
| :--- | :--- | :--- | :--- |
| loss | KL divergence scalar loss. | Consistent with input | `(1,), scalar`|

## Constraints

- The data types of `target_score` and `index_probs` must be consistent, and their shapes must be consistent.
- Supported shapes are (B, S, K) or (T, K), where B is in the range 1\~512, and the last dimension K is in the range 1\~8192.
- `eps` must be greater than 0.
- `weight_type` must be `'logits'` or `'probs'`.
- `target_score` and `index_probs` must be on the same NPU device.

## Deterministic Computation

- Deterministic computation is not supported by default; it can be enabled via the PyTorch switch (torch.use_deterministic_algorithms).

## Example

### Single Operator Call (Default logits Weight)

```python
import torch
import torch_npu
from cann_ops_transformer import lightning_indexer_kl_loss

torch_npu.npu.set_device(0)
device = torch.device("npu:0")

B, H, D = 4, 10, 128

target_score = torch.rand(B, H, D, dtype=torch.float32, device=device)
index_probs = torch.softmax(torch.randn(B, H, D, dtype=torch.float32, device=device), dim=-1)

# Use logits weight (default), equivalent to loss = sum((log_P - log_Y) * y)
loss = lightning_indexer_kl_loss(target_score, index_probs)
torch.npu.synchronize()
print(f"KL loss (logits): {loss.item()}")
```
