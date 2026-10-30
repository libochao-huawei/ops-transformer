# block_attn_res_update

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T08:05:21.745Z -->

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

  Phase 2 of the two-phase historical residual attention computation for Attention Residuals: accumulates `delta` into `partial_block` in place and computes the RMSNorm score of the updated residual; then computes the output `h` together with the softmax statistics `numerator`, `logit_max`, and `exp_sum` (corresponding to O, M, and L, respectively) returned by the first-phase API `block_attn_res_prepare`, and writes the updated residual back to `partial_block` in place.
- **Formula**:

  Let `T` denote the number of tokens and `D` denote the hidden size. The main computation process is as follows:

  1. Update the current `partial_block`:

     $$
     p[t,d] = partial\_block[t,d] + \operatorname{FP32}(delta[t,d])
     $$
  2. Compute the RMSNorm denominator and the current score:

     $$
     r[t] = \sqrt{\frac{1}{D}\sum_{d=0}^{D-1}p[t,d]^2 + eps}
     $$

     $$
     score[t] = \frac{\sum_{d=0}^{D-1}p[t,d] \times pseudo\_query[d]}{r[t]}
     $$
  3. Merge the current score with the historical online softmax states:

     $$
     m[t] = \max(logit\_max[t], score[t])
     $$

     $$
     alpha[t] = \exp(logit\_max[t] - m[t]), \qquad beta[t] = \exp(score[t] - m[t])
     $$

     $$
     ell[t] = exp\_sum[t] \times alpha[t] + beta[t]
     $$

     $$
     updated\_numerator[t,d] = numerator[t,d] \times alpha[t] + p[t,d] \times beta[t]
     $$
  4. Generate the output:

     $$
     partial\_block[t,d] \leftarrow p[t,d]
     $$

     $$
     h[t,d] = \operatorname{Cast}_{delta.dtype}\left(\frac{updated\_numerator[t,d]}{ell[t]}\right)
     $$

## Function Prototype

```python
cann_ops_transformer.block_attn_res_update(
    partial_block,
    delta,
    pseudo_query,
    numerator,
    logit_max,
    exp_sum,
    *,
    eps=1e-6,
) -> Tensor
```

## Parameters

| Name       | Type      | Type | Description                                                | Data Type          | Dimension (Shape) |
| --------------- | --------- | ----------------- | ---------------------------------------------------------- | ------------------ | ----------------- |
| `partial_block` | Tensor    | Required          | Accumulated `partial_block`. Updated in place after the call completes. | `torch.float32`    | `(T, D)`    |
| `delta`         | Tensor    | Required          | `delta` to be accumulated in this call.                    | `torch.bfloat16`   | `(T, D)`    |
| `pseudo_query`  | Tensor    | Required          | `pseudo_query` used to compute logits of the current `partial_block`. | `torch.float32`    | `(D,)`      |
| `numerator`     | Tensor    | Required          | Historical online softmax weighted sum output by `block_attn_res_prepare`. | `torch.float32`    | `(T, D)`    |
| `logit_max`     | Tensor    | Required          | Historical maximum logit output by `block_attn_res_prepare`. | `torch.float32`    | `(T,)`      |
| `exp_sum`       | Tensor    | Required          | Historical cumulative softmax denominator output by `block_attn_res_prepare`. | `torch.float32`    | `(T,)`      |
| `eps`           | `float`   | Optional          | Numerical stability term in the RMSNorm computation, with a default value of `1e-6`. | `float64`          | -           |

## Return Values

| Name | Type | Type | Description | Data Type | Dimension (Shape) |
| ------ | -------- | --------- | ---------------------------------------------------------------------- | ---------------- | ----------- |
| `h`    | Tensor   | Required      | Output of `block_attn_res_update`. Its data type and shape are the same as those of `delta`. It is a contiguous tensor. | `torch.bfloat16` | `(T, D)`    |

## Constraints

- This API supports use in inference scenarios.
- This API supports calls in single-operator mode and aclgraph mode.
- All tensor inputs and outputs must be contiguous.
- `T` must satisfy `T >= 0`, and `D` must satisfy `0 <= D <= 8192`.
- An empty tensor can be returned when `T == 0` or `D == 0`.
- The shapes of `partial_block`, `delta`, and `numerator` must be the same, all `(T, D)`.
- The shape of `pseudo_query` must be `(D,)`, and the shapes of `logit_max` and `exp_sum` must be `(T,)`.
- `eps` must be a finite positive number; if omitted, the default value `1e-6` is used. The Torch C++ bridge explicitly converts it to float32 before calling ACLNN.
- Tensor data type combinations must satisfy:

  | partial_block     | delta              | pseudo_query      | numerator         | logit_max         | exp_sum           | h                  |
  | ----------------- | ------------------ | ----------------- | ----------------- | ----------------- | ----------------- | ------------------ |
  | `torch.float32`   | `torch.bfloat16`   | `torch.float32`   | `torch.float32`   | `torch.float32`   | `torch.float32`   | `torch.bfloat16`   |

## Deterministic Computation

Defaults to a deterministic implementation.

## Example

- Call in single-operator mode:

  ```python
  import torch
  import torch_npu
  import cann_ops_transformer

  T = 32
  D = 7168

  # Create inputs.
  partial_block = torch.rand((T, D), dtype=torch.float32).npu()
  delta = torch.rand((T, D), dtype=torch.bfloat16).npu()
  pseudo_query = torch.rand((D,), dtype=torch.float32).npu()
  numerator = torch.rand((T, D), dtype=torch.float32).npu()
  logit_max = torch.rand((T,), dtype=torch.float32).npu()
  exp_sum = torch.rand((T,), dtype=torch.float32).npu()

  # Call the single-operator API.
  h = cann_ops_transformer.block_attn_res_update(
      partial_block,
      delta,
      pseudo_query,
      numerator,
      logit_max,
      exp_sum,
  )
  ```

- Call in aclgraph mode:

  ```python
  import torch
  import torch_npu
  import cann_ops_transformer


  # Define the network to be compiled.
  class OneOp(torch.nn.Module):
      def forward(
          self,
          partial_block,
          delta,
          pseudo_query,
          numerator,
          logit_max,
          exp_sum,
      ):
          return cann_ops_transformer.block_attn_res_update(
              partial_block,
              delta,
              pseudo_query,
              numerator,
              logit_max,
              exp_sum,
              eps=1e-6,
          )


  # Compile the network with the npugraph_ex backend.
  compiled_op = torch.compile(
      OneOp(),
      backend="npugraph_ex",
      fullgraph=True,
      dynamic=False,
      options={
          "static_kernel_compile": True,
      },
  )

  T = 8
  D = 7168

  # Create inputs.
  partial_block = torch.rand((T, D), dtype=torch.float32).npu()
  delta = torch.rand((T, D), dtype=torch.bfloat16).npu()
  pseudo_query = torch.rand((D,), dtype=torch.float32).npu()
  numerator = torch.rand((T, D), dtype=torch.float32).npu()
  logit_max = torch.rand((T,), dtype=torch.float32).npu()
  exp_sum = torch.rand((T,), dtype=torch.float32).npu()

  # Execute the compiled network.
  h = compiled_op(
      partial_block,
      delta,
      pseudo_query,
      numerator,
      logit_max,
      exp_sum,
  )
  ```
