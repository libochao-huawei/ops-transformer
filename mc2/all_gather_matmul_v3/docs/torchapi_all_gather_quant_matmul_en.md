# all_gather_quant_matmul

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T12:52:14.727Z -->

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

  `cann_ops_transformer.all_gather_quant_matmul` performs the fused computation of AllGather communication and MX quantized matrix multiplication. In a multi-device use case, each device first collects the left matrix and its MX scale factors from all devices through URMA all-gather, then performs MX FP8/FP4 quantized matrix multiplication with the local right matrix, and outputs BF16 or FP16 results.
- **Formula**:

  This operator currently only supports the MX quantization scenario: `x1` is of shape `(M, K)`, `x2` is of shape `(N, K)`, `x1_scale` is of shape `(M, ceilDiv(K, 64), 2)`, and `x2_scale` is of shape `(N, ceilDiv(K, 64), 2)`. After AllGather is performed on inputs `x1` and `x1_scale`, MatMul is performed on `x1` and `x2`, and then dequant is performed by grouping along the K axis (group size = 32):

  $$
  output=\sum_{0}^{\left \lfloor \frac{K}{blockSize=32} \right \rfloor}(AllGather(x1)@x2*(AllGather(x1\_scale)*x2\_scale)) + bias
  $$

  `bias` is an optional accumulation term, and the output shape is `(M * rankSize, N)`, where rankSize is the number of devices participating in the allgather matmul computation.

## Function Prototype

```python
cann_ops_transformer.all_gather_quant_matmul(
    x1,
    x2,
    group,
    *,
    bias=None,
    x1_scale=None,
    x2_scale=None,
    group_sizes=None,
    x1_dtype=None,
    x2_dtype=None,
    x1_scale_dtype=None,
    x2_scale_dtype=None,
    y_dtype=None,
    comm_mode=None,
) -> (Tensor, Tensor, Tensor)
```

## Mandatory

| Name | Type | Required/Optional | Description | Data Type | Data Format | Dimension (Shape) |
| ------------------ | ------------ | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | -------------- | ------------------------ |
| `x1` | Tensor | Required | Left matrix, corresponding to the local `x1[M, K]` on each device. In the MXFP4 scenario, data is packed and stored as uint8 (2 fp4 values occupy 1 byte), and the actual K dimension size passed in is `K/2`. It must be used together with `x1_dtype=296` (`float4_e2m1`). | `float8_e4m3fn`, `float8_e5m2`, `float4_e2m1` | ND | `(M, K)` |
| `x2` | Tensor | Required | Right matrix, corresponding to X2 in the formula. It must be passed in as a transposed view (non-contiguous tensor), that is, construct `(N, K)` first and then call `.t()`. In the MXFP4 scenario, data is packed and stored as uint8, and the actual K dimension size passed in is `K/2`. It must be used together with `x2_dtype=296` (`float4_e2m1`), and `x1`/`x2` must both be FP4. | `float8_e4m3fn`, `float8_e5m2`, `float4_e2m1` | ND | `(N, K)` |
| `group` | ProcessGroup | Required | Distributed communicator, used for AllGather communication. | - | - | - |
| `bias` | Tensor | Optional | Bias term, added after matrix multiplication; default value is `None`. If `None` is passed in, no bias is added. | `float32` | ND | `(N,)` |
| `x1_scale` | Tensor | Optional | MX quantization scale of `x1`; default value is `None`, and it is required in the MX quantization scenario. | `float8_e8m0` | ND | `(M, ceil(K / 64), 2)` |
| `x2_scale` | Tensor | Optional | MX quantization scale of `x2`; default value is `None`, and it is required in the MX quantization scenario. | `float8_e8m0` | ND | `(N, ceil(K / 64), 2)` |
| `group_sizes` | List[int] | Optional | MX quantization group size. Only `[1, 1, 32]` is supported in the MX scenario; default value is `None`. If `None` is passed in, it is automatically derived internally. For the derivation rules, see Constraints. | int | - | - |
| `x1_dtype` | int | Optional | dtype of `x1`; default value is `None`. If `None` is passed in, the dtype of input `x1` is used. In the MXFP8 scenario, pass in `None` (torch natively supports float8_e4m3fn/float8_e5m2, so passing is not required); in the MXFP4 scenario, pass in `296` (float4_e2m1). | int | - | - |
| `x2_dtype` | int | Optional | dtype of `x2`. If `None` is passed in, the dtype of input `x2` is used. In the FP8 scenario, pass in `None` (torch natively supports float8_e4m3fn/float8_e5m2, so passing is not required); in the MXFP4 scenario, pass in `296` (float4_e2m1). | int | - | - |
| `x1_scale_dtype` | int | Optional | dtype of `x1_scale`. If `None` is passed in, the dtype of `x1_scale` is used. If torch has no native float8_e8m0 type, pass in `293` (float8_e8m0). | int | - | - |
| `x2_scale_dtype` | int | Optional | dtype of `x2_scale`. If `None` is passed in, the dtype of `x2_scale` is used. If torch has no native float8_e8m0 type, pass in `293` (float8_e8m0). | int | - | - |
| `y_dtype` | int | Optional | Data type of output `y`. Only `15` (BF16) or `5` (FP16) can be passed in; if `None` is passed in, the default value is BF16. | int | - | - |
| `comm_mode` | str | Optional | Communication mode; default value is `"aiv_urma"`, and currently only `"aiv_urma"` is supported. | string | - | - |

## Returns

| Name      | Type     | Description                                                                             | Data Type                       | Data Format    | Dimension (Shape)     |
| -------------- | -------- | -------------------------------------------------------------------------------------- | ------------------------------- | -------------- | --------------------- |
| `y`          | Tensor   | Output of the fused AllGather and matmul computation.                                  | Specified by `y_dtype` (BF16 or FP16) | ND             | `(M * rankSize, N)` |
| `gather_out` | Tensor   | Intermediate output of AllGather communication; reserved parameter, not implemented in the current version (returns an empty placeholder tensor with shape `(0,)`). | Same as `x1`                    | ND             | `(0,)`              |
| `amax_out`   | Tensor   | amax output; reserved parameter, not implemented in the current version (returns an empty placeholder tensor with shape `(0,)`).                               | `float32`               | ND             | `(0,)`              |

## Constraints

- Applicable use cases: This API is applicable to training and inference.
- Call method: This API supports the single-operator mode and must be used in a multi-device communicator.
- Only Ascend 950PR&950DT products are supported.
- The data format of input and output tensors supports only ND.
- Input `x1` is 2-dimensional with shape `(M, K)`; input `x2` is 2-dimensional and you must pass in a `.t()` transpose view of data stored as `[N, K]`. The K axis of `x1` equals that of `x2`, and the value range of `K` is [256, 65535). In FP4 scenarios, `K` must be an even number.
- `M` (M-axis length on a single device) and `N` are in the range [1, 2147483647] (INT32_MAX).
- `bias` is 1-dimensional with shape `(N,)` and dtype `float32`.
- `x1_scale` and `x2_scale` are 3-dimensional with shapes `(M, ceil(K/64), 2)` and `(N, ceil(K/64), 2)`, respectively, and the dtype is `float8_e8m0` (if not natively supported by torch, use uint8 storage with scale_dtype=293).
- This operator currently only supports single-operator calls; graph mode is not supported.
- Output `y` is 2-dimensional, with shape `(M * rank_size, N)`, and only BF16/FP16 dtypes are supported.
- `x1` and `x2` must both be FP8 (float8_e4m3fn and float8_e5m2 can be mixed) or both FP4.
- `rank_size` must be one of 2, 4, 8, and 16.
- `comm_mode` currently only supports `"aiv_urma"`.
- `group_sizes` supports only `[1, 1, 32]` in MX use cases (both explicitly passed in values and automatically derived results must meet this requirement). `group_sizes` can be expressed as `[groupSizeM, groupSizeN, groupSizeK]`. If one or more values in the list are 0, groupSizeM, groupSizeN, and groupSizeK are reset based on the input shapes of x1/x2/x1Scale/x2Scale for computation. Setting principle: `x1_scale`/`x2_scale` are 3-dimensional, with shape `(dim0, ceil(K/64), 2)`, where dimension 0 (dim0) corresponds to the M (or N) axis, dimension 1 (dim1, i.e., `ceil(K/64)`) is the number of groups along the K axis, and dimension 2 (dim2, i.e., `2`) is the number of scales in each group. If groupSizeM=0, the m-directional quantization group value is determined by interface derivation, and the derivation formula is `groupSizeM = x1.M / x1_scale.dim0` (must ensure that x1.M is divisible by x1_scale.dim0), where x1.M is consistent with M in the x1 shape, and x1_scale.dim0 is dimension 0 of the x1Scale shape (whose value is M); if groupSizeK=0, the k-directional quantization group value is determined by interface derivation, and the derivation formula is `groupSizeK = x1.K / (x1_scale.dim1 × x1_scale.dim2)` (must ensure that x1.K is divisible by `x1_scale.dim1 × x1_scale.dim2`), where x1.K is consistent with K in the x1 shape, and `x1_scale.dim1 × x1_scale.dim2` is the product of dimensions 1 and 2 of the x1Scale shape (i.e., `ceil(K/64) × 2`, indicating the total number of scale elements along the k direction); if groupSizeN=0, the n-directional quantization group value is determined by interface derivation, and the derivation formula is `groupSizeN = x2.N / x2_scale.dim0` (must ensure that x2.N is divisible by x2_scale.dim0), where x2.N is consistent with N in the x2 shape, and x2_scale.dim0 is dimension 0 of the x2Scale shape (whose value is N).
- The name length of communicator `group` must be in the range `[1, 128)`.
- Communication buffer size constraint: This operator uses the built-in HCCL communication buffer to perform AllGather communication. The buffer size is controlled by the environment variable `HCCL_BUFFSIZE` (unit: MB; the default value is 200 MB if not configured; when the value is passed in to the framework layer, twice the value is allocated, that is, 400 MB). The requirement is `total communication data volume + 2MB <= HCCL_BUFFSIZE * 2`, where the total communication data volume is `rank_size * M * (K*sizeof(x1_dtype) + ceil(K / 64) * 2) + 2MB` bytes, where `sizeof(x1_dtype)` is 1 byte in FP8 use cases and 0.5 bytes in FP4 use cases. Set the required `HCCL_BUFFSIZE` environment variable using `export HCCL_BUFFSIZE=<value>`.

## Deterministic Computation

Defaults to a deterministic implementation.

## Call Instructions

- Call example in single-operator mode (MX FP8 scenario, 4 devices). The file name is demo.py. The details are as follows:

  ```python
  import os
  import math
  from torch.multiprocessing import Process

  os.environ.setdefault("HCCL_BUFFSIZE", "2000")  # Communication buffer, in MB.

  import torch
  import torch_npu
  import torch.distributed as dist
  import cann_ops_transformer


  def run_all_gather_quant_matmul(rank, world_size, m, k, n):
      torch_npu.npu.set_device(rank % 4)

      dist.init_process_group(
          backend="hccl",
          rank=rank,
          world_size=world_size,
          init_method="tcp://127.0.0.1:29500",
      )
      group = dist.new_group(ranks=list(range(world_size)))

      # Construct inputs (each rank generates different data).
      torch.manual_seed(rank)
      x1 = torch.randn(m, k).to(torch.float8_e4m3fn).npu()
      x2 = torch.randn(n, k).to(torch.float8_e4m3fn).npu()
      x1_scale = torch.randint(0, 256, (m, math.ceil(k / 64), 2), dtype=torch.uint8).npu()
      x2_scale = torch.randint(0, 256, (n, math.ceil(k / 64), 2), dtype=torch.uint8).npu()

      # Call the operator (x2 must be passed in as the .t() transpose view stored in [N, K]).
      y, gather_out, amax_out = cann_ops_transformer.all_gather_quant_matmul(
          x1,
          x2.t(),
          group,
          x1_scale=x1_scale,
          x2_scale=x2_scale,
          group_sizes=[1, 1, 32],
          x1_scale_dtype=293,   # fp8_e8m0 (stored as uint8, need to override dtype with enum).
          x2_scale_dtype=293,
          y_dtype=15,           # BF16
          comm_mode="aiv_urma",
      )
      torch.npu.synchronize()

      print(f"[rank {rank}] y.shape={tuple(y.shape)}, y.dtype={y.dtype}")
      dist.barrier()
      dist.destroy_process_group()


  if __name__ == "__main__":
      world_size = 4
      m, k, n = 2048, 4096, 2560

      torch.multiprocessing.set_start_method("spawn", force=True)

      procs = []
      for rank in range(world_size):
          p = Process(target=run_all_gather_quant_matmul, args=(rank, world_size, m, k, n))
          p.start()
          procs.append(p)

      for p in procs:
          p.join()
  ```

  How to run:

  ```bash
  # Must be executed in a multi-device NPU environment (4 devices).
  python3 demo.py
  ```
