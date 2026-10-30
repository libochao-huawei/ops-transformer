# all_to_all_quant_matmul

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T13:00:11.658Z -->

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

  `cann_ops_transformer.all_to_all_quant_matmul` performs the fused computation of All-to-All communication and MX quantization matrix multiplication. In multi-device use cases, rows and columns of the left matrix are redistributed across devices through URMA all-to-all, and MX FP8/FP4 quantization matrix multiplication is then performed with the right matrix to output BF16 or FP16 results.
- **Formula**:

  The detailed computation process is as follows:

  1. Each device holds a local shard of the left matrix `A_local[BS, H]`, which is redistributed through URMA all-to-all communication to obtain `A_permuted[BS/rankSize, H*rankSize]`.
  2. MX dequantization is performed on `A_permuted` with `x1_scale` and `x2_scale`, and the result is multiplied with the right matrix `X2`.
  3. `bias` (optional) is added to the matmul result to obtain the output `Y`.

  Here, rankSize is the number of NPU devices, that is, world_size.

## Function Prototype

```python
cann_ops_transformer.all_to_all_quant_matmul(
    x1,
    x2,
    group,
    *,
    bias=None,
    x1_scale=None,
    x2_scale=None,
    x1_quant_mode=None,
    x2_quant_mode=None,
    group_sizes=None,
    x1_dtype=None,
    x2_dtype=None,
    x1_scale_dtype=None,
    x2_scale_dtype=None,
    y_dtype=None,
    comm_mode=None,
    precision_mode=None,
) -> (Tensor, Tensor)
```

## Mandatory

| Name | Type | Required/Optional | Description | Data Type | Data Format | Dimension (Shape) |
| ------------------ | ------------ | --------- | ------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------- | -------------- | --------------------------------- |
| `x1`             | Tensor       | Required      | Left matrix, corresponding to the local A_local on each device. In MXFP4 use cases, data is packed and stored as uint8 (2 fp4 values occupy 1 byte), and the actual size of the K dimension passed in is `H/2`, which must be used together with `x1_dtype=296` (`float4_e2m1`). | `float8_e4m3fn`, `float8_e5m2`, `float4_e2m1` | ND             | `(BS, H)`                       |
| `x2`             | Tensor       | Required      | Right matrix, corresponding to X2 in the formula. It must be passed in as a transposed view (non-contiguous tensor), that is, construct `(N, H*rankSize)` first and then call `.t()`. In MXFP4 use cases, data is packed and stored as uint8, and the actual size of the K dimension passed in is `H*rankSize/2`, which must be used together with `x2_dtype=296` (`float4_e2m1`).                  | `float8_e4m3fn`, `float8_e5m2`, `float4_e2m1` | ND             | `(H*rankSize, N)` (transposed view)               |
| `group`          | ProcessGroup | Required      | torch.distributed communicator, used for All-to-All communication.                                                                                  | -                                              | -              | -                                 |
| `bias`           | Tensor       | Optional      | Bias, added after the matrix multiplication. Default value is `None`.                                                                                       | `float32`                              | ND             | `(N,)`                          |
| `x1_scale`       | Tensor       | Optional      | MX quantization scale of `x1`. Default value is `None`.                                                                                        | `float8_e8m0`                       | ND        | `(BS, ceil(H / 64), 2)`         |
| `x2_scale`       | Tensor       | Optional      | MX quantization scale of `x2`. Default value is `None`.                                                                                        | `float8_e8m0`                       | ND        | `(N, ceil(H*rankSize / 64), 2)` |
| `x1_quant_mode`  | int          | Optional      | Quantization mode of `x1`. Default value is 0; pass 6 for MX quantization.                                                                                         | int                                            | -              | -                                 |
| `x2_quant_mode`  | int          | Optional      | Quantization mode of `x2`. Default value is 0; pass 6 for MX quantization.                                                                                         | int                                            | -              | -                                 |
| `group_sizes`    | List[int]    | Optional      | MX quantization group size. Default value is `None`; pass `[1, 1, 32]` for MX use cases.                                                                    | int                                            | -              | -                                 |
| `x1_dtype`       | int          | Optional      | dtype of `x1`. Default value is -1 (use the tensor dtype); pass 296 (float4_e2m1) for MXFP4 use cases.                                                 | int                                            | -              | -                                 |
| `x2_dtype`       | int          | Optional      | dtype of `x2`. Default value is -1 (use the tensor dtype); pass 296 (float4_e2m1) for MXFP4 use cases.                                                 | int                                            | -              | -                                 |
| `x1_scale_dtype` | int          | Optional      | dtype of `x1_scale`. Default value is -1; pass 293 (float8_e8m0) for MX use cases.                                                           | int                                            | -              | -                                 |
| `x2_scale_dtype` | int          | Optional      | dtype of `x2_scale`. Default value is -1; pass 293 (float8_e8m0) for MX use cases.                                                           | int                                            | -              | -                                 |
| `y_dtype`        | int          | Optional      | Data type of output `y`. Default value is fp32; explicitly passing values other than 15 (BF16) or 5 (FP16) is rejected by operator validation.                                           | int                                            | -              | -                                 |
| `comm_mode`      | str          | Optional      | Communication mode. Default value is `"aiv_urma"`; currently only `"aiv_urma"` is supported.             | string                                         | -              | -                                 |
| `precision_mode` | int          | Optional      | Precision mode. Default value is 0. For 0/1/2, performance increases from low to high and precision decreases from high to low.                                                                         | int                                            | -              | -                                 |

## Return Values

| Name       | Type | Description                                                  | Data Type                       | Data Format   | Dimension (Shape)    |
| --------------- | -------- | ------------------------------------------------------------ | ------------------------------- | -------------- | -------------------- |
| `y`           | Tensor   | Matmul output.                                                 | Specified by `y_dtype` (BF16 or FP16) | ND             | `(BS/rankSize, N)` |

## Constraints

- Applicable use cases: This API supports both training and inference.
- Call mode: This API supports single-operator mode and must be used in a multi-device communicator.
- Only Ascend 950PR&950DT products are supported.
- `world_size` must be 2, 4, 8, or 16.
- `BS` must be divisible by `world_size`.
- `H` must be divisible by 64.
- `x1` and `x2` must be 2D tensors.
- `x1` and `x2` must both be FP8 or both FP4.
- Communication buffer size constraint: This operator uses the built-in HCCL communication buffer to complete All-to-All communication. The buffer size is controlled by the environment variable `HCCL_BUFFSIZE` (unit: MB; default value is 200 MB if not configured; twice the value, i.e., 400 MB, is allocated when it is passed to the framework layer). The commBuffer on each rank must hold the data written by all ranks, which requires: `HCCL_BUFFSIZE * 2` >= `BS * (H * sizeof(x1_dtype) + ceil(H / 64) * 2) + 2MB` (bytes), where `sizeof(x1_dtype)` is 1 byte in the FP8 use case and 0.5 byte in the FP4 use case, and `ceil(H / 64) * 2` is the communication volume of x1_scale (1-byte scale per 64 elements, 2 bytes for the last dimension). If the requirement is not met, the operator reports an error in the tiling phase during execution. In this case, run `export HCCL_BUFFSIZE=<suggested value>` to increase the buffer size and retry.
- The API returns the output tensor y and a placeholder None. For details, see the call description.

## Deterministic Computation

Defaults to a deterministic implementation.

## Calling Instructions

- Single-operator mode call example (MX FP8 use case, 2 devices), file name demo.py. The details are as follows:

  ```python
  import os
  import math
  from torch.multiprocessing import Process

  os.environ.setdefault("HCCL_BUFFSIZE", "2000")  # communication buffer, in MB

  import torch
  import torch_npu
  import torch.distributed as dist
  import cann_ops_transformer


  def run_all_to_all_quant_matmul(rank, world_size, bs, h, n):
      torch_npu.npu.set_device(rank % 2)

      dist.init_process_group(
          backend="hccl",
          rank=rank,
          world_size=world_size,
          init_method="tcp://127.0.0.1:29500",
      )
      group = dist.new_group(ranks=list(range(world_size)))

      # Construct inputs (each rank generates different data).
      torch.manual_seed(rank)
      x1 = torch.randn(bs, h).to(torch.float8_e4m3fn).npu()
      # x2 must be passed in as a transposed view (non-contiguous tensor): first construct (N, H*world_size),
      # then call .t() to obtain the transposed view of (H*world_size, N). The operator processes it per the transpose_x2 semantics.
      x2 = torch.randn(n, h * world_size).to(torch.float8_e4m3fn).npu()
      x1_scale = torch.randint(0, 256, (bs, math.ceil(h / 64), 2), dtype=torch.uint8).npu()
      x2_scale = torch.randint(0, 256, (n, math.ceil(h * world_size / 64), 2), dtype=torch.uint8).npu()

      # Call the operator (x2 must be passed in as the .t() transposed view stored in [N, H*world_size]),
      # The outputs are y and a reserved None.
      y, _ = cann_ops_transformer.all_to_all_quant_matmul(
          x1,
          x2.t(),
          group,
          x1_scale=x1_scale,
          x2_scale=x2_scale,
          y_dtype=15,           # BF16
          x1_quant_mode=6,      # MX
          x2_quant_mode=6,      # MX
          group_sizes=[1, 1, 32],
          x1_scale_dtype=293,   # fp8_e8m0 (stored as uint8, must override the dtype with enum)
          x2_scale_dtype=293,
          comm_mode="aiv_urma",
          precision_mode=0,
      )
      torch.npu.synchronize()

      print(f"[rank {rank}] y.shape={tuple(y.shape)}, y.dtype={y.dtype}")
      dist.barrier()
      dist.destroy_process_group()


  if __name__ == "__main__":
      world_size = 2
      bs, h, n = 128, 128, 64

      torch.multiprocessing.set_start_method("spawn", force=True)

      procs = []
      for rank in range(world_size):
          p = Process(target=run_all_to_all_quant_matmul, args=(rank, world_size, bs, h, n))
          p.start()
          procs.append(p)

      for p in procs:
          p.join()
  ```

  How to run:

  ```bash
  # Must be run in a multi-device NPU environment (2 devices).
  python3 demo.py
  ```
