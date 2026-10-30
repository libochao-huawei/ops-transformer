# grouped_matmul_activation_quant

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T12:25:06.908Z -->

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

  `cann_ops_transformer.grouped_matmul_activation_quant` integrates GroupedMatmul, an activation function, and quantization computation. It is currently used for the `gelu_tanh` activation with weight on the NZ path and for MXFP8 and MXFP4 input scenarios. The computation is completed by calling the underlying `aclnnGroupedMatmulActivationQuantWeightNz` API.

- **Formula**:

  The specific computation flow is as follows:
  1. Divide `x` into multiple groups along the M dimension according to `group_list`.
  2. Multiply each group with the corresponding `weight`, and perform MX dequantization with `x_scale` and `weight_scale`.
  3. Apply the `gelu_tanh` activation to the matrix multiplication result.
  4. Perform MX dynamic quantization on the activation result and output `y` and `y_scale`.

## Function Prototype

```python
cann_ops_transformer.grouped_matmul_activation_quant(
    x,
    group_list,
    weight,
    weight_scale,
    activation_type,
    *,
    bias=None,
    x_scale=None,
    group_list_type=0,
    tuning_config=None,
    quant_mode=None,
    y_dtype=None,
    round_mode="rint",
    scale_alg=0,
    dst_type_max=0.0,
    x_dtype=None,
    weight_dtype=None,
    weight_scale_dtype=None,
    x_scale_dtype=None,
) -> (Tensor, Tensor)
```

## Mandatory

### Common Parameters

| Name | Type | Type | Description | Data Type | Dimension (Shape) |
| --- | --- | --- | --- | --- | --- |
| `x` | Tensor | Required | Left matrix. In the MXFP4 quantization scenario, `torch.uint8` carries FLOAT4 data, and the corresponding type is specified through `x_dtype`. | `torch.float8_e4m3fn`, `torch.float8_e5m2`, `torch_npu.float4_e2m1fn_x2`, `torch_npu.float4_e1m2fn_x2` | 2 |
| `group_list` | Tensor | Required | Grouping information. | `torch.int64` | `(E,)` |
| `weight` | List[Tensor] | Required | Right-matrix TensorList. Only length 1 is supported. The caller must pass in `weight` in FRACTAL_NZ format. In the MXFP4 quantization scenario, `torch.uint8` carries FLOAT4 data, and the corresponding type is specified through `weight_dtype`. In the transposed scenario, run `npu_format_cast(..., 29)` on the source Weight first, then transpose the last two dimensions and pass in the view. | `torch.float8_e4m3fn`, `torch_npu.float4_e2m1fn_x2`, `torch_npu.float4_e1m2fn_x2` | 3 |
| `weight_scale` | List[Tensor] | Required | Quantization scale of `weight`. The tensorList only supports a length of 1. In the transposed scenario, the input is obtained by transposing the middle two dimensions of the source scale. | Parsed as `torch_npu.float8_e8m0fnu` through `weight_scale_dtype` | 4 |
| `activation_type` | str | Required | Activation function type. Currently only supports `"gelu_tanh"`. | string | - |
| `bias` | List[Tensor] | Optional | bias TensorList. The default value is `None`. In the current MX scenario, it must be empty. Supports `None`, an empty TensorList, or a single empty Tensor. | `torch.float32` | - |
| `x_scale` | Tensor | Optional | Quantization scale of `x`. Currently only supports MX scenarios. A valid Tensor must be passed in. | Parsed as `torch_npu.float8_e8m0fnu` through `x_scale_dtype` | 3 |
| `group_list_type` | int | Optional | Semantic type of `group_list`. The default value is 0. Supports 0 or 1. | int | - |
| `tuning_config` | List[int] | Optional | Reserved tuning parameter. The default value is `None`. | int | - |
| `quant_mode` | str | Optional | Quantization mode. The default value is `None`. Currently supports `"mx"` or `None`. | string | - |
| `y_dtype` | torch.dtype | Optional | Data type of the output `y`. The default value is `None`, in which case it is inferred to be the same data type as `x`. | `torch.float8_e4m3fn`, `torch.float8_e5m2`, `torch_npu.float4_e2m1fn_x2`, `torch_npu.float4_e1m2fn_x2` | - |
| `round_mode` | str | Optional | Rounding mode. The default value is `"rint"`. Currently only supports `"rint"`. | string | - |
| `scale_alg` | int | Optional | MX quantization scale algorithm. The default value is 0. Supports 0, 1, and 2: 0 indicates the OCP implementation; 1 indicates the cuBLAS implementation, which only supports FLOAT8 output; 2 indicates the FLOAT4 dynamic range implementation, which only supports FLOAT4_E2M1 output. | int | - |
| `dst_type_max` | float | Optional | Indicates the value of maxType, corresponding to Amax(DType) in the formula. The default value is 0.0. When `scale_alg=2`, 0.0 or `[6.0, 12.0]` is supported. 0.0 indicates using the maximum value 6.0 of FLOAT4_E2M1. When `scale_alg=0 or 1`, only 0.0 is supported. | float | - |
| `x_dtype` | torch.dtype/int | Optional | The default value is `None`. In the MXFP4 quantization scenario, the FLOAT4 type must be specified. In the MXFP8 quantization scenario, this parameter is not required. | `torch_npu.float4_e2m1fn_x2`, `torch_npu.float4_e1m2fn_x2` | - |
| `weight_dtype` | torch.dtype/int | Optional | The default value is `None`. In the MXFP4 quantization scenario, the FLOAT4 type must be specified. In the MXFP8 quantization scenario, this parameter is not required. | `torch_npu.float4_e2m1fn_x2`, `torch_npu.float4_e1m2fn_x2` | - |
| `weight_scale_dtype` | int | Optional | The default value is `None`. In the current MX scenario, `torch_npu.float8_e8m0fnu` must be passed in. | `torch_npu.float8_e8m0fnu` | - |
| `x_scale_dtype` | int | Optional | The default value is `None`. In the current MX scenario, `torch_npu.float8_e8m0fnu` must be passed in. | `torch_npu.float8_e8m0fnu` | - |

The following table lists the parameter types and shape constraints in different quantization scenarios. For descriptions of common parameters, see the preceding table.

MXFP8 quantization scenario: the MX quantization scenario in which input x and weight are both of type `torch.float8_e4m3fn` or `torch.float8_e5m2`;

MXFP4 quantization scenario: the MX quantization scenario in which input x and weight are both of type `torch_npu.float4_e2m1fn_x2` or `torch_npu.float4_e1m2fn_x2`

### MXFP8 Quantization Scenario

| Name | Data Type | Dimension (Shape) | Scenario Constraints |
| --- | --- | --- | --- |
| `x` | `torch.float8_e4m3fn`, `torch.float8_e5m2` | `(M, K)` | - |
| `weight` | `torch.float8_e4m3fn` | non-transposed: `(E, K, N)`<br>transposed: `(E, N, K)` | FRACTAL_NZ format must be passed in by the caller. |
| `weight_scale` | `torch_npu.float8_e8m0fnu` | non-transposed: `(E, ceil(K / 64), N, 2)`<br>transposed: `(E, N, ceil(K / 64), 2)` | The transposed attribute must be the same as that of `weight`. |
| `x_scale` | `torch_npu.float8_e8m0fnu` | `(M, ceil(K / 64), 2)` | Must be passed in. |
| `y` | `torch.float8_e4m3fn`, `torch.float8_e5m2` | `(M, N)` | The output type is specified by `y_dtype`. |
| `y_scale` | `torch_npu.float8_e8m0fnu` | `(M, ceil(N / 64), 2)` | - |
| `y_dtype` | `torch.float8_e4m3fn`, `torch.float8_e5m2` | - | Follows `x` when set to `None`. |
| `x_dtype` | `None` | - | Not required. |
| `weight_dtype` | `None` | - | Not required. |
| `scale_alg` | int | - | Supports 0 and 1. |
| `dst_type_max` | float | - | Only supports 0.0. |

### MXFP4 Quantization Scenario

| Name | Data Type | Dimension (Shape) | Scenario Constraints |
| --- | --- | --- | --- |
| `x` | `torch_npu.float4_e2m1fn_x2`, `torch_npu.float4_e1m2fn_x2` | `(M, K)` | Uses `torch.uint8` to carry FLOAT4 data. The type must be specified through `x_dtype`. |
| `weight` | `torch_npu.float4_e2m1fn_x2`, `torch_npu.float4_e1m2fn_x2` | Non-transposed: `(E, K, N)`<br>Transposed: `(E, N, K)` | Uses `torch.uint8` to carry FLOAT4 data. The type must be specified through `weight_dtype`, and the FRACTAL_NZ format must be passed in. |
| `weight_scale` | `torch_npu.float8_e8m0fnu` | Non-transposed: `(E, ceil(K / 64), N, 2)`<br>Transposed: `(E, N, ceil(K / 64), 2)` | The transposed attribute must be consistent with `weight`. |
| `x_scale` | `torch_npu.float8_e8m0fnu` | `(M, ceil(K / 64), 2)` | Must be passed in. |
| `y` | `torch.float8_e4m3fn`, `torch.float8_e5m2`, `torch_npu.float4_e2m1fn_x2`, `torch_npu.float4_e1m2fn_x2` | `(M, N)` | The FLOAT4 output uses `torch.uint8` to carry data. |
| `y_scale` | `torch_npu.float8_e8m0fnu` | `(M, ceil(N / 64), 2)` | - |
| `y_dtype` | `torch.float8_e4m3fn`, `torch.float8_e5m2`, `torch_npu.float4_e2m1fn_x2`, `torch_npu.float4_e1m2fn_x2` | - | Follows `x` when set to `None`. |
| `x_dtype` | `torch_npu.float4_e2m1fn_x2`, `torch_npu.float4_e1m2fn_x2` | - | Must be passed in. |
| `weight_dtype` | `torch_npu.float4_e2m1fn_x2`, `torch_npu.float4_e1m2fn_x2` | - | Must be passed in. |
| `scale_alg` | int | - | Supports 0, 1, and 2. When `y_dtype` is `torch.float8_e4m3fn` or `torch.float8_e5m2`, `scale_alg` supports 0 or 1. When `y_dtype` is `torch_npu.float4_e1m2fn_x2`, `scale_alg` only supports 0. When `y_dtype` is `torch_npu.float4_e2m1fn_x2`, `scale_alg` only supports 0 or 2. |
| `dst_type_max` | float | - | Supports 0.0 or `[6.0, 12.0]` when `scale_alg=2`. Only 0.0 is supported for other `scale_alg` values. |

## Returns

| Return Value | Return Value Type | Required/Optional | Description | Data Type | Dimension (Shape) |
| --- | --- | --- | --- | --- | --- |
| `y` | Tensor | Required | Quantized output after activation. In the MXFP4 scenario, `torch.uint8` carries FLOAT4 data. | Specified by `y_dtype`. | 2D |
| `y_scale` | Tensor | Required | Quantization scale corresponding to output `y`. | `torch_npu.float8_e8m0fnu` | `(M, ceil(N / 64), 2)` |

## Constraints

- Applicable use cases: This API supports training and inference.
- Calling method: This API supports calls in single-operator mode.
- `N` must be a multiple of 64, and the value range of `E` is `[1, 1024]`.
- `K` must be an even number and cannot be 2.
- `scale_alg=0` supports FLOAT4 and FLOAT8 output; `scale_alg=1` only supports FLOAT8 output; `scale_alg=2` only supports FLOAT4_E2M1 output, and `dst_type_max` can only be 0.0 or a value in `[6.0, 12.0]`.
- Empty tensors with M or N equal to 0 are supported; when M or N is 0, K can be 0.
- When `group_list_type=0`, the value indicates the cumulative end position of each group on the M axis, and the last value must not exceed the first dimension of the tensor in x; when `group_list_type=1`, the value indicates the M axis length of each group, and the sum of values must not exceed the first dimension of the tensor in x. When `group_list_type=0`, the last value must not exceed the first dimension of the tensor in x; when groupListType is 1, the sum of values must not exceed the first dimension of the tensor in x. The values in group_list constrain the valid part of the output data, and the parts not specified in group_list do not participate in the update.

## Deterministic Computation

Defaults to a deterministic implementation.

## Calling Example

- Call in MXFP8 single-operator mode:

  ```python
  import math
  import torch
  import torch_npu
  import cann_ops_transformer

  E = 1
  M = 64
  K = 128
  N = 128

  x = torch.randint(-8, 8, (M, K), dtype=torch.int8).to(torch.float8_e4m3fn).npu()
  weight = torch.randint(-8, 8, (E, K, N), dtype=torch.int8).to(torch.float8_e4m3fn).npu()
  weight = torch_npu.npu_format_cast(weight, 29, customize_dtype=torch.float8_e4m3fn)
  weight_scale = torch.randint(-8, 8, (E, math.ceil(K / 64), N, 2), dtype=torch.int8).npu()
  x_scale = torch.randint(-8, 8, (M, math.ceil(K / 64), 2), dtype=torch.int8).npu()
  group_list = torch.tensor([M], dtype=torch.int64).npu()

  y, y_scale = cann_ops_transformer.grouped_matmul_activation_quant(
      x,
      group_list,
      [weight],
      [weight_scale],
      "gelu_tanh",
      bias=None,
      x_scale=x_scale,
      quant_mode="mx",
      y_dtype=None,
      weight_scale_dtype=torch_npu.float8_e8m0fnu,
      x_scale_dtype=torch_npu.float8_e8m0fnu,
  )
  ```

- Call in single-operator mode with MXFP4 input packed as `uint8`, transposed weight NZ, and `scale_alg=2`:

  ```python
  import math
  import torch
  import torch_npu
  import cann_ops_transformer

  # K is even and not equal to 2; N is a multiple of 64.
  E = 3
  M = 52
  K = 82
  N = 128
  group_list = torch.tensor([1, 31, 20], dtype=torch.int64).npu()

  # MXFP4 uses torch.uint8 to carry FLOAT4 data.
  x = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8).npu()
  x_scale = torch.randint(120, 127, (M, math.ceil(K / 64), 2), dtype=torch.int8).npu()

  # Transpose weight NZ: first run npu_format_cast(..., 29) on the source Weight, then transpose the last two dimensions.
  weight_source = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8).npu()
  weight = torch_npu.npu_format_cast(weight_source, 29).transpose(-1, -2)
  weight_scale_source = torch.randint(
      120, 127, (E, N, math.ceil(K / 64), 2), dtype=torch.int8
  ).npu()
  weight_scale = weight_scale_source.transpose(-3, -2)

  y, y_scale = cann_ops_transformer.grouped_matmul_activation_quant(
      x,
      group_list,
      [weight],
      [weight_scale],
      "gelu_tanh",
      bias=None,
      x_scale=x_scale,
      group_list_type=1,
      quant_mode="mx",
      y_dtype=torch_npu.float4_e2m1fn_x2,
      round_mode="rint",
      scale_alg=2,
      dst_type_max=9.0,
      x_dtype=torch_npu.float4_e2m1fn_x2,
      weight_dtype=torch_npu.float4_e2m1fn_x2,
      weight_scale_dtype=torch_npu.float8_e8m0fnu,
      x_scale_dtype=torch_npu.float8_e8m0fnu,
  )
  torch.npu.synchronize()
  # y uses torch.uint8 to carry FLOAT4 data; the shape of y_scale is (M, ceil(N / 64), 2).
  print(y.shape, y.dtype, y_scale.shape, y_scale.dtype)
  ```
