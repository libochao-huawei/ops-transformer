# aclnnGroupedMatmulV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T11:46:23.376Z -->

**Notice: This API will be deprecated in later versions. Use the latest aclnnGroupedMatmulV5 API instead.**

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
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: Implements grouped matrix multiplication, where the dimension sizes of each group can differ. The basic function is matrix multiplication, as in $y_i[m_i,n_i]=x_i[m_i,k_i] \times weight_i[k_i,n_i], i=1...g$, where g is the number of groups and $m_i/k_i/n_i$ are the corresponding shapes.
    Compared with the [GroupedMatmul](aclnnGroupedMatmul_en.md) API, **this API adds**:

  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
    - Supports different grouping axes, indicated by groupType.
    - In the Non-quantized Scenario, supports Transpose of x and weight (Transpose refers to the scenario where the shape is [M, K] but the stride is [1, M] and the data layout is [K, M]).
    - In the Non-quantized Scenario, supports x and weight inputs both of the float32 data type.
    - In quantization and pseudo-quantization scenarios, weight transposition is supported, and weight as a single tensor is supported.
  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT products</term>:
    - Supports different grouping axes, indicated by groupType.
    - In the Non-quantized Scenario, supports x and weight inputs both of the float32 data type.
    - In pseudo-quantization scenarios, weight transposition is supported, and x, weight, and y as single tensors are all supported.
  <!-- end id8 -->

- Formula:
  - **Non-quantized scenario:**

  $$
   y_i=x_i\times weight_i + bias_i
  $$

  - **Quantization use case:**

  $$
   y_i=(x_i\times weight_i + bias_i) * scale_i + offset_i
  $$

  - **Dequantized scenario:**

  $$
   y_i=(x_i\times weight_i + bias_i) * scale_i
  $$

  - **Pseudo-Quantized Scenario:**

  $$
   y_i=x_i\times (weight_i + antiquant\_offset_i) * antiquant\_scale_i + bias_i
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnGroupedMatmulV2GetWorkspaceSize" API first to obtain input parameters and calculate the required workspace size based on the computation process, and then call the "aclnnGroupedMatmulV2" API to perform the computation.

```cpp
aclnnStatus aclnnGroupedMatmulV2GetWorkspaceSize(
  const aclTensorList   *x,
  const aclTensorList   *weight,
  const aclTensorList   *biasOptional,
  const aclTensorList   *scaleOptional,
  const aclTensorList   *offsetOptional,
  const aclTensorList   *antiquantScaleOptional,
  const aclTensorList   *antiquantOffsetOptional,
  const aclIntArray     *groupListOptional,
  int64_t                splitItem,
  int64_t                groupType,
  const aclTensorList   *y,
  uint64_t              *workspaceSize,
  aclOpExecutor         **executor)
```

```cpp
aclnnStatus aclnnGroupedMatmulV2(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream      stream)
```

## aclnnGroupedMatmulV2GetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1483px"><colgroup>
  <col style="width: 240px">
  <col style="width: 90px">
  <col style="width: 370px">
  <col style="width: 232px">
  <col style="width: 339px">
  <col style="width: 86px">
  <col style="width: 92px">
  <col style="width: 64px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x (aclTensorList *)</td>
      <td>Input</td>
      <td>The input x in the formula.</td>
      <td>The tensorList length supports [1, 128] or [1, 1024].</td>
      <td>FLOAT16, BFLOAT16, FLOAT32, INT8</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>weight (aclTensorList *)</td>
      <td>Input</td>
      <td>The weight in the formula.</td>
      <td>The tensorList length supports [1, 128] or [1, 1024].</td>
      <td>FLOAT16, BFLOAT16, FLOAT32, INT8</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>biasOptional (aclTensorList *)</td>
      <td>Input</td>
      <td>The bias in the formula.</td>
      <td>Same length as weight.</td>
      <td>INT32, BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleOptional (aclTensorList *)</td>
      <td>Input</td>
      <td>Represents the scale factor in quantization parameters.</td>
      <td>Generally, the length is the same as that of weight.</td>
      <td>UINT64</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>offsetOptional (aclTensorList *)</td>
      <td>Input</td>
      <td>Represents the offset in quantization parameters.</td>
      <td>The length is the same as that of weight.</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>antiquantScaleOptional (aclTensorList *)</td>
      <td>Input</td>
      <td>Represents the scale factor in pseudo-quantization parameters.</td>
      <td>The length is the same as that of weight.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>antiquantOffsetOptional (aclTensorList *)</td>
      <td>Input</td>
      <td>Represents the offset in pseudo-quantization parameters.</td>
      <td>The length is the same as that of weight.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupListOptional (aclIntArray *)</td>
      <td>Input</td>
      <td>Represents the distribution of matmul sizes along the group axis of inputs and outputs.</td>
      <td>      When the length of the TensorList in the output is 1, the last value in groupListOptional constrains the valid part of the output data, and the part not specified in groupListOptional will not participate in the update
      </td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>splitItem(int64_t)</td>
      <td>Input</td>
      <td>Integer parameter, indicating whether to split the output into tensors.</td>
      <td>      0/1 indicates multiple output tensors,
      2/3 indicates a single output tensor
      </td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupType (int64_t)</td>
      <td>Input</td>
      <td>Integer parameter that specifies the axis to be grouped.</td>
      <td>If the matrix multiplication is C[m,n]=A[m,k]xB[k,n], groupType can be -1 (no grouping), 0 (grouping along the m axis), or 2 (grouping along the k axis).</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y (aclTensorList *)</td>
      <td>Output</td>
      <td>Output y in the formula.</td>
      <td>The length of tensorList is [1, 128] or [1, 1024].</td>
      <td>FLOAT16, BFLOAT16, INT8, FLOAT32, INT32</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t)</td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor)</td>
      <td>Output</td>
      <td>Returns the op executor that contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  <!-- npu="A3,910b" id9 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
    - x supports FLOAT16, BFLOAT16, INT8, and FLOAT32.
    - weight supports FLOAT16, BFLOAT16, INT8, and FLOAT32.
    - biasOptional supports FLOAT16, FLOAT32, and INT32.
    - y supports FLOAT16, BFLOAT16, INT8, and FLOAT32.
    - The input parameters x and weight, and the output parameter y support up to 128 tensors.
  <!-- end id9 -->
  <!-- npu="950" id10 -->
  - <term>Ascend 950PR&950DT products</term>:
    - x supports FLOAT16, BFLOAT16, and FLOAT32.
    - weight supports FLOAT16, BFLOAT16, FLOAT32, and INT8.
    - biasOptional supports FLOAT16, BFLOAT16, and FLOAT32.
    - y supports FLOAT16, BFLOAT16, and FLOAT32.
    - scaleOptional and offsetOptional are not supported.
    - groupType supports m-axis grouping and no grouping. Only Non-quantized supports k-axis grouping.
    - The input parameters x and weight, and the output parameter y support up to 1024 tensors in the Non-quantized Scenario and up to 128 tensors in the Pseudo-quantized Scenario.

  <!-- end id10 -->

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1055px"><colgroup>
  <col style="width: 242px">
  <col style="width: 78px">
  <col style="width: 735px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Values</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>A required input, output, or attribute is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>The data type or data format of x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional, antiquantOffsetOptional, groupListOptional, or y is not within the supported range.</td>
    </tr>
    <tr>
      <td>The length of weight is not within the supported range.</td>
    </tr>
    <tr>
      <td>If bias is not empty, the length of bias is not equal to the length of weight.</td>
    </tr>
    <tr>
      <td>The dimension of groupListOptional is 1.</td>
    </tr>
    <tr>
      <td>In use cases where splitItem is 2 or 3, the length of out is not 1.</td>
    </tr>
    <tr>
      <td>In use cases where splitItem is 0 or 1, the length of out is not equal to the length of weight, or the length of groupListOptional is not equal to the length of weight.</td>
    </tr>
  </tbody>
  </table>

## aclnnGroupedMatmulV2

- **Parameters**
    <table style="undefined;table-layout: fixed; width: 834px"><colgroup>
    <col style="width: 118px">
    <col style="width: 87px">
    <col style="width: 629px">
    </colgroup>
    <thead>
      <tr>
        <th>Parameters</th>
        <th>Input/Output</th>
        <th>Description</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>workspace</td>
        <td>Input</td>
        <td>Workspace memory address allocated on the device side.</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Workspace size allocated on the device side, obtained by calling the first-phase API aclnnGroupedMatmulV2GetWorkspaceSize.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Executor for the op, containing the computation flow of the operator.</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream for executing tasks.</td>
      </tr>
    </tbody>
    </table>

- **Return Value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnGroupedMatmulV2 defaults to a deterministic implementation.

<!-- npu="A3,910b" id11 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
  - The supported input types for the non-quantized scenario are:
    - x is FLOAT16, weight is FLOAT16, biasOptional is FLOAT16, scaleOptional is empty, offsetOptional is empty, antiquantScaleOptional is empty, antiquantOffsetOptional is empty, y is FLOAT16;
    - x is BFLOAT16, weight is BFLOAT16, biasOptional is FLOAT32, scaleOptional is empty, offsetOptional is empty, antiquantScaleOptional is empty, antiquantOffsetOptional is empty, y is BFLOAT16;
    - x is FLOAT32, weight is FLOAT32, biasOptional is FLOAT32, scaleOptional is empty, offsetOptional is empty, antiquantScaleOptional is empty, antiquantOffsetOptional is empty, y is FLOAT32.
  - The input types supported in the quantized scenario are:

    - x is INT8, weight is INT8, biasOptional is INT32, scaleOptional is UINT64, offsetOptional is empty, antiquantScaleOptional is empty, antiquantOffsetOptional is empty, and y is INT8.
  - The input types supported in the pseudo-quantized scenario are:
    - x is FLOAT16, weight is INT8, biasOptional is FLOAT16, scaleOptional is empty, offsetOptional is empty, antiquantScaleOptional is FLOAT16, antiquantOffsetOptional is FLOAT16, and y is FLOAT16.
    - x is BFLOAT16, weight is INT8, biasOptional is FLOAT32, scaleOptional is empty, offsetOptional is empty, antiquantScaleOptional is BFLOAT16, antiquantOffsetOptional is BFLOAT16, and y is BFLOAT16.
    - If groupListOptional is passed in, it must be a non-negative increasing sequence, and its length cannot be 1.
  - Supported scenarios for different groupType values:
    - Quantized and pseudo-quantized scenarios only support groupType values of -1 and 0.
    - In supported scenarios, "Single" indicates a single tensor, "Multi" indicates multiple tensors, and the order represents x, weight, and y. For example, "Single Multi Single" indicates a scenario where x is a single tensor, weight consists of multiple tensors, and y is a single tensor.

      | groupType | Applicable scenario | scenario restriction |
      |:---------:|:-------:| :-------|
      | -1 | multiple-multiple-multiple |1) Only splitItem 0/1 is supported<br>2) Tensors in x must have the same number of dimensions, from 2 to 6; tensors in weight must be 2-dimensional; tensors in y must have the same number of dimensions as x<br>3) groupListOptional must be left empty |
      | 0 | single-single-single |1) Only splitItem 2/3 is supported<br>2) Tensors in weight must be 3-dimensional; tensors in x and y must be 2-dimensional<br>3) groupListOptional is mandatory, and its last value must be equal to the first dimension of tensors in x |
      | 0 | single-multiple-single |1) Only splitItem 2/3 is supported.<br>2) groupListOptional must be passed, and its last value must be equal to the first dimension of the tensor in x.<br>3) Tensors in x, weight, and y must be 2-dimensional.<br>4) The N axis of each tensor in weight must be equal. |
      | 0 | Single Multi Multi |1) Only splitItem 0/1 is supported<br>2) groupListOptional is mandatory, and the differences between its values must correspond one-to-one to the first dimensions of tensors in y<br>3) Tensors in x, weight, and y must be 2-dimensional |
      | 0 | multiple-multiple-single |1) Only splitItem 2/3 is supported.<br>2) Tensors in x, weight, and y must be 2-dimensional.<br>3) The N axis of each tensor in weight must be equal.<br>4) If groupListOptional is passed, the differences of groupListOptional must have a one-to-one correspondence with the first dimension of the tensors in x. |
      | 2 | single-single-single |1) Only splitItem 2/3 is supported.<br>2) Tensors in x and weight must be 2-dimensional. Tensors in y must be 3-dimensional.<br>3) groupListOptional must be passed, and its last value must be equal to the second dimension of the tensor in x.|

  - The size of the last dimension of each group of tensors in x and weight must be less than 65536. For $x_i$, the last dimension refers to the K axis when the attribute transpose_x is false, or the M axis when transpose_x is true. For $weight_i$, the last dimension refers to the N axis when the attribute transpose_weight is false, or the K axis when transpose_weight is true.
  - The size of each dimension of each group of tensors in x and weight, after 32-byte alignment, must be less than the maximum int32 value 2147483647.

<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950PR&950DT products</term>:

    <details>
    <summary>Non-quantized Scenario Constraints</summary>
      <a id="non-quantized-scenario-constraints"></a>

    - The data types supported in non-quantized scenarios are:
      - If groupListOptional is passed, it must be a non-negative monotonically non-decreasing sequence, and its length cannot be 1
      - The following input parameters are empty: scaleOptional, offsetOptional, antiQuantScaleOptional, antiQuantOffsetOptional
      - The supported data type combinations for non-empty parameters must satisfy the following table:

          |groupType| x       | weight  | biasOptional | y     |
          |:-------:|:-------:|:-------:| :------      |:------ |
          |-1/0/2   |BFLOAT16     |BFLOAT16     |BFLOAT16/FLOAT32/null    | BFLOAT16|
          |-1/0/2   |FLOAT16     |FLOAT16     |FLOAT16/FLOAT32/null    | FLOAT16|
          |-1/0/2   |FLOAT32     |FLOAT32     |FLOAT32/null    | FLOAT32|

    </details>

    <details>
    <summary>Pseudo-quantized Scenario Constraints</summary>
      <a id="pseudo-quantized-scenario-constraints"></a>

    - The data types supported in the pseudo-quantized scenario are:
      - The following input parameters are empty: scaleOptional and offsetOptional.
      - The supported data type combinations for non-empty parameters must satisfy the following table:

        |groupType| x       | weight  | antiquantScaleOptional | antiquantOffsetOptional | biasOptional | y     |
        |:-------:|:-------:|:-------:| :------  | :------ | :------     |:------ |
        |-1/0   |BFLOAT16     |INT8   |BFLOAT16  |BFLOAT16/null  |BFLOAT16/FLOAT32/null   | BFLOAT16|
        |-1/0   |FLOAT16     |INT8  |FLOAT16 |FLOAT16/null   |FLOAT16/null    | FLOAT16|

      - antiquantScaleOptional and non-null biasOptional and antiquantOffsetOptional must satisfy the following table (where g is the number of matmul groups, i.e., the number of groups):

        |groupType| usage scenario | Shape Constraint |
        |:---------:|:---------:| :------ |
        |-1|Weight multi-tensor|Each tensor is 1D with shape ($n_i$). It is not allowed that some tensors in a tensorList have shape ($n_i$) while others are empty. |
        |0 |Single weight tensor|Each tensor is 2D with shape (g, N).|

      - Only Single Single Single and Multi Multi Multi scenarios are supported.
    </details>

    <details>
    <summary>groupType Scenario Constraints</summary>
      <a id="grouptype-scenario-constraints"></a>

    - In supported scenarios, "Single" indicates a single tensor, "Multi" indicates multiple tensors, and the order represents x, weight, and y. For example, "Single Multi Single" indicates a scenario where x is a single tensor, weight consists of multiple tensors, and y is a single tensor.

      | groupType | Applicable scenario | scenario restriction |
      |:---------:|:-------:| :-------|
      | -1 | multiple-multiple-multiple |1) Only splitItem 0/1 is supported.<br>2) In the non-quantized scenario, tensors in x and out must be 2-dimensional, with shapes of ($m_i$, $k_i$) and ($m_i$, $n_i$) respectively. In the pseudo-quantized scenario, tensors in x must have consistent dimensions, supporting 2 to 6 dimensions, and tensors in y must have the same dimensions as x. Tensors in weight must be 2-dimensional, with a shape of ($n_i$, $k_i$) or ($k_i$, $n_i$). Tensors in bias must be 1-dimensional, with a shape of ($n_i$).<br>3) groupListOptional must be passed as empty.<br>4) Only ND input and ND output is supported.<br>5) Weight transpose is supported, but whether each tensor in the weight tensorList is transposed must keep consistent.<br>6) Transpose is not supported for x. |
      | 0 | single-single-single |1) Only splitItem 2/3 is supported<br>2) Tensors in weight must be 3-dimensional, with shape (g, N, K) or (g, K, N); tensors in x and y must be 2-dimensional, with shapes (M, K) and (M, N) respectively; tensors in bias must be 2-dimensional, with shape (g, N)<br>3) groupListOptional is mandatory, and its last value must not be greater than the first dimension of tensors in x<br>4) The first dimension of groupListOptional supports up to 1024, that is, a maximum of 1024 groups<br>5) Only ND-to-ND is supported<br>6) weight transpose is supported<br>7) x transpose is not supported|
      | 0 | single-multiple-single |1) Only splitItem 2/3 is supported.<br>2) groupListOptional must be passed, and its last value must equal the first dimension of the tensor in x, with a maximum length of 1024.<br>3) Tensors in x and y must be 2-dimensional, with shapes of (M, K) and (M, N) respectively. Tensors in weight must be 2-dimensional, with a shape of (N, K) or (K, N). Tensors in bias must be 1-dimensional, with a shape of (N).<br>4) The N axis of each tensor in weight must be equal.<br>5) Only ND input and ND output is supported.<br>6) Weight transpose is supported, but whether each tensor in the weight tensorList is transposed must keep consistent.<br>7) Transpose is not supported for x.<br>8) Only the non-quantized scenario is supported.|
      | 0 | Single Multi Multi |1) Only splitItem 0/1 is supported.<br>2) groupListOptional must be passed, and the differences of groupListOptional must have a one-to-one correspondence with the first dimension of the tensors in y, with a maximum length of 1024.<br>3) Tensors in x and y must be 2-dimensional, with shapes of (M, K) and (M, N) respectively. Tensors in weight must be 2-dimensional, with a shape of (N, K) or (K, N). Tensors in bias must be 1-dimensional, with a shape of (N).<br>4) Only ND input and ND output is supported.<br>5) Weight transpose is supported, but whether each tensor in the weight tensorList is transposed must keep consistent.<br>6) Transpose is not supported for x.<br>7) Only the non-quantized scenario is supported.|
      | 0 | multiple-multiple-single |1) Only splitItem 2 is supported.<br>2) Tensors in x and y must be 2-dimensional, with shapes of (M, K) and (M, N) respectively. Tensors in weight must be 2-dimensional, with a shape of (N, K) or (K, N). Tensors in bias must be 1-dimensional, with a shape of (N).<br>3) The N axis of each tensor in weight must be equal.<br>4) If groupListOptional is passed, the differences of groupListOptional must have a one-to-one correspondence with the first dimension of the tensors in x, with a maximum length of 1024.<br>5) Only ND input and ND output is supported.<br>6) Weight transpose is supported, but whether each tensor in the weight tensorList is transposed must keep consistent.<br>7) Transpose is not supported for x.<br>8) Only the non-quantized scenario is supported.|
      | 2 | single-single-single |1) Only splitItem 2/3 is supported.<br>2) Tensors in x and weight must be 2D with shapes (K, M) and (K, N) respectively. Tensors in y must be 3D with shape (g, M, N).<br>3) groupListOptional must be passed, and the last value must not exceed the first dimension of the tensor in x.<br>4) The first dimension of groupListOptional supports a maximum of 1024, that is, a maximum of 1024 groups.<br>5) x must be transposed and weight cannot be transposed.<br>6) Only ND in, ND out is supported.<br>7) Only non-quantized scenarios are supported.|

    </details>

<!-- end id12 -->

## Example

The sample code for calling the API is as follows, for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

  ```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul_v2.h"

#define CHECK_RET(cond, return_expr) \
    do {                               \
      if (!(cond)) {                   \
        return_expr;                   \
      }                                \
    } while (0)

#define LOG_PRINT(message, ...)     \
    do {                              \
      printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
    // Boilerplate for resource initialization.
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate Device-side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy Host-side data to Device-side memory.
    std::vector<T> hostData(GetShapeSize(shape), 0);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create the aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}


int CreateAclTensorList(const std::vector<std::vector<int64_t>>& shapes, void** deviceAddr,
                        aclDataType dataType, aclTensorList** tensor) {
    int size = shapes.size();
    std::vector<aclTensor*> tensors(size);
    for (int i = 0; i < size; i++) {
        int ret = CreateAclTensor<uint16_t>(shapes[i], deviceAddr + i, dataType, tensors.data() + i);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    *tensor = aclCreateTensorList(tensors.data(), size);
    return ACL_SUCCESS;
}


int main() {
    // 1. (Boilerplate) Initialize device/stream. For details, see the acl API manual.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as needed.
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs. Custom construction is required based on the API.
    std::vector<std::vector<int64_t>> xShape = {{1, 16}, {4, 32}};
    std::vector<std::vector<int64_t>> weightShape= {{16, 24}, {32, 16}};
    std::vector<std::vector<int64_t>> biasShape = {{24}, {16}};
    std::vector<std::vector<int64_t>> yShape = {{1, 24}, {4, 16}};
    void* xDeviceAddr[2];
    void* weightDeviceAddr[2];
    void* biasDeviceAddr[2];
    void* yDeviceAddr[2];
    aclTensorList* x = nullptr;
    aclTensorList* weight = nullptr;
    aclTensorList* bias = nullptr;
    aclIntArray* groupedList = nullptr;
    aclTensorList* scale = nullptr;
    aclTensorList* offset = nullptr;
    aclTensorList* antiquantScale = nullptr;
    aclTensorList* antiquantOffset = nullptr;
    aclTensorList* y = nullptr;
    int64_t splitItem = 0;
    int64_t groupType = -1;

    // Create the x aclTensorList.
    ret = CreateAclTensorList(xShape, xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create weight aclTensorList.
    ret = CreateAclTensorList(weightShape, weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create bias aclTensorList.
    ret = CreateAclTensorList(biasShape, biasDeviceAddr, aclDataType::ACL_FLOAT16, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create y aclTensorList.
    ret = CreateAclTensorList(yShape, yDeviceAddr, aclDataType::ACL_FLOAT16, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 3. Call the CANN operator library API.
    // Call the first-phase API of aclnnGroupedMatmulV2.
    ret = aclnnGroupedMatmulV2GetWorkspaceSize(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupedList, splitItem, groupType, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnGroupedMatmulV2.
    ret = aclnnGroupedMatmulV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulV2 failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output values and copy the result from the Device to the Host. Modify the code according to the specific API definition.
    for (int i = 0; i < 2; i++) {
        auto size = GetShapeSize(yShape[i]);
        std::vector<uint16_t> resultData(size, 0);
        ret = aclrtMemcpy(resultData.data(), size * sizeof(resultData[0]), yDeviceAddr[i],
                          size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t j = 0; j < size; j++) {
            LOG_PRINT("result[%ld] is: %d\n", j, resultData[j]);
        }
    }

    // 6. Release aclTensor and aclScalar. Modify the code according to the specific API definition.
    aclDestroyTensorList(x);
    aclDestroyTensorList(weight);
    aclDestroyTensorList(bias);
    aclDestroyTensorList(y);

    // 7. Release device resources. Modify it according to the specific API definition.
    for (int i = 0; i < 2; i++) {
        aclrtFree(xDeviceAddr[i]);
        aclrtFree(weightDeviceAddr[i]);
        aclrtFree(biasDeviceAddr[i]);
        aclrtFree(yDeviceAddr[i]);
    }
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
  ```
