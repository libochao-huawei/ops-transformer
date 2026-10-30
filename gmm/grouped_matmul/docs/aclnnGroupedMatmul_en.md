# aclnnGroupedMatmul

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T11:29:34.934Z -->

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
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: Implements grouped matrix multiplication, where the dimension sizes of each group can differ. The basic function is matrix multiplication, as in $y_i[m_i,n_i]=x_i[m_i,k_i] \times weight_i[k_i,n_i], i=1...g$, where g is the number of groups and $m_i/k_i/n_i$ are the corresponding shapes. The following four scenarios are supported based on the number of tensors for x, weight, and y:

  - x, weight, and y are all multi-tensor, meaning the tensor corresponding to each group's data is independent.
  - x is a single tensor, while weight and y are multiple tensors. In this case, use the optional parameter group_list to specify how x is grouped along the row dimension. For example, group_list[0]=10 indicates that the first 10 rows of x participate in the first group of matrix multiplication.
  - x and weight are multiple tensors, while y is a single tensor. In this case, the results of each group of matrix multiplication are stored contiguously in the same tensor.
  - x and y are single tensors, while weight is multiple tensors. This is a combination of the previous two cases.

    **NOTE** A single tensor means that all grouped tensors in a tensor list are merged into one along the M axis; otherwise, they are multiple tensors.
- Formula:

  - **Non-quantized scenario:**

    $$
      y_i=x_i\times weight_i + bias_i
    $$

  - **Quantized field:**

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

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnGroupedMatmulGetWorkspaceSize" API to obtain the input parameters and calculate the required workspace size based on the computation process, and then call the "aclnnGroupedMatmul" API to execute the computation.

```cpp
aclnnStatus aclnnGroupedMatmulGetWorkspaceSize(
    const aclTensorList  *x,
    const aclTensorList  *weight,
    const aclTensorList  *biasOptional,
    const aclTensorList  *scaleOptional,
    const aclTensorList  *offsetOptional,
    const aclTensorList  *antiquantScaleOptional,
    const aclTensorList  *antiquantOffsetOptional,
    const aclIntArray    *groupListOptional,
    int64_t               splitItem,
    const aclTensorList  *y,
    uint64_t             *workspaceSize,
    aclOpExecutor        **executor)
```

```cpp
aclnnStatus aclnnGroupedMatmul(
    void            *workspace,
    uint64_t         workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream      stream)
```

## aclnnGroupedMatmulGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed;width: 1540px"><colgroup>
    <col style="width: 240px">
    <col style="width: 120px">
    <col style="width: 300px">
    <col style="width: 330px">
    <col style="width: 212px">
    <col style="width: 100px">
    <col style="width: 190px">
    <col style="width: 118px">
    </colgroup>
    <thead>
      <tr>
        <th>Parameter</th>
        <th style="white-space: nowrap">Input/Output</th>
        <th>Description</th>
        <th>Instruction</th>
        <th>Data Type</th>
        <th><a href="../../../docs/en/context/data_format.md" target="_blank">Data Format</a></th>
        <th style="white-space: nowrap">Dimension (shape)</th>
        <th><a href="../../../docs/en/context/non_contiguous_tensor.md" target="_blank">Non-contiguous Tensor</a></th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>x (aclTensorList *)</td>
        <td>Input</td>
        <td>Input x in the formula.</td>
        <td>          <ul>
            <li>The maximum supported length is 128.</li>
          </ul>
        </td>
        <td>FLOAT16, BFLOAT16, FLOAT32, INT8</td>
        <td>ND</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>weight (aclTensorList *)</td>
        <td>Input</td>
        <td>Weight in the formula.</td>
        <td>          <ul>
            <li>The maximum supported length is 128.</li>
          </ul>
        </td>
        <td>FLOAT16, BFLOAT16, FLOAT32, INT8</td>
        <td>ND</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>biasOptional (aclTensorList *)</td>
        <td>Optional input</td>
        <td>The bias in the formula.</td>
        <td>          <ul>
            <li>The length is the same as weight.</li>
          </ul>
        </td>
        <td>FLOAT16, FLOAT32, INT32, BFLOAT16</td>
        <td>ND</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>scaleOptional (aclTensorList *)</td>
        <td>Optional input</td>
        <td>Represents the scaling factor in quantization parameters.</td>
        <td>          <ul>
            <li>The length is the same as that of weight.</li>
          </ul>
        </td>
        <td>UINT64</td>
        <td>ND</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>offsetOptional (aclTensorList *)</td>
        <td>Optional input</td>
        <td>Represents the offset in quantization parameters.</td>
        <td>          <ul>
            <li>Length is the same as weight.</li>
          </ul>
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>antiquantScaleOptional (aclTensorList *)</td>
        <td>Optional input</td>
        <td>Represents the scale factor in the pseudo-quantization parameters.</td>
        <td>          <ul>
            <li>Same length as weight.</li>
          </ul>
        </td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>antiquantOffsetOptional (aclTensorList *)</td>
        <td>Optional input</td>
        <td>Represents the offset in the pseudo-quantization parameters.</td>
        <td>          <ul>
            <li>Same length as weight.</li>
          </ul>
        </td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>groupListOptional (aclIntArray *)</td>
        <td>Optional input</td>
        <td>Represents the matmul index along the M dimension of inputs and outputs.</td>
        <td>          <ul>
            <li>The values must form a non-negative, monotonically non-decreasing sequence, representing the cumsum result (cumulative sum) of the grouping axis size.</li>
            <li>When the length of the TensorList in the output is 1, the last value in groupListOptional limits the valid part of the output data. The part not specified in groupListOptional does not participate in the update.</li>
          </ul>
        </td>
        <td>INT64</td>
        <td>ND</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>splitItem (int64_t)</td>
        <td>Input</td>
        <td>Integer parameter. Represents whether the output tensors are split.</td>
        <td>          <ul>
            <li>0/1 represents multiple output tensors; 2/3 represents a single output tensor.</li>
          </ul>
        </td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>y(aclTensorList *)</td>
        <td>Output</td>
        <td>Output y in the formula.</td>
        <td>          <ul>
            <li>A maximum of 128 tensors are supported.</li>
          </ul>
        </td>
        <td>FLOAT16, BFLOAT16, INT8, FLOAT32</td>
        <td>ND</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize(uint64_t)</td>
        <td>Output</td>
        <td>Returns the workspace size to be requested on the Device side.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor (aclOpExecutor)</td>
        <td>Output</td>
        <td>Returns the op executor, which contains the operator computation flow.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>

  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
    - x and weight support FLOAT16, BFLOAT16, and INT8.
    - y supports FLOAT16, BFLOAT16, INT8, and FLOAT32.
  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT products</term>:
    - x supports FLOAT16, BFLOAT16, and FLOAT32.
    - weight supports FLOAT16, BFLOAT16, FLOAT32, and INT8.
    - y supports FLOAT16, BFLOAT16, and FLOAT32.
    - scaleOptional and offsetOptional are not supported.

  <!-- end id8 -->

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 1030px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 650px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>A required input, output, or required attribute is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data type or data format of x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional, antiquantOffsetOptional, groupListOptional, splitItem, or y is not supported.</td>
    </tr>
    <tr>
      <td>The length of weight is out of the supported range.</td>
    </tr>
    <tr>
      <td>If bias is not null, the length of bias is not equal to the length of weight.</td>
    </tr>
    <tr>
      <td>If splitItem is 0 or 1, the length of y is not equal to the length of weight, or the length of groupListOptional is not equal to the length of weight.</td>
    </tr>
    <tr>
      <td>If splitItem is 2 or 3, the length of y is not 1.</td>
    </tr>
  </tbody></table>

## aclnnGroupedMatmul

- **Parameters:**
    <table>
    <thead>
      <tr><th>Parameter</th><th>Input/Output</th><th>Description</th></tr>
    </thead>
    <tbody>
      <tr><td>workspace</td><td>Input</td><td>Workspace memory address requested on the device.</td></tr>
      <tr><td>workspaceSize</td><td>Input</td><td>Workspace size requested on the device, obtained by the first-phase API aclnnGroupedMatmulGetWorkspaceSize.</td></tr>
      <tr><td>executor</td><td>Input</td><td>Op executor. Contains the operator computation flow.</td></tr>
      <tr><td>stream</td><td>Input</td><td>Specifies the Stream for task execution.</td></tr>
    </tbody>
    </table>

- **Return value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnGroupedMatmul defaults to a deterministic implementation.

<!-- npu="A3,910b" id9 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
  - The supported input types for the non-quantized scenario are:
    - x is FLOAT16, weight is FLOAT16, biasOptional is FLOAT16, scaleOptional is empty, offsetOptional is empty, antiquantScaleOptional is empty, antiquantOffsetOptional is empty, and y is FLOAT16.
    - x is BFLOAT16, weight is BFLOAT16, biasOptional is FLOAT32, scaleOptional is empty, offsetOptional is empty, antiquantScaleOptional is empty, antiquantOffsetOptional is empty, and y is BFLOAT16.
  - The input types supported in the quantized scenario are:

    - x is INT8, weight is INT8, biasOptional is INT32, scaleOptional is UINT64, offsetOptional is empty, antiquantScaleOptional is empty, antiquantOffsetOptional is empty, and y is INT8.
  - The input types supported in the pseudo-quantized scenario are:
    - x is FLOAT16, weight is INT8, biasOptional is FLOAT16, scaleOptional is empty, offsetOptional is empty, antiquantScaleOptional is FLOAT16, antiquantOffsetOptional is FLOAT16, and y is FLOAT16.
    - x is BFLOAT16, weight is INT8, biasOptional is FLOAT32, scaleOptional is empty, offsetOptional is empty, antiquantScaleOptional is BFLOAT16, antiquantOffsetOptional is BFLOAT16, and y is BFLOAT16.
  - If groupListOptional is passed, groupListOptional must be a non-negative, monotonically non-decreasing sequence, and the length of groupListOptional cannot be 1.
  - Currently supported scenarios:
      In the supported scenarios, "single" indicates a single tensor and "multiple" indicates multiple tensors. The order is x, weight, y. For example, "single-multiple-single" indicates that x is a single tensor, weight consists of multiple tensors, and y is a single tensor.

      | supported scenarios | scenario restriction |
      |:-------:| :-------|
      | multiple-multiple-multiple |1) only splitItem 0/1 is supported<br>2) tensors in x must have the same dimension, 2-6 dimensions are supported, tensors in weight must be 2-dimensional, and the dimension of tensors in y must be the same as that of x<br>3) if x contains a tensor with more than 2 dimensions, groupListOptional must be null<br>4) if tensors in x are 2-dimensional and groupListOptional is passed, the difference between elements of groupListOptional must correspond one-to-one to the first dimension of tensors in x |
      | single-multiple-single |1) Only splitItem 2/3 is supported.<br>2) groupListOptional must be passed, and its last value must be equal to the first dimension of the tensor in x.<br>3) Tensors in x, weight, and y must be 2-dimensional.<br>4) The N axis of each tensor in weight must be equal. |
      | Single Multi Multi |1) only splitItem 0/1 is supported<br>2) must pass groupListOptional, and the difference between elements of groupListOptional must correspond one-to-one to the first dimension of tensors in y<br>3) tensors in x, weight, and y must be 2-dimensional |
      | multiple-multiple-single |1) Only splitItem 2/3 is supported.<br>2) Tensors in x, weight, and y must be 2-dimensional.<br>3) The N axis of each tensor in weight must be equal.<br>4) If groupListOptional is passed, the differences of groupListOptional must have a one-to-one correspondence with the first dimension of the tensors in x. |

  - The size of the last dimension of each group of tensors in x and weight must be less than 65536. The last dimension of $x_i$ refers to the K dimension of $x_i$ when the attribute transpose_x is false, or the M dimension of $x_i$ when transpose_x is true. The last dimension of $weight_i$ refers to the N dimension of $weight_i$ when the attribute transpose_weight is false, or the K dimension of $weight_i$ when transpose_weight is true.
  - The size of each dimension of each group of tensors in x and weight, after 32-byte alignment, must be less than the maximum int32 value 2147483647.

<!-- end id9 -->
<!-- npu="950" id10 -->
- <term>Ascend 950PR&950DT products</term>:

    <details>
    <summary>Non-quantized Scenario Constraints</summary>
      <a id="non-quantized-scenario-constraints"></a>

    - The data types supported in non-quantized scenarios are:
      - If groupListOptional is passed, groupListOptional must be a non-negative, monotonically non-decreasing sequence, and the length of groupListOptional cannot be 1
      - The following input parameters are empty: scaleOptional, offsetOptional, antiQuantScaleOptional, antiQuantOffsetOptional
      - The supported data type combinations for non-empty parameters must satisfy the following table:

        | x       | weight  | biasOptional | y     |
        |:-------:|:-------:| :------      |:------ |
        |BFLOAT16     |BFLOAT16     |BFLOAT16/FLOAT32/null    | BFLOAT16|
        |FLOAT16     |FLOAT16     |FLOAT16/FLOAT32/null    | FLOAT16|
        |FLOAT32     |FLOAT32     |FLOAT32/null    | FLOAT32|

    </details>

    <details>
    <summary>Pseudo-quantized Scenario Constraints</summary>
      <a id="pseudo-quantized-scenario-constraints"></a>

    - The data types supported in the pseudo-quantized scenario are:
      - The following input parameters are empty: scaleOptional and offsetOptional.
      - The supported data type combinations for non-empty parameters must satisfy the following table:

        | x       | weight  | biasOptional | antiquantScaleOptional | antiquantOffsetOptional | y     |
        |:-------:|:-------:| :------      |:------ |:------ |:------ |
        |BFLOAT16    |INT8     |BFLOAT16/FLOAT32/null    | BFLOAT16 | BFLOAT16 | BFLOAT16 |
        |FLOAT16     |INT8     |FLOAT16/null             | FLOAT16  | FLOAT16  | FLOAT16  |

      - antiquantScaleOptional and non-empty biasOptional and antiquantOffsetOptional must satisfy the following table:

        | usage scenario | Shape Constraint |
        |:---------:| :------ |
        |Weight multi-tensor|Each tensor is 1D with shape ($n_i$). It is not allowed that some tensors in a tensorList have shape ($n_i$) while others are empty. |

    - Only supports the multi-multi-multi scenario.
    </details>

    <details>
    <summary>Supported Scenario Constraints</summary>
      <a id="supported-scenario-constraints"></a>

    - In supported scenarios, "Single" indicates a single tensor, "Multi" indicates multiple tensors, and the order represents x, weight, and y. For example, "Single Multi Single" indicates a scenario where x is a single tensor, weight consists of multiple tensors, and y is a single tensor.

      |supported scenarios |scenario restriction |
      |:-------:|:-------|
      | multiple-multiple-multiple |1) only splitItem 0/1 is supported<br>2) in the pseudo-quantized field, tensors in x must have the same dimension, 2-6 dimensions are supported, and the dimension of tensors in y must be the same as that of x; in the non-quantized field, tensors in x and y must be 2-dimensional, with shapes ($m_i$, $k_i$) and ($m_i$, $n_i$) respectively; tensors in weight must be 2-dimensional, with shape ($n_i$, $k_i$) or ($k_i$, $n_i$); tensors in bias must be 1-dimensional, with shape ($n_i$)<br>3) if x contains a tensor with more than 2 dimensions, groupListOptional must be null<br>4) if tensors in x are 2-dimensional and groupListOptional is passed, the difference between elements of groupListOptional must correspond one-to-one to the first dimension of tensors in x, and the length must not exceed 128<br>5) only ND input and ND output are supported<br>6) transpose of x is not supported, and transpose of weight is not supported |
      | single-multiple-single |1) Only supports splitItem 2/3.<br>2) groupListOptional must be passed, the last value must equal the first dimension of the tensor in x, and the maximum length is 128.<br>3) Tensors in x and y must be 2-dimensional with shapes (M, K) and (M, N) respectively. Tensors in weight must be 2-dimensional with shape (N, K) or (K, N). Tensors in bias must be 1-dimensional with shape (N).<br>4) The N axis of each tensor in weight must be the same.<br>5) Only supports ND in, ND out.<br>6) Only supports non-quantized scenarios.<br>7) Does not support x transposition. Supports weight transposition (if weight consists of multiple tensors, whether each tensor is transposed must be consistent).|
      | Single Multi Multi |1) Only supports splitItem 0/1.<br>2) groupListOptional must be passed, the last value must equal the first dimension of the tensor in x, and the maximum length is 128.<br>3) Tensors in x and y must be 2-dimensional with shapes (M, K) and (M, N) respectively. Tensors in weight must be 2-dimensional with shape (N, K) or (K, N). Tensors in bias must be 1-dimensional with shape (N).<br>4) Only supports ND in, ND out.<br>5) Only supports non-quantized scenarios.<br>6) Does not support x transposition. Supports weight transposition (if weight consists of multiple tensors, whether each tensor is transposed must be consistent).|
      | multiple-multiple-single |1) Only supports splitItem 2/3.<br>2) Tensors in x and y must be 2-dimensional with shapes (M, K) and (M, N) respectively. Tensors in weight must be 2-dimensional with shape (N, K) or (K, N). Tensors in bias must be 1-dimensional with shape (N).<br>3) The N axis of each tensor in weight must be the same.<br>4) If groupListOptional is passed, the differences of groupListOptional must have a one-to-one correspondence with the first dimension of tensors in x, and the maximum length is 128.<br>5) Only supports ND in, ND out.<br>6) Only supports non-quantized scenarios.<br>7) Does not support x transposition. Supports weight transposition (if weight consists of multiple tensors, whether each tensor is transposed must be consistent).|

    </details>

<!-- end id10 -->

## Example

The sample code is as follows, for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul.h"

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
    // Boilerplate for resource initialization
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
    // Call aclrtMalloc to allocate device-side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy host-side data to the device-side memory.
    std::vector<T> hostData(GetShapeSize(shape), 0);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of contiguous tensors.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor.
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
    // 1. (Boilerplate) Initialize device/stream. For details, see the acl API reference.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as needed.
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs. Customize the construction based on the API.
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

    // Create the x aclTensorList.
    ret = CreateAclTensorList(xShape, xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the weight aclTensorList.
    ret = CreateAclTensorList(weightShape, weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the bias aclTensorList.
    ret = CreateAclTensorList(biasShape, biasDeviceAddr, aclDataType::ACL_FLOAT16, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the y aclTensorList.
    ret = CreateAclTensorList(yShape, yDeviceAddr, aclDataType::ACL_FLOAT16, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 3. Call the CANN operator library API.
    // Call the first-phase API of aclnnGroupedMatmul.
    ret = aclnnGroupedMatmulGetWorkspaceSize(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupedList, splitItem, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnGroupedMatmul.
    ret = aclnnGroupedMatmul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmul failed. ERROR: %d\n", ret); return ret);

    // 4. (boilerplate) Synchronize and wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value, copy the result from the Device side to the Host side, and modify it based on the specific API definition.
    for (int i = 0; i < 2; i++) {
        auto size = GetShapeSize(yShape[i]);
        std::vector<uint16_t> resultData(size, 0);
        ret = aclrtMemcpy(resultData.data(), size * sizeof(resultData[0]), yDeviceAddr[i],
                          size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t j = 0; j < size; j++) {
            LOG_PRINT("result[%ld] is: %hu\n", j, resultData[j]);
        }
    }

    // 6. Release aclTensor and aclScalar, and modify them based on the specific API definition.
    aclDestroyTensorList(x);
    aclDestroyTensorList(weight);
    aclDestroyTensorList(bias);
    aclDestroyTensorList(y);

    // 7. Release device resources, and modify them based on the specific API definition.
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
