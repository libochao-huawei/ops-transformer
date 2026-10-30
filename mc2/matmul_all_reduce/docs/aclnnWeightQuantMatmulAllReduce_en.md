# aclnnWeightQuantMatmulAllReduce

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T13:36:42.535Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
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

**NOTE**: When using this API, ensure that the driver firmware package and the CANN package are both of the matching 8.0.RC2 version or a matching later version. Otherwise, errors such as BUS ERROR will occur.

## Function

- **Description**: Performs pseudo-quantization computation on the input x2, and then completes Matmul and AllReduce computation. Supports pertensor, perchannel, and pergroup quantization modes.

- **Formula**:

  $$
  output = AllReduce(x1 @ ((x2 + antiquantOffset) * antiquantScale) + bias + x3)
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call `aclnnWeightQuantMatmulAllReduceGetWorkspaceSize` to obtain the workspace size required for the computation and the executor that contains the computation process of the operator. Then call `aclnnWeightQuantMatmulAllReduce` to perform the computation.

```cpp
aclnnStatus aclnnWeightQuantMatmulAllReduceGetWorkspaceSize(
    const aclTensor  *x1,
    const aclTensor  *x2,
    const aclTensor  *bias,
    const aclTensor  *antiquantScale,
    const aclTensor  *antiquantOffset,
    const aclTensor  *x3,
    const char       *group,
    const char       *reduceOp,
    int64_t          commTurn,
    int64_t          streamMode,
    int64_t          antiquantGroupSize,
    const aclTensor *output,
    uint64_t        *workspaceSize,
    aclOpExecutor **executor)
```

```cpp
aclnnStatus aclnnWeightQuantMatmulAllReduce(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnWeightQuantMatmulAllReduceGetWorkspaceSize

- **Parameters**
    <table style="undefined;table-layout: fixed; width: 1567px"><colgroup>
      <col style="width: 170px">
      <col style="width: 120px">
      <col style="width: 300px">
      <col style="width: 330px">
      <col style="width: 212px">
      <col style="width: 100px">
      <col style="width: 190px">
      <col style="width: 145px">
      </colgroup>
      <thead>
        <tr>
          <th>Parameter</th>
          <th>Input/Output</th>
          <th>Description</th>
          <th>Instruction</th>
          <th>data type</th>
          <th>data format</th>
          <th>Dimension (shape)</th>
          <th>non-contiguous tensor</th>
        </tr></thead>
      <tbody>
        <tr>
          <td>x1</td>
          <td>Input</td>
          <td>Left matrix of the MatMul computation, that is, x1 in the formula.</td>
          <td><ul><li>The current version supports only 2D or 3D inputs.</li><li>Supports the non-transpose scenario.</li></ul></td>
          <td>BFLOAT16, FLOAT16</td>
          <td>See <a href="#constraints">Constraints</a>.</td>
          <td>2-3</td>
          <td>×</td>
        </tr>
        <tr>
          <td>x2</td>
          <td>Input</td>
          <td>Right matrix of the MatMul computation, that is, x2 in the formula.</td>
          <td><ul><li>The current version supports only two-dimensional inputs.</li><li>Supports transpose and non-transpose scenarios.</li><li>For ND format, non-contiguous tensors with the last two axes transposed are supported. Other non-contiguous tensors are not supported.</li></ul></td>
          <td>See <a href="#constraints">Constraints</a>.</td>
          <td>ND, FRACTAL_NZ</td>
          <td>2</td>
          <td>×</td>
        </tr>
        <tr>
          <td>bias</td>
          <td>Input</td>
          <td>Corresponds to the bias offset in the formula, that is, the bias in the formula.</td>
          <td><ul><li>Supports passing null pointers. In non-empty scenarios, the current version supports only one-dimensional inputs.</li></ul></td>
          <td>See <a href="#constraints">Constraints</a>.</td>
          <td>ND</td>
          <td>1</td>
          <td>√</td>
        </tr>
        <tr>
          <td>antiquantScale</td>
          <td>Input</td>
          <td>Refers to antiquantScale in the formula.</td>
          <td><ul><li>pertensor scenario: shape (1).</li><li>perchannel scenario: shape (n)/(1,n), where n is the size of the last dimension of x2.</li><li>pergroup scenario: shape (ceil(k,antiquantGroupSize),n).</li></ul></td>
          <td>BFLOAT16, FLOAT16</td>
          <td>ND</td>
          <td>1-2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>antiquantOffset</td>
          <td>Input</td>
          <td>Offset parameter for the pseudo-quantization computation on x2, corresponding to antiquantOffset in the formula.</td>
          <td><ul><li>A null pointer is supported. If the pointer is non-null, the shape must be the same as that of antiquantScale.</li><li>This parameter is not supported when the data format of x2 is FLOAT8_E4M3FN or HIFLOAT8. Set it to a null pointer.</li></ul></td>
          <td>BFLOAT16, FLOAT16</td>
          <td>ND</td>
          <td>1-2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>x3</td>
          <td>Input</td>
          <td>The add computation after MatMul, that is, x3 in the formula.</td>
          <td><ul><li>Null pointers can be passed in. If it is not null, the shape is the same as the shape after the mm computation.</li></ul></td>
          <td>See <a href="#constraints">Constraints</a>.</td>
          <td>ND</td>
          <td>2-3</td>
          <td>√</td>
        </tr>
        <tr>
          <td>group</td>
          <td>Input</td>
          <td>Communicator name.</td>
          <td><ul><li>Obtained through the API provided by Hccl: "extern HcclResult HcclGetCommName(HcclComm comm, char* commName);", where commName is the group.</li></ul></td>
          <td>String</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>reduceOp</td>
          <td>Input</td>
          <td>Reduce operation type.</td>
          <td><ul><li>Current version supports only the input "sum".</li></ul></td>
          <td>String</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>commTurn</td>
          <td>Input</td>
          <td>Number of data splits for communication, that is, total data volume/communication volume per call.</td>
          <td><ul><li>Current version supports only input 0.</li></ul></td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>streamMode</td>
          <td>Input</td>
          <td>Enumeration of stream modes.</td>
          <td><ul><li>Only enum value 1 is supported in the current version.</li></ul></td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>antiquantGroupSize</td>
          <td>Input</td>
          <td>groupSize input for the dequantization computation on x2 in pergroup pseudo-quantization mode.</td>
          <td><ul><li>This parameter is required in pergroup quantization scenarios. The value ranges [32,min(k-1,INT_MAX)] and must be a multiple of 32; the value range of k is the same as that of the mm API, that is, [1,65535].</li><li>Only 0 is supported in non-pergroup quantization scenarios.</li></ul></td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>Output</td>
          <td>Output</td>
          <td>Result of the MatMul computation and AllReduce communication, that is, output in the formula.</td>
          <td><ul><li>The dimension of output is the same as that of x1.</li></ul></td>
          <td>-</td>
          <td>ND</td>
          <td>2-3</td>
          <td>√</td>
        </tr>
        <tr>
          <td>workspaceSize</td>
          <td>Output</td>
          <td>Returns the workspace size to be requested on the Device.</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>executor</td>
          <td>Output</td>
          <td>Returns the op executor, which contains the operator computation flow.</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
      </tbody>
    </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 282px">
  <col style="width: 120px">
  <col style="width: 747px">
  </colgroup>
  <thead>
  <tr>
      <th>Return value</th>
      <th>Error code</th>
      <th>Description</th>
  </tr></thead>
  <tbody>
  <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>x1, x2, antiquantScale, or output is a null pointer.</td>
  </tr>
  <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type of x1, x2, bias, antiquantScale, antiquantOffset, x3, or output does not meet the requirement.</td>
  </tr>
  <tr>
      <td>reduceOp, streamMode, or antiquantGroupSize is out of the valid range.</td>
  </tr>
  <tr>
      <td>The shape of x1, x2, bias, antiquantScale, antiquantOffset, x3, output, or antiquantGroupSize does not meet the constraint requirements.</td>
  </tr>
  </tbody>
  </table>

## aclnnWeightQuantMatmulAllReduce

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 168px">
    <col style="width: 128px">
    <col style="width: 854px">
    </colgroup>
    <thead>
    <tr>
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>workspace</td>
        <td>Input</td>
        <td>Workspace memory address requested on the Device side.</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Workspace size requested on the Device side. It is obtained by calling the first-phase API <code>aclnnWeightQuantMatmulAllReduceGetWorkspaceSize</code>.</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>Input</td>
        <td>The op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Specifies the stream for task execution.</td>
    </tr>
    </tbody></table>

- **Return Value**

    Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:

  <!-- npu="910b" id7 -->
  - <term>Atlas A2 products</term>: `aclnnWeightQuantMatmulAllReduce` defaults to a non-deterministic implementation. Deterministic computation can be enabled by setting the `HCCL_DETERMINISTIC` environment variable to true.
  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - Ascend 950PR&950DT products: `aclnnWeightQuantMatmulAllReduce` defaults to a deterministic implementation.
  <!-- end id8 -->

- MC2 is disabled in incremental scenarios and enabled in full scenarios.
- The input x1 can be two-dimensional or three-dimensional, with a shape of (b, s, k) or (m, k).
- x2 must be two-dimensional. Its shape is (k, n), where the k-axis meets the input requirements of the mm operator, the k-axes are equal, m ranges from [1, 2147483647], and k and n range from [1, 65535].
- The passed x1, x2, antiquantScale, or output is not a null pointer.
- When the shape of input x1 is (b, s, k), the shape of x3 (non-null scenario) and output is (b, s, n); when the shape of input x1 is (m, k), the shape of x3 (non-null scenario) and output is (m, n).
- If bias is non-null, its shape size equals the size of the last dimension of output. The shape of antiquantScale is (1) in the pertensor scenario, (1, n)/(n) in the perchannel scenario, and (ceil(k, antiquantGroupSize), n) in the pergroup scenario. If antiquantOffset is non-null, its shape is the same as that of antiquantScale.
- The data types and data formats of x1, x2, x3 (Non-Null Scenario), antiquantScale, antiquantOffset (Non-Null Scenario), output, and bias (Non-Null Scenario) must fall within the supported ranges.
- x1, antiquantScale, antiquantOffset (Non-Null Scenario), x3 (Non-Null Scenario), bias (Non-Null Scenario), and output have the same data type. The value of antiquantGroupSize must fall within the valid range and be a multiple of 32.
- In the pergroup scenario, when x2 is transposed, antiquantScale and antiquantOffset must be transposed together to maintain continuity.
- In long sequence scenarios, as b/s or m increases, OOM or computation timeout may occur.
- Only HCCS all-mesh networking is supported.

    <!-- npu="910b" id9 -->
    - <term>Atlas A2 products</term>: Supports 1, 2, 4, or 8 cards.
    <!-- end id9 -->
    <!-- npu="950" id10 -->
    - <term>Ascend 950PR&950DT products</term>: Supports 1, 2, 4, 8, 16, 32, or 64 cards.
    <!-- end id10 -->

<!-- npu="910b" id11 -->
- <term>Atlas A2 products</term>:
  - The general-purpose computation fused MC2 operators in a model support only the same communication domain.
  - The data format of input x2 supports ND (only two-dimensional input is supported in the current version) and `FRACTAL_NZ` format (only four-dimensional input is supported in the current version). When the data format of x2 is `FRACTAL_NZ`, use aclnnCalculateMatmulWeightSizeV2 and aclnnTransMatmulWeight to convert the input from ND to NZ. Non-contiguous tensors support only the transpose scenario.
<!-- end id11 -->
<!-- npu="950" id12 -->
- <term>Ascend 950PR&950DT products</term>:
  - The data format of input x2 supports ND (2D input only). In the current version, when the data type is INT8, N and K must be 32-aligned; when the data type is INT4, N and K must be 64-aligned.
<!-- end id12 -->
- Null tensor support:
  - Only the scenario where k is 0 is supported, with the output being bias + x3. Null tensor inputs where bs, m, or n is 0 are not supported.

The input and output support the following data type combinations:

<!-- npu="910b" id13 -->
- <term>Atlas A2 products</term>:

  <table style="undefined;table-layout: fixed; width: 600px">
    <col style="width: 90px">
    <col style="width: 125px">
    <col style="width: 125px">
    <col style="width: 125px">
    <col style="width: 125px">
    <col style="width: 125px">
    <col style="width: 100px">
    <col style="width: 100px">
    <thead>
      <tr>
        <th>x1</th>
        <th>x2</th>
        <th>bias</th>
         <th>antiquantScale</th>
        <th>antiquantOffset</th>
        <th>x3</th>
        <th>output</th>
        <th>Restrictions</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>BFLOAT16</td>
        <td>INT8, INT4</td>
        <td>null, BFLOAT16</td>
        <td>BFLOAT16</td>
        <td>null, BFLOAT16</td>
        <td>null, BFLOAT16</td>
        <td>BFLOAT16</td>
        <td>-</td>
      </tr>
      <tr>
        <td>FLOAT16</td>
        <td>INT8, INT4</td>
        <td>null, FLOAT16</td>
        <td>FLOAT16</td>
        <td>null, FLOAT16</td>
        <td>null, FLOAT16</td>
        <td>FLOAT16</td>
        <td>-</td>
      </tr>
    </tbody>
  </table>

<!-- end id13 -->
- Ascend 950PR&950DT products
  <table style="undefined;table-layout: fixed; width: 600px">
    <col style="width: 90px">
    <col style="width: 125px">
    <col style="width: 125px">
    <col style="width: 125px">
    <col style="width: 125px">
    <col style="width: 125px">
    <col style="width: 100px">
    <col style="width: 100px">
    <thead>
      <tr>
        <th>x1</th>
        <th>x2</th>
        <th>bias</th>
        <th>antiquantScale</th>
        <th>antiquantOffset</th>
        <th>x3</th>
        <th>output</th>
        <th>Restrictions</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>BFLOAT16</td>
        <td>INT8, INT4</td>
        <td>null, BFLOAT16</td>
        <td>BFLOAT16</td>
        <td>null, BFLOAT16</td>
        <td>null, BFLOAT16</td>
        <td>BFLOAT16</td>
        <td>Supports pertensor, perchannel, and pergroup quantization scenarios.</td>
      </tr>
      <tr>
        <td>BFLOAT16</td>
        <td>FLOAT8_E4M3FN, HIFLOAT8</td>
        <td>null, BFLOAT16</td>
        <td>BFLOAT16</td>
        <td>null, BFLOAT16</td>
        <td>null, BFLOAT16</td>
        <td>BFLOAT16</td>
        <td>Only the perchannel quantization scenario is supported.</td>
      </tr>
      <tr>
        <td>FLOAT16</td>
        <td>INT8, INT4</td>
        <td>null, FLOAT16</td>
        <td>FLOAT16</td>
        <td>null, FLOAT16</td>
        <td>null, FLOAT16</td>
        <td>FLOAT16</td>
        <td>Pertensor, perchannel, and pergroup quantization scenarios are supported.</td>
      </tr>
      <tr>
        <td>FLOAT16</td>
        <td>FLOAT8_E4M3FN, HIFLOAT8</td>
        <td>null, FLOAT16</td>
        <td>FLOAT16</td>
        <td>null, FLOAT16</td>
        <td>null, FLOAT16</td>
        <td>FLOAT16</td>
        <td>Only the perchannel quantization scenario is supported.</td>
      </tr>
    </tbody>
  </table>

## Example

The following is sample code for reference only. For details about compilation and execution, see Compile and Run Samples.

NOTE: The sample code calls some HCCL collective communication library APIs: HcclGetCommName, HcclCommInitAll, and HcclCommDestroy. For details, see [*HCCL API (C)*](https://hiascend.com/en/document/redirect/CannCommunityHcclCppApi).

<!-- npu="950,910b" id14 -->
- <term>Atlas A2 products</term>, <term>Ascend 950PR&950DT products</term>:

  ```Cpp
  #include <iostream>
  #include <vector>
  #include <thread>
  #include <string.h>
  #include "hccl/hccl.h"
  #include "aclnn/opdev/fp16_t.h"
  #include "aclnnop/aclnn_weight_quant_matmul_all_reduce.h"

  int ndev = 2;

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

  int64_t GetShapeSize(const std::vector<int64_t> &shape) {
      int64_t shapeSize = 1;
      for (auto i: shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  template<typename T>
  int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                      aclDataType dataType, aclTensor **tensor) {
      auto size = GetShapeSize(shape) * sizeof(T);
      // Call aclrtMalloc to allocate device-side memory.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy host-side data to device-side memory.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
      // Compute the strides of the contiguous tensor
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }
      // Call the aclCreateTensor API to create an aclTensor
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
      return 0;
  }

  struct Args {
      uint32_t rankId;
      HcclComm hcclComm;
      aclrtStream stream;
      aclrtContext context;
  };

  int launchOneThreadweightQuantmatmulAllReduce(Args &args) {
      int ret;
      ret = aclrtSetCurrentContext(args.context);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
      char hcom_name[128];
      ret = HcclGetCommName(args.hcclComm, hcom_name);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d \n", ret); return -1);
      LOG_PRINT("[INFO] rank %d hcom: %s stream: %p, context : %p\n", args.rankId, hcom_name, args.stream,
              args.context);

      std::vector<int64_t> x1Shape = {32, 64};
      std::vector<int64_t> x2Shape = {64, 128};
      std::vector<int64_t> biasShape = {128};
      std::vector<int64_t> antiquantScaleShape = {128};
      std::vector<int64_t> antiquantOffsetShape = {128};
      std::vector<int64_t> x3Shape = {32, 128};
      std::vector<int64_t> outShape = {32, 128};
      void *x1DeviceAddr = nullptr;
      void *x2DeviceAddr = nullptr;
      void *biasDeviceAddr = nullptr;
      void *antiquantScaleDeviceAddr = nullptr;
      void *antiquantOffsetDeviceAddr = nullptr;
      void *x3DeviceAddr = nullptr;
      void *outDeviceAddr = nullptr;
      aclTensor *x1 = nullptr;
      aclTensor *x2 = nullptr;
      aclTensor *bias = nullptr;
      aclTensor *antiquantScale = nullptr;
      aclTensor *antiquantOffset = nullptr;
      aclTensor *x3 = nullptr;
      aclTensor *out = nullptr;

      int64_t commTurn = 0;
      int64_t streamMode = 1;
      int64_t antiquantGroupSize = 0;
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor;
      void *workspaceAddr = nullptr;

      long long x1ShapeSize = GetShapeSize(x1Shape);
      long long x2ShapeSize = GetShapeSize(x2Shape);
      long long biasShapeSize = GetShapeSize(biasShape);
      long long antiquantScaleShapeSize = GetShapeSize(antiquantScaleShape);
      long long antiquantOffsetShapeSize = GetShapeSize(antiquantOffsetShape);
      long long x3ShapeSize = GetShapeSize(x3Shape);
      long long outShapeSize = GetShapeSize(outShape);
      std::vector<op::fp16_t> x1HostData(x1ShapeSize, 1);
      std::vector<int8_t> x2HostData(x2ShapeSize, 1);
      std::vector<op::fp16_t> biasHostData(biasShapeSize, 1);
      std::vector<op::fp16_t> antiquantScaleHostData(antiquantScaleShapeSize, 1);
      std::vector<op::fp16_t> antiquantOffsetHostData(antiquantOffsetShapeSize, 1);
      std::vector<op::fp16_t> x3HostData(x3ShapeSize, 1);
      std::vector<op::fp16_t> outHostData(outShapeSize, 0);
      // Create a tensor
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT16, &bias);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(antiquantScaleHostData, antiquantScaleShape, &antiquantScaleDeviceAddr,
                          aclDataType::ACL_FLOAT16, &antiquantScale);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(antiquantOffsetHostData, antiquantOffsetShape, &antiquantOffsetDeviceAddr,
                          aclDataType::ACL_FLOAT16, &antiquantOffset);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(x3HostData, x3Shape, &x3DeviceAddr, aclDataType::ACL_FLOAT16, &x3);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Call the first-phase API
      ret = aclnnWeightQuantMatmulAllReduceGetWorkspaceSize(x1, x2, bias, antiquantScale, antiquantOffset, x3, hcom_name,
                                                          "sum", commTurn, streamMode, antiquantGroupSize, out,
                                                          &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnWeightQuantMatmulAllReduceGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // Allocate device memory based on the workspaceSize calculated by the first-phase API
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // Call the second-phase API
      ret = aclnnWeightQuantMatmulAllReduce(workspaceAddr, workspaceSize, executor, args.stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantMatmulAllReduce failed. ERROR: %d\n", ret); return     ret);
      // (Boilerplate) Synchronously wait for the task execution to complete
      ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
      LOG_PRINT("device%d aclnnWeightQuantMatmulAllReduce execute success \n", args.rankId);
      // Release device resources. Modify the code based on the specific API definition
      if (x1 != nullptr) {
          aclDestroyTensor(x1);
      }
      if (x2 != nullptr) {
          aclDestroyTensor(x2);
      }
      if (bias != nullptr) {
          aclDestroyTensor(bias);
      }
      if (antiquantScale != nullptr) {
          aclDestroyTensor(antiquantScale);
      }
      if (antiquantOffset != nullptr) {
          aclDestroyTensor(antiquantOffset);
      }
      if (x3 != nullptr) {
          aclDestroyTensor(x3);
      }
      if (out != nullptr) {
          aclDestroyTensor(out);
      }
      if (x1DeviceAddr != nullptr) {
          aclrtFree(x1DeviceAddr);
      }
      if (x2DeviceAddr != nullptr) {
          aclrtFree(x2DeviceAddr);
      }
      if (biasDeviceAddr != nullptr) {
          aclrtFree(biasDeviceAddr);
      }
      if (antiquantScaleDeviceAddr != nullptr) {
          aclrtFree(antiquantScaleDeviceAddr);
      }
      if (antiquantOffsetDeviceAddr != nullptr) {
          aclrtFree(antiquantOffsetDeviceAddr);
      }
      if (x3DeviceAddr != nullptr) {
          aclrtFree(x3DeviceAddr);
      }
      if (outDeviceAddr != nullptr) {
          aclrtFree(outDeviceAddr);
      }
      if (workspaceSize > 0) {
          aclrtFree(workspaceAddr);
      }
      aclrtDestroyStream(args.stream);
      HcclCommDestroy(args.hcclComm);
      aclrtDestroyContext(args.context);
      aclrtResetDevice(args.rankId);
      return 0;
  }

  int main(int argc, char *argv[]) {
      int ret;
      int32_t devices[ndev];
      for (int i = 0; i < ndev; i++) {
          devices[i] = i;
      }
      HcclComm comms[128];
      ret = aclInit(nullptr);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
      // Initialize the communicator
      for (int i = 0; i < ndev; i++) {
          ret = aclrtSetDevice(devices[i]);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
      }
      ret = HcclCommInitAll(ndev, devices, comms);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("HcclCommInitAll failed. ERROR: %d\n", ret); return ret);
      Args args[ndev];
      aclrtStream stream[ndev];
      aclrtContext context[ndev];
      for (uint32_t rankId = 0; rankId < ndev; rankId++) {
          ret = aclrtSetDevice(rankId);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
          ret = aclrtCreateContext(&context[rankId], rankId);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
          ret = aclrtCreateStream(&stream[rankId]);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
      }
      // Start multiple threads
      std::vector<std::unique_ptr<std::thread>> threads(ndev);
      for (uint32_t rankId = 0; rankId < ndev; rankId++) {
          args[rankId].rankId = rankId;
          args[rankId].hcclComm = comms[rankId];
          args[rankId].stream = stream[rankId];
          args[rankId].context = context[rankId];
          threads[rankId].reset(
                  new(std::nothrow) std::thread(&launchOneThreadweightQuantmatmulAllReduce, std::ref(args[rankId])));
      }
      for (uint32_t rankId = 0; rankId < ndev; rankId++) {
          threads[rankId]->join();
      }
      aclFinalize();
      return 0;
  }
  ```

<!-- end id14 -->
