# aclnnQuantMatmulAllReduceV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T13:32:55.118Z -->

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

- **Description**: The `aclnnQuantMatmulAllReduceV2` API extends the functions of the `aclnnQuantMatmulAllReduce` API and adds support for the pertoken quantization mode. `aclnnQuantMatmulAllReduceV2` supports the pertensor, perchannel, and pertoken [quantization modes](../../../docs/en/context/quant_mode_introduction.md).

- **Formula**:

    There are two cases:

    - Scenario 1: After performing MatMul on the quantized inputs x1 and x2, dequantization is performed, followed by an Add operation with x3, and finally an AllReduce computation.

        $$
        output= AllReduce(dequantScale*(x1_{int8}@x2_{int8} + bias_{int32}) + x3)
        $$

    - Scenario 2: After performing MatMul on the quantized inputs x1 and x2, dequantization and per-token computation are performed, followed by an Add operation with x3, and finally an AllReduce computation.

        $$
        output= AllReduce(dequantScale * pertokenScaleOptional * (x1_{int8}@x2_{int8} + biasOptional_{int32}) + x3Optional)
        $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call `aclnnQuantMatmulAllReduceV2GetWorkspaceSize` to obtain the workspace size required for the computation and the executor containing the computation process of the operator, and then call `aclnnQuantMatmulAllReduceV2` to execute the computation.

```cpp
aclnnStatus aclnnQuantMatmulAllReduceV2GetWorkspaceSize(
    const aclTensor  *x1,
    const aclTensor  *x2,
    const aclTensor  *biasOptional,
    const aclTensor  *x3Optional,
    const aclTensor  *dequantScale,
    const aclTensor  *pertokenScaleOptional,
    const char       *group,
    const char       *reduceOp,
    int64_t           commTurn,
    int64_t           streamMode,
    const aclTensor  *output,
    uint64_t         *workspaceSize,
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnQuantMatmulAllReduceV2(
    void              *workspace,
    uint64_t           workspaceSize,
    aclOpExecutor     *executor,
    const aclrtStream  stream)
```

## aclnnQuantMatmulAllReduceV2GetWorkspaceSize

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
          <th>Data Type</th>
          <th>Data Format</th>
          <th>Dimension (shape)</th>
          <th>Non-contiguous Tensor</th>
        </tr></thead>
      <tbody>
        <tr>
          <td>x1</td>
          <td>Input</td>
          <td>Left matrix of the MatMul computation, that is, x1 in the formula.</td>
          <td><ul><li>The current version supports only 2D or 3D inputs.</li><li>Supports the non-transpose use case.</li></ul></td>
          <td>INT8</td>
          <td>ND</td>
          <td>2-3</td>
          <td>×</td>
        </tr>
        <tr>
          <td>x2</td>
          <td>Input</td>
          <td>Right matrix of the MatMul computation, that is, x2 in the formula.</td>
          <td><ul><li>The current version supports only 2D inputs.</li><li>Transposed and non-transposed use cases are supported.</li><li>In ND format, non-contiguous tensors with the last two axes transposed are supported; other non-contiguous tensors are not supported.</li></ul></td>
          <td>INT8</td>
          <td>ND, FRACTAL_NZ</td>
          <td>2</td>
          <td>×</td>
        </tr>
        <tr>
          <td>biasOptional</td>
          <td>Input</td>
          <td>biasOptional in the formula.</td>
          <td>The current version supports only 1D inputs.</td>
          <td>INT32</td>
          <td>ND</td>
          <td>0-1</td>
          <td>√</td>
        </tr>
        <tr>
          <td>x3Optional</td>
          <td>Input</td>
          <td>The add computation after the MatMul computation, that is, x3Optional in the computation formula.</td>
          <td>The shape is the same as the shape after the MatMul computation.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>dequantScale</td>
          <td>Input</td>
          <td>Dequantization coefficient after MatMul computation, that is, dequantScale in the formula.</td>
          <td><ul><li>shape is (1) for the pertensor use case and (n)/(1, n) for the PerChannel use case</li><li>When the output is BFLOAT16, directly pass the dequantScale of the BFLOAT16 type to this API.</li><li>When the output is FLOAT16, if pertokenScaleOptional is not empty, directly pass the dequantScale of the FLOAT32 type to this API. If pertokenScaleOptional is empty, call the aclnn API of the TransQuantParamV2 operator in advance to convert dequantScale to the INT64/UINT64 data type.</li></ul></td>
          <td>INT64, UINT64, FLOAT32, BFLOAT16</td>
          <td>ND</td>
          <td>2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>pertokenScaleOptional</td>
          <td>Input</td>
          <td>pertoken dequantization coefficient after the MatMul computation, that is, pertokenScaleOptional in the formula.</td>
          <td>If x1 is (b, s, k), the shape is (b*s); if x1 is (m, k), the shape is (m).</td>
          <td>FLOAT32</td>
          <td>ND</td>
          <td>2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>group</td>
          <td>Input</td>
          <td>Communicator name.</td>
          <td>Obtained through the API provided by Hccl: "extern HcclResult HcclGetCommName(HcclComm comm, char* commName);", where commName is the group.</td>
          <td>String</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>reduceOp</td>
          <td>Input</td>
          <td>Reduce operation type.</td>
          <td>The current version supports only "sum" as the input.</td>
          <td>String</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>commTurn</td>
          <td>Input</td>
          <td>Number of communication data splits, that is, total data volume/single communication volume.</td>
          <td>The current version supports only input 0.</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>streamMode</td>
          <td>Input</td>
          <td>Enum of the stream mode.</td>
          <td>The current version supports only enum value 1.</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>Output</td>
          <td>Output</td>
          <td>Result of the MatMul computation and AllReduce communication, that is, output in the formula.</td>
          <td>output has the same number of dimensions as x1.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>2-3</td>
          <td>√</td>
        </tr>
        <tr>
          <td>workspaceSize</td>
          <td>Output</td>
          <td>Returns the workspace size to be requested on the device.</td>
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

    <!-- npu="910b" id7 -->
    - <term>Atlas A2 products</term>:

        - The data format of input x2 supports ND (only two-dimensional input is supported in the current version) and `FRACTAL_NZ` format (only four-dimensional input is supported in the current version). When the data format of x2 is `FRACTAL_NZ`, use `aclnnCalculateMatmulWeightSizeV2` and `aclnnTransMatmulWeight` to complete the conversion from ND data format to NZ data format. Non-contiguous tensors only support the transpose scenario.

    <!-- end id7 -->
    <!-- npu="950" id8 -->
    - <term>Ascend 950PR&950DT products</term>:

        - The data format of input x2 only supports the ND format (the current version only supports two-dimensional input).

    <!-- end id8 -->

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
        <th>Return Value</th>
        <th>Error Code</th>
        <th>Description</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>x1, x2, dequantScale, or output is a null pointer.</td>
    </tr>
    <tr>
        <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="3">161002</td>
        <td>The data types of x1, x2, biasOptional, dequantScale, pertokenScaleOptional, x3Optional, or output are not within the supported range.</td>
    </tr>
    <tr>
        <td>streamMode is not within the valid range.</td>
    </tr>
    <tr>
        <td>The shapes of x1, x2, biasOptional, dequantScale, pertokenScaleOptional, x3Optional, or output do not meet the constraint requirements.</td>
    </tr>
    </tbody>
    </table>

## aclnnQuantMatmulAllReduceV2

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
        <td>Address of the workspace memory allocated on the device.</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Size of the workspace allocated on the device, obtained by calling the first-phase API <code>aclnnQuantMatmulAllReduceV2GetWorkspaceSize</code>.</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Op executor, which contains the computation flow of the operator.</td>
    </tr>
    <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Specifies the stream on which the task is executed.</td>
    </tr>
    </tbody></table>

- **Return Value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:

  <!-- npu="910b" id9 -->
  - <term>Atlas A2 products</term>: `aclnnQuantMatmulAllReduceV2` defaults to a non-deterministic implementation. Deterministic computation can be enabled by setting the `HCCL_DETERMINISTIC` environment variable to true.
  <!-- end id9 -->
  <!-- npu="950" id10 -->
  - Ascend 950PR&950DT products: `aclnnQuantMatmulAllReduceV2` defaults to a deterministic implementation.
  <!-- end id10 -->

- MC2 is disabled in incremental use cases and enabled in full use cases.
- The input x1 can be two-dimensional or three-dimensional and must be a non-empty tensor, with its shape being (b, s, k) or (m, k). x2 must be two-dimensional and a non-empty tensor. Its shape is (k, n), and the k axis must satisfy the input requirements of the mm operator, meaning the k axes must be equal.
- The size of m must not exceed 2147483647. The last dimension size of x1 and x2 must not exceed 65535. The last dimension of x1 refers to k, and the last dimension of x2 refers to k when transposed or n when not transposed.
- If bias is non-empty, its shape is (n). If x3 is non-empty, its shape must be the same as that of output.
- When the shape of input x1 is (b, s, k), the shape of output is (b, s, n). When the shape of input x1 is (m, k), the shape of output is (m, n).
- The passed x1, x2, dequantScale, or output must not be null pointers.
- The data types and data formats of x1, x2, dequantScale, output, bias (when not null), and x3 (when not null) must fall within the supported ranges.
- If the output type is FLOAT16, when pertokenScaleOptional is empty, the type of dequantScale is INT64 or UINT64; when pertokenScaleOptional is not empty, the type of dequantScale is FLOAT32.
- If the output type is BFLOAT16, the type of dequantScale is BFLOAT16.
- When the shape of x1 is (b, s, k), the shape of pertokenScaleOptional is (b*s); when the shape of x1 is (m, k), the shape of pertokenScaleOptional is (m).
- Only HCCS all-mesh networking is supported.

    <!-- npu="910b" id11 -->
    - <term>Atlas A2 products</term>: Supports 1, 2, 4, or 8 cards.
    <!-- end id11 -->
    <!-- npu="950" id12 -->
    - <term>Ascend 950PR&950DT products</term>: Supports 1, 2, 4, 8, 16, 32, or 64 cards.
    <!-- end id12 -->

<!-- npu="910b" id13 -->
- <term>Atlas A2 products</term>: A communication-computation fused MC2 operator in a model only supports the same communication domain.
<!-- end id13 -->
- Null tensor support:
  - Empty tensors are not supported.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

Note: The sample code calls some HCCL collective communication library APIs: HcclGetCommName, HcclCommInitAll, and HcclCommDestroy. For details, see [<<HCCL API (C)>>](https://hiascend.com/en/document/redirect/CannCommunityHcclCppApi).

<!-- npu="950,910b" id14 -->
- <term>Atlas A2 products</term>, <term>Ascend 950PR&950DT products</term>:

    ```Cpp
    #include <iostream>
    #include <vector>
    #include <thread>
    #include "hccl/hccl.h"
    #include "aclnn/opdev/fp16_t.h"
    #include "aclnnop/aclnn_trans_matmul_weight.h"
    #include "aclnnop/aclnn_quant_matmul_all_reduce_v2.h"

    int ndev = 2;

    #define ACL_CHECK(ret)                                                                                     \
        do {                                                                                                   \
            auto retcode = ret;                                                                                \
            if (retcode != ACL_SUCCESS) {                                                                      \
                printf("[ERROR] acl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, retcode); \
                return retcode;                                                                                \
            }                                                                                                  \
        } while (0)

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

    struct Args {
        uint32_t rankId;
        HcclComm hcclComm;
        aclrtStream stream;
        aclrtContext context;
        std::string format;
    };

    template<typename T>
    int CreateWeightNzAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                                aclDataType dataType, aclTensor **tensor, Args &args) {
        auto size = GetShapeSize(shape) * sizeof(T);
        const aclIntArray *mat2Size = aclCreateIntArray(shape.data(), shape.size());
        auto ret = aclnnCalculateMatmulWeightSizeV2(mat2Size, ACL_INT8, &size);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCalculateMatmulWeightSizeV2 failed. ERROR: %d\n", ret); return    ret);

        // Call aclrtMalloc to allocate device memory.
        ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

        // Call aclrtMemcpy to copy data from the host to the device memory.
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

        // Compute the strides of a contiguous tensor.
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = shape.size() - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
        // Call aclCreateTensor to create an aclTensor.
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);

        uint64_t transWorkspaceSize;
        aclOpExecutor *executor;
        void *transWorkspaceAddr = nullptr;
        ret = aclnnTransMatmulWeightGetWorkspaceSize(*tensor, &transWorkspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS && transWorkspaceSize > 0,
                printf("[ERROR] aclnnTransMatmulWeightGetWorkspaceSize failed. ret = %d \n", ret); return ret);
        ACL_CHECK(aclrtMalloc(&transWorkspaceAddr, transWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
        ret = aclnnTransMatmulWeight(transWorkspaceAddr, transWorkspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnTransMatmulWeight failed. ret = %d \n", ret);return ret);
        ACL_CHECK(aclrtSynchronizeStreamWithTimeout(args.stream, 20000));

        return 0;
    }

    template<typename T>
    int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                        aclDataType dataType, aclTensor **tensor) {
        auto size = GetShapeSize(shape) * sizeof(T);
        // Call aclrtMalloc to allocate device-side memory.
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
        // Call aclrtMemcpy to copy data from the host to the device memory.
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
        // Compute the strides of a contiguous tensor.
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = shape.size() - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
        // Call aclCreateTensor to create an aclTensor.
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
        return 0;
    }

    int launchOneThreadQuantMatmulAllReduce(Args &args) {
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
        std::vector<int64_t> dequantScaleShape = {128};
        std::vector<int64_t> pertokenScaleShape = {32};
        std::vector<int64_t> x3Shape = {32, 128};
        std::vector<int64_t> outShape = {32, 128};
        void *x1DeviceAddr = nullptr;
        void *x2DeviceAddr = nullptr;
        void *biasDeviceAddr = nullptr;
        void *dequantScaleDeviceAddr = nullptr;
        void *pertokenScaleDeviceAddr = nullptr;
        void *x3DeviceAddr = nullptr;
        void *outDeviceAddr = nullptr;
        aclTensor *x1 = nullptr;
        aclTensor *x2 = nullptr;
        aclTensor *bias = nullptr;
        aclTensor *dequantScale = nullptr;
        aclTensor *pertokenScale = nullptr;
        aclTensor *x3 = nullptr;
        aclTensor *out = nullptr;

        int64_t commTurn = 0;
        int64_t streamMode = 1;
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;
        void *workspaceAddr = nullptr;

        long long x1ShapeSize = GetShapeSize(x1Shape);
        long long x2ShapeSize = GetShapeSize(x2Shape);
        long long biasShapeSize = GetShapeSize(biasShape);
        long long dequantScaleShapeSize = GetShapeSize(dequantScaleShape);
        long long pertokenScaleShapeSize = GetShapeSize(pertokenScaleShape);
        long long x3ShapeSize = GetShapeSize(x3Shape);
        long long outShapeSize = GetShapeSize(outShape);

        std::vector<int8_t> x1HostData(x1ShapeSize, 1);
        std::vector<int8_t> x2HostData(x2ShapeSize, 1);
        std::vector<int32_t> biasHostData(biasShapeSize, 1);
        std::vector<float> dequantScaleHostData(dequantScaleShapeSize, 1);
        std::vector<float> pertokenScaleHostData(pertokenScaleShapeSize, 1);
        std::vector<op::fp16_t> x3HostData(x3ShapeSize, 1);
        std::vector<op::fp16_t> outHostData(outShapeSize, 0);
        // Create a tensor.
        ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        if (args.format == "NZ") {
            ret = CreateWeightNzAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2, args);
        } else {
            ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
        }
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_INT32, &bias);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(dequantScaleHostData, dequantScaleShape, &dequantScaleDeviceAddr,
                            aclDataType::ACL_FLOAT, &dequantScale);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(pertokenScaleHostData, pertokenScaleShape, &pertokenScaleDeviceAddr,
                            aclDataType::ACL_FLOAT, &pertokenScale);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x3HostData, x3Shape, &x3DeviceAddr, aclDataType::ACL_FLOAT16, &x3);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Call the first-phase API
        ret = aclnnQuantMatmulAllReduceV2GetWorkspaceSize(x1, x2, bias, x3, dequantScale, pertokenScale,
                                                        hcom_name, "sum", commTurn, streamMode, out,
                                                        &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclnnQuantMatmulAllReduceV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
        // Allocate device memory based on the workspaceSize computed by the first-phase API
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        // Call the second-phase API
        ret = aclnnQuantMatmulAllReduceV2(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulAllReduceV2 failed. ERROR: %d\n", ret); return ret);
        // (boilerplate) Synchronously wait for the task to complete
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
        LOG_PRINT("device%d aclnnQuantMatmulAllReduceV2 execute success \n", args.rankId);
        // Release device resources. Modify based on the definition of the specific API
        if (x1 != nullptr) {
            aclDestroyTensor(x1);
        }
        if (x2 != nullptr) {
            aclDestroyTensor(x2);
        }
        if (bias != nullptr) {
            aclDestroyTensor(bias);
        }
        if (dequantScale != nullptr) {
            aclDestroyTensor(dequantScale);
        }
        if (pertokenScale != nullptr) {
            aclDestroyTensor(pertokenScale);
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
        if (dequantScaleDeviceAddr != nullptr) {
            aclrtFree(dequantScaleDeviceAddr);
        }
        if (pertokenScaleDeviceAddr != nullptr) {
            aclrtFree(pertokenScaleDeviceAddr);
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
                    new(std::nothrow) std::thread(&launchOneThreadQuantMatmulAllReduce, std::ref(args[rankId])));
        }
        for (uint32_t rankId = 0; rankId < ndev; rankId++) {
            threads[rankId]->join();
        }
        aclFinalize();
        return 0;
    }
    ```

<!-- end id14 -->
