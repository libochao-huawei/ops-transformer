# aclnnQuantMatmulAllReduceV3

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T13:34:23.454Z -->

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

- **Description**: The `aclnnQuantMatmulAllReduceV3` API extends the function of the `aclnnQuantMatmulAllReduceV2` API by adding support for low-bit communication: the MatMul computation result sequentially undergoes AllToAll communication, ReduceSum computation, AllGather communication, and dequant, replacing the original process of performing dequant and pertoken computation first and then AllReduce communication. Supports pertensor, perchannel, and pertoken [quantization modes](../../../docs/en/context/quant_mode_introduction.md).

- **Formula**:

    into the following three scenarios:

    - Scenario 1: After performing MatMul on the quantized inputs x1 and x2, dequantization is performed, followed by an Add operation with x3, and finally an AllReduce computation.

      $$
      output= AllReduce(dequantScale*(x1_{int8}@x2_{int8} + bias_{int32}) + x3)
      $$

    - Scenario 2: After performing MatMul on the quantized inputs x1 and x2, dequantization and per-token computation are performed, followed by an Add operation with x3, and finally an AllReduce computation.

      $$
      output= AllReduce(dequantScale * pertokenScaleOptional * (x1_{int8}@x2_{int8} + biasOptional_{int32}) + x3Optional)
      $$

    - Scenario 3: Perform MatMul, dequant, and pertoken computation on the quantized inputs x1 and x2, then perform an Add operation with x3, apply perchannel quantization to the output, then perform AllToAll communication, perform ReduceSum computation on the first communication result, then perform AllGather communication, and finally dequant the second communication result to obtain the final output.

      $$
      matmulAddOutPut = (dequantScale * pertokenScaleOptional * (x1_{int8}@x2_{int8} + biasOptional_{int32}) + x3Optional);
      $$

      $$
      alltoallOutPut_{int8} = AllToAll(matmulAddOutPut / commQuantScale1Optional);
      $$

      $$
      reduceSumOutPut_{int8} = (add(alltoallOutPut_{int8}) * (commQuantScale1Optional / commQuantScale2Optional));
      $$

      $$
      outPut = (AllGather(reduceSumOutPut_{int8}) * commQuantScale2Optional);
      $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the `aclnnQuantMatmulAllReduceV3GetWorkspaceSize` API first to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the `aclnnQuantMatmulAllReduceV3` API to execute the computation.

```cpp
aclnnStatus aclnnQuantMatmulAllReduceV3GetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *biasOptional,
    const aclTensor *x3Optional,
    const aclTensor *dequantScale,
    const aclTensor *pertokenScaleOptional,
    const aclTensor *commQuantScale1Optional,
    const aclTensor *commQuantScale2Optional,
    const char      *group,
    const char      *reduceOp,
    int64_t          commTurn,
    int64_t          streamMode,
    const aclTensor *output,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnQuantMatmulAllReduceV3(
    void              *workspace,
    uint64_t           workspaceSize,
    aclOpExecutor     *executor,
    const aclrtStream  stream)
```

## aclnnQuantMatmulAllReduceV3GetWorkspaceSize

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
          <td>Left matrix of the MatMul computation, that is, x1 in the computation formula.</td>
          <td><ul><li>The current version only supports two-dimensional or three-dimensional input.</li><li>Supports the non-transpose use case.</li></ul></td>
          <td>INT8</td>
          <td>ND</td>
          <td>2-3</td>
          <td>×</td>
        </tr>
        <tr>
          <td>x2</td>
          <td>Input</td>
          <td>Right matrix of the MatMul computation, that is, x2 in the computation formula.</td>
          <td><ul><li>Supports transpose/no-transpose use cases.</li><li>Supports non-contiguous tensors with the last two axes transposed in ND format. Other non-contiguous tensors are not supported.</li></ul></td>
          <td>INT8</td>
          <td>ND, FRACTAL_NZ</td>
          <td>2</td>
          <td>×</td>
        </tr>
        <tr>
          <td>biasOptional</td>
          <td>Input</td>
          <td>Bias offset, that is, biasOptional in the computation formula.</td>
          <td>The current version only supports one-dimensional input.</td>
          <td>INT32</td>
          <td>ND</td>
          <td>1</td>
          <td>√</td>
        </tr>
        <tr>
          <td>x3Optional</td>
          <td>Input</td>
          <td>Add computation computed after MatMul, which is x3Optional in the computation formula.</td>
          <td>The data type and dimensions must be consistent with output.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>dequantScale</td>
          <td>Input</td>
          <td>Dequantization scale computed by MatMul, that is, dequantScale in the computation formula.</td>
          <td><ul><li>Shape is (1) in the pertensor use case and (n)/(1, n) in the perchannel use case.</li><li>When the output is BFLOAT16, directly pass dequantScale of the BFLOAT16 type to this API.</li><li>When the output is FLOAT16, if pertokenScale is not empty, directly pass dequantScale of the FLOAT32 type to this API; if pertokenScale is empty, call the aclnn API of the TransQuantParamV2 operator in advance to convert dequantScale to the INT64/UINT64 data type.</li></ul></td>
          <td>INT64, UINT64, FLOAT32, BFLOAT16</td>
          <td>ND</td>
          <td>2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>pertokenScaleOptional</td>
          <td>Input</td>
          <td>pertoken dequantization scale computed by MatMul, that is, pertokenScaleOptional in the computation formula.</td>
          <td>If x1 is (b, s, k), the shape is (b*s); if x1 is (m, k), the shape is (m).</td>
          <td>FLOAT32</td>
          <td>ND</td>
          <td>2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>commQuantScale1Optional</td>
          <td>Input</td>
          <td>perchannel quantization scale computed by matmulAdd, that is, commQuantScale1Optional in the computation formula.</td>
          <td>When x2 is (k, n), the shape can be (n) or (1,n).</td>
          <td>BFLOAT16, FLOAT16</td>
          <td>ND</td>
          <td>2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>commQuantScale2Optional</td>
          <td>Input</td>
          <td>Perchannel quantization scale computed by allGather, that is, commQuantScale2Optional in the computation formula.</td>
          <td>When x2 is (k, n), the shape can be (n) or (1,n).</td>
          <td>BFLOAT16, FLOAT16</td>
          <td>ND</td>
          <td>2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>group</td>
          <td>Input</td>
          <td>Communicator name.</td>
          <td>Obtained through the Hccl API "extern HcclResult HcclGetCommName(HcclComm comm, char* commName);", where commName is the group.</td>
          <td>String</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>reduceOp</td>
          <td>Input</td>
          <td>Reduce operation type.</td>
          <td>Currently, the input only supports "sum".</td>
          <td>String</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>commTurn</td>
          <td>Input</td>
          <td>Number of communication data splits, that is, total data volume/single communication volume.</td>
          <td>The current version only supports input 0.</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>streamMode</td>
          <td>Input</td>
          <td>Enumeration of stream modes.</td>
          <td>Currently, only enumeration value 1 is supported.</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>Output</td>
          <td>Output</td>
          <td>Result of MatMul computation and AllReduce communication, that is, output in the computation formula.</td>
          <td>The number of dimensions of output is consistent with that of x1.</td>
          <td>FLOAT16, BFLOAT16</td>
          <td>ND</td>
          <td>2-3</td>
          <td>√</td>
        </tr>
        <tr>
          <td>workspaceSize</td>
          <td>Output</td>
          <td>Returns the workspace size that needs to be side-requested on the Device.</td>
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

      - The data format of input x2 supports ND and `FRACTAL_NZ` formats. The shape rules for the input are as follows:
        - When the data format of x2 is `FRACTAL_NZ`, the current version only supports four-dimensional input. Use `aclnnCalculateMatmulWeightSizeV2` and `aclnnTransMatmulWeight` to complete the conversion from ND to NZ. Non-contiguous tensors only support the transpose scenario.
        - When the data format of x2 is ND, the current version only supports two-dimensional input.

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
        <th>Return value</th>
        <th>Error code</th>
        <th>Description</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>The passed-in x1, x2, dequantScale, or output is a null pointer.</td>
    </tr>
    <tr>
        <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="3">161002</td>
        <td>The data type of x1, x2, biasOptional, dequantScale, pertokenScaleOptional, x3Optional, commQuantScale1Optional, commQuantScale2Optional, or output is not supported.</td>
    </tr>
    <tr>
        <td>streamMode is not within the valid range.</td>
    </tr>
    <tr>
        <td>The shape of x1, x2, biasOptional, dequantScale, pertokenScaleOptional, x3Optional, commQuantScale1Optional, commQuantScale2Optional, or output does not meet the constraints.</td>
    </tr>
    </tbody>
    </table>

## aclnnQuantMatmulAllReduceV3

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
        <td>The Device-side-requested workspace memory address.</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>The Device-side-requested workspace size, obtained by the first-phase API <code>aclnnQuantMatmulAllReduceV3GetWorkspaceSize</code>.</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream for executing tasks.</td>
    </tr>
    </tbody></table>

- **Return Value**

    Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:

  <!-- npu="910b" id9 -->
  - <term>Atlas A2 products</term>: `aclnnQuantMatmulAllReduceV3` defaults to a non-deterministic implementation. Deterministic computation can be enabled by setting the `HCCL_DETERMINISTIC` environment variable to true.
  <!-- end id9 -->
  <!-- npu="950" id10 -->
  - Ascend 950PR&950DT products: `aclnnQuantMatmulAllReduceV3` defaults to a deterministic implementation.
  <!-- end id10 -->

- MC2 is disabled in incremental use cases and enabled in full use cases.
- The input x1 can be 2D or 3D and must not be an empty tensor, with shape (b, s, k) or (m, k). x2 must be 2D and must not be an empty tensor, with shape (k, n). The k axes must satisfy the input requirements of the mm operator, meaning the k axes must be equal.
- The size of m must not exceed 2147483647. The last dimension size of x1 and x2 must not exceed 65535. The last dimension of x1 refers to k, and the last dimension of x2 refers to k when transposed or n when not transposed.
- The passed x1, x2, dequantScale, or output must not be null pointers.
- The data types and data formats of x1, x2, dequantScale, output, biasOptional (when not null), and x3 (when not null) must fall within the supported ranges.
- If the output type is FLOAT16: when pertokenScale is null, the type of dequantScale is INT64 or UINT64; when pertokenScale is not null, the type of dequantScale is FLOAT32. If the output type is BFLOAT16, the type of dequantScale is BFLOAT16 and the type of x3 is BFLOAT16.
- The passed commQuantScale1 and commQuantScale2 must be either both null pointers or both non-null pointers. If both commQuantScale1 and commQuantScale2 are non-null pointers, the shapes of the two quantization parameters must be consistent, their types must be consistent with the operator output type, and the inputs on each card must be consistent.
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
- int8 low-bit communication provides performance gains only in communication-bound cases. It is not recommended in compute-bound cases, that is, do not pass in commQuantScale1 and commQuantScale2.
- Null tensor support:
  - Empty tensors are not supported.

## Example

The following sample code is for reference only. For details about the compilation and execution procedure, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

NOTE: This sample code calls some HCCL collective communication APIs: HcclGetCommName, HcclCommInitAll, and HcclCommDestroy. For details, see [*HCCL API (C)*](https://hiascend.com/en/document/redirect/CannCommunityHcclCppApi).

<!-- npu="950,910b" id14 -->
- <term>Atlas A2 products</term>, <term>Ascend 950PR&950DT products</term>:

    ```Cpp
    #include <iostream>
    #include <vector>
    #include <thread>
    #include "hccl/hccl.h"
    #include "aclnn/opdev/fp16_t.h"
    #include "aclnnop/aclnn_trans_matmul_weight.h"
    #include "aclnnop/aclnn_quant_matmul_all_reduce_v3.h"

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

        // Call aclrtMemcpy to copy host-side data to device-side memory.
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

        // Calculate the strides of a contiguous tensor.
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
        // Call aclrtMemcpy to copy host-side data to device-side memory.
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
        // Calculate the strides of a contiguous tensor.
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
        std::vector<int64_t> commQuantScale1Shape = {128};
        std::vector<int64_t> commQuantScale2Shape = {128};
        std::vector<int64_t> x3Shape = {32, 128};
        std::vector<int64_t> outShape = {32, 128};
        void *x1DeviceAddr = nullptr;
        void *x2DeviceAddr = nullptr;
        void *biasDeviceAddr = nullptr;
        void *dequantScaleDeviceAddr = nullptr;
        void *pertokenScaleDeviceAddr = nullptr;
        void *commQuantScale1DeviceAddr = nullptr;
        void *commQuantScale2DeviceAddr = nullptr;
        void *x3DeviceAddr = nullptr;
        void *outDeviceAddr = nullptr;
        aclTensor *x1 = nullptr;
        aclTensor *x2 = nullptr;
        aclTensor *bias = nullptr;
        aclTensor *dequantScale = nullptr;
        aclTensor *pertokenScale = nullptr;
        aclTensor *commQuantScale1 = nullptr;
        aclTensor *commQuantScale2 = nullptr;
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
        long long commQuantScale1ShapeSize = GetShapeSize(commQuantScale1Shape);
        long long commQuantScale2ShapeSize = GetShapeSize(commQuantScale2Shape);
        long long x3ShapeSize = GetShapeSize(x3Shape);
        long long outShapeSize = GetShapeSize(outShape);

        std::vector<int8_t> x1HostData(x1ShapeSize, 1);
        std::vector<int8_t> x2HostData(x2ShapeSize, 1);
        std::vector<int32_t> biasHostData(biasShapeSize, 1);
        std::vector<float> dequantScaleHostData(dequantScaleShapeSize, 1);
        std::vector<float> pertokenScaleHostData(pertokenScaleShapeSize, 1);
        std::vector<op::fp16_t> commQuantScale1HostData(commQuantScale1ShapeSize, 1);
        std::vector<op::fp16_t> commQuantScale2HostData(commQuantScale2ShapeSize, 1);
        std::vector<op::fp16_t> x3HostData(x3ShapeSize, 1);
        std::vector<op::fp16_t> outHostData(outShapeSize, 0);
        // Create tensor.
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
        ret = CreateAclTensor(commQuantScale1HostData, commQuantScale1Shape, &commQuantScale1DeviceAddr,
                              aclDataType::ACL_FLOAT16, &commQuantScale1);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(commQuantScale2HostData, commQuantScale2Shape, &commQuantScale2DeviceAddr,
                              aclDataType::ACL_FLOAT16, &commQuantScale2);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x3HostData, x3Shape, &x3DeviceAddr, aclDataType::ACL_FLOAT16, &x3);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Call the first-phase API.
        ret = aclnnQuantMatmulAllReduceV3GetWorkspaceSize(x1, x2, bias, x3, dequantScale, pertokenScale,
                                                          commQuantScale1, commQuantScale2, hcom_name,
                                                          "sum", commTurn, streamMode, out,
                                                          &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("aclnnQuantMatmulAllReduceV3GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
        // Allocate device memory based on the workspaceSize computed by the first-phase API.
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        // Call the second-phase API.
        ret = aclnnQuantMatmulAllReduceV3(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulAllReduceV3 failed. ERROR: %d\n", ret); return ret);
        // (Boilerplate) Wait synchronously for the task to finish.
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
        LOG_PRINT("device%d aclnnQuantMatmulAllReduceV3 execute success \n", args.rankId);
        // Release device resources; modify based on the specific API definition.
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
        if (commQuantScale1 != nullptr) {
            aclDestroyTensor(commQuantScale1);
        }
        if (commQuantScale2 != nullptr) {
            aclDestroyTensor(commQuantScale2);
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
        if (commQuantScale1DeviceAddr != nullptr) {
            aclrtFree(commQuantScale1DeviceAddr);
        }
        if (commQuantScale2DeviceAddr != nullptr) {
            aclrtFree(commQuantScale2DeviceAddr);
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
        // Initialize the communicator.
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
        // Start multiple threads.
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
