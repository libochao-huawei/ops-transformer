# aclnnGroupedMatMulAlltoAllv

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T13:28:07.467Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
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

- Description: Fuses GroupedMatMul, Unpermute, and AlltoAllv for routed experts and fuses in parallel with MatMul for shared experts, with **computation before communication**.

- Formula:

    - Routed experts:

    $$
    gmmY = gmmX \times gmmWeight \\
    unpermuteOut = Unpermute(gmmY) \\
    y = AlltoAllv(unpermuteOut)
    $$

    - Shared expert:

    $$
    mmY = mmX \times mmWeight
    $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call `aclnnGroupedMatMulAlltoAllvGetWorkspaceSize` first to obtain the input parameters and calculate the required workspace size based on the calculation process, and then call `aclnnGroupedMatMulAlltoAllv` to execute the computation.

```cpp
aclnnStatus aclnnGroupedMatMulAlltoAllvGetWorkspaceSize(
    const aclTensor*   gmmX,
    const aclTensor*   gmmWeight,
    const aclTensor*   sendCountsTensorOptional,
    const aclTensor*   recvCountsTensorOptional,
    const aclTensor*   mmXOptional,
    const aclTensor*   mmWeightOptional,
    const char*        group,
    int64_t            epWorldSize,
    const aclIntArray* sendCounts,
    const aclIntArray* recvCounts,
    bool               transGmmWeight,
    bool               transMmWeight,
    aclTensor*         y,
    aclTensor*         mmYOptional,
    uint64_t*          workspaceSize,
    aclOpExecutor**    executor)
```

```cpp
aclnnStatus aclnnGroupedMatMulAlltoAllv(
    void*           workspace,
    uint64_t        workspaceSize,
    aclOpExecutor*  executor,
    aclrtStream     stream)
```

## aclnnGroupedMatMulAlltoAllvGetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1440px"> <colgroup>
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 200px">
    <col style="width: 250px">
    <col style="width: 150px">
    <col style="width: 180px">
    <col style="width: 180px">
    <col style="width: 180px">
    </colgroup>
    <thead>
    <tr>
    <th>Parameter</th>
    <th>Input/Output</th>
    <th>Description</th>
    <th>Instruction</th>
    <th>Data Type</th>
    <th>Data Format</th>
    <th>Shape</th>
    <th>Non-contiguous Tensor</th>
    </tr></thead>
    <tbody>
    <tr>
    <td>gmmX(aclTensor*)</td>
    <td>Input</td>
    <td>The left matrix of the GroupedMatMul computation.</td>
    <td><ul><li>The data type must be the same as that of gmmWeight.</li><li>Supports two dimensions, with the shape (A, H1).</li></ul></td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
    </tr>
    <tr>
    <td>gmmWeight(aclTensor*)</td>
    <td>Input</td>
    <td>The right matrix of the GroupedMatMul computation.</td>
    <td><ul><li>Data type is consistent with gmmX.</li><li>Supports 3 dimensions, with shape (e, H1, N1).</li></ul></td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>3</td>
    <td>×</td>
    </tr>
    <tr>
    <td>sendCountsTensorOptional (aclTensor*)</td>
    <td>Optional input</td>
    <td>Shape is (e * epWorldSize,).</td>
    <td>Not supported in the current version. Must be set to nullptr.</td>
    <td>INT32, INT64</td>
    <td>ND</td>
    <td>1</td>
    <td>×</td>
    </tr>
    <tr>
    <td>recvCountsTensorOptional (aclTensor*)</td>
    <td>Optional input</td>
    <td>The shape is (e * epWorldSize,).</td>
    <td>Not supported in the current version. Pass nullptr.</td>
    <td>INT32, INT64</td>
    <td>ND</td>
    <td>1</td>
    <td>×</td>
    </tr>
    <tr>
    <td>mmXOptional(aclTensor*)</td>
    <td>Optional input</td>
    <td>Left matrix in the shared expert MatMul computation.</td>
    <td><ul><li>Must be passed together with mmWeightOptional and mmYOptional, or all set to nullptr.</li><li>The data type must be the same as gmmX.</li><li>Supports 2 dimensions with shape (BS, H2).</li></ul></td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
    </tr>
    <tr>
    <td>mmWeightOptional(aclTensor*)</td>
    <td>Optional input</td>
    <td>Right matrix in the shared expert MatMul computation.</td>
    <td><ul><li>Must be passed together with mmXOptional and mmYOptional, or all set to nullptr.</li><li>Data type must be the same as that of gmmX.</li><li>Supports 2 dimensions, with shape (H2, N2).</li></ul></td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
    </tr>
    <tr>
    <td>group(char*)</td>
    <td>Input</td>
    <td>Communicator name for expert parallelism.</td>
    <td>The string length must be greater than 0 and less than 128.</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>epWorldSize (int64_t)</td>
    <td>Input</td>
    <td>Size of the ep communicator.</td>
    <td><ul><li><term>Atlas A3 products</term> supports 8, 16, 32, 64, and 128.</li><li><term>Ascend 950PR&950DT products</term> supports 2, 4, 8, 16, 32, and 64.</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sendCounts(aclIntArray*)</td>
    <td>Input</td>
    <td>Number of tokens sent to other devices.</td>
    <td><ul><li>Input type must be aclIntArray*, with elements of type INT64.</li><li>List size is e * epWorldSize, up to a maximum of 256.</li></ul></td>
    <td>aclIntArray* (element type INT64)</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>recvCounts(aclIntArray*)</td>
    <td>Input</td>
    <td>Indicates the number of tokens received from other devices.</td>
    <td><ul><li>The input type must be aclIntArray*, with the element type INT64.</li><li>The list size is e * epWorldSize, with a maximum of 256.</li></ul></td>
    <td>aclIntArray* (element type INT64)</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>transGmmWeight(bool)</td>
    <td>Input</td>
    <td>Whether gmmWeight needs to be transposed.</td>
    <td>true indicates transpose is required, and false indicates no transpose.</td>
    <td>BOOL</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>transMmWeight(bool)</td>
    <td>Input</td>
    <td>Whether shared expert mmWeightOptional needs to be transposed.</td>
    <td>true indicates that transpose is required, and false indicates no transpose.</td>
    <td>BOOL</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>y(aclTensor*)</td>
    <td>Output</td>
    <td>Final computation result.</td>
    <td><ul><li>The data type is the same as that of input gmmX.</li><li>Supports 2 dimensions, with shape (BSK, N1).</li></ul></td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
    </tr>
    <tr>
    <td>mmYOptional(aclTensor*)</td>
    <td>Output</td>
    <td>Output of the shared expert MatMul.</td>
    <td><ul><li>Data type is the same as that of mmXOptional.</li><li>Supports 2 dimensions, with shape (BS, N2).</li><li>Output only when both mmXOptional and mmWeightOptional are passed; otherwise, nullptr must be passed.</li></ul></td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
    </tr>
    <tr>
    <td>workspaceSize(uint64_t*)</td>
    <td>Output</td>
    <td>Returns the workspace size to be allocated on the Device.</td>
    <td>-</td>
    <td>UINT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor(aclOpExecutor**)</td>
    <td>Output</td>
    <td>Returns the op executor, which contains the computation flow of the operator.</td>
    <td>-</td>
    <td>aclOpExecutor*</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    </tbody></table>

- **Return Value**

    Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API performs input parameter validation and returns an error in the following scenarios:

    <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
    <col style="width: 282px">
    <col style="width: 120px">
    <col style="width: 747px">
    </colgroup>
    <thead>
    <tr>
    <th>Return Values</th>
    <th>Error Codes</th>
    <th>Description</th>
    </tr></thead>
    <tbody>
    <tr>
    <td>ACLNN_ERR_PARAM_NULLPTR</td>
    <td>161001</td>
    <td>The parameter is a required input, output, or required attribute, but a null pointer is passed.</td>
    </tr>
    <tr>
    <td>ACLNN_ERR_PARAM_INVALID</td>
    <td>161002</td>
    <td>The data type, data format, or dimension of gmmX, gmmWeight, sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional, group, epWorldSize, sendCounts, or recvCounts is out of the supported range.</td>
    </tr>
    </tbody></table>

## aclnnGroupedMatMulAlltoAllv

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
    <td>Workspace memory address allocated on the device.</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>Input</td>
    <td>Size of the workspace allocated on the device, obtained by calling the first-phase API <code>aclnnGroupedMatMulAlltoAllvGetWorkspaceSize</code>.</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>Input</td>
    <td>op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
    <td>stream</td>
    <td>Input</td>
    <td>Stream used to execute tasks.</td>
    </tr>
    </tbody></table>

- **Return Value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnGroupedMatMulAlltoAllv defaults to a deterministic implementation.
- Communication engine constraints:

  <!-- npu="A3" id7 -->
  - Atlas A3 products: Support AI_CPU communication.
  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - Ascend 950PR&950DT products: Supports AI_CPU communication.

  <!-- end id8 -->

- Variables used in shape in the parameter description:
  - BSK: The number of tokens received by the current card, which is the cumulative sum of the recvCounts parameter. Value range: (0, 52428800].
  - H1: Indicates the hidden size of the routed expert. Value range: (0, 65536).
  - H2: Indicates the hidden size of the shared expert. Value range: (0, 12288].
  - e: Indicates the number of experts on a single card. e ≤ 32, and e * epWorldSize supports a maximum of 256.
  - N1: Indicates the head_num of the routed expert. Value range: (0, 65536).
  - N2: Indicates the head_num of the shared expert. Value range: (0, 65536).
  - BS: batch sequence size.
  - K: indicates the number of TopK experts selected. K ranges from [2, 8].
  - A: the number of tokens sent by the current card, which is the cumulative sum of the sendCounts parameter.
  - The sum of the A parameter across all devices in the ep communicator equals the sum of the BSK parameter across all devices.

<!-- npu="A3" id9 -->
- <term>Atlas A3 products</term>: Performance may degrade when the communication volume per device is below 2MB.

<!-- end id9 -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

NOTE: The sample code calls some HCCL collective communication library APIs: HcclGetCommName, HcclCommInitAll, and HcclCommDestroy. For details, see [<<HCCL API (C)>>](https://hiascend.com/en/document/redirect/CannCommunityHcclCppApi).

<!-- npu="950,A3" id10 -->
- <term>Atlas A3 products</term> and <term>Ascend 950PR&950DT products</term>:

    ```Cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <vector>
    #include "acl/acl.h"
    #include "hccl/hccl.h"
    #include "aclnnop/aclnn_grouped_mat_mul_allto_allv.h"

    #define CHECK_RET(cond, return_expr) \
        do {                             \
            if (!(cond)) {               \
                return_expr;             \
            }                            \
        } while (0)

    #define LOG_PRINT(message, ...)          \
        do {                                 \
            printf(message, ##__VA_ARGS__);  \
        } while (0)

    int64_t GetShapeSize(const std::vector<int64_t> &shape)
    {
        int64_t shape_size = 1;
        for (auto i : shape) {
            shape_size *= i;
        }
        return shape_size;
    }

    template <typename T>
    int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
        aclDataType dataType, aclTensor **tensor)
    {
        auto size = GetShapeSize(shape) * sizeof(T);
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret);
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = shape.size() - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
        *tensor = aclCreateTensor(shape.data(),
            shape.size(),
            dataType,
            strides.data(),
            0,
            aclFormat::ACL_FORMAT_ND,
            shape.data(),
            shape.size(),
            *deviceAddr);
        return 0;
    }

    struct Args {
        int rankId;
        HcclComm hcclComm;
        aclrtStream stream;
        aclrtContext context;
    };

    // Basic shape information
    constexpr int64_t EP_WORLD_SIZE = 8;
    constexpr int64_t BS = 4096;
    constexpr int64_t K = 2;
    constexpr int64_t H = 7168;
    constexpr int64_t e = 4;
    constexpr int64_t N1 = 4096;
    constexpr int64_t N2 = 4096;
    constexpr int64_t A = BS * K;

    std::vector<int16_t> pGmmYData(BS *K *N1, 0);
    std::vector<int16_t> pmmYData(BS *N2, 0);

    int LaunchOneThreadAlltoAllvGmm(Args &args)
    {
        int ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret: %d\n", ret); return ret);
        char hcomName[128] = {0};
        ret = HcclGetCommName(args.hcclComm, hcomName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret: %d\n", ret); return -1);

        std::vector<int64_t> gmmXShape = {A, H};
        std::vector<int64_t> gmmWShape = {e, H, N1};
        std::vector<int64_t> gmmYShape = {BS * K, N1};
        std::vector<int64_t> permuteShape = {A, H};
        std::vector<int64_t> mmXShape = {BS, H};
        std::vector<int64_t> mmWShape = {H, N2};
        std::vector<int64_t> mmYShape = {BS, N2};

        std::vector<int64_t> sendCountsShape = {EP_WORLD_SIZE * e};
        std::vector<int64_t> recvCountsShape = {EP_WORLD_SIZE * e};

        std::vector<int64_t> sendCountsList(EP_WORLD_SIZE * e, A / (EP_WORLD_SIZE * e));
        std::vector<int64_t> recvCountsList(EP_WORLD_SIZE * e, A / (EP_WORLD_SIZE * e));

        void *gmmXDeviceAddr = nullptr;
        void *gmmWDeviceAddr = nullptr;
        void *gmmYDeviceAddr = nullptr;
        void *permuteDeviceAddr = nullptr;
        void *mmXDeviceAddr = nullptr;
        void *mmWDeviceAddr = nullptr;
        void *mmYDeviceAddr = nullptr;

        aclTensor *gmmX = nullptr;
        aclTensor *gmmW = nullptr;
        aclTensor *gmmY = nullptr;

        aclTensor *mmX = nullptr;
        aclTensor *mmW = nullptr;
        aclTensor *mmY = nullptr;

        aclTensor *sendCountsTensor = nullptr;
        aclTensor *recvCountsTensor = nullptr;

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        void *workspaceAddr = nullptr;

        long long gmmXShapeSize = GetShapeSize(gmmXShape);
        long long gmmWShapeSize = GetShapeSize(gmmWShape);
        long long gmmYShapeSize = GetShapeSize(gmmYShape);

        long long mmXShapeSize = GetShapeSize(mmXShape);
        long long mmWShapeSize = GetShapeSize(mmWShape);
        long long mmYShapeSize = GetShapeSize(mmYShape);

        std::vector<uint16_t> gmmXHostData(gmmXShapeSize, (args.rankId + 1) * 1024);  // BF16, FP16
        std::vector<uint16_t> gmmWHostData(gmmWShapeSize, (args.rankId + 1) * 512);
        std::vector<uint16_t> gmmYHostData(gmmYShapeSize, 65535);

        std::vector<uint16_t> mmXHostData(mmXShapeSize, (args.rankId + 1) * 1024);  // BF16, FP16
        std::vector<uint16_t> mmWHostData(mmWShapeSize, (args.rankId + 1) * 512);
        std::vector<uint16_t> mmYHostData(mmYShapeSize, 0);

        ret = CreateAclTensor(gmmXHostData, gmmXShape, &gmmXDeviceAddr, aclDataType::ACL_FLOAT16, &gmmX);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gmmWHostData, gmmWShape, &gmmWDeviceAddr, aclDataType::ACL_FLOAT16, &gmmW);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gmmYHostData, gmmYShape, &gmmYDeviceAddr, aclDataType::ACL_FLOAT16, &gmmY);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(mmXHostData, mmXShape, &mmXDeviceAddr, aclDataType::ACL_FLOAT16, &mmX);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(mmWHostData, mmWShape, &mmWDeviceAddr, aclDataType::ACL_FLOAT16, &mmW);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(mmYHostData, mmYShape, &mmYDeviceAddr, aclDataType::ACL_FLOAT16, &mmY);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
        aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());

        // Call the first-phase API.
        ret = aclnnGroupedMatMulAlltoAllvGetWorkspaceSize(gmmX,
            gmmW,
            sendCountsTensor,
            recvCountsTensor,
            mmX,
            mmW,
            hcomName,
            EP_WORLD_SIZE,
            sendCounts,
            recvCounts,
            false,
            false,
            gmmY,
            mmY,
            &workspaceSize,
            &executor);
        CHECK_RET(
            ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnGroupedMatMulAlltoAllvGetWorkspaceSize failed. ret = %d \n", ret);
            return ret);

        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }
        // Call the second-phase API.
        ret = aclnnGroupedMatMulAlltoAllv(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnGroupedMatMulAlltoAllv failed. ret = %d \n", ret);
                return ret);
        //(boilerplate) Synchronize and wait for task execution to complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
                return ret);
        // Release device resources. Modify them based on the specific API definition.
        if (args.rankId == 0) {
            size_t size = A * N1 * sizeof(int16_t);
            aclrtMemcpy(pGmmYData.data(), size, gmmYDeviceAddr, size, ACL_MEMCPY_DEVICE_TO_HOST);
        }
        if (gmmX != nullptr) {
            aclDestroyTensor(gmmX);
        }
        if (gmmW != nullptr) {
            aclDestroyTensor(gmmW);
        }
        if (gmmY != nullptr) {
            aclDestroyTensor(gmmY);
        }
        if (mmX != nullptr) {
            aclDestroyTensor(mmX);
        }
        if (mmW != nullptr) {
            aclDestroyTensor(mmW);
        }
        if (mmY != nullptr) {
            aclDestroyTensor(mmY);
        }
        if (gmmXDeviceAddr != nullptr) {
            aclrtFree(gmmXDeviceAddr);
        }
        if (gmmWDeviceAddr != nullptr) {
            aclrtFree(gmmWDeviceAddr);
        }
        if (gmmYDeviceAddr != nullptr) {
            aclrtFree(gmmYDeviceAddr);
        }
        if (mmXDeviceAddr != nullptr) {
            aclrtFree(mmXDeviceAddr);
        }
        if (mmWDeviceAddr != nullptr) {
            aclrtFree(mmWDeviceAddr);
        }
        if (mmYDeviceAddr != nullptr) {
            aclrtFree(mmYDeviceAddr);
        }
        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }
        HcclCommDestroy(args.hcclComm);
        aclrtDestroyStream(args.stream);
        aclrtDestroyContext(args.context);
        aclrtResetDevice(args.rankId);
        return 0;
    }

    int main(int argc, char *argv[])
    {
        int ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d \n", ret); return ret);
        aclrtStream stream[EP_WORLD_SIZE];
        aclrtContext context[EP_WORLD_SIZE];
        for (uint32_t rankId = 0; rankId < EP_WORLD_SIZE; rankId++) {
            ret = aclrtSetDevice(rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);
            ret = aclrtCreateContext(&context[rankId], rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d \n", ret); return ret);
            ret = aclrtCreateStream(&stream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d \n", ret); return ret);
        }

        int32_t devices[EP_WORLD_SIZE];
        for (int i = 0; i < EP_WORLD_SIZE; i++) {
            devices[i] = i;
        }
        //Initialize the communicator.
        HcclComm comms[EP_WORLD_SIZE];
        ret = HcclCommInitAll(EP_WORLD_SIZE, devices, comms);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll failed. ret = %d \n", ret); return ret);

        Args args[EP_WORLD_SIZE];
        // Start multiple threads.
        std::vector<std::unique_ptr<std::thread>> threads(EP_WORLD_SIZE);
        for (uint32_t rankId = 0; rankId < EP_WORLD_SIZE; rankId++) {
            args[rankId].rankId = rankId;
            args[rankId].hcclComm = comms[rankId];
            args[rankId].stream = stream[rankId];
            args[rankId].context = context[rankId];
            threads[rankId].reset(new std::thread(&LaunchOneThreadAlltoAllvGmm, std::ref(args[rankId])));
        }
        for (uint32_t rankId = 0; rankId < EP_WORLD_SIZE; rankId++) {
            threads[rankId]->join();
        }
        aclFinalize();
        return 0;
    }
    ```

<!-- end id10 -->
