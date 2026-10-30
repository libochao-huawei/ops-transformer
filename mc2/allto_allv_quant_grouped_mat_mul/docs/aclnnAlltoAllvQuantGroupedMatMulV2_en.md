# aclnnAlltoAllvQuantGroupedMatMulV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T13:06:24.580Z -->

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

- **Description**: Completes routed-expert AlltoAllv and quantized GroupedMatMul fusion, and implements parallel fusion with quantized MatMul of the shared expert, **communication before computation**.

- **Formula**:
Assume that the total number of devices in the communicator is epWorldSize and the number of routed experts on each device after communication is e. The grouped matrix multiplication on each device computes only the experts on that device. The formula for each device is as follows:
  - Grouped matrix multiplication computation for the shared expert on the local device

    $$
    mm\_y=(mm\_x × mm\_x\_scale) @ (mm\_weight × mm\_weight\_scale)
    $$

  - Alltoallv communication and permute

    $$
    permute\_out=Alltoallv(gmm\_x)
    $$

  - Grouped matrix multiplication computation for local routed experts by expert dimension

    $$
    gmm\_y=(permute\_out × gmm\_x\_scale) @ (gmm\_weight × gmm\_weight\_scale)
    $$

Compared with the `aclnnAlltoAllvQuantGroupedMatMul` API, this API has the following changes:

- A new `commMode` parameter is added for users to specify the communication engine used by the chip.

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>: supports `ai_cpu` and `ccu`.

  <!-- end id7 -->

## Function Prototype

Each operator is a two-phase API. Call the `aclnnAlltoAllvQuantGroupedMatMulV2GetWorkspaceSize` API first to obtain the input parameters and calculate the required workspace size based on the computation flow, and then call the `aclnnAlltoAllvQuantGroupedMatMulV2` API to execute the computation.

```cpp
aclnnStatus aclnnAlltoAllvQuantGroupedMatMulV2GetWorkspaceSize(
    const aclTensor*   gmmX,
    const aclTensor*   gmmWeight,
    const aclTensor*   gmmXScale,
    const aclTensor*   gmmWeightScale,
    const aclTensor*   sendCountsTensorOptional,
    const aclTensor*   recvCountsTensorOptional,
    const aclTensor*   mmXOptional,
    const aclTensor*   mmWeightOptional,
    const aclTensor*   mmXScaleOptional,
    const aclTensor*   mmWeightScaleOptional,
    int64_t            gmmXQuantMode,
    int64_t            gmmWeightQuantMode,
    int64_t            mmXQuantMode,
    int64_t            mmWeightQuantMode,
    const char*        group,
    const char*        commMode,
    int64_t            epWorldSize,
    const aclIntArray* sendCounts,
    const aclIntArray* recvCounts,
    bool               transGmmWeight,
    bool               transMmWeight,
    int64_t            groupSize,
    bool               permuteOutFlag,
    aclTensor*         gmmY,
    aclTensor*         mmYOptional,
    aclTensor*         permuteOutOptional,
    uint64_t*          workspaceSize,
    aclOpExecutor**    executor)
```

```cpp
aclnnStatus aclnnAlltoAllvQuantGroupedMatMulV2(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
```

## aclnnAlltoAllvQuantGroupedMatMulV2GetWorkspaceSize

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
        <td>gmmX</td>
        <td>Input</td>
        <td>This input serves as the left matrix of the GroupedMatMul computation after AlltoAllv communication and Permute operations.</td>
        <td>The current version supports only 2D inputs with shape (BSK, H1), and only the non-transpose use case is supported.</td>
        <td>HIFLOAT8, FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1</td>
        <td>ND</td>
        <td>2</td>
        <td>x</td>
    </tr>
    <tr>
        <td>gmmWeight</td>
        <td>Input</td>
        <td>Right matrix of the GroupedMatMul computation.</td>
        <td>The current version supports only 3D input. The shape is (e, H1, N1) without transpose and (e, N1, H1) with transpose.</td>
        <td>HIFLOAT8, FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1</td>
        <td>ND</td>
        <td>3</td>
        <td>√ (only applicable to the transposed use case)</td>
    </tr>
    <tr>
        <td>gmmXScale</td>
        <td>Input</td>
        <td>Quantization coefficient of gmmX.</td>
        <td><ul><li>The pertensor quantization scenario supports 1D input with shape (1).
            </li><li>The mx quantization scenario supports 3D input with shape (BSK, ceildiv(H1, 64), 2).
            </li></ul></td>
        <td>FLOAT32, FLOAT8_E8M0</td>
        <td>ND</td>
        <td>1, 3</td>
        <td>x</td>
    </tr>
    <tr>
        <td>gmmWeightScale</td>
        <td>Input</td>
        <td>Quantization coefficients of gmmWeight.</td>
        <td><ul><li>In the pertensor quantization scenario, 1 dimension is supported and the shape is (1).
            </li><li>In the mx quantization scenario, 4 dimensions are supported. The non-transpose shape is (e, ceildiv(H1, 64), N1, 2), and the transpose shape is (e, N1, ceildiv(H1, 64), 2).
            </li></ul></td>
        <td>FLOAT32, FLOAT8_E8M0</td>
        <td>ND</td>
        <td>1, 4</td>
        <td>√ (applicable only to the transpose scenario)</td>
    </tr>
    <tr>
        <td>sendCountsTensorOptional</td>
        <td>Input</td>
        <td>Reserved parameter. Only nullptr can be passed in the current version.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>recvCountsTensorOptional</td>
        <td>Input</td>
        <td>Reserved parameter. Only nullptr is supported in the current version.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>mmXOptional</td>
        <td>Input</td>
        <td>Optional input. Left matrix in the shared expert MatMul computation.</td>
        <td><ul><li>Supports 2 dimensions, with shape (BS, H2).
            </li><li>Must be passed together with mmWeightOptional or both set to nullptr.
            </li></ul></td>
        <td>Same as gmmX</td>
        <td>ND</td>
        <td>2</td>
        <td>x</td>
    </tr>
    <tr>
        <td>mmWeightOptional</td>
        <td>Input</td>
        <td>Optional input, the right matrix in the shared expert MatMul computation.</td>
        <td><ul><li>Supports 2 dimensions, with shape (H2, N2) for non-transpose and shape (N2, H2) for transpose.
            </li><li>Must be passed together with mmXOptional or both set to nullptr.
            </li></ul></td>
        <td>Same as gmmWeight</td>
        <td>ND</td>
        <td>2</td>
        <td>√ (applicable only to the transpose scenario).</td>
    </tr>
    <tr>
        <td>mmXScaleOptional</td>
        <td>Input</td>
        <td>Optional input, quantization coefficients of mmX.</td>
        <td><ul><li>Supports 1 dimension in the pertensor quantization scenario, with the shape of (1).
            </li><li>Supports 3 dimensions in the mx quantization scenario, with the shape of (BS, ceildiv(H2, 64), 2).
            </li><li>Must be passed in together with mmXOptional, or both set to nullptr.
            </li></ul></td>
        <td>FLOAT32, FLOAT8_E8M0</td>
        <td>ND</td>
        <td>1, 3</td>
        <td>x</td>
    </tr>
    <tr>
        <td>mmWeightScaleOptional</td>
        <td>Input</td>
        <td>Optional input. Quantization scale of mmWeight.</td>
        <td><ul><li>In pertensor quantization scenario, 1 dimension is supported, with shape (1).
            </li><li>In mx quantization scenario, 3 dimensions are supported, with shape (ceildiv(H2, 64), N2, 2) for non-transpose and shape (N2, ceildiv(H2, 64), 2) for transpose.
            </li><li>Must be passed in together with mmXOptional, or both must be nullptr.
            </li></ul></td>
        <td>FLOAT32, FLOAT8_E8M0</td>
        <td>ND</td>
        <td>1, 3</td>
        <td>√ (transpose use case only)</td>
    </tr>
    <tr>
        <td>gmmXQuantMode</td>
        <td>Input</td>
        <td>Quantization mode of gmmX.</td>
        <td>The current version supports 1 (pertensor quantization scenario) and 6 (mx quantization scenario).</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>gmmWeightQuantMode</td>
        <td>Input</td>
        <td>Quantization mode of gmmWeight.</td>
        <td>The current version supports 1 (pertensor quantization scenario) and 6 (mx quantization scenario).</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>mmXQuantMode</td>
        <td>Input</td>
        <td>Quantization mode of mmX.</td>
        <td>The current version supports 1 (pertensor quantization scenario) and 6 (mx quantization scenario).</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>mmWeightQuantMode</td>
        <td>Input</td>
        <td>Quantization mode of mmWeight.</td>
        <td>The current version supports 1 (pertensor quantization scenario) and 6 (mx quantization scenario).</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>group</td>
        <td>Input</td>
        <td>Communication domain name for expert parallelism. The string length must be (0, 128).</td>
        <td>Obtained through the Hccl API "extern HcclResult HcclGetCommName(HcclComm comm, char* commName);", where commName is group.</td>
        <td>STRING</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commMode</td>
        <td>Input</td>
        <td>Specifies the current communication type.</td>
        <td>Supports "ai_cpu" and "ccu".</td>
        <td>STRING</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>epWorldSize</td>
        <td>Input</td>
        <td>Size of the ep communicator.</td>
        <td><term>Ascend 950PR&950DT products</term> supports 2, 4, 8, 16, 32, 64, 128, 256.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>sendCounts</td>
        <td>Input</td>
        <td>Indicates the number of tokens sent to other devices.</td>
        <td>The data type is INT64. The length is e * epWorldSize, with a maximum of 256. The input type must be a list.</td>
        <td>aclIntArray* (element type INT64)</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>recvCounts</td>
        <td>Input</td>
        <td>Indicates the number of tokens received from other devices.</td>
        <td>INT64 data type is supported. The length is e * epWorldSize, with a maximum of 256. The input type must be list.</td>
        <td>aclIntArray* (element type INT64)</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>transGmmWeight</td>
        <td>Input</td>
        <td>Whether the right matrix of GroupedMatMul needs to be transposed.</td>
        <td>true indicates that transpose is required, and false indicates no transpose.</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>transMmWeight</td>
        <td>Input</td>
        <td>Whether the right matrix of the shared expert MatMul needs to be transposed.</td>
        <td>true indicates that transposition is required; false indicates no transposition.</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>groupSize</td>
        <td>Input</td>
        <td>Indicates how many numbers of the gmmX/gmmWeight/mmX/mmWeight input in the corresponding dimension can be quantized using one number of the gmmXScale/gmmWeightScale/mmXScale/mmWeightScale input in that dimension.</td>
        <td>The groupSize input consists of three values, groupSizeM, groupSizeN, and groupSizeK, in three directions. Each value occupies 16 bits. The calculation formula is groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>permuteOutFlag</td>
        <td>Input</td>
        <td>Specifies whether permuteOutOptional needs to be output.</td>
        <td>true indicates that output is required; false indicates that output is not required.</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>gmmY</td>
        <td>Output</td>
        <td>Output computed by the routed expert.</td>
        <td>Supports 2 dimensions, with shape (A, N1).</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>x</td>
    </tr>
    <tr>
        <td>mmYOptional</td>
        <td>Output</td>
        <td>Output of the shared expert computation.</td>
        <td>Supports 2 dimensions, with shape (BS, N2). Output is generated only when mmXOptional and mmWeightOptional are passed in.</td>
        <td>Same as gmmY</td>
        <td>ND</td>
        <td>2</td>
        <td>x</td>
    </tr>
    <tr>
        <td>permuteOutOptional</td>
        <td>Output</td>
        <td>Output after permute.</td>
        <td>Supports 2 dimensions, shape (A, H1). Output only when permuteOutFlag is true.</td>
        <td>Same as gmmX</td>
        <td>ND</td>
        <td>2</td>
        <td>x</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the workspace size to be allocated on the Device.</td>
        <td>-</td>
        <td>UINT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Returns the op executor, which contains the computation flow of the operator.</td>
        <td>-</td>
        <td>aclOpExecutor*</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    </tbody></table>

- The relationship between the enumeration values of gmmXQuantMode, gmmWeightQuantMode, mmXQuantMode, and mmWeightQuantMode and the quantization mode is as follows:
  - 0: non-quantization
  - 1: pertensor
  - 2: perchannel
  - 3: pertoken
  - 4: pergroup
  - 5: perblock
  - 6: mx quantization
  - 7: pertoken dynamic quantization

- **Return Value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md). The first-phase API validates input parameters. Errors occur in the following cases:

    <table style="undefined;table-layout: fixed; width: 1180px"> <colgroup>
    <col style="width: 250px">
    <col style="width: 130px">
    <col style="width: 800px">
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
    <td>The parameter is a required input, output, or attribute, but a null pointer is passed in.</td>
    </tr>
    <tr>
    <td>ACLNN_ERR_PARAM_INVALID</td>
    <td>161002</td>
    <td>The data type, data format, or dimension of gmmX, gmmWeight, sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional, group, epWorldSize, sendCounts, or recvCounts is not in the supported range.</td>
    </tr>
    </tbody></table>

## aclnnAlltoAllvQuantGroupedMatMulV2

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1180px"> <colgroup>
    <col style="width: 250px">
    <col style="width: 130px">
    <col style="width: 800px">
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
    <td>Workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>Input</td>
    <td>Size of the workspace allocated on the Device, obtained by the first-phase API <code>aclnnAlltoAllvQuantGroupedMatMulV2GetWorkspaceSize</code>.</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>Input</td>
    <td>Op executor, containing the operator computation flow.</td>
    </tr>
    <tr>
    <td>stream</td>
    <td>Input</td>
    <td>Specifies the Stream for task execution.</td>
    </tr>
    </tbody></table>

- **Return Value**

    Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Communication engine constraints: Supports CCU communication and AI_CPU communication. CCU supports intra-machine UB domain interconnection only, while AI_CPU supports inter-machine UB domain interconnection.

- Deterministic computation:
  - `aclnnAlltoAllvQuantGroupedMatMulV2` defaults to a deterministic implementation.

- Variables used in shape in the parameter description:
  - BSK: the number of tokens sent by the current card, which is the cumulative sum of the sendCounts parameter. Value range: (0, 52428800).
  - H1: Indicates the hidden size of the routed expert. Value range: (0, 65536).
  - H2: Indicates the hidden size of the shared expert. Value range: (0, 12288].
  - e: number of experts on a single card, value range (0, 32], and the maximum supported value of e * epWorldSize is 256.
  - N1: Indicates the head_num of the routed expert. Value range: (0, 65536).
  - N2: Indicates the head_num of the shared expert. Value range: (0, 65536).
  - BS: batch sequence size.
  - K: indicates the number of TopK experts selected. K ranges from [2, 8].
  - A: the number of tokens received by the current card, which is the cumulative sum of the recvCounts parameter.
  - The sum of parameter A across all devices in the ep communicator equals the sum of parameter BSK across all devices.
  - For mx quantization when gmmX and gmmWeight are FLOAT4_E2M1, H1 and H2 must be even and cannot be 2. When both transGmmWeight and transMmWeight are false, N1 and N2 must be even.
  - The transpose states of gmmWeight and gmmWeightScale must be consistent: both transposed or both not transposed. The transpose states of mmWeight and mmWeightScale must also be consistent.
  - groupSize:
    - The groupSize value is valid only when the gmmXScale/gmmWeightScale/mmXScale/mmWeightScale inputs are 2D or higher. Pass 0 in other use cases.
    - The groupSize value supports formula derivation: the input groupSize is internally decomposed into groupSizeM, groupSizeN, and groupSizeK according to the following formula. If one or more of them is 0, groupSizeM, groupSizeN, and groupSizeK are reset for computation based on the input shapes of gmmX/gmmWeight/mmX/mmWeight/gmmXScale/gmmWeightScale/mmXScale/mmWeightScale. Reset rules: If groupSizeM=0, the group value of directional quantization in the m dimension is determined by interface derivation. The derivation formula is groupSizeM = m / scaleM (m must be divisible by scaleM), where m is the same as m in the gmmX/mmX shape, and scaleM is the same as m in the gmmXScale/mmXScale shape; if groupSizeK=0, the group value of directional quantization in the k dimension is determined by interface derivation. The derivation formula is groupSizeK = k / scaleK (k must be divisible by scaleK), where k is the same as k in the gmmX/mmX shape, and scaleK is the same as k in the gmmXScale/mmXScale shape; if groupSizeN=0, the group value of directional quantization in the n dimension is determined by interface derivation. The derivation formula is groupSizeN = n / scaleN (n must be divisible by scaleN), where n is the same as n in the gmmWeight/mmWeight shape, and scaleN is the same as n in the gmmWeightScale/mmWeightScale shape.
    $$
    groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
    $$
    - If the reset condition is met and the gmmXScale/gmmWeightScale/mmXScale/mmWeightScale inputs are 2D or higher with the data type FLOAT8_E8M0 for all, the [groupSizeM, groupSizeN, groupSizeK] combination is derived as [1, 1, 32], corresponding to a groupSize value of 4295032864.
- Quantization parameter constraints:
  - The current version supports pertensor quantization and mx quantization.
- Type constraints
  - pertensor quantization

    | gmmX | gmmWeight | gmmXScale | gmmWeightScale | mmXScale | mmWeightScale | gmmY |
    | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
    | HIFLOAT8 | HIFLOAT8 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT16/BFLOAT16 |

  - mx quantization

    | gmmX | gmmWeight | gmmXScale | gmmWeightScale | mmXScale | mmWeightScale | gmmY |
    | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
    | FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT16/BFLOAT16 |
    | FLOAT8_E4M3FN | FLOAT8_E5M2 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT16/BFLOAT16 |
    | FLOAT8_E5M2 | FLOAT8_E4M3FN | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT16/BFLOAT16 |
    | FLOAT8_E5M2 | FLOAT8_E5M2 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT16/BFLOAT16 |
    | FLOAT4_E2M1 | FLOAT4_E2M1 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT16/BFLOAT16 |

  - mmX has the same data type as gmmX, mmWeight has the same data type as gmmWeight, mmY has the same data type as gmmY, and permuteOut has the same data type as gmmX.

## Example

The following is sample code for reference only. For details about compilation and execution, see Compile and Run Samples.

Note: The quantization API supports only <term>Ascend 950PR&950DT products</term>. The following example is implemented based on this series. The sample code uses 2 devices as an example. Modify `EP_WORLD_SIZE` according to the actual number of devices in the environment.

```cpp
#include <thread>
#include <iostream>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "aclnnop/aclnn_allto_allv_quant_grouped_mat_mul_v2.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)           \
    do {                                  \
        printf(message, ##__VA_ARGS__);   \
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
constexpr int64_t EP_WORLD_SIZE = 2;
constexpr int64_t BS = 4096;
constexpr int64_t K = 2;
constexpr int64_t H = 7168;
constexpr int64_t e = 4;
constexpr int64_t N1 = 4096;
constexpr int64_t N2 = 4096;
constexpr int64_t A = BS * K;

int LaunchOneThreadAlltoAllvQuantGroupedMatMul(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret: %d\n", ret); return ret);
    char hcomName[128] = {0};
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret: %d\n", ret); return -1);

    std::vector<int64_t> gmmXShape = {BS * K, H};
    std::vector<int64_t> gmmWShape = {e, H, N1};
    std::vector<int64_t> gmmYShape = {A, N1};
    std::vector<int64_t> permuteShape = {A, H};
    std::vector<int64_t> mmXShape = {BS, H};
    std::vector<int64_t> mmWShape = {H, N2};
    std::vector<int64_t> mmYShape = {BS, N2};
    std::vector<int64_t> scaleShape = {1}; // Scale factor shape

    std::vector<int64_t> sendCountsList(EP_WORLD_SIZE * e, BS * K / (EP_WORLD_SIZE * e));
    std::vector<int64_t> recvCountsList(EP_WORLD_SIZE * e, BS * K / (EP_WORLD_SIZE * e));

    void *gmmXDeviceAddr = nullptr;
    void *gmmWDeviceAddr = nullptr;
    void *gmmYDeviceAddr = nullptr;
    void *permuteDeviceAddr = nullptr;
    void *mmXDeviceAddr = nullptr;
    void *mmWDeviceAddr = nullptr;
    void *mmYDeviceAddr = nullptr;
    void *gmmXScaleDeviceAddr = nullptr;
    void *gmmWScaleDeviceAddr = nullptr;
    void *mmXScaleDeviceAddr = nullptr;
    void *mmWScaleDeviceAddr = nullptr;

    aclTensor *gmmX = nullptr;
    aclTensor *gmmW = nullptr;
    aclTensor *gmmY = nullptr;
    aclTensor *mmX = nullptr;
    aclTensor *mmW = nullptr;
    aclTensor *mmY = nullptr;
    aclTensor *permute = nullptr;
    aclTensor *gmmXScale = nullptr;
    aclTensor *gmmWScale = nullptr;
    aclTensor *mmXScale = nullptr;
    aclTensor *mmWScale = nullptr;
    int64_t groupSize = 0;

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;

    long long gmmXShapeSize = GetShapeSize(gmmXShape);
    long long gmmWShapeSize = GetShapeSize(gmmWShape);
    long long gmmYShapeSize = GetShapeSize(gmmYShape);
    long long permuteShapeSize = GetShapeSize(permuteShape);
    long long mmXShapeSize = GetShapeSize(mmXShape);
    long long mmWShapeSize = GetShapeSize(mmWShape);
    long long mmYShapeSize = GetShapeSize(mmYShape);

    // HIFLOAT8 data (simulated using uint8_t)
    std::vector<uint8_t> gmmXHostData(gmmXShapeSize, (args.rankId + 1) * 10);
    std::vector<uint8_t> gmmWHostData(gmmWShapeSize, (args.rankId + 1) * 5);
    std::vector<uint8_t> mmXHostData(mmXShapeSize, (args.rankId + 1) * 10);
    std::vector<uint8_t> mmWHostData(mmWShapeSize, (args.rankId + 1) * 5);

    // Output data (FLOAT16/BFLOAT16)
    std::vector<uint16_t> gmmYHostData(gmmYShapeSize, 65535);
    std::vector<uint16_t> mmYHostData(mmYShapeSize, 0);
    std::vector<uint8_t> permuteHostData(permuteShapeSize, 255);

    // Scale factor data (FLOAT32)
    std::vector<float> gmmXScaleHostData(1, 1.0f);
    std::vector<float> gmmWScaleHostData(1, 1.0f);
    std::vector<float> mmXScaleHostData(1, 1.0f);
    std::vector<float> mmWScaleHostData(1, 1.0f);

    // Create tensors.
    ret = CreateAclTensor(gmmXHostData, gmmXShape, &gmmXDeviceAddr, ACL_HIFLOAT8, &gmmX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gmmWHostData, gmmWShape, &gmmWDeviceAddr, ACL_HIFLOAT8, &gmmW);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gmmYHostData, gmmYShape, &gmmYDeviceAddr, ACL_FLOAT16, &gmmY);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(mmXHostData, mmXShape, &mmXDeviceAddr, ACL_HIFLOAT8, &mmX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(mmWHostData, mmWShape, &mmWDeviceAddr, ACL_HIFLOAT8, &mmW);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(mmYHostData, mmYShape, &mmYDeviceAddr, ACL_FLOAT16, &mmY);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(permuteHostData, permuteShape, &permuteDeviceAddr, ACL_HIFLOAT8, &permute);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gmmXScaleHostData, scaleShape, &gmmXScaleDeviceAddr, ACL_FLOAT, &gmmXScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gmmWScaleHostData, scaleShape, &gmmWScaleDeviceAddr, ACL_FLOAT, &gmmWScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(mmXScaleHostData, scaleShape, &mmXScaleDeviceAddr, ACL_FLOAT, &mmXScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(mmWScaleHostData, scaleShape, &mmWScaleDeviceAddr, ACL_FLOAT, &mmWScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());

    // Call the first-phase API.
    ret = aclnnAlltoAllvQuantGroupedMatMulV2GetWorkspaceSize(gmmX,
        gmmW,
        gmmXScale,
        gmmWScale,
        nullptr, // sendCountsTensorOptional
        nullptr, // recvCountsTensorOptional
        mmX,
        mmW,
        mmXScale,
        mmWScale,
        1, // gmmXQuantMode
        1, // gmmWeightQuantMode
        1, // mmXQuantMode
        1, // mmWeightQuantMode
        hcomName,
        "ai_cpu",
        EP_WORLD_SIZE,
        sendCounts,
        recvCounts,
        false, // transGmmWeight
        false, // transMmWeight
        groupSize,
        true,  // permuteOutFlag
        gmmY,
        mmY,
        permute,
        &workspaceSize,
        &executor);
    CHECK_RET(
        ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnAlltoAllvQuantGroupedMatMulV2GetWorkspaceSize failed. ret = %d \n", ret);
        return ret);

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
    }

    // Call the second-phase API.
    ret = aclnnAlltoAllvQuantGroupedMatMulV2(workspaceAddr, workspaceSize, executor, args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnAlltoAllvQuantGroupedMatMulV2 failed. ret = %d \n", ret);
            return ret);

    // Synchronize and wait for the task to complete.
    ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000000);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
            return ret);

    // Release device resources.
    if (gmmX != nullptr) aclDestroyTensor(gmmX);
    if (gmmW != nullptr) aclDestroyTensor(gmmW);
    if (gmmY != nullptr) aclDestroyTensor(gmmY);
    if (mmX != nullptr) aclDestroyTensor(mmX);
    if (mmW != nullptr) aclDestroyTensor(mmW);
    if (mmY != nullptr) aclDestroyTensor(mmY);
    if (permute != nullptr) aclDestroyTensor(permute);
    if (gmmXScale != nullptr) aclDestroyTensor(gmmXScale);
    if (gmmWScale != nullptr) aclDestroyTensor(gmmWScale);
    if (mmXScale != nullptr) aclDestroyTensor(mmXScale);
    if (mmWScale != nullptr) aclDestroyTensor(mmWScale);

    if (gmmXDeviceAddr != nullptr) aclrtFree(gmmXDeviceAddr);
    if (gmmWDeviceAddr != nullptr) aclrtFree(gmmWDeviceAddr);
    if (gmmYDeviceAddr != nullptr) aclrtFree(gmmYDeviceAddr);
    if (mmXDeviceAddr != nullptr) aclrtFree(mmXDeviceAddr);
    if (mmWDeviceAddr != nullptr) aclrtFree(mmWDeviceAddr);
    if (mmYDeviceAddr != nullptr) aclrtFree(mmYDeviceAddr);
    if (permuteDeviceAddr != nullptr) aclrtFree(permuteDeviceAddr);
    if (gmmXScaleDeviceAddr != nullptr) aclrtFree(gmmXScaleDeviceAddr);
    if (gmmWScaleDeviceAddr != nullptr) aclrtFree(gmmWScaleDeviceAddr);
    if (mmXScaleDeviceAddr != nullptr) aclrtFree(mmXScaleDeviceAddr);
    if (mmWScaleDeviceAddr != nullptr) aclrtFree(mmWScaleDeviceAddr);
    if (workspaceSize > 0) aclrtFree(workspaceAddr);

    HcclCommDestroy(args.hcclComm);
    aclrtDestroyStream(args.stream);
    aclrtDestroyContext(args.context);
    aclrtResetDevice(args.rankId);
    return 0;
}

int main(int argc, char *argv[])
{
    // This sample is implemented based on Ascend 950PR&950DT products.
    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d \n", ret); return ret);

    constexpr uint32_t WORLD_SIZE = EP_WORLD_SIZE;
    aclrtStream stream[WORLD_SIZE];
    aclrtContext context[WORLD_SIZE];

    for (uint32_t rankId = 0; rankId < WORLD_SIZE; rankId++) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);
        ret = aclrtCreateContext(&context[rankId], rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d \n", ret); return ret);
        ret = aclrtCreateStream(&stream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d \n", ret); return ret);
    }

    int32_t devices[WORLD_SIZE];
    for (int i = 0; i < WORLD_SIZE; i++) {
        devices[i] = i;
    }

    // Initialize the communicator.
    HcclComm comms[WORLD_SIZE];
    ret = HcclCommInitAll(WORLD_SIZE, devices, comms);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll failed. ret = %d \n", ret); return ret);

    Args args[WORLD_SIZE];
    // Start multiple threads.
    std::vector<std::unique_ptr<std::thread>> threads(WORLD_SIZE);
    for (uint32_t rankId = 0; rankId < WORLD_SIZE; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].hcclComm = comms[rankId];
        args[rankId].stream = stream[rankId];
        args[rankId].context = context[rankId];
        threads[rankId].reset(new std::thread(&LaunchOneThreadAlltoAllvQuantGroupedMatMul, std::ref(args[rankId])));
    }

    for (uint32_t rankId = 0; rankId < WORLD_SIZE; rankId++) {
        threads[rankId]->join();
    }

    aclFinalize();
    return 0;
}

```
