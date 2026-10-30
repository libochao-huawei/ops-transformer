# aclnnQuantGroupedMatMulAlltoAllv

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T14:42:18.527Z -->

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

- Description:
    - Completes the fusion of quantized routed expert GroupedMatMul, Unpermute, and AlltoAllv, and achieves parallel fusion with shared expert MatMul. Computes first, then communicates. Supports pertensor-pertensor and mx [quantization mode](../../../docs/en/context/quant_mode_introduction.md).

- Formula:
    - Routed experts:
        $$
        gmmY = (gmmX @ gmmWeight) * gmmXScale * gmmWeightScale \\
        unpermuteOut = Unpermute(gmmY) \\
        y = AlltoAllv(unpermuteOut)
        $$

    - Shared expert:

        $$
        mmY = (mmX @ mmWeight) * mmXScaleOptional * mmWeightScaleOptional
        $$

## Function Prototype

This operator consists of a two-phase API. Call the `aclnnQuantGroupedMatMulAlltoAllvGetWorkspaceSize` API first to obtain the input parameters, calculate the required workspace size based on the computation process, and obtain the executor that contains the operator computation process. Then, call the `aclnnQuantGroupedMatMulAlltoAllv` API to perform the computation.

```cpp
aclnnStatus aclnnQuantGroupedMatMulAlltoAllvGetWorkspaceSize(
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
    const aclTensor*   commQuantScaleOptional,
    int64_t            gmmXQuantMode,
    int64_t            gmmWeightQuantMode,
    int64_t            mmXQuantMode,
    int64_t            mmWeightQuantMode,
    int64_t            commQuantMode,
    int64_t            commQuantDtypeOptional,
    int64_t            groupSize,
    const char*        group,
    int64_t            epWorldSize,
    const aclIntArray* sendCounts,
    const aclIntArray* recvCounts,
    bool               transGmmWeight,
    bool               transMmWeight,
    const aclTensor*   y,
    const aclTensor*   mmYOptional,
    uint64_t*          workspaceSize,
    aclOpExecutor**    executor)
```

```cpp
aclnnStatus aclnnQuantGroupedMatMulAlltoAllv(
    void*           workspace,
    uint64_t        workspaceSize,
    aclOpExecutor*  executor,
    aclrtStream     stream)
```

## aclnnQuantGroupedMatMulAlltoAllvGetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1200px">
        <colgroup>
            <col style="width: 188px">
            <col style="width: 85px">
            <col style="width: 220px">
            <col style="width: 280px">
            <col style="width: 200px">
            <col style="width: 90px">
            <col style="width: 100px">
            <col style="width: 80px">
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
            </tr>
        </thead>
        <tbody>
            <!-- gmmX -->
            <tr>
                <td>gmmX</td>
                <td>Input</td>
                <td>Input gmmX in the formula.</td>
                <td>Shape (A, H1).</td>
                <td>HIFLOAT8, FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1</td>
                <td>ND</td>
                <td>2</td>
                <td>x</td>
            </tr>
            <!-- gmmWeight -->
            <tr>
                <td>gmmWeight</td>
                <td>Input</td>
                <td>Input gmmWeight in the formula.</td>
                <td>shape (e, H1, N1). e indicates the number of experts deployed on each device, H1 indicates the hidden size, and N1 indicates the FFN intermediate dimension of routed experts.</td>
                <td>HIFLOAT8, FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1</td>
                <td>ND</td>
                <td>3</td>
                <td>x</td>
            </tr>
            <!-- gmmXScale -->
            <tr>
                <td>gmmXScale</td>
                <td>Input</td>
                <td>Quantization coefficient of gmmX.</td>
                <td><li>pertensor quantization: shape (1).</li><li>mx quantization: shape (A, ceil(H1/64), 2)</li></td>
                <td>FLOAT32, FLOAT8_E8M0</td>
                <td>ND</td>
                <td><li>per-tensor quantization: 1</li><li>mx quantization: 3</li></td>
                <td>x</td>
            </tr>
            <!-- gmmWeightScale -->
            <tr>
                <td>gmmWeightScale</td>
                <td>Input</td>
                <td>Quantization coefficient of gmmWeight.</td>
                <td><li>per-tensor quantization: shape (1).</li><li>mx quantization: shape (e, N1, ceil(H1/64), 2), and shape (e, ceil(H1/64), N1, 2) when weight is transposed.</li></td>
                <td>FLOAT32, FLOAT8_E8M0</td>
                <td>ND</td>
                <td><li>per-tensor quantization: 1</li><li>mx quantization: 3</li></td>
                <td>x</td>
            </tr>
            <!-- sendCountsTensorOptional -->
            <tr>
                <td>sendCountsTensorOptional</td>
                <td>Input</td>
                <td>The send count used by AlltoAllv.</td>
                <td>Currently, only empty is supported. shape (e * ep, ), where e is the number of experts deployed on each device and ep is the size of the ep domain.</td>
                <td>INT64</td>
                <td>ND</td>
                <td>1</td>
                <td>x</td>
            </tr>
            <!-- recvCountsTensorOptional -->
            <tr>
                <td>recvCountsTensorOptional</td>
                <td>Input</td>
                <td>The recv count used by AlltoAllv.</td>
                <td>Currently, only empty is supported. shape (e * ep, ). e indicates the number of experts deployed on each device, and ep indicates the size of the ep domain.</td>
                <td>INT64</td>
                <td>ND</td>
                <td>1</td>
                <td>x</td>
            </tr>
            <!-- mmXOptional -->
            <tr>
                <td>mmXOptional</td>
                <td>Input</td>
                <td>Input mmX in the formula.</td>
                <td>shape (BS, H2). BS indicates the batch sequence size, and H2 indicates the hidden size.</td>
                <td>HIFLOAT8, FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1</td>
                <td>ND</td>
                <td>2</td>
                <td>x</td>
            </tr>
            <!-- mmWeightOptional -->
            <tr>
                <td>mmWeightOptional</td>
                <td>Input</td>
                <td>Input mmWeight in the formula.</td>
                <td>shape (H2, N2). H2 indicates the hidden size, and N2 indicates the intermediate layer dimension of the shared expert FFN.</td>
                <td>HIFLOAT8, FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1</td>
                <td>ND</td>
                <td>2</td>
                <td>x</td>
            </tr>
            <!-- mmXScaleOptional -->
            <tr>
                <td>mmXScaleOptional</td>
                <td>Input</td>
                <td>Quantization coefficient of mmX.</td>
                <td><li>pertensor quantization: shape (1).</li><li>mx quantization: shape (BS, ceil(H2/64), 2)</li></td>
                <td>FLOAT32, FLOAT8_E8M0</td>
                <td>ND</td>
                <td><li>pertensor quantization: 1</li><li>mx quantization: 3</li></td>
                <td>x</td>
            </tr>
            <!-- mmWeightScaleOptional -->
            <tr>
                <td>mmWeightScaleOptional</td>
                <td>Input</td>
                <td>Quantization coefficient of mmWeight.</td>
                <td><li>pertensor quantization: shape(1).</li><li>mx quantization: shape (N2, ceil(H2/64), 2); (ceil(H2/64), N2, 2) when weight is transposed</li></td>
                <td>FLOAT32, FLOAT8_E8M0</td>
                <td>ND</td>
                <td><li>pertensor quantization: 1</li><li>mx quantization: 3</li></td>
                <td>x</td>
            </tr>
            <!-- commQuantScaleOptional -->
            <tr>
                <td>commQuantScaleOptional</td>
                <td>Input</td>
                <td>Quantization coefficient for low-bit communication.</td>
                <td>Reserved parameter. Currently, only null is supported.</td>
                <td>FLOAT32</td>
                <td>ND</td>
                <td>1</td>
                <td>x</td>
            </tr>
            <!-- gmmXQuantMode -->
            <tr>
                <td>gmmXQuantMode</td>
                <td>Input</td>
                <td>Quantization mode of gmmX.</td>
                <td>The quantization mode must be passed in. Currently, 1 (per-tensor quantization) and 6 (MX quantization) are supported.</td>
                <td>INT64</td>
                <td>-</td>
                <td>1</td>
                <td>x</td>
            </tr>
            <!-- gmmWeightQuantMode -->
            <tr>
                <td>gmmWeightQuantMode</td>
                <td>Input</td>
                <td>Quantization mode of gmmWeight.</td>
                <td>The quantization mode must be passed in. Currently, 1 (per-tensor quantization) and 6 (MX quantization) are supported.</td>
                <td>INT64</td>
                <td>-</td>
                <td>1</td>
                <td>x</td>
            </tr>
            <!-- mmXQuantMode -->
            <tr>
                <td>mmXQuantMode</td>
                <td>Input</td>
                <td>Quantization mode of mmX.</td>
                <td>If mmX is not empty, the quantization mode must be passed in. Currently, 1 (pertensor quantization) and 6 (mx quantization) are supported.</td>
                <td>INT64</td>
                <td>-</td>
                <td>1</td>
                <td>x</td>
            </tr>
            <!-- mmWeightQuantMode -->
            <tr>
                <td>mmWeightQuantMode</td>
                <td>Input</td>
                <td>Quantization mode of mmWeight.</td>
                <td>If mmWeight is not empty, the quantization mode must be passed in. Currently, 1 (pertensor quantization) and 6 (mx quantization) are supported.</td>
                <td>INT64</td>
                <td>-</td>
                <td>1</td>
                <td>x</td>
            </tr>
            <!-- commQuantMode -->
            <tr>
                <td>commQuantMode</td>
                <td>Input</td>
                <td>Low-bit communication quantization mode.</td>
                <td>The low-bit feature is reserved. The value must be 0, indicating no quantization.</td>
                <td>INT64</td>
                <td>-</td>
                <td>1</td>
                <td>x</td>
            </tr>
            <!-- commQuantDtypeOptional -->
            <tr>
                <td>commQuantDtypeOptional</td>
                <td>Input</td>
                <td>Data type of low-bit communication.</td>
                <td>Currently reserved for the low-bit function. The value must be -1.</td>
                <td>INT64</td>
                <td>-</td>
                <td>1</td>
                <td>x</td>
            </tr>
            <!-- groupSize (new) -->
            <tr>
                <td>groupSize</td>
                <td>Input</td>
                <td>PerGroup quantization group size.</td>
                <td>Indicates the quantization group sizes in three directions for the Matmul computation. This parameter is reserved. It is valid only when the quantization scale input is 2D or higher; in other use cases, pass 0. The groupSize input consists of three concatenated values: groupSizeM, groupSizeN, and groupSizeK. Each value occupies 16 bits, and the three values together occupy the lower 48 bits of the int64_t type groupSize (the upper 16 bits are invalid). The formula is: groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32.</td>
                <td>INT64</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
            </tr>
            <!-- group -->
            <tr>
                <td>group</td>
                <td>Input</td>
                <td>Communicator identifier.</td>
                <td>String length greater than 0 and less than 128</td>
                <td>char*</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
            </tr>
            <!-- epWorldSize -->
            <tr>
                <td>epWorldSize</td>
                <td>Input</td>
                <td>communicator size.</td>
                <td>Supports 2/4/8/16/32/64/128/256.</td>
                <td>INT64</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
            </tr>
            <!-- sendCounts -->
            <tr>
                <td>sendCounts</td>
                <td>Input</td>
                <td>Send count used by AlltoAllv. Indicates the number of tokens sent by other ranks to each expert on the current rank.</td>
                <td>The supported dimension is e * ep. The data is flattened into one dimension in the order of <code>sendCounts[fromRank][expertId]</code>. For example, when e=3, the order is <code>e0,e1,e2,e0,e1,e2, ...</code></td>
                <td>aclIntArray* (element type INT64)</td>
                <td>ND</td>
                <td>-</td>
                <td>-</td>
            </tr>
            <!-- recvCounts -->
            <tr>
                <td>recvCounts</td>
                <td>Input</td>
                <td>Recv count used by AlltoAllv. Indicates the number of tokens that need to be received by the current device after AlltoAllv.</td>
                <td>The supported dimension is e * ep. Flattened to one dimension according to <code>recvCounts[fromRank][expertId]</code>. For example, when e=3, the order is <code>e0,e1,e2,e0,e1,e2, ...</code>.</td>
                <td>aclIntArray* (element type INT64)</td>
                <td>ND</td>
                <td>-</td>
                <td>-</td>
            </tr>
            <!-- transGmmWeight -->
            <tr>
                <td>transGmmWeight</td>
                <td>Input</td>
                <td>Whether the right matrix of gmm is transposed.</td>
                <td>Must be specified. No default value.</td>
                <td>BOOL</td>
                <td>ND</td>
                <td>-</td>
                <td>-</td>
            </tr>
            <!-- transMmWeight -->
            <tr>
                <td>transMmWeight</td>
                <td>Input</td>
                <td>Whether the right matrix of mm is transposed.</td>
                <td>Required. No default value.</td>
                <td>BOOL</td>
                <td>ND</td>
                <td>-</td>
                <td>-</td>
            </tr>
            <!-- y -->
            <tr>
                <td>y</td>
                <td>Output</td>
                <td>Output of the GroupedMatMul computation.</td>
                <td>Empty tensors are not supported. shape (BSK, N1).</td>
                <td>FLOAT16, BFLOAT16</td>
                <td>ND</td>
                <td>2</td>
                <td>x</td>
            </tr>
            <!-- mmYOptional -->
            <tr>
                <td>mmYOptional</td>
                <td>Output</td>
                <td>Matmul computation output.</td>
                <td>shape (bs, N2).</td>
                <td>FLOAT16, BFLOAT16</td>
                <td>ND</td>
                <td>2</td>
                <td>x</td>
            </tr>
            <!-- workspaceSize -->
            <tr>
                <td>workspaceSize</td>
                <td>Output</td>
                <td>Returns the size of the workspace to be allocated on the Device.</td>
                <td>-</td>
                <td>UINT64</td>
                <td>ND</td>
                <td>-</td>
                <td>-</td>
            </tr>
            <!-- executor -->
            <tr>
                <td>executor</td>
                <td>Output</td>
                <td>Returns the op executor, which contains the operator computation flow.</td>
                <td>-</td>
                <td>aclOpExecutor*</td>
                <td>ND</td>
                <td>-</td>
                <td>-</td>
            </tr>
        </tbody>
    </table>

  The enumeration values of gmmXQuantMode, gmmWeightQuantMode, mmXQuantMode, mmWeightQuantMode, and commQuantMode are related to [quantization mode](../../../docs/en/context/quant_mode_introduction.md) as follows:
  - 0: non-quantization
  - 1: pertensor
  - 2: perchannel
  - 3: pertoken
  - 4: pergroup
  - 5: perblock
  - 6: mx quantization
  - 7: pertoken dynamic quantization

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
    <td>The tensor of a required input or output parameter is a null pointer.</td>
    </tr>
    <tr>
    <td>ACLNN_ERR_PARAM_INVALID</td>
    <td>161002</td>
    <td>The data type of an input or output is not within the supported range.</td>
    </tr>
    </tbody></table>

## aclnnQuantGroupedMatMulAlltoAllv

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
    <td>Workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>Input</td>
    <td>Size of the workspace allocated on the Device, obtained by the first-phase API <code>aclnnQuantGroupedMatMulAlltoAllvGetWorkspaceSize</code>.</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>Input</td>
    <td>Executor of the operator, containing the operator computation flow.</td>
    </tr>
    <tr>
    <td>stream</td>
    <td>Input</td>
    <td>Specifies the Stream for executing the task.</td>
    </tr>
    </tbody></table>

- **Return Value**

    Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnQuantGroupedMatMulAlltoAllv defaults to a deterministic implementation.
- Communication engine constraints:

  <!-- npu="950" id7 -->
  - Ascend 950PR&950DT products: AI_CPU communication is supported.
  <!-- end id7 -->

- e * epWorldSize supports a maximum of 256, where e (number of experts per device) supports a maximum of 32, and epWorldSize supports 2/4/8/16/32/64/128/256;
- The shape of gmmX is (A, H1), where A is the sum of sendCounts and the value range of H1 is (0, 65536);
- The shape of gmmWeight is (e, H1, N1), and the value range of N1 is (0, 65536);
- The shape of y is (BSK, N1), where the range of K in the first dimension is [2, 8] and BSK is the sum of recvCounts;
- mmX is the left matrix of the shared expert, with shape (BS, H2), and the value range of H2 is (0, 12288];
- mmWeight is the right matrix of the shared expert, with shape (H2, N2), and the value range of N2 is (0, 65536);
- sendCounts indicates the number of tokens sent to other devices, and the array size is e * epWorldSize;
- recvCounts indicates the number of tokens received from other devices, and the array size is e * epWorldSize;
- The quantization Scale, Mode, etc. of routed experts and shared experts are all required;
- For low-bit communication, Mode is a required parameter, and DType and Scale are optional. DType and Scale are required when Mode is non-zero.
- Variables used in shape in the parameter description:
  - BSK: The number of tokens received by the current card, which is the cumulative sum of the recvCounts parameter. Value range: (0, 52428800].
  - H1: Indicates the hidden size of the routed expert. Value range: (0, 65536).
  - H2: Indicates the hidden size of the shared expert. Value range: (0, 12288].
  - e: indicates the number of experts on each device, 0<e<=32, and e * epWorldSize supports a maximum of 256.
  - N1: indicates the intermediate layer dimension of the routing expert FFN, value range (0, 65536).
  - N2: indicates the intermediate layer dimension of the shared expert FFN, value range (0, 65536).
  - BS: batch sequence size.
  - K: indicates the number of TopK experts selected. K ranges from [2, 8].
  - A: the number of tokens sent by the current card, which is the cumulative sum of the sendCounts parameter.
  - The sum of parameter A on all devices in the ep communicator is equal to the sum of parameter BSK on all devices.
  - In mx quantization, when gmmX and gmmWeight are FLOAT4_E2M1, H1 and H2 must be even numbers and cannot be 2. When transGmmWeight and transMmWeight are false, N1 and N2 must be even numbers.
  - The transpose states of gmmWeight and gmmWeightScale must be consistent: both transposed or both non-transposed. The transpose states of mmWeight and mmWeightScale must also be consistent.
  - groupSize:
    - The groupSize value is valid only when all the gmmXScale/gmmWeightScale/mmXScale/mmWeightScale inputs are 2D or higher. Pass in 0 in other cases.
    - The groupSize value supports formula derivation: the input groupSize is internally decomposed into groupSizeM, groupSizeN, and groupSizeK according to the following formula. If one or more of them is 0, groupSizeM, groupSizeN, and groupSizeK are reset based on the shapes of the gmmX/gmmWeight/mmX/mmWeight/gmmXScale/gmmWeightScale/mmXScale/mmWeightScale inputs for computation. Resetting rules: If groupSizeM=0, the m-directional quantization group value is derived by the API, and the derivation formula is groupSizeM = m / scaleM (m must be divisible by scaleM), where m is consistent with the m in the gmmX/mmX shape and scaleM is consistent with the m in the gmmXScale/mmXScale shape; if groupSizeK=0, the k-directional quantization group value is derived by the API, and the derivation formula is groupSizeK = k / scaleK (k must be divisible by scaleK), where k is consistent with the k in the gmmX/mmX shape and scaleK is consistent with the k in the gmmXScale/mmXScale shape; if groupSizeN=0, the n-directional quantization group value is derived by the API, and the derivation formula is groupSizeN = n / scaleN (n must be divisible by scaleN), where n is consistent with the n in the gmmWeight/mmWeight shape and scaleN is consistent with the n in the gmmWeightScale/mmWeightScale shape.
    $$
    groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
    $$
    - If the reset condition is met, when the gmmXScale/gmmWeightScale/mmXScale/mmWeightScale inputs are 2D or higher and all are of the FLOAT8_E8M0 data type, the [groupSizeM, groupSizeN, groupSizeK] value combination is derived as [1, 1, 32], corresponding to a groupSize value of 4295032864.

- Quantization parameter constraints:
  - The current version supports pertensor quantization and MX quantization.
- Type constraints
  - pertensor quantization

    | gmmX | gmmWeight | gmmXScale | gmmWeightScale | mmXScale | mmWeightScale | y |
    | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
    | HIFLOAT8 | HIFLOAT8 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT16/BFLOAT16 |

  - mx quantization

    | gmmX | gmmWeight | gmmXScale | gmmWeightScale | mmXScale | mmWeightScale | y |
    | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
    | FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT16/BFLOAT16 |
    | FLOAT8_E4M3FN | FLOAT8_E5M2 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT16/BFLOAT16 |
    | FLOAT8_E5M2 | FLOAT8_E4M3FN | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT16/BFLOAT16 |
    | FLOAT8_E5M2 | FLOAT8_E5M2 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT16/BFLOAT16 |
    | FLOAT4_E2M1 | FLOAT4_E2M1 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT16/BFLOAT16 |

  - The data type of mmX is consistent with that of gmmX, the data type of mmWeight is consistent with that of gmmWeight, and the data type of mmY is consistent with that of y.

## Example

The following is sample code for reference only. For details about compilation and execution, see Compile and Run Samples.

NOTE: Because the quantization APIs support only Ascend 950PR&950DT products, the following example is implemented based on this product series. This sample code uses two devices as an example. Modify EP_WORLD_SIZE based on the actual number of devices in the environment.

<!-- npu="950" id8 -->
- <term>Ascend 950PR&950DT products</term>:

<!-- end id8 -->

```cpp
#include <thread>
#include <iostream>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "aclnnop/aclnn_quant_grouped_mat_mul_allto_allv.h"

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

// Basic shape information.
constexpr int64_t EP_WORLD_SIZE = 2;
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
    std::vector<int64_t> gmmXScaleShape = {1};
    std::vector<int64_t> gmmWScaleShape = {1};
    std::vector<int64_t> yShape = {BS * K, N1};
    std::vector<int64_t> mmXShape = {BS, H};
    std::vector<int64_t> mmWShape = {H, N2};
    std::vector<int64_t> mmXScaleShape = {1};
    std::vector<int64_t> mmWScaleShape = {1};
    std::vector<int64_t> mmYShape = {BS, N2};

    std::vector<int64_t> sendCountsShape = {EP_WORLD_SIZE * e};
    std::vector<int64_t> recvCountsShape = {EP_WORLD_SIZE * e};

    std::vector<int64_t> sendCountsList(EP_WORLD_SIZE * e, A / (EP_WORLD_SIZE * e));
    std::vector<int64_t> recvCountsList(EP_WORLD_SIZE * e, A / (EP_WORLD_SIZE * e));

    void *gmmXDeviceAddr = nullptr;
    void *gmmWDeviceAddr = nullptr;
    void *gmmXScaleDeviceAddr = nullptr;
    void *gmmWScaleDeviceAddr = nullptr;
    void *yDeviceAddr = nullptr;
    void *mmXDeviceAddr = nullptr;
    void *mmWDeviceAddr = nullptr;
    void *mmXScaleDeviceAddr = nullptr;
    void *mmWScaleDeviceAddr = nullptr;
    void *mmYDeviceAddr = nullptr;

    aclTensor *gmmX = nullptr;
    aclTensor *gmmW = nullptr;
    aclTensor *gmmXScale = nullptr;
    aclTensor *gmmWScale = nullptr;
    aclTensor *y = nullptr;

    aclTensor *mmX = nullptr;
    aclTensor *mmW = nullptr;
    aclTensor *mmXScale = nullptr;
    aclTensor *mmWScale = nullptr;
    aclTensor *mmY = nullptr;

    aclTensor *sendCountsTensor = nullptr;
    aclTensor *recvCountsTensor = nullptr;
    aclTensor *commQuantScaleOptional = nullptr;

    int64_t gmmXQuantMode = 1;
    int64_t gmmWQuantMode = 1;
    int64_t mmXQuantMode = 1;
    int64_t mmWQuantMode = 1;
    int64_t commQuantMode = 0;
    int64_t commQuantDtypeOptional = -1;
    int64_t groupSize = 0;

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    void *workspaceAddr = nullptr;

    long long gmmXShapeSize = GetShapeSize(gmmXShape);
    long long gmmWShapeSize = GetShapeSize(gmmWShape);
    long long gmmXScaleShapeSize = GetShapeSize(gmmXScaleShape);
    long long gmmWScaleShapeSize = GetShapeSize(gmmWScaleShape);
    long long yShapeSize = GetShapeSize(yShape);

    long long mmXShapeSize = GetShapeSize(mmXShape);
    long long mmWShapeSize = GetShapeSize(mmWShape);
    long long mmXScaleShapeSize = GetShapeSize(mmXScaleShape);
    long long mmWScaleShapeSize = GetShapeSize(mmWScaleShape);
    long long mmYShapeSize = GetShapeSize(mmYShape);

    std::vector<uint8_t> gmmXHostData(gmmXShapeSize, (args.rankId + 1) * 1024);  // HIFLOAT8
    std::vector<uint8_t> gmmWHostData(gmmWShapeSize, (args.rankId + 1) * 512);
    std::vector<float> gmmXScaleHostData(gmmXScaleShapeSize, 1);
    std::vector<float> gmmWScaleHostData(gmmWScaleShapeSize, 1);
    std::vector<int16_t> yHostData(yShapeSize, 65535);

    std::vector<uint8_t> mmXHostData(mmXShapeSize, (args.rankId + 1) * 1024);  // HIFLOAT8
    std::vector<uint8_t> mmWHostData(mmWShapeSize, (args.rankId + 1) * 512);
    std::vector<float> mmXScaleHostData(mmXScaleShapeSize, 1);
    std::vector<float> mmWScaleHostData(mmWScaleShapeSize, 1);
    std::vector<int16_t> mmYHostData(mmYShapeSize, 0);

    ret = CreateAclTensor(gmmXHostData, gmmXShape, &gmmXDeviceAddr, aclDataType::ACL_HIFLOAT8, &gmmX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gmmWHostData, gmmWShape, &gmmWDeviceAddr, aclDataType::ACL_HIFLOAT8, &gmmW);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gmmXScaleHostData, gmmXScaleShape, &gmmXScaleDeviceAddr, aclDataType::ACL_FLOAT, &gmmXScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gmmWScaleHostData, gmmWScaleShape, &gmmWScaleDeviceAddr, aclDataType::ACL_FLOAT, &gmmWScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT16, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(mmXHostData, mmXShape, &mmXDeviceAddr, aclDataType::ACL_HIFLOAT8, &mmX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(mmWHostData, mmWShape, &mmWDeviceAddr, aclDataType::ACL_HIFLOAT8, &mmW);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(mmXScaleHostData, mmXScaleShape, &mmXScaleDeviceAddr, aclDataType::ACL_FLOAT, &mmXScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(mmWScaleHostData, mmWScaleShape, &mmWScaleDeviceAddr, aclDataType::ACL_FLOAT, &mmWScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(mmYHostData, mmYShape, &mmYDeviceAddr, aclDataType::ACL_FLOAT16, &mmY);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclIntArray *sendCounts = aclCreateIntArray(sendCountsList.data(), sendCountsList.size());
    aclIntArray *recvCounts = aclCreateIntArray(recvCountsList.data(), recvCountsList.size());

    // Call the first-phase API.
    ret = aclnnQuantGroupedMatMulAlltoAllvGetWorkspaceSize(
        gmmX,
        gmmW,
        gmmXScale,
        gmmWScale,
        sendCountsTensor,
        recvCountsTensor,
        mmX,
        mmW,
        mmXScale,
        mmWScale,
        commQuantScaleOptional,
        gmmXQuantMode,
        gmmWQuantMode,
        mmXQuantMode,
        mmWQuantMode,
        commQuantMode,
        commQuantDtypeOptional,
        groupSize,
        hcomName,
        EP_WORLD_SIZE,
        sendCounts,
        recvCounts,
        false,
        false,
        y,
        mmY,
        &workspaceSize,
        &executor);
    CHECK_RET(
        ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnQuantGroupedMatMulAlltoAllvGetWorkspaceSize failed. ret = %d \n", ret);
        return ret);

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
    }
    // Call the second-phase API.
    ret = aclnnQuantGroupedMatMulAlltoAllv(workspaceAddr, workspaceSize, executor, args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnQuantGroupedMatMulAlltoAllv failed. ret = %d \n", ret);
            return ret);
    //(Boilerplate) Wait for task execution to complete synchronously.
    ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000000);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
            return ret);
    LOG_PRINT("[INFO] device_%d aclnnQuantGroupedMatMulAlltoAllv execute successfully.\n", args.rankId);
    // Release device resources. Modify based on the specific API definition.
    if (args.rankId == 0) {
        size_t size = A * N1 * sizeof(int16_t);
        aclrtMemcpy(pGmmYData.data(), size, yDeviceAddr, size, ACL_MEMCPY_DEVICE_TO_HOST);
    }
    if (gmmX != nullptr) {
        aclDestroyTensor(gmmX);
    }
    if (gmmW != nullptr) {
        aclDestroyTensor(gmmW);
    }
    if (gmmXScale != nullptr) {
        aclDestroyTensor(gmmXScale);
    }
    if (gmmWScale != nullptr) {
        aclDestroyTensor(gmmWScale);
    }
    if (y != nullptr) {
        aclDestroyTensor(y);
    }
    if (mmX != nullptr) {
        aclDestroyTensor(mmX);
    }
    if (mmW != nullptr) {
        aclDestroyTensor(mmW);
    }
    if (mmXScale != nullptr) {
        aclDestroyTensor(mmXScale);
    }
    if (mmWScale != nullptr) {
        aclDestroyTensor(mmWScale);
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
    if (gmmXScaleDeviceAddr != nullptr) {
        aclrtFree(gmmXScaleDeviceAddr);
    }
    if (gmmWScaleDeviceAddr != nullptr) {
        aclrtFree(gmmWScaleDeviceAddr);
    }
    if (yDeviceAddr != nullptr) {
        aclrtFree(yDeviceAddr);
    }
    if (mmXDeviceAddr != nullptr) {
        aclrtFree(mmXDeviceAddr);
    }
    if (mmWDeviceAddr != nullptr) {
        aclrtFree(mmWDeviceAddr);
    }
    if (mmXScaleDeviceAddr != nullptr) {
        aclrtFree(mmXScaleDeviceAddr);
    }
    if (mmWScaleDeviceAddr != nullptr) {
        aclrtFree(mmWScaleDeviceAddr);
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
