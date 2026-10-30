# aclnnQuantGroupedMatMulAlltoAllvV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T14:42:34.725Z -->

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
    - Fuses quantized routed expert GroupedMatMul, Unpermute, and AlltoAllv, and achieves parallel fusion with shared expert MatMul. Computation is performed before communication. Supports the pertensor-pertensor and mx [quantization modes](../../../docs/en/context/quant_mode_introduction.md).

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

Compared with the `aclnnQuantGroupedMatMulAlltoAllv` API, changes in this API are as follows:

- Adds the `commMode` parameter, which users can use to specify the communication engine used by the device.

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>: supports `ai_cpu` and `ccu`.

  <!-- end id7 -->

## Function Prototype

This operator is a two-phase API. Call the `aclnnQuantGroupedMatMulAlltoAllvV2GetWorkspaceSize` API first to obtain the input parameters and calculate the required workspace size and the executor that contains the operator computation flow based on the computation flow. Then call the `aclnnQuantGroupedMatMulAlltoAllvV2` API to execute the computation.

```cpp
aclnnStatus aclnnQuantGroupedMatMulAlltoAllvV2GetWorkspaceSize(
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
    const char*        commMode,
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
aclnnStatus aclnnQuantGroupedMatMulAlltoAllvV2(
    void*           workspace,
    uint64_t        workspaceSize,
    aclOpExecutor*  executor,
    aclrtStream     stream)
```

## aclnnQuantGroupedMatMulAlltoAllvV2GetWorkspaceSize

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
                <td>shape (e, H1, N1). e is the number of experts deployed per card, H1 is the hidden size, and N1 is the intermediate dimension of the routed expert FFN.</td>
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
                <td><li>pertensor quantization: 1</li><li>mx quantization: 3</li></td>
                <td>x</td>
            </tr>
            <!-- gmmWeightScale -->
            <tr>
                <td>gmmWeightScale</td>
                <td>Input</td>
                <td>Quantization coefficient of gmmWeight.</td>
                <td><li>pertensor quantization: shape (1).</li><li>mx quantization: shape (e, N1, ceil(H1/64), 2), or (e, ceil(H1/64), N1, 2) when weight is transposed</li></td>
                <td>FLOAT32, FLOAT8_E8M0</td>
                <td>ND</td>
                <td><li>pertensor quantization: 1</li><li>mx quantization: 3</li></td>
                <td>x</td>
            </tr>
            <!-- sendCountsTensorOptional -->
            <tr>
                <td>sendCountsTensorOptional</td>
                <td>Input</td>
                <td>Send count used by AlltoAllv.</td>
                <td>Currently only supports empty. shape (e * ep, ), where e is the number of experts deployed per card and ep is the size of the ep domain.</td>
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
                <td>Currently only supports empty. Shape (e * ep, ). e indicates the number of experts deployed per card, and ep indicates the size of the ep domain.</td>
                <td>INT64</td>
                <td>ND</td>
                <td>1</td>
                <td>x</td>
            </tr>
            <!-- mmXOptional -->
            <tr>
                <td>mmXOptional</td>
                <td>Input</td>
                <td>The input mmX in the formula.</td>
                <td>shape (BS, H2). BS is the batch sequence size, and H2 is the hidden size.</td>
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
                <td>shape (H2, N2). H2 is the hidden size, and N2 is the intermediate layer dimension of the shared expert FFN.</td>
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
                <td><li>pertensor quantization: shape(1).</li><li>mx quantization: shape (N2, ceil(H2/64), 2), (ceil(H2/64), N2, 2) when weight is transposed.</li></td>
                <td>FLOAT32, FLOAT8_E8M0</td>
                <td>ND</td>
                <td><li>pertensor quantization: 1</li><li>mx quantization: 3</li></td>
                <td>x</td>
            </tr>
            <!-- commQuantScaleOptional -->
            <tr>
                <td>commQuantScaleOptional</td>
                <td>Input</td>
                <td>Low-bit communication quantization coefficient.</td>
                <td>Reserved parameter. Currently only supports empty.</td>
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
                <td>The quantization mode must be passed in. Currently supports 1 (pertensor quantization) and 6 (mx quantization).</td>
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
                <td>The quantization mode must be passed in. Currently supports 1 (pertensor quantization) and 6 (mx quantization).</td>
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
                <td>If mmX is not empty, the quantization mode must be passed in. Currently only 1 (pertensor quantization) and 6 (mx quantization) are supported.</td>
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
                <td>If mmWeight is not empty, the quantization mode must be passed in. Currently supports 1 (pertensor quantization) and 6 (mx quantization).</td>
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
                <td>Currently reserved for the low-bit feature. 0 must be passed in, indicating no quantization.</td>
                <td>INT64</td>
                <td>-</td>
                <td>1</td>
                <td>x</td>
            </tr>
            <!-- commQuantDtypeOptional -->
            <tr>
                <td>commQuantDtypeOptional</td>
                <td>Input</td>
                <td>Data type for low-bit communication.</td>
                <td>Currently reserved for the low-bit feature. -1 must be passed in.</td>
                <td>INT64</td>
                <td>-</td>
                <td>1</td>
                <td>x</td>
            </tr>
            <!-- groupSize (newly added) -->
            <tr>
                <td>groupSize</td>
                <td>Input</td>
                <td>PerGroup quantization group size.</td>
                <td>Quantization group sizes in three directions for the Matmul computation. Reserved parameter. The value is valid only when the quantization Scale input has 2 dimensions and above; 0 must be passed in for other use cases. The groupSize input is formed by concatenating three values for the three directions: groupSizeM, groupSizeN, and groupSizeK. Each value occupies 16 bits, together occupying the lower 48 bits of the int64_t groupSize (the upper 16 bits are invalid). The calculation formula is: groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32.</td>
                <td>INT64</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
            </tr>
            <!-- group -->
            <tr>
                <td>group</td>
                <td>Input</td>
                <td>Communicator ID.</td>
                <td>String length must be greater than 0 and less than 128.</td>
                <td>char*</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
            </tr>
            <!-- commMode -->
            <tr>
                <td>commMode</td>
                <td>Input</td>
                <td>Specifies the current communication type.</td>
                <td>Supports "ai_cpu" and "ccu" as input.</td>
                <td>char*</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
            </tr>
            <!-- epWorldSize -->
            <tr>
                <td>epWorldSize</td>
                <td>Input</td>
                <td>Size of the communicator.</td>
                <td>Supported values: 2/4/8/16/32/64/128/256</td>
                <td>INT64</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
            </tr>
            <!-- sendCounts -->
            <tr>
                <td>sendCounts</td>
                <td>Input</td>
                <td>send count used by AlltoAllv. Indicates the number of tokens sent by other ranks to each expert on the current rank.</td>
                <td>The supported dimension is e * ep. Flattened into one dimension according to <code>sendCounts[fromRank][expertId]</code>. For example, when e=3, the order is <code>e0,e1,e2,e0,e1,e2, ...</code>.</td>
                <td>aclIntArray* (element type INT64)</td>
                <td>ND</td>
                <td>-</td>
                <td>-</td>
            </tr>
            <!-- recvCounts -->
            <tr>
                <td>recvCounts</td>
                <td>Input</td>
                <td>recv count used by AlltoAllv. Indicates the number of tokens that the current device needs to receive after AlltoAllv.</td>
                <td>The supported dimension is e * ep. Flattened into one dimension according to <code>recvCounts[fromRank][expertId]</code>. For example, when e=3, the order is <code>e0,e1,e2,e0,e1,e2, ...</code>.</td>
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
                <td>Must be passed in, with no default value.</td>
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
                <td>Must be passed in. No default value.</td>
                <td>BOOL</td>
                <td>ND</td>
                <td>-</td>
                <td>-</td>
            </tr>
            <!-- y -->
            <tr>
                <td>y</td>
                <td>Output</td>
                <td>GroupedMatMul computation output.</td>
                <td>Empty tensors are not supported. Shape (BSK, N1).</td>
                <td>FLOAT16, BFLOAT16</td>
                <td>ND</td>
                <td>2</td>
                <td>x</td>
            </tr>
            <!-- mmYOptional -->
            <tr>
                <td>mmYOptional</td>
                <td>Output</td>
                <td>matmul computation output.</td>
                <td>Shape (bs, N2).</td>
                <td>FLOAT16, BFLOAT16</td>
                <td>ND</td>
                <td>2</td>
                <td>x</td>
            </tr>
            <!-- workspaceSize -->
            <tr>
                <td>workspaceSize</td>
                <td>Output</td>
                <td>Returns the size of the workspace that needs to be applied on the Device side.</td>
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

  The mapping between the enumeration values of gmmXQuantMode, gmmWeightQuantMode, mmXQuantMode, mmWeightQuantMode, commQuantMode and [quantization modes](../../../docs/en/context/quant_mode_introduction.md) is as follows:
  - 0: non-quantized
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
    <td>A required input or output tensor parameter is a null pointer.</td>
    </tr>
    <tr>
    <td>ACLNN_ERR_PARAM_INVALID</td>
    <td>161002</td>
    <td>The data type of an input or output is not within the supported range.</td>
    </tr>
    </tbody></table>

## aclnnQuantGroupedMatMulAlltoAllvV2

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
    <td>Workspace memory address applied on the Device side.</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>Input</td>
    <td>Size of the workspace applied on the Device side, obtained by the first-phase API <code>aclnnQuantGroupedMatMulAlltoAllvV2GetWorkspaceSize</code>.</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>Input</td>
    <td>op executor, which contains the operator computation process.</td>
    </tr>
    <tr>
    <td>stream</td>
    <td>Input</td>
    <td>Specifies the Stream for executing tasks.</td>
    </tr>
    </tbody></table>

- **Return Value**

    Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnQuantGroupedMatMulAlltoAllvV2 defaults to a deterministic implementation.
- Communication engine constraints:

  <!-- npu="950" id8 -->
  - Ascend 950PR&950DT products: supports CCU communication and AI_CPU communication. CCU supports only interconnection within the UB domain of a single node, while AI_CPU supports interconnection within the UB domain across nodes.
  <!-- end id8 -->

- The maximum product of e * epWorldSize is 256, where e (the number of experts per card) supports a maximum of 32, and epWorldSize supports 2/4/8/16/32/64/128/256;
- The shape of gmmX is (A, H1), where A is the sum of sendCounts, and the value range of H1 is (0, 65536);
- The shape of gmmWeight is (e, H1, N1), and the value range of N1 is (0, 65536);
- The shape of y is (BSK, N1), where K in the first dimension has a range of [2, 8], and BSK is the sum of recvCounts;
- mmX is the left matrix of the shared expert, with the shape (BS, H2). The value range of H2 is (0, 12288];
- mmWeight is the right matrix of the shared expert, with the shape (H2, N2). The value range of N2 is (0, 65536);
- sendCounts indicates the number of tokens sent to other devices, and the array size is e * epWorldSize;
- recvCounts indicates the number of tokens received from other devices, and the array size is e * epWorldSize;
- The quantization Scale, Mode, and other parameters of routed experts and shared experts are all mandatory;
- low-bit communication Mode is a mandatory parameter; DType and Scale are optional and must be provided when Mode is non-zero;
- Variables used in shape in the parameter description:
  - BSK: The number of tokens received by the current card, which is the cumulative sum of the recvCounts parameter. Value range: (0, 52428800].
  - H1: Indicates the hidden size of the routed expert. Value range: (0, 65536).
  - H2: Indicates the hidden size of the shared expert. Value range: (0, 12288].
  - e: indicates the number of experts on a single device. 0<e<=32, and e * epWorldSize supports a maximum of 256.
  - N1: indicates the intermediate layer dimension of the routing expert FFN. Value range: (0, 65536).
  - N2: indicates the intermediate layer dimension of the shared expert FFN. Value range: (0, 65536).
  - BS: batch sequence size.
  - K: indicates the number of TopK experts selected. K ranges from [2, 8].
  - A: the number of tokens sent by the current card, which is the cumulative sum of the sendCounts parameter.
  - The sum of the A parameters of all devices in the ep communicator equals the sum of the BSK parameters on all devices.
  - For mx quantization, when gmmX and gmmWeight are FLOAT4_E2M1, H1 and H2 must be even numbers and cannot be 2. In addition, when transGmmWeight and transMmWeight are false, N1 and N2 must be even numbers.
  - The transpose states of gmmWeight and gmmWeightScale must be consistent: both transposed or both not transposed. The transpose states of mmWeight and mmWeightScale must also be consistent.
  - groupSize:
    - The value of groupSize is valid only when the gmmXScale/gmmWeightScale/mmXScale/mmWeightScale inputs are all 2 dimensions and above. In other use cases, 0 must be passed in.
    - The groupSize value supports formula derivation: the input groupSize is internally decomposed into groupSizeM, groupSizeN, and groupSizeK according to the following formula. If one or more of them is 0, the groupSizeM, groupSizeN, and groupSizeK used for computation are reset based on the shapes of the gmmX/gmmWeight/mmX/mmWeight/gmmXScale/gmmWeightScale/mmXScale/mmWeightScale inputs. Setting principle: If groupSizeM=0, the quantization group value in the m direction is determined by interface derivation, and the derivation formula is groupSizeM = m / scaleM (m must be divisible by scaleM), where m is consistent with m in the gmmX/mmX shape and scaleM is consistent with m in the gmmXScale/mmXScale shape; if groupSizeK=0, the quantization group value in the k direction is determined by interface derivation, and the derivation formula is groupSizeK = k / scaleK (k must be divisible by scaleK), where k is consistent with k in the gmmX/mmX shape and scaleK is consistent with k in the gmmXScale/mmXScale shape; if groupSizeN=0, the quantization group value in the n direction is determined by interface derivation, and the derivation formula is groupSizeN = n / scaleN (n must be divisible by scaleN), where n is consistent with n in the gmmWeight/mmWeight shape and scaleN is consistent with n in the gmmWeightScale/mmWeightScale shape.
    $$
    groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
    $$
    - If the reset condition is met, when the gmmXScale/gmmWeightScale/mmXScale/mmWeightScale inputs are 2 dimensions and above and their data types are all FLOAT8_E8M0, the [groupSizeM, groupSizeN, groupSizeK] value combination is derived as [1, 1, 32], and the corresponding groupSize value is 4295032864.

- Quantization parameter constraints:
  - The current version supports pertensor quantization and mx quantization.
- Type constraints
  - Pertensor quantization

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

  - The mmX type and gmmX type must be consistent, the mmWeight type and gmmWeight type must be consistent, and the mmY type and y type must be consistent.

## Example

The following is sample code for reference only. For details about compilation and execution, see Compile and Run Samples.

NOTE: The sample code calls some HCCL collective communication library APIs, including HcclGetCommName, HcclCommInitAll, and HcclCommDestroy. For details, see [*HCCL API (C)*](https://hiascend.com/en/document/redirect/CannCommunityHcclCppApi).

<!-- npu="950" id9 -->
- <term>Ascend 950PR&950DT products</term>:

    ```cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <vector>
    #include "acl/acl.h"
    #include "hccl/hccl.h"
    #include "aclnnop/aclnn_quant_grouped_mat_mul_allto_allv_v2.h"

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
        ret = aclnnQuantGroupedMatMulAlltoAllvV2GetWorkspaceSize(
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
            "ai_cpu",
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
            ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnQuantGroupedMatMulAlltoAllvV2GetWorkspaceSize failed. ret = %d \n", ret);
            return ret);

        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }
        // Call the second-phase API.
        ret = aclnnQuantGroupedMatMulAlltoAllvV2(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnQuantGroupedMatMulAlltoAllvV2 failed. ret = %d \n", ret);
                return ret);
        // (boilerplate) Synchronize and wait for task execution to complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
                return ret);
        LOG_PRINT("[INFO] device_%d aclnnQuantGroupedMatMulAlltoAllvV2 execute successfully.\n", args.rankId);
        // Release device resources, which need to be modified based on the definition of the specific API.
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

<!-- end id9 -->
