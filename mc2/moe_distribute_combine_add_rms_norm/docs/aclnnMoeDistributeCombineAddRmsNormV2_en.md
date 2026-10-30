# aclnnMoeDistributeCombineAddRmsNormV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T13:57:17.516Z -->

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

- Description: When TP domain communication exists, ReduceScatterV communication is performed first, followed by AllToAllV communication, and finally the received data is combined (multiplied by weights and summed). When TP domain communication does not exist, AllToAllV communication is performed, and finally the received data is combined (multiplied by weights and summed), after which the Add + RmsNorm fusion is completed. The operator supports special expert scenarios.

- Formula:

$$
rsOut = ReduceScatterV(expandX)\\
ataOut = AllToAllV(rsOut)\\
combineOut = Sum(expertScales * ataOut + expertScales * sharedExpertX)\\
x = combineOut + residualX\\
y = \frac{x}{RMS(x)} * gamma,\quad\text{where}\ RMS(x) = \sqrt{\frac{1}{H}\sum_{i=1}^{H}x_{i}^{2}+normEps}
$$

> Note: This API must be used together with `aclnnMoeDistributeDispatchV3`. It is equivalent to returning data along the same path as the data collection in the `aclnnMoeDistributeDispatchV3` API.

Compared with the `aclnnMoeDistributeCombineAddRmsNorm` API, this API introduces the following changes:

- Added support for special expert scenarios:
  - **zeroExpert ≠ 0**: Enabled by passing a value greater than 0 to the `zeroExpertNum` parameter.

    $$
        Moe(oriXOptional) = 0
    $$

  - **copyExpert ≠ 0**: Enabled by passing a value greater than 0 to the `copyExpertNum` parameter, and a valid `oriXOptional` parameter must be passed.

    $$
        Moe(oriXOptional) = oriXOptional
    $$

  - **constExpert ≠ 0**: Enabled by passing a value greater than 0 to the `constExpertNum` parameter, and valid `oriXOptional`, `constExpertAlpha1Optional`, `constExpertAlpha2Optional`, and `constExpertVOptional` parameters must be passed.

    $$
        Moe(oriXOptional) = constExpertAlpha1Optional * oriXOptional + constExpertAlpha2Optional * constExpertVOptional
    $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call the "aclnnMoeDistributeCombineAddRmsNormV2GetWorkspaceSize" API to obtain the workspace size required for the computation and the executor containing the operator computation process, and then call the "aclnnMoeDistributeCombineAddRmsNormV2" API to perform the computation.

```cpp
aclnnStatus aclnnMoeDistributeCombineAddRmsNormV2GetWorkspaceSize(
    const aclTensor* expandX,
    const aclTensor* expertIds,
    const aclTensor* assistInfoForCombine,
    const aclTensor* epSendCounts,
    const aclTensor* expertScales,
    const aclTensor* residualX,
    const aclTensor* gamma,
    const aclTensor* tpSendCountsOptional,
    const aclTensor* xActiveMaskOptional,
    const aclTensor* activationScaleOptional,
    const aclTensor* weightScaleOptional,
    const aclTensor* groupListOptional,
    const aclTensor* expandScalesOptional,
    const aclTensor* sharedExpertXOptional,
    const aclTensor* elasticInfoOptional,
    const aclTensor* oriXOptional,
    const aclTensor* constExpertAlpha1Optional,
    const aclTensor* constExpertAlpha2Optional,
    const aclTensor* constExpertVOptional,
    const char*      groupEp,
    int64_t          epWorldSize,
    int64_t          epRankId,
    int64_t          moeExpertNum,
    const char*      groupTp,
    int64_t          tpWorldSize,
    int64_t          tpRankId,
    int64_t          expertShardType,
    int64_t          sharedExpertNum,
    int64_t          sharedExpertRankNum,
    int64_t          globalBs,
    int64_t          outDtype,
    int64_t          commQuantMode,
    int64_t          groupListType,
    const char*      commAlg,
    float            normEps,
    int64_t          zeroExpertNum,
    int64_t          copyExpertNum,
    int64_t          constExpertNum,
    aclTensor*       yOut,
    aclTensor*       rstdOut,
    aclTensor*       xOut,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnMoeDistributeCombineAddRmsNormV2(
    void*           workspace,
    uint64_t        workspaceSize,
    aclOpExecutor*  executor,
    aclrtStream     stream)
```

## aclnnMoeDistributeCombineAddRmsNormV2GetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1550px">
    <colgroup>
    <col style="width: 180px">
    <col style="width: 120px">
    <col style="width: 280px">
    <col style="width: 320px">
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 140px">
    <col style="width: 140px">
    </colgroup>
    <thead>
    <tr>
    <th>Parameter</th>
    <th>Input/Output</th>
    <th>Description</th>
    <th>Instruction</th>
    <th>Data Type</th>
    <th>Data Format</th>
    <th>Dimension</th>
    <th>Non-contiguous</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>expandX</td>
    <td>Input</td>
    <td>Token features expanded based on expertIds.</td>
    <td>2D tensor on the device, with shape (A, H).</td>
    <td>BFLOAT16</td>
    <td>ND</td>
    <td>2</td>
    <td>√</td>
    </tr>
    <tr>
    <td>expertIds</td>
    <td>Input</td>
    <td>TopK expert indexes of each token.</td>
    <td>Device-side 2D Tensor with shape (Bs, K).</td>
    <td>INT32</td>
    <td>ND</td>
    <td>2</td>
    <td>√</td>
    </tr>
    <tr>
    <td>assistInfoForCombine</td>
    <td>Input</td>
    <td>Corresponding to the <code>assistInfoForCombineOut</code> output of <code>aclnnMoeDistributeDispatchV3</code>.</td>
    <td>Device-side 1D Tensor with shape (A * 128,).</td>
    <td>INT32</td>
    <td>ND</td>
    <td>1</td>
    <td>√</td>
    </tr>
    <tr>
    <td>epSendCounts</td>
    <td>Input</td>
    <td>Corresponds to the <code>epRecvCounts</code> output of <code>aclnnMoeDistributeDispatchV3</code>.</td>
    <td>Device-side 1D tensor with shape (epWorldSize * localExpertNum,).</td>
    <td>INT32</td>
    <td>ND</td>
    <td>1</td>
    <td>√</td>
    </tr>
    <tr>
    <td>expertScales</td>
    <td>Input</td>
    <td>TopK expert weights of each token.</td>
    <td>Device-side 2D tensor with shape (Bs, K).</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>2</td>
    <td>√</td>
    </tr>
    <tr>
    <td>residualX</td>
    <td>Input</td>
    <td>Right-hand matrix of Add in AddRmsNorm.</td>
    <td>3D tensor on the device, with shape (Bs, 1, H).</td>
    <td>BFLOAT16</td>
    <td>ND</td>
    <td>3</td>
    <td>√</td>
    </tr>
    <tr>
    <td>gamma</td>
    <td>Input</td>
    <td>gamma parameter in RmsNorm.</td>
    <td>Device-side 1D tensor with shape (H,).</td>
    <td>BFLOAT16</td>
    <td>ND</td>
    <td>1</td>
    <td>√</td>
    </tr>
    <tr>
    <td>tpSendCountsOptional</td>
    <td>Input</td>
    <td>Reserved Parameter. TP-domain communication is no longer supported.</td>
    <td>Not supported in the current version. Pass null pointer.</td>
    <td>INT32</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>xActiveMaskOptional</td>
    <td>Input</td>
    <td>Indicates whether tokens participate in communication.</td>
    <td><ul><li>Valid data or a null pointer can be passed. By default, all tokens participate in communication. The shape is (BS,) for 1D and (BS, K) for 2D.</li><li>If BS differs across cards, all tokens must be valid.</li></ul></td>
    <td>BOOL</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>activationScaleOptional</td>
    <td>Input</td>
    <td>Reserved Parameter.</td>
    <td>Not supported in the Current Version. Pass Null Pointer.</td>
    <td>-</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>weightScaleOptional</td>
    <td>Input</td>
    <td>Reserved parameter.</td>
    <td>Not supported in the current version. Pass a null pointer.</td>
    <td>-</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>groupListOptional</td>
    <td>Input</td>
    <td>Reserved parameter.</td>
    <td>Not supported in the current version. Pass a null pointer.</td>
    <td>-</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expandScalesOptional</td>
    <td>Input</td>
    <td>Corresponding to the <code>expandScales</code> output of <code>aclnnMoeDistributeDispatchV3</code>. Reserved parameter.</td>
    <td>Not supported in the current version. Pass a null pointer.</td>
    <td>-</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sharedExpertXOptional</td>
    <td>Input</td>
    <td>Tokens computed by the Shared Expert.</td>
    <td>Valid data or null pointer can be passed. On the Device side, the shape is (Bs, H) for 2D or (Bs, 1, H) for 3D.</td>
    <td>BFLOAT16</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>elasticInfoOptional</td>
    <td>Input</td>
    <td>Dynamic scale-in information of the EP communicator.</td>
    <td>Not supported in the current version. Pass a null pointer.</td>
    <td>INT32</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>oriXOptional</td>
    <td>Input</td>
    <td>Token data that does not pass through the FFN (oriXOptional in the formula).</td>
    <td>Mandatory when copyExpertNum/constExpertNum>0; if passed, it is a 2D tensor with shape (Bs, H).</td>
    <td>BFLOAT16</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>constExpertAlpha1Optional</td>
    <td>Input</td>
    <td>Calculation coefficient for the constExpert use case (constExpertAlpha1Optional in the formula).</td>
    <td>Required when constExpertNum>0; if passed, it is a 2D tensor with shape (constExpertNum, H).</td>
    <td>BFLOAT16</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>constExpertAlpha2Optional</td>
    <td>Input</td>
    <td>Computation coefficient for the constExpert use case (constExpertAlpha2Optional in the formula).</td>
    <td>Required when constExpertNum>0; if passed, it is a 2D tensor with shape (constExpertNum, H).</td>
    <td>BFLOAT16</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>constExpertVOptional</td>
    <td>Input</td>
    <td>Calculation coefficient for the constExpert use case (constExpertVOptional in the formula).</td>
    <td>Required when constExpertNum>0; if provided, it is a 2D tensor with shape (constExpertNum, H).</td>
    <td>BFLOAT16</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>groupEp</td>
    <td>Input</td>
    <td>Name of the EP communicator (expert parallelism).</td>
    <td>String length [1, 128)</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>epWorldSize</td>
    <td>Input</td>
    <td>Size of the EP communicator.</td>
    <td>Value range: (1, 768].</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>epRankId</td>
    <td>Input</td>
    <td>ID of the current device in the EP domain.</td>
    <td>Value range [0, epWorldSize). Values are unique within the same communicator.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>moeExpertNum</td>
    <td>Input</td>
    <td>Number of MoE experts.</td>
    <td>Value range (0, 1024], and <code>moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0</code> must be satisfied.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>groupTp</td>
    <td>Input</td>
    <td>TP communicator name (Reserved Parameter, not supported in the Current Version).</td>
    <td>String length [1, 128).</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>tpWorldSize</td>
    <td>Input</td>
    <td>Size of the TP communicator (Reserved Parameter. Not supported in the Current Version. Only 0 or 1 is allowed).</td>
    <td>Reserved Parameter. Not supported in the Current Version. Pass 0.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>tpRankId</td>
    <td>Input</td>
    <td>TP domain local device ID (reserved parameter, not supported in the current version, pass 0).</td>
    <td>Reserved parameter, not supported in the current version, pass 0.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expertShardType</td>
    <td>Input</td>
    <td>Distribution type of shared expert devices.</td>
    <td>Only 0 is supported (Shared Expert devices are placed before MoE expert devices).</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sharedExpertNum</td>
    <td>Input</td>
    <td>Number of Shared Experts.</td>
    <td>Not supported in the Current Version. Only 0 can be passed in.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sharedExpertRankNum</td>
    <td>Input</td>
    <td>Number of shared expert devices.</td>
    <td>Not supported in the Current Version. Only 0 can be passed.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>globalBs</td>
    <td>Input</td>
    <td>Global batch size in the EP domain.</td>
    <td><ul><li>When Bs is the same on each card: <code>globalBs = Bs*epWorldSize</code> or 0.</li><li>When Bs differs on each card: <code>globalBs = maxBs*epWorldSize</code> (maxBs is the maximum Bs on a single card).</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>outDtype</td>
    <td>Input</td>
    <td>Reserved parameter. Specifies the data type of output x.</td>
    <td>Not supported in the current version. Only 0 is supported.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>commQuantMode</td>
    <td>Input</td>
    <td>Communication quantization type.</td>
    <td>Not supported in the Current Version. Only 0 can be passed.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>groupListType</td>
    <td>Input</td>
    <td>Reserved Parameter, group List format.</td>
    <td>Not supported in the Current Version. Only 0 can be passed.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>commAlg</td>
    <td>Input</td>
    <td>Communication affinity memory layout algorithm.</td>
    <td>Reserved field. Not supported in the current version. Pass null pointer.</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>normEps</td>
    <td>Input</td>
    <td>Parameter used to prevent division-by-zero errors in AddRmsNorm.</td>
    <td>The value can be 1e-6.</td>
    <td>FLOAT</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>zeroExpertNum</td>
    <td>Input</td>
    <td>Number of zero experts.</td>
    <td>Value range [0, MAX_INT32), where MAX_INT32 is 2147483647. Valid expert ID range: <code>[moeExpertNum, moeExpertNum+zeroExpertNum)</code>.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>copyExpertNum</td>
    <td>Input</td>
    <td>Number of copy experts.</td>
    <td>Value Range [0, MAX_INT32), where the value of MAX_INT32 is 2147483647. The valid expert ID range is <code>[moeExpertNum+zeroExpertNum, moeExpertNum+zeroExpertNum+copyExpertNum)</code>.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>constExpertNum</td>
    <td>Input</td>
    <td>Number of constant experts.</td>
    <td>Value Range [0, MAX_INT32), where the value of MAX_INT32 is 2147483647. The valid expert ID range is <code>[moeExpertNum+zeroExpertNum+copyExpertNum, moeExpertNum+zeroExpertNum+copyExpertNum+constExpertNum)</code>.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>yOut</td>
    <td>Output</td>
    <td>Result after RmsNorm.</td>
    <td>Device-side 3D tensor, with shape (Bs, 1, H).</td>
    <td>BFLOAT16</td>
    <td>ND</td>
    <td>3</td>
    <td>-</td>
    </tr>
    <tr>
    <td>rstdOut</td>
    <td>Output</td>
    <td>Normalization coefficient computed by RmsNorm.</td>
    <td>Device-side 3D tensor with shape (Bs, 1, 1).</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>3</td>
    <td>√</td>
    </tr>
    <tr>
    <td>xOut</td>
    <td>Output</td>
    <td>Result after Add.</td>
    <td>Device-side 3D tensor with shape (Bs, 1, H).</td>
    <td>BFLOAT16</td>
    <td>ND</td>
    <td>3</td>
    <td>-</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>Output</td>
    <td>Size of the workspace to be requested on the Device side.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>Output</td>
    <td>Returns the op executor, which contains the computation flow of the operator.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    </tbody>
    </table>

    <!-- npu="A3" id7 -->
    <details>
    <summary><term>Atlas A3 products</term>  :</summary>

    - groupTp is a reserved parameter. It is not supported in the current version. Pass an empty string.
    - tpWorldSize is a reserved parameter. It is not supported in the current version. Pass 0.
    - tpRankId is a reserved parameter. It is not supported in the current version. Pass 0.
    </details>
    <!-- end id7 -->

    <!-- npu="950" id8 -->
    <details>
    <summary><term>Ascend 950PR&950DT products</term>:</summary>

    - groupTp is not supported in the current version. Pass an empty string.
    - tpWorldSize is not supported in the current version. Pass 0.
    - tpRankId is not supported in the current version. Pass 0.
    </details>
    <!-- end id8 -->

- **Return Value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API performs input parameter validation and returns an error in the following scenarios:

    <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
    <col style="width: 305px">
    <col style="width: 119px">
    <col style="width: 725px">
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
    <td>1. The Tensor of a required input or output parameter is a null pointer.</td>
    </tr>
    <tr>
    <td>ACLNN_ERR_PARAM_INVALID</td>
    <td>161002</td>
    <td>1. The data type of inputs or outputs is not supported.</td>
    </tr>
    <tr>
    <td class="merged-cell" rowspan="2">ACLNN_ERR_INNER_TILING_ERROR</td>
    <td class="merged-cell" rowspan="2">561002</td>
    <td>1. The shape of inputs or outputs is not supported;</td>
    </tr>
    <tr>
    <td>2. The parameter value is not supported.</td>
    </tr>
    </tbody>
    </table>

## aclnnMoeDistributeCombineAddRmsNormV2

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
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>workspace</td>
    <td>Input</td>
    <td>Workspace memory address allocated on the Device side.</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>Input</td>
    <td>Workspace size allocated on the Device side, obtained by calling the first-phase API <code>aclnnMoeDistributeCombineAddRmsNormV2GetWorkspaceSize</code>.</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>Input</td>
    <td>op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
    <td>stream</td>
    <td>Input</td>
    <td>Specifies the AscendCL Stream for executing tasks.</td>
    </tr>
    </tbody>
    </table>

- **Return Value**

    Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnMoeDistributeCombineAddRmsNormV2 defaults to a deterministic implementation.

- API compatibility constraints:
  - The aclnnMoeDistributeDispatchV3 API and the aclnnMoeDistributeCombineAddRmsNormV2 API must be used together. For details, see [Example](#example).

- Parameter Consistency Constraints
  - During the API call, the parameters expertIds, xActiveMaskOptional, elasticInfoOptional, groupEp, epWorldSize, moeExpertNum, expertShardType, sharedExpertNum, sharedExpertRankNum, globalBs, commAlg, zeroExpertNum, copyExpertNum, and constExpertNum, as well as the value of HCCL_BUFFSIZE, must be the same on all cards, consistent across different layers in the network, and consistent with the corresponding parameters of DispatchV3.

- Product-Specific Constraints

  <!-- npu="A3" id9 -->
  - <term>Atlas A3 products</term>: In this use case, each card contains dual DIEs (also referred to as dies), so "the current card" in the parameter description indicates a single DIE.

  <!-- end id9 -->

- Shape Variable Constraints:
    - A: the maximum number of tokens that the current card needs to distribute. The value range is as follows:
        - If globalBs is 0, ensure A >= BS *epWorldSize* min(localExpertNum, K);
        - When globalBs is not 0, A >= globalBs * min(localExpertNum, K).

    - H: the hidden size. The value range is [1024, 8192].
    - Bs: the batch sequence size, that is, the number of tokens finally output by the current card. The value range is 0 < Bs ≤ 512.
    - K: indicates the number of topK experts selected. The value range is 0 < K ≤ 16 and also satisfies 0 < K ≤ moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum.
    - localExpertNum: indicates the number of experts on the current card.
        - For a shared expert card, localExpertNum = 1.
        - For MoE expert devices, localExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum). TP domain communication is not supported in the current version.

- Environment variable constraints:
    - HCCL_BUFFSIZE: Before calling this API, check whether the value of the HCCL_BUFFSIZE environment variable is reasonable. This environment variable specifies the memory size occupied by a single communication domain, in MB. The default value is 200 MB when not configured.
        - In the ep communicator: the size setting must be >= 2 and satisfy 1024 ^ 2 *(HCCL_BUFFSIZE - 2) / 2 >= BS* 2 *(H + 128)* (epWorldSize * localExpertNum + K + 1), where localExpertNum indicates the number of local experts on each MoE expert device.
        - In the tp communicator: reserved parameter. Not supported in the current version.

- Communication domain usage constraints:
    - Within a model, aclnnMoeDistributeCombineAddRmsNormV2 and aclnnMoeDistributeDispatchV3 only support the same EP communication domain, and no other operators are allowed in this communication domain.
    - TP domain communication is not supported currently.

    <!-- npu="A3" id10 -->
    - <term>Atlas A3 products</term>: Nodes in one communicator must be in one SuperPoD. Cross-SuperPoD is not supported.

    <!-- end id10 -->

- Communication mode constraints:

    <!-- npu="950" id11 -->
    - <term>Ascend 950PR&950DT products</term>: Only UB Memory communication is supported.

    <!-- end id11 -->

- Other constraints:
    - moeExpertNum + zeroExpertNum + copyExpertNum + constExpertNum < MAX_INT32, where the value of MAX_INT32 is 2147483647.

    <!-- npu="950,A3" id12 -->
    - <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>: Dynamic scale-in is not currently supported.

    <!-- end id12 -->

## Example

<term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>: call the aclnnMoeDistributeDispatchV3 and aclnnMoeDistributeCombineAddRmsNormV2 APIs. For the compilation and execution procedure, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

- The sample code is as follows, for reference only.

    ```Cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <vector>
    #include <unordered_set>
    #include "acl/acl.h"
    #include "hccl/hccl.h"
    #include "aclnnop/aclnn_moe_distribute_dispatch_v3.h"
    #include "aclnnop/aclnn_moe_distribute_combine_add_rms_norm_v2.h"

    #define CHECK_RET(cond, return_expr) \
        do {                             \
            if (!(cond)) {               \
                return_expr;             \
            }                            \
        } while (0)

    #define LOG_PRINT(message, ...)         \
        do {                                \
            printf(message, ##__VA_ARGS__); \
        } while (0)

    struct Args {
        uint32_t rankId;
        uint32_t epRankId;
        HcclComm hcclEpComm;
        aclrtStream dispatchStream;
        aclrtStream combineStream;
        aclrtContext context;
    };

    constexpr uint32_t EP_WORLD_SIZE = 2;
    constexpr uint32_t DEV_NUM = EP_WORLD_SIZE;

    int64_t GetShapeSize(const std::vector<int64_t> &shape)
    {
        int64_t shape_size = 1;
        for (auto i : shape) {
            shape_size *= i;
        }
        return shape_size;
    }

    template<typename T>
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
        *tensor = aclCreateTensor(
            shape.data(), shape.size(), dataType, strides.data(), 0,
            aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
        return 0;
    }

    int LaunchOneProcessDispatchAndCombine(Args &args)
    {
        int ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed, ret %d\n", ret); return ret);

        char hcomEpName[128] = {0};
        ret = HcclGetCommName(args.hcclEpComm, hcomEpName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetEpCommName failed, ret %d\n", ret); return -1);
        LOG_PRINT(
            "[INFO] rank = %d, hcomEpName = %s, dispatchStream = %p, combineStream = %p, context = %p\n",
            args.rankId, hcomEpName, args.dispatchStream, args.combineStream, args.context
        );

        int64_t BS = 8;
        int64_t H = 7168;
        int64_t K = 3;
        int64_t expertShardType = 0;
        int64_t sharedExpertNum = 0;
        int64_t sharedExpertRankNum = 0;
        int64_t moeExpertNum = 8;
        int64_t quantMode = 0;
        int64_t globalBS = BS * EP_WORLD_SIZE;
        int64_t expertTokenNumsType = 1;
        int64_t outDtype = 0;
        int64_t commQuantMode = 0;
        int64_t groupList_type = 1;
        int64_t localExpertNum;
        int64_t A;
        int64_t zeroExpertNum = 7;
        int64_t copyExpertNum = 8;
        int64_t constExpertNum = 10;
        if (args.epRankId < sharedExpertRankNum) {
            // Shared expert device
            localExpertNum = 1;
            A = globalBS / sharedExpertRankNum;
        } else {
            // Moe expert devices
            localExpertNum = moeExpertNum / (EP_WORLD_SIZE - sharedExpertRankNum);
            A = globalBS * (localExpertNum < K ? localExpertNum : K);
        }

        /* Construct device-side input and output variables based on the current use case. */
        // Declare device-side input and output variables.
        void *xDeviceAddr = nullptr;
        void *expertIdsDeviceAddr = nullptr;
        void *scalesDeviceAddr = nullptr;
        void *expertScalesDeviceAddr = nullptr;
        void *expandXDeviceAddr = nullptr;
        void *dynamicScalesDeviceAddr = nullptr;
        void *expandIdxDeviceAddr = nullptr;
        void *expertTokenNumsDeviceAddr = nullptr;
        void *epRecvCountsDeviceAddr = nullptr;
        void *expandScalesDeviceAddr = nullptr;
        void *residualXDeviceAddr = nullptr;
        void *sharedExpertXDeviceAddr = nullptr;
        void *gammaDeviceAddr = nullptr;

        void *elasticInfoDeviceAddr = nullptr;
        void *oriXDeviceAddr = nullptr;
        void *constExpertAlpha1DeviceAddr = nullptr;
        void *constExpertAlpha2DeviceAddr = nullptr;
        void *constExpertVDeviceAddr = nullptr;
        void *tpRecvCountsDeviceAddr = nullptr;

        void *yOutDeviceAddr = nullptr;
        void *rstdOutDeviceAddr = nullptr;
        void *xOutDeviceAddr = nullptr;

        aclTensor *x = nullptr;
        aclTensor *expertIds = nullptr;
        aclTensor *scales = nullptr;
        aclTensor *expertScales = nullptr;
        aclTensor *expandX = nullptr;
        aclTensor *dynamicScales = nullptr;
        aclTensor *expandIdx = nullptr;
        aclTensor *expertTokenNums = nullptr;
        aclTensor *epRecvCounts = nullptr;
        aclTensor *expandScales = nullptr;
        aclTensor *residualX = nullptr;
        aclTensor *sharedExpertX = nullptr;
        aclTensor *gamma = nullptr;

        aclTensor *elasticInfo = nullptr;
        aclTensor *oriX = nullptr;
        aclTensor *constExpertAlpha1 = nullptr;
        aclTensor *constExpertAlpha2 = nullptr;
        aclTensor *constExpertV = nullptr;
        aclTensor *tpRecvCounts = nullptr;

        aclTensor *yOut = nullptr;
        aclTensor *rstdOut = nullptr;
        aclTensor *xOut = nullptr;

        // Define the dimensions of each variable for the current use case.
        std::vector<int64_t> xShape{BS, H};
        std::vector<int64_t> expertIdsShape{BS, K};
        std::vector<int64_t> scalesShape{(sharedExpertRankNum > 0) ? moeExpertNum + 1 : moeExpertNum, H};
        std::vector<int64_t> expertScalesShape{BS, K};
        std::vector<int64_t> expandXShape{A, H};
        std::vector<int64_t> dynamicScalesShape{A};
        std::vector<int64_t> expandIdxShape{A * 128};
        std::vector<int64_t> expertTokenNumsShape{localExpertNum};
        std::vector<int64_t> epRecvCountsShape{localExpertNum * EP_WORLD_SIZE};
        std::vector<int64_t> expandScalesShape{A};
        std::vector<int64_t> residualXShape{BS, 1, H};
        std::vector<int64_t> sharedExpertXShape{BS, 1, H};
        std::vector<int64_t> gammaShape{H};

        std::vector<int64_t> elasticInfoShape{4 + 2 * EP_WORLD_SIZE};
        std::vector<int64_t> oriXShape{BS, H};
        std::vector<int64_t> constExpertAlpha1Shape{constExpertNum, H};
        std::vector<int64_t> constExpertAlpha2Shape{constExpertNum, H};
        std::vector<int64_t> constExpertVShape{constExpertNum, H};
        std::vector<int64_t> tpRecvCountsShape{1};

        std::vector<int64_t> yOutShape{BS, 1, H};
        std::vector<int64_t> rstdOutShape{BS, 1, 1};
        std::vector<int64_t> xOutShape{BS, 1, H};

        int64_t xShapeSize = GetShapeSize(xShape);
        int64_t expertIdsShapeSize = GetShapeSize(expertIdsShape);
        int64_t scalesShapeSize = GetShapeSize(scalesShape);
        int64_t expertScalesShapeSize = GetShapeSize(expertScalesShape);
        int64_t expandXShapeSize = GetShapeSize(expandXShape);
        int64_t dynamicScalesShapeSize = GetShapeSize(dynamicScalesShape);
        int64_t expandIdxShapeSize = GetShapeSize(expandIdxShape);
        int64_t expertTokenNumsShapeSize = GetShapeSize(expertTokenNumsShape);
        int64_t epRecvCountsShapeSize = GetShapeSize(epRecvCountsShape);
        int64_t expandScalesShapeSize = GetShapeSize(expandScalesShape);
        int64_t residualXShapeSize = GetShapeSize(residualXShape);
        int64_t sharedExpertXShapeSize = GetShapeSize(sharedExpertXShape);
        int64_t gammaShapeSize = GetShapeSize(gammaShape);

        int64_t elasticInfoSize = GetShapeSize(elasticInfoShape);
        int64_t oriXSize = GetShapeSize(oriXShape);
        int64_t constExpertAlpha1Size = GetShapeSize(constExpertAlpha1Shape);
        int64_t constExpertAlpha2Size = GetShapeSize(constExpertAlpha2Shape);
        int64_t constExpertVSize = GetShapeSize(constExpertVShape);
        int64_t tpRecvCountsShapeSize = GetShapeSize(tpRecvCountsShape);

        int64_t yOutShapeSize = GetShapeSize(yOutShape);
        int64_t rstdOutShapeSize = GetShapeSize(rstdOutShape);
        int64_t xOutShapeSize = GetShapeSize(xOutShape);

        // Construct host-side variables.
        std::vector<int16_t> xHostData(xShapeSize, 1);
        std::vector<int32_t> expertIdsHostData;
        for (int32_t token_id = 0; token_id < expertIdsShape[0]; token_id++) {
            // Each token is sent to moe experts {0, 1, ... k - 1}.
            for (int32_t k_id = 0; k_id < expertIdsShape[1]; k_id++) {
                expertIdsHostData.push_back(k_id);
            }
        }

        std::vector<float> scalesHostData(scalesShapeSize, 0.1);
        std::vector<float> expertScalesHostData(expertScalesShapeSize, 0.1);
        std::vector<int16_t> expandXHostData(expandXShapeSize, 0);
        std::vector<float> dynamicScalesHostData(dynamicScalesShapeSize, 0);
        std::vector<int32_t> expandIdxHostData(expandIdxShapeSize, 0);
        std::vector<int64_t> expertTokenNumsHostData(expertTokenNumsShapeSize, 0);
        std::vector<int32_t> epRecvCountsHostData(epRecvCountsShapeSize, 0);
        std::vector<float> expandScalesHostData(expandScalesShapeSize, 0);
        std::vector<int16_t> residualXHostData(residualXShapeSize, 1);
        std::vector<int16_t> sharedExpertXHostData(sharedExpertXShapeSize, 1);
        std::vector<int16_t> gammaHostData(gammaShapeSize, 1);

        std::vector<int16_t> oriXHostData(oriXSize, 1);
        std::vector<int16_t> constExpertAlpha1HostData(constExpertAlpha1Size, 0);
        std::vector<int16_t> constExpertAlpha2HostData(constExpertAlpha2Size, 0);
        std::vector<int16_t> constExpertVHostData(constExpertVSize, 0);
        std::vector<int32_t> tpRecvCountsHostData(tpRecvCountsShapeSize, 0);

        std::vector<int16_t> yOutHostData(yOutShapeSize, 0);
        std::vector<float> rstdOutHostData(rstdOutShapeSize, 0);
        std::vector<int16_t> xOutHostData(xOutShapeSize, 0);

        // Construct device-side variables.
        ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expertIdsHostData, expertIdsShape, &expertIdsDeviceAddr, aclDataType::ACL_INT32, &expertIds);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(scalesHostData, scalesShape, &scalesDeviceAddr, aclDataType::ACL_FLOAT, &scales);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expertScalesHostData, expertScalesShape, &expertScalesDeviceAddr, aclDataType::ACL_FLOAT, &expertScales);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expandXHostData, expandXShape, &expandXDeviceAddr, (quantMode > 0) ? aclDataType::ACL_INT8 : aclDataType::ACL_BF16, &expandX);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(dynamicScalesHostData, dynamicScalesShape, &dynamicScalesDeviceAddr, aclDataType::ACL_FLOAT, &dynamicScales);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expandIdxHostData, expandIdxShape, &expandIdxDeviceAddr, aclDataType::ACL_INT32, &expandIdx);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expertTokenNumsHostData, expertTokenNumsShape, &expertTokenNumsDeviceAddr, aclDataType::ACL_INT64, &expertTokenNums);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(epRecvCountsHostData, epRecvCountsShape, &epRecvCountsDeviceAddr, aclDataType::ACL_INT32, &epRecvCounts);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expandScalesHostData, expandScalesShape, &expandScalesDeviceAddr, aclDataType::ACL_FLOAT, &expandScales);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(residualXHostData, residualXShape, &residualXDeviceAddr, aclDataType::ACL_BF16, &residualX);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(sharedExpertXHostData, sharedExpertXShape, &sharedExpertXDeviceAddr, aclDataType::ACL_BF16, &sharedExpertX);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_BF16, &gamma);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        ret = CreateAclTensor(oriXHostData, oriXShape, &oriXDeviceAddr, aclDataType::ACL_BF16, &oriX);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(constExpertAlpha1HostData, constExpertAlpha1Shape, &constExpertAlpha1DeviceAddr, aclDataType::ACL_BF16, &constExpertAlpha1);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(constExpertAlpha2HostData, constExpertAlpha2Shape, &constExpertAlpha2DeviceAddr, aclDataType::ACL_BF16, &constExpertAlpha2);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(constExpertVHostData, constExpertVShape, &constExpertVDeviceAddr, aclDataType::ACL_BF16, &constExpertV);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(tpRecvCountsHostData, tpRecvCountsShape, &tpRecvCountsDeviceAddr, aclDataType::ACL_INT32, &tpRecvCounts);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(yOutHostData, yOutShape, &yOutDeviceAddr, aclDataType::ACL_BF16, &yOut);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(rstdOutHostData, rstdOutShape, &rstdOutDeviceAddr, aclDataType::ACL_FLOAT, &rstdOut);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(xOutHostData, xOutShape, &xOutDeviceAddr, aclDataType::ACL_BF16, &xOut);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        /* Declare variables required for operator execution. */
        uint64_t dispatchWorkspaceSize = 0;
        aclOpExecutor *dispatchExecutor = nullptr;
        void *dispatchWorkspaceAddr = nullptr;

        uint64_t combineAddRmsNormWorkspaceSize = 0;
        aclOpExecutor *combineAddRmsNormExecutor = nullptr;
        void *combineWorkspaceAddr = nullptr;

        /**************************************** Call dispatch warm up********************************************/
        ret = aclnnMoeDistributeDispatchV3GetWorkspaceSize(x, expertIds, (quantMode > 0 ? scales : nullptr), nullptr,
                expertScales, nullptr, hcomEpName, EP_WORLD_SIZE, args.epRankId, moeExpertNum, "", 0,
                0, expertShardType, sharedExpertNum, sharedExpertRankNum, quantMode, globalBS,
                expertTokenNumsType, nullptr, zeroExpertNum, copyExpertNum, constExpertNum, expandX, dynamicScales, expandIdx, expertTokenNums, epRecvCounts,
                tpRecvCounts, expandScales, &dispatchWorkspaceSize, &dispatchExecutor);

        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] warm up aclnnMoeDistributeDispatchV3GetWorkspaceSize failed. ret = %d \n", ret); return ret);

        if (dispatchWorkspaceSize > 0) {
            ret = aclrtMalloc(&dispatchWorkspaceAddr, dispatchWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] warm up aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }
        // Call the second-phase API.
        ret = aclnnMoeDistributeDispatchV3(dispatchWorkspaceAddr, dispatchWorkspaceSize,
                                            dispatchExecutor, args.dispatchStream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] warm up aclnnMoeDistributeDispatchV3 failed. ret = %d \n", ret);  \
                return ret);
        ret = aclrtSynchronizeStreamWithTimeout(args.dispatchStream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] warm up aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);  \
            return ret);

        /**************************************** Call dispatch. ********************************************/
        // Call the first-phase API.
        ret = aclnnMoeDistributeDispatchV3GetWorkspaceSize(x, expertIds, (quantMode > 0 ? scales : nullptr), nullptr,
                expertScales, elasticInfo, hcomEpName, EP_WORLD_SIZE, args.epRankId, moeExpertNum, "", 0,
                0, expertShardType, sharedExpertNum, sharedExpertRankNum, quantMode, globalBS,
                expertTokenNumsType, nullptr, zeroExpertNum, copyExpertNum, constExpertNum, expandX, dynamicScales, expandIdx, expertTokenNums, epRecvCounts,
                tpRecvCounts, expandScales, &dispatchWorkspaceSize, &dispatchExecutor);

        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchV3GetWorkspaceSize failed. ret = %d \n", ret); return ret);

        if (dispatchWorkspaceSize > 0) {
            ret = aclrtMalloc(&dispatchWorkspaceAddr, dispatchWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }
        // Call the second-phase API.
        ret = aclnnMoeDistributeDispatchV3(dispatchWorkspaceAddr, dispatchWorkspaceSize,
                                            dispatchExecutor, args.dispatchStream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchV3 failed. ret = %d \n", ret);  \
            return ret);
        ret = aclrtSynchronizeStreamWithTimeout(args.dispatchStream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] dispatch aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);  \
            return ret);


        /**************************************** Call combineAddRmsNorm. ********************************************/
        // Call the first-phase API.
        ret = aclnnMoeDistributeCombineAddRmsNormV2GetWorkspaceSize(
            expandX, expertIds, expandIdx, epRecvCounts, expertScales, residualX, gamma, tpRecvCounts, nullptr, nullptr,
            nullptr, nullptr, nullptr, sharedExpertX, elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV,
            hcomEpName, EP_WORLD_SIZE, args.epRankId, moeExpertNum, "", 0,
            0, expertShardType, sharedExpertNum, sharedExpertRankNum, globalBS, outDtype, commQuantMode,
            groupList_type, nullptr, 1e-6, zeroExpertNum, copyExpertNum, constExpertNum, yOut, rstdOut, xOut,
            &combineAddRmsNormWorkspaceSize, &combineAddRmsNormExecutor);
        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnMoeDistributeCombineAddRmsNormV2GetWorkspaceSize failed. ret = %d \n", ret); return ret);
        // Allocate device memory based on the workspaceSize calculated by the first-phase API.
        if (combineAddRmsNormWorkspaceSize > 0) {
            ret = aclrtMalloc(&combineWorkspaceAddr, combineAddRmsNormWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }

        // Call the second-phase API.
        ret = aclnnMoeDistributeCombineAddRmsNormV2(combineWorkspaceAddr, combineAddRmsNormWorkspaceSize, combineAddRmsNormExecutor, args.combineStream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeCombineAddRmsNormV2 failed. ret = %d \n", ret);
            return ret);
        // (boilerplate) Synchronize and wait for the task execution to complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.combineStream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
            return ret);
        LOG_PRINT("[INFO] device_%d aclnnMoeDistributeDispatchV3 and aclnnMoeDistributeCombineAddRmsNormV2                      \
                    execute successfully.\n", args.rankId);


        // Release device resources.
        if (dispatchWorkspaceSize > 0) {
            aclrtFree(dispatchWorkspaceAddr);
        }
        if (combineAddRmsNormWorkspaceSize > 0) {
            aclrtFree(combineWorkspaceAddr);
        }
        if (x != nullptr) {
            aclDestroyTensor(x);
        }
        if (expertIds != nullptr) {
            aclDestroyTensor(expertIds);
        }
        if (scales != nullptr) {
            aclDestroyTensor(scales);
        }
        if (expertScales != nullptr) {
            aclDestroyTensor(expertScales);
        }
        if (expandX != nullptr) {
            aclDestroyTensor(expandX);
        }
        if (dynamicScales != nullptr) {
            aclDestroyTensor(dynamicScales);
        }
        if (expandIdx != nullptr) {
            aclDestroyTensor(expandIdx);
        }
        if (expertTokenNums != nullptr) {
            aclDestroyTensor(expertTokenNums);
        }
        if (epRecvCounts != nullptr) {
            aclDestroyTensor(epRecvCounts);
        }
        if (expandScales != nullptr) {
            aclDestroyTensor(expandScales);
        }
        if (residualX != nullptr) {
            aclDestroyTensor(residualX);
        }
        if (sharedExpertX != nullptr) {
            aclDestroyTensor(sharedExpertX);
        }
        if (gamma != nullptr) {
            aclDestroyTensor(gamma);
        }

        if (elasticInfo != nullptr) {
            aclDestroyTensor(elasticInfo);
        }
        if (oriX != nullptr) {
            aclDestroyTensor(oriX);
        }
        if (constExpertAlpha1 != nullptr) {
            aclDestroyTensor(constExpertAlpha1);
        }
        if (constExpertAlpha2 != nullptr) {
            aclDestroyTensor(constExpertAlpha2);
        }
        if (constExpertV != nullptr) {
            aclDestroyTensor(constExpertV);
        }
        if (tpRecvCounts != nullptr) {
            aclDestroyTensor(tpRecvCounts);
        }

        if (yOut != nullptr) {
            aclDestroyTensor(yOut);
        }
        if (rstdOut != nullptr) {
            aclDestroyTensor(rstdOut);
        }
        if (xOut != nullptr) {
            aclDestroyTensor(xOut);
        }
        if (xDeviceAddr != nullptr) {
            aclrtFree(xDeviceAddr);
        }
        if (expertIdsDeviceAddr != nullptr) {
            aclrtFree(expertIdsDeviceAddr);
        }
        if (scalesDeviceAddr != nullptr) {
            aclrtFree(scalesDeviceAddr);
        }
        if (expertScalesDeviceAddr != nullptr) {
            aclrtFree(expertScalesDeviceAddr);
        }
        if (expandXDeviceAddr != nullptr) {
            aclrtFree(expandXDeviceAddr);
        }
        if (dynamicScalesDeviceAddr != nullptr) {
            aclrtFree(dynamicScalesDeviceAddr);
        }
        if (expandIdxDeviceAddr != nullptr) {
            aclrtFree(expandIdxDeviceAddr);
        }
        if (expertTokenNumsDeviceAddr != nullptr) {
            aclrtFree(expertTokenNumsDeviceAddr);
        }
        if (epRecvCountsDeviceAddr != nullptr) {
            aclrtFree(epRecvCountsDeviceAddr);
        }
        if (expandScalesDeviceAddr != nullptr) {
            aclrtFree(expandScalesDeviceAddr);
        }
        if (residualXDeviceAddr != nullptr) {
            aclrtFree(residualXDeviceAddr);
        }
        if (sharedExpertXDeviceAddr != nullptr) {
            aclrtFree(sharedExpertXDeviceAddr);
        }
        if (gammaDeviceAddr != nullptr) {
            aclrtFree(gammaDeviceAddr);
        }

        if (elasticInfoDeviceAddr != nullptr) {
            aclrtFree(elasticInfoDeviceAddr);
        }
        if (oriXDeviceAddr != nullptr) {
            aclrtFree(oriXDeviceAddr);
        }
        if (constExpertAlpha1DeviceAddr != nullptr) {
            aclrtFree(constExpertAlpha1DeviceAddr);
        }
        if (constExpertAlpha2DeviceAddr != nullptr) {
            aclrtFree(constExpertAlpha2DeviceAddr);
        }
        if (constExpertVDeviceAddr != nullptr) {
            aclrtFree(constExpertVDeviceAddr);
        }
        if (tpRecvCountsDeviceAddr != nullptr) {
            aclrtFree(tpRecvCountsDeviceAddr);
        }

        if (yOutDeviceAddr != nullptr) {
            aclrtFree(yOutDeviceAddr);
        }
        if (rstdOutDeviceAddr != nullptr) {
            aclrtFree(rstdOutDeviceAddr);
        }
        if (xOutDeviceAddr != nullptr) {
            aclrtFree(xOutDeviceAddr);
        }
        HcclCommDestroy(args.hcclEpComm);
        aclrtDestroyStream(args.dispatchStream);
        aclrtDestroyStream(args.combineStream);
        aclrtDestroyContext(args.context);
        aclrtResetDevice(args.rankId);

        return 0;
    }

    int main(int argc, char *argv[])
    {
        int ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed, ret = %d\n", ret); return ret);

        aclrtStream dispatchStream[DEV_NUM];
        aclrtStream combineStream[DEV_NUM];
        aclrtContext context[DEV_NUM];
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            ret = aclrtSetDevice(rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed, ret = %d\n", ret); return ret);
            ret = aclrtCreateContext(&context[rankId], rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed, ret = %d\n", ret); return ret);
            ret = aclrtCreateStream(&dispatchStream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed, ret = %d\n", ret); return ret);
            ret = aclrtCreateStream(&combineStream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed, ret = %d\n", ret); return ret);
        }

        int32_t devicesEp[EP_WORLD_SIZE];
        for (int32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
            devicesEp[epId] = epId;
        }

        HcclComm commsEp[EP_WORLD_SIZE];
        ret = HcclCommInitAll(EP_WORLD_SIZE, devicesEp, commsEp);
        CHECK_RET(ret == ACL_SUCCESS,
                    LOG_PRINT("[ERROR] HcclCommInitAll ep failed, ret %d\n", ret); return ret);

        Args args[DEV_NUM];
        // Each thread calls each card to execute the operator.
        std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            uint32_t epRankId = rankId;

            args[rankId].rankId = rankId;
            args[rankId].epRankId = epRankId;
            args[rankId].hcclEpComm = commsEp[epRankId];
            args[rankId].dispatchStream = dispatchStream[rankId];
            args[rankId].combineStream = combineStream[rankId];
            args[rankId].context = context[rankId];
            threads[rankId].reset(new (std::nothrow) std::thread(&LaunchOneProcessDispatchAndCombine, std::ref(args[rankId])));
        }

        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            threads[rankId]->join();
        }

        aclFinalize();
        LOG_PRINT("[INFO] aclFinalize success\n");

        return 0;
    }
    ```
