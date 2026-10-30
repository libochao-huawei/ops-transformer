# aclnnMoeDistributeDispatch

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T14:16:08.033Z -->

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

- Description: Quantizes (optional) the Token data and performs AllToAllV communication in the EP (Expert Parallelism) domain. TP (Tensor Parallelism) domain communication is no longer supported in the current version, and TP-related parameters are reserved parameters.

- Formula:

    - Case 1: If quantMode=0 (non-quantization scenario):

    $$
    expandXOut = AllToAllV(X)
    $$

    - Scenario 2: If quantMode=2 (per-token dynamic quantization scenario):

    $$
    xFp32 = CastToFp32(X) \times scales \\
    dynamicScales = dstTypeMax/Max(Abs(xFp32)) \\
    quantOut = CastToInt8(xFp32 \times dynamicScalesValue) \\
    allToAllXOut = AllToAllV(quantOut) \\
    allToAllDynamicScalesOut = AllToAllV(1.0/dynamicScales) \\
    expandXOut = AllToAllV(quantOut) \\
    dynamicScalesOut = allToAllDynamicScalesOut
    $$

>Note: This API must be used together with aclnnMoeDistributeCombine.

## Function Prototype

Each operator provides a two-phase API. Call "aclnnMoeDistributeDispatchGetWorkspaceSize" first to obtain the workspace size required for computation and the executor that contains the operator computation process. Then call "aclnnMoeDistributeDispatch" to perform the computation.

```cpp
aclnnStatus aclnnMoeDistributeDispatchGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* expertIds,
    const aclTensor* scales,
    const aclTensor* xActiveMask,
    const aclTensor* expertScales,
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
    int64_t          quantMode,
    int64_t          globalBS,
    int64_t          expertTokenNumsType,
    aclTensor*       expandX,
    aclTensor*       dynamicScales,
    aclTensor*       expandIdx,
    aclTensor*       expertTokenNums,
    aclTensor*       epRecvCounts,
    aclTensor*       tpRecvCounts,
    aclTensor*       expandScales,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnMoeDistributeDispatch(
    void            *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream     stream)
```

## aclnnMoeDistributeDispatchGetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1567px"> <colgroup>
    <col style="width: 120px">
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
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>x</td>
    <td>Input</td>
    <td>Token data sent by the local device.</td>
    <td>Required to be a 2D tensor.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>(BS, H) (BS is the batch size, H is the hidden layer size)</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expertIds</td>
    <td>Input</td>
    <td>The topK expert indexes of each token.</td>
    <td>Required to be a 2D Tensor.</td>
    <td>INT32</td>
    <td>ND</td>
    <td>(BS, K)</td>
    <td>-</td>
    </tr>
    <tr>
    <td>scales</td>
    <td>Input</td>
    <td>Smoothing weight of each expert, quantization coefficient of fused quantization smoothing weight, or quantization coefficient.</td>
    <td>Required to be a 2D Tensor.</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>Value is (moeExpertNum, H) when there is no shared expert device; value is (sharedExpertNum + moeExpertNum, H) when shared expert devices exist.</td>
    <td>-</td>
    </tr>
    <tr>
    <td>xActiveMask</td>
    <td>Input</td>
    <td>Reserved parameter.</td>
    <td>Not supported in the current version. Pass a null pointer.</td>
    <td>-</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expertScales</td>
    <td>Input</td>
    <td>TopK expert weights of each token.</td>
    <td>Required to be a 2D tensor.</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>(BS, K)</td>
    <td>-</td>
    </tr>
    <tr>
    <td>groupEp</td>
    <td>Input</td>
    <td>Name of the EP communicator (expert parallelism communicator).</td>
    <td>String length range: [1, 128).</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>epWorldSize</td>
    <td>Input</td>
    <td>Size of the EP communicator.</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>epRankId</td>
    <td>Input</td>
    <td>Local device ID in the EP domain.</td>
    <td><ul><li>Value range [0, epWorldSize)</li><li>The epRankId of each device in the same EP communicator domain must be unique.</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>moeExpertNum</td>
    <td>Input</td>
    <td>Number of MoE experts.</td>
    <td>Value range (0, 512], and moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>groupTp</td>
    <td>Input</td>
    <td>TP communicator name (reserved parameter, not supported in the current version).</td>
    <td>The default value is "".</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>tpWorldSize</td>
    <td>Input</td>
    <td>Size of the TP communicator (reserved parameter; not supported in the current version).</td>
    <td>Only 0 or 1 can be passed in. The default value is 0.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>tpRankId</td>
    <td>Input</td>
    <td>Local device ID in the TP domain (reserved parameter; not supported in the current version).</td>
    <td>Pass 0. Default value is 0.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expertShardType</td>
    <td>Input</td>
    <td>Indicates the distribution type of shared expert devices.</td>
    <td>Currently, only 0 is supported, indicating that shared expert devices are placed before MoE expert devices.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sharedExpertNum</td>
    <td>Input</td>
    <td>Number of shared experts (one shared expert can be replicated and deployed on multiple devices).</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sharedExpertRankNum</td>
    <td>Input</td>
    <td>Indicates the number of shared expert devices.</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>quantMode</td>
    <td>Input</td>
    <td>Quantization mode.</td>
    <td>Supported values: 0: no quantization, 2: per-token dynamic quantization.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>globalBS</td>
    <td>Input</td>
    <td>Global batch size in the EP domain.</td>
    <td><ul><li>If BS is the same on all ranks, globalBS = BS * epWorldSize or 0.</li><li>If BS differs across ranks, globalBS = maxBS * epWorldSize (maxBS is the maximum BS of a single device/rank).</li></ul></td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expertTokenNumsType</td>
    <td>Input</td>
    <td>Semantic type of the values in output expertTokenNums.</td>
    <td>Supported values: 0: the output in expertTokenNums is the prefix sum of the number of tokens processed by each expert; 1: the output in expertTokenNums is the number of tokens processed by each expert.</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expandX</td>
    <td>Output</td>
    <td>Token features expanded according to expertIds.</td>
    <td>Required to be a 2D tensor.</td>
    <td>INT8, FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>dynamicScales</td>
    <td>Output</td>
    <td>In quantization use cases, indicates the quantization scale of Tokens output by the local device.</td>
    <td>Required to be a 1D or 2D tensor.</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>-</td>
    <td>√</td>
    </tr>
    <tr>
    <td>expandIdx</td>
    <td>Output</td>
    <td>Indicates the number of tokens sent to the same expert (corresponding to expandIdx in aclnnMoeDistributeCombine).</td>
    <td>Required to be a 1D tensor.</td>
    <td>INT32</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expertTokenNums</td>
    <td>Output</td>
    <td>Indicates the number of tokens received by each expert.</td>
    <td>Required to be a 1D tensor.</td>
    <td>INT64</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>epRecvCounts</td>
    <td>Output</td>
    <td>Number of tokens received from each device in the EP communicator (corresponding to epSendCounts in aclnnMoeDistributeCombine).</td>
    <td>Required to be a 1D tensor.</td>
    <td>INT32</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>tpRecvCounts</td>
    <td>Output</td>
    <td>Reserved parameter. No longer supported for TP domain communication.</td>
    <td>Not supported in the current version. Pass a null pointer.</td>
    <td>INT32</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expandScales</td>
    <td>Output</td>
    <td>Weights of Tokens output by the local device (corresponding to expandScales in aclnnMoeDistributeCombine).</td>
    <td>Required to be a 1D tensor.</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
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
    </tbody>
    </table>

    <!-- npu="910b" id7 -->
    - <term>Atlas A2 products</term>:
        - Does not support the shared expert scenario, and does not support the `expertShardType`, `sharedExpertNum`, and `sharedExpertRankNum` attributes.
        - Only supports the EP domain with no TP domain. Does not support the `groupTp`, `tpWorldSize`, and `tpRankId` attributes. `tpRecvCounts` is invalid.
        - `expandScales` is valid only when the environment variables `HCCL_INTRA_PCIE_ENABLE` = 1 and `HCCL_INTRA_ROCE_ENABLE` = 0 are set.

    <!-- end id7 -->
    <!-- npu="A3" id8 -->
    - <term>Atlas A3 products</term>  :
        - `expandScales` is not supported.
        - The `xActiveMask` input is not supported.
        - The current value range of `sharedExpertNum` is [0, 1], where 0 indicates no shared expert and 1 indicates one shared expert. Only 1 is supported in the current version.
        - The current value range of `sharedExpertRankNum` is [0, epWorldSize). When it is not 0, epWorldSize % sharedExpertRankNum = 0 must be satisfied.

    <!-- end id8 -->
    <!-- npu="950" id9 -->
    - <term>Ascend 950PR&950DT products</term>:
        - `expandScales` is not supported.
        - The `xActiveMask` input is not supported.
        - The current value range of `sharedExpertNum` is [0, 1], where 0 indicates no shared expert and 1 indicates one shared expert. Only 1 is supported in the current version.
        - The current value range of `sharedExpertRankNum` is [0, epWorldSize). When it is not 0, epWorldSize % sharedExpertRankNum = 0 must be satisfied.
        - TP domain communication is not currently supported. The `groupTp`, `tpWorldSize`, and `tpRankId` attributes are not supported, and `tpSendCounts` is invalid.

    <!-- end id9 -->

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
    <th>Returns</th>
    <th>Error Code</th>
    <th>Description</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>ACLNN_ERR_PARAM_NULLPTR</td>
    <td>161001</td>
    <td>The required input or output tensor parameter is a null pointer.</td>
    </tr>
    <tr>
    <td>ACLNN_ERR_PARAM_INVALID</td>
    <td>161002</td>
    <td>The data type of the input or output is not in the supported range.</td>
    </tr>
    <tr>
    <td rowspan="2">ACLNN_ERR_INNER_TILING_ERROR</td>
    <td rowspan="2">561002</td>
    <td>The input and output shapes are not within the supported range.</td>
    </tr>
    <tr>
        <td>The parameter value is not within the supported range.</td>
    </tr>
    </tbody>
    </table>

## aclnnMoeDistributeDispatch

- **Parameters:**

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
    <td>Workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>Input</td>
    <td>Size of the workspace allocated on the Device, obtained by the first-phase API <code>aclnnMoeDistributeDispatchGetWorkspaceSize</code>.</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>Input</td>
    <td>op executor, containing the operator computation flow.</td>
    </tr>
    <tr>
    <td>stream</td>
    <td>Input</td>
    <td>Specifies the stream for executing tasks.</td>
    </tr>
    </tbody>
    </table>

## Constraints

- Deterministic computation:
  - aclnnMoeDistributeDispatch defaults to a deterministic implementation.

- Driver constraints:
  - The driver versions of all nodes in the operator communication domain must be the same.

- The `MoeDistributeDispatch` operator must be used together with the `MoeDistributeCombine` operator. For details, see the example.

- Across different product models, communication algorithms, or versions, the element values in the `MoeDistributeDispatch` tensor outputs `expandIdx`, `epRecvCounts`, `tpRecvCounts`, and `expandScales` may differ. When using them, simply pass the above tensors to the corresponding parameters of `MoeDistributeCombine`. Other model business logic should not depend on these values.

- The `groupEp`, `epWorldSize`, `moeExpertNum`, `groupTp`, `tpWorldSize`, `expertShardType`, `sharedExpertNum`, `sharedExpertRankNum`, and `globalBS` attribute values used during the operator call must be consistent across all cards and across different layers in the network, and must also be consistent with the corresponding parameters of `MoeDistributeCombine`.

- Description of the shape format in Parameters:
    - `A`: Indicates the maximum number of tokens that the current card may receive. The value range is as follows:
        - For shared experts, ensure that `A` = `BS` \* `epWorldSize` \* `sharedExpertNum` / `sharedExpertRankNum`.
        - For MoE experts, when `globalBS` is 0, ensure that `A` >= `BS` \* `epWorldSize` \* min(`localExpertNum`, `K`); when `globalBS` is not 0, ensure that `A` >= `globalBS` \* min(`localExpertNum`, `K`).
    - `localExpertNum`: Indicates the number of experts on the current card.
        - For shared expert cards, `localExpertNum` = 1.
        - For MoE expert devices, `localExpertNum` = `moeExpertNum` / (`epWorldSize` - `sharedExpertRankNum`)`; TP-domain communication is not supported in the current version.

- The "/" in the formulas in this document denotes integer division.

- Communication domain usage constraints:
    - `MoeDistributeCombine` and `MoeDistributeDispatch` in a model only support the same EP communication domain, and no other operators are allowed in this communication domain.
    - TP domain communication is not supported currently.

    <!-- npu="A3" id10 -->
    - <term>Atlas A3 products</term>  : Nodes in a communicator must reside within one SuperPoD; cross-SuperPoD communication is not supported.

    <!-- end id10 -->

- Communication mode constraints:

    <!-- npu="950" id11 -->
    - <term>Ascend 950PR&950DT products</term>: Only UB Memory communication is supported.

    <!-- end id11 -->

<!-- npu="910b" id12 -->
- <term>Atlas A2 products</term>:
    - Description of the shape format in Parameters:
        - `H`: Indicates the hidden size. The value range is (0, 7168] and must be an integer multiple of 32.
        - `BS`: Indicates the batch sequence size, that is, the number of tokens finally output by the current card. The value range is [1, 256].
        - `K`: Indicates the top-K experts selected. It must satisfy 0 < `K` ≤ moeExpertNum, with a value range of [1, 16].
    - `HCCL_BUFFSIZE`: Before calling this operator, check whether the value of the `HCCL_BUFFSIZE` environment variable is reasonable. This environment variable indicates the memory size occupied by a single communication domain, in MB. It defaults to 200 MB when not configured, and must be >= 2 \* (`BS` \* `epWorldSize` \* min(`localExpertNum`, `K`) \* `H` \* sizeof(uint16) + 2 MB).
    - `HCCL_INTRA_PCIE_ENABLE` and `HCCL_INTRA_ROCE_ENABLE`: Setting the environment variables `HCCL_INTRA_PCIE_ENABLE` = 1 and `HCCL_INTRA_ROCE_ENABLE` = 0 can reduce the amount of cross-machine communication data and may improve operator performance. In this case, `HCCL_BUFFSIZE` must be >= `moeExpertNum` \* `BS` \* (`H` \* sizeof(dtypeX) + 4 \* ((`K` + 7) / 8 \* 8) \* sizeof(uint32)) + 4 MB + 100 MB. Additionally, for the input parameter `moeExpertNum`, only `moeExpertNum` % `epWorldSize` = 0 is required, and `moeExpertNum` / `epWorldSize` <= 24 is not required, but the `scales` feature is not supported.
    - `epWorldSize`: Supported values are 16, 32, and 64.
    - Constraints related to `quantMode`:
        - When `quantMode` is set to 2, it indicates a per-token dynamic quantization scenario, and the data type of `expandX` supports `INT8`.
            - A null pointer can be passed for the input `scales`.
            - If valid data is passed to input `scales`, the shape is (`moeExpertNum`, `H`).
    - Networking constraint: In multi-machine scenarios, only switch networking is supported. Direct dual-machine connection is not supported.

<!-- end id12 -->
<!-- npu="A3" id13 -->
- <term>Atlas A3 products</term>  :
    - In this scenario, a single card contains two DIE (short for "die" or "bare die"), so "this card" in the parameter description refers to a single DIE.
    - Description of the shape format in Parameters:
        - `H`: Indicates the hidden size, with a value of 7168.
        - `BS`: Indicates the batch sequence size, i.e., the number of tokens finally output by the current card, with a value range of [1, 512].
        - `K`: Indicates the top-K experts to select. It must satisfy 0 < `K` ≤ moeExpertNum, with a value range of [1, 8].
    - `HCCL_BUFFSIZE`: Before calling this API, check whether the value of the `HCCL_BUFFSIZE` environment variable is reasonable. This environment variable indicates the memory size occupied by a single communication domain, in MB. The default value is 200 MB when not configured.
        - Within the ep communicator: Required to be >= 2 and satisfy 1024 ^ 2 \* (`HCCL_BUFFSIZE` - 2) / 2 >= `BS` \* 2 \* (`H` + 128) \* (`epWorldSize` \* `localExpertNum` + `K` + 1). `localExpertNum` must be the number of local experts on the MoE expert device.
        - Within the tp communicator: Reserved parameter. Not supported in the current version.
    - `epWorldSize`: Supported values include 8, 16, 32, 64, 128, 144, 256, and 288.
    - Constraints related to `quantMode`:
        - When `quantMode` is set to 2, it indicates a per-token dynamic quantization scenario, and the data type of `expandX` supports `INT8`.
            - A null pointer can be passed for the input `scales`.
            - If valid data is passed to input `scales` and shared expert devices exist, the shape is (`sharedExpertNum` + `moeExpertNum`, `H`).
            - If valid data is passed to input `scales` and no shared expert devices exist, the shape is (`moeExpertNum`, `H`).

<!-- end id13 -->
<!-- npu="950" id14 -->
- <term>Ascend 950PR&950DT products</term>:
    - Description of the shape format in Parameters:
        - `H`: Indicates the hidden size, with a value of 7168.
        - `BS`: Indicates the batch sequence size, i.e., the number of tokens finally output by the current card, with a value range of [1, 512].
        - `K`: Indicates the selection of top-K experts, with a value range of [1, 8], and must satisfy 0 < `K` ≤ moeExpertNum.
    - `epWorldSize`: Supported values are 2, 4, 8, 16, 32, 64, 128, 144, 256, and 288.
    - `HCCL_BUFFSIZE`: Before calling this operator, check whether the value of the `HCCL_BUFFSIZE` environment variable is reasonable. This environment variable indicates the memory size occupied by a single communication domain, in MB. When not configured, it defaults to 200 MB. It must be ≥ 2 and satisfy 1024 ^ 2 \* (`HCCL_BUFFSIZE` - 2) / 2 >= `BS` \* 2 * (`H` + 128) \* (`epWorldSize` \* `localExpertNum` + `K` + 1), where `localExpertNum` uses the number of experts on the current card of the MoE expert card.
    - Constraints related to `quantMode`:
        - When `quantMode` is set to 0, it indicates a non-quantization scenario, and the data type of `expandX` supports `FLOAT16` and `BFLOAT16`.
            - When the data type of `expandX` is `FLOAT16` or `BFLOAT16`, a null pointer must be passed for the input `scales`.
        - When `quantMode` is set to 2, it indicates a per-token dynamic quantization scenario, and the data type of `expandX` supports `INT8`.
            - A null pointer can be passed for the input `scales`.
            - If valid data is passed to input `scales` and shared expert devices exist, the shape is (`sharedExpertNum` + `moeExpertNum`, `H`).
            - If valid data is passed for the input `scales` and there is no shared expert device, the shape is (`moeExpertNum`, `H`).

<!-- end id14 -->

## Example

<!-- npu="910b" id15 -->
- <term>Atlas A2 products</term>:

    - File preparation:

        1. Create and modify the rank_table_m2.json file as instructed below.

        2. Copy the project to both servers, and configure the rank_table_m2.json file content based on the device IP of each machine. Ensure that the rank_table_m2.json files on both servers remain consistent.

        3. Install the CANN package and compile and run the sample by following the instructions below.

    - About rankTable:

        1. Developers can use a ranktable file to configure the NPU resource information of devices participating in collective communication. For details, see "Communication Function Development > Cluster Information Configuration > Configuring Resource Information Using the ranktable File" in [*Collective Communication User Guide*](https://hiascend.com/en/document/redirect/CannCommunityHcclUg).

        2. Use `cat /etc/hccn.conf` or `for i in seq 0 7; do echo "===================> dev$i, NPU$((i+1))"; hccn_tool -i $i -ip -g; done` to query the device IP of the machine. Then fill in the JSON file by referring to the collective communication documentation.

        > Note: In a two-machine 16-card scenario, the device_id of both machines is 0 to 7, the rank_id of one machine is 0 to 7, and the rank_id of the other machine is 8 to 15. In a single-machine 16-card scenario, both device_id and rank_id are 0 to 15.

    - Environment variable configuration:

        ```bash
        # Set the following three environment variables before running.
        ## FIRST_RANK_ID description: For example, with two machines and 16 devices in total, set the variable to 0 on one machine and 8 on the other.
        ## For example, export FIRST_RANK_ID=0
        export RANK_TABLE_FILE=/home/path/to/rank_table_m2.json
        export FIRST_RANK_ID=<starting rank_id of the device>
        ## ENV_DEV_NUM description: Set this variable based on the number of devices on the current machine. For example, with two machines and 16 devices in total, set it to 16 on both machines.
        export ENV_DEV_NUM=16
        ```

    - Number of machines:
        In a two-machine 16-NPU scenario, set the MACHINE_NUM parameter to 2, that is,

        ```Cpp
        const uint32_t MACHINE_NUM = 2;
        ```

        No modification is required for a single-machine 16-NPU scenario.

    - Operator compilation and execution:
        Compile the operators on all machines. The compilation command is as follows. Both the moe_distribute_dispatch and moe_distribute_combine operators need to be compiled, and these two operators must be executed in pairs:

        ```bash
        bash build.sh --pkg --soc=ascend910b --ops=moe_distribute_dispatch,moe_distribute_combine
        ```

        Run the operator example on all machines (in a two-machine scenario, you need to execute the operator in terminals simultaneously). The execution command is as follows:

        ```bash
        bash build.sh --run_example --ops=moe_distribute_dispatch eager cust
        ```

<!-- end id15 -->
<!-- npu="A3" id16 -->
- <term>Atlas A3 products</term>  :

    - Environment variable configuration:

        ```bash
        # Set the environment variable ENV_DEV_NUM before running. There is no need to configure the ranktable file or the environment variables RANK_TABLE_FILE and FIRST_RANK_ID.
        ## ENV_DEV_NUM description: Set this variable based on the number of devices on the current machine. For example, for two machines with 16 devices each, set the variable to 16 on both machines.
        export ENV_DEV_NUM=16
        ```

<!-- end id16 -->

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

<!-- npu="950,A3,910b" id17 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>:

    ```Cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <vector>
    #include <memory>
    #include <cstdio>
    #include "acl/acl.h"
    #include "hccl/hccl.h"
    #include "aclnn/opdev/fp16_t.h"
    #include "aclnnop/aclnn_moe_distribute_dispatch.h"
    #include "aclnnop/aclnn_moe_distribute_combine.h"

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

    const uint32_t MACHINE_NUM = 1;
    const char* rank_table_file = std::getenv("RANK_TABLE_FILE");
    const char* first_rank_id = std::getenv("FIRST_RANK_ID");
    const char* env_dev_num = std::getenv("ENV_DEV_NUM");

    uint32_t EP_WORLD_SIZE = 0;
    uint32_t DEV_NUM = 0;

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
            aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr
        );
        return 0;
    }

    int launchOneThreadDispatchAndCombine(Args &args)
    {
        int ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed. ret: %d\n", ret); return ret);

        char hcomEpName[128] = {0};
        ret = HcclGetCommName(args.hcclEpComm, hcomEpName);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetEpCommName failed. ret: %d\n", ret); return -1);
        LOG_PRINT(
            "[INFO] rank = %d, hcomEpName = %s, dispatchStream = %p, combineStream = %p, context = %p\n",
            args.rankId, hcomEpName, args.dispatchStream, args.combineStream, args.context
        );

        // Set the use case.
        int64_t BS = 8;
        int64_t H = 7168;
        int64_t K = 1;
        int64_t expertShardType = 0;
        int64_t sharedExpertNum = 0;
        int64_t sharedExpertRankNum = 0;
        if (!rank_table_file && !first_rank_id) {
            sharedExpertNum = 1;
            sharedExpertRankNum = 1;
        }
        if (rank_table_file && !first_rank_id) {
            sharedExpertNum = 1;
            sharedExpertRankNum = 0;
        }
        int64_t moeExpertNum = EP_WORLD_SIZE - sharedExpertRankNum;
        int64_t quantMode = 0;
        int64_t globalBS = BS * EP_WORLD_SIZE;
        int64_t expertTokenNumsType = 0;
        int64_t outDtype = 0;
        int64_t commQuantMode = 0;
        int64_t groupListType = 0;
        int64_t localExpertNum;
        int64_t A;
        if (args.epRankId < sharedExpertRankNum) {
            // Shared expert device
            localExpertNum = 1;
            A = globalBS / sharedExpertRankNum;
        } else {
            // MoE expert device
            localExpertNum = moeExpertNum / (EP_WORLD_SIZE - sharedExpertRankNum);
            A = globalBS * (localExpertNum < K ? localExpertNum : K);
        }

        /* Construct the input and output variables on the device based on the current use case. */
        // Declare the input and output variables on the device.
        void *xDeviceAddr = nullptr;
        void *expertIdsDeviceAddr = nullptr;
        void *scalesDeviceAddr = nullptr;
        void *expertScalesDeviceAddr = nullptr;
        void *expandXDeviceAddr = nullptr;
        void *dynamicScalesDeviceAddr = nullptr;
        void *expandIdxDeviceAddr = nullptr;
        void *expertTokenNumsDeviceAddr = nullptr;
        void *epRecvCountsDeviceAddr = nullptr;
        void *tpRecvCountsDeviceAddr = nullptr;
        void *expandScalesDeviceAddr = nullptr;

        aclTensor *x = nullptr;
        aclTensor *expertIds = nullptr;
        aclTensor *scales = nullptr;
        aclTensor *expertScales = nullptr;
        aclTensor *expandX = nullptr;
        aclTensor *dynamicScales = nullptr;
        aclTensor *expandIdx = nullptr;
        aclTensor *expertTokenNums = nullptr;
        aclTensor *epRecvCounts = nullptr;
        aclTensor *tpRecvCounts = nullptr;
        aclTensor *expandScales = nullptr;

        // Define the dimensions of each variable in the current use case.
        std::vector<int64_t> xShape{BS, H};
        std::vector<int64_t> expertIdsShape{BS, K};
        std::vector<int64_t> scalesShape{(sharedExpertRankNum > 0) ? 1 + moeExpertNum : moeExpertNum, H};
        std::vector<int64_t> expertScalesShape{BS, K};
        std::vector<int64_t> expandXShape{A, H};
        std::vector<int64_t> dynamicScalesShape{A};
        std::vector<int64_t> expandIdxShape{BS * K};
        std::vector<int64_t> expertTokenNumsShape{localExpertNum};
        std::vector<int64_t> epRecvCountsShape{localExpertNum * EP_WORLD_SIZE};
        std::vector<int64_t> tpRecvCountsShape{1};
        std::vector<int64_t> expandScalesShape{A};

        long long xShapeSize = GetShapeSize(xShape);
        long long expertIdsShapeSize = GetShapeSize(expertIdsShape);
        long long scalesShapeSize = GetShapeSize(scalesShape);
        long long expertScalesShapeSize = GetShapeSize(expertScalesShape);
        long long expandXShapeSize = GetShapeSize(expandXShape);
        long long dynamicScalesShapeSize = GetShapeSize(dynamicScalesShape);
        long long expandIdxShapeSize = GetShapeSize(expandIdxShape);
        long long expertTokenNumsShapeSize = GetShapeSize(expertTokenNumsShape);
        long long epRecvCountsShapeSize = GetShapeSize(epRecvCountsShape);
        long long tpRecvCountsShapeSize = GetShapeSize(tpRecvCountsShape);
        long long expandScalesShapeSize = GetShapeSize(expandScalesShape);

        // Construct host-side variables.
        std::vector<op::fp16_t> xHostData(xShapeSize, 1);
        std::vector<int32_t> expertIdsHostData;
        for (int32_t token_id = 0; token_id < expertIdsShape[0]; token_id++) {
            // Each token is sent to MoE experts {0, 1, ... k - 1}.
            for (int32_t k_id = 0; k_id < expertIdsShape[1]; k_id++) {
                expertIdsHostData.push_back(k_id);
            }
        }
        std::vector<float> scalesHostData(scalesShapeSize, 0);
        std::vector<float> expertScalesHostData(expertScalesShapeSize, 0);
        std::vector<op::fp16_t> expandXHostData(expandXShapeSize, 0);
        std::vector<float> dynamicScalesHostData(dynamicScalesShapeSize, 0);
        std::vector<int32_t> expandIdxHostData(expandIdxShapeSize, 0);
        std::vector<int64_t> expertTokenNumsHostData(expertTokenNumsShapeSize, 0);
        std::vector<int32_t> epRecvCountsHostData(epRecvCountsShapeSize, 0);
        std::vector<int32_t> tpRecvCountsHostData(tpRecvCountsShapeSize, 0);
        std::vector<float> expandScalesHostData(expandScalesShapeSize, 0);

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
        ret = CreateAclTensor(tpRecvCountsHostData, tpRecvCountsShape, &tpRecvCountsDeviceAddr, aclDataType::ACL_INT32, &tpRecvCounts);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expandScalesHostData, expandScalesShape, &expandScalesDeviceAddr, aclDataType::ACL_FLOAT, &expandScales);
        CHECK_RET(ret == ACL_SUCCESS, return ret);

        /* Declare variables required for operator execution. */
        uint64_t dispatchWorkspaceSize = 0;
        aclOpExecutor *dispatchExecutor = nullptr;
        void *dispatchWorkspaceAddr = nullptr;

        uint64_t combineWorkspaceSize = 0;
        aclOpExecutor *combineExecutor = nullptr;
        void *combineWorkspaceAddr = nullptr;

        /* Execute the dispatch and combine operators in sequence. */
        // Call the first-phase API of the dispatch operator.
        ret = aclnnMoeDistributeDispatchGetWorkspaceSize(
            x, expertIds,
            (quantMode > 0 ? scales : nullptr), nullptr,
            expertScales,
            hcomEpName, EP_WORLD_SIZE, args.epRankId,
            moeExpertNum, "", 0,
            0, expertShardType, sharedExpertNum,
            sharedExpertRankNum, quantMode, globalBS,
            expertTokenNumsType,
            expandX, dynamicScales,
            expandIdx, expertTokenNums,
            epRecvCounts, tpRecvCounts,
            expandScales, &dispatchWorkspaceSize,
            &dispatchExecutor
        );
        CHECK_RET(
            ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchGetWorkspaceSize failed. ret = %d\n", ret); return ret
        );
        // Allocate device memory based on the workspaceSize calculated by the first-phase API of the dispatch operator.
        if (dispatchWorkspaceSize > 0) {
            ret = aclrtMalloc(&dispatchWorkspaceAddr, dispatchWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret = %d\n", ret); return ret);
        }
        // Call the second-phase API of the dispatch operator.
        ret = aclnnMoeDistributeDispatch(dispatchWorkspaceAddr, dispatchWorkspaceSize, dispatchExecutor, args.dispatchStream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeDispatch failed. ret = %d\n", ret); return ret);
        // (Boilerplate) Wait for the task execution to complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.dispatchStream, 10000);
        CHECK_RET(
            ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d\n", ret); return ret
        );

        // Call the first-phase API of the combine operator.
        ret = aclnnMoeDistributeCombineGetWorkspaceSize(expandX, expertIds, expandIdx, epRecvCounts, expertScales, tpRecvCounts,
            nullptr, nullptr, nullptr, nullptr, nullptr,
            hcomEpName, EP_WORLD_SIZE, args.epRankId, moeExpertNum, "", 0, 0,
            expertShardType, sharedExpertNum, sharedExpertRankNum, globalBS, outDtype, commQuantMode, groupListType,
            x, &combineWorkspaceSize, &combineExecutor);
        CHECK_RET(
            ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnMoeDistributeCombineGetWorkspaceSize failed. ret = %d\n", ret); return ret
        );
        // Allocate device memory based on the workspaceSize calculated by the first-phase API of the combine operator
        if (combineWorkspaceSize > 0) {
            ret = aclrtMalloc(&combineWorkspaceAddr, combineWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret = %d\n", ret); return ret);
        }
        // Call the second-phase API of the combine operator
        ret = aclnnMoeDistributeCombine(combineWorkspaceAddr, combineWorkspaceSize, combineExecutor, args.combineStream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeCombine failed. ret = %d\n", ret); return ret);
        // (boilerplate) Synchronize and wait until the task is complete
        ret = aclrtSynchronizeStreamWithTimeout(args.combineStream, 10000);
        CHECK_RET(
            ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d\n", ret); return ret
        );

        LOG_PRINT("[INFO] device_%d aclnnMoeDistributeDispatch and aclnnMoeDistributeCombine execute successfully.\n", args.rankId);

        // Release device resources
        if (dispatchWorkspaceSize > 0) {
            aclrtFree(dispatchWorkspaceAddr);
        }
        if (combineWorkspaceSize > 0) {
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
        if (tpRecvCounts != nullptr) {
            aclDestroyTensor(tpRecvCounts);
        }
        if (expandScales != nullptr) {
            aclDestroyTensor(expandScales);
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
        if (tpRecvCountsDeviceAddr != nullptr) {
            aclrtFree(tpRecvCountsDeviceAddr);
        }
        if (expandScalesDeviceAddr != nullptr) {
            aclrtFree(expandScalesDeviceAddr);
        }

        HcclCommDestroy(args.hcclEpComm);
        aclrtDestroyStream(args.dispatchStream);
        aclrtDestroyStream(args.combineStream);
        aclrtDestroyContext(args.context);
        aclrtResetDevice(args.rankId);

        return 0;
    }

    int run_example_on_A2(int rankId, const char* RANK_TABLE_FILE, const char* FIRST_RANK_ID)
    {
        Args args;
        aclrtStream dispatchStream;
        aclrtStream combineStream;
        aclrtContext context;

        int ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d", ret));
        ret = aclrtCreateContext(&context, rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d", ret));
        ret = aclrtCreateStream(&dispatchStream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d", ret));
        ret = aclrtCreateStream(&combineStream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d", ret));

        int first_rank_id = std::stoi(std::string(FIRST_RANK_ID));
        HcclComm hcclComm = nullptr;
        int rank_id = rankId + first_rank_id;
        ret = HcclCommInitClusterInfo(RANK_TABLE_FILE, rank_id, &hcclComm);
        if (ret != HCCL_SUCCESS) {
            std::cout << "[ERROR] HCCL CommInitClusterInfo failed. ret = " << ret << std::endl;
            return ret;
        }
        std::cout << "[INFO] HcclCommInitClusterInfo success, rank_id:" << rank_id << ", rankSize:" << DEV_NUM
                << ", hcclComm:" << hcclComm << std::endl;
        uint32_t epRankId = rank_id;

        args.rankId = rankId;
        args.epRankId = epRankId;
        args.hcclEpComm = hcclComm;
        args.dispatchStream = dispatchStream;
        args.combineStream = combineStream;
        args.context = context;

        launchOneThreadDispatchAndCombine(args);
        return 0;
    }

    int run_example_on_A3A5()
    {
        int ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d\n", ret); return ret);

        aclrtStream dispatchStream[DEV_NUM];
        aclrtStream combineStream[DEV_NUM];
        aclrtContext context[DEV_NUM];
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            ret = aclrtSetDevice(rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d\n", ret); return ret);
            ret = aclrtCreateContext(&context[rankId], rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed. ret = %d\n", ret); return ret);
            ret = aclrtCreateStream(&dispatchStream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d\n", ret); return ret);
            ret = aclrtCreateStream(&combineStream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d\n", ret); return ret);
        }

        int32_t devicesEp[EP_WORLD_SIZE];
        for (int32_t epId = 0; epId < EP_WORLD_SIZE; epId++) {
            devicesEp[epId] = epId;
        }
        // Initialize the EP communicator.
        HcclComm commsEp[EP_WORLD_SIZE];
        ret = HcclCommInitAll(EP_WORLD_SIZE, devicesEp, commsEp);
        CHECK_RET(
            ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] HcclCommInitAll ep failed. ret = %d\n", ret); return ret
        );

        Args args[DEV_NUM];
        // Each thread calls each device to execute the operator
        std::vector<std::unique_ptr<std::thread>> threads(DEV_NUM);
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            uint32_t epRankId = rankId;

            args[rankId].rankId = rankId;
            args[rankId].epRankId = epRankId;
            args[rankId].hcclEpComm = commsEp[epRankId];
            args[rankId].dispatchStream = dispatchStream[rankId];
            args[rankId].combineStream = combineStream[rankId];
            args[rankId].context = context[rankId];
            threads[rankId].reset(new (std::nothrow) std::thread(&launchOneThreadDispatchAndCombine, std::ref(args[rankId])));
        }
        for (uint32_t rankId = 0; rankId < DEV_NUM; rankId++) {
            threads[rankId]->join();
        }
        aclFinalize();
        LOG_PRINT("[INFO] aclFinalize success\n");
        return 0;
    }


    int main(int argc, char *argv[])
    {
        const char* env_var_name = "RANK_TABLE_FILE and FIRST_RANK_ID";
        if (!env_dev_num) {
            LOG_PRINT("[WARNING] Please check whether environment variable ENV_DEV_NUM is set correctly.\n");
            return 0;
        }
        int actual_env_dev_num = std::stoi(std::string(env_dev_num));
        if (actual_env_dev_num < DEV_NUM) {
            LOG_PRINT("[INFO] ENV_DEV_NUM = %d is less than %d, currently not supported\n", actual_env_dev_num, DEV_NUM);
            return 0;
        }
        if (!rank_table_file && !first_rank_id) {
            LOG_PRINT("[INFO] %s are not identified and example on <Atlas A3> will be executed!\n", env_var_name);
            EP_WORLD_SIZE = 8;
            DEV_NUM = EP_WORLD_SIZE;
            int ret = run_example_on_A3A5();
        }
        else if (rank_table_file && !first_rank_id) {
            LOG_PRINT("[INFO] %s are not identified and example on <Atlas A5> will be executed!\n", env_var_name);
            EP_WORLD_SIZE = 2;
            DEV_NUM = 2;
            int ret = run_example_on_A3A5();
        }
        else if (rank_table_file && first_rank_id) {
            LOG_PRINT("[INFO] %s are identified and example on <Atlas A2> will be executed!\n", env_var_name);
            EP_WORLD_SIZE = 16;
            DEV_NUM = EP_WORLD_SIZE;
            uint32_t single_machine_dev_num = EP_WORLD_SIZE / MACHINE_NUM;
            std::vector<std::unique_ptr<std::thread>> threads(single_machine_dev_num);
            int ret = aclInit(nullptr);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d\n", ret); return ret);
            for (int rankId = 0; rankId < single_machine_dev_num; ++rankId) {
                threads[rankId] = std::make_unique<std::thread>([rankId,&ret]()
                {
                    int ret = run_example_on_A2(rankId, rank_table_file, first_rank_id);
                });
            }
            for (int rankId = 0; rankId < single_machine_dev_num; ++rankId) {
                threads[rankId]->join();
            }
            aclFinalize();
            LOG_PRINT("[INFO] aclFinalize success\n");
        }
        else {
            LOG_PRINT("[WARNING] Please check whether %s are set correctly.\n", env_var_name);
        }

        return 0;
    }
    ```

<!-- end id17 -->
