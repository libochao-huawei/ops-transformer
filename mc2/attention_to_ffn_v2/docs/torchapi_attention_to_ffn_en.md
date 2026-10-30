# attention_to_ffn

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T13:12:07.292Z -->

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

- **Description**:

  - The attention_to_ffn operator sends token data on the Attention node to the FFN node. It is used for forward data communication when Attention and FFN are deployed separately in MoE (Mixture of Experts) use cases. Based on the expert index selected by each token and the mapping table from experts to FFN devices, the operator routes token data to the target FFN device. It supports optional quantized transmission to reduce communication bandwidth overhead.
  - This operator provides APIs such as attention_to_ffn and get_buffer_for_attention_to_ffn, which are used together.
  - get_buffer_for_attention_to_ffn: Encapsulates input parameters and creates a communication context (context). It generates information required for running the attention_to_ffn operator, such as `context` and `ccl_buffer_size`, and returns a buffer object.

- **Formula**:

  - Input:
    - $\mathbf{X} \in \mathbb{R}^{\text{X} \times \text{BS} \times \text{H}}$: Token data matrix on the Attention node, corresponding to input parameter `x`. $\text{X}$ is the micro batch sequence size, $\text{BS}$ is the batch sequence size, $\text{H}$ is the hidden dimension.
    - $\mathbf{E} \in \mathbb{Z}^{\text{X} \times \text{BS} \times \text{K}}$: Index matrix of the topK experts selected by each token, corresponding to input parameter `expert_ids`. $\text{K}$ is the quantity of experts selected by each token.
    - $\mathbf{R} \in \mathbb{Z}^{\text{L} \times (\text{moeExpertNum} + \text{sharedExpertNum}) \times \text{M}}$: Deployment mapping table from experts to FFN devices, corresponding to input parameter `expert_rank_table`. $\text{L}$ is the number of model layers, $\text{M}$ is the length of the last dimension of the mapping table.
    - $\text{sessionId}$: Id of the current Attention Worker node, corresponding to input parameter `session_id`.
    - $\text{microBatchId}$: Id of the current micro batch, corresponding to input parameter `micro_batch_id`.
    - $\text{layerId}$: Id of the current model layer, corresponding to input parameter `layer_id`.
  - Output:
    - No host-visible output. Data is sent to the target FFN device through the HCCL window, and the FFN device reads token data from its CCL window.
  - Conventions:
    - $\text{expertNumPerToken} = \text{K} + \text{sharedExpertNum}$, indicating the total number of experts (including shared experts) corresponding to each token.
    - $\text{attnWorkerNum}$: quantity of Attention Workers.
    - $\text{ffnWorkerNum}$: quantity of FFN Workers.
    - $\text{ffnStartRankId}$: starting Rank Id of FFN nodes.

- Computation description:

    **Data routing**

    For each token $\text{token}_i$ on the Attention Worker ($i \in \{0, 1, \dots, \text{BS}-1\}$), based on its selected $k$-th expert $e_{i,k} = \mathbf{E}[\text{microBatchId}, i, k]$ ($k \in \{0, \dots, \text{K}-1\}$), find the FFN device $\text{toRankId}_{i,k}$ where the expert resides through the expert-to-FFN-device mapping table $\mathbf{R}$.

    <details>
    <summary> Non-quantization scenario (quantMode=0)</summary>

    In the non-quantization scenario, token data is transmitted directly at the original precision (FP16/BF16). For each token $\text{token}_i$ and its selected $k$-th expert, write the token data $\mathbf{x}_i$ to the corresponding token data area in the CCL window of the target FFN device $\text{toRankId}_{i,k}$, and write the metadata (sessionId, microBatchId, tokenId, expertOffset, etc.) to the information table area of the FFN device.

    $$\text{sendData}_{i,k} = \mathbf{x}_i \quad \in \mathbb{R}^{1 \times \text{H}}$$

    </details>

    <details>
    <summary> PERTOKEN+INT8 quantization scenario (quantMode=2)</summary>

    In the PERTOKEN quantization scenario, a scaling factor is dynamically computed for the hidden state vector of each token. The vector is then quantized to INT8 and transmitted.

    For token $\text{token}_i$, compute its per-token scaling factor:

    $$s^{X}_i = \frac{\max(|\mathbf{X}[\text{microBatchId}, i, :]|)}{127} \in \mathbb{R}$$

    Then quantize to obtain the INT8 representation:

    $$\mathbf{q}_i = \left\lfloor \frac{\mathbf{X}[\text{microBatchId}, i, :]}{s^{X}_i} \right\rceil \quad \in \left(\mathbb{Z}_8^{\text{sym}}\right)^{1 \times \text{H}}$$

    Send the quantized data $\mathbf{q}_i$ and the scaling factor $s^{X}_i$ together to the target FFN device.

    </details>

    <details>
    <summary> MXFP quantization scenario (quantMode=3/4/5)</summary>

    In the MX (Microscaling) quantization scenario, token data is divided into groups of 32 consecutive elements. A shared exponent scaling factor (E8M0 format) is computed for each group, and the data is quantized to the target low-precision type.

    For the hidden state vector $\mathbf{x}_i \in \mathbb{R}^{1 \times \text{H}}$ of token $\text{token}_i$, group the elements with group size = 32:

    $$\text{shared\_exp}_g = \text{floor}(\log_2(\max(|\mathbf{x}_{i,g}|))) - \text{emax}, \quad g \in \{0, 1, \dots, \lceil \text{H}/32 \rceil - 1\}$$

    $$s^{X}_{i,g} = 2^{\text{shared\_exp}_g}$$

    where $\text{emax}$ is the maximum exponent value of the target data type (7 for FP8_E5M2, 3 for FP8_E4M3, and 1 for FP4_E2M1), and $\mathbf{x}_{i,g}$ indicates the 32 elements in group $g$.

    The quantization result is:

    $$\mathbf{q}_{i,g} = \text{Cast}\left(\frac{\mathbf{x}_{i,g}}{s^{X}_{i,g}}, \text{dstType}\right)$$

    Output data type for each quantMode:

    | quantMode | Output Data Type | Description |
    | :-------: | :----------: | :---: |
    | 3 | FLOAT8_E5M2 | MX quantization, output FP8_E5M2. |
    | 4 | FLOAT8_E4M3 | MX quantization, output FP8_E4M3. |
    | 5 | FLOAT4_E2M1 | MX quantization, output FP4_E2M1. |

    </details>

    <details>
    <summary> MX_CLIP quantization scenario (quantMode=6/7)</summary>

    The MX_CLIP mode introduces 1e-4 lower-limit clipping based on MX quantization to prevent precision loss caused by excessively small scale factors.

    $$\text{max\_abs}_g = \max(|\mathbf{x}_{i,g}|)$$

    $$\text{max\_abs\_clamp}_g = \max(\text{max\_abs}_g, 10^{-4})$$

    $$\text{shared\_exp}_g = \text{ceil}(\log_2(\text{max\_abs\_clamp}_g / \text{fp8\_max}))$$

    $$s^{X}_{i,g} = 2^{\text{shared\_exp}_g}$$

    $$\mathbf{q}_{i,g} = \text{Cast}\left(\frac{\mathbf{x}_{i,g}}{s^{X}_{i,g}}, \text{dstType}\right)$$

    Output data types corresponding to each quantMode:

    | quantMode | Output Data Type | Description |
    | :-------: | :----------: | :---: |
    | 6 | FLOAT8_E5M2 | MX_CLIP quantization, output FP8_E5M2. |
    | 7 | FLOAT8_E4M3 | MX_CLIP quantization, output FP8_E4M3. |

    > In MX_CLIP mode, only FP8 output is supported; FP4 is not supported.

    </details>

## Function Prototype

Use the get_buffer_for_attention_to_ffn API to encapsulate input parameters and create a communication context (buffer). Then, call the attention_to_ffn API to send data.

```python
get_buffer_for_attention_to_ffn(group, world_size, ffn_token_info_table_shape, ffn_token_data_shape, *, quant_mode=0) -> AttentionToFfnBuffer
```

```python
attention_to_ffn(buffer, x, session_id, micro_batch_id, layer_id, expert_ids, expert_rank_table, attn_token_info_table_shape, moe_expert_num, *, sync_flag=0, ffn_start_rank_id=0, scales=None, active_mask=None) -> None
```

## Mandatory

### get_buffer_for_attention_to_ffn

<table style="undefined;table-layout: fixed; width:840px"><colgroup>
<col style="width: 180px">
<col style="width: 140px">
<col style="width: 80px">
<col style="width: 440px">
</colgroup>
<thead>
<tr>
    <th>Parameter</th>
    <th>Parameter Type</th>
    <th>Optional/Required</th>
    <th>Description</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>group</td>
        <td>torch.distributed.ProcessGroup</td>
        <td>required</td>
        <td>ProcessGroup object of the EP communicator.</td>
    </tr>
    <tr>
        <td>world_size</td>
        <td>int</td>
        <td>Required</td>
        <td>Size of the communicator.</td>
    </tr>
    <tr>
        <td>ffn_token_info_table_shape</td>
        <td>List[int]</td>
        <td>Required</td>
        <td>Shape of the token info table on the FFN node. The length is 3, and the format is <code>[attnWorkerNum, X, infoTableLastDim]</code>.</td>
    </tr>
    <tr>
        <td>ffn_token_data_shape</td>
        <td>List[int]</td>
        <td>Required</td>
        <td>Shape of the token data table on the FFN node. The length is 5, and the format is <code>[attnWorkerNum, microBatchNum, BS, K+sharedExpertNum, HS]</code>.</td>
    </tr>
    <tr>
        <td>quant_mode</td>
        <td>int</td>
        <td>Optional</td>
        <td>Quantization mode. 0 indicates non-quantization, 2 indicates PERTOKEN+INT8, 3 indicates MX+FP8_E5M2, 4 indicates MX+FP8_E4M3, 5 indicates MX+FP4_E2M1, 6 indicates MX_CLIP+FP8_E5M2, and 7 indicates MX_CLIP+FP8_E4M3. The default value is 0.</td>
    </tr>
</tbody>
</table>

This API returns an <code>AttentionToFfnBuffer</code> object. The CCL communication buffer size is automatically calculated and a communication context is created internally for use by <code>attention_to_ffn</code>.

### attention_to_ffn

<table style="undefined;table-layout: fixed; width:1400px"><colgroup>
<col style="width: 120px">
<col style="width: 120px">
<col style="width: 90px">
<col style="width: 320px">
<col style="width: 160px">
<col style="width: 120px">
<col style="width: 260px">
</colgroup>
<thead>
<tr>
    <th>Parameter</th>
    <th>Type</th>
    <th>Optional/Required</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Data Format</th>
    <th>Dimension (shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>buffer</td>
        <td>AttentionToFfnBuffer</td>
        <td>Required</td>
        <td>Communication buffer created by <a href="#get_buffer_for_attention_to_ffn">get_buffer_for_attention_to_ffn</a>, which internally encapsulates <code>context</code>, <code>group</code>, <code>world_size</code>, <code>ffn_token_info_table_shape</code>, <code>ffn_token_data_shape</code>, <code>quant_mode</code>, and <code>ccl_buffer_size</code>.</td>
        <td>int32</td>
        <td>ND</td>
        <td>(contextSize,)</td>
    </tr>
    <tr>
        <td>x</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Token data sent by the current device.</td>
        <td>float16, bfloat16</td>
        <td>ND</td>
        <td>(X, BS, H)</td>
    </tr>
    <tr>
        <td>session_id</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>The ID of the current Attention Worker node.</td>
        <td>int32</td>
        <td>ND</td>
        <td>(X,)</td>
    </tr>
    <tr>
        <td>micro_batch_id</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>ID of the current micro batch.</td>
        <td>int32</td>
        <td>ND</td>
        <td>(X,)</td>
    </tr>
    <tr>
        <td>layer_id</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>ID of the current model layer.</td>
        <td>int32</td>
        <td>ND</td>
        <td>(X,)</td>
    </tr>
    <tr>
        <td>expert_ids</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Indices of topK experts for each token. The value range of each element is <code>[0, moe_expert_num)</code>.</td>
        <td>int32</td>
        <td>ND</td>
        <td>(X, BS, K)</td>
    </tr>
    <tr>
        <td>expert_rank_table</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Mapping table from expert IDs to FFN device deployment (the caller must ensure the values are correct).</td>
        <td>int32</td>
        <td>ND</td>
        <td>(L, moeExpertNum + sharedExpertNum, M)</td>
    </tr>
    <tr>
        <td>attn_token_info_table_shape</td>
        <td>List[int]</td>
        <td>Required</td>
        <td>Shape of the token information table on the Attention node. The length is 3.</td>
        <td>int</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>moe_expert_num</td>
        <td>int</td>
        <td>Required</td>
        <td>Quantity of MoE routing experts.</td>
        <td>int</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>sync_flag</td>
        <td>int</td>
        <td>Optional</td>
        <td>Sync mode. 0 indicates asynchronous, and 1 indicates synchronous. The default value is 0.</td>
        <td>int</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>ffn_start_rank_id</td>
        <td>int</td>
        <td>Optional</td>
        <td>Start Rank Id of the FFN node. Default value is 0.</td>
        <td>int</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>scales</td>
        <td>Tensor</td>
        <td>Optional</td>
        <td>Quantization smoothing parameter of each expert. Passing in values is not supported in MX/MX_CLIP quantization modes; the value must be None. Default value is None.</td>
        <td>float32</td>
        <td>ND</td>
        <td>(L, moeExpertNum + sharedExpertNum, H)</td>
    </tr>
    <tr>
        <td>active_mask</td>
        <td>Tensor</td>
        <td>Optional</td>
        <td>Indicates whether tokens participate in the communication. Default value is None.</td>
        <td>bool</td>
        <td>ND</td>
        <td>(X, BS)</td>
    </tr>
</tbody>
</table>

## Return Values

No return value. Data is sent to the target FFN device through the HCCL window. There is no host-visible output tensor.

## Constraints

- During the operator call, the `group`, `world_size`, `ffn_token_info_table_shape`, `ffn_token_data_shape`, and `attn_token_info_table_shape` parameters and the `ccl_buffer_size` value must be consistent across all devices and also across different layers in the network. The buffer created by `get_buffer_for_attention_to_ffn` encapsulates `group`, `world_size`, `ffn_token_info_table_shape`, `ffn_token_data_shape`, `quant_mode`, and `ccl_buffer_size`. These parameters do not need to be passed separately when calling `attention_to_ffn`.

- `ccl_buffer_size` is the **total size** (Bytes) of the CCL communication buffer allocated on HBM. It is internally and automatically computed by `get_buffer_for_attention_to_ffn` and must meet the following requirements:

$$ccl\_buffer\_size \ge \mathrm{CeilAlign}(\mathrm{tokenInfoSize} + \mathrm{tokenDataSize},\ 2\,\mathrm{MB})$$

Where:

  - `tokenInfoSize = attnWorkerNum × microBatchNum × infoTableLastDim × 4B`
  - Non-quantization mode: `tokenDataSize = attnWorkerNum × microBatchNum × BS × (K+shared) × HS × 2B`
  - INT8/FP8 quantization mode: `tokenDataSize = attnWorkerNum × microBatchNum × BS × (K+shared) × HS × 1B`
  - FP4 quantization mode: `HS` is already calculated in packed bytes, and `tokenDataSize = attnWorkerNum × microBatchNum × BS × (K+shared) × HS × 1B`

This size is internally and automatically computed by `get_buffer_for_attention_to_ffn`. You do not need to calculate or set `ccl_buffer_size` yourself.

- Quantization mode description:

| quantMode | Algorithm | Output Type | Description |
| :--------: | :--: | :------: | :---: |
| 0 | Non-quantized | Follows the input | FP16/BF16 data is transmitted directly. |
| 2 | PERTOKEN | INT8 | The scale is dynamically computed for each token and quantized to INT8. |
| 3 | MX | FP8_E5M2 | The MX scale (E8M0 format) is computed for each group of 32 elements, and the output is FP8_E5M2. |
| 4 | MX | FP8_E4M3 | The MX scale (E8M0 format) is computed for each group of 32 elements, and the output is FP8_E4M3. |
| 5 | MX | FP4_E2M1 | The MX scale (E8M0 format) is computed for each group of 32 elements, and the output is FP4_E2M1. |
| 6 | MX_CLIP | FP8_E5M2 | The MX scale (with 1e-4 lower limit clipping) is computed for each group of 32 elements, and the output is FP8_E5M2. |
| 7 | MX_CLIP | FP8_E4M3 | The MX scale (with 1e-4 lower limit clipping) is computed for each group of 32 elements, and the output is FP8_E4M3. |

- MX/MX_CLIP quantization constraints:
  - The scales parameter is not supported in MX mode (quantMode 3/4/5) or MX_CLIP mode (quantMode 6/7) and must be None.
  - For FP4 output (quantMode=5), the H dimension must be even.
  - MX_CLIP mode (quantMode 6/7) supports only FP8 output (E5M2/E4M3), not FP4.

- Description of the shape format in Parameters:
    - `X`: micro batch sequence size (number of token groups). The current version supports only `X` = 1.
    - `BS`: batch sequence size (number of tokens finally output on the current device). The value range is 0 < `BS` ≤ 512.
    - `K`: number of topK experts to select. The value range is 0 < `K` ≤ 16 and 0 < `K` ≤ moeExpertNum.
    - `H`: hidden size (hidden layer size). The value range is 1024 ≤ `H` ≤ 8192.
    - `L`: number of model layers. The current version supports only `L` = 1.
    - `M`: Indicates the length of the last dimension of expertRankTable, that is, the length of the deployment information list of the expert deployed in the largest quantity on the FFN node.
    - `moeExpertNum`: Indicates the quantity of MoE experts. The value range is 0 < `moeExpertNum` ≤ 1024.
    - `sharedExpertNum`: Indicates the quantity of shared experts. The value range is 0 ≤ `sharedExpertNum` ≤ 4.
    - `worldSize`: Communicator size. The value range is [2, 1024].

- Communication domain usage constraints:
    - No other operators are allowed in the communicator of the attention_to_ffn operator.
    - All nodes in the communicator must have the same driver version.

## Deterministic Computation

Defaults to a deterministic implementation.

## Example

- Call in single-operator mode:

  The following example shows the complete call flow of attention_to_ffn: initialize the communicator, create the communication buffer by using get_buffer_for_attention_to_ffn, and then call attention_to_ffn to send token data.

  ```python
  import os
  import torch
  import torch_npu
  import torch.distributed as dist
  import torch.multiprocessing as mp
  from torch.multiprocessing import Process, Manager
  from cann_ops_transformer.ops import (
      attention_to_ffn,
      get_buffer_for_attention_to_ffn,
  )

  WORLD_SIZE = 8
  FFN_WORKER_NUM = 5
  ATTENTION_WORKER_NUM = WORLD_SIZE - FFN_WORKER_NUM
  X = 1
  BS = 8
  H = 7168
  K = 4
  sharedExpertNum = 1
  moeExpertNum = 8
  expertNumPerToken = K + sharedExpertNum
  quantMode = 0
  syncFlag = 0
  ffnStartRankId = 0


  def set_device(rank):
      torch_npu.npu.set_device(rank % (WORLD_SIZE // 2))


  def init_hccl_comm(rank):
      dist.init_process_group(
          backend="hccl",
          rank=rank,
          world_size=WORLD_SIZE,
          init_method="tcp://127.0.0.1:50001",
      )
      ep_group = dist.new_group(backend="hccl", ranks=list(range(WORLD_SIZE)))
      return ep_group


  def run_attention_to_ffn(rank):
      set_device(rank)
      ep_group = init_hccl_comm(rank)

      ffn_token_info_table_shape = [ATTENTION_WORKER_NUM, X, 2 + BS * expertNumPerToken]
      ffn_token_data_shape = [ATTENTION_WORKER_NUM, X, BS, expertNumPerToken, H]
      attn_token_info_table_shape = [X, BS, expertNumPerToken]

      # Step 1: Create the communication buffer.
      buffer = get_buffer_for_attention_to_ffn(
          ep_group,
          WORLD_SIZE,
          ffn_token_info_table_shape,
          ffn_token_data_shape,
          quant_mode=quantMode,
      )

      # Step 2: Construct the input data.
      x = torch.randn((X, BS, H), dtype=torch.bfloat16, device="npu")
      session_id = torch.tensor([rank - FFN_WORKER_NUM], dtype=torch.int32, device="npu")
      micro_batch_id = torch.tensor([0], dtype=torch.int32, device="npu")
      layer_id = torch.tensor([0], dtype=torch.int32, device="npu")
      expert_ids = torch.randint(0, moeExpertNum, (X, BS, K), dtype=torch.int32, device="npu")
      M = 2 * (FFN_WORKER_NUM - 1) + 1
      expert_rank_table = torch.randint(0, FFN_WORKER_NUM, (1, moeExpertNum + sharedExpertNum, M),
                                        dtype=torch.int32, device="npu")

      # Step 3: Call attention_to_ffn to send token data.
      attention_to_ffn(
          buffer,
          x,
          session_id,
          micro_batch_id,
          layer_id,
          expert_ids,
          expert_rank_table,
          attn_token_info_table_shape,
          moeExpertNum,
          sync_flag=syncFlag,
          ffn_start_rank_id=ffnStartRankId,
      )

      torch.npu.synchronize()
      buffer.destroy()
      dist.barrier()
      dist.destroy_process_group()
      print(f"[INFO] rank {rank} attention_to_ffn finished")


  if __name__ == "__main__":
      mp.set_start_method("forkserver", force=True)
      proc_list = []
      for rank in range(WORLD_SIZE):
          proc = Process(target=run_attention_to_ffn, args=(rank,))
          proc.start()
          proc_list.append(proc)
      for proc in proc_list:
          proc.join()
  ```
