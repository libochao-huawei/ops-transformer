# ffn_to_attention

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T13:26:13.206Z -->

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

  - The ffn_to_attention operator sends token data on the FFN node to the Attention node. It is used for backward data communication when Attention and FFN are deployed separately in MoE (Mixture of Experts) use cases. Based on the index of the Attention Worker belonging to each token, the operator routes token data back to the target Attention device after the FFN calculation is complete. This completes the data transmission from FFN to Attention.
  - This operator provides APIs such as ffn_to_attention and get_buffer_for_ffn_to_attention for use together.
  - get_buffer_for_ffn_to_attention: encapsulates input parameters and creates the communication context (context), generates information required for running the ffn_to_attention operator, such as `context` and `ccl_buffer_size`, and returns a buffer object.

- **Formula**:

  - Input:
    - $\mathbf{X} \in \mathbb{R}^{\text{Y} \times \text{H}}$: token data matrix on the FFN node, corresponding to the input parameter `x`. $\text{Y}$ is the maximum quantity of tokens that the local device needs to dispatch, and $\text{H}$ is the hidden layer size.
    - $\mathbf{S} \in \mathbb{Z}^{\text{Y}}$: index of the Attention Worker that each token belongs to, corresponding to the input parameter `session_ids`. Value range: $[0, \text{attnRankNum}-1]$.
    - $\mathbf{MB} \in \mathbb{Z}^{\text{Y}}$: microBatch index of each token, corresponding to the input parameter `micro_batch_ids`. Value range: $[0, \text{microBatchNum}-1]$.
    - $\mathbf{T} \in \mathbb{Z}^{\text{Y}}$: token index of each token in the microBatch, corresponding to the input parameter `token_ids`. Value range: $[0, \text{BS}-1]$.
    - $\mathbf{EO} \in \mathbb{Z}^{\text{Y}}$: offset of each token in the expert dimension, corresponding to the input parameter `expert_offsets`. Value range: $[0, \text{expertNumPerToken}-1]$.
    - $\text{N}$: actual total quantity of tokens in the local card transmission, corresponding to the input parameter `actual_token_num`. Value range: $[0, \text{Y}]$.
  - Output:
    - No host-visible output. Data is sent to the target Attention device through the HCCL window, and the Attention device reads token data from its CCL window.
  - Conventions:
    - $\text{attnRankNum}$: Quantity of Attention Workers.
    - $\text{ffnRankNum}$: Quantity of FFN Workers.
    - $\text{microBatchNum}$: Quantity of micro batches.
    - $\text{expertNumPerToken}$: Total number of experts for each token (including shared experts).
    - $\text{HS}$: Hidden layer size of the token data table (including scale storage space), satisfying $\text{HS} \ge \text{H}$.

- Calculation description:

    **Data routing**

    For each token $\text{token}_i$ on the FFN Worker ($i \in \{0, 1, \dots, \text{N}-1\}$), the token data $\mathbf{x}_i$ is routed to the target Attention device based on the index $\mathbf{S}[i]$ of the Attention Worker it belongs to.

    The Rank Id of the target Attention device is determined as follows:
    - If `attn_rank_table` is provided: $\text{toRankId}_i = \text{attnRankTable}[\mathbf{S}[i]]$
    - If `attn_rank_table` is not provided (default): $\text{toRankId}_i = \mathbf{S}[i]$, that is, the Attention Worker index equals the Rank Id

    The token data is written to the corresponding data area in the CCL window of the target Attention device. The location is jointly determined by $\mathbf{MB}[i]$ (microBatch index), $\mathbf{T}[i]$ (token index), and $\mathbf{EO}[i]$ (expert offset).

    $$\text{sendData}_i = \mathbf{x}_i \quad \in \mathbb{R}^{1 \times \text{H}}$$

    The token data is directly transmitted at the original precision (FP16/BF16). Quantization is not supported.

## Function Prototype

Use the get_buffer_for_ffn_to_attention API to encapsulate the input parameters and create the communication context (buffer). Then call the ffn_to_attention API to send data.

```python
get_buffer_for_ffn_to_attention(group, world_size, token_info_table_shape, token_data_shape) -> FFNToAttentionBuffer
```

```python
ffn_to_attention(buffer, x, session_ids, micro_batch_ids, token_ids, expert_offsets, actual_token_num, *, attn_rank_table=None) -> None
```

## Mandatory

### get_buffer_for_ffn_to_attention

<table style="undefined;table-layout: fixed; width:840px"><colgroup>
<col style="width: 180px">
<col style="width: 140px">
<col style="width: 80px">
<col style="width: 440px">
</colgroup>
<thead>
<tr>
    <th>Parameter</th>
    <th>Type</th>
    <th>Optional/Required</th>
    <th>Description</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>group</td>
        <td>torch.distributed.ProcessGroup</td>
        <td>Required</td>
        <td>ProcessGroup object of the EP communicator.</td>
    </tr>
    <tr>
        <td>world_size</td>
        <td>int</td>
        <td>Required</td>
        <td>Communicator size.</td>
    </tr>
    <tr>
        <td>token_info_table_shape</td>
        <td>List[int]</td>
        <td>Required</td>
        <td>Shape of the token information table. The length is 3. The format is <code>[microBatchNum, BS, expertNumPerToken]</code>.</td>
    </tr>
    <tr>
        <td>token_data_shape</td>
        <td>List[int]</td>
        <td>Required</td>
        <td>Shape of the Token data table. The length is 4. The format is <code>[microBatchNum, BS, expertNumPerToken, HS]</code>.</td>
    </tr>
</tbody>
</table>

This API returns an <code>FFNToAttentionBuffer</code> object. The object internally and automatically calculates the size of the CCL communication buffer and creates the communication context for <code>ffn_to_attention</code> to use.

### ffn_to_attention

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
    <th>Shape</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>buffer</td>
        <td>FFNToAttentionBuffer</td>
        <td>Required</td>
        <td>Communication buffer created by <a href="#get_buffer_for_ffn_to_attention">get_buffer_for_ffn_to_attention</a>. It internally encapsulates <code>context</code>, <code>group</code>, <code>world_size</code>, <code>token_info_table_shape</code>, <code>token_data_shape</code>, and <code>ccl_buffer_size</code>.</td>
        <td>int32</td>
        <td>ND</td>
        <td>(contextSize,)</td>
    </tr>
    <tr>
        <td>x</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Token data for local card transmission.</td>
        <td>float16, bfloat16</td>
        <td>ND</td>
        <td>(Y, H)</td>
    </tr>
    <tr>
        <td>session_ids</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Index of the Attention Worker belonging to each token. The element value range is <code>[0, attnRankNum-1]</code>.</td>
        <td>int32</td>
        <td>ND</td>
        <td>(Y,)</td>
    </tr>
    <tr>
        <td>micro_batch_ids</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Index of the microBatch that each token belongs to. The element value range is <code>[0, microBatchNum-1]</code>.</td>
        <td>int32</td>
        <td>ND</td>
        <td>(Y,)</td>
    </tr>
    <tr>
        <td>token_ids</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Token index of each token in the microBatch. The element value range is <code>[0, BS-1]</code>.</td>
        <td>int32</td>
        <td>ND</td>
        <td>(Y,)</td>
    </tr>
    <tr>
        <td>expert_offsets</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Offset of each token in the expert dimension. Element value range is <code>[0, expertNumPerToken-1]</code>.</td>
        <td>int32</td>
        <td>ND</td>
        <td>(Y,)</td>
    </tr>
    <tr>
        <td>actual_token_num</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Actual total number of tokens sent by local card transmission. The value range is <code>[0, Y]</code>.</td>
        <td>int64</td>
        <td>ND</td>
        <td>(1,)</td>
    </tr>
    <tr>
        <td>attn_rank_table</td>
        <td>Tensor</td>
        <td>Optional</td>
        <td>Maps the device ID corresponding to each Attention Worker. If it is None, the default policy is used: the ID of each device is used as the ID of the corresponding Attention Worker. Default value: None.</td>
        <td>int32</td>
        <td>ND</td>
        <td>(attnRankNum,)</td>
    </tr>
</tbody>
</table>

## Return Values

No return value. Data is sent to the target Attention device through the HCCL window. There is no host-visible output tensor.

## Constraints

- The `group`, `world_size`, `token_info_table_shape`, `token_data_shape` parameters and the `ccl_buffer_size` value used during the operator call must be the same on all devices and across different layers of the network. `group`, `world_size`, `token_info_table_shape`, `token_data_shape`, and `ccl_buffer_size` are encapsulated in the buffer created by `get_buffer_for_ffn_to_attention`. You do not need to pass them separately when calling `ffn_to_attention`.

- `ccl_buffer_size` is the **total size** (Bytes) of the CCL communication buffer allocated on HBM. It is automatically calculated internally by `get_buffer_for_ffn_to_attention` and must satisfy:

$$ccl\_buffer\_size \ge \mathrm{CeilAlign}(\mathrm{tokenInfoSize} + \mathrm{tokenDataSize},\ 2\,\mathrm{MB})$$

Where:

  - `tokenInfoSize = microBatchNum × BS × expertNumPerToken × 4B`
  - `tokenDataSize = microBatchNum × BS × expertNumPerToken × HS × 2B`

The size is automatically calculated internally by `get_buffer_for_ffn_to_attention`. You do not need to calculate or set `ccl_buffer_size` yourself.

- Description of the shape format in Parameters:
    - `Y`: indicates the maximum quantity of tokens to be dispatched on the local device.
    - `BS`: indicates the token quantity on each Attention node. The value range is 0 < `BS` ≤ 512.
    - `H`: indicates the hidden size (hidden layer size). The value range is 1024 ≤ `H` ≤ 8192.
    - `HS`: Indicates the hidden layer size of hidden and scale. The value range is 1152 ≤ `HS` ≤ 8320, satisfying `HS` ≥ `H`.
    - `microBatchNum`: Indicates the quantity of micro batches.
    - `expertNumPerToken`: Indicates the total quantity of experts corresponding to each token (including shared experts).
    - `attnRankNum`: Indicates the quantity of Attention Workers. The value range is 0 < `attnRankNum` < `worldSize`.
    - `ffnRankNum`: Indicates the quantity of FFN Workers. The value range is 0 < `ffnRankNum` < `worldSize`.
    - `sharedExpertNum`: Indicates the quantity of shared experts. The value range is 0 ≤ `sharedExpertNum` ≤ 4.
    - `worldSize`: Size of the communicator. The value range is [2, 768].

- Communication domain usage constraints:
    - The communicator of the ffn_to_attention operator must not contain other operators.
    - The driver versions of all nodes in the communicator must be the same.

## Deterministic Computation

Defaults to a deterministic implementation.

## Example

- Call in single-operator mode:

  The following example shows the complete call flow of ffn_to_attention: first initialize the communicator, then create the communication buffer with get_buffer_for_ffn_to_attention, and finally call ffn_to_attention to send token data.

  ```python
  import os
  import torch
  import torch_npu
  import torch.distributed as dist
  import torch.multiprocessing as mp
  from torch.multiprocessing import Process
  from cann_ops_transformer.ops import (
      ffn_to_attention,
      get_buffer_for_ffn_to_attention,
  )

  WORLD_SIZE = 8
  ATTENTION_WORKER_NUM = 3
  FFN_WORKER_NUM = WORLD_SIZE - ATTENTION_WORKER_NUM
  BS = 8
  H = 7168
  HS = 7168
  K = 4
  sharedExpertNum = 1
  expertNumPerToken = K + sharedExpertNum
  microBatchNum = 1


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


  def run_ffn_to_attention(rank):
      set_device(rank)
      ep_group = init_hccl_comm(rank)

      token_info_table_shape = [microBatchNum, BS, expertNumPerToken]
      token_data_shape = [microBatchNum, BS, expertNumPerToken, HS]

      # Step 1: Create the communication buffer.
      buffer = get_buffer_for_ffn_to_attention(
          ep_group,
          WORLD_SIZE,
          token_info_table_shape,
          token_data_shape,
      )

      # The FFN Worker sends token data to the Attention Worker.
      if rank >= ATTENTION_WORKER_NUM:
          # Step 2: Construct the input data.
          tokens_per_rank = (
              BS * microBatchNum * ATTENTION_WORKER_NUM * expertNumPerToken // FFN_WORKER_NUM
          )
          x = torch.randn((tokens_per_rank, H), dtype=torch.bfloat16, device="npu")
          session_ids = torch.randint(
              0, ATTENTION_WORKER_NUM, (tokens_per_rank,), dtype=torch.int32, device="npu"
          )
          micro_batch_ids = torch.zeros(tokens_per_rank, dtype=torch.int32, device="npu")
          token_ids = torch.randint(0, BS, (tokens_per_rank,), dtype=torch.int32, device="npu")
          expert_offsets = torch.randint(
              0, expertNumPerToken, (tokens_per_rank,), dtype=torch.int32, device="npu"
          )
          actual_token_num = torch.tensor([tokens_per_rank], dtype=torch.int64, device="npu")
          attn_rank_table = torch.arange(ATTENTION_WORKER_NUM, dtype=torch.int32, device="npu")

          # Step 3: Call ffn_to_attention to send token data.
          ffn_to_attention(
              buffer,
              x,
              session_ids,
              micro_batch_ids,
              token_ids,
              expert_offsets,
              actual_token_num,
              attn_rank_table=attn_rank_table,
          )

      torch.npu.synchronize()
      buffer.destroy()
      dist.barrier()
      dist.destroy_process_group()
      print(f"[INFO] rank {rank} ffn_to_attention finished")


  if __name__ == "__main__":
      mp.set_start_method("forkserver", force=True)
      proc_list = []
      for rank in range(WORLD_SIZE):
          proc = Process(target=run_ffn_to_attention, args=(rank,))
          proc.start()
          proc_list.append(proc)
      for proc in proc_list:
          proc.join()
  ```
