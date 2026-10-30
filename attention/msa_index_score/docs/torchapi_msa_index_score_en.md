# msa\_index\_score

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T11:54:15.798Z -->

## Applicable Products

<!-- npu="910b" id1 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id3 -->

## Function

- **Description**: Wraps `aclnnMsaIndexScore` to compute the block score of the MSA Index Branch.
- **Formula**:

$$
score = Maxpool[(scale\cdot)Q_{idx}@K_{idx}^{T}+atten\_mask]+local\_mask
$$

- `sparse_mode=0`: non-causal; `sparse_mode=3`: rightDownCausal (requires the `[2048,2048]` `atten_mask`).
- `start_loc[B]`: logical block index of the current query, used by `local_mask`.
- `init_blocks` / `local_blocks`: forced high-score blocks after Maxpool; set to 0 for alignment with the Triton raw score. Defaults to `0` / `1`.
- Ascend 950PR&950DT products additionally supports query/key in the same type `torch.float8_e4m3fn` / `torch.float8_e5m2` / `torch_npu.hifloat8` (without `scale`). `hifloat8.npu()` is currently unsafe and is skipped by the test script; the kernel is registered.
- `q_len` / `kv_len` can be 0 (including the entire batch). The corresponding requests skip QK; the score of empty KV is `-inf`.
- For short decode on A2/A3 and Ascend 950PR&950DT products, MIX is launched based on the estimated number of M-tasks without saturating idle cores. For Ascend 950PR&950DT products with short M and long S, `kvChunks` are further split by KV stile.
- PageAttention `key` allows dim0 (physical page) to be non-contiguous and must be kept as a view (do not call `.contiguous()`); non-leading axes with size>1 must be contiguous, otherwise tiling rejects it. TND does not allow non-contiguous input.
- The second dimension of PageAttention `block_table` can be greater than the actual number of KV logical blocks (for example, a wide table pre-allocated by vLLM). On 950, C2UB flushes by window when the last dimension of score exceeds 256 columns.

## Function Prototype

```python
cann_ops_transformer.msa_index_score(
    query,
    key,
    start_loc,
    *,
    block_table=None,
    scale=None,
    atten_mask=None,
    actual_seq_qlen=None,
    actual_seq_klen=None,
    layout_key="BBND",
    sparse_mode=3,
    init_blocks=0,
    local_blocks=1,
) -> Tensor
```

## Parameters

<table style="undefined;table-layout: fixed; width: 1180px"><colgroup>
<col style="width: 160px">
<col style="width: 120px">
<col style="width: 360px">
<col style="width: 280px">
<col style="width: 260px">
</colgroup>
<thead>
  <tr>
    <th>Parameter</th>
    <th>Optional/Required</th>
    <th>Description</th>
    <th>dtype</th>
    <th>shape</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>query</td>
    <td>Required</td>
    <td>$Q_{idx}$ TND</td>
    <td>fp16/bf16; 950 also supports fp8_e4m3fn / fp8_e5m2 / hifloat8</td>
    <td><code>[T1,N1,D]</code></td>
  </tr>
  <tr>
    <td>key</td>
    <td>Required</td>
    <td>$K_{idx}$ PA BBND/BNBD or TND. PA on A2/A3 and 950 supports non-contiguous dim0.</td>
    <td>Same as query, or int8 (int8 with fp16 query)</td>
    <td><code>[NP,P,N2,D]</code> / <code>[NP,N2,P,D]</code> / <code>[T2,N2,D]</code></td>
  </tr>
  <tr>
    <td>start_loc</td>
    <td>Required</td>
    <td>Logical block where the query is located.</td>
    <td>int32</td>
    <td><code>[B]</code></td>
  </tr>
  <tr>
    <td>block_table</td>
    <td>Required for PA</td>
    <td>Logical → physical page; not passed for TND.</td>
    <td>int32</td>
    <td><code>[B,MB]</code></td>
  </tr>
  <tr>
    <td>scale</td>
    <td>Required for quantization</td>
    <td>Dequantization.</td>
    <td>float32</td>
    <td>PA <code>[NP,N2,P]</code>; TND <code>[T2,N2]</code></td>
  </tr>
  <tr>
    <td>atten_mask</td>
    <td>Required for mode=3</td>
    <td>Compressed lower triangular template.</td>
    <td>int8</td>
    <td><code>[2048,2048]</code></td>
  </tr>
  <tr>
    <td>actual_seq_qlen</td>
    <td>Required for TND query</td>
    <td>Query prefix sum.</td>
    <td>int32</td>
    <td><code>[B+1]</code></td>
  </tr>
  <tr>
    <td>actual_seq_klen</td>
    <td>Required</td>
    <td>S2 of each request for PA; key prefix sum for TND.</td>
    <td>int32</td>
    <td><code>[B]</code> / <code>[B+1]</code></td>
  </tr>
  <tr>
    <td>layout_key</td>
    <td>optional</td>
    <td>Key layout: <code>TND</code> / <code>BBND</code> / <code>BNBD</code>, default <code>BBND</code>.</td>
    <td>str</td>
    <td>-</td>
  </tr>
  <tr>
    <td>sparse_mode</td>
    <td>optional</td>
    <td>0 / 3, default 3.</td>
    <td>int</td>
    <td>-</td>
  </tr>
  <tr>
    <td>init_blocks</td>
    <td>optional</td>
    <td>Number of forced head blocks of <code>local_mask</code>. Defaults to 0</td>
    <td>int</td>
    <td>-</td>
  </tr>
  <tr>
    <td>local_blocks</td>
    <td>optional</td>
    <td>Local window of <code>local_mask</code>. Defaults to 1</td>
    <td>int</td>
    <td>-</td>
  </tr>
</tbody>
</table>

## Output

`[N1, T1, RoundUp(MB,16)]` float32

## Example

- PageAttention BBND / BNBD:

```python
import torch
import torch_npu
import cann_ops_transformer

T1, N1, N2, D, P = 32, 8, 1, 128, 128
B, NP, MB = 1, 8, 2
query = torch.randn(T1, N1, D, dtype=torch.float16).npu()
key_bbnd = torch.randn(NP, P, N2, D, dtype=torch.float16).npu()
# BNBD and BBND differ only in the N/P axis swap: key_bnbd = key_bbnd.permute(0, 2, 1, 3).contiguous()
block_table = torch.arange(B * MB, dtype=torch.int32).view(B, MB).npu()
actual_seq_qlen = torch.tensor([0, T1], dtype=torch.int32).npu()
actual_seq_klen = torch.tensor([256], dtype=torch.int32).npu()
start_loc = torch.tensor([1], dtype=torch.int32).npu()
atten_mask = torch.zeros(2048, 2048, dtype=torch.int8).npu()
score = cann_ops_transformer.msa_index_score(
    query, key_bbnd, start_loc,
    block_table=block_table, atten_mask=atten_mask,
    actual_seq_qlen=actual_seq_qlen, actual_seq_klen=actual_seq_klen,
    layout_key="BBND")
```

- TND packed key (`layout_key="TND"`, `block_table` not passed, `actual_seq_klen` is the `[B+1]` prefix sum):

```python
T2 = 256
key_tnd = torch.randn(T2, N2, D, dtype=torch.float16).npu()
actual_seq_klen_tnd = torch.tensor([0, T2], dtype=torch.int32).npu()
score_tnd = cann_ops_transformer.msa_index_score(
    query, key_tnd, start_loc,
    atten_mask=atten_mask,
    actual_seq_qlen=actual_seq_qlen, actual_seq_klen=actual_seq_klen_tnd,
    layout_key="TND")
```
