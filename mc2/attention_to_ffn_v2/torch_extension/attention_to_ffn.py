# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import Optional, List

import torch
import torch_npu
from torch.library import impl
from torch_npu.utils._error_code import ErrCode, ops_error
from cann_ops_transformer.op_builder import OpBuilder, get_as_library
from ..common import CommContextManager


class _AttentionToFfnOpBuilder(OpBuilder):
    def __init__(self):
        super(_AttentionToFfnOpBuilder, self).__init__(
            "npu_attention_to_ffn", category="mc2"
        )

    def sources(self):
        return ["csrc/mc2/attention_to_ffn.cpp"]

    def schema(self) -> str:
        return (
            "npu_attention_to_ffn(Tensor(a!) context, Tensor x, Tensor session_id, "
            "Tensor micro_batch_id, Tensor layer_id, Tensor expert_ids, Tensor expert_rank_table, "
            "str group, int world_size, "
            "int[] ffn_token_info_table_shape, int[] ffn_token_data_shape, int[] attn_token_info_table_shape, "
            "int moe_expert_num, int quant_mode, int sync_flag, int ffn_start_rank_id, int ccl_buffer_size, *, "
            "Tensor? scales=None, Tensor? active_mask=None) -> ()"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def npu_attention_to_ffn_meta(
            context,
            x,
            session_id,
            micro_batch_id,
            layer_id,
            expert_ids,
            expert_rank_table,
            group,
            world_size,
            ffn_token_info_table_shape,
            ffn_token_data_shape,
            attn_token_info_table_shape,
            moe_expert_num,
            quant_mode,
            sync_flag,
            ffn_start_rank_id,
            ccl_buffer_size,
            scales=None,
            active_mask=None,
        ):
            torch._check(
                world_size > 0,
                lambda: (
                    f"world_size should be greater than 0, {ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                moe_expert_num > 0,
                lambda: (
                    f"moe_expert_num should be greater than 0, {ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                x.dim() == 3,
                lambda: (
                    f"x should be 3D, but got {x.dim()}D, {ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                len(ffn_token_info_table_shape) == 3,
                lambda: (
                    f"ffn_token_info_table_shape should have 3 elements, "
                    f"but got {len(ffn_token_info_table_shape)}, {ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                len(ffn_token_data_shape) == 5,
                lambda: (
                    f"ffn_token_data_shape should have 5 elements, "
                    f"but got {len(ffn_token_data_shape)}, {ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                len(attn_token_info_table_shape) == 3,
                lambda: (
                    f"attn_token_info_table_shape should have 3 elements, "
                    f"but got {len(attn_token_info_table_shape)}, {ops_error(ErrCode.VALUE)}."
                ),
            )


_attention_to_ffn_op_builder = _AttentionToFfnOpBuilder()
_attention_to_ffn_op_builder._ensure_initialized()


@impl(get_as_library(), _attention_to_ffn_op_builder.name, "PrivateUse1")
def _npu_attention_to_ffn(
    context,
    x,
    session_id,
    micro_batch_id,
    layer_id,
    expert_ids,
    expert_rank_table,
    group,
    world_size,
    ffn_token_info_table_shape,
    ffn_token_data_shape,
    attn_token_info_table_shape,
    moe_expert_num,
    quant_mode,
    sync_flag,
    ffn_start_rank_id,
    ccl_buffer_size,
    scales=None,
    active_mask=None,
):
    _op_module = _attention_to_ffn_op_builder.load()
    _op_module.npu_attention_to_ffn(
        context,
        x,
        session_id,
        micro_batch_id,
        layer_id,
        expert_ids,
        expert_rank_table,
        scales,
        active_mask,
        group,
        world_size,
        ffn_token_info_table_shape,
        ffn_token_data_shape,
        attn_token_info_table_shape,
        moe_expert_num,
        quant_mode,
        sync_flag,
        ffn_start_rank_id,
        ccl_buffer_size,
    )


def _get_hccl_comm_name(group, rank_id):
    """Get HCCL communicator name from a torch.distributed ProcessGroup."""
    if hasattr(group, "get_hccl_comm_name"):
        return group.get_hccl_comm_name(rank_id, init_comm=False)
    get_backend = getattr(group, "".join(["_get", "_backend"]))
    return get_backend(torch.device("npu")).get_hccl_comm_name(rank_id, init_comm=False)


def _get_attention_to_ffn_ccl_buffer_size(
    ffn_token_info_table_shape: List[int],
    ffn_token_data_shape: List[int],
    quant_mode: int = 0,
) -> int:
    _op_module = _attention_to_ffn_op_builder.load()
    return _op_module.get_attention_to_ffn_ccl_buffer_size(
        ffn_token_info_table_shape, ffn_token_data_shape, quant_mode
    )


class AttentionToFfnBuffer:
    """Communication buffer for attention_to_ffn, wraps CommContextManager.

    Uses the HCCL Channel backend (HcclCommMemReg + HcclChannelGetRemoteMems)
    to allocate and exchange window memory, compatible with Ascend 950.

    When ``window_addr`` and ``window_size`` are provided, the user memory at
    ``window_addr`` (``window_size`` bytes, the attention-side window on this
    card) is used as the local communication window instead of an internally
    allocated buffer: it is zeroed once, registered into the HCCL
    communication domain, and never freed by this class. The caller owns the
    memory and must keep it alive while the buffer is in use. The FFN-side CCL
    buffer size is always computed internally from the FFN-side token shapes
    and exposed as ``ccl_buffer_size`` (the size of the remote FFN windows the
    attention_to_ffn op writes to).

    The buffer also owns the async tensor keepalive state: a registered
    (device-writable) host status table that kernels use to publish completed
    send rounds, and the per-round references of in-flight input tensors.
    Call ``destroy()`` (or drop the last reference) to wait for the final
    round's in-flight sends and release everything.
    """

    def __init__(
        self,
        group: "torch.distributed.ProcessGroup",
        world_size: int,
        ffn_token_info_table_shape: List[int],
        ffn_token_data_shape: List[int],
        quant_mode: int = 0,
        window_addr: Optional[int] = None,
        window_size: Optional[int] = None,
    ):
        self.group = group
        self.rank_id = torch.distributed.get_rank(group)
        self.group_name = _get_hccl_comm_name(group, self.rank_id)
        self.ep_world_size = world_size
        self.ffn_token_info_table_shape = list(ffn_token_info_table_shape)
        self.ffn_token_data_shape = list(ffn_token_data_shape)
        self.quant_mode = quant_mode

        torch._check(
            (window_addr is None) == (window_size is None),
            lambda: (
                f"window_addr and window_size must be provided together, but got "
                f"window_addr={window_addr}, window_size={window_size}, {ops_error(ErrCode.VALUE)}."
            ),
        )
        if window_addr is not None:
            torch._check(
                window_addr > 0,
                lambda: (
                    f"window_addr should be a positive device address, but got {window_addr}, "
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                window_size > 0,
                lambda: (
                    f"window_size should be greater than 0, but got {window_size}, "
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )

        # FFN-side CCL buffer size, computed internally from the FFN-side token
        self.ccl_buffer_size = _get_attention_to_ffn_ccl_buffer_size(
            ffn_token_info_table_shape, ffn_token_data_shape, quant_mode
        )
        # Local attention-side window: user-provided memory
        self.window_addr = window_addr
        self.window_size = (
            window_size if window_size is not None else self.ccl_buffer_size
        )

        # Async send-completion status table: registered (device-writable) host memory. The
        # kernel publishes the highest fully-completed send round into slot 0, and the Python
        # side polls it without any device synchronization to release keepalive references.
        op_module = _attention_to_ffn_op_builder.load()
        self._status_table = op_module.AttentionToFfnHostStatusTable()
        host_status_table_addr = self._status_table.device_ptr()
        # 0 would silently disable the kernel-side completion report, so the keepalive
        # references would never be released; fail loudly instead.
        torch._check(
            host_status_table_addr > 0,
            lambda: (
                f"hostStatusTableAddr should be a positive device address "
                f"(status table device_ptr()), but got {host_status_table_addr}, "
                f"{ops_error(ErrCode.VALUE)}."
            ),
        )

        self._ctx_manager = CommContextManager(
            self.group_name,
            self.ep_world_size,
            backend={
                "Ascend910B": "kfc",
                "Ascend910_93": "kfc",
                "Ascend950": "channel",
            },
            commAlg="urma",
            opName="attention_to_ffn",
            customCclBufferSize=self.window_size,
            customDeviceBufferAddr=window_addr if window_addr is not None else 0,
            multiChannel=True,
            hostStatusTableAddr=host_status_table_addr,
        )
        self.context = self._ctx_manager.create_context()
        self._device = self.context.device

        # Tensor keepalive for async sends: round_id -> tensors still read by in-flight WQEs.
        # Only references are held (no copy); entries are dropped once the device confirms the
        # round completed. Round ids match the kernel launch ordinal on this buffer's context
        # (both start at 0 and advance once per call, launches are stream-ordered), so all
        # sends of one buffer must go through the attention_to_ffn wrapper of this buffer.
        self._keepalive = {}
        self._next_round_id = 0
        self._launch_stream = None

    def destroy(self):
        """Release the buffer and every retained resource.

        This is the explicit end-of-life wait of the async tensor keepalive: a device-wide
        synchronization retires all kernels and every in-flight WQE first, after which the
        keepalive references are dropped and the registered status-table host memory is
        unregistered. Skipping ``destroy`` and dropping the last reference also works (the
        status table's destructor performs the same device drain), but the explicit call
        makes the wait observable and deterministic.
        """
        if self.context is None:
            return
        torch.npu.synchronize(self._device)
        self._keepalive.clear()
        self._status_table = None
        self._ctx_manager.destroy()
        self.context = None
        self._launch_stream = None

    def get_window_addr(self) -> int:
        """Get the local window memory device address from the context."""
        import ctypes

        if self.context is None:
            return 0

        host_tensor = self.context.cpu()
        context_ptr = host_tensor.data_ptr()
        if context_ptr == 0:
            return 0

        offset_epHcclBuffer = 2 * 4 + 8
        offset_target = offset_epHcclBuffer + self.rank_id * 8
        addr_bytes = ctypes.c_uint64.from_address(context_ptr + offset_target)
        return int(addr_bytes.value)


def get_buffer_for_attention_to_ffn(
    group: "torch.distributed.ProcessGroup",
    world_size: int,
    ffn_token_info_table_shape: List[int],
    ffn_token_data_shape: List[int],
    *,
    quant_mode: int = 0,
    window_addr: Optional[int] = None,
    window_size: Optional[int] = None,
) -> AttentionToFfnBuffer:
    """Create a communication buffer for attention_to_ffn.

    This is a thin factory around ``AttentionToFfnBuffer`` that computes the
    required FFN-side CCL buffer size internally, creates the communication
    context, and returns a buffer object holding all the information
    ``attention_to_ffn`` needs.

    Args:
        group (ProcessGroup): HCCL process group.
        world_size (int): Communication domain size.
        ffn_token_info_table_shape (List[int]): 3-element list.
        ffn_token_data_shape (List[int]): 5-element list
            [attnWorkerNum, microBatchNum, BS, K+shared, HS].
        quant_mode (int): Quantization mode. 0=none, 2=pertoken+INT8,
            3=mx+FP8_E5M2, 4=mx+FP8_E4M3, 5=mx+FP4_E2M1,
            6=mx_clip+FP8_E5M2, 7=mx_clip+FP8_E4M3. Default 0.
        window_addr (int, optional): Device address of the user-provided local
            (attention-side) communication window memory on this card. Must be
            provided together with ``window_size``. The memory is zeroed once,
            registered into the HCCL communication domain and never freed by
            the framework; the caller must keep it alive while the buffer is
            in use. Only supported on the channel backend (Ascend950).
            Default None, meaning the framework allocates the window
            internally.
        window_size (int, optional): Size in bytes of the memory at
            ``window_addr``, i.e. the attention-side memory size. It must be
            no smaller than the attention-side window size required to receive
            the ffn_to_attention token data (computed from
            [microBatchNum, BS, expertNumPerToken] and
            [microBatchNum, BS, expertNumPerToken, HS], 2MB aligned). The
            FFN-side memory size is computed internally from
            ``ffn_token_info_table_shape``/``ffn_token_data_shape`` and
            ``quant_mode`` and exposed as ``ccl_buffer_size`` of the returned
            buffer. Default None, meaning the framework allocates the window
            internally.

    Returns:
        AttentionToFfnBuffer: Buffer holding the communication context,
        CCL buffer size and the FFN-side token shapes.
    """
    return AttentionToFfnBuffer(
        group,
        world_size,
        ffn_token_info_table_shape,
        ffn_token_data_shape,
        quant_mode=quant_mode,
        window_addr=window_addr,
        window_size=window_size,
    )


def _release_completed_rounds(buffer: AttentionToFfnBuffer) -> None:
    """Drop keepalive references of rounds the device already confirmed complete.

    Non-blocking: reads slot 0 of the registered host status table (a volatile host-memory
    read, no device synchronization). The kernel maintains the invariant that every WQE
    posted before the launch of round N is retired once the table reports N - 1, so
    releasing rounds <= completed is safe even though the current round is still in flight.
    """
    status_table = buffer._status_table
    if status_table is None:
        return
    completed = status_table.completed_round()
    if completed < 0 or not buffer._keepalive:
        return
    for round_id in [r for r in buffer._keepalive if r <= completed]:
        del buffer._keepalive[round_id]


def _check_launch_stream(buffer: AttentionToFfnBuffer) -> None:
    """All launches on one buffer must share the launch stream.

    The keepalive round ids mirror the kernel launch ordinal on the buffer's context, which
    only holds while launches are stream-ordered; the operator itself (shared workspace and
    cross-AIV SyncAll) already assumes sequential same-stream execution.
    """
    stream = torch.npu.current_stream(buffer._device)
    if buffer._launch_stream is None:
        buffer._launch_stream = stream
    else:
        torch._check(
            stream == buffer._launch_stream,
            lambda: (
                "attention_to_ffn must use the stream of its first launch on this buffer "
                f"while async sends are in flight, {ops_error(ErrCode.VALUE)}."
            ),
        )


def attention_to_ffn(
    buffer: AttentionToFfnBuffer,
    x: torch.Tensor,
    session_id: torch.Tensor,
    micro_batch_id: torch.Tensor,
    layer_id: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_rank_table: torch.Tensor,
    attn_token_info_table_shape: List[int],
    moe_expert_num: int,
    *,
    sync_flag: int = 0,
    ffn_start_rank_id: int = 0,
    scales: Optional[torch.Tensor] = None,
    active_mask: Optional[torch.Tensor] = None,
) -> None:
    """Send token data from Attention workers to FFN workers via HCCL window.

    Uses the CommContextManager-based context tensor to pass window memory
    addresses to the kernel, compatible with Ascend 950 (Channel backend).

    Args:
        buffer (AttentionToFfnBuffer): Communication buffer created by
            ``get_buffer_for_attention_to_ffn``. Holds the context tensor,
            group name, world size, FFN-side token shapes, quant mode and CCL
            buffer size (the FFN-side window size, computed internally).
        x (Tensor): Token data to send, shape (1, BS, H), dtype float16/bfloat16.
        session_id (Tensor): Session ID, shape (1,), dtype int32.
        micro_batch_id (Tensor): Micro batch ID, shape (1,), dtype int32.
        layer_id (Tensor): Layer ID, shape (1,), dtype int32.
        expert_ids (Tensor): Per-token topK expert indices, shape (1, BS, K), dtype int32.
        expert_rank_table (Tensor): Expert-to-rank mapping table,
            shape (L, expertNum, expRankTableM), dtype int32.
        attn_token_info_table_shape (List[int]): 3-element list.
        moe_expert_num (int): Number of MOE (routed) experts.
        sync_flag (int): Sync flag, 0=async, 1=sync. Default 0.
        ffn_start_rank_id (int): Starting rank ID of FFN workers. Default 0.
        scales (Tensor, optional): Smooth scales for quantization,
            shape (expertNum, expRankTableM, H), dtype float32.
        active_mask (Tensor, optional): Token activation mask,
            shape (1, BS), dtype bool.

    Returns:
        None. Data is sent to peer ranks via the HCCL window; no host-visible output.

    Note:
        The send is asynchronous: the kernel may return while the communication engine is
        still reading ``x`` (non-quant direct send reads the x GM rows through the relay
        WQEs). This wrapper therefore keeps ``x`` alive per round and releases it only after
        the device confirms the round's sends completed (polled lazily on the next call, or
        on ``AttentionToFfnBuffer.destroy()`` for the final round). Launches on one buffer
        must stay on the launch stream, and every launch must go through this wrapper so the
        round bookkeeping matches the kernel.
    """
    torch._check(
        buffer.context is not None,
        lambda: (
            "attention_to_ffn cannot be called after the buffer has been destroyed, "
            f"{ops_error(ErrCode.VALUE)}."
        ),
    )
    torch._check(
        x.device == buffer._device,
        lambda: (
            "attention_to_ffn requires x and buffer to be on the same device, "
            f"but got x.device={x.device}, buffer.device={buffer._device}, "
            f"{ops_error(ErrCode.VALUE)}."
        ),
    )
    _check_launch_stream(buffer)
    round_id = buffer._next_round_id
    buffer._next_round_id += 1
    # Keepalive: hold the reference only (no copy); the in-flight WQEs of this round may
    # still read x's storage after the kernel returns.
    buffer._keepalive[round_id] = x

    torch.ops.cann_ops_transformer.npu_attention_to_ffn(
        buffer.context,
        x,
        session_id,
        micro_batch_id,
        layer_id,
        expert_ids,
        expert_rank_table,
        buffer.group_name,
        buffer.ep_world_size,
        buffer.ffn_token_info_table_shape,
        buffer.ffn_token_data_shape,
        attn_token_info_table_shape,
        moe_expert_num,
        buffer.quant_mode,
        sync_flag,
        ffn_start_rank_id,
        buffer.ccl_buffer_size,
        scales=scales,
        active_mask=active_mask,
    )

    # Lazy release: the kernel (when it already ran) published the completed round to the
    # host status table; finished rounds' references are dropped here without blocking.
    _release_completed_rounds(buffer)
