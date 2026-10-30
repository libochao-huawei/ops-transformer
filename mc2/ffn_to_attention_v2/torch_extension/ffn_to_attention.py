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


class _FFNToAttentionOpBuilder(OpBuilder):
    def __init__(self):
        super(_FFNToAttentionOpBuilder, self).__init__(
            "npu_ffn_to_attention", category="mc2"
        )

    def sources(self):
        return ["csrc/mc2/ffn_to_attention.cpp"]

    def schema(self) -> str:
        return (
            "npu_ffn_to_attention(Tensor(a!) context, Tensor x, Tensor session_ids, "
            "Tensor micro_batch_ids, Tensor token_ids, Tensor expert_offsets, Tensor actual_token_num, "
            "str group, int world_size, "
            "int[] token_info_table_shape, int[] token_data_shape, int ccl_buffer_size, *, "
            "Tensor? attn_rank_table=None) -> ()"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def npu_ffn_to_attention_meta(
            context,
            x,
            session_ids,
            micro_batch_ids,
            token_ids,
            expert_offsets,
            actual_token_num,
            group,
            world_size,
            token_info_table_shape,
            token_data_shape,
            ccl_buffer_size,
            attn_rank_table=None,
        ):
            torch._check(
                world_size > 0,
                lambda: (
                    f"world_size should be greater than 0, {ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                x.dim() == 2,
                lambda: (
                    f"x should be 2D, but got {x.dim()}D, {ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                len(token_info_table_shape) == 3,
                lambda: (
                    f"token_info_table_shape should have 3 elements, "
                    f"but got {len(token_info_table_shape)}, {ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                len(token_data_shape) == 4,
                lambda: (
                    f"token_data_shape should have 4 elements, "
                    f"but got {len(token_data_shape)}, {ops_error(ErrCode.VALUE)}."
                ),
            )


_ffn_to_attention_op_builder = _FFNToAttentionOpBuilder()
_ffn_to_attention_op_builder._ensure_initialized()


@impl(get_as_library(), _ffn_to_attention_op_builder.name, "PrivateUse1")
def _npu_ffn_to_attention(
    context,
    x,
    session_ids,
    micro_batch_ids,
    token_ids,
    expert_offsets,
    actual_token_num,
    group,
    world_size,
    token_info_table_shape,
    token_data_shape,
    ccl_buffer_size,
    attn_rank_table=None,
):
    _op_module = _ffn_to_attention_op_builder.load()
    _op_module.npu_ffn_to_attention(
        context,
        x,
        session_ids,
        micro_batch_ids,
        token_ids,
        expert_offsets,
        actual_token_num,
        attn_rank_table,
        group,
        world_size,
        token_info_table_shape,
        token_data_shape,
        ccl_buffer_size,
    )


def _get_hccl_comm_name(group, rank_id):
    """Get HCCL communicator name from a torch.distributed ProcessGroup."""
    if hasattr(group, "get_hccl_comm_name"):
        return group.get_hccl_comm_name(rank_id, init_comm=False)
    get_backend = getattr(group, "".join(["_get", "_backend"]))
    return get_backend(torch.device("npu")).get_hccl_comm_name(rank_id, init_comm=False)


def _calc_window_size(token_info_table_shape, token_data_shape, world_size):
    """Calculate the conservative FFN-side window capacity."""
    _op_module = _ffn_to_attention_op_builder.load()
    return _op_module.get_ffn_to_attention_ccl_buffer_size(
        token_info_table_shape, token_data_shape, world_size
    )


class FFNToAttentionBuffer:
    """Communication buffer for ffn_to_attention, wraps CommContextManager.

    Uses the HCCL Channel backend (HcclCommMemReg + HcclChannelGetRemoteMems)
    to allocate and exchange window memory, compatible with Ascend 950.

    When ``window_addr`` and ``window_size`` are provided, the user memory at
    ``window_addr`` (``window_size`` bytes, the FFN-side window on this card)
    is used as the local communication window instead of an internally
    allocated buffer: it is zeroed once, registered into the HCCL
    communication domain, and never freed by this class. The caller owns the
    memory and must keep it alive while the buffer is in use. ``ccl_buffer_size``
    always equals the registered window size: the user-provided ``window_size``
    when custom memory is used, or the internally computed conservative
    capacity otherwise (it only hosts the attention_to_ffn dispatch receiving
    area for the maximum possible attention worker count). Per-call flag
    staging, address tables and count rows live in operator workspace. This
    class intentionally shares its
    registration tag with attention_to_ffn so peer ranks can discover this
    rank's window; create only one buffer per group in this process (attn
    workers create the attention_to_ffn buffer, FFN workers create this one).

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
        token_info_table_shape: List[int],
        token_data_shape: List[int],
        window_addr: Optional[int] = None,
        window_size: Optional[int] = None,
    ):
        if world_size != torch.distributed.get_world_size(group):
            raise ValueError("world_size must match the communication group size")
        self.group = group
        self.rank_id = torch.distributed.get_rank(group)
        self.group_name = _get_hccl_comm_name(group, self.rank_id)
        self.ep_world_size = world_size
        self.token_info_table_shape = list(token_info_table_shape)
        self.token_data_shape = list(token_data_shape)

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

        self.ccl_buffer_size = (
            window_size
            if window_size is not None
            else _calc_window_size(token_info_table_shape, token_data_shape, world_size)
        )

        self._capacity_shapes = (
            tuple(self.token_info_table_shape),
            tuple(self.token_data_shape),
        )
        # Local FFN-side window: user-provided memory (or internally allocated with the same size).
        self.window_addr = window_addr
        self.window_size = self.ccl_buffer_size

        # Async send-completion status table: registered (device-writable) host memory. The
        # kernel publishes the highest fully-completed send round into slot 0, and the Python
        # side polls it without any device synchronization to release keepalive references.
        op_module = _ffn_to_attention_op_builder.load()
        self._status_table = op_module.FfnToAttentionHostStatusTable()
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
        # sends of one buffer must go through the ffn_to_attention wrapper of this buffer.
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


def get_buffer_for_ffn_to_attention(
    group: "torch.distributed.ProcessGroup",
    world_size: int,
    token_info_table_shape: List[int],
    token_data_shape: List[int],
    *,
    window_addr: Optional[int] = None,
    window_size: Optional[int] = None,
) -> FFNToAttentionBuffer:
    """Create a communication buffer for ffn_to_attention.

    This is a thin factory around ``FFNToAttentionBuffer`` that computes the
    required FFN-side CCL buffer size internally, creates the communication
    context, and returns a buffer object holding all the information
    ``ffn_to_attention`` needs.

    Args:
        group (ProcessGroup): HCCL process group.
        world_size (int): Communication domain size.
        token_info_table_shape (List[int]): 3-element list [microBatchNum, BS, expertNumPerToken].
        token_data_shape (List[int]): 4-element list [microBatchNum, BS, expertNumPerToken, HS].
        window_addr (int, optional): Device address of the user-provided local
            (FFN-side) communication window memory on this card. Must be
            provided together with ``window_size``. The memory is zeroed once,
            registered into the HCCL communication domain and never freed by
            the framework; the caller must keep it alive while the buffer is
            in use. Only supported on the channel backend (Ascend950).
            Default None, meaning the framework allocates the window
            internally.
        window_size (int, optional): Size in bytes of the memory at
            ``window_addr``, i.e. the FFN-side memory size. It must be no
            smaller than the FFN-side window size required to host the
            attention_to_ffn dispatch receiving area (sized by the peer's
            actual quant_mode; use the conservative capacity computed by
            ``get_ffn_to_attention_ccl_buffer_size`` when the quant mode is
            unknown); per-call flag staging, address tables and count rows use
            operator workspace. The op tiling only validates the window
            against this op's own attention-side landing zone (one session's
            flags plus the fp16/bf16 return data). When omitted, the
            framework allocates the window with the internally computed
            conservative capacity (covering ``world_size - 1`` sessions) and
            exposes it as ``ccl_buffer_size`` of the returned buffer.
            Default None.

    Returns:
        FFNToAttentionBuffer: Buffer holding the communication context, group
        name, world size, token table shapes and CCL buffer size (the
        registered FFN-side window size).
    """
    return FFNToAttentionBuffer(
        group,
        world_size,
        token_info_table_shape,
        token_data_shape,
        window_addr=window_addr,
        window_size=window_size,
    )


def _release_completed_rounds(buffer: FFNToAttentionBuffer) -> None:
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


def _check_launch_stream(buffer: FFNToAttentionBuffer) -> None:
    """All launches on one buffer must share the launch stream.

    The keepalive round ids mirror the kernel launch ordinal on the buffer's context, which
    only holds while launches are stream-ordered; the shared channel state, context round
    counter and cross-AIV synchronization assume sequential same-stream execution.
    """
    stream = torch.npu.current_stream(buffer._device)
    if buffer._launch_stream is None:
        buffer._launch_stream = stream
    else:
        torch._check(
            stream == buffer._launch_stream,
            lambda: (
                "ffn_to_attention must use the stream of its first launch on this buffer "
                f"while async sends are in flight, {ops_error(ErrCode.VALUE)}."
            ),
        )


def ffn_to_attention(
    buffer: FFNToAttentionBuffer,
    x: torch.Tensor,
    session_ids: torch.Tensor,
    micro_batch_ids: torch.Tensor,
    token_ids: torch.Tensor,
    expert_offsets: torch.Tensor,
    actual_token_num: torch.Tensor,
    *,
    attn_rank_table: Optional[torch.Tensor] = None,
) -> None:
    """Send token data from FFN workers to Attention workers via HCCL window.

    Uses the CommContextManager-based context tensor to pass window memory
    addresses to the kernel, compatible with Ascend 950 (Channel backend).

    Args:
        buffer (FFNToAttentionBuffer): Communication buffer created by
            ``get_buffer_for_ffn_to_attention``. Holds the context tensor,
            group name, world size, token table shapes and CCL buffer size
            (the registered FFN-side window size).
        x (Tensor): Token data to send, shape (Y, H), dtype float16/bfloat16.
        session_ids (Tensor): Per-token attention worker index, shape (Y,), dtype int32.
        micro_batch_ids (Tensor): Per-token micro batch index, shape (Y,), dtype int32.
        token_ids (Tensor): Per-token batch offset, shape (Y,), dtype int32.
        expert_offsets (Tensor): Per-token topk expert offset, shape (Y,), dtype int32.
        actual_token_num (Tensor): Actual token count to send, shape (1,), dtype int64.
        attn_rank_table (Tensor, optional): Maps attention worker id to rank id,
            shape (attnRankNum,), dtype int32. If None, worker id == rank id.

    Returns:
        None. Data is sent to peer ranks via the HCCL window; no host-visible output.

    Note:
        The send is asynchronous: the kernel may return while the communication engine is
        still reading ``x`` (the relay data WQEs read the x GM rows directly). This wrapper
        therefore keeps ``x`` alive per round and releases it only after the device confirms
        the round's sends completed (polled lazily on the next call, or on
        ``FFNToAttentionBuffer.destroy()`` for the final round). Launches on one buffer must
        stay on the launch stream, and every launch must go through this wrapper so the round
        bookkeeping matches the kernel.
    """
    torch._check(
        buffer.context is not None,
        lambda: (
            "ffn_to_attention cannot be called after the buffer has been destroyed, "
            f"{ops_error(ErrCode.VALUE)}."
        ),
    )
    torch._check(
        x.device == buffer._device,
        lambda: (
            "ffn_to_attention requires x and buffer to be on the same device, "
            f"but got x.device={x.device}, buffer.device={buffer._device}, "
            f"{ops_error(ErrCode.VALUE)}."
        ),
    )
    if (
        tuple(buffer.token_info_table_shape),
        tuple(buffer.token_data_shape),
    ) != buffer._capacity_shapes:
        raise ValueError(
            "Window shapes cannot change after buffer creation; create a replacement buffer"
        )
    torch._check(x.dim() == 2, lambda: "x must be a 2D tensor")
    torch._check(
        x.shape[1] <= buffer.token_data_shape[3],
        lambda: "x.shape[1] exceeds the window token slot width HS",
    )
    _check_launch_stream(buffer)
    round_id = buffer._next_round_id
    buffer._next_round_id += 1
    # Keepalive: hold the reference only (no copy); the in-flight WQEs of this round may
    # still read x's storage after the kernel returns.
    buffer._keepalive[round_id] = x

    torch.ops.cann_ops_transformer.npu_ffn_to_attention(
        buffer.context,
        x,
        session_ids,
        micro_batch_ids,
        token_ids,
        expert_offsets,
        actual_token_num,
        buffer.group_name,
        buffer.ep_world_size,
        buffer.token_info_table_shape,
        buffer.token_data_shape,
        buffer.ccl_buffer_size,
        attn_rank_table=attn_rank_table,
    )

    # Lazy release: the kernel (when it already ran) published the completed round to the
    # host status table; finished rounds' references are dropped here without blocking.
    _release_completed_rounds(buffer)
