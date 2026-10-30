#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License).
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""pool_key_indexer golden: numpy 向量化参考实现(冒烟套精简版)。

数值口径: 非量化 score = (sum_g fp32(w)*relu(qk_fp32))*fp32(1/sqrt(128));
int8 qk=INT32 精确点积 -> fp16(2^-10)(DEQF16) -> relu -> wq=fp16(w)*fp16(dq) 预乘
-> FP32 聚合 -> (spool*dk)*(1/sqrt(128)*1024)(kernel 内 2^-10 中间缩放, 输出补偿 2^10)。
结构语义: pool 聚合/尾 pool AppendTail/索引为 batch 内局部口径/PA 逻辑块 bt[b, j//rpb]
映射/TND 前缀和/mask3 的 pool 级因果 + tail 可见数 vtk=clamp(tail_k, 0, g-(L_k-tail_k)+1)
(按尾池真实起点 L_k-tail_k 因果截断)。index_version: 1=legacy / 2=compact。
"""

import numpy as np

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - 纯 numpy 环境兜底
    torch = None
    _HAS_TORCH = False

FP16_MAX = 65504.0
# host tiling.cpp 口径: float scale = 1.0f / sqrtf(128.0f); int8 时 scale *= 1024.0f
_SCALE_BASE = np.float32(1.0) / np.sqrt(np.float32(128.0))
QK_SCALE_INT8 = _SCALE_BASE * np.float32(1024.0)
QK_SCALE_NO_QUANT = _SCALE_BASE


def _to_numpy(x):
    if x is None:
        return None
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        if x.dtype == torch.bfloat16:
            # numpy 无 bf16: bf16 -> fp32 逐位精确（非量化路径本就升 fp32 计算）
            return x.detach().cpu().to(torch.float32).numpy()
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _batch_lengths(seq, is_accum):
    """kernel GetActualSeqLen 口径: accum -> [seq[0], seq[1]-seq[0], ...]; 非 accum -> 逐 batch 长度。"""
    s = np.asarray(seq, dtype=np.int64).reshape(-1)
    if is_accum:
        return np.diff(np.concatenate(([0], s)))
    return s.copy()


def _f32(x):
    """标量 -> np.float32（经 fp64 精确中转, 单次舍入, 与硬件 fp32 逐位一致）。"""
    return np.float32(np.float64(x))


class GoldenResult:
    """golden 输出 + 比对所需中间信息（pool 分数 / batch 归属 / 行可见性 / 边界分数）。"""

    def __init__(
        self,
        indices,
        values,
        pool_scores,
        pool_batch,
        pool_local_idx,
        row_batch,
        nvalid,
        boundary,
        lk_per_row,
        expand_total=None,
        tail_tokens=None,
        index_version=1,
    ):
        self.indices = indices  # np.int32 [lead..., topk+ps-1]
        self.values = values  # np.float32 [lead..., kp] 或 shape (0,)
        self.pool_scores = (
            pool_scores  # np.float32 [S, P]（候选 pool 分数, -inf 为无效）
        )
        self.pool_batch = pool_batch  # np.int64 [P]
        self.pool_local_idx = pool_local_idx  # np.int64 [P]（batch 内局部 pool 序）
        self.row_batch = row_batch  # np.int64 [S]（query 行所属 batch）
        self.nvalid = nvalid  # np.int64 [S]（每行有效 pool 数）
        self.boundary = (
            boundary  # np.float32 [S]（该行第 k' 名边界分数; 无有效行 -inf）
        )
        self.lk_per_row = lk_per_row  # np.int64 [S]（每行所属 batch 的 L_k）
        # 紧凑布局比对辅助: 展开区长度(= tail 写入起点) 与 尾区 token 期望值
        self.expand_total = expand_total  # np.int64 [S]（min(nvalid, kp)*ps）
        self.tail_tokens = tail_tokens  # np.int64 [S, ps-1]（L_k-tail_k+t, 不可见 -1）
        # 索引输出布局版本: 1=legacy(V1) / 2=compact(V2 version=2), 比对口径据此分支
        self.index_version = index_version

    def pool_score(self, row, pool):
        """查询 (query 行, batch 内局部 pool 序) 的 golden 分数；不存在返回 None。"""
        r, p = int(row), int(pool)
        mask = (self.pool_batch == self.row_batch[r]) & (self.pool_local_idx == p)
        cols = np.nonzero(mask)[0]
        if cols.size == 0:
            return None
        return float(self.pool_scores[r, cols[0]])


def _row_batch_ids(qc):
    qcut = np.cumsum(qc)
    total = int(qc.sum())
    return np.minimum(
        np.searchsorted(qcut, np.arange(total), side="right"), len(qc) - 1
    )


def pki_golden(
    query,
    pool_key,
    weights,
    pool_tail_k,
    actual_seq_q=None,
    actual_seq_k=None,
    block_table=None,
    q_descale=None,
    k_descale=None,
    layout_q="BSND",
    layout_k="BSND",
    topk=2048,
    pool_size=16,
    mask_mode=3,
    quant_mode=-1,
    return_value=False,
    index_version=1,
):
    """pool_key_indexer golden 主入口（numpy 向量化实现）。

    参数与 aclnnPoolKeyIndexer / torch_extension 一致（tensor 或 numpy 均可，CPU 语义）。
    index_version: 索引输出布局版本（尾块追加模式）——1=legacy（tail 固定偏移 topk，
    短序列展开区与尾区之间有 -1 缝隙，V1 接口行为）；2=compact（tail 紧跟有效展开，
    -1 全在行尾，V2 接口 version=2 行为）。两种版本有效 token 集合一致。
    返回 GoldenResult（.indices / .values / 比对辅助信息）。
    """
    q_np = _to_numpy(query)
    k_np = _to_numpy(pool_key)
    w_np = _to_numpy(weights)
    dq_np = _to_numpy(q_descale)
    dk_np = _to_numpy(k_descale)
    asq_np = _to_numpy(actual_seq_q)
    ask_np = _to_numpy(actual_seq_k)
    bt_np = _to_numpy(block_table)
    ptk_np = _to_numpy(pool_tail_k)

    ps = int(pool_size)
    kp = int(topk) // ps
    is_int8 = quant_mode == 2
    N1 = int(q_np.shape[-2])
    D = int(q_np.shape[-1])
    lead = tuple(q_np.shape[:-2])
    S = int(np.prod(lead)) if lead else 1
    q3 = q_np.reshape(S, N1, D)

    # ---- batch 结构 ----
    if layout_q == "BSND":
        B = int(q_np.shape[0])
        qc = np.repeat(q_np.shape[1], B).astype(np.int64)
    else:  # TND
        B = int(asq_np.reshape(-1).shape[0])
        qc = _batch_lengths(asq_np, is_accum=True)
    if layout_k == "BSND":
        pc = np.repeat(k_np.shape[1], B).astype(np.int64)
    elif layout_k == "TND":
        pc = _batch_lengths(ask_np, is_accum=True)
    else:  # PA_BBND: actual_seq_k 为每 batch pool 数（非前缀和）
        pc = _batch_lengths(ask_np, is_accum=False)

    qbid = _row_batch_ids(qc)
    qcut = np.cumsum(qc)
    qpos = np.arange(S) - (qcut - qc)[qbid]

    tk = np.clip(ptk_np.astype(np.int64).reshape(-1), 0, ps - 1)[:B]
    Lk = pc * ps + tk  # 每 batch token 总数

    # ---- pool 逻辑序列（统一为候选列轴 [0, P)）----
    poff = np.concatenate(([0], np.cumsum(pc)[:-1]))
    totalP = int(pc.sum())
    pa_bid = np.repeat(np.arange(B), pc)
    pa_idx = np.arange(totalP) - poff[pa_bid]  # batch 内局部 pool 序
    if layout_k == "PA_BBND":
        rpb = int(k_np.shape[1]) * int(k_np.shape[2])  # blockSize * N2（每 block 行数）
        krows = k_np.reshape(-1, D)  # 逻辑展平（0 轴 stride padding 自动剔除）
        kdrows = dk_np.reshape(-1) if is_int8 else None
        bt2 = bt_np.astype(np.int64).reshape(B, -1)
        blk = np.minimum(pa_idx // rpb, bt2.shape[1] - 1)
        btg = bt2[pa_bid, blk]
        phys = btg * rpb + pa_idx % rpb
        klog = krows[phys]
        kdlog = kdrows[phys] if is_int8 else np.ones(totalP, dtype=np.float32)
        pbid, pidx = pa_bid, pa_idx
    else:
        klog = k_np.reshape(-1, D)
        kdlog = (
            dk_np.reshape(-1) if is_int8 else np.ones(klog.shape[0], dtype=np.float32)
        )
        pbid = np.minimum(
            np.searchsorted(np.cumsum(pc), np.arange(klog.shape[0]), side="right"),
            B - 1,
        )
        pidx = np.arange(klog.shape[0]) - poff[pbid]
    P = klog.shape[0]

    # ---- 数值六步 ----
    if is_int8:
        qi = q3.astype(np.int32)
        ki = klog.astype(np.int32)
        qk = np.matmul(qi, ki.transpose())  # [S, N1, P] INT32 精确
        t = qk.astype(np.float32) * np.float32(2.0**-10)
        sh = (
            np.clip(t, -FP16_MAX, FP16_MAX).astype(np.float16).astype(np.float32)
        )  # DEQF16(RNE)
        sh = np.maximum(sh, np.float32(0.0))  # reluPre（cast 后）
        wq = w_np.astype(np.float16).reshape(S, N1) * dq_np.astype(np.float16).reshape(
            S, N1
        )  # FP16 域预乘
        spool = np.einsum("sg,sgp->sp", wq.astype(np.float32), sh)  # FP32 聚合
        dkv = kdlog.astype(np.float32)
        score = (
            spool * dkv[None, :]
        ) * QK_SCALE_INT8  # 先 ×dk 再 ×qkScale（kernel 两次乘顺序）
    else:
        qf = q3.astype(np.float32)
        kf = klog.astype(np.float32)
        qk = np.matmul(qf, kf.transpose())  # [S, N1, P] FP32
        sh = np.maximum(qk, np.float32(0.0))
        wf = w_np.astype(np.float32).reshape(S, N1)
        spool = np.einsum("sg,sgp->sp", wf, sh)
        score = spool * QK_SCALE_NO_QUANT

    # ---- 候选有效性: batch 归属 ⊕ mask_mode=3 pool 级因果 ----
    g = Lk[qbid] - qc[qbid] + qpos  # query 在 batch 内的全局位置
    batch_ok = pbid[None, :] == qbid[:, None]
    if mask_mode == 3:
        visible = (pidx * ps + ps - 1)[None, :] <= g[:, None]
        valid = np.logical_and(batch_ok, visible)
    else:
        valid = batch_ok
    nvalid = valid.sum(axis=1)
    cand = np.where(valid, score, -np.inf)

    # ---- TopK: FP32 稳定降序（等值取小 pool 索引），不足补 -inf ----
    pad = max(0, kp - P)
    if pad > 0:
        candp = np.concatenate(
            [cand, np.full((S, pad), -np.inf, dtype=np.float32)], axis=1
        )
    else:
        candp = cand
    order = np.argsort(-candp, kind="stable", axis=1)
    topsel = order[:, :kp]
    topval = np.take_along_axis(candp, topsel, axis=1)

    # 边界分数: 该行第 k' 名（最后一个被选中的有效 pool）分数
    ex_cnt = np.minimum(nvalid, kp)
    boundary = np.where(
        ex_cnt > 0,
        np.take_along_axis(topval, np.clip(ex_cnt - 1, 0, kp - 1)[:, None], axis=1)[
            :, 0
        ],
        -np.inf,
    ).astype(np.float32)

    # ---- Expand: 槽位 i -> pool * ps + i % ps; 槽位块 >= ex_cnt 填 -1 ----
    islot = np.arange(int(topk))
    kslot = islot // ps
    sel_cand = topsel[:, kslot]  # [S, topk]
    sel_local = np.where(sel_cand < P, pidx[np.clip(sel_cand, 0, P - 1)], -1)
    ok_ex = kslot[None, :] < ex_cnt[:, None]
    tok_ex = np.where(ok_ex, sel_local * ps + (islot % ps)[None, :], -1)

    # ---- AppendTail: 尾 token = L_k - tail_k + t, t < vtk ----
    tslot = np.arange(ps - 1)
    if mask_mode == 0:
        vtk = tk[qbid]
    else:
        # 尾池真实起点 L_k-tail_k 的因果截断(修复后 kernel 口径): decode 末行
        # g=L_k-1 全尾可见; 早期行不含未来尾 token
        vtk = np.maximum(0, np.minimum(tk[qbid], g - (Lk[qbid] - tk[qbid]) + 1))
    tok_tl = (Lk[qbid] - tk[qbid])[:, None] + tslot[None, :]
    ok_tl = tslot[None, :] < vtk[:, None]
    tok_tl = np.where(ok_tl, tok_tl, -1)

    # ---- 输出布局组装 ----
    # index_version 1(legacy): [0, expandTotal) 展开 token | -1 缝隙 [expandTotal, topk)
    #   | tail token 固定偏移 topk | 行尾 -1 —— V1 接口布局
    # index_version 2(compact): [0, expandTotal) 展开 token | tail token(紧跟有效展开)
    #   | 其余 -1 全部位于行尾 —— V2 接口 version=2 布局
    expand_total = ex_cnt * ps
    width = int(topk) + ps - 1
    if index_version == 2:
        col = np.arange(width)[None, :]
        toks = np.full((S, width), -1, dtype=np.int64)
        toks[:, : int(topk)] = tok_ex
        in_tail = (col >= expand_total[:, None]) & (col < (expand_total + vtk)[:, None])
        tail_val = (Lk[qbid] - tk[qbid])[:, None] + (col - expand_total[:, None])
        toks = np.where(in_tail, tail_val, toks)
    else:
        toks = np.concatenate([tok_ex, tok_tl], axis=1)
    sparse_indices = toks.astype(np.int32).reshape(lead + (int(topk) + ps - 1,))
    if return_value:
        vout = np.where(
            np.arange(kp)[None, :] < nvalid[:, None], topval, -np.inf
        ).astype(np.float32)
        sparse_values = vout.reshape(lead + (kp,))
    else:
        sparse_values = np.zeros(0, dtype=np.float32)

    return GoldenResult(
        indices=sparse_indices,
        values=sparse_values,
        pool_scores=cand.astype(np.float32),
        pool_batch=pbid,
        pool_local_idx=pidx,
        row_batch=qbid,
        nvalid=nvalid,
        boundary=boundary,
        lk_per_row=Lk[qbid],
        expand_total=expand_total.astype(np.int64),
        tail_tokens=tok_tl.astype(np.int64),
        index_version=int(index_version),
    )
