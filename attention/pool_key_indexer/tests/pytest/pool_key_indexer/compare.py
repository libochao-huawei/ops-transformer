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
"""比对器: indices -1 掩码逐位一致 + 展开区结构校验(池连续段/越界) + token->pool
集合比较(并列翻转以边界分数差 <=1e-3 容忍, 差集不等量=真错误) + compact 紧凑性
不变量(有效 token 行首连续, -1 全在行尾); values isclose(rtol=5e-3, atol=2.5e-5)
通过率 >=99.5%; 通用不变量 indices>=-1。布局版本随 golden 的 index_version 自适应。
"""

import numpy as np

INDICES_RTOL = 0.0  # INT32 集合比较（无逐元素容差，走边界分数容忍）
BOUNDARY_THRES = 1e-3  # 集合差异元素 vs 边界分数相对误差容忍
VALUES_ATOL = 2.5e-05
VALUES_RTOL = 0.005
VALUES_PASS_RATE = 0.995


def _to_numpy(x):
    if hasattr(x, "detach"):  # torch.Tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)


class CompareReport:
    """比对结果汇总（断言失败时携带明细）。"""

    def __init__(self):
        self.failures = []
        self.warnings = []
        self.rows_total = 0
        self.rows_set_equal = 0
        self.rows_boundary_tolerated = 0
        self.values_total = 0
        self.values_passed = 0

    @property
    def passed(self):
        return not self.failures

    def fail(self, msg):
        self.failures.append(msg)

    def summary(self):
        lines = [
            f"rows: total={self.rows_total}, set_equal={self.rows_set_equal}, "
            f"boundary_tolerated={self.rows_boundary_tolerated}",
        ]
        if self.values_total:
            rate = self.values_passed / self.values_total * 100.0
            lines.append(
                f"values: passed={self.values_passed}/{self.values_total} ({rate:.4f}%), "
                f"thd>={VALUES_PASS_RATE * 100:.1f}%"
            )
        for w in self.warnings:
            lines.append(f"[warn] {w}")
        for f in self.failures:
            lines.append(f"[FAIL] {f}")
        return "\n".join(lines)


def _check_row_structure(row_idx, row_tokens, pool_size, lk, expand_limit, compact):
    """展开区 [0, expand_limit) 结构校验: 每 ps 槽位为同一 pool 的连续完整 token 段;
    无重复; 无越界。
    compact: 该区间内不应有 -1(全部池位有效展开);
    legacy: 允许整块 -1(短序列缝隙), 但不允许 -1 与有效 token 混块。"""
    problems = []
    pools = []
    for k in range(0, expand_limit, pool_size):
        seg = row_tokens[k : k + pool_size]
        if np.all(seg == -1):
            if compact:
                problems.append(
                    f"row {row_idx}: 展开区槽位块 {k // pool_size} 含 -1 (有效展开数不足或越界): {seg.tolist()}"
                )
            continue
        if np.any(seg == -1):
            problems.append(
                f"row {row_idx}: slot block {k // pool_size} 混合 -1 与有效 token: {seg.tolist()}"
            )
            continue
        base_pool = int(seg[0]) // pool_size
        expect = [base_pool * pool_size + t for t in range(pool_size)]
        if seg.tolist() != expect:
            problems.append(
                f"row {row_idx}: slot block {k // pool_size} 非连续完整段 "
                f"{expect}: {seg.tolist()}"
            )
            continue
        if base_pool * pool_size + pool_size > lk:
            problems.append(f"row {row_idx}: pool {base_pool} token 越界 (>= L_k={lk})")
            continue
        pools.append(base_pool)
    if len(set(pools)) != len(pools):
        problems.append(f"row {row_idx}: 展开区存在重复 pool: {sorted(pools)}")
    return problems


def compare_indices(golden_res, npu_indices, topk, pool_size):
    """sparse_indices 比对(布局版本自适应): -1 掩码逐位 + 展开区结构/集合比较
    (边界容忍 1e-3) + 尾区逐位; compact 版本附加紧凑性不变量(无中间空洞)。"""
    report = CompareReport()
    g_idx = (
        _to_numpy(golden_res.indices).reshape(-1, topk + pool_size - 1).astype(np.int64)
    )
    n_idx = _to_numpy(npu_indices).reshape(-1, topk + pool_size - 1).astype(np.int64)
    if g_idx.shape != n_idx.shape:
        report.fail(
            f"indices shape mismatch: golden {g_idx.shape} vs npu {n_idx.shape}"
        )
        return report
    if np.any(n_idx < -1):
        report.fail(f"indices 存在 < -1 的元素: min={int(n_idx.min())}")

    ps = pool_size
    kp = topk // ps
    compact = int(getattr(golden_res, "index_version", 1)) == 2
    for r in range(g_idx.shape[0]):
        g_row, n_row = g_idx[r], n_idx[r]
        report.rows_total += 1
        lk = int(golden_res.lk_per_row[r])
        # 展开区长度 = 有效展开 token 数; legacy 的展开区覆盖 [0, topk)(含缝隙),
        # compact 的展开区仅 [0, expandTotal)
        expand_total = int(min(int(golden_res.nvalid[r]), kp) * ps)
        expand_limit = expand_total if compact else topk
        # tail 写入起点: legacy 固定 topk; compact = expandTotal
        tail_start = topk if not compact else expand_total

        # (2) -1 无效位逐位(等价校验展开数与 vtk 两侧一致)
        g_mask, n_mask = g_row == -1, n_row == -1
        if not np.array_equal(g_mask, n_mask):
            report.fail(
                f"row {r}: -1 无效位不一致 golden={np.where(g_mask)[0].tolist()} "
                f"npu={np.where(n_mask)[0].tolist()}"
            )
            continue

        if compact:
            # (6) 紧凑性不变量(NPU): 有效 token 从行首连续, -1 全在行尾 —— 无中间缝隙
            n_valid = int((n_row >= 0).sum())
            if np.any(n_row[:n_valid] == -1) or np.any(n_row[n_valid:] != -1):
                first_gap = (
                    n_valid + int(np.argmax(n_row[n_valid:] != -1))
                    if n_valid < len(n_row)
                    else -1
                )
                report.fail(
                    f"row {r}: 非紧凑布局(有效 token 与行尾 -1 之间存在空洞), "
                    f"首个空洞位 {first_gap}: {n_row.tolist()}"
                )
                continue

        # (4) 尾区([tail_start, 行尾)) 与 golden 逐位(tail token 值 + 行尾 -1)
        if not np.array_equal(g_row[tail_start:], n_row[tail_start:]):
            report.fail(
                f"row {r}: AppendTail 尾区逐位不一致 golden={g_row[tail_start:].tolist()} "
                f"npu={n_row[tail_start:].tolist()}"
            )
            continue

        # (3) 展开区结构校验（NPU 侧, 范围 [0, expand_limit)）
        problems = _check_row_structure(r, n_row, ps, lk, expand_limit, compact)
        if problems:
            for p in problems:
                report.fail(p)
            continue

        # (5) 展开区 pool 集合比较（-1 剔除）
        g_pools = set((g_row[:expand_limit][g_row[:expand_limit] != -1] // ps).tolist())
        n_pools = set((n_row[:expand_limit][n_row[:expand_limit] != -1] // ps).tolist())
        if g_pools == n_pools:
            report.rows_set_equal += 1
            continue
        boundary = float(golden_res.boundary[r])
        diff_pools = (n_pools - g_pools) | (g_pools - n_pools)
        tolerated = True
        detail = []
        for p in sorted(diff_pools):
            score = golden_res.pool_score(r, p)
            if score is None or not np.isfinite(score):
                tolerated = False
                detail.append(f"pool {p}: score={score} 不可评估")
                continue
            if boundary == 0.0:
                ok = score == 0.0
            elif np.isfinite(boundary):
                ok = abs(score - boundary) / abs(boundary) <= BOUNDARY_THRES
            else:
                ok = False
            detail.append(
                f"pool {p}: score={score:.7f} boundary={boundary:.7f} "
                f"rel_err={abs(score - boundary) / max(abs(boundary), 1e-30):.3e} "
                f"{'tolerated' if ok else 'EXCEEDED'}"
            )
            tolerated &= ok
        if tolerated:
            report.rows_boundary_tolerated += 1
            report.warnings.append(
                f"row {r}: 集合差异在边界容忍内 -> " + "; ".join(detail)
            )
        else:
            report.fail(
                f"row {r}: 展开区 pool 集合不一致 golden={sorted(g_pools)} npu={sorted(n_pools)}; "
                + "; ".join(detail)
            )
    return report


def compare_values(golden_values, npu_values, return_value):
    """sparse_values 比对: FP32 isclose(atol=2.5e-5, rtol=0.005) 通过率 >= 99.5%。"""
    report = CompareReport()
    g_val = _to_numpy(golden_values)
    n_val = _to_numpy(npu_values)
    if not return_value:
        if n_val.shape != (0,) or n_val.size != 0:
            report.fail(
                f"return_value=false 时 values 应为 (0,) 空 tensor, got shape={n_val.shape}"
            )
        return report
    if g_val.shape != n_val.shape:
        report.fail(f"values shape mismatch: golden {g_val.shape} vs npu {n_val.shape}")
        return report
    if n_val.size == 0:
        return report
    g_flat = g_val.reshape(-1).astype(np.float32)
    n_flat = n_val.reshape(-1).astype(np.float32)
    close = np.isclose(
        n_flat, g_flat, rtol=VALUES_RTOL, atol=VALUES_ATOL, equal_nan=True
    )
    report.values_total = int(g_flat.size)
    report.values_passed = int(close.sum())
    rate = report.values_passed / report.values_total
    if rate < VALUES_PASS_RATE:
        bad = np.nonzero(~close)[0]
        detail = ", ".join(
            f"[{i}] golden={g_flat[i]:.7f} npu={n_flat[i]:.7f}" for i in bad[:8]
        )
        report.fail(
            f"values 通过率 {rate * 100:.4f}% < {VALUES_PASS_RATE * 100:.1f}% "
            f"({len(bad)}/{g_flat.size} 失败): {detail}"
        )
    # 附加断言: 有效前缀沿末轴非增（monotone_topk_invariance）
    n2 = n_val.reshape(-1, n_val.shape[-1])
    with np.errstate(invalid="ignore"):  # -inf - -inf = nan 不参与判定
        non_monotone = np.diff(n2, axis=-1) > 0
    if n2.shape[1] > 1 and np.any(non_monotone):
        r, c = np.argwhere(non_monotone)[0]
        report.fail(
            f"values 非单调非增: [{r},{c}] {n2[r, c]} < [{r},{c + 1}] {n2[r, c + 1]}"
        )
    return report


def compare_outputs(golden_res, npu_indices, npu_values, topk, pool_size, return_value):
    """整算子输出比对入口: indices + values + invariants。返回 (passed, report)。"""
    idx_report = compare_indices(golden_res, npu_indices, topk, pool_size)
    val_report = compare_values(golden_res.values, npu_values, return_value)
    passed = idx_report.passed and val_report.passed
    report = CompareReport()
    report.failures = idx_report.failures + val_report.failures
    report.warnings = idx_report.warnings + val_report.warnings
    report.rows_total = idx_report.rows_total
    report.rows_set_equal = idx_report.rows_set_equal
    report.rows_boundary_tolerated = idx_report.rows_boundary_tolerated
    report.values_total = val_report.values_total
    report.values_passed = val_report.values_passed
    return passed, report
