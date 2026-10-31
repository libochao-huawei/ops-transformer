#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import logging
import math

import torch

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s]-%(filename)s:%(lineno)04d - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)

_EPS = 1e-10


def cal_relative_diff_torch(real_data, expect_data):
    diff = abs(float(real_data) - float(expect_data))
    return diff / (abs(float(expect_data)) + _EPS)


def display_output_torch(real_data, expect_data, start, end, expect_fp32_data=None):
    has_fp32 = expect_fp32_data is not None

    logger.info(
        "---------------------------------------------------------------------------------------"
    )
    logger.info(
        "Loop \t ExpFP32Out \t ExpFP16Out \t NPUOut \tFpDiff(min) \t RateDiff"
        if has_fp32
        else "Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff"
    )
    logger.info(
        "---------------------------------------------------------------------------------------"
    )

    split_count = int(end - start)

    def display_row(i):
        j = start + i
        real_value = float(real_data[j])
        expect_value = float(expect_data[j])
        diff_rate = cal_relative_diff_torch(real_value, expect_value)
        idx = start + i + 1

        if not torch.isfinite(expect_data[j]).item():
            diff_abs = "inf" if torch.isinf(expect_data[j]).item() else "nan"
            if has_fp32:
                logger.info(
                    "%08d \t %-7s \t %-7s \t %-7s \t %-7s \t %-7s",
                    idx,
                    float(expect_fp32_data[j]),
                    expect_value,
                    real_value,
                    diff_abs,
                    diff_rate,
                )
            else:
                logger.info(
                    "%08d \t %-7s \t %-7s \t %-7s \t %-7s",
                    idx,
                    expect_value,
                    real_value,
                    diff_abs,
                    diff_rate,
                )
        else:
            diff_abs = abs(expect_value - real_value)
            if has_fp32:
                logger.info(
                    "%08d \t %0.7f \t %0.7f \t %0.7f \t %0.7f \t %0.7f",
                    idx,
                    float(expect_fp32_data[j]),
                    expect_value,
                    real_value,
                    diff_abs,
                    diff_rate,
                )
            else:
                logger.info(
                    "%08d \t %0.7f \t %0.7f \t %0.7f \t %0.7f",
                    idx,
                    expect_value,
                    real_value,
                    diff_abs,
                    diff_rate,
                )

    if split_count <= 20:
        for i in range(split_count + 1):
            display_row(i)
    else:
        for i in range(10):
            display_row(i)
        logger.info("...   \t   ...   \t   ...   \t   ...    \t   ...")
        for i in range(split_count - 10 + 1, split_count + 1):
            display_row(i)


def display_error_output_torch(real_data, expect_data, err_idx, relative_diff):
    logger.info(
        "Error Line-----------------------------------------------------------------------------"
    )
    logger.info("Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff")
    logger.info(
        "---------------------------------------------------------------------------------------"
    )

    len_err = int(err_idx.numel())
    for count, i in enumerate(err_idx.tolist(), 1):
        if count < 10 or (90 < count < 100):
            expect_value = float(expect_data[i])
            real_value = float(real_data[i])
            logger.info(
                "%08d \t %.7f \t %.7f \t %.7f \t %.7f",
                i,
                expect_value,
                real_value,
                abs(expect_value - real_value),
                float(relative_diff[count - 1].item()),
            )
        elif count == 10 or (count == 100 and len_err > 100):
            logger.info(
                "%08s \t %07s \t %07s \t %07s \t %07s",
                "...",
                "...",
                "...",
                "...",
                "...",
            )
        elif count > 100:
            break

    logger.info(
        "Max-RE line:---------------------------------------------------------------------------"
    )
    max_error = (
        float(torch.max(relative_diff).item()) if relative_diff.numel() > 0 else 0.0
    )
    m_idx_list = err_idx[relative_diff == max_error]
    for m_count, m_idx in enumerate(m_idx_list.tolist()):
        if m_count >= 3:
            break
        expect_value = float(expect_data[m_idx])
        real_value = float(real_data[m_idx])
        logger.info(
            "%08d \t %.7f \t %.7f \t %.7f \t %.7f",
            m_idx,
            expect_value,
            real_value,
            abs(expect_value - real_value),
            max_error,
        )
    logger.info(
        "---------------------------------------------------------------------------------------"
    )


def check_result(expect, npu_result):
    diff_thd = 0.005
    pct_thd = 0.005
    max_diff_hd = 10
    rtol = 0.005
    atol = 0.000025

    real_data = npu_result.detach().cpu().flatten()
    expect_data = (
        expect.detach().cpu().flatten()
        if isinstance(expect, torch.Tensor)
        else torch.as_tensor(expect).flatten()
    )

    if real_data.numel() == 0:
        logger.info(
            'The npu_output is [],and it is same as bm_output, the result of data_compare is "Pass"'
        )
        return "Pass", 100.0, 0

    if real_data.numel() != expect_data.numel():
        logger.info(
            "Error,the size of npu output[%s] and benchmark[%s] is not equal.",
            real_data.numel(),
            expect_data.numel(),
        )
        return "Failed", 0.0, 0

    start = 0
    end = real_data.numel() - 1
    if end < start:
        end = start
    split_count = end - start + 1 if end != start else 1
    max_error = 0.0

    overflows_count = int(
        torch.isinf(expect_data).sum().item() + torch.isnan(expect_data).sum().item()
    )
    if overflows_count > 0:
        logger.info(
            "Overflow,size:%s,benchmark_output:%s, %s",
            overflows_count,
            expect_data[torch.isinf(expect_data)][0:10],
            expect_data[torch.isnan(expect_data)][0:10],
        )

    logger.info("split_count:%s; max_diff_hd:%s;", float(split_count), max_diff_hd)

    real_float = real_data.to(torch.float32)
    expect_float = expect_data.to(torch.float32)

    if npu_result.dtype == torch.bfloat16:
        rtol = 0.0078125
        atol = 0.0001

    diff_result = torch.isclose(
        real_float, expect_float, rtol=rtol, atol=atol, equal_nan=True
    )
    err_idx = torch.nonzero(~diff_result, as_tuple=False).flatten()

    if expect_data.dtype == torch.bool:
        expect_data = expect_data.to(torch.int8)
        real_data = real_data.to(torch.int8)
        real_float = real_data.to(torch.float32)
        expect_float = expect_data.to(torch.float32)

    diff_abs = torch.abs(expect_float - real_float)
    b1 = torch.maximum(torch.abs(real_float), torch.abs(expect_float))
    b2 = float((1.0 / (1 << 14)) / diff_thd)
    b = torch.maximum(b1, torch.tensor(b2, dtype=torch.float32)) + _EPS
    err_diff = (diff_abs / b)[err_idx]
    err_diff = torch.where(
        torch.isnan(err_diff) | torch.isinf(err_diff),
        torch.tensor(float("inf"), dtype=torch.float32),
        err_diff,
    )

    fulfill_percent = float(split_count - err_idx.numel()) / float(split_count) * 100.0

    display_output_torch(real_data, expect_data, start, end)

    pct_thd = (1 - pct_thd) * 100.0
    result = "Pass" if (fulfill_percent >= pct_thd) else "Failed"

    if err_diff.numel() > 0:
        max_error = float(torch.max(err_diff).item())
        if max_error >= max_diff_hd:
            result = "Failed"

    logger.info(
        "---------------------------------------------------------------------------------------"
    )
    logger.info("Rtol   \t Atol   \t PctThd   \t PctRlt   \t Result")
    logger.info(
        "---------------------------------------------------------------------------------------"
    )
    logger.info(
        "%.4f    \t %.6f  \t %.2f%%   \t %.6f%%   \t %s",
        rtol,
        atol,
        pct_thd,
        fulfill_percent,
        result,
    )

    if err_diff.numel() > 0:
        logger.info(
            "Max-RelativeError is: %s. Threshold is: %s.", max_error, max_diff_hd
        )

    if result == "Failed":
        display_error_output_torch(real_data, expect_data, err_idx, err_diff)

    return result, fulfill_percent, max_error


# ==============================================================================
# 三方 cross_check 判据
# 以 fp64 bench 为共同基准, 比较 target(NPU) / third(CPU golden) 相对 bench 的误差,
# 判据为 "target 相对 bench 的误差不超过 third 相对 bench 误差的若干倍"。
# 判定算法与阈值对齐 attention/quant_flash_mla/tests/README_cross_check.md
# (L0: mare/mere/rmse = 10/2/2, 分母按 dtype 的 small_value 夹底)。
# ==============================================================================
CROSS_CHECK_TOLERANCE = {
    dtype: {
        "mare_ratio": 10.0,
        "mere_ratio": 2.0,
        "rmse_ratio": 2.0,
        "small_value": small_value,
        "small_value_atol": small_value_atol,
        "small_value_ratio": 2.0,
    }
    for dtype, small_value, small_value_atol in (
        ("float16", 2**-10, 2**-16),
        ("bfloat16", 2**-10, 2**-16),
        ("float32", 2**-14, 2**-30),
    )
}

_REL_EPS = 1e-7


def _flatten_fp32(data):
    if not isinstance(data, torch.Tensor):
        data = torch.as_tensor(data)
    return data.detach().cpu().to(torch.float32).flatten()


def _safe_div(num, den, err):
    """三比值除法: 分母按 small_value 夹底, 避免参考噪声低于噪声地板时比值被放大"""
    if math.isnan(num) or math.isnan(den):
        return float("inf")
    if math.isinf(num) and math.isinf(den):
        return 1.0
    if math.isinf(num):
        return float("inf")
    if num == 0 and den == 0:
        return 1.0
    return float(num / max(den, err))


def cross_check_result(bench, target, third, output_name="output"):
    """三方判据: target(NPU) / third(CPU golden) 相对 bench(fp64) 的误差比值

    bench: fp64 高精度基准
    target: NPU 输出
    third: CPU golden 输出
    返回 {"result", "detail", "metrics"}；result 为 "Pass"/"Failed"
    """
    g = _flatten_fp32(bench)
    t = _flatten_fp32(target)
    b = _flatten_fp32(third)

    if not (g.numel() == t.numel() == b.numel()):
        detail = (
            f"{output_name}: size mismatch bench={g.numel()} "
            f"target={t.numel()} third={b.numel()}"
        )
        logger.info("[CROSS_CHECK] %s", detail)
        return {"result": "Failed", "detail": detail, "metrics": {}}

    if g.numel() == 0:
        return {"result": "Pass", "detail": f"{output_name}: empty", "metrics": {}}

    dtype = (
        str(target.dtype).removeprefix("torch.")
        if isinstance(target, torch.Tensor)
        else "float32"
    )
    tol = CROSS_CHECK_TOLERANCE.get(dtype, CROSS_CHECK_TOLERANCE["bfloat16"])
    sv = tol["small_value"]
    atol = tol["small_value_atol"]

    # ---- Path A: 特殊位(NaN/Inf)三方交叉校验 ----
    t_fin, g_fin, b_fin = torch.isfinite(t), torch.isfinite(g), torch.isfinite(b)
    special = ~(t_fin & g_fin & b_fin)
    t_nan, g_nan, b_nan = torch.isnan(t), torch.isnan(g), torch.isnan(b)
    t_pinf, g_pinf, b_pinf = torch.isposinf(t), torch.isposinf(g), torch.isposinf(b)
    t_ninf, g_ninf, b_ninf = torch.isneginf(t), torch.isneginf(g), torch.isneginf(b)
    # target 与 third 一致或 target 与 bench 一致即通过
    tb_match = (
        (t_nan & b_nan)
        | (t_pinf & b_pinf)
        | (t_ninf & b_ninf)
        | (t_fin & b_fin & (t == b))
    )
    tg_match = (
        (t_nan & g_nan)
        | (t_pinf & g_pinf)
        | (t_ninf & g_ninf)
        | (t_fin & g_fin & (t == g))
    )
    special_fail = int((special & ~(tb_match | tg_match)).sum())

    finite = t_fin & g_fin & b_fin
    large = finite & (g.abs() >= sv)
    small = finite & (g.abs() < sv)

    metrics = {
        "mare": 1.0,
        "mere": 1.0,
        "rmse": 1.0,
        "small_over_target": 0,
        "small_over_third": 0,
        "small_ratio": 1.0,
        "special_fail": special_fail,
        "large_count": int(large.sum()),
        "small_count": int(small.sum()),
    }

    # ---- Path B-1: 大值域相对误差三比值 ----
    large_ok = True
    if large.any():
        denom = g[large].abs() + _REL_EPS
        rel_n = (t[large] - g[large]).abs() / denom
        rel_p = (b[large] - g[large]).abs() / denom
        diff_n = (t[large] - g[large]).abs()
        diff_p = (b[large] - g[large]).abs()
        mare = _safe_div(float(rel_n.max()), float(rel_p.max()), sv)
        mere = _safe_div(float(rel_n.mean()), float(rel_p.mean()), sv)
        rmse = _safe_div(
            float(torch.sqrt((diff_n**2).mean())),
            float(torch.sqrt((diff_p**2).mean())),
            sv,
        )
        metrics.update(mare=mare, mere=mere, rmse=rmse)
        large_ok = (
            mare <= tol["mare_ratio"]
            and mere <= tol["mere_ratio"]
            and rmse <= tol["rmse_ratio"]
        )

    # ---- Path B-2: 小值域超差计数比 ----
    if small.any():
        metrics["small_over_target"] = int(((t[small] - g[small]).abs() > atol).sum())
        metrics["small_over_third"] = int(((b[small] - g[small]).abs() > atol).sum())
        metrics["small_ratio"] = float(
            metrics["small_over_target"] / max(metrics["small_over_third"], 1)
        )

    passed = (
        large_ok
        and metrics["small_ratio"] <= tol["small_value_ratio"]
        and special_fail == 0
    )

    detail = (
        f"{output_name}: dtype={dtype} "
        f"mare={metrics['mare']:.4f}(<={tol['mare_ratio']:g}) "
        f"mere={metrics['mere']:.4f}(<={tol['mere_ratio']:g}) "
        f"rmse={metrics['rmse']:.4f}(<={tol['rmse_ratio']:g}) "
        f"small={metrics['small_over_target']}/{metrics['small_over_third']}"
        f"(<={tol['small_value_ratio']:g}) "
        f"special_fail={special_fail}"
    )
    logger.info("[CROSS_CHECK] %s → %s", detail, "Pass" if passed else "Failed")

    return {
        "result": "Pass" if passed else "Failed",
        "detail": detail,
        "metrics": metrics,
    }
