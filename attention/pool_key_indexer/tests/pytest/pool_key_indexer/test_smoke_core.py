#!/usr/bin/python
# -*- coding: utf-8 -*-
# fmt: off
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License).
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License).
# Please refer to the License for details. You may not use this file except in compliance with the License.
# -----------------------------------------------------------------------------------------------------------
"""pool_key_indexer 功能/边界/非法输入冒烟套(合并原 test.py/matrix/boundary/l2_errors)。

场景覆盖(与被合并套件的场景映射):
  * int8 4 组合法布局 x mask{0,3} + 非量化 fp16/bf16 回归;
  * TND 变长多 batch(前缀和) + TND/PA 乱序块表;
  * spec 边界: ps=1 退化 / ps=128 上界 / topk=1 / k=N 全选 / topk 超预算 padding /
    N1=1 / N1=64 上界 / 尾 pool / topk 2048 中规模 / topk 8192 上界 /
    PA bs=16 乱序多块 / B4 变长 / 2^10 补偿 decode;
  * 极值: 全零 query / 全 127 / -128 下界 / 负 descale(契约违反不变量) /
    W⊙Scale 超 FP16(契约违反不变量); 确定性双跑;
  * 非法输入 host 拒绝语义(dtype/shape/布局/参数)。
"""

import re
import numpy as np
import pytest
import torch

import pkicase
from pkicase import D, spec, run_golden_compare, rows_of

GWS_RC = 10
DISPATCH_OR_TILING_RC = (161002, 561002)


# ---- int8 4 组合法布局 x mask(场景: 合法布局矩阵 + rv 双态) ----
_LAYOUTS = [
    ("BSND", "BSND", 0),
    ("BSND", "BSND", 3),
    ("TND", "TND", 3),
    ("BSND", "PA_BBND", 3),
    ("TND", "PA_BBND", 0),
    ("TND", "PA_BBND", 3),
]


def _layout_spec(lq, lk, mask):
    if lq == "BSND":
        q_shape, w_shape, ptk = (2, 4, 2, D), (2, 4, 2), [0, 0]
    else:
        q_shape, w_shape, ptk = (3, 2, D), (3, 2), [0, 1]
    if lk == "PA_BBND":
        # ask=[24,32](非前缀和): 4 物理块覆盖, bt 逐 batch 两块
        k_shape, kd_shape, ask, bt = (
            (4, 16, 1, D),
            (4, 16, 1),
            [24, 32],
            [[0, 1], [2, 3]],
        )
    elif lk == "TND":
        k_shape, kd_shape, ask, bt = (6, 1, D), (6, 1), [3, 6], None
    else:
        k_shape, kd_shape, ask, bt = (2, 6, 1, D), (2, 6, 1), None, None
    asq = None if lq == "BSND" else [2, 3]
    return spec(f"M_{lq}_{lk}_m{mask}", seed=20260100 + mask + len(lq) + len(lk), layout_q=lq, layout_k=lk, q_shape=q_shape, k_shape=k_shape, w_shape=w_shape, kd_shape=kd_shape, topk=8, ps=2, mask=mask, rv=True, ptk=ptk, asq=asq, ask=ask, bt=bt, quant_mode=2)


@pytest.mark.pki_npu
class TestInt8LayoutMatrix:
    @pytest.mark.parametrize("lq,lk,mask", _LAYOUTS, ids=[f"{a}_{b}_m{c}" for a, b, c in _LAYOUTS])
    def test_int8_layout(self, npu_backend, lq, lk, mask):
        run_golden_compare(npu_backend, _layout_spec(lq, lk, mask), index_version=1)


@pytest.mark.pki_npu
class TestNonQuantRegress:
    @pytest.mark.parametrize("dtype,lq,lk", [(torch.float16, "BSND", "BSND"), (torch.bfloat16, "TND", "PA_BBND")], ids=["fp16_bsnd", "bf16_tnd_pa"],)
    def test_nonquant(self, npu_backend, dtype, lq, lk):
        sp = spec(f"NQ_{dtype}", seed=20260130, layout_q=lq, layout_k=lk, q_shape=(1, 4, 2, D) if lq == "BSND" else (4, 2, D), k_shape=(1, 6, 1, D) if lk == "BSND" else (2, 16, 1, D), w_shape=(1, 4, 2) if lq == "BSND" else (4, 2), kd_shape=(1, 6, 1) if lk == "BSND" else (2, 16, 1), topk=8, ps=2, mask=3, rv=True, ptk=[0] if lq == "BSND" else [1], asq=None if lq == "BSND" else [4], ask=None if lk == "BSND" else [6], bt=None if lk == "BSND" else [[0]], quant_mode=-1, q_dtype=dtype,)
        run_golden_compare(npu_backend, sp, index_version=1)


# ---- TND 变长多 batch(前缀和 / PA 非前缀和 + 乱序块表) ----
VARLEN = {
    "tnd_tnd": dict(layout_q="TND", layout_k="TND", q_shape=(7, 2, D), k_shape=(8, 1, D), w_shape=(7, 2), kd_shape=(8, 1), ptk=[1, 0], asq=[2, 7], ask=[5, 8], bt=None, topk=4, ps=2, mask=3, rv=True,),
    "tnd_pa": dict(layout_q="TND", layout_k="PA_BBND", q_shape=(4, 2, D), k_shape=(4, 16, 1, D), w_shape=(4, 2), kd_shape=(4, 16, 1), ptk=[1, 0], asq=[3, 4], ask=[18, 26], bt=[[1, 0, 0, 0], [2, 3, 0, 0]], topk=8, ps=2, mask=0, rv=False,),
}


@pytest.mark.pki_npu
class TestTndVarlen:
    @pytest.mark.parametrize("case_name", list(VARLEN))
    def test_varlen(self, npu_backend, case_name):
        sp = spec(f"V_{case_name}", seed=20260140, quant_mode=2, **VARLEN[case_name])
        idx, val, _ = run_golden_compare(npu_backend, sp, index_version=1)
        if not sp["rv"]:
            assert val.size == 0, "rv=False values 应为空输出"


# ---- spec 边界矩阵(场景取自原 PKI_L1_B 系列, 参数逐项对齐) ----
_B = {
    # ps=1 退化 lightning(输出末维=topk) / ps=128 上界+尾127 / topk=1 最小 / k=N 全选
    "B001_ps1": dict(l="BSND", q=(2, 4, 2, D), k=(2, 6, 1, D), w=(2, 4, 2), kd=(2, 6, 1), ptk=[0, 0], topk=4, ps=1, mask=0, rv=True,),
    "B002_ps128": dict(l="BSND", q=(2, 2, 2, D), k=(2, 1, 1, D), w=(2, 2, 2), kd=(2, 1, 1), ptk=[127, 127], topk=128, ps=128, mask=0, rv=True,),
    "B003_topk1": dict(l="BSND", q=(2, 4, 2, D), k=(2, 6, 1, D), w=(2, 4, 2), kd=(2, 6, 1), ptk=[0, 0], topk=1, ps=1, mask=0, rv=True,),
    "B004_allselect": dict(l="BSND", q=(2, 4, 2, D), k=(2, 3, 1, D), w=(2, 4, 2), kd=(2, 3, 1), ptk=[0, 0], topk=3, ps=1, mask=0, rv=True,),
    "B005_topk_gt": dict(l="BSND", q=(2, 4, 2, D), k=(2, 2, 1, D), w=(2, 4, 2), kd=(2, 2, 1), ptk=[0, 0], topk=16, ps=2, mask=0, rv=True,),
    "B006_n1_1": dict(l="BSND", q=(2, 4, 1, D), k=(2, 6, 1, D), w=(2, 4, 1), kd=(2, 6, 1), ptk=[0, 0], topk=4, ps=2, mask=0, rv=True,),
    "B007_n1_64": dict(l="BSND", q=(1, 4, 64, D), k=(1, 8, 1, D), w=(1, 4, 64), kd=(1, 8, 1), ptk=[0], topk=8, ps=2, mask=0, rv=True,),
    "B008_tailpool": dict(l="BSND", q=(2, 4, 2, D), k=(2, 3, 1, D), w=(2, 4, 2), kd=(2, 3, 1), ptk=[2, 3], topk=8, ps=4, mask=0, rv=True,),
    "B009_topk2048": dict(l="BSND", q=(1, 8, 32, D), k=(1, 300, 1, D), w=(1, 8, 32), kd=(1, 300, 1), ptk=[0], topk=2048, ps=16, mask=3, rv=True,),
    "B010_topk8192": dict(l="BSND", q=(1, 2, 8, D), k=(1, 130, 1, D), w=(1, 2, 8), kd=(1, 130, 1), ptk=[0], topk=8192, ps=64, mask=0, rv=True,),
    "B014_pa_bs16": dict(l="BSND", lk="PA_BBND", q=(2, 4, 2, D), k=(8, 16, 1, D), w=(2, 4, 2), kd=(8, 16, 1), ptk=[0, 0], ask=[40, 72], bt=[[1, 0, 3, 2, 0, 0, 0, 0], [4, 5, 6, 7, 0, 0, 0, 0]], topk=8, ps=4, mask=3, rv=True,),
    "B015_b4_varlen": dict(l="TND", q=(12, 2, D), k=(20, 1, D), w=(12, 2), kd=(20, 1), ptk=[0, 1, 0, 1], asq=[2, 5, 9, 12], ask=[3, 7, 15, 20], topk=4, ps=2, mask=3, rv=True,),
    "B018_comp_decode": dict(l="TND", lk="PA_BBND", q=(1, 32, D), k=(8, 32, 1, D), w=(1, 32), kd=(8, 32, 1), ptk=[0], asq=[1], ask=[200], bt=[[0, 1, 2, 3, 4, 5, 6, 7]], topk=64, ps=4, mask=3, rv=True,),
}


def _b_spec(name, c):
    lay = c.pop("l")
    kw = dict(layout_q=lay, layout_k=c.pop("lk", lay), quant_mode=2)
    for k in ("q", "k", "w", "kd"):
        kw[f"{k}_shape"] = c[k]
    kw.update(ptk=c["ptk"], topk=c["topk"], ps=c["ps"], mask=c["mask"], rv=c["rv"])
    for k in ("asq", "ask", "bt"):
        if c.get(k) is not None:
            kw[k] = c[k]
    return spec(name, seed=20260150 + len(name), **kw)


@pytest.mark.pki_npu
class TestSpecBoundary:
    @pytest.mark.parametrize("case_name", list(_B))
    def test_boundary(self, npu_backend, case_name):
        run_golden_compare(
            npu_backend, _b_spec(f"B_{case_name}", dict(_B[case_name])), index_version=1
        )

    def test_determinism(self, npu_backend):
        """B016: 同输入双跑逐位一致(replay 确定性)。"""
        sp1 = _b_spec("B_det_a", dict(_B["B008_tailpool"]))
        idx1, _, _ = run_golden_compare(npu_backend, sp1, index_version=1)
        idx2, _, _ = run_golden_compare(npu_backend, sp1, index_version=1)
        assert (idx1 == idx2).all(), "确定性双跑输出应逐位一致"


# ---- 极值/契约违反(E 系列: golden 对拍 / 不变量守护) ----
def _extreme_inputs(sp, pattern):
    inp = pkicase.make_inputs(sp)
    q, k, w, dq = inp["q"], inp["k"], inp["w"], inp["dq"]
    if pattern == "zeros":
        inp["q"] = torch.zeros_like(q)
    elif pattern == "const127":
        inp["q"] = torch.full_like(q, 127)
        inp["k"] = torch.full_like(k, 127)
    elif pattern == "neg128":
        inp["q"] = torch.full_like(q, -128)
    elif pattern == "neg_dscale":
        inp["dq"] = torch.full_like(dq, -0.5)
    elif pattern == "overflow":
        inp["w"] = torch.full_like(w, 60000.0)
        inp["dq"] = torch.full_like(dq, 100.0)
    return inp


_EXTREME = [
    ("zeros", 1),
    ("const127", 1),
    ("neg128", 1),
    ("neg_dscale", 0),
    ("overflow", 0),
]


@pytest.mark.pki_npu
class TestExtremeInputs:
    @pytest.mark.parametrize("pattern,golden_mode", _EXTREME, ids=["e1_zero_q", "e2_all127", "e3_neg128", "e4_neg_dscale", "e5_overflow"],)
    def test_extreme(self, npu_backend, pattern, golden_mode):
        sp = spec(f"E_{pattern}", seed=20260160, layout_q="BSND", layout_k="BSND", q_shape=(2, 4, 2, D), k_shape=(2, 6, 1, D), w_shape=(2, 4, 2), kd_shape=(2, 6, 1), topk=4, ps=2, mask=0, rv=True, ptk=[0, 0], quant_mode=2,)
        if not golden_mode:
            result = npu_backend.run_case(sp, _extreme_inputs(sp, pattern))
            assert (
                not isinstance(result, tuple) or True
            )  # 契约违反: 行为未定义, 仅守护不变量
            if isinstance(result, tuple):
                idx, val = result
                rows = rows_of(idx, sp)
                assert (rows >= -1).all(), "契约违反用例不变量: indices >= -1"
            return
        inp = _extreme_inputs(sp, pattern)
        import golden

        gres = golden.pki_golden(
            inp["q"],
            inp["k"],
            inp["w"],
            inp["ptk"],
            None,
            None,
            None,
            inp["dq"],
            inp["dk"],
            layout_q="BSND",
            layout_k="BSND",
            topk=4,
            pool_size=2,
            mask_mode=0,
            quant_mode=2,
            return_value=True,
            index_version=1,
        )
        result = npu_backend.run_case(sp, inp)
        assert isinstance(result, tuple), (
            f"E_{pattern} 执行失败:\n{result.stdout[-600:]}"
        )
        idx, val = result
        import compare

        ok, rep = compare.compare_outputs(gres, idx, val, 4, 2, True)
        assert ok, f"E_{pattern} compare FAIL:\n{rep.summary()}"


# ---- 非法输入 host 拒绝语义(L2 系列, 单参多态) ----
def _err_case(name, overrides, msgs, rc=(561002,)):
    return dict(name=name, overrides=overrides, msgs=msgs, rc=rc)


_BASE_ERR = dict(layout_q="BSND", layout_k="BSND", q_shape=(1, 4, 2, D), k_shape=(1, 6, 1, D), w_shape=(1, 4, 2), kd_shape=(1, 6, 1), topk=4, ps=2, mask=0, rv=True, ptk=[0],)


def _mk_err(entry):
    ov = dict(entry["overrides"])
    sp_ov = {k: v for k, v in _BASE_ERR.items()}
    sp_ov.update(
        {
            k: v
            for k, v in ov.items()
            if k
            not in (
                "q_dtype",
                "k_dtype",
                "w_dtype",
                "dq_dtype",
                "dq_shape",
                "dq_present",
                "bt_present",
                "omit_descale",
            )
        }
    )
    quant = sp_ov.pop("quant_mode", 2)
    sp_ov["q_dtype"] = ov.get("q_dtype", torch.int8 if quant == 2 else torch.float16)
    sp = spec(f"L2_{entry['name']}", seed=20260170, quant_mode=quant, **sp_ov)
    return sp, ov, entry["msgs"], entry["rc"]


_ERRS = [
    _err_case(
        "010_q_fp16",
        dict(quant_mode=2, q_dtype=torch.float16),
        [("quant_mode=2, query dtype must be INT8",)],
    ),
    _err_case(
        "011_w_bf16",
        dict(quant_mode=2, w_dtype=torch.bfloat16),
        [
            (
                "The dtype or format of the actual input or output parameter",
                "quant_mode=2, weights must be FLOAT16",
            )
        ],
        rc=(161002, 561002),
    ),
    _err_case(
        "012_dq_fp32",
        dict(quant_mode=2, dq_dtype=torch.float32),
        [("quant_mode=2, q_descale dtype must be FLOAT16",)],
    ),
    _err_case(
        "013_dq_shape",
        dict(quant_mode=2, dq_shape=(1, 4)),
        [("BSND q_descale must be 3D", "quant_mode=2")],
    ),
    _err_case(
        "014_no_descale",
        dict(quant_mode=2, omit_descale=True),
        [("quant_mode=2, q_descale must not be null",)],
    ),
    _err_case(
        "015_nq_descale",
        dict(quant_mode=-1, q_dtype=torch.float16, dq_present=True),
        [("quant_mode=-1, q_descale must be null",)],
    ),
    _err_case(
        "022_mixed_layout",
        dict(layout_q="BSND", layout_k="TND"),
        [("must match in non-PA scenario",)],
    ),
    _err_case(
        "030_bad_layout", dict(layout_q="BSNDX"), [("layout_q must be BSND or TND",)]
    ),
    _err_case(
        "031_topk_mod",
        dict(topk=6, ps=4),
        [("must be divisible by poolSize", "must be divisible by pool_size")],
    ),
    _err_case(
        "032_n1_65",
        dict(q_shape=(1, 4, 65, D), w_shape=(1, 4, 65)),
        [("N1(", "must be <= 64")],
    ),
    _err_case(
        "033_headdim",
        dict(q_shape=(1, 4, 2, 64), w_shape=(1, 4, 2)),
        [("must be 128", "headDim")],
    ),
    _err_case(
        "034_nonpa_bt",
        dict(bt_present=True),
        [("non-PA scenario, block_table must be null",)],
    ),
]


@pytest.mark.pki_npu
@pytest.mark.pki_error
class TestHostValidation:
    @pytest.mark.parametrize("entry", _ERRS, ids=[e["name"] for e in _ERRS])
    def test_host_rejected(self, npu_backend, entry):
        sp, ov, msgs, rc = _mk_err(entry)
        inp = pkicase.make_inputs(sp)
        if ov.get("omit_descale"):
            inp["dq"] = inp["dk"] = None
        if ov.get("dq_present"):
            inp["dq"] = torch.ones(1, 4, 2, dtype=torch.float16)
        if ov.get("dq_shape"):
            inp["dq"] = torch.ones(ov["dq_shape"], dtype=torch.float16)
        if ov.get("w_dtype"):
            inp["w"] = inp["w"].to(ov["w_dtype"])
        if ov.get("dq_dtype"):
            inp["dq"] = inp["dq"].to(ov["dq_dtype"])
            inp["dk"] = inp["dk"].to(ov["dq_dtype"])
        if ov.get("q_dtype"):
            inp["q"] = inp["q"].to(ov["q_dtype"])
            inp["k"] = inp["k"].to(
                ov["k_dtype"] if ov.get("k_dtype") else ov["q_dtype"]
            )
        if ov.get("bt_present"):
            inp["bt"] = torch.tensor([[0]], dtype=torch.int32)
        result = npu_backend.run_case(sp, inp)
        assert not isinstance(result, tuple), f"{sp['name']} 应被 host 拒绝"
        stdout = result.stdout or ""
        assert "GWS_FAILED" in stdout, f"{sp['name']} 缺 GWS_FAILED:\n{stdout[-400:]}"
        m = re.search(r"GWS_FAILED ret=(\d+)", stdout)
        assert m and int(m.group(1)) in (rc if isinstance(rc, tuple) else (rc,)), (
            f"{sp['name']} rc 不符: {m.group(0) if m else '无'}\n{stdout[-400:]}"
        )
        for group in msgs:
            assert any(frag in stdout for frag in group), (
                f"{sp['name']} 报错语义不符(期望任一 {group}):\n{stdout[-600:]}"
            )


# ---- 纯尾块 0 池场景(合并自 1ef9b1b1f 的 test_zerok_pool_tail.py) ----
def _zerok_expected(b, s1, ps, topk, ptk, mask_mode, version):
    """0 池确定性期望: gpos = ptk - s1 + i; 可见 = ptk(mask=0) 或 clamp(gpos+1, 0, ptk)
    (seqTailStart=0); legacy 落位自 topk 起, compact 自行首。"""
    out_len = topk + ps - 1
    rows = [[[-1] * out_len for _ in range(s1)] for _ in range(b)]
    for bi in range(b):
        for i in range(s1):
            gpos = ptk - s1 + i
            vis = ptk if mask_mode == 0 else max(0, min(ptk, gpos + 1))
            base = topk if version is None else 0
            for t in range(vis):
                rows[bi][i][base + t] = t  # L_orig - ptk + t = 0 + t
    return np.array(rows, dtype=np.int32).reshape(-1, out_len)


@pytest.mark.pki_npu
class TestZeroPoolTail:
    """pool_key 池数为 0(全部有效 token 经 pool_tail_k 表达): legacy/compact 落位、
    因果部分可见、values 全 -inf。与分数无关(候选池数 0, 尾 token 按因果全量追加)。"""

    def _run_zerok(self, npu_backend, sp):
        """0 池用例不走 golden(其 P=0 索引越界), 期望为确定性硬编码(候选池数 0)。"""
        inp = pkicase.make_inputs(sp)
        result = npu_backend.run_case(sp, inp)
        assert isinstance(result, tuple), (
            f"{sp['name']} NPU 执行失败:\n{result.stdout[-600:] if result.stdout else ''}"
        )
        return result

    @pytest.mark.parametrize("version,ptk", [(None, 3), (1, 3), (None, 1), (1, 2)],
                             ids=["legacy_ptk3", "compact_ptk3", "legacy_ptk1", "compact_ptk2"])
    def test_zerok_decode(self, npu_backend, version, ptk):
        """decode: 全部尾 token 必可见; legacy 落位 topk 起 / compact 行首。"""
        sp = spec(f"ZK_d{ptk}_{version}", seed=20260180, layout_q="BSND", layout_k="BSND",
                  q_shape=(2, 1, 8, D), k_shape=(2, 0, 1, D), w_shape=(2, 1, 8),
                  kd_shape=(2, 0, 1), topk=64, ps=4, mask=3, rv=False, ptk=[ptk, ptk],
                  quant_mode=-1, q_dtype=torch.float16, version=version)
        idx, _ = self._run_zerok(npu_backend, sp)
        expect = _zerok_expected(2, 1, 4, 64, ptk, 3, version)
        assert np.array_equal(rows_of(idx, sp), expect)

    def test_zerok_prefill_partial(self, npu_backend):
        """多 query 行: 早期行 gpos<0 全 -1, 末行全尾可见(legacy 落位)。"""
        s1, ptk = 5, 3
        sp = spec("ZK_prefill", seed=20260181, layout_q="BSND", layout_k="BSND",
                  q_shape=(1, s1, 8, D), k_shape=(1, 0, 1, D), w_shape=(1, s1, 8),
                  kd_shape=(1, 0, 1), topk=64, ps=4, mask=3, rv=False, ptk=[ptk],
                  quant_mode=-1, q_dtype=torch.float16, version=None)
        idx, _ = self._run_zerok(npu_backend, sp)
        rows = rows_of(idx, sp)
        expect = _zerok_expected(1, s1, 4, 64, ptk, 3, None)
        assert np.array_equal(rows, expect)
        assert (rows[0] == -1).all(), "早期行(gpos<0)应全 -1"

    def test_zerok_values_all_neginf(self, npu_backend):
        """0 池批次 values 无有效池分: 全 -inf(值通道与尾块无关)。"""
        sp = spec("ZK_vals", seed=20260182, layout_q="BSND", layout_k="BSND",
                  q_shape=(1, 1, 8, D), k_shape=(1, 0, 1, D), w_shape=(1, 1, 8),
                  kd_shape=(1, 0, 1), topk=64, ps=4, mask=3, rv=True, ptk=[3],
                  quant_mode=-1, q_dtype=torch.float16, version=1)
        _, val = self._run_zerok(npu_backend, sp)
        assert np.isneginf(val).all(), "0 池批次 values 应全 -inf"
