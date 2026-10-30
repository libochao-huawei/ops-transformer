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
"""int8 交付特性 + 多核/LD 机制冒烟套(合并原 c8_features F2/F4 + multicore + ld_defect_probe)。

场景覆盖:
  * int8 x V2 compact 交联(BSND ps{4,3,16} 三条展开路径 + TND/PA decode);
  * int8 2^10 补偿哨兵(values 与 golden/1024 显式不接近);
  * S2 多核切分三态: 单并行单元 LD 跨核归并(128K decode 缩影) / 多 batch 批并行不切分 /
    多 batch 双段跨核;
  * LD 跨调用残留: 同 runner 进程大 shape -> 小 shape 序列, 小 shape 与 fresh 逐位一致;
  * LD 链式部分归并(acc<核数)链长=3 int8。
"""

import numpy as np
import pytest
import torch

import pkicase
from pkicase import D, spec, run_golden_compare, rows_of

CORE_910B3 = 20


@pytest.mark.pki_npu
class TestInt8CompactCombo:
    @pytest.mark.parametrize("tag,lq,lk,ps,k_shape,kd_shape,ptk,asq,ask,bt", [ ( "bsnd_ps4", "BSND", "BSND", 4, (1, 3, 1, D), (1, 3, 1), [3], None, None, None, ), ( "bsnd_ps3", "BSND", "BSND", 3, (1, 3, 1, D), (1, 3, 1), [2], None, None, None, ), ( "bsnd_ps16", "BSND", "BSND", 16, (1, 1, 1, D), (1, 1, 1), [15], None, None, None, ), ( "tnd_pa", "TND", "PA_BBND", 4, (4, 16, 1, D), (4, 16, 1), [0, 3], [1, 2], [16, 32], [[0, 1], [2, 3]], ), ], ids=["bsnd_ps4", "bsnd_ps3", "bsnd_ps16", "tnd_pa"],)
    def test_combo(
        self, npu_backend, tag, lq, lk, ps, k_shape, kd_shape, ptk, asq, ask, bt
    ):
        q_shape = (1, 4, 2, D) if lq == "BSND" else (2, 8, D)
        w_shape = (1, 4, 2) if lq == "BSND" else (2, 8)
        topk = {4: 16, 3: 12, 16: 32}[ps]
        sp = spec(f"I8C_{tag}", seed=20260300 + ps, layout_q=lq, layout_k=lk, q_shape=q_shape, k_shape=k_shape, w_shape=w_shape, kd_shape=kd_shape, topk=topk, ps=ps, mask=3 if lk == "PA_BBND" else 0, rv=True, ptk=ptk, asq=asq, ask=ask, bt=bt, quant_mode=2, version=1,)
        idx, _, _ = run_golden_compare(npu_backend, sp, index_version=2)
        for r, row in enumerate(rows_of(idx, sp)):
            n_valid = int((row >= 0).sum())
            assert not np.any(row[:n_valid] == -1) and not np.any(
                row[n_valid:] != -1
            ), f"{tag} row{r}: 紧凑布局存在中间空洞"

    def test_compensation_sentinel(self, npu_backend):
        """int8 2^10 补偿哨兵: values 与 golden 一致, 且与 golden/1024(未补偿形态)不接近。"""

        sp = spec("I8S_comp", 20260310, layout_q="BSND", layout_k="BSND", q_shape=(1, 4, 2, D), k_shape=(1, 8, 1, D), w_shape=(1, 4, 2), kd_shape=(1, 8, 1), topk=16, ps=4, mask=0, rv=True, ptk=[0], quant_mode=2,)
        _, val, gres = run_golden_compare(npu_backend, sp, index_version=1)
        gvals = np.asarray(gres.values, dtype=np.float32)
        valid = np.isfinite(gvals) & (np.abs(gvals) > 1e-4)
        assert valid.any(), "golden 有效值过小, 哨兵无判别力"
        npu_v = np.asarray(val, dtype=np.float32)[valid]
        assert not np.allclose(npu_v, gvals[valid] / 1024.0, rtol=0.05, atol=1e-6), (
            "values 呈 2^-10 未补偿形态(补偿语义回归)"
        )
        assert np.allclose(npu_v, gvals[valid], rtol=0.02, atol=2.5e-5), (
            "values 与补偿后 golden 偏差超限"
        )


def _permuted_bt(g, logical, physical):
    ids = torch.randperm(physical, generator=g)[:logical]
    bt = torch.zeros(1, physical, dtype=torch.int32)
    bt[0, :logical] = ids
    return bt.tolist()


@pytest.mark.pki_npu
class TestMultiCoreLD:
    def test_mc1_ld_split(self, npu_backend):
        """128K decode 缩影: S2=20480 pool(40 块) / 20 核 = 2 块/核, LD 归并 20 个局部 TopK。"""
        g = torch.Generator().manual_seed(20260320)
        logical, bs = 640, 32
        sp = spec("MC1_ld", 20260320, layout_q="TND", layout_k="PA_BBND", q_shape=(1, 32, D), k_shape=(logical + 8, bs, 1, D), w_shape=(1, 32), kd_shape=(logical + 8, bs, 1), topk=2048, ps=4, mask=3, rv=True, ptk=[0], asq=[1], ask=[logical * bs], bt=_permuted_bt(g, logical, logical + 8), quant_mode=2,)
        assert (sp["ask"][0] + 511) // 512 >= CORE_910B3, "应启用多核切分"
        run_golden_compare(npu_backend, sp, index_version=1)

    def test_mc2_batch_parallel_nosplit(self, npu_backend):
        """B=24 >= 20 核: batch 并行优先, S2 跨核规避保持单核完整段。"""
        b = 24
        sp = spec("MC2_nosplit", 20260321, layout_q="BSND", layout_k="BSND", q_shape=(b, 1, 32, D), k_shape=(b, 520, 1, D), w_shape=(b, 1, 32), kd_shape=(b, 520, 1), topk=4, ps=2, mask=0, rv=True, ptk=[0] * b, quant_mode=2,)
        run_golden_compare(npu_backend, sp, index_version=1)

    def test_mc5_dual_span_split(self, npu_backend):
        """B=2 TND 各 2048 pool(4 块): 两个 LD 段首核分别归并(rv=False 纯 indices)。"""
        ppb = 2048
        sp = spec("MC5_dual", 20260322, layout_q="TND", layout_k="TND", q_shape=(2, 32, D), k_shape=(2 * ppb, 1, D), w_shape=(2, 32), kd_shape=(2 * ppb, 1), topk=2048, ps=4, mask=3, rv=False, ptk=[0, 0], asq=[1, 2], ask=[ppb, 2 * ppb], quant_mode=2,)
        idx, val, _ = run_golden_compare(npu_backend, sp, index_version=1)
        assert val.size == 0

    def test_ld_seq_residue(self, npu_backend):
        """跨调用残留: 同进程 [L=20 块写满 LD ws -> L=6 块], 小 shape 与 fresh 结果逐位一致。"""

        def mk(name, blocks):
            return spec(name, 20260323, layout_q="TND", layout_k="TND", q_shape=(1, 32, D), k_shape=(blocks * 512, 1, D), w_shape=(1, 32), kd_shape=(blocks * 512, 1), topk=2048, ps=4, mask=3, rv=True, ptk=[0], asq=[1], ask=[blocks * 512], quant_mode=2,)

        sp_big, sp_small = mk("SEQ_big", 20), mk("SEQ_small", 6)
        results = npu_backend.run_seq(
            [
                (sp_big, pkicase.make_inputs(sp_big)),
                (sp_small, pkicase.make_inputs(sp_small)),
            ]
        )
        import golden
        import compare

        for sp, (case_dir, rc, stdout) in zip((sp_big, sp_small), results):
            assert rc == 0, f"{sp['name']} 序列执行失败 rc={rc}:\n{stdout[-600:]}"
        idx_small, val_small = npu_backend._parse_outputs(sp_small, results[1][0])
        i = pkicase.make_inputs(sp_small)
        gres = golden.pki_golden(
            i["q"],
            i["k"],
            i["w"],
            i["ptk"],
            i["asq"],
            i["ask"],
            i["bt"],
            i["dq"],
            i["dk"],
            layout_q="TND",
            layout_k="TND",
            topk=2048,
            pool_size=4,
            mask_mode=3,
            quant_mode=2,
            return_value=True,
            index_version=1,
        )
        ok, rep = compare.compare_outputs(gres, idx_small, val_small, 2048, 4, True)
        assert ok, f"序列末条与 fresh golden 不一致(跨调用残留):\n{rep.summary()}"

    def test_ld_chain_partial_merge(self, npu_backend):
        """P-① 链式部分归并: B=1 decode, S2=3 块(链长=3, acc=2<4), int8。"""
        blocks = 3
        sp = spec("P1_chain3", 20260324, layout_q="TND", layout_k="TND", q_shape=(1, 32, D), k_shape=(blocks * 512, 1, D), w_shape=(1, 32), kd_shape=(blocks * 512, 1), topk=2048, ps=4, mask=0, rv=True, ptk=[0], asq=[1], ask=[blocks * 512], quant_mode=2,)
        run_golden_compare(npu_backend, sp, index_version=1)
