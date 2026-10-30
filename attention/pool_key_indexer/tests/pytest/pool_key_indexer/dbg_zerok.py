#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License).
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See License in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""0 池(纯尾块)场景缺陷特征诊断: NPU 输出 vs 合同逐行 diff。"""

import sys

sys.path.insert(0, ".")
import numpy as np
import torch

import pkicase
from pkicase import D

from conftest import _cann_home, _vendor_dir, _build_runner, AclnnBackend, TEST_DIR


def contract(b, s1, topk, ps, ptk, mask, version):
    """合同: 候选池数 0, 尾 token 按因果可见; legacy 落位 topk, compact 落位行首。"""
    out_len = topk + ps - 1
    rows = np.full((b, s1, out_len), -1, dtype=np.int32)
    for bi in range(b):
        for i in range(s1):
            gpos = ptk - s1 + i
            vis = ptk if mask == 0 else max(0, min(ptk, gpos + 1))
            base = topk if version is None else 0
            for t in range(vis):
                rows[bi, i, base + t] = t
    return rows.reshape(b, -1)


def main():
    cann = _cann_home()
    vendor = _vendor_dir(cann)
    backend = AclnnBackend(
        _build_runner(cann, vendor), 4, TEST_DIR / "build" / "cases_dbg"
    )

    cases = [
        (1, 1, None, 1),
        (2, 1, None, 1),
        (2, 1, None, 3),
        (2, 1, 1, 1),
        (1, 5, None, 3),
        (2, 5, None, 3),
        (2, 5, 1, 3),
        (1, 1, 1, 3),
        (3, 1, None, 1),
    ]
    for B, s1, version, ptk in cases:
        mode = "legacy" if version is None else "compact"
        sp = pkicase.spec(
            f"dbg{B}_{s1}_{mode}_{ptk}",
            seed=20260180,
            layout_q="BSND",
            layout_k="BSND",
            q_shape=(B, s1, 8, D),
            k_shape=(B, 0, 1, D),
            w_shape=(B, s1, 8),
            kd_shape=(B, 0, 1),
            topk=64,
            ps=4,
            mask=3,
            rv=False,
            ptk=[ptk] * B,
            quant_mode=-1,
            q_dtype=torch.float16,
            version=version,
        )
        inp = pkicase.make_inputs(sp)
        result = backend.run_case(sp, inp)
        if not isinstance(result, tuple):
            print(
                f"B={B} s1={s1} {mode} ptk={ptk}: RUN FAILED\n{(result.stdout or '')[-300:]}"
            )
            continue
        idx, _ = result
        rows = idx.reshape(B, -1)
        exp = contract(B, s1, 64, 4, ptk, 3, version)
        diffs = []
        for r in range(B):
            d = np.where(rows[r] != exp[r])[0]
            if d.size:
                diffs.append(
                    f"row{r}: pos={d[:10].tolist()} npu={rows[r][d[:10]].tolist()} exp={exp[r][d[:10]].tolist()}"
                )
        status = "OK" if not diffs else "MISMATCH"
        print(f"B={B} s1={s1} {mode} ptk={ptk}: {status}")
        for d in diffs:
            print(f"    {d}")


if __name__ == "__main__":
    main()
