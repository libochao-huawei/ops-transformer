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
"""pytest 冒烟套共享用例构造: spec 字典 + 输入生成(数量口径与 golden 对拍一致)。"""

import torch

D = 128


def spec(
    name,
    seed,
    layout_q,
    layout_k,
    q_shape,
    k_shape,
    w_shape,
    kd_shape,
    topk,
    ps,
    mask,
    rv,
    ptk,
    asq=None,
    ask=None,
    bt=None,
    quant_mode=2,
    q_dtype=torch.int8,
    version=None,
    expect_error=False,
):
    """用例 spec: version 键承载 sparse_indices_mode(0=legacy/1=compact), None 走 V1。"""
    return dict(name=name, seed=seed, layout_q=layout_q, layout_k=layout_k, q_shape=q_shape, k_shape=k_shape, w_shape=w_shape, kd_shape=kd_shape, topk=topk, ps=ps, mask=mask, rv=rv, ptk=ptk, asq=asq, ask=ask, bt=bt, quant_mode=quant_mode, q_dtype=q_dtype, version=version, expect_error=expect_error,)


def make_inputs(sp):
    """按 spec 生成 CPU 输入(int8: q/k~U[-127,127], w/descale=FP16; 非量化: 全体同 dtype ~U[-1,1])。"""
    g = torch.Generator().manual_seed(sp["seed"])
    if sp["quant_mode"] == 2:
        inp = dict(q=torch.randint(-127, 128, sp["q_shape"], dtype=torch.int8, generator=g), k=torch.randint(-127, 128, sp["k_shape"], dtype=torch.int8, generator=g), w=(torch.rand(sp["w_shape"], generator=g) * 2 - 1).half(), ptk=None, asq=None, ask=None, bt=None, dq=(torch.rand(sp["w_shape"], generator=g) * 0.099 + 0.001).half(), dk=(torch.rand(sp["kd_shape"], generator=g) * 0.099 + 0.001).half(),)
    else:
        dt = sp["q_dtype"]
        inp = dict(q=(torch.rand(sp["q_shape"], generator=g) * 2 - 1).to(dt), k=(torch.rand(sp["k_shape"], generator=g) * 2 - 1).to(dt), w=(torch.rand(sp["w_shape"], generator=g) * 2 - 1).to(dt), ptk=None, asq=None, ask=None, bt=None, dq=None, dk=None,)
    if sp["ptk"] is not None:
        inp["ptk"] = torch.tensor(sp["ptk"], dtype=torch.int64)
    for key in ("asq", "ask", "bt"):
        if sp[key] is not None:
            inp[key] = torch.tensor(
                sp[key], dtype=torch.int32 if key == "bt" else torch.int64
            )
    return inp


def rows_of(idx, sp):
    """indices 重排为逐 query 行 [.., topk+ps-1]。"""
    return idx.reshape(-1, sp["topk"] + sp["ps"] - 1)


def run_golden_compare(backend, sp, index_version):
    """golden 对拍一步式入口, 返回 (npu_idx, npu_val, golden_res)。"""
    import golden

    inp = make_inputs(sp)
    gres = golden.pki_golden(
        inp["q"],
        inp["k"],
        inp["w"],
        inp["ptk"],
        inp["asq"],
        inp["ask"],
        inp["bt"],
        inp["dq"],
        inp["dk"],
        layout_q=sp["layout_q"],
        layout_k=sp["layout_k"],
        topk=sp["topk"],
        pool_size=sp["ps"],
        mask_mode=sp["mask"],
        quant_mode=sp["quant_mode"],
        return_value=sp["rv"],
        index_version=index_version,
    )
    result = backend.run_case(sp, inp)
    assert isinstance(result, tuple), (
        f"{sp['name']} NPU 执行失败:\n{result.stdout[-800:] if result.stdout else ''}"
    )
    import compare

    idx, val = result
    ok, rep = compare.compare_outputs(gres, idx, val, sp["topk"], sp["ps"], sp["rv"])
    assert ok, f"{sp['name']} compare FAIL:\n{rep.summary()}"
    return idx, val, gres
