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
"""torch_extension e2e 冒烟套(压缩自原 test_torch_e2e, 场景对齐)。

场景覆盖:
  * E1 IntArrayRef 数组路径(BSND int8, ptk 值输入全 CPU);
  * E2 Tensor 变体(TND/PA, 值输入全 NPU, 无 D2H) + E2b 负向(weights=BF16 csrc 拒绝);
  * E3 PA pool_key 0 轴非连续(key_stride0 由 csrc 直读 stride(0), 中间块 padding);
  * E6 V2 compact(torch kwarg sparse_indices_mode=1)与 V1 布局互异/集合一致;
  * E7 torchair graph 模式: 动态长度 replay(覆盖 pc<sc / ==sc / >sc 三段),
    fullgraph 捕获 + 同形状二次调用逐位一致 + 逐长度尾池/前缀合同。
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parents[4]
E2E_SITE = REPO_ROOT / "torch_extension" / "build" / "e2e_site"
EXT_CACHE = REPO_ROOT / "torch_extension" / "build" / "torch_ext_cache"
D = 128
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(EXT_CACHE))
os.makedirs(EXT_CACHE, exist_ok=True)


def _ensure_e2e_site():
    if (E2E_SITE / "cann_ops_transformer_gjw_pki").is_dir():
        return
    import shutil
    import subprocess

    env = dict(os.environ, TORCH_EXTENSION_OPS="pool_key_indexer", TORCH_EXTENSION_VENDOR="gjw_pki",)
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "-n"],
        cwd=REPO_ROOT / "torch_extension",
        env=env,
        check=True,
    )
    whls = sorted(
        (REPO_ROOT / "torch_extension").glob("dist/cann_ops_transformer_gjw_pki-*.whl")
    )
    if not whls:
        raise RuntimeError("whl 构建失败: dist/ 下无产物")
    if E2E_SITE.exists():
        shutil.rmtree(E2E_SITE)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(E2E_SITE),
            str(whls[-1]),
        ],
        check=True,
    )


_ensure_e2e_site()
sys.path.insert(0, str(E2E_SITE))

import torch  # noqa: E402

import compare  # noqa: E402
import golden  # noqa: E402

_NPU_AVAILABLE = any(Path(f"/dev/davinci{i}").exists() for i in range(8))
_OP = None


def _load_op():
    """装载隔离 torch_extension(canary: 隔离包须先于共享包导入, schema 须含 quant_mode)。"""
    global _OP
    if _OP is not None:
        return _OP
    assert not any(
        m == "cann_ops_transformer" or m.startswith("cann_ops_transformer.")
        for m in sys.modules
    ), "共享包已先导入, 路由不可保证"

    op = torch.ops.cann_ops_transformer.pool_key_indexer
    assert "quant_mode" in str(op.default._schema), (
        "schema 缺 quant_mode, 疑似路由旧 csrc"
    )
    _OP = op
    return op


@pytest.fixture(scope="module")
def pki_device(request):
    if not _NPU_AVAILABLE:
        pytest.skip("no NPU device found (/dev/davinci*), skip e2e cases")
    device = int(request.config.getoption("--pki-device"))
    torch.npu.set_device(device)
    return device


def _to_npu(inp, keys):
    for key in keys:
        if inp[key] is not None:
            inp[key] = inp[key].npu()
    return inp


def _call_op(op, inp, layout_q, layout_k, topk, ps, mask, rv, quant_mode, mode=None):
    kwargs = {
        kw: inp[key]
        for kw, key in (
            ("actual_seq_q", "asq"),
            ("actual_seq_k", "ask"),
            ("block_table", "bt"),
            ("q_descale", "dq"),
            ("k_descale", "dk"),
        )
        if inp[key] is not None
    }
    if mode is not None:
        kwargs["sparse_indices_mode"] = mode  # 0=legacy / 1=compact(torch schema 同名)
    return op(
        inp["q"],
        inp["k"],
        inp["w"],
        inp["ptk"],
        **kwargs,
        layout_q=layout_q,
        layout_k=layout_k,
        topk=topk,
        pool_size=ps,
        mask_mode=mask,
        quant_mode=quant_mode,
        return_value=rv,
    )


def _golden_of(inp, layout_q, layout_k, topk, ps, mask, rv, quant_mode, iv=1):
    return golden.pki_golden(
        inp["q"],
        inp["k"],
        inp["w"],
        inp["ptk"],
        inp["asq"],
        inp["ask"],
        inp["bt"],
        inp["dq"],
        inp["dk"],
        layout_q=layout_q,
        layout_k=layout_k,
        topk=topk,
        pool_size=ps,
        mask_mode=mask,
        quant_mode=quant_mode,
        return_value=rv,
        index_version=iv,
    )


def _check(idx_out, val_out, gres, lead_shape, topk, ps, rv):
    idx = idx_out.cpu().numpy()
    val = val_out.cpu().numpy() if rv else np.zeros(0, dtype=np.float32)
    assert idx.dtype.name == "int32" and tuple(idx.shape) == tuple(lead_shape) + (
        topk + ps - 1,
    )
    ok, rep = compare.compare_outputs(gres, idx, val, topk, ps, rv)
    assert ok, f"compare FAIL:\n{rep.summary()}"
    return idx


def _c1_inputs():
    g = torch.Generator().manual_seed(20260101)
    return dict(q=torch.randint(-127, 128, (2, 4, 2, D), dtype=torch.int8, generator=g), k=torch.randint(-127, 128, (2, 6, 1, D), dtype=torch.int8, generator=g), w=(torch.rand((2, 4, 2), generator=g) * 50.0 - 25.0).half(), ptk=torch.tensor([0, 0], dtype=torch.int64), asq=None, ask=None, bt=None, dq=(torch.rand((2, 4, 2), generator=g) * 0.099 + 0.001).half(), dk=(torch.rand((2, 6, 1), generator=g) * 0.099 + 0.001).half(),)


@pytest.mark.pki_npu
class TestTorchE2E:
    def test_e1_array_path_int8(self, pki_device):
        """E1: IntArrayRef 数组路径(ptk 值输入留在 CPU, 其余 NPU)。"""
        inp = _to_npu(_c1_inputs(), ["q", "k", "w", "dq", "dk"])
        idx, val = _call_op(_load_op(), inp, "BSND", "BSND", 4, 2, 0, True, 2)
        _check(
            idx,
            val,
            _golden_of(inp, "BSND", "BSND", 4, 2, 0, True, 2),
            (2, 4),
            4,
            2,
            True,
        )

    def test_e2_tensor_variant_pa(self, pki_device):
        """E2: Tensor 变体(TND/PA 块表乱序, 值输入全 NPU 无 D2H)。"""
        g = torch.Generator().manual_seed(20260102)
        inp = dict(q=torch.randint(-127, 128, (1, 32, D), dtype=torch.int8, generator=g), k=torch.randint(-127, 128, (8, 32, 1, D), dtype=torch.int8, generator=g), w=(torch.rand((1, 32), generator=g) * 50.0 - 25.0).half(), ptk=torch.tensor([0], dtype=torch.int64), asq=torch.tensor([1], dtype=torch.int64), ask=torch.tensor([36], dtype=torch.int64), bt=torch.tensor([[5, 2]], dtype=torch.int32), dq=(torch.rand((1, 32), generator=g) * 0.099 + 0.001).half(), dk=(torch.rand((8, 32, 1), generator=g) * 0.099 + 0.001).half(),)
        _to_npu(inp, ["q", "k", "w", "dq", "dk", "bt", "ask"])
        idx, val = _call_op(_load_op(), inp, "TND", "PA_BBND", 64, 4, 3, False, 2)
        _check(
            idx,
            val,
            _golden_of(inp, "TND", "PA_BBND", 64, 4, 3, False, 2),
            (1,),
            64,
            4,
            False,
        )

    def test_e2b_weights_bf16_rejected(self, pki_device):
        """E2b 负向: mode=2 weights=BF16 被 csrc 拒绝。"""
        inp = _to_npu(_c1_inputs(), ["q", "k", "w", "dq", "dk"])
        inp["w"] = inp["w"].to(torch.bfloat16)
        with pytest.raises(RuntimeError, match="weights must be torch.float16"):
            _call_op(_load_op(), inp, "BSND", "BSND", 4, 2, 0, True, 2)

    def test_e3_pa_stride0_noncontig(self, pki_device):
        """E3: PA pool_key 0 轴非连续(stride0=4096, 逻辑块隔块放置), k_descale 连续。"""
        g = torch.Generator().manual_seed(20260103)
        contig0 = 16 * D
        stride0 = 2 * contig0
        q = torch.randint(-127, 128, (2, 4, 2, D), dtype=torch.int8, generator=g)
        k_logical = torch.randint(
            -127, 128, (4, 16, 1, D), dtype=torch.int8, generator=g
        )
        storage = torch.zeros((8, 16, 1, D), dtype=torch.int8)
        flat = storage.reshape(-1)
        for i in range(4):
            flat[i * stride0 : i * stride0 + contig0] = k_logical[i].reshape(-1)
        k_view = (
            flat.reshape(8, 16, 1, D)
            .npu()
            .as_strided((4, 16, 1, D), (stride0, D, D, 1))
        )
        assert not k_view.is_contiguous() and k_view.stride(0) == stride0
        inp = dict(q=q.npu(), k=k_view, w=(torch.rand((2, 4, 2), generator=g) * 2 - 1).half().npu(), ptk=torch.tensor([0, 0], dtype=torch.int64), asq=None, ask=torch.tensor([20, 28], dtype=torch.int64).npu(), bt=torch.tensor([[1, 0, 0, 0], [2, 3, 0, 0]], dtype=torch.int32).npu(), dq=(torch.rand((2, 4, 2), generator=g) * 0.099 + 0.001).half().npu(), dk=(torch.rand((4, 16, 1), generator=g) * 0.099 + 0.001).half().npu(),)
        idx, val = _call_op(_load_op(), inp, "BSND", "PA_BBND", 4, 2, 0, True, 2)
        ginp = dict(q=q, k=k_logical, w=inp["w"].cpu(), ptk=inp["ptk"], asq=None, ask=inp["ask"].cpu(), bt=inp["bt"].cpu(), dq=inp["dq"].cpu(), dk=inp["dk"].cpu(),)
        _check(
            idx,
            val,
            _golden_of(ginp, "BSND", "PA_BBND", 4, 2, 0, True, 2),
            (2, 4),
            4,
            2,
            True,
        )

    def test_e6_v2_compact_torch(self, pki_device):
        """E6: 同输入 V2(compact) 与 V1 布局互异、有效集合一致, values 对拍。"""
        g = torch.Generator().manual_seed(20260206)
        inp = dict(q=torch.randint(-127, 128, (1, 4, 2, D), dtype=torch.int8, generator=g), k=torch.randint(-127, 128, (1, 3, 1, D), dtype=torch.int8, generator=g), w=(torch.rand((1, 4, 2), generator=g) * 50.0 - 25.0).half(), ptk=torch.tensor([3], dtype=torch.int64), asq=None, ask=None, bt=None, dq=(torch.rand((1, 4, 2), generator=g) * 0.099 + 0.001).half(), dk=(torch.rand((1, 3, 1), generator=g) * 0.099 + 0.001).half(),)
        _to_npu(inp, ["q", "k", "w", "dq", "dk", "ptk"])
        op = _load_op()
        idx_v1, _ = _call_op(op, inp, "BSND", "BSND", 16, 4, 0, True, 2)
        idx_v2, val_v2 = _call_op(op, inp, "BSND", "BSND", 16, 4, 0, True, 2, mode=1)
        _check(
            idx_v2,
            val_v2,
            _golden_of(inp, "BSND", "BSND", 16, 4, 0, True, 2, iv=2),
            (1, 4),
            16,
            4,
            True,
        )
        a1 = idx_v1.cpu().numpy().reshape(-1, 19)
        a2 = idx_v2.cpu().numpy().reshape(-1, 19)
        diff = 0
        for r in range(a1.shape[0]):
            assert sorted(a1[r][a1[r] >= 0].tolist()) == sorted(
                a2[r][a2[r] >= 0].tolist()
            )
            diff += not (a1[r] == a2[r]).all()
        assert diff > 0, "短序列下 V1/V2 布局应有差异(tail 位置重排)"

    def test_e7_graph_replay(self, pki_device):
        """E7: torchair graph 动态长度 replay(pc<sc/==sc/>sc), fullgraph + replay 逐位一致
        + 尾池因果 + compact 前缀合同(非量化 fp16 路径)。"""
        pytest.importorskip("torchair")
        import torchair

        old_limit = torch._dynamo.config.cache_size_limit
        torch._dynamo.config.cache_size_limit = max(old_limit, 64)
        topk, ps, sc = 64, 4, 16

        def call_fn(inp):
            return _call_op(
                _load_op(), inp, "BSND", "BSND", topk, ps, 3, False, -1, mode=1
            )

        compiled = torch.compile(
            call_fn, backend=torchair.get_npu_backend(), fullgraph=True, dynamic=False
        )
        try:
            for n_pool in (4, 8, 15, 16, 17, 20):  # pc<sc / ==sc / >sc
                tk = n_pool % ps
                L = n_pool * ps + tk
                g = torch.Generator().manual_seed(20263500 + n_pool)
                inp = dict(q=(torch.rand((1, 1, 4, D), generator=g) * 2 - 1).half(), k=(torch.rand((1, n_pool, 1, D), generator=g) * 2 - 1).half(), w=(torch.rand((1, 1, 4), generator=g) * 2 - 1).half(), ptk=torch.tensor([tk], dtype=torch.int64), asq=None, ask=None, bt=None, dq=None, dk=None,)
                _to_npu(inp, ["q", "k", "w", "ptk"])
                idx_a, _ = compiled(inp)
                idx_b, _ = compiled(inp)
                assert torch.equal(idx_a, idx_b), f"pools={n_pool}: replay 输出不一致"
                row = idx_a[0, 0].cpu()
                assert int(row.max()) <= L - 1, f"pools={n_pool}: 含未来 token"
                expect_tail = list(range(L - tk, L)) if tk > 0 else []
                assert sorted(int(t) for t in row if t >= L - tk) == expect_tail, (
                    f"pools={n_pool}: 尾池违约"
                )
                arr = row.numpy()
                gap = int((arr == -1).argmax()) if (arr == -1).any() else len(arr)
                assert not (arr[:gap] == -1).any() and bool((arr[gap:] == -1).all()), (
                    f"pools={n_pool}: 前缀合同违约"
                )
                valid = sorted(int(t) for t in arr[:gap])
                if n_pool < sc:
                    assert valid == list(range(L)), f"pools={n_pool}: 预算不足应全选"
                else:
                    assert len(valid) == topk + tk, f"pools={n_pool}: 应 topk+vtk"
        finally:
            torch._dynamo.config.cache_size_limit = old_limit
