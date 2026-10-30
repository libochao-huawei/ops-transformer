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
"""pool_key_indexer pytest conftest（T1-3）: 设备选择 / golden 自测开关 / NPU 执行后端。

NPU 执行后端（aclnn 直调, 对齐 examples/ 与 910B3 已验证 smoke 载体）::
    主线 torch_extension csrc 的 isQuantPath 仅覆盖 quant_mode 0/1（HEAD 60139c985;
    int8 e2e 适配属迭代三 T3-3），quant_mode=2 会被
    "quant_mode=-1 requires q_descale and k_descale to be null" 拒绝 —— 本框架
    按任务预案改用 aclnn 直调路径执行 C1~C5 全部 NPU 用例：
      * aclnn_runner.cpp 编译为通用执行器（manifest + bin 输入 → aclnn 调用 → bin 输出）
      * 算子包: gjw_pki（需在 opp/vendors load_priority 首位, ISSUE-002）
"""

import os
import subprocess
from pathlib import Path

import pytest

TEST_DIR = Path(__file__).resolve().parent
RUNNER_SRC = TEST_DIR / "aclnn_runner.cpp"
RUNNER_BIN = TEST_DIR / "build" / "aclnn_runner"


def pytest_addoption(parser):
    parser.addoption(
        "--pki-device",
        action="store",
        default=os.environ.get("PKI_DEVICE", "0"),
        help="NPU device id (default: env PKI_DEVICE or '0')",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "pki_npu: case requires a real NPU device")
    config.addinivalue_line("markers", "pki_error: case expects host-side rejection")


def _cann_home():
    home = os.environ.get("ASCEND_TOOLKIT_HOME") or os.environ.get("ASCEND_HOME_PATH")
    if home and Path(home).is_dir():
        return Path(home)
    for cand in ("/opt/gjw/cann/cann-9.3.0", "/usr/local/Ascend/ascend-toolkit/latest"):
        if Path(cand).is_dir():
            return Path(cand)
    return None


def _vendor_dir(cann_home):
    """定位含 aclnn_pool_key_indexer.h 的 vendor 包。

    按 opp/vendors/config.ini 的 load_priority 顺序优先（ISSUE-002:
    gjw_pki 需在首位, 避免命中他人同名旧版算子包）; 无 config 时按字典序扫描。
    """
    opp = cann_home / "opp" / "vendors"
    if not opp.is_dir():
        return None

    def has_header(vendor):
        return (
            vendor / "op_api" / "include" / "aclnnop" / "aclnn_pool_key_indexer.h"
        ).is_file()

    config = opp / "config.ini"
    if config.is_file():
        for line in config.read_text().splitlines():
            line = line.strip()
            if line.startswith("load_priority"):
                for name in line.split("=", 1)[1].split(","):
                    vendor = opp / name.strip()
                    if vendor.is_dir() and has_header(vendor):
                        return vendor
                break
    for vendor in sorted(opp.iterdir()):
        if has_header(vendor):
            return vendor
    return None


def _npu_available():
    return (
        any(Path(f"/dev/davinci{i}").exists() for i in range(8))
        or Path("/dev/davinci").exists()
    )


def _build_runner(cann_home, vendor):
    RUNNER_BIN.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "g++",
        "-O2",
        "-std=c++17",
        "-o",
        str(RUNNER_BIN),
        str(RUNNER_SRC),
        f"-I{vendor / 'op_api' / 'include'}",
        f"-I{cann_home / 'include'}",
        f"-L{vendor / 'op_api' / 'lib'}",
        "-lcust_opapi",
        f"-L{cann_home / 'lib64'}",
        "-lascendcl",
        "-lnnopbase",
        "-lc_sec",
        f"-Wl,-rpath,{vendor / 'op_api' / 'lib'}",
        f"-Wl,-rpath,{cann_home / 'lib64'}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"aclnn_runner compile failed:\n{proc.stderr[-2000:]}")
    return RUNNER_BIN


class AclnnBackend:
    """aclnn 直调后端: 写 case 目录 → 跑 runner → 读输出。"""

    ENV_RETRY_MARKERS = (
        "TsdOpen failed",
        "acl init / set device failed",
        "hdc service",
        "DeviceRetain",
    )
    ENV_MAX_RETRIES = 3

    def __init__(self, runner_bin, device_id, workdir):
        import shutil

        self.runner_bin = str(runner_bin)
        self.device_id = int(device_id)
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        for stale in self.workdir.glob("case_*"):
            shutil.rmtree(stale, ignore_errors=True)
        self._case_seq = 0

    def _exec(self, case_dir):
        import time

        env = dict(os.environ)
        # host 侧 OP_LOGE 打到 stdout(非法用例报错语义断言用)
        env.setdefault("ASCEND_GLOBAL_LOG_LEVEL", "1")
        env.setdefault("ASCEND_SLOG_PRINT_TO_STDOUT", "1")
        proc = subprocess.run(
            [self.runner_bin, str(case_dir), str(self.device_id)],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
        (case_dir / "run.stdout").write_text(proc.stdout[-20000:])
        (case_dir / "run.stderr").write_text(proc.stderr[-20000:])
        # 共享设备瞬时故障(TsdOpen/hdc 类)重试, 不掩盖真实算子失败
        retries = 0
        while (
            proc.returncode == 1
            and retries < self.ENV_MAX_RETRIES
            and any(m in (proc.stdout + proc.stderr) for m in self.ENV_RETRY_MARKERS)
        ):
            retries += 1
            time.sleep(2.0)
            proc = subprocess.run(
                [self.runner_bin, str(case_dir), str(self.device_id)],
                capture_output=True,
                text=True,
                timeout=300,
                env=env,
            )
            (case_dir / "run.stdout").write_text(proc.stdout[-20000:])
            (case_dir / "run.stderr").write_text(proc.stderr[-20000:])
        if retries:
            print(
                f"[conftest] env-failure retried {retries} time(s) for {case_dir.name}"
            )
        return proc

    @staticmethod
    def _tensor_bin(t):
        """torch CPU tensor → (numpy 兼容字节, manifest dtype 名)。"""
        import torch

        if t.dtype == torch.int8:
            return t.numpy().tobytes(), "int8"
        if t.dtype == torch.int32:
            return t.numpy().tobytes(), "int32"
        if t.dtype == torch.int64:
            return t.numpy().tobytes(), "int64"
        if t.dtype == torch.float16:
            return t.numpy().tobytes(), "float16"
        if t.dtype == torch.bfloat16:
            return t.view(torch.uint16).numpy().tobytes(), "bfloat16"
        if t.dtype == torch.float32:
            return t.numpy().tobytes(), "float32"
        raise ValueError(f"unsupported dtype: {t.dtype}")

    def write_case(self, spec, inputs):
        """写 case 目录(manifest + 输入 bin), 返回 case_dir。"""
        self._case_seq += 1
        case_dir = self.workdir / f"case_{self._case_seq:03d}_{spec['name']}"
        case_dir.mkdir(parents=True, exist_ok=True)

        manifest = []
        tensor_keys = [
            ("query", "q"),
            ("pool_key", "k"),
            ("weights", "w"),
            ("pool_tail_k", "ptk"),
            ("actual_seq_q", "asq"),
            ("actual_seq_k", "ask"),
            ("block_table", "bt"),
            ("q_descale", "dq"),
            ("k_descale", "dk"),
        ]
        for key, ik in tensor_keys:
            t = inputs.get(ik)
            if t is None:
                continue
            data, dt_name = self._tensor_bin(t)
            (case_dir / f"{key}.bin").write_bytes(data)
            dims = " ".join(str(int(d)) for d in t.shape)
            manifest.append(f"tensor {key} {dt_name} {len(t.shape)} {dims}")
        manifest.append(f"attr topk {spec['topk']}")
        manifest.append(f"attr pool_size {spec['ps']}")
        manifest.append(f"attr mask_mode {spec['mask']}")
        manifest.append(f"attr quant_mode {spec['quant_mode']}")
        manifest.append(f"attr return_value {1 if spec['rv'] else 0}")
        manifest.append("attr key_stride0 -1")
        manifest.append("attr k_descale_stride0 -1")
        # sparse_indices_mode(索引输出布局模式, V2 接口专有): 1=紧凑布局;
        # 缺省不写走 V1(legacy)。用例 spec 的 "version" 键承载取值(0/1),
        # 与 aclnnPoolKeyIndexerV2 的 sparseIndicesMode 参数一一对应
        if spec.get("version") is not None:
            manifest.append(f"attr sparse_indices_mode {spec['version']}")
        # 非法用例输出 shape 覆盖(空/rank-0 query): aclnn executor 对全空输出
        # 跳过 tiling/kernel, 须以非空输出到达算子 host 校验(T3-2 L2_006/L2_007)
        if spec.get("out_dim0") is not None:
            manifest.append(f"attr out_dim0 {spec['out_dim0']}")
        if spec.get("out_dim1") is not None:
            manifest.append(f"attr out_dim1 {spec['out_dim1']}")
        manifest.append(f"str layout_q {spec['layout_q']}")
        manifest.append(f"str layout_k {spec['layout_k']}")
        (case_dir / "manifest.txt").write_text("\n".join(manifest) + "\n")
        return case_dir

    def _parse_outputs(self, spec, case_dir):
        """读 case 输出为 (indices np.int32, values np.float32)。"""
        import numpy as np

        lead = tuple(spec["q_shape"][:-2])
        idx_shape = lead + (spec["topk"] + spec["ps"] - 1,)
        idx = np.frombuffer(
            (case_dir / "indices.bin").read_bytes(), dtype="<i4"
        ).reshape(idx_shape)
        if spec["rv"]:
            val_shape = lead + (spec["topk"] // spec["ps"],)
            val = np.frombuffer(
                (case_dir / "values.bin").read_bytes(), dtype="<f4"
            ).reshape(val_shape)
        else:
            val = np.zeros(0, dtype=np.float32)
        return idx, val

    def run_seq(self, spec_inputs):
        """序列模式: 同一 runner 进程顺序执行多条用例, 每条之间释放 device 内存
        (LD 跨调用残留探针 P-② 用)。返回 [(case_dir, rc, stdout), ...]。

        spec_inputs: [(spec, inputs), ...]
        """
        case_dirs = [self.write_case(spec, inputs) for spec, inputs in spec_inputs]
        env = dict(os.environ)
        env.setdefault("ASCEND_GLOBAL_LOG_LEVEL", "1")
        env.setdefault("ASCEND_SLOG_PRINT_TO_STDOUT", "1")
        proc = subprocess.run(
            [self.runner_bin, "--seq", str(self.device_id)]
            + [str(d) for d in case_dirs],
            capture_output=True,
            text=True,
            timeout=900,
            env=env,
        )
        (case_dirs[-1] / "seq.stdout").write_text(proc.stdout[-40000:])
        (case_dirs[-1] / "seq.stderr").write_text(proc.stderr[-40000:])
        results = []
        for d in case_dirs:
            rc = None
            for line in proc.stdout.splitlines():
                if line.startswith("SEQ_CASE ") and line.split(" ", 2)[2] == str(d):
                    rc = int(line.split(" ", 2)[1])
            results.append((d, rc, proc.stdout))
        return results

    def run_case(self, spec, inputs):
        """执行一条用例。返回 (indices np.int32, values np.float32); 非法用例返回 subprocess 结果。"""
        case_dir = self.write_case(spec, inputs)

        proc = self._exec(case_dir)
        if proc.returncode != 0:
            return proc

        return self._parse_outputs(spec, case_dir)


@pytest.fixture(scope="session")
def npu_backend(request):
    """NPU 执行后端: 无卡 / 无 CANN / 无 gjw_pki 算子包时 skip。"""
    if not _npu_available():
        pytest.skip("no NPU device found (/dev/davinci*), skip NPU cases")
    cann_home = _cann_home()
    if cann_home is None:
        pytest.skip(
            "CANN toolkit not found (ASCEND_TOOLKIT_HOME / /opt/gjw/cann), skip NPU cases"
        )
    vendor = _vendor_dir(cann_home)
    if vendor is None:
        pytest.skip(
            f"gjw_pki vendor package (aclnn_pool_key_indexer.h) not found under {cann_home}/opp/vendors"
        )
    runner = _build_runner(cann_home, vendor)
    device = request.config.getoption("--pki-device")
    backend = AclnnBackend(runner, device, TEST_DIR / "build" / "cases")
    print(
        f"[conftest] aclnn backend ready: runner={runner}, vendor={vendor.name}, device={device}"
    )
    return backend
