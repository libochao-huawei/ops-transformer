#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# pool_key_indexer pytest 冒烟套入口(4 文件 68 例: 功能/边界/V2+tail/int8+LD/torch e2e)
# test_boundary.py + test_l2_errors.py; T3-3 扩展 test_torch_e2e.py——torch_extension
# e2e, 需先按 test_torch_e2e.py docstring 构建 whl + pip --target 隔离安装;
# R4' 扩展 test_multicore.py——S2 多核切分 + LD 归并专项, DESIGN §8;
# R4'+ 扩展 test_ld_defect_probe.py——LD 归并缺陷存在性探针转正, 33 条,
# 含 runner --seq 序列模式跨调用残留用例）
#
# 用法:
#   ./run.sh                        # 全量用例（需 910B3 实卡 + gjw_ 环境 + gjw_pki 算子包）
#   ./run.sh --collect-only         # 仅收集用例（无 NPU 也可执行）
#   ./run.sh -k C1                  # 按关键字筛选
#   ./run.sh --pki-device 1         # 指定 NPU 设备号
#   ./run.sh --pki-skip-golden-selfcheck   # 跳过 golden 自测
#   ./run.sh test_torch_e2e.py      # 仅跑 torch_extension e2e（追加文件参数）
#
# 环境要求（gjw_ conda 环境已内置 ASCEND_TOOLKIT_HOME=/opt/gjw/cann/cann-9.3.0）:
#   - CANN: /opt/gjw/cann（gjw_pki_transformer 需在 opp/vendors load_priority 首位, ISSUE-002）
#   - conda: gjw_（torch + torch_npu）
#   - 算子包: build.sh --pkg --soc=ascend910b --ops=pool_key_indexer --vendor_name=gjw_pki 产物已安装

set -euo pipefail
cd "$(dirname "$0")"

PY="${PKI_PYTHON:-python3}"

# conda gjw_ 环境激活（已在 gjw_ 内则跳过）
if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "gjw_" ]]; then
    if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
        source /root/miniconda3/etc/profile.d/conda.sh
        conda activate gjw_
    fi
fi

# gjw_pki vendor 运行环境（ASCEND_CUSTOM_OPP_PATH + opapi LD_LIBRARY_PATH）:
# torch_extension csrc 的 V2 接口符号经 GetOpApiFuncAddr 解析, 依赖本环境变量;
# 未设置时 V2 用例(e6 等)报 "not in opapi lib"（E5 torchair 偶发先行加载会掩盖）
VENDOR_ENV="/opt/gjw/cann/cann-9.3.0/vendors/gjw_pki_transformer/bin/set_env.bash"
if [[ -f "${VENDOR_ENV}" ]]; then
    # vendor 脚本在 set -u 下引用未初始化的 ASCEND_CUSTOM_OPP_PATH 会报 unbound, 临时关 u
    set +u
    # shellcheck disable=SC1090
    source "${VENDOR_ENV}"
    set -u
fi

exec "${PY}" -m pytest test_smoke_core.py test_v2_tail_model.py test_quant_ld.py test_torch_e2e.py "$@"
