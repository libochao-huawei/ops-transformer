# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# arch35 (ascend950) kernel UT。
#
# 状态: 默认关闭。本仓库 CANN 9.3.0 的 tikicpulib CPU 仿真头
#   asc/impl/basic_api/kernel_operator_fixpipe_intf_impl.h 只提供
#   Fixpipe(..., FixpipeParamsV220) 与 Fixpipe(..., FixpipeParamsArch3510<config.format>)
#   两类重载, 没有生产 arch35 kernel 使用的 C310 双目的
#   Fixpipe(..., FixpipeParamsC310<config.format>) 重载(该重载仅设备端 bisheng
#   头有)。因此 pool_key_indexer arch35 kernel 无法在本 CANN 版本做 CPU 仿真,
#   强行开启会编译报 "no matching function for call to Fixpipe(...,FixpipeParamsC310)"。
#   同族 arch35 cube 算子(lightning_indexer / quant_lightning_indexer)同样未接入
#   CPU kernel UT, 印证是仿真器能力缺口, 非算子或合并问题。
#
# A5 侧对本合并的真实校验: `--pkg --soc=ascend950` 设备端编译(已通过) + 真机 ST。
# 待 CANN 仿真头补齐 C310 Fixpipe 重载后, 用 -DPKI_ENABLE_ARCH35_KUT=ON 打开本用例。
option(PKI_ENABLE_ARCH35_KUT "Enable pool_key_indexer arch35 CPU kernel UT (needs CANN C310 Fixpipe sim support)" OFF)

if ((UT_TEST_ALL OR OP_KERNEL_UT) AND PKI_ENABLE_ARCH35_KUT)
    set(pool_key_indexer_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../op_host/pool_key_indexer_tiling.cpp
        ${PROJECT_SOURCE_DIR}/third_party/opbase/src/op_common/log/log.cpp
    )
    AddOpTestCase(pool_key_indexer "Ascend950PR_9599" "" "${pool_key_indexer_tiling_files}")
endif()
