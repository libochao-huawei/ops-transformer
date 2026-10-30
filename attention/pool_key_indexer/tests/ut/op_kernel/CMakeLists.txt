# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# arch22 (ascend910b / ascend910_93) kernel UT: AddOpTestCase 在 cmake/ut.cmake 中把
# op_kernel/ 目录下的用例路由到"构建不含 arch35"的场景(即 A2/A3 构建); A5 构建会跳过
# 本目录用例并递归到 arch35/ 子目录。
if (UT_TEST_ALL OR OP_KERNEL_UT)
    if (NOT ARCH_DIRECTORY MATCHES "arch35")
        set(pool_key_indexer_tiling_files
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/pool_key_indexer_tiling.cpp
            ${PROJECT_SOURCE_DIR}/third_party/opbase/src/op_common/log/log.cpp
        )
        # param1: 算子名(param2: 与 tikicpulib 一致的 soc) param3: 编译宏 param4: tiling 源码
        AddOpTestCase(pool_key_indexer "Ascend910B1" "" "${pool_key_indexer_tiling_files}")
    endif()
endif()

# 递归 arch 子目录(arch35 用例只在 A5 构建生效, 目录内自行判定)
file(GLOB CURRENT_SOURCE_DIRS LIST_DIRECTORIES true ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(SUB_DIR ${CURRENT_SOURCE_DIRS})
    if(EXISTS "${SUB_DIR}/CMakeLists.txt")
        add_subdirectory(${SUB_DIR})
    endif()
endforeach()
