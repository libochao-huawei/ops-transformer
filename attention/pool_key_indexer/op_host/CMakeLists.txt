# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
add_op_to_compiled_list()

# def 进 op_host_aclnnInner 库：opbuild 自动生成单一 inner 接口 aclnnInnerPoolKeyIndexer
# （全量原型，含 sparse_indices_mode），外层 aclnnPoolKeyIndexer（V1 兼容签名）与
# aclnnPoolKeyIndexerV2（全量原型）由 op_api 手写提供
# （参考 lightning_indexer_v2 手写 aclnn 接口的同款结构）
if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE
    pool_key_indexer_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME PoolKeyIndexer
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -Werror
            -mllvm -cce-vf-remove-membar=false
            -mllvm -cce-aicore-hoist-movemask=false
)

if (NOT BUILD_OPS_RTY_KERNEL)
    # ACLNNTYPE aclnn_inner: 外层 aclnn 接口不走 opbuild 自动生成, 由 op_api 手写;
    # def 不带 Version 标记, opbuild 只生成单一全量 inner 原型
    # (参考 lightning_indexer_v2 / quant_lightning_indexer 的 aclnn_inner 配置)
    add_modules_sources(OPTYPE pool_key_indexer ACLNNTYPE aclnn_inner)

    # 手写 op_api 头文件打包进 vendor 包 op_api/include/aclnnop
    # (aclnn_inner 算子不走 generate_aclnn_headers 的自动安装路径)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/op_api/aclnn_pool_key_indexer.h
                  ${CMAKE_CURRENT_SOURCE_DIR}/op_api/aclnn_pool_key_indexer_v2.h
            DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL)
endif()
