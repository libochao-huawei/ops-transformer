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

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
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
    # ACLNN_EXTRA_VERSION v2: def 中 version 属性带 .Version(2) 标记, opbuild 会
    # 额外生成 aclnn_pool_key_indexer_v2.{cpp,h}(V2 接口), 须经此声明才会编入
    # ops_aclnn / 打包进 op_api(参考 kv_rms_norm_rope_cache 同款配置)
    add_modules_sources(OPTYPE pool_key_indexer ACLNNTYPE aclnn ACLNN_EXTRA_VERSION v2)
endif()
