/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_POOL_KEY_INDEXER_V2_H
#define ACLNN_POOL_KEY_INDEXER_V2_H

#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief V2（全量原型）接口的第一段：计算 workspace 大小并构建执行器。
 * 手写 aclnn 接口：外层签名冻结为已发布的 V2 原型（含 sparse_indices_mode 属性），
 * 内部转发 def 自动生成的单一全量 inner 接口 aclnnInnerPoolKeyIndexer
 * （def 进 op_host_aclnnInner 库，见 op_host/CMakeLists.txt）。
 * @param sparseIndicesMode 索引输出布局模式：0=legacy（tail 固定偏移 topk），
 *                          1=compact（tail 紧跟有效展开 token，-1 缝隙全部位于行尾）
 * @domain aclnn_ops_infer
 */
__attribute__((visibility("default"))) aclnnStatus aclnnPoolKeyIndexerV2GetWorkspaceSize(
    const aclTensor* query, const aclTensor* poolKey, const aclTensor* weights, const aclIntArray* poolTailK,
    const aclIntArray* actualSeqQOptional, const aclIntArray* actualSeqKOptional, const aclTensor* blockTableOptional,
    const aclTensor* qDescaleOptional, const aclTensor* kDescaleOptional, int64_t topk, int64_t poolSize,
    char* layoutQOptional, char* layoutKOptional, int64_t maskMode, int64_t quantMode, bool returnValue,
    int64_t keyStride0, int64_t kDescaleStride0, int64_t sparseIndicesMode, const aclTensor* sparseIndicesOut,
    const aclTensor* sparseValuesOut, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief V2 Tensor 变体的第一段：pool_tail_k/actual_seq_q/actual_seq_k 以 aclTensor
 * 传入（计算设备上的 int64 连续 tensor，无 D2H，支持图捕获）。
 * @domain aclnn_ops_infer
 */
__attribute__((visibility("default"))) aclnnStatus aclnnPoolKeyIndexerV2TensorGetWorkspaceSize(
    const aclTensor* query, const aclTensor* poolKey, const aclTensor* weights, const aclTensor* poolTailK,
    const aclTensor* actualSeqQOptional, const aclTensor* actualSeqKOptional, const aclTensor* blockTableOptional,
    const aclTensor* qDescaleOptional, const aclTensor* kDescaleOptional, int64_t topk, int64_t poolSize,
    char* layoutQOptional, char* layoutKOptional, int64_t maskMode, int64_t quantMode, bool returnValue,
    int64_t keyStride0, int64_t kDescaleStride0, int64_t sparseIndicesMode, const aclTensor* sparseIndicesOut,
    const aclTensor* sparseValuesOut, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief V2 接口的第二段：执行计算。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnPoolKeyIndexerV2(void* workspace, uint64_t workspaceSize,
                                                                         aclOpExecutor* executor,
                                                                         const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_POOL_KEY_INDEXER_V2_H
