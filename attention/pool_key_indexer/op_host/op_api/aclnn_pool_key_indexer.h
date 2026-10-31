/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_POOL_KEY_INDEXER_H
#define ACLNN_POOL_KEY_INDEXER_H

#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief V1（模型侧存量兼容）接口的第一段：计算 workspace 大小并构建执行器。
 * 手写 aclnn 接口：签名不含 sparse_indices_mode 属性（该属性仅在
 * aclnnPoolKeyIndexerV2 签名中暴露），恒以 legacy 布局运行，与历史版本
 * 输出逐比特一致；内部转发 aclnnInnerPoolKeyIndexer（sparse_indices_mode 固定 0）。
 * @domain aclnn_ops_infer
 */
__attribute__((visibility("default"))) aclnnStatus aclnnPoolKeyIndexerGetWorkspaceSize(
    const aclTensor* query, const aclTensor* poolKey, const aclTensor* weights, const aclIntArray* poolTailK,
    const aclIntArray* actualSeqQOptional, const aclIntArray* actualSeqKOptional, const aclTensor* blockTableOptional,
    const aclTensor* qDescaleOptional, const aclTensor* kDescaleOptional, int64_t topk, int64_t poolSize,
    char* layoutQOptional, char* layoutKOptional, int64_t maskMode, int64_t quantMode, bool returnValue,
    int64_t keyStride0, int64_t kDescaleStride0, const aclTensor* sparseIndicesOut, const aclTensor* sparseValuesOut,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief V1 Tensor 变体的第一段：pool_tail_k/actual_seq_q/actual_seq_k 以 aclTensor
 * 传入（计算设备上的 int64 连续 tensor，无 D2H，支持图捕获）。
 * @domain aclnn_ops_infer
 */
__attribute__((visibility("default"))) aclnnStatus aclnnPoolKeyIndexerTensorGetWorkspaceSize(
    const aclTensor* query, const aclTensor* poolKey, const aclTensor* weights, const aclTensor* poolTailK,
    const aclTensor* actualSeqQOptional, const aclTensor* actualSeqKOptional, const aclTensor* blockTableOptional,
    const aclTensor* qDescaleOptional, const aclTensor* kDescaleOptional, int64_t topk, int64_t poolSize,
    char* layoutQOptional, char* layoutKOptional, int64_t maskMode, int64_t quantMode, bool returnValue,
    int64_t keyStride0, int64_t kDescaleStride0, const aclTensor* sparseIndicesOut, const aclTensor* sparseValuesOut,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief V1 接口的第二段：执行计算。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnPoolKeyIndexer(void* workspace, uint64_t workspaceSize,
                                                                       aclOpExecutor* executor,
                                                                       const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_POOL_KEY_INDEXER_H
