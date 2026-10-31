/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_pool_key_indexer_v2.cpp
 * \brief pool_key_indexer 手写 aclnn V2 接口：签名冻结为已发布的 V2 原型
 * （含 sparse_indices_mode 属性），内部转发 def 自动生成的单一全量 inner 接口
 * aclnnInnerPoolKeyIndexer（IntArray/Tensor 双变体），执行器构建、入参校验与
 * tiling 全部复用 inner 实现。
 */

#include <string.h>
#include "graph/types.h"
#include "aclnn_pool_key_indexer_v2.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

// def 自动生成的单一全量 inner 接口（编入 ops_aclnn 静态库，随 libcust_opapi.so
// 一并链接）：含 sparse_indices_mode 在内的全部属性
extern aclnnStatus aclnnInnerPoolKeyIndexerGetWorkspaceSize(
    const aclTensor* query, const aclTensor* poolKey, const aclTensor* weights, const aclIntArray* poolTailK,
    const aclIntArray* actualSeqQOptional, const aclIntArray* actualSeqKOptional, const aclTensor* blockTableOptional,
    const aclTensor* qDescaleOptional, const aclTensor* kDescaleOptional, int64_t topk, int64_t poolSize,
    char* layoutQOptional, char* layoutKOptional, int64_t maskMode, int64_t quantMode, bool returnValue,
    int64_t keyStride0, int64_t kDescaleStride0, int64_t sparseIndicesMode, const aclTensor* sparseIndicesOut,
    const aclTensor* sparseValuesOut, uint64_t* workspaceSize, aclOpExecutor** executor);

extern aclnnStatus aclnnInnerPoolKeyIndexerTensorGetWorkspaceSize(
    const aclTensor* query, const aclTensor* poolKey, const aclTensor* weights, const aclTensor* poolTailK,
    const aclTensor* actualSeqQOptional, const aclTensor* actualSeqKOptional, const aclTensor* blockTableOptional,
    const aclTensor* qDescaleOptional, const aclTensor* kDescaleOptional, int64_t topk, int64_t poolSize,
    char* layoutQOptional, char* layoutKOptional, int64_t maskMode, int64_t quantMode, bool returnValue,
    int64_t keyStride0, int64_t kDescaleStride0, int64_t sparseIndicesMode, const aclTensor* sparseIndicesOut,
    const aclTensor* sparseValuesOut, uint64_t* workspaceSize, aclOpExecutor** executor);

extern aclnnStatus aclnnInnerPoolKeyIndexer(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                            const aclrtStream stream);

aclnnStatus aclnnPoolKeyIndexerV2GetWorkspaceSize(
    const aclTensor* query, const aclTensor* poolKey, const aclTensor* weights, const aclIntArray* poolTailK,
    const aclIntArray* actualSeqQOptional, const aclIntArray* actualSeqKOptional, const aclTensor* blockTableOptional,
    const aclTensor* qDescaleOptional, const aclTensor* kDescaleOptional, int64_t topk, int64_t poolSize,
    char* layoutQOptional, char* layoutKOptional, int64_t maskMode, int64_t quantMode, bool returnValue,
    int64_t keyStride0, int64_t kDescaleStride0, int64_t sparseIndicesMode, const aclTensor* sparseIndicesOut,
    const aclTensor* sparseValuesOut, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    if (query == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "query pointer is null, cannot get data type!");
        return ge::GRAPH_FAILED;
    }
    if (poolKey == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "poolKey pointer is null, cannot get data type!");
        return ge::GRAPH_FAILED;
    }
    if (weights == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "weights pointer is null, cannot get data type!");
        return ge::GRAPH_FAILED;
    }
    if (poolTailK == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "poolTailK pointer is null, cannot get data type!");
        return ge::GRAPH_FAILED;
    }
    if (sparseIndicesOut == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "sparseIndicesOut cannot be nullptr.");
        return ge::GRAPH_FAILED;
    }
    if (sparseValuesOut == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "sparseValuesOut cannot be nullptr.");
        return ge::GRAPH_FAILED;
    }
    if (workspaceSize == nullptr || executor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "workspaceSize or executor cannot be nullptr.");
        return ge::GRAPH_FAILED;
    }

    return aclnnInnerPoolKeyIndexerGetWorkspaceSize(
        query, poolKey, weights, poolTailK, actualSeqQOptional, actualSeqKOptional, blockTableOptional,
        qDescaleOptional, kDescaleOptional, topk, poolSize, layoutQOptional, layoutKOptional, maskMode, quantMode,
        returnValue, keyStride0, kDescaleStride0, sparseIndicesMode, sparseIndicesOut, sparseValuesOut, workspaceSize,
        executor);
}

aclnnStatus aclnnPoolKeyIndexerV2TensorGetWorkspaceSize(
    const aclTensor* query, const aclTensor* poolKey, const aclTensor* weights, const aclTensor* poolTailK,
    const aclTensor* actualSeqQOptional, const aclTensor* actualSeqKOptional, const aclTensor* blockTableOptional,
    const aclTensor* qDescaleOptional, const aclTensor* kDescaleOptional, int64_t topk, int64_t poolSize,
    char* layoutQOptional, char* layoutKOptional, int64_t maskMode, int64_t quantMode, bool returnValue,
    int64_t keyStride0, int64_t kDescaleStride0, int64_t sparseIndicesMode, const aclTensor* sparseIndicesOut,
    const aclTensor* sparseValuesOut, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    if (query == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "query pointer is null, cannot get data type!");
        return ge::GRAPH_FAILED;
    }
    if (poolKey == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "poolKey pointer is null, cannot get data type!");
        return ge::GRAPH_FAILED;
    }
    if (weights == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "weights pointer is null, cannot get data type!");
        return ge::GRAPH_FAILED;
    }
    if (poolTailK == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "poolTailK pointer is null, cannot get data type!");
        return ge::GRAPH_FAILED;
    }
    if (sparseIndicesOut == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "sparseIndicesOut cannot be nullptr.");
        return ge::GRAPH_FAILED;
    }
    if (sparseValuesOut == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "sparseValuesOut cannot be nullptr.");
        return ge::GRAPH_FAILED;
    }
    if (workspaceSize == nullptr || executor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "workspaceSize or executor cannot be nullptr.");
        return ge::GRAPH_FAILED;
    }

    return aclnnInnerPoolKeyIndexerTensorGetWorkspaceSize(
        query, poolKey, weights, poolTailK, actualSeqQOptional, actualSeqKOptional, blockTableOptional,
        qDescaleOptional, kDescaleOptional, topk, poolSize, layoutQOptional, layoutKOptional, maskMode, quantMode,
        returnValue, keyStride0, kDescaleStride0, sparseIndicesMode, sparseIndicesOut, sparseValuesOut, workspaceSize,
        executor);
}

aclnnStatus aclnnPoolKeyIndexerV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                  const aclrtStream stream)
{
    return aclnnInnerPoolKeyIndexer(workspace, workspaceSize, executor, stream);
}

} // namespace

#ifdef __cplusplus
}
#endif
