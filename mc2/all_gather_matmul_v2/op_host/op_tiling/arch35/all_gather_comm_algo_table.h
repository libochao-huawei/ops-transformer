/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file all_gather_comm_algo_table.h
 * \brief all_gather_matmul / all_gather_matmul_v2 共享的 AllGather 通信算法表。
 */

#ifndef ALL_GATHER_COMM_ALGO_TABLE_H
#define ALL_GATHER_COMM_ALGO_TABLE_H

#include <cstdint>
#include <string>
#include "mc2_comm_algo_selector.h"

namespace Mc2Tiling {

inline constexpr const char* ALLGATHER_DEFAULT_ALGO_NAME = "AllGather=level0:fullmesh";

const Mc2Hcom::CommAlgoEntry* GetAllGatherCommAlgoTable(uint32_t& count);

// 计算单轮AllGather通信数据量tileM×K×dtypeSize(带回绕钳位)，查表选择通信算法并返回算法名；
// 未命中表项时回退默认算法ALLGATHER_DEFAULT_ALGO_NAME
std::string SelectAllGatherAlgoName(const std::string& opName, const std::string& group, uint8_t commEngine,
                                    uint64_t tileM, uint64_t kValue, uint64_t dtypeSize, uint32_t rankDim);

} // namespace Mc2Tiling

#endif // ALL_GATHER_COMM_ALGO_TABLE_H
