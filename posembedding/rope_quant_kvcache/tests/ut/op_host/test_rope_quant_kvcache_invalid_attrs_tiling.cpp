/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <vector>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_host/rope_quant_kvcache_tiling.h"

namespace {
gert::TilingContextPara MakeContextPara(const std::vector<gert::TilingContextPara::OpAttr>& attrs, void* compileInfo)
{
    return gert::TilingContextPara("RopeQuantKvcache",
                                   {
                                       {{{4, 1, 1280}, {4, 1, 1280}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                       {{{4, 1, 1, 128}, {4, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                       {{{4, 1, 1, 128}, {4, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                       {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                       {{{128}, {128}}, ge::DT_INT32, ge::FORMAT_ND},
                                       {{{4, 2048, 1, 128}, {4, 2048, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
                                       {{{4, 2048, 1, 128}, {4, 2048, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
                                       {{{4, 1}, {4, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                   },
                                   {
                                       {{{4, 1, 8, 128}, {4, 1, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                       {{{4, 1, 1, 128}, {4, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                       {{{4, 1, 1, 128}, {4, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                       {{{4, 2048, 1, 128}, {4, 2048, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
                                       {{{4, 2048, 1, 128}, {4, 2048, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
                                   },
                                   attrs, compileInfo);
}

class RopeQuantKvcacheInvalidAttrsTiling : public testing::TestWithParam<std::vector<int64_t>> {};

TEST(RopeQuantKvcacheMissingAttrs, Tiling)
{
    optiling::RopeQuantKvcacheCompileInfo compileInfo{};
    auto para = MakeContextPara({}, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_P(RopeQuantKvcacheInvalidAttrsTiling, InvalidSizeSplits)
{
    optiling::RopeQuantKvcacheCompileInfo compileInfo{};
    auto para = MakeContextPara(
        {{"size_splits", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(GetParam())}}, &compileInfo);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

INSTANTIATE_TEST_SUITE_P(InvalidSizeSplits, RopeQuantKvcacheInvalidAttrsTiling,
                         testing::Values(std::vector<int64_t>{}, std::vector<int64_t>{1024},
                                         std::vector<int64_t>{1024, 128}, std::vector<int64_t>{1024, 128, 128, 128}));
} // namespace
