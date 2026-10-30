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
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"

namespace {
gert::InfershapeContextPara MakeContextPara(const std::vector<gert::InfershapeContextPara::OpAttr>& attrs)
{
    return gert::InfershapeContextPara("RopeQuantKvcache",
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
                                           {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                           {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                           {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                           {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
                                           {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
                                       },
                                       attrs);
}

class RopeQuantKvcacheInvalidAttrsInferShape : public testing::TestWithParam<std::vector<int64_t>> {};

TEST(RopeQuantKvcacheMissingAttrs, InferShape)
{
    auto para = MakeContextPara({});
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST_P(RopeQuantKvcacheInvalidAttrsInferShape, InvalidSizeSplits)
{
    auto para =
        MakeContextPara({{"size_splits", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(GetParam())}});
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

INSTANTIATE_TEST_SUITE_P(InvalidSizeSplits, RopeQuantKvcacheInvalidAttrsInferShape,
                         testing::Values(std::vector<int64_t>{}, std::vector<int64_t>{1024},
                                         std::vector<int64_t>{1024, 128}, std::vector<int64_t>{1024, 128, 128, 128}));
} // namespace
