/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "error_code_test_utils.h"

class DequantRopeQuantKvcache : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "DequantRopeQuantKvcache SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "DequantRopeQuantKvcache TearDown" << std::endl;
    }
};

static gert::InfershapeContextPara MakeValidInfershapeContext()
{
    return gert::InfershapeContextPara(
        "DequantRopeQuantKvcache",
        {{{{4, 1, 1280}, {4, 1, 1280}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 128}, {4, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 128}, {4, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 2048, 1, 128}, {4, 2048, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{4, 2048, 1, 128}, {4, 2048, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND}},
        {{"size_splits", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1024, 128, 128})},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("static")},
         {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"kv_output", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}});
}

TEST_F(DequantRopeQuantKvcache, DequantRopeQuantKvcache_infershape_case_0)
{
    gert::InfershapeContextPara infershapeContextPara(
        "DequantRopeQuantKvcache",
        {{{{4, 1, 1280}, {4, 1, 1280}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 128}, {4, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 1, 1, 128}, {4, 1, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 2048, 1, 128}, {4, 2048, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{4, 2048, 1, 128}, {4, 2048, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{128}, {128}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND}},
        {{"size_splits", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1024, 128, 128})},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("static")},
         {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"kv_output", Ops::Transformer::AnyValue::CreateFrom<bool>(false)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 1, 8, 128}};
    drqk_test::ClearErrorMessages();
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
    drqk_test::ExpectNoError();
}

TEST_F(DequantRopeQuantKvcache, DequantRopeQuantKvcache_infershape_case_1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "DequantRopeQuantKvcache",
        {{{{4, 1280}, {4, 1280}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 128}, {4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 128}, {4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{4, 2048, 1, 128}, {4, 2048, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{4, 2048, 1, 128}, {4, 2048, 1, 128}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
         {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{128}, {128}}, ge::DT_INT32, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND}},
        {{"size_splits", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1024, 128, 128})},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("static")},
         {"layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("BSND")},
         {"kv_output", Ops::Transformer::AnyValue::CreateFrom<bool>(true)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 8, 128}};
    drqk_test::ClearErrorMessages();
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
    drqk_test::ExpectNoError();
}

TEST_F(DequantRopeQuantKvcache, DequantRopeQuantKvcache_infershape_invalid_x_rank)
{
    auto para = MakeValidInfershapeContext();
    para.inputTensorDesc_[0] = gert::InfershapeContextPara::TensorDescription(
        {{{4, 1, 1, 1280}, {4, 1, 1, 1280}}, ge::DT_FLOAT16, ge::FORMAT_ND});
    drqk_test::ClearErrorMessages();
    ExecuteTestCase(para, ge::GRAPH_FAILED);
    drqk_test::ExpectSingleError("EZ0011", {{"param_name", "x"},
                                            {"op_name", "DequantRopeQuantKvcache"},
                                            {"incorrect_dim", "4"},
                                            {"correct_dim", "2D or 3D"}});
}

TEST_F(DequantRopeQuantKvcache, DequantRopeQuantKvcache_infershape_invalid_size_splits_length)
{
    auto para = MakeValidInfershapeContext();
    para.attrs_[0].attr_ = Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1024, 128});
    drqk_test::ClearErrorMessages();
    ExecuteTestCase(para, ge::GRAPH_FAILED);
    drqk_test::ExpectSingleError("EZ0025", {{"param_name", "size_splits"},
                                            {"op_name", "DequantRopeQuantKvcache"},
                                            {"incorrect_size", "2"},
                                            {"correct_size", "3"}});
}

TEST_F(DequantRopeQuantKvcache, DequantRopeQuantKvcache_infershape_negative_size_splits)
{
    auto para = MakeValidInfershapeContext();
    para.attrs_[0].attr_ = Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({1024, -128, 128});
    drqk_test::ClearErrorMessages();
    ExecuteTestCase(para, ge::GRAPH_FAILED);
    drqk_test::ExpectSingleError("EZ0026", {{"param_name", "size_splits[1]"},
                                            {"op_name", "DequantRopeQuantKvcache"},
                                            {"incorrect_value", "-128"},
                                            {"reason", "size_splits elements must be greater than or equal to 0"}});
}

TEST_F(DequantRopeQuantKvcache, DequantRopeQuantKvcache_infershape_null_context)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infershapeFunc = spaceRegistry->GetOpImpl("DequantRopeQuantKvcache")->infer_shape;
    drqk_test::ClearErrorMessages();
    EXPECT_EQ(infershapeFunc(nullptr), ge::GRAPH_FAILED);
    drqk_test::ExpectSingleError(
        "EZ1014", {{"opName", "DequantRopeQuantKvcache"}, {"reason", "InferShape context is unavailable"}});
}
