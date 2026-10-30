/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <initializer_list>
#include <utility>
#include "experiment/metadef/common/util/error_manager/error_manager.h"
#include "../matmul_reduce_scatter_v2_host_ut_param.h"
#include "mc2_tiling_case_executor.h"

namespace matmul_reduce_scatter_v2_ut {

namespace {
template <typename Reports>
void CheckFormalReport(const Reports& reports, const std::string& errorId,
                       std::initializer_list<std::pair<std::string, std::string>> exactFields,
                       std::initializer_list<std::string> nonEmptyFields = {},
                       std::initializer_list<std::pair<std::string, std::string>> containsFields = {})
{
    ASSERT_EQ(reports.size(), 1U);
    ASSERT_EQ(reports[0].error_id, errorId);
    const auto& fields = reports[0].args_map;
    for (const auto& item : exactFields) {
        ASSERT_EQ(fields.count(item.first), 1U) << item.first;
    }
    for (const auto& key : nonEmptyFields) {
        ASSERT_EQ(fields.count(key), 1U) << key;
    }
    for (const auto& item : containsFields) {
        ASSERT_EQ(fields.count(item.first), 1U) << item.first;
    }
    for (const auto& item : exactFields) {
        EXPECT_EQ(fields.at(item.first), item.second) << item.first;
    }
    for (const auto& key : nonEmptyFields) {
        EXPECT_FALSE(fields.at(key).empty()) << key;
    }
    for (const auto& item : containsFields) {
        EXPECT_NE(fields.at(item.first).find(item.second), std::string::npos) << item.first;
    }
}

} // namespace

struct MatmulReduceScatterV2Arch22Param : public MatmulReduceScatterV2TilingUtParam {
    std::string expectedErrorId;
    std::string expectedParamName;
    explicit MatmulReduceScatterV2Arch22Param(const csv_map& csvMap)
        : MatmulReduceScatterV2TilingUtParam(csvMap),
          expectedErrorId(ReadMap(csvMap, "expectedErrorId")),
          expectedParamName(ReadMap(csvMap, "expectedParamName"))
    {}
};

class MatmulReduceScatterV2Arch22TilingTest : public testing::TestWithParam<MatmulReduceScatterV2Arch22Param> {
protected:
    void SetUp() override
    {
        Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();
        (void)ErrorManager::GetInstance().GetRawErrorMessages();
    }
    void TearDown() override
    {
        Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();
        (void)ErrorManager::GetInstance().GetRawErrorMessages();
    }

    static void SetUpTestCase()
    {
        std::cout << "MatmulReduceScatterV2 Arch22TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MatmulReduceScatterV2 Arch22TilingTest TearDown" << std::endl;
    }
};

TEST_P(MatmulReduceScatterV2Arch22TilingTest, param)
{
    auto param = GetParam();
    struct MatmulReduceScatterV2CompileInfo {};
    MatmulReduceScatterV2CompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            param.x1,
            param.x2,
            param.bias,
            param.x1_scale,
            param.x2_scale,
            param.quant_scale,
        },
        {
            param.y,
            param.amax_out,
        },
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.group)},
         {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.reduce_op)},
         {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_trans_a)},
         {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_trans_b)},
         {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.comm_turn)},
         {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.rank_size)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.block_size)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.group_size)},
         {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_amax_out)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.y_dtype)},
         {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.comm_mode)}},
        param.inputInstance, param.outputInstance, &compileInfo, param.soc, param.coreNum, param.ubsize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", param.rank_size}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, param.expectResult, param.expectTilingKey,
                       param.expectTilingDataHash, {}, MC2_TILING_DATA_RESERVED_LEN, true);
    if (!param.expectedErrorId.empty()) {
        ASSERT_TRUE(param.expectedErrorId == "EZ0004" || param.expectedErrorId == "EZ0020") << param.expectedErrorId;
        const auto reports = ErrorManager::GetInstance().GetRawErrorMessages();
        if (param.expectedErrorId == "EZ0020") {
            CheckFormalReport(reports, param.expectedErrorId, {{"param_name", param.expectedParamName}},
                              {"incorrect_dtype"}, {{"reason", "FLOAT16"}});
        } else {
            CheckFormalReport(reports, param.expectedErrorId, {{"param_name", param.expectedParamName}});
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    MatmulReduceScatterV2, MatmulReduceScatterV2Arch22TilingTest,
    testing::ValuesIn(GetCasesFromCsv<MatmulReduceScatterV2Arch22Param>(ReplaceFileExtension2Csv(__FILE__))),
    GetCaseInfoString<MatmulReduceScatterV2Arch22Param>);

} // namespace matmul_reduce_scatter_v2_ut
