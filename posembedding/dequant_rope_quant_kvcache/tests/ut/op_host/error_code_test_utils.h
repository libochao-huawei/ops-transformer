/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DEQUANT_ROPE_QUANT_KVCACHE_ERROR_CODE_TEST_UTILS_H
#define DEQUANT_ROPE_QUANT_KVCACHE_ERROR_CODE_TEST_UTILS_H

#include <map>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "experiment/metadef/common/util/error_manager/error_manager.h"

namespace drqk_test {
inline void ClearErrorMessages()
{
    (void)ErrorManager::GetInstance().GetRawErrorMessages();
}

inline void ExpectNoError()
{
    auto errors = ErrorManager::GetInstance().GetRawErrorMessages();
    EXPECT_TRUE(errors.empty());
}

inline void ExpectSingleError(const std::string& expectedCode, const std::map<std::string, std::string>& expectedArgs)
{
    auto errors = ErrorManager::GetInstance().GetRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors.front().error_id, expectedCode);
    for (const auto& expected : expectedArgs) {
        const auto actual = errors.front().args_map.find(expected.first);
        ASSERT_NE(actual, errors.front().args_map.end());
        EXPECT_EQ(actual->second, expected.second);
    }
}
} // namespace drqk_test

#endif // DEQUANT_ROPE_QUANT_KVCACHE_ERROR_CODE_TEST_UTILS_H
