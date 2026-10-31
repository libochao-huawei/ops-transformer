/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include "register/op_def_registry.h"

namespace ops {
class PoolKeyIndexer : public OpDef {
public:
    explicit PoolKeyIndexer(const char *name)
        : OpDef(name)
    {
        // ---- Base config: A2/A3 (ascend910b, ascend910_93) ----
        // 3 combinations by position:
        //   1) FP16  quant_mode=-1 (no quant) : q/k/weights=FP16,    descale 不传入(占位不生效)
        //   2) BF16  quant_mode=-1 (no quant) : q/k/weights=BF16,    descale 不传入(占位不生效)
        //   3) INT8  quant_mode=2  (int8)     : q/k=INT8, weights/descale=FP16
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8})
            .FormatList({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("pool_key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8})
            .FormatList({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("pool_tail_k")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .ValueDepend(OPTIONAL)
            .AutoContiguous();
        this->Input("actual_seq_q")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .ValueDepend(OPTIONAL)
            .AutoContiguous();
        this->Input("actual_seq_k")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .ValueDepend(OPTIONAL)
            .AutoContiguous();
        this->Input("block_table")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        // descale 为 OPTIONAL: 多组合 dtype 对齐须用 DataType(逐组合枚举),
        // DataTypeList 多元素会触发 opbuild 段错误(对齐 fused_infer_attention_score 的做法)
        this->Input("q_descale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("k_descale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Output("sparse_indices")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Output("sparse_values")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        this->Attr("topk").AttrType(OPTIONAL).Int(2048);
        this->Attr("pool_size").AttrType(OPTIONAL).Int(16);
        this->Attr("layout_q").AttrType(OPTIONAL).String("BSND");
        this->Attr("layout_k").AttrType(OPTIONAL).String("BSND");
        this->Attr("mask_mode").AttrType(OPTIONAL).Int(3);
        this->Attr("quant_mode").AttrType(OPTIONAL).Int(-1);
        this->Attr("return_value").AttrType(OPTIONAL).Bool(false);
        // pool_key 0轴非连续 stride(元素单位), 由 torch_extension 层从
        // at::Tensor::stride(0) 直读传入(参考 compressor 方案); -1 表示未指定
        // (连续输入), tiling 回退到 shape 推导。
        this->Attr("key_stride0").AttrType(OPTIONAL).Int(-1);
        // k_descale 0轴非连续 stride(元素单位, 量化场景), 与 key_stride0 同机制:
        // eager/aclnn 路径 GetDynamicInputStride 返回 nullptr, 属性是唯一来源;
        // -1 表示未指定(连续输入), tiling 回退到 shape 推导。
        this->Attr("k_descale_stride0").AttrType(OPTIONAL).Int(-1);
        // 索引输出布局模式(尾块追加模式):
        //   0 = legacy(默认): tail 固定偏移 topk, 短序列 [expandTotal, topk) 为 -1 缝隙,
        //       与历史版本输出逐比特一致(V1 兼容签名 aclnnPoolKeyIndexer 不暴露本属性,
        //       恒以默认值运行, 模型侧存量调用零感知)
        //   1 = compact    : tail 紧跟有效展开 token, -1 全部位于行尾(消除中间空洞)
        // def 声明全量原型, opbuild 生成单一 inner 接口 aclnnInnerPoolKeyIndexer;
        // 外层 aclnnPoolKeyIndexer(V1 兼容签名, 冻结 legacy)/aclnnPoolKeyIndexerV2
        // (全量原型)由 op_api 手写提供(参考 lightning_indexer_v2 手写 aclnn 结构);
        // GE 图模式经 op_graph proto 声明可直接设置。两种模式有效 token 集合一致。
        this->Attr("sparse_indices_mode").AttrType(OPTIONAL).Int(0);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend910b", aicoreConfig);
        this->AICore().AddConfig("ascend910_93", aicoreConfig);

        // ---- 950 config: override only the inputs that differ from base (FP8 combos) ----
        //   7 combinations:
        //   1) FP16  quant_mode=-1 (no quant)     : q/k/weights=FP16,    descale=FLOAT
        //   2) BF16  quant_mode=-1 (no quant)     : q/k/weights=BF16,    descale=FLOAT
        //   3) FP8   quant_mode=0  (per-token FP8) : q/k=FP8_E4M3FN, weights=FP16, descale=FLOAT
        //   4) FP8   quant_mode=0  (per-token FP8) : q/k=FP8_E4M3FN, weights=BF16, descale=FLOAT
        //   5) FP8   quant_mode=1  (mxFP8)         : q/k=FP8_E4M3FN, weights=FP16, descale=FLOAT8_E8M0
        //   6) FP8   quant_mode=1  (mxFP8)         : q/k=FP8_E4M3FN, weights=BF16, descale=FLOAT8_E8M0
        //   7) INT8  quant_mode=2  (int8)          : q/k=INT8, weights=FP16, descale=FP16
        // Note: pool_tail_k / actual_seq_q / actual_seq_k (DT_INT64 + ValueDepend) and
        //       block_table / outputs inherit from the top-level declaration; not
        //       re-declared here. INT64 is the canonical host-value dtype required by
        //       the ValueDepend convention (see fused_infer_attention_score et al.).
        // Note: 量化场景 weights 规格(文档+tiling)允许 FP16/BF16, kernel entry 按
        //       ORIG_DTYPE_WEIGHTS 分发 W_T(half/bfloat16_t), 与 quant_lightning_indexer
        //       的 weights 双 dtype 模式一致; int8 组合 weights 仅 FP16。
        OpAICoreConfig aicoreConfig950;
        aicoreConfig950.Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_INT8})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicoreConfig950.Input("pool_key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_INT8})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        aicoreConfig950.Input("weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicoreConfig950.Input("q_descale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT8_E8M0,
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicoreConfig950.Input("k_descale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT8_E8M0,
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        aicoreConfig950.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend950", aicoreConfig950);
    }
};
OP_ADD(PoolKeyIndexer);
} // namespace ops
