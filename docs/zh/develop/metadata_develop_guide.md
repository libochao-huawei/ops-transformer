# Metadata算子开发指南

## 概述

> 说明：
>
> 1. Metadata算子是一种前置调度算子：其在主算子（如`flash_attn`）执行之前运行，读取变长序列信息与形状等参数，完成任务切分与调度计算，并将切分与调度结果写入INT32的metadata张量，作为主算子的输入。
> 2. Metadata算子的开发方式与AI CPU算子一致，本指南仅介绍Metadata类算子**区别于普通AI CPU算子的部分**。算子开发涉及的基本概念、CpuKernel基类用法、算子注册、UT验证等重复内容，详情请参考[《AI CPU算子开发指南》](./aicpu_develop_guide.md)；主算子（AI Core侧）的Tiling/Kernel开发，详情请参考[《AI Core算子开发指南》](./aicore_develop_guide.md)。
> 3. build.sh：算子开发过程中涉及的命令可通过`bash build.sh --help`查看，功能参数介绍参考[build参数说明](../install/build.md)。

本指南以`FlashAttnMetadata`算子开发为例，介绍新算子开发流程以及涉及的交付件，完整示例代码请参考仓内`attention/flash_attn_metadata`目录。

### 背景与原理

随着主算子计算逻辑日趋复杂，对计算任务的组织与调度提出了更高要求。为此引入Metadata算子，在主算子执行之前完成任务切分与调度决策，实现调度逻辑与计算逻辑的解耦。

Metadata算子是一类运行于AI CPU的前置算子。其执行过程为：读取变长序列信息与形状等参数，执行任务切分算法，并将切分结果写入INT32的metadata张量；主算子启动后依据metadata所描述的任务区间执行计算。

该机制的核心价值包括：

1. **数据就近访问**：变长序列等信息的真实数值存储于Device侧，Host侧Tiling阶段仅能获取张量形状而无法读取数值；AI CPU同样运行于Device侧，可直接读取张量数值完成精确切分。切分结果（metadata张量）亦在Device侧生成并由主算子消费，全程无需Host与Device之间的数据搬运。
2. **执行流并行**：Metadata算子可在独立流中调度执行，与AI Core上除主算子外的其他算子的计算相互重叠，减少对关键计算路径的占用，提升NPU整体利用率。
3. **输出结果复用**：在多层attention等场景下，若各层输入的形状、变长、掩码、核数等信息保持不变，Metadata算子的输出可直接复用，无需重复执行切分计算。

### 指南导读

1. [工程创建](#工程创建)：开发算子前，需完成环境部署并创建算子目录。
2. [算子定义](#算子定义)：确定算子功能与原型定义。
3. [Kernel实现](#kernel实现)：实现Device侧（AI CPU）的分核计算与metadata张量写入。
4. [aclnn适配](#aclnn适配)：Metadata算子的两段式aclnn接口、参数校验。
5. [torch适配](#torch适配)：将aclnn封装为PyTorch算子。
6. [编译部署](#编译部署)：通过工程编译脚本完成算子编译和安装。
7. [负载均衡](#负载均衡)：介绍负载均衡的原理与处理流程。
8. [使用方式](#使用方式)：介绍Metadata算子的使用方式。
9. [注意事项](#注意事项)：Metadata算子开发的注意事项。

## 工程创建

环境部署与目录创建流程与AI CPU算子完全一致，详情请参考[《AI CPU算子开发指南》](./aicpu_develop_guide.md)的[工程创建](./aicpu_develop_guide.md#工程创建)章节。

## 算子定义

算子定义要求与AI CPU算子一致，详情请参考[《AI CPU算子开发指南》](./aicpu_develop_guide.md)的[算子定义](./aicpu_develop_guide.md#算子定义)章节。

## Kernel实现

Kernel实现与AI CPU算子一致，详情请参考[《AI CPU算子开发指南》](./aicpu_develop_guide.md)的[Kernel实现](./aicpu_develop_guide.md#Kernel实现)章节。

## aclnn适配

aclnn接口遵循两段式规范，与AI CPU算子完全一致，详情请参考[《AI CPU算子开发指南》](./aicpu_develop_guide.md)的[aclnn适配](./aicpu_develop_guide.md#aclnn适配)章节。

## torch适配

torch适配方式与AI CPU算子一致。Metadata算子作为前置算子与主算子高度绑定，建议将两个算子的torch层适配代码集中编写，通常置于主算子的torch适配文件中。在调用方式上，Metadata算子支持Aclgraph与GE入图，详情请参考[《图模式适配指南》](./graph_develop_guide.md)。

## 编译部署

编译部署与AI CPU算子一致，详情请参考[《AI CPU算子开发指南》](./aicpu_develop_guide.md)的[编译部署](./aicpu_develop_guide.md#编译部署)章节。

## 负载均衡

负载均衡是Metadata算子的核心职责：依据各任务的计算代价，将计算任务均衡地分配至参与计算的各核，以最大化硬件利用率、最小化整体计算延迟。切分算法可由开发者自行设计。本仓针对Attention类算子已提供成熟的负载均衡实现，详见`attention/common/op_kernel/load_balance`。

以TND（变长序列）场景为例，不同Batch的序列长度存在差异。下图假设Q/K/V头数均为1，任务空间按(Batch, QuerySeq, KVSeq)三维划分，用以示意负载均衡的任务切分方式。

![负载均衡示意图](../figures/load_balance_compute_example.png "负载均衡")

负载均衡的处理流程如下：

1. 依据Tiling确定的分块尺寸对输入进行划分，得到图中Batch维度上的各计算子块。
2. 选取合适的代价模型计算各子块的负载，并汇总得到总负载。
3. 以总负载与核数之比作为各核的平均负载，将子块均衡地分配至各核，使各核负载趋近于该平均值。
4. 任务切分可能使单个Batch的计算跨越多个核，须待全部核计算完成后执行归约，此类计算称为归约任务。归约任务的负载依据其涉及的QuerySeq与KVSeq规模计算，并分配至可用的AIV核执行。

## 使用方式

Metadata算子作为主算子的前序算子，调用时须遵循严格的先后顺序：先生成metadata，再由主算子消费。下面以`flash_attn`的torch接口调用为例进行说明。

```python
import torch
import torch_npu
from cann_ops_transformer.ops import flash_attn, flash_attn_metadata

dev = torch.device('npu:0')

dtype = torch.bfloat16
batch_size = 1
q_len = 1024
kv_len = 1024
q_heads = 1
kv_heads = 1
head_dim = 128
mask_mode = 0
win_left = -1
win_right = -1
softmax_scale = 1.0 / (head_dim ** 0.5)

metadata = torch.ops.cann_ops_transformer.flash_attn_metadata(
    num_heads_q=q_heads,
    num_heads_kv=kv_heads,
    head_dim=head_dim,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    batch_size=batch_size,
    max_seqlen_q=q_len,
    max_seqlen_kv=kv_len,
    mask_mode=mask_mode,
    win_left=win_left,
    win_right=win_right,
    layout_q='BNSD',
    layout_kv='BNSD',
    layout_out='BNSD',
)

q = torch.randn(batch_size, q_heads, q_len, head_dim, dtype=dtype, device=dev)
k = torch.randn(batch_size, kv_heads, kv_len, head_dim, dtype=dtype, device=dev)
v = torch.randn(batch_size, kv_heads, kv_len, head_dim, dtype=dtype, device=dev)

attn_out, softmax_lse = torch.ops.cann_ops_transformer.flash_attn(
    q,
    k,
    v,
    block_table=None,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    sinks=None,
    attn_mask=None,
    metadata=metadata,
    softmax_scale=softmax_scale,
    mask_mode=mask_mode,
    win_left=win_left,
    win_right=win_right,
    max_seqlen_q=q_len,
    max_seqlen_kv=kv_len,
    layout_q='BNSD',
    layout_kv='BNSD',
    layout_out='BNSD',
    return_softmax_lse=False,
)
torch_npu.npu.synchronize()
```

## 注意事项

1. **入参一致性**：Metadata算子与主算子以两个独立接口分段调用，主算子**无法校验**metadata是否与本次调用的参数匹配。调用时须确保两侧入参完全一致，否则将引发未定义行为（精度错误、非法内存访问、程序崩溃）。
2. **metadata张量布局一致性**：metadata张量的布局协议由写入侧（Metadata算子）与解析侧（主算子）共享，修改布局时两侧必须同步更新，否则将引发未定义行为（精度错误、非法内存访问、程序崩溃）。
3. **动态核数**：metadata张量的大小可能依赖AIC/AIV核数，计算张量尺寸时必须按实际核数进行适配。
4. **fallback函数**：Metadata算子可能不含Tensor入参，此时torch适配需注册fallback函数，以确保dispatcher能够正确路由至NPU实现。实现示例请参考`attention/flash_attn/torch_extension/flash_attn.py`。
5. **metadata缺省回退**：若主算子未接收到metadata，推荐在其Tiling侧预置一套简易的分核算法作为备选方案；当metadata未传入时，回退至该备选方案执行，以保证主算子功能可用。
