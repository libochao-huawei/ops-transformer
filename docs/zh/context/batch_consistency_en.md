# Batch Consistency

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T13:18:07.984Z -->

## Introduction

During application development, to achieve higher performance, partial operators may produce inconsistent computing results for the same Token across different batch sizes or at different positions within the same batch.
Currently, for partial operators, provided that the running environment conditions are the same, you can configure the computing process to use the Batch consistency algorithm so that the computing results remain exactly the same no matter how the inputs are combined.
**Batch consistency algorithm**: For a given Token, the output must be bit-for-bit identical regardless of the Token's position in the batch, the batch size, or the other Tokens batched together with it.

## Notes

- Generally, it is recommended not to enable Batch consistency, because enabling Batch consistency computation for an operator causes some performance degradation, and therefore the single-run performance of the model may decrease. However, in use cases such as experiments, debugging, and regression testing, where multiple runs must produce the same results for problem locating and algorithm testing, Batch consistency computation improves efficiency.

- The current configuration is a process-level switch configuration.

- **Version constraints**: TorchNPU 26.2.0 or later and CANN 9.2.0 or later.

## Usage

Currently, CANN operators are mainly called through the aclnn API or PyTorch API (torch_extension). Partial operator APIs implement Batch consistency by default, while others implement non-Batch consistency by default. For operators with non-Batch consistency implementations, some can enable Batch consistency through manual configuration.

- **Calling the aclnn API**

  In this use case, configure Batch consistency through the "Runtime Configuration > aclrtSetSysParamOpt" API (process-level) in [*Runtime API*](https://hiascend.com/en/document/redirect/CannCommunityRuntimeApi). Specifically, set `ACL_OPT_DETERMINISTIC=3` to enable Batch consistency computation.

- **Calling the PyTorch API**

  In this use case, enable Batch consistency computation through the `torch_npu.npu.set_deterministic_level` API in [*TorchNPU Custom APIs*](https://www.hiascend.com/document/detail/en/Pytorch/latest/apiref/customapi/docs/en/custom_APIs/overview.md).

For operator APIs of different frameworks, refer to the specific aclnn API or PyTorch API documentation for the default Batch consistency computation implementation strategy.
