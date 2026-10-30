# Deterministic Computation

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T13:24:37.651Z -->

## Introduction

During application development, due to differences such as CANN versions or NPU models, results of multiple runs of the same operator may not be exactly the same; even under the same conditions with the same random seed, results of multiple NPU executions may differ. This difference usually stems from asynchronous multi-threaded execution in operator implementations, which changes the order of floating-point accumulation.

In some special use cases (such as experiments, debugging, and regression tests), operators need to produce the same output across multiple runs to locate problems or perform algorithm analysis. This approach is generally called "deterministic computation".

## Notes

- Enabling deterministic computation is generally not recommended, because the same operator generally executes more slowly with deterministic computation than with non-deterministic computation, which may degrade the performance of a single model run. However, in use cases such as experiments, debugging, and regression tests, where consistent results across multiple runs are required to locate problems or analyze algorithms, deterministic computation can improve efficiency.

- Thread notes: In the same thread, the deterministic state can be set only once; if it is set multiple times, the last valid setting prevails. A valid setting means that after the deterministic state is set, an operator task is actually dispatched. If only the setting is made without dispatching an operator task, the deterministic variable is enabled but not delivered to the operator, and the operator will not execute. Setting deterministic computation multiple times in one thread is not recommended for now. This issue exists regardless of whether the binary is enabled or disabled, and will be fixed in later versions.

## Usage

Currently, CANN operators are mainly called through the aclnn API or the PyTorch API (torch_extension). Some operator APIs default to a deterministic implementation, and others default to a non-deterministic implementation. For operators with a non-deterministic implementation, deterministic computation can be enabled through manual configuration for some of them.

- **Calling the aclnn API**

  In this use case, enable deterministic computation by configuring the "Runtime Configuration > aclrtSetSysParamOpt" API (process-level) in [*Runtime APIs*](https://hiascend.com/en/document/redirect/CannCommunityRuntimeApi). Specifically, set `ACL_OPT_DETERMINISTIC=1`.

- **Calling the PyTorch API**

  In this use case, enable deterministic computation through the native PyTorch switch (`torch.use_deterministic_algorithms`).

For operator APIs of different frameworks, refer to the specific aclnn API documentation or PyTorch API documentation for the default implementation strategy of deterministic computation.
