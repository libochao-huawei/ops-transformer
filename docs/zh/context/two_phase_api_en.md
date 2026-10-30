# Two-Phase API

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T13:25:18.491Z -->

When calling an operator API based on the single-operator API execution mode, the process typically follows a "two-phase" pattern, as shown below:

```cpp
aclnnStatus aclxxXxxGetWorkspaceSize(const aclTensor *src, ..., aclTensor *out, ..., uint64_t *workspaceSize, aclOpExecutor **executor);
aclnnStatus aclxxXxx(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
```

You must first call the first-stage interface aclxxXxxGetWorkspaceSize to calculate the amount of workspace memory required for this API call. After obtaining the required workspaceSize, allocate NPU memory based on workspaceSize, and then call the second-stage interface aclxxXxx to perform the computation.

Here, "aclxx" represents the operator interface prefix, such as aclnn, and "Xxx" represents the corresponding operator type, such as the Add operator.

> NOTE:
>
> - workspace refers to the temporary memory, excluding inputs/outputs, required by an operator to complete computation on the NPU. workspaceSize indicates the size of the temporary memory.
> - The second-phase API aclxxXxx(...) cannot be called repeatedly. An exception occurs in the following call pattern:
>
>   ```cpp
>   aclxxXxxGetWorkspaceSize(...)
>   aclxxXxx(...)
>   aclxxXxx(...)
