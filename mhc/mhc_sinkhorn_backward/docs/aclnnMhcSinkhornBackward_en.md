# aclnnMhcSinkhornBackward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:02:08.818Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: MhcSinkhornBackward is the backward operator of MhcSinkhorn. The MhcSinkhorn operator in the mHC (Manifold-Constrained Hyper-Connections) architecture performs the sinkhorn transform on the input matrix to obtain a doubly stochastic matrix $\mathbf{H}_{\text{res}}$. All elements of the output doubly stochastic matrix are ≥ 0, the sum of each row is 1, and the sum of each column is 1 (it has three key properties: norm preservation, closure under composition, and a geometric interpretation in terms of convex combinations). The gradient of the doubly stochastic matrix $\mathbf{H}_{\text{res}}$ in the mHC architecture is fed into the backward computation of the sinkhorn transform to obtain the gradient of the input $\mathbf{H}'_{\text{res}}$.

- Formula

  - The Sinkhorn-Knopp algorithm achieves doubly stochastic projection through $\mathbf{num\_iters}$ iterations of normalization in forward computation. In the iterative computation of backward propagation:

  - The first $\mathbf{num\_iters}-1$ iterations:

  $$
  \begin{aligned}
      \mathbf{dot\_prod}_{2i+1} &= \sum_{\dim=-2,\text{keepdim}=\text{True}} (\mathbf{grad}_{curr}\ {⋅}\ \mathbf{norm\_out}_{2i+1}),  \\
      \mathbf{grad}_{curr} &← \frac{\mathbf{grad}_{curr} - \mathbf{dot\_prod}_{2i+1}}{\mathbf{sum\_out}_{2i+1}}, \\
      \mathbf{dot\_prod}_{2i} &= \sum_{\dim=-1,\text{keepdim}=\text{True}} (\mathbf{grad}_{curr}\ {⋅}\ \mathbf{norm\_out}_{2i}),  \\
      \mathbf{grad}_{curr} &← \frac{\mathbf{grad}_{curr} - \mathbf{dot\_prod}_{2i}}{\mathbf{sum\_out}_{2i}}, \\
  \end{aligned}
  $$

  - The last iteration:

  $$
  \begin{aligned}
      \mathbf{dot\_prod}_{1} &= \sum_{\dim=-2,\text{keepdim}=\text{True}} (\mathbf{grad}_{curr}\ {⋅}\ \mathbf{norm\_out}_{1}),  \\
      \mathbf{grad}_{curr} &← \frac{\mathbf{grad}_{curr} - \mathbf{dot\_prod}_{1}}{\mathbf{sum\_out}_{1}}, \\
      \mathbf{dot\_prod}_{0} &= \sum_{\dim=-1,\text{keepdim}=\text{True}} (\mathbf{grad}_{curr}\  {⋅}\  \mathbf{norm\_out}_{0}),  \\
      \mathbf{grad}_{input} &← ({\mathbf{grad}_{curr} - \mathbf{dot\_prod}_{0}})\  {⋅}\  \mathbf{norm\_out}_{0} \\
  \end{aligned}
  $$

  - Where: $\mathbf{grad}_\text{curr}$ is the initial gradient, $\mathbf{grad}_\text{input}$ is the output gradient, $\mathbf{norm\_out}_\text{k}$ is the normalization direction vector of the $k$-th iteration, and $\mathbf{sum\_out}_\text{k}$ is the corresponding scaling coefficient.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnMhcSinkhornBackwardGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor containing the operator computation process, and then call the "aclnnMhcSinkhornBackward" API to perform the computation.

```c++
aclnnStatus aclnnMhcSinkhornBackwardGetWorkspaceSize(
    const aclTensor* gradOutput,
    const aclTensor* normOut,
    const aclTensor* sumOut,
    aclTensor*       out,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```c++
aclnnStatus aclnnMhcSinkhornBackward(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
```

## aclnnMhcSinkhornBackwardGetWorkspaceSize

- **Parameters:**

  | Name | Input/Output | Description | Usage Instructions | Data Type | Data Format | Dimension (Shape) | Non-contiguous Tensor |
  |:--- |:--- |:--- |:--- |:--- |:--- |:--- |:--- |
  | gradOutput | Input | Initial gradient of the Sinkhorn transform output. | empty Tensor not supported | FLOAT32 | ND | (B,S,n,n), (T,n,n) | √ |
  | normOut| Input | Intermediate result norm saved in the forward computation of the Sinkhorn transform.| empty Tensor not supported | FLOAT32 | ND | (2\*num_iters\*n\*align_n\*B\*S), (2\*num_iters\*n\*align_n\*T) | √ |
  | sumOut| Input | Intermediate result sum saved in the forward computation of the Sinkhorn transform. | empty Tensor not supported | FLOAT32 | ND | (2\*num_iters\*align_n\*B\*S), (2\*num_iters\*align_n\*T) | √ |
  | out| Output | Gradient of the input $\mathbf{H}_{\text{res}}$ to the Sinkhorn transform. | empty Tensor not supported | FLOAT32 | ND | (B,S,n,n), (T,n,n) | √ |
  | workspaceSize | Output | Returns the workspace size to be applied on the Device side. | - | - | - | - | - |
  | executor | Output | Returns the op executor, which contains the operator computation flow. | - | - | - | - | - |

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table>
    <thead>
      <tr>
        <th style="width: 250px">Return Value</th>
        <th style="width: 130px">Error Code</th>
        <th style="width: 850px">Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td rowspan="1"> ACLNN_ERR_PARAM_NULLPTR </td>
        <td rowspan="1"> 161001 </td>
        <td> gradOutput, normOut, sumOut, or out is a null pointer. </td>
      </tr>
      <tr>
        <td rowspan="1"> ACLNN_ERR_PARAM_INVALID </td>
        <td rowspan="1"> 161002 </td>
        <td> The data type or data format of input variables such as gradOutput, normOut, or sumOut is out of the supported range. </td>
      </tr>
    </tbody>
  </table>

## aclnnMhcSinkhornBackward

- **Parameters:**

  | Name | Input/Output | Description |
  |:--- |:--- |:--- |
  | workspace | Input | Device-side allocated workspace memory address. |
  | workspaceSize | Input | Size of the workspace memory allocated on the device, obtained by calling the first-phase API aclnnMhcSinkhornBackwardGetWorkspaceSize. |
  | executor | Input | Op executor that contains the operator computation flow. |
  | stream | Input | Stream on which the task is executed.|

- **Return value**

  Returns a status code of the `aclnnStatus` type. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- **Deterministic computation:**

  aclnnMhcSinkhornBackward defaults to a deterministic implementation.

- **Specification constraints:**

  - num_iters: value range 1-100. An invalid parameter error is reported if the value is out of range.
  - n: sizes of the last two dimensions of the input matrix. Only 4, 6, or 8 is supported.
  - align_n: fixed to 8. It is the value of n aligned to the FP32 block size of 32 bytes.
  - Number of dimensions: only 3D (T,n,n) or 4D (B,S,n,n) is supported for the input gradOutput.

## Example

  The following sample code is for reference only. For details about how to compile and run it, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

  ```c++
  #include <iostream>
  #include <vector>
  #include <cmath>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_mhc_sinkhorn_backward.h"

  #define CHECK_RET(cond, return_expr) \
    do {                               \
      if (!(cond)) {                   \
        return_expr;                   \
      }                                \
    } while (0)

  #define LOG_PRINT(message, ...)     \
    do {                              \
      printf(message, ##__VA_ARGS__); \
    } while (0)

  // Compute the total number of elements for the Tensor shape.
  int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t size = 1;
    for (int64_t dim : shape) {
      size *= dim;
    }
    return size;
  }

  // Round up to a multiple of align.
  int64_t CeilAlign(int64_t value, int64_t align) {
    return (value + align - 1) / align * align;
  }

  // Copy the Tensor data from the Device side to the Host side and print it.
  void PrintTensorData(const std::vector<int64_t>& shape, void* device_addr, const char* name) {
    int64_t size = GetShapeSize(shape);
    std::vector<float> host_data(size, 0.0f);

    // Device -> Host data copy
    aclError ret = aclrtMemcpy(
        host_data.data(), size * sizeof(float),
        device_addr, size * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST
    );
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("Memcpy device to host failed, error: %d\n", ret);
              return);

    // Print the first 10 elements.
    LOG_PRINT("%s (first 10 elements): ", name);
    for (int i = 0; i < std::min((int64_t)10, size); ++i) {
      LOG_PRINT("%.6f ", host_data[i]);
    }
    LOG_PRINT("\n");
  }

  // Initialize the AscendCL environment (Device/Context/Stream).
  int InitAcl(int32_t device_id, aclrtContext& context, aclrtStream& stream) {
    // 1. Initialize ACL.
    aclError ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclInit failed, error: %d\n", ret);
              return -1);

    // 2. Set the Device.
    ret = aclrtSetDevice(device_id);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtSetDevice failed, error: %d\n", ret);
              return -1);

    // 3. Create the Context.
    ret = aclrtCreateContext(&context, device_id);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtCreateContext failed, error: %d\n", ret);
              return -1);

    // 4. Set the current Context.
    ret = aclrtSetCurrentContext(context);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtSetCurrentContext failed, error: %d\n", ret);
              return -1);

    // 5. Create a Stream.
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtCreateStream failed, error: %d\n", ret);
              return -1);

    return 0;
  }

  // Create the device-side aclTensor (including data copy).
  int CreateAclTensor(
      const std::vector<float>& host_data,
      const std::vector<int64_t>& shape,
      void*& device_addr,
      aclTensor*& tensor) {
    // 1. Calculate the memory size.
    int64_t size = GetShapeSize(shape) * sizeof(float);

    // 2. Allocate device-side memory.
    aclError ret = aclrtMalloc(&device_addr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtMalloc failed, error: %d\n", ret);
              return -1);

    // 3. Copy data from host to device.
    ret = aclrtMemcpy(
        device_addr, size,
        host_data.data(), size,
        ACL_MEMCPY_HOST_TO_DEVICE
    );
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtMemcpy failed, error: %d\n", ret);
              return -1);

    // 4. Calculate the strides of the Tensor (contiguous Tensor).
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }

    // 5. Create an aclTensor.
    tensor = aclCreateTensor(
        shape.data(), shape.size(),
        ACL_FLOAT, strides.data(), 0,
        ACL_FORMAT_ND, shape.data(), shape.size(),
        device_addr
    );
    CHECK_RET(tensor != nullptr,
              LOG_PRINT("aclCreateTensor failed\n");
              return -1);

    return 0;
  }

  // Create the output Tensor (only allocate memory, no initial data).
  int CreateOutputTensor(
      const std::vector<int64_t>& shape,
      void*& device_addr,
      aclTensor*& tensor) {
    int64_t size = GetShapeSize(shape) * sizeof(float);

    aclError ret = aclrtMalloc(&device_addr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtMalloc output failed, error: %d\n", ret);
              return -1);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }

    tensor = aclCreateTensor(
        shape.data(), shape.size(),
        ACL_FLOAT, strides.data(), 0,
        ACL_FORMAT_ND, shape.data(), shape.size(),
        device_addr
    );
    CHECK_RET(tensor != nullptr,
              LOG_PRINT("aclCreateTensor output failed\n");
              return -1);

    return 0;
  }

  int main() {
    // ========== 1. Initialize the environment. ==========
    int32_t device_id = 0;  // Adjust based on the actual Device ID.
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;

    int ret = InitAcl(device_id, context, stream);
    CHECK_RET(ret == 0,
              LOG_PRINT("InitAcl failed, error: %d\n", ret);
              return -1);

    LOG_PRINT("ACL environment initialized successfully!\n");

    // ========== 2. Define operator parameters ==========
    // The grad_y shape can be (T, N, N) or (B, S, N, N).
    // The T shape is used as an example here.
    int64_t T = 128;       // total length
    int64_t N = 4;         // n size (N x N matrix)
    int64_t num_iters = 20; // Number of Sinkhorn iterations

    // Calculate the aligned N size (aligned to 8).
    int64_t align_unit = 8;
    int64_t n_align_size = CeilAlign(N, align_unit);

    LOG_PRINT("Parameters: T=%lld, N=%lld, num_iters=%lld, n_align_size=%lld\n",
              T, N, num_iters, n_align_size);

    // ========== 3. Construct input/output tensors ==========
    // Input grad_y shape: (T, N, N)
    std::vector<int64_t> grad_y_shape = {T, N, N};
    int64_t grad_y_size = GetShapeSize(grad_y_shape);
    std::vector<float> grad_y_host_data(grad_y_size, 1.0f);
    for (int64_t i = 0; i < grad_y_size; ++i) {
      grad_y_host_data[i] = 0.1f * static_cast<float>(i % 100);  // Fill test data.
    }

    // Input norm shape: [2 * num_iters * T * N * n_align_size]
    int64_t norm_size = 2 * num_iters * T * N * n_align_size;
    std::vector<int64_t> norm_shape = {norm_size};
    std::vector<float> norm_host_data(norm_size, 0.5f);
    for (int64_t i = 0; i < norm_size; ++i) {
      norm_host_data[i] = 1.0f / (1.0f + static_cast<float>(i % 1000) * 0.001f);  // Populate test data.
    }

    // Input sum shape: [2 * num_iters * T * n_align_size]
    int64_t sum_size = 2 * num_iters * T * n_align_size;
    std::vector<int64_t> sum_shape = {sum_size};
    std::vector<float> sum_host_data(sum_size, 1.0f);
    for (int64_t i = 0; i < sum_size; ++i) {
      sum_host_data[i] = 1.0f + static_cast<float>(i % 500) * 0.01f;  // Populate test data.
    }

    // Output grad_input shape, same as grad_y: (T, N, N)
    std::vector<int64_t> grad_input_shape = grad_y_shape;

    // Create the Device-side Tensor.
    void* grad_y_device_addr = nullptr;
    aclTensor* grad_y_tensor = nullptr;
    ret = CreateAclTensor(grad_y_host_data, grad_y_shape, grad_y_device_addr, grad_y_tensor);
    CHECK_RET(ret == 0, LOG_PRINT("Create grad_y_tensor failed\n"); return -1);

    void* norm_device_addr = nullptr;
    aclTensor* norm_tensor = nullptr;
    ret = CreateAclTensor(norm_host_data, norm_shape, norm_device_addr, norm_tensor);
    CHECK_RET(ret == 0, LOG_PRINT("Create norm_tensor failed\n"); return -1);

    void* sum_device_addr = nullptr;
    aclTensor* sum_tensor = nullptr;
    ret = CreateAclTensor(sum_host_data, sum_shape, sum_device_addr, sum_tensor);
    CHECK_RET(ret == 0, LOG_PRINT("Create sum_tensor failed\n"); return -1);

    void* grad_input_device_addr = nullptr;
    aclTensor* grad_input_tensor = nullptr;
    ret = CreateOutputTensor(grad_input_shape, grad_input_device_addr, grad_input_tensor);
    CHECK_RET(ret == 0, LOG_PRINT("Create grad_input_tensor failed\n"); return -1);

    LOG_PRINT("All tensors created successfully!\n");
    LOG_PRINT("grad_y shape: [%lld, %lld, %lld]\n", grad_y_shape[0], grad_y_shape[1], grad_y_shape[2]);
    LOG_PRINT("norm shape: [%lld]\n", norm_shape[0]);
    LOG_PRINT("sum shape: [%lld]\n", sum_shape[0]);
    LOG_PRINT("grad_input shape: [%lld, %lld, %lld]\n", grad_input_shape[0], grad_input_shape[1], grad_input_shape[2]);

    // ========== 4. Call the first-phase API: obtain the Workspace size ==========
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclnn_ret = aclnnMhcSinkhornBackwardGetWorkspaceSize(
        grad_y_tensor,
        norm_tensor,
        sum_tensor,
        grad_input_tensor,
        &workspace_size,
        &executor
    );
    CHECK_RET(aclnn_ret == ACL_SUCCESS,
              LOG_PRINT("aclnnMhcSinkhornBackwardGetWorkspaceSize failed, error: %d\n", aclnn_ret);
              return -1);

    LOG_PRINT("Workspace size: %lu bytes\n", workspace_size);

    // ========== 5. Allocate Workspace memory ==========
    void* workspace_addr = nullptr;
    if (workspace_size > 0) {
      ret = aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclrtMalloc workspace failed, error: %d\n", ret);
                return -1);
    }

    // ========== 6. Call the second-phase API: run the MhcSinkhornBackward computation ==========
    aclnn_ret = aclnnMhcSinkhornBackward(
        workspace_addr,
        workspace_size,
        executor,
        stream
    );
    CHECK_RET(aclnn_ret == ACL_SUCCESS,
              LOG_PRINT("aclnnMhcSinkhornBackward failed, error: %d\n", aclnn_ret);
              return -1);

    // ========== 7. Synchronize the stream ==========
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtSynchronizeStream failed, error: %d\n", ret);
              return -1);

    LOG_PRINT("MhcSinkhornBackward compute success!\n");

    // ========== 8. Print the result ==========
    PrintTensorData(grad_input_shape, grad_input_device_addr, "grad_input");

    // ========== 9. Release resources ==========
    // Destroy the Tensor
    aclDestroyTensor(grad_y_tensor);
    aclDestroyTensor(norm_tensor);
    aclDestroyTensor(sum_tensor);
    aclDestroyTensor(grad_input_tensor);

    // Release the Device memory
    aclrtFree(grad_y_device_addr);
    aclrtFree(norm_device_addr);
    aclrtFree(sum_device_addr);
    aclrtFree(grad_input_device_addr);
    if (workspace_size > 0) {
      aclrtFree(workspace_addr);
    }

    // Destroy the Stream/Context and reset the Device
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(device_id);
    aclFinalize();

    LOG_PRINT("All resources released successfully!\n");
    return 0;
  }
  ```
