# aclnnLightningIndexerKLLoss

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T11:02:27.754Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Not supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
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

- Description: `lightning_indexer_kl_loss` computes the KL divergence loss between the teacher distribution and the student distribution in the Lightning Indexer.

  - **teacher side** (target_score): unnormalized raw primary attention scores of the compressed segment (sum ≠ 1). `clamp_min` is used to prevent NaN caused by log(0) at y=0.
  - **student side** (index_probs): probability distribution after indexer softmax. `+eps` is used to preserve the gradient at Y→0.

- Formula:

  $$
  y = \text{target\_score}, \quad Y = \text{index\_probs}
  $$

  $$
  P = \frac{y}{\text{sum}(y, \text{dim}=-1, \text{keepdim=True}) + \varepsilon}
  $$

  $$
  \log\_P = \log(\text{clamp\_min}(\tilde{y}, \varepsilon))
  $$

  $$
  \log\_Y = \log(Y + \varepsilon)
  $$

  $$
  \text{loss} = \sum((\log\_P - \log\_Y) \cdot \text{weight})
  $$

  where $\varepsilon$ is the `eps` parameter.

  The weight is selected by `weight_type`:

  - `'logits'`: weight = y, i.e., the raw unnormalized score
  - `'probs'`: weight = P, i.e., the normalized probability

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call the "aclnnLightningIndexerKLLossGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call the "aclnnLightningIndexerKLLoss" API to perform the computation.

```c++
aclnnStatus aclnnLightningIndexerKLLossGetWorkspaceSize(
  const aclTensor   *targetScore,
  const aclTensor   *indexProbs,
  double             eps,
  const char        *weightType,
  const aclTensor   *loss,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnLightningIndexerKLLoss(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  aclrtStream       stream)
```

## aclnnLightningIndexerKLLossGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; max-width: 1550px">
  <colgroup>
    <col style="width: 146px">
    <col style="width: 135px">
    <col style="width: 326px">
    <col style="width: 246px">
    <col style="width: 275px">
    <col style="width: 101px">
    <col style="width: 190px">
    <col style="width: 146px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>targetScore</td>
      <td>Input</td>
      <td>Raw primary attention scores of the teacher without normalization.</td>
      <td>Supported shapes: (T, K) or (B, S, K).</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>indexProbs</td>
      <td>Input</td>
      <td>Probability distribution of the student after softmax.</td>
      <td>Same shape as targetScore.</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>Attribute</td>
      <td>Constant for numerical stability.</td>
      <td>Prevents NaN caused by log(0).</td>
      <td>double</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>weightType</td>
      <td>Attribute</td>
      <td>Outer weight selection: 'logits' or 'probs'.</td>
      <td>'logits' uses the raw target_score as the outer weight, and 'probs' uses the normalized probability P = target_score / sum(target_score) as the outer weight.</td>
      <td>string</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>loss</td>
      <td>Output</td>
      <td>Loss function value, a scalar.</td>
      <td>Data type is the same as that of targetScore. Shape is (1,).</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to be side-applied on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor containing the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## aclnnLightningIndexerKLLoss

- **Parameters**

  <table style="undefined;table-layout: fixed; max-width: 1100px"><colgroup>
  <col style="width: 200px">
  <col style="width: 130px">
  <col style="width: 770px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>The Device side-applied workspace memory address.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>The Device side-applied workspace size, obtained by the first-phase API aclnnLightningIndexerKLLossGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which contains the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnLightningIndexerKLLoss defaults to a non-deterministic implementation. Deterministic computation can be enabled by calling aclrtSetSysParamOpt.
- Input shape constraints:
  - Supports shapes (B, S, K) or (T, K). The value range of B is 1\~512, and the value range of the last dimension K is 1\~8192.
- `eps` must be greater than 0.
- `weight_type` must be `'logits'` or `'probs'`.

## Example

The following sample code shows how to call the operator through the aclnn API, for reference only. For details about how to compile and run the sample, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_lightning_indexer_kl_loss.h"

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

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtContext* context, aclrtStream* stream) {
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateContext(context, deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetCurrentContext(*context);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. Initialize the device/context/stream
  int32_t deviceId = 0;
  aclrtContext context;
  aclrtStream stream;
  auto ret = Init(deviceId, &context, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output
  // target_score: (B, S, K) = (2, 4, 128), float32, non-normalized raw attention scores
  // index_probs:  (B, S, K) = (2, 4, 128), float32, probability distribution after softmax
  // loss:         scalar (1,), float32
  int64_t b = 2;
  int64_t s = 4;
  int64_t k = 128;
  std::vector<int64_t> inputShape = {b, s, k};
  std::vector<int64_t> lossShape = {1};

  std::vector<float> targetScoreHostData(b * s * k, 0);
  for (int64_t i = 0; i < b * s * k; i++) {
    targetScoreHostData[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f;
  }

  std::vector<float> indexProbsHostData(b * s * k, 0);
  for (int64_t bs = 0; bs < b * s; bs++) {
    float sum = 0.0f;
    for (int64_t j = 0; j < k; j++) {
      indexProbsHostData[bs * k + j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
      sum += indexProbsHostData[bs * k + j];
    }
    for (int64_t j = 0; j < k; j++) {
      indexProbsHostData[bs * k + j] /= sum;
    }
  }

  std::vector<float> lossHostData(1, 0.0f);

  void* targetScoreDeviceAddr = nullptr;
  void* indexProbsDeviceAddr = nullptr;
  void* lossDeviceAddr = nullptr;

  aclTensor* targetScore = nullptr;
  aclTensor* indexProbs = nullptr;
  aclTensor* loss = nullptr;

  ret = CreateAclTensor(targetScoreHostData, inputShape, &targetScoreDeviceAddr,
                        aclDataType::ACL_FLOAT, &targetScore);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(indexProbsHostData, inputShape, &indexProbsDeviceAddr,
                        aclDataType::ACL_FLOAT, &indexProbs);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(lossHostData, lossShape, &lossDeviceAddr,
                        aclDataType::ACL_FLOAT, &loss);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // eps parameter
  double eps = 1e-9;

  // 3. Call the C(A)NN operator library API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API aclnnLightningIndexerKLLossGetWorkspaceSize
  ret = aclnnLightningIndexerKLLossGetWorkspaceSize(
      targetScore, indexProbs, eps, "logits", loss, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnLightningIndexerKLLossGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);

  // Allocate device memory based on workspaceSize
  void* workspaceAddr = nullptr;
  if (workspaceSize > static_cast<uint64_t>(0)) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API aclnnLightningIndexerKLLoss
  ret = aclnnLightningIndexerKLLoss(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLightningIndexerKLLoss failed. ERROR: %d\n", ret); return ret);

  // 4. Synchronize and wait for the task to complete
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value
  PrintOutResult(lossShape, &lossDeviceAddr);

  // 6. Release the aclTensor
  aclDestroyTensor(targetScore);
  aclDestroyTensor(indexProbs);
  aclDestroyTensor(loss);

  // 7. Release the device resources
  aclrtFree(targetScoreDeviceAddr);
  aclrtFree(indexProbsDeviceAddr);
  aclrtFree(lossDeviceAddr);
  if (workspaceSize > static_cast<uint64_t>(0)) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtDestroyContext(context);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
