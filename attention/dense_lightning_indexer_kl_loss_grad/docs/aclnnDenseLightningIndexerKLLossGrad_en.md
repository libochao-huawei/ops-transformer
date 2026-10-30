# aclnnDenseLightningIndexerKLLossGrad

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T08:31:50.254Z -->

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

- Description: The DenseLightningIndexerKLLossGrad operator is the backward computation operator of the LightningIndexer KL Loss (Dense mode). This API takes q, k, and w from the Lightning Indexer branch, as well as attnSoftmaxL1Norm and softmaxLse precomputed by the main Attention branch, and computes and outputs dq, dk, dw, and softmaxOut. This API performs dense attention computation over all keys and no longer relies on sparseIndices for top-k filtering; it does not need to recompute the `Q @ K -> softmax -> reduce/l1 norm` of the main Attention inside the kernel, and no longer outputs loss. The target distribution of the main Attention is provided by the attnSoftmaxL1Norm input, and softmaxOut can be used for subsequent loss computation.

- Formula:
    Indexer logits can be expressed as:

   $$
   S_{t,:}=q_{t,:}@K^{T}
   $$

   $$
   I_{t,:}=W_{t,:}@\mathrm{ReLU}(S_{t,:})
   $$

   where $q$ and $K$ correspond to q and k of this API respectively, and $W$ corresponds to w of this API. The softmax output of the Indexer branch is:

   $$
   y_{t,:}=\operatorname{Softmax}(I_{t,:})
   $$

   This API writes $y$ to softmaxOut. The target distribution $p$ is provided by the attnSoftmaxL1Norm input, which is equivalent to the result obtained inside the legacy kernel by summing the main attention score over heads and applying L1 normalization. If KL Loss is subsequently computed, its form remains the same as the legacy version:

   $$
   L(I){=}\sum_tD_{KL}(p_{t,:}||\operatorname{Softmax}(I_{t,:}))
   $$

   $$
   D_{KL}(a||b){=}\sum_ia_i\mathrm{log}{\left(\frac{a_i}{b_i}\right)}
   $$

   The gradient expression of the loss can be obtained by derivation:

   $$
   dI_{t,:}=\operatorname{Softmax}(I_{t,:})-p_{t,:}
   $$

   The gradients of the w, q, and k matrices can be computed using the chain rule:

   $$
   dW_{t,:}=dI_{t,:}\text{@}\left(\mathrm{ReLU}(S_{t,:})\right)^{T}
   $$

   $$
   dq_{t,:}=dS_{t,:}@K
   $$

   $$
   dK=\sum_t\left(dS_{t,:}\right)^{T}@q_{t,:}
   $$

   dK is accumulated directly to the corresponding key position, without scatter-add.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnDenseLightningIndexerKLLossGradGetWorkspaceSize" API to obtain the input parameters and calculate the required workspace size based on the computation process, and then call the "aclnnDenseLightningIndexerKLLossGrad" API to execute the computation.

```c++
aclnnStatus aclnnDenseLightningIndexerKLLossGradGetWorkspaceSize(
    const aclTensor     *q,
    const aclTensor     *k,
    const aclTensor     *w,
    const aclTensor     *attnSoftmaxL1Norm,
    const aclTensor     *softmaxLse,
    const aclTensor     *cuSeqLensQOptional,
    const aclTensor     *cuSeqLensKOptional,
    const aclTensor     *seqUsedQOptional,
    const aclTensor     *seqUsedKOptional,
    const aclTensor     *cmpResidualKOptional,
    const aclTensor     *metadataOptional,
    char                *layoutQ,
    char                *layoutK,
    int64_t              maskMode,
    int64_t              cmpRatio,
    const aclTensor     *dq,
    const aclTensor     *dk,
    const aclTensor     *dw,
    const aclTensor     *softmaxOut,
    uint64_t            *workspaceSize,
    aclOpExecutor       **executor)
```

```c++
aclnnStatus aclnnDenseLightningIndexerKLLossGrad(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    aclrtStream stream)
```

## aclnnDenseLightningIndexerKLLossGradGetWorkspaceSize

- **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1550px">
        <colgroup>
            <col style="width: 220px">
            <col style="width: 120px">
            <col style="width: 300px">
            <col style="width: 400px">
            <col style="width: 212px">
            <col style="width: 100px">
            <col style="width: 190px">
            <col style="width: 145px">
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
                <th>Non-Contiguous Tensor</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>q</td>
                <td>Input</td>
                <td>Query input of the Lightning Indexer branch.</td>
                <td>                    <ul>
                        <li>The data type must be the same as that of k, dq, and dk.</li>
                        <li>Empty tensor is not supported.</li>
                    </ul>
                </td>
                <td>FLOAT16, BFLOAT16</td>
                <td>ND</td>
                <td>(B,S1,N1,D), (T1,N1,D)</td>
                <td>√</td>
            </tr>
            <tr>
                <td>k</td>
                <td>Input</td>
                <td>Key input of the Lightning Indexer branch.</td>
                <td>                    <ul>
                        <li>The data type must be the same as q, dq, and dk.</li>
                        <li>Currently, only N2=1 is supported.</li>
                        <li>Empty tensors are not supported.</li>
                    </ul>
                </td>
                <td>FLOAT16, BFLOAT16</td>
                <td>ND</td>
                <td>(B,S2,N2,D), (T2,N2,D)</td>
                <td>√</td>
            </tr>
            <tr>
                <td>w</td>
                <td>Input</td>
                <td>Head weight of Indexer logits.</td>
                <td>Empty tensors are not supported.</td>
                <td>FLOAT32</td>
                <td>ND</td>
                <td>(B,S1,N1), (T1,N1)</td>
                <td>√</td>
            </tr>
            <tr>
                <td>attnSoftmaxL1Norm</td>
                <td>Input</td>
                <td>Target distribution precomputed by the main Attention branch.</td>
                <td>                    <ul>
                        <li>Used to replace the main Attention recomputation result inside the legacy kernel.</li>
                        <li>Dimensions are (B,S1,N2,S2) or (T1,N2,maxSeqLenK), covering all keys.</li>
                        <li>Empty tensors are not supported.</li>
                    </ul>
                </td>
                <td>FLOAT32</td>
                <td>ND</td>
                <td>(B,S1,N2,S2), (T1,N2,maxSeqLenK)</td>
                <td>√</td>
            </tr>
            <tr>
                <td>softmaxLse</td>
                <td>Input</td>
                <td>softmax log-sum-exp value, precomputed by DenseLightningIndexerSoftmaxLse.</td>
                <td>                    <ul>
                        <li>Used for numerically stable computation.</li>
                        <li>The dimension is (B,N2,S1) or (N2,T1).</li>
                        <li>Empty Tensor Not Supported.</li>
                    </ul>
                </td>
                <td>FLOAT32</td>
                <td>ND</td>
                <td>(B,N2,S1), (N2,T1)</td>
                <td>√</td>
            </tr>
            <tr>
                <td>cuSeqLensQOptional</td>
                <td>Optional Input</td>
                <td>Cumulative sequence length of q under the TND layout.</td>
                <td>Required when layoutQ is TND. The length is B+1, and the first element is 0.</td>
                <td>INT32</td>
                <td>ND</td>
                <td>(B+1,)</td>
                <td>√</td>
            </tr>
            <tr>
                <td>cuSeqLensKOptional</td>
                <td>Optional Input</td>
                <td>Cumulative sequence length of k under the TND layout.</td>
                <td>Required when layoutK is TND. The length is B+1, and the first element is 0.</td>
                <td>INT32</td>
                <td>ND</td>
                <td>(B+1,)</td>
                <td>√</td>
            </tr>
            <tr>
                <td>seqUsedQOptional</td>
                <td>Optional Input</td>
                <td>Represents the actual q length used in each batch.</td>
                <td>The length is B.</td>
                <td>INT32</td>
                <td>ND</td>
                <td>(B,)</td>
                <td>√</td>
            </tr>
            <tr>
                <td>seqUsedKOptional</td>
                <td>Optional Input</td>
                <td>Indicates the length of k actually used in each batch.</td>
                <td>The length is B.</td>
                <td>INT32</td>
                <td>ND</td>
                <td>(B,)</td>
                <td>√</td>
            </tr>
            <tr>
                <td>cmpResidualKOptional</td>
                <td>Optional Input</td>
                <td>Must be provided when maskMode=3 and cmpRatio!=1.</td>
                <td>Length: B.</td>
                <td>INT32</td>
                <td>ND</td>
                <td>(B,)</td>
                <td>√</td>
            </tr>
            <tr>
                <td>metadataOptional</td>
                <td>Input</td>
                <td>Core splitting information generated by aclnnDenseLightningIndexerKLLossGradMetadata.</td>
                <td>Mandatory. If passed in, the kernel uses the core splitting defined in metadata.</td>
                <td>INT32</td>
                <td>ND</td>
                <td>(x,)</td>
                <td>√</td>
            </tr>
            <tr>
                <td>layoutQ</td>
                <td>Input</td>
                <td>Layout of q, w, attnSoftmaxL1Norm, softmaxLse, dq, dw, and softmaxOut.</td>
                <td>Only BSND and TND are supported.</td>
                <td>STRING</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
            </tr>
            <tr>
                <td>layoutK</td>
                <td>Input</td>
                <td>Layout of k and dk.</td>
                <td>Only BSND and TND are supported.</td>
                <td>STRING</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
            </tr>
            <tr>
                <td>maskMode</td>
                <td>Input</td>
                <td>Mask mode.</td>
                <td>For details about supported mask modes, see <a href="#constraints">Constraints</a>.</td>
                <td>INT64</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
            </tr>
            <tr>
                <td>cmpRatio</td>
                <td>Input</td>
                <td>Compression ratio.</td>
                <td>Value range: 1-128.</td>
                <td>INT64</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
            </tr>
            <tr>
                <td>dq</td>
                <td>Output</td>
                <td>Gradient of q.</td>
                <td>Data type and shape are the same as q.</td>
                <td>FLOAT16, BFLOAT16</td>
                <td>ND</td>
                <td>(B,S1,N1,D), (T1,N1,D)</td>
                <td>x</td>
            </tr>
            <tr>
                <td>dk</td>
                <td>Output</td>
                <td>Gradient of k.</td>
                <td>Data type and shape are the same as k.</td>
                <td>FLOAT16, BFLOAT16</td>
                <td>ND</td>
                <td>(B,S2,N2,D), (T2,N2,D)</td>
                <td>x</td>
            </tr>
            <tr>
                <td>dw</td>
                <td>Output</td>
                <td>Gradient of w.</td>
                <td>Data type and shape are the same as w.</td>
                <td>FLOAT32</td>
                <td>ND</td>
                <td>(B,S1,N1), (T1,N1)</td>
                <td>x</td>
            </tr>
            <tr>
                <td>softmaxOut</td>
                <td>Output</td>
                <td>Softmax output of the Indexer branch.</td>
                <td>The shape is the same as that of attnSoftmaxL1Norm.</td>
                <td>FLOAT32</td>
                <td>ND</td>
                <td>(B,S1,N2,S2), (T1,N2,S2)</td>
                <td>x</td>
            </tr>
            <tr>
                <td>workspaceSize</td>
                <td>Output</td>
                <td>Returns the size of the workspace that the user needs to allocate on the Device.</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
            </tr>
            <tr>
                <td>executor</td>
                <td>Output</td>
                <td>Op executor, including the computation process of the operator.</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
            </tr>
        </tbody>
    </table>

- **Return value**

    Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API performs input parameter validation and returns an error in the following scenarios:

    <table style="undefined;table-layout: fixed;width: 1155px">
        <colgroup>
            <col style="width: 319px">
            <col style="width: 144px">
            <col style="width: 671px">
        </colgroup>
        <thead>
            <tr>
                <th>Return Value</th>
                <th>Error Code</th>
                <th>Description</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>ACLNN_ERR_PARAM_NULLPTR</td>
                <td>161001</td>
                <td>An invalid nullptr exists in the parameters.</td>
            </tr>
            <tr>
                <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
                <td rowspan="3">161002</td>
                <td>The input data type is not a supported type.</td>
            </tr>
            <tr>
                <td>Required inputs q, k, w, attnSoftmaxL1Norm, or softmaxLse are not passed.</td>
            </tr>
            <tr>
                <td>Required outputs dq, dk, dw, or softmaxOut are not passed.</td>
            </tr>
        </tbody>
    </table>

## aclnnDenseLightningIndexerKLLossGrad

- **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1155px">
        <colgroup>
            <col style="width: 144px">
            <col style="width: 125px">
            <col style="width: 700px">
        </colgroup>
        <thead>
            <tr>
                <th>Parameter</th>
                <th>Input/Output</th>
                <th>Description</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>workspace</td>
                <td>Input</td>
                <td>Workspace memory address allocated on the Device side.</td>
            </tr>
            <tr>
                <td>workspaceSize</td>
                <td>Input</td>
                <td>Workspace size allocated on the Device side, obtained from the first-phase API aclnnDenseLightningIndexerKLLossGradGetWorkspaceSize.</td>
            </tr>
            <tr>
                <td>executor</td>
                <td>Input</td>
                <td>Op executor, which contains the operator computation flow.</td>
            </tr>
            <tr>
                <td>stream</td>
                <td>Input</td>
                <td>Stream for executing tasks.</td>
            </tr>
        </tbody>
    </table>

- **Return value**

    Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:

    <!-- npu="950" id7 -->
    - <term>Ascend 950PR&950DT products</term>: aclnnDenseLightningIndexerKLLossGrad defaults to a non-deterministic implementation. Deterministic computation can be enabled through aclrtSetSysParamOpt.

    <!-- end id7 -->

- Common constraints:
  - The data types of q, k, dq, and dk must be the same. FLOAT16 and BFLOAT16 are supported.
  - The data types of w, dw, attnSoftmaxL1Norm, softmaxLse, and softmaxOut must be FLOAT32.
  - The data types of cuSeqLensQOptional, cuSeqLensKOptional, seqUsedQOptional, seqUsedKOptional, cmpResidualKOptional, and metadataOptional must be INT32.
  - BSND and TND are currently supported for layoutQ and layoutK.
  - When layoutQ is TND, pass in cuSeqLensQOptional; when layoutK is TND, pass in cuSeqLensKOptional.
  - The dimensions of attnSoftmaxL1Norm are (B,S1,N2,S2) or (T1,N2,maxSeqLenK), covering the target distribution over all keys.
  - softmaxLse is precomputed and provided by the DenseLightningIndexerSoftmaxLse operator, with dimensions of (B,N2,S1) or (N2,T1), for numerically stable computation.
  - The metadata input must be passed in to compute core splitting information.

    <table style="undefined;table-layout: fixed; width: 942px">
        <colgroup>
            <col style="width: 100px">
            <col style="width: 740px">
            <col style="width: 360px">
        </colgroup>
        <thead>
            <tr>
                <th>maskMode</th>
                <th>Meaning</th>
                <th>Remarks</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>0</td>
                <td>defaultMask mode, no causal mask is applied.</td>
                <td>Supported</td>
            </tr>
            <tr>
                <td>1</td>
                <td>allMask, the complete attenmask matrix must be passed in.</td>
                <td>Not supported</td>
            </tr>
            <tr>
                <td>2</td>
                <td>Mask in leftUpCausal mode.</td>
                <td>Not supported</td>
            </tr>
            <tr>
                <td>3</td>
                <td>Mask of the rightDownCausal mode, corresponding to the lower triangle scenario with the right vertex as the dividing point.</td>
                <td>Supported</td>
            </tr>
            <tr>
                <td>4</td>
                <td>Mask of the band mode.</td>
                <td>Not supported</td>
            </tr>
            <tr>
                <td>5</td>
                <td>prefix mode.</td>
                <td>Not supported</td>
            </tr>
            <tr>
                <td>6</td>
                <td>global mode.</td>
                <td>Not supported</td>
            </tr>
            <tr>
                <td>7</td>
                <td>dilated mode.</td>
                <td>Not supported</td>
            </tr>
            <tr>
                <td>8</td>
                <td>block_local mode.</td>
                <td>Not supported</td>
            </tr>
        </tbody>
    </table>

- Specification constraints:
    <table style="undefined;table-layout: fixed; width: 942px">
        <colgroup>
            <col style="width: 100px">
            <col style="width: 300px">
            <col style="width: 360px">
        </colgroup>
        <thead>
            <tr>
                <th>Specification Item</th>
                <th>Value</th>
                <th>Description</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>B</td>
                <td>-</td>
                <td>-</td>
            </tr>
            <tr>
                <td>S1, S2</td>
                <td>-</td>
                <td>S1 and S2 support unequal lengths.</td>
            </tr>
            <tr>
                <td>N1</td>
                <td>1-128</td>
                <td>MQA use case</td>
            </tr>
            <tr>
                <td>N2</td>
                <td>1</td>
                <td>Currently, only N2=1 is supported.</td>
            </tr>
            <tr>
                <td>D</td>
                <td>128</td>
                <td>-</td>
            </tr>
            <tr>
                <td>cmpRatio</td>
                <td>1-128</td>
                <td>-</td>
            </tr>
        </tbody>
    </table>

## Example

The following sample code is for reference only. For details about how to compile and run the code, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_dense_lightning_indexer_kl_loss_grad.h"
#include "aclnnop/aclnn_dense_lightning_indexer_kl_loss_grad_metadata.h"

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

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

void PrintOutResult(std::vector<int64_t>& shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < std::min<int64_t>(size, 5); i++) {
    LOG_PRINT("softmaxOut[%ld] is: %f\n", i, resultData[i]);
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

int main() {
  int32_t deviceId = 0;
  aclrtContext context;
  aclrtStream stream;
  auto ret = Init(deviceId, &context, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  const int64_t batchSize = 1;
  const int64_t t1 = 16;
  const int64_t t2 = 4;
  const int64_t numHeadsQ = 8;
  const int64_t numHeadsK = 1;
  const int64_t headDim = 128;
  const int64_t maskMode = 3;
  const int64_t cmpRatio = 1;

  std::vector<int64_t> qShape = {t1, numHeadsQ, headDim};
  std::vector<int64_t> kShape = {t2, numHeadsK, headDim};
  std::vector<int64_t> wShape = {t1, numHeadsQ};
  std::vector<int64_t> l1NormShape = {t1, numHeadsK, t2};
  std::vector<int64_t> softmaxLseShape = {numHeadsK, t1};
  std::vector<int64_t> softmaxShape = {t1, numHeadsK, t2};
  std::vector<int64_t> cuSeqLensShape = {batchSize + 1};
  std::vector<int64_t> metadataShape = {64};

  std::vector<uint16_t> qHostData(GetShapeSize(qShape), 0x3C00);
  std::vector<uint16_t> kHostData(GetShapeSize(kShape), 0x3C00);
  std::vector<float> wHostData(GetShapeSize(wShape), 1.0f);
  std::vector<float> attnSoftmaxL1NormHostData(GetShapeSize(l1NormShape), 0.0f);
  for (int64_t t = 0; t < t1; ++t) {
    auto validK = std::min<int64_t>(std::max<int64_t>(1, (t + 1) / cmpRatio), t2);
    for (int64_t kIdx = 0; kIdx < validK; ++kIdx) {
      attnSoftmaxL1NormHostData[t * numHeadsK * t2 + kIdx] = 1.0f / static_cast<float>(validK);
    }
  }
  std::vector<float> softmaxLseHostData(GetShapeSize(softmaxLseShape), 0.0f);
  std::vector<int32_t> cuSeqLensQHostData = {0, static_cast<int32_t>(t1)};
  std::vector<int32_t> cuSeqLensKHostData = {0, static_cast<int32_t>(t2)};
  std::vector<int32_t> metadataHostData(GetShapeSize(metadataShape), 0);
  std::vector<uint16_t> dqHostData(GetShapeSize(qShape), 0);
  std::vector<uint16_t> dkHostData(GetShapeSize(kShape), 0);
  std::vector<float> dwHostData(GetShapeSize(wShape), 0.0f);
  std::vector<float> softmaxOutHostData(GetShapeSize(softmaxShape), 0.0f);

  void* qDeviceAddr = nullptr;
  void* kDeviceAddr = nullptr;
  void* wDeviceAddr = nullptr;
  void* attnSoftmaxL1NormDeviceAddr = nullptr;
  void* softmaxLseDeviceAddr = nullptr;
  void* cuSeqLensQDeviceAddr = nullptr;
  void* cuSeqLensKDeviceAddr = nullptr;
  void* metadataDeviceAddr = nullptr;
  void* dqDeviceAddr = nullptr;
  void* dkDeviceAddr = nullptr;
  void* dwDeviceAddr = nullptr;
  void* softmaxOutDeviceAddr = nullptr;

  aclTensor* q = nullptr;
  aclTensor* k = nullptr;
  aclTensor* w = nullptr;
  aclTensor* attnSoftmaxL1Norm = nullptr;
  aclTensor* softmaxLse = nullptr;
  aclTensor* cuSeqLensQ = nullptr;
  aclTensor* cuSeqLensK = nullptr;
  aclTensor* metadata = nullptr;
  aclTensor* dq = nullptr;
  aclTensor* dk = nullptr;
  aclTensor* dw = nullptr;
  aclTensor* softmaxOut = nullptr;

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(wHostData, wShape, &wDeviceAddr, aclDataType::ACL_FLOAT, &w);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attnSoftmaxL1NormHostData, l1NormShape, &attnSoftmaxL1NormDeviceAddr,
                        aclDataType::ACL_FLOAT, &attnSoftmaxL1Norm);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxLseHostData, softmaxLseShape, &softmaxLseDeviceAddr,
                        aclDataType::ACL_FLOAT, &softmaxLse);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cuSeqLensQHostData, cuSeqLensShape, &cuSeqLensQDeviceAddr, aclDataType::ACL_INT32, &cuSeqLensQ);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cuSeqLensKHostData, cuSeqLensShape, &cuSeqLensKDeviceAddr, aclDataType::ACL_INT32, &cuSeqLensK);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(metadataHostData, metadataShape, &metadataDeviceAddr, aclDataType::ACL_INT32, &metadata);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dqHostData, qShape, &dqDeviceAddr, aclDataType::ACL_FLOAT16, &dq);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dkHostData, kShape, &dkDeviceAddr, aclDataType::ACL_FLOAT16, &dk);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dwHostData, wShape, &dwDeviceAddr, aclDataType::ACL_FLOAT, &dw);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxOutHostData, softmaxShape, &softmaxOutDeviceAddr, aclDataType::ACL_FLOAT, &softmaxOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  char layoutQ[4] = {'T', 'N', 'D', 0};
  char layoutK[4] = {'T', 'N', 'D', 0};

  uint64_t metadataWorkspaceSize = 0;
  aclOpExecutor* metadataExecutor = nullptr;
  ret = aclnnDenseLightningIndexerKLLossGradMetadataGetWorkspaceSize(
      cuSeqLensQ, cuSeqLensK, nullptr, nullptr, nullptr, batchSize, t1, t2, numHeadsQ, numHeadsK, headDim,
      layoutQ, layoutK, maskMode, cmpRatio, metadata, &metadataWorkspaceSize, &metadataExecutor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnDenseLightningIndexerKLLossGradMetadataGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);

  void* metadataWorkspaceAddr = nullptr;
  if (metadataWorkspaceSize > 0) {
    ret = aclrtMalloc(&metadataWorkspaceAddr, metadataWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate metadata workspace failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnDenseLightningIndexerKLLossGradMetadata(metadataWorkspaceAddr, metadataWorkspaceSize,
                                                      metadataExecutor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnDenseLightningIndexerKLLossGradMetadata failed. ERROR: %d\n", ret); return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  ret = aclnnDenseLightningIndexerKLLossGradGetWorkspaceSize(
      q, k, w, attnSoftmaxL1Norm, softmaxLse, cuSeqLensQ, cuSeqLensK, nullptr, nullptr, nullptr, metadata,
      layoutQ, layoutK, maskMode, cmpRatio, dq, dk, dw, softmaxOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnDenseLightningIndexerKLLossGradGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnDenseLightningIndexerKLLossGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnDenseLightningIndexerKLLossGrad failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  PrintOutResult(softmaxShape, &softmaxOutDeviceAddr);

  aclDestroyTensor(q);
  aclDestroyTensor(k);
  aclDestroyTensor(w);
  aclDestroyTensor(attnSoftmaxL1Norm);
  aclDestroyTensor(softmaxLse);
  aclDestroyTensor(cuSeqLensQ);
  aclDestroyTensor(cuSeqLensK);
  aclDestroyTensor(metadata);
  aclDestroyTensor(dq);
  aclDestroyTensor(dk);
  aclDestroyTensor(dw);
  aclDestroyTensor(softmaxOut);

  aclrtFree(qDeviceAddr);
  aclrtFree(kDeviceAddr);
  aclrtFree(wDeviceAddr);
  aclrtFree(attnSoftmaxL1NormDeviceAddr);
  aclrtFree(softmaxLseDeviceAddr);
  aclrtFree(cuSeqLensQDeviceAddr);
  aclrtFree(cuSeqLensKDeviceAddr);
  aclrtFree(metadataDeviceAddr);
  aclrtFree(dqDeviceAddr);
  aclrtFree(dkDeviceAddr);
  aclrtFree(dwDeviceAddr);
  aclrtFree(softmaxOutDeviceAddr);
  if (metadataWorkspaceSize > 0) {
    aclrtFree(metadataWorkspaceAddr);
  }
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtDestroyContext(context);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
