# aclnnDenseLightningIndexerSoftmaxLse

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T08:41:21.871Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
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

- Description: The DenseLightningIndexerSoftmaxLse operator is a branch operator of the DenseLightningIndexerGradKlLoss operator for computing the Softmax input.

- Formula:

  $$
  \text{res}=\text{AttentionMask}\left(\text{ReduceSum}\left(W\odot\text{ReLU}\left(Q_{index}@K_{index}^T\right)\right)\right)
  $$

  $$
  \text{maxIndex}=\text{max}\left(res\right)
  $$

  $$
  \text{sumIndex}=\text{ReduceSum}\left(\text{exp}\left(res-maxIndex\right)\right)
  $$

  maxIndex and sumIndex are passed as outputs to the DenseLightningIndexerGradKlLoss operator as inputs for Softmax computation.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnDenseLightningIndexerSoftmaxLseGetWorkspaceSize" API to obtain the input parameters and calculate the required workspace size based on the computation flow, and then call the "aclnnDenseLightningIndexerSoftmaxLse" API to execute the computation.

```c++
aclnnStatus aclnnDenseLightningIndexerSoftmaxLseGetWorkspaceSize(
    const aclTensor   *queryIndex,
    const aclTensor   *keyIndex,
    const aclTensor   *weight,
    const aclIntArray *actualSeqLengthsQueryOptional,
    const aclIntArray *actualSeqLengthsKeyOptional,
    char              *layoutOptional,
    int64_t            sparseMode,
    int64_t            preTokens,
    int64_t            nextTokens,
    const aclTensor   *softmaxMaxOut,
    const aclTensor   *softmaxSumOut,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor);
```

```c++
aclnnStatus aclnnDenseLightningIndexerSoftmaxLse(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream);
```

## aclnnDenseLightningIndexerSoftmaxLse

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
      <th>Non-contiguous Tensor</th>
    </tr></thead>
    <tbody>
     <tr>
      <td>queryIndex (aclTensor*)</td>
      <td>Input</td>
      <td>Input queryIndex of the lightningIndexer structure.</td>
      <td><ul><li>B: supports generalization and keeps consistent with B of query.</li><li>Nidx1: 64, 32, 16, 8.</li><li>D: 128.</li><li>T1: sum of S1 accumulated over multiple batches.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>(B,S1,Nidx1,D);(T1,Nidx1,D)</td>
      <td>×</td>
     </tr>
     <tr>
      <td>keyIndex (aclTensor*)</td>
      <td>Input</td>
      <td>Input keyIndex of the lightningIndexer structure.</td>
      <td><ul><li>B: Support generalization and keep consistent with the B of queryIndex.</li> <li>S2: Support generalization.</li><li>Nidx2: 1.</li><li>D: 128.</li><li>T2: Accumulation of S2 across multiple batches.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>(B,S2,Nidx2,D);(T2,Nidx2,D)</td>
      <td>×</td>
     </tr>
     <tr>
      <td>weight (aclTensor*)</td>
      <td>Input</td>
      <td>Weight.</td>
      <td><ul><li>B: Support generalization and keep consistent with the B of queryIndex.</li><li>S1: Support generalization and keep consistent with the S1 of queryIndex.</li><li>Nidx1: 64, 32, 16, 8.</li><li>T1: Accumulation of S1 across multiple batches.</li></ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>(B,S1,Nidx1);(T1,Nidx1)</td>
      <td>×</td>
     </tr>
     <tr>
      <td>actualSeqLengthsQueryOptional (aclIntArray*)</td>
      <td>Input</td>
      <td>Number of valid tokens of the Query in each Batch.</td>
      <td><ul><li>Value dependent.</li><li>The length must keep consistent with B.</li><li>In TND format, the last element is the accumulated sum, and the accumulated sum is less than or equal to T1.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>-</td>
     </tr>
     <tr>
      <td>actualSeqLengthsKeyOptional (aclIntArray*)</td>
      <td>Input</td>
      <td>Number of valid tokens in Key for each batch.</td>
      <td><ul><li>Value dependency.</li><li>The length is consistent with B.</li><li>In TND format, the last element is the accumulated sum, and the accumulated sum is less than or equal to T2.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>-</td>
     </tr>
     <tr>
      <td>layoutOptional (char*)</td>
      <td>Input</td>
      <td>layout format.</td>
      <td>Only BSND and TND formats are supported.</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
     </tr>
     <tr>
      <td>sparseMode (int64_t)</td>
      <td>Input</td>
      <td>Sparse mode.</td>
      <td><ul><li>Indicates the sparse mode. For details about different sparse modes, see <a href="#constraints">Constraints</a>.</li><li>Only mode 3 is supported.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
     </tr>
     <tr>
       <td>preTokens (int64_t)</td>
       <td>Input</td>
       <td>Used for sparse computation. Indicates the number of preceding tokens with which Attention needs to compute correlation.</td>
       <td>Same definition as preTokens in Attention. Takes effect when sparseMode = 0 or 4. Only 2^63-1 is supported.</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
      </tr>
     <tr>
       <td>nextTokens (int64_t)</td>
       <td>Input</td>
       <td>Used for sparse computation. Indicates the number of subsequent tokens that Attention needs to compute correlation with.</td>
       <td>Same definition as nextTokens in Attention. Takes effect when sparseMode = 0 or 4. Only 2^63-1 is supported.</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
     </tr>
     <tr>
      <td>softmaxMaxOut(aclTensor*)</td>
      <td>Output</td>
      <td>Max value used in the softmax computation.</td>
      <td><ul><li>B: Supports generalization and keeps consistent with the B of queryIndex.</li><li>Nidx2: Keeps consistent with the Nidx2 of keyIndex.</li><li>S1: Supports generalization and keeps consistent with the S1 of queryIndex.</li><li>T1: Accumulated S1 across multiple batches.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(B,Nidx2,S1);(Nidx2,T1)</td>
      <td>×</td>
     </tr>
     <tr>
      <td>softmaxSumOut(aclTensor*)</td>
      <td>Output</td>
      <td>Sum value used in softmax computation.</td>
      <td><ul><li>B: Supports generalization and keeps consistent with the B of query.</li><li>Nidx2: Keeps consistent with the Nidx2 of keyIndex.</li><li>S1: Supports generalization and keeps consistent with the S1 of queryIndex.</li><li>T1: Sum of S1 accumulated across multiple batches.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(B,Nidx2,S1);(Nidx2,T1)</td>
      <td>×</td>
     </tr>
     <tr>
      <td>workspaceSize(uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     </tbody>
    </table>

- **Return value**

    Returns an aclnnStatus status code. For details, see [aclnnReturn Codes](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API performs input parameter validation and returns an error in the following scenarios:

    <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
    <col style="width: 319px">
    <col style="width: 144px">
    <col style="width: 671px">
    </colgroup>
    <thead>
     <th>Return Value</th>
     <th>Error Code</th>
     <th>Description</th>
    </thead>
    <tbody>
     <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>A required parameter or output is a null pointer.</td>
      </tr>
     <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data types and data formats of input variables such as queryIndex, keyIndex, and weight are out of the supported range.</td>
     </tr>
     <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>The shape information of multiple input tensors does not match (for details, see Parameters).</td>
     </tr>
     </tbody>
    </table>

## aclnnDenseLightningIndexerSoftmaxLse

- **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
    <col style="width: 184px">
    <col style="width: 134px">
    <col style="width: 833px">
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
      <td>Workspace memory address allocated on the device.</td>
     </tr>
     <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the device, obtained by calling the first-phase API aclnnDenseLightningIndexerSoftmaxLseGetWorkspaceSize.</td>
     </tr>
     <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, containing the operator computation flow.</td>
     </tr>
     <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for executing tasks.</td>
     </tr>
    </tbody>
    </table>

- **Return value**

    Returns an aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

  - The data types of queryIndex and keyIndex must be consistent.

  - If weight is not float32, the data types of queryIndex, keyIndex, and weight should keep consistent.

  - Deterministic computation:
    aclnnDenseLightningIndexerSoftmaxLse defaults to a deterministic implementation.

  - General Constraints
    - Handling of empty input parameters:
        - queryIndex is an empty tensor: Returns directly.
        - The handling of empty input parameters in SFAG general constraints is consistent with FAG.

    <table style="undefined;table-layout: fixed; width: 901px"><colgroup>
    <col style="width: 168px">
    <col style="width: 565px">
    <col style="width: 168px">
    </colgroup>
    <thead>
     <tr>
      <th>sparseMode</th>
      <th>Meaning</th>
      <th>Remarks</th>
     </tr>
    </thead>
    <tbody>
     <tr>
      <td>0</td>
      <td>defaultMask mode. If attenmask is not passed in, no mask operation is performed and preTokens and nextTokens are ignored; if attenmask is passed in, the complete attenmask matrix need to pass in, indicating that the part between preTokens and nextTokens needs to be computed.</td>
      <td>Not supported</td>
     </tr>
     <tr>
      <td>1</td>
      <td>allMask. The complete attenmask matrix must be passed in.</td>
      <td>Not supported</td>
     </tr>
     <tr>
      <td>2</td>
      <td>leftUpCausal mode mask. The optimized attenmask matrix need to pass in.</td>
      <td>Not supported</td>
     </tr>
     <tr>
      <td>3</td>
      <td>Mask in rightDownCausal mode, corresponding to the lower-triangle use case divided by the right vertex. Need to pass in the optimized attenmask matrix.</td>
      <td>Supported</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Mask in band mode. Need to pass in the optimized attenmask matrix.</td>
      <td>Not supported</td>
    </tr>
    <tr>
      <td>5</td>
      <td>prefix</td>
      <td>Not supported</td>
    </tr>
    <tr>
      <td>6</td>
      <td>global</td>
      <td>Not supported</td>
    </tr>
    <tr>
      <td>7</td>
      <td>dilated</td>
      <td>Not supported</td>
    </tr>
    <tr>
      <td>8</td>
      <td>block_local</td>
      <td>Not supported</td>
    </tr>
    </tbody>
    </table>

  - Specification constraints

    <table style="undefined;table-layout: fixed; width: 909px"><colgroup>
    <col style="width: 125px">
    <col style="width: 182px">
    <col style="width: 602px">
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
      <td>1-256</td>
      <td>-</td>
    </tr>
    <tr>
      <td>S1, S2</td>
      <td>0-1M</td>
      <td>S1 and S2 can be of unequal lengths. When layout is BSND, S1<=S2; when layout is TND, actualSeqLengthsQuery is less than or equal to the value of actualSeqLengthsKey at the same index, and S1<=S2 at the same index.</td>
    </tr>
    <tr>
      <td>Nidx1</td>
      <td>8, 16, 32, 64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Nidx2</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>D</td>
      <td>128</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout</td>
      <td>BSND/TND</td>
      <td>-</td>
    </tr>
    </tbody>
    </table>

  - Typical values

    <table style="undefined;table-layout: fixed; width: 903px"><colgroup>
    <col style="width: 164px">
    <col style="width: 739px">
    </colgroup>
    <thead>
    <tr>
      <th>Specification Item</th>
      <th>Typical Value</th>
    </tr>
    </thead>
    <tbody>
    <tr>
      <td>queryIndex</td>
      <td>N1 = 64/32;  D = 128 ; S1 = 64k/128k</td>
    </tr>
    <tr>
      <td>keyIndex</td>
      <td>D = 128.</td>
    </tr>
    </tbody>
    </table>

## Example

The sample code is as follows (for reference only). For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_dense_lightning_indexer_softmax_lse.h"

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
  std::vector<aclFloat16> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, aclFloat16ToFloat(resultData[i]));
  }
}

int Init(int32_t deviceId, aclrtContext* context, aclrtStream* stream) {
  // Initialize AscendCL (fixed boilerplate)
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
  // Call aclrtMalloc to allocate device memory
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host data to device memory
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor API to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize device/context/stream. Refer to the AscendCL external API list.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtContext context;
  aclrtStream stream;
  auto ret = Init(deviceId, &context, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs. Customize the construction based on the API.
  int64_t s1 = 4096;
  int64_t s2 = 4096;
  int64_t n1Index = 8;
  int64_t n2Index = 1;
  int64_t dQueryIndex = 128;
  int64_t t1 = s1;
  int64_t t2 = s2;
  int64_t G = n1Index / n2Index;

  std::vector<int64_t> qIndexShape = {t1, n1Index, dQueryIndex};
  std::vector<int64_t> kIndexShape = {t2, n2Index, dQueryIndex};
  std::vector<int64_t> weightShape = {t1, n1Index};
  std::vector<int64_t> softmaxMaxIndexShape = {n2Index, t1};
  std::vector<int64_t> softmaxSumIndexShape = {n2Index, t1};

  void* qIndexDeviceAddr = nullptr;
  void* kIndexDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* softmaxMaxIndexDeviceAddr = nullptr;
  void* softmaxSumIndexDeviceAddr = nullptr;

  aclTensor* qIndex = nullptr;
  aclTensor* kIndex = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* softmaxMaxIndex = nullptr;
  aclTensor* softmaxSumIndex = nullptr;

  std::vector<aclFloat16> qIndexHostData(t1 * n1Index * dQueryIndex, aclFloatToFloat16(0.2));
  std::vector<aclFloat16> kIndexHostData(t2 * n2Index * dQueryIndex, aclFloatToFloat16(0.1));
  std::vector<aclFloat16> weightHostData(t1 * n1Index, aclFloatToFloat16(0.005));

  std::vector<float> softmaxMaxIndexHostData(t1 * n2Index, 25.4483f);
  std::vector<float> softmaxSumIndexHostData(t1 * n2Index, 1.0f);

  ret = CreateAclTensor(qIndexHostData, qIndexShape, &qIndexDeviceAddr, aclDataType::ACL_FLOAT16, &qIndex);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kIndexHostData, kIndexShape, &kIndexDeviceAddr, aclDataType::ACL_FLOAT16, &kIndex);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxMaxIndexHostData, softmaxMaxIndexShape, &softmaxMaxIndexDeviceAddr,
      aclDataType::ACL_FLOAT, &softmaxMaxIndex);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxSumIndexHostData, softmaxSumIndexShape, &softmaxSumIndexDeviceAddr,
      aclDataType::ACL_FLOAT, &softmaxSumIndex);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t>  acSeqQLenOp = {t1};
  std::vector<int64_t>  acSeqKvLenOp = {t2};
  aclIntArray* acSeqQLen = aclCreateIntArray(acSeqQLenOp.data(), acSeqQLenOp.size());
  aclIntArray* acSeqKvLen = aclCreateIntArray(acSeqKvLenOp.data(), acSeqKvLenOp.size());
  int64_t preTokens = 9223372036854775807;
  int64_t nextTokens = 9223372036854775807;
  int64_t sparseMode = 3;

  char layOut[5] = {'T', 'N', 'D', 0};

  // 3. Call the CANN operator library API. Replace it with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API aclnnDenseLightningIndexerSoftmaxLseGetWorkspaceSize.
  ret = aclnnDenseLightningIndexerSoftmaxLseGetWorkspaceSize(
            qIndex, kIndex, weight, acSeqQLen, acSeqKvLen, layOut,
            sparseMode, preTokens, nextTokens, softmaxMaxIndex, softmaxSumIndex,
            &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnDenseLightningIndexerSoftmaxLseGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);

  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API aclnnDenseLightningIndexerSoftmaxLse.
  ret = aclnnDenseLightningIndexerSoftmaxLse(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDenseLightningIndexerSoftmaxLse failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait synchronously for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value, copy the result from the device memory to the host, and modify it based on the specific API definition.
  PrintOutResult(softmaxMaxIndexShape, &softmaxMaxIndexDeviceAddr);
  PrintOutResult(softmaxSumIndexShape, &softmaxSumIndexDeviceAddr);

  // 6. Release aclTensor and aclScalar, and modify it based on the specific API definition.
  aclDestroyTensor(qIndex);
  aclDestroyTensor(kIndex);
  aclDestroyTensor(weight);
  aclDestroyTensor(softmaxMaxIndex);
  aclDestroyTensor(softmaxSumIndex);

  // 7. Release device resources.
  aclrtFree(qIndexDeviceAddr);
  aclrtFree(kIndexDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(softmaxMaxIndexDeviceAddr);
  aclrtFree(softmaxSumIndexDeviceAddr);

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
