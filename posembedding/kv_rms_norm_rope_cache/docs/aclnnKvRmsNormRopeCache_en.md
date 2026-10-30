# aclnnKvRmsNormRopeCache

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T15:50:02.805Z -->

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

- Description: Splits the last axis of the input tensor (kv) into the left half for RMS Norm computation and the right half for RoPE computation, then scatters the results into two caches respectively.

- Formula:

  (1) interleaveRope:

  $$
  x=kv[...,Dv:]
  $$

  $$
  x1=x[...,::2]
  $$

  $$
  x2=x[...,1::2]
  $$

  $$
  x\_part1=torch.cat((x1,x2),dim=-1)
  $$

  $$
  x\_part2=torch.cat((-x2,x1),dim=-1)
  $$

  $$
  y=x\_part1*cos+x\_part2*sin
  $$

  (2) rmsNorm:

  $$
  x=kv[...,:Dv]
  $$

  $$
  square\_x=x*x
  $$

  $$
  mean\_square\_x=square\_x.mean(dim=-1,keepdim=True)
  $$

  $$
  rms=torch.sqrt(mean\_square\_x+epsilon)
  $$

  $$
  y=(x/rms)*gamma
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnKvRmsNormRopeCacheGetWorkspaceSize" API first to obtain the input parameters and calculate the required workspace size based on the process, and then call the "aclnnKvRmsNormRopeCache" API to perform the computation.

```Cpp
aclnnStatus aclnnKvRmsNormRopeCacheGetWorkspaceSize(
  const aclTensor* kv,
  const aclTensor* gamma,
  const aclTensor* cos,
  const aclTensor* sin,
  const aclTensor* index,
  aclTensor*       kCacheRef,
  aclTensor*       ckvCacheRef,
  const aclTensor* kRopeScaleOptional,
  const aclTensor* cKvScaleOptional,
  const aclTensor* kRopeOffsetOptional,
  const aclTensor* cKvOffsetOptional,
  double           epsilon,
  char*            cacheModeOptional,
  bool             isOutputKv,
  const aclTensor* kRopeOut,
  const aclTensor* cKvOut,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnKvRmsNormRopeCache(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnKvRmsNormRopeCacheGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1532px"><colgroup>
  <col style="width: 162px">
  <col style="width: 121px">
  <col style="width: 403px">
  <col style="width: 403px">
  <col style="width: 275px">
  <col style="width: 118px">
  <col style="width: 138px">
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
      <td>kv</td>
      <td>Input</td>
      <td>Input data used in the formula to split out Dv required for rms_norm computation and Dk required for RoPE computation.</td>
      <td><ul><li>Only the 4-dimension shape [Bkv,N,Skv,D] is supported.</li><li>Empty tensors are not supported.</li></ul></td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>Input data for rms_norm computation in the formula.</td>
      <td><ul><li>Consistent with the data type of kv, with the 1D shape [Dv,].</li><li>Empty tensors are not supported.</li></ul></td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cos</td>
      <td>Input</td>
      <td>Input data used for RoPE computation in the formula, performing cosine transformation on the input tensor Dk.</td>
      <td><ul><li>Consistent with the data type of kv, with the 4D shape [Bkv,1,Skv,Dk] or [Bkv,1,1,Dk].</li><li>Empty tensors are not supported.</li></ul></td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sin</td>
      <td>Optional input</td>
      <td>Input data used for RoPE computation in the formula. Performs sine transformation on the Dk dimension of the input tensor.</td>
      <td><ul><li>Data type consistent with kv. The shape is 4-dimensional [Bkv,1,Skv,Dk] or [Bkv,1,1,Dk], consistent with the shape of cos.</li><li>Empty tensors are not supported.</li></ul></td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>index</td>
      <td>Input</td>
      <td>Specifies the index position for writing to the cache. If the value of index is -1, the update is skipped.</td>
      <td><ul><li>If cacheModeOptional is Norm, the shape is 2D [Bkv,Skv].</li><li>If cacheModeOptional is PA_BNSD or PA_NZ, the shape is 1D [Bkv * Skv].</li><li>If cacheModeOptional is PA_BLK_BNSD or PA_BLK_NZ, the shape is 1D [Bkv\*ceil_div(Skv,BlockSize)].</li><li>Empty tensors are not supported.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1-2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kCacheRef</td>
      <td>input/output</td>
      <td>Cache allocated in advance. The input and output reuse the same address.</td>
      <td><ul><li>In non-quantization scenarios, the data type is consistent with that of input kv.</li><li>In quantization scenarios, the data type is INT8, HIFLOAT8, FLOAT8E5M2, or FLOAT8E4M3FN.</li><li>When cacheModeOptional indicates the PA scenario (cacheModeOptional is PA, PA_BNSD, PA_NZ, PA_BLK_BNSD, or PA_BLK_NZ), the shape is 4D [BlockNum,BlockSize,N,Dk].</li><li>When cacheModeOptional indicates the Norm scenario, the shape is 4D [Bcache,N,Scache,Dk].</li><li>Empty tensors are not supported.</li></ul></td>
      <td>FLOAT16, BFLOAT16, INT8, HIFLOAT8, FLOAT8E5M2, FLOAT8E4M3FN</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>ckvCacheRef</td>
      <td>input/output</td>
      <td>Cache allocated in advance. The input and output reuse the same address.</td>
      <td><ul><li>In non-quantization scenarios, the data type is consistent with input kv.</li><li>In quantization scenarios, the data type is INT8, HIFLOAT8, FLOAT8E5M2, or FLOAT8E4M3FN.</li><li>When cacheModeOptional is the PA scenario (cacheModeOptional is PA, PA_BNSD, PA_NZ, PA_BLK_BNSD, or PA_BLK_NZ), the shape is 4-dimensional [BlockNum,BlockSize,N,Dv].</li><li>When cacheModeOptional is the Norm scenario, the shape is 4-dimensional [Bcache,N,Scache,Dv].</li><li>Empty tensors are not supported.</li></ul></td>
      <td>FLOAT16, BFLOAT16, INT8, HIFLOAT8, FLOAT8E5M2, FLOAT8E4M3FN</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kRopeScaleOptional</td>
      <td>Input</td>
      <td>This input parameter is required when the data type of kCacheRef is INT8, HIFLOAT8, FLOAT8E5M2, or FLOAT8E4M3FN.</td>
      <td><ul><li>The shape is 2-dimensional [N,Dk]; or 1-dimensional [Dk,]; or 1-dimensional [1,].</li><li>Empty tensors are not supported.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>ckvScaleOptional</td>
      <td>Input</td>
      <td>This input parameter is required when the data type of ckvCacheRef is INT8, HIFLOAT8, FLOAT8E5M2, or FLOAT8E4M3FN.</td>
      <td><ul><li>shape is 2D [N,Dv]; or 1D [Dv,]; or 1D [1,].</li><li>Empty tensors are not supported.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kRopeOffsetOptional</td>
      <td>Input</td>
      <td>This parameter is required when the data type of kCacheRef is INT8, HIFLOAT8, FLOAT8E5M2, or FLOAT8E4M3FN, and the corresponding kRopeScaleOptional input exists and the quantization scenario is asymmetric quantization.</td>
      <td><ul><li>The shape is 2D [N,Dk]; or the shape is 1D [Dk,]; or the shape is 1D [1,].</li><li>Empty tensors are not supported.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cKvOffsetOptional</td>
      <td>Input</td>
      <td>This parameter is required when the data type of ckvCacheRef is INT8, HIFLOAT8, FLOAT8E5M2, or FLOAT8E4M3FN, the corresponding ckvScaleOptional input exists, and the quantization scenario is asymmetric quantization.</td>
      <td><ul><li>The shape is 2D [N,Dv]; or 1D [Dv,]; or 1D [1,].</li><li>Empty tensors are not supported.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>attribute</td>
      <td>Prevents division by zero in the rms_norm computation.</td>
      <td>The recommended value is 1e-5.</td>
      <td>double</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cacheModeOptional</td>
      <td>attribute</td>
      <td>Flag for selecting the cache format.</td>
      <td>Types: Norm, PA, PA_BNSD, PA_NZ, PA_BLK_BNSD, PA_BLK_NZ. The recommended value is Norm.</td>
      <td>char*</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>isOutputKv</td>
      <td>attribute</td>
      <td>Output control flag of kRopeOut and cKvOut.</td>
      <td>When isOutputKv is true, kRopeOut and cKvOut need to be output. It is recommended to set it to false.</td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>kRopeOut</td>
      <td>Output</td>
      <td>Controlled by isOutputKv, corresponding to `y` in the interleaveRope formula.<br>Must be output when isOutputKv is true. The data type is consistent with input kv.</td>
      <td>The shape is 4D [Bkv,N,Skv,Dk].</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cKvOut</td>
      <td>Output</td>
      <td>Controlled by isOutputKv, corresponding to `y` in the rmsNorm formula.<br>Output when isOutputKv is true. The data type is consistent with input kv.</td>
      <td>Shape is 4D [Bkv,N,Skv,Dv].</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns an op executor, which contains the operator computation process.</td>
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

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1260px"><colgroup>
  <col style="width: 325px">
  <col style="width: 126px">
  <col style="width: 809px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input or output tensor is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The input and output data types are not in the supported range.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="4">561002</td>
      <td>The shape of parameters kv, gamma, cos, sin, index, kCacheRef, ckvCacheRef, kRopeScaleOptional, ckvScaleOptional, kRopeOffsetOptional, or cKvOffsetOptional fails validation.</td>
    </tr>
    <tr>
      <td>The dtype of parameters kv, gamma, cos, sin, index, kCacheRef, ckvCacheRef, kRopeScaleOptional, ckvScaleOptional, kRopeOffsetOptional, or cKvOffsetOptional fails validation.</td>
    </tr>
    <tr>
      <td>In the PA use case (cacheModeOptional is PA, PA_BNSD, PA_NZ, PA_BLK_BNSD, or PA_BLK_NZ), the BlockSize dimension value of cache fails validation.</td>
    </tr>
    <tr>
      <td>In the NZ use case (cacheModeOptional is PA_NZ or PA_BLK_NZ), the dimension values of Dk and Dv fail validation.</td>
    </tr>
  </tbody>
  </table>

## aclnnKvRmsNormRopeCache

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
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
      <td>Workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device, obtained by calling the first-phase API aclnnKvRmsNormRopeCacheGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

  - This operator defaults to a deterministic implementation.
  - Description of the shape format in Parameters:

      - Bkv is the batch size of the input kv, and Skv is the sequence length of the input kv. Their sizes are determined by the user's input scenario, with no explicit limit.
      - N is the head number of the input kv. This operator is strongly related to the DeepSeekV3 network structure and only supports the scenario where N=1; scenarios where N is not 1 do not exist.
      - D is the head dim of input kv. Dv required for the rms_norm computation and Dk required for the RoPE computation are split from D of input kv. Therefore, the sizes of Dk and Dv must satisfy Dk+Dv=D. In addition, Dk must satisfy the rope rule. According to the rope rule, Dk must be an even number. If cacheModeOptional is an NZ use case (cacheModeOptional is PA_NZ or PA_BLK_NZ), Dk and Dv must be 32 Byte aligned.
      - If cacheModeOptional is a PA scenario (cacheModeOptional is PA, PA_BNSD, PA_NZ, PA_BLK_BNSD, or PA_BLK_NZ), BlockSize must be 32B-aligned.
      - For the 32B-alignment cases above, the alignment value is determined by the data type of the cache. Taking BlockSize as an example, if the data type of the cache is INT8, HIFLOAT8, FLOAT8E5M2, or FLOAT8E4M3FN, BlockSize%32=0 is required. If the data type of the cache is float16 or bfloat16, BlockSize%16=0 is required. If the dtype of the kCacheRef and ckvCacheRef parameters is inconsistent, BlockSize must satisfy both BlockSize%32=0 and BlockSize%16=0.
      - Bcache is the batch size of the input cache, and Scache is the sequence length of the input cache. Their sizes are determined by the user input scenario, with no explicit limit.
      - BlockNum is the number of memory blocks written to the cache. Its size is determined by the user input scenario, with no explicit limit.
      - None of the input tensors can be empty.
      - All inputs do not support invalid values, including but not limited to ±inf and nan.

  - Index-related constraints:

      - When cacheModeOptional is Norm, the shape is 2D [Bkv, Skv], and the value range of index must be [-1, Scache). Values can be duplicated across different Bkv.
      - When cacheModeOptional is PA_BNSD or PA_NZ, the shape is 1D [Bkv *Skv]. The values of index must be in the range [-1,BlockNum* BlockSize). Duplicate values are not allowed.
      - When cacheModeOptional is PA_BLK_BNSD or PA_BLK_NZ, the shape is 1D [Bkv *ceil_div(Skv,BlockSize)]. The values of index must be in the range [-1,BlockNum* BlockSize). Duplicate value/BlockSize results are not allowed.

  - Constraints for quantization scenarios:

      - Quantization scenario case 1: The data type of kCacheRef is FLOAT16 or BFLOAT16, and the data type of ckvCacheRef is INT8, HIFLOAT8, FLOAT8E5M2, or FLOAT8E4M3FN.
      - Quantization scenario case 2: The data type of ckvCacheRef is FLOAT16 or BFLOAT16, and the data type of kCacheRef is INT8, HIFLOAT8, FLOAT8E5M2, or FLOAT8E4M3FN.
      - Quantization scenario case 3: The data types of kCacheRef and ckvCacheRef are consistent, being INT8, HIFLOAT8, FLOAT8E5M2, or FLOAT8E4M3FN.

  - Supported data types of cache:

    - Non-quantization mode: The cache type must be consistent with that of kv.
      - <term>Ascend 950PR&950DT products</term>, <term>Atlas A3 products</term>, and <term>Atlas A2 products</term>: BFLOAT16 and FLOAT16 are supported.
      - <term>Kirin X90/Kirin 9030 processors</term>: Only FLOAT16 is supported.
    - Quantization mode:
      - <term>Ascend 950PR&950DT products</term>: INT8, HIFLOAT8, FLOAT8E5M2, and FLOAT8E4M3FN are supported.
      - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>, and <term>Kirin X90/Kirin 9030 processors</term>: Only INT8 is supported.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_kv_rms_norm_rope_cache.h"

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
  std::vector<aclFloat16> resultData(size, aclFloatToFloat16(0.0f));
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, aclFloat16ToFloat(resultData[i]));
  }
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // Boilerplate: resource initialization.
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate device memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy data from the host to device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of a contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. Boilerplate: initialize the device/stream. For details, see the acl API documentation.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output according to the API definition.
  std::vector<int64_t> kvShape = {181,1,1,576};
  std::vector<int64_t> gammaShape = {512,};
  std::vector<int64_t> cosShape = {181,1,1,64};
  std::vector<int64_t> sinShape = {181,1,1,64};
  std::vector<int64_t> indexShape = {181,1};
  std::vector<int64_t> kpeCacheShape = {181,1,1,64};
  std::vector<int64_t> ckvCacheShape = {181,1,1,512};
  std::vector<int64_t> kRopeShape = {181,1,1,64};
  std::vector<int64_t> cKvShape = {181,1,1,512};

  std::vector<int16_t> kvHostData(181*1*1*576,0);
  std::vector<int16_t> gammaHostData(512,0);
  std::vector<int16_t> cosHostData(181*1*1*64,0);
  std::vector<int16_t> sinHostData(181*1*1*64,0);
  std::vector<int64_t> indexHostData(181*1,0);
  std::vector<int16_t> kpeCacheHostData(181*1*1*64,0);
  std::vector<int16_t> ckvCacheHostData(181*1*1*512,0);
  std::vector<int16_t> kRopeHostData(181*1*1*64,0);
  std::vector<int16_t> cKvHostData(181*1*1*512,0);

  void* kvDeviceAddr = nullptr;
  void* gammaDeviceAddr = nullptr;
  void* cosDeviceAddr = nullptr;
  void* sinDeviceAddr = nullptr;
  void* indexDeviceAddr = nullptr;
  void* kpeCacheDeviceAddr = nullptr;
  void* ckvCacheDeviceAddr = nullptr;
  void* kRopeDeviceAddr = nullptr;
  void* cKvDeviceAddr = nullptr;

  aclTensor* kv = nullptr;
  aclTensor* gamma = nullptr;
  aclTensor* cos = nullptr;
  aclTensor* sin = nullptr;
  aclTensor* index = nullptr;
  aclTensor* kpeCache = nullptr;
  aclTensor* ckvCache = nullptr;
  aclTensor* kRope = nullptr;
  aclTensor* cKv = nullptr;


  double epsilon = 1e-5;
  char cacheMode[] = "Norm";
  bool isOutputKv = false;

  ret = CreateAclTensor(kvHostData, kvShape, &kvDeviceAddr, aclDataType::ACL_FLOAT16, &kv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT16, &gamma);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cosHostData, cosShape, &cosDeviceAddr, aclDataType::ACL_FLOAT16, &cos);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sinHostData, sinShape, &sinDeviceAddr, aclDataType::ACL_FLOAT16, &sin);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(indexHostData, indexShape, &indexDeviceAddr, aclDataType::ACL_INT64, &index);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kpeCacheHostData, kpeCacheShape, &kpeCacheDeviceAddr, aclDataType::ACL_FLOAT16, &kpeCache);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(ckvCacheHostData, ckvCacheShape, &ckvCacheDeviceAddr, aclDataType::ACL_FLOAT16, &ckvCache);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kRopeHostData, kRopeShape, &kRopeDeviceAddr, aclDataType::ACL_FLOAT16, &kRope);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cKvHostData, cKvShape, &cKvDeviceAddr, aclDataType::ACL_FLOAT16, &cKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the specific API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnKvRmsNormRopeCache.
  ret = aclnnKvRmsNormRopeCacheGetWorkspaceSize(kv,gamma,cos,sin,index,
                                                kpeCache,ckvCache,nullptr,nullptr,nullptr,nullptr,epsilon,cacheMode,isOutputKv,kRope,cKv,&workspaceSize,&executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnKvRmsNormRopeCacheGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory using the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnKvRmsNormRopeCache.
  ret = aclnnKvRmsNormRopeCache(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnKvRmsNormRopeCache failed. ERROR: %d\n", ret); return ret);

  // 4. Boilerplate: wait synchronously for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output values and copy the result from device memory to host memory. Modify as required by the specific API definition.
  PrintOutResult(kpeCacheShape, &kpeCacheDeviceAddr);
  PrintOutResult(ckvCacheShape, &ckvCacheDeviceAddr);

  // 6. Release aclTensor and aclScalar. Modify as required by the specific API definition.
  aclDestroyTensor(kv);
  aclDestroyTensor(gamma);
  aclDestroyTensor(cos);
  aclDestroyTensor(sin);
  aclDestroyTensor(index);
  aclDestroyTensor(kpeCache);
  aclDestroyTensor(ckvCache);
  aclDestroyTensor(kRope);
  aclDestroyTensor(cKv);

  // 7. Release device resources.
  aclrtFree(kvDeviceAddr);
  aclrtFree(gammaDeviceAddr);
  aclrtFree(cosDeviceAddr);
  aclrtFree(sinDeviceAddr);
  aclrtFree(indexDeviceAddr);
  aclrtFree(kpeCacheDeviceAddr);
  aclrtFree(ckvCacheDeviceAddr);
  aclrtFree(kRopeDeviceAddr);
  aclrtFree(cKvDeviceAddr);


  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
