# aclnnScatterPaCache

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T12:58:43.355Z -->

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

- Description: Updates the key at the specified position in KCache.

- Formula:
  - Use case 1:

    ```python
    key:[batch * seq_len, num_head, k_head_size]
    keyCache:[num_blocks, block_size, num_head, k_head_size]
    slotMapping:[batch * seq_len]
    cacheMode:"Norm"
    ```

    $$
    keyCache = slotMapping(key)
    $$

  - Scenario 2:

    ```python
    key:[batch, seq_len, num_head, k_head_size]
    keyCache:[num_blocks, block_size, 1, k_head_size]
    slotMapping:[batch, num_head]
    compressLensOptional:[batch, num_head]
    compressSeqOffsetOptional:[batch * num_head]
    seqLensOptional:[batch]
    cacheMode:"Norm"
    ```

    $$
    \begin{aligned}
    keyCache =\ & slotMapping(key[: compressSeqOffset], \\
    & ReduceMean(key[compressSeqOffset : compressSeqOffset + compressLens]), \\
    & key[compressSeqOffset + compressLens : seqLens])
    \end{aligned}
    $$

  - Scenario 3:

    ```python
    key:[batch, seq_len, num_head, k_head_size]
    keyCache:[num_blocks, block_size, 1, k_head_size]
    slotMapping:[batch, num_head]
    compressLensOptional:[batch * num_head]
    seqLensOptional:[batch]
    cacheMode:"Norm"
    ```

    $$
    keyCache = slotMapping(key[seqLens - compressLens : seqLens])
    $$

  The above scenarios are distinguished based on the constructed parameters. The first input parameter construction corresponds to Scenario 1, the second construction corresponds to Scenario 2, and the third construction corresponds to Scenario 3. Scenario 1 does not include the three optional inputs: compressLensOptional, seqLensOptional, and compressSeqOffsetOptional. Scenario 3 does not include the compressSeqOffsetOptional optional input.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnScatterPaCacheGetWorkspaceSize" first to obtain the workspace size required for the computation and the executor containing the computation flow of the operator, and then call "aclnnScatterPaCache" to perform the computation.

```c++
aclnnStatus aclnnScatterPaCacheGetWorkspaceSize(
  const aclTensor *key,
  aclTensor *keyCacheRef,
  const aclTensor *slotMapping,
  const aclTensor *compressLensOptional,
  const aclTensor *compressSeqOffsetOptional,
  const aclTensor *seqLensOptional,
  char            *cacheMode,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```c++
aclnnStatus aclnnScatterPaCache(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnScatterPaCacheGetWorkspaceSize

- **Parameters**

  <table class="tg" style="undefined;table-layout: fixed; width: 1548px"><colgroup>
  <col style="width: 265px">
  <col style="width: 86px">
  <col style="width: 269px">
  <col style="width: 462px">
  <col style="width: 172px">
  <col style="width: 111px">
  <col style="width: 87px">
  <col style="width: 96px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">Parameter</th>
      <th class="tg-0pky">Input/Output</th>
      <th class="tg-0pky">Description</th>
      <th class="tg-0pky">Instruction</th>
      <th class="tg-0pky">Data Type</th>
      <th class="tg-0pky">Data Format</th>
      <th class="tg-0pky">Shape</th>
      <th class="tg-0pky">Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">key(aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Key to be updated, key in the formula.</td>
      <td class="tg-0pky"><ul><li>Supports null tensor.</li><li>HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1, and FLOAT4_E1M2 support only the use case where key is 3-dimensional.</li><li>The shape must be [batch * seq_len, num_head, k_head_size] or [batch, seq_len, num_head, k_head_size]. For FLOAT4_E2M1 and FLOAT4_E1M2, k_head_size must be an even number.</li></ul></td>
      <td class="tg-0pky">FLOAT16, FLOAT, BFLOAT16, INT8, UINT8, INT16, UINT16, INT32, UINT32, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1, FLOAT4_E1M2</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">3-4</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">keyCacheRef(aclTensor*)</td>
      <td class="tg-0pky">Input/Output</td>
      <td class="tg-0pky">keyCache to be updated, keyCache in the formula.</td>
      <td class="tg-0pky"><ul><li>Supports null tensor.</li><li>If key is a 3D tensor, the shape must be [num_blocks, block_size, num_head, k_head_size]; if key is a 4D tensor, the shape must be [num_blocks, block_size, 1, k_head_size].</li></ul></td>
      <td class="tg-0pky">Same as key.</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">4</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">slotMapping(aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Storage offset of each token of key in the cache, slotMapping in the formula.</td>
      <td class="tg-0pky"><ul><li>Supports null tensor.</li><li>If key is a 3D tensor, the shape must be [batch * seq_len]; if key is a 4D tensor, the shape must be [batch, num_head].</li><li>The value range is [0, num_blocks * block_size-1], and element values must not be duplicated. Correctness is not guaranteed if they are duplicated.</li></ul></td>
      <td class="tg-0pky">INT32, INT64</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-2</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">compressLensOptional(aclTensor*)</td>
      <td class="tg-0pky">optional input</td>
      <td class="tg-0pky">Compression length, which is compressLens in the formula.</td>
      <td class="tg-0pky"><ul><li>Supports null tensor.</li><li>When key is 4-dimensional and compressSeqOffsetOptional is not a null pointer, the shape is [batch, num_head]; when key is 4-dimensional and compressSeqOffsetOptional is a null pointer, the shape is [batch * num_head].</li><li>Pass a null pointer in use case 1.</li></ul></td>
      <td class="tg-0pky">Consistent with slotMapping.</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-2</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">compressSeqOffsetOptional(aclTensor*)</td>
      <td class="tg-0pky">optional input</td>
      <td class="tg-0pky">Compression start point of each head in each batch, which is compressSeqOffset in the formula.</td>
      <td class="tg-0pky"><ul><li>Supports null tensor.</li><li>The shape is [batch * num_head].</li><li>Pass a null pointer for use case 1 and use case 3.</li></ul></td>
      <td class="tg-0pky">Same as slotMapping.</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">seqLensOptional(aclTensor*)</td>
      <td class="tg-0pky">Optional input</td>
      <td class="tg-0pky">Actual seqLens of each batch, i.e., seqLens in the formula.</td>
      <td class="tg-0pky"><ul><li>Supports null tensor.</li><li>The shape is [batch].</li><li>A null pointer is passed in use case 1.</li></ul></td>
      <td class="tg-0pky">Same as slotMapping.</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">cacheMode(char*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Memory layout format of keyCacheRef.</td>
      <td class="tg-0pky">Reserved parameter. Currently, only the ND data format is supported, and this parameter does not take effect.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">workspaceSize (uint64_t*)</td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">Returns the size of the workspace to be requested on the Device side.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">executor(aclOpExecutor**)</td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">Returns the op executor, which contains the operator computation process.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
  </tbody></table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1152px"><colgroup>
  <col style="width: 302px">
  <col style="width: 119px">
  <col style="width: 731px">
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
      <td>The input key, keyCacheRef, or slotMapping is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type of key is not in the supported range.</td>
    </tr>
    <tr>
      <td>The data types of key and keyCacheRef are inconsistent.</td>
    </tr>
    <tr>
      <td>The data types of slotMapping, compressLensOptional, compressSeqOffsetOptional, and seqLensOptional are inconsistent.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>561002</td>
      <td>The number of dimensions of key is not 3 or 4.</td>
    </tr>
  </tbody>
  </table>

## aclnnScatterPaCache

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
      <td>workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>workspace size allocated on the Device, obtained by the first-phase API aclnnScatterPaCacheGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream where the task is executed.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation: aclnnScatterPaCache defaults to a deterministic implementation.
- The variables used in shape descriptions in Parameters are as follows:
  - batch: the number of sequences in the current input (the number of samples processed at a time), which is a positive integer.
  - seq_len: the length of the sequence, which is a positive integer.
  - num_head: Number of "heads" in multi-head attention, which is a positive integer.
  - k_head_size: Feature dimension of key per attention head (length of a single-head key), which is a positive integer.
  - num_blocks: Total number of pre-allocated blocks in keyCache for storing key data of all sequences, which is a positive integer.
  - block_size: Number of tokens contained in each cache block, which is a positive integer.
- Input value range restriction: Each element value in seqLensOptional and compressLensOptional must satisfy the formula: reduceSum(seqLensOptional[i] - compressLensOptional[i] + 1) <= num_blocks * block_size (corresponding to scenarios 2 and 3).

## Example

The sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT products</term>:

  ```c++
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_scatter_pa_cache.h"

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

  int Init(int32_t deviceId, aclrtStream* stream) {
    // Boilerplate for resource initialization
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
    // Call aclrtMalloc to allocate device-side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host-side data to the device-side memory.
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
    // 1. (Boilerplate) Device/stream initialization. For details, see the acl API manual.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct inputs and outputs. Customize the construction as required by the API.
    std::vector<int64_t> keyShape = {2, 2, 3, 7};
    std::vector<int64_t> keyCacheShape = {2, 2, 1, 7};
    std::vector<int64_t> slotMappingShape = {2, 3};

    std::vector<int64_t> compressLensShape = {2, 3};
    std::vector<int64_t> compressSeqOffsetShape = {6};
    std::vector<int64_t> seqLensShape = {2};
    void* keyDeviceAddr = nullptr;
    void* slotMappingDeviceAddr = nullptr;
    void* keyCacheDeviceAddr = nullptr;
    void* compressLensDeviceAddr = nullptr;
    void* compressSeqOffsetDeviceAddr = nullptr;
    void* seqLensDeviceAddr = nullptr;

    aclTensor* key = nullptr;
    aclTensor* slotMapping = nullptr;
    aclTensor* keyCache = nullptr;
    aclTensor* compressLens = nullptr;
    aclTensor* compressSeqOffset = nullptr;
    aclTensor* seqLens = nullptr;
    char* cacheMode = const_cast<char*>("Norm");

    std::vector<float> hostKey = {1};
    std::vector<int32_t> hostSlotMapping = {0, 3, 6, 9, 12, 15};
    std::vector<float> hostKeyCacheRef = {1};
    std::vector<int32_t> hostCompressLens = {1, 0, 0, 0, 1, 0};
    std::vector<int32_t> hostCompressSeqOffset = {0, 0, 1, 0, 1, 1};
    std::vector<int32_t> hostSeqLens = {2, 1};

    // Create the key aclTensor.
    ret = CreateAclTensor(hostKey, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT, &key);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the slotMapping aclTensor.
    ret = CreateAclTensor(hostSlotMapping, slotMappingShape, &slotMappingDeviceAddr, aclDataType::ACL_INT32, &slotMapping);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the keyCache aclTensor.
    ret = CreateAclTensor(hostKeyCacheRef, keyCacheShape, &keyCacheDeviceAddr, aclDataType::ACL_FLOAT, &keyCache);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the compressLens aclTensor.
    ret = CreateAclTensor(hostCompressLens, compressLensShape, &compressLensDeviceAddr, aclDataType::ACL_INT32, &compressLens);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the compressSeqOffset aclTensor.
    ret = CreateAclTensor(hostCompressSeqOffset, compressSeqOffsetShape, &compressSeqOffsetDeviceAddr, aclDataType::ACL_INT32, &compressSeqOffset);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the seqLens aclTensor.
    ret = CreateAclTensor(hostSeqLens, seqLensShape, &seqLensDeviceAddr, aclDataType::ACL_INT32, &seqLens);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Replace it with the actual API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the aclnnScatterPaCache first-phase API.
    ret = aclnnScatterPaCacheGetWorkspaceSize(key, keyCache, slotMapping, compressLens, compressSeqOffset, seqLens, cacheMode, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScatterPaCacheGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the aclnnScatterPaCache second-phase API.
    ret = aclnnScatterPaCache(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScatterPaCache failed. ERROR: %d\n", ret); return ret);

    // 4. (boilerplate) Synchronize and wait for the task to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value, and copy the result from device memory to host memory. Modify it according to the actual API definition.
    auto size = GetShapeSize(keyShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), keyDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
      LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release the aclTensor and aclScalar. Modify it according to the actual API definition.
    aclDestroyTensor(key);
    aclDestroyTensor(slotMapping);
    aclDestroyTensor(keyCache);
    aclDestroyTensor(compressLens);
    aclDestroyTensor(compressSeqOffset);
    aclDestroyTensor(seqLens);
    // 7. Release device resources. Set parameters according to the definition of the specific API.
    aclrtFree(keyDeviceAddr);
    aclrtFree(slotMappingDeviceAddr);
    aclrtFree(keyCacheDeviceAddr);
    aclrtFree(compressLensDeviceAddr);
    aclrtFree(compressSeqOffsetDeviceAddr);
    aclrtFree(seqLensDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
  }
  ```

<!-- end id7 -->
