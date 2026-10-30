# aclnnScatterPaKvCacheWithKScale

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T13:03:20.503Z -->

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

- Description: Updates the key and value at specified positions in the KV cache, and updates the scale value of the key.

- The input and output support the following scenarios:
  - Use case 1:

    ```python
    key:[batch * seq_len, num_head, head_size]
    value:[batch * seq_len, num_head, head_size]
    key_cache:[num_blocks, num_head, block_size, head_size]
    value_cache:[num_blocks, num_head, block_size, head_size]
    slot_mapping:[batch * seq_len]
    key_scale:[batch * seq_len, num_head]
    key_scale_cache:[num_blocks, num_head, block_size, 1]
    cache_layout:"BNBD"
    ```

    The dtype of key and value is FLOAT8_E5M2 or FLOAT8_E4M3FN, and the dtype of key_scale and key_scale_cache is FLOAT.

    Formula:

    For each token (i ∈ [0, num_tokens)) and each head (j ∈ [0, num_head)):

    ```python
    block_idx = slot_mapping[i] // block_size
    block_offset = slot_mapping[i] % block_size

    key_cache[block_idx][j][block_offset][:] = key[i][j][:]
    value_cache[block_idx][j][block_offset][:] = value[i][j][:]
    key_scale_cache[block_idx][j][block_offset][0] = key_scale[i][j]
    ```

    Where:
    - num_tokens = batch * seq_len
    - block_idx: index of the block that slot_mapping maps to.
    - block_offset: offset within the block.

<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT products</term>: Only use case 1 is supported.

<!-- end id7 -->

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call the "aclnnScatterPaKvCacheWithKScaleGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call the "aclnnScatterPaKvCacheWithKScale" API to execute the computation.

```Cpp
aclnnStatus aclnnScatterPaKvCacheWithKScaleGetWorkspaceSize(
  const aclTensor   *key,
  const aclTensor   *value,
  aclTensor         *keyCacheRef,
  aclTensor         *valueCacheRef,
  const aclTensor   *slotMapping,
  aclTensor         *keyScale,
  aclTensor         *keyScaleCacheRef,
  char              *cacheLayoutOptional,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnScatterPaKvCacheWithKScale(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnScatterPaKvCacheWithKScaleGetWorkspaceSize

- **Parameters**

  <table class="tg" style="undefined;table-layout: fixed; width: 1565px"><colgroup>
  <col style="width: 230px">
  <col style="width: 120px">
  <col style="width: 270px">
  <col style="width: 350px">
  <col style="width: 200px">
  <col style="width: 115px">
  <col style="width: 135px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">Parameter</th>
      <th class="tg-0pky">Input/Output</th>
      <th class="tg-0pky">Description</th>
      <th class="tg-0pky">Instruction</th>
      <th class="tg-0pky">Data Type</th>
      <th class="tg-0pky">Data Format</th>
      <th class="tg-0pky">Dimension (shape)</th>
      <th class="tg-0pky">Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">key (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Key to be updated, that is, the keys of multiple tokens in the current step.</td>
      <td class="tg-0pky"><ul><li>Does not support empty tensors.</li></ul></td>
      <td class="tg-0pky">FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">3</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">value (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Value to be updated, that is, the values of multiple tokens in the current step.</td>
      <td class="tg-0pky"><ul><li>Does not support empty tensors.</li></ul></td>
      <td class="tg-0pky">FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">3</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">keyCacheRef (aclTensor*)</td>
      <td class="tg-0pky">Input/Output</td>
      <td class="tg-0pky">Key cache that needs to be updated, that is, the key cache of the current layer.</td>
      <td class="tg-0pky"><ul><li>Does not support empty tensors.</li></ul></td>
      <td class="tg-0pky">Same as key</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">4</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">valueCacheRef (aclTensor*)</td>
      <td class="tg-0pky">Input/Output</td>
      <td class="tg-0pky">Value cache that needs to be updated, the value cache of the current layer.</td>
      <td class="tg-0pky"><ul><li>Does not support empty tensors.</li></ul></td>
      <td class="tg-0pky">Consistent with value</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">4</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">slotMapping (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Storage offset of the key or value of each token in the cache.</td>
      <td class="tg-0pky"><ul><li>Does not support empty tensors.</li></ul></td>
      <td class="tg-0pky">INT32, INT64</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">keyScale (aclTensor*)</td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Key scale values to be updated, that is, key scales of multiple tokens in the current step.</td>
      <td class="tg-0pky"><ul><li>Does not support empty tensors.</li><li>Two-dimensional tensor; the last axis can be non-contiguous.</li></ul></td>
      <td class="tg-0pky">FLOAT</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">2</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">keyScaleCacheRef (aclTensor*)</td>
      <td class="tg-0pky">Input/Output</td>
      <td class="tg-0pky">Key scale cache that needs to be updated, that is, the key scale cache of the current layer.</td>
      <td class="tg-0pky"><ul><li>Does not support empty tensors.</li><li>4-D tensor with the last dimension of 1.</li></ul></td>
      <td class="tg-0pky">FLOAT</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">4</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">cacheLayoutOptional (char*)</td>
      <td class="tg-0lax">Input</td>
      <td class="tg-0lax">Indicates the memory layout of keyCacheRef and valueCacheRef.</td>
      <td class="tg-0lax"><ul><li>If an empty pointer or "BNBD" is passed, the format of keyCacheRef and valueCacheRef is [num_blocks, num_head, block_size, head_size].</li></ul></td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">workspaceSize (uint64_t*)</td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">Returns the size of the workspace that needs to be applied on the Device side.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">executor (aclOpExecutor**)</td>
      <td class="tg-0lax">Output</td>
      <td class="tg-0lax">Returns the op executor containing the operator computation flow.</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
  </tbody></table>

- **Return value**

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
      <td>key, value, keyCacheRef, valueCacheRef, slotMapping, keyScale, or keyScaleCacheRef is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data type of key and value is not within the supported range (FLOAT8_E5M2, FLOAT8_E4M3FN).</td>
    </tr>
    <tr>
      <td>The data types of key, keyCacheRef, value, and valueCacheRef are inconsistent.</td>
    </tr>
    <tr>
      <td>The data type of slotMapping is not within the supported range (INT32, INT64).</td>
    </tr>
    <tr>
      <td>The data types of keyScale and keyScaleCacheRef are inconsistent or not within the supported range (FLOAT).</td>
    </tr>
    <tr>
      <td>The dims of key or value are not 3, or the dims of slotMapping is not 1.</td>
    </tr>
  </tbody>
  </table>

## aclnnScatterPaKvCacheWithKScale

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
      <td>Workspace memory address applied on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size applied on the Device side, obtained by the first-phase API aclnnScatterPaKvCacheWithKScaleGetWorkspaceSize.</td>
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

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnScatterPaKvCacheWithKScale defaults to a deterministic implementation.
- The data types of key, value, keyCacheRef, and valueCacheRef must be consistent.
- The value range of slotMapping is [0, num_blocks*block_size-1], and element values in slotMapping are guaranteed to be unique. Correctness is not guaranteed if duplicates exist.
- The first two dimensions of key and value must be the same;
- keyScale is a two-dimensional tensor with shape [batch * seq_len, num_head]. The trailing axis can be non-contiguous;
- keyScaleCacheRef is a four-dimensional tensor with shape [num_blocks, num_head, block_size, 1]. The last dimension must be 1, and the trailing axis must be contiguous.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

<!-- npu="950" id8 -->
- <term>Ascend 950PR&950DT products</term>:

  ```c++
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_scatter_pa_kv_cache_with_k_scale.h"

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

  int64_t GetShapeSize(const std::vector<int64_t> &shape)
  {
    int64_t shapeSize = 1;
    for (auto i : shape) {
      shapeSize *= i;
    }
    return shapeSize;
  }

  int Init(int32_t deviceId, aclrtStream* stream) {
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
  }

  template <typename T>
  int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape,
                      void** deviceAddr, aclDataType dataType, aclTensor** tensor, aclFormat format) {
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
      strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
  }

  int main() {
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> keyShape = {2, 2, 64};
    std::vector<int64_t> valueShape = {2, 2, 64};
    std::vector<int64_t> keyCacheShape = {4, 2, 16, 64};
    std::vector<int64_t> valueCacheShape = {4, 2, 16, 64};
    std::vector<int64_t> slotMappingShape = {2};
    std::vector<int64_t> keyScaleShape = {2, 2};
    std::vector<int64_t> keyScaleCacheShape = {4, 2, 16, 1};

    void* keyDeviceAddr = nullptr;
    void* valueDeviceAddr = nullptr;
    void* slotMappingDeviceAddr = nullptr;
    void* keyCacheDeviceAddr = nullptr;
    void* valueCacheDeviceAddr = nullptr;
    void* keyScaleDeviceAddr = nullptr;
    void* keyScaleCacheDeviceAddr = nullptr;

    aclTensor* key = nullptr;
    aclTensor* value = nullptr;
    aclTensor* slotMapping = nullptr;
    aclTensor* keyCache = nullptr;
    aclTensor* valueCache = nullptr;
    aclTensor* keyScale = nullptr;
    aclTensor* keyScaleCache = nullptr;
    char * cacheLayout = "BNBD";

    std::vector<uint8_t> hostKey(256, 1);
    std::vector<uint8_t> hostValue(256, 1);
    std::vector<int32_t> hostSlotMapping(4, 1);
    std::vector<uint8_t> hostKeyCacheRef(8192, 1);
    std::vector<uint8_t> hostValueCacheRef(8192, 1);
    std::vector<float> hostKeyScale(2, 1.0f);
    std::vector<float> hostKeyScaleCacheRef(128, 1.0f);

    ret = CreateAclTensor(hostKey, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT8_E5M2, &key, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostValue, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT8_E5M2, &value, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostSlotMapping, slotMappingShape, &slotMappingDeviceAddr, aclDataType::ACL_INT32, &slotMapping, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostKeyCacheRef, keyCacheShape, &keyCacheDeviceAddr, aclDataType::ACL_FLOAT8_E5M2, &keyCache, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostValueCacheRef, valueCacheShape, &valueCacheDeviceAddr, aclDataType::ACL_FLOAT8_E5M2, &valueCache, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostKeyScale, keyScaleShape, &keyScaleDeviceAddr, aclDataType::ACL_FLOAT, &keyScale, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostKeyScaleCacheRef, keyScaleCacheShape, &keyScaleCacheDeviceAddr, aclDataType::ACL_FLOAT, &keyScaleCache, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnScatterPaKvCacheWithKScaleGetWorkspaceSize(key, value, keyCache, valueCache,
                                                         slotMapping, keyScale, keyScaleCache,
                                                         cacheLayout, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScatterPaKvCacheWithKScaleGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnScatterPaKvCacheWithKScale(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScatterPaKvCacheWithKScale failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    auto size = GetShapeSize(keyCacheShape);
    std::vector<uint8_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), keyCacheDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size && i < 10; i++) {
      LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }

    aclDestroyTensor(key);
    aclDestroyTensor(value);
    aclDestroyTensor(slotMapping);
    aclDestroyTensor(keyCache);
    aclDestroyTensor(valueCache);
    aclDestroyTensor(keyScale);
    aclDestroyTensor(keyScaleCache);

    aclrtFree(keyDeviceAddr);
    aclrtFree(valueDeviceAddr);
    aclrtFree(slotMappingDeviceAddr);
    aclrtFree(keyCacheDeviceAddr);
    aclrtFree(valueCacheDeviceAddr);
    aclrtFree(keyScaleDeviceAddr);
    aclrtFree(keyScaleCacheDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
  }
  ```

<!-- end id8 -->
