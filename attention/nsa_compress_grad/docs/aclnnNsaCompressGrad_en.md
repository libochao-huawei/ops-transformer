# aclnnNsaCompressGrad

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T12:01:08.889Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Not supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
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

- Description: backward computation of the aclnnNsaCompress operator.

- Formula:
  The forward formula for selected-attention is as follows:

    $$
    \text{dw} = \text{dk\_cmp} \cdot K^\top
    $$

    $$
    \text{dk} = W^\top \cdot \text{dk\_cmp}
    $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First, call "aclnnNsaCompressGradGetWorkspaceSize" to obtain the workspace size required for computation and the executor that contains the operator computation flow. Then, call "aclnnNsaCompressGrad" to execute the computation.

```c++
aclnnStatus aclnnNsaCompressGradGetWorkspaceSize(
  const aclTensor   *outputGrad,
  const aclTensor   *input,
  const aclTensor   *weight,
  const aclIntArray *actSeqLenOptionalOptional,
  int64_t            compressBlockSize,
  int64_t            compressStride,
  int64_t            actSeqLenType,
  char              *layoutOptionalOptional,
  const aclTensor   *inputGradOut,
  const aclTensor   *weightGradOut,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnNsaCompressGrad(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnNsaCompressGradGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1565px">
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
        <td>outputGrad</td>
        <td>Input</td>
        <td>Backward gradient of the forward operator output.</td>
        <td>-</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>Input</td>
        <td>Input</td>
        <td>Tensor to be compressed.</td>
        <td>-</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weight</td>
        <td>Input</td>
        <td>Weight for compression. Its shape must satisfy the broadcast relationship with the shape of input.</td>
        <td>Data type is the same as that of input.</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>actSeqLenOptionalOptional</td>
        <td>Input</td>
        <td>Specifies the S size for each batch. Required input when batch lengths differ.</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td></td>
      </tr>
      <tr>
        <td>compressBlockSize</td>
        <td>Input</td>
        <td>Compression sliding window size.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td></td>
      </tr>
      <tr>
        <td>compressStride</td>
        <td>Input</td>
        <td>Interval between two compression sliding windows.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td></td>
      </tr>
      <tr>
        <td>actSeqLenType</td>
        <td>Input</td>
        <td>Value: 0 or 1. Currently, only 0 is supported.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td></td>
      </tr>
      <tr>
        <td>layoutOptionalOptional</td>
        <td>Input</td>
        <td>Represents the data layout format of the input tensor. TND is supported.</td>
        <td>-</td>
        <td>string</td>
        <td>-</td>
        <td>-</td>
        <td></td>
      </tr>
      <tr>
        <td>inputGradOut</td>
        <td>Output</td>
        <td>Gradient of input, with the same shape as input.</td>
        <td>Same data type as input.</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weightGradOut</td>
        <td>Output</td>
        <td>Gradient of weight, with the same shape as weight.</td>
        <td>Same data type as weight.</td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the Device side-applied workspace size.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td></td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Returns the op executor, which contains the operator computation flow.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td></td>
      </tr>
    </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
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
      <td>input, weight, outputGrad, inputGrad, or weightGrad is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type of input or weight is not within the supported range.</td>
    </tr>
    <tr>
      <td>The shapes of input and weight cannot be broadcast.</td>
    </tr>
    <tr>
      <td>layoutOptional is invalid.</td>
    </tr>
  </tbody>
  </table>

## aclnnNsaCompressGrad

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
      <td>Memory address of the workspace side-applied on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace side-applied on the Device, obtained by the first-phase API aclnnNsaCompressGradGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream that executes the task.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnNsaCompressGrad defaults to a deterministic implementation.
- Non-contiguous output is not supported by this API.
- compressBlockSize and compressStride must be integer multiples of 16, and compressBlockSize > compressStride.

## Example

The following sample code shows how to call a single operator using aclnn (taking Atlas A2 training products as an example) and is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <fcntl.h>

#include "acl/acl.h"
#include "aclnnop/aclnn_nsa_compress_grad.h"

#define SUCCESS 0
#define FAILED 1

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR]  " fmt "\n", ##args)

#define CHECK_RET(cond, return_expr)     \
    do {                                 \
        if (!(cond)) {                   \
            return_expr;                 \
        }                                \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

bool ReadFile(const std::string &filePath, size_t &fileSize, void *buffer, size_t bufferSize)
{
    struct stat sBuf;
    int fileStatus = stat(filePath.data(), &sBuf);
    if (fileStatus == -1) {
        ERROR_LOG("failed to get file %s", filePath.c_str());
        return false;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        ERROR_LOG("%s is not a file, please enter a file", filePath.c_str());
        return false;
    }

    std::ifstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open()) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    std::filebuf *buf = file.rdbuf();
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    if (size == 0) {
        ERROR_LOG("file size is 0");
        file.close();
        return false;
    }
    if (size > bufferSize) {
        ERROR_LOG("file size is larger than buffer size");
        file.close();
        return false;
    }
    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char *>(buffer), size);
    fileSize = size;
    file.close();
    return true;
}

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
      shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtContext* context, aclrtStream* stream) {
    // Boilerplate for acl initialization.
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
                    aclDataType dataType, aclTensor** xOrResult) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate device-side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host-side data to the device-side memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of contiguous xOrResult.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor API to create an aclTensor.
    *xOrResult = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                 shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
    // 1. (Boilerplate) Initialize device/context/stream. For details, see the list of external acl APIs.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output. The construction is custom based on the API.
    int64_t headNum = 64;
    int64_t headDim = 128;
    int64_t blockSize = 32;
    int64_t blockStride = 16;
    int64_t blockNum = 15;
    int64_t seqLensSum = 272;
    int64_t seqLen = 3;
    std::vector<int64_t> outputGradShape = {blockNum, headNum, headDim};
    std::vector<int64_t> inputKVShape = {seqLensSum, headNum, headDim};
    std::vector<int64_t> weightShape = {blockSize, headNum};
    std::vector<int64_t> inputGradOutShape = {seqLensSum, headNum, headDim};
    std::vector<int64_t> weightGradOutShape = {blockSize, headNum};
    int64_t SeqLenType = 0;
    char layOut[] = "TND";

    void* outputGradDeviceAddr = nullptr;
    void* inputKVDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* inputGradOutDeviceAddr = nullptr;
    void* weightGradOutDeviceAddr = nullptr;

    aclTensor* outputGrad = nullptr;
    aclTensor* inputKV = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* inputGradOut = nullptr;
    aclTensor* weightGradOut = nullptr;

    std::vector<float> inputGradOutHostData(seqLensSum * headNum * headDim, 0.0);
    std::vector<float> weightGradOutHostData(blockSize * headNum, 0.0);

    std::vector<float> outputGradHostData(blockNum * headNum * headDim, 1.0);
    std::vector<float> inputKVHostData(seqLensSum * headNum * headDim, 1.0);
    std::vector<float> weightHostData(blockSize * headNum, 1.0);
    std::vector<int64_t> actSeqLenOptionalHostData = {0, 128, 272};

    aclIntArray *actSeqLenOptional = aclCreateIntArray(actSeqLenOptionalHostData.data(), actSeqLenOptionalHostData.size());

    // Create the dy aclTensor.
    ret = CreateAclTensor(outputGradHostData, outputGradShape, &outputGradDeviceAddr, aclDataType::ACL_FLOAT16,
                          &outputGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the x aclTensor.
    ret = CreateAclTensor(inputKVHostData, inputKVShape, &inputKVDeviceAddr, aclDataType::ACL_FLOAT16, &inputKV);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a gelu aclTensor.
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(inputGradOutHostData, inputGradOutShape, &inputGradOutDeviceAddr, aclDataType::ACL_FLOAT16, &inputGradOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(weightGradOutHostData, weightGradOutShape, &weightGradOutDeviceAddr, aclDataType::ACL_FLOAT16, &weightGradOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Replace Api with the actual API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnNsaCompressGrad.
    ret = aclnnNsaCompressGradGetWorkspaceSize(
        outputGrad, inputKV, weight, actSeqLenOptional, blockSize, blockStride, SeqLenType, layOut,
        inputGradOut, weightGradOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaCompressGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnNsaCompressGrad.
    ret = aclnnNsaCompressGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaCompressGrad failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Wait for the task execution to complete synchronously.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from device memory to host memory. Modify the code based on the API definition.
    auto size = GetShapeSize(inputGradOutShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), inputGradOutDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release the aclTensor and aclScalar. Modify the code based on the API definition.
    aclDestroyTensor(outputGrad);
    aclDestroyTensor(inputKV);
    aclDestroyTensor(weight);
    aclDestroyTensor(inputGradOut);
    aclDestroyTensor(weightGradOut);

    // 7. Release the device resources. Modify the code based on the API definition.
    aclrtFree(outputGradDeviceAddr);
    aclrtFree(inputKVDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(inputGradOutDeviceAddr);
    aclrtFree(weightGradOutDeviceAddr);
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
