# aclnnKdaInputProj

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T10:49:05.430Z -->

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

- **Description**: Pre-processing computation of Recurrent KDA in inference use cases. Projects the input $x$ to obtain `qkv`, `beta`, `gate`, and `g`, respectively:
    - Stage1: AIC computes `X @ W_beta`, `X @ W_gate`, and `X @ W_g` in parallel (BF16 Cube, FP32 accumulation); AIV executes DynamicMxQuant in parallel.
    - Stage2: AIV performs element-wise Sigmoid on `beta` and writes the result back in FP32; AIC executes QuantMatmul(`qkv`).
- **Formula**:

    $$
    \mathbf{beta}_{raw} = X W_{\beta}^{\mathrm{T}},\quad
    \mathbf{gate} = X W_{gate}^{\mathrm{T}},\quad
    \mathbf{g} = X W_{g}^{\mathrm{T}}
    $$

    $$
    \mathbf{beta} = \sigma(\mathbf{beta}_{raw}) = \frac{1}{1 + e^{-\mathbf{beta}_{raw}}}
    $$

    $$
    \mathbf{qkv} = \mathrm{QuantMatmul}\bigl(\mathrm{DynamicMxQuant}(X),\; W_{qkv},\; \mathrm{weight\_qkv\_scale}\bigr)
    $$

    where $X$ is a $[T,K]$ BF16 tensor; `gate`/`g` are BF16; `beta` is kept in FP32 before Sigmoid is applied.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnKdaInputProjGetWorkspaceSize" API to obtain the input parameters and calculate the required workspace size based on the process, and then call the "aclnnKdaInputProj" API to perform the computation.

```cpp
aclnnStatus aclnnKdaInputProjGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *weightQkv,
  const aclTensor *weightBeta,
  const aclTensor *weightGate,
  const aclTensor *weightG,
  const aclTensor *weightQkvScale,
  const aclTensor *qkvOut,
  const aclTensor *betaOut,
  const aclTensor *gateOut,
  const aclTensor *gOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnKdaInputProj(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnKdaInputProjGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1700px"><colgroup>
  <col style="width: 181px">
  <col style="width: 121px">
  <col style="width: 301px">
  <col style="width: 331px">
  <col style="width: 217px">
  <col style="width: 111px">
  <col style="width: 240px">
  <col style="width: 150px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Hidden layer input, corresponding to $X$ in the formula.</td>
      <td>Required. Empty tensors are not supported. AutoContiguous is performed internally on the input.</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>[T, K]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>weightQkv</td>
      <td>Input</td>
      <td>qkv projection weight, corresponding to $W_{qkv}$ in the formula.</td>
      <td>Required. Empty tensors are not supported. The K dimension must be the same as the K dimension of x.</td>
      <td>FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>[K, N_qkv]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weightBeta</td>
      <td>Input</td>
      <td>Beta projection weight, corresponding to $W_{\beta}$ in the formula.</td>
      <td>Required. Empty tensors are not supported. The K dimension must be the same as the K dimension of x.</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>[K, N_beta]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weightGate</td>
      <td>Input</td>
      <td>Gate projection weight, corresponding to $W_{gate}$ in the formula.</td>
      <td>Required. Empty tensors are not supported. The K dimension must be the same as that of x.</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>[K, N_gate]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weightG</td>
      <td>Input</td>
      <td>Projection weight for g, corresponding to $W_{g}$ in the formula.</td>
      <td>Required. Empty tensors are not supported. The K dimension must be the same as that of x.</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>[K, N_g]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weightQkvScale</td>
      <td>Input</td>
      <td>MX quantization scaling factor of weightQkv. The last dimension is packed with high and low bits.</td>
      <td>Required. Empty tensors are not supported. The last dimension is fixed to 2. The MX block size is 64.</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>When weightQkv is a transposed view: [N_qkv, Ceil(K/64), 2]; otherwise: [Ceil(K/64), N_qkv, 2]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>qkvOut</td>
      <td>Output</td>
      <td>qkv projection output.</td>
      <td></td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>[T, N_qkv]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>betaOut</td>
      <td>Output</td>
      <td>Output after Sigmoid is applied to the beta projection, corresponding to $\mathbf{beta}$ in the formula.</td>
      <td>Preallocated by the caller. The output is FP32.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>[T, N_beta]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>gateOut</td>
      <td>Output</td>
      <td>gate projection output, corresponding to $\mathbf{gate}$ in the formula.</td>
      <td>Pre-allocated by the caller.</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>[T, N_gate]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>gOut</td>
      <td>Output</td>
      <td>Projection output g, corresponding to $\mathbf{g}$ in the formula.</td>
      <td>Pre-allocated by the caller.</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>[T, N_g]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output parameter</td>
      <td>Returns the size of the workspace that needs to be allocated on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output parameter</td>
      <td>Returns the op executor, which contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:
  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 275px">
  <col style="width: 125px">
  <col style="width: 755px">
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
      <td>A null pointer exists in the required parameters.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The shape or dtype of the input parameters is not within the supported range, or the K dimension of the weight is inconsistent with x.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>An exception occurs when the API calls the NPU runtime API for memory operations.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>A tiling exception occurs when the dtype or shape of the input parameters is incorrect.</td>
    </tr>
  </tbody>
  </table>

## aclnnKdaInputProj

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 173px">
  <col style="width: 133px">
  <col style="width: 849px">
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
      <td>Workspace memory address allocated on the device. A null pointer can be passed in when workspaceSize is 0.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the device, obtained by calling the first-phase API aclnnKdaInputProjGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, containing the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnKdaInputProj defaults to a deterministic implementation.
- Meanings and constraints of shape fields
    - T: number of tokens, greater than 0.
    - K: hidden size, greater than 0.
    - N_qkv / N_beta / N_gate / N_g: number of output features for each projection, greater than 0.
- Data type constraints
    - x, weightBeta, weightGate, weightG, gateOut, gOut, qkvOut: BF16.
    - weightQkv: FLOAT8_E4M3FN.
    - weightQkvScale: FLOAT8_E8M0.
    - betaOut: FLOAT.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md). For the complete source code, see [test_aclnn_kda_input_proj.cpp](../examples/test_aclnn_kda_input_proj.cpp).

```Cpp
#include <cstdint>
#include <cstdio>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_kda_input_proj.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...) printf(message, ##__VA_ARGS__)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * static_cast<int64_t>(sizeof(T));
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), static_cast<int64_t>(shape.size()), dataType, strides.data(), 0,
                              ACL_FORMAT_ND, shape.data(), static_cast<int64_t>(shape.size()), *deviceAddr);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    const int64_t T = 1;
    const int64_t K = 256;
    const int64_t Nqkv = 128;
    const int64_t Nbeta = 16;
    const int64_t Ngate = 32;
    const int64_t Ng = 32;
    const int64_t mxK = (K + 63) / 64;

    std::vector<int64_t> xShape = {T, K};
    std::vector<int64_t> wQkvShape = {K, Nqkv};
    std::vector<int64_t> wBetaShape = {K, Nbeta};
    std::vector<int64_t> wGateShape = {K, Ngate};
    std::vector<int64_t> wGShape = {K, Ng};
    std::vector<int64_t> weightQkvScaleShape = {mxK, Nqkv, 2};
    std::vector<int64_t> qkvShape = {T, Nqkv};
    std::vector<int64_t> betaShape = {T, Nbeta};
    std::vector<int64_t> gateShape = {T, Ngate};
    std::vector<int64_t> gShape = {T, Ng};

    std::vector<uint16_t> xHost(T * K, 0x3F80);
    std::vector<uint8_t> wQkvHost(Nqkv * K, 0);
    std::vector<uint16_t> wBetaHost(Nbeta * K, 0);
    std::vector<uint16_t> wGateHost(Ngate * K, 0);
    std::vector<uint16_t> wGHost(Ng * K, 0);
    std::vector<uint8_t> weightQkvScaleHost(Nqkv * mxK * 2, 0);
    std::vector<uint16_t> qkvHost(T * Nqkv, 0);
    std::vector<float> betaHost(T * Nbeta, 0.0f);
    std::vector<uint16_t> gateHost(T * Ngate, 0);
    std::vector<uint16_t> gHost(T * Ng, 0);

    void *xDev = nullptr, *wQkvDev = nullptr, *wBetaDev = nullptr, *wGateDev = nullptr, *wGDev = nullptr;
    void *weightQkvScaleDev = nullptr, *qkvDev = nullptr, *betaDev = nullptr, *gateDev = nullptr, *gDev = nullptr;
    aclTensor *x = nullptr, *wQkv = nullptr, *wBeta = nullptr, *wGate = nullptr, *wG = nullptr;
    aclTensor *weightQkvScale = nullptr, *qkv = nullptr, *beta = nullptr, *gate = nullptr, *g = nullptr;

    CreateAclTensor(xHost, xShape, &xDev, ACL_BF16, &x);
    CreateAclTensor(wQkvHost, wQkvShape, &wQkvDev, ACL_FLOAT8_E4M3FN, &wQkv);
    CreateAclTensor(wBetaHost, wBetaShape, &wBetaDev, ACL_BF16, &wBeta);
    CreateAclTensor(wGateHost, wGateShape, &wGateDev, ACL_BF16, &wGate);
    CreateAclTensor(wGHost, wGShape, &wGDev, ACL_BF16, &wG);
    CreateAclTensor(weightQkvScaleHost, weightQkvScaleShape, &weightQkvScaleDev, ACL_FLOAT8_E8M0, &weightQkvScale);
    CreateAclTensor(qkvHost, qkvShape, &qkvDev, ACL_BF16, &qkv);
    CreateAclTensor(betaHost, betaShape, &betaDev, ACL_FLOAT, &beta);
    CreateAclTensor(gateHost, gateShape, &gateDev, ACL_BF16, &gate);
    CreateAclTensor(gHost, gShape, &gDev, ACL_BF16, &g);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnKdaInputProjGetWorkspaceSize(x, wQkv, wBeta, wGate, wG, weightQkvScale, qkv, beta, gate, g, &workspaceSize,
                                         &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void *workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    ret = aclnnKdaInputProj(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnKdaInputProj failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclDestroyTensor(x);
    aclDestroyTensor(wQkv);
    aclDestroyTensor(wBeta);
    aclDestroyTensor(wGate);
    aclDestroyTensor(wG);
    aclDestroyTensor(weightQkvScale);
    aclDestroyTensor(qkv);
    aclDestroyTensor(beta);
    aclDestroyTensor(gate);
    aclDestroyTensor(g);
    aclrtFree(xDev);
    aclrtFree(wQkvDev);
    aclrtFree(wBetaDev);
    aclrtFree(wGateDev);
    aclrtFree(wGDev);
    aclrtFree(weightQkvScaleDev);
    aclrtFree(qkvDev);
    aclrtFree(betaDev);
    aclrtFree(gateDev);
    aclrtFree(gDev);
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
