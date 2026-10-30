# aclnnChunkKdaFwd

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T08:24:58.211Z -->

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

- Description: Performs chunked forward computation of KDA without CP splitting, and outputs the attention result, final status, and intermediate quantities required for backward computation.

- Formula:

  Each sequence is divided into $M$ chunks by $C=chunkSize$. Taking the $c$-th chunk as an example, $i$ and $j$ denote the token subscripts within the chunk, and $l_c$ denotes the subscript of the last valid token in the chunk. The following formulas omit the batch and head subscripts. When $HV>H$, it is a Grouped Value Attention (GVA) use case: $HV/H$ consecutive Value heads in each group share one Query/Key head; when head numbering starts from 0, the Value head numbered $h_v$ corresponds to the Query/Key head numbered $\lfloor h_v/(HV/H)\rfloor$.

  Let $x_{c,i,d}=g_{c,i,d}+dtBias_d$. If `dtBiasOptional` is not passed in, set $dtBias_d=0$. The activated gate is:

  $$
  \gamma_{c,i,d}=
  \begin{cases}
  g_{c,i,d}, & useGateInKernel=false,\\
  -\exp(A_{log})\operatorname{softplus}(x_{c,i,d}),
      & useGateInKernel=true,\ safeGate=false,\\
  lowerBound\operatorname{sigmoid}(\exp(A_{log})x_{c,i,d}),
      & useGateInKernel=true,\ safeGate=true.
  \end{cases}
  $$

  `gkOut` is the base-2 cumulative gate within a chunk:

  $$
  gk_{c,i,d}=\frac{1}{\ln 2}\sum_{t=0}^{i}\gamma_{c,t,d}.
  $$

  `qgOut` and `kgOut` are the gate-scaled Query and Key, respectively:

  $$
  qg_{c,i,d}=q_{c,i,d}2^{gk_{c,i,d}},\qquad
  kg_{c,i,d}=k_{c,i,d}2^{gk_{c,l_c,d}-gk_{c,i,d}}.
  $$

  `aqkOut` is the lower-triangular Query-Key coefficient matrix including the diagonal:

  $$
  Aqk_{c,i,j}=\mathbb{1}_{j\le i}\cdot scale
  \sum_d q_{c,i,d}k_{c,j,d}2^{gk_{c,i,d}-gk_{c,j,d}}.
  $$

  Define the strictly lower-triangular matrix $L_c$. `akkOut` is the inverse of $I+L_c$:

  $$
  L_{c,i,j}=\mathbb{1}_{j<i}\cdot\beta_{c,i}
  \sum_d k_{c,i,d}k_{c,j,d}2^{gk_{c,i,d}-gk_{c,j,d}},\qquad
  Akk_c=(I+L_c)^{-1}.
  $$

  Define $w^{seed}$ and $u^{seed}$:

  $$
  w^{seed}_{c,i,d}=\beta_{c,i}k_{c,i,d}2^{gk_{c,i,d}},\qquad
  u^{seed}_{c,i,e}=\beta_{c,i}v_{c,i,e}.
  $$

  `wOut` and `uOut` are:

  $$
  w_c=Akk_c\,w^{seed}_c,\qquad u_c=Akk_c\,u^{seed}_c.
  $$

  Let $h_c$ be the status before the $c$-th chunk is computed. If `initialStateOptional` is passed in, $h_0=initialStateOptional$; otherwise, $h_0=0$. `vNewOut`, the status of the next chunk, `hOut`, and `finalStateOut` are:

  $$
  v^{new}_c=u_c-w_c\,h_c,
  $$

  $$
  h_{c+1}=2^{gk_{c,l_c}}\odot h_c+kg_c^T\,v^{new}_c,
  $$

  $$
  hOut[c]=h_c,\qquad finalStateOut=h_M.
  $$

  where, $2^{gk_{c,l_c}}\odot h_c$ denotes the status matrix scaled along the $K$ dimension. `attnOut` is:

  $$
  attnOut_c=scale\cdot qg_c\,h_c+Aqk_c\,v^{new}_c.
  $$

- Symbol description:

  | Symbol | Meaning |
  | --- | --- |
  | B | Batch Size. |
  | S | Sequence length of the four-dimensional input. |
  | T | Total number of tokens of the three-dimensional input. |
  | H | Number of heads of Query and Key. |
  | HV | Number of heads of Value and gate. |
  | K | Head dim of Query and Key. |
  | V | Head dim of Value. |
  | N | Number of logical sequences. |
  | C | Chunk size. |
  | NC | Number of chunks in each sequence when `cuSeqlensOptional` is not passed in $\lceil S/C\rceil$; total number of chunks in all logical sequences when it is passed in. |

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the `aclnnChunkKdaFwdGetWorkspaceSize` API first to obtain the workspace size required for computation and the executor containing the operator computation flow, and then call the `aclnnChunkKdaFwd` API to perform the computation.

```cpp
aclnnStatus aclnnChunkKdaFwdGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional,
    const aclTensor *initialStateOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    const char *layout,
    double scale,
    int64_t chunkSize,
    bool safeGate,
    double lowerBound,
    bool useGateInKernel,
    bool stateVFirst,
    const aclTensor *attnOut,
    const aclTensor *finalStateOut,
    const aclTensor *gkOut,
    const aclTensor *aqkOut,
    const aclTensor *akkOut,
    const aclTensor *wOut,
    const aclTensor *uOut,
    const aclTensor *qgOut,
    const aclTensor *kgOut,
    const aclTensor *vNewOut,
    const aclTensor *hOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
```

```cpp
aclnnStatus aclnnChunkKdaFwd(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
```

## aclnnChunkKdaFwdGetWorkspaceSize

- **Parameters**

  | Name | Input/Output | Description | Instructions | Data Type | Data Format | Dimension (shape) | Non-contiguous Tensor |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | q | Input | Query in the formula. | Does not support null tensor. The shape is interpreted by `layout`. | FLOAT16, BFLOAT16 | ND | BSND: [B,S,H,K]<br>BNSD: [B,H,S,K]<br>TND: [T,H,K]<br>NTD: [H,T,K] | √ |
  | k | Input | Key in the formula. | Does not support null tensor. The shape and data type must be consistent with `q`. | FLOAT16, BFLOAT16 | ND | Same as `q` | √ |
  | v | Input | Value in the formula. | Does not support null tensor. The data type must be consistent with `q`. | FLOAT16, BFLOAT16 | ND | BSND: [B,S,HV,V]<br>BNSD: [B,HV,S,V]<br>TND: [T,HV,V]<br>NTD: [HV,T,V] | √ |
  | g | Input | Activated natural logarithm gate when `useGateInKernel=false`; raw gate when `useGateInKernel=true`. | Does not support null tensor. | FLOAT32, BFLOAT16 | ND | BSND: [B,S,HV,K]<br>BNSD: [B,HV,S,K]<br>TND: [T,HV,K]<br>NTD: [HV,T,K] | √ |
  | beta | Input | Delta coefficient in the formula $\beta$. | Does not support null tensor. | FLOAT32, BFLOAT16 | ND | BSND: [B,S,HV]<br>BNSD: [B,HV,S]<br>TND: [T,HV]<br>NTD: [HV,T] | √ |
  | aLogOptional | Optional input | Gate decay parameter $A_{log}$. | Must be passed in when `useGateInKernel=true`; otherwise, nullptr can be passed in. | FLOAT32 | ND | [HV] | √ |
  | dtBiasOptional | Optional input | Gate bias. | nullptr can be passed in, indicating that the bias is 0; participates in computation only when `useGateInKernel=true`. | FLOAT32 | ND | [HV×K] | √ |
  | initialStateOptional | Optional input | Initial status $h_0$ of each logical sequence. | nullptr can be passed in, indicating that the initial status is 0. The order of the last two dimensions is determined by `stateVFirst`. | FLOAT32 | ND | `stateVFirst=false`: [N,HV,K,V]<br>`stateVFirst=true`: [N,HV,V,K] | √ |
  | cuSeqlensOptional (aclIntArray*) | Optional input | Cumulative length of variable-length sequences. | nullptr can be passed in, indicating fixed-length sequences. If passed in, elements must be monotonically non-decreasing, the first element is 0, and the last element is S or T; four-dimensional variable-length input requires B=1. | INT64 | - | [N+1] | - |
  | chunkIndicesOptional (aclIntArray*) | Optional input | Chunk indices flattened by `(seq_id, chunk_id)`. | nullptr can be passed in. If passed in, `cuSeqlensOptional` must be passed in at the same time, and every chunk must be included in the sequence-major canonical order. | INT64 | - | [2×NC] | - |
  | layout | Input | Input layout. | Supports `BSND`, `BNSD`, `TND`, and `NTD`, which are case-sensitive. This parameter describes only the layout of `q`, `k`, `v`, `g`, and `beta`. | STRING | - | - | - |
  | scale | Input | Query scaling coefficient. | Usually takes $K^{-0.5}$. | DOUBLE | - | - | - |
  | chunkSize | Input | Chunk size. | Only supports 64 and 128. | INT64 | - | - | - |
  | safeGate | Input | Whether to use a bounded gate. | Takes the value true or false. | BOOL | - | - | - |
  | lowerBound | Input | Lower bound of the bounded gate. | When `safeGate=true` and `useGateInKernel=true`, the value range is [-5,0). | DOUBLE | - | - | - |
  | useGateInKernel | Input | Whether to compute the activated gate from the raw gate inside the operator. | Takes the value true or false. | BOOL | - | - | - |
  | stateVFirst | Input | Whether the status tensor uses (V,K) as the order of the last two dimensions. | Takes the value true or false. | BOOL | - | - | - |
  | attnOut | Output | Attention output in the formula. | Does not support null tensor. The data type must be consistent with `q` and is fixed to the sequence-major layout. | FLOAT16, BFLOAT16 | ND | Four-dimensional input: [B,S,HV,V]<br>Three-dimensional input: [T,HV,V] | × |
  | finalStateOut | Optional Outputs | Status $h_M$ updated by the last chunk of each logical sequence. | nullptr can be passed in, indicating not exporting this result. The order of the last two dimensions is determined by `stateVFirst`. | FLOAT32 | ND | `stateVFirst=false`: [N,HV,K,V]<br>`stateVFirst=true`: [N,HV,V,K] | × |
  | gkOut | Optional Outputs | Cumulative gate with base 2 within a chunk. | nullptr can be passed in, indicating not exporting this result. Fixed to the head-major layout. | FLOAT32 | ND | Four-dimensional input: [B,HV,S,K]<br>Three-dimensional input: [HV,T,K] | × |
  | aqkOut | Output | Query-Key coefficient matrix $Aqk$ within a chunk. | Does not support null tensor. The data type must be consistent with `q` and is fixed to the head-major layout. | FLOAT16, BFLOAT16 | ND | Four-dimensional input: [B,HV,S,C]<br>Three-dimensional input: [HV,T,C] | × |
  | akkOut | Output | Triangular inversion result $Akk$ within a chunk. | Does not support null tensor. The data type must be consistent with `q` and is fixed to the head-major layout. | FLOAT16, BFLOAT16 | ND | Four-dimensional input: [B,HV,S,C]<br>Three-dimensional input: [HV,T,C] | × |
  | wOut | Optional Outputs | Intermediate quantity $w$ in the formula. | nullptr can be passed in, indicating not exporting this result. If passed in, the data type must be consistent with `q` and is fixed to the head-major layout. | FLOAT16, BFLOAT16 | ND | Four-dimensional input: [B,HV,S,K]<br>Three-dimensional input: [HV,T,K] | × |
  | uOut | Optional Outputs | Intermediate quantity $u$ in the formula. | nullptr can be passed in, indicating not exporting this result. If passed in, the data type must be consistent with `q` and is fixed to the head-major layout. | FLOAT16, BFLOAT16 | ND | Four-dimensional input: [B,HV,S,V]<br>Three-dimensional input: [HV,T,V] | × |
  | qgOut | Optional Outputs | Gate-scaled Query. | nullptr can be passed in, indicating not exporting this result. If passed in, the data type must be consistent with `q` and is fixed to the head-major layout. | FLOAT16, BFLOAT16 | ND | Four-dimensional input: [B,HV,S,K]<br>Three-dimensional input: [HV,T,K] | × |
  | kgOut | Optional Outputs | Gate-scaled Key. | nullptr can be passed in, indicating not exporting this result. If passed in, the data type must be consistent with `q` and is fixed to the head-major layout. | FLOAT16, BFLOAT16 | ND | Four-dimensional input: [B,HV,S,K]<br>Three-dimensional input: [HV,T,K] | × |
  | vNewOut | Optional Outputs | Value after status correction. | nullptr can be passed in, indicating not exporting this result. If passed in, the data type must be consistent with `q` and is fixed to the head-major layout. | FLOAT16, BFLOAT16 | ND | Four-dimensional input: [B,HV,S,V]<br>Three-dimensional input: [HV,T,V] | × |
  | hOut | Optional Outputs | Status $h_c$ before computation of each chunk. | nullptr can be passed in, indicating not exporting this result. If passed in, the data type must be consistent with `q`, and it is fixed to the sequence-major layout, with the order of the last two dimensions determined by `stateVFirst`. | FLOAT16, BFLOAT16 | ND | Four-dimensional input and `stateVFirst=false`: [B,NC,HV,K,V]<br>Four-dimensional input and `stateVFirst=true`: [B,NC,HV,V,K]<br>Three-dimensional input and `stateVFirst=false`: [NC,HV,K,V]<br>Three-dimensional input and `stateVFirst=true`: [NC,HV,V,K] | × |
  | workspaceSize | Output | Returns the workspace size to be applied on the Device side. | The unit is byte. | - | - | - | - |
  | executor | Output | Returns the operator executor. | Contains the operator computation flow. | - | - | - | - |

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  | Return Value | Error Code | Description |
  | --- | --- | --- |
  | ACLNN_ERR_PARAM_NULLPTR | 161001 | `q`, `k`, `v`, `g`, `beta`, `attnOut`, `aqkOut`, or `akkOut` is a null pointer; or `aLogOptional` is a null pointer when `useGateInKernel=true`. |
  | ACLNN_ERR_PARAM_INVALID | 161002 | The shape, data type, data format, layout, chunk metadata, or attribute value of the input or output is out of the supported range. |

## aclnnChunkKdaFwd

- **Parameters**

  | Name | Input/Output | Description |
  | --- | --- | --- |
  | workspace | Input | Memory address of the workspace applied on the Device side. nullptr can be passed in when `workspaceSize` is 0. |
  | workspaceSize | Input | Size of the workspace applied on the Device side, obtained by the first-phase API `aclnnChunkKdaFwdGetWorkspaceSize`. |
  | executor | Input | Operator executor, containing the operator computation flow. |
  | stream | Input | Stream on which the task is executed. |

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - `aclnnChunkKdaFwd` defaults to a deterministic implementation.
- Input layout:
  - `layout` only supports `BSND`, `BNSD`, `TND`, and `NTD`.
  - `layout` only describes the `q`, `k`, `v`, `g`, and `beta` inputs; each output layout is fixed as listed in the parameter table.
- Dimension Constraints:
  - B, S, T, H, HV, K, and V must all be greater than 0. Null tensors are not supported.
  - H and HV must satisfy $0<H\le HV\le128$ and $HV\bmod H=0$.
  - The value range of K and V is [16,256], and K and V must be multiples of 16.
  - `chunkSize` only supports 64 and 128.
  - The four-dimensional variable-length input requires B=1.
  - Variable-length inputs support a maximum of 1024 logical sequences.
- Parameter combination constraints:
  - The data types of `q`, `k`, and `v` must be the same.
  - When `useGateInKernel=true`, `aLogOptional` must be passed in, and `dtBiasOptional` can be passed in as nullptr.
  - When `safeGate=true` and `useGateInKernel=true`, the value range of `lowerBound` is [-5,0).
  - When `chunkIndicesOptional` is passed in, `cuSeqlensOptional` must also be passed in, and chunk indices must use the sequence-major canonical order.

## Example

The sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <random>
#include "acl/acl.h"
#include "aclnnop/aclnn_chunk_kda_fwd.h"

using namespace std;

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
    // Boilerplate: AscendCL initialization.
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
    // Check whether the shape is valid.
    if (shape.empty()) {
        LOG_PRINT("CreateAclTensor: ERROR - shape is empty\n");
        return -1;
    }
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] <= 0) {
            LOG_PRINT("CreateAclTensor: ERROR - shape[%zu]=%ld is invalid\n", i, shape[i]);
            return -1;
        }
    }

    auto size = GetShapeSize(shape) * sizeof(T);

    // Check whether the hostData size matches.
    if (hostData.size() != static_cast<size_t>(GetShapeSize(shape))) {
        LOG_PRINT("CreateAclTensor: ERROR - hostData size mismatch: %zu vs %ld\n",
                  hostData.size(), GetShapeSize(shape));
        return -1;
    }

    // Call aclrtMalloc to allocate device memory.
    *deviceAddr = nullptr;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy host data to device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
              aclrtFree(*deviceAddr); *deviceAddr = nullptr; return ret);

    // Calculate the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    if (shape.size() > 1) {
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
    }

    // Call the aclCreateTensor API to create an aclTensor.
    *tensor = nullptr;
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed - returned nullptr\n");
              aclrtFree(*deviceAddr); *deviceAddr = nullptr; return -1);
    return 0;
}

float Bf16ToFloat(int16_t value) {
    // Convert bfloat16 to float: pad the low 16 bits with 0s and then interpret as float.
    uint32_t bits = static_cast<uint32_t>(static_cast<uint16_t>(value)) << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

int16_t FloatToBf16(float value) {
    // Convert float to bfloat16: round to nearest and then take the high 16 bits.
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t rounded = bits + 0x7FFF + ((bits >> 16) & 1);
    return static_cast<int16_t>(rounded >> 16);
}

float Sigmoid(float value) {
    return 1.0f / (1.0f + std::exp(-value));
}

void FreeResource(aclTensor *qTensor, aclTensor *kTensor, aclTensor *vTensor, aclTensor *gTensor,
                  aclTensor *betaTensor, aclTensor *aLogTensor, aclTensor *dtBiasTensor,
                  aclTensor *attnOutTensor, aclTensor *aqkTensor, aclTensor *akkTensor,
                  void *qAddr, void *kAddr, void *vAddr, void *gAddr, void *betaAddr,
                  void *aLogAddr, void *dtBiasAddr, void *attnOutAddr, void *aqkAddr, void *akkAddr,
                  void *workspaceAddr, int32_t deviceId, aclrtStream *stream)
{
    if (qTensor) aclDestroyTensor(qTensor);
    if (kTensor) aclDestroyTensor(kTensor);
    if (vTensor) aclDestroyTensor(vTensor);
    if (gTensor) aclDestroyTensor(gTensor);
    if (betaTensor) aclDestroyTensor(betaTensor);
    if (aLogTensor) aclDestroyTensor(aLogTensor);
    if (dtBiasTensor) aclDestroyTensor(dtBiasTensor);
    if (attnOutTensor) aclDestroyTensor(attnOutTensor);
    if (aqkTensor) aclDestroyTensor(aqkTensor);
    if (akkTensor) aclDestroyTensor(akkTensor);

    if (qAddr) aclrtFree(qAddr);
    if (kAddr) aclrtFree(kAddr);
    if (vAddr) aclrtFree(vAddr);
    if (gAddr) aclrtFree(gAddr);
    if (betaAddr) aclrtFree(betaAddr);
    if (aLogAddr) aclrtFree(aLogAddr);
    if (dtBiasAddr) aclrtFree(dtBiasAddr);
    if (attnOutAddr) aclrtFree(attnOutAddr);
    if (aqkAddr) aclrtFree(aqkAddr);
    if (akkAddr) aclrtFree(akkAddr);
    if (workspaceAddr) aclrtFree(workspaceAddr);

    aclrtDestroyStream(*stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int main() {
    // 1. device/stream initialization
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs.
    int64_t batch = 1;
    int64_t seqLen = 64;
    int64_t headNum = 2;
    int64_t headDim = 128;
    int64_t chunkSize = 64;

    std::vector<int64_t> qkShape = {batch, seqLen, headNum, headDim};       // BSND
    std::vector<int64_t> vShape = {batch, seqLen, headNum, headDim};        // BSND
    std::vector<int64_t> gShape = {batch, seqLen, headNum, headDim};        // BSND
    std::vector<int64_t> betaShape = {batch, seqLen, headNum};              // BSND
    std::vector<int64_t> attnOutShape = {batch, seqLen, headNum, headDim};  // BSND
    std::vector<int64_t> aqkShape = {batch, headNum, seqLen, chunkSize};    // (B, HV, S, C)
    std::vector<int64_t> akkShape = aqkShape;
    std::vector<int64_t> aLogShape = {headNum};                             // (HV,)
    std::vector<int64_t> dtBiasShape = {headNum * headDim};                 // (HV*K,)

    // Use a fixed seed to generate small random numbers as inputs, consistent with the pytest data construction.
    std::mt19937 gen(20260808);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<int16_t> qHostData(GetShapeSize(qkShape));
    std::vector<int16_t> kHostData(GetShapeSize(qkShape));
    std::vector<int16_t> vHostData(GetShapeSize(vShape));
    std::vector<float> gHostData(GetShapeSize(gShape));
    std::vector<float> betaHostData(GetShapeSize(betaShape));
    for (size_t i = 0; i < qHostData.size(); i++) {
        qHostData[i] = FloatToBf16(dist(gen) * 0.05f);
        kHostData[i] = FloatToBf16(dist(gen) * 0.05f);
        vHostData[i] = FloatToBf16(dist(gen) * 0.05f);
        gHostData[i] = dist(gen) * 0.02f - 0.5f;
        betaHostData[i] = Sigmoid(dist(gen));
    }
    std::vector<float> aLogHostData = {-0.5f, 0.5f};
    std::vector<float> dtBiasHostData(GetShapeSize(dtBiasShape));
    for (int64_t i = 0; i < static_cast<int64_t>(dtBiasHostData.size()); i++) {
        dtBiasHostData[i] = -0.1f + 0.2f * static_cast<float>(i) / static_cast<float>(dtBiasHostData.size() - 1);
    }
    std::vector<int16_t> attnOutHostData(GetShapeSize(attnOutShape), 0);
    std::vector<int16_t> aqkHostData(GetShapeSize(aqkShape), 0);
    std::vector<int16_t> akkHostData(GetShapeSize(akkShape), 0);

    // 3. Create the input and output tensors.
    void* qAddr = nullptr;
    void* kAddr = nullptr;
    void* vAddr = nullptr;
    void* gAddr = nullptr;
    void* betaAddr = nullptr;
    void* aLogAddr = nullptr;
    void* dtBiasAddr = nullptr;
    void* attnOutAddr = nullptr;
    void* aqkAddr = nullptr;
    void* akkAddr = nullptr;
    void* workspaceAddr = nullptr;

    aclTensor* qTensor = nullptr;
    aclTensor* kTensor = nullptr;
    aclTensor* vTensor = nullptr;
    aclTensor* gTensor = nullptr;
    aclTensor* betaTensor = nullptr;
    aclTensor* aLogTensor = nullptr;
    aclTensor* dtBiasTensor = nullptr;
    aclTensor* attnOutTensor = nullptr;
    aclTensor* aqkTensor = nullptr;
    aclTensor* akkTensor = nullptr;

    ret = CreateAclTensor(qHostData, qkShape, &qAddr, ACL_BF16, &qTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("create q tensor failed\n");
              FreeResource(qTensor, kTensor, vTensor, gTensor, betaTensor, aLogTensor, dtBiasTensor,
                           attnOutTensor, aqkTensor, akkTensor, qAddr, kAddr, vAddr, gAddr, betaAddr,
                           aLogAddr, dtBiasAddr, attnOutAddr, aqkAddr, akkAddr, workspaceAddr, deviceId, &stream);
              return ret);
    ret = CreateAclTensor(kHostData, qkShape, &kAddr, ACL_BF16, &kTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("create k tensor failed\n");
              FreeResource(qTensor, kTensor, vTensor, gTensor, betaTensor, aLogTensor, dtBiasTensor,
                           attnOutTensor, aqkTensor, akkTensor, qAddr, kAddr, vAddr, gAddr, betaAddr,
                           aLogAddr, dtBiasAddr, attnOutAddr, aqkAddr, akkAddr, workspaceAddr, deviceId, &stream);
              return ret);
    ret = CreateAclTensor(vHostData, vShape, &vAddr, ACL_BF16, &vTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("create v tensor failed\n");
              FreeResource(qTensor, kTensor, vTensor, gTensor, betaTensor, aLogTensor, dtBiasTensor,
                           attnOutTensor, aqkTensor, akkTensor, qAddr, kAddr, vAddr, gAddr, betaAddr,
                           aLogAddr, dtBiasAddr, attnOutAddr, aqkAddr, akkAddr, workspaceAddr, deviceId, &stream);
              return ret);
    ret = CreateAclTensor(gHostData, gShape, &gAddr, ACL_FLOAT, &gTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("create g tensor failed\n");
              FreeResource(qTensor, kTensor, vTensor, gTensor, betaTensor, aLogTensor, dtBiasTensor,
                           attnOutTensor, aqkTensor, akkTensor, qAddr, kAddr, vAddr, gAddr, betaAddr,
                           aLogAddr, dtBiasAddr, attnOutAddr, aqkAddr, akkAddr, workspaceAddr, deviceId, &stream);
              return ret);
    ret = CreateAclTensor(betaHostData, betaShape, &betaAddr, ACL_FLOAT, &betaTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("create beta tensor failed\n");
              FreeResource(qTensor, kTensor, vTensor, gTensor, betaTensor, aLogTensor, dtBiasTensor,
                           attnOutTensor, aqkTensor, akkTensor, qAddr, kAddr, vAddr, gAddr, betaAddr,
                           aLogAddr, dtBiasAddr, attnOutAddr, aqkAddr, akkAddr, workspaceAddr, deviceId, &stream);
              return ret);
    ret = CreateAclTensor(aLogHostData, aLogShape, &aLogAddr, ACL_FLOAT, &aLogTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("create aLog tensor failed\n");
              FreeResource(qTensor, kTensor, vTensor, gTensor, betaTensor, aLogTensor, dtBiasTensor,
                           attnOutTensor, aqkTensor, akkTensor, qAddr, kAddr, vAddr, gAddr, betaAddr,
                           aLogAddr, dtBiasAddr, attnOutAddr, aqkAddr, akkAddr, workspaceAddr, deviceId, &stream);
              return ret);
    ret = CreateAclTensor(dtBiasHostData, dtBiasShape, &dtBiasAddr, ACL_FLOAT, &dtBiasTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("create dtBias tensor failed\n");
              FreeResource(qTensor, kTensor, vTensor, gTensor, betaTensor, aLogTensor, dtBiasTensor,
                           attnOutTensor, aqkTensor, akkTensor, qAddr, kAddr, vAddr, gAddr, betaAddr,
                           aLogAddr, dtBiasAddr, attnOutAddr, aqkAddr, akkAddr, workspaceAddr, deviceId, &stream);
              return ret);
    ret = CreateAclTensor(attnOutHostData, attnOutShape, &attnOutAddr, ACL_BF16, &attnOutTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("create attnOut tensor failed\n");
              FreeResource(qTensor, kTensor, vTensor, gTensor, betaTensor, aLogTensor, dtBiasTensor,
                           attnOutTensor, aqkTensor, akkTensor, qAddr, kAddr, vAddr, gAddr, betaAddr,
                           aLogAddr, dtBiasAddr, attnOutAddr, aqkAddr, akkAddr, workspaceAddr, deviceId, &stream);
              return ret);
    ret = CreateAclTensor(aqkHostData, aqkShape, &aqkAddr, ACL_BF16, &aqkTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("create aqk tensor failed\n");
              FreeResource(qTensor, kTensor, vTensor, gTensor, betaTensor, aLogTensor, dtBiasTensor,
                           attnOutTensor, aqkTensor, akkTensor, qAddr, kAddr, vAddr, gAddr, betaAddr,
                           aLogAddr, dtBiasAddr, attnOutAddr, aqkAddr, akkAddr, workspaceAddr, deviceId, &stream);
              return ret);
    ret = CreateAclTensor(akkHostData, akkShape, &akkAddr, ACL_BF16, &akkTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("create akk tensor failed\n");
              FreeResource(qTensor, kTensor, vTensor, gTensor, betaTensor, aLogTensor, dtBiasTensor,
                           attnOutTensor, aqkTensor, akkTensor, qAddr, kAddr, vAddr, gAddr, betaAddr,
                           aLogAddr, dtBiasAddr, attnOutAddr, aqkAddr, akkAddr, workspaceAddr, deviceId, &stream);
              return ret);

    // 4. Call the first-phase API: compute the workspace size.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnChunkKdaFwdGetWorkspaceSize(
        qTensor,
        kTensor,
        vTensor,
        gTensor,
        betaTensor,
        aLogTensor,
        dtBiasTensor,
        nullptr,  // initialStateOptional
        nullptr,  // cuSeqlensOptional
        nullptr,  // chunkIndicesOptional
        "BSND",
        1.0 / std::sqrt(static_cast<double>(headDim)),
        chunkSize,
        true,   // safeGate
        -5.0,   // lowerBound
        true,   // useGateInKernel
        false,  // stateVFirst
        attnOutTensor,
        nullptr,  // finalStateOut
        nullptr,  // gkOut
        aqkTensor,
        akkTensor,
        nullptr,  // wOut
        nullptr,  // uOut
        nullptr,  // qgOut
        nullptr,  // kgOut
        nullptr,  // vNewOut
        nullptr,  // hOut
        &workspaceSize,
        &executor
    );
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("GetWorkspaceSize failed. ERROR: %d\n", ret);
              FreeResource(qTensor, kTensor, vTensor, gTensor, betaTensor, aLogTensor, dtBiasTensor,
                           attnOutTensor, aqkTensor, akkTensor, qAddr, kAddr, vAddr, gAddr, betaAddr,
                           aLogAddr, dtBiasAddr, attnOutAddr, aqkAddr, akkAddr, workspaceAddr, deviceId, &stream);
              return ret);
    CHECK_RET(executor != nullptr, LOG_PRINT("executor is null after GetWorkspaceSize\n");
              FreeResource(qTensor, kTensor, vTensor, gTensor, betaTensor, aLogTensor, dtBiasTensor,
                           attnOutTensor, aqkTensor, akkTensor, qAddr, kAddr, vAddr, gAddr, betaAddr,
                           aLogAddr, dtBiasAddr, attnOutAddr, aqkAddr, akkAddr, workspaceAddr, deviceId, &stream);
              return -1);
    LOG_PRINT("Workspace size required: %lu bytes\n", workspaceSize);

    // 5. Allocate the workspace.
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                  FreeResource(qTensor, kTensor, vTensor, gTensor, betaTensor, aLogTensor, dtBiasTensor,
                               attnOutTensor, aqkTensor, akkTensor, qAddr, kAddr, vAddr, gAddr, betaAddr,
                               aLogAddr, dtBiasAddr, attnOutAddr, aqkAddr, akkAddr, workspaceAddr, deviceId, &stream);
                  return ret);
    }

    // 6. Call the second-phase API: perform the computation.
    LOG_PRINT("Calling aclnnChunkKdaFwd...\n");
    ret = aclnnChunkKdaFwd(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnChunkKdaFwd failed. ERROR: %d\n", ret);
              FreeResource(qTensor, kTensor, vTensor, gTensor, betaTensor, aLogTensor, dtBiasTensor,
                           attnOutTensor, aqkTensor, akkTensor, qAddr, kAddr, vAddr, gAddr, betaAddr,
                           aLogAddr, dtBiasAddr, attnOutAddr, aqkAddr, akkAddr, workspaceAddr, deviceId, &stream);
              return ret);

    // 7. Synchronize the Stream and wait for the task to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
              FreeResource(qTensor, kTensor, vTensor, gTensor, betaTensor, aLogTensor, dtBiasTensor,
                           attnOutTensor, aqkTensor, akkTensor, qAddr, kAddr, vAddr, gAddr, betaAddr,
                           aLogAddr, dtBiasAddr, attnOutAddr, aqkAddr, akkAddr, workspaceAddr, deviceId, &stream);
              return ret);

    // 8. Copy the results back to the Host side for printing.
    ret = aclrtMemcpy(attnOutHostData.data(), attnOutHostData.size() * sizeof(int16_t), attnOutAddr,
                      attnOutHostData.size() * sizeof(int16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed.\n");
              FreeResource(qTensor, kTensor, vTensor, gTensor, betaTensor, aLogTensor, dtBiasTensor,
                           attnOutTensor, aqkTensor, akkTensor, qAddr, kAddr, vAddr, gAddr, betaAddr,
                           aLogAddr, dtBiasAddr, attnOutAddr, aqkAddr, akkAddr, workspaceAddr, deviceId, &stream);
              return ret);

    LOG_PRINT("Execution Success! Output results (first 5 elements of attnOut):\n");
    for (uint64_t i = 0; i < 5 && i < attnOutHostData.size(); i++) {
        LOG_PRINT("  attnOut index %lu: %f\n", i, Bf16ToFloat(attnOutHostData[i]));
    }

    // 9. Release all resources.
    FreeResource(qTensor, kTensor, vTensor, gTensor, betaTensor, aLogTensor, dtBiasTensor,
                 attnOutTensor, aqkTensor, akkTensor, qAddr, kAddr, vAddr, gAddr, betaAddr,
                 aLogAddr, dtBiasAddr, attnOutAddr, aqkAddr, akkAddr, workspaceAddr, deviceId, &stream);

    LOG_PRINT("ChunkKdaFwd Test completed successfully!\n");
    return 0;
}
```
