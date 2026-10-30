# aclnnAlltoAllQuantMatmul

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T12:57:20.621Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
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

- Description: Fuses AlltoAll communication, Permute (ensuring contiguous addresses after communication), Quant, Matmul, and Dequant computations. **Communication first, then computation.** Supports K-C quantization, K-C dynamic quantization, and mx [Quantization Mode](../../../docs/en/context/quant_mode_introduction.md).
- Formula: Assume the shape of input x1 is (BS, H). In the mx quantization scenario, the shape of input x1ScaleOptional is (BS, ceil(H/64), 2), and rankSize is the number of NPU devices.

  <!-- npu="910b" id7 -->
  - <term>Atlas A2 products</term>:

    - K-C Quantization Scenario:

      $$
      commOut = AlltoAll(x1.view(rankSize, BS/rankSize, H)) \\
      permutedOut = commOut.permute(1, 0, 2).view(BS/rankSize, rankSize*H) \\
      output_{quant} = x1 @ x2 \\
      output = output_{quant} \times x1_{scale} \times x2_{scale} \\
      output = output + bias
      $$

    - K-C Dynamic Quantization scenario:

      $$
      commOut = AlltoAll(x1.view(rankSize, BS/rankSize, H)) \\
      permutedOut = commOut.permute(1, 0, 2).view(BS/rankSize, rankSize*H) \\
      x1_{quant}, x1_{scale} = Quant(permutedOut) \\
      output_{quant} = x1_{quant} @ x2 \\
      output = output_{quant} \times x1_{scale} \times x2_{scale} \\
      output = output + bias
      $$

  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT products</term>:

    - K-C Dynamic Quantization scenario:

      $$
      commOut = AlltoAll(x1.view(rankSize, BS/rankSize, H)) \\
      permutedOut = commOut.permute(1, 0, 2).view(BS/rankSize, rankSize*H) \\
      dynQuantX1, dynQuantX1Scale = dynamicQuant(permutedOut) \\
      output = (dynQuantX1@x2 + bias) \times dynQuantX1Scale \times x2Scale
      $$

    - mx quantization scenario:

      $$
      commOut = AlltoAll(x1.view(rankSize, BS/rankSize, H)) \\
      permutedOut = commOut.permute(1, 0, 2).view(BS/rankSize, rankSize*H) \\
      commScale = AlltoAll(x1Scale.view(rankSize, BS/rankSize, ceil(H/64), 2)) \\
      permutedScale = commScale.permute(1, 0, 2, 3).view(BS/rankSize, ceil(H/64)*rankSize, 2) \\
      output = \sum_{0}^{\left \lfloor \frac{k}{blockSize=32} \right \rfloor} (permutedOut @ x2 * (permutedScale * x2Scale)) + bias
      $$

  <!-- end id8 -->

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call "aclnnAlltoAllQuantMatmulGetWorkspaceSize" to obtain the workspace size required for computation and the executor containing the operator computation process. Then call "aclnnAlltoAllQuantMatmul" to execute the computation.

```cpp
aclnnStatus aclnnAlltoAllQuantMatmulGetWorkspaceSize(
  const aclTensor*   x1,
  const aclTensor*   x2,
  const aclTensor*   biasOptional,
  const aclTensor*   x1ScaleOptional,
  const aclTensor*   x2Scale,
  const aclTensor*   commScaleOptional,
  const aclTensor*   x1OffsetOptional,
  const aclTensor*   x2OffsetOptional,
  const char*        group,
  const aclIntArray* alltoAllAxesOptional,
  int64_t            x1QuantMode,
  int64_t            x2QuantMode,
  int64_t            commQuantMode,
  int64_t            commQuantDtype,
  int64_t            x1QuantDtype,
  int64_t            groupSize,
  bool               transposeX1,
  bool               transposeX2,
  const aclTensor*   output,
  const aclTensor*   alltoAllOutOptional,
  uint64_t*          workspaceSize,
  aclOpExecutor**    executor)
```

```cpp
aclnnStatus aclnnAlltoAllQuantMatmul(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnAlltoAllQuantMatmulGetWorkspaceSize

- **Parameters**:

    <table style="table-layout: fixed; width: 1556px"> <colgroup>
    <col style="width: 154px">
    <col style="width: 123px">
    <col style="width: 270px">
    <col style="width: 325px">
    <col style="width: 245px">
    <col style="width: 120px">
    <col style="width: 203px">
    <col style="width: 116px">
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
    <td>x1</td>
    <td>Input</td>
    <td>Left Matrix input of the fused operator, corresponding to x1 in the formula.</td>
    <td>This input undergoes AlltoAll communication and the Permute operation. The result serves as the Left Matrix input of the MatMul computation. The data type restrictions vary by device model. For details, see <a href="#constraints">Constraints</a>.</td>
    <td>FLOAT16, BFLOAT16, FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1, INT4</td>
    <td>ND</td>
    <td>2 dimensions, shape (BS, H)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>x2</td>
    <td>Input</td>
    <td>Right Matrix input of the fused operator, also the Right Matrix of the MatMul computation, corresponding to x2 in the formula.</td>
    <td>Serves as the Right Matrix input for the MatMul computation. Data type and non-contiguous tensor restrictions vary by product model. For details, see <a href="#constraints">Constraints</a>.</td>
    <td>FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1, INT8, INT4</td>
    <td>ND</td>
    <td>2D, shape (H*rankSize, N)</td>
    <td>Support varies by product model. For details, see <a href="#constraints">Constraints</a>.</td>
    </tr>
    <tr>
    <td>biasOptional</td>
    <td>Input</td>
    <td>Optional Input. Bias accumulated after the matrix multiplication, corresponding to bias in the formula.</td>
    <td>Data type restrictions vary by product model. For details, see <a href="#constraints">Constraints</a>.</td>
    <td>FLOAT16, BFLOAT16, FLOAT32</td>
    <td>ND</td>
    <td>1D, shape (N)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>x1ScaleOptional</td>
    <td>Input</td>
    <td>Optional input, the quantization coefficient of the Left Matrix.</td>
    <td>Must be configured in K-C quantization and mx quantization scenarios. In the K-C Dynamic Quantization scenario, x1ScaleOptional can be passed in as smoothScale. In this case, the data type must be the same as x1.</td>
    <td>FLOAT32, FLOAT16, BFLOAT16, FLOAT8_E8M0</td>
    <td>ND</td>
    <td>1D/3D. In K-C quantization scenarios, the shape is (BS). In K-C Dynamic Quantization scenarios, the shape is (H*rankSize). In mx quantization scenarios, the shape is (BS, ceil(H/64), 2).</td>
    <td>x</td>
    </tr>
    <tr>
    <td>x2Scale</td>
    <td>Input</td>
    <td>Quantization scale of the right matrix.</td>
    <td>x2Scale in the corresponding formula.</td>
    <td>FLOAT32, FLOAT8_E8M0</td>
    <td>ND</td>
    <td>1D/3D. The shape is (N) for K-C quantization and K-C dynamic quantization. The shape is (N, ceil(H*rankSize/64), 2) for the mx quantization scenario</td>
    <td>x</td>
    </tr>
    <tr>
    <td>commScaleOptional</td>
    <td>Input</td>
    <td>Optional input, quantization coefficients for low bit communication.</td>
    <td>Reserved parameter. Low bit communication is not supported yet.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>x1OffsetOptional</td>
    <td>Input</td>
    <td>Optional input, quantization offset of the left matrix.</td>
    <td>Reserved parameter, currently not supported.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>x2OffsetOptional</td>
    <td>Input</td>
    <td>Optional input, the quantization offset of the right matrix.</td>
    <td>Reserved parameter, currently not supported.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>alltoAllAxesOptional</td>
    <td>Input</td>
    <td>Optional input. Direction of AlltoAll and Permute data exchange.</td>
    <td>Only empty or [-2, -1] is supported. If empty is passed, [-2, -1] is used by default, indicating that the input is converted from (BS, H) to (BS/rankSize, rankSize*H).</td>
    <td>aclIntArray* (element type INT64)</td>
    <td>-</td>
    <td>1 dimension, shape (2)</td>
    <td>-</td>
    </tr>
    <tr>
    <td>group</td>
    <td>Input</td>
    <td>String on the Host identifying the column group, that is, the communicator name. Obtain commName through the HCCL API HcclGetCommName and pass it as this parameter.</td>
    <td>String length requirement: (0, 128).</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>x1QuantMode</td>
    <td>Input</td>
    <td>Quantization mode of the left matrix.</td>
    <td>The value is subject to different restrictions depending on the device model. For details, see <a href="#constraints">Constraints</a>.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>x2QuantMode</td>
    <td>Input</td>
    <td>Quantization mode of the right matrix.</td>
    <td>Value restrictions differ by device model. For details, see <a href="#constraints">Constraints</a>.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>commQuantMode</td>
    <td>Input</td>
    <td>Quantization mode of low-bit communication.</td>
    <td>Reserved parameter. Currently, only 0 is supported, indicating no quantization.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>commQuantDtype</td>
    <td>Input</td>
    <td>Quantization type for Low Bit communication.</td>
    <td>Reserved Parameter. Currently, only -1 can be configured, indicating ACL_DT_UNDEFINED.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>x1QuantDtype</td>
    <td>Input</td>
    <td>Quantization type of the left matrix of QuantMatmul.</td>
    <td>After the AlltoAll communication and Permute operations, the result is quantized based on this parameter configuration and used as the left matrix input of the MatMul computation. Value restrictions vary by product model. For details, see <a href="#constraints">Constraints</a>.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>groupSize</td>
    <td>Input</td>
    <td>Quantization group sizes along the three directions of the Matmul computation, composed of the three values groupSizeM, groupSizeN, and groupSizeK. Each value occupies 16 bits, and the three values together occupy the lower 48 bits of the int64_t groupSize (the upper 16 bits of groupSize are invalid).</td>
    <td><ul><li>In the mx quantization scenario, only [groupSizeM, groupSizeN, groupSizeK] = [1, 1, 32] is supported. For the corresponding groupSize value, see <a href="#constraints">Constraints</a>. In other quantization scenarios, the default value is 0 and does not take effect.</li><li>Parameter auto-derivation is supported. If one or more of groupSizeM, groupSizeN, and groupSizeK decomposed by the formula are 0, the operator automatically derives these values. For details, see <a href="#constraints">Constraints</a>.</li></ul></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>transposeX1</td>
    <td>Input</td>
    <td>Indicates whether the left matrix is transposed.</td>
    <td>Setting this parameter to True is not supported yet.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>transposeX2</td>
    <td>Input</td>
    <td>Indicates whether the right matrix is transposed.</td>
    <td>When set to True, the shape of the right matrix is (N, rankSize*H). In mx quantization mode, this parameter must be set to True.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>Output</td>
    <td>Output</td>
    <td>The final computation result.</td>
    <td>Data types are restricted differently depending on the product model. For details, see <a href="#constraints">Constraints</a>.</td>
    <td>FLOAT16, BFLOAT16, FLOAT32</td>
    <td>ND</td>
    <td>2D, shape (BS/rankSize, N)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>alltoAllOutOptional</td>
    <td>Output</td>
    <td>Receives the content after AlltoAll and Permute. The data type is the same as that of input x1.</td>
    <td>If nullptr is passed, no communication output is returned.</td>
    <td>FLOAT16, BFLOAT16, FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1, INT4</td>
    <td>ND</td>
    <td>2 dimensions, shape (BS/rankSize, rankSize*H)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>Output</td>
    <td>Returns the workspace size to be requested on the Device side.</td>
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
    </tbody></table>

  The enumeration values of x1QuantMode, x2QuantMode, and commQuantMode correspond to the [quantization mode](../../../docs/en/context/quant_mode_introduction.md) as follows:
  - 0: no quantization
  - 1: pertensor
  - 2: perchannel
  - 3: pertoken
  - 4: pergroup
  - 5: perblock
  - 6: mx quantization
  - 7: pertoken dynamic quantization

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1030px"><colgroup>
    <col style="width:250px">
    <col style="width:130px">
    <col style="width:650px">
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
      <td>A required input or output tensor parameter is a null pointer.</td>
    </tr>
    <tr>
        <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="7">161002</td>
        <td>The data types of the input and output are out of the supported range.</td>
    </tr>
    <tr>
        <td>The input tensor is an empty tensor.</td>
    </tr>
    <tr>
        <td>alltoAllAxesOptional is invalid.</td>
    </tr>
    <tr>
        <td>transposeX1 is true.</td>
    </tr>
    <tr>
        <td>The communicator length is invalid.</td>
    </tr>
    <tr>
        <td>The dimensions of the input and output tensors are invalid.</td>
    </tr>
    <tr>
        <td>The format of the input and output tensors is a private format.</td>
    </tr>
      </tbody>
  </table>

## aclnnAlltoAllQuantMatmul

- **Parameters:**

    <table style="table-layout: fixed; width: 1150px"><colgroup>
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
        <td>Workspace memory address allocated on the device.</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Workspace size allocated on the device, obtained by calling the first-phase API aclnnAlltoAllQuantMatmulGetWorkspaceSize.</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Executor that contains the operator computation flow.</td>
    </tr>
    <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream for executing tasks.</td>
    </tr>
    </tbody>
    </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- aclnnAlltoAllQuantMatmul supports deterministic computation by default.
- Number of NPU cards (rankSize), with different restrictions based on the device model:

  <!-- npu="910b" id9 -->
  - <term>Atlas A2 products</term>: Supports 2, 4, or 8 cards.
  <!-- end id9 -->
  <!-- npu="950" id10 -->
  - <term>Ascend 950PR&950DT products</term>: Supports 2, 4, 8, or 16 cards.
  <!-- end id10 -->

- The variable BS used in the shape in Parameters must be divisible by rankSize.
- The values of BS and N must not exceed 2147483647 (INT32_MAX). The value of BS must not be less than 2, and the value of N must not be less than 1.
- Empty tensors are not supported.
- Support for non-contiguous tensors varies by device model:

  <!-- npu="910b" id11 -->
  - <term>Atlas A2 products</term>: No non-contiguous tensors are supported.
  <!-- end id11 -->
  <!-- npu="950" id12 -->
  - <term>Ascend 950PR&950DT products</term>: Only x2 is supported as a non-contiguous tensor. Other non-contiguous tensors are not supported.
  <!-- end id12 -->

- The input x1, x2, x2Scale, and output must not be null pointers, and

  <!-- npu="950" id13 -->
  - <term>Ascend 950PR&950DT products</term>: In the per-token dynamic quantization scenario where x1QuantMode is set, x1ScaleOptional is not supported.
  <!-- end id13 -->

- Constraints on groupSize:
  - The groupSize value is valid only when both the x1ScaleOptional and x2Scale inputs have two or more dimensions. In other cases, pass 0.
  - The passed-in groupSize is internally decomposed into groupSizeM, groupSizeN, and groupSizeK using the following formula. If one or more of them is 0, groupSizeM, groupSizeN, and groupSizeK are recalculated based on the shapes of the x1/x2/x1ScaleOptional/x2Scale inputs for computation. Principle: If groupSizeM=0, the quantization group value along the m dimension is inferred by the API. The inference formula is groupSizeM = m / scaleM (m must be divisible by scaleM), where m is the same as the m in the x1 shape, and scaleM is the same as the m in the x1ScaleOptional shape. The same applies to the k and n dimensions.
    $$
    groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
    $$
  - If the input x and scale satisfy divisibility along all dimensions, and the automatically deduced groupSizeM, groupSizeN, and groupSizeK are [1,1,32], groupSize in the mx quantization scenario supports the following values:

    | groupSize | Based on the Calculation Formula [gsM,gsN,gsK] | Based on Automatic Deduction [gsM,gsN,gsK] |
    | :------: | :------: | :------: |
    | 4295032864 | [1,1,32] | - |
    | 0 | [0,0,0] | [1,1,32] |
    | 32 | [0,0,32] | [1,1,32] |
    | 65536 | [0,1,0] | [1,1,32] |
    | 65568 | [0,1,32] | [1,1,32] |
    | 4294967296 | [1,0,0] | [1,1,32] |
    | 4294967328 | [1,0,32] | [1,1,32] |
    | 4295032832 | [1,1,0] | [1,1,32] |

- The data types, data dimensions, and quantization modes of the input and output of this operator have different restrictions depending on the device model:

  <!-- npu="910b" id14 -->
  - <term>Atlas A2 products</term>:
    - Quantization Mode:
      - Currently supports per-token quantization and per-token dynamic quantization for the left matrix, with x1QuantMode=3 or 7; and per-channel quantization for the right matrix, with x2QuantMode=2.
    - Type constraints:
      - The data types of x1 and alltoAllOutOptional must be consistent.
      - If x1, x2, or alltoallout is of the int32 type, it is treated as eight packed int4 values and will be reinterpreted as int4.
      - In A16W8 and A16W4 modes, for the smoothQuant scenario, the data types of x1ScaleOptional and x1 must be consistent.
      - For A16W8, the supported data type combinations of x1, x2, biasOptional, and output are:

        | x1 | x2 | biasOptional | output |
        | :------: | :------: | :------: | :------: |
        | FLOAT16 | INT8 | FLOAT16 | FLOAT16 |
        | FLOAT16 | INT8 | FLOAT32 | FLOAT16 |
        | BFLOAT16 | INT8 | BFLOAT16 | BFLOAT16 |
        | BFLOAT16 | INT8 | FLOAT32 | BFLOAT16 |

      - For A16W4, the supported data type combinations of x1, x2, biasOptional, and output are:

        | x1 | x2 | biasOptional | output |
        | :------: | :------: | :------: | :------: |
        | FLOAT16 | INT4 | FLOAT16 | FLOAT16 |
        | FLOAT16 | INT4 | FLOAT32 | FLOAT16 |
        | BFLOAT16 | INT4 | BFLOAT16 | BFLOAT16 |
        | BFLOAT16 | INT4 | FLOAT32 | BFLOAT16 |

      - For A4W4, x1ScaleOptional supports only FLOAT32. The supported data type combinations of x1, x2, biasOptional, and output are:

        | x1 | x2 | biasOptional | output |
        | :------: | :------: | :------: | :------: |
        | INT4 | INT4 | FLOAT16 | FLOAT16 |
        | INT4 | INT4 | FLOAT32 | FLOAT16 |
        | INT4 | INT4 | BFLOAT16 | BFLOAT16 |
        | INT4 | INT4 | FLOAT32 | BFLOAT16 |

    - Dimension Constraints:
      - For A16W8, rankSize *H must be divisible by 16. Value range of rankSize* H: [1, 35000].
      - For A16W4, rankSize *H must be divisible by 16. N must be an even number. Value range of rankSize* H: [1, 35000].
      - For A4W4, H and N must be even numbers. Value range of rankSize * H: [1, 35000].
  <!-- end id14 -->
  <!-- npu="950" id15 -->
  - <term>Ascend 950PR&950DT products</term>:
    - Quantization Mode:
      - Currently supported: K-C dynamic quantization, left matrix perToken dynamic quantization, x1QuantMode=7, right matrix perChannel quantization, x2QuantMode=2; mx quantization, left matrix mx quantization, x1QuantMode=6, right matrix mx quantization, x2QuantMode=6.
    - Type constraints:
      - The data types of x1 and alltoAllOutOptional must be consistent.
      - x1QuantDtype takes effect in the K-C Dynamic Quantization scenario, supporting 35 (indicating aclDataType.ACL_FLOAT8_E5M2) and 36 (indicating aclDataType.ACL_FLOAT8_E4M3FN). It does not take effect in other quantization scenarios.
      - biasOptional can be empty.
      - The supported input/output data type combinations are:
        - K-C Dynamic Quantization:

          | x1 | x2 | biasOptional | output | x1QuantMode | x2QuantMode | x1ScaleOptional | x2Scale |
          | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
          | FLOAT16 | FLOAT8_E4M3FN | FLOAT32 | FLOAT16 | 7 | 2 | - | FLOAT32 |
          | FLOAT16 | FLOAT8_E4M3FN | FLOAT32 | BFLOAT16 | 7 | 2 | - | FLOAT32 |
          | FLOAT16 | FLOAT8_E4M3FN | FLOAT32 | FLOAT32 | 7 | 2 | - | FLOAT32 |
          | FLOAT16 | FLOAT8_E5M2 | FLOAT32 | FLOAT16 | 7 | 2 | - | FLOAT32 |
          | FLOAT16 | FLOAT8_E5M2 | FLOAT32 | BFLOAT16 | 7 | 2 | - | FLOAT32 |
          | FLOAT16 | FLOAT8_E5M2 | FLOAT32 | FLOAT32 | 7 | 2 | - | FLOAT32 |
          | BFLOAT16 | FLOAT8_E4M3FN | FLOAT32 | FLOAT16 | 7 | 2 | - | FLOAT32 |
          | BFLOAT16 | FLOAT8_E4M3FN | FLOAT32 | BFLOAT16 | 7 | 2 | - | FLOAT32 |
          | BFLOAT16 | FLOAT8_E4M3FN | FLOAT32 | FLOAT32 | 7 | 2 | - | FLOAT32 |
          | BFLOAT16 | FLOAT8_E5M2 | FLOAT32 | FLOAT16 | 7 | 2 | - | FLOAT32 |
          | BFLOAT16 | FLOAT8_E5M2 | FLOAT32 | BFLOAT16 | 7 | 2 | - | FLOAT32 |
          | BFLOAT16 | FLOAT8_E5M2 | FLOAT32 | FLOAT32 | 7 | 2 | - | FLOAT32 |

        - mx Quantization:

          | x1 | x2 | biasOptional | output | x1QuantMode | x2QuantMode | x1ScaleOptional | x2Scale |
          | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
          | FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT32 | FLOAT16 | 6 | 6 | FLOAT8_E8M0 | FLOAT8_E8M0 |
          | FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT32 | BFLOAT16 | 6 | 6 | FLOAT8_E8M0 | FLOAT8_E8M0 |
          | FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT32 | FLOAT32 | 6 | 6 | FLOAT8_E8M0 | FLOAT8_E8M0 |
          | FLOAT8_E4M3FN | FLOAT8_E5M2 | FLOAT32 | FLOAT16 | 6 | 6 | FLOAT8_E8M0 | FLOAT8_E8M0 |
          | FLOAT8_E4M3FN | FLOAT8_E5M2 | FLOAT32 | BFLOAT16 | 6 | 6 | FLOAT8_E8M0 | FLOAT8_E8M0 |
          | FLOAT8_E4M3FN | FLOAT8_E5M2 | FLOAT32 | FLOAT32 | 6 | 6 | FLOAT8_E8M0 | FLOAT8_E8M0 |
          | FLOAT8_E5M2 | FLOAT8_E4M3FN | FLOAT32 | FLOAT16 | 6 | 6 | FLOAT8_E8M0 | FLOAT8_E8M0 |
          | FLOAT8_E5M2 | FLOAT8_E4M3FN | FLOAT32 | BFLOAT16 | 6 | 6 | FLOAT8_E8M0 | FLOAT8_E8M0 |
          | FLOAT8_E5M2 | FLOAT8_E4M3FN | FLOAT32 | FLOAT32 | 6 | 6 | FLOAT8_E8M0 | FLOAT8_E8M0 |
          | FLOAT8_E5M2 | FLOAT8_E5M2 | FLOAT32 | FLOAT16 | 6 | 6 | FLOAT8_E8M0 | FLOAT8_E8M0 |
          | FLOAT8_E5M2 | FLOAT8_E5M2 | FLOAT32 | BFLOAT16 | 6 | 6 | FLOAT8_E8M0 | FLOAT8_E8M0 |
          | FLOAT8_E5M2 | FLOAT8_E5M2 | FLOAT32 | FLOAT32 | 6 | 6 | FLOAT8_E8M0 | FLOAT8_E8M0 |
          | FLOAT4_E2M1 | FLOAT4_E2M1 | FLOAT32 | FLOAT16 | 6 | 6 | FLOAT8_E8M0 | FLOAT8_E8M0 |
          | FLOAT4_E2M1 | FLOAT4_E2M1 | FLOAT32 | BFLOAT16 | 6 | 6 | FLOAT8_E8M0 | FLOAT8_E8M0 |
          | FLOAT4_E2M1 | FLOAT4_E2M1 | FLOAT32 | FLOAT32 | 6 | 6 | FLOAT8_E8M0 | FLOAT8_E8M0 |

    - Dimension Constraints:
      - The range of rankSize * H only supports [1, 65535].
      - In the mx quantization scenario, H must be divisible by 64.
      - In the mx quantization scenario, x2 must be transposed, with shape (H*rankSize, N) and transposeX2 set to True.
  <!-- end id15 -->

- The communication-computation fusion operator does not support concurrent calls, and different communication-computation fusion operators also do not support concurrent calls.
- Cross-supernode communication is not supported. Only intra-supernode communication is supported.
- Communication constraints:

   <!-- npu="910b" id16 -->
   - <term>Atlas A2 products</term>: MTE communication is supported, and the communication buffer size must be greater than or equal to 200 MB.
   <!-- end id16 -->
   <!-- npu="950" id17 -->
   - <term>Ascend 950PR&950DT products</term>: AI_CPU communication is supported.

   <!-- end id17 -->

## Example

The following is sample code for reference only. For details about compilation and execution, see Compile and Run Samples.

NOTE: The sample code calls some APIs of the HCCL collective communication library, including HcclGetCommName, HcclCommInitAll, and HcclCommDestroy. For details, see [*HCCL API (C)*](https://hiascend.com/en/document/redirect/CannCommunityHcclCppApi).

<!-- npu="910b" id18 -->
- <term>Atlas A2 products</term>:

    ```cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <cstring>
    #include <vector>
    #include <acl/acl.h>
    #include <hccl/hccl.h>
    #include "aclnn/opdev/fp16_t.h"
    #include "aclnnop/aclnn_allto_all_quant_matmul.h"

    int ndev = 2;

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

    int64_t GetShapeSize(const std::vector<int64_t> &shape) {
        int64_t shapeSize = 1;
        for (auto i: shape) {
            shapeSize *= i;
        }
        return shapeSize;
    }

    template<typename T>
    int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                        aclDataType dataType, aclTensor **tensor) {
        auto size = GetShapeSize(shape) * sizeof(T);
        // Call aclrtMalloc to allocate memory on the device.
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
        // Call aclrtMemcpy to copy data from the host to the device.
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

    struct Args {
        uint32_t rankId;
        HcclComm hcclComm;
        aclrtStream stream;
        aclrtContext context;
    };

    int launchOneThreadAlltoAllQuantMatmul(Args &args)
    {
        int ret;
        ret = aclrtSetCurrentContext(args.context);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
        char hcom_name[128] = {0};
        ret = HcclGetCommName(args.hcclComm, hcom_name);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d \n", ret); return -1);
        LOG_PRINT("[INFO] rank %d hcom: %s stream: %p, context : %p\n", args.rankId, hcom_name, args.stream,
                args.context);

        std::vector<int64_t> x1Shape = {32, 64};
        std::vector<int64_t> x2Shape = {64 * ndev, 128}; // ndev = 2; the shape of x2Shape remains unchanged before and after transposition.
        std::vector<int64_t> biasShape = {128};
        std::vector<int64_t> x2ScaleShape = {128};
        std::vector<int64_t> outShape = {32 / ndev, 128};
        std::vector<int64_t> allToAllOutShape = {32 / ndev, 64 * ndev};
        void *x1DeviceAddr = nullptr;
        void *x2DeviceAddr = nullptr;
        void *biasDeviceAddr = nullptr;
        void *x2ScaleDeviceAddr = nullptr;
        void *outDeviceAddr = nullptr;
        void *allToAllOutDeviceAddr = nullptr;
        aclTensor *x1 = nullptr;
        aclTensor *x2 = nullptr;
        aclTensor *bias = nullptr;
        aclTensor *x1ScaleOptional = nullptr;
        aclTensor *x2Scale = nullptr;
        aclTensor* commScaleOptional = nullptr;
        aclTensor* x1OffsetOptional = nullptr;
        aclTensor* x2OffsetOptional = nullptr;
        aclTensor *out = nullptr;
        aclTensor *allToAllOut = nullptr;

        int64_t a2aAxes[2] = {-2, -1};
        aclIntArray* alltoAllAxesOptional = aclCreateIntArray(a2aAxes, static_cast<uint64_t>(2));
        uint64_t workspaceSize = 0;
        int64_t x1QuantMode = 3;
        int64_t x2QuantMode = 2;
        int64_t commQuantMode = 0;
        int64_t commQuantDtype = -1;
        int64_t x1QuantDtype = 2;
        int64_t groupSize = 0;
        aclOpExecutor *executor;
        void *workspaceAddr = nullptr;

        long long x1ShapeSize = GetShapeSize(x1Shape);
        long long x2ShapeSize = GetShapeSize(x2Shape);
        long long biasShapeSize = GetShapeSize(biasShape);
        long long x2ScaleShapeSize = GetShapeSize(x2ScaleShape);
        long long outShapeSize = GetShapeSize(outShape);
        long long allToAllOutShapeSize = GetShapeSize(allToAllOutShape);
        std::vector<op::fp16_t> x1HostData(x1ShapeSize, 1);
        std::vector<int8_t> x2HostData(x2ShapeSize, 1);
        std::vector<op::fp16_t> biasHostData(biasShapeSize, 1);
        std::vector<float> x2ScaleHostData(x2ScaleShapeSize, 1);
        std::vector<op::fp16_t> outHostData(outShapeSize, 0);
        std::vector<op::fp16_t> allToAllOutHostData(allToAllOutShapeSize, 0);

        // Create a tensor.
        ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT16, &bias);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x2Scale);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(allToAllOutHostData, allToAllOutShape, &allToAllOutDeviceAddr, aclDataType::ACL_FLOAT16, &allToAllOut);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Call the first-phase API.
        ret = aclnnAlltoAllQuantMatmulGetWorkspaceSize(x1, x2, bias, x1ScaleOptional, x2Scale, commScaleOptional, x1OffsetOptional, x2OffsetOptional,
                                                hcom_name, alltoAllAxesOptional, x1QuantMode, x2QuantMode, commQuantMode, commQuantDtype, x1QuantDtype,
                                                groupSize, false, true,
                                                out, allToAllOut, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclnnAlltoAllQuantMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
        // Allocate device memory using the workspaceSize computed by the first-phase API.
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        // Call the second-phase API.
        ret = aclnnAlltoAllQuantMatmul(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAlltoAllQuantMatmul failed. ERROR: %d\n", ret); return ret);
        // (Boilerplate) Synchronously wait for the task to complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
        LOG_PRINT("device%d aclnnAlltoAllQuantMatmul execute success \n", args.rankId);
        // Release device resources. Modify this part based on the definition of the specific API.
        if (x1 != nullptr) {
            aclDestroyTensor(x1);
        }
        if (x2 != nullptr) {
            aclDestroyTensor(x2);
        }
        if (bias != nullptr) {
            aclDestroyTensor(bias);
        }
        if (x2Scale != nullptr) {
            aclDestroyTensor(x2Scale);
        }
        if (out != nullptr) {
            aclDestroyTensor(out);
        }
        if (allToAllOut != nullptr) {
            aclDestroyTensor(allToAllOut);
        }
        if (x1DeviceAddr != nullptr) {
            aclrtFree(x1DeviceAddr);
        }
        if (x2DeviceAddr != nullptr) {
            aclrtFree(x2DeviceAddr);
        }
        if (biasDeviceAddr != nullptr) {
            aclrtFree(biasDeviceAddr);
        }
        if (outDeviceAddr != nullptr) {
            aclrtFree(outDeviceAddr);
        }
        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }
        aclrtDestroyStream(args.stream);
        HcclCommDestroy(args.hcclComm);
        aclrtDestroyContext(args.context);
        aclrtResetDevice(args.rankId);
        return 0;
    }

    int main(int argc, char *argv[])
    {
        // This sample is implemented based on Atlas A2 and must run on Atlas A2.
        int ret = aclInit(nullptr);
        int32_t devices[ndev];
        for (int i = 0; i < ndev; i++) {
            devices[i] = i;
        }
        HcclComm comms[128];
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
        // Initialize the communicator.
        for (int i = 0; i < ndev; i++) {
            ret = aclrtSetDevice(devices[i]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
        }
        ret = HcclCommInitAll(ndev, devices, comms);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("HcclCommInitAll failed. ERROR: %d\n", ret); return ret);
        Args args[ndev];
        aclrtStream stream[ndev];
        aclrtContext context[ndev];
        for (uint32_t rankId = 0; rankId < ndev; rankId++) {
            ret = aclrtSetDevice(rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
            ret = aclrtCreateContext(&context[rankId], rankId);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
            ret = aclrtCreateStream(&stream[rankId]);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
        }
        // Start multiple threads.
        std::vector<std::unique_ptr<std::thread>> threads(ndev);
        for (uint32_t rankId = 0; rankId < ndev; rankId++) {
            args[rankId].rankId = rankId;
            args[rankId].hcclComm = comms[rankId];
            args[rankId].stream = stream[rankId];
            args[rankId].context = context[rankId];
            threads[rankId].reset(new(std::nothrow) std::thread(&launchOneThreadAlltoAllQuantMatmul, std::ref(args[rankId])));
        }
        for (uint32_t rankId = 0; rankId < ndev; rankId++) {
            threads[rankId]->join();
        }
        aclFinalize();
        return 0;
    }
    ```

<!-- end id18 -->
<!-- npu="950" id19 -->
- <term>Ascend 950PR&950DT products</term>:

    ```cpp
    #include <thread>
    #include <iostream>
    #include <string>
    #include <cstring>
    #include <vector>
    #include <acl/acl.h>
    #include <hccl/hccl.h>
    #include "aclnn/opdev/fp16_t.h"
    #include "aclnnop/aclnn_allto_all_quant_matmul.h"

    int ndev = 2;

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

    int64_t GetShapeSize(const std::vector<int64_t> &shape) {
    int64_t shapeSize = 1;
    for (auto i: shape) {
    shapeSize *= i;
    }
    return shapeSize;
    }

    template<typename T>
    int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclTensor **tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate device memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host data to device memory.
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

    struct Args {
    uint32_t rankId;
    HcclComm hcclComm;
    aclrtStream stream;
    aclrtContext context;
    };

    int launchOneThreadAlltoAllQuantMatmul(Args &args)
    {
    int ret;
    ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
    char hcom_name[128] = {0};
    ret = HcclGetCommName(args.hcclComm, hcom_name);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed. ret = %d \n", ret); return -1);
    LOG_PRINT("[INFO] rank %d hcom: %s stream: %p, context : %p\n", args.rankId, hcom_name, args.stream,
    args.context);

        std::vector<int64_t> x1Shape = {32, 64};
        std::vector<int64_t> x2Shape = {64 * ndev, 128};
        std::vector<int64_t> biasShape = {128};
        std::vector<int64_t> x2ScaleShape = {128};
        std::vector<int64_t> outShape = {32 / ndev, 128};
        std::vector<int64_t> allToAllOutShape = {32 / ndev, 64 * ndev};
        void *x1DeviceAddr = nullptr;
        void *x2DeviceAddr = nullptr;
        void *biasDeviceAddr = nullptr;
        void *x2ScaleDeviceAddr = nullptr;
        void *outDeviceAddr = nullptr;
        void *allToAllOutDeviceAddr = nullptr;
        aclTensor *x1 = nullptr;
        aclTensor *x2 = nullptr;
        aclTensor *bias = nullptr;
        aclTensor *x1ScaleOptional = nullptr;
        aclTensor *x2Scale = nullptr;
        aclTensor* commScaleOptional = nullptr;
        aclTensor* x1OffsetOptional = nullptr;
        aclTensor* x2OffsetOptional = nullptr;
        aclTensor *out = nullptr;
        aclTensor *allToAllOut = nullptr;

        int64_t a2aAxes[2] = {-2, -1};
        aclIntArray* alltoAllAxesOptional = aclCreateIntArray(a2aAxes, static_cast<uint64_t>(2));
        uint64_t workspaceSize = 0;
        int64_t x1QuantMode = 7;
        int64_t x2QuantMode = 2;
        int64_t commQuantMode = 0;
        int64_t commQuantDtype = -1;
        int64_t x1QuantDtype = 35;
        int64_t groupSize = 0;
        aclOpExecutor *executor;
        void *workspaceAddr = nullptr;

        long long x1ShapeSize = GetShapeSize(x1Shape);
        long long x2ShapeSize = GetShapeSize(x2Shape);
        long long biasShapeSize = GetShapeSize(biasShape);
        long long x2ScaleShapeSize = GetShapeSize(x2ScaleShape);
        long long outShapeSize = GetShapeSize(outShape);
        long long allToAllOutShapeSize = GetShapeSize(allToAllOutShape);
        std::vector<op::fp16_t> x1HostData(x1ShapeSize, 1);
        std::vector<int8_t> x2HostData(x2ShapeSize, 1);
        std::vector<float> biasHostData(biasShapeSize, 1);
        std::vector<float> x2ScaleHostData(x2ScaleShapeSize, 1);
        std::vector<float> outHostData(outShapeSize, 0);
        std::vector<op::fp16_t> allToAllOutHostData(allToAllOutShapeSize, 0);
        // Create a tensor.
        ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT8_E5M2, &x2);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x2Scale);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(allToAllOutHostData, allToAllOutShape, &allToAllOutDeviceAddr, aclDataType::ACL_FLOAT16, &allToAllOut);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Call the first-phase API.
        ret = aclnnAlltoAllQuantMatmulGetWorkspaceSize(x1, x2, bias, x1ScaleOptional, x2Scale, commScaleOptional, x1OffsetOptional, x2OffsetOptional,
                                                hcom_name, alltoAllAxesOptional, x1QuantMode, x2QuantMode, commQuantMode, commQuantDtype, x1QuantDtype,
                                                groupSize, false, false,
                                                out, allToAllOut, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclnnAlltoAllQuantMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
        // Allocate device memory based on the workspaceSize calculated by the first-phase API.
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        // Call the second-phase API.
        ret = aclnnAlltoAllQuantMatmul(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAlltoAllQuantMatmul failed. ERROR: %d\n", ret); return ret);
        // (boilerplate) Synchronize and wait for task execution to complete.
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
        LOG_PRINT("device%d aclnnAlltoAllQuantMatmul execute success \n", args.rankId);
        // Release device resources. Modify it based on the specific API definition.
        if (x1 != nullptr) {
            aclDestroyTensor(x1);
        }
        if (x2 != nullptr) {
            aclDestroyTensor(x2);
        }
        if (bias != nullptr) {
            aclDestroyTensor(bias);
        }
        if (x2Scale != nullptr) {
            aclDestroyTensor(x2Scale);
        }
        if (out != nullptr) {
            aclDestroyTensor(out);
        }
        if (allToAllOut != nullptr) {
            aclDestroyTensor(allToAllOut);
        }
        if (x1DeviceAddr != nullptr) {
            aclrtFree(x1DeviceAddr);
        }
        if (x2DeviceAddr != nullptr) {
            aclrtFree(x2DeviceAddr);
        }
        if (biasDeviceAddr != nullptr) {
            aclrtFree(biasDeviceAddr);
        }
        if (outDeviceAddr != nullptr) {
            aclrtFree(outDeviceAddr);
        }
        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }
        aclrtDestroyStream(args.stream);
        HcclCommDestroy(args.hcclComm);
        aclrtDestroyContext(args.context);
        aclrtResetDevice(args.rankId);
        return 0;
    }

    int main(int argc, char *argv[])
    {
    // This sample is implemented based on Ascend 950PR&950DT products and must be run on Ascend 950PR&950DT products.
    int ret = aclInit(nullptr);
    int32_t devices[ndev];
    for (int i = 0; i < ndev; i++) {
    devices[i] = i;
    }
    HcclComm comms[128];
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    // Initialize the communicator.
    for (int i = 0; i < ndev; i++) {
    ret = aclrtSetDevice(devices[i]);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    }
    ret = HcclCommInitAll(ndev, devices, comms);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("HcclCommInitAll failed. ERROR: %d\n", ret); return ret);
    Args args[ndev];
    aclrtStream stream[ndev];
    aclrtContext context[ndev];
    for (uint32_t rankId = 0; rankId < ndev; rankId++) {
    ret = aclrtSetDevice(rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateContext(&context[rankId], rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(&stream[rankId]);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    }
    // Start multiple threads.
    std::vector<std::unique_ptr<std::thread>> threads(ndev);
    for (uint32_t rankId = 0; rankId < ndev; rankId++) {
    args[rankId].rankId = rankId;
    args[rankId].hcclComm = comms[rankId];
    args[rankId].stream = stream[rankId];
    args[rankId].context = context[rankId];
    threads[rankId].reset(new(std::nothrow) std::thread(&launchOneThreadAlltoAllQuantMatmul, std::ref(args[rankId])));
    }
    for (uint32_t rankId = 0; rankId < ndev; rankId++) {
    threads[rankId]->join();
    }
    aclFinalize();
    return 0;
    }
    ```

<!-- end id19 -->
