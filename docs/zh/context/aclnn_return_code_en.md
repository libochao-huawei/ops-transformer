# aclnn Return Codes

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T13:25:25.185Z -->

When you call an aclnn API, common API return codes are listed in [Table 1](#table1).
For exception status code values, you can obtain the exception information through the aclGetRecentErrMsg API ([*Runtime API*](https://hiascend.com/en/document/redirect/CannCommunityRuntimeApi)), and troubleshoot the problem based on the error message or contact technical support.

**Table 1** Return status codes <a id="table1"></a>

| Status Code Name | Status Code Value | Status Code Description |
| ----- | ----- | ------ |
| ACLNN_SUCCESS | 0 | Success. |
| ACLNN_ERR_PARAM_NULLPTR | 161001 | Parameter validation error: an invalid nullptr exists in the parameters. |
| ACLNN_ERR_PARAM_INVALID | 161002 | Parameter validation error: for example, the data types of the two inputs do not satisfy the input type deduction relationship. |
| ACLNN_ERR_RUNTIME_ERROR | 361001 | An exception occurs when the API internally calls the NPU runtime API. |
| ACLNN_ERR_INNER_XXX | 561xxx | An internal exception occurs in the API. |

For details about status codes of the ACLNN_ERR_INNER_XXX type, see [Table 2](#table2).

**Table 2** Exception status codes <a id="table2"></a>

| Status Code Name | Status Code Value | Status Code Description |
| ----- | ----- | ------ |
| ACLNN_ERR_INNER |   561000     |    Internal Exception: An internal exception occurs in the API.     |
| ACLNN_ERR_INNER_INFERSHAPE_ERROR     |  561001      |  Internal Exception: An error occurs when the API performs output shape inference internally.       |
| ACLNN_ERR_INNER_TILING_ERROR     |   561002     |   Internal Exception: An exception occurs when the API performs tiling of the npu kernel internally.     |
| ACLNN_ERR_INNER_FIND_KERNEL_ERROR   |   561003  |  Internal Exception: An exception occurs when the API searches for the npu kernel internally (possibly because the operator binary package is not installed).    |
| ACLNN_ERR_INNER_CREATE_EXECUTOR     |    561101    |   Internal Exception: The API fails to create aclOpExecutor internally (possibly due to an operating system exception).      |
| ACLNN_ERR_INNER_NOT_TRANS_EXECUTOR     |  561102      |   Internal Exception: the ReleaseTo of uniqueExecutor is not called in the API.      |
| ACLNN_ERR_INNER_NULLPTR            |   561103     |    Internal Exception: An exception occurs inside the aclnn API, with a nullptr exception.     |
| ACLNN_ERR_INNER_WRONG_ATTR_INFO_SIZE     |   561104     |   Internal Exception: An exception occurs inside the aclnn API, with an abnormal number of operator attributes.      |
| ACLNN_ERR_INNER_KEY_CONFLICT (deprecated)     |   561105     |    **Deprecated. Use the latest ACLNN_ERR_INNER_KEY_CONFLICT instead.**     |
| ACLNN_ERR_INNER_KEY_CONFLICT |   561105     |   Internal Exception: An exception occurs inside the aclnn API, with a hash key conflict in operator kernel matching. |
| ACLNN_ERR_INNER_INVALID_IMPL_MODE     |   561106     |   Internal Exception: An exception occurs inside the aclnn API, with an invalid implementation mode parameter of the operator.    |
| ACLNN_ERR_INNER_OPP_PATH_NOT_FOUND     |  561107     |  Internal Exception: An exception occurs inside the aclnn API, because the required environment variable ASCEND_OPP_PATH is not detected.       |
| ACLNN_ERR_INNER_LOAD_JSON_FAILED     |  561108      |   Internal Exception: An exception occurs inside the aclnn API, and the operator information JSON file in the operator kernel library fails to be loaded.      |
| ACLNN_ERR_INNER_JSON_VALUE_NOT_FOUND     |   561109      |   Internal Exception: An exception occurs inside the aclnn API, and a field in the operator information JSON file in the operator kernel library fails to be loaded.      |
| ACLNN_ERR_INNER_JSON_FORMAT_INVALID     |  561110     |     Internal Exception: An exception occurs inside the aclnn API, and the format of the operator information JSON file in the operator kernel library is set to an invalid value.    |
| ACLNN_ERR_INNER_JSON_DTYPE_INVALID     |    561111      |     Internal Exception: An exception occurs inside the aclnn API, and the dtype of the operator information JSON file in the operator kernel library is set to an invalid value.    |
| ACLNN_ERR_INNER_OPP_KERNEL_PKG_NOT_FOUND     |   561112    |    Internal Exception: An exception occurs inside the aclnn API, and the binary kernel library of the operator fails to be loaded.     |
| ACLNN_ERR_INNER_OP_FILE_INVALID     |  561113     |   Internal Exception: An exception occurs inside the aclnn API when the fields of the operator JSON file are loaded.   |
| ACLNN_ERR_INNER_ATTR_NUM_OUT_OF_BOUND     |  561114      |  Internal Exception: An exception occurs inside the aclnn API. The number of operator attributes is inconsistent with that in the operator information JSON and exceeds the number of attrs specified in the JSON.       |
| ACLNN_ERR_INNER_ATTR_LEN_NOT_ENOUGH     |   561115      |   Internal Exception: An exception occurs inside the aclnn API. The number of operator attributes is inconsistent with that in the operator information JSON and is less than the number of attrs specified in the JSON.      |
| ACLNN_ERR_INNER_INPUT_NUM_IN_JSON_TOO_LARGE     |   561116     |   Internal Exception: An exception occurs inside the aclnn API, and the number of operator inputs exceeds the limit of 32.      |
| ACLNN_ERR_INNER_INPUT_JSON_IS_NULL     |   561117     |  Internal Exception: An exception occurs inside the aclnn API, and the description in the operator information JSON file is incomplete.       |
| ACLNN_ERR_INNER_STATIC_WORKSPACE_INVALID     |  561118     |    Internal Exception: An exception occurs inside the aclnn API when the workspace information in the static binary JSON file is parsed.     |
| ACLNN_ERR_INNER_STATIC_BLOCK_DIM_INVALID     |  561119      |    Internal Exception: An exception occurs inside the aclnn API when the core usage information in the static binary JSON file is parsed.  |
