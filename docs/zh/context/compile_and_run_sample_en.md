# Compile and Run Samples

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T13:19:04.287Z -->

## Prerequisites

- To compile and execute operator APIs, ensure that the basic environment is set up, including the driver, firmware, CANN software package, and ops package.

- For details about the operator API call process and compilation and running operations, see "Single Operator Call > Single Operator API Execution > Example Code for Calling aclnn APIs" in [*Application Development (C&C++)*](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/others/acldevg/aclcppdevg_000006.html).

## Preparations Before Compilation

This section uses a combined development and running environment as an example. That is, the machine with an AI processor serves as both the development environment and the running environment. In this use case, code development and code running are performed on the same machine. The **FlashAttentionScore operator** is used as an example. The invocation logic, process, and build script of other operators are similar to those of the FlashAttentionScore operator. Modify the API invocation script (\*.cpp) and build script (CMakeLists) as required.

- **Sample Code**

   The FlashAttentionScore operator implements the computation of self-attention:

  - If psetype=1, add is performed before mul.
  - If psetype≠1, mul is performed before add.

- Formula:

  The forward calculation formula of attention is as follows:

  - If psetype=1, the formula is as follows:
      $$
      attention\_out = Dropout(Softmax(Mask(scale*(pse+query*key^T), atten\_mask)), keep\_prob)*value
      $$

  - If psetype≠1, the formula is as follows:
      $$
      attention\_out=Dropout(Softmax(Mask(scale*(query*key^T) + pse),atten\_mask),keep\_prob)*value
      $$

  You can obtain the invocation sample code from the [aclnnFlashAttentionScore](../../../attention/flash_attention_score/docs/aclnnFlashAttentionScore_en.md) document and name the code file **test\_flash_attention_score.cpp**.

- **CMakeLists File**

    The following is an example of the CMake file. Modify it based on the actual situation:

    ```bash
    # Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

    # CMake lowest version requirement
    cmake_minimum_required(VERSION 3.18.4)

    # Set the project name.
    project(ACLNN_EXAMPLE)

    # Compile options
    add_compile_options(-std=c++11)

    # Set compile options
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "./bin")
    set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
    set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

    # Set the executable file name (e.g., opapi_test) and the directory where the *.cpp file of the operator to be run resides
    add_executable(opapi_test
                   test_flash_attention_score.cpp)

    # Set ASCEND_PATH (CANN software package directory; change it to the actual path) and INCLUDE_BASE_DIR (header file directory)
    if(NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
        set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
    else()
        set(ASCEND_PATH "/usr/local/Ascend/cann")
    endif()
    set(INCLUDE_BASE_DIR "${ASCEND_PATH}/include")
    include_directories(
        ${INCLUDE_BASE_DIR}
        ${INCLUDE_BASE_DIR}/aclnn
        ${INCLUDE_BASE_DIR}/aclnnop
        # For the custom package, add extra reference paths as shown below, where ${vendor_name} must be changed to the value specified when building the custom package; the default is "custom"
        # ${ASCEND_PATH}/opp/vendors/${vendor_name}_transformer/op_api/include
    )

    # Set the library paths to link
    target_link_libraries(opapi_test PRIVATE
                          ${ASCEND_PATH}/lib64/libascendcl.so
                          ${ASCEND_PATH}/lib64/libnnopbase.so
                          ${ASCEND_PATH}/lib64/libopapi_math.so
                          ${ASCEND_PATH}/lib64/libopapi_transformer.so
                          ${ASCEND_PATH}/lib64/libc_sec.so
                          # For the custom package, replace libopapi_transformer.so with libcust_opapi.so. The reference path is as follows,
                          # where ${vendor_name} must be changed to the value specified when building the custom package; the default is "custom"
                          # ${ASCEND_PATH}/opp/vendors/${vendor_name}_transformer/op_api/lib/libcust_opapi.so
                          )

    # The executable file is in the bin directory under the directory where the CMakeLists file resides
    install(TARGETS opapi_test DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    ```

    When building the operator sample in this project, additionally link the libopapi_math.so dynamic library. The operator calls some L0 APIs in its implementation, and these APIs are encapsulated in libopapi_math.so. Therefore, this dependency must be explicitly declared at the compilation and linking stage.

    Operators that integrate collective communication with MatMul computation and parallelism are collectively referred to as communication-computation fusion operators (MC2 operators for short), including AllGatherMatmul, AlltoAllAllGatherBatchMatMul, BatchMatMulReduceScatterAlltoAll, MatmulAllReduce, MatmulAllReduceAddRmsNorm, MatmulReduceScatter, and others. When calling APIs of such operators, multi-threading and HCCL (Huawei Collective Communication Library) are typically involved. Therefore, the CMake file must additionally import the following content; otherwise, compilation will fail.

  ```bash
  # Set the library paths to link
  find_package(Threads REQUIRED)
  target_link_libraries(opapi_test PRIVATE
                        ${ASCEND_PATH}/lib64/libascendcl.so
                        ${ASCEND_PATH}/lib64/libnnopbase.so
                        ${ASCEND_PATH}/lib64/libopapi_math.so
                        ${ASCEND_PATH}/lib64/libopapi_transformer.so
                        ${ASCEND_PATH}/lib64/libc_sec.so
                        # For the custom package, replace libopapi_transformer.so with libcust_opapi.so. The reference path is as follows,
                        # where ${vendor_name} needs to be changed to the value specified when building the custom package, with a default of "custom"
                        # ${ASCEND_PATH}/opp/vendors/${vendor_name}_transformer/op_api/lib/libcust_opapi.so
                        -Wl,--no-as-needed
                        ${ASCEND_PATH}/lib64/libhccl.so      # Collective communication library file
                        ${ASCEND_PATH}/lib64/libhccl_fwk.so
                        ${CMAKE_THREAD_LIBS_INIT})           # Library file that multithreading depends on
  ```

  Here, "find_package(Threads REQUIRED)" is a CMake command for locating the thread library, which automatically links the header files that the thread library depends on or the indirectly dependent library files.

## Compile and Run

  1. Prepare the operator invocation code (\*.cpp) and the build script (CMakeLists.txt) in advance.
  2. Configure environment variables.

     After installing the CANN software, log in to the CANN runtime environment as the user and execute the following command to make the environment variables take effect.

        ```bash
        source ${INSTALL_DIR}/set_env.sh
        ```

     Where `${INSTALL_DIR}` is the storage path of the CANN software after installation. Replace it with the actual path as needed.
  3. Compile and run.
      - Go to the directory where CMakeLists.txt is located and run the following command to create a build directory for storing the generated compilation files.

        ```bash
        mkdir -p build
        ```

      - Go to the build directory, run the cmake command to compile, and then run the make command to generate the executable file.

        ```bash
        cd build
        cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE
        make
        ```

        After successful compilation, the opapi\_test executable file is generated in the bin folder under the build directory.

      - Go to the bin directory and run the executable file opapi\_test.

        ```bash
        cd bin
        ./opapi_test
        ```

        The following example shows the run result of the FlashAttentionScore operator:

        ```text
        result[0] is: 256.000000
        result[1] is: 256.000000
        result[2] is: 256.000000
        result[3] is: 256.000000
        result[4] is: 256.000000

        ...
        result[65532] is: 256.000000
        result[65533] is: 256.000000
        result[65534] is: 256.000000
        result[65535] is: 256.000000
        ```

        If an error occurs during execution and the expected result is not obtained, call the aclGetRecentErrMsg API to obtain detailed error information.
        The following is an example of the exception information obtained when the call to aclnnFlashAttentionScoreGetWorkspaceSize fails:

        ```text
        // q is nullptr
        ret = aclnnFlashAttentionScoreGetWorkspaceSize(
          q, k, v, pse, dropMask, padding, attenmask, prefix, scaleValue,
          keepProb, preTokens, nextTokens, headNum, layOut, innerPrecise,
          sparseMode, softmaxMax, softmaxSum, softmaxOut, attentionOut, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionScoreGetWorkspaceSize failed. ERROR: %d.\n[ERROR msg]%s", ret, aclGetRecentErrMsg()); return ret);
        ```

        The following is an example of obtaining error information for the null pointer issue described above:

        ```text
        aclnnFlashAttentionScoreGetWorkspaceSize failed. ERROR: 161001
        [ERROR msg][PID:xxxx] xxx(timestamp) AclNN_Parameter_Error(EZ1001): The query cannot be nullptr.
        ```
