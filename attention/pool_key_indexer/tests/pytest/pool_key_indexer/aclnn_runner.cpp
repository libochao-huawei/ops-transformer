/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * pool_key_indexer pytest 框架的 aclnn 直调执行器（T1-3 NPU 后端）。
 *
 * 背景: 主线 torch_extension csrc 的 isQuantPath 仅覆盖 quant_mode 0/1（int8 e2e 属
 * 迭代三 T3-3），quant_mode=2 会被 "requires q_descale and k_descale to be null" 拒绝，
 * 故 pytest 采用 aclnn 直调路径（对齐 examples/ 与 910B3 已验证 smoke 载体）。
 *
 * 用法: aclnn_runner <case_dir> [device_id]
 *   case_dir/manifest.txt —— 行式清单:
 *     tensor <key> <dtype> <ndims> <d0> ...        (对应 <key>.bin, 缺省即 nullptr)
 *     attr <name> <int>                            (topk/pool_size/mask_mode/quant_mode/
 *                                                  return_value/key_stride0/k_descale_stride0)
 *     str <name> <value>                           (layout_q/layout_k)
 *   dtype: int8 | int32 | int64 | float16 | bfloat16 | float32
 *   bfloat16 以 uint16 原始位型存储于 .bin。
 * 输出: case_dir/indices.bin (int32 原始数据) + case_dir/values.bin (float32; rv=false 为空)
 * 退出码: 0=成功; 10=GetWorkspaceSize 拒绝(非法用例预期); 1=运行失败; 2=输入/IO 错误
 *
 * 序列模式(LD 缺陷探针 P-② 跨调用残留用): aclnn_runner --seq <device_id> <dir1> <dir2> ...
 *   同一进程顺序执行多条用例, 每条结束后释放其全部 device 内存(输入/输出/workspace)
 *   再进入下一条 —— 模拟生产环境同尺寸 workspace/输出内存的分配器复用, 使上一条
 *   用例的 LD workspace / 输出区残留可被下一条用例观测。每条用例输出落各自
 *   case_dir, stdout 逐条打印 SEQ_CASE <rc> <dir>。
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_pool_key_indexer.h"
#include "aclnnop/aclnn_pool_key_indexer_v2.h"

namespace {

struct TensorSpec {
    std::string dtype;
    std::vector<int64_t> dims;
    std::vector<char> bytes;
};

struct Case {
    std::map<std::string, TensorSpec> tensors;
    std::map<std::string, int64_t> attrs;
    std::map<std::string, std::string> strs;
};

int64_t Attr(const Case& c, const std::string& name, int64_t dft)
{
    auto it = c.attrs.find(name);
    return it == c.attrs.end() ? dft : it->second;
}

std::string Str(const Case& c, const std::string& name, const std::string& dft)
{
    auto it = c.strs.find(name);
    return it == c.strs.end() ? dft : it->second;
}

aclDataType ToAclDtype(const std::string& t)
{
    if (t == "int8")
        return ACL_INT8;
    if (t == "int32")
        return ACL_INT32;
    if (t == "int64")
        return ACL_INT64;
    if (t == "float16")
        return ACL_FLOAT16;
    if (t == "bfloat16")
        return ACL_BF16;
    if (t == "float32")
        return ACL_FLOAT;
    fprintf(stderr, "unknown dtype: %s\n", t.c_str());
    exit(2);
}

bool ReadFile(const std::string& path, std::vector<char>& out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }
    f.seekg(0, std::ios::end);
    std::streamoff n = f.tellg();
    f.seekg(0, std::ios::beg);
    out.resize(static_cast<size_t>(n));
    f.read(out.data(), n);
    return true;
}

aclTensor* MakeAclTensor(const TensorSpec& spec, void** devMem)
{
    const int64_t* dims = spec.dims.data();
    size_t ndim = spec.dims.size();
    std::vector<int64_t> strides(ndim, 1);
    for (int64_t i = static_cast<int64_t>(ndim) - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    size_t bytes = spec.bytes.size();
    void* dev = nullptr;
    // 空 tensor(shape 含 0 维)按 1 字节兜底分配, 避免 aclrtMalloc(0) 参数非法
    // (T3-2 空 tensor 用例 L2_006; 分配量不影响 host 校验拒绝路径)
    if (aclrtMalloc(&dev, bytes == 0 ? 1 : bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        fprintf(stderr, "aclrtMalloc failed (%zu bytes)\n", bytes);
        exit(1);
    }
    if (aclrtMemcpy(dev, bytes, spec.bytes.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        fprintf(stderr, "aclrtMemcpy failed\n");
        exit(1);
    }
    *devMem = dev;
    return aclCreateTensor(dims, ndim, ToAclDtype(spec.dtype), strides.data(), 0, ACL_FORMAT_ND, dims, ndim, dev);
}

aclIntArray* MakeIntArray(const TensorSpec& spec)
{
    size_t n = spec.bytes.size() / sizeof(int64_t);
    std::vector<int64_t> vals(n);
    memcpy(vals.data(), spec.bytes.data(), n * sizeof(int64_t));
    return aclCreateIntArray(vals.data(), n);
}

} // namespace

// 单条用例执行: 解析 manifest → 建 tensor → GetWorkspaceSize/run → dump 输出 →
// 释放本条全部 device 内存(序列模式跨调用残留探针依赖此释放, 单用例模式仅作资源卫生)。
// 返回 0=成功 / 10=GWS 拒绝 / 1=运行失败 / 2=IO 错误。
static int RunOneCase(const std::string& dir, aclrtStream stream)
{
    Case cs;
    {
        std::ifstream f(dir + "/manifest.txt");
        if (!f) {
            fprintf(stderr, "cannot open %s/manifest.txt\n", dir.c_str());
            return 2;
        }
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }
            std::istringstream iss(line);
            std::string kind;
            iss >> kind;
            if (kind == "tensor") {
                TensorSpec spec;
                std::string key;
                iss >> key >> spec.dtype;
                int64_t nd = 0;
                iss >> nd;
                for (int64_t i = 0; i < nd; i++) {
                    int64_t d = 0;
                    iss >> d;
                    spec.dims.push_back(d);
                }
                if (!ReadFile(dir + "/" + key + ".bin", spec.bytes)) {
                    fprintf(stderr, "cannot read %s/%s.bin\n", dir.c_str(), key.c_str());
                    return 2;
                }
                cs.tensors[key] = spec;
            } else if (kind == "attr") {
                std::string name;
                int64_t v = 0;
                iss >> name >> v;
                cs.attrs[name] = v;
            } else if (kind == "str") {
                std::string name, v;
                iss >> name >> v;
                cs.strs[name] = v;
            }
        }
    }

    int64_t topk = Attr(cs, "topk", 2048);
    int64_t poolSize = Attr(cs, "pool_size", 16);
    bool returnValue = Attr(cs, "return_value", 0) != 0;
    std::string layoutQ = Str(cs, "layout_q", "BSND");
    std::string layoutK = Str(cs, "layout_k", "BSND");

    std::vector<void*> devMems;
    std::vector<aclTensor*> aclTensors;
    std::vector<aclIntArray*> aclArrays;
    auto cleanup = [&devMems, &aclTensors, &aclArrays]() {
        for (auto* arr : aclArrays) {
            aclDestroyIntArray(arr);
        }
        for (auto* t : aclTensors) {
            if (t != nullptr) {
                aclDestroyTensor(t);
            }
        }
        for (auto* m : devMems) {
            if (m != nullptr) {
                aclrtFree(m);
            }
        }
    };

    // ---- 输入 ----
    auto mk = [&](const char* key) {
        auto it = cs.tensors.find(key);
        if (it == cs.tensors.end()) {
            return static_cast<aclTensor*>(nullptr);
        }
        void* mem = nullptr;
        aclTensor* t = MakeAclTensor(it->second, &mem);
        devMems.push_back(mem);
        aclTensors.push_back(t);
        return t;
    };
    aclTensor* tq = mk("query");
    aclTensor* tk = mk("pool_key");
    aclTensor* tw = mk("weights");
    aclTensor* tbt = mk("block_table");
    aclTensor* tdq = mk("q_descale");
    aclTensor* tdk = mk("k_descale");

    auto mkArr = [&](const char* key) {
        auto it = cs.tensors.find(key);
        if (it == cs.tensors.end()) {
            return static_cast<aclIntArray*>(nullptr);
        }
        aclIntArray* arr = MakeIntArray(it->second);
        aclArrays.push_back(arr);
        return arr;
    };
    aclIntArray* arrPtk = mkArr("pool_tail_k");
    aclIntArray* arrAsq = mkArr("actual_seq_q");
    aclIntArray* arrAsk = mkArr("actual_seq_k");

    // ---- 输出 shape（对齐 op_host infershape / torch_extension 构造逻辑）----
    // 非法用例(如 rank-0 标量 query)会被 GetWorkspaceSize 拒绝, 输出 shape 仅需
    // 结构合法: 维数不足时以 0 占位(T3-2 L2_007), 避免越界访问空 dims。
    // out_dim0/out_dim1: 空/rank-0 query 用例的输出 shape 覆盖 —— aclnn executor
    // 对全空输出会跳过 tiling/kernel(CANN 框架行为), 须以非空输出 shape 到达
    // 算子 host 校验分支(T3-2 L2_006/L2_007)
    const TensorSpec& qSpec = cs.tensors["query"];
    int64_t indicesLen = topk + poolSize - 1;
    int64_t valuesLen = topk / poolSize;
    int64_t qDim0 = qSpec.dims.size() > 0 ? qSpec.dims[0] : 0;
    int64_t qDim1 = qSpec.dims.size() > 1 ? qSpec.dims[1] : 0;
    if (cs.attrs.find("out_dim0") != cs.attrs.end()) {
        qDim0 = Attr(cs, "out_dim0", qDim0);
    }
    if (cs.attrs.find("out_dim1") != cs.attrs.end()) {
        qDim1 = Attr(cs, "out_dim1", qDim1);
    }
    std::vector<int64_t> idxShape, valShape;
    if (layoutQ == "BSND") {
        idxShape = {qDim0, qDim1, indicesLen};
        valShape = {qDim0, qDim1, valuesLen};
    } else {
        idxShape = {qDim0, indicesLen};
        valShape = {qDim0, valuesLen};
    }

    auto mkOut = [&](const std::vector<int64_t>& shape, aclDataType dt, void** mem) {
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        size_t n = 1;
        for (int64_t d : shape) {
            n *= static_cast<size_t>(d);
        }
        size_t bytes = n * ((dt == ACL_INT32 || dt == ACL_FLOAT) ? 4 : 2);
        if (aclrtMalloc(mem, bytes == 0 ? 1 : bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            fprintf(stderr, "aclrtMalloc output failed\n");
            cleanup();
            return static_cast<aclTensor*>(nullptr);
        }
        devMems.push_back(*mem);
        aclTensor* t = aclCreateTensor(shape.data(), shape.size(), dt, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                                       shape.size(), *mem);
        aclTensors.push_back(t);
        return t;
    };
    void* idxMem = nullptr;
    void* valMem = nullptr;
    aclTensor* tIdx = mkOut(idxShape, ACL_INT32, &idxMem);
    aclTensor* tVal = mkOut(returnValue ? valShape : std::vector<int64_t>{0}, ACL_FLOAT, &valMem);

    // ---- GetWorkspaceSize + run ----
    char bufLq[16];
    char bufLk[16];
    snprintf(bufLq, sizeof(bufLq), "%s", layoutQ.c_str());
    snprintf(bufLk, sizeof(bufLk), "%s", layoutK.c_str());
    uint64_t wsSize = 0;
    aclOpExecutor* exec = nullptr;
    // sparse_indices_mode 属性(索引输出布局模式, V2 接口专有): manifest 显式带
    // sparse_indices_mode 即走 aclnnPoolKeyIndexerV2(值透传, 0=legacy / 1=compact);
    // 缺省走 V1(aclnnPoolKeyIndexer, legacy 布局, 模型侧兼容口径)
    const bool useV2 = cs.attrs.find("sparse_indices_mode") != cs.attrs.end();
    const int64_t sparseIndicesMode = Attr(cs, "sparse_indices_mode", 0);
    aclnnStatus ret;
    if (useV2) {
        ret = aclnnPoolKeyIndexerV2GetWorkspaceSize(
            tq, tk, tw, arrPtk, arrAsq, arrAsk, tbt, tdq, tdk, topk, poolSize, bufLq, bufLk, Attr(cs, "mask_mode", 3),
            Attr(cs, "quant_mode", -1), returnValue, Attr(cs, "key_stride0", -1), Attr(cs, "k_descale_stride0", -1),
            sparseIndicesMode, tIdx, tVal, &wsSize, &exec);
    } else {
        ret = aclnnPoolKeyIndexerGetWorkspaceSize(tq, tk, tw, arrPtk, arrAsq, arrAsk, tbt, tdq, tdk, topk, poolSize,
                                                  bufLq, bufLk, Attr(cs, "mask_mode", 3), Attr(cs, "quant_mode", -1),
                                                  returnValue, Attr(cs, "key_stride0", -1),
                                                  Attr(cs, "k_descale_stride0", -1), tIdx, tVal, &wsSize, &exec);
    }
    if (ret != ACL_SUCCESS) {
        printf("GWS_FAILED ret=%d\n", static_cast<int>(ret));
        fflush(stdout);
        cleanup();
        return 10;
    }
    void* ws = nullptr;
    if (wsSize > 0 && aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        fprintf(stderr, "workspace malloc failed (%lu)\n", wsSize);
        cleanup();
        return 1;
    }
    if (ws != nullptr) {
        devMems.push_back(ws);
    }
    int runRc = 0;
    if (useV2) {
        if (aclnnPoolKeyIndexerV2(ws, wsSize, exec, stream) != ACL_SUCCESS) {
            fprintf(stderr, "aclnnPoolKeyIndexerV2 run failed\n");
            runRc = 1;
        }
    } else {
        if (aclnnPoolKeyIndexer(ws, wsSize, exec, stream) != ACL_SUCCESS) {
            fprintf(stderr, "aclnnPoolKeyIndexer run failed\n");
            runRc = 1;
        }
    }
    // 成败路径均同步: 残留异步任务落地后再释放内存, 防释放竞态
    aclrtSynchronizeStream(stream);

    // ---- dump 输出 ----
    if (runRc == 0) {
        size_t idxBytes = 1;
        for (int64_t d : idxShape) {
            idxBytes *= static_cast<size_t>(d);
        }
        idxBytes *= sizeof(int32_t);
        size_t valBytes = 0;
        if (returnValue) {
            valBytes = 1;
            for (int64_t d : valShape) {
                valBytes *= static_cast<size_t>(d);
            }
            valBytes *= sizeof(float);
        }
        {
            std::vector<char> host(idxBytes);
            if (idxBytes > 0) {
                aclrtMemcpy(host.data(), idxBytes, idxMem, idxBytes, ACL_MEMCPY_DEVICE_TO_HOST);
            }
            std::ofstream f(dir + "/indices.bin", std::ios::binary);
            f.write(host.data(), static_cast<std::streamsize>(idxBytes));
        }
        {
            std::vector<char> host(valBytes);
            if (valBytes > 0) {
                aclrtMemcpy(host.data(), valBytes, valMem, valBytes, ACL_MEMCPY_DEVICE_TO_HOST);
            }
            std::ofstream f(dir + "/values.bin", std::ios::binary);
            f.write(host.data(), static_cast<std::streamsize>(valBytes));
        }
        printf("RUN_OK idx_bytes=%zu val_bytes=%zu\n", idxBytes, valBytes);
    }
    fflush(stdout);
    cleanup();
    return runRc;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s <case_dir> [device_id]\n"
                "       %s --seq <device_id> <dir1> <dir2> ...\n",
                argv[0], argv[0]);
        return 2;
    }

    // ---- 序列模式: 同进程多条用例, 逐条释放 device 内存(跨调用残留探针) ----
    if (std::string(argv[1]) == "--seq") {
        if (argc < 4) {
            fprintf(stderr, "usage: %s --seq <device_id> <dir1> <dir2> ...\n", argv[0]);
            return 2;
        }
        int32_t deviceId = atoi(argv[2]);
        if (aclInit(nullptr) != ACL_SUCCESS || aclrtSetDevice(deviceId) != ACL_SUCCESS) {
            fprintf(stderr, "acl init / set device failed\n");
            return 1;
        }
        aclrtStream stream;
        aclrtCreateStream(&stream);
        int finalRc = 0;
        for (int i = 3; i < argc; i++) {
            int rc = RunOneCase(argv[i], stream);
            printf("SEQ_CASE %d %s\n", rc, argv[i]);
            fflush(stdout);
            if (rc != 0) {
                finalRc = rc;
            }
        }
        return finalRc;
    }

    // ---- 单用例模式(原行为) ----
    std::string dir = argv[1];
    int32_t deviceId = (argc >= 3) ? atoi(argv[2]) : 0;
    if (aclInit(nullptr) != ACL_SUCCESS || aclrtSetDevice(deviceId) != ACL_SUCCESS) {
        fprintf(stderr, "acl init / set device failed\n");
        return 1;
    }
    aclrtStream stream;
    aclrtCreateStream(&stream);
    return RunOneCase(dir, stream);
}
