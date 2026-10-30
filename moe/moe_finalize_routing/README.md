# MoeFinalizeRouting

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT系列产品</term>|      ×     |
|<term>Atlas A3系列产品</term>|      √     |
|<term>Atlas A2系列产品</term>|      √     |
|<term>Atlas 200I/500 A2推理产品</term>|      ×     |
|<term>Atlas推理系列产品</term>|      ×     |
|<term>Atlas训练系列产品</term>|      ×     |
|<term>Kirin X90处理器系列产品</term> | √ |
|<term>Kirin 9030处理器系列产品</term> | √ |

## 功能说明

- **算子功能**：MoE计算中，最后处理合并MoE FFN的输出结果。
- **计算公式**：

  $$
  expertid=expandedExpertIdx[i,k]
  $$

  $$
  out(i,j)=x1_{i,j}+x2Optional_{i,j}+\sum_{k=0}^{K-1}(scales_{i,k}*(expandedX_{expandedRowIdx_{i+k*num_rows},j}+bias_{expertid,j}))
  $$

## 参数说明

  <table style="table-layout: auto; width: 100%">
    <thead>
      <tr>
        <th style="white-space: nowrap">参数名</th>
        <th style="white-space: nowrap">输入/输出/属性</th>
        <th style="white-space: nowrap">描述</th>
        <th style="white-space: nowrap">数据类型</th>
        <th style="white-space: nowrap">数据格式</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>expandedX</td>
        <td>输入</td>
        <td>公式中的`expandedX` ，MoE的FFN输出。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>x1</td>
        <td>输入</td>
        <td>公式中的`x1`。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>x2Optional</td>
        <td>输入</td>
        <td>公式中的`x2Optional`。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>bias</td>
        <td>输入</td>
        <td>公式中的`bias`。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>scales</td>
        <td>输入</td>
        <td>公式中的`scales`。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expandedRowIdx</td>
        <td>输入</td>
        <td>公式中的`expandedRowIdx`。</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>expandedExpertIdx</td>
        <td>输入</td>
        <td>公式中的`expandedExpertIdx`。</td>
        <td>INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>out</td>
        <td>输出</td>
        <td>公式中的`out`。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
      </tr>
    </tbody>
  </table>

## 约束说明

- 输入 expandedX、x1、x2Optional、bias、scales、expandedExpertIdx 为 2D Tensor，expandedRowIdx 为 1D Tensor，其中 x2Optional 为可选输入。
- x1、x2Optional、scales、expandedExpertIdx 的 0 轴大小相同。
- x1、x2Optional、expandedX、bias 的 1 轴大小相同。
- scales 与 expandedExpertIdx 的 1 轴大小相同，即公式中的 K。
- expandedX 与 expandedRowIdx 的 0 轴大小相同。
- expandedRowIdx、expandedExpertIdx 的数据类型为 INT32，其余输入与输出 out 的数据类型为 FLOAT16、BFLOAT16 或 FLOAT32。
- <term>Kirin X90处理器系列产品</term>、<term>Kirin 9030处理器系列产品</term>: 不支持BFLOAT16。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_moe_finalize_routing.cpp](examples/test_aclnn_moe_finalize_routing.cpp) | 通过[aclnnMoeFinalizeRouting](docs/aclnnMoeFinalizeRouting.md)接口方式调用MoeFinalizeRouting算子。 |
