---
title: Triton Kernel 编程
domain: triton
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-05-30
---

# Triton Kernel 编程

Triton 是本知识库中手写 AI Kernel 的核心专题。它适合用来理解一个算子如何从张量语义落到 block、tile、program、memory access 和硬件执行效率上。

## 建议主题

- Triton program model、program id、block pointer、mask
- block size、num warps、num stages、tile shape
- memory coalescing、shared memory、register pressure、occupancy
- matmul、attention、layernorm、softmax、embedding、MoE dispatch
- fused kernel、persistent kernel、pipeline、prefetch
- auto-tuning、shape specialization、dynamic shape 处理
- 与 CUDA、CUTLASS、cuBLAS、TorchInductor codegen 的关系
- Triton profiler、IR、PTX、SASS、Nsight 分析

## 关键问题

- 这个算子是否适合用 Triton 手写，而不是调用库函数或依赖编译器生成。
- 当前瓶颈是访存、计算、同步、寄存器压力还是 launch overhead。
- tile shape 是否匹配输入 shape、cache 行为和矩阵单元。
- mask、layout 和边界处理是否引入额外开销。
- auto-tuning 搜索空间是否覆盖真实 workload，而不是只覆盖单一 shape。

## 学习顺序

1. 从 vector add、copy、reduce 理解 program model。
2. 用 matmul 理解 tiling、block size、num warps 和 memory reuse。
3. 用 softmax、layernorm 理解 memory-bound 算子。
4. 用 attention 或 fused MLP 理解 fusion、pipeline 和中间结果落地成本。
5. 用 profiler 对比 Triton、PyTorch eager、TorchInductor 和 vendor library。

## 记录模板

| 字段 | 说明 |
| --- | --- |
| Operator | 算子语义和输入输出 shape |
| Bottleneck | compute / memory / launch / register / occupancy |
| Baseline | PyTorch、cuBLAS、cuDNN、CUTLASS 或 Inductor 结果 |
| Triton Config | block size、num warps、num stages、precision |
| Evidence | profiler、roofline、吞吐、延迟、显存访问 |
| Portability | 不同 GPU、shape、precision 下是否稳定 |
