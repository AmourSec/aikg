---
title: TorchInductor 与 PyTorch 编译栈
domain: torchinductor
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-05-30
---

# TorchInductor 与 PyTorch 编译栈

TorchInductor 是理解 PyTorch 2.x 编译优化路径的核心专题。它连接上层 PyTorch graph 和底层 kernel codegen，适合研究图捕获、算子融合、调度、layout、Triton 代码生成和性能回归定位。

## 建议主题

- `torch.compile`、TorchDynamo、FX Graph、AOTAutograd
- TorchInductor graph lowering、scheduler、fusion、layout planning
- Triton codegen、C++ codegen、external kernel call
- guard、graph break、dynamic shape、symbolic shape
- operator decomposition、pattern matching、epilogue fusion
- memory planning、buffer reuse、layout transform
- compile time、cache、warmup、runtime performance
- debug tools、generated code inspection、IR 分析、profiler 关联

## 关键问题

- `torch.compile` 是否真的捕获到目标计算图，是否存在 graph break。
- Inductor 的 fusion 是否减少中间 tensor 落地，是否引入额外 layout transform。
- 生成的 Triton kernel 是否适合当前 shape、precision 和硬件。
- 编译开销、warmup、cache 命中和运行时收益是否平衡。
- dynamic shape、控制流、非标准算子是否导致性能退化。

## 学习顺序

1. 从 eager vs `torch.compile` 的执行差异开始。
2. 观察 graph break、FX graph 和 operator decomposition。
3. 分析 Inductor 生成的代码、kernel 数量和 fusion 边界。
4. 用 profiler 拆分 compile time、kernel time、memory copy 和 synchronization。
5. 对比 Inductor 自动生成 kernel 与手写 Triton kernel 的差异。

## 记录模板

| 字段 | 说明 |
| --- | --- |
| Model Fragment | 被编译的模型片段或算子组合 |
| Graph Capture | 是否有 graph break、guard、dynamic shape |
| Generated Kernels | kernel 数量、fusion 边界、Triton/C++ codegen |
| Bottleneck | compile time / runtime / memory / layout / graph break |
| Evidence | FX graph、Inductor IR、生成代码、profiler |
| Decision | 保持 eager、使用 Inductor、手写 Triton 或调用外部库 |
