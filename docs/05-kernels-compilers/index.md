---
title: Kernel、算子与编译优化
domain: kernels-compilers
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-06-12
---

# Kernel、算子与编译优化

本目录关注 AI workload 在单机和单卡上的执行效率：算子如何映射到硬件、Kernel 为什么慢、编译器如何做图优化和融合，以及自动调优如何找到更好的实现。Triton 和 TorchInductor 是本目录的核心专题，分别对应手写高性能 Kernel 和 PyTorch 编译栈中的自动代码生成路径。

## 技术分层

| 层级 | 关注问题 | 代表技术 |
| --- | --- | --- |
| 算子语义 | workload 需要什么计算和数据访问模式 | GEMM、Attention、LayerNorm、Softmax、Embedding、MoE dispatch |
| 手写 Kernel | 如何针对具体 shape 和硬件写出高效实现 | Triton、CUDA、CUTLASS |
| 图编译与代码生成 | 如何从 PyTorch graph 自动融合、调度和生成代码 | TorchDynamo、AOTAutograd、TorchInductor、Triton backend |
| 性能诊断 | 如何证明 Kernel 或编译优化真的有效 | profiler、roofline、IR/PTX/SASS inspection |

## 建议主题

- GEMM、Attention、LayerNorm、Softmax、Embedding、MoE dispatch
- tiling、vectorization、memory coalescing、shared memory、register pressure
- occupancy、warp divergence、bank conflict、kernel launch overhead
- FlashAttention、fused operator、persistent kernel
- Triton、CUDA、ROCm、CUTLASS、cuDNN、cuBLAS
- TorchDynamo、AOTAutograd、TorchInductor、XLA、TVM、MLIR、ONNX Runtime
- graph capture、operator fusion、layout transform、auto-tuning
- kernel profiling、roofline、SASS/PTX/IR inspection

## 关键问题

- 算子是 compute-bound 还是 memory-bound。
- 是否存在多余的 memory read/write、layout transform 或 kernel launch。
- Fusion 是否减少访存，是否引入寄存器压力和 occupancy 下降。
- Triton Kernel 或 TorchInductor 生成的 kernel 是否符合 workload shape。
- 优化在不同 batch、sequence length、precision 下是否仍然成立。

## 专题入口

- [Attention 机制与计算模式](attention-computation-patterns.md)：区分 attention pattern、exact/approx、kernel 实现和 KV Cache 管理，理解 Dense/Sparse/Flash/PagedAttention、MHA/MQA/GQA、Prefill/Decode、长上下文、显存 IO 和 benchmark 方法。
- [Triton Kernel 编程](triton.md)：面向 AI workload 的手写 Kernel，覆盖 block program、JIT specialization、launch grid、tiling、资源模型、数值、autotune、PyTorch/Inductor 集成和端到端 benchmark。
- [TorchInductor 与 PyTorch 编译栈](torchinductor.md)：面向 `torch.compile`、graph capture、fusion、scheduler、Triton codegen 和性能调试。
