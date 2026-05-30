---
title: Kernel、算子与编译优化
domain: kernels-compilers
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-05-30
---

# Kernel、算子与编译优化

本目录关注 AI workload 在单机和单卡上的执行效率：算子如何映射到硬件、Kernel 为什么慢、编译器如何做图优化和融合，以及自动调优如何找到更好的实现。

## 建议主题

- GEMM、Attention、LayerNorm、Softmax、Embedding、MoE dispatch
- tiling、vectorization、memory coalescing、shared memory、register pressure
- occupancy、warp divergence、bank conflict、kernel launch overhead
- FlashAttention、fused operator、persistent kernel
- CUDA、ROCm、Triton、CUTLASS、cuDNN、cuBLAS
- TorchInductor、XLA、TVM、MLIR、ONNX Runtime
- graph capture、operator fusion、layout transform、auto-tuning
- kernel profiling、roofline、SASS/PTX/IR inspection

## 关键问题

- 算子是 compute-bound 还是 memory-bound。
- 是否存在多余的 memory read/write、layout transform 或 kernel launch。
- Fusion 是否减少访存，是否引入寄存器压力和 occupancy 下降。
- 编译器生成的 kernel 是否符合 workload shape。
- 优化在不同 batch、sequence length、precision 下是否仍然成立。
