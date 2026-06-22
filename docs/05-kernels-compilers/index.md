---
title: Kernel、算子与编译优化
domain: kernels-compilers
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-06-22
---

# Kernel、算子与编译优化

本目录关注 AI workload 在单机和单卡上的执行效率：算子如何映射到硬件、Kernel 为什么慢、编译器如何做图优化和融合，IR 如何承载 lowering，tile/schedule 如何影响硬件效率，以及 MegaKernel 这类更激进的融合路径什么时候值得研究。

本章的主线是：

```text
先理解 Attention 等核心 workload 的计算模式
-> 再理解 Triton / TileLang 这类 kernel 编程方式
-> 再理解 TorchInductor / MLIR 这类编译表示和生成路径
-> 最后评估 Persistent Kernel / MegaKernel / 自动生成
```

## 技术分层

| 层级 | 关注问题 | 代表技术 |
| --- | --- | --- |
| 算子语义 | workload 需要什么计算和数据访问模式 | GEMM、Attention、LayerNorm、Softmax、Embedding、MoE dispatch |
| 手写 Kernel | 如何针对具体 shape 和硬件写出高效实现 | Triton、CUDA、CUTLASS |
| Tile 与 Schedule | 如何表达 tile、layout、pipeline、tensorization 和 autotuning | TileLang、Triton、CUTLASS、TVM |
| 图编译与代码生成 | 如何从 PyTorch graph 自动融合、调度和生成代码 | TorchDynamo、AOTAutograd、TorchInductor、Triton backend |
| 编译 IR 与 Lowering | 如何在多层抽象中表达、验证和降低计算 | MLIR、StableHLO、Linalg、MemRef、Vector、GPU dialect |
| 跨算子融合 | 如何把多个算子或子图变成更少、更长驻的执行单元 | operator fusion、persistent kernel、MegaKernel |
| 性能诊断 | 如何证明 Kernel 或编译优化真的有效 | profiler、roofline、IR/PTX/SASS inspection |

## 建议主题

- GEMM、Attention、LayerNorm、Softmax、Embedding、MoE dispatch
- tiling、vectorization、memory coalescing、shared memory、register pressure
- occupancy、warp divergence、bank conflict、kernel launch overhead
- FlashAttention、fused operator、persistent kernel
- Triton、CUDA、ROCm、CUTLASS、cuDNN、cuBLAS
- TorchDynamo、AOTAutograd、TorchInductor、XLA、TVM、MLIR、ONNX Runtime
- TileLang、tile DSL、schedule、pipeline、tensorization
- graph capture、operator fusion、layout transform、auto-tuning、lowering
- MegaKernel、persistent kernel、device-side scheduling、automatic kernel generation
- kernel profiling、roofline、SASS/PTX/IR inspection

## 关键问题

- 算子是 compute-bound 还是 memory-bound。
- 是否存在多余的 memory read/write、layout transform 或 kernel launch。
- Fusion 是否减少访存，是否引入寄存器压力和 occupancy 下降。
- Triton Kernel 或 TorchInductor 生成的 kernel 是否符合 workload shape。
- IR/lowering 层是否保留了必要语义，是否过早丢失优化空间。
- Tile、layout、pipeline 和 tensorization 是否真正适合目标硬件。
- MegaKernel 是否真的减少 launch 和 HBM round trip，还是只把问题合进一个更难 debug 的 kernel。
- 优化在不同 batch、sequence length、precision 下是否仍然成立。

## 专题入口

- [Attention 机制与计算模式](attention-computation-patterns.md)：区分 attention pattern、exact/approx、kernel 实现和 KV Cache 管理，理解 Dense/Sparse/Flash/PagedAttention、MHA/MQA/GQA、Prefill/Decode、长上下文、显存 IO 和 benchmark 方法。
- [Triton Kernel 编程](triton.md)：面向 AI workload 的手写 Kernel，覆盖 block program、JIT specialization、launch grid、tiling、资源模型、数值、autotune、PyTorch/Inductor 集成和端到端 benchmark。
- [TorchInductor 与 PyTorch 编译栈](torchinductor.md)：面向 `torch.compile`、TorchDynamo、FX/ATen 图、AOTAutograd、Inductor lowering/fusion/scheduler/codegen、guard/recompile、dynamic shape、CUDA Graph、AOTInductor、生产上线和性能调试。
- [MLIR 与 AI 编译 IR](mlir-ai-compiler-ir.md)：理解为什么 AI 编译需要多层 IR、dialect、lowering、bufferization、vector/GPU 后端，以及它和 Triton、TorchInductor、TileLang、MegaKernel 的关系。
- [TileLang：面向 AI Kernel 的 Tile 编程模型](tilelang.md)：理解 tile、layout、pipeline、tensorization、autotuning 如何共同决定 GEMM、Attention、Dequant GEMM、Sparse MM 等 kernel 的硬件效率。
- [MegaKernel、Persistent Kernel 与自动生成](megakernel-persistent-automatic-generation.md)：区分普通 fusion、persistent kernel、CUDA Graph 和 MegaKernel，理解 Triton MegaKernel、Ascend C MegaKernel-style 实现和自动生成路径的收益与风险。
