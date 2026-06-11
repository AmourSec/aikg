---
title: 计算单元：SIMT、Tensor Core 与矩阵引擎
domain: accelerators-architecture
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-06-11
---

# 计算单元：SIMT、Tensor Core 与矩阵引擎

AI 加速器的“算力”来自计算单元。理解计算单元，不是为了背芯片规格，而是为了知道为什么同一个模型在不同 shape、dtype、batch、kernel 下性能差异巨大。

这一篇回答：

- SIMD、SIMT、warp 是什么。
- Tensor Core / Matrix Core / Systolic Array 为什么适合 AI。
- 为什么矩阵单元峰值很高，但很多 workload 吃不满。
- 为什么分支、稀疏、不规则访存、小矩阵会降低效率。
- 算子和编译器如何配合计算单元。

## 从标量计算到并行计算

最简单的 CPU 标量计算可以想成：

```text
一次执行一个或少量操作
```

AI workload 不同。它有大量同构计算：

- 矩阵乘。
- 向量加。
- softmax。
- normalization。
- embedding。
- attention。

这些计算可以并行。

并行的核心问题是：

```text
如何让大量计算单元同时做有用工作？
```

这就是 SIMD、SIMT 和矩阵引擎存在的原因。

## SIMD

SIMD 是 Single Instruction, Multiple Data。

意思是：

```text
同一条指令，同时处理多个数据元素
```

例如一条向量加法指令：

```text
[a0, a1, a2, a3] + [b0, b1, b2, b3]
-> [c0, c1, c2, c3]
```

SIMD 很适合规则向量计算。

优点：

- 控制开销低。
- 数据并行效率高。
- 适合 dense、规则、连续数据。

限制：

- 分支不一致会浪费 lane。
- 数据不连续会降低访存效率。
- 不规则稀疏访问不友好。

CPU AVX、某些 NPU vector unit、GPU 内部某些执行路径都可以用 SIMD 思想理解。

## SIMT

NVIDIA CUDA 常用 SIMT 概念：Single Instruction, Multiple Threads。

SIMT 看起来像很多线程各自写标量代码，但硬件会把一组线程组织起来一起执行。

CUDA 中，一个 warp 通常包含 32 个 threads。一个 warp 内线程执行同一条指令，但每个 thread 操作自己的数据。

可以理解为：

```text
thread 0: c[0] = a[0] + b[0]
thread 1: c[1] = a[1] + b[1]
...
thread 31: c[31] = a[31] + b[31]

warp executes these together
```

SIMT 的好处是编程模型比直接写 SIMD lane 更自然，同时硬件仍能批量执行。

## Warp

Warp 是 GPU 执行调度的重要单位。

一个 CUDA block 中有多个 threads，硬件把它们分成 warp 执行：

```text
block
  warp 0: thread 0-31
  warp 1: thread 32-63
  warp 2: thread 64-95
  ...
```

性能上，warp 有几个关键点。

### Warp Divergence

如果一个 warp 内线程走不同分支：

```python
if condition_per_thread:
    path_a()
else:
    path_b()
```

硬件通常需要分批执行不同路径。没有走当前路径的 lane 暂时空着。

结果：

- 控制流越分散，效率越低。
- 分支越依赖数据，越难优化。
- 不规则 sparse / routing / mask 可能引发 divergence。

这就是为什么 GPU 喜欢规则、同构计算。

### Memory Coalescing

Warp 内线程最好访问连续地址。

好模式：

```text
thread 0 reads x[0]
thread 1 reads x[1]
...
thread 31 reads x[31]
```

差模式：

```text
thread 0 reads x[random0]
thread 1 reads x[random1]
...
```

连续访问可以合并 memory transaction，提高带宽利用。随机访问会浪费带宽和 cache。

Embedding、scatter/gather、MoE dispatch、KV cache paging 等场景经常要处理这类问题。

## SM：Streaming Multiprocessor

NVIDIA GPU 由多个 SM 组成。每个 SM 包含：

- warp scheduler。
- CUDA cores。
- Tensor Cores。
- load/store units。
- registers。
- shared memory / L1。
- special function units。

一个 kernel launch 后，thread blocks 会被分配到 SM 上执行。

优化 GPU kernel 时，经常关注：

- 每个 block 用多少 threads。
- 每个 thread 用多少 registers。
- 每个 block 用多少 shared memory。
- 每个 SM 能同时驻留多少 blocks / warps。
- occupancy。
- Tensor Core utilization。
- memory pipeline。

SM 是很多 GPU 性能问题的基本观察单位。

## Occupancy

Occupancy 指 SM 上活跃 warps 占最大可支持 warps 的比例。

高 occupancy 有助于隐藏 latency：

```text
warp A 等内存
warp B 继续计算
warp C 发起 load
...
```

但 occupancy 不是越高越好。

低 occupancy 可能由：

- registers 太多。
- shared memory 太多。
- block size 太大。
- 每个 program 占用资源太多。

高 occupancy 也可能不快：

- 算子 memory-bound。
- Tensor Core 没吃满。
- instruction mix 不好。
- 数据复用差。

优化时要看 occupancy、SM utilization、memory bandwidth、Tensor Core utilization 和端到端时间，不要只追一个指标。

## Tensor Core / Matrix Core

AI 模型的大部分计算来自矩阵乘：

```text
C = A @ B
```

普通 CUDA core 可以做标量/向量运算，但矩阵乘可以由专门矩阵单元更高效完成。

NVIDIA Tensor Core、AMD Matrix Core、TPU systolic array、各种 NPU matrix engine 都是这个方向。

它们通常执行类似：

```text
D = A * B + C
```

也就是 matrix multiply-accumulate。

优势：

- 吞吐高。
- 能效高。
- 适合 FP16/BF16/FP8/INT8 等低精度。
- 对 Transformer 的 Linear/Attention/MLP 特别重要。

限制：

- shape 要合适。
- dtype 要匹配。
- alignment 要满足。
- 数据要按合适 layout 进入。
- 小矩阵难吃满。

## 为什么矩阵单元峰值很高

矩阵乘有很高的数据复用。

例如 A 的一个元素可以和 B 的多个元素相乘，B 的一个元素也会被多个输出位置使用。

如果 tile 设计合理：

```text
load A tile once
load B tile once
do many multiply-accumulate
store C tile once
```

同样的数据搬运可以支撑很多 FLOPs。

这就是高 arithmetic intensity，也是矩阵单元能接近峰值的基础。

## 为什么很多 workload 吃不满矩阵单元

### 矩阵太小

Decode 阶段经常 batch 小、token 数少。GEMM 的 M 维可能很小。

结果：

- Tensor Core tile 不饱满。
- kernel launch overhead 占比高。
- occupancy 低。
- memory latency 难隐藏。

### 切分太碎

Tensor Parallel 过大、MoE expert token 分布不均、micro-batch 太小，都可能把大 GEMM 切成小 GEMM。

小 GEMM 可能导致：

- TFLOPS 下降。
- kernel 数量增加。
- launch overhead 上升。
- L2/cache 复用变差。

### dtype 不匹配

如果代码看起来是 BF16/FP16，但某些 op 实际走 FP32 或不支持低精度路径，就不能用到高峰值。

需要检查：

- autocast。
- kernel dtype。
- accumulator dtype。
- scale / quantization。
- library 是否支持该 dtype。

### layout 不匹配

矩阵单元喜欢规则 layout。

如果输入需要频繁 transpose、contiguous、pack/unpack，就会把收益吃掉。

## Systolic Array

TPU 的核心计算思想常用 systolic array 解释。

它像一个二维阵列：

```text
A 数据从左流入
B 数据从上流入
每个计算单元做乘加
partial sum 在阵列中移动
```

这种结构适合矩阵乘，因为数据能在阵列内部被多次使用。

优点：

- 控制规则。
- 数据复用高。
- 能效高。
- 对 dense GEMM 友好。

代价：

- shape/layout 需要适配阵列。
- 控制流不规则时效率差。
- sparse、gather/scatter、动态 shape 不一定适合。
- 编译器/runtime 要把计算切成合适 tile。

GPU Tensor Core 和 TPU systolic array 不是同一实现，但共同点是：都希望 workload 变成规则矩阵计算。

## Vector Unit 与 Scalar Unit

除了矩阵单元，AI 加速器仍需要处理：

- elementwise。
- activation。
- normalization。
- softmax。
- mask。
- index。
- sampling。
- routing。
- control logic。

这些不一定适合矩阵单元。

如果一个模型只有 GEMM 快，但 softmax、norm、routing、KV cache、sampling 很慢，端到端仍然不快。

所以现代加速器需要组合：

```text
matrix engine
vector unit
scalar/control unit
load/store unit
memory hierarchy
interconnect
```

只强化矩阵单元，不能解决所有 AI workload。

## 稀疏和不规则计算

稀疏计算理论上减少 FLOPs，但硬件上不一定更快。

原因：

- index 读取增加。
- 访存不连续。
- warp divergence。
- load imbalance。
- small block 太多。
- 不能高效用矩阵单元。

结构化稀疏更容易加速，例如 block sparse、N:M sparsity。

非结构化稀疏更难，因为计算被打碎。

MoE 是典型例子：

- 每个 token 只去少数 expert，计算上是稀疏激活。
- 但系统上要做 routing、dispatch、AllToAll、grouped GEMM、combine。
- 如果 expert token 数不均，小 GEMM 和通信会拖慢。

所以稀疏能不能快，取决于稀疏模式能否映射成硬件友好的 block。

## 分支和动态控制流

GPU/NPU 喜欢规则执行。

动态控制流会带来：

- warp divergence。
- 编译器难以优化。
- graph capture 难。
- kernel fusion 难。
- load imbalance。

典型场景：

- variable length sequence。
- conditional computation。
- dynamic routing。
- agent tool loop。
- variable image/video resolution。
- sparse attention pattern。

系统优化通常会做：

- bucketing。
- padding。
- grouping。
- sorting。
- compaction。
- mask fusion。
- specialized kernels。

目标不是消除动态性，而是把动态性组织成硬件可执行的规则块。

## 计算单元和编译器的关系

硬件峰值需要编译器把程序映射到正确指令。

编译器/runtime 负责：

- 选择矩阵指令。
- 选择 tile。
- 分配 register。
- 使用 shared memory。
- 安排 load/store。
- 做 fusion。
- 选择 layout。
- 插入同步。
- 生成低精度路径。

Triton、TorchInductor、XLA、TVM、MLIR、CUTLASS 等工具，本质都在做这件事。

如果编译器没把 matmul 映射到 Tensor Core，硬件峰值再高也没用。

如果编译器 fusion 过度导致 register spill，kernel 也可能变慢。

硬件和编译器必须一起看。

## 训练与推理中的计算单元压力

### 训练

训练中大矩阵多：

- forward GEMM。
- backward activation gradient GEMM。
- backward weight gradient GEMM。
- optimizer。

训练通常更容易吃满矩阵单元，因为 batch 和 token 数更大。

但训练也有挑战：

- activation memory。
- optimizer state。
- all-reduce/reduce-scatter。
- mixed precision 稳定性。
- checkpointing 重计算。

### 推理 Prefill

Prefill 处理 prompt，通常一次处理多个 token。

它包含较大的 GEMM 和 attention：

- QKV projection。
- QK。
- softmax。
- PV。
- MLP。

Prefill 相对容易用到矩阵单元，但长上下文会增加 attention 和 KV 相关 IO。

### 推理 Decode

Decode 每次生成一个 token。

特点：

- batch 可能动态。
- GEMM M 维小。
- 读大量 KV Cache。
- latency 重要。
- launch overhead 重要。

Decode 更容易 memory-bound 或 launch-bound，而不是 compute-bound。

这就是为什么同一块硬件，Prefill 和 Decode 的优化策略不同。

## 如何判断矩阵单元是否吃满

指标包括：

- Tensor Core utilization。
- achieved TFLOP/s。
- SM utilization。
- occupancy。
- instruction mix。
- warp stall reason。
- memory throughput。
- kernel shape。
- p50/p99 latency。

工具：

- Nsight Compute。
- Nsight Systems。
- PyTorch Profiler。
- vendor profiler。
- kernel benchmark。

要结合看：

```text
Tensor Core utilization 高 + memory bandwidth 低
  -> 可能 compute-bound

Tensor Core utilization 低 + memory bandwidth 高
  -> 可能 memory-bound

二者都低 + kernel 很多很短
  -> 可能 launch-bound / shape 太小

Tensor Core utilization 低 + branch/warp stall 高
  -> 可能 divergence / 不规则访问
```

## Benchmark 设计

分析计算单元时，至少测三类。

### Peak GEMM

测硬件在理想 GEMM 下的上限。

关注：

- BF16/FP16/FP8/INT8。
- large M/N/K。
- Tensor Core 使用。
- TFLOP/s。

### Real Shape GEMM

测模型真实 shape。

例如：

- batch。
- sequence length。
- hidden size。
- intermediate size。
- TP 切分后 shape。
- MoE expert token count。

真实 shape 比峰值 GEMM 更有意义。

### End-to-End Kernel Mix

测完整层或完整 step：

- Linear。
- Attention。
- Softmax。
- LayerNorm。
- MLP。
- Communication。
- KV Cache。

端到端结果决定真实价值。

## 常见误区

### 误区一：有 Tensor Core 就一定快

不一定。要看 kernel 是否使用 Tensor Core，shape 是否合适，数据是否及时送到。

### 误区二：GPU 线程越多越好

不一定。线程多但寄存器压力高、访存不连续、warp divergence 多，效率仍然低。

### 误区三：稀疏一定省算力也省时间

不一定。稀疏可能减少数学 FLOPs，但增加 index、访存、调度和负载均衡成本。

### 误区四：Occupancy 高就说明 kernel 好

不一定。要看 Tensor Core、memory bandwidth、stall 和端到端时间。

### 误区五：训练快的硬件推理一定快

不一定。训练通常大 GEMM 多，decode 推理小 batch 和 KV Cache 读写多，瓶颈不同。

## 设计检查清单

分析计算单元映射时，可以逐项确认：

- 主要算子能否映射到矩阵单元？
- dtype 是否匹配高吞吐路径？
- 矩阵 shape 是否足够大？
- TP/EP/PP 切分后 shape 是否变小？
- 是否存在大量小 GEMM？
- 是否有 warp divergence？
- 访存是否 coalesced？
- sparse pattern 是否结构化？
- kernel 是否 launch-bound？
- Tensor Core utilization 是多少？
- achieved TFLOP/s 与峰值差距多少？
- memory bandwidth 是否打满？
- register/shared memory 是否限制 occupancy？
- 编译器是否生成预期 kernel？
- 端到端 step time 或 TPOT 是否改善？

## 小结

计算单元决定 AI 加速器的算力来源，但真实性能取决于 workload 能不能把计算喂饱。

关键结论：

- SIMD/SIMT 适合规则数据并行，分支和不规则访存会浪费 lane。
- Warp 是 GPU 调度和性能分析的重要单位。
- Tensor Core / Matrix Engine / Systolic Array 通过矩阵乘复用数据，提供高吞吐和高能效。
- 大 GEMM 容易吃满矩阵单元，小 GEMM、decode、稀疏和动态 routing 更难。
- Occupancy、Tensor Core utilization、memory bandwidth 和端到端时间必须一起看。
- 编译器和 kernel 实现决定高层模型是否真的走到硬件高效路径。

理解计算单元后，再看 Triton、TorchInductor、FlashAttention、MoE dispatch 和多卡并行，会更容易判断优化到底是在提高有效算力，还是只是在移动瓶颈。

## 参考资料

- [NVIDIA CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [NVIDIA: Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [Google Cloud: Introduction to Cloud TPU](https://cloud.google.com/tpu/docs/intro-to-tpu)
- [In-Datacenter Performance Analysis of a Tensor Processing Unit](https://arxiv.org/abs/1704.04760)
