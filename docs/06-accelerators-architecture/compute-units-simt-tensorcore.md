---
title: 计算单元：SIMT、Tensor Core 与矩阵引擎
domain: accelerators-architecture
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-06-12
---

# 计算单元：SIMT、Tensor Core 与矩阵引擎

AI 加速器的“算力”来自计算单元。理解计算单元，不是为了背芯片规格，而是为了知道为什么同一个模型在不同 shape、dtype、batch、kernel 下性能差异巨大。

这一篇回答：

- SIMD、SIMT、warp 是什么。
- Tensor Core / Matrix Core / Systolic Array 为什么适合 AI。
- 为什么矩阵单元峰值很高，但很多 workload 吃不满。
- 为什么分支、稀疏、不规则访存、小矩阵会降低效率。
- 算子和编译器如何配合计算单元。

## 计算单元在系统里的位置

计算单元不是孤立存在的。它夹在上层 workload 和下层存储/互连之间：

```text
model layer
-> operator / graph
-> kernel / library call
-> thread block / warp / tile
-> matrix/vector/scalar units
-> registers / SRAM / cache / HBM
```

所以讨论“算力”时要同时问：

- workload 是否能分解成规则并行计算。
- kernel 是否能把计算映射到矩阵单元或向量单元。
- 数据是否能按正确 layout 和 tile 喂给计算单元。
- 寄存器、shared memory、cache、HBM 是否支撑得住。
- 编译器是否生成了目标硬件的高效指令。
- 端到端系统是否被 launch、通信、调度或 IO 限制。

这也是为什么同一块硬件上，大 GEMM 可能接近峰值，而 decode、embedding、MoE dispatch、sampling 可能远低于峰值。

## 执行层级：从 Grid 到 Lane

以 GPU 为例，可以把执行层级粗略理解为：

| 层级 | 含义 | 性能关注 |
| --- | --- | --- |
| grid | 一次 kernel launch 的全部工作 | launch 数量、全局并行度 |
| block / CTA | 被调度到 SM 的线程组 | block size、shared memory、同步 |
| warp | 一组一起发射指令的 threads | divergence、coalescing、latency hiding |
| thread / lane | 每个元素或小任务的执行上下文 | register、分支、地址计算 |
| instruction | 实际发射的算术、访存、矩阵指令 | issue rate、stall、pipeline |

Triton 的 `program instance`、CUDA 的 thread block、某些 NPU 的 tile program、TPU 的 XLA tile，都可以看成把大 workload 切成硬件可调度的小块。

关键问题是：

```text
切得太大 -> 资源占用高，occupancy 下降
切得太小 -> 计算单元吃不满，launch 和调度开销上升
```

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

## SIMD、SIMT 与 Masked Execution 的区别

SIMD 和 SIMT 都是“多数据并行”，但抽象层级不同。

| 维度 | SIMD | SIMT |
| --- | --- | --- |
| 程序员视角 | 明确写向量 lane 或由编译器向量化 | 写很多线程，每个线程像标量程序 |
| 硬件执行 | 一条向量指令控制多个 lane | 一组线程组成 warp/wavefront 同步发射 |
| 分支处理 | mask/predicate 控制 lane 是否生效 | warp divergence 或 predication |
| 常见场景 | CPU vector、NPU vector unit | GPU CUDA/HIP 风格模型 |

在 AI kernel 里，很多边界处理都依赖 mask：

```text
tile 内有效位置参与计算
tile 外无效位置被 mask 掉
```

mask 本身不是问题，但如果大量 lane 被 mask 掉，硬件仍可能消耗发射和调度资源。尾块、变长序列、ragged batch、MoE token imbalance 都可能出现这种浪费。

所以高性能实现常会做：

- padding 到硬件友好 shape。
- bucketing 让同组样本长度接近。
- packing/compaction 减少空 lane。
- block sparse 而不是完全随机 sparse。
- 对常见 shape 做 specialized kernel。

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

## Warp Scheduler 与 Latency Hiding

GPU 不是让一个 warp 从头跑到尾。SM 上通常有多个 resident warps，warp scheduler 会在它们之间切换。

直觉：

```text
warp A 等 HBM load
warp B 发射 Tensor Core 指令
warp C 做地址计算
warp D 做 store
```

这样可以隐藏内存和指令 pipeline latency。

但 latency hiding 需要足够的可调度工作。如果一个 kernel：

- resident warps 太少。
- 每个 warp 都在等同一个 long latency 事件。
- register/shared memory 占用太高导致 occupancy 低。
- 访存完全随机，cache miss 多。
- 小 shape 导致总 work 不足。

那么 scheduler 就没有足够工作可切换，SM 会空等。

这就是为什么 profiler 里要看 stall reason。常见 stall 方向包括：

- memory dependency。
- execution dependency。
- not selected。
- barrier/synchronization。
- instruction fetch/dispatch。
- tensor pipe not active。

不同 stall 指向不同优化方向，不应只看一个 utilization 数字。

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

## Register File、Shared Memory 与 Block 资源

每个 kernel 的 block 或 Triton program 会消耗 SM 资源：

| 资源 | 来自哪里 | 影响 |
| --- | --- | --- |
| registers | thread 局部变量、accumulator、地址、中间值 | 太多会降低 resident warps，甚至 spill |
| shared memory / SRAM | tile staging、software pipeline、跨 thread 协作 | 太多会降低 resident blocks |
| warps/threads | block size、num_warps | 影响并行度和调度 |
| barriers | thread/block 内同步 | 影响 pipeline 和 stall |
| instruction mix | load/store、matrix、vector、special function | 决定各 pipeline 压力 |

高性能 kernel 通常在几种资源之间取舍：

```text
更大 tile
-> 更高数据复用
-> 更多 accumulator/register
-> occupancy 可能下降

更多 shared memory staging
-> 更少 HBM 往返
-> resident blocks 可能下降
```

因此 occupancy 不是单独目标，而是资源模型的一个结果。

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

## Tensor Core 的执行直觉

Tensor Core 不是“更快的普通加法器”，而是专门执行小矩阵块乘加的单元。

概念上，它处理的是类似：

```text
acc_tile += a_tile @ b_tile
```

一个高性能 matmul kernel 会把大矩阵切成很多 tile：

```text
global memory A/B
-> load tile to register/shared memory
-> matrix instruction consumes fragments
-> accumulator tile in registers
-> epilogue
-> store C tile
```

关键对象：

| 对象 | 含义 |
| --- | --- |
| A/B fragments | 矩阵乘输入小块 |
| accumulator fragment | 累积输出小块，常用更高精度 |
| warp / warp group | 协作发射矩阵指令的线程组 |
| shared memory / SRAM | staging A/B tile，减少 HBM 重复读取 |
| epilogue | bias、activation、scale、quant、store 等后处理 |

NVIDIA CUDA guide 和 Hopper 架构资料都强调，Tensor Core 路径依赖特定数据类型、tile 形状和指令选择。Hopper 还引入了面向 warp group 的矩阵乘累加路径，用更大的线程协作粒度提高大矩阵吞吐。

对系统工程师来说，不必先记每条指令，但要记住：

```text
Tensor Core 利用率 = 数学上有 matmul + kernel 真的用矩阵指令 + 数据供应不断流
```

缺一项，峰值就只是纸面数字。

## Tile、Fragment 与 Accumulator

矩阵乘的输出 tile 通常由多个 K 维小块累积而来：

```text
for k_tile in K:
    acc += A_tile[:, k_tile] @ B_tile[k_tile, :]
```

这里的 `acc` 通常放在 register 中，直到 K 维累积完成才写回。

这带来几个取舍：

- 输出 tile 越大，accumulator 越多，register pressure 越高。
- K tile 越大，数据复用可能更好，但 shared memory/register 占用也更高。
- epilogue 融合越多，中间变量越多，可能降低 occupancy。
- accumulator dtype 越高，数值更稳，但资源占用和吞吐可能变化。

所以 matmul 调优经常围绕：

```text
BLOCK_M / BLOCK_N / BLOCK_K
num_warps
num_stages
layout
dtype
epilogue fusion
```

这些参数不是单纯软件细节，它们决定矩阵单元能否被连续喂饱。

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

### 数据供应不足

矩阵单元计算很快，常见瓶颈变成“数据送不过来”。

可能原因：

- HBM bandwidth 不够。
- L2/shared memory reuse 差。
- load/store 指令和矩阵指令 pipeline 没有 overlap。
- tile staging 不合理。
- shared memory bank conflict。
- register spill。
- scale/metadata 读取额外增加 bytes。

这类问题在 profiler 中可能表现为：

- Tensor Core utilization 低。
- memory throughput 高。
- warp stall memory dependency 高。
- achieved TFLOP/s 远低于峰值。

解决方向通常不是“再加 Tensor Core”，而是改善数据复用、layout、prefetch、pipeline 和 fusion 边界。

## 小矩阵效率曲线

矩阵单元有一个常见现象：矩阵越大越容易接近峰值，小矩阵效率快速下降。

可以这样理解：

```text
大矩阵:
  launch overhead 被摊薄
  tile 数多
  SM 都有活干
  数据复用高

小矩阵:
  tile 数少
  尾块比例高
  launch overhead 占比高
  Tensor Core tile 不饱满
```

AI 系统里小矩阵来自：

- decode 每次一个 token。
- micro-batch 太小。
- Tensor Parallel 过大。
- MoE expert token count 不均。
- LoRA/adapter 小 rank 矩阵。
- small batch rerank/embedding。

优化方向：

- batching / continuous batching。
- grouped GEMM。
- persistent kernel。
- CUDA Graph 或减少 launch。
- 对常见小 shape 做 specialized kernel。
- 改并行切分，避免把矩阵切碎。

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

## Systolic Array 与 Scratchpad

TPU 论文和 Cloud TPU 文档都强调 TPU 面向矩阵计算，核心思想是让数据在阵列中规则流动，并通过片上 memory/scratchpad 降低外部内存访问。

从系统角度看，systolic array 的优势来自：

- 阵列内部复用 A/B 数据。
- 控制流简单规则。
- 数据移动路径可预测。
- 对 dense matmul 能效高。

但它通常更依赖编译器/runtime：

- 需要把高层 graph 切成阵列友好的 tile。
- 需要安排数据进入 scratchpad。
- 需要处理 padding 和 shape 对齐。
- 需要把不规则 op 放到其它执行路径或 fallback。

所以 TPU/NPU/ASIC 的问题不是“有没有矩阵阵列”，而是：

```text
真实 workload 有多少比例能稳定映射到这个阵列？
剩下的 vector、control、gather/scatter、通信怎么处理？
```

## NPU / ASIC 矩阵引擎的常见取舍

专用 AI 加速器通常会围绕矩阵引擎做取舍。

| 设计方向 | 好处 | 风险 |
| --- | --- | --- |
| 更大矩阵阵列 | 高 dense GEMM 吞吐和能效 | 小 shape、不规则 workload 利用率低 |
| 更大片上 SRAM | 提高 tile 复用、降低 HBM traffic | 面积成本高，容量仍有限 |
| 更强低精度 | 提高 TOPS、降低 bytes | scale/accumulator/质量验证复杂 |
| 更多 vector/scalar 单元 | 支持 norm、softmax、routing | 面积和功耗分散 |
| 更灵活 gather/scatter | 支持 embedding、MoE、sparse | 控制复杂、带宽压力大 |
| 专用 attention/KV 单元 | 优化推理关键路径 | 泛化性和软件生态要求高 |

GPU 更通用，NPU/ASIC 更容易针对目标 workload 提高能效，但也更依赖软件栈和 workload 稳定性。

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

## Load/Store Unit 与地址计算

很多 AI kernel 的性能受 load/store 和地址计算限制，而不是受乘加限制。

典型场景：

- embedding lookup。
- KV Cache block table。
- paged attention。
- MoE dispatch/combine。
- packed sequence。
- masked attention。
- quant/dequant scale 读取。

这些场景需要大量地址计算：

```text
base_ptr + block_id * block_stride + offset
```

如果地址规则，硬件可以合并访问、cache 命中更高。

如果地址随机，问题会变成：

- memory transaction 增多。
- cache line 利用率低。
- warp 内 lane 等待不同地址。
- load imbalance。
- 指令中地址计算占比上升。

所以计算单元评估必须看 load/store pipeline。一个 kernel achieved TFLOP/s 低，不一定是算术单元差，也可能是 load/store 把算术单元饿住了。

## Special Function Unit

AI workload 也有一些非乘加操作：

- exp / log。
- rsqrt。
- sin/cos。
- activation 中的特殊函数。
- sampling 中的随机数和概率变换。

Softmax、normalization、某些 activation 会用到这些路径。

如果特殊函数占比高，Tensor Core peak 对它帮助有限。高性能 kernel 常会：

- 用近似实现。
- 做 fusion 减少读写。
- 避免重复计算。
- 把特殊函数限制在必要位置。

这解释了为什么 softmax、layernorm、sampling 不应该按矩阵峰值估算性能。

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

## 结构化稀疏和非结构化稀疏

稀疏加速要区分模式。

| 稀疏类型 | 硬件友好度 | 原因 |
| --- | --- | --- |
| N:M sparsity | 较友好 | 模式规则，硬件可固定跳过部分元素 |
| block sparse | 较友好 | 以 block 为单位，仍能用矩阵 tile |
| coarse-grained MoE | mixed | expert GEMM 规则，但 dispatch/AllToAll 不规则 |
| unstructured sparse | 较难 | index 多、访存乱、负载不均 |
| dynamic token pruning | 较难 | 每 batch pattern 变化 |

稀疏的收益公式不是：

```text
少 50% FLOPs -> 快 2x
```

而更像：

```text
节省的计算
- index/metadata 开销
- gather/scatter 开销
- 负载不均
- kernel launch 和调度开销
- 矩阵单元利用率下降
```

只有剩下的净收益为正，稀疏才会变快。

## Block Sparse Attention 的计算单元视角

Block sparse attention 把 attention 矩阵分成 block，只计算部分 block。

它比完全随机 sparse 更适合硬件，因为每个有效 block 内仍然是 dense matmul。

但难点包括：

- block mask metadata。
- 不同 query block 的有效 key block 数不同。
- load imbalance。
- block table 访问。
- softmax 需要跨有效 block 做归一化。
- backward 更复杂。

所以 block sparse attention 的性能取决于：

- block size 是否适合矩阵单元。
- sparsity pattern 是否足够规则。
- metadata 访问是否轻量。
- 调度是否能平衡 workload。
- softmax 和输出累积是否融合得好。

这再次说明：稀疏不是数学上少算就够，还要能组织成硬件友好的 tile。

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

## Dynamic Workload 的规整化

AI 系统常把动态 workload 变得更规则。

| 动态来源 | 规整化方法 |
| --- | --- |
| 变长文本 | bucketing、padding、packing |
| 多模态不同分辨率 | resize、patch bucketing、dynamic batch |
| MoE token 分布 | expert grouping、capacity、token sorting |
| RAG/Agent 请求差异 | 请求分级、batch by route、缓存 |
| Decode 动态 batch | continuous batching、paged KV、CUDA graph 分桶 |
| Sparse attention pattern | block sparse、固定 pattern、local/global block |

规整化的目的不是让算法变笨，而是让硬件看到更大的连续块、更少分支、更可预测的访存。

这是 runtime、compiler 和硬件共同完成的事情。

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

## 从 Transformer Layer 看计算单元映射

一个 Transformer layer 可以粗略拆成：

```text
QKV projection
-> attention QK
-> softmax / mask
-> attention PV
-> output projection
-> norm / residual
-> MLP up/gate/down
```

映射到计算单元：

| 部分 | 常见执行单元 | 瓶颈 |
| --- | --- | --- |
| QKV / output projection | Tensor Core / Matrix Engine | shape、dtype、layout |
| QK / PV | Tensor Core / Matrix Engine | sequence length、tile、attention kernel |
| softmax / mask | vector/SFU/load-store | HBM、特殊函数、fusion |
| norm / residual | vector/load-store | memory-bound、fusion |
| MLP | Tensor Core / Matrix Engine | 大 GEMM，通常较友好 |
| KV Cache read | load-store/cache | HBM bandwidth、layout、paging |

训练时 batch 和 token 数较大，projection/MLP 往往占主导，矩阵单元更容易吃满。

推理 decode 时，每步 token 少，KV Cache 读和小 GEMM 更明显，矩阵单元更难吃满。

## 并行切分会改变矩阵形状

并行策略会改变每张卡看到的矩阵。

例如 Tensor Parallel 把大矩阵切到多卡：

```text
原始 GEMM: [M, K] x [K, N]
TP 后每卡可能变成:
  [M, K] x [K, N/tp]
或
  [M, K/tp] x [K/tp, N]
```

TP 增大后，每卡矩阵可能变小。

好处：

- 单卡显存降低。
- 单卡计算量降低。
- 模型能扩到更多 GPU。

风险：

- 小矩阵效率下降。
- 通信更频繁。
- Tensor Core utilization 降低。
- launch overhead 增加。

MoE 的 Expert Parallel 也类似。每个 expert 看到的 token 数越少，expert GEMM 越小，矩阵单元越难吃满。

所以并行策略不是只看“能不能放下模型”，还要看切分后的 per-rank shape 是否仍然硬件友好。

## Kernel Fusion 对计算单元的影响

Fusion 可以减少 HBM 读写和 launch，但也会改变计算单元压力。

正向效果：

- 减少中间 tensor 写回。
- 减少小 kernel launch。
- 提高 arithmetic intensity。
- 把 epilogue 合并到 GEMM 后处理。

负向风险：

- register pressure 上升。
- shared memory 占用上升。
- occupancy 下降。
- Tensor Core pipeline 被额外 vector work 打断。
- 编译器生成代码更复杂。

例如 GEMM epilogue 融合 bias、activation、scale 通常有价值；但把太多复杂逻辑塞进主 kernel，可能让矩阵计算本身变慢。

因此 fusion 要用 profiler 验证，不是越多越好。

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

## Profiler 指标怎么连到优化动作

可以把 profiler 指标翻译成优化方向。

| 现象 | 可能原因 | 优先动作 |
| --- | --- | --- |
| Tensor Core utilization 低，memory throughput 高 | memory-bound | fusion、tiling、减少 HBM、低精度、layout |
| Tensor Core utilization 低，SM utilization 低 | shape 太小或 launch-bound | batching、grouped GEMM、CUDA graph、persistent kernel |
| occupancy 低，register spill 高 | tile/fusion 过大 | 减小 tile、拆 fusion、降低中间变量 |
| occupancy 低，shared memory 高 | staging 过重 | 调整 tile、num_stages、shared memory layout |
| L2 hit 低，memory transaction 多 | 访存不连续 | 改 layout、coalescing、重排数据 |
| warp divergence 高 | 分支/稀疏/变长 | bucketing、padding、compaction、block sparse |
| Tensor Core high 但端到端不快 | 其它路径瓶颈 | 看 norm、softmax、KV、通信、data pipeline |
| 通信暴露 | 并行策略或拓扑问题 | overlap、rank mapping、减少通信、调整 TP/EP/DP |

指标只提供线索，最终还是要回到端到端：

```text
step time
tokens/s
TTFT / TPOT
p99 latency
energy/token
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

### Split Shape Benchmark

还要测并行切分后的 shape。

例如：

- TP=1/2/4/8 后的 per-rank GEMM。
- MoE 每个 expert 的 token count 分布。
- micro-batch 改变后的 M 维。
- sequence parallel/context parallel 后的 attention shape。
- vocab parallel 后的 logits shape。

很多系统在单卡大 GEMM 上很好，但并行切分后变成大量中小 GEMM，矩阵单元效率明显下降。

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

## 硬件评估问题清单

评估一个计算单元设计时，可以问：

- dense GEMM 峰值是多少，真实 sustained GEMM 能到多少？
- 小 GEMM、batched GEMM、grouped GEMM 表现如何？
- BF16/FP16/FP8/INT8 路径分别是否成熟？
- accumulator 和 scale 机制是否满足训练/推理质量？
- vector/SFU/load-store 单元是否足够支撑 softmax/norm/routing？
- gather/scatter、block sparse、MoE dispatch 是否有硬件或软件支持？
- register/shared memory 容量是否能支撑目标 tile？
- 编译器能否稳定生成矩阵指令和高效 memory pipeline？
- profiler 是否能暴露 Tensor Core、stall、memory、occupancy 等指标？
- 多卡互连是否会让矩阵单元等待通信？

这些问题能把“硬件算力”拆成可验证的工程项。

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

### 误区六：矩阵单元越大越好

不一定。更大的矩阵阵列需要更高数据复用、更稳定 shape、更强 memory pipeline。目标 workload 如果小矩阵、不规则访问、动态控制流多，阵列可能利用率不高。

### 误区七：编译器能自动解决所有映射问题

不一定。编译器能做很多优化，但 shape、layout、dtype、parallelism、runtime 调度和硬件资源上限仍然约束最终性能。

## 设计检查清单

分析计算单元映射时，可以逐项确认：

- 主要算子能否映射到矩阵单元？
- dtype 是否匹配高吞吐路径？
- 矩阵 shape 是否足够大？
- 真实并行切分后的 per-rank shape 是否仍然足够大？
- TP/EP/PP 切分后 shape 是否变小？
- 是否存在大量小 GEMM？
- 是否存在大量 vector/SFU/load-store 主导的非矩阵路径？
- 是否有 warp divergence？
- 访存是否 coalesced？
- sparse pattern 是否结构化？
- MoE token 分布是否导致 grouped GEMM 负载不均？
- kernel 是否 launch-bound？
- Tensor Core utilization 是多少？
- achieved TFLOP/s 与峰值差距多少？
- memory bandwidth 是否打满？
- register/shared memory 是否限制 occupancy？
- 编译器是否生成预期 kernel？
- fusion 是否提高了数据复用，还是引入了 register/shared memory 压力？
- profiler 的 stall reason 是否支持你的瓶颈判断？
- 端到端 step time 或 TPOT 是否改善？

## 小结

计算单元决定 AI 加速器的算力来源，但真实性能取决于 workload 能不能把计算喂饱。

关键结论：

- SIMD/SIMT 适合规则数据并行，分支和不规则访存会浪费 lane。
- Warp 是 GPU 调度和性能分析的重要单位，warp scheduler 依赖足够 resident work 隐藏 latency。
- Tensor Core / Matrix Engine / Systolic Array 通过矩阵乘复用数据，提供高吞吐和高能效。
- Tensor Core 能否吃满，取决于 tile、dtype、layout、accumulator、数据供应和编译器指令选择。
- 大 GEMM 容易吃满矩阵单元，小 GEMM、decode、稀疏和动态 routing 更难。
- Vector、scalar、special function、load/store 单元决定 softmax、norm、routing、KV Cache、sampling 等非矩阵路径。
- 并行切分会改变每卡 shape，可能把大 GEMM 变成低效小 GEMM。
- Occupancy、Tensor Core utilization、memory bandwidth 和端到端时间必须一起看。
- 编译器和 kernel 实现决定高层模型是否真的走到硬件高效路径。

理解计算单元后，再看 Triton、TorchInductor、FlashAttention、MoE dispatch 和多卡并行，会更容易判断优化到底是在提高有效算力，还是只是在移动瓶颈。

## 参考资料

- [NVIDIA CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [NVIDIA: Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [Google Cloud: Introduction to Cloud TPU](https://cloud.google.com/tpu/docs/intro-to-tpu)
- [In-Datacenter Performance Analysis of a Tensor Processing Unit](https://arxiv.org/abs/1704.04760)
