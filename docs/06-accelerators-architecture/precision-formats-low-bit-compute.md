---
title: 精度格式：FP16、BF16、FP8 与量化计算
domain: accelerators-architecture
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-06-12
---

# 精度格式：FP16、BF16、FP8 与量化计算

AI 加速器性能和精度格式强相关。很多芯片的峰值算力会按 FP32、TF32、FP16、BF16、FP8、INT8、INT4 分开标注。低精度通常更快、更省显存、更省带宽，但也更容易带来数值误差、溢出、下溢和模型质量下降。

核心问题不是“精度越低越好”，而是：

> 哪些 tensor 可以低精度存储？哪些计算可以低精度执行？哪些累积必须高精度？scale 如何管理？硬件是否真的走到了对应低精度高吞吐路径？

这篇从硬件和系统角度解释常见精度格式。

## 精度格式在系统里的位置

精度格式不是一个孤立的模型参数。它会同时进入五条路径：

| 路径 | 典型问题 |
| --- | --- |
| 模型数学 | attention、MLP、normalization、loss、optimizer 对误差是否敏感 |
| Kernel dispatch | 框架是否选择 Tensor Core、Matrix Engine、INT dot product 等高吞吐 kernel |
| 存储与搬运 | 权重、activation、gradient、KV Cache、temporary buffer 占多少 HBM 和带宽 |
| 分布式通信 | AllReduce、ReduceScatter、AllGather、AllToAll、KV transfer 用什么 dtype |
| Artifact 管理 | checkpoint、量化 scale、amax history、calibration 数据和部署配置如何保存 |

所以一个成熟的低精度方案应该写成协议，而不是一句“用 FP8”：

```text
storage dtype
compute input dtype
accumulation dtype
output dtype
communication dtype
scale metadata
fallback op policy
checkpoint metadata
```

只有这些都明确，才知道系统是否真的在省显存、省带宽、提吞吐，并且没有把数值风险转移到不可观测的位置。

## 精度格式影响什么

精度格式会影响四类成本：

| 成本 | 影响 |
| --- | --- |
| 计算吞吐 | 低精度矩阵单元通常峰值更高 |
| HBM 容量 | 每个元素 bytes 更少，能放更多权重、activation、KV Cache |
| HBM 带宽 | 同样 bandwidth 下能搬更多元素 |
| 网络通信 | gradients、activations、KV transfer 更小 |

也会影响四类风险：

| 风险 | 含义 |
| --- | --- |
| precision loss | 有效数字太少，误差变大 |
| overflow | 数值超过最大可表示范围 |
| underflow | 小数太小，被冲成 0 或 subnormal |
| quantization error | 映射到低比特时产生离散误差 |

所以低精度是系统优化，也是数值工程。

## 位宽省下来的不只是容量

把元素从 16 bit 降到 8 bit，直观上是同样数量的值只占一半空间。但系统收益不止容量：

- HBM 同一时间能搬更多元素。
- L2/cache/SRAM 能容纳更多有效 tile。
- KV Cache、activation、temporary buffer 的驻留压力下降。
- 网络通信 payload 变小。
- 某些硬件矩阵单元对低精度提供更高峰值吞吐。
- 同样显存下可以放更大 batch、更多并发请求或更长上下文。

但位宽下降也会引入额外成本：

- scale、zero point、amax history 也要存。
- 量化和反量化会消耗 kernel 时间。
- packing/unpacking 会影响访存对齐。
- 某些 op 需要 cast 回 BF16/FP32。
- 更细粒度 scale 会让 kernel 更复杂。

因此低精度的收益要看端到端关键路径，而不是只看元素 bytes。

## 先区分存储、计算和累积

很多人说“用 FP16 训练”或“用 FP8 推理”，但这句话不够精确。

应该区分：

- 权重用什么 dtype 存。
- activation 用什么 dtype 存。
- GEMM 输入用什么 dtype。
- GEMM accumulator 用什么 dtype。
- reduction 用什么 dtype。
- gradient 用什么 dtype。
- optimizer state 用什么 dtype。
- master weight 是否保留 FP32。
- 通信时用什么 dtype。
- checkpoint 保存什么 dtype。

例如一个常见混合精度训练路径：

```text
weights: BF16
activations: BF16
matmul input: BF16
matmul accumulation: FP32 or higher internal precision
optimizer states: FP32
master weights: maybe FP32
```

这不是“全 BF16”。它是混合精度。

## 浮点格式的三个部分

浮点数通常由三部分组成：

| 部分 | 作用 | 直觉 |
| --- | --- | --- |
| sign | 正负号 | 这个数是正还是负 |
| exponent | 指数 | 能表示多大或多小的数，也就是动态范围 |
| mantissa / significand | 尾数 | 在某个数量级附近能分多细，也就是有效精度 |

动态范围和有效精度不是一回事。

BF16 和 FP16 都是 16 bit，但设计取舍不同：

- BF16 保留了接近 FP32 的 exponent 位数，所以范围大，不容易 overflow。
- FP16 给 mantissa 更多位，所以在可表示范围内更细，但范围小，训练更容易遇到 overflow/underflow。

这也是为什么大模型训练常偏向 BF16：训练过程里梯度、activation、loss scale 的数值范围变化很大，范围不够会比尾数少几位更麻烦。

## 常见格式速查

| 格式 | 大致位宽 | 类型 | 常见位置 | 主要优势 | 主要风险 |
| --- | --- | --- | --- | --- | --- |
| FP32 | 32 bit | 浮点 | baseline、optimizer state、敏感 reduction | 稳定、范围和精度都较好 | 成本高、矩阵吞吐低 |
| TF32 | 约 FP32 输入路径 | Tensor Core 计算路径 | FP32 matmul 加速 | 少改代码获得 Tensor Core 加速 | 不是通用存储格式，精度低于 FP32 |
| FP16 | 16 bit | 浮点 | 训练/推理 GEMM、activation | 吞吐高、支持广 | 动态范围小，训练常需 loss scaling |
| BF16 | 16 bit | 浮点 | 大模型训练默认低精度 | 范围接近 FP32，训练稳定 | 尾数少，敏感 op 仍需高精度 |
| FP8 E4M3 | 8 bit | 浮点 | weight/activation GEMM 输入 | 比 FP16 更省带宽，精度相对 E5M2 更好 | 范围较小，需要 scale |
| FP8 E5M2 | 8 bit | 浮点 | gradient 或范围更大的值 | 范围比 E4M3 大 | 有效精度更低 |
| INT8 | 8 bit | 整数 | 推理 weight/activation | 成熟、吞吐高、容量低 | calibration、outlier、dequant 开销 |
| INT4 / NF4 / FP4 | 4 bit | 整数或低比特浮点 | weight-only、QLoRA、实验性训练/推理 | 权重容量大幅下降 | 误差大，scale 和 kernel 成本高 |

这张表只给直觉。实际性能取决于硬件、kernel、shape、layout、scale 粒度和端到端瓶颈。

## FP32

FP32 是经典单精度浮点。

特点：

- 动态范围大。
- 精度较高。
- 数值稳定。
- 存储和带宽成本高。
- AI 矩阵吞吐通常低于低精度路径。

FP32 常用于：

- 参考 baseline。
- 某些敏感 reduction。
- optimizer master weight。
- optimizer state。
- loss / metric。
- 调试数值问题。

现代 AI 训练通常不会全 FP32，因为成本太高。但 FP32 仍然是很多关键状态和累积的稳定锚点。

## TF32

TF32 是 NVIDIA Ampere 以后常见的 Tensor Core 格式路径，用来加速 FP32 风格矩阵计算。

直觉：

- 输入动态范围接近 FP32。
- mantissa 精度少于 FP32。
- 使用 Tensor Core 加速。

TF32 常用于：

- 不想改代码但希望 FP32 matmul 更快。
- 训练早期基线。
- 对精度要求高但又想获得部分 Tensor Core 加速的场景。

它不是存储格式的主线，更多是计算路径。

## FP16

FP16 是半精度浮点，16 bit。

优点：

- 存储减半。
- 带宽需求减半。
- 矩阵单元吞吐高。
- 推理和训练都广泛支持。

缺点：

- 动态范围小。
- 容易 overflow/underflow。
- 训练常需要 loss scaling。
- 某些 reduction 和 normalization 不稳定。

FP16 训练经常配合：

- autocast。
- GradScaler。
- FP32 master weights。
- FP32 optimizer states。
- 敏感 op 保持 FP32。

PyTorch AMP 文档也强调 `autocast` 会为不同 op 选择合适 dtype，而 `GradScaler` 负责缓解 FP16 gradient underflow。

## BF16

BF16 也是 16 bit，但设计重点和 FP16 不同。

直觉：

```text
BF16: 动态范围接近 FP32，但有效精度更低
FP16: 有效精度比 BF16 多一些，但动态范围小很多
```

BF16 的优势：

- 动态范围大，不太需要 loss scaling。
- 训练稳定性通常好于 FP16。
- 存储和带宽仍然是 16 bit。
- 现代 GPU/TPU/NPU 支持成熟。

缺点：

- mantissa 少，精度有限。
- 某些数值敏感计算仍需 FP32。
- 不同硬件 BF16 路径性能不同。

大模型训练中，BF16 常作为默认训练精度，因为它比 FP16 更省心。

## FP16 与 BF16 怎么选

| 维度 | FP16 | BF16 |
| --- | --- | --- |
| 动态范围 | 小 | 接近 FP32 |
| 有效精度 | 比 BF16 多 | 比 FP16 少 |
| loss scaling | 常需要 | 通常不需要 |
| 稳定性 | 更敏感 | 更稳 |
| 硬件支持 | 很广 | 新硬件更好 |
| 常见训练 | 可以，但要调 scaler | LLM 训练常用 |

经验上：

- 现代大模型训练优先 BF16。
- 老硬件或特定推理路径可能用 FP16。
- FP16 需要更认真处理 overflow/underflow。
- 无论 FP16/BF16，reduction、optimizer、loss 仍要关注高精度。

## FP8

FP8 是 8 bit 浮点。它比 FP16/BF16 更省带宽和显存，也可能获得更高矩阵吞吐。

常见 FP8 格式包括：

- E4M3：更多 mantissa，范围较小。
- E5M2：范围更大，精度较低。

直觉：

| 格式 | 特点 |
| --- | --- |
| E4M3 | 更适合 activation/weight 等需要精度的值 |
| E5M2 | 更适合 gradient 等范围更大的值 |

实际选择依赖硬件和框架。

NVIDIA Transformer Engine 文档把 FP8 训练作为低精度训练能力之一，并提供 delayed scaling、current scaling、block scaling、MXFP8、NVFP4 等 recipe。重点不是“把 tensor cast 成 FP8”，而是整个训练路径要管理 scale、amax、transpose、分布式通信和 checkpoint。

更具体地看，FP8 的难点在三件事：

1. 格式本身太小，必须把数值分布压到可表示范围内。
2. 不同 tensor 的分布差异很大，不能指望一个全局 scale 解决所有问题。
3. 训练会不断改变分布，scale 也要随时间更新。

所以 FP8 更像“低精度计算体系”，而不是一个简单 dtype。

## FP8 E4M3 与 E5M2 怎么理解

E4M3 和 E5M2 的名字可以拆开看：

```text
E4M3 = 4 bit exponent + 3 bit mantissa
E5M2 = 5 bit exponent + 2 bit mantissa
```

直觉是：

- exponent 多，能表示的数值范围更大。
- mantissa 多，在同一数量级附近分得更细。

因此：

| 场景 | 更常见选择 | 原因 |
| --- | --- | --- |
| weight | E4M3 | 权重分布相对可控，更需要有效精度 |
| activation | E4M3 或按 recipe 选择 | activation 有 outlier，但也需要精度 |
| gradient | E5M2 更常见 | gradient 范围变化更大 |
| 中间敏感 op | BF16/FP32 | FP8 误差可能影响稳定性 |

这不是固定规则。真正的规则来自 recipe、硬件实现和质量验证。

## FP8 Scale 的粒度

Scale 粒度决定一个 scale 覆盖多少元素。

| 粒度 | 优点 | 缺点 | 常见理解 |
| --- | --- | --- | --- |
| per tensor | metadata 最少，简单 | 容易被一个 outlier 影响整个 tensor | 粗粒度、便宜但误差大 |
| per channel | 对通道分布更友好 | scale 数量增加 | 权重/activation 常见 |
| per group | 在精度和成本之间折中 | 需要 group 设计和 kernel 支持 | 低比特权重量化常见 |
| per block | 更局部，能处理分布不均 | metadata 和访存更复杂 | FP8 block scaling、MX 类格式常见 |

粒度越细，量化误差通常越小，但系统成本越高：

- scale metadata 更多。
- scale 读取会占 cache 和 bandwidth。
- kernel 要在计算中加载、广播或应用 scale。
- checkpoint 和通信要保存更多附加状态。
- 编译器和 runtime 更难 fusion。

所以 scale 粒度不是越细越好，要和硬件访存、kernel layout、质量目标一起选。

## FP8 为什么需要 Scale

FP8 只有 8 bit，表示范围和精度都有限。

为了让 tensor 的数值落入可表示范围，通常需要 scale：

```text
real_value ~= fp8_value * scale
```

或者：

```text
fp8_value ~= real_value / scale
```

Scale 可以按不同粒度管理：

- per tensor。
- per channel。
- per block。
- per group。

粒度越细：

- 表示更准确。
- metadata 更多。
- kernel 更复杂。
- 通信和存储更麻烦。

粒度越粗：

- 简单。
- 但容易被 outlier 影响。

## Amax 与 Delayed Scaling

FP8 训练常用 amax 统计：

```text
amax = max(abs(tensor))
```

通过历史 amax 估计 scale。

Delayed scaling 的思路是：

- 记录一段历史 amax。
- 根据历史最大值或策略选择 scale。
- 下一次或后续使用这个 scale 量化。

好处：

- scale 比较稳定。
- 不需要每次立即精确同步所有统计。

风险：

- 分布变化快时 scale 滞后。
- outlier 可能拉大 scale，降低有效精度。
- 分布式训练中 amax 同步和一致性要处理。

除了 delayed scaling，实际系统还可能使用 current scaling 或 block scaling。

| 方法 | 思路 | 优点 | 风险 |
| --- | --- | --- | --- |
| current scaling | 用当前 tensor 统计直接决定 scale | 适应快 | 统计和同步成本更高 |
| delayed scaling | 用历史 amax 决定下一次 scale | 稳定、工程上常用 | 分布突变时滞后 |
| block scaling | 每个 block 使用局部 scale | 精度更好，抗 outlier | metadata 和 kernel 更复杂 |

如果分布式训练里每个 rank 的 scale 不一致，还会引入更隐蔽的问题：同一个逻辑 tensor 在不同 rank 上被不同规则量化，后续 collective、checkpoint resume 和数值排查都会变复杂。

## FP8 不等于全模型 FP8

实际 FP8 训练/推理通常是混合的。

可能：

- GEMM 输入 FP8。
- accumulation 更高精度。
- 某些 activation BF16。
- normalization BF16/FP32。
- softmax BF16/FP32。
- optimizer state FP32/BF16。
- master weight BF16/FP32。

敏感部分不一定适合 FP8：

- LayerNorm / RMSNorm。
- Softmax。
- loss。
- small reduction。
- optimizer state。
- 某些 attention projection。
- logits。

所以评估 FP8 要看：

```text
哪些 GEMM 真的 FP8？
哪些 op 回退 BF16/FP32？
回退是否成为瓶颈？
质量是否稳定？
```

## 端到端低精度数据流

低精度 GEMM 的端到端路径通常长这样：

```text
低精度存储
  -> 读取 packed / quantized tensor
  -> 读取 scale / zero point / metadata
  -> dequant 或在矩阵单元内部应用 scale
  -> 低精度输入执行 matmul
  -> 高精度 accumulator 累积
  -> epilogue: bias / activation / residual / requant / cast
  -> 以目标 dtype 写回 HBM 或 cache
```

每一步都可能成为瓶颈：

- 如果读取 scale 打乱连续访存，带宽收益会下降。
- 如果 dequant 单独占一个 kernel，launch 和 HBM 往返会抵消收益。
- 如果 epilogue 不能 fusion，低精度 GEMM 后还要多次读写。
- 如果输出又马上 cast 回高精度，显存节省可能只发生在一小段生命周期。
- 如果 shape 不满足 Tensor Core 或矩阵引擎对齐要求，硬件峰值用不上。

因此低精度优化要画真实数据流，而不是只比较 dtype 位宽。

## 量化方法类型

量化不只有一种。

| 方法 | 含义 | 常见场景 | 风险点 |
| --- | --- | --- | --- |
| PTQ | Post-Training Quantization，训练后量化 | 已有模型转 INT8/INT4 推理 | calibration 数据不匹配会掉质量 |
| QAT | Quantization-Aware Training，训练中模拟量化误差 | 对质量敏感的部署 | 训练成本更高，工程链路更复杂 |
| Dynamic Quantization | 运行时按输入动态决定量化参数 | activation 分布变化大时 | 运行时统计和量化开销 |
| Weight-only Quantization | 只量化权重，activation 保持 FP16/BF16 | LLM 推理常见 INT4/INT8 权重 | Decode 可能仍受 KV Cache 限制 |
| W8A8 | weight 和 activation 都 INT8 | 高吞吐推理 | activation outlier 和 calibration 更关键 |
| W4A16 / W4A8 | 权重 4 bit，activation 16/8 bit | 大模型压显存 | kernel、scale、质量折中明显 |
| KV Cache Quantization | 量化 K/V cache | 长上下文和高并发推理 | 长程 attention 质量和 dequant 开销 |

训练、推理、微调、KV Cache 的量化目标不同，不能把一个 recipe 直接搬到另一个场景。

## INT8

INT8 是推理量化中非常常见的格式。

它不是浮点，而是整数。

基本思想：

```text
real_value ~= scale * (int8_value - zero_point)
```

常见量化粒度：

- per tensor。
- per channel。
- per group。

INT8 的收益：

- 权重更小。
- 带宽更低。
- 矩阵乘吞吐高。
- 推理成本低。

风险：

- calibration 不好会掉质量。
- activation outlier 影响大。
- 某些 op 不适合 INT8。
- dequant/quant overhead 可能抵消收益。
- kernel 支持和 layout 很重要。

INT8 还要区分对称量化和非对称量化：

| 类型 | 公式直觉 | 特点 |
| --- | --- | --- |
| 对称量化 | `real_value ~= scale * int_value` | 简单，硬件友好，常用于权重 |
| 非对称量化 | `real_value ~= scale * (int_value - zero_point)` | 能处理分布偏移，但多一个 zero point |

对 LLM 来说，activation outlier 很常见。一个小比例的大值就可能拉大 scale，让多数普通值的有效刻度变少。很多 INT8 方案的核心都不是“怎么把数变成 int8”，而是“怎么让 outlier 不毁掉大多数值的表示精度”。

## INT4 / FP4

INT4、FP4、NF4、NVFP4 等 4 bit 格式更激进。

收益：

- 权重容量大幅降低。
- HBM 带宽压力降低。
- 可支持更大模型或更高并发。

风险：

- 精度损失更明显。
- 需要更细粒度 scale。
- activation 量化更难。
- kernel 更复杂。
- metadata 和 packing/unpacking 成本更高。

4 bit 常见于：

- 权重量化推理。
- LoRA / QLoRA 类训练或微调。
- KV Cache 或 activation 的实验性低比特路径。
- 新一代硬件支持的 FP4/NVFP4 训练探索。

需要区分：

```text
weight-only quantization
weight + activation quantization
training with low-bit compute
low-bit storage but higher precision compute
```

这些不是同一个问题。

### NF4 与 QLoRA 的直觉

NF4 常和 QLoRA 一起出现。它的直觉不是“用普通整数均匀切分数轴”，而是用更适合神经网络权重分布的 4 bit 表示来存基础模型权重；计算时再反量化到更高精度参与前向和反向。

QLoRA 这类方法通常关注微调成本：

- 冻结大部分基础模型权重。
- 基础权重以 4 bit 形式存储。
- 训练 LoRA adapter。
- optimizer state 主要落在小量 adapter 参数上。

所以它主要解决“低成本微调和显存占用”，不等价于“全模型 4 bit 训练”。

## Accumulator 精度

矩阵乘通常不是用输入 dtype 累积到底。

例如：

```text
FP16 input * FP16 input -> FP32 accumulation
BF16 input * BF16 input -> FP32 accumulation
FP8 input * FP8 input -> higher precision accumulation
INT8 input * INT8 input -> INT32 accumulation
```

Accumulator 决定误差是否会在长 reduction 中积累。

对大矩阵乘：

```text
C[i, j] = sum_k A[i, k] * B[k, j]
```

如果 `K` 很大，累积误差很重要。

所以精度配置要写清楚：

- input dtype。
- output dtype。
- accumulation dtype。
- reduction dtype。
- storage dtype。

只写“FP8 GEMM”不够。

## Master Weight

训练中常见 master weight：

```text
forward/backward 用低精度 weight
optimizer 更新高精度 master weight
再 cast 回低精度 weight
```

这样做的原因：

- 参数更新可能很小。
- 低精度权重可能无法表达微小更新。
- 高精度 master weight 保存长期累积信息。

FP16 训练中，FP32 master weight 很常见。

BF16 训练中，是否保留 master weight 取决于框架和策略。

FP8/低比特训练中，通常更需要明确 master weight 或高精度权重副本的设计。

## Outlier 问题

低精度量化最怕 outlier。

假设一个 tensor 大部分值在：

```text
-1 到 1
```

但有少数值是：

```text
100
```

如果 per-tensor scale 被 100 决定，大部分小值会分到很少的有效刻度，误差变大。

解决方向：

- per-channel scale。
- per-block scale。
- outlier-aware quantization。
- 保留 outlier channel 高精度。
- activation clipping。
- smooth quant。
- 改模型结构降低 outlier。

FP8、INT8、INT4 都会遇到这个问题，只是严重程度不同。

## Scale Metadata 也是系统成本

低比特格式本身很小，但 scale metadata 不是免费的。

例如 4 bit 权重量化时，如果每 64 个元素共享一个 scale，那么除了 packed weight 之外，还要保存每组 scale。若还需要 zero point、group index、outlier map 或 double quantization metadata，实际显存节省会小于理论值。

Scale metadata 会影响：

- HBM 占用。
- HBM 读取带宽。
- L2/cache 命中。
- kernel register 和 shared memory 使用。
- checkpoint 大小。
- 分布式通信 payload。
- resume 后数值一致性。

因此评估低比特方案时，显存表要把这些都列出来：

```text
quantized weights
scales
zero points
amax history
outlier buffers
temporary dequant buffers
kernel workspace
checkpoint metadata
```

如果只统计 quantized weights，很容易高估收益。

## 低精度和硬件路径

低精度是否快，取决于是否真的走硬件高吞吐路径。

需要确认：

- 硬件支持该 dtype。
- library/kernel 支持该 dtype。
- shape 满足对齐要求。
- tensor layout 正确。
- 没有额外 cast/transpose。
- dequant/quant 开销可控。
- accumulator 路径高效。
- fallback op 不成为瓶颈。

例如 INT8 权重很小，但如果每次 GEMM 前都要昂贵 dequant，或者 shape 不支持高效 INT8 Tensor Core，端到端未必快。

## 精度格式和硬件选型

看一块 AI 加速器是否适合低精度 workload，不能只看产品页上的峰值算力。需要逐项确认：

| 维度 | 要问的问题 |
| --- | --- |
| dtype 覆盖 | 是否支持 FP16、BF16、FP8、INT8、INT4、FP4，分别支持训练还是推理 |
| accumulator | 低精度 GEMM 用什么精度累积，是否可配置 |
| shape 约束 | M/N/K、batch、head_dim 是否要满足 tile 对齐 |
| layout 约束 | 是否要求特定 memory layout、packing 或 transpose |
| op 覆盖 | matmul 之外的 attention、normalization、softmax、MoE、embedding 是否高效 |
| scale 支持 | scale 应用是在矩阵单元内完成，还是需要额外 kernel |
| compiler/library | cuBLASLt、CUTLASS、Triton、TensorRT-LLM、vLLM、框架后端是否支持 |
| profiler 可见性 | 能否看到实际 kernel dtype、Tensor Core 利用率、fallback 和 cast |

对硬件研发或选型来说，真正重要的是目标模型的有效吞吐：

```text
effective throughput = 真实端到端 tokens/s 或 step/s
```

而不是某个 dtype 的孤立峰值。

## 低精度和并行策略

低精度会影响并行系统的边界。

在 Tensor Parallel 中：

- Column/Row Parallel 的通信 tensor 是否低精度。
- AllReduce / ReduceScatter 前后是否要 cast。
- 不同 TP rank 是否共享相同 scale 规则。

在 FSDP / ZeRO 中：

- parameter shard 存储 dtype 和 all-gather dtype 是否一致。
- optimizer state 是否保持 FP32。
- mixed precision policy 是否和 checkpoint/resume 对齐。

在 MoE / Expert Parallel 中：

- token dispatch 的 activation 是否量化。
- expert grouped GEMM 是否支持目标 dtype。
- router logits 和 load balancing loss 是否保持高精度。
- AllToAll 中 scale 是否随 token payload 一起传递。

这些问题决定低精度优化能否和并行策略组合，而不是只在单卡 microbenchmark 中成立。

## 训练中的低精度策略

训练比推理难，因为 backward 和 optimizer 对数值更敏感。

训练状态包括：

- weights。
- activations。
- gradients。
- optimizer states。
- master weights。
- loss scaler。
- amax/scale history。

常见路线：

### BF16 Baseline

大模型训练常用：

```text
BF16 forward/backward
FP32 optimizer state
selected FP32 reductions
```

优点是稳定、简单、硬件支持成熟。

### FP16 AMP

常用：

```text
FP16 autocast
GradScaler
FP32 master weights
FP32 optimizer state
```

适合硬件或框架 FP16 路径成熟，但要处理 loss scaling。

### FP8 Training

常见：

```text
部分 GEMM FP8
scale/amax 管理
敏感 op BF16/FP32
higher precision accumulation
framework recipe
```

关键是稳定性和覆盖率。

如果只有少数 GEMM FP8，收益有限。

如果太多 op FP8，可能不稳定。

## 推理中的低精度策略

推理更常用量化，因为没有 backward 和 optimizer。

常见：

- FP16/BF16 weights。
- FP8 weights/activations。
- INT8 weight + activation。
- INT4 weight-only。
- KV Cache quantization。
- logits / sampling 保持较高精度。

推理需要同时看：

- TTFT。
- TPOT。
- throughput。
- p99 latency。
- quality。
- memory footprint。
- dequant overhead。
- batching。

Weight-only INT4 可以显著降低模型权重占用，但 Decode 阶段如果瓶颈在 KV Cache 读取，单纯压权重不一定显著改善 TPOT。

## KV Cache 量化

KV Cache 可能是长上下文推理的主要 HBM 消耗。

量化 KV Cache 可以：

- 增加并发。
- 支持更长上下文。
- 降低 HBM bandwidth。
- 降低 P/D 分离的 KV 传输成本。

风险：

- attention quality 下降。
- 长上下文误差积累。
- scale metadata 增加。
- decode kernel 更复杂。
- quant/dequant overhead。

KV Cache 量化要按端到端推理指标评估，而不是只看显存下降。

## 通信中的低精度

分布式训练/推理中，通信也可以低精度：

- gradient compression。
- activation communication。
- expert dispatch。
- KV transfer。
- parameter all-gather。

低精度通信收益：

- 减少网络带宽。
- 降低通信时间。
- 更容易 overlap。

风险：

- 训练收敛变化。
- all-reduce 数值误差。
- scale 同步成本。
- 不同 rank scale 不一致。
- debug 难度提高。

通信低精度不是简单 cast，需要协议和数值策略配合。

## Checkpoint 与 Artifact Metadata

低精度方案必须能被恢复和复现。Checkpoint 不能只保存权重文件。

训练场景通常要保存：

- model weight dtype。
- master weight dtype 和内容。
- optimizer state dtype。
- GradScaler 或 loss scale 状态。
- FP8 scale、amax history、recipe 配置。
- quantization group size、block size、zero point。
- mixed precision policy。
- kernel/library 版本。
- 分布式 parallel group 和 shard metadata。

推理场景通常要保存：

- 权重量化格式。
- 每层、每 group 或每 block 的 scale。
- calibration 数据版本。
- outlier 处理策略。
- KV Cache dtype 和 scale 策略。
- 推理引擎版本。
- tokenizer、模型 config、rope scaling 等部署配置。

如果 artifact metadata 不完整，模型可能“能加载”，但数值行为已经不是原来的模型。这类问题很难从报错中看出来，只会表现为质量下降、长上下文退化或少数请求异常。

## 数值监控指标

低精度上线后，需要持续看数值健康度。

训练中建议观察：

- loss 是否出现 spike。
- NaN / Inf 的数量和首次出现 step。
- gradient norm、parameter norm、update norm。
- FP16 loss scale 的增长、下降和 skipped step。
- FP8 amax、scale history、saturation 比例。
- activation 的 max、min、mean、zero ratio。
- optimizer state 是否异常膨胀或变成 NaN。
- resume 后前几个 step 是否和预期一致。

推理中建议观察：

- logits 是否出现 NaN / Inf。
- sampling 前的 logits 范围。
- 长上下文回答质量。
- 量化前后 perplexity 或任务指标差异。
- 不同 batch size、context length 下的误差变化。
- p99 latency 是否因为 dequant/cast 抖动。
- 低精度 kernel 覆盖率和 fallback 比例。

低精度问题通常不是一开始就崩溃，而是在特定长度、特定 batch、特定 rank、特定 outlier 输入上暴露。监控要覆盖这些边界条件。

## Benchmark 应该怎么做

低精度 benchmark 不能只看 kernel TFLOPS。

至少看四类。

### 性能

- GEMM throughput。
- end-to-end step time。
- tokens/s。
- TTFT / TPOT。
- p99 latency。
- memory bandwidth。
- communication time。
- low-precision kernel coverage。
- cast / dequant / quant time。
- fallback op time。

### 显存

- model weights。
- activations。
- gradients。
- optimizer states。
- KV Cache。
- scale metadata。
- temporary buffers。
- packed weight layout。
- calibration / quantization artifacts。

### 质量

- training loss。
- validation loss。
- downstream eval。
- perplexity。
- task accuracy。
- long context quality。
- generation stability。
- quantized vs baseline 差异。
- 不同 context length、batch size、prompt 类型下的误差。

### 稳定性

- NaN/Inf。
- overflow。
- loss spike。
- gradient norm。
- amax/scale history。
- saturation ratio。
- zero ratio。
- outlier statistics。
- resume 后是否稳定。

低精度优化最终要看：

```text
达到同等质量所需真实时间和成本是否下降
```

而不只是单步更快。

## 常见误区

### 误区一：位数减半，速度就翻倍

不一定。可能瓶颈在通信、KV Cache、kernel launch、dequant 或非量化 op。

### 误区二：FP8 比 BF16 一定更好

不一定。FP8 需要 scale 管理和硬件支持，敏感 op 回退后收益可能有限。

### 误区三：INT4 权重小，所以推理一定快

不一定。如果 decode 瓶颈是 KV Cache bandwidth 或 dequant overhead，weight-only INT4 的收益会受限。

### 误区四：低精度只影响模型质量

不只。它还影响 kernel 选择、layout、metadata、通信、checkpoint 和恢复。

### 误区五：硬件支持某 dtype，所有 op 都能高效用

不一定。很多 op 仍可能回退到高精度或通用 kernel。

## 设计检查清单

设计低精度方案时，逐项确认：

- 目标是训练还是推理？
- 低精度用于 weights、activations、gradients、KV Cache 还是 optimizer state？
- input dtype 是什么？
- accumulator dtype 是什么？
- output dtype 是什么？
- scale 粒度是什么？
- amax/scale 如何更新？
- scale metadata 占多少显存和带宽？
- outlier 如何处理？
- 哪些 op 必须保持 BF16/FP32？
- 硬件是否支持该 dtype 高吞吐路径？
- kernel/library 是否支持真实 shape？
- 是否有额外 quant/dequant/cast/transpose？
- 显存节省是否包含 scale metadata？
- profiler 能否证明低精度 kernel 覆盖率？
- 分布式并行中 scale 是否跨 rank 一致？
- benchmark 是否看端到端时间？
- 质量是否达到目标？
- checkpoint 是否保存必要 scale 和 metadata？
- 多卡通信是否需要同步 scale？
- resume、回滚和模型导出是否能复现同一套量化配置？

## 小结

精度格式是硬件性能、显存容量、带宽、通信和数值稳定性的交汇点。

关键结论：

- FP32 稳定但成本高，常作为累积、optimizer 或 baseline。
- FP16 快但动态范围小，训练常需要 loss scaling。
- BF16 动态范围大，是大模型训练的常见默认低精度。
- FP8 可以进一步提高吞吐、降低带宽，但需要 scale/amax/recipe。
- INT8/INT4 常用于推理量化，但收益取决于 kernel、dequant、质量和真实瓶颈。
- Accumulator、master weight、sensitive op、scale metadata、checkpoint metadata 必须单独设计。
- 低精度和 TP/FSDP/ZeRO/MoE 等并行策略会相互影响，尤其是通信 dtype 和 scale 一致性。
- 低精度收益必须用端到端性能和质量共同验证。

真正成熟的低精度方案，不是把所有 tensor 强行变小，而是知道哪些值可以粗略表示，哪些计算必须保留精度，以及硬件是否真的因此减少了关键路径成本。

## 参考资料

- [PyTorch: Automatic Mixed Precision package](https://docs.pytorch.org/docs/2.12/amp.html)
- [NVIDIA Transformer Engine Documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html)
- [NVIDIA: Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [PyTorch: Quantization](https://docs.pytorch.org/docs/2.12/quantization.html)
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)
- [FP8-LM: Training FP8 Large Language Models](https://arxiv.org/abs/2310.18313)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
