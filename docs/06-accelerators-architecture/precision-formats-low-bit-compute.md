---
title: 精度格式：FP16、BF16、FP8 与量化计算
domain: accelerators-architecture
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-06-11
---

# 精度格式：FP16、BF16、FP8 与量化计算

AI 加速器性能和精度格式强相关。很多芯片的峰值算力会按 FP32、TF32、FP16、BF16、FP8、INT8、INT4 分开标注。低精度通常更快、更省显存、更省带宽，但也更容易带来数值误差、溢出、下溢和模型质量下降。

核心问题不是“精度越低越好”，而是：

> 哪些 tensor 可以低精度存储？哪些计算可以低精度执行？哪些累积必须高精度？scale 如何管理？硬件是否真的走到了对应低精度高吞吐路径？

这篇从硬件和系统角度解释常见精度格式。

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

### 显存

- model weights。
- activations。
- gradients。
- optimizer states。
- KV Cache。
- scale metadata。
- temporary buffers。

### 质量

- training loss。
- validation loss。
- downstream eval。
- perplexity。
- task accuracy。
- long context quality。
- generation stability。

### 稳定性

- NaN/Inf。
- overflow。
- loss spike。
- gradient norm。
- amax/scale history。
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
- outlier 如何处理？
- 哪些 op 必须保持 BF16/FP32？
- 硬件是否支持该 dtype 高吞吐路径？
- kernel/library 是否支持真实 shape？
- 是否有额外 quant/dequant/cast/transpose？
- 显存节省是否包含 scale metadata？
- benchmark 是否看端到端时间？
- 质量是否达到目标？
- checkpoint 是否保存必要 scale 和 metadata？
- 多卡通信是否需要同步 scale？

## 小结

精度格式是硬件性能、显存容量、带宽、通信和数值稳定性的交汇点。

关键结论：

- FP32 稳定但成本高，常作为累积、optimizer 或 baseline。
- FP16 快但动态范围小，训练常需要 loss scaling。
- BF16 动态范围大，是大模型训练的常见默认低精度。
- FP8 可以进一步提高吞吐、降低带宽，但需要 scale/amax/recipe。
- INT8/INT4 常用于推理量化，但收益取决于 kernel、dequant、质量和真实瓶颈。
- Accumulator、master weight、sensitive op、scale metadata 必须单独设计。
- 低精度收益必须用端到端性能和质量共同验证。

真正成熟的低精度方案，不是把所有 tensor 强行变小，而是知道哪些值可以粗略表示，哪些计算必须保留精度，以及硬件是否真的因此减少了关键路径成本。

## 参考资料

- [PyTorch: Automatic Mixed Precision package](https://docs.pytorch.org/docs/2.12/amp.html)
- [NVIDIA Transformer Engine Documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html)
- [NVIDIA: Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)
- [FP8-LM: Training FP8 Large Language Models](https://arxiv.org/abs/2310.18313)
