---
title: Optimizer 与 Scheduler 系统成本
domain: training-systems
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-06-12
---

# Optimizer 与 Scheduler 系统成本

训练系统里，forward/backward 通常最显眼。

但在大模型训练中，optimizer step 也可能成为显存、带宽、step time、checkpoint 和恢复成本的大头。

尤其是 AdamW 这类自适应优化器，它不仅更新参数，还要维护长期 optimizer state。

一句话理解：

> Optimizer 与 Scheduler 的系统成本，来自参数更新本身、optimizer state 读写、混合精度 master weight、梯度裁剪、分布式 sharding/offload、checkpoint 保存和 step 计数语义。

本篇不重点推导优化算法，而从系统角度回答：

- optimizer 为什么占这么多显存？
- optimizer step 为什么可能慢？
- fused / foreach optimizer 解决什么？
- gradient clipping 为什么是同步点？
- scheduler 在 gradient accumulation 下为什么容易错？
- ZeRO/FSDP 如何切 optimizer state？
- checkpoint/resume 时 optimizer 和 scheduler 要保存什么？
- benchmark 时如何判断 optimizer 是否是瓶颈？

## 先给结论

训练系统里，optimizer/scheduler 要当成一等公民，而不是 backward 后面的“小步骤”。

原因：

| 问题 | 影响 |
| --- | --- |
| AdamW optimizer state 很大 | 每个参数可能额外维护 FP32 master、m、v。 |
| optimizer step 多为 elementwise | FLOPs 不高，但 HBM 读写重，容易 memory-bound。 |
| 参数 tensor 太碎 | Python loop 和 kernel launch 开销明显。 |
| gradient clipping 需要全局规约 | 引入额外读写和同步点。 |
| scheduler step 语义容易错 | gradient accumulation、resume、token-based schedule 会被影响。 |
| ZeRO/FSDP sharding 改变状态布局 | 省显存，但增加通信、resharding 和 checkpoint 复杂度。 |
| checkpoint 保存 optimizer state 很重 | AdamW state 常比模型权重大很多。 |

实用原则：

```text
先把 optimizer state 显存算清楚
再看 optimizer step 是否在 timeline 上暴露
然后决定 fused/foreach、sharding、offload、低精度 optimizer 或 scheduler 语义调整
```

不要只优化 forward/backward。

如果 optimizer step、gradient clipping、checkpoint optimizer state 或 scheduler 语义不对，长期训练仍然会慢、不稳或不可复现。

## 一个 Optimizer Step 做了什么

一次 optimizer step 通常发生在 backward 和梯度同步之后：

```text
forward
loss
backward
gradient sync
unscale gradients, if AMP
gradient clipping
optimizer step
scheduler step
zero grad
```

以 AdamW 为例，每个参数大致需要：

1. 读取参数 `p`。
2. 读取梯度 `g`。
3. 读取一阶矩 `m`。
4. 读取二阶矩 `v`。
5. 更新 `m`。
6. 更新 `v`。
7. 做 bias correction。
8. 应用 Adam update。
9. 应用 decoupled weight decay。
10. 写回参数和 optimizer states。

这是一组大量逐元素读写。

它通常不是 compute-bound，而是 memory-bound。

## AdamW 的显存成本

Adam 系列优化器维护两个状态：

```text
m: first moment
v: second moment
```

混合精度训练还可能有 FP32 master weight。

假设模型有 `N` 个参数，训练权重和梯度用 BF16/FP16，AdamW 状态用 FP32：

| 项 | 每参数字节 |
| --- | --- |
| 参数 | 2 |
| 梯度 | 2 |
| FP32 master weight | 4 |
| Adam m | 4 |
| Adam v | 4 |

合计：

```text
2 + 2 + 4 + 4 + 4 = 16 bytes / parameter
```

7B 参数就是：

```text
7B * 16 bytes = 112GB
```

这还没算：

- activations。
- temporary buffers。
- communication buffers。
- CUDA allocator reserved memory。
- fragmentation。
- checkpoint staging。

所以 AdamW 的系统成本不是“多做一点数学”，而是每个参数维护多份长期状态。

## Adam 与 AdamW 的系统差异

Adam 更新一阶矩和二阶矩：

```text
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad^2
param = param - lr * m_hat / (sqrt(v_hat) + eps)
```

AdamW 的关键是 decoupled weight decay。

它把 weight decay 从梯度自适应缩放中拆出来：

```text
param = param - lr * weight_decay * param
param = param - lr * adam_update
```

系统上要关注：

- decay 参数组和 no-decay 参数组。
- weight decay 是否作用在 embedding、bias、norm。
- fused optimizer 是否支持这些参数组。
- checkpoint 中参数组顺序是否稳定。
- resume 后 parameter group 的 learning rate / weight decay 是否一致。

常见参数分组：

| 参数 | 常见策略 |
| --- | --- |
| Linear / attention / MLP weight | weight decay。 |
| bias | no decay。 |
| LayerNorm / RMSNorm weight | no decay。 |
| embedding | 视 recipe 而定。 |
| MoE router | 视稳定性而定。 |
| LoRA adapter | 通常独立参数组。 |

不要只记录：

```text
optimizer = AdamW
```

还要记录参数组策略。

## Optimizer Step 为什么可能慢

optimizer step 慢的常见原因：

- 参数数量巨大。
- AdamW 读写 `p/g/m/v/master`。
- 参数 tensor 很碎，kernel launch 多。
- Python for-loop optimizer 调度开销高。
- gradient clipping 额外扫描所有梯度。
- AMP unscale 和 overflow check 增加扫描。
- ZeRO/FSDP 需要处理 shard、reduce-scatter、all-gather。
- CPU/NVMe offload 带来 PCIe/NVMe 读写。
- checkpoint 同步保存 optimizer state。

和 GEMM 不同，optimizer 通常不是大型 Tensor Core 矩阵乘。

它更像：

```text
读很多内存
做少量 elementwise math
写很多内存
```

所以它容易被 HBM 带宽、kernel launch 和状态布局限制。

## Fused Optimizer

普通 optimizer 可能按参数 tensor 循环：

```text
for param in params:
    update m
    update v
    compute denom
    apply weight decay
    update param
```

Fused optimizer 把多个逐元素操作融合到更少 kernel 中。

收益：

- 减少 kernel launch。
- 减少中间结果读写。
- 提高内存访问效率。
- 对大量小 tensor 更友好。
- 可能更适合 CUDA graph / compile 路径。

代价：

- dtype 支持有限。
- device 支持有限。
- 参数布局要求更多。
- 调试更难。
- 可能不支持所有 optimizer 选项。
- 版本差异明显。

PyTorch `AdamW` 提供 `foreach` 和 `fused` 相关选项。官方文档也说明，如果用户没有显式指定，CUDA 上通常会优先尝试 foreach，因为它一般比 for-loop 快，但会使用额外 tensorlist 中间内存；fused 可能更快，但成熟度和支持范围需要看版本和 dtype。

## Foreach Optimizer

foreach 实现把多个 tensor 的同类操作批量提交。

直觉：

```text
many small tensor ops
-> batched tensorlist ops
```

优点：

- 比 Python for-loop 更少调度开销。
- 对很多参数 tensor 更快。
- PyTorch 原生支持较好。

代价：

- 会使用额外 tensorlist 中间内存。
- peak memory 可能上升。
- 不一定适合显存紧张场景。

所以选择时要 benchmark：

| 实现 | 可能收益 | 风险 |
| --- | --- | --- |
| for-loop | 最通用 | 慢，launch 多。 |
| foreach | 通常更快 | peak memory 可能更高。 |
| fused | 可能最快 | 支持范围和版本依赖更强。 |

## Capturable 与 Differentiable

PyTorch AdamW 还有 `capturable`、`differentiable` 等选项。

系统含义：

- `capturable=True`：让 optimizer step 更适合 CUDA graph capture 等场景。
- `differentiable=True`：允许 optimizer step 参与 autograd，用于高阶优化等特殊场景。

普通大模型训练通常不需要 `differentiable=True`。

`capturable=True` 是否需要，取决于：

- 是否使用 CUDA graphs。
- optimizer state 是否在 capture 兼容位置。
- 学习率等标量是否 tensor 化。
- fused optimizer 是否支持。

这些选项会影响性能和兼容性，应该写入 benchmark manifest。

## Gradient Clipping 的系统成本

大模型训练常用 global norm clipping。

流程：

1. 遍历所有 gradient。
2. 计算平方和。
3. 跨 DP/FSDP ranks 做规约。
4. 得到 global norm。
5. 如果超过阈值，缩放所有 gradient。

这会引入：

- 额外 gradient 读写。
- 额外 reduction。
- 同步点。
- AMP unscale 顺序要求。
- 分布式 group 语义。

混合精度下，通常要先 unscale gradients，再 clip。

错误顺序：

```text
clip scaled gradients
then unscale
```

会改变 clipping 语义。

正确直觉：

```text
unscale -> compute grad norm -> clip -> optimizer step
```

相关内容见：[混合精度训练](mixed-precision-training.md)

## Gradient Norm 是监控信号

虽然 gradient clipping 有成本，但 grad norm 是非常重要的健康信号。

它能帮助发现：

- loss spike。
- bad batch。
- NaN/Inf 前兆。
- learning rate 过高。
- FP16 overflow。
- FP8 scale 异常。
- MoE router 不稳。
- 梯度同步异常。

建议记录：

- global grad norm。
- clipped / unclipped norm。
- clipping ratio。
- per-layer grad norm。
- skipped step。
- AMP loss scale。
- FP8 amax / scale。

如果训练系统为了省一点时间完全不记录 grad norm，排查稳定性会困难很多。

## Scheduler 做了什么

Scheduler 控制 learning rate 随训练进度变化。

常见策略：

- warmup。
- cosine decay。
- linear decay。
- constant with warmup。
- polynomial decay。
- step decay。
- inverse square root。
- token-based schedule。

Scheduler 本身计算成本通常很小。

真正重要的是 step 语义。

训练中至少有这些计数：

| 计数 | 含义 |
| --- | --- |
| micro-step | 一个 micro-batch forward/backward。 |
| gradient accumulation step | 累积中的第几个 micro-step。 |
| optimizer step | 参数实际更新一次。 |
| global step | 日志/checkpoint/eval 使用的全局步，需明确定义。 |
| consumed samples | 已消费样本数。 |
| consumed tokens | 已消费 token 数。 |
| loss tokens | 真正参与 loss 的 token 数。 |

Scheduler 通常应该跟 optimizer step 或 consumed tokens 对齐，而不是跟 micro-step 混淆。

## Gradient Accumulation 下的 Scheduler

假设：

```text
gradient_accumulation_steps = 8
```

流程：

```text
micro-step 1: forward/backward
micro-step 2: forward/backward
...
micro-step 8: forward/backward
optimizer step
scheduler step
zero grad
```

如果 scheduler 每个 micro-step 都 step，一次有效 batch 内学习率会变化 8 次。

这通常不是预期。

尤其在比较不同 gradient accumulation 配置时：

```text
global batch 相同
gradient accumulation 不同
```

如果 scheduler 绑定 micro-step，学习率曲线会变，实验不可比。

## Token-based Scheduler

大模型训练常按 token budget 管理。

例如：

```text
warmup 10B tokens
train 1T tokens
cosine decay by consumed tokens
```

这比按 step 更稳，因为 step 可能随下面变化：

- sequence length。
- packing ratio。
- micro-batch。
- gradient accumulation。
- world size。
- data parallel size。
- loss mask。

但 token-based scheduler 也要求系统准确记录：

- consumed input tokens。
- consumed loss tokens。
- skipped batches。
- dropped samples。
- resume 后 data cursor。

如果用 padded tokens 代替 loss tokens，scheduler 语义可能偏离真实训练量。

相关内容见：[训练数据混合、采样与有效 Token](training-data-mixing-sampling-effective-tokens.md)

## Scheduler 与 Skipped Step

AMP overflow 或数值 guardrail 可能导致 optimizer step 被跳过。

问题是：

```text
optimizer step skipped 时，scheduler 是否 step？
```

通常如果参数没有更新，scheduler 也不应该像正常 optimizer step 一样前进。

否则学习率曲线会消耗掉一个 step，但模型没有对应更新。

同样，异常 batch 被丢弃、OOM 重试、rollback，也会影响 scheduler 语义。

建议在日志中记录：

- attempted optimizer steps。
- completed optimizer steps。
- skipped optimizer steps。
- scheduler steps。
- consumed tokens。
- current lr。

## 参数分组与 Scheduler

Optimizer 里常有多个 parameter groups。

不同 group 可能有：

- 不同 learning rate。
- 不同 weight decay。
- 不同 betas。
- 不同 optimizer。
- 是否冻结。
- adapter / LoRA 单独参数组。
- router / expert 单独参数组。

Scheduler 通常要更新每个 group 的 lr。

这带来两个问题：

1. checkpoint/resume 时 group 顺序必须稳定。
2. 新增或删除参数时，旧 checkpoint 的 optimizer state 如何映射？

如果 parameter group 不稳定，resume 可能出现：

- 某些参数拿到错误 lr。
- weight decay 策略错。
- optimizer state 对不上。
- adapter 参数没有正确恢复。

因此参数组构造必须可复现、可记录。

## 分布式 Optimizer

Data Parallel 默认每个 rank 保存完整 optimizer state。

AdamW 下这很贵。

ZeRO / FSDP 通过 sharding 降低重复：

```text
Rank 0: optimizer state shard 0
Rank 1: optimizer state shard 1
Rank 2: optimizer state shard 2
Rank 3: optimizer state shard 3
```

收益：

- 每卡 optimizer state 显存下降。
- 支持更大模型。
- optimizer state 和 gradient shard 结合更自然。

代价：

- optimizer step 依赖 shard layout。
- gradient reduce-scatter 语义更复杂。
- checkpoint 需要保存 sharded optimizer state。
- resume 需要恢复分片映射。
- world size 改变时可能要 reshard。

相关内容见：[ZeRO 与 FSDP](zero-fsdp.md)

## ZeRO Stage 与 Optimizer State

ZeRO 的直觉：

| Stage | 切什么 |
| --- | --- |
| ZeRO-1 | optimizer states。 |
| ZeRO-2 | optimizer states + gradients。 |
| ZeRO-3 | optimizer states + gradients + parameters。 |

Optimizer 成本和 ZeRO 强相关：

- ZeRO-1 主要解决 Adam state 显存。
- ZeRO-2 进一步降低 gradient 显存。
- ZeRO-3 还要处理参数 all-gather。

如果 optimizer state 是最大显存项，ZeRO-1 就可能带来明显收益。

如果参数本身也放不下，需要 ZeRO-3/FSDP。

但更高 stage 也会增加通信和调度复杂度。

## Offload Optimizer State

Optimizer state 可以 offload 到：

- CPU memory。
- NVMe。

收益：

- GPU 显存压力下降。
- 更大模型能跑起来。

代价：

- PCIe / NVLink-C2C / CPU memory bandwidth 成瓶颈。
- NVMe latency / bandwidth 成瓶颈。
- optimizer step 更慢。
- prefetch / overlap 更复杂。
- checkpoint 和 resume 更慢。

Offload 适合：

- 显存绝对不够。
- 训练吞吐不是第一优先。
- 有高带宽 host memory / NVMe。
- 可以接受更复杂的调度。

如果目标是最大 tokens/s，offload 往往不是首选。

## 低精度 Optimizer State

为了降低 AdamW state 成本，可以考虑：

- 8-bit optimizer state。
- BF16 optimizer state。
- factored optimizer，如 Adafactor。
- optimizer state quantization。
- Muon 等不同状态结构。

收益：

- optimizer state 显存下降。
- checkpoint 变小。
- optimizer memory bandwidth 下降。

风险：

- 数值稳定性变化。
- 收敛曲线变化。
- 对超参数更敏感。
- 分布式 sharding 和 checkpoint 更复杂。

低精度 optimizer 不是单纯系统优化，也会改变训练算法行为。

必须做质量和稳定性验证。

相关内容见：[Muon 优化器](muon-optimizer.md)

## Fused Optimizer 与 Compile / CUDA Graph

现代训练栈会尝试减少 Python overhead：

- fused optimizer。
- foreach optimizer。
- CUDA graphs。
- `torch.compile`。
- custom optimizer kernel。

这些技术可能互相影响：

- optimizer step 是否 capturable。
- learning rate 是否 tensor 化。
- parameter group 是否动态变化。
- gradient 是否 set_to_none。
- fused kernel 是否支持当前 dtype。
- FSDP/ZeRO 是否能和 graph capture 兼容。

如果 optimizer step 在 Python 侧开销很大，graph/capture 可能有帮助。

但分布式训练中，通信、动态 shape、异常处理、scheduler、checkpoint 等都会增加 capture 难度。

## zero_grad 的语义

`zero_grad` 看起来简单，但也有系统影响。

两种常见方式：

```python
optimizer.zero_grad()
optimizer.zero_grad(set_to_none=True)
```

`set_to_none=True` 会把 `.grad` 设为 `None`，而不是填 0。

可能收益：

- 避免清零写入。
- 降低 memory bandwidth。
- 让下一次 backward 直接分配/写入 gradient。

注意：

- 某些代码假设 `.grad` 一定是 tensor。
- optimizer 对 `grad is None` 和 `grad == 0` 的行为可能不同。
- 分布式 hook、manual grad 操作要兼容。

大模型训练中，`set_to_none=True` 很常见，但要明确代码是否支持。

## Checkpoint / Resume 要保存什么

Optimizer/scheduler 相关 checkpoint 至少包括：

- optimizer state。
- parameter groups。
- master weights。
- scheduler state。
- current lr。
- optimizer step。
- scheduler step。
- GradScaler state。
- gradient accumulation state。
- skipped step count。
- consumed tokens。
- sharding metadata。
- offload metadata。

如果 optimizer state 丢失，只加载 model weights，那不是 exact resume，而是 warm start。

Resume 后常见错误：

- lr 从头 warmup。
- Adam m/v 丢失。
- master weights 丢失。
- parameter group 顺序变化。
- GradScaler scale 重置。
- ZeRO/FSDP shard mapping 不一致。
- scheduler 多 step 或少 step。

相关内容见：[Checkpoint、Resume 与容错](checkpoint-resume-fault-tolerance.md)

## Optimizer 与 Checkpoint 体积

AdamW checkpoint 通常远大于模型权重。

例如 BF16 参数 + FP32 Adam state：

| 对象 | 大小直觉 |
| --- | --- |
| BF16 model weights | 2 bytes / param |
| FP32 master weights | 4 bytes / param |
| Adam m | 4 bytes / param |
| Adam v | 4 bytes / param |

只保存模型权重：

```text
2 bytes / param
```

保存完整训练状态：

```text
2 + 4 + 4 + 4 = 14 bytes / param
```

这还不包括：

- scheduler。
- RNG。
- dataloader state。
- FP8 scale。
- metadata。

因此 checkpoint 保存 optimizer state 是长期训练存储系统的关键压力。

## Benchmark 时看什么

optimizer/scheduler benchmark 不能只看端到端 step time。

要拆：

| 指标 | 说明 |
| --- | --- |
| optimizer step time | 参数更新耗时。 |
| gradient clipping time | norm 计算和缩放耗时。 |
| unscale / overflow check time | AMP 相关成本。 |
| scheduler step time | 通常小，但语义要验证。 |
| zero_grad time | 是否有大量清零写。 |
| optimizer state memory | m/v/master 占用。 |
| optimizer temporary memory | foreach/fused 中间内存。 |
| checkpoint optimizer write time | 保存 optimizer state 的 I/O。 |
| resume optimizer load time | 恢复训练耗时。 |
| skipped step count | 数值稳定性和有效吞吐。 |

建议做 A/B：

| 对比 | 目的 |
| --- | --- |
| for-loop AdamW vs foreach AdamW | 看调度和 tensorlist 收益。 |
| foreach vs fused AdamW | 看 fused 是否更快且稳定。 |
| full optimizer state vs ZeRO/FSDP sharded | 看显存/通信取舍。 |
| no clipping vs clipping | 看稳定收益和成本。 |
| step-based vs token-based scheduler | 看语义和实验可比性。 |
| GPU optimizer vs CPU/NVMe offload | 看显存和吞吐取舍。 |

## Profiler 里怎么看

Profiler 重点看：

- optimizer step 是否在 timeline 上明显暴露。
- AdamW kernel 是否碎片化。
- foreach/fused 是否减少 kernel 数。
- HBM bandwidth 是否接近瓶颈。
- gradient clipping 是否是同步点。
- FSDP/ZeRO optimizer step 是否有通信等待。
- offload 是否出现 H2D/D2H 或 NVMe 等待。
- zero_grad 是否有大块 memset。
- scheduler 是否在 skipped step 时错误推进。

常见现象：

| 现象 | 可能原因 |
| --- | --- |
| optimizer step 很长 | Adam state 读写重、未 fused、参数碎。 |
| foreach 更快但 OOM | foreach 中间 tensorlist 增加 peak memory。 |
| fused 不生效 | dtype/device/版本不支持或参数组不兼容。 |
| grad clipping 暴露 | global norm reduction 成同步点。 |
| resume 后 loss 跳 | optimizer/scheduler/master/scaler 状态没恢复。 |

## 常见优化方向

### 1. 先算 Optimizer State

在调 optimizer 之前，先算显存：

```text
params + grads + master weights + m + v
```

明确瓶颈是：

- optimizer state。
- activation。
- communication buffer。
- temporary buffer。
- logits/loss。

如果 optimizer state 是主因，再考虑 ZeRO/FSDP、低精度 state 或 offload。

### 2. 使用 Foreach / Fused Optimizer

在支持的 dtype/device 上 benchmark：

```text
for-loop AdamW
foreach AdamW
fused AdamW
```

同时记录：

- step time。
- peak memory。
- kernel 数量。
- correctness。
- checkpoint/resume。

不要只看单 step 速度，foreach 的额外内存可能在长上下文训练中导致 OOM。

### 3. 稳定 Parameter Groups

参数组应可复现。

建议记录：

- group name。
- parameter name pattern。
- lr。
- weight decay。
- betas。
- optimizer type。
- trainable / frozen。

不要只靠 Python dict 遍历顺序和隐式规则。

### 4. 明确 Scheduler 计数

Scheduler 需要明确绑定：

- optimizer step。
- consumed tokens。
- epoch。
- wall-clock。

推荐在 Run Manifest 中记录：

```yaml
scheduler:
  type: cosine_with_warmup
  step_unit: optimizer_step
  warmup_steps: 2000
  max_steps: 200000
```

或：

```yaml
scheduler:
  type: cosine_with_warmup
  step_unit: loss_tokens
  warmup_tokens: 10000000000
  max_tokens: 1000000000000
```

### 5. 保存完整 Resume State

如果目标是 exact resume，必须保存：

- optimizer state。
- scheduler state。
- GradScaler state。
- global step。
- optimizer step。
- consumed tokens。
- parameter group metadata。

只保存 model weights 不够。

### 6. 将 Optimizer 纳入 Benchmark

训练 benchmark 报告应拆出：

- forward。
- backward。
- gradient sync。
- clipping。
- optimizer step。
- scheduler。
- checkpoint。

如果 optimizer step 占比 10%-20% 以上，就值得单独优化。

## 常见误区

### 误区一：Optimizer 只是很小的后处理

不对。

AdamW state 可能是训练显存最大项之一，optimizer step 也可能明显占 step time。

### 误区二：Fused 一定更好

不一定。

fused optimizer 依赖 dtype、device、shape、参数组、框架版本。

必须 benchmark。

### 误区三：Scheduler 每个 batch step 一下就行

不一定。

gradient accumulation 下，batch/micro-step 和 optimizer step 不是一回事。

### 误区四：Resume 只需要模型权重

不对。

只加载模型权重是 warm start，不是 exact resume。

长期训练恢复必须包含 optimizer 和 scheduler state。

### 误区五：Gradient clipping 免费

不免费。

global norm clipping 要扫描梯度并做分布式规约。它是稳定性工具，也是一项系统成本。

### 误区六：低精度 optimizer state 只是系统优化

不完全是。

低精度 optimizer state 可能改变训练算法行为和收敛，需要质量验证。

## 落地检查表

配置前：

- [ ] 模型参数量是多少？
- [ ] AdamW 每参数状态多少字节？
- [ ] 是否需要 FP32 master weights？
- [ ] optimizer state 是否是显存主因？
- [ ] scheduler 按 optimizer step 还是 token 计数？
- [ ] gradient accumulation 是否改变 scheduler 语义？

配置时：

- [ ] 参数组是否稳定、可记录？
- [ ] weight decay/no-decay 规则是否明确？
- [ ] optimizer 实现是 for-loop、foreach 还是 fused？
- [ ] foreach/fused 是否支持当前 dtype/device？
- [ ] gradient clipping 是否在 AMP unscale 后？
- [ ] ZeRO/FSDP sharding 是否覆盖 optimizer state？
- [ ] 是否使用 CPU/NVMe offload？
- [ ] scheduler state 是否保存？

验证时：

- [ ] optimizer step time 是否下降？
- [ ] peak memory 是否变化？
- [ ] foreach/fused 是否引入额外显存？
- [ ] loss 曲线是否和 baseline 对齐？
- [ ] grad norm / clipping ratio 是否正常？
- [ ] scheduler lr 曲线是否正确？
- [ ] skipped step 时 scheduler 是否正确处理？
- [ ] checkpoint/resume 后 lr、m/v、master weights 是否连续？

Benchmark 报告：

- [ ] optimizer type 和实现。
- [ ] parameter group rules。
- [ ] optimizer state dtype。
- [ ] master weight dtype。
- [ ] gradient clipping 配置。
- [ ] scheduler 类型和 step unit。
- [ ] optimizer step time。
- [ ] optimizer memory。
- [ ] checkpoint optimizer state 时间。
- [ ] run manifest。

## 小结

Optimizer 与 Scheduler 是训练系统里的关键成本中心。

要抓住几件事：

- AdamW 显存贵，因为每个参数维护 m/v/master 等状态。
- optimizer step 多为 memory-bound elementwise 读写。
- foreach/fused optimizer 可以减少调度和 kernel 开销，但要看显存和支持范围。
- gradient clipping 是稳定性工具，也是额外同步和读写成本。
- scheduler 的 step 语义必须和 optimizer step、gradient accumulation、tokens 对齐。
- ZeRO/FSDP 通过 sharding optimizer state 降低显存，但增加通信和 checkpoint 复杂度。
- checkpoint/resume 必须保存 optimizer、scheduler、GradScaler、parameter group 和 step 计数状态。

训练系统优化不能只看 forward/backward。

一个完整的 step breakdown 应该把 optimizer、scheduler、clipping 和 checkpoint 都拆出来，否则很容易低估长期训练成本。

## 参考资料

- [PyTorch: AdamW](https://docs.pytorch.org/docs/2.12/generated/torch.optim.AdamW.html)
- [PyTorch: clip_grad_norm_](https://docs.pytorch.org/docs/2.12/generated/torch.nn.utils.clip_grad_norm_.html)
- [PyTorch: Learning Rate Scheduler](https://docs.pytorch.org/docs/2.12/optim.html#how-to-adjust-learning-rate)
- [ZeRO 与 FSDP](zero-fsdp.md)
- [混合精度训练](mixed-precision-training.md)
