---
title: Muon 优化器
domain: training-systems
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-06-11
---

# Muon 优化器

Muon 是一种面向矩阵参数的优化器。它的核心不是给所有参数都换一个 optimizer，而是对神经网络 hidden layers 里的二维权重矩阵，用“动量 + 矩阵正交化”的方式产生更新方向。

如果只从训练系统角度记一句话：

> Muon 把 optimizer step 从逐元素更新，变成了矩阵级更新；它可能减少 optimizer state 并改善训练效率，但会引入 Newton-Schulz 矩阵乘、参数分组、分布式 gather、更新尺度和 checkpoint 语义问题。

这篇不深入证明 Muon 的优化理论。重点回答：

- Muon 和 SGD Momentum、AdamW 的差别是什么。
- 为什么它只适合部分矩阵参数。
- Newton-Schulz 正交化在系统里带来什么成本。
- Muon 和 ZeRO、FSDP、TP、MoE 训练如何互相影响。
- 评估 Muon 时不能只看 loss，还要看 wall-clock time、step time 和稳定性。

## 先从普通 optimizer step 说起

训练时，backward 得到每个参数的 gradient。Optimizer 用这些 gradient 更新权重。

最简单的 SGD 可以理解成：

```text
new_weight = old_weight - lr * gradient
```

SGD Momentum 会额外维护一个动量：

```text
momentum = beta * old_momentum + gradient
new_weight = old_weight - lr * momentum
```

AdamW 会维护一阶矩和二阶矩：

```text
m = beta1 * old_m + (1 - beta1) * gradient
v = beta2 * old_v + (1 - beta2) * gradient^2
update = m / sqrt(v + eps)
new_weight = weight_decay_then(old_weight) - lr * update
```

从系统角度看，AdamW 的特点是：

- 对每个参数元素维护 `m` 和 `v`。
- 混合精度训练里还可能维护 FP32 master weight。
- 更新主要是逐元素操作，适合 foreach/fused optimizer。
- optimizer state 显存很高，但更新逻辑对参数形状不敏感。

Muon 的不同点在于：它不只是逐元素缩放 gradient，而是把二维权重矩阵的动量看成一个矩阵，再对这个矩阵做近似正交化。

## Muon 的核心流程

一个简化的 Muon step 可以理解成：

```text
gradient matrix G
-> momentum matrix B
-> Newton-Schulz approximate orthogonalization
-> orthogonalized update O
-> update weight matrix W
```

用图表示：

```mermaid
flowchart TD
    A["2D weight matrix W"] --> B["backward 得到 gradient G"]
    B --> C["更新 momentum matrix B"]
    C --> D["Newton-Schulz 迭代<br/>近似正交化 B"]
    D --> E["得到 orthogonalized update O"]
    E --> F["按 learning rate 和 weight decay 更新 W"]
```

这里的“正交化”可以先用直觉理解：

- 普通动量矩阵可能被少数方向主导，更新看起来像“主要沿几个方向动”。
- 正交化会把矩阵更新方向变得更均衡，避免所有神经元或通道过度挤在少数方向上更新。
- Muon 实际使用 Newton-Schulz 迭代近似这个过程，而不是每一步都做昂贵的精确 SVD。

Keller Jordan 的原始介绍把 Muon 定义为 MomentUm Orthogonalized by Newton-Schulz。PyTorch 2.12 的 `torch.optim.Muon` 也把 Newton-Schulz steps、momentum、Nesterov、weight decay、update scale adjustment 暴露为 optimizer 参数。

## 为什么是矩阵参数

Muon 关心的是二维权重矩阵，例如 Transformer 里的很多线性层：

- attention 的 `q_proj`、`k_proj`、`v_proj`、`o_proj`。
- MLP 的 up projection、gate projection、down projection。
- MoE expert 里的 FFN 矩阵。

这些参数天然是矩阵。矩阵有行、列和方向结构，正交化才有明确含义。

不适合直接交给 Muon 的参数通常包括：

- bias。
- LayerNorm / RMSNorm 权重。
- embedding。
- LM head / output projection，具体是否纳入要看实现和实验设计。
- 标量、向量或非典型 shape 参数。

这也是为什么实践中常见做法不是“全模型 Muon”，而是：

```text
2D hidden layer weights -> Muon
other params             -> AdamW
```

PyTorch 文档示例也是按 `p.ndim == 2` 和 `p.ndim != 2` 把参数分到 Muon 与 AdamW 两组。

## 和 AdamW 的系统差异

AdamW 对每个元素保存两个状态：

```text
parameter p
gradient g
first moment m
second moment v
optional master weight
```

Muon 更像：

```text
matrix parameter W
gradient matrix G
momentum matrix B
optional master weight
temporary matrices for Newton-Schulz
```

重要差异有三类。

### 状态数量不同

AdamW 通常有 `m` 和 `v` 两个长期状态。Muon 至少需要 momentum buffer，但不一定需要 AdamW 那样的二阶矩状态。

所以在某些实现里，Muon 的长期 optimizer state 可能比 AdamW 少。Moonshot 的 Muon scaling 报告也强调，Distributed Muon 使用一个 momentum buffer，而 AdamW 有两个 momentum buffers。

但这不等于 Muon 一定更省显存。原因是：

- 可能仍有 FP32 master weight。
- Newton-Schulz 会产生临时矩阵。
- foreach/fused 实现可能有临时 buffer。
- 分布式 gather 可能在 optimizer step 中短暂形成 full matrix。

评估显存时要看 peak memory，而不是只看长期 optimizer state。

### 更新粒度不同

AdamW 是逐元素更新：

```text
update[i, j] depends mostly on g[i, j], m[i, j], v[i, j]
```

Muon 是矩阵级更新：

```text
update matrix O depends on the whole momentum matrix B
```

这意味着 Muon 对参数矩阵完整形状更敏感。如果一个矩阵被分布式切碎，只在本地 shard 上做 Muon，结果未必等价于在完整矩阵上做 Muon。

### 更新尺度不同

正交化会改变更新矩阵的谱结构，也会让不同 shape 的矩阵更新 RMS 不一致。Muon scaling 工作提出要用 weight decay 和 per-parameter update scale adjustment 来让 Muon 更稳定地扩展到大模型训练。

工程含义是：

- Muon 的 `lr` 不能简单照搬所有 AdamW 配置。
- 如果实现提供 `adjust_lr_fn`，要明确选择哪种尺度策略。
- 不同矩阵 shape、GQA/MLA 切法、MoE expert shape 都可能影响更新尺度。

## Newton-Schulz 迭代是什么

严格说，正交化可以通过 SVD 得到。如果动量矩阵 `B` 的 SVD 是：

```text
B = U S V^T
```

正交化更新方向可以近似看成：

```text
O = U V^T
```

但每个 optimizer step 对大量权重矩阵做 SVD 太贵。Muon 使用 Newton-Schulz 迭代，用少量矩阵乘近似这个正交化方向。

这就是系统上最重要的点：

> Muon 把一部分 optimizer step 变成了小到中等规模的矩阵乘。

这和 AdamW 的逐元素更新完全不同。

### Newton-Schulz 的成本

一次 Newton-Schulz 迭代大致包含：

- 矩阵乘 `X @ X.T`。
- 多项式组合。
- 再乘回 `X`。

如果做 5 次迭代，就会增加多轮矩阵乘。矩阵很大时，这些计算不可忽略；矩阵很小时，kernel launch 和调度开销可能更明显。

所以 Muon 的性能不只取决于 FLOPs，还取决于：

- 每个矩阵 shape。
- 参数矩阵数量。
- 是否有 fused / batched 实现。
- 临时矩阵 dtype。
- optimizer step 是否能与通信或其他计算重叠。
- small GEMM 是否能吃满 GPU。

### 为什么不是直接 SVD

SVD 更精确，但系统成本通常太高：

- 延迟更高。
- 实现复杂。
- 对大量不同 shape 的小矩阵不友好。
- 分布式训练中更难组合。

Newton-Schulz 的意义是用固定次数的矩阵乘换取足够好的近似。2026 年 ICLR 接收的 Muon with Newton-Schulz 收敛分析，也正是在解释“少量 Newton-Schulz step 为什么在理论上接近精确极分解方向”。

## 一个极简训练循环

Muon 与 AdamW 混用时，训练循环通常会出现两组 optimizer：

```python
muon_params = []
adamw_params = []

for name, p in model.named_parameters():
    if p.ndim == 2 and "embed" not in name and "lm_head" not in name:
        muon_params.append(p)
    else:
        adamw_params.append(p)

optim_muon = torch.optim.Muon(muon_params, lr=muon_lr, weight_decay=muon_wd)
optim_adamw = torch.optim.AdamW(adamw_params, lr=adamw_lr, weight_decay=adamw_wd)

loss.backward()
optim_muon.step()
optim_adamw.step()
optim_muon.zero_grad(set_to_none=True)
optim_adamw.zero_grad(set_to_none=True)
```

实际系统还要加上：

- gradient accumulation。
- gradient clipping。
- mixed precision / GradScaler。
- distributed gradient sync。
- scheduler。
- checkpoint。
- parameter group 日志。

关键不是这段代码，而是参数分组必须可解释、可复现、可 checkpoint。

## 参数分组怎么做

Muon 的参数分组比 AdamW 更重要。因为一旦分错，优化语义就变了。

一个保守策略是：

| 参数类型 | 建议 |
| --- | --- |
| Transformer hidden linear weights | 优先考虑 Muon |
| Attention Q/K/V/O projection | 可考虑 Muon |
| MLP / FFN projection | 可考虑 Muon |
| MoE expert FFN weights | 可考虑 Muon，但要关注 expert 分布式切分 |
| embedding | 通常 AdamW |
| LM head | 通常 AdamW 或单独实验 |
| bias | AdamW 或不 decay |
| RMSNorm / LayerNorm | AdamW 或不 decay |
| rope / scalar 参数 | 不交给 Muon |

参数分组最好不要只靠 `ndim == 2`，因为 embedding 和 LM head 也可能是二维矩阵。

更稳妥的方式是按模块名、模块类型和参数名共同判断：

```text
include:
  transformer.blocks.*.attn.*_proj.weight
  transformer.blocks.*.mlp.*_proj.weight
  transformer.blocks.*.experts.*.weight

exclude:
  *.embed*
  *.lm_head*
  *.norm*
  *.bias
```

每次实验都应把最终参数分组打印出来：

```text
Muon params:
  count
  total elements
  dtype
  top modules

AdamW params:
  count
  total elements
  dtype
  excluded reasons
```

否则 loss 出问题时很难判断是 optimizer 本身、参数分组，还是学习率配置的问题。

## Weight Decay 和更新尺度

早期 Muon 实现并不一定使用 decoupled weight decay。但大规模 LLM 训练报告指出，weight decay 对 Muon 扩展到大模型很重要。

直觉是：

- 正交化更新可能让某些权重范数长期增长。
- 大模型长时间训练时，权重 RMS 或层输出 RMS 失控会影响 BF16 数值范围和稳定性。
- Decoupled weight decay 给权重规模一个持续约束。

更新尺度同样关键。正交化后的矩阵更新不是 AdamW 那种逐元素归一化，它的 RMS 会受矩阵 shape 影响。

工程上要记录：

- Muon learning rate。
- AdamW learning rate。
- Muon weight decay。
- AdamW weight decay。
- `adjust_lr_fn` 或等价 scaling 规则。
- Newton-Schulz steps。
- momentum / nesterov。
- 哪些参数使用 Muon。

如果只记录“用了 Muon”，实验不可复现。

## 混合精度与数值稳定性

Muon 常见实现会在 Newton-Schulz 中使用 BF16，以降低成本。但这并不表示所有状态都可以随便低精度。

需要区分：

- 模型权重 dtype。
- gradient dtype。
- momentum buffer dtype。
- Newton-Schulz temporary dtype。
- master weight dtype。
- reduce-scatter / all-gather dtype。

几个常见风险：

- BF16 下矩阵范数过大或过小，正交化输入尺度不稳定。
- 临时矩阵乘产生 NaN/Inf。
- update scale 对某些矩阵过大，训练早期 loss spike。
- 混合 Muon + AdamW 时，两组参数的 LR scheduler 语义不一致。
- checkpoint 恢复后 optimizer state dtype 或 param group 顺序错位。

监控项至少包括：

- loss / grad norm。
- weight RMS。
- update RMS。
- Muon update norm。
- Newton-Schulz 输出是否出现 NaN/Inf。
- 各 parameter group 的 learning rate。

## 和 ZeRO/FSDP 的关系

AdamW 的更新是逐元素的，所以 ZeRO-1 / FSDP shard optimizer state 很自然：每个 rank 只更新自己负责的一段参数和状态。

Muon 麻烦在于：正交化需要看到完整矩阵，至少需要看到一个语义完整的矩阵块。

如果一个矩阵 `W` 被 DP 维度切成 shard：

```text
rank 0: W_part_0
rank 1: W_part_1
rank 2: W_part_2
rank 3: W_part_3
```

对每个 shard 单独做 Newton-Schulz，和对完整 `W` 做 Newton-Schulz 再切回 shard，结果通常不同。

所以分布式 Muon 需要明确选择：

1. gather full gradient / momentum matrix，再计算 Muon update。
2. 只对本地 shard 做近似 Muon，接受语义变化。
3. 按参数切分方式设计矩阵块，使每个块仍然有合理语义。

Moonshot 的 Distributed Muon 方案采用类似 ZeRO-1 的状态切分，同时增加 DP gather 来恢复 full gradient matrix，再做 Newton-Schulz update，然后只保留本 rank 负责的分片。

这带来一个典型流程：

```mermaid
flowchart TD
    A["各 rank 本地 gradient shard"] --> B["DP reduce-scatter<br/>得到正确梯度分片"]
    B --> C["更新本地 momentum shard"]
    C --> D["DP gather<br/>恢复完整矩阵"]
    D --> E["Newton-Schulz<br/>计算完整 Muon update"]
    E --> F["切回本 rank 分片"]
    F --> G["更新本地参数和状态"]
    G --> H["必要时 all-gather 参数"]
```

系统代价包括：

- 额外 gather。
- full matrix 临时显存。
- Newton-Schulz 计算。
- update 分片丢弃带来的中间结果浪费。
- 与已有 FSDP/ZeRO 状态机的集成复杂度。

## 和 Tensor Parallel 的关系

Tensor Parallel 会把一个线性层的矩阵按行或列切开。例如 column parallel linear：

```text
W = [W0, W1, W2, W3]
```

每个 rank 只持有一部分输出列。此时 Muon 有几种可能：

- 在每个 TP shard 上单独做 Muon。
- gather TP shard 后对完整矩阵做 Muon。
- 把 TP shard 视为独立矩阵，承认语义和 full matrix Muon 不同。

没有免费的选择。

如果 gather：

- 语义更接近完整矩阵。
- 通信和临时显存增加。
- optimizer step 更复杂。

如果 shard-local：

- 实现简单。
- 通信少。
- 更新方向依赖切分方式，TP size 改变可能导致优化行为改变。

所以使用 Muon 做 TP 训练时，benchmark 必须固定 TP size 和切分策略。不能假设 `TP=1` 的调参结果直接搬到 `TP=8`。

## 和 Pipeline Parallel 的关系

Pipeline Parallel 对 Muon 的影响相对间接。

主要问题是：

- 不同 stage 的参数 shape 不同，Muon optimizer time 可能不均衡。
- 某些 stage 有 embedding / lm_head，Muon 参数比例较低。
- optimizer step 通常发生在一次 global batch 完成后，可能造成 stage 间等待。
- 如果 optimizer step 很重，pipeline bubble 之外还会出现 optimizer tail。

排查方式：

- 分 stage 统计 optimizer step time。
- 分 stage 统计 Muon 参数量。
- 看 optimizer step 是否成为 iteration 末尾的 exposed time。
- 对 embedding-heavy stage 单独处理 AdamW 参数和 checkpoint。

## 和 MoE 训练的关系

MoE 模型里，专家 FFN 是大量二维矩阵，理论上很适合 Muon。但 MoE 训练本身已经有 Expert Parallel、AllToAll、load balance、token dispatch/combine 等复杂性，Muon 会叠加新的系统问题。

需要重点关注：

- expert weight 是否按 EP 切分。
- 每个 rank 持有哪些 expert。
- expert matrix 是否完整在单卡上。
- Muon momentum state 是否随 expert 放置。
- expert 迁移、checkpoint、resume 是否保持 optimizer state 对齐。
- 大 EP / 小 EP 改变后，Muon 的参数分组和更新语义是否变化。

如果每个 expert 的矩阵完整放在某个 rank 上，Muon 实现会简单很多。如果 expert 矩阵还被 TP 或 FSDP 切开，就要处理 full matrix 语义。

MoE 场景还要监控：

- router 是否稳定。
- expert load balance loss。
- 不同 expert 的 weight RMS / update RMS。
- hot expert 和 cold expert 的 optimizer state 是否有异常。

## Checkpoint 与恢复

Muon checkpoint 不能只保存模型权重。

至少要保存：

- Muon optimizer param groups。
- momentum buffers。
- AdamW param groups。
- AdamW `m/v` states。
- master weights，如果有。
- scheduler state。
- gradient scaler state，如果用 FP16 AMP。
- global step / consumed tokens。
- 参数分组规则版本。

对于 Muon 特别要注意：

- 参数顺序变了，momentum buffer 会错位。
- 模块重命名后，state dict 可能加载到错误参数。
- Muon 与 AdamW 参数组划分改变，会导致老 checkpoint 难以恢复。
- TP/FSDP/ZeRO world size 改变时，sharded optimizer state 需要 reshard。
- `adjust_lr_fn`、`ns_steps`、momentum 等配置应写进 checkpoint metadata。

推荐在 checkpoint metadata 中记录：

```yaml
optimizer:
  muon:
    enabled: true
    param_rule_version: v1
    ns_steps: 5
    momentum: 0.95
    nesterov: true
    adjust_lr_fn: match_rms_adamw
    weight_decay: 0.1
  adamw:
    used_for:
      - embedding
      - lm_head
      - norm
      - bias
```

这能避免半年后只剩一个 checkpoint，却不知道哪些参数当时到底用了哪个 optimizer。

## Benchmark 应该怎么做

Muon 的 benchmark 不能只比较“相同步数下 loss 更低”。训练系统关心的是更快、更省、更稳定。

至少做四组对比：

| 对比项 | 目的 |
| --- | --- |
| AdamW baseline | 有强基线，避免把弱 AdamW 调参当成 Muon 优势 |
| Muon sample efficiency | 看相同 token / step 下 loss |
| Muon wall-clock efficiency | 看达到目标 loss 的真实时间 |
| Muon system overhead | 看 optimizer step time、显存、通信和稳定性 |

### 关键指标

训练指标：

- training loss。
- validation loss。
- downstream eval，如果有。
- tokens to target loss。
- steps to target loss。

系统指标：

- step time。
- optimizer step time。
- forward time。
- backward time。
- communication time。
- peak memory。
- tokens/s。
- MFU。
- checkpoint size。
- checkpoint save/load time。

稳定性指标：

- NaN/Inf 次数。
- loss spike。
- grad norm。
- weight RMS。
- update RMS。
- resume 后 loss 是否连续。

### 需要固定的变量

如果比较 AdamW 和 Muon，必须固定：

- 模型结构。
- dataset 和 tokenization。
- sequence length。
- global batch tokens。
- warmup steps。
- scheduler。
- precision。
- parallelism layout。
- hardware。
- profiler 采样窗口。
- checkpoint/resume 策略。

否则很容易比较到别的变量。

## 性能剖析重点

Profiler 里要单独看 Muon optimizer step。

重点问题：

1. Newton-Schulz 的 GEMM 是否占明显时间。
2. small GEMM 是否过碎。
3. 是否有大量 tensor reshape / transpose / clone。
4. gather full matrix 是否造成通信尾巴。
5. temporary buffer 是否造成 peak memory。
6. Muon step 是否可以和参数 all-gather 或其他通信重叠。
7. checkpoint 保存 optimizer state 是否明显变大或变慢。

如果看到 optimizer step time 上升，不一定表示 Muon 不值得。还要看：

- 是否更少 token 达到目标 loss。
- 是否总体 wall-clock 更短。
- 是否降低长期 optimizer state 显存。
- 是否允许更大 batch 或更长 context。

系统评估要以“达到目标质量的总成本”为准。

## 常见优化方向

### 使用成熟实现

优先使用框架或经过验证的实现。Muon 涉及参数分组、Newton-Schulz、dtype、weight decay、update scale、checkpoint 和分布式切分，手写很容易漏细节。

### 减少小矩阵碎片

大量小矩阵逐个做 Newton-Schulz 会产生明显 launch overhead。可以考虑：

- batched GEMM。
- grouped kernel。
- fused optimizer step。
- 对过小矩阵回退 AdamW。

### 控制临时显存

Newton-Schulz 临时矩阵可能造成峰值显存上升。优化方向包括：

- BF16 temporary。
- in-place buffer reuse。
- 按参数分组分批执行。
- 避免在同一时间保留 full update 和 full gradient。

### 和分布式通信重叠

Distributed Muon 可能需要 gather。可以探索：

- gather 和局部 momentum update 重叠。
- optimizer reduce-scatter 和 parameter all-gather 重叠。
- 按 layer 顺序流水执行 optimizer step。
- 把 Muon step 放进更大的训练 runtime 调度里。

### 保守选择参数范围

初次引入 Muon 时，不建议一次性覆盖所有二维参数。更稳妥路线：

1. 只覆盖 Transformer hidden linear weights。
2. embedding、norm、bias、lm_head 保持 AdamW。
3. 记录每类参数的 loss、update RMS、weight RMS。
4. 稳定后再探索更多参数。

## 常见误区

### 误区一：Muon 就是 AdamW 的直接替代

不完全是。Muon 主要面向二维 hidden layer 参数，实践中经常和 AdamW 混用。

### 误区二：只要 `p.ndim == 2` 就全部用 Muon

不稳妥。Embedding 和 LM head 也可能是二维矩阵，但它们的训练语义和 hidden layer weight 不同。

### 误区三：Muon state 少，所以一定更快

不一定。Muon 可能减少长期状态，但 Newton-Schulz、临时矩阵、gather 和 small GEMM 都会增加 optimizer step 成本。

### 误区四：在 shard 上做 Muon 等价于 full matrix Muon

通常不等价。Muon 是矩阵级操作，切分方式会影响更新方向。

### 误区五：只看 step loss 就能判断 Muon 是否更好

不够。要看 tokens to target loss、wall-clock to target loss、系统开销和稳定性。

## 设计检查清单

引入 Muon 前，可以逐项确认：

- 哪些参数使用 Muon？
- 哪些参数继续使用 AdamW？
- 参数分组是否按名字和模块类型记录？
- Muon LR、AdamW LR 是否分别记录？
- weight decay 是否启用？
- `adjust_lr_fn` 或更新尺度策略是什么？
- Newton-Schulz steps 是多少？
- momentum / nesterov 配置是什么？
- Muon momentum dtype 是什么？
- Newton-Schulz temporary dtype 是什么？
- 是否有 FP32 master weight？
- FSDP/ZeRO/TP 下是否需要 full matrix gather？
- world size 改变后 checkpoint 能否恢复？
- profiler 是否单独统计 optimizer step？
- benchmark 是否比较 wall-clock to target loss？

## 小结

Muon 的价值不只是“换一个优化器”。它代表一种训练系统趋势：optimizer 不再只是逐元素更新参数，而是利用矩阵结构改变更新方向。

对训练系统来说，Muon 的关键点是：

- 它主要适合 hidden layer 的二维矩阵参数。
- 它通过 momentum matrix 的 Newton-Schulz 正交化产生更新方向。
- 它可能减少 AdamW 二阶矩状态，但会增加矩阵乘、临时 buffer 和分布式 gather。
- 它和 ZeRO/FSDP/TP 的组合必须明确 full matrix 语义。
- 它的 benchmark 必须同时看收敛、wall-clock、显存、通信和稳定性。

如果只把 Muon 当成一行 optimizer 替换，很容易踩坑。把它当成“矩阵级 optimizer subsystem”来设计，才更接近大模型训练系统里的真实问题。

## 参考资料

- [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/)
- [PyTorch: torch.optim.Muon](https://docs.pytorch.org/docs/2.12/generated/torch.optim.Muon.html)
- [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)
- [Convergence of Muon with Newton-Schulz](https://arxiv.org/abs/2601.19156)
- [The Newton-Muon Optimizer](https://arxiv.org/abs/2604.01472)
