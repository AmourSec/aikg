---
title: Optimizer 与 Scheduler 系统成本
domain: training-systems
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-06-11
---

# Optimizer 与 Scheduler 系统成本

训练系统里，forward/backward 通常最显眼，但 optimizer step 也可能是大头。特别是 AdamW 这类自适应优化器，它不仅更新参数，还要维护大量 optimizer state。

一句话理解：

> Optimizer 与 Scheduler 的系统成本，来自参数更新本身、optimizer state 读写、混合精度 master weight、梯度裁剪、分布式 sharding、checkpoint 和 step 语义。

这篇不重点推导优化算法，而是回答系统问题：为什么 optimizer 这么占显存和时间，如何判断它是不是瓶颈，scheduler 在 gradient accumulation 和恢复训练中为什么容易出错。

## 一个 optimizer step 做了什么

一次训练 step 的后半段通常是：

```text
backward -> gradient sync -> gradient clipping -> optimizer step -> scheduler step -> zero grad
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

这意味着 optimizer step 是一个大量逐元素读写的阶段。它不一定 FLOPs 很高，但非常吃显存带宽和 kernel launch。

## AdamW 为什么显存贵

Adam 系列优化器通常维护两个动量状态：

```text
m: first moment
v: second moment
```

混合精度训练还可能有 FP32 master weight。

假设一个模型有 `N` 个参数，训练权重和梯度用 BF16/FP16，AdamW 状态用 FP32：

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

这还没算 activation、temporary buffers、通信 buffer 和碎片。

所以 AdamW 的系统成本不仅是“多做一点数学”，而是每个参数要维护长期状态。ZeRO/FSDP 很大程度上就是为了切这些状态。

## Adam 和 AdamW 的差别

Adam 维护一阶矩和二阶矩：

```text
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad^2
```

然后用：

```text
param = param - lr * m_hat / (sqrt(v_hat) + eps)
```

AdamW 的关键是 decoupled weight decay。它把 weight decay 从 Adam 的梯度自适应缩放里拆出来，而不是简单把 `weight_decay * param` 加进梯度。

系统上，这意味着 optimizer update 里有额外的参数衰减操作，但更重要的是配置语义不同：

- Adam + L2 regularization 不等价于 AdamW。
- weight decay 和 learning rate schedule 的交互更清楚。
- 参数分组时，哪些参数 decay、哪些不 decay 必须明确。

常见做法是：

- Linear / attention / MLP weight 做 weight decay。
- bias、LayerNorm/RMSNorm weight、embedding 等常常不做 decay，具体取决于训练 recipe。

## Optimizer step 为什么可能慢

Optimizer step 可能慢，原因包括：

- 参数数量巨大，逐元素读写量大。
- AdamW 需要读写 `p/g/m/v/master`。
- 参数被拆成很多小 tensor，kernel launch 多。
- Python for-loop optimizer 开销高。
- gradient clipping 额外扫描所有梯度。
- ZeRO/FSDP 需要 shard 聚合、通信或 CPU/NVMe offload。
- checkpoint 保存 optimizer state 时写入量巨大。

普通 GEMM 是大块矩阵乘，Tensor Core 吞吐高。Optimizer step 往往是很多 elementwise 操作，更容易受内存带宽和框架调度影响。

## Fused optimizer 是什么

普通 optimizer 可能对每个参数 tensor 分别执行多个操作：

```text
update m
update v
compute denom
apply weight decay
update param
```

Fused optimizer 把多个逐元素操作融合到更少 kernel 中。

好处：

- 减少 kernel launch。
- 减少中间 tensor 写回。
- 提高内存访问效率。
- 对大量小 tensor 更友好。

代价：

- dtype、device、参数布局支持有限。
- 调试更难。
- 和 graph capture、compile、FSDP/ZeRO 组合时要验证。
- 可能对 peak memory 有不同影响。

PyTorch 的 AdamW 提供 foreach 和 fused 实现选项。foreach 通常比 Python for-loop 快，但可能使用额外 tensorlist 中间内存；fused 理论上进一步减少开销，但具体支持和稳定性要看版本、dtype、device。

## Gradient clipping 的成本

大模型训练常用 gradient clipping，例如按 global norm 裁剪。

global norm 需要：

1. 遍历所有梯度。
2. 计算平方和。
3. 跨 data parallel ranks 做规约。
4. 得到 norm。
5. 如果超过阈值，缩放所有梯度。

这不是免费操作。它会增加：

- 梯度读写。
- 额外 reduction。
- 同步点。
- 与 AMP unscale 的顺序要求。

混合精度下通常要先 unscale gradients，再做 gradient clipping。否则裁剪的是 scaled gradient，语义错误。

## Scheduler 做了什么

Scheduler 控制 learning rate 随 step 变化。

常见策略：

- warmup。
- cosine decay。
- linear decay。
- constant with warmup。
- polynomial decay。
- step decay。

Scheduler 本身计算成本通常很小，但系统语义很重要。

训练中有多个 step 概念：

```text
micro-step:
  每个 micro-batch forward/backward

optimizer step:
  梯度累积完成后更新一次参数

global step:
  日志、checkpoint、eval 使用的训练步数，定义需要明确

token step:
  已训练 token 数
```

Scheduler 通常应该按 optimizer step 更新，而不是每个 micro-batch 更新。否则 gradient accumulation 改变时，学习率曲线也会变。

## Gradient accumulation 下的 step 语义

假设：

```text
gradient_accumulation_steps = 8
```

训练流程是：

```text
micro-step 1: forward/backward
micro-step 2: forward/backward
...
micro-step 8: forward/backward
optimizer step
scheduler step
zero grad
```

如果 scheduler 每个 micro-step 都 step，一次有效 batch 内学习率会变化 8 次，这通常不是预期语义。

同样，logging、checkpoint、eval 也要明确按什么 step 计数：

- 每多少 optimizer steps 保存 checkpoint？
- 每多少 tokens 做 eval？
- warmup 是按 steps 还是 tokens？
- resume 后 scheduler 是否从正确位置继续？

这些问题不是数学细节，而是长期训练复现的基础。

## 分布式 optimizer

Data Parallel 中，每个 rank 默认保存完整 optimizer state。对 AdamW 来说，这很贵。

ZeRO/FSDP 的做法是把 optimizer state shard 到不同 rank：

```text
Rank 0: optimizer state shard 0
Rank 1: optimizer state shard 1
Rank 2: optimizer state shard 2
Rank 3: optimizer state shard 3
```

好处：

- 每卡 optimizer state 显存下降。
- 支持更大模型。

代价：

- optimizer step 需要处理 sharded 参数。
- 梯度 reduce-scatter 和参数同步更复杂。
- checkpoint 需要保存 sharded optimizer state。
- resume 时必须恢复分片映射。
- offload 到 CPU/NVMe 会引入带宽和延迟瓶颈。

分布式 optimizer 的核心不是“optimizer 算法变了”，而是 optimizer state 的放置和更新路径变了。

## CPU / NVMe offload 的取舍

如果 GPU 显存不够，可以把 optimizer state offload 到 CPU，甚至 NVMe。

收益：

- GPU 显存大幅下降。
- 可以训练更大模型。

代价：

- CPU/GPU 或 NVMe/GPU 传输可能成为瓶颈。
- optimizer step 可能变慢。
- 需要 overlap 和 prefetch。
- checkpoint 和恢复路径更复杂。
- 训练稳定性更依赖系统 I/O。

Offload 是容量工具，不是性能优化的默认答案。它适合“否则放不下”的场景。

## 参数分组的系统影响

Optimizer param groups 不只是超参数管理，也影响实现效率。

参数分组常用于：

- 对某些参数关闭 weight decay。
- 给 embedding 或 head 不同学习率。
- 冻结部分参数。
- LoRA / adapter 参数单独优化。
- Muon 和 AdamW 混合使用。

分组太碎会带来：

- 更多 Python 调度。
- 更少 fusion 机会。
- optimizer state 更复杂。
- checkpoint 映射更难。

但分组太粗，又可能把不该 decay 的参数放进 decay group。

工程上要在训练 recipe 清晰和系统效率之间平衡。

## Optimizer 与 checkpoint

完整恢复训练通常需要保存：

- model parameters。
- optimizer states。
- scheduler state。
- GradScaler state。
- RNG state。
- data loader / sampler state。
- global step / consumed tokens。

Optimizer state 往往是 checkpoint 最大部分之一。AdamW 的 `m/v/master` 可能比模型参数本身还大。

保存 checkpoint 时要注意：

- full checkpoint 是否会在 rank 0 聚合导致 OOM？
- sharded checkpoint 是否能跨并行配置恢复？
- optimizer state 是否和 param group 顺序绑定？
- scheduler state 是否在 optimizer 后正确加载？
- resume 后 LR、loss scale、global step 是否正确？

PyTorch 文档也提醒 optimizer state 和 scheduler 加载顺序会影响 LR。这个细节在长期训练恢复中很实际。

## Benchmark 时看什么

评估 optimizer/scheduler 成本至少看：

| 指标 | 作用 |
| --- | --- |
| Optimizer step time | 参数更新是否瓶颈 |
| Scheduler time | 通常小，但异常时要看 |
| Grad clipping time | 全局 norm 和缩放成本 |
| Peak memory | optimizer state 和中间 tensor |
| Memory bandwidth | optimizer 是否带宽受限 |
| Kernel count | 是否小 kernel 太多 |
| CPU time | Python for-loop 或 offload 是否瓶颈 |
| Checkpoint size | optimizer state 保存压力 |
| Resume correctness | LR、step、state 是否恢复一致 |

实验要固定：

- optimizer 类型。
- foreach/fused 配置。
- param groups。
- mixed precision 配置。
- FSDP/ZeRO stage。
- offload 配置。
- gradient accumulation。
- scheduler step 语义。

## 常见优化方向

### 使用 fused 或 foreach optimizer

优先避免 Python for-loop 式逐 tensor 更新。对大模型，fused/foreach 通常能减少 kernel launch 和调度开销。但要同时看 peak memory 和版本支持。

### 减少 param group 碎片

把参数分组保持在必要范围内。不要为每个小模块创建独立 param group，除非训练 recipe 真的需要。

### Shard optimizer state

如果 optimizer state 是主要显存瓶颈，考虑 ZeRO/FSDP 的 optimizer state sharding。比单纯减 batch 更直接。

### 谨慎使用 offload

Offload 解决容量，但可能牺牲吞吐。使用前要估算 PCIe/NVMe 带宽和 optimizer step time。

### 统一 step 语义

明确 optimizer step、global step、tokens consumed、scheduler step、checkpoint interval 的关系。长期训练最怕 step 语义漂移。

## 常见误区

### 误区一：optimizer step 只是很小的收尾

不一定。对大模型 AdamW，optimizer step 要读写大量状态，可能成为明显瓶颈。

### 误区二：显存不够先减 batch

如果 OOM 主要来自 optimizer state，减 micro-batch 不解决根因。应该考虑 ZeRO/FSDP、optimizer sharding 或 offload。

### 误区三：scheduler 每个 micro-step 调一次也差不多

不对。gradient accumulation 下，这会改变学习率曲线。scheduler step 语义必须和 optimizer step 对齐。

### 误区四：fused optimizer 一定更省显存

fused 主要减少 kernel 和中间读写，不一定总是降低 peak memory。foreach 有时还会增加 tensorlist 中间内存。

### 误区五：只保存模型参数就能恢复训练

不能完整恢复。缺 optimizer state、scheduler state、loss scaler、RNG 和数据位置，训练轨迹会变。

## 设计检查表

设计 optimizer/scheduler 时，可以逐项检查：

- optimizer 是 AdamW、SGD、Muon，还是混合？
- optimizer state 每参数多少字节？
- 是否保留 FP32 master weights？
- foreach/fused/capturable 配置是什么？
- param groups 是否必要且不过度碎片？
- weight decay 排除了哪些参数？
- gradient clipping 在 unscale 后还是前？
- scheduler 按 optimizer step、micro-step 还是 token step？
- gradient accumulation 改变时 LR 曲线是否不变？
- FSDP/ZeRO 如何 shard optimizer state？
- checkpoint 是否保存 optimizer 和 scheduler state？
- resume 后 LR、global step、tokens consumed 是否正确？

## 小结

Optimizer 与 Scheduler 是训练系统的核心组成，不是训练循环最后的附属细节。

系统上要抓住：

- AdamW 贵在 `m/v/master` 等长期状态。
- optimizer step 常常受内存带宽、kernel launch 和 Python 调度影响。
- fused/foreach optimizer 能改善执行效率，但要看显存和兼容性。
- ZeRO/FSDP 通过 sharding optimizer state 解决容量问题。
- scheduler 的成本小，但 step 语义非常重要。
- checkpoint/resume 必须恢复 optimizer、scheduler 和 step 状态。

如果训练系统只优化 forward/backward，而忽略 optimizer step、scheduler step 和 checkpoint state，就很容易在大模型训练后半段遇到显存、吞吐和复现问题。

## 参考资料

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- [PyTorch: AdamW](https://docs.pytorch.org/docs/2.12/generated/torch.optim.AdamW.html)
- [DeepSpeed Configuration JSON](https://www.deepspeed.ai/docs/config-json/)
- [DeepSpeed ZeRO-Offload](https://www.deepspeed.ai/tutorials/zero-offload/)
