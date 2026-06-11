---
title: Collective 通信原语与通信量模型
domain: training-systems
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-06-12
---

# Collective 通信原语与通信量模型

分布式训练不是“把同一个 Python 脚本跑在多张 GPU 上”这么简单。多个 rank 之间必须不断交换数据，才能让模型参数、梯度、activation、expert token、optimizer state 保持正确关系。

这些交换通常不是随意发送，而是通过一组标准通信原语完成：

- AllReduce。
- ReduceScatter。
- AllGather。
- Broadcast。
- Reduce。
- AllToAll。
- Barrier。
- Point-to-point send / recv。

这些原语是理解 Data Parallel、FSDP / ZeRO、Tensor Parallel、Pipeline Parallel、Expert Parallel、Sequence / Context Parallel 和通信重叠的公共基础。

如果不理解 collective，本章后面的很多现象都会显得零散：

- 为什么 DDP 用 AllReduce 同步梯度。
- 为什么 ZeRO / FSDP 喜欢 ReduceScatter 和 AllGather。
- 为什么 Tensor Parallel 每层都可能通信。
- 为什么 MoE 的 AllToAll 会变成尾延迟瓶颈。
- 为什么某个 rank 晚到 collective 会拖慢所有 rank。
- 为什么小消息很多时，带宽看起来很差。
- 为什么通信不慢，但 step time 仍然被 exposed communication 卡住。

## 先给结论

Collective 通信可以理解为多个 rank 共同参与的一次数据交换。它的关键点是：

1. 同一个 process group 里的所有 rank 必须以一致顺序进入相同 collective。
2. 通信 tensor 的 shape、dtype、设备、语义必须匹配。
3. 通信耗时不只取决于字节数，还取决于消息数量、算法、拓扑、rank placement 和 overlap。
4. AllReduce 常用于复制式状态同步，ReduceScatter / AllGather 常用于 sharded 状态，AllToAll 常用于 token / sequence / expert 交换。
5. 小消息通常 latency-bound，大消息通常 bandwidth-bound。
6. collective 本身可能很快，但如果暴露在关键路径上，仍然会拖慢 step。

粗略理解可以用一句话：

> 分布式训练优化，本质上经常是在选择“哪些数据复制、哪些数据切分、什么时候通信、通信能不能被计算覆盖”。

## Rank、Process Group 与 Collective

先区分几个概念。

| 概念 | 含义 |
| --- | --- |
| rank | 分布式训练中的一个进程编号。通常一个 GPU 对应一个 rank。 |
| world size | 全部参与训练的 rank 数量。 |
| process group | 一组 rank 的通信集合。collective 只在这个 group 内发生。 |
| communicator | 底层通信库维护的通信上下文，例如 NCCL communicator。 |
| collective | group 内多个 rank 共同参与的通信操作。 |

真实训练里不会只有一个 group。一个 3D 并行训练可能同时有：

- DP group。
- TP group。
- PP group。
- EP group。
- CP / SP group。
- optimizer / checkpoint 相关 group。

示例：

```text
world ranks: 0 1 2 3 4 5 6 7

TP group 0: 0 1
TP group 1: 2 3
TP group 2: 4 5
TP group 3: 6 7

DP group 0: 0 2 4 6
DP group 1: 1 3 5 7
```

同一个 rank 可能属于多个 group。它在不同时间参与不同 collective。

## Collective 的基本规则

Collective 的危险之处在于：它要求多个 rank 协同。

常见规则包括：

- group 内所有 rank 都要调用。
- 调用顺序要一致。
- op 类型要一致。
- tensor shape 要匹配。
- dtype 要匹配。
- device 要匹配。
- stream / async 使用要正确。
- timeout 和错误处理要明确。

如果 rank 0 调用 AllReduce，但 rank 1 调用 AllGather，程序通常不会报一个清晰的 Python exception，而是可能 hang、timeout 或 NCCL 报错。

如果 rank 0 的 tensor 是 `[1024]`，rank 1 的 tensor 是 `[2048]`，也可能出问题。

所以调分布式训练时，经常要问：

```text
哪些 rank 在哪个 group 里？
它们是否以同样顺序进入 collective？
每个 collective 的 tensor shape / dtype 是否一致？
有没有某个 rank 在 collective 前慢了？
```

## 通信量模型：Latency 与 Bandwidth

通信耗时常用一个简化模型理解：

```text
time ≈ latency + bytes / bandwidth
```

更细一点：

```text
time ≈ alpha * num_messages + beta * bytes + congestion + synchronization_wait
```

其中：

- `alpha` 表示每次通信启动、调度、协议、同步带来的固定开销。
- `beta` 表示每字节传输成本。
- `num_messages` 表示消息数量。
- `bytes` 表示总数据量。
- `congestion` 表示链路竞争、网络拥塞、多 rail 不均衡等。
- `synchronization_wait` 表示等待慢 rank 进入 collective 的时间。

这解释了两个常见现象：

1. 小消息很多时，带宽利用率很差，因为 latency 占主导。
2. 大消息时，更接近受互连带宽限制。

训练系统优化不能只说“通信数据量是多少”，还要看：

- 消息是否太碎。
- collective 是否足够大。
- 是否有很多小 bucket。
- 是否有 rank 晚到。
- 是否和计算重叠。
- 是否跨节点。
- 是否走了慢链路。

## 常见 Collective 总览

下面先给一个总览。

| 原语 | 输入 | 输出 | 常见训练用途 |
| --- | --- | --- | --- |
| AllReduce | 每个 rank 有一个完整 tensor | 每个 rank 得到 reduce 后的完整 tensor | DDP 梯度同步、loss/grad norm 聚合。 |
| ReduceScatter | 每个 rank 有完整或分块输入 | 每个 rank 得到 reduce 后的一片 shard | ZeRO/FSDP 梯度切分、AllReduce 分解。 |
| AllGather | 每个 rank 有一片 shard | 每个 rank 得到完整拼接结果 | FSDP 参数 all-gather、TP 输出拼接。 |
| Broadcast | root rank 有数据 | 所有 rank 得到 root 数据 | 初始化参数、分发配置或小状态。 |
| Reduce | 每个 rank 有数据 | root rank 得到 reduce 结果 | 指标聚合、checkpoint manifest 聚合。 |
| AllToAll | 每个 rank 有发给其他 rank 的分块 | 每个 rank 收到来自所有 rank 的分块 | MoE token dispatch、sequence/context exchange。 |
| Barrier | 无主要数据 | 所有 rank 等待同步 | 调试、阶段同步，不适合滥用。 |
| Send / Recv | 点对点 | 点对点 | Pipeline stage 间 activation / grad 传递。 |

## AllReduce

AllReduce 的语义是：

```text
每个 rank 都有一个 tensor
对这些 tensor 做 reduce
每个 rank 都得到 reduce 后的完整结果
```

以 SUM 为例：

```text
Rank 0: [1, 2]
Rank 1: [3, 4]
Rank 2: [5, 6]
Rank 3: [7, 8]

AllReduce SUM 后，每个 rank 都得到：
[16, 20]
```

如果要平均梯度，再除以 world size：

```text
avg = sum / world_size
```

AllReduce 是普通 DDP 最常见的梯度同步方式。

### AllReduce 为什么适合 DDP

DDP 中，每个 rank 有一份完整模型副本。

每个 rank 用不同数据算出本地梯度：

```text
rank 0 grad = g0
rank 1 grad = g1
rank 2 grad = g2
rank 3 grad = g3
```

为了让所有模型副本继续一致，需要所有 rank 使用同一份平均梯度：

```text
g = (g0 + g1 + g2 + g3) / 4
```

AllReduce 正好能让每个 rank 都拿到这个结果。

### AllReduce 的通信量直觉

如果 tensor 大小是 `N` bytes，rank 数是 `P`。

不同算法的实际路径不同，但一个常见直觉是：

```text
Ring AllReduce per-rank 传输量约为 2 * (P - 1) / P * N
```

当 `P` 很大时，接近：

```text
2N
```

这不是说网络总共只传 2N，而是每个 rank 视角下大约发送/接收这个量级。实际还要看 ring、tree、hierarchical、NVLink、IB/RoCE、多 rail、chunk size 等。

### AllReduce 常见问题

- tensor 太小，latency 占比高。
- bucket 太大，启动太晚，overlap 差。
- bucket 太小，collective 数量太多。
- 某个 rank backward 慢，其他 rank 等它进入 collective。
- 跨节点 AllReduce 走慢链路。
- rank placement 没匹配 GPU / NIC 拓扑。

## ReduceScatter

ReduceScatter 可以理解为：

```text
先 reduce
再 scatter
每个 rank 只拿到结果的一片 shard
```

例子：

```text
Rank 0 input: [a0, a1, a2, a3]
Rank 1 input: [b0, b1, b2, b3]
Rank 2 input: [c0, c1, c2, c3]
Rank 3 input: [d0, d1, d2, d3]

reduce sum:
[a0+b0+c0+d0,
 a1+b1+c1+d1,
 a2+b2+c2+d2,
 a3+b3+c3+d3]

scatter:
Rank 0 gets shard 0
Rank 1 gets shard 1
Rank 2 gets shard 2
Rank 3 gets shard 3
```

它的意义是：每个 rank 不再保存完整 reduce 结果。

### ReduceScatter 为什么适合 ZeRO / FSDP

ZeRO / FSDP 的核心是 sharding。

如果每个 rank 最终只负责一部分参数或梯度，就没必要让每个 rank 都拿到完整同步梯度。

普通 DDP：

```text
AllReduce -> 每个 rank 都有完整 gradient
```

ZeRO / FSDP：

```text
ReduceScatter -> 每个 rank 只保留自己的 gradient shard
```

这能降低梯度显存，也能减少后续 optimizer state 更新的重复。

### AllReduce 与 ReduceScatter + AllGather

很多实现上，可以把 AllReduce 理解为：

```text
AllReduce = ReduceScatter + AllGather
```

也就是说：

1. 先把 reduce 后的结果切成 shards。
2. 再把所有 shards gather 回每个 rank。

如果系统只需要 shard，就可以停在 ReduceScatter，不再做 AllGather。

这就是 sharded training 节省显存和通信的关键之一。

## AllGather

AllGather 的语义是：

```text
每个 rank 有一片 shard
所有 rank 收集全部 shard
每个 rank 都得到完整结果
```

例子：

```text
Rank 0: shard A
Rank 1: shard B
Rank 2: shard C
Rank 3: shard D

AllGather 后，每个 rank 都有：
[A, B, C, D]
```

AllGather 常见于：

- FSDP forward 前 all-gather 参数。
- ZeRO-3 需要完整参数时。
- Tensor Parallel 需要恢复完整输出时。
- Sequence / Context Parallel 需要收集 K/V 或 sequence shard 时。
- checkpoint 或 eval 时临时聚合 sharded state。

### AllGather 的显存峰值

AllGather 不只消耗网络带宽，还会消耗显存。

假设每个 rank 持有参数 shard：

```text
local shard = N / P
```

AllGather 后临时得到：

```text
full tensor = N
```

所以 FSDP / ZeRO-3 的显存峰值经常和 all-gather 时机有关：

- all-gather 太早，完整参数驻留时间长。
- prefetch 太激进，多个模块完整参数重叠。
- reshard 太晚，释放不及时。
- activation checkpointing 可能导致 backward 重算时再次 all-gather。

## Broadcast

Broadcast 的语义是：

```text
root rank 有一个 tensor
把它发送给 group 内所有 rank
```

常见用途：

- 初始化模型参数。
- 分发随机种子或小配置。
- 分发 tokenizer / metadata 的 hash。
- 恢复 checkpoint 后确保某些状态一致。

Broadcast 和 AllGather 的区别是：

- Broadcast 只有一个 root 源。
- AllGather 每个 rank 都贡献一片数据。

## Reduce

Reduce 的语义是：

```text
每个 rank 有数据
做 reduce
只有 root rank 得到结果
```

它适合只需要一个 rank 拿结果的场景：

- 聚合统计指标。
- 聚合错误计数。
- 生成 checkpoint manifest。
- 只在 rank 0 打印日志。

如果所有 rank 都需要结果，就用 AllReduce。

## AllToAll

AllToAll 的语义是：

```text
每个 rank 都把不同分块发给其他 rank
每个 rank 都从所有 rank 收到给自己的分块
```

它不是“大家得到一样的数据”，而是“每个 rank 和每个 rank 互换一部分数据”。

MoE Expert Parallel 是典型场景：

```text
每个 rank 本地有一些 tokens
router 决定每个 token 去哪个 expert
expert 分布在不同 rank
AllToAll 把 token 发给目标 expert 所在 rank
expert compute 后
再 AllToAll 把结果发回原来的 rank / 位置
```

AllToAll 对系统很敏感：

- token 分布不均会造成某些 rank 收到很多数据。
- message size 可能不均匀。
- 小消息多时 latency 开销高。
- 跨节点 AllToAll 容易打满网络。
- 某些 rank 晚到会拖慢整个 group。
- dispatch 和 combine 两次通信都要看。

MoE 性能问题经常不是 expert GEMM 不够快，而是 AllToAll 和负载不均。

## Barrier

Barrier 的语义是：

```text
所有 rank 到达 barrier 后才能继续
```

它没有主要数据交换，但会强制同步。

Barrier 在调试时有用：

- 确认所有 rank 到达某个阶段。
- 缩小 hang 的范围。
- 在 benchmark 前后对齐阶段。

但训练代码里不要滥用 barrier。

原因是：

- 它会破坏自然 overlap。
- 它会把慢 rank 的影响显式放大。
- 它可能掩盖真正的异步错误位置。
- 它会让 profiler timeline 不代表真实训练。

## Point-to-Point Send / Recv

不是所有通信都是 collective。

Pipeline Parallel 经常使用点对点通信：

```text
stage 0 send activation to stage 1
stage 1 recv activation from stage 0
stage 1 backward 后 send activation gradient back to stage 0
```

P2P 的特点是：

- 只有发送方和接收方参与。
- 需要匹配 send / recv 顺序。
- 容易和 pipeline schedule 绑定。
- 可以和 micro-batch 流水线交错。

如果 send / recv 配对错，也会 hang。

## 不同并行策略对应哪些通信

把原语映射回训练策略：

| 并行策略 | 常见通信 | 数据对象 |
| --- | --- | --- |
| DDP / Data Parallel | AllReduce | gradients。 |
| ZeRO-1/2 | ReduceScatter / AllReduce | gradients / optimizer state。 |
| ZeRO-3 / FSDP | AllGather / ReduceScatter | parameters / gradients。 |
| Tensor Parallel | AllReduce / AllGather / ReduceScatter | layer outputs / input grads / vocab shards。 |
| Pipeline Parallel | Send / Recv | activations / activation gradients。 |
| Expert Parallel / MoE | AllToAll | routed tokens / expert outputs。 |
| Sequence Parallel | AllGather / ReduceScatter | sequence / hidden shards。 |
| Context Parallel | P2P / ring / AllGather / ReduceScatter | K/V、context shards、attention partial results。 |
| Checkpoint resharding | AllGather / scatter / object store IO | sharded states。 |

训练系统设计时，应该先画出：

```text
哪些数据对象被切分？
哪些 rank 持有哪些 shard？
在哪一步需要完整数据？
在哪一步可以只保留 shard？
```

然后再选择通信原语。

## 通信量不是唯一瓶颈

很多人看通信只看 bytes，但这不够。

一个 collective 的端到端成本包括：

- 消息大小。
- 消息数量。
- 算法选择。
- 拓扑路径。
- rank placement。
- 慢 rank 等待。
- 网络拥塞。
- GPU kernel 调度。
- CPU launch 和 runtime 调度。
- stream dependency。
- 是否暴露在关键路径上。

两个通信量一样的方案，性能可能完全不同。

例如：

```text
一次 1GB AllReduce
```

和：

```text
1024 次 1MB AllReduce
```

总 bytes 都是 1GB，但后者更容易被 latency 和 launch overhead 拖慢。

## Ring、Tree 与 Hierarchical 直觉

底层通信库会根据消息大小、拓扑和配置选择算法。常见直觉包括：

- ring 适合大消息，带宽利用率好。
- tree 适合某些小消息或跨节点 reduction。
- hierarchical 会先做节点内通信，再做节点间通信。
- NVLink / NVSwitch、PCIe、IB / RoCE 的拓扑会影响路径。
- 多 rail 需要 rank mapping 和网卡绑定配合。

不用一开始就记住所有算法细节，但要知道：

```text
collective 不是抽象函数调用那么简单
底层算法和拓扑会改变真实性能
```

同一段训练脚本，在不同节点拓扑、NCCL 版本、驱动、网络配置下，通信表现可能不同。

## Async Collective 与 Overlap

很多框架支持异步 collective。例如 PyTorch distributed 中常见：

```python
work = dist.all_reduce(tensor, async_op=True)
...
work.wait()
```

这表示通信可以异步启动，但不代表通信一定被完全隐藏。

要区分：

- collective 是否异步发起。
- GPU 上通信 kernel 是否已经开始。
- 后续计算是否真的不依赖通信结果。
- 使用通信结果前是否正确 wait。
- 通信是否和计算在不同 stream 上有可重叠空间。
- 是否存在隐式同步。

常见误区：

```text
async_op=True
```

不等于：

```text
communication is free
```

真正要看 profiler timeline：

- NCCL kernel 是否和 compute kernel 重叠。
- backward 结束后是否仍有通信暴露。
- `wait()` 是否卡住关键路径。
- 某个 rank 是否晚到 collective。

## Collective Ordering 与 Hang

分布式 hang 很多来自 collective 顺序不一致。

例子：

```text
Rank 0:
  AllReduce(A)
  AllGather(B)

Rank 1:
  AllGather(B)
  AllReduce(A)
```

两个 rank 都调用了相同的 collective，但顺序不同，仍然会出问题。

动态控制流也危险：

```python
if local_loss_is_nan:
    dist.all_reduce(flag)
```

如果只有某些 rank 进入这个分支，其他 rank 没有进入，程序可能 hang。

更安全的写法是让所有 rank 都参与：

```python
flag = torch.tensor([local_loss_is_nan], device=device)
dist.all_reduce(flag, op=dist.ReduceOp.MAX)
```

然后所有 rank 根据全局 flag 做同样决策。

## Shape、Dtype 与 Device 一致性

Collective 对 tensor 元信息很敏感。

建议每个关键 collective 前记录或断言：

- tensor shape。
- dtype。
- device。
- numel。
- process group size。
- local rank / global rank。
- collective name。

尤其在动态 batch、sequence packing、MoE token routing、variable length sequence、checkpoint resharding 中，shape 不一致很常见。

MoE AllToAll 还要特别处理变长 token 数：

- 先 exchange counts。
- 根据 counts 分配接收 buffer。
- 再交换真正 token 数据。
- combine 时按原始顺序放回。

如果 counts 错，后续 tensor shape 可能全乱。

## 通信 Buffer 与显存

通信不只消耗网络，也会消耗显存。

常见通信相关显存包括：

- DDP gradient bucket。
- FSDP all-gather full parameter buffer。
- ReduceScatter output shard。
- AllGather output buffer。
- MoE dispatch / combine buffer。
- NCCL 内部 buffer。
- 临时 pack / unpack buffer。

所以看到 OOM 时，不能只看 model parameters 和 activations。

例如 FSDP 中：

```text
当前模块参数 all-gather
下一个模块参数 prefetch
activation checkpointing 重算
gradient reduce-scatter buffer
```

这些可能在某个时间点叠加，形成峰值。

## Benchmark 应该怎么做

Collective benchmark 需要分层。

### Microbenchmark

测单个 collective：

- AllReduce。
- ReduceScatter。
- AllGather。
- AllToAll。
- Broadcast。

记录：

- message size。
- dtype。
- rank 数。
- 节点数。
- 每节点 GPU 数。
- 拓扑。
- backend。
- 算法配置。
- warmup。
- steady-state bandwidth。
- p50 / p95 / p99 latency。

Microbenchmark 能回答：

```text
这套硬件和通信栈，单个 collective 大概能跑到什么水平？
```

### Component Benchmark

测训练组件：

- DDP bucket AllReduce。
- FSDP all-gather + forward + reduce-scatter。
- TP layer communication。
- MoE dispatch + expert GEMM + combine。
- PP send/recv pipeline。

它能回答：

```text
通信和计算放在一起时，是否能 overlap？
```

### End-to-End Benchmark

测完整训练 step：

- step time。
- MFU。
- tokens/s。
- communication exposed time。
- scaling efficiency。
- peak memory。

它能回答：

```text
通信是否真正影响端到端训练效率？
```

## Profiler 中怎么看 Collective

在 profiler timeline 里，重点看：

- NCCL / RCCL kernel 出现在哪里。
- 通信是否和 compute 重叠。
- backward 结束后是否还有通信。
- 不同 rank 是否同时进入 collective。
- 某些 rank 是否在 collective 前有长 idle。
- collective size 是否太小太碎。
- AllToAll 是否有明显长尾。
- CPU thread 是否延迟 launch。

一个有用的判断：

```text
communication time != exposed communication time
```

如果通信完全被计算覆盖，它对 step time 影响可能小。

如果通信暴露在关键路径上，即使总通信量不大，也可能拖慢 step。

## 常见故障

### Collective Hang

可能原因：

- 某些 rank 没进入 collective。
- collective 调用顺序不同。
- process group 不一致。
- tensor shape / dtype 不一致。
- 某个 rank 先前发生错误但其他 rank 仍在等。
- P2P send / recv 不匹配。

排查：

- 在 collective 前后打印 rank-aware log。
- 打印 group size、tensor shape、dtype。
- 缩小到最小复现。
- 设置合理 timeout。
- 打开 NCCL / framework debug log。
- 检查异常是否只在某个 rank 出现。

### AllReduce 很慢

可能原因：

- bucket 太小或太多。
- 跨节点网络慢。
- rank mapping 不合理。
- 某个 rank 晚到 collective。
- NCCL 算法或拓扑选择不适合。
- 和数据加载、checkpoint、其他 job 抢资源。

排查：

- 看 profiler 多 rank timeline。
- 对比 nccl-tests 或框架 microbenchmark。
- 检查 GPU/NIC affinity。
- 检查链路带宽和错误计数。
- 调整 bucket size 和 overlap 策略。

### AllGather 造成 OOM

可能原因：

- 临时 full tensor 太大。
- prefetch 过多。
- all-gather 和 activation peak 叠加。
- checkpointing 重算阶段再次 all-gather。
- reshard 太晚。

排查：

- 记录 all-gather 前后 peak memory。
- 检查 FSDP wrap 粒度。
- 调整 prefetch / reshard 策略。
- 降低 micro-batch 或使用 activation checkpointing。

### AllToAll 长尾

可能原因：

- token routing 不均。
- expert load imbalance。
- 每 rank send/recv counts 差异大。
- 小包太多。
- 跨节点 expert 分布不合理。
- 网络拥塞。

排查：

- 记录每 expert token count。
- 记录 AllToAll send/recv bytes per rank。
- 看 router load balance loss。
- 调整 EP size、expert placement、capacity factor。
- 对比同机 EP 和跨节点 EP。

## 常见优化方向

### 合并小消息

小 collective 多时，latency 占比高。可以考虑：

- DDP gradient bucket。
- 合并小参数。
- 减少 tiny all-reduce。
- 避免每个 scalar metric 都单独 all-reduce。

### 选择合适 Sharding

如果每个 rank 不需要完整结果，就不要 AllReduce 后保留完整 tensor。

可考虑：

- ReduceScatter 代替 AllReduce。
- AllGather 只在真正需要完整参数时发生。
- 让 layout 在多个层之间保持 sharded，减少来回 gather。

### 改善 Rank Mapping

让通信 group 匹配硬件拓扑：

- TP group 优先放在高速互连内。
- EP group 如果跨节点，要评估 AllToAll 成本。
- DP group 跨节点时要关注 AllReduce 带宽。
- GPU 到 NIC affinity 要正确。
- 多 rail 配置要均衡。

### 增加 Overlap

通信未必都要暴露：

- DDP bucket overlap。
- FSDP all-gather prefetch。
- reduce-scatter 和 backward overlap。
- PP send/recv 与 compute overlap。
- MoE dispatch 和 expert compute 的细粒度 overlap。

但 overlap 要靠 profiler 证明，不能只看配置开关。

### 减少跨节点通信

跨节点通常比节点内慢。可以：

- 把高频 TP 放节点内。
- 把 PP stage 映射到减少跨节点 activation 传输的位置。
- 控制 EP group 跨节点范围。
- 使用 hierarchical collective。
- 对 checkpoint / eval 等低频通信做异步化。

## 实践检查清单

设计或排查训练通信时，至少回答：

1. 当前有哪些 process groups？
2. 每个 group 包含哪些 rank？
3. 每种并行策略对应哪些 collective？
4. 每个 collective 的 tensor shape、dtype、bytes 是多少？
5. collective 是节点内、跨节点，还是混合？
6. 哪些通信在 critical path 上暴露？
7. 是否有很多小消息？
8. 是否有 rank 晚到 collective？
9. 是否存在 AllGather full tensor 导致显存峰值？
10. AllToAll 是否有 send/recv 不均？
11. benchmark 是否区分了 micro、component 和 end-to-end？
12. profiler 是否覆盖多 rank，而不是只看单 rank？

## 与本章其他主题的关系

建议把本篇和这些内容连起来读：

- [分布式训练启动与运行时：torchrun、Rank、Process Group 与 NCCL](distributed-training-runtime.md)：理解 rank、process group 和 backend 如何初始化。
- [Data Parallel 与梯度同步](data-parallel-gradient-sync.md)：理解 DDP 如何把 gradients 放进 buckets 并 AllReduce。
- [ZeRO 与 FSDP](zero-fsdp.md)：理解 sharded training 为什么依赖 ReduceScatter 和 AllGather。
- [Tensor Parallel](tensor-parallel.md)：理解层内切分为什么引入 AllReduce / AllGather / ReduceScatter。
- [Expert Parallel 与 MoE 训练](expert-parallel-moe-training.md)：理解 AllToAll 和 token dispatch。
- [通信与计算重叠](communication-computation-overlap.md)：理解 collective 如何被放进 backward 或 prefetch 中重叠。
- [FLUX 通信重叠与 Kernel Fusion](flux-kernel-fusion.md)：理解普通 collective overlap 不够时的更细粒度方向。
- [训练性能剖析与 Benchmark](training-benchmark-profiling.md)：理解如何用 profiler 验证通信瓶颈。

## 参考资料

- [NVIDIA NCCL Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)
- [PyTorch Distributed Communication Package](https://docs.pytorch.org/docs/stable/distributed.html)
- [NVIDIA NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)
