---
title: EP Size 与大 EP / 小 EP
domain: inference-systems
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-06-10
---

# EP Size 与大 EP / 小 EP

EP 是 Expert Parallelism 的缩写，通常翻译为专家并行。它主要用于 MoE 模型：把不同专家放到不同 GPU 上，让一个推理实例不必在每张 GPU 上保存全部专家权重。

一句话理解：

> EP size 决定一个 MoE 副本里的专家被切到多少张 GPU 上；大 EP 更省单卡专家显存，但通信范围更大，小 EP 通信更短、更适合低延迟，但每卡要放更多专家。

“大 EP / 小 EP”不是一个严格标准名词，更像工程团队在讨论部署取舍时的口语。它不是说某个固定数字一定大或小，而是相对于模型专家数、GPU 显存、节点拓扑、并发规模和延迟目标而言。

## EP Size 是什么

EP size 可以理解为一个 MoE expert parallel group 里的 GPU 数量。

假设一个 MoE layer 有 `E` 个 routed experts，EP size 是 `P_ep`，不考虑专家复制时，每张 GPU 大约放：

```text
experts per GPU = E / P_ep
```

例如：

| routed experts | EP size | 每张 GPU 放多少专家 |
| --- | --- | --- |
| 64 | 1 | 64 |
| 64 | 4 | 16 |
| 64 | 8 | 8 |
| 64 | 16 | 4 |
| 64 | 64 | 1 |

EP size 越大，每张 GPU 存的专家越少。这样能降低单卡专家权重显存，但也意味着一次 MoE layer 的 token dispatch/combine 可能跨越更多 GPU。

## 大 EP 和小 EP 的直觉

可以先用一个简单例子理解。

假设有 64 个专家和 8 张 GPU。

### 小 EP

如果 EP size = 2，那么一个 EP group 里只有 2 张 GPU。每张 GPU 要放 32 个专家。

特点是：

- 通信只在 2 张 GPU 之间发生。
- all-to-all group 小。
- 更容易限制在单机或高速互联内。
- 每张 GPU 专家权重显存压力大。
- 同样 GPU 总数下，可以有更多 DP replica。

### 大 EP

如果 EP size = 8，那么一个 EP group 里有 8 张 GPU。每张 GPU 只放 8 个专家。

特点是：

- 每张 GPU 专家权重更少。
- 可以容纳更大专家参数量。
- all-to-all group 变大。
- token dispatch/combine 跨更多 GPU。
- 如果跨节点，网络延迟和带宽压力会明显上升。
- 同样 GPU 总数下，DP replica 数会减少。

所以，大 EP 和小 EP 的核心矛盾是：

```text
专家权重显存  <->  MoE 通信范围
```

大 EP 用更大的通信范围换更低的单卡专家显存；小 EP 用更高的单卡专家显存换更短的通信路径。

## EP 改变了什么

EP 主要改变四件事。

### 1. 专家权重放置

EP size 越大，每张 GPU 放的专家越少。

这对超大 MoE 很重要。MoE 的总参数可能很大，虽然每个 token 只激活 top-k 个专家，但部署时专家权重必须放在某些 GPU 上。

如果所有专家都放不进单卡，就必须使用 EP 或专家 offload / 专家复制等策略。

### 2. Token Dispatch 范围

MoE layer 里，router 会为每个 token 选择专家。如果被选中的专家在别的 GPU 上，token hidden state 就要发送过去。

EP size 越大，token 可能被发送到的 GPU 越多。

这会影响：

- dispatch latency。
- combine latency。
- all-to-all 消息数量。
- 单次消息大小。
- 网络拥塞。
- 尾延迟。

### 3. 每个专家的子 batch 形态

MoE 的专家计算不是一个大 batch 直接进同一个 MLP，而是 token 被 router 分散到不同专家。

某个专家实际拿到多少 token，取决于：

- 总 token 数。
- top-k routing。
- 专家数量。
- router 分布。
- batch / continuous batching 规模。
- Prefill 还是 Decode。

EP size 越大，每张 GPU 上专家更少，但 all-to-all 更宽。对于 Decode 这种每轮 token 很少的场景，专家子 batch 可能很碎，通信 latency 更容易成为瓶颈。

### 4. DP Replica 数

在固定 GPU 总数下，EP size 越大，一个模型副本占用的 GPU 越多，能部署的 DP replica 通常越少。

如果总 GPU 数是 `N`，暂时忽略 TP/PP，那么：

```text
DP replica 数约等于 N / EP size
```

这意味着：

- 大 EP：单副本能容纳更大专家，但副本数少。
- 小 EP：副本数多，服务并发和隔离可能更好，但每副本专家显存更高。

实际部署还要乘上 TP、PP、CP 等并行维度，但直觉一样：EP size 会影响一个 replica 占多少 GPU。

## 大 EP 的优势

大 EP 适合这些情况：

- 专家权重太大，小 EP 放不下。
- 总专家数很多，每卡不能保存太多专家。
- Prefill token 多，MoE 通信可以被较大 token batch 摊薄。
- 更关注吞吐或模型容量，而不是极低 TPOT。
- GPU 间互联足够强，例如单机 NVLink / NVSwitch 内。
- 有高效 all-to-all、DeepEP、通信计算重叠或 grouped GEMM 支持。

大 EP 的核心价值是“容量”。它让一个 MoE 模型的专家参数分摊到更多 GPU 上。

## 大 EP 的风险

大 EP 的风险主要来自通信。

常见问题包括：

- all-to-all group 太大，通信延迟高。
- 跨节点 EP 让 IB/RDMA 成为瓶颈。
- Decode 阶段 token 少，通信启动开销无法摊薄。
- 最慢 EP rank 拖住整层。
- 专家热点导致某个 rank 成为 straggler。
- DP replica 数减少，整体服务并发下降。
- 故障域变大，一个 rank 异常影响整个 EP group。

大 EP 不是天然更快。它通常是为了放下模型或扩大专家容量，不是为了降低单请求延迟。

## 小 EP 的优势

小 EP 适合这些情况：

- 每张 GPU 有足够显存放更多专家。
- 希望 all-to-all 限制在单机或少数 GPU 内。
- 目标是降低 Decode TPOT 和 p99 latency。
- 请求 batch 较小，专家子 batch 很碎。
- 希望部署更多 DP replica 提高并发。
- 希望缩小故障影响范围。

小 EP 的核心价值是“低通信”和“更多副本”。对在线推理尤其是 Decode 阶段，它经常比大 EP 更容易获得稳定延迟。

## 小 EP 的风险

小 EP 的风险主要来自显存和局部过载。

常见问题包括：

- 每张 GPU 要放更多专家，显存压力高。
- 专家权重带宽压力更大。
- 单 rank 上专家计算更多，可能拖慢 step。
- 如果热门专家集中在同一 GPU，热点更明显。
- 模型太大时根本放不下。

所以小 EP 也不是越小越好。它需要专家权重、非专家层、KV Cache 和 runtime buffer 都能放进显存。

## Prefill 和 Decode 下的差异

MoE 推理里，Prefill 和 Decode 对 EP 的敏感点不同。

### Prefill

Prefill 一次处理用户输入 prompt，token 数通常比较多。

这意味着：

- 每个 MoE layer 的 routed token 多。
- 专家子 batch 更容易变大。
- all-to-all 消息更大但更容易摊薄启动开销。
- 专家 GEMM 更容易形成较好形态。

因此 Prefill 对大 EP 的容忍度通常比 Decode 更高。长 prompt 下，大 EP 的专家容量和并行计算可能更有价值。

### Decode

Decode 是逐 token 生成。每轮每个请求通常只新增一个 token。

即使有 continuous batching，一轮中的 routed token 仍可能不多。

这意味着：

- 专家子 batch 更小。
- all-to-all 消息更碎。
- 通信 latency 更难摊薄。
- 一个热门专家或慢 rank 更容易影响 TPOT。

因此 Decode 阶段经常更偏好小 EP、更多 replica、较短通信路径和更强专家负载控制。

## EP 与 TP / DP / PP 的组合

MoE 模型通常不会只用 EP。

常见组合包括：

| 并行方式 | 作用 |
| --- | --- |
| TP | 切 attention 或 dense MLP 的大矩阵 |
| EP | 切 MoE experts |
| DP | 复制完整推理实例，提高并发 |
| PP | 按层切分模型 |
| CP | 按序列或上下文切分长上下文计算 |

一个简化部署可能是：

```text
Attention 层：TP
MoE FFN 层：EP
多个模型副本：DP
```

EP size 的选择会影响其他并行维度：

- EP 变大，一个 replica 占用 GPU 更多，DP replica 变少。
- TP 变大，dense 层通信变重，EP 通信之外又多一类通信。
- 如果 TP 和 EP 都跨节点，网络压力会叠加。
- PP 会引入 stage 间通信和 pipeline bubble。

因此不能孤立选择 EP size。要把它放在完整并行配置里看。

## 跨节点 EP 为什么敏感

大 EP 最危险的情况通常是跨节点。

单机内 GPU 可能有 NVLink / NVSwitch，延迟和带宽较好。跨节点则依赖 IB/RDMA 或其他网络，延迟更高，拥塞更复杂。

MoE EP 的 all-to-all 通信每层都可能发生。如果每个 MoE layer 都跨节点 dispatch/combine，那么网络会在每一层反复进入关键路径。

所以经验上常见优先级是：

1. 能把 EP group 放在单机内，就先放在单机内。
2. 如果必须跨节点，优先确保网络拓扑、带宽、通信库和 overlap 能支撑。
3. 如果 Decode p99 很差，优先检查是否跨节点 EP 过大。

跨节点 EP 不是不能做，但必须用真实 workload benchmark 验证。

## 大 EP / 小 EP 的选择方法

可以按下面顺序选择。

### 1. 先看能不能放下

先计算单 GPU 显存是否能容纳：

- 非专家层权重。
- 本 GPU 上的专家权重。
- KV Cache。
- runtime buffer。
- 通信 buffer。

如果小 EP 放不下，就必须增大 EP size 或使用量化、offload、专家复制/分层放置等策略。

### 2. 尽量限制通信范围

在能放下的前提下，EP size 不宜无脑增大。

优先让 EP group 落在高速互联范围内，例如单机 8 卡或一个 NVSwitch island。

### 3. 分开看 Prefill 和 Decode

如果 Prefill 吞吐不错但 Decode TPOT 差，要重点怀疑：

- EP group 太大。
- Decode token 太少。
- all-to-all latency 高。
- 专家子 batch 太碎。
- 热门专家造成 straggler。

如果 Prefill 本身很慢，则还要看：

- dispatch/combine 带宽。
- 专家 GEMM 效率。
- expert load skew。
- context length。

### 4. 看 DP 副本数

如果 EP size 过大，DP replica 数减少，服务整体并发能力可能下降。

在线服务常常需要在两种方案之间比较：

```text
方案 A：大 EP，少 replica，单副本模型容量大
方案 B：小 EP，多 replica，通信短，并发更好
```

最终要看 SLO 下的 goodput，而不是只看单副本 tokens/s。

### 5. 用真实 expert load 判断

MoE 路由不一定均匀。理论上每个专家平均收到 token，不代表真实 workload 下均匀。

必须观察：

- 每个专家 token 数。
- 每个专家 latency。
- 每个 rank 的 expert load。
- 热门专家是否稳定存在。
- 不同业务流量是否激活不同专家。

没有 expert-level metrics，就无法严肃判断 EP size 是否合理。

## 一个例子

假设有 128 个 routed experts，集群里每台机器 8 张 GPU。

### 方案 A：EP size = 8

每张 GPU 放 16 个专家。EP group 正好在单机内。

特点：

- all-to-all 只在单机内。
- 每张 GPU 专家显存较高。
- DP replica 可以按节点扩展。
- Decode 延迟相对容易控制。

### 方案 B：EP size = 32

每张 GPU 放 4 个专家。一个 EP group 跨 4 台机器。

特点：

- 单卡专家显存显著下降。
- 可以容纳更大专家参数。
- 每个 MoE layer 可能跨节点 all-to-all。
- Decode TPOT 和 p99 更容易被网络拖慢。
- 同样 GPU 总数下 DP replica 更少。

这两个方案没有绝对对错。

如果模型放不下，方案 B 可能是必要的。如果模型在方案 A 下能放下，并且目标是在线低延迟，方案 A 可能更稳。

## 和专家复制的关系

EP 只是把专家分散到不同 GPU。它不自动解决热门专家问题。

如果某些专家长期热门，可以考虑专家复制：

- 热门专家在多个 GPU 上放副本。
- router 或 runtime 在副本之间分配 token。
- 减少单个专家 rank 的热点。

但专家复制会增加显存占用。它和 EP size 是一组取舍：

- 大 EP 降低每卡专家数，但通信更宽。
- 专家复制增加专家副本，但可缓解热点。
- 小 EP 通信短，但每卡专家更多。

实际部署可能组合使用：先选合适 EP size，再对热门专家做局部复制。

## 和 DeepEP / 通信优化的关系

DeepEP、all-to-all kernel、通信计算重叠、token grouping、grouped GEMM 等技术，可以改善 EP 的通信和计算效率。

但它们不能消除 EP size 的基本取舍。

如果 EP group 很大且跨节点，通信仍然会进入关键路径。通信优化可以降低成本，但不能把宽通信变成零成本。

所以选 EP size 时，不应只问“有没有高性能 EP 通信库”，还要问：

- 通信范围多大。
- 每轮 token 有多少。
- 消息是否足够大。
- 是否跨节点。
- Decode p99 是否能接受。
- 失败域是否过大。

## 常见误区

### 1. 大 EP 一定更快

不一定。大 EP 更省单卡专家显存，但通信更宽。Decode 场景里，大 EP 可能让 TPOT 和 p99 更差。

### 2. 小 EP 一定更好

也不一定。小 EP 每卡专家更多，可能放不下模型，也可能让单 rank 专家计算过重。

### 3. EP size 只影响显存

不对。EP size 同时影响专家权重显存、all-to-all 通信范围、DP replica 数、专家 batch 形态和故障域。

### 4. 平均专家负载均衡就够了

不够。要看 p95/p99、热门专家、rank-level latency 和不同请求类型。平均值容易掩盖 straggler。

### 5. Prefill 的最优 EP 一定适合 Decode

不一定。Prefill token 多，Decode token 少。一个配置可能 Prefill 吞吐很好，但 Decode TPOT 很差。

## 应该观察哪些指标

评估 EP size 时，建议观察：

| 指标 | 说明 |
| --- | --- |
| EP size | 一个 expert parallel group 的 GPU 数 |
| experts per GPU | 每张 GPU 放多少专家 |
| DP replicas | 固定 GPU 总数下能部署多少副本 |
| dispatch latency | token 发往专家的耗时 |
| combine latency | 专家结果合并耗时 |
| all-to-all bytes | MoE 通信量 |
| all-to-all p95/p99 | 通信长尾 |
| cross-node traffic | 跨节点通信量和占比 |
| expert load | 每个专家收到多少 token |
| expert load skew | 热门专家与冷门专家差异 |
| expert batch size | 每个专家实际 token batch |
| rank latency | 每个 EP rank 的耗时 |
| GPU memory | 专家权重、KV Cache 和 buffer 占用 |
| TTFT / TPOT | Prefill 和 Decode 对用户体验的影响 |
| goodput | 满足 SLO 的有效吞吐 |

这些指标要按 Prefill/Decode、batch size、input/output length 和请求类型分组看。

## 小结

EP size 是 MoE 推理部署里的关键参数。它决定专家权重如何分布，也决定 token dispatch/combine 的通信范围。

核心结论：

- 大 EP：每卡专家少，显存压力低，但 all-to-all 范围大，跨节点后尾延迟风险高。
- 小 EP：通信短、DP replica 多、Decode 更容易稳，但每卡专家显存压力高。
- EP size 不能孤立选择，必须和 TP、DP、PP、KV Cache、Prefill/Decode、网络拓扑一起看。
- Prefill 和 Decode 可能偏好不同 EP 配置。
- 真实 expert load 和 p95/p99 latency 比平均吞吐更重要。

对关注高效推理的人来说，大 EP / 小 EP 的本质不是名词，而是一个系统取舍：用多少 GPU 共同承载专家，以及为此愿意付出多少通信、延迟和副本数成本。

## 参考资料

- [DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale](https://arxiv.org/abs/2201.05596)
- [A Hybrid Tensor-Expert-Data Parallelism Approach to Optimize Mixture-of-Experts Training](https://arxiv.org/abs/2303.06318)
- [Speculative MoE: Communication Efficient Parallel MoE Inference with Speculative Token and Expert Pre-scheduling](https://arxiv.org/abs/2503.04398)
- [MoE Parallel Folding: Heterogeneous Parallelism Mappings for Efficient Large-Scale MoE Model Training with Megatron Core](https://arxiv.org/abs/2504.14960)
