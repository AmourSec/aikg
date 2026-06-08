---
title: KV Cache
domain: inference-systems
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-06-08
---

# KV Cache

KV Cache 是 LLM 推理系统里最重要的缓存之一。它保存历史 token 在 Attention 里会反复用到的 key 和 value，让 Decode 阶段不用每生成一个 token 都重新计算全部上下文。

一句话理解：

> KV Cache 用显存换计算时间，让模型在逐 token 生成时可以复用已经读过的上下文。

它的价值很大，但代价也很明显：KV Cache 会随着并发数、输入长度和输出长度增长，占用大量显存，并直接影响最大并发、长上下文能力、调度策略和服务稳定性。

## KV Cache 在哪里出现

KV Cache 出现在 Transformer 的 Attention 里。模型处理 token 时，会把每个 token 的表示变成 query、key、value。当前 token 用 query 去和历史 token 的 key 做匹配，再根据匹配结果读取 value。

简化流程如下：

```mermaid
flowchart TB
    A["输入 prompt"] --> B["Prefill"]
    B --> C["计算每层 Attention 的 K/V"]
    C --> D["写入 KV Cache"]
    D --> E["Decode step"]
    E --> F["新 token 产生新的 K/V"]
    F --> G["追加到 KV Cache"]
    G --> H["下一步 Decode 读取历史 KV"]
    H --> E
```

Prefill 阶段会一次性为输入 prompt 写入 KV Cache。Decode 阶段每生成一个新 token，就会把这个新 token 对应的 key/value 追加到 KV Cache 里。

## 如果没有 KV Cache 会怎样

假设用户输入了 1000 个 token，模型要继续生成 200 个 token。

如果没有 KV Cache：

1. 生成第 1 个输出 token 时，模型处理 1000 个输入 token。
2. 生成第 2 个输出 token 时，模型重新处理 1000 个输入 token 和第 1 个输出 token。
3. 生成第 200 个输出 token 时，模型重新处理前面 1199 个 token。

这样会重复计算大量历史上下文。

有了 KV Cache 后：

1. Prefill 先处理 1000 个输入 token，并保存它们的 KV。
2. Decode 第 1 步只处理新生成位置，同时读取已有 1000 个 token 的 KV。
3. Decode 第 2 步继续读取历史 KV，并追加第 1 个输出 token 的 KV。
4. 后续每一步都只追加新 KV，不重新计算全部历史 token。

所以 KV Cache 的核心收益是减少重复计算，让 Decode 可以持续生成。

## K、V 到底缓存了什么

在每一层 Transformer 里，Attention 会为 token 计算三类向量：

- **Q（Query）**：当前 token 用来“提问”的向量。
- **K（Key）**：历史 token 用来“被匹配”的向量。
- **V（Value）**：历史 token 被读取出来的信息。

Decode 时，当前 token 的 query 会去看历史 token 的 key，然后根据匹配权重读取 value。历史 token 的 key/value 每一步都会用到，所以适合缓存。

需要注意：

- KV Cache 通常按层保存。
- 每一层都有自己的 K 和 V。
- 每个请求都有自己的上下文和 KV Cache。
- 输出越长，KV Cache 会继续增长。

这也是为什么 KV Cache 不只是一个小缓存，而是推理显存管理的核心对象。

## KV Cache 为什么占显存

KV Cache 保存的是模型中间状态，不是原始文本。它的大小和几个因素相关：

| 因素 | 影响 |
| --- | --- |
| batch size / 并发请求数 | 同时服务的请求越多，总 KV Cache 越大 |
| sequence length | 输入越长、输出越长，每个请求缓存越大 |
| 模型层数 | 每层都要保存 K 和 V，层数越多越大 |
| hidden size / attention heads | 每个 token 的 K/V 向量越大，缓存越大 |
| precision | FP16、BF16、FP8、INT8 等精度会影响每个元素占用 |

一个直观估算是：

```text
KV Cache 大小
  约等于 batch 中的总 token 数
  × 模型层数
  × 每层 K/V 的向量大小
  × 2（K 和 V）
  × 每个元素的字节数
```

这个估算不需要死记，重要的是理解增长方向：**并发越高、上下文越长、模型越大，KV Cache 越容易成为显存瓶颈。**

## KV Cache 和 Prefill / Decode 的关系

Prefill 和 Decode 都会使用 KV Cache，但方式不同。

| 阶段 | KV Cache 行为 | 系统影响 |
| --- | --- | --- |
| Prefill | 为输入 prompt 一次性生成并写入 KV Cache | 长 prompt 会带来大规模写入和显存占用 |
| Decode | 每生成一个 token，读取历史 KV，并追加新 token 的 KV | 输出越长，读取和追加越多 |

Prefill 更像“初始化缓存”，Decode 更像“不断读取和扩展缓存”。

如果输入很长，Prefill 会一次性写入大量 KV Cache。如果输出很长，Decode 会不断追加 KV Cache。如果并发很高，很多请求的 KV Cache 会同时驻留在显存里。

## KV Cache 如何影响最大并发

在线服务里，最大并发通常不只由模型计算速度决定，还受 KV Cache 显存容量限制。

假设 GPU 显存里已经放了：

- 模型权重。
- runtime workspace。
- 当前 batch 的临时计算空间。

剩下的显存才能给 KV Cache 使用。每多接一个请求，系统就要为它的上下文预留或分配 KV Cache。请求越多、上下文越长，剩余显存越少。

如果 KV Cache 占满显存，系统可能出现：

- 新请求无法进入。
- 长上下文请求被拒绝。
- batch size 被迫降低。
- Decode 并发下降。
- 触发 OOM 或频繁驱逐。

因此，KV Cache 管理直接决定“同一块 GPU 能同时服务多少请求”。

## KV Cache 如何影响长上下文

长上下文推理不只是 Attention 计算更贵，也会让 KV Cache 更大。

例如：

- 4K 上下文需要缓存几千个 token 的 KV。
- 32K 上下文需要缓存几万个 token 的 KV。
- 如果还有多轮对话、RAG 文档和工具调用记录，KV Cache 会继续增长。

长上下文场景常见问题包括：

- 首 token 慢，因为 Prefill 要处理更多输入。
- 显存占用高，因为输入 token 的 KV 都要保存。
- Decode 每步读取的历史 KV 更多。
- 少数超长请求会挤占大量显存，拖累其他请求。

所以长上下文服务通常需要更严格的 max context length、请求准入、prefix cache、PagedAttention、KV Cache offload 或 KV Cache quantization。

## KV Cache 生命周期

从系统角度看，一个请求的 KV Cache 有生命周期：

1. **创建**：请求进入 Prefill，开始为输入 prompt 生成 KV。
2. **增长**：Decode 每生成一个 token，就追加新 token 的 KV。
3. **读取**：每个 Decode step 都会读取历史 KV。
4. **复用**：如果有 prefix cache，相同前缀的 KV 可能被多个请求复用。
5. **释放**：请求结束、取消、超时或失败后，KV Cache 必须释放。

如果释放不及时，已经结束的请求还占着显存，就会造成隐性容量下降。系统看起来没有很多活跃请求，但显存却很紧张。

## KV Cache 分配为什么困难

KV Cache 管理难，不只是因为它大，还因为请求形态很不规则。

在线请求有几个特点：

- 到达时间不同。
- 输入长度不同。
- 输出长度无法提前准确知道。
- 有些请求会很快结束，有些请求会生成很久。
- 有些请求会被用户取消或超时中断。

如果系统给每个请求预留一大块连续显存，短请求可能浪费很多空间；如果按需增长，又可能造成显存碎片。连续空间不够时，即使总剩余显存看起来还不少，也可能无法放下新的 KV Cache。

这就是后面要讲 PagedAttention 的原因：它把 KV Cache 分成块，减少连续分配带来的浪费和碎片问题。

## KV Cache 和 Batching 的关系

Batching 会让多个请求同时执行，也会让多个请求的 KV Cache 同时驻留。

Batching 的收益是 GPU 吞吐更高，但代价是显存压力更高：

- batch 里请求越多，总 KV Cache 越大。
- batch 里长上下文越多，KV Cache 越大。
- Decode 并发越高，活跃序列越多。
- 输出越长，每个请求的 KV Cache 越长。

因此，调 batch size 时不能只看 tokens/s，也要看 KV Cache usage、OOM、拒绝率、p95/p99 和 active sequences。

一个 batch 策略如果让吞吐提高 20%，但显存长期接近上限、尾延迟恶化、超长请求频繁失败，它就不一定是好策略。

## KV Cache 和调度的关系

调度器不只是决定谁先算，还要考虑谁占多少 KV Cache。

常见调度问题包括：

- 是否允许超长上下文请求进入。
- 是否优先服务快要完成的请求，释放 KV Cache。
- 是否限制单个租户占用过多 KV Cache。
- 是否把长 prompt 请求和短请求分开调度。
- 是否根据剩余生成长度预测未来 KV Cache 占用。

如果调度器只看请求数量，不看 KV Cache 占用，就可能让少数长上下文请求占满显存，导致大量短请求排队。

## 常见优化方向

KV Cache 相关优化可以分成几类：

| 方向 | 作用 | 代价或风险 |
| --- | --- | --- |
| PagedAttention / block 管理 | 减少显存碎片，提高容量利用率 | runtime 更复杂 |
| Prefix cache | 复用相同 prompt 前缀的 KV | 需要前缀匹配和缓存管理 |
| KV Cache quantization | 用更低精度保存 KV | 可能影响输出质量或稳定性 |
| KV Cache offload | 把部分 KV 放到 CPU 或其他存储 | 可能增加访问延迟 |
| Context length limit | 限制最大上下文 | 会限制应用能力 |
| Cache eviction | 驱逐低价值或过期缓存 | 可能导致重新计算 |
| Request admission | 拒绝超出容量的请求 | 需要明确策略和用户反馈 |

这些优化不是互斥的。实际系统通常会组合使用，例如 PagedAttention 管理显存块，prefix cache 复用公共前缀，准入控制限制超长请求。

## 该观察哪些指标

分析 KV Cache 问题时，建议观察：

| 指标 | 说明 |
| --- | --- |
| KV Cache used memory | 当前 KV Cache 占用多少显存 |
| KV Cache utilization | 分配出去的 KV 块是否被有效使用 |
| active sequences | 当前活跃生成序列数量 |
| total context tokens | 当前系统里所有请求的上下文 token 总量 |
| max context length | 当前请求是否接近上下文上限 |
| cache hit rate | prefix cache 或其他缓存命中率 |
| eviction count | KV 或 prefix cache 被驱逐次数 |
| OOM count | 是否因为 KV Cache 触发显存不足 |
| rejection rate | 是否因为容量限制拒绝请求 |
| p95 / p99 latency | KV 压力是否导致尾延迟变差 |

KV Cache 的问题经常不是单个指标能说明的。比如显存占用高可能是正常高并发，也可能是长请求过多、释放不及时、碎片严重或 cache 策略不合理。

## 一个最小例子

假设一个服务同时处理 3 个请求：

| 请求 | 输入长度 | 已生成输出 | 当前 KV Cache 长度 |
| --- | --- | --- | --- |
| A | 500 tokens | 20 tokens | 520 tokens |
| B | 4000 tokens | 10 tokens | 4010 tokens |
| C | 200 tokens | 200 tokens | 400 tokens |

虽然只有 3 个请求，但 B 的 KV Cache 远大于 A 和 C。调度器如果只看请求数，会低估 B 对显存和 Decode 的影响。

如果 B 是 RAG 请求，带了大量检索上下文，它可能显著拉高显存占用和尾延迟。系统可能需要对这类请求做单独队列、上下文压缩、prefix cache 或更严格的准入控制。

## 常见误区

- **误区一：KV Cache 只是一个小优化。**
  实际上它决定 Decode 是否能高效运行，也决定长上下文和高并发能否承载。

- **误区二：有了 KV Cache，Decode 就没有历史上下文成本。**
  KV Cache 避免重复计算历史 token，但每步 Decode 仍然要读取历史 KV。

- **误区三：显存够放模型权重就够了。**
  在线推理还需要大量 KV Cache。很多时候显存瓶颈来自 KV Cache，而不是权重。

- **误区四：只限制请求数就能控制容量。**
  不同请求上下文长度差异巨大，控制请求数不等于控制 KV Cache 占用。

- **误区五：请求结束后系统自然就没压力了。**
  如果 KV Cache 释放、复用或驱逐做不好，结束请求也可能留下显存管理问题。

读完这一节，应该能回答五个问题：

- KV Cache 保存的是什么，为什么 Decode 需要它。
- KV Cache 为什么会随着并发、输入长度和输出长度增长。
- KV Cache 如何影响最大并发、长上下文和显存容量。
- KV Cache 管理为什么会遇到碎片、释放和调度问题。
- PagedAttention、prefix cache、quantization、offload 分别在解决什么方向的问题。
