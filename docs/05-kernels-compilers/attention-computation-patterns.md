---
title: Attention 机制与计算模式
domain: kernels-compilers
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-06-12
---

# Attention 机制与计算模式

Attention 的核心问题是：每个 token 在更新自己的表示时，应该读取哪些其他 token 的信息，以及这个读取过程如何高效执行。

一句话理解：

> Dense Attention、Sparse Attention 和 FlashAttention 讨论的是不同层面的问题：Dense/Sparse 主要决定“看哪些 token”，FlashAttention 主要决定“同样的 attention 计算如何更省显存、更快执行”。

这篇文章放在 Kernel、算子与编译优化章节里，因为 Dense、Sparse、FlashAttention 不只是模型结构名词，也直接决定 attention kernel 的计算量、访存模式、显存占用和硬件效率。

## 先分清四个维度

Attention 相关名词很多，容易混在一起。学习和优化时，建议先分成四个维度：

| 维度 | 关注问题 | 例子 |
| --- | --- | --- |
| Attention pattern | 哪些 query 可以看哪些 key | dense、causal、sliding window、block sparse、global token |
| 数学是否精确 | 是否仍然计算原始 attention 结果 | exact attention、approximate attention |
| Kernel 实现 | 同样数学如何在 GPU 上执行 | naive SDPA、FlashAttention、Triton fused attention |
| 状态管理 | 历史 KV 怎么存、怎么复用、怎么释放 | KV Cache、PagedAttention、Prefix Cache |

这四个维度可以组合。

例如：

```text
dense causal attention + FlashAttention kernel + KV Cache + PagedAttention allocator
```

这句话的含义是：

- attention pattern 是 dense causal。
- kernel 用 FlashAttention 类 IO-aware exact attention。
- 推理时保存历史 K/V。
- KV Cache 用 page/block 管理。

如果不分维度，就会把 FlashAttention、Sparse Attention、PagedAttention 当成同类技术，后面做系统设计会很混乱。

## 先回到 Attention 在算什么

在 Transformer 里，每个 token 会产生三类向量：

- Query，表示“我想找什么信息”。
- Key，表示“我这里有什么特征可被匹配”。
- Value，表示“如果别人关注我，我能提供什么内容”。

简化流程如下：

```mermaid
flowchart TB
    A["Token 表示"] --> B["Linear 得到 Q / K / V"]
    B --> C["Q 与 K 做匹配"]
    C --> D["得到 attention score"]
    D --> E["Mask / Softmax"]
    E --> F["按权重读取 V"]
    F --> G["更新 token 表示"]
```

更具体一点，attention 会做三件事：

1. 用 Q 和 K 计算匹配分数。
2. 对分数做 mask 和 softmax，得到每个 token 应该关注谁。
3. 用这个权重加权求和 V，得到新的表示。

不同 attention 计算模式，主要差在第 1 步和第 2 步：哪些 Q/K 对会被计算，哪些位置会被 mask 掉，以及这些计算如何映射到 GPU。

## Q/K/V 的 Shape 与成本直觉

实际实现里，Q/K/V 通常带 batch、head、sequence、head dimension 这些维度。

常见形状可以写成：

```text
Q: [B, Hq, S_q, D]
K: [B, Hk, S_k, D]
V: [B, Hk, S_k, Dv]
O: [B, Hq, S_q, Dv]
```

含义：

- `B` 是 batch size。
- `Hq` 是 query head 数。
- `Hk` 是 key/value head 数。
- `S_q` 是 query 序列长度。
- `S_k` 是 key/value 序列长度。
- `D` 是每个 head 的维度。

普通 multi-head attention 里常见 `Hq = Hk`。GQA/MQA 会让 `Hk` 小于 `Hq`。

核心计算是：

```text
scores = Q @ K^T
prob   = softmax(scores + mask)
O      = prob @ V
```

如果是 dense attention，粗略计算量主要来自两次矩阵乘：

```text
QK^T:  B * H * S_q * S_k * D
PV:    B * H * S_q * S_k * Dv
```

当 `S_q = S_k = S` 时，attention 的这一部分随 `S^2` 增长。

这就是为什么 sequence length 增长很快会压垮 attention：不是多几个 token 这么简单，而是很多 query-key 组合都要重新计算或读取。

## 三种常见 Attention 形态

Attention 不只有 decoder self-attention。

| 类型 | Q 来自 | K/V 来自 | 常见场景 |
| --- | --- | --- | --- |
| bidirectional self-attention | 当前序列 | 当前序列 | BERT、encoder、图像/多模态 encoder |
| causal self-attention | 当前序列 | 当前序列历史位置 | GPT/LLM prefill 和训练 |
| cross-attention | decoder hidden | encoder output 或外部 memory | encoder-decoder、多模态、检索/条件生成 |

这三类的 mask 和 shape 不同。

Bidirectional self-attention 通常可以看完整输入。Causal self-attention 不能看未来。Cross-attention 的 `S_q` 和 `S_k` 可以完全不同，例如 decoder 的每个 token 看 encoder 的全部图像 patch 或文本表示。

做 kernel benchmark 时要写清楚是哪一种。一个对 causal self-attention 很快的实现，不一定适合 cross-attention。

## Attention Pattern 是什么

Attention pattern 描述“每个 token 能看哪些 token”。

在语言模型里，常见约束是 causal mask：第 `t` 个 token 只能看自己和它之前的 token，不能偷看未来。

例如生成第 5 个 token 时，可以看：

```text
token 1, token 2, token 3, token 4, token 5
```

不能看：

```text
token 6, token 7, ...
```

如果从矩阵角度看，attention score 是一个 `sequence_length x sequence_length` 的矩阵。矩阵里的一个格子表示“某个 query token 是否关注某个 key token”。

Dense Attention 和 Sparse Attention 的差别，就是这个矩阵里到底有多少格子需要计算。

## Mask 不只是 Causal Mask

Attention mask 会直接影响计算模式和 kernel 路径。

常见 mask 包括：

| Mask | 含义 | 系统影响 |
| --- | --- | --- |
| causal mask | token 不能看未来 | decoder 训练和推理核心路径 |
| padding mask | padding token 不参与 attention | tokens/s 口径和有效计算相关 |
| document mask | packed batch 中不同文档互相不可见 | packing 后防止样本串扰 |
| sliding window mask | 只看局部窗口 | 长上下文降低 pattern 规模 |
| prefix / bidirectional mask | prefix 可双向，生成部分因果 | PrefixLM、部分多模态模型 |
| block diagonal mask | 多个短序列 pack 到一个 batch | 需要 block-aware kernel 才高效 |

Mask 如果只是逻辑上存在，但 kernel 仍然计算完整 dense matrix 再把某些位置设为 `-inf`，计算量并不会真正下降。

要让 mask 带来系统收益，kernel 必须利用 mask 结构减少计算或减少访存。

## Dense Attention：全量注意力

Dense Attention 也可以叫 full attention、全量注意力或稠密注意力。它的意思是：在允许的 mask 范围内，每个 token 都可以关注所有相关 token。

对于双向编码器模型，例如 BERT 风格模型，一个 token 通常可以看整个输入序列。

对于因果语言模型，例如 GPT 风格模型，第 `t` 个 token 可以看前面所有 token 和自己。

可以用下图理解 causal dense attention：

```text
Query token 1: 看 1
Query token 2: 看 1, 2
Query token 3: 看 1, 2, 3
Query token 4: 看 1, 2, 3, 4
Query token 5: 看 1, 2, 3, 4, 5
```

它是“dense”的原因是：只要 causal mask 允许，位置之间都可以建立 attention 连接。

## Dense Attention 的成本

Dense Attention 的主要问题是序列长度越长，成本增长越快。

如果序列长度是 `n`，每个 token 都要和很多 token 做匹配，attention score 的规模大约是：

```text
n x n
```

也就是 `O(n^2)`。

这会带来两个问题。

第一是计算量。序列越长，QK 匹配和 attention weighted sum 的计算越多。

第二是显存和访存。如果直接把完整 attention score 或 softmax 矩阵写到显存，长序列时会产生大量 HBM 读写。

所以长上下文训练和推理里，Dense Attention 常常成为核心瓶颈。

### Naive Dense Attention 的中间矩阵

最直接的 dense attention 实现会产生几个大中间对象：

```text
scores: [B, H, S_q, S_k]
prob:   [B, H, S_q, S_k]
O:      [B, H, S_q, D]
```

其中 `scores` 和 `prob` 都随 `S_q * S_k` 增长。

对长序列来说，瓶颈往往不只是 FLOPs，还有：

- `scores` 写入 HBM。
- `scores` 从 HBM 读出做 softmax。
- `prob` 写入 HBM。
- `prob` 从 HBM 读出乘 V。
- backward 还要保存或重算中间状态。

这就是 FlashAttention 类 kernel 出现的原因：attention 的优化不能只看算术量，还要看中间矩阵是否被物化，以及 HBM IO 有多大。

## Dense Attention 的优点

Dense Attention 成本高，但它也有重要优点。

- 表达能力强，每个 token 都有机会读取任意上下文。
- 模型结构简单，训练和推理行为一致。
- kernel 和框架支持成熟。
- 对任务质量更稳，不需要提前假设哪些 token 重要。

很多主流 LLM 的基础 attention 仍然是 dense causal attention，只是在实现层面用 FlashAttention、KV Cache、PagedAttention、GQA/MQA 等方法优化系统成本。

因此不能简单说 Dense Attention 落后。它的缺点主要是成本高，而不是机制不合理。

## Sparse Attention：稀疏注意力

Sparse Attention 的思想是：不是每个 token 都看所有 token，而是只看一部分 token。

它改变的是 attention pattern。

例如，一个 token 只看附近窗口：

```text
token t 只看 t-2, t-1, t, t+1, t+2
```

或者只看某些全局 token、随机 token、固定间隔 token。

如果每个 token 只看 `w` 个 token，而不是看 `n` 个 token，那么 attention 成本可以从大约 `O(n^2)` 降到接近：

```text
O(n * w)
```

当 `w` 远小于 `n` 时，这对长序列很有吸引力。

## 常见 Sparse Attention Pattern

Sparse Attention 有很多形式。下面是常见几类。

| 模式 | 思想 | 适合直觉 |
| --- | --- | --- |
| Local / Sliding Window | 每个 token 只看附近窗口 | 语言和图像中局部信息很重要 |
| Global Tokens | 少数特殊 token 可以看全局，也被全局看到 | 分类 token、摘要 token、重要锚点 |
| Strided / Dilated | 按固定间隔看远处 token | 低成本获得长距离感受野 |
| Block Sparse | 把序列分块，只计算部分 block 之间的 attention | 更适合 GPU 上块状计算 |
| Random Sparse | 随机连接部分远距离 token | 增加全局连通性 |
| Learned / Dynamic Sparse | 根据内容动态决定看谁 | 更灵活，但系统实现更复杂 |

不同 sparse pattern 的质量和性能差异很大。不能只看复杂度公式，还要看模型是否适应这种模式，以及 kernel 是否真的高效。

## Sparse Attention 的收益和代价

Sparse Attention 的收益很直观：

- 降低 attention 计算量。
- 降低 attention score 显存需求。
- 支持更长上下文。
- 在某些任务中能把有限算力集中到更重要位置。

但代价也很明显：

- 模型看不到所有 token，可能损失质量。
- pattern 设计不当会漏掉关键远距离依赖。
- 训练和推理必须匹配，否则容易出现行为不一致。
- 动态稀疏会让 batching、kernel 和缓存管理更复杂。
- 理论复杂度降低，不代表 GPU wall-clock 一定更快。

最后一点很重要。GPU 擅长规则、密集、块状的大矩阵计算。如果 sparse pattern 很碎、很不规则，虽然计算次数少了，但硬件利用率也可能下降。

### Sparse Attention 的 Kernel 难点

Sparse Attention 在论文公式里看起来很省，但在 GPU 上未必容易快。

原因包括：

- token 级稀疏会导致不规则访存。
- 每个 query 看到的 key 数不同，容易造成 warp divergence。
- 小块过多会增加 kernel launch 和调度开销。
- 稀疏索引本身要占内存和带宽。
- backward 也要支持同样 sparse pattern。
- batching 时不同样本的稀疏结构可能不同。

所以工程上更偏好 block sparse，而不是完全任意稀疏。

Block sparse 的优势是把稀疏性提升到块级别，让 GPU 仍然可以做较规则的 tile 计算。但它也要求 pattern 适合块状表达，否则会出现很多空算或碎片。

## Local / Sliding Window Attention

Sliding Window Attention 是最容易理解的一类 sparse attention。

每个 token 只看附近窗口，例如前后各 `w` 个 token。

```text
token 10 只看 token 6 ~ token 14
```

它适合处理局部依赖明显的序列。对于很长文档，邻近内容通常最相关，因此 local attention 可以显著减少成本。

但它的风险是远距离信息传递变慢。如果第 100 个 token 需要直接引用第 1 个 token，纯 sliding window 可能做不到，除非堆叠很多层，或者加入 global token、dilated pattern 等补充连接。

## Block Sparse Attention

Block Sparse Attention 把 attention 矩阵切成块，只计算某些块。

例如序列长度很长时，不再按单个元素决定是否计算，而是按 `block x block` 决定。

这样做的原因是 GPU 更喜欢块状计算。相比零散的 token 级稀疏，block sparse 更容易映射到高效 kernel。

可以把它理解为：

```text
不是问“这个 token 看不看那个 token”
而是问“这一块 token 看不看那一块 token”
```

Block sparse 常见于长上下文模型、稀疏 Transformer 变体和某些高性能 attention kernel 设计中。

## MHA、MQA 与 GQA

Attention pattern 之外，还有 head 结构问题。

常见三种：

| 结构 | K/V head 数 | 直觉 |
| --- | --- | --- |
| MHA, Multi-Head Attention | `Hk = Hq` | 每个 query head 有独立 K/V |
| MQA, Multi-Query Attention | `Hk = 1` | 所有 query head 共享一组 K/V |
| GQA, Grouped-Query Attention | `1 < Hk < Hq` | 多个 query head 共享一组 K/V |

MHA 表达灵活，但 KV Cache 大。

MQA/GQA 会减少 K/V head 数，从而减少：

- KV Cache 显存。
- Decode 阶段读取 KV 的带宽。
- 多机推理时 KV 相关通信或拷贝压力。

代价是 K/V 表达能力减少，模型质量和训练 recipe 需要验证。

从 kernel 角度看，GQA/MQA 会改变 Q/K/V 的 shape 和广播方式。一个 attention kernel 如果只按 MHA 假设实现，可能不能高效支持 GQA/MQA。

## FlashAttention：不是 Sparse Attention

FlashAttention 经常和 sparse attention 混在一起讨论，因为它也能让 attention 更快、更省显存。但它们不是一类东西。

FlashAttention 的核心是：在不改变 dense attention 数学结果的前提下，用 IO-aware 的 tiling 方法减少 HBM 和片上 SRAM 之间的数据搬运，并避免把完整 attention 矩阵写回显存。

也就是说：

- Dense Attention / Sparse Attention 讨论“哪些位置参与 attention”。
- FlashAttention 讨论“同样的位置参与 attention 时，如何更高效地算”。

FlashAttention 默认是 exact attention。它不是通过少看 token 来省成本，而是通过更好的 kernel 算法和内存访问方式来省显存和提升速度。

## FlashAttention 为什么省显存

普通实现可能会显式生成 attention score 矩阵：

```text
QK^T -> attention scores -> softmax -> attention weights -> weights * V
```

当序列长度是 `n` 时，attention score 是 `n x n`。长序列下，这个矩阵很大。

FlashAttention 的做法是分块计算，并在块内维护 softmax 所需的中间统计量。它不需要把完整 attention 矩阵一次性存到 HBM。

简化理解：

```mermaid
flowchart TB
    A["Q block"] --> C["块内计算 QK"]
    B["K/V block"] --> C
    C --> D["在线更新 softmax 统计量"]
    D --> E["累积输出 O block"]
    E --> F["写回最终结果"]
```

这样做的收益是减少大规模中间矩阵的显存读写。对 GPU 来说，很多 attention 的瓶颈不是算术运算本身，而是数据在 HBM 和片上存储之间搬来搬去。

FlashAttention 的关键启发是：attention kernel 要关心 IO，不只是关心 FLOPs。

### Online Softmax 的直觉

普通 softmax 需要先知道一整行 score 的最大值，再做指数和归一化：

```text
softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
```

如果分块计算，就不能一次看到完整行。

FlashAttention 的关键技巧之一，是在遍历 K/V block 时维护每个 query 行的统计量：

```text
current max
current exp sum
current output accumulator
```

当新 block 到来时，更新 max 和 exp sum，并把旧 accumulator 重新缩放到新的归一化基准上。

这样就可以边读 K/V block，边更新 softmax 和输出，不需要把整张 `S x S` score 矩阵写回 HBM。

### FlashAttention 的 Kernel 视角

从 GPU kernel 角度看，FlashAttention 关心：

- Q block 怎么驻留在 SRAM/register。
- K/V block 怎么从 HBM 分块读入。
- score block 怎么在片上计算。
- softmax 统计量怎么在线更新。
- 输出 O block 怎么累计。
- backward 怎么用保存的少量统计量重算或恢复梯度。

它的核心收益是减少 HBM 读写，尤其是避免大 attention matrix 的 materialization。

这也解释了为什么 FlashAttention 在长序列场景收益明显：`S x S` 中间矩阵越大，减少 IO 的价值越高。

### FlashAttention-2 的方向

FlashAttention-2 继续优化的是并行度和工作划分。

直觉上，FlashAttention-1 已经减少了大量 HBM IO，但 GPU 上还要让更多 warps/blocks 高效工作，减少非矩阵乘部分的开销，并改进不同 sequence/head 维度上的并行切分。

这说明 attention kernel 优化不只是一条原则，而是多层问题：

- IO 是否减少。
- tile 是否合适。
- work partition 是否均衡。
- Tensor Core 是否充分利用。
- 非 matmul 部分是否成为瓶颈。
- 不同 sequence length 和 head dim 是否都高效。

## FlashAttention、Dense、Sparse 的关系

三者关系可以这样看：

| 概念 | 改变 attention pattern 吗 | 是否近似 | 主要解决 |
| --- | --- | --- | --- |
| Dense Attention | 不改变，全量看允许位置 | 否 | 表达能力强，但成本高 |
| Sparse Attention | 改变，只看部分位置 | 通常是结构性近似 | 降低长序列计算和显存 |
| FlashAttention | 不改变默认 pattern | 否 | 降低 IO 和中间显存，提高 exact attention 执行效率 |

也可以这样记：

- Dense Attention 是“都看”。
- Sparse Attention 是“只看一部分”。
- FlashAttention 是“还是都看，但算得更聪明”。

## FlashAttention 与 PagedAttention 的区别

这两个名字都带 Attention，也都出现在推理系统里，但它们解决的问题不同。

| 概念 | 主要对象 | 解决问题 |
| --- | --- | --- |
| FlashAttention | attention kernel 计算过程 | 减少 attention 中间矩阵和 HBM IO |
| PagedAttention | KV Cache 显存管理 | 用 block/page 管理请求的 KV Cache，减少碎片和预留浪费 |

FlashAttention 关注“attention 这次怎么算更高效”。

PagedAttention 关注“历史 KV Cache 存在哪里、怎么分配、怎么共享、怎么释放”。

所以 PagedAttention 不是 sparse attention，也不是 FlashAttention。它们可以在同一个推理系统中配合出现。

### PagedAttention 的 Block Table 直觉

自回归推理中，每个请求都会不断增长 KV Cache。

朴素做法容易要求每个请求预留一段连续显存：

```text
request A: [KV KV KV KV .... reserved ....]
request B: [KV KV .... reserved ....]
```

问题是不同请求长度不同，容易产生碎片和浪费。

PagedAttention 的直觉是把 KV Cache 切成固定大小的 block/page，再用 block table 记录逻辑位置到物理 block 的映射：

```text
logical token blocks:
  block 0 -> physical block 17
  block 1 -> physical block 42
  block 2 -> physical block 09
```

这样一个请求的 KV Cache 不需要物理连续。调度器可以按需分配、释放、复用 block。

它改变的是 KV Cache 管理，不是 attention pattern。模型仍然可以 dense 地看历史 token，只是 K/V 的存储更像分页内存。

## 训练和推理中的差异

Attention 在训练和推理中的系统压力不同。

### 训练

训练要做 forward 和 backward。Attention 不只要计算输出，还要为反向传播保存或重算中间状态。

训练中关注：

- attention forward 时间。
- attention backward 时间。
- activation 显存。
- sequence length 对显存的平方级压力。
- FlashAttention 是否支持 backward。
- sparse pattern 是否影响收敛和质量。

训练里的 backward 很重要。即使 forward 用了高效 kernel，backward 仍然要计算：

- `dQ`。
- `dK`。
- `dV`。
- softmax 相关梯度。
- mask 后梯度传播。

如果普通实现保存完整 attention probability，activation 显存会很高。如果使用 FlashAttention 类实现，通常会保存少量统计量，在 backward 中重算部分中间结果，以减少显存占用。

所以训练 benchmark 不能只测 attention forward。必须测 forward + backward，并记录 activation memory。

### 推理

推理尤其是自回归 Decode，会大量依赖 KV Cache。

推理中关注：

- Prefill 阶段长 prompt 的 attention 计算。
- Decode 阶段读取历史 KV Cache 的带宽压力。
- KV Cache 显存占用。
- PagedAttention / KV Cache block 管理。
- sliding window 是否能减少需要保留的 KV。
- FlashAttention 是否用于 Prefill 或批量 Decode。

因此，训练里的 attention 优化和推理里的 attention 优化不能完全混用。训练更关心 forward/backward 和 activation；推理更关心 Prefill、Decode、KV Cache 和在线调度。

### Prefill 与 Decode 的 Attention 形态

推理里 Prefill 和 Decode 的 attention 形态不同。

Prefill：

```text
S_q = prompt length
S_k = prompt length
```

它更像训练时的 causal dense attention，能用 FlashAttention 类 kernel 优化长 prompt。

Decode：

```text
S_q = 1 或很小
S_k = history length
```

每一步只生成少量新 token，但要读取很长历史 KV。此时瓶颈常常从 compute 变成 KV Cache 读取带宽、batching 和 cache 管理。

所以：

- FlashAttention 对长 prompt prefill 很重要。
- PagedAttention / KV Cache layout 对 decode 很重要。
- MQA/GQA 对 decode 的 KV 带宽和显存很重要。

把 Prefill 和 Decode 混在一个 attention benchmark 里，会很难解释结果。

## 对长上下文的影响

长上下文是 attention 优化最重要的场景之一。

序列长度从 4K 增加到 32K，不只是增加 8 倍 token。对于 dense attention 的某些计算和中间状态，压力可能接近平方级增长。

常见优化方向包括：

- 用 FlashAttention 降低 exact dense attention 的 IO 和中间显存。
- 用 sliding window 或 block sparse 降低 attention pattern 的规模。
- 用 GQA / MQA 降低 KV Cache 大小。
- 用 PagedAttention 降低 KV Cache 显存管理浪费。
- 用 sequence/context parallel 把长序列分到多 GPU。
- 用 activation checkpointing 降低训练 activation 显存。

这些方法解决的问题不同，通常可以组合使用。

## Attention 与并行

长上下文和大模型训练里，attention 也会和多 GPU 并行耦合。

常见方式包括：

| 并行方式 | 和 attention 的关系 |
| --- | --- |
| Tensor Parallel | 按 head 或 projection 矩阵切分 Q/K/V/O |
| Sequence Parallel | 把部分 activation 按 sequence 维度切分，常和 TP 组合 |
| Context Parallel | 把长上下文切到多个 GPU，attention 需要跨 context 通信 |
| Pipeline Parallel | attention layer 跟随 layer stage 分布 |
| Expert Parallel | MoE 层相关，通常与 attention 层交替出现 |

对于 attention，最敏感的是 TP 和 CP。

TP 常按 head 或 hidden dimension 切，让单卡处理更少 head 或更小矩阵。但 TP 可能带来层内 AllReduce/AllGather。

CP 面向长上下文，把 sequence/context 切开。它能降低单卡长序列压力，但 attention 需要跨设备交换 K/V 或中间结果。

因此 attention kernel 的性能不能只看单卡。多 GPU 下还要看：

- Q/K/V projection 是否需要 collective。
- attention 输出是否需要 reduce 或 gather。
- 长上下文是否引入 ring/context 通信。
- communication 是否能和 compute overlap。
- rank mapping 是否让高频通信留在高速互连内。

## Attention Kernel 的输入约束

一个 attention kernel 往往只在某些 shape 上高效。

需要记录：

- dtype：FP16、BF16、FP8。
- head dimension：例如 64、80、96、128、256。
- sequence length。
- causal / non-causal。
- dropout 是否启用。
- GQA/MQA 是否支持。
- sliding window 是否支持。
- variable length batch 是否支持。
- block diagonal mask 是否支持。
- forward-only 还是 forward+backward。

例如某个 kernel 对 `head_dim=128` 很快，不代表对 `head_dim=96` 也快。某个 prefill kernel 很快，不代表 decode 高效。

这也是为什么 PyTorch SDPA、xFormers、FlashAttention、Triton 自定义 kernel 常常需要根据 shape、dtype、mask 和硬件选择不同 backend。

## 什么时候用 Dense

Dense Attention 适合：

- 对质量要求高，不希望限制 token 间连接。
- 序列长度还在可承受范围内。
- 模型预训练就是 dense attention。
- 需要简单稳定的训练和推理行为。
- 有 FlashAttention 等高效 exact attention kernel 支持。

Dense Attention 的核心优势是稳。很多情况下，先把 dense exact attention 的实现优化好，比直接改成 sparse pattern 更可控。

## 什么时候考虑 Sparse

Sparse Attention 适合：

- 上下文很长，dense attention 成本不可接受。
- 任务天然有局部结构，例如长文档、视频、图像 patch 或时间序列。
- 可以接受或通过训练弥补 attention pattern 的限制。
- sparse pattern 能映射到高效 kernel。
- 模型架构从训练阶段就使用相同 sparse pattern。

不建议在没有验证的情况下，把 dense 模型推理时直接替换成 sparse attention。训练和推理不匹配，可能导致质量明显下降。

## 什么时候关注 FlashAttention

FlashAttention 适合：

- 仍然希望保持 exact attention。
- attention score 中间矩阵显存压力大。
- 长序列训练或 Prefill 成本高。
- GPU HBM IO 成为瓶颈。
- 框架和硬件已经支持对应 kernel。

如果系统瓶颈在 tokenizer、KV Cache 显存、网络、调度或工具调用，FlashAttention 不一定能解决问题。它主要优化 attention kernel 本身。

## 什么时候考虑 PagedAttention

PagedAttention 适合：

- 推理服务需要同时处理大量变长请求。
- KV Cache 显存碎片和预留浪费明显。
- 请求会频繁进入、增长、完成和释放。
- 需要 prefix/cache sharing 或更灵活的 KV block 管理。
- Decode 阶段 KV Cache 成为容量瓶颈。

PagedAttention 不直接降低模型数学计算量。它的价值在服务系统里更明显，尤其是 continuous batching、多请求混部和长上下文推理。

## 性能分析时看什么

分析 attention 相关性能时，不要只看理论复杂度。

建议同时看：

| 维度 | 要看什么 |
| --- | --- |
| 算法复杂度 | attention score 数量是否随 `n^2` 增长 |
| 显存 | 是否物化 `n x n` attention 矩阵，activation 占多少 |
| HBM IO | 是否频繁读写大中间矩阵 |
| Kernel 效率 | occupancy、tensor core 使用、warp divergence |
| Pattern 规则性 | sparse pattern 是否适合块状 GPU 计算 |
| 质量影响 | sparse pattern 是否影响任务效果 |
| 训练/推理一致性 | 训练 pattern 和推理 pattern 是否一致 |

对于长上下文模型，只看 FLOPs 往往不够。访存、缓存、通信和 kernel 实现会决定真实 wall-clock 性能。

## Attention Benchmark 方法

Attention benchmark 至少要区分四类场景：

| 场景 | 测什么 |
| --- | --- |
| training forward+backward | 训练端到端 attention 成本和 activation 显存 |
| prefill forward | 长 prompt attention kernel 吞吐 |
| decode attention | `S_q` 很小、读取历史 KV 的带宽和 latency |
| serving workload | KV Cache 管理、batching、请求长度分布 |

报告中应写清楚：

```yaml
attention_benchmark:
  dtype: bf16
  batch_size: 8
  num_q_heads: 32
  num_kv_heads: 8
  head_dim: 128
  query_length: 4096
  key_length: 4096
  causal: true
  dropout: false
  mask: causal
  backend: flash_attention
  measure: forward_backward
```

还要记录：

- latency。
- TFLOPs 或有效 tokens/s。
- peak memory。
- HBM bandwidth 指标。
- kernel 数量。
- 是否发生 fallback。
- 数值误差。
- 端到端模型 step time 是否真的改善。

Microbenchmark 只证明 kernel 在某些 shape 下快。最终仍要回到训练 step 或推理服务 workload 验证。

## 常见优化决策表

| 现象 | 可能方向 |
| --- | --- |
| 长序列训练 OOM | FlashAttention、activation checkpointing、SP/CP、降低 batch |
| Attention forward 显存高 | 避免 materialize score/prob，使用 fused/Flash kernel |
| Decode KV Cache 爆显存 | GQA/MQA、PagedAttention、sliding window、KV quantization |
| Prefill 慢 | FlashAttention、better SDPA backend、batching、TP/CP |
| Sparse 理论快但实际慢 | 检查 pattern 是否 block-friendly、kernel 是否 fallback |
| 多 GPU attention 扩展差 | 检查 TP/CP 通信、rank mapping、overlap |
| 端到端无收益 | attention 不是瓶颈，检查 MLP、optimizer、data、通信 |

## 常见误区

### 1. FlashAttention 是稀疏注意力

不是。FlashAttention 默认是 exact attention，它不靠少看 token 来近似，而是通过 IO-aware tiling 更高效地计算。

### 2. PagedAttention 是稀疏注意力

不是。PagedAttention 管理 KV Cache block。它改变 KV Cache 的存储和访问方式，不等于减少模型关注的历史 token。

### 3. Sparse Attention 一定更快

不一定。稀疏 pattern 如果不规则，GPU 利用率可能很差。理论计算量少，不代表实际运行更快。

### 4. Dense Attention 一定不能做长上下文

不准确。Dense Attention 成本高，但 FlashAttention、GQA/MQA、并行、显存优化和高端硬件可以把可用长度推高。只是成本仍然需要评估。

### 5. 只要 attention 快，模型就快

不一定。Transformer 还有 MLP、LayerNorm、embedding、sampling、通信、KV Cache、数据输入等环节。attention 只是重要瓶颈之一。

### 6. Mask 变稀疏就一定减少计算

不一定。如果 kernel 仍然计算完整 score matrix，再把某些位置 mask 掉，计算量没有真正下降。要看 kernel 是否利用 mask 结构跳过计算或减少 IO。

### 7. Prefill 和 Decode 可以用同一个结论判断

不稳妥。Prefill 通常是长序列 dense attention；Decode 通常是少量 query 读取长 KV Cache。两者的瓶颈不同。

## 一个最小理解例子

假设序列长度是 8192。

Dense Attention 的直觉是：每个 token 都可以看前面所有 token。这样信息最完整，但 attention score 数量很大。

Sparse Attention 的直觉是：每个 token 只看附近 512 个 token，再加少量全局 token。这样成本低很多，但模型可能看不到某些远距离信息。

FlashAttention 的直觉是：仍然让每个 token 看所有允许的 token，但不要把完整 `8192 x 8192` 中间矩阵写到显存，而是分块算、边算边归一化、最后写回结果。

这三个方案解决的是不同问题：

- Dense：保证全量连接。
- Sparse：减少连接数量。
- FlashAttention：优化全量连接的执行方式。

## 学习路径建议

如果刚开始学习 attention 计算模式，可以按这个顺序：

1. 先理解普通 Q/K/V attention。
2. 再理解 causal mask 和 attention matrix。
3. 学习 Dense Attention 为什么是 `O(n^2)`。
4. 学习 Sparse Attention 如何改变 attention pattern。
5. 学习 FlashAttention 为什么是 IO-aware exact attention。
6. 学习 MHA/MQA/GQA 如何影响 KV Cache 和 kernel shape。
7. 区分训练 forward/backward、推理 Prefill 和 Decode。
8. 最后区分 FlashAttention、PagedAttention、KV Cache、Prefix Cache 各自解决什么问题。

这样能避免把所有带 Attention 的名词混成一类。

## 小结

Dense Attention、Sparse Attention 和 FlashAttention 是理解高效 Transformer 的基础概念。

核心结论：

- Dense Attention 在允许范围内全量关注上下文，表达能力强，但长序列成本高。
- Sparse Attention 通过限制 attention pattern 降低成本，但可能影响质量，也不一定天然适合 GPU。
- FlashAttention 不改变 attention pattern，而是用 IO-aware kernel 更高效地计算 exact attention。
- PagedAttention 是 KV Cache 管理方法，不是稀疏注意力。
- MQA/GQA 通过减少 K/V head 降低 KV Cache 和 decode 带宽压力。
- Prefill、Decode、训练 backward 的 attention 瓶颈不同，benchmark 必须分开。
- 训练和推理中的 attention 瓶颈不同，优化方法要结合具体 workload。

对关注高效计算的人来说，真正重要的是分清楚：到底是在改模型能看的位置，还是在改同一计算的执行方式，还是在改 KV Cache 的存储管理。

## 参考资料

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509)
- [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)
- [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062)
- [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [PyTorch: scaled_dot_product_attention](https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.scaled_dot_product_attention.html)
