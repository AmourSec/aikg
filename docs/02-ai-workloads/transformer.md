---
title: Transformer 流程与原理
domain: transformer
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-06-01
---

# Transformer 流程与原理

Transformer 是当前大语言模型和很多多模态模型的核心结构。学习它时不要一开始就背公式，而要先抓住主线：

```text
把一串 token 变成一串向量；
让每个位置用 Attention 读取上下文；
再用 MLP 对每个位置做非线性变换；
重复很多层；
最后把当前位置向量映射成下一个 token 的概率。
```

本页面面向刚进入 AI Infra 方向的读者。目标不是完整推导论文，而是让你能看懂 Transformer 的数据流、关键张量形状、训练和推理差异，以及为什么它会影响推理系统、Kernel、编译器和硬件架构。

## 学习目标

读完本页，你应该能回答：

- 一个 token 从输入到 logits 经过哪些步骤。
- Attention 中 Q、K、V 分别从哪里来，为什么会有 `QK^T`。
- Multi-Head Attention、MLP、Residual、LayerNorm 在一个 block 里如何配合。
- Encoder-only、Decoder-only、Encoder-Decoder 的区别。
- 训练、prefill、decode 阶段为什么性能瓶颈不同。
- sequence length、hidden size、head 数如何影响计算量和显存。

## 为什么需要 Transformer

早期序列模型常用 RNN/LSTM/GRU。它们按时间步顺序处理 token：第 `i` 个位置依赖第 `i-1` 个位置的状态。这种结构有两个问题：

- **并行性差**：必须从左到右逐步算，长序列训练很难充分利用 GPU/TPU。
- **长距离依赖困难**：很远的 token 要经过很多步状态传递，信息容易衰减。

卷积模型可以并行，但单层卷积只能覆盖局部窗口；要让远距离 token 交互，需要堆很多层。

Transformer 的核心想法是：在每一层里，让序列中任意两个可见位置都可以直接交互。这个交互由 Self-Attention 完成。原始论文提出的 Transformer 完全基于 Attention 机制，不再使用 recurrence 或 convolution；它在机器翻译任务上获得更好的质量，同时更容易并行训练。

## 三种常见 Transformer 形态

原始 Transformer 是 Encoder-Decoder 架构，但今天的大模型经常只使用其中一部分。

| 架构 | 代表任务 | Attention 形式 | 输出方式 |
| --- | --- | --- | --- |
| Encoder-Decoder | 翻译、摘要、T5 类模型 | Encoder 双向 self-attention；Decoder masked self-attention + cross-attention | Decoder 自回归生成 |
| Encoder-only | BERT 类理解模型 | 双向 self-attention，可以看左右上下文 | 输出每个位置或整体表示 |
| Decoder-only | GPT、LLaMA 类语言模型 | masked self-attention，只能看当前位置和历史位置 | 逐 token 预测下一个 token |

对 AI Infra 来说，最常遇到的是 **Decoder-only LLM**，因为它对应在线推理中的 prefill、decode、KV Cache、batching 和长上下文问题。但理解 Encoder-Decoder 有助于读原始论文和很多教程。

## 总体数据流

以 Decoder-only 语言模型为例，推理时的数据流可以写成：

```text
文本
  -> tokenizer
  -> token IDs: [B, S]
  -> token embedding: [B, S, D]
  -> 加入位置信息
  -> Transformer Block x L
       -> LayerNorm
       -> Masked Multi-Head Self-Attention
       -> Residual Add
       -> LayerNorm
       -> MLP / FFN
       -> Residual Add
  -> final LayerNorm
  -> LM Head: [B, S, Vocab]
  -> logits
  -> sampling / greedy / beam search
  -> 下一个 token
```

其中：

- `B` 是 batch size。
- `S` 是 sequence length。
- `D` 是 hidden size，也叫 `d_model`。
- `L` 是 Transformer block 层数。
- `Vocab` 是词表大小。

## Token、Embedding 和位置信息

模型不能直接处理文本，第一步是 tokenizer 把文本切成 token，并映射成整数 ID。

```text
"Transformer 很重要"
  -> ["Transformer", " 很", "重要"]
  -> [15342, 421, 9231]
```

Embedding 层是一个查表：

```text
embedding_table: [Vocab, D]
token_ids:       [B, S]
output:          [B, S, D]
```

每个 token ID 会查到一个 `D` 维向量。到这一步，模型知道“每个位置是什么 token”，但还不知道“它在第几个位置”。

Self-Attention 本身对顺序不敏感。如果把输入 token 打乱，只看一组向量之间的相似度，模型不能天然知道谁在前谁在后。因此 Transformer 需要加入位置信息。常见方式包括：

- **Sinusoidal positional encoding**：原始论文使用的正弦/余弦位置编码。
- **Learned positional embedding**：位置向量也作为参数学习。
- **RoPE**：现代 LLM 常见的旋转位置编码，便于建模相对位置。

在工程上，你会经常看到两类写法：

```text
x = token_embedding + position_embedding
```

或在 Attention 里对 Q/K 注入 RoPE。

## Self-Attention 的直觉

Self-Attention 回答的问题是：

> 当我更新当前位置的表示时，应该从哪些上下文位置读取信息？各读多少？

例如句子：

```text
The animal didn't cross the street because it was tired.
```

当模型处理 `it` 时，它需要知道 `it` 更可能指向 `animal`，而不是 `street`。Self-Attention 允许 `it` 这个位置直接查看上下文中其他位置，并根据相关性加权读取信息。

对第 `i` 个位置，可以把 Attention 理解成：

```text
第 i 个位置的新表示
  = 对所有可见位置 j 的 value 向量做加权求和
```

权重不是固定参数，而是由当前位置和候选位置的内容动态算出来。

## Attention 在做什么

给定输入 `X: [B, S, D]`，Attention 会用三组线性投影生成 Q、K、V：

```text
Q = X Wq    # Query
K = X Wk    # Key
V = X Wv    # Value
```

可以用一个检索类比理解：

- `Q`：当前位置想查什么。
- `K`：每个位置提供什么匹配线索。
- `V`：如果被关注，真正被读走的内容。

但要注意：Q/K/V 不是人工设计的数据库字段，而是模型通过训练学出来的线性变换结果。

Attention 的核心公式是：

```text
scores = Q K^T / sqrt(d_head)
weights = softmax(scores + mask)
output = weights V
```

含义是：

1. `QK^T` 计算每个 query 位置和每个 key 位置的匹配分数。
2. 除以 `sqrt(d_head)` 是缩放，避免点积值过大导致 softmax 梯度不稳定。
3. `mask` 用于屏蔽 padding 或未来 token。
4. `softmax` 把分数变成权重。
5. 权重乘以 `V`，得到每个位置从上下文读到的新信息。

## 用 3 个 token 走一遍 Self-Attention

为了避免公式太抽象，先用一个短序列做心智模型：

```text
tokens = [我, 喜欢, AI]
hidden = [x0, x1, x2]
```

假设现在要更新位置 2，也就是 `AI` 这个 token 的表示。

第一步，每个位置都会从自己的 hidden state 生成 Q、K、V：

```text
x0 -> q0, k0, v0
x1 -> q1, k1, v1
x2 -> q2, k2, v2
```

第二步，用当前位置的 query `q2` 去匹配所有可见位置的 key：

```text
score(2, 0) = q2 dot k0
score(2, 1) = q2 dot k1
score(2, 2) = q2 dot k2
```

第三步，对这些分数做缩放、mask 和 softmax，得到权重：

```text
weights(2) = [0.20, 0.30, 0.50]
```

第四步，用权重加权求和 value：

```text
z2 = 0.20 * v0 + 0.30 * v1 + 0.50 * v2
```

`z2` 就是位置 2 从上下文读取后的新信息。它会再经过 output projection、residual add、norm 和 MLP，变成下一层的输入。

注意两个关键点：

- 对位置 2 来说，attention 权重是一行分布：它表示位置 2 读取各个历史位置的比例。
- 实际实现不会用 Python for-loop 一个 token 一个 token 算，而是把所有位置打包成矩阵，一次性算出 `QK^T`。

## 张量形状：从 X 到 Attention 输出

假设：

- `B = batch size`
- `S = sequence length`
- `D = hidden size`
- `H = num_heads`
- `Dh = D / H`

典型 Multi-Head Attention 的形状如下：

| 张量 | 形状 | 含义 |
| --- | --- | --- |
| `X` | `[B, S, D]` | 输入 hidden states |
| `Q, K, V` | `[B, S, D]` | 一次线性层后通常仍是 D 维 |
| reshape 后 | `[B, H, S, Dh]` | 拆成多个 head |
| `QK^T` | `[B, H, S, S]` | 每个 head 的注意力分数矩阵 |
| `softmax` 后 | `[B, H, S, S]` | 每个 query 对所有 key 的权重 |
| `weights V` | `[B, H, S, Dh]` | 每个 head 的输出 |
| concat 后 | `[B, S, D]` | 多个 head 拼回 hidden size |
| output projection | `[B, S, D]` | 经过 `Wo` 混合多个 head |

这张表对性能分析非常重要：`[B, H, S, S]` 说明 prefill 阶段的注意力分数矩阵会随序列长度平方增长。

## Causal Mask：为什么 Decoder 不能看未来

语言模型训练目标通常是预测下一个 token。

```text
输入:  [我, 喜欢, AI]
目标:  [喜欢, AI, <eos>]
```

训练时为了并行，模型会一次性看到整段序列。但预测第 `i` 个位置时不能偷看第 `i+1` 个位置，否则训练目标就泄漏了答案。

Decoder-only 模型使用 causal mask：

```text
位置 0 只能看 0
位置 1 只能看 0, 1
位置 2 只能看 0, 1, 2
...
```

矩阵上就是屏蔽右上三角。被屏蔽的位置在 softmax 前通常被加上一个极小值，使其权重接近 0。

## Multi-Head Attention 为什么需要多个头

单个 attention head 会产生一套权重分布。Multi-Head Attention 则把 hidden size 分成多个子空间，让多个 head 并行做 Attention：

```text
head_1 = Attention(Q Wq_1, K Wk_1, V Wv_1)
head_2 = Attention(Q Wq_2, K Wk_2, V Wv_2)
...
output = Concat(head_1, ..., head_H) Wo
```

直观上，不同 head 可以学习不同关系：

- 一个 head 关注当前位置附近的词。
- 一个 head 关注主语和谓语。
- 一个 head 关注括号、代码缩进或引用关系。

这只是帮助理解的类比。实际模型中，head 的功能由训练数据和目标函数共同决定，不一定能被简单命名。

系统视角下，如果总 hidden size `D` 固定，多头通常不是把计算量乘以 `H`，因为每个 head 的维度是 `Dh = D / H`。但是 head 数会影响张量布局、并行粒度、Kernel 实现和 KV Cache 排布。

## MLP 在做什么

Attention 负责 token 之间的信息交互；MLP/FFN 负责对每个 token 的表示做非线性变换。

原始 Transformer 的 FFN 是：

```text
FFN(x) = LinearDown(Activation(LinearUp(x)))
```

典型形状：

```text
[B, S, D]
  -> LinearUp:   [B, S, Dff]
  -> Activation: [B, S, Dff]
  -> LinearDown: [B, S, D]
```

`Dff` 通常比 `D` 大，例如 4 倍。现代 LLM 常用 SwiGLU / GeGLU 等门控 MLP 变体，但主线仍然是“升维 -> 激活/门控 -> 降维”。

注意 MLP 是 **逐位置独立** 的：第 `i` 个 token 的 MLP 不直接读取第 `j` 个 token。跨 token 的信息交换主要发生在 Attention。

系统视角下：

- MLP 主要是大 GEMM，通常更偏 compute-bound。
- Attention 涉及 QK、softmax、mask、PV 和 KV Cache，更容易受显存带宽、layout 和 kernel fusion 影响。
- LayerNorm、RMSNorm、RoPE、residual add 等算子单次计算量不大，但在 decode 小 batch 场景中也可能显著影响延迟。

## Residual 和 LayerNorm 为什么重要

Transformer block 会堆很多层。深层网络训练时容易出现梯度不稳定，所以每个子层周围通常有：

```text
x -> Sublayer -> + x -> Norm
```

或现代 LLM 更常见的 Pre-Norm：

```text
x -> Norm -> Sublayer -> + x
```

Residual connection 让信息和梯度可以跨过子层直接流动。LayerNorm/RMSNorm 则稳定每个 token 的 hidden state 分布。

原始 Transformer 使用 Add & Norm，也就是 Post-Norm；很多现代大模型为了更稳定地训练深层网络，使用 Pre-Norm 或 RMSNorm。

## 一个 Decoder-only Block 的完整流程

现代 Decoder-only LLM 的一个 block 可以概括为：

```text
输入 x: [B, S, D]

1. Attention 子层
   a = Norm(x)
   q, k, v = Linear(a)
   q, k = apply_position_info(q, k)  # 例如 RoPE
   attn_out = CausalSelfAttention(q, k, v)
   x = x + OutputProjection(attn_out)

2. MLP 子层
   m = Norm(x)
   mlp_out = MLP(m)
   x = x + mlp_out

输出 x: [B, S, D]
```

堆叠 `L` 层后，每个 token 的向量已经多次读取上下文，并经过多次非线性变换。最后 LM Head 把 hidden state 映射到词表维度：

```text
logits = hidden @ W_vocab^T
```

推理时通常只关心最后一个位置的 logits，因为它决定下一个 token。

## Training、Prefill、Decode 的区别

Transformer 的同一套结构，在训练和推理时运行方式不同。

### 训练

训练时通常用 teacher forcing。模型一次性处理一整段 token，并在每个位置预测下一个 token。

```text
输入:  [t0, t1, t2, t3]
目标:  [t1, t2, t3, t4]
```

因为有 causal mask，所以每个位置不能看未来，但所有位置可以并行计算 loss。训练的主要显存压力来自：

- 参数。
- activation。
- gradient。
- optimizer state。
- attention 中间张量。

训练时真正被学习的是模型参数，包括 embedding、Q/K/V 投影矩阵、MLP 权重、Norm 参数和 LM Head。模型先根据当前参数给出下一个 token 的概率分布，再用目标 token 计算 loss，最后通过反向传播更新这些参数。经过大量样本后，Q/K/V 投影会逐渐学会“什么样的 token 应该匹配什么上下文”，MLP 会学会对每个位置的表示做更复杂的变换。

推理时参数不再更新。模型只是重复执行前向计算：读入 prompt，得到 logits，采样下一个 token，把新 token 接回输入，再继续。

### Prefill

在线推理时，用户先给一个 prompt。模型需要一次性处理整段 prompt，这叫 prefill。

```text
prompt: [t0, t1, ..., tS]
输出: 最后位置的 logits
同时保存每层 K/V 到 KV Cache
```

Prefill 的 Attention 会处理长度为 `S` 的序列，计算量和 `S^2` 强相关，但矩阵较大，比较容易利用 GPU 并行。

### Decode

生成第一个新 token 后，模型继续生成下一个 token。每次 decode 通常只输入一个新 token。

```text
新 token -> 生成 Q
历史 token 的 K/V 从 KV Cache 读取
Q attend to all cached K/V
输出下一个 token
```

Decode 每步计算量比 prefill 小，但要重复很多次，并且需要不断读取 KV Cache。对于长上下文和大 batch，decode 很容易变成显存带宽和调度问题。

## KV Cache 是什么

在自回归生成中，历史 token 的 K/V 不会变。没有 KV Cache 时，每生成一个新 token 都要重新计算整个历史序列的 K/V，代价很高。

KV Cache 保存每一层、每个历史 token 的 K 和 V：

```text
K cache: [L, B, H, S, Dh]
V cache: [L, B, H, S, Dh]
```

粗略显存估算：

```text
KV Cache bytes ~= 2 * L * B * S * D * bytes_per_element
```

这里的 `2` 来自 K 和 V。这个公式解释了为什么：

- 上下文越长，KV Cache 越大。
- batch 越大，KV Cache 越大。
- 层数和 hidden size 越大，KV Cache 越大。
- FP16/BF16 比 FP32 省一半，FP8/量化 KV Cache 还能继续降低压力。

## 为什么 Transformer 对 Infra 很重要

从系统角度看，Transformer 不是一个抽象模型，而是一组非常具体的工作负载。

| 模块 | 主要算子 | 常见瓶颈 | 优化方向 |
| --- | --- | --- | --- |
| Embedding | gather / lookup | 随机访存、cache miss | layout、缓存、并行读取 |
| QKV Projection | GEMM | 算力利用率 | Tensor Core、融合 QKV |
| Attention Scores | QK GEMM | `S^2` 计算和中间矩阵 | FlashAttention、tiling、mask fusion |
| Softmax | exp / reduce | 内存访问、数值稳定 | online softmax、fusion |
| Attention Output | PV GEMM | 带宽和算力 | tiling、layout |
| RoPE / Norm / Residual | elementwise | 小算子开销 | kernel fusion |
| MLP | GEMM + activation | 算力、显存带宽 | Tensor Core、SwiGLU fusion、quantization |
| LM Head | GEMM / top-k | 大词表输出 | vocab parallel、采样优化 |
| KV Cache | read/write | 显存容量和带宽 | paged attention、cache quantization、调度 |

这也是为什么后续会学习 FlashAttention、Triton、TorchInductor、Tensor Core、HBM、interconnect、batching 和 serving scheduler。

## 常见误区

### 误区 1：Transformer 只有 Attention

Attention 很关键，但 MLP、Residual、Norm、位置编码、训练目标都很重要。很多参数和计算量集中在 MLP 上。

### 误区 2：Attention 权重就是完整解释

Attention 权重能提示模型在某层某头关注了哪里，但不能直接等同于“模型为什么做出这个预测”。真正的表示会经过多层、多头、MLP 和 residual stream 混合。

### 误区 3：Encoder-Decoder 和 GPT 是同一个结构

原始 Transformer 是 Encoder-Decoder。GPT/LLaMA 这类模型通常是 Decoder-only，去掉 cross-attention，只保留 masked self-attention 和 MLP。

### 误区 4：训练和推理性能是一回事

训练通常大 batch、大矩阵、需要 backward 和 optimizer。推理分 prefill 和 decode；decode 小步迭代、依赖 KV Cache，瓶颈经常完全不同。

### 误区 5：长上下文只是多放一些 token

长上下文会同时放大 prefill 计算、attention 中间状态、KV Cache 容量和 decode 读带宽。它是模型能力问题，也是系统问题。

## 关键问题

- 当前模型是 Encoder-only、Decoder-only 还是 Encoder-Decoder。
- 当前问题发生在训练、prefill 还是 decode。
- 当前瓶颈来自 Attention、MLP、Embedding、Norm、LM Head 还是 KV Cache。
- `B`、`S`、`D`、`H`、`L`、precision 如何改变计算量、显存和带宽。
- Attention kernel 是否避免了物化完整 `[B, H, S, S]` 矩阵。
- KV Cache 的 layout 是否适合当前 batch、beam、paged allocation 和并发调度。
- 小算子是否被融合，是否存在 launch overhead。
- 当前 shape 是否能让 Tensor Core、编译器和调度器跑满。

## 推荐阅读顺序

1. 先读 [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)：用图理解 encoder、decoder、Q/K/V、多头和位置编码。
2. 再读 [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)：把论文结构对应到可运行代码。
3. 再读 [Transformers from Scratch](https://peterbloem.nl/blog/transformers)：从最小 self-attention 和 PyTorch 实现理解矩阵化。
4. 再读 [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)：理解 Decoder-only 语言模型、masked self-attention 和逐 token 生成。
5. 需要系统化教材时，读 [Dive into Deep Learning: Transformer Architecture](https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html)。
6. 想补 Attention 历史和变体时，读 [Lilian Weng: Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/)。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Vaswani et al., 2017.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/), Jay Alammar.
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html), Harvard NLP.
- [Transformers from Scratch](https://peterbloem.nl/blog/transformers), Peter Bloem.
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/), Jay Alammar.
- [Dive into Deep Learning: The Transformer Architecture](https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html).
- [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/), Lilian Weng.
