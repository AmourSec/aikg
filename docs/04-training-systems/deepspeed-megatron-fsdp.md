---
title: DeepSpeed、Megatron-LM 与 PyTorch FSDP
domain: training-systems
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-06-11
---

# DeepSpeed、Megatron-LM 与 PyTorch FSDP

DeepSpeed、Megatron-LM / Megatron Core、PyTorch FSDP 都是大模型训练系统里的重要工具。但它们不是同一种东西。

最容易犯的错误是把框架名当成架构答案：

```text
我们用 DeepSpeed
我们用 Megatron
我们用 FSDP
```

这些说法都不够。真正要回答的是：

- 参数、梯度、optimizer state 怎么切？
- 单层矩阵计算怎么切？
- layer 怎么切？
- expert 怎么切？
- 通信怎么重叠？
- checkpoint 怎么保存和恢复？
- profiler 证据如何证明配置有效？

这篇把三者放在同一个训练系统视角下比较。

## 先给结论

粗略说：

| 工具 | 更像什么 | 核心强项 |
| --- | --- | --- |
| DeepSpeed | 训练 runtime + ZeRO 生态 | ZeRO、offload、optimizer、checkpoint、易接入 |
| Megatron-LM / Megatron Core | 大模型并行训练栈 | TP/PP/CP/EP、Transformer/MoE 结构、Megatron parallel layout |
| PyTorch FSDP | PyTorch 原生 sharded data parallel | 参数/梯度/optimizer state sharding、PyTorch 集成、组合灵活 |

更具体：

- 想在 PyTorch 模型上快速获得 ZeRO、offload、mixed precision、checkpoint 和 runtime 配置，DeepSpeed 很常见。
- 想训练 GPT/Llama/MoE 这类大 Transformer，并系统使用 TP/PP/EP/CP 等并行策略，Megatron 更核心。
- 想留在 PyTorch 原生生态，用 sharded data parallel 降低显存，FSDP 是主线选择。

实际项目中它们也可能组合。例如早期常见 Megatron-LM + DeepSpeed ZeRO；现在 Megatron Core 自己也提供 distributed optimizer、FSDP、dist checkpointing 等能力。

## 三者解决的问题不完全一样

一个大模型训练系统至少有这些层：

```text
model definition
parallelism layout
distributed runtime
memory sharding
optimizer
mixed precision
kernel/fusion
data pipeline
checkpoint
profiling
launch/orchestration
```

三者覆盖范围不同：

| 层次 | DeepSpeed | Megatron-LM / Core | PyTorch FSDP |
| --- | --- | --- | --- |
| model definition | 接入已有 PyTorch 模型 | 提供 GPT/MoE/Transformer 结构和组件 | 接入已有 PyTorch 模型 |
| data parallel | 支持 | 支持 | 核心能力 |
| ZeRO / sharding | 核心能力 | distributed optimizer / FSDP 能力 | 核心能力 |
| tensor parallel | 可与 Megatron 等组合 | 核心能力 | 需要额外 DTensor/TP 组合 |
| pipeline parallel | 支持 | 核心能力 | 非核心，需要另行组合 |
| expert parallel | 支持 MoE 生态 | 核心能力之一 | 非核心，需要另行组合 |
| offload | ZeRO-Offload / ZeRO-Infinity | CPU offload 等能力 | CPU offload |
| checkpoint | DeepSpeed checkpoint | dist checkpointing | PyTorch DCP/FSDP state_dict |
| kernel/fusion | fused optimizer、kernel 生态 | Transformer/MoE/fusion 深度优化 | 依赖 PyTorch/compiler/外部 kernel |
| 易接入 | 高 | 中等到高，取决于模型是否贴近 Megatron | 高 |

所以选型不是“哪个更高级”，而是 workload 需要哪几层能力。

## DeepSpeed：训练 runtime 与 ZeRO 生态

DeepSpeed 的核心价值之一是把一批训练系统能力封装成 runtime：

- distributed training。
- mixed precision。
- gradient accumulation。
- ZeRO。
- optimizer 和 fused optimizer。
- offload。
- activation checkpointing。
- checkpoint。
- monitoring / flops profiler / communication logging。

DeepSpeed 官方文档把 ZeRO 作为核心内存优化能力：把 optimizer states、gradients、parameters 等模型状态按 data parallel rank 切分，减少数据并行中的重复显存。

### DeepSpeed 最典型的使用场景

1. 已有 PyTorch/Hugging Face 模型，希望用 ZeRO 降低显存。
2. 需要 ZeRO-1/2/3、CPU/NVMe offload。
3. 希望用配置文件控制 mixed precision、batch、optimizer、checkpoint。
4. 不想重写模型结构来适配 TP/PP。
5. 想快速让较大模型跑起来。

一个典型 DeepSpeed 配置关注：

```json
{
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 8,
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,
    "overlap_comm": true,
    "contiguous_gradients": true
  },
  "gradient_clipping": 1.0
}
```

这里真正重要的不是 JSON，而是它背后的系统语义：

- micro batch 和 global batch 如何确定。
- ZeRO stage 切哪些状态。
- 通信是否 overlap。
- optimizer state 是否 offload。
- checkpoint 是 DeepSpeed shard 还是 consolidated state。

### ZeRO stage 视角

可以把 ZeRO 简化理解为：

| Stage | 切什么 | 主要收益 |
| --- | --- | --- |
| ZeRO-1 | optimizer states | 减少 AdamW `m/v/master` 重复 |
| ZeRO-2 | optimizer states + gradients | 进一步减少 gradient 重复 |
| ZeRO-3 | optimizer states + gradients + parameters | 最大化显存节省 |

代价是通信和运行时复杂度增加。

ZeRO-3 不一定总是最快。它会在 forward/backward 周期内 all-gather 参数，通信模式更复杂。小模型、短序列或网络较弱时，ZeRO-2 甚至 ZeRO-1 可能更快。

### DeepSpeed 的边界

DeepSpeed 不是自动解决所有大模型并行问题。

如果模型单层矩阵太大，单卡算不动或放不下，只靠 ZeRO 不一定够。这时需要 Tensor Parallel。

如果层数太多，单卡放不下整段模型，可能需要 Pipeline Parallel。

如果是 MoE，大量 expert 和 token dispatch/combine 需要 Expert Parallel。

DeepSpeed 可以和这些策略组合，但你仍然要设计 parallelism layout。

## Megatron-LM / Megatron Core：大 Transformer 并行栈

Megatron-LM 最初以大规模 Transformer 训练和模型并行著名。Megatron Core 则把许多能力模块化：

- tensor parallel。
- pipeline parallel。
- context parallel。
- expert parallel。
- Transformer layers。
- MoE。
- distributed optimizer。
- dist checkpointing。
- fused kernels。
- optimizer scheduler。
- data pipeline 和 tokenization 相关组件。

Megatron 的核心不是“一个训练脚本”，而是围绕大 Transformer 结构，把并行策略嵌入模型层实现和 runtime。

### Megatron 最典型的使用场景

1. 训练 GPT/Llama/Mistral 类 dense Transformer。
2. 训练 MoE 模型。
3. 需要 TP + PP + DP 的组合。
4. 需要 EP/CP 等更细的并行策略。
5. 关心 Transformer 层内通信和 kernel/fusion。
6. 希望使用 Megatron 已有的高性能训练 recipe。

Megatron 的配置通常围绕并行度：

```text
tensor_model_parallel_size
pipeline_model_parallel_size
context_parallel_size
expert_model_parallel_size
data_parallel_size
micro_batch_size
global_batch_size
num_layers
hidden_size
num_attention_heads
seq_length
```

这些值不是孤立参数，而是训练架构。

### Tensor Parallel 是 Megatron 的核心能力之一

Tensor Parallel 把单层矩阵切到多个 GPU。

例如：

```text
MLP up projection:
  W_up split by output dimension

MLP down projection:
  W_down split by input dimension
```

这会改变每层的通信：

- Column Parallel 后可能需要后续组合。
- Row Parallel 后常见 AllReduce / ReduceScatter。
- Attention head 可以按 head 切。
- vocab parallel 可以切输出词表。

Megatron 的价值在于它不仅“切 tensor”，还把切分嵌入 Transformer 层的计算图里。

### Pipeline Parallel 处理层切分

Pipeline Parallel 把不同层放在不同 stage 上：

```text
stage 0: layers 0-7
stage 1: layers 8-15
stage 2: layers 16-23
stage 3: layers 24-31
```

Megatron 支持 pipeline schedules、micro-batch 流水和更复杂的 layout。它适合模型层数多、权重和 activation 压力大的场景。

代价：

- pipeline bubble。
- stage balance。
- P2P 通信。
- virtual pipeline / interleaving 调参。
- checkpoint 与 layer mapping 复杂。

### Expert Parallel 和 MoE

MoE 模型需要把专家分布到不同 GPU。Megatron Core 文档中 MoE 是高级能力之一，和 TP/PP/DP/CP 组合时复杂度更高。

MoE 训练关注：

- router。
- top-k。
- token dispatch/combine。
- AllToAll。
- expert load balance。
- capacity。
- token dropping。
- expert parallel size。
- grouped GEMM。

Megatron 的优势是这些机制和 Transformer/MoE 层耦合更深，适合大规模 MoE 训练。

## PyTorch FSDP：原生 Sharded Data Parallel

FSDP 是 PyTorch 原生 Fully Sharded Data Parallel。

PyTorch 文档把 FSDP 描述为跨 data parallel workers shard module parameters 的 wrapper，并说明它受 ZeRO Stage 3 思路启发。

FSDP 的核心动作：

```text
parameters are sharded outside computation
before forward: all-gather full params for current module
after forward/backward: reshard params
after backward: reduce-scatter gradients
optimizer state updated locally per rank
```

具体取决于 sharding strategy。

### FSDP 最典型的使用场景

1. 希望留在 PyTorch 原生生态。
2. 模型结构比较自定义，不想迁移到 Megatron。
3. 需要参数、梯度、optimizer state sharding。
4. 想和 PyTorch Distributed Checkpoint、DTensor、torch.compile 等生态组合。
5. 主要并行策略是 data parallel sharding，而不是深度 TP/PP。

FSDP 的基本用法是 wrapper：

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = build_model()
model = FSDP(model, auto_wrap_policy=...)
optimizer = torch.optim.AdamW(model.parameters(), lr=...)
```

注意 optimizer 要在 FSDP wrap 后初始化，因为 FSDP 会改变参数对象。

### FSDP 的关键配置

| 配置 | 含义 |
| --- | --- |
| auto_wrap_policy | 决定哪些 module 成为 FSDP unit |
| sharding_strategy | FULL_SHARD、SHARD_GRAD_OP、HYBRID_SHARD 等 |
| mixed_precision | 参数、reduce、buffer dtype |
| backward_prefetch | backward all-gather 是否提前 |
| forward_prefetch | forward all-gather 是否提前 |
| limit_all_gathers | 限制预取，防止 all-gather 造成显存过高 |
| cpu_offload | 是否把状态 offload 到 CPU |
| use_orig_params | 参数视图和 optimizer 兼容相关 |

FSDP 性能很依赖 wrap 粒度。

如果 wrap 太粗：

- all-gather 大。
- peak memory 高。
- overlap 差。

如果 wrap 太细：

- collective 太碎。
- kernel launch 和通信调度开销大。
- norm/bias 等小模块可能引入额外通信。

通常 Transformer block 是一个常见起点，但不是唯一答案。

## 三者的系统对比

### 显存节省

| 问题 | DeepSpeed | Megatron | FSDP |
| --- | --- | --- | --- |
| optimizer state 过大 | ZeRO-1/2/3 | distributed optimizer | sharded optimizer state |
| gradient 过大 | ZeRO-2/3 | distributed optimizer | reduce-scatter / shard |
| parameter 过大 | ZeRO-3 | TP/PP/FSDP | FULL_SHARD |
| activation 过大 | activation checkpointing | activation checkpointing + TP/PP/CP | activation checkpointing |
| 单层矩阵过大 | 需组合 TP | TP 核心能力 | 需组合 TP/DTensor |

### 通信模式

| 工具 | 常见通信 |
| --- | --- |
| DeepSpeed ZeRO | reduce-scatter、all-gather、all-reduce、offload IO |
| Megatron TP | layer 内 all-reduce/all-gather/reduce-scatter |
| Megatron PP | stage 间 P2P |
| Megatron EP | AllToAll |
| FSDP | module 级 all-gather、reduce-scatter |

不同通信模式对网络要求不同。

- TP 最怕跨节点小频繁通信。
- EP AllToAll 对网络和负载均衡敏感。
- FSDP/ZeRO-3 对 all-gather 时机和 overlap 敏感。
- PP 对 stage balance 和 bubble 敏感。

### 接入成本

| 场景 | 接入成本较低的选择 |
| --- | --- |
| Hugging Face 模型，想快速省显存 | DeepSpeed 或 FSDP |
| 自定义 PyTorch 模型，想保持原生 | FSDP |
| GPT/Llama 大规模预训练 | Megatron |
| MoE 大规模预训练 | Megatron Core |
| 单机/少机中等模型 | FSDP / DeepSpeed |
| 多千卡混合并行 | Megatron 体系更常见 |

## 怎么选

可以按问题选，而不是按名气选。

### 模型是否单卡能放下权重？

如果能放下，但 optimizer state 放不下：

- ZeRO-1/2。
- FSDP SHARD_GRAD_OP。
- distributed optimizer。

如果权重本身放不下：

- ZeRO-3。
- FSDP FULL_SHARD。
- TP/PP。

如果单层矩阵也太大：

- TP 更自然。
- Megatron 更适合。

### 是否需要 TP/PP/EP？

如果只需要 sharded data parallel：

- FSDP 或 DeepSpeed ZeRO。

如果需要 TP/PP：

- Megatron。
- 或 DeepSpeed + Megatron 组合。

如果需要大规模 MoE EP：

- 优先看 Megatron Core / 专门 MoE runtime。

### 是否高度自定义模型？

如果模型结构非常自定义，迁移到 Megatron 成本高：

- 先考虑 FSDP。
- 或 DeepSpeed ZeRO。

如果模型贴近 GPT/Llama/MoE 标准结构：

- Megatron 的收益更大。

### 团队维护能力如何？

Megatron 类训练栈性能强，但需要理解：

- parallel group。
- rank mapping。
- tensor slicing。
- pipeline schedule。
- MoE dispatch。
- checkpoint reshard。

如果团队还没有这些经验，从 FSDP/DeepSpeed 开始可能更现实。

但如果目标是大规模高效预训练，迟早要理解这些细节。

## 常见组合方式

### FSDP + PyTorch 原生生态

适合：

- 研究代码。
- 自定义模型。
- 中大规模 dense 模型。
- 希望减少框架依赖。

组合：

- FSDP。
- PyTorch Distributed Checkpoint。
- torch.profiler。
- torch.compile / Inductor，视模型而定。
- 自定义 Triton/FlashAttention kernels。

关注：

- auto_wrap_policy。
- sharding_strategy。
- mixed_precision。
- checkpoint state_dict 类型。
- overlap 和 all-gather peak。

### DeepSpeed ZeRO + Hugging Face

适合：

- 快速训练/微调大模型。
- 资源有限，需要 offload。
- 希望配置驱动。

组合：

- HF Trainer / Accelerate。
- DeepSpeed ZeRO-2/3。
- CPU/NVMe offload。
- DeepSpeed checkpoint。

关注：

- ZeRO stage。
- offload 带宽。
- DeepSpeed checkpoint 转换。
- parameter group。
- gradient accumulation。

### Megatron Core 预训练栈

适合：

- GPT/Llama dense 预训练。
- MoE 预训练。
- 千卡级训练。
- TP/PP/EP/CP 组合。

组合：

- Megatron Core model components。
- Tensor Parallel。
- Pipeline Parallel。
- Expert Parallel。
- Distributed optimizer。
- dist checkpointing。
- Transformer Engine / fused kernels。

关注：

- 并行度搜索。
- rank mapping。
- stage balance。
- EP load balance。
- MFU。
- checkpoint resharding。

### Megatron + DeepSpeed

这种组合历史上很常见：Megatron 提供模型并行，DeepSpeed 提供 ZeRO 和 runtime 能力。

适合：

- 想用 Megatron TP/PP。
- 同时想用 DeepSpeed ZeRO。
- 已有相关脚本和经验。

风险：

- 两套配置系统。
- checkpoint 格式复杂。
- optimizer state 和并行组语义要对齐。
- debug 难度更高。

## 框架选型矩阵

| 需求 | 推荐优先级 |
| --- | --- |
| 快速让 HF 大模型微调跑起来 | DeepSpeed / FSDP |
| 自定义 PyTorch 模型省显存 | FSDP |
| 单机多卡 7B-70B 实验 | FSDP 或 DeepSpeed |
| 大规模 dense LLM 预训练 | Megatron Core |
| 大规模 MoE 预训练 | Megatron Core |
| 需要 CPU/NVMe offload | DeepSpeed |
| 需要深度 TP/PP/EP/CP 组合 | Megatron Core |
| 希望 PyTorch 原生维护 | FSDP |
| 需要和 torch.compile 深度配合 | FSDP / PyTorch 原生路径优先 |
| 想最少改模型代码 | DeepSpeed / FSDP |

## 选型时必须做 benchmark

不要只按经验选。

至少比较：

- 能否跑通目标 batch 和 sequence length。
- peak memory。
- step time。
- tokens/s。
- MFU。
- exposed communication。
- checkpoint save/load。
- resume 正确性。
- p99 step time。
- 代码复杂度。

一个小规模 benchmark 可能这样：

```text
model: 7B dense
seq_len: 4096
global_batch_tokens: fixed
hardware: 8xH100, 2 nodes

A: FSDP FULL_SHARD
B: DeepSpeed ZeRO-2
C: DeepSpeed ZeRO-3
D: Megatron TP=2 PP=2 DP=2
```

输出：

```text
fits? y/n
step time
tokens/s
MFU
peak memory
communication exposed time
checkpoint time
code/config complexity
```

不要只比较默认配置。默认配置通常不是最优配置。

## 常见误区

### 误区一：用了 DeepSpeed 就等于训练系统优化好了

不对。DeepSpeed 提供能力，但 ZeRO stage、offload、overlap、batch、checkpoint、网络拓扑仍需要调。

### 误区二：FSDP 就是 ZeRO-3 的完全替代

FSDP 和 ZeRO-3 思路相近，但实现、API、checkpoint、wrap 粒度、overlap 行为和生态不同。不能简单等同。

### 误区三：Megatron 只是一套模型代码

Megatron 更重要的是并行训练结构。TP/PP/EP/CP、distributed optimizer、checkpoint 和 Transformer/MoE 实现共同构成训练栈。

### 误区四：最省显存的方案就是最优方案

不一定。更省显存通常意味着更多通信、更多 all-gather 或 offload IO。最终要看 time to target 和成本。

### 误区五：框架可以替代系统理解

不行。框架隐藏了一些复杂性，但瓶颈仍然来自显存、通信、计算、数据、checkpoint 和故障恢复。

## 设计检查清单

选型时逐项确认：

- 模型类型是 dense、MoE 还是 multimodal？
- 参数量、层数、hidden size、head 数、sequence length 是多少？
- 单层矩阵是否需要 TP？
- 总层数是否需要 PP？
- 是否需要 EP/CP？
- global batch tokens 是否固定？
- optimizer state 是否是主要显存瓶颈？
- activation 是否是主要显存瓶颈？
- checkpoint 是否需要跨并行度恢复？
- 团队是否能维护 Megatron 级别并行配置？
- 是否需要 Hugging Face 生态兼容？
- 是否需要 PyTorch 原生 API？
- benchmark 是否比较了至少两个候选？
- profiler 是否证明瓶颈和选择匹配？

## 小结

DeepSpeed、Megatron-LM / Megatron Core、PyTorch FSDP 都很重要，但它们回答的问题不同。

关键结论：

- DeepSpeed 强在 ZeRO、offload、runtime 配置和快速接入。
- Megatron 强在大 Transformer/MoE 的 TP/PP/EP/CP 组合和高性能训练栈。
- FSDP 强在 PyTorch 原生 sharded data parallel 和自定义模型集成。
- 三者可以组合，但组合会增加 checkpoint、配置和 debug 复杂度。
- 选型必须用 workload、硬件、并行需求和 benchmark 证据决定。

框架不是训练架构本身。真正的架构是模型、并行策略、显存策略、通信策略、checkpoint 策略和可观测性共同组成的系统。

## 参考资料

- [DeepSpeed: Training Overview and Features](https://www.deepspeed.ai/training/)
- [DeepSpeed: ZeRO Tutorial](https://www.deepspeed.ai/tutorials/zero/)
- [Megatron Core User Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/index.html)
- [Megatron Core: Tensor Parallel package](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/tensor_parallel.html)
- [Megatron Core: Pipeline Parallel package](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/pipeline_parallel.html)
- [Megatron Core: Mixture of Experts package](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/moe.html)
- [PyTorch: FullyShardedDataParallel](https://docs.pytorch.org/docs/2.12/fsdp.html)
