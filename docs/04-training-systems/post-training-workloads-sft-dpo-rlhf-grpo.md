---
title: 后训练工作负载：SFT、DPO、RLHF 与 GRPO 系统视角
domain: training-systems
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-06-12
---

# 后训练工作负载：SFT、DPO、RLHF 与 GRPO 系统视角

后训练，英文常叫 post-training，指基础模型预训练完成之后，用指令数据、偏好数据、奖励模型或在线采样继续调整模型行为的一组训练流程。

这篇不讨论“怎样让模型更聪明”或“哪种对齐算法效果最好”。这里关心的是系统问题：

> SFT、DPO、RLHF、GRPO 会把普通语言模型训练变成不同的数据流和计算流；它们改变 GPU 显存、推理-训练耦合、batch 形态、checkpoint、调度和 benchmark 方法。

如果只把后训练理解成“再训练几轮”，很容易低估系统复杂度。SFT 比较接近普通监督训练；DPO 会引入 chosen/rejected 成对样本和 reference logprob；RLHF/PPO 会把在线推理 rollout、reward model、reference model、value model 和 policy update 串起来；GRPO 又用同一 prompt 的多组回答构造相对优势，减少某些 value/critic 相关成本，但增加 group generation 和样本组织压力。

## 先把后训练放到训练生命周期里

大模型训练可以粗略分成：

```text
pretraining
-> post-training
-> evaluation
-> serving
-> feedback / data collection
```

预训练更像大规模 next-token prediction：数据量极大，训练时间长，step 结构稳定，系统重点是长期吞吐、并行扩展、checkpoint 和稳定性。

后训练更像一组短到中等长度的实验流水线：数据规模相对小，版本多，评估频繁，训练方法变化快，可能还会把推理服务式的 rollout 放进训练循环。

```mermaid
flowchart TB
    A["预训练基础模型"] --> B["SFT<br/>指令监督微调"]
    B --> C["偏好数据 / 人类或模型反馈"]
    C --> D{"后训练方法"}
    D --> E["DPO / IPO / KTO 等<br/>离线偏好优化"]
    D --> F["RLHF / PPO<br/>在线 rollout + reward"]
    D --> G["GRPO<br/>group rollout + relative advantage"]
    E --> H["评估与部署"]
    F --> H
    G --> H
```

从系统视角看，后训练不是单一算法，而是一组 workload family。它们对硬件和平台的要求不同。

## 后训练的两个基本维度：离线训练与在线采样

理解后训练系统，先不要急着记 SFT、DPO、PPO、GRPO 的公式。更实用的分法是看两个维度。

第一个维度：训练信号来自哪里。

| 训练信号 | 例子 | 系统含义 |
| --- | --- | --- |
| 人工或模型写好的目标回答 | SFT | 数据集比较稳定，训练像普通 supervised learning。 |
| chosen/rejected 偏好对 | reward model、DPO | 一个 prompt 对应多条 response，logprob 和 pairwise 数据更重要。 |
| rollout 后由 reward/verifier 打分 | RLHF/PPO、GRPO | 训练过程需要在线生成，推理系统进入训练闭环。 |

第二个维度：是否需要在训练过程中生成新样本。

| 类型 | 是否在线生成 | 典型方法 | 系统特点 |
| --- | --- | --- | --- |
| 离线后训练 | 否 | SFT、reward model、DPO | 主要是 DataLoader、forward/backward、logprob 和 checkpoint。 |
| 在线后训练 | 是 | PPO/RLHF、GRPO | 需要 rollout engine、reward/verifier、样本队列、policy version 管理。 |

这两个维度比算法名字更能解释系统成本。DPO 和 SFT 都是离线训练，但 DPO 的样本是成对 response，训练时要处理 reference logprob；PPO 和 GRPO 都是在线训练，但 PPO 常有 value/critic 路径，GRPO 则把同一 prompt 的多条回答组织成 group。

一句话：

> 离线后训练像“更复杂的训练数据”；在线后训练像“推理服务和训练服务互相喂数据”。

## 几种后训练方式的系统差异

| 方法 | 数据形态 | 是否在线生成 | 额外模型 | 主要系统压力 |
| --- | --- | --- | --- | --- |
| SFT | prompt + target response | 通常否 | 无或少量 reference | 普通 forward/backward、数据 packing、loss mask。 |
| Reward Model 训练 | prompt + 多个回答 + 偏好排序 | 通常否 | reward model | pairwise/ranking loss、长回答编码、数据质量。 |
| DPO | prompt + chosen/rejected | 否 | reference model 或缓存的 reference logprob | 成对样本、双 response forward、reference logprob、显存和吞吐。 |
| RLHF/PPO | prompt + policy rollout + reward | 是 | policy、reference、reward、value/critic | rollout 推理、KV Cache、reward scoring、PPO update、推理训练切换。 |
| GRPO | prompt + group responses | 是 | policy、reference、reward/verifier | 同 prompt 多回答采样、group advantage、rollout 吞吐、样本分组。 |

这张表的重点不是算法优劣，而是系统成本来源不同：

- SFT 最像普通训练。
- DPO 把一个样本扩展成 chosen/rejected 两条 response。
- PPO/RLHF 把推理服务的生成阶段塞进训练循环。
- GRPO 减少或避免某些 value model 训练路径，但会增加同 prompt 多样本生成和分组管理。

## SFT：最接近普通监督训练

SFT 是 Supervised Fine-Tuning。它用人工或模型生成的高质量指令回答，让模型学习“给定 prompt 后应该输出什么”。

典型样本是：

```text
prompt: 用户问题 / 系统指令 / 上下文
response: 目标回答
```

训练时通常把 prompt 和 response 拼成一段 token 序列，但 loss 只打在 response 部分：

```text
[prompt tokens][response tokens]
 loss mask: 0 0 0 ... 1 1 1
```

SFT 的系统形态接近普通 causal language modeling：

1. 读入样本。
2. tokenize。
3. 按长度 packing 或 padding。
4. forward。
5. 只在 response token 上算 loss。
6. backward。
7. optimizer step。

它的瓶颈通常来自：

- sequence length 分布不均。
- prompt 很长但 loss token 很少。
- padding 浪费。
- 多轮对话模板导致 token 数膨胀。
- 小数据集频繁 eval 和 checkpoint。
- LoRA/QLoRA 配置与全量微调的取舍。

### SFT 的有效 token

SFT benchmark 不能只看总 tokens/s。因为 prompt token 也参与 forward，但通常不参与 loss。

建议区分：

```text
input tokens: prompt + response + padding
loss tokens: 真正参与 loss 的 response tokens
effective tokens: 去掉 padding 后的真实 tokens
```

如果一个 batch 里 prompt 很长、response 很短，GPU 计算了大量 prompt token，但训练信号只来自少量 response token。系统吞吐看起来不低，训练效率可能并不高。

### SFT 的 chat template 和 loss mask

SFT 里经常有多轮对话格式，例如：

```text
system: ...
user: ...
assistant: ...
user: ...
assistant: ...
```

这些文本不会直接进入模型。它们会先经过 chat template 变成具体 token 序列。模板会决定：

- system prompt 是否存在。
- user / assistant 的分隔符。
- 特殊 token 放在哪里。
- 多轮对话如何拼接。
- 哪些位置参与 loss。

这会影响系统层面的三个问题。

第一，token 数会变化。同样的文本，模板越复杂，prompt token 越多，forward 成本越高。

第二，loss mask 会变化。有些训练只对 assistant 回答打 loss；有些训练可能对某些工具调用、思维过程或结构化字段打 loss。loss mask 一变，有效训练 token 就变了。

第三，DPO/RLHF 的 logprob 也依赖同样模板。如果 SFT 和 DPO 使用不同模板，后面比较 logprob 时就不是同一个 token 序列。

所以 SFT 数据集版本至少要记录：

```text
raw text
chat template revision
tokenizer revision
max length / truncation rule
loss mask rule
packing rule
```

不要只保存一个 JSONL 文本文件。对训练系统来说，模板和 mask 也是数据的一部分。

### SFT Packing 的收益和风险

SFT 数据经常长短差异很大。如果每个样本单独 padding 到同一长度，会浪费大量计算。Packing 的做法是把多个短样本拼到同一个序列里，提高 token 利用率。

```text
sample A + EOS + sample B + EOS + sample C
```

收益：

- padding token 变少。
- tokens/s 更接近真实计算利用率。
- 小样本数据集吞吐更好。

风险：

- attention mask 必须避免样本之间错误互相看见。
- loss mask 必须正确分段。
- position id / RoPE 处理要符合训练设计。
- 如果后续要做 DPO/RLHF，packing 后的样本边界更难追踪。

因此 SFT benchmark 最好同时报告：

| 指标 | 说明 |
| --- | --- |
| raw input tokens/s | 包含 padding 的输入吞吐。 |
| non-padding tokens/s | 去掉 padding 后的真实 token 吞吐。 |
| loss tokens/s | 真正参与 loss 的 token 吞吐。 |
| packing ratio | packing 后有效 token 占比。 |

如果只报告 raw tokens/s，packing 前后的对比可能会失真。

## Reward Model 训练

RLHF 流程里常有 reward model。它学习判断哪个回答更好。

常见数据形态是：

```text
prompt
chosen response
rejected response
```

reward model 对 chosen 和 rejected 分别打分，再用 pairwise loss 让 chosen 分数高于 rejected。

系统上，它和普通 SFT 不同：

- 一个样本包含两条或多条 response。
- chosen/rejected 长度可能不同。
- 可以把 prompt 共享，但实现上常常还是分别编码两条完整序列。
- 如果 response 很长，batch padding 浪费明显。
- reward model 本身也可能是大模型，需要独立训练和 checkpoint。

Reward model 训练不是所有后训练方法都必须有。例如 DPO 直接用偏好对优化 policy，不需要显式训练一个 reward model；但 RLHF/PPO 类流程通常会用 reward model 或 verifier 给 rollout 打分。

## DPO：离线偏好优化

DPO 是 Direct Preference Optimization。它的工程直觉是：

> 不再先训练 reward model、再用 PPO 做在线强化学习，而是直接用 chosen/rejected 偏好对优化 policy。

典型输入是：

```text
prompt
chosen response
rejected response
```

训练时要比较当前 policy 对 chosen/rejected 的 log probability，同时通常还需要 reference model 对 chosen/rejected 的 log probability 作为基准。

简化数据流：

```mermaid
flowchart TB
    A["偏好样本<br/>prompt + chosen + rejected"] --> B["Policy model forward<br/>chosen logprob"]
    A --> C["Policy model forward<br/>rejected logprob"]
    A --> D["Reference logprob<br/>在线计算或预先缓存"]
    B --> E["DPO loss"]
    C --> E
    D --> E
    E --> F["Backward 更新 policy"]
```

DPO 的系统特点：

- 一个样本至少有 chosen 和 rejected 两条 response。
- policy 需要对两条 response 计算 logprob。
- reference logprob 可以在线算，也可以离线缓存。
- 如果在线跑 reference model，计算和显存成本会明显增加。
- 如果缓存 reference logprob，数据管线和版本一致性更重要。

### Reference logprob 的两种做法

第一种是在线计算 reference logprob：

```text
每个 batch:
  policy forward chosen/rejected
  reference forward chosen/rejected
  compute DPO loss
  backward policy
```

优点：

- 实现语义清楚。
- 不需要提前生成缓存。
- reference model 和 tokenizer 直接来自当前环境。

代价：

- 额外 forward 成本。
- 显存可能同时容纳 policy 和 reference。
- 多机分布式时 reference 的并行策略也要管理。

第二种是预先缓存 reference logprob：

```text
离线:
  reference model -> logprob cache

训练:
  读取 prompt/chosen/rejected + cached reference logprob
  policy forward
  compute DPO loss
```

优点：

- 训练时少跑 reference。
- 可以降低训练 step time 和显存。

代价：

- cache 必须和 base model、tokenizer、chat template、sequence truncation 完全一致。
- 数据版本管理更复杂。
- 一旦模板或 tokenizer 改了，cache 可能全部失效。

从系统角度看，DPO 的关键问题是：reference 成本放在线上训练循环里，还是提前转成数据 artifact。

### DPO 的 logprob 计算路径

DPO 的系统成本集中在 logprob。对每个样本，至少要拿到：

```text
policy_logprob(chosen)
policy_logprob(rejected)
reference_logprob(chosen)
reference_logprob(rejected)
```

这些 logprob 不是只取最后一个 token，而是通常要对 response token 的 token-level logprob 求和或平均。简化看：

```text
prompt + chosen -> token logprob -> chosen sequence logprob
prompt + rejected -> token logprob -> rejected sequence logprob
```

这带来几个工程影响：

- chosen 和 rejected 长度不同，batch padding 更复杂。
- 一个训练样本会变成两条 sequence 的 forward。
- 如果 online reference，reference 也要跑两条 sequence。
- 如果 cache reference，cache 必须和 tokenizer/template/truncation 对齐。
- logprob 需要从 `[tokens, vocab]` logits 中 gather target token，输出层和 loss 路径会被频繁调用。

一个 DPO batch 的粗略计算量可以这样理解：

```text
policy forward: chosen + rejected
reference forward: chosen + rejected   # 如果没有缓存
policy backward: chosen + rejected
```

如果 reference logprob 已缓存，则训练时可以省掉 reference forward：

```text
policy forward/backward: chosen + rejected
read cached reference logprob
```

所以 DPO 性能优化经常不是先改 optimizer，而是先决定 reference logprob 的位置。

### DPO 数据缓存的版本边界

缓存 reference logprob 能省训练成本，但它也把一部分模型计算结果变成了数据 artifact。缓存必须绑定这些信息：

| 字段 | 原因 |
| --- | --- |
| reference model id/revision | 换 reference 后 logprob 全变。 |
| tokenizer revision | token 切分变了，logprob 无法复用。 |
| chat template revision | prompt/response 拼接变了。 |
| truncation rule | 被截断 token 变了。 |
| response mask | 哪些 token 计入 logprob 变了。 |
| dtype / numerical policy | 极端情况下会影响可比性。 |

如果这些字段没记录，后面看到一个 `ref_logprob.npy` 或 parquet 列，很难判断它能不能复用。

推荐把 reference cache 当成一等数据集版本：

```yaml
reference_cache:
  model_id: ""
  model_revision: ""
  tokenizer_revision: ""
  chat_template_revision: ""
  truncation_rule: ""
  response_mask_rule: ""
  created_by_code_commit: ""
  checksum: ""
```

DPO 的很多“训练异常”其实不是优化器问题，而是 chosen/rejected、template、reference cache 不一致。

## RLHF / PPO：把推理放进训练循环

RLHF 的经典流程通常包括三段：

1. SFT 训练初始 policy。
2. 用人类偏好数据训练 reward model。
3. 用 PPO 之类的强化学习方法，让 policy 生成回答，再根据 reward 更新 policy。

PPO 训练循环和普通 supervised training 很不一样。它需要先用当前 policy 生成 rollout：

```text
prompt -> policy.generate() -> response
```

然后对 response 打分：

```text
reward model(prompt, response) -> reward
reference model(prompt, response) -> KL / reference logprob
value model(prompt, response) -> value estimate
```

再用这些结果构造 PPO loss 更新 policy。

```mermaid
flowchart TB
    A["Prompt batch"] --> B["Policy rollout<br/>生成 responses"]
    B --> C["Reward / verifier scoring"]
    B --> D["Reference model logprob<br/>KL 约束"]
    B --> E["Value / critic<br/>估计 value"]
    C --> F["Advantage / return"]
    D --> F
    E --> F
    F --> G["PPO minibatch update"]
    G --> H["更新 policy"]
    H --> A
```

这会带来一个重要变化：

> RLHF/PPO 的训练循环里有大量推理生成，而不只是 forward/backward。

### PPO 系统成本来自哪里

PPO 类后训练通常同时涉及：

- policy model：被训练的模型。
- reference model：用于 KL 约束，通常冻结。
- reward model：给生成结果打分。
- value model / critic：估计 value 或 advantage，某些实现和 policy 共享 backbone。
- rollout engine：负责生成 response。
- training engine：负责 PPO update。

系统成本包括：

| 成本 | 说明 |
| --- | --- |
| Rollout 推理 | Prefill/Decode、KV Cache、采样、变长输出。 |
| 多模型显存 | policy、reference、reward、value 可能同时占资源。 |
| 数据回流 | rollout 结果要进入训练 batch。 |
| 旧策略 logprob | PPO 需要记录 rollout 时 policy 的 logprob。 |
| KL 计算 | 需要 reference logprob。 |
| 多 epoch update | 同一批 rollout 可能被切成多个 minibatch 更新。 |
| 同步与版本 | rollout policy 和 training policy 版本不能混乱。 |

这就是为什么 RLHF/PPO 工程上常比 SFT/DPO 难很多。它不是一个普通 `DataLoader -> forward -> backward` 循环，而是一个“推理系统 + 训练系统 + 数据流水线”的组合。

## Rollout 是 RLHF 的系统核心

Rollout 看起来像推理服务：

- 输入 prompt。
- 运行 prefill。
- 逐 token decode。
- 做 sampling。
- 保存生成 token、logprob、attention mask 等。

但它又不是普通在线推理服务：

- 它的目标不是低用户延迟，而是高训练吞吐。
- 输出要回流到训练。
- 需要记录训练所需的 logprob、mask、reward、value。
- prompt 通常来自训练数据队列，不是随机用户请求。
- policy 会持续更新，engine 权重版本要跟上。

因此 rollout engine 要关注：

- 每秒生成 token 数。
- prompt 和 response 长度分布。
- KV Cache 显存。
- batch/continuous batching。
- 采样参数。
- policy 权重更新频率。
- rollout 与训练之间的队列积压。

如果 rollout 慢，训练 GPU 会等数据。如果训练慢，rollout 结果会堆积，甚至因为 policy 版本过旧而不能使用。

### Rollout 和 Update 的版本一致性

在线后训练最容易出错的地方是 policy 版本。Rollout 使用某个时刻的 policy 生成样本，training update 又会持续改变 policy。

如果系统没有记录 rollout policy version，就会出现：

```text
policy_v10 生成 response
policy_v20 训练时误以为这是当前 policy 样本
```

这会影响 PPO/GRPO 的训练语义。至少要记录：

| 对象 | 应记录字段 |
| --- | --- |
| Rollout sample | prompt id、response tokens、sampling config、policy version、生成时间。 |
| Logprob | rollout-time policy logprob、reference logprob、mask。 |
| Reward | reward model/verifier version、score、失败原因。 |
| Training batch | 使用哪些 rollout sample、是否过期、是否重复使用。 |

分离式架构里通常需要设置样本过期规则：

```text
max_policy_lag_steps
max_queue_wait_time
max_reuse_epochs
```

这不是纯算法细节，而是系统稳定性问题。样本太新，rollout 和 training 耦合太紧；样本太旧，训练信号可能不匹配当前 policy。

### Rollout Engine 和普通推理服务的差异

Rollout engine 可以借用推理服务技术，但目标不同。

| 项目 | 在线推理服务 | 后训练 rollout |
| --- | --- | --- |
| 主要目标 | 用户延迟、SLA、吞吐 | 训练样本生成吞吐和版本一致性。 |
| 请求来源 | 用户流量 | prompt 数据集或任务队列。 |
| 输出 | 返回给用户 | 回流到训练。 |
| 需要保存 | 最终文本即可 | tokens、logprob、mask、sampling metadata。 |
| 权重版本 | 相对稳定 | policy 持续更新。 |
| cache 使用 | 关注用户命中率 | 关注 prompt 分布和批量生成效率。 |

因此 rollout engine 的指标也不同：

- generated tokens/s。
- prompts/s。
- average response length。
- p95 / p99 response length。
- KV Cache 使用率。
- policy reload time。
- stale sample ratio。
- reward scoring backlog。

如果只用普通推理服务的 TTFT/TPOT 指标，无法完整解释在线后训练效率。

## GRPO：Group Relative Policy Optimization

GRPO 常见于近年的 reasoning / math 后训练工作。它的核心直觉是：对同一个 prompt 生成多条回答，用同组回答之间的相对得分估计优势，而不是依赖一个单独 value model 给每个 token 学 value。

简化流程：

```text
同一个 prompt
-> 生成 G 条 responses
-> 对每条 response 打 reward / verifier score
-> 在同组内归一化或比较 reward
-> 得到 relative advantage
-> 更新 policy
```

用图表示：

```mermaid
flowchart TB
    A["Prompt"] --> B["Policy 采样 G 条 responses"]
    B --> C["Reward / verifier scoring"]
    C --> D["组内比较<br/>relative advantage"]
    D --> E["Policy update<br/>带 KL 或 reference 约束"]
```

从系统角度看，GRPO 的收益和代价都很清楚。

收益：

- 可以减少或避免单独 value model / critic 的训练成本。
- advantage 来自同组 response 的相对质量，适合有明确 verifier/reward 的任务。
- 训练链路可能比 PPO 少一个复杂模型组件。

代价：

- 每个 prompt 要生成多条 response。
- rollout token 数按 group size 放大。
- batch 组织要保留 prompt group 边界。
- reward/verifier scoring 可能成为瓶颈。
- 长 reasoning response 会让 rollout KV Cache 和存储压力增加。

如果 group size 是 `G`，每个 prompt 平均生成 `L` 个 token，那么 rollout 生成 token 数大致是：

```text
num_prompts * G * L
```

因此 GRPO 减少 value model 训练成本，不等于整体一定更便宜。它把成本更多放到了多样本生成、reward/verifier 和样本分组上。

### GRPO 的 batch 组织

GRPO 的关键不是“多生成几条回答”这么简单，而是要保留 group 结构。

普通 batch 可能是：

```text
prompt_1 -> response_1
prompt_2 -> response_2
prompt_3 -> response_3
```

GRPO batch 更像：

```text
prompt_1 -> response_1_1, response_1_2, ..., response_1_G
prompt_2 -> response_2_1, response_2_2, ..., response_2_G
```

优势估计要在同一个 prompt 的 group 内做比较。如果数据管线打乱了 group 边界，训练语义就会错。

系统上要记录：

- `prompt_id`
- `group_id`
- `sample_index_in_group`
- reward / verifier score
- group mean / std 或相对 rank
- response length
- policy version

这会影响 batching：

- 同组 response 长度差异可能很大。
- padding 浪费会随着 group size 放大。
- verifier 需要按 response 批量打分，但结果要回填到 group。
- 训练 update 要能从 group 结构计算 advantage。

所以 GRPO 的数据对象不只是 `(prompt, response)`，而是：

```text
prompt -> group[responses, rewards, logprobs, masks, metadata]
```

### GRPO 与 Verifier 的关系

GRPO 常用于数学、代码、推理等可以用规则或模型打分的任务。在这些场景里，reward/verifier 可能是：

- exact match。
- 单元测试。
- 规则检查器。
- 另一个模型 judge。
- reward model。

这让 verifier 成为系统关键组件。它决定了训练信号，也可能决定吞吐。

常见瓶颈包括：

- 单元测试运行慢。
- sandbox 隔离成本高。
- 模型 judge 推理成本高。
- verifier timeout。
- 部分样本无法解析。
- reward 分布过于稀疏，很多 group 全 0 或全 1。

如果 group 内所有 response 得分都一样，相对优势信号会变弱。系统监控要看：

| 指标 | 为什么重要 |
| --- | --- |
| group reward variance | 判断同组是否有可学习差异。 |
| all-zero group ratio | verifier 太严格或模型太弱时常见。 |
| all-one group ratio | 任务太简单或 verifier 太宽松。 |
| verifier latency | 可能成为端到端瓶颈。 |
| invalid response ratio | 影响有效训练样本。 |

因此 GRPO 平台不是只扩 rollout GPU。很多时候要先扩 verifier，或优化 verifier 的批处理与缓存。

## 为什么后训练会影响集群调度

普通大规模预训练通常是稳定长作业。后训练平台则更像混合 workload：

- SFT：中短训练作业。
- DPO：偏好数据训练，可能需要 reference cache。
- RLHF/PPO：rollout + training 双系统。
- GRPO：高 rollout token 生成量 + reward/verifier。
- Eval：频繁、短、指标多。
- 数据处理：去重、过滤、模板化、采样。

调度系统要面对：

- 训练 GPU 和 rollout GPU 是否共池。
- 推理引擎是否常驻。
- reward model / verifier 是否独立部署。
- rollout 任务和 policy update 是否需要 gang scheduling。
- 短任务是否被大训练任务挤压。
- adapter 微调和 RL 后训练是否共享基础模型缓存。
- eval 是否独占资源或低优先级运行。

在后训练平台里，只看 GPU utilization 很容易误判。一个 PPO 任务可能训练 GPU 利用率不高，但瓶颈在 rollout；一个 GRPO 任务可能 rollout GPU 很忙，但训练 update 很短。

## 后训练平台的端到端数据流

把后训练平台画成一条链路，会发现它比普通训练多了很多中间 artifact。

```mermaid
flowchart TB
    A["Prompt / Preference Dataset"] --> B["Template + Tokenization"]
    B --> C{"训练方法"}
    C --> D["SFT / DPO<br/>离线训练 batch"]
    C --> E["Rollout Queue<br/>在线生成请求"]
    E --> F["Rollout Engine<br/>policy generate"]
    F --> G["Reward / Verifier Scoring"]
    G --> H["Rollout Artifact<br/>tokens, logprobs, rewards, metadata"]
    D --> I["Policy Update"]
    H --> I
    I --> J["Checkpoint / Adapter"]
    J --> K["Eval"]
    K --> L["Model Registry / Serving Candidate"]
```

这条链路里，每个节点都可能变成瓶颈：

| 节点 | 常见瓶颈 |
| --- | --- |
| Template + Tokenization | CPU 处理慢、模板版本不一致、loss mask 错。 |
| Rollout Queue | 训练等 rollout，或 rollout 积压过期。 |
| Rollout Engine | KV Cache、decode 吞吐、policy reload。 |
| Reward / Verifier | 打分慢、timeout、reward 分布异常。 |
| Policy Update | backward、optimizer、显存、分布式同步。 |
| Eval | 评估频繁、生成式 eval 慢、资源抢占。 |
| Registry | checkpoint、adapter、reference cache、eval report 版本不清。 |

后训练系统的难点不只是“某个环节快”，而是这些环节速率要匹配。否则就会出现：

- rollout 生成太慢，training GPU 等数据。
- rollout 生成太快，样本队列堆积并过期。
- verifier 太慢，rollout 结果无法进入 update。
- eval 太频繁，训练主循环被打断。
- checkpoint 太慢，短任务周转时间被存储拖垮。

## 资源池怎么拆

后训练可以把资源拆成几个池：

| 资源池 | 承担任务 | 典型优化目标 |
| --- | --- | --- |
| Training pool | SFT/DPO/PPO/GRPO policy update | backward/update 吞吐、显存、稳定性。 |
| Rollout pool | policy generate | decode tokens/s、KV Cache、policy reload。 |
| Reward/verifier pool | reward model、规则检查、单元测试 | scoring throughput、timeout、隔离。 |
| Eval pool | validation、benchmark、生成式评估 | 稳定可比、低干扰。 |
| Data pool | tokenize、packing、reference cache | 数据吞吐、版本治理。 |

小规模实验可以共用一个 GPU pool。大规模后训练则通常需要拆池，因为不同环节的硬件需求不同：

- training 需要高显存和高带宽，关注 backward 和 optimizer。
- rollout 更像推理服务，关注 KV Cache、decode 和 batching。
- verifier 可能是 CPU-heavy、GPU-heavy 或 sandbox-heavy。
- eval 要稳定，不应该被训练主循环频繁打断。

拆池之后，调度问题会变复杂。要监控每个池的利用率和队列长度，而不是只看总体 GPU utilization。

## 后训练里的多模型驻留

RLHF/PPO/GRPO 经常同时出现多个模型角色：

| 角色 | 是否训练 | 作用 |
| --- | --- | --- |
| Policy | 是 | 被优化的模型。 |
| Reference | 否 | 提供 KL / reference logprob 约束。 |
| Reward model | 通常否，可能单独训练 | 对 response 打分。 |
| Value / critic | PPO 中可能训练 | 估计 value / advantage。 |
| Verifier | 通常否 | 规则、测试或模型 judge。 |

这些角色可以有不同部署方式：

| 部署方式 | 优点 | 代价 |
| --- | --- | --- |
| 全部共置在训练进程 | 简单，数据传输少 | 显存压力大，模型互相干扰。 |
| reference/reward 独立服务化 | training 进程轻 | RPC、版本一致性、服务可用性。 |
| rollout 独立推理引擎 | 可用 vLLM/SGLang 等优化 decode | 权重同步、队列和过期样本复杂。 |
| reward/verifier 批处理服务 | scoring 吞吐高 | 需要请求聚合、timeout 和结果回填。 |

选择时要看瓶颈：

- 如果显存放不下多个模型，先拆 reference/reward。
- 如果 rollout decode 是瓶颈，拆 rollout pool。
- 如果 verifier 慢，拆 verifier pool 并做批处理。
- 如果实验很小，过度拆分会让调试和版本管理更难。

## Online 后训练的队列设计

在线后训练通常至少有两个队列：

```text
prompt queue -> rollout engine -> rollout result queue -> training update
```

如果有 reward/verifier，还会多一段：

```text
rollout result queue -> scoring queue -> scored sample queue
```

每个队列都要有元数据：

```yaml
sample_id: ""
prompt_id: ""
policy_version: ""
sampling_config_hash: ""
created_at: ""
expires_at: ""
status: "generated"  # generated / scored / consumed / expired / failed
retry_count: 0
```

队列不是越长越好。队列太短，训练可能断粮；队列太长，样本可能 stale。常见控制策略：

- 限制最大队列长度。
- 限制最大 policy lag。
- 对过期样本直接丢弃或降权。
- 按 prompt 难度或长度做采样。
- 对 verifier 失败样本单独统计。
- 记录每条样本是否被训练消费。

如果系统没有 sample-level lineage，后面很难解释一次训练为什么变好或变坏。

## 后训练的数据版本更重要

后训练数据通常比预训练数据更“结构化”，也更容易因为细节变化影响结果。

需要记录：

| 数据对象 | 为什么重要 |
| --- | --- |
| Prompt | 生成和训练的输入。 |
| Chosen/rejected response | DPO、reward model 的核心偏好数据。 |
| Reward / verifier score | RLHF/GRPO 的训练信号。 |
| Chat template | 改变 token 序列和 loss mask。 |
| Tokenizer revision | 改变 token boundary 和 logprob。 |
| Truncation 规则 | 决定哪些 token 参与训练。 |
| Reference logprob cache | 必须和 reference model、模板、tokenizer 对齐。 |
| Policy version | rollout 时使用的模型版本。 |

尤其要警惕：看起来一样的文本，在不同 chat template 下可能变成不同 token 序列。DPO/RLHF 里的 logprob 对 token 序列非常敏感，不能只记录原始文本。

## 显存与吞吐怎么估算

后训练的显存不能只按一个模型估算。

SFT 的显存类似普通训练：

```text
policy weights
+ gradients
+ optimizer states
+ activations
+ temporary buffers
```

DPO 可能增加：

```text
chosen/rejected activations
+ reference model weights or reference cache
+ longer batch sequence packing
```

RLHF/PPO 可能涉及：

```text
policy training memory
+ rollout policy serving memory
+ reference model memory
+ reward model memory
+ value model memory
+ KV Cache
+ rollout storage buffers
```

GRPO 可能把 rollout token 数放大：

```text
prompt_count * group_size * response_length
```

所以容量评估要分开看：

- training step time。
- rollout tokens/s。
- reward scoring tokens/s。
- update tokens/s。
- end-to-end samples/s。
- 每轮迭代 wall-clock time。

单独报告 policy update 的 tokens/s 对 RLHF/GRPO 不够，因为真正瓶颈常常在生成和打分。

## 后训练 checkpoint 与 artifact

后训练 checkpoint 不只是 policy 权重。不同方法需要保存的状态不同。

SFT/DPO 至少保存：

```text
policy weights or adapter
optimizer state
scheduler state
global step / consumed tokens
dataset position
tokenizer/template revision
training config
eval report
```

DPO 还要保存或引用：

```text
reference model revision
reference logprob cache version
chosen/rejected data revision
```

PPO/GRPO 还要保存或记录：

```text
rollout policy version
rollout queue state
reward/verifier version
sampling config
KL/reference config
advantage normalization config
```

如果后训练使用 LoRA/QLoRA，还要把 adapter 纳入模型版本：

```text
base_model_revision
adapter_revision
adapter_config
merge/unmerge state
serving compatibility
```

一个可追踪的后训练 artifact registry 至少回答：

- 当前 policy 来自哪个 base model。
- 是否经过 SFT、DPO、RLHF/GRPO，每一步使用什么数据。
- reference model 是哪个版本。
- reward/verifier 是哪个版本。
- rollout 使用了什么 sampling 配置。
- eval 用了哪些 benchmark 和 decoding 配置。
- 这个 checkpoint 是否可 resume，还是只能部署。

后训练实验数量通常很多，单个实验时间可能不长。没有 artifact 规范时，后面会留下大量无法解释的 adapter、checkpoint 和 eval 报告。

## 分离部署还是共置部署

后训练系统常见两种架构。

第一种是共置：

```text
同一批 GPU 同时承担 rollout 和 training
```

优点：

- 部署简单。
- 数据传输路径短。
- 小规模实验容易跑。

代价：

- rollout 和 training 资源互相干扰。
- 显存要同时考虑推理 KV Cache 和训练 activation。
- 权重更新和推理 engine reload 容易影响效率。

第二种是分离：

```text
rollout GPU pool -> 生成样本队列 -> training GPU pool
reward/verifier pool -> 打分
```

优点：

- 每类 workload 可以独立扩缩。
- rollout 可以使用 serving engine 优化。
- training GPU 更专注于 backward/update。

代价：

- 队列和数据传输复杂。
- policy 版本一致性要管理。
- rollout 过期样本要处理。
- 监控和调度更复杂。

选型原则：

| 场景 | 更适合 |
| --- | --- |
| 小模型、小实验、低并发 | 共置。 |
| 大规模 RLHF/GRPO、rollout 很重 | 分离。 |
| reward/verifier 很慢 | 独立 scoring pool。 |
| policy 更新频繁、样本易过期 | 更紧耦合或短队列。 |
| 需要复用高性能 serving engine | 分离 rollout pool。 |

## Rollout/Training 分离时的权重同步

分离部署的关键是权重同步。Training 更新 policy 后，rollout engine 需要拿到新权重，否则会持续用旧 policy 生成样本。

常见同步方式：

| 方式 | 说明 | 风险 |
| --- | --- | --- |
| 周期性保存 checkpoint，再由 rollout 加载 | 实现简单 | checkpoint I/O 和 reload 时间可能很高。 |
| 只同步 adapter | LoRA/QLoRA 场景很轻 | rollout engine 必须支持动态 adapter。 |
| 内存/网络直接广播权重 | 延迟低 | 工程复杂，要求训练和推理框架深度集成。 |
| 固定若干 update 后同步 | 降低同步频率 | policy lag 增大。 |

同步频率不是越高越好：

- 太高会让 rollout engine 频繁 reload，decode 吞吐下降。
- 太低会让 rollout 样本过期，训练信号变旧。

因此要监控：

```text
policy_version_in_training
policy_version_in_rollout
average_policy_lag
reload_time
stale_sample_ratio
rollout_queue_depth
```

这类指标比单纯的 GPU 利用率更能解释在线后训练是否健康。

## LoRA/QLoRA 与后训练

后训练经常和 LoRA/QLoRA 组合。

原因很直接：

- SFT/DPO 实验多，adapter checkpoint 小，便于快速迭代。
- RLHF/GRPO 的 policy update 可能只训练 adapter，降低 optimizer state。
- 多任务、多数据版本可以共享基础模型。

但组合后也有额外问题：

- reference model 是否包含同一个 base、不包含 adapter，还是某个固定 adapter。
- rollout engine 是否支持动态 LoRA。
- cache key 是否包含 adapter id。
- reward/verifier 是否基于同一 tokenizer/template。
- merge/unmerge 是否影响 policy update。
- adapter checkpoint 是否能和 rollout engine 快速同步。

如果后训练系统把 adapter 当作普通小文件，而不是模型版本的一部分，就容易出现 policy rollout、reference logprob、reward scoring 和训练 update 版本不一致。

## Benchmark 方法

后训练 benchmark 要按方法分层。

SFT/DPO 可以报告：

- step time。
- tokens/s。
- effective loss tokens/s。
- peak memory。
- trainable parameters。
- eval time。
- checkpoint size。

RLHF/PPO/GRPO 还要报告：

- rollout prompts/s。
- rollout output tokens/s。
- reward scoring samples/s。
- policy update tokens/s。
- 每轮 rollout/update 的 wall-clock time。
- rollout queue wait。
- stale rollout ratio。
- 每个 prompt 的 responses 数。
- average response length。
- KL、reward、advantage 分布。
- end-to-end time to target eval score。

一个可复现的后训练 benchmark 至少固定：

```yaml
method: "dpo"  # sft / reward_model / dpo / ppo / grpo
base_model: ""
policy_model: ""
reference_model: ""
reward_model: ""
tokenizer_revision: ""
chat_template_revision: ""
dataset_revision: ""
prompt_length_distribution: ""
response_length_distribution: ""
max_prompt_length: null
max_response_length: null
packing: true
precision: "bf16"
parallelism:
  dp: null
  fsdp: null
  tp: null
  pp: null
peft:
  enabled: false
  method: null
rollout:
  enabled: false
  num_responses_per_prompt: null
  sampling_config: {}
  rollout_engine: ""
metrics:
  step_time: null
  rollout_tokens_per_second: null
  update_tokens_per_second: null
  end_to_end_iteration_time: null
  peak_gpu_memory_gb: null
```

## 常见优化方向

### 优化 SFT/DPO 数据组织

重点是减少 padding 和无效 token：

- 按长度分桶。
- packing 多个短样本。
- 明确 loss mask。
- 区分 input tokens 和 loss tokens。
- 预先 tokenization。
- 对超长 prompt/response 做规则化截断。

### 缓存 reference logprob

对 DPO 类训练，如果 reference model 不变，可以考虑预先计算 reference logprob。

收益：

- 减少训练时 reference forward。
- 降低显存和 step time。

风险：

- tokenizer/template/truncation 一变，cache 失效。
- cache 体积可能很大。
- 需要记录 reference model revision。

### 分离 rollout 和 update

对 RLHF/GRPO，如果 rollout 占大头，可以把 rollout engine 独立出来，使用推理系统优化方法：

- continuous batching。
- KV Cache 管理。
- prefix cache。
- speculative decoding，视任务语义谨慎使用。
- 多副本 rollout pool。
- reward/verifier 批量打分。

但 rollout 与 update 分离后，要处理样本过期和版本一致性。

### 减少多模型同时驻留

RLHF/PPO 里 policy、reference、reward、value 都可能吃显存。

常见做法：

- reference 使用 frozen 低精度权重。
- reward model 独立服务化。
- value head 与 policy 共享 backbone。
- LoRA 只训练 adapter。
- 分阶段运行 rollout/scoring/update。
- 用 CPU/offload 或模型并行放下大模型。

### 让 reward/verifier 成为可观测组件

很多后训练瓶颈不是 policy update，而是 reward/verifier scoring。

需要监控：

- scoring latency。
- scoring throughput。
- reward 分布。
- failed / invalid sample ratio。
- verifier timeout。
- prompt/response 长度对 scoring 的影响。

如果 reward/verifier 不稳定，训练曲线可能看起来像优化问题，本质却是数据和打分系统问题。

### 控制 response length

后训练尤其是 RLHF/GRPO，很容易被 response length 拖垮。

response 越长：

- rollout decode token 越多。
- KV Cache 占用越大。
- reward/verifier scoring 越慢。
- DPO/RLHF logprob 计算越重。
- batch padding 浪费越明显。

常见控制方法：

- 设置合理的 `max_new_tokens`。
- 对 prompt 按长度分桶。
- 对超长 response 做截断策略。
- 记录 stop reason。
- 把 average / p95 / p99 response length 放入指标。
- 对无效长输出设置惩罚或过滤。

只优化模型训练 step，不控制 response length，在线后训练端到端成本会失控。

### 把 Eval 从训练主循环中拆出来

后训练 eval 往往比预训练 eval 更复杂。它可能包括：

- 多轮生成式评估。
- reward model 打分。
- verifier / 单元测试。
- 人工偏好抽检。
- 安全或格式约束检查。

如果每隔很少 step 就同步跑完整 eval，训练吞吐会很差。常见做法是：

- 小 eval 高频跑，完整 eval 低频跑。
- eval pool 独立调度。
- eval 只消费稳定 checkpoint 或 adapter。
- eval report 绑定 checkpoint version。
- 训练主循环只等待关键 gating 指标，不等待所有报告。

这样能避免后训练平台被 eval 拖成串行流水线。

## 常见坑

### 把 RLHF 当成普通训练

普通训练只有 dataset batch。RLHF 有 rollout、reward、reference、value、policy update。只测 backward step time 不能代表端到端效率。

### 忘记 reference 版本

DPO/RLHF 的 reference model 是训练语义的一部分。更换 reference 或 chat template 后，旧 logprob cache 可能不能用。

### 只看总 tokens/s

SFT 要区分 loss tokens。RLHF/GRPO 要区分 rollout tokens 和 update tokens。把所有 token 混成一个吞吐指标，会掩盖瓶颈。

### 让 rollout 样本过期

如果 policy 已经更新很多步，旧 policy 生成的 rollout 可能不再适合继续训练。分离架构必须记录 rollout policy version，并设置过期规则。

### 忽略 response 长度分布

后训练 response 长度波动很大。少数超长回答可能拖慢整个 batch，增加 KV Cache 和 activation 压力。

### 没有记录 sampling 配置

temperature、top-p、max tokens、stop words 都会影响 rollout 数据分布。RLHF/GRPO 实验必须记录这些配置。

## 学习顺序建议

建议按这个顺序学：

1. 先理解 SFT，因为它最接近普通监督训练。
2. 再理解 reward model 训练，知道偏好数据如何变成打分模型。
3. 然后理解 DPO，重点看 chosen/rejected 和 reference logprob。
4. 再看 RLHF/PPO，重点看 rollout、reward、reference、value 和 policy update 的闭环。
5. 最后看 GRPO，理解 group responses 如何改变 rollout 数量和 advantage 计算。

从系统角度记住一句话：

> 后训练不是一种训练 step，而是一组数据、推理、打分和优化循环；不同方法的瓶颈可能完全不在同一个地方。

## 参考资料

- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- [Hugging Face TRL documentation](https://huggingface.co/docs/trl/index)
- [Hugging Face TRL DPO Trainer](https://huggingface.co/docs/trl/main/en/dpo_trainer)
- [Hugging Face TRL PPO Trainer](https://huggingface.co/docs/trl/main/en/ppo_trainer)
- [Hugging Face TRL GRPO Trainer](https://huggingface.co/docs/trl/main/en/grpo_trainer)
