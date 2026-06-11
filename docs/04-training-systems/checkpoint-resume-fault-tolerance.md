---
title: Checkpoint、Resume 与容错
domain: training-systems
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-06-11
---

# Checkpoint、Resume 与容错

训练系统不能只追求每一步跑得快。大模型训练可能持续数天、数周甚至更久，中间一定会遇到机器故障、网络抖动、作业抢占、存储异常、代码升级、配置修改和人工误操作。

Checkpoint 的目标不是“保存一个模型文件”，而是：

> 在训练中断后，系统能尽量少丢进度，并且用可验证的状态恢复训练。

这篇重点讨论训练系统里的 checkpoint、resume 和容错设计。它不讲推理模型导出，也不讲怎么把模型上传到模型仓库。这里关心的是长期训练如何活下来。

## Checkpoint 保存的到底是什么

最小的模型 checkpoint 可能只保存参数：

```text
model weights
```

这对推理可能够用，但对继续训练远远不够。训练 checkpoint 至少要保存：

| 状态 | 为什么需要 |
| --- | --- |
| model parameters | 恢复模型权重 |
| optimizer state | 恢复 AdamW 的 `m/v`、Muon momentum、master weights 等 |
| scheduler state | 恢复学习率曲线位置 |
| scaler state | FP16 AMP 需要恢复 loss scaling |
| RNG state | 让 dropout、数据增强、随机采样尽量连续 |
| dataloader / sampler state | 避免重复或跳过数据 |
| global step | 决定日志、eval、scheduler、checkpoint cadence |
| consumed samples / tokens | 大模型训练通常按 token 计进度 |
| parallelism metadata | TP/PP/DP/EP/FSDP/ZeRO 配置 |
| code/config metadata | 记录这份状态属于哪个训练配置 |

所以完整训练 checkpoint 更像一个目录：

```text
checkpoint-00012000/
  metadata.json
  model/
  optimizer/
  scheduler/
  scaler/
  rng/
  dataloader/
  trainer_state.json
```

分布式训练里还会有每个 rank 或每个 shard 的文件。

## 为什么只保存权重不够

假设一个 AdamW 训练在 step 10000 中断。如果只保存模型权重，然后重新初始化 optimizer，从 step 10000 接着跑，会发生什么？

- AdamW 的一阶矩 `m` 丢失。
- AdamW 的二阶矩 `v` 丢失。
- 学习率 scheduler 可能回到 warmup 起点或错误位置。
- loss scaler 可能回到默认值。
- dataloader 可能从错误位置继续。
- dropout 和随机增强轨迹会变化。

训练当然可能还能跑，但它不再是同一条训练轨迹。

对于研究和系统调优，这会带来两个问题：

1. 实验不可复现。
2. resume 后 loss 曲线可能出现异常，但很难判断是模型问题、数据问题还是状态缺失。

所以需要区分两种 checkpoint：

| 类型 | 目的 | 内容 |
| --- | --- | --- |
| training checkpoint | 继续训练 | model + optimizer + scheduler + scaler + RNG + data state |
| inference/export checkpoint | 部署或评测 | 通常只需要模型权重和必要配置 |

不要用 inference checkpoint 当作长期训练容错方案。

## 一次保存的生命周期

一次 checkpoint save 通常经历：

```mermaid
flowchart TD
    A["触发保存条件"] --> B["暂停或标记训练状态"]
    B --> C["收集 model state"]
    C --> D["收集 optimizer/scheduler/scaler/RNG/data state"]
    D --> E["按 rank/shard 写入临时目录"]
    E --> F["写 metadata 和 manifest"]
    F --> G["校验文件完整性"]
    G --> H["原子切换 latest 指针"]
    H --> I["清理旧 checkpoint"]
```

这里有几个关键点：

- 不应该先覆盖旧的 latest。
- 不应该写一半就让系统认为 checkpoint 可用。
- metadata 和 manifest 要能描述每个 shard。
- 失败时要么保留旧 checkpoint，要么明确标记新 checkpoint 不完整。

## 一次恢复的生命周期

一次 resume 通常经历：

```mermaid
flowchart TD
    A["找到 checkpoint"] --> B["读取 metadata/manifest"]
    B --> C["校验配置兼容性"]
    C --> D["初始化模型和并行组"]
    D --> E["加载 model state"]
    E --> F["加载 optimizer/scheduler/scaler"]
    F --> G["恢复 RNG/data/global step"]
    G --> H["跑一次轻量 sanity check"]
    H --> I["继续训练"]
```

恢复时最危险的不是 load 报错，而是 load 成功但状态错位。

比如：

- 参数名对上了，但 shape 对不上。
- 参数 shape 对上了，但 optimizer state 对错参数。
- scheduler 加载了，但 global step 不一致。
- 数据恢复了，但 consumed tokens 不一致。
- world size 变化了，但 checkpoint 格式不支持 reshard。

所以 resume 后要做验证，而不是只看程序能继续跑。

## 完整性：哪些状态必须保存

### Model State

模型参数是最基本部分。

需要记录：

- 参数名。
- shape。
- dtype。
- sharding 信息。
- tied weights 关系。
- embedding / lm_head 是否共享。
- FP8 / quantized training 的额外 scale 状态，如果有。

FSDP、ZeRO、TP、PP 下，模型参数可能是 sharded state，而不是单个完整 `state_dict`。

### Optimizer State

Optimizer state 往往比模型权重更大。

AdamW 可能包括：

- first moment `m`。
- second moment `v`。
- FP32 master weights。
- step counter。
- parameter group hyperparameters。

Muon 可能包括：

- momentum buffer。
- master weights，如果有。
- parameter group 信息。
- Newton-Schulz / update scaling 相关配置。

训练 checkpoint 如果不保存 optimizer state，只能算 warm-start，不能算严格 resume。

### Scheduler State

Scheduler 看似很小，但语义非常关键。

必须知道学习率处在：

- warmup 第几步。
- cosine decay 第几步。
- token-based schedule 的哪个 token 位置。
- gradient accumulation 后第几个 optimizer step。

如果 scheduler 按 micro-step 计数，但 resume 按 optimizer step 恢复，学习率曲线会漂移。

### AMP / GradScaler State

FP16 训练常用 dynamic loss scaling。GradScaler 保存当前 scale 和调整历史。

如果恢复时丢失 scaler state，可能出现：

- resume 后短期 overflow。
- scale 重新从默认值爬升，影响吞吐。
- NaN/Inf 处理行为和中断前不同。

BF16 通常不需要 loss scaling，但仍要保存其他 mixed precision 配置。

### RNG State

RNG state 包括：

- Python random。
- NumPy random。
- PyTorch CPU RNG。
- CUDA RNG。
- 每个 rank 的 CUDA RNG。
- model/tensor parallel 相关 RNG tracker。

它影响：

- dropout。
- data augmentation。
- masked language modeling mask。
- MoE routing 中的随机策略，如果有。
- sequence packing 的随机性。

严格复现需要保存 RNG。工程容错至少要知道是否保存了 RNG。

### Data State

数据状态经常被忽略，但长期训练很重要。

需要保存：

- epoch。
- dataset shard。
- sampler position。
- consumed samples。
- consumed tokens。
- packing buffer 状态。
- shuffle seed。
- streaming dataset cursor。
- tokenizer / data version。

如果训练按 token budget 管理，`consumed_tokens` 比 `global_step` 更重要。

## Sharded Checkpoint

单机小模型可以保存一个文件。大模型训练通常不能这样做。

原因：

- 单 rank 聚合完整模型可能 OOM。
- 单文件写入速度慢。
- optimizer state 太大。
- 多节点网络会把 rank0 打成瓶颈。
- 恢复时再广播完整 state 很慢。

Sharded checkpoint 的思路是：

```text
rank 0 writes shard 0
rank 1 writes shard 1
rank 2 writes shard 2
...
metadata describes all shards
```

PyTorch Distributed Checkpoint 的 `save` 会处理 `ShardedTensor` 和 `DTensor`，让每个 rank 保存本地 shard。DeepSpeed 也强调训练 checkpoint 下所有进程都要调用保存接口，因为每个进程都有自己的 master weights、scheduler 和 optimizer states。

### Sharded Checkpoint 的关键元数据

一个 shard 文件本身不够，还需要 metadata：

- checkpoint version。
- world size。
- rank layout。
- TP/PP/DP/EP size。
- 每个 tensor 的 global shape。
- 每个 shard 的 offsets。
- 每个 shard 的 dtype。
- 每个 shard 对应的 parameter FQN。
- optimizer state 与 parameter 的映射。
- 保存时的 framework/runtime 版本。

没有这些元数据，shard 只是一堆难以重组的 tensor。

## Resharding：并行度变化后的恢复

真实训练中，经常需要从一个并行配置恢复到另一个并行配置。

例如：

```text
原训练:
  TP=4, PP=8, DP=16

新训练:
  TP=8, PP=4, DP=16
```

或者从 64 张 GPU 扩到 128 张 GPU。

这时 checkpoint 需要 reshard：

```text
old shards -> global tensor view -> new shards
```

Resharding 的难点：

- 每个 tensor 的切分维度不同。
- TP 切线性层，PP 切 layer，DP 切 optimizer state，EP 切 expert。
- optimizer state 也要同步 reshard。
- tied weights 不能被拆坏。
- 大规模 all-gather / redistribute 会很慢。

Megatron Core 的 distributed checkpoint 文档明确区分了不同 optimizer checkpoint format：有的保存加载快，但不能改变 model parallelism；有的 fully reshardable，支持任意改变 model parallelism，但更慢。

工程选择很直接：

- 平时高频保存，用快格式。
- 需要迁移并行度时，保存一份可重分片格式。
- 不要等故障后才发现 checkpoint 不能在新资源上恢复。

## 保存频率怎么定

Checkpoint 太频繁，会拖慢训练。太不频繁，故障后丢失太多进度。

保存频率要考虑：

- 平均故障间隔。
- 单次 checkpoint 保存时间。
- checkpoint 对 step time 的影响。
- 可接受的最大丢失 token。
- 存储容量。
- 恢复时间。
- 是否有异步保存。

可以用一个简单公式估算：

```text
lost_work_time <= checkpoint_interval
```

如果每 2 小时保存一次，故障最坏丢 2 小时，平均可能丢 1 小时。

大模型训练更常按 token 设计：

```text
checkpoint every N tokens
```

因为 global step 会受 batch size、sequence length、gradient accumulation 影响，而 token 是训练预算的真实单位。

## 同步保存与异步保存

### 同步保存

同步保存简单：

```text
stop training -> write checkpoint -> continue training
```

优点：

- 语义清楚。
- 容易调试。
- 失败边界明确。

缺点：

- 保存期间 GPU 可能等待。
- 大 checkpoint 会造成明显 step time spike。

### 异步保存

异步保存试图把 checkpoint 写入和训练重叠：

```text
stage data -> background thread/process writes -> training continues
```

PyTorch DCP 提供 `async_save`，会先把 state_dict staging 到存储位置或 CPU，再在后台执行保存路径。

异步保存的工程问题：

- staging 会占 CPU 内存或 GPU/CPU 带宽。
- 后台写入失败要能上报。
- 退出训练前必须等待最后一次保存完成。
- 不能覆盖仍在后台写入的数据。
- checkpoint metadata 只能在所有 shard 成功后标记可用。

异步保存不是“免费保存”，只是把部分时间挪到后台。必须 benchmark。

## 原子性与 latest 指针

Checkpoint 的目录命名常见：

```text
checkpoint-00010000/
checkpoint-00011000/
checkpoint-00012000/
latest
```

`latest` 可以是文件，也可以是软链接，指向最近完整 checkpoint。

正确流程应类似：

1. 写入临时目录 `checkpoint-00012000.tmp/`。
2. 所有 rank 写完自己的 shard。
3. 写 manifest。
4. 校验 manifest。
5. rename 为 `checkpoint-00012000/`。
6. 原子更新 `latest`。

错误流程是：

1. 先更新 `latest`。
2. 再慢慢写文件。
3. 写到一半作业失败。

这样下次 resume 会读到半成品。

## 容错：故障发生时系统怎么恢复

分布式训练常见故障：

- 某个 worker 进程崩溃。
- 某台节点宕机。
- NCCL collective hang。
- GPU Xid / ECC error。
- 网络短暂不可用。
- 存储写入超时。
- 作业被调度系统抢占。
- 代码抛异常。

容错策略大致分三层。

### 训练脚本层

训练脚本需要：

- 定期保存 checkpoint。
- 启动时自动查找 latest。
- 支持从指定 checkpoint 恢复。
- 恢复后校验 global step / consumed tokens。
- 遇到可恢复异常时退出给外层重启。

### 分布式 launcher 层

例如 `torchrun` elastic 会在 worker failure 时停止并重启 worker group。PyTorch 文档也提醒，failure 或 membership change 发生时，存活 workers 也会被杀掉，脚本需要 checkpoint 进度；rank 不稳定，不能硬编码 rank 与数据或文件的固定关系。

这意味着：

- checkpoint 不能依赖“rank 0 永远是同一台机器”。
- 数据 shard 不能只靠 rank 编号恢复。
- world size 变化时，脚本不能假设旧 world size 仍然成立。

### 调度系统层

Kubernetes、Slurm、Ray、内部调度系统等要负责：

- 重启作业。
- 分配新节点。
- 注入 checkpoint path。
- 控制最大重启次数。
- 报告失败原因。
- 清理不完整输出。

训练代码不应该把调度系统行为写死，但要暴露足够清晰的 resume 接口。

## Rank 不稳定带来的问题

Elastic training 中，重启后 rank 可能变化。PyTorch 文档明确提醒 `RANK` 不是稳定身份。

如果代码这样保存：

```text
rank_0.pt
rank_1.pt
rank_2.pt
```

并且假设恢复时同一个 rank 读同一个文件，就会有风险。

更稳妥的方式是：

- metadata 描述 shard，而不是只描述 rank。
- shard 与 global tensor range 绑定。
- rank 恢复时根据当前并行布局读取需要的 shard。
- 数据进度按 global sample/token cursor 保存，而不是按 rank 私有计数保存。

Rank 是运行时角色，不应该是 checkpoint 的长期身份。

## 存储系统设计

Checkpoint 是典型的大规模写入 workload。

要考虑：

- 写入带宽。
- 小文件数量。
- 元数据服务压力。
- 多节点并发写。
- 对象存储一致性。
- 本地 NVMe staging。
- 远端持久化。
- 删除旧 checkpoint 的速度。

常见策略：

### 本地 NVMe + 后台上传

先写本地 NVMe，尽快释放训练进程，再异步上传对象存储。

风险：

- 节点坏了，本地 checkpoint 丢失。
- 后台上传失败必须报警。
- latest 指针要以远端完整 checkpoint 为准。

### 直接写共享文件系统

实现简单，但要关注：

- 元数据瓶颈。
- 并发小文件。
- 单目录文件数量。
- 写入抖动影响训练。

### 对象存储

适合持久化和跨集群恢复，但要关注：

- multipart upload。
- 最终一致性或 list 延迟。
- manifest 原子性。
- 大量小对象成本。

不管哪种方式，都要把 checkpoint save/load 时间纳入训练 benchmark。

## Checkpoint 保留策略

不能无限保留所有 checkpoint。

常见保留策略：

- 最近 N 个 checkpoint 全保留。
- 每隔 M 个 checkpoint 保留一个长期点。
- 关键阶段结束保留。
- eval 最优保留。
- 出现异常前后的 checkpoint 保留。
- 不完整 `.tmp` checkpoint 定期清理。

示例：

```text
keep:
  latest 5 checkpoints
  every 100B tokens
  stage boundary checkpoints
  best validation checkpoint
delete:
  tmp older than 24h
  failed manifests
```

长期训练还要记录删除操作，避免误删唯一可恢复点。

## Resume 后如何验证

恢复不是 load 成功就结束。

建议做以下检查：

### 静态检查

- checkpoint version 是否支持。
- model config 是否一致。
- tokenizer / vocab 是否一致。
- parallelism config 是否兼容。
- parameter names 是否匹配。
- missing/unexpected keys 是否为 0，或在白名单内。
- optimizer state 数量是否匹配。
- dtype 是否符合预期。

### 动态检查

- resume 后第一步 loss 是否连续。
- learning rate 是否连续。
- grad norm 是否连续。
- scaler scale 是否合理。
- consumed tokens 是否继续增加。
- dataloader 是否没有回退到开头。
- eval 小样本结果是否接近中断前。

### 分布式检查

- 所有 rank 都加载到对应 shard。
- TP/PP/DP/EP group 重新构建正确。
- rank-local checkpoint 文件不是错读。
- optimizer state sharding 与当前 world size 匹配。
- all ranks 的 global step 一致。

最好把这些检查做成 resume sanity check，而不是人工看日志。

## 配置变更与兼容性

训练过程中可能修改配置，例如：

- batch size。
- gradient accumulation。
- sequence length。
- TP/PP/DP size。
- optimizer 参数。
- scheduler。
- tokenizer。
- model architecture。

不是所有变化都能安全 resume。

| 配置变化 | 风险 |
| --- | --- |
| global batch 变化 | optimizer/scheduler 语义改变 |
| sequence length 变化 | consumed tokens、position embedding、data packing 变化 |
| TP/PP 变化 | 需要 reshard |
| optimizer 改变 | optimizer state 不兼容 |
| scheduler 改变 | LR 曲线不连续 |
| tokenizer 改变 | 数据语义改变 |
| vocab size 改变 | embedding/lm_head shape 改变 |
| model layer 数改变 | checkpoint 不能直接匹配 |

建议将配置分成：

- resume-compatible。
- warm-start-only。
- forbidden unless conversion script exists。

这比在运行时临时猜更可靠。

## Warm-start 和 Resume 的区别

很多系统把两者混在一起，实际差别很大。

| 行为 | Resume | Warm-start |
| --- | --- | --- |
| model weights | 加载 | 加载 |
| optimizer state | 加载 | 通常不加载 |
| scheduler state | 加载 | 重新设置 |
| RNG/data state | 尽量恢复 | 通常不恢复 |
| 目标 | 延续同一次训练 | 从已有权重开始新训练 |
| loss 曲线 | 应连续 | 可以不连续 |

如果只加载模型权重继续训练，应明确叫 warm-start，不要叫 resume。

## Checkpoint 性能指标

训练 benchmark 里应记录：

- checkpoint size。
- save time。
- load time。
- save bandwidth。
- load bandwidth。
- checkpoint interval。
- training stall time。
- async staging time。
- CPU memory peak。
- storage error rate。
- failed checkpoint count。
- resume success rate。

Checkpoint 的代价可以折算到训练吞吐：

```text
effective_training_time = compute_time + checkpoint_stall_time + recovery_time
```

如果每小时训练 55 分钟、保存 checkpoint 卡 5 分钟，理论上已经损失约 8.3% 时间。

## 常见优化方向

### Sharded Save/Load

避免 rank0 聚合完整模型。每个 rank 保存自己的 shard，并用 manifest 记录全局视图。

### 异步保存

把写入放到后台，但必须处理 staging 内存、失败上报和最终一致性。

### 降低小文件数量

大量小文件会压垮元数据服务。可以按 rank 合并、按 tensor group 合并，或使用适合对象存储的格式。

### 本地缓存和分层存储

短期恢复用本地或近端存储，长期归档用远端对象存储。

### 可重分片格式

当训练资源经常变化时，优先支持 resharding。代价是保存/加载可能更慢。

### 保存轻重分层

高频保存轻量 checkpoint，低频保存完整可重分片 checkpoint。

例如：

```text
every 1B tokens:
  fast sharded training checkpoint

every 20B tokens:
  fully reshardable checkpoint

stage boundary:
  export/inference checkpoint
```

## 常见误区

### 误区一：只保存 rank0 就够了

DDP 下也许可以，但 ZeRO/FSDP/TP/PP/MoE 下通常不行。DeepSpeed 文档明确说明训练 checkpoint 要所有进程都调用保存接口。

### 误区二：load 成功就说明恢复正确

不够。Load 成功只说明文件能读，不说明 optimizer、scheduler、data cursor、RNG 和 parallelism 都正确。

### 误区三：checkpoint 越频繁越安全

过于频繁会拖慢训练，也可能给存储系统造成压力。频率要由故障率、保存时间、可接受丢失进度共同决定。

### 误区四：rank 编号是稳定身份

Elastic training 下 rank 会变化。Checkpoint 应按全局 tensor/shard metadata 恢复，而不是按旧 rank 假设恢复。

### 误区五：换 world size 后一定能恢复

不一定。要看 checkpoint 格式是否支持 resharding，optimizer state 是否支持迁移。

## 设计检查清单

设计训练 checkpoint 时，可以逐项确认：

- 是否保存 model weights？
- 是否保存 optimizer state？
- 是否保存 scheduler state？
- 是否保存 AMP scaler？
- 是否保存 RNG？
- 是否保存 data cursor / consumed tokens？
- 是否记录 TP/PP/DP/EP/FSDP/ZeRO 配置？
- 是否记录代码版本、配置 hash、数据版本？
- 是否支持 sharded checkpoint？
- 是否支持 world size 改变后的 reshard？
- 是否有 manifest？
- 是否有 atomic latest 更新？
- 是否清理不完整 checkpoint？
- 是否测试过从最新 checkpoint resume？
- 是否测试过从旧 checkpoint resume？
- 是否测试过某个 rank 失败后的恢复？
- 是否测试过 checkpoint 写满磁盘或对象存储失败？
- 是否监控 save/load time？

## 小结

Checkpoint、Resume 与容错是训练系统的可靠性基础。

关键结论：

- 训练 checkpoint 必须保存完整训练状态，不只是模型权重。
- Sharded checkpoint 是大模型训练的常态。
- Resume 要验证 optimizer、scheduler、RNG、data cursor 和 parallelism metadata。
- Elastic 环境下 rank 和 world size 可能变化，checkpoint 不能依赖旧 rank 身份。
- 保存频率、保存格式和存储后端都会影响有效训练吞吐。
- 可重分片 checkpoint 更灵活，但可能更慢；快速 checkpoint 更高效，但迁移能力有限。

真正可靠的训练系统，不是永远不失败，而是失败后能用清楚、可验证、低损失的方式继续。

## 参考资料

- [PyTorch: Distributed Checkpoint](https://docs.pytorch.org/docs/2.12/distributed.checkpoint.html)
- [PyTorch: torchrun Elastic Launch](https://docs.pytorch.org/docs/2.12/elastic/run.html)
- [DeepSpeed: Model Checkpointing](https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html)
- [Megatron Core: dist_checkpointing package](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/dist_checkpointing.html)
