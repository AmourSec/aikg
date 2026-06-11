---
title: 存储、数据缓存与 Checkpoint：NVMe、并行文件系统与对象存储
domain: cluster-infra
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-06-12
---

# 存储、数据缓存与 Checkpoint：NVMe、并行文件系统与对象存储

AI 集群里的存储不是“找个盘放文件”。训练数据、模型权重、checkpoint、容器镜像、日志、embedding、RAG 索引和推理缓存都有不同的访问模式、容量需求、延迟要求和一致性要求。

存储设计要回答的问题是：

> 哪些数据应该放在对象存储，哪些放在共享文件系统，哪些放在本地 NVMe，哪些必须进入 GPU HBM？数据什么时候预取、缓存、校验、清理和恢复？

如果存储设计不好，GPU 会等数据，训练会等 checkpoint，推理扩容会等模型加载，RAG 会等索引读取，集群会在高峰期被小文件和元数据请求拖慢。

## 一张总图

```mermaid
flowchart TB
    Source["Source Data<br/>raw corpus / images / logs / tables"]
    Object["Object Store / Data Lake<br/>S3 / OSS / COS / MinIO"]
    ParallelFS["Parallel FS / Shared FS<br/>dataset shards / checkpoints / shared workspace"]
    Registry["Image / Model Registry<br/>container image / model artifacts"]
    Local["Node Local NVMe<br/>hot cache / staging / shuffle / checkpoint buffer"]
    Host["Host Memory<br/>DataLoader / tokenizer / page cache"]
    GPU["GPU HBM<br/>batch tensors / KV cache / activations"]
    Remote["Remote Consumers<br/>training / serving / eval / RAG"]

    Source --> Object
    Object --> ParallelFS
    Object --> Local
    Registry --> Local
    ParallelFS --> Local
    Local --> Host
    Host --> GPU
    GPU --> Local
    Local --> ParallelFS
    ParallelFS --> Object
    Object --> Remote
```

这张图表达几个关键点：

- 对象存储适合大容量、低成本、版本化和长期保存。
- 共享/并行文件系统适合多节点共同访问和 checkpoint。
- 本地 NVMe 适合热数据、临时数据和高速 staging。
- Host memory 和 page cache 是 CPU 侧过渡层。
- GPU HBM 只适合最热的数据和当前计算状态。
- 数据路径应该有生命周期，而不是所有东西都永久堆在一个目录里。

## AI 集群里有哪些数据对象

先区分对象，再选存储。

| 数据对象 | 典型大小 | 访问模式 | 关键问题 |
| --- | --- | --- | --- |
| 原始数据 | TB-PB | 顺序写、多次读 | 版本、权限、成本 |
| 训练样本 shard | GB-TB | 高吞吐顺序读 | 吞吐、shuffle、metadata |
| 小文件数据集 | 大量 KB-MB 文件 | metadata 密集 | 元数据瓶颈、打包 |
| tokenizer / vocab | MB-GB | 多任务重复读 | 缓存、版本 |
| 模型权重 | GB-TB | 启动时读、部署时分发 | 拉取速度、校验、灰度 |
| checkpoint | GB-TB/次 | 周期性写、恢复读 | 原子性、吞吐、保留策略 |
| optimizer state | 模型参数数倍 | checkpoint 写入 | sharding、恢复一致性 |
| activation offload | GB-TB 临时 | step 内读写 | 延迟、带宽、生命周期 |
| KV Cache | GB-TB 热数据 | decode 高频读写 | latency、容量、淘汰 |
| embedding / vector index | GB-TB | 查询密集、批量更新 | p99、更新、分片 |
| container image | GB-十GB | 节点启动时拉取 | registry、预热、层缓存 |
| logs / traces | 持续写 | 小写入、查询 | retention、索引成本 |

不同对象混在同一个共享文件系统里，是很多 AI 集群的早期常见错误。

## 数据对象生命周期

AI 存储设计要先按生命周期划分对象，而不是先选某个存储产品。

一个对象通常会经历：

```text
生成 / 采集
  -> 清洗 / 转换
  -> 版本化
  -> 分发 / 缓存
  -> 训练 / 推理 / 评测使用
  -> 归档 / 删除
```

不同阶段需要不同策略：

| 阶段 | 关注点 | 常见位置 |
| --- | --- | --- |
| 原始采集 | 不丢数据、权限、审计、低成本 | object store / data lake |
| 处理后数据 | 版本、schema、shard、可复现 | object store + metadata catalog |
| 热训练数据 | 高吞吐、低 metadata 压力 | parallel FS / local NVMe cache |
| 临时中间数据 | 快速读写、自动清理 | local NVMe / scratch FS |
| checkpoint 当前版本 | 写入原子性、快速恢复 | parallel FS / fast object tier |
| checkpoint 归档 | 成本、保留策略、合规 | object store / archive tier |
| 模型权重 | 版本、digest、灰度、分发速度 | model registry + node cache |
| 日志和 trace | 查询、成本、保留时间 | log system / object archive |

生命周期还决定删除策略。

- raw data 通常长期保留，但要做权限和合规治理。
- processed dataset 要按版本保留，避免训练结果无法复现。
- local cache 可以删除，但必须能从远端重建。
- checkpoint 不能只按时间删除，要考虑是否仍被实验、评估、模型发布或回滚引用。
- 临时 shard、staging 文件、失败 checkpoint 要自动清理，否则会吃掉本地 NVMe 和共享文件系统容量。

一个成熟平台应该能回答：某个文件是事实来源、缓存、临时文件、checkpoint、artifact，还是日志。不同身份决定了它能不能删、什么时候删、谁能读、读错后如何恢复。

## 存储层次

### 对象存储

对象存储适合长期保存和大规模数据管理。

常见用途：

- raw dataset。
- processed dataset。
- model artifact。
- checkpoint archive。
- evaluation output。
- logs archive。
- RAG 文档原文。

对象存储的特点：

- 容量大。
- 成本相对低。
- 适合按 bucket/key 管理对象。
- 易于做版本、生命周期、权限和跨区域复制。
- 不提供 POSIX 文件系统语义。
- 大量小对象可能效率差。
- 高性能训练通常需要缓存或打包。

Amazon S3 官方文档说明，S3 以 bucket 和 object/key 组织数据，并对对象 PUT/DELETE 提供强读后写一致性；同时也说明单个 key 的更新是原子的，但跨 key 不提供事务式原子更新。这对 checkpoint 设计很重要：不能假设多个文件一起写就是原子事务。

### 并行文件系统 / 共享文件系统

共享文件系统适合多节点共同访问。

常见用途：

- 训练数据 shard。
- checkpoint 当前版本。
- 多节点共享 workspace。
- 评测结果。
- 用户 home / project space。

常见类型：

- Lustre。
- GPFS / IBM Spectrum Scale。
- BeeGFS。
- NFS。
- CephFS。
- 云厂商托管并行文件系统。

它们差异很大，不能只说“共享盘”。

关键指标：

- 顺序读写吞吐。
- 小文件 metadata 性能。
- 多客户端并发。
- 单目录文件数。
- stripe / block 配置。
- failure recovery。
- snapshot。
- quota。
- POSIX 语义。
- 与 GDS / RDMA / CSI 的支持程度。

并行文件系统不等于无限快。多个训练任务同时启动、同时 checkpoint、同时扫描小文件，会让 metadata server 和存储网络成为瓶颈。

### 本地 NVMe

本地 NVMe 是每台计算节点上的高速本地存储。

适合：

- dataset hot cache。
- model weight cache。
- checkpoint staging。
- shuffle buffer。
- tokenizer cache。
- temporary files。
- spill / offload。
- RAG index shard。

本地 NVMe 优点：

- 延迟低。
- 带宽高。
- 不经过共享存储网络。
- 适合反复读取热数据。

缺点：

- 节点本地，不天然共享。
- 节点故障会丢失临时数据。
- 调度迁移后 cache miss。
- 容量有限。
- 需要清理和配额。

所以本地 NVMe 应作为 cache/staging 层，而不是唯一持久层。

### 存储策略契约

平台应该把存储能力抽象成少数明确的策略，而不是让用户直接面对一堆路径。

例如：

| 策略 | 语义 | 适合对象 |
| --- | --- | --- |
| `dataset-cold` | 大容量、低成本、可版本化 | raw data、processed dataset archive |
| `dataset-hot` | 高吞吐读取、适合训练 | packed shard、epoch hot set |
| `checkpoint-fast` | 高写入吞吐、快速恢复 | 当前训练 checkpoint |
| `checkpoint-archive` | 成本优先、长期保留 | 重要里程碑、发布候选 |
| `local-cache` | 本地 NVMe，可丢弃，可重建 | dataset shard、model weight cache |
| `model-artifact` | digest、版本、灰度、权限 | 模型权重、tokenizer、配置 |
| `image-cache` | 镜像层缓存和预拉取 | container image layer |
| `logs-traces` | 查询和保留策略 | stdout、profile、trace、event |

这个契约要说明：

- 是否持久。
- 是否可丢弃。
- 是否支持多节点共享。
- 是否要求 POSIX 语义。
- 是否支持原子 rename 或 success marker。
- 吞吐、延迟、metadata 能力大概是什么级别。
- quota 和计费归属如何计算。
- 备份、归档、生命周期如何执行。

有了契约，用户提交任务时只需要声明“这是 dataset hot cache”或“这是 checkpoint fast path”，平台再映射到底层对象存储、并行文件系统、本地 NVMe 或缓存服务。否则用户会把所有东西都写到同一个路径，最后无法区分性能、成本和删除策略。

### Host Memory / Page Cache

Linux page cache 和 host memory 也在数据路径上。

它们影响：

- DataLoader。
- tokenizer。
- CPU preprocessing。
- mmap dataset。
- pinned memory。
- H2D copy。

如果数据集反复读取，page cache 可以提高性能；但如果多个任务互相污染 cache，或者数据大于内存，性能会抖动。

### GPU HBM

GPU HBM 是最热层。

适合：

- 当前 batch tensor。
- activation。
- parameter shard。
- KV Cache hot block。
- temporary kernel buffer。

不适合：

- 长期保存数据集。
- 大量冷 KV。
- 长期 checkpoint。
- 低频访问的模型历史版本。

推理系统里，KV Cache 是最典型的 HBM 容量压力来源。把冷 KV 分层到 host memory、local NVMe 或远端 cache 要非常谨慎，因为 Decode 每步都可能访问 KV，延迟会直接影响 TPOT。

## 数据集读取路径

训练数据常见路径：

```text
object store / parallel FS
  -> node local NVMe cache
  -> host memory / DataLoader
  -> pinned memory
  -> GPU HBM
```

每一层都可能成为瓶颈。

### 小文件问题

AI 数据集常来自图片、文本片段、JSON、日志、小样本文件。大量小文件会造成：

- metadata 请求多。
- open/close 开销大。
- directory scan 慢。
- 分布式多 worker 同时扫描时放大压力。
- object store 请求数量暴涨。

常见优化：

- 打包成 shard。
- 使用 tar / WebDataset。
- 使用 Parquet / Arrow / TFRecord / RecordIO 等格式。
- 每个 shard 足够大，减少 metadata。
- shard 内顺序读。
- 预生成 index。
- 避免每个 rank 重复扫描全目录。

目标是让训练读数据更像高吞吐顺序读，而不是大量随机小文件请求。

### Sharding 与 Shuffle

训练需要随机性，但存储喜欢顺序读。

常见折中：

- dataset 分成较大 shard。
- shard 内顺序读。
- shard 级别 shuffle。
- buffer shuffle。
- epoch 间重新排列 shard。
- 每个 rank 读取不同 shard。

错误做法是所有 rank 都从同一个共享目录随机读小文件。这会把存储和 metadata 打爆。

### DataLoader 与 CPU

DataLoader 可能成为瓶颈。

要看：

- worker 数。
- CPU core。
- NUMA。
- tokenizer 速度。
- decompression。
- image decode。
- augmentation。
- Python GIL。
- pinned memory。
- prefetch factor。
- host-to-device copy。

如果 GPU utilization 周期性掉零，可能不是 GPU 问题，而是 DataLoader、存储或 CPU preprocessing 跟不上。

## 数据缓存策略

缓存不是“复制一份数据”这么简单。要定义缓存对象、失效策略、预热方式和一致性。

### Read-through Cache

任务首次读取数据时，如果本地没有，就从远端拉取到本地 NVMe，后续复用。

优点：

- 简单。
- 不需要预先知道所有数据。

缺点：

- 第一次访问慢。
- 多任务同时 cache miss 会冲击远端。
- 需要控制本地容量和淘汰。

### Pre-staging

任务启动前，把需要的数据 shard 预先拉到本地 NVMe。

优点：

- 训练开始后更稳定。
- 可提前发现数据缺失。

缺点：

- 启动时间变长。
- 如果任务被抢占，预热成本浪费。
- 需要和调度器结合。

### Shared Cache

一组节点共享一个缓存层，例如高性能缓存文件系统或缓存服务。

优点：

- 多节点复用。
- 避免每台机器重复拉取。

缺点：

- 仍然可能成为集中瓶颈。
- 需要一致性和淘汰策略。

### Cache Key

缓存必须有清晰 key。

常见 key：

- dataset version。
- shard id。
- preprocessing version。
- tokenizer version。
- model artifact digest。
- quantization config。

如果 cache key 不包含数据版本和处理逻辑版本，就可能出现“读到了旧数据但没人发现”的问题。

### 缓存治理

缓存要被治理，否则它会从优化手段变成新的故障源。

需要定义：

| 问题 | 需要的策略 |
| --- | --- |
| 谁可以写缓存 | 防止低优任务污染公共 cache |
| cache key 怎么生成 | dataset、preprocess、tokenizer、model digest 必须进入 key |
| 何时预热 | 调度前、任务启动前、低峰期、模型发布前 |
| 如何淘汰 | LRU、按项目 quota、按热度、按过期时间 |
| 如何限流 | 避免大量 cache miss 同时冲击 object store |
| 如何校验 | checksum、size、manifest、success marker |
| 如何清理 | 任务结束、节点 drain、磁盘水位线、版本过期 |
| 如何计费 | 本地 NVMe、共享缓存、远端读取成本归属 |

缓存治理里最重要的是避免缓存雪崩。

例如一个热门模型发布新版本后，很多推理副本同时启动。如果所有副本都发现本地 cache miss，然后同时从对象存储拉 100 GB 权重，网络、对象存储和节点磁盘都会被打爆。

常见保护：

- 分批预热。
- 每节点只允许一个下载者，其他进程等待本地结果。
- 先写临时路径，校验通过后再原子切换到缓存路径。
- cache miss 限速和排队。
- 远端源做 mirror 或分层缓存。
- 缓存 key 使用 digest，避免覆盖旧版本。
- 发布系统先预热再切流量。

缓存命中率本身也不够。还要看 cache miss 放大、预热耗时、远端读取峰值、本地盘水位、错误缓存和缓存清理延迟。

## Checkpoint 是存储系统压力源

Checkpoint 不是简单保存一个文件。

训练 checkpoint 可能包括：

- model parameters。
- optimizer state。
- scheduler state。
- RNG state。
- dataloader state。
- sampler state。
- global step。
- distributed rank/shard metadata。
- AMP / grad scaler。
- tokenizer / config。
- framework / code version。

大模型训练中，optimizer state 和 sharded parameters 会让 checkpoint 非常大。多个 rank 同时写 checkpoint，会制造巨大的突发写入。

### Checkpoint 写入模式

常见模式：

| 模式 | 特点 |
| --- | --- |
| single file | 简单，但大模型不现实 |
| per-rank file | 每个 rank 写自己的 shard，恢复要依赖 metadata |
| sharded checkpoint | 参数/optimizer state 分片保存 |
| async checkpoint | 后台写入，减少阻塞 |
| local staging + remote commit | 先写本地 NVMe，再上传共享存储 |
| incremental checkpoint | 只保存变化部分，复杂度更高 |

分布式 checkpoint 的关键是：

- 写入是否原子可见。
- 最新 checkpoint 指针如何更新。
- rank 文件是否完整。
- metadata 是否和数据 shard 一致。
- 写入失败如何清理。
- 恢复时是否能发现不完整 checkpoint。

### Atomic Latest

常见做法：

```text
checkpoint-000100/
  rank-00000.pt
  rank-00001.pt
  ...
  metadata.json
  _SUCCESS

latest -> checkpoint-000100
```

原则：

- 先写新目录。
- 每个 rank 写自己的文件。
- metadata 写完后再写 `_SUCCESS`。
- 最后原子更新 latest 指针或 manifest。
- 恢复时只读取有 `_SUCCESS` 的 checkpoint。

对象存储没有跨 key 原子事务，所以更要依赖 manifest、success marker 和幂等恢复逻辑。

### Checkpoint 状态机

Checkpoint 最好被实现成状态机，而不是“写完就算”。

一个可解释的状态机可以是：

```text
INIT
  -> WRITING_LOCAL
  -> UPLOADING_REMOTE
  -> VERIFYING
  -> COMMITTED
  -> MARK_LATEST
  -> CLEANUP_OLD
```

失败分支包括：

- `FAILED_LOCAL_WRITE`：本地 NVMe 写满、权限错误、进程退出。
- `FAILED_UPLOAD`：对象存储超时、共享文件系统错误、网络中断。
- `FAILED_VERIFY`：checksum 不匹配、缺 shard、metadata 不一致。
- `STALE_INCOMPLETE`：训练任务已结束，但临时 checkpoint 没清理。

状态机的关键约束：

- 未进入 `COMMITTED` 的 checkpoint 不能被恢复流程读取。
- `latest` 只能指向已验证的 checkpoint。
- metadata 要先描述期望 shard，再由 success marker 表示完成。
- 清理旧 checkpoint 前要确认没有 eval、resume、发布、归档引用。
- 恢复流程要能跳过未完成 checkpoint，并给出明确原因。

这套状态机让 checkpoint 从“文件集合”变成“恢复协议”。尤其在对象存储、多 rank 分片、异步上传和抢占恢复场景下，它比简单保存路径可靠得多。

### Checkpoint 频率

Checkpoint 太频繁：

- step time 增加。
- 存储写入压力大。
- 网络拥塞。
- 影响其他任务。

Checkpoint 太少：

- 故障后 lost work 大。
- 抢占成本高。
- 长任务风险大。

合理频率取决于：

- job failure rate。
- checkpoint 写入时间。
- 训练成本。
- 抢占概率。
- 存储带宽。
- 恢复时间。

一个实用指标：

```text
checkpoint overhead = checkpoint_time / checkpoint_interval
```

例如每 30 分钟 checkpoint 一次，每次阻塞 3 分钟，overhead 就是 10%。这可能无法接受。

## 异步 Checkpoint

异步 checkpoint 的思路是把训练主流程和写入解耦。

典型路径：

1. 训练进程把状态快照到内存或本地 NVMe。
2. 训练继续下一步。
3. 后台线程/进程上传到共享文件系统或对象存储。
4. 上传完成后更新 manifest。

收益：

- 减少训练阻塞。
- 平滑远端存储压力。

风险：

- 占用额外内存或本地 NVMe。
- 后台写入失败要被发现。
- 训练进程退出时要处理未完成写入。
- 恢复时不能引用未完成 checkpoint。
- 多个异步 checkpoint 可能堆积。

异步 checkpoint 不是把问题消灭，而是把阻塞移动到后台，需要可靠状态机。

## 恢复与 Resharding

恢复比保存更难。

要考虑：

- 节点数是否变化。
- GPU 数是否变化。
- TP/PP/DP/EP group 是否变化。
- FSDP/ZeRO shard 是否变化。
- optimizer state 是否能重新分片。
- dataloader 是否从正确位置继续。
- RNG 是否一致。
- checkpoint 是否跨版本兼容。

如果训练从 64 GPU 恢复到 128 GPU，checkpoint 可能需要 resharding。保存时只按 rank 写文件，不记录足够 metadata，会让恢复变得困难。

建议保存：

- global topology。
- parallel config。
- shard metadata。
- tensor shape / dtype。
- model config。
- optimizer config。
- software version。
- checksum。

### Checkpoint Manifest

checkpoint manifest 应该描述“恢复所需的一切事实”，而不是只列文件名。

建议包含：

- run id、global step、epoch、token count。
- model config、tokenizer、vocab、训练代码 commit。
- parallelism config：DP、TP、PP、EP、FSDP/ZeRO。
- world size、rank 数、保存时节点数。
- 每个 tensor shard 的 shape、dtype、offset、checksum。
- optimizer、scheduler、grad scaler、RNG、sampler 状态。
- 数据集版本、shard 顺序、dataloader 进度。
- 保存时的 framework、CUDA、NCCL、driver、容器镜像 digest。
- checkpoint 状态：writing、verified、committed、latest。
- 父 checkpoint 或增量链。

manifest 的作用是让恢复流程先检查可恢复性，再加载大文件。

例如：

- 当前 world size 和保存时不同，是否支持 reshard。
- 模型代码版本是否兼容。
- 所有 rank shard 是否完整。
- optimizer state 是否缺失。
- 数据采样位置是否可恢复。
- checkpoint 是否来自已验证状态。

没有 manifest，恢复失败往往发生在已经下载大量文件之后，排查成本很高。

## 模型权重分发

推理服务扩容时，模型权重分发常成为瓶颈。

一个 100 GB 级别模型，如果 100 个 replica 同时启动，可能同时从对象存储或模型仓库拉取数 TB 数据。

常见优化：

- 节点本地模型缓存。
- 镜像和模型分离。
- 模型 artifact 用 digest 校验。
- 分层加载。
- lazy loading。
- peer-to-peer / registry mirror。
- 预热常用模型。
- 灰度发布时错峰拉取。
- 多版本保留和清理策略。

模型加载影响：

- cold start。
- autoscaling。
- rolling update。
- failure recovery。
- 多租户模型服务成本。

推理平台要把模型权重当作一等存储对象，而不是每个服务自己随便下载。

### 权重分发防雪崩

模型权重分发最怕“同时启动、同时 miss、同时下载”。

典型场景：

- autoscaling 触发大量新 replica。
- 滚动升级同时替换很多节点。
- 热门模型发布新版本。
- 节点维护后大量 cache 清空。
- registry 或对象存储短时抖动，重试风暴放大流量。

防雪崩策略：

| 策略 | 作用 |
| --- | --- |
| 分批发布 | 控制同时拉取权重的 replica 数 |
| 节点级 singleflight | 同一节点只允许一个进程下载同一 digest |
| 分层缓存 | registry/object store -> rack cache -> node cache |
| 预热任务 | 在切流量前把权重拉到目标节点 |
| digest pinning | 保证缓存对象不可变，避免读到半更新版本 |
| 校验后原子切换 | 下载到临时路径，checksum 通过后再暴露 |
| 失败退避 | 避免所有副本同时重试 |
| cache waterline | 磁盘接近水位线时提前清理冷模型 |

权重分发也要有指标：

- download time。
- cache hit rate。
- bytes from remote / local / peer。
- checksum failure。
- concurrent downloads。
- cold start latency。
- rollout duration。
- rollback readiness。

如果推理平台没有权重分发控制，扩容速度会被存储和网络限制，甚至在流量高峰期引发级联故障。

## 容器镜像存储

容器镜像也会影响 AI 集群。

问题包括：

- 镜像很大。
- 每个节点重复拉取。
- CUDA / framework 层变化导致缓存失效。
- registry 带宽不足。
- 镜像漏洞扫描和签名。
- 多团队维护重复镜像。

优化方向：

- 统一 base image。
- 分层稳定依赖。
- 节点预拉取。
- registry mirror。
- 镜像清理。
- digest pinning。
- SBOM 和签名。
- driver 和 CUDA 兼容矩阵。

镜像拉取慢会直接影响调度启动时间和推理扩容速度。

## GPUDirect Storage

GPUDirect Storage (GDS) 让 storage 和 GPU memory 之间有直接 DMA 数据路径，避免经过 CPU bounce buffer。NVIDIA GDS 文档说明，这可以减少 CPU 负载、降低延迟，并缓解系统带宽瓶颈。

GDS 适合：

- GPU 直接消费大块数据。
- 数据处理 pipeline 已经迁移到 GPU。
- 大吞吐顺序 IO。
- local NVMe 或支持 GDS 的分布式文件系统。

但 GDS 不是自动加速所有场景。

需要关注：

- 文件系统是否支持。
- 是否使用 `O_DIRECT` 或满足对齐条件。
- GPU 与 NVMe/NIC 的 PCIe 拓扑。
- IO size 是否足够。
- 是否有 fallback 到 CPU 路径。
- 应用是否使用 cuFile。
- 是否有足够并发 saturate 链路。

GDS 的价值在于减少 CPU 中转和提高直接路径效率，但它仍然需要应用、文件系统、驱动、拓扑和 benchmark 配合。

## Kubernetes 存储抽象

Kubernetes 用 PV/PVC/StorageClass 抽象持久存储。

官方文档把 PersistentVolume 描述为集群中的一块存储资源，可以由管理员静态创建，也可以通过 StorageClass 动态创建；PersistentVolumeClaim 是用户对存储的请求。这个抽象让用户不必直接知道底层 NFS、iSCSI 或云存储细节。

对 AI 来说，Kubernetes 存储要关注：

- access mode。
- volume mode。
- StorageClass。
- CSI driver。
- dynamic provisioning。
- node affinity。
- reclaim policy。
- snapshot / clone。
- performance class。
- quota。

AI workload 常见问题：

- PVC 创建了，但性能不适合训练。
- 多 pod 共享 RWX volume，metadata 压力大。
- checkpoint volume 与 dataset volume 混用。
- pod 被调度到远离存储的节点。
- local PV 生命周期不清晰。
- storage class 名字隐藏了真实性能差异。

所以 AI 平台应该定义清晰的存储类别，例如：

- `dataset-readonly`。
- `checkpoint-fast`。
- `local-nvme-cache`。
- `model-artifact-cache`。
- `logs-archive`。

不要让用户只看到一个泛泛的 `standard` storage class。

## 存储调度契约

存储也要进入调度契约。一个 AI job 不只是需要 GPU，还需要数据、缓存、checkpoint 和模型权重路径满足条件。

调度前应该知道：

| 问题 | 示例 |
| --- | --- |
| 数据在哪里 | object store、parallel FS、local cache、shared cache |
| 需要什么吞吐 | 每 rank 读取 MB/s，整体 GB/s |
| 是否需要预热 | dataset shard、model weight、container image |
| checkpoint 写到哪里 | local staging、parallel FS、object store |
| checkpoint 是否可抢占恢复 | interval、RPO、RTO、resume command |
| 是否需要 RWX / POSIX | 多 pod 共享写、rename、metadata |
| 是否要求数据 locality | 同 rack cache、local NVMe、特定 storage gateway |
| 缓存是否可丢弃 | local cache 可重建，checkpoint 不可丢 |

平台可以把这些信息转成：

- node affinity：优先调度到已有缓存的节点。
- pre-staging job：任务启动前拉数据和模型。
- storage class：选择 checkpoint-fast 或 dataset-hot。
- queue policy：checkpoint-heavy job 错峰运行。
- local NVMe quota：防止单任务占满节点缓存。
- cleanup policy：任务结束后清理临时文件和失败 checkpoint。

如果调度只看 GPU，任务可能拿到了 GPU 但等数据、等模型、等镜像、等 checkpoint。最终表现为 GPU allocation 很高，实际有效吞吐很低。

## 存储可观测性

需要采集：

### 数据集读取

- read throughput。
- open/close rate。
- metadata ops。
- cache hit rate。
- DataLoader wait time。
- H2D copy time。
- GPU idle due to input。

### Checkpoint

- checkpoint duration。
- checkpoint size。
- write bandwidth。
- async backlog。
- failed checkpoint count。
- restore time。
- latest pointer update。
- cleanup status。

### 存储系统

- filesystem metadata latency。
- object store request rate。
- object store error / throttle。
- storage network utilization。
- NVMe utilization。
- disk full。
- inode usage。
- quota usage。

### 推理模型加载

- model download time。
- local cache hit rate。
- model load time。
- cold start latency。
- artifact checksum failure。

没有这些指标，用户看到的是“GPU 利用率低”或“任务启动慢”，平台无法判断到底是存储、网络、CPU 还是代码问题。

## 故障模式与归因

存储问题经常表现为训练慢、GPU 空等、推理冷启动慢，而不是直接报“存储坏了”。

常见故障模式：

| 现象 | 可能原因 | 证据 |
| --- | --- | --- |
| GPU utilization 周期性掉零 | DataLoader、metadata、小文件、CPU decode 慢 | DataLoader wait、open/close rate、CPU profile |
| 首个 epoch 很慢 | cache cold、pre-staging 不足、对象存储限流 | cache hit rate、object GET、远端吞吐 |
| checkpoint 阻塞很久 | 写入带宽不足、metadata 压力、rank 写入不均 | checkpoint duration、rank write time、FS latency |
| 恢复失败 | shard 缺失、manifest 不一致、latest 指向未完成版本 | success marker、checksum、manifest 状态 |
| 推理扩容慢 | 权重下载雪崩、registry 限流、本地缓存 miss | download concurrency、registry QPS、node cache hit |
| 节点磁盘满 | cache 清理失败、临时文件残留、失败 checkpoint 没清理 | NVMe waterline、orphan file、job cleanup log |
| p99 抖动 | KV 分层、RAG 索引读取、模型 lazy load | request timeline、cache miss、storage latency |
| 对象存储成本异常 | 小对象太多、重复下载、cache key 不稳定 | request count、egress、cache miss source |

归因顺序建议：

```text
先确认 workload 阶段
  -> 是读数据、写 checkpoint、拉模型、拉镜像还是查 RAG
  -> 再看本地 cache 命中和远端请求
  -> 再看 metadata / 小文件 / shard 分布
  -> 再看 storage network 和远端服务错误
  -> 最后看应用参数和存储系统调优
```

不要只看“存储带宽”。很多 AI 存储问题不是大文件吞吐不足，而是 metadata、cache miss、并发下载、写入原子性或生命周期清理出了问题。

## Benchmark 方法

存储 benchmark 要贴近 workload。

### Microbenchmark

测：

- 顺序读写带宽。
- 随机读写。
- metadata ops。
- 小文件 open/close。
- object GET/PUT。
- local NVMe bandwidth。
- network filesystem bandwidth。
- GDS direct read/write。

### Dataset Benchmark

测真实数据集：

- shard 数量。
- shard 大小。
- worker 数。
- 每 rank 读取速率。
- shuffle 策略。
- decode / tokenizer 成本。
- cache warm / cold 差异。

### Checkpoint Benchmark

测：

- 单次 checkpoint size。
- 保存时间。
- 恢复时间。
- async checkpoint backlog。
- 多 job 同时 checkpoint。
- checkpoint 失败恢复。
- latest manifest 正确性。

### End-to-End Benchmark

最终看：

- training step time。
- GPU idle time。
- checkpoint overhead。
- restore time objective。
- 推理 cold start。
- model rollout 时间。
- p99 受存储影响程度。

只测 `fio` 不够。`fio` 能测存储设备，但不能代表 DataLoader、tokenizer、shuffle、checkpoint metadata 和多租户干扰。

### Benchmark Manifest

存储 benchmark 也要保存 manifest。

建议记录：

- 数据集版本、shard 数、shard 大小、小文件数量。
- 读取路径：object store、parallel FS、本地 NVMe、shared cache。
- cache 状态：cold、warm、partial warm。
- 每节点、每 rank、每 worker 的并发度。
- DataLoader 参数、tokenizer / decode / augmentation 版本。
- storage class、filesystem、mount option、stripe 配置。
- object store endpoint、bucket、prefix、request rate。
- checkpoint 大小、rank 数、文件数、manifest 状态机。
- driver、kernel、filesystem client、CSI driver 版本。
- 网络拓扑和是否与其他 job 干扰。
- raw metrics：throughput、latency、metadata ops、cache hit、error。

没有 manifest 的存储 benchmark 很难复现。一次测试结果可能只是因为 cache 是热的、节点刚好靠近存储、对象存储没有限流，或者没有其他任务同时 checkpoint。

## 常见优化方向

### 把小文件打包成 Shard

减少 metadata 压力，让读取变成顺序流。

### 本地 NVMe 热缓存

把高频读取的 dataset shard、模型权重、tokenizer 和临时结果放到本地缓存。

### Checkpoint Staging

先写本地 NVMe，再异步上传远端，减少训练主流程阻塞。

### 错峰 Checkpoint

避免多个大训练任务同时 checkpoint。调度系统可以参与 checkpoint window 管理。

### 分离训练、推理、日志和归档存储

不同流量不要全部打到同一个文件系统。

### Manifest 与 Checksum

模型、数据集、checkpoint 都应有 manifest、版本、checksum，避免读到半写入或错误版本。

### 预热模型与数据

对高频模型和数据集做预拉取，降低 cold start 和首 epoch 抖动。

## 常见误区

### 误区一：共享文件系统能解决所有问题

共享文件系统方便，但大量小文件、并发 checkpoint 和多租户扫描会造成 metadata 和吞吐瓶颈。

### 误区二：对象存储可以当 POSIX 文件系统用

对象存储是 bucket/key/object 模型，不是本地文件系统。路径 rename、目录语义、多文件原子更新都不能简单假设。

### 误区三：本地 NVMe 是持久存储

本地 NVMe 适合 cache 和 staging。节点故障或任务迁移后，数据可能不可用，必须有远端持久层。

### 误区四：Checkpoint 越频繁越安全

Checkpoint 太频繁会拖慢训练并冲击存储。要平衡 lost work 和 checkpoint overhead。

### 误区五：异步 checkpoint 没有代价

异步写入需要内存/NVMe、后台状态机、失败检测和恢复逻辑。否则会生成不可用 checkpoint。

### 误区六：只看存储带宽

AI 存储还要看 metadata、cache hit、DataLoader wait、checkpoint overhead、恢复时间、cold start 和多租户干扰。

## 设计检查清单

设计 AI 存储系统时，可以检查：

- 原始数据、训练 shard、checkpoint、模型权重、镜像、日志是否分层。
- 每类对象是否定义生命周期：事实来源、缓存、临时、归档、删除。
- 是否有清晰存储策略契约：dataset-hot、checkpoint-fast、local-cache、model-artifact。
- 小文件是否打包。
- dataset version 和 preprocessing version 是否明确。
- 本地 NVMe cache 是否有容量、淘汰和清理策略。
- cache key 是否包含 dataset、preprocess、tokenizer、model digest。
- cache miss 是否有限流、singleflight 和预热机制。
- checkpoint 是否有 `_SUCCESS` 或 manifest。
- checkpoint 是否实现 writing、verifying、committed、latest 的状态机。
- latest 指针是否原子或幂等。
- 恢复是否能识别不完整 checkpoint。
- 是否支持 sharded checkpoint 和 resharding。
- checkpoint 频率是否基于 lost work 和 overhead 计算。
- 多 job 同时 checkpoint 是否压测过。
- 模型权重是否有 digest、缓存和预热。
- 模型权重分发是否防止发布、扩容、重试时雪崩。
- 镜像是否有 base image 策略和 registry mirror。
- Kubernetes StorageClass 是否表达真实性能类别。
- 存储需求是否进入调度契约，而不是只看 GPU。
- 对象存储是否考虑一致性、权限、版本和生命周期。
- 是否采集 DataLoader wait、checkpoint duration、cache hit、restore time。
- benchmark 是否保存 storage manifest、cache 状态、shard 信息和 raw data。
- 故障归因是否能区分数据读取、checkpoint、模型加载、镜像拉取和 RAG/index 访问。

## 小结

AI 存储设计的核心不是“盘够不够大”，而是：

```text
数据对象
  -> 生命周期
  -> 访问模式
  -> 一致性要求
  -> 性能要求
  -> 放置层次
  -> 缓存策略
  -> 恢复策略
```

好的存储系统会让 GPU 少等数据、训练少等 checkpoint、推理少等模型加载，并在故障后快速恢复。差的存储系统会把所有瓶颈伪装成“GPU 利用率低”。

第 7 章后续的环境可复现、混合集群和成本治理，都需要建立在清晰的数据生命周期和存储分层之上。

## 延伸阅读

- [Kubernetes Persistent Volumes](https://kubernetes.io/docs/concepts/storage/persistent-volumes/)
- [Kubernetes Storage Classes](https://kubernetes.io/docs/concepts/storage/storage-classes/)
- [NVIDIA GPUDirect Storage Overview Guide](https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html)
- [PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)
- [Amazon S3 User Guide](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html)
- [Amazon S3 Performance Guidelines](https://docs.aws.amazon.com/AmazonS3/latest/userguide/optimizing-performance.html)
