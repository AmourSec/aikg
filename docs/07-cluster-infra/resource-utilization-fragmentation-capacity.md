---
title: 资源利用率、碎片与容量治理：从 GPU 分配到有效吞吐
domain: cluster-infra
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-06-12
---

# 资源利用率、碎片与容量治理：从 GPU 分配到有效吞吐

AI 集群最容易被一个指标误导：GPU 利用率。

很多团队会先问：

> 集群 GPU 利用率有多少？

但这个问题本身不够精确。这里的利用率可能指：

- GPU 被分配给任务的比例。
- GPU 上是否有进程。
- SM 是否在忙。
- HBM 带宽是否在跑。
- Tensor Core 是否在跑。
- 训练 step 是否有效推进。
- 推理服务是否在稳定产出 token。
- 用户是否得到了有价值的实验结果。

这些都可以叫“利用率”，但它们表达的含义完全不同。

一个 GPU 被任务占着，不代表它在计算；GPU 在计算，不代表它在做有价值的计算；训练任务在跑，不代表集群资源配置是健康的。如果只盯分配率，集群会很快变成“看起来很忙，实际有效吞吐很低”。

这篇文章讨论的是 AI 集群运营视角下的资源治理：如何理解利用率、碎片、排队、公平性、SLA、容量、成本和能效之间的关系。

## 一张总图

```mermaid
flowchart TB
    Demand["Demand<br/>jobs / services / notebooks / batch"]
    Policy["Policy<br/>quota / priority / SLA / budget"]
    Queue["Queue<br/>pending / admitted / backfill"]
    Allocation["Allocation<br/>GPU / CPU / memory / network / storage"]
    Runtime["Runtime<br/>warmup / run / checkpoint / idle / cleanup"]
    Telemetry["Telemetry<br/>DCGM / scheduler / storage / network / app"]
    Metrics["Metrics<br/>utilization / fragmentation / wait / throughput"]
    Decisions["Decisions<br/>resize / rebalance / autoscale / deprecate"]
    Capacity["Capacity Plan<br/>node pool / reservation / burst / upgrade"]

    Demand --> Policy
    Policy --> Queue
    Queue --> Allocation
    Allocation --> Runtime
    Runtime --> Telemetry
    Telemetry --> Metrics
    Metrics --> Decisions
    Decisions --> Capacity
    Capacity --> Policy
```

这张图表达一个运营闭环：

- 需求进入队列。
- 策略决定谁能用资源。
- 调度器完成分配。
- 运行时产生真实消耗和产出。
- 监控收集事实。
- 指标暴露问题。
- 决策改变配额、节点池、镜像、任务模板和容量计划。

如果没有这个闭环，集群治理就会停留在“资源不够，加机器”或“GPU 利用率低，催用户”的粗糙阶段。

## 利用率不是一个指标

AI 集群至少要区分五种利用率。

| 指标 | 含义 | 容易误判 |
| --- | --- | --- |
| 分配率 | GPU 被 scheduler 分配出去的比例 | 分配了不代表在计算 |
| 驻留率 | GPU 上有进程或显存占用的比例 | 显存占用不代表 SM 忙 |
| 活跃率 | GPU SM、Tensor Core、HBM 正在工作的比例 | 活跃不代表任务有效推进 |
| 有效吞吐 | step/token/sample/query 等业务产出 | 需要应用指标配合 |
| 价值利用率 | 单位成本产出的有效训练/推理结果 | 最难测，但最接近治理目标 |

举例：

```text
GPU allocation rate: 95%
GPU SM utilization: 35%
training step efficiency: 20%
```

这说明 GPU 大部分时间被任务占住，但任务实际推进慢。原因可能是：

- DataLoader 等数据。
- checkpoint 卡住。
- 通信同步等待。
- CPU tokenization 成为瓶颈。
- 小 batch 导致 GPU 不饱和。
- 多租户干扰导致网络或存储抖动。
- Notebook 占着 GPU 但很少计算。

所以“GPU 分配率高”不是好消息，也不是坏消息。它只是第一个信号。

## 四本账：容量、需求、使用和产出

资源治理不能只看监控曲线，还要建立“账本”。账本的目标是让平台能回答：资源有多少、谁要用、实际怎么用了、产出了什么。

| 账本 | 记录什么 | 典型问题 |
| --- | --- | --- |
| Capacity Ledger | 集群真实可提供的资源 | 哪些 GPU 健康、哪些节点 drain、哪些 flavor 可调度 |
| Demand Ledger | 用户和服务提出的资源需求 | 哪些队列长期排队、需要什么 GPU、gang size 多大 |
| Usage Ledger | 资源实际被谁占用和消耗 | 谁用了多少 GPU hour、是否 idle、是否被抢占 |
| Output Ledger | 资源产生了什么结果 | 训练了多少 token、推理服务了多少请求、成功实验多少 |

四本账要能通过统一字段关联：

```text
tenant / project / queue / workload_type / node_pool / gpu_flavor / image / model / dataset
```

例如一次训练任务同时出现在四本账里：

- Capacity Ledger：它使用了 `h100-training` node pool 的 64 张 H100。
- Demand Ledger：它请求 64 GPU、预计 36 小时、允许抢占。
- Usage Ledger：它实际运行 41 小时，中间失败重启 1 次。
- Output Ledger：它完成 2.3T token，产出 checkpoint 和 eval 结果。

如果只有 Usage Ledger，就会把失败重试也算作“使用”；如果只有 Output Ledger，就不知道它占用了多少资源；如果只有 Capacity Ledger，就无法判断资源是否给了高价值任务。

AI 集群的治理成熟度，往往取决于这四本账能否对齐。

## 从分配到有效吞吐

可以把 GPU 资源使用分成一条链：

```text
available GPU
  -> schedulable GPU
  -> allocated GPU
  -> process resident GPU
  -> active GPU
  -> productive GPU
  -> valuable output
```

每一层都会损耗。

### Available GPU

物理存在且健康的 GPU。

排除：

- 故障 GPU。
- drain 节点。
- driver 异常节点。
- 被维护窗口占用的节点。
- 被系统保留的节点。

### Schedulable GPU

调度器认为可以分配的 GPU。

排除：

- taint 不允许使用。
- 节点池不匹配。
- MIG 切分形态不匹配。
- GPU 型号不满足任务要求。
- CPU/memory/NVMe 不足导致 GPU 无法被调度。

### Allocated GPU

已经被 job 或 service 占用的 GPU。

这是很多集群报表里的“利用率”，但它只能说明资源被占住。

### Resident GPU

任务在 GPU 上有进程、上下文或显存占用。

Notebook 经常停在这一层：显存占着，SM 不忙。

### Active GPU

GPU 真的在执行 kernel、访问 HBM 或通信。

这一层需要 DCGM、NVML、框架 profiler 或系统 telemetry。

### Productive GPU

GPU 活跃时间里，有多少在推动目标 workload。

例如训练里：

- forward/backward/optimizer 是 productive。
- DataLoader wait 不是。
- checkpoint blocking 不是。
- NCCL 等慢 rank 不完全是。
- 失败后重跑的一部分可能不是。

推理里：

- prefill/decode 是 productive。
- 模型冷加载不是。
- 空 batch 等待不是。
- 被取消请求的计算价值较低。

### Valuable Output

最终产出：

- 训练 token。
- 有效 step。
- eval score。
- 推理 token。
- 成功请求。
- 构建好的 embedding index。
- 通过验收的模型版本。

这才是资源治理真正想提升的东西。

## 核心指标体系

### 资源供给指标

资源供给回答“集群能提供什么”。

| 指标 | 说明 |
| --- | --- |
| total GPU | 物理 GPU 总量 |
| healthy GPU | 健康可用 GPU |
| schedulable GPU | 调度器可分配 GPU |
| GPU by type | H100/A100/L40S/MI300 等型号分布 |
| GPU by node pool | train/infer/dev/batch/system |
| CPU/GPU ratio | 每 GPU 对应 CPU 核数 |
| memory/GPU ratio | 每 GPU 对应 host memory |
| local NVMe/GPU ratio | 每 GPU 对应本地缓存容量 |
| network bandwidth/GPU | 每 GPU 对应网络能力 |

AI 集群不能只统计 GPU 数量。GPU 很强但 CPU、内存、网络或存储不足，会形成“假容量”。

### 资源需求指标

资源需求回答“用户想要什么”。

| 指标 | 说明 |
| --- | --- |
| submitted jobs | 提交任务数 |
| requested GPU hours | 请求的 GPU 小时 |
| requested GPU type | 请求的 GPU 型号 |
| requested gang size | 需要同时分配的 GPU 数 |
| requested duration | 用户预估或历史推断运行时间 |
| queue wait time | 排队等待 |
| pending reason | 等待原因 |
| deadline / SLA | 任务时限 |

大训练、短微调、Notebook、推理服务的需求曲线完全不同。如果只看总 GPU 需求，会掩盖结构性问题。

### 资源使用指标

资源使用回答“资源实际怎么被用掉”。

| 指标 | 说明 |
| --- | --- |
| allocation rate | GPU 被分配比例 |
| SM utilization | GPU 计算单元活跃程度 |
| HBM utilization | 显存带宽使用 |
| GPU memory used | 显存占用 |
| power draw | 功耗 |
| PCIe/NVLink/RDMA traffic | 互连与网络流量 |
| CPU utilization | CPU 使用 |
| memory pressure | host memory 压力 |
| storage throughput | 数据和 checkpoint 吞吐 |
| checkpoint duration | checkpoint 时间 |
| job failure/retry | 失败和重试 |

NVIDIA DCGM exporter 常用于把 GPU telemetry 暴露给 Prometheus；Kubernetes resource metrics pipeline 主要提供 CPU/memory 等资源指标，不能替代 GPU、网络、存储和应用层 telemetry。

### 产出指标

产出指标回答“这些资源产生了什么”。

| Workload | 产出指标 |
| --- | --- |
| 预训练 | tokens/sec、step time、MFU、loss 曲线 |
| 微调 | samples/sec、epoch time、eval score |
| 推理 | tokens/sec、requests/sec、p50/p95/p99、错误率 |
| 离线推理 | samples/hour、cost/sample |
| 数据预处理 | files/sec、GB/hour、shard 产出 |
| RAG index | vectors/sec、index build time、query recall |
| Notebook | active session、idle time、GPU active ratio |
| Benchmark | 固定 workload 的可比较吞吐 |

没有产出指标，就无法判断高利用率是否有意义。

## 一个实用分解

可以用几个简单比例定位问题：

```text
allocation_util = allocated_gpu_hours / schedulable_gpu_hours
active_util = active_gpu_time / allocated_gpu_time
productive_util = productive_time / active_gpu_time
useful_output_efficiency = useful_output / allocated_gpu_hours
```

不同组合代表不同问题。

| 现象 | 可能含义 |
| --- | --- |
| allocation 高，active 低 | 占用但不计算，Notebook idle、数据瓶颈、任务卡住 |
| allocation 低，pending 高 | 碎片、约束过强、gang 调度等待 |
| active 高，throughput 低 | 通信、低效 kernel、batch 太小、memory bound |
| throughput 高，SLA 差 | 平均吞吐好但尾延迟差 |
| utilization 高，成本高 | 高价值任务和低价值任务混在一起，需要成本归因 |

这套分解的价值是：不要把所有问题都归结成“GPU 利用率低”。

## GPU Hour Ledger 与浪费分类

GPU hour 是成本和容量规划中最常用的单位，但它必须拆开看。否则失败、空转、等待和有效计算会混在一起。

一个实用拆分是：

```text
allocated_gpu_hours
  = productive_gpu_hours
  + idle_gpu_hours
  + waiting_gpu_hours
  + failed_gpu_hours
  + overhead_gpu_hours
  + system_reserved_gpu_hours
```

各类含义：

| 类型 | 含义 | 常见原因 |
| --- | --- | --- |
| productive | 推动训练、推理、数据处理等目标产出 | forward/backward、decode、embedding、index build |
| idle | GPU 被占用但几乎不计算 | Notebook idle、任务 hang、等待人工操作 |
| waiting | 任务已启动但在等外部资源 | DataLoader、存储、网络、慢 rank、模型加载 |
| failed | 最终失败或被废弃的计算 | OOM、环境错误、节点故障、训练不稳定 |
| overhead | 必要但不直接产出的开销 | checkpoint、编译、warmup、启动、恢复 |
| reserved | 为 SLA 或系统稳定保留 | 推理 headroom、benchmark 池、维护窗口 |

治理动作应该针对不同浪费类型：

- idle 高：做 Notebook idle 回收和 hang 检测。
- waiting 高：查 CPU、存储、网络、rank skew。
- failed 高：查环境、节点健康、OOM、数值稳定性。
- overhead 高：优化镜像、缓存、checkpoint、编译和模型加载。
- reserved 高：评估 headroom 是否过大，或是否服务于明确 SLA。

“浪费”不是绝对坏事。headroom 和 checkpoint 都可能是必要成本。关键是把它们从 productive 中分出来，避免用一个总利用率掩盖真实问题。

## 资源碎片

碎片是 AI 集群最常见的隐性损耗。

### 节点内碎片

例如 8 卡节点上只剩 1 张 GPU。小任务可以跑，但 8 卡训练任务无法调度。

碎片来源：

- 小任务随意占用大节点。
- CPU/memory request 不匹配。
- 本地 NVMe 已满。
- MIG 切分形态不匹配。
- GPU 与 NIC/NUMA 拓扑不匹配。

### 节点间碎片

总 GPU 数够，但分散在不同 rack、不同网络域或不同节点池里，不能组成一个拓扑一致的大 job。

例如：

```text
free GPU total: 64
needed: 64 GPU, same high-speed fabric island
actual: 8 islands x 8 free GPU, network topology not suitable
```

这时“空闲 GPU 总数”会误导容量判断。

### 型号碎片

不同 GPU 型号混杂：

- H100。
- A100 80GB。
- A100 40GB。
- L40S。
- 推理卡。
- MIG 实例。

如果任务只接受一种 flavor，其他 GPU 就是不可用容量。

### 配额碎片

资源在不同 team/project quota 中静态切分。某个团队空闲，另一个团队排队，但资源不能借用。

解决方式通常是：

- guaranteed quota。
- borrowable quota。
- fairshare。
- preemption。
- queue-level resource flavor。

### 时间碎片

短任务和长任务混合时，调度器可能为了等待大 job 而保留资源，也可能为了提高即时利用率不断塞小任务，导致大 job 长期无法启动。

这就是 backfill、gang scheduling 和 deadline-aware scheduling 要处理的问题。

## 碎片指标

可以从多个角度度量碎片。

### Free-but-unusable GPU

```text
unusable_free_gpu = free_gpu - gpu_that_can_satisfy_pending_jobs
```

这比简单的 `free_gpu` 更有意义。

### Largest Contiguous Allocation

能立即分配给单个 job 的最大连续资源。

例如：

```text
total free: 128 GPU
largest contiguous allocation: 32 GPU
```

如果队列里有多个 64 GPU 任务，集群看似空闲，实际不可用。

### Pending Reason 分布

统计 pending 原因：

- GPU 不足。
- 特定 GPU 型号不足。
- CPU/memory 不足。
- 节点亲和性不满足。
- quota 不足。
- gang size 无法满足。
- topology 不满足。
- storage mount 失败。
- image pull 慢。

Kueue、Slurm、Kubernetes 的事件和队列状态都可以提供这类信号。关键是要把它们转成可聚合的报表，而不是让用户自己读日志。

### Fragmentation Ratio

一个简单定义：

```text
fragmentation_ratio = 1 - schedulable_for_pending / total_free
```

它不一定适合所有场景，但能提醒你：空闲资源不等于可用资源。

## 碎片治理 Playbook

碎片不是一个单一问题，不能只靠“提高利用率”解决。需要先判断碎片类型，再选择动作。

| 碎片类型 | 现象 | 治理动作 |
| --- | --- | --- |
| 节点内碎片 | 8 卡节点剩 1-2 张卡，大 job 进不来 | 小任务优先放 dev/batch 池，完整节点优先给大任务 |
| CPU/GPU 碎片 | GPU 空闲但 CPU/memory 不足 | 资源模板 right-sizing，限制 CPU 超配，补齐内存节点 |
| 拓扑碎片 | 总 GPU 够，但不在同一 rack/fabric | topology-aware scheduling，保留 fabric island，限制随意放置 |
| Flavor 碎片 | 某型号排队，其他型号空闲 | 扩大兼容 flavor，建立替代规则，避免过度指定型号 |
| MIG 碎片 | MIG instance 形态不匹配 | 统一 MIG profile，周期性重切，避免任意切分 |
| 配额碎片 | 某 team 空闲，另一个 team 排队 | borrowing、cohort、fairshare、可解释抢占 |
| 时间碎片 | 大 job 一直等，短任务不断插入 | backfill with deadline，reservation，gang-aware admission |

治理要注意两点：

1. 不要为了短期分配率牺牲长期可调度性。例如把 8 卡节点塞满零散小任务，可能让下一个 64 卡训练等很久。
2. 不要把所有大节点都保留给大任务。否则短任务体验会恶化，空闲窗口也不能被利用。

更好的做法是给调度器明确偏好：

```text
large gang job: prefer full nodes / same topology domain
small job: prefer fragmented nodes / dev pool / smaller GPU flavor
notebook: prefer preemptible fragmented capacity
benchmark: require clean dedicated node pool
```

这样碎片治理就从人工搬任务，变成可配置的放置策略。

## 排队指标

排队是资源不足、策略不合理和碎片的综合结果。

核心指标：

- queue length。
- queue wait time p50/p95/p99。
- wait time by queue。
- wait time by workload type。
- wait time by GPU type。
- wait time by gang size。
- admitted vs pending。
- backfill 成功率。
- preemption 次数。
- starvation task 数量。

平均等待时间很容易误导。一个队列可能平均等待 10 分钟，但 512 GPU 训练任务等待 3 天。AI 集群应该按 job size 和 workload type 拆分排队指标。

## 队列健康度

队列治理不应该只看 queue length。一个健康队列应该同时满足：

- 可解释：用户知道为什么 pending。
- 可预测：等待时间有合理估计。
- 可恢复：被抢占或失败后能重新进入队列。
- 不饥饿：低优先级任务也不会无限期等待。
- 不阻塞：大 job 不会因为零散小任务永远无法启动。

建议增加这些队列指标：

| 指标 | 说明 |
| --- | --- |
| admission latency | 从提交到被队列接受的时间 |
| scheduling latency | 从 admitted 到开始运行的时间 |
| pending reason age | 同一个 pending reason 持续多久 |
| starvation age | 最老 pending job 等了多久 |
| deadline miss rate | 有 deadline 的任务超时比例 |
| requeue count | 被抢占、失败或重试后重新排队次数 |
| backfill useful time | backfill 任务实际利用了多少空隙 |
| reservation waste | reservation 留出的资源有多少没被用上 |

一个实用规则是：

```text
queue length tells pressure
pending reason tells bottleneck
wait time distribution tells user pain
starvation age tells fairness failure
```

当 pending reason 长期集中在 `insufficient GPU`，可能是真容量不足；如果集中在 `quota`，可能是配额策略问题；如果集中在 `topology`，则是碎片和放置问题。

## 公平性指标

公平性不是“每个人一样多”，而是资源分配和组织目标一致。

常见公平性指标：

- 每个 team 的 GPU hour。
- 每个 project 的 quota 使用率。
- guaranteed quota 满足率。
- borrowed quota 使用量。
- fairshare score。
- 每个 team 的等待时间。
- 每个 team 被抢占次数。
- 每个 workload type 的资源占比。

公平性要结合时间窗口：

```text
1 hour: 当前是否拥塞
1 day: 日常使用是否平衡
1 week: 项目是否持续超用
1 month: 成本和预算归因
```

短时间内不公平可能是合理的，因为大训练需要一次性拿到很多 GPU；长期不公平才需要调整策略。

## Request Right-Sizing

资源请求不准，会直接制造碎片和低效。

常见问题：

- CPU request 过低，DataLoader 或 tokenizer 拖慢 GPU。
- CPU request 过高，让节点 CPU 先满，GPU 调不出去。
- memory request 过低，任务 OOM 或频繁 page cache 抖动。
- memory request 过高，造成虚假碎片。
- shared memory 没设够，训练或推理数据管道异常。
- GPU request 过大，小任务占完整节点。
- 本地 NVMe 未声明，运行后把节点缓存打满。

right-sizing 不是让用户手工猜，而是基于历史运行给出建议：

| 信号 | 建议动作 |
| --- | --- |
| GPU active 低、CPU 满 | 提高 CPU/GPU 比例或优化数据管道 |
| GPU memory 长期低于 30% | 建议更小 GPU/MIG/更小 batch 规格 |
| CPU request 高但实际低 | 下调默认 CPU request，释放可调度容量 |
| OOM/restart 多 | 提高 memory request 或检查 batch/loader |
| checkpoint 挤爆本地盘 | 显式声明 ephemeral storage/NVMe |
| 大量小任务占大节点 | 引导到小 GPU、MIG 或 dev pool |

Kubernetes 的 request/limit 决定调度和资源隔离的基础；Slurm 的 job request 同样决定排队、优先级和资源占用。AI 平台应该提供标准规格，而不是让每个用户从零填写。

示例规格：

```text
notebook-small: 1 GPU, 8 CPU, 64Gi memory, preemptible
finetune-8g: 8 GPU, 128 CPU, 1Ti memory, checkpoint required
train-64g: 64 GPU, topology aware, same fabric island preferred
infer-prod: GPU + CPU reserved, no best-effort, fixed node pool
```

规格越统一，调度器越容易做 bin packing、排队预测和容量规划。

## SLA 指标

不同 workload 的 SLA 不同。

### 推理服务

关注：

- p50/p95/p99 latency。
- time to first token。
- tokens/sec。
- request error rate。
- timeout rate。
- cold start time。
- model loading time。
- batch queue delay。

推理服务的 SLA 经常和低优先级混部冲突。平均 GPU 利用率提升 10%，如果 p99 latency 翻倍，通常不值得。

### 训练任务

关注：

- queue wait time。
- time to first step。
- step time。
- tokens/sec。
- MFU。
- checkpoint duration。
- failure recovery time。
- successful run rate。

训练 SLA 不一定是实时延迟，而是周转时间和可恢复性。

### Notebook

关注：

- session 启动时间。
- idle GPU 时间。
- interactive latency。
- 最大连续运行时间。
- 从 Notebook 转 batch 的比例。

Notebook 的治理重点是减少“占而不用”。

## 容量治理

容量治理不是简单地看平均利用率。

需要回答：

```text
现有容量能满足哪些 workload？
哪些需求长期排队？
瓶颈是 GPU、CPU、内存、网络还是存储？
新增资源应该买什么型号？
是扩训练池、推理池、dev 池，还是优化调度？
空闲资源是否真的可用？
```

### Demand Model

一个基础模型：

```text
demand = arrival_rate * requested_gpu * expected_duration
```

但 AI workload 还要考虑：

- gang size。
- GPU type。
- topology。
- checkpoint 周期。
- job failure rate。
- peak/off-peak。
- deadline。
- priority。

同样 1024 GPU hours，可能是：

- 1 个 1024 GPU 任务跑 1 小时。
- 128 个 8 GPU 任务跑 1 小时。
- 1 个 8 GPU 任务跑 128 小时。

它们对调度和碎片的压力完全不同。

### Headroom

集群不能追求 100% 分配率。

需要保留 headroom：

- 推理突发。
- 失败重试。
- 高优先级任务。
- 节点维护。
- checkpoint 高峰。
- 数据重处理。
- 大 job gang allocation。

headroom 不是浪费，而是 SLA 和稳定性的成本。

### Capacity by Flavor

容量必须按 flavor 管理：

```text
H100-SXM-80GB
A100-SXM-80GB
A100-PCIe-40GB
L40S
MIG-1g.10gb
CPU-only high-memory
NVMe-heavy
network-heavy
```

如果只看总 GPU 数，很容易把不可替代的 H100 需求用普通 GPU 空闲量掩盖。

### 可用容量状态机

容量不是只有“可用/不可用”。建议把 GPU 或节点容量分成状态：

```text
physical
  -> healthy
  -> schedulable
  -> allocatable
  -> usable_for_workload
  -> productive
```

对应解释：

| 状态 | 含义 |
| --- | --- |
| physical | 资产上存在 |
| healthy | 通过健康检查，driver/GPU/NIC 正常 |
| schedulable | 没有 drain/taint/维护限制，调度器可见 |
| allocatable | CPU、memory、storage、GPU flavor 组合可分配 |
| usable_for_workload | 满足某个 workload 的拓扑、环境、权限和队列要求 |
| productive | 运行后产生有效产出 |

容量报表应该同时展示这些状态。例如：

```text
H100 physical: 2048
healthy: 2016
schedulable: 1968
allocatable now: 812
usable for 256-GPU training: 512
largest contiguous allocation: 128
```

这比“还有 812 张空闲 GPU”更接近真实容量。

## 成本指标

成本治理要从“花了多少钱”走向“每单位有效产出多少钱”。

常见指标：

| 指标 | 含义 |
| --- | --- |
| cost / GPU hour | 单位 GPU 时间成本 |
| cost / training token | 训练单位 token 成本 |
| cost / successful run | 成功实验成本 |
| cost / inference token | 推理 token 成本 |
| cost / request | 单请求成本 |
| cost / eval point | 评测成本 |
| wasted cost | 失败、idle、重试、无效 checkpoint 成本 |

成本归因维度：

- team。
- project。
- workload type。
- queue。
- node pool。
- GPU type。
- model。
- dataset。
- environment image。
- priority class。

成本报表不能只按 namespace 或账号聚合。AI 系统更需要看到“哪个模型、哪个实验、哪个服务、哪个数据流程”在消耗资源。

## Showback 与 Chargeback

资源成本展示可以分两种：

- Showback：把成本展示给团队，让大家知道资源消耗。
- Chargeback：把成本真正计入预算或账单。

研究平台早期通常先做 showback，避免一开始就让成本规则阻碍探索；生产平台或大规模共享集群则需要逐步进入 chargeback。

建议至少展示：

| 维度 | 指标 |
| --- | --- |
| Team / Project | GPU hour、成本、等待时间、失败成本 |
| Workload Type | train、infer、notebook、batch、benchmark |
| Model / Dataset | 哪些模型和数据流程最耗资源 |
| Queue / Priority | 高优先级资源是否被滥用 |
| Waste Class | idle、failed、waiting、overhead、reserved |
| Output | token、request、checkpoint、eval、index |

Chargeback 要谨慎处理几个问题：

- 失败重试成本算谁。
- 平台故障导致的浪费是否计入用户。
- headroom 和系统保留成本如何分摊。
- 共享模型缓存、镜像缓存、数据缓存如何归因。
- 抢占造成的重算如何归因。

如果成本规则不透明，用户会为了省账面成本做出错误行为，例如低估 request、绕过 checkpoint、隐藏任务类型。成本治理必须和准入、模板、观测一起设计。

## 能效指标

AI 集群能效不只是数据中心 PUE，也包括 workload 级别的有效计算。

常见指标：

- GPU power draw。
- energy / training token。
- energy / inference token。
- energy / request。
- tokens per joule。
- step time per watt。
- idle power。
- cooling headroom。
- thermal throttling。

功耗指标和利用率要一起看：

| 现象 | 含义 |
| --- | --- |
| 高功耗、高吞吐 | 可能正常 |
| 高功耗、低吞吐 | 可能 kernel、通信、数据瓶颈 |
| 低功耗、GPU 分配高 | 任务占着但没跑 |
| 功耗周期性锯齿 | 可能数据读取或 checkpoint 周期造成 |
| 频率下降 | 可能温度、电源或功耗限制 |

能效治理不是简单降频，而是提高单位能耗的有效产出。

## 可观测性实现

### Kubernetes

Kubernetes resource metrics pipeline 提供 CPU、内存等资源指标，常用于 autoscaling 和 `kubectl top`。但 AI 集群还需要：

- GPU metrics。
- queue metrics。
- object state metrics。
- storage metrics。
- network metrics。
- application metrics。

常见组件包括：

- Metrics Server：基础 CPU/memory 指标。
- kube-state-metrics：Kubernetes 对象状态。
- DCGM exporter：GPU telemetry。
- CNI / network exporter：网络指标。
- CSI / storage exporter：存储指标。
- scheduler / queue metrics：调度和排队指标。
- application metrics：训练和推理产出。

### DCGM Exporter

NVIDIA DCGM exporter 可以把 GPU 指标暴露给 Prometheus，包括利用率、显存、功耗、温度、错误等。它适合做集群 GPU 监控，但仍要和应用指标结合。

例如：

```text
DCGM says GPU busy
training says tokens/sec low
storage says read latency high
```

这三者合在一起才说明：GPU 忙不等于训练有效。

### Kueue

Kueue 提供队列、准入、资源 flavor 和 workload 相关指标。它适合观察：

- workload 是否 admitted。
- queue 是否拥塞。
- resource flavor 是否不足。
- pending workload 分布。
- borrowing 和 quota 使用。

这些指标能把“为什么任务还没跑”从用户感受转成平台信号。

### Slurm

Slurm 常用：

- `squeue` 看当前队列。
- `sacct` 看历史 accounting。
- `sinfo` 看节点和 partition。
- `sshare` 看 fairshare。
- `sstat` 看运行中 job 指标。

Slurm accounting 对长期资源归因很重要，但 GPU active utilization、HBM、功耗等仍需要 DCGM 或其他 GPU telemetry 补齐。

## 指标数据质量

资源治理依赖指标，但指标本身也会出问题。常见数据质量问题包括：

- GPU 指标缺卡、缺节点或标签不一致。
- scheduler 记录的 allocation 和实际进程驻留时间不一致。
- job 名字、team、project、queue 标签缺失。
- Notebook 长期运行但没有 workload type。
- 推理服务指标没有 model/version 标签。
- 训练任务没有 run_id，无法和 checkpoint、日志、成本关联。
- 节点时钟不同步，导致时间窗口错位。
- 采样间隔太粗，漏掉短任务和尖峰。
- failed job 没有记录失败原因。

建议给所有 workload 统一注入标签：

```text
tenant
project
queue
workload_type
run_id
image_digest
model_id
dataset_id
node_pool
gpu_flavor
priority_class
```

同时要建立指标对账：

| 对账项 | 目的 |
| --- | --- |
| scheduler allocation vs DCGM process/GPU active | 找出占用但不计算、或监控缺失 |
| queue admitted time vs pod/job start time | 找出调度和启动延迟 |
| job runtime vs application step/token logs | 找出启动、等待、失败和恢复开销 |
| cost ledger vs usage ledger | 避免成本漏算或重复计算 |
| node inventory vs telemetry targets | 发现监控采集缺口 |

没有数据质量治理，dashboard 会给人一种“看起来很全”的错觉，但真正排障时却缺少关键字段。

## Dashboard 设计

AI 集群 dashboard 不应该只有一张 GPU 利用率图。

建议分层：

### Executive View

面向管理和容量决策：

- total GPU hours。
- effective utilization。
- cost by team/project。
- wait time by queue。
- SLA violation。
- node pool health。
- trend and forecast。

### Operator View

面向平台运维：

- unhealthy nodes。
- pending reason。
- fragmentation。
- scheduler latency。
- image pull failure。
- storage/network saturation。
- GPU error。
- checkpoint spikes。

### Research View

面向用户和课题组：

- 我的 quota。
- 我的 waiting jobs。
- 我的 GPU hour。
- 我的 idle notebook。
- 我的失败重试。
- 推荐队列和资源规格。

### Workload View

面向单个任务：

- queue wait。
- startup time。
- step time / tokens/sec。
- GPU SM/HBM/power。
- DataLoader wait。
- communication time。
- checkpoint duration。
- failure/restart。

不同视图对应不同问题。把所有指标堆在一张图上，通常谁也看不懂。

## 告警策略

告警要避免“利用率低就报警”的粗糙规则。

更有价值的告警：

- schedulable GPU 突然下降。
- unhealthy GPU 增多。
- pending time p95 超阈值。
- 某队列长期 starvation。
- fragmentation ratio 高。
- 推理 p99 与低优先级混部相关。
- checkpoint duration 异常升高。
- DataLoader wait 大面积升高。
- storage metadata rate 异常。
- network retransmit / congestion 异常。
- GPU 分配高但功耗和 SM utilization 低。
- system node pool 资源压力。

告警要能指向动作：

```text
drain node
rebalance queue
increase quota
disable low-priority backfill
investigate storage
roll back image
expand node pool
```

没有动作的告警会变成噪声。

## 常见治理动作

### 清理 Idle

适合 Notebook 和 dev job：

- idle timeout。
- GPU active ratio 低于阈值提醒。
- 超时自动释放。
- 长任务转 batch。

### 调整 Request

很多任务 request 过大或过小：

- CPU request 太低导致 DataLoader 慢。
- memory request 太低导致 OOM。
- GPU request 太大导致浪费。
- shared memory 不足导致 dataloader/推理问题。

可以基于历史运行推荐资源规格。

### Backfill

用短任务填补大任务等待期间的空隙。

要求：

- 任务时长可估计。
- 可抢占或短时完成。
- 不破坏大 job 的 gang allocation。

### Defragmentation

通过驱逐、迁移或等待策略减少碎片。

适用于：

- 大 job 长期 pending。
- 小 job 占据大节点。
- MIG 形态不匹配。
- 多节点拓扑被打散。

### Queue Rebalancing

根据需求和等待时间调整：

- guaranteed quota。
- borrow limit。
- priority。
- node pool mapping。
- fairshare weight。

### Node Pool 调整

当某类资源长期短缺，而其他资源长期空闲，应调整 node pool：

- 推理池扩容。
- dev 池缩容。
- benchmark 池固定。
- H100 和 A100 分层。
- NVMe-heavy 节点给数据任务。

## 治理闭环：从指标到动作

资源治理最终要形成闭环，而不是停在报表。

一个实用流程：

```text
detect
  -> classify
  -> attribute
  -> decide
  -> act
  -> verify
```

示例：

| 信号 | 分类 | 归因 | 动作 | 验证 |
| --- | --- | --- | --- | --- |
| 8 卡训练长期 pending | 碎片 | 小任务占满大节点 | 调整小任务 node affinity，启用 backfill 限制 | largest contiguous allocation 上升 |
| GPU 分配高但 SM 低 | idle/waiting | Notebook idle、DataLoader 慢 | idle 回收、提高 CPU/GPU、优化数据路径 | active_util 上升 |
| 推理 p99 上升 | SLA | 与 batch 混部和模型加载冲突 | 暂停低优先级 backfill，增加推理 headroom | p99 回落 |
| checkpoint duration 暴涨 | 存储瓶颈 | 多 job 同时写共享 FS | checkpoint 错峰、异步保存、本地 staging | checkpoint p95 下降 |
| 某 team 长期等待 | 公平性 | quota 过小或 borrowing 不足 | 调整 guaranteed/borrow limit | wait time p95 下降 |

每个治理动作都要有回看指标，否则无法判断它是优化还是只是移动问题。

## 自动化策略

随着集群变大，纯人工治理会失效。可以逐步引入自动化，但要避免直接让自动化改动高风险策略。

适合自动化的动作：

- Notebook idle 提醒和回收。
- 长期 pending reason 聚合。
- 资源 request 推荐。
- failed job 原因分类。
- image pull / startup 慢任务标记。
- GPU idle 但显存占用的任务告警。
- 低优先级 backfill 开关建议。

需要人工审批或灰度的动作：

- 调整队列 quota。
- 改 priority 和 preemption 策略。
- 改 node pool 归属。
- 驱逐运行中的大训练。
- 改 production inference headroom。
- 调整采购或扩容计划。

自动化的第一目标不是“自动做决定”，而是把事实整理清楚，让平台团队更快做正确决定。

## 容量规划方法

容量规划可以按三步做。

### 1. 记录历史需求

至少记录：

- job type。
- requested GPU。
- GPU type。
- duration。
- wait time。
- success/failure。
- preemption。
- output throughput。
- cost。

### 2. 建立需求画像

按 workload type 建模：

```text
large training: needs gang, topology sensitive, long duration
fine-tuning: medium GPU, frequent, deadline moderate
notebook: interactive, idle heavy, preemptible
online inference: SLA strict, bursty
batch inference: throughput-oriented, preemptible
data pipeline: storage-heavy, often CPU-bound
```

### 3. 做情景分析

典型问题：

- 如果新增 20% 推理流量，推理池是否够。
- 如果启动一个 1024 GPU 训练，其他队列等待多久。
- 如果一个 rack 维护，哪些任务受影响。
- 如果低优先级任务可抢占，能释放多少 headroom。
- 如果 H100 需求增长，A100 空闲能否替代。

容量规划不是一次性表格，而是持续更新的模型。

## 采购与扩容决策

容量规划最终会进入采购、租用或扩容决策。这里最容易犯的错误是：看到 GPU 排队，就直接买同一种 GPU。

更稳妥的决策问题是：

| 问题 | 可能结论 |
| --- | --- |
| 排队是否由某个 flavor 造成 | 买对应 flavor，或放宽兼容规则 |
| 排队是否由 gang/topology 造成 | 增加同构 rack/fabric island，而不是零散加卡 |
| GPU 空闲但作业慢 | 先补 CPU、内存、存储、网络或优化数据管道 |
| 推理 p99 不稳 | 增加 headroom、模型缓存、路由能力，而不只是 GPU |
| Notebook idle 高 | 先做治理和回收，不急着扩容 dev 池 |
| checkpoint 挤占训练 | 扩存储带宽或调整 checkpoint 策略 |
| H100 排队、A100 空闲 | 判断 workload 是否真的需要 H100 |

扩容方案也要按用途拆分：

```text
training scale-out: topology-consistent GPU islands
inference scale-out: low-jitter nodes + local model cache
dev capacity: preemptible / smaller GPU / MIG
storage-heavy jobs: CPU + memory + NVMe + network
benchmark: small but stable dedicated pool
```

买更多 GPU 只是容量治理的一种动作，不是默认答案。

## 常见误区

### 误区一：GPU 分配率越高越好

分配率高但有效吞吐低，说明资源被占住却没有产出。

### 误区二：空闲 GPU 就等于浪费

有些 headroom 是为了推理 SLA、高优先级任务、维护和大 job gang allocation。

### 误区三：平均利用率能代表集群健康

平均值会掩盖队列、型号、节点池和租户之间的结构性问题。

### 误区四：只买更多 GPU 就能解决排队

如果瓶颈是 CPU、存储、网络、碎片、调度策略或环境启动，新增 GPU 可能只是制造更多空转。

### 误区五：成本只按 GPU hour 算

失败重试、idle、低效数据管道、长 checkpoint、冷启动和人工排障都是真实成本。

### 误区六：能效就是降低功耗

AI 集群能效的关键是单位能耗的有效产出，而不是单纯让 GPU 少耗电。

## 设计检查清单

- 是否区分 GPU 分配率、活跃率、有效吞吐和价值产出。
- 是否建立 capacity、demand、usage、output 四本账。
- 是否能把 GPU hour 拆成 productive、idle、waiting、failed、overhead、reserved。
- 是否采集 DCGM/NVML 级 GPU 指标。
- 是否采集训练、推理和数据任务的应用层产出指标。
- 是否能看到 pending reason。
- 是否能计算 starvation age、admission latency 和 scheduling latency。
- 是否能按 queue、team、project、GPU type 拆分等待时间。
- 是否能度量资源碎片，而不只是空闲 GPU。
- 是否能计算 largest contiguous allocation。
- 是否有碎片治理 playbook。
- 是否基于历史运行做 request right-sizing。
- 是否区分 guaranteed quota、borrowed quota 和实际使用。
- 是否能看到 Notebook idle GPU。
- 是否能看到 checkpoint duration 和 DataLoader wait。
- 是否能把成本归因到 team/project/workload/model。
- 是否区分 showback 和 chargeback。
- 是否有推理 p99 与混部任务的关联分析。
- 是否有 headroom 策略。
- 是否能展示 physical、healthy、schedulable、allocatable、usable_for_workload 等容量状态。
- 是否按 GPU flavor 做容量规划。
- 是否能识别 GPU 分配高但功耗/SM 利用低的任务。
- 是否能识别高功耗低吞吐任务。
- 是否检查指标标签、run_id、tenant、model、dataset 等数据质量。
- 是否有 queue rebalancing 和 defragmentation 流程。
- 是否有从指标到动作再到验证的治理闭环。
- 是否有面向管理、运维、用户和单任务的不同 dashboard。

## 小结

AI 集群治理要从单一“GPU 利用率”升级为多层指标体系：

```text
capacity
  -> demand
  -> queue
  -> allocation
  -> active usage
  -> productive usage
  -> output
  -> cost and energy
```

真正有价值的目标不是让每张 GPU 都被占满，而是让高价值 workload 更快、更稳、更便宜地产生结果。

当你能解释“为什么空闲 GPU 不能用”“为什么分配率高但吞吐低”“为什么某个队列长期等待”“为什么推理 p99 被混部影响”“为什么新增 GPU 不能解决瓶颈”时，集群才进入可治理状态。

## 延伸阅读

- [Kubernetes Resource Metrics Pipeline](https://kubernetes.io/docs/tasks/debug/debug-cluster/resource-metrics-pipeline/)
- [NVIDIA DCGM Exporter](https://docs.nvidia.com/datacenter/dcgm/latest/gpu-telemetry/dcgm-exporter.html)
- [Kueue Prometheus Metrics](https://kueue.sigs.k8s.io/docs/reference/metrics/)
- [Slurm sacct](https://slurm.schedmd.com/sacct.html)
- [Slurm squeue](https://slurm.schedmd.com/squeue.html)
- [Kubernetes Resource Management for Pods and Containers](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/)
