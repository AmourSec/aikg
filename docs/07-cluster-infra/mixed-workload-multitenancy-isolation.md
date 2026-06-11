---
title: 混合集群与多租户隔离：训练、推理、Notebook 与批处理共存
domain: cluster-infra
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-06-12
---

# 混合集群与多租户隔离：训练、推理、Notebook 与批处理共存

AI 集群很少只跑一种任务。真实环境里通常同时存在：

- 大规模训练任务。
- 在线推理服务。
- Notebook 和交互式实验。
- 数据预处理和离线评测。
- embedding、RAG index 构建和批量推理。
- 系统监控、日志采集、镜像缓存、存储网关等基础设施任务。

这些任务共用 GPU、CPU、内存、网络、存储、镜像仓库和调度系统，但它们的目标完全不同。

训练任务关心吞吐、扩展效率和故障恢复；推理任务关心 p99 latency、稳定性和弹性扩缩容；Notebook 关心响应速度和灵活性；批处理关心成本和排队时间；系统任务关心高可用和低干扰。

因此，混合集群的核心问题不是“能不能把所有任务都提交进去”，而是：

> 如何在同一套集群资源上，让不同租户、不同优先级、不同 SLA、不同资源形态的 AI workload 共存，同时控制干扰、碎片、故障半径和治理复杂度？

## 一张总图

```mermaid
flowchart TB
    Users["Tenants<br/>research / platform / product / batch"]
    Portal["Submission Layer<br/>notebook / CLI / API / CI"]
    Policy["Policy Layer<br/>quota / priority / admission / security"]
    Queues["Queues<br/>team queue / service queue / best-effort queue"]
    Scheduler["Scheduler<br/>Slurm / Kubernetes / Ray / Kueue / Volcano"]
    Pools["Node Pools<br/>training / inference / dev / system / spare"]
    Runtime["Runtime Isolation<br/>cgroup / namespace / GPU device / MIG / MPS"]
    Infra["Shared Infra<br/>network / storage / registry / observability"]
    Workloads["Workloads<br/>train / serve / notebook / eval / ETL / RAG"]
    Signals["Feedback<br/>utilization / pending reason / SLA / fairness"]

    Users --> Portal
    Portal --> Policy
    Policy --> Queues
    Queues --> Scheduler
    Scheduler --> Pools
    Pools --> Runtime
    Runtime --> Workloads
    Infra --> Workloads
    Workloads --> Signals
    Signals --> Policy
    Signals --> Scheduler
```

这张图强调三点：

- 多租户不是只靠调度器完成，入口、策略、队列、运行时隔离和可观测性都要参与。
- 混部不是简单提高利用率，而是在利用率、SLA、公平性、隔离和可维护性之间取舍。
- 集群治理应该形成反馈闭环：资源使用、排队原因、SLA 违约和故障复盘要反过来修正队列和策略。

## 先区分 workload 类型

混合集群设计的第一步，是承认不同 workload 的行为差异。

| Workload | 典型资源 | 时间特征 | 主要目标 | 主要风险 |
| --- | --- | --- | --- | --- |
| LLM 训练 | 多机多卡、网络、存储 | 长时运行 | step time、扩展效率、可恢复 | gang 等待、故障重启、网络拥塞 |
| 微调 / LoRA | 少量 GPU、存储 | 中短任务 | 周转时间、环境灵活 | 碎片化、依赖漂移 |
| 在线推理 | GPU、CPU、内存、网络 | 长驻服务 | p99 latency、可用性、吞吐 | 被抢占、batch 抖动、模型加载慢 |
| 离线推理 | GPU、存储、网络 | 批量任务 | 成本、吞吐 | 与在线服务争资源 |
| Notebook | GPU、CPU、内存 | 交互式、空闲多 | 响应速度、易用性 | 占而不用、手工改环境 |
| 数据预处理 | CPU、内存、存储、网络 | 批处理 | 数据吞吐、成本 | 小文件、元数据压力 |
| RAG index 构建 | CPU/GPU、存储、向量库 | 批处理或周期性 | 构建速度、一致性 | 存储和网络尖峰 |
| Benchmark | 固定硬件、低干扰 | 短时或周期性 | 可比较、可解释 | 被混部噪声污染 |
| 系统组件 | CPU、内存、网络 | 长驻 | 稳定性 | 被用户任务挤压 |

这些任务不能都放在一个默认队列里。否则会出现：

- Notebook 长期占用 GPU，训练任务排队。
- 低优先级离线推理把在线推理 p99 拉高。
- 数据预处理把共享文件系统打满，训练 GPU 等数据。
- 多个大训练任务同时 checkpoint，存储瞬间拥塞。
- Benchmark 被后台任务干扰，结论不可用。

## Workload Contract：混部的最小声明

混合集群里最容易出问题的任务，往往不是“资源申请很大”的任务，而是“没有说清楚自己是什么”的任务。

所以入口层应该要求每个任务声明 workload contract。它不需要很复杂，但至少要把调度器、准入策略和运维人员需要判断的内容写清楚。

示例：

```yaml
workload:
  name: sft-run-2026-06-12
  type: training
  tenant: research-a
  priority: normal
  queue: train-h100
  interruptible: true
  checkpoint:
    enabled: true
    interval_minutes: 20
    path: s3://bucket/checkpoints/sft-run-2026-06-12/

resources:
  gpu:
    type: h100
    count: 64
    topology: same-rack-preferred
  cpu_per_gpu: 16
  memory_per_gpu: 128Gi
  local_nvme: 2Ti

isolation:
  data_class: internal
  network_profile: training-rdma
  storage_profile: shared-fs-high-throughput
  environment_family: train-h100-cu122

slo:
  max_queue_wait: 12h
  expected_duration: 36h
  preemption_grace: 15m
```

这个 contract 的价值在于：

- 调度系统知道任务需要什么资源和拓扑。
- 准入策略知道它能不能进入某个队列。
- 抢占策略知道它是否可中断。
- 存储和网络策略知道它会制造什么负载。
- 事故复盘时能知道它为什么被允许运行。

不同 workload 的 contract 重点不同：

| Workload | contract 必须讲清楚 |
| --- | --- |
| 在线推理 | SLA、模型 artifact、扩缩容边界、是否可驱逐、流量入口 |
| 大训练 | GPU 拓扑、gang 需求、checkpoint、最大运行时间、恢复策略 |
| Notebook | idle timeout、最大运行时间、数据权限、是否允许临时装包 |
| 数据预处理 | 存储路径、IOPS/吞吐预算、小文件数量、输出提交方式 |
| Benchmark | 固定节点池、固定镜像、输入 trace、低干扰要求 |

没有 contract 的任务应该进入低优先级默认队列，不能直接使用生产推理池、高优先级、privileged 权限或关键数据路径。

## 多租户要隔离什么

多租户隔离不只是“不同团队用不同 namespace”。至少要分七个维度。

### 资源隔离

控制每个租户能用多少：

- GPU 数量和 GPU 类型。
- CPU、内存、ephemeral storage。
- 本地 NVMe cache。
- 网络带宽或 RDMA 能力。
- 共享文件系统配额。
- 对象存储 bucket/prefix。
- 镜像仓库和构建资源。

Kubernetes ResourceQuota 可以限制 namespace 级别的对象数量和资源总量；Slurm 可以用 account、partition、QoS、fairshare 等方式控制租户使用。

资源隔离解决的是“谁能用多少”，但不自动解决“用得是否高效”。

### 性能隔离

性能隔离关注一个任务是否会拖慢另一个任务。

典型干扰包括：

- 多个任务共享同一 PCIe switch 或 NIC。
- 多个 job 同时走同一个 RDMA uplink。
- 本地 NVMe cache 被别的任务清空。
- 共享文件系统 metadata server 被小文件打爆。
- CPU thread、DataLoader、tokenizer 抢占推理服务 CPU。
- 推理服务和训练任务共享 GPU 时互相影响。

性能隔离通常比资源隔离更难，因为它依赖拓扑和运行时行为。

### 故障隔离

故障隔离关注失败是否扩散。

例如：

- 一个团队的错误镜像是否拖垮节点。
- 一个任务的日志量是否打满磁盘。
- 一个 job 的 NCCL hang 是否占住整批 GPU。
- 某个数据集路径异常是否影响所有训练任务。
- 一个租户的恶意或错误配置是否影响 host。

故障隔离需要限额、超时、健康检查、自动清理和权限边界共同完成。

### 安全隔离

AI 集群常常保存训练数据、模型权重、日志、实验结果和密钥。安全隔离包括：

- 身份认证和授权。
- namespace / account / project 边界。
- Secret 访问控制。
- 容器权限限制。
- host path、privileged container、host network 的限制。
- 网络访问控制。
- 数据集和模型 artifact 权限。

Kubernetes 官方多租户文档强调，隔离既包括控制面隔离，也包括数据面隔离。对于 AI 集群，数据面隔离尤其重要，因为很多任务会挂载大规模数据和模型。

### 环境隔离

环境隔离关注依赖是否互相污染：

- 是否共享 Conda 环境。
- 是否允许运行时 `pip install -U`。
- Notebook 环境和 batch 环境是否一致。
- 生产推理是否使用 immutable image。
- 自定义 CUDA extension 是否写入共享路径。
- 编译缓存是否按镜像、模型、GPU 架构区分。

环境隔离与上一篇“环境可复现”直接相关。混合集群里越允许灵活环境，越需要明确哪些任务可以灵活，哪些必须固定。

### 数据隔离

数据隔离关注谁能读写哪些数据，以及不同任务的数据生命周期。

包括：

- 数据集只读挂载。
- checkpoint 写入路径隔离。
- 模型权重发布路径隔离。
- RAG index 构建和发布路径隔离。
- 临时数据清理。
- 日志和 trace 脱敏。

训练和推理常常共享模型 artifact，但不应该共享所有中间路径。否则一次错误覆盖可能影响线上服务。

### 控制面隔离

控制面隔离关注用户能否影响调度和集群状态。

例如：

- 用户是否能创建高优先级任务。
- 用户是否能使用 privileged pod。
- 用户是否能修改 node selector、toleration、priority class。
- 用户是否能创建大量小任务压垮 API server。
- 用户是否能绕过队列直接提交到节点池。

很多混合集群问题不是底层算力不够，而是控制面没有准入策略。

## 租户模型：Namespace、Account、Project 与 Queue

多租户治理首先要统一“租户”到底是什么。不同系统的抽象不同：

- Kubernetes 常用 namespace 表达边界。
- Slurm 常用 account、association、partition、QoS。
- 云平台常用 project、folder、billing account。
- 队列系统常用 LocalQueue、ClusterQueue、cohort、resource flavor。

如果这些概念各自独立，就会出现权限、配额和成本归因对不上的问题。例如：

```text
Kubernetes namespace: team-a-dev
Slurm account: research-a
Object storage prefix: /datasets/group-alpha/
Cost tag: project-123
Queue: shared-gpu
```

这些名字如果没有映射关系，后续很难回答“这个任务属于谁、消耗谁的 quota、能访问哪些数据、成本算到哪里”。

更好的做法是建立一张租户映射表：

| 层 | 字段 | 示例 |
| --- | --- | --- |
| Identity | tenant_id | `research-a` |
| Kubernetes | namespace / service account | `research-a-train`、`research-a-dev` |
| Slurm | account / QoS | `research-a`、`normal` |
| Queue | local queue / cluster queue | `team-a-train` -> `h100-training` |
| Storage | bucket / prefix / quota | `s3://ai-data/research-a/` |
| Registry | image namespace | `registry.example.com/research-a/*` |
| Cost | billing tag | `tenant=research-a` |
| Security | data class | public / internal / restricted |

租户模型要同时服务四件事：

- 授权：谁能提交什么任务，能访问哪些资源。
- 配额：谁有多少 guaranteed quota，能借多少。
- 审计：谁在什么时候运行了什么。
- 归因：资源、成本、故障和 SLA 影响算到谁。

Kubernetes ResourceQuota 可以限制 namespace 内资源和对象总量；Kueue 的 LocalQueue/ClusterQueue 可以把用户入口和集群级资源池解耦；Slurm 的 account/QoS/fairshare 可以把历史使用量纳入优先级。AI 平台通常需要把这些层统一起来，而不是只依赖其中一个。

## 节点池设计

混合集群通常不会把所有节点放在一个池子里。更常见的是把节点按用途、硬件和隔离级别分组。

| Node Pool | 典型任务 | 设计重点 |
| --- | --- | --- |
| Training Pool | 大规模训练、微调 | 高带宽网络、拓扑一致、gang scheduling |
| Inference Pool | 在线推理 | 稳定性、模型缓存、低抖动、弹性扩缩容 |
| Dev Pool | Notebook、小实验 | 灵活、低优先级、可抢占、自动回收 |
| Batch Pool | 数据预处理、离线推理 | 成本、吞吐、可抢占 |
| Benchmark Pool | 性能测试 | 低干扰、固定环境、固定拓扑 |
| System Pool | 监控、日志、控制面组件 | 高可靠、避免被用户任务挤压 |
| Spare / Burst Pool | 临时扩容、抢占式资源 | 成本优化、容忍中断 |

节点池太少会导致互相干扰；节点池太多会导致碎片和管理成本上升。

一个常见折中是：

```text
关键在线服务和 benchmark 硬隔离
大训练和普通批处理软隔离
Notebook 和低优先级任务可抢占
系统组件独立节点池
```

## 单集群、多集群与虚拟集群

混部不一定意味着所有东西必须在一个 Kubernetes 或 Slurm 集群里。常见形态有三种。

| 形态 | 优点 | 风险 | 适用场景 |
| --- | --- | --- | --- |
| 单集群多节点池 | 资源池大，调度灵活，统一观测 | 控制面影响面大，策略复杂 | 中小规模统一平台 |
| 多物理集群 | 故障隔离强，策略简单 | 资源碎片、跨集群调度和成本归因复杂 | 生产推理、核心训练、强安全边界 |
| 虚拟集群 / 逻辑租户 | 用户体验接近独立集群，底层仍共享 | 需要额外控制面和策略治理 | 多团队共享研究平台 |

选择时可以按三个问题判断：

1. 这个 workload 的事故会不会影响生产服务？
2. 它是否需要和其他 workload 共享同一批昂贵 GPU 才能提高利用率？
3. 控制面、网络、存储和安全策略是否能承受混部复杂度？

经验上：

- 生产在线推理和关键 benchmark 更适合独立集群或强隔离节点池。
- 研发训练、微调、离线推理适合共享集群加队列治理。
- Notebook 可以共享底层资源，但要通过虚拟边界、idle 回收和权限限制降低风险。
- 安全等级不同的数据任务不应只靠队列隔离，必要时要物理或网络隔离。

## Hard Isolation 与 Soft Sharing

混部策略可以放在一条轴上理解。

```text
hard isolation
  -> dedicated node pool
  -> quota + queue
  -> quota borrowing
  -> opportunistic sharing
  -> best-effort preemptible
soft sharing
```

### Hard Isolation

硬隔离适合：

- 在线推理核心服务。
- 关键 benchmark。
- 安全级别高的数据。
- 需要固定拓扑的大训练。
- 系统控制面组件。

优点是可预测、容易排障；缺点是利用率可能低。

### Soft Sharing

软共享适合：

- 离线批处理。
- Notebook。
- 小规模微调。
- 低优先级评测。
- 资源空闲时的 opportunistic job。

优点是利用率高；缺点是需要更强的抢占、隔离和可观测性。

### Quota Borrowing

很多队列系统支持“有配额上限，但空闲资源可以借用”。

例如：

```text
team-a guaranteed: 64 GPU
team-a max: 128 GPU
team-b guaranteed: 64 GPU
team-b max: 128 GPU
shared cluster: 192 GPU
```

当 team-b 空闲时，team-a 可以临时使用超过 guaranteed 的资源；当 team-b 提交任务时，系统可以通过排队或抢占逐步收回借用资源。

这比静态切分资源更高效，但必须回答：

- 借用资源是否可抢占。
- 抢占谁。
- 抢占前是否有 grace period。
- 在线服务是否允许被抢占。
- 大训练被抢占后 checkpoint 是否足够快。

## 隔离等级：不要只有“隔离/不隔离”

现实中很少只有两种状态。更实用的是定义隔离等级，让不同 workload 按风险进入不同等级。

| 等级 | 典型 workload | 隔离要求 | 调度策略 |
| --- | --- | --- | --- |
| Tier 0 | 控制面、核心在线推理、关键网关 | 独立节点池、强权限边界、不可抢占 | 最高优先级，保留容量 |
| Tier 1 | 重要训练、生产评测、正式 benchmark | 固定环境、固定节点池或拓扑、低干扰 | 高优先级，谨慎抢占 |
| Tier 2 | 常规训练、微调、离线推理 | 队列隔离、quota、可观测 | 正常优先级，可借用 |
| Tier 3 | Notebook、小实验、数据探索 | dev pool、idle timeout、权限限制 | 低优先级，可抢占 |
| Tier 4 | opportunistic batch、spot 任务 | best effort、可中断、自动重试 | 使用空闲资源，随时回收 |

隔离等级要写进准入策略，而不是只写在文档里。例如：

- Tier 0/1 任务必须使用镜像 digest 和受支持环境族。
- Tier 0/1 任务不能和未知 Notebook 混部。
- Tier 2 可以借用资源，但要接受排队和部分抢占。
- Tier 3 必须有 idle timeout 和最大运行时间。
- Tier 4 必须声明可中断和重试方式。

这样平台不需要对每个任务单独争论，而是把风险和权利用等级表达出来。

## Queue、Quota、Priority、Preemption

混合集群治理最核心的四个词是：

```text
queue
quota
priority
preemption
```

### Queue

Queue 是用户提交任务时看到的入口。

好的队列应该表达：

- 属于哪个团队或项目。
- 面向哪类 workload。
- 可用哪些节点池。
- 默认优先级。
- 是否允许借用资源。
- 是否允许抢占或被抢占。
- 是否有最大运行时间。

Kueue 这类 Kubernetes 批处理队列系统把 LocalQueue、ClusterQueue、ResourceFlavor 等概念分开：用户提交到本地队列，平台侧把不同队列映射到集群资源和 flavor。这个模型适合把“用户入口”和“底层资源池”解耦。

### Quota

Quota 表示租户或队列的资源边界。

Quota 可以是：

- 硬上限：不能超过。
- 软上限：可以借用空闲资源。
- 最小保障：资源紧张时仍应保留。
- 周期预算：按天、周、月统计资源用量。
- 对象数量限制：限制 pod/job/PVC/configmap 数量。

Kubernetes ResourceQuota 适合 namespace 内的资源和对象数量治理；AI 批处理场景往往还需要队列层的 GPU quota、fairshare 和借用策略。

### Priority

Priority 决定资源紧张时谁先运行，谁可以等待。

高优先级通常给：

- 在线推理。
- 生产评测。
- 关键发布流程。
- 快到 deadline 的任务。
- 系统组件。

低优先级通常给：

- Notebook 空闲 session。
- 离线批处理。
- 可重试评测。
- opportunistic training。

Kubernetes Pod Priority and Preemption 允许高优先级 Pod 抢占低优先级 Pod，但对 AI workload 要谨慎：抢占一个普通 Web Pod 和抢占一个 512 GPU 训练 job 的代价完全不同。

### Preemption

Preemption 是提高利用率的强工具，也是制造事故的强工具。

适合被抢占的任务：

- 有 checkpoint 的训练。
- 可重跑的数据处理。
- 离线推理。
- Notebook idle session。
- 低优先级评测。

不适合被抢占的任务：

- 无冗余在线推理。
- 关键 benchmark。
- 控制面组件。
- 没有 checkpoint 的长训练。
- 正在写关键 artifact 的任务。

抢占策略必须配合：

- checkpoint 周期。
- graceful termination。
- 最大运行时间。
- retry policy。
- artifact 原子提交。
- 用户可见的 pending/preempted reason。

### AI 任务的抢占生命周期

AI 任务抢占不能只发送一个 kill 信号。一个更稳妥的生命周期是：

```text
preempt requested
  -> mark draining
  -> stop accepting new work
  -> checkpoint / flush output
  -> release lease
  -> terminate
  -> requeue / reschedule
  -> restore
  -> validate progress
```

不同任务对应的动作不同：

| Workload | 抢占前动作 | 恢复动作 |
| --- | --- | --- |
| 训练 | 保存 checkpoint、记录 global step、停止写入非原子 artifact | 从 checkpoint 恢复，校验 step 和 optimizer state |
| 离线推理 | flush 已完成分片、提交分片 manifest | 跳过已完成分片，继续剩余分片 |
| 数据预处理 | 原子提交输出目录，避免半成品覆盖正式数据 | 从输入 manifest 重新切分或续跑 |
| Notebook | 保存 session 状态，提示用户迁移 | 重新分配低优先级资源 |
| 在线推理 | 从负载均衡摘除副本，等待请求 drain | 新副本预热后再接流量 |

抢占策略至少要定义：

- `termination_grace_period` 是否足够完成 checkpoint。
- 超时后是否强杀。
- 被抢占后是否自动 requeue。
- 连续抢占次数是否有限制。
- 抢占原因是否进入 run manifest。
- 抢占造成的浪费如何计入成本。

如果一个 workload 没有恢复策略，就不应该被标记为 preemptible。否则调度器看似提高了利用率，实际只是把计算浪费转移到用户身上。

## 在线推理与训练混部

训练和推理混部是最有吸引力、也最容易出问题的场景。

### 为什么想混部

因为在线推理的负载有波峰波谷：

- 白天高峰，夜间低谷。
- 活动期间高峰，平时低谷。
- 不同业务峰值时间不同。

空闲 GPU 如果完全保留给推理，会降低利用率。低谷时可以运行低优先级训练、离线推理或数据任务。

### 为什么难

在线推理和训练的性能目标冲突：

| 维度 | 训练 | 在线推理 |
| --- | --- | --- |
| 目标 | 平均吞吐、step time | p95/p99 latency、可用性 |
| 资源使用 | 长时高占用 | 随请求波动 |
| 容错 | 可 checkpoint 重跑 | 请求不能随意失败 |
| 调度 | gang、拓扑敏感 | 弹性、滚动发布 |
| 性能风险 | 网络/存储拥塞 | 抖动、冷启动、batch 延迟 |

因此，常见策略是：

```text
推理保留基础容量
  + 空闲资源运行低优先级批任务
  + 高峰时回收低优先级任务
  + 推理扩容优先
  + 批任务必须可中断
```

### 混部边界

推理和训练混部时要设边界：

- 不让训练任务使用推理服务所在节点的所有 CPU。
- 不让训练 checkpoint 挤占模型加载和日志路径。
- 不让训练 NCCL 流量影响推理入口网络。
- 推理节点上的低优先级任务必须可驱逐。
- 推理服务要有模型预热和冷启动预算。
- 低优先级任务不能使用生产推理的 secret。

如果这些边界没有建立，混部带来的利用率提升很容易被 p99 抖动和事故成本抵消。

## Notebook 治理

Notebook 是研发效率工具，也是 GPU 利用率治理难点。

典型问题：

- 用户启动后忘记关闭。
- GPU 被进程占用但显存和算力利用率很低。
- Notebook 内临时安装依赖，环境不可复现。
- 开发 session 使用生产数据。
- 多个 Notebook 共享同一机器，互相影响。

常见治理策略：

- 默认使用 dev node pool。
- 设置 idle timeout。
- 设置最大连续运行时间。
- 对 Notebook GPU 配额单独统计。
- 长时间训练必须转成 batch job。
- 关键实验必须固化为镜像和 manifest。
- Notebook 镜像和生产镜像分离。

一个实用规则是：

```text
Notebook 用来探索，不用来承载关键长任务。
```

当探索变成可复现实验，应进入 batch 或 training queue。

## 系统任务保护

混合集群必须保护系统任务。

系统任务包括：

- DNS、CNI、CSI。
- GPU device plugin。
- DCGM exporter。
- 日志 agent。
- metrics collector。
- ingress / gateway。
- 镜像缓存和 registry mirror。
- 调度器和队列控制器。

如果系统任务和用户任务争抢资源，会出现很难排查的问题：

- 节点心跳抖动。
- 日志丢失。
- 监控空洞。
- CSI mount 超时。
- GPU plugin 异常导致资源不可见。
- 调度器延迟升高。

因此应考虑：

- system node pool。
- system priority class。
- resource requests/limits。
- taints/tolerations。
- pod disruption budget。
- 和用户任务分离的日志/监控资源。

系统任务不应该靠“刚好还有一点 CPU”运行。

## Kubernetes 隔离工具箱

Kubernetes 提供了很多通用隔离工具，但 AI 场景要组合使用。

| 工具 | 解决什么 | AI 场景注意点 |
| --- | --- | --- |
| Namespace | 逻辑分组 | 不是安全边界本身，需要 RBAC/Policy |
| ResourceQuota | namespace 资源上限 | GPU extended resource 也要纳入治理 |
| LimitRange | 默认 request/limit | 避免用户漏写 CPU/memory |
| PriorityClass | 优先级 | 谨慎给用户高优先级权限 |
| Preemption | 高优先级抢占低优先级 | 大训练抢占代价高，要配 checkpoint |
| QoS Class | Guaranteed/Burstable/BestEffort | 推理服务应避免 BestEffort |
| NodeSelector/Affinity | 节点选择 | 表达 GPU 型号、拓扑、节点池 |
| Taints/Tolerations | 控制哪些任务能上节点 | 保护系统池、推理池、benchmark 池 |
| NetworkPolicy | 网络访问控制 | 限制租户间访问和外部访问 |
| Pod Security Standards | 限制容器权限 | 控制 privileged、host path、host network |

这些工具的共同问题是：它们只提供机制，不自动给出策略。AI Infra 的工作是把策略固化成默认模板、准入规则和队列约束。

### Kubernetes 中的推荐落地边界

在 Kubernetes 里做 AI 多租户时，不建议让用户直接提交任意 Pod。更稳的路径是：

```text
user intent
  -> workload template
  -> admission policy
  -> queue object
  -> generated pod/job
```

也就是说，用户提交“我要跑一个训练任务”，平台根据模板生成 Pod、Job、RayJob、PyTorchJob 或 Kueue Workload，而不是让用户手写所有底层字段。

建议默认固化：

- namespace、service account 和 tenant 的映射。
- node selector、toleration 和 topology 选择。
- resource requests/limits。
- priority class。
- image digest 和环境族。
- network policy。
- secret mount 白名单。
- volume mount 白名单。
- termination grace period。
- sidecar、日志和指标采集。

用户可以改业务参数，但不应该随意改：

- `hostNetwork`
- `hostPID`
- `privileged`
- 任意 `hostPath`
- system node pool toleration
- 高优先级 priority class
- 未授权 secret
- 绕过队列的 node selector

这些字段一旦开放，namespace 就很难作为可靠边界。

## Slurm 隔离工具箱

很多 AI 训练集群仍然以 Slurm 为主。Slurm 的优势是 HPC 批处理、拓扑、accounting、fairshare 和大型 gang job。

常见抽象包括：

- Partition：把节点分组。
- Account：按组织、项目或团队统计资源。
- QOS：限制优先级、最大资源、最大运行时间等。
- Fairshare：根据历史使用量调整优先级。
- GRES：表示 GPU 等通用资源。
- Reservation：预留资源。
- Preemption：抢占低优先级 job。

AI 场景下要关注：

- 大训练 job 需要 gang 分配。
- 交互式 job 不能长期占住高价值节点。
- Fairshare 要结合 GPU 类型和 GPU hour，而不只是 job 数。
- Preemption 必须和 checkpoint 能力匹配。
- Partition 不宜过碎，否则排队和碎片严重。

### Slurm 与 Kubernetes 混用

不少 AI 平台会同时有 Slurm 和 Kubernetes：

- Slurm 承载大规模训练和 HPC 风格批处理。
- Kubernetes 承载在线推理、Notebook、控制面和云原生服务。
- Ray 可能横跨二者，用于数据处理、分布式推理或实验平台。

混用时最大的风险不是“两个系统都能调度”，而是它们看不到彼此的真实资源状态。

常见问题：

- Kubernetes 认为某节点空闲，但 Slurm 已经保留或占用。
- Slurm job 和 Kubernetes Pod 共享同一节点导致 GPU/CPU/网络争用。
- 两边的 quota、account、priority、cost tag 不一致。
- 日志、指标和故障归因被拆散。
- 用户可以选择更宽松的一边绕过策略。

更稳的做法是明确边界：

| 模式 | 做法 |
| --- | --- |
| 物理隔离 | 一批节点归 Slurm，一批节点归 Kubernetes |
| 时间隔离 | 维护窗口或批处理窗口内切换资源归属 |
| 控制器协调 | 用统一平台在两边创建任务并维护资源状态 |
| 单向承载 | Kubernetes 只跑服务，Slurm 只跑大训练，不共享节点 |

如果必须共享节点，至少要有统一的 node ownership、资源锁、GPU 设备可见性控制和成本归因。不建议让两个调度器在同一批 GPU 上独立做最终决策。

## GPU 共享：MIG、MPS 与 Time Slicing

GPU 共享是混部中的敏感点。

### MIG

MIG 把支持的 GPU 切成硬件隔离的实例，适合：

- 小模型推理。
- Notebook。
- 小规模实验。
- 多租户隔离要求较高的场景。

限制是：

- 不是所有 GPU 都支持。
- 切分形态固定。
- 大模型训练不适合。
- 切分和调度策略需要配合。

### MPS

MPS 让多个 CUDA 进程更高效共享 GPU，适合某些小 kernel、低占用场景。

但它的隔离性弱于 MIG，使用时要注意：

- 显存控制。
- 错误传播。
- 性能干扰。
- 用户权限边界。

### Time Slicing

Time slicing 可以让多个容器轮流使用 GPU，适合低优先级、交互式或轻量任务。

但对延迟敏感推理和 benchmark 来说，time slicing 可能带来不可接受的抖动。

结论是：

```text
GPU 共享适合提高小任务利用率，不适合默认用于所有任务。
```

## 存储与网络的多租户

很多混部问题表面上是 GPU 争用，根因其实是网络或存储。

### 存储隔离

需要治理：

- 每个租户的容量配额。
- checkpoint 写入速率。
- 小文件数量。
- 本地 NVMe cache 使用。
- 对象存储 API 请求量。
- 共享文件系统 metadata 压力。

如果只限制 GPU 数量，不限制 checkpoint 和数据读取，高优先级训练仍然可能被低优先级数据任务拖慢。

### 网络隔离

需要治理：

- RDMA 网络和普通业务网络分离。
- 训练 all-reduce 流量和推理入口流量分离。
- 多租户网络访问控制。
- QoS / traffic class。
- 拓扑感知调度。
- 拥塞可观测性。

RoCE 集群尤其要小心。一个错误配置或突发流量可能造成 PFC pause、ECN 异常和尾延迟恶化。

## 可观测性：看见干扰

混合集群必须能看见干扰，否则只能靠用户抱怨。

需要监控的指标包括：

### 资源使用

- GPU utilization。
- GPU memory。
- SM occupancy。
- HBM bandwidth。
- CPU utilization。
- memory pressure。
- local NVMe usage。
- network throughput。
- storage IOPS 和 metadata rate。

### 调度状态

- pending job 数量。
- pending reason。
- queue wait time。
- quota 使用率。
- fairshare 变化。
- preemption 次数。
- gang job 等待时间。
- fragmentation。

### SLA

- 推理 p50/p95/p99。
- 请求错误率。
- 模型加载时间。
- 训练 step time。
- checkpoint duration。
- DataLoader wait。
- job restart 次数。

### 多租户公平性

- 每个租户 GPU hour。
- 每个租户等待时间。
- 每个租户被抢占次数。
- 每个租户 quota borrowing。
- 每个租户资源浪费。

没有这些指标，混部策略无法调优。

## Noisy Neighbor：干扰归因

混合集群里的典型抱怨是：

```text
我的代码没变，但今天变慢了。
```

这时如果没有干扰归因，只能在代码、数据、环境、网络和邻居任务之间猜。更好的做法是把 noisy neighbor 当成一类正式故障处理。

常见干扰源：

| 干扰源 | 表现 | 证据 |
| --- | --- | --- |
| GPU 共享 | SM/HBM 抖动、推理 p99 上升 | 同卡进程、MIG/MPS/time slicing 配置 |
| CPU 争用 | tokenizer/DataLoader 慢、服务线程排队 | CPU throttling、run queue、cgroup 指标 |
| 网络拥塞 | NCCL timeout、all-reduce 变慢、推理入口抖动 | fabric metrics、ECN/PFC、NCCL log |
| 存储拥塞 | checkpoint 慢、DataLoader wait 高 | IOPS、吞吐、metadata rate、对象存储请求 |
| 控制面压力 | Pod 创建慢、调度延迟、watch 超时 | API server、scheduler、queue controller 指标 |
| 镜像风暴 | 大量 Pod 同时拉镜像，节点启动慢 | registry、node image cache、pull duration |
| 日志风暴 | 节点磁盘或日志系统压力 | log agent queue、磁盘使用率、drop count |

排查时建议按时间窗口对齐：

```text
victim workload slowdown window
  -> same node / same rack workloads
  -> same storage path workloads
  -> same network fabric workloads
  -> same control plane events
  -> recent scheduling / preemption / rollout events
```

对线上推理尤其要记录：

- 同节点是否有低优先级 batch。
- 同时段是否有大规模镜像拉取。
- 模型加载路径是否被训练 checkpoint 挤占。
- ingress、tokenizer、scheduler、GPU worker 哪一段变慢。
- prefix cache 或 KV cache 是否被重建。

对训练任务则要记录：

- 是否和其他大 job 共用 rack uplink。
- 是否有同时 checkpoint 的任务。
- 是否发生过重调度、抢占或节点替换。
- 是否跨了不同 GPU 拓扑或不同 node profile。

Noisy neighbor 治理不是追求完全无干扰，而是让干扰可见、可归因、可用策略缓解。

## 常见策略模板

### 策略一：训练主导型集群

适合研究和模型开发阶段。

```text
training pool: majority GPU
dev pool: small, preemptible
inference pool: separate minimal
batch pool: uses idle capacity
benchmark pool: small dedicated
```

关注点：

- 大 job 排队效率。
- GPU 拓扑一致性。
- checkpoint 和恢复。
- fairshare。
- Notebook 自动回收。

### 策略二：推理主导型集群

适合模型上线和服务平台。

```text
inference pool: reserved + autoscaling
batch inference: low priority
training/fine-tune: separate or preemptible
model cache: local NVMe
canary pool: deployment validation
```

关注点：

- p99 latency。
- 模型加载和预热。
- rollback。
- 低优先级任务驱逐。
- 线上和离线网络隔离。

### 策略三：共享研究平台

适合多个课题组或团队共用。

```text
team queues
guaranteed quota
borrowable quota
dev queue
benchmark reservation
shared storage quota
```

关注点：

- 公平性。
- 配额透明。
- 排队原因可解释。
- 数据权限。
- 环境模板。

### 策略四：成本优化型集群

适合大量离线任务和弹性资源。

```text
base reserved capacity
burst spot/preemptible capacity
checkpoint-aware training
best-effort batch jobs
aggressive idle cleanup
```

关注点：

- 任务是否可中断。
- checkpoint 成本。
- 重试风暴。
- 队列退避。
- 成本归因。

## 准入控制

准入控制决定任务能否进入集群。

AI 任务常见准入检查：

- 是否指定队列。
- 是否指定镜像 digest。
- 是否声明 GPU/CPU/memory。
- 是否超过租户 quota。
- 是否使用未授权节点池。
- 是否使用禁止的 host path。
- 是否设置 checkpoint 路径。
- 是否设置最大运行时间。
- 是否使用高优先级。
- 是否使用生产 secret。
- 是否符合环境矩阵。

准入控制的目标不是增加门槛，而是把事故挡在运行前。

### 准入策略模板

可以把准入策略分成三类。

| 类型 | 作用 | 示例 |
| --- | --- | --- |
| 必填字段 | 保证任务可治理 | queue、tenant、workload type、resource request、image digest |
| 禁止字段 | 防止越权和破坏隔离 | privileged、host network、未授权 host path、高优先级滥用 |
| 条件规则 | 根据 workload type 调整要求 | training 必须有 checkpoint，inference 必须有 readiness，benchmark 必须固定节点池 |

示例规则：

```text
if workload.type == "training" and gpu.count >= 8:
  require checkpoint.enabled == true
  require max_runtime is set
  require queue in allowed_training_queues

if workload.type == "online-inference":
  require priority in ["prod", "critical"]
  require readiness_probe
  require model_artifact_digest
  deny preemptible == true

if workload.type == "notebook":
  require idle_timeout <= 4h
  deny production_secret_mount
  require queue == "dev"

if workload.tier in ["Tier 0", "Tier 1"]:
  require image_digest
  require supported_environment_family
  deny best_effort_qos
```

准入系统还应该给出可理解的拒绝原因：

```text
rejected: notebook workload cannot mount production secret "prod-model-key"
rejected: training job requesting 64 GPUs must define checkpoint policy
rejected: benchmark workload must use benchmark node pool
```

可解释性很重要。否则用户只会看到“任务提交失败”，然后绕过平台规则。

## 资源碎片

混合集群很容易出现碎片：

- 8 卡节点上剩 1 张卡，大训练用不了。
- 某些节点 CPU 不够，GPU 空闲。
- 某些节点本地 NVMe 满了，GPU 空闲。
- GPU 型号分散，任务可用范围缩小。
- MIG 切分后形态不匹配。
- 大 job 需要同一拓扑，资源分散在多个 rack。

碎片治理手段包括：

- bin packing 和 gang scheduling。
- topology-aware scheduling。
- 大任务优先匹配完整节点。
- 小任务使用小 GPU、MIG 或 dev pool。
- 定期 defragmentation。
- 限制随意 node selector。
- 统一 GPU flavor 命名。

利用率指标不能只看“GPU 是否分配”。如果 GPU 已分配但任务 pending、DataLoader 等待或网络拥塞，实际有效利用率仍然低。

## 成本与归因

多租户集群必须能回答：

```text
谁用了多少资源？
用于什么 workload？
是否符合配额？
等待时间是多少？
浪费在哪里？
性能损失来自哪里？
```

常见归因维度：

- team / project / namespace / account。
- workload type。
- queue。
- GPU type。
- node pool。
- priority class。
- image/environment。
- dataset/model artifact。

成本治理不是单纯压低使用量，而是让资源从低价值占用流向高价值任务。

## 常见误区

### 误区一：所有任务放一个队列最公平

表面公平，实际会让长任务、短任务、在线服务和 Notebook 互相伤害。公平必须结合 workload 类型和 SLA。

### 误区二：GPU quota 够了就不需要别的隔离

网络、存储、CPU、内存、镜像拉取和控制面都可能成为共享瓶颈。

### 误区三：抢占一定能提高利用率

如果任务没有 checkpoint、恢复慢或写 artifact 不原子，抢占会制造更多浪费。

### 误区四：推理和训练混部只要设置优先级

优先级只能影响调度和抢占，不能自动隔离网络、存储、CPU、模型缓存和尾延迟。

### 误区五：Notebook 是小事

Notebook 常常是长期 GPU 占用、环境漂移和数据权限问题的来源。

### 误区六：利用率高就是好

高利用率可能来自低价值任务占满资源。AI 集群更应关注有效吞吐、SLA、公平性和可恢复性。

## 设计检查清单

- 是否明确区分训练、推理、Notebook、批处理、benchmark 和系统任务。
- 是否要求任务声明 workload contract。
- 是否有统一 tenant_id，并映射 namespace、account、queue、storage、registry 和 cost tag。
- 是否有清晰的 node pool 策略。
- 是否决定哪些 workload 放在单集群、多集群或虚拟集群。
- 是否定义 Tier 0 到 Tier 4 等隔离等级。
- 是否有 team/project queue。
- 是否有 guaranteed quota、max quota 和 borrowing 策略。
- 是否定义 priority class，并限制谁能使用高优先级。
- 哪些任务可以被抢占，哪些不能。
- 被抢占任务是否有 checkpoint、graceful termination、requeue 和恢复验证。
- 在线推理是否与低优先级任务有明确隔离边界。
- Notebook 是否有 idle timeout 和最大运行时间。
- 系统任务是否有独立资源保障。
- Kubernetes 用户是否只能通过模板和队列提交，而不是任意 Pod。
- Slurm 和 Kubernetes 是否有清晰节点归属或统一资源锁。
- GPU、CPU、memory、storage、network 是否都纳入治理。
- 共享文件系统和对象存储是否有租户配额。
- 是否有 NetworkPolicy 或等价网络隔离。
- 是否限制 privileged、host path、host network。
- 是否记录 pending reason、preemption、quota borrowing 和 SLA。
- 是否能定位 noisy neighbor 干扰。
- 是否有资源碎片指标。
- 是否能按 team/project/workload 做成本归因。
- 是否有准入控制阻止危险配置。

## 小结

混合集群的目标不是“把所有任务塞进一个大池子”，而是在共享中建立边界：

```text
workload classification
  -> queue and quota
  -> priority and preemption
  -> node pool and topology
  -> runtime isolation
  -> storage/network governance
  -> observability and feedback
```

对 AI Infra 来说，混部的价值在于提升资源有效利用率，而不是追求表面分配率。好的混合集群会让高优先级任务有保障，低优先级任务能利用空闲资源，研发探索不阻塞生产服务，系统组件不被用户任务拖垮。

一旦这些边界不清，集群就会从“共享资源池”退化成“互相干扰的黑盒”。

## 延伸阅读

- [Kubernetes Multi-tenancy](https://kubernetes.io/docs/concepts/security/multi-tenancy/)
- [Kubernetes Resource Quotas](https://kubernetes.io/docs/concepts/policy/resource-quotas/)
- [Kubernetes Pod Priority and Preemption](https://kubernetes.io/docs/concepts/scheduling-eviction/pod-priority-preemption/)
- [Kubernetes Pod QoS Classes](https://kubernetes.io/docs/concepts/workloads/pods/pod-qos/)
- [Kubernetes Network Policies](https://kubernetes.io/docs/concepts/services-networking/network-policies/)
- [Kubernetes Pod Security Standards](https://kubernetes.io/docs/concepts/security/pod-security-standards/)
- [Kueue LocalQueue](https://kueue.sigs.k8s.io/docs/concepts/local_queue/)
- [Kueue ClusterQueue](https://kueue.sigs.k8s.io/docs/concepts/cluster_queue/)
- [Slurm Multifactor Priority Plugin](https://slurm.schedmd.com/priority_multifactor.html)
- [Slurm Fair Tree](https://slurm.schedmd.com/fair_tree.html)
