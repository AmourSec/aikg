---
title: 节点生命周期与集群运维：交付、验收、入池、维护与下线
domain: cluster-infra
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-06-11
---

# 节点生命周期与集群运维：交付、验收、入池、维护与下线

AI 集群不是把 GPU 服务器上架、装好驱动、加入调度器就结束了。一个计算节点从进入机房到退出生产，会经历交付、验收、基线安装、压力测试、入池、运行、维护、升级、故障隔离、归还资源和下线。

如果这个生命周期没有被设计清楚，集群会出现很多“看似偶发”的问题：

- 新节点刚入池就出现 NCCL timeout。
- GPU 型号一样，但某批节点吞吐低。
- 训练任务随机 OOM 或 Xid error。
- 维护时任务被粗暴杀掉，checkpoint 不完整。
- 驱动和 CUDA 组合漂移，某些镜像突然不可用。
- 节点已经有硬件错误，却还在被调度。
- 下线节点仍保留数据、secret、缓存或监控噪声。

节点生命周期治理的目标是：

> 让每一台 AI 计算节点在可验证、可追溯、可维护、可回滚的状态下进入和退出资源池。

这篇文章关注集群运维流程，不深入故障复盘和可观测性体系。后者更适合放在可靠性章节展开。

## 一张总图

```mermaid
flowchart TB
    Arrival["Arrival<br/>rack / cable / asset"]
    Inventory["Inventory<br/>serial / GPU / NIC / disk / firmware"]
    Baseline["Baseline Install<br/>OS / kernel / driver / runtime"]
    BurnIn["Burn-in & Acceptance<br/>GPU / RDMA / NVMe / thermal / power"]
    Label["Classify & Label<br/>node pool / topology / flavor"]
    Join["Join Scheduler<br/>Kubernetes / Slurm / Ray"]
    Serve["Serve Workloads<br/>train / infer / dev / batch"]
    Monitor["Health Monitoring<br/>node / GPU / network / storage"]
    Drain["Cordon & Drain<br/>stop new work / evict safely"]
    Maintain["Maintenance<br/>repair / upgrade / config change"]
    Validate["Post-maint Validation<br/>smoke / benchmark / compare"]
    Return["Return to Pool<br/>un-drain / enable scheduling"]
    Retire["Retire<br/>wipe / de-register / asset close"]

    Arrival --> Inventory
    Inventory --> Baseline
    Baseline --> BurnIn
    BurnIn --> Label
    Label --> Join
    Join --> Serve
    Serve --> Monitor
    Monitor --> Drain
    Drain --> Maintain
    Maintain --> Validate
    Validate --> Return
    Return --> Serve
    Monitor --> Retire
```

这张图表达一个基本原则：节点不是“可用/不可用”两个状态，而是一组受控阶段。

每个阶段都应该有：

- 进入条件。
- 验证命令或指标。
- 失败处理。
- 责任人。
- 记录位置。
- 回滚方式。

## 为什么 AI 节点生命周期更复杂

普通计算节点关注 CPU、内存、磁盘、网络和操作系统。AI 节点还要额外关注：

- GPU 型号、显存、SM 架构。
- GPU firmware、NVSwitch firmware。
- NVIDIA driver 和 CUDA compatibility。
- NCCL/RCCL、RDMA、GPUDirect RDMA。
- PCIe、NUMA、NVLink、NVSwitch 拓扑。
- HBM ECC、Xid、retired pages、thermal throttling。
- 本地 NVMe cache。
- 大功耗、大热量和供电冗余。
- 大模型加载和 checkpoint 流量。
- 节点间拓扑对训练性能的影响。

这意味着 AI 节点不能只通过“能 SSH、能看到 GPU”就入池。它必须通过面向 AI workload 的验收。

## 生命周期阶段

### 1. 到货与资产登记

节点进入机房后，首先要把物理世界的信息变成平台可查询的资产记录。

至少记录：

- 资产编号。
- 厂商、型号、序列号。
- GPU 型号、数量、显存。
- CPU 型号、内存容量。
- NIC/HCA 型号、端口速率。
- 本地磁盘和 NVMe。
- 电源、PDU、机柜、U 位。
- 交换机端口、线缆编号。
- 保修和支持信息。
- 计划归属的 node pool。

资产记录不是财务表格而已。后续排查“某批节点性能异常”时，往往要按服务器批次、GPU 批次、NIC firmware、机柜、交换机端口做关联分析。

### 2. 物理验收

物理验收关注硬件和布线是否符合设计。

检查项包括：

- GPU 数量是否正确。
- NIC 是否插在预期 PCIe slot。
- NVLink/NVSwitch 是否识别。
- RDMA 网口是否连到正确网络。
- 管理网、业务网、存储网是否分离。
- 电源是否双路或符合冗余要求。
- 风道、温度、功耗是否正常。
- BIOS/UEFI 设置是否符合集群基线。

很多性能问题来自物理层：

- 网线插错交换机。
- NIC 插在带宽不足的 PCIe slot。
- NUMA 亲和性不符合预期。
- BIOS 省电策略导致频率异常。
- 风道问题导致 GPU 降频。

这些问题如果在入池后才发现，排查成本会高很多。

### 3. 软件基线安装

软件基线包括：

- OS image。
- kernel version。
- kernel 参数。
- NVIDIA driver。
- container runtime。
- NVIDIA Container Toolkit。
- RDMA/OFED stack。
- 文件系统和挂载参数。
- 安全 agent。
- 监控 agent。
- GPU device plugin 或 Slurm GRES 配置。

关键是基线要可声明、可重建、可比较。

不建议把节点变成“手工调出来的状态”。更好的方式是：

```text
node profile
  -> base image
  -> driver bundle
  -> runtime config
  -> health checks
  -> labels / taints
```

例如：

| Node Profile | 说明 |
| --- | --- |
| `h100-sxm-cu12-train` | H100 训练节点，RDMA 和 NCCL 优化 |
| `a100-cu11-legacy` | 旧 CUDA 11 任务兼容池 |
| `l40s-infer` | 推理节点，强调模型 cache 和服务稳定性 |
| `cpu-storage-etl` | 数据预处理节点，强调 CPU、内存和存储吞吐 |

### 4. Burn-in 与验收测试

Burn-in 是节点入池前的压力测试。它的目标不是跑一个 demo，而是尽早暴露硬件、散热、供电、网络和驱动问题。

建议分层测试。

| 层 | 测什么 | 示例 |
| --- | --- | --- |
| 基础识别 | 设备是否完整 | GPU/NIC/NVMe/NUMA |
| GPU 健康 | 显存、ECC、温度、功耗 | DCGM diagnostics、压力测试 |
| 计算性能 | 矩阵乘、Tensor Core | GEMM、框架 smoke test |
| GPU 拓扑 | NVLink/NVSwitch/PCIe | topo matrix、bandwidth |
| 网络 | RDMA、带宽、延迟 | ib_write_bw、NCCL all-reduce |
| 存储 | 本地 NVMe、共享 FS | fio、dataset read、checkpoint |
| 长稳 | 长时间运行 | 多小时训练/通信压力 |

AI 节点至少应通过：

- 单卡计算 smoke test。
- 多卡 collective test。
- 多节点 NCCL/RCCL test。
- 本地 NVMe 读写测试。
- 共享存储读写测试。
- 监控指标上报测试。
- 温度和功耗稳定性测试。

如果节点是训练池，还要验证多节点拓扑；如果节点是推理池，还要验证模型加载、服务启动、冷启动和 p99 latency。

### 5. 分类、打标签与入池

节点验收通过后，才应该加入可调度资源池。

Kubernetes 中通常会使用：

- labels 表达 GPU 型号、节点池、拓扑、zone。
- taints 限制哪些 workload 能调度上来。
- device plugin 暴露 GPU。
- Node Feature Discovery 暴露硬件特征。
- ResourceQuota/queue 系统控制谁能使用。

Slurm 中通常会使用：

- partition。
- node features。
- GRES。
- account/QoS。
- reservation。

节点标签不能随意命名。建议统一资源 flavor：

```text
gpu.vendor=nvidia
gpu.product=h100
gpu.memory=80gb
gpu.interconnect=nvswitch
node.pool=training
network.rdma=true
storage.local-nvme=true
```

标签是调度、可观测性、成本归因和故障分析的共同语言。

### 6. 运行期健康管理

节点入池后，要持续判断它是否仍适合承载 workload。

Kubernetes Node Status 包含 conditions、addresses、capacity/allocatable、info 等信息。常见 condition 包括 Ready、DiskPressure、MemoryPressure、PIDPressure、NetworkUnavailable。

AI 集群还需要额外健康信号：

- GPU Xid error。
- ECC 错误。
- retired pages。
- GPU 温度和降频。
- NVLink/NVSwitch 错误。
- RDMA port error。
- NCCL test 异常。
- 本地 NVMe wear、IO error、容量满。
- 共享存储 mount 异常。
- driver/device plugin 异常。
- container runtime 异常。

健康管理不是“看到问题报警”就够。平台要能根据严重程度采取动作：

| 严重程度 | 动作 |
| --- | --- |
| 轻微指标异常 | 告警、标记观察 |
| 可疑性能下降 | 降低调度优先级、进入 suspect |
| 单次可恢复错误 | 自动恢复、记录事件 |
| 反复错误 | cordon、drain、人工检查 |
| 明确硬件故障 | 下线、报修、替换 |

### 7. Cordon、Drain 与维护

维护节点前，不能直接杀进程或关机。

Kubernetes 里常见动作：

```text
cordon: 不再调度新 Pod
drain: 驱逐已有 Pod
maintenance: 做维护动作
uncordon: 恢复调度
```

Kubernetes 官方文档提供 `kubectl drain` 用于安全地从节点驱逐 Pod，并把节点标记为不可调度。AI 任务使用 drain 时要特别注意：

- 训练任务是否有 checkpoint。
- 推理服务是否有足够副本。
- PodDisruptionBudget 是否允许驱逐。
- eviction grace period 是否足够。
- local storage 是否需要保留或清理。
- job controller 是否会重试。
- checkpoint 是否已经完成。

Slurm 中类似动作包括：

- drain node。
- resume node。
- 设置 node reason。
- 停止新 job。
- 等待已有 job 结束或主动 requeue。

对 AI workload 来说，安全 drain 的难点是“任务状态大”。一个大训练任务可能需要几分钟甚至更久才能保存 checkpoint；一个推理节点可能需要先把流量摘除、等待请求清空、再停止服务。

### 8. 维护后验证

维护后不要直接恢复调度。需要先验证：

- 节点是否 Ready。
- GPU 是否全部识别。
- driver/CUDA/NCCL 版本是否符合 profile。
- device plugin 是否正常上报。
- RDMA 是否可用。
- 本地 NVMe 和共享存储是否正常。
- 基础 smoke test 是否通过。
- 与同类节点相比性能是否异常。
- 监控和日志是否恢复。

建议把维护后验证分成两类：

```text
smoke validation: 快速判断能不能回池
performance validation: 判断是否和同类节点一致
```

前者适合每次维护后都跑；后者适合驱动、firmware、kernel、网络、存储变更后跑。

### 9. 下线与退役

节点下线不是简单从调度器删除。

需要处理：

- 从调度器移除。
- 从监控和告警中移除。
- 清理本地 NVMe cache。
- 清理镜像缓存。
- 清理临时数据。
- 撤销节点证书和凭据。
- 更新资产系统。
- 回收或销毁磁盘。
- 关闭保修/维修工单。
- 记录下线原因。

如果下线流程不完整，会留下：

- 噪声告警。
- 失效节点仍显示在 dashboard。
- 数据残留。
- 证书和 secret 残留。
- 成本和资产统计错误。

## 节点状态模型

建议为 AI 节点定义比 Ready/NotReady 更细的状态。

| 状态 | 含义 |
| --- | --- |
| `new` | 新到货，未验收 |
| `installing` | 正在装系统或配置 |
| `burn-in` | 正在压力测试 |
| `ready-to-join` | 验收通过，待入池 |
| `active` | 正常承载 workload |
| `suspect` | 有异常信号，限制调度 |
| `cordoned` | 不接收新任务 |
| `draining` | 正在迁出任务 |
| `maintenance` | 正在维护 |
| `validating` | 维护后验证 |
| `retired` | 已下线 |

这个状态模型可以映射到：

- Kubernetes labels/taints。
- Slurm node state/reason。
- 资产系统状态。
- CMDB。
- 运维工单。
- 监控告警抑制规则。

核心是让系统、用户和运维看到同一事实。

## 入池验收清单

新节点入池前建议至少检查：

- 资产信息完整。
- 机柜、端口、线缆记录完整。
- OS/kernel/driver 符合 node profile。
- GPU 数量、型号、显存正确。
- GPU ECC/Xid/retired pages 正常。
- GPU 拓扑符合设计。
- CPU NUMA 与 GPU/NIC 亲和性符合设计。
- NIC/RDMA link up，速率正确。
- RDMA bandwidth/latency 通过。
- NCCL/RCCL 单机和多机测试通过。
- 本地 NVMe 性能和容量正常。
- 共享存储挂载正常。
- container runtime 正常。
- GPU device plugin 或 Slurm GRES 正常。
- 监控 agent 正常上报。
- 日志 agent 正常上报。
- 温度、功耗、风扇稳定。
- 长稳测试通过。
- 节点 labels/taints/features 正确。
- 入池记录和验收报告归档。

## 健康检查设计

健康检查要区分“节点是否存活”和“节点是否适合跑 AI 任务”。

### Liveness

节点是否还活着：

- kubelet 或 slurmd 是否在线。
- SSH/管理面是否可达。
- 心跳是否正常。
- container runtime 是否可用。

### Readiness

节点是否能接任务：

- GPU 是否可见。
- device plugin 是否正常。
- driver 是否匹配。
- 本地存储是否可写。
- 网络是否可用。
- 必要 daemon 是否运行。

### Performance Health

节点性能是否正常：

- GPU clock 是否异常。
- HBM 带宽是否异常。
- NCCL bandwidth 是否异常。
- RDMA error counter 是否异常。
- NVMe latency 是否异常。
- 温度/功耗是否触发 throttle。

### Workload Health

节点上的 workload 是否正常：

- 训练 step time 是否异常。
- 推理 p99 是否异常。
- DataLoader wait 是否异常。
- checkpoint duration 是否异常。
- 失败和重启是否集中在某些节点。

如果多个 job 在同一节点上表现异常，节点应进入 suspect，而不是让用户反复踩坑。

## 故障隔离策略

节点异常时，不同故障要有不同策略。

| 故障 | 典型动作 |
| --- | --- |
| kubelet/slurmd 短暂异常 | 自动恢复、记录事件 |
| 单次 GPU Xid | 标记观察，必要时重启 GPU/节点 |
| 反复 GPU Xid | cordon/drain，硬件检查 |
| ECC 错误增加 | 降级或下线检查 |
| RDMA 错误 | 排查 NIC、线缆、交换机、PFC/ECN |
| NVMe 错误 | 停止使用本地 cache，维护更换 |
| 温度异常 | 降载、检查风道和功耗 |
| 性能异常但无硬错误 | suspect、跑 benchmark 对比 |

故障隔离的关键不是“一出错就重启”，而是避免坏节点反复进入生产队列。

## 升级策略

节点升级包括：

- OS。
- kernel。
- NVIDIA driver。
- CUDA userspace。
- NCCL/RCCL。
- container runtime。
- GPU Operator/device plugin。
- OFED/RDMA stack。
- firmware。
- BIOS。
- 监控 agent。

升级风险很高，因为 AI workload 依赖复杂。

推荐流程：

```text
candidate profile
  -> canary node
  -> smoke test
  -> representative benchmark
  -> small canary workload
  -> limited node pool rollout
  -> full rollout
  -> old profile deprecation
```

不要直接全量升级训练池。先选择少量 canary 节点，跑：

- GPU smoke test。
- NCCL/RDMA test。
- 数据读取测试。
- checkpoint save/restore。
- 典型训练 step。
- 推理服务启动和压测。
- 自定义 CUDA/Triton kernel 编译。

升级完成后要能比较：

- step time 是否变化。
- NCCL bandwidth 是否变化。
- 推理 p99 是否变化。
- 错误率是否变化。
- 功耗和温度是否变化。

## 配置漂移

配置漂移是长期运维的大敌。

常见漂移：

- 某批节点 driver 版本不同。
- 某些节点 kernel 参数不同。
- 手工安装了调试包。
- RDMA 配置不同。
- BIOS 设置不同。
- 容器运行时配置不同。
- 节点 label 被临时修改后没有恢复。
- 监控 agent 版本不一致。

治理方法：

- 节点 profile 声明化。
- 定期配置审计。
- drift detection。
- immutable node image。
- 禁止手工修复不留记录。
- 所有临时变更必须有过期时间。
- 维护后自动验证并更新资产状态。

AI 集群里，配置漂移经常表现为“只有某些节点慢”。这类问题很难靠用户自己定位。

## 自动扩缩容与电源管理

部分集群会使用弹性节点、云上实例、spot/preemptible 资源，或者 Slurm power saving。

自动扩缩容要处理：

- 节点启动时间。
- 镜像预热。
- 驱动和 device plugin 准备。
- 模型权重和数据 cache 预热。
- 节点入池前健康检查。
- 空闲多久后缩容。
- 缩容前如何 drain。
- spot/preemptible 中断通知。

AI 节点启动慢，尤其是推理节点。模型权重很大，冷启动会显著影响 SLA。所以扩容不能只看“节点 Ready”，还要看：

- GPU device ready。
- 模型 cache ready。
- 服务 warmup ready。
- 网络和存储 ready。

## 数据与缓存清理

节点维护和下线时必须处理本地数据。

常见本地数据：

- dataset cache。
- model cache。
- compiled kernel cache。
- TensorRT engine。
- checkpoint staging。
- temporary shard。
- logs。
- core dump。
- 用户手工文件。

清理策略要区分：

- 正常维护：保留可复用 cache，提高回池速度。
- 故障排查：保留现场，避免覆盖证据。
- 租户切换：清理敏感数据。
- 下线退役：彻底清除数据和凭据。

不要让本地 NVMe 变成无人管理的共享垃圾场。容量满会导致节点看似 Ready，但任务启动失败或性能下降。

## Runbook

节点运维需要标准 runbook。

### 新节点入池

```text
register asset
install baseline
run burn-in
apply labels/taints
join scheduler
run smoke test
enable scheduling
archive acceptance report
```

### 计划维护

```text
announce maintenance
cordon node
drain workloads
confirm checkpoint/service handoff
apply maintenance
run validation
uncordon node
watch metrics
close ticket
```

### 异常节点隔离

```text
detect anomaly
mark suspect
collect diagnostics
cordon if needed
drain if workload-safe
run hardware/software checks
repair or retire
post-maint validation
return or replace
```

### 节点退役

```text
cordon and drain
remove from scheduler
wipe local data
revoke credentials
remove monitoring targets
update asset system
close capacity record
```

## 常见误区

### 误区一：节点 Ready 就能跑 AI 任务

Ready 只说明节点达到调度系统的基本条件。GPU、RDMA、NCCL、NVMe、热稳定性和应用性能还需要额外验证。

### 误区二：维护就是重启

重启可能暂时掩盖问题，但不会解释根因，也不会保证节点适合回池。

### 误区三：坏节点让用户重试就行

如果节点持续异常，重试只会浪费更多 GPU hour，并污染用户对平台的信任。

### 误区四：驱动升级可以全量推

AI stack 对 driver/CUDA/NCCL/框架组合敏感。升级必须有 canary、benchmark 和回滚。

### 误区五：本地缓存不需要管

本地 NVMe cache 会影响启动、数据读取、checkpoint staging 和隐私边界。必须有容量和清理策略。

### 误区六：资产系统和调度系统无关

资产、拓扑、故障、成本和容量规划都依赖节点身份一致。

## 设计检查清单

- 是否有统一的节点 profile。
- 新节点是否必须 burn-in 后才能入池。
- 是否有 GPU、RDMA、NVMe、NCCL、温度、功耗验收测试。
- 节点 labels/taints/features 是否标准化。
- 是否能从节点追溯到资产、机柜、交换机端口和硬件批次。
- 是否区分 Ready、suspect、maintenance、retired 等状态。
- 是否有自动健康检查和异常隔离策略。
- 是否记录 GPU Xid、ECC、retired pages、thermal throttling。
- 是否能发现某节点上的 workload 性能持续异常。
- 是否有安全的 cordon/drain 流程。
- 维护前是否考虑 checkpoint、PDB、grace period、服务摘流。
- 维护后是否有 smoke validation 和 performance validation。
- 驱动、firmware、kernel 升级是否有 canary。
- 是否有配置漂移检测。
- 本地 NVMe cache 是否有容量、保留和清理策略。
- 节点退役是否清理数据、证书、监控和资产记录。
- 是否有新节点入池、计划维护、异常隔离、退役 runbook。

## 小结

AI 节点生命周期可以概括为：

```text
asset
  -> baseline
  -> burn-in
  -> classify
  -> join pool
  -> serve workload
  -> monitor health
  -> isolate / maintain
  -> validate
  -> return or retire
```

节点治理的关键是把“节点能不能跑”变成可验证的工程事实。只有当入池、维护、升级和下线都有标准流程，集群才不会被少数坏节点、漂移配置和粗暴维护拖慢。

这也是后续可靠性、可观测性和故障复盘章节的基础。

## 延伸阅读

- [Kubernetes Node Status](https://kubernetes.io/docs/reference/node/node-status/)
- [Kubernetes Safely Drain a Node](https://kubernetes.io/docs/tasks/administer-cluster/safely-drain-node/)
- [Kubernetes Node Shutdown](https://kubernetes.io/docs/concepts/cluster-administration/node-shutdown/)
- [Kubernetes Monitor Node Health](https://kubernetes.io/docs/tasks/debug/debug-cluster/monitor-node-health/)
- [Kubernetes Upgrade Nodes](https://kubernetes.io/docs/tasks/administer-cluster/kubeadm/upgrading-linux-nodes/)
- [NVIDIA GPU Operator Overview](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/overview.html)
- [NVIDIA DCGM Diagnostics](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html#diagnostics)
- [Slurm Power Saving Guide](https://slurm.schedmd.com/power_save.html)
