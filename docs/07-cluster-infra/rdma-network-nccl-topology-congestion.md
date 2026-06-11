---
title: RDMA 网络与 NCCL 拓扑：InfiniBand、RoCE 与拥塞控制
domain: cluster-infra
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-06-12
---

# RDMA 网络与 NCCL 拓扑：InfiniBand、RoCE 与拥塞控制

多机 AI 训练和分布式推理的瓶颈，常常不在单卡算力，而在网络。

当一个模型跨越多台服务器后，GPU 之间必须同步梯度、参数、activation、expert token、KV Cache 或中间状态。网络如果慢、抖动、拥塞或拓扑不匹配，就会让 GPU 等通信，最终表现为 step time 变长、tokens/s 下降、p99 变差、扩展效率低。

这篇关注的问题是：

> AI 集群网络如何支撑 NCCL/RCCL collective、RDMA、InfiniBand/RoCE、GPU Direct RDMA、拓扑感知调度和拥塞控制？

## 一张总图

```mermaid
flowchart TB
    subgraph NodeA["Node A"]
        GA0["GPU 0"]
        GA1["GPU 1"]
        NICA0["NIC 0"]
        NICA1["NIC 1"]
        GA0 <--> NICA0
        GA1 <--> NICA1
    end

    subgraph NodeB["Node B"]
        GB0["GPU 0"]
        GB1["GPU 1"]
        NICB0["NIC 0"]
        NICB1["NIC 1"]
        GB0 <--> NICB0
        GB1 <--> NICB1
    end

    subgraph Fabric["Cluster Network Fabric"]
        Leaf0["Leaf / ToR 0"]
        Leaf1["Leaf / ToR 1"]
        Spine0["Spine 0"]
        Spine1["Spine 1"]
        Leaf0 <--> Spine0
        Leaf0 <--> Spine1
        Leaf1 <--> Spine0
        Leaf1 <--> Spine1
    end

    NICA0 <--> Leaf0
    NICA1 <--> Leaf1
    NICB0 <--> Leaf0
    NICB1 <--> Leaf1

    App["Training / Serving Runtime<br/>DDP / FSDP / TP / EP / P-D"]
    NCCL["NCCL / RCCL / MPI<br/>Collectives and P2P"]
    RDMA["RDMA Verbs / Driver / NIC"]

    App --> NCCL
    NCCL --> RDMA
    RDMA --> NICA0
```

这张图表达几个关键点：

- AI 网络不是普通东西向流量，它承载同步、突发、大带宽的 collective。
- NCCL/RCCL 会根据 GPU、NIC、PCIe、NVLink 和网络拓扑选择通信路径。
- 多 rail、多 NIC、多交换机只有被正确使用时才有价值。
- 拥塞、错误和尾延迟会直接暴露到训练 step time 或推理 p99。

## RDMA 是什么

RDMA 是 Remote Direct Memory Access。核心思想是让一台机器的网卡直接读写另一台机器的内存区域，尽量绕过 CPU copy 和内核协议栈开销。

对 AI 集群来说，RDMA 的价值是：

- 降低通信延迟。
- 减少 CPU 参与。
- 提高大 tensor 传输带宽。
- 支撑 NCCL/RCCL/MPI 等通信库。
- 和 GPU Direct RDMA 结合，让 NIC 直接访问 GPU memory。

简化路径：

```text
普通路径:
GPU memory -> host memory -> kernel/network stack -> NIC -> network

理想 GPU Direct RDMA 路径:
GPU memory -> NIC -> network
```

实际路径受 GPU/NIC 拓扑、driver、IOMMU、ACS、firmware、通信库和系统配置影响。不能只因为机器有 RDMA NIC，就假设所有通信都走到了理想路径。

## GPU Direct RDMA 路径要验证

GPU Direct RDMA 的目标是让 NIC 能直接访问 GPU memory，减少经过 host memory 的拷贝路径。它对多机训练很重要，但它不是“装了 RDMA 网卡就自动成立”。

要让路径真正高效，通常要同时满足：

- GPU、NIC、PCIe 拓扑支持。
- driver、firmware、CUDA、OFED/rdma-core 版本兼容。
- IOMMU、ACS、BAR、peer memory 等系统配置正确。
- 通信库识别到可用的 GPU/NIC 路径。
- 容器运行时没有屏蔽必要设备和权限。
- NUMA 绑定和 NIC 选择没有走远端路径。

可以把路径分成三种层级理解：

| 路径 | 直觉 | 风险 |
| --- | --- | --- |
| GPU Direct RDMA | GPU memory 直接通过 NIC 访问网络 | 最理想，但依赖拓扑和软件栈 |
| GPU -> Host pinned memory -> NIC | 经过 host memory staging | CPU、NUMA、内存带宽可能成为瓶颈 |
| GPU -> Host pageable memory -> kernel/network stack | 普通路径 | 延迟高、CPU 开销大，不适合大规模 collective |

平台应该在节点验收和升级后验证 GPU Direct RDMA，而不是等用户训练慢了再排查。

常见验证包括：

- 同节点检查 GPU/NIC affinity。
- 用 RDMA microbenchmark 测 host memory 与 GPU memory 路径。
- 用 NCCL tests 测单机、多机、单 rail、多 rail。
- 保存 benchmark 时的 driver、CUDA、NCCL、OFED、firmware、kernel、BIOS 配置。
- 对照 topology manifest，确认性能异常不是来自错误放置。

如果没有路径验证，问题很容易被误判为“模型代码慢”或“框架通信差”。实际上可能只是某批节点没走到预期的 GPUDirect RDMA 路径。

## InfiniBand 与 RoCE

AI 集群常见两类 RDMA fabric：

| 类型 | 直觉 | 常见特点 |
| --- | --- | --- |
| InfiniBand | 专用 RDMA/HPC 网络 | 性能和拥塞控制能力强，HPC/训练集群常见 |
| RoCE | 在 Ethernet 上承载 RDMA | 结合以太网生态，但对无损/拥塞配置要求高 |

### InfiniBand

InfiniBand 是专用高性能网络 fabric。它有自己的链路层、交换体系、Subnet Manager、服务等级和拥塞管理能力。

适合：

- 大规模同步训练。
- 高带宽 AllReduce / ReduceScatter。
- 低延迟多节点通信。
- HPC 风格集群。
- 对网络性能和可预测性要求高的场景。

InfiniBand 的优势通常在于它是为 HPC/RDMA 这类 workload 设计的，网络语义和工具链更专用。

### RoCE

RoCE 是 RDMA over Converged Ethernet。它把 RDMA 语义运行在 Ethernet 上。

RoCE 的吸引力是：

- 使用以太网生态。
- 可以和现有数据中心网络能力结合。
- RoCEv2 可以跨三层网络。
- 设备和运维体系更接近普通数据中心网络。

但 RoCE 的挑战在于：RDMA 对丢包和重传非常敏感，RoCE 网络通常需要正确配置 PFC、ECN、QoS、buffer、DSCP/priority、交换机队列和拥塞控制。NVIDIA RoCE 文档也强调，RoCE 需要某种流控形式，常用方式是 Priority Flow Control，并且路径上的端点和交换机都要启用。

### 怎么选择

粗略判断：

- 大规模训练、追求极致稳定和性能：InfiniBand 常更自然。
- 数据中心以太网能力强、希望统一网络生态：RoCE 值得评估。
- 小规模集群：两者都可能可行，关键看团队运维能力。
- 推理东西向流量和训练混合：要重点看隔离、QoS 和拥塞。

不要只比较单口带宽。要比较端到端：

- collective performance。
- 拥塞行为。
- 多租户隔离。
- 故障排查工具。
- 交换机配置复杂度。
- 运维团队经验。
- 成本和供应链。

## NCCL/RCCL 承载什么通信

NCCL/RCCL 是 GPU collective 通信库，常用于多 GPU 和多节点训练/推理。

常见通信模式：

| 模式 | 用途 |
| --- | --- |
| AllReduce | DDP 梯度同步 |
| ReduceScatter | FSDP/ZeRO 梯度分片、部分 TP |
| AllGather | FSDP 参数收集、TP 输出拼接 |
| AllToAll | MoE expert dispatch/combine |
| Broadcast | 参数初始化、配置同步 |
| P2P send/recv | Pipeline Parallel、KV transfer |

不同通信模式对网络压力不同。

AllReduce 可以用 ring/tree/hierarchical 算法优化。ReduceScatter 和 AllGather 常和 sharding 绑定。AllToAll 对网络 bisection bandwidth 和负载均衡特别敏感。P2P 对特定路径和尾延迟敏感。

训练和推理常见映射：

| 并行/系统模式 | 网络压力 |
| --- | --- |
| DDP | step 内 gradient AllReduce |
| FSDP / ZeRO | 多层 AllGather / ReduceScatter |
| Tensor Parallel | 层内高频 collective，跨节点代价很高 |
| Pipeline Parallel | stage 间 activation P2P |
| Expert Parallel / MoE | AllToAll，突发且易不均衡 |
| P/D 分离推理 | KV Cache transfer |
| RAG / Agent | 服务间 RPC、检索网络和 LLM serving 网络叠加 |

## 通信模式到网络需求的映射

不同通信模式对网络的要求不同。设计网络和调度策略时，不能只看“总带宽”。

| 通信模式 | 主要敏感点 | 常见症状 |
| --- | --- | --- |
| AllReduce | 带宽、算法、rank 数、tail rank | step time 随节点数上升过快 |
| ReduceScatter / AllGather | 带宽、bucket 粒度、overlap | backward 中通信气泡变大 |
| AllToAll | bisection bandwidth、负载均衡、拥塞 | MoE step 抖动、某些 rank 长尾 |
| P2P send/recv | 路径延迟、stage 邻近性 | pipeline bubble、KV transfer 慢 |
| Broadcast | fan-out、初始化阶段拥塞 | job 启动慢、模型加载同步慢 |
| Parameter / checkpoint traffic | 存储网络、突发写入、metadata | checkpoint 时间拉长，影响下一步 |
| Serving RPC | p95/p99、排队、服务网络隔离 | 在线推理尾延迟抖动 |

这张表的意义是：同样是“网络慢”，原因可能完全不同。

- DDP 慢，通常先看 AllReduce 和跨节点带宽。
- FSDP/ZeRO 慢，要看 ReduceScatter/AllGather、bucket 和 overlap。
- MoE 慢，要重点看 AllToAll、token imbalance 和 bisection bandwidth。
- P/D 分离慢，要看 KV transfer 的路径、批量粒度和网络隔离。
- checkpoint 慢，可能不是训练网络，而是存储网络、metadata 或对象存储入口。

因此，网络 benchmark 要覆盖真实通信模式，而不是只跑单个带宽数字。

## NCCL Topology

NCCL 不只是调用网络，它会理解拓扑并选择路径。

NCCL 需要考虑：

- GPU 与 GPU 的关系。
- GPU 与 NIC 的关系。
- PCIe switch。
- CPU NUMA。
- NVLink/NVSwitch。
- 多 NIC。
- network interface。
- rank mapping。
- collective algorithm。
- channel 数量。

NCCL 文档中也有大量与网络和拓扑相关的环境变量，例如 `NCCL_IB_HCA`、`NCCL_SOCKET_IFNAME`、`NCCL_CROSS_NIC`、`NCCL_TOPO_FILE`、`NCCL_TOPO_DUMP_FILE`、`NCCL_DEBUG`、`NCCL_DEBUG_SUBSYS` 等。

这些变量不是让用户随便调，而是说明 NCCL 的性能和拓扑、网卡选择、调试信息强相关。

### Topology Dump

排查问题时，可以让 NCCL dump topology，结合：

- `nvidia-smi topo -m`。
- NCCL debug log。
- hostfile / rank order。
- network interface。
- switch port counters。
- profiler timeline。

目标是确认：

- NCCL 识别到的拓扑是否正确。
- 是否选到了预期 NIC。
- 是否多 NIC 都在使用。
- 是否跨了错误 NUMA。
- 是否使用了 P2P/NVLink。
- collective 是否分层。

### NCCL 环境变量治理

NCCL 环境变量很强，但不应该变成“每个用户自己试参数”。

平台应把环境变量分成三类：

| 类别 | 例子 | 管理方式 |
| --- | --- | --- |
| 平台默认 | network interface、HCA 选择、debug 基线、async error handling | 由镜像、启动器或队列模板统一注入 |
| workload 可调 | algorithm/protocol、channel、buffer、timeout | 只允许有经验用户在 benchmark 中调整 |
| 排查专用 | topology dump、debug subsystem、trace | 临时打开，避免长期污染日志和性能 |

治理目标不是禁止用户调参，而是让默认路径足够正确：

- 默认选择训练网络接口，而不是管理网口。
- 默认选择拓扑邻近的 HCA。
- 默认支持多 rail 或明确禁用。
- 默认打开必要的异步错误处理。
- benchmark 和生产任务记录所有通信相关环境变量。
- 变更 NCCL/OFED/driver 后重新跑通信验收。

如果平台没有默认治理，用户会复制彼此的环境变量。某个参数在 A 集群有效，到了 B 集群可能错误选择网卡、禁用某些路径，或者掩盖真正的拥塞问题。

## 网络拓扑：不是所有节点等价

集群网络有拓扑。

常见形态：

- single switch。
- leaf-spine / Clos。
- fat-tree。
- dragonfly。
- torus / mesh。
- multi-rail。
- rail-optimized topology。

AI workload 特别关心：

- bisection bandwidth。
- oversubscription ratio。
- path diversity。
- hop count。
- rack locality。
- rail symmetry。
- congestion domain。
- failure domain。

一个 32 节点 job 如果分布在同 rack，可能表现很好；如果分散跨多个 rack、多个 pod，通信路径变长、拥塞变多，step time 可能变差。

所以调度要能表达：

- 尽量同 rack。
- 尽量同 rail。
- 尽量同网络 island。
- 避免跨低带宽边界。
- 大 job 独占一段拓扑。
- MoE/AllToAll 避免被放到 bisection bandwidth 弱的位置。

## Network Topology Manifest

和 GPU 拓扑一样，网络拓扑也应该被记录成 manifest，而不是只存在于网络团队的图纸里。

一个 AI 集群的 network topology manifest 可以包含：

| 字段 | 示例 | 用途 |
| --- | --- | --- |
| Node location | rack、leaf、pod、row、room | 调度同 rack / 同 pod / 跨故障域 |
| NIC inventory | NIC 型号、端口、速率、PCI bus、NUMA | 绑定 GPU/NIC、定位硬件差异 |
| Rail mapping | NIC0 -> rail0，NIC1 -> rail1 | 多 rail 对齐和排查不均衡 |
| Switch port | leaf port、spine path、LAG/ECMP 信息 | 定位拥塞、错误和链路故障 |
| Fabric type | IB、RoCE、Ethernet、混合 | 决定配置和 benchmark 方法 |
| QoS class | PFC priority、ECN profile、DSCP/PCP | 确认流量分类一致 |
| Oversubscription | rack、pod、fabric 级别比例 | 判断大 job 是否跨越弱边界 |
| Failure domain | leaf、spine、rail、power domain | 服务副本分散和训练风险评估 |
| Health state | link flap、drop、error、degraded port | 自动隔离异常节点或链路 |

这个 manifest 应该能和 job manifest 对上：

- job 分配到了哪些节点。
- 每个 rank 使用哪个 GPU 和 NIC。
- 这些 NIC 属于哪个 rail。
- 流量经过哪些 rack / leaf / pod。
- 当时 QoS、PFC、ECN、driver、firmware 配置是什么。

没有 network manifest，平台很难解释“同样 64 张 GPU，为什么这次慢 20%”。可能原因是跨了更多 rack、走了不同 rail、某个 leaf 有拥塞、某些端口降速，或者 RoCE QoS 配置漂移。

## Multi-Rail 网络

Multi-rail 指一台服务器有多张 NIC，连接到多个网络 rail。

理想目标：

```text
GPU 0/1/2/3 -> NIC 0 -> Rail 0
GPU 4/5/6/7 -> NIC 1 -> Rail 1
跨节点通信均匀使用多个 rail
```

多 rail 的好处：

- 提高总带宽。
- 降低单 rail 拥塞。
- 增加路径多样性。
- 更好匹配 GPU/NIC affinity。

但多 rail 不是自动生效。

需要确认：

- 每张 NIC 是否连到正确 rail。
- NCCL/RCCL 是否使用多个 NIC。
- rank mapping 是否均匀。
- 每个 rail 的交换机配置一致。
- RoCE 的 PFC/ECN/QoS 是否在所有 rail 一致。
- 监控是否能分 rail 看流量和错误。

一个常见问题是硬件有多张 NIC，但通信库实际只用了其中一张。

## 拥塞为什么严重

AI 训练网络有同步特征。

例如一个 FSDP 训练 step：

1. 多个 rank 同时 backward。
2. 多个 bucket 同时 ReduceScatter。
3. 下一层又触发 AllGather。
4. 所有节点周期性产生相似流量。

这种流量不是平滑 Web 请求，而是同步、周期性、突发的。

拥塞会导致：

- collective time 增加。
- tail rank 拖慢所有 rank。
- GPU 空等。
- step time 抖动。
- NCCL timeout。
- RoCE pause storm。
- p99 latency 变差。
- 多 job 互相影响。

MoE AllToAll 更容易制造拥塞，因为每个 rank 都可能向多个 rank 发送不同大小的数据，且 token routing 可能不均衡。

## 拥塞域与故障域

网络拥塞不是均匀发生的。它通常发生在某个拥塞域里。

常见拥塞域包括：

- 单个 NIC。
- 单个 PCIe root complex。
- 单条 rail。
- 单个 leaf switch。
- leaf-spine 上行。
- 单个 rack。
- 存储入口。
- 对象存储 gateway。
- 服务网关或负载均衡器。

故障域和拥塞域也不总是相同。

例如：

- 一个 leaf switch 是故障域；它的上行带宽也是拥塞域。
- 两条 rail 能增加带宽；但如果 rank mapping 不均衡，某条 rail 会变成拥塞域。
- 存储网络和训练网络共用交换机时，checkpoint 写入可能让 collective 进入同一个拥塞域。
- 推理服务跨 rack 分散更利于高可用，但 P/D 分离或 KV transfer 可能更希望靠近。

调度系统需要知道这些边界：

- 大训练 job 尽量不要跨多个弱 bisection boundary。
- 多个 AllToAll-heavy job 不要堆在同一个 leaf 或 rail。
- 生产推理副本要跨故障域分散，但不能让关键路径延迟过大。
- checkpoint-heavy job 和 collective-heavy job 要避免同时冲击同一网络/存储入口。

把拥塞域纳入调度，比事后调 NCCL 参数更有效。

## RoCE 拥塞控制：PFC、ECN 与 QoS

RoCE 在 Ethernet 上跑 RDMA，需要认真配置无损或低丢包网络行为。

常见机制：

- PFC：Priority Flow Control，按优先级 pause 流量。
- ECN：Explicit Congestion Notification，用标记提示拥塞。
- QoS：把不同流量放入不同 priority / traffic class。
- DSCP / PCP：标记流量优先级。
- Buffer tuning：交换机 buffer 和队列配置。

PFC 能减少丢包，但配置不好会带来 pause storm、head-of-line blocking 和跨流量影响。ECN 配合端侧拥塞控制可以提前降速。QoS 用来把训练、存储、管理、服务流量隔离到不同 class。

RoCE 运维要点：

- 端点和交换机配置一致。
- PFC 只对需要的 priority 开启。
- ECN 阈值合理。
- DSCP/PCP 映射一致。
- 训练流量和存储/服务流量不要混成同一优先级。
- 监控 pause、ECN mark、drop、buffer、retransmit。

RoCE 的难点不是跑通，而是长期稳定跑快。

### RoCE 配置漂移

RoCE 集群很怕配置漂移。端点、交换机和队列策略只要有一处不一致，就可能出现“部分节点慢、部分 job 超时、偶发 pause storm”。

需要持续检查：

- 主机侧 priority、DSCP、PCP、traffic class 是否一致。
- NIC firmware、driver、OFED/rdma-core 是否一致。
- 交换机端口 PFC/ECN 阈值是否一致。
- 所有 rail 是否使用同样 QoS 策略。
- 训练流量是否真的打上预期 priority。
- 存储、服务和管理流量是否误入训练 priority。
- 新节点入池是否完成网络基线验证。

建议把 RoCE 配置纳入 cluster manifest，并在节点入池、交换机变更、驱动升级后自动跑通信验收。否则平台很难区分“模型变慢”和“网络配置漂移”。

## InfiniBand 拥塞与路由

InfiniBand 也会拥塞，尤其是在大规模同步 collective、AllToAll、多个 job 同时运行时。

关注点包括：

- Subnet Manager。
- routing。
- adaptive routing。
- service level。
- virtual lane。
- link error。
- port counters。
- credit / flow control。
- congestion control。
- SHARP / in-network reduction。

InfiniBand 的优势是体系更面向 HPC/RDMA，但仍需要拓扑规划、路由策略、故障监控和 job placement。

大 job 是否跨越多个网络 island，rank 是否和 rail 对齐，都会影响性能。

## 流量隔离

AI 集群常有多类网络流量：

- 训练 collective。
- checkpoint 写入。
- dataset 读取。
- model loading。
- 推理请求。
- RAG 检索。
- monitoring/logging。
- control plane。

这些流量不应无保护地混在一起。

典型隔离策略：

- 管理网络和训练网络分离。
- 存储网络和训练网络分离或 QoS 隔离。
- 推理服务网络与 batch 网络分离。
- checkpoint 流量限速或错峰。
- 大训练 job 独占特定网络分区。
- 多租户按 queue / namespace / project 做流量策略。

网络隔离的目标不是复杂化，而是防止一个 workload 的突发流量拖垮另一个 workload。

## 调度与网络拓扑

调度器需要网络信息。

至少要知道：

- 节点属于哪个 rack。
- 节点属于哪个 leaf / pod。
- 节点有哪些 NIC。
- NIC 属于哪个 rail。
- GPU/NIC affinity。
- 节点间是否同网络 island。
- 哪些节点适合组成大训练 job。

调度策略例子：

- TP 跨节点任务优先放在同一高带宽网络 island。
- MoE AllToAll 任务避免跨低 bisection boundary。
- 大训练 job 尽量获得连续节点集合。
- 多个大 job 不放在同一拥塞域。
- 推理服务避免和 checkpoint-heavy 任务共享关键网络。

如果调度器完全不知道网络拓扑，NCCL 再努力也只能在已经分配的节点上优化。

## 网络调度契约

一个通信密集 workload 应该和平台形成明确的网络调度契约。

这个契约至少回答：

| 问题 | 示例 |
| --- | --- |
| 需要什么网络能力 | IB、RoCE、GPU Direct RDMA、多 rail、低 oversubscription |
| 需要什么拓扑范围 | same rack、same pod、same rail、single island |
| 哪些是硬约束 | 64 GPU gang、H100-IB 节点、双 rail |
| 哪些是软偏好 | 尽量同 rack、尽量避开拥塞 leaf |
| 是否可与其他 job 混跑 | benchmark 独占、MoE 避免同拥塞域、普通 DDP 可共享 |
| 失败或慢时如何解释 | 保存 node/rank/NIC/rail/topology manifest |

平台可以把这些需求翻译成：

- queue / resource flavor。
- node label / affinity。
- rack 或 pod 级放置策略。
- gang scheduling。
- network-aware placement。
- NCCL/RCCL 启动模板。
- QoS class。
- 监控和告警规则。

没有契约时，用户只会写“我要 64 张 GPU”。调度器可能给到 64 张 GPU，但这些 GPU 跨越多个网络岛、rail 不均衡、GPU/NIC affinity 很差，最后表现为扩展效率低。

## 可观测性

网络问题必须可观测。

需要采集：

### NIC / Host

- RDMA bandwidth。
- packet drop。
- retry / retransmission。
- completion error。
- CQ error。
- RoCE GID / traffic class。
- NIC temperature。
- PCIe error。

### Switch

- port utilization。
- port error。
- pause frame。
- PFC storm。
- ECN mark。
- buffer occupancy。
- link flap。
- congestion counter。

### Communication Library

- NCCL debug log。
- collective type。
- collective duration。
- algorithm / protocol。
- channel count。
- selected NIC。
- topology dump。
- timeout / async error。

### Workload

- step time。
- exposed communication time。
- p95/p99 latency。
- rank skew。
- GPU idle time。
- tokens/s。
- job placement。

没有这些指标，网络问题会被误判成“模型慢”“GPU 有问题”或“框架不稳定”。

## 故障模式与归因

AI 网络故障经常不是完全不可用，而是性能退化。

常见故障模式：

| 现象 | 可能原因 | 证据 |
| --- | --- | --- |
| 某些 rank 长尾 | rank mapping 跨远端 NIC、某条 rail 拥塞、某端口错误 | NCCL log、rank timeline、switch port counter |
| step time 周期性抖动 | checkpoint、数据读取、其他 job 同步流量冲击 | 存储指标、网络利用率、job timeline |
| NCCL timeout | 丢包、PFC storm、链路 flap、某 rank 卡死 | NCCL async error、NIC error、switch log |
| 多 rail 带宽不均 | NCCL 只用单 NIC、rail mapping 错、ECMP 不均 | NIC bytes、NCCL selected HCA、topology manifest |
| RoCE 偶发慢 | PFC/ECN 配置漂移、buffer 阈值不一致 | pause frame、ECN mark、QoS config diff |
| 同节点正常，跨 rack 慢 | bisection bandwidth 不足、跨 pod oversubscription | placement、rack path、collective benchmark |
| 推理 p99 抖动 | 训练/存储流量混跑、P/D 传输跨远路径 | service metrics、KV transfer time、network class |

归因顺序建议：

```text
先确认 workload 和单机性能
  -> 再确认 rank/GPU/NIC/rail 映射
  -> 再看 NCCL/RCCL 选择的算法和接口
  -> 再看 NIC 与交换机计数器
  -> 再看同 rack / 跨 rack / 多 job 对比
  -> 最后调整环境变量、QoS 或调度策略
```

不要反过来一上来就改 NCCL 参数。参数可能掩盖问题，但不能替代拓扑、链路、拥塞和调度证据。

## Benchmark 方法

网络 benchmark 要从 micro 到 end-to-end。

### Microbenchmark

测基本能力：

- point-to-point RDMA bandwidth。
- RDMA latency。
- GPU Direct RDMA bandwidth。
- single NIC / multi NIC。
- same rack / cross rack。
- same rail / cross rail。

### Collective Benchmark

测 NCCL/RCCL：

- AllReduce。
- ReduceScatter。
- AllGather。
- AllToAll。
- Broadcast。
- P2P send/recv。

要覆盖不同 message size，因为小消息受 latency 影响，大消息受 bandwidth 影响。

### Component Benchmark

测真实 AI 组件：

- DDP gradient sync。
- FSDP bucket。
- TP layer collective。
- MoE dispatch/combine。
- P/D KV transfer。
- checkpoint write burst。

### End-to-End Benchmark

最终看：

- training step time。
- scaling efficiency。
- MFU/HFU。
- TTFT / TPOT。
- p95/p99。
- tokens/s/GPU。
- timeout / failure rate。
- 多 job 干扰。

单 job 网络 benchmark 很好，不代表多租户集群里也好。必须在接近真实负载下测拥塞。

### Benchmark Manifest

网络 benchmark 必须记录环境，否则结果不可复现。

建议记录：

- 节点列表、rack、leaf、pod、rail。
- GPU UUID、NIC、PCI bus id、NUMA。
- driver、CUDA、NCCL/RCCL、OFED/rdma-core、firmware、kernel。
- NCCL/RCCL 环境变量。
- RoCE PFC/ECN/QoS 或 InfiniBand routing/service level。
- message size、collective type、rank 数、process per node。
- 是否启用 GPU Direct RDMA。
- 是否有其他 job、checkpoint、存储流量干扰。
- switch/NIC counters 的采集时间窗口。
- raw data、脚本、commit、容器镜像。

Benchmark 结果不要只保存平均带宽。至少要保存 p50/p95/p99、最慢 rank、失败率、重试/丢包/pause/ECN、每 rail 带宽。AI 训练往往被最慢 rank 拖住，平均值会掩盖真正问题。

## 常见排查路径

看到多机训练慢，可以按下面查：

1. 单机性能是否正常。
2. GPU/NIC affinity 是否合理。
3. NCCL 是否选到预期 NIC。
4. 多 NIC 是否都在使用。
5. rank mapping 是否跨错拓扑。
6. NCCL debug 是否有 warning / timeout。
7. switch port 是否有 drop / pause / ECN。
8. RoCE PFC/ECN/QoS 是否一致。
9. 是否与 checkpoint 或其他 job 同时拥塞。
10. 同 rack 与跨 rack 性能差异是否异常。

看到推理 p99 抖动，可以查：

- P/D KV transfer 是否堵塞。
- 推理服务是否和训练共享网络。
- model loading 是否冲击网络。
- RAG 检索是否和 LLM serving 抢网络。
- 某些 replica 是否被放到远端网络路径。

## 常见优化方向

### 拓扑感知分配

把重通信 job 放在网络邻近节点上，比盲目增加节点更有效。

### 分层 collective

先节点内规约，再节点间通信，再节点内分发，通常比扁平跨所有 GPU 通信更符合硬件拓扑。

### 多 rail 对齐

让 GPU、NIC、rail、rank order 对齐，避免所有流量挤到一张 NIC 或一个 rail。

### 通信与计算重叠

训练中把 gradient bucket、FSDP prefetch、reduce-scatter 和 backward 计算重叠。推理中把 KV transfer 和调度、precompute、cache 管理重叠。

### 流量隔离和 QoS

把训练、存储、服务和管理流量分开或设置不同优先级。checkpoint 可以限速或错峰。

### 降低通信量

减少通信比优化网络更直接：

- 使用合适并行策略。
- 避免跨节点 TP。
- 使用 FSDP/ZeRO 合理 bucket。
- MoE 做 load balance。
- 梯度或 activation 压缩要谨慎评估精度和收益。
- KV transfer 做 cache 和 locality 优化。

## 常见误区

### 误区一：有 RDMA 就一定快

RDMA 只是能力。是否走到 GPU Direct RDMA、是否选到正确 NIC、网络是否拥塞、PFC/ECN 是否正确，都会影响性能。

### 误区二：单链路带宽代表集群能力

AI job 关心的是多节点同时通信时的 bisection bandwidth、collective efficiency 和 tail rank。

### 误区三：RoCE 只是普通以太网

RoCE 对无损/拥塞控制要求高。普通以太网能 ping 通，不代表 RoCE collective 稳定。

### 误区四：网络问题只看 NCCL log

NCCL log 重要，但还要看 NIC、switch、PFC、ECN、drop、pause、rank mapping 和 job placement。

### 误区五：训练网络和存储网络混用没关系

Checkpoint 和 dataset 读取也会产生大流量。它们可能和 collective 互相干扰。

### 误区六：网络调优全靠用户环境变量

平台应该提供默认正确的 topology、interface、QoS、rank mapping 和监控，而不是让每个用户自己摸索。

## 设计检查清单

设计 AI 集群网络时，可以检查：

- 使用 InfiniBand、RoCE 还是二者混合。
- 是否支持 GPU Direct RDMA。
- 是否有 GPU Direct RDMA 路径验收，而不只是安装 RDMA 网卡。
- GPU/NIC affinity 是否清晰。
- 是否有多 rail。
- NCCL/RCCL 是否能使用所有预期 NIC。
- NCCL/RCCL 环境变量是否有平台默认、用户可调和排查专用的分层治理。
- 是否维护 network topology manifest：rack、leaf、pod、rail、QoS、failure domain。
- 节点、rack、leaf、pod 拓扑是否进入调度。
- 是否定义网络调度契约：硬约束、软偏好、资源池、QoS class。
- 大 job 是否能获得拓扑连续资源。
- 是否识别拥塞域：NIC、rail、leaf、rack、存储入口、对象存储 gateway。
- RoCE 的 PFC/ECN/QoS 是否端到端一致。
- RoCE 配置漂移是否能被检测。
- 训练、存储、推理、管理流量是否隔离。
- checkpoint 是否会冲击训练网络。
- 是否采集 switch port、NIC、NCCL、job placement 指标。
- 是否有 NCCL topology dump 和 debug 流程。
- 是否有 micro、collective、component、end-to-end benchmark。
- benchmark 是否保存 network manifest、NCCL env、QoS、firmware、raw data。
- 是否在多 job 干扰下测过拥塞。
- timeout、drop、pause、ECN、retry 是否有告警。

## 小结

AI 集群网络的核心不是“网卡速率是多少”，而是：

```text
通信模式
  -> GPU/NIC 拓扑
  -> 网络 fabric
  -> collective algorithm
  -> 拥塞控制
  -> 调度放置
  -> 可观测性
  -> 端到端 step time / p99
```

如果网络设计和调度不匹配，GPU 会等通信；如果拥塞不可观测，问题会长期被误判；如果多租户流量不隔离，一个 job 的 checkpoint 或 AllToAll 可能拖慢整个集群。

高质量 AI 网络设计，应把 RDMA、NCCL topology、拥塞控制、调度和 benchmark 作为一个系统一起看。

## 延伸阅读

- [NVIDIA NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)
- [NVIDIA NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [NVIDIA GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/)
- [NVIDIA RDMA over Converged Ethernet Documentation](https://docs.nvidia.com/networking/display/mlnxofedv23103220lts/rdma+over+converged+ethernet+(roce))
- [NVIDIA Networking Documentation](https://docs.nvidia.com/networking/)
- [OpenFabrics Alliance](https://www.openfabrics.org/)
