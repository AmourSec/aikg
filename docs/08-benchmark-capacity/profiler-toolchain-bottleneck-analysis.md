---
title: Profiler 工具链与瓶颈定位：Nsight、PyTorch Profiler、DCGM、perf 与 eBPF
domain: benchmark-capacity
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-06-12
---

# Profiler 工具链与瓶颈定位：Nsight、PyTorch Profiler、DCGM、perf 与 eBPF

AI 系统性能问题很少能靠一个指标直接解释。

常见现象包括：

- GPU utilization 看起来很高，但 tokens/s 很低。
- GPU utilization 看起来很低，但 CPU、网络或存储并没有明显打满。
- 平均延迟正常，但 p99 经常抖动。
- 单机 benchmark 很好，多机训练一扩展就掉效率。
- 同样代码在一批机器上快，在另一批机器上慢。
- 某次升级后吞吐下降，但日志没有明显错误。

Profiler 的作用不是“截几张漂亮的 trace 图”，而是建立一条证据链：

```text
现象是什么
  -> 哪个指标证明它存在
  -> 哪个阶段贡献最多
  -> 哪个资源成为瓶颈
  -> 哪个代码路径或 kernel 导致瓶颈
  -> 哪个修改能改善
  -> 改完后 benchmark 是否复现提升
```

本篇重点回答：

> 面对 AI 推理、训练和集群性能问题时，应该如何组合 PyTorch Profiler、Nsight Systems、Nsight Compute、DCGM、perf、eBPF、NVTX 标注和 benchmark，形成可复现的瓶颈定位方法？

Profiler 工作最容易失败的原因，不是工具不会用，而是没有把问题、workload、采样窗口和验证标准固定下来。结果是 trace 很大、指标很多，但结论仍然靠猜。

所以本篇的核心原则是：

```text
先定义要证明什么，再决定采什么。
```

一个好的 profiler 结论应该满足：

- 能解释一个明确的性能现象。
- 能定位到具体阶段、资源或代码路径。
- 能排除几个更简单的解释。
- 能通过最小变更和 benchmark 验证。
- 能沉淀为报告、回归检测或 runbook。

## Profiling Contract

正式采集 profiler 前，建议先写一个 Profiling Contract。它比 Benchmark Contract 更偏诊断，目标是让采集有边界、有假设、有可复查的产物。

示例：

```yaml
profile_id: infer-p99-regression-2026-06-12

problem:
  symptom: p99_e2e_latency_regression
  baseline: 3.2s
  current: 4.1s
  target: <= 3.3s
  affected_workload: high_concurrency_long_context

workload:
  model: llama-like-70b
  precision: fp8
  input_tokens_p50_p95: [2048, 8192]
  output_tokens_p50_p95: [256, 1024]
  qps: 120
  concurrency: 256
  cache_state: prefix_cache_warm

environment:
  gpu_type: h100-sxm
  gpu_count: 8
  driver: recorded
  cuda: recorded
  framework_or_engine: recorded
  image_digest: recorded
  node_ids: recorded

capture:
  benchmark_window: steady_state
  warmup_duration: 120s
  capture_duration: 30s
  target_processes: selected_worker
  target_ranks: [slowest_rank, median_rank]
  tools:
    - application_metrics
    - nsight_systems
    - dcgm
    - perf_if_cpu_gap
  nvtx_required: true

success:
  evidence_required:
    - before_after_benchmark
    - trace_showing_bottleneck_change
    - no_regression_on_throughput
```

这个 contract 让 profiling 从“打开工具看一看”变成一组可验证问题：

- 要解释哪个指标，而不是泛泛地找慢。
- 采集哪个 workload，而不是随便跑一个样例。
- 采集哪个时间窗口，而不是把 warmup、steady state 和异常混在一起。
- 采哪些 rank、进程或节点，而不是全量采集到不可分析。
- 什么证据才足够支持结论。

如果问题很小，contract 可以很短；但至少要明确：

```text
symptom
workload
environment
capture window
success criterion
artifacts
```

## 一张总图

```mermaid
flowchart TB
    Symptom["Symptom<br/>latency / tokens/s / step time / jitter"]
    AppMetric["Application Metrics<br/>TTFT / TPOT / p99 / MFU / goodput"]
    Phase["Phase Split<br/>prefill / decode / forward / backward / comm / data"]
    Framework["Framework Profiler<br/>operator / module / memory / call stack"]
    Timeline["GPU Timeline<br/>CPU thread / CUDA API / kernel / NCCL / NVTX"]
    Kernel["Kernel Profile<br/>occupancy / Tensor Core / HBM / stall / roofline"]
    System["System Signals<br/>DCGM / CPU / storage / network / OS"]
    Hypothesis["Hypothesis<br/>why this bottleneck happens"]
    Fix["Fix and Validate<br/>change / A-B / regression guard"]
    Report["Report<br/>config / trace / evidence / before-after"]

    Symptom --> AppMetric
    AppMetric --> Phase
    Phase --> Framework
    Framework --> Timeline
    Timeline --> Kernel
    Timeline --> System
    Kernel --> Hypothesis
    System --> Hypothesis
    Hypothesis --> Fix
    Fix --> Report
    Report --> AppMetric
```

这张图强调两个原则：

- 先用指标确认问题，再用 profiler 解释原因。
- profiler 结论必须回到 benchmark 验证，不能只停留在 trace 观察。

## 工具各自回答什么问题

不同工具处在不同层级。选错工具，会让定位过程变慢。

| 工具 | 主要回答的问题 | 典型用途 | 不适合做什么 |
| --- | --- | --- | --- |
| PyTorch Profiler | 框架层时间花在哪些 operator、module、CPU/GPU activity 上 | 训练 step、推理 forward、显存、shape、operator 热点 | 深入解释单个 CUDA kernel 为什么慢 |
| Nsight Systems | 整机时间线如何流动，CPU 线程、CUDA API、GPU kernel、NCCL 是否并行 | 找 idle gap、CPU launch overhead、同步点、通信等待、跨进程关系 | 分析单个 kernel 的寄存器、访存和指令瓶颈 |
| Nsight Compute | 单个 CUDA kernel 的微观性能瓶颈是什么 | occupancy、Tensor Core、memory throughput、warp stall、kernel 对比 | 看端到端请求生命周期 |
| DCGM / DCGM Exporter | GPU 在生产或集群中是否健康、忙碌、降频、报错 | utilization、HBM、power、temperature、ECC、Xid、长期监控 | 替代 profiler 解释代码级瓶颈 |
| perf | CPU 侧热点、系统调用、内核路径和用户态采样 | tokenization、DataLoader、scheduler、runtime、网络/存储 CPU 开销 | 分析 GPU kernel 内部 |
| eBPF | 低侵入地观察生产系统的内核事件、网络、块设备、系统调用 | 线上排查 syscall、TCP、磁盘 I/O、调度延迟 | 替代训练/推理框架 profiler |
| NVTX | 给 trace 增加业务语义标签 | 标注 prefill、decode、step、rank、request、pipeline stage | 单独产生性能结论 |

一个实用判断：

- 不知道问题有没有发生：先看 monitoring。
- 知道问题发生但不确定是否可复现：做 benchmark。
- benchmark 确认变慢但不知道原因：做 profiling。
- profiling 找到疑似原因：做最小变更，再 benchmark 验证。

## 证据链分层

Profiler 工具链不是并列堆工具，而是分层收集证据。

| 层级 | 证据 | 能说明什么 | 不能说明什么 |
| --- | --- | --- | --- |
| 现象层 | QPS、latency、tokens/s、step time、MFU、error rate | 问题是否存在，影响多大 | 根因是什么 |
| 阶段层 | prefill/decode、forward/backward、communication、data、checkpoint | 问题在哪个阶段 | 阶段内部为什么慢 |
| 时间线层 | CPU thread、CUDA API、GPU kernel、NCCL、NVTX | 是否有 idle、同步、通信暴露、launch gap | 单个 kernel 的微观瓶颈 |
| kernel 层 | occupancy、Tensor Core、HBM/L2 throughput、stall reason | kernel 为什么没有达到预期 | 端到端收益是否足够 |
| 系统层 | DCGM、CPU、network、storage、OS、container | 是否有环境、硬件或系统干扰 | 具体模型代码是否低效 |
| 验证层 | before/after benchmark、A/B、回归检测 | 修改是否真正改善目标指标 | 不能自动解释所有 workload |

一个可靠结论通常长这样：

```text
现象层：
  p99 latency 回退 28%，只发生在长上下文高并发 workload。

阶段层：
  回退主要来自 TTFT，不是 decode TPOT。

时间线层：
  Nsight Systems 显示 prefill 前有 CPU launch gap 和 tokenizer 阻塞。

系统层：
  perf 显示 CPU 时间集中在 tokenizer path，DCGM 未见降频或 Xid。

验证层：
  tokenizer cache 后 p99 恢复，throughput 未回退。
```

反过来，下面这种结论不够强：

```text
Nsight Compute 看到某个 kernel occupancy 低，所以系统瓶颈是这个 kernel。
```

它缺少端到端占比、关键路径位置和 before/after 验证。

## 工具选择矩阵

实际定位时，可以按症状选择第一批工具。

| 症状 | 首先看 | 进一步工具 |
| --- | --- | --- |
| 推理 TTFT 变差 | 请求阶段指标、queue、prefill time | Nsight Systems、PyTorch Profiler、CPU perf |
| 推理 TPOT 变差 | decode batch、KV Cache、kernel timeline | Nsight Systems、Nsight Compute、DCGM |
| 训练 step time 变慢 | step breakdown、data/compute/comm | PyTorch Profiler、Nsight Systems、NCCL/network counters |
| MFU 低 | FLOPs 口径、kernel 占比、shape | Nsight Compute、Roofline、编译器日志 |
| p99 抖动 | 分位数、请求长度、节点维度 | distributed trace、DCGM、eBPF |
| 多机扩展效率差 | scaling curve、communication exposed time | multi-rank Nsight Systems、NCCL logs、网络 counters |
| 个别节点慢 | node-level benchmark、DCGM、拓扑 | nvidia-smi topo、fabric counters、环境 manifest |
| checkpoint spike | step timeline、storage metrics | Nsight Systems、eBPF block I/O、存储监控 |

这个矩阵的意思不是限制工具，而是避免一开始就采集过多数据。先用最能缩小范围的工具，再进入更细层级。

## 先从症状和阶段拆分开始

Profiler 之前要先回答两个问题：

1. 哪个指标坏了？
2. 这个指标属于哪个阶段？

推理系统可以先拆成：

```text
request arrival
  -> queueing
  -> scheduling
  -> prefill
  -> decode loop
  -> postprocess
  -> response streaming
```

训练系统可以先拆成：

```text
data loading
  -> forward
  -> loss
  -> backward
  -> gradient communication
  -> optimizer
  -> checkpoint / eval
```

集群层可以先拆成：

```text
scheduling
  -> image / env startup
  -> data access
  -> GPU execution
  -> network communication
  -> storage checkpoint
  -> failure / retry
```

如果没有阶段拆分，trace 很容易变成一张很复杂但无法决策的图。

例如“p99 延迟变差”至少要区分：

- 排队变长。
- prefill 变慢。
- decode 每 token 变慢。
- KV Cache miss 变多。
- speculative decoding 接受率下降。
- 远端存储或 tokenizer 抖动。
- 某些节点降频或出现 Xid 错误。

这些原因对应的工具完全不同。

## PyTorch Profiler：先看框架层热点

PyTorch Profiler 适合回答：

- 一个训练 step 中哪些 operator 最耗时。
- CPU 和 CUDA 时间分别花在哪里。
- 是否存在意外的 CPU hotspot。
- 哪些 operator 显存占用大。
- 某些 shape 是否导致 kernel 选择异常。
- forward、backward、optimizer 的时间比例是否合理。

它的优势是和框架语义贴近。对于新手来说，PyTorch Profiler 通常比直接看 Nsight 更容易入门，因为它能直接显示 operator 名称、调用栈、shape、memory 和时间统计。

使用时要注意几个点。

### 不要从第一个 step 直接判断

第一个 step 常常包含：

- CUDA context 初始化。
- kernel lazy loading。
- JIT / graph capture / compile。
- dataloader warmup。
- cache 填充。

因此需要设置 warmup、active window 和重复采样窗口。只看第一个 step，很容易把初始化成本误判为稳定瓶颈。

### 先看粗粒度，再看细粒度

常见顺序是：

1. 看 CPU total、CUDA total 和 self time。
2. 看最耗时 operator。
3. 看显存峰值和分配次数。
4. 看 shape 是否符合预期。
5. 看调用栈定位到模型代码、数据代码或 runtime 代码。

如果发现时间集中在 `aten::matmul`、attention、layernorm、embedding、copy、all_reduce 等操作上，再进一步用 Nsight Systems 或 Nsight Compute。

### 警惕 profiler 自身开销

Profiler 会带来额外开销，尤其是记录 shape、memory、stack、trace 时。采样窗口越长，trace 越大，对系统扰动越明显。

实用做法：

- 用短窗口抓稳定阶段。
- 不在所有进程、所有 rank、所有 step 上同时开完整 trace。
- 先跑无 profiler 的 benchmark，再跑 profiler 解释原因。
- profiler 数据只用于定位，不直接当作最终性能指标。

### PyTorch Profiler 产物怎么读

PyTorch Profiler 的输出通常包括：

- operator table。
- CPU time / CUDA time。
- self time / total time。
- memory allocation。
- input shape。
- call stack。
- Chrome trace / TensorBoard trace。

读表时要注意：

| 字段 | 常见含义 | 误读风险 |
| --- | --- | --- |
| self CPU time | operator 自身 CPU 时间 | 不包含子调用时容易低估总成本 |
| CPU total | CPU 侧累计时间 | 可能包含等待 GPU 或同步 |
| CUDA total | 关联 CUDA kernel 时间 | 不一定等于端到端关键路径 |
| memory | 分配/释放行为 | profiler 开启 memory 记录会增加开销 |
| input shape | operator 的实际 shape | dynamic shape 下要看分布，不只看单个样本 |

建议先回答三个问题：

```text
CPU 时间是否异常高？
CUDA 时间集中在哪些 operator？
operator shape 是否符合 workload contract？
```

如果 PyTorch Profiler 已经显示 CPU path 是主因，就不要急着进入 Nsight Compute；如果显示某个 fused kernel 或 Triton kernel 占比很高，再进入 GPU timeline 或 kernel 级分析。

## Nsight Systems：看端到端时间线

Nsight Systems 适合回答：

- CPU 是否在及时 launch CUDA kernel。
- GPU 是否存在 idle gap。
- CUDA API 是否被同步点阻塞。
- NCCL 通信是否和计算重叠。
- 多进程、多线程、多 GPU 的时间线是否协调。
- NVTX 标注的业务阶段是否和 kernel 执行对应。

它最擅长发现“时间线形态问题”。

常见模式包括：

### GPU 中间有大段空白

可能原因：

- CPU preprocessing 太慢。
- dataloader 供应不上。
- Python 逻辑或 GIL 开销大。
- scheduler 决策慢。
- 同步等待。
- 数据从 CPU 到 GPU 拷贝没有重叠。

处理方向：

- 用 perf 或 PyTorch Profiler 看 CPU 侧热点。
- 增加 dataloader worker、prefetch、pinned memory。
- 减少 Python 调度开销。
- 合并小 kernel 或使用 CUDA Graph。
- 检查是否有不必要的 `cudaSynchronize` 或 `.item()`。

### CUDA API 调用很多且 kernel 很碎

可能原因：

- batch 太小。
- operator 没有融合。
- dynamic shape 过多。
- 推理 decode 阶段单 token 粒度导致 launch overhead 显著。
- 框架 fallback 到低效路径。

处理方向：

- 使用 fused kernel。
- 使用 CUDA Graph。
- 调整 batch 或 micro-batch。
- 稳定 shape。
- 检查 torch.compile、Inductor、Triton 或专用推理引擎是否生效。

### 通信没有被计算覆盖

训练中常见现象是 all-reduce、reduce-scatter、all-gather 或 all-to-all 暴露在计算路径上。

可能原因：

- bucket 太大或太小。
- 并行策略不匹配网络拓扑。
- rank mapping 不合理。
- 某个 rank 变成 straggler。
- MoE expert imbalance。
- 网络拥塞或链路异常。

处理方向：

- 调整 bucket、overlap 策略和通信粒度。
- 优化 DP/TP/PP/EP/FSDP/ZeRO 配置。
- 检查 NCCL topology、multi-rail、IB/RoCE 状态。
- 对比最快 rank 和最慢 rank 的 trace。

### Nsight Systems 采集窗口

Nsight Systems trace 很容易变大，采集前要控制窗口。

推荐方式：

```text
benchmark warmup
  -> 进入 steady state
  -> 用 NVTX 或时间范围触发 capture
  -> 只采关键 10-60 秒
  -> 结束后继续跑 benchmark 观察是否恢复
```

对于训练任务，可以按 step id 采集：

```text
capture steps 100-120
```

对于推理服务，可以按时间窗口或 request id 采集：

```text
capture p99 request window
capture burst window
capture cache-miss window
```

采集策略建议：

- 先只采一个代表进程或少数 rank。
- 如果怀疑 straggler，同时采最快 rank 和最慢 rank。
- 如果怀疑通信，保留 rank mapping、NCCL 环境变量和网络 counters。
- 如果怀疑 CPU launch gap，保留 CPU thread 和 CUDA API 事件。
- 如果 trace 过大，减少窗口而不是降低 workload 真实性。

Nsight Systems 的价值在于看到时间关系；采集窗口太大反而会让时间关系变难分析。

### 频繁出现同步点

常见来源：

- `.item()`、打印 tensor 值、同步日志。
- CPU 读取 GPU 结果。
- 不必要的 barrier。
- allocator 或 memory copy 引发同步。
- 评测、保存、采样逻辑插入训练主路径。

处理方向是减少同步、把必要同步移出关键路径，或把同步成本显式纳入指标。

## Nsight Compute：深入单个 kernel

当 Nsight Systems 或 PyTorch Profiler 已经定位到某个 kernel 或 operator 可疑时，再用 Nsight Compute。

Nsight Compute 适合回答：

- kernel 是否真正使用 Tensor Core。
- occupancy 是否过低。
- 是否受 HBM bandwidth 限制。
- 是否受 shared memory、register、L2、instruction issue 限制。
- warp stall 主要来自 memory dependency、barrier、not selected 还是其他原因。
- 同一个 kernel 在不同 shape、不同参数下为什么性能不同。

它不适合作为第一步，因为单个 kernel 优化可能对端到端性能没有意义。

一个常见误区是：发现某个 kernel 只有 50% occupancy，就认为它一定是瓶颈。实际要看：

- 这个 kernel 占端到端时间多少。
- 它是否 memory-bound。
- 它是否已经接近可达到的 roofline。
- 它是否在关键路径上。
- 优化它是否会被别的瓶颈抵消。

Nsight Compute 分析前建议写清 kernel selection：

```text
target_kernel:
  name: fused_attention_fwd
  reason: accounts_for_35_percent_of_prefill_time
  shape: batch=..., seq=..., heads=..., head_dim=...
  dtype: bf16
  baseline_runtime: ...
  expected_runtime_or_roofline: ...
```

不要对随机选中的 kernel 做深度优化。kernel 级优化成本高，必须先证明它：

1. 占端到端时间足够多。
2. 位于关键路径上。
3. 有明确可优化空间。
4. 修改不会破坏其他 workload。

### Nsight Compute 指标怎么归类

不同指标对应不同优化方向：

| 现象 | 可能含义 | 常见方向 |
| --- | --- | --- |
| Tensor Core 使用低 | dtype/layout/shape 不匹配，fallback 到普通 CUDA core | 检查 dtype、layout、矩阵维度、kernel selection |
| HBM throughput 高且 SM 等待访存 | memory-bound | fusion、tiling、减少读写、cache reuse |
| occupancy 低 | register/shared memory/block 配置限制 | 调整 tile、num warps、寄存器压力 |
| warp stall barrier 高 | 同步或 shared memory 使用不当 | 减少 barrier、优化流水 |
| instruction issue 受限 | 指令混合或依赖链问题 | kernel 重写、调整 unroll/pipeline |
| L2 hit rate 异常 | 访问模式或数据复用差 | layout、blocking、prefetch |

这些指标要和 Roofline、端到端占比一起看。单个指标异常不等于端到端瓶颈。

对 AI 计算来说，Nsight Compute 常用于分析：

- attention kernel。
- GEMM / matmul。
- layernorm / rmsnorm。
- softmax。
- embedding。
- MoE dispatch / combine。
- quantization / dequantization kernel。
- sampling kernel。
- 自定义 Triton kernel。

## DCGM：生产和集群层的 GPU 事实

DCGM 更像是 GPU 侧的健康和资源观测基础设施。它适合长期、低频、集群级监控。

常见指标包括：

- GPU utilization。
- SM active。
- memory utilization。
- HBM bandwidth。
- power draw。
- temperature。
- clocks。
- ECC errors。
- Xid errors。
- PCIe / NVLink 相关信号。

DCGM 能帮助回答：

- 某个节点是否降频。
- 某张卡是否异常报错。
- GPU 是否长期空闲。
- 显存是否接近上限。
- 功耗和温度是否导致性能波动。
- workload 是否真的跑在预期 GPU 上。

但 DCGM 不能告诉你“哪个 operator 慢”或“哪个 kernel 访存模式不好”。它更适合做：

- profiler 前的异常筛查。
- benchmark 期间的环境记录。
- 生产环境的告警。
- 容量模型的长期校准。

### DCGM 指标的解释边界

DCGM 很适合做健康筛查，但要避免过度解释。

例如：

| DCGM 现象 | 可能说明 | 下一步 |
| --- | --- | --- |
| SM active 低 | GPU 空闲、CPU/data/queue 等待 | 看 Nsight Systems idle gap、CPU perf、data metrics |
| HBM bandwidth 高 | memory-bound 或大量 copy | 看 kernel timeline 和 Nsight Compute |
| power draw 低 | workload 不饱和或限频策略 | 查 clocks、utilization、应用阶段 |
| clocks 下降 | thermal/power throttle 或策略限制 | 查温度、功耗、throttle reason |
| ECC/Xid 增加 | 硬件或驱动异常 | 隔离节点并做健康检查 |

DCGM 的结论通常是“环境或资源信号”，不是“代码根因”。它要和 trace、benchmark、环境记录一起使用。

## perf 与 eBPF：CPU、系统调用、网络和存储

AI 系统不是只有 GPU。

很多瓶颈发生在 CPU 和操作系统层：

- tokenization。
- prompt parsing。
- JSON 序列化。
- Python 调度。
- DataLoader。
- 图像/音频解码。
- 文件系统读取。
- 网络收发。
- 进程调度。
- 内存拷贝。
- page fault。
- syscall 过多。

`perf` 适合做 CPU sampling，回答“CPU 时间花在哪些函数上”。如果 GPU trace 里有大量 idle gap，而 DCGM 显示 GPU 没打满，perf 常常是下一步。

eBPF 适合做线上低侵入观测，尤其是：

- syscall latency。
- TCP retransmit。
- block I/O latency。
- scheduler latency。
- socket queue。
- 文件系统热点。
- 容器和进程维度的系统行为。

它们和 GPU profiler 是互补关系。

一个典型例子：

```text
Nsight Systems 看到 GPU 每隔一段时间空闲
  -> PyTorch Profiler 看到 data loading 时间变长
  -> perf 看到 CPU 时间集中在 image decode
  -> eBPF 看到对象存储读取存在尾延迟
  -> benchmark 复现：本地 NVMe cache 后 step time 抖动下降
```

### perf 与 eBPF 的使用边界

`perf` 更适合离线或受控环境中的 CPU 采样；eBPF 更适合线上低侵入观测。

| 工具 | 更适合 | 注意事项 |
| --- | --- | --- |
| perf top / record | CPU 热点、用户态/内核态函数、采样火焰图 | 需要符号信息，采样本身有开销 |
| eBPF syscall tracing | syscall 延迟、open/read/write/connect 等路径 | 要限制过滤条件，避免线上数据量过大 |
| eBPF TCP tracing | retransmit、连接延迟、socket queue | 需要结合服务连接拓扑解释 |
| eBPF block I/O tracing | 磁盘延迟、队列深度、tail I/O | 需要和文件系统/对象存储层区分 |
| scheduler tracing | run queue、context switch、CPU starvation | 需要结合 cgroup、NUMA、CPU pinning |

如果 GPU trace 显示 idle gap，但应用指标没有数据阶段拆分，perf/eBPF 常常是把问题从“GPU 不忙”推进到“CPU/系统哪里阻塞”的关键工具。

## NVTX：让 trace 看得懂

没有业务标注的 trace，通常很难解释。

NVTX 的作用是把业务阶段写进 timeline，例如：

- `request_id=...`
- `prefill`
- `decode_step`
- `sampling`
- `forward`
- `backward`
- `optimizer`
- `all_reduce`
- `checkpoint_save`
- `rank=7`
- `pipeline_stage=2`
- `expert_dispatch`

好的 NVTX 标注应该满足：

- 名称稳定，便于跨版本对比。
- 粒度适中，不要每行代码都标。
- 包含关键维度，如 rank、stage、batch、sequence length。
- 和 benchmark 指标中的阶段定义一致。

在推理服务中，建议至少标注：

```text
queueing
  -> prefill
  -> decode
  -> sampling
  -> detokenize
  -> stream response
```

在训练任务中，建议至少标注：

```text
data
  -> forward
  -> loss
  -> backward
  -> optimizer
  -> checkpoint
  -> eval
```

如果没有 NVTX，Nsight Systems 里看到的只是线程、CUDA API 和 kernel 名称；有了 NVTX，才能把底层事件映射回业务阶段。

## 时间对齐与 Trace 关联

AI 性能定位往往同时采集多类数据：

- 应用指标。
- profiler trace。
- DCGM。
- CPU perf。
- eBPF。
- 网络和存储监控。
- 日志。

如果这些数据不能关联，证据链会断。

建议统一记录几个关键 id：

| ID | 用途 |
| --- | --- |
| run_id | 关联一次 benchmark 或生产采样 |
| request_id | 关联推理请求、p99 请求和 trace |
| step_id | 关联训练 step、checkpoint 和 profiler window |
| rank_id | 关联分布式训练进程 |
| node_id | 关联硬件、DCGM、网络和存储指标 |
| model_version / commit | 关联代码和权重版本 |

采集时建议在 NVTX、日志和 metrics 里使用同一组 id：

```text
run_id=bench-2026-06-12-001
step_id=1200
rank=7
stage=backward
```

或者：

```text
request_id=req-abc
phase=prefill
input_tokens=8192
output_tokens=1024
node_id=node-17
```

### 时间戳纪律

多节点系统还要注意时间戳：

- 节点时间是否同步。
- profiler 时间是否和日志时间可映射。
- metrics scrape interval 是否足够细。
- p99 请求窗口是否覆盖到 trace 采集窗口。
- benchmark warmup、measurement、cooldown 是否明确。

如果 DCGM 是 15 秒粒度，而问题发生在 200 ms 的 decode spike 上，DCGM 只能用于环境背景，不能直接解释 spike。

### 产物目录

建议每次 profiling 产物使用稳定目录结构：

```text
profiles/
  2026-06-12-infer-p99-regression/
    contract.yaml
    benchmark-before.json
    benchmark-after.json
    traces/
      nsys-rank7.qdrep
      pytorch-profiler-worker3.json
    metrics/
      dcgm.csv
      network.csv
      storage.csv
    logs/
      service.log
      nccl.log
    report.md
```

这不是为了形式，而是为了让其他人和 AI 能在几个月后重新理解这次定位。

## 一套推荐定位流程

下面是一套比较稳的流程。

### 1. 明确症状和成功标准

先写清楚：

- 哪个 workload。
- 哪个版本。
- 哪个指标变差。
- 变差多少。
- 是否稳定复现。
- 优化成功的判断标准是什么。

例如：

```text
workload: Llama-like 70B, input 2048, output 512
symptom: p99 E2E latency from 3.2s to 4.1s
scope: only high-concurrency requests
success: p99 returns below 3.3s with no throughput regression
```

### 2. 复现并控制变量

复现时要记录：

- 模型、权重、tokenizer。
- 输入/输出长度分布。
- batch、并发、随机种子。
- GPU、CPU、内存、网络、存储。
- driver、CUDA、NCCL、PyTorch、engine 版本。
- 容器镜像 digest。
- 环境变量。

如果环境不固定，profiler 很容易追着噪声跑。

### 3. 先拆阶段

不要一开始就看 kernel。先确认问题在哪个阶段。

推理：

```text
queueing latency
prefill latency
decode TPOT
sampling/postprocess
network streaming
```

训练：

```text
data time
forward time
backward time
communication time
optimizer time
checkpoint/eval time
```

如果阶段指标已经能解释 80% 的问题，再深入该阶段。

### 4. 用框架 profiler 找热点

用 PyTorch Profiler 或推理引擎内部 profiler 看：

- operator 时间。
- CPU/GPU 时间比例。
- memory allocation。
- shape。
- module path。
- 关键阶段的算子组成。

这一步的目标是缩小范围，不是最终定论。

### 5. 用 Nsight Systems 看时间线

重点看：

- GPU 是否空闲。
- CPU launch 是否连续。
- CUDA API 是否阻塞。
- kernel 是否碎片化。
- 通信是否暴露。
- rank 之间是否同步等待。
- NVTX 阶段是否符合预期。

这一步通常能判断瓶颈属于：

- CPU 供应不足。
- GPU kernel 低效。
- 通信暴露。
- 同步过多。
- 数据/存储阻塞。
- 调度和排队问题。

### 6. 必要时用 Nsight Compute 看 kernel

只有当一个 kernel 端到端占比足够高，或者它是一个关键路径上的自定义 kernel，才值得深入 Nsight Compute。

要记录：

- 输入 shape。
- block/grid 配置。
- dtype。
- kernel 名称和版本。
- occupancy。
- achieved occupancy。
- Tensor Core 使用情况。
- HBM/L2 throughput。
- stall reason。
- 与理论或历史版本对比。

### 7. 用系统信号排除环境问题

同时检查：

- DCGM：降频、温度、功耗、ECC、Xid。
- 网络：IB/RoCE counters、重传、拥塞、链路速率。
- 存储：吞吐、IOPS、尾延迟。
- CPU：load、context switch、syscall、NUMA。
- 容器：cgroup 限制、CPU pinning、共享节点干扰。

AI 性能问题里，环境问题很常见。尤其是“同样代码有些节点慢”的情况，必须先排环境。

### 8. 做最小变更并回到 benchmark

定位结论要通过前后对比验证。

报告中至少写：

- 改了什么。
- 为什么这个修改对应前面的证据。
- 修改前指标。
- 修改后指标。
- 是否影响其他指标。
- 是否只对特定 workload 有效。
- 是否加入回归检测。

### 9. 给结论分级

不是所有 profiler 发现都能直接成为根因结论。建议给发现分级。

| 等级 | 含义 | 可采取动作 |
| --- | --- | --- |
| Observation | trace 或指标里看到的现象 | 继续收集证据，不直接下结论 |
| Hypothesis | 可能解释现象的原因 | 设计验证实验 |
| Supported Cause | 多个证据支持，且排除明显替代解释 | 做最小修复并 A/B |
| Confirmed Root Cause | 修复后 benchmark 恢复，且保护指标无回退 | 写报告、加回归检测 |

例子：

```text
Observation:
  Nsight Systems 看到 prefill 前 CPU gap。

Hypothesis:
  tokenizer path 阻塞 GPU launch。

Supported Cause:
  PyTorch Profiler / perf 显示 tokenizer CPU time 上升，DCGM 无降频，输入长度分布不变。

Confirmed Root Cause:
  tokenizer cache 后 p99 TTFT 恢复，吞吐无回退。
```

这个分级能避免把“我看到了一个异常”过早写成“根因就是它”。

### 10. 把经验转成防线

一次 profiling 的终点不是报告，而是防止同类问题重复发生。

常见沉淀方式：

- benchmark case：把复现 workload 加入 nightly 或 release benchmark。
- alert：把关键指标加到监控告警，如 p99、DCGM Xid、checkpoint spike。
- runbook：把定位步骤写成排障手册。
- dashboard：把应用指标、GPU、网络、存储放到同一视图。
- regression guard：在 CI 或发布门禁里加入微基准或组件基准。
- node quarantine：把异常硬件自动隔离。

如果没有沉淀，下一次团队还会从零开始看 trace。

## 常见瓶颈模式

### GPU 空闲，但 CPU 很忙

可能原因：

- tokenization 太慢。
- DataLoader 解码太慢。
- Python 控制流过重。
- 请求 scheduler 单线程瓶颈。
- 小 batch 导致 launch overhead 占比高。
- CPU 到 GPU 拷贝没有重叠。

工具组合：

```text
DCGM -> Nsight Systems -> PyTorch Profiler -> perf / eBPF
```

优化方向：

- 预处理或缓存 tokenizer 结果。
- 增加并行度和 prefetch。
- 使用 pinned memory。
- 减少 Python 路径。
- 合并小算子。
- 使用 CUDA Graph。

### GPU 很忙，但吞吐不高

可能原因：

- kernel 低效。
- memory-bound。
- 没有使用 Tensor Core。
- shape 不适合当前 kernel。
- fallback 到通用实现。
- quant/dequant 开销抵消收益。
- batch 或 sequence length 导致资源利用差。

工具组合：

```text
PyTorch Profiler -> Nsight Systems -> Nsight Compute
```

优化方向：

- fused kernel。
- 更合适的 attention 实现。
- 调整 shape 和 padding。
- 使用 torch.compile / Inductor / Triton。
- 使用专用推理引擎。
- 检查 dtype、layout 和 kernel selection。

### 平均值正常，p99 抖动

可能原因：

- queueing burst。
- KV Cache eviction。
- prefix cache miss。
- 某些请求输入极长。
- GC 或 allocator 抖动。
- checkpoint/eval 干扰。
- 网络或存储尾延迟。
- 个别节点降频或硬件异常。

工具组合：

```text
application metrics -> distributed trace -> DCGM -> eBPF -> targeted profiler
```

优化方向：

- 按输入/输出长度分桶分析。
- 分离长短请求。
- 增加 cache hit 指标。
- 做节点维度对比。
- 对 p99 请求单独采样 trace。
- 做隔离和限流。

### 多机训练扩展效率差

可能原因：

- 通信暴露。
- rank mapping 不合理。
- NCCL topology 不匹配。
- 网络拥塞。
- straggler。
- pipeline bubble。
- MoE expert imbalance。
- checkpoint 阻塞。

工具组合：

```text
step metrics -> Nsight Systems multi-rank trace -> NCCL logs/counters -> DCGM/network metrics
```

优化方向：

- 调整 DP/TP/PP/EP/FSDP/ZeRO 组合。
- 改善 compute-communication overlap。
- 调整 bucket。
- 做拓扑感知 placement。
- 检查 slow rank。
- 优化 checkpoint 和 eval 调度。

### 同样代码在不同节点速度不同

可能原因：

- GPU clocks 不同。
- 温度或功耗限制。
- PCIe/NVLink/IB 链路异常。
- NUMA 绑定错误。
- CPU 频率策略不同。
- 驱动、固件、容器镜像不一致。
- 邻居任务干扰。

工具组合：

```text
DCGM -> nvidia-smi topo / fabric counters -> environment manifest -> benchmark A-B
```

优化方向：

- 固化节点基线。
- 上线前 burn-in。
- 定期健康检查。
- 自动隔离异常节点。
- benchmark 结果按 node id 记录。

### 显存接近上限且性能抖动

可能原因：

- allocator 频繁分配/释放。
- KV Cache 或 activation 内存碎片。
- batch size 动态变化导致内存规划不稳定。
- checkpoint、eval 或 logging 临时占用显存。
- FSDP/ZeRO all-gather 峰值超出预期。
- 推理服务中 eviction 或 swap 触发。

工具组合：

```text
application memory metrics -> PyTorch Profiler memory -> Nsight Systems -> DCGM memory
```

优化方向：

- 固定或分桶 shape。
- 预分配内存池。
- 使用 paged KV cache 或更稳定的 cache policy。
- 减少临时 tensor。
- 调整 activation checkpointing。
- 降低 micro-batch 或并发上限。

显存问题不一定表现为 OOM。很多时候它表现为周期性抖动、allocator stall、cache miss 或 batch 被迫缩小。

### 网络或存储尾延迟影响训练/推理

可能原因：

- 对象存储或并行文件系统 p99 抖动。
- checkpoint 同时写入造成拥塞。
- RDMA/RoCE 重传或拥塞控制异常。
- 多租户共享网络导致干扰。
- 数据 cache miss。
- 日志、评测和 checkpoint 与主路径争资源。

工具组合：

```text
application phase metrics -> eBPF/network/storage metrics -> DCGM -> Nsight Systems
```

优化方向：

- 数据和 checkpoint 错峰。
- 本地 NVMe cache。
- 分层 checkpoint。
- 限制后台 I/O 并发。
- 拓扑感知 placement。
- 对长尾 I/O 做 retry/backoff 和隔离。

这类问题常常不是 GPU profiler 单独能解释的。GPU trace 只能看到“等待”，原因要到系统层找。

### 动态 shape 或编译缓存导致性能不稳定

可能原因：

- 输入长度分布变化。
- request batching 产生过多 shape。
- torch.compile / Inductor 反复 recompile。
- CUDA Graph capture 失败。
- Triton autotune 或 kernel cache 重新触发。
- 推理服务 warmup 没覆盖真实 shape。

工具组合：

```text
application shape stats -> framework logs -> PyTorch Profiler -> Nsight Systems
```

优化方向：

- shape bucketing。
- warmup 覆盖真实分布。
- 限制 dynamic shape 范围。
- 固化 batch policy。
- 记录 compile/cache hit 指标。
- 对 recompile 做告警。

动态 shape 问题常见于 LLM 推理、RAG/Agent、多模态和变长训练。它会让“同一模型”在不同 workload 下表现完全不同。

## 分布式 profiling 的特殊问题

分布式训练和多副本推理不能简单地“每个进程都开完整 profiler”。

问题包括：

- trace 文件巨大。
- profiler 开销影响原始行为。
- 所有 rank 同时采样会放大干扰。
- 不同机器时钟可能不完全一致。
- 最慢 rank 才是真正关键，但不一定容易提前知道。

更实用的做法：

- 先用全局指标找异常时间窗口。
- 只对少数代表 rank 开 trace。
- 同时采样最快 rank 和最慢 rank。
- 用统一 step id、request id、rank id 做 NVTX 标注。
- 对通信问题保留 NCCL 拓扑、环境变量和网络 counters。
- 对 MoE 问题记录 expert load、token routing、all-to-all 时间。
- 对 pipeline 问题记录 stage id、microbatch id、bubble。

分布式 profiling 的目标不是收集所有数据，而是收集能解释差异的数据。

## Profiler 报告模板

一份有用的 profiler 报告可以按下面结构写。

```text
Title:
  short problem statement

Workload:
  model / batch / sequence length / concurrency / training config

Environment:
  hardware / driver / CUDA / framework / engine / image / topology

Symptom:
  which metric regressed or missed target

Reproduction:
  command / dataset / seed / duration / warmup / repetitions

Evidence:
  application metrics
  profiler trace links
  system metrics
  before-after tables

Bottleneck:
  phase
  root cause hypothesis
  why other explanations were ruled out

Fix:
  code/config/runtime change

Result:
  metric before
  metric after
  confidence and caveats

Regression Guard:
  benchmark or alert added
```

这个模板的价值是逼迫结论闭环：有现象、有证据、有修改、有验证。

建议再补一个证据表，避免报告只写叙述。

| Evidence | Artifact | Supports | Limitations |
| --- | --- | --- | --- |
| benchmark before/after | `benchmark-before.json`, `benchmark-after.json` | 证明指标变化 | 不能单独解释原因 |
| Nsight Systems trace | `nsys-rank7.qdrep` | 证明 idle gap 或通信暴露 | 采样窗口有限 |
| PyTorch Profiler | `trace.json` | 证明 operator / CPU hotspot | profiler 有开销 |
| DCGM metrics | `dcgm.csv` | 排除降频、Xid、温度 | 粒度可能较粗 |
| perf/eBPF | `perf.data`, `ebpf.csv` | 解释 CPU/syscall/I/O | 需要符号和过滤 |

报告里还应该写“排除项”：

```text
ruled_out:
  - input length distribution unchanged
  - no GPU clock throttling
  - no Xid/ECC increase
  - same container image and model commit
  - regression only appears when prefix cache disabled
```

排除项能显著提高结论质量。很多性能问题不是找到一个证据就够了，而是要证明更简单的解释不成立。

## 常见错误

### 只看 GPU utilization

GPU utilization 高不等于有效吞吐高。

它可能代表：

- 有效矩阵计算。
- 低效 kernel 忙碌。
- 访存等待。
- 小 kernel 频繁执行。
- 通信或同步导致其他资源等待。

必须结合 tokens/s、latency、SM active、HBM bandwidth、kernel timeline 和业务阶段看。

### 用 profiler trace 代替 benchmark

Profiler 会改变系统行为。trace 适合解释原因，不适合作为最终性能数字。

最终性能仍应来自受控 benchmark。

### 优化非关键路径

某个 kernel 看起来很慢，但如果它只占端到端 1%，优化 50% 也几乎没有收益。

优先级应该按：

```text
端到端占比 * 可优化空间 * 修改风险
```

### 忽略 workload 分布

推理服务中，输入长度、输出长度、并发、cache 命中率、RAG 文档数和工具调用都会改变瓶颈。

训练任务中，sequence length、batch、并行策略、数据格式和 checkpoint 频率都会改变瓶颈。

一个 workload 上的 profiler 结论，不能自动外推到所有 workload。

### 没有保存环境信息

没有环境信息的 trace 很难复现。

至少保存：

- commit hash。
- container image digest。
- driver/CUDA/NCCL/PyTorch 版本。
- GPU 型号和数量。
- CPU、内存、NUMA。
- 网络和存储配置。
- benchmark 参数。
- profiler 参数。

### 只采一次 trace 就下结论

单次 trace 可能碰到偶发噪声。

更稳的做法：

- 先确认 benchmark 可重复复现。
- 至少采集多个窗口或多个 run。
- 对 p99 问题采集命中 p99 的请求窗口。
- 对分布式问题采集 slow rank 和 normal rank 对照。
- 对 A/B 修改做 paired comparison。

如果现象本身是间歇性的，报告要写明触发条件和复现概率。

## 最小检查清单

开始 profiler 前：

- 是否写了 Profiling Contract？
- 是否已经明确症状指标？
- 是否有可复现 workload？
- 是否记录环境和版本？
- 是否区分 warmup 和 steady state？
- 是否知道要观察哪个阶段？
- 是否定义了成功标准和保护指标？

采集 profiler 时：

- 是否控制 trace 时间窗口？
- 是否避免全量 rank 长时间采样？
- 是否加了 NVTX 标注？
- 是否同步采集应用指标和系统指标？
- 是否保留原始 trace 和命令？
- 是否记录 run_id、request_id、step_id、rank_id、node_id？
- 是否记录 profiler 自身开销？

分析后：

- 是否证明瓶颈在关键路径上？
- 是否排除了更简单的解释？
- 是否做了 before/after benchmark？
- 是否说明适用范围？
- 是否把经验沉淀成 benchmark、告警或 runbook？
- 是否保存 evidence table、产物路径和排除项？
- 是否把 confirmed root cause 和 observation 区分开？

## 小结

Profiler 工具链的核心不是工具名字，而是方法顺序。

一条可靠路径是：

```text
monitoring 发现症状
  -> benchmark 复现和量化
  -> 阶段拆分缩小范围
  -> PyTorch Profiler 找框架热点
  -> Nsight Systems 看端到端时间线
  -> Nsight Compute 深入关键 kernel
  -> DCGM/perf/eBPF 排查系统层
  -> 最小修改
  -> benchmark 验证
  -> 写入报告和回归检测
```

当团队形成这样的证据链后，性能优化就不再依赖经验猜测，而可以变成可复现、可审查、可积累的工程能力。

## 参考资料

- [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html)
- [NVIDIA DCGM Documentation](https://docs.nvidia.com/datacenter/dcgm/latest/)
- [Linux perf Wiki](https://perf.wiki.kernel.org/index.php/Main_Page)
- [eBPF Introduction](https://ebpf.io/what-is-ebpf/)
