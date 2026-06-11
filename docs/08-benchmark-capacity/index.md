---
title: 性能分析、Benchmark 与容量建模
domain: benchmark-capacity
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-05-30
---

# 性能分析、Benchmark 与容量建模

本目录是整个知识库的度量核心。任何“更快、更省、更稳”的结论，都必须回到可复现的指标、实验设计、性能剖析和容量模型。

## 建议主题

- latency、throughput、TTFT、TPOT、p95/p99、tokens/s
- step time、MFU、scaling efficiency、communication ratio
- GPU utilization、SM occupancy、HBM bandwidth、memory footprint
- power、energy per token、thermal、frequency throttling
- profiler：Nsight、PyTorch Profiler、NVIDIA DCGM、perf、eBPF
- Roofline model、queueing model、capacity planning
- Benchmark workload design、warmup、sample size、confidence interval
- A/B comparison、ablation、regression detection

## 关键问题

- 指标定义是否清晰，是否能被别人复现。
- Benchmark workload 是否代表真实请求和训练任务。
- 结果是否区分平均值、尾部、波动和异常值。
- 性能瓶颈是否有 profiler 证据，而不是只看现象。
- 容量模型能否解释现有结果并预测扩容后的表现。

## 专题入口

- [性能分析与 Benchmark 方法论：指标、实验设计与瓶颈定位](performance-analysis-benchmark-methodology.md)：解释 benchmark、profiling、monitoring 的区别，如何从问题出发定义指标、设计 workload、控制变量、处理 warmup、做统计、A/B、ablation、profiler 证据和容量建模输入。
- [推理容量建模：QPS、并发、TTFT、TPOT 与 GPU 副本数](inference-capacity-modeling.md)：解释如何用请求分布、SLA、单副本 goodput 曲线、KV Cache 容量、prefill/decode 约束、headroom、冷启动、路由和生产反馈推导推理副本数。
- [训练容量建模：Tokens/s、Step Time、MFU 与扩展效率](training-capacity-scaling-efficiency.md)：解释如何用训练目标、global batch、step time、tokens/s、MFU、强/弱扩展效率、并行策略、checkpoint、eval、故障恢复、排队等待、成本和能效推导训练容量。
- [Profiler 工具链与瓶颈定位：Nsight、PyTorch Profiler、DCGM、perf 与 eBPF](profiler-toolchain-bottleneck-analysis.md)：解释如何把应用指标、PyTorch Profiler、Nsight Systems、Nsight Compute、DCGM、perf/eBPF、NVTX 标注和分布式 trace 组合成可复现的瓶颈定位证据链。
- [Roofline 分析：算力、带宽与瓶颈上限](roofline-analysis-compute-bandwidth.md)：解释如何用 FLOPs、bytes moved、arithmetic intensity、ridge point、profiler 指标和端到端 benchmark 判断 AI workload 更接近算力上限、带宽上限还是系统开销瓶颈。
- [排队模型与尾延迟：QPS、并发、利用率和 p99](queueing-model-tail-latency.md)：解释如何用 Little's Law、利用率、服务时间波动、batching、队列长度、load shedding、open-loop benchmark 和 goodput at SLA 分析 AI 推理服务的 p95/p99 与容量边界。
- [能效、功耗与热限制：Power、Energy per Token 与持续吞吐](energy-power-thermal-benchmark.md)：解释如何定义 GPU/node/rack 测量边界，采集 power、energy、clocks、temperature、throttle reason，并用 steady-state benchmark、power cap sweep、joules/token 和 goodput at SLA 分析 AI 系统能效。
- [Benchmark 负载设计与 Trace Replay：从玩具测试到真实 Workload](benchmark-workload-design-trace-replay.md)：解释如何设计 synthetic、sampled、trace replay workload，定义 input/output token 分布、arrival process、cache 状态、warmup、measurement window、load generator 校验和报告 caveats。
- [A/B 对比、消融实验与性能回归检测](ab-testing-ablation-regression-detection.md)：解释如何用 A/A 噪声基线、A/B 对比、消融实验、主指标/保护指标/解释指标、paired runs、阈值、baseline、CI/nightly/release gate、canary 和 profiler 证据建立性能回归防线。
