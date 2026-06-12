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

- [性能分析与 Benchmark 方法论：指标、实验设计与瓶颈定位](performance-analysis-benchmark-methodology.md)：解释 benchmark、profiling、monitoring 的区别，如何从问题出发定义 Benchmark Contract、测量边界、指标、workload、open-loop/closed-loop 负载、实验矩阵、控制变量、warmup、统计判定、A/A 噪声基线、A/B、ablation、profiler 证据、噪声控制、raw data lineage、发布门禁和容量建模输入。
- [推理容量建模：QPS、并发、TTFT、TPOT 与 GPU 副本数](inference-capacity-modeling.md)：解释如何用 Capacity Contract、请求分布、SLA、offered load/throughput/goodput、单副本 goodput 曲线、请求分桶、KV Cache 预算、prefill/decode 约束、headroom、故障域、rolling update、autoscaling 滞后、路由效率、多模型、过载保护、成本和生产反馈推导推理副本数。
- [训练容量建模：Tokens/s、Step Time、MFU 与扩展效率](training-capacity-scaling-efficiency.md)：解释如何用 Capacity Contract、训练目标、token 口径、global batch、step time、sustained/effective throughput、goodput、MFU、强/弱扩展曲线、边际收益、并行策略、checkpoint、eval、故障恢复、资源准入、headroom、成本、能效和生产校准推导训练 GPU 需求与总完成时间。
- [Profiler 工具链与瓶颈定位：Nsight、PyTorch Profiler、DCGM、perf 与 eBPF](profiler-toolchain-bottleneck-analysis.md)：解释如何用 Profiling Contract、证据链分层、工具选择矩阵、PyTorch Profiler、Nsight Systems、Nsight Compute、DCGM、perf/eBPF、NVTX、时间对齐、分布式 trace、瓶颈模式、结论分级、报告模板和回归防线建立可复现的 AI 系统瓶颈定位方法。
- [Roofline 分析：算力、带宽与瓶颈上限](roofline-analysis-compute-bandwidth.md)：解释如何用 Roofline Analysis Contract、对象层级、FLOPs/bytes 口径、arithmetic intensity、ridge point、理论/实测 roof、多重 roof、profiler 指标、bound efficiency、端到端收益上限、AI 算子案例、推理/训练场景、硬件评估和 before/after benchmark 判断 AI workload 更接近算力、带宽、通信、launch 还是系统开销瓶颈。
- [排队模型与尾延迟：QPS、并发、利用率和 p99](queueing-model-tail-latency.md)：解释如何用 Queueing Contract、offered load/throughput/goodput、Little's Law、Kingman/VUT 直觉、多阶段队列、Prefill/Decode、KV Cache、batching、head-of-line blocking、deadline-aware scheduling、retry amplification、load shedding、open-loop benchmark、goodput 曲线、队列上限和分桶指标分析 AI 推理服务的 p95/p99 与容量边界。
- [能效、功耗与热限制：Power、Energy per Token 与持续吞吐](energy-power-thermal-benchmark.md)：解释如何用 Energy Benchmark Contract 定义 GPU-only、node-level、rack/facility-level 测量边界、idle baseline、telemetry 和热稳态窗口，采集 power、energy、clocks、temperature、throttle reason，并用 power cap sweep、Pareto frontier、prefill/decode 分阶段能耗、cache hit/miss、energy to target、goodput at SLA 和 power-aware scheduling 分析 AI 训练与推理系统能效。
- [Benchmark 负载设计与 Trace Replay：从玩具测试到真实 Workload](benchmark-workload-design-trace-replay.md)：解释如何用 Workload Benchmark Contract 设计 synthetic、sampled、trace replay workload，定义 input/output token 分布、请求类型混合、arrival process、open-loop/closed-loop、coordinated omission、trace 时间语义、cache 状态、RAG/Agent 与多模态负载、训练数据路径、集群 job trace、warmup、measurement window、load generator 校验和报告 caveats。
- [A/B 对比、消融实验与性能回归检测](ab-testing-ablation-regression-detection.md)：解释如何用 Experiment Contract、A/A 噪声基线、A/B 对比、消融矩阵、主指标/保护指标/解释指标、paired runs、paired difference、effect size、practical threshold、baseline 生命周期、rerun policy、CI/nightly/release gate、canary、shadow traffic、回归规则模板和 profiler 证据建立性能实验与回归防线。
- [成本模型与单位经济性：Cost per Token、GPU Hour 与有效产出](cost-model-unit-economics.md)：解释如何用 Cost Model Contract 定义 GPU-only、node、service、workflow、cluster 成本边界，建立 rate card、摊销、边际成本、effective output、cost/request、cost/input token、cost/output token、cost/successful run、cost to target、RAG/Agent workflow 成本、训练有效 token、checkpoint/eval/failure 成本、headroom、缓存 ROI、共享成本归因、showback/chargeback、浪费分类和 dashboard 指标。
- [Benchmark 数据治理与实验记录：Run Manifest、Raw Data 与可复现报告](benchmark-data-governance-run-records.md)：解释如何用 benchmark data contract、数据模型、run manifest、raw data、schema version、artifact digest、lineage、quality gates、baseline registry、retention、CI 记录和 AI-readable run card，让 benchmark 结论可复现、可审查、可比较、可回归检测、可被 AI 检索。
