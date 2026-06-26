---
title: AI 知识地图
status: reviewed
owner: maintainers
license: CC-BY-4.0
updated: 2026-06-26
---

# AI 知识地图

这张地图面向 AI Systems、AI Infrastructure 和高效 AI 计算方向的新生。它分成三层：第一层用思维导图建立全局方向，第二层按章节列出可点击的分组导航，第三层按学习目标给出推荐路径。

前置科普层只负责讲清 AI 基础概念、Transformer、训练、推理和多模态五件事；后续章节再进入推理系统、训练系统、Kernel、具体硬件平台、通用加速器架构、集群、Benchmark、可靠性和案例沉淀。

## 总览思维导图

<nav class="kg-mindmap" aria-label="AI Knowledge Graph mind map">
  <a class="kg-mindmap-root" href="../">
    <strong>AI Knowledge Graph</strong>
    <span>更快 / 更省 / 更稳 / 可复现</span>
  </a>
  <div class="kg-mindmap-branches">
    <section class="kg-mindmap-branch">
      <a class="kg-branch-title" href="../01-getting-started/">学习入口</a>
      <ul>
        <li class="kg-node-main"><a href="../01-getting-started/">1 入门导读</a></li>
        <li class="kg-node-note"><span>关键词：问题意识 / 阅读方法 / 实验纪律</span></li>
      </ul>
    </section>
    <section class="kg-mindmap-branch">
      <a class="kg-branch-title" href="../02-ai-workloads/">工作负载基础</a>
      <ul>
        <li class="kg-node-main"><a href="../02-ai-workloads/">2 AI 计算工作负载基础</a></li>
        <li class="kg-node-child"><a href="../02-ai-workloads/ai-fundamentals/">2.1 AI 基础概念</a></li>
        <li class="kg-node-child"><a href="../02-ai-workloads/transformer/">2.2 Transformer 流程与原理</a></li>
        <li class="kg-node-child"><a href="../02-ai-workloads/training-primer/">2.3 训练过程与原理</a></li>
        <li class="kg-node-child"><a href="../02-ai-workloads/inference-primer/">2.4 推理过程与原理</a></li>
        <li class="kg-node-child"><a href="../02-ai-workloads/multimodal-primer/">2.5 多模态原理</a></li>
        <li class="kg-node-note"><span>关键词：token / embedding / tensor / logits / loss / forward / backward / Attention / 逐 token 生成 / 多模态理解与生成</span></li>
      </ul>
    </section>
    <section class="kg-mindmap-branch">
      <a class="kg-branch-title" href="../12-hardware-basics/">硬件基础</a>
      <ul>
        <li class="kg-node-main"><a href="../12-hardware-basics/">硬件基础：GPU / NPU / 昇腾平台</a></li>
        <li class="kg-node-child"><a href="../12-hardware-basics/gpu-architecture-basics/">GPU 架构基础</a></li>
        <li class="kg-node-child"><a href="../12-hardware-basics/npu-basics/">NPU 基础概念</a></li>
        <li class="kg-node-child"><a href="../12-hardware-basics/gpu-npu-comparison/">GPU 与 NPU 异同点</a></li>
        <li class="kg-node-child"><a href="../12-hardware-basics/ascend-npu-models/">昇腾 NPU 型号与架构映射</a></li>
        <li class="kg-node-child"><a href="../12-hardware-basics/ascend-910-series/">Ascend 910 系列</a></li>
        <li class="kg-node-child"><a href="../12-hardware-basics/ascend-950-series/">Ascend 950 系列</a></li>
        <li class="kg-node-child"><a href="../12-hardware-basics/cann-stack/">CANN 软件栈与开发入口</a></li>
        <li class="kg-node-child"><a href="../12-hardware-basics/ai-skills-sample/">NPU 相关 AI Skills 样例</a></li>
        <li class="kg-node-note"><span>关键词：GPU / SM / warp / Tensor Core / NPU / Ascend / Atlas / CANN / SocVersion / NpuArch / __NPU_ARCH__ / skills</span></li>
      </ul>
    </section>
    <section class="kg-mindmap-branch">
      <a class="kg-branch-title" href="../03-inference-systems/">推理系统</a>
      <ul>
        <li class="kg-node-main"><a href="../03-inference-systems/">3 推理系统与优化</a></li>
        <li class="kg-node-child"><a href="../03-inference-systems/request-lifecycle/">请求生命周期</a></li>
        <li class="kg-node-child"><a href="../03-inference-systems/prefill-decode/">Prefill / Decode</a></li>
        <li class="kg-node-child"><a href="../03-inference-systems/kv-cache/">KV Cache 与缓存复用</a></li>
        <li class="kg-node-child"><a href="../03-inference-systems/scheduling/">Batching 与调度</a></li>
        <li class="kg-node-child"><a href="../03-inference-systems/single-node-serving-architecture/">单机 / 多机部署</a></li>
        <li class="kg-node-child"><a href="../03-inference-systems/vllm/">推理引擎与框架</a></li>
        <li class="kg-node-child"><a href="../03-inference-systems/benchmark-profiling/">Benchmark 与性能剖析</a></li>
        <li class="kg-node-note"><span>关键词：TTFT / TPOT / batching / scheduling / KV Cache / PagedAttention / Prefix Cache / PDD / AFD / 沙箱快启 / PVM / vLLM / TensorRT-LLM / SGLang</span></li>
      </ul>
    </section>
    <section class="kg-mindmap-branch">
      <a class="kg-branch-title" href="../04-training-systems/">训练系统</a>
      <ul>
        <li class="kg-node-main"><a href="../04-training-systems/">4 训练系统与优化</a></li>
        <li class="kg-node-child"><a href="../04-training-systems/training-lifecycle/">训练生命周期与数据</a></li>
        <li class="kg-node-child"><a href="../04-training-systems/backward-autograd-gradient-lifecycle/">Backward 与梯度</a></li>
        <li class="kg-node-child"><a href="../04-training-systems/collective-communication-primitives/">分布式运行时与通信</a></li>
        <li class="kg-node-child"><a href="../04-training-systems/hybrid-parallelism-composition/">并行策略组合</a></li>
        <li class="kg-node-child"><a href="../04-training-systems/memory-composition-optimization/">显存、精度与稳定性</a></li>
        <li class="kg-node-child"><a href="../04-training-systems/flux-kernel-fusion/">通信重叠与 FLUX</a></li>
        <li class="kg-node-child"><a href="../04-training-systems/muon-optimizer/">优化器、微调与后训练</a></li>
        <li class="kg-node-child"><a href="../04-training-systems/training-benchmark-profiling/">复现、Checkpoint 与 Benchmark</a></li>
        <li class="kg-node-note"><span>关键词：DP / FSDP / ZeRO / TP / PP / EP / CP / MoE / activation checkpointing / mixed precision / Muon / LoRA / RLHF</span></li>
      </ul>
    </section>
    <section class="kg-mindmap-branch">
      <a class="kg-branch-title" href="../05-kernels-compilers/">Kernel 与编译</a>
      <ul>
        <li class="kg-node-main"><a href="../05-kernels-compilers/">5 Kernel、算子与编译优化</a></li>
        <li class="kg-node-child"><a href="../05-kernels-compilers/attention-computation-patterns/">Attention 机制与计算模式</a></li>
        <li class="kg-node-child"><a href="../05-kernels-compilers/triton/">Triton Kernel 编程</a></li>
        <li class="kg-node-child"><a href="../05-kernels-compilers/torchinductor/">TorchInductor 与 PyTorch 编译栈</a></li>
        <li class="kg-node-child"><a href="../05-kernels-compilers/mlir-ai-compiler-ir/">MLIR 与 AI 编译 IR</a></li>
        <li class="kg-node-child"><a href="../05-kernels-compilers/tilelang/">TileLang</a></li>
        <li class="kg-node-child"><a href="../05-kernels-compilers/megakernel-persistent-automatic-generation/">MegaKernel 与 Persistent Kernel</a></li>
        <li class="kg-node-note"><span>关键词：Dense Attention / Sparse Attention / FlashAttention / Triton / Triton 编译器 / Triton AI 算子生成 / TorchInductor / MLIR / TileLang / OpenTileIR / Persistent Kernel / MegaKernel</span></li>
      </ul>
    </section>
    <section class="kg-mindmap-branch">
      <a class="kg-branch-title" href="../06-accelerators-architecture/">计算架构</a>
      <ul>
        <li class="kg-node-main"><a href="../06-accelerators-architecture/">6 AI 加速器与计算架构</a></li>
        <li class="kg-node-child"><a href="../06-accelerators-architecture/performance-model-roofline/">性能模型与 Roofline</a></li>
        <li class="kg-node-child"><a href="../06-accelerators-architecture/compute-units-simt-tensorcore/">计算单元</a></li>
        <li class="kg-node-child"><a href="../06-accelerators-architecture/memory-hierarchy-data-reuse/">存储层次</a></li>
        <li class="kg-node-child"><a href="../06-accelerators-architecture/interconnect-communication-architecture/">互连与通信架构</a></li>
        <li class="kg-node-note"><span>关键词：GPU / NPU / Tensor Core / HBM / SRAM / interconnect / power / workload mapping</span></li>
      </ul>
    </section>
    <section class="kg-mindmap-branch">
      <a class="kg-branch-title" href="../07-cluster-infra/">集群基础设施</a>
      <ul>
        <li class="kg-node-main"><a href="../07-cluster-infra/">7 集群、网络、存储与调度</a></li>
        <li class="kg-node-child"><a href="../07-cluster-infra/ai-cluster-architecture-overview/">集群架构总览</a></li>
        <li class="kg-node-child"><a href="../07-cluster-infra/scheduling-queues-resource-management/">调度与资源队列</a></li>
        <li class="kg-node-child"><a href="../07-cluster-infra/rdma-network-nccl-topology-congestion/">RDMA 与 NCCL 网络</a></li>
        <li class="kg-node-child"><a href="../07-cluster-infra/storage-data-cache-checkpoint/">存储、缓存与 Checkpoint</a></li>
        <li class="kg-node-note"><span>关键词：Slurm / Kubernetes / GPU topology / RDMA / NCCL / storage / multitenancy / utilization</span></li>
      </ul>
    </section>
    <section class="kg-mindmap-branch">
      <a class="kg-branch-title" href="../08-benchmark-capacity/">度量与可靠性</a>
      <ul>
        <li class="kg-node-main"><a href="../08-benchmark-capacity/">8 性能分析、Benchmark 与容量建模</a></li>
        <li class="kg-node-main"><a href="../09-reliability-observability/">9 可靠性、可观测性与故障复盘</a></li>
        <li class="kg-node-main"><a href="../10-papers-cases/">10 论文复现与系统案例</a></li>
        <li class="kg-node-child"><a href="../08-benchmark-capacity/profiler-toolchain-bottleneck-analysis/">Profiler 工具链</a></li>
        <li class="kg-node-child"><a href="../09-reliability-observability/observability-overview-signals/">可观测性信号</a></li>
        <li class="kg-node-child"><a href="../10-papers-cases/ai-system-architecture/">系统论文与架构分析</a></li>
        <li class="kg-node-note"><span>关键词：profiling / roofline / capacity / SLO / failure modes / runbook / paper card / ADR</span></li>
      </ul>
    </section>
    <section class="kg-mindmap-branch">
      <a class="kg-branch-title" href="../11-knowledge-index/">知识沉淀</a>
      <ul>
        <li class="kg-node-main"><a href="../11-knowledge-index/">11 知识组织、模板与 AI 可读索引</a></li>
        <li class="kg-node-main"><a href="../99-templates/knowledge-note/">99 模板与资源</a></li>
        <li class="kg-node-note"><span>关键词：metadata / provenance / llms.txt / vector index / knowledge graph / ADR / benchmark report / AI-readable skills</span></li>
      </ul>
    </section>
  </div>
</nav>

## 地图逻辑

| 主线 | 组织逻辑 | 对应模块 |
| --- | --- | --- |
| 学习入口 | 先建立 AI Infra 的问题意识、阅读方法和实验纪律。 | [1 入门导读](01-getting-started/index.md) |
| 工作负载基础 | 只做入门科普：讲清 AI 基础概念、Transformer、训练、推理和多模态原理。 | [2 AI 计算工作负载基础](02-ai-workloads/index.md) |
| 具体硬件平台 | 研究 GPU/NPU 执行模型、GPU 与 NPU 差异、昇腾型号映射、CANN 软件栈、平台证据收集，以及如何把硬件适配经验沉淀成 AI skill。 | [硬件基础](12-hardware-basics/index.md) |
| 推理系统 | 从请求生命周期出发，理解 Prefill/Decode、指标、batching、KV Cache、调度、部署架构、推理引擎和 RAG/Agent 负载。 | [3 推理系统与优化](03-inference-systems/index.md) |
| 训练系统 | 从训练 step 出发，理解数据、batch、loss、backward、分布式通信、并行策略、显存、稳定性、优化器、后训练和 benchmark。 | [4 训练系统与优化](04-training-systems/index.md) |
| 单机执行与编译 | 研究 Attention、Triton、TorchInductor、MLIR、TileLang、MegaKernel 如何决定 kernel 级性能和自动生成能力。 | [5 Kernel、算子与编译优化](05-kernels-compilers/index.md) |
| 架构与集群 | 研究计算单元、存储层次、互连、功耗、架构取舍，以及集群调度、网络、存储和资源治理。 | [6 AI 加速器与计算架构](06-accelerators-architecture/index.md)、[7 集群、网络、存储与调度](07-cluster-infra/index.md) |
| 度量与沉淀 | 用 Benchmark、Profiling、容量模型、可靠性、故障复盘、论文复现和 ADR 把经验变成可复用知识。 | [8 性能分析、Benchmark 与容量建模](08-benchmark-capacity/index.md)、[9 可靠性、可观测性与故障复盘](09-reliability-observability/index.md)、[10 论文复现与系统案例](10-papers-cases/index.md)、[11 知识组织、模板与 AI 可读索引](11-knowledge-index/index.md) |

## 分章导航

### 1 入门导读

- [入门导读](01-getting-started/index.md)：学习路线、术语约定、实验纪律和贡献方法。

### 2 AI 计算工作负载基础

- 基础概念：[AI 基础概念](02-ai-workloads/ai-fundamentals.md)
- 模型原理：[Transformer 流程与原理](02-ai-workloads/transformer.md)
- 任务流程：[训练过程与原理](02-ai-workloads/training-primer.md)、[推理过程与原理](02-ai-workloads/inference-primer.md)、[多模态原理](02-ai-workloads/multimodal-primer.md)

### 硬件基础

- GPU 入门：[GPU 架构基础](12-hardware-basics/gpu-architecture-basics.md)
- NPU 入门：[NPU 基础概念](12-hardware-basics/npu-basics.md)
- GPU/NPU 对比：[GPU 与 NPU 异同点](12-hardware-basics/gpu-npu-comparison.md)
- 昇腾平台：[昇腾 NPU 型号与架构映射](12-hardware-basics/ascend-npu-models.md)、[Ascend 910 系列](12-hardware-basics/ascend-910-series.md)、[Ascend 950 系列](12-hardware-basics/ascend-950-series.md)
- 软件栈与 AI 输入：[CANN 软件栈与开发入口](12-hardware-basics/cann-stack.md)、[NPU 相关 AI Skills 样例](12-hardware-basics/ai-skills-sample.md)

### 3 推理系统与优化

- 请求链路：[推理请求生命周期](03-inference-systems/request-lifecycle.md)、[Prefill 与 Decode](03-inference-systems/prefill-decode.md)、[指标体系](03-inference-systems/metrics.md)
- 批处理与调度：[Batching](03-inference-systems/batching.md)、[调度策略](03-inference-systems/scheduling.md)
- KV 与缓存：[KV Cache](03-inference-systems/kv-cache.md)、[PagedAttention](03-inference-systems/paged-attention.md)、[Prefix Cache](03-inference-systems/prefix-cache.md)、[缓存体系](03-inference-systems/cache-system.md)
- 推理优化技术：[量化推理](03-inference-systems/quantization.md)、[Speculative Decoding](03-inference-systems/speculative-decoding.md)、[MoE 模型推理优化](03-inference-systems/moe-inference-optimization.md)、[EP Size 与大 EP / 小 EP](03-inference-systems/ep-size-large-small-ep.md)
- 部署架构：[Prefill/Decode 分离部署](03-inference-systems/prefill-decode-disaggregation.md)、[单机推理服务架构](03-inference-systems/single-node-serving-architecture.md)、[多机分布式推理](03-inference-systems/distributed-inference.md)
- 推理引擎：[vLLM](03-inference-systems/vllm.md)、[TensorRT-LLM](03-inference-systems/tensorrt-llm.md)、[SGLang](03-inference-systems/sglang.md)
- 复合负载与评测：[RAG 与 Agent 推理负载](03-inference-systems/rag-agent-workloads.md)、[Benchmark 方法](03-inference-systems/benchmark-methodology.md)、[Benchmark 方法与性能剖析](03-inference-systems/benchmark-profiling.md)

### 4 训练系统与优化

- 训练闭环与数据：[训练任务生命周期](04-training-systems/training-lifecycle.md)、[数据输入与 Data Pipeline](04-training-systems/data-pipeline.md)、[训练数据混合、采样与有效 Token](04-training-systems/training-data-mixing-sampling-effective-tokens.md)、[Batch、Micro-batch 与 Gradient Accumulation](04-training-systems/batch-gradient-accumulation.md)、[大词表输出层、Logits 与 Cross Entropy 系统优化](04-training-systems/vocab-output-cross-entropy.md)
- Backward、显存与数值：[Backward、Autograd Graph 与梯度生命周期](04-training-systems/backward-autograd-gradient-lifecycle.md)、[显存组成与优化总览](04-training-systems/memory-composition-optimization.md)、[Activation Checkpointing](04-training-systems/activation-checkpointing.md)、[混合精度训练](04-training-systems/mixed-precision-training.md)、[训练稳定性与数值异常](04-training-systems/training-stability-numerical-debugging.md)
- 分布式运行时与通信：[分布式训练启动与运行时](04-training-systems/distributed-training-runtime.md)、[Collective 通信原语与通信量模型](04-training-systems/collective-communication-primitives.md)、[通信与计算重叠](04-training-systems/communication-computation-overlap.md)、[FLUX 通信重叠与 Kernel Fusion](04-training-systems/flux-kernel-fusion.md)
- 并行策略：[Data Parallel 与梯度同步](04-training-systems/data-parallel-gradient-sync.md)、[ZeRO 与 FSDP](04-training-systems/zero-fsdp.md)、[Tensor Parallel](04-training-systems/tensor-parallel.md)、[Sequence Parallel 与 Context Parallel](04-training-systems/sequence-context-parallel.md)、[Pipeline Parallel](04-training-systems/pipeline-parallel.md)、[Expert Parallel 与 MoE 训练](04-training-systems/expert-parallel-moe-training.md)、[并行策略组合](04-training-systems/hybrid-parallelism-composition.md)
- 优化器、微调与后训练：[Optimizer 与 Scheduler 系统成本](04-training-systems/optimizer-scheduler-cost.md)、[Muon 优化器](04-training-systems/muon-optimizer.md)、[参数高效微调：LoRA、QLoRA 与 Adapter 系统优化](04-training-systems/parameter-efficient-finetuning-lora-qlora.md)、[后训练工作负载：SFT、DPO、RLHF 与 GRPO 系统视角](04-training-systems/post-training-workloads-sft-dpo-rlhf-grpo.md)
- 评估、复现与工程治理：[Evaluation、Validation 与 Checkpoint Selection](04-training-systems/evaluation-validation-checkpoint-selection.md)、[训练可复现性、随机性与 Run Manifest](04-training-systems/training-reproducibility-randomness-run-manifest.md)、[Checkpoint、Resume 与容错](04-training-systems/checkpoint-resume-fault-tolerance.md)、[训练性能指标与扩展效率](04-training-systems/training-performance-metrics-scaling.md)、[训练性能剖析与 Benchmark](04-training-systems/training-benchmark-profiling.md)、[DeepSpeed、Megatron-LM 与 PyTorch FSDP](04-training-systems/deepspeed-megatron-fsdp.md)

### 5 Kernel、算子与编译优化

- 计算模式：[Attention 机制与计算模式](05-kernels-compilers/attention-computation-patterns.md)
- Kernel 编程：[Triton Kernel 编程](05-kernels-compilers/triton.md)、[TileLang：面向 AI Kernel 的 Tile 编程模型](05-kernels-compilers/tilelang.md)
- 编译栈与 IR：[TorchInductor 与 PyTorch 编译栈](05-kernels-compilers/torchinductor.md)、[MLIR 与 AI 编译 IR](05-kernels-compilers/mlir-ai-compiler-ir.md)
- 激进融合与自动生成：[MegaKernel、Persistent Kernel 与自动生成](05-kernels-compilers/megakernel-persistent-automatic-generation.md)

### 6 AI 加速器与计算架构

- 性能上限：[AI 加速器性能模型：算力、带宽与 Roofline](06-accelerators-architecture/performance-model-roofline.md)
- 计算与存储：[计算单元：SIMT、Tensor Core 与矩阵引擎](06-accelerators-architecture/compute-units-simt-tensorcore.md)、[存储层次：HBM、SRAM、Cache 与数据复用](06-accelerators-architecture/memory-hierarchy-data-reuse.md)、[精度格式：FP16、BF16、FP8 与量化计算](06-accelerators-architecture/precision-formats-low-bit-compute.md)
- 系统架构：[互连与通信架构](06-accelerators-architecture/interconnect-communication-architecture.md)、[功耗、散热、频率与可靠性](06-accelerators-architecture/power-thermal-reliability.md)、[架构取舍](06-accelerators-architecture/accelerator-architecture-tradeoffs.md)、[Workload Mapping](06-accelerators-architecture/workload-mapping-compiler-runtime-interface.md)

### 7 集群、网络、存储与调度

- 集群与调度：[AI 集群架构总览](07-cluster-infra/ai-cluster-architecture-overview.md)、[调度系统与资源队列](07-cluster-infra/scheduling-queues-resource-management.md)
- 节点与网络：[GPU 拓扑、NUMA、MIG/MPS 与资源隔离](07-cluster-infra/gpu-topology-numa-mig-mps-isolation.md)、[RDMA 网络与 NCCL 拓扑](07-cluster-infra/rdma-network-nccl-topology-congestion.md)
- 存储与环境：[存储、数据缓存与 Checkpoint](07-cluster-infra/storage-data-cache-checkpoint.md)、[环境可复现](07-cluster-infra/environment-reproducibility-containers.md)
- 平台治理：[混合集群与多租户隔离](07-cluster-infra/mixed-workload-multitenancy-isolation.md)、[资源利用率、碎片与容量治理](07-cluster-infra/resource-utilization-fragmentation-capacity.md)、[节点生命周期与集群运维](07-cluster-infra/node-lifecycle-health-maintenance.md)

### 8 性能分析、Benchmark 与容量建模

- 方法论：[性能分析与 Benchmark 方法论](08-benchmark-capacity/performance-analysis-benchmark-methodology.md)、[Benchmark 负载设计与 Trace Replay](08-benchmark-capacity/benchmark-workload-design-trace-replay.md)、[Benchmark 数据治理与实验记录](08-benchmark-capacity/benchmark-data-governance-run-records.md)
- 容量建模：[推理容量建模](08-benchmark-capacity/inference-capacity-modeling.md)、[训练容量建模](08-benchmark-capacity/training-capacity-scaling-efficiency.md)、[排队模型与尾延迟](08-benchmark-capacity/queueing-model-tail-latency.md)
- 瓶颈定位：[Profiler 工具链与瓶颈定位](08-benchmark-capacity/profiler-toolchain-bottleneck-analysis.md)、[Roofline 分析](08-benchmark-capacity/roofline-analysis-compute-bandwidth.md)
- 工程决策：[能效、功耗与热限制](08-benchmark-capacity/energy-power-thermal-benchmark.md)、[A/B 对比、消融实验与性能回归检测](08-benchmark-capacity/ab-testing-ablation-regression-detection.md)、[成本模型与单位经济性](08-benchmark-capacity/cost-model-unit-economics.md)

### 9 可靠性、可观测性与故障复盘

- [AI 系统可观测性总览](09-reliability-observability/observability-overview-signals.md)
- [SLO、SLI、错误预算与告警策略](09-reliability-observability/slo-sli-error-budget-alerting.md)
- [AI 系统故障模式](09-reliability-observability/ai-failure-modes-gpu-nccl-network-storage.md)
- [Incident Response、Runbook 与故障复盘](09-reliability-observability/incident-response-runbook-postmortem.md)

### 10 论文复现与系统案例

- [AI 系统论文与架构](10-papers-cases/ai-system-architecture.md)
- [技术决策记录](10-papers-cases/adr.md)
- [故障案例库](10-papers-cases/failure-cases.md)

### 11 知识组织、模板与 AI 可读索引

- 知识组织：[知识组织、模板与 AI 可读索引](11-knowledge-index/index.md)
- 模板：[知识点模板](99-templates/knowledge-note.md)、[技术决策模板](99-templates/adr.md)、[基准实验报告模板](99-templates/benchmark-report.md)

## 按目标导航

| 当前目标 | 优先阅读 |
| --- | --- |
| 刚进入 AI Infra 方向 | [1 入门导读](01-getting-started/index.md) -> [AI 基础概念](02-ai-workloads/ai-fundamentals.md) -> [Transformer 流程与原理](02-ai-workloads/transformer.md) -> [训练过程与原理](02-ai-workloads/training-primer.md) -> [推理过程与原理](02-ai-workloads/inference-primer.md) |
| 想先搞懂训练和推理怎么回事 | [训练过程与原理](02-ai-workloads/training-primer.md) -> [推理过程与原理](02-ai-workloads/inference-primer.md) -> [推理请求生命周期](03-inference-systems/request-lifecycle.md) -> [训练任务生命周期](04-training-systems/training-lifecycle.md) |
| 想降低 LLM 推理延迟 | [Prefill 与 Decode](03-inference-systems/prefill-decode.md) -> [指标体系](03-inference-systems/metrics.md) -> [Batching](03-inference-systems/batching.md) -> [KV Cache](03-inference-systems/kv-cache.md) -> [调度策略](03-inference-systems/scheduling.md) -> [Benchmark 方法与性能剖析](03-inference-systems/benchmark-profiling.md) |
| 想提高推理吞吐和并发 | [PagedAttention](03-inference-systems/paged-attention.md) -> [Prefix Cache](03-inference-systems/prefix-cache.md) -> [单机推理服务架构](03-inference-systems/single-node-serving-architecture.md) -> [多机分布式推理](03-inference-systems/distributed-inference.md) -> [vLLM](03-inference-systems/vllm.md) |
| 想优化 MoE 推理 | [MoE 模型推理优化](03-inference-systems/moe-inference-optimization.md) -> [EP Size 与大 EP / 小 EP](03-inference-systems/ep-size-large-small-ep.md) -> [调度策略](03-inference-systems/scheduling.md) -> [多机分布式推理](03-inference-systems/distributed-inference.md) |
| 想做训练系统 | [训练任务生命周期](04-training-systems/training-lifecycle.md) -> [Backward、Autograd Graph 与梯度生命周期](04-training-systems/backward-autograd-gradient-lifecycle.md) -> [分布式训练启动与运行时](04-training-systems/distributed-training-runtime.md) -> [Collective 通信原语与通信量模型](04-training-systems/collective-communication-primitives.md) -> [训练性能剖析与 Benchmark](04-training-systems/training-benchmark-profiling.md) |
| 想做分布式训练并行策略 | [Data Parallel 与梯度同步](04-training-systems/data-parallel-gradient-sync.md) -> [ZeRO 与 FSDP](04-training-systems/zero-fsdp.md) -> [Tensor Parallel](04-training-systems/tensor-parallel.md) -> [Pipeline Parallel](04-training-systems/pipeline-parallel.md) -> [Expert Parallel 与 MoE 训练](04-training-systems/expert-parallel-moe-training.md) -> [并行策略组合](04-training-systems/hybrid-parallelism-composition.md) |
| 想降低训练显存和提升稳定性 | [显存组成与优化总览](04-training-systems/memory-composition-optimization.md) -> [Activation Checkpointing](04-training-systems/activation-checkpointing.md) -> [混合精度训练](04-training-systems/mixed-precision-training.md) -> [训练稳定性与数值异常](04-training-systems/training-stability-numerical-debugging.md) |
| 想做 Kernel 或编译优化 | [Attention 机制与计算模式](05-kernels-compilers/attention-computation-patterns.md) -> [Triton Kernel 编程](05-kernels-compilers/triton.md) -> [TorchInductor 与 PyTorch 编译栈](05-kernels-compilers/torchinductor.md) -> [MLIR 与 AI 编译 IR](05-kernels-compilers/mlir-ai-compiler-ir.md) -> [TileLang：面向 AI Kernel 的 Tile 编程模型](05-kernels-compilers/tilelang.md) |
| 想研究 MegaKernel / Persistent Kernel | [Attention 机制与计算模式](05-kernels-compilers/attention-computation-patterns.md) -> [Triton Kernel 编程](05-kernels-compilers/triton.md) -> [MLIR 与 AI 编译 IR](05-kernels-compilers/mlir-ai-compiler-ir.md) -> [MegaKernel、Persistent Kernel 与自动生成](05-kernels-compilers/megakernel-persistent-automatic-generation.md) |
| 想做 NPU 或昇腾平台适配 | [GPU 与 NPU 异同点](12-hardware-basics/gpu-npu-comparison.md) -> [NPU 基础概念](12-hardware-basics/npu-basics.md) -> [昇腾 NPU 型号与架构映射](12-hardware-basics/ascend-npu-models.md) -> [CANN 软件栈与开发入口](12-hardware-basics/cann-stack.md) -> [NPU 相关 AI Skills 样例](12-hardware-basics/ai-skills-sample.md) |
| 想做 AI 加速器或硬件架构 | [GPU 架构基础](12-hardware-basics/gpu-architecture-basics.md) -> [NPU 基础概念](12-hardware-basics/npu-basics.md) -> [GPU 与 NPU 异同点](12-hardware-basics/gpu-npu-comparison.md) -> [AI 加速器性能模型](06-accelerators-architecture/performance-model-roofline.md) -> [计算单元](06-accelerators-architecture/compute-units-simt-tensorcore.md) -> [存储层次](06-accelerators-architecture/memory-hierarchy-data-reuse.md) -> [互连与通信架构](06-accelerators-architecture/interconnect-communication-architecture.md) -> [Workload Mapping](06-accelerators-architecture/workload-mapping-compiler-runtime-interface.md) |
| 想建设稳定集群或实验平台 | [AI 集群架构总览](07-cluster-infra/ai-cluster-architecture-overview.md) -> [调度系统与资源队列](07-cluster-infra/scheduling-queues-resource-management.md) -> [GPU 拓扑、NUMA、MIG/MPS 与资源隔离](07-cluster-infra/gpu-topology-numa-mig-mps-isolation.md) -> [RDMA 网络与 NCCL 拓扑](07-cluster-infra/rdma-network-nccl-topology-congestion.md) -> [存储、数据缓存与 Checkpoint](07-cluster-infra/storage-data-cache-checkpoint.md) |
| 想做 Benchmark 和容量规划 | [性能分析与 Benchmark 方法论](08-benchmark-capacity/performance-analysis-benchmark-methodology.md) -> [Profiler 工具链与瓶颈定位](08-benchmark-capacity/profiler-toolchain-bottleneck-analysis.md) -> [推理容量建模](08-benchmark-capacity/inference-capacity-modeling.md) -> [训练容量建模](08-benchmark-capacity/training-capacity-scaling-efficiency.md) -> [成本模型与单位经济性](08-benchmark-capacity/cost-model-unit-economics.md) |
| 想做可靠性和事故复盘 | [AI 系统可观测性总览](09-reliability-observability/observability-overview-signals.md) -> [SLO、SLI、错误预算与告警策略](09-reliability-observability/slo-sli-error-budget-alerting.md) -> [AI 系统故障模式](09-reliability-observability/ai-failure-modes-gpu-nccl-network-storage.md) -> [Incident Response、Runbook 与故障复盘](09-reliability-observability/incident-response-runbook-postmortem.md) |
| 想复现系统论文或沉淀决策 | [AI 系统论文与架构](10-papers-cases/ai-system-architecture.md) -> [技术决策记录](10-papers-cases/adr.md) -> [故障案例库](10-papers-cases/failure-cases.md) -> [知识组织、模板与 AI 可读索引](11-knowledge-index/index.md) |

## 模块关系

| 模块 | 上游依赖 | 主要产出 |
| --- | --- | --- |
| 1 入门导读 | 无 | 学习路线、术语约定、实验纪律、贡献方法 |
| 2 AI 计算工作负载基础 | 1 | AI 基础概念、Transformer、训练流程、推理流程和多模态原理 |
| 硬件基础 | 2、5、6、8 | GPU 架构、NPU 基础、GPU/NPU 对比、昇腾型号映射、Ascend 910/950、CANN 软件栈、平台证据收集和 AI skill 样例 |
| 3 推理系统与优化 | 2、5、6、8、硬件基础 | 推理链路、KV Cache、调度、缓存、量化、MoE、部署架构、推理引擎和 benchmark |
| 4 训练系统与优化 | 2、5、6、7、8、硬件基础 | 数据、batch、loss、backward、并行策略、通信、显存、稳定性、优化器、后训练、checkpoint 和 benchmark |
| 5 Kernel、算子与编译优化 | 2、6、8、硬件基础 | Attention 计算模式、Triton Kernel、TorchInductor、MLIR、TileLang、MegaKernel、算子实现、图优化和自动调优 |
| 6 AI 加速器与计算架构 | 2、5、8、硬件基础 | 计算、存储、互连、精度、功耗、体系结构取舍和 workload mapping |
| 7 集群、网络、存储与调度 | 3、4、6、8 | 资源调度、拓扑、网络、存储、环境、隔离、容量治理和节点生命周期 |
| 8 性能分析、Benchmark 与容量建模 | 2、3、4、5、6、7 | 指标体系、profiling、roofline、容量估算、trace replay、A/B、能效和成本模型 |
| 9 可靠性、可观测性与故障复盘 | 3、4、7、8 | 监控、告警、SLO、错误预算、故障模式、事故响应、runbook 和复盘 |
| 10 论文复现与系统案例 | 全部模块 | 系统论文阅读框架、paper card、机制卡片、复现协议、ADR、failure case 和 evidence pack |
| 11 知识组织、模板与 AI 可读索引 | 全部模块 | 文档类型、元数据、标签、引用溯源、证据等级、llms.txt、向量索引、知识图谱和 AI skills |
