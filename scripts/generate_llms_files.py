#!/usr/bin/env python3
"""Generate AI-readable entry files for the documentation repository."""

from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SITE_URL = "https://amoursec.github.io/aikg/"
REPO_URL = "https://github.com/AmourSec/aikg"
RAW_URL = "https://raw.githubusercontent.com/AmourSec/aikg/main"

PRIORITY_DOCS = [
    "index.md",
    "knowledge-map.md",
    "01-getting-started/index.md",
    "02-ai-workloads/index.md",
    "02-ai-workloads/ai-fundamentals.md",
    "02-ai-workloads/transformer.md",
    "02-ai-workloads/training-primer.md",
    "02-ai-workloads/inference-primer.md",
    "02-ai-workloads/multimodal-primer.md",
    "03-inference-systems/index.md",
    "03-inference-systems/request-lifecycle.md",
    "03-inference-systems/prefill-decode.md",
    "03-inference-systems/metrics.md",
    "03-inference-systems/batching.md",
    "03-inference-systems/kv-cache.md",
    "03-inference-systems/paged-attention.md",
    "03-inference-systems/prefix-cache.md",
    "03-inference-systems/scheduling.md",
    "03-inference-systems/quantization.md",
    "03-inference-systems/speculative-decoding.md",
    "03-inference-systems/prefill-decode-disaggregation.md",
    "03-inference-systems/moe-inference-optimization.md",
    "03-inference-systems/ep-size-large-small-ep.md",
    "03-inference-systems/single-node-serving-architecture.md",
    "03-inference-systems/distributed-inference.md",
    "03-inference-systems/cache-system.md",
    "03-inference-systems/benchmark-methodology.md",
    "03-inference-systems/vllm.md",
    "03-inference-systems/tensorrt-llm.md",
    "03-inference-systems/sglang.md",
    "03-inference-systems/rag-agent-workloads.md",
    "03-inference-systems/benchmark-profiling.md",
    "04-training-systems/index.md",
    "04-training-systems/training-lifecycle.md",
    "04-training-systems/data-pipeline.md",
    "04-training-systems/training-data-mixing-sampling-effective-tokens.md",
    "04-training-systems/batch-gradient-accumulation.md",
    "04-training-systems/vocab-output-cross-entropy.md",
    "04-training-systems/memory-composition-optimization.md",
    "04-training-systems/backward-autograd-gradient-lifecycle.md",
    "04-training-systems/distributed-training-runtime.md",
    "04-training-systems/collective-communication-primitives.md",
    "04-training-systems/data-parallel-gradient-sync.md",
    "04-training-systems/zero-fsdp.md",
    "04-training-systems/tensor-parallel.md",
    "04-training-systems/sequence-context-parallel.md",
    "04-training-systems/pipeline-parallel.md",
    "04-training-systems/expert-parallel-moe-training.md",
    "04-training-systems/hybrid-parallelism-composition.md",
    "04-training-systems/activation-checkpointing.md",
    "04-training-systems/mixed-precision-training.md",
    "04-training-systems/training-stability-numerical-debugging.md",
    "04-training-systems/communication-computation-overlap.md",
    "04-training-systems/flux-kernel-fusion.md",
    "04-training-systems/optimizer-scheduler-cost.md",
    "04-training-systems/parameter-efficient-finetuning-lora-qlora.md",
    "04-training-systems/post-training-workloads-sft-dpo-rlhf-grpo.md",
    "04-training-systems/muon-optimizer.md",
    "04-training-systems/evaluation-validation-checkpoint-selection.md",
    "04-training-systems/training-reproducibility-randomness-run-manifest.md",
    "04-training-systems/checkpoint-resume-fault-tolerance.md",
    "04-training-systems/training-performance-metrics-scaling.md",
    "04-training-systems/training-benchmark-profiling.md",
    "04-training-systems/deepspeed-megatron-fsdp.md",
    "05-kernels-compilers/index.md",
    "05-kernels-compilers/attention-computation-patterns.md",
    "05-kernels-compilers/triton.md",
    "05-kernels-compilers/torchinductor.md",
    "06-accelerators-architecture/index.md",
    "06-accelerators-architecture/performance-model-roofline.md",
    "06-accelerators-architecture/compute-units-simt-tensorcore.md",
    "06-accelerators-architecture/memory-hierarchy-data-reuse.md",
    "06-accelerators-architecture/precision-formats-low-bit-compute.md",
    "06-accelerators-architecture/interconnect-communication-architecture.md",
    "06-accelerators-architecture/power-thermal-reliability.md",
    "06-accelerators-architecture/accelerator-architecture-tradeoffs.md",
    "06-accelerators-architecture/workload-mapping-compiler-runtime-interface.md",
    "07-cluster-infra/index.md",
    "07-cluster-infra/ai-cluster-architecture-overview.md",
    "07-cluster-infra/scheduling-queues-resource-management.md",
    "07-cluster-infra/gpu-topology-numa-mig-mps-isolation.md",
    "07-cluster-infra/rdma-network-nccl-topology-congestion.md",
    "07-cluster-infra/storage-data-cache-checkpoint.md",
    "07-cluster-infra/environment-reproducibility-containers.md",
    "07-cluster-infra/mixed-workload-multitenancy-isolation.md",
    "07-cluster-infra/resource-utilization-fragmentation-capacity.md",
    "07-cluster-infra/node-lifecycle-health-maintenance.md",
    "08-benchmark-capacity/index.md",
    "08-benchmark-capacity/performance-analysis-benchmark-methodology.md",
    "08-benchmark-capacity/inference-capacity-modeling.md",
    "08-benchmark-capacity/training-capacity-scaling-efficiency.md",
    "08-benchmark-capacity/profiler-toolchain-bottleneck-analysis.md",
    "08-benchmark-capacity/roofline-analysis-compute-bandwidth.md",
    "08-benchmark-capacity/queueing-model-tail-latency.md",
    "08-benchmark-capacity/energy-power-thermal-benchmark.md",
    "08-benchmark-capacity/benchmark-workload-design-trace-replay.md",
    "08-benchmark-capacity/ab-testing-ablation-regression-detection.md",
    "08-benchmark-capacity/cost-model-unit-economics.md",
    "08-benchmark-capacity/benchmark-data-governance-run-records.md",
    "09-reliability-observability/index.md",
    "09-reliability-observability/observability-overview-signals.md",
    "09-reliability-observability/slo-sli-error-budget-alerting.md",
    "09-reliability-observability/ai-failure-modes-gpu-nccl-network-storage.md",
    "09-reliability-observability/incident-response-runbook-postmortem.md",
    "10-papers-cases/index.md",
    "10-papers-cases/ai-system-architecture.md",
    "10-papers-cases/adr.md",
    "10-papers-cases/failure-cases.md",
    "11-knowledge-index/index.md",
    "99-templates/knowledge-note.md",
    "99-templates/adr.md",
    "99-templates/benchmark-report.md",
]

DESCRIPTIONS = {
    "index.md": "知识库首页，说明目标、写作原则和推荐阅读路径。",
    "knowledge-map.md": "面向人和 AI 的总览知识地图，适合先建立全局结构。",
    "01-getting-started/index.md": "新读者入口，说明如何使用知识库。",
    "02-ai-workloads/index.md": "AI 计算工作负载基础目录。",
    "02-ai-workloads/ai-fundamentals.md": "AI 基础概念面向刚入门读者解释 AI、机器学习、深度学习、模型、参数、数据、样本、特征、标签、token、tokenizer、embedding、tensor、logits、概率、loss、梯度、训练、推理、评估、prompt、上下文、batch、并发、吞吐、模型知识边界、模型能力与系统能力的区别，以及这些概念如何连接到 Transformer、训练系统、推理系统、Benchmark 和 AI Infra 成本。",
    "02-ai-workloads/transformer.md": "Transformer 流程与原理面向新手解释 decoder-only 大语言模型的数据流，覆盖文本到 token、token id、embedding、位置信息、self-attention、Q/K/V、attention weights、scaled dot-product attention 直觉、causal mask、multi-head attention、MLP、残差连接、归一化、Transformer block、hidden states、LM Head、logits、训练时的 next-token prediction、推理时的自回归生成、KV Cache 直觉、encoder-only/decoder-only/encoder-decoder 区别、Transformer 的并行性和系统成本来源。",
    "02-ai-workloads/training-primer.md": "训练如何先预测、计算错误、反向传播并更新参数。",
    "02-ai-workloads/inference-primer.md": "推理如何读取 prompt，并逐 token 生成回答。",
    "02-ai-workloads/multimodal-primer.md": "多模态理解如何读懂图片、音频、视频，多模态生成如何生成新内容。",
    "03-inference-systems/index.md": "推理系统与优化主题入口。",
    "03-inference-systems/request-lifecycle.md": "推理请求从 API 接入到 Decode、流式返回和资源释放的端到端生命周期。",
    "03-inference-systems/prefill-decode.md": "Prefill 与 Decode 两阶段的计算形态、指标关系和系统瓶颈。",
    "03-inference-systems/metrics.md": "推理系统的 TTFT、TPOT、吞吐、尾延迟、资源、成本和稳定性指标体系。",
    "03-inference-systems/batching.md": "推理系统中 static、dynamic、continuous batching 的原理、收益和延迟取舍。",
    "03-inference-systems/kv-cache.md": "KV Cache 在 Decode 中复用历史上下文，并影响显存、并发、长上下文和调度。",
    "03-inference-systems/paged-attention.md": "PagedAttention 用 block table 和块式 KV Cache 管理减少显存碎片与预留浪费。",
    "03-inference-systems/prefix-cache.md": "Prefix Cache 复用相同 prompt 前缀的 KV Cache，降低重复 Prefill 和 TTFT。",
    "03-inference-systems/scheduling.md": "推理调度策略如何在队列、优先级、SLO、缓存、准入控制和公平性之间取舍。",
    "03-inference-systems/quantization.md": "量化推理用低精度权重、激活或 KV Cache 降低显存、带宽和推理成本。",
    "03-inference-systems/speculative-decoding.md": "Speculative Decoding 用 draft 和 verify 减少目标模型逐 token Decode 等待。",
    "03-inference-systems/prefill-decode-disaggregation.md": "Prefill/Decode 分离部署用独立资源池和 KV 传输减少两阶段互相干扰。",
    "03-inference-systems/moe-inference-optimization.md": "MoE 推理优化关注专家路由、dispatch/combine、负载不均、通信和专家权重放置。",
    "03-inference-systems/ep-size-large-small-ep.md": "EP Size 与大 EP / 小 EP 说明 MoE 专家并行大小如何影响每卡专家显存、all-to-all 通信范围、DP 副本、Prefill/Decode 和尾延迟。",
    "03-inference-systems/single-node-serving-architecture.md": "单机推理服务架构梳理 API、tokenizer、scheduler、GPU executor、KV Cache、streaming 和可观测性。",
    "03-inference-systems/distributed-inference.md": "多机分布式推理关注 data/tensor/pipeline/expert parallel、通信、路由、KV Cache 状态和故障恢复。",
    "03-inference-systems/cache-system.md": "缓存体系统一梳理 query、embedding、retrieval、prefix、KV、tool result、response 和 model artifact cache。",
    "03-inference-systems/benchmark-methodology.md": "Benchmark 方法说明如何设计可复现的推理压测 workload、指标、SLO 和实验记录。",
    "03-inference-systems/vllm.md": "vLLM 作为现代 LLM serving 引擎案例，串联 PagedAttention、continuous batching、KV Cache、OpenAI-compatible serving 和性能调优。",
    "03-inference-systems/tensorrt-llm.md": "TensorRT-LLM 作为 NVIDIA GPU 高性能推理栈案例，串联 engine/runtime、in-flight batching、paged KV cache、量化、kernel 优化和多 GPU 并行。",
    "03-inference-systems/sglang.md": "SGLang 作为结构化生成和高性能 runtime 案例，串联 RadixAttention、prefix reuse、structured outputs、RAG/Agent 和多模态 serving。",
    "03-inference-systems/rag-agent-workloads.md": "RAG 与 Agent 作为复合推理 workload，拆解检索、rerank、context packing、工具调用、多轮 LLM 调用、缓存、容量、可靠性和端到端 benchmark。",
    "03-inference-systems/benchmark-profiling.md": "Benchmark 方法与性能剖析连接 benchmark 现象、trace、profiling、队列分析、GPU timeline、KV Cache、网络瓶颈和容量模型。",
    "04-training-systems/index.md": "训练系统与优化主题入口，覆盖训练生命周期、数据混合采样、大词表输出层/loss、backward/autograd、collective 通信原语、并行策略、通信重叠、FLUX、参数高效微调、后训练工作负载、Muon 优化器、evaluation/checkpoint selection、训练可复现性/run manifest、checkpoint 和训练 benchmark。",
    "04-training-systems/training-lifecycle.md": "训练任务生命周期从数据读取、forward、loss、backward、gradient sync、optimizer step 到 checkpoint 建立训练系统端到端视角。",
    "04-training-systems/data-pipeline.md": "数据输入与 Data Pipeline 解释训练数据从存储到 GPU 的链路、DataLoader、tokenization、packing、H2D copy、有效 token 和数据瓶颈定位。",
    "04-training-systems/training-data-mixing-sampling-effective-tokens.md": "训练数据混合、采样与有效 Token 解释多数据源 mixture ratio、sample/token/loss-token 口径、weighted sampling、curriculum、distributed sampler、shuffle buffer、packing、loss mask、resume 数据状态、world size 变化、多模态采样、raw/input/loss tokens/s 和数据侧 benchmark manifest。",
    "04-training-systems/batch-gradient-accumulation.md": "Batch、Micro-batch 与 Gradient Accumulation 解释 global batch 公式、显存与吞吐关系、梯度累积流程、DDP 同步、loss 归一化和 benchmark 可比性。",
    "04-training-systems/vocab-output-cross-entropy.md": "大词表输出层、Logits 与 Cross Entropy 系统优化解释 LM head、[tokens, vocab] logits、shift labels、loss mask、ignore_index、fused CE、chunked loss、vocab parallel CE、数值稳定、后训练 logprob 和 benchmark 方法。",
    "04-training-systems/memory-composition-optimization.md": "显存组成与优化总览拆解 parameters、gradients、optimizer states、master weights、activations、temporary buffers、allocator overhead，并说明不同显存优化技术节省哪类对象。",
    "04-training-systems/backward-autograd-gradient-lifecycle.md": "Backward、Autograd Graph 与梯度生命周期解释 autograd graph、saved tensors、leaf parameter、param.grad、gradient accumulation、zero_grad、retain_graph、backward 显存生命周期、DDP/FSDP/ZeRO/TP/PP/MoE backward 通信、AMP unscale、gradient clipping 和 benchmark 方法。",
    "04-training-systems/distributed-training-runtime.md": "分布式训练启动与运行时解释 torchrun、launcher、rank、world size、local rank、node rank、rendezvous、init_process_group、process group、NCCL backend、CUDA_VISIBLE_DEVICES、分布式 sampler、rank-aware logging、checkpoint 保存、Slurm/Kubernetes 启动、DeepSpeed/Megatron/FSDP runtime 关系和常见启动/通信故障排查。",
    "04-training-systems/collective-communication-primitives.md": "Collective 通信原语与通信量模型解释 AllReduce、ReduceScatter、AllGather、Broadcast、Reduce、AllToAll、Barrier、P2P、process group、通信量 latency/bandwidth 模型、ring/tree/hierarchical 直觉、async collective、collective ordering、shape/dtype 一致性、通信 buffer、benchmark、profiler 和 hang 排查。",
    "04-training-systems/data-parallel-gradient-sync.md": "Data Parallel 与梯度同步解释 DDP、AllReduce、ReduceScatter、gradient bucket、backward overlap、gradient accumulation 同步时机、多机通信瓶颈和排查方法。",
    "04-training-systems/zero-fsdp.md": "ZeRO 与 FSDP 解释 sharded data parallel 如何切分 parameters、gradients、optimizer states，以及 ZeRO-1/2/3、FSDP all-gather/reduce-scatter、wrap 粒度、offload、checkpoint 和调优取舍。",
    "04-training-systems/tensor-parallel.md": "Tensor Parallel 解释如何把 Transformer 层内矩阵、MLP、Attention head 和词表输出切到多个 GPU，以及 Column/Row Parallel、AllReduce/AllGather/ReduceScatter、TP group、跨节点通信和性能排查。",
    "04-training-systems/sequence-context-parallel.md": "Sequence Parallel 与 Context Parallel 解释长序列训练中如何把 token sequence 或 full context 切到多个 GPU，区分 SP 和 CP 的目标、通信、attention 正确性、position id、RoPE、packing、TP/PP/FSDP/EP 组合、rank mapping、benchmark、常见优化方向和排查清单。",
    "04-training-systems/pipeline-parallel.md": "Pipeline Parallel 解释如何把模型按层切成 pipeline stages，并用 micro-batch、GPipe、1F1B、interleaving、stage balance 和 rank mapping 降低显存压力与 pipeline bubble。",
    "04-training-systems/expert-parallel-moe-training.md": "Expert Parallel 与 MoE 训练解释 router、top-k routing、token dispatch/combine、AllToAll、capacity factor、token dropping、load balance loss、EP size 和 MoE benchmark。",
    "04-training-systems/hybrid-parallelism-composition.md": "并行策略组合解释 3D/4D/5D Parallelism 如何把 DP/FSDP、TP、PP、EP、Sequence/Context Parallel、activation checkpointing、mixed precision、optimizer、checkpoint 和 rank mapping 放到同一个训练系统设计中，并根据显存、计算、通信、global batch、pipeline bubble、AllToAll、长上下文和硬件拓扑选择组合。",
    "04-training-systems/activation-checkpointing.md": "Activation Checkpointing 解释 activation memory 为什么在训练中形成峰值，如何用 forward 重算换显存，覆盖 checkpoint 数据流、显存/计算交换、block/segment/submodule/selective 粒度、PyTorch reentrant 与 non-reentrant、RNG/dropout 正确性、纯函数约束、长上下文、micro-batch、FSDP/ZeRO、TP、PP、SP/CP、MoE、mixed precision/FP8、torch.compile/min-cut、实现方式、benchmark、profiler、优化方向和落地检查表。",
    "04-training-systems/mixed-precision-training.md": "混合精度训练解释 FP32、TF32、FP16、BF16、FP8 在训练系统中的 compute/storage/communication 角色，覆盖 AMP autocast、GradScaler、loss scaling、dynamic loss scaling、master weights、敏感算子高精度保留、FP8 E4M3/E5M2、scale/amax/delayed scaling/block scaling、transpose handling、FSDP/ZeRO、TP、PP、MoE、activation checkpointing、checkpoint/resume state、NaN/Inf 排查、benchmark、profiler、优化方向和落地检查表。",
    "04-training-systems/training-stability-numerical-debugging.md": "训练稳定性与数值异常解释如何把 NaN、Inf、loss spike、gradient norm、parameter/update norm、activation statistics、AMP loss scale、FP8 amax、bad batch、distributed rank 异常、MoE router、长上下文、kernel fusion、checkpoint rollback、latest good checkpoint、stability guardrail 和 benchmark stability metrics 纳入训练系统治理。",
    "04-training-systems/communication-computation-overlap.md": "通信与计算重叠解释 backward bucket overlap、DDP AllReduce、FSDP/ZeRO all-gather/reduce-scatter、TP/PP/MoE 通信、async collective 限制、profiler timeline 和 exposed communication time。",
    "04-training-systems/flux-kernel-fusion.md": "FLUX 通信重叠与 Kernel Fusion 解释依赖通信为何难以用普通 stream overlap 隐藏，并从 over-decomposition、tile/chunk scheduling、kernel fusion、dense MLP 的 AllGather+GEMM 与 GEMM+ReduceScatter、MoE dispatch/grouped GEMM/combine、NCCL/NVSHMEM/CUTLASS、Ampere/Hopper 差异、TP/EP 场景、数值正确性、显存 buffer、benchmark 和落地检查表理解 kernel-level communication-computation overlap。",
    "04-training-systems/optimizer-scheduler-cost.md": "Optimizer 与 Scheduler 系统成本解释 Adam/AdamW 的参数、梯度、FP32 master weight、m/v optimizer state 显存和 memory-bound step 开销，覆盖 parameter groups、weight decay/no-decay、foreach/fused/capturable optimizer、gradient clipping、grad norm health metrics、micro-step/optimizer-step/token-based scheduler 语义、skipped step、ZeRO/FSDP optimizer state sharding、CPU/NVMe offload、低精度 optimizer state、CUDA graph/compile、zero_grad(set_to_none)、checkpoint/resume、benchmark、profiler、优化方向和落地检查表。",
    "04-training-systems/parameter-efficient-finetuning-lora-qlora.md": "参数高效微调解释 LoRA、QLoRA 与 Adapter 如何通过冻结基础模型、训练低秩 adapter 降低 gradients、optimizer states、通信量和 checkpoint 成本，覆盖 PEFT 作业生命周期、LoRA 参数量/显存估算、初始化与缩放、QLoRA/NF4/double quantization/paged optimizer、DP/FSDP/ZeRO/TP 选择、FSDP-QLoRA 约束、adapter-only 与可恢复 checkpoint、artifact registry、serving cache key、多 adapter 推理、benchmark、profiler 和优化方向。",
    "04-training-systems/post-training-workloads-sft-dpo-rlhf-grpo.md": "后训练工作负载解释 SFT、reward model、DPO、RLHF/PPO 与 GRPO 如何形成不同训练系统数据流，覆盖离线后训练与在线采样、chat template/loss mask、packing 与有效 token、chosen/rejected 偏好数据、reference logprob 计算与缓存、rollout/update 版本一致性、GRPO group batch 与 verifier、后训练平台资源池、rollout/training 分离部署、权重同步、artifact/checkpoint、LoRA/QLoRA 组合、benchmark、队列指标和常见优化方向。",
    "04-training-systems/muon-optimizer.md": "Muon 优化器解释 hidden layer 二维矩阵参数的矩阵动量正交化，覆盖 Muon 在训练 step 中的位置、PyTorch Muon 参数、Newton-Schulz 迭代与 ns_steps 成本、QKV/MLP/MoE 参数分组、update scale 与 weight decay、混合 Muon/AdamW、learning rate/scheduler、RMS 监控、ZeRO/FSDP/TP 的 full-matrix 与 shard-local 语义、MoE expert 状态、checkpoint/resume、benchmark、运行时告警、编译/CUDA Graph 和优化方向。",
    "04-training-systems/evaluation-validation-checkpoint-selection.md": "Evaluation、Validation 与 Checkpoint Selection 解释训练系统中的评估控制面和数据面，覆盖 validation loss、指标分层、eval dataset 版本/污染治理、样本级结果、token-based eval cadence、同步/异步 eval、eval scheduler/backlog、资源池容量、loss/generation/judge eval、pass@k、decoding 与 determinism、checkpoint selection policy、Pareto frontier、stop/pause/rollback、分布式 eval 加权汇总、eval cost model、artifact/report 模板、regression detection、eval overhead 和常见故障排查。",
    "04-training-systems/training-reproducibility-randomness-run-manifest.md": "训练可复现性、随机性与 Run Manifest 解释训练系统如何从 seed 走向可审计复现，覆盖 bitwise/step-level/statistical/auditable reproducibility、复现目标协议、实验比较协议、Run Manifest 生命周期/schema/event log、code/config/tokenizer/data manifest、streaming data cursor、packing state、RNG state inventory、随机流命名、DataLoader worker seed、rank seed policy、deterministic algorithms、determinism budget、floating point nondeterminism、mixed precision/TF32/FP8、环境锁定、activation checkpointing、MoE、rank mapping、通信拓扑 manifest、checkpoint/resume、eval/benchmark 可复现性、manifest quality gate 和自动可复现性评级。",
    "04-training-systems/checkpoint-resume-fault-tolerance.md": "Checkpoint、Resume 与容错解释训练 checkpoint 作为恢复协议的系统设计，覆盖完整训练状态、状态分层、manifest、two-phase save、committed/latest 语义、sharded checkpoint、coordinator/rank-local metadata、resharding、RPO/RTO、同步/异步保存、backpressure、elastic restart、rank 不稳定、故障模型、分层存储、容量模型、保留/删除引用检查、resume sanity protocol、损坏校验、preemption-aware checkpoint、恢复演练和 benchmark 指标。",
    "04-training-systems/training-performance-metrics-scaling.md": "训练性能指标与扩展效率解释如何用固定 workload、warmup/steady/end-to-end 测量窗口、step time breakdown、samples/s、tokens/s、instantaneous/sustained/effective throughput、goodput、tokens/day、time to target quality、quality-normalized throughput、MFU/HFU、FLOPs 分子分母纪律、dense/MoE FLOPs 估算、strong/weak scaling、scaling 实验矩阵、并行策略影响、显存 headroom、通信 exposed time 与有效带宽、synthetic/real data 差异、optimizer/checkpoint overhead、stability metrics、straggler/tail metrics、cost efficiency、性能回归门槛和报告模板评价训练系统。",
    "04-training-systems/training-benchmark-profiling.md": "训练性能剖析与 Benchmark 解释如何用问题驱动方式设计可复现训练 benchmark，覆盖 micro/component/end-to-end benchmark、torch.utils.benchmark、固定 workload、测量窗口、随机性、环境记录、benchmark manifest、重复测量与噪声、synthetic/real data、工具选择矩阵、PyTorch Profiler、Nsight Systems、Nsight Compute、NVTX 命名、多 rank trace、step time breakdown、通信 exposed time 与归因、data pipeline 对照实验、memory/optimizer/checkpoint profiling、straggler 定位、paired A/B、profiling overhead、trace 资产治理、benchmark 报告模板、性能回归处理流程和常见误区。",
    "04-training-systems/deepspeed-megatron-fsdp.md": "DeepSpeed、Megatron-LM 与 PyTorch FSDP 从训练系统职责拆分出发比较 runtime、model definition、parallel layout、memory sharding、optimizer、kernel/fusion、checkpoint/resume 和 observability，覆盖 DeepSpeed ZeRO/offload/runtime 配置、ZeRO stage、DeepSpeed 适用边界，Megatron Core 的 TP/PP/EP/CP、Transformer/MoE model-parallel training stack、distributed optimizer/FSDP/dist checkpointing，PyTorch FSDP 的 wrap policy、sharding strategy、PyTorch 原生生态、DTensor/Tensor Parallel/DeviceMesh/DCP、FSDP checkpoint 语义、parallel group ownership、框架组合方式、迁移成本、框架选型矩阵、benchmark 失败路径、选型报告模板、常见误区和设计检查清单。",
    "05-kernels-compilers/index.md": "Kernel、算子与编译优化主题入口，覆盖算子语义、手写 Kernel、图编译与代码生成、性能诊断、Attention、Triton 和 TorchInductor。",
    "05-kernels-compilers/attention-computation-patterns.md": "Attention 机制与计算模式区分 attention pattern、exact/approx、kernel 实现和状态管理四个维度，解释 Q/K/V shape、dense/causal/cross attention、mask 类型、Dense Attention 的 O(n^2) 成本和中间矩阵、Sparse Attention pattern 与 block sparse kernel 难点、MHA/MQA/GQA、FlashAttention 的 IO-aware tiling、online softmax、FlashAttention-2 work partition、PagedAttention block table、训练 forward/backward、推理 Prefill/Decode、KV Cache、长上下文、TP/SP/CP 并行、attention kernel 输入约束、Dense/Sparse/Flash/Paged 使用场景、性能分析维度、attention benchmark 方法和常见优化决策。",
    "05-kernels-compilers/triton.md": "Triton Kernel 编程解释 blocked program 执行模型、JIT 生命周期、launch grid、program_id、block tensor、pointer arithmetic、mask、shape specialization、AI workload 模式、tiling、资源模型、数值与 dtype、fused softmax、matmul、autotune、PyTorch/Inductor 集成、debugging、benchmark、端到端验证和 profiler 方法。",
    "05-kernels-compilers/torchinductor.md": "TorchInductor 与 PyTorch 编译栈解释 torch.compile、编译区域、TorchDynamo、FX/ATen 图、Fake Tensor、AOTAutograd、TorchInductor lowering/fusion/scheduler/codegen、graph break、guard、recompile、dynamic shape、CUDA Graph、compile mode、Generated Code、AOTInductor、生产上线策略、tlparse、TORCH_LOGS、torch.compiler API、profiler 和消融排查方法。",
    "06-accelerators-architecture/index.md": "AI 加速器与计算架构主题入口，关注 GPU、NPU、TPU、ASIC、FPGA 的计算、存储、互连、能效和可编程性。",
    "06-accelerators-architecture/performance-model-roofline.md": "AI 加速器性能模型用 arithmetic intensity、ridge point、多重 Roofline、compute/memory/network/energy roof、FLOPs/bytes 口径、compute-bound、memory-bound、launch-bound、HBM、片上存储、矩阵单元、低精度、Prefill/Decode、训练、MoE、互连、能效、硬件选型和 benchmark 分析硬件真实性能上限。",
    "06-accelerators-architecture/compute-units-simt-tensorcore.md": "计算单元：SIMT、Tensor Core 与矩阵引擎解释 SIMD/SIMT、masked execution、warp、warp scheduler、SM、occupancy、register/shared memory、Tensor Core tile/fragment/accumulator、Matrix Core、systolic array、NPU/ASIC 取舍、vector/scalar/load-store/SFU、稀疏、不规则访存、动态控制流、Transformer/MoE/Prefill/Decode 映射、并行切分、kernel fusion、profiler 指标和真实矩阵单元利用率。",
    "06-accelerators-architecture/memory-hierarchy-data-reuse.md": "存储层次：HBM、SRAM、Cache 与数据复用解释 register、register spill、SRAM/shared memory、bank conflict、async copy、L1/L2 cache locality、HBM 容量/带宽预算、host memory、pinned memory、UVM/page fault、offload、remote storage、数据对象生命周期、temporary buffer、FlashAttention/online softmax、KV Cache layout/量化、fusion、coalescing/stride/alignment、MoE token grouping、activation checkpointing、ZeRO/FSDP、memory planning、profiler 证据和 memory-bound benchmark。",
    "06-accelerators-architecture/precision-formats-low-bit-compute.md": "精度格式：FP16、BF16、FP8 与量化计算解释 FP32/TF32/FP16/BF16、FP8 E4M3/E5M2、INT8、INT4/NF4/FP4、accumulator、master weight、scale/amax、outlier、PTQ/QAT/weight-only/W8A8/KV Cache 量化、scale metadata、低精度通信、checkpoint metadata、硬件路径验证和 benchmark。",
    "06-accelerators-architecture/interconnect-communication-architecture.md": "互连与通信架构解释数据对象、通信频率、带宽/延迟模型、PCIe、NVLink/NVSwitch、GPU Direct RDMA、RDMA/InfiniBand/RoCE、CXL、NoC、chiplet interconnect、多 rail、collective 算法、NCCL/RCCL/MPI、rank mapping、拓扑感知并行、通信重叠、network benchmark、故障排查和互连可观测性。",
    "06-accelerators-architecture/power-thermal-reliability.md": "功耗、散热、频率与可靠性解释从单卡到整柜的功耗/散热约束链、power limit、power capping、clock、throttling、遥测指标、热稳态实验、训练/推理能效、tokens/s/W、joules/token、ECC、RAS、HBM 错误、SDC、稳态 benchmark、故障诊断、调度治理和 power-aware scheduling。",
    "06-accelerators-architecture/accelerator-architecture-tradeoffs.md": "架构取舍解释 GPU、TPU、NPU、AI ASIC、FPGA 的通用性/专用性、执行模型、存储层次、精度支持、软件栈成熟度、算子覆盖、编译器/runtime、动态性、扩展方式、迁移成本、锁定风险、训练/推理分开评估、TCO、自研 ASIC 判断、混合架构、benchmark 报告和 workload 匹配。",
    "06-accelerators-architecture/workload-mapping-compiler-runtime-interface.md": "Workload Mapping 解释模型、IR、算子、kernel、layout、tiling、fusion、compiler lowering、autotuning、guard/recompile、memory planning、runtime interface、training/inference mapping、dynamic shape、fallback、parallel mapping、profiler 证据、mapping manifest、correctness gate 和硬件有效吞吐之间的关系。",
    "07-cluster-infra/index.md": "集群、网络、存储与调度主题入口。",
    "07-cluster-infra/ai-cluster-architecture-overview.md": "AI 集群架构总览解释 workload、AI Job 生命周期、resource flavor、容量池、故障域、网络平面、存储层次、admission/placement/orchestration、gang scheduling、拓扑感知、资源碎片、多租户、manifest、可观测性和容量规划闭环。",
    "07-cluster-infra/scheduling-queues-resource-management.md": "调度系统与资源队列解释 Slurm、Kubernetes、Ray、Volcano、Kueue 在 AI workload 的 job spec、admission、queue、priority、quota、fairshare、resource flavor、capacity pool、backfill、preemption、gang scheduling、topology-aware scheduling、fragmentation、pending reason、抢占恢复协议和策略迭代中的作用。",
    "07-cluster-infra/gpu-topology-numa-mig-mps-isolation.md": "GPU 拓扑、NUMA、MIG/MPS 与资源隔离解释 GPU-to-GPU、GPU-to-NIC、CPU NUMA、本地 NVMe、nvidia-smi topo、topology manifest、MIG 生命周期和碎片、MPS、time slicing、共享 GPU 治理、Kubernetes device plugin/GPU Operator/Topology Manager、Slurm GRES、rank mapping、可观测性和故障归因。",
    "07-cluster-infra/rdma-network-nccl-topology-congestion.md": "RDMA 网络与 NCCL 拓扑解释 InfiniBand、RoCE、GPU Direct RDMA 路径验收、NCCL/RCCL collective、通信模式到网络需求映射、multi-rail、bisection bandwidth、network topology manifest、PFC、ECN、QoS、RoCE 配置漂移、拥塞域、网络调度契约、网络可观测性、故障归因和 benchmark manifest。",
    "07-cluster-infra/storage-data-cache-checkpoint.md": "存储、数据缓存与 Checkpoint 解释对象存储、并行文件系统、本地 NVMe、数据对象生命周期、存储策略契约、DataLoader、dataset shard、cache key、缓存治理、checkpoint 状态机、checkpoint manifest、异步保存、恢复与 resharding、模型权重分发防雪崩、容器镜像、GPUDirect Storage、Kubernetes PV/PVC/StorageClass、存储调度契约、可观测性、故障归因和 benchmark manifest。",
    "07-cluster-infra/environment-reproducibility-containers.md": "环境可复现解释 AI 任务的 node profile、image family、run manifest、Driver/CUDA 支持矩阵、镜像供应链与 SBOM、image digest、Python/Conda lock、artifact manifest、随机性控制、benchmark 可复现、编译缓存、环境升级回滚和环境漂移归因。",
    "07-cluster-infra/mixed-workload-multitenancy-isolation.md": "混合集群与多租户隔离解释训练、在线推理、Notebook、离线批处理、数据预处理、benchmark、系统任务共存时的 workload contract、tenant 映射、单/多/虚拟集群取舍、隔离等级、queue、quota、priority、preemption 生命周期、node pool、GPU 共享、存储网络隔离、noisy neighbor 归因、准入控制、资源碎片和成本归因。",
    "07-cluster-infra/resource-utilization-fragmentation-capacity.md": "资源利用率、碎片与容量治理解释 GPU 分配率、驻留率、活跃率、有效吞吐、capacity/demand/usage/output 四本账、GPU hour 浪费分类、pending reason、队列健康度、fragmentation、largest contiguous allocation、碎片治理、request right-sizing、公平性、SLA、成本归因、showback/chargeback、能效、DCGM、Kueue/Slurm 指标、指标数据质量、dashboard、告警、治理闭环和容量规划。",
    "07-cluster-infra/node-lifecycle-health-maintenance.md": "节点生命周期与集群运维解释 AI 计算节点从 Node Manifest、资产登记、物理验收、基线即代码、burn-in 验收报告、入池门禁、Kubernetes/Slurm 状态、健康分级、cordon/drain、维护窗口、批次升级与回滚、配置漂移处理、自动扩缩容、RMA、缓存清理、安全擦除、运维证据链到退役下线的流程。",
    "08-benchmark-capacity/index.md": "性能分析、Benchmark 与容量建模主题入口。",
    "08-benchmark-capacity/performance-analysis-benchmark-methodology.md": "性能分析与 Benchmark 方法论解释 benchmark、profiling、monitoring 的区别，如何定义决策问题、Benchmark Contract、测量边界、latency/throughput/efficiency/cost/energy 指标、设计 workload、open-loop/closed-loop 负载、实验矩阵、记录环境、处理 warmup、做统计判定、A/A 噪声基线、A/B、ablation、profiler 证据、噪声控制、瓶颈定位、raw data lineage、发布门禁、回归检测和容量建模输入。",
    "08-benchmark-capacity/inference-capacity-modeling.md": "推理容量建模解释如何用 Capacity Contract、QPS、concurrency、TTFT、TPOT、E2E latency、offered load/throughput/goodput、输入/输出长度分布、请求分桶、单副本 goodput 曲线、latency/error 拐点、prefill/decode、KV Cache token occupancy 和预算、batching、headroom、failure domain、rolling update、autoscaling 滞后、冷启动、routing efficiency、多模型、过载保护、成本和生产校准推导 GPU 副本数。",
    "08-benchmark-capacity/training-capacity-scaling-efficiency.md": "训练容量建模解释如何用 Capacity Contract、total tokens、raw/processed/loss tokens、global batch、micro batch、gradient accumulation、step time、sustained/effective throughput、goodput、tokens/s/GPU、MFU、strong/weak scaling、扩展曲线拐点、边际 GPU 收益、DP/TP/PP/EP/FSDP/ZeRO、checkpoint interval、eval、failure/restart、queue wait、资源准入、headroom、成本、能效、容量报告模板和生产校准推导训练总时间与 GPU 需求。",
    "08-benchmark-capacity/profiler-toolchain-bottleneck-analysis.md": "Profiler 工具链与瓶颈定位解释如何用 Profiling Contract、应用指标、证据链分层、工具选择矩阵、PyTorch Profiler、Nsight Systems、Nsight Compute、DCGM、perf/eBPF、NVTX 标注、时间对齐、run/request/step/rank/node id、分布式 trace、kernel metrics、CPU/网络/存储信号、常见瓶颈模式、结论分级、报告模板、排除项、前后对比 benchmark 和回归防线建立 AI 系统性能瓶颈证据链。",
    "08-benchmark-capacity/roofline-analysis-compute-bandwidth.md": "Roofline 分析解释如何用 Roofline Analysis Contract、分析对象层级、FLOPs/bytes 口径、useful/executed FLOPs、arithmetic intensity、ridge point、theoretical/sustained compute roof、measured memory roof、dtype roof、Nsight/profiler 指标、achieved FLOP/s、achieved bandwidth、bound efficiency、端到端收益上限、多重 roof、communication/launch roofline、GEMM、KV Cache、fusion、低精度、MoE、sparse attention、推理 prefill/decode、训练 forward/backward/optimizer、硬件评估和 before/after benchmark 判断瓶颈上限与优化方向。",
    "08-benchmark-capacity/queueing-model-tail-latency.md": "排队模型与尾延迟解释如何用 Queueing Contract、offered load、throughput、goodput at SLA、Little's Law、concurrency、service time、utilization、queueing latency、deadline miss、Kingman/VUT 直觉、多阶段队列、Prefill/Decode 资源冲突、KV Cache occupancy、batching、head-of-line blocking、长度分桶、deadline-aware scheduling、priority class、retry amplification、load shedding、retry budget、open-loop benchmark、coordinated omission、阶梯压测、goodput 曲线、队列上限、请求长度分布、分桶指标、target utilization 和故障场景验证分析 AI 推理服务的 p95/p99 与容量边界。",
    "08-benchmark-capacity/energy-power-thermal-benchmark.md": "能效、功耗与热限制解释如何用 Energy Benchmark Contract 定义 GPU-only、node-level、rack/facility-level 测量边界、idle baseline、telemetry 和热稳态窗口，采集 power draw、energy、clocks、temperature、throttle reason、DCGM/NVML/nvidia-smi telemetry，并用 power cap sweep、Pareto frontier、prefill/decode 分阶段能耗、cache hit/miss、joules/token、tokens/joule、energy to target、goodput at SLA、thermal headroom 和 power-aware scheduling 分析 AI 训练与推理系统能效。",
    "08-benchmark-capacity/benchmark-workload-design-trace-replay.md": "Benchmark 负载设计与 Trace Replay 解释如何用 Workload Benchmark Contract 把决策问题转成 workload spec，设计 synthetic workload、sampled workload、trace replay、input/output token 分布、request mix、arrival process、open-loop/closed-loop、coordinated omission 防护、trace 时间语义、cache 状态、RAG/Agent 与多模态负载、训练数据路径、集群 job trace、warmup、measurement window、load generator 校验和 benchmark 报告 caveats。",
    "08-benchmark-capacity/ab-testing-ablation-regression-detection.md": "A/B 对比、消融实验与性能回归检测解释如何用 Experiment Contract、A/A test、noise floor、A/B comparison、ablation matrix、primary metrics、guardrail metrics、explanatory metrics、paired runs、paired difference、effect size、practical threshold、baseline lifecycle、rerun policy、CI smoke benchmark、nightly benchmark、release gate、production canary、shadow traffic、regression rule template 和 profiler evidence 防止 AI 系统性能回归。",
    "08-benchmark-capacity/cost-model-unit-economics.md": "成本模型与单位经济性解释如何用 Cost Model Contract 定义 GPU-only、node-level、service-level、workflow-level、cluster-level 成本边界，建立 rate card、摊销、边际成本、effective output、cost/request、cost/input token、cost/output token、cost/training token、cost/successful run、cost to target quality、RAG/Agent workflow 成本、训练有效 token、checkpoint/eval/failure 成本、headroom、缓存 ROI、共享成本归因、showback/chargeback、浪费分类、dashboard 指标、多租户治理和研发速度成本。",
    "08-benchmark-capacity/benchmark-data-governance-run-records.md": "Benchmark 数据治理与实验记录解释如何用 benchmark data contract、数据模型、run manifest、run id、raw per-request/per-step/system metrics、summary schema、schema versioning、environment metadata、artifact digest、lineage/provenance、timestamp alignment、privacy redaction、retention、baseline registry、data quality gates、dashboard/report separation、CI regression records 和 AI-readable run card 让 AI benchmark 结论可复现、可审查、可比较、可回归检测和可检索。",
    "09-reliability-observability/index.md": "可靠性、可观测性与故障复盘主题入口。",
    "09-reliability-observability/observability-overview-signals.md": "AI 系统可观测性总览解释 metrics、logs、traces、profiles、events 五类 telemetry signals 在 AI 推理、训练和集群系统中的分工，覆盖 monitoring/observability/debugging 区别、latency/traffic/errors/saturation 黄金信号、分层观测、black-box/white-box、关联字段、cardinality、指标命名、SLO/burn-rate alerting、dashboard 组织、sampling/retention、benchmark 联动和最小可观测性闭环。",
    "09-reliability-observability/slo-sli-error-budget-alerting.md": "SLO、SLI、错误预算与告警策略解释如何为 AI 推理、训练和集群平台定义 SLI、SLO、SLA、error budget、burn rate 与 alert policy，覆盖 SLO contract、用户旅程、推理 availability/TTFT/TPOT/goodput/degraded mode、训练 job start/progress/checkpoint freshness/recovery、集群 GPU readiness/scheduling/storage、good/bad/total events、occurrence/time-slice 预算、多窗口 burn-rate 告警、告警分级、发布治理、容量规划、benchmark 联动和常见误区。",
    "09-reliability-observability/ai-failure-modes-gpu-nccl-network-storage.md": "AI 系统故障模式解释如何从 symptom、scope、timeline、signals、layer、evidence pack 和 action 定位 AI 推理、训练与集群故障，覆盖 GPU Xid/ECC/HBM/OOM/thermal/power throttling、DCGM diagnostics、NCCL timeout/hang/debug logs/topology、RDMA/RoCE/NIC/switch 网络问题、checkpoint/dataset/model load 存储问题、runtime scheduler/cache/tokenizer/DataLoader 状态机、placement 和环境漂移，并沉淀 runbook、自动隔离、SLO 关联和 benchmark 健康检查。",
    "09-reliability-observability/incident-response-runbook-postmortem.md": "Incident Response、Runbook 与故障复盘解释 AI 系统事故如何从 detect、declare、stabilize、diagnose、mitigate、recover、postmortem 到 improve 形成闭环，覆盖 severity、incident commander、ops/comms/scribe 角色、live incident state document、AI 推理/训练/集群事故类型、stop-the-bleeding mitigation、runbook 设计、handoff、postmortem 模板、contributing factors、blameless accountability、action item verification、事故到 benchmark/alert/runbook/架构改进、incident knowledge card 和事故数据库。",
    "10-papers-cases/index.md": "论文复现与系统案例主题入口。",
    "10-papers-cases/ai-system-architecture.md": "AI 系统论文与架构解释如何把一篇 AI Systems / AI Infra 论文拆成 workload、系统瓶颈、核心机制、data plane、control plane、状态对象、成本模型、正确性边界、evaluation 证据、复现等级、reproduction contract、baseline、公平 benchmark、sanity check、sweep、ablation、raw data、paper card、mechanism card、experiment card、ADR 输入和 AI-readable 知识资产。",
    "10-papers-cases/adr.md": "技术决策记录解释 AI Infra 中如何用 ADR 保存关键技术选择的工程判断，覆盖 decision question、scope、workload contract、decision drivers、guardrail metrics、候选方案、证据等级、benchmark/profiler/trace/incident evidence、decision status、rollout、rollback、confirmation、revisit condition、evidence pack、decision readiness/done、ADR review、与 benchmark/论文复现/SLO/error budget/成本模型的关系、AI-readable ADR card、文件组织、常见误区和检查清单。",
    "10-papers-cases/failure-cases.md": "故障案例库解释如何把 AI Infra 的线上事故、压测失败、论文复现失败、性能回归、训练异常、NCCL hang、checkpoint 恢复失败、kernel/compiler 回归和成本异常沉淀为 failure case，覆盖 case/postmortem/incident 区分、案例分类法、failure case card、workload contract、symptom/evidence 分离、分层定位、contributing factors、复现等级、evidence pack、从案例到 benchmark/test/runbook/ADR/可观测性、AI-readable case card、case registry、案例评审、脱敏、常见误区和检查清单。",
    "11-knowledge-index/index.md": "知识组织、模板与 AI 可读索引解释如何把 AI Infra Markdown 知识库组织成同时适合人类学习和 AI 检索的系统，覆盖文档类型、Diataxis 视角、front matter、domain/doc_type/status/workload/system_layer/hardware/software/metrics/sources 元数据、证据等级、文档状态流转、文档关系、人类导航、knowledge-map、llms.txt、llms-full.txt、AI 引用策略、向量索引、知识图谱、source of truth、provenance、FAIR、文件命名、链接规范、模板体系、写作规范、质量门禁、维护节奏、public/internal 分层、安全隐私、AI Agent 使用协议、AI 检索失败模式和演进路线。",
    "99-templates/knowledge-note.md": "知识点模板说明如何撰写 AI Infra 普通知识点文章，覆盖适用场景、最小模板、完整推荐模板、front matter、domain/doc_type/status/workload/system_layer/metrics/sources 字段、系统对象、成本模型、指标体系、适用范围、相关技术、Benchmark 方法、证据来源、AI-readable knowledge card、写作规则、常见误区和模板质量门禁。",
    "99-templates/adr.md": "技术决策模板说明如何撰写 AI Infra ADR，覆盖使用场景、文件命名、状态流转、最小模板、完整推荐模板、decision question、context、workload contract、system scope、decision drivers、options、evidence matrix、benchmark evidence、evidence level、decision、consequences、cost model、reliability impact、implementation plan、rollout、rollback、confirmation、revisit condition、AI-readable ADR card、字段说明、写作规则、常见误区、质量门禁以及与知识点、Benchmark、Failure Case、Runbook 和 Paper Note 的关系。",
    "99-templates/benchmark-report.md": "基准实验报告模板说明如何撰写 AI Infra benchmark report，覆盖 benchmark 类型、证据等级、最小模板、完整推荐模板、benchmark question、hypothesis、validity scope、workload contract、hardware/software/runtime environment、baseline、variants、primary/guardrail/debugging metrics、experiment design、warmup、repetitions、sweep、stop conditions、commands、run manifest、raw data、summary data、results、statistical treatment、profiler and bottleneck analysis、correctness and quality guardrails、cost and energy、conclusion、artifact index、AI-readable benchmark card、字段说明、写作规则、常见误区和质量门禁。",
}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def strip_front_matter(text: str) -> str:
    return re.sub(r"\A---\n.*?\n---\n\n?", "", text, flags=re.S)


def front_matter(text: str) -> dict[str, str]:
    match = re.match(r"\A---\n(.*?)\n---\n", text, flags=re.S)
    if not match:
        return {}
    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip()
    return fields


def title_for(rel_path: str) -> str:
    text = read_text(DOCS / rel_path)
    metadata = front_matter(text)
    if metadata.get("title"):
        return metadata["title"]
    for line in strip_front_matter(text).splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return rel_path


def site_link(rel_path: str) -> str:
    if rel_path == "index.md":
        return SITE_URL
    path = rel_path.removesuffix("index.md").removesuffix(".md").rstrip("/")
    return f"{SITE_URL}{path}/"


def raw_link(rel_path: str) -> str:
    return f"{RAW_URL}/docs/{rel_path}"


def doc_paths() -> list[str]:
    available = {str(path.relative_to(DOCS)) for path in DOCS.rglob("*.md")}
    ordered = [path for path in PRIORITY_DOCS if path in available]
    ordered.extend(sorted(available - set(ordered)))
    return ordered


def build_llms_txt(paths: list[str]) -> str:
    primary = paths[:24]
    optional = paths[24:]
    lines = [
        "# AI Knowledge Graph",
        "",
        "> AI infrastructure and efficient computing knowledge base for systems-oriented graduate students, engineers, and AI-assisted retrieval.",
        "",
        "This repository is a public Markdown knowledge base rendered as a MkDocs site. Use `llms.txt` as the entry index and `llms-full.txt` when a single-file context dump is preferred.",
        "",
        "## Canonical Locations",
        "",
        f"- [Documentation site]({SITE_URL}): human-readable MkDocs site.",
        f"- [GitHub repository]({REPO_URL}): Markdown source, templates, configuration, and version history.",
        f"- [Full LLM context]({SITE_URL}llms-full.txt): aggregated Markdown source for AI ingestion.",
        f"- [Knowledge map]({SITE_URL}knowledge-map/): high-level map and navigation entry.",
        "",
        "## Recommended Reading Path",
        "",
        "1. Start with `首页` and `知识地图` to understand the scope and structure.",
        "2. Read `AI 基础概念`, `Transformer 流程与原理`, `训练过程与原理`, `推理过程与原理`, and `多模态原理` for baseline AI workload understanding.",
        "3. For efficient computing, move to `推理系统与优化`, `Kernel、算子与编译优化`, `AI 加速器与计算架构`, and `性能分析、Benchmark 与容量建模`.",
        "4. For long-term knowledge capture, use `论文复现与系统案例`, `知识组织、模板与 AI 可读索引`, and the templates.",
        "",
        "## Primary Documents",
        "",
    ]
    for rel_path in primary:
        lines.append(f"- [{title_for(rel_path)}]({site_link(rel_path)}): {DESCRIPTIONS.get(rel_path, '知识库文档。')}")
    if optional:
        lines.extend(["", "## Optional Documents", ""])
        for rel_path in optional:
            lines.append(f"- [{title_for(rel_path)}]({site_link(rel_path)}): {DESCRIPTIONS.get(rel_path, '补充文档或模板。')}")
    lines.extend(
        [
            "",
            "## Markdown Source",
            "",
            "Prefer Markdown source when exact wording, metadata, or citations are needed.",
            "",
        ]
    )
    for rel_path in paths:
        lines.append(f"- [{rel_path}]({raw_link(rel_path)})")
    lines.extend(
        [
            "",
            "## AI Usage Notes",
            "",
            "- Treat front matter fields as metadata for domain, status, owner, license, and update time.",
            "- Prefer citing source document paths when answering from this knowledge base.",
            "- Performance claims should preserve workload, batch shape, sequence length, precision, hardware, software version, and benchmark context when available.",
            "- Draft pages are useful for orientation, but should not be treated as verified conclusions unless the page status says reviewed or verified.",
            "",
        ]
    )
    return "\n".join(lines)


def build_llms_full(paths: list[str]) -> str:
    lines = [
        "# AI Knowledge Graph - Full LLM Context",
        "",
        f"Repository: {REPO_URL}",
        f"Documentation site: {SITE_URL}",
        "Generated from Markdown source files in `docs/`.",
        "",
        "Use this file as a compact ingestion target when an AI system cannot crawl the full repository. For exact source locations, each document section includes its repository path and site URL.",
        "",
        "## Document Index",
        "",
    ]
    for rel_path in paths:
        lines.append(f"- {rel_path} - {title_for(rel_path)}")
    lines.extend(["", "---", ""])
    for rel_path in paths:
        source = DOCS / rel_path
        text = read_text(source)
        metadata = front_matter(text)
        body = strip_front_matter(text).strip()
        lines.extend(
            [
                f"# Document: {title_for(rel_path)}",
                "",
                f"Source: docs/{rel_path}",
                f"URL: {site_link(rel_path)}",
            ]
        )
        for key in ["domain", "status", "owner", "license", "updated"]:
            if key in metadata:
                lines.append(f"{key}: {metadata[key]}")
        lines.extend(["", body, "", "---", ""])
    return "\n".join(lines)


def write_pair(filename: str, content: str) -> None:
    for directory in [ROOT, DOCS]:
        (directory / filename).write_text(content, encoding="utf-8")


def main() -> None:
    paths = doc_paths()
    write_pair("llms.txt", build_llms_txt(paths))
    write_pair("llms-full.txt", build_llms_full(paths))


if __name__ == "__main__":
    main()
