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
    "04-training-systems/batch-gradient-accumulation.md",
    "04-training-systems/memory-composition-optimization.md",
    "04-training-systems/data-parallel-gradient-sync.md",
    "04-training-systems/zero-fsdp.md",
    "04-training-systems/tensor-parallel.md",
    "04-training-systems/pipeline-parallel.md",
    "04-training-systems/expert-parallel-moe-training.md",
    "04-training-systems/activation-checkpointing.md",
    "04-training-systems/mixed-precision-training.md",
    "04-training-systems/communication-computation-overlap.md",
    "04-training-systems/flux-kernel-fusion.md",
    "04-training-systems/optimizer-scheduler-cost.md",
    "04-training-systems/muon-optimizer.md",
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
    "07-cluster-infra/index.md",
    "08-benchmark-capacity/index.md",
    "09-reliability-observability/index.md",
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
    "02-ai-workloads/ai-fundamentals.md": "AI 基础概念，面向刚入门读者。",
    "02-ai-workloads/transformer.md": "Transformer 如何读取上下文、更新 token 表示并预测下一个 token。",
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
    "04-training-systems/index.md": "训练系统与优化主题入口，覆盖训练生命周期、并行策略、通信重叠、FLUX、Muon 优化器、checkpoint 和训练 benchmark。",
    "04-training-systems/training-lifecycle.md": "训练任务生命周期从数据读取、forward、loss、backward、gradient sync、optimizer step 到 checkpoint 建立训练系统端到端视角。",
    "04-training-systems/data-pipeline.md": "数据输入与 Data Pipeline 解释训练数据从存储到 GPU 的链路、DataLoader、tokenization、packing、H2D copy、有效 token 和数据瓶颈定位。",
    "04-training-systems/batch-gradient-accumulation.md": "Batch、Micro-batch 与 Gradient Accumulation 解释 global batch 公式、显存与吞吐关系、梯度累积流程、DDP 同步、loss 归一化和 benchmark 可比性。",
    "04-training-systems/memory-composition-optimization.md": "显存组成与优化总览拆解 parameters、gradients、optimizer states、master weights、activations、temporary buffers、allocator overhead，并说明不同显存优化技术节省哪类对象。",
    "04-training-systems/data-parallel-gradient-sync.md": "Data Parallel 与梯度同步解释 DDP、AllReduce、ReduceScatter、gradient bucket、backward overlap、gradient accumulation 同步时机、多机通信瓶颈和排查方法。",
    "04-training-systems/zero-fsdp.md": "ZeRO 与 FSDP 解释 sharded data parallel 如何切分 parameters、gradients、optimizer states，以及 ZeRO-1/2/3、FSDP all-gather/reduce-scatter、wrap 粒度、offload、checkpoint 和调优取舍。",
    "04-training-systems/tensor-parallel.md": "Tensor Parallel 解释如何把 Transformer 层内矩阵、MLP、Attention head 和词表输出切到多个 GPU，以及 Column/Row Parallel、AllReduce/AllGather/ReduceScatter、TP group、跨节点通信和性能排查。",
    "04-training-systems/pipeline-parallel.md": "Pipeline Parallel 解释如何把模型按层切成 pipeline stages，并用 micro-batch、GPipe、1F1B、interleaving、stage balance 和 rank mapping 降低显存压力与 pipeline bubble。",
    "04-training-systems/expert-parallel-moe-training.md": "Expert Parallel 与 MoE 训练解释 router、top-k routing、token dispatch/combine、AllToAll、capacity factor、token dropping、load balance loss、EP size 和 MoE benchmark。",
    "04-training-systems/activation-checkpointing.md": "Activation Checkpointing 解释为什么训练要保存 activation，如何用重计算换显存，以及 checkpoint 粒度、selective recomputation、RNG 正确性、FSDP/TP/PP 组合和 benchmark 方法。",
    "04-training-systems/mixed-precision-training.md": "混合精度训练解释 FP16、BF16、FP8、autocast、loss scaling、master weights、敏感算子高精度保留、分布式 dtype 策略、NaN/Inf 排查和性能评估。",
    "04-training-systems/communication-computation-overlap.md": "通信与计算重叠解释 backward bucket overlap、DDP AllReduce、FSDP/ZeRO all-gather/reduce-scatter、TP/PP/MoE 通信、async collective 限制、profiler timeline 和 exposed communication time。",
    "04-training-systems/flux-kernel-fusion.md": "FLUX 通信重叠与 Kernel Fusion 解释为什么 stream-level overlap 难以隐藏依赖通信，以及过分解、chunk-level overlap、kernel fusion、TP/MoE 场景、硬件依赖和 benchmark 方法。",
    "04-training-systems/optimizer-scheduler-cost.md": "Optimizer 与 Scheduler 系统成本解释 Adam/AdamW 状态、master weights、fused/foreach optimizer、gradient clipping、scheduler step 语义、分布式 optimizer、offload、checkpoint 和恢复。",
    "04-training-systems/muon-optimizer.md": "Muon 优化器解释矩阵动量正交化、Newton-Schulz 迭代、参数分组、更新尺度、混合 Muon/AdamW、ZeRO/FSDP/TP/MoE 集成、checkpoint 和 benchmark 方法。",
    "04-training-systems/checkpoint-resume-fault-tolerance.md": "Checkpoint、Resume 与容错解释完整训练状态、sharded checkpoint、resharding、异步保存、atomic latest、elastic restart、rank 不稳定、存储设计和恢复验证。",
    "04-training-systems/training-performance-metrics-scaling.md": "训练性能指标与扩展效率解释 step time、tokens/s、time to target quality、MFU/HFU、strong/weak scaling、显存效率、通信效率、数据效率、checkpoint overhead 和成本效率。",
    "04-training-systems/training-benchmark-profiling.md": "训练性能剖析与 Benchmark 解释 micro/component/end-to-end benchmark、PyTorch Profiler、Nsight Systems、Nsight Compute、NVTX 标注、通信/数据/显存/optimizer/checkpoint 剖析和 A/B 实验方法。",
    "04-training-systems/deepspeed-megatron-fsdp.md": "DeepSpeed、Megatron-LM 与 PyTorch FSDP 对比 ZeRO、FSDP、TP、PP、EP、runtime、checkpoint、offload、PyTorch 原生集成和框架选型 benchmark。",
    "05-kernels-compilers/index.md": "Kernel、算子与编译优化主题入口。",
    "05-kernels-compilers/attention-computation-patterns.md": "Attention 机制与计算模式区分 Dense Attention、Sparse Attention、FlashAttention 和 PagedAttention，解释 attention pattern、长上下文成本、显存 IO 和 kernel 执行效率。",
    "05-kernels-compilers/triton.md": "Triton Kernel 编程解释 blocked program 执行模型、program_id、block tensor、pointer arithmetic、mask、constexpr、tiling、fused softmax、matmul、autotune、debugging、benchmark 和 profiler 方法。",
    "05-kernels-compilers/torchinductor.md": "TorchInductor 与 PyTorch 编译栈解释 torch.compile、TorchDynamo、FX Graph、AOTAutograd、TorchInductor lowering/fusion/codegen、graph break、guard、recompile、dynamic shape、tlparse、TORCH_LOGS 和 profiler 排查方法。",
    "06-accelerators-architecture/index.md": "AI 加速器与计算架构主题入口，关注 GPU、NPU、TPU、ASIC、FPGA 的计算、存储、互连、能效和可编程性。",
    "06-accelerators-architecture/performance-model-roofline.md": "AI 加速器性能模型用 arithmetic intensity、Roofline、compute-bound、memory-bound、HBM、片上存储、矩阵单元、互连和 benchmark 分析硬件真实性能上限。",
    "06-accelerators-architecture/compute-units-simt-tensorcore.md": "计算单元：SIMT、Tensor Core 与矩阵引擎解释 SIMD/SIMT、warp、SM、occupancy、Tensor Core、Matrix Core、systolic array、稀疏、不规则访存、动态控制流和真实矩阵单元利用率。",
    "06-accelerators-architecture/memory-hierarchy-data-reuse.md": "存储层次：HBM、SRAM、Cache 与数据复用解释 register、SRAM/shared memory、L1/L2 cache、HBM、host memory、offload、KV Cache、fusion、IO-aware attention 和 memory-bound benchmark。",
    "06-accelerators-architecture/precision-formats-low-bit-compute.md": "精度格式：FP16、BF16、FP8 与量化计算解释 FP32/TF32/FP16/BF16/FP8/INT8/INT4、accumulator、master weight、scale、amax、outlier、KV Cache 量化、低精度通信和硬件路径验证。",
    "06-accelerators-architecture/interconnect-communication-architecture.md": "互连与通信架构解释 PCIe、NVLink/NVSwitch、RDMA/InfiniBand/RoCE、CXL、NoC、chiplet interconnect、collective、rank mapping、拓扑感知并行、通信重叠和网络 benchmark。",
    "06-accelerators-architecture/power-thermal-reliability.md": "功耗、散热、频率与可靠性解释 power limit、thermal limit、clock、throttling、ECC、RAS、稳态 benchmark、tokens/s/W、joules/token、power-aware scheduling 和持续吞吐。",
    "07-cluster-infra/index.md": "集群、网络、存储与调度主题入口。",
    "08-benchmark-capacity/index.md": "性能分析、Benchmark 与容量建模主题入口。",
    "09-reliability-observability/index.md": "可靠性、可观测性与故障复盘主题入口。",
    "10-papers-cases/index.md": "论文复现与系统案例主题入口。",
    "10-papers-cases/ai-system-architecture.md": "AI 系统论文与架构案例索引。",
    "10-papers-cases/adr.md": "技术决策记录入口。",
    "10-papers-cases/failure-cases.md": "故障复盘入口。",
    "11-knowledge-index/index.md": "知识组织、模板、元数据和 AI 可读索引说明。",
    "99-templates/knowledge-note.md": "知识点模板。",
    "99-templates/adr.md": "技术决策模板。",
    "99-templates/benchmark-report.md": "基准实验报告模板。",
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
