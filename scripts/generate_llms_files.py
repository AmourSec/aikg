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
    "03-inference-systems/single-node-serving-architecture.md",
    "03-inference-systems/distributed-inference.md",
    "03-inference-systems/rag-agent-workloads.md",
    "04-training-systems/index.md",
    "05-kernels-compilers/index.md",
    "05-kernels-compilers/triton.md",
    "05-kernels-compilers/torchinductor.md",
    "06-accelerators-architecture/index.md",
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
    "03-inference-systems/single-node-serving-architecture.md": "单机推理服务架构梳理 API、tokenizer、scheduler、GPU executor、KV Cache、streaming 和可观测性。",
    "03-inference-systems/distributed-inference.md": "多机分布式推理关注 data/tensor/pipeline/expert parallel、通信、路由、KV Cache 状态和故障恢复。",
    "03-inference-systems/rag-agent-workloads.md": "RAG 与 Agent 推理负载的系统特征。",
    "04-training-systems/index.md": "训练系统与优化主题入口。",
    "05-kernels-compilers/index.md": "Kernel、算子与编译优化主题入口。",
    "05-kernels-compilers/triton.md": "Triton Kernel 编程学习入口。",
    "05-kernels-compilers/torchinductor.md": "TorchInductor 与 PyTorch 编译栈学习入口。",
    "06-accelerators-architecture/index.md": "AI 加速器与计算架构主题入口。",
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
