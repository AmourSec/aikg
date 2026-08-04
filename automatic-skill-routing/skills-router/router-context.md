# Skills Router Context

This file is auto-generated. Do not edit manually.
Regenerate with: python3 automatic-skill-routing/skills-router/scripts/generate_router_context.py

---

## Routing Protocol

When a user describes a task, follow these steps:

### Step 1 — RECALL
Scan the Skills Catalog below. Identify candidates whose description
relates to the user's task. Use keyword matching AND semantic judgment.
Return at most 10 candidates. Prefer over-recalling; Step 2 will filter.

### Step 2 — SELECT
For each candidate, decide if it truly fits the task. You MAY return
0, 1, or N skills. For multiple skills, assign `order` (1, 2, 3...) to
indicate usage sequence. Give a `reason` for each selected and rejected
skill. Output this JSON (do not skip):

```json
{
  "selected": [
    {"name": "<skill-name>", "order": 1, "reason": "<why>"},
    {"name": "<skill-name>", "order": 2, "reason": "<why>"}
  ],
  "rejected": [
    {"name": "<skill-name>", "reason": "<why not>"}
  ],
  "confirm_required": false,
  "confirm_reason": ""
}
```

Rules:
- `selected[].name` MUST exist in the catalog below.
- If nothing fits, return empty `selected` and say so in natural language.
- `confirm_required` = true if ANY selected skill has `confirm: true`.
- Do NOT select skills solely on keyword hits. Use semantic judgment.

### Step 3 — NOTIFY / CONFIRM
Before loading any skill, output this message:

```
准备使用以下 Skills：
- <name>：<one-line purpose>（需确认：<reason>）   ← only if confirm: true
- <name>：<one-line purpose>
```

- If `confirm_required` is false: show the message, then continue to Step 4.
- If `confirm_required` is true: show the message, then STOP and wait for
  the user to explicitly agree (e.g., "继续" / "yes"). Do NOT proceed to
  Step 4 until the user agrees.
- If the user refuses a confirmed skill, remove it from selected and
  re-evaluate whether the remaining skills can complete the task.

### Step 4 — LOAD
For each selected skill (in order), read the file at its `path` to load
the full SKILL.md content. If the skill references `references/`,
`scripts/`, or `assets/` directories, read those on demand.

Only load selected skills. Do NOT load the entire catalog's full text.

### Step 5 — EXECUTE
Follow the loaded skill's own instructions to complete the task.
If a skill requires sub-step confirmations, follow its own rules.

---

## Skills Catalog


_Total: 386 skills from 3 active sources._

### Source: ascend-agent-skills (196 skills)

- **RAGSkill**
  - path: `.skills-cache/ascend-agent-skills/official/MindSeriesSDK/RAGSDK/rag-skill/SKILL.md`
  - desc: 基于FastAPI知识库服务实现的RAG知识管理与检索技能，支持文档上传、解析、入库、删除、列表查询、内容检索、全文检索等能力，兼容PDF/DOCX/TXT/MD等格式，支持图片解析与多粒度描述入库，可直接对接Agent/机器人进行知识问答。

- **adapt-ascend-op**
  - path: `.skills-cache/ascend-agent-skills/official/vllm-ascend/adapt-ascend-op/SKILL.md`
  - desc: 将 Ascend-Kernel 单算子工程算子适配到 vllm-ascend 仓库，创建 ACLNN API 层、PyTorch 绑定和构建注册。 当用户提到"适配算子"、"接入算子"、"注册算子到 vllm-ascend"、"将算子适配 vllm-ascend"、 "添加自定义算子"、"算子集成"、"ACLNN 适配"时使用此 Skill。

- **agent-perf-analyzer**
  - path: `.skills-cache/ascend-agent-skills/official/MindSeriesSDK/AgentSDK/agent-perf-analyzer/SKILL.md`
  - desc: Analyze AgentSDK/Agentic RL codebases for precision and performance issues on Ascend NPU.
Use when users want to: diagnose training/inference precision problems (NaN, overflow, dtype mismatches) on NPU,
identify performance bottlenecks (memory, latency, throughput), optimize model training speed ...

- **api-consistency**
  - path: `.skills-cache/ascend-agent-skills/official/PyTorch/api-consistency/SKILL.md`
  - desc: Use for PyTorch Ascend (torch_npu) API 一致性解单 across A2/A3/A5 servers, including DTS intake, Gate 0 routing between ATK and non-ATK script reproduction, evidence alignment, batch ticket dispatch, tool handoff, and final report navigation across this bundle.

- **api-consistency-analyze-functional**
  - path: `.skills-cache/ascend-agent-skills/official/PyTorch/api-consistency/tools/api-consistency-analyze-functional/SKILL.md`
  - desc: Use for ATK-only functional or environment triage after flow-atk identifies FAILED xlsx rows, including aclnn errors, environment failures, A2/A3 lookup prompts, OOM/timeout evidence, and retest routing.

- **api-consistency-analyze-precision**
  - path: `.skills-cache/ascend-agent-skills/official/PyTorch/api-consistency/tools/api-consistency-analyze-precision/SKILL.md`
  - desc: Use for ATK-only precision analysis when report xlsx rows are successful but precision columns fail, including precision detail review, save_data artifacts, dump/UT replay, and output_grad evidence.

- **api-consistency-analyze-single-op**
  - path: `.skills-cache/ascend-agent-skills/official/PyTorch/api-consistency/tools/api-consistency-analyze-single-op/SKILL.md`
  - desc: Use for ATK-only single-op defining after reproduce or flow-atk routing, including failed case_id enumeration from xlsx, targeted atk -wl reruns, aclnn forward/backward counting, memory-stomp checks, and summary tables.

- **api-consistency-batch-orchestrator**
  - path: `.skills-cache/ascend-agent-skills/official/PyTorch/api-consistency/skills/api-consistency-batch-orchestrator/SKILL.md`
  - desc: Use when handling two or more related API consistency DTS tickets as a batch, with shared read-only setup, per-ticket Gate 0 dispatch, independent ticket outputs, and batch summary aggregation.

- **api-consistency-orchestrator**
  - path: `.skills-cache/ascend-agent-skills/official/PyTorch/api-consistency/skills/api-consistency-orchestrator/SKILL.md`
  - desc: Use as the single-ticket API consistency intake and routing workflow: collect the opening contract, classify Gate 0 into ATK, non-ATK script reproduction, or narrow exit, and hand off to the matching flow.

- **api-consistency-report**
  - path: `.skills-cache/ascend-agent-skills/official/PyTorch/api-consistency/report/api-consistency-report/SKILL.md`
  - desc: Use when writing final API consistency ticket closure artifacts, including 分析结论.md and 复现记录.md, for ATK or non-ATK script reproduction branches after evidence has been collected.

- **api-consistency-reproduce**
  - path: `.skills-cache/ascend-agent-skills/official/PyTorch/api-consistency/tools/api-consistency-reproduce/SKILL.md`
  - desc: Use for ATK-only API consistency reproduction, including log/xlsx reuse, controlled full reruns, runtime baseline setup, -mt/-to tuning, save_data artifacts, and UT replay templates.

- **api-consistency-torchrun-functional**
  - path: `.skills-cache/ascend-agent-skills/official/PyTorch/api-consistency/tools/api-consistency-torchrun-functional/SKILL.md`
  - desc: Use for Gate 0 branch B non-ATK script reproduction of API consistency tickets, including torchrun/python/pytest/bash entrypoints, tee logs, four-source evidence matching, retry limits, and report handoff.

- **arxiv-recommendation-npu**
  - path: `.skills-cache/ascend-agent-skills/official/MindSeriesSDK/RecSDK/arxiv-recommendation-npu/SKILL.md`
  - desc: 自动化推荐系统论文发现流水线。抓取 arxiv 推荐论文，检测源码，生成待迁移任务清单，由 npu-model-migration skill 完成 NPU 适配。

- **ascend-communication-analysis**
  - path: `.skills-cache/ascend-agent-skills/official/MindStudio/skills/ascend-communication-analysis/SKILL.md`
  - desc: Analyze Ascend NPU collective communication profiling data with a DB-first workflow. Use when the user provides `cluster_analysis_output/cluster_analysis.db`, rank-level `analysis.db`, rank-level `ascend_pytorch_profiler_{rank_id}.db`, together with `profiler_info.json`, and asks about HCCL or hc...

- **ascend-computation-analysis**
  - path: `.skills-cache/ascend-agent-skills/official/MindStudio/skills/ascend-computation-analysis/SKILL.md`
  - desc: Analyze Ascend NPU computation-side profiling data for single-card runs or a selected rank from multi-card runs. Use this skill when the user asks to diagnose computation bottlenecks, AI Core / AI Vector / AICPU hotspots, dynamic shape overhead, block dim issues, redundant TransData/Transpose/Cas...

- **ascend-detectron2-install**
  - path: `.skills-cache/ascend-agent-skills/official/MindSpeed/drivingsdk-ascend-model-migration/ascend-mmlab-install-suite/detectron2/SKILL.md`
  - desc: 在昇腾NPU容器中从源码安装detectron2。适用于实例分割、目标检测等模型的开发。

- **ascend-docker**
  - path: `.skills-cache/ascend-agent-skills/official/Common/ascend-docker/SKILL.md`
  - desc: Create Docker containers for Huawei Ascend NPU development with proper device mappings and volume mounts. Use when setting up Ascend development environments in Docker, running CANN applications in containers, or creating isolated NPU development workspaces. Supports privileged mode (default), ba...

- **ascend-github-explorer**
  - path: `.skills-cache/ascend-agent-skills/community/Tools/ascend-github-explorer/SKILL.md`
  - desc: 昇腾 AI / NPU 生态下 GitHub 信息的系统化查询技能。覆盖 vLLM（vllm-project/vllm, vllm-project/vllm-ascend） 和 SGLang（sgl-project/sglang, sgl-project/sgl-kernel-npu）四大核心仓库。 当用户提到以下任一场景时**必须**使用此技能： - 在昇腾/NPU/Ascend 上遇到某个问题，想查社区有没有类似 case（"ascend 上 xxx 报错/oom/hang/精度不对"） - 想了解某个框架对某个模型或特性的支持程度（"vllm/sglang 对 deepseek v...

- **ascend-inference-repos-copilot**
  - path: `.skills-cache/ascend-agent-skills/official/Common/ascend-inference-repos-copilot/SKILL.md`
  - desc: 昇腾（Ascend）推理生态开源代码仓库智能问答专家旨在为 vLLM、vLLM-Ascend、MindIE-LLM、MindIE-SD、MindIE-Motor、MindIE-Turbo 以及 msModelSlim (MindStudio-ModelSlim) 等仓库提供专家级且易于理解的解释。在处理昇腾（Ascend）推理生态相关项目的用户询问时，务必触发此技能（Skill），可解答使用方法、部署流程、支持模型、支持特性、系统架构、配置管理、调试、测试、故障排查、性能优化、定制开发、源码解析以及其他技术问题。支持中英文双语回复，并可借助 deepwiki MCP 工具检索仓库知识库，...

- **ascend-mmcv-install**
  - path: `.skills-cache/ascend-agent-skills/official/MindSpeed/drivingsdk-ascend-model-migration/ascend-mmlab-install-suite/mmcv/SKILL.md`
  - desc: 在昇腾NPU容器中编译安装mmcv-full，支持NPU算子。适用于需要mmcv作为依赖的其他OpenMMLab库安装前的前置步骤。

- **ascend-mmdet-install**
  - path: `.skills-cache/ascend-agent-skills/official/MindSpeed/drivingsdk-ascend-model-migration/ascend-mmlab-install-suite/mmdet/SKILL.md`
  - desc: 在昇腾NPU容器中安装mmdetection。适用于目标检测模型的开发。

- **ascend-mmdet3d-install**
  - path: `.skills-cache/ascend-agent-skills/official/MindSpeed/drivingsdk-ascend-model-migration/ascend-mmlab-install-suite/mmdet3d/SKILL.md`
  - desc: 在昇腾NPU容器中安装mmdetection3d（含mmsegmentation依赖）。适用于3D目标检测模型的开发。

- **ascend-mmlab-install-suite**
  - path: `.skills-cache/ascend-agent-skills/official/MindSpeed/drivingsdk-ascend-model-migration/ascend-mmlab-install-suite/SKILL.md`
  - desc: 昇腾NPU环境安装OpenMMLab系列库套件（mmcv/mmdet/mmdet3d/detectron2），支持本地+远程混合开发模式

- **ascend-model-migration**
  - path: `.skills-cache/ascend-agent-skills/official/MindSpeed/drivingsdk-ascend-model-migration/SKILL.md`
  - desc: Ascend NPU model migration suite. Invoke when user wants to migrate/train models on Ascend NPU, setup environment, or deploy models from open-source repositories.

- **ascend-npu-driver-install**
  - path: `.skills-cache/ascend-agent-skills/official/Common/ascend-npu-driver-install/SKILL.md`
  - desc: 能完成昇腾NPU驱动和固件安装部署，实现安装包正则匹配提取、按需添加可执行权限、Python+Shell双重包校验、系统依赖先验后装、适配CentOS/RHEL/Ubuntu/Debian系统，适用于昇腾NPU驱动和固件安装部署。

- **ascend-om-deployer**
  - path: `.skills-cache/ascend-agent-skills/official/Common/ascend-om-deployer/SKILL.md`
  - desc: Orchestrate end-to-end Ascend OM deployment, including source analysis, ONNX-to-OM conversion, pipeline reintegration, validation, and final delivery. Use when involving the deployment of models to Ascend NPUs using .om files, the conversion of ONNX models to OM format for production, the constru...

- **ascend-om-pipeline-adapter**
  - path: `.skills-cache/ascend-agent-skills/official/Common/ascend-om-pipeline-adapter/SKILL.md`
  - desc: This sub-skill of ascend-om-deployer is designed for source-side analysis and OM reintegration across single-stage, multi-stage, stateful, streaming, service-style, CV, OCR, speech, tracking, and multimodal systems. It identifies the actual inference entry points, source-side subgraph contracts, ...

- **ascend-onnx-atc-pipeline**
  - path: `.skills-cache/ascend-agent-skills/official/Common/ascend-onnx-atc-pipeline/SKILL.md`
  - desc: Environment preparation process and the ONNX-to-OM conversion pipeline for Ascend. Use when verifying compatibility between CANN and Python, repairing the ONNX toolchain, inspecting ONNX input-output contracts, running onnxslim and auto_optimizer, executing ATC with an appropriate fallback order,...

- **ascend-operator-ut-gen**
  - path: `.skills-cache/ascend-agent-skills/official/Common/ascend-operator-ut-gen/SKILL.md`
  - desc: 为 Ascend 自定义算子生成或迁移 framework_normal UT，覆盖 op_host infershape/tiling 与手写 op_api UT。先识别仓库目录、算子类型、arch、CMake/UT 框架，再按本算子的 def/infershape/tiling/op_api 实现生成用例；不要默认 attention、ops-transformer、arch22 或固定 build target。

- **ascend-profiling-anomaly**
  - path: `.skills-cache/ascend-agent-skills/official/Common/ascend-profiling-anomaly/SKILL.md`
  - desc: Analyze Huawei Ascend NPU profiling data to discover hidden performance anomalies and produce a detailed model architecture report reverse-engineered from profiling. Trigger on Ascend profiling traces, NPU bottlenecks, device idle gaps, host-device issues, kernel_details.csv / trace_view.json / o...

- **ascend-schedule-analysis**
  - path: `.skills-cache/ascend-agent-skills/official/MindStudio/skills/ascend-schedule-analysis/SKILL.md`
  - desc: Analyze Ascend NPU schedule, operator dispatch, operator launch, and Host Bound profiling issues in Ascend profiling data. Use when need to diagnose device Free time, framework/operator dispatch latency, launch latency, PYTORCH_API/CANN_API launch gaps, aclrtSynchronizeStream stalls, task queue b...

- **ascend-transformer-boost**
  - path: `.skills-cache/ascend-agent-skills/official/Common/ascend-transformer-boost/SKILL.md`
  - desc: 昇腾 Transformer 加速库（ATB）核心技能集索引（Index Skill）。 整合 16 大核心技能（v1.12.0）：CANN 安装部署、ATB 测试框架编译、Pybind 绑定自动生成、 ATB→ACLNN 算子替换设计文档生成、ATB→ACLNN 算子迁移、ATK 测试用例自动生成、 tbe_adapter 跨 CANN/3rdparty 适配，覆盖昇腾 NPU 开发全链路。


- **ascend-verl-env-check**
  - path: `.skills-cache/ascend-agent-skills/official/Common/ascend-verl-env-check/SKILL.md`
  - desc: 检查verl训练环境是否就绪，包括容器状态、依赖工具安装

- **ascend-verl-env-preparation**
  - path: `.skills-cache/ascend-agent-skills/official/verl/ascend-verl-env-preparation/SKILL.md`
  - desc: 准备verl训练环境，包括拉取镜像、创建启动脚本、启动容器

- **ascend-verl-image-list**
  - path: `.skills-cache/ascend-agent-skills/official/verl/ascend-verl-image-list/SKILL.md`
  - desc: 查询quay.io上可用的VERL镜像列表，支持过滤和版本更新

- **ascend-verl-model-download**
  - path: `.skills-cache/ascend-agent-skills/official/verl/ascend-verl-model-download/SKILL.md`
  - desc: 从ModelScope或HuggingFace下载模型权重

- **ascend-verl-prepare-data**
  - path: `.skills-cache/ascend-agent-skills/official/verl/ascend-verl-prepare-data/SKILL.md`
  - desc: 在verl容器中下载并处理数据集为parquet格式

- **ascend-verl-training**
  - path: `.skills-cache/ascend-agent-skills/official/verl/ascend-verl-training/SKILL.md`
  - desc: 拉起verl训练，包含环境准备、环境检查、数据集处理、模型下载和训练执行；触发关键词：verl训练、使用verl拉起训练

- **ascend_pytorch_profiler_db_explorer**
  - path: `.skills-cache/ascend-agent-skills/official/MindStudio/skills/ascend-profiler-db-explorer/SKILL.md`
  - desc: 面向 Ascend PyTorch Profiler / msprof DB（如 ascend_pytorch_profiler*.db、msprof_*.db）的 SQL 分析技能。将自然语言问题（算子耗时、通信、下发、调度、schema/table 查询）转为安全可执行 SQL，并按需从官方文档提取表结构详情。

- **ascendc-mssanitizer**
  - path: `.skills-cache/ascend-agent-skills/community/Op/ascendc-operator-mssanitizer/SKILL.md`
  - desc: Ascend C 算子 mssanitizer 内存检测分析技能。用于检测和分析算子内存问题：非法内存访问、非法释放、内存泄漏、UB地址越界，生成问题报告。自动识别算子工程类型（ops算子仓用GE IR模式，自定义算子用Python模式）。触发关键词：mssanitizer、内存检测、内存泄漏、非法访问、illegal free、内存错误。

- **ascendc-op-doc-generator**
  - path: `.skills-cache/ascend-agent-skills/community/Op/ascendc-op-doc-generator/SKILL.md`
  - desc: Ascend算子资料文档生成。当用户需要为Ascend CANN算子生成aclnn接口文档(aclnnXxx.md)或开源算子README时触发。输入算子目录路径，自动从源码提取信息并生成两篇标准文档。

- **ascendc-op-doc-incremental**
  - path: `.skills-cache/ascend-agent-skills/community/Op/ascendc-op-doc-incremental/SKILL.md`
  - desc: Ascend算子文档增量更新。当用户需要对比源码变更与已有文档、增量修改已有aclnn MD或README文档时触发。输入算子目录路径，自动提取源码信息、对比已有文档、输出增量修改后的文件和对比报告。

- **ascendc-operator-A5-migration**
  - path: `.skills-cache/ascend-agent-skills/official/Common/ascendc-operator-A5-migration/SKILL.md`
  - desc: AscendC 算子从 A2/A3(910b/910_93) 迁移到 A5(950) 的 L1+L2+L3 级别改造。当用户需要对 AscendC 算子做 950 适配、A5 迁移、ascend950 编译、算子跨芯片移植、MicroAPI 重写、RegBase 改造、SIMT 优化时使用。触发词：迁移、950、A5、RegBase、arch35、L1、L2、L3、MicroAPI、RegTensor、CastTrait、SIMT、Scatter、Gather。

- **ascendc-operator-code-gen**
  - path: `.skills-cache/ascend-agent-skills/community/Op/ascendc-operator-code-gen/SKILL.md`
  - desc: 根据设计文档生成 AscendC 算子完整代码实现并完成框架适配。TRIGGER when: 设计文档已完成，需要生成 op_host/op_kernel 代码、注册到 PyTorch 框架、编译测试。关键词：代码生成、op_host、op_kernel、tiling、kernel、框架适配、算子注册。

- **ascendc-operator-code-review**
  - path: `.skills-cache/ascend-agent-skills/community/Op/ascendc-operator-code-review/SKILL.md`
  - desc: Ascend C 代码检视技能。基于假设检验方法论对代码进行安全规范检视。调用时必须明确提供：代码片段和检视规则描述。TRIGGER when: 用户要求代码检视、代码review、询问代码安全问题、检查编码规范、或需要检查特定代码问题（如内存泄漏、整数溢出、空指针等）。关键词：Ascend C、代码检视、代码review、安全规范、内存、指针、溢出、泄漏、编码规范。

- **ascendc-operator-compile-debug**
  - path: `.skills-cache/ascend-agent-skills/community/Op/ascendc-operator-compile-debug/SKILL.md`
  - desc: 编译安装 AscendC 算子并执行精度测试。TRIGGER when: 算子代码生成完成后需要编译验证、安装 whl 包、运行精度测试，或编译/测试失败需要排查。关键词：build.sh、编译、安装、whl、pytest、精度测试、编译错误、NPU 测试。

- **ascendc-operator-design**
  - path: `.skills-cache/ascend-agent-skills/community/Op/ascendc-operator-design/SKILL.md`
  - desc: 完成AscendC算子设计 - 帮助用户完成算子的架构设计、接口定义和性能规划。当用户提到算子设计、算子开发、tiling策略、内存规划、AscendC kernel设计、两级tiling、核间切分、核内切分时，使用此skill。

- **ascendc-operator-dev**
  - path: `.skills-cache/ascend-agent-skills/community/Op/ascendc-operator-dev/SKILL.md`
  - desc: AscendC算子端到端开发编排器。当用户需要开发新算子、实现自定义算子、或完成从需求到测试的完整流程时使用。关键词：算子开发、operator development、端到端、完整流程、工作流编排、新建算子。

- **ascendc-operator-doc-gen**
  - path: `.skills-cache/ascend-agent-skills/community/Op/ascendc-operator-doc-gen/SKILL.md`
  - desc: 为AscendC算子生成PyTorch风格的接口文档（README.md）。触发场景：编译调试通过后需要生成接口文档，或用户提到"生成算子文档"、"创建README"、"文档化算子"、"帮我写文档"（算子上下文）、"算子文档"时使用。

- **ascendc-operator-doc-writer**
  - path: `.skills-cache/ascend-agent-skills/community/Op/ascendc-operator-doc-writer/SKILL.md`
  - desc: Write README-style technical documentation for AscendC custom operators by reading local source files and adapting an existing template. Use when Codex needs to document an AscendC operator, compare a target operator repo against a reference README, turn `op_host` and `op_kernel` implementations ...

- **ascendc-operator-performance-eval**
  - path: `.skills-cache/ascend-agent-skills/community/Op/ascendc-operator-performance-eval/SKILL.md`
  - desc: 在 ascend-kernel 的 csrc/ops/<op>/test 下维护仅含 JSONL 的 profiler 性能用例，使用 torch_npu.profiler（固定 warmup=5、active=5）采集，汇总 ASCEND_PROFILER_OUTPUT/op_statistic.csv 的 Total Time(us)，输出含 DType 列的统一 Markdown 对比报告（自定义算子 vs 标杆）。不生成 perf_cases.json 与 *_profiler_results.json。参考实现见 examples/layer_norm_profiler_ref...

- **ascendc-operator-precision-debug**
  - path: `.skills-cache/ascend-agent-skills/community/Op/ascendc-operator-precision-debug/SKILL.md`
  - desc: AscendC 算子精度问题调试与根因定位。当算子精度测试失败（allclose 不通过、结果偏差、输出全零/NaN 等）时使用。流程：误差分布分析 → 代码易错点审查 → 实验隔离 → printf/DumpTensor 插桩 → 修复验证。关键词：精度调试、精度问题、结果不一致、误差定位、allclose 失败、输出偏差、NaN、全零、precision debug。

- **ascendc-operator-precision-eval**
  - path: `.skills-cache/ascend-agent-skills/community/Op/ascendc-operator-precision-eval/SKILL.md`
  - desc: AscendC算子精度评估。对已编译安装的算子生成全面的精度测试用例集（≥30例），运行并生成精度验证报告。关键词：精度测试、precision evaluation、精度报告、accuracy、误差分析。执行完成后 MUST 在当前对话中展示总览、失败摘要与关键发现，不得仅附报告路径。

- **ascendc-operator-project-init**
  - path: `.skills-cache/ascend-agent-skills/community/Op/ascendc-operator-project-init/SKILL.md`
  - desc: 初始化 AscendC 算子工程并创建可编译的算子骨架。触发场景：(1) 用户要求创建新算子；(2) 关键词：ascendc算子、新建算子、算子目录、算子初始化；(3) 需要基于 ascend-kernel 模板快速落地。本 skill 不只建目录，还输出“可继续开发”的标准文件与检查清单。

- **ascendc-operator-st-gen**
  - path: `.skills-cache/ascend-agent-skills/community/Op/ascendc-operator-st-gen/SKILL.md`
  - desc: 为昇腾 CANN 算子创建或修正 ATK ST 双标杆精度工程。当用户提到 ATK ST、aclnn ST、pyaclnn、双标杆、atk case、atk task、smoke_case.json、自定义执行方式、-cp 签名、cv_fused_double_benchmark 时使用；不足时可经用户同意查阅 AscendTest/ATK 官方仓；不修改算子 kernel/tiling。

- **ascendc-operator-testcase-gen**
  - path: `.skills-cache/ascend-agent-skills/community/Op/ascendc-operator-testcase-gen/SKILL.md`
  - desc: 完成AscendC算子验证用例生成 - 帮助用户完成testcase设计。当用户提到用例设计、泛化用例生成、算子标杆、UT用例、精度用例、性能用例时，使用此skill。

- **atb-aclnn-operator-migration**
  - path: `.skills-cache/ascend-agent-skills/official/Common/ascend-transformer-boost/skills/atb-aclnn-operator-migration/SKILL.md`
  - desc: 自动执行 ATB 算子到 ACLNN 的迁移操作，在 910B/950 设备上启用 ACLNN 加速。 支持参数映射、ACLNN Runner 实现、设备检测切换和功能/性能验证全流程。


- **atb-aclnn-operator-replacement-designer**
  - path: `.skills-cache/ascend-agent-skills/official/Common/ascend-transformer-boost/skills/atb-aclnn-operator-replacement-designer/SKILL.md`
  - desc: 自动生成 ATB 到 ACLNN 算子替换的详细设计文档。接收用户提供的 ATB 和 ACLNN 接口文档链接， 输出包含参数映射、开发自测、风险评估的 7 章结构化设计文档。 TRIGGER when: 用户需要将 ATB 算子替换为 ACLNN 算子并撰写设计文档。


- **atb-atk-testcase-generator**
  - path: `.skills-cache/ascend-agent-skills/official/Common/ascend-transformer-boost/skills/atb-atk-testcase-generator/SKILL.md`
  - desc: ATB ATK 测试用例生成主控技能。负责 6-Gate 流程编排和 HIL 门禁控制。 详细实现拆分到 checks/references/templates/scripts 资源目录，避免超长单文件。


- **atb-csv-testcase-generator**
  - path: `.skills-cache/ascend-agent-skills/official/Common/ascend-transformer-boost/skills/atb-csv-testcase-generator/SKILL.md`
  - desc: ATB CSV 测试用例生成技能。当用户需要为 ATB 算子创建 CSV 格式的泛化测试用例时调用此技能。 覆盖：正例设计、反例设计、性能测试用例、CSV 格式规范。


- **atb-csv-tester**
  - path: `.skills-cache/ascend-agent-skills/official/Common/ascend-transformer-boost/skills/atb-csv-tester/SKILL.md`
  - desc: 运行 ATB (Ascend Transformer Boost) CSV 测试。当用户需要执行 CSV 格式的 ATB 测试用例、 验证算子正确性、或运行任何ATB下的 CSV 测试文件时调用此技能。 需配合 CANN 环境和已编译的 ATB 测试框架使用。


- **atb-debug-guide**
  - path: `.skills-cache/ascend-agent-skills/official/Common/ascend-transformer-boost/skills/atb-debug-guide/SKILL.md`
  - desc: ATB 调试指南技能。当用户遇到 ATB 算子测试问题、需要分析错误原因、或需要了解 ATB 环境配置时调用此技能。 覆盖：GDB core dump 分析、NZ格式丢失、异步析构segfault、TaskQueue OpCommand 陷阱、ATK ATB perf 栈与 NPUBackend 继承关系、PerformanceConfig 合并阶段 TypeError/KeyError 等场景。


- **atb-golden-developer**
  - path: `.skills-cache/ascend-agent-skills/official/Common/ascend-transformer-boost/skills/atb-golden-developer/SKILL.md`
  - desc: ATB CSV 测试 Golden 参考实现开发指南。覆盖 DataGen 类的 customize/golden/case_preprocess 三件套开发模式、hostData 注入机制（hosttensor binder）、NO_ERROR vs I:NO_ERROR 区别、 kernel 对齐原则、精度调试完整流程。当用户需要为算子编写 CSV 正例 golden 参考实现时调用此技能。


- **atb-knowledge-generalize**
  - path: `.skills-cache/ascend-agent-skills/official/Common/ascend-transformer-boost/skills/atb-knowledge-generalize/SKILL.md`
  - desc: 批量泛化知识提取。基于模式库将已验证的模板 Op 知识泛化到同分类其他 Op， 按缺口盘点（PENDING/STALE/OK）批量编排提取任务，经质量门禁自动接受或标记人工审核。


- **atb-nnal-installer**
  - path: `.skills-cache/ascend-agent-skills/official/Common/ascend-transformer-boost/skills/atb-nnal-installer/SKILL.md`
  - desc: 昇腾 NPU NNAL（ATB 加速库）安装技能。依赖 cann-operator-env-config 提供 Toolkit+Kernels 环境，本技能仅负责 NNAL 包的安装、环境变量配置与验证。


- **atb-op-knowledge-extract**
  - path: `.skills-cache/ascend-agent-skills/official/Common/ascend-transformer-boost/skills/atb-op-knowledge-extract/SKILL.md`
  - desc: 从 ATB 算子源码自动提取结构化知识条目。读取路由文件确定源码清单， 分析 C++ 源码提取参数约束、Computation Pipeline（dtype 流转）、 执行路径、Kernel 依赖、已知问题，最终组装为 9 章节 Markdown 知识条目。 TRIGGER when: 用户需要提取/生成某个 ATB 算子的知识条目，或批量提取分类下全部算子。


- **atb-op-knowledge-update**
  - path: `.skills-cache/ascend-agent-skills/official/Common/ascend-transformer-boost/skills/atb-op-knowledge-update/SKILL.md`
  - desc: 检测 ATB 源码变更并增量更新知识条目。通过 git diff 识别变更文件， 映射到受影响的 Op，按 11 种变更模式精准重提取受影响章节。 TRIGGER when: ATB 仓有 git commit 变更，或用户要求更新某个 Op 的知识条目。


- **atb-ops-to-aclnn-migration-workflow**
  - path: `.skills-cache/ascend-agent-skills/official/Common/ascend-transformer-boost/skills/atb-ops-to-aclnn-migration-workflow/SKILL.md`
  - desc: ATB OPS→ACLNN 迁移标准化工作流主模板。整合前置学习、设计文档生成、CSV用例设计、 实际迁移、编译验证、测试验证全流程，提供明确的阶段 Gates 和用户确认机制。


- **atb-pybind-bindgen**
  - path: `.skills-cache/ascend-agent-skills/official/Common/ascend-transformer-boost/skills/atb-pybind-bindgen/SKILL.md`
  - desc: ATB torch_atb pybind11 绑定自动生成工作流。当用户需要自动绑定、pybind 代码生成、 infer_op_params.h 新增算子同步、torch_atb 绑定维护、bindgen 端到端验证时调用。 触发词：自动绑定、pybind、torch_atb、infer_op_params、bindgen、bindgen 验证、verify_bindgen。


- **atb-pybind-test-generator**
  - path: `.skills-cache/ascend-agent-skills/official/Common/ascend-transformer-boost/skills/atb-pybind-test-generator/SKILL.md`
  - desc: 生成 ATB pybind 测试用例（op_param_test + bindgen golden forward 测试）。 触发词：pybind 测试生成、op_param_test、bindgen_test、golden recipe。


- **atb-tbe-adapter-adapt**
  - path: `.skills-cache/ascend-agent-skills/official/Common/ascend-transformer-boost/skills/atb-tbe-adapter-adapt/SKILL.md`
  - desc: ATB tbe_adapter 跨 CANN/3rdparty 大版本适配：release compile、stubs/CMake、 3rdparty 分支对齐、AutoTiling 隔离、CSV/加载运行时验证与 pack。 用于 tbe_adapter / libtbe_adapter 编译或加载失败、ops-nn/canndev/metadef/opbase 等 release 依赖仓与 CANN 不同步，或 CANN 升级后 tbe 相关 csv（如 ScatterElementsV2、 elewise、activation）失败等场景。


- **atb-testframework-build**
  - path: `.skills-cache/ascend-agent-skills/official/Common/ascend-transformer-boost/skills/atb-testframework-build/SKILL.md`
  - desc: 编译 ATB (Ascend Transformer Boost) 测试框架。当用户需要编译 ATB 测试框架、 torch_atb、运行 CSV 测试、或构建 atb_test_framework 时调用。 支持全量编译和增量编译两种模式。需在 Docker 容器内配合 CANN 环境执行。


- **atc-model-converter**
  - path: `.skills-cache/ascend-agent-skills/official/Common/atc-model-converter/SKILL.md`
  - desc: Complete toolkit for Huawei Ascend NPU model conversion and end-to-end inference adaptation. Workflow 1 auto-discovers input shapes and parameters from user source code. Workflow 2 exports PyTorch models to ONNX. Workflow 3 converts ONNX to .om via ATC with multi-CANN version support. Workflow 4 ...

- **auto-bug-fixer**
  - path: `.skills-cache/ascend-agent-skills/official/Common/auto-bug-fixer/SKILL.md`
  - desc: Use when encountering bugs, test failures, or error logs that need root cause analysis and fix generation

- **auto-develop-test-gen**
  - path: `.skills-cache/ascend-agent-skills/official/Common/auto-develop-test-gen/SKILL.md`
  - desc: 开发者测试自动补全技能 - 为函数和类生成高质量单元测试，分析覆盖率盲区并生成高价值补充测试，提升有效覆盖率。

- **cann-nnal-installer**
  - path: `.skills-cache/ascend-agent-skills/official/Common/cann-nnal-installer/SKILL.md`
  - desc: 昇腾NPU CANN Toolkit+Kernels+NNAL安装部署技能。支持从官网下载run包安装和从Docker镜像提取两种方式，覆盖驱动检查、包下载、安装、环境变量配置与验证全流程。当用户需要安装CANN全套组件或指定版本CANN到自定义路径时调用。

- **cann-operator-env-config**
  - path: `.skills-cache/ascend-agent-skills/official/Common/cann-operator-env-config/SKILL.md`
  - desc: 提供昇腾NPU的CANN安装指导。当用户需要安装CANN、配置昇腾环境或解决安装问题时调用。

- **catlass-operator-code-gen**
  - path: `.skills-cache/ascend-agent-skills/community/Op/catlass-operator-code-gen/SKILL.md`
  - desc: 根据CATLASS算子设计文档生成算子工程交付件

- **catlass-operator-design**
  - path: `.skills-cache/ascend-agent-skills/community/Op/catlass-operator-design/SKILL.md`
  - desc: 将用户基于CATLASS开发算子的需求转变为具体的设计文档

- **catlass-operator-dev**
  - path: `.skills-cache/ascend-agent-skills/community/Op/catlass-operator-dev/SKILL.md`
  - desc: Catlass 算子端到端开发编排器。基于 ascend-kernel（csrc/ops），串联 catlass 设计、catlass-operator-code-gen 与 ascendc 子 skill，完成从工程初始化到文档、精度、性能的闭环。关键词：Catlass、端到端、ascend-kernel、算子开发、工作流编排。

- **catlass-operator-performance-optim**
  - path: `.skills-cache/ascend-agent-skills/community/Op/catlass-operator-performance-optim/SKILL.md`
  - desc: 指导 Catlass 算子性能调优。流程：阅读 catlass 优化指南、获取/更新 profiler 基线、按指南修改 tiling、重新编译、**强制产出并展示性能对比报告**、迭代对比。调优策略以 catlass 文档为准。条件不明则追问。

- **ci-static-errors-fix**
  - path: `.skills-cache/ascend-agent-skills/official/PyTorch/ci-static-errors-fix/SKILL.md`
  - desc: Use when the user asks to check or fix CI lint/static style errors in a PyTorch-style repository, including lintrunner, flake8/PEP8, whitespace, formatting, or codespell issues. Limit checks and fixes to files modified in the current working tree, use lintrunner auto-fix only for approved rules, ...

- **cluster-fast-slow-rank-detector**
  - path: `.skills-cache/ascend-agent-skills/official/MindStudio/skills/cluster-fast-slow-rank-detector/SKILL.md`
  - desc: 专门用于 Ascend 集群 Profiling 性能数据的“快慢卡”诊断专家技能。当用户提供【集群性能数据目录/路径】并要求分析【快慢卡】、【慢节点】、【负载不均衡】或【集群瓶颈】时，必须触发此技能。该技能会自动接收集群路径，调度相关工具输出快慢卡的宏观定性与微观根因（如 Host 下发瓶颈、算子计算劣化）。

- **code-comprehension**
  - path: `.skills-cache/ascend-agent-skills/official/MindSpeed/mindspeed-llm-auto-ut-skills/skills/code-comprehension/SKILL.md`
  - desc: 在多个尺度上理解和总结代码功能，从函数级到模块级到系统级，帮助快速掌握陌生代码库。特别适用于大语言模型训练框架、分布式训练系统、深度学习框架等复杂代码库的分析。

- **com-auto-binary-pr**
  - path: `.skills-cache/ascend-agent-skills/official/PyTorch/com-auto-binary-pr/SKILL.md`
  - desc: Use when performing PTA problem binary localization with COM auto binary PR tooling, including automated whl build/test loops, binary_config.json, binary_result.json, and auto_whl workflow navigation.

- **community-pr-test-tracking**
  - path: `.skills-cache/ascend-agent-skills/official/PyTorch/pytorch-pr-analysis/community-pr-test-tracking/SKILL.md`
  - desc: Track and test community PyTorch PRs on NPU. Use when the user asks to check community follow-up, community case tracking, follow PyTorch PR tests, or run community tests by PR number. Fetches PR diff from GitHub, extracts test files and test cases, injects the NPU adaptation snippet, runs tests,...

- **core-requirement-analyze**
  - path: `.skills-cache/ascend-agent-skills/official/PyTorch/core-requirement-analyze/SKILL.md`
  - desc: Use for Core 组需求分析 when analyzing PyTorch or torch_npu modules, mechanisms, commits, or source paths and producing a source-backed Chinese requirement/design/implementation analysis document.

- **coverage**
  - path: `.skills-cache/ascend-agent-skills/official/MindSpeed/mindspeed-llm-auto-ut-skills/skills/coverage/SKILL.md`
  - desc: Use when working with coverage

- **deterministic-calculation-analysis**
  - path: `.skills-cache/ascend-agent-skills/official/MindStudio/skills/deterministic-calculation-analysis/SKILL.md`
  - desc: 执行msProbe数据比对并分析比对结果，定位确定性计算问题首个输入一致输出不一致的API。

- **document-ux-review**
  - path: `.skills-cache/ascend-agent-skills/official/MindStudio/skills/document-ux-review/SKILL.md`
  - desc: 当用户希望你像第一次接触项目的人一样，真实按仓库的 README、安装文档或 quick start 跑一遍，并判断“新人能不能走通”“文档是否可用”“哪里会卡住”“安装/启动说明是否对新手友好”时，使用这个 skill。它适用于 repo onboarding audit、documentation UX review、quickstart validation、README walkthrough、按文档验证安装与运行并输出问题报告的场景；即使用户只是说“按 README 试一下”“帮我检查这个仓库文档能不能跑通”“看看 quick start 为什么带不动新人”，也应触发。不要用于...

- **error-analyzer**
  - path: `.skills-cache/ascend-agent-skills/official/MindSeriesSDK/AgentSDK/error-analyzer/SKILL.md`
  - desc: Analyzes user-provided error messages, logs, and environment information to identify root causes 
and generate customer-friendly responses for Ascend NPU hardware scenarios. Use when: (1) User provides 
error logs, stack traces, or crash reports, (2) User describes a problem with environment/cont...

- **fault_diagnose**
  - path: `.skills-cache/ascend-agent-skills/official/Common/fault_diagnose/SKILL.md`
  - desc: Ascend 故障诊断工具，提供日志采集、清洗、诊断全流程。支持集群/单机/超节点故障诊断，当用户需要排查 NPU 训练推理故障或性能劣化问题时调用。

- **gen-evaluation-cfg**
  - path: `.skills-cache/ascend-agent-skills/official/MindStudio/skills/gen-evaluation-cfg/SKILL.md`
  - desc: Generate msmodelslim evaluation YAML configuration (service_oriented + aisbench + vllm-ascend). Use when user asks for evaluation config generation.

- **generate-unit-test**
  - path: `.skills-cache/ascend-agent-skills/official/MindSpeed/mindspeed-llm-auto-ut-skills/skills/generate-unit-test/SKILL.md`
  - desc: 为函数和类生成高质量单元测试，覆盖正常路径、边界条件和异常场景

- **gitcode-code-reviewer**
  - path: `.skills-cache/ascend-agent-skills/official/MindStudio/skills/gitcode-code-reviewer/SKILL.md`
  - desc: 用于审查 GitCode PR，并结合 PR metadata、diff 与整个代码仓上下文生成深度审查结论或发布逐行评论。当用户希望 review GitCode PR、检查某个 GitCode PR 链接、分析变更风险、或将审查意见发布到 GitCode PR 时使用。典型触发方式包括“review this PR”“检视这个 PR”“检查 PR”，或直接提供 GitCode PR 链接，例如 https://gitcode.com/owner/repo/pull/123 。

- **github-raw-fetch**
  - path: `.skills-cache/ascend-agent-skills/official/MindStudio/skills/github-raw-fetch/SKILL.md`
  - desc: 当用户提供 GitHub 文件页面链接，或希望读取某个仓库中的源码、配置、README、Markdown、docs 内容时，使用此技能。技能不仅支持将 `github.com/<owner>/<repo>/blob/<ref>/...` 转换为 `raw.githubusercontent.com` 链接，还要求在读取仓库 docs 前优先读取同仓库同 ref 的 `agent_router.md`，根据其中声明的目录结构或路由规则拼出真实路径，并优先通过 `curl` 获取内容。

- **hccl-test**
  - path: `.skills-cache/ascend-agent-skills/official/Common/hccl-test/SKILL.md`
  - desc: HCCL (Huawei Collective Communication Library) performance testing for Ascend NPU clusters. Use for testing distributed communication bandwidth, verifying HCCL functionality, and benchmarking collective operations like AllReduce, AllGather. Covers MPI installation, multi-node pre-flight checks (S...

- **k8s-check-fix**
  - path: `.skills-cache/ascend-agent-skills/official/MindCluster/k8s-check-fix/SKILL.md`
  - desc: Kubernetes 集群健康检查与安全修复 — 诊断问题，用户确认后执行修复

- **large_scale_deploy**
  - path: `.skills-cache/ascend-agent-skills/official/Common/large_scale_deploy/SKILL.md`
  - desc: 自动化大规模集群安装部署工具，用于 ascend-deployer 组件批量部署。当用户需要跨集群部署组件或执行批量安装操作时调用。

- **manual-connect-npu-server**
  - path: `.skills-cache/ascend-agent-skills/official/PyTorch/api-consistency/tools/manual-connect-npu-server/SKILL.md`
  - desc: Use when an API consistency ticket needs remote NPU server access, SSH key-based connection setup, optional existing container/env entry, read-only environment inspection, and remote_torch_npu metadata capture.

- **megatron-change-analyzer**
  - path: `.skills-cache/ascend-agent-skills/official/MindSpeed/megatron-change-analyzer/SKILL.md`
  - desc: Analyze official Megatron-LM commits, PRs, and branch change sets to identify feature evolution, candidate breaking changes, and migration-relevant events. Use when Codex already has a normalized Megatron change set and needs to explain what changed, which new features matter, and which changes s...

- **megatron-commit-tracker**
  - path: `.skills-cache/ascend-agent-skills/official/MindSpeed/megatron-commit-tracker/SKILL.md`
  - desc: Track and normalize change requests against the official Megatron-LM repository by branch, PR, commit, commit range, or time window. Use when Codex needs to collect the exact upstream change set before deeper analysis, especially for branch-aware Megatron and MindSpeed migration work, daily/perio...

- **megatron-impact-mapper**
  - path: `.skills-cache/ascend-agent-skills/official/MindSpeed/megatron-impact-mapper/SKILL.md`
  - desc: Map migration-relevant Megatron changes onto the official MindSpeed repository by resolving branch alignment, locating affected subsystems, and identifying concrete adaptation points. Use when Codex has structured Megatron change events and needs to decide whether MindSpeed already covers them, w...

- **megatron-migration-generator**
  - path: `.skills-cache/ascend-agent-skills/official/MindSpeed/megatron-migration-generator/SKILL.md`
  - desc: Generate migration deliverables for bringing relevant Megatron changes into MindSpeed after branch alignment and impact mapping are complete. Use when Codex already has a confirmed MindSpeed-to-Megatron branch pairing and needs to produce a migration report, candidate patch, or guarded workspace ...

- **mindspeed-fsdp2-config-migration**
  - path: `.skills-cache/ascend-agent-skills/official/MindSpeed/mindspeed-mm-fsdp2-migration/mindspeed-fsdp2-config-migration/SKILL.md`
  - desc: 用于将源训练设置映射到 MindSpeed-MM FSDP2 YAML 契约。适用于创建或修复 model_id/dataset_type/plugin 对齐、strict/extra 分层与分片配置时。

- **mindspeed-fsdp2-data-migration**
  - path: `.skills-cache/ascend-agent-skills/official/MindSpeed/mindspeed-mm-fsdp2-migration/mindspeed-fsdp2-data-migration/SKILL.md`
  - desc: 用于将数据预处理与数据加载契约迁移到 MindSpeed-MM FSDP2。适用于实现数据集注册、预处理复用、collate 行为与输入字段兼容时。

- **mindspeed-fsdp2-migration-main**
  - path: `.skills-cache/ascend-agent-skills/official/MindSpeed/mindspeed-mm-fsdp2-migration/mindspeed-fsdp2-migration-main/SKILL.md`
  - desc: 用于统筹 MindSpeed-MM FSDP2 端到端迁移。适用于需要协同模型、数据、配置与验证子流程迁移任意新模型时。

- **mindspeed-fsdp2-model-migration**
  - path: `.skills-cache/ascend-agent-skills/official/MindSpeed/mindspeed-mm-fsdp2-migration/mindspeed-fsdp2-model-migration/SKILL.md`
  - desc: 用于模型侧迁移到 MindSpeed-MM FSDP2 注册与加载契约。适用于实现模型插件、加载签名兼容、token/embedding 更新与前向兼容时。

- **mindspeed-fsdp2-verification**
  - path: `.skills-cache/ascend-agent-skills/official/MindSpeed/mindspeed-mm-fsdp2-migration/mindspeed-fsdp2-verification/SKILL.md`
  - desc: 用于执行 MindSpeed-MM FSDP2 迁移的功能与可靠性验收门禁。适用于模型/数据/配置改动后，验证一次分布式端到端成功并留存证据时。

- **mindspeed-llm-st-generator**
  - path: `.skills-cache/ascend-agent-skills/official/MindSpeed/mindspeed-llm-st-generator/SKILL.md`
  - desc: 为 MindSpeed-LLM 的 FSDP2 模型自动生成系统测试(ST)用例及基线数据。当需要为新模型添加 pretrain ST 用例、生成基线数据、或识别缺失 ST 覆盖的模型时调用此技能。触发词：生成ST用例、补全ST、FSDP2测试用例、缺失ST模型、生成基线数据、ST baseline、generate ST case、FSDP2 ST、missing ST coverage。

- **mindstudio-cpu-binding**
  - path: `.skills-cache/ascend-agent-skills/official/MindStudio/skills/mindstudio-cpu-binding/SKILL.md`
  - desc: Use when diagnosing NPU + PyTorch or LLM Serving Host CPU affinity, NUMA locality, cgroup/cpuset constraints, CPU range conflicts, PyTorch/runtime threading, DataLoader, tokenizer, scheduler, vLLM-Ascend, SGLang, TTFT, TPOT, tokens/s, QPS, or multi-rank/multi-worker CPU binding issues.

- **mindstudio_profiler_data_check**
  - path: `.skills-cache/ascend-agent-skills/official/MindStudio/skills/mindstudio_profiler_data_check/SKILL.md`
  - desc: 当用户提供 MindStudio profiler 采集的性能数据（框架 profiler、msprof 命令行）时，对数据完整性、采集状态及关键配置进行校验，确保后续分析工具能正常运行。

- **model-migration**
  - path: `.skills-cache/ascend-agent-skills/official/MindSpeed/drivingsdk-ascend-model-migration/model-migration/SKILL.md`
  - desc: Model code migration for Ascend NPU. Invoke when user needs to clone open-source repo and apply NPU adaptation patches.

- **model-training**
  - path: `.skills-cache/ascend-agent-skills/official/MindSpeed/drivingsdk-ascend-model-migration/model-training/SKILL.md`
  - desc: Model training on Ascend NPU. Invoke when user wants to launch training script and monitor training progress.

- **modelscope-cli**
  - path: `.skills-cache/ascend-agent-skills/official/Common/modelscope-cli/SKILL.md`
  - desc: ModelScope CLI 模型与数据集下载工具。当用户需要从 ModelScope 下载模型或数据集、批量下载模型、校验文件完整性、统计模型参数量、或进行网络诊断时使用。

- **msmodeling-device-config**
  - path: `.skills-cache/ascend-agent-skills/official/MindStudio/skills/msmodeling-device-config/SKILL.md`
  - desc: 在需要根据自然语言规格为未支持硬件新增或更新 DeviceProfile 设备画像条目时使用

- **msmodeling-env-installer**
  - path: `.skills-cache/ascend-agent-skills/official/MindStudio/skills/msmodeling-env-installer/SKILL.md`
  - desc: Install and verify the msmodeling development environment. Use when the user explicitly asks to install msmodeling dependencies, set up this repository, create `myenv` with `uv`, install this repository's `requirements.txt`, set project `PYTHONPATH`, or configure `HF_ENDPOINT`; if the user only s...

- **msmodeling-throughput-optimizer-executor**
  - path: `.skills-cache/ascend-agent-skills/official/MindStudio/skills/msmodeling-throughput-optimizer-executor/SKILL.md`
  - desc: Interactively gather parameters for `python -m cli.inference.throughput_optimizer`, generate a deployment simulation command, explain assumptions, ask for execution confirmation, then run the simulation and summarize the best parallel strategy. Use when the user wants to evaluate a model on one o...

- **msmodelslim-adapter-verification**
  - path: `.skills-cache/ascend-agent-skills/official/MindStudio/skills/msmodelslim-adapter-verification/SKILL.md`
  - desc: 为 msModelSlim 适配器执行功能性验证。适用于基础适配器开发完成后，自动执行四步验证（测试模型、全回退量化、权重一致性与可加载/保存、实际量化规则校验）并输出通过/失败结论。

- **msmodelslim-layer-wise-quantization**
  - path: `.skills-cache/ascend-agent-skills/official/MindStudio/skills/msmodelslim-layer-wise-quantization/SKILL.md`
  - desc: 为 msModelSlim 适配器实现逐层量化（按层加载/懒加载）能力。仅在用户明确要求逐层量化或基础适配因 CPU 内存不足无法全量加载权重时使用。该特性为高阶可选项，不是基础适配必需项。

- **msmodelslim-model-adapt**
  - path: `.skills-cache/ascend-agent-skills/official/MindStudio/skills/msmodelslim-model-adapt/SKILL.md`
  - desc: 为 msModelSlim 创建基础 Transformers 模型适配器（Model Adapter）。 包含创建适配器、实现必需接口与注册安装流程。 适用：Decoder-only LLM、理解类 VLM（仅 LLM/text 部分）。 不适用：多模态生成模型（图像/视频/语音生成）、Encoder-only、非 Transformers 架构。

- **msmodelslim-model-analysis**
  - path: `.skills-cache/ascend-agent-skills/official/MindStudio/skills/msmodelslim-model-analysis/SKILL.md`
  - desc: 在实现适配器前对候选模型做分析。确定模型实现来源（transformers 或模型目录）、结构特征、内存约束下的逐层量化建议（可选）及 MoE 融合权重风险。适用于用户询问模型适配可行性或做适配前分析时使用。

- **msmodelslim-model-dequant**
  - path: `.skills-cache/ascend-agent-skills/official/MindStudio/skills/msmodelslim-model-dequant/SKILL.md`
  - desc: 为 msModelSlim 适配流程注入反量化能力。先识别模型权重是否可反量化，再实现反量化脚本并接入 model_adapter。当前仅覆盖 FP8 的 per-block 与 per-channel 两类；若格式不确定或无公开反量化规则，要求用户提供反量化脚本或浮点权重。

- **msmodelslim-quick-quant**
  - path: `.skills-cache/ascend-agent-skills/official/MindStudio/skills/msmodelslim-quick-quant/SKILL.md`
  - desc: 提供 msModelSlim 的通用快速量化指引，包含安装、最简 YAML 配置与基础执行校验。适用于用户询问 msmodelslim 安装、快速量化、配置 yaml、linear_quant 或 minmax 基础参数时。

- **msot-msopprof-operator-profiler**
  - path: `.skills-cache/ascend-agent-skills/official/MindStudio/skills/msot-msopprof-operator-profiler/SKILL.md`
  - desc: 当用户希望使用 msOpProf（`msprof op` / `msprof op simulator`）对昇腾 AI 算子做上板或仿真性能调优、解释 `aic-metrics`/`trace.json`/`visualize_data.bin`、选择 device vs simulator 路径、排查 `--soc-version`/`--export`/`signal 6`/`Bad address`/热点图或流水图相关问题，或要求生成固定分析报告模板（算子基本信息 / 关键数据 TOP5 / 核心瓶颈 TOP5 / 优化建议 TOP5）时，使用本技能。它负责先判定模式、输入形态与芯...

- **msprof-analyze-cli**
  - path: `.skills-cache/ascend-agent-skills/official/MindStudio/skills/msprof-analyze-cli/SKILL.md`
  - desc: MindStudio Profiler Analyze（msprof-analyze）是面向 AI 训练与推理场景的性能分析工具，基于采集得到的 profiling 数据进行统计、比对和诊断，帮助定位计算、通信、调度及集群场景下的性能瓶颈。

- **msverl-daily-regression-triage**
  - path: `.skills-cache/ascend-agent-skills/official/Common/msverl-daily-regression-triage/SKILL.md`
  - desc: Triage a daily msverl regression run by reading the baseline comparison log, stopping on success, extracting the most relevant training failure evidence from the daily training log when needed, collecting recent commits from verl main and MindSpeed master, and ranking the most likely culprit comm...

- **nan-overflow-detection**
  - path: `.skills-cache/ascend-agent-skills/official/MindStudio/skills/nan-overflow-detection/SKILL.md`
  - desc: 多卡分布式训练中的 loss/gnorm 精度溢出检测与根因追溯。基于 MSProbe dump 数据，先跨 rank 定位首次出现 NaN 的源卡，再在源卡上追溯具体的溢出根因算子。
当用户需要：(1) 多卡分布式训练场景下的 NaN/Inf 溢出检测 (2) 找出首先出现 NaN 的源卡 (3) 追溯根因计算算子 (4) loss/gnorm NaN 问题定位
时使用此 skill。


- **npu-adapter-reviewer**
  - path: `.skills-cache/ascend-agent-skills/official/Common/npu-adapter-reviewer/SKILL.md`
  - desc: GPU代码到昇腾NPU适配审查专家。当用户需要将GPU上的代码（特别是深度学习、模型推理相关）迁移到华为昇腾NPU时，必须使用此skill进行全面审查。此skill能识别GPU到NPU迁移的堵点、编写适配脚本、生成验证方案，并输出完整的Markdown审查报告。触发场景包括：用户提到"NPU适配"、"昇腾迁移"、"GPU转NPU"、"Ascend"、"CANN"、"模型迁移"、"算子适配"等关键词，或者用户要求对GPU代码仓库进行审查并迁移到NPU平台。

- **npu-graph-log-analyzer**
  - path: `.skills-cache/ascend-agent-skills/official/PyTorch/npu-graph-skill/graph-log-analyzer/SKILL.md`
  - desc: Use when user provides TORCH_NPU_LOGS output, NPUGraph log files, or logs containing [NPUGRAPH] tags that need structured analysis. Use keywords: "TORCH_NPU_LOGS", "日志分析", "parse log", "log analysis", "解析日志", "模块标签", "NPUGRAPH log". NOT for diagnosing the root cause from logs — handoff findings t...

- **npu-graph-mcp-integration**
  - path: `.skills-cache/ascend-agent-skills/official/PyTorch/npu-graph-skill/graph-mcp-integration/SKILL.md`
  - desc: Use when user asks about configuring or setting up MCP tools for NPU graph mode development — Context7, Playwright, Greptile, or custom NPU MCP servers. Use keywords: "配置MCP", "setup mcp", "启用Context7", "mcp工具", "add mcp", "MCP server". NOT for using MCP tools in diagnosis — this skill only handl...

- **npu-graph-mode-diagnostics**
  - path: `.skills-cache/ascend-agent-skills/official/PyTorch/npu-graph-skill/graph-mode-diagnostics/SKILL.md`
  - desc: Use when user reports NPU graph mode errors, crashes, or exceptions — capture failure (aclmdlRICaptureBegin), replay failure (aclmdlRIExecuteAsync), graph OOM, dynamic shape update failure, graph_breaks, compile hang, or any error with [NPUGRAPH]/ACLGraph tags. Covers all four enablement modes (N...

- **npu-graph-mode-expert**
  - path: `.skills-cache/ascend-agent-skills/official/PyTorch/npu-graph-skill/graph-mode-expert/SKILL.md`
  - desc: Use when user asks conceptual questions about NPU graph mode — how NPUGraph/ACLGraph works, capture/replay lifecycle, graph tree management, update mechanism, FA3 graph integration, super kernel, memory model. Also use when user needs code navigation — locating specific files/functions in torch_n...

- **npu-graph-performance-profiling**
  - path: `.skills-cache/ascend-agent-skills/official/PyTorch/npu-graph-skill/graph-performance-profiling/SKILL.md`
  - desc: Use when user reports NPU graph mode performance issues — slow capture, replay regression, QPS drop, compile overhead, memory inefficiency, or requests profiling/bottleneck analysis. Use keywords: "图模式慢", "性能回退", "regression", "profile", "耗时分析", "compare performance", "benchmark", "QPS下降", "bottl...

- **npu-graph-skill**
  - path: `.skills-cache/ascend-agent-skills/official/PyTorch/npu-graph-skill/SKILL.md`
  - desc: NPU 图模式技能套件。Use when user asks about NPU graph mode — NPUGraph, ACLGraph, capture/replay, graph compile, graph OOM, graph update, FA3 graph, graph performance, graph logs, MCP config. Covers torch_npu full stack (Python API -> C++ core -> op-plugin -> ACL runtime).

- **npu-model-migration**
  - path: `.skills-cache/ascend-agent-skills/official/MindSeriesSDK/RecSDK/npu-model-migration/SKILL.md`
  - desc: 自动化将 PyTorch 模型迁移到华为昇腾 NPU。Use when: 用户请求将模型迁移到 NPU、适配 NPU、在 NPU 上跑通模型、迁移到昇腾。

- **npu-smi**
  - path: `.skills-cache/ascend-agent-skills/official/Common/npu-smi/SKILL.md`
  - desc: Huawei Ascend NPU npu-smi command reference. Use for device queries (health, temperature, power, memory, processes, ECC), configuration (thresholds, modes, fan), firmware upgrades (MCU, bootloader, VRD), virtualization (vNPU), and certificate management.

- **op-adaptation**
  - path: `.skills-cache/ascend-agent-skills/official/PyTorch/op-adaptation/SKILL.md`
  - desc: Integrate CANN (Ascend) operators into torch_npu backend. Two modes: (1) CREATE — new operators from .md + _def.cpp files; (2) MODIFY — existing operators with interface changes, new variants, backward supplements, or bug fixes. Use when the user asks to add, adapt, modify, or fix CANN/aclnn oper...

- **op-apidoc**
  - path: `.skills-cache/ascend-agent-skills/official/PyTorch/op-apidoc/SKILL.md`
  - desc: Generate Chinese API reference documentation for torch_npu custom operators, following the standard structure used in docs/zh/custom_APIs/torch_npu/. Use when the user asks to write or generate API docs for a torch_npu operator. Covers product support tables, function signatures, parameter descri...

- **op-mfu-calculator**
  - path: `.skills-cache/ascend-agent-skills/official/MindStudio/skills/op-mfu-calculator/SKILL.md`
  - desc: 计算算子（如 matmul/GEMM）的 MFU（Machine FLOP Utilization），并给出清晰的公式和推导过程。

- **op-test**
  - path: `.skills-cache/ascend-agent-skills/official/PyTorch/op-test/SKILL.md`
  - desc: Generate and execute UT test cases for torch_npu operators. Use when the user asks to write, create, or generate unit tests for a torch_npu operator (e.g. "write tests for npu_xxx", "generate UT for this operator"). Automatically determines the correct output directory, generates test file with c...

- **pytest-writer**
  - path: `.skills-cache/ascend-agent-skills/official/MindSpeed/mindspeed-llm-auto-ut-skills/skills/pytest-writer/SKILL.md`
  - desc: 专业的pytest测试用例编写助手，用于创建、编写和优化Python测试用例。当需要编写测试文件、创建测试代码、重构优化测试、调试失败测试、使用fixtures、参数化测试、断言技巧、测试覆盖率分析时使用此技能。

- **python-refactoring**
  - path: `.skills-cache/ascend-agent-skills/official/Common/python-refactoring/SKILL.md`
  - desc: Python 代码重构技能，覆盖代码坏味道识别、设计模式应用、可读性改进和实战经验。当用户要求"重构代码"、"refactor"、"代码优化"、"改善代码质量"、"code smell review"、"应用设计模式"、"提升可读性"，或提交代码审查请求时使用此技能。支持在重构完成后输出结构化重构文档（"输出重构文档"、"生成重构报告"）。包含基于 vllm-ascend 仓库 20+ 个真实重构 PR 提炼的实战模式。

- **pytorch-pr-analysis**
  - path: `.skills-cache/ascend-agent-skills/official/PyTorch/pytorch-pr-analysis/SKILL.md`
  - desc: Analyze PyTorch PRs merged during a specified time range and evaluate impacts on torch-npu.

- **quant-tuning-evaluate**
  - path: `.skills-cache/ascend-agent-skills/official/MindStudio/skills/quant-tuning-evaluate/SKILL.md`
  - desc: 执行模型测评。通过 scripts/run_evaluation.py 依据 Evaluation YAML 对量化模型进行评测。

- **quant-tuning-quantize**
  - path: `.skills-cache/ascend-agent-skills/official/MindStudio/skills/quant-tuning-quantize/SKILL.md`
  - desc: 执行模型量化。通过 msmodelslim quant 依据 Practice YAML 对模型进行量化。

- **quantization-accuracy-tuning-orchestrator**
  - path: `.skills-cache/ascend-agent-skills/official/MindStudio/skills/quantization-accuracy-tuning-orchestrator/SKILL.md`
  - desc: End-to-end automated model quantization and accuracy tuning workflow. Use when user asks for automated model quantization and accuracy tuning, e.g. "自动量化", "量化调优", "一键量化", "精度调优", etc.

- **rl-consistency-analysis**
  - path: `.skills-cache/ascend-agent-skills/official/MindStudio/skills/rl-consistency-analysis/SKILL.md`
  - desc: 训练与推理数据不一致的端到端根因分析。当模块映射/值比较不足以定位问题时使用，可追踪首个可信的分歧边界，过滤融合或结构性误报，遵循生产者-消费者链，并生成包含具体假设和证据的根因报告。

- **run-mindspeed-llm-test**
  - path: `.skills-cache/ascend-agent-skills/official/MindSpeed/mindspeed-llm-auto-ut-skills/skills/run-mindspeed-llm-test/SKILL.md`
  - desc: 运行MindSpeed-LLM项目的测试用例。当需要运行测试用例、扫描项目代码覆盖率时调用此技能

- **security-code-review**
  - path: `.skills-cache/ascend-agent-skills/official/Common/security-code-review/SKILL.md`
  - desc: 多语言安全代码审查 (Security Code Review)。对 Python、C++、Shell、Markdown 文件进行系统性安全漏洞检测与修复指导。覆盖 OWASP Top 10、CWE Top 25、CERT 安全编码标准。当用户提及以下内容时，务必使用此技能：安全审查、安全代码审查、security review、code review 中的安全检查、漏洞扫描、安全合规检查（CWE/CERT/OWASP）、编写安全代码、检查代码安全性、推理服务安全审计、多模态 Token 安全校验、JSON 嵌套深度攻击防护。即使用户没有明确说'安全审查'，只要涉及代码安全性评估、漏洞检...

- **shmem-ops-code-gen**
  - path: `.skills-cache/ascend-agent-skills/community/Op/shmem-ops-code-gen/SKILL.md`
  - desc: 根据 design.md 生成 SHMEM 算子代码、目录结构、CMake 和 README。关键词：代码生成、code-gen、实现、kernel、main.cpp。

- **shmem-ops-code-review**
  - path: `.skills-cache/ascend-agent-skills/community/Op/shmem-ops-code-review/SKILL.md`
  - desc: SHMEM 算子实现与设计一致性走读，生成 review-report.md。关键词：code review、design review、一致性检查、走读、交付报告。

- **shmem-ops-compile-debug**
  - path: `.skills-cache/ascend-agent-skills/community/Op/shmem-ops-compile-debug/SKILL.md`
  - desc: 编译、运行和调试 SHMEM 算子。关键词：编译、compile、debug、build、运行、link、失败定位。

- **shmem-ops-correctness-eval**
  - path: `.skills-cache/ascend-agent-skills/community/Op/shmem-ops-correctness-eval/SKILL.md`
  - desc: 执行 SHMEM 算子正确性契约验证并生成报告。关键词：正确性验证、correctness、测试执行、精度验证。

- **shmem-ops-design**
  - path: `.skills-cache/ascend-agent-skills/community/Op/shmem-ops-design/SKILL.md`
  - desc: 设计 SHMEM 通信算子。将需求转化为 design.md。关键词：设计、design、DSL、capability mapping、gap analysis、contract。

- **shmem-ops-dev**
  - path: `.skills-cache/ascend-agent-skills/community/Op/shmem-ops-dev/SKILL.md`
  - desc: SHMEM 通信算子端到端开发编排器。用户 @ 指定本 skill 后：先 Read askquestion-template.md，再 verbatim AskQuestion 五项（零跳过）。关键词：shmem、.agents/skills、Phase 0 intake。

- **shmem-ops-performance-eval**
  - path: `.skills-cache/ascend-agent-skills/community/Op/shmem-ops-performance-eval/SKILL.md`
  - desc: SHMEM 算子性能采集、baseline 对比、聊天自动输出和瓶颈分析。关键词：性能采集、performance、baseline、bandwidth、steady_bus、自动输出、优化轮次。

- **shmem-ops-performance-optim**
  - path: `.skills-cache/ascend-agent-skills/community/Op/shmem-ops-performance-optim/SKILL.md`
  - desc: SHMEM 算子性能优化迭代（Phase 6.5，须 performance_auto_optim:true 且未达标）；机制优化轮次由 design.md performance.max_opt_rounds 控制（默认 5，上限 5）；每轮 MUST 聊天自动输出 Δ%。关键词：性能优化、optimization、steady_bus、自动输出。

- **shmem-ops-testcase-gen**
  - path: `.skills-cache/ascend-agent-skills/community/Op/shmem-ops-testcase-gen/SKILL.md`
  - desc: 生成 SHMEM 算子正确性测试计划、case matrix、golden/checker 和测试脚本。关键词：测试用例、case matrix、golden、checker、gen_data、check_result。

- **shmem-ops-torch-bind**
  - path: `.skills-cache/ascend-agent-skills/community/Op/shmem-ops-torch-bind/SKILL.md`
  - desc: 将 SHMEM 算子封装为 PyTorch CustomClass，生成 Python 测试脚本并验证正确性。关键词：torch、PyTorch、CustomClass、torch_binding、python_extension、接入 torch。

- **simple-vector-triton-gpu-to-npu**
  - path: `.skills-cache/ascend-agent-skills/community/Op/simple-vector-triton-gpu-to-npu/SKILL.md`
  - desc: 将简单Vector类型Triton算子从GPU迁移到昇腾NPU。当用户需要迁移Triton代码到NPU、提到GPU到NPU迁移、Triton迁移、昇腾适配时使用。注意：无法自动迁移存在编译问题的算子。

- **skill-auditor**
  - path: `.skills-cache/ascend-agent-skills/official/Common/skill-auditor/SKILL.md`
  - desc: Comprehensive security auditor for AI agent skills, prompts, and instructions. Checks for typosquatting, dangerous permissions, prompt injection, supply chain risks, and data exfiltration patterns — before you use any agent or skill.

- **ssh-dev-suite**
  - path: `.skills-cache/ascend-agent-skills/official/MindSpeed/drivingsdk-ascend-model-migration/ssh-connection/SKILL.md`
  - desc: SSH远程开发套件，连接管理、命令执行、文件传输、部署、隧道、调试

- **ssh-dev-suite/connect**
  - path: `.skills-cache/ascend-agent-skills/official/MindSpeed/drivingsdk-ascend-model-migration/ssh-connection/connect/SKILL.md`
  - desc: SSH连接管理、远程命令执行、文件传输、后台任务

- **ssh-dev-suite/debug**
  - path: `.skills-cache/ascend-agent-skills/official/MindSpeed/drivingsdk-ascend-model-migration/ssh-connection/debug/SKILL.md`
  - desc: 结构化远程服务器问题排查流程，支持上下文感知的环境检查和容器内调试

- **ssh-dev-suite/deploy**
  - path: `.skills-cache/ascend-agent-skills/official/MindSpeed/drivingsdk-ascend-model-migration/ssh-connection/deploy/SKILL.md`
  - desc: 本地项目部署到远程服务器，支持增量同步、部署钩子、回滚

- **ssh-dev-suite/long-task**
  - path: `.skills-cache/ascend-agent-skills/official/MindSpeed/drivingsdk-ascend-model-migration/ssh-connection/long-task/SKILL.md`
  - desc: 长耗时任务管理，支持checkpoint记忆、agent休息与恢复

- **ssh-dev-suite/tunnel**
  - path: `.skills-cache/ascend-agent-skills/official/MindSpeed/drivingsdk-ascend-model-migration/ssh-connection/tunnel/SKILL.md`
  - desc: SSH通道管理，支持本地/远程端口转发、SOCKS代理、反向代理

- **step0-check**
  - path: `.skills-cache/ascend-agent-skills/official/PyTorch/pytorch-pr-analysis/.pr-analysis/skills/step0-check/SKILL.md`
  - desc: Step 0 强制前置检查。检查 Token、解析时间区间、确认参考文件。

- **step3-analyze**
  - path: `.skills-cache/ascend-agent-skills/official/PyTorch/pytorch-pr-analysis/.pr-analysis/skills/step3-analyze/SKILL.md`
  - desc: Step 3 AI 分析主控。读取 PR 数据，分批并行调用子代理分析，合并结果。

- **step3-batch-analyze**
  - path: `.skills-cache/ascend-agent-skills/official/PyTorch/pytorch-pr-analysis/.pr-analysis/skills/step3-batch-analyze/SKILL.md`
  - desc: Step 3 子代理：分析一批 PR（20条），逐条生成 impact_analysis 等字段。

- **step3-quality-ref**
  - path: `.skills-cache/ascend-agent-skills/official/PyTorch/pytorch-pr-analysis/.pr-analysis/skills/step3-quality-ref/SKILL.md`
  - desc: Step 3 质量参考：高质量案例和判定标准。被 step3-batch-analyze 引用。

- **step3-rules**
  - path: `.skills-cache/ascend-agent-skills/official/PyTorch/pytorch-pr-analysis/.pr-analysis/skills/step3-rules/SKILL.md`
  - desc: Step 3 分析规则：功能模块判定、版本归属、影响级别判定。被 step3-batch-analyze 引用。

- **step5r-review**
  - path: `.skills-cache/ascend-agent-skills/official/PyTorch/pytorch-pr-analysis/.pr-analysis/skills/step5r-review/SKILL.md`
  - desc: Step 5.7 适配/验证/优先级评估。所有信息回填后，AI 子代理分批为每条 PR 判定 need_adaptation/need_verification/priority，原地写回 analysis.json 供 Excel。impact_tag 不在此重新评估。

- **swanlab-setup**
  - path: `.skills-cache/ascend-agent-skills/official/Common/swanlab-setup/SKILL.md`
  - desc: SwanLab 实验追踪平台配置与登录管理。触发场景：(1) 配置 SwanLab 登录凭据 (2) 在容器内安装/登录 SwanLab (3) 为指定容器配置 SwanLab (4) 检查 SwanLab 连接状态。支持多种配置获取方式：环境变量、配置文件、交互式输入。可被其他 skill 通过 source scripts/functions.sh 调用。

- **tilelang-vector-ascend-ops-migration**
  - path: `.skills-cache/ascend-agent-skills/community/Op/tilelang-vector-ascend-ops-migration/SKILL.md`
  - desc: 指导 TileLang 算子从 GPU（CUDA）迁移到华为昇腾 NPU，分析现有实现并生成对应的 NPU 适配代码。

- **torch-npu-api-target-analyse**
  - path: `.skills-cache/ascend-agent-skills/official/PyTorch/torch-npu-api-target-analyse/SKILL.md`
  - desc: 独立的 Torch-NPU API 前期静态任务分析技能：读取一个或多个 API，准备 PyTorch 与 torch-npu 六个版本 source，检索官方定义、测试、NPU 适配、release patch 和 native docs，按规则分类 issue-only、upstream-patch、new-test 或 pending，并生成供外部开发者执行的任务分析说明.md。用于 API 补齐前的场景判断和交付清单；不修改代码、测试、docs、Issue/PR，不运行 API 或 NPU 验证。

- **torch-npu-diagnose**
  - path: `.skills-cache/ascend-agent-skills/official/PyTorch/torch-npu-issue-pipeline-distributed/skills/diagnose/SKILL.md`
  - desc: Diagnoses root causes of PyTorch UT failures on NPU and provides fix strategies. Routes failures to 5 root cause categories: missing dispatch, behavior mismatch, memory issue, env/build issue, upstream PyTorch issue. When the case touches torch.compile/Inductor (inductor/dynamo keywords), routes ...

- **torch-npu-doc-writer**
  - path: `.skills-cache/ascend-agent-skills/official/PyTorch/torch-npu-doc-writer/SKILL.md`
  - desc: 为 Ascend torch-npu API 的多版本 NPU 测试适配一次性生成一对 Markdown 文档：Issue 分析报告和 PR 描述。要求明确提供本地 PyTorch 项目和 torch-npu Git 仓库；task_report 可选，提供时优先参考，并在信息不全或结论矛盾时使用本地 PyTorch 源码与测试补充、核验。测试输入支持测试目录/文件，以及弃用、空实现或无需 NPU 专用适配等无测试特殊说明。支持 torch-npu commit、patch/diff、完整 Python 测试和日志；不运行测试、不提交代码、不创建 PR、不修改项目仓库，也不分析 PyTo...

- **torch-npu-graph-mode-inductor**
  - path: `.skills-cache/ascend-agent-skills/official/PyTorch/torch-npu-issue-pipeline-distributed/skills/tools/graph-mode-inductor/SKILL.md`
  - desc: Forces inspection of torch.compile / Inductor debug artifacts (torch_compile_debug/) when a failing NPU test case touches graph-mode paths (torch/_inductor, torch/_dynamo, torch/_functorch, torch/_export, torch/_higher_order_ops), and cross-references torch_npu Inductor lowering / fallback config...

- **torch-npu-issue-pipeline-core**
  - path: `.skills-cache/ascend-agent-skills/official/PyTorch/torch-npu-issue-pipeline-core/SKILL.md`
  - desc: Use when the user needs PyTorch-NPU 社区用例解单（Core组）for an upstream PyTorch UT failure on Ascend NPU, with setup, reproduce, diagnose, verify, and delivery handled as one workflow.

- **torch-npu-issue-pipeline-distributed**
  - path: `.skills-cache/ascend-agent-skills/official/PyTorch/torch-npu-issue-pipeline-distributed/SKILL.md`
  - desc: Use when the user needs PyTorch-NPU 社区用例解单（分布式或图模式）, especially for distributed or HCCL-related upstream PyTorch UT failures, or graph-mode failures that follow the reproduce → diagnose → verify workflow.

- **torch-npu-pipeline**
  - path: `.skills-cache/ascend-agent-skills/official/PyTorch/torch-npu-issue-pipeline-distributed/skills/SKILL.md`
  - desc: Use when the user asks for PyTorch-NPU 社区用例解单（分布式）or end-to-end repair of a distributed/HCCL-related failing upstream PyTorch test case, especially when they provide a full parameterized case name, issue log, or ask to run the reproduce → diagnose → verify pipeline.

- **torch-npu-pr-review**
  - path: `.skills-cache/ascend-agent-skills/official/PyTorch/torch-npu-pr-review/SKILL.md`
  - desc: 审核 Ascend torch-npu PR，支持单PR审核和多PR批量审核+跨分支对比，基于验收标准规范、其他要求规范和注意点进行逐项检查，输出检验报告。

- **torch-npu-remote-runner**
  - path: `.skills-cache/ascend-agent-skills/official/PyTorch/torch-npu-remote-runner/SKILL.md`
  - desc: 在 Windows、Linux 或 macOS 本地通过 Python 和 OpenSSH，对已配置 SSH 免密登录的远程 Linux 昇腾服务器执行 torch-npu 多版本一键验证：校验非交互 SSH，上传按版本目录组织的测试和内置 server runner，在远程宿主机或用户指定 Docker 容器内自动发现同名 conda 环境并执行测试，下载本轮日志后分析结果。用户要求从本地电脑远程上传、运行、下载并分析 torch-npu 测试，或要求远端宿主机/Docker 多版本验证时使用；不用于普通 SSH、密码登录、单条远程命令、容器创建或非测试任务。

- **torch-npu-reproduce**
  - path: `.skills-cache/ascend-agent-skills/official/PyTorch/torch-npu-issue-pipeline-distributed/skills/reproduce/SKILL.md`
  - desc: Reproduces PyTorch upstream UT failures on NPU (Ascend) devices. Converts CUDA-based test cases to NPU-runnable form, executes them, and generates structured reproduction reports. Use when user says "reproduce", "复现", "NPU test failure", "UT fails on NPU", "run PyTorch test on NPU", or "test_xxx ...

- **torch-npu-server-runner**
  - path: `.skills-cache/ascend-agent-skills/official/PyTorch/torch-npu-server-runner/SKILL.md`
  - desc: 在已经登录的 Linux 昇腾服务器上，通过宿主机或用户指定 Docker 容器内各版本 conda 环境的 Python 解释器，批量执行按版本目录组织的 torch-npu Python 测试，将输出写入测试根目录的 result_log，并分析本轮测试结果。用户要求直接运行 torch-npu 多版本测试、在指定容器中执行测试方法、收集或分析服务器本轮测试日志时使用；实际执行后必须读取 result_log，全部通过时仅报告结果，存在失败时区分测试用例、API 实现、环境或执行问题并给出解决方案；不需要服务器地址、登录用户名、SSH 或文件传输参数，不用于远程执行、普通 conda...

- **torch-npu-verify**
  - path: `.skills-cache/ascend-agent-skills/official/PyTorch/torch-npu-issue-pipeline-distributed/skills/verify/SKILL.md`
  - desc: Verifies code fixes for PyTorch-NPU adaptation issues. Runs regression tests on the originally failing case, validates neighbor test cases, performs build verification if needed, and generates PR-ready documentation. Use when user says "verify", "验证", "regression test", "回归测试", "check fix", "gene...

- **tune-practice-cfg**
  - path: `.skills-cache/ascend-agent-skills/official/MindStudio/skills/tune-practice-cfg/SKILL.md`
  - desc: Use when 量化调优闭环中需要生成或修改一轮调优所需的 Practice YAML，包括敏感层分析、策略决策、写出 YAML 文件和校验。

- **unittest-writer**
  - path: `.skills-cache/ascend-agent-skills/official/MindSpeed/mindspeed-llm-auto-ut-skills/skills/unittest-writer/SKILL.md`
  - desc: Python unittest 框架的专业测试用例编写助手。用于创建、编写和优化 Python 单元测试，包括测试用例结构、断言方法、测试组织、setUp/tearDown 模式以及命令行执行。当需要编写测试文件、创建测试代码、重构优化测试、调试失败测试时使用此技能。

- **vLLM-ascend_FAQ_Generator**
  - path: `.skills-cache/ascend-agent-skills/official/vllm-ascend/vLLM-ascend_FAQ_Generator/SKILL.md`
  - desc: 为 vLLM-ascend 项目构建自动化工作流，处理已关闭的Issue并生成Debug FAQ。Use when users want to process closed issues from vLLM-ascend repository, generate debug FAQ, categorize issues, or analyze issue patterns.

- **vector-triton-ascend-ops-optimizer**
  - path: `.skills-cache/ascend-agent-skills/community/Op/vector-triton-ascend-ops-optimizer/SKILL.md`
  - desc: 昇腾（Ascend） NPU 上 Triton 算子深度性能优化技能（Skill），致力于实现用户要求的 Triton 算子性能提升。核心技术包括但不限于 Unified Buffer (UB) 容量规划、多 Tokens 并行处理、MTE/Vector 流水并行、mask（掩码）优化等。当用户提及以下内容时，务必触发此技能（Skill）：昇腾（Ascend）NPU 上 Vector 类 Triton 算子性能优化。

- **verl-async-dapo**
  - path: `.skills-cache/ascend-agent-skills/official/verl/verl-async-dapo/SKILL.md`
  - desc: Verl 单异步 DAPO 训练配置生成器。触发场景：(1) 启动单异步 DAPO 训练 (2) 生成训练脚本 (3) 配置特性参数 (4) 训练前检查。**特性策略**：用户未指定时默认开启性能特性（flash_attn/dynamic_batch/remove_padding/gradient_checkpointing），显存特性（offload/recompute）默认关闭。OOM 时自动追加显存特性重试。**训练监控**：启动后输出 SwanLab 链接供用户自行查看，仅在错误时通知用户。**依赖 skill**：SwanLab 配置通过 swanlab-setup skill...

- **verl-deploy**
  - path: `.skills-cache/ascend-agent-skills/official/verl/verl-feature-deploy/SKILL.md`
  - desc: Verl 分布式训练服务一键拉起与配置。触发场景：(1) 用户要启动 Verl 训练任务或部署 RLHF/DAPO 训练环境 (2) 在 NPU 集群上拉起 Verl 训练容器 (3) 配置 Ray 集群和 SwanLab 监控 (4) 根据 7 位二进制掩码灵活配置加速特性。支持 Qwen3-8B 等 Megatron 模型的 DAPO 训练全流程。

- **vllm-ascend-deploy**
  - path: `.skills-cache/ascend-agent-skills/official/vllm-ascend/vllm-ascend-deploy/SKILL.md`
  - desc: 昇腾 NPU 平台 vLLM 大模型推理服务一键部署。触发：用户说'部署 模型名'、'NPU 部署模型'、'vllm serve'。流程：SSH检查 → NPU检查 → 配置发现(必须验证) → 用户确认 → 部署 → cron监控 → 验证。约束：(1) 配置必须从官方文档验证，禁止猜测；(2) 后台启动必须用cron监控，禁止手动轮询。支持 Qwen/Qwen3.5、GLM、DeepSeek、Kimi。

- **vllm-tests-failure-analysis**
  - path: `.skills-cache/ascend-agent-skills/official/vllm-ascend/vllm-tests-failure-analysis/SKILL.md`
  - desc: Analyze and debug upstream vLLM test failures on Ascend NPUs. Adapt test cases from `vllm/tests/` for the vllm-ascend plugin, and identify tests that are compatible with the `vllm-ascend` continuous integration (CI) pipeline. This skill should be used to analyze whether upstream vLLM mainline tes...

### Source: cannbot-skills (189 skills)

- **aiss-tiling-solver**
  - path: `.skills-cache/cannbot-skills/ops/aiss-tiling-solver/SKILL.md`
  - desc: 使用 AISS-TilingSolver 工具自动求解 Ascend C 算子（MatMul / Vector）的最优 Tiling 参数，包括下载安装、构造 JSON 输入、运行求解、结果解读与故障排查。触发：当用户使用 TilingSolver 工具求解时。

- **aog-a3-author**
  - path: `.skills-cache/cannbot-skills/plugins-community/ascendc-port-orchestrator/skills/aog-a3-author/SKILL.md`
  - desc: Generate per-op `run_a3_reference.py` + `input_gen.py` + `manifest.json` for an arch22→arch35 migration workspace by parsing the upstream ops-nn op directory's `examples/test_aclnn_{op}.cpp` (aclnn signature) and `op_host/{op}_proto.cpp` (output shape inference). Eliminates the per-op manual-auth...

- **aog-input-gen-builder**
  - path: `.skills-cache/cannbot-skills/plugins-community/ascendc-port-orchestrator/skills/aog-input-gen-builder/SKILL.md`
  - desc: Phase O2.5 input_gen.py + edge dataset generator for the bundled orchestrator. Reads a source-architecture AscendC package or a differentiable forward spec, infers the case_gen SCHEMA (tensor_inputs, scalar_inputs, shape_derive, invariants), and emits a ready-to-run input_gen.py under `workspace/...

- **aog-knowledge-maintain**
  - path: `.skills-cache/cannbot-skills/plugins-community/ascendc-port-orchestrator/skills/aog-knowledge-maintain/SKILL.md`
  - desc: Review AscendC runtime findings and stage user-local c-tier entries; audit the release-owned bundled knowledge base without mutating it. Four modes: "update" (fast), "scan" (thorough), "validate" (single entry), "learn" (web scraping). Use when an operator run produces new knowledge or the Ascend...

- **aog-op-classify**
  - path: `.skills-cache/cannbot-skills/plugins-community/ascendc-port-orchestrator/skills/aog-op-classify/SKILL.md`
  - desc: Classify an op by reading its source (Python / PyTorch / AscendC) and emit `op_classification.json` with KB recommendations as the load-bearing output. Invoked by the Python orchestrator at Phase O1.7, or by a human for one-off inspection: `Skill(name="aog-op-classify", args="workspace/{op}")`. U...

- **aog-perf-eval**
  - path: `.skills-cache/cannbot-skills/plugins-community/ascendc-port-orchestrator/skills/aog-perf-eval/SKILL.md`
  - desc: AscendC 算子双方法性能评估（profiler + wall clock）。适用于 op-gen 输出目录（output/{project}/src/kernels/{op}/）。 使用 torch_npu.profiler（V5: kernel_details.csv, AI_VECTOR_CORE/AI_CORE 含 TensorMove）采集 device 侧 kernel 时间， 同时使用 time.perf_counter() wall-clock 采集 host 侧端到端时间。生成双方法并排对比的 HTML 报告。 调用方式: /aog-perf-eval {outpu...

- **aog-prior-art-verify**
  - path: `.skills-cache/cannbot-skills/plugins-community/ascendc-port-orchestrator/skills/aog-prior-art-verify/SKILL.md`
  - desc: Use when an arch22→arch35 migration can reuse an existing arch35 candidate. Scan, provenance-stage, build, measure, and learn from it without replacing fresh arch22 truth capture or customer-facing verification.

- **aog-report-gen**
  - path: `.skills-cache/cannbot-skills/plugins-community/ascendc-port-orchestrator/skills/aog-report-gen/SKILL.md`
  - desc: Generate or refresh a project-level REPORT.md for an output/{project}/ directory (cross-generation migration or backward-generation project). Wraps src/scripts/gen_report_tables.py (table injection via <!-- BEGIN-GEN:* --> markers) and the canonical 9-section structure defined in OUTPUT_PROJECT_L...

- **aog-self-critic**
  - path: `.skills-cache/cannbot-skills/plugins-community/ascendc-port-orchestrator/skills/aog-self-critic/SKILL.md`
  - desc: Self-supervision skill — invoke to audit the current working session against recurring failure patterns user has had to correct across prior sessions. Goal: catch reward-hacking, priority drift, infrastructure bypass, and premature-conclusion smells BEFORE the user has to correct them again. Use ...

- **ascend-call-generation**
  - path: `.skills-cache/cannbot-skills/plugins-community/collaborative-agent-kernel-evolution/skills/ascend-call-generation/SKILL.md`
  - desc: Generate AscendC project scaffold (pybind, CMake, host code, kernel skeleton) from functional PyTorch for pure Vector operators. Use after functional-conversion, before dsl-baseline-generation.

- **ascendc-api-best-practices**
  - path: `.skills-cache/cannbot-skills/ops/ascendc-api-best-practices/SKILL.md`
  - desc: Ascend C API 使用最佳实践。提供算术、归约、数据搬运、Buffer管理、精度转换等 API 的正确用法和限制说明。触发：用户询问具体 API 用法（如"DataCopy 怎么用"）、遇到 API 参数错误或限制报错（如 repeatTimes、对齐问题）、需要查看 API 最佳实践或避坑指南时。

- **ascendc-backward-gen**
  - path: `.skills-cache/cannbot-skills/plugins-community/ascendc-port-orchestrator/skills/ascendc-backward-gen/SKILL.md`
  - desc: 正向→反向 AscendC 算子生成入口。由一个可微 PyTorch 正向算子，自动生成其反向（梯度）AscendC 算子并在 NPU 上验证精度。可用自然语言指定目标芯片（a3/a5 或 arch22/arch35）。 触发：当用户需要为某正向算子生成对应反向算子时使用。


- **ascendc-blaze-best-practice**
  - path: `.skills-cache/cannbot-skills/ops/ascendc-blaze-best-practice/SKILL.md`
  - desc: Blaze/tensor_api 路径的 Matmul 类算子开发指南（Ascend 950 / DAV_3510）。覆盖框架认知、模板目录、开发指南和扩展开发。触发：在 A5 平台开发 matmul 类算子（普通 matmul、MX 量化 matmul、Grouped matmul）及 C+V 模式融合算子（上述三类 matmul + vector epilogue）时。不适用于纯 Vector 算子和 A2/A3 平台。

- **ascendc-blaze-migration**
  - path: `.skills-cache/cannbot-skills/ops/ascendc-blaze-migration/SKILL.md`
  - desc: 将 ops-nn、ops-transformer 等仓中的 Ascend 950 / DAV_3510 Matmul、BatchMatMul、GroupMatmul AscendC、CMCT 或 CGMCT 核函数等价迁移到 ops-tensor Blaze/tensor_api。使用场景：迁移边界、tiling/ABI 冻结、GM_ADDR/ListTensor、Blaze 规范事实源、Scheduler/Kernel/BlockMmad/Epilogue 复用或扩展、CMCT/CGMCT 清理、逐字节一致性、性能和双仓 PR 门禁。证据不全为待确认（unknown），冲突为受阻（bl...

- **ascendc-code-review**
  - path: `.skills-cache/cannbot-skills/ops/ascendc-code-review/SKILL.md`
  - desc: Ascend C 代码检视技能。触发：检视代码、检视 PR、检查是否有问题、快速检视。支持文件检视、PR 检视、大型PR自动切换、快速定向检视、设计一致性检查。自动识别代码侧别、提取适用条例、执行假设检验驱动的逐条检视。

- **ascendc-crash-debug**
  - path: `.skills-cache/cannbot-skills/ops/ascendc-crash-debug/SKILL.md`
  - desc: Ascend C 算子卡死/崩溃/内存错误调试路由技能。用于处理程序无法运行完或执行异常崩溃的场景：(1) 程序卡死/挂起/超时，Kernel 无响应，(2) 程序崩溃（Segmentation Fault、Abort），(3) Buffer 冲突/死锁导致的核心挂起，(4) 需要解析 plog 日志定位卡死/崩溃位置，(5) 偶发崩溃/结果异常（怀疑内存越界踩踏），需要主动检测内存错误。触发关键词：卡死、挂起、超时、崩溃、hang、crash、deadlock、Segmentation Fault、Abort、Kernel hang、内存越界、plog、memcheck、内存检查、内存...

- **ascendc-cross-gen-port**
  - path: `.skills-cache/cannbot-skills/plugins-community/ascendc-port-orchestrator/skills/ascendc-cross-gen-port/SKILL.md`
  - desc: 跨代际 AscendC 算子移植入口。把一个 AscendC 算子从来源架构移植到用户指定的目标架构/产品 （当前 arch22→arch35，如 910C/V220→950PR/V300；规划更多目标与反向跨代）。用户用自然语言 指定目标（arch35 / 950PR / A5 / SoC编号 / 代际皆可）；来源架构由代码分析自动识别。 触发：当用户需要把某 AscendC 算子跨代际移植到指定目标架构时使用。


- **ascendc-direct-invoke-template**
  - path: `.skills-cache/cannbot-skills/ops/ascendc-direct-invoke-template/SKILL.md`
  - desc: Kernel直调工程模板，用于创建 Ascend C Kernel 直调工程项目。提供经过验证的 Vector 样例工程（add_custom）、Blaze Matmul 工程模板（纯 Matmul / 融合 / MX 量化 / GroupMatmul）和 Kirin Vector 模板。触发：当用户需要创建 Kernel 直调工程、学习 Ascend C 编程、快速原型验证、或提及"Kernel直调"、"<<<>>>内核调用"、"Blaze Matmul"、"matmul 模板"时使用本 skill。

- **ascendc-direct-invoke-to-registry-invoke**
  - path: `.skills-cache/cannbot-skills/ops/ascendc-direct-invoke-to-registry-invoke/SKILL.md`
  - desc: 当用户想把`<<<>>>` kernel 直调形式改造成自定义算子工程时使用。触发：用户提到"kernel直调转自定义算子"、"为kernel直调工程接入ACLNN/GEIR接口"、"`<<<>>>` 改自定义算子工程"等。不适用于从零开发新算子

- **ascendc-docs-gen**
  - path: `.skills-cache/cannbot-skills/ops/ascendc-docs-gen/SKILL.md`
  - desc: Ascend C 算子文档写作参考。提供需求分析、详细设计、迭代计划、aclnnAPI接口文档、算子README的标准模板。当用户需要生成算子文档、aclnnAPI文档、算子README、参考文档模板或了解算子文档规范时触发此技能。

- **ascendc-docs-search**
  - path: `.skills-cache/cannbot-skills/ops/ascendc-docs-search/SKILL.md`
  - desc: Ascend C 开发资源检索技能。通过本地 API 文档索引、示例代码映射和在线文档兜底搜索定位开发资料，优先查本地、缺失时再查在线。当需要查询 API 用法、示例代码、兼容性信息、官方资料入口或定位文档来源时使用。

- **ascendc-env-check**
  - path: `.skills-cache/cannbot-skills/ops/ascendc-env-check/SKILL.md`
  - desc: Ascend C 算子开发环境检查技能。用于：(1) 通过 npu-smi 查询 NPU 设备信息（设备列表、状态、资源使用），(2) 检查 CANN 环境配置（CANN Toolkit、Ops、自定义算子包），(3) 验证开发依赖是否完整，(4) 运行时检测当前设备 NPU 架构。触发关键词：环境检查、NPU设备、npu-smi、CANN安装、设备查询、资源监控、检查CANN环境变量、NPU架构、npu arch。

- **ascendc-evaluation**
  - path: `.skills-cache/cannbot-skills/plugins-community/collaborative-agent-kernel-evolution/skills/ascendc-evaluation/SKILL.md`
  - desc: Multi-case operator evaluation with precision testing and performance profiling, use when you want to evaluate the performance and correctness of your AscendC operator implementation.

- **ascendc-mc2-best-practice**
  - path: `.skills-cache/cannbot-skills/ops/ascendc-mc2-best-practice/SKILL.md`
  - desc: Ascend C MC2 通算融合算子（多卡通信+计算融合 Kernel 直调）开发最佳实践。仅支持 Ascend 950（npu-arch=dav-3510）。当用户需要开发 MC2 通算融合算子、多卡通信+计算融合的 Kernel 直调算子，或提及"MC2"、"SHMEM"、"通算融合"、"多卡通信直调"、"UDMA"、"URMA"、"AllToAll+Matmul"、"多卡集合通信直调"时必须使用。强制约束：通信走 SHMEM（禁止 HCCL 高阶 API），Matmul 走 Blaze 模板（禁止 asc-devkit matmul API），开发必须按 CANNBot AGEN...

- **ascendc-op-debug**
  - path: `.skills-cache/cannbot-skills/plugins-community/collaborative-agent-kernel-evolution/skills/ascendc-op-debug/SKILL.md`
  - desc: Unified AscendC operator runtime debug skill. Diagnose precision errors, runtime crashes, hangs, and multicore inconsistencies using a hypothesis-driven protocol with 3-layer evidence (code review → log analysis → tools). Covers hypothesis patterns for Vector operators (cache line conflict, works...

- **ascendc-perf-optimize**
  - path: `.skills-cache/cannbot-skills/ops/ascendc-perf-optimize/SKILL.md`
  - desc: Ascend C 算子性能优化策略制定。结合 Tiling 建模与流水分析（仿真图 + profiling 数据），按卡间/核间/核内三层流水制定性能优化策略，并回修 Tiling 参数。触发：算子性能调优、流水分析、Tiling 修正、bound 诊断、MC² 通算融合算子优化、卡间流水配平时。

- **ascendc-performance-best-practices**
  - path: `.skills-cache/cannbot-skills/ops/ascendc-performance-best-practices/SKILL.md`
  - desc: Ascend C 算子性能优化最佳实践库。按算子族组织优化经验与参考代码总结，供性能优化实施阶段查询。触发：查询某类算子的性能优化参考实现、实施某项优化时需加载对应优化经验时。

- **ascendc-precision-debug**
  - path: `.skills-cache/cannbot-skills/ops/ascendc-precision-debug/SKILL.md`
  - desc: Ascend C 算子精度调试技能，提供精度问题诊断和解决方法。触发：输出异常（全为0、随机值、未初始化）、精度验证失败（rtol/atol 不达标）、FP16 精度差于预期、Cast 后数据错误、需要排查流水线同步（EnQue/DeQue）或 DataCopy 对齐问题。

- **ascendc-regbase-best-practice**
  - path: `.skills-cache/cannbot-skills/ops/ascendc-regbase-best-practice/SKILL.md`
  - desc: 当需要为 DAV_3510 RegBase 算子确认 API 约束、实现结构、排查常见陷阱或选择真实参考算子时使用。

- **ascendc-registry-invoke-template**
  - path: `.skills-cache/cannbot-skills/ops/ascendc-registry-invoke-template/SKILL.md`
  - desc: 完整自定义算子工程模板。通过提供标准工程结构、代码模板、UT/ST 样例和多芯片架构参考，帮助快速搭建并实现 registry-invoke 方式的自定义算子工程。当需要创建完整自定义算子工程、参考标准目录结构、补齐 UT/ST、适配多芯片架构或查找工程样例时使用。

- **ascendc-registry-invoke-to-direct-invoke**
  - path: `.skills-cache/cannbot-skills/ops/ascendc-registry-invoke-to-direct-invoke/SKILL.md`
  - desc: 当用户想把自定义算子工程中的 kernel 模板改造成 `<<<>>>` kernel 直调形式，或从自定义算子工程中抽取某个 kernel 模板并转换成 `<<<>>>` 直调方式时使用。触发：用户提到"自定义算子转直调"、"从算子工程抽 kernel"、"kernel 模板改 `<<<>>>`"等。不适用于从零开发新算子

- **ascendc-runtime-debug**
  - path: `.skills-cache/cannbot-skills/ops/ascendc-runtime-debug/SKILL.md`
  - desc: Ascend C 算子运行时错误调试技能。用于处理算子运行时问题：(1) aclnn 返回错误码（161xxx/361xxx/561xxx，包括环境配置、Tiling、Kernel 查找等错误），(2) 需要解析 plog 日志定位问题。触发关键词：运行时错误、错误码、Tiling错误、Kernel查找失败、环境变量、plog。

- **ascendc-simt-best-practices**
  - path: `.skills-cache/cannbot-skills/ops/ascendc-simt-best-practices/SKILL.md`
  - desc: AscendC SIMT 最佳实践与 API 导航。提供 SIMT 算子开发的实践经验总结和专有 API 分类索引：VF函数声明与调用、线程排布模式、索引计算API、数据搬入搬出、计算指令映射、多核同步、核内共享内存、TensorList处理、调测接口、内置宏等。触发：开发SIMT算子kernel代码、设计SIMT编程模式、查询SIMT API分类或线程排布时。

- **ascendc-simt-tiling-design**
  - path: `.skills-cache/cannbot-skills/ops/ascendc-simt-tiling-design/SKILL.md`
  - desc: AscendC SIMT 算子切分设计指南。提供 SIMT 算子独有的核数切分、线程数设置、DCache/UB空间分配方法。SIMT切分与SIMD完全不同（线程级并行 vs 向量级UB切分），本skill聚焦SIMT切分范式。触发：设计SIMT算子Tiling策略、设置SIMT线程数、规划SIMT核数切分时。

- **ascendc-st-design**
  - path: `.skills-cache/cannbot-skills/ops/ascendc-st-design/SKILL.md`
  - desc: Ascend C 算子系统测试（ST）设计技能。基于 aclnn 接口文档，完成算子参数定义、测试因子提取、约束关系分析、测试用例生成（L0/L1/L2）的完整流程。当需要以下任务时使用此技能：设计算子测试用例、生成ST用例。

- **ascendc-tiling-design**
  - path: `.skills-cache/cannbot-skills/ops/ascendc-tiling-design/SKILL.md`
  - desc: Ascend C 算子 Tiling 设计指南。提供算子分类体系和 Tiling 核心要素（多核切分、UB切分、Buffer规划、分支覆盖）的详细设计方法。触发：算子设计阶段、设计 Tiling 策略（多核切分/UB切分）、规划 Buffer 分配、查阅某类算子的 Tiling 方法论时。

- **ascendc-ut-develop**
  - path: `.skills-cache/cannbot-skills/ops/ascendc-ut-develop/SKILL.md`
  - desc: Ascend C 算子 UT 开发与覆盖率增强技能。通过分析 op_host / op_api / op_kernel 的测试空白、生成或补充 UT 用例并定位未覆盖代码来提升覆盖率并支持生成覆盖率报告。当用户提及 UT、单元测试、覆盖率、补测、未覆盖代码或需要新增/完善 UT 时使用，不适用于 ST 测试。

- **ascendc-whitebox-design**
  - path: `.skills-cache/cannbot-skills/ops/ascendc-whitebox-design/SKILL.md`
  - desc: Ascend C 算子白盒测试用例生成系统。分析算子源码提取参数维度，自动枚举参数组合，生成可执行的白盒测试用例。自动两套输出：low 档位（路径覆盖+网络+空tensor，全normal）与 high 档位（data_range 展开，信息性验证）。触发场景：(1) "为 X 算子生成白盒测试用例" (2) "算子白盒用例生成" (3) "generate whitebox test cases for operator"。

- **cake-code-review**
  - path: `.skills-cache/cannbot-skills/plugins-community/collaborative-agent-kernel-evolution/skills/cake-code-review/SKILL.md`
  - desc: Review AscendC kernel code for structural red-line violations (P0–P3) and algorithm correctness. Use after dsl-lowering, before ascendc-evaluation. Outputs rectification report; guides fix → recompile → unit test → system test.

- **cake-docs-search**
  - path: `.skills-cache/cannbot-skills/plugins-community/collaborative-agent-kernel-evolution/skills/cake-docs-search/SKILL.md`
  - desc: Ascend C 开发资源检索技能。通过本地 API 文档索引、示例代码映射和在线文档兜底搜索定位开发资料，优先查本地、缺失时再查在线。当需要查询 API 用法、示例代码、兼容性信息、官方资料入口或定位文档来源时使用。本技能（含 scripts/ 脚本）源自 cannbot-skills 仓库（gitcode.com/cann/cannbot-skills）的 ops/ascendc-docs-search，重命名而来，为自研代码。

- **cake-evo**
  - path: `.skills-cache/cannbot-skills/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/SKILL.md`
  - desc: Evolutionary AscendC operator generation — spawn parallel variants with different optimization strategies and select the best. Use as the top-level orchestrator for multi-round kernel optimization. 触发：需要多轮并行进化以优化内核性能时。

- **cake-review**
  - path: `.skills-cache/cannbot-skills/plugins-community/collaborative-agent-kernel-evolution/skills/cake-review/SKILL.md`
  - desc: Comprehensive review of the entire kernel generation process after all stages complete. Analyzes problems encountered during generation, compile/environment issues,  and skill/agent document quality. Produces a REVIEW.md in the kernel output directory. Use when a formal review of the kernel gener...

- **cann-env-setup**
  - path: `.skills-cache/cannbot-skills/ops/cann-env-setup/SKILL.md`
  - desc: 昇腾 NPU 上 CANN 安装与环境配置指导。触发场景：需安装 CANN、配置开发环境或排查安装问题时。

- **cannbot-skill-reviewer**
  - path: `.skills-cache/cannbot-skills/infra/cannbot-skill-reviewer/SKILL.md`
  - desc: 审查新提交或修改的 CANNBot Skill 是否符合入库质量要求。当用户需要评审 SKILL.md、检查新增 cannbot skill、审查 GitCode PR 中的技能变更、验证测试/开发/NPU/Ascend 相关 skill 是否合格时使用；输出结构门禁、九维评分、阻塞问题和可执行整改建议。

- **catlass-op-design**
  - path: `.skills-cache/cannbot-skills/ops/catlass-op-design/SKILL.md`
  - desc: Analyze operator requirements and select CATLASS components (ArchTag, DispatchPolicy, TileShape, BlockMmad, BlockEpilogue, BlockScheduler, Kernel type). Use when designing new CATLASS-based Ascend C operators, selecting DispatchPolicy, determining TileShape, choosing Kernel type, or picking Epilo...

- **catlass-op-develop**
  - path: `.skills-cache/cannbot-skills/ops/catlass-op-develop/SKILL.md`
  - desc: Generate CATLASS kernel code from design selections. For prerequisite-reading questions, answer directly: before implementation must read workspace `./catlass/README.md` (library positioning and directory structure), `./catlass/docs/` (operator assembly knowledge and implementation constraints), ...

- **catlass-op-perf-tune**
  - path: `.skills-cache/cannbot-skills/ops/catlass-op-perf-tune/SKILL.md`
  - desc: Tune CATLASS kernel performance by adjusting TileShape, DispatchPolicy, Swizzle, and Kernel type parameters. Change one variable at a time for attribution. Use when optimizing CATLASS kernel performance, analyzing profiler bottlenecks, or exploring tiling configurations.

- **code-performance-advisor**
  - path: `.skills-cache/cannbot-skills/plugins-community/collaborative-agent-kernel-evolution/skills/code-performance-advisor/SKILL.md`
  - desc: Diagnose AscendC kernel performance bottlenecks from profiling data (msprof). Matches expert rules first, falls back to LLM analysis. Use when speedup is below target after correctness passes.

- **dsl-baseline-generation**
  - path: `.skills-cache/cannbot-skills/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-baseline-generation/SKILL.md`
  - desc: Generate initial AscendDSL code (class structure, compute, tiling) from functional PyTorch for pure Vector operators. Use after ascend-call-generation creates the project scaffold.

- **dsl-lowering**
  - path: `.skills-cache/cannbot-skills/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/SKILL.md`
  - desc: Translate the operator DSL into AscendC code through multiple passes. Also used when diagnosing compilation errors.

- **dsl-optimization**
  - path: `.skills-cache/cannbot-skills/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-optimization/SKILL.md`
  - desc: Iteratively optimize AscendDSL code for performance — tiling tuning, vectorization, pipeline adjustments. Use after evaluation shows correctness passes but speedup is below target.

- **evolution-knowledge**
  - path: `.skills-cache/cannbot-skills/plugins-community/ops-perf-evolution/skills/evolution-knowledge/SKILL.md`
  - desc: Domain knowledge base for AscendC evolution optimization covering hardware architecture, algorithm insights, API pitfalls, optimization patterns, and proven solutions for A3 (910B) architecture. 当进行 AscendC 进化优化需要查询硬件架构、算法洞察、API 陷阱、优化模式或已验证方案时使用。

- **evolution-report**
  - path: `.skills-cache/cannbot-skills/plugins-community/ops-perf-evolution/skills/evolution-report/SKILL.md`
  - desc: 进化优化完成后自动生成标准化 HTML 可视化报告（全自动脚本生成，含自检，支持 ops-evo 和 lingxi-evo 两种 pipeline）。当进化轮次结束或需要汇总基线与各轮变体的性能对比、决策树与资源消耗时使用。

- **evolution-strategies**
  - path: `.skills-cache/cannbot-skills/plugins-community/ops-perf-evolution/skills/evolution-strategies/SKILL.md`
  - desc: AscendC kernel optimization strategy library with 61+ strategies across D/P/A/R/X series, supporting tiered retrieval, compatibility checking, and autonomous strategy discovery. 当需要为 AscendC kernel 选择优化策略、按瓶颈/算子族筛选策略、检查策略兼容性或登记进化发现的新策略时使用。

- **evolution-world-model**
  - path: `.skills-cache/cannbot-skills/plugins-community/ops-perf-evolution/skills/evolution-world-model/SKILL.md`
  - desc: World model decision tree tools and reference documentation for evidence-driven AscendC kernel evolution, providing CLI operations (select, validate, summary, deep-profiling) and schema/operations reference. 当进行证据驱动的 AscendC 内核进化优化，需要初始化、选择、验证、更新世界模型决策树或查询其 schema/操作规范时使用。

- **fixer-broken-link**
  - path: `.skills-cache/cannbot-skills/plugins-community/ops-qa-suite/.opencode/skills/fixer-broken-link/SKILL.md`
  - desc: Ascend C 算子仓库 Markdown 断链修复技能。用于扫描 ops-math/ops-nn/ops-transformer/ops-cv 算子仓库的 Markdown 文件断链，自动修复并创建 PR。当用户需要修复文档断链、检查链接有效性时使用。

- **functional-conversion**
  - path: `.skills-cache/cannbot-skills/plugins-community/collaborative-agent-kernel-evolution/skills/functional-conversion/SKILL.md`
  - desc: Convert PyTorch nn.Module reference implementation to stateless functional API (module_fn + get_inputs). Use after reference-generation, before ascend-call-generation.

- **git-version-management**
  - path: `.skills-cache/cannbot-skills/plugins-community/collaborative-agent-kernel-evolution/skills/git-version-management/SKILL.md`
  - desc: Git 版本管理 - 算子工作区初始化、逐阶段提交、worktree 并行隔离与审计追踪

- **gitcode-issue-gen**
  - path: `.skills-cache/cannbot-skills/infra/gitcode-issue-gen/SKILL.md`
  - desc: 根据用户输入自动判断走两条路径之一：(PR路径) 用户提供 GitCode PR 链接时，按变更类型自动选用 Issue 模板，通过 GitCode API 创建 Issue 并完成 PR ↔ Issue 双向关联；(手动路径) 用户直接描述问题或要求"提 Issue / 生成草稿"时，交互式收集信息、生成草稿、查重，经确认后提交。当用户提供 PR 链接、要求"创建 Issue / 关联 Issue / 给 PR 建 Issue"，或用户直接描述问题、要求"提单 / 生成草稿"时触发此 skill。

- **gitcode-issue-handler**
  - path: `.skills-cache/cannbot-skills/infra/gitcode-issue-handler/SKILL.md`
  - desc: GitCode Issue 端到端处置工具，根据 Issue 内容自动判断走两条路径之一：(PR 路径) 克隆 fork → 代码定位 → 最小改动 → 跑测试 → 提交并推送 → 创建 PR，覆盖 bug 修复 / 功能增强 / 文档补全等任何需要代码变更的诉求；(Comment 路径) 仅克隆上游主仓只读分析 → 起草答复 → 提交评论，覆盖答疑 / 设计澄清 / 用法说明等不需改代码的诉求。当用户提到"处理 Issue / 跟进 Issue / 从 Issue 提 PR / 端到端处理 Issue / 修复 Issue"或仅给出 issue_url 让 Claude 判断要不要改代...

- **gitcode-pr-handler**
  - path: `.skills-cache/cannbot-skills/infra/gitcode-pr-handler/SKILL.md`
  - desc: 根据 GitCode PR 的代码变更，重新生成符合约定式提交规范的 PR 标题与符合仓库 PR 模板的 PR 描述（body），然后通过 GitCode API 写回 PR。当用户提供 PR 链接、要求"更新 PR / 生成标题 / 生成描述 / 改 PR 文案 / 重写 PR 标题描述"时触发此 skill。

- **gitcode-toolkit**
  - path: `.skills-cache/cannbot-skills/infra/gitcode-toolkit/SKILL.md`
  - desc: GitCode 协作通用基础参考（内部参考，不直接触发）。提供 GitCode API、Token 配置、URL 解析、日志规范、变更展示，Git 克隆/分支/diff/log/remote 通用操作，以及 PR 创建工作流和 Issue 创建工作流（API/模板/head 格式等）等共享文档与确定性脚本。供 gitcode-pr-handler、gitcode-issue-gen、gitcode-issue-handler 等 GitCode 协作类 skill 引用使用，本 skill 自身不响应用户触发。

- **knowledge-issue-report**
  - path: `.skills-cache/cannbot-skills/plugins-community/cannbot-knowledge/skills/knowledge-issue-report/SKILL.md`
  - desc: 生成并校验 cannbot-knowledge 知识库 GitCode Issue 提交材料。触发：当用户要提交 Issue、反馈知识内容错误/缺失、检索错误、图谱错误、知识编译错误、治理 lint 错误、debug 下游 agent 使用问题，或要求打包复现与测试材料时使用。

- **knowledge-lint**
  - path: `.skills-cache/cannbot-skills/plugins-community/cannbot-knowledge/skills/knowledge-lint/SKILL.md`
  - desc: Use when 用户需要在摄入、生成、勘误或提交 PR 前，检查 Ascend NPU 算子 OKF 知识库的结构、溯源、索引或图谱是否合规。

- **knowledge-query**
  - path: `.skills-cache/cannbot-skills/plugins-community/cannbot-knowledge/skills/knowledge-query/SKILL.md`
  - desc: Use when 用户正在处理昇腾 NPU 上的 Ascend C/CANN 算子任务，并询问 API 名称或可用性、签名、参数、头文件、调用方式、平台/版本支持、Tiling、数据搬运、AICore kernel、实现样例、编译/运行错误、精度、性能或 Profiling；在搜索本地 CANN/Toolkit/SDK 安装包或源码前触发。

- **model-infer-fusion**
  - path: `.skills-cache/cannbot-skills/model/model-infer-fusion/SKILL.md`
  - desc: 基于 PyTorch 框架的昇腾 NPU 模型推理融合算子优化技能。分析模型代码，识别可替换为 torch_npu 融合算子的计算模式，生成替换方案。触发场景：torch_npu 融合算子替换、MoE/Attention/FFN/Norm 等模块的推理算子适配、torch_npu API 使用咨询。基于仓库已有模型的融合算子经验，按计算语义推荐最佳方案。

- **model-infer-graph-mode**
  - path: `.skills-cache/cannbot-skills/model/model-infer-graph-mode/SKILL.md`
  - desc: 基于 PyTorch 框架的昇腾 NPU 模型推理图模式适配技能。将模型适配到 torch.compile 图模式以加速推理性能。触发场景：npugraph_ex 或 GE 图模式适配、torch.compile 在昇腾 NPU 上的使用、图中断（Graph Break）修复、aclgraph 图编译问题。LLM 模型的图模式适配优先阅读 LLM 模型改造指南。

- **model-infer-harmony**
  - path: `.skills-cache/cannbot-skills/model/model-infer-harmony/SKILL.md`
  - desc: 麒麟 NPU 端侧（Kirin9030 / HarmonyOS）ASR 模型 4bit W4A16 量化、omg 离线模型转换与 CANNPAK 打包全流程技能。覆盖 dopt PTQ 标定 → 导量化参数 → 导 ONNX → 图改写 → MatMul 补维 → 容器内 omg 转 omc → 多 omc 打包成端侧单 bin，并含量化精度修复与转换报错调试。触发：当用户在 Kirin9030 NPU 上量化部署 ASR 的 encoder / decoder / 标点（punc）模型、把浮点 PyTorch 模型转成端侧 omc、遇到 UINT4 越界 / MatMul don't ...

- **model-infer-kvcache**
  - path: `.skills-cache/cannbot-skills/model/model-infer-kvcache/SKILL.md`
  - desc: 基于 PyTorch 框架的昇腾 NPU 模型推理 KVCache 优化技能。分析并改造 LLM 推理模型的 KVCache 实现，覆盖 Legacy 连续缓存与分页注意力（Paged Attention）配 FA 融合算子、MLA 压缩缓存、SlidingWindow / 多 attn_type 混合。触发场景：KVCache 管理实现、分页注意力接入、KV 压缩、FA 融合算子、OOM / 性能问题、block_table / slot_mapping 构造。支持框架部署与独立部署两种模式。

- **model-infer-migrator**
  - path: `.skills-cache/cannbot-skills/model/model-infer-migrator/SKILL.md`
  - desc: 基于 PyTorch 框架的昇腾 NPU 模型推理适配与部署基线技能。支持两种部署模式：框架部署模式（接入 cann-recipes-infer 的 executor/core/）和独立部署模式（自管 Runner 不依赖框架）。从 HF 链接或本地代码适配为可运行的标准模型目录，并采集性能基线。触发场景：新模型适配到昇腾 NPU 推理框架、已有模型的部署基线采集、模型迁移和初始跑通验证。

- **model-infer-multi-stream**
  - path: `.skills-cache/cannbot-skills/model/model-infer-multi-stream/SKILL.md`
  - desc: NPU 多流技术知识技能。提供整网模块 / 算子 DAG 拆解、模块间与模块内并行性判断、多流候选编排派生、TorchAir(Ascend IR/GE) 与 npugraph_ex/aclgraph 的多流 API 路由、切流 / 同步 / 控核实现，以及假并行（overlap_pct）排查与 Profile 验证等技术规则。供两类工作引用：形成多流优化候选 plan，以及实施 / review 多流改造。触发场景包括：多流、双流、stream overlap、控核、limit_core_num、整网 DAG、模块拆解、npu_stream_switch、npu_wait_tensor、...

- **model-infer-parallel-analysis**
  - path: `.skills-cache/cannbot-skills/model/model-infer-parallel-analysis/SKILL.md`
  - desc: 基于 PyTorch 框架的昇腾 NPU 模型推理并行策略分析技能。分析模型架构参数和昇腾硬件规格，推荐最优的 TP/EP/DP 并行配置（parallel_config）。触发场景：新模型需要确定并行策略、现有配置需要优化、部署卡数或硬件变更后需要重新评估。输出为结构化的 parallel_config 推荐及定量依据。

- **model-infer-parallel-impl**
  - path: `.skills-cache/cannbot-skills/model/model-infer-parallel-impl/SKILL.md`
  - desc: 基于 PyTorch 框架的昇腾 NPU 模型推理并行切分实施技能。根据已确认的 parallel_config，实施模型代码的并行化改造，包括并行线性层替换、MoE 并行模式适配、通信组创建、Embedding/LMHead 并行、YAML 配置生成和权重转换。支持 infer 仓框架部署与独立部署两种模式。触发场景：model-infer-parallel-analysis 完成后需要实施改造、现有模型需要支持新的并行配置。

- **model-infer-perf-breakdown**
  - path: `.skills-cache/cannbot-skills/model/model-infer-perf-breakdown/SKILL.md`
  - desc: NPU 性能数据拆解技能。把 kernel_details.csv 按用户描述的模型结构切成 component
实例（每层 attn / ffn / moe…），再按用户给的 cluster 规则把 component 内部算子
分桶，生成 wall_ms / bubble_ms 中位数 + 异常 layer 的单页 HTML。
触发场景：分析 NPU prof / 拆解大模型性能 / 找抖动 layer。


- **model-infer-precision-debug**
  - path: `.skills-cache/cannbot-skills/model/model-infer-precision-debug/SKILL.md`
  - desc: 基于 PyTorch 框架的昇腾 NPU 模型推理精度问题诊断技能。当前主要覆盖 KVCache / FlashAttention 相关精度问题，包括 Prefill/Decode 对齐、cache 更新错误、slot/block mapping 错误、attention 路径切换后的精度异常等。触发场景：优化改造后精度验证未通过、模型输出与基线存在显著偏差、Prefill 和 Decode 精度表现不一致、出现 NaN/Inf、量化模式下精度异常放大等。

- **model-infer-prefetch**
  - path: `.skills-cache/cannbot-skills/model/model-infer-prefetch/SKILL.md`
  - desc: 为模型添加 torch_npu.npu_prefetch 权重预取优化特性。触发：profiling 显示 MatMul/QBMM/GMM 算子存在 memory-bound 热点、需要为模型添加权重预取、将 prefetch 模式迁移到新模型时使用。

- **model-infer-profiling**
  - path: `.skills-cache/cannbot-skills/model/model-infer-profiling/SKILL.md`
  - desc: NPU 性能分析数据采集技能。用于华为昇腾NPU上的 PyTorch 模型性能分析。

触发场景：
- 用户需要采集 NPU 性能数据
- 用户提到 profiling、性能分析、tensorboard
- 用户需要分析模型推理性能瓶颈
- 用户使用 torch_npu.profiler
- 用户遇到 profiler 解析失败或 JSON 截断问题

关键要求：必须用 ExperimentalConfig(Level1 + PipeUtilization)，否则 kernel_details.csv 只有 9 列、op_statistic/api_statistic 不生成。采集前先判...

- **model-infer-quantization**
  - path: `.skills-cache/cannbot-skills/model/model-infer-quantization/SKILL.md`
  - desc: infer 仓模型量化适配改造技能。分析并接入既有 compressed-tensors 量化方案和权重，完成量化产物契约检查、结构参考匹配、量化 runtime 映射、权重加载、post-load 处理、融合算子量化冲突回退、真实生效验证和收益评估。触发：模型优化流程中的量化初评估、量化改造任务、compressed-tensors 量化产物接入时使用；不重新设计上游量化算法，不实现 compressed-tensors 之外的量化路线。

- **model-infer-runtime-debug**
  - path: `.skills-cache/cannbot-skills/model/model-infer-runtime-debug/SKILL.md`
  - desc: 基于 PyTorch 框架的昇腾 NPU 模型推理运行时错误诊断与修复技能。系统化排查模型加载、初始化、推理执行全链路的运行时错误，包括 aicore timeout、HCCL 通信错误、OOM、算子约束违反、推理卡住等。触发场景：NPU 运行时错误（RuntimeError、aicore timeout 507014、HCCL timeout、device synchronize 失败、kernel crash、EZ9999/EE9999 错误码）、推理过程卡住不返回、权重加载阶段 crash、模型加载成功但 forward 失败、分布式推理某些 rank 挂死等。

- **model-infer-superkernel**
  - path: `.skills-cache/cannbot-skills/model/model-infer-superkernel/SKILL.md`
  - desc: SuperKernel 适配技能。当用户需要启用 SuperKernel 算子二进制融合技术优化 NPU 推理性能时使用此技能。触发场景包括：用户询问 SuperKernel、算子融合、二进制融合、启用 superkernel、superkernel_scope、减少任务调度开销、优化 decode 性能等。SuperKernel 仅支持 ge_graph 模式、Atlas A3 硬件、PyTorch 框架，且仅在 decode 阶段生效。

- **model-train-accuracy-debug**
  - path: `.skills-cache/cannbot-skills/model/model-train-accuracy-debug/SKILL.md`
  - desc: 用于定位 PyTorch on NPU 大模型训练中“有基线可对照”的精度异常定位。只要用户提到训练精度异常、 loss/grad norm 曲线偏离、换算子/并行策略/分支/CANN 版本后精度异常，或出现 NaN 且有基线对照，就应触发本技能，并按代码审查 + detect_anomaly + msprobe dump/compare 流程定界根因。

- **model-train-log-visualization**
  - path: `.skills-cache/cannbot-skills/model/model-train-log-visualization/SKILL.md`
  - desc: 用于 NPU 大模型训练的日志可视化。当用户提到训练日志作图、loss/grad_norm 曲线、两份训练日志对比、误差曲线，或需要从 torchtitan 风格训练日志按 step 提取并可视化性能指标（含 memory/tps/tflops/mfu/elapsed_time_per_step/indexer_loss）时，优先使用本技能；即使用户只说“画训练日志曲线”“对比两份训练日志”也应触发。

- **model-train-oom-analysis**
  - path: `.skills-cache/cannbot-skills/model/model-train-oom-analysis/SKILL.md`
  - desc: 用于诊断 PyTorch on NPU 大模型训练中的 NPU OOM（Out of Memory）问题。当用户报告训练因内存不足崩溃、出现 OOM 相关错误（OutOfMemoryError / NPU out of memory / workspace allocator / HCCL memory）、或需要进行训练内存优化时，触发本技能。按照日志分类 → 静态估算 → snapshot 深度分析 → 优化建议的流程定位和解决问题。

- **npu-arch**
  - path: `.skills-cache/cannbot-skills/ops/npu-arch/SKILL.md`
  - desc: Ascend NPU 架构知识查询技能。通过芯片型号映射、架构代际划分和 archXX 特性说明，帮助判断目标平台能力、特性支持与条件编译策略。当需要确认芯片型号、NpuArch/SocVersion、架构差异、特性支持或编译分支条件时使用。

- **op-dashboard**
  - path: `.skills-cache/cannbot-skills/plugins-community/collaborative-agent-kernel-evolution/skills/op-dashboard/SKILL.md`
  - desc: Generate self-contained interactive HTML dashboard (4 tabs: algo flow, UB tiling, precision, performance) from an AscendC operator output directory. Use when asked to visualize or report operator results. Works for both precision-pass and precision-fail states.

- **op-desc-generation**
  - path: `.skills-cache/cannbot-skills/plugins-community/collaborative-agent-kernel-evolution/skills/op-desc-generation/SKILL.md`
  - desc: Generate operator description JSON (shapes, dtypes, attributes) from API description or user specification. First step in the cake agent pipeline.

- **ops-direct-invoke-flash**
  - path: `.skills-cache/cannbot-skills/plugins-official/ops-direct-invoke-flash/skills/ops-direct-invoke-flash/SKILL.md`
  - desc: 当需要从 CPU 函数、数学公式、代码片段或文本描述出发构建新的 Ascend C 或 Ascend950 Reg API 核函数时使用。覆盖从规格说明到经验证的 NPU 核函数的完整路径。

- **ops-direct-invoke-workflow**
  - path: `.skills-cache/cannbot-skills/plugins-community/cuda2ascend/skills/ops-direct-invoke-workflow/SKILL.md`
  - desc: 直调算子开发工作流编排。承载从需求分析到上库的完整流程：阶段划分、各环节角色/输入/输出/交付件、CP 验收与回退、状态机。触发：用户要求开发新算子、实现某算子接口，或推进算子开发流程时。

- **ops-direct-invoke-workflow-maintain**
  - path: `.skills-cache/cannbot-skills/plugins-community/cuda2ascend/skills/ops-direct-invoke-workflow-maintain/SKILL.md`
  - desc: 工作流维护技能。任何对工作流文件的新增、修改、删除，都必须先触发本技能再执行，包括修改本技能自身。禁止未触发直接修改。触发：任何对工作流文件（基类的 AGENTS.md / agents / skills / init.sh，或算子仓 agent/ 下的 override 实现）的新增、修改、删除

- **ops-easyasc-dsl**
  - path: `.skills-cache/cannbot-skills/plugins-community/ops-easyasc-dsl/skill/SKILL.md`
  - desc: easyasc DSL to AscendC workflow. Author, debug, and validate Ascend NPU kernels written in the easyasc Python DSL, then lower them to AscendC runtime.

- **ops-evaluation**
  - path: `.skills-cache/cannbot-skills/plugins-community/ops-perf-evolution/skills/ops-evaluation/SKILL.md`
  - desc: 从 ops-nn/cv/math/transformer 等算子仓库构建、部署并评估 AscendC 算子，比较基线与进化后的性能差异。当需要构建 ops 仓库算子、运行正确性验证与性能评测、或对比基线版本与进化版本差异时使用。

- **ops-knowledge-cv-ingest**
  - path: `.skills-cache/cannbot-skills/plugins-community/cannbot-knowledge/skills/ops-knowledge-cv-ingest/SKILL.md`
  - desc: 为 AscendC CV（cube-vector）融合算子生成「cube↔vector 融合设计」wiki（手动触发）。

- **ops-knowledge-ingest**
  - path: `.skills-cache/cannbot-skills/plugins-community/cannbot-knowledge/skills/ops-knowledge-ingest/SKILL.md`
  - desc: 知识库摄入顶层编排器：大原则（新 source 接入 / commit 级增量 / 大版本升级三路由）+ reference/ops 生产 skill 与 runbooks 治理边界。持跨树共享图引擎脚本（okf_graph.py / okf_judge_aggregate.py）。不亲自产卡——正文著作委派对应生产 skill。手动触发。

- **ops-knowledge-reference-ingest**
  - path: `.skills-cache/cannbot-skills/plugins-community/cannbot-knowledge/skills/ops-knowledge-reference-ingest/SKILL.md`
  - desc: Use when 用户需要将 asc-devkit、CANN 或 Profiling 上游文档摄入并编译为 Ascend NPU 算子 OKF reference 知识卡。

- **ops-knowledge-vv-ingest**
  - path: `.skills-cache/cannbot-skills/plugins-community/cannbot-knowledge/skills/ops-knowledge-vv-ingest/SKILL.md`
  - desc: 为 AscendC 多模板 VV 算子从官方 golden 代码生成解耦的两层知识：算子特定 wiki（ops/{repo}/{category}/{op}.md，逐模板全链路 + mermaid UB 布局图）和泛化优化点 runbook（runbooks/operator-optimization/vv-fusion-common.md，跨算子单一共享、增量合并的 NPU 垂域优化点库）。仅适用于有多模板分发（TilingKey 多分支）的 VV 纯 Vector 算子。手动触发。

- **ops-precision-standard**
  - path: `.skills-cache/cannbot-skills/ops/ops-precision-standard/SKILL.md`
  - desc: 算子精度标准。描述 Ascend C 算子各种 dtype 输出对应的精度比对标准(混合容差 atol/rtol)。当需要(1)评估算子精度是否达标,(2)编写 ST 测试验证精度,(3)处理 FP16/FP32/BF16 等不同数据类型精度问题,(4)确认算子精度验收标准时触发。

- **ops-profiling**
  - path: `.skills-cache/cannbot-skills/ops/ops-profiling/SKILL.md`
  - desc: NPU 性能采集与分析，融合 msprof 算子级瓶颈定位与 kernel-level 对比测试，用于采集算子性能数据、对比自定义算子 vs 标杆加速比、定位性能瓶颈并给出优化建议。当用户在算子开发过程中提到"上板性能"、"算子性能测试"、"硬件性能验证"、"NPU性能采集"、"NPU profiling"、"性能对比"、"加速比"等场景时触发。

- **ops-simulator**
  - path: `.skills-cache/cannbot-skills/ops/ops-simulator/SKILL.md`
  - desc: NPU 仿真器技能。提供 CANN Simulator 的使用指导，包括精度仿真、性能仿真、流水线分析。当需要在无 NPU 硬件环境下验证算子功能、分析性能瓶颈、定位精度问题时使用。也用于分析已有的 cannsim 性能报告（summary.json）并给出优化建议。

- **ops-spec-gen**
  - path: `.skills-cache/cannbot-skills/ops/ops-spec-gen/SKILL.md`
  - desc: 生成或校验算子 spec.yaml（算子的 L0 数学约束唯一真值）。当用户提及：生成 spec.yaml、新算子 spec 骨架、scaffold spec、validate spec.yaml、spec schema 校验、算子规格校验 时触发。

- **perf-analyzer**
  - path: `.skills-cache/cannbot-skills/ops/pypto-op-perf-tune/perf-analyzer/SKILL.md`
  - desc: 分析 PyPTO 算子的性能指标。适用于分析 PyPTO 算子的性能指标，从性能数据文件中提取关键指标，计算性能评级，并提供性能瓶颈分析和优化建议时。

- **precision-binary-search**
  - path: `.skills-cache/cannbot-skills/ops/pypto-precision-compare/precision-binary-search/SKILL.md`
  - desc: PyPTO 算子上板二分定位法。通过在 kernel 函数中添加检查点 tensor 作为输入参数进行原地修改，对比中间结果的精度。专注：修改 kernel 签名 → 修改 golden 返回值 → 修改测试函数 → 二分定位首个出错的 op。适用于上板二分定位算子精度问题时。

- **precision-pass**
  - path: `.skills-cache/cannbot-skills/ops/pypto-precision-compare/precision-pass/SKILL.md`
  - desc: Pass精度校验子技能。开启PreCheck/PostCheck进行全链路Pass校验，通过pass校验定位报错Pass，使用pass_compare逐Op对比定位具体问题Op，支持动态shape上板数据打印验证。适用于 Pass 精度校验并定位问题 Op 时。

- **precision-verify**
  - path: `.skills-cache/cannbot-skills/ops/pypto-precision-compare/precision-verify/SKILL.md`
  - desc: PyPTO 算子精精度工具对比法。使用 pypto.pass_verify_save 和 torch.save 保存中间结果到文件，然后使用对比工具分析。专注：插入检查点 → 运行测试 → 对比分析 → 定位首个失败点。适用于用精度工具对比定位首个失败点时。

- **pypto-api-explore**
  - path: `.skills-cache/cannbot-skills/ops/pypto-api-explore/SKILL.md`
  - desc: 探索 PyPTO API，为算子开发提供 API 映射、约束检查和 Tiling 需求分析。当需要查找 PyPTO 是否支持某个操作、验证 API 约束、分析算子可行性时使用。触发词：API 探索、查找 API、PyPTO 有没有 xxx、支持什么 dtype、约束是什么、tiling 怎么配、API 映射、可行性分析、这个算子能做吗。

- **pypto-docs-search**
  - path: `.skills-cache/cannbot-skills/ops/pypto-docs-search/SKILL.md`
  - desc: 检索 PyPTO 算子开发资源——API 文档、按错误码排障、教程/安装/工具文档、算子参考实现与 golden。当要查 PyPTO 文档、读算子或 API 用法、按错误码排障、找算子参考实现或 golden、或问"这份资源在哪、有哪些"时使用，即使没明说"文档站"也应触发。触发：当需要查 PyPTO 文档、算子/API 用法、错误码排障或算子参考实现时。

- **pypto-general-debug**
  - path: `.skills-cache/cannbot-skills/ops/pypto-general-debug/SKILL.md`
  - desc: PyPTO debugging router for stuck or opaque failures, including tile-shape/L0/L1/alignment/set_cube_tile_shapes issues; route through DEBUG_GUIDEBOOK.md to the matching topic reference or sub-skill. Use when a PyPTO kernel run is stuck or failing opaquely.

- **pypto-golden-generate**
  - path: `.skills-cache/cannbot-skills/ops/pypto-golden-generate/SKILL.md`
  - desc: 当需要生成 golden 参考实现时使用此 skill。基于算子规格信息，生成 torch + torch_npu NPU 参考实现 `{op}_golden.py`，导出 `{op}_golden()` 函数，作为精度验证基准。计算在 NPU 上执行；torch_npu 未安装时直接报错引导安装，仅无 NPU 硬件时回退 CPU。触发词：生成 golden、生成参考实现、写 golden 函数、golden script、golden reference、reference implementation、generate golden、torch 参考、验证基准、baseline im...

- **pypto-intent-understand**
  - path: `.skills-cache/cannbot-skills/ops/pypto-intent-understand/SKILL.md`
  - desc: PyPTO 算子需求意图理解。将用户的自然语言算子描述转化为结构化需求文档。当用户描述要开发、实现、创建某个算子时触发，例如：'开发一个 sinh 算子'、'实现 GELU'、'参考 PyTorch 的 F.scaled_dot_product_attention'、'根据论文实现算子'、'创建自定义算子'

- **pypto-memory-template**
  - path: `.skills-cache/cannbot-skills/ops/pypto-memory-template/SKILL.md`
  - desc: Template for the PyPTO Kernel operator-specific memory file (custom/{op}/MEMORY.md). Defines required sections, machine-readable fields, and update cadence. Use when creating or updating an operator's MEMORY.md.

- **pypto-op-construct**
  - path: `.skills-cache/cannbot-skills/ops/pypto-op-construct/SKILL.md`
  - desc: PyPTO Kernel module decomposition and construction. Covers semantic module decomposition (split by meaning, define contracts, freeze) and module construction (one module at a time, validate, cross-check golden inventory). Use when decomposing an operator into modules and constructing them.

- **pypto-op-design**
  - path: `.skills-cache/cannbot-skills/ops/pypto-op-design/SKILL.md`
  - desc: 当需要设计 PyPTO 算子实现方案时使用。通过迭代式约束收敛，生成 DESIGN.md（含 API 映射、精度路由、Tiling 推导、Loop 结构设计）。触发词：生成设计方案、生成 design、设计方案、写 DESIGN.md、算子设计、API 映射、Tiling 策略、tiling 推导、Loop 结构、数据流设计、精度路由。

- **pypto-op-develop**
  - path: `.skills-cache/cannbot-skills/ops/pypto-op-develop/SKILL.md`
  - desc: PyPTO 算子 impl 编码手册。coder agent 收到调度时使用，先 per-Phase 累计构建 `{op}_module{k}_impl.py`，最后一个 Phase 通过验证后 cleanup 整理出 `{op}_impl.py` + `README.md`。基于 Layer A–L 设计规范，配合 `impl_template.py` 模板生成符合规范的 PyPTO 实现代码。触发关键词：实现算子、写 kernel、编写实现、写 impl、算子编码、code the op、op develop、kernel 实现。

- **pypto-op-knowledge**
  - path: `.skills-cache/cannbot-skills/ops/pypto-op-knowledge/SKILL.md`
  - desc: PyPTO 算子开发知识库查询技能。串行查询经验表和问题查找表，有结果立即返回，无结果返回"无匹配"。触发关键词：查经验表、查问题表、知识库查询、knowledge query。

- **pypto-op-perf-tune**
  - path: `.skills-cache/cannbot-skills/ops/pypto-op-perf-tune/SKILL.md`
  - desc: PyPTO 算子性能分析和自动调优技能。用于对生成及新开发的算子进行性能分析及自动调优，包括算子用例执行及精度校验、性能数据采集及分析、分步骤性能调优和生成性能分析报告。当用户需要分析 PyPTO 算子性能、进行性能调优、生成性能报告时使用此技能。触发词：算子性能调优、性能分析、自动调优、性能优化、泳道图分析。

- **pypto-op-plan**
  - path: `.skills-cache/cannbot-skills/ops/pypto-op-plan/SKILL.md`
  - desc: PyPTO Kernel requirement planning — structurally-similar example search and feasibility setup. Use when planning a new operator's development.

- **pypto-op-review**
  - path: `.skills-cache/cannbot-skills/ops/pypto-op-review/SKILL.md`
  - desc: Op-by-op PyPTO call extraction plus the layout / structure requirements for custom/{operator}/ kernels. Use when reviewing or debugging a custom operator's kernels.

- **pypto-op-verify**
  - path: `.skills-cache/cannbot-skills/ops/pypto-op-verify/SKILL.md`
  - desc: Validation runner requirements, detailed_tensor_compare usage, success criteria, required deliverables, and required output structure for the PyPTO Complex Kernel Workflow. Use when validating a PyPTO kernel and assembling deliverables.

- **pypto-orchestration-manual**
  - path: `.skills-cache/cannbot-skills/ops/pypto-orchestration-manual/SKILL.md`
  - desc: pypto-op-orchestrator entry point for PyPTO Kernel development. Bundles the 3 control documents (principles, team roster, mandatory rules) as one skill with progressive-disclosure references. Read this file first, then load the references on demand. Use when orchestrating the PyPTO operator-devel...

- **pypto-precision-compare**
  - path: `.skills-cache/cannbot-skills/ops/pypto-precision-compare/SKILL.md`
  - desc: PyPTO 算子精度问题调试技能。提供两种精度对比方法：文件保存方法（使用 pypto.pass_verify_save 和 torch.save）和二分对比方法（使用检查点 tensor）。当需要调试 PyPTO 算子精度、定位精度差异来源、进行中间结果对比时使用此技能。

- **pypto-precision-debug**
  - path: `.skills-cache/cannbot-skills/ops/pypto-precision-debug/SKILL.md`
  - desc: PyPTO 算子精度问题排查技能。专注于用户代码层面的语法逻辑检查和规避方法尝试。当算子精度验证失败、输出结果异常、计算错误、数值偏差、或任何与精度相关的问题时使用此技能。

- **reference-generation**
  - path: `.skills-cache/cannbot-skills/plugins-community/collaborative-agent-kernel-evolution/skills/reference-generation/SKILL.md`
  - desc: Generate reference PyTorch implementation (nn.Module with get_init_inputs/get_random_inputs) from operator description JSON. Faithfully reproduces the Golden definition when provided. Use after op-desc-generation.

- **remote-cann-development**
  - path: `.skills-cache/cannbot-skills/plugins-community/collaborative-agent-kernel-evolution/skills/remote-cann-development/SKILL.md`
  - desc: Unified remote NPU development — sync, exec, and test across multiple NPU backends (docker containers, hdspace cloud) via a single Python CLI. Use when: (1) syncing code to any remote NPU server (2) executing build/test commands on remote NPU (3) probing NPU platform info (4) managing multiple re...

- **repo-build-guide**
  - path: `.skills-cache/cannbot-skills/plugins-community/cuda2ascend/skills/repo-build-guide/SKILL.md`
  - desc: 仓库代码结构与构建指南，介绍本仓算子代码的目录/文件结构与编译验证方法。触发：了解算子代码结构、搭建工程、编译与运行验证时加载。

- **repo-coding-rules**
  - path: `.skills-cache/cannbot-skills/plugins-community/cuda2ascend/skills/repo-coding-rules/SKILL.md`
  - desc: 本仓编码红线，列出算子编码中明确禁止的红线条款与触发后的解决方案，供编码与代码检视对照。触发：编写/修复算子代码、代码检视逐条核对红线时加载。

- **repo-knowledge**
  - path: `.skills-cache/cannbot-skills/plugins-community/cuda2ascend/skills/repo-knowledge/SKILL.md`
  - desc: 仓库领域知识，提供本仓算子涉及的领域标准、概念与背景。触发：需要算子族设计方法论、目标芯片架构概念、精度领域标准等背景知识时加载。

- **repo-op-templates**
  - path: `.skills-cache/cannbot-skills/plugins-community/cuda2ascend/skills/repo-op-templates/SKILL.md`
  - desc: 算子代码模板库，提供代码模板与模板选择规则，作为算子代码开发的起点。触发：开始实现算子代码、搭建工程骨架前，先取模板复制到工作区，以此为起点开发。

- **repo-test-develop**
  - path: `.skills-cache/cannbot-skills/plugins-community/cuda2ascend/skills/repo-test-develop/SKILL.md`
  - desc: 仓库测试开发指导，介绍本仓测试框架的使用与测试代码的开发方法。触发：实现 golden、编写分级功能用例、补全白盒测试、搭建性能采集框架、执行精度测试时加载。

- **runtime_migration**
  - path: `.skills-cache/cannbot-skills/runtime/runtime_migration/SKILL.md`
  - desc: 将用户合法拥有或已获授权的 CUDA 应用中的 Runtime 层迁移到 CANN Runtime，包括 API、类型、错误码、初始化、内存、stream、event、IPC、构建配置和兼容层适配。仅在处理 Runtime 层迁移或可行性分析时使用；不迁移 CUDA kernel、device helper 或任何算子实现。

- **scan-cmake**
  - path: `.skills-cache/cannbot-skills/plugins-community/ops-qa-suite/.opencode/skills/scan-cmake/SKILL.md`
  - desc: CMake 配置问题扫描技能。用于扫描 ops-math/ops-nn/ops-transformer/ops-cv 仓库的 CMakeLists.txt 文件，检测 OPTYPE 参数错误、UT 构建配置问题、目标冲突、源文件缺失等问题。当用户询问 CMake 构建错误、UT 构建失败、CMakeLists 问题检测时使用。

- **scan-examples-analysis**
  - path: `.skills-cache/cannbot-skills/plugins-community/ops-qa-suite/.opencode/skills/scan-examples-analysis/SKILL.md`
  - desc: Ascend C 算子 examples 缺失分析技能。用于分析 ops-math/ops-nn/ops-transformer/ops-cv 算子的 examples 需求，判断何时需要 examples 测试用例，识别 examples 缺失情况。当用户询问 examples 缺失、调用示例、测试用例时使用。

- **scan-examples-test**
  - path: `.skills-cache/cannbot-skills/plugins-community/ops-qa-suite/.opencode/skills/scan-examples-test/SKILL.md`
  - desc: Ascend C 算子仓库全量 examples 测试执行与报告生成技能。用于执行 ops-math/ops-nn/ops-transformer/ops-cv 仓库的全量算子 examples 测试（test_aclnn_*.cpp/test_geir_*.cpp），记录测试结果，识别失败问题，按标准格式生成测试报告，自动为失败问题生成 Issue 文件。核心原则：1) 所有问题都创建 Issue；2) 报告后询问提交；3) 同类问题合并选项。当用户需要执行全量 examples 测试、分析测试失败原因、生成 examples 测试报告时使用。

- **scan-op-api-list**
  - path: `.skills-cache/cannbot-skills/plugins-community/ops-qa-suite/.opencode/skills/scan-op-api-list/SKILL.md`
  - desc: 算子接口列表一致性扫描技能。用于扫描 ops-math/ops-nn/ops-transformer/ops-cv 仓库的 docs/zh/op_api_list.md 文档，验证 aclnn 接口名、接口说明、确定性说明与实际代码实现的一致性。当用户需要验证aclnn接口文档准确性、检查op_api_list表格与实际实现匹配时使用。

- **scan-op-list**
  - path: `.skills-cache/cannbot-skills/plugins-community/ops-qa-suite/.opencode/skills/scan-op-list/SKILL.md`
  - desc: 算子列表一致性扫描技能。用于扫描 ops-math/ops-nn/ops-transformer/ops-cv 仓库的 docs/zh/op_list.md 文档，验证算子目录、分类、实现状态标记(√×)、硬件单元说明与实际代码实现的一致性。当用户需要验证算子列表文档准确性、检查op_list表格与实际实现匹配时使用。

- **scan-repo-docs**
  - path: `.skills-cache/cannbot-skills/plugins-community/ops-qa-suite/.opencode/skills/scan-repo-docs/SKILL.md`
  - desc: Ascend C 算子仓库文档质量扫描技能。用于扫描 ops-math/ops-nn/ops-transformer/ops-cv 仓库的资料正确性、资料易理解性、资料规范性，生成详细报告。当用户需要仓库文档质量评估、问题检测、验证修复效果时使用。

- **scan-ut-analysis**
  - path: `.skills-cache/cannbot-skills/plugins-community/ops-qa-suite/.opencode/skills/scan-ut-analysis/SKILL.md`
  - desc: Ascend C 算子 UT 类型分析与缺失检测技能。用于分析 ops-math/ops-nn/ops-transformer/ops-cv 算子的 UT 需求，判断何时需要 _infershape/_tiling/op_kernel/op_api UT，识别 UT 缺失情况。当用户询问 UT 类型判断、UT 缺失分析、测试覆盖策略时使用。

- **scan-ut-test**
  - path: `.skills-cache/cannbot-skills/plugins-community/ops-qa-suite/.opencode/skills/scan-ut-test/SKILL.md`
  - desc: Ascend C 算子仓库全量 UT 测试执行与报告生成技能。用于执行 ops-math/ops-nn/ops-transformer/ops-cv 仓库的全量 UT 测试（op_host/op_api/op_kernel），记录测试结果，识别阻塞问题，按标准格式生成测试报告，自动为阻塞问题生成 Issue 文件。核心原则：1) 所有问题都创建 Issue；2) 报告后询问提交；3) 同类问题合并选项。当用户需要执行全量 UT 测试、分析测试失败原因、生成 UT 测试报告时使用。

- **science-model-npu-migration**
  - path: `.skills-cache/cannbot-skills/plugins-community/science-model-npu-migration/SKILL.md`
  - desc: 面向华为 Ascend 的 NPU 代码级迁移（环境门禁、脚本适配、精度/性能对比）。Use when the user asks for NPU/Ascend migration, torch_npu, MindSpore Ascend, or NPU adaptation checks.

- **skill-trace**
  - path: `.skills-cache/cannbot-skills/plugins-community/collaborative-agent-kernel-evolution/skills/skill-trace/SKILL.md`
  - desc: Track skill invocations, durations, and outcomes during operator generation. Records which skills were called, their inputs/outputs, and correlates with final results. MUST use when: (1) Starting any skill step, (2) Completing any skill step, (3) Final summary to correlate skills with outcomes.


- **spec-to-design**
  - path: `.skills-cache/cannbot-skills/plugins-official/ops-registry-invoke/skills/spec-to-design/SKILL.md`
  - desc: 从算子仓 operators/{operator}/docs/spec.yaml 生成或更新中文 DESIGN.md 和 PLAN.md 的方案设计技能。当用户要求根据 spec.yaml 生成设计文档、方案设计、迭代计划、spec-to-design、更新 DESIGN.md 或执行 ops-registry-invoke 的 1.3 方案设计时触发。

- **task-progress**
  - path: `.skills-cache/cannbot-skills/plugins-community/collaborative-agent-kernel-evolution/skills/task-progress/SKILL.md`
  - desc: Track and manage task progress via a PROGRESS.md. MUST use when: (1) Starting a new task or operator workflow, (2) Entering or completing any step/stage/substage, (3) After context compaction or reset — read PROGRESS.md first to recover state. Also call when the user asks about current progress o...

- **tilelang-api-best-practices**
  - path: `.skills-cache/cannbot-skills/ops/tilelang-api-best-practices/SKILL.md`
  - desc: TileLang Ascend API 使用最佳实践。提供内存分配、数据搬运、矩阵计算、归约、元素级运算、同步、调度原语等 API 的正确用法和最佳实践。触发：使用 TileLang API 编写 Ascend NPU kernel 时或遇到 API 相关问题时。

- **tilelang-env-check**
  - path: `.skills-cache/cannbot-skills/ops/tilelang-env-check/SKILL.md`
  - desc: TileLang-Ascend 环境检查与配置验证技能。检查代码仓库完整性、编译安装状态、环境变量配置，并运行简单测试验证环境。发现问题会自动调用相关 skill 进行修复，并按依赖顺序重新执行后续步骤。触发关键词："环境检查"、"检查环境"、"验证环境"、"环境配置"、"环境搭建"、"env check"、"check environment"、"verify environment"、"setup environment"。

- **tilelang-op-design**
  - path: `.skills-cache/cannbot-skills/ops/tilelang-op-design/SKILL.md`
  - desc: 根据算子需求生成 TileLang-Ascend 算子设计文档（design.md）。涵盖编程模式选型（Developer/Expert/混合）、API 映射、内存层级规划、Tiling 策略、循环结构、同步策略、验证方案等。触发：设计算子、生成 design.md、算子方案设计、新算子开发、算子实现方案。

- **tilelang-op-develop**
  - path: `.skills-cache/cannbot-skills/ops/tilelang-op-develop/SKILL.md`
  - desc: 基于设计文档生成 TileLang-Ascend 算子实现代码与测试。从 design.md 中提取关键信息，结合 examples/ 中的参考实现生成可运行代码。触发：实现算子、写 kernel、生成代码、算子编码、根据设计文档实现。

- **tilelang-op-test-design**
  - path: `.skills-cache/cannbot-skills/ops/tilelang-op-test-design/SKILL.md`
  - desc: TileLang-Ascend 算子测试设计技能。支持多种场景：(1) 从 design.md 设计测试配置 (2) 从 custom/{op}/*.py 补充测试 (3) 手动提供算子信息生成测试 (4) 测试覆盖率分析。理解算子实现逻辑后智能判断测试策略。触发：设计算子测试、生成测试用例、补充测试、测试覆盖率不足。

- **tilelang-perf-optimization**
  - path: `.skills-cache/cannbot-skills/ops/tilelang-perf-optimization/SKILL.md`
  - desc: TileLang 算子性能调优与潜在性能劣化模式检查。提供性能数据采集、瓶颈诊断、优化实施、效果验证能力；也用于生成或评审算子时对照常见性能劣化模式示例检查当前 kernel 代码。触发：算子精度通过后需要优化性能、性能不及预期时。

- **tilelang-programming-model-guide**
  - path: `.skills-cache/cannbot-skills/ops/tilelang-programming-model-guide/SKILL.md`
  - desc: TileLang Ascend Developer/Expert 模式选择与 pass_configs 配置指南。当需要确定编程模式、配置 pass_configs、或在两种模式之间转换时触发。API 详情请参考 tilelang-api-best-practices skill。

- **tilelang-review**
  - path: `.skills-cache/cannbot-skills/ops/tilelang-review/SKILL.md`
  - desc: 检查代码格式是否符合 CI 规则。适用于 TileLang NPU kernel 开发时的代码规范检查和格式化。自动检测并安装缺失工具（ruff、clang-format），先运行检查生成报告，使用醒目方式询问用户后，仅在用户同意时执行修复。工作流程：检测环境→自动安装缺失工具→运行检查→生成报告→醒目询问→用户确认→执行修复。使用此技能当：用户要求"格式检查"、"格式化代码"、"代码格式化"、"检查代码格式"、"代码 review"、"代码审查"、"修复格式"、"fix format"、"lint code"、"检查代码规范"、提交 PR 前验证、或需要检查 Python/C++ 代码...

- **tilelang-submodule-pull**
  - path: `.skills-cache/cannbot-skills/ops/tilelang-submodule-pull/SKILL.md`
  - desc: Automatically pull tilelang repository and its third-party code. Provides scheduled pull script supporting git pull --recurse-submodules and git submodule update --init --recursive with automatic error detection and retry. Triggers when user mentions "重新拉取三方库", "自动重拉", "重新拉取子模块", "auto retry pull...

- **tilelang2ascend-case-simplifier**
  - path: `.skills-cache/cannbot-skills/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-case-simplifier/SKILL.md`
  - desc: 测试用例精简专家 Skill。读取 `{output_dir}` 中与算子对应的 `.json` 文件， 对其中的输入 cases（JSON Lines 格式，每行一个 `{"inputs": [...]}` 对象）进行精简， 使 case 数量尽量不超过 10 个，同时保证覆盖度。 当测试用例数量过多需要精简时，使用此 skill。


- **tilelang2ascend-operator-project-init**
  - path: `.skills-cache/cannbot-skills/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-operator-project-init/SKILL.md`
  - desc: 初始化 AscendC 算子工程并创建可编译的算子骨架。触发场景：(1) 用户要求创建新算子；(2) 关键词：ascendc算子、新建算子、算子目录、算子初始化；(3) 需要基于 ascend-kernel 模板快速落地。本 skill 不只建目录，还输出“可继续开发”的标准文件与检查清单。

- **tilelang2ascend-precision-tuning**
  - path: `.skills-cache/cannbot-skills/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-precision-tuning/SKILL.md`
  - desc: 用于DumpTensor进行AscendC算子精度的调试。
Use when:
- AscendC kernel / 算子精度失败，结果不对，数值错误，部分位置错误，或 NaN/Inf
- 需要用 DumpTensor / dumptensor / dump tensor 看中间结果、GM/UB/Workspace 数据
- 需要分段定位 `Cube输入/中间/输出`、`Vector输入/中间/输出` 哪一段出问题


- **tilelang2ascend-tilelang-designer**
  - path: `.skills-cache/cannbot-skills/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-tilelang-designer/SKILL.md`
  - desc: TileLang kernel 设计与实现专家 Skill。为 PyTorch Model 设计并实现自定义 TileLang kernel： 完成 block-level 设计、tile-level 设计，并生成 model_new_tilelang.py 调用自定义 TileLang kernel。 当需要为复杂算子设计 TileLang kernel 时，使用此 skill。


- **tilelang2ascend-trace-recorder**
  - path: `.skills-cache/cannbot-skills/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-trace-recorder/SKILL.md`
  - desc: 执行 trace 记录员 Skill。在算子任务完成后，回顾整个执行过程， 生成结构化的 trace 记录供 meta-agent 优化使用。 当算子任务完成后需要记录执行过程时，使用此 skill。


- **tilelang2ascend-translator**
  - path: `.skills-cache/cannbot-skills/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/SKILL.md`
  - desc: AscendC kernel 转译与实现专家 Skill。将 TileLang 设计转译为 AscendC kernel， 并生成 model_new_ascendc.py 调用 AscendC kernel。 当 TileLang 设计完成需要转译为 AscendC kernel 时，使用此 skill。


- **tool-link-checker**
  - path: `.skills-cache/cannbot-skills/plugins-community/ops-qa-suite/.opencode/skills/tool-link-checker/SKILL.md`
  - desc: Ascend C 算子仓库 Markdown 断链扫描与修复技能。用于扫描 ops-math/ops-nn/ops-transformer/ops-cv 算子仓库的 Markdown 文件内部链接，检测断链并分类统计，提供一键修复脚本。当用户询问断链检查、链接修复、文档可达性问题时使用。

- **tool-reports-to-issue**
  - path: `.skills-cache/cannbot-skills/plugins-community/ops-qa-suite/.opencode/skills/tool-reports-to-issue/SKILL.md`
  - desc: 扫描报告转 Issue 工具。根据扫描报告或问题列表批量生成 GitCode Issue，支持模板查询、智能合并、自动填充。核心能力：问题类型→模板匹配、同类问题合并策略、Issue 内容生成。通用能力（模板查询、API 提交）引用 gitcode-toolkit（软链接 infra）。当用户提供扫描报告、问题列表、要求"创建 Issue / 根据报告创建 Issue / 批量创建 Issue"时触发此 skill。

- **torch-ascendc-op-extension**
  - path: `.skills-cache/cannbot-skills/ops/torch-ascendc-op-extension/SKILL.md`
  - desc: 将已有 Ascend C <<<>>> 直调工程通过 TORCH_LIBRARY 对接到 PyTorch，实现 torch.ops.npu.xxx() 调用。触发：用户提到 TORCH_LIBRARY、.asc 对接 PyTorch、Python 调用 Ascend C 算子、注册到 torch、算子接入 PyTorch dispatch、PyTorch binding、torch extension、或想在 Python 中用 torch.ops.xxx() 调用已有 Ascend C kernel。不适用：从零建工程（用 ascendc-direct-invoke-template）...

- **torch-custom-ops-guide**
  - path: `.skills-cache/cannbot-skills/graph/torch-custom-ops-guide/SKILL.md`
  - desc: 自定义算子入图完整指南。覆盖从零开发、Eager 算子适配 npugraph_ex 图模式（torch.library.custom_op / torch.library.Library）、Meta 推导函数编写等全流程。适用于两种纯 Python 自定义算子注册场景。关键词：custom_op、torch.library.Library、register_fake、meta、mutates_args。

- **torch-npugraph-ex-compile-error-diagnosis**
  - path: `.skills-cache/cannbot-skills/graph/torch-npugraph-ex-compile-error-diagnosis/SKILL.md`
  - desc: PyTorch 昇腾 NPU npugraph_ex 编译期报错诊断。覆盖 torch.compile 触发后 TorchDynamo / FX / AOTAutograd / npugraph_ex backend / ACL graph capture 阶段的报错排查，包括 Unsupported / graph break / BackendCompilerFailed / Meta 推导失败 / capture 失败等场景。本 skill 由 dfx-triage 路由进入。触发：当用户遇到 npugraph_ex 入图失败、graph break、BackendCompiler...

- **torch-npugraph-ex-dfx-triage**
  - path: `.skills-cache/cannbot-skills/graph/torch-npugraph-ex-dfx-triage/SKILL.md`
  - desc: PyTorch 昇腾 NPU npugraph_ex DFX 问题分诊入口。统一执行首轮全量日志收集与最少闭环信息核对，按报错栈和现象将问题路由到 compile-error / runtime-error / accuracy / performance 四个专科 sub-skill。本 skill 不输出最终诊断结论，只完成「采集 + 分类 + 加载下游 skill」。触发：当用户报告 npugraph_ex 相关报错、断图、精度差异或性能回退、需要 debug/dump 定位时加载。关键词：问题定位、报错、断图、精度、性能、debug、dump、aot_eager。

- **torch-npugraph-ex-knowledge**
  - path: `.skills-cache/cannbot-skills/graph/torch-npugraph-ex-knowledge/SKILL.md`
  - desc: npugraph_ex（aclgraph）模式使用指南。采用 Capture & Replay 方式将算子任务下沉至 Device 执行，减少 Host 调度开销，适用于固定 shape 在线推理低延迟场景。涵盖模式配置、FX Pass、编译缓存、多流并行、内存复用、静态 Kernel 编译、限核、性能优化、调试定位、自定义算子入图等。关键词：npugraph_ex、aclgraph、backend="npugraph_ex"、capture、replay、reduce-overhead、config.aclgraph_config。

- **torch-npugraph-ex-performance-diagnosis**
  - path: `.skills-cache/cannbot-skills/graph/torch-npugraph-ex-performance-diagnosis/SKILL.md`
  - desc: PyTorch 昇腾 NPU npugraph_ex 性能诊断（FX 图静态审计，聚焦 reinplace 未命中导致的冗余 tensor move）。处理「5 step 全部通过、但推理慢/Device 利用率低」阶段：基于 TORCH_COMPILE_DEBUG=1 产出的 FX 图序列与 debug.log，定位图里冗余的 tensor move——重点是 reinplace_inplaceable_ops_pass（out-of-place→in-place）与 reinplace_input_mutated_ops（折叠输入侧 copy_）回填未完成留下的 copy_/clon...

- **torch-npugraph-ex-runtime-error-diagnosis**
  - path: `.skills-cache/cannbot-skills/graph/torch-npugraph-ex-runtime-error-diagnosis/SKILL.md`
  - desc: PyTorch 昇腾 NPU npugraph_ex 运行时报错诊断。覆盖 ACL graph 已 capture 成功之后，replay / kernel launch / 通信 / 内存 / device API 阶段的报错排查，包括 aclnnXxx 算子失败、HCCL 错误、stream/event 同步、segfault、device side assert、OOM 等场景。本 skill 由 dfx-triage 路由进入。触发：当用户遇到 npugraph_ex replay / aclnn / HCCL / stream / OOM 等运行时报错时加载。关键词：runti...

- **torch-npugraph-ex-template**
  - path: `.skills-cache/cannbot-skills/graph/torch-npugraph-ex-template/SKILL.md`
  - desc: npugraph_ex 模式的 MRE（最小可复现示例）代码模板。包含标准 npugraph_ex 编译模板和 npugraph_ex 编译缓存（cache_compile）模板。触发：当用户需要从零生成 npugraph_ex 模式代码、做概念解释、对比分析或配置指导时加载。

- **torch-ops-profiler**
  - path: `.skills-cache/cannbot-skills/ops/torch-ops-profiler/SKILL.md`
  - desc: 使用 torch_npu.profiler（warmup/active=5）维护 JSONL 用例并输出自定义算子 vs 标杆的性能报告。触发场景：需要 profiler 对比算子性能时。细节见正文与 examples/。

- **triton-latency-optimizer**
  - path: `.skills-cache/cannbot-skills/ops/triton-latency-optimizer/SKILL.md`
  - desc: 擅长在 Ascend NPU 平台上编写高效 Triton 算子的性能优化专家。 按照严格的顺序逐步优化 Triton 代码，每次只尝试一个优化点， 确保优化前后功能一致、精度一致。 ⚠️ 只能使用本 skill 规定的优化方式，禁止使用任何超出本 skill 之外的优化方式。 触发：当用户需要对 Ascend NPU 上的 Triton 算子代码进行性能优化、降低时延、提升吞吐时使用。


- **triton-npu-convert**
  - path: `.skills-cache/cannbot-skills/plugins-community/triton-optimizer/skills/triton-npu-convert/SKILL.md`
  - desc: Use when the user asks to convert a PyTorch operator into a Triton NPU-backed PyTorch operator. Preserve the trailing input-helper block, and validate the converted output through standalone or differential testing.

- **triton-npu-optimize**
  - path: `.skills-cache/cannbot-skills/plugins-community/triton-optimizer/skills/triton-npu-optimize/SKILL.md`
  - desc: Iteratively optimize a Triton Ascend NPU operator with correctness and performance gates. Use for operator optimization tasks that need repeated correctness validation, benchmark validation, multi-round experiment tracking, reusable optimization notes, and profiler-backed performance analysis whe...

- **triton-op-coding**
  - path: `.skills-cache/cannbot-skills/ops/triton-op-coding/SKILL.md`
  - desc: Triton Ascend 算子代码生成 Skill — 根据算子任务格式任务描述生成高性能 Triton Ascend 内核代码。支持首次生成和基于错误反馈的迭代优化。 触发：当用户需要根据任务描述生成或迭代修复 Triton Ascend 内核代码时使用。


- **triton-op-designer**
  - path: `.skills-cache/cannbot-skills/ops/triton-op-designer/SKILL.md`
  - desc: Triton Ascend 算子算法草图设计 Skill — 根据任务描述设计高质量的算法草图（sketch）， 用于指导后续代码生成。支持首次设计和基于历史上下文的迭代优化。 触发：当用户需要为 Triton Ascend 算子设计算法草图或在已有 sketch 基础上迭代时使用。


- **triton-op-verifier**
  - path: `.skills-cache/cannbot-skills/ops/triton-op-verifier/SKILL.md`
  - desc: 算子代码验证 Skill — 按照标准验证流程验证生成的内核代码。 创建验证项目文件，调用 scripts/verify.py 运行验证，验证通过后 调用 scripts/benchmark.py 进行性能测试并收集结果。 触发：当用户需要验证 Triton 算子代码功能正确性或采集其性能数据时使用。


- **triton-precision-debug**
  - path: `.skills-cache/cannbot-skills/ops/triton-precision-debug/SKILL.md`
  - desc: Triton-Ascend 算子精度对齐调试专家。当算子精度校验（MERE/MARE）不通过时， 按照系统化的五阶隔离法定位 ULP 级差异根因，并提供修复方案。 尤其擅长处理编译器 scalar/vector 浮点行为差异、常量除法精度偏差等隐蔽问题。


- **triton-simulator-optimizer**
  - path: `.skills-cache/cannbot-skills/ops/triton-simulator-optimizer/SKILL.md`
  - desc: Triton-Ascend 算子的 simulator 流水**采集与诊断**专家。用 msprof op simulator 采集 per-instruction pipe 统计，定位真实瓶颈（而非猜测），产出**诊断报告**：瓶颈类型 + 热源码行 + 修复方向（指向 triton-latency-optimizer 的已有优化点编号）。 本 skill **只采集 + 诊断，不自带优化技术**——修复落地一律走 latency-optimizer， 禁止在本 skill 重复定义第二个优化技术目录。禁止在未采集前 declaring "硬件极限"。 触发：当用户需要通过 simul...

- **triton-task-extractor**
  - path: `.skills-cache/cannbot-skills/ops/triton-task-extractor/SKILL.md`
  - desc: 从用户 PyTorch/Python 代码中提取算子实现，构建为算子任务格式的标准化 任务文件。支持两种模式：单 case（单一自包含 .py，get_inputs 返回单组）和 多 case（.py + 同名 .json 配对，get_input_groups 返回多组）。 触发：当用户需要将 PyTorch/Python 代码提取并转换为标准化算子任务文件时使用。


- **tune-frontend**
  - path: `.skills-cache/cannbot-skills/ops/pypto-op-perf-tune/tune-frontend/SKILL.md`
  - desc: PyPTO 算子开箱性能调优技能。主要关注代码级的调优、前端写法不同导致的性能差异，包括 loop 写法优化、TileShape 设置优化、数据操作优化等。当用户需要进行算子初始开发性能优化、开箱性能调优时使用此技能。触发词：开箱性能调优、代码级优化、loop 优化、TileShape 设置、前端优化。

- **tune-incore**
  - path: `.skills-cache/cannbot-skills/ops/pypto-op-perf-tune/tune-incore/SKILL.md`
  - desc: PyPTO 算子核内性能调优技能。通过分析单 task 的实现指令及 operation，完成核内的性能调优，包括指令级优化、核内流水优化、特殊 Shape 处理等。当用户需要进行核内性能调优、单 task 耗时分析、指令级优化时使用此技能。触发词：核内性能调优、单 task 优化、指令级优化、核内流水、Operation 实现优化。

- **tune-orchestrator**
  - path: `.skills-cache/cannbot-skills/ops/pypto-op-perf-tune/tune-orchestrator/SKILL.md`
  - desc: PyPTO 算子性能调优编排器。控制调优流程按固定顺序严格执行、迭代轮次完整执行、Todo 清单实时更新。负责流程推进、完成条件校验、状态机转移，不负责具体优化建议或代码修改。适用于 PyPTO 算子性能调优流程编排时。

- **tune-swimlane**
  - path: `.skills-cache/cannbot-skills/ops/pypto-op-perf-tune/tune-swimlane/SKILL.md`
  - desc: PyPTO 算子深度性能调优技能。通过泳道图分析及调优性能，包括 Stitch 调优、TileShape 深度调优、合图调优、调度策略调优等。当用户需要进行深度性能调优、泳道图分析、Stitch 优化、合图优化时使用此技能。触发词：深度性能调优、泳道图分析、Stitch 调优、合图调优、调度优化。

- **workflow-agent-permissions**
  - path: `.skills-cache/cannbot-skills/plugins-community/cuda2ascend/skills/workflow-agent-permissions/SKILL.md`
  - desc: 派发写类任务前加载，判断目标角色是否具备对应目录写权限，避免无效派发；同时是权限规格真值源，init 物化后由权限插件加载执行。子仓可整体 override；仓内可单独调整。

- **workflow-cp0**
  - path: `.skills-cache/cannbot-skills/plugins-community/cuda2ascend/skills/workflow-cp0/SKILL.md`
  - desc: CP0 环境确认的验收标准 skill，供 QA 在环境确认点加载。触发：仅在明确调用时触发，不主动触发。

- **workflow-cp1**
  - path: `.skills-cache/cannbot-skills/plugins-community/cuda2ascend/skills/workflow-cp1/SKILL.md`
  - desc: CP1 需求确认的验收标准 skill，供 QA 在需求确认点加载。触发：仅在明确调用时触发，不主动触发。

- **workflow-cp2-1**
  - path: `.skills-cache/cannbot-skills/plugins-community/cuda2ascend/skills/workflow-cp2-1/SKILL.md`
  - desc: CP2.1 测试检查的验收标准 skill，供 QA 在测试方案检查点加载。触发：仅在明确调用时触发，不主动触发。

- **workflow-cp2-2**
  - path: `.skills-cache/cannbot-skills/plugins-community/cuda2ascend/skills/workflow-cp2-2/SKILL.md`
  - desc: CP2.2 方案检查的验收标准 skill，供 QA 在开发方案检查点加载。触发：仅在明确调用时触发，不主动触发。

- **workflow-cp3**
  - path: `.skills-cache/cannbot-skills/plugins-community/cuda2ascend/skills/workflow-cp3/SKILL.md`
  - desc: CP3 功能验收的验收标准 skill，供 QA 在功能验收点加载。触发：仅在明确调用时触发，不主动触发。

- **workflow-cp4**
  - path: `.skills-cache/cannbot-skills/plugins-community/cuda2ascend/skills/workflow-cp4/SKILL.md`
  - desc: CP4 性能验收的验收标准 skill，供 QA 在性能验收点加载。触发：仅在明确调用时触发，不主动触发。

- **workflow-cp5**
  - path: `.skills-cache/cannbot-skills/plugins-community/cuda2ascend/skills/workflow-cp5/SKILL.md`
  - desc: CP5 代码检视的验收标准 skill，供 QA 在代码检视点加载。触发：仅在明确调用时触发，不主动触发。

- **workflow-cp6**
  - path: `.skills-cache/cannbot-skills/plugins-community/cuda2ascend/skills/workflow-cp6/SKILL.md`
  - desc: CP6 CI 通过确认的验收标准 skill，供 QA 在上库前的收口确认点加载。触发：仅在明确调用时触发，不主动触发。

- **workflow-doc-templates**
  - path: `.skills-cache/cannbot-skills/plugins-community/cuda2ascend/skills/workflow-doc-templates/SKILL.md`
  - desc: 交付件模板，提供设计文档、验收报告等中间交付件的格式模板。触发：产出需求/方案/验收报告/算子文档/开发日志/Issue 等交付件时，先加载对应模板作为格式基准。

### Source: local (1 skills)

- **npu-arch-capability-check**
  - path: `skills/npu-arch-capability-check/SKILL.md`
  - desc: Use when asked to judge Ascend NPU model, SocVersion, NpuArch, architecture capability, architecture-specific code branches, or whether a CANN/Ascend C/model path can run on a target NPU. This skill is for evidence-based capability checks, not general NPU education.

### Conflicts (excluded from selection)

- `ascendc-operator-performance-optim`: 2 duplicate definitions
- `torch-npu-bisect`: 2 duplicate definitions
- `torch-npu-memory-analyzer`: 2 duplicate definitions
- `torch-npu-missing-dispatch-loop`: 2 duplicate definitions
- `torch-npu-op-integration`: 2 duplicate definitions

### Errors (excluded from selection)

- .skills-cache/ascend-agent-skills/official/MindSpeed/mindspeed-llm-auto-ut-skills/skills/analyse-coverage/SKILL.md: description too short (<20 chars)
- .skills-cache/ascend-agent-skills/official/MindStudio/skills/msmodeling-text-generate-executor/SKILL.md: missing frontmatter delimiters
- .skills-cache/ascend-agent-skills/official/PyTorch/npu-graph-skill/.claude/skills/npu-graph-log-analyzer/SKILL.md: missing frontmatter delimiters
- .skills-cache/ascend-agent-skills/official/PyTorch/npu-graph-skill/.claude/skills/npu-graph-mcp-integration/SKILL.md: missing frontmatter delimiters
- .skills-cache/ascend-agent-skills/official/PyTorch/npu-graph-skill/.claude/skills/npu-graph-mode-diagnostics/SKILL.md: missing frontmatter delimiters
- .skills-cache/ascend-agent-skills/official/PyTorch/npu-graph-skill/.claude/skills/npu-graph-mode-expert/SKILL.md: missing frontmatter delimiters
- .skills-cache/ascend-agent-skills/official/PyTorch/npu-graph-skill/.claude/skills/npu-graph-performance-profiling/SKILL.md: missing frontmatter delimiters
- .skills-cache/ascend-agent-skills/official/PyTorch/npu-graph-skill/.claude/skills/npu-graph-skill/SKILL.md: missing frontmatter delimiters
- .skills-cache/cannbot-skills/plugins-official/ops-registry-invoke/workflow/SKILL.md: description too short (<20 chars)
