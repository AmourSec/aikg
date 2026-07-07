---
title: AI Skills 编写指南
domain: knowledge-index
doc_type: guide
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-07-07
related:
  - ../12-hardware-basics/ai-skills-sample.md
  - ../../skills/npu-arch-capability-check/SKILL.md
---

# AI Skills 编写指南

AI Skills 是给 AI Agent 看的“任务工作流”。它不是科普文章，也不是把所有背景知识塞给模型，而是把团队反复做的事情沉淀成可触发、可执行、可验证的步骤。

在这个知识库里，`docs/` 负责让人和 AI 理解知识，`skills/` 负责让 AI 在具体任务中按流程行动。

可以先记住一句话：

```text
普通文档讲清楚知识；skill 规定 AI 怎么做事。
```

## 放在哪

本仓库的 skill 放在根目录的 `skills/` 下：

```text
skills/
  README.md
  <skill-name>/
    SKILL.md
    references/   # 可选
    scripts/      # 可选
    assets/       # 可选
```

当前已有样例：

```text
skills/npu-arch-capability-check/SKILL.md
```

它对应硬件章节里的说明页：[硬件适配 AI Skills 样例](../12-hardware-basics/ai-skills-sample.md)。

## 什么内容适合写成 skill

适合写成 skill 的内容通常有五个特征：

| 判断问题 | 适合写 skill 的信号 |
| --- | --- |
| 是否重复发生 | 同类任务会反复出现，例如环境基线、模型迁移、性能剖析、故障定位、算子 review。 |
| 是否有稳定输入 | 日志、配置、benchmark、profiler、代码 diff、设备信息、错误栈等可以作为输入。 |
| 是否有固定步骤 | 先收集证据，再判断版本，再跑最小验证，再输出结论。 |
| 是否有输出模板 | 输出可以稳定分成结论、证据、风险、缺失信息、下一步实验。 |
| 是否需要 AI 行动 | 需要 AI 搜索文件、检查配置、执行命令、比较证据，而不是只复述知识。 |

不适合写成 skill 的内容：

- 纯概念解释；
- 入门科普；
- 只发生一次的临时聊天记录；
- 没有输入、步骤和输出边界的经验片段；
- 尚未验证、没有证据来源的猜想。

这些内容应该先写成普通文档、benchmark report、ADR 或 failure case。等同类任务重复出现，再抽象成 skill。

## skill 与文档类型的关系

| 类型 | 主要读者 | 作用 |
| --- | --- | --- |
| Primer / Concept | 人和 AI | 建立概念、解释原理、提供背景。 |
| Benchmark Report | 人和 AI | 保存实验设计、环境、数据、结果和结论边界。 |
| ADR | 人和 AI | 保存技术决策、证据、取舍、回滚条件。 |
| Failure Case | 人和 AI | 保存故障现象、定位过程、证据链、修复和预防。 |
| Runbook | 人和 AI | 指导人或系统按步骤处理事故或例行任务。 |
| Skill | AI | 指导 AI 在特定任务中收集证据、执行步骤、输出结论。 |

实际维护时，建议顺序是：

```mermaid
flowchart LR
  A["真实工作记录\n日志 / 命令 / 结论"] --> B["普通文档或案例\nbenchmark / ADR / failure case"]
  B --> C["Checklist\n可重复步骤"]
  C --> D["Skill\nAI 可执行工作流"]
```

不要从一段概念解释直接跳到 skill。skill 应该来自真实任务和反复验证过的流程。

## 命名规则

skill 名称使用小写字母、数字和短横线：

```text
npu-env-baseline
npu-model-migration-baseline
npu-operator-porting-review
inference-benchmark-pack
training-hang-triage
adr-review
failure-case-writer
```

命名要表达“这个 skill 让 AI 做什么”。不要用太泛的名字：

| 不建议 | 更好 |
| --- | --- |
| `npu` | `npu-env-baseline` |
| `benchmark` | `inference-benchmark-pack` |
| `debug` | `training-hang-triage` |
| `docs` | `failure-case-writer` |

目录名和 front matter 里的 `name` 保持一致：

```text
skills/npu-env-baseline/SKILL.md
```

```yaml
---
name: npu-env-baseline
description: ...
---
```

## 最小目录结构

最小 skill 只需要一个文件：

```text
skills/npu-env-baseline/
  SKILL.md
```

当内容变复杂，再加资源目录：

| 目录 | 放什么 | 什么时候需要 |
| --- | --- | --- |
| `references/` | 长参考资料、字段说明、案例、检查表。 | `SKILL.md` 太长，或者不同场景需要不同资料。 |
| `scripts/` | 可执行脚本。 | 操作需要稳定、可重复，靠 AI 临时写代码风险高。 |
| `assets/` | 模板、配置、图片、示例文件。 | skill 执行时需要复制、生成或引用固定资产。 |

不要在 skill 目录里堆 `README.md`、`CHANGELOG.md`、`QUICKSTART.md` 这类给人看的附属文档。skill 目录的目标是让 AI 快速行动，不是再做一个小型文档站。

## SKILL.md 标准结构

`SKILL.md` 分成两部分：

1. YAML front matter；
2. Markdown 正文。

### Front Matter

只放必要字段：

```yaml
---
name: npu-env-baseline
description: Use when asked to collect or verify Ascend NPU environment baseline, including CANN, driver, firmware, runtime, torch_npu, device model, SocVersion, NpuArch, container, and framework versions. This skill is for evidence collection before debugging or benchmarking.
---
```

`description` 很重要，因为 AI 会先看到它，再决定是否加载正文。写 `description` 时要包含：

- 这个 skill 做什么；
- 什么时候应该使用；
- 典型触发词；
- 不要用于什么场景。

不要只写：

```yaml
description: NPU environment skill.
```

这种描述太模糊，AI 不知道什么时候该触发。

### 正文

正文建议用下面结构：

````markdown
# NPU Environment Baseline

## Scope

Use this skill for ...
Do not use this skill for ...

## Required Inputs

- ...

## Workflow

1. ...
2. ...
3. ...

## Output Template

```markdown
...
```

## Local Knowledge References

- `docs/...`
````

正文要短，优先写流程。不要把整篇背景知识复制进来。

## 最小模板

可以直接复制下面模板：

````markdown
---
name: your-skill-name
description: Use when asked to <task>. This skill is for <scope>, especially when the task mentions <trigger words>. Do not use it for <out-of-scope>.
---

# Your Skill Name

## Scope

Use this skill when the task requires ...

Do not use this skill for ...

## Required Inputs

Ask for or collect:

- ...
- ...
- ...

## Workflow

1. Identify the task boundary.
2. Collect required evidence.
3. Check the relevant files, logs, configs, or benchmark artifacts.
4. Separate confirmed facts from assumptions.
5. Produce the output using the template below.

## Output Template

```markdown
## Summary

- Conclusion:
- Confidence:

## Evidence

- ...

## Missing Inputs

- ...

## Risks

- ...

## Next Steps

1. ...
2. ...
```

## Local Knowledge References

- `docs/...`
````

## 稍完整示例：NPU 环境基线

````markdown
---
name: npu-env-baseline
description: Use when asked to collect or verify Ascend NPU environment baseline, including CANN, driver, firmware, runtime, torch_npu, device model, SocVersion, NpuArch, container, and framework versions. This skill is for evidence collection before debugging or benchmarking, not for general NPU education.
---

# NPU Environment Baseline

## Scope

Use this skill before NPU debugging, benchmarking, model migration, or operator performance analysis.

Do not use this skill to explain basic NPU architecture. Use documentation pages for primers.

## Required Inputs

Ask for or collect:

- device inventory, such as `npu-smi info`;
- CANN Toolkit, Runtime, Driver, and firmware versions;
- framework version, such as PyTorch and `torch_npu`;
- container image or host OS;
- target workload command;
- relevant environment variables;
- error log, profiler trace, or benchmark report if available.

## Workflow

1. Record hardware identity.
2. Record software stack versions.
3. Record workload command and key runtime flags.
4. Check whether the versions are mutually compatible.
5. Mark missing evidence instead of guessing.
6. Output a baseline report.

## Output Template

```markdown
## NPU Environment Baseline

### Summary
- Target:
- Status:
- Confidence:

### Hardware
- Device:
- SocVersion:
- NpuArch:
- Topology:

### Software
- CANN:
- Driver:
- Firmware:
- Framework:
- torch_npu:
- Runtime / engine:

### Workload
- Command:
- Model:
- Precision:
- Shape / batch / sequence:

### Evidence
- ...

### Missing Inputs
- ...

### Next Steps
1. ...
```

## Local Knowledge References

- `docs/12-hardware-basics/ascend-npu-models.md`
- `docs/12-hardware-basics/cann-stack.md`
- `docs/12-hardware-basics/ascend-910-series.md`
- `docs/12-hardware-basics/ascend-950-series.md`
````

## references 怎么写

如果 skill 需要较长参考资料，不要塞进 `SKILL.md`，放到 `references/`：

```text
skills/npu-operator-porting-review/
  SKILL.md
  references/
    tiling-checklist.md
    architecture-branches.md
    common-failure-patterns.md
```

在 `SKILL.md` 里写清楚什么时候读取：

```markdown
## References

- Read `references/tiling-checklist.md` when reviewing tiling logic.
- Read `references/architecture-branches.md` when code contains `__NPU_ARCH__`, `DAV_`, `SocVersion`, or `archXX`.
- Read `references/common-failure-patterns.md` when the task includes compile errors, runtime errors, or wrong results.
```

引用规则：

- 一层引用即可，避免 references 里再链接一堆更深文件；
- 长文件开头加目录；
- 一个 reference 只解决一类问题；
- 不要复制普通文档已有的大段内容，优先链接 `docs/` 路径。

## scripts 怎么放

如果某个步骤需要稳定执行，放脚本比让 AI 每次临时写更可靠。

适合放脚本的场景：

- 解析 benchmark manifest；
- 汇总 profiler 输出；
- 检查 front matter；
- 生成环境 baseline；
- 对日志做脱敏；
- 把固定格式报告转换为 Markdown。

目录示例：

```text
skills/inference-benchmark-pack/
  SKILL.md
  scripts/
    summarize_benchmark.py
    validate_manifest.py
```

`SKILL.md` 里要写清楚脚本如何调用：

```markdown
Run `scripts/validate_manifest.py <manifest.json>` before summarizing results.
If validation fails, report the exact missing fields and do not draw benchmark conclusions.
```

脚本必须能独立运行，并且有明确输入输出。不要把实验结论硬编码进脚本。

## assets 怎么放

`assets/` 放 AI 执行任务时要用的固定素材，例如：

- 报告模板；
- 配置模板；
- dashboard JSON；
- 示例输入；
- 代码骨架；
- 图片或格式文件。

示例：

```text
skills/adr-review/
  SKILL.md
  assets/
    adr-review-template.md
```

如果只是给 AI 读取的文字说明，优先放 `references/`，不要放 `assets/`。

## 从真实工作转成 skill

推荐流程：

1. 先写真实记录。
   - 背景；
   - 环境；
   - 输入；
   - 命令；
   - 证据；
   - 结论；
   - 复盘。

2. 抽出 checklist。
   - 哪些信息必须收集；
   - 哪些判断必须有证据；
   - 哪些错误最常见；
   - 输出应该长什么样。

3. 写成 `SKILL.md`。
   - 触发条件；
   - 输入要求；
   - 工作流；
   - 输出模板；
   - 参考文档。

4. 用真实问题验证。
   - AI 是否先补证据；
   - 是否避免凭空下结论；
   - 输出是否可复查；
   - 是否知道缺失信息时应该停在“未知”。

5. 根据验证结果修改。
   - 太泛就收窄 scope；
   - 太长就拆 references；
   - 容易误触发就改 description；
   - 输出不稳定就收紧模板。

## 更新 AI 入口

新增 skill 后，需要让 `llms.txt` 和 `llms-full.txt` 收录。

本仓库的生成脚本会扫描：

```text
skills/**/SKILL.md
```

新增后运行：

```bash
python3 scripts/generate_llms_files.py
```

如果希望 `llms.txt` 里的描述更准确，修改 `scripts/generate_llms_files.py`：

```python
SKILL_DESCRIPTIONS = {
    "skills/npu-arch-capability-check/SKILL.md": "...",
    "skills/npu-env-baseline/SKILL.md": "NPU 环境基线 skill，指导 AI 收集设备、CANN、driver、runtime、framework 和 workload 证据。",
}
```

如果这个 skill 很重要，也可以加入 `PRIORITY_SKILLS`，让它在索引里更靠前。

最后构建检查：

```bash
.venv/bin/mkdocs build --strict
git diff --check
```

## 质量检查清单

新增或修改 skill 时检查：

- [ ] 目录名是否小写短横线；
- [ ] `SKILL.md` 是否存在；
- [ ] front matter 是否只有 `name` 和 `description`；
- [ ] `name` 是否和目录名一致；
- [ ] `description` 是否写清触发条件；
- [ ] 是否明确“不适用”的范围；
- [ ] 是否列出 required inputs；
- [ ] workflow 是否是可执行步骤；
- [ ] 输出模板是否稳定；
- [ ] 是否区分事实、推断、缺失信息；
- [ ] 是否引用相关 `docs/` 文档；
- [ ] 长内容是否拆到 `references/`；
- [ ] 脚本是否可运行；
- [ ] 是否更新 `llms.txt` / `llms-full.txt`；
- [ ] 是否用真实问题验证过。

## 常见错误

### 把科普文章写成 skill

错误信号：

- 全文都在解释概念；
- 没有输入要求；
- 没有步骤；
- 没有输出模板。

处理方式：放回 `docs/`，不要放 `skills/`。

### description 太短

错误示例：

```yaml
description: Benchmark skill.
```

更好的写法：

```yaml
description: Use when asked to design, run, review, or package an AI inference benchmark, including workload contract, metrics, warmup, repetitions, profiler evidence, result tables, and conclusion boundaries. This skill is for benchmark workflow control, not general performance theory.
```

### 把全部知识塞进 SKILL.md

`SKILL.md` 太长会浪费上下文。处理方式：

- 核心流程留在 `SKILL.md`；
- 长表格放 `references/`；
- 可执行逻辑放 `scripts/`；
- 背景知识链接到 `docs/`。

### 输出没有证据边界

AI 很容易在信息不足时给“看起来合理”的结论。skill 必须要求：

- 缺证据时输出 `unknown`；
- benchmark 结论必须有测量口径；
- 性能瓶颈必须有 profiler 或消融证据；
- 版本兼容性必须有日志、文档或实际命令输出。

### skill 过大

一个 skill 只做一类任务。不要写成：

```text
npu-all-in-one
```

应该拆成：

```text
npu-env-baseline
npu-model-migration-baseline
npu-operator-porting-review
npu-inference-profiling-pack
npu-training-hang-triage
```

## 推荐起步方向

结合当前知识库，后续可以优先补这些 skill：

| Skill | 作用 |
| --- | --- |
| `npu-env-baseline` | 收集 CANN、driver、runtime、torch_npu、设备和容器信息。 |
| `npu-model-migration-baseline` | 建立模型迁移的功能、精度、性能和回退路径。 |
| `npu-operator-porting-review` | 检查自定义算子的架构分支、tiling、片上存储、同步和测试覆盖。 |
| `npu-inference-profiling-pack` | 组织推理压测、profiler、KV Cache、调度和内存证据。 |
| `training-hang-triage` | 整理分布式训练 hang、collective、rank、网络和 checkpoint 证据。 |
| `benchmark-report-review` | 检查 benchmark 问题、workload、指标、实验设计和结论边界。 |
| `adr-review` | 检查技术决策是否有足够证据、回滚条件和复盘条件。 |
| `failure-case-writer` | 把事故、压测失败或性能回归整理成可复查案例。 |

先从最常重复、最容易标准化的工作开始，不要一次性追求完整体系。

## 最小提交流程

新增一个 skill 后，建议一次提交包括：

```text
skills/<skill-name>/SKILL.md
skills/<skill-name>/references/...      # 如果有
skills/<skill-name>/scripts/...         # 如果有
scripts/generate_llms_files.py          # 如果补了描述或优先级
docs/llms.txt
docs/llms-full.txt
llms.txt
llms-full.txt
```

命令：

```bash
python3 scripts/generate_llms_files.py
.venv/bin/mkdocs build --strict
git diff --check
git add skills scripts/generate_llms_files.py docs/llms.txt docs/llms-full.txt llms.txt llms-full.txt
git commit -m "Add <skill-name> skill"
```

如果只是新增普通文档，不需要放进 `skills/`。如果新增的是 AI 可执行工作流，就应该同步更新 AI 入口索引。
