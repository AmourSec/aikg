---
title: NPU 相关 AI Skills 样例
domain: knowledge-management
doc_type: guide
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-06-25
sources:
  - https://gitcode.com/cann/cannbot-skills
  - https://gitcode.com/cann/cannbot-skills/blob/master/README.md
---

# NPU 相关 AI Skills 样例

不是每篇知识库文章都需要写成 skill。科普文章适合给人建立理解，也适合 AI 检索背景知识；skill 更适合沉淀“可重复执行的工作流程”。

当前硬件基础章节先打一个样：`skills/npu-arch-capability-check/SKILL.md`。它不负责科普 NPU，而是指导 AI 在遇到昇腾 NPU 架构、型号、SocVersion、NpuArch、条件编译和能力判断问题时，按固定流程收集证据并输出结论。

## 什么内容适合写成 skill

满足下面条件的内容，适合从普通文档升级为 skill：

- 这件事会反复发生，例如环境检查、模型迁移、性能剖析、精度调试、故障定位。
- 它有稳定的输入，例如日志、配置、设备信息、benchmark、profiler、代码 diff。
- 它有明确步骤，例如先判断版本，再查映射，再验证最小 workload。
- 它有输出模板，例如“结论、证据、风险、下一步实验”。
- 它需要 AI 在多个文档之间做选择，而不是只复述一段知识。

不适合写成 skill 的内容：

- 只解释概念的科普文章。
- 尚未验证的个人猜想。
- 只对一次临时问题有效的聊天记录。
- 缺少输入、步骤和输出边界的经验片段。

## skill 与普通文档的区别

| 类型 | 面向谁 | 主要作用 |
| --- | --- | --- |
| 普通知识文档 | 人和 AI | 解释概念、流程、原理、背景和引用来源。 |
| Benchmark report | 人和 AI | 保存实验设计、环境、原始数据、结果和结论边界。 |
| ADR | 人和 AI | 保存技术选择、证据、取舍、回滚条件和复盘条件。 |
| Failure case | 人和 AI | 保存故障现象、定位过程、证据链、修复和预防措施。 |
| Skill | AI | 在特定任务中指导 AI 如何行动、追问、检查、输出。 |

## 本仓库的样例结构

```text
skills/
  README.md
  npu-arch-capability-check/
    SKILL.md
```

这个样例刻意保持很小，只覆盖一个任务：判断昇腾 NPU 架构能力和条件编译路径。后续可以继续扩展：

- `npu-env-baseline`：收集 CANN、driver、runtime、torch_npu、设备和容器信息。
- `npu-model-migration-baseline`：建立模型迁移的功能、精度、性能基线。
- `npu-inference-profiling-pack`：组织推理压测、profiler、KV Cache、调度和内存证据。
- `npu-operator-porting-review`：检查自定义算子的架构分支、tiling、片上存储和测试覆盖。
- `npu-training-hang-triage`：整理分布式训练 hang、collective、rank、网络和 checkpoint 证据。

## 程序员后续怎么补 skill

1. 先用普通 Markdown 记录一次真实工作：背景、环境、现象、操作、证据、结论。
2. 如果同类问题重复出现，把它抽象成 checklist。
3. 如果 checklist 需要 AI 按步骤执行，就写成 `skills/<skill-name>/SKILL.md`。
4. 在 skill 中写清楚触发条件、输入要求、流程、输出模板、风险边界和引用文档。
5. 更新 `llms.txt` / `llms-full.txt`，让 AI 能看到新的 skill 入口。
6. 用一个真实问题提示 AI，检查它是否会先补证据、再判断，而不是凭空给结论。

## 参考资料

- [CANNBot Skills 项目](https://gitcode.com/cann/cannbot-skills) 是 CANN 生态 skill 化组织方式的参考。
- [CANNBot README](https://gitcode.com/cann/cannbot-skills/blob/master/README.md) 展示了算子开发、模型推理优化、Triton、TileLang、profiling、simulator 等任务如何拆成多个 skill。

