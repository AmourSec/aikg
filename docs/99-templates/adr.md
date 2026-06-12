---
title: 技术决策模板：从问题、证据到可追溯决策
domain: template
doc_type: template
status: reviewed
owner: maintainers
license: CC-BY-4.0
updated: 2026-06-12
---

# 技术决策模板：从问题、证据到可追溯决策

这个模板用于撰写 AI Infra 知识库中的 ADR。

ADR 是 Architecture Decision Record，也可以更宽泛地理解为 technical decision record。

它记录的不是“我们做了什么”，而是：

> 在什么背景下，面对什么约束，比较了哪些方案，基于什么证据，为什么选择这个方案，并且以后在什么条件下要重新评估。

AI Infra 的 ADR 和普通软件系统 ADR 有相同的骨架：

```text
context -> options -> decision -> consequences -> follow-up
```

但它还必须特别关注：

- workload；
- SLO；
- benchmark；
- profiler；
- capacity；
- cost；
- hardware topology；
- reliability；
- rollout；
- rollback；
- reproducibility。

如果没有这些字段，AI Infra 决策很容易变成“某人觉得这个方案更先进”。

高质量 ADR 的目标不是证明当前选择永远正确，而是让未来读者和 AI 能够判断：

- 当时的问题是什么；
- 当时有哪些硬约束；
- 当时为什么没选其他方案；
- 证据够不够强；
- 决策影响哪些系统对象；
- 什么时候应该废弃或替换这个决策。

## 使用方式

新增 ADR 时，建议按下面流程：

1. 先写清 decision question。
2. 再写 scope 和 workload contract。
3. 列出真实候选方案，包括“不做”。
4. 明确 decision drivers。
5. 给每个候选方案填证据。
6. 做出 decision。
7. 写 consequences、rollout、rollback 和 revisit condition。
8. 加入 AI-readable ADR card。
9. 更新相关知识点、benchmark、failure case 或 runbook 链接。

本模板适合：

- 推理引擎选型；
- 训练框架选型；
- 并行策略选择；
- KV Cache 管理策略选择；
- 量化策略选择；
- 网络、存储、调度系统选择；
- compiler、kernel、runtime 路线选择；
- 硬件规格、集群拓扑和容量模型选择；
- SLO、降级策略和可靠性策略选择。

本模板不适合：

- 解释一个概念；
- 记录一次实验结果；
- 记录一次故障；
- 写操作手册；
- 写会议纪要；
- 写没有长期影响的小改动。

如果只是解释机制，使用知识点模板。

如果只是记录实验，使用基准实验报告模板。

如果只是记录事故，使用 failure case 或 postmortem 结构。

## 文件命名

推荐命名：

```text
ADR-0001-select-vllm-for-general-llm-serving.md
ADR-0002-use-paged-kv-cache-for-long-context-serving.md
ADR-0003-adopt-fsdp-for-7b-post-training.md
```

命名规则：

- 编号单调递增；
- 编号不要复用；
- 标题使用动词或决策对象；
- 文件名不要只写 `decision.md`；
- 被替代的 ADR 保留，不要删除。

推荐标题：

```text
ADR-0000: 选择 vLLM 作为通用 LLM Serving 默认引擎
```

不推荐标题：

```text
vLLM 调研
```

原因是 ADR 记录的是决策，不是调研过程。

## 状态流转

ADR 状态建议使用固定枚举。

| 状态 | 含义 |
| --- | --- |
| proposed | 已提出，尚未接受 |
| accepted | 已接受，进入实施或已实施 |
| rejected | 明确不采用 |
| deferred | 暂缓，等待更多证据或条件成熟 |
| superseded | 被后续 ADR 替代 |
| deprecated | 不再推荐，但可能仍存在历史依赖 |
| abandoned | 决策未完成，且不再推进 |

状态流转示例：

```text
proposed -> accepted -> superseded
proposed -> rejected
proposed -> deferred -> accepted
accepted -> deprecated -> superseded
```

状态不是形式字段。

它决定读者和 AI 是否应该继续引用这篇 ADR。

例如：

- `accepted` 可以作为当前架构事实；
- `proposed` 只能作为讨论材料；
- `superseded` 必须链接替代 ADR；
- `deprecated` 必须说明仍然残留在哪里。

## 最小可复制模板

下面是最小版本。

适合快速记录一个已经讨论清楚的技术决策。

````markdown
---
title: "ADR-0000: "
domain: technical-decision-records
doc_type: adr
status: proposed
owner: maintainers
license: CC-BY-4.0
updated: 2026-06-12
decision_status: proposed
workload: []
system_layer: []
hardware: []
software: []
metrics: []
evidence_level: E1
related:
  - ""
supersedes: []
superseded_by: []
---

# ADR-0000: 标题

## Status

Proposed.

## Decision Question

我们要在什么范围内决定什么？

> 面对什么 workload、SLO、硬件和软件约束，我们是否选择某个方案？

## Context

说明背景和约束。

- workload:
- target metrics:
- current bottleneck:
- hardware:
- software:
- constraints:

## Options

| Option | Summary | Pros | Cons | Evidence |
| --- | --- | --- | --- | --- |
| A |  |  |  |  |
| B |  |  |  |  |
| Do nothing |  |  |  |  |

## Decision

我们决定：

原因：

不选择其他方案的原因：

## Consequences

正面影响：

-

负面影响：

-

中性影响：

-

## Rollout and Rollback

上线方式：

回滚方式：

## Follow-up

- [ ] 需要补充的 benchmark：
- [ ] 需要新增的监控：
- [ ] 需要更新的文档：
- [ ] 需要复审的条件：

## References

-
````

## 完整推荐模板

下面是完整版本。

适合 AI Infra 中影响较大、成本较高、会被后续系统或 AI 反复引用的决策。

````markdown
---
title: "ADR-0000: "
domain: technical-decision-records
doc_type: adr
status: proposed
owner: maintainers
license: CC-BY-4.0
updated: 2026-06-12

decision_status: proposed
decision_date: 2026-06-12
deciders: []
reviewers: []
consulted: []
informed: []

workload:
  - ""
system_layer:
  - ""
hardware:
  - ""
software:
  - ""
metrics:
  - ""

evidence_level: E1
evidence:
  - type: ""
    title: ""
    url: ""
related:
  - ""
supersedes: []
superseded_by: []
revisit:
  date: ""
  condition: ""
---

# ADR-0000: 标题

## Status

Proposed / Accepted / Rejected / Deferred / Superseded / Deprecated.

如果被替代，写：

```text
Superseded by ADR-0000: ...
```

如果替代旧决策，写：

```text
Supersedes ADR-0000: ...
```

## Decision Summary

用 3 到 5 句话说明：

1. 我们决定什么。
2. 为什么现在需要决定。
3. 主要证据是什么。
4. 主要收益和代价是什么。
5. 什么时候需要重新评估。

## Decision Question

用一句话写清楚本文回答的问题。

建议格式：

```text
在 <workload / scale / SLO / hardware / software> 条件下，
我们是否应该采用 <option>，
以改善 <primary metric>，
同时不破坏 <guardrail metrics>？
```

例：

```text
在长上下文 LLM serving 场景下，我们是否应该采用 PagedAttention 管理 KV Cache，
以提升 goodput at SLO，同时控制 TTFT、显存碎片和实现复杂度？
```

## Context

说明背景，只写事实和约束。

建议包含：

- 当前系统是什么；
- 当前瓶颈是什么；
- 这个问题为什么现在必须处理；
- 不做决策会发生什么；
- 已有决策有哪些约束；
- 哪些范围不在本文讨论。

### Workload Contract

| 字段 | 内容 |
| --- | --- |
| workload |  |
| model |  |
| input shape |  |
| output shape |  |
| concurrency |  |
| precision |  |
| traffic pattern |  |
| SLO |  |
| failure budget |  |

### System Scope

| 层级 | 是否涉及 | 说明 |
| --- | --- | --- |
| model |  |  |
| runtime |  |  |
| scheduler |  |  |
| kernel |  |  |
| compiler |  |  |
| memory |  |  |
| communication |  |  |
| storage |  |  |
| cluster |  |  |
| observability |  |  |

## Decision Drivers

列出影响决策的因素。

每个 driver 都要说明权重或优先级。

| Driver | Priority | Why it matters | Metric or evidence |
| --- | --- | --- | --- |
| latency | must-have / should-have / nice-to-have |  |  |
| throughput |  |  |  |
| cost |  |  |  |
| reliability |  |  |  |
| reproducibility |  |  |  |
| maintainability |  |  |  |
| portability |  |  |  |

注意：

- driver 不是愿望清单；
- 不可能所有目标都是最高优先级；
- 如果目标冲突，要明确取舍顺序。

## Options Considered

至少包含：

- 选择 A；
- 选择 B；
- 继续保持现状；
- 分阶段方案；
- 暂缓决策。

| Option | Description | Pros | Cons | Evidence | Decision |
| --- | --- | --- | --- | --- | --- |
| A |  |  |  |  | selected / rejected |
| B |  |  |  |  | selected / rejected |
| Do nothing |  |  |  |  | selected / rejected |

## Evidence

### Evidence Matrix

| Evidence | Source | Supports | Limitations |
| --- | --- | --- | --- |
| paper |  |  |  |
| official docs |  |  |  |
| benchmark |  |  |  |
| profiler |  |  |  |
| trace |  |  |  |
| incident |  |  |  |
| source code |  |  |  |
| expert review |  |  |  |

### Benchmark Evidence

如果决策依赖 benchmark，必须写清：

```yaml
benchmark:
  question: ""
  workload: ""
  baseline: ""
  variants: []
  metrics:
    primary: []
    guardrail: []
  environment:
    hardware: ""
    software: ""
    config: ""
  raw_data: ""
  run_manifest: ""
  profiler_trace: ""
```

不要只写：

```text
实验显示 A 更快。
```

应该写：

```text
在 workload W、硬件 H、版本 V 和配置 C 下，A 相比 baseline 在 primary metric M 上改善，
同时 guardrail metrics G1/G2 未越过阈值。该结论尚未覆盖 workload W2 和硬件 H2。
```

### Evidence Level

建议使用证据等级。

| 等级 | 含义 |
| --- | --- |
| E0 | 直觉、经验判断或未验证假设 |
| E1 | 文档、论文、公开 benchmark 或专家判断 |
| E2 | 本地可复现实验，有 workload、baseline、metrics 和环境记录 |
| E3 | 多场景 sweep、profiler、稳定性和成本证据完整 |
| E4 | 线上灰度或生产数据验证，并有回滚和监控闭环 |

ADR 可以在 E1 阶段被 proposed。

进入 accepted 前，通常至少需要 E2。

高风险、不可逆或高成本决策通常需要 E3 或 E4。

## Decision

清楚写出最终选择。

建议使用主动句：

```text
我们决定采用 <option> 作为 <scope> 的默认方案。
```

同时写清：

- 决策范围；
- 生效条件；
- 不包含的范围；
- 哪些方案被拒绝；
- 拒绝原因；
- 是否分阶段实施；
- 是否需要 feature flag；
- 是否需要 fallback。

## Consequences

Nygard 风格 ADR 强调 consequences 不只写正面后果。

AI Infra 决策尤其要把代价写出来。

### Positive Consequences

-

### Negative Consequences

-

### Neutral Consequences

-

### Cost Model Impact

| 成本项 | 影响 |
| --- | --- |
| compute |  |
| memory capacity |  |
| memory bandwidth |  |
| network communication |  |
| storage |  |
| engineering complexity |  |
| operational cost |  |
| power / energy |  |

### Reliability Impact

| 方面 | 影响 |
| --- | --- |
| failure domain |  |
| recovery |  |
| fallback |  |
| observability |  |
| debugging |  |
| upgrade risk |  |

## Implementation Plan

ADR 不需要写成详细任务拆解，但必须说明如何落地。

| Phase | Scope | Exit Criteria |
| --- | --- | --- |
| phase 0 | prototype |  |
| phase 1 | benchmark |  |
| phase 2 | limited rollout |  |
| phase 3 | default path |  |

## Rollout Plan

写清上线策略。

- 灰度范围：
- feature flag：
- traffic split：
- fallback path：
- monitoring：
- alert threshold：
- owner：

## Rollback Plan

每个高影响 ADR 都应该有回滚策略。

写清：

- 如何判断需要回滚；
- 谁能触发回滚；
- 回滚命令或配置在哪里；
- 回滚后是否需要数据修复；
- 回滚是否会丢失状态；
- 回滚后如何复盘。

## Confirmation

说明如何证明这个决策被正确执行。

| Claim | Evidence |
| --- | --- |
| implementation shipped |  |
| benchmark passed |  |
| SLO not degraded |  |
| dashboard exists |  |
| alert exists |  |
| runbook updated |  |
| docs updated |  |

## Revisit Conditions

写清什么时候要重新评估。

常见条件：

- workload 变化；
- 模型规模变化；
- context length 变化；
- 新硬件上线；
- runtime 主版本升级；
- driver/CUDA/ROCm/NCCL 变化；
- SLO 变化；
- 成本模型变化；
- 事故或性能回归；
- 替代方案成熟。

例：

```text
如果长上下文请求占比超过 40%，或 p99 TTFT 连续两周超过 SLO，
则重新评估当前 KV Cache 管理策略。
```

## Links

链接相关资料。

| 类型 | 链接 |
| --- | --- |
| 知识点 |  |
| benchmark report |  |
| failure case |  |
| runbook |  |
| paper note |  |
| source code |  |
| dashboard |  |

## AI-readable ADR Card

重要 ADR 建议包含结构化卡片。

```yaml
adr_card:
  id: "ADR-0000"
  title: ""
  status: "proposed"
  decision: ""
  scope:
    workload: []
    system_layer: []
    hardware: []
    software: []
  drivers:
    primary: []
    guardrail: []
  options:
    selected: ""
    rejected: []
  evidence:
    level: "E1"
    sources: []
    benchmark_reports: []
  consequences:
    positive: []
    negative: []
    neutral: []
  rollout:
    strategy: ""
    fallback: ""
  revisit:
    date: ""
    conditions: []
  related_docs: []
```

AI 检索时，这个 card 应该能回答：

- 当前决策是什么；
- 是否仍然有效；
- 适用范围是什么；
- 证据强度如何；
- 为什么没选替代方案；
- 什么时候要重新评估。

## Checklist

### Decision Readiness

- [ ] decision question 是否明确？
- [ ] scope 是否明确？
- [ ] workload contract 是否完整？
- [ ] decision drivers 是否有优先级？
- [ ] 是否列出至少两个真实候选方案？
- [ ] 是否包含 do nothing？
- [ ] 是否写明不可讨论范围？

### Evidence

- [ ] 是否说明 evidence level？
- [ ] 是否区分事实、推论和假设？
- [ ] benchmark 是否包含 workload、baseline、metrics 和环境？
- [ ] profiler 或 trace 是否可追溯？
- [ ] 是否说明证据的局限性？
- [ ] 是否避免把供应商宣传材料当成验证结论？

### Decision Quality

- [ ] decision 是否用主动句写清？
- [ ] 是否说明为什么选中该方案？
- [ ] 是否说明为什么拒绝其他方案？
- [ ] 是否说明正面、负面和中性后果？
- [ ] 是否说明成本影响？
- [ ] 是否说明可靠性影响？

### Execution

- [ ] rollout 是否可执行？
- [ ] rollback 是否可执行？
- [ ] confirmation 是否有可观测证据？
- [ ] 是否新增必要监控？
- [ ] 是否更新 runbook？
- [ ] 是否更新相关知识点和 benchmark 报告？

### Maintenance

- [ ] 状态是否正确？
- [ ] 如果 superseded，是否链接替代 ADR？
- [ ] 是否写明 revisit condition？
- [ ] 是否更新 `updated`？
- [ ] 是否加入 `mkdocs.yml`？
- [ ] 是否更新 `scripts/generate_llms_files.py`？
- [ ] 是否运行生成和构建命令？

## References

-
````

## 字段说明

### decision_status

`decision_status` 是给 AI 和自动化脚本读取的状态字段。

建议与正文 `Status` 保持一致。

```yaml
decision_status: accepted
```

允许值：

- proposed；
- accepted；
- rejected；
- deferred；
- superseded；
- deprecated；
- abandoned。

### decision_date

表示决策日期，不一定等于文档最后更新时间。

```yaml
decision_date: 2026-06-12
updated: 2026-06-12
```

如果后续只更新链接或补充证据，`updated` 可以变化，但 `decision_date` 不应随意变化。

### deciders、reviewers、consulted、informed

这些字段用于保留决策责任和沟通边界。

```yaml
deciders:
  - runtime-maintainers
reviewers:
  - serving-platform
consulted:
  - benchmark-team
  - cluster-ops
informed:
  - users
```

公开知识库不一定保留真实个人姓名，可以使用角色或维护者组。

### workload

AI Infra 决策必须写 workload。

同一个方案在不同 workload 下可能结论相反。

例如：

```yaml
workload:
  - llm-serving
  - long-context
  - rag
```

不要只写：

```yaml
workload:
  - ai
```

### system_layer

写清决策影响的系统层。

```yaml
system_layer:
  - runtime
  - scheduler
  - memory
  - communication
```

### metrics

只写和决策有关的指标。

```yaml
metrics:
  - ttft_p99
  - tpot_p99
  - goodput_at_slo
  - gpu_memory_peak
  - cost_per_output_token
```

指标应分为：

- primary metrics；
- guardrail metrics；
- debugging metrics。

### evidence_level

证据等级用于避免“看起来合理”的决策被误认为已验证结论。

```yaml
evidence_level: E2
```

如果 evidence_level 只有 E0 或 E1，ADR 可以存在，但应保持 `proposed` 或 `deferred`，除非决策风险很低。

### supersedes 和 superseded_by

用于表达决策演化。

```yaml
supersedes:
  - ADR-0003
superseded_by:
  - ADR-0012
```

旧 ADR 不应删除。

旧 ADR 是未来理解系统演化的重要证据。

## 写作规则

### 一篇 ADR 只回答一个决策问题

不好：

```text
选择推理引擎、KV Cache 策略、量化策略和部署方式。
```

好：

```text
选择通用 LLM Serving 默认推理引擎。
```

如果多个决策互相依赖，可以拆成多篇 ADR，并用 `related` 链接。

### 把 Context 写成事实，不要写成辩论

Context 应该描述约束。

它不应该提前替某个方案辩护。

不好：

```text
方案 A 明显更先进，所以我们应该迁移。
```

好：

```text
当前系统在 8k context、p99 TTFT 和显存碎片方面存在压力。方案 A 提供了针对该问题的机制，但需要额外验证迁移成本和稳定性。
```

### Options 要包含被拒绝方案

ADR 最有价值的部分往往不是“选了什么”，而是“为什么没选别的”。

未来系统变化后，被拒绝方案可能重新变得合理。

因此，每个 rejected option 都应该写：

- 拒绝原因；
- 当前证据；
- 未来可能改变结论的条件。

### Decision 要能执行

不好：

```text
我们倾向于使用方案 A。
```

好：

```text
我们决定在在线 LLM serving 的通用路径中采用方案 A，先覆盖 7B/13B dense 模型，MoE 和 long-context 变体暂不纳入默认路径。
```

### Consequences 必须写负面后果

只写好处的 ADR 不可信。

任何技术选择都会带来：

- 维护成本；
- 迁移成本；
- 兼容性成本；
- 观测成本；
- debug 成本；
- 回滚成本；
- 机会成本。

这些成本不是反对决策的理由，而是让决策可管理的前提。

### Rollback 不是补充项

AI Infra 决策常常影响线上服务、训练任务或集群资源。

如果决策不可回滚，必须明确说明：

- 为什么不可回滚；
- 是否需要更高证据等级；
- 是否需要更小灰度范围；
- 是否需要停机窗口；
- 是否需要人工审批。

### 不把 Benchmark 写成绝对事实

Benchmark 只能证明被测条件下的结果。

应该写：

```text
在本次 workload、硬件和版本组合下，方案 A 的 p99 TTFT 优于 baseline。
```

不要写：

```text
方案 A 的延迟更低。
```

后者会误导读者和 AI。

## 常见误区

### 误区一：把调研报告当 ADR

调研报告可以没有结论。

ADR 必须有 decision 或明确 deferred/rejected。

如果还没有决策，就写 `proposed` 或 `deferred`，不要假装已经 accepted。

### 误区二：只记录最终方案

只记录最终方案会丢失最重要的信息：

- 当时有哪些替代方案；
- 为什么没有选择它们；
- 证据是否充分；
- 未来什么条件会改变判断。

### 误区三：没有适用边界

AI Infra 决策强依赖 workload。

没有适用边界的 ADR 很容易被错误复用到不同模型、不同硬件、不同流量和不同 SLO 上。

### 误区四：证据不可复现

如果 ADR 说“Benchmark 显示收益明显”，但没有 raw data、run manifest 和环境信息，这个证据不能复用。

至少要链接：

- benchmark report；
- raw data；
- run manifest；
- profiler trace；
- 环境配置。

### 误区五：没有复审条件

没有 revisit condition 的 ADR 会逐渐变成历史包袱。

推荐每篇高影响 ADR 都写：

- 时间复审；
- workload 复审；
- 版本复审；
- 事故复审；
- 成本复审。

### 误区六：让 AI 读到过期决策

如果旧 ADR 已被替代，但没有 `superseded_by`，AI 可能继续引用旧结论。

状态字段和替代链接是 AI 可读知识库里的关键安全措施。

## 质量门禁

写完 ADR 后，至少检查：

- `title` 和 H1 是否一致；
- `doc_type` 是否为 `adr`；
- `decision_status` 是否和正文一致；
- 是否有 decision question；
- 是否有 workload contract；
- 是否有 system scope；
- 是否列出 options；
- 是否包含 do nothing；
- 是否说明 evidence level；
- 是否链接 benchmark、paper、docs、source code 或 incident；
- 是否说明 rejected options；
- 是否说明 positive、negative 和 neutral consequences；
- 是否有 rollout；
- 是否有 rollback；
- 是否有 confirmation；
- 是否有 revisit condition；
- 如果 superseded，是否链接替代 ADR；
- 是否需要更新知识地图或章节入口；
- 是否更新 `scripts/generate_llms_files.py`；
- 是否运行 `python3 scripts/generate_llms_files.py`；
- 是否运行 `.venv/bin/mkdocs build --strict`；
- 是否运行 `git diff --check`。

## 和其他模板的关系

| 模板 | 什么时候使用 | ADR 如何引用 |
| --- | --- | --- |
| 知识点模板 | 解释概念、机制或组件 | 链接为背景资料 |
| Benchmark 报告模板 | 记录实验问题、环境、结果和分析 | 作为证据来源 |
| Failure Case | 记录事故、性能回归或复现失败 | 作为风险和复审触发 |
| Runbook | 记录操作步骤、诊断和回滚 | 作为 rollout/rollback 执行依据 |
| Paper Note | 记录论文机制和复现实验 | 作为候选方案或理论依据 |

ADR 不应该吞掉这些文档。

它应该把它们组织成一条清晰证据链。

## 参考资料

- [Documenting Architecture Decisions](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
- [Markdown Architectural Decision Records](https://github.com/adr/madr)
- [Architecture decision record examples and templates](https://github.com/architecture-decision-record/architecture-decision-record)
- [One Size Fits All? An Empirical Comparison of ADR Templates regarding Comprehension, Usability, and Ease of Adoption](https://arxiv.org/abs/2604.27333)
- [Context Matters: Evaluating Context Strategies for Automated ADR Generation Using LLMs](https://arxiv.org/abs/2604.03826)
