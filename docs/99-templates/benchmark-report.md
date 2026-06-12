---
title: 基准实验报告模板：从问题设计到可复现实验
domain: template
doc_type: template
status: reviewed
owner: maintainers
license: CC-BY-4.0
updated: 2026-06-12
---

# 基准实验报告模板：从问题设计到可复现实验

这个模板用于撰写 AI Infra 知识库中的 benchmark report。

Benchmark report 不是简单的“跑分记录”。

它应该回答：

> 在什么 workload、硬件、软件、配置和测量方法下，某个系统问题被怎样测量，结果是否支持结论，结论能否被别人复现，能否支撑后续 ADR、容量规划或优化决策。

AI Infra 里的 benchmark 很容易误导人。

常见原因包括：

- workload 不是目标 workload；
- baseline 不公平；
- warmup 和测量混在一起；
- 只看平均值，不看尾延迟；
- 只看吞吐，不看 SLO；
- 只看 GPU utilization，不看有效产出；
- 只给图表，不给 raw data；
- 只记录结果，不记录环境；
- 只写“更快”，不写适用边界；
- 单次实验被写成普遍结论。

高质量 benchmark report 的目标不是证明某个方案更好，而是把一个可验证的问题变成可复用的证据。

它应该能被三类对象使用：

- 人类读者：理解实验问题、方法、结果、边界和结论；
- 后续工程决策：作为 ADR、容量规划、SLO 或优化工作的证据；
- AI Agent：在检索时能识别 workload、metrics、baseline、evidence level、raw data 和 caveats。

## 使用方式

新增 benchmark report 时，建议按下面流程：

1. 先写 benchmark question。
2. 再写 hypothesis。
3. 固定 workload contract。
4. 固定 baseline。
5. 选择 primary metrics、guardrail metrics 和 debugging metrics。
6. 写清变量控制。
7. 记录 run manifest。
8. 保存 raw data、summary data 和 profiler trace。
9. 分析结果和误差。
10. 给出结论、适用范围和后续动作。

本模板适合：

- 推理引擎对比；
- Batching、KV Cache、Prefix Cache、量化、Speculative Decoding 等推理优化评估；
- 训练吞吐、MFU、扩展效率、通信重叠、checkpoint 开销评估；
- kernel、compiler、runtime 优化评估；
- GPU、NPU、网络、存储、集群拓扑评估；
- 容量规划和成本模型输入；
- 性能回归检测；
- trace replay 和真实流量压测；
- 能效和功耗评估。

本模板不适合：

- 解释一个概念；
- 做技术选型结论；
- 记录一次事故；
- 写操作手册；
- 写无法复现的临时观察；
- 写没有 baseline 的宣传式结果。

如果要解释机制，使用知识点模板。

如果要做决策，使用 ADR 模板。

如果要记录故障，使用 failure case 或 postmortem 结构。

## Benchmark 类型

不同 benchmark 类型回答的问题不同。

不要把它们混在一张表里直接比较。

| 类型 | 回答的问题 | 常见风险 |
| --- | --- | --- |
| Microbenchmark | 单个 kernel、算子或组件的上限是多少 | 可能和真实系统瓶颈无关 |
| Component benchmark | 某个模块在受控输入下表现如何 | 容易忽略上下游等待 |
| End-to-end benchmark | 用户请求或训练 step 的整体表现如何 | 难以定位瓶颈 |
| Capacity benchmark | 在 SLO 下能承载多少请求、token、step 或 job | 需要明确流量模型 |
| Scaling benchmark | 增加 GPU、节点或并行度后效率如何 | 容易被通信、负载不均或数据输入掩盖 |
| Regression benchmark | 新版本是否退化 | 需要稳定 baseline 和阈值 |
| Power benchmark | 单位有效产出的能耗是多少 | 需要稳定功耗采样和温度条件 |
| Trace replay | 真实流量分布下系统表现如何 | 需要处理脱敏、采样偏差和重放保真度 |

报告开头必须写清 benchmark 类型。

同一篇报告可以包含多个类型，但必须分段呈现，不要用一个结论覆盖所有类型。

## 结果可信度

建议给 benchmark report 标注证据等级。

| 等级 | 含义 |
| --- | --- |
| E0 | 手动临时观察，没有可复现记录 |
| E1 | 有命令和摘要结果，但缺少 raw data、manifest 或重复实验 |
| E2 | 有 workload、baseline、metrics、环境、raw data 和 run manifest |
| E3 | 有多次重复、sweep、profiler、误差分析和 guardrail metrics |
| E4 | 有生产灰度、真实 trace replay、SLO 和成本闭环验证 |

知识库中的 benchmark report 至少应达到 E2。

如果只有 E0 或 E1，应明确标注为 exploratory，不要作为 ADR 的主要证据。

## 最小可复制模板

下面是最小版本。

适合快速记录一次可复现实验。

````markdown
---
title: ""
domain: benchmark-capacity
doc_type: benchmark-report
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-06-12
benchmark_type: ""
evidence_level: E2
workload: []
system_layer: []
hardware: []
software: []
metrics: []
baseline: ""
variants: []
raw_data: ""
run_manifest: ""
profiler_trace: ""
related:
  - ""
---

# 标题

## Summary

用 3 到 5 句话说明：

- 本次 benchmark 要回答什么问题；
- 使用什么 workload 和环境；
- baseline 和 variants 是什么；
- 主要结果是什么；
- 结论适用于什么范围。

## Benchmark Question

一句话写清楚实验问题。

> 在什么 workload 和环境下，比较什么方案，观察什么指标？

## Hypothesis

实验前的假设：

-

可能推翻假设的证据：

-

## Workload

| 字段 | 内容 |
| --- | --- |
| model / operator |  |
| dataset / trace |  |
| input shape |  |
| output shape |  |
| batch / concurrency |  |
| precision |  |
| traffic pattern |  |
| duration |  |

## Environment

| 项目 | 配置 |
| --- | --- |
| hardware |  |
| software |  |
| runtime |  |
| driver |  |
| configuration |  |
| git commit |  |
| container image |  |

## Metrics

Primary metrics:

-

Guardrail metrics:

-

Debugging metrics:

-

## Experiment Design

Baseline:

Variants:

Controlled variables:

Changed variables:

Warmup:

Repetitions:

## Commands

```bash
# benchmark commands
```

## Results

| Run | Variant | Primary metric | Guardrail metrics | Notes |
| --- | --- | --- | --- | --- |
|  | baseline |  |  |  |
|  | variant-a |  |  |  |

## Analysis

- 结果是否支持 hypothesis？
- 主要瓶颈是什么？
- 是否有 profiler、trace 或日志证据？
- 误差和不确定性来自哪里？
- 结论不能推广到哪些场景？

## Conclusion

Supported:

-

Not supported:

-

Next steps:

-

## Artifacts

- raw data:
- run manifest:
- profiler trace:
- logs:
- dashboard:

## References

-
````

## 完整推荐模板

下面是完整版本。

适合能支撑 ADR、容量规划、性能回归治理或公开复现实验的重要 benchmark。

````markdown
---
title: ""
domain: benchmark-capacity
doc_type: benchmark-report
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-06-12

benchmark_id: ""
benchmark_type: ""
evidence_level: E2
question: ""
hypothesis: ""

workload:
  - ""
system_layer:
  - ""
hardware:
  - ""
software:
  - ""
metrics:
  primary: []
  guardrail: []
  debugging: []

baseline: ""
variants: []
raw_data: ""
summary_data: ""
run_manifest: ""
profiler_trace: ""
dashboard: ""
related:
  - ""
---

# 标题

## Executive Summary

用 5 到 8 句话说明：

1. 本次 benchmark 的问题。
2. workload、硬件、软件和测量窗口。
3. baseline 和 variants。
4. primary metrics 的结果。
5. guardrail metrics 是否被破坏。
6. 主要瓶颈和证据。
7. 结论适用范围。
8. 后续动作。

## Benchmark Question

写成可验证问题。

建议格式：

```text
在 <workload / traffic / hardware / software / config> 下，
<variant> 相比 <baseline> 是否能改善 <primary metric>，
并且不恶化 <guardrail metrics>？
```

例：

```text
在 8k input、512 output、Poisson arrival、p99 TTFT SLO 为 2s 的 LLM serving 场景下，
启用 prefix cache 是否能提高 goodput at SLO，
并且不显著增加 p99 TPOT 和 GPU memory peak？
```

## Hypothesis

写实验前假设。

| 假设 | 推导依据 | 可被什么证据推翻 |
| --- | --- | --- |
|  |  |  |

规则：

- hypothesis 不能在看完结果后倒写；
- 如果结果推翻 hypothesis，也要保留；
- 不要把 hypothesis 写成结论。

## Validity Scope

说明本报告适用范围。

| 范围 | 内容 |
| --- | --- |
| model / operator |  |
| workload distribution |  |
| input / output shape |  |
| hardware |  |
| software version |  |
| runtime config |  |
| cluster scale |  |
| traffic pattern |  |
| benchmark duration |  |

明确不覆盖：

-

## Workload Contract

Benchmark 首先是 workload contract。

如果 workload 不清楚，结果不可解释。

### Inference Workload

| 字段 | 内容 |
| --- | --- |
| model |  |
| tokenizer |  |
| input length distribution |  |
| output length distribution |  |
| arrival process |  |
| concurrency / QPS |  |
| request mix |  |
| streaming |  |
| sampling params |  |
| SLO |  |

### Training Workload

| 字段 | 内容 |
| --- | --- |
| model |  |
| dataset |  |
| global batch size |  |
| micro batch size |  |
| sequence length |  |
| precision |  |
| optimizer |  |
| parallelism strategy |  |
| checkpoint cadence |  |
| evaluation cadence |  |

### Kernel or Compiler Workload

| 字段 | 内容 |
| --- | --- |
| operator |  |
| tensor shape |  |
| dtype |  |
| layout |  |
| stride |  |
| sparsity |  |
| fusion pattern |  |
| compile mode |  |

## Environment

### Hardware

| 项目 | 内容 |
| --- | --- |
| accelerator |  |
| accelerator count |  |
| CPU |  |
| memory |  |
| storage |  |
| network |  |
| topology |  |
| power limit |  |
| thermal condition |  |

### Software

| 项目 | 内容 |
| --- | --- |
| OS |  |
| driver |  |
| CUDA / ROCm / vendor runtime |  |
| framework |  |
| serving runtime / training stack |  |
| compiler |  |
| communication library |  |
| container image |  |
| git commit |  |
| dependency lock |  |

### Runtime Configuration

| 配置 | 值 |
| --- | --- |
| batch size / max batch size |  |
| max tokens / max seq len |  |
| parallelism |  |
| cache policy |  |
| quantization |  |
| scheduler policy |  |
| environment variables |  |
| feature flags |  |

## Baseline and Variants

Baseline 必须是可解释的比较对象。

| Variant | Description | Changed variables | Expected effect |
| --- | --- | --- | --- |
| baseline |  |  |  |
| variant-a |  |  |  |
| variant-b |  |  |  |

规则：

- baseline 不等于“旧版本随便跑一次”；
- 每个 variant 只改变必要变量；
- 如果多个变量同时改变，要说明为什么无法拆开；
- 不公平 baseline 只能作为历史参考，不能支撑结论。

## Metrics

### Primary Metrics

直接回答 benchmark question。

| Metric | Definition | Unit | Aggregation |
| --- | --- | --- | --- |
|  |  |  |  |

### Guardrail Metrics

防止单点优化伤害其他目标。

| Metric | Threshold | Reason |
| --- | --- | --- |
|  |  |  |

### Debugging Metrics

用于定位瓶颈，不直接作为结论。

| Metric | Source | Used for |
| --- | --- | --- |
|  |  |  |

常见指标：

- latency：TTFT、TPOT、E2E latency、queue wait、p50/p95/p99；
- throughput：requests/s、input tokens/s、output tokens/s、samples/s、steps/s；
- goodput：满足 SLO 的有效 requests/s 或 tokens/s；
- resource：GPU utilization、SM occupancy、HBM bandwidth、memory peak、KV cache occupancy；
- training：step time、tokens/s、MFU、scaling efficiency、communication ratio；
- cost：cost per request、cost per output token、GPU hour per training token；
- reliability：error rate、timeout、OOM、retry、preemption；
- energy：power、energy per token、frequency throttling、thermal throttling。

## Experiment Design

### Variables

| 类型 | 变量 |
| --- | --- |
| independent variables |  |
| dependent variables |  |
| controlled variables |  |
| uncontrolled variables |  |

### Warmup

写清：

- warmup 时长；
- warmup 请求数或 step 数；
- 是否包含 compilation；
- 是否包含 engine build；
- 是否包含 cache population；
- warmup 数据是否从统计中排除。

### Repetitions

写清：

- 重复次数；
- 每次持续时间；
- 是否重启进程；
- 是否清理 cache；
- 是否随机化顺序；
- 是否跨节点或跨时间重复。

### Sweep Plan

如果做 sweep，写清变量范围。

| Variable | Values |
| --- | --- |
| concurrency |  |
| input length |  |
| output length |  |
| batch size |  |
| GPU count |  |
| context length |  |
| precision |  |

### Stop Conditions

写清何时停止实验。

- error rate 超过阈值；
- OOM；
- p99 latency 超过 SLO；
- GPU thermal throttling；
- benchmark duration 达到要求；
- result variance 超过阈值，需要增加重复次数。

## Commands

记录完整命令。

```bash
# Environment

# Start service / runtime

# Run benchmark

# Collect profiler trace

# Export raw data
```

不要只写：

```text
按默认参数跑 benchmark。
```

默认参数会随版本变化。

## Run Manifest

run manifest 是复现实验的最小事实包。

```yaml
run_manifest:
  benchmark_id: ""
  run_id: ""
  timestamp: ""
  operator: ""
  git_commit: ""
  container_image: ""
  hardware:
    accelerator: ""
    count: ""
    topology: ""
  software:
    driver: ""
    cuda_or_rocm: ""
    framework: ""
    runtime: ""
  config:
    command: ""
    env: {}
    flags: {}
  workload:
    dataset_or_trace: ""
    request_count: ""
    duration_sec: ""
  artifacts:
    raw_data: ""
    logs: ""
    profiler_trace: ""
```

没有 run manifest 的结果很难被 AI 或团队复用。

## Raw Data and Summary Data

### Raw Data

raw data 应保存每次运行、每个请求、每个 step 或每个采样窗口的原始记录。

最小字段：

```csv
run_id,variant,timestamp,request_id,input_tokens,output_tokens,ttft_ms,tpot_ms,e2e_ms,error
```

训练场景可使用：

```csv
run_id,variant,step,tokens,step_time_ms,loss,grad_norm,gpu_memory_gb,mfu,comm_time_ms
```

### Summary Data

summary data 应从 raw data 生成。

不要只保存截图。

```yaml
summary:
  variant: ""
  samples: 0
  primary:
    metric: ""
    mean: null
    p50: null
    p95: null
    p99: null
  guardrail:
    error_rate: null
    memory_peak_gb: null
  confidence:
    method: ""
    interval: ""
```

## Results

### Main Results

| Variant | Primary metric | Guardrail metrics | Result | Notes |
| --- | --- | --- | --- | --- |
| baseline |  |  |  |  |
| variant-a |  |  |  |  |

### Distribution

如果指标有分布，不要只给平均值。

| Variant | p50 | p95 | p99 | max | samples |
| --- | --- | --- | --- | --- | --- |
| baseline |  |  |  |  |  |
| variant-a |  |  |  |  |  |

### Sweep Results

| Variable | Value | Baseline | Variant | Delta | Guardrail |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |

### Failure Results

| Variant | Failure | Count | Condition | Notes |
| --- | --- | --- | --- | --- |
|  | OOM / timeout / error |  |  |  |

## Statistical Treatment

写清：

- 样本数量；
- 聚合方法；
- 是否去掉 outlier；
- outlier 标准；
- 方差或置信区间；
- 多次 run 的合并方法；
- 是否做显著性判断；
- 结果是否受随机种子影响。

如果无法做统计判断，要写：

```text
本报告仅描述观察结果，不声称差异具有统计显著性。
```

## Profiler and Bottleneck Analysis

Benchmark report 应尽量说明“为什么是这个结果”。

| Evidence | Observation | Interpretation |
| --- | --- | --- |
| profiler trace |  |  |
| GPU metrics |  |  |
| network metrics |  |  |
| storage metrics |  |  |
| logs |  |  |

常见瓶颈：

- compute bound；
- HBM bandwidth bound；
- kernel launch overhead；
- synchronization；
- NCCL / network；
- KV Cache capacity；
- memory fragmentation；
- tokenizer / data pipeline；
- storage read；
- scheduler queueing；
- CPU overhead；
- compilation overhead。

## Correctness and Quality Guardrails

性能结果必须和正确性或质量边界一起解释。

| Guardrail | Result | Pass / Fail |
| --- | --- | --- |
| accuracy / loss / perplexity |  |  |
| output quality |  |  |
| numerical error |  |  |
| deterministic behavior |  |  |
| error rate |  |  |

例如：

- 量化推理要记录质量或数值误差；
- 训练优化要记录 loss 曲线和稳定性；
- speculative decoding 要记录 acceptance rate 和输出一致性；
- compiler 优化要记录 correctness test；
- kernel 优化要记录误差阈值。

## Cost and Energy

如果 benchmark 用于工程决策，应记录成本。

| Metric | Value | Notes |
| --- | --- | --- |
| cost per request |  |  |
| cost per output token |  |  |
| GPU hour per training token |  |  |
| energy per token |  |  |
| average power |  |  |
| peak power |  |  |

不要把吞吐提升自动等同于成本下降。

如果需要更多副本、更大显存或更复杂运维，成本可能上升。

## Interpretation

按下面顺序解释结果：

1. primary metrics 是否改善；
2. guardrail metrics 是否保持在阈值内；
3. profiler 是否解释了改善原因；
4. 结果在哪些 sweep 点成立；
5. 哪些场景没有覆盖；
6. 结论能否支撑 ADR 或容量模型。

## Conclusion

### Supported Conclusions

-

### Unsupported Conclusions

-

### Caveats

-

### Decision Impact

本报告可以支持：

-

本报告不能支持：

-

### Next Steps

- [ ] 补充 sweep：
- [ ] 补充 profiler：
- [ ] 更新 ADR：
- [ ] 更新容量模型：
- [ ] 加入回归 benchmark：
- [ ] 更新 runbook：

## Artifacts

| Artifact | Path / URL | Required |
| --- | --- | --- |
| raw data |  | yes |
| summary data |  | yes |
| run manifest |  | yes |
| benchmark script |  | yes |
| logs |  | recommended |
| profiler trace |  | recommended |
| dashboard |  | optional |
| report notebook |  | optional |

## AI-readable Benchmark Card

重要 benchmark report 建议包含结构化卡片。

```yaml
benchmark_card:
  benchmark_id: ""
  title: ""
  benchmark_type: ""
  question: ""
  hypothesis: ""
  evidence_level: "E2"
  workload:
    type: ""
    model_or_operator: ""
    shape: ""
    traffic: ""
  environment:
    hardware: []
    software: []
    config: {}
  baseline: ""
  variants: []
  metrics:
    primary: []
    guardrail: []
    debugging: []
  results:
    summary: ""
    primary_delta: ""
    guardrail_status: ""
  artifacts:
    raw_data: ""
    run_manifest: ""
    profiler_trace: ""
  caveats: []
  supported_decisions: []
  related_docs: []
```

AI 检索时，这个 card 应该能回答：

- 这个 benchmark 测了什么；
- 是否有可复现证据；
- 结果支持什么；
- 结果不支持什么；
- raw data 和 manifest 在哪里；
- 能否作为 ADR 证据。

## Checklist

### Question

- [ ] benchmark question 是否明确？
- [ ] hypothesis 是否在结果前定义？
- [ ] benchmark 类型是否明确？
- [ ] 适用范围是否明确？
- [ ] 不覆盖范围是否明确？

### Workload

- [ ] 模型、算子或系统对象是否明确？
- [ ] input/output shape 是否明确？
- [ ] traffic pattern 或 batch/step 设置是否明确？
- [ ] 数据集、trace 或合成数据生成方式是否明确？
- [ ] SLO 或目标指标是否明确？

### Environment

- [ ] 硬件规格是否完整？
- [ ] 软件版本是否完整？
- [ ] driver、CUDA/ROCm、framework、runtime 是否完整？
- [ ] git commit、镜像和依赖是否可追溯？
- [ ] 关键配置和环境变量是否记录？

### Experiment

- [ ] baseline 是否公平？
- [ ] variants 是否只改变必要变量？
- [ ] warmup 是否和正式测量分离？
- [ ] repetitions 是否足够？
- [ ] sweep 范围是否记录？
- [ ] stop conditions 是否明确？

### Data

- [ ] raw data 是否保存？
- [ ] summary data 是否由 raw data 生成？
- [ ] run manifest 是否保存？
- [ ] profiler trace 是否保存或说明缺失原因？
- [ ] logs 是否保存？

### Analysis

- [ ] primary metrics 是否回答问题？
- [ ] guardrail metrics 是否检查？
- [ ] 是否报告分布而不只是平均值？
- [ ] 是否说明误差、方差或置信度？
- [ ] 是否说明结果不能推广到哪些场景？
- [ ] 是否区分观察、解释和结论？

### Maintenance

- [ ] 是否更新相关知识点？
- [ ] 是否更新 ADR？
- [ ] 是否更新容量模型？
- [ ] 是否加入回归 benchmark？
- [ ] 是否更新 `scripts/generate_llms_files.py`？
- [ ] 是否运行生成和构建命令？

## References

-
````

## 字段说明

### benchmark_id

benchmark_id 用于追踪一次实验。

推荐格式：

```yaml
benchmark_id: "bench-2026-06-12-vllm-prefix-cache-001"
```

它应出现在：

- 报告；
- raw data；
- run manifest；
- profiler trace；
- dashboard；
- ADR 引用。

### benchmark_type

建议使用固定枚举。

```yaml
benchmark_type: capacity
```

可选值：

- microbenchmark；
- component；
- end-to-end；
- capacity；
- scaling；
- regression；
- power；
- trace-replay；
- exploratory。

### evidence_level

表示证据强度。

```yaml
evidence_level: E3
```

如果结果要支撑 ADR，建议至少 E2。

如果结果要影响生产默认路径，建议 E3 或 E4。

### workload

workload 是 benchmark 的核心。

同一个优化在不同 workload 下可能结论相反。

不要只写：

```yaml
workload:
  - llm
```

建议写：

```yaml
workload:
  - llm-serving
  - long-context
  - poisson-arrival
  - input-8k-output-512
```

### metrics

metrics 应分层。

```yaml
metrics:
  primary:
    - goodput_at_slo
  guardrail:
    - ttft_p99
    - error_rate
    - gpu_memory_peak
  debugging:
    - queue_wait_ms
    - kv_cache_occupancy
```

不要把所有指标混成一组。

### raw_data

raw_data 是可复现分析的基础。

```yaml
raw_data: "artifacts/bench-2026-06-12/raw.csv"
```

如果 raw data 太大，应提供：

- 存储路径；
- schema；
- sample；
- checksum；
- retention policy。

### run_manifest

run_manifest 是复现实验环境和命令的基础。

```yaml
run_manifest: "artifacts/bench-2026-06-12/manifest.yaml"
```

没有 manifest 的 benchmark 不应作为强证据。

### profiler_trace

profiler_trace 用于解释性能结果。

```yaml
profiler_trace: "artifacts/bench-2026-06-12/nsys.qdrep"
```

如果没有 profiler trace，要说明原因。

例如：

- benchmark 是容量压测，profiler 开销会改变流量；
- 当前结论只依赖黑盒指标；
- profiler 将在后续实验补充。

## 写作规则

### 先写问题，再看结果

Benchmark report 的问题必须先于结果。

不好：

```text
方案 A 比方案 B 快 20%，所以我们写一个 benchmark 证明它。
```

好：

```text
我们要判断方案 A 在 workload W 下是否能改善 goodput at SLO，同时不恶化 p99 TTFT 和 error rate。
```

### 一个 benchmark 只回答一个主问题

如果同时比较推理引擎、量化、batching、硬件和网络，结果通常无法解释。

应拆成：

- engine benchmark；
- quantization benchmark；
- batching benchmark；
- hardware benchmark；
- network benchmark。

### Baseline 要公平

公平 baseline 至少要求：

- workload 相同；
- 硬件相同；
- 软件版本可追溯；
- 关键配置对等；
- warmup 和测量方法相同；
- 数据处理路径一致。

如果 baseline 不公平，必须标注为 historical baseline 或 operational baseline。

### 不要只报告峰值

峰值吞吐通常不能代表可用能力。

AI 系统更关心：

- SLO 下的 goodput；
- p95/p99 尾延迟；
- 长时间稳定性；
- error rate；
- memory headroom；
- recovery behavior；
- cost per useful unit。

### 图表不能替代 raw data

图表用于阅读。

raw data 用于复查。

summary data 用于自动化和 AI 检索。

三者都应该保留。

### 区分观察、解释和结论

观察：

```text
variant-a 的 p99 TTFT 在本次 workload 下低于 baseline。
```

解释：

```text
profiler 显示 queue wait 和 KV cache allocation 时间下降，可能解释了 TTFT 改善。
```

结论：

```text
在该 workload 和环境下，variant-a 可以作为下一阶段灰度候选，但尚不能推广到 32k context。
```

### 报告失败结果

失败结果很有价值。

应记录：

- OOM；
- timeout；
- crash；
- accuracy failure；
- numerical error；
- thermal throttling；
- network congestion；
- data loader stall；
- checkpoint failure；
- profiler overhead。

这些结果常常比成功样本更能指导系统设计。

### 不把 synthetic workload 当真实流量

synthetic workload 有价值，但不能自动代表真实流量。

如果使用合成数据，要说明：

- 分布假设；
- 采样方法；
- 与真实 trace 的差异；
- 哪些结论只能用于上限估计；
- 是否需要 trace replay 验证。

## 常见误区

### 误区一：问题不明确

只写“测试一下性能”不是 benchmark question。

应写清：

- 比较对象；
- workload；
- 指标；
- 成功标准；
- guardrail。

### 误区二：指标太多但没有主指标

指标多不等于严谨。

必须有 primary metric。

否则读者不知道实验到底回答了什么。

### 误区三：只看平均延迟

平均延迟可能掩盖尾部问题。

在线推理服务通常至少要看 p95/p99、queue wait、error rate 和 goodput at SLO。

### 误区四：没有版本和配置

AI Infra 结果高度依赖版本和配置。

缺少以下信息时，结果几乎不可复现：

- git commit；
- container image；
- driver；
- CUDA/ROCm；
- runtime version；
- kernel/compiler flags；
- environment variables；
- parallelism config；
- scheduler config。

### 误区五：只保存最终表格

最终表格不能回答：

- outlier 来自哪里；
- 是否有随机波动；
- 某个请求为什么慢；
- 某个 step 为什么卡；
- 某个节点是否异常。

必须保存 raw data 和日志。

### 误区六：没有质量或正确性 guardrail

某些优化可以提升速度，但改变数值、质量或稳定性。

例如：

- 量化；
- speculative decoding；
- compiler fusion；
- 自定义 kernel；
- 混合精度训练；
- data pipeline 改写。

这些场景必须检查 guardrail。

### 误区七：把单点结果推广成普遍结论

如果只测了 8k context，不应声称对所有 long-context 有效。

如果只测了某一代 GPU，不应声称对所有硬件有效。

如果只测了合成流量，不应声称对生产流量有效。

## 质量门禁

写完 benchmark report 后，至少检查：

- `title` 和 H1 是否一致；
- `doc_type` 是否为 `benchmark-report`；
- 是否有 benchmark_id；
- 是否有 benchmark_type；
- 是否有 benchmark question；
- 是否有 hypothesis；
- 是否有 workload contract；
- 是否有 baseline；
- 是否有 variants；
- 是否有 primary metrics；
- 是否有 guardrail metrics；
- 是否说明 warmup；
- 是否说明 repetitions；
- 是否有 raw data；
- 是否有 run manifest；
- 是否有 summary data；
- 是否有 profiler trace 或缺失说明；
- 是否报告误差或方差；
- 是否说明适用范围；
- 是否说明不支持的结论；
- 是否链接相关知识点、ADR 或容量模型；
- 是否更新 `scripts/generate_llms_files.py`；
- 是否运行 `python3 scripts/generate_llms_files.py`；
- 是否运行 `.venv/bin/mkdocs build --strict`；
- 是否运行 `git diff --check`。

## 和其他模板的关系

| 模板 | 什么时候使用 | Benchmark report 如何引用 |
| --- | --- | --- |
| 知识点模板 | 解释机制、指标或系统组件 | 链接为背景资料 |
| ADR 模板 | 做技术选择 | Benchmark report 作为证据 |
| Failure Case | 记录失败、事故或性能回归 | Benchmark 复现失败现象 |
| Runbook | 记录压测、回滚或排障操作 | Benchmark 命令可转为操作步骤 |
| Paper Note | 记录论文结果和复现实验 | Benchmark 验证论文机制在本地环境是否成立 |

Benchmark report 不应该替代 ADR。

它提供证据，但不自动产生决策。

## 参考资料

- [MLPerf Inference Benchmark Suite](https://docs.mlcommons.org/inference/index_gh/)
- [MLPerf Training](https://mlcommons.org/benchmarks/training/)
- [TensorRT-LLM Benchmarking](https://nvidia.github.io/TensorRT-LLM/performance/perf-benchmarking.html)
- [PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
