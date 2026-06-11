---
title: 推理容量建模：QPS、并发、TTFT、TPOT 与 GPU 副本数
domain: benchmark-capacity
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-06-12
---

# 推理容量建模：QPS、并发、TTFT、TPOT 与 GPU 副本数

推理系统容量规划最常见的错误，是用“峰值吞吐”推算生产容量。

例如：

```text
单副本 benchmark: 3000 output tokens/s
目标业务: 30000 output tokens/s
结论: 需要 10 个副本
```

这个结论通常不可靠，因为它忽略了：

- 输入长度分布。
- 输出长度分布。
- TTFT 和 TPOT 的 SLA。
- 请求到达方式。
- 并发上限。
- KV Cache 容量。
- prefix cache 命中率。
- batching 策略。
- 冷启动和模型加载。
- tail latency。
- 错误率和重试。
- 保留 headroom。

推理容量建模的核心不是“这张 GPU 最多能吐多少 token”，而是：

> 在指定模型、请求分布、服务策略和 SLA 下，单个副本能稳定承载多少有效请求，然后需要多少副本满足目标流量和冗余要求。

## 一张总图

```mermaid
flowchart TB
    Traffic["Traffic Model<br/>QPS / burst / concurrency / input-output length"]
    SLA["SLA<br/>TTFT / TPOT / E2E / error rate"]
    Engine["Serving Engine<br/>batching / scheduler / KV cache / cache policy"]
    Replica["One Replica Benchmark<br/>qps at SLA / tokens/s / memory / tail"]
    Model["Capacity Model<br/>replicas / headroom / autoscaling / cost"]
    Deploy["Deployment Plan<br/>GPU type / TP size / node pool / routing"]
    Observe["Production Feedback<br/>actual QPS / p99 / cache hit / saturation"]

    Traffic --> SLA
    SLA --> Engine
    Engine --> Replica
    Replica --> Model
    Model --> Deploy
    Deploy --> Observe
    Observe --> Traffic
    Observe --> Model
```

这张图表达一个闭环：

- 先定义业务流量和 SLA。
- 再测单副本在该约束下的能力。
- 用单副本能力推导副本数、headroom 和成本。
- 部署后用生产反馈修正模型。

容量模型不是一次性表格，而是持续校准的工程模型。

## 先定义服务目标

推理容量建模必须先回答：

```text
我们要服务什么模型？
请求长什么样？
SLA 是什么？
可接受的成本和错误率是多少？
```

一个可用的服务目标例子：

```text
model: 70B dense model
precision: FP8 / W8A8 / FP16
engine: vLLM / TensorRT-LLM / SGLang
input length p50/p95: 1k / 8k tokens
output length p50/p95: 256 / 1024 tokens
target QPS: 120
TTFT p95: < 800 ms
TPOT p95: < 50 ms/token
error rate: < 0.1%
availability: tolerate 1 node failure
```

如果只有“目标 QPS 120”，容量模型无法成立。

## Capacity Contract

容量建模开始前，建议写一个 Capacity Contract。它和 benchmark contract 类似，但目标更偏部署和容量决策。

示例：

```yaml
capacity_plan: chat-70b-prod-2026-06

service:
  model: llama-like-70b
  engine: vLLM
  parallelism:
    tensor_parallel: 4
    replicas_per_node: 2
  node_profile: h100-sxm-infer

traffic:
  target_qps_steady: 80
  target_qps_peak: 160
  burst:
    qps: 240
    duration: 5m
  arrival_process: production_trace_replay
  input_tokens:
    p50: 1024
    p95: 8192
    p99: 16384
  output_tokens:
    p50: 256
    p95: 1024
    p99: 2048

slo:
  ttft_p95: 800ms
  tpot_p95: 50ms
  e2e_p95: 8s
  error_rate: 0.1%

capacity_policy:
  headroom: 30%
  tolerate_node_failure: 1
  rolling_update_unavailable: 20%
  min_warm_replicas: 4
```

这个 contract 让容量讨论从“要多少卡”变成一组可验证问题：

- 单副本在这个 workload 下的 goodput 是多少。
- 需要保留多少 headroom。
- 失去一个节点后是否仍满足 SLA。
- 滚动升级时允许多少副本不可用。
- autoscaling 是否来得及应对 burst。

容量模型的输出也应该回写 contract：

```text
required_replicas
required_nodes
expected_cost
expected_headroom
known_risks
validation_result
```

## 核心指标

### QPS

QPS 是每秒请求数。

但推理请求不是等价单位。一个 100 token 输入、20 token 输出的请求，和一个 32k 输入、4k 输出的请求，对系统压力完全不同。

所以 QPS 必须绑定请求分布：

```text
QPS at request mix X
QPS at input/output length distribution Y
QPS at SLA Z
```

### Concurrency

Concurrency 是系统中同时存在的请求数，包括：

- 排队中的请求。
- prefill 中的请求。
- decode 中的请求。
- 等待输出或后处理的请求。

排队论里的 Little's Law 可以作为直觉：

```text
concurrency ~= arrival_rate * latency
```

例如：

```text
QPS = 100 req/s
E2E latency = 2 s
concurrency ~= 200 requests
```

这说明如果目标 QPS 高、请求又长，系统必须能容纳足够并发。并发上限往往受 KV Cache 和调度策略约束。

### TTFT

TTFT 是 time to first token，从请求进入服务到第一个 token 返回。

它通常包括：

```text
request queue
  + tokenization / preprocessing
  + routing
  + prefill wait
  + prefill compute
  + first decode step
  + network / streaming overhead
```

TTFT 主要受输入长度、prefill 计算、batching 等待、queueing 和缓存命中影响。

### TPOT

TPOT 是 time per output token，常用于衡量 decode 阶段持续生成速度。

它通常和：

- decode batch size。
- KV Cache 读写。
- memory bandwidth。
- sampling。
- tensor parallel 通信。
- speculative decoding。
- CUDA graph / kernel fusion。

有关。

### E2E Latency

端到端延迟可以粗略拆成：

```text
E2E ~= queue_time + TTFT_compute + output_tokens * TPOT + postprocess_time
```

更细一点：

```text
E2E
  ~= queue
   + tokenize
   + prefill_wait
   + prefill_compute
   + decode_wait
   + decode_steps
   + detokenize
   + network
```

容量规划时要明确 SLA 是看 TTFT、TPOT、E2E，还是三者都看。

### Goodput

Goodput 是满足 SLA 的有效吞吐。

例如：

```text
系统总吞吐: 150 QPS
满足 TTFT/TPOT SLA 的请求: 120 QPS
错误或超时请求: 5 QPS
goodput = 120 QPS
```

容量规划应该使用 goodput，而不是总吞吐。

### Load、Throughput 与 Goodput

容量建模要区分三个量：

| 名称 | 含义 |
| --- | --- |
| Offered load | 外部打进来的请求量，例如目标 QPS |
| Throughput | 系统实际处理的总请求或 token |
| Goodput | 成功且满足 SLA 的有效请求或 token |

当 offered load 超过系统能力时，throughput 可能还在增长，但 goodput 会下降：

```text
load rises
  -> queue grows
  -> p95/p99 latency breaks SLA
  -> timeout/retry increases
  -> goodput stops increasing or drops
```

因此找容量边界时，不要只看曲线最高点，而要找满足 SLA 的最后一个稳定点。

## 单副本能力不是一个数

单副本能力通常是一条曲线，而不是一个标量。

例如：

| QPS | TTFT p95 | TPOT p95 | Error Rate | 结论 |
| --- | --- | --- | --- | --- |
| 20 | 300 ms | 30 ms | 0% | 宽松 |
| 40 | 500 ms | 38 ms | 0% | 可用 |
| 60 | 900 ms | 55 ms | 0.1% | 可能超 SLA |
| 80 | 2000 ms | 90 ms | 1% | 不可用 |

如果 SLA 是 TTFT p95 < 800 ms、TPOT p95 < 50 ms，那么单副本容量不是 80 QPS，而是 40 QPS 左右，或者介于 40 和 60 之间，需要更细测试。

所以容量公式应该写成：

```text
replicas = ceil(target_qps / per_replica_goodput_at_sla)
```

而不是：

```text
replicas = ceil(target_peak_tokens_per_sec / peak_tokens_per_sec)
```

### Goodput 曲线与拐点

单副本压测通常会得到类似曲线：

| Offered QPS | Throughput | Goodput | TTFT p95 | TPOT p95 | KV Occupancy | 结论 |
| --- | --- | --- | --- | --- | --- | --- |
| 20 | 20 | 20 | 300 ms | 30 ms | 35% | 低负载 |
| 40 | 40 | 40 | 500 ms | 38 ms | 55% | 健康 |
| 60 | 59 | 52 | 900 ms | 55 ms | 75% | 接近边界 |
| 80 | 66 | 20 | 2400 ms | 90 ms | 95% | 过载 |

容量规划应选择 `40` 或在 `40-60` 之间进一步细测，而不是选择 throughput 最高的 `66`。

要关注两个拐点：

- latency knee：QPS 增加一点，TTFT/TPOT 突然变差。
- error knee：timeout、OOM、拒绝请求或重试开始明显增加。

生产容量应该远离拐点，而不是贴着拐点运行。贴着拐点运行时，输入长度、cache 命中率或邻居负载稍微变化，就会把系统推入过载区。

## 推理容量的三个约束

一个推理副本通常同时受三个约束：

```text
compute capacity
memory capacity
latency capacity
```

### Compute Capacity

计算能力决定 prefill 和 decode 的算力上限。

prefill 通常更像大矩阵计算，输入越长，prefill 成本越高。

decode 每步只生成一个 token，但要访问所有层的 KV Cache，常常更受内存带宽和调度影响。

粗略看：

```text
prefill_cost ~ input_tokens
decode_cost ~ output_tokens * active_sequences
```

真正的成本还取决于模型结构、attention 实现、并行方式和缓存命中。

### Memory Capacity

显存限制包括：

- 模型权重。
- KV Cache。
- runtime workspace。
- activation 临时空间。
- CUDA graph / engine cache。
- fragmentation。

KV Cache 通常是并发的主要限制。

粗略估算：

```text
kv_bytes_per_token
  ~= 2 * num_layers * num_kv_heads * head_dim * bytes_per_element
```

其中 2 代表 K 和 V。如果用了 tensor parallel，单 GPU 的 KV 负载还要考虑切分方式。不同架构、GQA/MQA、engine 实现和 cache block 管理都会改变实际值。

可容纳 token 数可以粗略写成：

```text
max_cached_tokens
  ~= available_kv_memory / kv_bytes_per_token
```

并发上限取决于：

```text
sum(input_tokens + generated_tokens for active_requests)
```

这就是为什么长上下文会迅速压低并发容量。

### Latency Capacity

即使算力和显存还没满，SLA 也可能先被打爆。

例如增加 batch 可以提高吞吐，但会带来：

- 请求排队更久。
- prefill 等待更久。
- decode step 被更大 batch 拖慢。
- p99 latency 上升。

因此服务容量常常不是硬件满载点，而是 SLA 刚好可接受前的点。

### 约束优先级会随 workload 改变

同一个模型在不同请求分布下，瓶颈可能不同：

| Workload | 常见主瓶颈 |
| --- | --- |
| 短输入、短输出、高 QPS | scheduler、batching、CPU/tokenizer、decode |
| 长输入、短输出 | prefill compute、TTFT、prompt cache |
| 短输入、长输出 | decode、KV cache 带宽、TPOT |
| 长输入、长输出 | KV capacity、prefill+decode 双瓶颈 |
| 多轮对话 | KV/cache 生命周期、路由粘性、prefix cache |
| RAG 请求 | 输入长度尾部、检索延迟、上下文拼接 |
| Agent 请求 | 多次模型调用、工具延迟、burst fan-out |

所以容量模型不能只写“70B 模型每卡多少 QPS”。必须绑定具体 workload bucket。

## Workload 模型

推理容量模型必须从真实 workload 出发。

### 输入长度分布

不要只记录平均输入长度。

需要至少记录：

- p50。
- p90。
- p95。
- p99。
- 最大值。
- 是否有多模态输入。
- 是否有 RAG 拼接。
- 是否有系统 prompt 和历史对话。

长输入主要影响 TTFT 和 KV Cache。

### 输出长度分布

输出长度决定 decode 时间和持续占用。

需要记录：

- p50/p95/p99 输出 token。
- max tokens 设置。
- stop 条件。
- 用户取消请求比例。
- streaming 是否开启。

如果 benchmark 固定输出 128 token，而生产 p95 是 2048 token，容量会严重高估。

### 到达过程

请求不是均匀到达。

常见模型：

- 固定并发。
- 固定 QPS。
- Poisson arrival。
- trace replay。
- burst traffic。

固定并发容易测吞吐上限；固定 QPS 更接近 SLA 验证；trace replay 最接近真实业务。

### Cache 命中

需要建模：

- prefix cache 命中率。
- system prompt 复用。
- 多轮对话历史。
- RAG context 复用。
- KV cache reuse。

cache 命中会显著改变 TTFT 和显存压力。不能用高 cache 命中 benchmark 去承诺低 cache 命中业务。

## 请求分桶建模

真实流量通常不能用一个平均请求代表。建议把请求分成 bucket。

示例：

| Bucket | 占比 | 输入 | 输出 | 特征 |
| --- | --- | --- | --- | --- |
| short chat | 50% | <= 1k | <= 256 | 常规对话 |
| medium chat | 30% | 1k-8k | 256-1k | 多轮上下文 |
| long context | 10% | 8k-32k | <= 1k | RAG/长文档 |
| long generation | 8% | <= 4k | 1k-4k | 写作/代码生成 |
| agent burst | 2% | variable | variable | 多次工具调用 |

然后分别测或估算每个 bucket 的：

```text
TTFT
TPOT
E2E
KV token occupancy
goodput
error rate
```

整体容量不能只按占比平均，因为长尾请求会制造非线性影响：

- 长输入会阻塞 prefill。
- 长输出会长期占用 decode slot。
- 长上下文会占用大量 KV cache。
- Agent fan-out 会把一个用户请求放大成多个模型请求。

容量模型至少要回答：

```text
如果 long context 占比从 10% 变成 20%，副本数需要增加多少？
如果 output p95 从 1024 变成 2048，TPOT 和 KV occupancy 是否还能满足 SLA？
```

## 容量建模步骤

### 1. 定义目标流量

示例：

```text
target_qps_p50 = 80
target_qps_peak = 160
burst_duration = 5 min
request_mix = production_trace_2026_06
```

### 2. 定义 SLA

示例：

```text
TTFT p95 < 800 ms
TPOT p95 < 50 ms/token
E2E p95 < 8 s
error_rate < 0.1%
```

SLA 要和业务体验绑定。内部批处理可以放宽 TTFT；在线聊天通常不能。

### 3. 测单副本曲线

不要只测一个点。至少扫：

- QPS。
- concurrency。
- input/output length。
- cache hit rate。
- batch 参数。
- max context。

得到：

```text
per_replica_goodput_at_sla
max_concurrency_at_sla
kv_cache_saturation_point
decode_saturation_point
```

### 4. 计算基础副本数

```text
base_replicas = ceil(target_peak_qps / per_replica_goodput_at_sla)
```

### 5. 加 headroom

需要考虑：

- 流量突发。
- 单节点故障。
- 滚动升级。
- cache miss。
- 长请求比例上升。
- 模型热切换。
- 监控误差。

示例：

```text
replicas = ceil(base_replicas * (1 + headroom_ratio))
```

如果要求 N+1：

```text
replicas = base_replicas + replicas_lost_in_one_failure_domain
```

### 更完整的副本数公式

生产副本数通常要同时考虑多项约束：

```text
base = ceil(target_peak_qps / per_replica_goodput_at_sla)

with_headroom = ceil(base / (1 - headroom_ratio))

with_rolling_update = ceil(with_headroom / (1 - max_unavailable_ratio))

with_failure_domain = with_rolling_update + replicas_lost_in_one_failure_domain

required_replicas = max(
  with_failure_domain,
  min_warm_replicas,
  replicas_required_by_kv_capacity,
  replicas_required_by_burst_absorption
)
```

这里有两个容易写错的地方：

- `headroom_ratio` 如果表示保留 30% 空余，更合理的写法是除以 `1 - 0.3`，而不是乘以 `1.3`。
- rolling update 会让一部分副本暂时不可用，必须和故障域分开考虑。

示例：

```text
target_peak_qps = 160
per_replica_goodput_at_sla = 20
headroom = 30%
max_unavailable_during_rollout = 20%
one_node_failure_loses = 2 replicas

base = ceil(160 / 20) = 8
with_headroom = ceil(8 / 0.7) = 12
with_rolling_update = ceil(12 / 0.8) = 15
with_failure_domain = 15 + 2 = 17
```

这个结果比简单 `160 / 20 = 8` 大很多，但它更接近生产部署现实。

### 6. 验证集群级容量

副本数算出来后，还要验证：

- 调度是否能放下这些副本。
- GPU 型号和显存是否满足。
- 节点池是否有足够 headroom。
- 模型权重加载是否会打爆存储。
- 路由层是否能均衡。
- cache 命中率是否符合假设。
- autoscaler 是否足够快。

容量模型只算 GPU 副本数是不够的。

## 一个简化例子

目标：

```text
peak_qps = 120
SLA: TTFT p95 < 1s, TPOT p95 < 60ms
single replica goodput at SLA = 18 QPS
headroom = 30%
one node failure loses 2 replicas
```

计算：

```text
base_replicas = ceil(120 / 18) = 7
headroom_replicas = ceil(7 * 1.3) = 10
failure_aware_replicas = 10 + 2 = 12
```

这说明生产部署至少需要 12 个副本，而不是按峰值 tokens/s 推出的 7 个。

## TTFT 预算

TTFT 可以拆预算：

```text
TTFT_budget = queue + tokenize + prefill_wait + prefill_compute + first_decode + network
```

例如：

| 阶段 | 预算 |
| --- | --- |
| queue | 100 ms |
| tokenize | 50 ms |
| prefill wait | 150 ms |
| prefill compute | 500 ms |
| first decode | 50 ms |
| network/stream | 50 ms |
| total | 900 ms |

如果 TTFT p95 超标，先看哪一段超：

- queue 高：容量不足或 batching 等待过长。
- tokenize 高：CPU 或 tokenizer 服务瓶颈。
- prefill compute 高：输入过长、batch 太大或算力不足。
- first decode 高：调度或 kernel 问题。

## TPOT 预算

TPOT 可以理解为 decode 阶段每 token 的预算。

影响因素：

- active sequences。
- KV Cache 访问。
- memory bandwidth。
- tensor parallel communication。
- sampling。
- batch shape。
- CUDA graph。
- speculative decoding。

TPOT p95 高常常说明 decode 阶段被压得太满。

如果 TPOT 高但 TTFT 正常，可能是：

- decode batch 太大。
- 长输出请求过多。
- KV Cache 内存压力大。
- GPU memory bandwidth 饱和。
- tensor parallel 通信开销高。

## Batching 与容量

Batching 是容量模型里最重要的变量之一。

增大 batch 往往会：

- 提高吞吐。
- 降低单位 token 成本。
- 增加排队时间。
- 增加 tail latency。
- 增加 KV Cache 压力。

所以不能问：

```text
最大 batch 能跑多少？
```

应该问：

```text
在 TTFT/TPOT SLA 下，batching 策略能承载多少 goodput？
```

推理服务的调度策略通常是在吞吐和延迟之间动态折中。

## Prefill/Decode 分离的容量影响

Prefill 和 decode 的资源特征不同。

| 阶段 | 特征 | 容量风险 |
| --- | --- | --- |
| Prefill | 计算密集、输入长度敏感 | 长 prompt 抬高 TTFT |
| Decode | memory/KV 敏感、持续迭代 | 并发和长输出抬高 TPOT |

分离部署后，可以分别扩容：

```text
prefill replicas = ceil(prefill_load / prefill_capacity_at_sla)
decode replicas = ceil(decode_load / decode_capacity_at_sla)
```

但分离也带来：

- KV 转移成本。
- 网络开销。
- 调度复杂度。
- failure handling。
- 更复杂的观测指标。

容量模型要把转移成本计入 TTFT 和 E2E。

## KV Cache 容量模型

KV Cache 是 LLM 推理容量的关键约束。

假设：

```text
available_kv_memory = 40 GiB
kv_bytes_per_token = 1 MiB
max_cached_tokens ~= 40K tokens
```

如果平均 active request 占用：

```text
input_tokens + generated_tokens = 2K
```

理论并发上限约：

```text
max_concurrency ~= 40K / 2K = 20 requests
```

如果 p95 请求占用 16K tokens，并发会大幅下降。

所以容量模型必须看 token occupancy，而不是只看请求数。

### KV Cache 预算表

建议为每个副本做 KV cache 预算：

| 项目 | 示例 |
| --- | --- |
| total GPU memory | 80 GiB |
| model weights | 45 GiB |
| runtime workspace / graphs | 5 GiB |
| safety reserve | 4 GiB |
| available for KV | 26 GiB |
| kv bytes per token | 0.8 MiB |
| theoretical cached tokens | ~33K |
| usable cached tokens | ~26K after fragmentation/reserve |

usable cached tokens 要留余量，因为实际系统还有：

- block fragmentation。
- allocator reserve。
- speculative decoding 额外状态。
- LoRA/adapter 或多模型显存。
- runtime 临时 buffer。

容量模型中最好设置一个 KV occupancy 上限，例如：

```text
do not exceed 80% sustained KV occupancy
scale out above 70% p95 KV occupancy
reject or route long-context requests before OOM
```

KV cache 接近满时，系统不一定马上报错，但 tail latency、调度失败和 eviction 行为会变得不可预测。

## Autoscaling

推理 autoscaling 常用信号：

- QPS。
- in-flight requests。
- queue length。
- TTFT p95。
- TPOT p95。
- GPU memory/KV occupancy。
- GPU utilization。
- tokens/sec。

只用 GPU utilization 做 autoscaling 往往不够，因为：

- GPU utilization 高时可能已经超过 tail SLA。
- GPU utilization 低时也可能因为 queueing 或 cache miss 导致 TTFT 高。
- decode memory-bound 时 SM utilization 不一定高。

更稳妥的是组合信号：

```text
scale out if:
  queue_length high
  OR TTFT p95 above target
  OR KV occupancy near limit
  OR goodput demand exceeds capacity
```

缩容也要小心：

- 不要缩掉热 cache。
- 不要在流量波谷刚开始就缩。
- 不要影响正在生成的请求。
- 要考虑模型加载冷启动时间。

## 冷启动与扩容时间

推理副本从创建到真正可服务，可能经历：

- 调度 Pod。
- 拉镜像。
- 加载模型权重。
- 初始化 engine。
- 构建或加载 TensorRT engine。
- CUDA graph capture。
- warmup。
- 注册到路由。

如果冷启动需要 5 分钟，而流量突发在 30 秒内到来，autoscaling 只能事后补救。

容量模型应包含：

```text
scale_out_time
model_load_time
warmup_time
traffic_ramp_rate
```

高价值在线服务通常需要预留 warm replicas，而不是完全依赖冷启动。

### 扩缩容滞后与滞回

Autoscaling 要建模滞后：

```text
scale_out_latency
  = scheduling_time
  + image_pull_time
  + model_load_time
  + engine_init_time
  + warmup_time
  + route_registration_time
```

如果 `scale_out_latency = 6 min`，而流量 spike 在 1 分钟内到达，自动扩容无法保护 p99。此时需要：

- warm pool。
- predictive scaling。
- scheduled scaling。
- 更高 steady-state headroom。
- 更快模型分发和本地缓存。

缩容也需要 hysteresis：

```text
scale out fast
scale in slow
keep warm replicas for cooldown window
do not terminate replicas with high KV/cache value immediately
```

否则系统会在流量波动时反复扩缩容，造成模型加载风暴和 cache miss。

## 路由与负载均衡

容量不是副本数相加那么简单。路由会影响实际利用率。

常见问题：

- 长请求集中到少数副本。
- 某些副本 KV Cache 满，其他副本空闲。
- sticky session 导致倾斜。
- prefix cache 命中和负载均衡冲突。
- 多模型共享 GPU 时互相影响。

路由策略要在两件事之间折中：

```text
load balance
cache locality
```

如果为了 prefix cache 命中把请求都打到同一副本，可能牺牲 tail latency。

### 路由效率折扣

容量模型可以引入 routing efficiency：

```text
effective_cluster_goodput
  = replicas * per_replica_goodput_at_sla * routing_efficiency
```

`routing_efficiency` 反映负载不均、sticky session、cache locality、长请求倾斜和 failure domain 带来的损失。

例如：

```text
replicas = 20
per_replica_goodput = 20 QPS
routing_efficiency = 0.85
effective_goodput = 340 QPS
```

如果按 400 QPS 规划，会高估容量。

需要监控：

- 每副本 in-flight requests。
- 每副本 KV occupancy。
- 每副本 TTFT/TPOT。
- 每副本输入/输出长度分布。
- 每副本 cache hit rate。
- 长请求是否集中。

容量问题有时不是副本少，而是路由没有把负载均衡到可用容量上。

## 多模型容量

多模型服务比单模型复杂。

需要考虑：

- 每个模型的权重显存。
- 每个模型的请求分布。
- 热模型和冷模型。
- 模型加载/卸载成本。
- KV Cache 是否隔离。
- 多模型 batch 是否可合并。
- 低流量模型是否独占 GPU。

多模型容量常见策略：

- 热模型独立副本。
- 中等流量模型共享池。
- 冷模型按需加载。
- 批处理模型离线服务。

容量模型要避免一个低频大模型占住大量显存，导致高频模型扩不起来。

## 降级与过载保护

容量模型还要定义过载时怎么保护服务。

常见策略：

- admission control：超过队列或 KV 阈值时拒绝新请求。
- max context limit：限制极长输入。
- max output limit：高峰时降低输出上限。
- priority queue：高优先级请求优先。
- load shedding：主动丢弃无法满足 SLA 的低价值请求。
- fallback model：降级到更小模型。
- cache-only / retrieval-only fallback：部分场景返回已有结果。

如果没有过载保护，系统会进入恶性循环：

```text
load exceeds capacity
  -> queue grows
  -> latency timeout
  -> client retry
  -> load grows further
```

容量规划不是保证永远不过载，而是定义过载时系统如何稳定退化。

## 成本模型

基础公式：

```text
cost_per_request = replica_cost_per_second / goodput_req_per_second
cost_per_output_token = replica_cost_per_second / output_tokens_per_second_at_sla
```

但生产成本还包括：

- headroom。
- 冷备副本。
- 模型加载带宽。
- cache miss。
- 失败重试。
- 日志和 tracing。
- 多可用区冗余。
- 在线/离线混部带来的干扰成本。

成本优化不能只追求最少副本。副本太少导致 p99 变差、重试增加、用户取消请求，反而可能更贵。

## Benchmark 设计

推理容量 benchmark 应该覆盖：

- 固定 QPS。
- 固定并发。
- 真实 trace replay。
- 多输入长度。
- 多输出长度。
- cache hit/miss。
- burst。
- long context。
- cold start。
- rolling update。

建议结果表：

| QPS | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | E2E p95 | Error | Goodput |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 20 | | | | | | | |
| 40 | | | | | | | |
| 60 | | | | | | | |

用这张曲线找到“满足 SLA 的最大 goodput”。

### 容量报告模板

推理容量报告建议包含：

```text
1. Capacity Contract
2. Traffic Distribution
3. SLA
4. Single Replica Benchmark Curve
5. KV Cache Budget
6. Replica Formula
7. Failure Domain / Rolling Update Assumptions
8. Autoscaling and Cold Start
9. Routing Efficiency
10. Cost Estimate
11. Risks and Caveats
12. Production Calibration Plan
```

报告结论不要只写“需要 17 个副本”，而要写：

```text
17 replicas can serve 160 peak QPS under trace v3,
with 30% headroom, one-node failure tolerance,
and 20% rolling-update max unavailable.
This assumes long-context bucket <= 10% and prefix cache hit >= 35%.
If long-context bucket reaches 20%, required replicas increase to 23.
```

容量结论必须带假设。假设一变，副本数就要重新算。

## 生产校准

上线后要持续比较：

```text
predicted_goodput vs actual_goodput
predicted_TTFT vs actual_TTFT
predicted_TPOT vs actual_TPOT
predicted_KV_occupancy vs actual_KV_occupancy
predicted_replicas vs actual_replicas
```

如果偏差很大，常见原因是：

- 生产输入长度分布变了。
- 输出长度变了。
- cache 命中率不同。
- 路由倾斜。
- 长尾请求比例高。
- 某些副本性能异常。
- 节点上有混部干扰。
- benchmark 没有覆盖真实 burst。

容量模型必须随生产数据更新。

### 校准频率

建议在这些事件后重新校准：

- 模型版本变化。
- tokenizer/chat template 变化。
- serving engine 升级。
- driver/CUDA/TensorRT 变更。
- 输入/输出长度分布变化。
- 新增 RAG、Agent、多模态功能。
- traffic pattern 明显变化。
- prefix cache 命中率变化。
- 节点池硬件变化。
- p95/p99 或 error rate 持续偏离预测。

容量模型不是写完就结束。它应该像 SLO 和成本模型一样，随生产事实更新。

## 常见误区

### 误区一：用峰值 tokens/s 算副本数

峰值吞吐通常不满足 tail latency SLA。容量要用 goodput at SLA。

### 误区二：平均输入输出长度足够

长尾长度决定 p95/p99 和 KV Cache 压力。必须看分布。

### 误区三：并发越高越好

并发提高可能增加吞吐，也可能让 TTFT、TPOT 和 KV occupancy 失控。

### 误区四：GPU 利用率低就该少副本

decode memory-bound、tail latency、headroom、cold start 都可能要求保留副本。

### 误区五：cache 命中 benchmark 可以代表全部请求

cache hit 和 cache miss 的容量完全不同，需要分开测。

### 误区六：autoscaling 可以解决所有容量问题

如果模型加载和 warmup 很慢，autoscaling 无法应对快速突发。

## 设计检查清单

- 是否有 Capacity Contract。
- 是否定义目标模型、engine、GPU 类型和并行方式。
- 是否记录输入长度和输出长度分布。
- 是否按请求 bucket 建模，而不是只用平均长度。
- 是否定义 QPS、burst、并发和 arrival pattern。
- 是否明确 TTFT、TPOT、E2E、error rate SLA。
- 是否区分 offered load、throughput 和 goodput。
- 是否用 goodput at SLA 计算容量。
- 是否找到 goodput 曲线和 latency/error 拐点。
- 是否扫 QPS/并发曲线，而不是只测单点。
- 是否单独建模 prefill 和 decode。
- 是否估算 KV Cache token occupancy。
- 是否有 KV Cache 预算表和 occupancy 上限。
- 是否考虑 prefix cache 命中率。
- 是否考虑冷启动、模型加载和 warmup。
- 是否建模 autoscaling 滞后和 hysteresis。
- 是否加入 headroom 和 failure domain。
- 是否考虑 rolling update max unavailable。
- 是否验证路由和负载均衡。
- 是否估算 routing efficiency。
- 是否考虑多模型共享。
- 是否定义过载保护和降级策略。
- 是否把成本按 request/token 归因。
- 是否有容量报告模板和假设说明。
- 是否上线后用生产数据校准模型。

## 小结

推理容量模型可以简化成：

```text
traffic distribution
  -> SLA
  -> single replica goodput curve
  -> KV/cache/concurrency constraints
  -> headroom and failure domain
  -> replica count
  -> production feedback
```

关键原则是：

```text
不要用峰值吞吐算容量。
用满足 SLA 的持续 goodput 算容量。
```

当模型、输入长度、输出长度、cache 命中率或路由策略变化时，容量模型也要重新校准。

## 延伸阅读

- [vLLM Benchmarking](https://docs.vllm.ai/en/latest/contributing/benchmarks.html)
- [NVIDIA Triton Performance Analyzer](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/docs/README.html)
- [NVIDIA Triton Model Analyzer](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/model_analyzer/docs/README.html)
- [MLPerf Inference Benchmark](https://mlcommons.org/benchmarks/inference-datacenter/)
- [Ray Serve Autoscaling](https://docs.ray.io/en/latest/serve/autoscaling-guide.html)
