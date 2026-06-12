---
title: 可靠性、可观测性与故障复盘
domain: reliability-observability
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-05-30
---

# 可靠性、可观测性与故障复盘

本目录关注 AI 系统如何长期稳定运行。性能优化如果不能被观测、不能恢复、不能复盘，就无法沉淀成可靠的基础设施能力。

## 建议主题

- metrics、logs、traces、profiles、events
- GPU health、ECC、Xid、temperature、power、clock、memory error
- OOM、NCCL timeout、hang、kernel failure、driver reset
- queue backlog、tail latency、straggler、checkpoint failure
- alerting、SLO、error budget、runbook
- fault injection、chaos test、rollback、fallback、graceful degradation
- incident review、root cause analysis、corrective action
- 性能回归、容量耗尽、版本升级风险

## 关键问题

- 失败是否能被快速发现、定位、隔离和恢复。
- 监控指标是否覆盖 workload、runtime、kernel、hardware 和 cluster。
- 故障复盘是否给出可验证的改进项。
- 可靠性改动是否影响延迟、吞吐或资源利用率。
- 线上故障经验是否能转化为 Benchmark、测试用例和知识库条目。

## 专题入口

- [AI 系统可观测性总览：Metrics、Logs、Traces、Profiles 与 Events](observability-overview-signals.md)：解释 AI 推理、训练和集群系统如何使用 metrics、logs、traces、profiles、events 五类观测信号，建立 latency/traffic/errors/saturation 黄金信号、分层观测、black-box/white-box、关联字段、cardinality 治理、告警、dashboard、sampling、retention、benchmark 联动和最小可观测性闭环。
