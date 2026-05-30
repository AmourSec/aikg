---
title: 技术决策记录
domain: technical-decision-records
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-05-30
---

# 技术决策记录

本页用于记录 AI Infra 方向的重要技术选择。决策记录应基于明确 workload、Benchmark 结果和约束条件，而不是只记录主观偏好。

## 适用场景

- 选择推理 runtime、训练框架、通信库、调度系统或存储方案。
- 决定是否采用量化、KV Cache 策略、batching 策略或 speculative decoding。
- 决定某个 Kernel、编译器优化或硬件特性的投入方向。
- 决定 Benchmark 口径、容量模型或线上 SLO。

## 必填信息

- 决策背景和目标指标。
- workload、硬件、软件和数据路径约束。
- 候选方案和拒绝原因。
- Benchmark 证据、profiler 证据或论文依据。
- 风险、回滚方式和后续验证计划。

## 关键问题

- 这个决策优化的是延迟、吞吐、成本、能效、可靠性还是可维护性。
- 证据是否足以支持当前 workload，而不是只支持 toy benchmark。
- 未来模型规模、上下文长度、并发或硬件变化后，决策是否仍然成立。
