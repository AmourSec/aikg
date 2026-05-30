---
title: AI 计算架构与硬件基础
domain: ai-compute-infra
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-05-30
---

# AI 计算架构与硬件基础

本目录关注 AI 计算背后的计算机体系结构和硬件基础，连接模型计算模式、系统软件和加速器设计。

## 建议主题

- CPU、GPU、NPU、ASIC、FPGA
- Tensor Core、SIMT、Systolic Array、Vector Unit
- HBM、DDR、SRAM、Cache、NUMA、Memory Wall
- PCIe、CXL、NVLink、InfiniBand、RoCE、NoC
- Roofline、算术强度、带宽瓶颈和通信瓶颈
- 能效、可靠性、可扩展性和故障模型

## 关键问题

- 某类模型或算子受限于计算、内存还是互连
- 加速器架构如何影响训练、推理和编译器设计
- 性能指标如何从 FLOPS、带宽、延迟扩展到能效和可靠性
