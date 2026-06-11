---
title: AI 加速器与计算架构
domain: accelerators-architecture
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-06-11
---

# AI 加速器与计算架构

本目录关注 GPU、NPU、TPU、ASIC、FPGA 等 AI 加速器如何支撑模型训练和推理。重点是计算、存储、互连、能效和可编程性如何约束上层系统。

## 建议主题

- SIMD、SIMT、Tensor Core、Systolic Array、Matrix Engine
- HBM、SRAM、cache、register file、memory hierarchy
- PCIe、NVLink、CXL、NoC、RDMA、chiplet interconnect
- FP16、BF16、FP8、INT8、INT4、混合精度
- arithmetic intensity、roofline、memory wall、data reuse
- power、thermal、frequency、reliability、ECC
- workload mapping、operator support、compiler/runtime interface
- GPU、NPU、TPU、ASIC、FPGA 的体系结构取舍

## 关键问题

- 给定 workload 的瓶颈是否与硬件计算峰值、带宽或容量匹配。
- 算子能否充分使用矩阵单元，数据搬运是否抵消计算收益。
- 精度格式、存储层次和互连如何影响训练和推理效率。
- 硬件特性是否需要编译器、runtime 或上层调度配合。
- Benchmark 是否能真实反映目标负载，而不是只测峰值指标。

## 专题入口

- [AI 加速器性能模型：算力、带宽与 Roofline](performance-model-roofline.md)：用 arithmetic intensity、compute-bound、memory-bound、HBM、片上存储、矩阵单元和 Roofline 建立硬件性能分析入口。
- [计算单元：SIMT、Tensor Core 与矩阵引擎](compute-units-simt-tensorcore.md)：解释 SIMD/SIMT、warp、SM、occupancy、Tensor Core、systolic array、稀疏和动态控制流如何影响真实算力。
- [存储层次：HBM、SRAM、Cache 与数据复用](memory-hierarchy-data-reuse.md)：解释 register、SRAM/shared memory、cache、HBM、host memory、offload、KV Cache、fusion 和 IO-aware kernel 如何共同决定数据搬运成本。
