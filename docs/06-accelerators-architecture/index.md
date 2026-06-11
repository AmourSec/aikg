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

- [AI 加速器性能模型：算力、带宽与 Roofline](performance-model-roofline.md)：用 arithmetic intensity、ridge point、多重 Roofline、compute/memory/network/energy roof、HBM、片上存储、矩阵单元、Prefill/Decode、训练、MoE 和 benchmark 建立硬件性能分析入口。
- [计算单元：SIMT、Tensor Core 与矩阵引擎](compute-units-simt-tensorcore.md)：解释 SIMD/SIMT、warp、SM、occupancy、register/shared memory、Tensor Core tile、systolic array、vector/load-store 单元、稀疏、动态控制流、并行切分和 profiler 指标如何影响真实算力。
- [存储层次：HBM、SRAM、Cache 与数据复用](memory-hierarchy-data-reuse.md)：解释 register spill、SRAM/shared memory、bank conflict、cache locality、HBM 容量/带宽预算、host/offload/UVM、KV Cache layout、fusion、IO-aware kernel、memory planning 和 profiler 证据如何共同决定数据搬运成本。
- [精度格式：FP16、BF16、FP8 与量化计算](precision-formats-low-bit-compute.md)：解释 FP32、TF32、FP16、BF16、FP8 E4M3/E5M2、INT8、INT4/NF4/FP4、accumulator、master weight、scale/amax、outlier、量化方法、KV Cache 量化、低精度通信、checkpoint metadata、硬件路径验证和 benchmark。
- [互连与通信架构：PCIe、NVLink、CXL、RDMA 与 NoC](interconnect-communication-architecture.md)：解释片内 NoC、chiplet interconnect、PCIe、NVLink/NVSwitch、CXL、GPU Direct RDMA、InfiniBand/RoCE、多 rail、collective 算法、NCCL/RCCL、rank mapping、拓扑感知并行、通信 benchmark、故障排查和互连可观测性。
- [功耗、散热、频率与可靠性：从峰值算力到持续吞吐](power-thermal-reliability.md)：解释从单卡到整柜的功耗/散热约束链、power limit、power capping、clock、throttling、遥测指标、热稳态实验、训练/推理能效、ECC、RAS、SDC、稳态 benchmark、故障诊断和 power-aware scheduling。
- [架构取舍：GPU、NPU、TPU、ASIC 与 FPGA](accelerator-architecture-tradeoffs.md)：解释 GPU、TPU、NPU、AI ASIC、FPGA 的通用性/专用性、执行模型、存储层次、精度支持、软件栈成熟度、算子覆盖、动态性、扩展方式、迁移成本、锁定风险、训练/推理分开评估、TCO、混合架构和 workload 匹配。
- [Workload Mapping：算子、Compiler、Runtime 与硬件执行](workload-mapping-compiler-runtime-interface.md)：解释模型、IR、算子、kernel、layout、tiling、fusion、runtime、dynamic shape、fallback 和 parallel mapping 如何共同决定硬件有效吞吐。
