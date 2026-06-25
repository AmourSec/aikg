---
title: Ascend 950 系列
domain: hardware
doc_type: knowledge
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-06-25
sources:
  - https://gitcode.com/cann/cannbot-skills/blob/master/ops/npu-arch/SKILL.md
  - https://www.tomshardware.com/tech-industry/semiconductors/huawei-unveils-ascend-roadmap-backed-by-in-house-hbm
  - https://www.techradar.com/pro/huawei-debuts-its-atlas-950-ai-superpod-at-mwc-2026-taking-the-ai-data-center-fight-to-nvidia-and-amd
---

# Ascend 950 系列

Ascend 950 系列在公开资料和 CANNBot skill 中已经出现较多线索，但对工程知识库来说，第一原则是：把它当作“需要持续验证的新平台”，不要把媒体报道或规划信息直接写成最终硬件事实。

本页只建立学习框架，后续应根据真实设备、CANN 版本、runtime API、platform config、benchmark 和 profiler 结果逐步补充。

## 公开线索怎么读

目前可见线索大致分成两类：

- CANNBot 的 NPU 架构 skill 把 Ascend950PR、Ascend950DT 与 `ASCEND950`、`DAV_3510` 等软件识别线索关联起来，并把 950PR/950DT 放入新一代 Atlas A5 方向。
- 媒体报道提到 Huawei Ascend 950PR/950DT、Atlas 950、HBM、低精度和大规模 SuperPoD / SuperCluster 等规划或发布信息。

这两类信息用途不同。CANNBot 更适合启发“软件栈如何识别和适配架构”；媒体报道更适合了解产业背景、产品路线和系统形态。真正做工程结论时，仍然要以当前 CANN、driver、runtime、设备查询和实测结果为准。

## 950PR 与 950DT 的学习切入点

| 方向 | 重点问题 |
| --- | --- |
| Ascend950PR | 更关注 Prefill、低精度推理、长上下文、大显存和高吞吐推理场景。 |
| Ascend950DT | 更关注 Decode、训练、通信、集群扩展和大规模系统形态。 |
| `DAV_3510` | 更关注架构能力、片上存储、数据格式、编译宏、SIMT/Regbase 等软件可见特性。 |
| Atlas 950 / A5 | 更关注系统级组合：卡、服务器、互连、机柜、集群和软件生态。 |

这里的“更关注”不是严格功能边界，而是学习和实验优先级。真实部署时，一个平台最终能跑什么 workload，取决于硬件形态、CANN 支持、驱动、框架、推理引擎和模型适配状态。

## 950 系列最该验证什么

新平台上不要先问“理论峰值是多少”，而要先验证这些问题：

1. `工具链可用性`：当前 CANN、compiler、simulator、profiler、torch_npu、推理引擎是否支持目标平台。
2. `模型路径`：目标模型是否能从 eager、图模式、推理引擎或自定义算子路径稳定运行。
3. `精度路径`：FP16、BF16、FP8、FP4 或量化路径是否可用，误差口径如何定义。
4. `内存路径`：权重、activation、KV Cache、temporary buffer、通信 buffer 是否符合预期。
5. `Prefill/Decode`：两阶段瓶颈是否不同，是否需要 PD 分离、异构部署或专门调度。
6. `通信与扩展`：多卡、多机、MoE、TP/PP/EP/DP 组合是否被通信、拓扑或 runtime 限制。
7. `profiling 证据`：是否能获得可解释的 kernel、runtime、memory、communication timeline。

## 适合沉淀成 skill 的方向

950 系列还处在快速演进阶段，最值得写成 skill 的不是“参数大全”，而是可重复执行的判断流程：

- `架构能力判断`：从设备日志、CANN 版本和 NpuArch 判断能否启用某个代码路径。
- `迁移基线建立`：把模型迁移到 950 系列时，先建立功能、精度、性能和回退路径。
- `Prefill/Decode 资源判断`：判断一个 workload 是否适合按 Prefill/Decode 分离或异构部署。
- `低精度验证`：判断 FP8/FP4/量化路径是否真的端到端生效。
- `profiling 证据收集`：把工具链输出组织成 AI 能够继续诊断的证据包。

## 最小实验模板

```text
hardware:
  platform: Atlas / Ascend 950 系列实际设备名
  device_count:
  topology:
software:
  cann:
  driver:
  runtime:
  framework:
  engine:
architecture:
  soc_version:
  npu_arch:
  compile_target:
workload:
  model:
  precision:
  input_length_distribution:
  output_length_distribution:
  batch_or_concurrency:
  parallel_strategy:
results:
  correctness:
  ttft_or_step_time:
  tpot_or_tokens_per_second:
  memory:
  profiler_artifacts:
  known_limits:
```

## 参考资料

- [CANNBot npu-arch skill](https://gitcode.com/cann/cannbot-skills/blob/master/ops/npu-arch/SKILL.md) 是了解 Ascend 架构识别、NpuArch、SocVersion 和条件编译组织方式的重要参考。
- [Tom's Hardware: Huawei Ascend roadmap](https://www.tomshardware.com/tech-industry/semiconductors/huawei-unveils-ascend-roadmap-backed-by-in-house-hbm) 报道了 Ascend 950PR/950DT 和 Atlas 950 等公开路线图线索。
- [TechRadar: Atlas 950 SuperPoD](https://www.techradar.com/pro/huawei-debuts-its-atlas-950-ai-superpod-at-mwc-2026-taking-the-ai-data-center-fight-to-nvidia-and-amd) 报道了 Atlas 950 系统形态、互连和大规模 AI 集群背景。
