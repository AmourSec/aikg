---
title: Ascend 型号、SocVersion 与 NpuArch
domain: hardware
doc_type: knowledge
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-06-29
sources:
  - https://gitcode.com/cann/cannbot-skills/blob/master/ops/npu-arch/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/master/ops/npu-arch/references/npu-hardware-params.md
---

# Ascend 型号、SocVersion 与 NpuArch

做昇腾平台适配时，最容易混淆的是“产品名、芯片型号、软件识别名、架构号、编译宏”这几层。工程上不能只写“910B”或“950”，因为同一代产品可能有多个子型号，同一个架构号也可能覆盖多个具体设备形态。

## 五层命名

| 层次 | 示例 | 作用 |
| --- | --- | --- |
| 产品或平台系列 | Atlas A2、Atlas A3、Atlas A5 | 面向产品和系统交付的名字，适合沟通硬件平台大类。 |
| 芯片或 SKU 名 | Ascend910B、Ascend910_93、Ascend950PR、Ascend950DT | 面向实际设备、卡、服务器或平台配置的名字。 |
| SocVersion | `ASCEND910B`、`ASCEND950` | 软件栈识别 SoC 版本时常见的枚举或字符串。 |
| NpuArch | `DAV_2201`、`DAV_3510` | 更接近架构代际和指令/微架构能力的标识。 |
| 编译宏或目录简写 | `__NPU_ARCH__=2201`、`arch22` | 用于条件编译、架构特化和算子工程目录组织。 |

这几层不是一一对应关系。一个 NpuArch 可能对应多个芯片型号；一个产品系列也可能因为卡形态、内存、频率、核心数量不同而表现不同。

## 当前重点映射

下表是面向服务器训练、推理和 AI Infra 学习的简化映射。它用于建立方向感，不替代 CANN 文档、platform config、runtime API 和实际设备查询。

| 平台线索 | 常见用途 | 软件识别重点 | 学习重点 |
| --- | --- | --- | --- |
| Ascend 910 | 早期训练平台 | `ASCEND910` / `DAV_1001` | 了解 910 系列历史和与后续 910B 的差异。 |
| Ascend 910B | 训练、推理、集群实验 | `ASCEND910B` / `DAV_2201` | 重点掌握 CANN、torch_npu、分布式训练、推理服务和算子适配。 |
| Ascend 910_93 | Atlas A3 相关训练/推理平台 | 通常仍要关注 `ASCEND910B` / `DAV_2201` 路径 | 不要只看营销型号，要确认 runtime 返回值和平台配置。 |
| Ascend 950PR | 新一代推理侧平台线索 | `ASCEND950` / `DAV_3510` 线索 | 关注 Prefill、低精度、HBM、CANN 支持状态和实际可用工具链。 |
| Ascend 950DT | 新一代训练/Decode 侧平台线索 | `ASCEND950` / `DAV_3510` 线索 | 关注 Decode、训练、通信、集群形态和 simulator/真实硬件差异。 |

## 判断硬件能力不要硬编码

硬件能力判断应该遵循这个顺序：

1. 收集当前设备和软件栈信息：设备型号、CANN、driver、runtime、framework、推理引擎、编译目标。
2. 通过 runtime 或平台接口确认 SocVersion、NpuArch、核心数、片上存储、内存等关键能力。
3. 查阅对应版本的 CANN 文档、platform config 和 release notes。
4. 检查代码里是否存在 `__NPU_ARCH__`、`archXX`、SocVersion 分支或硬编码参数。
5. 用最小 workload 验证功能、精度和性能，再推广到真实模型。

不能只根据设备名推断所有参数。原因很简单：同一架构下的子型号可能有不同核心数、内存容量、频率和系统形态；同一模型在框架图模式、算子融合、量化或自定义 kernel 路径上也可能走不同执行路径。

## 适合写入实验记录的字段

| 字段 | 示例写法 |
| --- | --- |
| 硬件平台 | `Atlas A2 / Ascend910B`，或实际集群资产名。 |
| 设备查询 | 保存 `npu-smi info`、拓扑、设备数量和内存信息。 |
| 软件栈 | CANN Toolkit、Runtime、Driver、torch_npu、PyTorch、推理引擎版本。 |
| 架构识别 | SocVersion、NpuArch、编译宏、arch 目录。 |
| Workload | 模型、精度、batch、sequence length、并行策略、输入输出长度分布。 |
| 结论边界 | 该结论只对哪个硬件、CANN 版本、shape 和 workload 成立。 |

## 与 AI skill 的关系

型号映射非常适合写成 AI skill，因为它不是科普，而是一个可重复执行的判断流程：

- 用户给出设备型号或日志。
- AI 先识别平台命名层次。
- AI 再要求补齐缺失的 CANN、driver、runtime、SocVersion、NpuArch 信息。
- AI 检查是否存在错误的硬编码或错误的架构条件分支。
- AI 输出“可判断结论、缺失证据、验证步骤、风险边界”。

本仓库的 `skills/npu-arch-capability-check/SKILL.md` 就是这种打样。

## 参考资料

- [CANNBot Skills 项目](https://gitcode.com/cann/cannbot-skills) 提供了面向 CANN 开发的 skill 组织方式。
- [CANNBot npu-arch skill](https://gitcode.com/cann/cannbot-skills/blob/master/ops/npu-arch/SKILL.md) 展示了如何把 NpuArch、SocVersion、编译宏和架构差异组织成可调用知识。
- [CANNBot NPU 硬件参数参考](https://gitcode.com/cann/cannbot-skills/blob/master/ops/npu-arch/references/npu-hardware-params.md) 给出了硬件映射和参数来源说明；本仓库只抽象组织方法，不复制其内部表格作为真源。
