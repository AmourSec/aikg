---
title: Ascend 910 系列平台要点
domain: hardware
doc_type: knowledge
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-06-29
sources:
  - https://gitcode.com/cann/cannbot-skills/blob/master/ops/npu-arch/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/master/README.md
---

# Ascend 910 系列平台要点

Ascend 910 系列是学习服务器侧昇腾 AI 计算时最应该先掌握的一条线。对 AI Infra 来说，重点不是背型号参数，而是建立三件事：

- 设备型号和软件栈识别之间的对应关系。
- 训练、推理、算子和编译路径在该平台上分别怎么落地。
- 每个性能或正确性结论依赖哪个 CANN、driver、framework、模型 shape 和并行策略。

## 910、910B、910_93 怎么看

| 名称 | 工程上更应该关注什么 |
| --- | --- |
| Ascend 910 | 早期训练平台。用于理解 910 系列历史、CANN 路径和与后续平台的差异。 |
| Ascend 910B | 当前服务器训练、推理和平台适配中更常见的学习对象。需要重点记录 `ASCEND910B`、`DAV_2201`、CANN 和 framework 版本。 |
| Ascend 910_93 | 常见于 Atlas A3 相关平台线索。工程判断时不要只看名字，要确认 runtime 和 CANN 返回的 SocVersion/NpuArch。 |

同一个“910B 方向”的问题，可能发生在不同层：

- 模型层：模型结构、精度、sequence length、batch、MoE 路由或 KV Cache 使用方式。
- 框架层：PyTorch、torch_npu、图模式、fallback、算子覆盖、autograd 或 collective 路径。
- 编译层：算子选择、fusion、tiling、动态 shape、架构条件编译。
- Runtime 层：stream、内存、设备队列、通信、错误码、profiling 事件。
- 集群层：节点拓扑、网络、rank mapping、调度、故障域。

## 训练场景重点

910 系列训练问题通常沿着下面几条线排查：

1. `数据路径`：DataLoader、tokenization、packing、host 到 device 的搬运是否形成瓶颈。
2. `显存结构`：parameters、gradients、optimizer states、activation、temporary buffer 哪一项占主导。
3. `并行策略`：DP/FSDP/ZeRO/TP/PP/EP/CP 是否匹配设备数量、网络和模型结构。
4. `通信成本`：AllReduce、ReduceScatter、AllGather、AllToAll 是否与计算重叠。
5. `数值稳定`：混合精度、loss scaling、梯度裁剪、NaN/Inf、loss spike 是否被监控。
6. `checkpoint`：保存、恢复、resharding、故障恢复是否可复现。

这些内容已经分别放在 [训练系统与优化](../04-training-systems/index.md) 中。本页只强调：在 NPU 平台上做结论时必须加上硬件和 CANN 上下文。

## 推理场景重点

910 系列推理问题通常围绕：

- Prefill 与 Decode 的资源形态不同。
- KV Cache 占用显存，并影响并发、长上下文和调度。
- Continuous batching、PagedAttention、Prefix Cache、Speculative Decoding 需要推理引擎和后端 kernel 支持。
- MoE 推理会引入 expert placement、token dispatch/combine、EP size、AllToAll 和尾延迟问题。
- 单机推理与多机分布式推理要分别看调度、通信、缓存和容错。

如果问题是“某个模型迁移到 910B 后慢或报错”，不要直接进入 kernel 调优。更稳的顺序是：先建立功能基线和精度基线，再打开 profiling，看瓶颈是在框架 fallback、算子覆盖、kernel、通信、内存还是调度。

## 算子与编译场景重点

在 910 系列上做算子或编译优化，常见输入是：

- 某个 PyTorch 算子没有高效路径。
- 模型图里有大量小算子，launch、搬运或中间 tensor 成本过高。
- Attention、LayerNorm、RMSNorm、RoPE、MLP、MoE dispatch 等热点需要融合。
- 动态 shape 或特定 layout 导致图编译和 kernel 选择不稳定。
- 自定义算子需要针对 NpuArch 做 tiling 和条件编译。

这时应该把“设备型号”和“架构能力”作为设计约束，而不是事后补充说明。

## 910 系列实验清单

| 阶段 | 必要检查 |
| --- | --- |
| 建立基线 | 记录设备、CANN、driver、framework、模型、精度、shape、并行策略。 |
| 功能验证 | 用小 batch、小输入、固定随机种子验证能否稳定运行。 |
| 精度验证 | 对比 CPU/GPU/已知正确输出，记录误差口径和容忍范围。 |
| 性能验证 | 分别记录 warmup、测量窗口、吞吐、延迟、显存、功耗和 profiler。 |
| 瓶颈定位 | 区分 data、framework、compiler、kernel、memory、communication、scheduler。 |
| 结论沉淀 | 形成 benchmark report、failure case、ADR 或 skill。 |

## 参考资料

- [CANNBot README](https://gitcode.com/cann/cannbot-skills/blob/master/README.md) 将昇腾相关 skill 分成算子开发、PyPTO、TileLang、Triton、NPU 模型推理优化、代码检视、torch.compile 等场景。
- [CANNBot npu-arch skill](https://gitcode.com/cann/cannbot-skills/blob/master/ops/npu-arch/SKILL.md) 展示了 SocVersion、NpuArch、编译宏和架构差异的组织方式。
