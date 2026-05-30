---
title: 训练算法、微调与对齐
domain: training-finetuning-alignment
status: draft
owner: maintainers
license: CC-BY-4.0
updated: 2026-05-30
---

# 训练算法、微调与对齐

本目录整理从预训练到参数高效微调、偏好优化和对齐的核心算法，强调目标函数、优化稳定性、实验设计和复现细节。

## 建议主题

- 预训练、继续预训练、监督微调
- LoRA、QLoRA、Adapter、Prompt Tuning
- RLHF、DPO、RLAIF、偏好优化
- 分布式训练、混合精度、梯度检查点
- Optimizer、Learning Rate Schedule、Regularization
- Checkpoint、随机性控制、实验追踪、可复现训练
- 训练稳定性、过拟合、灾难性遗忘、能力退化

## 关键问题

- 训练目标、数据分布和评测指标是否一致
- 论文结果是否能被独立复现
- 超参数、随机种子和硬件环境是否记录完整
- 微调是否改变基础能力、校准性和安全性
- 对齐方法是否引入新的偏差、奖励黑客或泛化问题
