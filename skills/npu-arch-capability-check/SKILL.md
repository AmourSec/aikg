---
name: npu-arch-capability-check
description: Use when asked to judge Ascend NPU model, SocVersion, NpuArch, architecture capability, architecture-specific code branches, or whether a CANN/Ascend C/model path can run on a target NPU. This skill is for evidence-based capability checks, not general NPU education.
---

# NPU Architecture Capability Check

## Scope

Use this skill for Ascend NPU hardware and software capability questions, especially when the task mentions:

- Ascend 910, 910B, 910_93, 950PR, 950DT, Atlas A2, Atlas A3, or Atlas A5.
- SocVersion, NpuArch, `__NPU_ARCH__`, `DAV_2201`, `DAV_3510`, `archXX`, or architecture-specific branches.
- CANN, Ascend C, torch_npu, custom operators, model migration, profiling, simulator, or NPU inference optimization.

Do not use this skill to explain basic AI concepts. Use the documentation first for primers.

## Required Inputs

Ask for missing inputs before making a hard conclusion:

- Device model and device query output, such as `npu-smi info` or an equivalent inventory record.
- CANN Toolkit, Runtime, Driver, firmware, PyTorch, torch_npu, and inference engine versions.
- Target workload: model, precision, batch/concurrency, sequence length, input/output length distribution, and parallel strategy.
- The exact code path or question: model migration, custom operator, graph compile, profiling, precision, runtime error, or performance issue.
- Any available SocVersion, NpuArch, compile target, `__NPU_ARCH__`, `archXX`, CMake, build log, runtime log, profiler output, or platform config evidence.

## Workflow

1. Identify the naming layer.
   - Separate product/platform name, chip/SKU name, SocVersion, NpuArch, compile macro, and source tree directory.
   - Do not treat a marketing name as a complete hardware capability statement.

2. Map the target conservatively.
   - Ascend 910B and Ascend 910_93 commonly require checking the `ASCEND910B` / `DAV_2201` path.
   - Ascend 950PR and Ascend 950DT commonly require checking the `ASCEND950` / `DAV_3510` path.
   - If the evidence is only a media article, mark the conclusion as background information, not an engineering fact.

3. Check software-stack compatibility.
   - Verify CANN, driver, runtime, framework, and engine versions.
   - Check whether the target feature requires a newer CANN version, graph mode, simulator support, custom kernel path, or runtime flag.

4. Inspect architecture-specific code.
   - Search for `__NPU_ARCH__`, `DAV_`, `SocVersion`, `GetCurNpuArch`, `GetSocVersion`, `arch22`, `arch35`, and hardcoded memory/core parameters.
   - Treat hardcoded hardware parameters as suspect unless they are explicitly guarded and verified by runtime APIs.

5. Validate with a minimal workload.
   - Prefer a small functional test before performance tuning.
   - Add precision checks before benchmark claims.
   - Add profiler evidence before bottleneck claims.

6. Report evidence and gaps.
   - Separate confirmed facts, likely mappings, missing evidence, and required validation.
   - Include the exact file paths, commands, logs, or documentation references used.

## Output Template

```markdown
## Capability Check

### Summary
- Target:
- Likely mapping:
- Confidence:

### Evidence
- Hardware evidence:
- Software evidence:
- Code evidence:
- Runtime or profiler evidence:

### Decision
- Supported / not supported / unknown:
- Required conditions:
- Risk:

### Missing Inputs
- ...

### Validation Plan
1. ...
2. ...
3. ...

### References
- docs/12-hardware-basics/ascend-npu-models.md
- docs/12-hardware-basics/cann-stack.md
```

## Local Knowledge References

- `docs/12-hardware-basics/npu-basics.md`
- `docs/12-hardware-basics/ascend-npu-models.md`
- `docs/12-hardware-basics/ascend-910-series.md`
- `docs/12-hardware-basics/ascend-950-series.md`
- `docs/12-hardware-basics/cann-stack.md`

