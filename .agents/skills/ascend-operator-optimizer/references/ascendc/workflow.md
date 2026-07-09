# Ascend C Workflow Pack

本包把 Ascend/agent-skills 的端到端 AscendC skill 蒸馏成当前单 Skill 可用的检查表。不要把上游多 skill 工作流原样编排进 harness；harness 只负责执行 gate、保存证据和比较结果。

## 上游 Skill 到本项目的映射

| 上游类型 | 本项目吸收方式 |
| --- | --- |
| `ascendc-operator-dev` 七阶段流程 | `inspect -> env_smoke -> build -> probes -> correctness -> benchmark -> profile -> compare` 检查表 |
| `code-gen` references | `api-usage.md`、`data-copy.md`、短代码模式 |
| `performance-optim` references | `tiling-grid.md`、`memory.md`、`pipeline.md`、性能诊断顺序 |
| `performance-eval` | `harness.md` benchmark/profile 契约 |
| `precision-eval` / `precision-debug` | `precision.md` 精度指标和症状定位 |
| `mssanitizer` | `precision.md` 运行时内存/同步 probe |
| `ascend-profiling-anomaly` | `profiling.md` profiler CSV 分析和软标签 |

## Harness Gate

当前 harness 的 AscendC 顺序是：

```text
env_smoke -> build -> launch_probe -> correctness -> benchmark -> profile -> compare
```

- `env_smoke`：可选，检查 CANN、torch-npu、编译器、设备可见性和基础导入。
- `build`：可选，构建或 JIT 编译。
- `launch_probe`：可选 gate；`msopgen-aclnn-dynamic` launch profile 默认必需。
- `correctness`：必需，失败时不进入 benchmark。
- `benchmark`：必需，必须写 `$OPT_HARNESS_RESULT`。
- `profile`：可选，生成 profiler trace；harness 可解析 `kernel_details.csv` / `op_statistic.csv`。
- `compare`：evaluate 阶段默认要求和 baseline 同 case、同单位、同 metadata 语义。

## 使用边界

- `SKILL.md` 负责路由和一轮一个假设的优化循环。
- `references/ascendc/*` 负责知识包，不直接驱动命令。
- `scripts/harness.py` 不读上游 skill，不做自动决策，只验证 gate 结果。
- 性能收益必须来自 benchmark compare；profile 标签只用于诊断。
