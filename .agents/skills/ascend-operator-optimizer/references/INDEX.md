# Ascend Operator Optimizer References

本目录是 `ascend-operator-optimizer` 的紧凑知识库，随 Skill 单目录复制。

读取顺序：

1. `playbook.md`：优化循环、门禁和报告规则。
2. `constraints.md`：硬性停止条件和实现侧禁忌。
3. `harness.md`：确定性 baseline/evaluate 契约。
4. `hardware.md`：通用 Ascend 性能模型。
5. `api-lookup.md`：官方 API 快照的高效检索流程和查询词路由。
6. `ascendc.md`：Ascend C 算子设计、调试、优化总览。
7. `ascendc/`：按场景拆分的 AscendC knowledge packs。
8. `cases/`：实际优化案例，含 AWQ W4A16 no-cache 与 fused Cube/MMAD 探索经验。
9. `ascendc-examples.md`：Ascend C 高频代码模式示例。
10. `sources.md`：来源映射和蒸馏边界。

## AscendC Knowledge Pack 路由

| 场景 | 读取文件 |
| --- | --- |
| 确认端到端流程和 gate | `ascendc/workflow.md` |
| 判断 custom op launch 语义 | `ascendc/launch-profiles.md` |
| 性能评估或 profiler 诊断 | `ascendc/profiling.md` |
| 改 tiling、block/grid、tail | `ascendc/tiling-grid.md` |
| 改 GM/UB 搬运、stride、padding | `ascendc/data-copy.md` |
| 改 AscendC API、TQue/TBuf、Atomic | `ascendc/api-usage.md` |
| 改 UB/L1/L0/BT/FP buffer 策略 | `ascendc/memory.md` |
| 改 CopyIn/Compute/CopyOut 流水 | `ascendc/pipeline.md` |
| 精度、MSSanitizer、同步错误 | `ascendc/precision.md` |
| 查询 API 签名、重载、参数单位、ISASI 限制 | `api-lookup.md` |
| AWQ/W4A16、int4 unpack、no-cache 反量化、TSCM/Cube/MMAD 探索 | `cases/awq-w4a16-ascendc.md` |

在目标算子仓库工作时，不依赖本 Skill 目录之外的文件。若仓库根目录存在 `reference/asc-devkit-api-9.0.0/docs/api`，按 `api-lookup.md` 把它作为离线 API 证据源。
