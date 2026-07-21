# Reference Index

Load only the rows needed for the current task. Files are one level below `SKILL.md` so every route remains discoverable.

| Scenario | Required references |
| --- | --- |
| Every task | `environment-cann-900.md`, `hardware-ascend910b.md`, `workflow-contract.md` |
| New operator or redesign | `operator-classification.md`, `host-code-rules.md`, `kernel-code-rules.md`, matching `tiling-*.md` |
| Elementwise/pointwise | `tiling-elementwise.md`, `data-movement.md`, `pipeline-scheduling.md` |
| Rowwise/reduction/normalize | `tiling-row-reduction.md`, `numerical-precision.md`, `memory-and-alignment.md` |
| Gather/scatter/index/transpose | `tiling-index.md`, `memory-and-alignment.md`, `data-movement.md` |
| Cube/Matmul/mixed AIC-AIV | `tiling-cube.md`, `api-selection.md`, `pipeline-scheduling.md`, `framework-integration.md` |
| Host or TilingData changes | `host-code-rules.md`, `tiling-general.md`, `api-selection.md` |
| Kernel/API changes | `kernel-code-rules.md`, `api-selection.md`, `data-movement.md` |
| Build or launch failure | `compile-and-debug.md`, `failure-patterns.md`, `framework-integration.md` |
| Runtime 507015 or memory fault | `failure-patterns.md`, `memory-and-alignment.md`, `pipeline-scheduling.md` |
| Precision failure | `precision-validation.md`, `numerical-precision.md`, `failure-patterns.md` |
| Performance work | `profiling-metrics.md`, `optimization-playbook.md`, `optimization-patterns.md` |
| AWQ/W4A16/int4 local-B | `optimization-patterns-awq-w4a16.md`, `tiling-cube.md`, `api-selection.md` |
| Harness or run state | `harness-contract.md`, `workflow-contract.md` |
| Source provenance | `sources.md` |

## Search Before Loading More

For files longer than 100 lines, inspect headings first and search for the API, memory position, format, error code, or pipeline name. Useful terms include:

```text
DataCopyPad TQue TQueBind TSCM VECOUT SetTensorB Matmul Mmad Fixpipe
PlatformAscendC GetCoreNum GetUbSize GetLibApiWorkSpaceSize 507015
```

Do not recursively read the whole directory. Start from this table, search the selected file, then read the relevant section in full.
