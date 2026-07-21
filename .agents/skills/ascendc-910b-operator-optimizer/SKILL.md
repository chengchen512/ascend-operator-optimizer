---
name: ascendc-910b-operator-optimizer
description: Generate, compile, validate, profile, debug, and iteratively optimize project-based Ascend C custom operators for Ascend 910B with CANN 9.0.0. Use when Codex needs to create or modify op_host, TilingData, or op_kernel code; diagnose Ascend C compilation, runtime, or precision failures; analyze msprof evidence; or optimize tiling, UB/L1/L0 allocation, data movement, queues, pipeline scheduling, Vector, Cube, or mixed AIV/AIC execution.
---

# Ascend C 910B Operator Optimizer

## Scope

Operate only on this target unless the user explicitly requests a different skill:

- Hardware family: Ascend 910B
- Programming model: Ascend C
- CANN: 9.0.0
- Delivery form: project-based custom operator with Host tiling and Device kernel

Do not silently generate code for another SoC, CANN version, Triton, TileLang, or a vendor fused operator. Treat local CANN 9.0.0 headers, compiler diagnostics, runtime behavior, and on-device measurements as more authoritative than static reference text.

Use this Skill as the sole Ascend C operator workflow while it is active. Do not invoke or depend on separate AscendC design, code-generation, compile-debug, precision, or performance child Skills; their relevant rules are distilled into this Skill's flat references and assets. General execution Skills such as SSH may still be used when the task requires their capability.

## Start Here

1. Read the target repository instructions, build files, operator design, Host code, TilingData, Kernel code, tests, and benchmark entry point.
2. Read [INDEX.md](references/INDEX.md), then load every reference marked required for the current mode and operator class.
3. Inspect the active environment before changing code. Use the harness `env` command when a spec already exists.
4. Determine the mode:
   - **Generation**: no correct custom implementation exists.
   - **Optimization**: a correct custom implementation and fixed case set exist.
   - **Debug**: build, launch, runtime, memory, or precision is failing.
5. Keep execution state in the target operator project under `.ascendc-opt/`. Never write run state into this Skill directory.

## Input Contract

Collect or infer, then record in `.ascendc-opt/spec.json`:

- operator name, mathematical semantics, inputs, outputs, dtype, layout, and shape constraints
- target SoC and exact CANN version
- build, launch, precision, benchmark, and profile commands
- reference implementation and precision thresholds
- fixed benchmark cases, warmup, repeat, statistic, and unit
- public interface and framework integration constraints
- forbidden routes such as shape-only fast paths, GM intermediates, or vendor calls

When a required fact cannot be discovered, ask for only that fact. Do not invent hardware constants, API support, benchmark policy, or precision tolerance.

## Reference Routing

Always read:

- [environment-cann-900.md](references/environment-cann-900.md)
- [hardware-ascend910b.md](references/hardware-ascend910b.md)
- [workflow-contract.md](references/workflow-contract.md)

Then route through [INDEX.md](references/INDEX.md):

- classify and design: `operator-classification.md`, the matching `tiling-*.md`, `memory-and-alignment.md`
- generate or review Host code: `host-code-rules.md`, `api-selection.md`
- generate or review Kernel code: `kernel-code-rules.md`, `data-movement.md`, `pipeline-scheduling.md`
- compile or launch failure: `compile-and-debug.md`, `failure-patterns.md`, `framework-integration.md`
- precision failure: `numerical-precision.md`, `precision-validation.md`, `failure-patterns.md`
- performance optimization: `profiling-metrics.md`, `optimization-playbook.md`, `optimization-patterns.md`
- AWQ/W4A16/int4 local-B work: `optimization-patterns-awq-w4a16.md`

Search local CANN headers and the curated API source map before guessing an API signature. Load only the relevant reference files; do not ingest the whole knowledge base by default.

## Generation Mode

Execute these stages in order:

1. **Environment gate**: verify CANN 9.0.0 activation, SoC, compiler, device, and profiling tools.
2. **Specification gate**: normalize semantics, interface, constraints, and test/benchmark cases.
3. **Classification gate**: choose Vector, rowwise/reduction, index/data-movement, Cube/Matmul, or mixed/fused.
4. **Design gate**: write `design.md` with Host tiling, Device data path, queue/buffer lifecycle, UB/L1/L0 budget, tails, synchronization, and fallback behavior.
5. **Code gate**: generate Host, TilingData, Kernel, registration, framework wrapper, and tests from the closest asset template.
6. **Build gate**: compile with the active CANN environment and retain raw logs.
7. **Correctness gate**: pass functional, precision, boundary, and repeated-run checks.
8. **Baseline gate**: freeze the benchmark suite and establish custom and reference baselines.

Do not optimize while the implementation is still failing build or correctness.

## Optimization Mode

For each candidate:

1. Restore or branch from the last known-good candidate.
2. Run the fixed baseline when no comparable baseline exists.
3. Profile enough representative cases to identify one bottleneck class.
4. State one hypothesis, expected metric movement, changed region, and rejection condition.
5. Create a candidate snapshot before editing.
6. Apply the smallest coherent change.
7. Build and run precision before benchmarking.
8. Benchmark with identical cases and measurement metadata.
9. Compare baseline, current candidate, best candidate, and vendor/reference when available.
10. Accept or reject the candidate and append one record to `.ascendc-opt/history.jsonl`.
11. Repeat until a completion condition is met, then restore the best accepted candidate.

Use evidence in this order:

```text
fixed benchmark -> msprof/torch_npu profile -> code and resource model -> hypothesis
```

Profiling tags diagnose a bottleneck; they do not prove a speedup.

## Debug Mode

1. Reproduce with the smallest real launch path that still fails.
2. Preserve the command, environment, case, exit code, and raw logs in a run directory.
3. Classify the failure as environment, Host/tiling, compile, launch, memory/alignment, synchronization, precision, or performance measurement.
4. Run a focused probe such as launch coverage, single-core/single-tile, repeated execution, or MSSanitizer.
5. Fix the root contract before resuming generation or optimization.

Do not turn broad barriers, disabled cores, or hardcoded shapes into production fixes unless they are required by a documented contract and covered by general tests.

## Harness

Use `scripts/harness.py` as the deterministic execution layer:

```bash
python scripts/harness.py doctor
python scripts/harness.py init --workspace /path/to/operator --spec assets/operator-spec.example.yaml
python scripts/harness.py env --config /path/to/operator/.ascendc-opt/spec.json
python scripts/harness.py baseline --config /path/to/operator/.ascendc-opt/spec.json
python scripts/harness.py evaluate --config /path/to/operator/.ascendc-opt/spec.json --label candidate-001
python scripts/harness.py profile-analyze --profile-dir /path/to/trace
python scripts/harness.py report --config /path/to/operator/.ascendc-opt/spec.json
```

The harness records commands and evidence. It must not choose optimization strategies or rewrite source code.

## Hard Gates

- Never optimize before baseline correctness passes.
- Never accept a candidate that fails build, precision, repeated-run, or case-comparability checks.
- Never overwrite the last known-good or best candidate.
- Never report speedup from different shapes, dtype, implementation path, warmup, repeat, statistic, unit, or synchronization policy.
- Never infer core counts or memory capacities when `PlatformAscendC` or the active environment can query them.
- Never hardcode a target shape as the only fast path unless the user explicitly requests it and general fallback remains valid.
- Never introduce a GM intermediate that violates the operator's bandwidth objective.
- Never use vendor output as the candidate implementation.
- Never change multiple unrelated optimization dimensions in one candidate.
- Never treat compilation success, simulator output, a single timing sample, or profiler wait time as a final performance conclusion.
- Never leave queue, tensor, event, atomic, or Matmul lifecycle incomplete.

## State Contract

The target project owns this generated state:

```text
.ascendc-opt/
├── spec.json
├── environment.json
├── baseline.json
├── best.json
├── history.jsonl
├── candidates/
├── runs/
└── reports/final-report.md
```

Every history record must include candidate, parent, hypothesis, changes, build status, precision status, benchmark metric, baseline metric, speedup, and acceptance decision.

## Completion

Stop when one condition is true:

- the requested performance target is reached
- the optimization budget is exhausted
- consecutive evidence-backed candidates provide no accepted improvement
- the remaining bottleneck is outside the operator
- further changes violate precision, interface, memory, synchronization, or resource constraints

Deliver the best accepted implementation, `design.md`, fixed tests/cases, run evidence, and `.ascendc-opt/reports/final-report.md`. Report unsuccessful hypotheses as well as successful ones.
