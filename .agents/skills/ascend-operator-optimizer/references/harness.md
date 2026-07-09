# Harness Contract

The harness is `scripts/harness.py`. It provides deterministic run records for
baseline and candidate evaluation.

## Commands

```bash
python .agents/skills/ascend-operator-optimizer/scripts/harness.py \
  baseline --config operator-optim.json

python .agents/skills/ascend-operator-optimizer/scripts/harness.py \
  evaluate --config operator-optim.json --label iter-1
```

Run records are written under `.operator-optim/`.

## Configuration

Create `operator-optim.json` at the target repository root from
`assets/harness.example.json`. Edit only commands and local metadata.

Commands:

- `env_smoke`: optional string; checks CANN/torch-npu/device/compiler basics.
- `build`: optional string; empty means skipped.
- `launch_probe`: optional string; required by default for `ascendc-msopgen-aclnn-dynamic`.
- `correctness`: required command; non-zero exit stops the run.
- `benchmark`: required command; must write JSON to `$OPT_HARNESS_RESULT`.
- `profile`: optional string; writes profiler trace under `$OPT_HARNESS_PROFILE_DIR`.

Implementation metadata:

```json
{
  "implementation": {
    "type": "ascendc",
    "launch_profile": "ascendc-standard",
    "op_name": "OperatorName"
  }
}
```

Known `launch_profile` values:

- `ascendc-standard`
- `ascendc-msopgen-aclnn-dynamic`
- `ascendc-cube-matmul`
- `ascendc-vector`

The harness runs commands with these environment variables:

| Variable | Meaning |
| --- | --- |
| `OPT_HARNESS_PHASE` | `baseline` or `evaluate` |
| `OPT_HARNESS_LABEL` | run label |
| `OPT_HARNESS_ROOT` | absolute target workspace directory |
| `OPT_HARNESS_RUN_DIR` | absolute current run directory |
| `OPT_HARNESS_RESULT` | path where benchmark command must write JSON |
| `OPT_HARNESS_PROBE_RESULT` | path where launch probe command should write JSON |
| `OPT_HARNESS_PROFILE_DIR` | directory where profile command should write trace artifacts |
| `OPT_HARNESS_PROFILE_RESULT` | optional JSON result path for profile command |
| `OPT_HARNESS_IMPLEMENTATION_TYPE` | configured implementation type |
| `OPT_HARNESS_LAUNCH_PROFILE` | configured launch profile |
| `OPT_HARNESS_OP_NAME` | configured operator name |

## Launch Probe JSON

`launch_probe` should exercise the target repository's real launch path. For
msopgen/aclnn dynamic custom ops, this gate catches blockDim/GetBlockIdx
assumptions before kernel math optimization.

Minimal JSON:

```json
{
  "name": "launch_coverage_probe",
  "status": "pass",
  "launch_profile": "ascendc-msopgen-aclnn-dynamic",
  "block_dim": 8,
  "coverage": 1.0,
  "observed_block_idx": [0, 1, 2, 3, 4, 5, 6, 7],
  "observed_block_num": 8,
  "notes": []
}
```

When `gates.require_launch_probe=true`, or when `launch_profile` is
`ascendc-msopgen-aclnn-dynamic`, the harness fails if this JSON is missing,
not passing, or below `probes.launch_coverage_min` (default `1.0`).

## Benchmark JSON

The benchmark command writes:

```json
{
  "unit": "us",
  "metadata": {
    "profiler": "torch_npu.profiler",
    "warmup": 5,
    "repeat": 1,
    "active": 5,
    "statistic": "mean"
  },
  "cases": [
    {
      "case": "0",
      "shape": [128, 4096],
      "dtype": "float16",
      "mode": "default",
      "custom": 9.75,
      "reference": 10.10
    }
  ]
}
```

`cases` is required. Each case needs enough identity fields to match baseline
and candidate results. The default match fields are `case`, `shape`, `dtype`,
and `mode`.

The default candidate metric is `custom`. You can override it through
`benchmark.primary_metric` in the config. Smaller is better by default.

## Baseline

`baseline` runs build, correctness, benchmark, validates benchmark JSON, and
writes:

- `.operator-optim/baseline.json`
- `.operator-optim/runs/<run-id>/run.json`
- `.operator-optim/runs/<run-id>/benchmark.json`
- `.operator-optim/runs/<run-id>/summary.md`

## Evaluate

`evaluate` runs the same gates, then compares candidate cases with
`.operator-optim/baseline.json`. It fails only for execution or comparability
errors; regressions are recorded for the agent to accept or revert.

Outputs include:

- `.operator-optim/latest-evaluation.json`
- `.operator-optim/runs/<run-id>/comparison.json`
- `.operator-optim/runs/<run-id>/summary.md`

## Profile

`profile` is a diagnostic gate. It must not replace benchmark comparison.

Recommended trace location:

```text
$OPT_HARNESS_PROFILE_DIR
```

If `commands.profile` is configured or `profiling.analyze=true`, the harness
tries to parse:

```text
**/kernel_details.csv
**/op_statistic.csv
```

Outputs:

- `.operator-optim/runs/<run-id>/profile_analysis.json`
- `.operator-optim/runs/<run-id>/profile_summary.md`

The built-in profile analyzer records service time, device busy union, kernel
sum, wait sum, underfeed ratio, largest internal bubble, top kernels, and soft
tags such as `DEVICE_IDLE_GAP_HEAVY`, `INTERNAL_BUBBLE_HEAVY`, and
`WAIT_ANCHOR_FALSE_HOTSPOT`.

You can analyze an existing trace directory:

```bash
python .agents/skills/ascend-operator-optimizer/scripts/harness.py \
  profile-analyze --profile-dir .operator-optim/runs/<run-id>/profile
```

## Artifact Collection

Optional config:

```json
{
  "artifacts": {
    "collect": [
      "generated/**/binary_info_config.json",
      "generated/**/*ops-info.json",
      "logs/**",
      "profile/**"
    ]
  }
}
```

Collected files are copied into the current run directory under `collected/`.

## Comparable Measurement Rules

- Same case identity fields.
- Same unit.
- Same primary metric field.
- Same profiler/statistic semantics when metadata is available.
- No missing or non-numeric metric values.

