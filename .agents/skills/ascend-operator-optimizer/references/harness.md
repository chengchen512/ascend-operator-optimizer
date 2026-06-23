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

Required commands:

- `build`: optional string; empty means skipped.
- `correctness`: required command; non-zero exit stops the run.
- `benchmark`: required command; must write JSON to `$OPT_HARNESS_RESULT`.

The harness runs commands with these environment variables:

| Variable | Meaning |
| --- | --- |
| `OPT_HARNESS_PHASE` | `baseline` or `evaluate` |
| `OPT_HARNESS_LABEL` | run label |
| `OPT_HARNESS_ROOT` | absolute target workspace directory |
| `OPT_HARNESS_RUN_DIR` | absolute current run directory |
| `OPT_HARNESS_RESULT` | path where benchmark command must write JSON |

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

## Comparable Measurement Rules

- Same case identity fields.
- Same unit.
- Same primary metric field.
- Same profiler/statistic semantics when metadata is available.
- No missing or non-numeric metric values.

