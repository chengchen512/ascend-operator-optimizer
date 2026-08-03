# Harness Contract

`scripts/harness.py` is the deterministic execution layer. It records evidence but does not select or apply optimizations.

## State

All generated state lives under the target workspace:

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

## Commands

```bash
python scripts/harness.py doctor
python scripts/harness.py init --workspace /path/to/op --spec assets/operator-spec.example.yaml
python scripts/harness.py env --config /path/to/op/.ascendc-opt/spec.json
python scripts/harness.py build --config /path/to/op/.ascendc-opt/spec.json
python scripts/harness.py precision --config /path/to/op/.ascendc-opt/spec.json
python scripts/harness.py benchmark --config /path/to/op/.ascendc-opt/spec.json --label smoke
python scripts/harness.py profile --config /path/to/op/.ascendc-opt/spec.json --label smoke
python scripts/harness.py baseline --config /path/to/op/.ascendc-opt/spec.json
python scripts/harness.py evaluate --config /path/to/op/.ascendc-opt/spec.json --label candidate-001
python scripts/harness.py compare --config /path/to/op/.ascendc-opt/spec.json --candidate result.json
python scripts/harness.py report --config /path/to/op/.ascendc-opt/spec.json
```

## Benchmark Result

The benchmark command writes JSON to `$ASCENDC_OPT_RESULT`:

```json
{
  "unit": "us",
  "metadata": {"warmup": 5, "repeat": 30, "statistic": "mean"},
  "cases": [
    {"case": "m16-k4096-n22016", "shape": [16, 4096, 22016], "dtype": "float16", "mode": "auto", "candidate": 298.3, "reference": 123.3}
  ]
}
```

Case keys, unit, and configured metadata must match the baseline. Missing or extra cases invalidate comparison.

## Execution Environment

Commands receive:

- `ASCENDC_OPT_PHASE`, `ASCENDC_OPT_LABEL`
- `ASCENDC_OPT_ROOT`, `ASCENDC_OPT_RUN_DIR`
- `ASCENDC_OPT_RESULT`, `ASCENDC_OPT_PROBE_RESULT`
- `ASCENDC_OPT_PROFILE_DIR`, `ASCENDC_OPT_PROFILE_RESULT`
- compatibility aliases using the former `OPT_HARNESS_*` names

## Baseline and Evaluate

`baseline` runs environment smoke, build, launch probe when required, precision, benchmark, and optional profile. `evaluate` runs the same gates, compares with the saved baseline, and writes a candidate history record. Regression does not erase evidence; acceptance remains an explicit agent decision.
