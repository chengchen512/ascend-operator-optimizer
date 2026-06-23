# codex-ascend-operator-optimizer

A single Codex Skill for measured Ascend operator performance optimization.

## Design

- **One decision layer:** `.agents/skills/ascend-operator-optimizer/SKILL.md`
- **One compact knowledge base:** `.agents/skills/ascend-operator-optimizer/references/`
- **One deterministic harness:** `.agents/skills/ascend-operator-optimizer/scripts/harness.py`
- **One rule:** correctness first; performance claims require comparable measurements

The Skill implements this loop:

```text
inspect -> baseline -> diagnose -> one hypothesis -> patch
        -> correctness gate -> benchmark gate -> accept/revert -> repeat (max 3)
```

Supported operator implementations:

- Ascend C
- Triton-Ascend
- TileLang-Ascend

## Repository layout

```text
.agents/skills/ascend-operator-optimizer/
├── SKILL.md
├── assets/harness.example.json
├── references/
│   ├── INDEX.md
│   ├── hardware.md
│   ├── playbook.md
│   ├── constraints.md
│   ├── harness.md
│   ├── ascendc.md
│   ├── ascendc-examples.md
│   └── sources.md
└── scripts/harness.py

knowledge/   # legacy/source material, not required by the copied Skill
harness/     # legacy task/report templates, not required by the copied Skill
optimctl     # legacy local helper for the old task-template flow
```

## Use in an operator repository

Copy the single Skill directory into the target repository:

```bash
mkdir -p .agents/skills
cp -R /path/to/codex-ascend-operator-optimizer/.agents/skills/ascend-operator-optimizer \
  .agents/skills/
```

Create the harness configuration at the target repository root:

```bash
cp .agents/skills/ascend-operator-optimizer/assets/harness.example.json \
  operator-optim.json
```

Edit only the build, correctness, and benchmark commands. The benchmark command must write JSON to `$OPT_HARNESS_RESULT`.

Start Codex with:

```text
$ascend-operator-optimizer optimize this operator
```

## Harness commands

Check the copied Skill layout:

```bash
python .agents/skills/ascend-operator-optimizer/scripts/harness.py doctor
```

Create a baseline:

```bash
python .agents/skills/ascend-operator-optimizer/scripts/harness.py \
  baseline --config operator-optim.json
```

Evaluate a candidate patch:

```bash
python .agents/skills/ascend-operator-optimizer/scripts/harness.py \
  evaluate --config operator-optim.json --label iter-1
```

Run records are stored under `.operator-optim/` and should normally be excluded from source control.

## Benchmark result contract

The benchmark command receives these environment variables:

- `OPT_HARNESS_PHASE`: `baseline` or `evaluate`
- `OPT_HARNESS_LABEL`: stable run label
- `OPT_HARNESS_ROOT`: repository root from the config file
- `OPT_HARNESS_RUN_DIR`: artifact directory for this run
- `OPT_HARNESS_RESULT`: JSON file path the benchmark command must write

Minimal result shape:

```json
{
  "cases": [
    {
      "name": "case-id",
      "latency_us": 123.4
    }
  ]
}
```

Case names and units must stay stable across baseline and evaluation. Missing, extra, or incomparable cases invalidate a performance claim.

## Local compatibility helper

`./optimctl` is kept for compatibility with the earlier task-file workflow:

```bash
./optimctl doctor
./optimctl init --task-dir /path/to/repo/test --framework ascend-c --operator-name <op_name>
./optimctl status --task-dir /path/to/repo/test
```

New optimization runs should prefer the self-contained Skill harness above.

## Scope

The compact knowledge base covers the common Ascend performance model, tiling, data movement, memory residency, pipeline overlap, kernel constraints, correctness gates, benchmark comparability, and Ascend C-specific operator notes. Device-specific constants must be obtained from the active environment rather than copied into code.

