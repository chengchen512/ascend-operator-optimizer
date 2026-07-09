# codex-ascend-operator-optimizer

A Codex Skill project for measured Ascend operator performance optimization.
It packages the optimization workflow, compact references, and validation
harness needed to optimize Ascend operator implementations repeatably.

## Design

- **One decision layer:** `.agents/skills/ascend-operator-optimizer/SKILL.md`
- **One compact knowledge base:** `.agents/skills/ascend-operator-optimizer/references/`
- **One deterministic harness:** `.agents/skills/ascend-operator-optimizer/scripts/harness.py`
- **One rule:** correctness first; performance claims require comparable measurements

The Skill implements this loop:

```text
inspect -> env_smoke -> build -> probes -> baseline -> diagnose
        -> one hypothesis -> patch -> correctness -> benchmark
        -> profile -> compare -> accept/revert -> repeat (max 3)
```

Supported operator implementations:

- Ascend C
- Triton-Ascend
- TileLang-Ascend

## Repository layout

```text
.agents/skills/ascend-operator-optimizer/
├── SKILL.md           # entry point and routing rules
├── assets/            # harness config example
├── references/        # optimization knowledge base
└── scripts/           # benchmark/profile harness

ascend-kernel/  # local AscendC operator workspace and probes
reference/      # curated skills/API references
skill_usage.md
knowledge/       # legacy/source material
harness/         # legacy task/report templates
optimctl         # compatibility helper for the old workflow
```

`_work/` and `asc-devkit/` are local experiment/upstream clones and are ignored.

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

Edit only commands, implementation metadata, and local artifact globs. The benchmark command must write JSON to `$OPT_HARNESS_RESULT`.

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

Analyze an existing Ascend profiler trace directory:

```bash
python .agents/skills/ascend-operator-optimizer/scripts/harness.py \
  profile-analyze --profile-dir .operator-optim/runs/<run-id>/profile
```

## Benchmark Contract

The benchmark command must write a JSON file to `$OPT_HARNESS_RESULT`:

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

## Scope

The compact knowledge base covers the common Ascend performance model, tiling, data movement, memory residency, pipeline overlap, kernel constraints, correctness gates, benchmark comparability, and Ascend C-specific operator notes. Device-specific constants must be obtained from the active environment rather than copied into code.
