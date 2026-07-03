# codex-ascend-operator-optimizer

A single Codex Skill for measured Ascend operator performance optimization.

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
├── SKILL.md
├── assets/harness.example.json
├── references/
│   ├── INDEX.md
│   ├── hardware.md
│   ├── playbook.md
│   ├── constraints.md
│   ├── harness.md
│   ├── api-lookup.md
│   ├── ascendc/
│   │   ├── workflow.md
│   │   ├── tiling-grid.md
│   │   ├── data-copy.md
│   │   ├── api-usage.md
│   │   ├── memory.md
│   │   ├── pipeline.md
│   │   ├── precision.md
│   │   ├── profiling.md
│   │   └── launch-profiles.md
│   ├── cases/
│   │   └── awq-w4a16-ascendc.md
│   ├── ascendc.md
│   ├── ascendc-examples.md
│   └── sources.md
└── scripts/
    ├── harness.py
    └── api_lookup.py

knowledge/   # legacy/source material, not required by the copied Skill
reference/   # archived source skill contents used by the AWQ W4A16 optimization
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

Search the optional offline Ascend C API snapshot without loading all API files:

```bash
python .agents/skills/ascend-operator-optimizer/scripts/api_lookup.py DataCopyPad
python .agents/skills/ascend-operator-optimizer/scripts/api_lookup.py TSCM Matmul
```

## Benchmark result contract

The benchmark command receives these environment variables:

- `OPT_HARNESS_PHASE`: `baseline` or `evaluate`
- `OPT_HARNESS_LABEL`: stable run label
- `OPT_HARNESS_ROOT`: repository root from the config file
- `OPT_HARNESS_RUN_DIR`: artifact directory for this run
- `OPT_HARNESS_RESULT`: JSON file path the benchmark command must write
- `OPT_HARNESS_PROBE_RESULT`: JSON file path the launch probe command should write
- `OPT_HARNESS_PROFILE_DIR`: directory for profiler traces
- `OPT_HARNESS_PROFILE_RESULT`: optional JSON file path for profile command output
- `OPT_HARNESS_IMPLEMENTATION_TYPE`: implementation type from config
- `OPT_HARNESS_LAUNCH_PROFILE`: launch profile from config
- `OPT_HARNESS_OP_NAME`: operator name from config

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

## AscendC launch profiles

`operator-optim.json` can declare:

```json
{
  "implementation": {
    "type": "ascendc",
    "launch_profile": "ascendc-msopgen-aclnn-dynamic",
    "op_name": "OperatorName"
  }
}
```

For `ascendc-msopgen-aclnn-dynamic`, `commands.launch_probe` is required by default. The probe should exercise the real launch path and write coverage JSON to `$OPT_HARNESS_PROBE_RESULT`.

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

## Integrated AWQ W4A16 experience

This branch adds a concrete AscendC AWQ W4A16 optimization case:

- `.agents/skills/ascend-operator-optimizer/references/cases/awq-w4a16-ascendc.md`: distilled no-cache AWQ, workspace-aware wrapper, pack8 unpack, TSCM/Cube/MMAD lessons.
- `knowledge/awq-w4a16-ascendc-experience.md`: full Chinese experience report.
- `knowledge/awq-skill-usage.md`: skill usage, remote paths, build/test/performance results.
- `knowledge/awq-w4a16/`: copied design and benchmark reports.
- `reference/skills/`: archived contents of the skills actually used in the AWQ work.
- `reference/asc-devkit-api-9.0.0/docs/api/`: offline snapshot of the Ascend C API docs from GitCode `cann/asc-devkit` branch `9.0.0`.
- `.agents/skills/ascend-operator-optimizer/references/api-lookup.md`: lookup protocol that tells agents how to query the API snapshot efficiently.

The main reusable constraint is: when quantization is meant to save HBM and bandwidth, do not treat a persistent dequantized half-weight cache as an acceptable final optimization.

