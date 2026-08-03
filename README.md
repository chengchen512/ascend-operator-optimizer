# ascendc-910b-operator-optimizer

A single Codex Skill for generating, validating, profiling, debugging, and iteratively optimizing project-based Ascend C operators on Ascend 910B with CANN 9.0.0.

## Structure

```text
.agents/skills/ascendc-910b-operator-optimizer/
├── SKILL.md                 # workflow, routing, and hard gates
├── agents/openai.yaml       # Codex UI metadata
├── references/              # flat, selectively loaded knowledge
├── assets/                  # spec, cases, report, scaffold, templates
└── scripts/harness.py       # deterministic execution and run records

tests/
├── skill-cases/             # routing/behavior contract cases
└── test_skill_contract.py   # layout and harness tests
```

The Skill deliberately does not split design, code generation, debug, evaluation, and optimization into separate Skills. One entry point keeps the target contract and candidate history coherent; detailed knowledge is loaded through `references/INDEX.md` only when needed.

## Target

- Ascend 910B hardware family
- CANN 9.0.0
- Ascend C Host tiling and Device kernel development
- project-based custom operators

Local CANN headers, compiler behavior, and on-device evidence take priority over static notes.

## Use

Invoke:

```text
$ascendc-910b-operator-optimizer generate and optimize this Ascend C operator
```

Initialize deterministic state in an operator workspace:

```bash
python .agents/skills/ascendc-910b-operator-optimizer/scripts/harness.py \
  init \
  --workspace /path/to/operator \
  --spec .agents/skills/ascendc-910b-operator-optimizer/assets/operator-spec.example.yaml \
  --cases .agents/skills/ascendc-910b-operator-optimizer/assets/benchmark-cases.example.jsonl
```

The target workspace receives:

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

Establish and evaluate candidates:

```bash
python .agents/skills/ascendc-910b-operator-optimizer/scripts/harness.py \
  baseline --config /path/to/operator/.ascendc-opt/spec.json

python .agents/skills/ascendc-910b-operator-optimizer/scripts/harness.py \
  evaluate \
  --config /path/to/operator/.ascendc-opt/spec.json \
  --label candidate-001 \
  --hypothesis "increase tile size to reduce MTE2 task count" \
  --change "tile_length: 4096 -> 8192"
```

The benchmark command writes structured JSON to `$ASCENDC_OPT_RESULT`. Baseline and candidate case identities, units, and configured measurement metadata must match.

## Validate

```bash
python .agents/skills/ascendc-910b-operator-optimizer/scripts/harness.py doctor
python -m unittest discover -s tests -p "test_*.py" -v
python C:/Users/CHENG/.codex/skills/.system/skill-creator/scripts/quick_validate.py \
  .agents/skills/ascendc-910b-operator-optimizer
```

Operator experiments may live elsewhere in this repository, but the Skill itself is self-contained and never stores mutable run state inside its own directory.
