# Workflow Contract

## Modes

- **Generation** ends at a correct, benchmarked custom baseline.
- **Optimization** starts from a correct baseline and produces evidence-backed candidate decisions.
- **Debug** repairs environment, build, launch, memory, synchronization, or precision contracts before returning to the active mode.

## Required Sequence

```text
environment -> spec -> classification -> design -> code -> build
            -> precision -> baseline -> profile -> one hypothesis
            -> candidate -> build -> precision -> benchmark -> accept/reject
```

Each arrow is a gate. A failed gate blocks downstream work.

## Generation Artifacts

- normalized `.ascendc-opt/spec.json`
- `design.md` with exact data path and resource budget
- Host tiling and TilingData
- Device kernel and registration/wrapper code
- precision and benchmark case set
- correct custom baseline

## Candidate Contract

Record one optimization dimension per candidate:

```json
{
  "candidate": "candidate-003",
  "parent": "candidate-001",
  "hypothesis": "Increase tile length to reduce MTE2 transaction count",
  "expected_metric": "MTE2 task count decreases without UB overflow",
  "changes": ["tile_length: 4096 -> 8192", "buffer_num: 1 -> 2"],
  "build": "pass",
  "precision": "pass",
  "latency_us": 18.4,
  "baseline_us": 24.7,
  "speedup": 1.342,
  "accepted": true
}
```

Use `changes` for one coherent hypothesis. A double-buffer candidate may change queue count and tile size together only when both are required by that single pipeline hypothesis.

## Acceptance Gate

Accept only when:

- the candidate builds through the real framework path
- all required precision and repeated-run cases pass
- benchmark cases and metadata match the baseline
- the primary objective improves beyond the configured noise threshold
- protected cases have no disallowed regression
- the route respects interface, memory, and forbidden-path constraints

Keep profiler output as diagnostic evidence. Benchmark comparison decides acceptance.

## Failure Records

Persist facts only: command, exit code, environment, case, raw excerpt, observed symptom, and next probe. Keep speculative root-cause language in the working conversation until code and trace evidence agree.
