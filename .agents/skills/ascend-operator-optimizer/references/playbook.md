# Optimization Playbook

## Loop

```text
inspect -> baseline -> diagnose -> one hypothesis -> patch
        -> correctness gate -> benchmark gate -> accept/revert -> repeat (max 3)
```

Correctness comes before performance. Performance claims require comparable
measurements against the same cases and measurement method.

## Inspect Gate

Before modifying code:

- Identify implementation type: `ascend-c`, `triton-ascend`, or `tilelang-ascend`.
- Read target repository instructions, README, build scripts, tests, benchmark entry points, and target operator source.
- Find the correctness command and benchmark command.
- Identify benchmark cases, dtype coverage, shape coverage, reference path, profiler, unit, warmup/repeat/active, statistic, and trace path.

Do not patch before this gate is complete.

## Baseline Gate

Before optimization:

- Run the baseline with a fixed case set.
- Ensure custom operator and reference/baseline both execute on NPU.
- Save raw command logs and structured benchmark JSON under `.operator-optim/`.
- If no baseline exists, do not claim speedup.

## Diagnose Order

1. Algorithm.
2. Tiling/grid.
3. Data movement.
4. UB/local memory.
5. Pipeline overlap.
6. Framework-specific API constraints.

Output candidate issues ordered by expected payoff. Choose one hypothesis per round.

## Patch Round

Rules:

- Small scoped changes only.
- Preserve public semantics, supported dtype/shape, and call interface.
- No single-shape hardcoding.
- No implementation bypass, such as replacing a Triton kernel with external Python composition.
- Update run records with the target, changed files, commands, and observed results.

Stop after three rounds if there is no defensible improvement.

## Correctness Gate

Before benchmarking a patch:

- Run the target correctness command.
- Cover original dtype set and small/medium/large/boundary shapes where the target test suite supports them.
- Ascend C should prefer MERE/MARE reporting when available.
- Triton-Ascend can use `torch.testing.assert_close(..., rtol=1e-3, atol=1e-3)` unless the target repo defines stricter thresholds.

If correctness fails, fix or revert before benchmarking.

## Benchmark Gate

The benchmark after a patch must use the baseline case set and compatible metrics.

Required comparison:

```text
Case | Shape | DType | Mode | Baseline | Candidate | Reference | Candidate/Baseline | Candidate/Reference
```

Conclusions must be based on the table across cases, not on a single fastest result.

## Failure Logs

Record facts only:

- command and exit code
- log paths
- raw excerpts
- case, shape, dtype, mode
- observed symptom
- next check

Keep inference in the conversation, clearly marked as inference. Do not write speculative root cause into run artifacts.
