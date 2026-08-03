# Optimization Playbook

Choose candidates from evidence, in this order:

```text
profile evidence -> bottleneck class -> candidate action -> rejection condition
```

## Launch or Scalar Overhead

Evidence: many short tasks, high host/device gap, small useful work per call.

Actions: reduce kernel/public API calls, enlarge useful tile, hoist invariant setup, simplify tails/masks, cache Host legality/tiling work.

Reject when larger tiles break local lifetime, general shapes, or increase tail waste enough to regress protected cases.

## MTE2 or GM Read

Evidence: MTE2 longest, low useful bytes per task, Vector/Cube idle.

Actions: contiguous/stride DMA, larger transfers, better split direction, remove repeated input reads, add double buffering when overlap exists.

Reject on UB overflow, harmful tile shrinkage, alignment bugs, or no measured overlap.

## Vector

Evidence: Vector is longest after movement is healthy.

Actions: select fused/scalar APIs, reduce casts and temporaries, Counter mode, supported multi-row patterns, retain intermediates locally.

Reject on tolerance changes without justification, repeat overflow, or bank/resource regression.

## Cube

Evidence: Cube dominates or many small Matmul calls/atomics surround low-utilization Cube work.

Actions: improve M/N/K tiles, reduce Matmul calls, increase L1/L0 reuse, accumulate in L0C, use supported local input/epilogue paths.

Reject on full-load/format violations, repeat hangs, incorrect local lifetime, or general-shape loss.

## Output and Atomic

Evidence: MTE3/atomic tasks scale with K partitions or partial results.

Actions: local/L0C accumulation, fewer partitions, Fixpipe/epilogue fusion, initialize and write final output once.

Reject if local capacity is exceeded or accumulation precision changes beyond the contract.

## Synchronization

Evidence: internal bubbles, nondeterminism, repeat hangs, excessive event tasks.

Actions: correct queue ownership, narrow dependencies, pipeline independent stages, reduce queue objects/events.

Reject any candidate that only passes once or relies on a broad barrier as the final design.
