# Pipeline Scheduling

## Objective

Overlap independent MTE, Vector, and Cube work while preserving data lifetime. Buffer count alone does not create a pipeline.

## Queue Lifecycle

Every queue-managed tensor must complete:

```text
AllocTensor -> produce -> EnQue -> DeQue -> consume -> FreeTensor
```

Use queue depth for the maximum number of consecutive enqueues without dequeue. Treat buffer number as storage for overlap; depth 1 can still use double buffering when supported.

## Double Buffer Gate

Enable only when:

- each core has at least two useful iterations
- producer and consumer use independently schedulable pipelines
- profiling shows overlap opportunity
- the doubled live set fits without harmful tile shrinkage
- queue/event and Matmul-object limits remain valid

Reject when small shapes finish in one tile, one pipeline dominates completely, or local memory pressure causes smaller/slower tiles.

## Mixed AIV/AIC

Specify which engine owns allocation, transformation, enqueue/dequeue, Matmul setup, and release. Verify producer completion before Cube reads and Cube completion before reuse/free. Single-run success is insufficient; run repeated correctness and benchmark loops to expose stale-event and lifetime bugs.

## Direct Mmad Event Chain

For manually managed A1/B1/C1 and L0 tensors, preserve the narrow engine chain:

```text
GM -> L1 DataCopy       Set MTE2_MTE1
L1 -> L0 LoadData       Wait MTE2_MTE1; Set MTE1_M
Mmad                    Wait A/B MTE1_M; Set M_FIX
Fixpipe CO1 -> GM       Wait M_FIX
```

Use separate event IDs for independently moving A, B, and bias. Reuse an L1 or L0 region only after its last asynchronous consumer completes. A final `PipeBarrier<PIPE_ALL>()` does not repair a missing stage dependency and is not a substitute for these events.

## Diagnosis

- Nondeterministic results: missing dependency or uninitialized data.
- Single pass works, repeat hangs: event exhaustion, incomplete lifecycle, or local tensor reuse.
- Device gaps: host underfeed, excessive public API calls, or synchronization boundaries.
- No speedup from double buffer: no overlap opportunity or tile shrinkage offset the gain.
