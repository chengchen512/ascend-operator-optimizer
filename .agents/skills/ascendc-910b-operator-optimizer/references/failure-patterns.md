# Failure Patterns

## Compile Failure

Observed: overload/template/namespace errors or unsupported instruction.

Check: exact local header, dtype, `TPosition`, format, product guard, compiler flags, and minimal probe. Do not patch around an unavailable API with an unverified cast.

## 507015 / MPU Address Invalid

Observed: `Synchronize stream failed`, error code `507015`, plog MPU/fixp address error.

Check in order:

1. `InitBuffer` bytes versus typed element count.
2. LocalTensor offsets, aligned row pitch, and API-required extent.
3. queue depth, buffer number, allocation/free lifecycle, and event ownership.
4. copy block/stride units and destination position.
5. TSCM full-load size, ND/NZ requirement, and Matmul single tile.
6. L0/L1/UB capacity and overlap between manually partitioned buffers.

Use print/probe and plog to locate the failing stage. A broad barrier may isolate synchronization but is not a production fix.

## Single Call Passes, Repeat Hangs

Likely classes: leaked/incomplete event lifecycle, LocalTensor freed or reused before Matmul completion, queue imbalance, or excessive route runtime. Run a bounded repeat probe and inspect device activity before declaring deadlock.

## Short Benchmark Fast, Full Repeat Times Out

Reject the candidate. Treat it as unstable until correctness and the fixed repeat protocol both complete. Do not publish the short-run number.

## First Error at Boundary

Check block/tile offset equations, valid versus aligned length, GlobalTensor element indexing, CopyOut pitch, and padding. Reproduce the first failing boundary with one neighboring tile on each side.

## Multicore Only Failure

Check exact coverage, block index semantics, output/workspace partition, atomics, and cross-core collisions. A single-core workaround is diagnostic only.

## Performance Regression

First prove comparable metadata. Then inspect tile shrinkage, additional public API calls, new synchronization, extra casts/temp buffers, smaller DMA, and tail behavior. Revert rejected candidates to the last known-good snapshot.
