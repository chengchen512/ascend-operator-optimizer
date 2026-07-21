# Elementwise Tiling

Assume memory-bound until profiling disproves it.

## Block Level

- Flatten contiguous tensors only when layout and broadcasting semantics allow it.
- Partition contiguous ranges with cache-friendly starts and one explicit tail core.
- Avoid launching more blocks than useful tiles.

## UB Level

- Size tiles for useful DMA bursts and balanced input/output queues.
- Include all broadcast operands and FP32 temporaries in the budget.
- Use aligned capacity for buffer layout and valid length for Vector count and CopyOut.

## Pipeline

Start with one buffer. Enable double buffering only when there are at least two tiles per core, copy and compute can overlap, and the larger UB footprint does not shrink tiles enough to erase the gain.

## Candidate Order

1. Eliminate repeated GM passes.
2. Make accesses contiguous and copies larger.
3. Reduce Scalar loop/mask work.
4. Fuse Vector operations in UB.
5. Add overlap after proving pipeline slack.
