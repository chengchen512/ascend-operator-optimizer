# Index and Data-Movement Tiling

## Correctness First

Validate index bounds, signedness, axis normalization, duplicates, collision semantics, and output initialization before optimizing traffic.

## Partitioning

- Partition by contiguous output ownership when possible.
- For gather, group or tile indices to improve source locality only when output order remains correct.
- For scatter, define collision handling explicitly; do not introduce races by spreading colliding updates across cores.
- For transpose/layout conversion, choose tiles that satisfy both source and destination stride/alignment constraints.

## Local Budget

Include index tensors, address/offset vectors, masks, staging tiles, and sort/reorder temporaries. Index bookkeeping can consume more UB than payload for small dtypes.

## Candidate Order

1. Replace per-element GM access with block/stride movement.
2. Improve locality and coalescing.
3. Reduce index/mask materialization.
4. Fuse layout conversion into a required upstream/downstream copy.
5. Add overlap only after collision and ownership semantics are proven.
