# Cube and Matmul Tiling

## Tile Axes

Choose M/N/K tiles from queried L1/L0A/L0B/L0C resources, supported Cube formats, operand reuse, and output accumulation strategy. Minimize public Matmul call count without exceeding local capacity or violating LocalTensor lifetime.

## Residency

- Keep the smaller or more reused operand resident in L1/TSCM when full-load and format constraints are satisfied.
- Accumulate K partitions in L0C where the API permits; avoid GM writeback plus AtomicAdd for every small K tile.
- Use BT/FP/Fixpipe fusion only when supported by the exact CANN 9.0.0 product path.

## Direct Mmad Geometry

Separate logical matrix extents, packed GM bytes, and aligned L1/L0 extents. For the supplied CANN `Mmad` example, the starting Cube geometry was:

| Input dtype | C0 elements | Fractals per 16-element edge |
| --- | ---: | ---: |
| `int4b_t` | 64 | 4 |
| `int8_t` | 32 | 2 |
| `bfloat16_t` | 16 | 1 |
| `float` | 8 | 2 |

Treat this as a same-version design pattern, not a portable hardware constant. Confirm the local CANN 9.0.0 type/layout contract before using it. Derive A1/B1 and A2/B2 sizes independently from transpose choice and required NZ/ZZ/ZN layout; derive CO1 from `align(M,16) * align(N,16)`. Keep `MmadParams.m/n/k` at valid logical sizes except where the exact dtype/layout contract requires an aligned parameter.

## Vector-to-Cube Handoff

For AIV preprocessing feeding Cube, specify all stages:

```text
GM -> VECIN/VECCALC -> VECOUT or TSCM -> Matmul/Mmad -> L0C -> final output
```

Record queue type/depth/buffer count, allocation size, `TPosition`, ND/NZ format, event ownership, producer/consumer order, and release point. `SetTensorB(LocalTensor)` support alone does not prove repeat-safe handoff.

## TSCM

Treat TSCM input as full-load: the entire configured single matrix tile must coexist in L1/TSCM. Follow `AllocTensor -> copy/transform -> EnQue -> DeQue -> Matmul -> FreeTensor`. Verify required NZ/ND format and TSCM event limits.

## Candidate Order

1. Reduce Matmul/Mmad invocation count.
2. Remove GM intermediates and repeated atomic accumulation.
3. Improve L1/L0 reuse and tile aspect ratio.
4. Fuse Vector preprocessing or epilogue through a supported local path.
5. Overlap producer/consumer work with explicit event evidence.
