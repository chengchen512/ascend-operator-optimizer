# Ascend 910B Hardware Model

## Query, Do Not Freeze

Use `PlatformAscendC` or active environment data for AIC/AIV counts, UB/L1/L0 capacities, workspace, and library workspace. Product variants differ. Resource values observed on one 910B SKU are evidence for that device, not portable constants.

## Compute Roles

- **AIV/Vector**: unpack, cast, elementwise, reduction, layout preprocessing, antiquant, and postprocessing.
- **AIC/Cube**: Matmul/Mmad and Cube-oriented reduction or convolution paths.
- **Scalar**: tiling interpretation, loop control, address arithmetic, masks, and API setup.
- **MTE2/MTE3**: GM/L2/L1/UB/L0 data movement depending on the route.

For MIX kernels, reason about paired physical resources and event traffic, not just nominal block count.

## Memory Roles

| Memory | Primary use | Main risk |
| --- | --- | --- |
| GM/HBM | inputs, outputs, required workspace | bandwidth and unnecessary intermediates |
| L2 | cross-core/cache reuse | split direction and cache thrash |
| L1/TSCM | Cube inputs and explicit reuse | full-load/format/lifecycle constraints |
| L0A/L0B | Cube operand staging | tile shape and layout constraints |
| L0C | Cube accumulation | capacity and premature writeback |
| UB | Vector queues, temporaries, layout conversion | capacity, alignment, bank conflict, event count |
| BT/FP | bias/quant/fixpipe support | API and product support |

## Alignment Model

- Treat 32 bytes as the base DataBlock alignment for local buffers and many copy strides.
- Inspect 256-byte Vector repeat/mask requirements where the selected API demands them.
- Prefer GM starts and partitions friendly to cache-line transactions; verify 512-byte heuristics with profiling.
- Keep semantic lengths separate from padded storage lengths.

## Bottleneck Classes

Classify evidence into launch/scalar overhead, MTE2 input, Vector, Cube, MTE3 output, synchronization gaps, UB conflict/pressure, L2 behavior, or inter-core imbalance. Optimize the longest constrained path, not the most visible source loop.

## Dynamic Queries

Host tiling should query the platform and serialize only the values the kernel needs. Do not branch the kernel on guessed SKU names when a resource or capability check can drive tiling or TilingKey selection.
