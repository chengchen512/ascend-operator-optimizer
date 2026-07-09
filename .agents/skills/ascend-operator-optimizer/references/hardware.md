# Hardware Model

## Common Path

- Main data path: GM/HBM -> local storage -> Vector/Cube -> local storage -> GM/HBM.
- Typical units: Scalar, Vector, Cube, MTE/DMA, L2, L1, L0A, L0B, L0C, UB, BT Buffer, FP Buffer.
- Capacity and core counts are device dependent. Query the active platform or profiling logs; do not copy constants into kernel code.
- Most operator bottlenecks show up as one of: Scalar overhead, MTE bandwidth/latency, Vector time, Cube time, or synchronization gaps.

## Optimization Order

1. Algorithm: remove repeated GM reads/writes, avoid multi-pass work when one pass is possible, avoid per-element GM access.
2. Tiling/grid: map logical work to physical AIV/AIC cores, keep work balanced, handle tails explicitly.
3. Data movement: contiguous access, large enough DMA bursts, correct mask/stride/padding.
4. UB/local memory: account for inputs, outputs, temporaries, FP32 buffers, index/mask/offset tensors, and double buffering.
5. Pipeline: overlap CopyIn, Compute, and CopyOut through TQue/queues where there is enough loop count to benefit.

## Ascend C Signals

- Prefer AIV core count for pure Vector kernels, AIC core count for pure Cube kernels, and physical core groups for MIX kernels.
- Plan GM offsets around 512B boundaries where possible; UB buffers around 32B boundaries.
- Single DMA transfers should generally be at least 16KB to approach useful bandwidth.
- Keep chained Vector intermediates in UB instead of round-tripping through GM.

## Framework Signals

- Triton-Ascend: watch BLOCK/grid choices, contiguous masks, UB pressure from offsets/masks/index tensors, and host synchronizations.
- TileLang-Ascend: avoid elementwise scalar-style kernels; check `T.copy`, `T.alloc_shared`, `T.Pipelined`, and NPU API differences.
