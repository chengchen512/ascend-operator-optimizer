# General Tiling

## Host Responsibilities

Host tiling must validate shape/dtype/layout, query platform resources, choose an implementation key, calculate block and local tiles, serialize TilingData, set blockDim, and request user/library workspace.

Use 64-bit arithmetic for shapes, offsets, products, and byte counts. Reject overflow before narrowing API fields.

## Block Tiling

- Map independent work to AIV for pure Vector and AIC for pure Cube; model physical grouping for MIX kernels.
- Use fewer cores for small work when launch and tail cost would dominate.
- Prove exact coverage with no overlap: former/tail equations must equal the semantic workload.
- Preserve contiguous GM ranges where possible; do not trade one balanced tail for many strided micro-copies.

## Local Tiling

Calculate from an explicit memory budget:

```text
input queues + output queues + temporaries + FP32 workspace
+ masks/index/offset + API tmp + double-buffer copies + safety/alignment
<= queried local capacity
```

Keep these values distinct:

- tile capacity
- current valid length
- aligned storage length
- per-row pitch
- copy block length and stride

## TilingKey

Use TilingKey for materially different implementation routes: aligned/general, Vector/Cube, full/tail, or local/global source. Do not create a key for minor numeric constants that TilingData can express.

Preserve historical TilingData field order. Append fields and guard new semantics with a key/version when compatibility matters.

## Verification

Test tiny, exact-tile, one-element-tail, block-tail, large, and overflow-adjacent shapes. Validate Host tiling independently before interpreting kernel output.
