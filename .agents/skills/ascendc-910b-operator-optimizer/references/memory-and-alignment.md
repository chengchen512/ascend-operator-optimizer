# Memory and Alignment

## Budget Table

Every `design.md` must include a table with position, owner, dtype, logical shape, aligned bytes, buffer count, lifetime, and reuse. Sum by UB, L1/TSCM, L0A/B/C, workspace, BT, and FP resources.

## UB Rules

Budget input/output queues, `TBuf` temporaries, upcast storage, masks/index/offset tensors, API tmp, and every double-buffer copy. Prefer offset-based subviews inside one safely sized allocation when event or queue-object limits are tighter than raw bytes.

## Alignment Rules

- Align local allocations and pitches to required DataBlock boundaries.
- Keep valid elements separate from aligned elements.
- Apply API-specific 256-byte mask/repeat requirements only where documented.
- Use bytes for copy block lengths and documented stride units; use elements for `GlobalTensor` indexing.

For direct Cube code, budget A1, B1, and C1 in their shared L1 address space with explicit non-overlapping byte offsets. A2, B2, C2/BT, and CO1 are distinct physical positions, so offset zero may be valid independently in each position; never interpret identical local offsets as aliasing without checking `TPosition`.

In the supplied A2/A3-family CANN example, C1-to-C2 bias `blockLen` uses a 64-byte unit and the C1 allocation is rounded to `align(N * sizeof(T), 64)`. Keep this product-specific rule separate from ordinary 32-byte DataBlock rules and verify the active 910B header/overload before coding it.

## Lifetime

Model the peak set of simultaneously live tensors, not the sum of sequential phases. A buffer can be reused only after the final asynchronous consumer is complete and its queue/event lifecycle has closed.

## GM/Workspace

Use workspace for required framework/library state or true cross-core/global staging. Do not materialize an intermediate in GM when the optimization objective relies on quantized or fused bandwidth savings.

## 507015 Checklist

Check exact allocation size, typed element count, aligned pitch, queue depth/buffer number, LocalTensor subview bounds, full-load matrix size, format conversion destination, and FreeTensor timing before changing synchronization globally.
