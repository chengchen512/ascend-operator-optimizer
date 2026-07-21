# Precision Validation

## Case Set

Cover tiny, normal, large, exact-alignment, each tail type, extreme values, zeros, negative values, repeated runs, and every supported dtype/layout/route. Include shapes that force the generic Host autotiling fallback.

## Sequence

1. Compare output shape, dtype, layout, and finite-value counts.
2. Locate first mismatch and tile/core boundary.
3. Report max/mean absolute error and the repository's relative metric.
4. Repeat the same input to detect nondeterminism.
5. Isolate one core/tile or intermediate only after preserving a failing real-path case.

## Symptom Routing

| Symptom | First checks |
| --- | --- |
| all zero/constant | output offset, write path, atomic mode |
| first error at tile boundary | valid length, offset, CopyOut pitch |
| only tail fails | padding, aligned versus valid extent |
| single core passes, multicore fails | overlap, omission, workspace partition, atomic init |
| repeated results differ | queue/event lifecycle, uninitialized memory |
| NaN/Inf | domain, overflow, dirty padding, invalid accumulation |
| FP32 passes, FP16/BF16 fails | accumulation dtype and cast round mode |

Do not benchmark a failing candidate. Record and reject it or repair the same hypothesis before measuring performance.
