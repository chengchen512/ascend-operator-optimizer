# Numerical Precision

## Accumulation

Use FP32 intermediates for FP16/BF16 reductions, normalization, variance, transcendental composites, and long accumulations unless the operator contract and measured error justify another route. Pure copies and index payloads do not require numeric upcast.

## Cast

Record source type, destination type, round mode, saturation behavior, and where rounding occurs. Moving a cast across a reduction or multiply changes numerical semantics even when algebra looks equivalent.

## Padding

Choose padding values by operation:

- sum: zero
- max/argmax: non-winning low extreme
- min/argmin: non-winning high extreme
- product: one when padded lanes participate
- normalization: exclude padded lanes from denominator and statistics

## Quantization

Record signed nibble interpretation, packing order, group axis/size, zero-point convention, scale/offset dtype, accumulation dtype, and epilogue rounding. A faster unpack route is invalid when it changes any of these.

## Error Policy

Prefer the target repository's metric and threshold. Otherwise record max/mean absolute error, relative metrics with a near-zero policy, NaN/Inf counts, and mismatch location. Do not relax tolerance solely to accept a performance candidate; justify any change from model-level accuracy requirements or vendor/reference behavior.
