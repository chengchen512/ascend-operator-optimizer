# Rowwise and Reduction Tiling

## Ownership

Prefer one core to own complete output rows when that avoids cross-core reduction. Split a row only when row length or parallelism requires it and the merge cost is justified.

## Row Tile

- Keep valid columns and aligned columns separate.
- Batch rows only while Vector repeat limits, `blockCount`, UB capacity, and tmpBuffer limits remain valid.
- Split batches when an API field narrows to `uint8_t`/`uint16_t`.

## Precision

Use FP32 accumulation for FP16/BF16 reductions, normalization, variance, and nonlinear composite math unless the operator contract and evidence justify lower precision. Reserve independent destination and reduction tmp buffers.

## Tail

Fill padded lanes with a semantic identity or non-winning extreme. Never allow padding to alter sums, maxima, minima, indices, or normalization denominators.

## Candidate Order

1. Remove redundant row reads.
2. Improve row batching and stride DMA.
3. Use supported pattern/multi-row reductions.
4. Reduce temporary footprint and Scalar repeat setup.
5. Split oversized rows with an explicit merge design.
