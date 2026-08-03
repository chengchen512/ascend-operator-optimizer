# Data Movement

## Copy Contract

Prefer `DataCopyPad` for non-trivial GM/local movement. Verify the selected overload and parameter units:

| Field | Typical unit |
| --- | --- |
| `blockLen` | bytes |
| GM stride | bytes |
| UB/local stride | 32-byte DataBlocks where documented |
| `GlobalTensor[index]` | elements |

Use `DataCopy` only when the alignment and supported path are proven.

## Throughput

- Combine contiguous rows into one stride DMA instead of a kernel loop of small copies.
- Increase useful transfer size before adding concurrency; use profiling rather than a fixed threshold as proof.
- Partition GM so core starts and row pitches remain transaction-friendly.
- Keep chained intermediates in UB/L1/L0 when semantics and lifetime allow.

## ND/NZ and Local Paths

When a copy performs layout conversion, record source/destination position, format, dtype, shape, padded extent, and product support. Confirm whether conversion is performed by `DataCopyPad`, an ISASI overload, Matmul internals, or an explicit transform.

For a direct `Mmad` path on the supplied A2/A3-family branch, use the layout chain as an explicit design object:

```text
GM ND -> A1/B1 with Nd2NzParams -> A2/B2 with LoadData or
LoadDataWithTranspose -> Mmad -> CO1 -> Fixpipe row-major GM
```

Do not infer `LoadData` direction from helper names such as `SplitB`; record the actual source and destination formats. Handle a non-aligned transformed axis with a separate tail conversion whose GM offset uses valid elements and whose local offset uses the aligned destination pitch.

For `int4b_t`, distinguish logical elements from physical transfer bytes. The example keeps logical `K` for Cube geometry and `MmadParams.k`, but halves the packed GM D extent (`K/2`) in the `Nd2NzParams` copy. Validate the exact overload before applying this rule; never halve logical loop bounds, MMAD K, and storage counts indiscriminately.

When bias GM length is not directly copy-aligned and `DataCopyPad` does not support the GM-to-C1 path, use a supported ND-to-NZ conversion to create an aligned C1 representation rather than reading beyond valid GM data.

## Synchronization

Copies are asynchronous. Express dependencies with queues/events:

```text
producer: AllocTensor -> copy -> EnQue
consumer: DeQue -> compute/copy -> FreeTensor
```

Use a broad barrier only as a temporary diagnosis. Replace it with the narrow documented dependency before accepting the candidate.
