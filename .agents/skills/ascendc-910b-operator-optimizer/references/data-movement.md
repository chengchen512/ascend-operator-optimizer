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

## Synchronization

Copies are asynchronous. Express dependencies with queues/events:

```text
producer: AllocTensor -> copy -> EnQue
consumer: DeQue -> compute/copy -> FreeTensor
```

Use a broad barrier only as a temporary diagnosis. Replace it with the narrow documented dependency before accepting the candidate.
