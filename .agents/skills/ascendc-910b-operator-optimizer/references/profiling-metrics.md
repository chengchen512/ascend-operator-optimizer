# Profiling Metrics

## Measurement Contract

Use the same shape, dtype, implementation path, synchronization boundary, warmup, active/repeat count, statistic, and unit for baseline and candidate. Run custom and vendor/reference on device.

Use benchmark results for acceptance. Use `msprof op` or `torch_npu.profiler` evidence to classify bottlenecks.

## Clock Accounting

- `service_us`: first device task to last device task.
- `device_busy_union_us`: union of device task intervals across streams.
- `kernel_sum_us`: sum of kernel durations; may exceed busy union.
- `wait_sum_us`: wait fields, not compute work.
- `underfeed_ratio`: idle portion of service window.
- `largest_internal_bubble_us`: largest gap between merged busy intervals.

## Evidence Map

| Evidence | Likely class | Next checks |
| --- | --- | --- |
| many short kernels/calls | launch/scalar | fuse calls, enlarge tile, reduce public Matmul count |
| MTE2 longest | input movement | copy size, continuity, padding, double buffer |
| Vector longest | SIMD compute | API choice, cast/tmp cost, fusion, repeat setup |
| Cube longest with low utilization | Cube tiling | M/N/K aspect, reuse, L0/L1 occupancy |
| MTE3/atomic heavy | output path | local accumulation, Fixpipe, fewer partial writes |
| busy union much smaller than service | underfeed/sync | host gaps, queue/event waits, launch fragmentation |
| core durations vary | imbalance | block tiling and tail ownership |

## Guardrails

Do not treat wait time as kernel busy time, compare different profiler case sets, normalize inconsistently, select the single fastest sample, or use a profile tag as proof of speedup.
