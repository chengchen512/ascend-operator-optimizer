# Constraints

## Hard Stops

- Implementation type is unknown.
- Target operator source and tests have not been read.
- No baseline exists but a performance improvement is being claimed.
- Correctness failed but benchmark is being run or reported.
- Baseline and candidate benchmark use different cases.
- Reference/baseline path does not execute on NPU.
- Patch requires unrelated public build/tooling/operator changes.

## General Prohibitions

- Do not hardcode a single shape, core count, UB capacity, or L2 capacity.
- Do not drop dtype/shape/semantic generality.
- Do not introduce CPU-NPU synchronization in the hot path.
- Do not use external Python composition to bypass the target implementation.
- Do not use one benchmark sample as an improvement claim.
- Do not maintain multiple inconsistent benchmark case formats.

## Ascend C Prohibitions

- Do not use `std::min/max/abs/sqrt/exp/log` in kernel hot paths.
- Do not use dynamic allocation in kernel code.
- Do not let FP16/BF16 execute complex math, reductions, or normalization directly; cast to FP32 first.
- Do not pass `repeatTime > 255` to high-dimensional Vector APIs.
- Do not use GM `GetValue()` / `SetValue()` in performance paths.
- Do not use `DataCopy` as the default GM <-> UB production path; prefer `DataCopyPad`.
- Do not mix up `DataCopyPad` units: `blockLen` is bytes, GM stride is bytes, UB stride is 32B DataBlock.
- Do not pass rvalues, temporary objects, literals, or expressions to `EXEC_KERNEL_CMD`.
- Do not use `int32_t` for GM offsets, shape sizes, or products that can exceed 2GB.
- Do not leave `PipeBarrier<PIPE_ALL>()` as a final sync fix.
- Do not change existing TilingID field order or historical semantics.
- Do not skip tail tile handling; separate `tileLength` from `curTileLength`.
- Do not alias `ReduceSum/ReduceMax` destination with `tmpBuffer`.
- Do not call high-dimensional `Compare` on data regions that are not padded to 256B.
- Do not exceed `DataCopyExtParams.blockCount` limits; split large row batches.
- Do not leave class members or stack state uninitialized in kernel paths.
- Do not use AtomicAdd on dirty GM/UB data or forget to restore AtomicNone.
- Do not dereference optional input desc/tensor or allocated output/workspace before null checks.

## Triton-Ascend Prohibitions

- Do not optimize from a single scale only.
- Do not use `tensor.item()` in the hot path.
- Do not ignore UB pressure from offset, mask, or index tensors.
- Do not default to non-contiguous access.
- Do not add branch-heavy loop logic without checking compiled behavior.

## TileLang-Ascend Prohibitions

- Do not write scalar per-element kernels instead of blocked kernels.
- Do not reference function-local variables in `T.Kernel(...)` grid parameters.
- Do not use missing scalar/vector APIs; broadcast scalars first.
- Do not use `T.copy` for broadcasting.
- Do not omit tail bounds checks.

## Acceptance Conditions

A kept round must satisfy:

- build/JIT/compile success
- correctness pass
- same benchmark cases as baseline
- most core cases have no regression
- run artifacts capture commands, results, and decision


