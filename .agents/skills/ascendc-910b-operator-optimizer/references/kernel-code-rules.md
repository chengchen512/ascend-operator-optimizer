# Kernel Code Rules

## Structure

- Keep `TPipe` at kernel-entry scope where practical and pass it to the implementation object.
- Initialize every member and state variable.
- Use `TQue` for synchronized producer/consumer storage and `TBuf` for scratch.
- Pair every allocation, queue transition, atomic mode, Matmul lifecycle, and event operation.

## Hot Path

- No dynamic allocation, STL containers, host math, or per-element `GlobalTensor::GetValue/SetValue`.
- Reduce Scalar address/mask/setup work inside the innermost tile loop.
- Split Vector API repetitions before narrow repeat fields overflow.
- Keep destination and API tmp buffers disjoint when required.

## Lengths and Offsets

Use 64-bit GM offsets and checked API parameters. Use valid lengths for semantic compute and final output; use aligned lengths only for storage, pitch, and APIs that explicitly consume padded extents.

## Synchronization

Do not rely on API return as completion for asynchronous copy/compute. Use documented queue/event dependencies. Remove diagnostic `PipeBarrier<PIPE_ALL>()` before accepting a candidate unless it is specifically required.

## Atomic and Accumulation

Initialize accumulation destinations, ensure padded source lanes are clean, enter the correct atomic mode, and restore `AtomicNone`. Prefer local/L0C accumulation over repeated GM atomics when capacity and API semantics allow.

## Generality

Handle block and tile tails explicitly. Do not embed one benchmark shape in loop bounds, queue sizes, or implementation selection without a correct general route.
