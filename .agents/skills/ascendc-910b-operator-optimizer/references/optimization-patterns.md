# Optimization Patterns

## UB Fusion

Fuse consecutive Vector transforms while data is resident. Count the saved GM bytes and added UB live range before implementation.

## Double Buffer

Use two storage slots with a proven producer/consumer overlap and at least two iterations. Queue depth need not equal buffer number; follow the exact API contract.

## L1/TSCM Residency

Keep a full configured matrix tile resident when reuse justifies it and format/event constraints are satisfied. Free only after the last Matmul consumer.

## L0C Accumulation

Accumulate K-split partial products before final writeback. Prefer this over repeated GM atomic updates when L0C/API capacity permits.

## VECOUT Local Matmul Input

Produce a supported ND/NZ local tensor, enqueue/dequeue it, pass it through `SetTensorA/B(LocalTensor)`, and retain it until Matmul completes. Benchmark repeat behavior, not only a single call.

## Host Autotiling

Choose routes from shape, dtype, queried resources, and API tiling legality. Cache repeat decisions. Preserve a correct custom fallback without prohibited intermediates.

## Candidate Discipline

Each pattern requires a baseline, resource delta, expected profiler movement, precision gate, benchmark comparison, and explicit reject condition.
