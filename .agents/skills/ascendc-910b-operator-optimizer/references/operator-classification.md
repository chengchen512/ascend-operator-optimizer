# Operator Classification

Classify before selecting templates or tiling. Fused operators may have a dominant class plus a preprocessing/postprocessing class.

| Class | Signals | Primary design axis |
| --- | --- | --- |
| Elementwise/pointwise | independent output elements | GM bandwidth, contiguous tiles, launch overhead |
| Rowwise/reduction | rows reduced or normalized | row ownership, FP32 workspace, repeat limits |
| Index/data movement | gather, scatter, transpose, sort staging | access locality, bounds, aligned layout |
| Cube/Matmul | dense block products | M/N/K tiles, L1/L0 residency, Matmul calls |
| Mixed/fused | AIV preprocessing feeds AIC or AIC output feeds AIV | local handoff, queues/events, synchronization |

## Decision Questions

1. Which output dimensions are independent across cores?
2. Which input dimension carries reuse?
3. Is arithmetic intensity high enough to justify Cube?
4. Does Vector preprocessing reduce GM traffic, or merely move work before Cube?
5. Can an intermediate remain in UB/L1/L0, or must it be materialized?
6. Which tails alter API, layout, or precision behavior?

## Design Output

State the selected class, rejected classes, expected bottleneck, Host tiling fields, Device queues/buffers, and fallback route in `design.md`. A template is usable only when its manifest constraints match the operator.
