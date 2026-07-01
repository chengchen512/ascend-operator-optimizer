# awq_w4a16_linear performance report

Current default route is AscendC vector dequant (`int4b_t -> half`) with a versioned dequant-weight cache, followed by generic NPU `at::mm`. Set `AWQ_DISABLE_DEQ_CACHE=1` to measure the cache-miss two-stage route.

| Shape | Route | Correctness | Candidate us | Vendor us | Ratio | Warmup | Repeat |
|---|---|---|---:|---:|---:|---:|---:|
| `[2,256,256]` | cache enabled | pass | 7.339999824762344 | 8.449999988079071 | 0.8686390337416957 | 5 | 20 |
| `[2,256,256]` | cache disabled | pass | 11.35300025343895 | 8.60000029206276 | 1.3201162637072268 | 5 | 20 |
| `[16,4096,22016]` | cache enabled | pass | 80.7804183959961 | 87.57350158691406 | 0.9224299238031948 | 100 | 1000 |
| `[16,4096,22016]` | cache disabled | pass | 252.05120086669922 | 87.62419700622559 | 2.8765022616845357 | 20 | 100 |

Cache invalidation probe passed after `scale.add_(0.01)`: max abs vs new vendor reference remained `0.00390625`, while output changed by `2.625` vs the old cached output.

Remaining cache-miss gap is caused by materializing `half[K,N]` in GM before matmul. First-call performance still needs fused antiquant with Cube/MMAD.