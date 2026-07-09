# awq_w4a16_linear fresh remote test report

## Environment

- Remote workspace: `/root/awq_w4a16_fresh_skill_20260703`
- Device: 910B2C
- CANN: `/usr/local/Ascend/cann-9.0.0`
- SOC: `Ascend910_9382`
- Python: `/root/ascend-operator-optim/.venv`

## Build

`bash build.sh Ascend910_9382` succeeded and produced:

```text
output/ascend_kernel-2026.3.9-cp311-cp311-linux_x86_64.whl
```

## Correctness

| Case | Route | Tolerance | Result |
| --- | --- | --- | --- |
| `[2,256,256]` | stream | `rtol=1e-2, atol=2e-2` | pass, `max_abs=0.0048828125`, `mean_abs=0.0008409023284912109` |
| `[2,256,256]` | scalar | `rtol=1e-2, atol=2e-2` | pass, `max_abs=0.00390625`, `mean_abs=0.0007225815206766129` |
| `[16,4096,22016]` | stream | `rtol=5e-2, atol=5e-2` | pass, `max_abs=0.0234375`, `mean_abs=0.003105091629549861` |

## Performance

| Case | Route / tile | Candidate us | Vendor us | Ratio |
| --- | ---: | ---: | ---: | ---: |
| `[2,256,256]` | stream | `18.1554` | `7.9832` | `2.27x` |
| `[2,256,256]` | scalar | `1936.17` | `7.9620` | `243.18x` |
| `[16,4096,22016]` | stream default 12288 | `189.7602` | `87.3658` | `2.17x` |

Tile sweep on `[16,4096,22016]`:

| `AWQ_STREAM_N_TILE` | Candidate us | Vendor us | Ratio |
| ---: | ---: | ---: | ---: |
| `4096` | `289.4320` | `88.7664` | `3.26x` |
| `8192` | `224.9380` | `89.1820` | `2.52x` |
| `12288` | `189.7168` | `88.2316` | `2.15x` |
| `16384` | `200.6744` | `87.6700` | `2.29x` |
| `22016` | `230.4136` | `88.3488` | `2.61x` |

## Conclusion

The fresh implementation reached a correct no-cache stream baseline. It is much faster than the scalar-pack diagnostic route but still about `2.17x` slower than vendor because the dequantized half tile is written to and read from HBM before matmul. The next useful optimization is a fused tile-level unpack/antiquant path feeding TSCM/Cube/MMAD without a GM dequant tile.

## 2026-07-04 official API route review

Official references checked:

- CANN 9.0.0 Matmul usage: A/B matrices can use `GM`, `VECOUT`, or `TSCM` on the relevant A2/A3 product line; TSCM input format is constrained to `CubeFormat::NZ`, and local inputs must fit the single-core computation.
- CANN 9.0.0 `SetTensorB`: the `LocalTensor` overload supports `TSCM/VECOUT` B operands and skips the GM-to-L1 copy when TSCM is already set.
- `asc-devkit` 9.0.0 `matmul_int4`: public int4 Matmul requires both A and B to be `int4b_t`, so it does not implement W4A16.
- `asc-devkit` 9.0.0 `matmul_s4_950`: the closest official example to vendor behavior is a mixed AIV/AIC flow, but the current PyTorch workspace wrapper probes have not yet made this handoff correct.

Additional optimization tried:

| Case | Route | Candidate us | Vendor us | Ratio | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| `[2,256,256]` | split-offset | `35.7520` | `8.0592` | `4.44x` | reject |
| `[16,4096,22016]` | split-offset | `234.3387` | `87.9427` | `2.66x` | reject |

The split-offset route computes `(q + offset) * scale` as `q * scale` plus a host-side `mm(sum_group(x), offset * scale)`. It passed correctness but increased total latency, so the default implementation was restored to stream dequant.

Restored default verification:

| Case | Route | Tolerance | Result |
| --- | --- | --- | --- |
| `[2,256,256]` | default stream | `rtol=1e-2, atol=2e-2` | pass, `max_abs=0.0048828125`, `mean_abs=0.0008409023284912109`, candidate `18.1350 us`, vendor `8.4440 us`, ratio `2.15x` |
| `[16,4096,22016]` | default stream | `rtol=5e-2, atol=5e-2` | pass, `max_abs=0.0234375`, `mean_abs=0.003105091629549861`, candidate `189.7180 us`, vendor `89.4527 us`, ratio `2.12x` |

Historical route decision at this point: keep no-cache stream dequant as the
stable checked-in path. This was later superseded by the no-GM-dequant VECOUT
default route below. Public int4 Matmul and the split-offset algebraic path are
not the right mainline for this W4A16 operator.

## 2026-07-04 no-GM-dequant VECOUT default route

Hard requirement:

- The optimized operator must not write the antiquantized B intermediate back to
  GM. Writing that half tile to HBM defeats the bandwidth saving of W4A16.

Implementation promoted to default:

- `awq_w4a16_linear_vecout` is a workspace-aware kernel.
- The host wrapper keeps the Python interface unchanged and launches the
  workspace kernel with parameters `..., m, k, n, workspace, tiling`.
- For each N tile (`N_TILE=256`) and K group (`K_GROUP=128`), the kernel copies
  int4 weights to local memory, casts to half, applies `(q + offset) * scale` in
  local Vector memory, enqueues the result as `TPosition::VECOUT`, and passes it
  to Cube through `SetTensorB(LocalTensor)`.
- K groups accumulate into the final output by `IterateAll(...,
  enAtomic=kg!=0)`. This writes only the final C tile to GM; the dequantized B
  tile never goes to GM.
- The previous stream implementation remains available only with
  `AWQ_ASCENDC_ROUTE=stream` for regression comparison.

Remote verification on `/root/awq_w4a16_fresh_skill_20260703/ascend-kernel`:

| Case | Route | Tolerance | Result |
| --- | --- | --- | --- |
| `[16,4096,22016]` | default VECOUT/Cube | `rtol=8e-2, atol=8e-2` | pass, `max_abs=0.0625`, `mean_abs=0.005369903519749641` |

Wall-clock timing uses `perf_counter + torch.npu.synchronize()` because NPU
Event timing has hung on this route:

| Case | Route | Avg us | Min us | Max us | Repeat |
| --- | --- | ---: | ---: | ---: | ---: |
| `[16,4096,22016]` | vendor WQMM | `154.924967` | `120.505923` | `213.849009` | 3 |
| `[16,4096,22016]` | default VECOUT/Cube | `466.850004` | `410.337001` | `567.427021` | 3 |

Current ratio is about `3.01x` vendor on short wall-clock timing. This is slower
than the old stream baseline but is the first mainline route that satisfies the
no-GM-dequant bandwidth constraint.

Rejected or reverted follow-ups in this pass:

- `TSCM/NZ` fused path: still timed out or failed in earlier stage probes.
- `N_TILE=512`: either `GetTiling` failed or runtime hit `507015` with UB
  address out-of-bounds.
- L0C partial-sum accumulation with `Iterate(enPartialSum)` and `GetTensorC`:
  produced incorrect values and was not a missing-finalize issue.
- Hoisting `SetOrgShape/SetSingleShape` outside the K-group loop compiled and
  passed single correctness, but repeated candidate timing timed out; it was
  reverted.

## 2026-07-04 VECOUT/Cube K-merge optimization

Goal:

- Continue optimizing under the hard requirement that the antiquantized B tile
  must not be written back to GM.

Accepted implementation:

- Default tile changed from `K_GROUP=128, N_TILE=256` to
  `MATMUL_K=256, N_TILE=128`.
- The VECOUT B tile size stays constant at `32768` half elements, about 64KB.
- Two 128-K quant groups are unpacked/antiquantized into one VECOUT tile, then
  consumed through `Matmul.SetTensorB(LocalTensor)`.
- This reduces final output atomic accumulation traffic while preserving the
  no-GM-dequant dataflow.

Final verification:

| Case | Route | Tolerance | Result |
| --- | --- | --- | --- |
| `[16,4096,22016]` | default VECOUT/Cube K256/N128 | `rtol=8e-2, atol=8e-2` | pass, `max_abs=0.046875`, `mean_abs=0.004495220258831978` |

Final wall-clock timing, `perf_counter + torch.npu.synchronize()`,
`warmup=2`, `repeat=10`:

| Case | Route | Avg us | Min us | Max us | Repeat |
| --- | --- | ---: | ---: | ---: | ---: |
| `[16,4096,22016]` | vendor WQMM | `129.779603` | `122.887082` | `157.117029` | 10 |
| `[16,4096,22016]` | default VECOUT/Cube K256/N128 | `371.544517` | `354.437972` | `421.473058` | 10 |

Current ratio is about `2.86x vendor`. This is a clear improvement over the
previous VECOUT/Cube K128/N256 short timing level (`466.850004 us`, repeat=3)
while still avoiding the GM half-tile intermediate.

Rejected follow-ups:

| Variant | Result | Decision |
| --- | --- | --- |
| FusedMulAdd antiquant | correctness passed, candidate avg `466.838984 us` repeat=3, `max_abs=0.078125` | reject: no speedup and less precision margin |
| blockDim stride over N tiles | AIV-core blockDim produced incorrect target output; single tile/block passed | reject: public Matmul object is not safely reusable across multiple N tiles in one block |
| async `IterateAll<false>` | single correctness passed, repeat timing timed out | reject |
| `K=512,N=64` | correctness passed, `max_abs=0.03125`; candidate avg `513.910665 us` repeat=3 | reject: N too narrow for this public Matmul shape |
| `K=256,N=256` | host `GetTiling failed` before kernel launch | reject: 128KB VECOUT B tile shape not accepted |

Bottleneck update:

- The remaining gap is dominated by many small public Matmul calls inside the
  kernel, Vector unpack/antiquant work, and final C atomic accumulation.
- K merging helps because it reduces output accumulation without increasing the
  local B tile. Larger B tiles would further reduce Matmul/atomic overhead, but
  the tested 128KB VECOUT shape is currently blocked by public tiling/resource
  limits.

## 2026-07-04 bottleneck review and Matmul config tuning

Current bottleneck:

- The implementation now satisfies the no-GM-dequant requirement. The
  antiquantized B tile stays in VECOUT and only final C is written to GM.
- The remaining performance gap is from small public Matmul granularity,
  repeated `SetTensorB(LocalTensor)` handoff, final C GM atomic accumulation
  across K chunks, and Vector unpack/antiquant work.
- `MATMUL_K=256,N_TILE=128` is the best accepted tile so far because it keeps
  the VECOUT B tile at about 64KB. Moving to 128KB B tiles is blocked.

Accepted change:

- Changed the public Matmul config from `CFG_MDL` to `CFG_NORM`.
- Tile shape and no-GM-dequant dataflow are unchanged:
  `MATMUL_K=256,N_TILE=128`, `BType=TPosition::VECOUT`.

Final verification:

| Case | Route | Tolerance | Result |
| --- | --- | --- | --- |
| `[16,4096,22016]` | default VECOUT/Cube K256/N128 + CFG_NORM | `rtol=8e-2, atol=8e-2` | pass, `max_abs=0.046875`, `mean_abs=0.004495220258831978` |

Final wall-clock timing:

| Case | Route | Avg us | Min us | Max us | Warmup | Repeat |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `[16,4096,22016]` | vendor WQMM | `132.503733` | `124.302111` | `157.230999` | 2 | 10 |
| `[16,4096,22016]` | VECOUT/Cube K256/N128 + CFG_NORM | `362.916302` | `351.460999` | `391.397043` | 2 | 10 |
| `[16,4096,22016]` | vendor WQMM | `126.019326` | `120.301964` | `168.345054` | 5 | 30 |
| `[16,4096,22016]` | VECOUT/Cube K256/N128 + CFG_NORM | `364.990005` | `355.703058` | `422.097975` | 5 | 30 |

The best stable comparison is about `2.74x vendor` on repeat=10 and about
`2.90x vendor` on the repeat=30 rerun. Compared with the previous accepted
`CFG_MDL` K256/N128 result (`371.544517 us`, repeat=10), `CFG_NORM` improves
candidate latency by about `2.3%`.

Rejected variants in this pass:

| Variant | Result | Decision |
| --- | --- | --- |
| `K=512,N=128` | build passed, host `GetTiling failed` | reject |
| combined scale/offset param queue | correctness passed, candidate avg `374.777394 us` repeat=10 | reject |
| `CFG_NORM + enUnitFlag=false` | correctness passed, candidate avg `367.278303 us` repeat=10 | reject |
| `CFG_MDL + enUnitFlag=true` | correctness passed, candidate avg `490.663690 us` repeat=10 with large jitter | reject |

Next realistic optimization boundary:

- Public Matmul/VECOUT has little room left unless a larger local B tile becomes
  accepted or L0C/local accumulation can be made correct.
- To close the remaining gap to vendor, the likely required step is a lower
  overhead fused mixed AIV/AIC implementation that keeps K accumulation local
  longer than the current `IterateAll(..., enAtomic)` public API path.

## 2026-07-06 VECIN/VECOUT double-buffer experiment

Change:

- The previous VECOUT/Cube kernel did not use ping-pong buffers: all explicit
  queues were initialized with one buffer.
- Kept queue depth at `TQue<TPosition::*, 1>` and changed only the explicit
  queue buffer number to `BufferNum=2`:
  `weight4Queue`, `scaleQueue`, `offsetQueue`, and `vecoutQueue`.
- Tile shape, Matmul config, no-GM-dequant dataflow, and Python interface are
  unchanged: `MATMUL_K=256,N_TILE=128`, `CFG_NORM`,
  `BType=TPosition::VECOUT`.

Correctness:

| Case | Route | Tolerance | Result |
| --- | --- | --- | --- |
| `[16,4096,22016]` | VECOUT/Cube K256/N128 + CFG_NORM + BufferNum=2 | `rtol=8e-2, atol=8e-2` | pass, `max_abs=0.046875`, `mean_abs=0.004495220258831978` |

Wall-clock timing:

| Case | Route | Avg us | Min us | Max us | Warmup | Repeat |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `[16,4096,22016]` | vendor WQMM | `129.159982` | `120.381010` | `162.045006` | 2 | 10 |
| `[16,4096,22016]` | VECOUT/Cube + BufferNum=2 | `349.860790` | `333.667966` | `422.499026` | 2 | 10 |
| `[16,4096,22016]` | vendor WQMM | `131.627242` | `125.606079` | `170.603977` | 5 | 30 |
| `[16,4096,22016]` | VECOUT/Cube + BufferNum=2 | `351.601928` | `344.735919` | `370.361959` | 5 | 30 |

Decision:

- Keep the change. Against the previous `CFG_NORM` repeat=10 baseline
  (`362.916302 us`), this is about `3.6%` faster. Against the previous
  repeat=30 rerun (`364.990005 us`), this is about `3.7%` faster.
- This is still not a full cross-iteration software pipeline: the code keeps
  the same AllocTensor -> EnQue -> DeQue -> FreeTensor order inside each K tile.
  It only gives the TPipe/queue allocator ping-pong buffer resources. The
  remaining bottleneck is still small public Matmul granularity, VECOUT handoff,
  and GM atomic accumulation across K chunks.

## 2026-07-06 AWQ subgroup unroll and sync tuning

Accepted change:

- Manually unrolled the two AWQ groups inside each `MATMUL_K=256` tile.
- Kept `scaleLocal0/offsetLocal0` live while computing group 1, then released
  both groups after the final vector barrier. This removes one intermediate
  post-Mul barrier/release point and avoids the inner `subCol` loop because
  `VectorCols == TileN == 128`.
- Hoisted invariant `weight4Copy`, `int4Pad`, and vector repeat parameters out
  of the K loop.

Correctness:

| Case | Route | Tolerance | Result |
| --- | --- | --- | --- |
| `[16,4096,22016]` | VECOUT/Cube + BufferNum=2 + subgroup unroll | `rtol=8e-2, atol=8e-2` | pass, `max_abs=0.046875`, `mean_abs=0.004495220258831978` |

Wall-clock timing:

| Case | Route | Avg us | Min us | Max us | Warmup | Repeat |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `[16,4096,22016]` | vendor WQMM | `124.493905` | `119.999051` | `142.717035` | 2 | 10 |
| `[16,4096,22016]` | VECOUT/Cube + subgroup unroll | `346.333301` | `333.725940` | `376.813929` | 2 | 10 |
| `[16,4096,22016]` | vendor WQMM | `122.475457` | `115.014962` | `143.576995` | 5 | 30 |
| `[16,4096,22016]` | VECOUT/Cube + subgroup unroll | `346.482347` | `335.765071` | `359.333004` | 5 | 30 |
| `[16,4096,22016]` | VECOUT/Cube + subgroup unroll, final installed candidate-only rerun | `349.562642` | `339.133083` | `379.764009` | 5 | 30 |

Decision:

- Keep the subgroup unroll. Compared with the prior double-buffer repeat=30
  result (`351.601928 us`), the same both-path test improved to
  `346.482347 us` (`~1.5%`). A final candidate-only rerun after reinstall was
  `349.562642 us`, still slightly faster than the double-buffer-only baseline.

Rejected follow-ups:

- Early prefetch of `scale0/offset0` before weight `DeQue` and `scale1/offset1`
  before group 0 vector compute: correctness passed, but candidate repeat=10
  regressed to `363.333209 us`; reverted.
- Replacing the final post-Mul `PipeBarrier` with VECOUT `EnQue/DeQue` before
  freeing scale/offset: correctness passed, but candidate repeat=10 regressed
  to `351.491303 us`; reverted.

## 2026-07-06 static basicK and high-risk route probes

Accepted change:

- Keep the compliant no-GM-dequant route: `int4 GM -> VECIN`, `Cast int4 -> half`
  into `VECOUT`, fused tile-level AWQ `(q + offset) * scale` in local memory,
  then `SetTensorB(VECOUT LocalTensor)` into public Matmul.
- Change the static Matmul shape from `basicK=128` to `basicK=256` while
  keeping `MATMUL_K=256,N_TILE=128,BufferNum=2`.
- Remove the explicit `enableL1BankConflictOptimise=true`; repeat=10 showed no
  measurable difference on this route.

Final verification:

| Case | Route | Tolerance | Result |
| --- | --- | --- | --- |
| `[16,4096,22016]` | VECOUT/Cube K256/N128 + `basicK=256` | `rtol=8e-2, atol=8e-2` | pass, `max_abs=0.046875`, `mean_abs=0.004495220258831978` |

Wall-clock timing:

| Case | Route | Avg us | Min us | Max us | Warmup | Repeat |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `[16,4096,22016]` | VECOUT/Cube K256/N128 + `basicK=256` | `298.384465` | `294.136000` | `304.985093` | 5 | 30 |
| `[16,4096,22016]` | VECOUT/Cube K256/N128 + `basicK=256`, final rerun | `302.789232` | `295.892009` | `336.866942` | 5 | 30 |
| `[16,4096,22016]` | vendor WQMM | `121.796411` | `117.152929` | `137.661002` | 5 | 30 |

Decision:

- Keep `basicK=256`. It is the best stable public-API route so far, with
  repeat=30 candidate around `298-303 us`.
- This is still only about `40-41%` of vendor throughput
  (`vendor_us / candidate_us`), short of the `60%` target.

Rejected probes:

| Probe | Result | Decision |
| --- | --- | --- |
| `CFG_NORM + enUnitFlag=false` | correctness passed, candidate repeat=10 `311.878300 us` | reject; slower than restored baseline |
| `CONFIG_MDL` with K256/N128 | correctness passed, repeat=10 `308.756716 us` | reject; no gain |
| K512/N128, `BufferNum=1`, 4 AWQ groups per Matmul | correctness passed, repeat=3 `273.307708 us`, but repeat=10 timed out even with `waitIterateAll=true` | reject; unstable VECOUT/Matmul lifetime |
| K384 main + K256 tail | correctness could be fixed by ending main Matmul before tail, quick timing about `307.515 us`, repeat=10 timed out | reject; unstable |
| N256/K256 with two 128-col vector segments | compiled, but correctness failed with `inf` / large diff even with `basicK=128` | reject; public VECOUT ND Matmul shape not reliable here |
| block-stride scheduling with fewer host blocks | correctness script timed out, even when `mm.End()` was moved inside each N tile | reject; Matmul object reuse across N tiles is not safe in this pattern |
| A16W8 route: int4 -> int8 VECOUT + `SetAntiQuantVector` | compile failed: `SetAntiQuantVector` is not exposed for `BType=TPosition::VECOUT,int8_t`; `Cast(int4b_t -> int8_t)` also has no matching implementation | reject for public API route |

Current bottleneck:

- The quantized B tile is not written back to GM, so HBM intermediate traffic is
  not the active bottleneck.
- The remaining public route still performs `172 N tiles * 16 K chunks = 2752`
  Matmul calls and GM atomic accumulations. K512 short-run speed confirms that
  reducing Matmul call count helps, but the public VECOUT handoff becomes
  unstable when the local B tile grows to K512.
- Reaching `>=60%` vendor likely requires a lower-level fused A16W4/A16W8 Cube
  path that supports local int4/int8 B and per-group antiquant inside the Cube
  pipeline, or an official public API that exposes `SetAntiQuantVector` for
  local VECOUT/TSCM int8 B.

## 2026-07-07 Host autotiling and generic-shape fallback

Accepted change:

- Host wrapper now selects a route from shape instead of enforcing VECOUT-only
  constraints during common input validation.
- `AWQ_ASCENDC_ROUTE` unset or `auto` selects VECOUT/Cube only when
  `M <= 16`, `K % 256 == 0`, `N % 128 == 0`, and host Matmul `GetTiling`
  accepts the `GM x VECOUT -> GM` tile.
- If VECOUT is not legal, auto falls back to scalar direct custom compute. This
  fallback writes only final output and does not write a dequantized B tile to
  GM. Explicit `stream` and `split` routes remain diagnostic and are not chosen
  by auto because they use GM scratch.
- The host Matmul tiling legality check is cached per process. A first repeat
  benchmark without the cache hung after repeatedly building host tiling inside
  the Python loop; the cached version completed normally.

Correctness:

| Case | Route | Tolerance | Result |
| --- | --- | --- | --- |
| `[16,4096,22016]` | auto -> VECOUT/Cube K256/N128 | `rtol=8e-2, atol=8e-2` | pass, `max_abs=0.046875`, `mean_abs=0.004495220258831978` |
| `[3,256,136]` | auto -> scalar direct | `rtol=8e-2, atol=8e-2` | pass, `max_abs=0.00244140625`, `mean_abs=0.00038888410199433565` |
| `[17,256,128]` | auto -> scalar direct | `rtol=8e-2, atol=8e-2` | pass, `max_abs=0.00390625`, `mean_abs=0.0003516251454129815` |

Wall-clock timing:

| Case | Route | Avg us | Min us | Max us | Warmup | Repeat |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `[16,4096,22016]` | auto -> VECOUT/Cube, candidate-only smoke | `288.596191` | `284.781912` | `297.735911` | 2 | 5 |
| `[16,4096,22016]` | vendor WQMM | `123.275661` | `119.426055` | `142.335892` | 5 | 30 |
| `[16,4096,22016]` | auto -> VECOUT/Cube | `298.327790` | `290.949130` | `302.972971` | 5 | 30 |

Decision:

- Keep Host autotiling. It broadens supported shapes without hiding a GM
  dequant fallback and keeps target-shape performance in the existing stable
  range (`~298 us` candidate vs `~123 us` vendor in this rerun).
- This is a correctness/generalization improvement, not a throughput
  breakthrough. The performance bottleneck remains public Matmul call count,
  VECOUT handoff overhead, and GM atomic accumulation.
