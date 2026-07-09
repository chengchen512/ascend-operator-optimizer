# awq_w4a16_linear fresh design

## Scope

This fresh workspace implements the operator required by `task.md`:

```python
torch.ops.npu.awq_w4a16_linear(x, weight, antiquant_scale, antiquant_offset)
```

Inputs are `[M, K]` fp16 activation, `[K, N/8]` int32 int4pack weight, `[K/128, N]` fp16 scale and offset. Bias is not supported because the task fixes `bias=None`. The output is fp16 `[M, N]`.

## Constraints

- Do not call `torch_npu.npu_weight_quant_batchmatmul` in the implementation.
- Do not add a target shape fast path.
- Do not keep a persistent dequantized `half[K, N]` weight cache.
- Preserve the Python interface while using a host wrapper that can pass workspace to the kernel.
- Compare correctness against `torch_npu.npu_weight_quant_batchmatmul` with the same generated case.

## Current implementation

The current implementation keeps the Python interface unchanged and makes the
workspace-aware VECOUT/Cube route the default:

- Host autotiling: the Python API still exposes one operator, but the host now
  selects a route from the runtime shape. `AWQ_ASCENDC_ROUTE` unset or `auto`
  first checks whether the accepted VECOUT tile is legal (`M <= 16`,
  `K % 256 == 0`, `N % 128 == 0`) and whether host Matmul `GetTiling` accepts
  the `GM x VECOUT -> GM` tile. The host caches this tiling legality check per
  process, so benchmark loops do not repeatedly call the public tiling API. If
  yes, it launches the fast VECOUT route. If not, it falls back to the pure
  custom scalar direct kernel, which computes output directly and does not
  materialize an antiquantized B tile in GM. The legacy `stream` and `split`
  routes are explicit diagnostics only and are not selected automatically
  because they use a GM half scratch tile.
- Default fast route: fused tile-level `int4 -> half antiquant` into a
  `TQue<TPosition::VECOUT, 1>` LocalTensor. The kernel then calls
  `matmul::Matmul` with `B = MatmulType<TPosition::VECOUT, CubeFormat::ND,
  half>` and `SetTensorB(LocalTensor)`. The antiquantized B tile is never
  written to GM. K groups are accumulated into the final output by
  `IterateAll(..., enAtomic=kg!=0)`. The current accepted tile is
  `MATMUL_K=256, N_TILE=128` with the public `CFG_NORM` Matmul config.
  The explicit VECIN/VECOUT queues now allocate two buffers per queue
  (`BufferNum=2`) so MTE local copies have ping-pong buffer resources while
  preserving queue depth 1 and the original tensor lifecycle. The two AWQ
  groups inside each `MATMUL_K=256` tile are manually unrolled so the first
  group's scale/offset buffers are kept live until the second group's vector
  compute finishes; this removes one intermediate source-buffer release
  barrier without changing the numerical dataflow.
- Legacy stream route: set `AWQ_ASCENDC_ROUTE=stream` to run the old no-cache
  GM scratch implementation. It is kept only for regression evidence because
  it writes the antiquantized half tile to HBM.
- Diagnostic route: set `AWQ_ASCENDC_ROUTE=scalar` to run a pure scalar-pack
  custom kernel. It computes 8 adjacent columns for one row at a time and is
  used only as a correctness anchor.
- Rejected experiment: set `AWQ_ASCENDC_ROUTE=split` to dequantize
  `q * scale` only and compute the offset term as
  `mm(sum_group(x), offset * scale)`. This is correct, but target-shape
  benchmark regressed from about `2.12x vendor` to about `2.66x vendor`, so it
  is kept only as evidence.
- Workspace: host allocates `GetLibApiWorkSpaceSize()+512` bytes and passes it
  to the VECOUT/Cube kernel. The generated wrapper is workspace-aware, with
  kernel parameters ordered as `..., m, k, n, workspace, tiling`.
- Rejected wide-N routes: `AWQ_ASCENDC_ROUTE=n192` and `n144` were tested as
  no-GM-dequant VECOUT/Cube experiments to reduce public Matmul call count.
  N192 would reduce the target shape from `2752` to `1840` calls, and N144 to
  `2448` calls. Both compiled, but both timed out during target-shape
  correctness. N192 also failed after reducing `basicK` from 128 to 64, so the
  failure is likely the hidden public Matmul/VECOUT handoff boundary for wider
  local-B tiles, not just L0B basicK pressure. These routes must not be
  promoted unless a future CANN/public-Matmul contract change makes them pass
  correctness and repeat timing.

The default route is now compliant with the hard bandwidth constraint: no
antiquantized B intermediate may be copied back to GM.

Generic shape support is deliberately separated from the fast path: shape
families outside the accepted VECOUT tile are supported by `auto -> scalar`
for correctness and API coverage, while performance-sensitive shapes remain on
the VECOUT/Cube route. This avoids hiding a GM-dequant fallback behind
autotiling.

## Current data path and bottleneck

The accepted default data path is:

```text
weight int4 GM
  -> DataCopyPad row-strided tile into weight4Queue(VECIN)
  -> Cast int4 -> half into vecoutQueue(VECOUT)
  -> DataCopyPad scale/offset into VECIN
  -> Add/Mul antiquant in VECOUT
  -> SetTensorB(VECOUT LocalTensor)
  -> Matmul internal VECOUT -> Cube path
  -> IterateAll final y GM, atomic for kg > 0
```

For `[16,4096,22016]`, the stable N128 route launches one block per 128-column
tile and runs 16 K chunks per block. That produces 2752 public Matmul calls and
15 atomic accumulations per output tile. The latest stable timing is about
`298-303 us` versus vendor `121.8 us`, so the remaining gap is dominated by
public Matmul granularity, VECOUT handoff overhead, and GM atomic accumulation,
not by a dequantized B HBM intermediate.

The N192 experiment attacks exactly that call-count bottleneck without changing
the bandwidth contract. It is intentionally route-gated because previous larger
tiles (`K512/N128`, `K256/N256`) either timed out or failed correctness even
when short-run timing looked promising.

## Official API route selection

Official CANN 9.0.0 Matmul/TSCM documents and `asc-devkit` samples were used to narrow the route:

- `SetTensorB(LocalTensor)` supports a local B matrix from `TSCM` or `VECOUT`; with TSCM, Matmul assumes the matrix is already resident in L1 and does not perform GM-to-L1 copy.
- Matmul `A/B` can be `GM`, `VECOUT`, or `TSCM` on A2/A3 style products, but a TSCM input matrix must be fully resident for the single-core computation, and TSCM input format is constrained to `CubeFormat::NZ`.
- The public `matmul_int4` route is not a match for this operator because it requires both A and B to be `int4b_t`; this task is W4A16, namely fp16 activation times int4 weight with per-group fp16 antiquant.
- The official `matmul_s4_950` compatibility sample shows that a mixed AIV/AIC design can unpack/extend int4 on Vector side and then feed Cube, but our workspace-aware PyTorch wrapper probes using cross-core AIV-to-AIC handoff currently either time out or hit `507015`.

Therefore the checked-in operator uses the public VECOUT `SetTensorB` route
instead of the previous GM scratch route. It is correct and respects the
no-GM-dequant requirement, but it is still slower than vendor because each
128-wide K group is a separate Matmul accumulation and the final C tile is
updated with GM atomic add. The next route most likely to approach vendor is
not public `int4 Matmul`; it is a lower-overhead fused mixed kernel/wrapper
that keeps Cube accumulation local or uses an official TSCM/NZ handoff without
falling back to a GM half B tile.

## Next optimization route

The next accepted performance route must keep the same no-GM-dequant dataflow
and reduce the overhead in the current VECOUT/Cube path:

1. Keep generic N tiling; the stable value is currently `N_TILE=128` with
   two AWQ groups fused into each `MATMUL_K=256` Cube call.
2. For each active block, unpack int4 into a local half B tile and apply AWQ
   scale/offset in Vector local memory only.
3. Feed B through `MatmulType<TPosition::VECOUT, CubeFormat::ND, half>` or a
   proven `MatmulType<TPosition::TSCM, CubeFormat::NZ, half>` path.
4. Avoid promoting any route that writes the antiquantized half B tile to GM,
   even as a temporary workspace.
5. Never promote a Cube route unless correctness passes before benchmark, and
   never hide a target-shape special case behind route selection.

Current candidate order:

1. Keep the host-autotiled N128/K256/`basicK=256` public VECOUT route as the
   accepted performance baseline.
   Remote repeat=30 after wide-N experiments was `296.904 us` versus vendor
   `122.936 us`, with correctness unchanged (`max_abs=0.046875`).
2. Keep scalar direct as the generic-shape fallback, not as a performance
   candidate. It is slow but respects the no-GM-dequant constraint.
3. Do not continue wider-N public VECOUT tiles without a new API insight:
   N144 and N192 both compile but time out on target-shape correctness.
4. Return to lower-level fused AIV/AIC or an
   official local int8/int4 antiquant API path; do not return to GM scratch.

## Test plan

- Small correctness: `[M,K,N] = [2,256,256]`, vendor reference, `rtol=1e-2`, `atol=2e-2`.
- Autotiling fallback correctness: one non-VECOUT shape such as
  `[3,128,136]` should pass through `auto -> scalar` with vendor reference.
- Task correctness: `[16,4096,22016]`, vendor reference, relaxed only if documented.
- Benchmark: prefer wall-clock `perf_counter + torch.npu.synchronize()` for the
  VECOUT/Cube path because NPU Event timing has hung in this route.

## Skill notes

This run uses the cloned `ascend-operator-optimizer` skill flow:

- inspect task and implementation type;
- avoid hard-stop violations;
- keep probe failures isolated;
- record API lookups instead of reading all official docs;
- write back skill improvements when workflow gaps are found.
