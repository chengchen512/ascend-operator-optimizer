# AWQ W4A16 Local-B Pattern

Use for AWQ/weight-only int4 Linear where the custom candidate must avoid vendor WQMM, target-shape-only fast paths, persistent half caches, and GM materialization of dequantized B.

## Contract

```text
x:                [M,K] fp16
weight:           [K,N/8] int32 int4pack
antiquant_scale:  [K/128,N] fp16
antiquant_offset: [K/128,N] fp16
y:                [M,N] fp16

y = x @ ((int4_unpack(weight) + offset) * scale)
group_size = 128
```

Record signed nibble interpretation and zero-point convention before changing unpack.

## Stable Public-API Route

```text
int4 weight GM
  -> row-strided DataCopyPad into VECIN
  -> Cast int4 to half into VECOUT
  -> scale/offset GM to VECIN
  -> Vector antiquant in VECOUT
  -> Matmul SetTensorB(VECOUT LocalTensor)
  -> internal Cube path
  -> final y GM; later K chunks accumulate
```

On one validated Ascend 910B2C/CANN 9.0.0 environment, the stable tile was M<=16, N tile 128, K tile 256, two AWQ groups per Matmul, and explicit queue buffer number 2. These are measured candidate parameters, not portable constants; Host autotiling must query resources and validate Matmul tiling legality.

## Host Autotiling

- Select the VECOUT/Cube route only when shape/resource/API legality checks pass.
- Cache legality/tiling results by relevant shape and route.
- Preserve a correct custom fallback for unsupported shapes.
- Never let automatic selection choose a route that writes half dequantized B to GM.

## Measured Boundary

For `[16,4096,22016]`, a validated route used 172 N tiles and 16 K chunks, producing 2752 public Matmul calls. It measured about 298 us against a roughly 123 us vendor reference in that environment. The primary bottleneck was public Matmul granularity, VECOUT handoff, and GM atomic accumulation, not half-B GM traffic.

Rejected experiments from the same environment:

| Experiment | Evidence | Decision |
| --- | --- | --- |
| K512/N128 | short run faster, repeated run timed out | reject; local-B/event lifetime suspect |
| K384 main plus K256 tail | short run completed, repeat timed out | reject |
| K256/N256 | incorrect or `inf` | reject |
| N144/N192 | compiled, target correctness timed out | reject |
| GM scratch stream | correct, but writes dequantized B to GM | diagnostic only |
| offset split with extra matmul | correct but slower and violates fusion intent | reject |

## TSCM Boundary

Official TSCM Matmul is useful when the configured single matrix tile is fully resident and in the required format. A minimal GM(pre-NZ)->TSCM->Cube probe can pass, but that does not prove AIV-produced AWQ ND half can be handed directly to AIC.

For `VECIN/VECOUT -> TSCM -> SetTensorB`, require a standalone probe that proves:

- conversion destination and NZ/ND format
- full-load bytes fit L1/TSCM
- `AllocTensor -> EnQue -> DeQue -> Matmul -> FreeTensor`
- repeat correctness and fixed-repeat benchmark completion
- no `507015`, timeout, or event leak

## Direct-int4 Mmad Design Gate

A direct `int4b_t` Cube candidate can avoid both public Matmul granularity and half-B GM materialization, but accept it only after proving the full contract:

1. Preserve signed packed-nibble interpretation from framework weight storage to `GlobalTensor<int4b_t>`.
2. Keep logical `K` for tiling and MMAD while using packed byte extents only in APIs whose copy fields are byte-based.
3. Prove GM ND -> B1 converted layout -> B2 ZN/required Cube layout with a standalone non-square and tail probe.
4. Budget B1/B2 from int4 Cube C0 geometry and prove all local offsets in bytes.
5. Preserve `MTE2_MTE1 -> MTE1_M -> M_FIX` dependencies and repeated-run event reuse.
6. Accumulate K partitions in CO1 with the documented `cmatrixInitVal/cmatrixSource` combination when capacity permits; use Fixpipe only after the final local accumulation.

Do not copy the example's compile-time M/N/K, single-block launch, broad final barrier, or architecture branches into production. Use it to derive a probe, then query core count, tile resources, and exact CANN 9.0.0 overloads for the target operator.

## Next High-Value Direction

Seek a supported lower-overhead fused AIV/AIC or local int8/int4 antiquant path that reduces public Matmul calls and GM atomic partial writes. Do not return to a dequantized-GM cache to improve a bandwidth-saving operator.
