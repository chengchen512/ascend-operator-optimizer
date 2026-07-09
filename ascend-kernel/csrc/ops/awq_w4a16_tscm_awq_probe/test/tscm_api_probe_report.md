# awq_w4a16_tscm_awq_probe API probe report

## Scope

This isolated probe tests the public AscendC route:

```text
tile-level int4 unpack + antiquant in VECIN
  -> VECIN to TSCM ND2NZ copy
  -> Matmul SetTensorB(LocalTensor from TSCM)
  -> Cube/MMAD output
```

It is intentionally separate from `awq_w4a16_linear`; the main stream baseline is not changed.

## API evidence

- `DataCopyPad(ISASI).md` documents `DataCopyPad(LocalTensor dst, LocalTensor src, DataCopyExtParams, Nd2NzParams)` for Local Memory -> Local Memory, with actual VECIN/VECOUT -> GM -> TSCM and ND -> NZ on the GM -> TSCM leg.
- `随路转换ND2NZ搬运.md` documents `DataCopy(LocalTensor dst, LocalTensor src, Nd2NzParams)` for Local Memory -> Local Memory and supports VECIN/VECCALC/VECOUT -> TSCM on Atlas A2.
- `TSCM简介.md` defines `TSCM<TPosition::VECIN, 1>` as a `TQueBind` alias and requires strict `AllocTensor -> EnQue -> DeQue -> FreeTensor` lifecycle; depth 1 is recommended.
- `SetTensorB.md` documents `SetTensorB(LocalTensor<SrcBT>&)` and says LocalTensor input supports TSCM/VECOUT; setting a TSCM address avoids GM -> A1/B1 loading inside Iterate.
- `Matmul使用说明.md` states that when A or B is `TPosition::TSCM`, the format only supports `CubeFormat::NZ`.

## Implemented variants

All variants compiled successfully with CANN 9.0.0 on Ascend910_9382:

1. `BType = MatmulType<TPosition::TSCM, CubeFormat::NZ, half, ..., TPosition::VECIN>`
2. AIV unpacked int4 pack8 into int8 with `uint64_t` stores, then `Cast/Add/Mul` to half.
3. Variant A: `DataCopyPad(bScm, bHalf, DataCopyExtParams, Nd2NzParams)`.
4. Variant B: `DataCopy(bScm, bHalf, Nd2NzParams)`, matching the shorter TSCM example.
5. Variant C: `TSCM<TPosition::VECIN, 1, 4>` to set mask bit2 and map VECTOR0/VECTOR1 to the same Cube-side memory position.

## Remote validation

Remote workspace:

```text
/root/awq_w4a16_fresh_skill_20260703/ascend-kernel
```

Build:

```text
bash build.sh Ascend910_9382
```

Result:

```text
Build passed; workspace_kernel generated aclrtlaunch_awq_w4a16_tscm_awq_probe.h.
```

Runtime command:

```text
python tests/test_awq_w4a16_tscm_awq_probe.py --benchmark --warmup 20 --repeat 200
```

Observed failure for all three TSCM variants:

```text
RuntimeError: AclrtSynchronizeDeviceWithTimeout, error code is 507015
The MPU address access is invalid.
```

The device log reported an aivector-side exception, not a Python reference or vendor-op parameter failure.

## Baseline preservation

After the failing TSCM probe, the stable main operator was re-run:

```text
python tests/test_awq_w4a16_linear.py --m 2 --k 256 --n 256 --benchmark --warmup 5 --repeat 20
```

Result:

| Shape | Correctness | Candidate | Vendor | Ratio |
|---|---|---:|---:|---:|
| `[2,256,256]` | pass | `17.768999934 us` | `8.643999696 us` | `2.055645599x` |

## Conclusion

The public TSCM direct route is not yet acceptable in this project: it compiles, but VECIN -> TSCM producer/consumer execution still triggers `507015 MPU address access is invalid`, even after matching the documented NZ Matmul format and trying the TSCM mask mapping option.

The safe next optimization route remains the already validated GM scratch + Cube/MMAD probe, while continuing to treat direct TSCM as a separate feasibility investigation.

## 507015 debug update

The follow-up debug pass added an `AWQ_TSCM_STAGE` host parameter:

- Stage 0: unpack int4 -> int8, cast/add/mul antiquant to half in UB, then return.
- Stage 1: TSCM queue-only probe. This is not a valid standalone signal because EnQue without a producer copy can hang.
- Stage 2: UB/VEC -> TSCM ND2NZ copy, then DeQue/Free, no Matmul.
- Stage 3: full UB/VEC -> TSCM ND2NZ -> Matmul path.

Observed results:

- Stage 0 passed, proving the vector unpack/antiquant block itself is not the 507015 source.
- Stage 2 failed consistently with `AclrtSynchronizeDeviceWithTimeout, error code is 507015`; therefore Matmul/Cube consumption is not required to trigger the failure.
- `mssanitizer -t memcheck --kernel-name=awq_w4a16_tscm_awq_probe` did not report an illegal read/write in its log. It only warned that the kernel missed debug line information and that some registers were not reset to defaults.

Fix attempts that still failed:

- Added `V_MTE3` event synchronization before TSCM copy.
- Changed `Nd2NzParams.dstNzMatrixStride` from `0` to `1` per the documented range.
- Switched `DataCopyPad` to row-wise `DataCopyExtParams{blockCount=K, blockLen=N*sizeof(half)}` instead of a single full-matrix block.
- Added 1 KiB padding to UB half source and TSCM destination buffers.
- Added `halfQueue.EnQue/DeQue` after Vector antiquant and before TSCM copy to match the official source-queue lifecycle more closely.
- Tried `TSCM<TPosition::GM, 1>` and matching MatmulType source position.
- Re-tested basic `DataCopy(bScm, bHalf, nd2nz)` with corrected `dstNzMatrixStride=1`.

The failure remains localized to the UB/VEC -> TSCM ND2NZ transfer path. The next useful experiment should be a minimal copy-only kernel copied as closely as possible from the official TSCM example, using static input data and no AWQ logic, before reintegrating antiquant.

## VECOUT public Matmul probe

`AWQ_TSCM_STAGE=4` was added to test another internal path:

```text
AIV unpack/antiquant -> TQue<VECOUT> -> public matmul::Matmul SetTensorB(LocalTensor)
```

Results:

- Direct `MatmulImpl<B=VECOUT/ND>` does not compile. Its static `CopyCubeInB` path still instantiates `SetInput(GlobalTensor)` for B, so `MatmulImpl` is not the right wrapper for LocalTensor B.
- Public `matmul::Matmul<A=GM/ND, B=VECOUT/ND, C=GM/ND, MM_CFG=RunCfg>` plus `REGIST_MATMUL_OBJ` compiles when using typed null tiling: `static_cast<TCubeTiling *>(nullptr)`.
- Runtime no longer raises 507015, but correctness fails badly. Observed output stats for the fixed `[16,128] x [128,64]` probe: `out nan=490/1024`, finite values range roughly `[-32560, 58176]`, while the reference is finite and within about `[-4.88, 4.17]`.

Interpretation:

- The public Matmul/KFC VECOUT path can be compiled and launched, so it is a different failure mode from the direct TSCM copy-only 507015.
- The AIC side is not consuming the AIV-produced VECOUT UB data correctly in this setup. This is likely an address/message/layout constraint of the VECOUT LocalTensor path, not a tolerance issue.
- Remaining internal data-path candidates are lower-level `VECOUT -> B1/A1 -> LoadData -> Mmad`, or a public Matmul/TSCM path copied exactly from the official sample before reintegrating AWQ.

Follow-up after checking the official `SetTensorB(LocalTensor)` contract and the
`matmul_vecout` sample:

- Manual `VECOUT -> B1 -> B2` experiments were stopped; that path belongs inside
  the Matmul high-level API.
- Stage 4 was corrected to pass a local `TCubeTiling` object to
  `REGIST_MATMUL_OBJ` instead of a null tiling pointer. The host-generated GM
  tiling is copied into a local `TCubeTiling` first, matching the official sample
  pattern.
- Added `SetLocalWorkspace()` with a `TBuf<VECCALC>` buffer before `IterateAll`.
- Changed the stage-4 Matmul C type and host tiling C type to `float`, again
  matching the official `matmul_vecout` sample.
- Removed the custom `RunCfg` from the public Matmul client and used the default
  `matmul::Matmul<A, B, C, Bias>` configuration.

These changes all compiled and installed successfully on the remote
`Ascend910_9382` environment, but stage 4 still produced NaNs and failed
correctness:

```text
AWQ_TSCM_STAGE=4 python tests/test_awq_w4a16_tscm_awq_probe.py --stage 4 --rtol 0.08 --atol 0.08
Mismatched elements: 953 / 1024 (93.1%)
Greatest absolute difference: nan at index (0, 4)
```

Current interpretation: the code now follows the documented `SetTensorB` route
much more closely. The remaining failure is likely a platform/sample-support
boundary or an unresolved MIX-mode VECOUT producer/consumer contract issue, not
the previously attempted manual B1/B2 copy path.

## Low-level B1/B2/MMAD probe

`AWQ_TSCM_STAGE=5` and `AWQ_TSCM_STAGE=6` were added to isolate the lower-level
Cube route:

- stage 5: AIV unpack/antiquant to VECOUT, then ND->NZ copy into B1; AIC only
  dequeues/frees B1. This isolates whether the internal VECOUT-to-B1 handoff can
  avoid the previous TSCM 507015 path.
- stage 6: AIC copies A from GM to A1, loads A1->A2, loads B1->B2 with
  `LoadDataWithTranspose`, computes `Mmad`, and writes with `Fixpipe`.
- stage 7: use `TQueBind<VECOUT, B1>` as the producer/consumer queue and let
  Vector write directly into the allocated tensor, without an explicit
  `DataCopy`.
- stage 8: bypass VECOUT and test whether AIV can directly allocate/fill a
  plain B1 queue for AIC to consume.

This remains an isolated probe and does not change the accepted
`awq_w4a16_linear` implementation.

Observed results:

- stage 5 with `DataCopy(b1, bVec, Nd2NzParams)` failed with 507015 on the
  aivector side.
- stage 5 with contiguous `DataCopy(b1, bVec, BElems)` also failed with 507015
  on the aivector side, so the problem is not only ND->NZ conversion.
- replacing the plain B1 queue with `TQueBind<VECOUT, B1>` but keeping explicit
  DataCopy still failed with 507015.
- stage 7 compiled but timed out after 240 seconds, consistent with the AIC side
  not receiving a consumable queue item.
- stage 8 failed with 507015 on the aicore side, indicating a plain B1 queue is
  not a valid AIV-produced/AIC-consumed buffer in this kernel shape.

Current conclusion: direct AIV-produced internal-memory B tiles are still not a
usable route in this environment. The stable path remains the GM scratch route;
future Cube work should either use GM->B1/Matmul-managed copy-in or start from an
official minimal mixed AIV/AIC sample that demonstrates VECOUT->B1 consumption
before reintroducing AWQ.

Baseline re-check after these probes:

| Shape | Correctness | Candidate | Vendor | Ratio |
|---|---|---:|---:|---:|
| `[2,256,256]` | pass | `25.964000821 us` | `12.685999274 us` | `2.046665797x` |

## Official-style TSCM input follow-up

The next pass used the official `SCM输入的矩阵乘` and `batch_matmul_tscm.asc`
patterns more closely:

- host tiling generated `B=TSCM/NZ`, `C=float`;
- kernel used public `matmul::Matmul<A=GM/ND, B=TSCM/NZ, C=GM/ND float>`;
- TSCM queue was changed to `TSCM<TPosition::VECIN, 1, 0x04>`;
- Matmul object was registered with a local `TCubeTiling`;
- the antiquantized B tile was fully resident before `SetTensorB`.

Two variants were tested:

- `AWQ_TSCM_STAGE=9`: `DataCopy(bScm, bHalf, Nd2NzParams)`.
- `AWQ_TSCM_STAGE=10`: `DataCopyPad(bScm, bHalf, DataCopyExtParams, Nd2NzParams)`.

Both variants compiled and installed successfully on the remote
`Ascend910_9382` environment. Both failed at runtime before Matmul correctness
could be evaluated:

```text
RuntimeError: AclrtSynchronizeDeviceWithTimeout, error code is 507015
The MPU address access is invalid.
fftsplus aivector error
```

The stage-10 failure proves that the documented `DataCopyPad(ISASI)`
Local->Local route does not by itself fix the current AIV-produced
`VECIN -> TSCM` MPU error. The failure remains on the producer side.

An additional temporary main-operator experiment reduced `kDeqTileN` from 512
to 256 to test whether the target-shape NaNs were caused by UB pressure. It did
not fix the target random-shape NaN signal and regressed small-shape performance,
so it was reverted. The restored small-shape baseline passed:

| Shape | Correctness | Candidate | Vendor | Ratio |
|---|---|---:|---:|---:|
| `[2,256,256]` | pass | `18.041999638 us` | `8.504000306 us` | `2.121589721x` |

The `[16,4096,22016]` random benchmark generator produced NaNs in both the
vendor reference and candidate, so that generated data distribution is not a
valid allclose signal unless the scale range is reduced or NaNs are handled
explicitly.

## Minimal GM/A1/TSCM direct Matmul probe

To test whether the official `SCM输入的矩阵乘` route is usable independently of
AWQ and AIV-produced data, an isolated PyTorch op
`matmul_tscm_direct_probe` was added:

- fixed shape `M=N=K=64`;
- A half `[64,64]`, B half `[64,64]`, bias float `[64]`, C float `[64,64]`;
- A full-load size is `64 * 64 * sizeof(half) = 8192` bytes, satisfying the
  "TSCM input matrix must be fully resident in L1/TSCM" constraint;
- host tiling uses `SetAType(TSCM, NZ)`, `SetBType(GM, ND)`,
  `SetCType(GM, ND float)`, and workspace-aware launch.

Observed variants:

1. `TSCM<TPosition::GM,1>` with direct `DataCopy(scmTensor, aGlobal, M*K)`
   compiled and ran, but failed correctness with ordinary ND input. This is
   expected because Matmul declares A as `TSCM/NZ`.
2. `TSCM<TPosition::GM,1>` with `DataCopy(scmTensor, aGlobal, Nd2NzParams)`
   compiled and ran without 507015, but correctness was unstable. Typical
   repeated-call diagnostics included `max_abs=2.184064865`, `mean_abs=0.089693`,
   `bad=496/4096`, with occasional larger bad blocks.
3. The same direct-copy variant with Python-side pre-NZ packing was tested
   against several common NZ linear layouts. None matched the Matmul result;
   representative failures were `>4000/4096` bad elements and max error around
   `1.8e3`.
4. Replacing `TSCM<TPosition::GM,1>` with `TQue<TPosition::A1,1>` plus
   GM->A1 `DataCopy(..., Nd2NzParams)` also compiled and ran, but failed
   correctness: `3584/4096` mismatched, max error `62.335853576`.

Current conclusion:

- The official-style full-load L1/TSCM Matmul path is compileable and does not
  reproduce the earlier AIV-side 507015 in the pure GM-source probe.
- It became numerically correct after reproducing the official asc-devkit
  `matmul_tscm/scripts/gen_data.py` layout exactly. That script writes A to GM
  as pre-packed NZ:
  `A.reshape(M/16,16,K/16,16).transpose(2,0,1,3)`, then the kernel performs a
  plain `DataCopy(scmTensor, aGlobal, M*K)` into `TSCM<TPosition::GM,1>`.
- The README table saying A is ND is therefore misleading for the actual direct
  TSCM/NZ sample input. For `MatmulType<TSCM, NZ>`, the source data copied into
  TSCM must already be in the NZ linear order expected by Matmul.

Remote validation after the fix:

| Probe | Correctness | Max abs | Mean abs | Candidate | Reference | Ratio |
|---|---|---:|---:|---:|---:|---:|
| `matmul_tscm_direct_probe` `[64,64,64]` | pass | `5.722e-6` | `4.623e-7` | `8.08-8.15 us` | `16.41-21.78 us` | `0.37-0.49x` |

## AIV-produced NZ follow-up

After the direct GM-source TSCM probe was fixed, four AWQ fused variants were
added to test whether the same NZ contract can be used for AIV-produced B:

- `AWQ_TSCM_STAGE=11`: AIV scalar-unpacks/dequants B directly into official NZ
  order in VECIN, then performs contiguous `DataCopy(VECIN -> TSCM)`.
- `AWQ_TSCM_STAGE=12`: AIV writes the pre-NZ values directly into a TSCM
  `LocalTensor` before enqueueing it for Matmul.
- `AWQ_TSCM_STAGE=13`: AIV writes a pre-NZ B tile to GM user workspace, signals
  AIC with the official `CrossCoreSetFlag<4, PIPE_MTE3>` pattern from
  `matmul_s4_950`, then AIC copies GM->TSCM and calls Matmul.
- `AWQ_TSCM_STAGE=14`: same as stage 13, but uses the documented AIC/AIV
  CrossCore mode id `2`.

Results on `/root/awq_w4a16_fresh_skill_20260703/ascend-kernel`:

| Stage | Result | Interpretation |
|---|---|---|
| 11 | timed out before JSON output | Contiguous `VECIN -> TSCM` remains unusable in this mixed probe even when data is pre-NZ. |
| 12 | failed with `507015`, aivector MPU invalid | AIV direct writes to TSCM LocalTensor are not valid in this setup. |
| 13 | timed out before JSON output | The official `CrossCoreSetFlag<4>` pattern does not transfer cleanly to this PyTorch/workspace-wrapper mixed kernel. |
| 14 | timed out before JSON output | Switching CrossCore mode to documented AIC/AIV mode `2` does not resolve the hang. |

The new hard boundary is: public `GM(pre-NZ) -> TSCM -> Cube` works and is
correct; public AIV-produced `VECIN/VECOUT/TSCM/B1` handoff paths remain blocked
by MPU errors or mixed-kernel synchronization hangs. The stable operator should
therefore keep the no-cache GM tile route until a true official mixed
AIV-to-Cube sample can be reproduced under the same wrapper ABI.

Post-failure sanity checks:

- `matmul_tscm_direct_probe` still passed after failed fused probes:
  candidate `10.0308 us`, reference `27.8616 us`, max abs `5.722e-6`
  (`warmup=10`, `repeat=50`).
- Stable `awq_w4a16_linear` still passed `[2,256,256]`:
  candidate `18.0698 us`, vendor `8.1760 us`, ratio `2.2101x`, max abs
  `0.0048828125`.
