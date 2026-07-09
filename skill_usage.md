# Skill usage log

## Used skills

- `ascend-operator-optimizer` cloned from `chengchen512/ascend-operator-optimizer`, branch `codex/ascendc-awq-experience`.
- `ascendc-operator-project-init` reference template from the cloned skill repository.
- `ssh-remote-connect` from this project for remote configuration and connectivity checks.

## Read references

- `.agents/skills/ascend-operator-optimizer/SKILL.md`
- `references/INDEX.md`
- `references/playbook.md`
- `references/constraints.md`
- `references/harness.md`
- `references/hardware.md`
- `references/api-lookup.md`
- `references/ascendc.md`
- `references/ascendc/workflow.md`
- `references/ascendc/launch-profiles.md`
- `references/ascendc/tiling-grid.md`
- `references/ascendc/memory.md`
- `references/ascendc/pipeline.md`
- `references/ascendc/api-usage.md`
- `references/ascendc/data-copy.md`
- `references/ascendc/precision.md`
- `references/ascendc/profiling.md`
- `references/cases/awq-w4a16-ascendc.md`
- `references/ascendc-examples.md`

## API lookups

Used the cloned `api_lookup.py` against `reference/asc-devkit-api-9.0.0/docs/api`:

- `TSCM`
- `Matmul`
- `DataCopy`
- `Cast`

The design keeps TSCM/Cube/MMAD as the next route, but only after a dedicated TSCM ND-to-NZ copy and Matmul consumer probe passes.

## Current result

Created a fresh `ascend-kernel` workspace with:

- `csrc/ops/awq_w4a16_linear/op_host/awq_w4a16_linear.cpp`
- `csrc/ops/awq_w4a16_linear/op_kernel/awq_w4a16_linear.cpp`
- `csrc/ops/awq_w4a16_linear/design.md`
- `tests/test_awq_w4a16_linear.py`

The current kernel has a workspace-aware ABI and two routes:

- default: custom no-cache stream dequant tile kernel + per-tile matmul;
- `AWQ_ASCENDC_ROUTE=scalar`: pure custom scalar-pack diagnostic route.

It does not call vendor fused AWQ in implementation, does not use a target-shape fast path, and does not keep a persistent dequantized weight cache.

## Remote results

Remote workspace: `/root/awq_w4a16_fresh_skill_20260703`.

- Build: pass, CANN 9.0.0, SOC `Ascend910_9382`, Python venv `/root/ascend-operator-optim/.venv`.
- Small case `[2,256,256]`, default stream route: correctness pass, `max_abs=0.0048828125`, `mean_abs=0.0008409023284912109`, `candidate_us=18.1554`, `vendor_us=7.9832`, ratio `2.27x`.
- Small case `[2,256,256]`, scalar route: correctness pass, `candidate_us=1936.17`, `vendor_us=7.9620`, ratio `243.18x`.
- Target case `[16,4096,22016]`, default stream route: correctness pass at `rtol=5e-2, atol=5e-2`, `max_abs=0.0234375`, `mean_abs=0.003105091629549861`.
- Target benchmark `[16,4096,22016]`: `candidate_us=189.7602`, `vendor_us=87.3658`, ratio `2.17x`.
- Tile sweep on target case: `4096 -> 289.43 us`, `8192 -> 224.94 us`, `12288 -> 189.72 us`, `16384 -> 200.67 us`, `22016 -> 230.41 us`.

## Remote workflow findings

- SFTP failed with `EOF during negotiation`; Paramiko exec + stdin upload worked.
- Windows-created tar preserved CRLF, so remote shell scripts failed until text files were converted to LF.

## Remaining gap

The stream route is a stable no-cache baseline but still writes dequantized half tiles to HBM and remains about `2.17x` slower than vendor. The next required experiment is a minimal TSCM/Cube/MMAD producer-consumer probe that keeps the dequantized tile off GM.

## 2026-07-03 TSCM API direct probe

Additional API documents checked:

- `DataCopyPad(ISASI).md`
- `随路转换ND2NZ搬运.md`
- `TSCM简介.md`
- `SetTensorB.md`
- `Matmul使用说明.md`

Implemented an isolated `awq_w4a16_tscm_awq_probe` using the documented route:

- `BType = TPosition::TSCM, CubeFormat::NZ`
- AIV int4 pack8 unpack + direct vector antiquant
- VECIN local half tile to TSCM with `DataCopyPad(..., Nd2NzParams)` and then `DataCopy(..., Nd2NzParams)`
- AIC `SetTensorB(LocalTensor)` and `MatmulImpl`
- TSCM mask variant `TSCM<VECIN, 1, 4>`

Remote build passed on `/root/awq_w4a16_fresh_skill_20260703/ascend-kernel`, but every direct TSCM variant failed at runtime with `507015 MPU address access is invalid` on the aivector side. The stable main stream route was re-run after the failed probe and still passed small-shape correctness with `candidate_us=17.768999934`, `vendor_us=8.643999696`, ratio `2.055645599x`.

Failure evidence is recorded in `ascend-kernel/csrc/ops/awq_w4a16_tscm_awq_probe/test/tscm_api_probe_report.md`.

## 2026-07-03 507015 TSCM debug pass

Used skills:

- `ssh-remote-connect`: remote upload, build, wheel install, and NPU execution on `/root/awq_w4a16_fresh_skill_20260703`.
- `ascendc-mssanitizer`: memcheck pass for `awq_w4a16_tscm_awq_probe` stage 2.

The 507015 debug pass followed the MPU invalid-address guidance:

- Added `AWQ_TSCM_STAGE` to isolate antiquant-only, TSCM copy-only, and full Matmul paths.
- Stage 0 (`unpack + antiquant` only) passed synchronization, so the basic UB allocation and Vector compute path are not the failing block.
- Stage 2 (`VEC/UB -> TSCM ND2NZ`, no Matmul) still failed with `AclrtSynchronizeDeviceWithTimeout, error code is 507015`, so the current failure is before Cube consumption.
- `mssanitizer -t memcheck --kernel-name=awq_w4a16_tscm_awq_probe` produced no illegal read/write record; it only reported missing debug-line information and register reset warnings.
- Tried and ruled out these suspected fixes: `V_MTE3` synchronization before TSCM copy, `dstNzMatrixStride=1`, `DataCopyPad` row-wise `DataCopyExtParams`, 1 KiB UB/TSCM padding, source `halfQueue.EnQue/DeQue`, `TSCM<TPosition::GM>`, and basic `DataCopy(..., Nd2NzParams)`.

Current conclusion: direct fused VECIN/UB -> TSCM ND2NZ is still a blocked feasibility path in this environment. The useful signal is that stage 0 is clean and stage 2 fails consistently with `mte error info` / `fixp_error0` on the aivector side, so future attempts should focus on a minimal official-style TSCM copy kernel or a GM scratch -> Cube path, not on the AWQ dequant math.

## 2026-07-03 VECOUT public Matmul data-path probe

Used skills:

- `ascendc-operator-performance-optim`: selected the next internal-memory path to probe after direct TSCM failed.
- `ssh-remote-connect`: uploaded the isolated probe, rebuilt the wheel, installed it, and ran NPU tests remotely.

Additional API/code evidence checked:

- `SetTensorB.md`: LocalTensor B is documented for `TSCM/VECOUT`.
- `Matmul使用说明.md`: A/B can be `VECOUT`; TSCM requires NZ, VECOUT can use ND.
- CANN 9.0.0 headers: public `matmul::Matmul` has `SetTensorB(LocalTensor)`; `MatmulImpl` static copy-in still chose `GlobalTensor` for the VECOUT B probe.

Experiment:

- Added `AWQ_TSCM_STAGE=4`.
- AIV writes the dequantized half B tile into `TQue<TPosition::VECOUT, 1>`.
- AIC consumes it through public `matmul::Matmul<A=GM/ND, B=VECOUT/ND, C=GM/ND, MM_CFG=RunCfg>` and `REGIST_MATMUL_OBJ`.

Remote result:

- Build/install passed after switching to public Matmul/KFC and typed null tiling: `static_cast<TCubeTiling *>(nullptr)`.
- Runtime did not hit 507015, but correctness failed with invalid output distribution: `out nan=490/1024`, finite values roughly `[-32560, 58176]` versus finite reference roughly `[-4.88, 4.17]`.

Conclusion:

- `VECOUT -> public Matmul` is not blocked at compile/launch level, but this implementation does not yet transfer the AIV-produced UB data to AIC correctly.
- Direct `MatmulImpl<VECOUT>` is ruled out for this use because it does not accept the LocalTensor B operand in the needed copy-in path.
- Next viable internal route should be lower-level `VECOUT -> B1/A1 -> LoadData -> Mmad`, where the UB-to-L1 and L1-to-L0 movement are explicit, or a minimal official-style public Matmul/TSCM sample before reintroducing AWQ.

## 2026-07-03 low-level VECOUT/B1/B2/MMAD probe

Used skills:

- `ascendc-operator-performance-optim`: selected and scoped the lower-level
  Cube data-path experiment.
- `ssh-remote-connect`: remote upload/build/test workflow.

API references checked:

- CANN 9.0.0 `TQueBind简介`: documents `VECOUT -> A1/B1/C1` as an explicit
  internal data path.
- CANN 9.0.0 `Load2D`, `LoadDataWithTranspose`, `Mmad`, `Fixpipe`: used to
  build an explicit `A1/A2/B1/B2/CO1` Cube path.
- Code-gen `data-copy-api`, `resource-management-api`, `sync-control-api`:
  queue lifecycle, alignment, and synchronization checks.

Experiment:

- Added `AWQ_TSCM_STAGE=5` for VECOUT-to-B1 copy-only synchronization.
- Added `AWQ_TSCM_STAGE=6` for explicit low-level Cube compute:
  `A GM->A1->A2`, `B VECOUT->B1->B2`, `Mmad`, `Fixpipe`.
- Purpose: determine whether AIV-generated antiquantized half tiles can bypass
  the failing TSCM path and feed Cube through B1/B2 without persistent HBM cache.

Remote results:

- `stage=5` with VECOUT-to-B1 ND->NZ DataCopy failed with 507015 on the
  aivector side.
- `stage=5` with contiguous VECOUT-to-B1 DataCopy still failed with 507015, so
  the failure is not limited to ND->NZ parameters.
- Changing the B1 resource to `TQueBind<VECOUT, B1>` while keeping explicit
  DataCopy also failed with 507015.
- `stage=7`, where Vector wrote directly into the tensor allocated by
  `TQueBind<VECOUT, B1>`, compiled but timed out after 240 seconds.
- `stage=8`, where AIV directly filled a plain B1 queue and AIC dequeued it,
  failed with 507015 on the aicore side.

Decision:

- Treat direct AIV-produced TSCM/B1/VECOUT-to-Cube paths as blocked until an
  official minimal mixed AIV/AIC sample demonstrates the queue handoff on this
  CANN/SOC combination.
- Keep the accepted `awq_w4a16_linear` GM scratch route as the stable no-cache
  baseline. Re-run after probes passed correctness on `[2,256,256]`:
  `candidate_us=25.964000821`, `vendor_us=12.685999274`, ratio `2.046665797x`.

## 2026-07-03 corrected VECOUT SetTensorB probe

User correction:

- Do not hand-write `VECOUT -> B1 -> B2`; use `matmulObj.SetTensorB(LocalTensor)`
  and let Matmul high-level API do the internal VECOUT-to-Cube path.

References checked:

- Official `SetTensorB`: LocalTensor overload supports `TSCM/VECOUT`, with
  TensorB address space at least `singleK * singleN`.
- Official `Matmul使用说明`: Matmul object registration must use matching
  tiling and then `SetTensorA/SetTensorB/Iterate/End`.
- Official `matmul_vecout` sample: copies GM tiling into local `TCubeTiling`,
  registers Matmul with that local tiling, uses `TQue<VECOUT>`, calls
  `SetLocalWorkspace`, and uses float C.

Changes made to stage 4:

- Copied host-generated GM tiling into local `TCubeTiling` before
  `REGIST_MATMUL_OBJ`.
- Added `TBuf<VECCALC>` and `mmClient.SetLocalWorkspace()`.
- Switched stage-4 C/Bias MatmulType and host tiling C type to float.
- Removed custom `RunCfg` from the public Matmul client and used the default
  `matmul::Matmul<A, B, C, Bias>`.

Remote result:

- Build/install succeeded for each correction.
- Runtime still failed correctness with NaNs:
  `AWQ_TSCM_STAGE=4`, mismatch `953/1024`, greatest absolute difference `nan`.

Conclusion:

- The VECOUT `SetTensorB(LocalTensor)` route has now been tried in the official
  shape, but still fails on the current `Ascend910_9382` environment. This points
  to an unresolved platform/MIX producer-consumer contract issue rather than a
  manual B1/B2 implementation issue.

## 2026-07-03 official-style TSCM input follow-up

User references:

- Official `SCM输入的矩阵乘`: TSCM input requires the matrix to be fully resident
  in TSCM and Matmul consumes the user-managed TSCM address directly.
- Official `batch_matmul_tscm.asc` sample: uses `TSCM<TPosition::GM, 1, 0x04>`,
  `MatmulType<TSCM, NZ, ...>`, host `SetAType(TSCM, NZ)`, and local
  `TCubeTiling` registration.
- Official `DataCopyPad(ISASI)`: documents Local->Local
  `DataCopyPad(LocalTensor, LocalTensor, DataCopyExtParams, Nd2NzParams)` for
  `VECIN/VECOUT -> TSCM`.

Changes tested:

- Added `AWQ_TSCM_STAGE=9`: public `matmul::Matmul` with
  `B=TSCM/NZ`, `C=float`, `TSCM<VECIN,1,0x04>`, and
  `DataCopy(Local, Local, Nd2NzParams)` from AIV antiquantized half tile into
  TSCM.
- Added `AWQ_TSCM_STAGE=10`: same Matmul/TSCM setup, but replaced the TSCM copy
  with `DataCopyPad(Local, Local, DataCopyExtParams, Nd2NzParams)`.
- Temporarily changed main `awq_w4a16_linear` dequant tile from 512 to 256 to
  test whether target-shape NaNs were caused by UB pressure; this was reverted.

Remote results:

- Stage 9 compiled and installed, but failed at runtime with 507015. Device log
  identified `fftsplus aivector error` and `The MPU address access is invalid`.
- Stage 10 also compiled and installed, but failed with the same aivector-side
  507015. This rules out a simple `DataCopy` vs `DataCopyPad(ISASI)` API swap.
- Target random benchmark shape `[16,4096,22016]` produced NaNs in both vendor
  reference and candidate with the current random scale/offset generator, so
  that generated case is not a valid allclose signal without reducing scale or
  allowing equal NaNs.
- `kDeqTileN=256` did not fix the target NaN signal and regressed small-shape
  performance to about `2.94x` vendor, so it was reverted to 512.
- Final restored baseline on `[2,256,256]` passed:
  `candidate_us=18.041999638`, `vendor_us=8.504000306`, ratio `2.121589721x`.

Decision:

- Keep stage 9/10 only as isolated evidence. Do not promote AIV-produced
  TSCM/VECOUT/B1 tiles into the main operator on this `Ascend910_9382` setup.
- The stable main implementation remains the GM scratch + torch/mm route with
  no persistent dequant cache and no target-shape fast path.

## 2026-07-03 minimal TSCM direct Matmul probe

Skills used:

- `ascendc-operator-performance-optim`: performance/debug workflow and report
  discipline.
- `ssh-remote-connect`: remote upload, build, wheel install, and NPU execution.

Files added for the isolated probe:

- `ascend-kernel/csrc/ops/matmul_tscm_direct_probe/op_kernel/matmul_tscm_direct_probe.cpp`
- `ascend-kernel/csrc/ops/matmul_tscm_direct_probe/op_host/matmul_tscm_direct_probe.cpp`
- `ascend-kernel/tests/test_matmul_tscm_direct_probe.py`

What was tested:

- A pure `M=N=K=64` Matmul with A fully resident in L1/TSCM
  (`8192` bytes), B from GM, bias from GM, C float.
- `TSCM<TPosition::GM,1>` direct copy, `TSCM<TPosition::GM,1>` with GM->TSCM
  ND2NZ, Python-side pre-NZ packing, and `TQue<TPosition::A1,1>` with GM->A1
  ND2NZ.

Remote result:

- All variants compiled and launched in the workspace-aware host wrapper.
- The pure GM-source variants did not hit the previous AIV-side 507015, so the
  failure mode changed from illegal access to wrong numerical results.
- None of the tested variants passed correctness. The best repeated diagnostic
  for GM->TSCM ND2NZ still had `max_abs=2.184064865`, `mean_abs=0.089693`, and
  `496/4096` bad elements, with occasional larger bad blocks.
- The A1 queue variant also failed correctness (`3584/4096` bad elements,
  max error `62.335853576`).

Decision:

- Treat official-style TSCM direct Matmul as not yet usable in this
  PyTorch/workspace-wrapper project.
- Do not integrate it into `awq_w4a16_linear` until the exact official
  `matmul_tscm` sample is reproduced in this build framework or the missing
  `TSCM/NZ` layout contract is identified.

## 2026-07-04 official API route review and default restoration

Skills used:

- `ascendc-operator-design`: selected the main implementation route from the
  official Matmul/TSCM/int4 constraints.
- `ascendc-operator-performance-optim`: benchmarked the split-offset route and
  restored the faster stable baseline.
- `ascendc-operator-code-gen`: adjusted the host route switch and kept the
  experimental mode isolated behind `AWQ_ASCENDC_ROUTE=split`.
- `ssh-remote-connect`: uploaded the modified sources and ran remote build,
  wheel install, correctness, and benchmark tests.

Official references checked:

- CANN 9.0.0 `SetTensorB`: `LocalTensor` B supports `TSCM/VECOUT`.
- CANN 9.0.0 `Matmul使用说明`: local `TSCM/VECOUT` Matmul inputs must fit the
  single-core computation; TSCM input format is constrained to `CubeFormat::NZ`.
- `asc-devkit` 9.0.0 `matmul_int4`: public int4 Matmul requires both A and B to
  be `int4b_t`, so it does not implement fp16 x int4 W4A16.
- `asc-devkit` 9.0.0 `matmul_s4_950`: mixed AIV/AIC is the closest official
  direction to vendor, but this PyTorch workspace wrapper has not yet made that
  handoff correct.

Experiment:

- Added split-offset mode: dequantize `q * scale` in the custom kernel, then add
  `mm(sum_group(x), offset * scale)` on host.
- Correctness passed, but target performance regressed to `234.3387 us` versus
  vendor `87.9427 us` (`2.66x`), so the default route was restored.

Restored remote results:

- `[2,256,256]`: pass, `max_abs=0.0048828125`,
  `mean_abs=0.0008409023284912109`, candidate `18.1350 us`,
  vendor `8.4440 us`, ratio `2.15x`.
- `[16,4096,22016]`: pass at `rtol=5e-2, atol=5e-2`,
  `max_abs=0.0234375`, `mean_abs=0.003105091629549861`,
  candidate `189.7180 us`, vendor `89.4527 us`, ratio `2.12x`.

Decision:

- Keep the no-cache stream dequant route as the stable checked-in operator.
- The route most likely to approach vendor is a dedicated fused mixed AIV/AIC
  kernel/wrapper that performs tile-level unpack/antiquant and feeds
  `TSCM/VECOUT -> Cube/MMAD` without writing a half B tile to GM.
- Public int4 Matmul and split-offset are not suitable as the main route for
  this W4A16 task.

## 2026-07-04 no-GM-dequant VECOUT/Cube default

Skills used:

- `ascendc-operator-performance-optim`: reviewed the new hard bandwidth
  constraint, validated the VECOUT/Cube default route, benchmarked it against
  vendor, and reverted an unstable micro-optimization.
- `ascendc-operator-code-gen`: updated the workspace-aware kernel and host
  launch shape, and added the wall-clock timing helper.
- `ssh-remote-connect`: uploaded the modified files, ran remote build/install,
  correctness, timing, and cleanup commands.

Hard constraint:

- The optimized operator must not copy the antiquantized B tile back to GM.
  The legacy GM scratch stream route is therefore no longer the default and is
  kept only behind `AWQ_ASCENDC_ROUTE=stream` for comparison.

Promoted route:

- `awq_w4a16_linear_vecout` is now the default implementation for
  `torch.ops.npu.awq_w4a16_linear`.
- Dataflow:
  `int4 weight GM -> VECIN LocalTensor -> vector Cast/Add/Mul antiquant ->
  VECOUT LocalTensor -> matmul SetTensorB(LocalTensor) -> final y GM`.
- The antiquantized B tile stays in local memory. Only the final output C is
  written to GM; K groups use Matmul atomic accumulation on the final output.
- Host launch uses the workspace-aware order
  `..., m, k, n, workspace, tiling`.

Remote results:

- Build/install passed on
  `/root/awq_w4a16_fresh_skill_20260703/ascend-kernel`,
  CANN 9.0.0, SOC `Ascend910_9382`.
- Correctness for `[16,4096,22016]` passed at `rtol=8e-2, atol=8e-2`,
  `max_abs=0.0625`, `mean_abs=0.005369903519749641`.
- Wall-clock timing uses `perf_counter + torch.npu.synchronize()` because NPU
  Event timing can hang on this route.
- Short timing, repeat=3:
  - vendor: avg `154.924967 us`, min `120.505923 us`, max `213.849009 us`;
  - VECOUT/Cube default: avg `466.850004 us`, min `410.337001 us`,
    max `567.427021 us`;
  - ratio: about `3.01x vendor`.

Rejected or reverted:

- Moving `SetOrgShape/SetSingleShape` outside the K-group loop compiled and
  passed one correctness run, but repeated candidate timing timed out. Reverted.
- `N_TILE=512` remains rejected due `GetTiling` failure or `507015` UB
  out-of-bounds.
- TSCM/NZ and direct AIV-produced TSCM/B1 handoff remain blocked by timeout or
  `507015`.
- L0C partial-sum accumulation did not produce correct values.

Decision:

- Keep VECOUT/Cube as the default because it is the only current mainline route
  that satisfies the no-GM-dequant constraint.
- Do not promote any future optimization that writes the half B tile to GM,
  even if it is faster in short benchmarks.

## 2026-07-04 VECOUT/Cube K-merge optimization

Skills used:

- `ascendc-operator-performance-optim`: selected K-merge tile shapes from the
  no-GM-dequant bottleneck and rejected unstable or slower variants.
- `ascendc-operator-code-gen`: updated host tiling constants and the VECOUT
  kernel antiquant loop while preserving the workspace-aware Python interface.
- `ssh-remote-connect`: uploaded each variant and ran remote build, wheel
  install, correctness, and wall-clock timing.

Accepted change:

- Default VECOUT/Cube tile changed from `K=128,N=256` to `K=256,N=128`.
- The dequantized B tile remains `32768` half elements, about 64KB, so the
  route still avoids a GM half-tile intermediate and does not increase VECOUT
  B capacity.
- Two AWQ quant groups are unpacked and antiquantized into one VECOUT B tile,
  then consumed by one `SetTensorB(LocalTensor)` Matmul. This halves final C
  atomic accumulation traffic compared with one 128-K group per Matmul.

Final remote result on `[16,4096,22016]`:

- Correctness: pass at `rtol=8e-2, atol=8e-2`, `max_abs=0.046875`,
  `mean_abs=0.004495220258831978`.
- Candidate wall-clock: repeat=10, warmup=2, avg `371.544517 us`, min
  `354.437972 us`, max `421.473058 us`.
- Vendor wall-clock under the same repeat/warmup settings: avg
  `129.779603 us`, min `122.887082 us`, max `157.117029 us`.
- Current gap: about `2.86x vendor`.

Rejected variants:

- `FusedMulAdd` antiquant: correctness passed, but performance was essentially
  unchanged and `max_abs` rose to `0.078125`, so Add+Mul was restored.
- BlockDim stride over multiple N tiles per block: default AIV-core blockDim
  produced incorrect output; one N tile per block remained the correct mapping.
- Async `IterateAll<false>`: one correctness run passed, repeated timing
  timed out.
- `K=512,N=64`: correctness passed with `max_abs=0.03125`, but candidate
  slowed to `513.910665 us`.
- `K=256,N=256`: host `GetTiling failed`; the 128KB VECOUT B tile shape is not
  accepted by the current public Matmul tiling path.

Bottleneck update:

- The dominant remaining cost is the high count of small public Matmul calls
  plus Vector unpack/antiquant and output atomic accumulation overhead. The
  accepted K-merge reduces output accumulation cost without increasing B tile
  memory. Larger B tiles would be the natural next step, but current public
  tiling/local-memory limits reject the 128KB attempt.

## 2026-07-04 bottleneck review and Matmul config tuning

Skills used:

- `ascendc-operator-performance-optim`: reviewed bottlenecks and compared tile,
  queue, and Matmul config variants with the same target case.
- `ascendc-operator-code-gen`: updated Matmul config and queue experiments while
  checking DataCopy/Vector/resource constraints.
- `ssh-remote-connect`: uploaded sources, rebuilt the wheel, installed it, and
  ran remote correctness/performance tests.

Bottleneck summary:

- The no-GM-dequant constraint is now satisfied. The dequantized half B tile is
  not the HBM bottleneck anymore.
- Remaining cost is mainly many small public Matmul calls, GM atomic
  accumulation of final C across K chunks, AIV/AIC synchronization around
  `SetTensorB(LocalTensor)`, and Vector unpack/antiquant.
- `K=256,N=128` is the current practical capacity point: it keeps the B tile at
  about 64KB. Attempts to move to 128KB VECOUT B tiles are rejected by tiling or
  performance.

Accepted change:

- Switched the VECOUT Matmul object from `CFG_MDL` to `CFG_NORM`.
- Tile/dataflow remains `MATMUL_K=256,N_TILE=128`; B still stays in VECOUT and
  never goes to GM.

Final remote result:

- Correctness `[16,4096,22016]`: pass at `rtol=8e-2, atol=8e-2`,
  `max_abs=0.046875`, `mean_abs=0.004495220258831978`.
- Candidate repeat=10: avg `362.916302 us`, min `351.460999 us`, max
  `391.397043 us`.
- Vendor repeat=10: avg `132.503733 us`, min `124.302111 us`, max
  `157.230999 us`.
- Candidate repeat=30 final rerun: avg `364.990005 us`, min `355.703058 us`,
  max `422.097975 us`.
- Vendor repeat=30: avg `126.019326 us`, min `120.301964 us`, max
  `168.345054 us`.

Rejected variants:

- `K=512,N=128`: build passed but host `GetTiling failed`.
- Combined scale/offset param queue: correctness passed, but candidate
  repeat=10 was `374.777394 us`, slower than baseline.
- `CFG_NORM + enUnitFlag=false`: correctness passed, but candidate repeat=10
  was `367.278303 us`, slower than default `CFG_NORM`.
- `CFG_MDL + enUnitFlag=true`: correctness passed, but candidate repeat=10
  was unstable and averaged `490.663690 us`.
## 2026-07-06 VECOUT/Cube double-buffer 实验

- 回答并验证当前默认 VECOUT/Cube kernel 是否使用 ping-pong buffer: 之前没有，
  四个显式队列均为 `InitBuffer(..., 1, ...)`。
- 保留优化: 将 `weight4Queue`、`scaleQueue`、`offsetQueue`、`vecoutQueue`
  的 buffer number 调整为 `BufferNum=2`，队列深度仍保持
  `TQue<TPosition::*, 1>`。
- 正确性 `[16,4096,22016]`: pass, `max_abs=0.046875`,
  `mean_abs=0.004495220258831978`。
- 性能 `[16,4096,22016]`:
  - repeat=10: candidate `349.860790 us`, vendor `129.159982 us`,
    ratio `2.7087x`。
  - repeat=30: candidate `351.601928 us`, vendor `131.627242 us`,
    ratio `2.6712x`。
- 相比上一版 `CFG_NORM` repeat=10 `362.916302 us` 和 repeat=30
  `364.990005 us`，candidate 分别提升约 `3.6%` 和 `3.7%`。

| Skill | 用途 | 读取的关键文件 |
|---|---|---|
| `ascendc-operator-performance-optim` | 按流水/双缓冲规则检查现有 queue，并用同目标 shape 对比优化前后性能 | `.agents/skills/ascendc-operator-performance-optim/SKILL.md`, `references/pipeline-prof.md` |
| `ascendc-operator-code-gen` | 修改 kernel queue buffer number 前核对 TQue/TPipe/DataCopyPad 资源和同步约束 | `.agents/skills/ascendc-operator-code-gen/SKILL.md`, `references/sync-control-api.md`, `references/resource-management-api.md`, `references/data-copy-api.md` |
| `ssh-remote-connect` | 上传单个 kernel 文件，在远端 CANN 9.0.0/Ascend910_9382 环境编译、安装、正确性和 wall-clock 测试 | `.agents/skills/ssh-remote-connect/SKILL.md`, `scripts/connect.ps1`, `scripts/remote_exec.py` |

## 2026-07-06 AWQ subgroup unroll 与同步微调

- 保留优化: 手动展开 `MATMUL_K=256` 内的两个 AWQ group，保持
  `scaleLocal0/offsetLocal0` 到 group1 计算结束后统一释放，减少一个中间
  post-Mul barrier/release 点，并移除 `subCol` 单次循环。
- 同时将不变的 `weight4Copy`、`int4Pad`、`aqParams` 移出 K 循环。
- 正确性 `[16,4096,22016]`: pass, `max_abs=0.046875`,
  `mean_abs=0.004495220258831978`。
- 性能:
  - repeat=10 both: candidate `346.333301 us`, vendor `124.493905 us`。
  - repeat=30 both: candidate `346.482347 us`, vendor `122.475457 us`。
  - 最终安装后 candidate-only repeat=30: `349.562642 us`。
- 对比 double-buffer-only repeat=30 `351.601928 us`，both 测试提升约
  `1.5%`；最终 candidate-only 复测仍略快于 double-buffer-only 基线。
- 回退实验:
  - 提前预取 `scale/offset`: 正确性通过，但 candidate repeat=10 回退到
    `363.333209 us`。
  - 用 VECOUT `EnQue/DeQue` 替代最终 post-Mul `PipeBarrier`: 正确性通过，
    但 candidate repeat=10 回退到 `351.491303 us`。

| Skill | 用途 | 读取的关键文件 |
|---|---|---|
| `ascendc-operator-performance-optim` | 继续围绕 Vector 同步、queue 生命周期和同目标 shape 性能做小步实验 | `.agents/skills/ascendc-operator-performance-optim/SKILL.md`, `references/pipeline-prof.md` |
| `ascendc-operator-code-gen` | 核对 PipeBarrier、EnQue/DeQue、TQue buffer 生命周期和 Vector repeat API 后修改 kernel | `.agents/skills/ascendc-operator-code-gen/SKILL.md`, `references/sync-control-api.md`, `references/resource-management-api.md`, `references/vector-compute-api.md` |
| `ssh-remote-connect` | 上传单个 kernel 文件，在远端编译安装并运行正确性、repeat=10/30 wall-clock 测试 | `.agents/skills/ssh-remote-connect/SKILL.md`, `scripts/connect.ps1`, `scripts/remote_exec.py` |

## 2026-07-06 910B2C / Ascend910_9382 资源地图整理

- 远端查询来源:
  - `npu-smi info -l/-m/info`
  - `npu-smi info -t memory/usages/board -i 7 -c 0`
  - `torch_npu.npu.get_device_properties(0)`
  - `/usr/local/Ascend/cann-9.0.0/x86_64-linux/data/platform_config/Ascend910_9382.ini`
- 确认当前设备: physical NPU ID `7`，PyTorch logical device `0`，
  chip `Ascend 910B2C`，CANN SoC `Ascend910_9382`。
- 关键资源: Cube core `24`，Vector core `48`，HBM `64GiB` 物理 /
  torch_npu 可分配约 `62420MiB`，L2 `192MiB`，UB `192KiB`，L1
  `512KiB`，L0A/L0B `64KiB`，L0C `128KiB`。
- 新增 resource-map:
  - `.agents/skills/ascendc-operator-performance-optim/references/ascend910-9382-resource-map.md`
  - `reference/skills/ascendc-operator-performance-optim/references/ascend910-9382-resource-map.md`
- 更新 `ascendc-operator-performance-optim/SKILL.md`: 当遇到
  `GetTiling failed`、`507015`、VECOUT/TSCM 资源边界、Matmul shape
  合法性、queue buffer/workspace/EventID 竞争时，按需加载该资源地图。

| Skill | 用途 | 读取的关键文件 |
|---|---|---|
| `skill-creator` | 将资源经验按 progressive disclosure 方式整理进已有 skill，而不是塞进主说明 | `C:/Users/CHENG/.codex/skills/.system/skill-creator/SKILL.md` |
| `ssh-remote-connect` | 连接远端并执行 npu-smi、torch_npu、CANN 配置查询 | `.agents/skills/ssh-remote-connect/SKILL.md`, `scripts/connect.ps1` |
| `npu-smi` | 确认 NPU ID、芯片名、HBM、board、usage 查询命令 | `.agents/skills/npu-smi/SKILL.md` |
| `ascendc-operator-performance-optim` | 将资源边界整理为后续 tiling/queue/Matmul 调优前置参考 | `.agents/skills/ascendc-operator-performance-optim/SKILL.md` |

## 2026-07-06 static basicK and high-risk route probes

Skills used:

- `ascendc-operator-performance-optim`: used the same-case correctness ->
  performance loop, kept the accepted baseline separate, and rejected unstable
  probes even when short-run timing looked better.
- `ssh-remote-connect`: uploaded only touched host/kernel files and ran remote
  CANN build/install/correctness/performance commands on the Ascend910_9382
  server.

Accepted result:

- Final stable route remains pure custom and does not write the antiquantized B
  tile to GM.
- Kept `MATMUL_K=256,N_TILE=128,BufferNum=2` and changed static Matmul
  `basicK` from `128` to `256`.
- Correctness on `[16,4096,22016]`: pass, `max_abs=0.046875`,
  `mean_abs=0.004495220258831978`, `rtol=8e-2`, `atol=8e-2`.
- Best repeat=30 candidate observed: `298.384465 us`; final rerun:
  `302.789232 us`. Same-run vendor repeat=30: `121.796411 us`.

Rejected route summary:

- `enUnitFlag=false`: pass but slower (`311.878300 us` repeat=10).
- `CONFIG_MDL`: pass but no gain (`308.756716 us` repeat=10).
- K512/N128: pass and fast short-run (`273.307708 us` repeat=3), but repeat=10
  timed out even with `waitIterateAll=true`.
- N256/K256: compiled but correctness failed with `inf`/large diffs.
- block-stride scheduling: correctness script timed out; reusing a Matmul object
  across N tiles is unsafe here.
- A16W8 public API route: compile failed because `SetAntiQuantVector` is not
  available for `BType=VECOUT,int8_t`, and `Cast(int4b_t -> int8_t)` has no
  matching implementation.

Current bottleneck:

- The bottleneck is no longer GM traffic for antiquantized B.
- The public VECOUT/Cube route is dominated by many small Matmul calls and GM
  atomic accumulation: `172 * 16 = 2752` calls for the target shape.
- Short K512 timing proves larger K granularity helps, but public VECOUT
  lifetime/synchronization becomes unstable at that size.

## 2026-07-07 N192 VECOUT/Cube route design

Skills used:

- `ascendc-operator-performance-optim`: reviewed the current N128 bottleneck,
  910B2C resource map, and public Matmul/VECOUT boundary before selecting a new
  route.
- `ascendc-operator-code-gen`: checked `SetTail`, `SetSingleShape`,
  DataCopyPad, Vector repeat, and TQue/TPipe resource constraints before
  modifying code.
- `ssh-remote-connect`: prepared the remote upload/build path, but upload was
  blocked by the execution environment's security review because it would
  transfer source files to the configured SSH host.

Local changes prepared:

- Added an experimental `AWQ_ASCENDC_ROUTE=n192` route.
- Added `awq_w4a16_linear_vecout_n192`, which keeps `MATMUL_K=256`, changes
  main `N_TILE` to 192, uses `basicK=128`, and splits Vector antiquant into
  128-column + 64-column segments.
- Added `colOffset` to the existing N128 VECOUT kernel so it can process a
  final 128-column tail after N192 main tiles.
- Updated `design.md` with the current data path, call-count bottleneck, N192
  assumptions, and promotion criteria.

Expected target-shape effect:

- Current stable route: `172 * 16 = 2752` public Matmul calls.
- N192 route on `[16,4096,22016]`: `(114 + 1) * 16 = 1840` public Matmul calls.
- The route still keeps antiquantized B in VECOUT and does not write the B half
  tile to GM.

Validation status:

- After explicit upload approval, the N192 route compiled and installed, but
  target-shape correctness timed out at 300 seconds. Reducing N192 `basicK`
  from 128 to 64 also compiled and installed, but timed out in the same way.
- Added and tested a more conservative N144 route. It would reduce target
  public Matmul calls from `2752` to `2448`, but it also timed out during
  target-shape correctness.
- Default N128 route was re-validated after the experiments:
  - small `[2,256,256]`: pass, `max_abs=0.0048828125`.
  - target `[16,4096,22016]`: pass, `max_abs=0.046875`,
    `mean_abs=0.004495220258831978`.
  - repeat=30 wall-clock: candidate `296.9040147339304 us`, vendor
    `122.93596131106217 us`, ratio `2.415x`.

Decision:

- Reject N144/N192 public VECOUT routes. The likely boundary is hidden
  public-Matmul/VECOUT local-B handoff/lifetime for B tiles above the stable
  64 KiB N128 tile.
- Keep N128/K256/`basicK=256` as the stable accepted route.

## 2026-07-07 Host autotiling before next route search

Skills used:

- `ascendc-operator-performance-optim`: reloaded the performance workflow and
  910B2C resource map before changing the host route selection.
- `ascendc-operator-code-gen`: checked kernel constraints and queue/sync notes
  before editing host/kernel-facing launch code.
- `ssh-remote-connect`: will be used for remote upload/build/correctness and
  same-case timing after local edits.

Checkpoint:

- Created inner operator-workspace git commit
  `048b32f Add VECOUT AWQ W4A16 optimization baseline` before new changes.

Host autotiling change:

- Relaxed common input validation from VECOUT-only constraints to generic AWQ
  constraints: `K % 128 == 0`, `N % 8 == 0` through the int4pack shape, and
  uint32-safe kernel parameters.
- Added host route selection. `AWQ_ASCENDC_ROUTE` unset or `auto` now checks
  whether the stable VECOUT tile is legal (`M <= 16`, `K % 256 == 0`,
  `N % 128 == 0`) and whether host Matmul `GetTiling` accepts the
  `GM x VECOUT -> GM` tile.
- Cached the host Matmul tiling legality check per process. A repeat benchmark
  initially hung after repeatedly constructing public Matmul tiling inside the
  Python loop; the cache keeps autotiling without paying that API cost every
  call.
- If the VECOUT route is not legal, auto selects the scalar direct custom
  kernel for correctness coverage. This path writes only the final output and
  does not materialize a half dequantized B tile in GM.
- Explicit `stream` and `split` routes remain diagnostic only; auto does not
  choose them because they use GM scratch for an antiquantized half tile.
