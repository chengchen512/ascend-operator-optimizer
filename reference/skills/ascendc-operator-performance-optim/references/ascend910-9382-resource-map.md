# Ascend910_9382 / 910B2C Resource Map

Use this reference when optimizing AscendC kernels on the remote 910B2C
environment, especially when `GetTiling failed`, `507015`, VECOUT/TSCM handoff
timeouts, queue buffer changes, or public Matmul shape changes appear.

This is not a full internal Huawei resource manual. It combines values verified
from the current machine with CANN 9.0.0 platform config and empirical AWQ
W4A16 boundary results. Treat hidden Matmul API resource use as black-box until
validated by compile, correctness, and timing.

## Verified Environment

Commands used:

```bash
npu-smi info -l
npu-smi info -m
npu-smi info
npu-smi info -t memory -i 7 -c 0
npu-smi info -t usages -i 7 -c 0
npu-smi info -t board -i 7
python - <<'PY'
import torch, torch_npu
print(torch_npu.npu.get_device_name(0))
print(torch_npu.npu.get_device_properties(0))
print(torch_npu._C._npu_get_soc_version())
PY
sed -n '1,90p' /usr/local/Ascend/cann-9.0.0/x86_64-linux/data/platform_config/Ascend910_9382.ini
```

Snapshot:

| Item | Value |
|---|---:|
| Physical NPU ID | `7` |
| PyTorch logical device | `0` |
| Chip name | `Ascend 910B2C` / `Ascend910B2C` |
| CANN SoC version used for build | `Ascend910_9382` |
| Short SoC version | `Ascend910_93` |
| NPU arch | `2201` |
| AIC compiler target | `dav-c220-cube` |
| AIV compiler target | `dav-c220-vec` |
| CANN path | `/usr/local/Ascend/cann-9.0.0` |
| npu-smi version | `26.0.rc1` |
| Board product | `IT21HMDB02-B2` |
| Firmware | `9.0.0.0.205` |

Important mapping: `npu-smi` uses physical card ID `7`, while `torch.npu` sees
one logical device at ordinal `0`. Use `npu-smi info -l` before forming `-i`.

## Core and Memory Resources

From `Ascend910_9382.ini` and `torch_npu.npu.get_device_properties(0)`:

| Resource | Value | Notes |
|---|---:|---|
| AI core count | `24` | CANN `ai_core_cnt` |
| Cube core count | `24` | `cube_core_cnt`, `cube_core_num` |
| Vector core count | `48` | `vector_core_cnt`, `vector_core_num`, `multi_processor_count` |
| Core organization | split Cube/Vector | `cube_vector_combine=split` |
| AI CPU count | `6` | CANN `ai_cpu_cnt` |
| Physical HBM | `64 GiB` | `memory_size=68719476736`, `npu-smi HBM Capacity=65536 MB` |
| torch_npu allocatable memory | `62420 MiB` | `total_memory=65452113920`; lower than physical due reserve |
| L2 cache | `192 MiB` | `l2_size=201326592`; torch props match |
| UB usable size | `192 KiB` | `ub_size=196608` per AI Core |
| L1 size | `512 KiB` | `l1_size=524288` per AI Core |
| L0A size | `64 KiB` | `l0_a_size=65536` |
| L0B size | `64 KiB` | `l0_b_size=65536` |
| L0C size | `128 KiB` | `l0_c_size=131072` |
| BT buffer | `1 KiB` | `bt_size=1024` |
| UB block size | `32 B` | `ubblock_size=32`; align UB/L1 addresses to 32B |
| UB bank size / count | `4096 B * 64` | Do not treat product as usable UB; use `ub_size` for capacity |
| UB bank groups | `16` | Useful when diagnosing bank conflicts |
| Cube frequency | `1800 MHz` | `cube_freq=1800` |
| Vector calc size | `128` | `vec_calc_size=128` |
| BF16 support | yes | `support_bf16=1` |
| Fixpipe support | yes | `support_fixpipe=1` |
| Sparsity support | yes | `sparsity=1` |

Mmad primitive shape hints from `[DtypeMKN]`:

| dtype | M,N,K |
|---|---|
| Default fp16-style | `16,16,16` |
| INT8 / UINT8 | `16,32,16` |
| INT4 | `16,64,16` |
| INT2 / UINT2 | `16,128,16` |
| UINT1 | `16,256,16` |

## Queue and Workspace Constraints to Apply

Use public API rules plus current environment observations:

- TQue `depth` and `InitBuffer(num, len)` are different. Keep `depth=1` unless
  there are consecutive EnQue calls without DeQue. Use `num=2` for double
  buffer/ping-pong resources.
- On Atlas A2/A3 style products, EventID count is 8 and TQue buffer count per
  queue should stay <= 8. Keep the sum of all explicit TPipe buffers <= 64.
- `InitBuffer` length is rounded to 32B. Still do explicit 32B alignment in
  sizing formulas.
- TSCM with Matmul competes with synchronization/event resources. For TSCM
  sourced from VECIN, keep `TSCM buffers + Matmul objects <= 10` unless a
  specific official sample proves otherwise.
- TSCM Matmul input must be fully resident for the single-core computation and
  uses NZ format constraints. Do not feed ordinary ND UB data to TSCM Matmul
  without a validated ND->NZ path.
- `GetLibApiWorkSpaceSize()` is required for public Matmul high-level API
  workspace. Allocate at least this system workspace before `REGIST_MATMUL_OBJ`.

Explicit UB queue budget formula:

```text
explicit_ub_bytes = sum(queue_buffer_num_i * per_buffer_len_i)
```

For the accepted AWQ VECOUT route (`K=256,N=128`):

| Queue | Per-buffer bytes | BufferNum=1 | BufferNum=2 |
|---|---:|---:|---:|
| `weight4Queue` VECIN | `256*128/2 = 16 KiB` | 16 KiB | 32 KiB |
| `scaleQueue` VECIN | `128*sizeof(half) = 256 B` | 256 B | 512 B |
| `offsetQueue` VECIN | `128*sizeof(half) = 256 B` | 256 B | 512 B |
| `vecoutQueue` VECOUT | `256*128*sizeof(half) = 64 KiB` | 64 KiB | 128 KiB |
| Total explicit queues |  | about 80.5 KiB | about 161 KiB |

This `BufferNum=2` case passes on `Ascend910_9382`, but it leaves only about
31 KiB of the 192 KiB UB before hidden API/runtime bookkeeping. Do not add
large VECCALC buffers on top without reducing tile size or validating first.

## Empirical Matmul/VECOUT Boundaries from AWQ W4A16

Target case: `[M,K,N]=[16,4096,22016]`, W4A16 AWQ, no GM dequant tile.

| Shape / route | Result | Interpretation |
|---|---|---|
| `K128,N256` VECOUT/Cube | correct, about `466.85 us` short run | works but many K atomics |
| `K256,N128`, `CFG_MDL` | correct, `371.54 us` repeat=10 | K merge reduces C atomic count |
| `K256,N128`, `CFG_NORM` | correct, `362.92 us` repeat=10 | current public Matmul config baseline before double buffer |
| `K256,N128`, `CFG_NORM`, `BufferNum=2` | correct, `349.86 us` repeat=10; `351.60 us` repeat=30 | keep; queue ping-pong resources help about 3.6-3.7% |
| `K256,N256` | host `GetTiling failed` | 128 KiB B tile shape rejected by public tiling/resource path |
| `K512,N128` | build passed, host `GetTiling failed` | larger K with same N also rejected |
| `K512,N64` | correct but about `513.91 us` | N too narrow; public Matmul shape inefficient |
| `async IterateAll<false>` | one correctness pass, repeat timing timeout | hidden AIV/AIC sync/lifetime constraint likely |
| TSCM/NZ probes | timeout or `507015` in AWQ handoff | TSCM path needs stricter full-load/NZ/event discipline |

Do not infer legality only from explicit UB size. Some rejected shapes appear
close to the 192 KiB UB limit, but the failing check happens in public
`TCubeTiling`/Matmul resource planning and includes L1/L0B/workspace/internal
handoff constraints that are not fully exposed.

## Optimization Procedure for Unknown Boundaries

When a new tile or queue plan is needed:

1. Record baseline: route, `M/N/K`, `MatmulType` positions, `CFG_*`, queue
   depth, `InitBuffer` number/length, workspace bytes, blockDim, correctness,
   and timing.
2. Change exactly one axis: `K`, `N`, queue buffer number, Matmul config, or
   async flag.
3. Call host `GetTiling` first. Treat `GetTiling failed` as a resource/shape
   boundary, not as kernel C++ logic failure.
4. If kernel launches but returns `507015`, inspect UB/L1 address sizing,
   queue lifecycle, DataCopyPad alignment, and TSCM full-load/NZ assumptions.
5. If single correctness passes but repeat timing hangs, suspect AIV/AIC
   synchronization or LocalTensor lifetime around `SetTensorB(LocalTensor)`.
6. Promote only variants that pass correctness before timing and improve the
   same benchmark case.

Suggested log row:

```text
date | soc | route | M,N,K | Apos/Bpos/Cpos | format | CFG | queue bytes |
workspace | blockDim | GetTiling | correctness | avg/min/max us | error
```

## Commands for Future Agents

Remote quick check:

```powershell
.\.agents\skills\ssh-remote-connect\scripts\connect.ps1 `
  "npu-smi info -l && npu-smi info -m && npu-smi info"
```

Detailed physical card query, after replacing `7` with the ID from
`npu-smi info -l`:

```powershell
.\.agents\skills\ssh-remote-connect\scripts\connect.ps1 `
  "npu-smi info -t memory -i 7 -c 0; npu-smi info -t usages -i 7 -c 0; npu-smi info -t board -i 7"
```

Runtime properties:

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source /root/ascend-operator-optim/.venv/bin/activate
python - <<'PY'
import torch, torch_npu
print(torch_npu.npu.get_device_name(0))
print(torch_npu.npu.get_device_properties(0))
print(torch_npu._C._npu_get_soc_version())
PY
```

CANN platform config:

```bash
sed -n '1,90p' /usr/local/Ascend/cann-9.0.0/x86_64-linux/data/platform_config/Ascend910_9382.ini
```
