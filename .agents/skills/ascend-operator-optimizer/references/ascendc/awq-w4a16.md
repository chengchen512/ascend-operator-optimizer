# Ascend C AWQ W4A16 Case Pack

读取本文件的场景：

- 用户优化 AWQ / W4A16 / weight-only quantized linear / int4pack matmul。
- 目标约束包括不调用 vendor WQMM、不使用 target-shape fast path、不把反量化 B 中间结果写回 GM。
- 遇到 `SetTensorB(LocalTensor)`、`VECOUT`、`TSCM`、`GetTiling failed`、`507015`、AIV/AIC handoff 或 public Matmul 小块调用开销。

## 已验证目标

Dense AWQ W4A16 Linear：

```text
x:                [M,K] fp16
weight:           [K,N/8] int32 int4pack
antiquant_scale:  [K/128,N] fp16
antiquant_offset: [K/128,N] fp16
y:                [M,N] fp16
```

默认语义：

```text
y = x @ ((int4_unpack(weight) + offset) * scale)
group_size = 128
bias = None
zero_point = True
```

## 硬约束

- 不调用 `torch_npu.npu_weight_quant_batchmatmul` 实现 candidate。
- 不加入只服务单个目标 shape 的 fast path。
- 不使用常驻 half dequant cache。
- 不自动 fallback 到会把 half dequantized B 写入 GM 的路线。
- 正确性未过时不跑性能；性能结论必须绑定同一 shape、warmup、repeat、计时方式和 vendor baseline。

## 推荐当前路线

在公开 AscendC API 下，优先使用：

```text
int4 weight GM
  -> DataCopyPad row-strided tile into VECIN
  -> Cast int4 -> half into VECOUT
  -> DataCopyPad scale/offset into VECIN
  -> Vector Add/Mul antiquant in VECOUT
  -> matmul.SetTensorB(VECOUT LocalTensor)
  -> public Matmul internal VECOUT -> Cube path
  -> IterateAll final y GM, atomic for later K chunks
```

已验证稳定 tile：

```text
M <= 16
N_TILE = 128
MATMUL_K = 256
AWQ groups per Matmul = 2
BasicK = 256
explicit queue BufferNum = 2
```

Host 侧应做 autotiling：

- `auto` 仅在 `M<=16, K%256==0, N%128==0` 且 host Matmul `GetTiling` 接受时选择 VECOUT/Cube。
- 对其它 shape，fallback 到 scalar direct custom compute 以保证 API 覆盖；该 fallback 只写最终输出，不写 GM dequant B。
- `stream` / `split` 等 GM scratch 方案只能显式诊断，不应被 auto 选择。
- Host Matmul tiling legality check 要按 shape/tile 缓存；不要在 benchmark loop 每次调用都重新构造 public tiling。

## 910B2C 经验边界

目标环境：Ascend 910B2C / CANN 9.0.0 / `Ascend910_9382`。

有用的资源事实：

- AIC 24，AIV 48，split Cube/Vector。
- UB 192 KiB，L1 512 KiB，L0B 64 KiB，L0C 128 KiB。
- 当前 N128/K256 route 的显式 queue 约 161 KiB（BufferNum=2），剩余 UB 很少，不要再叠大 VECCALC buffer。

已验证边界：

| Route | 结果 | 决策 |
| --- | --- | --- |
| K256/N128 VECOUT/Cube | 正确，target repeat=30 约 298 us | 当前稳定 baseline |
| K512/N128 | 短测快，但 repeat 会 timeout | 拒绝，疑似 local B 生命周期/同步边界 |
| K384 main + K256 tail | 可短测，repeat timeout | 拒绝 |
| K256/N256 | 编译/运行后结果错误或 `inf` | 拒绝 |
| N144/N192 wide-N | 编译通过，target correctness timeout | 拒绝 |
| `stream` GM scratch | 正确但写 GM dequant B | 只作诊断 |
| `split` offset term | 正确但变慢，且额外 torch mm | 拒绝默认 |

## TSCM 结论

官方 TSCM Matmul 可用的前提是输入矩阵对单核全载并满足 NZ 格式约束。

已验证：

- `GM(pre-NZ)->TSCM->Cube` 的最小官方式 probe 可通过。
- 普通 ND half tile 不能直接作为 TSCM Matmul 输入。
- AWQ AIV 侧直接 `VECIN/VECOUT -> TSCM -> AIC SetTensorB` 当前会 timeout 或 `507015`。
- `507015` 优先排查 UB/L1 地址越界、InitBuffer 大小、队列生命周期、对齐和 TSCM full-load/NZ 假设。

因此，除非先写出独立最小 handoff probe 并通过 correctness/repeat benchmark，不要把 TSCM AWQ handoff 晋升为主路线。

## 性能瓶颈

当前 no-GM-dequant route 的主要瓶颈不是 half B HBM 中间结果，而是：

- `N/128 * K/256` 次 public Matmul 调用。
- 每个 K chunk 后对 GM 输出做 atomic accumulation。
- `SetTensorB(VECOUT LocalTensor)` 的 public Matmul handoff 开销。

对 `[16,4096,22016]`：

```text
N tiles = 172
K chunks = 16
Matmul calls = 2752
vendor ≈ 123 us
candidate ≈ 298 us
```

要接近 vendor，下一步应优先寻找 lower-level fused AIV/AIC 或官方 local int8/int4 antiquant API，目标是减少 public Matmul 小调用和 GM atomic，而不是回到 GM dequant cache。

## 推进规则

新路线必须满足：

1. 不写 GM dequant B。
2. 先通过小 shape correctness，再跑 target correctness。
3. target correctness 通过后再跑 repeat benchmark。
4. 若出现单次通过、repeat timeout，按 AIV/AIC 同步或 LocalTensor 生命周期问题处理，不晋升。
5. 失败 probe 保留为 evidence，但不要污染默认 route。
