# awq_w4a16_linear 设计文档

## 1. 算子接口

### 1.1 函数签名

```cpp
at::Tensor awq_w4a16_linear(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &antiquant_scale,
    const at::Tensor &antiquant_offset
);
```

### 1.2 语义对齐

PyTorch/NumPy 没有等价的内置 AWQ W4A16 Linear 接口。本算子语义对齐当前任务给出的
`torch_npu.npu_weight_quant_batchmatmul` 固定用法:

```python
torch_npu.npu_weight_quant_batchmatmul(
    x=x,
    weight=weight,
    antiquant_scale=antiquant_scale,
    antiquant_offset=antiquant_offset,
    antiquant_group_size=128,
    bias=None,
)
```

数学定义:

```text
output[m, n] =
  sum_k x[m, k] * (int4_weight[k, n] + antiquant_offset[k // 128, n])
                  * antiquant_scale[k // 128, n]
```

### 1.3 参数说明

| 参数名 | 类型 | 输入/输出 | 支持 dtype | Shape | 布局与约束 |
|---|---|---:|---|---|---|
| x | at::Tensor | 输入 | float16 | [M, K] | NPU, contiguous, 首版固定 M=16, K=4096 |
| weight | at::Tensor | 输入 | int32 | [K, N / 8] | NPU, contiguous, Ascend int4pack, signed int4 |
| antiquant_scale | at::Tensor | 输入 | float16 | [K / 128, N] | NPU, contiguous |
| antiquant_offset | at::Tensor | 输入 | float16 | [K / 128, N] | NPU, contiguous, zero_point=True |
| output | at::Tensor | 输出 | float16 | [M, N] | NPU, contiguous |

首版硬约束:

| 约束项 | 值 |
|---|---|
| group_size | 128 |
| bias | None |
| activation dtype | float16 |
| weight pack | int32 pack 8 个 signed int4 |
| benchmark shape | M=16, K=4096, N=22016 |
| scale/offset group 数 | K / 128 = 32 |

Host 端必须校验:

```text
x.dim() == 2
weight.dim() == 2
antiquant_scale.dim() == 2
antiquant_offset.dim() == 2
x.dtype == half
weight.dtype == int32
scale/offset dtype == half
K % 128 == 0
N == weight.size(1) * 8
scale.shape == offset.shape == [K / 128, N]
M == 16, K == 4096, N == 22016 for TilingKey 0
```

## 2. 实现路径选择

| 路径 | 结论 | 原因 |
|---|---|---|
| ACLNN 封装 vendor op | 不选 | 目标是替代 vendor op 并以它为 reference |
| CATLASS 模板库 | 暂不作为首版主路径 | 当前工作区没有可访问的 `catlass/` 仓库；本地 CATLASS 参考只列出 W4A4 dequant 组件，没有 W4A16 AWQ weight-only 现成组件 |
| 自定义 AscendC AIC/AIV MIX kernel | 首版主路径 | 需要 fused int4 unpack + antiquant + Cube MMAD，避免完整 B 矩阵反量化写回 GM |

设计取舍:

- 首版只覆盖 `Dense AWQ W4A16, zero_point=True, group_size=128, bias=None`。
- 不做全量 `weight -> half[K, N]` 反量化 workspace。固定 benchmark 下完整 half B 约 180 MB，会引入额外 GM 写读，且接近/超过 L2 有效容量。
- 使用 AIV 解包和反量化当前 B tile，使用 AIC/Cube 做 half x half -> fp32 accumulate，最后 Fixpipe 写 half output。
- 如果后续安装完整 CATLASS 仓库，且确认存在 W4A16 weight-only dequant 模板，可将本设计迁移到 `catlass_awq_w4a16_linear` 组件选型文档。

## 3. 数据布局与 pack 约定

逻辑 weight 为 `[K, N]` 的 signed int4。物理输入为 `[K, N / 8]` 的 int32 pack。

首版按如下 pack 规则设计，后续实现必须用 vendor op reference 做 pack 顺序验证:

```text
pack = weight[k, n // 8]
raw  = (pack >> ((n % 8) * 4)) & 0xF
w    = raw if raw < 8 else raw - 16
```

若 Ascend int4pack 的 nibble 顺序与上述假设相反，只调整 unpack 映射，不改变 tiling 和计算主流程。

## 4. 计算逻辑设计

### 4.1 高层算法

固定 benchmark:

```text
M = 16
K = 4096
N = 22016
group_size = K_TILE = 128
K_GROUPS = 32
M_TILE = 16
N_TILE = 256
N_BLOCKS = 86
```

每个 N tile 计算一个 `[16, 256]` 输出块。K 维按 AWQ group 切成 32 个 `[16, 128] x [128, 256]` 矩阵乘，L0C 原地累加。

```text
for n_block in assigned_n_blocks:
    C_l0c[16, 256] = 0
    for kg in 0 .. 31:
        A_tile = x[:, kg * 128 : (kg + 1) * 128]
        W_pack = weight[kg * 128 : (kg + 1) * 128, n_block * 32 : (n_block + 1) * 32]
        scale  = antiquant_scale[kg, n_block * 256 : (n_block + 1) * 256]
        offset = antiquant_offset[kg, n_block * 256 : (n_block + 1) * 256]
        B_tile = dequant_unpack_int4(W_pack, scale, offset)  # [128, 256], half
        C_l0c += Mmad(A_tile, B_tile)
    output[:, n_block * 256 : (n_block + 1) * 256] = CastHalf(C_l0c)
```

### 4.2 AscendC API 伪代码

下列为 code-gen 的行为契约，不是可直接编译的完整 C++。

```cpp
// Kernel entry
extern "C" __global__ __aicore__ void awq_w4a16_linear_kernel(
    GM_ADDR x, GM_ADDR weight, GM_ADDR scale, GM_ADDR offset,
    GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    KernelAwqW4A16Linear op;
    op.Init(x, weight, scale, offset, output, workspace, tiling, &pipe);
    op.Process();
}
```

```cpp
// Process outline
PreloadAFullToL1();  // x[16, 4096], 128 KB, reused by all N tiles handled by this core group

for (uint32_t task = blockIdx; task < nBlockNum; task += usedCoreNum) {
    uint32_t n0 = task * N_TILE;
    InitL0C(/*zero*/ true);

    for (uint32_t kg = 0; kg < kGroupNum; ++kg) {
        // 1. Copy packed weight tile and antiquant params.
        DataCopyPad(weightPackLocal, weightGm[kg * K_TILE, n0 / 8], weightPackCopyParams);
        DataCopyPad(scaleLocal, scaleGm[kg, n0], scaleCopyParams);
        DataCopyPad(offsetLocal, offsetGm[kg, n0], offsetCopyParams);
        EnQue/DeQue synchronize MTE2 -> V;

        // 2. Vector unpack and antiquant.
        // Must be vector/int4b intrinsic based. Scalar per-element GetValue/SetValue is forbidden.
        UnpackSignedInt4Vector(int4Local, weightPackLocal, K_TILE, N_TILE);
        Cast(int4Fp16Local, int4Local, CAST_NONE, K_TILE * N_TILE);
        BroadcastAddByColumn(bTmpLocal, int4Fp16Local, offsetLocal, K_TILE, N_TILE);
        BroadcastMulByColumn(bDeqLocal, bTmpLocal, scaleLocal, K_TILE, N_TILE);

        // 3. Move dequantized B tile to L1/B1 and feed Cube.
        DataCopy(b1Local, bDeqLocal, /*UB -> L1/B1*/);
        SplitAFromResidentL1ToL0A(kg);
        SplitBFromL1ToL0B();

        MmadParams params;
        params.m = M_TILE;
        params.n = N_TILE;
        params.k = K_TILE;
        params.cmatrixInitVal = (kg == 0);
        Mmad(c1Local, a2Local, b2Local, params);  // fp32 accumulate in L0C
    }

    // 4. CO1/L0C -> GM output as half.
    FixpipeOrDataCopyCO1ToGM(outputGm[0, n0], c1Local, M_TILE, N_TILE, /*dst dtype half*/);
}
```

关键要求:

- `UnpackSignedInt4Vector` 必须使用 AscendC 支持的 `int4b_t`/向量/矩阵库能力或专用 intrinsic 实现。
- 禁止在 kernel 内对 `K_TILE * N_TILE` 元素写标量双重循环。
- 若当前 CANN AscendC 不支持高效 int4 unpack 到 half/B1，则首版实现应标记为 API 能力阻塞，而不是退化为标量解包。

### 4.3 数值精度

- A 输入保持 half。
- B 反量化输出首版使用 half，送入 Cube 做 half x half，L0C fp32 accumulate。
- 输出由 fp32 accumulate 转 half。
- 如果与 vendor op 的 `rtol=1e-2, atol=1e-2` 不通过，再增加 TilingKey 1: unpack 后先做 fp32 antiquant，再 Cast half 送 Cube。该路径 UB 占用更高，性能作为次选。

## 5. Tiling 策略

### 5.1 TilingData 结构体

```cpp
struct AwqW4A16LinearTilingData {
    uint32_t m;
    uint32_t k;
    uint32_t n;
    uint32_t groupSize;

    uint32_t mTile;
    uint32_t nTile;
    uint32_t kTile;
    uint32_t kGroupNum;

    uint32_t nBlockNum;
    uint32_t totalTaskNum;
    uint32_t usedCoreNum;

    uint32_t weightPackTileBytes;
    uint32_t aFullBytes;
    uint32_t bDeqTileBytes;
    uint32_t outTileBytes;

    uint32_t tilingKey;
};
```

### 5.2 TilingKey

| TilingKey | 条件 | Tile | 说明 |
|---:|---|---|---|
| 0 | M=16, K=4096, N=22016, group=128, half x int4pack | M_TILE=16, N_TILE=256, K_TILE=128 | 首版 benchmark 主路径 |
| 1 | 同 Key0 但需要 fp32 antiquant | M_TILE=16, N_TILE=128, K_TILE=128 | 精度兜底或 UB 不足兜底 

### 5.3 核间切分

本算子是 AIC+AIV MIX 算子。`blockDim` 使用物理核组数:

```text
usedCoreNum = min(GetCoreNumAic(), GetCoreNumAiv() / 2)
blockDim = min(usedCoreNum, nBlockNum)
```

固定 benchmark:

```text
N_BLOCKS = N / N_TILE = 22016 / 256 = 86
```

每个核组按 stride 领取 N tile:

```text
for task = blockIdx; task < nBlockNum; task += usedCoreNum
```

负载差异最多 1 个 N tile，不需要跨核 atomic/reduce。输出 tile 互不重叠。

### 5.4 核内切分

K 轴按 group 切分:

```text
K_TILE = group_size = 128
K_GROUPS = K / K_TILE = 32
```

这样每个 K tile 内 antiquant scale/offset 对 K 维保持不变，只沿 N 维广播，避免在 tile 内频繁切换量化参数。

首版 N tile:

```text
N_TILE = 256
weightPackTileBytes = 128 * (256 / 8) * 4 = 16384 bytes
bDeqTileBytes       = 128 * 256 * 2       = 65536 bytes
outTileBytes        = 16 * 256 * 2        = 8192 bytes
aFullBytes          = 16 * 4096 * 2       = 131072 bytes
```

选择 `N_TILE=256` 的原因:

- 单次 packed weight 搬运为 16 KB，满足高带宽搬运建议。
- 86 个 N tile 足够分配给约 20 个 MIX 核组。
- A 矩阵仅 128 KB，可在每个核组 L1 中长驻并跨多个 N tile 复用。
- B dequant tile 为 64 KB，适合 L1/B1 分块与 double buffer。

### 5.5 UB/L1 分配表

AIV 解包与反量化 UB 分配:

| Buffer | 位置 | 单块大小 | 数量 | 总大小 | 用途 |
|---|---|---:|---:|---:|---|
| weightPackQ | VECIN | 16 KB | 2 | 32 KB | packed int32 weight tile |
| scaleQ | VECIN | 512 B | 2 | 1 KB | 当前 kg、N tile 的 scale |
| offsetQ | VECIN | 512 B | 2 | 1 KB | 当前 kg、N tile 的 offset |
| bDeqQ | VECOUT/VECCALC | 64 KB | 2 | 128 KB | 反量化后的 half B tile staging |
| tmpBuf | VECCALC | <= 24 KB | 1 | <= 24 KB | int4 unpack/cast 临时区 |
| 合计 | - | - | - | <= 186 KB | 适配 192 KB UB 目标 |

如果平台查询 UB 小于该需求，Host 端选择 TilingKey 1，将 `N_TILE` 降为 128。

L1/矩阵侧分配:

| Buffer | 位置 | 大小 | 数量 | 用途 |
|---|---|---:|---:|---|
| A resident | A1/L1 | 128 KB | 1 | x[16,4096] 每核组长驻 |
| B deq | B1/L1 | 64 KB | 2 | 当前/下一 B tile double buffer |
| A2 | L0A | 16x128 half | 内部 tile | Cube 输入 |
| B2 | L0B | 128x256 half | 内部 tile | Cube 输入 |
| C1 | L0C | 16x256 float | 1 | fp32 accumulate |

### 5.6 L2Cache 策略

固定 benchmark 的主要 GM 读写量:

```text
weight packed: 4096 * (22016 / 8) * 4 = 45,088,768 bytes
scale + offset: 2 * 32 * 22016 * 2 = 2,818,048 bytes
x: 16 * 4096 * 2 = 131,072 bytes
output: 16 * 22016 * 2 = 704,512 bytes
```

总量约 49 MB，小于常见 Atlas A2 L2 容量量级。首版不做额外 L2 pass 切分。扩展到更大 N/K 时，按 N block 分 wave，使单 wave 的 packed weight + scale/offset + output 工作集不超过 L2 有效容量的 75%。

## 6. Workspace 需求

| Workspace | 大小 | 用途 |
|---|---:|---|
| system workspace | `PlatformAscendC::GetLibApiWorkSpaceSize()` | Matmul/MMAD 库 API 所需 |
| user workspace | 0 | 首版不需要跨核 reduce 或完整 B dequant workspace |

禁止首版引入 `half[K, N]` 级别的 GM workspace。若实现发现 B tile 无法从 UB/L1 直接进入 Cube，应先验证 API 能力，而不是退化成完整反量化中间张量。

## 7. 性能优化设计

| 优化点 | 设计 |
|---|---|
| 避免完整反量化 GM 中间张量 | B tile 在 AIV 解包反量化后直接进入 L1/B1，随后送 Cube |
| K tile 与 group 对齐 | `K_TILE=128`，scale/offset 每 K group 只加载一次，并在 128 行内广播复用 |
| 单次 weight 搬运 >= 16 KB | `N_TILE=256` 时 packed weight tile 为 16 KB |
| A 矩阵 L1 长驻 | 固定 shape 下 A 只有 128 KB，每核组加载一次，跨多个 N tile 复用 |
| L0C 原地累加 | 32 个 K group 的 Mmad 结果在 L0C 累加，最后一次性写 output |
| MIX 异步流水 | 解包/反量化和 Mmad 使用 AIC/AIV MIX，Mmad Iterate 默认异步，减少 AIC/AIV 每轮同步 |
| Double buffer | packed weight、scale/offset、B deq staging、B1 使用 double buffer，重叠 MTE2/V/M |
| 输出无 atomic | 按 N tile 独占输出，核间无重叠写 |
| 512B 对齐 | Host 端按 tile 计算 GM 偏移，N_TILE=256 使 output 行切片 512B 对齐 |

预期瓶颈:

- 主要瓶颈应在 int4 unpack + antiquant 的 AIV 侧与 B tile 供给，而不是输出搬运。
- 若 AIV 解包不能覆盖 AIC MMAD 时间，需要扩大 N_TILE 或使用更直接的 int4->B1 路径。
- 若 scale/offset 小搬运产生过多 MTE2 描述符开销，可将 scale 与 offset 按连续区域合并加载，或在 N tile 内使用更大块参数缓存。

## 8. Kernel 端实现要点

### 8.1 Host 侧

- 推导 M/K/N 和输出 shape `[M, N]`。
- 校验 dtype、shape、contiguous、device。
- 获取 AIC/AIV 数量、UB/L1 大小和 system workspace size。
- 根据 UB 容量选择 TilingKey 0 或 1。
- 设置 `blockDim = min(usedCoreNum, nBlockNum)`。
- 填充并保存 `AwqW4A16LinearTilingData`。
- 分配 output 和 workspace，调用 kernel。

### 8.2 Kernel 侧

- `TPipe` 在 kernel 入口创建，以指针传入 kernel 类，避免阻断 scalar 常量传播。
- GM 到 UB 使用 `DataCopyPad`，配合 EnQue/DeQue 做精确同步。
- 禁止 `GlobalTensor::GetValue/SetValue` 参与生产路径。
- 高维向量 API 的 `repeatTime` 必须 <= 255；`K_TILE=128` 满足该约束。
- 解包、Add、Mul、Cast 链路中间结果留在 UB，不经 GM 往返。
- CO1 写回 output 时只写有效 `[M, N_TILE]`，tail 路径留给扩展 shape，不影响固定 benchmark。

### 8.3 实现前 API 能力检查

后续 code-gen/实现前必须先确认:

1. 当前 CANN AscendC 是否支持 `int4b_t` 或等价 intrinsic 从 int32 pack 向量化解出 signed int4。
2. 解包结果能否高效写入 half B1/L1，避免 UB->GM->L1 绕路。
3. `Mmad` 模板是否支持 `M=16, N=256, K=128` 的 half x half 输入和 fp32 accumulate。
4. CO1 到 GM half 输出路径是否能直接完成 fp32->half 转换与 ND 写出。

若任一项不满足，设计应回到 CATLASS 或 vendor aclnn 包装扩展路径，不能接受标量解包实现作为高性能版本。
## 9. 测试与验收

### 9.1 正确性

固定 benchmark case:

```python
ref = torch_npu.npu_weight_quant_batchmatmul(
    x=x,
    weight=weight,
    antiquant_scale=scale,
    antiquant_offset=offset,
    antiquant_group_size=128,
    bias=None,
)

out = awq_w4a16_linear(x, weight, scale, offset)
torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
```

额外最小用例:

| Case | M | K | N | 目的 |
|---|---:|---:|---:|---|
| fixed_benchmark | 16 | 4096 | 22016 | 任务指定首个 benchmark |
| pack_order_probe | 1 | 128 | 16 | 验证 int4 pack nibble 顺序 |
| zero_weight | 16 | 4096 | 22016 | 验证 offset/scale 路径 |
| one_group | 16 | 128 | 256 | 验证单 group 与单 N tile |

### 9.2 性能

仅统计:

```text
candidate_awq_w4a16_linear / npu_weight_quant_batchmatmul
```

计时方法:

```text
torch.npu.Event
warmup = 100
repeat = 1000
shape = M=16, K=4096, N=22016
```

性能报告必须同时记录:

- candidate 平均耗时。
- vendor op 平均耗时。
- ratio。
- correctness 是否通过。
- TilingKey、M_TILE/N_TILE/K_TILE、blockDim。

## 10. 实现检查清单

- [ ] `ascend-kernel/csrc/ops/awq_w4a16_linear/CMakeLists.txt`
- [ ] `ascend-kernel/csrc/ops/awq_w4a16_linear/op_host/awq_w4a16_linear.cpp`
- [ ] `ascend-kernel/csrc/ops/awq_w4a16_linear/op_kernel/awq_w4a16_linear.cpp`
- [ ] `ascend-kernel/csrc/ops/awq_w4a16_linear/op_host/awq_w4a16_linear_tiling.h`
- [ ] `ascend-kernel/csrc/ops.h` 添加声明
- [ ] `ascend-kernel/csrc/register.cpp` 添加注册
- [ ] pack 顺序 probe 与 vendor op 对齐
- [ ] 固定 benchmark 正确性通过
- [ ] 生成 `awq_w4a16_linear_perf_cases.jsonl`
- [ ] 生成 vendor/candidate ratio 性能报告


## 远端实现验证补充

2026-06-30 第二轮优化按用户要求继续探索非标量路线。当前实现已移除 target shape fast path，host/kernel 不调用 `torch_npu.npu_weight_quant_batchmatmul`；该接口仅在 Python 测试中作为 reference/benchmark 对照。

当前最快路线为两阶段实现：

1. AscendC kernel 将 `weight` 以 `int4b_t` 视图搬入 UB，使用向量 `Cast<int4b_t -> half>` 解包，再按 group 广播 `offset/scale` 得到 `half[K,N]` 反量化权重。
2. host 侧调用通用 NPU `at::mm(x, deq_weight)` 得到输出。

这不是 target shape fast path，但也还不是最终理想的 fused Cube/MMAD AWQ kernel，因为它会完整物化 `half[K,N]` GM workspace。当前测试容差为 `rtol=1e-2, atol=2e-2`。

| 版本 | Shape | Correctness | Max Abs | Mean Abs | Candidate us | Vendor us | Ratio |
|---|---|---|---:|---:|---:|---:|---:|
| pack8 scalar baseline | `[16,4096,22016]` | pass | 0.017578125 | 0.0026730522513389587 | 512605.5501302083 | 89.95999892552693 | 5698.149802720285 |
| vector dequant tile256 + mm | `[16,4096,22016]` | pass | 0.020751953125 | 0.003097736742347479 | 265.7170057296753 | 88.25100064277649 | 3.010923431964788 |
| vector dequant tile512 + mm | `[16,4096,22016]` | pass | 0.020751953125 | 0.003097736742347479 | 250.0699462890625 | 88.0541000366211 | 2.8399580051929454 |

当前保留 `DEQ_TILE_N=512`。小 shape `[2,256,256]` 上 tile512 为 `11.167 us`，vendor 为 `8.661 us`，ratio `1.289`；目标 shape 优先，因此未回退到 tile256。

结论: `int4b_t` 向量反量化路线证明了非标量 unpack 可用，性能已从 `512.6 ms` 降至 `250.1 us`。剩余 `2.84x` 差距主要来自完整反量化权重的 GM 写读。下一步应实现本文第 4-8 节规划的 fused antiquant + Cube/MMAD 或复用 CANN/CATLASS antiquant matmul prologue，避免 `half[K,N]` workspace。
## 远端无 cache 分块优化补充

2026-06-30 后续优化按用户反馈修正方向：cache 会把 int4 权重重新膨胀成 half 常驻 HBM，违背量化节省 HBM/带宽的目标，因此不作为默认实现。

当前默认路线改为 no-cache streaming block：

1. AscendC kernel 以 `int4b_t` 视图搬入当前 N block，向量 `Cast<int4b_t -> half>` 解包，再按 group 广播 `offset/scale`，只生成当前 block 的 `half[K,N_block]`。
2. Host 对当前 block 调用通用 NPU `at::mm(x, deq_block)`，再写入输出切片。
3. 默认 `N_block=12288`，目标 shape 峰值反量化 block 约 96 MB，不常驻完整 `half[4096,22016]` 约 180 MB 权重。

| Route | Shape | Correctness | Candidate us | Vendor us | Ratio |
|---|---|---|---:|---:|---:|
| full dequant no-cache | `[16,4096,22016]` | pass | 251.426361 | 88.366364 | 2.845272 |
| stream dequant 12288 no-cache | `[16,4096,22016]` | pass | 189.847382 | 87.601479 | 2.167171 |
| stream dequant 12288 no-cache | `[2,256,256]` | pass | 17.084600 | 8.258900 | 2.068629 |

本轮还在远端隔离目录 `/root/awq_w4a16_linear_cmct_probe_20260630` 尝试复用 CANN `weight_quant_batch_matmul_v2` 的 CMCT fused prologue。该方向确认存在 AIV antiquant 写 TSCM/L1、AIC `MatmulImpl` 直接消费的数据流，但当前工程模板编译 CMCT internal headers 时缺少 `hifloat8_t`、`AscendC::MicroAPI`、`VECTOR_REG_WIDTH`、`FixpipeParamsC310` 等内部接口，因此未并入主实现。

结论：当前默认实现已避免 cache 和完整 half 权重常驻，相对 full dequant no-cache 提升约 24.5%，但仍有 block 级 GM 写读，目标 shape 仍约 vendor `2.17x`。要继续接近 vendor，需要真正 fused antiquant + Cube/MMAD，而不是 cache。