# AWQ W4A16 AscendC 算子优化经验总结

## 背景

本次工作围绕 `task.md` 中的 AWQ W4A16 Linear 算子展开，目标是在不使用 vendor fused op、不使用 target shape fast path、不常驻 dequant cache 的前提下，设计并优化纯自定义 AscendC 算子。

主要远端实验目录：

- `/root/awq_w4a16_linear_stream_20260630`
- `/root/awq_w4a16_linear_cmct_probe_20260630`
- `/root/awq_w4a16_linear_ubmmad_probe_20260701/ascend-kernel`

## 使用到的 Skill

| Skill | 使用阶段 | 主要作用 |
|---|---|---|
| `ssh-remote-connect` | 远端连接、编译、测试 | 通过项目内 SSH 脚本连接远端 Ascend 服务器，执行构建、安装、NPU 测试和日志采集。 |
| `ascendc-operator-project-init` | 工程初始化 | 补齐 `ascend-kernel` 标准工程结构和算子目录。 |
| `ascendc-operator-design` | 初始设计 | 梳理接口、数据流、tiling、workspace 和精度策略。 |
| `ascendc-operator-code-gen` | 代码生成与修改 | 参考 DataCopy、Vector、Resource、Sync、Kernel 约束修改 host/kernel/CMake。 |
| `ascendc-operator-compile-debug` | 编译和正确性验证 | 远端执行 `build.sh`、安装 wheel、运行 Python 正确性 smoke。 |
| `ascendc-operator-performance-optim` | 性能优化闭环 | 建立基线，按 tiling、搬运、API、内存、流水阶段排查并迭代优化。 |
| `catlass-operator-design` | 早期路线评估 | 判断当前场景是否适合 CATLASS；结论是无现成 W4A16 AWQ 组件可直接复用。 |

## 成功经验

### 1. 先保存可运行基线，再做高风险实验

TSCM、CMCT、Cube/MMAD 相关尝试容易触发编译失败、运行时 `507015` 或错误结果。每轮修改前保留可回退版本，失败后立即恢复，避免污染已验证路径。

有效做法：

- 每轮先记录 baseline 延迟和正确性。
- 修改前备份 host/kernel/CMake。
- 高风险路线单独在 probe 目录中做，不覆盖主线实现。
- 失败路线只记录证据，不纳入最终 benchmark 成果。

### 2. workspace-aware wrapper 必须匹配生成器参数约定

启用 `HAVE_WORKSPACE + HAVE_TILING` 后，AscendC 自动生成 wrapper 会把倒数第二个 kernel 参数当作 workspace，最后一个参数当作 tiling。

正确签名形态：

```cpp
extern "C" __global__ __aicore__ void awq_w4a16_tscm_awq_probe(
    GM_ADDR x, GM_ADDR weight, GM_ADDR scale, GM_ADDR offset, GM_ADDR y,
    uint32_t m, uint32_t k, uint32_t n,
    GM_ADDR workspace, GM_ADDR tiling);
```

host launch 也必须保持同样顺序：

```cpp
EXEC_KERNEL_CMD(awq_w4a16_tscm_awq_probe, block_dim,
    x, weight, antiquant_scale, antiquant_offset, y,
    m_u, k_u, n_u, workspace, tiling_dev);
```

### 3. 小 shape 下 host wrapper 开销可见

固定 probe case `[M,K,N]=[16,128,64]` 很小，每次 host 构造并 H2D 拷贝 `TCubeTiling` 会进入 event latency。把 tiling tensor 缓存为只读 device tensor 后，1000 次平均延迟从约 `55.99 us` 降到 `53.46 us`。

保留原则：

- 只缓存只读 tiling，不缓存 dequant half 权重。
- workspace 不做持久 cache，避免并发调用踩踏。
- workspace size 从固定 `64MB` 缩小为 `GetLibApiWorkSpaceSize() + 16KB + 512B`，减少临时 HBM 占用。

### 4. scalar unpack 的最大收益来自减少写次数

原始 int4 unpack 对每个 int4 lane 调一次 `SetValue`，`128 * 8 * 8 = 8192` 次 scalar store。优化后把每个 int32 pack 解成 8 个 int8 byte，用 `uint64_t` 一次写入 UB，将 scalar store 降到 `1024` 次。

保留代码模式：

```cpp
AscendC::LocalTensor<uint64_t> bInt8Pack = bInt8.ReinterpretCast<uint64_t>();
for (uint32_t kk = 0; kk < K; ++kk) {
    for (uint32_t packCol = 0; packCol < PACKED_N; ++packCol) {
        const uint32_t packIdx = kk * PACKED_N + packCol;
        bInt8Pack.SetValue(packIdx, DecodeInt4Pack8(wLocal.GetValue(packIdx)));
    }
}
```

该优化让固定 probe latency 从约 `53.46 us` 降到约 `42.08 us`，是本轮收益最大的优化点。

### 5. 简单固定广播可替换高阶 AntiQuant

在固定 `[K,N]=[128,64]` 场景下，`AscendAntiQuant` 可替换为：

1. `Cast(int8 -> half)`
2. 对 64 列 offset 做高维广播 `Add`
3. 对 64 列 scale 做高维广播 `Mul`

这样去掉了 `AscendAntiQuant` 的 64KB tmp buffer。延迟收益不大，但 UB 资源更干净，最终版本保留。

## 失败经验

### 1. TSCM 直连仍不稳定

尝试路径：

- AIV: `TSCM<TPosition::VECIN,1>::AllocTensor`
- AIV: 反量化 half tile `DataCopy` 到 TSCM
- AIV: `EnQue`
- AIC: `DeQue`
- AIC: `mm.SetTensorB`

结果：

- 编译通过。
- 运行失败：`507015 MPU address access is invalid`。

结论：

- 问题不是 no-workspace wrapper；切到 workspace-aware wrapper 后仍失败。
- 当前公开 TSCM 队列和 `MatmulImpl` 模板组合还不能作为可接受路径。
- 后续若继续走 TSCM，需要先做更小的独立 TSCM producer/consumer probe，而不是直接嵌入 AWQ kernel。

### 2. AIV 侧 ND2NZ 写 GM workspace 不可行

尝试把 ND->NZ 转换前移到 AIV 写 workspace 阶段，AIC 只做 contiguous GM->B1 copy。

失败原因：

```text
no matching function for call to DataCopy(GlobalTensor, LocalTensor, Nd2NzParams)
```

CANN 支持当前使用的 `DataCopy(LocalTensor, GlobalTensor, Nd2NzParams)`，但不支持 `DataCopy(GlobalTensor, LocalTensor, Nd2NzParams)`。

### 3. CMCT 内部 prologue 不能直接复用

远端 OPP 中存在 `weight_quant_batch_matmul_v2` 的 fused 数据流，但直接实例化内部模板缺少内部符号：

- `hifloat8_t`
- `AscendC::MicroAPI`
- `AscendC::VECTOR_REG_WIDTH`
- `AscendC::FixpipeParamsC310`

结论：

- 当前工程模板不能直接编译 CANN 内部 CMCT prologue。
- 后续需要官方 WQMM 编译模板，或者基于公开 AscendC API 重写低层 MMAD 数据通路。

### 4. 不要把 dequant cache 当作优化主线

versioned dequant cache 虽然能减少重复反量化，但会常驻 half 权重，违背量化节省 HBM 和带宽的目标。该路线不适合作为最终答案。

## 性能结果

### stream no-cache 主线

| Case | Candidate | Vendor | Ratio | 结论 |
|---|---:|---:|---:|---|
| 小 shape `[2,256,256]` | `17.084600 us` | `8.258900 us` | `2.07x` | 精度通过 |
| 目标 shape full dequant no-cache `[16,4096,22016]` | `251.426361 us` | `88.366364 us` | `2.85x` | 精度通过 |
| 目标 shape stream-12288 `[16,4096,22016]` | `189.847382 us` | `87.601479 us` | `2.17x` | 精度通过 |

### TSCM/MMAD probe 主线

| 版本 | 1000 avg latency |
|---|---:|
| workspace-aware GM scratch baseline | `55.988819 us` |
| host tiling cache | `53.455379 us` |
| pack8 unpack | `42.077782 us` |
| final: pack8 + direct vector antiquant | `41.949902 us` |

最终正确性：

- `max_abs=0.015625`
- `mean_abs=0.0012750625610351562`
- `allclose(rtol=5e-2, atol=5e-2)=True`

## 可复用流程

1. 先用 `ascendc-operator-design` 明确接口、shape、误差容忍和禁用路线。
2. 用 `ascendc-operator-code-gen` 生成最小可编译 host/kernel/CMake。
3. 用 `ascendc-operator-compile-debug` 跑通 build、wheel 安装和正确性。
4. 用 `ascendc-operator-performance-optim` 记录同一 case 的基线。
5. 每轮只改一个主要优化点，保留备份。
6. 先跑正确性，再跑性能；失败路线只记录，不纳入最终数据。
7. 高风险 Cube/MMAD/TSCM 实验必须隔离，不要污染稳定实现。

## 后续建议

- 继续降低 scalar unpack：寻找公开可用的 vector bit unpack 或 MicroAPI 等价实现。
- 继续探索低层 MMAD：绕开 `MatmulImpl` 固定开销，但需要更严格的 L1/L0 layout 验证。
- 若继续 TSCM，先写最小 AIV producer + AIC consumer + Matmul probe，单 translation unit、单路线验证。
- 保持 no-cache 原则，不引入常驻 half dequant cache。
- 所有性能结论必须绑定固定 case、固定 repeat、固定参考路径，避免把 probe 成功误报为完整算子成功。
