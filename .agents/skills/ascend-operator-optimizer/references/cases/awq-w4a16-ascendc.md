# AWQ W4A16 AscendC Case

本案例记录一次 AWQ W4A16 Linear 自定义 AscendC 算子的优化经验。适用场景是用户明确要求纯自定义算子、禁止 vendor fused op、禁止 target shape fast path、禁止常驻反量化 cache，并希望继续探索 fused antiquant + Cube/MMAD。

## 何时读取

- 目标是 int4 weight + FP16 activation 的 W4A16/AWQ Linear。
- 性能目标接近 `torch_npu.npu_weight_quant_batchmatmul` 或 CANN WQMM 类 vendor 路径。
- 需要在 no-cache 约束下减少 HBM 中间权重写读。
- 准备尝试 TSCM、Cube/MMAD、Matmul 高阶 API、ND/NZ 转换或 workspace-aware launch。

## 硬约束

- 不把完整 `half[K,N]` 反量化权重作为持久 cache。
- 不用单一 target shape fast path 冒充通用优化。
- 不把 vendor fused op 包装成自定义算子实现。
- 性能结论必须绑定同一批 case、同一 repeat/warmup、同一参考路径。
- Cube/TSCM 失败实验要隔离在 probe 目录，不覆盖已经通过正确性和性能验证的主线。

## 路线结论

| 路线 | 结果 | 经验 |
| --- | --- | --- |
| scalar unpack + FMA | 正确但极慢 | 只保留为诊断回退。kernel hot path 不应逐元素 `GetValue/SetValue`。 |
| `int4b_t -> half` vector dequant + `at::mm` | 可运行 | 能把毫秒级降到百微秒级，但完整反量化会写读约 `half[K,N]` 中间权重。 |
| no-cache N-block streaming dequant + `at::mm` | 当前稳定主线 | 目标 shape 比 full no-cache 快约 24.5%，但仍落后 vendor，瓶颈是 GM 中间权重和多次 matmul 调度。 |
| versioned dequant cache | 性能可接近 vendor | 被否决。常驻膨胀后的 half 权重违背量化节省 HBM 和带宽的目标。 |
| CANN CMCT/WQMM 内部 prologue | 编译失败 | 公开工程缺内部 `hifloat8_t`、`MicroAPI`、`VECTOR_REG_WIDTH`、`FixpipeParamsC310` 等符号。 |
| TSCM direct producer/consumer | 编译通过但运行失败 | `507015 MPU address access is invalid`。需要先做最小 TSCM producer + AIC consumer probe。 |
| AIV 侧 ND2NZ 写 GM workspace | 编译失败 | 当前可用 API 有 `DataCopy(LocalTensor, GlobalTensor, Nd2NzParams)`，没有对应的 GM dst overload。 |

## 关键性能数据

主线 no-cache stream：

| Case | Candidate | Vendor | Ratio |
| --- | ---: | ---: | ---: |
| 小 shape `[2,256,256]` | `17.084600 us` | `8.258900 us` | `2.07x` |
| 目标 shape full dequant no-cache `[16,4096,22016]` | `251.426361 us` | `88.366364 us` | `2.85x` |
| 目标 shape stream-12288 no-cache `[16,4096,22016]` | `189.847382 us` | `87.601479 us` | `2.17x` |

Cube/MMAD probe：

| 版本 | 1000 avg latency |
| --- | ---: |
| workspace-aware GM scratch baseline | `55.988819 us` |
| host tiling cache | `53.455379 us` |
| pack8 unpack | `42.077782 us` |
| pack8 + direct vector antiquant | `41.949902 us` |

最终 probe 正确性：`max_abs=0.015625`，`mean_abs=0.0012750625610351562`，`allclose(rtol=5e-2, atol=5e-2)=True`。

## 可复用优化点

### workspace-aware launch 参数顺序

`HAVE_WORKSPACE + HAVE_TILING` 的生成 wrapper 约定倒数第二个 kernel 参数是 workspace，最后一个参数是 tiling。保持 Python 接口不变时，host wrapper 必须在底层 launch 中显式补齐 workspace 和 tiling。

```cpp
extern "C" __global__ __aicore__ void awq_w4a16_tscm_awq_probe(
    GM_ADDR x, GM_ADDR weight, GM_ADDR scale, GM_ADDR offset, GM_ADDR y,
    uint32_t m, uint32_t k, uint32_t n,
    GM_ADDR workspace, GM_ADDR tiling);
```

```cpp
EXEC_KERNEL_CMD(awq_w4a16_tscm_awq_probe, block_dim,
    x, weight, antiquant_scale, antiquant_offset, y,
    m_u, k_u, n_u, workspace, tiling_dev);
```

### 只缓存只读 tiling

小 shape probe 中，host 每次构造并 H2D 拷贝 `TCubeTiling` 会进入 event latency。缓存只读 tiling device tensor 可把 1000 次平均延迟从约 `55.99 us` 降到 `53.46 us`。

保留边界：

- 只缓存只读 tiling，不缓存 dequant half 权重。
- workspace 不做持久 cache，避免并发调用踩踏。
- workspace size 按 `GetLibApiWorkSpaceSize() + 16KB + 512B` 申请，不固定 64MB。

### pack8 unpack 降低 scalar store

每个 int32 pack 包含 8 个 int4。把 8 lane 解成一个 `uint64_t` 并一次写入 UB，可将 `128 * 8 * 8 = 8192` 次 scalar store 降到 `1024` 次。

```cpp
AscendC::LocalTensor<uint64_t> bInt8Pack = bInt8.ReinterpretCast<uint64_t>();
for (uint32_t kk = 0; kk < K; ++kk) {
    for (uint32_t packCol = 0; packCol < PACKED_N; ++packCol) {
        const uint32_t packIdx = kk * PACKED_N + packCol;
        bInt8Pack.SetValue(packIdx, DecodeInt4Pack8(wLocal.GetValue(packIdx)));
    }
}
```

该优化是 probe 中收益最大的单点，延迟从约 `53.46 us` 降到约 `42.08 us`。

### 直接 Vector antiquant

固定列宽较小时，可以用 `Cast(int8 -> half)`、按列 `Add(offset)`、按列 `Mul(scale)` 替代高阶 AntiQuant，并去掉 64KB tmp buffer。收益不一定大，但 UB 资源更清晰，便于后续接 Cube/MMAD。

## 失败模式

- TSCM direct path 的运行时 `507015` 不是 no-workspace wrapper 单独导致；切到 workspace-aware 后仍失败。
- ND2NZ 不要假设存在 `DataCopy(GlobalTensor, LocalTensor, Nd2NzParams)`。先查目标 CANN API overload。
- 内部 CMCT/WQMM prologue 不能只靠 include 补全。缺私有 MicroAPI/Fixpipe 符号时应停止，不要继续把内部模板硬塞进普通工程。
- cache hit 性能好不代表路线正确。若任务目标是节省 HBM 和带宽，常驻 half 权重 cache 应视为违反需求。

## 建议的下一轮实验

1. 保留 no-cache stream 主线作为正确性和性能基线。
2. 写最小 TSCM producer/consumer + Matmul probe，不嵌入完整 AWQ kernel。
3. 若 TSCM 仍失败，转向公开 API 可编译的 GM scratch + Cube/MMAD，先验证 L1/L0 layout 和 L0C 累加。
4. 优先寻找 vector bit unpack 或公开 MicroAPI 等价方式，继续降低 int4 unpack scalar 比例。
5. 所有 probe 结果必须回写到固定 case 表，不能把小 shape probe 成功当作完整目标 shape 成功。

## 原始材料

本仓库根目录保留了更完整的复盘材料：

- `knowledge/awq-w4a16-ascendc-experience.md`
- `knowledge/awq-skill-usage.md`
- `knowledge/awq-w4a16/design.md`
- `knowledge/awq-w4a16/remote-test-report.md`
- `knowledge/awq-w4a16/optim-summary.md`
- `reference/skills/`
