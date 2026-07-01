# awq_w4a16_linear 性能优化报告

## 排查发现

1. pack8 baseline 使用 AIV 标量 `GlobalTensor::GetValue` 累加，没有进入 Cube/MMAD，目标 shape `512605.55 us`，不是高性能路线。
2. `int4b_t -> half` 向量 Cast 可用，能避免逐元素标量 unpack，把目标 shape 从毫秒级降到百微秒级。
3. 完整 `half[K,N]` 反量化再 `at::mm` 仍会写读约 180 MB GM 中间权重，不符合量化节省 HBM/带宽的目标。
4. versioned dequant cache 虽然稳态可接近/超过 vendor，但会常驻膨胀后的 half 权重，已按用户反馈从默认实现移除。
5. CANN 内置 WQMM/CMCT 有 fused antiquant + Cube 数据流，但当前 no-workspace 工程模板缺少内部 MicroAPI/Fixpipe 编译环境，隔离 probe 未通过编译。

## 优化前基线

| Shape | Correctness | Candidate us | Vendor us | Ratio |
|---|---|---:|---:|---:|
| `[2,256,256]` | pass | 1935.079956 | 8.511000 | 227.362229 |
| `[16,4096,22016]` | pass | 512605.550130 | 89.959999 | 5698.149803 |

## 迭代历史

### 第 1 轮: pack8 scalar baseline

- 优化目标: 修复正确性并减少同一 int32 pack 的重复读取。
- 代码修改: 一个 pack 一次计算 8 个 int4 lane，UB `DataCopyPad` 写回。
- 精度结果: 通过。
- 性能结果: 目标 shape `512605.55 us`，不可接受。
- 决策: 仅保留为 `AWQ_ASCENDC_ROUTE=scalar` 诊断回退。

### 第 2 轮: `int4b_t` vector dequant + mm

- 优化目标: 用 AscendC 向量 Cast 解包 int4，避免 kernel 内逐元素标量 unpack/FMA。
- 代码修改: AscendC kernel 反量化 `weight -> half[K,N]`，host 调用通用 NPU `at::mm`。
- 精度结果: 通过，目标 shape `max_abs=0.020751953125`, `mean_abs=0.003097736742347479`。
- 性能结果: tile512 目标 shape `251.43 us`，vendor `88.37 us`，ratio `2.85`。
- 决策: 保留为 `AWQ_ASCENDC_ROUTE=full` 对照路线，不作为默认。

### 第 3 轮: versioned dequant cache 实验

- 优化目标: 固定权重推理稳态下避免重复反量化。
- 代码修改: host 增加 TensorImpl + version 单项 cache，cache hit 直接 `at::mm(x, cached_deq_weight)`。
- 精度结果: cache invalidation probe 通过。
- 性能结果: 目标 shape 曾测得 `80.78 us`，vendor `87.57 us`，ratio `0.92`。
- 决策: 按用户反馈否决并移除默认实现。原因是 cache 常驻完整 half 权重，与量化节省 HBM/带宽的目标冲突。

### 第 4 轮: 无 cache N-block streaming dequant

- 优化目标: 不常驻完整 `half[K,N]`，降低峰值 HBM，同时保持非标量 `int4b_t` 向量解包。
- 代码修改: kernel 增加 `colOffset/outN`，只反量化当前 N block；host 按 block 执行 `at::mm` 并写入输出切片。默认 `AWQ_STREAM_N_TILE=12288`。
- 精度结果: 小 shape 和目标 shape 均通过 `rtol=1e-2, atol=2e-2`。
- 性能结果: 目标 shape `189.85 us`，vendor `87.60 us`，ratio `2.17`；相对 full dequant no-cache `251.43 us` 提升约 24.5%。
- 决策: 作为当前默认无 cache 交付路线。

### 第 5 轮: CANN CMCT fused probe

- 优化目标: 复用 CANN WQMM fused antiquant + Cube/MMAD prologue，避免反量化 block 写回 GM。
- 代码修改: 远端隔离目录中新增 `awq_w4a16_cmct_probe`，手动实例化 `InvokeKernel<0,false,false,3,true,false,false>`。
- 精度结果: 未进入运行阶段。
- 编译结果: include 自包含后仍缺 `hifloat8_t`、`AscendC::MicroAPI`、`VECTOR_REG_WIDTH`、`FixpipeParamsC310` 等内部接口。
- 决策: 本轮不并入主实现；需要官方 WQMM 编译模板或重写可公开编译的 fused MMAD kernel。

## 最终性能对比

| Route | Shape | Candidate us | Vendor us | Ratio | 备注 |
|---|---|---:|---:|---:|---|
| scalar baseline | `[16,4096,22016]` | 512605.550130 | 89.959999 | 5698.149803 | 标量回退，不作为性能路线 |
| full dequant no-cache | `[16,4096,22016]` | 251.426361 | 88.366364 | 2.845272 | 完整 `half[K,N]` 中间权重 |
| stream dequant 12288 no-cache | `[16,4096,22016]` | 189.847382 | 87.601479 | 2.167171 | 当前默认，峰值 block 约 96 MB |
| stream dequant 12288 no-cache | `[2,256,256]` | 17.084600 | 8.258900 | 2.068629 | 小 shape 调度开销占比更高 |

## 结论

1. 当前交付实现已移除 cache，不再常驻完整反量化 half 权重。
2. stream-12288 在目标 shape 上相对 full dequant no-cache 提升约 24.5%，但仍未达到 vendor；主要瓶颈仍是反量化 block 的 GM 写读和多次 `at::mm` 调度。
3. CMCT/WQMM fused 路线方向正确，但当前工程模板缺少内部 MicroAPI 编译环境，短期不能稳定并入。
4. 后续要接近 vendor，不能靠 cache，应改为真正 fused antiquant + Cube/MMAD：AIV 解包/反量化直接写 TSCM/L1，AIC `MatmulImpl` 消费并在 L0C 累加。