# awq_w4a16_linear 远端编译测试报告

## 远端环境

- 当前无 cache 分块目录: `/root/awq_w4a16_linear_stream_20260630`
- CMCT fused compile probe 目录: `/root/awq_w4a16_linear_cmct_probe_20260630`
- 历史两阶段目录: `/root/awq_w4a16_linear_route_20260630`
- 历史 scalar baseline 目录: `/root/awq_w4a16_linear_fresh_20260630`
- CANN: `/usr/local/Ascend/cann-9.0.0`
- Python venv: `/root/ascend-operator-optim/.venv`
- 当前 wheel: `/root/awq_w4a16_linear_stream_20260630/ascend-kernel/output/ascend_kernel-2026.3.9-cp311-cp311-linux_x86_64.whl`

## 当前实现状态

- 默认实现不使用 target shape fast path。
- host/kernel 不调用 `torch_npu.npu_weight_quant_batchmatmul`；该接口只在 Python 测试中作为 correctness reference 和 benchmark 对照。
- 默认路线: AscendC kernel 以 `int4b_t` 视图向量反量化当前 N block，host 对每个 block 调用通用 NPU `at::mm`，再写入输出切片。
- 默认 `AWQ_STREAM_N_TILE=12288`，目标 shape 峰值反量化 block 约 `4096*12288*2 = 96 MB`，不再常驻完整 `half[4096,22016]` 约 180 MB 权重。
- 诊断对照: `AWQ_ASCENDC_ROUTE=full` 会完整物化 `half[K,N]` 后 `at::mm`；仅用于对比，不作为默认方案。
- 回退路线: `AWQ_ASCENDC_ROUTE=scalar` 可运行旧 pack8 scalar kernel。
- cache 路线已被否决: 版本化 dequant cache 虽可在固定权重稳态达到约 vendor 水平，但会常驻膨胀后的 half 权重，违背量化节省 HBM/带宽的目标，已从默认实现移除。

## 验证结果

默认测试容差: `rtol=1e-2, atol=2e-2`。

| Route | Shape | Warmup | Repeat | Correctness | Max Abs | Mean Abs | Candidate us | Vendor us | Ratio |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| pack8 scalar baseline | `[2,256,256]` | 5 | 20 | pass | 0.00390625 | 0.0005837632343173027 | 1935.079956 | 8.511000 | 227.362229 |
| pack8 scalar baseline | `[16,4096,22016]` | 1 | 3 | pass | 0.017578125 | 0.0026730522513389587 | 512605.550130 | 89.959999 | 5698.149803 |
| full dequant + mm, no cache | `[16,4096,22016]` | 100 | 1000 | pass | 0.020751953125 | 0.003097736742347479 | 251.426361 | 88.366364 | 2.845272 |
| stream dequant 4096 + block mm | `[16,4096,22016]` | 20 | 100 | pass | 0.020751953125 | 0.003097736742347479 | 287.262802 | 87.665005 | 3.276824 |
| stream dequant 8192 + block mm | `[16,4096,22016]` | 20 | 100 | pass | 0.020751953125 | 0.003097736742347479 | 226.764793 | 88.310003 | 2.567827 |
| stream dequant 11008 + block mm | `[16,4096,22016]` | 20 | 100 | pass | 0.020751953125 | 0.003097736742347479 | 215.011406 | 88.411999 | 2.431926 |
| stream dequant 12288 + block mm | `[16,4096,22016]` | 100 | 1000 | pass | 0.020751953125 | 0.003097736742347479 | 189.847382 | 87.601479 | 2.167171 |
| stream dequant 12288 + block mm | `[2,256,256]` | 20 | 200 | pass | 0.00390625 | 0.0007003415375947952 | 17.084600 | 8.258900 | 2.068629 |

## CMCT fused probe

为继续尝试不写 GM 中间反量化 block 的路线，远端创建了隔离 probe `/root/awq_w4a16_linear_cmct_probe_20260630`，尝试手动实例化 CANN 内置 `WeightQuantBatchMatmulV2::InvokeKernel<0,false,false,3,true,false,false>`，对应 `A=half`、`W=int4b_t`、`scale/offset=half`、ND weight、per-group antiquant、带 offset。

结论: CANN OPP 中确实存在 fused antiquant + Cube 数据流，但当前 `ascend-kernel` no-workspace 工程模板不能直接编译 CMCT 内部 headers。已确认的阻塞:

- include 路径需要把 `common/cmct` 复制到 probe 后才能越过 `cmct/policy/dispatch_policy.h`。
- 进入 AIC/AIV 编译后，缺少 CMCT 依赖的内部符号: `hifloat8_t`、`AscendC::MicroAPI`、`AscendC::VECTOR_REG_WIDTH`、`AscendC::FixpipeParamsC310`。
- 补 `kernel_operator_intf.h` 和 `kernel_basic_intf.h` 后错误不变，说明需要官方 WQMM 编译模板或额外 devkit 宏/头环境，不能作为本轮稳定交付。

## 结论

1. cache 已从默认实现移除；当前默认不常驻完整 half 权重。
2. 分块路线相对 full dequant no-cache 从 `251.43 us` 降到 `189.85 us`，提升约 24.5%，同时把目标 shape 峰值反量化 workspace 从约 180 MB 降到约 96 MB。
3. 默认路线仍有 GM block 写读，目标 shape 仍为 vendor 的约 `2.17x`，没有达到 vendor 水平。
4. 要真正接近 vendor，下一步必须实现 fused antiquant + Cube/MMAD，或把工程切换到能编译 CMCT/WQMM internal prologue 的模板，而不是使用 cache。