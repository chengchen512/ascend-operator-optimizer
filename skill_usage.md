# Skill 使用记录

## 任务

- 输入需求: `task.md`
- 输出设计: `ascend-kernel/csrc/ops/awq_w4a16_linear/design.md`
- 日期: 2026-06-30

## 本次使用到的 skill

| Skill | 用途 | 读取的关键文件 |
|---|---|---|
| `ascendc-operator-design` | 主设计流程，确定接口、计算逻辑、Tiling、Workspace、实现检查清单 | `.agents/skills/ascendc-operator-design/SKILL.md`, `templates/design-template.md`, `references/general-tiling-principles.md` |
| `ascendc-operator-performance-optim` | 性能优化闭环，约束 no-cache、分块反量化、CMCT fused probe 和 benchmark 对比 | `.agents/skills/ascendc-operator-performance-optim/SKILL.md` |
| `ascendc-operator-code-gen` | 生成/修改 host、kernel、测试文件，并参考 AscendC API 约束 | `.agents/skills/ascendc-operator-code-gen/SKILL.md`, `references/data-copy-api.md`, `references/vector-compute-api.md`, `references/resource-management-api.md`, `references/kernel-constraints.md` |
| `ascendc-operator-compile-debug` | 远端编译安装 wheel，执行正确性与性能测试 | `.agents/skills/ascendc-operator-compile-debug/SKILL.md` |
| `ssh-remote-connect` | 通过 SSH/Paramiko 上传工程并在远端执行构建测试 | `.agents/skills/ssh-remote-connect/SKILL.md`, `scripts/connect.ps1`, `scripts/remote_exec.py` |
| `catlass-operator-design` | 早期评估 GEMM 类算子是否应走 CATLASS；当前工作区无可直接复用的 W4A16 AWQ 组件 | `.agents/skills/catlass-operator-design/SKILL.md` |
| `ascendc-operator-project-init` | 补齐标准 `ascend-kernel` 工程模板，创建算子目录结构 | `.agents/skills/ascendc-operator-project-init/SKILL.md`, `templates/ascend-kernel/*` |

## 选型结论

- 当前交付路线为纯自定义 AscendC no-cache 实现，不使用 target shape fast path，不调用 `torch_npu.npu_weight_quant_batchmatmul` 作为实现路径。
- 默认 host 路线按 N 维分块: AscendC `int4b_t` 向量反量化当前 block，通用 NPU `at::mm` 计算 block 输出。
- 默认 `AWQ_STREAM_N_TILE=12288`；目标 shape 峰值反量化 block 约 96 MB，不再常驻完整 `half[4096,22016]` 约 180 MB 权重。
- `AWQ_ASCENDC_ROUTE=full` 仅作为完整反量化 no-cache 对照；`AWQ_ASCENDC_ROUTE=scalar` 仅作为 pack8 scalar 诊断回退。
- versioned dequant cache 已按用户反馈否决并移出默认实现，原因是 cache 常驻 half 权重，违背量化节省 HBM/带宽目标。

## 远端验证结果

- 当前目录: `/root/awq_w4a16_linear_stream_20260630`
- CMCT probe 目录: `/root/awq_w4a16_linear_cmct_probe_20260630`
- 编译: `bash build.sh` 通过，wheel 已安装。
- 默认测试容差: `rtol=1e-2, atol=2e-2`。
- 小 shape `[2,256,256]` 默认 stream-12288: candidate `17.084600 us`，vendor `8.258900 us`，ratio `2.068629`，精度通过。
- 目标 shape `[16,4096,22016]` full dequant no-cache: candidate `251.426361 us`，vendor `88.366364 us`，ratio `2.845272`，精度通过。
- 目标 shape `[16,4096,22016]` 默认 stream-12288: candidate `189.847382 us`，vendor `87.601479 us`，ratio `2.167171`，精度通过，测试配置 `warmup=100, repeat=1000`。
- 相对 pack8 baseline: 目标 shape 从 `512605.55 us` 降至 `189.85 us`，约 `2700x` 提升。
- 相对 full dequant no-cache: stream-12288 从 `251.43 us` 降至 `189.85 us`，约 24.5% 提升。

## CMCT fused probe 记录

- 远端确认 CANN OPP 中存在 `weight_quant_batch_matmul_v2` fused 数据流: AIV antiquant 写 TSCM/L1，AIC `MatmulImpl` 直接消费。
- 隔离 probe 尝试手动实例化 `WeightQuantBatchMatmulV2::InvokeKernel<0,false,false,3,true,false,false>`。
- include 自包含后仍缺内部符号: `hifloat8_t`、`AscendC::MicroAPI`、`AscendC::VECTOR_REG_WIDTH`、`AscendC::FixpipeParamsC310`。
- 结论: 当前 no-workspace 工程模板不能直接编译 CMCT 内部 prologue；下一步需要官方 WQMM 编译模板或重写公开 AscendC fused MMAD kernel。

## 报告

- 远端测试报告: `ascend-kernel/csrc/ops/awq_w4a16_linear/remote_test_report.md`
- 优化汇总: `ascend-kernel/csrc/ops/awq_w4a16_linear/test/awq_w4a16_linear_optim_summary.md`