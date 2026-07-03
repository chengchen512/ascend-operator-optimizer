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

## 2026-07-02 workspace-aware host wrapper 更新

- 保持 Python 接口 `torch.ops.npu.awq_w4a16_tscm_awq_probe(x, weight, antiquant_scale, antiquant_offset)` 不变。
- 远端目录: `/root/awq_w4a16_linear_ubmmad_probe_20260701/ascend-kernel`。
- 底层实现已从 `no_workspace_kernel` 移至 `workspace_kernel`，启用 `HAVE_WORKSPACE` 和 `HAVE_TILING`。
- kernel 参数顺序调整为 `..., y, m, k, n, workspace, tiling`，匹配生成 wrapper 对倒数第二个参数作为 workspace 的约定。
- 生成 wrapper 已确认调用 `SetSysWorkspaceForce(workspace_param)` 和 `GetUserWorkspace(workspace_param)`。
- 编译安装通过，正确性 smoke: `max_abs=0.015625`，`mean_abs=0.0012750625610351562`，`allclose(rtol=5e-2, atol=5e-2)=True`。

| Skill | 用途 | 读取的关键文件 |
|---|---|---|
| `ascendc-operator-code-gen` | 修改 kernel/host/CMake 参数顺序与 workspace-aware launch 约定 | `.agents/skills/ascendc-operator-code-gen/SKILL.md`, `references/GUIDE.md` |
| `ascendc-operator-compile-debug` | 远端 `bash build.sh`、wheel 重装、正确性和延迟 smoke 验证 | `.agents/skills/ascendc-operator-compile-debug/SKILL.md` |
| `ssh-remote-connect` | 连接远端服务器并执行源码修改、编译、测试命令 | `.agents/skills/ssh-remote-connect/SKILL.md`, `scripts/connect.ps1` |

## 2026-07-02 继续优化 awq_w4a16_tscm_awq_probe

- 远端目录: `/root/awq_w4a16_linear_ubmmad_probe_20260701/ascend-kernel`。
- 本轮基线: `55.988819 us`，正确性 `max_abs=0.015625`，`mean_abs=0.0012750625610351562`，`allclose(5e-2,5e-2)=True`。
- 保留优化: host tiling tensor 缓存、workspace size 从 64MB 缩为 `GetLibApiWorkSpaceSize()+16KB+512B`、pack8 `uint64_t` unpack 写 UB、直接 vector `Cast/Add/Mul` antiquant。
- 最终结果: `41.949902 us`，相对本轮基线提升约 25.08%，正确性通过。
- 回退路线: TSCM 直连运行失败 `507015 MPU address access is invalid`；AIV 侧 ND2NZ 写 GM workspace 编译失败，CANN 无 `DataCopy(GlobalTensor, LocalTensor, Nd2NzParams)` 重载。
- 远端报告: `csrc/ops/awq_w4a16_tscm_awq_probe/test/awq_w4a16_tscm_awq_probe_optim_summary.md`。

| Skill | 用途 | 读取的关键文件 |
|---|---|---|
| `ascendc-operator-performance-optim` | 按 tiling/搬运/API/内存/流水阶段排查并迭代优化 | `.agents/skills/ascendc-operator-performance-optim/SKILL.md`, `references/tiling-prof.md`, `references/data-copy-prof.md`, `references/api-usage-prof.md`, `references/memory-prof.md`, `references/pipeline-prof.md` |
| `ascendc-operator-code-gen` | 修改 kernel/host 前核对 DataCopy、Vector、Sync、Resource、Kernel 约束 | `.agents/skills/ascendc-operator-code-gen/references/data-copy-api.md`, `vector-compute-api.md`, `sync-control-api.md`, `resource-management-api.md`, `basic-data-structures-api.md`, `kernel-constraints.md` |
| `ascendc-operator-compile-debug` | 远端 `build.sh`、wheel 重装、正确性和性能 smoke | `.agents/skills/ascendc-operator-compile-debug/SKILL.md` |
| `ssh-remote-connect` | 连接远端服务器执行源码修改、编译、测试和报告落盘 | `.agents/skills/ssh-remote-connect/SKILL.md`, `scripts/connect.ps1` |