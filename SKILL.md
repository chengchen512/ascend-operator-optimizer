---
name: ascend-operator-optim-codex
description: >-
  用 Codex 驱动 Ascend NPU 算子性能优化。当用户提到 Ascend C、
  Triton-Ascend、TileLang-Ascend、算子优化、tiling、UB、AIV/AIC、
  MTE/Vector/Scalar、流水、精度或性能回归时使用。
---

# Ascend 算子性能优化

## 目标

在目标算子仓库中完成可验证的性能优化。支持三类实现：

- Ascend C
- Triton-Ascend
- TileLang-Ascend

流程固定为：识别框架与算子边界，建立基线，诊断瓶颈，小步修改，精度验证，使用同一批 case 做性能对比。

## 启动前必读

1. 读取目标仓库的 `AGENTS.md`、README、构建脚本和测试入口。
2. 读取本仓库：
   - `harness/optim-harness.md`
   - `harness/report-templates.md`
   - `knowledge/constraints.md`
3. 按实现类型读取 `knowledge/ascend-operator-knowledge.md` 中对应章节。

## Context Gate

先识别实现类型：

| 实现类型 | 典型文件 | 重点 |
| --- | --- | --- |
| Ascend C | `op_host/*.cpp`、`op_kernel/*.cpp`、`design.md` | tiling、DataCopy、TPipe/TQue、L0/L1/UB |
| Triton-Ascend | `*.py` 中的 `@triton.jit` / `triton.language as tl` | BLOCK/grid、UB、mask、连续访存、msprof |
| TileLang-Ascend | `@tilelang.jit`、`T.prim_func`、`T.Kernel(..., is_npu=True)` | 分块、`T.alloc_shared`、`T.copy`、`T.Pipelined`、NPU API 差异 |

必须完成：

- 读取设计文档、说明文档或测试文档。
- 读取目标算子完整源码。
- 找到构建命令、精度测试入口和性能测试入口。
- 建立或更新 `optim_task.yaml`，格式参考 `harness/task-template.yaml`。

未完成 Context Gate 时，不修改代码。

## Baseline Gate

性能优化前必须有基线：

- 固定一批性能 case。
- 自定义算子和标杆都在 NPU 上运行。
- 按 `harness/report-templates.md` 保存基线报告。
- 记录 profiler 参数和执行命令。

没有基线时，不声明性能收益。

## Failure Log Gate

遇到构建、精度或性能失败时，必须先读取失败日志和失败 case，再更新任务文件。

写入文件的内容只允许包含：

- 命令与退出码。
- 日志路径。
- 原始日志片段。
- case、shape、dtype、模式。
- 可观察现象。
- 下一步核查项。

不要把推断性结论写入 `optim_task.yaml`、`precision_report.md`、`performance_report.md` 或 `optim_summary.md`。推断只允许在当前对话中说明，并且必须标明它是推断。

## Diagnose

按以下顺序排查：

1. 算法：是否存在明显多 pass、重复 GM 访问、逐元素处理。
2. Tiling/Grid：核数、block size、逻辑任务到物理核的映射。
3. Data Movement：连续访存、搬运粒度、mask、stride、GM/UB 往返。
4. UB/Local Memory：峰值占用、FP32 临时量、offset/mask/index 开销。
5. Pipeline：MTE/Vector/Scalar 或 CopyIn/Compute/CopyOut 是否可重叠。
6. Framework API：按 Ascend C、Triton-Ascend、TileLang-Ascend 的专属反模式检查。

输出待改问题列表，并按预期收益排序。每轮只选择一个主要优化点。

## Patch Round

每轮修改必须满足：

- 修改范围尽量小。
- 保留原有输入 shape、dtype、语义和调用接口。
- 不为了性能硬编码单一 shape。
- 不绕过目标实现类型，例如 Triton 问题不能直接改成外部 Python 拼接。
- 变更后更新 `optim_task.yaml` 的本轮目标、文件、命令和结果。

最多 3 轮。若 3 轮后仍无收益，停止并输出事实结果。

## Build Gate

按目标框架执行构建或编译：

- Ascend C：重新构建并安装算子包。
- Triton-Ascend：触发 JIT 编译，保留编译错误和 kernel 名称。
- TileLang-Ascend：执行 TileLang NPU 编译流程，保留 IR 或编译日志。

构建失败时先修复构建问题；同一轮构建排错最多 3 次。

## Precision Gate

性能测试前必须通过精度验证：

- 覆盖原有全部 dtype。
- 覆盖小、中、大和边界 shape。
- Triton-Ascend 默认用 `torch.testing.assert_close(..., rtol=1e-3, atol=1e-3)`，除非目标仓库已有更严格标准。
- Ascend C 优先使用 MERE 与 MARE 报告。
- TileLang-Ascend 对比 PyTorch-NPU 或目标仓库既有标杆。
- 精度报告必须使用 `harness/report-templates.md` 的固定结构。

精度失败时，不进入性能测试；先修复或回退本轮修改。

## Performance Gate

性能验证必须满足：

- 使用 Baseline Gate 的同一批 case。
- 使用同一套 profiler 或计时脚本。
- 记录单位、warmup、repeat/active、kernel name、统计方式和 trace 路径。
- 性能报告必须使用 `harness/report-templates.md` 的固定结构。
- 输出基线、优化后、标杆三方对比。
- 结论基于多 case，而不是单个最快结果。

## 输出要求

最终回复必须包含：

- 修改摘要。
- 精度结果。
- 性能对比摘要。
- 每轮优化目标和结论。
- 相关文件路径。

不得只给文件路径，不得在精度失败时声明性能成功。

## 自检清单

- [ ] 已识别实现类型：Ascend C / Triton-Ascend / TileLang-Ascend。
- [ ] 已读取目标仓库约束、设计或测试文档、目标算子源码。
- [ ] 已读取本 skill 的 harness、report templates 与 constraints。
- [ ] 已建立基线报告。
- [ ] 已按算法、tiling/grid、搬运、UB、流水、框架 API 排查。
- [ ] 每轮只改一个主要优化点。
- [ ] 构建或 JIT 编译通过。
- [ ] 精度验证通过。
- [ ] 失败 case 已按现象日志记录，文件中不写推断性结论。
- [ ] 性能使用同一批 case。
- [ ] 已输出基线、优化后、标杆三方对比。
