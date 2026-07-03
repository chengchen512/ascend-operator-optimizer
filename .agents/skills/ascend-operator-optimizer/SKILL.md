---
name: ascend-operator-optimizer
description: >-
  用 Codex 驱动 Ascend NPU 算子性能优化。当用户提到 Ascend C、
  Triton-Ascend、TileLang-Ascend、算子优化、tiling、UB、AIV/AIC、
  MTE/Vector/Scalar、流水、精度或性能回归时使用。
---

# Ascend Operator Optimizer

## 目标

在目标算子仓库中做可验证、可回退、可复现的 Ascend NPU 性能优化。核心规则只有一条：

**正确性优先；任何性能收益都必须来自同一批 case、同一套命令和可比较的测量。**

本 Skill 支持：

- Ascend C
- Triton-Ascend
- TileLang-Ascend

固定循环：

```text
inspect -> baseline -> diagnose -> one hypothesis -> patch
        -> correctness gate -> benchmark gate -> accept/revert -> repeat (max 3)
```

## 自包含结构

只依赖本 Skill 目录内的文件：

```text
.agents/skills/ascend-operator-optimizer/
├── SKILL.md
├── assets/harness.example.json
├── references/
│   ├── INDEX.md
│   ├── hardware.md
│   ├── playbook.md
│   ├── constraints.md
│   ├── harness.md
│   ├── api-lookup.md
│   ├── ascendc/
│   │   ├── workflow.md
│   │   ├── tiling-grid.md
│   │   ├── data-copy.md
│   │   ├── api-usage.md
│   │   ├── memory.md
│   │   ├── pipeline.md
│   │   ├── precision.md
│   │   ├── profiling.md
│   │   └── launch-profiles.md
│   ├── cases/
│   │   └── awq-w4a16-ascendc.md
│   ├── ascendc.md
│   ├── ascendc-examples.md
│   └── sources.md
└── scripts/
    ├── harness.py
    └── api_lookup.py
```

不要依赖目标仓库之外的 `knowledge/` 或 `harness/` 目录。若当前仓库中仍有旧版资料，只把它们当作人工参考，不作为执行契约。根目录 `reference/asc-devkit-api-9.0.0/docs/api` 是可选的官方 API 离线快照，不属于复制 Skill 的最小依赖；需要精确 API 证据时按 `references/api-lookup.md` 查找。

## 启动前必读

1. 读取目标仓库的 `AGENTS.md`、README、构建脚本、测试入口和 benchmark 入口。
2. 读取本 Skill 的：
   - `references/INDEX.md`
   - `references/playbook.md`
   - `references/constraints.md`
   - `references/harness.md`
   - `references/hardware.md`
   - `references/api-lookup.md`
3. 若目标是 Ascend C，先读取 `references/ascendc.md` 和 `references/ascendc/workflow.md`。
4. 按场景读取 AscendC knowledge pack：
   - custom op launch 或 blockDim 异常：`references/ascendc/launch-profiles.md`
   - 性能评估或 profiler 诊断：`references/ascendc/profiling.md`
   - tiling/grid/tail：`references/ascendc/tiling-grid.md`
   - GM/UB 搬运：`references/ascendc/data-copy.md`
   - API/TQue/TBuf/Atomic：`references/ascendc/api-usage.md`
   - UB/L1/L0/BT/FP buffer：`references/ascendc/memory.md`
   - 流水重叠：`references/ascendc/pipeline.md`
   - 精度或 MSSanitizer：`references/ascendc/precision.md`
   - AWQ/W4A16、int4 unpack、no-cache 反量化、TSCM/Cube/MMAD：`references/cases/awq-w4a16-ascendc.md`
5. 若需要确认 Ascend C API 签名、重载、参数单位或限制，按 `references/api-lookup.md` 使用 `scripts/api_lookup.py` 或 `rg` 查官方 API 快照；不要全量读取 `docs/api`。
6. 若需要修改 Ascend C 源码，再读取 `references/ascendc-examples.md` 中相关代码模式。
7. 若需要确认来源或外部资料边界，读取 `references/sources.md`。

未完成目标仓库上下文读取前，不修改代码。

## 在目标仓库安装

将整个 Skill 目录复制到目标仓库：

```bash
mkdir -p .agents/skills
cp -R /path/to/codex-ascend-operator-optimizer/.agents/skills/ascend-operator-optimizer \
  .agents/skills/
```

创建 harness 配置：

```bash
cp .agents/skills/ascend-operator-optimizer/assets/harness.example.json \
  operator-optim.json
```

只编辑配置中的命令、实现元数据和本地 artifact globs。benchmark 命令必须把 JSON 写入环境变量 `OPT_HARNESS_RESULT` 指向的路径。

## Harness 契约

建立基线：

```bash
python .agents/skills/ascend-operator-optimizer/scripts/harness.py \
  baseline --config operator-optim.json
```

评估一轮修改：

```bash
python .agents/skills/ascend-operator-optimizer/scripts/harness.py \
  evaluate --config operator-optim.json --label iter-1
```

运行记录写入 `.operator-optim/`，通常应排除在版本控制之外。

benchmark JSON 最少包含：

```json
{
  "cases": [
    {
      "name": "case-id",
      "latency_us": 123.4
    }
  ]
}
```

case 名称必须稳定。比较时只接受同一批 case；缺失、新增或单位不一致都不能声明性能收益。

## Context Gate

先识别实现类型和算子边界：

| 实现类型 | 典型文件 | 重点 |
| --- | --- | --- |
| Ascend C | `op_host/*.cpp`、`op_kernel/*.cpp`、`design.md` | tiling、DataCopy、TPipe/TQue、L0/L1/UB |
| Triton-Ascend | `*.py` 中的 `@triton.jit` / `triton.language as tl` | BLOCK/grid、mask、连续访存、UB、msprof |
| TileLang-Ascend | `@tilelang.jit`、`T.prim_func`、`T.Kernel(..., is_npu=True)` | 分块、`T.alloc_shared`、`T.copy`、`T.Pipelined`、NPU API 差异 |

必须完成：

- 读取设计文档、说明文档或测试文档。
- 读取目标算子完整源码。
- 找到构建命令、正确性测试入口和性能测试入口。
- 配好 `operator-optim.json` 的 `implementation.type` 和必要的 `implementation.launch_profile`。
- 对 `ascendc-msopgen-aclnn-dynamic` 这类路径，必须配置并通过 `commands.launch_probe`。
- 确认 benchmark 输出结构稳定。

## Baseline Gate

性能优化前必须有基线：

- 固定一批性能 case。
- 自定义算子和标杆都在 NPU 上运行。
- 用 harness 的 `baseline` 命令保存基线。
- 记录命令、环境、单位、warmup、repeat、统计方式和 profiler/trace 路径。

没有基线时，不声明性能收益。

## Harness Gate

基础顺序：

```text
env_smoke -> build -> launch_probe -> correctness -> benchmark -> profile -> compare
```

- `env_smoke`、`launch_probe`、`profile` 是可选 gate。
- `implementation.launch_profile=ascendc-msopgen-aclnn-dynamic` 时，`launch_probe` 默认必需。
- `profile` 只生成诊断证据，不替代 benchmark compare。

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
- 修改后先过 correctness，再进入 benchmark。

最多 3 轮。若 3 轮后仍无收益，停止并输出事实结果。

## Correctness Gate

性能测试前必须通过正确性验证：

- 覆盖目标仓库已有 dtype、shape 和边界 case。
- Triton-Ascend 默认使用目标仓库既有阈值；没有标准时才考虑 `rtol=1e-3, atol=1e-3` 级别的临时门槛。
- Ascend C 优先保留 MERE/MARE 或目标仓库已有精度报告。
- TileLang-Ascend 对比 PyTorch-NPU 或目标仓库既有标杆。

正确性失败时，不进入性能测试；先修复或回退本轮修改。

## Benchmark Gate

性能验证必须满足：

- 使用 Baseline Gate 的同一批 case。
- 使用同一套计时脚本或 profiler。
- `harness.py evaluate` 能与基线完成 case 对齐。
- 记录基线、优化后、标杆三方对比；没有标杆时明确写 `N/A`。
- 结论基于多 case，而不是单个最快结果。

如果收益来自噪声范围、case 不可比、单位不一致或正确性失败，必须拒绝该轮修改并回退。

## Failure Log Gate

遇到构建、正确性或性能失败时，先读取失败日志和失败 case。

文件记录只允许包含：

- 命令与退出码。
- 日志路径。
- 原始日志片段。
- case、shape、dtype、模式。
- 可观察现象。
- 下一步核查项。

不要把推断性结论写入执行产物。推断只允许在当前对话中说明，并且必须标明它是推断。

## 输出要求

最终回复必须包含：

- 修改摘要。
- 正确性结果。
- 性能对比摘要。
- 每轮优化目标和结论。
- 相关文件路径和 harness run 目录。

不得只给文件路径，不得在正确性失败时声明性能成功。

## 自检清单

- [ ] 已识别实现类型：Ascend C / Triton-Ascend / TileLang-Ascend。
- [ ] 已读取目标仓库约束、设计或测试文档、目标算子源码。
- [ ] 已读取本 Skill 的 `references/` 和 `harness.py` 契约。
- [ ] 已建立基线。
- [ ] 已按算法、tiling/grid、搬运、UB、流水、框架 API 排查。
- [ ] 每轮只改一个主要优化点。
- [ ] 构建或 JIT 编译通过。
- [ ] 正确性验证通过。
- [ ] 失败 case 已按现象日志记录，文件中不写推断性结论。
- [ ] 性能使用同一批 case。
- [ ] 已输出基线、优化后、标杆三方对比。

