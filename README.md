# ascend-operator-optimizer

一个面向 Codex 的单 skill 仓库，用于驱动 Ascend NPU 算子性能优化。

覆盖范围：

- Ascend C
- Triton-Ascend
- TileLang-Ascend

## 结构

```text
.
├── SKILL.md
├── knowledge/
│   ├── ascend-operator-knowledge.md
│   └── constraints.md
└── harness/
    ├── optim-harness.md
    ├── report-templates.md
    └── task-template.yaml
```

仓库只有一个 skill 入口：`SKILL.md`。`knowledge/` 提供硬件、tiling/grid、搬运、API、UB、流水、验证的先验知识；`harness/` 定义任务状态、用例、基线、精度和性能报告的文件契约。

## 安装

推荐使用符号链接，让 Codex 直接读取当前仓库：

```bash
mkdir -p ~/.agents/skills
ln -sfn /Users/songdehao/sdh-lab/code/ascend-operator-optimizer ~/.agents/skills/ascend-operator-optimizer
```

## 使用

在 Codex 中发起任务时，明确目标仓库、算子名和实现类型：

```text
使用 ascend-operator-optimizer，优化 /path/to/repo 里的 <op_name> 算子。
实现类型是 <ascend-c | triton-ascend | tilelang-ascend>。
先建立基线，再做精度和性能验证。
```

实现类型不确定时：

```text
使用 ascend-operator-optimizer，先识别 /path/to/repo 中 <op_name> 的实现类型，再优化性能。
```

## 输入

任务至少给出：

- 目标仓库路径
- 算子名或源码文件路径
- 实现类型，未知时交给 skill 识别

目标仓库中最好已有：

- 精度测试
- 性能测试或 benchmark
- PyTorch-NPU 标杆路径

## 执行产物

skill 会在目标算子的 `test/` 或 benchmark 附近维护：

- `optim_task.yaml`
- `failure_logs/`
- `perf_cases.jsonl`
- `baseline_report.md`
- `precision_report.md`
- `performance_report.md`
- `optim_summary.md`

报告结构由 `harness/report-templates.md` 固定。

这些是执行中间文件，当前仓库的 `.gitignore` 已默认忽略。
