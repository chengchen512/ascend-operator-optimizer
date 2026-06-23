# 优化 Harness

Harness 用于让 Codex 在算子优化过程中保留任务状态、固定输入、复用命令、比较指标，并避免性能结论漂移。

## 目标算子目录

优先把任务文件放在目标算子的测试目录；若仓库已有固定 benchmark 目录，则放在同级最接近的位置。

```text
test/
├── optim_task.yaml
├── failure_logs/
├── perf_cases.jsonl
├── baseline_report.md
├── precision_report.md
├── performance_report.md
└── optim_summary.md
```

已有命名规范优先，例如 Ascend C 可继续使用 `<op_name>_perf_cases.jsonl`。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `optim_task.yaml` | 实现类型、当前任务、命令、阶段状态、每轮结果 |
| `failure_logs/` | 构建、精度、性能失败日志和原始片段 |
| `perf_cases.jsonl` | 性能 case，优化前后必须复用 |
| `baseline_report.md` | 优化前性能基线 |
| `precision_report.md` | 优化后精度报告 |
| `performance_report.md` | 优化后性能报告 |
| `optim_summary.md` | 排查、修改、验证、结论摘要 |

`baseline_report.md`、`precision_report.md`、`performance_report.md` 必须使用 `harness/report-templates.md` 的固定结构。

## 失败现象日志

遇到失败 case 时，先读取日志，再记录现象。写入文件的内容只包含事实材料：

- 命令、退出码、日志路径。
- 原始日志片段。
- case、shape、dtype、模式。
- 可观察现象。
- 下一步核查项。

不要在任务文件或报告中写推断性结论。推断只在当前对话中说明，并标明它是推断。

## 阶段状态

`optim_task.yaml` 至少维护这些阶段：

```text
context -> baseline -> diagnose -> patch -> build -> precision -> performance -> summary
```

每个阶段记录：

- `status`: `pending` / `running` / `pass` / `fail`
- `evidence`: 关键文件或命令
- `notes`: 简短事实

## Round 规则

- 每轮只处理一个主要优化点。
- 每轮都有独立的目标、修改文件、构建命令、精度结果、性能结果。
- 最多 3 轮。
- 新一轮不得替换性能 case，除非重新建立基线。
- 每轮失败日志只记录现象和原始片段，不记录推断性结论。

## 性能比较规则

性能表至少包含：

```text
Case | Shape | DType | 基线(us) | 优化后(us) | 标杆(us) | 优化后/基线 | 优化后/标杆
```

结论只基于这张表。

## 报告结构

三类报告的章节和表格列由 `harness/report-templates.md` 定义。执行时可以追加补充章节，但不能删除模板中的章节和列。

## 框架记录

`optim_task.yaml` 必须记录：

- `framework`: `ascend-c` / `triton-ascend` / `tilelang-ascend`
- `operator_name`
- `operator_files`
- `precision_command`
- `performance_command`
- `profiler`
- `measurement`
- `failure_logs`

`measurement` 至少记录：

- `unit`
- `warmup`
- `repeat`
- `active`
- `kernel_name`
- `statistic`
- `trace_dir`

## Round 字段

每轮记录固定为：

```yaml
- index: 1
  target: ""
  observations: []
  changed_files: []
  failure_logs:
    - phase: ""
      case: ""
      command: ""
      exit_code: ""
      log_path: ""
      raw_excerpt: ""
      observed_symptom: ""
      next_check: ""
  build:
    command: ""
    status: pending
    evidence: []
  precision:
    command: ""
    status: pending
    report: ""
  performance:
    command: ""
    status: pending
    report: ""
    summary: ""
  decision: ""
```

## 汇报规则

最终回复展示：

- 排查发现。
- 每轮修改。
- 精度结果。
- 性能摘要。
- 保留的文件路径。

不得只给路径。
