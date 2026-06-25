# 报告模板

所有报告均为 Markdown。章节名和表格列必须保持稳定。字段不可用时填写 `N/A`，不要删除章节。

## baseline_report.md

优化前使用该报告记录基线。

```markdown
# 基线报告

## 元信息

| 字段 | 值 |
| --- | --- |
| 实现类型 | <ascend-c / triton-ascend / tilelang-ascend> |
| Launch Profile | <ascendc-standard / ascendc-msopgen-aclnn-dynamic / ascendc-cube-matmul / ascendc-vector / N/A> |
| 算子名称 | <operator_name> |
| 目标仓库 | <absolute_path> |
| 源码提交 | <git_commit_or_N/A> |
| 设备 | <device_model_or_N/A> |
| 性能工具 | <torch_npu.profiler / msprof / timer / other> |
| 单位 | us |
| Warmup | <value_or_N/A> |
| Repeat | <value_or_N/A> |
| Active | <value_or_N/A> |
| Kernel 名称 | <name_or_N/A> |
| 统计方式 | <mean / median / Total Time / Task Duration / other> |
| Trace 目录 | <path_or_N/A> |

## 用例来源

- 性能用例：`<path>`
- 用例数量：`<N>`
- 标杆路径：`<torch_npu_or_repo_baseline>`
- Launch Probe：`<path_or_N/A>`

## 基线表

| Case | Shape | DType | 模式 | 自定义算子(us) | 标杆(us) | 自定义/标杆 | 备注 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 0 | <shape> | <dtype> | <mode> | <value> | <value> | <ratio> | <notes> |

## 摘要

- 自定义算子平均耗时(us)：`<value>`
- 标杆平均耗时(us)：`<value>`
- 最慢用例：`<case_id>`
- 主要基线现象：`<short observation>`
- 是否可进入优化：`<yes/no>`
```

## precision_report.md

每轮修改后、性能测试前使用该报告记录精度结果。

```markdown
# 精度报告

## 元信息

| 字段 | 值 |
| --- | --- |
| 实现类型 | <ascend-c / triton-ascend / tilelang-ascend> |
| Launch Profile | <launch_profile_or_N/A> |
| 算子名称 | <operator_name> |
| 源码提交 | <git_commit_or_N/A> |
| 精度命令 | `<command>` |
| 参考路径 | <torch_npu_or_repo_reference> |

## 结果

| 总数 | 通过 | 失败 | 状态 |
| ---: | ---: | ---: | --- |
| <N> | <N> | <N> | PASS/FAIL |

## 用例表

| Case | Shape | DType | 模式 | 最大绝对误差 | 平均绝对误差 | 最大相对误差 | MERE | MARE | Rtol | Atol | 状态 | 备注 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0 | <shape> | <dtype> | <mode> | <value> | <value> | <value> | <value_or_N/A> | <value_or_N/A> | <value> | <value> | PASS/FAIL | <notes> |

## 失败现象日志

| Case | 日志路径 | 原始日志片段 | 可观察现象 | 下一步核查 |
| --- | --- | --- | --- | --- |
| <case_id> | <path_or_N/A> | <raw_excerpt> | <observed_symptom> | <next_check> |

## 摘要

- 精度状态：`PASS/FAIL`
- 最差用例：`<case_id_or_N/A>`
- 是否可进入性能测试：`<yes/no>`
```

## performance_report.md

精度通过后使用该报告记录性能结果。

```markdown
# 性能报告

## 元信息

| 字段 | 值 |
| --- | --- |
| 实现类型 | <ascend-c / triton-ascend / tilelang-ascend> |
| Launch Profile | <launch_profile_or_N/A> |
| 算子名称 | <operator_name> |
| 源码提交 | <git_commit_or_N/A> |
| 性能用例 | `<path>` |
| 性能工具 | <torch_npu.profiler / msprof / timer / other> |
| 单位 | us |
| Warmup | <value_or_N/A> |
| Repeat | <value_or_N/A> |
| Active | <value_or_N/A> |
| Kernel 名称 | <name_or_N/A> |
| 统计方式 | <mean / median / Total Time / Task Duration / other> |
| Trace 目录 | <path_or_N/A> |

## 对比表

| Case | Shape | DType | 模式 | 基线(us) | 优化后(us) | 标杆(us) | 相对基线加速比 | 优化后/标杆 | 状态 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | <shape> | <dtype> | <mode> | <value> | <value> | <value> | <ratio> | <ratio> | PASS/REGRESSION |

## 摘要

- 基线平均耗时(us)：`<value>`
- 优化后平均耗时(us)：`<value>`
- 标杆平均耗时(us)：`<value>`
- 相对基线平均加速比：`<ratio>`
- 最好用例：`<case_id>`
- 最差用例：`<case_id>`
- 回退用例数量：`<N>`
- 性能状态：`PASS/FAIL`

## 回退用例

| Case | 基线(us) | 优化后(us) | 变化 | 可观察现象 | 下一步核查 |
| --- | ---: | ---: | ---: | --- | --- |
| <case_id> | <value> | <value> | <percent> | <observed_symptom> | <next_check> |

## Profile 诊断

| 指标 | 值 |
| --- | --- |
| Trace 目录 | `<path_or_N/A>` |
| Service(us) | <value_or_N/A> |
| Device busy union(us) | <value_or_N/A> |
| Underfeed ratio | <value_or_N/A> |
| Largest internal bubble(us) | <value_or_N/A> |
| Tags | <tag_list_or_N/A> |

Profile 诊断只作为软信号；性能状态仍由同 case benchmark 对比决定。

## 决策

- 是否保留本轮修改：`<yes/no>`
- 下一步：`<continue/stop/revert/fix_precision/fix_build>`
```
