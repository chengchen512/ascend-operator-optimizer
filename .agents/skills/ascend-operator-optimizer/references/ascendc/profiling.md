# Ascend C Profiling Pack

Profiling 是诊断工具，不是性能结论来源。接受或回退修改只看 correctness 和 comparable benchmark。

## Benchmark Contract

推荐用 `torch_npu.profiler` 或 msprof 对同一批 case 分别采集 custom 和 reference。

最低要求：

- custom/reference 都在 NPU 上执行。
- 每个 case、每个实现使用独立 profiler 上下文和独立 trace 目录。
- 记录 `profiler`、`warmup`、`repeat`、`active`、`statistic`、`trace_dir`。
- 使用相同 case identity fields：默认 `case`、`shape`、`dtype`、`mode`。
- `benchmark.json` 的 metric 必须是已归一化后的可比数值。

上游 `performance-eval` 常用口径：

```text
torch_npu.profiler schedule: warmup=5, active=5
每轮调用 prof.step()
读取 *_ascend_pt/ASCEND_PROFILER_OUTPUT/op_statistic.csv
使用 Total Time(us)，按 active 或 active*repeat 归一化
```

## Harness Profile Gate

`commands.profile` 可选。命令应把 trace 写到：

```text
$OPT_HARNESS_PROFILE_DIR
```

如果还生成自定义摘要，写到：

```text
$OPT_HARNESS_PROFILE_RESULT
```

harness 会在 `profiling.analyze=true` 或配置了 `commands.profile` 时尝试解析：

```text
**/kernel_details.csv
**/op_statistic.csv
```

输出：

```text
profile_analysis.json
profile_summary.md
```

## Profile Analysis Signals

分析时维护多套时钟口径：

- `service_us`：第一个 device task 到最后一个 device task 的窗口。
- `device_busy_union_us`：跨 stream 合并后的 device busy 区间。
- `kernel_sum_us`：所有 kernel duration 求和，可能大于 busy union。
- `total_cost_sum_us`：duration + wait 的求和。
- `underfeed_ratio`：`(service_us - busy_union_us) / service_us`。
- `largest_internal_bubble_us`：合并 busy 区间之间最大的空洞。

常见软标签：

| 标签 | 含义 |
| --- | --- |
| `DEVICE_IDLE_GAP_HEAVY` | device busy union 明显小于 service window |
| `INTERNAL_BUBBLE_HEAVY` | device task 之间存在较大内部空洞 |
| `WAIT_ANCHOR_FALSE_HOTSPOT` | top total cost 主要由 wait 组成，不应当成 kernel 计算热点 |
| `INSUFFICIENT_TRACE_DETAIL` | 缺少 `kernel_details.csv`，只能看 op statistic |
| `NO_MAJOR_PROFILE_ANOMALY` | 未发现上述强信号 |

## 禁忌

- 不要把 wait time 当成 kernel busy time。
- 不要用不同 case 或不同 profiler 口径推断收益。
- 不要只看单个最快样本。
- 不要用 profile 标签替代 benchmark compare。
- profile 根因只能写成“可能/倾向/证据不足”，除非有代码和 trace 双重证据。
