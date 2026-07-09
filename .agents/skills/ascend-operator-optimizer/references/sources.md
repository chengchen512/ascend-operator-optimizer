# 来源与蒸馏边界

本 Skill 将旧仓库级资料和 Ascend/agent-skills 中的 AscendC 资料蒸馏为一个可复制 Skill 目录。

## 本地来源

- `knowledge/ascend-operator-knowledge.md`
- `knowledge/constraints.md`
- `knowledge/ascendc-agent-skills-distilled.md`
- `harness/optim-harness.md`
- `harness/report-templates.md`
- `harness/task-template.yaml`

## 外部来源

上游仓库：`https://github.com/Ascend/agent-skills/tree/master`

本次读取的浅克隆 HEAD：`155ac37bd169ddb89479af528297cfb2237400aa`

蒸馏范围限于 AscendC 算子相关内容：

- `skills/ascendc-operator-design/`：硬件抽象、通用 tiling、elementwise/reduction/index/pooling/sort tiling 思路。
- `skills/ascendc-operator-code-gen/references/`：基础数据结构、TPipe/TQue/TBuf、DataCopyPad、Vector API、同步控制、kernel 限制。
- `skills/ascendc-operator-performance-optim/references/`：tiling、搬运、API 使用、内存层级、流水优化。
- `skills/ascendc-operator-performance-eval/`：profiler 与指标、case JSON 格式。
- `skills/ascendc-operator-precision-debug/`：GM offset、tail、FP32 upcast、异步同步、多核 tiling 等精度定位案例。
- `skills/ascendc-operator-code-review/references/`：数值、指针、资源、输入校验、并发、接口兼容检查项。
- `skills/ascendc-operator-mssanitizer/`：mssanitizer 使用边界和错误类型。
- `skills/ascend-profiling-anomaly/`：kernel details、op statistic、trace gap、wait-anchor、AICPU 暴露/遮蔽等 profiling 异常分析方法。

## 蒸馏到新架构

- 端到端流程类 skill：蒸馏为 `references/ascendc/workflow.md`，不进入 harness 编排。
- code-gen references：蒸馏为 `references/ascendc/api-usage.md`、`data-copy.md` 和 `ascendc-examples.md`。
- performance references：蒸馏为 `tiling-grid.md`、`memory.md`、`pipeline.md`、`profiling.md`。
- compile/precision/mssanitizer：蒸馏为 correctness/probe 规则和 `precision.md`。
- profiling anomaly：蒸馏为 profile analyzer 的 clock accounting、bubble、wait-anchor 软标签。

未纳入核心 references 的内容：完整项目初始化脚手架、长模板、完整报告样例、与 AscendC 优化无直接关系的 Catlass/Triton/ATB 资料。

## 使用原则

- `ascendc.md` 只保留优化时需要加载的判断规则和诊断顺序。
- `ascendc-examples.md` 只保留短代码模式，不复制完整上游模板。
- 所有设备常量只作为示例语义，真实优化必须从目标环境或 profiler 获取。
