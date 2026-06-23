# 约束

## Hard Stop

遇到以下情况必须暂停当前阶段，先补证据或修复问题：

- 未识别目标实现类型：Ascend C / Triton-Ascend / TileLang-Ascend。
- 未读取目标算子源码和对应测试。
- 无性能基线却要求声明性能收益。
- 精度验证失败却准备运行或汇报性能结果。
- 性能前后使用了不同 case。
- 标杆路径不是 NPU 执行路径。
- 修改需要触碰公共构建、公共工具或无关算子。

## 禁止事项

- 不硬编码单一 shape、核数、UB 容量、L2 容量。
- 不牺牲原有 dtype、shape 和语义泛化能力。
- 不在性能路径引入 CPU-NPU 同步。
- 不用外部 Python 拼接绕开目标算子实现。
- 不把一次 benchmark 结果当成优化收益。
- 不把性能 case 维护成多套不一致格式。

## Ascend C 禁止事项

- 不在 kernel 性能路径使用 `std::min/max/abs/sqrt/exp/log`。
- 不在 kernel 使用动态内存分配。
- 不让 FP16/BF16 直接执行复杂数学计算。
- 不向高维切分 API 传入超过 255 的 `repeatTime`。
- 不在 GM 性能路径使用逐元素 `GetValue()` / `SetValue()`。

## Triton-Ascend 禁止事项

- 不只凭单一规模数据做优化决策。
- 不在 hot path 使用 `tensor.item()`。
- 不忽略 offset、mask、index tensor 的 UB 占用。
- 不在循环内用条件分支修改多个变量。
- 不使用非连续访存作为默认性能路径。

## TileLang-Ascend 禁止事项

- 不写逐元素 kernel 替代分块 kernel。
- 不在 `T.Kernel(...)` grid 参数中引用函数体内局部变量。
- 不用不存在的标量向量 API；标量先广播。
- 不用 `T.copy` 做广播。
- 不省略尾块边界检查。

## 修改边界

优先修改：

- 目标算子源码。
- 目标算子 test/perf/benchmark 文件。
- 目标算子局部配置。

默认不修改：

- 仓库级公共构建。
- 公共工具目录。
- 其他算子目录。
- 模型侧图优化路径。

## 通过条件

一轮优化必须同时满足：

- 构建或 JIT 编译成功。
- 精度报告通过。
- 性能使用同一批 case。
- 至少大部分核心 case 无回退。
- `optim_task.yaml` 记录本轮目标、修改、命令和结果。
