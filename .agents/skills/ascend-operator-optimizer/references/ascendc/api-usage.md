# Ascend C API Usage Pack

## Kernel 结构

- 推荐 `TPipe` 放在 kernel 入口栈上，再把指针传入 kernel 类。
- `TQue` 用于有生产/消费关系的 GM/UB 队列。
- `TBuf` 用于纯计算临时区、FP32 工作区、归约 tmp。
- `TQueBind` 适合纯搬运或 VECIN/VECOUT 可复用路径。
- 修改 API 调用前如需确认签名、模板参数、重载或限制，按 `../api-lookup.md` 检索官方 API 快照，不要全量读取 `docs/api`。

## Vector API

- 高维 Vector API 的 `repeatTime` 是 `uint8_t`，超过 255 必须分批。
- 多条同元素数 Vector 指令可考虑 Counter 模式，减少 tail mask scalar 开销。
- `Compare` 输入区域按 256B padding，不足部分填不会影响语义的极值。
- `ReduceSum/ReduceMax` 的 `dst` 不能和 `tmpBuffer` alias。

## Launch

- `EXEC_KERNEL_CMD` 不传 rvalue、临时对象、literal 或表达式。
- 不在 kernel hot path 使用 `std::` 数学函数、动态分配、`GlobalTensor::GetValue/SetValue`。
- Runtime2 attr 从运行时上下文读取；shape/dtype 从 desc/shape 读取，除非目标路径明确依赖 tensor data。

## Atomic

- AtomicAdd 前确认 GM 目标和 UB 源 padding 已初始化。
- AtomicAdd 后必须恢复 AtomicNone。
- 需要累加到 GM 已有矩阵时，再考虑 Matmul AtomicAdd；不要把它当默认写回方式。
