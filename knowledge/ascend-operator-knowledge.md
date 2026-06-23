# Ascend 算子优化知识库

## 1. 通用硬件模型

- AI Core 主要由 Scalar、Vector、Cube、MTE/DMA 和本地存储层级组成。
- 数据主路径通常是：GM/HBM -> 本地存储 -> Vector/Cube -> 本地存储 -> GM/HBM。
- 常见存储层级：L2、L1、L0A、L0B、L0C、UB、BT Buffer、FP Buffer。
- UB、L1、L0 等容量随平台和芯片而变，优先通过平台接口或编译日志确认。
- 性能瓶颈通常落在 Scalar、MTE、Vector、Cube 的某一类耗时上。

## 2. 通用优化顺序

1. 先看算法：避免多 pass、重复读写 GM、逐元素处理。
2. 再看 tiling/grid：让任务数量、block 大小和物理核匹配。
3. 再看搬运：连续访存、足够大的搬运粒度、正确 mask。
4. 再看 UB：统计输入、输出、中间量、offset、mask、index 和 FP32 临时量。
5. 最后看流水：让搬运、计算、写回尽量重叠。

## 3. Ascend C

- Tiling 分两层：核间切分和核内 UB 切分。
- `blockDim` 按算子类型选择：Vector 算子偏 AIV，Cube 算子偏 AIC，MIX 算子按物理核组限制。
- GM 分块按 512B 边界规划，UB 内部按 32B 边界规划。
- 单次 DMA 搬运量尽量达到 16KB 以上。
- 间隔搬运优先使用 `DataCopyParams` 的 `blockCount/blockLen/srcStride/dstStride`。
- GM <-> UB 的非规则搬运优先使用 `DataCopyPad`。
- `TPipe` 放在 kernel 入口函数中创建，再以指针传入 kernel 类。
- 纯搬运算子优先使用 `TQueBind`。
- Vector 指令优先使用 Counter 模式。
- 连续 Vector 运算的中间结果尽量留在 UB。
- `A1*B1 + A2*B2 + ...` 这类场景优先考虑 L0C 原地累加。
- CopyIn、Compute、CopyOut 通过 `TQue` 传递 LocalTensor；double buffer 启用前必须重新计算 UB 峰值。

## 4. Triton-Ascend

- 先用端到端计时筛选，再用 `msprof` 定位 kernel 侧瓶颈。
- BLOCK/grid 优先级：block size、连续访存、UB 复用、编译期常量。
- 矩阵类算子 BLOCK_M/N/K 通常取 16 的倍数，累加器使用 FP32。
- Vector 类算子要按 UB 峰值反推每轮处理的 token/block 数。
- 单轮 UB 建议控制在可 double buffer 的范围内；offset、mask、index tensor 必须计入。
- 所有 `tl.load` / `tl.store` 必须处理边界 mask。
- 归约类算子优先单 pass，避免多次加载同一数据。
- 热路径避免 `tensor.item()`，避免触发 CPU-NPU 同步。
- 避免 kernel 循环内用 `if` 分支修改变量；Triton 可能编译成低效 masked 操作。
- 避免用预计算 offset tensor 多次参与 2D broadcasting，可能触发编译器约束。
- 需要保留泛化性，不能只为单一 shape 写死参数。

## 5. TileLang-Ascend

- NPU 版本必须用分块处理，不使用逐元素 kernel。
- `T.Kernel(...)` 的 grid 参数应直接用函数参数和 `T.ceildiv` 表达，不依赖函数体内局部变量。
- `T.alloc_shared()` 和 `T.alloc_fragment()` 总量要按 UB 预算计算，建议预留 double buffer 空间。
- NPU 向量 API 以向量运算为主；标量参与运算时先用 `T.vbrc` 广播。
- `T.copy` 不用于广播，源和目标形状应一致。
- `T.vcmp` 需要显式输出 bool buffer，比较符使用小写字符串。
- 分块边界必须处理，最后一块不能越界访问。
- `T.Pipelined` 适合迭代次数足够的循环；迭代过少时收益有限。
- 大规模数据可固定物理核数量，每个核串行处理多个逻辑块。
- 输入 tensor 必要时转为 contiguous，避免非连续访存拖慢 MTE。

## 6. 验证

- 精度先于性能。
- 性能 case 优化前后必须一致。
- 自定义算子和标杆都必须在 NPU 上运行。
- 性能报告必须包含基线、优化后、标杆。
- 结论必须来自多 case 对比，不使用单个最快结果代表整体收益。
