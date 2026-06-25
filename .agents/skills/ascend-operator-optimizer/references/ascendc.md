# Ascend C 算子优化蒸馏知识

本文件面向“已有 Ascend C 算子性能优化”。它不是完整 API 手册，而是从 Ascend/agent-skills 中蒸馏出的高频判断、排错路径和优化规则。需要代码模式时读取 `ascendc-examples.md`。

细分知识包位于 `references/ascendc/`。按问题读取，不要每次加载所有文件：

- `workflow.md`：上游多 skill 到当前单 harness 的映射。
- `launch-profiles.md`：标准 AscendC、msopgen/aclnn dynamic、Cube、Vector launch profile。
- `profiling.md`：profiler 采集、CSV 分析、软标签。
- `tiling-grid.md`、`data-copy.md`、`api-usage.md`、`memory.md`、`pipeline.md`、`precision.md`：专项检查表。

## 读取目标

优化前必须读完：

- `op_host/*.cpp` 或 tiling 入口。
- `op_kernel/*.cpp` 或 kernel 类实现。
- 算子 `design.md`、shape/dtype 约束、测试和 benchmark。
- TilingData 定义、TilingKey/TilingID 历史兼容逻辑、workspace 申请逻辑。
- `operator-optim.json` 中的 `implementation.type`、`implementation.launch_profile`、`commands.launch_probe`。

缺少上下文时不要改代码。不要用示例里的常量替代目标环境查询结果。

## Launch Profile 先行

AscendC launch 路径至少区分：

- `ascendc-standard`：普通 ascend-kernel / `EXEC_KERNEL_CMD` 路径。
- `ascendc-msopgen-aclnn-dynamic`：msopgen + aclnn custom op，必须先通过 launch coverage probe。
- `ascendc-cube-matmul`：优先看 AIC/Cube、L0/L1/L0C、Fixpipe、BT/FP buffer。
- `ascendc-vector`：优先看 AIV、UB、DataCopyPad、TQue、流水。

如果 launch coverage probe 失败，不进入 kernel 数学、tiling 或性能优化；先修 launch path 或工程路径。

## 设计与 Tiling

Ascend C 常见两级切分：

- Block 级：把逻辑任务分配到 AIV/AIC 或 MIX core group，目标是负载均衡、GM/L2 访问友好、无遗漏无重叠。
- UB 级：把单核任务继续分成可放入 UB 的 tile，必须统计输入、输出、临时 buffer、FP32 工作区、index/mask、double buffer。

Tiling 必查项：

- `formerNum * formerLength + tailNum * tailLength` 或目标仓库等价字段必须刚好覆盖全部数据。
- `usedCoreNum` 不应超过平台可用核数；小 shape 不应强行铺满所有核。
- GM 偏移、shape 元素数、字节数乘积使用 `int64_t`，不要用 `int32_t` 承接可能超过 2GB 的值。
- Block 级优先考虑 512B GM/cache line 对齐；UB 内部按 32B 对齐。
- Host 侧通过平台接口获取 AIV/AIC 数、UB、L1、workspace/lib workspace 大小；不要写死 910B 示例值。
- Tail block 和 tail tile 必须显式处理，kernel 内计算和搬运使用 `curTileLength`，不要把对齐长度当有效长度。

Elementwise 默认是 memory-bound：优先让单次 DMA 足够大、访问连续、分核均匀。Reduction/Normalize 默认是数值和 UB 压力敏感：优先保证 FP32 中间精度、tmpBuffer 独立、行级对齐和 `repeatTime` 分批。

## Kernel 结构

推荐把 `TPipe` 放在 kernel 入口栈上，再以指针传给 kernel 类。把 `TPipe` 作为类成员可能削弱类内 Scalar 变量的常量传播，Scalar bound 场景尤其明显。

资源选择：

- `TQue<VECIN/VECOUT>`：GM 和 UB 之间的生产/消费队列，承担同步语义。
- `TBuf<VECCALC>`：纯计算临时区、归约 tmp、FP32 工作区，不入队出队。
- `TQueBind`：纯搬运或 VECIN/VECOUT 可复用场景，减少 UB 内冗余拷贝。
- Workspace：需要跨核同步、Matmul 系统 workspace 或中间 GM 缓存时，在 host tiling/shape 推导中申请并记录大小。

`TPipe::InitBuffer` 总 buffer 数、单 TPosition 连续 `AllocTensor` 数、EventID 数都有平台限制。优化时如果出现 buffer 不足，优先合并 buffer 后用 offset 切分，或释放不用的 queue 事件；不要盲目增加队列。

## DataCopy 与搬运

生产代码中 GM <-> UB 默认使用 `DataCopyPad`，`DataCopy` 只适合 UB 内部拷贝，或非常确定的 32B 对齐原型路径。

`DataCopyExtParams` 单位要分清：

| 字段 | 含义 | 单位 |
| --- | --- | --- |
| `blockCount` | 数据块数量，常用于行数 | 个数，最大通常按 API 限制处理 |
| `blockLen` | 每块有效搬运长度 | 字节 |
| `srcStride` | 源相邻块间隔 | GM 侧是字节，UB 侧是 32B DataBlock |
| `dstStride` | 目的相邻块间隔 | GM 侧是字节，UB 侧是 32B DataBlock |

非连续多行搬运优先用 stride 描述为一条 DMA，不要在 kernel 里循环发很多小 DMA。单次搬运量过小通常带宽利用率差，设计 tile 时优先让 GM <-> UB 搬运达到可观批量，经验阈值可用 16KB 作为检查线，但最终以目标 profiler 为准。

`rLength` 和 `rLengthAlign` 不可混用：

- `blockLen` 和 Vector API `count` 使用有效长度 `rLength`。
- UB 行偏移、InitBuffer、UB stride 使用对齐长度 `rLengthAlign`。
- CopyOut 的 UB `srcStride` 是块间 padding 间隔，不是完整行长度。
- CopyIn 使用 DataCopyPad 时，CopyOut 也应保持 DataCopyPad 语义，避免行错位。

GlobalTensor 下标是元素偏移，不是字节偏移。`xGm[tileOffset * sizeof(T)]` 这类写法通常是 bug。

## 同步与流水

MTE2/MTE3 搬运是异步的。`DataCopyPad` 返回不代表数据已经可被 Vector 读取。正确同步路径是：

```text
CopyIn:  AllocTensor -> DataCopyPad -> EnQue
Compute: DeQue -> Vector/Cube -> EnQue/FreeTensor
CopyOut: DeQue -> DataCopyPad -> FreeTensor
```

`PipeBarrier<PIPE_ALL>()` 可以作为定位实验，不应作为最终修复。最终修复应落到正确的 `EnQue/DeQue` 配对、TPosition 选择、事件同步或必要的细粒度 barrier。

Double buffer 的价值是让 MTE 和 Vector/Cube 重叠，不是简单把内存翻倍。满足以下条件时优先考虑：

- tile 循环次数至少 2。
- 搬运时间和计算时间接近，存在可重叠空间。
- UB 容量允许输入/输出队列各保留 2 块。

当计算远大于搬运、小 shape 一次完成、或 UB 已经紧张时，double buffer 可能没有收益。

MIX Matmul 场景中，`Iterate<false>()`/异步 Iterate 可减少 AIC/AIV 每轮同步消息；只有存在地址踩踏或严格顺序依赖时才回退同步模式。

## Vector 与数值规则

FP16/BF16 在复杂数学、归约、归一化、排序、池化等路径中应升到 FP32 中间精度。纯搬运或索引类算子，如果没有数学语义，可以把 FP16/BF16 当 2 字节 payload 处理。

高维 Vector API 的 `repeatTime` 是 `uint8_t`，超过 255 会静默截断。Host tiling 可限制每批行数，kernel 也要对超 255 的 batch 分段。

常用优先级：

- 行级 `x - scalar`、`x * scalar`、`x / scalar`：优先 `Adds/Muls`，避免 `Duplicate` 标量 tensor 再做二元运算。
- 同一元素数量的多条 Vector 指令：考虑 Counter 模式减少主块/尾块 mask 计算。
- `Compare`：参与比较数据区域按 256B 对齐，不足部分填充不会影响语义的极值。
- `ReduceSum/ReduceMax`：`count` 用有效长度，`dst` 不能与 `tmpBuffer` alias，源数据后续还要用时要避免被 API 覆盖。
- Pattern 归约：适合多行对齐数据，`alignedCols` 必须满足 32B 对齐；非对齐或复杂尾行退回 Level 2 逐行归约。

## 性能诊断顺序

按 profiler 或日志证据定位，不从直觉直接改：

1. Scalar 高：检查类内 `TPipe`、过多 mask/tail 分支、for 循环小 DMA、重复 offset 计算、`repeatTime` 分批是否合理。
2. MTE 高：检查单次搬运量、GM 512B 起始对齐、stride DMA、DataCopyPad 参数、是否 GM 往返中间结果。
3. Vector 高：检查 API 选择、FP32 临时量是否必要、Counter 模式、融合 `Adds/Muls`、UB 内链式计算。
4. Cube 高：检查 L0C 累加、较小矩阵 L1 常驻、Matmul AtomicAdd、BT/FP Buffer 融合。
5. 同步空洞：检查 CopyIn/Compute/CopyOut 是否有足够 tile 重叠、TQue 深度和 buffer 数是否匹配。

优化候选：

- UB 融合：连续 Vector 运算尽量留在 UB，避免每个子算子 GM 往返。
- L0C 累加：`A1*B1 + A2*B2 + ...` 类场景用 Mmad 累加，不把中间 C 搬到 GM 后再加。
- 小矩阵常驻 L1：L1 放不下两个矩阵时，优先让较小/复用高的一侧常驻。
- BT Buffer：bias 可融合进 Mmad 时，不单独走 CO1->GM->UB->Add。
- FP Buffer/Fixpipe：量化参数可放在搬出路径上融合，避免 UB 单独量化再写 GM。
- Matmul AtomicAdd：需要把结果累加到 GM 上已有矩阵时，优先考虑搬出路径原子累加。

## 精度与运行时调试

现象到排查方向：

| 现象 | 优先排查 |
| --- | --- |
| 全 1 输入通过，随机/等差失败 | GM offset、stride、tile 起点单位 |
| 首错在 tile 边界 | CopyIn/CopyOut 偏移或 `curTileLength` |
| 只在 tail 失败 | tail tile 长度、padding、CopyOut 有效长度 |
| 多次运行结果不同 | MTE 异步同步、EnQue/DeQue、未初始化变量 |
| 单核通过，多核失败 | block tiling 重叠/遗漏、workspace 分区、atomic 初始化 |
| FP32 通过，FP16/BF16 失败 | 缺少 FP32 中间精度或 Cast round mode 不合适 |
| 输出全 0 | CopyOut 偏移、AtomicNone 未恢复、输出 GM 未写到目标位置 |
| NaN/Inf | 除零、非法 domain、exp overflow、padding 值参与计算 |

调试顺序：先做误差分布和首错定位，再做受控实验。可以临时缩小到单 core、单 tile、固定小 shape、dump 0 核关键中间值；验证后删除调试代码。

MSSanitizer 适合检查非法访问、未初始化、泄漏、race 和同步问题。空日志通常表示该检查未发现问题，不等于精度一定正确。`--check-device-heap` 与 `--check-cann-heap` 不要同时开。

## 代码评审硬约束

Host/op_host：

- Optional input desc/tensor 使用前必须判空。
- workspace、输出 tensor、临时对象申请后必须检查返回值。
- 外部 shape/axis/index 进入数组索引或 tiling 计算前必须做范围检查。
- 不改变已有 TilingData 字段顺序和历史语义；新增字段优先追加并用 TilingKey 分支保护。

Kernel/op_kernel：

- 禁止 `std::vector/new/delete/malloc/free` 等动态分配。
- 禁止在 hot path 使用 `GlobalTensor::GetValue/SetValue`。
- 禁止未初始化成员变量；状态变量给默认值。
- AtomicAdd 前确认 UB src 和 GM dst 初始化，AtomicAdd 后恢复 AtomicNone。
- 资源申请和释放配对，`AllocTensor/FreeTensor`、`EnQue/DeQue` 不能跨异常分支失衡。
- 头块/尾块如与中间块逻辑不同，必须有明确注释或测试覆盖。

## 何时读取代码示例

需要修改具体 Ascend C 源码时读取 `ascendc-examples.md`：

- 写或改 Tiling 计算。
- 改 DataCopyPad 多行/非对齐搬运。
- 修复 tail、repeatTime、Compare、Reduce tmp、FP32 upcast。
- 做 TPipe 外置、TQue/TBuf、TQueBind、double buffer 或 Counter 模式优化。
