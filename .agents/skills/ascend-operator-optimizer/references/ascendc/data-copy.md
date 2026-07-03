# Ascend C Data Copy Pack

## 生产路径

- GM <-> UB 优先使用 `DataCopyPad`。
- `DataCopy` 只用于确定对齐的原型路径、UB 内部拷贝，或目标仓库已有稳定模式。
- GlobalTensor 下标是元素偏移，不是字节偏移。
- 涉及 ND2NZ/NZ2ND、非连续搬运或 GM/UB/TSCM 目的位置变化时，按 `../api-lookup.md` 查官方 `DataCopy`/`DataCopyPad` 文档确认 overload，不要假设源和目的反向后仍可编译。

## 参数单位

| 字段 | 单位 |
| --- | --- |
| `blockLen` | 字节 |
| GM `srcStride/dstStride` | 字节 |
| UB `srcStride/dstStride` | 32B DataBlock |

## 性能检查

- 单次 GM <-> UB 搬运尽量达到可观批量，可用 16KB 作为初筛线，最终以 profiler 为准。
- GM 起始地址尽量 512B 友好。
- 多行非连续搬运优先用 stride DMA，不要 kernel for-loop 发大量小 DMA。
- 中间结果能留在 UB 就不要 GM 往返。

## 同步

`DataCopyPad` 是异步搬运。正确路径应通过队列表达同步：

```text
CopyIn:  AllocTensor -> DataCopyPad -> EnQue
Compute: DeQue -> Vector/Cube -> EnQue/FreeTensor
CopyOut: DeQue -> DataCopyPad -> FreeTensor
```

`PipeBarrier<PIPE_ALL>()` 只能作为定位实验，不应作为最终修复。
