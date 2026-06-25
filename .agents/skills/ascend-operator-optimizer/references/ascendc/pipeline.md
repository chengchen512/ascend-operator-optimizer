# Ascend C Pipeline Pack

## 目标

流水优化的目标是让 MTE、Vector、Cube 的可重叠部分真实重叠，而不是简单增加 buffer 数。

## 基本结构

```text
CopyIn -> Compute -> CopyOut
```

队列深度和 buffer 数需要和循环次数匹配。`AllocTensor/FreeTensor`、`EnQue/DeQue` 必须配对。

## Double Buffer 适用条件

- tile 循环次数至少 2。
- 搬运时间和计算时间接近，profile 中存在可重叠空间。
- UB 容量允许输入/输出队列各保留 2 块。

不适用：

- 小 shape 一次完成。
- 计算远大于搬运。
- UB 已经被 FP32 临时量、mask、offset、tmpBuffer 占满。

## MIX / Matmul

- MIX Matmul 可评估异步 Iterate，减少 AIC/AIV 每轮同步消息。
- 有地址踩踏、严格顺序依赖或同步证据不足时，不要盲目异步化。
- profile 出现 device idle gap 时，先确认 launch/profile 口径，再检查 CopyIn/Compute/CopyOut 是否真的有重叠空间。
