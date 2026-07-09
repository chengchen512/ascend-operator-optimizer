# Ascend C Memory Pack

## 层级

- GM/HBM：容量大，访问慢；避免中间结果反复往返。
- L2/L1：关注复用和切分方向。
- L0A/L0B/L0C：Cube/Matmul 路径核心资源。
- UB：Vector、搬运、临时量和队列主要压力点。
- BT Buffer / FP Buffer：bias、量化和 Fixpipe 融合机会。

## UB 预算

UB 预算必须包含：

- 输入队列和输出队列。
- 临时 `TBuf`。
- FP32 upcast buffer。
- index/mask/offset tensor。
- double buffer 的额外副本。
- API 要求的 tmpBuffer。

## 常见优化

- 连续 Vector 子计算尽量 UB 内融合。
- `A1*B1 + A2*B2 + ...` 类 Matmul 使用 L0C 累加，不把中间 C 写回 GM 再加。
- L1 放不下多侧矩阵时，让更小或复用更高的一侧常驻。
- bias 可进 BT Buffer 时，不走 CO1->GM->UB->Add。
- 量化参数可在 FP Buffer/Fixpipe 路径融合时，不单独 UB 量化再写 GM。
