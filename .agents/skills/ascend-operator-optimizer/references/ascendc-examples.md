# Ascend C 代码模式示例

这些示例来自 Ascend/agent-skills 中 AscendC references 的蒸馏和改写，保留关键 API 形状与易错点。用于优化已有算子时按需套用，不要直接复制常量或假设目标 shape。

## 1. 对齐工具

```cpp
constexpr int64_t CeilDiv(int64_t x, int64_t y) {
    return (x + y - 1) / y;
}

constexpr int64_t AlignUp(int64_t x, int64_t align) {
    return CeilDiv(x, align) * align;
}

constexpr int64_t AlignDown(int64_t x, int64_t align) {
    return (x / align) * align;
}

int64_t AlignElemsDownByBytes(int64_t elems, int64_t dtypeSize, int64_t alignBytes) {
    int64_t elemsPerAlign = alignBytes / dtypeSize;
    return AlignDown(elems, elemsPerAlign);
}
```

使用规则：GM/cache line 相关检查常看 512B，UB 行和 LocalTensor 相关检查常看 32B。实际容量和核数从平台接口获取。

## 2. Host 侧 tiling 骨架

```cpp
struct OpTilingData {
    int64_t totalLength;
    int64_t usedCoreNum;
    int64_t formerNum;
    int64_t formerLength;
    int64_t tailNum;
    int64_t tailLength;
    int64_t tileLength;
};

void FillElementwiseTiling(const std::vector<int64_t>& shape,
                           int64_t dtypeSize,
                           int64_t coreNum,
                           int64_t ubSize,
                           int64_t bufferCoefficient,
                           OpTilingData& tiling) {
    tiling.totalLength = 1;
    for (int64_t dim : shape) {
        tiling.totalLength *= dim;
    }

    int64_t elemsPer512B = 512 / dtypeSize;
    int64_t perCore = AlignUp(CeilDiv(tiling.totalLength, coreNum), elemsPer512B);
    tiling.usedCoreNum = CeilDiv(tiling.totalLength, perCore);
    tiling.formerNum = tiling.usedCoreNum > 0 ? tiling.usedCoreNum - 1 : 0;
    tiling.formerLength = perCore;
    tiling.tailNum = tiling.usedCoreNum > 0 ? 1 : 0;
    tiling.tailLength = tiling.totalLength - tiling.formerNum * tiling.formerLength;

    int64_t maxTileElems = ubSize / bufferCoefficient;
    tiling.tileLength = AlignElemsDownByBytes(maxTileElems, dtypeSize, 32);
}
```

`bufferCoefficient` 来自本算子的 UB 分配表：输入队列、输出队列、临时区、FP32 工作区、double buffer 都要计入。

## 3. 外置 TPipe

```cpp
template <typename T>
class KernelOp {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const OpTilingData* tiling, AscendC::TPipe* pipeIn) {
        pipe = pipeIn;
        this->tiling = tiling;
        xGm.SetGlobalBuffer((__gm__ T*)x, tiling->totalLength);
        yGm.SetGlobalBuffer((__gm__ T*)y, tiling->totalLength);
        pipe->InitBuffer(inQueue, 2, tiling->tileLength * sizeof(T));
        pipe->InitBuffer(outQueue, 2, tiling->tileLength * sizeof(T));
    }

private:
    AscendC::TPipe* pipe{nullptr};
    const OpTilingData* tiling{nullptr};
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> yGm;
};

extern "C" __global__ __aicore__ void op_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR tilingGm) {
    AscendC::TPipe pipe;
    KernelOp<half> op;
    op.Init(x, y, reinterpret_cast<__gm__ OpTilingData*>(tilingGm), &pipe);
    op.Process();
}
```

## 4. 核内 tail 循环

```cpp
__aicore__ inline void Process() {
    int64_t blockIdx = AscendC::GetBlockIdx();
    int64_t blockLength = blockIdx < tiling->formerNum ? tiling->formerLength : tiling->tailLength;
    int64_t blockOffset = blockIdx < tiling->formerNum
        ? blockIdx * tiling->formerLength
        : tiling->formerNum * tiling->formerLength;

    int64_t tileNum = CeilDiv(blockLength, tiling->tileLength);
    for (int64_t i = 0; i < tileNum; ++i) {
        int64_t curOffset = blockOffset + i * tiling->tileLength;
        int64_t curLen = (i == tileNum - 1) ? blockLength - i * tiling->tileLength : tiling->tileLength;
        CopyIn(curOffset, curLen);
        Compute(curLen);
        CopyOut(curOffset, curLen);
    }
}
```

GlobalTensor 下标是元素偏移，`curOffset` 不要再乘 `sizeof(T)`。

## 5. 连续 GM 到 UB 的 DataCopyPad

```cpp
__aicore__ inline void CopyIn(int64_t gmOffset, int64_t curLen) {
    AscendC::LocalTensor<half> xLocal = inQueue.AllocTensor<half>();
    AscendC::DataCopyExtParams copyParams{
        1,
        static_cast<uint32_t>(curLen * sizeof(half)),
        0,
        0,
        0
    };
    AscendC::DataCopyPadExtParams<half> padParams{false, 0, 0, 0};
    AscendC::DataCopyPad(xLocal, xGm[gmOffset], copyParams, padParams);
    inQueue.EnQue(xLocal);
}

__aicore__ inline void CopyOut(int64_t gmOffset, int64_t curLen) {
    AscendC::LocalTensor<half> yLocal = outQueue.DeQue<half>();
    AscendC::DataCopyExtParams copyParams{
        1,
        static_cast<uint32_t>(curLen * sizeof(half)),
        0,
        0,
        0
    };
    AscendC::DataCopyPad(yGm[gmOffset], yLocal, copyParams);
    outQueue.FreeTensor(yLocal);
}
```

## 6. 多行 DataCopyPad 与 stride 单位

```cpp
__aicore__ inline void CopyRowsIn(uint32_t startRow, uint32_t rows) {
    AscendC::LocalTensor<half> xLocal = inQueue.AllocTensor<half>();
    AscendC::DataCopyExtParams copyParams{
        static_cast<uint16_t>(rows),
        static_cast<uint32_t>(cols * sizeof(half)),
        static_cast<uint32_t>(totalCols * sizeof(half)),
        static_cast<uint16_t>(alignedCols * sizeof(half) / 32),
        0
    };
    AscendC::DataCopyPadExtParams<half> padParams{
        true,
        0,
        static_cast<uint8_t>(alignedCols - cols),
        0
    };
    AscendC::DataCopyPad(xLocal, xGm[startRow * totalCols], copyParams, padParams);
    inQueue.EnQue(xLocal);
}

__aicore__ inline void CopyRowsOut(uint32_t startRow, uint32_t rows) {
    AscendC::LocalTensor<half> yLocal = outQueue.DeQue<half>();
    AscendC::DataCopyExtParams copyParams{
        static_cast<uint16_t>(rows),
        static_cast<uint32_t>(cols * sizeof(half)),
        static_cast<uint16_t>((alignedCols - cols) * sizeof(half) / 32),
        static_cast<uint32_t>(totalCols * sizeof(half)),
        0
    };
    AscendC::DataCopyPad(yGm[startRow * totalCols], yLocal, copyParams);
    outQueue.FreeTensor(yLocal);
}
```

注意：GM stride 是字节间隔；UB stride 是 32B DataBlock 间隔。CopyOut 的 UB `srcStride` 是 padding 间隔。

## 7. FP16/BF16 升精度计算

```cpp
AscendC::TBuf<AscendC::TPosition::VECCALC> xFp32Buf;
AscendC::TBuf<AscendC::TPosition::VECCALC> yFp32Buf;

__aicore__ inline void InitBuffers(int64_t tileLength) {
    pipe->InitBuffer(xFp32Buf, tileLength * sizeof(float));
    pipe->InitBuffer(yFp32Buf, tileLength * sizeof(float));
}

__aicore__ inline void Compute(int64_t curLen) {
    AscendC::LocalTensor<half> xLocal = inQueue.DeQue<half>();
    AscendC::LocalTensor<half> yLocal = outQueue.AllocTensor<half>();
    AscendC::LocalTensor<float> xFp32 = xFp32Buf.Get<float>();
    AscendC::LocalTensor<float> yFp32 = yFp32Buf.Get<float>();

    AscendC::Cast(xFp32, xLocal, AscendC::RoundMode::CAST_NONE, curLen);
    AscendC::Exp(yFp32, xFp32, curLen);
    AscendC::Cast(yLocal, yFp32, AscendC::RoundMode::CAST_ROUND, curLen);

    outQueue.EnQue(yLocal);
    inQueue.FreeTensor(xLocal);
}
```

复杂数学、归约和归一化优先用独立 FP32 工作区，避免用尺寸不足的 reinterpret 区域承载扩大后的类型。

## 8. repeatTime 超 255 分批

```cpp
__aicore__ inline void SubRows(AscendC::LocalTensor<float> dst,
                               AscendC::LocalTensor<float> src,
                               AscendC::LocalTensor<float> rowValue,
                               int64_t rows,
                               int64_t alignedCols) {
    int64_t remaining = rows;
    int64_t rowOffset = 0;
    uint64_t mask = alignedCols;
    AscendC::BinaryRepeatParams params{1, 1, 1,
        static_cast<uint16_t>(alignedCols / 8),
        static_cast<uint16_t>(alignedCols / 8),
        0};

    while (remaining > 0) {
        uint8_t batch = static_cast<uint8_t>(remaining > 255 ? 255 : remaining);
        AscendC::Sub(dst[rowOffset], src[rowOffset], rowValue, mask, batch, params);
        rowOffset += static_cast<int64_t>(batch) * alignedCols;
        remaining -= batch;
    }
}
```

## 9. Reduce tmpBuffer 独立分配

```cpp
uint32_t TmpElemsForReduce(uint32_t count, uint32_t dtypeSize) {
    uint32_t elemsPerBlock = 32 / dtypeSize;
    uint32_t elemsPerRepeat = 256 / dtypeSize;
    uint32_t firstRepeat = (count + elemsPerRepeat - 1) / elemsPerRepeat;
    return ((firstRepeat + elemsPerBlock - 1) / elemsPerBlock) * elemsPerBlock;
}

__aicore__ inline void ReduceOneRow(uint32_t rLength) {
    AscendC::LocalTensor<float> x = inQueue.DeQue<float>();
    AscendC::LocalTensor<float> out = outQueue.AllocTensor<float>();
    AscendC::LocalTensor<float> tmp = reduceTmpBuf.Get<float>();

    AscendC::ReduceSum(out, x, tmp, rLength);

    outQueue.EnQue(out);
    inQueue.FreeTensor(x);
}
```

`out` 和 `tmp` 必须不同。`rLength` 是有效长度，不是 32B 对齐长度。

## 10. Compare 256B padding

```cpp
uint32_t alignElems = 256 / sizeof(float);
uint32_t alignedCount = ((count + alignElems - 1) / alignElems) * alignElems;
if (alignedCount > count) {
    AscendC::Duplicate(xLocal[count], -FLT_MAX, alignedCount - count);
}
AscendC::Compare(maskLocal, xLocal, thresholdLocal, AscendC::CMPMODE::GT, alignedCount);
```

ArgMax 类 padding 用不会胜出的最小值，ArgMin 类 padding 用不会胜出的最大值。

## 11. Counter 模式减少尾块 Scalar 开销

```cpp
AscendC::BinaryRepeatParams params{1, 1, 1, 8, 8, 8};
AscendC::SetMaskCount();
AscendC::SetVectorMask<half, AscendC::MaskMode::COUNTER>(elementCount);
AscendC::Add<half, false>(zLocal, xLocal, yLocal,
    AscendC::MASK_PLACEHOLDER, 1, params);
AscendC::ResetMask();
```

多条相同元素数的 Vector 指令可以复用 Counter 思路；确认目标 API 和平台支持后再替换。

## 12. TQueBind 纯搬运模式

```cpp
AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, 1> bindQueue;
pipe->InitBuffer(bindQueue, 2, tileLength * sizeof(half));

__aicore__ inline void ProcessCopyOnly(int64_t inOffset, int64_t outOffset, int64_t curLen) {
    AscendC::LocalTensor<half> local = bindQueue.AllocTensor<half>();
    AscendC::DataCopyExtParams inParams{1, static_cast<uint32_t>(curLen * sizeof(half)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<half> padParams{false, 0, 0, 0};
    AscendC::DataCopyPad(local, xGm[inOffset], inParams, padParams);
    bindQueue.EnQue(local);

    local = bindQueue.DeQue<half>();
    AscendC::DataCopyExtParams outParams{1, static_cast<uint32_t>(curLen * sizeof(half)), 0, 0, 0};
    AscendC::DataCopyPad(yGm[outOffset], local, outParams);
    bindQueue.FreeTensor(local);
}
```

用于避免 VECIN 到 VECOUT 的额外 LocalTensor 拷贝；如果中间有真实 Vector 计算，仍按常规 TQue/TBuf 设计。

## 13. AtomicAdd 保护

```cpp
InitOutput(outGm, totalLength, static_cast<T>(0));

SetAtomicAdd<T>();
AscendC::DataCopyPad(outGm[gmOffset], partialLocal, copyParams);
SetAtomicNone();
```

AtomicAdd 前确认 GM 目标和 UB 源尾部 padding 不含脏数据。AtomicAdd 后必须恢复 AtomicNone，避免影响后续写出。


