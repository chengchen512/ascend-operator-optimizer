// Licensed under the BSD 3-Clause License (the "License").

#include "kernel_operator.h"

using AscendC::int4b_t;

constexpr uint32_t PACK_WIDTH = 8;
constexpr uint32_t TILE_PACK = 512;
constexpr uint32_t TILE_OUT = TILE_PACK * PACK_WIDTH;
constexpr uint32_t GROUP_SIZE = 128;
constexpr uint32_t DEQ_TILE_N = 512;
constexpr uint32_t DEQ_TILE_ELEMS = GROUP_SIZE * DEQ_TILE_N;
constexpr uint32_t DEQ_WEIGHT_BYTES = DEQ_TILE_ELEMS / 2;
constexpr uint32_t DEQ_HALF_BYTES = DEQ_TILE_ELEMS * sizeof(half);
constexpr uint32_t DEQ_PARAM_BYTES = DEQ_TILE_N * sizeof(half);

class KernelAwqW4A16Linear {
public:
    __aicore__ inline KernelAwqW4A16Linear() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR scale, GM_ADDR offset, GM_ADDR output,
                                uint32_t m, uint32_t k, uint32_t n, uint32_t total, uint32_t mode,
                                uint32_t colOffset, uint32_t outN, AscendC::TPipe *pipeIn)
    {
        this->pipe = pipeIn;
        this->m = m;
        this->k = k;
        this->n = n;
        this->mode = mode;
        this->colOffset = colOffset;
        this->outN = outN == 0U ? n : outN;
        this->packedN = n >> 3;
        this->groupNum = k >> 7;
        this->total = total;
        this->totalPack = m * this->packedN;
        xGm.SetGlobalBuffer((__gm__ half *)x, static_cast<uint64_t>(m) * k);
        weightGm.SetGlobalBuffer((__gm__ int32_t *)weight, static_cast<uint64_t>(k) * this->packedN);
        weight4Gm.SetGlobalBuffer((__gm__ int4b_t *)weight, static_cast<uint64_t>(k) * n);
        scaleGm.SetGlobalBuffer((__gm__ half *)scale, static_cast<uint64_t>(this->groupNum) * n);
        offsetGm.SetGlobalBuffer((__gm__ half *)offset, static_cast<uint64_t>(this->groupNum) * n);
        outputGm.SetGlobalBuffer((__gm__ half *)output, static_cast<uint64_t>(total));
        pipe->InitBuffer(outQueue, 1, TILE_OUT * sizeof(half));
        pipe->InitBuffer(deqWeightQueue, 1, DEQ_WEIGHT_BYTES);
        pipe->InitBuffer(deqScaleQueue, 1, DEQ_PARAM_BYTES);
        pipe->InitBuffer(deqOffsetQueue, 1, DEQ_PARAM_BYTES);
        pipe->InitBuffer(deqOutQueue, 1, DEQ_HALF_BYTES);
    }

    __aicore__ inline void Process()
    {
        if (this->mode == 1U) {
            ProcessDequant();
            return;
        }
        ProcessScalarLinear();
    }

private:
    __aicore__ inline void ProcessScalarLinear()
    {
        const uint32_t blockNum = AscendC::GetBlockNum();
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t blockStride = blockNum * TILE_PACK;
        for (uint32_t packBase = blockIdx * TILE_PACK; packBase < this->totalPack; packBase += blockStride) {
            uint32_t packCount = TILE_PACK;
            if (packBase + packCount > this->totalPack) {
                packCount = this->totalPack - packBase;
            }
            ComputePackTile(packBase, packCount);
        }
    }

    __aicore__ inline void ProcessDequant()
    {
        const uint32_t nTileCount = (this->outN + DEQ_TILE_N - 1U) / DEQ_TILE_N;
        const uint32_t taskCount = this->groupNum * nTileCount;
        const uint32_t blockNum = AscendC::GetBlockNum();
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        for (uint32_t task = blockIdx; task < taskCount; task += blockNum) {
            const uint32_t group = task / nTileCount;
            const uint32_t nTileIdx = task - group * nTileCount;
            const uint32_t outColBase = nTileIdx * DEQ_TILE_N;
            uint32_t tileN = DEQ_TILE_N;
            if (outColBase + tileN > this->outN) {
                tileN = this->outN - outColBase;
            }
            DequantGroupTile(group, this->colOffset + outColBase, outColBase, tileN);
        }
    }

    __aicore__ inline void DequantGroupTile(uint32_t group, uint32_t sourceColBase, uint32_t outColBase, uint32_t tileN)
    {
        AscendC::LocalTensor<int4b_t> wLocal = deqWeightQueue.AllocTensor<int4b_t>();
        AscendC::LocalTensor<half> scaleLocal = deqScaleQueue.AllocTensor<half>();
        AscendC::LocalTensor<half> offsetLocal = deqOffsetQueue.AllocTensor<half>();
        AscendC::LocalTensor<half> outLocal = deqOutQueue.AllocTensor<half>();

        const uint32_t rowBase = group * GROUP_SIZE;
        const uint32_t weightBlockBytes = (tileN + 1U) >> 1;
        AscendC::DataCopyExtParams weightCopy{GROUP_SIZE, weightBlockBytes,
            static_cast<uint32_t>((this->n - tileN) >> 1), 0, 0};
        AscendC::DataCopyPadExtParams<int4b_t> weightPad{false, 0, 0, 0};
        AscendC::DataCopyPad(wLocal, weight4Gm[rowBase * this->n + sourceColBase], weightCopy, weightPad);
        deqWeightQueue.EnQue<int4b_t>(wLocal);

        AscendC::DataCopyExtParams paramCopy{1, static_cast<uint32_t>(tileN * sizeof(half)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<half> halfPad{false, 0, 0, 0};
        AscendC::DataCopyPad(scaleLocal, scaleGm[group * this->n + sourceColBase], paramCopy, halfPad);
        deqScaleQueue.EnQue<half>(scaleLocal);
        AscendC::DataCopyPad(offsetLocal, offsetGm[group * this->n + sourceColBase], paramCopy, halfPad);
        deqOffsetQueue.EnQue<half>(offsetLocal);

        wLocal = deqWeightQueue.DeQue<int4b_t>();
        scaleLocal = deqScaleQueue.DeQue<half>();
        offsetLocal = deqOffsetQueue.DeQue<half>();

        const uint32_t elemCount = GROUP_SIZE * tileN;
        AscendC::Cast(outLocal, wLocal, AscendC::RoundMode::CAST_NONE, elemCount);
        AscendC::PipeBarrier<PIPE_V>();

        const uint8_t halfRowStrideBlocks = static_cast<uint8_t>((tileN * sizeof(half)) >> 5);
        AscendC::BinaryRepeatParams broadcastParams{1, 1, 1, halfRowStrideBlocks, halfRowStrideBlocks, 0};
        const uint32_t subTileCount = (tileN + 127U) >> 7;
        for (uint32_t sub = 0; sub < subTileCount; ++sub) {
            const uint32_t subCol = sub << 7;
            uint64_t mask = tileN - subCol;
            if (mask > 128U) {
                mask = 128U;
            }
            AscendC::Add(outLocal[subCol], outLocal[subCol], offsetLocal[subCol], mask, GROUP_SIZE, broadcastParams);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(outLocal[subCol], outLocal[subCol], scaleLocal[subCol], mask, GROUP_SIZE, broadcastParams);
            AscendC::PipeBarrier<PIPE_V>();
        }

        deqOutQueue.EnQue<half>(outLocal);
        outLocal = deqOutQueue.DeQue<half>();
        AscendC::DataCopyExtParams outCopy{GROUP_SIZE, static_cast<uint32_t>(tileN * sizeof(half)), 0,
            static_cast<uint32_t>((this->outN - tileN) * sizeof(half)), 0};
        AscendC::DataCopyPad(outputGm[rowBase * this->outN + outColBase], outLocal, outCopy);

        deqWeightQueue.FreeTensor(wLocal);
        deqScaleQueue.FreeTensor(scaleLocal);
        deqOffsetQueue.FreeTensor(offsetLocal);
        deqOutQueue.FreeTensor(outLocal);
    }

    __aicore__ inline int32_t DecodeSignedInt4(int32_t packed, uint32_t lane) const
    {
        uint32_t rawPacked = static_cast<uint32_t>(packed);
        uint32_t raw = (rawPacked >> (lane << 2)) & 0xFU;
        return raw < 8U ? static_cast<int32_t>(raw) : static_cast<int32_t>(raw) - 16;
    }

    __aicore__ inline void ComputeOnePack(AscendC::LocalTensor<half> outLocal, uint32_t localBase,
                                          uint32_t row, uint32_t packCol)
    {
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;
        float acc4 = 0.0f;
        float acc5 = 0.0f;
        float acc6 = 0.0f;
        float acc7 = 0.0f;

        for (uint32_t group = 0; group < this->groupNum; ++group) {
            float raw0 = 0.0f;
            float raw1 = 0.0f;
            float raw2 = 0.0f;
            float raw3 = 0.0f;
            float raw4 = 0.0f;
            float raw5 = 0.0f;
            float raw6 = 0.0f;
            float raw7 = 0.0f;
            float sumX = 0.0f;
            const uint32_t kBase = group << 7;
            for (uint32_t kk = 0; kk < 128U; ++kk) {
                const uint32_t kIndex = kBase + kk;
                const float xVal = static_cast<float>(xGm.GetValue(row * this->k + kIndex));
                const int32_t packed = weightGm.GetValue(kIndex * this->packedN + packCol);
                raw0 += xVal * static_cast<float>(DecodeSignedInt4(packed, 0));
                raw1 += xVal * static_cast<float>(DecodeSignedInt4(packed, 1));
                raw2 += xVal * static_cast<float>(DecodeSignedInt4(packed, 2));
                raw3 += xVal * static_cast<float>(DecodeSignedInt4(packed, 3));
                raw4 += xVal * static_cast<float>(DecodeSignedInt4(packed, 4));
                raw5 += xVal * static_cast<float>(DecodeSignedInt4(packed, 5));
                raw6 += xVal * static_cast<float>(DecodeSignedInt4(packed, 6));
                raw7 += xVal * static_cast<float>(DecodeSignedInt4(packed, 7));
                sumX += xVal;
            }
            const uint32_t paramBase = group * this->n + (packCol << 3);
            const float scale0 = static_cast<float>(scaleGm.GetValue(paramBase));
            const float scale1 = static_cast<float>(scaleGm.GetValue(paramBase + 1));
            const float scale2 = static_cast<float>(scaleGm.GetValue(paramBase + 2));
            const float scale3 = static_cast<float>(scaleGm.GetValue(paramBase + 3));
            const float scale4 = static_cast<float>(scaleGm.GetValue(paramBase + 4));
            const float scale5 = static_cast<float>(scaleGm.GetValue(paramBase + 5));
            const float scale6 = static_cast<float>(scaleGm.GetValue(paramBase + 6));
            const float scale7 = static_cast<float>(scaleGm.GetValue(paramBase + 7));
            const float offset0 = static_cast<float>(offsetGm.GetValue(paramBase));
            const float offset1 = static_cast<float>(offsetGm.GetValue(paramBase + 1));
            const float offset2 = static_cast<float>(offsetGm.GetValue(paramBase + 2));
            const float offset3 = static_cast<float>(offsetGm.GetValue(paramBase + 3));
            const float offset4 = static_cast<float>(offsetGm.GetValue(paramBase + 4));
            const float offset5 = static_cast<float>(offsetGm.GetValue(paramBase + 5));
            const float offset6 = static_cast<float>(offsetGm.GetValue(paramBase + 6));
            const float offset7 = static_cast<float>(offsetGm.GetValue(paramBase + 7));
            acc0 += (raw0 + sumX * offset0) * scale0;
            acc1 += (raw1 + sumX * offset1) * scale1;
            acc2 += (raw2 + sumX * offset2) * scale2;
            acc3 += (raw3 + sumX * offset3) * scale3;
            acc4 += (raw4 + sumX * offset4) * scale4;
            acc5 += (raw5 + sumX * offset5) * scale5;
            acc6 += (raw6 + sumX * offset6) * scale6;
            acc7 += (raw7 + sumX * offset7) * scale7;
        }

        outLocal.SetValue(localBase, static_cast<half>(acc0));
        outLocal.SetValue(localBase + 1, static_cast<half>(acc1));
        outLocal.SetValue(localBase + 2, static_cast<half>(acc2));
        outLocal.SetValue(localBase + 3, static_cast<half>(acc3));
        outLocal.SetValue(localBase + 4, static_cast<half>(acc4));
        outLocal.SetValue(localBase + 5, static_cast<half>(acc5));
        outLocal.SetValue(localBase + 6, static_cast<half>(acc6));
        outLocal.SetValue(localBase + 7, static_cast<half>(acc7));
    }

    __aicore__ inline void ComputePackTile(uint32_t packBase, uint32_t packCount)
    {
        AscendC::LocalTensor<half> outLocal = outQueue.AllocTensor<half>();
        for (uint32_t i = 0; i < packCount; ++i) {
            const uint32_t packIndex = packBase + i;
            const uint32_t row = packIndex / this->packedN;
            const uint32_t packCol = packIndex - row * this->packedN;
            ComputeOnePack(outLocal, i << 3, row, packCol);
        }
        outQueue.EnQue<half>(outLocal);
        outLocal = outQueue.DeQue<half>();
        const uint32_t outBase = packBase << 3;
        const uint32_t outCount = packCount << 3;
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(outCount * sizeof(half)), 0, 0, 0};
        AscendC::DataCopyPad(outputGm[outBase], outLocal, copyParams);
        outQueue.FreeTensor(outLocal);
    }

private:
    AscendC::TPipe *pipe;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> deqWeightQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> deqScaleQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> deqOffsetQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> deqOutQueue;
    AscendC::GlobalTensor<half> xGm;
    AscendC::GlobalTensor<int32_t> weightGm;
    AscendC::GlobalTensor<int4b_t> weight4Gm;
    AscendC::GlobalTensor<half> scaleGm;
    AscendC::GlobalTensor<half> offsetGm;
    AscendC::GlobalTensor<half> outputGm;
    uint32_t m;
    uint32_t k;
    uint32_t n;
    uint32_t mode;
    uint32_t colOffset;
    uint32_t outN;
    uint32_t packedN;
    uint32_t groupNum;
    uint32_t total;
    uint32_t totalPack;
};

extern "C" __global__ __aicore__ void awq_w4a16_linear(GM_ADDR x, GM_ADDR weight, GM_ADDR scale, GM_ADDR offset,
                                                        GM_ADDR output, uint32_t m, uint32_t k, uint32_t n,
                                                        uint32_t total, uint32_t mode, uint32_t colOffset,
                                                        uint32_t outN)
{
    AscendC::TPipe pipe;
    KernelAwqW4A16Linear op;
    op.Init(x, weight, scale, offset, output, m, k, n, total, mode, colOffset, outN, &pipe);
    op.Process();
}