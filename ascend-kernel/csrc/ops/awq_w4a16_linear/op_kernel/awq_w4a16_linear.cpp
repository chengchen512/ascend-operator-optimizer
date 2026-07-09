// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "kernel_operator.h"

using AscendC::int4b_t;

namespace {
constexpr uint32_t kGroupSize = 128;
constexpr uint32_t kInt4PerPack = 8;
constexpr uint32_t kTilePack = 512;
constexpr uint32_t kTileOut = kTilePack * kInt4PerPack;
constexpr uint32_t kDeqTileN = 512;
constexpr uint32_t kDeqTileElems = kGroupSize * kDeqTileN;
constexpr uint32_t kDeqWeightBytes = kDeqTileElems / 2;
constexpr uint32_t kDeqHalfBytes = kDeqTileElems * sizeof(half);
constexpr uint32_t kDeqParamBytes = kDeqTileN * sizeof(half);

__aicore__ inline float HalfToFloat(half value)
{
    return static_cast<float>(value);
}

__aicore__ inline half FloatToHalf(float value)
{
    return static_cast<half>(value);
}

__aicore__ inline int32_t DecodeSignedInt4(int32_t packed, uint32_t lane)
{
    int32_t v = (packed >> (lane * 4U)) & 0xF;
    return v >= 8 ? v - 16 : v;
}
}  // namespace

class KernelAwqW4A16Linear {
public:
    __aicore__ inline KernelAwqW4A16Linear() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR scale, GM_ADDR offset,
                                GM_ADDR y, GM_ADDR workspace,
                                uint32_t m, uint32_t k, uint32_t n,
                                uint32_t mode, uint32_t colOffset, uint32_t outN)
    {
        (void)workspace;
        m_ = m;
        k_ = k;
        n_ = n;
        mode_ = mode;
        colOffset_ = colOffset;
        outN_ = outN == 0U ? n : outN;
        packedN_ = n / kInt4PerPack;
        groupNum_ = k / kGroupSize;
        totalPack_ = m * packedN_;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(x), static_cast<uint64_t>(m) * k);
        weightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(weight), static_cast<uint64_t>(k) * packedN_);
        weight4Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int4b_t *>(weight), static_cast<uint64_t>(k) * n);
        scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(scale), static_cast<uint64_t>(k / kGroupSize) * n);
        offsetGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(offset), static_cast<uint64_t>(k / kGroupSize) * n);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(y), static_cast<uint64_t>(m) * outN_);
        pipe_.InitBuffer(outQueue_, 1, kTileOut * sizeof(half));
        pipe_.InitBuffer(deqWeightQueue_, 1, kDeqWeightBytes);
        pipe_.InitBuffer(deqScaleQueue_, 1, kDeqParamBytes);
        pipe_.InitBuffer(deqOffsetQueue_, 1, kDeqParamBytes);
        pipe_.InitBuffer(deqOutQueue_, 1, kDeqHalfBytes);
    }

    __aicore__ inline void Process()
    {
        if (mode_ == 1U || mode_ == 2U) {
            ProcessDequant();
            return;
        }
        ProcessScalarPack();
    }

private:
    __aicore__ inline void ProcessDequant()
    {
        const uint32_t nTileCount = (outN_ + kDeqTileN - 1U) / kDeqTileN;
        const uint32_t taskCount = groupNum_ * nTileCount;
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t blockNum = AscendC::GetBlockNum();
        for (uint32_t task = blockIdx; task < taskCount; task += blockNum) {
            const uint32_t group = task / nTileCount;
            const uint32_t nTileIdx = task - group * nTileCount;
            const uint32_t outColBase = nTileIdx * kDeqTileN;
            uint32_t tileN = kDeqTileN;
            if (outColBase + tileN > outN_) {
                tileN = outN_ - outColBase;
            }
            DequantGroupTile(group, colOffset_ + outColBase, outColBase, tileN);
        }
    }

    __aicore__ inline void DequantGroupTile(uint32_t group, uint32_t sourceColBase,
                                            uint32_t outColBase, uint32_t tileN)
    {
        AscendC::LocalTensor<int4b_t> wLocal = deqWeightQueue_.AllocTensor<int4b_t>();
        AscendC::LocalTensor<half> scaleLocal = deqScaleQueue_.AllocTensor<half>();
        AscendC::LocalTensor<half> offsetLocal = deqOffsetQueue_.AllocTensor<half>();
        AscendC::LocalTensor<half> outLocal = deqOutQueue_.AllocTensor<half>();

        const uint32_t rowBase = group * kGroupSize;
        const uint32_t weightBlockBytes = (tileN + 1U) >> 1;
        AscendC::DataCopyExtParams weightCopy{
            static_cast<uint16_t>(kGroupSize),
            weightBlockBytes,
            static_cast<uint32_t>((n_ - tileN) >> 1),
            0,
            0};
        AscendC::DataCopyPadExtParams<int4b_t> weightPad{false, 0, 0, 0};
        AscendC::DataCopyPad(wLocal, weight4Gm_[rowBase * n_ + sourceColBase], weightCopy, weightPad);
        deqWeightQueue_.EnQue<int4b_t>(wLocal);

        AscendC::DataCopyExtParams paramCopy{1, static_cast<uint32_t>(tileN * sizeof(half)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<half> halfPad{false, 0, 0, 0};
        AscendC::DataCopyPad(scaleLocal, scaleGm_[group * n_ + sourceColBase], paramCopy, halfPad);
        deqScaleQueue_.EnQue<half>(scaleLocal);
        AscendC::DataCopyPad(offsetLocal, offsetGm_[group * n_ + sourceColBase], paramCopy, halfPad);
        deqOffsetQueue_.EnQue<half>(offsetLocal);

        wLocal = deqWeightQueue_.DeQue<int4b_t>();
        scaleLocal = deqScaleQueue_.DeQue<half>();
        offsetLocal = deqOffsetQueue_.DeQue<half>();

        const uint32_t elemCount = kGroupSize * tileN;
        AscendC::Cast(outLocal, wLocal, AscendC::RoundMode::CAST_NONE, elemCount);
        AscendC::PipeBarrier<PIPE_V>();

        const uint8_t rowStrideBlocks = static_cast<uint8_t>((tileN * sizeof(half)) >> 5);
        AscendC::BinaryRepeatParams params{1, 1, 1, rowStrideBlocks, rowStrideBlocks, 0};
        const uint32_t subTileCount = (tileN + 127U) >> 7;
        for (uint32_t sub = 0; sub < subTileCount; ++sub) {
            const uint32_t subCol = sub << 7;
            uint64_t mask = tileN - subCol;
            if (mask > 128U) {
                mask = 128U;
            }
            if (mode_ != 2U) {
                AscendC::Add(outLocal[subCol], outLocal[subCol], offsetLocal[subCol], mask, kGroupSize, params);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Mul(outLocal[subCol], outLocal[subCol], scaleLocal[subCol], mask, kGroupSize, params);
            AscendC::PipeBarrier<PIPE_V>();
        }

        deqOutQueue_.EnQue<half>(outLocal);
        outLocal = deqOutQueue_.DeQue<half>();
        AscendC::DataCopyExtParams outCopy{
            static_cast<uint16_t>(kGroupSize),
            static_cast<uint32_t>(tileN * sizeof(half)),
            0,
            static_cast<uint32_t>((outN_ - tileN) * sizeof(half)),
            0};
        AscendC::DataCopyPad(yGm_[rowBase * outN_ + outColBase], outLocal, outCopy);

        deqWeightQueue_.FreeTensor(wLocal);
        deqScaleQueue_.FreeTensor(scaleLocal);
        deqOffsetQueue_.FreeTensor(offsetLocal);
        deqOutQueue_.FreeTensor(outLocal);
    }

    __aicore__ inline void ProcessScalarPack()
    {
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t blockNum = AscendC::GetBlockNum();
        const uint32_t blockStride = blockNum * kTilePack;
        for (uint32_t packBase = blockIdx * kTilePack; packBase < totalPack_; packBase += blockStride) {
            uint32_t packCount = kTilePack;
            if (packBase + packCount > totalPack_) {
                packCount = totalPack_ - packBase;
            }
            ComputePackTile(packBase, packCount);
        }
    }

    __aicore__ inline void ComputeOnePack(AscendC::LocalTensor<half> outLocal,
                                          uint32_t localBase,
                                          uint32_t row,
                                          uint32_t packCol)
    {
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;
        float acc4 = 0.0f;
        float acc5 = 0.0f;
        float acc6 = 0.0f;
        float acc7 = 0.0f;

        for (uint32_t group = 0; group < groupNum_; ++group) {
            float raw0 = 0.0f;
            float raw1 = 0.0f;
            float raw2 = 0.0f;
            float raw3 = 0.0f;
            float raw4 = 0.0f;
            float raw5 = 0.0f;
            float raw6 = 0.0f;
            float raw7 = 0.0f;
            float sumX = 0.0f;
            const uint32_t kBase = group * kGroupSize;
            for (uint32_t kk = 0; kk < kGroupSize; ++kk) {
                const uint32_t kIndex = kBase + kk;
                const float xVal = HalfToFloat(xGm_.GetValue(static_cast<uint64_t>(row) * k_ + kIndex));
                const int32_t packed = weightGm_.GetValue(static_cast<uint64_t>(kIndex) * packedN_ + packCol);
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

            const uint32_t paramBase = group * n_ + packCol * kInt4PerPack;
            acc0 += (raw0 + sumX * HalfToFloat(offsetGm_.GetValue(paramBase))) *
                HalfToFloat(scaleGm_.GetValue(paramBase));
            acc1 += (raw1 + sumX * HalfToFloat(offsetGm_.GetValue(paramBase + 1))) *
                HalfToFloat(scaleGm_.GetValue(paramBase + 1));
            acc2 += (raw2 + sumX * HalfToFloat(offsetGm_.GetValue(paramBase + 2))) *
                HalfToFloat(scaleGm_.GetValue(paramBase + 2));
            acc3 += (raw3 + sumX * HalfToFloat(offsetGm_.GetValue(paramBase + 3))) *
                HalfToFloat(scaleGm_.GetValue(paramBase + 3));
            acc4 += (raw4 + sumX * HalfToFloat(offsetGm_.GetValue(paramBase + 4))) *
                HalfToFloat(scaleGm_.GetValue(paramBase + 4));
            acc5 += (raw5 + sumX * HalfToFloat(offsetGm_.GetValue(paramBase + 5))) *
                HalfToFloat(scaleGm_.GetValue(paramBase + 5));
            acc6 += (raw6 + sumX * HalfToFloat(offsetGm_.GetValue(paramBase + 6))) *
                HalfToFloat(scaleGm_.GetValue(paramBase + 6));
            acc7 += (raw7 + sumX * HalfToFloat(offsetGm_.GetValue(paramBase + 7))) *
                HalfToFloat(scaleGm_.GetValue(paramBase + 7));
        }

        outLocal.SetValue(localBase, FloatToHalf(acc0));
        outLocal.SetValue(localBase + 1, FloatToHalf(acc1));
        outLocal.SetValue(localBase + 2, FloatToHalf(acc2));
        outLocal.SetValue(localBase + 3, FloatToHalf(acc3));
        outLocal.SetValue(localBase + 4, FloatToHalf(acc4));
        outLocal.SetValue(localBase + 5, FloatToHalf(acc5));
        outLocal.SetValue(localBase + 6, FloatToHalf(acc6));
        outLocal.SetValue(localBase + 7, FloatToHalf(acc7));
    }

    __aicore__ inline void ComputePackTile(uint32_t packBase, uint32_t packCount)
    {
        AscendC::LocalTensor<half> outLocal = outQueue_.AllocTensor<half>();
        for (uint32_t i = 0; i < packCount; ++i) {
            const uint32_t packIndex = packBase + i;
            const uint32_t row = packIndex / packedN_;
            const uint32_t packCol = packIndex - row * packedN_;
            ComputeOnePack(outLocal, i * kInt4PerPack, row, packCol);
        }
        outQueue_.EnQue<half>(outLocal);
        outLocal = outQueue_.DeQue<half>();
        const uint32_t outBase = packBase * kInt4PerPack;
        const uint32_t outCount = packCount * kInt4PerPack;
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(outCount * sizeof(half)), 0, 0, 0};
        AscendC::DataCopyPad(yGm_[outBase], outLocal, copyParams);
        outQueue_.FreeTensor(outLocal);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> deqWeightQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> deqScaleQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> deqOffsetQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> deqOutQueue_;
    AscendC::GlobalTensor<half> xGm_;
    AscendC::GlobalTensor<int32_t> weightGm_;
    AscendC::GlobalTensor<int4b_t> weight4Gm_;
    AscendC::GlobalTensor<half> scaleGm_;
    AscendC::GlobalTensor<half> offsetGm_;
    AscendC::GlobalTensor<half> yGm_;
    uint32_t m_{0};
    uint32_t k_{0};
    uint32_t n_{0};
    uint32_t mode_{0};
    uint32_t colOffset_{0};
    uint32_t outN_{0};
    uint32_t packedN_{0};
    uint32_t groupNum_{0};
    uint32_t totalPack_{0};
};

extern "C" __global__ __aicore__ void awq_w4a16_linear(GM_ADDR x, GM_ADDR weight, GM_ADDR scale, GM_ADDR offset,
                                                       GM_ADDR y, GM_ADDR workspace,
                                                       uint32_t m, uint32_t k, uint32_t n,
                                                       uint32_t mode, uint32_t colOffset, uint32_t outN)
{
    KernelAwqW4A16Linear op;
    op.Init(x, weight, scale, offset, y, workspace, m, k, n, mode, colOffset, outN);
    op.Process();
}
