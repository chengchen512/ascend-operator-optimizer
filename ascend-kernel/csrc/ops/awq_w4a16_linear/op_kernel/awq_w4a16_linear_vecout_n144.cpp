#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace AwqW4A16LinearVecoutN144 {
using namespace AscendC;
using namespace matmul;

constexpr uint32_t GroupK = 128;
constexpr uint32_t GroupsPerMatmul = 2;
constexpr uint32_t MatmulK = GroupK * GroupsPerMatmul;
constexpr uint32_t BasicK = 128;
constexpr uint32_t TileN = 144;
constexpr uint32_t FirstVectorCols = 128;
constexpr uint32_t SecondVectorCols = TileN - FirstVectorCols;
constexpr uint32_t MaxM = 16;
constexpr uint32_t BTileElems = MatmulK * TileN;
constexpr uint32_t BTileInt4Bytes = BTileElems / 2;
constexpr uint32_t BTileHalfBytes = BTileElems * sizeof(half);
constexpr uint32_t ParamTileBytes = TileN * sizeof(half);
constexpr uint32_t BufferNum = 1;
constexpr int32_t L1SizeBytes = 524288;
static_assert(GroupsPerMatmul == 2, "This kernel unrolls two AWQ groups per Matmul call.");
static_assert(SecondVectorCols == 16, "The N144 kernel expects a 128+16 vector split.");

using AType = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
using BType = MatmulType<TPosition::VECOUT, CubeFormat::ND, half, false>;
using CType = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, float>;

__aicore__ constexpr MatmulConfig MakeAwqConfig()
{
    constexpr MatmulShapeParams shapeParams{MaxM, TileN, MatmulK, MaxM, TileN, BasicK};
    MatmulConfig cfg = GetMMConfig<MatmulConfigMode::CONFIG_NORM>(shapeParams);
    cfg.enableSetBias = false;
    cfg.enableSetTail = false;
    cfg.enableQuantVector = false;
    cfg.enableSetDefineData = false;
    cfg.iterateMode = IterateMode::ITERATE_MODE_ALL;
    return cfg;
}

__aicore__ constexpr MatmulApiStaticTiling MakeAwqStaticTiling()
{
    return AscendC::GetMatmulApiTiling<AType, BType, CType, BiasType>(MakeAwqConfig(), L1SizeBytes);
}

constexpr MatmulApiStaticTiling AWQ_CFG = MakeAwqStaticTiling();

class Kernel {
public:
    using Mm = Matmul<AType, BType, CType, BiasType, AWQ_CFG>;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR scale, GM_ADDR offset, GM_ADDR y,
                                uint32_t m, uint32_t k, uint32_t n, uint32_t colOffset)
    {
        m_ = m;
        k_ = k;
        n_ = n;
        colOffset_ = colOffset;
        groupNum_ = k / GroupK;
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(x), static_cast<uint64_t>(m) * k);
        weight4Gm.SetGlobalBuffer(reinterpret_cast<__gm__ int4b_t *>(weight), static_cast<uint64_t>(k) * n);
        scaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(scale), static_cast<uint64_t>(groupNum_) * n);
        offsetGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(offset), static_cast<uint64_t>(groupNum_) * n);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(y), static_cast<uint64_t>(m) * n);
    }

    __aicore__ inline void Process(AscendC::TPipe *pipe)
    {
        const uint32_t blockIdx = matmul::GetBlockIdx();
        const uint32_t colBase = colOffset_ + blockIdx * TileN;
        if (colBase + TileN > n_) {
            return;
        }

        TQue<TPosition::VECIN, 1> weight4Queue;
        TQue<TPosition::VECIN, 1> scaleQueue;
        TQue<TPosition::VECIN, 1> offsetQueue;
        TQue<TPosition::VECOUT, 1> vecoutQueue;
        pipe->InitBuffer(weight4Queue, BufferNum, BTileInt4Bytes);
        pipe->InitBuffer(scaleQueue, BufferNum, ParamTileBytes);
        pipe->InitBuffer(offsetQueue, BufferNum, ParamTileBytes);
        pipe->InitBuffer(vecoutQueue, BufferNum, BTileHalfBytes);

        DataCopyExtParams paramCopy{1, ParamTileBytes, 0, 0, 0};
        DataCopyPadExtParams<half> halfPad{false, 0, 0, 0};
        DataCopyExtParams weight4Copy{static_cast<uint16_t>(MatmulK), static_cast<uint32_t>(TileN / 2U),
                                      static_cast<uint32_t>((n_ - TileN) / 2U), 0, 0};
        DataCopyPadExtParams<int4b_t> int4Pad{false, 0, 0, 0};
        const uint32_t matmulGroupNum = groupNum_ / GroupsPerMatmul;
        mm.SetOrgShape(m_, TileN, k_, MatmulK, n_);
        mm.SetSingleShape(m_, TileN, MatmulK);

        for (uint32_t kg = 0; kg < matmulGroupNum; ++kg) {
            const uint32_t kBase = kg * MatmulK;

            LocalTensor<int4b_t> w4Local = weight4Queue.AllocTensor<int4b_t>();
            DataCopyPad(w4Local, weight4Gm[kBase * n_ + colBase], weight4Copy, int4Pad);
            weight4Queue.EnQue<int4b_t>(w4Local);
            w4Local = weight4Queue.DeQue<int4b_t>();

            LocalTensor<half> bVec = vecoutQueue.AllocTensor<half>();
            Cast(bVec, w4Local, RoundMode::CAST_NONE, BTileElems);
            PipeBarrier<PIPE_V>();
            weight4Queue.FreeTensor(w4Local);

            const uint32_t quantGroup0 = kg * GroupsPerMatmul;
            LocalTensor<half> scaleLocal0 = scaleQueue.AllocTensor<half>();
            LocalTensor<half> offsetLocal0 = offsetQueue.AllocTensor<half>();
            DataCopyPad(scaleLocal0, scaleGm[quantGroup0 * n_ + colBase], paramCopy, halfPad);
            scaleQueue.EnQue<half>(scaleLocal0);
            DataCopyPad(offsetLocal0, offsetGm[quantGroup0 * n_ + colBase], paramCopy, halfPad);
            offsetQueue.EnQue<half>(offsetLocal0);
            scaleLocal0 = scaleQueue.DeQue<half>();
            offsetLocal0 = offsetQueue.DeQue<half>();

            ApplyAntiquantGroup(bVec, scaleLocal0, offsetLocal0, 0);

            const uint32_t quantGroup1 = quantGroup0 + 1U;
            constexpr uint32_t secondGroupOffset = GroupK * TileN;
            LocalTensor<half> scaleLocal1 = scaleQueue.AllocTensor<half>();
            LocalTensor<half> offsetLocal1 = offsetQueue.AllocTensor<half>();
            DataCopyPad(scaleLocal1, scaleGm[quantGroup1 * n_ + colBase], paramCopy, halfPad);
            scaleQueue.EnQue<half>(scaleLocal1);
            DataCopyPad(offsetLocal1, offsetGm[quantGroup1 * n_ + colBase], paramCopy, halfPad);
            offsetQueue.EnQue<half>(offsetLocal1);
            scaleLocal1 = scaleQueue.DeQue<half>();
            offsetLocal1 = offsetQueue.DeQue<half>();

            ApplyAntiquantGroup(bVec, scaleLocal1, offsetLocal1, secondGroupOffset);
            PipeBarrier<PIPE_V>();
            scaleQueue.FreeTensor(scaleLocal0);
            offsetQueue.FreeTensor(offsetLocal0);
            scaleQueue.FreeTensor(scaleLocal1);
            offsetQueue.FreeTensor(offsetLocal1);

            vecoutQueue.EnQue<half>(bVec);
            bVec = vecoutQueue.DeQue<half>();

            mm.SetTensorA(xGm[kBase], false);
            mm.SetTensorB(bVec, false);
            mm.IterateAll(yGm[colBase], static_cast<uint8_t>(kg != 0U));

            vecoutQueue.FreeTensor(bVec);
        }
        mm.End();
    }

    Mm mm;

private:
    __aicore__ inline void ApplyAntiquantGroup(LocalTensor<half> bVec, LocalTensor<half> scaleLocal,
                                               LocalTensor<half> offsetLocal, uint32_t groupBase)
    {
        constexpr uint32_t secondSegmentOffset = FirstVectorCols;
        constexpr uint8_t rowStrideBlocks = static_cast<uint8_t>((TileN * sizeof(half)) >> 5);
        BinaryRepeatParams aqParams{1, 1, 1, rowStrideBlocks, rowStrideBlocks, 0};
        Add(bVec[groupBase], bVec[groupBase], offsetLocal, FirstVectorCols, static_cast<uint8_t>(GroupK), aqParams);
        PipeBarrier<PIPE_V>();
        Mul(bVec[groupBase], bVec[groupBase], scaleLocal, FirstVectorCols, static_cast<uint8_t>(GroupK), aqParams);
        PipeBarrier<PIPE_V>();
        Add(bVec[groupBase + secondSegmentOffset], bVec[groupBase + secondSegmentOffset],
            offsetLocal[secondSegmentOffset], SecondVectorCols, static_cast<uint8_t>(GroupK), aqParams);
        PipeBarrier<PIPE_V>();
        Mul(bVec[groupBase + secondSegmentOffset], bVec[groupBase + secondSegmentOffset],
            scaleLocal[secondSegmentOffset], SecondVectorCols, static_cast<uint8_t>(GroupK), aqParams);
    }

    GlobalTensor<half> xGm;
    GlobalTensor<int4b_t> weight4Gm;
    GlobalTensor<half> scaleGm;
    GlobalTensor<half> offsetGm;
    GlobalTensor<half> yGm;
    uint32_t m_{0};
    uint32_t k_{0};
    uint32_t n_{0};
    uint32_t colOffset_{0};
    uint32_t groupNum_{0};
};
}  // namespace AwqW4A16LinearVecoutN144

extern "C" __global__ __aicore__ void awq_w4a16_linear_vecout_n144(
    GM_ADDR x, GM_ADDR weight, GM_ADDR scale, GM_ADDR offset, GM_ADDR y, uint32_t m, uint32_t k, uint32_t n,
    uint32_t colOffset, GM_ADDR workspace, GM_ADDR tilingGm)
{
    AscendC::SetSysWorkspace(workspace);

    AscendC::TPipe pipe;

    AwqW4A16LinearVecoutN144::Kernel kernel;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), kernel.mm, (AscendC::tiling::TCubeTiling *)nullptr);
    kernel.Init(x, weight, scale, offset, y, m, k, n, colOffset);
    kernel.Process(&pipe);
}
