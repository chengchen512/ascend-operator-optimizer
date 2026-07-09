#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace AwqTscmVecoutDirectProbe {
using namespace AscendC;
using namespace matmul;

constexpr uint32_t M = 16;
constexpr uint32_t K = 4096;
constexpr uint32_t GroupK = 128;
constexpr uint32_t NumKGroups = K / GroupK;
constexpr uint32_t N = 22016;
constexpr uint32_t TileN = 256;
constexpr uint32_t NumTilesN = N / TileN;
constexpr uint32_t PackedN = N / 8;
constexpr uint32_t PackedElems = K * PackedN;
constexpr uint32_t Weight4Elems = K * N;
constexpr uint32_t BTileElems = GroupK * TileN;
constexpr uint32_t CElems = M * N;
constexpr uint32_t ParamElems = NumKGroups * N;
constexpr uint32_t PackedTileBytes = GroupK * (TileN / 2);
constexpr uint32_t BTileInt4Bytes = BTileElems / 2;
constexpr uint32_t BTileHalfBytes = BTileElems * sizeof(half);
constexpr uint32_t ParamTileBytes = TileN * sizeof(half);

using AType = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
using BType = MatmulType<TPosition::VECOUT, CubeFormat::ND, half, false>;
using CType = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, float>;

__aicore__ inline void CopyTiling(AscendC::tiling::TCubeTiling *tiling, GM_ADDR tilingGm)
{
    uint32_t *dst = reinterpret_cast<uint32_t *>(tiling);
    auto src = reinterpret_cast<__gm__ uint32_t *>(tilingGm);
    for (uint32_t i = 0; i < sizeof(AscendC::tiling::TCubeTiling) / sizeof(uint32_t); ++i) {
        dst[i] = src[i];
    }
}

__aicore__ inline int32_t DecodeInt4Value(int32_t packed, uint32_t lane)
{
    const uint32_t nibble = (static_cast<uint32_t>(packed) >> (lane * 4U)) & 0xFU;
    return static_cast<int32_t>(nibble < 8U ? nibble : (nibble - 16U));
}

class Kernel {
public:
    using Mm = Matmul<AType, BType, CType, BiasType, CFG_MDL>;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR scale, GM_ADDR offset, GM_ADDR y, uint32_t mode,
                                const AscendC::tiling::TCubeTiling &tilingIn)
    {
        mode_ = mode;
        tiling = tilingIn;
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(x), M * K);
        weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(weight), PackedElems);
        weight4Gm.SetGlobalBuffer(reinterpret_cast<__gm__ int4b_t *>(weight), Weight4Elems);
        scaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(scale), ParamElems);
        offsetGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(offset), ParamElems);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(y), CElems);
    }

    __aicore__ inline void Process(AscendC::TPipe *pipe)
    {
        const uint32_t blockIdx = matmul::GetBlockIdx();
        if (blockIdx >= NumTilesN) {
            return;
        }
        const uint32_t colBase = blockIdx * TileN;

        TQue<TPosition::VECIN, 1> packedQueue;
        TQue<TPosition::VECIN, 1> weight4Queue;
        TQue<TPosition::VECIN, 1> scaleQueue;
        TQue<TPosition::VECIN, 1> offsetQueue;
        TQue<TPosition::VECOUT, 1> vecoutQueue;
        pipe->InitBuffer(packedQueue, 1, PackedTileBytes);
        pipe->InitBuffer(weight4Queue, 1, BTileInt4Bytes);
        pipe->InitBuffer(scaleQueue, 1, ParamTileBytes);
        pipe->InitBuffer(offsetQueue, 1, ParamTileBytes);
        pipe->InitBuffer(vecoutQueue, 1, BTileHalfBytes);

        DataCopyExtParams paramCopy{1, ParamTileBytes, 0, 0, 0};
        DataCopyPadExtParams<half> halfPad{false, 0, 0, 0};

        for (uint32_t kg = 0; kg < NumKGroups; ++kg) {
            const uint32_t kBase = kg * GroupK;

            LocalTensor<half> scaleLocal = scaleQueue.AllocTensor<half>();
            LocalTensor<half> offsetLocal = offsetQueue.AllocTensor<half>();
            DataCopyPad(scaleLocal, scaleGm[kg * N + colBase], paramCopy, halfPad);
            scaleQueue.EnQue<half>(scaleLocal);
            DataCopyPad(offsetLocal, offsetGm[kg * N + colBase], paramCopy, halfPad);
            offsetQueue.EnQue<half>(offsetLocal);
            scaleLocal = scaleQueue.DeQue<half>();
            offsetLocal = offsetQueue.DeQue<half>();

            LocalTensor<half> bVec = vecoutQueue.AllocTensor<half>();
            if (mode_ == 0U) {
                LocalTensor<int32_t> wLocal = packedQueue.AllocTensor<int32_t>();
                DataCopyExtParams packedCopy{static_cast<uint16_t>(GroupK),
                                             static_cast<uint32_t>((TileN / 8U) * sizeof(int32_t)),
                                             static_cast<uint32_t>((N - TileN) / 8U * sizeof(int32_t)), 0, 0};
                DataCopyPadExtParams<int32_t> intPad{false, 0, 0, 0};
                DataCopyPad(wLocal, weightGm[kBase * PackedN + colBase / 8U], packedCopy, intPad);
                packedQueue.EnQue<int32_t>(wLocal);
                wLocal = packedQueue.DeQue<int32_t>();

                for (uint32_t kk = 0; kk < GroupK; ++kk) {
                    for (uint32_t nn = 0; nn < TileN; ++nn) {
                        const uint32_t packIdx = kk * (TileN / 8U) + (nn / 8U);
                        const int32_t q = DecodeInt4Value(wLocal.GetValue(packIdx), nn & 7U);
                        const float dequant =
                            (static_cast<float>(q) + static_cast<float>(offsetLocal.GetValue(nn))) *
                            static_cast<float>(scaleLocal.GetValue(nn));
                        bVec.SetValue(kk * TileN + nn, static_cast<half>(dequant));
                    }
                }
                packedQueue.FreeTensor(wLocal);
            } else {
                LocalTensor<int4b_t> w4Local = weight4Queue.AllocTensor<int4b_t>();
                DataCopyExtParams weight4Copy{static_cast<uint16_t>(GroupK), static_cast<uint32_t>(TileN / 2U),
                                              static_cast<uint32_t>((N - TileN) / 2U), 0, 0};
                DataCopyPadExtParams<int4b_t> int4Pad{false, 0, 0, 0};
                DataCopyPad(w4Local, weight4Gm[kBase * N + colBase], weight4Copy, int4Pad);
                weight4Queue.EnQue<int4b_t>(w4Local);
                w4Local = weight4Queue.DeQue<int4b_t>();

                Cast(bVec, w4Local, RoundMode::CAST_NONE, BTileElems);
                PipeBarrier<PIPE_V>();
                constexpr uint8_t rowStrideBlocks = static_cast<uint8_t>((TileN * sizeof(half)) >> 5);
                BinaryRepeatParams aqParams{1, 1, 1, rowStrideBlocks, rowStrideBlocks, 0};
                for (uint32_t subCol = 0; subCol < TileN; subCol += 128U) {
                    Add(bVec[subCol], bVec[subCol], offsetLocal[subCol], 128, static_cast<uint8_t>(GroupK),
                        aqParams);
                    PipeBarrier<PIPE_V>();
                    Mul(bVec[subCol], bVec[subCol], scaleLocal[subCol], 128, static_cast<uint8_t>(GroupK),
                        aqParams);
                    PipeBarrier<PIPE_V>();
                }
                weight4Queue.FreeTensor(w4Local);
            }
            vecoutQueue.EnQue<half>(bVec);
            bVec = vecoutQueue.DeQue<half>();

            mm.SetOrgShape(M, TileN, K, GroupK, N);
            mm.SetSingleShape(M, TileN, GroupK);
            mm.SetTensorA(xGm[kBase], false);
            mm.SetTensorB(bVec, false);
            mm.IterateAll(yGm[colBase], static_cast<uint8_t>(kg != 0U));

            scaleQueue.FreeTensor(scaleLocal);
            offsetQueue.FreeTensor(offsetLocal);
            vecoutQueue.FreeTensor(bVec);
        }

        mm.End();
    }

    Mm mm;

private:
    GlobalTensor<half> xGm;
    GlobalTensor<int32_t> weightGm;
    GlobalTensor<int4b_t> weight4Gm;
    GlobalTensor<half> scaleGm;
    GlobalTensor<half> offsetGm;
    GlobalTensor<half> yGm;
    AscendC::tiling::TCubeTiling tiling;
    uint32_t mode_{0};
};
}  // namespace AwqTscmVecoutDirectProbe

extern "C" __global__ __aicore__ void awq_w4a16_tscm_vecout_direct_probe(
    GM_ADDR x, GM_ADDR weight, GM_ADDR scale, GM_ADDR offset, GM_ADDR y, uint32_t mode, GM_ADDR workspace,
    GM_ADDR tilingGm)
{
    AscendC::SetSysWorkspace(workspace);

    AscendC::tiling::TCubeTiling tiling;
    AwqTscmVecoutDirectProbe::CopyTiling(&tiling, tilingGm);
    AscendC::TPipe pipe;

    AwqTscmVecoutDirectProbe::Kernel kernel;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), kernel.mm, &tiling);
    kernel.Init(x, weight, scale, offset, y, mode, tiling);
    kernel.Process(&pipe);
}
