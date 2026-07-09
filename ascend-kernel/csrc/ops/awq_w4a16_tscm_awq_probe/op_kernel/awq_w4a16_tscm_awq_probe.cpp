#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace AwqTscmNzProbe {
using namespace AscendC;
using namespace matmul;

using AType = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
using BType = MatmulType<TPosition::TSCM, CubeFormat::NZ, half, false, LayoutMode::NONE, false,
                         TPosition::GM>;
using BScmVecType = MatmulType<TPosition::TSCM, CubeFormat::NZ, half, false, LayoutMode::NONE, false,
                               TPosition::VECIN>;
using BVecType = MatmulType<TPosition::VECOUT, CubeFormat::ND, half>;
using CType = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, half, false>;
using CFloatType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using BiasFloatType = MatmulType<TPosition::GM, CubeFormat::ND, float>;

constexpr uint32_t M = 16;
constexpr uint32_t K = 128;
constexpr uint32_t N = 64;
constexpr uint32_t AElems = M * K;
constexpr uint32_t PackedN = N / 8;
constexpr uint32_t PackedElems = K * PackedN;
constexpr uint32_t BElems = K * N;
constexpr uint32_t CElems = M * N;
constexpr uint32_t ParamElems = N;
constexpr uint32_t AHalfBytes = AElems * sizeof(half);
constexpr uint32_t PackedBytes = PackedElems * sizeof(int32_t);
constexpr uint32_t BInt8Bytes = BElems * sizeof(int8_t);
constexpr uint32_t BHalfBytes = BElems * sizeof(half);
constexpr uint32_t CFloatBytes = CElems * sizeof(float);
constexpr uint32_t BufferPadBytes = 1024;
constexpr uint32_t ParamBytes = ParamElems * sizeof(half);

constexpr MatmulConfig RunCfg = {
    .doNorm = false,
    .doBasicBlock = false,
    .doMultiDataLoad = true,
    .basicM = M,
    .basicN = N,
    .basicK = K,
    .intrinsicsCheck = false,
    .isNBatch = false,
    .enVecND2NZ = true,
    .doSpecialBasicBlock = false,
    .doMTE2Preload = 0,
    .singleCoreM = M,
    .singleCoreN = N,
    .singleCoreK = K,
    .stepM = 0,
    .stepN = 0,
    .baseMN = 0,
    .singleCoreMN = 0,
    .enUnitFlag = false,
    .isPerTensor = false,
    .hasAntiQuantOffset = false,
    .doIBShareNorm = false,
    .doSpecialMDL = false,
    .enableInit = true,
    .batchMode = BatchMode::NONE,
    .enableEnd = true,
    .enableGetTensorC = true,
    .enableSetOrgShape = true,
    .enableSetBias = false,
    .enableSetTail = true,
    .enableQuantVector = true,
    .enableSetDefineData = true,
    .iterateMode = IterateMode::ITERATE_MODE_DEFAULT,
    .enableReuse = true,
    .enableUBReuse = true,
    .enableL1CacheUB = false,
    .intraBlockPartSum = false,
    .iterateOrder = IterateOrder::UNDEF,
    .scheduleType = ScheduleType::INNER_PRODUCT,
    .enableDoubleCache = false,
    .isBiasBatch = true,
    .enableStaticPadZeros = false,
    .isPartialOutput = false,
    .enableMixDualMaster = false,
    .isA2B2Shared = false,
    .isEnableChannelSplit = false,
    .enableKdimReorderLoad = false,
    .isCO1Shared = false,
    .sharedCO1BufferSize = SHARED_CO1_BUFFER_SIZE,
    .bmmOutMode = BatchOutMode::SINGLE_BATCH,
    .enableL1BankConflictOptimise = false,
    .enableRelu = false,
    .decompMode = DecompressionMode::DECOMP_NONE
};

constexpr MatmulApiStaticTiling MakeTiling(const MatmulApiStaticTiling &base)
{
    MatmulApiStaticTiling tiling = base;
    tiling.stepM = 1;
    tiling.stepN = 1;
    tiling.stepKa = 1;
    tiling.stepKb = 1;
    tiling.depthA1 = 2;
    tiling.depthB1 = 2;
    tiling.isBias = 0;
    return tiling;
}

constexpr MatmulApiStaticTiling RunTiling = MakeTiling(GetMatmulApiTiling<AType, BType, CType, BiasType>(RunCfg));
using Mmad = matmul::MatmulImpl<AType, BType, CType, BiasType, RunTiling>;
using MmadClient = matmul::Matmul<AType, BVecType, CFloatType, BiasFloatType>;
using TscmClient = matmul::Matmul<AType, BScmVecType, CFloatType, BiasFloatType>;
using TscmGmClient = matmul::Matmul<AType, BType, CFloatType, BiasFloatType>;

__aicore__ inline uint64_t SignExtendInt4Byte(uint32_t raw4)
{
    return static_cast<uint64_t>(raw4 < 8U ? raw4 : (raw4 | 0xF0U));
}

__aicore__ inline uint64_t DecodeInt4Pack8(int32_t packed)
{
    const uint32_t raw = static_cast<uint32_t>(packed);
    const uint64_t b0 = SignExtendInt4Byte(raw & 0xFU);
    const uint64_t b1 = SignExtendInt4Byte((raw >> 4) & 0xFU);
    const uint64_t b2 = SignExtendInt4Byte((raw >> 8) & 0xFU);
    const uint64_t b3 = SignExtendInt4Byte((raw >> 12) & 0xFU);
    const uint64_t b4 = SignExtendInt4Byte((raw >> 16) & 0xFU);
    const uint64_t b5 = SignExtendInt4Byte((raw >> 20) & 0xFU);
    const uint64_t b6 = SignExtendInt4Byte((raw >> 24) & 0xFU);
    const uint64_t b7 = SignExtendInt4Byte((raw >> 28) & 0xFU);
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24) | (b4 << 32) | (b5 << 40) | (b6 << 48) |
           (b7 << 56);
}

__aicore__ inline int32_t DecodeInt4Value(int32_t packed, uint32_t lane)
{
    const uint32_t nibble = (static_cast<uint32_t>(packed) >> (lane * 4U)) & 0xFU;
    return static_cast<int32_t>(nibble < 8U ? nibble : (nibble - 16U));
}

__aicore__ inline uint32_t NzIndex(uint32_t rowK, uint32_t colN)
{
    constexpr uint32_t c0 = 16;
    const uint32_t nBlock = colN / c0;
    const uint32_t kBlock = rowK / c0;
    const uint32_t kInner = rowK % c0;
    const uint32_t nInner = colN % c0;
    return (((nBlock * (K / c0) + kBlock) * c0 + kInner) * c0 + nInner);
}

__aicore__ inline void BuildNd2NzParams(AscendC::Nd2NzParams &nd2nz)
{
    nd2nz.ndNum = 1;
    nd2nz.nValue = K;
    nd2nz.dValue = N;
    nd2nz.srcNdMatrixStride = 0;
    nd2nz.srcDValue = N;
    nd2nz.dstNzC0Stride = K;
    nd2nz.dstNzNStride = 1;
    nd2nz.dstNzMatrixStride = 1;
}

__aicore__ inline void BuildANd2NzParams(AscendC::Nd2NzParams &nd2nz)
{
    nd2nz.ndNum = 1;
    nd2nz.nValue = M;
    nd2nz.dValue = K;
    nd2nz.srcNdMatrixStride = 0;
    nd2nz.srcDValue = K;
    nd2nz.dstNzC0Stride = M;
    nd2nz.dstNzNStride = 1;
    nd2nz.dstNzMatrixStride = 0;
}

__aicore__ inline void BuildBNd2NzParams(AscendC::Nd2NzParams &nd2nz)
{
    nd2nz.ndNum = 1;
    nd2nz.nValue = K;
    nd2nz.dValue = N;
    nd2nz.srcNdMatrixStride = 0;
    nd2nz.srcDValue = N;
    nd2nz.dstNzC0Stride = K;
    nd2nz.dstNzNStride = 1;
    nd2nz.dstNzMatrixStride = 0;
}
}  // namespace AwqTscmNzProbe

extern "C" __global__ __aicore__ void awq_w4a16_tscm_awq_probe(GM_ADDR x, GM_ADDR weight, GM_ADDR scale,
                                                                GM_ADDR offset, GM_ADDR y, uint32_t m,
                                                                uint32_t k, uint32_t n, uint32_t debugStage,
                                                                GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);

    AscendC::TQue<AscendC::TPosition::VECIN, 1> packedQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> int8Queue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> halfQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> scaleQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> offsetQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> vecOutQueue;
    AscendC::TSCM<AscendC::TPosition::GM, 1> bScmQueue;
    AscendC::TSCM<AscendC::TPosition::VECIN, 1, 0x04> bScmVecQueue;
    AscendC::TQue<AscendC::TPosition::A1, 1> a1Queue;
    AscendC::TQue<AscendC::TPosition::A2, 1> a2Queue;
    AscendC::TQueBind<AscendC::TPosition::VECOUT, AscendC::TPosition::B1, 1> b1Queue;
    AscendC::TQue<AscendC::TPosition::B1, 1> b1PlainQueue;
    AscendC::TQue<AscendC::TPosition::B2, 1> b2Queue;
    AscendC::TQue<AscendC::TPosition::CO1, 1> co1Queue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> matmulLocalWorkspace;
    AwqTscmNzProbe::MmadClient mmClient;
    AwqTscmNzProbe::TscmClient tscmClient;
    AwqTscmNzProbe::TscmGmClient tscmGmClient;
    AscendC::tiling::TCubeTiling vecoutTiling;

    if (debugStage == 4 || debugStage >= 9) {
        AscendC::GlobalTensor<uint32_t> tilingGm;
        tilingGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(tiling),
                                 (sizeof(AscendC::tiling::TCubeTiling) + sizeof(uint32_t) - 1) / sizeof(uint32_t));
        uint32_t *localTilingWords = reinterpret_cast<uint32_t *>(&vecoutTiling);
        for (uint32_t i = 0;
             i < (sizeof(AscendC::tiling::TCubeTiling) + sizeof(uint32_t) - 1) / sizeof(uint32_t); ++i) {
            localTilingWords[i] = tilingGm.GetValue(i);
        }
        AscendC::SetSysWorkspace(workspace);
        if (debugStage == 4) {
            REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mmClient, &vecoutTiling);
        } else if (debugStage == 13 || debugStage == 14) {
            REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), tscmGmClient, &vecoutTiling);
        } else {
            REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), tscmClient, &vecoutTiling);
        }
    }

    pipe.InitBuffer(packedQueue, 1, AwqTscmNzProbe::PackedBytes);
    pipe.InitBuffer(int8Queue, 1, AwqTscmNzProbe::BInt8Bytes);
    pipe.InitBuffer(scaleQueue, 1, AwqTscmNzProbe::ParamBytes);
    pipe.InitBuffer(offsetQueue, 1, AwqTscmNzProbe::ParamBytes);
    if (debugStage < 4 || debugStage >= 9) {
        pipe.InitBuffer(halfQueue, 1, AwqTscmNzProbe::BHalfBytes + AwqTscmNzProbe::BufferPadBytes);
    }
    if (debugStage < 4 || debugStage == 13 || debugStage == 14) {
        pipe.InitBuffer(bScmQueue, 1, AwqTscmNzProbe::BHalfBytes + AwqTscmNzProbe::BufferPadBytes);
    }
    if (debugStage >= 9 && debugStage != 13 && debugStage != 14) {
        pipe.InitBuffer(bScmVecQueue, 1, AwqTscmNzProbe::BHalfBytes + AwqTscmNzProbe::BufferPadBytes);
    }
    if (debugStage == 4 || debugStage == 5 || debugStage == 6) {
        pipe.InitBuffer(vecOutQueue, 1, AwqTscmNzProbe::BHalfBytes + AwqTscmNzProbe::BufferPadBytes);
    }
    if (debugStage == 4 || debugStage >= 9) {
        pipe.InitBuffer(matmulLocalWorkspace, AwqTscmNzProbe::BHalfBytes);
    }
    if (debugStage >= 5) {
        pipe.InitBuffer(a1Queue, 1, AwqTscmNzProbe::AHalfBytes);
        pipe.InitBuffer(a2Queue, 1, AwqTscmNzProbe::AHalfBytes);
        pipe.InitBuffer(b1Queue, 1, AwqTscmNzProbe::BHalfBytes + AwqTscmNzProbe::BufferPadBytes);
        if (debugStage == 8) {
            pipe.InitBuffer(b1PlainQueue, 1, AwqTscmNzProbe::BHalfBytes + AwqTscmNzProbe::BufferPadBytes);
        }
        pipe.InitBuffer(b2Queue, 1, AwqTscmNzProbe::BHalfBytes);
        pipe.InitBuffer(co1Queue, 1, AwqTscmNzProbe::CFloatBytes);
    }

    if ASCEND_IS_AIV {
        AscendC::GlobalTensor<int32_t> weightGm;
        AscendC::GlobalTensor<half> scaleGm;
        AscendC::GlobalTensor<half> offsetGm;
        weightGm.SetGlobalBuffer((__gm__ int32_t *)weight, AwqTscmNzProbe::PackedElems);
        scaleGm.SetGlobalBuffer((__gm__ half *)scale, AwqTscmNzProbe::ParamElems);
        offsetGm.SetGlobalBuffer((__gm__ half *)offset, AwqTscmNzProbe::ParamElems);

        AscendC::LocalTensor<int32_t> wLocal = packedQueue.AllocTensor<int32_t>();
        AscendC::LocalTensor<int8_t> bInt8 = int8Queue.AllocTensor<int8_t>();
        AscendC::LocalTensor<half> bHalf;
        if (debugStage < 4 || debugStage >= 9) {
            bHalf = halfQueue.AllocTensor<half>();
        }
        AscendC::LocalTensor<half> scaleLocal = scaleQueue.AllocTensor<half>();
        AscendC::LocalTensor<half> offsetLocal = offsetQueue.AllocTensor<half>();

        AscendC::DataCopyExtParams packedCopy{1, AwqTscmNzProbe::PackedBytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> intPad{false, 0, 0, 0};
        AscendC::DataCopyPad(wLocal, weightGm[0], packedCopy, intPad);
        packedQueue.EnQue<int32_t>(wLocal);

        AscendC::DataCopyExtParams paramCopy{1, AwqTscmNzProbe::ParamBytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<half> halfPad{false, 0, 0, 0};
        AscendC::DataCopyPad(scaleLocal, scaleGm[0], paramCopy, halfPad);
        scaleQueue.EnQue<half>(scaleLocal);
        AscendC::DataCopyPad(offsetLocal, offsetGm[0], paramCopy, halfPad);
        offsetQueue.EnQue<half>(offsetLocal);

        wLocal = packedQueue.DeQue<int32_t>();
        scaleLocal = scaleQueue.DeQue<half>();
        offsetLocal = offsetQueue.DeQue<half>();

        if (debugStage == 11 || debugStage == 12 || debugStage == 13 || debugStage == 14) {
            AscendC::LocalTensor<half> bNz;
            if (debugStage == 11 || debugStage == 13 || debugStage == 14) {
                bNz = halfQueue.AllocTensor<half>();
            } else {
                bNz = bScmVecQueue.AllocTensor<half>();
            }

            for (uint32_t kk = 0; kk < AwqTscmNzProbe::K; ++kk) {
                for (uint32_t nn = 0; nn < AwqTscmNzProbe::N; ++nn) {
                    const uint32_t packIdx = kk * AwqTscmNzProbe::PackedN + (nn / 8U);
                    const int32_t q = AwqTscmNzProbe::DecodeInt4Value(wLocal.GetValue(packIdx), nn & 7U);
                    const float dequant =
                        (static_cast<float>(q) + static_cast<float>(offsetLocal.GetValue(nn))) *
                        static_cast<float>(scaleLocal.GetValue(nn));
                    bNz.SetValue(AwqTscmNzProbe::NzIndex(kk, nn), static_cast<half>(dequant));
                }
            }

            if (debugStage == 11) {
                halfQueue.EnQue<half>(bNz);
                bNz = halfQueue.DeQue<half>();
                AscendC::LocalTensor<half> bScm = bScmVecQueue.AllocTensor<half>();
                AscendC::DataCopy(bScm, bNz, AwqTscmNzProbe::BElems);
                bScmVecQueue.EnQue<half>(bScm);
                halfQueue.FreeTensor(bNz);
            } else if (debugStage == 13 || debugStage == 14) {
                halfQueue.EnQue<half>(bNz);
                bNz = halfQueue.DeQue<half>();
                AscendC::GlobalTensor<half> bNzWorkspace;
                bNzWorkspace.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(AscendC::GetUserWorkspace(workspace)),
                                             AwqTscmNzProbe::BElems);
                AscendC::DataCopy(bNzWorkspace, bNz, AwqTscmNzProbe::BElems);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
                if (debugStage == 13) {
                    AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(8);
                } else {
                    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(8);
                }
                halfQueue.FreeTensor(bNz);
            } else {
                bScmVecQueue.EnQue<half>(bNz);
            }

            packedQueue.FreeTensor(wLocal);
            int8Queue.FreeTensor(bInt8);
            scaleQueue.FreeTensor(scaleLocal);
            offsetQueue.FreeTensor(offsetLocal);
            return;
        }

        AscendC::LocalTensor<uint64_t> bInt8Pack = bInt8.ReinterpretCast<uint64_t>();
        for (uint32_t kk = 0; kk < AwqTscmNzProbe::K; ++kk) {
            for (uint32_t packCol = 0; packCol < AwqTscmNzProbe::PackedN; ++packCol) {
                const uint32_t packIdx = kk * AwqTscmNzProbe::PackedN + packCol;
                bInt8Pack.SetValue(packIdx, AwqTscmNzProbe::DecodeInt4Pack8(wLocal.GetValue(packIdx)));
            }
        }

        if (debugStage == 8) {
            AscendC::LocalTensor<half> b1Plain = b1PlainQueue.AllocTensor<half>();
            AscendC::Fill(b1Plain, {static_cast<uint16_t>(AwqTscmNzProbe::BHalfBytes / 32), 1, 0, 1});
            b1PlainQueue.EnQue<half>(b1Plain);

            packedQueue.FreeTensor(wLocal);
            int8Queue.FreeTensor(bInt8);
            scaleQueue.FreeTensor(scaleLocal);
            offsetQueue.FreeTensor(offsetLocal);
            return;
        }

        if (debugStage == 7) {
            AscendC::LocalTensor<half> bBind = b1Queue.AllocTensor<half>();
            AscendC::Cast(bBind, bInt8, AscendC::RoundMode::CAST_NONE, AwqTscmNzProbe::BElems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::BinaryRepeatParams aqParams{1, 1, 1, 4, 4, 0};
            AscendC::Add(bBind, bBind, offsetLocal, AwqTscmNzProbe::N,
                         static_cast<uint8_t>(AwqTscmNzProbe::K), aqParams);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(bBind, bBind, scaleLocal, AwqTscmNzProbe::N,
                         static_cast<uint8_t>(AwqTscmNzProbe::K), aqParams);
            AscendC::PipeBarrier<PIPE_V>();
            b1Queue.EnQue<half>(bBind);

            packedQueue.FreeTensor(wLocal);
            int8Queue.FreeTensor(bInt8);
            scaleQueue.FreeTensor(scaleLocal);
            offsetQueue.FreeTensor(offsetLocal);
            return;
        }

        if (debugStage == 4 || debugStage >= 5) {
            AscendC::LocalTensor<half> bVec = vecOutQueue.AllocTensor<half>();
            AscendC::Cast(bVec, bInt8, AscendC::RoundMode::CAST_NONE, AwqTscmNzProbe::BElems);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::BinaryRepeatParams aqParams{1, 1, 1, 4, 4, 0};
            AscendC::Add(bVec, bVec, offsetLocal, AwqTscmNzProbe::N,
                         static_cast<uint8_t>(AwqTscmNzProbe::K), aqParams);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(bVec, bVec, scaleLocal, AwqTscmNzProbe::N,
                         static_cast<uint8_t>(AwqTscmNzProbe::K), aqParams);
            AscendC::PipeBarrier<PIPE_V>();
            if (debugStage == 4) {
                vecOutQueue.EnQue<half>(bVec);
            } else {
                AscendC::LocalTensor<half> b1Local = b1Queue.AllocTensor<half>();
                if (debugStage == 5) {
                    AscendC::DataCopy(b1Local, bVec, AwqTscmNzProbe::BElems);
                } else {
                    AscendC::Nd2NzParams bNd2Nz;
                    AwqTscmNzProbe::BuildBNd2NzParams(bNd2Nz);
                    AscendC::DataCopy(b1Local, bVec, bNd2Nz);
                }
                b1Queue.EnQue<half>(b1Local);
                vecOutQueue.FreeTensor(bVec);
            }

            packedQueue.FreeTensor(wLocal);
            int8Queue.FreeTensor(bInt8);
            scaleQueue.FreeTensor(scaleLocal);
            offsetQueue.FreeTensor(offsetLocal);
            return;
        }

        AscendC::Cast(bHalf, bInt8, AscendC::RoundMode::CAST_NONE, AwqTscmNzProbe::BElems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BinaryRepeatParams aqParams{1, 1, 1, 4, 4, 0};
        AscendC::Add(bHalf, bHalf, offsetLocal, AwqTscmNzProbe::N,
                     static_cast<uint8_t>(AwqTscmNzProbe::K), aqParams);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(bHalf, bHalf, scaleLocal, AwqTscmNzProbe::N,
                     static_cast<uint8_t>(AwqTscmNzProbe::K), aqParams);
        AscendC::PipeBarrier<PIPE_V>();
        halfQueue.EnQue<half>(bHalf);
        bHalf = halfQueue.DeQue<half>();

        if (debugStage >= 9) {
            AscendC::LocalTensor<half> bScm = bScmVecQueue.AllocTensor<half>();
            AscendC::Nd2NzParams nd2nz;
            AwqTscmNzProbe::BuildBNd2NzParams(nd2nz);
            if (debugStage == 10) {
                AscendC::DataCopyExtParams copyParams{
                    static_cast<uint16_t>(AwqTscmNzProbe::K),
                    static_cast<uint32_t>(AwqTscmNzProbe::N * sizeof(half)),
                    0,
                    0,
                    0};
                AscendC::DataCopyPad(bScm, bHalf, copyParams, nd2nz);
            } else {
                AscendC::DataCopy(bScm, bHalf, nd2nz);
            }
            bScmVecQueue.EnQue<half>(bScm);

            packedQueue.FreeTensor(wLocal);
            int8Queue.FreeTensor(bInt8);
            halfQueue.FreeTensor(bHalf);
            scaleQueue.FreeTensor(scaleLocal);
            offsetQueue.FreeTensor(offsetLocal);
            return;
        }

        if (debugStage >= 1) {
            AscendC::TEventID eventId = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);

            AscendC::LocalTensor<half> bScm = bScmQueue.AllocTensor<half>();
            if (debugStage >= 2) {
                AscendC::Nd2NzParams nd2nz;
                AwqTscmNzProbe::BuildNd2NzParams(nd2nz);
                AscendC::DataCopy(bScm, bHalf, nd2nz);
            }
            bScmQueue.EnQue<half>(bScm);
        }

        packedQueue.FreeTensor(wLocal);
        int8Queue.FreeTensor(bInt8);
        halfQueue.FreeTensor(bHalf);
        scaleQueue.FreeTensor(scaleLocal);
        offsetQueue.FreeTensor(offsetLocal);
    }

    if ASCEND_IS_AIC {
        if (debugStage == 0) {
            return;
        }

        AscendC::GlobalTensor<half> xGm;
        AscendC::GlobalTensor<half> yGm;
        xGm.SetGlobalBuffer((__gm__ half *)x, static_cast<uint64_t>(m) * k);
        yGm.SetGlobalBuffer((__gm__ half *)y, static_cast<uint64_t>(m) * n);

        if (debugStage == 4) {
            AscendC::LocalTensor<half> bVec = vecOutQueue.DeQue<half>();
            AscendC::GlobalTensor<float> yFloatGm;
            yFloatGm.SetGlobalBuffer((__gm__ float *)y, static_cast<uint64_t>(m) * n);
            mmClient.SetOrgShape(m, n, k);
            mmClient.SetSingleShape(m, n, k);
            mmClient.SetTensorA(xGm, false);
            mmClient.SetTensorB(bVec, false);
            AscendC::LocalTensor<uint8_t> buf = matmulLocalWorkspace.Get<uint8_t>();
            mmClient.SetLocalWorkspace(buf);
            mmClient.IterateAll(yFloatGm, 0);
            mmClient.End();
            vecOutQueue.FreeTensor(bVec);
            return;
        }

        if (debugStage == 13 || debugStage == 14) {
            if (debugStage == 13) {
                AscendC::CrossCoreWaitFlag<4, PIPE_MTE2>(8);
            } else {
                AscendC::CrossCoreWaitFlag<2, PIPE_MTE2>(8);
            }
            AscendC::GlobalTensor<half> bNzWorkspace;
            bNzWorkspace.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(AscendC::GetUserWorkspace(workspace)),
                                         AwqTscmNzProbe::BElems);
            AscendC::LocalTensor<half> bScm = bScmQueue.AllocTensor<half>();
            AscendC::DataCopy(bScm, bNzWorkspace, AwqTscmNzProbe::BElems);
            bScmQueue.EnQue<half>(bScm);
            bScm = bScmQueue.DeQue<half>();

            AscendC::GlobalTensor<float> yFloatGm;
            yFloatGm.SetGlobalBuffer((__gm__ float *)y, static_cast<uint64_t>(m) * n);
            tscmGmClient.SetOrgShape(m, n, k);
            tscmGmClient.SetSingleShape(m, n, k);
            tscmGmClient.SetTensorA(xGm, false);
            tscmGmClient.SetTensorB(bScm, false);
            AscendC::LocalTensor<uint8_t> buf = matmulLocalWorkspace.Get<uint8_t>();
            tscmGmClient.SetLocalWorkspace(buf);
            tscmGmClient.IterateAll(yFloatGm, 0);
            tscmGmClient.End();
            bScmQueue.FreeTensor(bScm);
            return;
        }

        if (debugStage >= 9) {
            AscendC::LocalTensor<half> bScm = bScmVecQueue.DeQue<half>();
            AscendC::GlobalTensor<float> yFloatGm;
            yFloatGm.SetGlobalBuffer((__gm__ float *)y, static_cast<uint64_t>(m) * n);
            tscmClient.SetOrgShape(m, n, k);
            tscmClient.SetSingleShape(m, n, k);
            tscmClient.SetTensorA(xGm, false);
            tscmClient.SetTensorB(bScm, false);
            AscendC::LocalTensor<uint8_t> buf = matmulLocalWorkspace.Get<uint8_t>();
            tscmClient.SetLocalWorkspace(buf);
            tscmClient.IterateAll(yFloatGm, 0);
            tscmClient.End();
            bScmVecQueue.FreeTensor(bScm);
            return;
        }

        if (debugStage == 8) {
            AscendC::LocalTensor<half> b1Plain = b1PlainQueue.DeQue<half>();
            b1PlainQueue.FreeTensor(b1Plain);
            return;
        }

        if (debugStage >= 5) {
            AscendC::LocalTensor<half> b1Local = b1Queue.DeQue<half>();
            if (debugStage == 5 || debugStage == 7) {
                b1Queue.FreeTensor(b1Local);
                return;
            }

            AscendC::LocalTensor<half> a1Local = a1Queue.AllocTensor<half>();
            AscendC::Nd2NzParams aNd2Nz;
            AwqTscmNzProbe::BuildANd2NzParams(aNd2Nz);
            AscendC::DataCopy(a1Local, xGm, aNd2Nz);
            a1Queue.EnQue<half>(a1Local);

            a1Local = a1Queue.DeQue<half>();
            AscendC::LocalTensor<half> a2Local = a2Queue.AllocTensor<half>();
            AscendC::LoadData2dParams loadA;
            loadA.repeatTimes = AwqTscmNzProbe::AHalfBytes / 512;
            loadA.srcStride = 1;
            loadA.dstGap = 0;
            loadA.ifTranspose = false;
            AscendC::LoadData(a2Local, a1Local, loadA);
            a2Queue.EnQue<half>(a2Local);
            a1Queue.FreeTensor(a1Local);

            AscendC::LocalTensor<half> b2Local = b2Queue.AllocTensor<half>();
            AscendC::LoadData2dTransposeParams loadB;
            loadB.startIndex = 0;
            loadB.repeatTimes = AwqTscmNzProbe::K / 16;
            loadB.srcStride = 1;
            loadB.dstGap = 1;
            loadB.dstFracGap = 0;
            for (uint32_t blockN = 0; blockN < (AwqTscmNzProbe::N / 16); ++blockN) {
                AscendC::LoadDataWithTranspose(b2Local[blockN * 16 * 16],
                                               b1Local[blockN * AwqTscmNzProbe::K * 16], loadB);
            }
            b2Queue.EnQue<half>(b2Local);
            b1Queue.FreeTensor(b1Local);

            a2Local = a2Queue.DeQue<half>();
            b2Local = b2Queue.DeQue<half>();
            AscendC::LocalTensor<float> c1Local = co1Queue.AllocTensor<float>();
            AscendC::MmadParams mmadParams;
            mmadParams.isBias = false;
            mmadParams.m = AwqTscmNzProbe::M;
            mmadParams.n = AwqTscmNzProbe::N;
            mmadParams.k = AwqTscmNzProbe::K;
            AscendC::Mmad(c1Local, a2Local, b2Local, mmadParams);
            co1Queue.EnQue<float>(c1Local);
            a2Queue.FreeTensor(a2Local);
            b2Queue.FreeTensor(b2Local);

            c1Local = co1Queue.DeQue<float>();
            AscendC::FixpipeParamsV220 fixpipeParams;
            fixpipeParams.nSize = AwqTscmNzProbe::N;
            fixpipeParams.mSize = AwqTscmNzProbe::M;
            fixpipeParams.srcStride = AwqTscmNzProbe::M;
            fixpipeParams.dstStride = AwqTscmNzProbe::N;
            fixpipeParams.ndNum = 1;
            fixpipeParams.srcNdStride = 0;
            fixpipeParams.dstNdStride = 0;
            AscendC::Fixpipe(yGm, c1Local, fixpipeParams);
            co1Queue.FreeTensor(c1Local);
            return;
        }

        AscendC::LocalTensor<half> bScm = bScmQueue.DeQue<half>();
        if (debugStage < 3) {
            bScmQueue.FreeTensor(bScm);
            return;
        }

        AwqTscmNzProbe::Mmad mm;
        mm.SetSubBlockIdx(0);
        mm.Init((__gm__ AscendC::tiling::TCubeTiling *)tiling, &pipe);
        mm.SetOrgShape(m, n, k);
        mm.SetSingleShape(m, n, k);
        mm.SetTensorA(xGm, false);
        mm.SetTensorB(bScm, false);
        mm.IterateAll(yGm, 0);
        mm.End();
        bScmQueue.FreeTensor(bScm);
    }
}
