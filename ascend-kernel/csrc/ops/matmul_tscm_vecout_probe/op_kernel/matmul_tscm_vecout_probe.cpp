#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace MatmulTscmVecoutProbe {
using namespace AscendC;
using namespace matmul;

constexpr uint32_t M = 32;
constexpr uint32_t N = 256;
constexpr uint32_t K = 32;
constexpr bool IsTransA = false;
constexpr bool IsTransB = false;

__aicore__ inline void CopyTiling(AscendC::tiling::TCubeTiling *tiling, GM_ADDR tilingGm)
{
    uint32_t *dst = reinterpret_cast<uint32_t *>(tiling);
    auto src = reinterpret_cast<__gm__ uint32_t *>(tilingGm);
    for (uint32_t i = 0; i < sizeof(AscendC::tiling::TCubeTiling) / sizeof(uint32_t); ++i) {
        dst[i] = src[i];
    }
}

template <typename AType, typename BType, typename CType, typename BiasType>
class Kernel {
public:
    using Mm = Matmul<MatmulType<TPosition::TSCM, CubeFormat::NZ, AType, IsTransA, LayoutMode::NONE, false,
                                 TPosition::VECOUT>,
                      MatmulType<TPosition::GM, CubeFormat::ND, BType, IsTransB>,
                      MatmulType<TPosition::GM, CubeFormat::ND, CType>,
                      MatmulType<TPosition::GM, CubeFormat::ND, BiasType>, CFG_NORM>;

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c,
                                const AscendC::tiling::TCubeTiling &tilingIn)
    {
        tiling = tilingIn;
        aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ AType *>(a), tiling.M * tiling.Ka);
        bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ BType *>(b), tiling.Kb * tiling.N);
        cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ CType *>(c), tiling.M * tiling.N);
        biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ BiasType *>(bias), tiling.N);

        int32_t offsetA = 0;
        int32_t offsetB = 0;
        int32_t offsetC = 0;
        int32_t offsetBias = 0;
        CalcOffset(AscendC::GetBlockIdx(), offsetA, offsetB, offsetC, offsetBias);
        aGlobal = aGlobal[offsetA];
        bGlobal = bGlobal[offsetB];
        cGlobal = cGlobal[offsetC];
        biasGlobal = biasGlobal[offsetBias];
    }

    __aicore__ inline void Process(AscendC::TPipe *pipe)
    {
        if (matmul::GetBlockIdx() >= tiling.usedCoreNum) {
            return;
        }

        int32_t tailM = tiling.M - mCoreIndex * tiling.singleCoreM;
        tailM = tailM < tiling.singleCoreM ? tailM : tiling.singleCoreM;
        int32_t tailN = tiling.N - nCoreIndex * tiling.singleCoreN;
        tailN = tailN < tiling.singleCoreN ? tailN : tiling.singleCoreN;
        if (tailM < tiling.singleCoreM || tailN < tiling.singleCoreN) {
            mm.SetTail(tailM, tailN);
        }

        TQue<TPosition::VECIN, 1> vecin;
        pipe->InitBuffer(vecin, 1, tiling.M * tiling.Ka * sizeof(AType));
        auto vecinTensor = vecin.AllocTensor<AType>();
        DataCopy(vecinTensor, aGlobal, tiling.M * tiling.Ka);
        vecin.EnQue(vecinTensor);
        LocalTensor<AType> vecinLocal = vecin.DeQue<AType>();

        TQue<TPosition::VECOUT, 1> vecout;
        pipe->InitBuffer(vecout, 1, tiling.M * tiling.Ka * sizeof(AType));
        auto vecoutTensor = vecout.AllocTensor<AType>();
        DataCopy(vecoutTensor, vecinLocal, tiling.M * tiling.Ka);
        vecout.EnQue(vecoutTensor);
        LocalTensor<AType> vecoutLocal = vecout.DeQue<AType>();
        vecin.FreeTensor(vecinLocal);

        TSCM<TPosition::VECOUT, 1> scm;
        pipe->InitBuffer(scm, 1, tiling.M * tiling.Ka * sizeof(AType));
        auto scmTensor = scm.AllocTensor<AType>();
        DataCopy(scmTensor, vecoutLocal, tiling.M * tiling.Ka);
        scm.EnQue(scmTensor);
        LocalTensor<AType> scmLocal = scm.DeQue<AType>();

        mm.SetTensorA(scmLocal, IsTransA);
        mm.SetTensorB(bGlobal, IsTransB);
        if (tiling.isBias) {
            mm.SetBias(biasGlobal);
        }
        mm.IterateAll(cGlobal);
        mm.End();

        vecout.FreeTensor(vecoutLocal);
        scm.FreeTensor(scmLocal);
    }

    Mm mm;

private:
    __aicore__ inline void CalcOffset(int32_t blockIdx, int32_t &offsetA, int32_t &offsetB, int32_t &offsetC,
                                      int32_t &offsetBias)
    {
        auto mSingleBlocks = (tiling.M + tiling.singleCoreM - 1) / tiling.singleCoreM;
        mCoreIndex = blockIdx % mSingleBlocks;
        nCoreIndex = blockIdx / mSingleBlocks;

        offsetA = mCoreIndex * tiling.Ka * tiling.singleCoreM;
        if (IsTransA) {
            offsetA = mCoreIndex * tiling.singleCoreM;
        }
        offsetB = nCoreIndex * tiling.singleCoreN;
        if (IsTransB) {
            offsetB = nCoreIndex * tiling.Kb * tiling.singleCoreN;
        }
        offsetC = mCoreIndex * tiling.N * tiling.singleCoreM + nCoreIndex * tiling.singleCoreN;
        offsetBias = nCoreIndex * tiling.singleCoreN;
    }

    GlobalTensor<AType> aGlobal;
    GlobalTensor<BType> bGlobal;
    GlobalTensor<CType> cGlobal;
    GlobalTensor<BiasType> biasGlobal;
    AscendC::tiling::TCubeTiling tiling;
    int32_t mCoreIndex = 0;
    int32_t nCoreIndex = 0;
};
}  // namespace MatmulTscmVecoutProbe

extern "C" __global__ __aicore__ void matmul_tscm_vecout_probe(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c,
                                                                 GM_ADDR workspace, GM_ADDR tilingGm)
{
    AscendC::SetSysWorkspace(workspace);

    AscendC::tiling::TCubeTiling tiling;
    MatmulTscmVecoutProbe::CopyTiling(&tiling, tilingGm);
    AscendC::TPipe pipe;

    MatmulTscmVecoutProbe::Kernel<half, half, float, float> kernel;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), kernel.mm, &tiling);
    kernel.Init(a, b, bias, c, tiling);
    kernel.Process(&pipe);
}
