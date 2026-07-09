#include <cstdint>
#include <cstring>

#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "aclrtlaunch_matmul_tscm_vecout_probe.h"

namespace ascend_kernel {
namespace {
constexpr int64_t kM = 32;
constexpr int64_t kN = 256;
constexpr int64_t kK = 32;
constexpr int64_t kWorkspacePadBytes = 512;

struct TilingCache {
    at::Tensor tensor;
    uint32_t used_core_num;
};

int64_t GetWorkspaceBytes()
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, "matmul_tscm_vecout_probe: failed to initialize AscendC platform");
    return static_cast<int64_t>(platform->GetLibApiWorkSpaceSize()) + kWorkspacePadBytes;
}

TilingCache MakeTilingCache()
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, "matmul_tscm_vecout_probe: failed to initialize AscendC platform");

    AscendC::tiling::TCubeTiling cube_tiling;
    matmul_tiling::MultiCoreMatmulTiling tiling(*platform);
    tiling.SetDim(1);
    tiling.SetAType(matmul_tiling::TPosition::TSCM, matmul_tiling::CubeFormat::NZ,
                    matmul_tiling::DataType::DT_FLOAT16, false);
    tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_FLOAT16, false);
    tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_FLOAT);
    tiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                       matmul_tiling::DataType::DT_FLOAT);
    tiling.SetOrgShape(kM, kN, kK);
    tiling.SetShape(kM, kN, kK);
    tiling.EnableBias(true);
    tiling.SetBufferSpace(-1, -1, -1);
    TORCH_CHECK(tiling.GetTiling(cube_tiling) != -1, "matmul_tscm_vecout_probe: GetTiling failed");
    TORCH_CHECK(cube_tiling.usedCoreNum > 0, "matmul_tscm_vecout_probe: expected positive usedCoreNum");

    at::Tensor tiling_cpu = at::empty({static_cast<int64_t>(sizeof(cube_tiling))},
                                      at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    std::memcpy(tiling_cpu.data_ptr<uint8_t>(), &cube_tiling, sizeof(cube_tiling));
    return {TorchNpuHelper::CopyTensorHostToDevice(tiling_cpu), static_cast<uint32_t>(cube_tiling.usedCoreNum)};
}
}  // namespace

at::Tensor matmul_tscm_vecout_probe(const at::Tensor &a, const at::Tensor &b, const at::Tensor &bias)
{
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && bias.dim() == 1,
                "matmul_tscm_vecout_probe: expected a=[32,32], b=[32,256], bias=[256]");
    TORCH_CHECK(a.scalar_type() == at::kHalf && b.scalar_type() == at::kHalf,
                "matmul_tscm_vecout_probe: a/b must be float16");
    TORCH_CHECK(bias.scalar_type() == at::kFloat, "matmul_tscm_vecout_probe: bias must be float32");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && bias.is_contiguous(),
                "matmul_tscm_vecout_probe: all inputs must be contiguous");
    TORCH_CHECK(a.size(0) == kM && a.size(1) == kK && b.size(0) == kK && b.size(1) == kN &&
                    bias.numel() == kN,
                "matmul_tscm_vecout_probe: only supports M=32, N=256, K=32");

    at::Tensor c = at::empty({kM, kN}, a.options().dtype(at::kFloat));
    at::Tensor workspace = at::empty({GetWorkspaceBytes()}, a.options().dtype(at::kByte));
    static TilingCache tiling_cache = MakeTilingCache();
    uint32_t block_dim = tiling_cache.used_core_num;
    at::Tensor tiling_dev = tiling_cache.tensor;
    EXEC_KERNEL_CMD(matmul_tscm_vecout_probe, block_dim, a, b, bias, c, workspace, tiling_dev);
    return c;
}

}  // namespace ascend_kernel
