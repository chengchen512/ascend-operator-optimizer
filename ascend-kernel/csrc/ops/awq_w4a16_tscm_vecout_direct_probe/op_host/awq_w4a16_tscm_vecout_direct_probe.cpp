#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "aclrtlaunch_awq_w4a16_tscm_vecout_direct_probe.h"

namespace ascend_kernel {
namespace {
constexpr int64_t kM = 16;
constexpr int64_t kK = 4096;
constexpr int64_t kGroupK = 128;
constexpr int64_t kN = 22016;
constexpr int64_t kTileN = 256;
constexpr int64_t kWorkspacePadBytes = 512;

struct TilingCache {
    at::Tensor tensor;
    uint32_t used_core_num;
};

int64_t GetWorkspaceBytes()
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, "awq_w4a16_tscm_vecout_direct_probe: failed to initialize AscendC platform");
    return static_cast<int64_t>(platform->GetLibApiWorkSpaceSize()) + kWorkspacePadBytes;
}

TilingCache MakeTilingCache()
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, "awq_w4a16_tscm_vecout_direct_probe: failed to initialize AscendC platform");

    AscendC::tiling::TCubeTiling cube_tiling;
    matmul_tiling::MultiCoreMatmulTiling tiling(*platform);
    tiling.SetDim(1);
    tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_FLOAT16);
    tiling.SetBType(matmul_tiling::TPosition::VECOUT, matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_FLOAT16);
    tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_FLOAT16);
    tiling.EnableBias(false);
    tiling.SetShape(kM, kTileN, kGroupK);
    tiling.SetOrgShape(kM, kTileN, kGroupK);
    tiling.SetFixSplit(kM, kTileN, kGroupK);
    tiling.SetBufferSpace(-1, -1, -1);
    TORCH_CHECK(tiling.GetTiling(cube_tiling) != -1,
                "awq_w4a16_tscm_vecout_direct_probe: GetTiling failed");
    TORCH_CHECK(cube_tiling.usedCoreNum > 0,
                "awq_w4a16_tscm_vecout_direct_probe: expected positive usedCoreNum");

    at::Tensor tiling_cpu = at::empty({static_cast<int64_t>(sizeof(cube_tiling))},
                                      at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    std::memcpy(tiling_cpu.data_ptr<uint8_t>(), &cube_tiling, sizeof(cube_tiling));
    return {TorchNpuHelper::CopyTensorHostToDevice(tiling_cpu), static_cast<uint32_t>(kN / kTileN)};
}
}  // namespace

at::Tensor awq_w4a16_tscm_vecout_direct_probe(const at::Tensor &x,
                                               const at::Tensor &weight,
                                               const at::Tensor &antiquant_scale,
                                               const at::Tensor &antiquant_offset)
{
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 2,
                "awq_w4a16_tscm_vecout_direct_probe: x/weight must be 2D");
    TORCH_CHECK(x.scalar_type() == at::kHalf,
                "awq_w4a16_tscm_vecout_direct_probe: x must be float16");
    TORCH_CHECK(weight.scalar_type() == at::kInt,
                "awq_w4a16_tscm_vecout_direct_probe: weight must be int32 int4pack");
    TORCH_CHECK(antiquant_scale.scalar_type() == at::kHalf && antiquant_offset.scalar_type() == at::kHalf,
                "awq_w4a16_tscm_vecout_direct_probe: scale/offset must be float16");
    TORCH_CHECK(x.is_contiguous() && weight.is_contiguous() && antiquant_scale.is_contiguous() &&
                    antiquant_offset.is_contiguous(),
                "awq_w4a16_tscm_vecout_direct_probe: all inputs must be contiguous");
    TORCH_CHECK(x.size(0) == kM && x.size(1) == kK && weight.size(0) == kK && weight.size(1) == kN / 8,
                "awq_w4a16_tscm_vecout_direct_probe: only supports x=[16,4096], weight=[4096,2752]");
    TORCH_CHECK(antiquant_scale.numel() == (kK / kGroupK) * kN &&
                    antiquant_offset.numel() == (kK / kGroupK) * kN,
                "awq_w4a16_tscm_vecout_direct_probe: scale/offset must have [32,22016] elements");

    at::Tensor y = at::empty({kM, kN}, x.options());
    at::Tensor workspace = at::empty({GetWorkspaceBytes()}, x.options().dtype(at::kByte));
    static TilingCache tiling_cache = MakeTilingCache();
    at::Tensor tiling_dev = tiling_cache.tensor;
    uint32_t block_dim = tiling_cache.used_core_num;
    const char *route = std::getenv("AWQ_VECOUT_TSCM_ROUTE");
    uint32_t mode = (route != nullptr && std::strcmp(route, "scalar") == 0) ? 0U : 1U;
    EXEC_KERNEL_CMD(awq_w4a16_tscm_vecout_direct_probe, block_dim, x, weight, antiquant_scale, antiquant_offset, y,
                    mode, workspace, tiling_dev);
    return y;
}

}  // namespace ascend_kernel
