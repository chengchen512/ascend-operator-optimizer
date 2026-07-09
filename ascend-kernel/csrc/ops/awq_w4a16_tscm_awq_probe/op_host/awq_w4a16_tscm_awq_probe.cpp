// Isolated VECIN -> TSCM(NZ) -> Matmul AWQ W4A16 probe.
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "aclrtlaunch_awq_w4a16_tscm_awq_probe.h"

namespace ascend_kernel {
namespace {

constexpr int64_t kM = 16;
constexpr int64_t kK = 128;
constexpr int64_t kN = 64;
constexpr int64_t kBHalfBytes = kK * kN * sizeof(at::Half);
constexpr int64_t kWorkspacePadBytes = 512;

int64_t GetEnvInt64(const char *name, int64_t default_value)
{
    const char *raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        return default_value;
    }
    char *end = nullptr;
    long long value = std::strtoll(raw, &end, 10);
    if (end == raw || value < 0) {
        return default_value;
    }
    return value;
}

int64_t GetProbeWorkspaceBytes()
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, "awq_w4a16_tscm_awq_probe: failed to initialize AscendC platform");
    return static_cast<int64_t>(platform->GetLibApiWorkSpaceSize()) + kBHalfBytes + kWorkspacePadBytes;
}

at::Tensor MakeProbeTilingTensor(int64_t tiling_kind)
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, "awq_w4a16_tscm_awq_probe: failed to initialize AscendC platform");

    AscendC::tiling::TCubeTiling cube_tiling;
    matmul_tiling::MatmulApiTiling tiling(*platform);
    tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_FLOAT16);
    if (tiling_kind == 1) {
        tiling.SetBType(matmul_tiling::TPosition::VECOUT, matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_FLOAT16);
    } else {
        tiling.SetBType(matmul_tiling::TPosition::TSCM, matmul_tiling::CubeFormat::NZ,
                        matmul_tiling::DataType::DT_FLOAT16);
    }
    tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                    tiling_kind == 0 ? matmul_tiling::DataType::DT_FLOAT16 : matmul_tiling::DataType::DT_FLOAT);
    tiling.SetBias(false);
    tiling.SetShape(kM, kN, kK);
    tiling.SetOrgShape(kM, kN, kK);
    tiling.SetFixSplit(kM, kN, kK);
    tiling.SetBufferSpace(-1, -1, -1);
    TORCH_CHECK(tiling.GetTiling(cube_tiling) != -1, "awq_w4a16_tscm_awq_probe: GetTiling failed");

    at::Tensor tiling_cpu = at::empty({static_cast<int64_t>(sizeof(cube_tiling))},
                                      at::TensorOptions().dtype(at::kByte).device(at::kCPU));
    std::memcpy(tiling_cpu.data_ptr<uint8_t>(), &cube_tiling, sizeof(cube_tiling));
    return TorchNpuHelper::CopyTensorHostToDevice(tiling_cpu);
}

}  // namespace

at::Tensor awq_w4a16_tscm_awq_probe(const at::Tensor &x, const at::Tensor &weight,
                                     const at::Tensor &antiquant_scale,
                                     const at::Tensor &antiquant_offset)
{
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 2, "awq_w4a16_tscm_awq_probe: x/weight must be 2D");
    TORCH_CHECK(x.scalar_type() == at::kHalf, "awq_w4a16_tscm_awq_probe: x must be float16");
    TORCH_CHECK(weight.scalar_type() == at::kInt, "awq_w4a16_tscm_awq_probe: weight must be int32 int4pack");
    TORCH_CHECK(antiquant_scale.scalar_type() == at::kHalf && antiquant_offset.scalar_type() == at::kHalf,
                "awq_w4a16_tscm_awq_probe: scale/offset must be float16");
    TORCH_CHECK(x.is_contiguous() && weight.is_contiguous() && antiquant_scale.is_contiguous() &&
                    antiquant_offset.is_contiguous(),
                "awq_w4a16_tscm_awq_probe: all inputs must be contiguous");
    TORCH_CHECK(x.size(0) == kM && x.size(1) == kK && weight.size(0) == kK && weight.size(1) == kN / 8,
                "awq_w4a16_tscm_awq_probe: only supports x=[16,128], weight=[128,8]");
    TORCH_CHECK(antiquant_scale.numel() == kN && antiquant_offset.numel() == kN,
                "awq_w4a16_tscm_awq_probe: scale/offset must have 64 elements");

    uint32_t m = static_cast<uint32_t>(kM);
    uint32_t k = static_cast<uint32_t>(kK);
    uint32_t n = static_cast<uint32_t>(kN);
    uint32_t debug_stage = static_cast<uint32_t>(std::min<int64_t>(14, GetEnvInt64("AWQ_TSCM_STAGE", 3)));
    at::Tensor y = at::empty({kM, kN}, x.options().dtype((debug_stage == 4 || debug_stage >= 9) ? at::kFloat : at::kHalf));
    at::Tensor workspace = at::empty({GetProbeWorkspaceBytes()}, x.options().dtype(at::kByte));
    static at::Tensor cached_tscm_tiling_dev = MakeProbeTilingTensor(0);
    static at::Tensor cached_vecout_tiling_dev = MakeProbeTilingTensor(1);
    static at::Tensor cached_tscm_float_tiling_dev = MakeProbeTilingTensor(2);
    at::Tensor &tiling_dev = (debug_stage == 4)    ? cached_vecout_tiling_dev
                             : (debug_stage >= 9) ? cached_tscm_float_tiling_dev
                                                  : cached_tscm_tiling_dev;
    uint32_t block_dim = 1;
    EXEC_KERNEL_CMD(awq_w4a16_tscm_awq_probe, block_dim, x, weight, antiquant_scale, antiquant_offset, y,
                    m, k, n, debug_stage, workspace, tiling_dev);
    return y;
}

}  // namespace ascend_kernel
