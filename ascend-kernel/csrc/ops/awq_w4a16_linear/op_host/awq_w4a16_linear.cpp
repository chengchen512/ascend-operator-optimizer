// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>

#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

#include "aclrtlaunch_awq_w4a16_linear.h"
#include "aclrtlaunch_awq_w4a16_linear_vecout.h"
#include "aclrtlaunch_awq_w4a16_linear_vecout_n144.h"
#include "aclrtlaunch_awq_w4a16_linear_vecout_n192.h"

namespace ascend_kernel {
namespace {
constexpr int64_t kGroupSize = 128;
constexpr int64_t kInt4PerPack = 8;
constexpr int64_t kWorkspaceBytes = 512;
constexpr int64_t kVecoutTileN = 128;
constexpr int64_t kVecoutN144TileN = 144;
constexpr int64_t kVecoutN192TileN = 192;
constexpr int64_t kVecoutTileK = 256;

enum class AwqHostRoute {
    kScalar,
    kStream,
    kSplit,
    kVecout128,
    kVecoutN144,
    kVecoutN192,
};

struct AwqHostPlan {
    AwqHostRoute route;
    const char *name;
    bool auto_selected;
};

int64_t GetEnvInt64(const char *name, int64_t default_value)
{
    const char *raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        return default_value;
    }
    char *end = nullptr;
    long long value = std::strtoll(raw, &end, 10);
    if (end == raw || value <= 0) {
        return default_value;
    }
    return static_cast<int64_t>(value);
}

int64_t GetStreamTileN(int64_t n)
{
    int64_t tile_n = GetEnvInt64("AWQ_STREAM_N_TILE", 12288);
    tile_n = std::min<int64_t>(tile_n, n);
    if (tile_n > 512) {
        tile_n = (tile_n / 512) * 512;
    }
    return std::max<int64_t>(1, tile_n);
}

bool EnvEquals(const char *raw, const char *expected)
{
    return raw != nullptr && std::strcmp(raw, expected) == 0;
}

bool IsDebugTilingEnabled()
{
    const char *debug_tiling = std::getenv("AWQ_DEBUG_TILING");
    return debug_tiling != nullptr && debug_tiling[0] != '\0';
}

void CheckAwqInputs(const at::Tensor &x,
                    const at::Tensor &weight,
                    const at::Tensor &antiquant_scale,
                    const at::Tensor &antiquant_offset)
{
    TORCH_CHECK(x.dim() == 2, "awq_w4a16_linear: x must be 2D [M, K]");
    TORCH_CHECK(weight.dim() == 2, "awq_w4a16_linear: weight must be 2D [K, N/8]");
    TORCH_CHECK(antiquant_scale.dim() == 2, "awq_w4a16_linear: antiquant_scale must be 2D [K/128, N]");
    TORCH_CHECK(antiquant_offset.dim() == 2, "awq_w4a16_linear: antiquant_offset must be 2D [K/128, N]");
    TORCH_CHECK(x.scalar_type() == at::kHalf, "awq_w4a16_linear: x must be float16");
    TORCH_CHECK(weight.scalar_type() == at::kInt, "awq_w4a16_linear: weight must be int32 int4pack");
    TORCH_CHECK(antiquant_scale.scalar_type() == at::kHalf, "awq_w4a16_linear: antiquant_scale must be float16");
    TORCH_CHECK(antiquant_offset.scalar_type() == at::kHalf, "awq_w4a16_linear: antiquant_offset must be float16");
    TORCH_CHECK(x.is_contiguous(), "awq_w4a16_linear: x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "awq_w4a16_linear: weight must be contiguous");
    TORCH_CHECK(antiquant_scale.is_contiguous(), "awq_w4a16_linear: antiquant_scale must be contiguous");
    TORCH_CHECK(antiquant_offset.is_contiguous(), "awq_w4a16_linear: antiquant_offset must be contiguous");

    const int64_t m = x.size(0);
    const int64_t k = x.size(1);
    TORCH_CHECK(m > 0 && k > 0, "awq_w4a16_linear: empty x is not supported");
    TORCH_CHECK(k % kGroupSize == 0, "awq_w4a16_linear: K must be divisible by 128");
    TORCH_CHECK(weight.size(0) == k, "awq_w4a16_linear: weight.shape[0] must equal K");
    TORCH_CHECK(weight.size(1) > 0, "awq_w4a16_linear: packed N must be positive");
    const int64_t n = weight.size(1) * kInt4PerPack;
    TORCH_CHECK(antiquant_scale.size(0) == k / kGroupSize && antiquant_scale.size(1) == n,
                "awq_w4a16_linear: antiquant_scale shape must be [K/128, N]");
    TORCH_CHECK(antiquant_offset.size(0) == k / kGroupSize && antiquant_offset.size(1) == n,
                "awq_w4a16_linear: antiquant_offset shape must be [K/128, N]");
    TORCH_CHECK(k <= std::numeric_limits<uint32_t>::max() && n <= std::numeric_limits<uint32_t>::max() &&
                    m <= std::numeric_limits<uint32_t>::max(),
                "awq_w4a16_linear: shape exceeds uint32 kernel parameters");
    TORCH_CHECK(m <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) / weight.size(1),
                "awq_w4a16_linear: scalar fallback task count exceeds uint32");
}

bool TryBuildVecoutMatmulTiling(uint32_t m, uint32_t tile_n, uint32_t tile_k)
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, "awq_w4a16_linear: failed to initialize AscendC platform");

    AscendC::tiling::TCubeTiling cube_tiling{};
    matmul_tiling::MultiCoreMatmulTiling tiling(*platform);
    tiling.SetDim(1);
    tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_FLOAT16);
    tiling.SetBType(matmul_tiling::TPosition::VECOUT, matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_FLOAT16);
    tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_FLOAT16);
    tiling.EnableBias(false);
    tiling.SetShape(static_cast<int32_t>(m), static_cast<int32_t>(tile_n), static_cast<int32_t>(tile_k));
    tiling.SetOrgShape(static_cast<int32_t>(m), static_cast<int32_t>(tile_n), static_cast<int32_t>(tile_k));
    tiling.SetFixSplit(static_cast<int32_t>(m), static_cast<int32_t>(tile_n), -1);
    tiling.SetBufferSpace(-1, -1, -1);
    const int32_t tiling_ret = tiling.GetTiling(cube_tiling);
    if (IsDebugTilingEnabled() && tiling_ret != -1) {
        matmul_tiling::SysTilingTempBufSize buf_size;
        int32_t tmp_ret = MatmulGetTmpBufSizeV2(cube_tiling, buf_size);
        std::cerr << "awq_w4a16_linear host tiling: ret=" << tiling_ret
                  << " tileN=" << tile_n << " tileK=" << tile_k
                  << " usedCoreNum=" << cube_tiling.usedCoreNum
                  << " M=" << cube_tiling.M << " N=" << cube_tiling.N << " Ka=" << cube_tiling.Ka
                  << " Kb=" << cube_tiling.Kb << " singleCoreM=" << cube_tiling.singleCoreM
                  << " singleCoreN=" << cube_tiling.singleCoreN << " singleCoreK=" << cube_tiling.singleCoreK
                  << " baseM=" << cube_tiling.baseM << " baseN=" << cube_tiling.baseN
                  << " baseK=" << cube_tiling.baseK << " stepM=" << cube_tiling.stepM
                  << " stepN=" << cube_tiling.stepN << " transLength=" << cube_tiling.transLength
                  << " shareL1Size=" << cube_tiling.shareL1Size
                  << " shareL0CSize=" << cube_tiling.shareL0CSize
                  << " shareUbSize=" << cube_tiling.shareUbSize << " tmpRet=" << tmp_ret
                  << " tmpUb=" << buf_size.ubSize << " tmpL1=" << buf_size.l1Size
                  << " tmpL0C=" << buf_size.l0cSize << std::endl;
    } else if (IsDebugTilingEnabled()) {
        std::cerr << "awq_w4a16_linear host tiling: ret=" << tiling_ret
                  << " tileN=" << tile_n << " tileK=" << tile_k << std::endl;
    }
    return tiling_ret != -1;
}

bool CachedTryBuildVecoutMatmulTiling(uint32_t m, uint32_t tile_n, uint32_t tile_k)
{
    static std::mutex mutex;
    static bool has_cache = false;
    static uint32_t cached_m = 0;
    static uint32_t cached_tile_n = 0;
    static uint32_t cached_tile_k = 0;
    static bool cached_result = false;

    std::lock_guard<std::mutex> guard(mutex);
    if (has_cache && cached_m == m && cached_tile_n == tile_n && cached_tile_k == tile_k) {
        return cached_result;
    }
    cached_result = TryBuildVecoutMatmulTiling(m, tile_n, tile_k);
    cached_m = m;
    cached_tile_n = tile_n;
    cached_tile_k = tile_k;
    has_cache = true;
    return cached_result;
}

bool IsVecout128Candidate(uint32_t m, uint32_t k, uint32_t n)
{
    return m <= 16U && (k % static_cast<uint32_t>(kVecoutTileK)) == 0U &&
           (n % static_cast<uint32_t>(kVecoutTileN)) == 0U;
}

void CheckVecout128Shape(uint32_t m, uint32_t k, uint32_t n)
{
    TORCH_CHECK(m <= 16U, "awq_w4a16_linear: vecout128 route supports M <= 16");
    TORCH_CHECK((k % static_cast<uint32_t>(kVecoutTileK)) == 0U,
                "awq_w4a16_linear: vecout128 route requires K divisible by 256");
    TORCH_CHECK((n % static_cast<uint32_t>(kVecoutTileN)) == 0U,
                "awq_w4a16_linear: vecout128 route requires N divisible by 128");
    TORCH_CHECK(CachedTryBuildVecoutMatmulTiling(m, static_cast<uint32_t>(kVecoutTileN),
                                                 static_cast<uint32_t>(kVecoutTileK)),
                "awq_w4a16_linear: host GetTiling rejected vecout128 route");
}

void CheckWideVecoutShape(const char *route_name, uint32_t m, uint32_t k, uint32_t n, uint32_t tile_n)
{
    TORCH_CHECK(m <= 16U, "awq_w4a16_linear: ", route_name, " route supports M <= 16");
    TORCH_CHECK((k % static_cast<uint32_t>(kVecoutTileK)) == 0U, "awq_w4a16_linear: ", route_name,
                " route requires K divisible by 256");
    const uint32_t main_tiles = n / tile_n;
    const uint32_t tail_n = n - main_tiles * tile_n;
    TORCH_CHECK(tail_n == 0U || tail_n == kVecoutTileN,
                "awq_w4a16_linear: ", route_name, " route requires tail N to be 0 or 128");
}

AwqHostPlan SelectAwqHostPlan(uint32_t m, uint32_t k, uint32_t n)
{
    const char *route = std::getenv("AWQ_ASCENDC_ROUTE");
    const bool auto_route = route == nullptr || route[0] == '\0' || EnvEquals(route, "auto");
    if (auto_route) {
        if (IsVecout128Candidate(m, k, n) &&
            CachedTryBuildVecoutMatmulTiling(m, static_cast<uint32_t>(kVecoutTileN),
                                             static_cast<uint32_t>(kVecoutTileK))) {
            return {AwqHostRoute::kVecout128, "vecout128", true};
        }
        return {AwqHostRoute::kScalar, "scalar", true};
    }
    if (EnvEquals(route, "scalar")) {
        return {AwqHostRoute::kScalar, "scalar", false};
    }
    if (EnvEquals(route, "stream")) {
        return {AwqHostRoute::kStream, "stream", false};
    }
    if (EnvEquals(route, "split")) {
        return {AwqHostRoute::kSplit, "split", false};
    }
    if (EnvEquals(route, "vecout") || EnvEquals(route, "vecout128")) {
        CheckVecout128Shape(m, k, n);
        return {AwqHostRoute::kVecout128, "vecout128", false};
    }
    if (EnvEquals(route, "n144")) {
        CheckWideVecoutShape("n144", m, k, n, static_cast<uint32_t>(kVecoutN144TileN));
        return {AwqHostRoute::kVecoutN144, "n144", false};
    }
    if (EnvEquals(route, "n192")) {
        CheckWideVecoutShape("n192", m, k, n, static_cast<uint32_t>(kVecoutN192TileN));
        return {AwqHostRoute::kVecoutN192, "n192", false};
    }
    TORCH_CHECK(false, "awq_w4a16_linear: unknown AWQ_ASCENDC_ROUTE=", route);
    return {AwqHostRoute::kScalar, "scalar", false};
}

void LaunchScalarDirect(const at::Tensor &x,
                        const at::Tensor &weight,
                        const at::Tensor &antiquant_scale,
                        const at::Tensor &antiquant_offset,
                        at::Tensor &y,
                        uint32_t m,
                        uint32_t k,
                        uint32_t n,
                        uint32_t core_num)
{
    auto workspace = at::empty({kWorkspaceBytes}, x.options().dtype(at::kByte));
    uint32_t total_pack = static_cast<uint32_t>(m * (n / kInt4PerPack));
    uint32_t block_dim = std::min<uint32_t>(core_num, total_pack);
    uint32_t mode = 0;
    uint32_t col_offset = 0;
    uint32_t out_n = n;
    EXEC_KERNEL_CMD(awq_w4a16_linear, block_dim,
                    x, weight, antiquant_scale, antiquant_offset, y, workspace,
                    m, k, n, mode, col_offset, out_n);
}
}  // namespace

at::Tensor awq_w4a16_linear(const at::Tensor &x,
                            const at::Tensor &weight,
                            const at::Tensor &antiquant_scale,
                            const at::Tensor &antiquant_offset)
{
    CheckAwqInputs(x, weight, antiquant_scale, antiquant_offset);

    const uint32_t m = static_cast<uint32_t>(x.size(0));
    const uint32_t k = static_cast<uint32_t>(x.size(1));
    const uint32_t n = static_cast<uint32_t>(weight.size(1) * kInt4PerPack);
    auto y = at::empty({static_cast<int64_t>(m), static_cast<int64_t>(n)}, x.options());
    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(ascendc_platform != nullptr, "awq_w4a16_linear: failed to initialize AscendC platform");
    uint32_t core_num = static_cast<uint32_t>(std::max<int64_t>(1, ascendc_platform->GetCoreNumAiv()));

    const AwqHostPlan plan = SelectAwqHostPlan(m, k, n);
    if (IsDebugTilingEnabled()) {
        std::cerr << "awq_w4a16_linear host plan: route=" << plan.name
                  << " auto=" << static_cast<int>(plan.auto_selected)
                  << " M=" << m << " K=" << k << " N=" << n << std::endl;
    }

    if (plan.route == AwqHostRoute::kScalar) {
        LaunchScalarDirect(x, weight, antiquant_scale, antiquant_offset, y, m, k, n, core_num);
        return y;
    }

    if (plan.route != AwqHostRoute::kStream && plan.route != AwqHostRoute::kSplit) {
        const int64_t workspace_bytes = static_cast<int64_t>(ascendc_platform->GetLibApiWorkSpaceSize()) +
                                        kWorkspaceBytes;
        auto workspace = at::empty({workspace_bytes}, x.options().dtype(at::kByte));
        auto tiling_placeholder = at::empty({kWorkspaceBytes}, x.options().dtype(at::kByte));
        // The VECOUT kernels use compile-time Matmul tiling and no longer read tilingGm.
        // Keep a separate GM placeholder to preserve the generated launch signature without aliasing workspace.
        if (plan.route == AwqHostRoute::kVecoutN144 || plan.route == AwqHostRoute::kVecoutN192) {
            const int64_t main_tile_n = plan.route == AwqHostRoute::kVecoutN144 ? kVecoutN144TileN : kVecoutN192TileN;
            uint32_t main_tiles = static_cast<uint32_t>(n / main_tile_n);
            uint32_t tail_n = static_cast<uint32_t>(n - static_cast<uint32_t>(main_tiles * main_tile_n));
            TORCH_CHECK(tail_n == 0 || tail_n == kVecoutTileN,
                        "awq_w4a16_linear: wide-N route requires tail N to be 0 or 128");
            uint32_t col_offset = 0;
            if (main_tiles > 0) {
                if (plan.route == AwqHostRoute::kVecoutN144) {
                    EXEC_KERNEL_CMD(awq_w4a16_linear_vecout_n144, main_tiles, x, weight, antiquant_scale,
                                    antiquant_offset, y, m, k, n, col_offset, workspace, tiling_placeholder);
                } else {
                    EXEC_KERNEL_CMD(awq_w4a16_linear_vecout_n192, main_tiles, x, weight, antiquant_scale,
                                    antiquant_offset, y, m, k, n, col_offset, workspace, tiling_placeholder);
                }
            }
            if (tail_n == kVecoutTileN) {
                uint32_t tail_block_dim = 1;
                col_offset = static_cast<uint32_t>(main_tiles * main_tile_n);
                EXEC_KERNEL_CMD(awq_w4a16_linear_vecout, tail_block_dim, x, weight, antiquant_scale,
                                antiquant_offset, y, m, k, n, col_offset, workspace, tiling_placeholder);
            }
        } else {
            uint32_t block_dim = static_cast<uint32_t>(n / kVecoutTileN);
            uint32_t col_offset = 0;
            EXEC_KERNEL_CMD(awq_w4a16_linear_vecout, block_dim, x, weight, antiquant_scale, antiquant_offset, y,
                            m, k, n, col_offset, workspace, tiling_placeholder);
        }
        return y;
    }

    at::Tensor x_group_sum;
    if (plan.route == AwqHostRoute::kSplit) {
        x_group_sum = x.view({static_cast<int64_t>(m), static_cast<int64_t>(k / kGroupSize), kGroupSize}).sum(2);
    }

    const int64_t stream_tile_n = GetStreamTileN(n);
    for (int64_t col = 0; col < static_cast<int64_t>(n); col += stream_tile_n) {
        const int64_t block_n = std::min<int64_t>(stream_tile_n, static_cast<int64_t>(n) - col);
        at::Tensor deq_block = at::empty({static_cast<int64_t>(k), block_n}, x.options());
        uint32_t mode = plan.route == AwqHostRoute::kSplit ? 2U : 1U;
        uint32_t col_offset = static_cast<uint32_t>(col);
        uint32_t out_n = static_cast<uint32_t>(block_n);
        uint32_t deq_tasks = static_cast<uint32_t>((k / kGroupSize) * ((out_n + 511U) / 512U));
        uint32_t block_dim = std::min<uint32_t>(core_num, std::max<uint32_t>(1, deq_tasks));
        auto workspace = at::empty({kWorkspaceBytes}, x.options().dtype(at::kByte));
        EXEC_KERNEL_CMD(awq_w4a16_linear, block_dim,
                        x, weight, antiquant_scale, antiquant_offset, deq_block, workspace,
                        m, k, n, mode, col_offset, out_n);
        at::Tensor out_block = at::mm(x, deq_block);
        if (plan.route == AwqHostRoute::kSplit) {
            at::Tensor scale_block = antiquant_scale.slice(1, col, col + block_n);
            at::Tensor offset_block = antiquant_offset.slice(1, col, col + block_n);
            at::Tensor offset_scaled = at::mul(offset_block, scale_block);
            out_block = at::add(out_block, at::mm(x_group_sum, offset_scaled));
        }
        y.slice(1, col, col + block_n).copy_(out_block);
    }
    return y;
}

}  // namespace ascend_kernel
