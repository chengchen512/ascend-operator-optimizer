// Licensed under the BSD 3-Clause License (the "License").

#include <algorithm>
#include <cstdlib>
#include <cstdint>

#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_awq_w4a16_linear.h"

namespace ascend_kernel {
namespace {

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

}  // namespace

at::Tensor awq_w4a16_linear(const at::Tensor &x,
                            const at::Tensor &weight,
                            const at::Tensor &antiquant_scale,
                            const at::Tensor &antiquant_offset)
{
    TORCH_CHECK(x.dim() == 2, "awq_w4a16_linear: x must be 2D");
    TORCH_CHECK(weight.dim() == 2, "awq_w4a16_linear: weight must be 2D int4pack");
    TORCH_CHECK(antiquant_scale.dim() == 2, "awq_w4a16_linear: antiquant_scale must be 2D");
    TORCH_CHECK(antiquant_offset.dim() == 2, "awq_w4a16_linear: antiquant_offset must be 2D");
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
    const int64_t packed_n = weight.size(1);
    const int64_t n = packed_n * 8;
    constexpr int64_t group_size = 128;

    TORCH_CHECK(k > group_size, "awq_w4a16_linear: k must be greater than group_size=128 for vendor-compatible AWQ");
    TORCH_CHECK(k % group_size == 0, "awq_w4a16_linear: k must be divisible by 128");
    TORCH_CHECK(weight.size(0) == k, "awq_w4a16_linear: weight.shape[0] must equal K");
    TORCH_CHECK(antiquant_scale.size(0) == k / group_size && antiquant_scale.size(1) == n,
                "awq_w4a16_linear: antiquant_scale shape must be [K/128, N]");
    TORCH_CHECK(antiquant_offset.size(0) == k / group_size && antiquant_offset.size(1) == n,
                "awq_w4a16_linear: antiquant_offset shape must be [K/128, N]");
    TORCH_CHECK(m <= static_cast<int64_t>(UINT32_MAX) && k <= static_cast<int64_t>(UINT32_MAX) &&
                    n <= static_cast<int64_t>(UINT32_MAX) && m * packed_n <= static_cast<int64_t>(UINT32_MAX) &&
                    k * n <= static_cast<int64_t>(UINT32_MAX),
                "awq_w4a16_linear: shape too large for current uint32 tiling");

    at::Tensor output = at::empty({m, n}, x.options());
    if (m == 0 || n == 0) {
        return output;
    }

    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t core_num = static_cast<uint32_t>(std::max<int64_t>(1, ascendc_platform->GetCoreNumAiv()));

    uint32_t m32 = static_cast<uint32_t>(m);
    uint32_t k32 = static_cast<uint32_t>(k);
    uint32_t n32 = static_cast<uint32_t>(n);

    const char *route = std::getenv("AWQ_ASCENDC_ROUTE");
    const bool use_scalar_route = route != nullptr && route[0] == 's';
    const bool use_full_deq_route = route != nullptr && route[0] == 'f';
    if (!use_scalar_route) {
        uint32_t mode = 1;
        if (use_full_deq_route) {
            at::Tensor deq_weight = at::empty({k, n}, x.options());
            uint32_t deq_total = static_cast<uint32_t>(k * n);
            uint32_t deq_tasks = static_cast<uint32_t>((k / group_size) * ((n + 511) / 512));
            uint32_t block_dim = std::min<uint32_t>(core_num, std::max<uint32_t>(1, deq_tasks));
            uint32_t col_offset = 0;
            uint32_t out_n = n32;
            EXEC_KERNEL_CMD(awq_w4a16_linear, block_dim,
                            x, weight, antiquant_scale, antiquant_offset, deq_weight,
                            m32, k32, n32, deq_total, mode, col_offset, out_n);
            return at::mm(x, deq_weight);
        }

        const int64_t stream_tile_n = GetStreamTileN(n);
        for (int64_t col = 0; col < n; col += stream_tile_n) {
            const int64_t block_n = std::min<int64_t>(stream_tile_n, n - col);
            at::Tensor deq_block = at::empty({k, block_n}, x.options());
            uint32_t deq_total = static_cast<uint32_t>(k * block_n);
            uint32_t deq_tasks = static_cast<uint32_t>((k / group_size) * ((block_n + 511) / 512));
            uint32_t block_dim = std::min<uint32_t>(core_num, std::max<uint32_t>(1, deq_tasks));
            uint32_t col_offset = static_cast<uint32_t>(col);
            uint32_t out_n = static_cast<uint32_t>(block_n);
            EXEC_KERNEL_CMD(awq_w4a16_linear, block_dim,
                            x, weight, antiquant_scale, antiquant_offset, deq_block,
                            m32, k32, n32, deq_total, mode, col_offset, out_n);
            at::Tensor out_block = at::mm(x, deq_block);
            output.slice(1, col, col + block_n).copy_(out_block);
        }
        return output;
    }

    uint32_t total_pack = static_cast<uint32_t>(m * packed_n);
    uint32_t block_dim = std::min<uint32_t>(core_num, total_pack);
    uint32_t total32 = static_cast<uint32_t>(m * n);
    uint32_t mode = 0;
    uint32_t col_offset = 0;
    uint32_t out_n = n32;
    EXEC_KERNEL_CMD(awq_w4a16_linear, block_dim,
                    x, weight, antiquant_scale, antiquant_offset, output,
                    m32, k32, n32, total32, mode, col_offset, out_n);
    return output;
}

}  // namespace ascend_kernel