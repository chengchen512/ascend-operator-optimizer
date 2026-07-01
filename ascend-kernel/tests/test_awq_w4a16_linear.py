#!/usr/bin/env python3
import argparse
import json
import os
import pathlib

import torch
import torch_npu

try:
    import ascend_kernel  # noqa: F401
except ImportError:
    root = pathlib.Path(__file__).resolve().parents[1]
    so_files = sorted((root / "python" / "ascend_kernel" / "ascend_kernel" / "lib").glob("*.so"))
    if not so_files:
        raise
    torch.ops.load_library(str(so_files[0]))


def make_inputs(m: int, k: int, n: int):
    assert k % 128 == 0 and k > 128 and n % 8 == 0
    torch.manual_seed(20260630 + m + k + n)
    x = torch.randn((m, k), device="npu", dtype=torch.float16).contiguous()
    w_int = torch.randint(-8, 8, (k, n), dtype=torch.int32)
    weight = torch_npu.npu_convert_weight_to_int4pack(w_int.npu().contiguous()).contiguous()
    scale = (torch.rand((k // 128, n), device="npu", dtype=torch.float16) * 0.04).contiguous()
    offset = (torch.randn((k // 128, n), device="npu", dtype=torch.float16) * 0.5).contiguous()
    return x, weight, scale, offset


def vendor_awq(x, weight, scale, offset):
    return torch_npu.npu_weight_quant_batchmatmul(
        x=x,
        weight=weight,
        antiquant_scale=scale,
        antiquant_offset=offset,
        antiquant_group_size=128,
        bias=None,
    )


def candidate_awq(x, weight, scale, offset):
    return torch.ops.npu.awq_w4a16_linear(x, weight, scale, offset)


def time_us(fn, warmup: int, repeat: int):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.npu.synchronize()
    return float(start.elapsed_time(end) * 1000.0 / repeat)


def run_case(m: int, k: int, n: int, warmup: int, repeat: int, benchmark: bool, rtol: float, atol: float):
    x, weight, scale, offset = make_inputs(m, k, n)
    ref = vendor_awq(x, weight, scale, offset)
    out = candidate_awq(x, weight, scale, offset)
    torch.npu.synchronize()
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
    diff = (out - ref).abs().float()
    result = {
        "shape": [m, k, n],
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "correctness": "pass",
        "rtol": rtol,
        "atol": atol,
        "route_env": os.environ.get("AWQ_ASCENDC_ROUTE", "stream"),
        "stream_tile_n": os.environ.get("AWQ_STREAM_N_TILE", "12288"),
    }
    if benchmark:
        vendor_us = time_us(lambda: vendor_awq(x, weight, scale, offset), warmup, repeat)
        candidate_us = time_us(lambda: candidate_awq(x, weight, scale, offset), warmup, repeat)
        result.update({
            "warmup": warmup,
            "repeat": repeat,
            "candidate_us": candidate_us,
            "vendor_us": vendor_us,
            "candidate_over_vendor": candidate_us / vendor_us,
        })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=2e-2)
    args = parser.parse_args()
    torch.npu.set_device(0)
    result = run_case(args.m, args.k, args.n, args.warmup, args.repeat, args.benchmark, args.rtol, args.atol)
    print(json.dumps({"status": "ok", "case": result}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()