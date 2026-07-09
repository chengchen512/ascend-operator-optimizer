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


def time_us(fn, warmup, repeat):
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


def make_inputs():
    torch.manual_seed(20260704)
    m, k, n = 16, 4096, 22016
    x = torch.randn((m, k), device="npu", dtype=torch.float16).contiguous()
    w_int = torch.randint(-8, 8, (k, n), device="npu", dtype=torch.int32).contiguous()
    weight = torch_npu.npu_convert_weight_to_int4pack(w_int).contiguous()
    scale = (torch.rand((k // 128, n), device="npu", dtype=torch.float16) * 0.04).contiguous()
    offset = (torch.randn((k // 128, n), device="npu", dtype=torch.float16) * 0.5).contiguous()
    return x, weight, scale, offset


def reference_awq(x, weight, scale, offset):
    return torch_npu.npu_weight_quant_batchmatmul(
        x=x,
        weight=weight,
        antiquant_scale=scale,
        antiquant_offset=offset,
        antiquant_group_size=128,
        bias=None,
    )


def candidate_awq(x, weight, scale, offset):
    return torch.ops.npu.awq_w4a16_tscm_vecout_direct_probe(x, weight, scale, offset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--route", choices=["vector", "scalar"], default="vector")
    args = parser.parse_args()

    torch.npu.set_device(0)
    os.environ["AWQ_VECOUT_TSCM_ROUTE"] = args.route
    x, weight, scale, offset = make_inputs()
    out = candidate_awq(x, weight, scale, offset)
    ref = reference_awq(x, weight, scale, offset)
    torch.npu.synchronize()
    torch.testing.assert_close(out, ref, rtol=args.rtol, atol=args.atol)
    diff = (out - ref).abs().float()
    result = {
        "shape": [16, 4096, 22016],
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "correctness": "pass",
        "route": args.route,
        "rtol": args.rtol,
        "atol": args.atol,
    }
    if args.benchmark:
        result["reference_us"] = time_us(lambda: reference_awq(x, weight, scale, offset), args.warmup, args.repeat)
        result["candidate_us"] = time_us(lambda: candidate_awq(x, weight, scale, offset), args.warmup, args.repeat)
        result["candidate_over_reference"] = result["candidate_us"] / result["reference_us"]
        result["warmup"] = args.warmup
        result["repeat"] = args.repeat
    print(json.dumps({"status": "ok", "case": result}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
