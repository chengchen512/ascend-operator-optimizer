#!/usr/bin/env python3
import argparse
import json
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


def pack_a_like_official_tscm_sample(a):
    # Official asc-devkit matmul_tscm/gen_data.py writes A in NZ layout before
    # DataCopy(scmTensor, aGlobal, M*K): [M,K] -> [K/16, M/16, 16, 16].
    return a.reshape(4, 16, 4, 16).permute(2, 0, 1, 3).contiguous().reshape(64, 64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtol", type=float, default=3e-2)
    parser.add_argument("--atol", type=float, default=3e-2)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()

    torch.npu.set_device(0)
    torch.manual_seed(20260703)
    a = torch.randn((64, 64), device="npu", dtype=torch.float16).contiguous()
    b = torch.randn((64, 64), device="npu", dtype=torch.float16).contiguous()
    bias = torch.randn((64,), device="npu", dtype=torch.float32).contiguous()
    a_tscm = pack_a_like_official_tscm_sample(a)

    out = torch.ops.npu.matmul_tscm_direct_probe(a_tscm, b, bias)
    ref = torch.matmul(a.float(), b.float()) + bias
    torch.npu.synchronize()
    torch.testing.assert_close(out, ref, rtol=args.rtol, atol=args.atol)
    diff = (out - ref).abs().float()
    result = {
        "shape": [64, 64, 64],
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "correctness": "pass",
        "rtol": args.rtol,
        "atol": args.atol,
    }
    if args.benchmark:
        result["reference_us"] = time_us(lambda: torch.matmul(a.float(), b.float()) + bias, args.warmup, args.repeat)
        result["candidate_us"] = time_us(
            lambda: torch.ops.npu.matmul_tscm_direct_probe(a_tscm, b, bias), args.warmup, args.repeat
        )
        result["candidate_over_reference"] = result["candidate_us"] / result["reference_us"]
        result["warmup"] = args.warmup
        result["repeat"] = args.repeat
    print(json.dumps({"status": "ok", "case": result}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
