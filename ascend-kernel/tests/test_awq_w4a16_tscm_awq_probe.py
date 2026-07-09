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


def make_inputs():
    torch.manual_seed(20260703)
    m, k, n = 16, 128, 64
    x = torch.randn((m, k), device="npu", dtype=torch.float16).contiguous()
    w_int = torch.randint(-8, 8, (k, n), dtype=torch.int32)
    weight = torch_npu.npu_convert_weight_to_int4pack(w_int.npu().contiguous()).contiguous()
    scale = (torch.rand((1, n), device="npu", dtype=torch.float16) * 0.04).contiguous()
    offset = (torch.randn((1, n), device="npu", dtype=torch.float16) * 0.5).contiguous()
    return x, weight, w_int.npu().contiguous(), scale, offset


def reference_awq(x, unpacked_weight, scale, offset):
    dequant = (unpacked_weight.to(torch.float16) + offset) * scale
    return torch.mm(x, dequant)


def candidate_awq(x, weight, scale, offset):
    return torch.ops.npu.awq_w4a16_tscm_awq_probe(x, weight, scale, offset)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--stage", type=int, default=None)
    args = parser.parse_args()

    if args.stage is not None:
        os.environ["AWQ_TSCM_STAGE"] = str(args.stage)
    stage = int(os.environ.get("AWQ_TSCM_STAGE", "3"))

    torch.npu.set_device(0)
    x, weight, unpacked_weight, scale, offset = make_inputs()
    out = candidate_awq(x, weight, scale, offset)
    torch.npu.synchronize()
    if stage < 3 or stage == 5 or stage == 7 or stage == 8:
        print(json.dumps({"status": "ok", "stage": stage, "synchronize": "pass"}, indent=2, ensure_ascii=False))
        return

    ref = reference_awq(x, unpacked_weight, scale, offset).to(out.dtype)
    torch.testing.assert_close(out, ref, rtol=args.rtol, atol=args.atol)
    diff = (out - ref).abs().float()
    result = {
        "shape": [16, 128, 64],
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "correctness": "pass",
        "rtol": args.rtol,
        "atol": args.atol,
    }
    if args.benchmark:
        result["reference_us"] = time_us(lambda: reference_awq(x, unpacked_weight, scale, offset), args.warmup, args.repeat)
        result["candidate_us"] = time_us(lambda: candidate_awq(x, weight, scale, offset), args.warmup, args.repeat)
        result["candidate_over_reference"] = result["candidate_us"] / result["reference_us"]
        result["warmup"] = args.warmup
        result["repeat"] = args.repeat
    print(json.dumps({"status": "ok", "case": result}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
