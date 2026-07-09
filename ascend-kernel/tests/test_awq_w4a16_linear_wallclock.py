#!/usr/bin/env python3
import argparse
import json
import time

import torch

import test_awq_w4a16_linear as awq_test


def wallclock_us(fn, warmup: int, repeat: int):
    last = None
    for _ in range(warmup):
        last = fn()
    torch.npu.synchronize()

    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        last = fn()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1_000_000.0)
    if last is not None:
        _ = last
    return {
        "avg_us": sum(samples) / len(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "repeat": repeat,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--n", type=int, default=22016)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--which", choices=("vendor", "candidate", "both"), default="both")
    args = parser.parse_args()

    torch.npu.set_device(0)
    x, weight, scale, offset = awq_test.make_inputs(args.m, args.k, args.n)
    result = {
        "shape": [args.m, args.k, args.n],
        "method": "perf_counter_plus_torch_npu_synchronize",
        "which": args.which,
    }
    if args.which in ("vendor", "both"):
        print("running vendor", flush=True)
        result["vendor"] = wallclock_us(lambda: awq_test.vendor_awq(x, weight, scale, offset), args.warmup, args.repeat)
    if args.which in ("candidate", "both"):
        print("running candidate", flush=True)
        result["candidate"] = wallclock_us(lambda: awq_test.candidate_awq(x, weight, scale, offset), args.warmup, args.repeat)
    if "vendor" in result and "candidate" in result:
        result["candidate_over_vendor"] = result["candidate"]["avg_us"] / result["vendor"]["avg_us"]
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
