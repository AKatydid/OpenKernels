#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_utils import benchmark_eager_and_inductor, resolve_device  # noqa: E402


def sgemm_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("m", nargs="?", type=int, default=256)
    parser.add_argument("n", nargs="?", type=int, default=256)
    parser.add_argument("k", nargs="?", type=int, default=256)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = resolve_device(args.device)
    a = torch.randn(args.m, args.k, device=device, dtype=torch.float32)
    b = torch.randn(args.k, args.n, device=device, dtype=torch.float32)
    result = benchmark_eager_and_inductor(
        sgemm_op, (a, b), iters=args.iters, warmup=args.warmup, device=device
    )
    print(
        f"PYBENCH test_sgemm eager_ms={result.eager_ms:.6f} "
        f"inductor_ms={result.inductor_ms:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
