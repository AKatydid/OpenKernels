#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_utils import benchmark_eager_and_inductor, resolve_device  # noqa: E402


def rms_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("n", nargs="?", type=int, default=16)
    parser.add_argument("k", nargs="?", type=int, default=128)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = resolve_device(args.device)
    x = torch.randn(args.n, args.k, device=device, dtype=torch.float32)
    result = benchmark_eager_and_inductor(
        rms_norm, (x,), iters=args.iters, warmup=args.warmup, device=device
    )
    print(
        f"PYBENCH test_rms_norm eager_ms={result.eager_ms:.6f} "
        f"inductor_ms={result.inductor_ms:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
