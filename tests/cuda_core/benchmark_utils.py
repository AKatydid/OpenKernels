#!/usr/bin/env python3

from __future__ import annotations

import time
from dataclasses import dataclass

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required for Python benchmarks") from exc


@dataclass(frozen=True)
class BenchmarkResult:
    eager_ms: float
    inductor_ms: float


def resolve_device(device: str) -> torch.device:
    if device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is not available for Python benchmark")
        return torch.device("cuda")
    if device == "cpu":
        return torch.device("cpu")
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise SystemExit(f"unsupported device: {device}")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(fn, args, *, iters: int = 100, warmup: int = 20, device: torch.device) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            fn(*args)
        synchronize(device)
        start = time.perf_counter()
        for _ in range(iters):
            out = fn(*args)
        synchronize(device)
        end = time.perf_counter()
        if isinstance(out, torch.Tensor):
            out.reshape(-1)[0].item()
    return (end - start) * 1000.0 / iters


def benchmark_eager_and_inductor(fn, args, *, iters: int = 100, warmup: int = 20,
                                 device: torch.device) -> BenchmarkResult:
    eager_ms = benchmark(fn, args, iters=iters, warmup=warmup, device=device)
    if not hasattr(torch, "compile"):
        raise SystemExit("torch.compile is unavailable in this PyTorch build")
    try:
        compiled = torch.compile(fn, backend="inductor")
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"torch.compile failed: {exc}") from exc
    inductor_ms = benchmark(compiled, args, iters=iters, warmup=warmup, device=device)
    return BenchmarkResult(eager_ms=eager_ms, inductor_ms=inductor_ms)
