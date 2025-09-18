import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
from typing import Optional

lib = load(
    name="rms_norm_lib",
    sources=["rms_norm.cu"],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        norm = torch.sqrt(torch.mean(x ** 2, dim = 1, keepdim=True) + 1e-5)
        return x / norm


def run_check(
    x: torch.Tensor,
    perf_func: callable,
    ref_func: callable,
    tar: str,
    out: Optional[torch.Tensor] = None,
    atol: float = 1e-5,
    rtol: float = 1e-8
):
    with torch.no_grad():
        if out is not None:
            out.fill_(0)
        perf_func(x, out)
        torch.cuda.synchronize()

        ref = ref_func(x)
        torch.cuda.synchronize()
        closed = torch.allclose(out, ref, atol=atol, rtol=rtol)
        if closed:
            print(f"[Success] tar: {tar} Success!")
        else:
            print(f"[Error] max diff with {torch.max(out-ref).item()}")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_benchmark(
    perf_func: callable,
    x: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 100,
    show_all: bool = False,
):
    with torch.no_grad():
        if out is not None:
            out.fill_(0)

        # warmup
        if out is not None:
            for _ in range(warmup):
                perf_func(x, out)
        else:
            for _ in range(warmup):
                _ = perf_func(x)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()

        if out is not None:
            for _ in range(iters):
                perf_func(x, out)
        else:
            for _ in range(iters):
                out = perf_func(x)

        end_event.record()
        torch.cuda.synchronize()

        total_time = start_event.elapsed_time(end_event)
        mean_time = total_time / iters

        out_info = f"out_{tag}"
        out_val = out.flatten().detach().cpu().numpy().tolist()[:3]
        out_val = [round(v, 8) for v in out_val]
        out_val = [f"{v:<12}" for v in out_val]
        print(f"{out_info:>17}: {out_val}, time:{mean_time:.8f}ms")

        if show_all:
            print(out)

        return out, mean_time


m = Model()

print("-" * 85)
N, K = 112, 4096
print(" " * 40 + f"N={N}, K={K}")
x = torch.randn((N, K)).cuda().float().contiguous()
out = torch.zeros_like(x).cuda().float().contiguous()

run_check(x, lib.rmsnorm_f32, m, "f32", out)

run_benchmark(lib.rmsnorm_f32, x, "f32", out)
run_benchmark(m, x, "f32_th")
print("-" * 85)

N, K = 112, 40961
print(" " * 40 + f"N={N}, K={K}")
x = torch.randn((N, K)).cuda().float().contiguous()
out = torch.zeros_like(x).cuda().float().contiguous()

run_check(x, lib.rmsnorm_f32, m, "f32", out)

run_benchmark(lib.rmsnorm_f32, x, "f32", out)
run_benchmark(m, x, "f32_th")

print("-" * 85)
N, K = 112, 409611
print(" " * 40 + f"N={N}, K={K}")
x = torch.randn((N, K)).cuda().float().contiguous()
out = torch.zeros_like(x).cuda().float().contiguous()

run_check(x, lib.rmsnorm_f32, m, "f32", out)

run_benchmark(lib.rmsnorm_f32, x, "f32", out)
run_benchmark(m, x, "f32_th")
print("-" * 85)