import time
from functools import partial
from typing import Optional

import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(
    name="sgemv_lib",
    sources=["sgemv_new.cu"],
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


def run_check(
    perf_func: callable,
    ref_func: callable,
    A: torch.Tensor,
    B: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    atol: float = 1e-4,
    rtol: float = 1e-8,
):
    perf_func(A, B, out)
    y1 = out.clone()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    y2 = ref_func(A, B)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    is_close = torch.allclose(y1, y2, atol=atol, rtol=rtol)
    if not is_close:
        max_diff = torch.max(y1-y2).item()
        print(f"check_{tag:>10}: failed, max_diff: {max_diff}")
    else:
        print(f"check_{tag:>10}: passed")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return is_close


def run_benchmark(
    perf_func: callable,
    a: torch.Tensor,
    b: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 200,
    show_all: bool = False,
):
    if out is not None:
        out.fill_(0)
    
    # warmup
    if out is not None:
        for i in range(warmup):
            perf_func(a, b, out)
    else:
        for i in range(warmup):
            _ = perf_func(a, b)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start_event.record()

    # iters
    if out is not None:
        for i in range(iters):
            perf_func(a, b, out)
    else:
        for i in range(iters):
            out = perf_func(a, b)
    
    end_event.record()
    torch.cuda.synchronize()

    total_time = start_event.elapsed_time(end_event)
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:3]
    out_val = [round(v, 8) for v in out_val]
    print(f"{out_info:>13}: {out_val}, time:{mean_time:.8f}ms")
    if show_all:
        print(out)
    return out.clone(), mean_time


print("-" * 80)
M, N, K = 1024, 1, 128
a = torch.randn((M, K)).cuda().float().contiguous()
b = torch.randn((K, N)).cuda().float().contiguous()
c = torch.randn((M, N)).cuda().float().contiguous()

run_check(lib.sgemv_k32_f32, partial(torch.matmul, out=c), a, b, "k32f32", c)
run_check(lib.sgemv_k128_f32x4, partial(torch.matmul, out=c), a, b, "k128f32x4", c)

run_benchmark(lib.sgemv_k32_f32, a, b, "k32f32", c)
run_benchmark(lib.sgemv_k128_f32x4, a, b, "k128f32x4", c)
run_benchmark(partial(torch.matmul, out=c), a, b, "f32_th")
print("-" * 80)

M, N, K = 1024, 1, 16
a = torch.randn((M, K)).cuda().float().contiguous()
b = torch.randn((K, N)).cuda().float().contiguous()
c = torch.randn((M, N)).cuda().float().contiguous()

run_check(lib.sgemv_k16_f32, partial(torch.matmul, out=c), a, b, "k16f32", c)

run_benchmark(lib.sgemv_k16_f32, a, b, "k16f32", c)
run_benchmark(partial(torch.matmul, out=c), a, b, "f32_th")
print("-" * 80)

M, N, K = 1024, 1, 8192
print(" " * 30 + f"M={M}, K={K}")
a = torch.randn((M, K)).cuda().float().contiguous()
b = torch.randn((K, N)).cuda().float().contiguous()
c = torch.zeros((M, N)).cuda().float().contiguous()

run_check(lib.sgemv_k32_f32, partial(torch.matmul, out=c), a, b, "k32f32", c)
run_check(lib.sgemv_k128_f32x4, partial(torch.matmul, out=c), a, b, "k128f32x4", c)
run_check(lib.sgemv_splitk_f32, partial(torch.matmul, out=c), a, b, "splitk", c)
run_check(lib.sgemv_splitk_smem_f32, partial(torch.matmul, out=c), a, b, "splitk_smem", c)

run_benchmark(lib.sgemv_k32_f32, a, b, "k32f32", c)
run_benchmark(lib.sgemv_k128_f32x4, a, b, "k128f32x4", c)
run_benchmark(lib.sgemv_splitk_f32, a, b, "splitk", c)
run_benchmark(lib.sgemv_splitk_smem_f32, a, b, "splitk_smem", c)
run_benchmark(partial(torch.matmul, out=c), a, b, "f32_th")

print("-" * 80)
M, N, K = 2048, 1, 16384
print(" " * 30 + f"M={M}, K={K}")
a = torch.randn((M, K)).cuda().float().contiguous()
b = torch.randn((K, N)).cuda().float().contiguous()
c = torch.zeros((M, N)).cuda().float().contiguous()

run_check(lib.sgemv_k32_f32, partial(torch.matmul, out=c), a, b, "k32f32", c)
run_check(lib.sgemv_k128_f32x4, partial(torch.matmul, out=c), a, b, "k128f32x4", c)
run_check(lib.sgemv_splitk_f32, partial(torch.matmul, out=c), a, b, "splitk", c)
run_check(lib.sgemv_splitk_smem_f32, partial(torch.matmul, out=c), a, b, "splitk_smem", c)

run_benchmark(lib.sgemv_k32_f32, a, b, "k32f32", c)
run_benchmark(lib.sgemv_k128_f32x4, a, b, "k128f32x4", c)
run_benchmark(lib.sgemv_splitk_f32, a, b, "splitk", c)
run_benchmark(lib.sgemv_splitk_smem_f32, a, b, "splitk_smem", c)
run_benchmark(partial(torch.matmul, out=c), a, b, "f32_th")

print("-" * 80)
M, N, K = 2048, 1, 131072
print(" " * 30 + f"M={M}, K={K}")
a = torch.randn((M, K)).cuda().float().contiguous()
b = torch.randn((K, N)).cuda().float().contiguous()
c = torch.zeros((M, N)).cuda().float().contiguous()

run_check(lib.sgemv_k32_f32, partial(torch.matmul, out=c), a, b, "k32f32", c)
run_check(lib.sgemv_k128_f32x4, partial(torch.matmul, out=c), a, b, "k128f32x4", c)
run_check(lib.sgemv_splitk_f32, partial(torch.matmul, out=c), a, b, "splitk", c)
run_check(lib.sgemv_splitk_smem_f32, partial(torch.matmul, out=c), a, b, "splitk_smem", c)

run_benchmark(lib.sgemv_k32_f32, a, b, "k32f32", c)
run_benchmark(lib.sgemv_k128_f32x4, a, b, "k128f32x4", c)
run_benchmark(lib.sgemv_splitk_f32, a, b, "splitk", c)
run_benchmark(lib.sgemv_splitk_smem_f32, a, b, "splitk_smem", c)
run_benchmark(partial(torch.matmul, out=c), a, b, "f32_th")
print("-" * 80)