import time
from functools import partial
from typing import Optional

import torch.nn
import torch.utils
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(
    name="gelu_lib",
    sources=["gelu.cu"],
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
    x: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    atol: float = 1e-5,
    rtol: float = 1e-8,
):
    perf_func(x, out)
    y2 = ref_func(x)
    is_close = torch.allclose(out, y2, atol=atol, rtol=rtol)
    if not is_close:
        # print \max|out - y2||
        diff = torch.abs(out - y2)
        max_diff = torch.max(diff).item()
        max_idx = torch.argmax(diff).item()
        print(f"Max diff: {max_diff} at index {max_idx}")
        print(f"check_{tag:>10}: failed")
    else:
        print(f"check_{tag:>10}: passed")
    torch.cuda.synchronize()
    return is_close

def run_benchmark(
    perf_func: callable,
    x: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 1000,
    show_all: bool = False,
):
    if out is not None:
        out.fill_(0)
    # warmup
    if out is not None:
        for i in range(warmup):
            perf_func(x, out)
    else:
        for i in range(warmup):
            _ = perf_func(x)
    torch.cuda.synchronize()

    start = time.time()
    # iters
    if out is not None:
        for i in range(iters):
            perf_func(x, out)
    else:
        for i in range(iters):
            out = perf_func(x)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:3]
    out_val = [round(v, 8) for v in out_val]
    print(f"{out_info:>18}: {out_val}, time:{mean_time:.8f}ms")
    if show_all:
        print(out)
    return out, mean_time


Ss = [1024, 2048, 4096, 2025]
Ks = [1024, 2048, 4096, 1234]
SKs = [(S, K) for S in Ss for K in Ks]
torch.gelu = torch.nn.GELU("tanh")
for S, K in SKs:
    print("-" * 85)
    print(" " * 40 + f"S={S}, K={K}")
    x = torch.randn((S, K)).cuda().float().contiguous()
    y = torch.zeros_like(x).cuda().float().contiguous()

    run_check(lib.gelu_f32, partial(torch.gelu), x, "f32", y)
    run_check(lib.gelu_f32x4, partial(torch.gelu), x, "f32x4", y)

    run_benchmark(lib.gelu_f32, x, "f32", y)
    run_benchmark(lib.gelu_f32x4, x, "f32x4", y)
    run_benchmark(partial(torch.gelu), x, "f32_th")

    print("-" * 85)