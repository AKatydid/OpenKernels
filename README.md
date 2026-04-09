# OpenKernels

A pure C++/CUDA project for implementing and benchmarking common GPU kernels.

## Project Layout

```text
OpenKernels/
├── CMakeLists.txt
├── README.md
├── LICENSE
├── include/
│   └── openkernels/
│       ├── common_utils.cuh
│       └── kernels.h
├── src/
│   ├── CMakeLists.txt
│   ├── activation/
│   │   ├── gelu.cu
│   │   └── relu.cu
│   ├── normalization/
│   │   ├── rms_norm.cu
│   │   └── softmax.cu
│   ├── blas/
│   │   ├── sgemv.cu
│   │   ├── sgemm_naive.cu
│   │   ├── sgemm_v1_tiling.cu
│   │   ├── sgemm_v2_vectorize.cu
│   │   ├── sgemm_v3_swizzle.cu
│   │   └── sgemm_v4_double_buffer.cu
│   └── scan/
│       └── cumsum.cu
├── tests/
│   ├── CMakeLists.txt
│   ├── test_sgemm.cu
│   ├── test_softmax.cu
│   ├── test_relu.cu
│   ├── test_gelu.cu
│   ├── test_rms_norm.cu
│   ├── test_sgemv.cu
│   └── test_cumsum.cu
├── benchmarks/
│   ├── CMakeLists.txt
│   ├── bench_sgemm.cu
│   ├── bench_softmax.cu
│   └── common.h
├── scripts/
│   ├── build.sh
│   └── plot_performance.py
└── tensor_core/
```

## Build

```bash
./scripts/build.sh
```

Or manually:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Run

```bash
# correctness
./build/tests/test_softmax
./build/tests/test_sgemm 512 512 512

# performance
./build/benchmarks/bench_sgemm 2048 2048 2048
./build/benchmarks/bench_softmax 7 8 1024 50257 50
```

## Notes

- Public APIs for reusable kernels are declared in `include/openkernels/kernels.h`.
- Shared CUDA helpers (`CUDA_CHECK`, `CudaTimer`, random helpers) are in `include/openkernels/common_utils.cuh`.
- `bench_softmax.cu` is migrated from the previous `softmax_new.cu` standalone benchmark.
