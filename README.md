# OpenKernels

A pure C++/CUDA project for implementing and benchmarking common GPU kernels.

## Project Layout

```text
OpenKernels/
├── CMakeLists.txt
├── README.md
├── LICENSE
├── include/
│   └── cuda_core/
│       └── common_utils.cuh
├── src/
│   ├── CMakeLists.txt
│   ├── cuda_core/
│   │   ├── activation/         # ReLu, GELU
│   │   ├── blas/               # SGEMM, SGEMBV
│   │   ├── normalization/      # Softmax, BatchNorm
│   │   └── scan/               # Cumsum
│   └── tensor_core/
├── scripts/
│   ├── build.sh
│   └── make_clean.sh
└── build/
```

## Build

The build command compiles all files containing a `main` function in `src/cuda_core` and `src/tensor_core`.

```bash
./scripts/build.sh
```

Clean build artifacts.

```bash
./scripts/make_clean.sh
```

## Run

```bash
# Available SGEMM executables (generated from src/cuda_core/blas/*.cu with main)
./build/src/sgemm_v1_tiling
./build/src/sgemm_v2_vectorize
./build/src/sgemm_v3_swizzle
./build/src/sgemm_v4_double_buffer
```

All SGEMM executables currently use the same CLI format:

```bash
./build/src/sgemm_v4_double_buffer <M> <N> <K> [warm_up] [test_iter]
```

- `warm_up` is optional, default: `10`
- `test_iter` is optional, default: `200`

Example:

```bash
./build/src/sgemm_v4_double_buffer 4096 4096 4096
./build/src/sgemm_v4_double_buffer 4096 4096 4096 20 500
```

## Notes

- CMake scans `src/cuda_core/**/*.cu` and only builds files containing `main` as standalone executables.
- Shared CUDA helpers are in `include/cuda_core/common_utils.cuh`.
