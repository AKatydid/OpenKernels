cmake_minimum_required(VERSION 3.21)
project(OpenKernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(CUDAToolkit REQUIRED)

enable_testing()

option(OPENKERNELS_BUILD_TESTS "Build correctness tests" ON)
option(OPENKERNELS_BUILD_BENCHMARKS "Build performance benchmarks" ON)

add_subdirectory(src)

if(OPENKERNELS_BUILD_TESTS)
  add_subdirectory(tests)
endif()

if(OPENKERNELS_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
