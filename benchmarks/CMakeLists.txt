add_executable(bench_sgemm bench_sgemm.cu)
target_include_directories(bench_sgemm PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}
)
target_link_libraries(bench_sgemm PRIVATE CUDA::cudart CUDA::cublas)
target_compile_options(bench_sgemm PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

add_executable(bench_softmax bench_softmax.cu)
target_include_directories(bench_softmax PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(bench_softmax PRIVATE CUDA::cudart CUDA::cublas CUDA::cublasLt)
target_compile_options(bench_softmax PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)
