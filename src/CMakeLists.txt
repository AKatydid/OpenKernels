set(OPENKERNELS_CUDA_CORE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cuda_core)

set(OPENKERNELS_CORE_SOURCES
  ${OPENKERNELS_CUDA_CORE_DIR}/activation/gelu.cu
  ${OPENKERNELS_CUDA_CORE_DIR}/activation/relu.cu
  ${OPENKERNELS_CUDA_CORE_DIR}/normalization/rms_norm.cu
  ${OPENKERNELS_CUDA_CORE_DIR}/normalization/softmax.cu
  ${OPENKERNELS_CUDA_CORE_DIR}/blas/sgemv.cu
  ${OPENKERNELS_CUDA_CORE_DIR}/scan/cumsum.cu
)

add_library(openkernels STATIC)

target_sources(openkernels PRIVATE
  ${OPENKERNELS_CORE_SOURCES}
)

target_include_directories(openkernels PUBLIC
  ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(openkernels PUBLIC CUDA::cudart)

target_compile_options(openkernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)
