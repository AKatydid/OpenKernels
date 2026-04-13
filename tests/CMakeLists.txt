function(add_openkernels_test target source)
  add_executable(${target} ${source})
  set(TEST_UTILS_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/test_utils.h)
  target_include_directories(${target} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}
  )
  target_link_libraries(${target} PRIVATE openkernels CUDA::cudart)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:--pre-include=${TEST_UTILS_HEADER}>
  )
  add_test(NAME ${target} COMMAND ${target} --verify)
endfunction()

add_openkernels_test(test_relu cuda_core/activation/test_relu.cu)
add_openkernels_test(test_gelu cuda_core/activation/test_gelu.cu)
add_openkernels_test(test_rms_norm cuda_core/normalization/test_rms_norm.cu)
add_openkernels_test(test_sgemv cuda_core/blas/test_sgemv.cu)
add_openkernels_test(test_softmax cuda_core/normalization/test_softmax.cu)
add_openkernels_test(test_cumsum cuda_core/scan/test_cumsum.cu)

# Reuse the existing standalone SGEMM driver as a test entry.
add_executable(test_sgemm cuda_core/blas/test_sgemm.cu)
target_include_directories(test_sgemm PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}
)
target_link_libraries(test_sgemm PRIVATE CUDA::cudart)
target_compile_options(test_sgemm PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
  $<$<COMPILE_LANGUAGE:CUDA>:--pre-include=${CMAKE_CURRENT_SOURCE_DIR}/include/test_utils.h>
)
add_test(NAME test_sgemm COMMAND test_sgemm --verify 256 256 256)
