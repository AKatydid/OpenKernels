function(add_openkernels_test target source)
  add_executable(${target} ${source})
  target_include_directories(${target} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}
  )
  target_link_libraries(${target} PRIVATE openkernels CUDA::cudart)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
  )
  add_test(NAME ${target} COMMAND ${target})
endfunction()

add_openkernels_test(test_relu test_relu.cu)
add_openkernels_test(test_gelu test_gelu.cu)
add_openkernels_test(test_rms_norm test_rms_norm.cu)
add_openkernels_test(test_sgemv test_sgemv.cu)
add_openkernels_test(test_softmax test_softmax.cu)
add_openkernels_test(test_cumsum test_cumsum.cu)

# Reuse the existing standalone SGEMM driver as a test entry.
add_executable(test_sgemm test_sgemm.cu)
target_include_directories(test_sgemm PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}
)
target_link_libraries(test_sgemm PRIVATE CUDA::cudart CUDA::cublas)
target_compile_options(test_sgemm PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)
add_test(NAME test_sgemm COMMAND test_sgemm 512 512 512)
