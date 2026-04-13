set(OPENKERNELS_CUDA_CORE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cuda_core)

option(OPENKERNELS_BUILD_CUDA_EXECUTABLES "Build standalone CUDA executables under src/cuda_core" ON)

if(OPENKERNELS_BUILD_CUDA_EXECUTABLES)
  file(GLOB_RECURSE OPENKERNELS_CUDA_EXEC_SOURCES CONFIGURE_DEPENDS
    ${OPENKERNELS_CUDA_CORE_DIR}/*.cu
  )

  foreach(CUDA_SRC IN LISTS OPENKERNELS_CUDA_EXEC_SOURCES)
    file(READ ${CUDA_SRC} CUDA_SRC_CONTENT)
    string(REGEX MATCH "int[ \t\r\n]+main[ \t\r\n]*\\(" HAS_MAIN ${CUDA_SRC_CONTENT})
    if(NOT HAS_MAIN)
      continue()
    endif()

    get_filename_component(CUDA_EXE ${CUDA_SRC} NAME_WE)
    add_executable(${CUDA_EXE} ${CUDA_SRC})
    target_include_directories(${CUDA_EXE} PRIVATE
      ${PROJECT_SOURCE_DIR}/include
      ${PROJECT_SOURCE_DIR}/include/cuda_core
    )
    target_link_libraries(${CUDA_EXE} PRIVATE CUDA::cudart CUDA::cublas)
    target_compile_options(${CUDA_EXE} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    )
  endforeach()
endif()
