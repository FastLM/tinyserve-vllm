cmake_minimum_required(VERSION 3.16)
project(tinyserve LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(tinyserve
  selector/bounding_box.cpp
  selector/topk.cpp
  attention/sparse_attention.cpp
  cache/page_cache.cpp
  host/src/api.cpp
)
target_include_directories(tinyserve
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/host/include
)

add_executable(tinyserve_demo host/src/demo.cpp)
target_link_libraries(tinyserve_demo PRIVATE tinyserve)

enable_testing()
foreach(test_name selector attention)
  add_executable(test_${test_name} tests/test_${test_name}.cpp)
  target_link_libraries(test_${test_name} PRIVATE tinyserve)
  add_test(NAME ${test_name} COMMAND test_${test_name})
endforeach()

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  add_library(tinyserve_kernels STATIC kernels/tinyserve_kernels.cu)
  set_target_properties(tinyserve_kernels PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
  )
  target_include_directories(tinyserve_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/kernels)
  message(STATUS "CUDA kernels enabled")
else()
  message(STATUS "CUDA compiler not found; building the C++ kernels only")
endif()
