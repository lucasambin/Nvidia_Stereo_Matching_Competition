cmake_minimum_required(VERSION 3.10)
# create project
set(PROJECT_NAME sgm_nano)
project(${PROJECT_NAME})

set(TENSORRT_LIB_PATH "/usr/lib/aarch64-linux-gnu/" CACHE STRING "absolute path to tensorrt libraries")
set(TENSORRT_INCLUDE_PATH "/usr/include/aarch64-linux-gnu/" CACHE STRING "absolute path to tensorrt public header")

find_package( CUDA REQUIRED )
find_package( OpenCV REQUIRED )

# TensorRT
find_library(LIBNVINFER NAME nvinfer HINTS ${TENSORRT_LIB_PATH} REQUIRED)
find_library(LIBNVINFER_PLUGIN NAME nvinfer_plugin HINTS ${TENSORRT_LIB_PATH} REQUIRED)
find_library(LIBNVPARSERS NAME nvparsers HINTS ${TENSORRT_LIB_PATH} REQUIRED)
find_library(LIBNVONNXPARSER NAME nvonnxparser HINTS ${TENSORRT_LIB_PATH} REQUIRED)

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -lineinfo
    # NVIDIA GTX 950M (2 Gb)
    #-gencode=arch=compute_50,code=sm_50
    # NVIDIA Jetson Nano Maxwell (2 Gb)
    -gencode=arch=compute_53,code=sm_53
    )

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_INCLUDE_PATH})

#include_directories(/usr/lib/aarch64-linux-gnu/)
include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/inc ${CMAKE_CURRENT_SOURCE_DIR}/src )

include_directories(${TENSORRT_INCLUDE_DIR}/)



message(STATUS "
=============> Final Config <===============
TensorRT headers: ${TENSORRT_INCLUDE_DIR}
TensorRT library: ${LIBNVINFER}
============================================
")

#link_directories(/usr/lib/aarch64-linux-gnu/tegra)

file(GLOB SOURCES
  "${PROJECT_SOURCE_DIR}/inc/*.h"
  "${PROJECT_SOURCE_DIR}/src/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/*.cu"
)
  
cuda_add_executable(
    ${PROJECT_NAME}
    src/main.cpp 
    ${SOURCES}
    )

target_link_libraries(${PROJECT_NAME} 
                      ${OpenCV_LIBS}
                      #${CMAKE_THREAD_LIBS_INIT} 
                      #${NVINFER_LIBRARY}
                      #${NVONNXPARSER_LIBRARY}
                      ${LIBNVINFER}
                      ${LIBNVINFER_PLUGIN}
                      ${LIBNVPARSERS}
                      ${LIBNVONNXPARSER}
                      )

#target_link_libraries(Test PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${NVONNXPARSER_LIBRARY})

