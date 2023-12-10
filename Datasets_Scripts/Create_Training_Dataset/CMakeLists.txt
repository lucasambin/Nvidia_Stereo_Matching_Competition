cmake_minimum_required(VERSION 3.18)
# create project
set(PROJECT_NAME sgm_nano)
project(${PROJECT_NAME})

find_package( CUDA REQUIRED )
find_package( OpenCV REQUIRED )

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -lineinfo
    # NVIDIA GTX 950M (2 Gb)
    -gencode=arch=compute_50,code=sm_50
    # NVIDIA Jetson Nano Maxwell (2 Gb)
    #-gencode=arch=compute_53,code=sm_53
    )

include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/inc ${CMAKE_CURRENT_SOURCE_DIR}/src )

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
                      ${OpenCV_LIBS})
