cmake_minimum_required(VERSION 3.10)

# create project
set(PROJECT_NAME sgm_nano)
project(${PROJECT_NAME})

find_package( CUDA REQUIRED )
find_package( OpenCV REQUIRED )

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -lineinfo
    -gencode=arch=compute_53,code=sm_53
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

target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} ${CUDA_LIBRARIES})
