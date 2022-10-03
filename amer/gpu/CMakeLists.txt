cmake_minimum_required(VERSION 3.23)
project(American CUDA)

set(CMAKE_CUDA_STANDARD 14)
include_directories(Common)

# COMPILE AND LINK
add_executable(American American.cu)
set_target_properties(
        American
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)