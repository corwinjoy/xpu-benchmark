cmake_minimum_required(VERSION 3.23)
project(BlackScholes)

set(CMAKE_CUDA_STANDARD 23)
include_directories(Common)

find_package(CUDA REQUIRED)

set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)


# COMPILE AND LINK
cuda_add_executable(BlackScholes BlackScholes_gold.cpp BlackScholes.cu)
