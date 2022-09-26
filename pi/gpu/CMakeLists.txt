cmake_minimum_required(VERSION 3.23)
project(pi_gpu CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(pi_gpu pi.cu timer.h number_with_commas.h)

set_target_properties(pi_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
