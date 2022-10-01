add_definitions(-DSIMULATED_IPU)

cmake_minimum_required(VERSION 3.10)
project(BlackScholes)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_COMPILER_IS_GNUCXX)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -ansi -Wno-deprecated")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -Wall -ansi -Wno-deprecated -march=native -mtune=native -O3")
endif()

find_package(poplar REQUIRED)

add_executable(BlackScholes BlackScholes_ipu.cpp BlackScholes_cpu.cpp BlackScholes_vertex.h)
target_link_libraries(BlackScholes poplar poputil popops poprand)

