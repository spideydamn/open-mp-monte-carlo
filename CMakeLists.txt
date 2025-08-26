cmake_minimum_required(VERSION 3.28)
project(se_comp_arch24_omp_spideydamn)

OPTION (USE_OpenMP "Use OpenMP" ON)
IF(USE_OpenMP)
    FIND_PACKAGE(OpenMP)
    IF(OPENMP_FOUND)
        SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    ENDIF()
ENDIF()

set(CMAKE_CXX_STANDARD 20)

include_directories(.)

add_executable(se_comp_arch24_omp_spideydamn
        main.cpp)