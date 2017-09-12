cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(cpp_libdmet)

# This must be run using a conda environment
if("$ENV{CONDA_PREFIX}" STREQUAL "")
    message(FATAL_ERROR "This must be run within the MolSSI SSS conda prefix. "
                        "Delete the build directory and try again")
endif()

# Find the LAWrap for mkl_blas/lapack
find_package(LAWrap CONFIG REQUIRED)

# CMake will sometimes find the conda python version
# (rather the the python inside the sss environment)
set(PYTHON_EXECUTABLE $ENV{CONDA_PREFIX}/bin/python3)

# Find the pybind11 in the conda path
set(PYBIND11_CPP_STANDARD -std=c++11)
find_package(pybind11 CONFIG REQUIRED
             PATHS $ENV{CONDA_PREFIX}
             NO_DEFAULT_PATH)

message(STATUS "Found pybind11: ${pybind11_CONFIG}")

# Creates a python module named "module_name"
pybind11_add_module(libdmet MODULE libdmet.cpp)
target_link_libraries(libdmet mkl_intel_lp64 mkl_sequential mkl_core)