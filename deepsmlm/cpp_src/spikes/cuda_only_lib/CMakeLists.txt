cmake_minimum_required(VERSION 3.14)
# find_package(CUDA REQUIRED)
enable_language(CUDA)

project(dummy_cuda_lib LANGUAGES CUDA CXX)

add_library(dummy_cuda_lib SHARED spline_psf_gpu.cu)
target_compile_features(dummy_cuda_lib SHARED cuda_std_11)