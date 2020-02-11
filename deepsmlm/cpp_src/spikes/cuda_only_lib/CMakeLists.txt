cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(dummy_cuda_lib LANGUAGES CUDA CXX)

add_library(dummy_cuda_lib SHARED spline_psf_gpu.cu)
set_target_properties(dummy_exec PROPERTIES CUDA_SEPARABLE_COMPILATION ON)  # sometime off works, idk
target_compile_features(dummy_cuda_lib PUBLIC cxx_std_11)
target_compile_options(dummy_cuda_lib PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
                       -rdc=true
                       --gpu-architecture=sm_50
                       --expt-relaxed-constexpr
                       >)