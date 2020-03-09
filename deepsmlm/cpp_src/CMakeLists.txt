cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(splininger LANGUAGES CXX C)

include(CheckLanguage)
check_language(CUDA)

find_package(PythonInterp 3.6 REQUIRED)
find_package(PythonLibs 3.6 REQUIRED)
find_package(CUDA)

if(CUDA_FOUND)
   enable_language(CUDA)
   # find_package(CUDA REQUIRED)

   set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --gpu-architecture=sm_75")
   add_compile_definitions(CUDA_ENABLED)

else()
   message(WARNING "No CUDA compiler found. \nYou may need to specify its path like 'export CUDACXX=/usr/local/cuda/bin/nvcc'")

endif()


add_library(spline_psf_cpu_impl SHARED src/spline_psf.c)
target_include_directories(spline_psf_cpu_impl PUBLIC include)
target_compile_features(spline_psf_cpu_impl PUBLIC c_std_99)
set_property(TARGET spline_psf_cpu_impl PROPERTY POSITION_INDEPENDENT_CODE ON)

add_subdirectory(lib/pybind11)

if(CMAKE_CUDA_COMPILER)
   add_library(spline_psf_cu_impl SHARED src/spline_psf_gpu.cu)
   target_include_directories(spline_psf_cu_impl PUBLIC include)
   target_compile_features(spline_psf_cu_impl PUBLIC cxx_std_11)
   set_target_properties(spline_psf_cu_impl PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
   target_compile_options(spline_psf_cu_impl PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
                        -rdc=true
                     #    --gpu-architecture=sm_50
                        --expt-relaxed-constexpr
                        >)

   link_directories(/usr/local/cuda/lib64)
   include_directories("${CUDA_INCLUDE_DIRS}")

   pybind11_add_module(spline_psf_cuda src/pybind_spline.cpp)
   target_link_libraries(spline_psf_cuda PRIVATE spline_psf_cu_impl spline_psf_cpu_impl)

else()  # NO CUDA

   pybind11_add_module(spline_psf_cuda src/pybind_spline.cpp)
   target_link_libraries(spline_psf_cuda PRIVATE spline_psf_cpu_impl)

endif()
