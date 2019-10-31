cmake_minimum_required(VERSION 3.15)
project(deepsmlm_cpp)

# adds the preprocessor DEBUG when CMAKE_BUILD_TYPE=Debug
add_compile_options("$<$<CONFIG:DEBUG>:-DDEBUG>")

add_library(spline_psf STATIC src/spline_psf.c)
target_include_directories(spline_psf PUBLIC include)
target_compile_features(spline_psf PUBLIC c_std_99)
set_property(TARGET spline_psf PROPERTY POSITION_INDEPENDENT_CODE ON)  # otherwise I get an error when compiling the extension

add_library(multi_crlb STATIC src/multi_crlb.cpp)
target_include_directories(multi_crlb PUBLIC include lib/eigen)
target_compile_features(multi_crlb PUBLIC cxx_std_11)
target_link_libraries (multi_crlb spline_psf)
set_property(TARGET multi_crlb PROPERTY POSITION_INDEPENDENT_CODE ON)  # otherwise I get an error when compiling the extension