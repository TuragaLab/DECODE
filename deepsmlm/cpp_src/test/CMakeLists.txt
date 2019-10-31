cmake_minimum_required(VERSION 3.15)
project(tests)

# find_package(Torch REQUIRED)

# adds the preprocessor DEBUG when CMAKE_BUILD_TYPE=Debug
add_compile_options("$<$<CONFIG:DEBUG>:-DDEBUG>")

set(CATCH_INCLUDE_DIR ../lib/)
add_library(Catch2 INTERFACE)
target_include_directories(Catch2 INTERFACE ${CATCH_INCLUDE_DIR})

add_library(spline_psf STATIC ../src/spline_psf.c) #include/spline_psf.h)
target_include_directories(spline_psf PUBLIC ../include)
target_compile_features(spline_psf PUBLIC c_std_99)

add_executable(test_out test_spline.cpp)
target_link_libraries(test_out Catch2 spline_psf)
target_include_directories(test_out PUBLIC ../include)
target_compile_features(test_out PUBLIC cxx_std_11)

# add_library(torch_cubic STATIC ../src/torch_cubicspline.cpp) #include/spline_psf.h)
# target_include_directories(torch_cubic PUBLIC ../include)
# target_link_libraries(torch_cubic "${TORCH_LIBRARIES}")
# target_compile_definitions(torch_cubic PRIVATE TORCH_DIRECT=TRUE)
# target_compile_features(torch_cubic PUBLIC cxx_std_11)

# add_executable(test_out test_torchcubicspline.cpp)
# target_link_libraries(test_out Catch2 "${TORCH_LIBRARIES}" torch_cubic spline_psf)
# target_compile_features(test_out PUBLIC cxx_std_11)