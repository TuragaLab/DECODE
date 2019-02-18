from setuptools import setup
import sys
from distutils.core import setup, Extension
from torch.utils.cpp_extension import CppExtension, BuildExtension

static_libraries = ['cubic_spline']
static_lib_dir = 'lib'
libraries = ['']
library_dirs = ['']

if sys.platform == 'win32':

    libraries.extend(static_libraries)
    library_dirs.append(static_lib_dir)
    extra_objects = []
    extra_compile_args = []

elif sys.platform == 'darwin':
    extra_compile_args = ['-g', '-stdlib=libc++', '-std=c++11']
    extra_objects = ['{}/lib{}.a'.format(static_lib_dir, l) for l in static_libraries]

elif sys.platform == 'linux':
    extra_compile_args = []
    extra_objects = ['{}/lib{}.a'.format(static_lib_dir, l) for l in static_libraries]

setup(
    name='torch_cpp',
    ext_modules=[
        CppExtension(
            name='torch_cpp',
            sources=['pybind_wrapper.cpp', 'torch_boost.cpp', 'torch_cubicspline.cpp'],
            extra_compile_args=extra_compile_args,
            extra_objects=extra_objects)

        ],
    cmdclass={
        'build_ext': BuildExtension
    })


"""
call:

$ CC=clang CXX=clang++ NO_CUDA=1 python setup.py install

Compile cubic_spline and link statically (Linux: gcc, macOS: clang):

$ gcc -fPIC -g -c -Wall lib/cubic_spline.c -o lib/cubic_spline.o
$ ar rcs lib/libcubic_spline.a lib/cubic_spline.o

$ clang -o lib/cubic_spline.o -c -O3 -Wall -I/usr/local/include -fPIC lib/cubic_spline.c
$ ar rcs lib/libcubic_spline.a lib/cubic_spline.o

"""
