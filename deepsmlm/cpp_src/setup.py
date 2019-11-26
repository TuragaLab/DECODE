from setuptools import setup
import sys
from distutils.core import setup, Extension
from torch.utils.cpp_extension import CppExtension, BuildExtension

# https://stackoverflow.com/questions/4597228/how-to-statically-link-a-library-when-compiling-a-python-module-extension

static_libraries = ['spline_psf', 'multi_crlb']
static_lib_dir = 'build'
libraries = [] #['spline_psf', 'multi_crlb']
library_dirs = [] #['build']

if sys.platform == 'win32':

    libraries.extend(static_libraries)
    library_dirs.append(static_lib_dir)
    extra_objects = []
    extra_compile_args = []

elif sys.platform == 'darwin':
    extra_compile_args = ['-O3', '-g', '-stdlib=libc++', '-std=c++11']
    extra_linker_args = []
    extra_objects = ['{}/lib{}.a'.format(static_lib_dir, l) for l in static_libraries]

elif sys.platform == 'linux':
    extra_compile_args = ['-O3']
    extra_linker_args = []
    extra_objects = ['{}/lib{}.a'.format(static_lib_dir, l) for l in static_libraries]

setup(
    name='torch_cpp',
    ext_modules=[
        CppExtension(
            name='torch_cpp',
            sources=['src/pybind_wrapper.cpp', 'src/torch_boost.cpp', 'src/torch_cubicspline.cpp'],
            include_dirs=['include'],
            libraries=libraries,
            library_dirs=library_dirs,
            extra_include_paths=['include'],
            extra_compile_args=extra_compile_args,
            extra_objects=extra_objects)

        ],
    cmdclass={
        'build_ext': BuildExtension
    })


"""
call:

macOS:
$ CFLAGS='-stdlib=libc++' CC=clang CXX=clang++ NO_CUDA=1 python setup.py clean --all install

Linux:
$ python setup.py clean --all install

Linux Server
$ CXX=gcc-4.9 CC=gcc-4.9 python setup.py clean --all install

Linux without CUDA:
$ NO_CUDA=1 python setup.py clean --all install


Compile cubic_spline and link statically (Linux: gcc, macOS: clang):

@linux workstation
$ gcc -fPIC -g -c -Wall src/spline_psf.c -o build/spline_psf.o -Iinclude
$ ar rcs build/libspline_psf.a build/spline_psf.o

@gpu6
$ gcc-4.9 -fPIC -g -c -std=c99 -Wall src/spline_psf.c -o build/spline_psf.o -Iinclude
$ ar rcs build/libspline_psf.a build/spline_psf.o

@mac
$ clang -o build/spline_psf.o -c -O3 -Wall -I/usr/local/include -Iinclude -fPIC src/spline_psf.c
$ ar rcs build/libspline_psf.a build/spline_psf.o
"""
