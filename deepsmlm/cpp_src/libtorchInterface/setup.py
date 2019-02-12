from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='torch_cpp',
    ext_modules=[
        CppExtension(
            name='torch_cpp',
            sources=['pybind_wrapper.cpp', 'torch_boost.cpp'],
            extra_compile_args=['-g', '-stdlib=libc++', '-std=c++11']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })


# call: $ CC=clang CXX=clang++ NO_CUDA=1 python setup.py install
