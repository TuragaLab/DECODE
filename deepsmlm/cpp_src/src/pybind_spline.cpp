#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <math.h>
#include <stdbool.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// #include <cuda_runtime.h>

#include "spline_psf_gpu.cuh"

namespace py = pybind11;

struct PSFWrapper {

    spline *psf;

    PSFWrapper(int xsize, int ysize, int zsize, py::array_t<float> coeff) {
        psf = d_spline_init(xsize, ysize, zsize, coeff.data());
    }

    auto forward_psf(py::array_t<float> x, py::array_t<float> y, py::array_t<float> z, py::array_t<float> phot) -> void {
        int n = x.size();

        compute_rois_h(psf, n, x.data(), y.data(), z.data(), phot.data());
    }

};

PYBIND11_MODULE(spline_psf_cuda, m) {
    py::class_<PSFWrapper>(m, "PSFWrapper")
        .def(py::init<int, int, int, py::array_t<float>>())
        .def("forward_psf", &PSFWrapper::forward_psf);
}