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

namespace splinecpu {
    extern "C" {
        #include "spline_psf.h"
    }
}

namespace py = pybind11;

struct PSFWrapper {

    spline *psf;

    PSFWrapper(int xsize, int ysize, int zsize, 
               py::array_t<float, py::array::f_style | py::array::forcecast> coeff) {
        psf = d_spline_init(xsize, ysize, zsize, coeff.data());
    }

    auto forward_psf(py::array_t<float, py::array::c_style | py::array::forcecast> x, 
                     py::array_t<float, py::array::c_style | py::array::forcecast> y, 
                     py::array_t<float, py::array::c_style | py::array::forcecast> z, 
                     py::array_t<float, py::array::c_style | py::array::forcecast> phot) -> py::array_t<float> {
        int n = x.size();

        py::array_t<float> h_roi(n * 26 * 26);

        compute_rois_h(psf, n, x.data(), y.data(), z.data(), phot.data(), h_roi.mutable_data());

        return h_roi;
    }

};

struct PSFWrapperCPU {

    splinecpu::spline *psf;

    PSFWrapperCPU(int xsize, int ysize, int zsize,
                  py::array_t<float, py::array::f_style | py::array::forcecast> coeff) {
                      psf = splinecpu::initSpline(coeff.data(), xsize, ysize, zsize, 0.0, 0.0, 0.0, 0.0);
                  }

    auto forward_psf(py::array_t<float, py::array::c_style | py::array::forcecast> x, 
                     py::array_t<float, py::array::c_style | py::array::forcecast> y, 
                     py::array_t<float, py::array::c_style | py::array::forcecast> z, 
                     py::array_t<float, py::array::c_style | py::array::forcecast> phot) -> py::array_t<float> {

        int n = x.size();

        py::array_t<float> h_roi(n * 26 * 26);
        splinecpu::fPSF(psf, h_roi.mutable_data(), 26, x.data()[0], y.data()[0], z.data()[0], 0.0, 0.0, phot.data()[0]);

        return h_roi;        
    }

};

PYBIND11_MODULE(spline_psf_cuda, m) {
    py::class_<PSFWrapper>(m, "PSFWrapper")
        .def(py::init<int, int, int, py::array_t<float>>())
        .def("forward_psf", &PSFWrapper::forward_psf);

    py::class_<PSFWrapperCPU>(m, "PSFWrapperCPU")
        .def(py::init<int, int, int, py::array_t<float>>())
        .def("forward_psf", &PSFWrapperCPU::forward_psf);
}