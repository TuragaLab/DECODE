#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <math.h>
#include <stdbool.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// #include <cuda_runtime.h>

// ToDo: Namespace here did not work for a reason I don't yet understand.
#include "spline_psf_gpu.cuh"


namespace spc {
    extern "C" {
        #include "spline_psf.h"
    }
}

namespace spg = spline_psf_gpu;
namespace py = pybind11;

template <typename T>
class PSFWrapperBase {

    protected:
    
        T *psf;
        int roi_size_x;
        int roi_size_y;

        PSFWrapperBase(int rx, int ry) : roi_size_x(rx), roi_size_y(ry) { }

};

class PSFWrapperCUDA : public PSFWrapperBase<spg::spline> {

    public:

        explicit PSFWrapperCUDA(int coeff_xsize, int coeff_ysize, int coeff_zsize, int roi_size_x_, int roi_size_y_,
            py::array_t<float, py::array::f_style | py::array::forcecast> coeff) : PSFWrapperBase{roi_size_x_, roi_size_y_} {

                psf = spg::d_spline_init(coeff_xsize, coeff_ysize, coeff_zsize, coeff.data());

            }

        auto forward_psf(py::array_t<float, py::array::c_style | py::array::forcecast> x, 
                        py::array_t<float, py::array::c_style | py::array::forcecast> y, 
                        py::array_t<float, py::array::c_style | py::array::forcecast> z, 
                        py::array_t<float, py::array::c_style | py::array::forcecast> phot) -> py::array_t<float> {


            int n = x.size();  // number of ROIs
            py::array_t<float> h_rois(n * roi_size_x * roi_size_y);

            spg::forward_rois_host2host(psf, h_rois.mutable_data(), n, roi_size_x, roi_size_y, x.data(), y.data(), z.data(), phot.data());

            return h_rois;
        }

};

class PSFWrapperCPU : public PSFWrapperBase<spc::spline> {

    public:

        explicit PSFWrapperCPU(int coeff_xsize, int coeff_ysize, int coeff_zsize, int roi_size_x_, int roi_size_y_,
            py::array_t<float, py::array::f_style | py::array::forcecast> coeff) : PSFWrapperBase{roi_size_x_, roi_size_y_} {

                psf = spc::initSpline(coeff.data(), coeff_xsize, coeff_ysize, coeff_zsize, 0.0, 0.0, 0.0, 0.0);

            }

        auto forward_psf(py::array_t<float, py::array::c_style | py::array::forcecast> x, 
                        py::array_t<float, py::array::c_style | py::array::forcecast> y, 
                        py::array_t<float, py::array::c_style | py::array::forcecast> z, 
                        py::array_t<float, py::array::c_style | py::array::forcecast> phot) -> py::array_t<float> {

            int n = x.size();
            py::array_t<float> h_rois(n * roi_size_x * roi_size_y);

            spc::fPSF(psf, h_rois.mutable_data(), 26, x.data()[0], y.data()[0], z.data()[0], 0.0, 0.0, phot.data()[0]);

            return h_rois;
        }

};

PYBIND11_MODULE(spline_psf_cuda, m) {
    py::class_<PSFWrapperCUDA>(m, "PSFWrapperCUDA")
        .def(py::init<int, int, int, int, int, py::array_t<float>>())
        .def("forward_psf", &PSFWrapperCUDA::forward_psf);

    py::class_<PSFWrapperCPU>(m, "PSFWrapperCPU")
        .def(py::init<int, int, int, int, int, py::array_t<float>>())
        .def("forward_psf", &PSFWrapperCPU::forward_psf);
}