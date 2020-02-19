#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <math.h>
#include <stdexcept>
#include <stdbool.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#if CUDA_ENABLED
    #include "spline_psf_gpu.cuh"
    namespace spg = spline_psf_gpu;
#endif


namespace spc {
    extern "C" {
        #include "spline_psf.h"
    }
}

namespace py = pybind11;

template <typename T>
class PSFWrapperBase {

    protected:
    
        T *psf;
        const int roi_size_x;
        const int roi_size_y;
        int frame_size_x;
        int frame_size_y;

        PSFWrapperBase(int rx, int ry) : roi_size_x(rx), roi_size_y(ry) { }
        PSFWrapperBase(int rx, int ry, int fx, int fy) : roi_size_x(rx), roi_size_y(ry), frame_size_x(fx), frame_size_y(fy) { }

};

#if CUDA_ENABLED
    class PSFWrapperCUDA : public PSFWrapperBase<spg::spline> {

        public:

            explicit PSFWrapperCUDA(int coeff_xsize, int coeff_ysize, int coeff_zsize, int roi_size_x_, int roi_size_y_,
                py::array_t<float, py::array::f_style | py::array::forcecast> coeff) : PSFWrapperBase{roi_size_x_, roi_size_y_} {

                    psf = spg::d_spline_init(coeff.data(), coeff_xsize, coeff_ysize, coeff_zsize);

                }

            auto forward_rois(py::array_t<float, py::array::c_style | py::array::forcecast> x, 
                            py::array_t<float, py::array::c_style | py::array::forcecast> y, 
                            py::array_t<float, py::array::c_style | py::array::forcecast> z, 
                            py::array_t<float, py::array::c_style | py::array::forcecast> phot) -> py::array_t<float> {


                int n = x.size();  // number of ROIs
                py::array_t<float> h_rois(n * roi_size_x * roi_size_y);

                spg::forward_rois_host2host(psf, h_rois.mutable_data(), n, roi_size_x, roi_size_y, x.data(), y.data(), z.data(), phot.data());

                return h_rois;
            }

            auto forward_frames(const int fx, const int fy,
                            py::array_t<int, py::array::c_style | py::array::forcecast> frame_ix,
                            const int n_frames,
                            py::array_t<float, py::array::c_style | py::array::forcecast> xr,
                            py::array_t<float, py::array::c_style | py::array::forcecast> yr,
                            py::array_t<float, py::array::c_style | py::array::forcecast> z,
                            py::array_t<int, py::array::c_style | py::array::forcecast> x_ix,
                            py::array_t<int, py::array::c_style | py::array::forcecast> y_ix,
                            py::array_t<float, py::array::c_style | py::array::forcecast> phot) -> py::array_t<float> {

            frame_size_x = fx;
            frame_size_y = fy;
            const int n_emitters = xr.size();
            py::array_t<float> h_frames(n_frames * frame_size_x * frame_size_y);

            spg::forward_frames_host2host(psf, h_frames.mutable_data(), frame_size_x, frame_size_y, n_frames, n_emitters, roi_size_x, roi_size_y, 
                frame_ix.data(), xr.data(), yr.data(), z.data(), x_ix.data(), y_ix.data(), phot.data());

            return h_frames;
        }


    };
#else
    class PSFWrapperCUDA : public PSFWrapperBase<float> {

        public:
        
        explicit PSFWrapperCUDA(int coeff_xsize, int coeff_ysize, int coeff_zsize, int roi_size_x_, int roi_size_y_,
                py::array_t<float, py::array::f_style | py::array::forcecast> coeff) : PSFWrapperBase{roi_size_x_, roi_size_y_} {
                    throw_cuda_error();
                }

        auto forward_rois() -> void {
            throw_cuda_error();
        }
        auto forward_frames() -> void {
            throw_cuda_error();
        }
        
        auto throw_cuda_error() -> void {
            throw std::runtime_error("Extension not compiled with CUDA enabled. \nPlease use CPU version or recompile with CUDA if CUDA capable device is available on your machine.");
        }

    };
#endif

class PSFWrapperCPU : public PSFWrapperBase<spc::spline> {

    public:

        explicit PSFWrapperCPU(int coeff_xsize, int coeff_ysize, int coeff_zsize, int roi_size_x_, int roi_size_y_,
            py::array_t<float, py::array::f_style | py::array::forcecast> coeff) : PSFWrapperBase{roi_size_x_, roi_size_y_} {

                psf = spc::initSpline(coeff.data(), coeff_xsize, coeff_ysize, coeff_zsize);

            }

        auto forward_rois(py::array_t<float, py::array::c_style | py::array::forcecast> x, 
                          py::array_t<float, py::array::c_style | py::array::forcecast> y, 
                          py::array_t<float, py::array::c_style | py::array::forcecast> z, 
                          py::array_t<float, py::array::c_style | py::array::forcecast> phot) -> py::array_t<float> {

            const int n = x.size();
            py::array_t<float> h_rois(n * roi_size_x * roi_size_y);

            if (roi_size_x != roi_size_y) {
                throw std::invalid_argument("ROI size must be equal currently.");
            }

            spc::forward_rois(psf, h_rois.mutable_data(), n, roi_size_x, roi_size_y, x.data(), y.data(), z.data(), phot.data());

            return h_rois;
        }

        auto forward_frames(const int fx, const int fy,
                            py::array_t<int, py::array::c_style | py::array::forcecast> frame_ix,
                            const int n_frames,
                            py::array_t<float, py::array::c_style | py::array::forcecast> xr,
                            py::array_t<float, py::array::c_style | py::array::forcecast> yr,
                            py::array_t<float, py::array::c_style | py::array::forcecast> z,
                            py::array_t<int, py::array::c_style | py::array::forcecast> x_ix,
                            py::array_t<int, py::array::c_style | py::array::forcecast> y_ix,
                            py::array_t<float, py::array::c_style | py::array::forcecast> phot) -> py::array_t<float> {

            frame_size_x = fx;
            frame_size_y = fy;
            const int n_emitters = xr.size();
            py::array_t<float> h_frames(n_frames * frame_size_x * frame_size_y);

            spc::forward_frames(psf, h_frames.mutable_data(), frame_size_x, frame_size_y, n_frames, n_emitters, roi_size_x, roi_size_y, 
                frame_ix.data(), xr.data(), yr.data(), z.data(), x_ix.data(), y_ix.data(), phot.data());

            return h_frames;
        }

};

PYBIND11_MODULE(spline_psf_cuda, m) {
    py::class_<PSFWrapperCUDA>(m, "PSFWrapperCUDA")
        .def(py::init<int, int, int, int, int, py::array_t<float>>())
        .def("forward_rois", &PSFWrapperCUDA::forward_rois)
        .def("forward_frames", &PSFWrapperCUDA::forward_frames);

    py::class_<PSFWrapperCPU>(m, "PSFWrapperCPU")
        .def(py::init<int, int, int, int, int, py::array_t<float>>())
        .def("forward_rois", &PSFWrapperCPU::forward_rois)
        .def("forward_frames", &PSFWrapperCPU::forward_frames);
}