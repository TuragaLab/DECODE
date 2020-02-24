//
//  pybind_wrapper.cpp
//  torch_cpp
//
//  Created by Lucas Müller on 12.02.19.
//  Copyright © 2019 Lucas-Raphael Müller. All rights reserved.
//

#include <stdio.h>
#include <torch/extension.h>

#include "torch_boost.hpp"
#include "torch_cubicspline.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<spline> sp(m, "splineData");
    sp
      .def(py::init<>());

    m.def("split_tensor", &split_tensor, "Function to split tensor as described by another tensor.");

    m.def("initSpline", &init_spline_bind, "Torch wrapper function to init a cubic spline library in C.");
    m.def("fPSF", &fPSF_bind, "Function to generate cubic spline psf image.");

    m.def("f_spline", &fSpline_bind, "Function to debug img_spline_function, i.e. getting back single values." );
    m.def("f_spline_d", &fPSF_d_bind, "Returns derivatives of the PSF.");
    m.def("f_spline_fisher", &fPSF_fisher, "Construct the fisher matrix.");
    m.def("f_spline_crlb", &fPSF_crlb, "Calculate CRLB and image.");
}
