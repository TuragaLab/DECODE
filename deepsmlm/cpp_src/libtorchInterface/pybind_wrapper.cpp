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
    m.def("distribute_frames", &distribute_frames, "Function to distribute real valued emitters over the frames.");

    m.def("init_spline", &init_spline_bind, "Torch wrapper function to init a cubic spline library in C.");
    m.def("fPSF", &fPSF_bind, "Function to generate cubic spline psf image.");

    m.def("f_spline", &fSpline_bind, "Function to debug img_spline_function" );
}
