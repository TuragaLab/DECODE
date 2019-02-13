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
    py::class_<splineData> spline_data(m, "splineData");
    spline_data
      .def(py::init<>());

    m.def("split_tensor", &split_tensor, "Function to split tensor as described by another tensor.");
    m.def("init_spline", &initSplineTorch, "Torch wrapper function to init a cubic spline library in C.");
    m.def("img_spline", &imgSplineTorch, "Function to generate cubic spline psf image.");
}
