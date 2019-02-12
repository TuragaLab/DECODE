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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("split_tensor", &split_tensor, "Function to split tensor as described by another tensor.");
}
