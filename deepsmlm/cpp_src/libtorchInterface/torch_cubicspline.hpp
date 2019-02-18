//
//  torch_cubicspline.hpp
//  libtorchInterface
//
//  Created by Lucas Müller on 12.02.19.
//  Copyright © 2019 Lucas-Raphael Müller. All rights reserved.
//

#ifndef torch_cubicspline_hpp
#define torch_cubicspline_hpp

#include <stdio.h>

extern "C" {
    #include "lib/cubic_spline.h"
}

auto initSplineTorch(torch::Tensor coeff, std::array<int, 3>, std::array<double, 2>) -> splineData*;
auto imgSplineTorch(splineData *spline_data, torch::Tensor xyz, torch::Tensor phot, std::array<int, 2> img_size) -> torch::Tensor;

#endif /* torch_cubicspline_hpp */
