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

auto initSplineTorch(torch::Tensor coeff) -> splineData*;
auto imgSplineTorch(splineData *spline_data) -> torch::Tensor;

#endif /* torch_cubicspline_hpp */
