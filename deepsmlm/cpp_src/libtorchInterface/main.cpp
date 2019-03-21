//
//  main.cpp
//  libtorchInterface
//
//  Created by Lucas Müller on 07.02.19.
//  Copyright © 2019 Lucas-Raphael Müller. All rights reserved.
//

//#include <fenv.h>  // DEBUGGING
//#pragma STDC FENV_ACCESS ON

#include <iostream>
#include <torch/torch.h>

#include "torch_boost.hpp"
#include "torch_cubicspline.hpp"

extern "C" {
//    #include "cubic_spline.h"  // this shall be replaced by a better path specification
    #include "lib/spline_psf.h"
}

void dummy_function(torch::Tensor x) {
    x[0] = 1;
}

int main() {

    std::array<int, 2> img_size = {32, 32};
    std::array<float, 3> ref0_ix = {0, 0, 150};
    std::array<float, 2> pu = {-0.5, -0.5};
    torch::Tensor xyz = torch::zeros({10000, 3}, torch::kFloat);
    torch::Tensor phot = torch::ones({10000}, torch::kFloat);
    xyz[0][0] = 10.1;
    xyz[0][1] = 4.7;

    
//    auto coeff_t = torch::arange(1297920, torch::kDouble).view({26, 26, 30, 64}) / 1e6;
    auto coeff_t = torch::arange(1297920, torch::kFloat);
    float *coeff = coeff_t.data<float>();

    spline *sp = initSpline(coeff, 26, 26, 30, 0, 0, 0, 1);
    float f = fSpline3D(sp, 10.2, 11.4, 12.6);
    
    std::cout << f << std::endl;
    auto x = fPSF_bind(sp, xyz, phot, 32, pu);
    
    std::cout << x << std::endl;
    
    return 0;
}
