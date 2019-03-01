//
//  main.cpp
//  libtorchInterface
//
//  Created by Lucas Müller on 07.02.19.
//  Copyright © 2019 Lucas-Raphael Müller. All rights reserved.
//

#include <iostream>
#include <torch/torch.h>
#include "torch_boost.hpp"
#include "torch_cubicspline.hpp"

extern "C" {
    #include "cubic_spline.h"  // this shall be replaced by a better path specification
}
    

int main() {
    
    int x0, y0, z0;
    double xc, yc, zc;
    double x_delta, y_delta, z_delta;
    
    xc = 14.1;
    yc = 4.8;
    zc = 10;
    
    for (int i = 0; i < 10; i++) {
        x0 = (int)(floor(xc));
        y0 = (int)(floor(yc));
        z0 = (int)(floor(zc));
        
        x_delta = xc - (double)x0;
        y_delta = yc - (double)y0;
        z_delta = zc - (double)z0;
    }

    
    std::array<int, 2> img_size = {32, 32};
    std::array<int, 3> ref0_ix = {0, 0, 150};
    std::array<double, 2> pu = {-0.5, -0.5};
    torch::Tensor xyz = torch::zeros({1, 3}, torch::kDouble);
    torch::Tensor phot = torch::ones({1}, torch::kDouble);
    xyz[0][0] = 10.1;
    xyz[0][1] = 4.7;
    
    
    auto *y = initSplineTorch(torch::randn({26, 26, 300, 64}, torch::kDouble), ref0_ix, pu);
    auto t = imgSplineTorch(y, xyz, phot, img_size);
    
    std::cout << t << std::endl;
    
    return 0;
}

