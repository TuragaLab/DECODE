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

    torch::Tensor trial_tensor = torch::randint(40, {5, 6});  // does not work because the matrix must be sorted!
    torch::Tensor frames = torch::ones(5);
    frames[0] = 3.;
    frames[1] = 3.;
    frames[2] = 5.;
    frames[3] = 7.;
    frames[4] = 7.;
    
    auto x = split_tensor(trial_tensor, frames, 0, -1);
    
//
//    double aij[(10*10*10*64)] = {0};
//    splineData *s = initSpline3D(aij, 5, 5, 5);
//    printf("Hi this CPP. Xsize is: %d\n", s->xsize);
//
//    auto *y = initSplineTorch(torch::randn({20, 25, 250, 64}, torch::kDouble));
//    std::cout << y->xsize << std::endl;
//    std::cout << y->aij[5] << std::endl;
    
    
//    cubic_spline_img();
    
    /* Test the function split_frames */
    
    torch::Tensor trial_tensor = torch::randint(40, {5, 6});  // does not work because the matrix must be sorted!
    torch::Tensor frames = torch::ones(5);
    frames[0] = 3.;
    frames[1] = 3.;
    frames[2] = 5.;
    frames[3] = 7.;
    frames[4] = 7.;
    
    //    std::cout << std::get<1>(x) << std::endl;
    
    //    auto x = split_tensor(trial_tensor, frames, 0, -1);
    //    std::cout << trial_tensor << std::endl;
    //    std::cout << x << std::endl;
    
    auto *y = initSplineTorch(torch::randn({10, 10, 10, 64}, torch::kDouble));
    auto t = imgSplineTorch(y);
    
    std::cout << t << std::endl;
    //    std::cout << y->aij[5] << std::endl;
    
    return 0;
    
    return 0;
}

