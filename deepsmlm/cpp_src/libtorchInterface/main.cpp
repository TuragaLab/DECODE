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
    
    //    std::cout << std::get<1>(x) << std::endl;
    
    //    auto x = split_tensor(trial_tensor, frames, 0, -1);
    //    std::cout << trial_tensor << std::endl;
    //    std::cout << x << std::endl;
    
    std::array<int, 2> img_size = {32, 32};
    std::array<int, 3> ref0_ix = {14, 14, 150};
    torch::Tensor xyz = 32 * torch::rand({10000, 3}, torch::kDouble);
    torch::Tensor phot = torch::rand({10000}, torch::kDouble);
    
    auto *y = initSplineTorch(torch::randn({26, 26, 300, 64}, torch::kDouble), ref0_ix);
    auto t = imgSplineTorch(y, xyz, phot, img_size);
    
//    std::cout << t << std::endl;
    
    return 0;
}

