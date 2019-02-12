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

extern "C" {
    #include "cubic_spline.h"  // this shall be replaced by a better path specification
}

/**
 C++ Wrapper function to initialise 3D spline.

 @param coeff tensor (dtype: torch::kDouble) 4D
 @return Pointer to struct splineData (i.e. splineData*)
 */
auto initSplineTorch(torch::Tensor coeff) -> splineData* {
    
    const int xsize = static_cast<int>(coeff.size(0));
    const int ysize = static_cast<int>(coeff.size(1));
    const int zsize = static_cast<int>(coeff.size(2));
    
//    double coeff_[xsize * ysize * zsize];
    double* coeff_ = coeff.data<double>();
    
    return initSpline3D(coeff_, xsize, ysize, zsize);
    
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
    
    return 0;
}

