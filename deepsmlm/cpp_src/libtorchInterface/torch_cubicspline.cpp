//
//  torch_cubicspline.cpp
//  libtorchInterface
//
//  Created by Lucas Müller on 12.02.19.
//  Copyright © 2019 Lucas-Raphael Müller. All rights reserved.
//

#include <iostream>
#include <torch/torch.h>
#include "torch_cubicspline.hpp"

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

auto imgSplineTorch(splineData *spline_data) -> torch::Tensor {

    double img[(spline_data->xsize * spline_data->ysize)];

    imgSpline3D(spline_data, img, spline_data->xsize, spline_data->ysize, 0.0, 0.0, 0.0);

    torch::Tensor t = torch::empty({spline_data->xsize*spline_data->ysize,1}, torch::kDouble);
    std::memcpy(t.data_ptr(), img, t.numel() * sizeof(double));
    t = t.view({spline_data->xsize, spline_data->ysize});

    return t;
}
