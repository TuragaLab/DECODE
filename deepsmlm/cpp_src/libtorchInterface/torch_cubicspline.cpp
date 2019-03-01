//
//  torch_cubicspline.cpp
//  libtorchInterface
//
//  Created by Lucas Müller on 12.02.19.
//  Copyright © 2019 Lucas-Raphael Müller. All rights reserved.
//

#include <iostream>

#ifndef TORCH_DIRECT
    #include <torch/extension.h>
#else
    #include <torch/torch.h>
#endif

#include "torch_cubicspline.hpp"

/**
 C++ wrapper function to initialise 3D spline.

 @param coeff           tensor (dtype: torch::kDouble) 4D
 @param ref0_ix         std::array where the reference point of the psf is written into (i.e. the pseudo-midpoint)
 @param pos_up_left     upper left corner coordinate of the image. internally 0.5 will be added because spline references center of px
 
 @return Pointer to struct splineData (i.e. splineData*)
 */
auto initSplineTorch(torch::Tensor coeff, const std::array<int, 3> ref0_ix, const std::array<double, 2> pos_up_left) -> splineData* {

    const int xsize = static_cast<int>(coeff.size(0));
    const int ysize = static_cast<int>(coeff.size(1));
    const int zsize = static_cast<int>(coeff.size(2));

    double* coeff_ = coeff.data<double>();

    return initSpline3D(coeff_, xsize, ysize, zsize,
                        ref0_ix[0], ref0_ix[1], ref0_ix[2],
                        pos_up_left[0] + 0.5, pos_up_left[1] + 0.5);

}

/**
 C++ wrapper function to call C imgSpline function

 @param spline_data     pointer to splineData structure
 @param xyz             torch tensor N x 3 with xyz positions of the fluorophor (double tensor)
 @param img_size        array descrbing the image size in pixels
 
 @return img            torch::Tensor of size 1 x H x W
 */
auto imgSplineTorch(splineData *spline_data, torch::Tensor xyz, torch::Tensor phot, std::array<int, 2> img_size) -> torch::Tensor {

    double img[img_size[0] * img_size[1]];
    std::fill_n(img, img_size[0] * img_size[1], 0);
    
    /* Loop over all emitters */
    for (int i = 0; i < xyz.size(0); i++) {
        imgSpline3D(spline_data, img, img_size[0], img_size[1],
                    *xyz[i][2].data<double>(), *xyz[i][1].data<double>(), *xyz[i][0].data<double>(), *phot[i].data<double>());
    }

    torch::Tensor img_tensor = torch::empty({img_size[0] * img_size[1], 1}, torch::kDouble);
    std::memcpy(img_tensor.data_ptr(), img, img_tensor.numel() * sizeof(double));
    img_tensor = img_tensor.view({1, img_size[0], img_size[1]});

    return img_tensor;
}
