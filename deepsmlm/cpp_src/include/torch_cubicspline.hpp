//
//  torch_cubicspline.hpp
//  libtorchInterface
//
//  Created by Lucas Müller on 12.02.19.
//  Copyright © 2019 Lucas-Raphael Müller. All rights reserved.
//

#ifndef torch_cubicspline_hpp
#define torch_cubicspline_hpp

extern "C" {
    #include "spline_psf.h"
}

auto init_spline_bind(torch::Tensor coeff, std::array<float, 3> ref0_ix, float dz)->spline*;
auto fPSF_bind(spline *sp, torch::Tensor xyz, torch::Tensor phot, int img_size, std::array<float, 2> corner_coord)->torch::Tensor;
auto fSpline_bind(spline *sp, float x, float y, float z) -> float;
auto fPSF_d_bind(spline *sp, torch::Tensor xyz, torch::Tensor phot, torch::Tensor bg, int img_size, std::array<float, 2> corner_coord) -> std::tuple<torch::Tensor, torch::Tensor>;
auto fPSF_fisher(spline *sp, torch::Tensor xyz, torch::Tensor phot, torch::Tensor bg, int img_size, std::array<float, 2> corner_coord) -> std::tuple<torch::Tensor, torch::Tensor>;
auto fPSF_crlb(spline *sp, torch::Tensor xyz, torch::Tensor phot, torch::Tensor bg, int img_size, std::array<float, 2> corner_coord) -> std::tuple<torch::Tensor, torch::Tensor>;


#endif /* torch_cubicspline_hpp */
