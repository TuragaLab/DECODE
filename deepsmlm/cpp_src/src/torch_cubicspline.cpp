//
//  torch_cubicspline.cpp
//  libtorchInterface
//
//  Created by Lucas Müller on 12.02.19.
//  Copyright © 2019 Lucas-Raphael Müller. All rights reserved.
//

#include <iostream>
#include <stdio.h>
#include <tuple>

#ifndef TORCH_DIRECT
    #include <torch/extension.h>
#else
    #include <torch/torch.h>
#endif

#include "torch_cubicspline.hpp"
#include "multi_crlb.hpp"


auto init_spline_bind(torch::Tensor coeff, std::array<float, 3> ref0_ix, float dz) -> spline* {

    int xsize = static_cast<int>(coeff.size(0));
    int ysize = static_cast<int>(coeff.size(1));
    int zsize = static_cast<int>(coeff.size(2));

    float* coeff_ = coeff.data<float>();

    return initSpline(coeff_, xsize, ysize, zsize, ref0_ix[0], ref0_ix[1], ref0_ix[2], dz);
}

auto fPSF_bind(spline *sp,
                    torch::Tensor xyz, torch::Tensor phot, int img_size, std::array<float, 2> corner_coord) -> torch::Tensor {

    float img[img_size * img_size];
    std::fill_n(img, img_size * img_size, 0.0);
    
    /* Loop over all emitters */
    for (int i = 0; i < xyz.size(0); i++) {
        fPSF(sp, img, img_size, *xyz[i][0].data<float>(), 
             *xyz[i][1].data<float>(), *xyz[i][2].data<float>(), 
             corner_coord[0] + 0.5, 
             corner_coord[1] + 0.5, 
             *phot[i].data<float>());
    }
    
    torch::Tensor img_tensor = torch::empty({img_size * img_size, 1}, torch::kFloat);
    std::memcpy(img_tensor.data_ptr(), img, img_tensor.numel() * sizeof(float));
    img_tensor = img_tensor.view({1, img_size, img_size});

    return img_tensor;
}

auto fSpline_bind(spline *sp, float x, float y, float z) -> float {
    return fSpline3D(sp, x, y, z);
}

auto fPSF_d_bind(spline *sp, torch::Tensor xyz, torch::Tensor phot, torch::Tensor bg, int img_size, std::array<float, 2> corner_coord) -> std::tuple<torch::Tensor, torch::Tensor> {

    float img[img_size * img_size];
    std::fill_n(img, img_size * img_size, 0.0);

    float dudt[sp->NV_PSP * img_size * img_size];
    
    /* Loop over all emitters */
    // ToDo: Change loop.
    for (int i = 0; i < 1; i++) {
        f_derivative_PSF(sp, img, dudt, img_size, 
             *xyz[i][0].data<float>(), 
             *xyz[i][1].data<float>(), *xyz[i][2].data<float>(), 
             corner_coord[0] + 0.5, 
             corner_coord[1] + 0.5, 
             *phot[i].data<float>(),
             *bg[i].data<float>()
             );
    }
    
    torch::Tensor img_tensor = torch::empty({img_size * img_size, 1}, torch::kFloat);
    std::memcpy(img_tensor.data_ptr(), img, img_tensor.numel() * sizeof(float));
    img_tensor = img_tensor.view({1, img_size, img_size});

    torch::Tensor dpsf = torch::empty({sp->NV_PSP, img_size, img_size}, torch::kFloat);
    std::memcpy(dpsf.data_ptr(), dudt, dpsf.numel() * sizeof(float));

    return std::make_tuple(dpsf, img_tensor);
}

auto fPSF_fisher(spline *sp, torch::Tensor xyz, torch::Tensor phot, torch::Tensor bg, int img_size, std::array<float, 2> corner_coord) -> std::tuple<torch::Tensor, torch::Tensor> {
    
    int n_emitter = phot.size(0);

    float img[img_size * img_size];
    std::fill_n(img, img_size * img_size, 0.0);
    float hessian_blocked_matrix[sp->NV_PSP * n_emitter * sp->NV_PSP * n_emitter];
    std::fill_n(hessian_blocked_matrix, sp->NV_PSP * sp->NV_PSP * n_emitter * n_emitter, 0.0);

    // convert tensors to sted::vectors
    std::vector<std::array<float, 3>> xyz_;
    std::vector<float> phot_;
    std::vector<float> bg_;

    for (int i = 0; i < n_emitter; i++) {
        std::array<float, 3> xyz__ = { *xyz[i][0].data<float>(), *xyz[i][1].data<float>(), *xyz[i][2].data<float>() };
        xyz_.push_back(xyz__);
        phot_.push_back(*phot[i].data<float>());
        bg_.push_back(*bg[i].data<float>());
    }

    corner_coord[0] += 0.5;
    corner_coord[1] += 0.5;
    
    construct_multi_fisher(sp, xyz_, phot_, bg_, corner_coord, img_size, img, hessian_blocked_matrix);

    torch::Tensor hessian_tensor = torch::empty({sp->NV_PSP * n_emitter, sp->NV_PSP * n_emitter}, torch::kFloat);
    torch::Tensor img_tensor = torch::empty({img_size, img_size}, torch::kFloat);

    std::memcpy(img_tensor.data_ptr(), img, img_tensor.numel() * sizeof(float));
    std::memcpy(hessian_tensor.data_ptr(), hessian_blocked_matrix, hessian_tensor.numel() * sizeof(float));

    return std::make_tuple(hessian_tensor, img_tensor);
}

auto fPSF_crlb(spline *sp, torch::Tensor xyz, torch::Tensor phot, torch::Tensor bg, int img_size, std::array<float, 2> corner_coord) -> std::tuple<torch::Tensor, torch::Tensor> {
    
    int n_emitter = phot.size(0);

    float img[img_size * img_size];
    std::fill_n(img, img_size * img_size, 0.0);

    float crlb[sp->NV_PSP * n_emitter];
    std::fill_n(crlb, sp->NV_PSP * n_emitter, 0.0);

    // convert tensors to sted::vectors
    std::vector<std::array<float, 3>> xyz_;
    std::vector<float> phot_;
    std::vector<float> bg_;

    for (int i = 0; i < xyz.size(0); i++) {
        std::array<float, 3> xyz__ = { *xyz[i][0].data<float>(), *xyz[i][1].data<float>(), *xyz[i][2].data<float>() };
        xyz_.push_back(xyz__);
        phot_.push_back(*phot[i].data<float>());
        bg_.push_back(*bg[i].data<float>());
    }

    corner_coord[0] += 0.5;
    corner_coord[1] += 0.5;

    calc_crlb(sp, xyz_, phot_, bg_, corner_coord, img_size, img, crlb);

    torch::Tensor crlb_tensor = torch::empty({n_emitter, sp->NV_PSP}, torch::kFloat);
    torch::Tensor img_tensor = torch::empty({img_size, img_size}, torch::kFloat);

    std::memcpy(img_tensor.data_ptr(), img, img_tensor.numel() * sizeof(float));
    std::memcpy(crlb_tensor.data_ptr(), crlb, crlb_tensor.numel() * sizeof(float));

    return std::make_tuple(crlb_tensor, img_tensor);
}
