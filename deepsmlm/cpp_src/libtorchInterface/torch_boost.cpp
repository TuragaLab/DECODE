//
//  torch_boost.cpp
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

#include "torch_boost.hpp"


/**
 Function to split a tensor into vector / list of tensors acoording to an indexing split_data tensor.

 @param tensor_ tensor of size N x D  which should be splitted in rows, must be sorted(!)
 @param split_data tensor of size N
 @param bound_low first value of splitting (e.g. 0)
 @param bound_high last value of splitting
 
 @return vector of splitted tensor_
 */
auto split_tensor(torch::Tensor tensor_, torch::Tensor split_data, int bound_low, int bound_high) -> std::vector<torch::Tensor> {
    
    std::vector<torch::Tensor> tensor_split;
    
    int ix_start_frame = 0;
    int ix_end_frame = 0;
    
    for (int i = bound_low; i <= bound_high; i++) {
        
        int j = ix_start_frame;
        
        while ((j < split_data.size(0)) && (i >= static_cast<int>(*(split_data[j]).data<float>()))) {
            j++;
        }
        ix_end_frame = j;
        
        if (ix_start_frame != ix_end_frame){
            tensor_split.push_back(tensor_.slice(0, ix_start_frame, ix_end_frame));
        } else {
            tensor_split.push_back(torch::zeros({0, 6}));
        }
        
        ix_start_frame = ix_end_frame;
    }
    return tensor_split;
}

auto distribute_frames(torch::Tensor t0, torch::Tensor ontime, torch::Tensor xyz, torch::Tensor phot, torch::Tensor id) -> std::vector<torch::Tensor> {
    
    auto frame_start = torch::floor(t0);
    auto te = t0 + ontime;
    auto frame_last = torch::ceil(te);
    auto frame_dur = (frame_last - frame_start);
    
    long num_emitter = xyz.size(0);
    int num_rows = static_cast<int>(*frame_dur.sum().data<float>());
    
    auto _xyz = torch::zeros({num_rows, xyz.size(1)});
    auto _phot = torch::zeros({num_rows});
    auto _frame_ix = torch::zeros_like(_phot);
    auto _id = torch::zeros_like(_frame_ix);
    
    int c = 0;
    for (int i = 0; i < num_emitter; i++) {
        for (int j = 0; j < static_cast<int>(*(frame_dur[i]).data<float>()); j++) {
            for (int k = 0; k < 3; k++){
                _xyz[c][k] = xyz[i][k];
            }
            _frame_ix[c] = frame_start[i] + j;
            _id[c] = id[i];
            
            auto ontime_on_frame = torch::min(te[i], (_frame_ix[c] + 1)) - torch::max(t0[i], _frame_ix[c]);
            _phot[c] = ontime_on_frame / ontime[i] * phot[i];
            
            c++;
        }
    }
    
    std::vector<torch::Tensor> emitter;
    emitter.push_back(_xyz);
    emitter.push_back(_phot);
    emitter.push_back(_frame_ix);
    emitter.push_back(_id);
    return emitter;
    
}
