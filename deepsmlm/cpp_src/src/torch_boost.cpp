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
 Get range indices which equal to a value (element).

 @param vals tensor which are indices (must be sorted)
 @param element single value
 @param last_ix (input & return) index of last accessed item in vals
 @param ix_low (return) index of first item
 @param ix_up (return) index of last item
 */
void get_frame_range(torch::Tensor vals, int element, int& last_ix, int& ix_low, int& ix_up) {
    
    // first item to be accessed (other items have been looped through earlier)
    int j = last_ix + 1;
    // look out for first index of value 'element'
    while ((j <= vals.size(0) - 1) && (static_cast<int>(*(vals[j]).data<float>()) != element)) {
        j++;
    }
    // if outside range, return indicater
    if (j >= vals.size(0)) {
        ix_low = -1;
        ix_up = -1;
        return;
    } else {
        ix_low = j;
    }
    
    // search how many items have the value
    int k = ix_low;
    while ((k <= vals.size(0) - 1) && (static_cast<int>(*(vals[k]).data<float>()) == element)) {
        ix_up = k;
        k++;
    }
    last_ix = ix_up;
    return;
}


/**
 Function to split a tensor in a recotr of rows of the aforementioned tensor

 @param tensor_ tensor which should be splitted in rows as specified by the tensor split_data
 @param split_data indices for splitting
 @param bound_low lower bound of range
 @param bound_high high bound
 @return vector of tensors, each element being a tensor
 */
auto split_tensor(torch::Tensor tensor_, torch::Tensor split_data, int bound_low, int bound_high) -> std::vector<torch::Tensor> {
    
    int tensor_cols = tensor_.size(1);
    std::vector<torch::Tensor> tensor_split;
    
    // initiailise last accesed item in split_data
    int last_ix = -1;
    
    for (int i = bound_low; i <= bound_high; i++) {
        
        // call get_frame_range, outputs are ix_low, ix_up
        int ix_low, ix_up;
        get_frame_range(split_data, i, last_ix, ix_low, ix_up);
        
        // if value is not found in tensor, push empty
        if (ix_up == -1 || ix_low == -1) {
            tensor_split.push_back(torch::zeros({0, tensor_cols}));
        } else {
            tensor_split.push_back(tensor_.slice(0, ix_low, ix_up + 1));  // behaviour like Python, so excluding end index
        }
        
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
