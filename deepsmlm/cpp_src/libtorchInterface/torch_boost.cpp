//
//  torch_boost.cpp
//  libtorchInterface
//
//  Created by Lucas Müller on 12.02.19.
//  Copyright © 2019 Lucas-Raphael Müller. All rights reserved.
//

#include <iostream>
#include <torch/torch.h>
#include "torch_boost.hpp"


/**
 Function to split a tensor into vector / list of tensors acoording to an indexing split_data tensor.

 @param tensor_ tensor of size N x D  which should be splitted in rows, must be sorted(!)
 @param split_data tensor of size N
 @param bound_low first value of splitting (e.g. 0). if (-1) then it is the min value of split_data
 @param bound_high last value of splitting. if (-1) then it is the max value of split_data
 
 @return vector of splitted tensor_
 */
auto split_tensor(torch::Tensor tensor_, torch::Tensor split_data, int bound_low, int bound_high) -> std::vector<torch::Tensor> {
    
//    auto sort_ = torch::sort(split_data);
//    split_data = std::get<0>(sort_);

    
    if (bound_low == -1){
        bound_low = static_cast<int>(*(split_data.min()).data<float>());
    }
    if (bound_high == -1){
        bound_high = static_cast<int>(*(split_data.max()).data<float>());
    }
    
    //    int num_frames = bound_low - bound_high + 1;
    //    std::cout << emitter << std::endl;
    
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
        
        //        std::cout << em_p_frame.back() << std::endl;
        ix_start_frame = ix_end_frame;
    }
    return tensor_split;
}

int main() {
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
    
    return 0;
}
