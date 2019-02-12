//
//  torch_boost.hpp
//  libtorchInterface
//
//  Created by Lucas Müller on 12.02.19.
//  Copyright © 2019 Lucas-Raphael Müller. All rights reserved.
//

#ifndef torch_boost_hpp
#define torch_boost_hpp

#include <stdio.h>

auto split_tensor(torch::Tensor tensor_, torch::Tensor split_data, int bound_low=0, int bound_high=-1) -> std::vector<torch::Tensor>;

#endif /* torch_boost_hpp */
 
