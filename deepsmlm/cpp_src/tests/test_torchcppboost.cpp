//
//  test_torchcppboost.cpp
//  libtorchInterface
//
//  Created by Lucas Müller on 25.04.19.
//  Copyright © 2019 Lucas-Raphael Müller. All rights reserved.
//

#include <stdio.h>
#include <torch/torch.h>

#include "catch.hpp"

#include "torch_boost.hpp"


TEST_CASE( "Split a N x 5 tensor.", "[split_tensor]" ) {
    
    // arbitrary matrix which should be splitted in list
    torch::Tensor matrix = torch::randn({4, 5});
    // tensor which specifies which index is assigned to a row
    torch::Tensor ix = torch::arange(1, 5);
    
    int lower_bound = 0;
    int upper_bound = 5;
    
    std::vector<torch::Tensor> outcome = split_tensor(matrix, ix, lower_bound, upper_bound);
    
    REQUIRE(outcome.size() == 6); // list of 6 elements
    REQUIRE(outcome[0].size(0) == 0); // empty tensor
    REQUIRE(outcome[1].size(0) == 1); // one element tensor
    REQUIRE(outcome[5].size(0) == 0); // empty tensor
}
