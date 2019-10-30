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
    REQUIRE(outcome[2].size(0) == 1); // one element tensor
    REQUIRE(outcome[5].size(0) == 0); // empty tensor
    
    
    /* Test whether adjacent frames split works*/
    
    matrix = torch::randn({3, 5});
    ix = torch::zeros(3);
    ix[0] = -1;
    ix[1] = 0;
    ix[2] = 1;
    matrix[0][4] = ix[0];
    matrix[1][4] = ix[1];
    matrix[2][4] = ix[2];
    
    outcome = split_tensor(matrix, ix, -1, 1);
    REQUIRE(outcome[0][0][4].item().toInt() == -1);
    REQUIRE(outcome[1][0][4].item().toInt() == 0);
    REQUIRE(outcome[2][0][4].item().toInt() == 1);
    
    matrix = torch::randn({5, 5});
    ix = torch::zeros(5);
    ix[0] = -1;
    ix[1] = -1;
    ix[2] = 0;
    ix[3] = 1;
    ix[4] = 1;
    for (int i=0; i < ix.size(0); i++) {
        matrix[i][4] = ix[i];
    }
    
    outcome = split_tensor(matrix, ix, -1, 1);
    REQUIRE(outcome[0][0][4].item().toInt() == -1);
    REQUIRE(outcome[0][1][4].item().toInt() == -1);
    REQUIRE(outcome[1][0][4].item().toInt() == 0);
    REQUIRE(outcome[2][0][4].item().toInt() == 1);
    REQUIRE(outcome[2][1][4].item().toInt() == 1);
    
    outcome = split_tensor(matrix, ix, 0, 0);
    REQUIRE(outcome.size() == 1);
    REQUIRE(outcome[0][0][4].item().toInt() == 0);
    
}
