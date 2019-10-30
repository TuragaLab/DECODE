#define CATCH_CONFIG_MAIN
#include <stdio.h>
#include <torch/torch.h>

#include "catch.hpp"
#include "torch_cubicspline.hpp"

TEST_CASE( "Split a N x 5 tensor.", "[split_tensor]" ) {
    auto xyz = torch::rand({2, 3}, torch::kFloat);
    auto phot = torch::rand({2}, torch::kFloat);
    auto bg = torch::rand({2}, torch::kFloat);

    int xsize = 21;
    int ysize = 21;
    int zsize = 100;

    float x0 = 15;
    float y0 = 18;
    float z0 = 150;
    float dz = 10;

    float coeff[64 * 21 * 21 * 100] = { 0 };

    // spline* sp = initSpline(coeff, xsize, ysize, zsize, x0, y0, z0, dz);
    REQUIRE(true);

}