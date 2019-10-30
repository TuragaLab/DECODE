#define CATCH_CONFIG_MAIN

#include <stdio.h>
#include "catch.hpp"

extern "C" {
    #include "spline_psf.h"
}

TEST_CASE( "Initialise Spline",  "[spline-init]" ) {
    int xsize = 21;
    int ysize = 21;
    int zsize = 100;

    float x0 = 15;
    float y0 = 18;
    float z0 = 150;
    float dz = 10;

    float coeff[64 * 21 * 21 * 100] = { 0 };

    spline* sp = initSpline(coeff, xsize, ysize, zsize, x0, y0, z0, dz);

    REQUIRE(1 == 1);
}