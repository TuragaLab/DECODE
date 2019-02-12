//
//  main.cpp
//  cubicSpline
//
//  Created by Lucas Müller on 07.02.19.
//  Copyright © 2019 Lucas-Raphael Müller. All rights reserved.
//

#include <iostream>

extern "C" {
    #include "cubic_spline.h"
}

int main() {
    
    double aij[(10*10*10*64)] = {0};
    
    splineData *s = initSpline3D(aij, 10, 10, 10);
    printf("Hi this CPP. Xsize is: %d\n", s->xsize);
    
    return 0;
}
