//
//  main.c
//  cubicSplineLibrary
//
//  Created by Lucas Müller on 07.02.19.
//  Copyright © 2019 Lucas-Raphael Müller. All rights reserved.
//

#include <stdio.h>
#include "cubic_spline.h"

int main() {
    
    double aij[(10*10*10*64)] = {0};
    
    splineData *s = initSpline3D(aij, 0, 0, 0);
    printf("Hi. Xsize is: %f\n", s->aij[0]);
    

    return 0;
}
