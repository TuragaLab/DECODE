//
//  spline_psf.h
//  libtorchInterface
//
//  Created by Lucas Müller on 15.03.19.
//  Copyright © 2019 Lucas-Raphael Müller. All rights reserved.
//

#ifndef spline_psf_h
#define spline_psf_h

/* debugging */
#define TESTING 0
#define VERBOSE 0

/* Spline Structure */
/**
 * @brief defines the cubic spline
 * 
 **/
typedef struct {
        int xsize;  // size of the spline in x
        int ysize;  // size of the spline in y
        int zsize;  // size of the spline in z

        float x0;  // reference in x, y, z
        float y0;
        float z0;
        float dz;  // delta between the z slices

        float *coeff;  // coefficients
        float *delta_f;  // internal helper
} spline;

void kernel_computeDelta3D(spline *, float, float, float );
float fAt3Dj(spline *, int, int, int );

float fSpline3D(spline *, float, float, float );
void fPSF(spline *, float *, int, float, float, float, float, float, float);

spline* initSpline(float *, int, int, int, float, float, float, float);

#endif /* spline_psf_h */
