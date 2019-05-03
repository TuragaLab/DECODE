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
typedef struct {
        int xsize;
        int ysize;
        int zsize;

        float x0;
        float y0;
        float z0;
        float dz;

        float *coeff;
        float *delta_f;
} spline;

void kernel_computeDelta3D(spline *, float, float, float );
float fAt3Dj(spline *, int, int, int );

float fSpline3D(spline *, float, float, float );
void fPSF(spline *, float *, int, float, float, float, float, float, float);

spline* initSpline(float *, int, int, int, float, float, float, float);

#endif /* spline_psf_h */
