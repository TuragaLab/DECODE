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
#define spline_npar 5  // number of spline parameters as PSEUDO-Const. expr

#include <stdbool.h>

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

        float roi_out_eps;  // epsilon value outside the roi
        float roi_out_deriv_eps; // epsilon value of derivative values outside the roi
        int NV_PSP; // number of parameters

        bool add_bg_to_model;

        float *coeff;  // coefficients
        float *delta_f;  // internal helper
        float *delta_dxf;
        float *delta_dyf;
        float *delta_dzf;
} spline;

void kernel_computeDelta3D(spline *, float, float, float );
void kernel_DerivativeSpline(spline *, int xc, int yc, int zc, float *theta, float *dudt, float *model);
float fAt3Dj(spline *, int, int, int );

float fSpline3D(spline *, float, float, float );
void fPSF(spline *, float *, int, float, float, float, float, float, float);
void f_derivative_PSF(spline *sp, float *img, float *dudt, int npx, float xc, float yc, float zc, float corner_x0, float corner_y0, float phot, float bg);
void f_derivative_aggregate(spline *sp, float *dudt_px, float *dudt_agg, int npx, int npy);

spline* initSpline(float *, int, int, int, float, float, float, float);

#endif /* spline_psf_h */
