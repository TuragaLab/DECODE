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

spline* initSpline(const float *coeff, const int xsize, int const ysize, const int zsize);
void kernel_roi(spline *sp, float *img, const int roi_ix, const int npx, const int npy, const float xc, const float yc, const float zc, const float phot);
void forward_rois(spline *sp, float *rois, const int n_rois, const int npx, const int npy, const float *xc, const float *yc, const float *zc, const float *phot);
void forward_frames(spline *sp, float *frames, const int frame_size_x, const int frame_size_y, const int n_rois, const int roi_size_x, const int roi_size_y,
                    const int *frame_ix, const float *xr0, const float *yr0, const float *z0, const int *x_ix, const int *y_ix, const float *phot);

void f_derivative_PSF(spline *sp, float *img, float *dudt, int npx, float xc, float yc, float zc, float phot, float bg);
void f_derivative_aggregate(spline *sp, float *dudt_px, float *dudt_agg, int npx, int npy);

#endif /* spline_psf_h */
