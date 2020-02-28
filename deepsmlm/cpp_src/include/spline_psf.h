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

/// Spline struct holding all relevant attributes to it.
///
///
typedef struct {
    int xsize;  // size of the spline in x
    int ysize;  // size of the spline in y
    int zsize;  // size of the spline in z

    float roi_out_eps;  // epsilon value outside the roi
    float roi_out_deriv_eps; // epsilon value of derivative values outside the roi
    int n_par; // number of parameters

    float *coeff;  // coefficients
} spline;

/// Initialises spline struct
///
/// \param [in] coeff: spline coefficients
/// \param [in] xsize: size of the coefficients in x
/// \param [in] ysize: size of the coefficients in y
/// \param [in] zsize: size of the coefficients in z
/// \return spline*: pointer to spline struct
spline *initSpline(const float *coeff, int xsize, int ysize, int zsize);



void forward_rois(spline *sp, float *rois, int n_rois, int npx, int npy,
                  const float *xc, const float *yc, const float *zc, const float *phot);

void forward_drv_rois(spline *sp, float *rois, float *drv_rois, int n_rois, int npx, int npy,
                      const float *xc, const float *yc, const float *zc, const float *phot, const float *bg,
                      const bool add_bg);

void forward_frames(spline *sp, float *frames, int frame_size_x, int frame_size_y, int n_frames,
                    int n_rois, int roi_size_x, int roi_size_y,
                    const int *frame_ix, const float *xr0, const float *yr0, const float *z0, const int *x_ix,
                    const int *y_ix, const float *phot);


#endif /* spline_psf_h */
