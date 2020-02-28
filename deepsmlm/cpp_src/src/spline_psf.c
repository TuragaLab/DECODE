//
//  spline_psf.c
//  libtorchInterface
//
//  Created by Lucas Müller on 15.03.19.
//  Copyright © 2019 Lucas-Raphael Müller. All rights reserved.
//

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fenv.h>

#include "spline_psf.h"

// Internal Declarations
void kernel_computeDelta3D(spline *sp, float x_delta, float y_delta, float z_delta,
                           float *delta_f, float *delta_dxf, float *delta_dyf, float *delta_dzf);

void kernel_DerivativeSpline(spline *sp, const int xc, const int yc, int zc, const float *theta, float *dudt, float *model,
                             const float *delta_f, const float *delta_dxf, const float *delta_dyf, const float *delta_dzf,
                             const bool add_bg);

float fAt3Dj(spline *, int xc, int yc, int zc, float phot, const float *delta_f);

float fSpline3D(spline *sp, float xc, float yc, float zc);

void kernel_roi(spline *sp, float *rois, int roi_ix, int npx, int npy, float xc, float yc, float zc, float phot);

void kernel_drv_roi(spline *sp, float *rois, float *drv_rois, int roi_ix, int npx, int npy,
                    float xc, float yc, float zc, float phot, float bg, bool add_bg);

void roi_accumulator(float *frames, int frame_size_x, int frame_size_y, int n_frames, const float *rois,
                     int n_rois, const int *frame_ix, const int *x0, const int *y0, int roi_size_x, int roi_size_y);


// Definitions
spline *initSpline(const float *coeff, const int xsize, int const ysize, const int zsize) {

    spline *sp;
    sp = (spline *) malloc(sizeof(spline));

    sp->xsize = xsize;
    sp->ysize = ysize;
    sp->zsize = zsize;

    sp->roi_out_eps = 1e-10;
    sp->roi_out_deriv_eps = 0.0;
    sp->n_par = 5; // number of parameters

    int tsize = xsize * ysize * zsize * 64;
    sp->coeff = (float *) malloc(sizeof(float) * tsize);

    // Copy spline coefficients
    for (int i = 0; i < tsize; i++) {
        sp->coeff[i] = coeff[i];
    }

    return sp;
}

void kernel_computeDelta3D(spline *sp, const float x_delta, const float y_delta, const float z_delta,
                           float *delta_f, float *delta_dxf, float *delta_dyf, float *delta_dzf) {

    int i, j, k;
    float cx, cy, cz;

    cz = 1.0;
    for (i = 0; i < 4; i++) {
        cy = 1.0;
        for (j = 0; j < 4; j++) {
            cx = 1.0;
            for (k = 0; k < 4; k++) {
                delta_f[i * 16 + j * 4 + k] = cz * cy * cx;
                if (k < 3) {
                    delta_dxf[i * 16 + j * 4 + k + 1] = ((float) k + 1) * cz * cy * cx;
                }
                if (j < 3) {
                    delta_dyf[i * 16 + (j + 1) * 4 + k] = ((float) j + 1) * cz * cy * cx;
                }
                if (i < 3) {
                    delta_dzf[(i + 1) * 16 + j * 4 + k] = ((float) i + 1) * cz * cy * cx;
                }
                cx = cx * x_delta;
            }
            cy = cy * y_delta;
        }
        cz = cz * z_delta;
    }
}

void
kernel_DerivativeSpline(spline *sp, const int xc, const int yc, int zc, const float *theta, float *dudt, float *model, 
                        const float *delta_f, const float *delta_dxf, const float *delta_dyf, const float *delta_dzf, bool add_bg) {
    int i;
    float temp = 0;
    // make sure that dudt is initialised to zero

    if ((xc < 0) || (xc > sp->xsize - 1) || (yc < 0) || (yc > sp->ysize - 1)) {
        dudt[0] = sp->roi_out_deriv_eps;
        dudt[1] = sp->roi_out_deriv_eps;
        dudt[2] = sp->roi_out_deriv_eps;
        dudt[3] = sp->roi_out_deriv_eps;
        dudt[4] = sp->roi_out_deriv_eps;

        if (add_bg) {
            *model = sp->roi_out_eps * theta[2] + theta[3];
        } else {
            *model = sp->roi_out_eps * theta[2];
        }
        return;
    }

    zc = fmax(zc, 0);
    zc = fmin(zc, sp->zsize - 1);

    for (i = 0; i < 64; i++) {
        temp += delta_f[i] *
                sp->coeff[i * (sp->xsize * sp->ysize * sp->zsize) + zc * (sp->xsize * sp->ysize) + yc * sp->xsize + xc];
        dudt[0] += delta_dxf[i] *
                   sp->coeff[i * (sp->xsize * sp->ysize * sp->zsize) + zc * (sp->xsize * sp->ysize) + yc * sp->xsize +
                             xc];
        dudt[1] += delta_dyf[i] *
                   sp->coeff[i * (sp->xsize * sp->ysize * sp->zsize) + zc * (sp->xsize * sp->ysize) + yc * sp->xsize +
                             xc];
        dudt[4] += delta_dzf[i] *
                   sp->coeff[i * (sp->xsize * sp->ysize * sp->zsize) + zc * (sp->xsize * sp->ysize) + yc * sp->xsize +
                             xc];
    }
    dudt[0] *= -1 * theta[2];
    dudt[1] *= -1 * theta[2];
    dudt[4] *= theta[2];
    dudt[2] = temp;
    dudt[3] = 1;
    if (add_bg) {
        *model = theta[3] + theta[2] * temp;  // saves time
    } else {
        *model = theta[2] * temp;  // saves time
    }
}

float fAt3Dj(spline *sp, const int xc, const int yc, int zc, const float phot, const float *delta_f) {

    // If the lateral position is outside the calibration, return epsilon value
    if ((xc < 0) || (xc > sp->xsize - 1) || (yc < 0) || (yc > sp->ysize - 1)) {
        return sp->roi_out_eps;
    }

    zc = fmax(zc, 0);
    zc = fmin(zc, sp->zsize - 1);

    float fv = 0;

    for (int i = 0; i < 64; i++) {
        fv += delta_f[i] *
              sp->coeff[i * (sp->xsize * sp->ysize * sp->zsize) + zc * (sp->xsize * sp->ysize) + yc * sp->xsize + xc];
    }

    fv *= phot;
    return fv;
}

float fSpline3D(spline *sp, float xc, float yc, float zc) {

    int x0, y0, z0;
    float x_delta, y_delta, z_delta;

    x0 = (int) floor(xc);
    x_delta = xc - x0;

    y0 = (int) floor(yc);
    y_delta = yc - y0;

    z0 = (int) floor(zc);
    z_delta = zc - z0;

    float delta_f[64] = { 0 };
    float dxf[64] = { 0 };
    float dyf[64] = { 0 };
    float dzf[64] = { 0 };
    kernel_computeDelta3D(sp, x_delta, y_delta, z_delta, delta_f, dxf, dyf, dzf);

    float f = fAt3Dj(sp, x0, y0, z0, 1, delta_f);
    return f;
}



void kernel_roi(spline *sp, float *rois, const int roi_ix, const int npx, const int npy, const float xc, const float yc,
                const float zc, const float phot) {

    /* Compute delta. Will be the same for all following px */
    int x0 = (int) floor(xc);
    float x_delta = xc - x0;

    int y0 = (int) floor(yc);
    float y_delta = yc - y0;

    int z0 = (int) floor(zc);
    float z_delta = zc - z0;

    float delta_f[64] = { 0 };
    float dxf[64] = { 0 };
    float dyf[64] = { 0 };
    float dzf[64] = { 0 };

    kernel_computeDelta3D(sp, x_delta, y_delta, z_delta, delta_f, dxf, dyf, dzf);

    /* loop through all pixels */
    for (int i = 0; i < npx; i++) {
        for (int j = 0; j < npx; j++) {
            rois[roi_ix * npx * npy + i * npy + j] += fAt3Dj(sp, x0 + i, y0 + j, z0, phot, delta_f);
        }
    }
}

void kernel_drv_roi(spline *sp, float *rois, float *drv_rois, const int roi_ix, const int npx, const int npy,
                    const float xc, const float yc, const float zc, const float phot, const float bg, const bool add_bg) {

    int x0, y0, z0; // px indices
    float model;  // model value
    float theta[sp->n_par];

    /* Compute delta. Will be the same for all following px */
    x0 = (int) floor(xc);
    float x_delta = xc - x0;

    y0 = (int) floor(yc);
    float y_delta = yc - y0;

    z0 = (int) floor(zc);
    float z_delta = zc - z0;

    // thetas [x y phot bg z]
    theta[0] = xc;
    theta[1] = yc;
    theta[2] = phot;
    theta[3] = bg;
    theta[4] = zc;

    float delta_f[64] = { 0 };
    float dxf[64] = { 0 };
    float dyf[64] = { 0 };
    float dzf[64] = { 0 };

    kernel_computeDelta3D(sp, x_delta, y_delta, z_delta, delta_f, dxf, dyf, dzf);

    /* loop through all pixels */
    for (int i = 0; i < npx; i++) {
        for (int j = 0; j < npy; j++) {
            float deriv_px[5] = {0};
            kernel_DerivativeSpline(sp, x0 + i, y0 + j, z0, theta, deriv_px, &model, delta_f, dxf, dyf, dzf, add_bg);

            // set model image (almost for free when calc. gradient)
            rois[roi_ix * npx * npy + i * npy + j] += model; // unlike in kernel_roi, background and photons is already in theta.

            // store gradient
            for (int k = 0; k < sp->n_par; k++) {
                drv_rois[roi_ix * sp->n_par * npx * npy + k * npx * npy + i * npy + j] = deriv_px[k];
            }

        }
    }

}

void forward_rois(spline *sp, float *rois, const int n_rois, const int npx, const int npy,
                  const float *xc, const float *yc, const float *zc, const float *phot) {

    // init rois
    for (int i = 0; i < n_rois * npx * npy; i++) {
        rois[i] = 0.0;
    }

    for (int i = 0; i < n_rois; i++) {
        kernel_roi(sp, rois, i, npx, npy, xc[i], yc[i], zc[i], phot[i]);
    }

}

void forward_drv_rois(spline *sp, float *rois, float *drv_rois, const int n_rois, const int npx, const int npy,
                      const float *xc, const float *yc, const float *zc, const float *phot, const float *bg, const bool add_bg) {

    // init rois and pixelwise derivatives
    for (int i = 0; i < n_rois * npx * npy; i++) {
        rois[i] = 0;
    }

    for (int i = 0; i < n_rois * sp->n_par * npx * npy; i++) {
        drv_rois[i] = 0;
    }

    for (int i = 0; i < n_rois; i++) {
        kernel_drv_roi(sp, rois, drv_rois, i, npx, npy, xc[i], yc[i], zc[i], phot[i], bg[i], add_bg);
    }
}

void roi_accumulator(float *frames, const int frame_size_x, const int frame_size_y, const int n_frames,
                     const float *rois, const int n_rois, const int *frame_ix, const int *x0, const int *y0,
                     const int roi_size_x, const int roi_size_y) {

    // loop over rois
    for (int r = 0; r < n_rois; r++) {
        // loop over all roi px
        for (int i = 0; i < roi_size_x; i++) {
            for (int j = 0; j < roi_size_y; j++) {
                int ii = x0[r] + i;
                int jj = y0[r] + j;

                if ((frame_ix[r] < 0) || (frame_ix[r] >= n_frames)) {  // if frame ix is outside
                    continue;
                }

                if ((ii < 0) || (jj < 0) || (ii >= frame_size_x) ||
                    (jj >= frame_size_y)) {  // if outside frame throw away
                    continue;
                }

                frames[frame_ix[r] * frame_size_x * frame_size_y + ii * frame_size_y + jj] += rois[
                        r * roi_size_x * roi_size_y + i * roi_size_y + j];
            }
        }
    }

}

void forward_frames(spline *sp, float *frames, const int frame_size_x, const int frame_size_y, const int n_frames,
                    const int n_rois, const int roi_size_x, const int roi_size_y,
                    const int *frame_ix, const float *xr0, const float *yr0, const float *z0, const int *x_ix,
                    const int *y_ix, const float *phot) {

    // malloc rois
    int roi_px = n_rois * roi_size_x * roi_size_y;
    float *rois = (float *) malloc(roi_px * sizeof(float));

    // init frames
    for (int i = 0; i < frame_size_x * frame_size_y * n_frames; i++) {
        frames[i] = 0.0;
    }

    // forward rois and accumulate
    forward_rois(sp, rois, n_rois, roi_size_x, roi_size_y, xr0, yr0, z0, phot);
    roi_accumulator(frames, frame_size_x, frame_size_y, n_frames, rois, n_rois, frame_ix, x_ix, y_ix, roi_size_x,
                    roi_size_y);

    // free rois
    free(rois);

}