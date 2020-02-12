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

void kernel_computeDelta3D(spline *sp, float x_delta, float y_delta, float z_delta) {

    int i,j,k;
    float cx,cy,cz;

    cz = 1.0;
    for(i=0;i<4;i++){
        cy = 1.0;
        for(j=0;j<4;j++){
            cx = 1.0;
            for(k=0;k<4;k++){
                sp->delta_f[i*16+j*4+k] = cz * cy * cx;
                if(k<3){
					sp->delta_dxf[i*16+j*4+k+1] = ((float)k+1) * cz * cy * cx;
				}
				if(j<3){
					sp->delta_dyf[i*16+(j+1)*4+k] = ((float)j+1) * cz * cy * cx;
				}
				if(i<3){
					sp->delta_dzf[(i+1)*16+j*4+k] = ((float)i+1) * cz * cy * cx;
				}
                cx = cx * x_delta;
            }
            cy = cy * y_delta;
        }
        cz= cz * z_delta;
    }
}

void kernel_DerivativeSpline(spline *sp, int xc, int yc, int zc, float *theta, float *dudt, float *model)
{
    int i;
    float temp = 0;
    // make sure that dudt is initialised to zero

    if ((xc < 0) || (xc > sp->xsize-1) || (yc < 0) || (yc > sp->ysize-1)) {
        dudt[0] = sp->roi_out_deriv_eps;
        dudt[1] = sp->roi_out_deriv_eps;
        dudt[2] = sp->roi_out_deriv_eps;
        dudt[3] = sp->roi_out_deriv_eps;
        dudt[4] = sp->roi_out_deriv_eps;
        if (sp->add_bg_to_model) {
            *model = sp->roi_out_eps * theta[2] + theta[3];
        } else {
            *model = sp->roi_out_eps * theta[2]; // theta[3] + theta[2] * temp;  // saves time
        }
        return;
    }

    xc = fmax(xc, 0);
    xc = fmin(xc, sp->xsize - 1);

    yc = fmax(yc, 0);
    yc = fmin(yc, sp->ysize - 1);

    zc = fmax(zc, 0);
    zc = fmin(zc, sp->zsize - 1);

    for (i = 0; i < 64; i++)
    {
        temp += sp->delta_f[i] * sp->coeff[i * (sp->xsize * sp->ysize * sp->zsize) + zc * (sp->xsize * sp->ysize) + yc * sp->xsize + xc];
        dudt[0] += sp->delta_dxf[i] * sp->coeff[i * (sp->xsize * sp->ysize * sp->zsize) + zc * (sp->xsize * sp->ysize) + yc * sp->xsize + xc];
        dudt[1] += sp->delta_dyf[i] * sp->coeff[i * (sp->xsize * sp->ysize * sp->zsize) + zc * (sp->xsize * sp->ysize) + yc * sp->xsize + xc];
        dudt[4] += sp->delta_dzf[i] * sp->coeff[i * (sp->xsize * sp->ysize * sp->zsize) + zc * (sp->xsize * sp->ysize) + yc * sp->xsize + xc];
    }
    dudt[0] *= -1 * theta[2];
    dudt[1] *= -1 * theta[2];
    dudt[4] *= theta[2];
    dudt[2] = temp;
    dudt[3] = 1;
    if (sp->add_bg_to_model) {
        *model = theta[3] + theta[2] * temp;  // saves time
    } else {
        *model = theta[2] * temp;  // saves time
    }
}

float fAt3Dj(spline *sp, int xc, int yc, int zc) {
    float fv = 0;
    
    // Throw 0 for outside points (only x,y considered).
    if ((xc < 0) || (xc > sp->xsize-1) || (yc < 0) || (yc > sp->ysize-1)) {
        return sp->roi_out_eps;
    }

    xc = fmax(xc,0);
    xc = fmin(xc,sp->xsize-1);

    yc = fmax(yc,0);
    yc = fmin(yc,sp->ysize-1);

    zc = fmax(zc,0);
    zc = fmin(zc,sp->zsize-1);

    for (int i=0; i < 64; i++) {
        fv += sp->delta_f[i] * sp->coeff[i * (sp->xsize * sp->ysize * sp->zsize) + zc * (sp->xsize * sp->ysize) + yc * sp->xsize + xc];
    }
    return fv;
}

float fSpline3D(spline *sp, float xc, float yc, float zc) {

    /** Query points. Note that the delta is the same for all points on the grid. **/
    int x0,y0,z0;
    float x_delta,y_delta,z_delta;

    x0 = (int)floor(xc);
    x_delta = xc - x0;

    y0 = (int)floor(yc);
    y_delta = yc - y0;

    z0 = (int)floor(zc);
    z_delta = zc - z0;

    kernel_computeDelta3D(sp, x_delta, y_delta, z_delta);
    
    float f = fAt3Dj(sp, x0, y0, z0);
    return f;
}


void fPSF(spline *sp, float *img, int npx, float xc, float yc, float zc, float corner_x0, float corner_y0, float phot) {
    
    int x0, y0, z0;
    float x_delta,y_delta,z_delta;

    /* coordinate of upper left corner */
    // xc = 0 - xc + sp->x0 - corner_x0;
    // yc = 0 - yc + sp->y0 - corner_y0;
    // zc = zc / sp->dz + sp->z0;

    /* Compute delta. Will be the same for all following px */

    x0 = (int)floor(xc);
    x_delta = xc - x0;

    y0 = (int)floor(yc);
    y_delta = yc - y0;

    z0 = (int)floor(zc);
    z_delta = zc - z0;

    kernel_computeDelta3D(sp, x_delta, y_delta, z_delta);

    /* loop through all pixels */
    for (int i = 0; i < npx; i++) {
        for (int j = 0; j < npx; j++){
            img[i * npx + j] += phot * fAt3Dj(sp, x0 + i, y0 + j, z0);
        }
    }
}

void f_derivative_PSF(spline *sp, float *img, float *dudt, int npx, float xc, float yc, float zc, float corner_x0, float corner_y0, float phot, float bg) {

    int npy = npx;
    int x0, y0, z0; // px indices
    float x_delta,y_delta,z_delta;  // deltas
    float model;  // model value
    // float* theta = (float *)malloc(sizeof(float) * sp->NV_PSP);
    float theta[sp->NV_PSP];

    /* coordinate of upper left corner */
    // xc = 0 - xc + sp->x0 - corner_x0;
    // yc = 0 - yc + sp->y0 - corner_y0;
    // zc = zc / sp->dz + sp->z0;

    /* Compute delta. Will be the same for all following px */
    x0 = (int)floor(xc);
    x_delta = xc - x0;

    y0 = (int)floor(yc);
    y_delta = yc - y0;

    z0 = (int)floor(zc);
    z_delta = zc - z0;

    // thetas [x y phot bg z]
    theta[0] = xc;
    theta[1] = yc;
    theta[2] = phot;
    theta[3] = bg;
    theta[4] = zc;

    kernel_computeDelta3D(sp, x_delta, y_delta, z_delta);
    /* loop through all pixels */
    for (int i = 0; i < npx; i++) {
        for (int j = 0; j < npy; j++){
            float deriv_px[5] =  { 0 };     
            kernel_DerivativeSpline(sp, x0 + i, y0 + j, z0, theta, deriv_px, &model);
            
            // set model image (almost for free when calc. gradient)
            img[i * npx + j] += model; // unlike in fPSF, background and photons is already in theta.

            // store gradient
            for (int k = 0; k < sp->NV_PSP; k++) {
                dudt[k * npx * npx + i * npx + j] = deriv_px[k];
            }
  
        }
    }

    // free(theta);
}

/** Aggregate Derivatives (over pixels)
 * dudt_agg must be initialised!
 */
void f_derivative_aggregate(spline *sp, float *dudt_px, float *dudt_agg, int npx, int npy) {
    for (int i = 0; i < sp->NV_PSP; i++){
        for (int j = 0; j < npx; j++) {
            for (int k = 0; k < npx; k++) {
                dudt_agg[i] += dudt_px[i * npx * npx + j * npx + k];
            }
        }
    }
    
}

spline* initSpline(const float *coeff, const int xsize, int const ysize, const int zsize, 
                   const float x0, const float y0, const float z0, const float dz) {

    spline * sp;
    sp =(spline *)malloc(sizeof(spline));

    sp->xsize = xsize;
    sp->ysize = ysize;
    sp->zsize = zsize;

    sp->roi_out_eps = 1e-10;
    sp->roi_out_deriv_eps = 0.0;
    sp->NV_PSP = 5; // number of parameters

    sp->add_bg_to_model = false;
    
    sp->x0 = x0;
    sp->y0 = y0;
    sp->z0 = z0;
    sp->dz = dz; 

    int tsize = xsize * ysize * zsize * 64;
    sp->coeff = (float *)malloc(sizeof(float)*tsize);
    sp->delta_f = (float *)malloc(sizeof(float)*64);
    sp->delta_dxf = (float *)malloc(sizeof(float)*64);
    sp->delta_dyf = (float *)malloc(sizeof(float)*64);
    sp->delta_dzf = (float *)malloc(sizeof(float)*64);

    /* Copy spline coefficients. */
    for (int i=0; i<tsize; i++){
        sp->coeff[i] = coeff[i];
    }

    /* init */
    for (int i=0; i<64; i++){
        sp->delta_f[i] = 0.0;
        sp->delta_dxf[i] = 0.0;
        sp->delta_dyf[i] = 0.0;
        sp->delta_dzf[i] = 0.0;
    }

    return sp;
}
