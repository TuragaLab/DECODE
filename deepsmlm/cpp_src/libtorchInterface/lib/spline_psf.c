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
                cx = cx * x_delta;
            }
            cy = cy * y_delta;
        }
        cz= cz * z_delta;
    }
}

float fAt3Dj(spline *sp, int xc, int yc, int zc) {
    float fv = 0;
    
    // Throw 0 for outside points (only x,y considered).
    if ((xc < 0) || (xc > sp->xsize-1) || (yc < 0) || (yc > sp->ysize-1)) {
        return 0.0;
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
    xc = 0 - xc + sp->x0 - corner_x0;
    yc = 0 - yc + sp->y0 - corner_y0;
    zc = zc / sp->dz + sp->z0;

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

spline* initSpline(float *coeff, int xsize, int ysize, int zsize, float x0, float y0, float z0, float dz) {

    spline * sp;
    sp =(spline *)malloc(sizeof(spline));

    sp->xsize = xsize;
    sp->ysize = ysize;
    sp->zsize = zsize;
    
    sp->x0 = x0;
    sp->y0 = y0;
    sp->z0 = z0;
    sp->dz = dz;
    

    int tsize = xsize * ysize * zsize * 64;
    sp->coeff = (float *)malloc(sizeof(float)*tsize);
    sp->delta_f = (float *)malloc(sizeof(float)*64);

    /* Copy spline coefficients. */
    for (int i=0; i<tsize; i++){
        sp->coeff[i] = coeff[i];
    }

    /* init */
    for (int i=0; i<64; i++){
        sp->delta_f[i] = 0.0;
    }

    return sp;
}
