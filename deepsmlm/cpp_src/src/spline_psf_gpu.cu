//
//  spline_psf_gpu.cu
//
//  Created by Lucas Müller on 30.01.2020.
//  Copyright © 2019 Lucas-Raphael Müller. All rights reserved.
//
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

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
        
        int NV_PSP;  // number of parameters to fit
        int n_coeff;  // number of coefficients per pixel


} spline;

spline* initSpline(int xsize, int ysize, int zsize) {

    spline* sp;
    sp =(spline *)malloc(sizeof(spline));

    sp->xsize = xsize;
    sp->ysize = ysize;
    sp->zsize = zsize;

    sp->roi_out_eps = 1e-10;
    sp->roi_out_deriv_eps = 0.0;

    sp->NV_PSP = 5;  
    sp->n_coeff = 64;

    return sp;
}


__device__
void kernel_computeDelta3D(spline *sp, float* delta_f, float* delta_dxf, float* delta_dyf, float* delta_dzf, float x_delta, float y_delta, float z_delta) {

    int i,j,k;
    float cx,cy,cz;

    cz = 1.0;
    for(i=0;i<4;i++){
        cy = 1.0;
        for(j=0;j<4;j++){
            cx = 1.0;
            for(k=0;k<4;k++){
                delta_f[i*16+j*4+k] = cz * cy * cx;
                if(k<3){
					delta_dxf[i*16+j*4+k+1] = ((float)k+1) * cz * cy * cx;
				}
				if(j<3){
					delta_dyf[i*16+(j+1)*4+k] = ((float)j+1) * cz * cy * cx;
				}
				if(i<3){
					delta_dzf[(i+1)*16+j*4+k] = ((float)i+1) * cz * cy * cx;
				}
                cx = cx * x_delta;
            }
            cy = cy * y_delta;
        }
        cz= cz * z_delta;
    }
}

__global__
void fAt3Dj(spline *sp, float* rois, int roi_ix, int npx, int npy, float* coeff, 
    int xc, int yc, int zc, float phot, float x_delta, float y_delta, float z_delta) {
    
    int i = (blockIdx.x * blockDim.x + threadIdx.x) / npx;
    int j = (blockIdx.x * blockDim.x + threadIdx.x) % npx;

     // allocate space for df, dxf, dyf, dzf
    __shared__ float delta_f[64], dxf[64], dyf[64], dzf[64];
    if (i == 0 and j == 0) {
        kernel_computeDelta3D(sp, delta_f, dxf, dyf, dzf, x_delta, y_delta, z_delta);
    }
    __syncthreads();

    xc += i;
    yc += j;
    
    float fv = 0;
    // Throw 0 for outside points (only x,y considered).
    if ((xc < 0) || (xc > sp->xsize-1) || (yc < 0) || (yc > sp->ysize-1)) {
        rois[roi_ix * npx * npy + i * npy + j] = sp->roi_out_eps;
        return;
    }

    xc = fmax(xc,0);
    xc = fmin(xc,sp->xsize-1);

    yc = fmax(yc,0);
    yc = fmin(yc,sp->ysize-1);

    zc = fmax(zc,0);
    zc = fmin(zc,sp->zsize-1);

    for (int i=0; i < 64; i++) {
        fv += delta_f[i] * coeff[i * (sp->xsize * sp->ysize * sp->zsize) + zc * (sp->xsize * sp->ysize) + yc * sp->xsize + xc];
    }

    // write to global roi stack
    rois[roi_ix * npx * npy + i * npy + j] = phot * fv;
}

__global__
void fPSF(spline *sp, float *rois, int npx, int npy, float* coeff, float* xc_, float* yc_, float* zc_, float* phot_) {
    
    int r = blockIdx.x;  // roi index

    int x0, y0, z0;
    float xc, yc, zc, phot;
    float x_delta,y_delta,z_delta;

    xc = xc_[r];
    yc = yc_[r];
    zc = zc_[r];
    phot = phot_[r];

    /* Compute delta. Will be the same for all following px */
    x0 = (int)floor(xc);
    x_delta = xc - x0;

    y0 = (int)floor(yc);
    y_delta = yc - y0;

    z0 = (int)floor(zc);
    z_delta = zc - z0;

    /* rewrite to start cuda childs */
    fAt3Dj<<<1, npx * npy>>>(sp, rois, r, npx, npy, coeff, x0, y0, z0, phot, x_delta, y_delta, z_delta);
    cudaDeviceSynchronize();
}

int main() {

    // init cuda_err
    cudaError_t err = cudaSuccess;

    // initialise spline strtuct
    spline* sp = initSpline(13, 13, 150);

    // put spline and coefficients from host to device
    spline* d_sp;
    cudaMalloc((void **)&d_sp, sizeof(spline));
    cudaMemcpy(d_sp, sp, sizeof(spline), cudaMemcpyHostToDevice);

    float* coeff;
    cudaMalloc(&coeff, sp->n_coeff * sp->xsize * sp->ysize * sp->zsize * sizeof(float));
    cudaMemset(coeff, 0, sp->n_coeff * sp->xsize * sp->ysize * sp->zsize * sizeof(float));

    // setup n random localisations and ship them to GPU
    int n;
    std::cout << "Enter number of ROIs: ";
    std::cin >> n;
    std::cout << "\n";
    int roi_size_x = 13;
    int roi_size_y = 13;

    float *d_x, *d_y, *d_z, *d_phot;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));
    cudaMalloc(&d_z, n * sizeof(float));
    cudaMalloc(&d_phot, n * sizeof(float));
    cudaMemset(d_x, 0, n * sizeof(float));
    cudaMemset(d_y, 0, n * sizeof(float));
    cudaMemset(d_z, 0, n * sizeof(float));
    cudaMemset(d_phot, 0, n * sizeof(float));

    // add output rois on host and device; 
    float* d_rois;
    cudaMalloc(&d_rois, n * roi_size_x * roi_size_y * sizeof(float));
    cudaMemset(d_rois, 0, n * roi_size_x * roi_size_y * sizeof(float));

    std::vector<float> h_rois(n * roi_size_x * roi_size_y);  // host

    // start n blocks which itself start number of px childs
    fPSF<<<n, 1>>>(d_sp, d_rois, roi_size_x, roi_size_y, coeff, d_x, d_y, d_z, d_phot);
    cudaDeviceSynchronize();

    // put results to host
    cudaMemcpy(h_rois.data(), d_rois, n * roi_size_x * roi_size_y * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Success.\n";

    return 0;  
}
