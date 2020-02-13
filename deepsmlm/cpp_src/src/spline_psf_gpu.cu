//
//  Created by Lucas Müller on 12.02.2020
//  Copyright © 2020 Lucas-Raphael Müller. All rights reserved.
//
#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#include "spline_psf_gpu.cuh"

__device__
void kernel_computeDelta3D(spline *sp, 
float* delta_f, float* delta_dxf, float* delta_dyf, float* delta_dzf, 
float x_delta, float y_delta, float z_delta);

__global__
void fAt3Dj(spline *sp, float* rois, int roi_ix, int npx, int npy, 
int xc, int yc, int zc, float phot, float x_delta, float y_delta, float z_delta);

__global__
void fPSF(spline *sp, float *rois, int npx, int npy, 
float* xc_, float* yc_, float* zc_, float* phot_);

spline* d_spline_init(int xsize, int ysize, int zsize, const float *h_coeff) {

    check_host_coeff(h_coeff);
    

    spline* sp;
    sp = (spline *)malloc(sizeof(spline));

    sp->xsize = xsize;
    sp->ysize = ysize;
    sp->zsize = zsize;

    sp->roi_out_eps = 1e-10;
    sp->roi_out_deriv_eps = 0.0;

    sp->NV_PSP = 5;  
    sp->n_coeff = 64;

    int tsize = xsize * ysize * zsize * 64;

    float *d_coeff;
    cudaMalloc(&d_coeff, tsize * sizeof(float));
    cudaMemcpy(d_coeff, h_coeff, tsize * sizeof(float), cudaMemcpyHostToDevice);

    sp->coeff = d_coeff;

    spline* d_sp;
    cudaMalloc(&d_sp, sizeof(spline));
    cudaMemcpy(d_sp, sp, sizeof(spline), cudaMemcpyHostToDevice);

    return d_sp;
}

void check_host_coeff(const float *h_coeff) {
    // printf("Host coeff: \n");
    std::cout << "Host coeff: \n" << std::endl;
    for (int i = 0; i < 10000; i++) {
        printf("i: %d coeff: %f\n", i, h_coeff[i]);
    }
    printf("\n");
    return;
}

// Just a dummy for checking correct parsing from python
__global__
void check_spline(spline *d_sp) {
    printf("Checking spline ...\n");
    printf("\txs, ys, zs: %i %i %i\n", d_sp->xsize, d_sp->ysize, d_sp->zsize);

    printf("\tDevice coeff: \n");
    for (int i = 0; i < 10000; i++) {
        printf("i: %d coeff %f\n", d_sp->coeff[i]);
    }
    printf("\n");
}

__device__
void kernel_computeDelta3D(spline *sp, 
    float* delta_f, float* delta_dxf, float* delta_dyf, float* delta_dzf, 
    float x_delta, float y_delta, float z_delta) {

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
void fAt3Dj(spline *sp, float* rois, int roi_ix, int npx, int npy,
    int xc, int yc, int zc, float phot, float x_delta, float y_delta, float z_delta) {
    
    int i = (blockIdx.x * blockDim.x + threadIdx.x) / npx;
    int j = (blockIdx.x * blockDim.x + threadIdx.x) % npx;

    if ((i < 0) || (i >= 26) || (j < 0) || (j >= 26)) {
        assert((int) 0);
    }

     // allocate space for df, dxf, dyf, dzf
    __shared__ float delta_f[64];
    __shared__ float dxf[64];
    __shared__ float dyf[64];
    __shared__ float dzf[64];
    if (i == 0 and j == 0) {
        // initialise the shared memory
        for (int k = 0; k < 64; k++) {
            delta_f[k] = 0.0;
            dxf[k] = 0.0;
            dyf[k] = 0.0;
            dzf[k] = 0.0;
        }
        kernel_computeDelta3D(sp, delta_f, dxf, dyf, dzf, x_delta, y_delta, z_delta);
    }
    __syncthreads();

    // if (i == 1 and j == 1) {
    //     printf("df, dxf, dyf, dzf\n");
    //     for (int k = 0; k < 64; k++) {
    //         printf("(%f, %f, %f, %f)\n", delta_f[k], dxf[k], dyf[k], dzf[k]);
    //     }
    // }
    // __syncthreads();

    // printf("xc: %d\n", xc);

    xc = xc + i;
    yc = yc + j;

    // printf("xc: %d\n", xc);
    
    // Throw 0 for outside points (only x,y considered).
    if ((xc < 0) || (xc > sp->xsize-1) || (yc < 0) || (yc > sp->ysize-1)) {
        rois[roi_ix * npx * npy + i * npy + j] = sp->roi_out_eps;
        return;
    }

    // printf("Before max min.\n");
    // xc = max(xc,0);
    // xc = min(xc,sp->xsize-1);

    // yc = max(yc,0);
    // yc = min(yc,sp->ysize-1);

    // zc = max(zc,0);
    // zc = min(zc,sp->zsize-1);
    // // printf("After max min.\n");

    float fv = 0.0;

    // // for certain pixel
    // if ((xc == 10) && (yc == 10)) {
    //     printf("checking xc, yc: %d, %d\nchecking delta_f", xc, yc);
    //     for (int l = 0; l < 64; l++) {
    //         printf("%f\n", delta_f[l]);
    //     }
    //     // check coefficient indices
    //     printf("Checking indices and coeff values: \n");
    //     for (int l = 0; l < 64; l++) {
    //         int ix = l * (sp->xsize * sp->ysize * sp->zsize) + zc * (sp->xsize * sp->ysize) + yc * sp->xsize + xc;
    //         printf("%d %f\n", ix, sp->coeff[ix]);
    //     }
    // }
    // __syncthreads();

    for (int k = 0; k < 64; k++) {
        fv += delta_f[k] * sp->coeff[k * (sp->xsize * sp->ysize * sp->zsize) + zc * (sp->xsize * sp->ysize) + yc * sp->xsize + xc];
    }
    // write to global roi stack
    rois[roi_ix * npx * npy + i * npy + j] = phot * fv;
    return;
}

__global__
void fPSF(spline *sp, float *rois, int npx, int npy, float* xc_, float* yc_, float* zc_, float* phot_) {
    
    int r = blockIdx.x;  // roi number 'r'

    int x0, y0, z0;
    float x_delta,y_delta,z_delta;

    float xc = xc_[r];
    float yc = yc_[r];
    float zc = zc_[r];
    float phot = phot_[r];

    printf("Trafo coord: (%f %f %f)\n", xc, yc, zc);

    /* Compute delta. Will be the same for all following px */
    x0 = (int)floor(xc);
    x_delta = xc - x0;

    y0 = (int)floor(yc);
    y_delta = yc - y0;

    z0 = (int)floor(zc);
    z_delta = zc - z0;

    printf("r0: (%d %d %d)\n", x0, y0, z0);
    printf("rd: (%f %f %f)\n", x_delta, y_delta, z_delta);

    fAt3Dj<<<1, npx * npy>>>(sp, rois, r, npx, npy, x0, y0, z0, phot, x_delta, y_delta, z_delta);
    cudaDeviceSynchronize();

    cudaError_t err;
    err = cudaGetLastError();

    if (err != cudaSuccess) {
        printf("Error. Code %d, Message: %s\n", err, cudaGetErrorString(err));
    }

    return;
}

auto compute_rois(spline *d_sp, 
    const int n, const float *h_x, const float *h_y, const float *h_z, const float *h_phot) -> float* {

    // init cuda_err
    cudaError_t err = cudaSuccess;

    int roi_size_x = 26;
    int roi_size_y = 26;

    float *d_x, *d_y, *d_z, *d_phot;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));
    cudaMalloc(&d_z, n * sizeof(float));
    cudaMalloc(&d_phot, n * sizeof(float));
    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_phot, h_phot, n * sizeof(float), cudaMemcpyHostToDevice);

    // add output rois on host and device; 
    float* d_rois;
    cudaMalloc(&d_rois, n * roi_size_x * roi_size_y * sizeof(float));
    cudaMemset(d_rois, 0.0, n * roi_size_x * roi_size_y * sizeof(float));

    // #if DEBUG
    check_spline<<<1,1>>>(d_sp);
    cudaDeviceSynchronize();
    // #endif

    // start n blocks which itself start number of px childs (dynamic parallelism)
    fPSF<<<n, 1>>>(d_sp, d_rois, roi_size_x, roi_size_y, d_x, d_y, d_z, d_phot);
    cudaDeviceSynchronize();

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_phot);

    return d_rois;  
}

// Wrapper around compute_roi function to put the results back to host
// 
auto compute_rois_h(spline *d_sp, 
    const int n, const float *h_x, const float *h_y, const float *h_z, const float *h_phot,
    float *h_rois) -> void {

    int roi_size_x = 26;
    int roi_size_y = 26;
    auto d_rois = compute_rois(d_sp, n, h_x, h_y, h_z, h_phot);
    
    // put results to host
    // std::vector<float> h_rois(n * roi_size_x * roi_size_y);  // host
    cudaMemcpy(h_rois, d_rois, n * roi_size_x * roi_size_y * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(&d_rois);

    return;
}
