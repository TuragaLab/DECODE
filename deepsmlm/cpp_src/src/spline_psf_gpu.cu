//
//  Created by Lucas Müller on 12.02.2020
//  Copyright © 2020 Lucas-Raphael Müller. All rights reserved.
//
#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#include "spline_psf_gpu.cuh"
using namespace spline_psf_gpu;


// internal declarations
void check_host_coeff(const float *h_coeff);

auto forward_rois(spline *d_sp, float *d_rois, const int n, const int roi_size_x, const int roi_size_y, 
    const float *d_x, const float *d_y, const float *d_z, const float *d_phot) -> void;

__device__
auto kernel_computeDelta3D(spline *sp, 
    float* delta_f, float* delta_dxf, float* delta_dyf, float* delta_dzf, 
    float x_delta, float y_delta, float z_delta) -> void;

__global__
auto fAt3Dj(spline *sp, float* rois, int roi_ix, int npx, int npy, 
    int xc, int yc, int zc, float phot, float x_delta, float y_delta, float z_delta) -> void;

__global__
auto kernel_roi(spline *sp, float *rois, const int npx, const int npy, 
    const float* xc_, const float* yc_, const float* zc_, const float* phot_) -> void;

__global__
auto kernel_roi(spline *sp, float *rois, const int npx, const int npy, 
    const float* xc_, const float* yc_, const float* zc_, const float* phot_) -> void;

__global__
auto roi_accumulate(float *frames, const int frame_size_x, const int frame_size_y, const int n_frames,
    const float *rois, const int n_rois, 
    const int *frame_ix, const int *x0, const int *y0, 
    const int roi_size_x, const int roi_size_y) -> void;

namespace spline_psf_gpu {

    // Create struct and ship it to device
    auto d_spline_init(const float *h_coeff, int xsize, int ysize, int zsize) -> spline* {

        // allocate struct on host and ship it to device later
        // ToDo: C++11ify this
        spline* sp;
        sp = (spline *)malloc(sizeof(spline));

        sp->xsize = xsize;
        sp->ysize = ysize;
        sp->zsize = zsize;

        if ((sp->xsize > 32) || (sp->ysize > 32)) {
            // this is because we start threads per pixel and the limit is 1024 threads per block
            throw std::invalid_argument("Invalid ROI size. ROI size must not exceed 32 px in either dimension.");  
        }

        sp->roi_out_eps = 1e-10;
        sp->roi_out_deriv_eps = 0.0;

        sp->NV_PSP = 5;  
        sp->n_coeff = 64;

        int tsize = xsize * ysize * zsize * 64;

        float *d_coeff;
        cudaMalloc(&d_coeff, tsize * sizeof(float));
        cudaMemcpy(d_coeff, h_coeff, tsize * sizeof(float), cudaMemcpyHostToDevice);

        sp->coeff = d_coeff;  // for some reason this should happen here and not d_sp->coeff = d_coeff ...

        // ship to device
        spline* d_sp;
        cudaMalloc(&d_sp, sizeof(spline));
        cudaMemcpy(d_sp, sp, sizeof(spline), cudaMemcpyHostToDevice);

        return d_sp;
    }

    // Wrapper function to compute the ROIs on the device.
    // Takes in all the host arguments and returns leaves the ROIs on the device
    // 
    auto forward_rois_host2device(spline *d_sp, const int n, const int roi_size_x, const int roi_size_y,
        const float *h_x, const float *h_y, const float *h_z, const float *h_phot) -> float* {

        // allocate and copy coordinates and photons
        float *d_x, *d_y, *d_z, *d_phot;
        cudaMalloc(&d_x, n * sizeof(float));
        cudaMalloc(&d_y, n * sizeof(float));
        cudaMalloc(&d_z, n * sizeof(float));
        cudaMalloc(&d_phot, n * sizeof(float));
        cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_z, h_z, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_phot, h_phot, n * sizeof(float), cudaMemcpyHostToDevice);

        // allocate space for rois on device
        float* d_rois;
        cudaMalloc(&d_rois, n * roi_size_x * roi_size_y * sizeof(float));
        cudaMemset(d_rois, 0.0, n * roi_size_x * roi_size_y * sizeof(float));

        #if DEBUG
            check_spline<<<1,1>>>(d_sp);
            cudaDeviceSynchronize();
        #endif

        // call to actual implementation
        forward_rois(d_sp, d_rois, n, roi_size_x, roi_size_y, d_x, d_y, d_z, d_phot);

        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_z);
        cudaFree(d_phot);

        return d_rois;  
    }

    // Wrapper function to ocmpute the ROIs on the device and ships it back to the host
    // Takes in all the host arguments and returns the ROIs to the host
    // Allocation for rois must have happened outside
    // 
    auto forward_rois_host2host(spline *d_sp, float *h_rois, const int n, const int roi_size_x, const int roi_size_y,
        const float *h_x, const float *h_y, const float *h_z, const float *h_phot) -> void {

        auto d_rois = forward_rois_host2device(d_sp, n, roi_size_x, roi_size_y, h_x, h_y, h_z, h_phot);
        
        cudaMemcpy(h_rois, d_rois, n * roi_size_x * roi_size_y * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_rois);
        return;
    }

    auto forward_frames_host2host(spline *d_sp, float *h_frames, const int frame_size_x, const int frame_size_y, const int n_frames,
        const int n_rois, const int roi_size_x, const int roi_size_y,
        const int *h_frame_ix, const float *h_xr0, const float *h_yr0, const float *h_z0, 
        const int *h_x_ix, const int *h_y_ix, const float *h_phot) -> void {

        auto d_frames = forward_frames_host2device(d_sp, frame_size_x, frame_size_y, n_frames, 
            n_rois, roi_size_x, roi_size_y, h_frame_ix, h_xr0, h_yr0, h_z0, h_x_ix, h_y_ix, h_phot);

        cudaMemcpy(h_frames, d_frames, n_frames * frame_size_x * frame_size_y * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_frames);
        return;        
    }

    auto forward_frames_host2device(spline *d_sp, const int frame_size_x, const int frame_size_y, const int n_frames,
        const int n_rois, const int roi_size_x, const int roi_size_y,
        const int *h_frame_ix, const float *h_xr0, const float *h_yr0, const float *h_z0, 
        const int *h_x_ix, const int *h_y_ix, const float *h_phot) -> float* {

        cudaError_t err;
        
        // ToDo: maybe convert to stream
        float* d_frames;
        cudaMalloc(&d_frames, n_frames * frame_size_x * frame_size_y * sizeof(float));
        cudaMemset(d_frames, 0.0, n_frames * frame_size_x * frame_size_y * sizeof(float));

        // allocate indices
        int *d_xix, *d_yix, *d_fix;
        cudaMalloc(&d_xix, n_rois * sizeof(int));
        cudaMalloc(&d_yix, n_rois * sizeof(int));
        cudaMalloc(&d_fix, n_rois * sizeof(int));
        cudaMemcpy(d_xix, h_x_ix, n_rois * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_yix, h_y_ix, n_rois * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_fix, h_frame_ix, n_rois * sizeof(int), cudaMemcpyHostToDevice);

        auto d_rois = forward_rois_host2device(d_sp, n_rois, roi_size_x, roi_size_y, h_xr0, h_yr0, h_z0, h_phot);

        // accumulate rois into frames
        const int blocks = (n_rois * roi_size_x * roi_size_y) / 256 + 1;
        const int thread_p_block = 512;
        roi_accumulate<<<blocks, thread_p_block>>>(d_frames, frame_size_x, frame_size_y, n_frames, 
            d_rois, n_rois, d_fix, d_xix, d_yix, roi_size_x, roi_size_y);

        cudaDeviceSynchronize();

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cout << "Error during frame computation.\nCode: " << err << "Information: \n" << cudaGetErrorString(err) << std::endl;
        }

        cudaFree(d_xix);
        cudaFree(d_yix);
        cudaFree(d_fix);
        cudaFree(d_rois);

        return d_frames;
    }
} // namespace spline_psf_gpu


auto forward_rois(spline *d_sp, float *d_rois, const int n, const int roi_size_x, const int roi_size_y, 
    const float *d_x, const float *d_y, const float *d_z, const float *d_phot) -> void {
    
    // init cuda_err
    cudaError_t err = cudaSuccess;

    // throw error if roi size is too big
    if ((roi_size_x > 32) || (roi_size_y > 32)) {
        throw std::invalid_argument("ROI size (per PSF) must not exceed 32 pixels.");
    }

    // start n blocks which itself start threads corresponding to the number of px childs (dynamic parallelism)
    kernel_roi<<<n, 1>>>(d_sp, d_rois, roi_size_x, roi_size_y, d_x, d_y, d_z, d_phot);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Error during ROI computation.\nCode: " << err << "Information: \n" << cudaGetErrorString(err) << std::endl;
    }

    return;
}

// Just a dummy for checking correct parsing from python
// ... had to learn the hard way ...
__global__
auto check_spline(spline *d_sp) -> void {
    printf("Checking spline ...\n");
    printf("\txs, ys, zs: %i %i %i\n", d_sp->xsize, d_sp->ysize, d_sp->zsize);
    printf("\toutside-roi value: %f\n", d_sp->roi_out_eps);
    printf("\toutside-roi derivative value: %f\n", d_sp->roi_out_deriv_eps);

    printf("\tDevice coeff: \n");
    for (int i = 0; i < 100; i++) {
        printf("\t\ti: %d coeff %f\n", d_sp->coeff[i]);
    }
    printf("\n");
}

// kernel to compute common term for spline function (for all pixels this will stay the same)
__device__
auto kernel_computeDelta3D(spline *sp, float* delta_f, float* delta_dxf, float* delta_dyf, float* delta_dzf, 
    float x_delta, float y_delta, float z_delta) -> void {

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

// kernel to compute pixel-wise term
__global__
auto fAt3Dj(spline *sp, float* rois, const int roi_ix, const int npx, const int npy,
    int xc, int yc, int zc, float phot, float x_delta, float y_delta, float z_delta) -> void {
    
    int i = (blockIdx.x * blockDim.x + threadIdx.x) / npx;
    int j = (blockIdx.x * blockDim.x + threadIdx.x) % npx;

     // allocate space for df, dxf, dyf, dzf
    __shared__ float delta_f[64];
    __shared__ float dxf[64];
    __shared__ float dyf[64];
    __shared__ float dzf[64];

    if (i == 0 and j == 0) {

        for (int k = 0; k < 64; k++) {
            delta_f[k] = 0.0;
            dxf[k] = 0.0;
            dyf[k] = 0.0;
            dzf[k] = 0.0;
        }

        // This is different to the C library since we needed to rearrange a bit to account for the GPU parallelism
        kernel_computeDelta3D(sp, delta_f, dxf, dyf, dzf, x_delta, y_delta, z_delta);
    }
    __syncthreads();

    xc = xc + i;
    yc = yc + j;
    
    // If the lateral position is outside the calibration, return epsilon value
    if ((xc < 0) || (xc > sp->xsize-1) || (yc < 0) || (yc > sp->ysize-1)) {
        rois[roi_ix * npx * npy + i * npy + j] = sp->roi_out_eps;
        return;
    }

    // the following is unnecessary if we have the test with return as above
    // xc = max(xc,0);
    // xc = min(xc,sp->xsize-1);

    // yc = max(yc,0);
    // yc = min(yc,sp->ysize-1);

    zc = max(zc,0);
    zc = min(zc,sp->zsize-1);

    float fv = 0.0;

    for (int k = 0; k < 64; k++) {
        fv += delta_f[k] * sp->coeff[k * (sp->xsize * sp->ysize * sp->zsize) + zc * (sp->xsize * sp->ysize) + yc * sp->xsize + xc];
    }

    // write to global roi stack
    rois[roi_ix * npx * npy + i * npy + j] = phot * fv;
    return;
}

// kernel to compute psf for a single emitter
__global__
auto kernel_roi(spline *sp, float *rois, const int npx, const int npy, const float* xc_, const float* yc_, const float* zc_, const float* phot_) -> void {
    
    int r = blockIdx.x;  // roi number 'r'

    int x0, y0, z0;
    float x_delta,y_delta,z_delta;

    float xc = xc_[r];
    float yc = yc_[r];
    float zc = zc_[r];
    float phot = phot_[r];

    /* Compute delta. Will be the same for all following px */
    x0 = (int)floor(xc);
    x_delta = xc - x0;

    y0 = (int)floor(yc);
    y_delta = yc - y0;

    z0 = (int)floor(zc);
    z_delta = zc - z0;

    fAt3Dj<<<1, npx * npy>>>(sp, rois, r, npx, npy, x0, y0, z0, phot, x_delta, y_delta, z_delta);
    // cudaDeviceSynchronize();  // not needed. Device sync once for the forward_rois thing is sufficient

    return;
}

// accumulate rois to frames
__global__
auto roi_accumulate(float *frames, const int frame_size_x, const int frame_size_y, const int n_frames,
                    const float *rois, const int n_rois, 
                    const int *frame_ix, const int *x0, const int *y0, 
                    const int roi_size_x, const int roi_size_y) -> void {

        // kernel ix
        const long kx = (blockIdx.x * blockDim.x + threadIdx.x);
        if (kx >= n_rois * roi_size_x * roi_size_y) {
            return;
        }

        // roi index
        const long j = kx % roi_size_y;
        const long i = ((kx - j) / roi_size_y) % roi_size_x;
        const long r = (((kx - j) / roi_size_y) - i) / roi_size_x;

        const long ii = x0[r] + i;
        const long jj = y0[r] + j;

        if ((frame_ix[r] < 0) || (frame_ix[r] >= n_frames)) {  // if frame ix is outside
            return;
        }

        if ((ii < 0) || (jj < 0) || (ii >= frame_size_x) || (jj >= frame_size_y)) {  // if outside frame throw away
            return;
        }
        float val = rois[r * roi_size_x * roi_size_y + i * roi_size_y + j];
        atomicAdd(&frames[frame_ix[r] * frame_size_x * frame_size_y + ii * frame_size_y + jj], val);  // otherwise race condition 

        return;
    }


