
#include <iostream>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

__device__
float kernel_common_term()
{
    return 5.0;
}

__global__
void kernel_px_term(float *rois_glob, int roi_ix, int roi_size, float xc, float yc, float cterm) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= roi_size * roi_size) return;

     __shared__ float roi[16 * 16];
    __syncthreads();
    
    const int psf_size = 5;

    // convert to 2d index
    int ii = i / roi_size;
    int jj = i % roi_size;
    if (ii < xc + psf_size/2 && ii > xc - psf_size/2 && jj < yc + psf_size/2 && jj > yc - psf_size/2) {
        roi[i] = cterm;
    } else {
        roi[i] = 0.0;
    }

    // debug
    // __syncthreads();
    if (i == 0) {
        printf("ROI IX: %d\n", roi_ix);
        // for (int j = 0; j < roi_size * roi_size; j++) {
        //     printf("[%f]", roi[j]);
        // }
        // printf("\n");
    }

    // write to global roi
    rois_glob[roi_ix * roi_size * roi_size + i] = roi[i];

    return;
}

__global__
void roi_parent(float *rois_glob, int num_rois, int roi_size) {
    int n = blockIdx.x;

    // do common term
    float common = kernel_common_term();
    // start childs for every px
    kernel_px_term<<<1, 16 * 16>>>(rois_glob, n, roi_size, 5.0, 10.0, common);
    cudaDeviceSynchronize();

}


int main() {

    // suppose list of xyz coordinates
    const int n = 1;
    const int roi_size = 13;

    // initialise rois
    float *rois;
    cudaMalloc(&rois, n * roi_size * roi_size * sizeof(float));
    cudaMemset(rois, 0, n * roi_size * roi_size * sizeof(float));

    // initiailise parent processes for every roi
    roi_parent<<<n, 1>>>(rois, n, roi_size);
    cudaDeviceSynchronize();

    // cpy stuff back to host
    float *h_rois = (float *)malloc(n * roi_size * roi_size);
    cudaMemcpy(h_rois, rois, n * roi_size * roi_size, cudaMemcpyDeviceToHost);

    // show a couple of values
    int roi_ix = 5;
    for (int i = 0; i < n * roi_size * roi_size; i++) {
        std::cout << h_rois[i] << "\n";
    }

    return 0;
} 