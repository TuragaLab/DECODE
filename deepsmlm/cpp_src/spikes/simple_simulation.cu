
#include <iostream>
#include <stdio.h>
#include <vector>

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

void print_roi(std::vector<float> h_rois, int nl) {

    // show a couple of values
    std::cout << "ROI on Host: \n";
    for (int i = 0; i < nl; i++) {
        std::cout << h_rois[i] << " ";
    }
    std::cout << "\n";

}

__global__
void print_roi_device(float *d_rois, int nl) {

    printf("ROI on Device: \n");
    for (int i = 0; i < nl; i++) {
        printf("%.0f ", d_rois[i]);
    }
    printf("\n");

}


int main() {

    cudaError_t err = cudaSuccess;

    // suppose list of xyz coordinates
    const int n = 500;
    const int roi_size = 13;

    // initialise rois
    float *rois;
    cudaMalloc(&rois, n * roi_size * roi_size * sizeof(float));
    cudaMemset(rois, 0, n * roi_size * roi_size * sizeof(float));

    // on host
    std::vector<float> h_rois(n * roi_size * roi_size);  // no need to fill since all elements will be filled by cuda

    // initiailise parent processes for every roi
    roi_parent<<<n, 1>>>(rois, n, roi_size);
    cudaDeviceSynchronize();

    // print_roi_device<<<1,1>>>(rois, n * roi_size * roi_size);

    // cpy stuff back to host
    err = cudaMemcpy(h_rois.data(), rois, n * roi_size * roi_size * sizeof(float), cudaMemcpyDeviceToHost);

    // print_roi(h_rois, n * roi_size * roi_size);

    return 0;
} 