// Header file

#ifndef SPLINE_PSF_GPU_H_
#define SPLINE_PSF_GPU_H_

namespace spline_psf_gpu {

    /* Spline Structure */
    /**
     * @brief defines the cubic spline and holds its coefficients
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

        float *coeff;

    } spline;


    // Initialisation of Spline Coefficients on Device
    // Args:
    //      xsize, ysize, zsize:  size of the coefficients in the respective axis
    //      h_coeff: coefficients on host
    // Returns:
    //      spline*:    pointer to spline struct living on the device (!)
    auto d_spline_init(int xsize, int ysize, int zsize, const float *h_coeff) -> spline*;


    // Wrapper function to compute the ROIs on the device.
    // Takes in all the host arguments and returns leaves the ROIs on the device
    // 
    auto forward_rois_host2device(spline *d_sp, const int n, const int roi_size_x, const int roi_size_y,
        const float *h_x, const float *h_y, const float *h_z, const float *h_phot) -> float*;

    // Wrapper function to ocmpute the ROIs on the device and ships it back to the host
    // Takes in all the host arguments and returns the ROIs to the host
    // Allocation for rois must have happened outside
    // 
    auto forward_rois_host2host(spline *d_sp, float *h_rois, const int n, const int roi_size_x, const int roi_size_y,
        const float *h_x, const float *h_y, const float *h_z, const float *h_phot) -> void;
        
}

#endif  // SPLINE_PSF_GPU_H_