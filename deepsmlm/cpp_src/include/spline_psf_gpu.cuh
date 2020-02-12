// Header file

#ifndef SPLINE_PSF_GPU_H_
#define SPLINE_PSF_GPU_H_

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

    float *coeff;

} spline;


// Initialisation of Spline Coefficients on Device
// Args:
//      xsize, ysize, zsize:  size of the coefficients in the respective axis
//      h_coeff: coefficients on host
// Returns:
//      spline*:    pointer to spline struct living on the device (!)
spline* d_spline_init(int xsize, int ysize, int zsize, const float *h_coeff);

auto compute_rois(spline *d_sp, const int n, const float *h_x, const float *h_y, const float *h_z, const float *h_phot) -> float*;
auto compute_rois_h(spline *d_sp, const int n, const float *h_x, const float *h_y, const float *h_z, const float *h_phot) -> void;

#endif  // 