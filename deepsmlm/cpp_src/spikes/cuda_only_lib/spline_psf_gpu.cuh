// Header file

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


spline* initSpline(int xsize, int ysize, int zsize);

__device__
void kernel_computeDelta3D(spline *sp, 
float* delta_f, float* delta_dxf, float* delta_dyf, float* delta_dzf, 
float x_delta, float y_delta, float z_delta);

__global__
void fAt3Dj(spline *sp, float* rois, int roi_ix, int npx, int npy, float* coeff, 
int xc, int yc, int zc, float phot, float x_delta, float y_delta, float z_delta);

__global__
void fPSF(spline *sp, float *rois, int npx, int npy, 
float* coeff, float* xc_, float* yc_, float* zc_, float* phot_);

void run();