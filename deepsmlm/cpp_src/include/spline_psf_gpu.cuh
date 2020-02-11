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

    float *coeff;

} spline;


// spline* initSpline(int xsize, int ysize, int zsize);
spline* d_spline_init(int xsize, int ysize, int zsize, const float *h_coeff);

void forward(spline *d_sp);

