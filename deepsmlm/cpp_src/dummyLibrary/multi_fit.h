/*
 * Common constants for multiple peak fitting.
 *
 * Hazen 10/17
 *
 */

#ifndef MULTI_FIT_H
#define MULTI_FIT_H

/* debugging */
#define TESTING 0
#define VERBOSE 0

/* number of peak and results parameters. */
#define NFITTING 7
#define NPEAKPAR 9

/* indexes for peak fitting parameters. */
#define HEIGHT 0        /* Height */     
#define XCENTER 1       /* X center */
#define XWIDTH 2        /* Width in x, only relevant for gaussians */
#define YCENTER 3       /* Y center */
#define YWIDTH 4        /* Width in y, only relevant for gaussians */
#define BACKGROUND 5    /* Background level under the peak */
#define ZCENTER 6       /* Z center */

/* additional indexes for results. */
#define STATUS 7        /* Status flag, see below */
#define IERROR 8        /* Error in the fit (integrated over the AOI) */

/* peak status flags */
#define RUNNING 0
#define CONVERGED 1
#define ERROR 2

#define HYSTERESIS 0.6 /* In order to move the AOI or change it's size,
			  the new value must differ from the old value
			  by at least this much (<= 0.5 is no hysteresis). */

/* fitting constants. */
#define LAMBDASTART 1.0     /* Initial lambda value. */
#define LAMBDADOWN 0.75     /* Multiplier for decreasing lambda. */
#define LAMBDAMAX 1.0e+20   /* Maximum lambda value, if we hit this the peak is lost as un-fittable. */
#define LAMBDAMIN 1.0e-3    /* Minimum lambda value. */
#define LAMBDAUP 4.0        /* Multiplier for increasing lambda, if necessary. */

/* peak storage. */
#define INCNPEAKS 500       /* Storage grows in units of 500 peaks. */

/* convergence metric */
#define DELTA_CONVERGENCE 0 /* The default is check for fitting error convergence. Set this
                               to 1 for convergence based on peak fit deltas. */

/*
 * There is one of these for each peak to be fit.
 */
typedef struct peakData
{
  int added;                /* Counter for adding / subtracting the peak from the image. */
  int index;                /* Peak id. */
  int iterations;           /* Number of fitting iterations. */
  int status;               /* Status of the fit (running, converged, etc.). */
  int xi;                   /* Location of the fitting area in x (starting pixel). */
  int yi;                   /* Location of the fitting area in y (starting pixel). */

  double error;             /* Current error. */
  double lambda;            /* Levenberg-Marquadt lambda term. */

  double params[NFITTING];  /* [height x-center x-width y-center y-width background z-center] */
  double *psf;              /* The peaks PSF. */

  void *peak_model;         /* Pointer to peak model specific data (i.e. spline data, etc.) */
} peakData;


/*
 * This structure contains everything necessary to fit an array of peaks on an image.
 */
typedef struct fitData
{
  /* 
   * These are for diagnostics.
   *
   * FIXME: These may overflow if there are lot of frames with a lot of localizations.
   */
  int n_dposv;                  /* Number reset due to an error trying to solve Ax = b. */
  int n_iterations;             /* Number of iterations of fitting. */
  int n_lost;                   /* Number of fits that were lost altogether. */
  int n_margin;                 /* Number reset because they were too close to the edge of the image. */
  int n_neg_fi;                 /* Number reset due to a negative fi. */
  int n_neg_height;             /* Number reset due to negative height. */
  int n_non_converged;          /* Number of fits that did not converge. */
  int n_non_decr;               /* Number of restarts due to non-decreasing error.*/
  
  int fit_size_x;               /* Size of the fitting area in x in pixels. */
  int fit_size_y;               /* Size of the fitting area in y in pixels. */
  int image_size_x;             /* Size of the image in x (fast axis). */
  int image_size_y;             /* Size of the image in y (slow axis). */  
  int jac_size;                 /* The number of terms in the Jacobian. */
  int max_nfit;                 /* The (current) maximum number of peaks that we have storage for. */
  int nfit;                     /* Number of peaks to fit. */
  int roi_n_index;              /* Number of valid points in the ROI index arrays. */

  double minimum_height;        /* This is used to clamp the minimum allowed peak starting height. */

  double xoff;                  /* Offset between the peak center parameter in x and the actual center. */
  double yoff;                  /* Offset between the peak center parameter in y and the actual center. */
  double zoff;                  /* Offset between the peak center parameter in z and the actual center. */

  double tolerance;             /* Fit tolerance. */

  int *bg_counts;               /* Number of peaks covering a particular pixel. */
  int *roi_x_index;             /* Array that translates from ROI to image x coordinates. */
  int *roi_y_index;             /* Array that translates from ROI to image y coordinates. */
  int *stale;                   /* Error value at pixel i is stale and needs to be updated. */

  double *as_xi;                /* Anscombe transform of the image data. */  
  double *bg_data;              /* Fit (background) data. */
  double *bg_estimate;          /* Current background estimate (calculated externally). */
  double *err_i;                /* Current error at pixel i. */
  double *f_data;               /* Fit (foreground) data. */
  double *rqe;                  /* Pixel relative quantum efficiency. */
  double *scmos_term;           /* sCMOS calibration term for each pixel (var/gain^2). */
  double *t_fi;                 /* Transform of the fitting function. */
  double *x_data;               /* Image data. */

  peakData *working_peak;       /* Working copy of the peak that we are trying to improve the fit of. */
  peakData *fit;                /* The peaks to be fit to the image. */

  void *fit_model;              /* Other data/structures necessary to do the fitting, such as a cubic spline structure. */

  /* Specific fitter versions must provide these functions. */
  void (*fn_alloc_peaks)(struct peakData *, int);             /* Function for allocating storage for peaks. */
  void (*fn_calc_JH)(struct fitData *, double *, double *);   /* Function for calculating the Jacobian and the Hessian. */
  void (*fn_calc_peak_shape)(struct fitData *);               /* Function for calculating the current peak shape. */
  int (*fn_check)(struct fitData *);                          /* Function for checking the validity of the working peak parameters. */
  void (*fn_copy_peak)(struct fitData *, struct peakData *, struct peakData *); /* Function for copying peaks. */
  int (*fn_error_fn)(struct fitData *);                       /* Function for calculating the fitting error. */
  void (*fn_free_peaks)(struct peakData *, int);              /* Function for freeing storage for peaks. */
  void (*fn_update)(struct fitData *, double *);              /* Function for updating the working peak parameters. */
  
} fitData;


/*
 * Functions.
 *
 * These all start with mFit to make it easier to figure
 * out in libraries that use this code where they came from.
 */
void mFitAddPeak(fitData *);
double mFitAnscombe(double);
int mFitCalcErr(fitData *);
int mFitCalcErrALS(fitData *);
int mFitCheck(fitData *);
void mFitCleanup(fitData *);
void mFitCopyPeak(fitData *, peakData *, peakData *);
int mFitDeltaConvergence(fitData *, int);
void mFitEstimatePeakHeight(fitData *);
void mFitGetFitImage(fitData *, double *);
int mFitGetNError(fitData *);
void mFitGetPeakPropertyDouble(fitData *, double *, char *);
void mFitGetPeakPropertyInt(fitData *, int32_t *, char *);
void mFitGetResidual(fitData *, double *);
int mFitGetUnconverged(fitData *);
fitData *mFitInitialize(double *, double *, double, int, int);
void mFitInitializeROIIndexing(fitData *, int);
void mFitIterateOriginal(fitData *);
void mFitIterateLM(fitData *);
void mFitNewBackground(fitData *, double *);
void mFitNewImage(fitData *, double *, int);
void mFitNewPeaks(fitData *, int);
double mFitPeakBgSum(fitData *, peakData *);
double mFitPeakFgSum(fitData *, peakData *);
double mFitPeakSum(fitData *, peakData *);
void mFitRecenterPeaks(fitData *);
void mFitRemoveErrorPeaks(fitData *);
void mFitResetPeak(fitData *, int);
void mFitSetPeakStatus(fitData *, int32_t *);
int mFitSolve(double *, double *, int);
void mFitSubtractPeak(fitData *);
void mFitUpdate(peakData *);
void mFitUpdateParam(peakData *, double, int);

#endif
