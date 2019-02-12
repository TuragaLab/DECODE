/*
 * Core routines that are common to all of the fitters.
 *
 * Hazen 09/18
 */

/* Include */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "multi_fit.h"

/* LAPACK Functions */
extern void dposv_(char* uplo, int* n, int* nrhs, double* a, int* lda,
		   double* b, int* ldb, int* info);

/*
 * mFitAddPeak()
 *
 * Add working peak to the foreground and background data arrays. This
 * also adds the sCMOS variance term.
 *
 * fit_data - pointer to a fitData structure.
 */
void mFitAddPeak(fitData *fit_data)
{
  int i,j,k;
  double bg,mag,rqe;
  peakData *peak;

  peak = fit_data->working_peak;

  peak->added++;

  if(TESTING){
    if(peak->added != 1){
      printf("Peak count error detected in mFitAddPeak()! %d\n", peak->added);
      exit(EXIT_FAILURE);
    }
  }

  /* Add peak to the foreground and background arrays. */
  i = peak->yi * fit_data->image_size_x + peak->xi;
  bg = peak->params[BACKGROUND];
  mag = peak->params[HEIGHT];
  for(j=0;j<fit_data->roi_n_index;j++){
    k = fit_data->roi_y_index[j]*fit_data->image_size_x + fit_data->roi_x_index[j] + i;
    rqe = fit_data->rqe[k];
    fit_data->f_data[k] += mag * peak->psf[j] * rqe;
    fit_data->bg_counts[k] += 1;
    fit_data->bg_data[k] += bg * rqe + fit_data->scmos_term[k];
    fit_data->stale[k] = 1;      
  }
}

/*
 * mFitAnscombe()
 *
 * Calculate the Anscombe transform.
 *
 * x - Signal / function value.
 * var - (Gaussian) variance.
 */
double mFitAnscombe(double x)
{
  if(x < (-0.375 + 1.0e-6)){
    if(TESTING){
      printf(" Negative value in Anscombe transform! %.3f\n\n", x);
    }
    return 0.0;
  }
  else{
    return 2.0*sqrt(x + 0.375);
  }
}

/*
 * mFitCalcErr()
 *
 * Calculate the fit error of a peak. Technically this is actually the total error 
 * in the pixels that are covered by the peak. When peaks overlap substantially they 
 * will have similar errors.
 *
 * fit_data - pointer to a fitData structure.
 *
 * Returns 0 if there were no errors.
 */
int mFitCalcErr(fitData *fit_data)
{
  int i,j,k;
  double err,fi,xi;
  peakData *peak;

  peak = fit_data->working_peak;

  if(peak->status != RUNNING){
    return 0;
  }

  if(VERBOSE){
    printf("mFCE, xi - %d, yi - %d\n", peak->xi, peak->yi);
  }

  i = peak->yi * fit_data->image_size_x + peak->xi;
  err = 0.0;
  for(j=0;j<fit_data->roi_n_index;j++){
    k = fit_data->roi_y_index[j]*fit_data->image_size_x + fit_data->roi_x_index[j] + i;
    if(fit_data->stale[k] != 0){

      /*
       * f_data and bg_data already include the rqe correction and the 
       * sCMOS variance term.
       *
       * f_data[k] = fit function[k] * rqe[k]
       * bg_data[k] = bg_counts[k] * (fit_bg_term[k] * rqe[k] + variance[k])
       *
       * So after division by bg_counts[k] we have:
       * f_data[k] = fit_function[k] * rqe[k] + fit_bg_term[k] * rqe[k] + variance[k]
       */
      fi = fit_data->f_data[k] + fit_data->bg_data[k] / ((double)fit_data->bg_counts[k]);
      if(fi <= 0.0){
	/*
	 * This can happen because the fit background can be negative. I
	 * don't think it is a problem that merits crashing everything.
	 */
	if(TESTING){
	  printf(" Negative f detected!\n");
	  printf("  index %d\n", peak->index);
	  printf("      f %.3f\n", fi);
	  printf("    fit %.3f\n", fit_data->f_data[k]);
	  printf("     bg %.3f\n", fit_data->bg_data[k]);
	  printf("   cnts %d\n\n", fit_data->bg_counts[k]);
	}
	fit_data->n_neg_fi++;
	return 1;
      }
      fit_data->t_fi[k] = fi;
      xi = fit_data->x_data[k];
      /* 
       * This should not happen as the expectation is that negative image
       * values are eliminated upstream of this step.
       */
      if(TESTING){
	if(xi <= 0.0){
	  printf(" Negative x detected!\n");
	  printf("   xi %.3f\n\n", xi);
	  err = peak->error;
	  exit(EXIT_FAILURE);
	}
      }
      fit_data->err_i[k] = 2.0*((fi-xi)-xi*log(fi/xi));
      
      fit_data->stale[k] = 0;
    }
    err += fit_data->err_i[k];
    if(TESTING){
      /*
       * FIXME: Should also test for +- infinity?
       */
      if (isnan(err)){
	printf(" NAN error detected!\n");
	printf("  index %d\n", peak->index);
	printf("     fi %.3f\n", fi);
	printf("     xi %.3f\n\n", xi);
	exit(EXIT_FAILURE);
      }
    }
  }
  peak->error = err;
  
  return 0;
}


/*
 * mFitCalcErrALS()
 *
 * The Anscombe least-squares version of the error function.
 *
 * fit_data - pointer to a fitData structure.
 *
 * Returns 0 if there were no errors.
 */
int mFitCalcErrALS(fitData *fit_data)
{
  int i,j,k;
  double err,di,fi,as_fi;
  peakData *peak;

  peak = fit_data->working_peak;

  if(peak->status != RUNNING){
    return 0;
  }

  if(VERBOSE){
    printf("mFCEL, xi - %d, yi - %d\n", peak->xi, peak->yi);
  }

  i = peak->yi * fit_data->image_size_x + peak->xi;
  err = 0.0;
  for(j=0;j<fit_data->roi_n_index;j++){
    k = fit_data->roi_y_index[j]*fit_data->image_size_x + fit_data->roi_x_index[j] + i;
    if(fit_data->stale[k] != 0){

      /*
       * f_data and bg_data already include the rqe correction and the 
       * sCMOS variance term.
       *
       * f_data[k] = fit function[k] * rqe[k]
       * bg_data[k] = bg_counts[k] * (fit_bg_term[k] * rqe[k] + variance[k])
       *
       * So after division by bg_counts[k] we have:
       * f_data[k] = fit_function[k] * rqe[k] + fit_bg_term[k] * rqe[k] + variance[k]
       */      
      fi = fit_data->f_data[k] + fit_data->bg_data[k] / ((double)fit_data->bg_counts[k]);
      as_fi = mFitAnscombe(fi);

      /* For now, keep track of how often fi was negative. */
      if(as_fi == 0.0){
	if(TESTING){
	  printf(" Negative f detected!\n");
	  printf("  index %d\n", peak->index);
	  printf("      f %.3f\n", fi);
	  printf("    fit %.3f\n", fit_data->f_data[k]);
	  printf("     bg %.3f\n", fit_data->bg_data[k]);
	  printf("   cnts %d\n\n", fit_data->bg_counts[k]);
	}
	fit_data->n_neg_fi++;
      }
      fit_data->t_fi[k] = as_fi;
      di = as_fi - fit_data->as_xi[k];
      fit_data->err_i[k] = di*di;
      
      fit_data->stale[k] = 0;
    }
    err += fit_data->err_i[k];
  }
  peak->error = err;
  
  return 0;
}


/*
 * mFitCheck()
 *
 * Check that the parameters of working_peak are still valid.
 *
 * fit_data - pointer to a fitData structure.
 *
 * Return 0 if okay.
 */
int mFitCheck(fitData *fit_data)
{
  int xi,yi;
  peakData *peak;

  peak = fit_data->working_peak;  
  
  /*
   * Check that the peak hasn't moved to close to the edge of the image.
   */
  xi = peak->xi;
  yi = peak->yi;
  if((xi < 0)||(xi >= (fit_data->image_size_x - fit_data->fit_size_x))||(yi < 0)||(yi >= (fit_data->image_size_y - fit_data->fit_size_y))){
    fit_data->n_margin++;
    if(TESTING){
      printf("object outside margins, %d, %d, %d\n", peak->index, xi, yi);
    }
    return 1;
  }
  
  /* 
   * Check for negative height. 
   */
  if(peak->params[HEIGHT] < 0.0){
    fit_data->n_neg_height++;
    if(TESTING){
      printf("negative height, %d, %.3f\n", peak->index, peak->params[HEIGHT]);
    }
    return 1;
  }

  return 0;
}


/*
 * mfitCleanup()
 *
 * Frees the fitData structure.
 *
 * fit_data - pointer to a fitData structure.
 */
void mFitCleanup(fitData *fit_data)
{
  int i;
  
  /* 
   * Free individual peaks. Freeing of the fitter specialized parts
   * are handled by the fitter.
   */
  if(fit_data->fit != NULL){
    for(i=0;i<fit_data->max_nfit;i++){
      free(fit_data->fit[i].psf);
    }
    free(fit_data->fit);
  }

  free(fit_data->working_peak->psf);
  free(fit_data->working_peak);

  free(fit_data->bg_counts);
  free(fit_data->roi_x_index);
  free(fit_data->roi_y_index);
  free(fit_data->stale);
  
  free(fit_data->as_xi);
  free(fit_data->bg_data);
  free(fit_data->bg_estimate);
  free(fit_data->err_i);
  free(fit_data->f_data);
  free(fit_data->rqe);
  free(fit_data->scmos_term);
  free(fit_data->t_fi);
  free(fit_data->x_data);
  free(fit_data);
}


/*
 * mFitCopyPeak()
 *
 * Copy the contents of a peakData structure into another peakData 
 * structure. Note that the peak_model pointer is not copied as doing
 * this is the responsibility of the particular instantiation of the
 * the fitter.
 *
 * fit_data - pointer to a fitData structure.
 * original - pointer to a peakData structure.
 * copy - pointer to a peakData structure.
 */
void mFitCopyPeak(fitData *fit_data, peakData *original, peakData *copy)
{
  copy->added = original->added;
  copy->index = original->index;
  copy->iterations = original->iterations;
  copy->status = original->status;
  copy->xi = original->xi;
  copy->yi = original->yi;
  
  copy->error = original->error;

  copy->lambda = original->lambda;

  memcpy(copy->params, original->params, sizeof(double)*NFITTING);
  memcpy(copy->psf, original->psf, sizeof(double)*fit_data->roi_n_index);
}


/*
 * mFitDeltaConvergence()
 *
 * Check for fit convergence based on deltas of the peak fitting 
 * parameters. At least for now the scales of the different terms
 * are fixed by the tolerance parameter, which defaults to 1.0e-6.
 *
 * Returns 1 if converged, 0 otherwise.
 */
int mFitDeltaConvergence(fitData *fit_data, int index)
{
  double *old_params, *cur_params;

  old_params = (&fit_data->fit[index])->params;
  cur_params = fit_data->working_peak->params;

  /* 0.01 delta */
  if(fabs(old_params[HEIGHT] - cur_params[HEIGHT]) > (10000.0 * fit_data->tolerance)){
    return 0;
  }

  /* 0.01 delta */
  if(fabs(old_params[BACKGROUND] - cur_params[BACKGROUND]) > (10000.0 * fit_data->tolerance)){
    return 0;
  }

  /* 0.001 delta */  
  if(fabs(old_params[XCENTER] - cur_params[XCENTER]) > (1000.0 * fit_data->tolerance)){
    return 0;
  }

  /* 0.001 delta */    
  if(fabs(old_params[YCENTER] - cur_params[YCENTER]) > (1000.0 * fit_data->tolerance)){
    return 0;
  }

  /* 0.01 delta */    
  if(fabs(old_params[ZCENTER] - cur_params[ZCENTER]) > (100.0 * fit_data->tolerance)){
    return 0;
  }

  /* 0.001 delta */    
  if(fabs(old_params[XWIDTH] - cur_params[XWIDTH]) > (1000.0 * fit_data->tolerance)){
    return 0;
  }
  
  /* 0.001 delta */  
  if(fabs(old_params[YWIDTH] - cur_params[YWIDTH]) > (1000.0 * fit_data->tolerance)){
    return 0;
  }
  
  return 1;
}


/*
 * mFitEstimatePeakHeight()
 *
 * Calculate the area under the peak of unit height and compare this to
 * the area under (image - current fit - estimated background) x peak.
 *
 * We are minimizing : fi * (h*fi - xi)^2
 * where fi is the peak shape at pixel i, h is the height and xi is
 * is the data (minus the current fit & estimated background.
 *
 * Taking the derivative with respect to h gives fi*fi*xi/fi*fi*fi as the
 * value for h that will minimize the above.
 */
void mFitEstimatePeakHeight(fitData *fit_data)
{
  int i,j,k;
  double sp,sx,t1;
  peakData *peak;

  peak = fit_data->working_peak;

  i = peak->yi * fit_data->image_size_x + peak->xi;
  sp = 0.0;  /* This is fi*fi*fi. */
  sx = 0.0;  /* This is fi*fi*xi. */
  for(j=0;j<fit_data->roi_n_index;j++){
    k = fit_data->roi_y_index[j]*fit_data->image_size_x + fit_data->roi_x_index[j] + i;
    t1 = peak->psf[j];
    sp += t1*t1*t1;
    sx += t1*t1*(fit_data->x_data[k] - fit_data->f_data[k] - fit_data->bg_estimate[k]);
  }
  peak->params[HEIGHT] = sx/sp;

  if (VERBOSE){
    printf("mFEPH %.3f %.3f\n", sp, sx);
  }
}


/*
 * mFitGetFitImage()
 *
 * Return an image created from the current best fit peaks.
 */
void mFitGetFitImage(fitData *fit_data, double *fit_image)
{
  int i;
  
  for(i=0;i<(fit_data->image_size_x * fit_data->image_size_y);i++){
    fit_image[i] = fit_data->f_data[i];
  }
}


/*
 * mFitGetNError()
 *
 * Return the number of fits that are in the error state.
 *
 * fit_data - Pointer to a fitData structure.
 */
int mFitGetNError(fitData *fit_data)
{
  int i,count;

  count = 0;
  for(i=0;i<fit_data->nfit;i++){
    if(fit_data->fit[i].status==ERROR){
      count++;
    }
  }

  return count;
}


/*
 * mFitGetPeakPropertyDouble()
 *
 * Return requested peak property (double).
 *
 * fit_data - Pointer to a fitData structure.
 * values - Pre-allocated storage for the results.
 * what - Which property to get.
 */
void mFitGetPeakPropertyDouble(fitData *fit_data, double *values, char *what)
{
  int i;
  
  if (!strcmp(what, "background")){
    for(i=0;i<fit_data->nfit;i++){
      values[i] = fit_data->fit[i].params[BACKGROUND];
    }
  }
  else if (!strcmp(what, "bg_sum")){
    for(i=0;i<fit_data->nfit;i++){
      values[i] = mFitPeakBgSum(fit_data, &fit_data->fit[i]);
    }
  }
  else if (!strcmp(what, "error")){
    for(i=0;i<fit_data->nfit;i++){
      values[i] = fit_data->fit[i].error;
    }
  }
  else if (!strcmp(what, "fg_sum")){
    for(i=0;i<fit_data->nfit;i++){
      values[i] = mFitPeakFgSum(fit_data, &fit_data->fit[i]);
    }
  }
  else if (!strcmp(what, "height")){
    for(i=0;i<fit_data->nfit;i++){
      values[i] = fit_data->fit[i].params[HEIGHT];
    }
  }
  else if (!strcmp(what, "sum")){
    for(i=0;i<fit_data->nfit;i++){
      values[i] = mFitPeakSum(fit_data, &fit_data->fit[i]);
    }
  }
  else if (!strcmp(what, "x")){
    for(i=0;i<fit_data->nfit;i++){
      values[i] = fit_data->fit[i].params[XCENTER] + fit_data->xoff;
    }
  }
  else if (!strcmp(what, "xsigma")){
    for(i=0;i<fit_data->nfit;i++){
      values[i] = sqrt(1.0/(2.0*fit_data->fit[i].params[XWIDTH]));
    }
  }
  else if (!strcmp(what, "y")){
    for(i=0;i<fit_data->nfit;i++){
      values[i] = fit_data->fit[i].params[YCENTER] + fit_data->yoff;
    }
  }
  else if (!strcmp(what, "ysigma")){
    for(i=0;i<fit_data->nfit;i++){
      values[i] = sqrt(1.0/(2.0*fit_data->fit[i].params[YWIDTH]));
    }
  }
  else if (!strcmp(what, "z")){
    for(i=0;i<fit_data->nfit;i++){
      values[i] = fit_data->fit[i].params[ZCENTER] + fit_data->zoff;
    }
  }
  else{
    printf("Unrecognized parameter '%s'!\n", what);
  }
}


/*
 * mFitGetPeakPropertyInt()
 *
 * Return requested peak values as integers.
 *
 * fit_data - Pointer to a fitData structure.
 * values - Pre-allocated storage for the results.
 * what - Which property to get.
 */
void mFitGetPeakPropertyInt(fitData *fit_data, int32_t *values, char *what)
{
  int i;

  if (!strcmp(what, "iterations")){
    for(i=0;i<fit_data->nfit;i++){
      values[i] = fit_data->fit[i].iterations;
    }
  }
  else if (!strcmp(what, "status")){
    for(i=0;i<fit_data->nfit;i++){
      values[i] = fit_data->fit[i].status;
    }
  }
  else{
    printf("Unrecognized parameter '%s'!\n", what);
  }
}


/*
 * mFitGetResidual(residual).
 *
 * Returns image - fit.
 *
 * fit_data - Pointer to a fitData structure.
 * residual - Pre-allocated space to store the residual values.
 *            This should be square & the same size as the image.
 */
void mFitGetResidual(fitData *fit_data, double *residual)
{
  int i;

  //calcFit(fit_data);
  for(i=0;i<(fit_data->image_size_x * fit_data->image_size_y);i++){
    residual[i] = fit_data->x_data[i] - fit_data->f_data[i];
  }
}


/*
 * mFitGetUnconverged()
 *
 * Return the number of fits that have not yet converged.
 *
 * fit_data - Pointer to a fitData structure.
 */
int mFitGetUnconverged(fitData *fit_data)
{
  int i,count;

  count = 0;
  for(i=0;i<fit_data->nfit;i++){
    if(fit_data->fit[i].status==RUNNING){
      count++;
    }
  }

  return count;
}


/*
 * mFitInitialize()
 *
 * Initializes fitting things for fitting.
 *
 * rqe - Pixel relative quantum efficiency.
 * scmos_calibration - sCMOS calibration data, variance/gain^2 for each pixel in the image.
 * tol - The fitting tolerance.
 * im_size_x - size of the image in x.
 * im_size_y - size of the image in y.
 *
 * Returns - Pointer to the fitdata structure.
 */
fitData* mFitInitialize(double *rqe, double *scmos_calibration, double tol, int im_size_x, int im_size_y)
{
  int i;
  fitData* fit_data;
  
  /* Initialize fitData structure. */
  fit_data = (fitData*)malloc(sizeof(fitData));

  fit_data->n_dposv = 0;
  fit_data->n_iterations = 0;
  fit_data->n_lost = 0;
  fit_data->n_margin = 0;
  fit_data->n_neg_fi = 0;
  fit_data->n_neg_height = 0;
  fit_data->n_non_converged = 0;
  fit_data->n_non_decr = 0;

  fit_data->image_size_x = im_size_x;
  fit_data->image_size_y = im_size_y;
  fit_data->max_nfit = 0;
  fit_data->nfit = 0;
  fit_data->tolerance = tol;

  /*
   * These will get set by the fitting model when it initializes.
   */
  fit_data->fit_size_x = 0;
  fit_data->fit_size_y = 0;
  fit_data->roi_n_index = 0;
  fit_data->roi_x_index = NULL;
  fit_data->roi_y_index = NULL;
  
  /* 
   * The default behavior is to immediately ERROR out peaks that start 
   * with a negative height. However this is a problem for multi-plane
   * analysis where some peaks in a group could have negative heights 
   * due to noise and large z values, so multi-plane sets this to a 
   * small positive value.
   */
  fit_data->minimum_height = -1.0;
  
  fit_data->xoff = 0.0;
  fit_data->yoff = 0.0;
  fit_data->zoff = 0.0;

  /* Copy RQE data. */
  fit_data->rqe = (double *)malloc(sizeof(double)*im_size_x*im_size_y);
  for(i=0;i<(im_size_x*im_size_y);i++){
    fit_data->rqe[i] = rqe[i];
  }

  /* Copy sCMOS calibration data. */
  fit_data->scmos_term = (double *)malloc(sizeof(double)*im_size_x*im_size_y);
  for(i=0;i<(im_size_x*im_size_y);i++){
    fit_data->scmos_term[i] = scmos_calibration[i];
  }

  /* Allocate space for image, fit and background arrays. */
  fit_data->bg_counts = (int *)malloc(sizeof(int)*im_size_x*im_size_y);
  fit_data->stale = (int *)malloc(sizeof(int)*im_size_x*im_size_y);
  
  fit_data->as_xi = (double *)malloc(sizeof(double)*im_size_x*im_size_y);
  fit_data->bg_data = (double *)malloc(sizeof(double)*im_size_x*im_size_y);
  fit_data->bg_estimate = (double *)malloc(sizeof(double)*im_size_x*im_size_y);
  fit_data->err_i = (double *)malloc(sizeof(double)*im_size_x*im_size_y);
  fit_data->f_data = (double *)malloc(sizeof(double)*im_size_x*im_size_y);
  fit_data->t_fi = (double *)malloc(sizeof(double)*im_size_x*im_size_y);
  fit_data->x_data = (double *)malloc(sizeof(double)*im_size_x*im_size_y);

  /* Allocate space for the working peak. */
  fit_data->working_peak = (peakData *)malloc(sizeof(peakData));
  fit_data->working_peak->psf = NULL;
  fit_data->working_peak->peak_model = NULL;

  fit_data->fit = NULL;
  fit_data->fit_model = NULL;

  fit_data->fn_alloc_peaks = NULL;
  fit_data->fn_calc_JH = NULL;
  fit_data->fn_calc_peak_shape = NULL;
  fit_data->fn_check = NULL;
  fit_data->fn_copy_peak = NULL;
  fit_data->fn_error_fn = mFitCalcErr;
  fit_data->fn_free_peaks = NULL;
  fit_data->fn_update = NULL;
  
  return fit_data;
}


/*
 * mFitInitializeROIIndexing()
 *
 * Initializes the ROI indexing. The idea is to do fitting on an approximately circular
 * ROI instead of the usual square ROI. This will hopefully be less sensitive to noise 
 * and localization overlap as the points in the corners of the original square ROI are
 * now ignored. It should also be a little faster as there are fewer points in the ROI.
 *
 * Here we set the fitting area size, and create the arrays that we'll use to go from
 * ROI (1D) coordinate to an X,Y coordinate in the space of the image that is being
 * analyzed.
 *
 * fit_data - Pointer to a fitData structure.
 * roi_size - The size of the fitting ROI in pixels.
 */
void mFitInitializeROIIndexing(fitData *fit_data, int roi_size)
{
  int i,j,k;
  double cx,cy,dx,dy,rr;
  
  /* For now the ROI is always the same size in X/Y. */
  fit_data->fit_size_x = roi_size;
  fit_data->fit_size_y = roi_size;

  fit_data->roi_x_index = (int *)malloc(sizeof(int)*roi_size*roi_size);
  fit_data->roi_y_index = (int *)malloc(sizeof(int)*roi_size*roi_size);

  cx = 0.5*(double)roi_size;
  cy = 0.5*(double)roi_size;
  rr = cx*cx;
  
  k = 0;
  for(i=0;i<roi_size;i++){
    dy = (double)i - cy + 0.5;
    for(j=0;j<roi_size;j++){
      dx = (double)j - cx + 0.5;
      if((dx*dx + dy*dy)<=rr){
	fit_data->roi_x_index[k] = j;
	fit_data->roi_y_index[k] = i;
	k += 1;
      }
    }
  }
  fit_data->roi_n_index = k;
}


/*
 * mFitIterateLM
 *
 * Perform a single iteration of fitting update for each peaks.
 *
 * fit_data - Pointer to a fitData structure.
 */
void mFitIterateLM(fitData *fit_data)
{
  int i,j,k,l,m;
  int info;
  int n_add;

  double starting_error;               /* Initial error value for the peak. */
  
  double jacobian[NFITTING];           /* Jacobian */
  double w_jacobian[NFITTING];         /* Working copy of the Jacobian. */
  double hessian[NFITTING*NFITTING];   /* Hessian */
  double w_hessian[NFITTING*NFITTING]; /* Working copy of the Hessian. */

  if(VERBOSE){
    printf("mFILM\n");
  }

  for(i=0;i<fit_data->nfit;i++){

    if(VERBOSE){
      printf("\nmFILM peak - %d\n", i);
    }
    
    /* 
     * This is for debugging, to make sure that we not adding more times than
     * we are subtracting. 
     */
    n_add = 1;

    /* Skip ahead if this peak is not RUNNING. */
    if(fit_data->fit[i].status != RUNNING){
      continue;
    }

    /* Copy current peak into working peak. */
    fit_data->fn_copy_peak(fit_data, &fit_data->fit[i], fit_data->working_peak);

    /* 
     * Calculate initial error.
     *
     * Why? This might have changed from the previous cycle because the peak
     * background value could be shifted by neighboring peaks, creating a 
     * situation where it is impossible to improve on the old error value.
     */
    /* 
     * FIXME: We're ignoring the return value here and just assuming that
     *        the error function will work. Maybe okay because when we 
     *        calculate the updated error we check the return value? Not
     *        sure what would be the correct thing to do anyway.
     */
    fit_data->fn_error_fn(fit_data);
    starting_error = fit_data->working_peak->error;

    /* Calculate Jacobian and Hessian. This is expected to use 'working_peak'. */
    fit_data->fn_calc_JH(fit_data, jacobian, hessian);
    
    /* Subtract working peak out of image. */
    mFitSubtractPeak(fit_data);
    n_add--;

    j = 0;
    while(1){
      j++;

      if(VERBOSE){
	printf("  cycle %d %d\n", j, n_add);
      }

      /* Check if we are stuck on this peak, error it out if we are. */
      if(fit_data->working_peak->lambda > LAMBDAMAX){
	fit_data->n_lost++;
	fit_data->working_peak->status = ERROR;
	break;
      }
	
      /* Update peak iterations counter. */
      fit_data->working_peak->iterations++;
      
      /* Update total fitting iterations counter. */
      fit_data->n_iterations++;

      /*
       * Reset status flag. We only started this loop if the peak was RUNNING.
       * However the status might have been changed to error due to a Cholesky
       * solver issue or invalid peak parameters in a previous iteration of
       * this loop.
       */
      fit_data->working_peak->status = RUNNING;

      /* Copy Jacobian and Hessian. */
      for(k=0;k<fit_data->jac_size;k++){
	w_jacobian[k] = jacobian[k];
	m = k*fit_data->jac_size;
	for(l=0;l<fit_data->jac_size;l++){
	  if (k == l){
	    w_hessian[m+l] = (1.0 + fit_data->working_peak->lambda) * hessian[m+l];
	  }
	  else{
	    w_hessian[m+l] = hessian[m+l];
	  }
	}
      }
      
      /* 
       * Solve for update. Note that this also changes w_jacobian
       * which is one of the reasons why we made a copy.
       */
      info = mFitSolve(w_hessian, w_jacobian, fit_data->jac_size);
    
      if(info!=0){
	if(VERBOSE){
	  printf(" mFitSolve() failed %d\n", info);
	}
	fit_data->n_dposv++;
	fit_data->working_peak->status = ERROR;
	
	/* If the solver failed, try again with a larger lambda. */
        fit_data->working_peak->lambda = fit_data->working_peak->lambda * LAMBDAUP;
	continue;
      }
      
      /* Update 'working_peak'. mFitSolve returns the update in w_jacobian. */
      fit_data->fn_update(fit_data, w_jacobian);

      /* 
       * Check that it is still in the image, etc.. The fn_check function
       * should return 0 if everything is okay.
       */
      if(fit_data->fn_check(fit_data)){
	if(VERBOSE){
	  printf(" fn_check() failed\n");
	}
	/* 
	 * Try again with a larger lambda. We need to reset the 
	 * peak state because fn_update() changed it.
	 */
	mFitResetPeak(fit_data, i);

	/* Set status to ERROR in case this is the last iteration. */
	fit_data->working_peak->status = ERROR;
	
	continue;	
      }
      
      /* Add peak 'working_peak' back to fit image. */
      fit_data->fn_calc_peak_shape(fit_data);
      mFitAddPeak(fit_data);
      n_add++;

      /* 
       * Calculate error for 'working_peak' with the new parameters. This
       * will also check if the fit has converged. It will return 0 if
       * everything is okay (the fit image has no negative values).
       */
      if(fit_data->fn_error_fn(fit_data)){
	if(VERBOSE){
	  printf(" Error calculation failed\n");
	}
	/* Subtract 'working_peak' from the fit image. */
	mFitSubtractPeak(fit_data);
	n_add--;
	 
	/* 
	 * Try again with a larger lambda. We need to reset the peak 
	 * state because fn_update() and fn_add_peak() changed it.
	 */
	mFitResetPeak(fit_data, i);

	/* Set status to ERROR in case this is the last iteration. */
	fit_data->working_peak->status = ERROR;
	
	continue;	
      }
      
      /* Check whether the error improved. */
      if(fit_data->working_peak->error > starting_error){
	if(VERBOSE){
	  printf("    increasing error %.6e %.6e %.6e\n", fit_data->working_peak->error, starting_error, fit_data->working_peak->lambda);
	}

	/* 
	 * Check for error convergence. 
	 *
	 * Usually this will happen because the lambda term has gotten so 
	 * large that the peak will barely move in the update.
	 *
	 * FIXME: Why is this CONVERGED and not say ERROR?
	 */
	if(DELTA_CONVERGENCE){
	  if (mFitDeltaConvergence(fit_data, i)){
	    fit_data->working_peak->status = CONVERGED;
	    break;
	  }
	}
	else{
	  if (((fit_data->working_peak->error - starting_error)/starting_error) < fit_data->tolerance){
	    fit_data->working_peak->status = CONVERGED;
	    break;
	  }
	}

	fit_data->n_non_decr++;
	
	/* Subtract 'working_peak' from the fit image. */
	mFitSubtractPeak(fit_data);
	n_add--;
	
	/* 
	 * Try again with a larger lambda. We need to reset the 
	 * peak state because fn_update() changed it.
	 */
	mFitResetPeak(fit_data, i);
      }
      else{
	
	if(VERBOSE){
	  printf("    decreasing error %.6e %.6e %.6e\n", fit_data->working_peak->error, starting_error, fit_data->working_peak->lambda);
	}

	/* Check for error convergence. */
	if(DELTA_CONVERGENCE){
	  if (mFitDeltaConvergence(fit_data, i)){
	    fit_data->working_peak->status = CONVERGED;
	    break;
	  }	  
	}
	else{
	  if (((starting_error - fit_data->working_peak->error)/starting_error) < fit_data->tolerance){
	    fit_data->working_peak->status = CONVERGED;
	    break;
	  }
	}
	
	/* Decrease lambda. */
	if(fit_data->working_peak->lambda > LAMBDAMIN){
	  fit_data->working_peak->lambda = LAMBDADOWN * fit_data->working_peak->lambda;
	}

	/* Update successful, so exit while loop. */
	break;
      }
    }

    /* We expect n_add to be 1 if there were no errors, 0 otherwise. */
    if(TESTING){
      if(fit_data->working_peak->status == ERROR){
	if(n_add != 0){
	  printf("Problem detected in peak addition / subtraction logic, status == ERROR, counts = %d\n", n_add);
	  exit(EXIT_FAILURE);
	}
      }
      else{
	if(n_add != 1){
	  printf("Problem detected in peak addition / subtraction logic, status != ERROR, counts = %d\n", n_add);
	  exit(EXIT_FAILURE);
	}
      }
    }
    
    /* Copy updated working peak back into current peak. */
    fit_data->fn_copy_peak(fit_data, fit_data->working_peak, &fit_data->fit[i]);
  }

  /* Recenter peaks as necessary. */
  mFitRecenterPeaks(fit_data);
}


/*
 * mFitNewBackground
 *
 * Copy in a new estimate of the background.
 *
 * fit_data - Pointer to a fitData structure.
 * background - Pointer to the background data of size image_size_x by image_size_y.
 */
void mFitNewBackground(fitData *fit_data, double *background)
{
  int i;

  for(i=0;i<(fit_data->image_size_x*fit_data->image_size_y);i++){
    fit_data->bg_estimate[i] = background[i];
    fit_data->stale[i] = 1;
  }  
}


/*
 * mFitNewImage
 *
 * Copy in a new image to fit. We also reset everything that is fitting related
 * as this call indicates the start of a new cycle of fitting.
 *
 * fit_data - Pointer to a fitData structure.
 * new_image - Pointer to the image data of size image_size_x by image_size_y.
 * use_als - Doing Anscombe least-squares fitting.
 */
void mFitNewImage(fitData *fit_data, double *new_image, int use_als)
{
  int i;

  if(VERBOSE){
    printf("mFNI\n");
  }

  /* Copy the image & add scmos term (variance / gain * gain). */
  for(i=0;i<(fit_data->image_size_x*fit_data->image_size_y);i++){
    fit_data->x_data[i] = new_image[i] + fit_data->scmos_term[i];
  }
  
  if (use_als != 0){
    if (VERBOSE){
      printf("Using ALS fit\n");
    }
    for(i=0;i<(fit_data->image_size_x*fit_data->image_size_y);i++){
      fit_data->as_xi[i] = mFitAnscombe(fit_data->x_data[i]);
    }
  }

  /* Reset fitting arrays. */
  for(i=0;i<(fit_data->image_size_x*fit_data->image_size_y);i++){
    fit_data->bg_counts[i] = 0;
    fit_data->bg_data[i] = 0.0;
    fit_data->err_i[i] = 0.0;
    fit_data->f_data[i] = 0.0;
    fit_data->stale[i] = 1;
  }

  fit_data->nfit = 0;
}


/*
 * mFitNewPeaks
 *
 * These are new peaks to add to our current list of peaks.
 * 
 * fit_data - Pointer to a fitData structure.
 * n_peaks - The number of peaks.
 */
void mFitNewPeaks(fitData *fit_data, int n_peaks)
{
  int i,j,n_alloc,start,stop;
  peakData *peak,*new_peaks;

  if(VERBOSE){
    printf("mFNP %d\n", n_peaks);
  }
  
  /* 1. Check if we need more storage. */
  if ((fit_data->nfit + n_peaks) > fit_data->max_nfit){

    n_alloc = INCNPEAKS*((fit_data->nfit + n_peaks)/INCNPEAKS + 1);

    new_peaks = (peakData *)malloc(sizeof(peakData)*n_alloc);
    for(j=0;j<n_alloc;j++){
      new_peaks[j].psf = NULL;
    }
    fit_data->fn_alloc_peaks(new_peaks, n_alloc);

    i = 0;
    for(j=0;j<fit_data->nfit;j++){
      if(fit_data->fit[j].status != ERROR){
	fit_data->fn_copy_peak(fit_data, &fit_data->fit[j], &new_peaks[i]);
	i += 1;

	/* Check that this peak is in the image. */
	if(TESTING){
	  if(fit_data->fit[j].added == 0){
	    printf("Peak %d is not in the image.\n", j);
	    exit(EXIT_FAILURE);
	  }
	}
      }
      else{
	/* Check that this peak is not in the image. */
	if(TESTING){
	  if(fit_data->fit[j].added != 0){
	    printf("Peak %d is in error state, but still in the image.\n", j);
	    exit(EXIT_FAILURE);
	  }
	}
      }
    }
    
    /* Free old peak storage (if necessary). */
    if(fit_data->fit != NULL){
      fit_data->fn_free_peaks(fit_data->fit, fit_data->max_nfit);
      for(j=0;j<fit_data->max_nfit;j++){
	free(fit_data->fit[j].psf);
      }
      free(fit_data->fit);
    }

    /* Point to new peak storage and update counters. */
    fit_data->fit = new_peaks;
    fit_data->max_nfit = n_alloc;
    fit_data->nfit = i;
  }
  
  /* 2. Generic peak initialization. */
  start = fit_data->nfit;
  stop = fit_data->nfit + n_peaks;
  for(i=start;i<stop;i++){
    peak = &fit_data->fit[i];
    peak->added = 0;
    peak->index = i;
    peak->iterations = 0;

    /* Initial status. */
    peak->status = RUNNING;

    /* Initial error values. */
    peak->error = 0.0;

    /* Initial lambda value. */
    peak->lambda = LAMBDASTART;
  }
  
  /* 3. Caller must update the value of fit_data->nfit! */
}


/*
 * mFitPeakBgSum
 *
 * Return the background sum for the purposes of peak significance
 * calculations.
 *
 * This subtracts the fit shape off of the image. Then it multiplies
 * what is left by the PSF squared. The result is the number of
 * photo-electrons in the background if the background was filtered
 * by the PSF that describes the peak. Assuming Poisson statistics this
 * is also the variance. It is slightly complicated by the fact that
 * the PSF is not normalized.
 *
 * This is the same algorithm that is used in the Python peak finder
 * module for image segmentation.
 */
double mFitPeakBgSum(fitData *fit_data, peakData *peak)
{
  int i,j,k;
  double bg,bg_sum,fg,mag,psf,psf_sum;

  bg_sum = 0.0;
  psf_sum = 0.0;

  i = peak->yi * fit_data->image_size_x + peak->xi;
  mag = peak->params[HEIGHT];
  for(j=0;j<fit_data->roi_n_index;j++){
    k = fit_data->roi_y_index[j]*fit_data->image_size_x + fit_data->roi_x_index[j] + i;
    psf = peak->psf[j];
    psf_sum += psf;
    fg = mag * psf;

    /*
     * The background is the sum of the residual after subtracting the
     * foreground fit, and then convolved with the PSF.
     */
    bg = (fit_data->x_data[k] - fit_data->scmos_term[k]) - fg;
    bg_sum += bg*psf*psf;
  }

  if((bg_sum > 0.0)&&(psf_sum > 0.0)){
    
    /* Correct bg sum calculation for the PSF not being normalized. */
    bg_sum = bg_sum/(psf_sum*psf_sum);
  }
  else{
    if(TESTING){
      printf("mFPBS %.3f %.3f\n", bg_sum, psf_sum);
      printf("0 or negative background sum or PSF sum in peak background sum calculation.\n");
    }
    bg_sum = 1.0;
  }
  
  return bg_sum;
}


/*
 * mFitPeakFgSum
 *
 * Return the foreground sum for the purposes of peak significance
 * calculations.
 *
 * This convolves the fit peak shape with the PSF, mirroring the
 * calculation that is done to determine the background sum. It is
 * also slightly complicated by the fact that the PSF is not 
 * normalized.
 */
double mFitPeakFgSum(fitData *fit_data, peakData *peak)
{
  int i;
  double fg_sum,psf,psf_sum;

  fg_sum = 0.0;
  psf_sum = 0.0;

  for(i=0;i<fit_data->roi_n_index;i++){
    psf = peak->psf[i];
    psf_sum += psf;
    fg_sum += psf*psf;
  }
  fg_sum = peak->params[HEIGHT]*fg_sum;

  if(psf_sum > 0.0){
    
    /* Correct foreground sum calculation for the PSF not being normalized. */
    fg_sum = fg_sum/psf_sum;
  }
  else{
    fg_sum = 1.0;
    if(TESTING){
      printf("0 or negative psf sum in peak foreground sum calculation.\n");
    }
  }
  
  return fg_sum;
}


/*
 * mFitPeakSum()
 *
 * Return the integrated intensity of the requested peak.
 *
 * peak - the peak to sum.
 */
double mFitPeakSum(fitData *fit_data, peakData *peak)
{
  int i;
  double sum;

  sum = 0.0;
  for(i=0;i<fit_data->roi_n_index;i++){
    sum += peak->psf[i];
  }
  sum = sum * peak->params[HEIGHT];
  
  return sum;
}


/*
 * mFitRecenterPeaks()
 *
 * Recenter the ROIs of any RUNNING peaks that are not centered. This is
 * called after LM fitting to move the ROIs of peaks whose center has
 * moved too far from the ROI center. We don't force the center to be
 * exactly inside the pixel (i.e. between 0.0 - 1.0) as this causes
 * problems for peaks that are near the pixel boundary.
 */
void mFitRecenterPeaks(fitData *fit_data)
{
  int i;
  double dx,dy;
  peakData *peak;

  for(i=0;i<fit_data->nfit;i++){

    if(fit_data->fit[i].status != RUNNING){
      continue;
    }

    peak = &fit_data->fit[i];

    dx = peak->params[XCENTER] - peak->xi;
    dy = peak->params[YCENTER] - peak->yi;
    
    if((dx<-0.25)||(dx>1.25)||(dy<-0.25)||(dy>1.25)){

      /* Copy into working peak. */
      fit_data->fn_copy_peak(fit_data, peak, fit_data->working_peak);

      /* Subtract peak at current ROI. */
      mFitSubtractPeak(fit_data);

      /* Move the ROI. */
      fit_data->working_peak->xi = (int)floor(peak->params[XCENTER]);
      fit_data->working_peak->yi = (int)floor(peak->params[YCENTER]);

      /* 
       * Check that the ROI is still in the image. Mark it for removal if
       * it is not. 
       */
      if(mFitCheck(fit_data)){
	fit_data->n_lost += 1;
	fit_data->working_peak->status = ERROR;
      }
      else{
	fit_data->fn_calc_peak_shape(fit_data);
	mFitAddPeak(fit_data);
      }

      /* Copy working peak back. */
      fit_data->fn_copy_peak(fit_data, fit_data->working_peak, peak);
    }
  }
}

/*
 * mFitRemoveErrorPeaks()
 *
 * This removes all the peaks that are in the error state from the
 * peaks array. If none of the peaks are in the error state this
 * this is basically a NOP. If some are then there is a bit of
 * copying as it overwrites the error peaks with good peaks, shortening
 * the list of peaks in the process.
 *
 * The idea is that this is called after fitting and before you start
 * filtering out adjacent peaks, etc. in Python.
 */
void mFitRemoveErrorPeaks(fitData *fit_data)
{
  int i,j;
  
  i = 0;
  for(j=0;j<fit_data->nfit;j++){
    if(fit_data->fit[j].status != ERROR){
      if(j!=i){
	fit_data->fn_copy_peak(fit_data, &fit_data->fit[j], &fit_data->fit[i]);
      }
      i += 1;

      /* Check that this peak is in the image. */
      if(TESTING){
	if(fit_data->fit[j].added == 0){
	  printf("Peak %d is not in the image.\n", j);
	  exit(EXIT_FAILURE);
	}
      }
    }
    else{
      /* Check that this peak is not in the image. */
      if(TESTING){
	if(fit_data->fit[j].added > 0){
	  printf("Peak %d is in error state, but still in the image (%d).\n", j, fit_data->fit[j].added);
	  exit(EXIT_FAILURE);
	}
      }
    }
  }
  fit_data->nfit = i;
}


/*
 * mFitRemoveRunningPeaks()
 *
 * This removes all the peaks that are in the running state from the
 * peaks array. If none of the peaks are in the running state this
 * this is basically a NOP. If some are then there is a bit of
 * copying as it overwrites the running peaks with converged peaks, 
 * shortening the list of peaks in the process.
 *
 * The idea is that this is called at the end of the analysis before
 * you save the results.
 */
void mFitRemoveRunningPeaks(fitData *fit_data)
{
  int i,j;
  
  i = 0;
  for(j=0;j<fit_data->nfit;j++){
    if(fit_data->fit[j].status != RUNNING){
      if(j!=i){
	fit_data->fn_copy_peak(fit_data, &fit_data->fit[j], &fit_data->fit[i]);
      }
      i += 1;

      /* 
       * Check that this peak is not in the ERROR state. Peaks that
       * are in the error state should have been removed before this
       * function was called.
       */
      if(TESTING){
	if(fit_data->fit[j].status == ERROR){
	  printf("Peak %d is in the error state!\n", j);
	  exit(EXIT_FAILURE);
	}
      }
    }
    else{
      fit_data->n_non_converged++;
    }
  }
  fit_data->nfit = i;
}


/*
 * mFitResetPeak()
 *
 * This is used during fitting to restore the working peak to it's previous
 * state, but with increased lambda.
 */
void mFitResetPeak(fitData *fit_data, int index)
{
  int tmp_added;
  double tmp_lambda;

  tmp_added = fit_data->working_peak->added;
  tmp_lambda = fit_data->working_peak->lambda;
  fit_data->fn_copy_peak(fit_data, &fit_data->fit[index], fit_data->working_peak);
  fit_data->working_peak->added = tmp_added;
  fit_data->working_peak->lambda = tmp_lambda * LAMBDAUP;
}


/*
 * mFitSetPeakStatus()
 *
 * Set the status of the peaks.
 *
 * fit_data - Pointer to a fitData structure.
 * status - Integer array containing new status values.
 */
void mFitSetPeakStatus(fitData *fit_data, int32_t *status)
{
  int i;

  for(i=0;i<fit_data->nfit;i++){   
    /* 
     * If we marked this peak as ERROR and it was in the image we 
     * need to subtract it from the image.
     */
    if(status[i] == ERROR){
      if(VERBOSE){
	printf(" mFSPS %d %d\n", i, status[i]);
      }
      if(fit_data->fit[i].added > 0){
	fit_data->fn_copy_peak(fit_data, &fit_data->fit[i], fit_data->working_peak);
	mFitSubtractPeak(fit_data);
	fit_data->fn_copy_peak(fit_data, fit_data->working_peak, &fit_data->fit[i]);
      }
    }

    fit_data->fit[i].status = status[i];
  }
}


/*
 * mFitSolve
 *
 * Solve for update vector given jacobian and hessian.
 */
int mFitSolve(double *hessian, double *jacobian, int p_size)
{
  int i,j;
  
  /* Lapack */
  int n, nrhs = 1, lda, ldb, info;

  n = p_size;
  lda = p_size;
  ldb = p_size;

  /* Use Lapack to solve AX=B to calculate update vector. */
  dposv_("Lower", &n, &nrhs, hessian, &lda, jacobian, &ldb, &info);

  /* 
   * This can return NaNs with info = 0.. Need to catch this
   * and return as failure to solve.
   */
  if(info==0){
    for(i=0;i<p_size;i++){
      if(isnan(jacobian[i])){
	return 1;
      }
    }
  }
  
  if(VERBOSE){
    if(info!=0){
      printf(" dposv_ failed with %d\n", info);
      for(i=0;i<p_size;i++){
	printf("%.3f\t", jacobian[i]);
      }
      printf("\n\n");
      for(i=0;i<p_size;i++){
	for(j=0;j<p_size;j++){
	  printf("%.3f\t", hessian[i*p_size+j]);
	}
	printf("\n");
      }
      printf("\n");      
    }
  }
  
  return info;
}


/*
 * mFitSubtractPeak()
 *
 * Subtract working peak out of the current fit, basically 
 * this just undoes addPeak().
 *
 * fit_data - pointer to a fitData structure.
 */
void mFitSubtractPeak(fitData *fit_data)
{
  int i,j,k;
  double bg,mag,rqe;
  peakData *peak;

  peak = fit_data->working_peak;

  peak->added--;

  if(TESTING){
    if(peak->added != 0){
      printf("Peak count error detected in mFitSubtractPeak()! %d\n", peak->added);
      exit(EXIT_FAILURE);
    }
  }

  i = peak->yi * fit_data->image_size_x + peak->xi;
  bg = peak->params[BACKGROUND];
  mag = peak->params[HEIGHT];
  for(j=0;j<fit_data->roi_n_index;j++){
    k = fit_data->roi_y_index[j]*fit_data->image_size_x + fit_data->roi_x_index[j] + i;
    rqe = fit_data->rqe[k];
    fit_data->f_data[k] -= mag * peak->psf[j] * rqe;
    fit_data->bg_counts[k] -= 1;
    fit_data->bg_data[k] -= (bg * rqe + fit_data->scmos_term[k]);
    fit_data->stale[k] = 1;
  }
}


/*
 * mFitUpdate()
 *
 * Updates working_peak (integer) center. This is called in the 
 * LM loop. It moves the pixel location of the peak.
 *
 * peak - pointer to a peakData structure.
 */
void mFitUpdate(peakData *peak)
{
  double dx,dy;
  
  dx = peak->params[XCENTER] - peak->xi;
  if((dx<-0.5)||(dx>1.5)){
    peak->xi = (int)floor(peak->params[XCENTER]);
  }

  dy = peak->params[YCENTER] - peak->yi;
  if((dy<-0.5)||(dy>1.5)){
    peak->yi = (int)floor(peak->params[YCENTER]);
  }
}


/*
 * mFitUpdateParam
 *
 * Update peak parameter based on delta.
 */
void mFitUpdateParam(peakData *peak, double delta, int i)
{
  if(VERBOSE){
    printf("mFUP %d : %d %.3e %.3e\n", peak->index, i, peak->params[i], delta);
  }
  peak->params[i] -= delta;
}


/*
 * The MIT License
 *
 * Copyright (c) 2018 Zhuang Lab, Harvard University
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
