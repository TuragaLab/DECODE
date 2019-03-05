/*
 * API functions.
 *
 * Hazen 10/16
 *
 */

#ifndef CUBIC_SPLINE_H
#define CUBIC_SPLINE_H

/* Define */
#define S2D 0
#define S3D 1

/* debugging */
#define TESTING 0
#define VERBOSE 0

/* Spline Structure */
typedef struct{
    int type;
    int xsize;
    int ysize;
    int zsize;
    double x0_ix;
    double y0_ix;
    double z0_ix;
    double img_pos_x0;
    double img_pos_y0;
    double dz;
    double *aij;
    double *delta_f;
    double *delta_dxf;
    double *delta_dyf;
    double *delta_dzf;
} splineData;

/* Function Declarations */
void computeDelta2D(splineData *, double, double);
void computeDelta3D(splineData *, double, double, double);

double dxfAt2D(splineData *, int, int);
double dxfAt3D(splineData *, int, int, int);
double dxfSpline2D(splineData *,double, double);
double dxfSpline3D(splineData *,double, double, double);

double dyfAt2D(splineData *, int, int);
double dyfAt3D(splineData *, int, int, int);
double dyfSpline2D(splineData *, double, double);
double dyfSpline3D(splineData *, double, double, double);

double dzfAt3D(splineData *, int, int, int);
double dzfSpline3D(splineData *, double, double, double);

double fAt2D(splineData *, int, int);
double fAt3D(splineData *, int, int, int);
double fSpline2D(splineData *, double, double);
double fSpline3D(splineData *, double, double, double);
void imgSpline3D(splineData *, double*, int , int, double, double, double, double);
void gradSpline3D(splineData *, double *, double *, double *, int, double, double, double,
                  double *, double *);

int getXSize(splineData *);
int getYSize(splineData *);
int getZSize(splineData *);

splineData* initSpline2D(double *, int, int);
splineData* initSpline3D(double *, int, int, int, int, int, int, double, double, double);

void splineCleanup(splineData *);

#endif
