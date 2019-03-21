/*
 * Calculate cubic spline values and derivatives given spline coefficients.
 *
 * Notes:
 *
 *  1. Nominally xsize, ysize and zsize could be different, but this whether
 *     or not this works correctly has not been tested.
 *
 *  2. Unlike the Python version, this will not return the correct value if
 *     x is exactly at the maximum value that the spline covers.
 *
 *  3. This library is thread safe.. Pretty sure..
 *
 * Hazen 11/16
 *
 *
 * Compilation instructions:
 *
 * Linux:
 *  gcc -fPIC -g -c -Wall cubic_spline.c
 *  gcc -shared -Wl,-soname,cubic_spline.so.1 -o cubic_spline.so.1.0.1 cubic_spline.o -lc -lm
 *
 *  gcc -fPIC -g -c -Wall cubic_spline.c -o cubic_spline.o
 *
 *  ----- OR ---- Link statically
 *  ar rcs lib/libcubic_spline.a lib/cubic_spline.o
 *
 * macOS:
 *  gcc -shared -Wl,-install_name,testlib.so -o cubic_spline.so -fPIC cubic_spline.c
 *
 *  Lucas:
 *
 *  clang -o cubic_spline.o -c -O3 -Wall -I/usr/local/include -fPIC cubic_spline.c
 *  clang -o libcubic_spline.dylib -dynamiclib cubic_spline.o
 *
 *  ----- OR ---- Link statically
 *  ar rcs lib/libcubic_spline.a lib/cubic_spline.o
 *
 * Windows:
 *  gcc -c cubic_spline.c
 *  gcc -shared -o cubic_spline.dll cubic_spline.o
 */


/* Include */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//#include "multi_fit.h"

#include "cubic_spline.h"

/* Define */
#define RANGECHECK 1
#define RANGEWARN 0

/* Structures */

/* (Local) function declarations */

double dot(double *, double *, int);

double rangeCheckD(const char *, double, double, double);
int rangeCheckI(const char *, int, int, int);

/* Global variables */


/*
 * computeDelta2D()
 *
 * This computes the 16 cross-product values that you need to for
 * 2D splines and their derivatives.
 * i.e. 1,x,y,xx,xy,yy,..
 *
 * spline_data - Pointer to a spline data structure.
 * y_delta - The delta in y (0.0 - 1.0).
 * x_delta - The delta in x (0.0 - 1.0).
 */
void computeDelta2D(splineData *spline_data, double y_delta, double x_delta)
{
    int i,j;
    double cx,cy;
    
    if(RANGECHECK){
        x_delta = rangeCheckD("computeDelta2D,x_delta", x_delta, 0.0, 1.0);
        y_delta = rangeCheckD("computeDelta2D,y_delta", y_delta, 0.0, 1.0);
    }
    
    cx = 1.0;
    for(i=0;i<4;i++){
        cy = 1.0;
        for(j=0;j<4;j++){
            spline_data->delta_f[4*i+j] = cx * cy;
            if(i<3){
                spline_data->delta_dxf[4*(i+1)+j] = ((double)i+1) * cx * cy;
            }
            if(j<3){
                spline_data->delta_dyf[4*i+j+1] = ((double)j+1) * cx * cy;
            }
            cy = cy * y_delta;
        }
        cx = cx * x_delta;
    }
}

/*
 * computeDelta3D()
 *
 * This computes the 64 cross-product values that you need to for
 * 3D splines and their derivatives.
 * i.e. 1,x,y,z,xx,xy,yy,zx,zy,zz,..
 *
 * spline_data - Pointer to a spline data structure.
 * z_delta - The delta in z (0.0 - 1.0).
 * y_delta - The delta in y (0.0 - 1.0).
 * x_delta - The delta in x (0.0 - 1.0).
 */
void computeDelta3D(splineData *spline_data, double z_delta, double y_delta, double x_delta)
{
    int i,j,k;
    double cx,cy,cz;
    
    if(RANGECHECK){
        x_delta = rangeCheckD("computeDelta3D,x_delta", x_delta, 0.0, 1.0);
        y_delta = rangeCheckD("computeDelta3D,y_delta", y_delta, 0.0, 1.0);
        z_delta = rangeCheckD("computeDelta3D,z_delta", z_delta, 0.0, 1.0);
    }
    
    cx = 1.0;
    for(i=0;i<4;i++){
        cy = 1.0;
        for(j=0;j<4;j++){
            cz = 1.0;
            for(k=0;k<4;k++){
                spline_data->delta_f[i*16+j*4+k] = cx * cy * cz;
                if(i<3){
                    spline_data->delta_dxf[(i+1)*16+j*4+k] = ((double)i+1) * cx * cy * cz;
                }
                if(j<3){
                    spline_data->delta_dyf[i*16+(j+1)*4+k] = ((double)j+1) * cx * cy * cz;
                }
                if(k<3){
                    spline_data->delta_dzf[i*16+j*4+k+1] = ((double)k+1) * cx * cy * cz;
                }
                cz = cz * z_delta;
            }
            cy = cy * y_delta;
        }
        cx = cx * x_delta;
    }
}

/*
 * dot()
 *
 * Compute the dot product of two vectors.
 *
 * v1 - The first vector.
 * v2 - The second vector.
 * len - The number of elements.
 *
 * Returns the dot product.
 */
double dot(double *v1, double *v2, int len)
{
    int i;
    double pd;
    
    pd = 0.0;
    for (i=0;i<len;i++){
        pd += v1[i]*v2[i];
    }
    
    return pd;
}

/*
 * dxfAt2D()
 *
 * Compute the derivative of the spline in x at x,y coordinate (integer).
 * In order for this to work correctly computeDelta2D should have already
 * been called.
 *
 * spline_data - Pointer to a spline data structure.
 * yc - The y coordinate (integer).
 * xc - The x coordinate (integer).
 *
 * Return - The derivative in x.
 */
double dxfAt2D(splineData *spline_data, int yc, int xc)
{
    double yv;
    
    if(RANGECHECK){
        xc = rangeCheckI("dxfAt2D,xc", xc, 0, spline_data->xsize);
        yc = rangeCheckI("dxfAt2D,yc", yc, 0, spline_data->ysize);
    }
    
    yv = dot(&(spline_data->aij[(xc*spline_data->ysize+yc)*16]), spline_data->delta_dxf, 16);
    
    return yv;
}

/*
 * dxfAt3D()
 *
 * Compute the derivative of the spline in x at x,y,z coordinate (integer).
 * In order for this to work correctly computeDelta3D should have already
 * been called.
 *
 * spline_data - Pointer to a spline data structure.
 * zc - The z coordinate (integer).
 * yc - The y coordinate (integer).
 * xc - The x coordinate (integer).
 *
 * Return - The derivative in y.
 */
double dxfAt3D(splineData *spline_data, int zc, int yc, int xc)
{
    double yv;
    
    xc = rangeCheckI("dxfAt3D,xc", xc, 0, spline_data->xsize);
    yc = rangeCheckI("dxfAt3D,yc", yc, 0, spline_data->ysize);
    zc = rangeCheckI("dxfAt3D,zc", zc, 0, spline_data->zsize);
    
    yv = dot(&(spline_data->aij[(xc*(spline_data->ysize*spline_data->zsize)+yc*spline_data->zsize+zc)*64]), spline_data->delta_dxf, 64);
    
    return yv;
}

/*
 * dxfSpline2D()
 *
 * Return the derivative in x of the spline at coordinate x,y.
 *
 * spline_data - Pointer to a spline data structure.
 * y - The y coordinate (double).
 * x - The x coordinate (double).
 *
 * Return the derivative in x.
 */
double dxfSpline2D(splineData *spline_data, double y, double x)
{
    int xc,yc;
    double x_delta,y_delta,yv;
    
    if(RANGECHECK){
        x = rangeCheckD("dxfSpline2D,x", x, 0.0, (double)(spline_data->xsize));
        y = rangeCheckD("dxfSpline2D,y", y, 0.0, (double)(spline_data->ysize));
    }
    
    xc = (int)x;
    x_delta = x - (double)xc;
    
    yc = (int)y;
    y_delta = y - (double)yc;
    
    computeDelta2D(spline_data, y_delta, x_delta);
    yv = dxfAt2D(spline_data, yc, xc);
    return yv;
}

/*
 * dxfSpline3D()
 *
 * Return the derivative in x of the spline at coordinate x,y,z.
 *
 * spline_data - Pointer to a spline data structure.
 * z - The x coordinate (double).
 * y - The y coordinate (double).
 * x - The x coordinate (double).
 */
double dxfSpline3D(splineData *spline_data, double z, double y, double x)
{
    int xc,yc,zc;
    double x_delta,y_delta,z_delta,yv;
    
    if(RANGECHECK){
        x = rangeCheckD("dxfSpline3D,x", x, 0.0, (double)(spline_data->xsize));
        y = rangeCheckD("dxfSpline3D,y", y, 0.0, (double)(spline_data->ysize));
        z = rangeCheckD("dxfSpline3D,z", z, 0.0, (double)(spline_data->zsize));
    }
    
    xc = (int)x;
    x_delta = x - (double)xc;
    
    yc = (int)y;
    y_delta = y - (double)yc;
    
    zc = (int)z;
    z_delta = z - (double)zc;
    
    computeDelta3D(spline_data, z_delta, y_delta, x_delta);
    yv = dxfAt3D(spline_data, zc, yc, xc);
    return yv;
}

/*
 * dyfAt2D()
 *
 * Compute the derivative of the spline in y at x,y coordinate (integer).
 * In order for this to work correctly computeDelta2D should have already
 * been called.
 *
 * spline_data - Pointer to a spline data structure.
 * yc - The y coordinate (integer).
 * xc - The x coordinate (integer).
 *
 * Return - The derivative in y.
 */
double dyfAt2D(splineData *spline_data, int yc, int xc)
{
    double yv;
    
    if(RANGECHECK){
        xc = rangeCheckI("dyfAt2D,xc", xc, 0, spline_data->xsize);
        yc = rangeCheckI("dyfAt2D,yc", yc, 0, spline_data->ysize);
    }
    
    yv = dot(&(spline_data->aij[(xc*spline_data->ysize+yc)*16]), spline_data->delta_dyf, 16);
    
    return yv;
}

/*
 * dyfAt3D()
 *
 * Compute the derivative of the in spline y at x,y,z coordinate (integer).
 * In order for this to work correctly computeDelta3D should have already
 * been called.
 *
 * spline_data - Pointer to a spline data structure.
 * zc - The z coordinate (integer).
 * yc - The y coordinate (integer).
 * xc - The x coordinate (integer).
 *
 * Return - The derivative in y.
 */
double dyfAt3D(splineData *spline_data, int zc, int yc, int xc)
{
    double yv;
    
    if(RANGECHECK){
        xc = rangeCheckI("dyfAt3D,xc", xc, 0, spline_data->xsize);
        yc = rangeCheckI("dyfAt3D,yc", yc, 0, spline_data->ysize);
        zc = rangeCheckI("dyfAt3D,zc", zc, 0, spline_data->zsize);
    }
    
    yv = dot(&(spline_data->aij[(xc*(spline_data->ysize*spline_data->zsize)+yc*spline_data->zsize+zc)*64]), spline_data->delta_dyf, 64);
    
    return yv;
}

/*
 * dyfSpline2D()
 *
 * Return the derivative in y of the spline at coordinate x,y.
 *
 * spline_data - Pointer to a spline data structure.
 * y - The y coordinate (double).
 * x - The x coordinate (double).
 *
 * Return the derivative in y.
 */
double dyfSpline2D(splineData *spline_data, double y, double x)
{
    int xc,yc;
    double x_delta,y_delta,yv;
    
    if(RANGECHECK){
        x = rangeCheckD("dyfSpline2D,x", x, 0.0, (double)(spline_data->xsize));
        y = rangeCheckD("dyfSpline2D,y", y, 0.0, (double)(spline_data->ysize));
    }
    
    xc = (int)x;
    x_delta = x - (double)xc;
    
    yc = (int)y;
    y_delta = y - (double)yc;
    
    computeDelta2D(spline_data, y_delta, x_delta);
    yv = dyfAt2D(spline_data, yc, xc);
    return yv;
}

/*
 * dyfSpline3D()
 *
 * Return the derivative in y of the spline at coordinate x,y,z.
 *
 * spline_data - Pointer to a spline data structure.
 * z - The x coordinate (double).
 * y - The y coordinate (double).
 * x - The x coordinate (double).
 */
double dyfSpline3D(splineData *spline_data, double z, double y, double x)
{
    int xc,yc,zc;
    double x_delta,y_delta,z_delta,yv;
    
    if(RANGECHECK){
        x = rangeCheckD("dyfSpline3D,x", x, 0.0, (double)(spline_data->xsize));
        y = rangeCheckD("dyfSpline3D,y", y, 0.0, (double)(spline_data->ysize));
        z = rangeCheckD("dyfSpline3D,z", z, 0.0, (double)(spline_data->zsize));
    }
    
    xc = (int)x;
    x_delta = x - (double)xc;
    
    yc = (int)y;
    y_delta = y - (double)yc;
    
    zc = (int)z;
    z_delta = z - (double)zc;
    
    computeDelta3D(spline_data, z_delta, y_delta, x_delta);
    yv = dyfAt3D(spline_data, zc, yc, xc);
    return yv;
}

/*
 * dzfAt3D()
 *
 * Compute the derivative of the in spline z at x,y,z coordinate (integer).
 * In order for this to work correctly computeDelta3D should have already
 * been called.
 *
 * spline_data - Pointer to a spline data structure.
 * zc - The z coordinate (integer).
 * yc - The y coordinate (integer).
 * xc - The x coordinate (integer).
 *
 * Return - The derivative in y.
 */
double dzfAt3D(splineData *spline_data, int zc, int yc, int xc)
{
    double yv;
    
    if(RANGECHECK){
        xc = rangeCheckI("dzfAt3D,xc", xc, 0, spline_data->xsize);
        yc = rangeCheckI("dzfAt3D,yc", yc, 0, spline_data->ysize);
        zc = rangeCheckI("dzfAt3D,zc", zc, 0, spline_data->zsize);
    }
    
    yv = dot(&(spline_data->aij[(xc*(spline_data->ysize*spline_data->zsize)+yc*spline_data->zsize+zc)*64]), spline_data->delta_dzf, 64);
    
    return yv;
}

/*
 * dzfSpline3D()
 *
 * Return the derivative in z of the spline at coordinate x,y,z.
 *
 * spline_data - Pointer to a spline data structure.
 * z - The x coordinate (double).
 * y - The y coordinate (double).
 * x - The x coordinate (double).
 */
double dzfSpline3D(splineData *spline_data, double z, double y, double x)
{
    int xc,yc,zc;
    double x_delta,y_delta,z_delta,yv;
    
    if(RANGECHECK){
        x = rangeCheckD("dzfSpline3D,x", x, 0.0, (double)(spline_data->xsize));
        y = rangeCheckD("dzfSpline3D,y", y, 0.0, (double)(spline_data->ysize));
        z = rangeCheckD("dzfSpline3D,z", z, 0.0, (double)(spline_data->zsize));
    }
    
    xc = (int)x;
    x_delta = x - (double)xc;
    
    yc = (int)y;
    y_delta = y - (double)yc;
    
    zc = (int)z;
    z_delta = z - (double)zc;
    
    computeDelta3D(spline_data, z_delta, y_delta, x_delta);
    yv = dzfAt3D(spline_data, zc, yc, xc);
    return yv;
}

/*
 * fAt2D()
 *
 * Compute the spline at x,y coordinate (integer). In order
 * for this to work correctly computeDelta2D should have
 * already been called.
 *
 * spline_data - Pointer to a spline data structure.
 * yc - The y coordinate (integer).
 * xc - The x coordinate (integer).
 */
double fAt2D(splineData *spline_data, int yc, int xc)
{
    double yv;
    
    if(RANGECHECK){
        xc = rangeCheckI("fAt2D,xc", xc, 0, spline_data->xsize);
        yc = rangeCheckI("fAt2D,yc", yc, 0, spline_data->ysize);
    }
    
    yv = dot(&(spline_data->aij[(xc*spline_data->ysize+yc)*16]), spline_data->delta_f, 16);
    
    return yv;
}

/*
 * fAt3D()
 *
 * Compute the spline at x,y,z coordinate (integer). In order
 * for this to work correctly computeDelta3D should have
 * already been called.
 *
 * spline_data - Pointer to a spline data structure.
 * zc - The z coordinate (integer).
 * yc - The y coordinate (integer).
 * xc - The x coordinate (integer).
 */
double fAt3D(splineData *spline_data, int zc, int yc, int xc)
{
    double yv;
    
    if(RANGECHECK){
        xc = rangeCheckI("fAt3D,xc", xc, 0, spline_data->xsize);
        yc = rangeCheckI("fAt3D,yc", yc, 0, spline_data->ysize);
        zc = rangeCheckI("fAt3D,zc", zc, 0, spline_data->zsize);
    }
    
    yv = dot(&(spline_data->aij[(xc*(spline_data->ysize*spline_data->zsize)+yc*spline_data->zsize+zc)*64]), spline_data->delta_f, 64);
    
    return yv;
}



/*
 * fSpline2D()
 *
 * Return the spline value at coordinate x,y.
 *
 * spline_data - Pointer to a spline data structure.
 * y - The y coordinate (double).
 * x - The x coordinate (double).
 */
double fSpline2D(splineData *spline_data, double y, double x)
{
    int xc,yc;
    double x_delta,y_delta,yv;
    
    if(RANGECHECK){
        x = rangeCheckD("fSpline2D,x", x, 0.0, (double)(spline_data->xsize));
        y = rangeCheckD("fSpline2D,y", y, 0.0, (double)(spline_data->ysize));
    }
    
    xc = (int)x;
    x_delta = x - (double)xc;
    
    yc = (int)y;
    y_delta = y - (double)yc;
    
    computeDelta2D(spline_data, y_delta, x_delta);
    yv = fAt2D(spline_data, yc, xc);
    return yv;
}

/*
 * fSpline3D()
 *
 * Return the spline value at coordinate x,y,z.
 *
 * spline_data - Pointer to a spline data structure.
 * z - The z coordinate (double).
 * y - The y coordinate (double).
 * x - The x coordinate (double).
 */
double fSpline3D(splineData *spline_data, double z, double y, double x)
{
    int xc,yc,zc;
    double x_delta,y_delta,z_delta,yv;
    
    if(RANGECHECK){
        x = rangeCheckD("fSpline3D,x", x, 0.0, (double)(spline_data->xsize));
        y = rangeCheckD("fSpline3D,y", y, 0.0, (double)(spline_data->ysize));
        z = rangeCheckD("fSpline3D,z", z, 0.0, (double)(spline_data->zsize));
    }
    
    xc = (int)x;
    x_delta = x - (double)xc;
    
    yc = (int)y;
    y_delta = y - (double)yc;
    
    zc = (int)z;
    z_delta = z - (double)zc;
    
    computeDelta3D(spline_data, z_delta, y_delta, x_delta);
    yv = fAt3D(spline_data, zc, yc, xc);
    return yv;
}


/**
 Compute Image
 
 @param spline_data spline_data structure
 @param img image
 @param x_size image size in x
 @param y_size image size in y
 @param zc position of the emitter in z
 @param yc position of the emitter in y
 @param xc position of the emitter in x
 @param N number of photons
 */
void imgSpline3D(splineData *spline_data, double* img, int x_size, int y_size,
                 double zc, double yc, double xc, double N) {
    
    /** Query points. Note that the delta is the same for all points on the grid. **/
    int x0,y0,z0;
    double x_delta,y_delta,z_delta;
    
    /* Shift query points by 0 reference ix; shift by coordinate of upper left px (usually 0, 0)*/
    xc = 0 - xc + spline_data->x0_ix - spline_data->img_pos_x0;
    yc = 0 - yc + spline_data->y0_ix - spline_data->img_pos_y0;
    zc = zc / spline_data->dz + spline_data->z0_ix;
    
    x0 = (int)(floor(xc));
    y0 = (int)(floor(yc));
    z0 = (int)(floor(zc));
    
    x_delta = xc - (double)x0;
    y_delta = yc - (double)y0;
    z_delta = zc - (double)z0;
    
    computeDelta3D(spline_data, z_delta, y_delta, x_delta);
    
    //    int c = 0;
    for (int i = 0; i < x_size; i++) {
        for (int j = 0; j < y_size; j++){
            img[i * y_size + j] += N * fAt3D(spline_data, z0, y0 + j, x0 + i);
        }
    }
}

void kernel_computeDelta3D_yim(float x_delta, float y_delta, float z_delta, float *delta_f) {
    
    int i,j,k;
    float cx,cy,cz;
    
    cz = 1.0;
    for(i=0;i<4;i++){
        cy = 1.0;
        for(j=0;j<4;j++){
            cx = 1.0;
            for(k=0;k<4;k++){
                delta_f[i*16+j*4+k] = cz * cy * cx;
                cx = cx * x_delta;
            }
            cy = cy * y_delta;
        }
        cz= cz * z_delta;
    }
}

float fAt3Dj_yim2(int xc, int yc, int zc, int xsize, int ysize, int zsize, float*delta_f, float *coeff) {
    int i;
    float temp =0;
    
    xc = fmax(xc,0);
    xc = fmin(xc,xsize-1);
    
    yc = fmax(yc,0);
    yc = fmin(yc,ysize-1);
    
    zc = fmax(zc,0);
    zc = fmin(zc,zsize-1);
    
    
    
    for (i=0;i<64;i++){
        temp+=delta_f[i]*coeff[i*(xsize*ysize*zsize)+zc*(xsize*ysize)+yc*xsize+xc];
    }
    return temp;
}

float fSpline3D(float xc, float yc, float zc, float *coeff) {
    
    /** Query points. Note that the delta is the same for all points on the grid. **/
    int x0,y0,z0;
    float x_delta,y_delta,z_delta;
    float delta_f[64] = {0};
    
    x0 = (int)floor(xc);
    x_delta = xc - x0;
    
    y0 = (int)floor(yc);
    y_delta = yc - y0;
    
    z0 = (int)floor(zc);
    z_delta = zc - z0;
    
    kernel_computeDelta3D_yim(x_delta, y_delta, z_delta, delta_f);
    float f = fAt3Dj_yim2(x0, y0, z0, 26, 26, 30, delta_f, coeff);
    return f;
}

/**
 Compute Image -- after yimming.
 */
void imgSpline3D_yim(splineData *spline_data, double* img,
                 int x_size, int y_size,
                 double zcenter, double ycenter, double xcenter, double N) {
    
    /** Query points. Note that the delta is the same for all points on the grid. **/
    int x0,y0,z0;
    double xc, yc, zc;
    double x_delta,y_delta,z_delta;

    /* Shift query points by 0 reference ix; shift by coordinate of upper left px (usually 0, 0)*/
//    xc = 0 - xc + spline_data->x0_ix - spline_data->pos_x0;
//    yc = 0 - yc + spline_data->y0_ix - spline_data->pos_y0;
//    zc = zc + spline_data->z0_ix;
    const int Npixels = x_size;
    const int offset = (int)(floor(((spline_data->xsize + 1) - Npixels) / 2.0));

    xc = -1.0 * ((xcenter - ((double)Npixels) / 2) + 0.5);
    yc = -1.0 * ((ycenter - ((double)Npixels) / 2) + 0.5);
    zc = zcenter + spline_data->z0_ix;

    x0 = floor(xc);
    x_delta = xc - x0;

    y0 = floor(yc);
    y_delta = yc - y0;

    z0 = floor(zcenter);
    z_delta = zc - z0;

    kernel_computeDelta3D_yim(x_delta, y_delta, z_delta, spline_data->delta_f);

    for (int i = 0; i < x_size; i++) {
        for (int j = 0; j < y_size; j++) {
            img[y_size * j + i] += N * fAt3Dj_yim(x0 + i + offset, y0 + j + offset, z0, spline_data, spline_data->delta_f, spline_data->aij);
        }
    }
}

/*
 * gradSpline3D()
 *
 * Return None. Build gradient.
 *
 */
void gradSpline3D(splineData *spline_data, double *zg, double *yg, double *xg, int img_dim,
                  double zc, double yc, double xc, double * yq, double * xq) {
    
    for (int i = 0; i < img_dim; i++) {
        xg[i] = dxfSpline3D(spline_data, zc, (yq[i] - yc), (xq[i] - xc));
        yg[i] = dyfSpline3D(spline_data, zc, (yq[i] - yc), (xq[i] - xc));
        zg[i] = dzfSpline3D(spline_data, zc, (yq[i] - yc), (xq[i] - xc));
    }
}


/*
 * getXSize()
 *
 * Returns the x size of the current spline.
 *
 * spline_data - Pointer to a spline data structure.
 */
int getXSize(splineData *spline_data)
{
    return spline_data->xsize;
}

/*
 * getYSize()
 *
 * Returns the y size of the current spline.
 *
 * spline_data - Pointer to a spline data structure.
 */
int getYSize(splineData *spline_data)
{
    return spline_data->ysize;
}

/*
 * getZSize()
 *
 * Returns the z size of the current spline.
 *
 * spline_data - Pointer to a spline data structure.
 */
int getZSize(splineData *spline_data)
{
    return spline_data->zsize;
}

/*
 * initSpline2D()
 *
 * Initialize the spline coefficient values.
 *
 * new_aij - The spline values.
 * new_xsize - The number of cells in x.
 * new_ysize - The number of cells in y.
 */
splineData* initSpline2D(double *new_aij, int new_xsize, int new_ysize)
{
    int i, tsize;
    splineData *spline_data;
    
    tsize = new_xsize*new_ysize*16;
    spline_data =(splineData *)malloc(sizeof(splineData));
    
    spline_data->type = S2D;
    
    spline_data->xsize = new_xsize;
    spline_data->ysize = new_ysize;
    spline_data->zsize = 0;
    
    /*
     * Allocate storage for spline data.
     *
     * delta_dzf is not actually used, but we allocated it anyway to
     * keep the cleanup() function simple.
     */
    spline_data->aij = (double *)malloc(sizeof(double)*tsize);
    spline_data->delta_f = (double *)malloc(sizeof(double)*16);
    spline_data->delta_dxf = (double *)malloc(sizeof(double)*16);
    spline_data->delta_dyf = (double *)malloc(sizeof(double)*16);
    spline_data->delta_dzf = (double *)malloc(sizeof(double));
    
    /* Copy spline coefficients. */
    for (i=0;i<tsize;i++){
        spline_data->aij[i] = new_aij[i];
    }
    
    /* Initialize delta arrays. */
    for (i=0;i<16;i++){
        spline_data->delta_f[i] = 0.0;
        spline_data->delta_dxf[i] = 0.0;
        spline_data->delta_dyf[i] = 0.0;
    }
    
    return spline_data;
}

/*
 * initSpline3D()
 *
 * Initialize the spline coefficient values.
 *
 * new_aij - The spline values.
 * new_xsize - The number of cells in x.
 * new_ysize - The number of cells in y.
 * new_zsize - The number of cells in z.
 */
splineData* initSpline3D(double *new_aij,
                         int new_xsize, int new_ysize, int new_zsize,
                         const int x0_ix, const int y0_ix, const int z0_ix,
                         const double pos_x0, const double pos_y0, const double dz)
{
    int i, tsize;
    splineData *spline_data;
    
    tsize = new_xsize*new_ysize*new_zsize*64;
    spline_data =(splineData *)malloc(sizeof(splineData));
    
    spline_data->type = S3D;
    
    spline_data->xsize = new_xsize;
    spline_data->ysize = new_ysize;
    spline_data->zsize = new_zsize;
    
    
    /* Denote ix (x,y) of bright point (reference point). If wrong this induces a constant shift, most likely harmless.
     * In z, this denotes the focal plane.
     * Note that we allow for non-integer 'indices' because we might know the ground truth position and want to shift exactly.
     */
    spline_data->x0_ix = x0_ix;
    spline_data->y0_ix = y0_ix;
    spline_data->z0_ix = z0_ix;
    
    /* Position of upper left px of the image (px center). Useful for psf rendering */
    spline_data->img_pos_x0 = pos_x0;
    spline_data->img_pos_y0 = pos_y0;
    
    /* Delta z in nm between slices */
    spline_data->dz = dz;
    
    /* Allocate storage for spline data. */
    spline_data->aij = (double *)malloc(sizeof(double)*tsize);
    spline_data->delta_f = (double *)malloc(sizeof(double)*64);
    spline_data->delta_dxf = (double *)malloc(sizeof(double)*64);
    spline_data->delta_dyf = (double *)malloc(sizeof(double)*64);
    spline_data->delta_dzf = (double *)malloc(sizeof(double)*64);
    
    /* Copy spline coefficients. */
    for (i=0;i<tsize;i++){
        spline_data->aij[i] = new_aij[i];
    }
    
    /* Initialize delta arrays. */
    for (i=0;i<64;i++){
        spline_data->delta_f[i] = 0.0;
        spline_data->delta_dxf[i] = 0.0;
        spline_data->delta_dyf[i] = 0.0;
        spline_data->delta_dzf[i] = 0.0;
    }
    
    return spline_data;
}

/*
 * rangeCheckD()
 *
 * Checks whether the variable is in the expected range.
 * Prints a warning if it is not (double version).
 *
 * fname - The name of the current function.
 * value - The variable value.
 * vmin - The minimum expected variable value.
 * vmax - The maximum expected variable value.
 */
double rangeCheckD(const char *fname, double value, double vmin, double vmax)
{
    if (value < vmin){
        if ((RANGEWARN)||(TESTING)){
            printf("Double value out of range %f (%f) in %s\n", value, vmin, fname);
            if(TESTING){
                exit(EXIT_FAILURE);
            }
        }
        value = vmin;
    }
    if (value > vmax){
        if ((RANGEWARN)||(TESTING)){
            printf("Double value out of range %f (%f) in %s\n", value, vmax, fname);
            if(TESTING){
                exit(EXIT_FAILURE);
            }
        }
        value = vmax;
    }
    
    return value;
}

/*
 * rangeCheckI()
 *
 * Checks whether the variable is in the expected range.
 * Prints a warning if it is not (integer version).
 *
 * fname - The name of the current function.
 * value - The variable value.
 * vmin - The minimum expected variable value.
 * vmax - The maximum expected variable value.
 */
int rangeCheckI(const char *fname, int value, int vmin, int vmax)
{
    if (value < vmin){
        if ((RANGEWARN)||(TESTING)){
            printf("Int value out of range %d (%d) in %s\n", value, vmin, fname);
            if(TESTING){
                exit(EXIT_FAILURE);
            }
        }
        value = vmin;
    }
    if (value >= vmax){
        if ((RANGEWARN)||(TESTING)){
            printf("Int value out of range %d (%d) in %s\n", value, vmax, fname);
            if(TESTING){
                exit(EXIT_FAILURE);
            }
        }
        value = vmax - 1;
    }
    
    return value;
}

/*
 * splineCleanup()
 *
 * Free the allocated storage.
 *
 * spline_data - Pointer to a spline data structure.
 */
void splineCleanup(splineData *spline_data)
{
    if (spline_data->aij != NULL){
        free(spline_data->aij);
    }
    if (spline_data->delta_f != NULL){
        free(spline_data->delta_f);
        free(spline_data->delta_dxf);
        free(spline_data->delta_dyf);
        free(spline_data->delta_dzf);
    }
    free(spline_data);
}
