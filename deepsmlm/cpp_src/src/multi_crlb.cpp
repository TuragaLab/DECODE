#include <iostream>
#include <stdio.h>
#include <vector>
#include <array>
#include <Eigen/Eigen>

#include "multi_crlb.hpp"

// using namespace Eigen;

/**
 * Output format: x y phot bg z
 * 
 */
auto construct_multi_fisher(spline *sp, 
                            std::vector<std::array<float, 3>> xyz, 
                            std::vector<float> phot,
                            std::vector<float> bg,
                            std::array<float, 2> corner,
                            int npx, float *img, float *fisher_flat) -> void {
    
    const int n_emitters = phot.size();
    const int n_par = sp->NV_PSP * n_emitters;
    const int npxpx = npx * npx;
    std::fill_n(img, npx * npx, 0.0);
    std::fill_n(fisher_flat, sp->NV_PSP * n_emitters * sp->NV_PSP * n_emitters, 0.0);

    #if DEBUG
        std::cout << "Before 3D init." << std::endl;
    #endif // DEBUG

    // fisher px-wise
    float* fisher_px_ma = new float[n_par * n_par * npxpx]();

    #if DEBUG
        std::cout << "Finished initialising fisher block px flat." << std::endl;
    #endif // DEBUG
    
    // derivatives
    std::vector<float*> deriv_px_em; // derivatives of emitters by px

    for (int i = 0; i < n_emitters; i++) {
        // initialise derivatives
        deriv_px_em.push_back((float *)malloc(sizeof(float) * sp->NV_PSP * npx * npx));
        std::fill_n(deriv_px_em[i], sp->NV_PSP * npx * npx, 0.0);

        // calculate the derivatives for all emitters (and eventually also get the model for free)
        f_derivative_PSF(sp, img, deriv_px_em[i], npx, xyz[i][0], xyz[i][1], xyz[i][2], corner[0], corner[1], phot[i], bg[i]);

    }

    #if DEBUG
        std::cout << "Finished calc. derivates" << std::endl;
    #endif // DEBUG

    // flatten and expand px derivatives
    float deriv_px_em_total[n_par * npx * npx];
    std::fill_n(deriv_px_em_total, n_par * npx * npx, 0.0);
    for (int i = 0; i < n_emitters; i++) {
        for (int j = 0; j < sp->NV_PSP; j++) {
            for (int p = 0; p < npx * npx; p++) {
                deriv_px_em_total[i * sp->NV_PSP * npx * npx + j * npx * npx + p] = deriv_px_em[i][j * npx * npx + p];
            }
        }
    }

    #if DEBUG
        std::cout << "Finished flattening and expansion." << std::endl;
    #endif // DEBUG

    // fill the fisher px-wise
    for (int j = 0; j < n_par; j++) {  // parameter rows
        for (int k = 0; k < n_par; k++) { // parameter cols
            for (int p = 0; p < npx * npx; p++) { // px
                fisher_px_ma[j * n_par * npx * npx + k * npx * npx + p] = deriv_px_em_total[j * npx * npx + p] * deriv_px_em_total[k * npx * npx + p] / img[p];
            }
        }
    }

    #if DEBUG
        std::cout << "Finished filling the px-wise fisher." << std::endl;
    #endif // DEBUG

    // flatten out the fisher by aggregating the sample dimension
    for (int i = 0; i < n_par; i++) {
        for (int j = 0; j < n_par; j++) {
            for (int p = 0; p < npxpx; p++) {
                fisher_flat[i * n_par + j] += fisher_px_ma[i * n_par * npxpx + j * npxpx + p];
            }
        }
    }

    delete[] fisher_px_ma;

    #if DEBUG
        std::cout << "Finished aggregating to normal fisher." << std::endl;
    #endif // DEBUG

}

/** output format x y phot bg z
 * 
 */
auto calc_crlb(spline *sp, 
            std::vector<std::array<float, 3>> xyz, 
            std::vector<float> phot,
            std::vector<float> bg,
            std::array<float, 2> corner,
            int npx, float *img, float *crlb) -> void {
    
    int n_emitter = phot.size();
    float fisher_blockmat[sp->NV_PSP * n_emitter * sp->NV_PSP * n_emitter];

    construct_multi_fisher(sp, xyz, phot, bg, corner, npx, img, fisher_blockmat);

    // map the matrices
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> hessian_block(fisher_blockmat, sp->NV_PSP * n_emitter, sp->NV_PSP * n_emitter);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> crlb_(crlb, sp->NV_PSP * n_emitter, 1);
    
    // crlb_ = hessian_block.completeOrthogonalDecomposition().pseudoInverse().diagonal();
    // crlb_ = hessian_block.fullPivLu().inverse().diagonal();
    crlb_ = hessian_block.inverse().diagonal();

    #if DEBUG
        std::cout << "Hessian: \n" << hessian_block << std::endl;
        std::cout << "CRLB is: \n" << crlb_ << std::endl;
    #endif
}