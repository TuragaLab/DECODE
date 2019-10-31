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
    
    int n_emitters = phot.size();
    int n_par = sp->NV_PSP * n_emitters;
    std::fill_n(img, npx * npx, 0.0);
    std::fill_n(fisher_flat, sp->NV_PSP * n_emitters * sp->NV_PSP * n_emitters, 0.0);

    // fisher px-wise
    float fisher_block_px_flat[sp->NV_PSP * n_emitters][sp->NV_PSP * n_emitters][npx * npx];

    for (int i = 0; i < sp->NV_PSP * n_emitters; i++) {
        for (int j = 0; j < sp->NV_PSP * n_emitters; j++) {
            for (int p = 0; p < npx * npx; p++) {
                fisher_block_px_flat[i][j][p] = 0.0;
            }
        }
    }
    
    // derivatives
    std::vector<float*> deriv_px_em; // derivatives of emitters by px

    for (int i = 0; i < n_emitters; i++) {
        // initialise derivatives
        deriv_px_em.push_back((float *)malloc(sizeof(float) * sp->NV_PSP * npx * npx));
        std::fill_n(deriv_px_em[i], sp->NV_PSP * npx * npx, 0.0);

        // calculate the derivatives for all emitters (and eventually also get the model for free)
        f_derivative_PSF(sp, img, deriv_px_em[i], npx, xyz[i][0], xyz[i][1], xyz[i][2], corner[0], corner[1], phot[i], bg[i]);

    }

    // flatten and expand px derivatives
    // float deriv_px_em_total[n_par];
    // std::fill_n(deriv_px_em_total, n_par, 0.0);
    // for (int i = 0; i < n_emitters; i++) {
    //     for (int j = 0; j < sp->NV_PSP; j++) {
    //         deriv_px_em_total[i * sp->NV_PSP + j] = deriv_px_em[i][j];
    //     }
    // }

    // fill the fisher px-wise
    for (int i = 0; i < n_emitters; i++) {  // emitter block
        for (int j = 0; j < sp->NV_PSP; j++) {  // parameter rows
            for (int k = 0; k < sp->NV_PSP; k++) { // parameter cols
                for (int p = 0; p < npx * npx; p++) { // px

                    int block_ix = i * sp->NV_PSP;;
                    fisher_block_px_flat[block_ix + j][block_ix + k][p] = deriv_px_em[i][j * npx * npx + p] * deriv_px_em[i][k * npx * npx + p] / img[p];
                }
            }
        }
    }

    // flatten out the fisher by aggregating the sample dimension
    int n_rows = sp->NV_PSP * n_emitters, n_cols = sp->NV_PSP * n_emitters;
    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            for (int p = 0; p < npx * npx; p++) {
                fisher_flat[i * n_cols + j] += fisher_block_px_flat[i][j][p];
            }
        }
    }

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
    crlb_ = hessian_block.inverse().diagonal();

    #if DEBUG
        std::cout << "Hessian: \n" << hessian_block << std::endl;
        std::cout << "CRLB is: \n" << crlb_ << std::endl;
    #endif
}