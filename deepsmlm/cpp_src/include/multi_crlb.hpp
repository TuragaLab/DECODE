#ifndef multi_crlb_hpp
#define multi_crlb_hpp

extern "C" {
    #include "spline_psf.h"
}

auto construct_multi_fisher(spline *sp, 
                            std::vector<std::array<float, 3>> xyz, 
                            std::vector<float> phot,
                            std::vector<float> bg,
                            std::array<float, 2> corner,
                            int npx, float *img, float *hessian_blockmat) -> void;

auto calc_crlb(spline *sp, 
            std::vector<std::array<float, 3>> xyz, 
            std::vector<float> phot,
            std::vector<float> bg,
            std::array<float, 2> corner,
            int npx, float *img, float *crlb) -> void;

#endif /* multi_crlb_hpp */