#ifndef LGFSOLVERFACTORY_H_
#define LGFSOLVERFACTORY_H_

#include <AMReX_ParmParse.H>

#include <memory>
#include <string>

#include <LGFPoissonSolver.H>
#include <DirectSumLGF.H>
#include <LGFOpenBC.H>
#include <LGFFastMultipole.H>

/**
 * Builds the Poisson backend named in the inputs file, so that neither the
 * NSE solver nor the test bench needs to know any backend's parameters.
 *
 *   lgf.solver           fft | fmm | direct      default fft
 *   lgf.fft_len_quantum  LGFOpenBC: solver lengths rounded up to a
 *                        multiple of this                     default 1
 *   lgf.fmm_block_size   LGFFastMultipole: lattice points per block and
 *                        direction                            default 16
 *   lgf.fmm_n_interp     LGFFastMultipole: nodes per interpolation
 *                                                             default 10
 *
 * With no lgf.* entries this is LGFOpenBC exactly as the NSE solver built
 * it before, so existing inputs files run unchanged. 'direct' is O(N^2)
 * and meant for checking the others on small problems.
 */
std::unique_ptr<LGFPoissonSolver> makeLGFPoissonSolver(const amrex::Geometry& geom, const amrex::BoxArray& xsoln_ba,
                                                       int n_lookup, int n_ghost_max);

#endif
