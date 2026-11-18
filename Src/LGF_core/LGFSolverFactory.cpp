#include <LGFSolverFactory.H>

std::unique_ptr<LGFPoissonSolver> makeLGFPoissonSolver(const amrex::Geometry& geom, const amrex::BoxArray& xsoln_ba,
                                                       int n_lookup, int n_ghost_max)
{
    std::string solver_name = "fft";
    int len_quantum = 1;
    int n_block = 16;
    int n_interp = 10;

    amrex::ParmParse pp("lgf");
    pp.query("solver", solver_name);
    pp.query("fft_len_quantum", len_quantum);
    pp.query("fmm_block_size", n_block);
    pp.query("fmm_n_interp", n_interp);

    if (solver_name == "fft")
    {
        amrex::Print() << "LGF Poisson solver: fft (LGFOpenBC), len_quantum " << len_quantum << "\n";
        return std::make_unique<LGFOpenBC>(geom, xsoln_ba, n_lookup, n_ghost_max, len_quantum);
    }

    if (solver_name == "fmm")
    {
        amrex::Print() << "LGF Poisson solver: fmm (LGFFastMultipole), block size " << n_block
                       << ", n_interp " << n_interp << "\n";
        return std::make_unique<LGFFastMultipole>(geom, xsoln_ba, n_lookup, n_ghost_max, n_block, n_interp);
    }

    if (solver_name == "direct")
    {
        amrex::Print() << "LGF Poisson solver: direct (DirectSumLGF), O(N^2)\n";
        return std::make_unique<DirectSumLGF>(geom, n_lookup);
    }

    amrex::Abort("lgf.solver = '" + solver_name + "' is not one of fft, fmm, direct");
    return nullptr;
}
