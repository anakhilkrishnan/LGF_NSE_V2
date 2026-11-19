#include <MyFunctions.H>

using namespace amrex;

void initializeSourceMultiFab(amrex::MultiFab& phi_mf, const amrex::Geometry& phi_geom, SourceCase source_case)
{
    // adding profiling blocks for Tiny/Base profilers
    BL_PROFILE("<Setup> initializeSourceMultiFab");

    // extracting physical dx, physical domain lo for computing x,y,z
    GpuArray<amrex::Real, AMREX_SPACEDIM> dx = phi_geom.CellSizeArray();
    GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = phi_geom.ProbLoArray();

    // index i sits at prob_lo + (i + offset) * dx: offset 1/2 along cell-centred
    // directions and 0 along nodal ones, so staggered data is evaluated where it lives
    GpuArray<amrex::Real, AMREX_SPACEDIM> offset;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        offset[d] = phi_mf.ixType().cellCentered(d) ? 0.5 : 0.0;
    }

    // ghosts are zeroed here and filled by FillBoundary() below; LGFPoissonSolver's contract
    phi_mf.setVal(0.0);

#ifdef AMREX_USE_OMP
    #pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(phi_mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& vbx = mfi.tilebox();
        auto const& phiarr = phi_mf.array(mfi);
        ParallelFor(vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                amrex::ignore_unused(j, k);

                // physical coordinates of point (i,j,k) of this index type
                AMREX_D_TERM(const amrex::Real x = prob_lo[0] + (i + offset[0]) * dx[0];,
                             const amrex::Real y = prob_lo[1] + (j + offset[1]) * dx[1];,
                             const amrex::Real z = prob_lo[2] + (k + offset[2]) * dx[2];)
#if AMREX_SPACEDIM == 2
                const amrex::Real z = 0.0;
#endif

                phiarr(i,j,k) = sourceField(source_case, x, y, z);
            });
    }

    phi_mf.FillBoundary(phi_geom.periodicity());
}

void syncBCs(amrex::MultiFab& phi_mf_dest, amrex::MultiFab& phi_mf_src, amrex::Geometry& phi_geom, int n_ghost)
{
    // adding profiling blocks for Tiny/Base profilers
    BL_PROFILE("<Setup> Extracting FMM Boundaries for MLMG");
        
    for (MFIter mfi(phi_mf_dest, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) 
    {
        const Box& gbx = mfi.growntilebox(n_ghost);
        const Box& vbx = mfi.validbox(); // The strict interior box
        auto const& mlmg_arr = phi_mf_dest.array(mfi);
        auto const& fmm_arr = phi_mf_src.array(mfi);
        
        ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) 
        {
            // Only populate the ghost cells (the Dirichlet boundaries)
            // The interior remains 0.0 to force MLMG to solve it
            if (!vbx.contains(i, j, k)) {
                mlmg_arr(i,j,k) = fmm_arr(i,j,k);
            }
        });
    }
}