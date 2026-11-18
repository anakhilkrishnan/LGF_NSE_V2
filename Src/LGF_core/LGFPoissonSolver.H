#ifndef LGFPOISSONSOLVER_H_
#define LGFPOISSONSOLVER_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Math.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_BLProfiler.H>

/**
 * Free-space LGF Poisson solve, u = h^2 * G * f, shared by every backend.
 *
 * CONTRACT, THE SAME FOR EVERY BACKEND
 *
 *   - source and target share an index type and a DistributionMapping, and
 *     are built from the layout the backend was constructed (or last
 *     regridded) on: one cell-centred BoxArray, convert()ed to the index
 *     type of the solve.
 *   - tag_ba is cell-centred and names the boxes of that layout that hold
 *     source; convert(tag_ba, type) is the support for a given index type.
 *     The source is treated as zero everywhere else.
 *   - the source has at least one ghost cell, zeroed or filled by
 *     FillBoundary() before the solve: a backend may read it where
 *     neighbouring boxes share a plane (LGFOpenBC does, for cell-centred
 *     directions).
 *   - every target cell, ghost cells included, is overwritten.
 *
 * All index types go through solvePoisson(); a backend that cannot handle
 * one aborts there.
 */
class LGFPoissonSolver
{
public:
    virtual ~LGFPoissonSolver() = default;

    virtual void solvePoisson(const amrex::MultiFab& source, amrex::MultiFab& target,
                              const amrex::BoxArray& tag_ba) = 0;

    // rebuild any cached state after the snug domain changes. Backends that
    // hold nothing across solves (e.g. direct summation) need do nothing;
    // FFT and FMM backends rebuild their layout-dependent caches here.
    virtual void regridOnto(const amrex::Geometry& new_geom, const amrex::BoxArray& new_ba,
                            const amrex::DistributionMapping& new_dm)
    {
        amrex::ignore_unused(new_geom, new_ba, new_dm);
    }
};

#endif
