#ifndef TESTSUITE_H_
#define TESTSUITE_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_GpuDevice.H>

#include <memory>
#include <string>

#include <LGFSolverFactory.H>
#include <SourceField.H>

/**
 * POISSON TEST SUITE
 *
 * One problem: an unbounded Poisson solve with a compactly supported source,
 * the source tagged by boxes and the answer wanted on a layout grown around
 * the tags. Every case is that problem with a different source (SourceField.H),
 * run for each requested index type with the backend named by lgf.solver.
 *
 *   loadTestCase(name)       one case
 *   loadTestSuite(names)     several, back to back; "all" is every case
 *
 * PER CASE (buildCaseLayout)
 *
 *   1. the source is evaluated on a background layout covering the domain
 *      and tagSource() picks the boxes holding it (D_supp);
 *   2. layout = snug: the solve layout is D_supp grown by n_buffer_box
 *      boxes, overlaps removed and re-chopped, as DomainManager builds
 *      D_soln, then clipped to the domain. layout = full: the background.
 *
 * PER INDEX TYPE (runTestCase)
 *
 *   source and target on the solve layout convert()ed to the type; the
 *   solver is built (timed), then solvePoisson() runs n_iter times (each
 *   timed). The first solve carries the backend's setup, the rest are the
 *   repeated cost:
 *
 *     setup = t_build + t_first - mean(t_repeat)
 *
 *   With reuse_solver one solver serves the whole run and regridOnto()
 *   replaces the build from the second case on: the NSE solver's pattern.
 *
 * CHECKS
 *
 *   residual  max|f_tag - L_h u| / max|f_tag| over the valid cells, f_tag
 *             being f restricted to the tagged boxes: the problem posed
 *   error     max|u - u_ref| / max|u_ref|, ghosts included, against
 *             reference_solver (none: skipped)
 *   drift     max|u_last - u_first| / max|u_first|: repeated solves must
 *             agree, or something cached is going stale
 *   each against its tolerance; one failure fails the case.
 *
 * TIMING
 *
 *   timedCall(): ranks start together, the device is drained before the
 *   clock stops, and the slowest rank's time is what counts.
 */

// one test case: which source, and what it is there to test
struct TestCase
{
    std::string name;
    SourceCase  source;
    std::string description;
};

// every input of a bench run, read once by readBenchConfig()
struct BenchConfig
{
    int n_cell = 0;
    int max_grid_size = 0;
    int n_lookup = 0;
    int n_ghost = 1;
    amrex::Array<amrex::Real, AMREX_SPACEDIM> domain_lo{};
    amrex::Array<amrex::Real, AMREX_SPACEDIM> domain_hi{};
    amrex::Real tagging_threshold = 0.0;

    amrex::Vector<std::string> case_names;
    amrex::Vector<std::string> index_type_names;
    amrex::Vector<amrex::IndexType> index_types;

    std::string solver_name;          // lgf.solver
    std::string reference_name;       // reference_solver; "none" skips the error check
    std::string layout;               // snug | full
    int n_buffer_box = 2;
    int n_iter = 1;
    bool reuse_solver = false;

    amrex::Real tol_error = 1.0e-8;
    amrex::Real tol_residual = 1.0e-6;
    amrex::Real tol_drift = 1.0e-12;

    bool compare_MLMG = false;
    bool write_plot = false;
    std::string plot_prefix;
    std::string summary_file;         // empty: no csv
};

// the tags and the solve layout of one case
struct CaseLayout
{
    amrex::BoxArray supp_ba;          // tagged boxes of the background, cell centred
    amrex::BoxArray soln_ba;          // the layout solved on, cell centred
    amrex::DistributionMapping soln_dm;
    amrex::Long n_supp_cells = 0;
    amrex::Long n_soln_cells = 0;
    amrex::Long n_bbox_cells = 0;     // cells in the solve layout's bounding box
};

// what one (case, index type) measured
struct CaseResult
{
    std::string case_name;
    std::string type_name;
    std::string solver_name;
    amrex::Long n_supp_cells = 0;
    amrex::Long n_soln_cells = 0;
    amrex::Long n_bbox_cells = 0;
    int n_iter = 0;

    amrex::Real t_build = 0.0;        // construction, or regridOnto() with reuse_solver
    amrex::Real t_first = 0.0;
    amrex::Real t_repeat_mean = 0.0;  // the rest: zero when n_iter = 1
    amrex::Real t_repeat_min = 0.0;
    amrex::Real t_repeat_max = 0.0;

    amrex::Real residual = 0.0;
    amrex::Real error = -1.0;         // < 0: no reference
    amrex::Real drift = 0.0;
    bool is_pass = false;

    // build + first solve - repeated solve, never below zero
    amrex::Real setupTime() const;
};

// the cases
amrex::Vector<TestCase> testCatalogue();
TestCase loadTestCase(const std::string& name);
amrex::Vector<TestCase> loadTestSuite(const amrex::Vector<std::string>& names);

// cell | node | face_x | face_y | face_z | edge_x | edge_y | edge_z (faces and
// edges named by the direction they are normal to and run along)
amrex::IndexType indexTypeFromName(const std::string& name);

// the run
BenchConfig readBenchConfig();
CaseLayout buildCaseLayout(const TestCase& test_case, const BenchConfig& config, const amrex::Geometry& geom,
                           const amrex::BoxArray& background_ba, const amrex::DistributionMapping& background_dm);
// every index type of one case; shared_solver is used, and kept, with reuse_solver
amrex::Vector<CaseResult> runTestCase(const TestCase& test_case, const BenchConfig& config, const amrex::Geometry& geom,
                                      const CaseLayout& layout, std::unique_ptr<LGFPoissonSolver>& shared_solver);

// reporting
void printSummary(const amrex::Vector<CaseResult>& results, const BenchConfig& config);
// appends one csv row per result, with a header when the file is new
void writeSummary(const amrex::Vector<CaseResult>& results, const BenchConfig& config);

// max|a - b| / max|b| over the valid cells and n_ghost ghost cells
amrex::Real relativeDifference(const amrex::MultiFab& a, const amrex::MultiFab& b, int n_ghost);

// wall time of f(), as the slowest rank saw it. Ranks start together, and
// the device is drained before the clock stops, so asynchronous kernels are
// counted where they run
template <typename F>
amrex::Real timedCall(F&& f)
{
    amrex::ParallelDescriptor::Barrier();
    const amrex::Real t_start = amrex::second();

    f();

    amrex::Gpu::streamSynchronize();
    amrex::Real t_elapsed = amrex::second() - t_start;
    amrex::ParallelDescriptor::ReduceRealMax(t_elapsed);

    return t_elapsed;
}

#endif
