#include <MyFunctions.H>

amrex::Real CaseResult::setupTime() const
{
    if (n_iter < 2) { return t_build + t_first; }

    return amrex::max(amrex::Real(0.0), t_build + t_first - t_repeat_mean);
}

amrex::Vector<TestCase> testCatalogue()
{
    return {
        {"single_blob",  SourceCase::single_blob,  "one Gaussian, off-lattice centre: the plain monopole"},
        {"dipole",       SourceCase::dipole,       "two opposite Gaussians: zero net charge, dipole far field"},
        {"blob_cluster", SourceCase::blob_cluster, "32 blobs in the unit ball, nonzero net charge"},
        {"ring",         SourceCase::ring,         "Gaussian-core torus, vortex-ring shaped support"},
        {"shell",        SourceCase::shell,        "Gaussian shell: large, hollow support"},
        {"distant_pair", SourceCase::distant_pair, "two small clusters in opposite corners: sparse layout"}
    };
}

TestCase loadTestCase(const std::string& name)
{
    const amrex::Vector<TestCase> catalogue = testCatalogue();

    for (const TestCase& test_case : catalogue)
    {
        if (test_case.name == name) { return test_case; }
    }

    std::string known;
    for (const TestCase& test_case : catalogue) { known += " " + test_case.name; }
    amrex::Abort("test_cases: unknown case '" + name + "'; known cases are" + known + ", or all");

    return catalogue[0];
}

amrex::Vector<TestCase> loadTestSuite(const amrex::Vector<std::string>& names)
{
    if (names.size() == 1 && names[0] == "all") { return testCatalogue(); }

    amrex::Vector<TestCase> suite;
    for (const std::string& name : names) { suite.push_back(loadTestCase(name)); }

    return suite;
}

amrex::IndexType indexTypeFromName(const std::string& name)
{
    if (name == "cell") { return amrex::IndexType::TheCellType(); }
    if (name == "node") { return amrex::IndexType::TheNodeType(); }

    const std::string axis = "xyz";
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        // a face is nodal along its normal; an edge is cell centred along itself, nodal across
        if (name == std::string("face_") + axis[d])
        {
            return amrex::IndexType(amrex::IntVect::TheDimensionVector(d));
        }
#if AMREX_SPACEDIM == 3
        if (name == std::string("edge_") + axis[d])
        {
            return amrex::IndexType(amrex::IntVect(1) - amrex::IntVect::TheDimensionVector(d));
        }
#endif
    }

    amrex::Abort("index_types: unknown index type '" + name + "'; use cell, node, face_x, ...");
    return amrex::IndexType::TheCellType();
}

BenchConfig readBenchConfig()
{
    BenchConfig config;
    amrex::ParmParse pp;

    // inputs that changed meaning; caught rather than silently ignored
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!pp.contains("index_type"),
        "index_type is replaced by index_types = cell node face_x ... (a list)");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!pp.contains("solver_type"),
        "solver_type is replaced by lgf.solver = fft | fmm | direct");

    pp.get("n_cell", config.n_cell);
    pp.get("max_grid_size", config.max_grid_size);
    pp.get("domain_lo", config.domain_lo);
    pp.get("domain_hi", config.domain_hi);
    pp.get("tagging_threshold", config.tagging_threshold);
    pp.get("n_lookup", config.n_lookup);

    config.case_names = {"blob_cluster"};
    pp.queryarr("test_cases", config.case_names);

    config.index_type_names = {"cell"};
    pp.queryarr("index_types", config.index_type_names);
    for (const std::string& name : config.index_type_names)
    {
        config.index_types.push_back(indexTypeFromName(name));
    }

    config.reference_name = "none";
    config.layout = "snug";
    pp.query("reference_solver", config.reference_name);
    pp.query("layout", config.layout);
    pp.query("n_buffer_box", config.n_buffer_box);
    pp.query("n_iter", config.n_iter);
    pp.query("reuse_solver", config.reuse_solver);

    pp.query("tol_error", config.tol_error);
    pp.query("tol_residual", config.tol_residual);
    pp.query("tol_drift", config.tol_drift);

    config.plot_prefix = "./Results/plt";
    pp.query("compare_MLMG", config.compare_MLMG);
    pp.query("write_plot", config.write_plot);
    pp.query("plot_prefix", config.plot_prefix);
    pp.query("summary_file", config.summary_file);

    config.solver_name = "fft";
    amrex::ParmParse pp_lgf("lgf");
    pp_lgf.query("solver", config.solver_name);

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(config.n_iter >= 1, "n_iter must be at least 1");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(config.n_buffer_box >= 0, "n_buffer_box must not be negative");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(config.layout == "snug" || config.layout == "full",
        "layout must be snug or full");

    return config;
}

CaseLayout buildCaseLayout(const TestCase& test_case, const BenchConfig& config, const amrex::Geometry& geom,
                           const amrex::BoxArray& background_ba, const amrex::DistributionMapping& background_dm)
{
    BL_PROFILE("<Setup> buildCaseLayout()");

    CaseLayout layout;

    // the source over the whole background, cell centred, only to tag it;
    // freed before anything is solved
    {
        amrex::MultiFab background(background_ba, background_dm, 1, config.n_ghost);
        initializeSourceMultiFab(background, geom, test_case.source);
        tagSource(layout.supp_ba, background, config.tagging_threshold);
    }

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!layout.supp_ba.empty(),
        "buildCaseLayout: no box tagged -- is tagging_threshold too high?");

    if (config.layout == "full")
    {
        layout.soln_ba = background_ba;
    }
    else
    {
        // D_soln as DomainManager::growBoxArr builds it from D_supp: grown by
        // n_buffer_box boxes, overlap removed, re-chopped. Clipped to the
        // domain here, which the NSE solver's search domain guarantees itself
        amrex::BoxArray soln_ba = layout.supp_ba;
        soln_ba.grow(config.n_buffer_box * config.max_grid_size);
        soln_ba.removeOverlap(true);
        soln_ba = amrex::intersect(soln_ba, geom.Domain());
        soln_ba.maxSize(config.max_grid_size);

        layout.soln_ba = std::move(soln_ba);
    }

    layout.soln_dm = amrex::DistributionMapping(layout.soln_ba);
    layout.n_supp_cells = layout.supp_ba.numPts();
    layout.n_soln_cells = layout.soln_ba.numPts();
    layout.n_bbox_cells = layout.soln_ba.minimalBox().numPts();

    return layout;
}

amrex::Real relativeDifference(const amrex::MultiFab& a, const amrex::MultiFab& b, int n_ghost)
{
    amrex::MultiFab diff(a.boxArray(), a.DistributionMap(), 1, n_ghost);
    amrex::MultiFab::Copy(diff, a, 0, 0, 1, n_ghost);
    amrex::MultiFab::Subtract(diff, b, 0, 0, 1, n_ghost);

    const amrex::Real b_max = b.norm0(0, n_ghost);

    return (b_max > 0.0) ? diff.norm0(0, n_ghost) / b_max : diff.norm0(0, n_ghost);
}

namespace
{
    // fixed-width cell of the summary table; "-" for a value that does not apply
    std::string summaryCell(amrex::Real value, bool is_fixed, int width)
    {
        std::ostringstream cell;
        cell << std::setw(width);
        if (value < 0.0) { cell << "-"; return cell.str(); }

        if (is_fixed) { cell << std::fixed << std::setprecision(1) << value; }
        else          { cell << std::scientific << std::setprecision(2) << value; }

        return cell.str();
    }

    // 1 where the tags cover a point of the given index type, 0 elsewhere,
    // ghost cells included
    void fillTagRegion(amrex::MultiFab& tag_region, const amrex::BoxArray& supp_ba)
    {
        const amrex::BoxArray keep = amrex::convert(supp_ba, tag_region.ixType());
        std::vector<std::pair<int, amrex::Box>> isects;

        tag_region.setVal(0.0);
        for (amrex::MFIter mfi(tag_region); mfi.isValid(); ++mfi)
        {
            keep.intersections(mfi.fabbox(), isects);
            for (auto const& is : isects)
            {
                tag_region[mfi].setVal<amrex::RunOn::Device>(1.0, is.second, 0, 1);
            }
        }
    }

    void writeCasePlot(const BenchConfig& config, const CaseResult& result, const amrex::Geometry& geom,
                       const amrex::MultiFab& source, const amrex::MultiFab& target, const amrex::MultiFab& residual,
                       const amrex::BoxArray& supp_ba, const amrex::MultiFab* target_mlmg)
    {
        BL_PROFILE("<I/O> writingPlotfile");

        const int n_plot_comp = (target_mlmg != nullptr) ? 5 : 4;
        amrex::MultiFab plot_mf(target.boxArray(), target.DistributionMap(), n_plot_comp, 0);
        amrex::MultiFab tag_region(target.boxArray(), target.DistributionMap(), 1, 0);
        fillTagRegion(tag_region, supp_ba);

        amrex::MultiFab::Copy(plot_mf, source, 0, 0, 1, 0);
        amrex::MultiFab::Copy(plot_mf, target, 0, 1, 1, 0);
        amrex::MultiFab::Copy(plot_mf, tag_region, 0, 2, 1, 0);
        amrex::MultiFab::Copy(plot_mf, residual, 0, 3, 1, 0);

        amrex::Vector<std::string> varnames = {"Source_Phi", "Target_Phi", "Active_Box_Tag", "Residual"};
        if (target_mlmg != nullptr)
        {
            amrex::MultiFab::Copy(plot_mf, *target_mlmg, 0, 4, 1, 0);
            varnames.push_back("MLMG_Target_Phi");
        }

        // n_cell stays the only number in the name, for postProcess.py
        const std::string plotfile_name = amrex::Concatenate(config.plot_prefix + "_" + result.case_name + "_"
                                                             + result.type_name + "_", config.n_cell);
        WriteSingleLevelPlotfile(plotfile_name, plot_mf, varnames, geom, 0.0, 0);
        amrex::Print() << "    plotfile written to " << plotfile_name << "\n";
    }
}

amrex::Vector<CaseResult> runTestCase(const TestCase& test_case, const BenchConfig& config, const amrex::Geometry& geom,
                                      const CaseLayout& layout, std::unique_ptr<LGFPoissonSolver>& shared_solver)
{
    BL_PROFILE("<Compute> runTestCase()");

    amrex::Vector<CaseResult> results;
    const int n_ghost = config.n_ghost;

    // the reference backend solves every index type of the case; not timed
    std::unique_ptr<LGFPoissonSolver> reference;
    if (config.reference_name != "none")
    {
        reference = makeLGFPoissonSolver(config.reference_name, geom, layout.soln_ba, config.n_lookup, n_ghost);
    }

    for (int t = 0; t < static_cast<int>(config.index_types.size()); ++t)
    {
        const amrex::BoxArray typed_ba = amrex::convert(layout.soln_ba, config.index_types[t]);

        amrex::MultiFab source(typed_ba, layout.soln_dm, 1, n_ghost);
        amrex::MultiFab target(typed_ba, layout.soln_dm, 1, n_ghost);
        initializeSourceMultiFab(source, geom, test_case.source);

        CaseResult result;
        result.case_name = test_case.name;
        result.type_name = config.index_type_names[t];
        result.solver_name = config.solver_name;
        result.n_supp_cells = layout.n_supp_cells;
        result.n_soln_cells = layout.n_soln_cells;
        result.n_bbox_cells = layout.n_bbox_cells;
        result.n_iter = config.n_iter;

        // a fresh solver per (case, type), or one for the whole run that is
        // regridded once per case
        std::unique_ptr<LGFPoissonSolver> own_solver;
        LGFPoissonSolver* solver = nullptr;
        if (config.reuse_solver)
        {
            if (!shared_solver)
            {
                result.t_build = timedCall([&] () {
                    shared_solver = makeLGFPoissonSolver(config.solver_name, geom, layout.soln_ba, config.n_lookup, n_ghost);
                });
            }
            else if (t == 0)
            {
                result.t_build = timedCall([&] () {
                    shared_solver->regridOnto(geom, layout.soln_ba, layout.soln_dm);
                });
            }
            solver = shared_solver.get();
        }
        else
        {
            result.t_build = timedCall([&] () {
                own_solver = makeLGFPoissonSolver(config.solver_name, geom, layout.soln_ba, config.n_lookup, n_ghost);
            });
            solver = own_solver.get();
        }

        // the first solve carries the setup; the rest are the repeated cost,
        // and must reproduce the first
        amrex::MultiFab first_target;
        amrex::Real t_sum = 0.0;
        for (int it = 0; it < config.n_iter; ++it)
        {
            const amrex::Real t_solve = timedCall([&] () {
                solver->solvePoisson(source, target, layout.supp_ba);
            });

            if (it == 0)
            {
                result.t_first = t_solve;
                result.t_repeat_min = std::numeric_limits<amrex::Real>::max();
                if (config.n_iter > 1)
                {
                    first_target.define(typed_ba, layout.soln_dm, 1, n_ghost);
                    amrex::MultiFab::Copy(first_target, target, 0, 0, 1, n_ghost);
                }
                continue;
            }

            t_sum += t_solve;
            result.t_repeat_min = amrex::min(result.t_repeat_min, t_solve);
            result.t_repeat_max = amrex::max(result.t_repeat_max, t_solve);
        }

        if (config.n_iter > 1)
        {
            result.t_repeat_mean = t_sum / amrex::Real(config.n_iter - 1);
            result.drift = relativeDifference(target, first_target, n_ghost);
        }
        else
        {
            result.t_repeat_min = 0.0;
        }

        // against the reference, ghosts included, before FillBoundary touches
        // them. Direct summation handles cell-centred and nodal data only
        const amrex::IndexType ix = config.index_types[t];
        const bool has_reference = reference
            && (config.reference_name != "direct"
                || ix == amrex::IndexType::TheCellType() || ix == amrex::IndexType::TheNodeType());
        if (has_reference)
        {
            amrex::MultiFab ref_target(typed_ba, layout.soln_dm, 1, n_ghost);
            reference->solvePoisson(source, ref_target, layout.supp_ba);
            result.error = relativeDifference(target, ref_target, n_ghost);
        }

        // the residual of the problem actually posed: f restricted to the tags.
        // Against the unrestricted f it would also count what tagging dropped,
        // up to tagging_threshold * max|f|
        amrex::MultiFab posed_source(typed_ba, layout.soln_dm, 1, n_ghost);
        fillTagRegion(posed_source, layout.supp_ba);
        amrex::MultiFab::Multiply(posed_source, source, 0, 0, 1, n_ghost);

        target.FillBoundary(geom.periodicity());
        const amrex::MultiFab residual = computeResidual(posed_source, target, geom);
        const amrex::Real source_max = posed_source.norm0(0, 0);
        result.residual = residual.norm0(0, 0) / ((source_max > 0.0) ? source_max : 1.0);

        result.is_pass = (result.residual <= config.tol_residual)
                      && (result.drift <= config.tol_drift)
                      && (result.error <= config.tol_error);

        amrex::Print() << "  [" << result.case_name << " | " << result.type_name << " | " << result.solver_name << "]"
                       << " build " << result.t_build << " s, first solve " << result.t_first << " s";
        if (config.n_iter > 1)
        {
            amrex::Print() << ", repeat " << result.t_repeat_mean << " s (min " << result.t_repeat_min
                           << ", max " << result.t_repeat_max << ", over " << (config.n_iter - 1) << " repeat"
                           << ((config.n_iter > 2) ? "s)" : ")")
                           << ", setup " << 100.0 * result.setupTime() / (result.t_build + result.t_first)
                           << "% of build + first";
        }
        amrex::Print() << "\n    residual " << result.residual;
        if (has_reference) { amrex::Print() << ", error vs " << config.reference_name << " " << result.error; }
        else if (reference) { amrex::Print() << ", no reference (" << config.reference_name << " cannot do this type)"; }
        if (config.n_iter > 1) { amrex::Print() << ", drift " << result.drift; }
        amrex::Print() << " -> " << (result.is_pass ? "PASS" : "FAIL") << "\n";

        // MLMG needs the whole domain as its layout, and is cell centred
        amrex::MultiFab target_mlmg;
        const bool is_mlmg = config.compare_MLMG && config.layout == "full"
                          && config.index_types[t] == amrex::IndexType::TheCellType();
        if (is_mlmg)
        {
            target_mlmg.define(typed_ba, layout.soln_dm, 1, n_ghost);
            target_mlmg.setVal(0.0);
            amrex::Geometry geom_copy = geom;      // syncBCs() takes a non-const Geometry
            syncBCs(target_mlmg, target, geom_copy, n_ghost);

            const amrex::Real t_mlmg = timedCall([&] () { solveMLMG(source, target_mlmg, geom); });
            // MLMG reads the ghost values as Dirichlet data on the boundary faces,
            // half a cell from where the LGF puts them: O(h) differences are expected
            amrex::Print() << "    MLMG with LGF Dirichlet data: " << t_mlmg << " s, differs from "
                           << result.solver_name << " by " << relativeDifference(target_mlmg, target, 0)
                           << " (boundary data half a cell off, so O(h))\n";
        }
        else if (config.compare_MLMG && t == 0)
        {
            amrex::Print() << "    MLMG comparison skipped: it needs layout = full and cell data\n";
        }

        if (config.write_plot)
        {
            writeCasePlot(config, result, geom, source, target, residual, layout.supp_ba,
                          is_mlmg ? &target_mlmg : nullptr);
        }

        results.push_back(result);
    }

    return results;
}

void printSummary(const amrex::Vector<CaseResult>& results, const BenchConfig& config)
{
    int n_fail = 0;
    for (const CaseResult& result : results) { n_fail += result.is_pass ? 0 : 1; }

    amrex::Print() << "\n==== SUMMARY: " << results.size() << " runs, " << n_fail << " failed"
                   << " | lgf.solver = " << config.solver_name << ", layout = " << config.layout
                   << ", n_iter = " << config.n_iter << ", reuse_solver = " << config.reuse_solver << "\n";

    amrex::Print() << std::left << std::setw(14) << "case" << std::setw(8) << "type"
                   << std::right << std::setw(11) << "soln cells" << std::setw(8) << "bbox%"
                   << std::setw(11) << "build" << std::setw(11) << "first" << std::setw(11) << "repeat"
                   << std::setw(8) << "setup%" << std::setw(11) << "residual" << std::setw(11) << "error"
                   << std::setw(11) << "drift" << "  status\n";

    for (const CaseResult& result : results)
    {
        const amrex::Real bbox_pct = 100.0 * amrex::Real(result.n_soln_cells) / amrex::Real(result.n_bbox_cells);
        const amrex::Real setup_pct = 100.0 * result.setupTime() / (result.t_build + result.t_first);
        const bool has_repeat = (result.n_iter > 1);

        amrex::Print() << std::left << std::setw(14) << result.case_name << std::setw(8) << result.type_name
                       << std::right << std::setw(11) << result.n_soln_cells
                       << summaryCell(bbox_pct, true, 8)
                       << summaryCell(result.t_build, false, 11) << summaryCell(result.t_first, false, 11)
                       << summaryCell(has_repeat ? result.t_repeat_mean : -1.0, false, 11)
                       << summaryCell(has_repeat ? setup_pct : -1.0, true, 8)
                       << summaryCell(result.residual, false, 11) << summaryCell(result.error, false, 11)
                       << summaryCell(has_repeat ? result.drift : -1.0, false, 11)
                       << "  " << (result.is_pass ? "PASS" : "FAIL") << "\n";
    }
}

void writeSummary(const amrex::Vector<CaseResult>& results, const BenchConfig& config)
{
    if (config.summary_file.empty() || !amrex::ParallelDescriptor::IOProcessor()) { return; }

    const bool is_new = !std::ifstream(config.summary_file).good();
    std::ofstream out(config.summary_file, std::ios::app);

    if (is_new)
    {
        out << "dim,n_cell,case,index_type,solver,reference,layout,n_buffer_box,n_iter,reuse_solver,"
            << "n_procs,supp_cells,soln_cells,bbox_cells,t_build,t_first,t_repeat_mean,t_repeat_min,"
            << "t_repeat_max,t_setup,residual,error,drift,pass\n";
    }

    out << std::setprecision(6);
    for (const CaseResult& result : results)
    {
        out << AMREX_SPACEDIM << "," << config.n_cell << "," << result.case_name << "," << result.type_name << ","
            << result.solver_name << "," << config.reference_name << "," << config.layout << ","
            << config.n_buffer_box << "," << config.n_iter << "," << config.reuse_solver << ","
            << amrex::ParallelDescriptor::NProcs() << "," << result.n_supp_cells << "," << result.n_soln_cells << ","
            << result.n_bbox_cells << "," << result.t_build << "," << result.t_first << ","
            << result.t_repeat_mean << "," << result.t_repeat_min << "," << result.t_repeat_max << ","
            << result.setupTime() << "," << result.residual << "," << result.error << "," << result.drift << ","
            << (result.is_pass ? 1 : 0) << "\n";
    }

    amrex::Print() << "summary appended to " << config.summary_file << "\n";
}
