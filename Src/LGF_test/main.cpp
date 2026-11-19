#include <MyFunctions.H>

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    amrex::Print() << "Launching test run for LGF Core...\n";
    const int n_fail = extendedMain();

    amrex::Finalize();

    // non-zero when any check failed, so scripts can tell
    return (n_fail > 0) ? 1 : 0;
}

int extendedMain()
{
    // marking start to track runtimes
    auto start_time = amrex::second();

    // every input, read once; see TestSuite.H and Utils/inputs
    const BenchConfig config = readBenchConfig();

    // the background layout: the whole domain, cell centred. Every case is
    // tagged on it, and with layout = full it is also the layout solved on
    amrex::Box domain(amrex::IntVect(0), amrex::IntVect(config.n_cell - 1));
    amrex::BoxArray ba(domain);
    ba.maxSize(config.max_grid_size);

    amrex::RealBox real_box(config.domain_lo, config.domain_hi);
    amrex::Vector<int> is_periodic(AMREX_SPACEDIM, 0); // 0 = not periodic
    amrex::Geometry geom;
    geom.define(domain, &real_box, amrex::CoordSys::cartesian, is_periodic.data());

    amrex::DistributionMapping dm(ba);

    const amrex::Vector<TestCase> test_suite = loadTestSuite(config.case_names);

    amrex::Print() << "Running " << test_suite.size() << " case(s) x " << config.index_types.size()
                   << " index type(s), lgf.solver = " << config.solver_name
                   << ", reference_solver = " << config.reference_name
                   << ", layout = " << config.layout << ", n_iter = " << config.n_iter << "\n";

    // with reuse_solver, one solver follows the whole run
    std::unique_ptr<LGFPoissonSolver> shared_solver;
    amrex::Vector<CaseResult> results;

    for (const TestCase& test_case : test_suite)
    {
        const CaseLayout layout = buildCaseLayout(test_case, config, geom, ba, dm);

        amrex::Print() << "\n== " << test_case.name << ": " << test_case.description << "\n"
                       << "   " << layout.supp_ba.size() << " tagged boxes (" << layout.n_supp_cells << " cells), "
                       << "solve layout " << layout.soln_ba.size() << " boxes (" << layout.n_soln_cells << " cells, "
                       << 100.0 * amrex::Real(layout.n_soln_cells) / amrex::Real(layout.n_bbox_cells)
                       << "% of its bounding box)\n";

        const amrex::Vector<CaseResult> case_results = runTestCase(test_case, config, geom, layout, shared_solver);
        results.insert(results.end(), case_results.begin(), case_results.end());
    }

    printSummary(results, config);
    writeSummary(results, config);

    auto end_time = amrex::second();
    auto elapsed_time = end_time - start_time;

    // making copies to track slowest and fastest processor
    amrex::Real max_time = elapsed_time;
    amrex::Real min_time = elapsed_time;

    // performing a reduction over all the processors to track the slowest and
    // fastest MPI rank
    const int IOProc = amrex::ParallelDescriptor::IOProcessorNumber();
    amrex::ParallelDescriptor::ReduceRealMax(max_time, IOProc);
    amrex::ParallelDescriptor::ReduceRealMin(min_time, IOProc);

    amrex::Print() << "Max compute time (Slowest Rank): " << max_time << " s\n"
                   << "Min compute time (Fastest Rank): " << min_time << " s\n"
                   << "Time spread (Load Imbalance)   : " << (max_time - min_time) << " s\n";

    int n_fail = 0;
    for (const CaseResult& result : results) { n_fail += result.is_pass ? 0 : 1; }

    return n_fail;
}
