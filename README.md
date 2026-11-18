# 3D Unbounded-Domain Incompressible Viscous Flow Solver

Akhil Krishnan\
November 11, 2026\
Indian Institute of Science

[GitHub Repository](https://github.com/anakhilkrishnan/LGF_NSE_GPU.git)

## Introduction

The solver is written using [AMReX](https://amrex-codes.github.io). Consequently, the code is performant and portable across architectures. The solver is written to preserve AMReX's own performance benefits without introducing any  bottlenecks. 

The methods implemented in the solver are extensions to Liska and Colonius' work on [Unbounded Flow Solvers using LGFs](https://www.sciencedirect.com/science/article/pii/S002199911630626X), at Caltech.

The solver is part of a larger developmental process of applying LGF-based methods to simulate multiphase flows, flows past solid objects, bubbles and bubble clusters, and so on,  with a sharp interface treatment. A secondary goal is to keep the solver massively parallelizable on hundreds of GPUs. 

Development of the code began in March 2026, as a CFD course project. The progress thus far is the culmination of countless inputs from my PhD advisor, lab mates, friends, and popular LLM's, paired with long hours of writing, debugging, rewriting, and refining code (all while having 5 different code explanation websites open), and having a supercomputer to test and run those codes on. Ofcourse, none of this would be possible without the support of my loved ones, and my sister.

## Using the code

Building and running the code requires the following:
1. CUDA Toolkits/NVHPC kit
2. OpenMPI, OpenMP if desired
3. CMake (build system)
4. Latest GCC
5. FFTW, for CPU builds (cuFFT used for GPU builds)

AMReX is built along with the solver by default, pre-existing AMReX files can be provided using `-DFETCHCONTENT_SOURCE_DIR_AMREX_CODE=<path-to-'amrex'>` flag in build dir creation step.

To build the executable:
```bash
## start in the parent directory containing CMakeLists.txt and Src directory
mkdir Exec && cd Exec

#(a) for OpenMP CPU executable
cmake -B build_omp -S .. -DAMReX_SPACEDIM=3 -DAMReX_GPU_BACKEND=NONE

#(b) for pure MPI (no OpenMP) CPU executable
cmake -B build_mpi -S .. -DAMReX_SPACEDIM=3 -DAMReX_GPU_BACKEND=NONE -DAMReX_OMP=OFF

#(c) for GPU executable 
cmake -B build_gpu -S .. -DAMReX_SPACEDIM=3 -DAMReX_GPU_BACKEND=CUDA

## this creates the build dir, NSE_test and LGF_test inside Exec. Complete the build process
cmake --build <build_dir_name> -j <n_procs>
```

Both executables are built by default: `nserun` (the NSE solver, in `Exec/NSE_test`) and `lgfpoisson` (the stand-alone Poisson test bench, in `Exec/LGF_test`). Either can be switched off with `-DLGF_BUILD_NSE=OFF` or `-DLGF_BUILD_TEST=OFF`.

To run the executable:
```bash
## good practice: do not alter in NSE_test directory, make a copy of NSE_test with an apt name
cp -r NSE_test NSE_threeringrun && cd NSE_threeringrun

## edit inputs file
vim inputs

## good practice: update the description.txt file with a brief description of the particular run, with context on what this run achieves
vim description.txt

#(a) run directly
mpiexec -n <nprocs> nserun inputs | tee ./Logs/log.txt

#(b) submit a jobscript on SLURM (many options available: cpu_multi.sh, gpu_multi.sh, gpu_multi_rudra.sh)
sbatch <jobscript>.sh

## all plot/log/checkpoint files are written to folders specified in inputs file, typical values are:

#(a)plotsfiles are written in Results/
ls Results/ # shows all files written so far

#(b) logfiles are written in Logs/
vim Logs/log.txt # opens log file in vim

#(c) checkpoint files (main and alt) are written to Checkpoints/
vim Checkpoints/main/status.dat # shows if checkpoint was written safely without interrupts and saved time step
```

## Repository layout

| Directory | Contents |
|---|---|
| `Src/LGF_core` | LGF kernels and the Poisson backends, shared by both executables: `LGFPoissonSolver.H` (the interface), `fft/` (`LGFOpenBC`), `fmm/` (`LGFFastMultipole`), `directSum/` (`DirectSumLGF`), and `LGFSolverFactory` |
| `Src/NSE_core` | the NSE solver library |
| `Src/NSE_run` | `nserun`: main, I/O and case headers |
| `Src/LGF_test` | `lgfpoisson`: the Poisson test bench |
| `Docs` | Sphinx sources |

The repository was formed by merging LGF_NSE_GPU with LGF_Accelerate (the test bench), keeping the history of both; see `CHANGES.md`.

## Choosing the Poisson backend

Both executables build their Poisson solver through `makeLGFPoissonSolver()`, which reads these inputs:

```
lgf.solver = fft          # fft (LGFOpenBC) | fmm (LGFFastMultipole) | direct (DirectSumLGF, O(N^2), small checks only)
lgf.fft_len_quantum = 1   # fft: solver lengths rounded up to a multiple of this
lgf.fmm_block_size = 16   # fmm: lattice points per block and direction
lgf.fmm_n_interp = 10     # fmm: nodes per interpolation
```

With none of them given, the backend is `fft`, built exactly as before the merge. `fft` is fastest when the active region fills most of its bounding box; `fmm` costs O(N) in the active blocks and pays off when the region is sparse. The FMM's error (about 1e-10 relative at the defaults) must stay well below `support_threshold`, or it shows up as spurious support.

## Contributing to the code

Any suggestions and bug fixes are welcome. You may reach out to the author via email (kakhil@iisc.ac.in). Alternatively, you may create a new branch and raise a pull request.

Please do not push/force push changes to main.