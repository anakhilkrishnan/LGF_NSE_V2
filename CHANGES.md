# Merge of LGF_Accelerate into LGF_NSE_GPU

November 2026. Everything after the merge commit was written by Claude
(commits authored `Claude <noreply@anthropic.com>`), so `git log` and
`git blame` separate it from the earlier work.

## History

- The merge commit joins `LGF_NSE_GPU/main` (ab719cf, first parent) and
  `LGF_Accelerate/main` (464879b, second parent). All 183 earlier commits are
  kept with their authors and dates.
- `feat/SDF` (LGF_NSE_GPU, three commits ahead of its old `main`) is kept as
  it was. A trial merge into the new `main` goes through without conflicts.
- File moves after the merge are pure renames in their own commit, so
  `git log --follow` and `git blame` reach back into both histories.
- The bench's older `LGFOpenBC` (aaa17ff) was dropped in the merge in favour
  of the NSE solver's newer one (2867652). It stays reachable through the
  second parent.

## Layout

```
Src/LGF_core/   LGFKernels.H, LGFPoissonSolver.H, LGFSolverFactory.{H,cpp}, Utils/
                directSum/   DirectSumLGF, directSumKernel.H
                fft/         LGFOpenBC
                fmm/         LGFFastMultipole, fastMultipoleKernel.H   (new)
Src/NSE_core/   unchanged layout
Src/NSE_run/    unchanged layout
Src/LGF_test/   the bench, now also holding tagSource.cpp
Docs/           from LGF_Accelerate
```

## NSE solver

- `ProjectionWorkspace` holds `std::unique_ptr<LGFPoissonSolver>`, built by
  `makeLGFPoissonSolver()` from `lgf.solver = fft | fmm | direct`. With no
  `lgf.*` entries it builds `LGFOpenBC` exactly as before.
- `DomainManager::initializeVelField`, `refineVelField`,
  `checkAndRefreshVelocity` and `vor2vel` take `LGFPoissonSolver&`;
  `main.cpp` passes `*workspace.lgf_poisson_solver`.
- The search-domain solver in `initializeSnugDomain` stays a concrete
  `LGFOpenBC`: that domain is one dense box, the FFT's best case.
- `NSE_core/Utils/domainManager_diag3D.cpp` (not compiled) got the same
  signature change, so it is still a drop-in.
- `NSE_run/Utils/inputs` documents the `lgf.*` entries, commented out.

## LGF_core

- `LGFPoissonSolver` is the one interface. `solveNodalPoisson()` is gone:
  every index type goes through `solvePoisson()`. The header documents the
  contract the backends share: a cell-centred layout `convert()`ed per index
  type, a cell-centred `tag_ba`, a zeroed or filled source ghost layer, and
  every target cell overwritten. `tagSource()` moved to the bench.
- `LGFOpenBC` (the NSE solver's version) derives from it. Its code is
  otherwise untouched.
- `DirectSumLGF::solvePoisson()` dispatches on index type to
  `solveCellPoisson()` (the old body) or `solveNodalPoisson()`, and aborts
  for face and edge data.
- New: `LGFFastMultipole`, the block-structured LGF FMM, and the factory.

## Test bench (LGF_test)

These change what the bench computes:

- **Layout.** The bench now builds a cell-centred `n_cell^d` layout and
  `convert()`s it to `index_type`, as `DomainManager` does. Nodal directions
  now have `n_cell + 1` points instead of `n_cell`, so runs with non-cell
  index types are not point-for-point comparable with older bench results.
  `Geometry` now gets the cell-centred domain; the typed one failed AMReX's
  `dom.cellCentered()` debug assertion.
- **Source ghosts.** The source is zeroed, initialized and `FillBoundary()`'d
  before tagging and solving. `LGFOpenBC` reads that ghost layer, and before
  this the bench left it uninitialized.
- **Backend selection.** `solver_type` is replaced by `lgf.solver`:
  1 is now `direct`, 2 is now `fft`, and `fmm` is new. `Utils/inputs` is
  updated accordingly.
- **MLMG comparison.** It now runs for cell-centred data only.

`Utils/inputs` still has `n_lookup = 32`, set for 2D in 417c9aa. The 3D
lookup table stops at 16, so a 3D run of that file aborts in `computeLGF`;
pass `n_lookup=16` for 3D.

## Build

- One top-level `CMakeLists.txt` (LGF_NSE_GPU's) builds `LGF_core` and,
  through `LGF_BUILD_NSE` and `LGF_BUILD_TEST` (both ON), `nserun` and
  `lgfpoisson`. The bench therefore inherits the NSE build's rules:
  - `AMReX_GPU_BACKEND` and `AMReX_SPACEDIM` must be given;
  - C++20;
  - MPI, FFT and profiling forced on;
  - OpenMP on by default for CPU builds.
- `lgfpoisson` and its `Utils` files go to `Exec/LGF_test`, beside
  `Exec/NSE_test`.
- LGF_Accelerate's submodule-library mode and its `AMReX_ROOT` option are
  gone. `-DFETCHCONTENT_SOURCE_DIR_AMREX_CODE=<amrex>` points the build at an
  existing AMReX source tree instead.

## How this was checked

The checks were run in a sandbox: CPU only, one core, OpenMP with one
thread, against AMReX `development` as of 12 Nov 2026.

- **3D and 2D builds.** CPU with OpenMP, both executables.
- **NSE, default backend.** The pre-merge `ab719cf` and the merged build ran
  the same case: the three-ring initial condition on a 96^3 grid, 8^3 boxes,
  search domain ±7.68, `search_to_fine_refine_ratio = 2`, interface tracking
  off, two steps. The step-2 plotfiles (velocity, pressure, vorticity) are
  bitwise identical, checked with `fcompare`.
- **NSE, `lgf.solver = fmm`.** Same case, serial and on 2 MPI ranks. Against
  `fft`, velocity and pressure are identical and vorticity magnitude differs
  by 3e-14 relative. A deliberately coarse FMM (`fmm_block_size = 8`,
  `fmm_n_interp = 4`) raised the velocity-refresh error to about 1e-3 and the
  support grew every step. FMM error has to stay well below
  `support_threshold`.
- **Bench, all backends against `fft`.** Same setup as `main.cpp`:
  - 32^3: `fmm` 5e-16 and `direct` 3e-15 relative, cell and nodal;
  - 96^3: `fmm` 2.2e-10, cell, face and edge;
  - 64^3 on 3 MPI ranks with `fmm_block_size = 8`: `fmm` 1.3e-7, cell and
    edge.
- **Bench executable, end to end.** In 3D, `fft` and `fmm` run with the
  MLMG comparison on, and `direct` aborts cleanly on face data. In 2D, with
  `Utils/inputs` as committed at 128^2, the `Target_Phi` plotfiles differ
  from `fft` by 3.6e-10 relative for `fmm` and 4.2e-15 for `direct`
  (`fcompare`). The 2D `nserun` builds but was not run.

Not checked: GPU (CUDA) builds of anything, runs with interface tracking
on, restarting from a checkpoint (`refineVelField`), long runs, and the FMM
on a GPU. The FMM's block FFTs are not batched yet, which matters on a GPU.

## Left for the author

- `README.md` still links to the LGF_NSE_GPU repository.
- `NSE_core`'s CMake still puts `Src/NSE_run` on `NSE_core_lib`'s public
  include path (for `InitialVorticity.H` and `FreestreamVelocity.H`). Both
  apps also have a `MyFunctions.H`, which is fine while the bench does not
  link `NSE_core`.
- `Docs/` describes LGF_Accelerate and has not been updated for the merge.
