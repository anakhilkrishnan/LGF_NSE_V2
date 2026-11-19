# Signed Distance Function (SDF) implementation with WENO5 advection in AMReX

Akhil Krishnan\
October 27, 2026\
Indian Institute of Science

## Code Structure

The code is split into 5 files:

1. `interfaceTracker.cpp` : contains routines for initialization, filling boundary conditions, computing RHS for time integrator, updating with the snug domain, and checking if the unclipped band is inside the snug domain.
2. `InterfaceTracker.H` : contains class definition and all dependencies.
3. `InterfaceDiscretization.H` : contains WENO5/HOUC5 discretization functions.
4. `InitialSDF.H` : contains functions that provide expressions for initial conditions, including primitive shapes and combinations and the 'active case'.
5. `InterfaceReinit.H` : contains the device kernels of the closest-point reinitialization.

The time integration is in `ProjectionWorkspace` class, `main.cpp` handles initialization and `IOManager` class handles plotting.

## Implementation notes

### Inputs to the solver

1. Initial condition for $\psi$ (set in `InitialSDF.H` file at compile time).
2. `bool` switch to disable interface tracking if not needed.
3. `scheme` (WENO5 or HOUC5) as a runtime variable.
4. $\psi_\infty$ value.
5. Reinitialization triggers and interpolant degree (see Reinitialization).

### Data handling and boundary conditions

1. Signed Distance Function ($\psi$) stored in cell-centered index type, in every cell in the snug domain, including the buffer.
2. Each field is stored as a component of the same `amrex::MultiFab` object, with 3 ghost cells.
3. $\psi > 0$ for inside the shape, $\psi = 0$ for interface, and $\psi < 0$ for outside.
4. $\psi$ is clipped between limits ($\pm \psi_{\infty}$) at initialization.
5. The $\psi$ band ($|\psi| < \psi_\infty$) is required to stay inside the snug domain; the run aborts at initialization, restart or regrid otherwise. It should also stay inside the flow’s support region, since velocity in the buffer is inaccurate, but this is not checked. Ghost cells that overlap a neighbouring box take its values.
6. Boundary ghost cells are treated based on the flow. An inflow condition in the last valid cell fills the ghost cells with $-\psi_\infty$. An outflow condition extrapolates linearly and evaluates. This allows for WENO5 to be implemented throughout the domain, buffer included.
7. Each field is plotted for post processing.

### Advection scheme

1. WENO5/HOUC5 implemented with the same scheme as `Own-LS-Option-3D-Test-Advection.h`. Algebraically regrouped as 3 candidates of third-order stencils, computed from face-gradients of $\psi$.
2. The velocities are averaged to cell-centers.
3. Only the upwind derivative is computed. (`Own-LS-Option-3D-Test-Advection.h` computes both `Psix_L` and `Psix_R` for each cell) 
4. HOUC5 is computed identically to WENO5 with the optimal weights (0.1, 0.6, 0.3) without any smoothness indicators.
5. At each stage, velocity ($\vec{u}$) obtained (as a sum of velocity perturbation from the projection method and freestream) from IF-HERK integration is used to advect $\psi$ using the same Butcher Tableau as the HERK, without the action of the integrating factor.
6. On every snug-domain update, the regrid routine copies $\psi$ wherever the old and new grids overlap and assigns $-\psi_\infty$ to new cells. Beforehand it checks that removed cells, and cells on the edge of the old domain, all have $\psi \leq -\psi_\infty + 3\Delta x $ (the stencil reach), and aborts otherwise.
7. (Yet to be implemented) Per-step clipping of the $\psi$ field. UPDATE: Need to check whether this is needed, as its effect is known only between reinitializations.

### Reinitialization

1. Geometric rather than PDE-based: $\psi$ is reset to the distance to its own zero level set, rebuilt by local interpolation, so no pseudo-time solve moves the interface. The method is the high-order closest-point method of Saye (2014), as in the `algoim` library (`hocp.hpp`), restructured for GPUs: the seeds and kd-tree are replaced by a starting point from the gradient and by passes over neighbouring cells, so every cell is handled independently with a fixed amount of work. The device kernels are in `InterfaceReinit.H`; the driver is `InterfaceTracker::reinitialize()`.
2. Inputs: `interface_reinit_int` (reinitialize every $N$ steps), `interface_reinit_drift` (reinitialize when more than 1% of the cells with $|\psi| < 3\Delta x$ have $\big||\nabla\psi| - 1\big|$ above this value, by central differences) and `interface_reinit_degree` (3 or 5). Both triggers at 0 (the default) disables reinitialization. When enabled, $\psi$ is also reinitialized once after initialization (not on restart), and $\psi_\infty$ may be at most $16\Delta x$.
3. Called from `main.cpp` after each time step, before the snug-domain update, whose band check reads $\psi$.
4. For each interface:
   1. $\psi$ is copied with deep ghost cells (about $\psi_\infty/\Delta x$ plus the interpolation stencil; 19 for $\psi_\infty = 13.2\Delta x$). Ghost cells outside the snug domain take $-\psi_\infty$.
   2. The band is every cell within $\lceil \psi_\infty/\Delta x \rceil + 1$ cells (in each direction) of a sign change of $\psi$.
   3. Each band cell $\vec{x}$ finds its closest point $\vec{y}^*$ on the zero level set of a tensor-product Lagrange interpolant $p$ of $\psi$ ($4^3$ cells for degree 3, $6^3$ for degree 5, re-centred on the cell holding the current iterate): four projection steps $\vec{y} \leftarrow \vec{y} - p\,\nabla p/|\nabla p|^2$ starting from $\vec{x}$, then four Newton steps on the Lagrangian of $\min |\vec{y} - \vec{x}|^2$ subject to $p(\vec{y}) = 0$ (a $4\times4$ system per step).
   4. Three neighbour passes: a cell adopts the nearest of its 26 neighbours' closest points if it is nearer than its own (or it has none), and polishes it with Newton. This repairs failures and closest points that are only locally closest (near medial surfaces), and carries closest points out to cells the first pass could not resolve.
   5. $\psi(\vec{x}) = \mathrm{sign}(\psi_{old}(\vec{x}))\, \min(|\vec{x} - \vec{y}^*|, \psi_\infty)$, so the sign of a cell never changes. Cells outside the band, or beyond the reach of the passes, take $\pm\psi_\infty$.
   6. Zero-set correction: the interpolant of the new $\psi$ is evaluated at each cell's closest point (which lies on the old zero level set) and subtracted. Without it, the interpolation error (about $(\Delta x/r)^3\Delta x$ at a radius of curvature $r$, always to the same side) would make repeated reinitialization walk the interface.
5. Each reinitialization prints one line: the trigger, the band size, the unresolved cells, the largest change of $\psi$ next to the interface and the largest zero-set correction.
6. Verified with `reinitTest.cpp`: the distance is 4th order for degree 3 and 6th order for degree 5; with the correction, the zero level set drifts by $5\times10^{-7}\Delta x$ per reinitialization at 8 cells per tube radius (none measurable for degree 5). In the three-ring run to $t = 2$, reinitialized every 20 steps, the interfaces differ from a run without reinitialization by $0.003$–$0.004\,\Delta x$ on average. After correcting for the volume estimate's bias, the tube volumes are kept to about $5\times10^{-6}$, against a loss of $1$–$2\times10^{-4}$ without reinitialization. The cost is about 2.6% of the run time on CPUs.

### Tube/Band masking

The reinitialization restricts its work to the band, but its temporary copies and halo exchanges still cover the whole snug domain. Box-level masking is yet to be implemented.

### Refine at restart

Yet to be implemented for the signed distance function field, currently aborts run if attempted.
