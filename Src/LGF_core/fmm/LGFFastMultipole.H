#ifndef LGFFASTMULTIPOLE_H_
#define LGFFASTMULTIPOLE_H_

#include <LGFPoissonSolver.H>
#include <fastMultipoleKernel.H>

#include <AMReX_BoxList.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_TagParallelFor.H>
#include <AMReX_FFT.H>

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

/**
 * Block-structured fast multipole LGF Poisson solver.
 *
 *   u(n) = h^2 * sum_m G(n-m) f(m)
 *
 * evaluated with the FLGF method of Liska & Colonius (2014), laid out as in
 * IBLGF-AMR: a kernel-independent, interpolation-based FMM whose every tree
 * node is a block of regularly spaced nodes, so that all translations are
 * block-wise FFT convolutions. Cost is O(N) in the number of active points
 * and independent of the shape of the active region, where LGFOpenBC pays
 * for the bounding box. Single resolution only.
 *
 * THE TREE IS A STACK OF HAND-ROLLED BOXARRAYS
 *
 *   Level 0 chops index space into blocks of n_block^d lattice points,
 *   anchored at index 0, and keeps every block that holds a target index
 *   (xsoln_ba, its hi+1 planes, and n_ghost_max ghosts). Level l+1 is level
 *   l coarsened by 2. Each level is a BoxArray of unit boxes in block-index
 *   space (blk_ba), so parents, neighbours and interaction lists are plain
 *   BoxArray::intersections() queries, and the data BoxArray (node_ba) is
 *   blk_ba refined by the block size. The tree stops at the first level
 *   whose parents all fit in a 2-wide window: every pair that is still
 *   separated there is in each other's interaction list.
 *
 *   An interval on level l holds n_node = n_block + 1 nodes per direction
 *   at a spacing of 2^l lattice points, i.e. its blocks' footprint plus the
 *   first node of the next interval (L&C's one-point overlap). A parent's
 *   nodes are therefore every other node of its children, end points
 *   included, and no child node is ever extrapolated. On level 0 the nodes
 *   ARE the lattice points: the overlap node carries no source and its
 *   induced field is discarded, so level 0 is exact and its MultiFabs live
 *   on plain n_block^d lattice boxes.
 *
 * ONE SOLVE
 *
 *   1. copy in:   source -> src[0], zeroed outside the tagged support
 *   2. upward:    src[l+1] = sum over children of Reg(src[l])
 *   3. downward, from the top level to level 0:
 *                 tar[l]  = Interp(tar[l+1])
 *                 tar[l] += IFFT( sum_S FFT(src_S) * K^_{T-S} ),
 *      S running over the interaction list of T, plus on level 0 its
 *      near neighbours (the exact, uncompressed part)
 *   4. copy out:  tar[0] -> target, ghosts included
 *
 *   Reg and Interp are tensor products of one 1D operator, applied one
 *   direction at a time and in place. 'aux' is the only transfer scratch:
 *   it is the level's data on the DistributionMapping of the PARENTS, so
 *   both passes run entirely on the parent's rank and the only messages
 *   are aux <- src (upward) and tar <- aux (downward).
 *
 *   Source spectra are computed once per interval and level, held only
 *   while that level's interactions run, and only for intervals that hold
 *   sources. Intervals owned by other ranks are pulled in real space into
 *   'halo' (n_node^d reals instead of their spectra) and transformed here.
 *
 * ALL INDEX TYPES, NO RETYPE COPIES
 *
 *   G depends on index differences only, so the solve never looks at the
 *   index type: lattice point (i,j,k) of ANY type belongs to the level-0
 *   block that contains index (i,j,k). The hi+1 planes of the staggered
 *   directions and all ghost cells are simply more target indices. src[0]
 *   and tar[0] are aliased by header-only MultiFabs carrying the caller's
 *   index type (one pair per type, kept until the next regrid so AMReX's
 *   copy patterns stay cached); AMReX then copies caller <-> tree
 *   directly. Unlike LGFOpenBC there are no canonical scratch copies, no
 *   fabricated planes, and the source needs no ghost cells.
 *
 *   The layout follows LGFPoissonSolver's contract: one cell-centred
 *   BoxArray, convert()ed per index type, and convert(tag_ba, type) is the
 *   support. Two looser forms are accepted as well, since they cost
 *   nothing: a layout built directly from typed boxes (MultiFabs then share
 *   its index ranges, and cell-centred tag boxes are read in its index
 *   type), and tag boxes that already carry the source's index type. What
 *   is checked is canonicalBoxArray(), which agrees for every MultiFab
 *   built from the same layout.
 *
 * WHAT IS CACHED, AND WHEN IT IS REBUILT
 *
 *   kernel spectra   per (level, |block offset|); depend on n_block,
 *                    n_interp and n_lookup alone, so they survive every
 *                    regrid. Reflected offsets reuse the same spectrum.
 *   tree             on regridOnto(), or if the source's
 *                    DistributionMapping changes; built lazily.
 *   source plan      masks, interaction lists, halo layout: whenever
 *                    tag_ba differs from the last solve's. The plan is
 *                    shared by all index types, so it includes every block
 *                    that touches the support under ANY type; the blocks
 *                    that hold no source under the type at hand (those
 *                    reached only by a hi+1 plane it does not have) are
 *                    flagged per solve and skipped, src_live.
 *
 * ACCURACY
 *
 *   The only approximation is the Lagrange interpolation of the far field,
 *   using the n_interp nearest nodes. The error falls like
 *   n_node^(-n_interp) once n_node > n_interp and is largest next to block
 *   edges (L&C Fig. 1). n_interp = 10 is the largest value that keeps the
 *   equispaced Lebesgue constant harmless in double precision. Measured
 *   against LGFOpenBC with random sources, max norm relative to max |u|:
 *   2e-7 for n_block = 8 and 2e-10 for n_block = 16, in 2D and 3D alike.
 *   Anything the level-0 window reaches (the children of the parent's
 *   neighbours) is exact to round-off.
 *
 * MEMORY
 *
 *   Per level-0 block: src n_block^d, tar and aux (n_block + 2)^d reals;
 *   the upper levels add about 1/(2^d - 1) of that. Per source interval in
 *   reach of a rank, while its level is being convolved: one spectrum,
 *   (n_fft/2 + 1) * n_fft^(d-1) complex (0.39 MB for n_block = 16 in 3D).
 *   Kernel spectra: at most 4^d of the same size per level.
 */

namespace fmm_detail
{
    struct Level
    {
        // tree
        amrex::BoxArray            blk_ba;    // one unit box per interval, in block indices
        amrex::BoxArray            node_ba;   // data boxes: lattice indices on level 0, node slots above
        amrex::DistributionMapping dm;        // owner of each interval
        amrex::DistributionMapping par_dm;    // owner of each interval's parent
        amrex::Vector<int>         par_idx;   // each interval's parent on the next level
        bool                       is_par_local = true; // par_dm == dm: transfers skip aux
        int                        n_data = 0;          // nodes per direction that carry data

        amrex::MultiFab src;                  // regularized sources
        amrex::MultiFab tar;                  // induced field
        amrex::MultiFab aux;                  // on (node_ba, par_dm): transfer scratch

        // source plan
        amrex::Vector<int> src_mask;          // 1 where the interval can hold sources, whatever the index type
        amrex::Vector<int> src_live;          // 1 where it does in the solve in progress
        amrex::Vector<int> slot_idx;          // interval -> local spectrum slot, -1 if unused here
        int                n_slot_own = 0;    // slots filled from src
        int                n_slot = 0;        // ... plus those filled from halo
        amrex::MultiFab    halo;              // source intervals owned by other ranks

        // interaction lists of the local targets, CSR by local index
        amrex::Vector<int>            pair_ptr;
        amrex::Gpu::DeviceVector<int> pair_slot;   // spectrum slot of the source
        amrex::Gpu::DeviceVector<int> pair_code;   // (kernel slot << 3) | reflection bits
        amrex::Gpu::DeviceVector<int> slot_live;   // src_live, by spectrum slot
    };

    // header-only MultiFabs aliasing src[0] and tar[0] under another index type
    struct TypedView
    {
        amrex::IndexType type;
        amrex::MultiFab  src;
        amrex::MultiFab  tar;
    };
}

// the FMM's index bookkeeping (2 bits of block offset and 1 reflection bit
// per direction) and the far-field kernels are written for 2D and 3D
static_assert(AMREX_SPACEDIM >= 2, "LGFFastMultipole needs AMREX_SPACEDIM = 2 or 3");

class LGFFastMultipole : public LGFPoissonSolver
{
private:

    amrex::Geometry geom;
    amrex::BoxArray xsoln_ba_ref;             // local reference copy of the layout solved on, any index type
    amrex::BoxArray canon_ba;                 // canonicalBoxArray(xsoln_ba_ref); every index a solve may touch
    amrex::DistributionMapping xsoln_dm_ref;  // its DistributionMapping; known from the first solve on
    int n_lookup;
    int n_ghost_max;

    // FMM parameters, fixed for the lifetime of the object
    int n_block;      // lattice points per level-0 block and direction
    int n_interp;     // nodes used to interpolate one value
    int n_node;       // nodes per interval and direction: n_block + 1
    int n_fft;        // padded convolution length, >= 2*n_node - 1
    int n_half;       // n_fft/2 + 1: half-stored spectral length along x
    int n_batch;      // target intervals convolved per launch
    amrex::Long n_pad;   // points of one padded real block
    amrex::Long n_spec;  // points of one spectrum
    amrex::Box pad_box;
    amrex::Box spec_box;

    // tree and source plan
    amrex::Vector<fmm_detail::Level> levels;
    amrex::Vector<fmm_detail::TypedView> views;
    amrex::BoxArray cached_tag_ba;
    bool is_tree_built;
    bool is_plan_built;

    // 1D transfer operator, both child parities: row = parity * n_node + child node
    amrex::Gpu::DeviceVector<amrex::Real> interp_val;   // n_interp weights per row
    amrex::Gpu::DeviceVector<int>         interp_beg;   // first parent node of each row

    // FFT of one padded block; pad_work and spec_work are the fixed plan arrays
    std::unique_ptr<amrex::FFT::LocalR2C<amrex::Real>> fft;
    amrex::Gpu::DeviceVector<amrex::Real>         pad_work;
    amrex::Gpu::DeviceVector<fmm_detail::Complex> spec_work;
    amrex::Gpu::DeviceVector<fmm_detail::Complex> spec_src;   // source spectra of the level in progress
    amrex::Gpu::DeviceVector<fmm_detail::Complex> spec_sum;   // summed spectra of one batch of targets

    // kernel spectra per level, and where each |block offset| sits in them
    amrex::Vector<amrex::Gpu::DeviceVector<fmm_detail::Complex>> ker_spec;
    amrex::Vector<amrex::Vector<int>> ker_slot;
    amrex::Vector<int> n_ker;

    // HELPER functions that stay on the host
    // same boxes, retyped; amrex::convert() would move the hi index
    static amrex::BoxArray retypeBoxArray(const amrex::BoxArray& ba, amrex::IndexType type);
    // every index any index type built from ba can hold: [lo, hi+1], cell typed
    static amrex::BoxArray canonicalBoxArray(const amrex::BoxArray& ba);
    // tag boxes in the index type of the layout they come from
    amrex::BoxArray layoutTagBoxArray(const amrex::BoxArray& tag_ba) const;
    // unique unit boxes of the blocks that the boxes of ba touch
    static amrex::BoxArray blockBoxArray(const amrex::BoxArray& ba, int ratio);
    // smallest length >= n whose only prime factors are 2, 3 and 5
    static int nextFastLength(int n);
    // low corner of box i; BoxArray::operator[] returns a temporary
    static amrex::IntVect lowEnd(const amrex::BoxArray& ba, int i);
    // box one cell thick along dim, n_lo wide below it and n_hi wide above it
    static amrex::Box crossBox(const amrex::IntVect& lo, int dim, int n_lo, int n_hi);

    void buildInterpTables();
    void buildTree();
    void buildSourceMasks(const amrex::BoxArray& tag_ba);
    void buildInteractionLists(int lev);
    void buildSourcePlan(const amrex::BoxArray& tag_ba);
    fmm_detail::TypedView& typedView(amrex::IndexType type);

    template <int... dim>
    void regularizeLevel(int lev, std::integer_sequence<int, dim...>);
    template <int... dim>
    void interpolateLevel(int lev, std::integer_sequence<int, dim...>);

public:

    // xsoln_ba_in: the layout the source and target MultiFabs are built from,
    // of any index type. n_ghost_max_in: the most ghosts any target carries.
    // n_block_in must be >= 4: a leftover LGFOpenBC len_quantum of 1 is caught here
    LGFFastMultipole(const amrex::Geometry& geom_in, const amrex::BoxArray& xsoln_ba_in,
                     int n_look_in, int n_ghost_max_in, int n_block_in = 16, int n_interp_in = 10);

    // HELPER functions, kept public because GPU kernels are launched from them
    // spectrum of the convolution kernel for one |block offset| on one level
    void buildKernel(int lev, int ker_id);
    // source -> src[0] through the typed view, zeroed outside convert(tag_ba, type);
    // also flags the level-0 blocks that hold sources under this index type
    void copySource(const amrex::MultiFab& source, const amrex::BoxArray& tag_ba);
    // one direction of the upward pass, level lev -> lev+1
    template <int dim>
    void regularizePass(int lev);
    // one direction of the downward pass, level lev+1 -> lev
    template <int dim>
    void interpolatePass(int lev);
    // FFT of one interval's nodes into spectrum slot 'slot' of spec_src
    void transformSource(amrex::Array4<amrex::Real const> const& src, const amrex::IntVect& lo,
                         int n_src, int slot);
    // near-neighbour and interaction-list convolutions of one level, added to tar
    void interactLevel(int lev);

    // WRAPPER functions for external calling
    // builds the tree if it is stale; needs the DistributionMapping, so it
    // is a no-op until the first solve or regridOnto() has supplied one
    void buildSolver();
    // source and target: same index type, built from xsoln_ba_ref, same dm.
    // Every target cell, ghosts included, is overwritten
    void solvePoisson(const amrex::MultiFab& source, amrex::MultiFab& target, const amrex::BoxArray& tag_ba) override;
    void regridOnto(const amrex::Geometry& new_geom, const amrex::BoxArray& new_ba, const amrex::DistributionMapping& new_dm) override;

    // diagnostics
    int numLevels() const { return static_cast<int>(levels.size()); }
    amrex::Long numIntervals(int lev) const { return levels[lev].blk_ba.size(); }
};

#endif
