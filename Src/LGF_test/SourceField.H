#ifndef SOURCEFIELD_H_
#define SOURCEFIELD_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BCRec.H>
#include <AMReX_Math.H>

/**
 * CLUSTER-OF-BLOBS SOURCE FIELD
 *
 *   f(x) = sum_{n=0}^{N-1} c_n * exp( -|x - p_n|^2 / (2 sigma^2) )
 *
 * with the N centres p_n drawn uniformly inside the unit ball and the strengths
 * c_n drawn uniformly from +-[0.5, 1]. Sampled by rejection from
 * numpy.random.default_rng(20260921), so the table below is reproducible.
 *
 *   max |c_n|        0.9738
 *   max |p_n|        0.9967      (touches the unit sphere)
 *   min |p_i - p_j|  0.0872      (two near-coincident pairs, deliberately kept)
 *   sum c_n         -0.4061      (NET CHARGE IS NOT ZERO -- see below)
 *
 * WHY THE NET CHARGE MATTERS
 *
 *   sum c_n != 0 means u ~ (sum c_n) / (4 pi r) at large r: a monopole tail.
 *   This is the term that a periodic or homogeneous-Dirichlet solver gets
 *   catastrophically wrong and that the LGF gets exactly right, so it is the
 *   part of the field worth putting a free-space solver in front of. In
 *   physical units the net charge is (sum c_n) * (2 pi sigma^2)^{3/2} = -1.0e-4
 *   at the default sigma -- small in absolute terms because the blobs are
 *   narrow, but it is the LEADING far-field term, so a solver with the wrong
 *   outer BC gets it wrong by O(1) relative. It does
 *   NOT affect || L_h u - f ||_inf, which stays at the round-off floor either
 *   way -- the LGF is an exact inverse of the discrete operator regardless of
 *   what the right-hand side looks like. It shows up only when comparing
 *   against MLMG, or against the analytic far field.
 *
 *   If this is ever reused as an NSE vorticity field it must be made net-zero
 *   (compactly supported vorticity integrates to zero); flip the sign of any
 *   one c_n and rescale, or subtract the mean over the blobs.
 *
 * COMPACT SUPPORT
 *
 *   Each blob is cut off hard at |x - p_n|^2 > R_SUPP2, where R_SUPP2 is set so
 *   the discarded value is exp(-30) = 9.4e-14. Below the tag threshold, below
 *   anything the solve can see -- but the point is that f is then EXACTLY zero
 *   (not merely small) outside the union of 32 balls of radius
 *
 *       r_supp = sqrt(2 * 30) * sigma = 7.746 * sigma
 *
 *   so the tagging has a crisp, unambiguous support to find and the tagged
 *   fraction is a property of the field, not of the threshold.
 *
 * CHOOSING SIGMA
 *
 *   sigma trades resolution against how much of the domain gets tagged. With
 *   the default sigma = 0.025 on [-1,1]^3 (r_supp = 0.1936, union of the 32
 *   balls = 11.4% of the domain by volume):
 *
 *       N      h         sigma/h   cells across the 3-sigma core
 *       128    1.563e-2  1.6        4.8
 *       192    1.042e-2  2.4        7.2
 *       256    7.813e-3  3.2        9.6
 *       512    3.906e-3  6.4       19.2
 *      1024    1.953e-3  12.8      38.4
 *
 *   Under-resolving a blob does not degrade the residual -- it only means the
 *   discrete solution is a poor approximation to the continuum one. So 128^3 is
 *   fine for a tagging/timing comparison; use 512^3 or 1024^3 when the picture
 *   itself is the deliverable. Shrink sigma to tighten the tagged fraction
 *   (fraction scales roughly as sigma^3 until the balls stop overlapping).
 *
 * TWO DIFFERENT SPEEDUPS -- DON'T CONFUSE THEM
 *
 *   directSumLGF costs O(N_tagged_src * N_tgt), so restricting the source to
 *   the tagged boxes cuts the work by exactly the tagged fraction. That is the
 *   "tagging catches the cluster" demo, and with CLUSTER_R = 1 it works
 *   straight away.
 *
 *   LGFOpenBC costs O(V log V) where V is the DOUBLED BOUNDING BOX of
 *   (target boxes) U (tagged boxes) -- see LGFOpenBC::makeDomain. With the
 *   centres filling the unit ball and the answer wanted on all of [-1,1]^3,
 *   that bounding box is the whole domain whether or not anything is tagged,
 *   so tagging buys nothing but a cheaper RHS copy. The FFT-side lever is not
 *   tagging the SOURCE, it is shrinking the region the answer is WANTED in --
 *   which is exactly what vor2vel() does in the NSE solver. To demo that here,
 *   set CLUSTER_R = 0.35 and build the target on tag_ba: the bounding box goes
 *   from 2.0 to ~1.09 a side, i.e. (2.0/1.09)^3 = 6.2x less transform volume.
 *
 * 2D
 *
 *   AMREX_D_TERM drops the z term, which projects the ball of centres onto the
 *   unit disc. The file compiles and runs unchanged in 2D; the projected centre
 *   density is ~sqrt(1-r^2) rather than uniform, which is harmless for a test.
 */

namespace lgf_cluster
{
    // number of blobs
    static constexpr int   N_BLOBS = 32;

    // Gaussian width. See "CHOOSING SIGMA" above.
    static constexpr amrex::Real SIGMA = 0.025;

    // Radius of the ball the centres are scattered in. 1.0 fills [-1,1]^3, so
    // the cluster's bounding box IS the domain and LGFOpenBC gets no benefit
    // from tagging. Set this to ~0.35 and solve on a snug target to get the
    // FFT-side speedup instead (see "TWO DIFFERENT SPEEDUPS" above).
    static constexpr amrex::Real CLUSTER_R = 1.0;

    // exp(-CUT) is the largest value thrown away by the compact-support cut
    static constexpr amrex::Real CUT = 30.0;

    // derived: 1/(2 sigma^2) and the squared support radius
    static constexpr amrex::Real INV_2S2 = 1.0 / (2.0 * SIGMA * SIGMA);
    static constexpr amrex::Real R_SUPP2 = 2.0 * CUT * SIGMA * SIGMA;

    // {p_x, p_y, p_z, c}
    static constexpr amrex::Real blob[N_BLOBS][4] = {
        {+0.55801859, -0.49209070, -0.03398098, -0.76474523},
        {+0.85009778, -0.08890864, +0.05708035, -0.71987671},
        {-0.58536576, -0.26284485, +0.33759874, -0.95207512},
        {-0.26019527, -0.70893890, +0.26044668, +0.82617556},
        {-0.11630771, +0.78397511, +0.11783537, +0.85436457},
        {-0.09681709, -0.10637322, -0.79998940, -0.65700925},
        {-0.12687944, -0.91010084, +0.08725856, +0.94267293},
        {-0.26014187, -0.63440167, +0.30575437, -0.70049099},
        {+0.50282261, +0.24676520, -0.20457838, +0.88351534},
        {+0.26332787, +0.72547416, -0.54607203, -0.70689561},
        {+0.23703246, +0.44286825, -0.62406843, +0.52400914},
        {+0.78694269, -0.55353965, +0.21961446, +0.60885934},
        {-0.49292191, +0.55471884, -0.06482887, +0.60991838},
        {+0.47305337, -0.12877911, +0.41777724, -0.62119788},
        {-0.36279843, +0.24023394, -0.00964007, -0.60344080},
        {+0.56641997, +0.07679785, -0.26278851, -0.97380122},
        {-0.33374314, -0.48179779, -0.78139920, +0.77638025},
        {+0.86000692, +0.36708408, +0.34505393, +0.92150558},
        {+0.33644570, +0.62537152, -0.27805657, -0.87327926},
        {-0.20470781, +0.02455467, +0.60280705, -0.79802695},
        {-0.07023751, -0.33934438, +0.44961144, -0.54003799},
        {-0.86583376, +0.33312644, -0.22753614, -0.85332310},
        {+0.40382086, -0.12057052, +0.57628931, +0.89464482},
        {-0.06779163, -0.04023548, +0.04708517, +0.66639502},
        {-0.70411348, -0.67244504, -0.05625701, -0.55200768},
        {+0.01574823, -0.27637199, -0.74200422, +0.69185442},
        {+0.17165419, +0.65833524, +0.61952489, -0.90196731},
        {+0.18864616, +0.74369398, +0.28082898, +0.57557916},
        {+0.06235840, -0.23110072, -0.56773725, -0.93632914},
        {-0.36464817, -0.00845577, -0.92702906, +0.59232026},
        {-0.66054078, +0.32708146, -0.52472139, +0.69603880},
        {+0.28679498, -0.62204981, -0.00475310, +0.68420477}
    };
}


// Source field evaluated inside a ParallelFor. Same signature as the single
// Gaussian it replaces, so initializeSourceMultiFab() needs no change.
//
// constexpr table, not a device pointer: the compiler materialises `blob`
// directly into the kernel image (constant bank / immediates), so there is
// nothing to allocate, nothing to copy and nothing to free. Cost is ~32 * 10
// flops per cell with the early-out taken most of the time -- about 0.05 s at
// 1024^3 on one A100, i.e. lost in the noise against the transform.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real sourceField(amrex::Real x, amrex::Real y, amrex::Real z)
{
    amrex::ignore_unused(y, z);

    amrex::Real f = amrex::Real(0.0);

    for (int n = 0; n < lgf_cluster::N_BLOBS; ++n)
    {
        AMREX_D_TERM(const amrex::Real ddx = x - lgf_cluster::CLUSTER_R * lgf_cluster::blob[n][0];,
                     const amrex::Real ddy = y - lgf_cluster::CLUSTER_R * lgf_cluster::blob[n][1];,
                     const amrex::Real ddz = z - lgf_cluster::CLUSTER_R * lgf_cluster::blob[n][2];)

        const amrex::Real r2 = AMREX_D_TERM(ddx*ddx, + ddy*ddy, + ddz*ddz);

        // hard cut -> f is EXACTLY zero outside the union of the blob supports
        if (r2 >= lgf_cluster::R_SUPP2) { continue; }

        f += lgf_cluster::blob[n][3] * std::exp(-r2 * lgf_cluster::INV_2S2);
    }

    return f;
}

#endif

// #ifndef SOURCEFIELD_H_
// #define SOURCEFIELD_H_

// #include <AMReX_Geometry.H>
// #include <AMReX_MultiFab.H>
// #include <AMReX_BCRec.H>
// #include <AMReX_MultiFab.H>
// #include <AMReX_Math.H>


// // function for computing lgf kernel inside parallelFor
// AMREX_GPU_DEVICE AMREX_FORCE_INLINE
// amrex::Real sourceField(amrex::Real x, amrex::Real y, amrex::Real z)
//     {
//         // parameters of the gaussian distribution
//         amrex::Real cent_x = 0.0;
//         #if AMREX_SPACEDIM == 2
//             amrex::Real cent_y = 0.25;
//         #endif    
//         #if AMREX_SPACEDIM == 3
//             amrex::Real cent_y = 0.25;
//             amrex::Real cent_z = 0.0;
//         #endif
//         amrex::Real variance = 0.5;

//         amrex::Real r_sq = AMREX_D_TERM((x - cent_x)*(x - cent_x),
//                                            + (y - cent_y)*(y - cent_y),
//                                            + (z - cent_z)*(z - cent_z));
        
//         amrex::Real f = std::exp(-r_sq / variance); 
        
//         return f;                                
//     }
// #endif