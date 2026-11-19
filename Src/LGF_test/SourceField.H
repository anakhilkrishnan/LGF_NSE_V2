#ifndef SOURCEFIELD_H_
#define SOURCEFIELD_H_

#include <AMReX_REAL.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>

#include <cmath>

// ---------------------------------------------------------------------------
// Source fields for the Poisson test bench. Every case follows the same contract:
//   x, y, z : physical coordinates (the 2D build passes z = 0)
//   returns : f(x, y, z)
// f is compactly supported with a hard cut, so it is EXACTLY zero outside a
// known region and tagging finds a crisp support. Coordinates are absolute;
// the cases are laid out for the default domain [-5, 5]^d. Parameters live
// inside each case; edit them there. The cases are listed by name in
// TestSuite.H (loadTestCase), and sourceField() at the bottom dispatches.
// ---------------------------------------------------------------------------

namespace lgf_source
{
    // exp(-CUT) is the largest value thrown away by any compact-support cut
    static constexpr amrex::Real CUT = 30.0;
}

// strength * exp(-r2 / (2 sigma^2)), exactly zero where that is below exp(-CUT)
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real cutGaussian (amrex::Real r2, amrex::Real sigma, amrex::Real strength) noexcept
{
    const amrex::Real arg = r2 / (amrex::Real(2.0) * sigma * sigma);

    return (arg >= lgf_source::CUT) ? amrex::Real(0.0) : strength * std::exp(-arg);
}

// squared distance from (x, y, z) to (cx, cy, cz); z terms drop out in 2D
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real distanceSquared (amrex::Real x, amrex::Real y, amrex::Real z,
                             amrex::Real cx, amrex::Real cy, amrex::Real cz) noexcept
{
    amrex::ignore_unused(z, cz);

    return AMREX_D_TERM((x - cx)*(x - cx), + (y - cy)*(y - cy), + (z - cz)*(z - cz));
}

// ---------------------------------------------------------------------------
// single_blob: one Gaussian, centre off the lattice. The plain monopole,
// u ~ -1/(4 pi r) far away; the smallest support of the suite.
// ---------------------------------------------------------------------------
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real singleBlobSource (amrex::Real x, amrex::Real y, amrex::Real z) noexcept
{
    constexpr amrex::Real sigma = 0.1;
    constexpr amrex::Real cx = 0.1, cy = 0.25, cz = -0.15;

    return cutGaussian(distanceSquared(x, y, z, cx, cy, cz), sigma, 1.0);
}

// ---------------------------------------------------------------------------
// dipole: two opposite Gaussians, zero net charge. The monopole cancels, so
// the far field is the much weaker dipole term; checks that the far-field
// approximation of a solver does not leave a monopole-sized error behind.
// ---------------------------------------------------------------------------
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real dipoleSource (amrex::Real x, amrex::Real y, amrex::Real z) noexcept
{
    constexpr amrex::Real sigma = 0.1;
    constexpr amrex::Real cx = 0.05, cy = 0.1, cz = -0.05;
    constexpr amrex::Real half_sep = 0.3;      // along x

    return cutGaussian(distanceSquared(x, y, z, cx - half_sep, cy, cz), sigma,  1.0)
         + cutGaussian(distanceSquared(x, y, z, cx + half_sep, cy, cz), sigma, -1.0);
}

/**
 * blob_cluster: CLUSTER OF BLOBS
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
 *   the tagged boxes cuts the work by exactly the tagged fraction.
 *
 *   LGFOpenBC costs O(V log V) where V is the doubled bounding box of the
 *   TARGET layout, so its lever is shrinking the region the answer is wanted
 *   in, which is what vor2vel() does in the NSE solver. The bench now does
 *   the same: with layout = snug the target is the tagged boxes grown by
 *   n_buffer_box boxes (see buildCaseLayout), so the cluster's bounding box,
 *   not the domain's, is what the FFT pays for. layout = full restores the
 *   whole-domain target.
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

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real blobClusterSource (amrex::Real x, amrex::Real y, amrex::Real z) noexcept
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

// ---------------------------------------------------------------------------
// ring: Gaussian-core torus of radius R in the xy-plane, the shape of a
// vortex ring's vorticity support (an annulus in 2D). The bounding box is
// mostly empty, more so for the snug layout than for a blob of the same size.
// ---------------------------------------------------------------------------
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real ringSource (amrex::Real x, amrex::Real y, amrex::Real z) noexcept
{
    amrex::ignore_unused(z);

    constexpr amrex::Real R = 1.0;                  // ring radius
    constexpr amrex::Real sigma = 0.1;              // core
    constexpr amrex::Real cx = 0.05, cy = -0.1, cz = 0.1;

    const amrex::Real rho = std::sqrt((x - cx)*(x - cx) + (y - cy)*(y - cy));
#if AMREX_SPACEDIM == 3
    const amrex::Real s2 = (rho - R)*(rho - R) + (z - cz)*(z - cz);
#else
    amrex::ignore_unused(cz);
    const amrex::Real s2 = (rho - R)*(rho - R);
#endif

    return cutGaussian(s2, sigma, 1.0);
}

// ---------------------------------------------------------------------------
// shell: Gaussian shell of radius R around a point. A large, hollow support:
// the tagged region is a thin surface and the interior holds no source.
// ---------------------------------------------------------------------------
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real shellSource (amrex::Real x, amrex::Real y, amrex::Real z) noexcept
{
    constexpr amrex::Real R = 2.0;
    constexpr amrex::Real sigma = 0.08;
    constexpr amrex::Real cx = -0.1, cy = 0.05, cz = 0.1;

    const amrex::Real r = std::sqrt(distanceSquared(x, y, z, cx, cy, cz));

    return cutGaussian((r - R)*(r - R), sigma, 1.0);
}

// ---------------------------------------------------------------------------
// distant_pair: two small clusters of three blobs in opposite corners, of
// opposite sign. The support is tiny and its bounding box is most of the
// domain: the sparse layout an FFT pays the most for.
// ---------------------------------------------------------------------------
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real distantPairSource (amrex::Real x, amrex::Real y, amrex::Real z) noexcept
{
    constexpr amrex::Real sigma = 0.05;
    constexpr amrex::Real corner = 3.5;
    // blob offsets within a cluster, and their strengths
    constexpr amrex::Real offset[3][4] = {
        { 0.00,  0.00,  0.00, 1.0},
        { 0.15, -0.05,  0.10, 0.6},
        {-0.10,  0.12, -0.08, 0.8}
    };

    amrex::Real f = amrex::Real(0.0);
    for (int n = 0; n < 3; ++n)
    {
        f += cutGaussian(distanceSquared(x, y, z, -corner + offset[n][0], -corner + offset[n][1], -corner + offset[n][2]),
                         sigma,  offset[n][3]);
        f += cutGaussian(distanceSquared(x, y, z,  corner + offset[n][0],  corner + offset[n][1],  corner + offset[n][2]),
                         sigma, -offset[n][3]);
    }

    return f;
}

// ---------------------------------------------------------------------------
// The cases, by id; TestSuite.H maps names to these. Add a case by writing
// its function above, adding an id here (before n_cases) and a line in
// sourceField(), and listing it in loadTestCase().
// ---------------------------------------------------------------------------
enum class SourceCase : int
{
    single_blob = 0,
    dipole,
    blob_cluster,
    ring,
    shell,
    distant_pair,
    n_cases
};

// f(x, y, z) for the given case; called from inside a ParallelFor
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real sourceField (SourceCase source_case, amrex::Real x, amrex::Real y, amrex::Real z) noexcept
{
    switch (source_case)
    {
        case SourceCase::single_blob:  return singleBlobSource(x, y, z);
        case SourceCase::dipole:       return dipoleSource(x, y, z);
        case SourceCase::blob_cluster: return blobClusterSource(x, y, z);
        case SourceCase::ring:         return ringSource(x, y, z);
        case SourceCase::shell:        return shellSource(x, y, z);
        case SourceCase::distant_pair: return distantPairSource(x, y, z);
        default:                       return amrex::Real(0.0);
    }
}

#endif
