#ifndef FASTMULTIPOLEKERNEL_H_
#define FASTMULTIPOLEKERNEL_H_

#include <AMReX.H>
#include <AMReX_Math.H>
#include <AMReX_Box.H>
#include <AMReX_IntVect.H>
#include <AMReX_Array4.H>
#include <AMReX_GpuComplex.H>
#include <AMReX_GpuAtomic.H>

#include <directSumKernel.H>

// largest number of nodes an interval may hold per direction; it sizes the
// on-stack line buffers of the transfer kernels
inline constexpr int LGF_FMM_MAX_NODE = 65;

// the same for the half-stored spectral direction; the padded length stays
// within a few points of 2*LGF_FMM_MAX_NODE - 1
inline constexpr int LGF_FMM_MAX_HALF = LGF_FMM_MAX_NODE + 8;

namespace fmm_detail
{
    using Complex = amrex::GpuComplex<amrex::Real>;

    // one entry per (fab, region) to be zeroed; all regions of a level go out
    // in a single fused launch
    struct ZeroTag
    {
        amrex::Array4<amrex::Real> fab;
        amrex::Box                 region;

        [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        amrex::Box const& box () const noexcept { return region; }
    };

    // one entry per (target interval, slab of its spectrum). 'cross' is the
    // slab, one cell thick along the last direction, collapsed along x: the
    // kernel walks the x-line through each of its cells. [p_beg, p_end) are
    // the target's entries in the level's interaction lists
    struct SumTag
    {
        amrex::Array4<Complex> sum;
        amrex::Box             cross;
        int                    p_beg;
        int                    p_end;

        [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        amrex::Box const& box () const noexcept { return cross; }
    };

    // one entry per child interval and transfer direction. The launch runs
    // over 'cross', which is one cell thick along the transfer direction, and
    // the kernel walks the line through each of its cells
    struct LineTag
    {
        amrex::Array4<amrex::Real>       dst;
        amrex::Array4<amrex::Real const> src;
        amrex::Box                       cross;   // in dst indices
        amrex::IntVect                   shift;   // src index - dst index
        int                              parity;  // which child of its parent, along the line
        int                              n_in;    // nodes read along the line
        int                              n_out;   // nodes written along the line

        [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
        amrex::Box const& box () const noexcept { return cross; }
    };
}

// G(n) for a lattice offset passed as Reals. computeLGF() forms nx*nx + ...
// in int, which overflows for the strides of the upper tree levels, so every
// offset that cannot be in the lookup table goes straight to the expansion
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real fmmLatticeGreens(int n_lookup,
                             AMREX_D_DECL(amrex::Real nx, amrex::Real ny, amrex::Real nz))
{
    const amrex::Real n_max = amrex::max(AMREX_D_DECL(amrex::Math::abs(nx),
                                                      amrex::Math::abs(ny),
                                                      amrex::Math::abs(nz)));

    if (n_max < amrex::Real(n_lookup))
    {
        return computeLGF(n_lookup, AMREX_D_DECL(nx, ny, nz),
                                    AMREX_D_DECL(amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0)),
                                    AMREX_D_DECL(amrex::Real(1.0), amrex::Real(1.0), amrex::Real(1.0)));
    }

#if AMREX_SPACEDIM == 2
    return amrex::Real(farField2D(nx, ny));
#else
    return amrex::Real(farField3D(nx, ny, nz));
#endif
}

// position i of a circular kernel array of length n_fft -> signed node
// separation in [-(n_node-1), n_node-1]; flags the unused middle of the array
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
int fmmUnwrap(int i, int n_fft, int n_node, bool& is_inside) noexcept
{
    if (i < n_node) { return i; }
    if (i > n_fft - n_node) { return i - n_fft; }

    is_inside = false;
    return 0;
}

// wavenumber index of -k on a full (not half-stored) direction
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
int fmmNegate(int i, int n_fft) noexcept
{
    return (i == 0) ? 0 : n_fft - i;
}

// kernel of the block-wise convolution between two intervals of one level,
//   K(m) = G(stride * (off + m)),   m = target node - source node,
// laid out circularly. 'off' is the block offset times the block size and
// 'stride' the node spacing of the level, both in lattice units
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real fmmKernelValue(int i, int j, int k, int n_fft, int n_node, int n_lookup,
                           amrex::Real stride, amrex::Dim3 const& off)
{
    amrex::ignore_unused(k);

    bool is_inside = true;
    AMREX_D_TERM(const int mx = fmmUnwrap(i, n_fft, n_node, is_inside);,
                 const int my = fmmUnwrap(j, n_fft, n_node, is_inside);,
                 const int mz = fmmUnwrap(k, n_fft, n_node, is_inside);)

    if (!is_inside) { return amrex::Real(0.0); }

    return fmmLatticeGreens(n_lookup, AMREX_D_DECL(stride * amrex::Real(off.x + mx),
                                                   stride * amrex::Real(off.y + my),
                                                   stride * amrex::Real(off.z + mz)));
}

// zero-padded copy of one interval's nodes into the FFT work array
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real fmmPadSource(int i, int j, int k, amrex::Array4<amrex::Real const> const& src,
                         amrex::Dim3 const& lo, int n_src) noexcept
{
    amrex::ignore_unused(k);

    const bool is_inside = AMREX_D_TERM(i < n_src, && j < n_src, && k < n_src);

    return is_inside ? src(i + lo.x, j + lo.y, k + lo.z) : amrex::Real(0.0);
}

// one x-line of a target's spectrum: the sum over its interaction list of
// (source spectrum) * (kernel spectrum). The kernel is stored for
// non-negative block offsets only; a reflected offset reads the same
// spectrum at a reflected wavenumber, since G is even in each coordinate:
//   K_o(m) = K_|o|(s*m)  =>  K^_o(k) = K^_|o|(s*k),  s = sign(o) per direction.
// x is the half-stored direction, so its reflection goes through
//   K^(-kx, ky, kz) = conj(K^(kx, -ky, -kz))
// and a line along x is contiguous in both spectra whatever the reflection
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void fmmSumLine(int j, int k, fmm_detail::SumTag const& t,
                fmm_detail::Complex const* src_spec, fmm_detail::Complex const* ker_spec,
                int const* pair_slot, int const* pair_code, int const* slot_live,
                amrex::Long n_spec, int n_fft, int n_half) noexcept
{
    fmm_detail::Complex line[LGF_FMM_MAX_HALF];
    for (int i = 0; i < n_half; ++i) { line[i] = fmm_detail::Complex(0.0, 0.0); }

    for (int p = t.p_beg; p < t.p_end; ++p)
    {
        // intervals that hold no source in this solve have no spectrum
        if (!slot_live[pair_slot[p]]) { continue; }

        const int flip = pair_code[p] & 7;

        int jk = (flip & 2) ? fmmNegate(j, n_fft) : j;
#if AMREX_SPACEDIM == 3
        int kk = (flip & 4) ? fmmNegate(k, n_fft) : k;
#else
        int kk = 0;
#endif

        const bool is_conj = (flip & 1);
        if (is_conj)
        {
            jk = fmmNegate(jk, n_fft);
#if AMREX_SPACEDIM == 3
            kk = fmmNegate(kk, n_fft);
#endif
        }

        fmm_detail::Complex const* src = src_spec + pair_slot[p] * n_spec
                                       + n_half * (j + amrex::Long(n_fft) * k);
        fmm_detail::Complex const* ker = ker_spec + (pair_code[p] >> 3) * n_spec
                                       + n_half * (jk + amrex::Long(n_fft) * kk);

        if (is_conj)
        {
            for (int i = 0; i < n_half; ++i)
            {
                line[i] += src[i] * fmm_detail::Complex(ker[i].real(), -ker[i].imag());
            }
        }
        else
        {
            for (int i = 0; i < n_half; ++i)
            {
                line[i] += src[i] * ker[i];
            }
        }
    }

    for (int i = 0; i < n_half; ++i)
    {
        t.sum(i, j, k) = line[i];
    }
}

// Interp along direction 'dim': out(q) = sum_s w(q,s) in(beg(q) + s). The
// inputs are read into a line buffer first, so dst may alias src
template <int dim>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void fmmInterpLine(int i, int j, int k, fmm_detail::LineTag const& t,
                   amrex::Real const* w_val, int const* w_beg, int n_node, int n_interp) noexcept
{
    amrex::ignore_unused(k);

    amrex::Real line_in[LGF_FMM_MAX_NODE];

    amrex::IntVect iv_dst(AMREX_D_DECL(i, j, k));
    amrex::IntVect iv_src = iv_dst + t.shift;

    for (int p = 0; p < t.n_in; ++p)
    {
        line_in[p] = t.src(iv_src);
        ++iv_src[dim];
    }

    const int row = t.parity * n_node;
    for (int q = 0; q < t.n_out; ++q)
    {
        const amrex::Real* w = w_val + (row + q) * n_interp;
        const int beg = w_beg[row + q];

        amrex::Real val = 0.0;
        for (int s = 0; s < n_interp; ++s)
        {
            val += w[s] * line_in[beg + s];
        }

        t.dst(iv_dst) = val;
        ++iv_dst[dim];
    }
}

// Reg along direction 'dim', the adjoint of fmmInterpLine:
// out(beg(q) + s) += w(q,s) in(q). With is_accumulate the result is added
// to dst atomically, since the children of one parent all write to it
template <int dim>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void fmmRegLine(int i, int j, int k, fmm_detail::LineTag const& t,
                amrex::Real const* w_val, int const* w_beg, int n_node, int n_interp,
                bool is_accumulate) noexcept
{
    amrex::ignore_unused(k);

    amrex::Real line_in[LGF_FMM_MAX_NODE];
    amrex::Real line_out[LGF_FMM_MAX_NODE];

    amrex::IntVect iv_dst(AMREX_D_DECL(i, j, k));
    amrex::IntVect iv_src = iv_dst + t.shift;

    for (int q = 0; q < t.n_in; ++q)
    {
        line_in[q] = t.src(iv_src);
        ++iv_src[dim];
    }

    for (int p = 0; p < t.n_out; ++p) { line_out[p] = 0.0; }

    const int row = t.parity * n_node;
    for (int q = 0; q < t.n_in; ++q)
    {
        const amrex::Real* w = w_val + (row + q) * n_interp;
        const int beg = w_beg[row + q];

        for (int s = 0; s < n_interp; ++s)
        {
            line_out[beg + s] += w[s] * line_in[q];
        }
    }

    for (int p = 0; p < t.n_out; ++p)
    {
        if (is_accumulate)
        {
            amrex::Gpu::Atomic::AddNoRet(&t.dst(iv_dst), line_out[p]);
        }
        else
        {
            t.dst(iv_dst) = line_out[p];
        }
        ++iv_dst[dim];
    }
}

#endif
