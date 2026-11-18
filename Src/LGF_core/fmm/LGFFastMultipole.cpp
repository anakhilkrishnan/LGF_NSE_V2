#include <LGFFastMultipole.H>

LGFFastMultipole::LGFFastMultipole(const amrex::Geometry& geom_in, const amrex::BoxArray& xsoln_ba_in,
                                   int n_look_in, int n_ghost_max_in, int n_block_in, int n_interp_in)
    : geom(geom_in), xsoln_ba_ref(xsoln_ba_in), n_lookup(n_look_in), n_ghost_max(n_ghost_max_in),
      n_block(n_block_in), n_interp(n_interp_in), is_tree_built(false), is_plan_built(false)
{
    // targets convolved per launch: enough work to fill a device, few enough
    // on the host for one slab of the spectra involved to stay in cache
#ifdef AMREX_USE_GPU
    n_batch = 64;
#else
    n_batch = 8;
#endif

    AMREX_ALWAYS_ASSERT(n_ghost_max >= 0 && n_interp >= 2);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(n_block >= 4,
        "LGFFastMultipole: n_block < 4 -- the fifth argument is the FMM block size, not LGFOpenBC's len_quantum");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(n_block + 1 <= LGF_FMM_MAX_NODE,
        "LGFFastMultipole: n_block + 1 exceeds LGF_FMM_MAX_NODE");

    // one-point overlap: a parent's nodes are every other node of its children
    n_node   = n_block + 1;
    n_interp = amrex::min(n_interp, n_node);

    // a linear convolution of two n_node-long sequences needs 2*n_node - 1
    n_fft  = nextFastLength(2 * n_node - 1);
    n_half = n_fft / 2 + 1;
    AMREX_ALWAYS_ASSERT(n_half <= LGF_FMM_MAX_HALF);

    pad_box  = amrex::Box(amrex::IntVect(0), amrex::IntVect(n_fft - 1));
    spec_box = pad_box;
    spec_box.setBig(0, n_half - 1);
    n_pad  = pad_box.numPts();
    n_spec = spec_box.numPts();

    // the plan is tied to these two arrays, so every transform goes through them
    pad_work.resize(n_pad);
    spec_work.resize(n_spec);
    fft = std::make_unique<amrex::FFT::LocalR2C<amrex::Real>>(pad_box.length(), pad_work.data(), spec_work.data());

    buildInterpTables();

    canon_ba = canonicalBoxArray(xsoln_ba_ref);

    // no tree yet; the first solve builds it, once the DistributionMapping is known
}

amrex::BoxArray LGFFastMultipole::retypeBoxArray(const amrex::BoxArray& ba, amrex::IndexType type)
{
    amrex::BoxList bl(type);
    bl.reserve(ba.size());
    for (int i = 0; i < ba.size(); ++i)
    {
        const amrex::Box& b = ba[i];
        bl.push_back(amrex::Box(b.smallEnd(), b.bigEnd(), type));
    }

    return amrex::BoxArray(std::move(bl));
}

amrex::BoxArray LGFFastMultipole::canonicalBoxArray(const amrex::BoxArray& ba)
{
    // convert() only ever increments hi, so the nodal boxes hold the indices of every type
    return retypeBoxArray(amrex::convert(ba, amrex::IndexType::TheNodeType()), amrex::IndexType::TheCellType());
}

amrex::BoxArray LGFFastMultipole::layoutTagBoxArray(const amrex::BoxArray& tag_ba) const
{
    // a cell-centred tag box names indices of the layout, whatever its type
    if (!tag_ba.ixType().cellCentered()) { return tag_ba; }

    return retypeBoxArray(tag_ba, xsoln_ba_ref.ixType());
}

amrex::BoxArray LGFFastMultipole::blockBoxArray(const amrex::BoxArray& ba, int ratio)
{
    // sort + unique rather than removeOverlap(): the coarsened boxes are all
    // duplicates of one another, and the order must not depend on the rank
    std::vector<amrex::IntVect> blk;
    for (int i = 0; i < ba.size(); ++i)
    {
        amrex::LoopOnCpu(amrex::coarsen(ba[i], ratio), [&] (int ii, int jj, int kk)
        {
            amrex::ignore_unused(kk);
            blk.push_back(amrex::IntVect(AMREX_D_DECL(ii, jj, kk)));
        });
    }

    // x runs fastest, so consecutive blocks are neighbours
    std::sort(blk.begin(), blk.end(), [] (const amrex::IntVect& a, const amrex::IntVect& b)
    {
        for (int d = AMREX_SPACEDIM - 1; d >= 0; --d)
        {
            if (a[d] != b[d]) { return a[d] < b[d]; }
        }
        return false;
    });
    blk.erase(std::unique(blk.begin(), blk.end()), blk.end());

    amrex::BoxList bl;
    bl.reserve(blk.size());
    for (const amrex::IntVect& b : blk)
    {
        bl.push_back(amrex::Box(b, b));
    }

    return amrex::BoxArray(std::move(bl));
}

int LGFFastMultipole::nextFastLength(int n)
{
    for (int len = n; ; ++len)
    {
        int rem = len;
        for (int p : {2, 3, 5})
        {
            while (rem % p == 0) { rem /= p; }
        }
        if (rem == 1) { return len; }
    }
}

amrex::IntVect LGFFastMultipole::lowEnd(const amrex::BoxArray& ba, int i)
{
    const amrex::Box b = ba[i];
    return b.smallEnd();
}

amrex::Box LGFFastMultipole::crossBox(const amrex::IntVect& lo, int dim, int n_lo, int n_hi)
{
    amrex::IntVect len(1);
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        if (d < dim) { len[d] = n_lo; }
        if (d > dim) { len[d] = n_hi; }
    }

    return amrex::Box(lo, lo + len - amrex::IntVect(1));
}

void LGFFastMultipole::buildInterpTables()
{
    // child node q of the child with parity c sits at x = c*n_block + q, in
    // child spacings from the parent's first node; parent node p sits at 2p
    const int n_row = 2 * n_node;
    amrex::Vector<amrex::Real> h_val(n_row * n_interp, 0.0);
    amrex::Vector<int> h_beg(n_row, 0);

    for (int c = 0; c < 2; ++c)
    {
        for (int q = 0; q < n_node; ++q)
        {
            const int row = c * n_node + q;
            const int x   = c * n_block + q;

            if (x % 2 == 0)
            {
                // coincides with a parent node
                const int beg = amrex::min(x / 2, n_node - n_interp);
                h_beg[row] = beg;
                h_val[row * n_interp + (x / 2 - beg)] = 1.0;
                continue;
            }

            // midway between parent nodes (x-1)/2 and (x+1)/2: the n_interp
            // nearest nodes, shifted inwards next to the ends of the interval
            const int beg = amrex::Clamp((x - 1) / 2 - (n_interp - 1) / 2, 0, n_node - n_interp);
            h_beg[row] = beg;

            const long double xi = 0.5L * x;
            for (int s = 0; s < n_interp; ++s)
            {
                long double w = 1.0L;
                for (int t = 0; t < n_interp; ++t)
                {
                    if (t != s) { w *= (xi - (beg + t)) / static_cast<long double>(s - t); }
                }
                h_val[row * n_interp + s] = static_cast<amrex::Real>(w);
            }
        }
    }

    interp_val.resize(h_val.size());
    interp_beg.resize(h_beg.size());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_val.begin(), h_val.end(), interp_val.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_beg.begin(), h_beg.end(), interp_beg.begin());
}

void LGFFastMultipole::buildTree()
{
    BL_PROFILE("<Setup> LGFFastMultipole::buildTree()");

    // the views alias src[0] and tar[0], so they go first
    views.clear();
    levels.clear();
    is_plan_built = false;

    // level 0: every block holding a target index of any index type, ghosts included
    amrex::BoxArray target_ba = canon_ba;
    target_ba.grow(n_ghost_max);
    amrex::BoxArray blk_ba = blockBoxArray(target_ba, n_block);

    while (true)
    {
        fmm_detail::Level& lev = levels.emplace_back();
        lev.blk_ba  = blk_ba;
        lev.n_data  = (levels.size() == 1) ? n_block : n_node;
        lev.node_ba = amrex::refine(blk_ba, lev.n_data);

        // stop once all parents are mutual neighbours: nothing is left to separate
        amrex::BoxArray par_ba = blockBoxArray(blk_ba, 2);
        if (par_ba.minimalBox().length().allLE(amrex::IntVect(2))) { break; }
        blk_ba = std::move(par_ba);
    }

    const int n_lev = numLevels();
    std::vector<std::pair<int, amrex::Box>> isects;

    // level 0 lives with the xsoln box it overlaps most, so copy in/out stay
    // local. canon_ba is cell typed like the blocks, and in xsoln's box order
    {
        fmm_detail::Level& leaf = levels[0];
        amrex::Vector<int> pmap(leaf.node_ba.size());
        for (int i = 0; i < leaf.node_ba.size(); ++i)
        {
            canon_ba.intersections(leaf.node_ba[i], isects, false, amrex::IntVect(n_ghost_max));
            AMREX_ALWAYS_ASSERT(!isects.empty());

            int best = 0;
            for (int m = 1; m < static_cast<int>(isects.size()); ++m)
            {
                const amrex::Long n_m = isects[m].second.numPts();
                const amrex::Long n_b = isects[best].second.numPts();
                if (n_m > n_b || (n_m == n_b && isects[m].first < isects[best].first)) { best = m; }
            }
            pmap[i] = xsoln_dm_ref[isects[best].first];
        }
        leaf.dm = amrex::DistributionMapping(std::move(pmap));
    }

    // a parent lives with its first child
    for (int l = 0; l < n_lev - 1; ++l)
    {
        fmm_detail::Level& child = levels[l];
        fmm_detail::Level& par   = levels[l + 1];

        child.par_idx.resize(child.blk_ba.size());
        amrex::Vector<int> par_pmap(par.blk_ba.size(), -1);
        for (int i = 0; i < child.blk_ba.size(); ++i)
        {
            const amrex::IntVect b_par = amrex::coarsen(lowEnd(child.blk_ba, i), 2);
            par.blk_ba.intersections(amrex::Box(b_par, b_par), isects, true, amrex::IntVect(0));
            AMREX_ALWAYS_ASSERT(!isects.empty());

            const int i_par = isects[0].first;
            child.par_idx[i] = i_par;
            if (par_pmap[i_par] < 0) { par_pmap[i_par] = child.dm[i]; }
        }

        amrex::Vector<int> child_par_pmap(child.blk_ba.size());
        for (int i = 0; i < child.blk_ba.size(); ++i)
        {
            child_par_pmap[i] = par_pmap[child.par_idx[i]];
        }

        par.dm = amrex::DistributionMapping(std::move(par_pmap));

        child.is_par_local = (child_par_pmap == child.dm.ProcessorMap());
        child.par_dm = child.is_par_local ? child.dm : amrex::DistributionMapping(std::move(child_par_pmap));
    }

    amrex::Long n_interval = 0;
    for (int l = 0; l < n_lev; ++l)
    {
        fmm_detail::Level& lev = levels[l];

        // level 0 keeps one ghost so the overlap node has a slot during the transfers
        const int n_grow = (l == 0) ? 1 : 0;
        lev.src.define(lev.node_ba, lev.dm, 1, 0);
        lev.tar.define(lev.node_ba, lev.dm, 1, n_grow);
        if (l < n_lev - 1) { lev.aux.define(lev.node_ba, lev.par_dm, 1, n_grow); }

        n_interval += lev.blk_ba.size();
    }

    // kernel spectra are keyed on the level alone; keep what is already there
    const int n_ker_id = 1 << (2 * AMREX_SPACEDIM);
    while (static_cast<int>(ker_slot.size()) < n_lev)
    {
        ker_spec.emplace_back();
        ker_slot.emplace_back(n_ker_id, -1);
        n_ker.push_back(0);
    }

    is_tree_built = true;

    amrex::Print() << "LGFFastMultipole: built tree with " << n_lev << " levels, "
                   << levels[0].blk_ba.size() << " blocks of " << n_block << " on level 0, "
                   << n_interval << " intervals in total"
                   << " (n_node " << n_node << ", n_interp " << n_interp << ", n_fft " << n_fft << ")\n";
}

void LGFFastMultipole::buildSourceMasks(const amrex::BoxArray& tag_ba)
{
    // on level 0, any block touching the support under any index type; the
    // per-type restriction is applied to the data in copySource()
    const amrex::BoxArray keep = canonicalBoxArray(layoutTagBoxArray(tag_ba));

    fmm_detail::Level& leaf = levels[0];
    leaf.src_mask.resize(leaf.node_ba.size());
    for (int i = 0; i < leaf.node_ba.size(); ++i)
    {
        leaf.src_mask[i] = keep.intersects(leaf.node_ba[i]) ? 1 : 0;
    }

    // a parent holds sources if any of its children does
    for (int l = 0; l < numLevels() - 1; ++l)
    {
        const fmm_detail::Level& child = levels[l];
        fmm_detail::Level& par = levels[l + 1];

        par.src_mask.assign(par.blk_ba.size(), 0);
        for (int i = 0; i < child.blk_ba.size(); ++i)
        {
            if (child.src_mask[i]) { par.src_mask[child.par_idx[i]] = 1; }
        }
    }
}

void LGFFastMultipole::buildInteractionLists(int lev)
{
    fmm_detail::Level& level = levels[lev];
    const int my_proc = amrex::ParallelDescriptor::MyProc();
    const int n_procs = amrex::ParallelDescriptor::NProcs();

    amrex::Vector<int> h_src;    // source interval of each pair
    amrex::Vector<int> h_code;   // (kernel slot << 3) | reflection bits
    amrex::Vector<int> halo_idx; // source intervals to pull, in order of first use
    amrex::Vector<int> use(level.blk_ba.size(), 0);

    level.pair_ptr.assign(1, 0);
    std::vector<std::pair<int, amrex::Box>> isects;

    for (amrex::MFIter mfi(level.tar); mfi.isValid(); ++mfi)
    {
        // everything worth a convolution on this level sits among the
        // children of the parent's neighbours
        const amrex::IntVect b_tar = lowEnd(level.blk_ba, mfi.index());
        const amrex::IntVect b_par = amrex::coarsen(b_tar, 2);
        const amrex::Box window = amrex::refine(amrex::grow(amrex::Box(b_par, b_par), 1), 2);

        level.blk_ba.intersections(window, isects);
        for (auto const& is : isects)
        {
            const int i_src = is.first;
            if (!level.src_mask[i_src]) { continue; }

            // near neighbours are resolved on level 0 only; above it they
            // are left to the children
            const amrex::IntVect off = b_tar - lowEnd(level.blk_ba, i_src);
            const bool is_near = off.allGE(amrex::IntVect(-1)) && off.allLE(amrex::IntVect(1));
            if (is_near && lev > 0) { continue; }

            // kernels are stored for |off|; the signs become wavenumber reflections
            int ker_id = 0;
            int flip = 0;
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                ker_id |= std::abs(off[d]) << (2 * d);
                if (off[d] < 0) { flip |= (1 << d); }
            }
            if (ker_slot[lev][ker_id] < 0) { buildKernel(lev, ker_id); }

            if (use[i_src] == 0)
            {
                use[i_src] = 1;
                if (level.dm[i_src] != my_proc) { halo_idx.push_back(i_src); }
            }

            h_src.push_back(i_src);
            h_code.push_back((ker_slot[lev][ker_id] << 3) | flip);
        }

        level.pair_ptr.push_back(static_cast<int>(h_src.size()));
    }

    // spectrum slots: own sources first, in MFIter order, then the pulled ones
    level.slot_idx.assign(level.blk_ba.size(), -1);
    level.n_slot_own = 0;
    for (amrex::MFIter mfi(level.src); mfi.isValid(); ++mfi)
    {
        if (use[mfi.index()]) { level.slot_idx[mfi.index()] = level.n_slot_own++; }
    }
    level.n_slot = level.n_slot_own;
    for (int i_src : halo_idx) { level.slot_idx[i_src] = level.n_slot++; }

    for (int& s : h_src) { s = level.slot_idx[s]; }

    level.pair_slot.resize(h_src.size());
    level.pair_code.resize(h_code.size());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_src.begin(), h_src.end(), level.pair_slot.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_code.begin(), h_code.end(), level.pair_code.begin());

    // halo layout: every rank lists the boxes it wants, the lists are
    // concatenated in rank order, and each box is owned by the rank that
    // asked for it. A box wanted by several ranks appears once per rank
    amrex::Vector<amrex::Box> halo_boxes;
    halo_boxes.reserve(halo_idx.size());
    for (int i_src : halo_idx) { halo_boxes.push_back(level.node_ba[i_src]); }

    amrex::Vector<int> n_halo(n_procs, 0);
    n_halo[my_proc] = static_cast<int>(halo_boxes.size());
    amrex::ParallelDescriptor::ReduceIntSum(n_halo.data(), n_procs);
    amrex::AllGatherBoxes(halo_boxes);

    level.halo.clear();
    if (!halo_boxes.empty())
    {
        amrex::Vector<int> pmap;
        pmap.reserve(halo_boxes.size());
        for (int p = 0; p < n_procs; ++p)
        {
            pmap.insert(pmap.end(), n_halo[p], p);
        }

        amrex::BoxList bl(std::move(halo_boxes));
        level.halo.define(amrex::BoxArray(std::move(bl)), amrex::DistributionMapping(std::move(pmap)), 1, 0);
    }
}

void LGFFastMultipole::buildSourcePlan(const amrex::BoxArray& tag_ba)
{
    BL_PROFILE("<Setup> LGFFastMultipole::buildSourcePlan()");

    buildSourceMasks(tag_ba);

    amrex::Long n_pair = 0;
    amrex::Long n_halo = 0;
    for (int l = 0; l < numLevels(); ++l)
    {
        buildInteractionLists(l);

        n_pair += levels[l].pair_slot.size();
        n_halo += levels[l].halo.isDefined() ? levels[l].halo.size() : 0;
    }
    amrex::ParallelDescriptor::ReduceLongSum(n_pair);

    cached_tag_ba = tag_ba;
    is_plan_built = true;

    amrex::Long n_src = 0;
    for (int m : levels[0].src_mask) { n_src += m; }

    amrex::Print() << "LGFFastMultipole: built source plan, " << n_src << " of " << levels[0].blk_ba.size()
                   << " level-0 blocks hold sources, " << n_pair << " block convolutions per solve, "
                   << n_halo << " intervals pulled across ranks\n";
}

fmm_detail::TypedView& LGFFastMultipole::typedView(amrex::IndexType type)
{
    for (fmm_detail::TypedView& view : views)
    {
        if (view.type == type) { return view; }
    }

    // same boxes and the same memory as src[0] and tar[0], under the caller's
    // index type, so ParallelCopy to and from the caller needs no retype copy
    fmm_detail::Level& leaf = levels[0];
    const amrex::BoxArray typed_ba = retypeBoxArray(leaf.node_ba, type);

    fmm_detail::TypedView& view = views.emplace_back();
    view.type = type;
    view.src.define(typed_ba, leaf.dm, 1, leaf.src.nGrowVect(), amrex::MFInfo().SetAlloc(false));
    view.tar.define(typed_ba, leaf.dm, 1, leaf.tar.nGrowVect(), amrex::MFInfo().SetAlloc(false));

    for (amrex::MFIter mfi(leaf.src); mfi.isValid(); ++mfi)
    {
        view.src.setFab(mfi, amrex::FArrayBox(leaf.src.array(mfi), type));
        view.tar.setFab(mfi, amrex::FArrayBox(leaf.tar.array(mfi), type));
    }

    return view;
}

void LGFFastMultipole::buildKernel(int lev, int ker_id)
{
    BL_PROFILE("<Setup> LGFFastMultipole::buildKernel()");

    // grow the level's store a few spectra at a time
    const int slot = n_ker[lev]++;
    ker_slot[lev][ker_id] = slot;
    if (slot % 8 == 0) { ker_spec[lev].reserve((slot + 8) * n_spec); }
    ker_spec[lev].resize((slot + 1) * n_spec);

    // plain data for passing into GPU lambda
    const amrex::IntVect blk_off(AMREX_D_DECL(ker_id & 3, (ker_id >> 2) & 3, (ker_id >> 4) & 3));
    const amrex::Dim3 off = (blk_off * n_block).dim3();
    const amrex::Real stride = std::ldexp(amrex::Real(1.0), lev);   // node spacing, in lattice points
    const int nft = n_fft;
    const int nnd = n_node;
    const int nlk = n_lookup;

    auto const& pad = amrex::makeArray4(pad_work.data(), pad_box, 1);
    amrex::ParallelFor(pad_box,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            pad(i, j, k) = fmmKernelValue(i, j, k, nft, nnd, nlk, stride, off);
        });

    fft->forward(pad_work.data(), spec_work.data());

    fmm_detail::Complex const* work = spec_work.data();
    fmm_detail::Complex* store = ker_spec[lev].data() + slot * n_spec;
    amrex::ParallelFor(n_spec,
        [=] AMREX_GPU_DEVICE (amrex::Long m) noexcept
        {
            store[m] = work[m];
        });
}

void LGFFastMultipole::copySource(const amrex::MultiFab& source, const amrex::BoxArray& tag_ba)
{
    BL_PROFILE("<Compute> LGFFastMultipole::copySource()");

    fmm_detail::Level& leaf = levels[0];
    amrex::MultiFab& view = typedView(source.ixType()).src;

    view.ParallelCopy(source, 0, 0, 1, amrex::IntVect(0), amrex::IntVect(0));

    // the support under this index type. The plan's masks cover every type;
    // blocks reached only by a hi+1 plane this type does not have are dead.
    // convert() is the identity for tag boxes that already have the type
    const amrex::BoxArray keep = amrex::convert(layoutTagBoxArray(tag_ba), source.ixType());
    const amrex::BoxArray& typed_ba = view.boxArray();

    leaf.src_live.assign(typed_ba.size(), 0);
    for (int i = 0; i < typed_ba.size(); ++i)
    {
        if (leaf.src_mask[i] && keep.intersects(typed_ba[i])) { leaf.src_live[i] = 1; }
    }

    // inside the live blocks, the complement of the support must be zero,
    // not whatever the caller holds there
    amrex::Vector<fmm_detail::ZeroTag> tags;
    for (amrex::MFIter mfi(view); mfi.isValid(); ++mfi)
    {
        if (!leaf.src_live[mfi.index()]) { continue; }

        const amrex::BoxList outside = keep.complementIn(mfi.validbox());
        if (outside.isEmpty()) { continue; }

        auto const& fab = view.array(mfi);
        for (const amrex::Box& b : outside)
        {
            tags.push_back({fab, b});
        }
    }

    if (tags.empty()) { return; }

    amrex::ParallelFor(tags,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, fmm_detail::ZeroTag const& t) noexcept
        {
            t.fab(i, j, k) = 0.0;
        });
}

template <int dim>
void LGFFastMultipole::regularizePass(int lev)
{
    fmm_detail::Level& child = levels[lev];
    fmm_detail::Level& par   = levels[lev + 1];

    // directions below dim already hold parent nodes, those above it still
    // hold child nodes. The last direction lands in the parent itself
    const bool is_last = (dim == AMREX_SPACEDIM - 1);

    amrex::Vector<fmm_detail::LineTag> tags;
    for (amrex::MFIter mfi(child.aux); mfi.isValid(); ++mfi)
    {
        const int i_child = mfi.index();
        if (!child.src_live[i_child]) { continue; }

        const int i_par = child.par_idx[i_child];
        const amrex::IntVect lo_child = lowEnd(child.node_ba, i_child);
        const amrex::IntVect lo_par   = lowEnd(par.node_ba, i_par);
        const int parity = lowEnd(child.blk_ba, i_child)[dim] - 2 * lowEnd(par.blk_ba, i_par)[dim];

        auto const& aux = child.aux.array(mfi);
        if (is_last)
        {
            tags.push_back({par.src.array(i_par), aux, crossBox(lo_par, dim, n_node, child.n_data),
                            lo_child - lo_par, parity, child.n_data, n_node});
        }
        else
        {
            tags.push_back({aux, aux, crossBox(lo_child, dim, n_node, child.n_data),
                            amrex::IntVect(0), parity, child.n_data, n_node});
        }
    }

    if (tags.empty()) { return; }

    // plain data for passing into GPU lambda
    amrex::Real const* w_val = interp_val.data();
    int const* w_beg = interp_beg.data();
    const int nnd = n_node;
    const int nip = n_interp;

    amrex::ParallelFor(tags,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, fmm_detail::LineTag const& t) noexcept
        {
            fmmRegLine<dim>(i, j, k, t, w_val, w_beg, nnd, nip, is_last);
        });
}

template <int... dim>
void LGFFastMultipole::regularizeLevel(int lev, std::integer_sequence<int, dim...>)
{
    BL_PROFILE("<Compute> LGFFastMultipole::regularizeLevel()");

    fmm_detail::Level& child = levels[lev];
    fmm_detail::Level& par   = levels[lev + 1];

    // a parent holds sources if any of its children does
    par.src_live.assign(par.blk_ba.size(), 0);
    for (int i = 0; i < child.blk_ba.size(); ++i)
    {
        if (child.src_live[i]) { par.src_live[child.par_idx[i]] = 1; }
    }

    // children move to their parent's rank; aux is then regularized in place
    par.src.setVal(0.0);
    child.aux.ParallelCopy(child.src, 0, 0, 1, amrex::IntVect(0), amrex::IntVect(0));

    (regularizePass<dim>(lev), ...);
}

template <int dim>
void LGFFastMultipole::interpolatePass(int lev)
{
    fmm_detail::Level& child = levels[lev];
    fmm_detail::Level& par   = levels[lev + 1];

    // the first direction reads the parent, the others work in place. With
    // the parent on the child's rank the result goes straight into tar
    const bool is_first = (dim == 0);
    amrex::MultiFab& dst_mf = child.is_par_local ? child.tar : child.aux;

    amrex::Vector<fmm_detail::LineTag> tags;
    for (amrex::MFIter mfi(child.aux); mfi.isValid(); ++mfi)
    {
        const int i_child = mfi.index();
        const int i_par   = child.par_idx[i_child];
        const amrex::IntVect lo_child = lowEnd(child.node_ba, i_child);
        const amrex::IntVect lo_par   = lowEnd(par.node_ba, i_par);
        const int parity = lowEnd(child.blk_ba, i_child)[dim] - 2 * lowEnd(par.blk_ba, i_par)[dim];

        auto const& dst = dst_mf.array(i_child);
        if (is_first)
        {
            tags.push_back({dst, par.tar.const_array(i_par), crossBox(lo_child, dim, n_node, n_node),
                            lo_par - lo_child, parity, n_node, n_node});
        }
        else
        {
            tags.push_back({dst, dst, crossBox(lo_child, dim, n_node, n_node),
                            amrex::IntVect(0), parity, n_node, n_node});
        }
    }

    if (tags.empty()) { return; }

    // plain data for passing into GPU lambda
    amrex::Real const* w_val = interp_val.data();
    int const* w_beg = interp_beg.data();
    const int nnd = n_node;
    const int nip = n_interp;

    amrex::ParallelFor(tags,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, fmm_detail::LineTag const& t) noexcept
        {
            fmmInterpLine<dim>(i, j, k, t, w_val, w_beg, nnd, nip);
        });
}

template <int... dim>
void LGFFastMultipole::interpolateLevel(int lev, std::integer_sequence<int, dim...>)
{
    BL_PROFILE("<Compute> LGFFastMultipole::interpolateLevel()");

    fmm_detail::Level& child = levels[lev];

    // every node of every child is overwritten, so tar needs no zeroing
    (interpolatePass<dim>(lev), ...);

    if (!child.is_par_local)
    {
        child.tar.ParallelCopy(child.aux, 0, 0, 1, amrex::IntVect(0), amrex::IntVect(0));
    }
}

void LGFFastMultipole::transformSource(amrex::Array4<amrex::Real const> const& src, const amrex::IntVect& lo,
                                       int n_src, int slot)
{
    // plain data for passing into GPU lambda
    const amrex::Dim3 lo3 = lo.dim3();

    auto const& pad = amrex::makeArray4(pad_work.data(), pad_box, 1);
    amrex::ParallelFor(pad_box,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            pad(i, j, k) = fmmPadSource(i, j, k, src, lo3, n_src);
        });

    fft->forward(pad_work.data(), spec_work.data());

    fmm_detail::Complex const* work = spec_work.data();
    fmm_detail::Complex* store = spec_src.data() + slot * n_spec;
    amrex::ParallelFor(n_spec,
        [=] AMREX_GPU_DEVICE (amrex::Long m) noexcept
        {
            store[m] = work[m];
        });
}

void LGFFastMultipole::interactLevel(int lev)
{
    BL_PROFILE("<Compute> LGFFastMultipole::interactLevel()");

    fmm_detail::Level& level = levels[lev];

    // sources owned by other ranks; every rank takes part, with or without
    // targets. isDefined() rather than ok(): ok() is a collective MPI call,
    // and not every rank gets past the early return below
    if (level.halo.isDefined())
    {
        level.halo.ParallelCopy(level.src, 0, 0, 1, amrex::IntVect(0), amrex::IntVect(0));
    }

    if (level.pair_slot.empty()) { return; }

    // one forward transform per source interval in reach of a local target
    spec_src.resize(level.n_slot * n_spec);
    spec_sum.resize(n_batch * n_spec);

    amrex::Vector<int> h_live(level.n_slot, 0);
    for (int i = 0; i < level.blk_ba.size(); ++i)
    {
        if (level.slot_idx[i] >= 0) { h_live[level.slot_idx[i]] = level.src_live[i]; }
    }
    level.slot_live.resize(level.n_slot);
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_live.begin(), h_live.end(), level.slot_live.begin());

    for (amrex::MFIter mfi(level.src); mfi.isValid(); ++mfi)
    {
        const int slot = level.slot_idx[mfi.index()];
        if (slot < 0 || !h_live[slot]) { continue; }

        transformSource(level.src.const_array(mfi), lowEnd(level.node_ba, mfi.index()), level.n_data, slot);
    }

    if (level.halo.isDefined())
    {
        for (amrex::MFIter mfi(level.halo); mfi.isValid(); ++mfi)
        {
            const int slot = level.n_slot_own + mfi.LocalIndex();
            if (!h_live[slot]) { continue; }

            transformSource(level.halo.const_array(mfi), lowEnd(level.halo.boxArray(), mfi.index()), level.n_data, slot);
        }
    }

    // plain data for passing into GPU lambda
    const amrex::Real h2 = geom.CellSize(0) * geom.CellSize(0); // IMPORTANT: Assumes dx[0] = dx[1] = dx[2]
    const amrex::Real scale = h2 * fft->scalingFactor();
    const amrex::Long nsp = n_spec;
    const int nft = n_fft;
    const int nhf = n_half;

    fmm_detail::Complex const* src_spec = spec_src.data();
    fmm_detail::Complex const* ker = ker_spec[lev].data();
    int const* pair_slot = level.pair_slot.data();
    int const* pair_code = level.pair_code.data();
    int const* slot_live = level.slot_live.data();

    fmm_detail::Complex* work = spec_work.data();
    auto const& pad = amrex::makeArray4<amrex::Real const>(pad_work.data(), pad_box, 1);

    // local targets with at least one source in reach
    amrex::Vector<int> tar_idx;
    amrex::Vector<int> tar_loc;
    for (amrex::MFIter mfi(level.tar); mfi.isValid(); ++mfi)
    {
        if (level.pair_ptr[mfi.LocalIndex()] == level.pair_ptr[mfi.LocalIndex() + 1]) { continue; }

        tar_idx.push_back(mfi.index());
        tar_loc.push_back(mfi.LocalIndex());
    }

    const int n_tar  = static_cast<int>(tar_idx.size());
    const int n_slab = spec_box.length(AMREX_SPACEDIM - 1);
    amrex::Vector<fmm_detail::SumTag> tags;

    for (int b_beg = 0; b_beg < n_tar; b_beg += n_batch)
    {
        const int n_b = amrex::min(n_batch, n_tar - b_beg);

        // all convolutions onto the targets of the batch, summed in Fourier
        // space in one fused launch. Tags are ordered slab by slab: neighbouring
        // targets share most of their sources, so on the host one slab of every
        // spectrum the batch touches stays in cache while it is being used
        tags.clear();
        for (int s = 0; s < n_slab; ++s)
        {
            amrex::Box cross = spec_box;
            cross.setBig(0, 0);
            cross.setSmall(AMREX_SPACEDIM - 1, s);
            cross.setBig(AMREX_SPACEDIM - 1, s);

            for (int b = 0; b < n_b; ++b)
            {
                tags.push_back({amrex::makeArray4(spec_sum.data() + b * n_spec, spec_box, 1), cross,
                                level.pair_ptr[tar_loc[b_beg + b]], level.pair_ptr[tar_loc[b_beg + b] + 1]});
            }
        }

        amrex::ParallelFor(tags,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, fmm_detail::SumTag const& t) noexcept
            {
                amrex::ignore_unused(i);
                fmmSumLine(j, k, t, src_spec, ker, pair_slot, pair_code, slot_live, nsp, nft, nhf);
            });

        // back to real space one target at a time, through the plan arrays
        for (int b = 0; b < n_b; ++b)
        {
            fmm_detail::Complex const* sum = spec_sum.data() + b * n_spec;
            amrex::ParallelFor(n_spec,
                [=] AMREX_GPU_DEVICE (amrex::Long m) noexcept
                {
                    work[m] = sum[m];
                });

            fft->backward(spec_work.data(), pad_work.data());

            // the first n_data nodes per direction; on level 0 that drops the overlap node
            const amrex::Box tar_box = level.node_ba[tar_idx[b_beg + b]];
            const amrex::Dim3 lo = tar_box.smallEnd().dim3();
            auto const& tar = level.tar.array(tar_idx[b_beg + b]);

            amrex::ParallelFor(tar_box,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    tar(i, j, k) += scale * pad(i - lo.x, j - lo.y, k - lo.z);
                });
        }
    }
}

void LGFFastMultipole::buildSolver()
{
    if (is_tree_built || xsoln_dm_ref.empty()) { return; }

    buildTree();
}

void LGFFastMultipole::solvePoisson(const amrex::MultiFab& source, amrex::MultiFab& target,
                                    const amrex::BoxArray& tag_ba)
{
    BL_PROFILE("<Compute> LGFFastMultipole::solvePoisson()");

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(source.ixType() == target.ixType(),
        "LGFFastMultipole: source and target must share an index type");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(canonicalBoxArray(source.boxArray()) == canon_ba
                                  && canonicalBoxArray(target.boxArray()) == canon_ba,
        "LGFFastMultipole: source/target not built from xsoln_ba_ref -- was regridOnto() called?");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(source.DistributionMap() == target.DistributionMap(),
        "LGFFastMultipole: source and target DistributionMaps differ");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(target.nGrowVect().allLE(amrex::IntVect(n_ghost_max)),
        "LGFFastMultipole: target has more ghosts than n_ghost_max");

    if (tag_ba.empty()) { target.setVal(0.0); return; }   // no sources => u == 0 everywhere

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(tag_ba.ixType().cellCentered() || tag_ba.ixType() == source.ixType(),
        "LGFFastMultipole: tag_ba must be cell-centred or share the source's index type");
    // the canonical boxes overlap on their hi+1 planes, so not assumed disjoint
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(canon_ba.contains(canonicalBoxArray(layoutTagBoxArray(tag_ba)), false),
        "LGFFastMultipole: tag_ba reaches outside xsoln_ba");

    // the tree follows the caller's DistributionMapping
    if (xsoln_dm_ref != source.DistributionMap())
    {
        xsoln_dm_ref  = source.DistributionMap();
        is_tree_built = false;
    }
    buildSolver();

    if (!is_plan_built || tag_ba != cached_tag_ba) { buildSourcePlan(tag_ba); }

    const int n_lev = numLevels();
    const auto dims = std::make_integer_sequence<int, AMREX_SPACEDIM>{};

    copySource(source, tag_ba);

    // upward pass
    for (int l = 0; l < n_lev - 1; ++l)
    {
        regularizeLevel(l, dims);
    }

    // downward pass, level interactions folded in: a level's field is its
    // parent's, interpolated, plus what its own convolutions add
    levels[n_lev - 1].tar.setVal(0.0);
    for (int l = n_lev - 1; l >= 0; --l)
    {
        if (l < n_lev - 1) { interpolateLevel(l, dims); }
        interactLevel(l);
    }

    // tar[0] covers every target index, ghosts included
    target.ParallelCopy(typedView(target.ixType()).tar, 0, 0, 1, amrex::IntVect(0), target.nGrowVect());
}

void LGFFastMultipole::regridOnto(const amrex::Geometry& new_geom, const amrex::BoxArray& new_ba, const amrex::DistributionMapping& new_dm)
{
    geom = new_geom;
    xsoln_ba_ref = new_ba;
    xsoln_dm_ref = new_dm;
    canon_ba = canonicalBoxArray(xsoln_ba_ref);

    // the tree is rebuilt lazily on the next solve; the kernel spectra are
    // kept, since they depend on neither the layout nor dx
    views.clear();
    levels.clear();
    is_tree_built = false;
    is_plan_built = false;
}
