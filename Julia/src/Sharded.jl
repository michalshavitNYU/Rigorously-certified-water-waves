"""
Sharded.jl — memory-sharded driver for the residual bound
(docs/sharding_plan_2026-07-03.md).

`residual_bound_sharded` reproduces `_residual_bound_full` **bit-exactly**
while holding only one grid shard (plus one full column during the FFT
phase) in RAM. The invariants that make this possible:

* Per-point derivative rows never read `sup_bounds` — they are computed
  shard-by-shard by the *same* row kernels (the existing `TaylorSeries`
  operators applied to shard-sized objects, with `u_j`/`h` taken from the
  global grid).
* All `sup_bounds` content is either a pure scalar chain or a grid
  reduction (`tighten_sup_bounds!` maxima, `sup_bound`/`inf_bound` scans,
  `dmin_bound`'s minimum). Reductions are accumulated across shards in
  ascending order and the scalar chains are replayed once, via the same
  extracted kernels (`_leibniz_sup_bounds`, `_tighten_from_max!`,
  `_supbound_scan`, `_infbound_scan`), in the same op order.
* Only the two `fft_hilbert_series` applications are global; they run
  sequentially, one derivative level (= one full column) at a time, with
  inputs/outputs stored exactly on disk (`ShardStore`).

The pipeline is split into three **stages** that communicate only through
the `ShardStore` directory (columns + exact Arf/Arb state dumps):

  stage 1 `_stage_p1!`   — shard streaming: rows, maxima, scans, columns.
  stage 2 `_stage_fft!`  — scalar replay (P2a), `inf_bound` scan over the
                            stored D3 columns (P1.5), remaining scalars
                            (P2b), and the per-level FFTs (S2).
  stage 3 `_stage_s3`    — shard-by-shard R assembly + final sup scan.

`residual_bound_sharded` composes them in-process. Each stage can also be
run as a **separate OS process** (`scripts/stage_run.jl`), with the
reduction state round-tripped exactly through disk — process exit then
reclaims all allocator memory perfectly and `Sys.maxrss` gives a clean
per-stage peak-RSS measurement.
"""

# --------------------------------------------------------------------------
# Memory reclamation at shard/column boundaries.
#
# Arb values are C-heap allocations (FLINT/GMP malloc) released by Julia
# finalizers, which only run during GC. The sharded driver's *live* set is
# small, but without forcing collection the deferred Arb frees accumulate
# between GCs and inflate peak RSS. At each boundary the previous shard's
# objects are dead, so a full GC + malloc_trim reclaims them and returns the
# pages to the OS.
# --------------------------------------------------------------------------

function _reclaim!()
    GC.gc(true)               # full collection: runs Arb finalizers (arb_clear)
    @static if Sys.islinux()
        try
            ccall(:malloc_trim, Cint, (Csize_t,), 0)
        catch
        end
    end
    return nothing
end

# --------------------------------------------------------------------------
# Small helpers: reduction accumulators and scalar-op replicas
# --------------------------------------------------------------------------

"""Per-order running maxima accumulator for `tighten_sup_bounds!`."""
mutable struct _MaxAcc
    ubs::Vector{Arblib.Arf}
    init::Vector{Bool}
end
_MaxAcc(M::Int) = _MaxAcc([Arblib.Arf() for _ in 0:M], fill(false, M + 1))

_acc!(a::_MaxAcc, derivs::AbstractMatrix{Arb}, M::Int) =
    _grid_abs_max_ubs!(a.ubs, a.init, derivs, 1:size(derivs, 1), M)

# Scalar sup_bounds replicas of the TaylorSeries operator scalar parts.
# These mirror `Base.:+`, `Base.:*(c, ts)` and `Base.:-` exactly:
#   +  : entrywise Arb addition;
#   c* : abs(c) computed once, then entrywise multiplication;
#   -x : Arb(-1; prec) * x   (so a - b = a + (Arb(-1)*b)).
_sb_add(a::Vector{Arb}, b::Vector{Arb}) = [a[i] + b[i] for i in eachindex(a)]
function _sb_smul(c::Arb, sb::Vector{Arb})
    abs_c = abs(c)
    return [abs_c * b for b in sb]
end
_sb_neg(sb::Vector{Arb}, prec::Int) = _sb_smul(Arb(-1; prec=prec), sb)

# Row-matrix replicas of the operator row parts (same op sequence as the
# corresponding TaylorSeries operators; used in stage 3 where we assemble R
# rows from stored columns without full TaylorSeries objects).
function _rows_smul(c::Arb, mat::Matrix{Arb})
    out = Matrix{Arb}(undef, size(mat)...)
    for j in 1:size(mat, 1), m in 1:size(mat, 2)
        out[j, m] = c * mat[j, m]
    end
    return out
end
function _rows_add(a::Matrix{Arb}, b::Matrix{Arb})
    out = Matrix{Arb}(undef, size(a)...)
    for j in 1:size(a, 1), m in 1:size(a, 2)
        out[j, m] = a[j, m] + b[j, m]
    end
    return out
end
_rows_neg(mat::Matrix{Arb}, prec::Int) = _rows_smul(Arb(-1; prec=prec), mat)
_rows_sub(a::Matrix{Arb}, b::Matrix{Arb}, prec::Int) =
    _rows_add(a, _rows_neg(b, prec))

"""Closed-form sup_bounds vector of a TrigPoly at Taylor order `M_order` —
exactly the vector `trigpoly_to_taylor` computes (same chain, same ops)."""
function _trigpoly_sup_bounds(f::TrigPoly, M_order::Int)::Vector{Arb}
    f_d = Vector{TrigPoly}(undef, M_order + 2)
    f_d[1] = f
    for m in 1:(M_order + 1)
        f_d[m+1] = deriv(f_d[m])
    end
    return [supnorm_bound(f_d[m+1]) for m in 0:(M_order + 1)]
end

# --------------------------------------------------------------------------
# Setup: everything derivable from the coefficients alone (cheap, scalar).
# Recomputed identically by every stage.
# --------------------------------------------------------------------------

function _sharded_setup(y_coeffs::Vector{<:Real}, μ::Real, β::Real;
                        M_order::Int, N_grid::Int, prec::Int,
                        extra_orders::Int, shard_size::Int)
    p = prec
    shards = [a:min(a + shard_size - 1, N_grid) for a in 1:shard_size:N_grid]
    has_capillary = (β != 0.0)
    has_capillary && @assert M_order >= 1 "M_order must be ≥ 1 for β≠0"

    y   = cosine_series(y_coeffs; prec=p)
    yu  = deriv(y)
    Hyu = hilbert(yu)

    M_h = M_order + 2 + extra_orders
    M1  = M_order + 1
    M3  = M_order + 3 + extra_orders

    pi_arb = Arb(pi; prec=p)
    h_arb  = Arb(2; prec=p) * pi_arb / Arb(N_grid; prec=p)

    # TrigPoly derivative chains (shared by dmin_bound and the B3 block —
    # both monolithic constructions produce identical TrigPolys).
    yu_d  = Vector{TrigPoly}(undef, M3 + 2)
    Hyu_d = Vector{TrigPoly}(undef, M3 + 2)
    yu_d[1]  = yu
    Hyu_d[1] = Hyu
    for m in 1:(M3 + 1)
        yu_d[m+1]  = deriv(yu_d[m])
        Hyu_d[m+1] = deriv(Hyu_d[m])
    end

    # Global closed-form sup_bounds vectors (grid-independent scalars).
    sup_bounds_A = [supnorm_bound(yu_d[m+1]) for m in 0:(M3 + 1)]
    sup_bounds_B = Vector{Arb}(undef, M3 + 2)
    sup_bounds_B[1] = Arb(1; prec=p) + supnorm_bound(Hyu_d[1])
    for m in 1:(M3 + 1)
        sup_bounds_B[m+1] = supnorm_bound(Hyu_d[m+1])
    end

    # Binomial tables exactly as `Base.:*` builds them per product order.
    binom_M  = [binomial(m, k) for m in 0:(M_order+1), k in 0:(M_order+1)]
    binom_Mh = [binomial(m, k) for m in 0:(M_h+1),     k in 0:(M_h+1)]
    binom_M3 = [binomial(m, k) for m in 0:(M3+1),      k in 0:(M3+1)]
    binom_M1 = [binomial(m, k) for m in 0:(M1+1),      k in 0:(M1+1)]

    # Pre-reduction scalar chain needed DURING stage 1: the sup_bound(ts_D2)
    # scan requires ts_D2.sup_bounds top, which is the (untightened) Leibniz
    # top — tighten never touches the top entry, so this is shard-independent.
    sb_AA_base = _leibniz_sup_bounds(sup_bounds_A, sup_bounds_A, M3, p, binom_M3)
    sb_BB_base = _leibniz_sup_bounds(sup_bounds_B, sup_bounds_B, M3, p, binom_M3)
    sb_D2_top  = sb_AA_base[M3 + 2] + sb_BB_base[M3 + 2]

    return (p=p, shards=shards, has_capillary=has_capillary,
            N_grid=N_grid, M_order=M_order, M_h=M_h, M1=M1, M3=M3,
            extra_orders=extra_orders,
            μ=μ, β=β, y=y, yu=yu, Hyu=Hyu,
            pi_arb=pi_arb, h_arb=h_arb, yu_d=yu_d, Hyu_d=Hyu_d,
            sup_bounds_A=sup_bounds_A, sup_bounds_B=sup_bounds_B,
            binom_M=binom_M, binom_Mh=binom_Mh, binom_M3=binom_M3,
            binom_M1=binom_M1,
            sb_AA_base=sb_AA_base, sb_BB_base=sb_BB_base,
            sb_D2_top=sb_D2_top,
            dummy=Arb(1; prec=p))
end

# --------------------------------------------------------------------------
# Stage 1: shard streaming — rows, maxima, scans, column slices.
# --------------------------------------------------------------------------

"""
    _stage_p1!(store, su; shard_range=1:length(su.shards))

Stage 1 over the (contiguous, ascending) global shard indices in
`shard_range`. With the default range this is the full sequential stage 1.
With a sub-range it computes that range's column slices plus a *partial*
reduction state; partial states from a contiguous ascending cover of
`1:length(su.shards)` merge into the exact sequential state via
`_merge_p1_states` (all reductions are first-occurrence max/min folds, so
ordered merging reproduces the sequential result bit-exactly).
"""
function _stage_p1!(store::ShardStore, su;
                    shard_range::UnitRange{Int}=1:length(su.shards))
    p = su.p
    M_order = su.M_order

    acc_yHyu  = _MaxAcc(M_order)
    acc_yyuh  = _MaxAcc(su.M_h)
    acc_AA    = _MaxAcc(su.M3)
    acc_BB    = _MaxAcc(su.M3)
    acc_D3    = _MaxAcc(su.M3)
    acc_invD3 = _MaxAcc(su.M3)
    acc_G3    = _MaxAcc(su.M3)
    acc_yuD   = _MaxAcc(su.M1)

    # sup_bound(ts_D2) scan state (weights identical to the monolithic call).
    wD2, remD2 = _taylor_step_weights(su.M3, su.h_arb, p)
    fboundD2 = Arblib.ubound(su.sb_D2_top * remD2)
    supD2_best, supD2_ub = Arb(0; prec=p), Arblib.Arf()

    # dmin_bound grid-minimum state.
    D_min_arf = Arblib.Arf()
    D_min_init = false

    for si in shard_range
        jr = su.shards[si]
        P = length(jr)

        # --- β=0 chain rows (Steps 1–3 of the monolithic pipeline) ---
        ts_y_h  = trigpoly_to_taylor(su.y,  su.N_grid, su.M_h; prec=p, j_range=jr)
        ts_yu_h = trigpoly_to_taylor(su.yu, su.N_grid, su.M_h; prec=p, j_range=jr)
        ts_Hyu  = trigpoly_to_taylor(su.Hyu, su.N_grid, M_order; prec=p, j_range=jr)

        ts_y  = truncate_order(ts_y_h,  M_order)
        ts_yu = truncate_order(ts_yu_h, M_order)

        ts_y_Hyu  = ts_y * ts_Hyu
        ts_y_yu_h = ts_y_h * ts_yu_h
        ts_y_yu   = truncate_order(ts_y_yu_h, M_order)

        _acc!(acc_yHyu, ts_y_Hyu.derivs, M_order)
        _acc!(acc_yyuh, ts_y_yu_h.derivs, su.M_h)

        write_matrix_slices(store, "y",    si, ts_y.derivs)
        write_matrix_slices(store, "Hyu",  si, ts_Hyu.derivs)
        write_matrix_slices(store, "yHyu", si, ts_y_Hyu.derivs)
        write_matrix_slices(store, "yyu",  si, ts_y_yu.derivs)

        if su.has_capillary
            # --- dmin_bound rows (Step 6a) ---
            derivs_A = Matrix{Arb}(undef, P, su.M3 + 1)
            derivs_B = Matrix{Arb}(undef, P, su.M3 + 1)
            Dj_lbs = Vector{Arblib.Arf}(undef, P)
            j0 = first(jr) - 1
            Threads.@threads for r in 1:P
                j = j0 + r
                u_j = Arb(2; prec=p) * su.pi_arb * Arb(j - 1; prec=p) / Arb(su.N_grid; prec=p)
                for m in 0:su.M3
                    derivs_A[r, m+1] = tp_eval(su.yu_d[m+1], u_j)
                    if m == 0
                        derivs_B[r, 1] = Arb(1; prec=p) + tp_eval(su.Hyu_d[1], u_j)
                    else
                        derivs_B[r, m+1] = tp_eval(su.Hyu_d[m+1], u_j)
                    end
                end
                Dj_lbs[r] = Arblib.lbound(sqrt(derivs_A[r,1]^2 + derivs_B[r,1]^2))
            end
            for r in 1:P
                if !D_min_init || Dj_lbs[r] < D_min_arf
                    D_min_arf = Dj_lbs[r]
                    D_min_init = true
                end
            end

            ts_A = TaylorSeries(derivs_A, su.sup_bounds_A, su.M3, P, su.h_arb, p)
            ts_B = TaylorSeries(derivs_B, su.sup_bounds_B, su.M3, P, su.h_arb, p)

            ts_AA = ts_A * ts_A
            ts_BB = ts_B * ts_B
            _acc!(acc_AA, ts_AA.derivs, su.M3)
            _acc!(acc_BB, ts_BB.derivs, su.M3)
            ts_D2 = ts_AA + ts_BB

            supD2_best, supD2_ub = _supbound_scan(ts_D2.derivs, 1:P, su.M3, p,
                                                  wD2, fboundD2,
                                                  supD2_best, supD2_ub)

            # sqrt/inv rows never read the scalar arguments (those only feed
            # the sup_bounds chains, replayed globally in stage 2) — dummies.
            ts_D3 = sqrt_series(ts_D2, su.dummy; S_upper=su.dummy)
            _acc!(acc_D3, ts_D3.derivs, su.M3)
            write_matrix_slices(store, "D3", si, ts_D3.derivs)

            ts_invD3 = inv_series(ts_D3, su.dummy)
            _acc!(acc_invD3, ts_invD3.derivs, su.M3)

            # --- B3 (Step 6b) rows coincide with ts_B's rows (identical
            # tp_eval calls in the monolithic code) ---
            ts_B3 = TaylorSeries(derivs_B, su.sup_bounds_B, su.M3, P, su.h_arb, p)

            # --- G3, yuD, C1, Gu rows (Steps 6c–6d) ---
            ts_G3 = ts_B3 * ts_invD3
            _acc!(acc_G3, ts_G3.derivs, su.M3)

            ts_yu_M3 = trigpoly_to_taylor(su.yu, su.N_grid, su.M3; prec=p, j_range=jr)
            ts_yu_M1   = truncate_order(ts_yu_M3, su.M1)
            ts_invD_M1 = truncate_order(ts_invD3, su.M1)
            ts_yuD_M1  = ts_yu_M1 * ts_invD_M1
            _acc!(acc_yuD, ts_yuD_M1.derivs, su.M1)
            ts_C1 = deriv_series(ts_yuD_M1)

            ts_G_M1 = truncate_order(ts_G3, su.M1)
            ts_Gu   = deriv_series(ts_G_M1)

            write_matrix_slices(store, "C1", si, ts_C1.derivs)
            write_matrix_slices(store, "Gu", si, ts_Gu.derivs)
        end

        # Shard boundary: previous shard's TaylorSeries objects are now dead.
        _reclaim!()
    end

    return (acc_yHyu=acc_yHyu, acc_yyuh=acc_yyuh, acc_AA=acc_AA,
            acc_BB=acc_BB, acc_D3=acc_D3, acc_invD3=acc_invD3,
            acc_G3=acc_G3, acc_yuD=acc_yuD,
            supD2_best=supD2_best, supD2_ub=supD2_ub,
            D_min_arf=D_min_arf, D_min_init=D_min_init)
end

# --------------------------------------------------------------------------
# Stage 2: scalar replay (P2a), inf_bound over stored D3 (P1.5), remaining
# scalars (P2b), per-level FFTs (S2).
# --------------------------------------------------------------------------

function _stage_fft!(store::ShardStore, su, st)
    p = su.p
    M_order = su.M_order

    # Closed-form sup vectors (identical to the shard objects' vectors).
    sb_y_h  = _trigpoly_sup_bounds(su.y,  su.M_h)
    sb_yu_h = _trigpoly_sup_bounds(su.yu, su.M_h)
    sb_Hyu  = _trigpoly_sup_bounds(su.Hyu, M_order)

    sb_y  = sb_y_h[1:(M_order + 2)]
    sb_yu = sb_yu_h[1:(M_order + 2)]

    sb_y_Hyu = _leibniz_sup_bounds(sb_y, sb_Hyu, M_order, p, su.binom_M)
    _tighten_from_max!(sb_y_Hyu, st.acc_yHyu.ubs, M_order, su.h_arb, p)

    sb_y_yu_h = _leibniz_sup_bounds(sb_y_h, sb_yu_h, su.M_h, p, su.binom_Mh)
    _tighten_from_max!(sb_y_yu_h, st.acc_yyuh.ubs, su.M_h, su.h_arb, p)

    Gpp_bound_yyu  = sb_y_yu_h[M_order + 3]
    Gppp_bound_yyu = sb_y_yu_h[M_order + 4]
    sb_y_yu = sb_y_yu_h[1:(M_order + 2)]

    # fft_hilbert_series output sup_bounds for H[y·y_u] (pure scalars).
    factor_pi2 = su.pi_arb * su.pi_arb / Arb(3; prec=p)
    sb_Hyyu = Vector{Arb}(undef, M_order + 2)
    for m in 0:(M_order - 1)
        sb_Hyyu[m + 1] = sb_y_yu[m + 3] * factor_pi2
    end
    sb_Hyyu[M_order + 1] = Gpp_bound_yyu  * factor_pi2
    sb_Hyyu[M_order + 2] = Gppp_bound_yyu * factor_pi2

    # R sup_bounds, stage 1: μ·Hyu − y − y·Hyu − Hyyu (operator scalar parts).
    μ_arb = Arb(su.μ; prec=p)
    sb_R = _sb_add(_sb_smul(μ_arb, sb_Hyu), _sb_neg(sb_y, p))
    sb_R = _sb_add(sb_R, _sb_neg(sb_y_Hyu, p))
    sb_R = _sb_add(sb_R, _sb_neg(sb_Hyyu, p))

    D_lower = nothing
    sb_Gu = Vector{Arb}()
    Gpp_bound_Gu = su.dummy
    Gppp_bound_Gu = su.dummy

    if su.has_capillary
        sb_AA = copy(su.sb_AA_base)
        sb_BB = copy(su.sb_BB_base)
        _tighten_from_max!(sb_AA, st.acc_AA.ubs, su.M3, su.h_arb, p)
        _tighten_from_max!(sb_BB, st.acc_BB.ubs, su.M3, su.h_arb, p)
        sb_D2 = _sb_add(sb_AA, sb_BB)

        S_upper = st.supD2_best

        lip_correction_arb = supnorm_bound(su.yu_d[2]) + supnorm_bound(su.Hyu_d[2])
        d_lower_arb = Arb(st.D_min_arf; prec=p)
        Arblib.add_error!(d_lower_arb, Arblib.ubound(lip_correction_arb * su.h_arb))
        d_lower_init = Arb(Arblib.lbound(d_lower_arb); prec=p)
        if !Arblib.is_positive(d_lower_init)
            error("dmin_bound: D is not certified strictly positive (d_lower_init ≤ 0).")
        end

        d2_lower = d_lower_init * d_lower_init
        g_sup_sqrt = _sqrt_sup_bounds_on_interval(d2_lower, S_upper, su.M3 + 1, p)
        sb_D3 = _faa_di_bruno_derivs(g_sup_sqrt, sb_D2, su.M3 + 1, p)
        _tighten_from_max!(sb_D3, st.acc_D3.ubs, su.M3, su.h_arb, p)

        # ---- P1.5: inf_bound(ts_D3) scan over the stored D3 columns. ----
        wD3, remD3 = _taylor_step_weights(su.M3, su.h_arb, p)
        fboundD3 = Arblib.ubound(sb_D3[su.M3 + 2] * remD3)
        first_slice = read_matrix_slices(store, "D3", 1,
                                         length(su.shards[1]), su.M3 + 1)
        ib_best = first_slice[1, 1] - Arb(0; prec=p)
        ib_lb = Arblib.Arf()
        ib_init = false
        for (si, jr) in enumerate(su.shards)
            mat = si == 1 ? first_slice :
                  read_matrix_slices(store, "D3", si, length(jr), su.M3 + 1)
            ib_best, ib_lb, ib_init = _infbound_scan(mat, 1:length(jr), su.M3,
                                                     p, wD3, fboundD3,
                                                     ib_best, ib_lb, ib_init)
            si != 1 && _reclaim!()
        end
        ib = ib_best
        D_lower = Arblib.lbound(ib) > Arblib.lbound(d_lower_init) ? ib : d_lower_init

        # ---- P2b: remaining capillary scalars. ----
        g_sup_inv = _inv_sup_bounds_on_interval(D_lower, su.M3 + 1, p)
        sb_invD3 = _faa_di_bruno_derivs(g_sup_inv, sb_D3, su.M3 + 1, p)
        _tighten_from_max!(sb_invD3, st.acc_invD3.ubs, su.M3, su.h_arb, p)

        sb_G3 = _leibniz_sup_bounds(su.sup_bounds_B, sb_invD3, su.M3, p,
                                    su.binom_M3)
        _tighten_from_max!(sb_G3, st.acc_G3.ubs, su.M3, su.h_arb, p)

        _, G_sup_tight = tight_DG_sup_bounds(su.y, M_order + 4, D_lower; prec=p)
        Gpp_bound_chain  = sb_G3[M_order + 4]
        Gppp_bound_chain = sb_G3[M_order + 5]
        Gpp_bound_Gu  = Arblib.ubound(G_sup_tight[M_order + 4]) < Arblib.ubound(Gpp_bound_chain)  ? G_sup_tight[M_order + 4] : Gpp_bound_chain
        Gppp_bound_Gu = Arblib.ubound(G_sup_tight[M_order + 5]) < Arblib.ubound(Gppp_bound_chain) ? G_sup_tight[M_order + 5] : Gppp_bound_chain

        sb_yu_M3 = _trigpoly_sup_bounds(su.yu, su.M3)
        sb_yu_M1   = sb_yu_M3[1:(su.M1 + 2)]
        sb_invD_M1 = sb_invD3[1:(su.M1 + 2)]
        sb_yuD = _leibniz_sup_bounds(sb_yu_M1, sb_invD_M1, su.M1, p, su.binom_M1)
        _tighten_from_max!(sb_yuD, st.acc_yuD.ubs, su.M1, su.h_arb, p)
        sb_C1 = sb_yuD[2:end]

        sb_G_M1 = sb_G3[1:(su.M1 + 2)]
        sb_Gu   = sb_G_M1[2:end]

        sb_HGu = Vector{Arb}(undef, M_order + 2)
        for m in 0:(M_order - 1)
            sb_HGu[m + 1] = sb_Gu[m + 3] * factor_pi2
        end
        sb_HGu[M_order + 1] = Gpp_bound_Gu  * factor_pi2
        sb_HGu[M_order + 2] = Gppp_bound_Gu * factor_pi2

        # R sup_bounds, capillary stage: + β·C1 − β·HGu.
        β_arb = Arb(su.β; prec=p)
        sb_R = _sb_add(sb_R, _sb_smul(β_arb, sb_C1))
        sb_R = _sb_add(sb_R, _sb_neg(_sb_smul(β_arb, sb_HGu), p))
    end

    # ---- S2: sequential per-level certified FFT-Hilbert transforms. ----
    function _transform_column!(in_name::String, out_name::String,
                                F_top::Arb, Gpp::Arb)
        for m in 0:M_order
            col = Vector{Arb}(undef, su.N_grid)
            for (si, jr) in enumerate(su.shards)
                sl = read_slice(store, in_name, m, si, length(jr))
                for (i, j) in enumerate(jr)
                    col[j] = sl[i]
                end
            end
            p_m, B_m = _fft_level_params(M_order, m, F_top, Gpp)
            HG_col, _ = fft_hilbert(col, su.N_grid, p_m, B_m; prec=p)
            for (si, jr) in enumerate(su.shards)
                write_slice(store, out_name, m, si, view(HG_col, jr))
            end
            # Column boundary: the length-N input/output columns and the
            # 2N-1 AcbPoly convolution buffers are now dead.
            _reclaim!()
        end
    end

    _transform_column!("yyu", "Hyyu", sb_y_yu[M_order + 2], Gpp_bound_yyu)
    if su.has_capillary
        _transform_column!("Gu", "HGu", sb_Gu[M_order + 2], Gpp_bound_Gu)
    end

    return (sb_R=sb_R, D_lower=D_lower)
end

# --------------------------------------------------------------------------
# Stage 3: shard-by-shard residual assembly + final sup_bound scan.
# --------------------------------------------------------------------------

function _stage_s3(store::ShardStore, su, sb_R::Vector{Arb})
    p = su.p
    M_order = su.M_order

    wR, remR = _taylor_step_weights(M_order, su.h_arb, p)
    fboundR = Arblib.ubound(sb_R[M_order + 2] * remR)
    supR_best, supR_ub = Arb(0; prec=p), Arblib.Arf()

    ncols = M_order + 1
    μ_arb = Arb(su.μ; prec=p)
    β_arb = Arb(su.β; prec=p)
    for (si, jr) in enumerate(su.shards)
        P = length(jr)
        rows_Hyu  = read_matrix_slices(store, "Hyu",  si, P, ncols)
        rows_y    = read_matrix_slices(store, "y",    si, P, ncols)
        rows_yHyu = read_matrix_slices(store, "yHyu", si, P, ncols)
        rows_Hyyu = read_matrix_slices(store, "Hyyu", si, P, ncols)

        # ts_R = μ·Hyu − y − y·Hyu − Hyyu  (same operator op sequence).
        rows_R = _rows_smul(μ_arb, rows_Hyu)
        rows_R = _rows_sub(rows_R, rows_y, p)
        rows_R = _rows_sub(rows_R, rows_yHyu, p)
        rows_R = _rows_sub(rows_R, rows_Hyyu, p)

        if su.has_capillary
            rows_C1  = read_matrix_slices(store, "C1",  si, P, ncols)
            rows_HGu = read_matrix_slices(store, "HGu", si, P, ncols)
            # ts_R = ts_R + β·C1 − β·HGu.
            rows_R = _rows_add(rows_R, _rows_smul(β_arb, rows_C1))
            rows_R = _rows_sub(rows_R, _rows_smul(β_arb, rows_HGu), p)
        end

        write_matrix_slices(store, "R", si, rows_R)

        supR_best, supR_ub = _supbound_scan(rows_R, 1:P, M_order, p,
                                            wR, fboundR, supR_best, supR_ub)

        # Shard boundary: this shard's loaded columns and R rows are dead.
        _reclaim!()
    end

    return supR_best
end

# --------------------------------------------------------------------------
# Exact state (de)serialization for the process-per-stage mode.
# --------------------------------------------------------------------------

_dump_arf(x::Arblib.Arf) = Arblib.dump_string(x)
_load_arf(s::AbstractString) = Arblib.load_string!(Arblib.Arf(), String(s))

function _save_acc(io::IO, name::String, a::_MaxAcc)
    println(io, "acc ", name, " ", length(a.ubs))
    for i in eachindex(a.ubs)
        println(io, _dump_arf(a.ubs[i]), " | ", a.init[i] ? 1 : 0)
    end
end

function _load_acc(lines::Vector{String}, idx::Int)
    hdr = split(lines[idx])
    @assert hdr[1] == "acc"
    n = parse(Int, hdr[3])
    a = _MaxAcc(n - 1)
    for i in 1:n
        parts = split(lines[idx + i], " | ")
        a.ubs[i] = _load_arf(strip(parts[1]))
        a.init[i] = strip(parts[2]) == "1"
    end
    return String(hdr[2]), a, idx + n + 1
end

"""Save the stage-1 reduction state exactly (arf/arb dumps)."""
function _save_p1_state(path::String, st)
    open(path, "w") do io
        println(io, "supD2_best ", Arblib.dump_string(st.supD2_best))
        println(io, "supD2_ub ", _dump_arf(st.supD2_ub))
        println(io, "D_min ", _dump_arf(st.D_min_arf), " | ",
                st.D_min_init ? 1 : 0)
        for (name, a) in (("yHyu", st.acc_yHyu), ("yyuh", st.acc_yyuh),
                          ("AA", st.acc_AA), ("BB", st.acc_BB),
                          ("D3", st.acc_D3), ("invD3", st.acc_invD3),
                          ("G3", st.acc_G3), ("yuD", st.acc_yuD))
            _save_acc(io, name, a)
        end
    end
end

function _load_p1_state(path::String, prec::Int)
    lines = readlines(path)
    supD2_best = Arblib.load_string!(Arb(; prec=prec),
                                     String(split(lines[1], " "; limit=2)[2]))
    supD2_ub = _load_arf(split(lines[2], " "; limit=2)[2])
    dparts = split(split(lines[3], " "; limit=2)[2], " | ")
    D_min_arf = _load_arf(strip(dparts[1]))
    D_min_init = strip(dparts[2]) == "1"
    accs = Dict{String,_MaxAcc}()
    idx = 4
    while idx <= length(lines)
        name, a, idx = _load_acc(lines, idx)
        accs[name] = a
    end
    return (acc_yHyu=accs["yHyu"], acc_yyuh=accs["yyuh"], acc_AA=accs["AA"],
            acc_BB=accs["BB"], acc_D3=accs["D3"], acc_invD3=accs["invD3"],
            acc_G3=accs["G3"], acc_yuD=accs["yuD"],
            supD2_best=supD2_best, supD2_ub=supD2_ub,
            D_min_arf=D_min_arf, D_min_init=D_min_init)
end

"""Save the stage-2 outputs (R sup_bounds vector and D_lower) exactly."""
function _save_fft_state(path::String, res)
    open(path, "w") do io
        if res.D_lower === nothing
            println(io, "D_lower NONE")
        else
            println(io, "D_lower ", Arblib.dump_string(res.D_lower))
        end
        println(io, "sb_R ", length(res.sb_R))
        for b in res.sb_R
            println(io, Arblib.dump_string(b))
        end
    end
end

function _load_fft_state(path::String, prec::Int)
    lines = readlines(path)
    dl_payload = split(lines[1], " "; limit=2)[2]
    D_lower = dl_payload == "NONE" ? nothing :
              Arblib.load_string!(Arb(; prec=prec), String(dl_payload))
    n = parse(Int, split(lines[2])[2])
    sb_R = [Arblib.load_string!(Arb(; prec=prec), lines[2 + i]) for i in 1:n]
    return (sb_R=sb_R, D_lower=D_lower)
end

# --------------------------------------------------------------------------
# Parallel stage 1: worker ranges and ordered state merge.
# --------------------------------------------------------------------------

"""
    _worker_shard_range(S, i, W) -> UnitRange{Int}

Contiguous split of shard indices `1:S` into `W` chunks; chunk `i` (1-based).
The same function is used by the worker and the merger, so the cover is
consistent by construction.
"""
function _worker_shard_range(S::Int, i::Int, W::Int)
    @assert 1 <= i <= W
    base, rem = divrem(S, W)
    lo = (i - 1) * base + min(i - 1, rem) + 1
    hi = i * base + min(i, rem)
    return lo:hi
end

"""
    _merge_p1_states(parts) -> merged state

Fold partial stage-1 reduction states (from a contiguous ascending cover of
the shard range, in ascending order) into the exact sequential state.

Every reduction is a first-occurrence max/min fold with strict comparison,
so folding the partials with the same strict comparison in ascending order
selects exactly the value (and, for the sup scan, the ball) the sequential
scan would have selected:
  * `_MaxAcc` maxima: per-order max of exact Arf values — associative.
  * sup(D²) scan: sequential adoption on `ub > best_ub` picks the ball at
    the first global row attaining the maximal ubound; folding partial
    `(best, best_ub)` pairs with the same strict `>` picks the same ball.
  * dmin minimum: same argument with `<`.
"""
function _merge_p1_states(parts::Vector)
    @assert !isempty(parts)
    M(a::_MaxAcc) = length(a.ubs) - 1

    function merge_acc(accs::Vector{_MaxAcc})
        out = _MaxAcc(M(accs[1]))
        for a in accs
            for i in eachindex(out.ubs)
                if a.init[i] && (!out.init[i] || a.ubs[i] > out.ubs[i])
                    out.ubs[i] = a.ubs[i]
                    out.init[i] = true
                end
            end
        end
        return out
    end

    acc_yHyu  = merge_acc([st.acc_yHyu  for st in parts])
    acc_yyuh  = merge_acc([st.acc_yyuh  for st in parts])
    acc_AA    = merge_acc([st.acc_AA    for st in parts])
    acc_BB    = merge_acc([st.acc_BB    for st in parts])
    acc_D3    = merge_acc([st.acc_D3    for st in parts])
    acc_invD3 = merge_acc([st.acc_invD3 for st in parts])
    acc_G3    = merge_acc([st.acc_G3    for st in parts])
    acc_yuD   = merge_acc([st.acc_yuD   for st in parts])

    # sup(D²) scan: fold from the same initial state the sequential scan
    # uses, adopting on strict ubound improvement in ascending part order.
    prec = Arblib.precision(parts[1].supD2_best)
    supD2_best, supD2_ub = Arb(0; prec=prec), Arblib.Arf()
    for st in parts
        if st.supD2_ub > supD2_ub
            supD2_best = st.supD2_best
            supD2_ub = st.supD2_ub
        end
    end

    # dmin grid minimum: strict `<` fold.
    D_min_arf = Arblib.Arf()
    D_min_init = false
    for st in parts
        if st.D_min_init && (!D_min_init || st.D_min_arf < D_min_arf)
            D_min_arf = st.D_min_arf
            D_min_init = true
        end
    end

    return (acc_yHyu=acc_yHyu, acc_yyuh=acc_yyuh, acc_AA=acc_AA,
            acc_BB=acc_BB, acc_D3=acc_D3, acc_invD3=acc_invD3,
            acc_G3=acc_G3, acc_yuD=acc_yuD,
            supD2_best=supD2_best, supD2_ub=supD2_ub,
            D_min_arf=D_min_arf, D_min_init=D_min_init)
end

# --------------------------------------------------------------------------
# In-process driver (stage composition).
# --------------------------------------------------------------------------

"""
    residual_bound_sharded(y_coeffs, μ, β=0.0;
                           M_order, N_grid, prec=128, extra_orders=0,
                           shard_size=4096, workdir=mktempdir())
        -> NamedTuple (bound, D_lower, workdir)

Sharded, bit-exact equivalent of
`residual_bound(y_coeffs, μ, β; M_order, N_grid, prec, extra_orders)`:
returns the same `bound` (and for β ≠ 0 the same `D_lower`) as the
monolithic pipeline, with peak RAM ~O(shard_size + one grid column)
instead of ~O(20 full matrices).
"""
function residual_bound_sharded(y_coeffs::Vector{<:Real}, μ::Real, β::Real=0.0;
                                M_order::Int, N_grid::Int, prec::Int=128,
                                extra_orders::Int=0,
                                shard_size::Int=4096,
                                workdir::AbstractString=mktempdir())
    su = _sharded_setup(y_coeffs, μ, β;
                        M_order=M_order, N_grid=N_grid, prec=prec,
                        extra_orders=extra_orders, shard_size=shard_size)
    store = ShardStore(workdir, prec)
    st  = _stage_p1!(store, su)
    res = _stage_fft!(store, su, st)
    bound = _stage_s3(store, su, res.sb_R)
    return (bound = bound,
            D_lower = res.D_lower,
            workdir = workdir)
end

export residual_bound_sharded
