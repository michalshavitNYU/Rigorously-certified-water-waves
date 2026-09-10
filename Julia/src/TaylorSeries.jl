"""
TaylorSeries.jl — Certified Taylor-series layer for bounding smooth functions
between grid points using Arb ball arithmetic.

Given a smooth 2π-periodic function f, we represent it at each grid point u_j
by the vector of derivatives [f(u_j), f'(u_j), ..., f^{(M)}(u_j)] together with
a certified *vector* of global supnorm bounds
    sup_bounds[m+1] ≥ ‖f^{(m)}‖_∞    for m = 0..M_order+1
spanning every derivative level the local Taylor remainder formulas and
Leibniz/Faà-di-Bruno bounds need.

This is a port of the design used in cap-ccf/pkg/CCFBlowup.jl/src/taylor_series.jl.
The previous single-scalar `F_bound :: Arb` representation was structurally
insufficient for products, inverses, and roots: the formulas need bounds on
*all* intermediate derivatives, not just the top one.

Convention:
  derivs[j, m+1]    = f^{(m)}(u_j),   j = 1..M_grid, m = 0..M_order
  sup_bounds[m+1]   ≥ ‖f^{(m)}‖_∞,    m = 0..M_order+1   (length M_order+2)
  Grid points:  u_j = 2π·(j-1)/M_grid  for j = 1..M_grid
"""

using Arblib

# ---------------------------------------------------------------------------
# Struct
# ---------------------------------------------------------------------------

"""
    TaylorSeries

Certified Taylor series representation of a smooth 2π-periodic function f at
M_grid equispaced grid points.

Fields:
  derivs     :: Matrix{Arb}   # derivs[j, m+1] = f^{(m)}(u_j)
  sup_bounds :: Vector{Arb}   # sup_bounds[m+1] ≥ ‖f^{(m)}‖_∞ for m=0..M_order+1
  M_order    :: Int           # Taylor order (highest derivative stored)
  M_grid     :: Int           # number of grid points
  h          :: Arb           # grid spacing = 2π/M_grid
  prec       :: Int           # Arb working precision in bits
"""
struct TaylorSeries
    derivs     :: Matrix{Arb}
    sup_bounds :: Vector{Arb}
    M_order    :: Int
    M_grid     :: Int
    h          :: Arb
    prec       :: Int

    function TaylorSeries(derivs::Matrix{Arb}, sup_bounds::Vector{Arb},
                          M_order::Int, M_grid::Int, h::Arb, prec::Int)
        @assert size(derivs) == (M_grid, M_order + 1) "derivs shape must be (M_grid, M_order+1)"
        @assert length(sup_bounds) == M_order + 2 "sup_bounds must have length M_order+2"
        @assert M_order >= 0 "M_order must be ≥ 0"
        new(derivs, sup_bounds, M_order, M_grid, h, prec)
    end
end

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

"""Compute binomial coefficient C(n, k) as an Arb."""
function _binom_arb(n::Int, k::Int, prec::Int)::Arb
    return Arb(binomial(n, k); prec=prec)
end

"""Compute factorial m! as an Arb."""
function _fact_arb(m::Int, prec::Int)::Arb
    return Arb(factorial(big(m)); prec=prec)
end

"""
    _faa_di_bruno_derivs(g_d, C_d, order, prec) -> Vector{Arb}

Compute raw derivatives `[(f∘C)^(0), (f∘C)^(1), …, (f∘C)^(order)]` via
ArbSeries composition. Given
    g_d[k+1] = f^(k)(C(z))    (raw, not divided by k!)
    C_d[j+1] = C^(j)(z)       (raw)

returns the raw derivatives of f∘C at z. Requires `length(g_d) ≥ order+1` and
`length(C_d) ≥ order+1`.

When fed non-negative inputs (sup_bound vectors), the output bounds the
sup_bounds of `g ∘ C` because the Bell-polynomial expansion of Faà di Bruno
has non-negative integer coefficients.

Ported from cap-ccf/pkg/CCFBlowup.jl/src/taylor_series.jl.
"""
function _faa_di_bruno_derivs(g_d::Vector{Arb}, C_d::Vector{Arb},
                              order::Int, prec::Int)
    @assert length(g_d) >= order + 1 "g_d too short for requested order"
    @assert length(C_d) >= order + 1 "C_d too short for requested order"

    facts = [Arb(factorial(big(k)); prec=prec) for k in 0:order]

    # Δ(ε) = C(z+ε) − C(z) = Σ_{j=1}^{order} (C^(j)/j!)·ε^j     (Δ(0) = 0)
    Δ_coeffs = [Arb(0; prec=prec) for _ in 0:order]
    for j in 1:order
        Δ_coeffs[j + 1] = C_d[j + 1] / facts[j + 1]
    end
    Δ_ts = ArbSeries(Δ_coeffs; degree=order, prec=prec)

    # F(ε) = Σ_{k=0}^{order} (f^(k)(C(z))/k!)·Δ(ε)^k        truncated to degree order
    F_ts = ArbSeries([g_d[1]]; degree=order, prec=prec)
    Δ_pow = deepcopy(Δ_ts)                                  # Δ^1
    for k in 1:order
        scalar = g_d[k + 1] / facts[k + 1]
        term = scalar * Δ_pow
        F_ts = F_ts + term
        k < order && (Δ_pow = Δ_pow * Δ_ts)
    end

    return [F_ts[n] * facts[n + 1] for n in 0:order]
end

# ---------------------------------------------------------------------------
# sup_bound — certified upper bound on ‖f‖_∞
# ---------------------------------------------------------------------------

"""
    _taylor_step_weights(M, h, prec) -> (hpow_over_fact, rem_factor)

Shared precomputation for `sup_bound`/`inf_bound`: the per-order Taylor step
weights `h^m/m!` for m = 0..M and the remainder factor `h^{M+1}/(M+1)!`.
Extracted so the sharded driver reproduces the identical Arb op sequence.
"""
function _taylor_step_weights(M::Int, h::Arb, prec::Int)
    hpow_over_fact = Vector{Arb}(undef, M + 1)
    hpow = Arb(1; prec=prec)
    fact = Arb(1; prec=prec)
    for m in 0:M
        if m > 0
            hpow = hpow * h
            fact = fact * Arb(m; prec=prec)
        end
        hpow_over_fact[m+1] = hpow / fact
    end
    rem_factor = hpow * h / (fact * Arb(M + 1; prec=prec))
    return hpow_over_fact, rem_factor
end

"""
    _supbound_scan(derivs, jrows, M, prec, hpow_over_fact, fbound_radius_arf,
                   best, best_ub) -> (best, best_ub)

Row-scan kernel of `sup_bound` over the rows `jrows` of `derivs`, updating
the running `(best, best_ub)` state. Rows must be visited in ascending order
(within and across calls) for bit-identical tie-breaking with the monolithic
scan.
"""
function _supbound_scan(derivs::AbstractMatrix{Arb}, jrows, M::Int, prec::Int,
                        hpow_over_fact::Vector{Arb},
                        fbound_radius_arf::Arblib.Arf,
                        best::Arb, best_ub::Arblib.Arf)
    for j in jrows
        s = Arb(0; prec=prec)
        for m in 0:M
            s = s + abs(derivs[j, m+1]) * hpow_over_fact[m+1]
        end
        Arblib.add_error!(s, fbound_radius_arf)
        s_ub = Arblib.ubound(s)
        if s_ub > best_ub
            best    = s
            best_ub = s_ub
        end
    end
    return best, best_ub
end

"""
    sup_bound(ts::TaylorSeries) -> Arb

Return a certified upper bound on ‖f‖_∞ = sup_{u∈[0,2π]} |f(u)|.

For each interval [u_j, u_j+h], the Taylor expansion with remainder gives:
  |f(u)| ≤ Σ_{m=0}^{M_order} |f^{(m)}(u_j)|/m! · h^m
            +  sup_bounds[M_order+2] · h^{M_order+1}/(M_order+1)!

We take the maximum over all j.
"""
function sup_bound(ts::TaylorSeries)::Arb
    prec = ts.prec
    M = ts.M_order

    hpow_over_fact, rem_factor = _taylor_step_weights(M, ts.h, prec)

    # The Fbound·rem_factor contribution is a sup_bounds-derived correction;
    # by convention it must enter the result's RADIUS via add_error!, not the
    # midpoint via Arb addition, so the returned Arb's midpoint stays
    # informative (= the per-grid Taylor partial sum) and the radius carries
    # the looseness of the sup_bound chain.
    Fbound = ts.sup_bounds[M + 2]
    fbound_radius_arf = Arblib.ubound(Fbound * rem_factor)

    best, _ = _supbound_scan(ts.derivs, 1:(ts.M_grid), M, prec,
                             hpow_over_fact, fbound_radius_arf,
                             Arb(0; prec=prec), Arblib.Arf())
    return best
end

# ---------------------------------------------------------------------------
# inf_bound — certified lower bound on min f(u)
# ---------------------------------------------------------------------------

"""
    _infbound_scan(derivs, jrows, M, prec, hpow_over_fact, fbound_radius_arf,
                   best, best_lb, initialized) -> (best, best_lb, initialized)

Row-scan kernel of `inf_bound` over the rows `jrows` of `derivs`, updating
the running `(best, best_lb, initialized)` state. Rows must be visited in
ascending order (within and across calls) for bit-identical tie-breaking
with the monolithic scan.
"""
function _infbound_scan(derivs::AbstractMatrix{Arb}, jrows, M::Int, prec::Int,
                        hpow_over_fact::Vector{Arb},
                        fbound_radius_arf::Arblib.Arf,
                        best::Arb, best_lb::Arblib.Arf, initialized::Bool)
    for j in jrows
        f0 = derivs[j, 1]
        tail = Arb(0; prec=prec)
        for m in 1:M
            tail = tail + abs(derivs[j, m+1]) * hpow_over_fact[m+1]
        end
        lb = f0 - tail
        Arblib.add_error!(lb, fbound_radius_arf)
        lb_val = Arblib.lbound(lb)
        if !initialized || lb_val < best_lb
            best    = lb
            best_lb = lb_val
            initialized = true
        end
    end
    return best, best_lb, initialized
end

"""
    inf_bound(ts::TaylorSeries) -> Arb

Return a certified lower bound on inf_{u∈[0,2π]} f(u).

For each interval [u_j, u_j+h], the Taylor expansion gives:
  f(u) ≥ f(u_j) − Σ_{m=1}^{M_order} |f^{(m)}(u_j)|/m! · h^m
                  − sup_bounds[M_order+2] · h^{M_order+1}/(M_order+1)!
"""
function inf_bound(ts::TaylorSeries)::Arb
    prec = ts.prec
    M = ts.M_order

    hpow_over_fact, rem_factor = _taylor_step_weights(M, ts.h, prec)

    # See sup_bound for the convention: sup_bounds-derived correction enters
    # via add_error! (radius), not via Arb subtraction (midpoint).
    Fbound = ts.sup_bounds[M + 2]
    fbound_radius_arf = Arblib.ubound(Fbound * rem_factor)

    best, _, _ = _infbound_scan(ts.derivs, 1:(ts.M_grid), M, prec,
                                hpow_over_fact, fbound_radius_arf,
                                ts.derivs[1, 1] - Arb(0; prec=prec),
                                Arblib.Arf(), false)
    return best
end

# ---------------------------------------------------------------------------
# tighten_sup_bounds! — replace formula-based sup_bounds by grid+Lipschitz
# ---------------------------------------------------------------------------

"""
    _grid_abs_max_ubs!(max_ubs, initialized, derivs, jrows, M)

Grid-reduction kernel of `tighten_sup_bounds!`: fold the rows `jrows` of
`derivs` into the running per-order maxima `max_ubs[m+1] =
max_j ubound(|derivs[j, m+1]|)` (with `initialized[m+1]` tracking whether
any row has been seen). Pure max — shard-combinable in any order; ascending
kept for uniformity.
"""
function _grid_abs_max_ubs!(max_ubs::Vector{Arblib.Arf},
                            initialized::Vector{Bool},
                            derivs::AbstractMatrix{Arb}, jrows, M::Int)
    for m in M:-1:0
        max_abs_ub = max_ubs[m+1]
        init = initialized[m+1]
        for j in jrows
            ub = Arblib.ubound(abs(derivs[j, m+1]))
            if !init || ub > max_abs_ub
                max_abs_ub = ub
                init = true
            end
        end
        max_ubs[m+1] = max_abs_ub
        initialized[m+1] = init
    end
    return max_ubs, initialized
end

"""
    _tighten_from_max!(sup_bounds, max_ubs, M, h, prec)

Scalar kernel of `tighten_sup_bounds!`: given the per-order grid maxima
`max_ubs`, walk m = M..0 applying the grid+Lipschitz update to `sup_bounds`
in place (adopting only if strictly tighter). This is exactly the scalar
tail of the original loop; the sharded driver calls it with maxima
accumulated across shards.
"""
function _tighten_from_max!(sup_bounds::Vector{Arb},
                            max_ubs::Vector{Arblib.Arf},
                            M::Int, h::Arb, prec::Int)
    for m in M:-1:0
        # Compute the rigorous upper bound: grid_max + lipschitz·h.
        # We package it as a "point Arb" at the ubound (lbound = ubound = the
        # rigorous bound) — matching the cap-ccf convention that sup_bounds
        # entries are scalar upper bounds. (The informative-midpoint convention
        # from §1 of the convention doc applies to derived/consumed values like
        # the final `sup_bound(ts_R)`, not to intermediate `sup_bounds`
        # entries.)
        new_sb_full = Arb(max_ubs[m+1]; prec=prec)
        Arblib.add_error!(new_sb_full, Arblib.ubound(sup_bounds[m+2] * h))
        new_sb_ub = Arblib.ubound(new_sb_full)
        # Only adopt if strictly tighter than the existing entry.
        if new_sb_ub < Arblib.ubound(sup_bounds[m+1])
            sup_bounds[m+1] = Arb(new_sb_ub; prec=prec)
        end
    end
    return sup_bounds
end

"""
    tighten_sup_bounds!(ts::TaylorSeries) -> ts

Re-populate `ts.sup_bounds[1..M_order+1]` using the grid+Lipschitz bound
    ‖f^(m)‖_∞  ≤  max_j |f^(m)(u_j)|  +  ‖f^(m+1)‖_∞ · h
walking from m = M_order down to m = 0.  The top entry `sup_bounds[M_order+2]`
is left unchanged (it seeds the recursion; its looseness gets damped by
factor `h` at every subsequent step, so after `M_order+1` levels of damping
the bottom entry's looseness is the top's × h^(M_order+1)).

The midpoint of each new sup_bound entry is `ubound(max_j |f^(m)(u_j)|)`
— the tight grid maximum (informative).  The radius carries the Lipschitz
correction, in line with the project's add_error! convention
(`docs/changes_2026-05-22_midpoint_convention.md`).

Entries are updated in place only if the new bound is tighter than what
was already there (so calling this on a TaylorSeries with already-tight
sup_bounds is a no-op).  Idempotent in this sense.

Cross-reference: cap-ccf's `taylor_series.jl` uses the same grid+Lipschitz
pattern at every sup-bound construction site; this brings BabenkoRigorous's
sqrt_series / inv_series / * outputs into line with that convention.

Implemented as `_grid_abs_max_ubs!` (grid reduction) followed by
`_tighten_from_max!` (scalar walk) — the two halves are shared with the
sharded driver.
"""
function tighten_sup_bounds!(ts::TaylorSeries)
    M = ts.M_order
    max_ubs = [Arblib.Arf() for _ in 0:M]
    initialized = fill(false, M + 1)
    _grid_abs_max_ubs!(max_ubs, initialized, ts.derivs, 1:ts.M_grid, M)
    _tighten_from_max!(ts.sup_bounds, max_ubs, M, ts.h, ts.prec)
    return ts
end

# ---------------------------------------------------------------------------
# deriv_series — differentiate a TaylorSeries
# ---------------------------------------------------------------------------

"""
    deriv_series(ts::TaylorSeries) -> TaylorSeries

Return the TaylorSeries for f' given the TaylorSeries for f.

The new derivative array is the old one shifted one column to the left
(`new_derivs[j, m+1] = ts.derivs[j, m+2]`), and the new sup_bounds vector
is the old one shifted by one entry (`new_sup_bounds = ts.sup_bounds[2:end]`).
No looseness factor — the new top-level supnorm bound is exactly the old
sup_bounds[M_order+2].

The new M_order is `ts.M_order − 1`. Requires `ts.M_order ≥ 1`.

(NB: the previous implementation multiplied the bound by (M_order+1); that
factor was unnecessary looseness and has been removed.)
"""
function deriv_series(ts::TaylorSeries)::TaylorSeries
    @assert ts.M_order >= 1 "Cannot differentiate a TaylorSeries with M_order = 0"
    prec = ts.prec
    M_new = ts.M_order - 1
    M_grid = ts.M_grid

    new_derivs = Matrix{Arb}(undef, M_grid, M_new + 1)
    for j in 1:M_grid
        for m in 0:M_new
            new_derivs[j, m+1] = ts.derivs[j, m+2]
        end
    end
    new_sup_bounds = ts.sup_bounds[2:end]
    return TaylorSeries(new_derivs, new_sup_bounds, M_new, M_grid, ts.h, prec)
end

# ---------------------------------------------------------------------------
# truncate_order — restrict a TaylorSeries to a lower order
# ---------------------------------------------------------------------------

"""
    truncate_order(ts::TaylorSeries, M_new::Int) -> TaylorSeries

Restrict a TaylorSeries from its current `M_order` down to `M_new ≤ M_order`,
keeping the first `M_new + 1` derivative columns and the first `M_new + 2`
entries of `sup_bounds`. The result represents the same underlying function,
just stored at lower Taylor order.

Useful when an arithmetic chain has been built at a higher order (to expose
extra sup_bound entries) but the downstream consumer only needs a lower-order
series — avoids re-computing the chain from scratch.
"""
function truncate_order(ts::TaylorSeries, M_new::Int)::TaylorSeries
    @assert 0 <= M_new <= ts.M_order "M_new must satisfy 0 ≤ M_new ≤ ts.M_order"
    new_derivs = ts.derivs[:, 1:(M_new + 1)]
    new_sup_bounds = ts.sup_bounds[1:(M_new + 2)]
    return TaylorSeries(new_derivs, new_sup_bounds, M_new, ts.M_grid, ts.h, ts.prec)
end

# ---------------------------------------------------------------------------
# Arithmetic: addition / subtraction
# ---------------------------------------------------------------------------

"""
    Base.:+(ts1::TaylorSeries, ts2::TaylorSeries) -> TaylorSeries

Add two TaylorSeries pointwise. Both must have the same M_grid and M_order.
New sup_bounds[m+1] = ts1.sup_bounds[m+1] + ts2.sup_bounds[m+1]   (triangle inequality).
"""
function Base.:+(ts1::TaylorSeries, ts2::TaylorSeries)::TaylorSeries
    @assert ts1.M_grid == ts2.M_grid "TaylorSeries must have the same M_grid"
    @assert ts1.M_order == ts2.M_order "TaylorSeries must have the same M_order"
    prec = ts1.prec
    M_grid = ts1.M_grid
    M = ts1.M_order

    new_derivs = Matrix{Arb}(undef, M_grid, M + 1)
    for j in 1:M_grid
        for m in 0:M
            new_derivs[j, m+1] = ts1.derivs[j, m+1] + ts2.derivs[j, m+1]
        end
    end
    new_sup_bounds = [ts1.sup_bounds[m+1] + ts2.sup_bounds[m+1] for m in 0:M+1]
    return TaylorSeries(new_derivs, new_sup_bounds, M, M_grid, ts1.h, prec)
end

# ---------------------------------------------------------------------------
# Arithmetic: multiplication (TaylorSeries × TaylorSeries)
# ---------------------------------------------------------------------------

"""
    _leibniz_sup_bounds(sb1, sb2, M, prec, binom_table) -> Vector{Arb}

Scalar kernel of `Base.:*`: the exact-Leibniz combination of two sup_bounds
vectors (orders 0..M+1). Extracted so the sharded driver replays the
identical Arb op sequence on the global scalar vectors.
"""
function _leibniz_sup_bounds(sb1::Vector{Arb}, sb2::Vector{Arb},
                             M::Int, prec::Int, binom_table)::Vector{Arb}
    new_sup_bounds = Vector{Arb}(undef, M + 2)
    for m in 0:(M+1)
        s = Arb(0; prec=prec)
        for k in 0:m
            s = s + Arb(binom_table[m+1, k+1]; prec=prec) *
                    sb1[k+1] * sb2[m-k+1]
        end
        new_sup_bounds[m+1] = s
    end
    return new_sup_bounds
end

"""
    Base.:*(ts1::TaylorSeries, ts2::TaylorSeries) -> TaylorSeries

Multiply two TaylorSeries. The m-th derivative of f·g is:
  (fg)^{(m)} = Σ_{k=0}^m C(m,k) f^{(k)} g^{(m-k)}.

sup_bounds via the same *exact* Leibniz formula on the sup_bounds vectors:
  new_sup_bounds[m+1] = Σ_{k=0}^m C(m,k) · ts1.sup_bounds[k+1] · ts2.sup_bounds[m-k+1]
for m = 0..M_order+1. This is provably rigorous (the binomial coefficients are
non-negative, so applying triangle inequality term-by-term gives a true upper
bound) and is the sharpest bound that only uses the per-derivative supnorms.

(NB: the previous implementation used `2^{M+1}(F1·S2 + S1·F2)`, which only
bounded the k=0 and k=M+1 Leibniz terms and was structurally insufficient.
The new formula uses the full sup_bounds vector and is exact-Leibniz tight.)
"""
function Base.:*(ts1::TaylorSeries, ts2::TaylorSeries)::TaylorSeries
    @assert ts1.M_grid == ts2.M_grid "TaylorSeries must have the same M_grid"
    @assert ts1.M_order == ts2.M_order "TaylorSeries must have the same M_order"
    prec = ts1.prec
    M_grid = ts1.M_grid
    M = ts1.M_order

    binom_table = [binomial(m, k) for m in 0:(M+1), k in 0:(M+1)]

    new_derivs = Matrix{Arb}(undef, M_grid, M + 1)
    Threads.@threads for j in 1:M_grid
        for m in 0:M
            s = Arb(0; prec=prec)
            for k in 0:m
                s = s + Arb(binom_table[m+1, k+1]; prec=prec) *
                        ts1.derivs[j, k+1] * ts2.derivs[j, m-k+1]
            end
            new_derivs[j, m+1] = s
        end
    end

    new_sup_bounds = _leibniz_sup_bounds(ts1.sup_bounds, ts2.sup_bounds,
                                         M, prec, binom_table)

    out = TaylorSeries(new_derivs, new_sup_bounds, M, M_grid, ts1.h, prec)
    # Leibniz on sup_bounds compounds input looseness (~10× per order from
    # the trace's ts_y_yu_h); tighten via grid+Lip using the tight per-grid
    # product values already computed above.
    return tighten_sup_bounds!(out)
end

# ---------------------------------------------------------------------------
# Arithmetic: scalar multiplication
# ---------------------------------------------------------------------------

"""
    Base.:*(c::Arb, ts::TaylorSeries) -> TaylorSeries

Scalar multiplication. Scale all derivs by c and all sup_bounds by |c|.
"""
function Base.:*(c::Arb, ts::TaylorSeries)::TaylorSeries
    prec = ts.prec
    M_grid = ts.M_grid
    M = ts.M_order
    abs_c = abs(c)

    new_derivs = Matrix{Arb}(undef, M_grid, M + 1)
    for j in 1:M_grid
        for m in 0:M
            new_derivs[j, m+1] = c * ts.derivs[j, m+1]
        end
    end
    new_sup_bounds = [abs_c * b for b in ts.sup_bounds]
    return TaylorSeries(new_derivs, new_sup_bounds, M, M_grid, ts.h, prec)
end

Base.:*(ts::TaylorSeries, c::Arb) = c * ts

function Base.:*(c::Real, ts::TaylorSeries)::TaylorSeries
    return Arb(c; prec=ts.prec) * ts
end

Base.:*(ts::TaylorSeries, c::Real) = c * ts

# ---------------------------------------------------------------------------
# Negation, subtraction
# ---------------------------------------------------------------------------

Base.:-(ts::TaylorSeries) = Arb(-1; prec=ts.prec) * ts
Base.:-(ts1::TaylorSeries, ts2::TaylorSeries) = ts1 + (-ts2)

# ---------------------------------------------------------------------------
# inv_series — TaylorSeries for 1/f
# ---------------------------------------------------------------------------

"""
    _inv_sup_bounds_on_interval(d_lower::Arb, order::Int, prec::Int) -> Vector{Arb}

Return the sup_bounds vector `[max|g^{(k)}|]_{k=0..order}` for g(x) = 1/x on
the interval [d_lower, +∞):
  g^{(k)}(x) = (-1)^k k! / x^{k+1}    ⇒   max|g^{(k)}| on [d_lower, ∞) = k! / d_lower^{k+1}
"""
function _inv_sup_bounds_on_interval(d_lower::Arb, order::Int, prec::Int)::Vector{Arb}
    out = Vector{Arb}(undef, order + 1)
    inv_d = Arb(1; prec=prec) / d_lower
    pow = inv_d                       # d_lower^{-1}
    fact = Arb(1; prec=prec)          # 0!
    out[1] = pow                       # k=0: 1/d_lower
    for k in 1:order
        fact = fact * Arb(k; prec=prec)
        pow = pow * inv_d              # d_lower^{-(k+1)}
        out[k+1] = fact * pow
    end
    return out
end

"""
    inv_series(ts::TaylorSeries, d_lower::Arb) -> TaylorSeries

Compute the TaylorSeries for 1/f given that f(u) ≥ d_lower > 0 for all u.

Per-point derivatives are computed by the standard recursion
  g^{(0)} = 1/f(u_j)
  g^{(m)} = -1/f(u_j) · Σ_{k=1}^m C(m,k) f^{(k)}(u_j) · g^{(m-k)}    (m ≥ 1)

The sup_bounds vector is obtained by Faà di Bruno composition (via
`_faa_di_bruno_derivs`) on the sup_bounds of g(x) = 1/x (on the interval
[d_lower, ∞)) and `ts.sup_bounds`. Because every coefficient in the Bell
polynomial expansion is non-negative, feeding non-negative supnorm inputs
gives a rigorous upper bound on the supnorm of each derivative of 1/f.
"""
function inv_series(ts::TaylorSeries, d_lower::Arb)::TaylorSeries
    @assert Arblib.is_positive(d_lower) "d_lower must be strictly positive"
    prec = ts.prec
    M_grid = ts.M_grid
    M = ts.M_order

    binom_table = [binomial(m, k) for m in 0:M, k in 0:M]

    new_derivs = Matrix{Arb}(undef, M_grid, M + 1)
    Threads.@threads for j in 1:M_grid
        f0 = ts.derivs[j, 1]
        inv_f0 = Arb(1; prec=prec) / f0

        g_derivs = Vector{Arb}(undef, M + 1)
        g_derivs[1] = inv_f0
        for m in 1:M
            s = Arb(0; prec=prec)
            for k in 1:m
                s = s + Arb(binom_table[m+1, k+1]; prec=prec) *
                        ts.derivs[j, k+1] * g_derivs[m-k+1]
            end
            g_derivs[m+1] = -inv_f0 * s
        end
        for m in 0:M
            new_derivs[j, m+1] = g_derivs[m+1]
        end
    end

    # sup_bounds via Faà di Bruno on g(x)=1/x sup_bounds × ts.sup_bounds.
    # Output sup_bounds vector has length M+2 (orders 0..M+1).
    g_sup_bounds = _inv_sup_bounds_on_interval(d_lower, M + 1, prec)
    new_sup_bounds = _faa_di_bruno_derivs(g_sup_bounds, ts.sup_bounds, M + 1, prec)

    out = TaylorSeries(new_derivs, new_sup_bounds, M, M_grid, ts.h, prec)
    # Faà di Bruno gives an over-conservative sup_bound; tighten via grid+Lip.
    return tighten_sup_bounds!(out)
end

# ---------------------------------------------------------------------------
# sqrt_series — TaylorSeries for sqrt(f)
# ---------------------------------------------------------------------------

"""
    _sqrt_sup_bounds_on_interval(d_lower::Arb, S_upper::Arb, order::Int, prec::Int)
        -> Vector{Arb}

Return the sup_bounds vector `[max|g^{(k)}|]_{k=0..order}` for g(x) = √x on
the interval [d_lower, S_upper] with d_lower > 0:
  g^{(0)} on the interval has max = √S_upper.
  g^{(k)}(x) = (1/2)_k · x^{1/2 − k} = (-1)^{k-1} (2k-3)!! / 2^k · x^{1/2-k}     for k ≥ 1
where (2k-3)!! = 1·3·5·…·(2k-3) (with (-1)!! = 1, 1!! = 1).
For k ≥ 1 the function x ↦ x^{1/2 − k} is decreasing on (0,∞), so
  max|g^{(k)}| on [d_lower, S_upper] = ((2k-3)!! / 2^k) · d_lower^{1/2 − k}.
"""
function _sqrt_sup_bounds_on_interval(d_lower::Arb, S_upper::Arb,
                                      order::Int, prec::Int)::Vector{Arb}
    out = Vector{Arb}(undef, order + 1)
    out[1] = sqrt(S_upper)
    if order >= 1
        two_pow = Arb(2; prec=prec)            # 2^1
        dd_half = Arb(1; prec=prec) / sqrt(d_lower)  # d_lower^{-1/2}
        inv_d = Arb(1; prec=prec) / d_lower    # d_lower^{-1}
        d_pow = dd_half                         # holds d_lower^{1/2 - k}, start at k=1 → d_lower^{-1/2}
        dfact = Arb(1; prec=prec)               # (2k-3)!! for k=1 is (-1)!! = 1
        out[2] = (dfact / two_pow) * d_pow      # k=1: (1/2) · d_lower^{-1/2}
        for k in 2:order
            two_pow = two_pow * Arb(2; prec=prec)        # 2^k
            dfact = dfact * Arb(2*k - 3; prec=prec)      # (2k-3)!!
            d_pow = d_pow * inv_d                          # d_lower^{1/2 - k}
            out[k+1] = (dfact / two_pow) * d_pow
        end
    end
    return out
end

"""
    sqrt_series(ts::TaylorSeries, d_lower::Arb) -> TaylorSeries

Compute the TaylorSeries for sqrt(f) given that sqrt(f(u)) ≥ d_lower > 0 for
all u (equivalently, f(u) ≥ d_lower^2).

Per-point derivatives are computed by the standard recursion (from g^2 = f):
  g^{(0)} = √f(u_j)
  g^{(m)} = [f^{(m)}(u_j) − Σ_{k=1}^{m-1} C(m,k) g^{(k)} g^{(m-k)}] / (2 g^{(0)})

The sup_bounds vector is obtained by Faà di Bruno composition on the
sup_bounds of g(x) = √x on the interval [d_lower^2, sup_bound(ts)] and
`ts.sup_bounds`. Same rigour argument as `inv_series`.
"""
function sqrt_series(ts::TaylorSeries, d_lower::Arb;
                     S_upper::Union{Nothing,Arb}=nothing)::TaylorSeries
    @assert Arblib.is_positive(d_lower) "d_lower must be strictly positive"
    prec = ts.prec
    M_grid = ts.M_grid
    M = ts.M_order

    binom_table = [binomial(m, k) for m in 0:M, k in 0:M]

    new_derivs = Matrix{Arb}(undef, M_grid, M + 1)
    Threads.@threads for j in 1:M_grid
        f0 = ts.derivs[j, 1]
        g0 = sqrt(f0)
        g_derivs = Vector{Arb}(undef, M + 1)
        g_derivs[1] = g0
        two_g0 = Arb(2; prec=prec) * g0
        for m in 1:M
            inner = Arb(0; prec=prec)
            for k in 1:(m-1)
                inner = inner + Arb(binom_table[m+1, k+1]; prec=prec) *
                                g_derivs[k+1] * g_derivs[m-k+1]
            end
            g_derivs[m+1] = (ts.derivs[j, m+1] - inner) / two_g0
        end
        for m in 0:M
            new_derivs[j, m+1] = g_derivs[m+1]
        end
    end

    # sup_bounds via Faà di Bruno composition with g(x)=√x on [d_lower², S_upper].
    # `S_upper` is injectable (sharded driver supplies the globally-reduced
    # value); the default reproduces the original in-place computation.
    S_upper = S_upper === nothing ? sup_bound(ts) : S_upper
    d2_lower = d_lower * d_lower
    g_sup_bounds = _sqrt_sup_bounds_on_interval(d2_lower, S_upper, M + 1, prec)
    new_sup_bounds = _faa_di_bruno_derivs(g_sup_bounds, ts.sup_bounds, M + 1, prec)

    out = TaylorSeries(new_derivs, new_sup_bounds, M, M_grid, ts.h, prec)
    # Faà di Bruno is catastrophically loose for high-mode input
    # (~10^10× looseness at m=5 for our PINN case); tighten via grid+Lip.
    return tighten_sup_bounds!(out)
end

# ---------------------------------------------------------------------------
# dmin_bound — certified lower bound on D(u) = sqrt(y_u^2 + (1+H[y_u])^2)
# ---------------------------------------------------------------------------

"""
    dmin_bound(y::TrigPoly, M_grid::Int, M_order::Int; prec::Int=256)
              -> Tuple{Arb, TaylorSeries}

Return (D_lower, ts_D) where:
  - ts_D is the TaylorSeries for D(u) = sqrt(y_u(u)^2 + (1 + H[y_u](u))^2)
  - D_lower = inf_bound(ts_D) is a certified lower bound on D over [0, 2π]

Algorithm: build ts_A and ts_B for A = y_u and B = 1 + H[y_u] via the
TrigPoly chain, square and add to get ts_D2 = A² + B², establish a
certified positive lower bound on √(ts_D2), and apply sqrt_series.
"""
function dmin_bound(y::TrigPoly, M_grid::Int, M_order::Int; prec::Int=256)::Tuple{Arb, TaylorSeries}
    p = prec

    yu  = deriv(y)
    Hyu = hilbert(yu)

    # Build the derivative chains up to order M_order+1 (one beyond stored order).
    yu_d  = Vector{TrigPoly}(undef, M_order + 2)
    Hyu_d = Vector{TrigPoly}(undef, M_order + 2)
    yu_d[1]  = yu
    Hyu_d[1] = Hyu
    for m in 1:(M_order + 1)
        yu_d[m+1]  = deriv(yu_d[m])
        Hyu_d[m+1] = deriv(Hyu_d[m])
    end

    # Full sup_bounds vectors for ts_A (= y_u) and ts_B (= 1 + H[y_u]).
    # Note B^{(0)} = 1 + H[y_u], B^{(m)} = H[y_u]^{(m)} for m ≥ 1; the constant 1
    # only shifts the m=0 supnorm.
    sup_bounds_A = [supnorm_bound(yu_d[m+1]) for m in 0:(M_order + 1)]
    sup_bounds_B = Vector{Arb}(undef, M_order + 2)
    sup_bounds_B[1] = Arb(1; prec=p) + supnorm_bound(Hyu_d[1])
    for m in 1:(M_order + 1)
        sup_bounds_B[m+1] = supnorm_bound(Hyu_d[m+1])
    end

    pi_arb = Arb(pi; prec=p)
    h = Arb(2; prec=p) * pi_arb / Arb(M_grid; prec=p)

    derivs_A = Matrix{Arb}(undef, M_grid, M_order + 1)
    derivs_B = Matrix{Arb}(undef, M_grid, M_order + 1)

    # Per-grid D lower bounds collected in an array so the loop body is
    # purely write-to-disjoint-locations (thread-safe). Reduction to the
    # global min happens after.
    Dj_lbs = Vector{Arblib.Arf}(undef, M_grid)
    Threads.@threads for j in 1:M_grid
        u_j = Arb(2; prec=p) * pi_arb * Arb(j - 1; prec=p) / Arb(M_grid; prec=p)
        for m in 0:M_order
            derivs_A[j, m+1] = tp_eval(yu_d[m+1], u_j)
            if m == 0
                derivs_B[j, 1] = Arb(1; prec=p) + tp_eval(Hyu_d[1], u_j)
            else
                derivs_B[j, m+1] = tp_eval(Hyu_d[m+1], u_j)
            end
        end
        Dj_lbs[j] = Arblib.lbound(sqrt(derivs_A[j,1]^2 + derivs_B[j,1]^2))
    end
    # Serial reduction over the per-grid lower bounds.
    D_lb_min_arf = Dj_lbs[1]
    for j in 2:M_grid
        if Dj_lbs[j] < D_lb_min_arf
            D_lb_min_arf = Dj_lbs[j]
        end
    end

    ts_A = TaylorSeries(derivs_A, sup_bounds_A, M_order, M_grid, h, p)
    ts_B = TaylorSeries(derivs_B, sup_bounds_B, M_order, M_grid, h, p)

    ts_D2 = ts_A * ts_A + ts_B * ts_B

    # Lipschitz lower bound on min D via D_j minus h × supnorm(D').
    # |D'| ≤ |A'| + |B'| (Cauchy-Schwarz inside the sqrt(A²+B²)).
    # Convention: the sup-norm-derived Lipschitz correction enters as a
    # radius (via add_error!) so d_lower_arb's midpoint stays at the
    # actual D_lb_min, with the looseness recorded in its radius.
    lip_correction_arb = supnorm_bound(yu_d[2]) + supnorm_bound(Hyu_d[2])
    d_lower_arb  = Arb(D_lb_min_arf; prec=p)
    Arblib.add_error!(d_lower_arb, Arblib.ubound(lip_correction_arb * h))
    d_lower_init = Arb(Arblib.lbound(d_lower_arb); prec=p)

    if !Arblib.is_positive(d_lower_init)
        error("dmin_bound: D is not certified strictly positive (d_lower_init ≤ 0).")
    end

    ts_D = sqrt_series(ts_D2, d_lower_init)
    # We have two valid rigorous lower bounds on min D:
    #   d_lower_init  : Lipschitz-based, tight at moderate grid sizes.
    #   inf_bound(ts_D): Taylor-remainder-based, only tight when sqrt_series's
    #                    sup_bounds for D's high derivatives are tight.
    # For high-mode input the Faà di Bruno sup_bounds are very loose, so
    # inf_bound(ts_D)'s lbound can be wildly negative while d_lower_init is
    # comfortably positive. Return whichever has the larger (= tighter) lbound.
    ib = inf_bound(ts_D)
    D_lower = Arblib.lbound(ib) > Arblib.lbound(d_lower_init) ? ib : d_lower_init

    return (D_lower, ts_D)
end

# ---------------------------------------------------------------------------
# tight_DG_sup_bounds — direct algebraic sup_bounds for D and G via TrigPoly
# ---------------------------------------------------------------------------

"""
    tight_DG_sup_bounds(y::TrigPoly, max_m::Int, D_lower::Arb; prec::Int=128)
        -> (D_sup::Vector{Arb}, G_sup::Vector{Arb})

Tight sup_bounds for ‖D^(m)‖_∞ and ‖G^(m)‖_∞, m = 0..max_m, via algebraic
recurrences derived from
    D · D = A² + B²    (so D² = A² + B², a trig poly)
    G · D = B          (so G = B/D)
with A = y_u, B = 1 + H[y_u].

Leibniz on m derivatives:
    2·D·D^(m)   = (D²)^(m) − Σ_{k=1..m-1} C(m,k)·D^(k)·D^(m-k)
    G^(m)·D     = B^(m)    − Σ_{k=0..m-1} C(m,k)·G^(k)·D^(m-k)

Taking sup-norms with D ≥ D_lower > 0:
    D_sup[m+1] ≤ [‖(D²)^(m)‖∞ + Σ_{k=1..m-1} C(m,k)·D_sup[k+1]·D_sup[m-k+1]] / (2·D_lower)
    G_sup[m+1] ≤ [‖B^(m)‖∞   + Σ_{k=0..m-1} C(m,k)·G_sup[k+1]·D_sup[m-k+1]] / D_lower

with base cases D_sup[1] = √(‖D²‖∞), G_sup[1] = ‖B‖∞ / D_lower.

`‖(D²)^(m)‖∞` and `‖B^(m)‖∞` are computed exactly via TrigPoly
multiplication + `deriv` + `supnorm_bound` (triangle inequality on the
coefficient vectors). No Faà-di-Bruno expansion enters; the bounds avoid
the combinatorial blowup that's catastrophic for high-mode input
(~10¹¹× looseness at m=5 for the PINN solution via `sqrt_series`).

Intended use: feed `G_sup[M_order+4]` and `G_sup[M_order+5]` as
`Gpp_bound_Gu` and `Gppp_bound_Gu` into `fft_hilbert_series(ts_Gu; …)`,
replacing the loose top entries of `ts_G3`.

See `docs/changes_2026-05-22_midpoint_convention.md` §5.
"""
function tight_DG_sup_bounds(y::TrigPoly, max_m::Int, D_lower::Arb;
                              prec::Int=128)::Tuple{Vector{Arb}, Vector{Arb}}
    @assert max_m >= 0
    @assert Arblib.is_positive(D_lower) "D_lower must be strictly positive"

    A  = deriv(y)
    B  = add_const(hilbert(A), 1.0)
    D2 = A * A + B * B

    # ‖(D²)^(m)‖∞ and ‖B^(m)‖∞ for m = 0..max_m via TrigPoly diff + supnorm_bound.
    D2_sup = Vector{Arb}(undef, max_m + 1)
    B_sup  = Vector{Arb}(undef, max_m + 1)
    D2_curr = D2
    B_curr  = B
    D2_sup[1] = supnorm_bound(D2_curr)
    B_sup[1]  = supnorm_bound(B_curr)
    for m in 1:max_m
        D2_curr = deriv(D2_curr)
        B_curr  = deriv(B_curr)
        D2_sup[m + 1] = supnorm_bound(D2_curr)
        B_sup[m + 1]  = supnorm_bound(B_curr)
    end

    # D_sup recurrence.
    D_sup = Vector{Arb}(undef, max_m + 1)
    D_sup[1] = sqrt(D2_sup[1])
    two_Dlower = Arb(2; prec=prec) * D_lower
    for m in 1:max_m
        s = D2_sup[m + 1]
        for k in 1:(m - 1)
            cmk = Arb(binomial(m, k); prec=prec)
            s = s + cmk * D_sup[k + 1] * D_sup[m - k + 1]
        end
        D_sup[m + 1] = s / two_Dlower
    end

    # G_sup recurrence.
    G_sup = Vector{Arb}(undef, max_m + 1)
    G_sup[1] = B_sup[1] / D_lower
    for m in 1:max_m
        s = B_sup[m + 1]
        for k in 0:(m - 1)
            cmk = Arb(binomial(m, k); prec=prec)
            s = s + cmk * G_sup[k + 1] * D_sup[m - k + 1]
        end
        G_sup[m + 1] = s / D_lower
    end

    return (D_sup, G_sup)
end

# ---------------------------------------------------------------------------
# trigpoly_to_taylor — convert a TrigPoly to a TaylorSeries
# ---------------------------------------------------------------------------

"""
    trigpoly_to_taylor(f::TrigPoly, M_grid::Int, M_order::Int; prec::Int=128)
        -> TaylorSeries

Convert a TrigPoly `f` to a TaylorSeries by evaluating `f` and its iterated
derivatives at `M_grid` equispaced grid points. The sup_bounds vector is
populated via `supnorm_bound` on each derivative TrigPoly (rigorous because
TrigPoly derivative is exact and `supnorm_bound` is the rigorous triangle
inequality).
"""
function trigpoly_to_taylor(f::TrigPoly, M_grid::Int, M_order::Int;
                             prec::Int=128,
                             j_range::UnitRange{Int}=1:M_grid)::TaylorSeries
    p = prec

    f_d = Vector{TrigPoly}(undef, M_order + 2)
    f_d[1] = f
    for m in 1:(M_order + 1)
        f_d[m+1] = deriv(f_d[m])
    end

    sup_bounds = [supnorm_bound(f_d[m+1]) for m in 0:(M_order + 1)]

    pi_arb = Arb(pi; prec=p)
    h = Arb(2; prec=p) * pi_arb / Arb(M_grid; prec=p)

    # `j_range` selects a contiguous block of GLOBAL grid indices (sharded
    # evaluation): u_j and h always use the global M_grid, so each row is
    # bit-identical to the corresponding row of the full-grid call. The
    # returned TaylorSeries has `length(j_range)` rows.
    n_rows = length(j_range)
    j0 = first(j_range) - 1
    derivs = Matrix{Arb}(undef, n_rows, M_order + 1)
    # Per-grid TrigPoly evaluation is embarrassingly parallel — no shared
    # mutable state. Same pattern as cap-ccf's hermite_functions.jl
    # `Threads.@threads for pc in ...` over disjoint point chunks.
    Threads.@threads for r in 1:n_rows
        j = j0 + r
        u_j = Arb(2; prec=p) * pi_arb * Arb(j - 1; prec=p) / Arb(M_grid; prec=p)
        for m in 0:M_order
            derivs[r, m+1] = tp_eval(f_d[m+1], u_j)
        end
    end

    return TaylorSeries(derivs, sup_bounds, M_order, n_rows, h, p)
end

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

export TaylorSeries, sup_bound, inf_bound, deriv_series, truncate_order,
       tighten_sup_bounds!
export inv_series, sqrt_series, dmin_bound, tight_DG_sup_bounds
export trigpoly_to_taylor
