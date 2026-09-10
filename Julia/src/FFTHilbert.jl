"""
FFTHilbert.jl — Certified discrete Hilbert transform via AcbPoly arithmetic.

Milestone M4: Given Arb ball-valued samples of a smooth periodic G, compute
H[G] at the same grid points with a certified aliasing error bound added to
each output ball.

All arithmetic is performed in the Arb / Acb ball field. No Float64
intermediate values are used and no informal error constants enter
the computation. FLINT's `Arblib.mul!` provides certified O(N log N)
polynomial multiplication; every rounding error is tracked automatically
by Arb's interval arithmetic. The only mathematically-derived error term
added by hand is the aliasing bound, derived rigorously below.

Two main functions:
  fft_hilbert(G_vals, N_fft, p, Gp_sup; prec)
      ->  (HG_vals, ε_alias)

  fft_hilbert_series(ts_G, N_fft; Gpp_bound, Gppp_bound, prec)
      ->  TaylorSeries for H[G]
    The Gpp_bound and Gppp_bound kwargs supply bounds on G^{(M_order+2)} and
    G^{(M_order+3)} respectively; both are required to certify the aliasing
    at the top derivative level (Gpp_bound) and the output series' top-two
    sup_bounds via Fourier-tail decay (Gpp_bound and Gppp_bound).

Mathematical background
=======================
For a real 2π-periodic G with Fourier series G(u) = Σ_k c_k exp(iku), the
continuous Hilbert transform is

    H[G](u) = Σ_{k≠0} (-i sign k) c_k exp(iku).

With N equally-spaced samples u_j = 2π(j-1)/N (N even), the natural discrete
Hilbert transform — the one that exactly inverts the DFT-multiply-IDFT
pipeline — is the circular convolution

    (H_N[G])_j = Σ_{m=0}^{N-1} h_m G_{(j-m) mod N},

with kernel

    h_n = (2/N) cot(πn/N)  for n odd in [1, N-1]
    h_n = 0                 for n even (including n=0 and n=N/2).

Its DFT equals H_k = -i for k=1..N/2-1, +i for k=N/2+1..N-1, 0 at k=0 and
k=N/2 — exactly the spectral Hilbert multiplier, with Nyquist zeroed for
unambiguous sign.

Implementation
==============
The circular convolution h ⋆ G is implemented as a linear convolution
followed by folding modulo x^N − 1:

    Let p_G(x) = Σ_{j=0}^{N-1} G_j x^j, p_h(x) = Σ_{j=0}^{N-1} h_j x^j.
    Their product p_G · p_h has 2N − 1 coefficients c_0, ..., c_{2N-2},
    with c_k = Σ_{j+m=k} G_j h_m.
    Then (H_N[G])_j = c_j + c_{j+N}   for j = 0..N-2,
         (H_N[G])_{N-1} = c_{N-1}     (c_{2N-1} := 0).

The polynomial multiplication is performed by `Arblib.mul!` on `AcbPoly`
objects, which dispatches to FLINT's certified FFT for large N. Real inputs
are lifted to Acb with zero imaginary part; real outputs are extracted via
`Arblib.get_real!`.

Aliasing bound
==============
The DFT-based formula aliases modes: G_hat_k = Σ_l c_{k + lN}. Under our
multiplier convention, M_k equals the continuous multiplier M_∞(m) whenever
m mod N corresponds to a frequency in [-N/2+1, N/2-1] of the correct sign.
The per-mode disagreement |M_k − M_∞(m)| is at most 2 across all aliased
modes |m| ≥ N/2. Combining with the Fourier decay |c_m| ≤ Gp_sup / |m|^p
(valid for any p such that Gp_sup ≥ ‖G^{(p)}‖_∞),

    |H[G](u_j) − H_N[G](u_j)|
       ≤ 2 · Σ_{|m| ≥ N/2} |c_m|
       ≤ 4 · Σ_{m ≥ N/2} |c_m|                                (real G ⇒ |c_{-m}| = |c_m|)
       ≤ 4 · Gp_sup · Σ_{m ≥ N/2} m^{-p}
       ≤ 4 · Gp_sup · (N/2 − 1)^{1−p} / (p − 1)               (integral test, p ≥ 2),

using Σ_{m ≥ M} m^{-p} ≤ ∫_{M-1}^∞ x^{-p} dx = (M-1)^{1-p}/(p-1), valid for
any M ≥ 2 and decreasing integrand m → m^{-p}. The factor 2 in the first
step is the maximum multiplier disagreement; the additional factor 2 in the
second step is real-Fourier symmetry.
"""

using Arblib

# ---------------------------------------------------------------------------
# fft_hilbert
# ---------------------------------------------------------------------------

"""
    fft_hilbert(G_vals::Vector{Arb}, N_fft::Int, p::Int, Gp_sup::Arb;
                prec::Int=128) -> Tuple{Vector{Arb}, Arb}

Compute the Hilbert transform H[G] at N_fft equally-spaced grid points and
return certified Arb balls for each output value.

Arguments
  G_vals  — length-N_fft vector of Arb samples G(u_j), u_j = 2π(j-1)/N_fft.
            Each ball encloses the true value G(u_j); the radii propagate
            automatically through the Acb-poly arithmetic below.
  N_fft   — number of grid points (must equal length(G_vals); must be even ≥ 4).
  p       — smoothness order: |G^{(p)}(u)| ≤ Gp_sup for all u; must be ≥ 2.
  Gp_sup  — certified upper bound on ‖G^{(p)}‖_∞.
  prec    — Arb working precision in bits.

Returns (HG_vals, ε_alias) where each HG_vals[j] is a certified Arb ball
enclosing the continuous H[G](u_j), and ε_alias is the aliasing component
added to each output ball.
"""
function fft_hilbert(G_vals::Vector{Arb}, N_fft::Int, p::Int, Gp_sup::Arb;
                     prec::Int=128)::Tuple{Vector{Arb}, Arb}
    @assert length(G_vals) == N_fft "length(G_vals) must equal N_fft"
    @assert p >= 2 "smoothness order p must be ≥ 2"
    @assert N_fft >= 4 "N_fft must be ≥ 4"
    @assert iseven(N_fft) "N_fft must be even"

    # ---------------------------------------------------------------------
    # Step 1: build the discrete Hilbert kernel h ∈ Arb^N.
    #   h_n = (2/N) cot(πn/N) for n odd; h_n = 0 for n even.
    # sin/cos/π are certified by Arb.
    # ---------------------------------------------------------------------
    pi_arb     = Arb(pi; prec=prec)
    N_arb      = Arb(N_fft; prec=prec)
    two_over_N = Arb(2; prec=prec) / N_arb

    h = Vector{Arb}(undef, N_fft)
    for n in 0:(N_fft - 1)
        if iseven(n)
            h[n + 1] = Arb(0; prec=prec)
        else
            arg      = pi_arb * Arb(n; prec=prec) / N_arb
            h[n + 1] = two_over_N * cos(arg) / sin(arg)
        end
    end

    # ---------------------------------------------------------------------
    # Step 2: linear convolution c = G ⋆ h via AcbPoly multiplication.
    # Real inputs are lifted to Acb with zero imaginary part. `Arblib.mul!`
    # dispatches to certified FFT internally; every Acb rounding error is
    # tracked by the ball arithmetic.
    # ---------------------------------------------------------------------
    p1      = Arblib.AcbPoly(N_fft;         prec=prec)
    p2      = Arblib.AcbPoly(N_fft;         prec=prec)
    product = Arblib.AcbPoly(2 * N_fft - 1; prec=prec)

    zr = Arb(0; prec=prec)
    for j in 1:N_fft
        Arblib.set_coeff!(p1, j - 1, Acb(G_vals[j], zr))
        Arblib.set_coeff!(p2, j - 1, Acb(h[j],      zr))
    end
    Arblib.mul!(product, p1, p2)

    # ---------------------------------------------------------------------
    # Step 3: fold linear → circular convolution.
    #   (H_N[G])_j = c_j + c_{j+N}     for j = 0..N-2
    #   (H_N[G])_{N-1} = c_{N-1}       (c_{2N-1} = 0, out of range)
    # ---------------------------------------------------------------------
    HG_vals   = Vector{Arb}(undef, N_fft)
    coeff_tmp = Acb(; prec=prec)
    real_tmp  = Arb(; prec=prec)

    for j in 0:(N_fft - 1)
        Arblib.get_coeff!(coeff_tmp, product, j)
        HG_vals[j + 1] = Arb(; prec=prec)
        Arblib.get_real!(HG_vals[j + 1], coeff_tmp)

        wrap_idx = j + N_fft
        if wrap_idx <= 2 * N_fft - 2
            Arblib.get_coeff!(coeff_tmp, product, wrap_idx)
            Arblib.get_real!(real_tmp, coeff_tmp)
            HG_vals[j + 1] += real_tmp
        end
    end

    # ---------------------------------------------------------------------
    # Step 4: certified aliasing bound
    #   ε_alias = 4 · Gp_sup / ((p - 1) · (N/2 − 1)^{p-1})
    # See module docstring for derivation.
    # ---------------------------------------------------------------------
    M_minus_1 = Arb(N_fft ÷ 2 - 1; prec=prec)
    pm1       = Arb(p - 1;         prec=prec)
    ε_alias   = Arb(4; prec=prec) * Gp_sup / (pm1 * M_minus_1^(p - 1))

    # ---------------------------------------------------------------------
    # Step 5: add aliasing component to each output ball. All other error
    # sources (FFT rounding, propagation of input ball radii) are already
    # tracked automatically by Arb interval arithmetic.
    # ---------------------------------------------------------------------
    for j in 1:N_fft
        Arblib.add_error!(HG_vals[j], ε_alias)
    end

    return (HG_vals, ε_alias)
end

# ---------------------------------------------------------------------------
# fft_hilbert_series
# ---------------------------------------------------------------------------

"""
    _fft_level_params(M_order, m, F_top, Gpp_bound) -> (p_m, B_m)

Smoothness order and sup-bound source for level `m` of
`fft_hilbert_series` (shared with the sharded driver).
"""
function _fft_level_params(M_order::Int, m::Int, F_top::Arb, Gpp_bound::Arb)
    natural_p = M_order + 1 - m
    if natural_p >= 2
        return natural_p, F_top
    else
        # m = M_order: clamp to p_m = 2 using Gpp_bound on G^{(M_order+2)}.
        return 2, Gpp_bound
    end
end

"""
    fft_hilbert_series(ts_G::TaylorSeries, N_fft::Int;
                       Gpp_bound::Arb,
                       Gppp_bound::Arb,
                       prec::Int=128) -> TaylorSeries

Compute the TaylorSeries for H[G] from the TaylorSeries of G.

Uses (H[G])^{(m)} = H[G^{(m)}] for each derivative level m = 0..M_order.

Smoothness bookkeeping for per-grid-point aliasing.
For each level m we have |c_k(G^{(m)})| = |k|^m |c_k(G)|, and integration by
parts gives `|c_k(G)| ≤ ‖G^{(p)}‖_∞ / |k|^p` for any p.
- m ∈ {0, …, M_order − 1}: use the natural smoothness p_m = M_order + 1 − m
  and bound `ts_G.sup_bounds[M_order + 2]` ≥ ‖G^{(M_order+1)}‖_∞.
- m = M_order: natural p_m = 1 diverges; require `Gpp_bound ≥ ‖G^{(M_order+2)}‖_∞`
  and certify with p = 2.

Output sup_bounds (the new bit — fixes the Lebesgue-constant unsoundness).
For each level m of the output we need a bound on `‖(H[G])^{(m)}‖_∞ = ‖H[G^{(m)}]‖_∞`.
Two integrations by parts give `|c_k(G^{(m)})| ≤ ‖G^{(m+2)}‖_∞ / k²`, hence
    ‖H[G^{(m)}]‖_∞ ≤ Σ_{k≠0} |c_k(G^{(m)})| ≤ ‖G^{(m+2)}‖_∞ · 2 · Σ_{k≥1} k^{-2}
                                            = ‖G^{(m+2)}‖_∞ · π²/3.
The required sup_bounds source for each output level m:
    m = 0 .. M_order − 1   →   ts_G.sup_bounds[m + 3]   (= ‖G^{(m+2)}‖_∞)
    m = M_order            →   Gpp_bound                (= ‖G^{(M_order+2)}‖_∞)
    m = M_order + 1        →   Gppp_bound               (= ‖G^{(M_order+3)}‖_∞)

The caller is responsible for providing rigorous `Gpp_bound` and `Gppp_bound`.
In ResidualBound.jl, these are extracted from extended (`M_order + 1` and
`M_order + 2`) TaylorSeries products.

(NB: the previous implementation set the output F_bound to
`ts_G.F_bound · (2 log N + 2)` — the discrete Lebesgue constant. That bound
is unsound because the continuous Hilbert is unbounded on L^∞; the formula
above replaces it with a Fourier-tail decay that genuinely converges.)
"""
function fft_hilbert_series(ts_G::TaylorSeries, N_fft::Int;
                            Gpp_bound::Arb,
                            Gppp_bound::Arb,
                            prec::Int=128)::TaylorSeries
    M_order = ts_G.M_order
    M_grid  = ts_G.M_grid
    @assert M_grid == N_fft "Currently requires N_fft == ts_G.M_grid"

    # -------------------------------------------------------------------
    # Per-grid-point Hilbert at each derivative level m = 0..M_order
    # -------------------------------------------------------------------
    new_derivs = Matrix{Arb}(undef, M_grid, M_order + 1)
    F_top = ts_G.sup_bounds[M_order + 2]   # ‖G^{(M_order+1)}‖_∞

    for m in 0:M_order
        G_m_vals = [ts_G.derivs[j, m+1] for j in 1:M_grid]

        p_m, B_m = _fft_level_params(M_order, m, F_top, Gpp_bound)

        HG_m_vals, _ = fft_hilbert(G_m_vals, N_fft, p_m, B_m; prec=prec)

        for j in 1:M_grid
            new_derivs[j, m+1] = HG_m_vals[j]
        end
    end

    # -------------------------------------------------------------------
    # Output sup_bounds via Fourier-tail decay: bound for level m is
    #     ‖G^{(m+2)}‖_∞ · π²/3
    # -------------------------------------------------------------------
    pi_arb     = Arb(pi; prec=prec)
    factor_pi2 = pi_arb * pi_arb / Arb(3; prec=prec)   # π²/3

    new_sup_bounds = Vector{Arb}(undef, M_order + 2)
    for m in 0:(M_order - 1)
        # ts_G.sup_bounds has length M_order + 2; entry [m+3] bounds G^{(m+2)}.
        new_sup_bounds[m + 1] = ts_G.sup_bounds[m + 3] * factor_pi2
    end
    new_sup_bounds[M_order + 1] = Gpp_bound  * factor_pi2   # m = M_order
    new_sup_bounds[M_order + 2] = Gppp_bound * factor_pi2   # m = M_order + 1

    return TaylorSeries(new_derivs, new_sup_bounds, M_order, M_grid, ts_G.h, prec)
end

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

export fft_hilbert, fft_hilbert_series
