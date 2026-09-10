"""
ResidualBound.jl — Certified upper bound on ‖R‖_∞ for the Babenko equation.

M5: β = 0 (no capillary terms).
M6: β ≠ 0 (full gravity-capillary residual via Path B).

The Babenko equation:
  μ H[y_u] = y + y H[y_u] + H[y y_u] - β ∂_u(y_u/D) + β H[∂_u(G)]

where D(u) = sqrt(y_u² + (1+H[y_u])²) and G(u) = (1+H[y_u])/D.

The residual:
  R = μ H[y_u] - y - y H[y_u] - H[y y_u] + β ∂_u(y_u/D) - β H[∂_u(G)]

β = 0: only Path A terms (exact trig-poly arithmetic).
β ≠ 0: Path A terms plus Path B capillary terms via GDerivatives + FFTHilbert.
"""

using Arblib

"""
    residual_bound(y_coeffs, μ, β=0.0; M_order, N_grid, prec) -> Arb

Certified upper bound on ‖R‖_∞ for the Babenko residual:
  R = μ H[y_u] - y - y H[y_u] - H[y y_u] + β ∂_u(y_u/D) - β H[∂_u(G)]

`y_coeffs` follows the cosine_series convention:
  y_coeffs[1] = DC term, y_coeffs[k+1] = coefficient of cos(ku).

For β ≠ 0, requires D(u) > 0 everywhere (certified via dmin_bound). The
capillary terms are computed at Taylor order M_order+1 and then differentiated
once to maintain consistency with the M_order base residual.
"""
function residual_bound(y_coeffs::Vector{<:Real}, μ::Real, β::Real=0.0;
                        M_order::Int, N_grid::Int, prec::Int=128,
                        extra_orders::Int=0)::Arb
    return _residual_bound_full(y_coeffs, μ, β;
                                M_order=M_order, N_grid=N_grid, prec=prec,
                                extra_orders=extra_orders).bound
end

"""
    _residual_bound_full(y_coeffs, μ, β=0.0; M_order, N_grid, prec) -> NamedTuple

Internal entry point that runs the full residual pipeline and returns all the
intermediate quantities required to emit a Lean certificate (in addition to
`bound = sup_bound(ts_R)`).

Fields of the returned NamedTuple:
  bound          :: Arb              — certified upper bound on ‖R‖_∞
  ts_R           :: TaylorSeries     — full residual TaylorSeries (at M_order)
  ts_y           :: TaylorSeries     — y at order M_order
  ts_yu          :: TaylorSeries     — y_u at order M_order
  ts_Hyu         :: TaylorSeries     — H[y_u] at order M_order
  Gpp_bound_yyu  :: Arb              — ‖(y·y_u)^{(M_order+2)}‖_∞ bound
  Gppp_bound_yyu :: Arb              — ‖(y·y_u)^{(M_order+3)}‖_∞ bound
  ts_G3          :: Union{TaylorSeries, Nothing} — G at order M_order+3, β≠0 only
  Gpp_bound_Gu   :: Union{Arb, Nothing}          — β≠0 only
  Gppp_bound_Gu  :: Union{Arb, Nothing}          — β≠0 only
  D_lower        :: Union{Arb, Nothing}          — β≠0 only

`ts_R` and the various intermediate `TaylorSeries` provide all per-grid-point
derivative balls and sup_bounds vectors needed by `emit_certificate`.
"""
function _residual_bound_full(y_coeffs::Vector{<:Real}, μ::Real, β::Real=0.0;
                              M_order::Int, N_grid::Int, prec::Int=128,
                              extra_orders::Int=0)
    # `extra_orders` extends the chain that feeds Gpp_bound / Gppp_bound by
    # additional derivative orders, so the consumed sup_bounds entries sit
    # below the loose Faà-di-Bruno top of the chain and benefit from
    # `tighten_sup_bounds!`'s recursive grid+Lipschitz damping (factor h per
    # level). Default 0 reproduces the old behaviour; for high-mode inputs
    # like the PINN solution, ~4–8 extra orders bring the radius from 10^17
    # down to ~10^-6 (see docs/changes_2026-05-22_midpoint_convention.md §4).
    # ------------------------------------------------------------------
    # Step 1: build TrigPoly for y, y_u, H[y_u]  (Path A — exact)
    # ------------------------------------------------------------------
    y   = cosine_series(y_coeffs; prec=prec)
    yu  = deriv(y)
    Hyu = hilbert(yu)

    # ------------------------------------------------------------------
    # Step 2: convert exact TrigPolys to TaylorSeries.
    # ------------------------------------------------------------------
    # Build y and y_u at the extended order M_h = M_order + 2 + extra_orders
    # so the product ts_y_yu_h carries enough sup_bounds entries to feed
    # fft_hilbert_series's Gpp_bound / Gppp_bound; truncate to M_order for
    # the residual core.
    M_h = M_order + 2 + extra_orders
    ts_y_h  = trigpoly_to_taylor(y,  N_grid, M_h; prec=prec)
    ts_yu_h = trigpoly_to_taylor(yu, N_grid, M_h; prec=prec)
    ts_Hyu  = trigpoly_to_taylor(Hyu, N_grid, M_order; prec=prec)

    ts_y  = truncate_order(ts_y_h,  M_order)
    ts_yu = truncate_order(ts_yu_h, M_order)

    # ------------------------------------------------------------------
    # Step 3: products via Leibniz rule.  ts_y_yu is built at M_h to expose
    # the extra sup_bounds entries; ts_y_yu (at M_order) is the truncation.
    # ------------------------------------------------------------------
    ts_y_Hyu  = ts_y * ts_Hyu                          # y · H[y_u]  at M_order
    ts_y_yu_h = ts_y_h * ts_yu_h                       # y · y_u     at M_h
    ts_y_yu   = truncate_order(ts_y_yu_h, M_order)     # y · y_u     at M_order

    # ts_y_yu_h.sup_bounds has length M_h + 2 = M_order + 4 + extra_orders:
    #   entry [M_order+3] bounds (y·y_u)^{(M_order+2)} → Gpp_bound
    #   entry [M_order+4] bounds (y·y_u)^{(M_order+3)} → Gppp_bound
    # With extra_orders > 0 these consumed entries sit below the loose top,
    # so tighten_sup_bounds! has damped them via grid+Lipschitz.
    Gpp_bound_yyu  = ts_y_yu_h.sup_bounds[M_order + 3]
    Gppp_bound_yyu = ts_y_yu_h.sup_bounds[M_order + 4]

    # ------------------------------------------------------------------
    # Step 4: H[y·y_u] via certified DFT Hilbert transform
    # ------------------------------------------------------------------
    ts_Hyyu = fft_hilbert_series(ts_y_yu, N_grid;
                                 Gpp_bound=Gpp_bound_yyu,
                                 Gppp_bound=Gppp_bound_yyu,
                                 prec=prec)

    # ------------------------------------------------------------------
    # Step 5: assemble β=0 residual
    # ------------------------------------------------------------------
    ts_R = Arb(μ; prec=prec) * ts_Hyu - ts_y - ts_y_Hyu - ts_Hyyu

    # Capillary-only outputs (left as nothing for β=0).
    ts_G3_out          = nothing
    Gpp_bound_Gu_out   = nothing
    Gppp_bound_Gu_out  = nothing
    D_lower_out        = nothing

    # ------------------------------------------------------------------
    # Step 6: capillary terms for β ≠ 0  (Path B)
    # ------------------------------------------------------------------
    if β != 0.0
        @assert M_order >= 1 "M_order must be ≥ 1 for β≠0 (need to differentiate)"

        # Build the capillary chain once at the highest required order
        # M3 = M+3+extra_orders. Lower-order series (M1 for ts_yuD / ts_G,
        # M_order for ts_C1 / ts_Gu) are obtained by `truncate_order` +
        # `deriv_series`.
        M1 = M_order + 1
        M3 = M_order + 3 + extra_orders

        pi_arb = Arb(pi; prec=prec)
        h_arb  = Arb(2; prec=prec) * pi_arb / Arb(N_grid; prec=prec)

        # ------ 6a: certified D and its inverse, at order M3 ------
        D_lower, ts_D3 = dmin_bound(y, N_grid, M3; prec=prec)
        if !Arblib.is_positive(D_lower)
            error("D is not certified strictly positive; cannot compute capillary terms.")
        end
        ts_invD3 = inv_series(ts_D3, D_lower)

        # ------ 6b: B = 1 + H[y_u] at order M3 ------
        Hyu_d = Vector{TrigPoly}(undef, M3 + 2)
        Hyu_d[1] = Hyu
        for m in 1:(M3 + 1)
            Hyu_d[m+1] = deriv(Hyu_d[m])
        end
        sup_bounds_B3 = Vector{Arb}(undef, M3 + 2)
        sup_bounds_B3[1] = Arb(1; prec=prec) + supnorm_bound(Hyu_d[1])
        for m in 1:(M3 + 1)
            sup_bounds_B3[m+1] = supnorm_bound(Hyu_d[m+1])
        end

        derivs_B3 = Matrix{Arb}(undef, N_grid, M3 + 1)
        Threads.@threads for j in 1:N_grid
            u_j = Arb(2; prec=prec) * pi_arb * Arb(j - 1; prec=prec) / Arb(N_grid; prec=prec)
            derivs_B3[j, 1] = Arb(1; prec=prec) + tp_eval(Hyu_d[1], u_j)
            for m in 1:M3
                derivs_B3[j, m+1] = tp_eval(Hyu_d[m+1], u_j)
            end
        end
        ts_B3 = TaylorSeries(derivs_B3, sup_bounds_B3, M3, N_grid, h_arb, prec)

        # ------ 6c: G = B/D at order M3; read Gpp/Gppp from its sup_bounds ------
        ts_G3 = ts_B3 * ts_invD3
        # ts_G3.sup_bounds has length M3+2 = M_order+5; entry [k+1] bounds G^{(k)}.
        # fft_hilbert_series(ts_Gu at M_order) needs:
        #   Gpp_bound  ≥ ‖(G_u)^{(M_order+2)}‖∞ = ‖G^{(M_order+3)}‖∞  → sup_bounds[M_order+4]
        #   Gppp_bound ≥ ‖(G_u)^{(M_order+3)}‖∞ = ‖G^{(M_order+4)}‖∞  → sup_bounds[M_order+5]
        # Tight algebraic bounds on ‖G^(M_order+3)‖∞ and ‖G^(M_order+4)‖∞
        # via tight_DG_sup_bounds (TrigPoly recurrence) — bypasses the
        # catastrophically loose Faà di Bruno chain inside ts_G3.
        # Use whichever is tighter between the two (algebraic recurrence vs
        # the (now-tightened) sqrt_series/inv_series chain).
        _, G_sup_tight = tight_DG_sup_bounds(y, M_order + 4, D_lower; prec=prec)
        Gpp_bound_chain  = ts_G3.sup_bounds[M_order + 4]
        Gppp_bound_chain = ts_G3.sup_bounds[M_order + 5]
        Gpp_bound_Gu  = Arblib.ubound(G_sup_tight[M_order + 4]) < Arblib.ubound(Gpp_bound_chain)  ? G_sup_tight[M_order + 4] : Gpp_bound_chain
        Gppp_bound_Gu = Arblib.ubound(G_sup_tight[M_order + 5]) < Arblib.ubound(Gppp_bound_chain) ? G_sup_tight[M_order + 5] : Gppp_bound_chain

        # ------ 6d: derive ts_C1 = ∂_u(y_u/D) and ts_Gu = ∂_u G at order M_order ------
        # Build y_u at M3 once, truncate as needed.
        ts_yu_M3   = trigpoly_to_taylor(yu, N_grid, M3; prec=prec)
        ts_yu_M1   = truncate_order(ts_yu_M3, M1)
        ts_invD_M1 = truncate_order(ts_invD3, M1)
        ts_yuD_M1  = ts_yu_M1 * ts_invD_M1
        ts_C1      = deriv_series(ts_yuD_M1)     # at order M_order

        ts_G_M1    = truncate_order(ts_G3, M1)
        ts_Gu      = deriv_series(ts_G_M1)        # at order M_order

        ts_HGu = fft_hilbert_series(ts_Gu, N_grid;
                                    Gpp_bound=Gpp_bound_Gu,
                                    Gppp_bound=Gppp_bound_Gu,
                                    prec=prec)   # H[∂_u(G)]

        # ------ 6d: add capillary contribution ------
        β_arb = Arb(β; prec=prec)
        ts_R  = ts_R + β_arb * ts_C1 - β_arb * ts_HGu

        # Expose for certificate emission.
        ts_G3_out          = ts_G3
        Gpp_bound_Gu_out   = Gpp_bound_Gu
        Gppp_bound_Gu_out  = Gppp_bound_Gu
        D_lower_out        = D_lower
    end

    # ------------------------------------------------------------------
    # Step 7: certified sup bound on ‖R‖_∞
    # ------------------------------------------------------------------
    bound = sup_bound(ts_R)

    return (
        bound          = bound,
        ts_R           = ts_R,
        ts_y           = ts_y,
        ts_yu          = ts_yu,
        ts_Hyu         = ts_Hyu,
        Gpp_bound_yyu  = Gpp_bound_yyu,
        Gppp_bound_yyu = Gppp_bound_yyu,
        ts_G3          = ts_G3_out,
        Gpp_bound_Gu   = Gpp_bound_Gu_out,
        Gppp_bound_Gu  = Gppp_bound_Gu_out,
        D_lower        = D_lower_out,
    )
end

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

export residual_bound
