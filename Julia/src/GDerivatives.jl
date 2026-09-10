"""
GDerivatives.jl — Certified TaylorSeries for G(u) = (1 + H[y_u](u)) / D(u).

G appears in the capillary terms of the Babenko water-wave equation. This module
computes a certified TaylorSeries for G using the M2 primitives: dmin_bound,
inv_series, and the TaylorSeries multiplication operator.

Algorithm:
  1. dmin_bound(y, M_grid, M_order) -> (D_lower, ts_D)
  2. Build ts_B for B(u) = 1 + H[y_u](u) directly from TrigPoly derivatives
  3. ts_invD = inv_series(ts_D, D_lower)
  4. ts_G = ts_B * ts_invD
"""

using Arblib

"""
    compute_G_taylor(y::TrigPoly, M_grid::Int, M_order::Int; prec::Int=128) -> TaylorSeries

Compute a certified TaylorSeries for G(u) = (1 + H[y_u](u)) / D(u), where
D(u) = sqrt(y_u(u)^2 + (1 + H[y_u](u))^2).

Arguments:
  y       :: TrigPoly  — the conformal parametrisation y(u)
  M_grid  :: Int       — number of equispaced grid points
  M_order :: Int       — Taylor order (number of derivatives stored)
  prec    :: Int       — Arb working precision in bits (default 128)

Returns a TaylorSeries ts_G with:
  ts_G.derivs[j, m+1]     = G^{(m)}(u_j)   for j=1..M_grid, m=0..M_order
  ts_G.sup_bounds[m+1]    >= sup |G^{(m)}(u)|  for m=0..M_order+1

Throws an error if D is not certified to be strictly positive.
"""
function compute_G_taylor(y::TrigPoly, M_grid::Int, M_order::Int;
                          prec::Int=128)::TaylorSeries
    p = prec

    # ------------------------------------------------------------------
    # Step 1: certified lower bound and TaylorSeries for D(u)
    # ------------------------------------------------------------------
    D_lower, ts_D = dmin_bound(y, M_grid, M_order; prec=p)

    if !Arblib.is_positive(D_lower)
        error("D is not certified positive; cannot compute G.")
    end

    # ------------------------------------------------------------------
    # Step 2: build TaylorSeries ts_B for B(u) = 1 + H[y_u](u)
    # ------------------------------------------------------------------
    # Exact TrigPoly arithmetic
    yu  = deriv(y)
    Hyu = hilbert(yu)

    # Precompute iterated derivatives of Hyu: Hyu_d[m+1] = d^m(Hyu)/du^m
    # We need m = 0..M_order+1 for the F_bound (needs the (M_order+1)-th deriv).
    Hyu_d = Vector{TrigPoly}(undef, M_order + 2)
    Hyu_d[1] = Hyu
    for m in 1:(M_order + 1)
        Hyu_d[m+1] = deriv(Hyu_d[m])
    end

    # Full sup_bounds vector for B = 1 + Hyu: the constant +1 shifts only the m=0 supnorm.
    sup_bounds_B = Vector{Arb}(undef, M_order + 2)
    sup_bounds_B[1] = Arb(1; prec=p) + supnorm_bound(Hyu_d[1])
    for m in 1:(M_order + 1)
        sup_bounds_B[m+1] = supnorm_bound(Hyu_d[m+1])
    end

    pi_arb = Arb(pi; prec=p)
    h = Arb(2; prec=p) * pi_arb / Arb(M_grid; prec=p)

    derivs_B = Matrix{Arb}(undef, M_grid, M_order + 1)
    for j in 1:M_grid
        u_j = Arb(2; prec=p) * pi_arb * Arb(j - 1; prec=p) / Arb(M_grid; prec=p)
        derivs_B[j, 1] = Arb(1; prec=p) + tp_eval(Hyu_d[1], u_j)
        for m in 1:M_order
            derivs_B[j, m+1] = tp_eval(Hyu_d[m+1], u_j)
        end
    end

    ts_B = TaylorSeries(derivs_B, sup_bounds_B, M_order, M_grid, h, p)

    # ------------------------------------------------------------------
    # Step 3: ts_invD = 1 / D
    # ------------------------------------------------------------------
    ts_invD = inv_series(ts_D, D_lower)

    # ------------------------------------------------------------------
    # Step 4: ts_G = B * (1/D)
    # ------------------------------------------------------------------
    ts_G = ts_B * ts_invD

    return ts_G
end

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

export compute_G_taylor
