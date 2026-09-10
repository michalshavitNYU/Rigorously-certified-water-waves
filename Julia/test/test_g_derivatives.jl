"""
test_g_derivatives.jl — Tests for GDerivatives.jl (Milestone M3).

Run with:
  julia --project=. test/test_g_derivatives.jl
from the BabenkoRigorous root.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Test
using Arblib
using BabenkoRigorous

using BabenkoRigorous: TrigPoly, hilbert, deriv, tp_eval, supnorm_bound, cosine_series
using BabenkoRigorous: TaylorSeries, sup_bound, inf_bound, dmin_bound
using BabenkoRigorous: compute_G_taylor

prec = 128

# ---------------------------------------------------------------------------
# Helper: evaluate G directly at an Arb point u_j using exact TrigPoly ops
# ---------------------------------------------------------------------------

function G_direct_arb(y::TrigPoly, u::Arb; p::Int=128)
    yu  = deriv(y)
    Hyu = hilbert(yu)
    A_j = tp_eval(yu, u)
    B_j = Arb(1; prec=p) + tp_eval(Hyu, u)
    D_j = sqrt(A_j^2 + B_j^2)
    return B_j / D_j
end

# Helper: evaluate G at a Float64 point (returns Float64)
function G_direct_f64(y::TrigPoly, u_f64::Float64)
    u = Arb(u_f64; prec=64)
    val = G_direct_arb(y, u; p=64)
    return Float64(Arblib.midpoint(Arb, val))
end

# ---------------------------------------------------------------------------
# Setup: y = 0.3*cos(u) + 0.15*cos(2u), M_grid=64, M_order=4
# ---------------------------------------------------------------------------

y_main = cosine_series([0.0, 0.3, 0.15]; prec=prec)
M_grid_main = 64
M_order_main = 4

pi_arb = Arb(pi; prec=prec)

println("Computing ts_G for T3.1-T3.5 (y=0.3cos+0.15cos2, M_grid=64, M_order=4)...")
ts_G_main = compute_G_taylor(y_main, M_grid_main, M_order_main; prec=prec)
println("  ts_G computed. sup_bound(ts_G) = ", Float64(Arblib.ubound(sup_bound(ts_G_main))))

# ---------------------------------------------------------------------------
# T3.1 — G^{(0)} matches direct evaluation
# ---------------------------------------------------------------------------

@testset "T3.1: G^(0) ball contains direct evaluation at higher prec" begin
    # Strengthened: use Arblib.contains against a HIGHER-precision (prec=512)
    # G_direct_arb. The high-prec ball is essentially a point (radius ~10^{-150})
    # so containment is a genuine rigor check on the prec=128 implementation,
    # not the weak `overlaps` symmetric overlap.
    prec_ref = 512
    violations = 0
    for j in 1:M_grid_main
        u_j = Arb(2; prec=prec_ref) * Arb(pi; prec=prec_ref) *
              Arb(j - 1; prec=prec_ref) / Arb(M_grid_main; prec=prec_ref)
        G_dir = G_direct_arb(y_main, u_j; p=prec_ref)
        if !Arblib.contains(ts_G_main.derivs[j, 1], G_dir)
            violations += 1
        end
    end
    println("  T3.1 violations: $violations / $M_grid_main")
    @test violations == 0
end

# ---------------------------------------------------------------------------
# T3.2 — G^{(1)} validated by analytic chain rule (independent of compute_G_taylor)
# ---------------------------------------------------------------------------
# Strengthened from Float64 midpoint comparison to rigorous Arb containment.
# Uses the closed-form G' = B'/D − B·DDp/D³ with
#     A = y_u,  A' = y_uu,  B = 1 + H[y_u],  B' = H[y_uu] = (H[y_u])',
#     D = √(A² + B²),  DDp = A·A' + B·B'.
# Each piece is evaluated at u_j as an Arb at prec=512. The resulting ball is
# essentially a point at that precision (~10^{-150}), so `Arblib.contains`
# against the prec=128 ts_G.derivs ball is a genuine rigor check on the
# compute_G_taylor pipeline — not subject to FD truncation or cancellation.
# ---------------------------------------------------------------------------

@testset "T3.2: G^(1) ball contains analytic chain-rule G' (prec=512)" begin
    prec_ref = 512
    yu_tp    = deriv(y_main)
    Hyu_tp   = hilbert(yu_tp)
    y_uu_tp  = deriv(yu_tp)
    H_yuu_tp = deriv(Hyu_tp)        # = hilbert(y_uu_tp) (H and ∂ commute)

    violations = 0
    for j in 1:M_grid_main
        u_j = Arb(2; prec=prec_ref) * Arb(pi; prec=prec_ref) *
              Arb(j - 1; prec=prec_ref) / Arb(M_grid_main; prec=prec_ref)
        A  = tp_eval(yu_tp,    u_j)
        Ap = tp_eval(y_uu_tp,  u_j)
        B  = Arb(1; prec=prec_ref) + tp_eval(Hyu_tp,   u_j)
        Bp = tp_eval(H_yuu_tp, u_j)
        D  = sqrt(A^2 + B^2)
        DDp = A*Ap + B*Bp
        G_prime_ref = Bp / D - B * DDp / D^3
        if !Arblib.contains(ts_G_main.derivs[j, 2], G_prime_ref)
            violations += 1
            if violations <= 3
                println("  T3.2 violation at j=$j")
            end
        end
    end
    println("  T3.2 violations: $violations / $M_grid_main")
    @test violations == 0
end

# ---------------------------------------------------------------------------
# T3.3 — G^{(2)} sanity check via FD on the analytic G' (high-prec Arb)
# ---------------------------------------------------------------------------
# We use central FD on the analytic G' (from T3.2's chain rule) rather than
# an explicit closed form for G'' (which is unwieldy). To avoid catastrophic
# cancellation at extreme h, we use h_fd ≈ 10^{-50}, near optimal for prec=512
# arithmetic. The FD truncation O(h²·G^(4)) ≈ 10^{-100} and cancellation
# error ≈ 10^{-100} are both far below the prec=128 ball radius.
# ---------------------------------------------------------------------------

@testset "T3.3: G^(2) ball contains FD of analytic G' (h≈1e-50, prec=512)" begin
    prec_ref = 512
    h_fd = Arb(10; prec=prec_ref)^(-50)
    two_h = Arb(2; prec=prec_ref) * h_fd

    yu_tp    = deriv(y_main)
    Hyu_tp   = hilbert(yu_tp)
    y_uu_tp  = deriv(yu_tp)
    H_yuu_tp = deriv(Hyu_tp)

    function G_prime_arb(u::Arb)
        A  = tp_eval(yu_tp,    u)
        Ap = tp_eval(y_uu_tp,  u)
        B  = Arb(1; prec=prec_ref) + tp_eval(Hyu_tp,   u)
        Bp = tp_eval(H_yuu_tp, u)
        D  = sqrt(A^2 + B^2)
        DDp = A*Ap + B*Bp
        return Bp / D - B * DDp / D^3
    end

    violations = 0
    for j in 1:M_grid_main
        u_j = Arb(2; prec=prec_ref) * Arb(pi; prec=prec_ref) *
              Arb(j - 1; prec=prec_ref) / Arb(M_grid_main; prec=prec_ref)
        Gp_plus  = G_prime_arb(u_j + h_fd)
        Gp_minus = G_prime_arb(u_j - h_fd)
        G_pp_ref = (Gp_plus - Gp_minus) / two_h
        if !Arblib.contains(ts_G_main.derivs[j, 3], G_pp_ref)
            violations += 1
            if violations <= 3
                println("  T3.3 violation at j=$j")
            end
        end
    end
    println("  T3.3 violations: $violations / $M_grid_main")
    @test violations == 0
end

# ---------------------------------------------------------------------------
# T3.4 — Analytical case y = eps*cos(u), eps=0.1
# ---------------------------------------------------------------------------

@testset "T3.4: Analytical G for y=0.1*cos(u)" begin
    eps = 0.1
    y_eps = cosine_series([0.0, eps]; prec=prec)
    M_grid_eps = 128
    M_order_eps = 4

    println("  Computing ts_G for y=0.1*cos(u), M_grid=128...")
    ts_G_eps = compute_G_taylor(y_eps, M_grid_eps, M_order_eps; prec=prec)
    println("  Done.")

    # Reference computations use prec=512 (higher than the prec=128 test ball)
    # so `Arblib.contains` is a genuine rigor check, not just overlap.
    prec_ref = 512
    pi_ref = Arb(pi; prec=prec_ref)

    violations = 0
    for j in 1:M_grid_eps
        u_j = Arb(2; prec=prec_ref) * pi_ref *
              Arb(j - 1; prec=prec_ref) / Arb(M_grid_eps; prec=prec_ref)
        G_dir = G_direct_arb(y_eps, u_j; p=prec_ref)
        if !Arblib.contains(ts_G_eps.derivs[j, 1], G_dir)
            violations += 1
        end
    end
    println("  T3.4 violations (all grid pts): $violations / $M_grid_eps")
    @test violations == 0

    # Special points: find nearest grid points to 0, pi/2, pi
    j0 = 1  # u_0 = 0
    G_dir_0 = G_direct_arb(y_eps, Arb(0; prec=prec_ref); p=prec_ref)
    println("  T3.4 at u=0: G_ts=$(Float64(Arblib.midpoint(Arb, ts_G_eps.derivs[j0,1]))), G_direct=$(Float64(Arblib.midpoint(Arb, G_dir_0)))")
    @test Arblib.contains(ts_G_eps.derivs[j0, 1], G_dir_0)

    j_pi = 65
    u_pi = Arb(2; prec=prec_ref) * pi_ref * Arb(64; prec=prec_ref) / Arb(M_grid_eps; prec=prec_ref)
    G_dir_pi = G_direct_arb(y_eps, u_pi; p=prec_ref)
    println("  T3.4 at u~pi: G_ts=$(Float64(Arblib.midpoint(Arb, ts_G_eps.derivs[j_pi,1]))), G_direct=$(Float64(Arblib.midpoint(Arb, G_dir_pi)))")
    @test Arblib.contains(ts_G_eps.derivs[j_pi, 1], G_dir_pi)

    j_half = 33
    u_half = Arb(2; prec=prec_ref) * pi_ref * Arb(32; prec=prec_ref) / Arb(M_grid_eps; prec=prec_ref)
    G_dir_half = G_direct_arb(y_eps, u_half; p=prec_ref)
    exact_half = BigFloat(1) / sqrt(BigFloat(1) + BigFloat(eps)^2)
    println("  T3.4 at u~pi/2: G_ts=$(Float64(Arblib.midpoint(Arb, ts_G_eps.derivs[j_half,1]))), G_direct=$(Float64(Arblib.midpoint(Arb, G_dir_half))), exact=$(Float64(exact_half))")
    @test Arblib.contains(ts_G_eps.derivs[j_half, 1], G_dir_half)
end

# ---------------------------------------------------------------------------
# T3.5 — sup_bound on ts_G is finite and positive
# ---------------------------------------------------------------------------

@testset "T3.5: sup_bound(ts_G) is finite and positive" begin
    sb = sup_bound(ts_G_main)
    sb_ub = Float64(Arblib.ubound(sb))
    println("  T3.5: sup_bound(ts_G) upper bound = $sb_ub")
    @test Arblib.is_positive(sb)
    @test sb_ub < 1e6
end

println("\nAll M3 tests complete.")
