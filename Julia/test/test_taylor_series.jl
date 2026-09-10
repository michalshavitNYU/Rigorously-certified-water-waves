"""
test_taylor_series.jl — M3 milestone tests for TaylorSeries.jl

Tests:
  T2.1 — Taylor expansion contains true sin(u) value for all test points
  T2.2 — sup_bound(ts_sin) ≥ 1.0 (true ‖sin‖_∞)
  T2.3 — inf_bound for D on y = 0.1·cos(u) (true D_min = 0.9)
  T2.4 — Taylor remainder decreases with M_order
  T2.5 — deriv_series: cos' = -sin
  T2.6 — product: sin * cos agrees with sin(2u)/2
  T2.7 — inv_series: 1/(2+cos(u))
"""

using Test
using Arblib
import BabenkoRigorous

using BabenkoRigorous: TrigPoly, degree, hilbert, deriv, tp_eval, supnorm_bound,
                       cosine_series, add_const, trigpoly_equal
using BabenkoRigorous: TaylorSeries, sup_bound, inf_bound, deriv_series,
                       inv_series, sqrt_series, dmin_bound

# ---------------------------------------------------------------------------
# Helper: build TaylorSeries for a simple function analytically
# ---------------------------------------------------------------------------

"""
Build TaylorSeries for sin(u) at M_grid equispaced points with Taylor order M_order.
sin^{(m)}(u) cycles: sin, cos, -sin, -cos, sin, ...
"""
function make_sin_series(M_grid::Int, M_order::Int; prec::Int=128)::TaylorSeries
    p = prec
    pi_arb = Arb(pi; prec=p)
    h = Arb(2; prec=p) * pi_arb / Arb(M_grid; prec=p)

    derivs = Matrix{Arb}(undef, M_grid, M_order + 1)
    for j in 1:M_grid
        u_j = Arb(2; prec=p) * pi_arb * Arb(j - 1; prec=p) / Arb(M_grid; prec=p)
        for m in 0:M_order
            # sin^{(m)}(u): cycles sin, cos, -sin, -cos
            r = m % 4
            if r == 0
                derivs[j, m+1] = sin(u_j)
            elseif r == 1
                derivs[j, m+1] = cos(u_j)
            elseif r == 2
                derivs[j, m+1] = -sin(u_j)
            else  # r == 3
                derivs[j, m+1] = -cos(u_j)
            end
        end
    end
    # All derivatives of sin are bounded by 1 in absolute value
    sup_bounds = [Arb(1; prec=p) for _ in 0:(M_order + 1)]
    return TaylorSeries(derivs, sup_bounds, M_order, M_grid, h, p)
end

"""
Build TaylorSeries for cos(u) at M_grid equispaced points with Taylor order M_order.
cos^{(m)}(u) cycles: cos, -sin, -cos, sin, cos, ...
"""
function make_cos_series(M_grid::Int, M_order::Int; prec::Int=128)::TaylorSeries
    p = prec
    pi_arb = Arb(pi; prec=p)
    h = Arb(2; prec=p) * pi_arb / Arb(M_grid; prec=p)

    derivs = Matrix{Arb}(undef, M_grid, M_order + 1)
    for j in 1:M_grid
        u_j = Arb(2; prec=p) * pi_arb * Arb(j - 1; prec=p) / Arb(M_grid; prec=p)
        for m in 0:M_order
            # cos^{(m)}(u): cycles cos, -sin, -cos, sin
            r = m % 4
            if r == 0
                derivs[j, m+1] = cos(u_j)
            elseif r == 1
                derivs[j, m+1] = -sin(u_j)
            elseif r == 2
                derivs[j, m+1] = -cos(u_j)
            else  # r == 3
                derivs[j, m+1] = sin(u_j)
            end
        end
    end
    sup_bounds = [Arb(1; prec=p) for _ in 0:(M_order + 1)]
    return TaylorSeries(derivs, sup_bounds, M_order, M_grid, h, p)
end

# ---------------------------------------------------------------------------
# T2.1 — Taylor expansion contains true sin(u) value
# ---------------------------------------------------------------------------

@testset "T2.1 — Taylor expansion contains true value" begin
    prec = 128
    M_grid = 32
    M_order = 4
    ts = make_sin_series(M_grid, M_order; prec=prec)

    pi_arb = Arb(pi; prec=prec)
    h = ts.h
    M = M_order

    # Precompute h^{M+1}/(M+1)! for remainder
    hpow = h^(M + 1)
    fact_Mp1 = Arb(factorial(M + 1); prec=prec)
    remainder = ts.sup_bounds[end] * hpow / fact_Mp1

    # Test at 200 uniformly spaced points
    n_test = 200
    violations = 0
    for i in 0:(n_test - 1)
        u_float = 2π * i / n_test
        u = Arb(u_float; prec=prec)

        # Find nearest grid point to the left: j such that u_j <= u < u_{j+1}
        # u_j = 2π*(j-1)/M_grid, so j = floor(u * M_grid / (2π)) + 1
        j = clamp(floor(Int, u_float * M_grid / (2π)) + 1, 1, M_grid)
        u_j = Arb(2; prec=prec) * pi_arb * Arb(j - 1; prec=prec) / Arb(M_grid; prec=prec)
        du = u - u_j

        # Taylor sum S = Σ_{m=0}^{M} f^{(m)}(u_j)/m! * (u-u_j)^m
        S = Arb(0; prec=prec)
        du_pow = Arb(1; prec=prec)
        fact_m = Arb(1; prec=prec)
        for m in 0:M
            if m > 0
                du_pow = du_pow * du
                fact_m = fact_m * Arb(m; prec=prec)
            end
            S = S + ts.derivs[j, m+1] * du_pow / fact_m
        end

        # True value
        true_val = sin(u)

        # Check |sin(u) - S| ≤ remainder
        diff = abs(true_val - S)
        if !Arblib.le(diff, remainder)
            violations += 1
        end
    end
    @test violations == 0
end

# ---------------------------------------------------------------------------
# T2.2 — sup_bound ≥ true maximum of sin
# ---------------------------------------------------------------------------

@testset "T2.2 — sup_bound ≥ true sup(sin) = 1" begin
    prec = 128
    M_grid = 32
    for M_order in [2, 4, 6]
        ts = make_sin_series(M_grid, M_order; prec=prec)
        sb = sup_bound(ts)
        # True ‖sin‖_∞ = 1, so sup_bound must be ≥ 1
        one_arb = Arb(1; prec=prec)
        @test Arblib.ge(sb, one_arb)
        # Also: sup_bound should be finite and not astronomical
        @test Float64(Arblib.ubound(sb)) < 10.0
    end
end

# ---------------------------------------------------------------------------
# T2.3 — inf_bound for D on y = 0.1·cos(u)
# ---------------------------------------------------------------------------

@testset "T2.3 — inf_bound for D on y = 0.1·cos(u)" begin
    prec = 256
    # y = 0.1*cos(u): a[1]=0, a[2]=0.1 (DC stored as 2*const, cos(u) coeff = 0.1)
    # Actually: TrigPoly convention: a[1] = DC (eval uses a[1]/2), a[2] = cos(1*u) coeff
    # So for y = 0.1*cos(u): a = [0.0, 0.1], b = [0.0, 0.0]
    y = TrigPoly([0.0, 0.1], [0.0, 0.0]; prec=prec)

    M_grid = 256
    M_order = 4

    D_lower, ts_D = dmin_bound(y, M_grid, M_order; prec=prec)

    D_lower_f64 = Float64(Arblib.lbound(D_lower))

    # True D_min for y=0.1*cos(u):
    # y_u = -0.1*sin(u), H[y_u] = 0.1*cos(u)
    # D = sqrt(0.01*sin^2(u) + (1+0.1*cos(u))^2)
    # min over u: at u=π, D = sqrt(0 + (1-0.1)^2) = 0.9
    @test D_lower_f64 > 0.88    # conservative tolerance: lower bound is close to 0.9
    @test D_lower_f64 < 0.9     # it IS a lower bound, so strictly below true min
    @test D_lower_f64 > 0.0     # positive
end

# ---------------------------------------------------------------------------
# T2.4 — Taylor remainder decreases with M_order
# ---------------------------------------------------------------------------

@testset "T2.4 — sup_bound decreases with M_order" begin
    prec = 128
    M_grid = 32

    sb_vals = Float64[]
    for M_order in [1, 2, 3, 4, 5]
        ts = make_sin_series(M_grid, M_order; prec=prec)
        sb = sup_bound(ts)
        push!(sb_vals, Float64(Arblib.ubound(sb)))
    end

    # sup_bound should decrease as M_order increases (higher-order → tighter bound)
    # At minimum, M_order=5 bound should be strictly less than M_order=1 bound.
    @test sb_vals[5] < sb_vals[1]

    # All bounds should be ≥ 1 (the true sup norm)
    @test all(v >= 1.0 for v in sb_vals)

    # Check monotone decrease (or at worst non-increasing) — allow small non-monotonicity
    # due to grid effects, but the trend should be clear from order 1 to 5.
    @test sb_vals[5] <= sb_vals[3]   # higher order is better
    @test sb_vals[3] <= sb_vals[1]   # monotone on coarser check
end

# ---------------------------------------------------------------------------
# T2.5 — deriv_series: cos' = -sin
# ---------------------------------------------------------------------------

@testset "T2.5 — deriv_series: cos' = -sin" begin
    prec = 128
    M_grid = 32
    M_order = 5  # need at least 1 to differentiate

    ts_cos = make_cos_series(M_grid, M_order; prec=prec)
    ts_cos_prime = deriv_series(ts_cos)   # should be -sin

    # Check: ts_cos_prime.derivs[j, m+1] should equal (-sin)^{(m)}(u_j)
    # (-sin)^{(m)} cycles: -sin, -cos, sin, cos, -sin, ...
    pi_arb = Arb(pi; prec=prec)
    for j in 1:M_grid
        u_j = Arb(2; prec=prec) * pi_arb * Arb(j - 1; prec=prec) / Arb(M_grid; prec=prec)
        for m in 0:(M_order - 1)
            r = m % 4
            if r == 0
                expected = -sin(u_j)
            elseif r == 1
                expected = -cos(u_j)
            elseif r == 2
                expected = sin(u_j)
            else
                expected = cos(u_j)
            end
            diff = ts_cos_prime.derivs[j, m+1] - expected
            @test Arblib.contains_zero(diff)
        end
    end

    # Check M_order decreased by 1
    @test ts_cos_prime.M_order == M_order - 1
end

# ---------------------------------------------------------------------------
# T2.6 — product: ts_sin * ts_cos agrees with ts for sin(2u)/2
# ---------------------------------------------------------------------------

@testset "T2.6 — product ts_sin * ts_cos ≈ sin(2u)/2" begin
    prec = 128
    M_grid = 32
    M_order = 4

    ts_sin = make_sin_series(M_grid, M_order; prec=prec)
    ts_cos = make_cos_series(M_grid, M_order; prec=prec)
    ts_prod = ts_sin * ts_cos

    # sin(u)*cos(u) = sin(2u)/2
    # (sin(2u)/2)^{(m)} at u_j: sin(2u)/2, so m-th deriv = 2^{m-1} sin^{(m)}(2u)
    # More directly: (d^m/du^m)[sin(2u)/2] = 2^m/2 * [sin or cos cycle for 2u]
    # At each grid point, compare product derivs to expected sin(2u)/2 derivs.
    pi_arb = Arb(pi; prec=prec)
    tol = Arb(1e-10; prec=prec)   # loose tolerance (product bound is conservative)

    for j in 1:M_grid
        u_j = Arb(2; prec=prec) * pi_arb * Arb(j - 1; prec=prec) / Arb(M_grid; prec=prec)
        two_uj = Arb(2; prec=prec) * u_j

        for m in 0:M_order
            # (sin(2u)/2)^{(m)} = 2^{m-1} * sin^{(m)}(2u)  for m≥1, and sin(2u)/2 for m=0
            # sin^{(m)}(2u) cycles: sin(2u), cos(2u), -sin(2u), -cos(2u)
            two_pow = Arb(2; prec=prec)^m / Arb(2; prec=prec)
            r = m % 4
            if r == 0
                expected = sin(two_uj) * Arb(2; prec=prec)^m / Arb(2; prec=prec)
            elseif r == 1
                expected = cos(two_uj) * Arb(2; prec=prec)^m / Arb(2; prec=prec)
            elseif r == 2
                expected = -sin(two_uj) * Arb(2; prec=prec)^m / Arb(2; prec=prec)
            else
                expected = -cos(two_uj) * Arb(2; prec=prec)^m / Arb(2; prec=prec)
            end

            diff = ts_prod.derivs[j, m+1] - expected
            # The product is computed exactly from Arb sin/cos, so the difference
            # should be within Arb rounding (ball arithmetic, essentially zero).
            @test Arblib.contains_zero(diff)
        end
    end
end

# ---------------------------------------------------------------------------
# T2.7 — inv_series: 1/(2 + cos(u))
# ---------------------------------------------------------------------------

@testset "T2.7 — inv_series: 1/(2+cos(u))" begin
    prec = 128
    M_grid = 32
    M_order = 4
    p = prec

    # Build TaylorSeries for f(u) = 2 + cos(u)
    # f^{(0)} = 2 + cos(u), f^{(1)} = -sin(u), f^{(2)} = -cos(u), f^{(3)} = sin(u), ...
    # cos^{(m)} cycles: cos, -sin, -cos, sin
    pi_arb = Arb(pi; prec=p)
    h = Arb(2; prec=p) * pi_arb / Arb(M_grid; prec=p)

    derivs_f = Matrix{Arb}(undef, M_grid, M_order + 1)
    for j in 1:M_grid
        u_j = Arb(2; prec=p) * pi_arb * Arb(j - 1; prec=p) / Arb(M_grid; prec=p)
        for m in 0:M_order
            r = m % 4
            if m == 0
                derivs_f[j, 1] = Arb(2; prec=p) + cos(u_j)
            else
                if r == 1
                    derivs_f[j, m+1] = -sin(u_j)
                elseif r == 2
                    derivs_f[j, m+1] = -cos(u_j)
                elseif r == 3
                    derivs_f[j, m+1] = sin(u_j)
                else  # r == 0
                    derivs_f[j, m+1] = cos(u_j)
                end
            end
        end
    end
    # f = 2 + cos(u): ‖f‖_∞ = 3, ‖f^{(m)}‖_∞ = 1 for m ≥ 1 (sinusoidal derivatives).
    sup_bounds_f = Vector{Arb}(undef, M_order + 2)
    sup_bounds_f[1] = Arb(3; prec=p)
    for m in 1:(M_order + 1)
        sup_bounds_f[m+1] = Arb(1; prec=p)
    end
    ts_f = TaylorSeries(derivs_f, sup_bounds_f, M_order, M_grid, h, p)

    # f(u) = 2 + cos(u) ≥ 1 for all u, so d_lower = 1
    d_lower = Arb(1; prec=p)
    ts_inv = inv_series(ts_f, d_lower)

    # At each grid point, check that the 0-th derivative of inv_series is close to
    # 1/(2+cos(u_j))
    for j in 1:M_grid
        u_j = Arb(2; prec=p) * pi_arb * Arb(j - 1; prec=p) / Arb(M_grid; prec=p)
        expected = Arb(1; prec=p) / (Arb(2; prec=p) + cos(u_j))
        got = ts_inv.derivs[j, 1]
        diff = got - expected
        # Should be exactly equal (both computed from Arb arithmetic)
        @test Arblib.contains_zero(diff)
    end

    # Also check sup_bound and inf_bound are in the right range:
    # 1/(2+cos) ∈ [1/3, 1] on [0,2π]
    sb = sup_bound(ts_inv)
    ib = inf_bound(ts_inv)
    @test Float64(Arblib.ubound(sb)) > 1.0/3.0
    @test Float64(Arblib.lbound(ib)) > 0.0   # positive lower bound
end

# ---------------------------------------------------------------------------
# T2.8 — Product F_bound rigor (exact Leibniz on sup_bounds vectors)
# ---------------------------------------------------------------------------
# Verify that the new exact-Leibniz formula in `Base.:*` produces a sup_bounds
# vector that is a genuine upper bound on the true derivative supnorms.
# Test case: sin(u) · sin(u) = sin²(u) = (1 − cos(2u))/2. Then
#   ‖(sin²)^{(m)}‖_∞  =  1       for m = 0
#                   =  2^{m-1}  for m ≥ 1.
# The new formula gives Σ_k C(m,k) · ‖sin^{(k)}‖_∞ · ‖sin^{(m-k)}‖_∞ = 2^m, so
# the certified bound is at most 2× looser than the truth — always rigorous.
# This testset would have failed the previous `2^{M+1}(F1·S2 + S1·F2)` formula
# under high-frequency-input stress (see audit C4); the exact Leibniz cannot.
# ---------------------------------------------------------------------------

@testset "T2.8 — Product sup_bounds rigor (sin · sin = sin²)" begin
    prec = 128
    M_grid = 64
    for M_order in [2, 4, 6]
        ts_a = make_sin_series(M_grid, M_order; prec=prec)
        ts_b = make_sin_series(M_grid, M_order; prec=prec)
        ts_prod = ts_a * ts_b

        # True supnorm of (sin²)^{(m)}:
        # m=0: 1.  m ≥ 1: 2^{m-1}.
        for m in 0:(M_order + 1)
            true_sup = m == 0 ? Arb(1; prec=prec) : Arb(2.0^(m - 1); prec=prec)
            @test Arblib.le(true_sup, ts_prod.sup_bounds[m + 1])
        end
    end
end

# ---------------------------------------------------------------------------
# T2.9 — inv_series F_bound responds to input frequency (rigorous Faà di Bruno)
# ---------------------------------------------------------------------------
# Previous buggy formula `(M+1)! · (S/d)^{M+1} / d` used only S and d, so it
# was independent of the input function's high-frequency content. The new
# Faà di Bruno formula uses the full sup_bounds vector of `ts`, so for
# f = 2 + cos(N·u) the bound should grow with N. Concretely:
#   ‖f^{(k)}‖_∞ = N^k for k ≥ 1, so the Bell polynomial composition predicts
#   sup_bounds[end] for 1/f ≈ const · N^{M+1}, growing rapidly with N.
# This testset exercises the exact path that produced the audit's
# counter-example (formula off by 10+ orders of magnitude for high N).
# ---------------------------------------------------------------------------

@testset "T2.9 — inv_series sup_bounds grows with input frequency" begin
    prec = 128
    M_grid = 128
    M_order = 2

    function make_2_plus_cosN(N::Int)
        p = prec
        pi_arb = Arb(pi; prec=p)
        h = Arb(2; prec=p) * pi_arb / Arb(M_grid; prec=p)

        derivs = Matrix{Arb}(undef, M_grid, M_order + 1)
        for j in 1:M_grid
            u_j = Arb(2; prec=p) * pi_arb * Arb(j - 1; prec=p) / Arb(M_grid; prec=p)
            Nu = Arb(N; prec=p) * u_j
            for m in 0:M_order
                r = m % 4
                if m == 0
                    derivs[j, 1] = Arb(2; prec=p) + cos(Nu)
                else
                    Nk = Arb(N; prec=p)^m
                    if r == 1
                        derivs[j, m+1] = -Nk * sin(Nu)
                    elseif r == 2
                        derivs[j, m+1] = -Nk * cos(Nu)
                    elseif r == 3
                        derivs[j, m+1] = Nk * sin(Nu)
                    else  # r == 0
                        derivs[j, m+1] = Nk * cos(Nu)
                    end
                end
            end
        end
        # sup_bounds: ‖f‖_∞ = 3 (when cos=1); ‖f^{(m)}‖_∞ = N^m for m ≥ 1.
        sup_bounds = Vector{Arb}(undef, M_order + 2)
        sup_bounds[1] = Arb(3; prec=prec)
        for m in 1:(M_order + 1)
            sup_bounds[m+1] = Arb(N; prec=prec)^m
        end
        return TaylorSeries(derivs, sup_bounds, M_order, M_grid, h, prec)
    end

    d_lower = Arb(1; prec=prec)   # f = 2 + cos(Nu) ≥ 1
    bounds = Float64[]
    for N in [1, 3, 10]
        ts_f = make_2_plus_cosN(N)
        ts_inv = inv_series(ts_f, d_lower)
        push!(bounds, Float64(Arblib.ubound(ts_inv.sup_bounds[end])))
        println("  T2.9: N=$N, inv_series sup_bounds[end] = $(bounds[end])")
    end

    # Bound must increase with N (input frequency); ratio should be roughly
    # at least N^(M+1) factor. We assert just monotonicity here to keep the
    # test robust across precision changes.
    @test bounds[2] > bounds[1]
    @test bounds[3] > bounds[2]
    # For N=10 vs N=1: ratio should be ≥ 100 (much weaker than N^{M+1}=1000).
    @test bounds[3] / bounds[1] > 100.0
end

# ---------------------------------------------------------------------------
# T2.10 — sqrt_series F_bound responds to input frequency
# ---------------------------------------------------------------------------
# Same logic as T2.9 but for sqrt_series. Previous formula
# `(M+1)! · S^{M/2} / (2·d^{M+1})` was dimensionally wrong and ignored F,
# producing bounds 10× too small (audit C2). The new Faà di Bruno formula
# uses the full sup_bounds vector and should grow with input frequency.
# ---------------------------------------------------------------------------

@testset "T2.10 — sqrt_series sup_bounds grows with input frequency" begin
    prec = 128
    M_grid = 128
    M_order = 2

    function make_2_plus_cosN(N::Int)
        p = prec
        pi_arb = Arb(pi; prec=p)
        h = Arb(2; prec=p) * pi_arb / Arb(M_grid; prec=p)

        derivs = Matrix{Arb}(undef, M_grid, M_order + 1)
        for j in 1:M_grid
            u_j = Arb(2; prec=p) * pi_arb * Arb(j - 1; prec=p) / Arb(M_grid; prec=p)
            Nu = Arb(N; prec=p) * u_j
            for m in 0:M_order
                r = m % 4
                if m == 0
                    derivs[j, 1] = Arb(2; prec=p) + cos(Nu)
                else
                    Nk = Arb(N; prec=p)^m
                    if r == 1
                        derivs[j, m+1] = -Nk * sin(Nu)
                    elseif r == 2
                        derivs[j, m+1] = -Nk * cos(Nu)
                    elseif r == 3
                        derivs[j, m+1] = Nk * sin(Nu)
                    else
                        derivs[j, m+1] = Nk * cos(Nu)
                    end
                end
            end
        end
        sup_bounds = Vector{Arb}(undef, M_order + 2)
        sup_bounds[1] = Arb(3; prec=prec)
        for m in 1:(M_order + 1)
            sup_bounds[m+1] = Arb(N; prec=prec)^m
        end
        return TaylorSeries(derivs, sup_bounds, M_order, M_grid, h, prec)
    end

    d_lower = Arb(1; prec=prec)   # √(2+cos Nu) ≥ √1 = 1
    bounds = Float64[]
    for N in [1, 3, 10]
        ts_f = make_2_plus_cosN(N)
        ts_sqrt = sqrt_series(ts_f, d_lower)
        push!(bounds, Float64(Arblib.ubound(ts_sqrt.sup_bounds[end])))
        println("  T2.10: N=$N, sqrt_series sup_bounds[end] = $(bounds[end])")
    end

    @test bounds[2] > bounds[1]
    @test bounds[3] > bounds[2]
    @test bounds[3] / bounds[1] > 100.0
end

# ---------------------------------------------------------------------------
# T2.11 — Direct test of _faa_di_bruno_derivs (Bell-polynomial composition)
# ---------------------------------------------------------------------------
# Closes the audit gap: _faa_di_bruno_derivs was previously exercised only
# indirectly through inv_series and sqrt_series. This test feeds known
# analytic inputs and checks the exact raw-derivative output.
#
# Three cases:
# (a) g(x) = x, C(z) = z+1.  (g∘C)(z) = z+1; derivs at 0: [1, 1, 0, 0].
# (b) g(x) = x², C(z) = z.   (g∘C)(z) = z²; derivs at 0: [0, 0, 2, 0].
# (c) g(x) = exp(x), C(z) = z²+z.  (g∘C)(z) = exp(z²+z);
#     derivs at 0: [1, 1, 3, 7]  (verified analytically below).

const _faa_di_bruno_derivs = BabenkoRigorous._faa_di_bruno_derivs

@testset "T2.11 — _faa_di_bruno_derivs against analytic compositions" begin
    prec = 128
    arb(x) = Arb(x; prec=prec)

    # Case (a): g(x) = x, C(z) = z+1
    g_d_a = [arb(1), arb(1), arb(0), arb(0)]
    C_d_a = [arb(1), arb(1), arb(0), arb(0)]
    out_a = _faa_di_bruno_derivs(g_d_a, C_d_a, 3, prec)
    @test Arblib.overlaps(out_a[1], arb(1))
    @test Arblib.overlaps(out_a[2], arb(1))
    @test Arblib.contains_zero(out_a[3])
    @test Arblib.contains_zero(out_a[4])

    # Case (b): g(x) = x², C(z) = z
    g_d_b = [arb(0), arb(0), arb(2), arb(0)]
    C_d_b = [arb(0), arb(1), arb(0), arb(0)]
    out_b = _faa_di_bruno_derivs(g_d_b, C_d_b, 3, prec)
    @test Arblib.contains_zero(out_b[1])
    @test Arblib.contains_zero(out_b[2])
    @test Arblib.overlaps(out_b[3], arb(2))
    @test Arblib.contains_zero(out_b[4])

    # Case (c): g(x) = exp(x), C(z) = z²+z
    # f(z) = exp(z²+z): f'(z) = (2z+1)exp; f''(z) = [2+(2z+1)²]exp;
    # f'''(z) = [4(2z+1) + (2z+1)(2+(2z+1)²)] exp.
    # At z=0: f=1, f'=1, f''=3, f'''=7.
    g_d_c = [arb(1), arb(1), arb(1), arb(1)]   # exp^{(k)}(0) = 1 for all k
    C_d_c = [arb(0), arb(1), arb(2), arb(0)]   # C(0)=0, C'(0)=1, C''(0)=2, C'''(0)=0
    out_c = _faa_di_bruno_derivs(g_d_c, C_d_c, 3, prec)
    @test Arblib.overlaps(out_c[1], arb(1))
    @test Arblib.overlaps(out_c[2], arb(1))
    @test Arblib.overlaps(out_c[3], arb(3))
    @test Arblib.overlaps(out_c[4], arb(7))
    println("  T2.11 case (c): out = $(Float64.(Arblib.midpoint.(Arb, out_c))) (expect [1, 1, 3, 7])")
end

# ---------------------------------------------------------------------------
# T2.12 — dmin_bound throws on degenerate y (D not certified positive)
# ---------------------------------------------------------------------------
# Closes the audit gap: the error path at TaylorSeries.jl:625 was not exercised.
# For y = ε·cos(u) with ε close to 1, D(u) = √(ε²sin² + (1+εcos)²) can be
# close to zero near u = π (where (1+εcos u) → 1 - ε ≈ 0).

@testset "T2.12 — dmin_bound throws when D not certified positive" begin
    prec = 128
    # Large-amplitude y that makes D very small at u=π
    y_coeffs = [0.0, 0.99]   # y = 0.99 cos(u), so D_min ≈ 0.01
    y = cosine_series(y_coeffs; prec=prec)

    # At small M_grid the Lipschitz subtraction may push d_lower_init below 0,
    # which is the throw path. At larger M_grid h shrinks and it survives.
    # We try a small grid that's known to fail the positivity test.
    @test_throws ErrorException dmin_bound(y, 8, 2; prec=prec)
end

# ---------------------------------------------------------------------------
# T2.13 — M_order = 0 edge case (smallest legal)
# ---------------------------------------------------------------------------
# The new sup_bounds-vector arithmetic must work at M_order = 0.
# sup_bounds has length 2: bounds for f and f'.
# `deriv_series` should reject (asserts M_order ≥ 1); `*`, `+`, `inv_series`,
# `sqrt_series` should still work.

@testset "T2.13 — M_order=0 edge case" begin
    prec = 128
    M_grid = 8
    pi_arb = Arb(pi; prec=prec)
    h = Arb(2; prec=prec) * pi_arb / Arb(M_grid; prec=prec)

    # ts_sin0: f(u) = sin(u), M_order = 0 → only stores f(u_j), with
    # sup_bounds[1] = ‖sin‖∞ = 1, sup_bounds[2] = ‖cos‖∞ = 1.
    derivs0 = Matrix{Arb}(undef, M_grid, 1)
    for j in 1:M_grid
        u_j = Arb(2; prec=prec) * pi_arb * Arb(j - 1; prec=prec) / Arb(M_grid; prec=prec)
        derivs0[j, 1] = sin(u_j)
    end
    ts_sin0 = TaylorSeries(derivs0, [Arb(1; prec=prec), Arb(1; prec=prec)], 0, M_grid, h, prec)

    # sup_bound should give ≥ 1 (true ‖sin‖∞). Check via ubound — since the
    # 2026-05-22 convention fix, `sup_bound` returns an Arb whose midpoint is
    # the informative estimate (= grid Taylor sum) and radius is the rigorous
    # remainder; the upper bound on the truth is `ubound`, not `lbound`.
    @test Float64(Arblib.ubound(sup_bound(ts_sin0))) >= 1.0

    # Product ts_sin0 * ts_sin0 should give sup_bounds = [1, 2]
    # (Leibniz: m=0: C(0,0)·1·1=1; m=1: C(1,0)·1·1 + C(1,1)·1·1 = 2)
    ts_sq = ts_sin0 * ts_sin0
    @test ts_sq.M_order == 0
    @test length(ts_sq.sup_bounds) == 2
    @test Arblib.le(Arb(1; prec=prec), ts_sq.sup_bounds[1])
    @test Arblib.le(Arb(2; prec=prec), ts_sq.sup_bounds[2])

    # deriv_series should reject M_order=0
    @test_throws AssertionError deriv_series(ts_sin0)

    # inv_series on (2 + sin(u)) with M_order=0; result is 1/(2+sin), bound on
    # (1/f)' = -cos/(2+sin)^2; max at sin=-1, cos=0: 0/1=0. Max at sin=0, cos=±1:
    # 1/4. So ‖(1/f)'‖∞ = 1/4. Test that the bound is ≥ 1/4.
    derivs_f0 = [Arb(2; prec=prec) + sin(Arb(2; prec=prec) * pi_arb *
                                          Arb(j - 1; prec=prec) /
                                          Arb(M_grid; prec=prec))
                 for j in 1:M_grid]
    ts_f0 = TaylorSeries(reshape(derivs_f0, M_grid, 1),
                         [Arb(3; prec=prec), Arb(1; prec=prec)], 0, M_grid, h, prec)
    ts_inv0 = inv_series(ts_f0, Arb(1; prec=prec))
    @test ts_inv0.M_order == 0
    @test length(ts_inv0.sup_bounds) == 2
    @test Arblib.le(Arb(0.25; prec=prec), ts_inv0.sup_bounds[2])
end

# ---------------------------------------------------------------------------
# T2.14 — trigpoly_to_taylor direct test
# ---------------------------------------------------------------------------
# Audit gap: trigpoly_to_taylor is a public export with no direct tests.
# Build TrigPoly for sin(u), convert via trigpoly_to_taylor, verify derivs
# and sup_bounds agree with the analytic answer (sin's derivatives cycle
# through sin, cos, -sin, -cos with supnorm 1 each).

using BabenkoRigorous: trigpoly_to_taylor

@testset "T2.14 — trigpoly_to_taylor matches analytic sin" begin
    prec    = 128
    M_grid  = 32
    M_order = 4

    # TrigPoly for sin(u): a = [0,0], b = [0,1] (degree 1).
    a = [Arb(0; prec=prec), Arb(0; prec=prec)]
    b = [Arb(0; prec=prec), Arb(1; prec=prec)]
    sin_tp = TrigPoly(a, b, prec)

    ts_sin = trigpoly_to_taylor(sin_tp, M_grid, M_order; prec=prec)

    pi_arb = Arb(pi; prec=prec)
    # Verify derivs[j, m+1] = sin^{(m)}(u_j) at every grid point and level.
    for j in 1:M_grid
        u_j = Arb(2; prec=prec) * pi_arb * Arb(j - 1; prec=prec) / Arb(M_grid; prec=prec)
        for m in 0:M_order
            expected = m % 4 == 0 ?  sin(u_j) :
                       m % 4 == 1 ?  cos(u_j) :
                       m % 4 == 2 ? -sin(u_j) :
                                    -cos(u_j)
            @test Arblib.overlaps(ts_sin.derivs[j, m + 1], expected)
        end
    end

    # sup_bounds[m+1] should be ≥ 1 (the true supnorm of every sin derivative).
    for m in 0:(M_order + 1)
        @test Arblib.le(Arb(1; prec=prec), ts_sin.sup_bounds[m + 1])
    end
end

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

println("\nM3 TaylorSeries tests complete.")
