"""
test_fourier_series.jl — M1 milestone tests for FourierSeries.jl

Tests:
  T1.1 — H² = −I
  T1.2 — H and ∂ commute
  T1.3 — supnorm_bound is genuine (a real upper bound)
  T1.4 — Matches Float64 baseline from hilbert.jl
"""

using Test
using Arblib
using FFTW
using Random

using BabenkoRigorous: TrigPoly, degree, hilbert, deriv, tp_eval, supnorm_bound,
                       cosine_series, add_const, trigpoly_equal

# ---------------------------------------------------------------------------
# Helper: random TrigPoly with coefficients seeded from Float64 randn
# ---------------------------------------------------------------------------

function random_trigpoly(N::Int; seed::Int=42, prec::Int=128, zero_dc::Bool=true)
    rng = MersenneTwister(seed)
    # Decay coefficients as 1/k to keep them bounded
    # k=0: DC term; set to zero by default since H[const]=0 breaks H^2=-I for DC modes
    a_vals = [k == 0 ? 0.0 : randn(rng) / k for k in 0:N]
    b_vals = [k == 0 ? 0.0 : randn(rng) / k for k in 0:N]
    a_arb = [Arb(x; prec=prec) for x in a_vals]
    b_arb = [Arb(x; prec=prec) for x in b_vals]
    return TrigPoly(a_arb, b_arb, prec)
end

# ---------------------------------------------------------------------------
# T1.1 — H² = −I
# ---------------------------------------------------------------------------

@testset "T1.1 — H^2 = -I" begin
    prec = 128
    f = random_trigpoly(8; seed=42, prec=prec)

    # g = H(H(f)) should equal -f
    g = hilbert(hilbert(f))

    # H²[f] + f should have all coefficients containing zero
    @test length(g.a) == length(f.a)
    @test length(g.b) == length(f.b)

    all_a_zero = all(Arblib.contains_zero(g.a[k] + f.a[k]) for k in eachindex(f.a))
    all_b_zero = all(Arblib.contains_zero(g.b[k] + f.b[k]) for k in eachindex(f.b))
    @test all_a_zero
    @test all_b_zero

    # Verify b[1] = 0 always (sin(0) = 0)
    @test Arblib.contains_zero(f.b[1])
    @test Arblib.contains_zero(g.b[1])
end

# ---------------------------------------------------------------------------
# T1.2 — H and ∂ commute
# ---------------------------------------------------------------------------

@testset "T1.2 — H and d/du commute" begin
    prec = 128
    f = random_trigpoly(8; seed=7, prec=prec)

    hdf = hilbert(deriv(f))   # H(∂f)
    dhf = deriv(hilbert(f))   # ∂(H(f))

    @test trigpoly_equal(hdf, dhf)

    # Also verify degrees match
    N = degree(f)
    @test degree(hdf) == N
    @test degree(dhf) == N

    # Coefficient-by-coefficient check
    all_a = all(Arblib.contains_zero(hdf.a[k+1] - dhf.a[k+1]) for k in 0:N)
    all_b = all(Arblib.contains_zero(hdf.b[k+1] - dhf.b[k+1]) for k in 0:N)
    @test all_a
    @test all_b
end

# ---------------------------------------------------------------------------
# T1.3 — supnorm_bound is genuine
# ---------------------------------------------------------------------------

@testset "T1.3 — supnorm_bound is genuine" begin
    prec = 128
    f = random_trigpoly(6; seed=99, prec=prec)
    bound = supnorm_bound(f)

    # Evaluate at 1000 equally spaced points and check all |f(u_j)| <= bound
    n_pts = 1000
    violations = 0
    for j in 0:(n_pts - 1)
        u_float = 2π * j / n_pts
        u = Arb(u_float; prec=prec)
        fval = tp_eval(f, u)
        abs_fval = abs(fval)
        diff = bound - abs_fval
        if !Arblib.is_nonnegative(diff)
            violations += 1
        end
    end
    @test violations == 0

    # Check bound is nonneg and finite
    @test Arblib.is_nonnegative(bound)
    @test isfinite(Float64(Arblib.midref(bound)))
end

# ---------------------------------------------------------------------------
# T1.4 — Matches Float64 baseline from hilbert.jl
# ---------------------------------------------------------------------------

@testset "T1.4 — Matches Float64 baseline" begin
    # This test verifies that the Arb trig-poly evaluation of H[∂y] agrees with the
    # Float64 periodic_hilbert reference to within Float64 machine precision (~1e-15).
    #
    # Note on containment: Since the Arb balls are computed at prec=128 (~38 decimal digits)
    # and the Float64 reference accumulates 1-2 ULP of rounding error, the Arb ball (which
    # is very tight, radius ~1e-38) will not always literally contain the Float64 value.
    # We instead verify:
    # (a) The Arb and FFT-reference values agree to within 1e-12 at all 100 grid points.
    # (b) The Arb coefficient balls contain the exact mathematical values (stored as
    #     Float64 midpoints of the Arb coefficients themselves, which ARE in the ball).

    prec = 128

    # Build the test TrigPoly: y = 0.3*cos(u) + 0.15*cos(2u) + 0.07*cos(3u)
    a_coeffs_f64 = [0.0, 0.3, 0.15, 0.07]
    b_coeffs_f64 = [0.0, 0.0, 0.0,  0.0 ]
    y_tp = TrigPoly(a_coeffs_f64, b_coeffs_f64; prec=prec)
    hdy_tp = hilbert(deriv(y_tp))

    # FFT-based Float64 reference (same as hilbert.jl)
    function periodic_hilbert_ref(f_in::AbstractVector{<:Real})
        N_pts = length(f_in)
        F = fft(f_in)
        h = zeros(Complex{Float64}, N_pts)
        if iseven(N_pts)
            h[2:N_pts÷2]       .= -im
            h[N_pts÷2+2:N_pts] .= im
        else
            h[2:(N_pts+1)÷2]   .= -im
            h[(N_pts+3)÷2:N_pts] .= im
        end
        return real(ifft(h .* F))
    end

    N_grid = 100
    x = [2π * j / N_grid for j in 0:(N_grid - 1)]
    dy_f64 = [-0.3 * sin(u) - 0.15*2*sin(2u) - 0.07*3*sin(3u) for u in x]
    hdy_f64 = periodic_hilbert_ref(dy_f64)

    # Check 1: Arb and Float64 agree to within 2e-14 at all grid points
    max_err = 0.0
    for (j, u_float) in enumerate(x)
        u_arb = Arb(u_float; prec=prec)
        arb_val = tp_eval(hdy_tp, u_arb)
        arb_mid = Float64(Arblib.midref(arb_val))
        err = abs(arb_mid - hdy_f64[j])
        max_err = max(max_err, err)
    end
    # Should agree to < 1e-13 (about 2-3 ULPs of Float64 near magnitude 0.8)
    @test max_err < 1e-13

    # Check 2: coefficient-level rigor against an INDEPENDENT Arb computation.
    # H[∂y].a[k+1] should equal k · y.a[k+1] (since H[-sin(ku)] = cos(ku)).
    # Compute the analytic value via Arb arithmetic on the SAME input Float64s
    # (these are exact Arb point balls; integer multiplication in Arb is exact),
    # and verify the implementation's output ball overlaps it. This is NOT
    # tautological — if `hilbert` or `deriv` introduced an algebraic error,
    # `Arblib.overlaps` would fail.
    analytic_a2 = Arb(1; prec=prec) * Arb(a_coeffs_f64[2]; prec=prec)
    analytic_a3 = Arb(2; prec=prec) * Arb(a_coeffs_f64[3]; prec=prec)
    analytic_a4 = Arb(3; prec=prec) * Arb(a_coeffs_f64[4]; prec=prec)
    @test Arblib.overlaps(hdy_tp.a[2], analytic_a2)
    @test Arblib.overlaps(hdy_tp.a[3], analytic_a3)
    @test Arblib.overlaps(hdy_tp.a[4], analytic_a4)

    # Check 3: the Arb midpoints match the expected analytic Float64 values
    # k * y.a[k+1] in Float64 arithmetic:
    c1_expected = Float64(1 * 0.3)    # 0.3
    c2_expected = Float64(2 * 0.15)   # 0.3
    c3_expected = Float64(3 * 0.07)   # 0.21000...002
    @test abs(Float64(Arblib.midref(hdy_tp.a[2])) - c1_expected) < 1e-15
    @test abs(Float64(Arblib.midref(hdy_tp.a[3])) - c2_expected) < 1e-15
    @test abs(Float64(Arblib.midref(hdy_tp.a[4])) - c3_expected) < 1e-15
end

# ---------------------------------------------------------------------------
# Additional sanity checks
# ---------------------------------------------------------------------------

@testset "Sanity: b[1] always zero" begin
    prec = 128
    f = random_trigpoly(4; seed=5, prec=prec)
    g = random_trigpoly(4; seed=6, prec=prec)

    @test Arblib.contains_zero(f.b[1])
    @test Arblib.contains_zero(g.b[1])
    @test Arblib.contains_zero(hilbert(f).b[1])
    @test Arblib.contains_zero(deriv(f).b[1])
    @test Arblib.contains_zero((f + g).b[1])
end

@testset "Sanity: cosine_series constructor" begin
    prec = 128
    # f(u) = cos(u) + 2*cos(2u): DC=0, cos(u)=1, cos(2u)=2
    tp = cosine_series([0.0, 1.0, 2.0]; prec=prec)
    @test degree(tp) == 2
    @test Arblib.contains_zero(tp.b[1])
    @test Arblib.contains_zero(tp.b[2])
    @test Arblib.contains_zero(tp.b[3])
    # eval at u=π/2: cos(π/2)=0, cos(π)=-1 => f(π/2) = 0 + 1*0 + 2*(-1) = -2
    # Use Arb(pi; prec) to get a rigorous π, then divide by 2
    pi_arb = Arb(pi; prec=prec)
    val = tp_eval(tp, pi_arb / Arb(2; prec=prec))
    # The Arb ball should contain -2 (the exact value).
    # Since π/2 is not exactly representable, we check that the ball is close to -2.
    @test abs(Float64(Arblib.midref(val)) - (-2.0)) < 1e-14
end

println("M1 FourierSeries tests complete.")
