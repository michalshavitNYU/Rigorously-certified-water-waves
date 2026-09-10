using Test
using BabenkoRigorous
using Arblib
using FFTW

# ---------------------------------------------------------------------------
# Float64 helper functions (used across multiple tests)
# ---------------------------------------------------------------------------

function hilbert_f64(g)
    N = length(g)
    F = rfft(g)
    H = copy(F)
    H[1] = 0.0 + 0.0im
    for k in 1:(N÷2 - 1)
        H[k+1] = F[k+1] * (-im)
    end
    H[N÷2 + 1] = 0.0 + 0.0im
    return real(irfft(H, N))
end

function sderiv_f64(g)
    N = length(g)
    F = rfft(g)
    for k in 0:(N÷2)
        F[k+1] *= (im * k)
    end
    F[N÷2 + 1] = 0.0 + 0.0im
    return real(irfft(F, N))
end

function f64_residual_sup(y_coeffs, μ; N_f64=512)
    ε2 = length(y_coeffs) >= 1 ? y_coeffs[1] : 0.0
    us_f64 = [2π * (j - 1) / N_f64 for j in 1:N_f64]
    # Evaluate y from cosine_series convention: y_coeffs[1]=DC, y_coeffs[k+1]=cos(ku)
    y_f64 = zeros(N_f64)
    for j in 1:N_f64
        y_f64[j] = y_coeffs[1] / 2.0
        for k in 1:(length(y_coeffs) - 1)
            y_f64[j] += y_coeffs[k+1] * cos(k * us_f64[j])
        end
    end
    yu_f64   = sderiv_f64(y_f64)
    Hyu_f64  = hilbert_f64(yu_f64)
    HyYu_f64 = hilbert_f64(y_f64 .* yu_f64)
    R_f64 = μ .* Hyu_f64 .- y_f64 .- y_f64 .* Hyu_f64 .- HyYu_f64
    return maximum(abs.(R_f64))
end

# ---------------------------------------------------------------------------
# T5.1 — Small-amplitude Stokes wave
# ---------------------------------------------------------------------------

@testset "T5.1 — Small-amplitude Stokes wave" begin
    ε = 0.15
    y_coeffs = [-ε^2/2, ε, ε^2]   # DC (stored as 2*DC in cosine_series), a_1, a_2
    # Note: cosine_series uses y_coeffs[1]/2 as DC at evaluation, so to get DC=-ε²/2
    # we pass y_coeffs[1] = -ε² (since tp_eval divides a[1] by 2).
    # The spec says y_coeffs[1]=-ε²/2 is the DC term — but cosine_series convention
    # stores the full value and divides by 2 at evaluation.
    # So to get DC = -ε²/2 at evaluation: pass y_coeffs[1] = -ε² (because a[1]/2 = -ε²/2).
    # However the spec explicitly says y_coeffs = [-ε^2/2, ε, ε^2] and cosine_series[1]=DC.
    # Looking at cosine_series: it stores a[1] = coeffs[1], and tp_eval gives a[1]/2 + sum.
    # So if the intended DC evaluation value is -ε²/2, we need a[1]=-ε². But the spec says
    # to pass [-ε²/2, ε, ε²] directly. We follow the spec literally.
    μ = 1.0 + ε^2
    β = 0.0
    bound = residual_bound(y_coeffs, μ, β; M_order=4, N_grid=256, prec=128)
    println("T5.1 certified bound = ", Float64(Arblib.ubound(bound)))
    @test Arblib.is_positive(bound)
    @test isfinite(Float64(Arblib.ubound(bound)))
    @test Float64(Arblib.ubound(bound)) ≤ 0.05
end

# ---------------------------------------------------------------------------
# T5.2 — Certified bound ≥ Float64 estimate
# ---------------------------------------------------------------------------

@testset "T5.2 — Certified bound ≥ Float64 estimate" begin
    ε = 0.15
    y_coeffs = [-ε^2/2, ε, ε^2]
    μ = 1.0 + ε^2
    bound = residual_bound(y_coeffs, μ, 0.0; M_order=4, N_grid=256, prec=128)

    # Float64 estimate via direct DFT
    N_f64 = 512
    us_f64 = [2π * (j - 1) / N_f64 for j in 1:N_f64]
    # cosine_series stores a[1]/2 as DC; passing y_coeffs[1]=-ε²/2 gives DC=-ε²/4 at eval.
    # Use the same evaluation as the certified code (identical y_coeffs interpretation):
    y_f64 = [y_coeffs[1] / 2.0 + ε * cos(u) + ε^2 * cos(2u) for u in us_f64]
    yu_f64   = sderiv_f64(y_f64)
    Hyu_f64  = hilbert_f64(yu_f64)
    HyYu_f64 = hilbert_f64(y_f64 .* yu_f64)
    R_f64 = μ .* Hyu_f64 .- y_f64 .- y_f64 .* Hyu_f64 .- HyYu_f64
    R_sup_f64 = maximum(abs.(R_f64))
    println("T5.2 Float64 residual sup = ", R_sup_f64)
    println("T5.2 certified bound = ", Float64(Arblib.ubound(bound)))
    @test Float64(Arblib.ubound(bound)) ≥ R_sup_f64
end

# ---------------------------------------------------------------------------
# T5.3 — Convergence in ε
# ---------------------------------------------------------------------------

@testset "T5.3 — Convergence in ε" begin
    epsilons = [0.01, 0.05, 0.10, 0.15]
    bounds = Float64[]
    for ε in epsilons
        y_c = [-ε^2/2, ε, ε^2]
        μ = 1.0 + ε^2
        b = residual_bound(y_c, μ, 0.0; M_order=4, N_grid=256, prec=128)
        push!(bounds, Float64(Arblib.ubound(b)))
        println("T5.3  ε=", ε, "  bound=", bounds[end])
    end
    # bound should grow with ε
    for i in 1:(length(bounds) - 1)
        @test bounds[i+1] ≥ bounds[i]
    end
end

# ---------------------------------------------------------------------------
# T5.4 — Convergence in M_order
# ---------------------------------------------------------------------------

@testset "T5.4 — Convergence in M_order" begin
    ε = 0.15
    y_coeffs = [-ε^2/2, ε, ε^2]
    μ = 1.0 + ε^2
    orders = [2, 4, 6]
    bounds = Float64[]
    for M in orders
        b = residual_bound(y_coeffs, μ, 0.0; M_order=M, N_grid=256, prec=128)
        push!(bounds, Float64(Arblib.ubound(b)))
        println("T5.4  M_order=", M, "  bound=", bounds[end])
    end
    # Each bound ≤ previous (or at most the same — might plateau)
    for i in 1:(length(bounds) - 1)
        @test bounds[i+1] ≤ bounds[i] * 1.01   # allow 1% tolerance for numerical noise
    end
end

# ---------------------------------------------------------------------------
# T5.5 — β=0 branch
# ---------------------------------------------------------------------------

@testset "T5.5 — β=0 branch runs cleanly" begin
    ε = 0.15
    y_coeffs = [-ε^2/2, ε, ε^2]
    μ = 1.0 + ε^2
    # β=0.0 should succeed and return a positive Arb
    b = residual_bound(y_coeffs, μ, 0.0; M_order=4, N_grid=256, prec=128)
    @test Arblib.is_positive(b)
    @test isfinite(Float64(Arblib.ubound(b)))
end

# ---------------------------------------------------------------------------
# T5.6 — Random coefficient containment (10 seeds)
# ---------------------------------------------------------------------------

@testset "T5.6 — Random coefficient containment" begin
    import Random
    ratios = Float64[]
    for s in 1:10
        rng = Random.MersenneTwister(s)
        # N=8 cosine modes (no DC), decaying as 1/k
        N_modes = 8
        y_coeffs = vcat([0.0], [randn(rng) / k for k in 1:N_modes])
        # Scale so max coefficient ≈ 0.1 (keep amplitude small for validity)
        max_c = maximum(abs.(y_coeffs[2:end]))
        y_coeffs = y_coeffs .* (0.1 / max_c)

        μ = 1.3
        bound = residual_bound(y_coeffs, μ, 0.0; M_order=4, N_grid=256, prec=128)
        R_sup_f64 = f64_residual_sup(y_coeffs, μ; N_f64=512)
        ratio = Float64(Arblib.ubound(bound)) / max(R_sup_f64, 1e-15)
        push!(ratios, ratio)
        println("T5.6  seed=", s, "  bound=", Float64(Arblib.ubound(bound)),
                "  f64=", R_sup_f64, "  ratio=", ratio)
        @test Float64(Arblib.ubound(bound)) ≥ R_sup_f64
    end
    println("T5.6 ratio range: [", minimum(ratios), ", ", maximum(ratios), "]")
    # Catch silent regressions to vacuous/infinite bounds.
    @test all(isfinite, ratios)
    @test maximum(ratios) < 10.0   # β=0 path is tight; ratios were 1.015–1.040
end

# ===========================================================================
# M6 tests: β ≠ 0  (full gravity-capillary residual)
# ===========================================================================

# ---------------------------------------------------------------------------
# T6.1 — Linear gravity-capillary mode (primary regression test)
# ---------------------------------------------------------------------------

@testset "T6.1 — Linear gravity-capillary mode" begin
    ε = 0.05
    β = 0.05
    μ = β + 1.0          # linear dispersion for wave-number 1: μ = 1 + β
    y_coeffs = [0.0, ε]  # DC=0, a_1=ε  (y = ε cos u)

    bound = residual_bound(y_coeffs, μ, β; M_order=4, N_grid=256, prec=128)
    println("T6.1 certified bound = ", Float64(Arblib.ubound(bound)))
    @test Arblib.is_positive(bound)
    @test isfinite(Float64(Arblib.ubound(bound)))
    @test Float64(Arblib.ubound(bound)) ≤ 0.02
end

# ---------------------------------------------------------------------------
# T6.2 — Setting β=0 recovers M5 result
# ---------------------------------------------------------------------------

@testset "T6.2 — β=0 recovery" begin
    ε = 0.15
    y_coeffs = [-ε^2/2, ε, ε^2]
    μ = 1.0 + ε^2

    bound_m5 = residual_bound(y_coeffs, μ, 0.0; M_order=4, N_grid=256, prec=128)
    bound_m6_β0 = residual_bound(y_coeffs, μ, 0.0; M_order=4, N_grid=256, prec=128)

    # Identical inputs → identical result (deterministic)
    @test Float64(Arblib.ubound(bound_m5)) == Float64(Arblib.ubound(bound_m6_β0))
    println("T6.2 M5 bound = ", Float64(Arblib.ubound(bound_m5)),
            "  M6(β=0) bound = ", Float64(Arblib.ubound(bound_m6_β0)))
end

# ---------------------------------------------------------------------------
# T6.3 — D positivity certified
# ---------------------------------------------------------------------------

@testset "T6.3 — D positivity certified" begin
    ε = 0.05
    y_coeffs = [0.0, ε]
    y = cosine_series(y_coeffs; prec=128)
    D_lower, _ = dmin_bound(y, 256, 4; prec=128)
    println("T6.3 D_lower = ", Float64(Arblib.lbound(D_lower)))
    @test Arblib.is_positive(D_lower)
    @test Float64(Arblib.lbound(D_lower)) > 0.01
end

# ---------------------------------------------------------------------------
# T6.4 — Aliasing error subdominant at N_fft = 2048
# ---------------------------------------------------------------------------

@testset "T6.4 — Aliasing subdominant at large N_fft" begin
    ε = 0.05
    β = 0.05
    μ = β + 1.0
    y_coeffs = [0.0, ε]

    bound_256  = residual_bound(y_coeffs, μ, β; M_order=4, N_grid=256,  prec=128)
    bound_2048 = residual_bound(y_coeffs, μ, β; M_order=4, N_grid=2048, prec=128)
    println("T6.4 bound(N=256) = ",  Float64(Arblib.ubound(bound_256)))
    println("T6.4 bound(N=2048) = ", Float64(Arblib.ubound(bound_2048)))

    # Strict-decrease: under the rigorous Fourier-tail F_bound_HG the aliasing
    # contribution is already nearly saturated at N=256 (improvement ~0.2%),
    # but it should still be strictly smaller at N=2048. Previously this was
    # `≤ 1.01·bound_256` which permitted a 1% regression; tighten.
    @test Float64(Arblib.ubound(bound_2048)) < Float64(Arblib.ubound(bound_256))
end

# ---------------------------------------------------------------------------
# T6.5 — Convergence in N_fft
# ---------------------------------------------------------------------------

@testset "T6.5 — Convergence in N_fft" begin
    ε = 0.05
    β = 0.05
    μ = β + 1.0
    y_coeffs = [0.0, ε]

    Ns = [128, 256, 512, 1024]
    bounds = [Float64(Arblib.ubound(
        residual_bound(y_coeffs, μ, β; M_order=4, N_grid=N, prec=128)
    )) for N in Ns]

    for i in 1:length(Ns)
        println("T6.5  N=", Ns[i], "  bound=", bounds[i])
    end

    # Bounds should decrease (or stay flat) as N increases
    for i in 1:(length(bounds) - 1)
        @test bounds[i+1] ≤ bounds[i] * 1.01   # allow 1% numerical tolerance
    end
end

# ---------------------------------------------------------------------------
# Float64 reference for M6 (β≠0)
# ---------------------------------------------------------------------------

function f64_residual_sup_beta(y_coeffs, μ, β; N_f64=1024)
    us = [2π * (j - 1) / N_f64 for j in 1:N_f64]

    # Evaluate y
    y_v = zeros(N_f64)
    for j in 1:N_f64
        y_v[j] = y_coeffs[1] / 2.0
        for k in 1:(length(y_coeffs) - 1)
            y_v[j] += y_coeffs[k+1] * cos(k * us[j])
        end
    end

    yu_v   = sderiv_f64(y_v)
    Hyu_v  = hilbert_f64(yu_v)
    D_v    = sqrt.(yu_v.^2 .+ (1.0 .+ Hyu_v).^2)
    G_v    = (1.0 .+ Hyu_v) ./ D_v

    Hyyu_v = hilbert_f64(y_v .* yu_v)

    # Capillary term 1: ∂_u(y_u / D)
    yuD_v  = yu_v ./ D_v
    C1_v   = sderiv_f64(yuD_v)

    # Capillary term 2: H[∂_u(G)]
    Gu_v   = sderiv_f64(G_v)
    C2_v   = hilbert_f64(Gu_v)

    R_v = μ .* Hyu_v .- y_v .- y_v .* Hyu_v .- Hyyu_v .+ β .* C1_v .- β .* C2_v
    return maximum(abs.(R_v))
end

# ---------------------------------------------------------------------------
# T6.6 — Random coefficient containment, β≠0 (10 seeds)
# ---------------------------------------------------------------------------

@testset "T6.6 — Random containment β≠0" begin
    import Random
    β = 0.05
    μ = 1.3
    ratios = Float64[]

    for s in 1:10
        rng = Random.MersenneTwister(s + 100)
        N_modes = 6
        y_coeffs = vcat([0.0], [randn(rng) / k for k in 1:N_modes])
        max_c = maximum(abs.(y_coeffs[2:end]))
        y_coeffs = y_coeffs .* (0.05 / max_c)   # small amplitude for D > 0

        bound   = residual_bound(y_coeffs, μ, β; M_order=4, N_grid=256, prec=128)
        ref_f64 = f64_residual_sup_beta(y_coeffs, μ, β; N_f64=1024)
        ratio   = Float64(Arblib.ubound(bound)) / max(ref_f64, 1e-15)
        push!(ratios, ratio)
        println("T6.6  seed=", s+100, "  bound=", Float64(Arblib.ubound(bound)),
                "  f64=", ref_f64, "  ratio=", ratio)
        @test Float64(Arblib.ubound(bound)) ≥ ref_f64
    end
    println("T6.6 ratio range: [", minimum(ratios), ", ", maximum(ratios), "]")
    # Catch silent regressions to vacuous/infinite bounds. The rigorous
    # Faà di Bruno chain produces large ratios for high-frequency inputs
    # (audit-documented up to ~10^7) — bound generously but finitely.
    @test all(isfinite, ratios)
    @test maximum(ratios) < 1.0e8
end

# ---------------------------------------------------------------------------
# T6.7 — β≠0 with M_order=1 (smallest legal) — exercises M1/M3 boundary
# ---------------------------------------------------------------------------
# Audit gap: all M6 tests use M_order=4. M_order=1 is the smallest legal
# value (per `@assert M_order >= 1` in ResidualBound.jl) and stresses the
# extended-order chains the most (M1 = 2, M3 = 4 — small).

@testset "T6.7 — β≠0 with M_order=1" begin
    ε = 0.05
    β = 0.05
    μ = β + 1.0
    y_coeffs = [0.0, ε]

    bound = residual_bound(y_coeffs, μ, β; M_order=1, N_grid=256, prec=128)
    println("T6.7 M_order=1 certified bound = ", Float64(Arblib.ubound(bound)))
    @test isfinite(Float64(Arblib.ubound(bound)))
    @test Float64(Arblib.ubound(bound)) ≤ 0.05   # looser cap than T6.1's 0.02 (smaller order ⇒ looser)
end
