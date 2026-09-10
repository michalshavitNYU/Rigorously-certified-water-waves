"""
test_fft_hilbert.jl — Tests T4.1–T4.6 for the FFTHilbert module (Milestone M4).

Run via:
  julia --project=. test/runtests.jl
or directly:
  julia --project=. test/test_fft_hilbert.jl
"""

using Test
using Arblib
using FFTW

# Bring in the types/functions we need
using BabenkoRigorous: fft_hilbert, fft_hilbert_series
using BabenkoRigorous: TaylorSeries

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

"""Extract Float64 midpoint from Arb."""
arb_mid(x::Arb) = Float64(Arblib.midpoint(Arb, x))

"""Check whether an Arb ball contains a given Float64 value."""
function arb_contains(ball::Arb, val::Float64)
    return Arblib.contains(ball, Arb(val; prec=precision(ball)))
end

"""Compute H[G] reference values via large-N DFT (Float64)."""
function hilbert_reference(G_f64::Vector{Float64})
    N = length(G_f64)
    G_hat = FFTW.rfft(G_f64)
    H_hat = similar(G_hat)
    n_pos = length(G_hat)
    H_hat[1] = 0.0 + 0.0im
    for k in 1:(n_pos - 2)
        H_hat[k+1] = G_hat[k+1] * (-im)
    end
    H_hat[n_pos] = 0.0 + 0.0im
    return FFTW.irfft(H_hat, N)
end

# ---------------------------------------------------------------------------
# T4.1 — Trig poly input: near-zero aliasing
# ---------------------------------------------------------------------------

@testset "T4.1 — Trig poly input: near-zero aliasing" begin
    prec = 128
    N_fft = 256
    # G(u) = sin(3u), H[G] = -cos(3u)
    # G^{(4)} = 3^4 * sin(3u) => Gp_sup = 81
    p = 4
    Gp_sup = Arb(81.0; prec=prec)

    us = [2π * (j - 1) / N_fft for j in 1:N_fft]
    G_vals = [Arb(sin(3 * u); prec=prec) for u in us]

    HG_vals, ε_alias = fft_hilbert(G_vals, N_fft, p, Gp_sup; prec=prec)

    # Each ball must contain the exact value -cos(3u_j)
    all_contain = true
    for (j, u) in enumerate(us)
        exact = -cos(3 * u)
        if !arb_contains(HG_vals[j], exact)
            all_contain = false
            @warn "T4.1 containment failure at j=$j, u=$u, exact=$exact, ball=$(HG_vals[j])"
            break
        end
    end
    @test all_contain

    # Aliasing error bound: ε_alias = 4 * 81 / (3 * 127^3) ≈ 5.28e-5.
    # This is a rigorous (not tight) bound; the actual aliasing is zero for a
    # trig poly of degree 3 with N_fft=256, but the formula uses only the
    # p-th derivative norm, not the exact polynomial structure.
    # We verify: (a) the bound is finite and positive, and (b) it matches
    # the formula 4*Gp_sup / ((p-1)*(N_fft/2 - 1)^(p-1)).
    ε_alias_f64 = arb_mid(ε_alias)
    expected_ε = 4.0 * 81.0 / (3.0 * 127.0^3)
    @test ε_alias_f64 ≈ expected_ε rtol=1e-6
    @test ε_alias_f64 > 0.0
    println("  T4.1: ε_alias = $ε_alias_f64 (expected ≈ $expected_ε)")
end

# ---------------------------------------------------------------------------
# T4.2 — Non-poly function: convergence in N_fft
# ---------------------------------------------------------------------------

@testset "T4.2 — Non-poly function: convergence in N_fft" begin
    prec = 128
    p = 4
    # Conservative bound on G^{(4)} for G = (1+cos(u))/(2+sin(u))
    Gp_sup = Arb(50.0; prec=prec)

    # Large-N reference
    N_ref = 65536
    us_ref = [2π * (j - 1) / N_ref for j in 1:N_ref]
    G_ref_f64 = [(1 + cos(u)) / (2 + sin(u)) for u in us_ref]
    HG_ref_f64 = hilbert_reference(G_ref_f64)

    Ns = [256, 512, 1024, 2048]
    radii = Float64[]
    all_contain = true

    for N_fft in Ns
        us = [2π * (j - 1) / N_fft for j in 1:N_fft]
        G_vals = [Arb((1 + cos(u)) / (2 + sin(u)); prec=prec) for u in us]

        HG_vals, ε_alias = fft_hilbert(G_vals, N_fft, p, Gp_sup; prec=prec)

        # Check containment vs large-N reference at the same grid points
        for j in 1:N_fft
            # Reference at the j-th grid point of N_fft (nearest in ref grid)
            ref_idx = round(Int, (j - 1) * N_ref / N_fft) + 1
            ref_val = HG_ref_f64[ref_idx]
            if !arb_contains(HG_vals[j], ref_val)
                all_contain = false
                @warn "T4.2 containment failure at N_fft=$N_fft, j=$j"
                break
            end
        end

        # Record max ball radius (radref returns MagRef; convert via Float64)
        max_r = maximum(Float64(Arblib.radref(HG_vals[j])) for j in 1:N_fft)
        push!(radii, max_r)
        ε_f64 = arb_mid(ε_alias)
        println("  T4.2: N_fft=$N_fft, ε_alias=$ε_f64")
    end

    @test all_contain

    # Check radii decrease as N^{-(p-1)} = N^{-3} approximately.
    # Ratio of radii for consecutive N should be ≈ 2^{-3} = 0.125.
    for i in 2:length(Ns)
        ratio = radii[i] / radii[i-1]
        println("  T4.2: radius ratio (N=$(Ns[i])/N=$(Ns[i-1])) = $ratio (expect ~0.125)")
        @test ratio < 0.5  # weaker test since Gp_sup is conservative
    end
end

# ---------------------------------------------------------------------------
# T4.3 — Aliasing bound is rigorous
# ---------------------------------------------------------------------------

@testset "T4.3 — Aliasing bound is rigorous" begin
    prec = 128
    N_fft = 256
    p = 4
    Gp_sup = Arb(50.0; prec=prec)

    # Large-N reference
    N_ref = 65536
    us_ref = [2π * (j - 1) / N_ref for j in 1:N_ref]
    G_ref_f64 = [(1 + cos(u)) / (2 + sin(u)) for u in us_ref]
    HG_ref_f64 = hilbert_reference(G_ref_f64)

    us = [2π * (j - 1) / N_fft for j in 1:N_fft]
    G_vals = [Arb((1 + cos(u)) / (2 + sin(u)); prec=prec) for u in us]

    HG_vals, ε_alias = fft_hilbert(G_vals, N_fft, p, Gp_sup; prec=prec)
    ε_f64 = arb_mid(ε_alias)

    # Compare midpoints of HG_vals to the reference
    max_err = 0.0
    for j in 1:N_fft
        ref_idx = round(Int, (j - 1) * N_ref / N_fft) + 1
        ref_val = HG_ref_f64[ref_idx]
        err = abs(arb_mid(HG_vals[j]) - ref_val)
        max_err = max(max_err, err)
    end

    println("  T4.3: ε_alias = $ε_f64, max actual error = $max_err, rigorous: $(ε_f64 >= max_err)")
    @test ε_f64 >= max_err
end

# ---------------------------------------------------------------------------
# T4.4 — TaylorSeries output: derivatives match finite differences
# ---------------------------------------------------------------------------

@testset "T4.4 — TaylorSeries output: derivatives match finite differences" begin
    prec = 128
    N_fft = 256
    M_order = 2

    # G = (1+cos(u))/(2+sin(u)) and its first derivative:
    # G' = (-sin(u)*(2+sin(u)) - (1+cos(u))*cos(u)) / (2+sin(u))^2
    # G'' = ... (more complex, we compute numerically for the test)
    us = [2π * (j - 1) / N_fft for j in 1:N_fft]

    G_f64    = [(1 + cos(u)) / (2 + sin(u)) for u in us]
    Gp_f64   = [(-sin(u)*(2 + sin(u)) - (1 + cos(u))*cos(u)) / (2 + sin(u))^2 for u in us]
    # G'' via product rule (computed analytically)
    function Gpp_f64(u)
        D = 2 + sin(u)
        N = 1 + cos(u)
        Np = -sin(u)
        Dp = cos(u)
        # G = N/D, G' = (N'D - N*D')/D^2
        # G'' = (N''D^2 - 2D*D'*(N'D-N*D') - N*D''*D^2 + ... )
        # Let's compute from scratch:
        # G' = Np/D - N*Dp/D^2
        # G'' = Npp/D - 2*Np*Dp/D^2 - N*Dpp/D^2 + 2*N*Dp^2/D^3
        Npp = -cos(u)
        Dpp = -sin(u)
        return Npp/D - 2*Np*Dp/D^2 - N*Dpp/D^2 + 2*N*Dp^2/D^3
    end
    Gpp_vals = [Gpp_f64(u) for u in us]

    # Conservative sup_bounds for G = (1+cos(u))/(2+sin(u)) and its derivatives.
    # ‖G‖_∞ ≤ 2; higher derivatives grow but 200 is a conservative blanket bound
    # for ‖G^{(m)}‖_∞ over m=0..M_order+1 (i.e., 0..3) for this benign rational.
    pi_arb = Arb(pi; prec=prec)
    h = Arb(2; prec=prec) * pi_arb / Arb(N_fft; prec=prec)
    sup_bounds_G = [Arb(200.0; prec=prec) for _ in 0:(M_order + 1)]

    # Build ts_G manually
    derivs_G = Matrix{Arb}(undef, N_fft, M_order + 1)
    for j in 1:N_fft
        derivs_G[j, 1] = Arb(G_f64[j]; prec=prec)
        derivs_G[j, 2] = Arb(Gp_f64[j]; prec=prec)
        derivs_G[j, 3] = Arb(Gpp_vals[j]; prec=prec)
    end
    ts_G = TaylorSeries(derivs_G, sup_bounds_G, M_order, N_fft, h, prec)

    # Bound on G^{(M_order+2)} = G^{(4)} for the same rational function; the
    # value 50 is the bound already used as Gp_sup for G^{(4)} in T4.2/T4.3.
    # Gppp bounds G^{(M_order+3)} = G^{(5)}; 500 is conservative.
    Gpp_bound  = Arb(50.0;  prec=prec)
    Gppp_bound = Arb(500.0; prec=prec)
    ts_HG = fft_hilbert_series(ts_G, N_fft;
                               Gpp_bound=Gpp_bound, Gppp_bound=Gppp_bound, prec=prec)

    # Large-N reference for H[G]
    N_ref = 65536
    us_ref = [2π * (j - 1) / N_ref for j in 1:N_ref]
    G_ref_f64 = [(1 + cos(u)) / (2 + sin(u)) for u in us_ref]
    HG_ref_f64 = hilbert_reference(G_ref_f64)

    # Check H[G](u_j) containment
    all_contain = true
    for j in 1:N_fft
        ref_idx = round(Int, (j - 1) * N_ref / N_fft) + 1
        ref_val = HG_ref_f64[ref_idx]
        if !arb_contains(ts_HG.derivs[j, 1], ref_val)
            all_contain = false
            @warn "T4.4 H[G] containment failure at j=$j"
            break
        end
    end
    @test all_contain
    println("  T4.4: H[G] containment: $all_contain")

    # Check (H[G])' via large-N DFT of G' — strengthened to ball containment.
    # Previously this was only a Float64 midpoint comparison against the
    # reference (max_err_deriv < 1e-5), which would pass even if the certified
    # ball radius were wildly wrong. Containment is the actual rigor claim.
    Gp_ref_f64 = [(-sin(u)*(2 + sin(u)) - (1 + cos(u))*cos(u)) / (2 + sin(u))^2
                  for u in us_ref]
    HGp_ref_f64 = hilbert_reference(Gp_ref_f64)

    all_contain_deriv = true
    for j in 1:N_fft
        ref_idx = round(Int, (j - 1) * N_ref / N_fft) + 1
        ref_val = HGp_ref_f64[ref_idx]
        if !arb_contains(ts_HG.derivs[j, 2], ref_val)
            all_contain_deriv = false
            @warn "T4.4 (H[G])' containment failure at j=$j, ref=$ref_val, ball=$(ts_HG.derivs[j,2])"
            break
        end
    end
    @test all_contain_deriv
    println("  T4.4: (H[G])' containment: $all_contain_deriv")
end

# ---------------------------------------------------------------------------
# T4.5 — Error budget: aliasing dominates Arb rounding
# ---------------------------------------------------------------------------

@testset "T4.5 — Error budget: aliasing dominates Arb rounding" begin
    prec = 128
    N_fft = 2048
    p = 4
    Gp_sup = Arb(50.0; prec=prec)

    us = [2π * (j - 1) / N_fft for j in 1:N_fft]
    G_vals = [Arb((1 + cos(u)) / (2 + sin(u)); prec=prec) for u in us]

    HG_vals, ε_alias = fft_hilbert(G_vals, N_fft, p, Gp_sup; prec=prec)
    ε_alias_f64 = arb_mid(ε_alias)

    # Arb rounding error ~ machine epsilon * |H[G]|
    # The irfft sums N_fft terms, each with ~2.2e-16 relative error
    # so the total rounding is roughly N_fft * eps * max|HG_mid|
    max_HG_mid = maximum(abs(arb_mid(HG_vals[j])) for j in 1:N_fft)  # float64 ok here
    arb_rounding_estimate = N_fft * eps(Float64) * max_HG_mid * (log(N_fft) + 2)

    ratio = ε_alias_f64 / arb_rounding_estimate
    println("  T4.5: ε_alias = $ε_alias_f64, arb_rounding ~ $arb_rounding_estimate, ratio = $ratio")
    @test ratio > 100
end

# ---------------------------------------------------------------------------
# T4.6 — Containment on random G inputs (trig poly)
# ---------------------------------------------------------------------------

@testset "T4.6 — Containment on random G inputs" begin
    prec = 128
    N_fft = 256
    p = 4

    # Use fixed seed for reproducibility
    import Random
    rng = Random.MersenneTwister(42)

    all_pass = true
    for trial in 1:5
        # G_k(u) = Σ_{j=1}^8 r_j * cos(j*u) / j^2,  r_j ∈ [-1, 1]
        r = 2 * rand(rng, 8) .- 1
        coeffs = [r[j] / j^2 for j in 1:8]   # amplitude of cos(j*u)

        us = [2π * (k - 1) / N_fft for k in 1:N_fft]
        G_f64_vals = [sum(coeffs[j] * cos(j * u) for j in 1:8) for u in us]
        G_vals = [Arb(v; prec=prec) for v in G_f64_vals]

        # Analytic H[G]: H[cos(j*u)] = sin(j*u)
        HG_exact = [sum(coeffs[j] * sin(j * u) for j in 1:8) for u in us]

        # Gp_sup: G^{(4)} = Σ_{j=1}^8 r_j * j^4 * (-cos or sin)(j*u) / j^2
        # = Σ_{j=1}^8 r_j * j^2 * (±cos(j*u))
        # |G^{(4)}|_∞ ≤ Σ_{j=1}^8 |r_j| * j^2
        Gp_sup_f64 = sum(abs(r[j]) * j^2 for j in 1:8)
        Gp_sup = Arb(Gp_sup_f64 + 1.0; prec=prec)  # +1 for safety

        HG_vals, ε_alias = fft_hilbert(G_vals, N_fft, p, Gp_sup; prec=prec)

        for k in 1:N_fft
            if !arb_contains(HG_vals[k], HG_exact[k])
                all_pass = false
                @warn "T4.6 containment failure at trial=$trial, k=$k"
                break
            end
        end
    end

    println("  T4.6: all containment checks passed: $all_pass")
    @test all_pass
end

# ---------------------------------------------------------------------------
# T4.7 — fft_hilbert_series output sup_bounds against analytic G = sin(ku)
# ---------------------------------------------------------------------------
# Closes the audit gap: the Fourier-tail π²/3 output sup_bounds (the fix for
# the previous Lebesgue-constant unsoundness) had zero direct test coverage.
#
# For G(u) = sin(ku):  ‖G^{(m)}‖_∞ = k^m,  ‖H[G^{(m)}]‖_∞ = k^m  (H[sin] = -cos).
# We feed exact sup_bounds and the matching Gpp_bound = k^{M_order+2},
# Gppp_bound = k^{M_order+3}; the output sup_bounds[m+1] should bound k^m
# (in fact by factor k²·π²/3 ≈ 30 for k=3, since the Fourier-tail bound uses
# the (m+2)-th derivative).

using BabenkoRigorous: fft_hilbert_series, TaylorSeries

@testset "T4.7 — fft_hilbert_series output sup_bounds rigor (G = sin(ku))" begin
    prec    = 128
    M_order = 2
    N_fft   = 256
    k       = 3

    pi_arb = Arb(pi; prec=prec)
    h_arb  = Arb(2; prec=prec) * pi_arb / Arb(N_fft; prec=prec)
    k_arb  = Arb(k; prec=prec)

    # G^{(m)}(u) = k^m · {sin(ku), cos(ku), -sin(ku), -cos(ku)}[m mod 4]
    derivs_G = Matrix{Arb}(undef, N_fft, M_order + 1)
    for j in 1:N_fft
        u_j = Arb(2; prec=prec) * pi_arb * Arb(j - 1; prec=prec) / Arb(N_fft; prec=prec)
        ku  = k_arb * u_j
        for m in 0:M_order
            base = m % 4 == 0 ?  sin(ku) :
                   m % 4 == 1 ?  cos(ku) :
                   m % 4 == 2 ? -sin(ku) :
                                -cos(ku)
            derivs_G[j, m+1] = k_arb^m * base
        end
    end

    # sup_bounds[m+1] = k^m for m = 0..M_order+1  (length M_order+2 = 4)
    sup_bounds_G = [k_arb^m for m in 0:(M_order + 1)]

    ts_G = TaylorSeries(derivs_G, sup_bounds_G, M_order, N_fft, h_arb, prec)

    Gpp_bound  = k_arb^(M_order + 2)   # = k^4
    Gppp_bound = k_arb^(M_order + 3)   # = k^5

    ts_HG = fft_hilbert_series(ts_G, N_fft;
                               Gpp_bound=Gpp_bound, Gppp_bound=Gppp_bound, prec=prec)

    # Verify output sup_bounds[m+1] ≥ ‖H[G^{(m)}]‖_∞ = k^m for m = 0..M_order+1.
    # By the formula `sup_bounds[m+1] = ‖G^{(m+2)}‖_∞ · π²/3 = k^{m+2}·π²/3`,
    # we should see roughly k²·π²/3 ≈ 30 (for k=3) of headroom over the truth.
    for m in 0:(M_order + 1)
        true_sup = k_arb^m   # ‖H[G^{(m)}]‖_∞ = k^m
        @test Arblib.le(true_sup, ts_HG.sup_bounds[m + 1])
    end

    # Verify the predicted formula `‖G^{(m+2)}‖_∞ · π²/3` is achieved (within Arb rounding).
    factor_pi2 = pi_arb * pi_arb / Arb(3; prec=prec)
    for m in 0:(M_order + 1)
        Gm2_bound = m + 2 <= M_order + 1 ? sup_bounds_G[m + 3] :
                    m + 2 == M_order + 2 ? Gpp_bound : Gppp_bound
        expected = Gm2_bound * factor_pi2
        @test Arblib.overlaps(ts_HG.sup_bounds[m + 1], expected)
    end

    # Spot check: ts_HG.sup_bounds[end] should equal Gppp_bound · π²/3 = k^5 · π²/3
    expected_top = k_arb^(M_order + 3) * factor_pi2
    @test Arblib.overlaps(ts_HG.sup_bounds[end], expected_top)
    println("  T4.7: sup_bounds[end] = $(Float64(Arblib.midpoint(Arb, ts_HG.sup_bounds[end]))) " *
            "(expected ≈ k^5·π²/3 = $(Float64(Arblib.midpoint(Arb, expected_top))))")
end
