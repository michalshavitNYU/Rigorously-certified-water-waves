#!/usr/bin/env julia
# run_pinn.jl — Rigorous residual bound on the PINN-projected solution +
#               end-to-end midpoint-vs-Float64 verification harness.
#
# Reads cosine coefficients from BabenkoRigorous/pinn_a_coeffs_N110.txt
# (produced by scripts/pinn_cosine_projection.py) and:
#   1. Computes the Float64 ‖R‖∞ via FFT — the "truth" the Arb midpoint
#      should track (see docs/changes_2026-05-22_midpoint_convention.md).
#   2. Runs residual_bound(...) at the agreed PINN parameter convention
#      (μ = lamb², β = beta²; see docs/pinn_parameter_convention.md) and
#      prints both midpoint (informative) and ubound (rigorous).
#   3. Reports ratio Arb_midpoint / Float64_truth as the regression metric.
#
# Expected behaviour at M_order=4, N_grid=2048, prec=128 (as of 2026-05-22):
#   Float64 ‖R‖∞ ≈ 1.518e-7
#   Arb midpoint ≈ 1.576e-7  (matches truth to ~4%)
#   Arb ubound   ≈ 3.2e17    (loose, blocked on the sqrt_series sup_bound
#                             tightening task — see docs/changes... §5)
#
# Use this script as the canonical verification harness whenever you change
# anything in the BabenkoRigorous pipeline: the midpoint should always track
# the Float64 truth; if it drifts, the Arb arithmetic has a real bug (vs a
# loose bound, which only shows up in the radius).

using Printf
using FFTW
using BabenkoRigorous
using Arblib

const COEFF_PATH = joinpath(@__DIR__, "..", "pinn_a_coeffs_N110.txt")

# ---------------------------------------------------------------------------
# Load coefficients
# ---------------------------------------------------------------------------
function load_coeffs(path::String)
    coeffs = Float64[]
    for line in eachline(path)
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        push!(coeffs, parse(Float64, s))
    end
    return coeffs
end

# ---------------------------------------------------------------------------
# Float64 baseline residual (matches the convention in test_residual_bound.jl)
# ---------------------------------------------------------------------------
function f64_hilbert(g::AbstractVector{<:Real})
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

function f64_sderiv(g::AbstractVector{<:Real})
    N = length(g)
    F = rfft(g)
    for k in 0:(N÷2)
        F[k+1] *= (im * k)
    end
    F[N÷2 + 1] = 0.0 + 0.0im
    return real(irfft(F, N))
end

function f64_residual(y_coeffs::Vector{Float64}, μ::Float64, β::Float64; N=2048)
    us = [2π * (j - 1) / N for j in 1:N]
    y = zeros(N)
    for j in 1:N
        s = y_coeffs[1] / 2.0
        for k in 1:(length(y_coeffs) - 1)
            s += y_coeffs[k+1] * cos(k * us[j])
        end
        y[j] = s
    end
    yu  = f64_sderiv(y)
    Hyu = f64_hilbert(yu)
    D   = sqrt.(yu .^ 2 .+ (1 .+ Hyu) .^ 2)
    HyYu = f64_hilbert(y .* yu)
    R = μ .* Hyu .- y .- y .* Hyu .- HyYu
    if β != 0.0
        cap1 = f64_sderiv(yu ./ D)
        cap2 = f64_hilbert(f64_sderiv((1 .+ Hyu) ./ D))
        R = R .+ β .* cap1 .- β .* cap2
    end
    return (R_sup = maximum(abs.(R)),
            y_sup = maximum(abs.(y)),
            D_min = minimum(D),
            D_max = maximum(D))
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function main()
    coeffs = load_coeffs(COEFF_PATH)
    N = length(coeffs) - 1
    @printf("Loaded %d coefficients (degree N = %d) from %s\n",
            length(coeffs), N, COEFF_PATH)
    @printf("  a[0]   = %+.12e\n", coeffs[1])
    @printf("  a[1]   = %+.12e\n", coeffs[2])
    @printf("  a[%d] = %+.3e\n", N, coeffs[end])

    # PINN convention (Wilton_ripple notebook, cell 14 `stwv_gov_eqn`):
    #   -λ² H[y_x] + g·y + g·H[y·y_x] + g·y·H[y_x] + β² · (-cap1 + cap2) = 0
    # vs our `ResidualBound` form:
    #   μ H[y_u] − y − y H[y_u] − H(y·y_u) + β cap1 − β cap2 = R
    # ⇒ μ = λ²,  β_doc = β_PINN²,  with g = 1.
    lamb = 1.168786478294966
    beta = 0.498000000104982
    μ    = lamb^2
    β    = beta^2
    @printf("\nPINN λ = %.12f, β_PINN = %.12f\n", lamb, beta)
    @printf("→ Babenko μ = λ²    = %.12f\n", μ)
    @printf("→ Babenko β = β_PINN² = %.12f\n", β)

    # --- Float64 baseline ---
    fb = f64_residual(coeffs, μ, β; N=4096)
    @printf("\nFloat64 baseline (DFT on N=4096):\n")
    @printf("  ‖y‖∞              = %.6e\n", fb.y_sup)
    @printf("  D ∈ [%.4f, %.4f]\n", fb.D_min, fb.D_max)
    @printf("  ‖R‖∞ (Float64)    = %.6e\n", fb.R_sup)

    # --- Certified Arb bound ---
    # N_grid must be large enough for dmin_bound's Lipschitz argument: with
    # Lipschitz ≈ 29 and D_min ≈ 0.34, we need h · Lip < D_min ⇒ N_grid ≳ 600.
    # Use 2048 for safety margin (Lip·h ≈ 0.09).
    M_order      = 4
    N_grid       = 2048
    prec         = 128
    extra_orders = 0   # tighten + tight_DG path means extending the chain is counter-productive
    @printf("\nCertified bound: M_order=%d, N_grid=%d, prec=%d, extra_orders=%d ...\n",
            M_order, N_grid, prec, extra_orders)
    t_start = time()
    bound = residual_bound(coeffs, μ, β; M_order=M_order, N_grid=N_grid,
                           prec=prec, extra_orders=extra_orders)
    elapsed = time() - t_start
    ub  = Float64(Arblib.ubound(bound))
    lb  = Float64(Arblib.lbound(bound))
    mid = Float64(Arblib.midref(bound))
    rad = Float64(Arblib.radref(bound))
    @printf("\nResidual bound:\n")
    @printf("    midpoint  = %+.6e        (should track Float64 truth — convention check)\n", mid)
    @printf("    radius    = %.4e\n", rad)
    @printf("    [lbound, ubound] = [%+.6e, %+.6e]   (rigorous interval)\n", lb, ub)
    @printf("    rigorous upper bound on ‖R‖∞ = %.6e\n", ub)
    @printf("\nRegression check (midpoint vs Float64 truth):\n")
    @printf("    Arb_midpoint / Float64_sup = %.4f   (1.0 ± few %% expected)\n", mid / fb.R_sup)
    @printf("    Arb_ubound   / Float64_sup = %.3e   (informative; drives certificate size)\n", ub / fb.R_sup)
    @printf("    elapsed: %.1fs\n", elapsed)
end

main()
