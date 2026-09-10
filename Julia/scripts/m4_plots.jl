"""
m4_plots.jl — Figures for Milestone M4: FFT-based certified Hilbert transform.

Run from the BabenkoRigorous root:
  julia --project=. scripts/m4_plots.jl

Generates three figures in docs/figures/:
  fig_m4_1_convergence.pdf  — ball radius vs N_fft (log-log) with N^{-3} reference
  fig_m4_2_aliasing_rigor.pdf — max |error|/ε_alias ratio at each N_fft
  fig_m4_3_containment.pdf  — trig poly: exact H[G] vs Arb midpoints + band
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using BabenkoRigorous
using BabenkoRigorous: fft_hilbert
using Arblib, Plots, Printf, FFTW
import Random

gr()
default(linewidth=2, framestyle=:box, size=(700, 420),
        left_margin=5Plots.mm, bottom_margin=5Plots.mm)

OUTDIR = joinpath(@__DIR__, "..", "docs", "figures")
mkpath(OUTDIR)

prec = 128

arb_mid(x::Arb) = Float64(Arblib.midpoint(Arb, x))

"""Compute H[G] reference via large-N DFT (Float64)."""
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
# Shared setup: large-N reference
# ---------------------------------------------------------------------------
N_ref = 65536
us_ref = [2π * (j - 1) / N_ref for j in 1:N_ref]
G_ref_f64 = [(1 + cos(u)) / (2 + sin(u)) for u in us_ref]
HG_ref_f64 = hilbert_reference(G_ref_f64)

Ns = [256, 512, 1024, 2048]
p = 4
Gp_sup = Arb(50.0; prec=prec)

# Collect data for all N
radii_by_N  = Float64[]
max_err_by_N = Float64[]
eps_by_N    = Float64[]

for N_fft in Ns
    us = [2π * (j - 1) / N_fft for j in 1:N_fft]
    G_vals = [Arb((1 + cos(u)) / (2 + sin(u)); prec=prec) for u in us]

    HG_vals, ε_alias = fft_hilbert(G_vals, N_fft, p, Gp_sup; prec=prec)
    ε_f64 = arb_mid(ε_alias)

    # Max ball radius (radref returns MagRef, directly convertible to Float64)
    max_r = maximum(Float64(Arblib.radref(HG_vals[j])) for j in 1:N_fft)
    push!(radii_by_N, max_r)
    push!(eps_by_N, ε_f64)

    # Max actual error vs large-N reference
    max_err = 0.0
    for j in 1:N_fft
        ref_idx = round(Int, (j - 1) * N_ref / N_fft) + 1
        err = abs(arb_mid(HG_vals[j]) - HG_ref_f64[ref_idx])
        max_err = max(max_err, err)
    end
    push!(max_err_by_N, max_err)

    @printf("N_fft=%4d  ε_alias=%.3e  max_err=%.3e  max_radius=%.3e  ratio=%.3f\n",
            N_fft, ε_f64, max_err, max_r, max_err / ε_f64)
end

# ===========================================================================
# Figure 1: Log-log plot of ball radius vs N_fft with N^{-3} reference line
# ===========================================================================
println("\nGenerating fig_m4_1_convergence.pdf...")
let
    N_ref_line = [256.0, 2048.0]
    # Reference line: C * N^{-3}, anchored at first data point
    C = radii_by_N[1] * Ns[1]^3
    ref_line = C ./ N_ref_line.^3

    p1 = plot(Float64.(Ns), radii_by_N;
              xscale=:log10, yscale=:log10,
              marker=:circle, markersize=8, color=:royalblue, linewidth=2.5,
              label="max ball radius",
              xlabel="N_fft", ylabel="Ball radius",
              title="M4: Ball radius convergence for H[(1+cos u)/(2+sin u)]",
              xticks=(Ns, string.(Ns)))
    plot!(p1, N_ref_line, ref_line;
          color=:crimson, linestyle=:dash, linewidth=2,
          label="N^{-3} reference")
    savefig(p1, joinpath(OUTDIR, "fig_m4_1_convergence.pdf"))
    println("  Saved fig_m4_1_convergence.pdf")
end

# ===========================================================================
# Figure 2: Rigour ratio max|error|/ε_alias at each N_fft
# ===========================================================================
println("Generating fig_m4_2_aliasing_rigor.pdf...")
let
    rigour_ratios = max_err_by_N ./ eps_by_N

    p2 = bar(string.(Ns), rigour_ratios;
             color=:steelblue, legend=false,
             xlabel="N_fft", ylabel="max |error| / ε_alias",
             title="M4: Aliasing rigour ratio (must be ≤ 1)",
             ylim=(0, 1.2))
    hline!(p2, [1.0]; color=:red, linewidth=2, linestyle=:dash, label="limit = 1")
    savefig(p2, joinpath(OUTDIR, "fig_m4_2_aliasing_rigor.pdf"))
    println("  Saved fig_m4_2_aliasing_rigor.pdf")
end

# ===========================================================================
# Figure 3: Containment on a random trig poly
# ===========================================================================
println("Generating fig_m4_3_containment.pdf...")
let
    N_fft = 256
    rng = Random.MersenneTwister(42)
    r = 2 * rand(rng, 8) .- 1
    coeffs = [r[j] / j^2 for j in 1:8]

    us = [2π * (j - 1) / N_fft for j in 1:N_fft]
    G_f64_vals = [sum(coeffs[j] * cos(j * u) for j in 1:8) for u in us]
    G_vals = [Arb(v; prec=prec) for v in G_f64_vals]

    # Analytic H[G]
    HG_exact = [sum(coeffs[j] * sin(j * u) for j in 1:8) for u in us]

    # Gp_sup for p=4: |G^{(4)}|_∞ ≤ Σ |r_j| * j^2
    Gp_sup_f64 = sum(abs(r[j]) * j^2 for j in 1:8)
    Gp_sup_tp = Arb(Gp_sup_f64 + 1.0; prec=prec)

    HG_vals, ε_alias = fft_hilbert(G_vals, N_fft, p, Gp_sup_tp; prec=prec)

    HG_mids = [arb_mid(HG_vals[j]) for j in 1:N_fft]
    HG_radii = [Float64(Arblib.radref(HG_vals[j])) for j in 1:N_fft]

    p3 = plot(us, HG_exact;
              color=:black, linewidth=2, label="H[G] exact (analytic)",
              xlabel="u", ylabel="H[G](u)",
              title="M4: Containment for random trig poly G",
              xticks=(0:π/2:2π, ["0", "π/2", "π", "3π/2", "2π"]))
    scatter!(p3, us, HG_mids;
             color=:royalblue, markersize=3, markerstrokewidth=0,
             label="Arb midpoints")
    # Shaded band: midpoints ± radius
    plot!(p3, us, HG_mids .+ HG_radii;
          color=:royalblue, alpha=0.3, linewidth=0.5, linestyle=:dot, label="± radius")
    plot!(p3, us, HG_mids .- HG_radii;
          color=:royalblue, alpha=0.3, linewidth=0.5, linestyle=:dot, label=false)

    savefig(p3, joinpath(OUTDIR, "fig_m4_3_containment.pdf"))
    println("  Saved fig_m4_3_containment.pdf")
    println("  ε_alias = $(arb_mid(ε_alias))")
end

println("\nAll M4 figures saved to docs/figures/")
