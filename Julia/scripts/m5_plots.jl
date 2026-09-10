"""
m5_plots.jl — Three figures for Milestone M5 (ResidualBound, β=0).

Figure 1: Log-log plot of certified bound vs ε, with ε³ reference line.
Figure 2: Certified bound vs M_order at fixed ε=0.15, N_grid=256.
Figure 3: Scatter of ratio (certified_bound / Float64_estimate) for 10 random seeds.
"""

using BabenkoRigorous
using Arblib
using FFTW
using CairoMakie
import Random

# ---------------------------------------------------------------------------
# Float64 helper functions
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
    us_f64 = [2π * (j - 1) / N_f64 for j in 1:N_f64]
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
# Figure 1: certified bound vs ε (log-log), with ε³ reference
# ---------------------------------------------------------------------------

println("Figure 1: computing bounds for ε grid...")
epsilons = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
bounds_fig1 = Float64[]
for ε in epsilons
    y_c = [-ε^2/2, ε, ε^2]
    μ = 1.0 + ε^2
    b = residual_bound(y_c, μ, 0.0; M_order=4, N_grid=256, prec=128)
    push!(bounds_fig1, Float64(Arblib.ubound(b)))
    println("  ε=", ε, "  bound=", bounds_fig1[end])
end

fig1 = Figure(size=(600, 450))
ax1 = Axis(fig1[1, 1],
    xlabel="ε",
    ylabel="certified ‖R‖_∞ upper bound",
    title="Stokes wave residual bound vs amplitude",
    xscale=log10, yscale=log10)

scatterlines!(ax1, epsilons, bounds_fig1; label="certified bound", marker=:circle, markersize=10)

# ε³ reference line: normalise to pass through the first point
ref_scale = bounds_fig1[1] / epsilons[1]^3
eps_ref = range(epsilons[1], epsilons[end], length=50)
lines!(ax1, collect(eps_ref), ref_scale .* collect(eps_ref).^3;
       linestyle=:dash, label="ε³ reference")

axislegend(ax1; position=:lt)
save("docs/figures/fig_m5_1_stokes_convergence.pdf", fig1)
println("Saved fig_m5_1_stokes_convergence.pdf")

# ---------------------------------------------------------------------------
# Figure 2: certified bound vs M_order at fixed ε=0.15
# ---------------------------------------------------------------------------

println("Figure 2: computing bounds for M_order sweep...")
orders = [2, 3, 4, 5, 6]
ε = 0.15
y_c_fixed = [-ε^2/2, ε, ε^2]
μ_fixed = 1.0 + ε^2
bounds_fig2 = Float64[]
for M in orders
    b = residual_bound(y_c_fixed, μ_fixed, 0.0; M_order=M, N_grid=256, prec=128)
    push!(bounds_fig2, Float64(Arblib.ubound(b)))
    println("  M_order=", M, "  bound=", bounds_fig2[end])
end

fig2 = Figure(size=(600, 450))
ax2 = Axis(fig2[1, 1],
    xlabel="M_order",
    ylabel="certified ‖R‖_∞ upper bound",
    title="Residual bound vs Taylor order (ε=0.15, N_grid=256)",
    yscale=log10)

scatterlines!(ax2, orders, bounds_fig2; marker=:circle, markersize=10, color=:steelblue)
save("docs/figures/fig_m5_2_order_convergence.pdf", fig2)
println("Saved fig_m5_2_order_convergence.pdf")

# ---------------------------------------------------------------------------
# Figure 3: containment ratios for 10 random seeds
# ---------------------------------------------------------------------------

println("Figure 3: containment ratios for 10 seeds...")
seeds = 1:10
ratios_fig3 = Float64[]
for s in seeds
    rng = Random.MersenneTwister(s)
    N_modes = 8
    y_coeffs = vcat([0.0], [randn(rng) / k for k in 1:N_modes])
    max_c = maximum(abs.(y_coeffs[2:end]))
    y_coeffs = y_coeffs .* (0.1 / max_c)
    μ = 1.3
    b = residual_bound(y_coeffs, μ, 0.0; M_order=4, N_grid=256, prec=128)
    r_f64 = f64_residual_sup(y_coeffs, μ; N_f64=512)
    ratio = Float64(Arblib.ubound(b)) / max(r_f64, 1e-15)
    push!(ratios_fig3, ratio)
    println("  seed=", s, "  ratio=", ratio)
end

fig3 = Figure(size=(600, 450))
ax3 = Axis(fig3[1, 1],
    xlabel="seed index",
    ylabel="certified bound / Float64 estimate",
    title="Containment ratio: certified vs Float64 (must be ≥ 1)")

scatter!(ax3, collect(seeds), ratios_fig3; marker=:circle, markersize=12, color=:darkorange)
hlines!(ax3, [1.0]; linestyle=:dash, color=:red, label="ratio = 1")
axislegend(ax3; position=:lt)
save("docs/figures/fig_m5_3_containment.pdf", fig3)
println("Saved fig_m5_3_containment.pdf")

println("\nAll three figures saved to docs/figures/.")
println("Ratio range: [", minimum(ratios_fig3), ", ", maximum(ratios_fig3), "]")
