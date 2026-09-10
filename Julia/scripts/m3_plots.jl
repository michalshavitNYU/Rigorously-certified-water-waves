"""
m3_plots.jl — Figures for Milestone M3: TaylorSeries for G(u) = (1+H[y_u])/D.

Run from the BabenkoRigorous root:
  julia --project=. scripts/m3_plots.jl
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using BabenkoRigorous
using Arblib, Plots, Printf

gr()
default(linewidth = 2, framestyle = :box, size = (700, 420),
        left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)

using BabenkoRigorous: TrigPoly, hilbert, deriv, tp_eval, supnorm_bound, cosine_series
using BabenkoRigorous: TaylorSeries, sup_bound, inf_bound, dmin_bound
using BabenkoRigorous: compute_G_taylor

OUTDIR = joinpath(@__DIR__, "..", "docs", "figures")
mkpath(OUTDIR)

prec = 128
pi_arb = Arb(pi; prec=prec)

arb_mid(x::Arb) = Float64(Arblib.midpoint(Arb, x))
arb_ubound(x::Arb) = Float64(Arblib.ubound(x))
arb_lbound(x::Arb) = Float64(Arblib.lbound(x))

# ---------------------------------------------------------------------------
# Shared: evaluate G(u) from the nearest grid-point Taylor expansion (Arb)
# ---------------------------------------------------------------------------

"""
    eval_G_taylor(ts_G, u_val, M_order)

Evaluate G(u) from ts_G using the Taylor expansion from the nearest (left) grid point.
Returns Float64 midpoint.
"""
function eval_G_taylor(ts_G::TaylorSeries, u_f64::Float64)
    M_grid = ts_G.M_grid
    M = ts_G.M_order
    h_f64 = arb_mid(ts_G.h)

    # Find left grid index
    j = clamp(floor(Int, u_f64 * M_grid / (2pi)) + 1, 1, M_grid)
    u_j_f64 = (j - 1) * h_f64
    du = u_f64 - u_j_f64

    S = 0.0
    du_pow = 1.0
    fact_m = 1.0
    for m in 0:M
        if m > 0
            du_pow *= du
            fact_m *= m
        end
        S += arb_mid(ts_G.derivs[j, m+1]) * du_pow / fact_m
    end
    return S
end

"""Direct Float64 evaluation of G(u) from TrigPoly y."""
function G_f64(y::TrigPoly, u::Float64)
    u_arb = Arb(u; prec=64)
    yu = deriv(y)
    Hyu = hilbert(yu)
    A = tp_eval(yu, u_arb)
    B = Arb(1; prec=64) + tp_eval(Hyu, u_arb)
    D = sqrt(A^2 + B^2)
    return Float64(Arblib.midpoint(Arb, B / D))
end

# ===========================================================================
# fig_m3_1: G(u) for y = 0.3cos(u)+0.15cos(2u), Taylor vs direct
# ===========================================================================

println("Generating fig_m3_1...")
let
    y = cosine_series([0.0, 0.3, 0.15]; prec=prec)
    M_grid = 64
    M_order = 4
    ts_G = compute_G_taylor(y, M_grid, M_order; prec=prec)

    n_plot = 300
    us = range(0.0, 2pi; length=n_plot)

    G_taylor_vals = [eval_G_taylor(ts_G, u) for u in us]
    G_direct_vals = [G_f64(y, u) for u in us]

    p = plot(us, G_taylor_vals;
             label="G via TaylorSeries (M_order=$M_order)",
             color=:royalblue, linewidth=2.5,
             xlabel="u", ylabel="G(u)",
             title="G(u) = (1+H[y_u])/D via TaylorSeries (M3)",
             xticks=(0:pi/2:2pi, ["0", "pi/2", "pi", "3pi/2", "2pi"]))
    plot!(p, us, G_direct_vals;
          label="Direct Float64 evaluation",
          color=:crimson, linestyle=:dash, linewidth=1.8)
    savefig(p, joinpath(OUTDIR, "fig_m3_1.pdf"))
    println("  fig_m3_1 done, sup_bound(G) = ", arb_ubound(sup_bound(ts_G)))
end

# ===========================================================================
# fig_m3_2: G(u) for y = 0.1*cos(u) — Taylor vs analytic, error panel
# ===========================================================================

println("Generating fig_m3_2...")
let
    eps_val = 0.1
    y = cosine_series([0.0, eps_val]; prec=prec)
    M_grid = 128
    M_order = 4
    ts_G = compute_G_taylor(y, M_grid, M_order; prec=prec)

    n_plot = 300
    us = range(0.0, 2pi; length=n_plot)

    G_taylor_vals = [eval_G_taylor(ts_G, u) for u in us]

    # Exact G(u) = (1 + eps*cos(u)) / sqrt(eps^2*sin^2(u) + (1+eps*cos(u))^2)
    G_exact_vals = [(1 + eps_val * cos(u)) /
                    sqrt(eps_val^2 * sin(u)^2 + (1 + eps_val * cos(u))^2) for u in us]
    err_vals = abs.(G_taylor_vals .- G_exact_vals)

    p1 = plot(us, G_taylor_vals;
              label="G via TaylorSeries",
              color=:royalblue, linewidth=2.5,
              xlabel="u", ylabel="G(u)",
              title="T3.4: G for y=0.1*cos(u) -- Taylor vs analytic",
              xticks=(0:pi/2:2pi, ["0", "pi/2", "pi", "3pi/2", "2pi"]))
    plot!(p1, us, G_exact_vals;
          label="Exact G(u)",
          color=:crimson, linestyle=:dash, linewidth=1.8)

    p2 = plot(us, max.(err_vals, 1e-16);
              yscale=:log10, color=:darkorange, linewidth=2,
              label="|G_Taylor - G_exact|",
              xlabel="u", ylabel="|error|",
              title="Pointwise error (log scale)",
              xticks=(0:pi/2:2pi, ["0", "pi/2", "pi", "3pi/2", "2pi"]))

    p = plot(p1, p2; layout=(2, 1), size=(700, 560))
    savefig(p, joinpath(OUTDIR, "fig_m3_2.pdf"))
    println("  fig_m3_2 done, max error = ", maximum(err_vals))
end

# ===========================================================================
# fig_m3_3: sup_bound(ts_G) convergence in M_grid
# ===========================================================================

println("Generating fig_m3_3...")
let
    y = cosine_series([0.0, 0.3, 0.15]; prec=prec)
    M_order = 4
    M_grids_all = [16, 32, 64, 128, 256]
    M_grids = Int[]
    sb_vals = Float64[]

    for M_grid in M_grids_all
        ts_G = try
            compute_G_taylor(y, M_grid, M_order; prec=prec)
        catch e
            println("  M_grid=$M_grid: skipped ($(e.msg))")
            continue
        end
        push!(M_grids, M_grid)
        sb = sup_bound(ts_G)
        push!(sb_vals, arb_ubound(sb))
        @printf("  M_grid=%d  sup_bound=%.6f\n", M_grid, sb_vals[end])
    end

    p = plot(M_grids, sb_vals;
             xscale=:log10, marker=:circle, markersize=7,
             color=:royalblue, linewidth=2,
             label="sup_bound(G)",
             xlabel="M_grid", ylabel="sup_bound(ts_G)",
             title="sup_bound(G) convergence in M_grid  (M_order=$M_order)",
             xticks=(M_grids, string.(M_grids)))
    savefig(p, joinpath(OUTDIR, "fig_m3_3.pdf"))
    println("  fig_m3_3 done")
end

println("\nAll M3 figures saved to docs/figures/")
