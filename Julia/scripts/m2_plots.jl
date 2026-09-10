using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using BabenkoRigorous
using Arblib, Plots, Printf

gr()
default(linewidth = 2, framestyle = :box, size = (700, 380),
        left_margin = 4Plots.mm, bottom_margin = 4Plots.mm)

OUTDIR = joinpath(@__DIR__, "..", "docs", "figures")
mkpath(OUTDIR)

using BabenkoRigorous: TrigPoly, degree, hilbert, deriv, tp_eval, supnorm_bound,
                       cosine_series, add_const
using BabenkoRigorous: TaylorSeries, sup_bound, inf_bound, deriv_series,
                       inv_series, sqrt_series, dmin_bound

prec = 128

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

arb_mid(x::Arb) = Float64(Arblib.midpoint(Arb, x))
arb_ubound(x::Arb) = Float64(Arblib.ubound(x))
arb_lbound(x::Arb) = Float64(Arblib.lbound(x))

"""Build TaylorSeries for sin at M_grid equispaced points with Taylor order M_order."""
function make_sin_series(M_grid::Int, M_order::Int; p::Int=128)::TaylorSeries
    pi_arb = Arb(pi; prec=p)
    h = Arb(2; prec=p) * pi_arb / Arb(M_grid; prec=p)
    derivs = Matrix{Arb}(undef, M_grid, M_order + 1)
    for j in 1:M_grid
        u_j = Arb(2; prec=p) * pi_arb * Arb(j - 1; prec=p) / Arb(M_grid; prec=p)
        for m in 0:M_order
            r = m % 4
            derivs[j, m+1] = r == 0 ? sin(u_j) :
                              r == 1 ? cos(u_j) :
                              r == 2 ? -sin(u_j) : -cos(u_j)
        end
    end
    sup_bounds = [Arb(1; prec=p) for _ in 0:(M_order + 1)]
    TaylorSeries(derivs, sup_bounds, M_order, M_grid, h, p)
end

# ---------------------------------------------------------------------------
# fig_m2_1: Taylor approximation of sin(u) on subintervals (M_grid=32, M_order=4)
# ---------------------------------------------------------------------------

let
    M_grid = 32
    M_order = 4
    ts = make_sin_series(M_grid, M_order; p=prec)

    pi_arb = Arb(pi; prec=prec)
    h = ts.h
    h_f64 = arb_mid(h)
    M = M_order

    # Precompute remainder
    rem_f64 = arb_ubound(ts.sup_bounds[end] * h^(M+1) / Arb(factorial(M+1); prec=prec))

    # Dense set of u values for plotting
    n_plot = 600
    us = range(0, 2π; length=n_plot)

    true_vals = sin.(us)
    taylor_vals = zeros(n_plot)
    ub_vals     = zeros(n_plot)
    lb_vals     = zeros(n_plot)

    for (i, u_float) in enumerate(us)
        # Find left grid point
        j = clamp(floor(Int, u_float * M_grid / (2π)) + 1, 1, M_grid)
        u_j = (j - 1) * h_f64
        du = u_float - u_j

        # Taylor sum
        S = 0.0
        du_pow = 1.0
        fact_m = 1.0
        for m in 0:M
            if m > 0
                du_pow *= du
                fact_m *= m
            end
            S += arb_mid(ts.derivs[j, m+1]) * du_pow / fact_m
        end
        taylor_vals[i] = S
        ub_vals[i] = S + rem_f64
        lb_vals[i] = S - rem_f64
    end

    # Grid point markers
    grid_us = [(j - 1) * h_f64 for j in 1:M_grid]
    grid_sinvals = sin.(grid_us)

    p = plot(us, true_vals; label="sin(u)", color=:royalblue, linewidth=2.5,
             xlabel="u", ylabel="f(u)",
             title="Taylor series approximation of sin(u) on subintervals",
             xticks=(0:π/2:2π, ["0", "π/2", "π", "3π/2", "2π"]))
    plot!(p, us, taylor_vals; label="Taylor approx (M_order=$M_order)",
          color=:crimson, linestyle=:dash, linewidth=1.8)
    plot!(p, us, ub_vals; fillrange=lb_vals, fillalpha=0.25, fillcolor=:orange,
          label="Remainder band (±R)", color=:orange, linewidth=0.5, linealpha=0.6)
    scatter!(p, grid_us, grid_sinvals; label="Grid points u_j",
             color=:black, markersize=4, markerstrokewidth=0)
    savefig(p, joinpath(OUTDIR, "fig_m2_1.pdf"))
    println("fig_m2_1 done")
end

# ---------------------------------------------------------------------------
# fig_m2_2: Convergence — sup_bound overestimate vs Taylor order (log scale)
# ---------------------------------------------------------------------------

let
    M_grid = 32
    orders = [1, 2, 3, 4, 5]
    overestimates = Float64[]
    for M_order in orders
        ts = make_sin_series(M_grid, M_order; p=prec)
        sb = sup_bound(ts)
        overestimate = arb_ubound(sb) - 1.0   # true ‖sin‖_∞ = 1
        push!(overestimates, max(overestimate, 1e-16))  # floor for log scale
    end

    p = plot(orders, overestimates; yscale=:log10, marker=:circle,
             markersize=7, color=:royalblue, linewidth=2,
             xlabel="Taylor order M_order", ylabel="sup_bound(ts_sin) − 1",
             title="sup_bound overestimate vs Taylor order  (M_grid = $M_grid)",
             label="Overestimate above true ‖sin‖_∞ = 1",
             xticks=orders)
    savefig(p, joinpath(OUTDIR, "fig_m2_2.pdf"))
    println("fig_m2_2 done")
end

# ---------------------------------------------------------------------------
# fig_m2_3: D_lower convergence for y=0.1·cos(u) vs M_grid
# ---------------------------------------------------------------------------

let
    M_order = 4
    M_grids = [32, 64, 128, 256, 512]
    D_lower_vals = Float64[]

    for M_grid in M_grids
        y = TrigPoly([0.0, 0.1], [0.0, 0.0]; prec=256)
        D_lower, _ = dmin_bound(y, M_grid, M_order; prec=256)
        push!(D_lower_vals, arb_lbound(D_lower))
        @printf("  M_grid=%d  D_lower=%.6f\n", M_grid, D_lower_vals[end])
    end

    # True D_min = 0.9 (at u=π)
    D_true = 0.9

    # Lipschitz first-order bound: D(u) ≥ min_j D(u_j) - L*h
    # For y=0.1*cos(u): D'(u) = (y_u * y_uu + B * B') / D
    # where y_u = -0.1*sin(u), y_uu = -0.1*cos(u), B = 1+0.1*cos(u), B'=-0.1*sin(u)
    # We bound |D'| ≤ |y_uu| + |B'| ≤ 0.1 + 0.1 = 0.2 (very rough)
    # First-order bound at each M_grid: D_true - 0.2 * (2π/M_grid)
    D_lip = [D_true - 0.2 * (2π / M_grid) for M_grid in M_grids]

    p = plot(M_grids, D_lower_vals; xscale=:log10, marker=:circle,
             markersize=7, color=:royalblue, linewidth=2,
             label="Taylor inf_bound (M_order=$M_order)",
             xlabel="M_grid", ylabel="D_lower",
             title="D_lower convergence for y = 0.1·cos(u)  (true D_min = 0.9)")
    hline!(p, [D_true]; label="True D_min = 0.9", color=:black,
           linestyle=:dash, linewidth=1.5)
    plot!(p, M_grids, D_lip; label="First-order Lipschitz bound",
          color=:crimson, linestyle=:dot, linewidth=2, marker=:diamond, markersize=5)
    savefig(p, joinpath(OUTDIR, "fig_m2_3.pdf"))
    println("fig_m2_3 done")
end

println("\nAll M3 figures saved to docs/figures/")
