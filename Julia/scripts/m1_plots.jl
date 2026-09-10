using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using BabenkoRigorous
using Arblib, Plots, Printf, Random

gr()
default(linewidth = 2, framestyle = :box, size = (700, 380),
        left_margin = 4Plots.mm, bottom_margin = 4Plots.mm)

OUTDIR = joinpath(@__DIR__, "..", "docs", "figures")
mkpath(OUTDIR)

prec = 128
PREC = prec

# ── helpers ───────────────────────────────────────────────────────────────────
arb_mid(x::Arb) = Float64(Arblib.midpoint(Arb, x))
arb_rad(x::Arb) = Float64(Arblib.radius(Arb, x))

function random_trigpoly(N::Int; seed=42, prec=128)
    rng = MersenneTwister(seed)
    a = [Arb(randn(rng); prec) for _ in 0:N]
    b = [Arb(randn(rng); prec) for _ in 0:N]
    a[1] = Arb(0; prec)   # zero DC: H[const]=0 so H²≠-I on DC component
    b[1] = Arb(0; prec)   # b₀ = 0 always (sin(0·u) = 0)
    TrigPoly(a, b, prec)
end

u_grid(n=300) = range(0, 2π; length=n)

# ── Figure 1: Hilbert transform as a mode-by-mode π/2 phase shift ─────────────
# Use a pure cosine series so H[f] is the pure sine series with identical
# amplitudes — the two curves are visually "twins" shifted in phase.
let
    a_modes = [1.0, 0.6, 0.3]   # coefficients of cos(u), cos(2u), cos(3u)
    f  = cosine_series([0.0; a_modes]; prec)
    Hf = hilbert(f)
    us = u_grid()
    fv  = [arb_mid(tp_eval(f,  Arb(u; prec))) for u in us]
    Hfv = [arb_mid(tp_eval(Hf, Arb(u; prec))) for u in us]

    p = plot(us, fv;  label="f(u) = cos u + 0.6 cos 2u + 0.3 cos 3u",
             color=:royalblue)
    plot!(p, us, Hfv; label="H[f](u) = sin u + 0.6 sin 2u + 0.3 sin 3u",
          color=:crimson, linestyle=:dash)
    plot!(p; xlabel="u", ylabel="amplitude",
          title="H maps each cos(ku) → sin(ku): same amplitudes, π/2 phase shift",
          xticks=(0:π/2:2π, ["0","π/2","π","3π/2","2π"]),
          legend=:topright)
    savefig(p, joinpath(OUTDIR, "fig1_hilbert_example.pdf"))
    println("fig1 done")
end

# ── Figure 2: T1.1 — H² = −I (coefficients of f vs −H²f overlap exactly) ─────
# H is exact algebraically: coefficients of H²f + f are identically zero,
# so a log-error plot is meaningless.  Instead show that f and −H²f have
# identical coefficients, which is the content of the test.
let
    N = 6
    f = random_trigpoly(N; seed=42)
    g = hilbert(hilbert(f))          # should equal -f

    ks = 0:N
    fa = [arb_mid(f.a[k+1]) for k in ks]
    fb = [arb_mid(f.b[k+1]) for k in ks]
    ga = [arb_mid(g.a[k+1]) for k in ks]   # should equal -fa
    gb = [arb_mid(g.b[k+1]) for k in ks]   # should equal -fb

    # Cosine coefficients
    p1 = scatter(ks, fa;  label="f  (cosine)", color=:royalblue,
                 marker=:circle, markersize=7,
                 xlabel="mode k", ylabel="coefficient",
                 title="Cosine coefficients: f vs −H²f")
    scatter!(p1, ks, -ga; label="−H²f (cosine)", color=:crimson,
             marker=:xcross, markersize=7, markerstrokewidth=2)

    # Sine coefficients
    p2 = scatter(ks, fb;  label="f  (sine)", color=:royalblue,
                 marker=:circle, markersize=7,
                 xlabel="mode k", ylabel="coefficient",
                 title="Sine coefficients: f vs −H²f")
    scatter!(p2, ks, -gb; label="−H²f (sine)", color=:crimson,
             marker=:xcross, markersize=7, markerstrokewidth=2)

    p = plot(p1, p2; layout=(1,2), size=(1000,380))
    savefig(p, joinpath(OUTDIR, "fig2_hsquared.pdf"))
    println("fig2 done")
end

# ── Figure 3: T1.2 — H∂ = ∂H (pointwise comparison) ─────────────────────────
let
    f = random_trigpoly(6; seed=13)
    lhs = hilbert(deriv(f))   # H[f']
    rhs = deriv(hilbert(f))   # (Hf)'
    us  = u_grid()
    lv  = [arb_mid(tp_eval(lhs, Arb(u; prec))) for u in us]
    rv  = [arb_mid(tp_eval(rhs, Arb(u; prec))) for u in us]
    diff = abs.(lv .- rv)

    p1 = plot(us, lv; label="H[∂f]", color=:royalblue)
    plot!(p1, us, rv; label="∂H[f]", color=:crimson, linestyle=:dash)
    plot!(p1; xlabel="u", title="T1.2: H[∂f] vs ∂H[f]",
          xticks=(0:π/2:2π, ["0","π/2","π","3π/2","2π"]))

    p2 = plot(us, diff .+ 1e-40; yscale=:log10, color=:darkgreen,
              label="|H[∂f] − ∂H[f]|",
              xlabel="u", title="Pointwise difference (log scale)",
              xticks=(0:π/2:2π, ["0","π/2","π","3π/2","2π"]))

    p = plot(p1, p2; layout=(1,2), size=(1200,380))
    savefig(p, joinpath(OUTDIR, "fig3_commutation.pdf"))
    println("fig3 done")
end

# ── Figure 4: T1.4 — supnorm_bound is genuine ────────────────────────────────
let
    f = random_trigpoly(8; seed=99)
    bound = arb_mid(supnorm_bound(f))
    us  = range(0, 2π; length=1000)
    fv  = [arb_mid(tp_eval(f, Arb(u; prec))) for u in us]
    absfv = abs.(fv)

    p = plot(us, fv; label="f(u)", color=:royalblue, alpha=0.8,
             xlabel="u", title="T1.4: supnorm_bound ≥ |f(u)| at 1000 points",
             xticks=(0:π/2:2π, ["0","π/2","π","3π/2","2π"]))
    hline!(p, [bound];  label=@sprintf("supnorm\\_bound = %.4f", bound),
           color=:crimson, linestyle=:dash, linewidth=2.5)
    hline!(p, [-bound]; label="", color=:crimson, linestyle=:dash, linewidth=2.5)
    scatter!(p, us, absfv; label="", color=:royalblue, alpha=0.3,
             markersize=1, markerstrokewidth=0)
    savefig(p, joinpath(OUTDIR, "fig4_supnorm.pdf"))
    println("fig4 done")
end

# ── Figure 5: T1.4 — Arb matches Float64 baseline ────────────────────────────
# y(u) = 0.3*cos(u) + 0.15*cos(2u) + 0.07*cos(3u)
# H[y_u](u) = sum_k k*a_k*cos(ku)  (exact, analytic)
let
    # cosine_series: coeffs[1]=DC, coeffs[k+1]=cos(ku) coefficient
    a_modes = [0.3, 0.15, 0.07]          # a_1, a_2, a_3
    y_tp    = cosine_series([0.0; a_modes]; prec)
    Hyu_tp  = hilbert(deriv(y_tp))        # H[y_u] in Arb (exact, algebraic)

    n_pts = 100
    us    = range(0, 2π; length=n_pts+1)[1:n_pts]

    # Arb midpoints
    arb_vals = [arb_mid(tp_eval(Hyu_tp, Arb(u; prec))) for u in us]

    # Analytic Float64 reference: H[y_u] = sum_k k*a_k*cos(ku)
    Hyu_exact(u) = sum(k * a_modes[k] * cos(k*u) for k in 1:length(a_modes))
    f64_vals = Hyu_exact.(us)

    abserr = abs.(arb_vals .- f64_vals)

    p1 = plot(us, arb_vals; label="Arb midpoint", color=:royalblue)
    plot!(p1, us, f64_vals; label="Analytic Float64", color=:crimson,
          linestyle=:dash)
    plot!(p1; xlabel="u", title="T1.4: H[∂y] — Arb vs analytic Float64",
          xticks=(0:π/2:2π, ["0","π/2","π","3π/2","2π"]))

    p2 = plot(us, abserr .+ 1e-40; yscale=:log10, color=:darkgreen,
              label="|Arb − Float64|",
              xlabel="u", title="Absolute error (log scale)",
              xticks=(0:π/2:2π, ["0","π/2","π","3π/2","2π"]))
    hline!(p2, [1e-13]; label="1e-13 threshold", color=:gray, linestyle=:dot)

    p = plot(p1, p2; layout=(1,2), size=(1200,380))
    savefig(p, joinpath(OUTDIR, "fig5_float64_comparison.pdf"))
    println("fig5 done")
end

println("All figures saved to docs/figures/")
