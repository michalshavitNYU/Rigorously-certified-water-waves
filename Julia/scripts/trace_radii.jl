#!/usr/bin/env julia
# trace_radii.jl — walk the residual_bound chain step by step and compare
# each TaylorSeries's `sup_bounds` midpoints against FFT-computed
# ‖f^(m)‖∞ ground truth, to find the first place looseness enters.
#
# Output: for each (named-step, derivative order m): truth, Arb midpoint, ratio.
# Ratio = Arb_midpoint / Float64_truth; ~1 = tight, >>1 = loose.

using FFTW, Printf
using BabenkoRigorous
using Arblib

const COEFF_PATH = joinpath(@__DIR__, "..", "pinn_a_coeffs_N110.txt")

function load_coeffs(path)
    a = Float64[]
    for line in eachline(path); s=strip(line); (isempty(s)||startswith(s,"#"))&&continue; push!(a, parse(Float64,s)); end
    return a
end

function f64_hilbert(g)
    N=length(g); F=rfft(g); H=copy(F)
    H[1]=0im; for k in 1:(N÷2-1); H[k+1]=F[k+1]*(-im); end; H[N÷2+1]=0im
    return real(irfft(H,N))
end
function f64_deriv(g)
    N=length(g); F=rfft(g); for k in 0:(N÷2); F[k+1]*=(im*k); end; F[N÷2+1]=0im
    return real(irfft(F,N))
end

"""Build FFT-truth derivatives of `f` up to order M (returns vector of vectors)."""
function f64_derivs(f::Vector{Float64}, M::Int)
    out = Vector{Vector{Float64}}(undef, M+1)
    out[1] = f
    for m in 1:M
        out[m+1] = f64_deriv(out[m])
    end
    return out
end

f64_sup(v) = maximum(abs.(v))

"""Compare ts.sup_bounds[m+1] against ‖f^(m)‖∞ from FFT truth, AND report
max per-grid Arb radius at each derivative level. The per-grid radii capture
discretization / aliasing error injected via add_error! in fft_hilbert; the
sup_bounds capture the formula-based bound looseness (Leibniz, Faà di Bruno,
Fourier-tail)."""
function report(label, ts, truth_derivs)
    M_avail = length(ts.sup_bounds) - 1   # ts holds sup_bounds[0..M+1]
    M_truth = length(truth_derivs) - 1
    M_sup   = min(M_avail, M_truth)
    @printf("\n--- %s  (M_sup=%d, M_grid=%d) ---\n", label, M_sup, ts.M_order)
    @printf("  %-3s   %-12s   %-12s   %-10s   %-12s\n",
            "m", "FFT truth", "Arb sup_mid", "sup/truth", "max per-grid radius")
    for m in 0:M_sup
        truth = f64_sup(truth_derivs[m+1])
        sup_mid = Float64(Arblib.midref(ts.sup_bounds[m+1]))
        ratio = sup_mid / truth
        # per-grid radius — only available for m ≤ ts.M_order
        if m <= ts.M_order
            max_rad = maximum(Float64(Arblib.radref(ts.derivs[j, m+1])) for j in 1:ts.M_grid)
            @printf("  m=%d   %.4e     %.4e     %.3e   %.4e\n", m, truth, sup_mid, ratio, max_rad)
        else
            @printf("  m=%d   %.4e     %.4e     %.3e   (no derivs)\n", m, truth, sup_mid, ratio)
        end
    end
end

# -------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------
coeffs = load_coeffs(COEFF_PATH)
prec    = 128
M_order = 4
N_grid  = 2048
M_h     = M_order + 2
M1      = M_order + 1
M3      = M_order + 3

# FFT truth at high-res grid
N_fft = 8192
us    = [2π*(j-1)/N_fft for j in 1:N_fft]
y_f   = zeros(N_fft)
for j in 1:N_fft
    s = coeffs[1]/2.0
    for k in 1:(length(coeffs)-1); s += coeffs[k+1]*cos(k*us[j]); end
    y_f[j] = s
end
yu_f  = f64_deriv(y_f)
Hyu_f = f64_hilbert(yu_f)
y_Hyu_f  = y_f .* Hyu_f
y_yu_f   = y_f .* yu_f
Hy_yu_f  = f64_hilbert(y_yu_f)
D_f      = sqrt.(yu_f.^2 .+ (1 .+ Hyu_f).^2)
B_f      = 1 .+ Hyu_f
A_f      = yu_f
D2_f     = A_f.^2 .+ B_f.^2
invD_f   = 1 ./ D_f
G_f      = (1 .+ Hyu_f) ./ D_f
yuD_f    = yu_f ./ D_f
Gu_f     = f64_deriv(G_f)
C1_f     = f64_deriv(yuD_f)
HGu_f    = f64_hilbert(Gu_f)

# -------------------------------------------------------------------
# Build the Arb chain, mirroring _residual_bound_full
# -------------------------------------------------------------------
println("============================================================")
println("Tracing radii / sup_bounds tightness through residual pipeline")
println("M_order=$M_order, N_grid=$N_grid, prec=$prec")
println("Compare each step's ts.sup_bounds[m+1] midpoint vs FFT-truth ‖f^(m)‖∞.")
println("============================================================")

y   = cosine_series(coeffs; prec=prec)
yu  = BabenkoRigorous.deriv(y)
Hyu = BabenkoRigorous.hilbert(yu)

ts_y_h   = BabenkoRigorous.trigpoly_to_taylor(y,   N_grid, M_h;     prec=prec)
ts_yu_h  = BabenkoRigorous.trigpoly_to_taylor(yu,  N_grid, M_h;     prec=prec)
ts_Hyu   = BabenkoRigorous.trigpoly_to_taylor(Hyu, N_grid, M_order; prec=prec)

report("ts_y_h    (y)",   ts_y_h,   f64_derivs(y_f,   M_h+1))
report("ts_yu_h   (y_u)", ts_yu_h,  f64_derivs(yu_f,  M_h+1))
report("ts_Hyu    (H[y_u])", ts_Hyu, f64_derivs(Hyu_f, M_order+1))

ts_y      = BabenkoRigorous.truncate_order(ts_y_h,  M_order)
ts_yu     = BabenkoRigorous.truncate_order(ts_yu_h, M_order)
ts_y_Hyu  = ts_y * ts_Hyu
ts_y_yu_h = ts_y_h * ts_yu_h
ts_y_yu   = BabenkoRigorous.truncate_order(ts_y_yu_h, M_order)

report("ts_y_Hyu  (y · H[y_u])",  ts_y_Hyu,  f64_derivs(y_Hyu_f, M_order+1))
report("ts_y_yu_h (y · y_u at M_h)", ts_y_yu_h, f64_derivs(y_yu_f, M_h+1))

Gpp_bound_yyu  = ts_y_yu_h.sup_bounds[M_order + 3]
Gppp_bound_yyu = ts_y_yu_h.sup_bounds[end]
ts_Hyyu = BabenkoRigorous.fft_hilbert_series(ts_y_yu, N_grid;
                                             Gpp_bound=Gpp_bound_yyu,
                                             Gppp_bound=Gppp_bound_yyu,
                                             prec=prec)
report("ts_Hyyu   (H[y·y_u])", ts_Hyyu, f64_derivs(Hy_yu_f, M_order+1))

# ---- capillary branch ----
D_lower, ts_D3 = BabenkoRigorous.dmin_bound(y, N_grid, M3; prec=prec)
report("ts_D3     (D, from sqrt_series)", ts_D3, f64_derivs(D_f, M3+1))
@printf("\nD_lower: mid=%.4e  rad=%.2e   (FFT truth min D = %.4e)\n",
        Float64(Arblib.midref(D_lower)), Float64(Arblib.radref(D_lower)), minimum(D_f))

ts_invD3 = BabenkoRigorous.inv_series(ts_D3, D_lower)
report("ts_invD3  (1/D, from inv_series)", ts_invD3, f64_derivs(invD_f, M3+1))

# ts_B3 (1 + H[y_u]) at order M3 — direct construction in ResidualBound.jl
Hyu_d = Vector{BabenkoRigorous.TrigPoly}(undef, M3 + 2)
Hyu_d[1] = Hyu
for m in 1:(M3 + 1); Hyu_d[m+1] = BabenkoRigorous.deriv(Hyu_d[m]); end
sup_bounds_B3 = Vector{Arb}(undef, M3 + 2)
sup_bounds_B3[1] = Arb(1; prec=prec) + BabenkoRigorous.supnorm_bound(Hyu_d[1])
for m in 1:(M3 + 1)
    sup_bounds_B3[m+1] = BabenkoRigorous.supnorm_bound(Hyu_d[m+1])
end
pi_arb = Arb(pi; prec=prec)
h_arb  = Arb(2; prec=prec) * pi_arb / Arb(N_grid; prec=prec)
derivs_B3 = Matrix{Arb}(undef, N_grid, M3 + 1)
for j in 1:N_grid
    u_j = Arb(2; prec=prec) * pi_arb * Arb(j - 1; prec=prec) / Arb(N_grid; prec=prec)
    derivs_B3[j, 1] = Arb(1; prec=prec) + BabenkoRigorous.tp_eval(Hyu_d[1], u_j)
    for m in 1:M3
        derivs_B3[j, m+1] = BabenkoRigorous.tp_eval(Hyu_d[m+1], u_j)
    end
end
ts_B3 = BabenkoRigorous.TaylorSeries(derivs_B3, sup_bounds_B3, M3, N_grid, h_arb, prec)
report("ts_B3     (1 + H[y_u])", ts_B3, f64_derivs(B_f, M3+1))

ts_G3 = ts_B3 * ts_invD3
report("ts_G3     (G = B/D, ts_B3 * ts_invD3)", ts_G3, f64_derivs(G_f, M3+1))

# next: ts_C1 = ∂_u(y_u/D), ts_Gu = ∂_u G
ts_yu_M3   = BabenkoRigorous.trigpoly_to_taylor(yu, N_grid, M3; prec=prec)
ts_yu_M1   = BabenkoRigorous.truncate_order(ts_yu_M3, M1)
ts_invD_M1 = BabenkoRigorous.truncate_order(ts_invD3, M1)
ts_yuD_M1  = ts_yu_M1 * ts_invD_M1
report("ts_yuD_M1 (y_u/D at M1)", ts_yuD_M1, f64_derivs(yuD_f, M1+1))

ts_C1   = BabenkoRigorous.deriv_series(ts_yuD_M1)
report("ts_C1     (∂_u(y_u/D))", ts_C1, f64_derivs(C1_f, M_order+1))

ts_G_M1 = BabenkoRigorous.truncate_order(ts_G3, M1)
ts_Gu   = BabenkoRigorous.deriv_series(ts_G_M1)
report("ts_Gu     (∂_u G)", ts_Gu, f64_derivs(Gu_f, M_order+1))

Gpp_bound_Gu  = ts_G3.sup_bounds[M_order + 4]
Gppp_bound_Gu = ts_G3.sup_bounds[M_order + 5]
@printf("\nFor H[∂_u G] via Path B:\n")
@printf("  Gpp_bound_Gu (mid)  = %.4e   (FFT truth ‖G^(M_order+3)‖∞ ≈ %.4e)\n",
        Float64(Arblib.midref(Gpp_bound_Gu)), f64_sup(f64_derivs(G_f, M3+1)[M_order+4]))
@printf("  Gppp_bound_Gu (mid) = %.4e   (FFT truth ‖G^(M_order+4)‖∞ ≈ %.4e)\n",
        Float64(Arblib.midref(Gppp_bound_Gu)), f64_sup(f64_derivs(G_f, M3+1)[M_order+5]))

ts_HGu = BabenkoRigorous.fft_hilbert_series(ts_Gu, N_grid;
                                             Gpp_bound=Gpp_bound_Gu,
                                             Gppp_bound=Gppp_bound_Gu,
                                             prec=prec)
report("ts_HGu    (H[∂_u G])", ts_HGu, f64_derivs(HGu_f, M_order+1))

println("\n============================================================")
println("Done. Look for the first 'LOOSE' tag — that's where it enters.")
println("============================================================")
