#!/usr/bin/env julia
# sweep_M6.jl — sweep M_order=6 across N_grid to find the radius-vs-runtime curve
# on the PINN solution. Each iter logs midpoint, radius, runtime to stdout.

using Printf, BabenkoRigorous, Arblib

function load_coeffs(path::String)
    a = Float64[]
    for line in eachline(path)
        s = strip(line); (isempty(s)||startswith(s,"#")) && continue
        push!(a, parse(Float64, s))
    end
    return a
end

function main()
    coeffs = load_coeffs(joinpath(@__DIR__, "..", "pinn_a_coeffs_N110.txt"))
    μ = 1.168786478294966^2
    β = 0.498000000104982^2
    @printf("[%s] sweep_M6 start: threads=%d, |coeffs|=%d, μ=%.6f, β=%.6f\n",
            now_ts(), Threads.nthreads(), length(coeffs), μ, β)
    flush(stdout)
    # Sequence of (M_order, N_grid). M=6 is the PINN winner; also include a
    # couple M=4 anchors for direct comparison with the Mac results.
    configs = [
        (4,  2048),
        (4,  8192),
        (6,  4096),
        (6,  8192),
        (6, 16384),
        (6, 32768),
        (6, 65536),
        (6, 131072),
        (6, 262144),
    ]
    for (M, N) in configs
        t0 = time()
        try
            bound = residual_bound(coeffs, μ, β; M_order=M, N_grid=N, prec=128)
            dt = time() - t0
            mid = Float64(Arblib.midref(bound))
            rad = Float64(Arblib.radref(bound))
            ub  = Float64(Arblib.ubound(bound))
            @printf("[%s] M=%d N=%7d: mid=%.4e rad=%.4e ub=%.4e [%.0fs]\n",
                    now_ts(), M, N, mid, rad, ub, dt)
        catch e
            @printf("[%s] M=%d N=%7d: FAILED in %.0fs: %s\n",
                    now_ts(), M, N, time()-t0, sprint(showerror, e))
        end
        flush(stdout)
        GC.gc()
    end
    @printf("[%s] sweep_M6 done\n", now_ts())
end

now_ts() = Base.Libc.strftime("%H:%M:%S", time())

main()
