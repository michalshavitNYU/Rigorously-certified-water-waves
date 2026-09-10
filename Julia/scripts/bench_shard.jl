#!/usr/bin/env julia
# bench_shard.jl — run the shipped-certificate residual bound (M_order=6,
# N_grid=65536, prec=128, β≠0) either monolithically or sharded, print the
# exact bound/D_lower, and dump them for cross-mode bit-exactness comparison.
#
# Usage:
#   julia --project=. scripts/bench_shard.jl mono   [out_prefix]
#   julia --project=. scripts/bench_shard.jl shard  [out_prefix] [shard_size]
#
# Wrap the whole invocation in `/usr/bin/time -v` to capture peak RSS.

using Printf
using Arblib
using BabenkoRigorous
import BabenkoRigorous: _residual_bound_full

const COEFF_PATH = joinpath(@__DIR__, "..", "pinn_a_coeffs_N110.txt")

function load_coeffs(path::String)
    coeffs = Float64[]
    for line in eachline(path)
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        push!(coeffs, parse(Float64, s))
    end
    return coeffs
end

function main()
    mode = length(ARGS) >= 1 ? ARGS[1] : "mono"
    out_prefix = length(ARGS) >= 2 ? ARGS[2] : "bench_$(mode)"
    shard_size = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 4096

    # Shipped-certificate parameters (WaveCAP/Certificate.lean).
    coeffs       = load_coeffs(COEFF_PATH)
    lamb         = 1.168786478294966
    beta         = 0.498000000104982
    μ            = lamb^2
    β            = beta^2
    M_order      = 6
    N_grid       = 65536
    prec         = 128
    extra_orders = 0

    @printf("mode=%s  N_grid=%d  M_order=%d  prec=%d  shard_size=%s  threads=%d\n",
            mode, N_grid, M_order, prec,
            mode == "shard" ? string(shard_size) : "-", Threads.nthreads())
    flush(stdout)

    t0 = time()
    local bound, D_lower
    if mode == "mono"
        full = _residual_bound_full(coeffs, μ, β;
                                    M_order=M_order, N_grid=N_grid,
                                    prec=prec, extra_orders=extra_orders)
        bound   = full.bound
        D_lower = full.D_lower
    elseif mode == "shard"
        workdir = mktempdir(; prefix="shard_bench_")
        sh = residual_bound_sharded(coeffs, μ, β;
                                    M_order=M_order, N_grid=N_grid,
                                    prec=prec, extra_orders=extra_orders,
                                    shard_size=shard_size, workdir=workdir)
        bound   = sh.bound
        D_lower = sh.D_lower
        # Clean the shard store (its disk footprint is not what we measure).
        rm(workdir; recursive=true, force=true)
    else
        error("unknown mode: $mode (expected mono|shard)")
    end
    elapsed = time() - t0

    peak_rss_gb = Sys.maxrss() / 2^30
    @printf("elapsed = %.1f s\n", elapsed)
    @printf("peak_rss = %.3f GB  (Sys.maxrss)\n", peak_rss_gb)
    @printf("bound ubound = %.16e\n", Float64(Arblib.ubound(bound)))
    @printf("D_lower lbound = %.16e\n", Float64(Arblib.lbound(D_lower)))
    flush(stdout)

    # Exact dumps for bit-exact cross-mode comparison.
    open(out_prefix * "_bound.arbdump", "w") do io
        println(io, Arblib.dump_string(bound))
    end
    open(out_prefix * "_dlower.arbdump", "w") do io
        println(io, Arblib.dump_string(D_lower))
    end
    println("wrote ", out_prefix, "_{bound,dlower}.arbdump")
end

main()
