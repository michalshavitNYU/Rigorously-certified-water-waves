#!/usr/bin/env julia
# stage_run.jl — run ONE stage of the sharded residual pipeline as its own
# OS process, with reduction state round-tripped exactly through disk.
# Process exit reclaims all allocator memory perfectly, so Sys.maxrss gives
# a clean per-stage peak-RSS measurement.
#
# Usage (run in order, same workdir):
#   julia --project=. scripts/stage_run.jl p1  <workdir> [shard_size] [N_grid]
#   julia --project=. scripts/stage_run.jl fft <workdir> [shard_size] [N_grid]
#   julia --project=. scripts/stage_run.jl s3  <workdir> [shard_size] [N_grid]
#
# Parallel stage 1 (workers may run concurrently, then merge):
#   julia ... stage_run.jl p1part  <workdir> <shard_size> <N_grid> <i> <W>
#   julia ... stage_run.jl p1merge <workdir> <shard_size> <N_grid> <W>
# Worker i of W processes the contiguous shard chunk _worker_shard_range(S,i,W)
# and writes a partial reduction state; p1merge folds the partials in
# ascending order into the exact sequential state (bit-identical results).
#
# Parameters otherwise match the shipped certificate (M_order=6, prec=128,
# PINN coefficients, β≠0). Stage s3 prints bound/D_lower and writes
# <workdir>/staged_{bound,dlower}.arbdump for bit-exactness comparison.

using Printf
using Arblib
using BabenkoRigorous
import BabenkoRigorous: _sharded_setup, _stage_p1!, _stage_fft!, _stage_s3,
                        _save_p1_state, _load_p1_state,
                        _save_fft_state, _load_fft_state, ShardStore,
                        _worker_shard_range, _merge_p1_states

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
    stage      = ARGS[1]
    workdir    = ARGS[2]
    shard_size = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 4096
    N_grid     = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 65536

    coeffs = load_coeffs(COEFF_PATH)
    lamb = 1.168786478294966
    beta = 0.498000000104982
    μ = lamb^2
    β = beta^2
    M_order = 6
    prec = 128
    extra_orders = 0

    mkpath(workdir)
    su = _sharded_setup(coeffs, μ, β;
                        M_order=M_order, N_grid=N_grid, prec=prec,
                        extra_orders=extra_orders, shard_size=shard_size)
    store = ShardStore(workdir, prec)
    p1_state_path  = joinpath(workdir, "state_p1.txt")
    fft_state_path = joinpath(workdir, "state_fft.txt")

    @printf("stage=%s  N_grid=%d  M_order=%d  prec=%d  shard_size=%d  threads=%d\n",
            stage, N_grid, M_order, prec, shard_size, Threads.nthreads())
    flush(stdout)

    t0 = time()
    if stage == "p1"
        st = _stage_p1!(store, su)
        _save_p1_state(p1_state_path, st)
    elseif stage == "p1part"
        i = parse(Int, ARGS[5])
        W = parse(Int, ARGS[6])
        S = length(su.shards)
        rng = _worker_shard_range(S, i, W)
        @printf("worker %d/%d: shards %d:%d of %d\n", i, W, first(rng), last(rng), S)
        st = _stage_p1!(store, su; shard_range=rng)
        _save_p1_state(joinpath(workdir, "state_p1_part$(i)of$(W).txt"), st)
    elseif stage == "p1merge"
        W = parse(Int, ARGS[5])
        parts = [_load_p1_state(joinpath(workdir, "state_p1_part$(i)of$(W).txt"), prec)
                 for i in 1:W]
        merged = _merge_p1_states(collect(parts))
        _save_p1_state(p1_state_path, merged)
    elseif stage == "fft"
        st = _load_p1_state(p1_state_path, prec)
        res = _stage_fft!(store, su, st)
        _save_fft_state(fft_state_path, res)
    elseif stage == "s3"
        res = _load_fft_state(fft_state_path, prec)
        bound = _stage_s3(store, su, res.sb_R)
        @printf("bound ubound = %.16e\n", Float64(Arblib.ubound(bound)))
        if res.D_lower !== nothing
            @printf("D_lower lbound = %.16e\n",
                    Float64(Arblib.lbound(res.D_lower)))
        end
        open(joinpath(workdir, "staged_bound.arbdump"), "w") do io
            println(io, Arblib.dump_string(bound))
        end
        if res.D_lower !== nothing
            open(joinpath(workdir, "staged_dlower.arbdump"), "w") do io
                println(io, Arblib.dump_string(res.D_lower))
            end
        end
    else
        error("unknown stage: $stage (expected p1|fft|s3)")
    end
    elapsed = time() - t0

    @printf("stage=%s elapsed = %.1f s  peak_rss = %.3f GB (Sys.maxrss)\n",
            stage, elapsed, Sys.maxrss() / 2^30)
    flush(stdout)
end

main()
