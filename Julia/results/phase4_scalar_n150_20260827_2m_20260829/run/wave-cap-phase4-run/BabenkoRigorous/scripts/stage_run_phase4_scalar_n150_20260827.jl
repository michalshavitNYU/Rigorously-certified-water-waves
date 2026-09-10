#!/usr/bin/env julia
# stage_run_phase4_scalar_n150_20260827.jl
#
# Additive staged verifier for the phase4 2026-08-27 scalar N=150 candidate.
# It mirrors scripts/stage_run.jl, but uses the N=150 coefficient file and
# the matching phase4 lambda/beta.

using Printf
using Arblib
using BabenkoRigorous
import BabenkoRigorous: _sharded_setup, _stage_p1!, _stage_fft!, _stage_s3,
                        _save_p1_state, _load_p1_state,
                        _save_fft_state, _load_fft_state, ShardStore,
                        _worker_shard_range, _merge_p1_states

const DEFAULT_COEFF_PATH =
    joinpath(@__DIR__, "..", "pinn_a_coeffs_N150_phase4_20260827.txt")
const COEFF_PATH = get(ENV, "PHASE4_SCALAR_N150_COEFF_PATH", DEFAULT_COEFF_PATH)

const LAMBDA_PHASE4 = 1.1687864806007486
const BETA_PHASE4 = 0.498
const MU_BABENKO = LAMBDA_PHASE4^2
const BETA_BABENKO = BETA_PHASE4^2
const M_ORDER = 6
const PREC = 128
const EXTRA_ORDERS = 0
const CANDIDATE_LABEL = "phase4_20260827_scalar_N150"
const CERT_MODULE = "WaveCAP.CertificatePhase4ScalarN150Phase420260827"
const CERT_FILE = "CertificatePhase4ScalarN150Phase420260827.lean"

function load_coeffs(path::String)
    coeffs = Float64[]
    for line in eachline(path)
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        push!(coeffs, parse(Float64, s))
    end
    return coeffs
end

function maybe_write_certificate(path::String, coeffs, N_grid::Int, bound::Arb, d_lower)
    if isfile(path)
        @printf("certificate already exists; left unchanged = %s\n", path)
        return
    end

    text = format_certificate(;
        y_coeffs=coeffs,
        μ=MU_BABENKO,
        β=BETA_BABENKO,
        N_grid=N_grid,
        M_order=M_ORDER,
        prec=PREC,
        bound=bound,
        D_lower=d_lower,
        module_name=CERT_MODULE,
    )
    open(path, "w") do io
        write(io, text)
    end
    @printf("certificate = %s\n", path)
    @printf("certificate module = %s\n", CERT_MODULE)
end

function usage()
    println("""
usage:
  stage_run_phase4_scalar_n150_20260827.jl p1part  <workdir> <shard_size> <N_grid> <i> <W>
  stage_run_phase4_scalar_n150_20260827.jl p1merge <workdir> <shard_size> <N_grid> <W>
  stage_run_phase4_scalar_n150_20260827.jl fft     <workdir> <shard_size> <N_grid>
  stage_run_phase4_scalar_n150_20260827.jl s3      <workdir> <shard_size> <N_grid>
""")
end

function main()
    if length(ARGS) < 4
        usage()
        exit(2)
    end

    stage      = ARGS[1]
    workdir    = ARGS[2]
    shard_size = parse(Int, ARGS[3])
    N_grid     = parse(Int, ARGS[4])

    coeffs = load_coeffs(COEFF_PATH)
    mkpath(workdir)
    su = _sharded_setup(coeffs, MU_BABENKO, BETA_BABENKO;
                        M_order=M_ORDER, N_grid=N_grid, prec=PREC,
                        extra_orders=EXTRA_ORDERS, shard_size=shard_size)
    store = ShardStore(workdir, PREC)
    p1_state_path  = joinpath(workdir, "state_p1.txt")
    fft_state_path = joinpath(workdir, "state_fft.txt")

    @printf("candidate=%s\n", CANDIDATE_LABEL)
    @printf("coeff_path=%s\n", COEFF_PATH)
    @printf("stage=%s  coeff_count=%d  lambda=%.17g  beta=%.17g\n",
            stage, length(coeffs), LAMBDA_PHASE4, BETA_PHASE4)
    @printf("N_grid=%d  M_order=%d  prec=%d  shard_size=%d  threads=%d\n",
            N_grid, M_ORDER, PREC, shard_size, Threads.nthreads())
    flush(stdout)

    t0 = time()
    if stage == "p1"
        st = _stage_p1!(store, su)
        _save_p1_state(p1_state_path, st)
    elseif stage == "p1part"
        length(ARGS) >= 6 || error("p1part requires <i> <W>")
        i = parse(Int, ARGS[5])
        W = parse(Int, ARGS[6])
        S = length(su.shards)
        rng = _worker_shard_range(S, i, W)
        @printf("worker %d/%d: shards %d:%d of %d\n", i, W, first(rng), last(rng), S)
        st = _stage_p1!(store, su; shard_range=rng)
        _save_p1_state(joinpath(workdir, "state_p1_part$(i)of$(W).txt"), st)
    elseif stage == "p1merge"
        length(ARGS) >= 5 || error("p1merge requires <W>")
        W = parse(Int, ARGS[5])
        parts = [_load_p1_state(joinpath(workdir, "state_p1_part$(i)of$(W).txt"), PREC)
                 for i in 1:W]
        merged = _merge_p1_states(collect(parts))
        _save_p1_state(p1_state_path, merged)
    elseif stage == "fft"
        st = _load_p1_state(p1_state_path, PREC)
        res = _stage_fft!(store, su, st)
        _save_fft_state(fft_state_path, res)
    elseif stage == "s3"
        res = _load_fft_state(fft_state_path, PREC)
        bound = _stage_s3(store, su, res.sb_R)
        ub = Float64(Arblib.ubound(bound))
        lb = Float64(Arblib.lbound(bound))
        mid = Float64(Arblib.midref(bound))
        rad = Float64(Arblib.radref(bound))
        @printf("bound midpoint = %+.16e\n", mid)
        @printf("bound radius   = %.16e\n", rad)
        @printf("bound interval = [%+.16e, %+.16e]\n", lb, ub)
        @printf("bound ubound   = %.16e\n", ub)
        if res.D_lower !== nothing
            @printf("D_lower lbound = %.16e\n", Float64(Arblib.lbound(res.D_lower)))
        end
        open(joinpath(workdir, "staged_bound.arbdump"), "w") do io
            println(io, Arblib.dump_string(bound))
        end
        if res.D_lower !== nothing
            open(joinpath(workdir, "staged_dlower.arbdump"), "w") do io
                println(io, Arblib.dump_string(res.D_lower))
            end
        end
        maybe_write_certificate(joinpath(workdir, CERT_FILE), coeffs, N_grid, bound, res.D_lower)
    else
        error("unknown stage: $stage (expected p1|p1part|p1merge|fft|s3)")
    end
    elapsed = time() - t0

    @printf("stage=%s elapsed = %.1f s  peak_rss = %.3f GB (Sys.maxrss)\n",
            stage, elapsed, Sys.maxrss() / 2^30)
    flush(stdout)
end

main()
