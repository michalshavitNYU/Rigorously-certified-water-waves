#!/usr/bin/env julia

# Re-emit the completed phase-4 N=150 result with the DC coefficient converted
# for WaveCAP.y. This reads existing Arb dumps and never reruns the grid stages.

using Arblib
using BabenkoRigorous

include(joinpath(@__DIR__, "half_dc_certificate_tools.jl"))
using .HalfDCCertificateTools

const DEFAULT_COEFF_PATH =
    joinpath(@__DIR__, "..", "pinn_a_coeffs_N150_phase4_20260827.txt")
const CERT_BASENAME =
    "CertificatePhase4ScalarN150Phase420260827HalfDC.lean"
const CERT_MODULE =
    "WaveCAP.CertificatePhase4ScalarN150Phase420260827HalfDC"
const LAMBDA_PHASE4 = 1.1687864806007486
const BETA_PHASE4 = 0.498
const M_ORDER = 6
const PREC = 128

function load_coefficients(path::AbstractString)
    isfile(path) || error("coefficient file does not exist: $path")
    coeffs = Float64[]
    for line in eachline(path)
        value = strip(line)
        (isempty(value) || startswith(value, "#")) && continue
        push!(coeffs, parse(Float64, value))
    end
    length(coeffs) == 151 ||
        error("expected 151 coefficients, found $(length(coeffs)) in $path")
    return coeffs
end

function usage()
    println("""
usage:
  julia --project=. scripts/reemit_phase4_n150_half_dc_certificate.jl WORKDIR [OUTPUT]

WORKDIR must contain staged_bound.arbdump and staged_dlower.arbdump from the
completed run. OUTPUT defaults to WORKDIR/$CERT_BASENAME.

Environment overrides:
  PHASE4_SCALAR_N150_COEFF_PATH   source coefficient file
  N_GRID                          recorded grid size (default: 2000000)
""")
end

function main()
    if isempty(ARGS) || length(ARGS) > 2
        usage()
        exit(2)
    end

    workdir = abspath(ARGS[1])
    output = length(ARGS) == 2 ? abspath(ARGS[2]) : joinpath(workdir, CERT_BASENAME)
    coeff_path = get(ENV, "PHASE4_SCALAR_N150_COEFF_PATH", DEFAULT_COEFF_PATH)
    N_grid = parse(Int, get(ENV, "N_GRID", "2000000"))

    coeffs = load_coefficients(coeff_path)
    bound = load_arb_dump(joinpath(workdir, "staged_bound.arbdump"))
    D_lower = load_arb_dump(joinpath(workdir, "staged_dlower.arbdump"))

    text = format_half_dc_certificate(;
        y_coeffs=coeffs,
        mu=LAMBDA_PHASE4^2,
        beta=BETA_PHASE4^2,
        N_grid=N_grid,
        M_order=M_ORDER,
        prec=PREC,
        bound=bound,
        D_lower=D_lower,
        module_name=CERT_MODULE,
    )
    write_new_file(output, text)

    lean_coeffs = lean_coefficients_half_dc(coeffs)
    println("source coefficient file = $coeff_path")
    println("source a0              = $(repr(coeffs[1]))")
    println("Lean a0                = $(repr(lean_coeffs[1]))")
    println("bound upper endpoint   = $(Arblib.ubound(bound))")
    println("D lower endpoint       = $(Arblib.lbound(D_lower))")
    println("new certificate        = $output")
    println("certificate module     = $CERT_MODULE")
end

main()
