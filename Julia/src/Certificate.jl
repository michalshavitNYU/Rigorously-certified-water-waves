"""
Certificate.jl — Emit a Lean-readable certificate for the Babenko residual bound.

Milestone M7: serialize the *final scalar outputs* of the Arb-certified
interval-arithmetic pipeline to a self-contained Lean 4 module.

# Trust boundary

This file produces what the Lean proof treats as **trusted input** from the
Julia/Arblib side.  Lean's role is to verify the *universal analytic lemmas*
that justify the interval-arithmetic chain:

  - aliasing remainder: 4·Cp / ((p−1)(N/2−1)^{p−1})   (integral test)
  - Fourier-tail π²/3 bound for ‖H[g]‖_∞ from ‖g''‖_∞
  - grid-to-supnorm Taylor remainder
  - Leibniz on per-derivative supnorms
  - Faà di Bruno (Bell polynomial) for `1/f` and `√f`
  - integration by parts for Fourier-coefficient decay

These are universal — independent of any specific `a_cert`.  Once they are
proved in Lean, the Arb-arithmetic chain that produced ε₀_cert is justified
by the trust chain (Lean-verified lemmas + correctness of Arb interval ops),
**without Lean redoing the per-grid-point arithmetic**.

# What's in the certificate

Inputs (Lean cannot derive — these define the problem):
  N_cert      :: ℕ            number of cosine modes
  a_cert      :: Fin N → ℚ    cosine coefficients
  μ_cert      :: ℚ            wave speed parameter
  β_cert      :: ℚ            surface-tension parameter

Algorithm parameters (record what Julia ran with):
  N_fft_cert  :: ℕ            DFT/Taylor grid size
  M_order_cert :: ℕ           Taylor expansion order

Arb-certified outputs (Lean trusts; the trust chain above justifies them):
  ε₀_cert     :: ℚ            ‖R‖_∞ ≤ ε₀_cert
  DLower_cert :: ℚ            (β≠0 only)  D(u) ≥ DLower_cert > 0

Anything else (sup_bounds vectors, Gpp/Gppp, per-grid-point matrices,
intermediate Taylor data) is *Julia's internal scratch* — not in the
certificate.  Lean does not need it: those quantities are intermediate
values inside the Arb-certified routines, justified by the universal
lemmas Lean has verified.
"""

using Arblib

# ---------------------------------------------------------------------------
# Arf → Rational{BigInt} conversion
# ---------------------------------------------------------------------------

"""
Convert an `Arf` (the exact dyadic endpoint of an Arb ball) to an exact
`Rational{BigInt}` via a high-precision `BigFloat` round-trip. The
intermediate `BigFloat` is set to `bits` bits of precision (default 2048),
which is far above the working Arb precision used anywhere in this package
(usually 128 bits); at that level `BigFloat(arf)` is the exact value and
`Rational{BigInt}(bf)` is the exact dyadic rational `m / 2^k`.

Special cases: zero → `0//1`, ±∞ or NaN raise an error.
"""
function _arf_to_rational(arf::Arblib.ArfOrRef; bits::Int=2048)::Rational{BigInt}
    if !isfinite(arf)
        error("_arf_to_rational: non-finite Arf ($arf)")
    end
    bf = setprecision(BigFloat, bits) do
        BigFloat(arf)
    end
    return Rational{BigInt}(bf)
end

"""
    arb_upper_rational(x::Arb) -> Rational{BigInt}

A rational `r` satisfying `r ≥ ubound(x)`. Returns the exact dyadic
rational corresponding to the Arf upper endpoint of `x` (which by Arb
semantics is itself an upper bound on the true value enclosed by `x`).
Use for upper bounds (e.g. ε₀_cert).
"""
function arb_upper_rational(x::Arb)::Rational{BigInt}
    return _arf_to_rational(Arblib.ubound(x))
end

"""
    arb_lower_rational(x::Arb) -> Rational{BigInt}

A rational `r` satisfying `r ≤ lbound(x)`. Returns the exact dyadic
rational of the Arf lower endpoint. Use for lower bounds (e.g. DLower_cert).
"""
function arb_lower_rational(x::Arb)::Rational{BigInt}
    return _arf_to_rational(Arblib.lbound(x))
end

"""
    arb_midpoint_rational(x::Arb) -> Rational{BigInt}

The exact dyadic rational at the midpoint of the Arb ball `x`. **Not** a
certified bound — only use for inputs the caller has independently certified
(e.g. Float64-derived `y_coeffs`).
"""
function arb_midpoint_rational(x::Arb)::Rational{BigInt}
    midref = Arblib.midref(x)
    arf = Arblib.Arf(midref; prec=Arblib.precision(x))
    return _arf_to_rational(arf)
end

# ---------------------------------------------------------------------------
# Lean-formatted rational printing
# ---------------------------------------------------------------------------

"""
Format a `Rational{BigInt}` as a Lean `ℚ` literal.

Conventions:
  - 0 → "0"
  - integer (denominator 1) → just the numerator
  - other → "(p : ℚ) / q" with explicit type ascription so Lean's parser
    doesn't fall back to `ℕ` division
"""
function _format_rat(r::Rational{BigInt})::String
    p, q = numerator(r), denominator(r)
    if p == 0
        return "0"
    elseif q == 1
        return string(p)
    else
        return "((" * string(p) * " : ℚ) / " * string(q) * ")"
    end
end

# ---------------------------------------------------------------------------
# Helper: emit a Fin n → ℚ definition
# ---------------------------------------------------------------------------

function _emit_fin_rat_def(io::IO, name::AbstractString, rats::Vector{Rational{BigInt}},
                           n_expected::Int, doc::AbstractString)
    @assert length(rats) == n_expected "Vector length $(length(rats)) ≠ expected $n_expected for $name"
    if !isempty(doc)
        println(io, "/-- ", doc, " -/")
    end
    print(io, "def ", name, " : Fin ", n_expected, " → ℚ := ![")
    if n_expected == 0
        println(io, "]")
    else
        if n_expected <= 3
            print(io, " ", join(_format_rat.(rats), ", "), " ]")
            println(io)
        else
            println(io)
            for (i, r) in enumerate(rats)
                trail = i < length(rats) ? "," : ""
                println(io, "  ", _format_rat(r), trail)
            end
            println(io, "]")
        end
    end
end

# ---------------------------------------------------------------------------
# Main API: emit_certificate and write_certificate
# ---------------------------------------------------------------------------

"""
    format_certificate(; y_coeffs, μ, β, N_grid, M_order, prec,
                        bound::Arb, D_lower::Union{Arb,Nothing},
                        module_name="WaveCAP.Certificate") -> String

Format an **already-computed** certified `bound` (⇒ `ε₀_cert`) and `D_lower`
(⇒ `DLower_cert`, β≠0 only) into the self-contained Lean 4 certificate
module text. Pure formatting — the caller supplies the Arb scalars, so the
same formatter serves the monolithic (`emit_certificate`) and sharded
(`emit_certificate_sharded`) compute paths. Deterministic: byte-identical
for identical arguments.

**Phase-6 format.** The certificate ships only the inputs and the two final
scalars. The trust anchors `cert_residual_le` / `cert_D_ge` live in
`WaveCAP/CertificateAxioms.lean` as *derived theorems* (built from the
atomic `cert_arb_*` axioms), so — unlike the pre-Phase-6 emitter — this file
emits **no `axiom` declarations**; declaring them here would collide with
those theorems.
"""
function format_certificate(; y_coeffs::Vector{<:Real}, μ::Real, β::Real,
                            N_grid::Int, M_order::Int, prec::Int,
                            bound::Arb, D_lower::Union{Arb,Nothing},
                            module_name::AbstractString="WaveCAP.Certificate")::String
    N_cert = length(y_coeffs)
    a_cert_rats = [Rational{BigInt}(y_coeffs[k]) for k in 1:N_cert]
    μ_rat = Rational{BigInt}(μ)
    β_rat = Rational{BigInt}(β)
    ε₀_rat = arb_upper_rational(bound)
    has_capillary = (β != 0.0)
    has_capillary && D_lower === nothing &&
        error("format_certificate: β ≠ 0 requires a D_lower")
    DLower_rat = has_capillary ? arb_lower_rational(D_lower) : nothing

    io = IOBuffer()

    println(io, "-- Auto-generated by BabenkoRigorous.jl/Certificate.jl")
    println(io, "-- DO NOT EDIT BY HAND. Re-emit by running `write_certificate(...)`")
    println(io, "-- (small N) or `write_certificate_sharded(...)` (large N).")
    println(io, "--")
    println(io, "-- Inputs:")
    println(io, "--   y_coeffs = ", y_coeffs)
    println(io, "--   μ        = ", μ)
    println(io, "--   β        = ", β)
    println(io, "--   M_order  = ", M_order)
    println(io, "--   N_grid   = ", N_grid)
    println(io, "--   prec     = ", prec, " bits")
    println(io, "--")
    println(io, "-- This certificate contains only the final scalar outputs of the")
    println(io, "-- Arb-certified interval-arithmetic chain. Intermediate quantities")
    println(io, "-- (sup_bounds vectors, per-grid-point matrices, Gpp/Gppp bounds) are")
    println(io, "-- not shipped — Lean does not redo interval arithmetic; the trust")
    println(io, "-- chain is: Lean-verified universal lemmas (aliasing, Faà di Bruno,")
    println(io, "-- Leibniz, Taylor remainder, Fourier-tail π²/3) + correctness of Arb.")
    println(io)
    println(io, "import WaveCAP.Definitions")
    println(io, "import Mathlib.Data.Rat.Defs")
    println(io, "import Mathlib.LinearAlgebra.Matrix.Notation")
    println(io)
    println(io, "namespace ", module_name)
    println(io)

    # ---- Inputs ----
    println(io, "/-- Number of cosine modes (= length of input `y_coeffs`). -/")
    println(io, "def N_cert : ℕ := ", N_cert)
    println(io)
    _emit_fin_rat_def(io, "a_cert_ℚ", a_cert_rats, N_cert,
        "Cosine coefficients (exact dyadic rationals from Float64 input).")
    println(io)
    println(io, "/-- ℝ-typed view of `a_cert_ℚ`, used by the downstream trust anchor. -/")
    println(io, "noncomputable def a_cert : Fin N_cert → ℝ := fun i => (a_cert_ℚ i : ℝ)")
    println(io)
    println(io, "/-- Wave speed parameter μ (exact dyadic rational). -/")
    println(io, "def μ_cert_ℚ : ℚ := ", _format_rat(μ_rat))
    println(io)
    println(io, "noncomputable def μ_cert : ℝ := (μ_cert_ℚ : ℝ)")
    println(io)
    println(io, "/-- Surface-tension parameter β (exact dyadic rational; 0 disables the capillary path). -/")
    println(io, "def β_cert_ℚ : ℚ := ", _format_rat(β_rat))
    println(io)
    println(io, "noncomputable def β_cert : ℝ := (β_cert_ℚ : ℝ)")
    println(io)

    # ---- Algorithm parameters ----
    println(io, "/-- DFT / Taylor-grid size used by the Arb-arithmetic chain. -/")
    println(io, "def N_fft_cert : ℕ := ", N_grid)
    println(io)
    println(io, "/-- Taylor expansion order used by the Arb-arithmetic chain. -/")
    println(io, "def M_order_cert : ℕ := ", M_order)
    println(io)

    # ---- ε₀_cert ----
    println(io, "/-- Final certified upper bound: ‖R‖_∞ ≤ ε₀_cert. -/")
    println(io, "def ε₀_cert_ℚ : ℚ := ", _format_rat(ε₀_rat))
    println(io)
    println(io, "noncomputable def ε₀_cert : ℝ := (ε₀_cert_ℚ : ℝ)")
    println(io)

    # ---- DLower_cert (β≠0 only) ----
    if has_capillary
        println(io, "/-- Final certified lower bound on D(u) = √(y_u² + (1+H[y_u])²);")
        println(io, "    needed only for β ≠ 0 to certify that 1/D is well-defined. -/")
        println(io, "def DLower_cert_ℚ : ℚ := ", _format_rat(DLower_rat))
        println(io)
        println(io, "noncomputable def DLower_cert : ℝ := (DLower_cert_ℚ : ℝ)")
        println(io)
    end

    # ---- Phase-6 note (NO axioms emitted; see CertificateAxioms.lean) ----
    println(io, "-- **Phase 6:** The trust-anchor axioms `cert_residual_le` and `cert_D_ge`")
    println(io, "-- have been moved to `WaveCAP/CertificateAxioms.lean`, where they are now")
    println(io, "-- derived **theorems** built from a decomposed set of atomic `cert_arb_*`")
    println(io, "-- axioms (one per Julia/Arb primitive). This generated file therefore emits")
    println(io, "-- NO `axiom` declarations — the names `WaveCAP.Certificate.cert_residual_le`")
    println(io, "-- / `cert_D_ge` are re-introduced as theorems there and used by `Main.lean`.")
    println(io, "--")
    println(io, "-- Pre-Phase-6 monolithic axioms (kept here as comments for reference):")
    println(io, "--   axiom cert_residual_le :")
    println(io, "--       ∀ u, |WaveCAP.R a_cert μ_cert β_cert u| ≤ ε₀_cert")
    if has_capillary
        println(io, "--   axiom cert_D_ge :")
        println(io, "--       ∀ u, DLower_cert ≤ WaveCAP.D a_cert u")
    end
    println(io)
    println(io, "end ", module_name)

    return String(take!(io))
end

"""
    emit_certificate(y_coeffs, μ, β; M_order=4, N_grid=256, prec=128,
                     extra_orders=0, module_name="WaveCAP.Certificate") -> String

Run the **monolithic** Babenko residual pipeline at the given inputs and
return the Lean certificate text. Suitable for small `N_grid`; for large
grids that don't fit in RAM use [`emit_certificate_sharded`](@ref), which
produces byte-identical output (modulo the header) via the sharded pipeline.
"""
function emit_certificate(y_coeffs::Vector{<:Real}, μ::Real, β::Real;
                          M_order::Int=4,
                          N_grid::Int=256,
                          prec::Int=128,
                          extra_orders::Int=0,
                          module_name::AbstractString="WaveCAP.Certificate")::String
    full = _residual_bound_full(y_coeffs, μ, β;
                                M_order=M_order, N_grid=N_grid, prec=prec,
                                extra_orders=extra_orders)
    return format_certificate(; y_coeffs=y_coeffs, μ=μ, β=β,
                              N_grid=N_grid, M_order=M_order, prec=prec,
                              bound=full.bound, D_lower=full.D_lower,
                              module_name=module_name)
end

"""
    emit_certificate_sharded(y_coeffs, μ, β; M_order, N_grid, prec=128,
                             extra_orders=0, shard_size=4096,
                             workdir=mktempdir(),
                             module_name="WaveCAP.Certificate") -> String

Like [`emit_certificate`](@ref) but drives the **sharded** pipeline
(`residual_bound_sharded`), so the certificate for a large `N_grid` can be
generated with bounded RAM. The scalar outputs are bit-identical to the
monolithic pipeline, hence so is the certificate text (modulo the header).
"""
function emit_certificate_sharded(y_coeffs::Vector{<:Real}, μ::Real, β::Real;
                                  M_order::Int,
                                  N_grid::Int,
                                  prec::Int=128,
                                  extra_orders::Int=0,
                                  shard_size::Int=4096,
                                  workdir::AbstractString=mktempdir(),
                                  module_name::AbstractString="WaveCAP.Certificate")::String
    sh = residual_bound_sharded(y_coeffs, μ, β;
                                M_order=M_order, N_grid=N_grid, prec=prec,
                                extra_orders=extra_orders,
                                shard_size=shard_size, workdir=workdir)
    return format_certificate(; y_coeffs=y_coeffs, μ=μ, β=β,
                              N_grid=N_grid, M_order=M_order, prec=prec,
                              bound=sh.bound, D_lower=sh.D_lower,
                              module_name=module_name)
end

_atomic_write(path::AbstractString, text::AbstractString) = begin
    dir = dirname(abspath(path))
    isempty(dir) || isdir(dir) || mkpath(dir)
    tmp = path * ".tmp"
    open(tmp, "w") do f
        write(f, text)
    end
    mv(tmp, path; force=true)
    text
end

"""
    write_certificate(path, y_coeffs, μ, β; M_order=4, N_grid=256, prec=128,
                      extra_orders=0, module_name="WaveCAP.Certificate") -> String

Monolithic-pipeline `emit_certificate` written atomically to `path`.
"""
function write_certificate(path::AbstractString,
                           y_coeffs::Vector{<:Real}, μ::Real, β::Real;
                           M_order::Int=4,
                           N_grid::Int=256,
                           prec::Int=128,
                           extra_orders::Int=0,
                           module_name::AbstractString="WaveCAP.Certificate")::String
    text = emit_certificate(y_coeffs, μ, β;
                            M_order=M_order, N_grid=N_grid, prec=prec,
                            extra_orders=extra_orders, module_name=module_name)
    return _atomic_write(path, text)
end

"""
    write_certificate_sharded(path, y_coeffs, μ, β; M_order, N_grid, prec=128,
                              extra_orders=0, shard_size=4096,
                              workdir=mktempdir(),
                              module_name="WaveCAP.Certificate") -> String

Sharded-pipeline `emit_certificate_sharded` written atomically to `path`.
"""
function write_certificate_sharded(path::AbstractString,
                                   y_coeffs::Vector{<:Real}, μ::Real, β::Real;
                                   M_order::Int,
                                   N_grid::Int,
                                   prec::Int=128,
                                   extra_orders::Int=0,
                                   shard_size::Int=4096,
                                   workdir::AbstractString=mktempdir(),
                                   module_name::AbstractString="WaveCAP.Certificate")::String
    text = emit_certificate_sharded(y_coeffs, μ, β;
                                    M_order=M_order, N_grid=N_grid, prec=prec,
                                    extra_orders=extra_orders,
                                    shard_size=shard_size, workdir=workdir,
                                    module_name=module_name)
    return _atomic_write(path, text)
end

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

export emit_certificate, write_certificate
export emit_certificate_sharded, write_certificate_sharded, format_certificate
export arb_upper_rational, arb_lower_rational, arb_midpoint_rational
