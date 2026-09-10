module HalfDCCertificateTools

using Arblib
using BabenkoRigorous

"""
    lean_coefficients_half_dc(y_coeffs)

Return a new coefficient vector for Lean's full-DC convention. The input is
not mutated: only the first output coefficient is divided by two.
"""
function lean_coefficients_half_dc(y_coeffs::Vector{<:Real})
    isempty(y_coeffs) && throw(ArgumentError("the coefficient vector is empty"))
    lean_coeffs = collect(y_coeffs)
    lean_coeffs[1] = y_coeffs[1] / 2
    return lean_coeffs
end

"""Load an exact Arb value previously written with `Arblib.dump_string`."""
function load_arb_dump(path::AbstractString)::Arb
    isfile(path) || throw(ArgumentError("Arb dump does not exist: $path"))
    return Arblib.load_string(Arb, strip(read(path, String)))
end

"""
    format_half_dc_certificate(; y_coeffs, mu, beta, ...)

Format a new Lean certificate for an interval result computed with Julia's
half-DC convention. The Arb bound and D lower bound are reused exactly; only
the coefficient representation passed to Lean is converted.
"""
function format_half_dc_certificate(;
        y_coeffs::Vector{<:Real},
        mu::Real,
        beta::Real,
        N_grid::Int,
        M_order::Int,
        prec::Int,
        bound::Arb,
        D_lower::Union{Arb,Nothing},
        module_name::AbstractString)
    lean_coeffs = lean_coefficients_half_dc(y_coeffs)
    metadata = """-- DC-convention adapter certificate.
-- The interval run used y(u) = source_y_coeffs[1] / 2 + sum_{k>=1} a[k] cos(k*u).
-- This Lean module stores source_y_coeffs[1] / 2 as a_cert[0], because
-- WaveCAP.y evaluates its zeroth coefficient without an additional factor 1/2.
-- The original coefficient file and the Arb endpoint dumps are unchanged.
-- Source DC coefficient: $(repr(y_coeffs[1]))
-- Lean DC coefficient:   $(repr(lean_coeffs[1]))
--
"""
    certificate = BabenkoRigorous.format_certificate(;
        y_coeffs=lean_coeffs,
        μ=mu,
        β=beta,
        N_grid=N_grid,
        M_order=M_order,
        prec=prec,
        bound=bound,
        D_lower=D_lower,
        module_name=module_name,
    )
    return metadata * certificate
end

"""Write a new file and fail rather than replacing an existing path."""
function write_new_file(path::AbstractString, text::AbstractString)
    ispath(path) && throw(ArgumentError("refusing to overwrite existing path: $path"))
    parent = dirname(abspath(path))
    isdir(parent) || mkpath(parent)
    open(path, "w") do io
        write(io, text)
    end
    return path
end

export format_half_dc_certificate, lean_coefficients_half_dc
export load_arb_dump, write_new_file

end
