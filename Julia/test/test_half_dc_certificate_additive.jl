using Test
using Arblib
using BabenkoRigorous

include(joinpath(@__DIR__, "..", "scripts", "half_dc_certificate_tools.jl"))
using .HalfDCCertificateTools

@testset "half-DC certificate adapter" begin
    source = [-0.2, 0.3, -0.1]
    source_before = copy(source)
    converted = lean_coefficients_half_dc(source)

    @test source == source_before
    @test converted == [-0.1, 0.3, -0.1]

    bound = Arb("1.0e-12 +/- 1.0e-20"; prec=128)
    D_lower = Arb("0.34 +/- 1.0e-20"; prec=128)
    text = format_half_dc_certificate(;
        y_coeffs=source,
        mu=1.25,
        beta=0.25,
        N_grid=2_000_000,
        M_order=6,
        prec=128,
        bound=bound,
        D_lower=D_lower,
        module_name="WaveCAP.TestHalfDC",
    )

    @test occursin("namespace WaveCAP.TestHalfDC", text)
    @test occursin("Source DC coefficient: -0.2", text)
    @test occursin("Lean DC coefficient:   -0.1", text)
    @test occursin("def N_fft_cert : ℕ := 2000000", text)

    mktempdir() do dir
        dump_path = joinpath(dir, "bound.arbdump")
        open(dump_path, "w") do io
            println(io, Arblib.dump_string(bound))
        end
        loaded = load_arb_dump(dump_path)
        @test Arblib.dump_string(loaded) == Arblib.dump_string(bound)

        output = joinpath(dir, "HalfDC.lean")
        @test write_new_file(output, text) == output
        @test read(output, String) == text
        @test_throws ArgumentError write_new_file(output, text)
    end
end
