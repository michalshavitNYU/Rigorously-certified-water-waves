"""
ShardStore.jl — exact on-disk storage of Arb column slices for the sharded
residual pipeline (docs/sharding_plan_2026-07-03.md §3).

Serialization uses `Arblib.dump_string` / `Arblib.load_string!`
(wrapping `arb_dump_str` / `arb_load_str`): hex mantissa/exponent plus
radius — the round-trip is **bit-exact**, so storing intermediate balls on
disk does not affect rigor or reproducibility.
"""

struct ShardStore
    dir::String
    prec::Int
    function ShardStore(dir::AbstractString, prec::Int)
        mkpath(dir)
        new(String(dir), prec)
    end
end

_slice_path(store::ShardStore, name::AbstractString, m::Int, s::Int) =
    joinpath(store.dir, string(name, "_m", m, "_s", s, ".arbdump"))

"""
    write_slice(store, name, m, s, vals)

Write the column slice `vals` (level `m`, shard index `s`) to the store,
one exact Arb dump per line.
"""
function write_slice(store::ShardStore, name::AbstractString, m::Int, s::Int,
                     vals::AbstractVector{Arb})
    open(_slice_path(store, name, m, s), "w") do io
        for v in vals
            println(io, Arblib.dump_string(v))
        end
    end
    return nothing
end

"""
    read_slice(store, name, m, s, n) -> Vector{Arb}

Read back a column slice of length `n`. Values are bit-identical to what
was written.
"""
function read_slice(store::ShardStore, name::AbstractString, m::Int, s::Int,
                    n::Int)::Vector{Arb}
    vals = Vector{Arb}(undef, n)
    open(_slice_path(store, name, m, s), "r") do io
        for i in 1:n
            line = readline(io)
            vals[i] = Arblib.load_string!(Arb(; prec=store.prec), line)
        end
    end
    return vals
end

"""
    write_matrix_slices(store, name, s, derivs)

Write every column of the shard-local derivative matrix `derivs`
(`P × (M+1)`) as slices `name_m0..name_mM` for shard `s`.
"""
function write_matrix_slices(store::ShardStore, name::AbstractString, s::Int,
                             derivs::AbstractMatrix{Arb})
    for m in 0:(size(derivs, 2) - 1)
        write_slice(store, name, m, s, view(derivs, :, m + 1))
    end
    return nothing
end

"""
    read_matrix_slices(store, name, s, n, ncols) -> Matrix{Arb}

Read back a shard-local derivative matrix written by `write_matrix_slices`.
"""
function read_matrix_slices(store::ShardStore, name::AbstractString, s::Int,
                            n::Int, ncols::Int)::Matrix{Arb}
    out = Matrix{Arb}(undef, n, ncols)
    for m in 0:(ncols - 1)
        col = read_slice(store, name, m, s, n)
        for i in 1:n
            out[i, m + 1] = col[i]
        end
    end
    return out
end
