# Rigorously certified water waves

Publication repository for the Julia interval-arithmetic computations and Lean
proofs supporting certified water-wave residual estimates.

## Repository layout

```text
Lean/                     Lean 4 project, analytic proofs, and certificates
  WaveCAP/                Lean modules
  WaveCAP.lean            Library entry point
  lean-toolchain          Pinned Lean version
  lakefile.toml            Lake project configuration
  lake-manifest.json       Pinned Lean dependencies
Julia/                    BabenkoRigorous Julia package
  src/                    Interval arithmetic and residual algorithms
  test/                   Tests, including sharding and half-DC conversion
  scripts/                Verification and certificate-generation scripts
  results/                Archived Crunchy2 run evidence
  Project.toml            Julia package configuration
  Manifest.toml           Pinned Julia environment
docs/                     Mathematical notes and implementation documentation
PROVENANCE.md             Source snapshot and directory mapping
```

## Lean

Use the toolchain specified in `Lean/lean-toolchain`:
`leanprover/lean4:v4.29.0-rc6`, with the matching Mathlib revision.

```sh
cd Lean
lake exe cache get
lake build
lake build WaveCAP.MainPhase4ScalarN150Phase420260827HalfDC
lake build WaveCAP.AuditPhase4ScalarN150HalfDC
```

The dedicated publication certificate is
[`MainPhase4ScalarN150Phase420260827HalfDC.lean`](Lean/WaveCAP/MainPhase4ScalarN150Phase420260827HalfDC.lean).
The older `Main.lean` remains as part of the source snapshot. See
[the Lean guide](Lean/README.md) for the distinction and the numerical assumptions.

## Julia

The supplied manifest records Julia **1.12.5**. Use that version to reproduce the
locked environment. From the repository root:

```sh
julia --project=Julia -e 'using Pkg; Pkg.instantiate()'
julia --project=Julia Julia/test/runtests.jl
julia --project=Julia Julia/test/test_half_dc_certificate_additive.jl
julia --project=Julia Julia/test/test_sharded.jl
```

See [the Julia guide](Julia/README.md) for the production candidate, archived
results, and commands for rerunning the staged verifier. The two-million-point
computation is a separate, substantial run; it is not performed by repository
initialization.

## Certified result and scope

The archived phase-4 degree-150 candidate uses 151 coefficients, 2,000,000 grid
points, Taylor order 6, 128-bit Arb precision, and 1024-row shards. Its full
residual upper endpoint is approximately `1.7792768524432498e-13`, and its
denominator lower endpoint is approximately `0.34465883518530394`.

The dedicated Lean theorem proves the resulting global residual statement
conditional on two certificate-specific numerical assumptions: the denominator
lower bound and the residual bound on one period. The general analytic lemmas
are proved in Lean. The repository does not claim that Lean re-executes the Arb
computation, or that a small residual by itself proves existence of an exact
nearby water wave.

## Mathematical documentation

- [Detailed estimate of `y H[y_u]`, with ordinary theorems and proofs](docs/y_hilbert_yu_interval_estimate.pdf)
- [Standalone LaTeX source](docs/y_hilbert_yu_interval_estimate.tex)
- [Documentation index](docs/README.md)
- [Source provenance](PROVENANCE.md)
