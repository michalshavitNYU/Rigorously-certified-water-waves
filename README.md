# Rigorously certified water waves

Publication repository for the Julia interval-arithmetic computations and Lean
proofs supporting certified water-wave residual estimates.

## Certification plan

The paper separates **candidate discovery** from **rigorous certification**.
The neural search and a finite-dimensional Newton polish supply a degree-150
cosine polynomial; the theorem concerns that frozen polynomial and exact
interpretations of its stored coefficients and parameters. The verification
then has two complementary parts:

1. [Julia/Arb interval validation](Julia/README.md) encloses the full
   nonlinear Babenko residual, controls sampled Hilbert transforms by
   analytic aliasing bounds, proves a positive capillary denominator, and
   uses Taylor remainders to extend grid bounds to the whole period.
2. [Lean 4 formalization](Lean/README.md) proves reusable Hilbert,
   derivative, aliasing, denominator and grid-to-continuum mathematics.
   Its publication theorem imports the two final Arb numerical conclusions
   as explicit assumptions, then checks periodic extension, rational
   comparisons and the sign convention of the manuscript residual.

The [Supplementary Methods](docs/supplementary_computer_assisted_accuracy_proof.pdf)
give the detailed argument. The [archived production run](Julia/results/phase4_scalar_n150_20260827_2m_20260829/README.md)
supplies the input, logs and endpoints. The [source provenance](PROVENANCE.md)
maps the original WaveCAP layout to this publication repository. The result
is a certified **uniform residual bound**, not an existence theorem for a
nearby exact water wave; the latter would require a validated inverse bound.

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
lake build WaveCAP.PaperTheorems
lake build WaveCAP.PaperAudit
```

The dedicated publication certificate is
[`PaperTheorems.lean`](Lean/WaveCAP/PaperTheorems.lean).
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

Browse the [Crunchy2 run logs, scalar outputs, input, and original scripts](Julia/results/phase4_scalar_n150_20260827_2m_20260829/README.md).

The archived phase-4 degree-150 candidate uses 151 coefficients, 2,000,000 grid
points, Taylor order 6, 128-bit Arb precision, and 1024-row shards. Its full
residual upper endpoint is approximately `1.7792768524432498e-13`, and its
denominator lower endpoint is approximately `0.34465883518530394`.

**Current Lean status (2026-09-09): sorry-free proofs, two explicit numerical
axioms, and no end-to-end Lean verification of the Crunchy2 computation.**
The publication audit built successfully, and the final residual theorem has
no `sorryAx` dependency. Beyond Lean's standard `propext`, `Classical.choice`,
and `Quot.sound`, it depends on exactly two certificate-specific axioms:

- `cert_arb_dmin_bound`: positivity and the global denominator lower bound.
- `cert_arb_sup_bound_le_eps0`: the residual bound on one period, given the
  denominator bound.

These axioms record final numerical conclusions from Julia/Arb, not proofs of
individual interval operations inside Lean. The general analytic lemmas and
the extension to all real arguments are proved in Lean, but the dedicated
certificate does not instantiate the full analytic pipeline with serialized
Crunchy2 intermediate data. A small residual by itself does not prove
existence of an exact nearby water wave.

See [the Lean audit status](Lean/README.md#verification-status) for the checked
dependency list. The [historical Lean roadmap](docs/source_notes/lean_roadmap_2026-05-23.md)
has a current-status notice; its older milestone claims are not the present
publication verification status.

## Mathematical documentation

- [Computer-assisted accuracy proof (PDF)](docs/supplementary_computer_assisted_accuracy_proof.pdf)
- [LaTeX source for the supplement](docs/supplementary_computer_assisted_accuracy_proof.tex)
- [Documentation index](docs/README.md)
- [Source provenance](PROVENANCE.md)
