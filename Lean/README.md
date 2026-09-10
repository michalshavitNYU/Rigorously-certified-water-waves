# Lean proofs and certificates

This directory is a complete Lake project. Its source namespace remains
`WaveCAP`, independently of the publication directory name `Lean`.

## Build

Install Lean through Elan, then run from this directory:

```sh
lake exe cache get
lake build
lake build WaveCAP.MainPhase4ScalarN150Phase420260827HalfDC
lake build WaveCAP.AuditPhase4ScalarN150HalfDC
```

Elan reads the local `lean-toolchain` file; no global toolchain change is needed.
The pinned Lean and Mathlib versions are `v4.29.0-rc6`. The manifests are kept in
version control, while `.lake/` is a local dependency and build cache.

## Publication entry points

- `WaveCAP/CertificatePhase4ScalarN150Phase420260827HalfDC.lean` contains the
  exact candidate coefficients, parameters, and numerical certificate endpoints.
- `WaveCAP/CertificatePhase4ScalarN150Phase420260827HalfDCAxioms.lean` declares
  the two numerical assumptions and proves periodic extension.
- `WaveCAP/MainPhase4ScalarN150Phase420260827HalfDC.lean` states the global
  residual and denominator results, including the decimal inequalities.
- `WaveCAP/AuditPhase4ScalarN150HalfDC.lean` checks the certificate declarations,
  the manuscript residual sign convention, and their axiom dependencies.
- `WaveCAP/HalfDCConvention.lean` reconciles Julia's `a0/2` convention with
  Lean's full constant coefficient.

`WaveCAP/Main.lean` and `WaveCAP/CertificateAxioms.lean` are the older generic
certificate route. They are preserved for their proofs and context; the
dedicated half-DC modules above identify the two-million-point result.

## Analytic modules

The source includes Fourier and Hilbert identities, Leibniz and Faa di Bruno
bounds, Taylor remainders, grid-to-continuum estimates, denominator estimates,
and residual assembly. Start with `WaveCAP/Definitions.lean`,
`WaveCAP/TaylorSeriesCorrectness.lean`, and
`WaveCAP/ResidualAssemblyCorrectness.lean`.

The dedicated final theorem relies on `cert_arb_dmin_bound` and
`cert_arb_sup_bound_le_eps0`. They encode numerical claims verified by the
external interval-arithmetic calculation. Existence theorems for abstract
Taylor records do not identify those records with serialized Arb arrays.

The [standalone mathematical note](../docs/y_hilbert_yu_interval_estimate.pdf)
gives ordinary statements and proofs, along with exact source correspondences.
