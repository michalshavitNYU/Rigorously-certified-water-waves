# Lean proofs and certificates

This directory is a complete Lake project. Its source namespace remains
`WaveCAP`, independently of the publication directory name `Lean`.

See the [Lean proof design](lean_design.md) for the design document and updated overview.

## Build

Install Lean through Elan, then run from this directory:

```sh
lake exe cache get
lake build
lake build WaveCAP.PaperTheorems
lake build WaveCAP.PaperAudit
```

Elan reads the local `lean-toolchain` file; no global toolchain change is needed.
The pinned Lean and Mathlib versions are `v4.29.0-rc6`. The manifests are kept in
version control, while `.lake/` is a local dependency and build cache.

## Theorem chain

The source namespace remains <code>WaveCAP</code>. The table identifies
representative checked results and the separate numerical boundary of the
publication certificate. The generic correctness modules prove contracts
for the verifier's mathematics; the dedicated final theorem does not
instantiate those contracts from serialized intermediate Arb arrays.

| Step | File and representative theorem | Role |
| --- | --- | --- |
| Finite Hilbert algebra | [HilbertOnTrigPoly.lean](WaveCAP/HilbertOnTrigPoly.lean), <code>periodicHilbert_yu_eq_Hyu</code> | Identifies the explicit cosine expression with the Fourier-defined transform of $y_u$. |
| Derivative calculus | [LeibnizBound.lean](WaveCAP/LeibnizBound.lean), [FaaDiBrunoBound.lean](WaveCAP/FaaDiBrunoBound.lean) | Bounds derivatives of products and nonlinear compositions. |
| Sampled transform | [AliasingErrorBound.lean](WaveCAP/AliasingErrorBound.lean), <code>dft_hilbert_aliasing_bound</code> and <code>samplingDFTHilbert_eq_cot_convolution</code> | Controls unresolved modes and identifies the cot-kernel convolution. |
| Grid-to-continuum | [GridToSupNormTaylor.lean](WaveCAP/GridToSupNormTaylor.lean), <code>grid_to_supNorm_taylor</code> | Turns grid derivative data and a remainder into a continuum bound. |
| Generic denominator and residual contracts | [DminBoundCorrectness.lean](WaveCAP/DminBoundCorrectness.lean), <code>dmin_bound_correct</code>; [ResidualAssemblyCorrectness.lean](WaveCAP/ResidualAssemblyCorrectness.lean), <code>cert_residual_le_derived</code> | States correctness conditions for the assembled interval chain. |
| Numerical anchors and periodicity | [PaperCertificateAxioms.lean](WaveCAP/PaperCertificateAxioms.lean), <code>cert_D_ge</code> and <code>cert_residual_le</code> | Imports two final Arb conclusions and extends the one-period residual bound. |
| Paper-facing bounds | [PaperTheorems.lean](WaveCAP/PaperTheorems.lean), <code>residual_bounded_by_phase4_n150_half_dc_decimal</code> and <code>D_bounded_below_by_phase4_n150_half_dc_decimal</code> | Checks exact rational comparisons with the displayed constants. |
| Residual sign bridge | [PaperAudit.lean](WaveCAP/PaperAudit.lean), <code>paperBabenkoGOne_bounded_by_phase4_n150_half_dc_decimal</code> | Transfers the absolute bound to the manuscript's operator. |

The exact [certificate data module](WaveCAP/PaperCertificate.lean)
contains the coefficient vector, parameters and Arb endpoints as rational
constants. [HalfDCConvention.lean](WaveCAP/HalfDCConvention.lean) reconciles
Julia's $a_0/2$ with Lean's direct zeroth coefficient; the audit checks
the certificate-specific coefficient and residual sign bridge.

## Verification status

**Checked 2026-09-09: sorry-free proofs, two explicit numerical axioms, and no
end-to-end Lean verification of the Crunchy2 computation.** The audit target
`WaveCAP.PaperAudit` built successfully in the source checkout,
whose Lean sources and pinned build configuration were verified byte-for-byte
identical to this publication copy. The project proof sources contain no
`sorry` or `admit` placeholders. The audit's `#print axioms` output for
`WaveCAP.residual_bounded_by_phase4_n150_half_dc_certificate` lists exactly:

```text
propext
Classical.choice
Quot.sound
WaveCAP.PaperCertificate.cert_arb_dmin_bound
WaveCAP.PaperCertificate.cert_arb_sup_bound_le_eps0
```

The first three are Lean's standard logical axioms; there is no `sorryAx`
dependency. The remaining two are explicit, unproved-in-Lean numerical
assumptions declared in
[`PaperCertificateAxioms.lean`](WaveCAP/PaperCertificateAxioms.lean):

- `cert_arb_dmin_bound` asserts that `DLower_cert` is positive and is a lower
  bound for `D` at every real argument.
- `cert_arb_sup_bound_le_eps0` assumes that denominator assertion and supplies
  the residual bound on `[0, 2*pi]`.

Lean proves periodic extension and the decimal corollaries from these
assumptions. They record the final Julia/Arb conclusions, not merely the
correctness of primitive interval arithmetic. Lean does not replay the grid,
FFT, interval operations, or final Taylor scan, and this certificate does not
instantiate the full generic analytic pipeline with serialized intermediate
interval data. Removing the two numerical axioms would require additional
verification work; it is not a completed milestone.

Historical Markdown plans may contain illustrative `sorry` stubs and obsolete
axiom counts. Those examples are not compiled proof files. Use this section
and the current audit module for publication claims, rather than the older
milestone descriptions in the [Lean roadmap](../docs/source_notes/lean_roadmap_2026-05-23.md).

## Publication entry points

- `WaveCAP/PaperCertificate.lean` contains the
  exact candidate coefficients, parameters, and numerical certificate endpoints.
- `WaveCAP/PaperCertificateAxioms.lean` declares
  the two numerical assumptions and proves periodic extension.
- `WaveCAP/PaperTheorems.lean` states the global
  residual and denominator results, including the decimal inequalities.
- `WaveCAP/PaperAudit.lean` checks the certificate declarations,
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
