# Lean certificate audit: phase-4 N=150 half-DC result

## Executive conclusion

The installed Lean modules compile, the exact generated certificate is
byte-for-byte identical to the downloaded Crunchy artifact, and the stated
exact and decimal inequalities follow correctly from the assumptions present
in Lean.

The central qualification is that the current headline theorem is a formal
derivation from two certificate-specific numerical axioms. Lean does not
replay the two-million-point Arb computation and does not derive its final
Taylor-grid residual bound from serialized intermediate interval data. The
paper should call this a machine-checked theorem with an explicit two-anchor
numerical trust boundary, not an independent Lean verification of every
interval operation.

## Findings

### High: the final numerical inequalities are axioms

`CertificatePhase4ScalarN150Phase420260827HalfDCAxioms.lean` declares:

1. `cert_arb_dmin_bound`, which already asserts positivity and the global
   lower bound for `D`.
2. `cert_arb_sup_bound_le_eps0`, which already asserts the residual bound on
   `[0, 2*pi]` after receiving the denominator anchor.

The exact headline theorem applies the second anchor and then uses proved
periodicity to extend the result to all real arguments. Therefore, Lean
checks the post-processing of the numerical conclusions, but the correctness
of those conclusions remains in the external Arb trust boundary.

**Required manuscript wording:** state both anchors explicitly and say that
they are axioms recording the final Arb conclusions. Do not say that Lean
reruns the grid, FFT, interval arithmetic, or Taylor scan.

### High: the generic analytic library is not instantiated by this certificate

The project contains proved generic results for Fourier tails, aliasing,
Leibniz and Faa di Bruno bounds, Taylor remainders, denominator bounds, and
residual assembly. These modules build successfully and are mathematically
valuable. However, `#print axioms` for the certificate-facing theorem shows
only the two final numerical anchors, not a chain of serialized
certificate-specific intermediate data through those generic results.

The generated certificate contains coefficients and final endpoints, but not
the per-grid derivative matrices, sup-bound vectors, FFT enclosures, or
Taylor-series records required to instantiate the full structural pipeline in
Lean.

**Required manuscript wording:** distinguish "the analytic ingredients are
formalized generically" from "the present numerical instance is imported
through two final trust anchors."

### Medium: Lean's `R` has the opposite sign from the paper's `B`

For `g = 1`, `mu = c^2`, and `beta = kappa`, inspection of
`Definitions.lean` shows

```text
R(a, mu, beta; u) = -B(y(a))(u).
```

This does not affect an absolute residual bound. The new additive audit module
proves `R_eq_neg_paperBabenkoGOne`,
`abs_R_eq_abs_paperBabenkoGOne`, and exact/decimal certificate corollaries
stated directly for the manuscript operator.

### Medium: the half-DC relation is only partly certificate-specific in Lean

`HalfDCConvention.lean` proves the generic coefficient adapter and invariance
of `y_u`, `H[y_u]`, `D`, and `G`. The certificate-specific theorem proves the
zeroth coefficient is exactly one half of the source zeroth coefficient.

The source vector is not encoded as a second exact Lean vector, so Lean does
not currently prove a 151-component equality between named source and
certificate vectors. The generated certificate already contains the converted
vector. Externally, the installed file and downloaded generated file are
byte-identical, with SHA-256:

```text
2190a52f3520ab0b15683eab592ee1105797d737a22f2e24a869130f5a431ed2
```

**Required manuscript wording:** say that Lean proves the generic adapter and
the exact zeroth-entry identity. Do not claim that a separate full source
vector equality is currently a certificate theorem.

### Medium: the dedicated certificate is not exercised by default `lake build`

The default project build succeeds, but the new half-DC headline module is not
imported by the default library root. It was therefore checked directly:

```text
lake env lean WaveCAP/MainPhase4ScalarN150Phase420260827HalfDC.lean
lake env lean WaveCAP/AuditPhase4ScalarN150HalfDC.lean
```

Both commands succeed. Publication or CI scripts should include these direct
targets or import them from a dedicated verification root.

### Scope: this is not a nearby-solution existence theorem

The theorem proves a uniform residual bound for one explicit finite cosine
series. It does not provide an inverse bound for the linearized Babenko
operator, a contraction estimate, or a Newton-Kantorovich radius. The paper
must not infer existence or uniqueness of an exact nearby water wave from the
residual certificate alone.

## Verified theorem statements

The direct audit reports:

```text
forall u, |R a_cert mu_cert beta_cert u| <= epsilon0_cert
forall u, |R a_cert mu_cert beta_cert u| <= 178 / 10^15
forall u, 3446588 / 10^7 <= D a_cert u
```

The paper-notation bridge additionally proves:

```text
forall u, |paperBabenkoGOne a_cert mu_cert beta_cert u|
  <= epsilon0_cert
forall u, |paperBabenkoGOne a_cert mu_cert beta_cert u|
  <= 178 / 10^15
```

The exact constants are:

```text
epsilon0_cert =
  66570650447636094343693612891855023841 /
  374144419156711147060143317175368453031918731001856

DLower_cert =
  30024018999569627600479976468787845960585 /
  87112285931760246646623899502532662132736
```

## Axiom dependency audit

The exact and decimal residual theorems depend on:

```text
propext
Classical.choice
Quot.sound
cert_arb_dmin_bound
cert_arb_sup_bound_le_eps0
```

The denominator theorem depends on the first four entries except the residual
anchor. The half-DC and paper-sign alignment theorems depend only on the three
standard Mathlib principles.

## Reproducibility checks performed

- Lean toolchain: `leanprover/lean4:v4.29.0-rc6`.
- Mathlib revision: `v4.29.0-rc6`.
- Full default `lake build`: successful.
- Dedicated half-DC headline module: successful.
- Dedicated theorem/axiom audit module: successful.
- Installed/downloaded generated certificate comparison: byte-identical.
- Exact rational comparisons to `1.78e-13` and `0.3446588`: checked by Lean's
  `norm_num` in the theorem module.
