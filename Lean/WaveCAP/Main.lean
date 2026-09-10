import WaveCAP.HilbertOnTrigPoly
import WaveCAP.HilbertCommutesDeriv
import WaveCAP.AliasingErrorBound
import WaveCAP.FourierTailHilbert
import WaveCAP.FaaDiBrunoBound
import WaveCAP.LeibnizBound
import WaveCAP.TaylorRemainder
import WaveCAP.GridToSupNormTaylor
import WaveCAP.DLowerBound
import WaveCAP.Certificate
import WaveCAP.CertificateAxioms  -- Phase 6: cert_residual_le is now a theorem

/-!
# Main: certified sup-norm bound on the Babenko residual

This is the **headline statement** of the WaveCAP Lean side:

```
For the certificate-supplied (a_cert, μ_cert, β_cert):
  ∀ u, |R a_cert μ_cert β_cert u| ≤ ε₀_cert.
```

It is derived from the Phase-6 decomposed trust-anchor axioms
(`cert_arb_*`, see `WaveCAP/CertificateAxioms.lean`), composed with the
Lean-verified residual chain in Area G (`cert_residual_le_derived`).

## Scope

The original scope of WaveCAP is to deliver *rigorous bounds on the
residual*, not an existence theorem.  Upgrading the residual bound to
existence of a true Babenko solution requires inverting the linearisation
of the Babenko operator — an unbounded infinite-dimensional operator. That
is a substantial separate problem (paper-level analysis of the operator's
spectral properties, additional Arb-computed certificates for the
inverse-operator norm, and a Newton–Kantorovich fixed-point argument in
Lean).  We do *not* attempt it here.  See "Out of scope" below for what a
follow-up project would entail.

## Trust footprint (Phase 6, corrected 2026-07-03)

`#print axioms residual_bounded_by_certificate` shows the chain depends on:
  - `propext, Classical.choice, Quot.sound` (standard Mathlib), and
  - the **two** remaining `cert_arb_*` axioms of
    `WaveCAP/CertificateAxioms.lean`: `cert_arb_dmin_bound` and
    `cert_arb_sup_bound_le_eps0` — the two scalars Julia/Arb shipped.
    All five per-primitive statements (`trigpoly_to_taylor`, `ts_mul`,
    `ts_deriv`, `inv_series`, `fft_hilbert_series`) and the composite
    `cert_arb_residual_TS` are derived theorems (de-axiomatization
    targets P2–P6 of `docs/audit_2026-07-03.md`, completed 2026-07-03).

This is the floor for the certificate-axiom design: the entire
quantitative Arb trust is the two shipped scalars, and everything
structural is machine-checked (see "What the trust base actually is"
in `WaveCAP/CertificateAxioms.lean`).

Definitional fidelity: the hand-expanded cosine sum `Hyu` used inside
`R`, `D`, and `G` is certified to equal the true periodic Hilbert
transform of `y_u` by `periodicHilbert_yu_eq_Hyu`
(`WaveCAP/HilbertOnTrigPoly.lean`, proved on core axioms only), so the
formalized `R` is the Babenko residual with `periodicHilbert`
everywhere.

This Phase-6 decomposition replaces the old monolithic axiom
`WaveCAP.Certificate.cert_residual_le`, which has been retired and is
now a *derived theorem* with the same name and statement (see
`WaveCAP/CertificateAxioms.lean`). Trust-wise the two numeric axioms are
equivalent to the retired monolithic pair.

Everything between (N1–N5, the Hilbert-transform infrastructure, the
TaylorSeries grid-max wrapper) is proved against Mathlib with the same
three standard axioms.

## Out of scope (deliberately)

A true existence theorem of the form
```
∃ y_star, (∀ u, R y_star μ β u = 0) ∧ ‖y_star - a_cert‖ ≤ explicit_radius
```
would require:
  1. A Lean formalisation of the Fréchet derivative `R'` of the Babenko
     operator (built on `periodicHilbert` + multiplication operators).
  2. An Arb-computed bound `K_cert ≥ ‖R'(a_cert)⁻¹‖_op` shipped as an
     additional certificate input.
  3. Bounds on `‖R''‖` on a ball, drawing on N1–N5.
  4. The Banach fixed-point argument (Mathlib has `ContractingWith.fixedPoint`)
     applied to the Newton iterate `T(y) := y - R'(a_cert)⁻¹ R(y)`.

Step 2 is where the real work sits: proving an operator-norm bound on the
inverse of the linearisation of a nonlinear pseudodifferential operator is
not routine.  Such a result exists in the analytic CAP literature for the
Babenko equation, but formalising it is research-grade work.
-/

namespace WaveCAP

open WaveCAP.Certificate

/-- **Headline**: the Babenko residual is bounded by `ε₀_cert` at every point.

This is the rigorous-bounds statement the WaveCAP project delivers: the
sup-norm of the residual `R` evaluated at the certificate inputs is bounded
by the rational scalar `ε₀_cert` that the Julia/Arb pipeline computes.

The Lean trust chain reaching this statement consists of:
  - The universal analytic lemmas N1–N5 (proved against Mathlib with only
    the standard axioms `propext, Classical.choice, Quot.sound`).
  - The two numeric axioms `cert_arb_dmin_bound` and
    `cert_arb_sup_bound_le_eps0`, which carry the entire quantitative
    Arb trust — see `WaveCAP/CertificateAxioms.lean`. -/
theorem residual_bounded_by_certificate :
    ∀ u, |R a_cert μ_cert β_cert u| ≤ ε₀_cert :=
  cert_residual_le

end WaveCAP

-- Phase 6 trust footprint: the headline now depends on the atomic
-- `cert_arb_*` axioms, not the old monolithic `cert_residual_le` axiom.
#print axioms WaveCAP.residual_bounded_by_certificate
