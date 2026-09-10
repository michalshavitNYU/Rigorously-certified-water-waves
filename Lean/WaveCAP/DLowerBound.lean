import WaveCAP.Definitions
import WaveCAP.FaaDiBrunoBound

/-!
# `D` positivity and bounds on `G = (1 + H[y_u]) / D`

`D(u) = √(y_u(u)² + (1 + H[y_u](u))²)` must be bounded below by a strictly
positive constant for `G = (1+H[y_u])/D` to be well-defined and smooth.

In the certificate-driven workflow:

  - `BabenkoRigorous/src/TaylorSeries.jl` `dmin_bound(ts_D)` computes a
    rigorous Arb lower bound on `inf_𝕋 D(u)`.
  - The Lean side consumes that as a rational constant `DLower_cert : ℚ`
    in `Certificate.lean` (with axiom `cert_D_ge` stating
    `(DLower_cert : ℝ) ≤ D(u)` pointwise).
  - This file connects `cert_D_ge` to `G` being well-defined, with
    derivative bounds propagated by Faà di Bruno (applied to `φ = 1/x`)
    composed with the trig-poly `D`.

Cross-reference: `BabenkoRigorous/src/TaylorSeries.jl` `dmin_bound`,
`inv_series`; `BabenkoRigorous/src/GDerivatives.jl` `compute_G_taylor`.
-/

namespace WaveCAP

open Real

/-- `D(u) > 0` follows from `DLower_cert > 0` and the certificate-supplied
pointwise lower bound. -/
theorem D_pos_of_DLower_pos
    {N : ℕ} {a : Fin N → ℝ}
    {DLower : ℝ} (hDLower_pos : 0 < DLower)
    (hcert : ∀ u, DLower ≤ D a u)
    (u : ℝ) : 0 < D a u :=
  lt_of_lt_of_le hDLower_pos (hcert u)

/-- `G` unfolds to `(1 + H[y_u]) / D` by definition; the positivity of `D` is
needed when consumers want to reason about division being well-defined, not
for the unfolding itself. -/
theorem G_eq_div
    {N : ℕ} {a : Fin N → ℝ} (u : ℝ) :
    G a u = (1 + Hyu a u) / D a u := rfl

end WaveCAP
