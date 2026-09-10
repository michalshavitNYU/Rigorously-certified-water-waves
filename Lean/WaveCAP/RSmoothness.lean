import WaveCAP.PipelineSmoothness
import WaveCAP.HilbertSmoothness

/-!
# Smoothness of the Babenko residual `R`

This file proves `R_contDiff`: the residual

```
R(u) = μ · H[y_u](u) − y(u) − y(u) · H[y_u](u) − H[y · y_u](u)
       + β · ∂_u(y_u/D)(u) − β · H[∂_u G](u)
```

is `C^k` for every natural `k`, given a strictly positive uniform lower bound
on `D` (which makes the quotients `y_u/D` and `G = (1+H[y_u])/D` smooth).

The proof is a mechanical assembly of:

* `y_contDiff_all`, `yu_contDiff_all`, `Hyu_contDiff_all`, `D_contDiff_all`
  (from `PipelineSmoothness.lean`).
* `hilbert_contDiff` (from `HilbertSmoothness.lean`), which costs one
  derivative: `ContDiff (k+1) (periodic) ⇒ ContDiff k (H ∘ ·)`.
* `ContDiff.div` (Mathlib), with `D ≠ 0` discharged from the certificate
  positivity hypothesis.
* `ContDiff.deriv'` (Mathlib), which costs one derivative: `ContDiff (k+1)
  ⇒ ContDiff k (deriv ·)`.
* `Function.Periodic.mul` (Mathlib) for periodicity of `y · y_u`.

Smoothness budget. To get `ContDiff k R`, the most demanding terms are
the two `H[deriv (...)]` style terms (paths B), which need `ContDiff (k+2)`
on the inner non-derivative function — `+1` for the inner `deriv`, then
`+1` for the outer `H`. Since `y, y_u, H[y_u], D` are all `C^∞`, every
building block can be upcast as needed via `ContDiff.of_le`.

Cross-reference: `WaveCAP/Definitions.lean` for `R`; `PipelineSmoothness.lean`
and `HilbertSmoothness.lean` for the underlying ingredient lemmas.
-/

namespace WaveCAP

open Real

variable {N : ℕ} (a : Fin N → ℝ) (μ β : ℝ)

/-- Helper: `y · y_u` is `C^k` for every natural `k`. -/
theorem y_mul_yu_contDiff_all :
    ∀ k : ℕ, ContDiff ℝ k (fun u => y a u * yu a u) :=
  fun k => (y_contDiff_all a k).mul (yu_contDiff_all a k)

/-- Helper: `y · y_u` is `2π`-periodic. -/
theorem y_mul_yu_periodic :
    Function.Periodic (fun u => y a u * yu a u) (2 * Real.pi) :=
  (y_periodic a).mul (yu_periodic_re a)

/-- `H[y · y_u]` is `C^k` for every natural `k`. -/
theorem hilbert_y_mul_yu_contDiff_all :
    ∀ k : ℕ, ContDiff ℝ k (periodicHilbert (fun u => y a u * yu a u)) := by
  intro k
  exact hilbert_contDiff (k := k) (y_mul_yu_contDiff_all a (k + 1))
    (y_mul_yu_periodic a)

/-- `D ≠ 0` pointwise, from a strictly positive lower bound. -/
private lemma D_ne_zero
    {DLower : ℝ} (hDLower_pos : 0 < DLower)
    (hcert : ∀ u, DLower ≤ D a u) :
    ∀ u, D a u ≠ 0 :=
  fun u => ne_of_gt (lt_of_lt_of_le hDLower_pos (hcert u))

/-- `y_u / D` is `C^k` for every natural `k`. -/
theorem yu_div_D_contDiff_all
    {DLower : ℝ} (hDLower_pos : 0 < DLower)
    (hcert : ∀ u, DLower ≤ D a u) :
    ∀ k : ℕ, ContDiff ℝ k (fun u => yu a u / D a u) := fun k =>
  ContDiff.div (yu_contDiff_all a k) (D_contDiff_all a hDLower_pos hcert k)
    (D_ne_zero a hDLower_pos hcert)

/-- `G = (1 + H[y_u]) / D` is `C^k` for every natural `k`. -/
theorem G_contDiff_all
    {DLower : ℝ} (hDLower_pos : 0 < DLower)
    (hcert : ∀ u, DLower ≤ D a u) :
    ∀ k : ℕ, ContDiff ℝ k (G a) := by
  intro k
  have h_def : (G a) = fun u => (1 + Hyu a u) / D a u := rfl
  rw [h_def]
  exact ContDiff.div (contDiff_const.add (Hyu_contDiff_all a k))
    (D_contDiff_all a hDLower_pos hcert k)
    (D_ne_zero a hDLower_pos hcert)

/-- `D` is `2π`-periodic — re-exported here for convenience. -/
theorem D_periodic_aux : Function.Periodic (D a) (2 * Real.pi) :=
  D_periodic_re a

/-- `y_u / D` is `2π`-periodic. -/
theorem yu_div_D_periodic :
    Function.Periodic (fun u => yu a u / D a u) (2 * Real.pi) :=
  (yu_periodic_re a).div (D_periodic_re a)

/-- `G` is `2π`-periodic. -/
theorem G_periodic :
    Function.Periodic (G a) (2 * Real.pi) := by
  intro x
  change (1 + Hyu a (x + 2 * Real.pi)) / D a (x + 2 * Real.pi)
      = (1 + Hyu a x) / D a x
  rw [Hyu_periodic_re a x, D_periodic_re a x]

/-- `deriv (G a)` is `2π`-periodic (consequence of `G` periodic). -/
theorem G_deriv_periodic :
    Function.Periodic (deriv (G a)) (2 * Real.pi) :=
  WaveCAP.Periodic.deriv_periodic (G_periodic a)

/-- `deriv (fun u => y_u / D)` is `2π`-periodic. -/
theorem yu_div_D_deriv_periodic :
    Function.Periodic (deriv (fun u => yu a u / D a u)) (2 * Real.pi) :=
  WaveCAP.Periodic.deriv_periodic (yu_div_D_periodic a)

/-- `deriv G` is `C^k` for every natural `k`. -/
theorem G_deriv_contDiff_all
    {DLower : ℝ} (hDLower_pos : 0 < DLower)
    (hcert : ∀ u, DLower ≤ D a u) :
    ∀ k : ℕ, ContDiff ℝ k (deriv (G a)) := fun k =>
  ContDiff.deriv' (n := (k : WithTop ℕ∞)) (G_contDiff_all a hDLower_pos hcert (k + 1))

/-- `deriv (y_u / D)` is `C^k` for every natural `k`. -/
theorem yu_div_D_deriv_contDiff_all
    {DLower : ℝ} (hDLower_pos : 0 < DLower)
    (hcert : ∀ u, DLower ≤ D a u) :
    ∀ k : ℕ, ContDiff ℝ k (deriv (fun u => yu a u / D a u)) := fun k =>
  ContDiff.deriv' (n := (k : WithTop ℕ∞))
    (yu_div_D_contDiff_all a hDLower_pos hcert (k + 1))

/-- `H[deriv G]` is `C^k` for every natural `k`. -/
theorem hilbert_G_deriv_contDiff_all
    {DLower : ℝ} (hDLower_pos : 0 < DLower)
    (hcert : ∀ u, DLower ≤ D a u) :
    ∀ k : ℕ, ContDiff ℝ k (periodicHilbert (deriv (G a))) := fun k =>
  hilbert_contDiff (k := k) (G_deriv_contDiff_all a hDLower_pos hcert (k + 1))
    (G_deriv_periodic a)

/-- **Main theorem: the Babenko residual `R` is `C^k` for every `k`.**

Assumes a strictly positive uniform lower bound `DLower ≤ D` (which makes
the quotients `y_u/D` and `G = (1+H[y_u])/D` well-defined and smooth).

The decomposition mirrors the residual definition:
```
R = μ · Hyu - y - y · Hyu - H[y · yu] + β · deriv(yu/D) - β · H[deriv G]
```

* `μ · Hyu`             — `C^∞` (Hyu is a trig polynomial).
* `y`, `y · Hyu`        — `C^∞`.
* `H[y · yu]`           — `C^k` for any k via `hilbert_contDiff` (one
                          derivative cost).
* `β · deriv(yu/D)`     — `C^k` for any k via `ContDiff.deriv'` (one
                          derivative cost) on the smooth quotient `yu/D`.
* `β · H[deriv G]`      — `C^k` via `hilbert_contDiff ∘ deriv'` (two
                          derivative costs) on the smooth `G`.

All built-in ingredients are `C^∞`, so `.of_le` is never tight. -/
theorem R_contDiff
    {DLower : ℝ} (hDLower_pos : 0 < DLower)
    (hcert : ∀ u, DLower ≤ D a u)
    (k : ℕ) :
    ContDiff ℝ k (R a μ β) := by
  unfold R
  -- Each summand assembled as ContDiff ℝ k:
  have h_muHyu : ContDiff ℝ k (fun u => μ * Hyu a u) :=
    contDiff_const.mul (Hyu_contDiff_all a k)
  have h_y : ContDiff ℝ k (y a) := y_contDiff_all a k
  have h_yHyu : ContDiff ℝ k (fun u => y a u * Hyu a u) :=
    (y_contDiff_all a k).mul (Hyu_contDiff_all a k)
  have h_HyYu : ContDiff ℝ k (periodicHilbert (fun v => y a v * yu a v)) :=
    hilbert_y_mul_yu_contDiff_all a k
  have h_βderivYuD : ContDiff ℝ k (fun u => β * deriv (fun v => yu a v / D a v) u) :=
    contDiff_const.mul (yu_div_D_deriv_contDiff_all a hDLower_pos hcert k)
  have h_βHdG : ContDiff ℝ k (fun u => β * periodicHilbert (deriv (G a)) u) :=
    contDiff_const.mul (hilbert_G_deriv_contDiff_all a hDLower_pos hcert k)
  -- Now assemble: R = (((μ Hyu - y) - y · Hyu) - H[y · yu]) + β · deriv(yu/D)
  --                  - β · H[deriv G].
  exact (((((h_muHyu.sub h_y).sub h_yHyu).sub h_HyYu).add h_βderivYuD).sub h_βHdG)

end WaveCAP

-- ===========================================================================
-- Sanity checks
-- ===========================================================================

#check @WaveCAP.R_contDiff
#print axioms WaveCAP.R_contDiff
