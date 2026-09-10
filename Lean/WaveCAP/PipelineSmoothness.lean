import WaveCAP.Common
import WaveCAP.Definitions
import WaveCAP.TightDGSupBounds

/-!
# Pipeline smoothness helpers — closing Cat-2 hoists

This file proves a handful of small mathematical-gap lemmas that previously
needed to be hoisted as hypotheses to callers in `DminBoundCorrectness.lean`,
`FFTHilbertCorrectness.lean`, and `ResidualAssemblyCorrectness.lean`. The
lemmas live here so the pipeline-level theorems can call them directly:

  * `D_deriv_eq` — chain rule for `D = √(y_u² + (1 + H[y_u])²)`.
  * `yu_deriv_eq`, `Hyu_deriv_eq` — closed forms for `deriv (yu a)` and
    `deriv (Hyu a)` as trig polynomials.
  * `yu_deriv_bound`, `Hyu_deriv_bound` — sup-norm bounds on the above.
  * Periodicity helpers `yu_periodic`, `Hyu_periodic`, `D_periodic` — these
    already live in `TightDGSupBounds.lean`; the present file re-exports them
    (`yu_periodic_re`, etc.) under stable, file-local names for callers that
    do not want to import the whole Area-D file.
  * `y_contDiff_all`, `yu_contDiff_all`, `Hyu_contDiff_all`, `D_contDiff_all`
    — `C^∞` smoothness packaged as `∀ k : ℕ, ContDiff ℝ k ·`.

All proofs are mechanical: chain rule + termwise derivative of cos/sin +
triangle inequality.
-/

namespace WaveCAP

open Real

variable {N : ℕ} (a : Fin N → ℝ)

-- ===========================================================================
-- §1. Smoothness of `y`, `y_u`, `H[y_u]`, `D` as `C^∞` packages
-- ===========================================================================

/-- `y(u) = ∑ a_k · cos(k u)` is `C^∞`. -/
theorem y_contDiff {n : ℕ∞} : ContDiff ℝ n (y a) := by
  unfold y
  apply ContDiff.sum
  intro k _
  have hcos : ContDiff ℝ n (fun u : ℝ => Real.cos ((k : ℝ) * u)) :=
    Real.contDiff_cos.comp (contDiff_const.mul contDiff_id)
  exact contDiff_const.mul hcos

/-- `y` is `C^k` for every natural `k`. -/
theorem y_contDiff_all : ∀ k : ℕ, ContDiff ℝ k (y a) := fun _ => y_contDiff a

/-- `y_u` is `C^k` for every natural `k`. -/
theorem yu_contDiff_all : ∀ k : ℕ, ContDiff ℝ k (yu a) :=
  fun k => yu_contDiff a (n := (k : ℕ∞))

/-- `H[y_u]` is `C^k` for every natural `k`. -/
theorem Hyu_contDiff_all : ∀ k : ℕ, ContDiff ℝ k (Hyu a) :=
  fun k => Hyu_contDiff a (n := (k : ℕ∞))

/-- `D` is `C^k` for every natural `k`, given a positive lower bound. -/
theorem D_contDiff_all
    {DLower : ℝ} (hD : 0 < DLower) (hcert : ∀ u, DLower ≤ D a u) :
    ∀ k : ℕ, ContDiff ℝ k (D a) :=
  fun k => D_contdiff a hD hcert (n := (k : ℕ∞))

-- ===========================================================================
-- §2. Periodicity re-exports (already proved in TightDGSupBounds)
-- ===========================================================================

/-- `y` is `2π`-periodic. -/
theorem y_periodic : Function.Periodic (y a) (2 * Real.pi) := by
  intro x
  unfold y
  apply Finset.sum_congr rfl
  intro k _
  rw [cos_nat_mul_periodic_eq (k : ℕ) x]

/-- `y_u` is `2π`-periodic (re-exported from `TightDGSupBounds`). -/
theorem yu_periodic_re : Function.Periodic (yu a) (2 * Real.pi) := yu_periodic a

/-- `H[y_u]` is `2π`-periodic (re-exported from `TightDGSupBounds`). -/
theorem Hyu_periodic_re : Function.Periodic (Hyu a) (2 * Real.pi) := Hyu_periodic a

/-- `D` is `2π`-periodic (re-exported from `TightDGSupBounds`). -/
theorem D_periodic_re : Function.Periodic (D a) (2 * Real.pi) := D_periodic a

-- ===========================================================================
-- §3. Closed-form derivatives of `y_u` and `H[y_u]`
-- ===========================================================================

/-- `(d/du) y_u(u) = -∑_k (k * a_k) · k · cos(k u)`. -/
theorem yu_deriv_eq (u : ℝ) :
    deriv (yu a) u =
      -∑ k : Fin N, (k : ℝ) * a k * ((k : ℝ) * Real.cos ((k : ℝ) * u)) := by
  -- yu = - ∑_k (k * a_k) * sin(k ·)
  unfold yu
  rw [deriv.fun_neg]
  congr 1
  -- deriv (∑_k …) = ∑_k deriv (…)
  rw [deriv_fun_sum]
  · apply Finset.sum_congr rfl
    intro k _
    -- Each term: (k * a_k) * sin(k u). Use deriv_const_mul and chain rule.
    have hsin : HasDerivAt (fun u : ℝ => Real.sin ((k : ℝ) * u))
        ((k : ℝ) * Real.cos ((k : ℝ) * u)) u := by
      have h_id : HasDerivAt (fun u : ℝ => (k : ℝ) * u) (k : ℝ) u := by
        simpa using (hasDerivAt_id u).const_mul (k : ℝ)
      simpa [mul_comm] using h_id.sin
    have hmul := hsin.const_mul ((k : ℝ) * a k)
    -- Convert HasDerivAt to deriv.
    exact hmul.deriv
  · intro k _
    -- Differentiability at u: const * sin(k u) is differentiable.
    have hcos : ContDiff ℝ (1 : ℕ∞) (fun u : ℝ => Real.sin ((k : ℝ) * u)) :=
      Real.contDiff_sin.comp (contDiff_const.mul contDiff_id)
    have hcd : ContDiff ℝ (1 : ℕ∞)
        (fun u : ℝ => (k : ℝ) * a k * Real.sin ((k : ℝ) * u)) :=
      contDiff_const.mul hcos
    exact hcd.differentiable (by decide) u

/-- `(d/du) H[y_u](u) = -∑_k (k * a_k) · k · sin(k u)`. -/
theorem Hyu_deriv_eq (u : ℝ) :
    deriv (Hyu a) u =
      ∑ k : Fin N, (k : ℝ) * a k * (-((k : ℝ) * Real.sin ((k : ℝ) * u))) := by
  unfold Hyu
  rw [deriv_fun_sum]
  · apply Finset.sum_congr rfl
    intro k _
    -- (d/du) cos(k u) = -k sin(k u).
    have hcos : HasDerivAt (fun u : ℝ => Real.cos ((k : ℝ) * u))
        (-((k : ℝ) * Real.sin ((k : ℝ) * u))) u := by
      have h_id : HasDerivAt (fun u : ℝ => (k : ℝ) * u) (k : ℝ) u := by
        simpa using (hasDerivAt_id u).const_mul (k : ℝ)
      have := h_id.cos
      -- `HasDerivAt.cos` gives derivative `-sin (f x) * f' x`.
      simpa [mul_comm] using this
    have hmul := hcos.const_mul ((k : ℝ) * a k)
    exact hmul.deriv
  · intro k _
    have hcos : ContDiff ℝ (1 : ℕ∞) (fun u : ℝ => Real.cos ((k : ℝ) * u)) :=
      Real.contDiff_cos.comp (contDiff_const.mul contDiff_id)
    have hcd : ContDiff ℝ (1 : ℕ∞)
        (fun u : ℝ => (k : ℝ) * a k * Real.cos ((k : ℝ) * u)) :=
      contDiff_const.mul hcos
    exact hcd.differentiable (by decide) u

-- ===========================================================================
-- §4. Sup-norm bounds on `deriv (y_u)` and `deriv (H[y_u])`
-- ===========================================================================

/-- `|deriv (y_u) u| ≤ ∑_k k² · |a k|` uniformly in `u`. -/
theorem yu_deriv_bound :
    ∀ u, |deriv (yu a) u| ≤ ∑ k : Fin N, ((k : ℝ) ^ 2) * |a k| := by
  intro u
  rw [yu_deriv_eq a u, abs_neg]
  refine le_trans (Finset.abs_sum_le_sum_abs _ _) ?_
  apply Finset.sum_le_sum
  intro k _
  -- |k * a_k * (k * cos(k u))| = k² · |a k| · |cos(k u)| ≤ k² · |a k|.
  have hcos_abs : |Real.cos ((k : ℝ) * u)| ≤ 1 := Real.abs_cos_le_one _
  have hk_nn : (0 : ℝ) ≤ (k : ℝ) := Nat.cast_nonneg _
  have hk2_nn : (0 : ℝ) ≤ (k : ℝ) ^ 2 := by positivity
  have habs_nn : (0 : ℝ) ≤ |a k| := abs_nonneg _
  calc |(k : ℝ) * a k * ((k : ℝ) * Real.cos ((k : ℝ) * u))|
      = (k : ℝ) * |a k| * ((k : ℝ) * |Real.cos ((k : ℝ) * u)|) := by
        rw [abs_mul, abs_mul, abs_mul, abs_of_nonneg hk_nn]
    _ ≤ (k : ℝ) * |a k| * ((k : ℝ) * 1) := by
        gcongr
    _ = (k : ℝ) ^ 2 * |a k| := by ring

/-- `|deriv (H[y_u]) u| ≤ ∑_k k² · |a k|` uniformly in `u`. -/
theorem Hyu_deriv_bound :
    ∀ u, |deriv (Hyu a) u| ≤ ∑ k : Fin N, ((k : ℝ) ^ 2) * |a k| := by
  intro u
  rw [Hyu_deriv_eq a u]
  refine le_trans (Finset.abs_sum_le_sum_abs _ _) ?_
  apply Finset.sum_le_sum
  intro k _
  have hsin_abs : |Real.sin ((k : ℝ) * u)| ≤ 1 := Real.abs_sin_le_one _
  have hk_nn : (0 : ℝ) ≤ (k : ℝ) := Nat.cast_nonneg _
  calc |(k : ℝ) * a k * (-((k : ℝ) * Real.sin ((k : ℝ) * u)))|
      = (k : ℝ) * |a k| * ((k : ℝ) * |Real.sin ((k : ℝ) * u)|) := by
        rw [abs_mul, abs_mul, abs_neg, abs_mul, abs_of_nonneg hk_nn]
    _ ≤ (k : ℝ) * |a k| * ((k : ℝ) * 1) := by
        gcongr
    _ = (k : ℝ) ^ 2 * |a k| := by ring

-- ===========================================================================
-- §5. Chain rule for `D`
-- ===========================================================================

/-- Differentiability of `yu` at every point (consequence of `C^∞`). -/
private lemma yu_differentiable : Differentiable ℝ (yu a) :=
  (yu_contDiff a (n := (1 : ℕ∞))).differentiable (by decide)

/-- Differentiability of `Hyu` at every point. -/
private lemma Hyu_differentiable : Differentiable ℝ (Hyu a) :=
  (Hyu_contDiff a (n := (1 : ℕ∞))).differentiable (by decide)

/-- **Chain rule for `D`.** With `D = √(y_u² + (1 + H[y_u])²)`, if `D u > 0`
then
```
(d/du) D(u) = (y_u(u) · y_u'(u) + (1 + H[y_u](u)) · H[y_u]'(u)) / D(u).
```
The proof applies `Real.deriv_sqrt` with positivity of the argument and the
sum/chain rules on the inside.

This closes the `h_D_deriv_eq` hoist hypothesis on
`D_lipschitz_bound`/`dmin_bound_correct` in `DminBoundCorrectness.lean`. -/
theorem D_deriv_eq (u : ℝ) (hD_pos : 0 < D a u) :
    deriv (D a) u =
      (yu a u * deriv (yu a) u + (1 + Hyu a u) * deriv (Hyu a) u) / D a u := by
  -- Rewrite D pointwise as a square root and apply the sqrt chain rule.
  have hD_def : (D a) = fun v => Real.sqrt ((yu a v) ^ 2 + (1 + Hyu a v) ^ 2) := by
    funext v; rfl
  -- The argument of √ at u is nonzero (= D u² > 0).
  have hArg_eq : (yu a u) ^ 2 + (1 + Hyu a u) ^ 2 = (D a u) ^ 2 := (D_squared_eq a u).symm
  have hArg_ne : (yu a u) ^ 2 + (1 + Hyu a u) ^ 2 ≠ 0 := by
    rw [hArg_eq]; exact pow_ne_zero 2 hD_pos.ne'
  -- Differentiability of the inner.
  have h_yu_diff : DifferentiableAt ℝ (yu a) u := yu_differentiable a u
  have h_Hyu_diff : DifferentiableAt ℝ (Hyu a) u := Hyu_differentiable a u
  have h_yu_sq_diff : DifferentiableAt ℝ (fun v => (yu a v) ^ 2) u :=
    h_yu_diff.pow 2
  have h_1pH_diff : DifferentiableAt ℝ (fun v => 1 + Hyu a v) u :=
    differentiableAt_const _ |>.add h_Hyu_diff
  have h_1pH_sq_diff : DifferentiableAt ℝ (fun v => (1 + Hyu a v) ^ 2) u :=
    h_1pH_diff.pow 2
  have h_inner_diff : DifferentiableAt ℝ
      (fun v => (yu a v) ^ 2 + (1 + Hyu a v) ^ 2) u :=
    h_yu_sq_diff.add h_1pH_sq_diff
  -- Apply Real.deriv_sqrt.
  rw [hD_def, deriv_sqrt h_inner_diff hArg_ne]
  -- Now compute the derivative of the inner.
  have h_inner_deriv :
      deriv (fun v => (yu a v) ^ 2 + (1 + Hyu a v) ^ 2) u
        = 2 * yu a u * deriv (yu a) u
          + 2 * (1 + Hyu a u) * deriv (Hyu a) u := by
    rw [deriv_fun_add h_yu_sq_diff h_1pH_sq_diff]
    -- deriv ((yu a)^2) = 2 * yu a u * deriv (yu a) u
    have hpow_yu : deriv (fun v => (yu a v) ^ 2) u
        = 2 * yu a u * deriv (yu a) u := by
      rw [deriv_fun_pow (n := 2) h_yu_diff]
      ring
    -- deriv ((1 + Hyu a)^2) = 2 * (1 + Hyu a u) * deriv (1 + Hyu a) u
    have hpow_H : deriv (fun v => (1 + Hyu a v) ^ 2) u
        = 2 * (1 + Hyu a u) * deriv (Hyu a) u := by
      rw [deriv_fun_pow (n := 2) h_1pH_diff]
      -- deriv (fun v => 1 + Hyu a v) u = deriv (Hyu a) u
      have h_const_add :
          deriv (fun v => 1 + Hyu a v) u = deriv (Hyu a) u := by
        rw [deriv_fun_add (differentiableAt_const _) h_Hyu_diff]
        simp
      rw [h_const_add]
      ring
    rw [hpow_yu, hpow_H]
  rw [h_inner_deriv]
  -- Final algebra: cancel the 2's. Don't use field_simp (it expands the (1+Hyu)^2
  -- on the LHS); instead, rewrite the numerator.
  rw [show (2 * yu a u * deriv (yu a) u + 2 * (1 + Hyu a u) * deriv (Hyu a) u)
        = 2 * (yu a u * deriv (yu a) u + (1 + Hyu a u) * deriv (Hyu a) u) from by ring]
  exact mul_div_mul_left _ _ two_ne_zero

end WaveCAP

-- ===========================================================================
-- Sanity checks
-- ===========================================================================

#check @WaveCAP.D_deriv_eq
#check @WaveCAP.yu_deriv_bound
#check @WaveCAP.Hyu_deriv_bound
#check @WaveCAP.yu_periodic_re
#check @WaveCAP.Hyu_periodic_re
#check @WaveCAP.D_periodic_re
#check @WaveCAP.y_contDiff_all
#check @WaveCAP.yu_contDiff_all
#check @WaveCAP.Hyu_contDiff_all
#check @WaveCAP.D_contDiff_all

#print axioms WaveCAP.D_deriv_eq
#print axioms WaveCAP.yu_deriv_bound
#print axioms WaveCAP.Hyu_deriv_bound
