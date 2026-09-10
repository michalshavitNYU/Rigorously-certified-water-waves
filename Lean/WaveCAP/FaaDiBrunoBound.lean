import WaveCAP.Common

/-!
# N1 — Faà di Bruno sup-norm bound for `g ∘ f`

Pointwise bound on `|(g ∘ f)^{(n)}|` in terms of per-derivative bounds on
`g` (evaluated at `f x`) and on `f`.  This is the analytic content the Julia
routine `_faa_di_bruno_derivs` relies on when it propagates `sup_bounds` for
`inv_series` (`φ = 1/x`), `sqrt_series` (`φ = √x`), and any other
`ArbSeries`-composition.

The Lean statement parallels mathlib's
`iteratedDeriv_comp_eq_sum_orderedFinpartition`:

```
|iteratedDeriv n (g ∘ f) x|
  ≤ ∑ c : OrderedFinpartition n, Φ c.length * ∏ j, F (c.partSize j)
```

given per-`k` bounds `|g^(k)(f x)| ≤ Φ k` and `|f^(k)(x)| ≤ F k`.  Taking the
sup over `x` (with `Φ` interpreted as `sup over y ∈ range f ∘ Icc` of `|g^(k) y|`)
gives the sup-norm version.

Cross-reference: `BabenkoRigorous/src/TaylorSeries.jl` `_faa_di_bruno_derivs`,
`_inv_sup_bounds_on_interval`, `_sqrt_sup_bounds_on_interval`.
-/

namespace WaveCAP

open Real OrderedFinpartition

/-- Pointwise Faà di Bruno bound for the iterated derivative of a composition.

Mirrors the structure of mathlib's
`iteratedDeriv_comp_eq_sum_orderedFinpartition`: take `|·|`, push it through
the sum and the per-partition product, then apply the per-derivative bounds. -/
theorem abs_iteratedDeriv_comp_le
    {f g : ℝ → ℝ} {x : ℝ} {n : ℕ}
    (hg : ContDiffAt ℝ n g (f x)) (hf : ContDiffAt ℝ n f x)
    {Φ F : ℕ → ℝ}
    (hΦ : ∀ k ≤ n, |iteratedDeriv k g (f x)| ≤ Φ k)
    (hF : ∀ k ≤ n, |iteratedDeriv k f x| ≤ F k) :
    |iteratedDeriv n (g ∘ f) x| ≤
      ∑ c : OrderedFinpartition n, Φ c.length * ∏ j, F (c.partSize j) := by
  rw [iteratedDeriv_comp_eq_sum_orderedFinpartition (i := n) hg hf le_rfl]
  calc |∑ c : OrderedFinpartition n,
          iteratedDeriv c.length g (f x) * ∏ j, iteratedDeriv (c.partSize j) f x|
      ≤ ∑ c : OrderedFinpartition n,
            |iteratedDeriv c.length g (f x) * ∏ j, iteratedDeriv (c.partSize j) f x| :=
          Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ c : OrderedFinpartition n, Φ c.length * ∏ j, F (c.partSize j) := by
        apply Finset.sum_le_sum
        intro c _
        -- |g^(length) · ∏ f^(partSize j)| = |g^(length)| · ∏ |f^(partSize j)|
        rw [abs_mul, Finset.abs_prod]
        -- Bound the g-factor.
        have hgΦ : |iteratedDeriv c.length g (f x)| ≤ Φ c.length :=
          hΦ c.length c.length_le
        -- Bound each f-factor.
        have hf_each : ∀ j ∈ (Finset.univ : Finset (Fin c.length)),
            |iteratedDeriv (c.partSize j) f x| ≤ F (c.partSize j) :=
          fun j _ => hF (c.partSize j) (c.partSize_le j)
        -- Each |·| factor is nonneg (used for prod_le_prod).
        have hF_nonneg : ∀ j ∈ (Finset.univ : Finset (Fin c.length)),
            (0 : ℝ) ≤ |iteratedDeriv (c.partSize j) f x| :=
          fun _ _ => abs_nonneg _
        -- Φ c.length ≥ 0
        have hΦ_nonneg : 0 ≤ Φ c.length :=
          le_trans (abs_nonneg _) hgΦ
        -- ∏ |·| ≥ 0
        have hprod_nonneg : (0 : ℝ) ≤ ∏ j, |iteratedDeriv (c.partSize j) f x| :=
          Finset.prod_nonneg fun _ _ => abs_nonneg _
        -- Combine.
        calc |iteratedDeriv c.length g (f x)| *
                ∏ j, |iteratedDeriv (c.partSize j) f x|
            ≤ Φ c.length * ∏ j, |iteratedDeriv (c.partSize j) f x| :=
              mul_le_mul_of_nonneg_right hgΦ hprod_nonneg
          _ ≤ Φ c.length * ∏ j, F (c.partSize j) := by
              apply mul_le_mul_of_nonneg_left _ hΦ_nonneg
              exact Finset.prod_le_prod hF_nonneg hf_each

#check @abs_iteratedDeriv_comp_le

end WaveCAP

#print axioms WaveCAP.abs_iteratedDeriv_comp_le
