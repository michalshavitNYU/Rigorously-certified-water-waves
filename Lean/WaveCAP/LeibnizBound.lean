import WaveCAP.Common

/-!
# N2 — Leibniz product bound on iterated derivatives

Pointwise bound on `|(f·g)^{(n)}|` in terms of per-derivative bounds on `f` and
`g`.  This is the analytic content the Julia routine
`Base.:*(ts1::TaylorSeries, ts2::TaylorSeries)` relies on when it computes

```
new.sup_bounds[m+1] = Σ_{k=0}^m C(m, k) · ts1.sup_bounds[k+1] · ts2.sup_bounds[m-k+1]
```

(see `BabenkoRigorous/src/TaylorSeries.jl`, lines ~283–340).

The Lean theorem is the pointwise version (no sup-norm formalism needed):

```
|iteratedDeriv n (f * g) x|
  ≤ Σ_{k=0..n} C(n, k) · F k · G (n - k)
```

given per-`k` bounds `|f^(k)(x)| ≤ F k` and `|g^(k)(x)| ≤ G k`.  Taking the
sup over `x` gives the sup-norm version directly.
-/

namespace WaveCAP

open Real

/-- Pointwise Leibniz product bound on iterated derivatives.

For `f, g ∈ C^n` at `x` with `|f^(k)(x)| ≤ F k` and `|g^(k)(x)| ≤ G k`
for all `k ≤ n`, the iterated derivative of the product is bounded by the
Leibniz convolution: `Σ C(n, k) · F k · G (n - k)`.

Cross-reference: `BabenkoRigorous/src/TaylorSeries.jl` `Base.:*` (lines ~283–340).
-/
theorem abs_iteratedDeriv_mul_le
    {f g : ℝ → ℝ} {x : ℝ} {n : ℕ}
    (hf : ContDiffAt ℝ n f x) (hg : ContDiffAt ℝ n g x)
    {F G : ℕ → ℝ}
    (hF : ∀ k ≤ n, |iteratedDeriv k f x| ≤ F k)
    (hG : ∀ k ≤ n, |iteratedDeriv k g x| ≤ G k) :
    |iteratedDeriv n (f * g) x| ≤
      ∑ k ∈ Finset.range (n + 1), (n.choose k : ℝ) * F k * G (n - k) := by
  rw [iteratedDeriv_mul hf hg]
  calc |∑ k ∈ Finset.range (n + 1),
          (n.choose k : ℝ) * iteratedDeriv k f x * iteratedDeriv (n - k) g x|
      ≤ ∑ k ∈ Finset.range (n + 1),
            |(n.choose k : ℝ) * iteratedDeriv k f x * iteratedDeriv (n - k) g x| :=
          Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ k ∈ Finset.range (n + 1), (n.choose k : ℝ) * F k * G (n - k) := by
        apply Finset.sum_le_sum
        intro k hk_mem
        have hk_le_n : k ≤ n := Nat.lt_succ_iff.mp (Finset.mem_range.mp hk_mem)
        have hnk_le_n : n - k ≤ n := Nat.sub_le n k
        have hFk : |iteratedDeriv k f x| ≤ F k := hF k hk_le_n
        have hGk : |iteratedDeriv (n - k) g x| ≤ G (n - k) := hG (n - k) hnk_le_n
        have hFk_nonneg : 0 ≤ F k := le_trans (abs_nonneg _) hFk
        have hGk_nonneg : 0 ≤ G (n - k) := le_trans (abs_nonneg _) hGk
        have hbinom_nonneg : (0 : ℝ) ≤ (n.choose k : ℝ) := Nat.cast_nonneg _
        calc |(n.choose k : ℝ) * iteratedDeriv k f x * iteratedDeriv (n - k) g x|
            = (n.choose k : ℝ) * |iteratedDeriv k f x| *
                |iteratedDeriv (n - k) g x| := by
              rw [abs_mul, abs_mul, abs_of_nonneg hbinom_nonneg]
          _ ≤ (n.choose k : ℝ) * F k * G (n - k) := by
              apply mul_le_mul _ hGk (abs_nonneg _)
                (mul_nonneg hbinom_nonneg hFk_nonneg)
              exact mul_le_mul_of_nonneg_left hFk hbinom_nonneg

#check @abs_iteratedDeriv_mul_le

end WaveCAP

#print axioms WaveCAP.abs_iteratedDeriv_mul_le
