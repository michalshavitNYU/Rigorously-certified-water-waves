import WaveCAP.Common

/-!
# N5 (pointwise) — Taylor-remainder bound on a subinterval

Port of `CCFLeanPOC.PathBeta.abs_taylor_remainder_le` from
`ccf-lean-poc/CCFLeanPOC/PathBeta/Auxiliary/TaylorRemainderSubinterval.lean`
(commit verified sorry-free against Mathlib `v4.29.0-rc6`).

Statement (Lagrange form, using mathlib's `taylorWithinEval`):

```
abs_taylor_remainder_le {f : ℝ → ℝ} {a b : ℝ} (hab : a ≤ b) {M : ℕ}
    (hf : ContDiffOn ℝ (M + 1) f (Set.Icc a b)) {F : ℝ}
    (hF : ∀ t ∈ Set.Icc a b, |iteratedDerivWithin (M + 1) f (Set.Icc a b) t| ≤ F)
    {t : ℝ} (ht : t ∈ Set.Icc a b) :
    |f t - taylorPoly f a b M t| ≤ F * (t - a) ^ (M + 1) / (Nat.factorial (M + 1) : ℝ)
```

`taylorPoly` is defined as mathlib's `taylorWithinEval f M (Icc a b) a` so the
hypothesis is exactly what `taylor_mean_remainder_lagrange_iteratedDeriv`
expects.  The bridge lemma `iteratedDerivWithin_Icc_left_eq` lets us evaluate
the Lagrange remainder on the shorter interval `Icc a t` and transport back to
`Icc a b` (which is what we want for the grid-max wrapper).

This is the **pointwise** piece. The grid-max wrapper that lifts this to a sup
over a periodic grid lives in `WaveCAP/GridToSupNormTaylor.lean`.

Cross-reference: `BabenkoRigorous/src/TaylorSeries.jl` `sup_bound(ts)` (lines
~80–113) — the Julia routine that bounds `‖f^{(M+1)}‖_∞ · h^{M+1} / (M+1)!`
per grid point and takes a max, whose mathematical correctness this file's
headline lemma supplies.

Watch-points (from the CCF smoke-test agent):
  1. The denominator is `(M + 1)!` here (pointwise Lagrange); the *integral*
     form bumps to `(M + 2)!`.  We are not porting the integral form.
  2. `iteratedDerivWithin` (not `iteratedDeriv`) is needed at boundary points —
     the bridge handles the within/global swap at the interior point `t'`.
-/

namespace WaveCAP

open Real intervalIntegral MeasureTheory Set

/-- Pointwise Taylor expansion at `a` over `Icc a b`, as `taylorWithinEval`.

Note: this differs from a "plain `iteratedDeriv`" expansion at boundary points
unless `f` extends smoothly outside `Icc a b`. We use the within-version since
that's what `taylor_mean_remainder_lagrange_iteratedDeriv` accepts. -/
noncomputable def taylorPoly (f : ℝ → ℝ) (a b : ℝ) (M : ℕ) (t : ℝ) : ℝ :=
  taylorWithinEval f M (Set.Icc a b) a t

/-- Sum form for `taylorPoly` from `taylor_within_apply`.

Useful for the grid-max wrapper (`grid_to_supNorm_taylor`), which needs to see
`taylorPoly` as a finite sum of `iteratedDerivWithin` evaluated at `a`. -/
lemma taylorPoly_apply (f : ℝ → ℝ) (a b : ℝ) (M : ℕ) (t : ℝ) :
    taylorPoly f a b M t =
      ∑ k ∈ Finset.range (M + 1),
        ((Nat.factorial k : ℝ)⁻¹ * (t - a) ^ k) *
          iteratedDerivWithin k f (Set.Icc a b) a := by
  unfold taylorPoly
  rw [taylor_within_apply]
  simp [smul_eq_mul]

/-- Bridge lemma: for `a < t ≤ b`, the within-derivatives at `a` agree between
`Icc a t` and `Icc a b`. This works because `Icc a t = Icc a b ∩ Iic t` and
`Iic t` is a neighborhood of `a` (since `a < t`). -/
lemma iteratedDerivWithin_Icc_left_eq {f : ℝ → ℝ} {a t b : ℝ}
    (hat : a < t) (htb : t ≤ b) (k : ℕ) :
    iteratedDerivWithin k f (Set.Icc a t) a = iteratedDerivWithin k f (Set.Icc a b) a := by
  unfold iteratedDerivWithin
  congr 1
  -- iteratedFDerivWithin k f (Icc a t) a = iteratedFDerivWithin k f (Icc a b) a
  have hinter : Set.Icc a t = Set.Icc a b ∩ Set.Iic t := by
    ext x
    simp only [Set.mem_Icc, Set.mem_inter_iff, Set.mem_Iic]
    refine ⟨fun h => ⟨⟨h.1, h.2.trans htb⟩, h.2⟩, fun h => ⟨h.1.1, h.2⟩⟩
  rw [hinter]
  apply iteratedFDerivWithin_inter
  -- Iic t ∈ 𝓝 a: need a < t.
  exact Iic_mem_nhds hat

/-- Bridge: `taylorWithinEval f M (Icc a t) a x = taylorWithinEval f M (Icc a b) a x`
when `a < t ≤ b`. -/
lemma taylorWithinEval_Icc_eq {f : ℝ → ℝ} {a t b : ℝ}
    (hat : a < t) (htb : t ≤ b) (M : ℕ) (x : ℝ) :
    taylorWithinEval f M (Set.Icc a t) a x = taylorWithinEval f M (Set.Icc a b) a x := by
  rw [taylor_within_apply, taylor_within_apply]
  apply Finset.sum_congr rfl
  intro k _
  rw [iteratedDerivWithin_Icc_left_eq hat htb k]

/-- Pointwise Taylor-remainder bound, Lagrange form.

For `f ∈ C^{M+1}` on `Icc a b` with `|f^{(M+1)}| ≤ F` (within), the remainder
at any `t ∈ Icc a b` is bounded by `F · (t-a)^{M+1} / (M+1)!`. -/
theorem abs_taylor_remainder_le {f : ℝ → ℝ} {a b : ℝ} (hab : a ≤ b) {M : ℕ}
    (hf : ContDiffOn ℝ (M + 1) f (Set.Icc a b)) {F : ℝ}
    (hF : ∀ t ∈ Set.Icc a b, |iteratedDerivWithin (M + 1) f (Set.Icc a b) t| ≤ F)
    {t : ℝ} (ht : t ∈ Set.Icc a b) :
    |f t - taylorPoly f a b M t| ≤ F * (t - a) ^ (M + 1) / (Nat.factorial (M + 1) : ℝ) := by
  -- Special case t = a.
  rcases eq_or_lt_of_le ht.1 with hta | hat
  · subst hta
    unfold taylorPoly
    rw [taylorWithinEval_self, sub_self, abs_zero]
    apply div_nonneg
    · apply mul_nonneg _ (pow_nonneg (by linarith) _)
      have := hF a ⟨le_refl _, hab⟩
      linarith [abs_nonneg (iteratedDerivWithin (M + 1) f (Set.Icc a b) a)]
    · exact_mod_cast Nat.zero_le _
  -- Main case a < t.
  -- Apply Lagrange form on (Icc a t).
  have hsubset : Set.Icc a t ⊆ Set.Icc a b := Set.Icc_subset_Icc_right ht.2
  have hf_at : ContDiffOn ℝ (M + 1) f (Set.Icc a t) := hf.mono hsubset
  obtain ⟨t', ht', heq⟩ :=
    taylor_mean_remainder_lagrange_iteratedDeriv hat hf_at
  -- heq : f t - taylorWithinEval f M (Icc a t) a t =
  --       iteratedDeriv (M+1) f t' * (t-a)^(M+1) / (M+1)!
  rw [taylorWithinEval_Icc_eq hat ht.2] at heq
  -- t' ∈ Ioo a t ⊆ Icc a b. Convert iteratedDeriv to iteratedDerivWithin.
  have ht'_mem : t' ∈ Set.Icc a b := ⟨ht'.1.le, ht'.2.le.trans ht.2⟩
  have ht'_in_open_b : t' ∈ Set.Ioo a b := ⟨ht'.1, ht'.2.trans_le ht.2⟩
  have hu : UniqueDiffOn ℝ (Set.Icc a b) := uniqueDiffOn_Icc (lt_of_lt_of_le hat ht.2)
  have hcontDiffAt : ContDiffAt ℝ (M + 1) f t' :=
    hf.contDiffAt (Icc_mem_nhds_iff.2 ht'_in_open_b)
  have hbridge_t' : iteratedDeriv (M + 1) f t' =
      iteratedDerivWithin (M + 1) f (Set.Icc a b) t' :=
    (iteratedDerivWithin_eq_iteratedDeriv hu hcontDiffAt ht'_mem).symm
  rw [hbridge_t'] at heq
  -- Now use heq to compute |f t - taylorPoly f a b M t|.
  unfold taylorPoly
  rw [heq]
  -- Goal: |iteratedDerivWithin (M+1) f (Icc a b) t' * (t-a)^(M+1) / (M+1)!|
  --        ≤ F * (t-a)^(M+1) / (M+1)!
  have ht_a_nonneg : (0 : ℝ) ≤ t - a := sub_nonneg.mpr ht.1
  have hpow_nonneg : (0 : ℝ) ≤ (t - a) ^ (M + 1) := pow_nonneg ht_a_nonneg _
  have hfact_pos : (0 : ℝ) < (Nat.factorial (M + 1) : ℝ) := by
    exact_mod_cast Nat.factorial_pos _
  rw [abs_div, abs_mul, abs_pow, abs_of_nonneg ht_a_nonneg,
      abs_of_pos hfact_pos]
  apply div_le_div_of_nonneg_right
  · -- |iteratedDerivWithin| · (t-a)^(M+1) ≤ F · (t-a)^(M+1)
    apply mul_le_mul_of_nonneg_right (hF t' ht'_mem) hpow_nonneg
  · exact hfact_pos.le

#check @abs_taylor_remainder_le
#check @taylorPoly
#check @taylorPoly_apply

end WaveCAP

#print axioms WaveCAP.abs_taylor_remainder_le
