import WaveCAP.TaylorRemainder

/-!
# N5 (grid-max wrapper) — sup-norm bound from per-interval Taylor data

Lifts the pointwise Taylor-remainder bound `WaveCAP.abs_taylor_remainder_le`
to a sup-norm bound on `[a, a + N·h]` partitioned into `N` equispaced
sub-intervals of length `h`.

The headline (matching `BabenkoRigorous/src/TaylorSeries.jl` `sup_bound(ts)`):

```
∀ u ∈ [a, a + N·h], |f u|
  ≤ max_{j ∈ Fin N}
        Σ_{k=0..M} |f^{(k)}(u_j)| · h^k / k!
      + F · h^{M+1} / (M+1)!
```

where `u_j = a + j·h`, and the bound at index `j` is exactly the inner sum the
Julia routine computes for the `j`-th grid interval.

Statement-side, we expose this as
`∀ u ∈ Icc a (a + N·h), |f u| ≤ M_bound`
under the hypothesis `∀ j, per-interval-bound j ≤ M_bound`.  The user picks
`M_bound = max_j (per-interval-bound j)` and discharges the max separately.

Cross-reference: `BabenkoRigorous/src/TaylorSeries.jl` `sup_bound(ts)`
(lines ~120–167).
-/

namespace WaveCAP

open Real intervalIntegral MeasureTheory Set

-- ===========================================================================
-- Per-sub-interval bound (parametrised on `Icc a b`, with `(b - a)` in place
-- of the interval width)
-- ===========================================================================

/-- For `t ∈ [a, b]` with `f ∈ C^{M+1}` on `Icc a b`, the value `|f t|` is
bounded by the sum of Taylor-coefficient absolute values at `a`, each weighted
by `(b-a)^k / k!`, plus the Lagrange-remainder bound. -/
lemma abs_f_le_subinterval_taylor_sum
    {f : ℝ → ℝ} {a b : ℝ} (hab : a ≤ b) {M : ℕ}
    (hf : ContDiffOn ℝ (M + 1) f (Set.Icc a b))
    {F : ℝ} (hF : ∀ s ∈ Set.Icc a b,
                  |iteratedDerivWithin (M + 1) f (Set.Icc a b) s| ≤ F)
    {t : ℝ} (ht : t ∈ Set.Icc a b) :
    |f t| ≤
      (∑ k ∈ Finset.range (M + 1),
          |iteratedDerivWithin k f (Set.Icc a b) a|
            * (b - a) ^ k / (Nat.factorial k : ℝ))
      + F * (b - a) ^ (M + 1) / (Nat.factorial (M + 1) : ℝ) := by
  have hba_nonneg : (0 : ℝ) ≤ b - a := sub_nonneg.mpr hab
  have ht_a_nonneg : (0 : ℝ) ≤ t - a := sub_nonneg.mpr ht.1
  have ht_a_le_h : t - a ≤ b - a := by have := ht.2; linarith
  -- Pointwise Taylor remainder.
  have hrem := abs_taylor_remainder_le hab hf hF ht
  -- Sum-form for taylorPoly.
  have hsum := taylorPoly_apply f a b M t
  -- F nonneg (since F bounds |·| at a).
  have hF_nonneg : 0 ≤ F := by
    have := hF a ⟨le_refl _, hab⟩
    linarith [abs_nonneg (iteratedDerivWithin (M + 1) f (Set.Icc a b) a)]
  have hfact_pos : ∀ k : ℕ, (0 : ℝ) < (Nat.factorial k : ℝ) := by
    intro k; exact_mod_cast Nat.factorial_pos _
  -- Bound the remainder via (b-a)^(M+1) (monotone in (t-a)).
  have hrem' : |f t - taylorPoly f a b M t|
      ≤ F * (b - a) ^ (M + 1) / (Nat.factorial (M + 1) : ℝ) := by
    calc |f t - taylorPoly f a b M t|
        ≤ F * (t - a) ^ (M + 1) / (Nat.factorial (M + 1) : ℝ) := hrem
      _ ≤ F * (b - a) ^ (M + 1) / (Nat.factorial (M + 1) : ℝ) := by
        apply div_le_div_of_nonneg_right _ (hfact_pos (M + 1)).le
        apply mul_le_mul_of_nonneg_left _ hF_nonneg
        exact pow_le_pow_left₀ ht_a_nonneg ht_a_le_h (M + 1)
  -- Bound the Taylor polynomial as a sum.
  have hpoly_le : |taylorPoly f a b M t|
      ≤ ∑ k ∈ Finset.range (M + 1),
          |iteratedDerivWithin k f (Set.Icc a b) a|
            * (b - a) ^ k / (Nat.factorial k : ℝ) := by
    rw [hsum]
    calc
      |∑ k ∈ Finset.range (M + 1),
        ((Nat.factorial k : ℝ)⁻¹ * (t - a) ^ k) *
          iteratedDerivWithin k f (Set.Icc a b) a|
          ≤ ∑ k ∈ Finset.range (M + 1),
              |((Nat.factorial k : ℝ)⁻¹ * (t - a) ^ k) *
                iteratedDerivWithin k f (Set.Icc a b) a| :=
            Finset.abs_sum_le_sum_abs _ _
      _ = ∑ k ∈ Finset.range (M + 1),
            (Nat.factorial k : ℝ)⁻¹ * (t - a) ^ k *
              |iteratedDerivWithin k f (Set.Icc a b) a| := by
          apply Finset.sum_congr rfl
          intro k _
          rw [abs_mul, abs_mul]
          have hfac_inv_pos : (0 : ℝ) < (Nat.factorial k : ℝ)⁻¹ :=
            inv_pos.mpr (hfact_pos k)
          rw [abs_of_pos hfac_inv_pos, abs_pow, abs_of_nonneg ht_a_nonneg]
      _ ≤ ∑ k ∈ Finset.range (M + 1),
            (Nat.factorial k : ℝ)⁻¹ * (b - a) ^ k *
              |iteratedDerivWithin k f (Set.Icc a b) a| := by
          apply Finset.sum_le_sum
          intro k _
          apply mul_le_mul_of_nonneg_right
          · apply mul_le_mul_of_nonneg_left
              (pow_le_pow_left₀ ht_a_nonneg ht_a_le_h k)
              (le_of_lt (inv_pos.mpr (hfact_pos k)))
          · exact abs_nonneg _
      _ = ∑ k ∈ Finset.range (M + 1),
            |iteratedDerivWithin k f (Set.Icc a b) a| *
              (b - a) ^ k / (Nat.factorial k : ℝ) := by
          apply Finset.sum_congr rfl
          intro k _
          rw [div_eq_mul_inv]
          ring
  -- Combine: |f t| ≤ |taylorPoly| + |f t - taylorPoly|.
  have hsplit : |f t|
      ≤ |taylorPoly f a b M t| + |f t - taylorPoly f a b M t| := by
    calc |f t|
        = |taylorPoly f a b M t + (f t - taylorPoly f a b M t)| := by
          congr 1; ring
      _ ≤ |taylorPoly f a b M t| + |f t - taylorPoly f a b M t| :=
          abs_add_le _ _
  linarith

-- ===========================================================================
-- Grid covering: ∀ u ∈ [a, a+N·h], ∃ j ∈ Fin N, u ∈ [a + j·h, a + (j+1)·h]
-- ===========================================================================

/-- Exhibit a sub-interval containing `u`. -/
lemma exists_grid_subinterval
    {a : ℝ} {N : ℕ} (hN : 0 < N) {h : ℝ} (hh : 0 < h)
    {u : ℝ} (hu : u ∈ Set.Icc a (a + N * h)) :
    ∃ j : Fin N, u ∈ Set.Icc (a + (j : ℕ) * h) (a + ((j : ℕ) + 1) * h) := by
  obtain ⟨hu_lo, hu_hi⟩ := hu
  have hu_a_nonneg : (0 : ℝ) ≤ u - a := sub_nonneg.mpr hu_lo
  set q : ℝ := (u - a) / h with hq_def
  have hq_nonneg : (0 : ℝ) ≤ q := div_nonneg hu_a_nonneg hh.le
  -- Two cases: u = a + N·h (right endpoint) vs u < a + N·h.
  rcases eq_or_lt_of_le hu_hi with hu_eq | hu_lt
  · -- u = a + N·h: use j = N - 1.
    refine ⟨⟨N - 1, Nat.sub_lt hN (by norm_num)⟩, ?_, ?_⟩
    · -- a + (N-1)·h ≤ u = a + N·h
      have hN_ge_1 : 1 ≤ (N : ℝ) := by exact_mod_cast hN
      have hcast : (↑(N - 1) : ℝ) = (N : ℝ) - 1 := by
        rw [Nat.cast_sub hN]; push_cast; ring
      rw [hcast, hu_eq]
      have : ((N : ℝ) - 1) * h ≤ (N : ℝ) * h :=
        mul_le_mul_of_nonneg_right (by linarith) hh.le
      linarith
    · -- u = a + N·h ≤ a + (N-1+1)·h = a + N·h
      rw [hu_eq]
      have hcast : (↑(N - 1) : ℝ) + 1 = (N : ℝ) := by
        rw [Nat.cast_sub hN]; push_cast; ring
      rw [hcast]
  · -- u < a + N·h: use j = ⌊q⌋₊.
    have hq_lt_N : q < N := by
      rw [hq_def, div_lt_iff₀ hh]; linarith
    have hfloor_lt : ⌊q⌋₊ < N := (Nat.floor_lt hq_nonneg).mpr hq_lt_N
    refine ⟨⟨⌊q⌋₊, hfloor_lt⟩, ?_, ?_⟩
    · -- a + ⌊q⌋·h ≤ u
      have h1 : (⌊q⌋₊ : ℝ) ≤ q := Nat.floor_le hq_nonneg
      have h2 : (⌊q⌋₊ : ℝ) * h ≤ q * h := mul_le_mul_of_nonneg_right h1 hh.le
      have h3 : q * h = u - a := by rw [hq_def]; field_simp
      linarith
    · -- u ≤ a + (⌊q⌋ + 1)·h
      have h1 : q < (⌊q⌋₊ : ℝ) + 1 := Nat.lt_floor_add_one q
      have h2 : q * h < ((⌊q⌋₊ : ℝ) + 1) * h := mul_lt_mul_of_pos_right h1 hh
      have h3 : q * h = u - a := by rw [hq_def]; field_simp
      linarith

-- ===========================================================================
-- Headline: grid-max sup-norm wrapper
-- ===========================================================================

/-- Grid-max wrapper (the N5 sup-norm statement).

For `f` smooth on each sub-interval `[a + j·h, a + (j+1)·h]` with derivative
bound `F` on `f^{(M+1)}` and a per-interval Taylor-sum bound `≤ M_bound`,
the sup-norm of `|f|` on `[a, a + N·h]` is bounded by `M_bound`.

This is the Lean side of `BabenkoRigorous/src/TaylorSeries.jl` `sup_bound(ts)`:
the Julia routine computes the per-interval sum + remainder bound for each
`j`, takes the Arb-`max`, and returns it; this theorem says any such
upper-bound `M_bound` is a true ‖f‖_∞ bound. -/
theorem grid_to_supNorm_taylor
    {f : ℝ → ℝ} {a : ℝ} {N M : ℕ} (hN : 0 < N) {h : ℝ} (hh : 0 < h)
    (hf : ∀ j : Fin N, ContDiffOn ℝ (M + 1) f
                          (Set.Icc (a + (j : ℕ) * h) (a + ((j : ℕ) + 1) * h)))
    {F : ℝ} (hF : ∀ j : Fin N,
        ∀ s ∈ Set.Icc (a + (j : ℕ) * h) (a + ((j : ℕ) + 1) * h),
            |iteratedDerivWithin (M + 1) f
                (Set.Icc (a + (j : ℕ) * h) (a + ((j : ℕ) + 1) * h)) s| ≤ F)
    {M_bound : ℝ}
    (hM_bound : ∀ j : Fin N,
        (∑ k ∈ Finset.range (M + 1),
          |iteratedDerivWithin k f
              (Set.Icc (a + (j : ℕ) * h) (a + ((j : ℕ) + 1) * h))
              (a + (j : ℕ) * h)|
            * h ^ k / (Nat.factorial k : ℝ))
        + F * h ^ (M + 1) / (Nat.factorial (M + 1) : ℝ) ≤ M_bound) :
    ∀ u ∈ Set.Icc a (a + N * h), |f u| ≤ M_bound := by
  intro u hu
  obtain ⟨j, hu_mem⟩ := exists_grid_subinterval hN hh hu
  -- The per-sub-interval bound is in `(b - a)` form; for our grid,
  -- `b - a = (a + (j+1)·h) - (a + j·h) = h`.
  have hab : a + (j : ℕ) * h ≤ a + ((j : ℕ) + 1) * h := by
    have : (j : ℕ) * h ≤ ((j : ℕ) + 1) * h :=
      mul_le_mul_of_nonneg_right (by linarith) hh.le
    linarith
  have hba_eq : a + ((j : ℕ) + 1) * h - (a + (j : ℕ) * h) = h := by ring
  have hpt := abs_f_le_subinterval_taylor_sum hab (hf j) (hF j) hu_mem
  -- Rewrite `(b - a)` to `h` in the conclusion.
  rw [hba_eq] at hpt
  exact hpt.trans (hM_bound j)

#check @abs_f_le_subinterval_taylor_sum
#check @exists_grid_subinterval
#check @grid_to_supNorm_taylor

end WaveCAP

#print axioms WaveCAP.grid_to_supNorm_taylor
