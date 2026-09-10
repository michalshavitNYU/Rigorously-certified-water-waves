import WaveCAP.Common
import WaveCAP.TaylorSeriesCorrectness

/-!
# Area C — `tighten_sup_bounds!` correctness

Justifies Julia's `tighten_sup_bounds!` algorithm: starting from a possibly
loose top-level derivative bound `B_{M+1}`, the routine walks *down* from
`m = M` to `m = 0`, at each level combining a per-grid maximum of
`|f^{(m)}(u_j)|` with a Lipschitz step `B_{m+1} · h` to produce a tightened
sup-norm bound on `f^{(m)}`.

The mathematical core is the single-step inequality

  ∀ u ∈ [a, a + N·h],  |f^{(m)}(u)| ≤ max_j |f^{(m)}(u_j)| + B · h

where `B` is any bound on `|f^{(m+1)}|`. Iterating this from `m = M` downward
gives the recursion captured by `tightenSupBoundsRecursion` below.

**Plan reference:** `docs/lean_tighten_sup_bounds_plan.md`.
**Cross-reference Julia:** `BabenkoRigorous/src/TaylorSeries.jl:259–290`
(`tighten_sup_bounds!`).

Watch-points:
  1. The conservative spacing `h` (rather than the optimal `h/2`) mirrors
     Julia's implementation choice; both are valid upper bounds.
  2. The `min` in `TaylorSeriesData.tighten` mirrors Julia's "only adopt the
     new bound if strictly tighter than the existing entry" check — the `min`
     of two valid upper bounds is again a valid upper bound, so correctness
     is automatic regardless of which side wins.
-/

namespace WaveCAP

open Real intervalIntegral MeasureTheory Set

-- ===========================================================================
-- §1. Single-step grid + Lipschitz lemma
-- ===========================================================================

/-- Single-step grid+Lipschitz sup-norm bound.

If `f ∈ C^{m+1}` and `|f^{(m+1)}| ≤ B` everywhere, then for any grid spacing
`h > 0` the sup-norm of `|f^{(m)}|` over `[a, a + (N-1)·h + h]` is bounded by
the worst grid-point value plus `B · h`.

This is the workhorse called inductively in `tighten_sup_bounds_correct`.

Julia: `BabenkoRigorous/src/TaylorSeries.jl:259` (`tighten_sup_bounds!`,
inner loop). -/
theorem grid_lipschitz_sup_bound_step
    {f : ℝ → ℝ} {m : ℕ}
    (hf : ContDiff ℝ (m + 1) f)
    {h B : ℝ} (hh : 0 < h) (hB : 0 ≤ B)
    (hBound : ∀ u, |iteratedDeriv (m + 1) f u| ≤ B)
    {a : ℝ} {N : ℕ} (hN : 0 < N)
    (grid_max : ℝ)
    (hgrid : ∀ j : Fin N, |iteratedDeriv m f (a + j.val * h)| ≤ grid_max) :
    ∀ u ∈ Set.Icc a (a + (N - 1) * h + h),
      |iteratedDeriv m f u| ≤ grid_max + B * h := by
  -- Abbreviations.
  set g : ℝ → ℝ := iteratedDeriv m f with hg_def
  have hg_diff : Differentiable ℝ g :=
    ContDiff.differentiable_iteratedDeriv' m hf
  have hg_deriv_eq : ∀ x, deriv g x = iteratedDeriv (m + 1) f x := by
    intro x
    -- `iteratedDeriv (m+1) f = deriv (iteratedDeriv m f)`
    have heq : iteratedDeriv (m + 1) f = deriv (iteratedDeriv m f) :=
      iteratedDeriv_succ (f := f) (n := m)
    have hx : iteratedDeriv (m + 1) f x = deriv (iteratedDeriv m f) x :=
      congrArg (fun φ => φ x) heq
    simpa [g] using hx.symm
  -- The bound on `|deriv g|`.
  have hderiv_bound : ∀ x, ‖deriv g x‖ ≤ B := by
    intro x
    have h1 := hBound x
    rw [Real.norm_eq_abs, hg_deriv_eq x]
    exact h1
  -- Convex MVT instance (the whole real line).
  have hconv : Convex ℝ (Set.univ : Set ℝ) := convex_univ
  -- Per-pair Lipschitz inequality.
  have hLip : ∀ x y : ℝ, |g y - g x| ≤ B * |y - x| := by
    intro x y
    have h := Convex.norm_image_sub_le_of_norm_deriv_le
      (f := g) (s := (Set.univ : Set ℝ))
      (fun z _ => hg_diff z) (fun z _ => hderiv_bound z) hconv
      (Set.mem_univ x) (Set.mem_univ y)
    simpa [Real.norm_eq_abs] using h
  -- Now pick `u` in the covered interval and locate the grid index.
  intro u hu
  -- `Icc a (a + (N-1)*h + h)` simplifies to `Icc a (a + N*h)`.
  have hu_upper : u ≤ a + N * h := by
    have hu2 := hu.2
    have heq : a + ((N : ℝ) - 1) * h + h = a + (N : ℝ) * h := by ring
    linarith
  have hu_lower : a ≤ u := hu.1
  -- Define j = min ⌊(u - a) / h⌋ (N - 1). Then 0 ≤ j ≤ N-1 (as a Nat).
  set t : ℝ := (u - a) / h with ht_def
  have ht_nonneg : 0 ≤ t := by
    apply div_nonneg
    · linarith
    · linarith
  -- Floor of t as a natural number (since t ≥ 0).
  set jN : ℕ := min ⌊t⌋.toNat (N - 1) with hjN_def
  have hjN_lt : jN < N := by
    refine lt_of_le_of_lt (min_le_right _ _) ?_
    omega
  let j : Fin N := ⟨jN, hjN_lt⟩
  -- Goal: |g u| ≤ grid_max + B * h.
  -- Use: |g u| ≤ |g (a + j*h)| + |g u - g (a + j*h)| ≤ grid_max + B * h.
  have hgrid_j : |g (a + j.val * h)| ≤ grid_max := hgrid j
  have hLip_uj : |g u - g (a + j.val * h)| ≤ B * h := by
    have hLip0 := hLip (a + j.val * h) u
    -- Need |u - (a + j.val * h)| ≤ h.
    have hdiff : |u - (a + j.val * h)| ≤ h := by
      -- Case split on whether we used the floor or the cap N-1.
      by_cases hfloor : ⌊t⌋.toNat ≤ N - 1
      · -- jN = ⌊t⌋.toNat. By floor property: ⌊t⌋ ≤ t < ⌊t⌋ + 1.
        have hjN_eq : jN = ⌊t⌋.toNat := min_eq_left hfloor
        have hjN_real : (jN : ℝ) = (⌊t⌋ : ℝ) := by
          have h0 : (⌊t⌋.toNat : ℤ) = ⌊t⌋ :=
            Int.toNat_of_nonneg (Int.floor_nonneg.mpr ht_nonneg)
          have h1 : ((⌊t⌋.toNat : ℕ) : ℝ) = (⌊t⌋ : ℝ) := by exact_mod_cast h0
          calc (jN : ℝ) = ((⌊t⌋.toNat : ℕ) : ℝ) := by rw [hjN_eq]
            _ = (⌊t⌋ : ℝ) := h1
        have hfl_le_t : (jN : ℝ) ≤ t := by
          rw [hjN_real]; exact Int.floor_le t
        have ht_lt_succ : t < (jN : ℝ) + 1 := by
          rw [hjN_real]; exact Int.lt_floor_add_one t
        -- t = (u - a)/h, so u - a = t*h and (jN*h) ≤ u - a < (jN+1)*h.
        have hu_minus_a : u - a = t * h := by
          rw [ht_def]; field_simp
        have h_lower : (jN : ℝ) * h ≤ u - a := by
          have := mul_le_mul_of_nonneg_right hfl_le_t (le_of_lt hh)
          linarith [hu_minus_a]
        have h_upper : u - a < ((jN : ℝ) + 1) * h := by
          have := mul_lt_mul_of_pos_right ht_lt_succ hh
          linarith [hu_minus_a]
        have h_upper' : u - a - (jN : ℝ) * h < h := by linarith
        have h_lower' : 0 ≤ u - (a + (jN : ℝ) * h) := by linarith
        rw [show (j.val : ℝ) = (jN : ℝ) from rfl]
        rw [abs_of_nonneg (by linarith : (0 : ℝ) ≤ u - (a + (jN : ℝ) * h))]
        linarith
      · -- jN = N - 1 (we capped). Floor is ≥ N, so t ≥ N, so u ≥ a + N*h.
        push_neg at hfloor
        have hjN_eq : jN = N - 1 := min_eq_right (by omega)
        -- ⌊t⌋.toNat ≥ N, so ⌊t⌋ ≥ N, so t ≥ N.
        have hN_le_t : (N : ℝ) ≤ t := by
          have hfl : (N : ℤ) ≤ ⌊t⌋ := by
            have hfl_nonneg : (0 : ℤ) ≤ ⌊t⌋ := Int.floor_nonneg.mpr ht_nonneg
            have hN_le : (N : ℕ) ≤ ⌊t⌋.toNat := by omega
            have hcast : ((⌊t⌋.toNat : ℕ) : ℤ) = ⌊t⌋ := Int.toNat_of_nonneg hfl_nonneg
            calc (N : ℤ) ≤ (⌊t⌋.toNat : ℤ) := by exact_mod_cast hN_le
              _ = ⌊t⌋ := hcast
          have : (N : ℝ) ≤ (⌊t⌋ : ℝ) := by exact_mod_cast hfl
          exact this.trans (Int.floor_le t)
        -- So u - a ≥ N * h.
        have hu_minus_a : u - a = t * h := by rw [ht_def]; field_simp
        have hu_ge : u - a ≥ N * h := by
          have := mul_le_mul_of_nonneg_right hN_le_t (le_of_lt hh)
          linarith [hu_minus_a]
        -- But also u ≤ a + N * h (from hu_upper).
        have hu_eq_aN : u = a + N * h := by linarith
        -- And j.val = N - 1, so a + j.val*h = a + (N-1)*h.
        have hj_val : (j.val : ℝ) = (N : ℝ) - 1 := by
          show (jN : ℝ) = (N : ℝ) - 1
          rw [hjN_eq]
          have hN1 : 1 ≤ N := hN
          have hNcast : ((N - 1 : ℕ) : ℝ) = (N : ℝ) - 1 := by
            have : ((N - 1 : ℕ) : ℤ) = (N : ℤ) - 1 := by omega
            exact_mod_cast this
          exact hNcast
        have : u - (a + j.val * h) = h := by
          rw [hu_eq_aN, hj_val]; ring
        rw [this, abs_of_pos hh]
    calc |g u - g (a + j.val * h)| ≤ B * |u - (a + j.val * h)| := hLip0
      _ ≤ B * h := by
        apply mul_le_mul_of_nonneg_left hdiff hB
  -- Triangle: |g u| ≤ |g (a + j.val * h)| + |g u - g (a + j.val * h)|.
  have htri : |g u| ≤ |g (a + j.val * h)| + |g u - g (a + j.val * h)| := by
    have := abs_sub_abs_le_abs_sub (g u) (g (a + j.val * h))
    have h2 : |g u| - |g (a + j.val * h)| ≤ |g u - g (a + j.val * h)| := this
    linarith
  linarith

-- ===========================================================================
-- §2. Recursion: walk down from m = M to m = 0
-- ===========================================================================

/-- The tightened-bound recursion.

`tightenSupBoundsRecursion top_bound M h grid_max ⟨m, _⟩` gives the bound on
`|f^{(m)}|` produced by walking down from `m = M+1` (where the value is just
`top_bound`) to `m = 0`. At each intermediate `m ≤ M` the value is
`grid_max[m] + (next level) * h`.

Julia: `BabenkoRigorous/src/TaylorSeries.jl:259` (loop body). -/
def tightenSupBoundsRecursion
    (top_bound : ℝ) (M : ℕ) (h : ℝ)
    (grid_max : Fin (M + 1) → ℝ) :
    Fin (M + 2) → ℝ
  | ⟨k, hk⟩ =>
      if hkM : k = M + 1 then top_bound
      else
        have hk_le : k ≤ M := by omega
        grid_max ⟨k, by omega⟩ +
          tightenSupBoundsRecursion top_bound M h grid_max
            ⟨k + 1, by omega⟩ * h
  termination_by m => M + 2 - m.val

-- (Doc-comment for `tighten_sup_bounds_correct` follows the helper below.)

/-- Local (Icc-restricted) variant of `grid_lipschitz_sup_bound_step`. Uses
the same Convex-MVT pattern but on the convex set `Icc a (a + (N-1)*h + h)`
so the bound hypothesis `hBound` is only needed on that interval.

Used internally by `tighten_sup_bounds_correct`, where the recursion's IH
only provides an Icc-restricted bound on `|f^{(m+1)}|`. -/
private theorem grid_lipschitz_sup_bound_step_local
    {f : ℝ → ℝ} {m : ℕ}
    (hf : ContDiff ℝ (m + 1) f)
    {h B : ℝ} (hh : 0 < h) (hB : 0 ≤ B)
    {a : ℝ} {N : ℕ} (hN : 0 < N)
    (hBound : ∀ u ∈ Set.Icc a (a + (N - 1) * h + h),
      |iteratedDeriv (m + 1) f u| ≤ B)
    (grid_max : ℝ)
    (hgrid : ∀ j : Fin N, |iteratedDeriv m f (a + j.val * h)| ≤ grid_max) :
    ∀ u ∈ Set.Icc a (a + (N - 1) * h + h),
      |iteratedDeriv m f u| ≤ grid_max + B * h := by
  -- Abbreviations.
  set g : ℝ → ℝ := iteratedDeriv m f with hg_def
  set I : Set ℝ := Set.Icc a (a + ((N : ℝ) - 1) * h + h) with hI_def
  have hg_diff : Differentiable ℝ g :=
    ContDiff.differentiable_iteratedDeriv' m hf
  have hg_deriv_eq : ∀ x, deriv g x = iteratedDeriv (m + 1) f x := by
    intro x
    have heq : iteratedDeriv (m + 1) f = deriv (iteratedDeriv m f) :=
      iteratedDeriv_succ (f := f) (n := m)
    have hx : iteratedDeriv (m + 1) f x = deriv (iteratedDeriv m f) x :=
      congrArg (fun φ => φ x) heq
    simpa [g] using hx.symm
  -- The bound on `|deriv g|` over `I`.
  have hderiv_bound : ∀ x ∈ I, ‖deriv g x‖ ≤ B := by
    intro x hx
    rw [Real.norm_eq_abs, hg_deriv_eq x]
    exact hBound x hx
  -- Convex MVT on the Icc.
  have hconv : Convex ℝ I := convex_Icc _ _
  -- Per-pair Lipschitz inequality on `I`.
  have hLip : ∀ x ∈ I, ∀ y ∈ I, |g y - g x| ≤ B * |y - x| := by
    intro x hx y hy
    have h := Convex.norm_image_sub_le_of_norm_deriv_le
      (f := g) (s := I)
      (fun z _ => hg_diff z) hderiv_bound hconv hx hy
    simpa [Real.norm_eq_abs] using h
  -- Now pick `u ∈ I` and locate the grid index.
  intro u hu
  have hu_upper : u ≤ a + N * h := by
    have hu2 := hu.2
    have heq : a + ((N : ℝ) - 1) * h + h = a + (N : ℝ) * h := by ring
    linarith
  have hu_lower : a ≤ u := hu.1
  set t : ℝ := (u - a) / h with ht_def
  have ht_nonneg : 0 ≤ t := by
    apply div_nonneg
    · linarith
    · linarith
  set jN : ℕ := min ⌊t⌋.toNat (N - 1) with hjN_def
  have hjN_lt : jN < N := by
    refine lt_of_le_of_lt (min_le_right _ _) ?_
    omega
  let j : Fin N := ⟨jN, hjN_lt⟩
  have hgrid_j : |g (a + j.val * h)| ≤ grid_max := hgrid j
  -- The grid point also lies in `I`.
  have hj_in_I : (a + (j.val : ℝ) * h) ∈ I := by
    refine ⟨?_, ?_⟩
    · have : 0 ≤ (j.val : ℝ) * h := mul_nonneg (by exact_mod_cast Nat.zero_le _) (le_of_lt hh)
      linarith
    · -- a + j.val * h ≤ a + (N-1) * h + h since j.val ≤ N-1.
      have hj_le : j.val ≤ N - 1 := by
        show jN ≤ N - 1
        exact min_le_right _ _
      have hj_le_real : (j.val : ℝ) ≤ (N : ℝ) - 1 := by
        have : ((N - 1 : ℕ) : ℤ) = (N : ℤ) - 1 := by omega
        have h1 : ((N - 1 : ℕ) : ℝ) = (N : ℝ) - 1 := by exact_mod_cast this
        have h2 : (j.val : ℝ) ≤ ((N - 1 : ℕ) : ℝ) := by exact_mod_cast hj_le
        linarith
      have hmul : (j.val : ℝ) * h ≤ ((N : ℝ) - 1) * h := by
        apply mul_le_mul_of_nonneg_right hj_le_real (le_of_lt hh)
      linarith
  -- |u - (a + j.val * h)| ≤ h, exactly as in the universal version.
  have hdiff : |u - (a + (j.val : ℝ) * h)| ≤ h := by
    by_cases hfloor : ⌊t⌋.toNat ≤ N - 1
    · have hjN_eq : jN = ⌊t⌋.toNat := min_eq_left hfloor
      have hjN_real : (jN : ℝ) = (⌊t⌋ : ℝ) := by
        have h0 : (⌊t⌋.toNat : ℤ) = ⌊t⌋ :=
          Int.toNat_of_nonneg (Int.floor_nonneg.mpr ht_nonneg)
        have h1 : ((⌊t⌋.toNat : ℕ) : ℝ) = (⌊t⌋ : ℝ) := by exact_mod_cast h0
        calc (jN : ℝ) = ((⌊t⌋.toNat : ℕ) : ℝ) := by rw [hjN_eq]
          _ = (⌊t⌋ : ℝ) := h1
      have hfl_le_t : (jN : ℝ) ≤ t := by
        rw [hjN_real]; exact Int.floor_le t
      have ht_lt_succ : t < (jN : ℝ) + 1 := by
        rw [hjN_real]; exact Int.lt_floor_add_one t
      have hu_minus_a : u - a = t * h := by rw [ht_def]; field_simp
      have h_lower : (jN : ℝ) * h ≤ u - a := by
        have := mul_le_mul_of_nonneg_right hfl_le_t (le_of_lt hh)
        linarith [hu_minus_a]
      have h_upper : u - a < ((jN : ℝ) + 1) * h := by
        have := mul_lt_mul_of_pos_right ht_lt_succ hh
        linarith [hu_minus_a]
      rw [show (j.val : ℝ) = (jN : ℝ) from rfl]
      rw [abs_of_nonneg (by linarith : (0 : ℝ) ≤ u - (a + (jN : ℝ) * h))]
      linarith
    · push_neg at hfloor
      have hjN_eq : jN = N - 1 := min_eq_right (by omega)
      have hN_le_t : (N : ℝ) ≤ t := by
        have hfl : (N : ℤ) ≤ ⌊t⌋ := by
          have hfl_nonneg : (0 : ℤ) ≤ ⌊t⌋ := Int.floor_nonneg.mpr ht_nonneg
          have hN_le : (N : ℕ) ≤ ⌊t⌋.toNat := by omega
          have hcast : ((⌊t⌋.toNat : ℕ) : ℤ) = ⌊t⌋ := Int.toNat_of_nonneg hfl_nonneg
          calc (N : ℤ) ≤ (⌊t⌋.toNat : ℤ) := by exact_mod_cast hN_le
            _ = ⌊t⌋ := hcast
        have : (N : ℝ) ≤ (⌊t⌋ : ℝ) := by exact_mod_cast hfl
        exact this.trans (Int.floor_le t)
      have hu_minus_a : u - a = t * h := by rw [ht_def]; field_simp
      have hu_ge : u - a ≥ N * h := by
        have := mul_le_mul_of_nonneg_right hN_le_t (le_of_lt hh)
        linarith [hu_minus_a]
      have hu_eq_aN : u = a + N * h := by linarith
      have hj_val : (j.val : ℝ) = (N : ℝ) - 1 := by
        show (jN : ℝ) = (N : ℝ) - 1
        rw [hjN_eq]
        have hN1 : 1 ≤ N := hN
        have hNcast : ((N - 1 : ℕ) : ℝ) = (N : ℝ) - 1 := by
          have : ((N - 1 : ℕ) : ℤ) = (N : ℤ) - 1 := by omega
          exact_mod_cast this
        exact hNcast
      have : u - (a + j.val * h) = h := by
        rw [hu_eq_aN, hj_val]; ring
      rw [this, abs_of_pos hh]
  -- Combine.
  have hLip_uj : |g u - g (a + j.val * h)| ≤ B * h := by
    have hLip0 := hLip (a + j.val * h) hj_in_I u hu
    calc |g u - g (a + j.val * h)| ≤ B * |u - (a + j.val * h)| := hLip0
      _ ≤ B * h := by apply mul_le_mul_of_nonneg_left hdiff hB
  have htri : |g u| ≤ |g (a + j.val * h)| + |g u - g (a + j.val * h)| := by
    have := abs_sub_abs_le_abs_sub (g u) (g (a + j.val * h))
    linarith
  linarith

/-- Auxiliary lemma: `tightenSupBoundsRecursion` is nonneg at every level
when `top_bound`, all `grid_max`, and `h` are nonneg. -/
private lemma tightenSupBoundsRecursion_nonneg
    {top_bound : ℝ} (h_top_nonneg : 0 ≤ top_bound)
    (M : ℕ) {h : ℝ} (hh_nonneg : 0 ≤ h)
    {grid_max : Fin (M + 1) → ℝ}
    (h_grid_nonneg : ∀ m : Fin (M + 1), 0 ≤ grid_max m)
    (i : ℕ) (hi : i < M + 2) :
    0 ≤ tightenSupBoundsRecursion top_bound M h grid_max ⟨i, hi⟩ := by
  -- Generalize: prove `∀ i, M+1-i = d → ...` by induction on `d`.
  suffices H : ∀ d, ∀ (i' : ℕ) (hi' : i' < M + 2), M + 1 - i' = d →
      0 ≤ tightenSupBoundsRecursion top_bound M h grid_max ⟨i', hi'⟩ from
    H (M + 1 - i) i hi rfl
  intro d
  induction d with
  | zero =>
      intro i' hi' hd
      have hi_eq : i' = M + 1 := by omega
      subst hi_eq
      have hrec_eq : tightenSupBoundsRecursion top_bound M h grid_max
                       ⟨M + 1, hi'⟩ = top_bound := by
        unfold tightenSupBoundsRecursion
        simp
      rw [hrec_eq]
      exact h_top_nonneg
  | succ d ih =>
      intro i' hi' hd
      have hi_lt : i' < M + 1 := by omega
      have hrec_eq : tightenSupBoundsRecursion top_bound M h grid_max ⟨i', hi'⟩
                     = grid_max ⟨i', hi_lt⟩ +
                       tightenSupBoundsRecursion top_bound M h grid_max
                         ⟨i' + 1, by omega⟩ * h := by
        have hkne : i' ≠ M + 1 := by omega
        conv_lhs => rw [tightenSupBoundsRecursion]
        simp [hkne]
      rw [hrec_eq]
      have h1 : 0 ≤ grid_max ⟨i', hi_lt⟩ := h_grid_nonneg _
      have hd_eq : M + 1 - (i' + 1) = d := by omega
      have h2 : 0 ≤ tightenSupBoundsRecursion top_bound M h grid_max
                       ⟨i' + 1, by omega⟩ := ih (i' + 1) (by omega) hd_eq
      have h3 : 0 ≤ tightenSupBoundsRecursion top_bound M h grid_max
                       ⟨i' + 1, by omega⟩ * h :=
        mul_nonneg h2 hh_nonneg
      linarith

/-- Correctness of the downward recursion.

Given a (possibly loose) top-level bound `top_bound ≥ |f^{(M+1)}|` and a
per-level grid maximum `grid_max[m] ≥ max_j |f^{(m)}(u_j)|`, the recursion
output `tightenSupBoundsRecursion top_bound M h grid_max ⟨m, _⟩` is a valid
sup-norm bound on `|f^{(m)}|` for every `m ≤ M` and every `u` in the grid's
covered interval.

**Proof sketch.** Induct downward on `m`. Base `m = M`: apply
`grid_lipschitz_sup_bound_step` with `B = top_bound` (universal). Step
`m < M`: the IH gives an Icc-restricted bound on `|f^{(m+1)}|`, so apply
`grid_lipschitz_sup_bound_step_local` with that bound.

Julia: `BabenkoRigorous/src/TaylorSeries.jl:259` (whole routine). -/
theorem tighten_sup_bounds_correct
    {f : ℝ → ℝ} (M : ℕ) {h : ℝ} (hh : 0 < h)
    (hf : ContDiff ℝ (M + 1) f)
    {top_bound : ℝ} (h_top_nonneg : 0 ≤ top_bound)
    (h_top : ∀ u, |iteratedDeriv (M + 1) f u| ≤ top_bound)
    {a : ℝ} {N : ℕ} (hN : 0 < N)
    {grid_max : Fin (M + 1) → ℝ}
    (h_grid_nonneg : ∀ m : Fin (M + 1), 0 ≤ grid_max m)
    (h_grid : ∀ (m : Fin (M + 1)) (j : Fin N),
        |iteratedDeriv m.val f (a + j.val * h)| ≤ grid_max m) :
    ∀ (m : Fin (M + 1)) (u : ℝ),
        u ∈ Set.Icc a (a + (N - 1) * h + h) →
          |iteratedDeriv m.val f u| ≤
            tightenSupBoundsRecursion top_bound M h grid_max
              ⟨m.val, by omega⟩ := by
  -- Strategy: prove a stronger generic statement by downward induction on
  -- `k`. Specifically, for every `k ≤ M` and every `u` in the covered Icc,
  -- the recursion at level `k` bounds `|f^{(k)} u|`. We do this by induction
  -- on `j := M - k`.
  -- First, evaluate `tightenSupBoundsRecursion` at the two relevant patterns:
  -- (a) `⟨M+1, _⟩` → `top_bound`
  -- (b) `⟨k, _⟩` with `k ≤ M` → `grid_max ⟨k, _⟩ + recurse(k+1) * h`
  have recurse_top : tightenSupBoundsRecursion top_bound M h grid_max
                       ⟨M + 1, Nat.lt_succ_self _⟩ = top_bound := by
    unfold tightenSupBoundsRecursion
    simp
  have recurse_step : ∀ (k : ℕ) (hk : k < M + 1),
      tightenSupBoundsRecursion top_bound M h grid_max
        ⟨k, Nat.lt_succ_of_lt hk⟩ =
      grid_max ⟨k, hk⟩ +
        tightenSupBoundsRecursion top_bound M h grid_max
          ⟨k + 1, by omega⟩ * h := by
    intro k hk
    have hkne : k ≠ M + 1 := by omega
    conv_lhs => rw [tightenSupBoundsRecursion]
    simp [hkne]
  -- Generic statement keyed off the level `k` (rather than `M - j`).
  suffices H : ∀ (k : ℕ) (hk : k < M + 1),
      ∀ (u : ℝ), u ∈ Set.Icc a (a + ((N : ℝ) - 1) * h + h) →
        |iteratedDeriv k f u| ≤
          tightenSupBoundsRecursion top_bound M h grid_max
            ⟨k, by omega⟩ by
    intro m u hu
    have hmval_lt : m.val < M + 1 := m.isLt
    exact H m.val hmval_lt u hu
  -- Now prove H by downward induction on `j := M - k`.
  intro k hk_lt
  -- We strong-induct on `j := M - k`, generalizing `k`.
  suffices H' : ∀ (j : ℕ), ∀ (k' : ℕ) (hk' : k' < M + 1), M - k' = j →
      ∀ (u : ℝ), u ∈ Set.Icc a (a + ((N : ℝ) - 1) * h + h) →
        |iteratedDeriv k' f u| ≤
          tightenSupBoundsRecursion top_bound M h grid_max
            ⟨k', by omega⟩ from
    H' (M - k) k hk_lt rfl
  intro j
  induction j with
  | zero =>
      intro k' hk' hkj u hu
      -- M - k' = 0 means k' ≥ M; combined with k' < M + 1, gives k' = M.
      have hk'_eq : k' = M := by omega
      -- Avoid `rw [hk'_eq]` (dependent motive issue). Build the goal in
      -- terms of `M` directly.
      have hM_lt : M < M + 1 := Nat.lt_succ_self _
      have hgrid_M : ∀ jx : Fin N,
          |iteratedDeriv M f (a + jx.val * h)| ≤ grid_max ⟨M, hM_lt⟩ := by
        intro jx
        have := h_grid ⟨M, hM_lt⟩ jx
        simpa using this
      have hstep := grid_lipschitz_sup_bound_step
        (f := f) (m := M) hf hh h_top_nonneg h_top hN
        (grid_max ⟨M, hM_lt⟩) hgrid_M u hu
      -- Recurse at ⟨M, hM_lt⟩ = grid_max ⟨M, _⟩ + recurse ⟨M+1, _⟩ * h
      --                       = grid_max ⟨M, _⟩ + top_bound * h.
      have hres_eq : tightenSupBoundsRecursion top_bound M h grid_max
                       ⟨M, by omega⟩
                     = grid_max ⟨M, hM_lt⟩ + top_bound * h := by
        have hstep_eq := recurse_step M hM_lt
        have htop_at : tightenSupBoundsRecursion top_bound M h grid_max
                         ⟨M + 1, by omega⟩ = top_bound := recurse_top
        rw [hstep_eq, htop_at]
      -- Now massage the goal — replace k' with M in the Fin index.
      have hgoal :
          |iteratedDeriv M f u| ≤
            tightenSupBoundsRecursion top_bound M h grid_max ⟨M, by omega⟩ := by
        rw [hres_eq]; exact hstep
      -- Transport along hk'_eq.
      have : (⟨k', by omega⟩ : Fin (M + 2)) = ⟨M, by omega⟩ :=
        Fin.ext hk'_eq
      rw [this, hk'_eq]
      exact hgoal
  | succ d ih =>
      intro k' hk' hkj u hu
      -- M - k' = d + 1 means k' < M, so k' + 1 ≤ M, and M - (k' + 1) = d.
      have hk'_lt_M : k' < M := by omega
      have hk'_succ_lt : k' + 1 < M + 1 := by omega
      have hd_eq : M - (k' + 1) = d := by omega
      -- IH at k' + 1.
      have ih_app : ∀ (v : ℝ), v ∈ Set.Icc a (a + ((N : ℝ) - 1) * h + h) →
          |iteratedDeriv (k' + 1) f v| ≤
            tightenSupBoundsRecursion top_bound M h grid_max
              ⟨k' + 1, by omega⟩ := ih (k' + 1) hk'_succ_lt hd_eq
      -- Set B = recurse(k'+1).
      set B : ℝ := tightenSupBoundsRecursion top_bound M h grid_max
                     ⟨k' + 1, by omega⟩ with hB_def
      have hB_nonneg : 0 ≤ B :=
        tightenSupBoundsRecursion_nonneg h_top_nonneg M (le_of_lt hh)
          h_grid_nonneg (k' + 1) (by omega)
      have hcontdiff_k' : ContDiff ℝ (k' + 1) f := by
        apply hf.of_le
        have h' : (k' : ℕ) ≤ M := by omega
        gcongr <;> exact_mod_cast h'
      have hBound_Icc : ∀ v ∈ Set.Icc a (a + ((N : ℝ) - 1) * h + h),
          |iteratedDeriv (k' + 1) f v| ≤ B := ih_app
      have hgrid_k' : ∀ jx : Fin N,
          |iteratedDeriv k' f (a + jx.val * h)| ≤ grid_max ⟨k', hk'⟩ := by
        intro jx
        have := h_grid ⟨k', hk'⟩ jx
        simpa using this
      have hstep := grid_lipschitz_sup_bound_step_local
        (f := f) (m := k') hcontdiff_k' hh hB_nonneg hN hBound_Icc
        (grid_max ⟨k', hk'⟩) hgrid_k' u hu
      have hres_eq : tightenSupBoundsRecursion top_bound M h grid_max
                       ⟨k', by omega⟩
                     = grid_max ⟨k', hk'⟩ + B * h := by
        rw [recurse_step k' hk']
      rw [hres_eq]
      exact hstep

-- ===========================================================================
-- §3. Bridge to `TaylorSeriesData`
-- ===========================================================================

/-- Constructive `tighten` operation on a `TaylorSeriesData` record.

Recomputes `sup_bounds[m]` for `m = 0..M` using the grid+Lipschitz recursion,
taking `min` with the existing entry so we never weaken a bound that was
already tighter. The `M+1` slot (the input top bound) is preserved.

Mirrors the in-place mutation in
`BabenkoRigorous/src/TaylorSeries.jl:259` (`tighten_sup_bounds!`). -/
noncomputable def TaylorSeriesData.tighten {M N : ℕ}
    (ts : TaylorSeriesData M N) : TaylorSeriesData M N :=
  let new_sup_bounds : Fin (M + 2) → ℝ := fun m =>
    if hM1 : m.val = M + 1 then ts.sup_bounds m
    else
      have hm_le : m.val ≤ M := by omega
      -- For `N = 0` there is no grid to take the max over, so we fall back to
      -- the existing bound (which already covers the empty case trivially).
      if hN : 0 < N then
        have hNe : (Finset.univ : Finset (Fin N)).Nonempty :=
          ⟨⟨0, hN⟩, Finset.mem_univ _⟩
        let grid_max : ℝ :=
          Finset.univ.sup' hNe
            (fun j : Fin N =>
              |ts.derivs j ⟨m.val, by omega⟩| + ts.derivs_err j ⟨m.val, by omega⟩)
        min (ts.sup_bounds m)
          (grid_max +
            (tightenSupBoundsRecursion (ts.sup_bounds ⟨M + 1, by omega⟩) M ts.h
              (fun k => ts.sup_bounds ⟨k.val, by omega⟩)
              ⟨m.val + 1, by omega⟩) * ts.h)
      else
        ts.sup_bounds m
  ⟨ts.derivs, ts.derivs_err, new_sup_bounds, ts.h⟩

-- Local helper: `iteratedDeriv n f` is `T`-periodic when `f` is. Mirrors
-- `WaveCAP.Periodic.iteratedDeriv_periodic` from `HilbertCommutesDeriv.lean`,
-- duplicated here to avoid pulling in the heavyweight Hilbert dependency.
private lemma iteratedDeriv_periodic_of_periodic
    {f : ℝ → ℝ} {T : ℝ} (hper : Function.Periodic f T) (n : ℕ) :
    Function.Periodic (iteratedDeriv n f) T := by
  induction n with
  | zero => simpa [iteratedDeriv_zero] using hper
  | succ n ih =>
      rw [iteratedDeriv_succ]
      intro x
      have hcomp : (fun y : ℝ => iteratedDeriv n f (y + T)) = iteratedDeriv n f :=
        funext ih
      have h1 : deriv (fun y : ℝ => iteratedDeriv n f (y + T)) x
                = deriv (iteratedDeriv n f) (x + T) :=
        deriv_comp_add_const _ _ _
      have h2 : deriv (fun y : ℝ => iteratedDeriv n f (y + T)) x
                = deriv (iteratedDeriv n f) x := by rw [hcomp]
      linarith [h1, h2]

-- Local helper: extensional congruence for `tightenSupBoundsRecursion` in the
-- grid_max function — if two grid_max functions agree at every level `≥ k`,
-- their recursion values at level `k` agree. Proven by downward induction.
private lemma tightenSupBoundsRecursion_congr
    {top_bound : ℝ} (M : ℕ) {h : ℝ}
    {G1 G2 : Fin (M + 1) → ℝ}
    (k : ℕ) (hk : k < M + 2)
    (hG : ∀ (i : ℕ) (hi : i < M + 1), k ≤ i → G1 ⟨i, hi⟩ = G2 ⟨i, hi⟩) :
    tightenSupBoundsRecursion top_bound M h G1 ⟨k, hk⟩
      = tightenSupBoundsRecursion top_bound M h G2 ⟨k, hk⟩ := by
  -- Downward induction on `d := M + 1 - k`.
  suffices H : ∀ d, ∀ (k' : ℕ) (hk' : k' < M + 2), M + 1 - k' = d →
      (∀ (i : ℕ) (hi : i < M + 1), k' ≤ i → G1 ⟨i, hi⟩ = G2 ⟨i, hi⟩) →
      tightenSupBoundsRecursion top_bound M h G1 ⟨k', hk'⟩
        = tightenSupBoundsRecursion top_bound M h G2 ⟨k', hk'⟩ from
    H (M + 1 - k) k hk rfl hG
  intro d
  induction d with
  | zero =>
      intro k' hk' hd _hG
      have hk_eq : k' = M + 1 := by omega
      subst hk_eq
      unfold tightenSupBoundsRecursion
      simp
  | succ d ih =>
      intro k' hk' hd hG'
      have hk_lt : k' < M + 1 := by omega
      have hkne : k' ≠ M + 1 := by omega
      have hG_eq : G1 ⟨k', hk_lt⟩ = G2 ⟨k', hk_lt⟩ := hG' k' hk_lt le_rfl
      -- Unfold one step on both sides.
      have hLHS : tightenSupBoundsRecursion top_bound M h G1 ⟨k', hk'⟩
                  = G1 ⟨k', hk_lt⟩ +
                    tightenSupBoundsRecursion top_bound M h G1
                      ⟨k' + 1, by omega⟩ * h := by
        conv_lhs => rw [tightenSupBoundsRecursion]
        simp [hkne]
      have hRHS : tightenSupBoundsRecursion top_bound M h G2 ⟨k', hk'⟩
                  = G2 ⟨k', hk_lt⟩ +
                    tightenSupBoundsRecursion top_bound M h G2
                      ⟨k' + 1, by omega⟩ * h := by
        conv_lhs => rw [tightenSupBoundsRecursion]
        simp [hkne]
      rw [hLHS, hRHS, hG_eq]
      have hd_eq : M + 1 - (k' + 1) = d := by omega
      have hG_succ : ∀ (i : ℕ) (hi : i < M + 1), k' + 1 ≤ i →
          G1 ⟨i, hi⟩ = G2 ⟨i, hi⟩ := by
        intro i hi hle; exact hG' i hi (by omega)
      have := ih (k' + 1) (by omega) hd_eq hG_succ
      rw [this]

/-- Bridge: `tighten` preserves the `IsTaylorSeriesOf` predicate.

The `derivs`, `derivs_err`, and `h` fields are untouched, so their clauses
transfer directly. The `sup_bounds[M+1]` slot is preserved. For
`sup_bounds[m]` with `m ≤ M`, the new value is `min` of two valid upper
bounds (the old `sup_bounds[m]` and the freshly computed grid+Lipschitz
combination), so it is itself a valid upper bound by
`tighten_sup_bounds_correct`.

**Periodicity hypothesis.** The grid+Lipschitz bound only directly establishes
`|f^{(m)}(x)| ≤ grid_max + B·h` for `x ∈ Icc a (a + N·h)` (one grid period).
To extend this to a *global* sup-norm bound (as `sup_acc` demands), we assume
`f` is `(N·h)`-periodic. In the WaveCAP pipeline `h = 2π/N` and `f` is
`2π`-periodic, so `N·h = 2π` is the period — exactly the standard setting.

Julia: `BabenkoRigorous/src/TaylorSeries.jl:259` (`tighten_sup_bounds!`). -/
theorem tighten_sup_bounds_taylorseries_correct
    {M N : ℕ} {ts : TaylorSeriesData M N} {f : ℝ → ℝ} {a : ℝ}
    (h : IsTaylorSeriesOf ts f a)
    (hper : Function.Periodic f ((N : ℝ) * ts.h)) :
    IsTaylorSeriesOf (TaylorSeriesData.tighten ts) f a := by
  -- Notation.
  set tsT : TaylorSeriesData M N := TaylorSeriesData.tighten ts with htsT_def
  -- `tighten` preserves `derivs`, `derivs_err`, `h`; only `sup_bounds`
  -- changes. Verify these equalities once.
  have h_derivs_eq : tsT.derivs = ts.derivs := rfl
  have h_derivs_err_eq : tsT.derivs_err = ts.derivs_err := rfl
  have h_h_eq : tsT.h = ts.h := rfl
  refine ⟨?_, ?_, ?_, ?_⟩
  · -- contdiff: same as `ts`, since `f` is unchanged.
    exact h.contdiff
  · -- grid_acc: derivs, derivs_err, h are unchanged, so this is literally
    -- the original grid_acc.
    intro j m
    rw [h_h_eq]
    exact h.grid_acc j m
  · -- sup_acc: the core of the proof. Three cases:
    --   (i)   m.val = M + 1: new bound = old; trivial.
    --   (ii)  m.val ≤ M but N = 0: new bound = old; trivial.
    --   (iii) m.val ≤ M and 0 < N: new bound = min(old, grid+Lip);
    --                              old is valid by `h.sup_acc`; grid+Lip is
    --                              valid by `tighten_sup_bounds_correct` on
    --                              the Icc, extended to ℝ by periodicity.
    intro m x
    -- Unfold the new sup_bound at `m`.
    show |iteratedDeriv m.val f x| ≤ tsT.sup_bounds m
    by_cases hMm : m.val = M + 1
    · -- Case (i): the M+1 slot is preserved verbatim.
      have hsb_eq : tsT.sup_bounds m = ts.sup_bounds m := by
        show (if _ : m.val = M + 1 then ts.sup_bounds m else _) = ts.sup_bounds m
        simp [hMm]
      rw [hsb_eq]
      exact h.sup_acc m x
    · -- m.val ≤ M.
      have hm_le_M : m.val ≤ M := by have := m.isLt; omega
      by_cases hN : 0 < N
      · -- Case (iii): the main case.
        -- Spell out the new sup_bound value.
        have hNe : (Finset.univ : Finset (Fin N)).Nonempty :=
          ⟨⟨0, hN⟩, Finset.mem_univ _⟩
        set grid_max_m : ℝ :=
          Finset.univ.sup' hNe
            (fun j : Fin N =>
              |ts.derivs j ⟨m.val, by omega⟩|
                + ts.derivs_err j ⟨m.val, by omega⟩) with hgmm_def
        set recurse_m1 : ℝ :=
          tightenSupBoundsRecursion (ts.sup_bounds ⟨M + 1, by omega⟩) M ts.h
            (fun k => ts.sup_bounds ⟨k.val, by omega⟩)
            ⟨m.val + 1, by omega⟩ with hrec_def
        have hsb_eq : tsT.sup_bounds m = min (ts.sup_bounds m)
                        (grid_max_m + recurse_m1 * ts.h) := by
          show (if _ : m.val = M + 1 then _
                else if _ : 0 < N then min _ _ else _)
                = _
          simp [hMm, hN, hgmm_def, hrec_def]
        rw [hsb_eq]
        refine le_min ?_ ?_
        · -- |f^(m)(x)| ≤ ts.sup_bounds m: directly from h.sup_acc.
          exact h.sup_acc m x
        · -- |f^(m)(x)| ≤ grid_max_m + recurse_m1 * ts.h.
          -- Step 1: use periodicity to reduce x to the Icc.
          have h_per_m : Function.Periodic (iteratedDeriv m.val f)
                          ((N : ℝ) * ts.h) :=
            iteratedDeriv_periodic_of_periodic hper m.val
          have hNh_pos : 0 < (N : ℝ) * ts.h :=
            mul_pos (by exact_mod_cast hN) h.h_pos
          obtain ⟨y, hy_mem, hy_eq⟩ :=
            h_per_m.exists_mem_Ico hNh_pos x a
          -- y ∈ Ico a (a + N*ts.h) ⊆ Icc a (a + (N-1)*ts.h + ts.h).
          have hy_in_Icc :
              y ∈ Set.Icc a (a + ((N : ℝ) - 1) * ts.h + ts.h) := by
            refine ⟨hy_mem.1, ?_⟩
            have hy2 := hy_mem.2
            have heq : a + ((N : ℝ) - 1) * ts.h + ts.h = a + (N : ℝ) * ts.h := by
              ring
            linarith
          -- Step 2: apply `tighten_sup_bounds_correct` with a custom
          -- grid_max_func that equals grid_max_m at level m.val, and equals
          -- ts.sup_bounds at all other levels.
          set Gfun : Fin (M + 1) → ℝ := fun k =>
            if k.val = m.val then grid_max_m
            else ts.sup_bounds ⟨k.val, by omega⟩ with hGfun_def
          -- Hypotheses for `tighten_sup_bounds_correct`.
          have h_top_nonneg : 0 ≤ ts.sup_bounds ⟨M + 1, by omega⟩ := by
            have := h.sup_acc ⟨M + 1, by omega⟩ a
            exact le_trans (abs_nonneg _) this
          have h_top : ∀ u, |iteratedDeriv (M + 1) f u|
                          ≤ ts.sup_bounds ⟨M + 1, by omega⟩ :=
            fun u => h.sup_acc ⟨M + 1, by omega⟩ u
          have h_Gfun_nonneg : ∀ k : Fin (M + 1), 0 ≤ Gfun k := by
            intro k
            by_cases hk : k.val = m.val
            · -- Gfun k = grid_max_m; show grid_max_m ≥ 0.
              simp [hGfun_def, hk]
              -- grid_max_m = sup' of nonneg terms.
              have h0 : (⟨0, hN⟩ : Fin N) ∈ (Finset.univ : Finset (Fin N)) :=
                Finset.mem_univ _
              have : 0 ≤ |ts.derivs ⟨0, hN⟩ ⟨m.val, by omega⟩|
                        + ts.derivs_err ⟨0, hN⟩ ⟨m.val, by omega⟩ := by
                have habs : 0 ≤ |ts.derivs ⟨0, hN⟩ ⟨m.val, by omega⟩| := abs_nonneg _
                have herr : 0 ≤ ts.derivs_err ⟨0, hN⟩ ⟨m.val, by omega⟩ := by
                  have hga := h.grid_acc ⟨0, hN⟩ ⟨m.val, by omega⟩
                  exact le_trans (abs_nonneg _) hga
                linarith
              have hle : |ts.derivs ⟨0, hN⟩ ⟨m.val, by omega⟩|
                            + ts.derivs_err ⟨0, hN⟩ ⟨m.val, by omega⟩
                          ≤ Finset.univ.sup' hNe
                              (fun j : Fin N =>
                                |ts.derivs j ⟨m.val, by omega⟩|
                                  + ts.derivs_err j ⟨m.val, by omega⟩) :=
                Finset.le_sup' (f :=
                  fun j : Fin N =>
                    |ts.derivs j ⟨m.val, by omega⟩|
                      + ts.derivs_err j ⟨m.val, by omega⟩) h0
              linarith
            · -- Gfun k = ts.sup_bounds ⟨k.val, _⟩; nonneg from h.sup_acc.
              simp [hGfun_def, hk]
              have := h.sup_acc ⟨k.val, by omega⟩ a
              exact le_trans (abs_nonneg _) this
          have h_Gfun_grid : ∀ (k : Fin (M + 1)) (j : Fin N),
              |iteratedDeriv k.val f (a + j.val * ts.h)| ≤ Gfun k := by
            intro k j
            by_cases hk : k.val = m.val
            · -- Gfun k = grid_max_m. Use h.grid_acc and triangle.
              have hgrid := h.grid_acc j ⟨k.val, by omega⟩
              -- |f^(k)(u_j) - ts.derivs j k| ≤ ts.derivs_err j k
              -- so |f^(k)(u_j)| ≤ |ts.derivs j k| + ts.derivs_err j k.
              have hbound :
                  |iteratedDeriv k.val f (a + j.val * ts.h)|
                  ≤ |ts.derivs j ⟨k.val, by omega⟩|
                      + ts.derivs_err j ⟨k.val, by omega⟩ := by
                have := abs_sub_abs_le_abs_sub
                  (iteratedDeriv k.val f (a + j.val * ts.h))
                  (ts.derivs j ⟨k.val, by omega⟩)
                linarith
              -- Now the RHS is at most grid_max_m, using k.val = m.val.
              -- Substitute k.val = m.val in the indices.
              have hk_eq : (⟨k.val, by omega⟩ : Fin (M + 1)) = ⟨m.val, by omega⟩ :=
                Fin.ext hk
              rw [hk_eq] at hbound
              have hle : |ts.derivs j ⟨m.val, by omega⟩|
                            + ts.derivs_err j ⟨m.val, by omega⟩ ≤ grid_max_m := by
                exact Finset.le_sup' (f :=
                  fun jj : Fin N =>
                    |ts.derivs jj ⟨m.val, by omega⟩|
                      + ts.derivs_err jj ⟨m.val, by omega⟩) (Finset.mem_univ j)
              have : Gfun k = grid_max_m := by simp [hGfun_def, hk]
              rw [this]
              linarith
            · -- Gfun k = ts.sup_bounds ⟨k.val, _⟩. Use h.sup_acc.
              have : Gfun k = ts.sup_bounds ⟨k.val, by omega⟩ := by
                simp [hGfun_def, hk]
              rw [this]
              exact h.sup_acc ⟨k.val, by omega⟩ _
          -- Apply tighten_sup_bounds_correct.
          have hstep := tighten_sup_bounds_correct
            (f := f) M h.h_pos h.contdiff h_top_nonneg h_top hN
            h_Gfun_nonneg h_Gfun_grid ⟨m.val, by omega⟩ y hy_in_Icc
          -- hstep : |f^(m.val)(y)| ≤ tightenSupBoundsRecursion top_bound M ts.h Gfun ⟨m.val, _⟩.
          -- Use periodicity: |f^(m.val)(x)| = |f^(m.val)(y)|.
          have habs_eq :
              |iteratedDeriv m.val f x| = |iteratedDeriv m.val f y| := by
            rw [hy_eq]
          rw [habs_eq]
          -- Now reduce the recursion value to grid_max_m + recurse_m1 * ts.h.
          -- Step: tightenSupBoundsRecursion ... Gfun ⟨m.val, _⟩
          --        = Gfun ⟨m.val, _⟩
          --        + tightenSupBoundsRecursion ... Gfun ⟨m.val + 1, _⟩ * ts.h
          --        = grid_max_m
          --        + tightenSupBoundsRecursion ... (fun k => ts.sup_bounds ⟨k.val,_⟩)
          --              ⟨m.val + 1, _⟩ * ts.h
          --        = grid_max_m + recurse_m1 * ts.h.
          have hrec_step :
              tightenSupBoundsRecursion (ts.sup_bounds ⟨M + 1, by omega⟩) M ts.h
                Gfun ⟨m.val, by omega⟩
              = Gfun ⟨m.val, by omega⟩ +
                tightenSupBoundsRecursion (ts.sup_bounds ⟨M + 1, by omega⟩) M ts.h
                  Gfun ⟨m.val + 1, by omega⟩ * ts.h := by
            have hkne : m.val ≠ M + 1 := hMm
            conv_lhs => rw [tightenSupBoundsRecursion]
            simp [hkne]
          have hGfun_at_m : Gfun ⟨m.val, by omega⟩ = grid_max_m := by
            simp [hGfun_def]
          have hrec_congr :
              tightenSupBoundsRecursion (ts.sup_bounds ⟨M + 1, by omega⟩) M ts.h
                Gfun ⟨m.val + 1, by omega⟩
              = recurse_m1 := by
            rw [hrec_def]
            apply tightenSupBoundsRecursion_congr
            intro i hi hle
            -- At levels i ≥ m.val + 1, we have i ≠ m.val, so Gfun = ts.sup_bounds.
            have hi_ne : i ≠ m.val := by omega
            show Gfun ⟨i, hi⟩ = ts.sup_bounds ⟨i, by omega⟩
            simp [hGfun_def, hi_ne]
          rw [hrec_step, hGfun_at_m, hrec_congr] at hstep
          exact hstep
      · -- Case (ii): N = 0. New bound = old bound.
        have hsb_eq : tsT.sup_bounds m = ts.sup_bounds m := by
          show (if _ : m.val = M + 1 then _
                else if _ : 0 < N then _ else ts.sup_bounds m)
                = ts.sup_bounds m
          simp [hMm, hN]
        rw [hsb_eq]
        exact h.sup_acc m x
  · -- h_pos: ts.h is unchanged.
    rw [h_h_eq]
    exact h.h_pos

#check @grid_lipschitz_sup_bound_step
#check @tightenSupBoundsRecursion
#check @tighten_sup_bounds_correct
#check @TaylorSeriesData.tighten
#check @tighten_sup_bounds_taylorseries_correct

end WaveCAP
