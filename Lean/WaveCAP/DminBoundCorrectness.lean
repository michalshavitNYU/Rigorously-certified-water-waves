import WaveCAP.Common
import WaveCAP.Definitions
import WaveCAP.TaylorSeriesCorrectness
import WaveCAP.SeriesCorrectness
import WaveCAP.DLowerBound
import WaveCAP.TightenSupBoundsLemma
import WaveCAP.PipelineSmoothness

/-!
# Area E — `dmin_bound` correctness

This file justifies Julia's `dmin_bound` routine (in
`BabenkoRigorous/src/TaylorSeries.jl:655–749`), which produces a certified
positive lower bound on `D(u) = √(y_u(u)² + (1 + H[y_u](u))²)` over the torus.

## Julia algorithm summary

Given the trig-polynomial profile `y`, build derivative chains for `y_u` and
`H[y_u]`. Sample on a uniform grid `u_j = 2π·(j-1)/M_grid`, compute the
pointwise `D(u_j)`, and take the *minimum* over the grid:

  `D_lb_min = min_j √(y_u(u_j)² + (1 + H[y_u](u_j))²)`.

A Lipschitz correction is then subtracted to extend the grid-pointwise minimum
to a uniform lower bound:

  `d_lower_init = D_lb_min − (‖y_u'‖∞ + ‖H[y_u]'‖∞) · h,    h = 2π/M_grid`.

The proof that `|D'(u)| ≤ ‖y_u'‖∞ + ‖H[y_u]'‖∞` uses
  `D' = (y_u·y_u' + (1+H[y_u])·H[y_u]') / D`
with `|y_u/D|, |(1+H[y_u])/D| ≤ 1` (Cauchy–Schwarz inside the √).

The Julia routine then optionally returns `max(d_lower_init, inf_bound(ts_D))`
where `inf_bound(ts_D)` is the Taylor-remainder bound from `ts_D`. The Lean
side takes the `max` of any two valid lower bounds (lemma `max_of_two_bounds_correct`).

## Lean status

* `D_lipschitz_bound` — pointwise `|D'(u)| ≤ L`. Fully proved from the
  trig-poly derivative-supnorm hypotheses (caller supplies `L_yu` and `L_Hyu`).
* `d_lower_init_correct` — the Lipschitz-from-grid bound. Fully proved by
  MVT-on-grid against the periodic extension, mirroring the structure of
  `TightenSupBoundsLemma.grid_lipschitz_sup_bound_step`.
* `max_of_two_bounds_correct` — `max` of two valid lower bounds is a valid
  lower bound. Trivial.
* `dmin_bound_correct` — headline statement combining the above.

Caller hypotheses (to be discharged by future Phase 5 composition):

* `h_yu_lip`, `h_Hyu_lip`: that the supplied constants `L_yu`, `L_Hyu` are
  valid pointwise upper bounds on `|y_u'|` and `|H[y_u]'|`. These come from
  `supnorm_bound` applied to the derivative chain on the Julia side.
* `h_grid_min`: that the input `D_lb_min` is a valid pointwise lower bound on
  `D` *at the grid points*. This is what Julia's `lbound(sqrt(A_j² + B_j²))`
  computes.

Cross-reference: `BabenkoRigorous/src/TaylorSeries.jl:655–749` (`dmin_bound`).
-/

namespace WaveCAP

open Real Set

-- ===========================================================================
-- §1. Lipschitz bound on D' from sup-bounds of y_u' and H[y_u]'
-- ===========================================================================

/-- Pointwise Lipschitz bound on `D'`: for any positive `D` (witnessed by a
strict positive lower bound), `|D'(u)| ≤ 2 · (∑_k k² · |a k|)` for every `u`.

The key algebra is the chain rule for `D = √(A² + B²)` with `A = y_u`,
`B = 1 + H[y_u]`:

  `D'(u) = (A(u)·A'(u) + B(u)·B'(u)) / D(u)`,

and `|A|, |B| ≤ √(A² + B²) = D`, so `|A/D|, |B/D| ≤ 1`. Hence

  `|D'(u)| ≤ |A'(u)| · |A/D| + |B'(u)| · |B/D| ≤ |A'(u)| + |B'(u)|`.

The derivative bounds `|y_u'(u)|, |H[y_u]'(u)| ≤ ∑_k k² · |a k|` follow from
`yu_deriv_bound`/`Hyu_deriv_bound` in `PipelineSmoothness`, hence the overall
factor of `2`. The chain-rule identity comes from `D_deriv_eq`, dischargeable
given the positivity hypothesis.

Cross-reference: `BabenkoRigorous/src/TaylorSeries.jl:724` (comment block
"|D'| ≤ |A'| + |B'| (Cauchy-Schwarz inside the sqrt(A²+B²))"). -/
theorem D_lipschitz_bound
    {N : ℕ} {a : Fin N → ℝ}
    {DLower : ℝ} (hDLower_pos : 0 < DLower)
    (hcert : ∀ u, DLower ≤ D a u) :
    ∀ u, |deriv (D a) u| ≤ 2 * (∑ k : Fin N, ((k : ℝ) ^ 2) * |a k|) := by
  intro u
  -- D u > 0 everywhere, so we may divide.
  have hDpos : 0 < D a u := lt_of_lt_of_le hDLower_pos (hcert u)
  have hDne : D a u ≠ 0 := ne_of_gt hDpos
  -- Identity: D² = A² + B² with A = yu u, B = 1 + Hyu u.
  -- So |A| ≤ D and |B| ≤ D pointwise.
  have h_sq : D a u ^ 2 = (yu a u) ^ 2 + (1 + Hyu a u) ^ 2 := D_squared_eq a u
  have h_A_sq_le : (yu a u) ^ 2 ≤ D a u ^ 2 := by
    have : (1 + Hyu a u) ^ 2 ≥ 0 := sq_nonneg _
    linarith
  have h_B_sq_le : (1 + Hyu a u) ^ 2 ≤ D a u ^ 2 := by
    have : (yu a u) ^ 2 ≥ 0 := sq_nonneg _
    linarith
  have h_absA_le_D : |yu a u| ≤ D a u := by
    have h1 : |yu a u| ^ 2 ≤ D a u ^ 2 := by rw [sq_abs]; exact h_A_sq_le
    have h2 : 0 ≤ |yu a u| := abs_nonneg _
    exact abs_le_of_sq_le_sq' h1 hDpos.le |>.2
  have h_absB_le_D : |1 + Hyu a u| ≤ D a u := by
    have h1 : |1 + Hyu a u| ^ 2 ≤ D a u ^ 2 := by rw [sq_abs]; exact h_B_sq_le
    exact abs_le_of_sq_le_sq' h1 hDpos.le |>.2
  -- Discharge the chain-rule identity for D'.
  have h_D_deriv_eq_u :
      deriv (D a) u =
        (yu a u * deriv (yu a) u + (1 + Hyu a u) * deriv (Hyu a) u) / D a u :=
    D_deriv_eq a u hDpos
  -- Discharge the sup-bounds on |y_u'| and |H[y_u]'|.
  have h_yu_deriv_u : |deriv (yu a) u| ≤ ∑ k : Fin N, ((k : ℝ) ^ 2) * |a k| :=
    yu_deriv_bound a u
  have h_Hyu_deriv_u : |deriv (Hyu a) u| ≤ ∑ k : Fin N, ((k : ℝ) ^ 2) * |a k| :=
    Hyu_deriv_bound a u
  -- Now use the derivative formula and bound termwise.
  rw [h_D_deriv_eq_u]
  rw [abs_div, abs_of_pos hDpos]
  -- |A·A' + B·B'| ≤ |A|·|A'| + |B|·|B'| ≤ D·|A'| + D·|B'|
  have h_num_bound :
      |yu a u * deriv (yu a) u + (1 + Hyu a u) * deriv (Hyu a) u|
        ≤ D a u * |deriv (yu a) u| + D a u * |deriv (Hyu a) u| := by
    calc |yu a u * deriv (yu a) u + (1 + Hyu a u) * deriv (Hyu a) u|
        ≤ |yu a u * deriv (yu a) u| + |(1 + Hyu a u) * deriv (Hyu a) u| :=
          abs_add_le _ _
      _ = |yu a u| * |deriv (yu a) u| + |1 + Hyu a u| * |deriv (Hyu a) u| := by
          rw [abs_mul, abs_mul]
      _ ≤ D a u * |deriv (yu a) u| + D a u * |deriv (Hyu a) u| := by
          gcongr
  -- Divide by D > 0; the bound becomes |A'| + |B'| ≤ 2 · (∑ k² |a k|).
  have hDiv : (D a u * |deriv (yu a) u| + D a u * |deriv (Hyu a) u|) / D a u
              = |deriv (yu a) u| + |deriv (Hyu a) u| := by
    field_simp
  calc |yu a u * deriv (yu a) u + (1 + Hyu a u) * deriv (Hyu a) u| / D a u
      ≤ (D a u * |deriv (yu a) u| + D a u * |deriv (Hyu a) u|) / D a u := by
        apply div_le_div_of_nonneg_right h_num_bound hDpos.le
    _ = |deriv (yu a) u| + |deriv (Hyu a) u| := hDiv
    _ ≤ 2 * (∑ k : Fin N, ((k : ℝ) ^ 2) * |a k|) := by
        linarith [h_yu_deriv_u, h_Hyu_deriv_u]

-- ===========================================================================
-- §2. d_lower_init from grid + Lipschitz
-- ===========================================================================

/-- **Lipschitz-from-grid lower bound on `D`.**

Given:
  * a grid `u_j = j · h` for `j = 0, …, M_grid - 1`, with `h = 2π/M_grid`,
  * a pointwise lower bound `d_grid_min ≤ D(u_j)` at every grid point,
  * a global Lipschitz constant `L` for `D'` (with `L ≥ 0`),
  * `D` is `(M_grid · h)`-periodic (so `2π`-periodic, the standard case),
  * `D` is differentiable everywhere (which holds globally given the positive
    lower bound),

the value `d_grid_min - L · h` is a valid pointwise lower bound on `D(u)` for
all `u ∈ ℝ`.

This is the `dmin_bound`-side specialization of the universal grid+Lipschitz
pattern proved in `TightenSupBoundsLemma.grid_lipschitz_sup_bound_step`, but
applied to `D` itself (rather than to a generic `f^{(m)}`) and stated as a
*lower* bound rather than a sup-norm bound.

Cross-reference: `BabenkoRigorous/src/TaylorSeries.jl:728–731`. -/
theorem d_lower_init_correct
    {N : ℕ} {a : Fin N → ℝ}
    {DLower : ℝ} (hDLower_pos : 0 < DLower)
    (hcert : ∀ u, DLower ≤ D a u)
    {M_grid : ℕ} (hM_pos : 0 < M_grid)
    {h : ℝ} (hh_pos : 0 < h)
    (hh_eq : h = 2 * Real.pi / (M_grid : ℝ))
    {L : ℝ} (hL_nonneg : 0 ≤ L)
    (hL_bound : ∀ u, |deriv (D a) u| ≤ L)
    {d_grid_min : ℝ}
    (h_grid_min : ∀ j : Fin M_grid, d_grid_min ≤ D a ((j.val : ℝ) * h)) :
    ∀ u, d_grid_min - L * h ≤ D a u := by
  intro u
  -- D is C^∞ globally given the positive lower bound.
  have hD_contdiff : ContDiff ℝ (1 : ℕ) (D a) := D_contdiff a hDLower_pos hcert
  have hD_diff : Differentiable ℝ (D a) := hD_contdiff.differentiable (by norm_num)
  -- D is (M_grid · h)-periodic: since h = 2π/M_grid, M_grid · h = 2π, and
  -- D is 2π-periodic.
  have hMh_eq : (M_grid : ℝ) * h = 2 * Real.pi := by
    rw [hh_eq]; field_simp
  have hMh_pos : 0 < (M_grid : ℝ) * h :=
    mul_pos (by exact_mod_cast hM_pos) hh_pos
  -- D is 2π-periodic (follows from yu and Hyu being 2π-periodic).
  have hD_2pi_periodic : Function.Periodic (D a) (2 * Real.pi) := by
    intro x
    unfold D
    rw [yu_periodic a x, Hyu_periodic a x]
  have hD_periodic : Function.Periodic (D a) ((M_grid : ℝ) * h) := by
    rw [hMh_eq]; exact hD_2pi_periodic
  -- Reduce u to the standard interval [0, M_grid·h) via periodicity.
  obtain ⟨v, hv_mem, hv_eq⟩ := hD_periodic.exists_mem_Ico₀ hMh_pos u
  -- v ∈ Ico 0 (M_grid · h); hv_eq : D a u = D a v, so rewrite the goal.
  rw [hv_eq]
  -- Locate the nearest grid index: jN = min ⌊v / h⌋ (M_grid - 1).
  set t : ℝ := v / h with ht_def
  have ht_nonneg : 0 ≤ t := div_nonneg hv_mem.1 hh_pos.le
  set jN : ℕ := min ⌊t⌋.toNat (M_grid - 1) with hjN_def
  have hjN_lt : jN < M_grid := by
    refine lt_of_le_of_lt (min_le_right _ _) ?_
    omega
  let j : Fin M_grid := ⟨jN, hjN_lt⟩
  -- Bound the distance: |v - j*h| ≤ h, by the standard floor-grid argument.
  have hdist : |v - (j.val : ℝ) * h| ≤ h := by
    by_cases hfloor : ⌊t⌋.toNat ≤ M_grid - 1
    · -- jN = ⌊t⌋.toNat. By floor property: jN ≤ t < jN + 1.
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
      -- v = t * h.
      have hv_eq_th : v = t * h := by rw [ht_def]; field_simp
      have h_lower : (jN : ℝ) * h ≤ v := by
        have := mul_le_mul_of_nonneg_right hfl_le_t hh_pos.le
        linarith [hv_eq_th]
      have h_upper : v < ((jN : ℝ) + 1) * h := by
        have := mul_lt_mul_of_pos_right ht_lt_succ hh_pos
        linarith [hv_eq_th]
      show |v - (jN : ℝ) * h| ≤ h
      rw [abs_of_nonneg (by linarith : (0 : ℝ) ≤ v - (jN : ℝ) * h)]
      linarith
    · -- jN = M_grid - 1. v ∈ Ico 0 (M_grid·h), so this case requires t ≥ M_grid
      -- which contradicts v < M_grid·h.
      push_neg at hfloor
      -- t < M_grid (since v < M_grid · h and h > 0).
      have ht_lt_M : t < (M_grid : ℝ) := by
        rw [ht_def]
        rw [div_lt_iff₀ hh_pos]
        have := hv_mem.2
        linarith
      -- But hfloor says ⌊t⌋.toNat ≥ M_grid, hence t ≥ M_grid, contradiction.
      exfalso
      have hM_le_floor : (M_grid : ℕ) ≤ ⌊t⌋.toNat := by omega
      have hfl_nonneg : (0 : ℤ) ≤ ⌊t⌋ := Int.floor_nonneg.mpr ht_nonneg
      have hcast : ((⌊t⌋.toNat : ℕ) : ℤ) = ⌊t⌋ := Int.toNat_of_nonneg hfl_nonneg
      have hM_le_floor_int : (M_grid : ℤ) ≤ ⌊t⌋ := by
        calc (M_grid : ℤ) ≤ (⌊t⌋.toNat : ℤ) := by exact_mod_cast hM_le_floor
          _ = ⌊t⌋ := hcast
      have hfl_le_t : ((⌊t⌋ : ℤ) : ℝ) ≤ t := Int.floor_le t
      have hM_le_t : (M_grid : ℝ) ≤ t := by
        have : ((M_grid : ℤ) : ℝ) ≤ ((⌊t⌋ : ℤ) : ℝ) := by exact_mod_cast hM_le_floor_int
        have hM_cast : ((M_grid : ℤ) : ℝ) = (M_grid : ℝ) := by push_cast; ring
        rw [hM_cast] at this
        linarith
      linarith
  -- MVT step: |D v - D (j*h)| ≤ L · |v - j*h| ≤ L · h.
  have hLip_vj : |D a v - D a ((j.val : ℝ) * h)| ≤ L * h := by
    -- Use Convex.norm_image_sub_le_of_norm_deriv_le on Set.univ.
    have hconv : Convex ℝ (Set.univ : Set ℝ) := convex_univ
    have hderiv_bound : ∀ x ∈ (Set.univ : Set ℝ), ‖deriv (D a) x‖ ≤ L := by
      intro x _; rw [Real.norm_eq_abs]; exact hL_bound x
    have hMVT := Convex.norm_image_sub_le_of_norm_deriv_le
      (f := D a) (s := (Set.univ : Set ℝ))
      (fun z _ => hD_diff z) hderiv_bound hconv
      (Set.mem_univ ((j.val : ℝ) * h)) (Set.mem_univ v)
    have habs := (Real.norm_eq_abs _ ▸ hMVT)
    -- habs : |D v - D (j*h)| ≤ L * |v - j*h|
    calc |D a v - D a ((j.val : ℝ) * h)|
        ≤ L * |v - (j.val : ℝ) * h| := habs
      _ ≤ L * h := by apply mul_le_mul_of_nonneg_left hdist hL_nonneg
  -- D v ≥ D (j*h) - L * h ≥ d_grid_min - L * h.
  have h_grid_at_j : d_grid_min ≤ D a ((j.val : ℝ) * h) := h_grid_min j
  -- |D v - D (j*h)| ≤ L * h ⇒ D (j*h) - L * h ≤ D v.
  have h_lower :
      D a ((j.val : ℝ) * h) - L * h ≤ D a v := by
    have h_abs : -(L * h) ≤ D a v - D a ((j.val : ℝ) * h) := by
      have := abs_le.mp hLip_vj
      linarith [this.1]
    linarith
  linarith

-- ===========================================================================
-- §3. max-of-two lower bounds
-- ===========================================================================

/-- The pointwise `max` of two valid lower bounds on `D` is again a valid
lower bound. Trivial but used by `dmin_bound_correct`.

Cross-reference: `BabenkoRigorous/src/TaylorSeries.jl:746` (the `lbound(ib)`
vs `lbound(d_lower_init)` `max` selection). -/
theorem max_of_two_bounds_correct
    {N : ℕ} {a : Fin N → ℝ}
    {b1 b2 : ℝ}
    (h1 : ∀ u, b1 ≤ D a u) (h2 : ∀ u, b2 ≤ D a u) :
    ∀ u, max b1 b2 ≤ D a u := by
  intro u
  exact max_le (h1 u) (h2 u)

-- ===========================================================================
-- §4. Headline: dmin_bound correctness
-- ===========================================================================

/-- **Headline correctness of `dmin_bound`.**

The Julia routine `dmin_bound` returns a pair `(D_lower, ts_D)` where
`D_lower` is the `max` of two candidate lower bounds:

  * `d_lower_init` — Lipschitz-from-grid, justified by `d_lower_init_correct`.
  * `inf_bound(ts_D)` — Taylor-remainder-based, justified by
    `sup_bound_taylorseries_correct` on the `sqrt_series` output (the caller
    composes these in Phase 5; we hoist its validity as `h_ib_valid`).

The theorem packages both into a single pointwise lower bound on `D`. It is
parametric in:

  * `d_grid_min`: a valid pointwise lower bound on `D` at the grid points
    (caller-side, from `lbound(sqrt(A_j² + B_j²))` in Julia). **Trust anchor.**
  * `D_alt`: an alternate lower bound (e.g. `inf_bound(ts_D)`) supplied with
    its validity proof. **Trust anchor.**
  * `hDLower_pos`/`hcert`: a strictly-positive abstract `DLower ≤ D` witness,
    needed to invoke `D_contdiff`/`D_lipschitz_bound`. In the Phase 6 pipeline
    this is dischargeable from `cert_D_ge`; here it is a hypothesis so we don't
    depend on the certificate axiom.

The Lipschitz constant for `D'` is now the concrete `2 · (∑ k² |a k|)` produced
by `D_lipschitz_bound` (which internally discharges the chain-rule identity
and the derivative sup-bounds via `PipelineSmoothness`).

Cross-reference: `BabenkoRigorous/src/TaylorSeries.jl:655–749` (`dmin_bound`). -/
theorem dmin_bound_correct
    {N : ℕ} {a : Fin N → ℝ}
    {DLower : ℝ} (hDLower_pos : 0 < DLower)
    (hcert : ∀ u, DLower ≤ D a u)
    {M_grid : ℕ} (hM_pos : 0 < M_grid)
    {h : ℝ} (hh_pos : 0 < h)
    (hh_eq : h = 2 * Real.pi / (M_grid : ℝ))
    {d_grid_min : ℝ}
    (h_grid_min : ∀ j : Fin M_grid, d_grid_min ≤ D a ((j.val : ℝ) * h))
    {D_alt : ℝ}
    (h_alt_valid : ∀ u, D_alt ≤ D a u) :
    ∀ u, max (d_grid_min - 2 * (∑ k : Fin N, ((k : ℝ) ^ 2) * |a k|) * h) D_alt
            ≤ D a u := by
  -- Build the Lipschitz constant for D' (concrete: 2 · (∑ k² |a k|)).
  have hL_nonneg : 0 ≤ 2 * (∑ k : Fin N, ((k : ℝ) ^ 2) * |a k|) := by
    apply mul_nonneg (by norm_num)
    apply Finset.sum_nonneg
    intro k _
    exact mul_nonneg (by positivity) (abs_nonneg _)
  have hD_lip : ∀ u, |deriv (D a) u| ≤ 2 * (∑ k : Fin N, ((k : ℝ) ^ 2) * |a k|) :=
    D_lipschitz_bound hDLower_pos hcert
  have h_lower_init :=
    d_lower_init_correct hDLower_pos hcert hM_pos hh_pos hh_eq hL_nonneg
      hD_lip h_grid_min
  -- Combine via max_of_two_bounds_correct.
  exact max_of_two_bounds_correct h_lower_init h_alt_valid

-- ===========================================================================
-- Sanity checks
-- ===========================================================================

#check @D_lipschitz_bound
#check @d_lower_init_correct
#check @max_of_two_bounds_correct
#check @dmin_bound_correct

end WaveCAP

#print axioms WaveCAP.D_lipschitz_bound
#print axioms WaveCAP.d_lower_init_correct
#print axioms WaveCAP.max_of_two_bounds_correct
#print axioms WaveCAP.dmin_bound_correct
