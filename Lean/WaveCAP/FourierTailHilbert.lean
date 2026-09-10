import WaveCAP.Definitions

/-!
# N3 — Fourier-tail `π²/3` bound for `‖H[g]‖_∞`

```
For g ∈ C²(𝕋) 2π-periodic with ‖g''‖_∞ ≤ F_2:
  ‖periodicHilbert g‖_∞ ≤ F_2 · π² / 3.
```

This is the analytic content the Julia routine `fft_hilbert_series` uses to
bound each `sup_bound` level of `H[G]` in terms of the next-next-derivative
sup_bound of `G`:

```
new_sup_bounds[m+1] = ts_G.sup_bounds[m+3] · π²/3
```

(see `BabenkoRigorous/src/FFTHilbert.jl` lines ~270–280).

## Proof structure

1. **Set up.** Cast `g` to `g_circ : ℝ → ℂ` (via `gCircOfReal`). Its
   Fourier coefficients are `c n = fourierCoeffOn (h : 0 < 2π) g_circ n : ℂ`.

2. **IBP coefficient bound.** For `n ≠ 0`, `‖c n‖ ≤ F₂ / n²`. Uses
   `fourierCoeffOn_of_hasDerivAt` twice on the C²-smooth periodic `g`; the
   boundary terms `f(2π) - f(0)` vanish by periodicity for both `g` and `g'`.

3. **Summability.** `Summable (fun n : ℤ => if n = 0 then 0 else ‖c n‖)` by
   comparison with `Real.summable_one_div_int_pow (p := 2)`.

4. **Fourier-series bound on `H[g]`.** The periodic Hilbert transform sends
   the Fourier mode `c n · exp(i n u)` to `(-i · sgn n) · c n · exp(i n u)`,
   a unit-modulus multiplier. Hence pointwise
   `|periodicHilbert g u| ≤ ∑'_{n ≠ 0} ‖c n‖`.

   This is now `periodicHilbert_le_tsum_fourierCoeff`, **proved as a
   theorem** from the concrete Fourier-series definition of
   `periodicHilbert` (see `WaveCAP/Definitions.lean`): apply
   `Complex.abs_re_le_norm` to get `|Re(∑')| ≤ ‖∑'‖`, then
   `norm_tsum_le_tsum_norm`, then `‖hilbertMult n‖ = 1` for `n ≠ 0`
   (and `= 0` for `n = 0`) plus `‖exp(inu)‖ = 1`.

5. **Basel sum.** `2 · F₂ · (∑ k≥1 1/k²) = 2 · F₂ · π²/6 = F₂ · π²/3` by
   `hasSum_zeta_two`.

## Trust-chain status

`fourier_tail_hilbert_bound` now depends only on `propext, Classical.choice,
Quot.sound` — no `WaveCAP.*`-specific axioms remain in the N3 chain. Both
the multiplier-modulus content and the IBP coefficient bound are discharged
from standard Mathlib infrastructure (`fourierCoeffOn_of_hasDerivAt`,
`norm_tsum_le_tsum_norm`, `hasSum_zeta_two`, `Summable.of_nat_of_neg`).
-/

namespace WaveCAP

open Real Complex MeasureTheory intervalIntegral
open scoped Real

-- ===========================================================================
-- Step 1: Fourier-coefficient setup
-- ===========================================================================

-- Helpers `gCircOfReal` and `zero_lt_two_pi` are defined in `Definitions.lean`.

-- ===========================================================================
-- Step 2: IBP coefficient bound `‖c_n‖ ≤ F₂ / n²` for `n ≠ 0`
-- ===========================================================================

/-- Helper: complex cast of a real-differentiable function is differentiable
with the same (cast) derivative. Wraps `HasDerivAt.ofReal_comp`. -/
lemma hasDerivAt_complex_cast {f : ℝ → ℝ} {f' x : ℝ}
    (hf : HasDerivAt f f' x) :
    HasDerivAt (fun y : ℝ => (f y : ℂ)) (f' : ℂ) x :=
  hf.ofReal_comp

/-- The derivative of a periodic function is periodic with the same period.

Used here for `g` and `deriv g`; the proof works for any `T`-periodic function. -/
lemma Periodic.deriv_periodic {g : ℝ → ℝ} {T : ℝ}
    (hg_per : Function.Periodic g T) :
    Function.Periodic (deriv g) T := by
  intro x
  -- The function `y ↦ g (y + T)` equals `g` pointwise by periodicity, so
  -- their derivatives agree.  Use `deriv_comp_add_const` to evaluate the LHS.
  have hcomp : (fun y : ℝ => g (y + T)) = g := funext hg_per
  have h1 : deriv (fun y : ℝ => g (y + T)) x = deriv g (x + T) :=
    deriv_comp_add_const _ _ _
  have h2 : deriv (fun y : ℝ => g (y + T)) x = deriv g x := by rw [hcomp]
  linarith [h1, h2]

/-- **Single-IBP identity for periodic C¹ functions.**

For `g ∈ C¹(ℝ)` 2π-periodic and `n ≠ 0`, integration by parts gives the
identity `c_n(g) = (1 / (i n)) · c_n(g')`, where `g' = deriv g`. The boundary
term `g(2π) − g(0) = 0` vanishes by periodicity. -/
lemma fourierCoeffOn_eq_inv_n_mul_fourierCoeffOn_deriv
    (g : ℝ → ℝ) (hg : ContDiff ℝ 1 g)
    (hg_per : Function.Periodic g (2 * Real.pi))
    {n : ℤ} (hn : n ≠ 0) :
    fourierCoeffOn zero_lt_two_pi (gCircOfReal g) n
      = (1 / ((n : ℂ) * Complex.I)) *
        fourierCoeffOn zero_lt_two_pi (gCircOfReal (deriv g)) n := by
  -- Differentiability and continuity of g and deriv g.
  have hg_diff : Differentiable ℝ g := hg.differentiable (by decide)
  have hg'_cont : Continuous (deriv g) := hg.continuous_deriv (by decide)
  -- Abbreviations.
  set gC : ℝ → ℂ := gCircOfReal g with hgC
  set gC' : ℝ → ℂ := gCircOfReal (deriv g) with hgC'
  -- HasDerivAt of complex-cast g.
  have hgC_deriv : ∀ x, HasDerivAt gC (gC' x) x := by
    intro x
    have := hg_diff.differentiableAt (x := x)
    exact hasDerivAt_complex_cast this.hasDerivAt
  have hgC'_cont : Continuous gC' := Complex.continuous_ofReal.comp hg'_cont
  -- Boundary-term vanishing for g.
  have hboundary_g : gC (2 * Real.pi) - gC 0 = 0 := by
    have hg_eq : g (2 * Real.pi) = g 0 := by simpa using hg_per 0
    change ((g (2 * Real.pi) : ℂ) - (g 0 : ℂ)) = 0
    rw [hg_eq, sub_self]
  -- IBP: c_n(gC) = 1/(-2πin) · (e · (gC 2π - gC 0) - (2π - 0) · c_n(gC')).
  have hIBP :=
    fourierCoeffOn_of_hasDerivAt zero_lt_two_pi hn
      (f := gC) (f' := gC') (fun x _ => hgC_deriv x)
      (hgC'_cont.intervalIntegrable _ _)
  rw [hboundary_g, mul_zero, zero_sub] at hIBP
  -- After boundary vanishes: c_n(gC) = 1/(-2πin) · (-(2π - 0) · c_n(gC')).
  -- Simplify to: c_n(gC) = (1/(i n)) · c_n(gC').
  have hpi : (Real.pi : ℂ) ≠ 0 := Complex.ofReal_ne_zero.mpr Real.pi_pos.ne'
  have hnC : (n : ℂ) ≠ 0 := by exact_mod_cast hn
  have hI : Complex.I ≠ 0 := Complex.I_ne_zero
  rw [hIBP]
  have h2pi_ne : (2 * Real.pi : ℂ) ≠ 0 := mul_ne_zero (by norm_num) hpi
  push_cast
  field_simp
  ring

/-- **IBP identity (flipped form).** For `g ∈ C¹(ℝ)` 2π-periodic and `n ≠ 0`:
`c_n(g') = (i n) · c_n(g)`. -/
lemma fourierCoeffOn_deriv_eq
    (g : ℝ → ℝ) (hg : ContDiff ℝ 1 g)
    (hg_per : Function.Periodic g (2 * Real.pi))
    {n : ℤ} (hn : n ≠ 0) :
    fourierCoeffOn zero_lt_two_pi (gCircOfReal (deriv g)) n
      = ((n : ℂ) * Complex.I) *
        fourierCoeffOn zero_lt_two_pi (gCircOfReal g) n := by
  have h := fourierCoeffOn_eq_inv_n_mul_fourierCoeffOn_deriv g hg hg_per hn
  have hnC : (n : ℂ) ≠ 0 := by exact_mod_cast hn
  have hI : Complex.I ≠ 0 := Complex.I_ne_zero
  have hnI : (n : ℂ) * Complex.I ≠ 0 := mul_ne_zero hnC hI
  -- From c_n(g) = (1/(iN)) · c_n(g'), multiply both sides by (iN).
  rw [h]
  field_simp

/-- **Single-IBP step for periodic C¹ functions.**

For `g ∈ C¹(ℝ)` 2π-periodic and `n ≠ 0`, one integration by parts gives
`c_n(g) = (1 / (i n)) · c_n(g')`, where `g' = deriv g`. The boundary term
`g(2π) − g(0) = 0` vanishes by periodicity.

Phrased as a norm inequality: `‖c_n(g)‖ ≤ (1/|n|) · ‖c_n(g')‖`. -/
lemma fourierCoeff_norm_le_inv_n_mul_fourierCoeff_deriv
    (g : ℝ → ℝ) (hg : ContDiff ℝ 1 g)
    (hg_per : Function.Periodic g (2 * Real.pi))
    {n : ℤ} (hn : n ≠ 0) :
    ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal g) n‖
      ≤ (1 / |(n : ℝ)|) *
        ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal (deriv g)) n‖ := by
  rw [fourierCoeffOn_eq_inv_n_mul_fourierCoeffOn_deriv g hg hg_per hn, norm_mul]
  have h_norm_coef : ‖(1 / ((n : ℂ) * Complex.I) : ℂ)‖ = 1 / |(n : ℝ)| := by
    rw [norm_div, norm_one, norm_mul, Complex.norm_I, mul_one,
        Complex.norm_intCast]
  rw [h_norm_coef]

/-- **IBP Fourier-coefficient bound.**

For `g ∈ C²` 2π-periodic with `|g''| ≤ F₂`, the `n`-th Fourier coefficient of
`g` (viewed on `(0, 2π]`) satisfies `‖c_n‖ ≤ F₂ / n²` for every `n ≠ 0`.

The proof is two integrations by parts via `fourierCoeffOn_of_hasDerivAt`:
the boundary term `f(2π) - f(0) = 0` for `g` (by `hg_per`) and for `g'`
(by `(deriv g)`'s periodicity, derived above). -/
theorem fourierCoeff_norm_le_F₂_div_sq (g : ℝ → ℝ)
    (hg : ContDiff ℝ 2 g) (hg_per : Function.Periodic g (2 * Real.pi))
    (F₂ : ℝ) (hF₂ : ∀ u, |iteratedDeriv 2 g u| ≤ F₂)
    {n : ℤ} (hn : n ≠ 0) :
    ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal g) n‖ ≤ F₂ / (n : ℝ) ^ 2 := by
  -- F₂ ≥ 0.
  have hF₂_nn : 0 ≤ F₂ := le_trans (abs_nonneg _) (hF₂ 0)
  -- Smoothness ⇒ differentiability of g and deriv g.
  have hg_diff : Differentiable ℝ g := hg.differentiable (by decide)
  have hg'_diff : Differentiable ℝ (deriv g) := hg.differentiable_deriv_two
  -- Continuity of g, deriv g, deriv (deriv g).
  have hg_cont : Continuous g := hg.continuous
  have hg'_cont : Continuous (deriv g) := hg.continuous_deriv (by decide)
  have hg''_cont : Continuous (iteratedDeriv 2 g) :=
    hg.continuous_iteratedDeriv 2 (by decide)
  -- Periodicity of deriv g.
  have hg'_per : Function.Periodic (deriv g) (2 * Real.pi) :=
    Periodic.deriv_periodic hg_per
  -- Abbreviations.
  set gC : ℝ → ℂ := gCircOfReal g with hgC
  set gC' : ℝ → ℂ := gCircOfReal (deriv g) with hgC'
  set gC'' : ℝ → ℂ := gCircOfReal (iteratedDeriv 2 g) with hgC''
  -- iteratedDeriv 2 g = deriv (deriv g)
  have h_iter2 : iteratedDeriv 2 g = deriv (deriv g) := by
    rw [iteratedDeriv_succ, iteratedDeriv_one]
  -- HasDerivAt of complex-cast functions.
  have hgC_deriv : ∀ x, HasDerivAt gC (gC' x) x := by
    intro x
    have := hg_diff.differentiableAt (x := x)
    exact hasDerivAt_complex_cast this.hasDerivAt
  have hgC'_deriv : ∀ x, HasDerivAt gC' (gC'' x) x := by
    intro x
    have hd := hg'_diff.differentiableAt (x := x)
    have hcast : HasDerivAt (fun y : ℝ => ((deriv g y : ℝ) : ℂ)) ((deriv (deriv g) x : ℝ) : ℂ) x :=
      hasDerivAt_complex_cast hd.hasDerivAt
    -- gC'' x = (iteratedDeriv 2 g x : ℂ) = (deriv (deriv g) x : ℂ)
    have hrw : gC'' x = ((deriv (deriv g) x : ℝ) : ℂ) := by
      change ((iteratedDeriv 2 g x : ℝ) : ℂ) = _
      rw [h_iter2]
    rw [hrw]
    exact hcast
  -- Continuity of gC, gC', gC''.
  have hgC_cont : Continuous gC := Complex.continuous_ofReal.comp hg_cont
  have hgC'_cont : Continuous gC' := Complex.continuous_ofReal.comp hg'_cont
  have hgC''_cont : Continuous gC'' := Complex.continuous_ofReal.comp hg''_cont
  -- Boundary-term vanishing.
  have hboundary_g : gC (2 * Real.pi) - gC 0 = 0 := by
    have hg_eq : g (2 * Real.pi) = g 0 := by simpa using hg_per 0
    change ((g (2 * Real.pi) : ℂ) - (g 0 : ℂ)) = 0
    rw [hg_eq, sub_self]
  have hboundary_g' : gC' (2 * Real.pi) - gC' 0 = 0 := by
    have hg'_eq : deriv g (2 * Real.pi) = deriv g 0 := by simpa using hg'_per 0
    have h : ((deriv g (2 * Real.pi) : ℝ) : ℂ) - ((deriv g (0 : ℝ) : ℝ) : ℂ) = 0 := by
      rw [hg'_eq, sub_self]
    exact h
  -- IBP step 1: c_n(gC) = 1/(-2πin) · (e · (gC 2π - gC 0) - (2π - 0) · c_n(gC')).
  have hIBP1 :=
    fourierCoeffOn_of_hasDerivAt zero_lt_two_pi hn
      (f := gC) (f' := gC') (fun x _ => hgC_deriv x)
      (hgC'_cont.intervalIntegrable _ _)
  -- IBP step 2: c_n(gC') = 1/(-2πin) · (e · (gC' 2π - gC' 0) - (2π - 0) · c_n(gC'')).
  have hIBP2 :=
    fourierCoeffOn_of_hasDerivAt zero_lt_two_pi hn
      (f := gC') (f' := gC'') (fun x _ => hgC'_deriv x)
      (hgC''_cont.intervalIntegrable _ _)
  -- Substitute the vanishing boundary terms.
  rw [hboundary_g, mul_zero, zero_sub] at hIBP1
  rw [hboundary_g', mul_zero, zero_sub] at hIBP2
  -- Constants: pi ≠ 0, n ≠ 0 (over ℂ), so the denominator is nonzero.
  have hpi : (Real.pi : ℂ) ≠ 0 := Complex.ofReal_ne_zero.mpr Real.pi_pos.ne'
  have hnC : (n : ℂ) ≠ 0 := by exact_mod_cast hn
  -- Substitute hIBP2 into hIBP1, expressing c_n(gC) in terms of c_n(gC'').
  rw [hIBP2] at hIBP1
  -- The integral bound: ‖∫_0^{2π} fourier(-n)(x) • gC''(x) dx‖ ≤ 2π · F₂.
  -- Combined with the (1/2π) prefactor, this gives ‖c_n(gC'')‖ ≤ F₂.
  have h_norm_gC'' : ‖fourierCoeffOn zero_lt_two_pi gC'' n‖ ≤ F₂ := by
    rw [fourierCoeffOn_eq_integral]
    -- ‖(1/(2π)) • ∫₀^{2π} fourier(-n)(x) • gC''(x) dx‖ ≤ (1/(2π)) · ((2π) · F₂) = F₂.
    have h2pi_pos : (0 : ℝ) < 2 * Real.pi := zero_lt_two_pi
    -- Bound on the integrand: norm of fourier is 1, norm of gC'' x is |iteratedDeriv 2 g x|.
    have hbound : ∀ x ∈ Set.uIoc (0 : ℝ) (2 * Real.pi),
        ‖fourier (-n) ((x : ℝ) : AddCircle (2 * Real.pi - 0)) • gC'' x‖ ≤ F₂ := by
      intro x _
      rw [norm_smul]
      have h_fnorm : ‖fourier (-n) ((x : ℝ) : AddCircle (2 * Real.pi - 0))‖ = 1 := by
        rw [fourier_apply]; exact Circle.norm_coe _
      rw [h_fnorm, one_mul]
      have h_normgC : ‖gC'' x‖ = |iteratedDeriv 2 g x| :=
        Complex.norm_real _
      rw [h_normgC]; exact hF₂ x
    have hint_bound :
        ‖∫ x in (0 : ℝ)..(2 * Real.pi),
              fourier (-n) ((x : ℝ) : AddCircle (2 * Real.pi - 0)) • gC'' x‖
          ≤ F₂ * |2 * Real.pi - 0| :=
      intervalIntegral.norm_integral_le_of_norm_le_const hbound
    have h_abs_2pi : |2 * Real.pi - 0| = 2 * Real.pi := by
      rw [sub_zero]; exact abs_of_pos h2pi_pos
    rw [h_abs_2pi] at hint_bound
    -- Now bound ‖(1/(2π-0)) • ∫ … ‖ ≤ F₂.
    have h_eq_inv : (1 / (2 * Real.pi - 0) : ℝ) = 1 / (2 * Real.pi) := by rw [sub_zero]
    have h2pi_inv_pos : (0:ℝ) < 1 / (2 * Real.pi) := by positivity
    rw [norm_smul, h_eq_inv, Real.norm_eq_abs, abs_of_pos h2pi_inv_pos]
    calc (1 / (2 * Real.pi))
            * ‖∫ x in (0 : ℝ)..(2 * Real.pi),
                  fourier (-n) ((x : ℝ) : AddCircle (2 * Real.pi - 0)) • gC'' x‖
          ≤ (1 / (2 * Real.pi)) * (F₂ * (2 * Real.pi)) :=
            mul_le_mul_of_nonneg_left hint_bound h2pi_inv_pos.le
      _ = F₂ := by field_simp
  -- Express c_n(gC) = (-1/n²) · c_n(gC'') using the substituted hIBP1.
  have h_eq : fourierCoeffOn zero_lt_two_pi gC n
        = (-1 / (n : ℂ)^2) * fourierCoeffOn zero_lt_two_pi gC'' n := by
    rw [hIBP1]
    have h2pi_ne : (2 * Real.pi : ℂ) ≠ 0 := mul_ne_zero (by norm_num) hpi
    have hI : Complex.I ≠ 0 := Complex.I_ne_zero
    have hI_sq : Complex.I ^ 2 = -1 := Complex.I_sq
    push_cast
    field_simp
    linear_combination
      (4 * (Real.pi : ℂ) ^ 2 * fourierCoeffOn zero_lt_two_pi gC'' n) * hI_sq
  -- Take norms.
  rw [h_eq, norm_mul]
  have h_norm_coef : ‖(-1 / (n : ℂ) ^ 2 : ℂ)‖ = 1 / (n : ℝ) ^ 2 := by
    rw [norm_div, norm_neg, norm_one, norm_pow, Complex.norm_intCast]
    congr 1
    rw [← sq_abs (n : ℝ)]
  rw [h_norm_coef, div_mul_eq_mul_div, one_mul]
  rw [div_le_div_iff_of_pos_right (by positivity : (0 : ℝ) < (n : ℝ) ^ 2)]
  exact h_norm_gC''

-- ===========================================================================
-- Step 3+5: summability and Basel sum bound
-- ===========================================================================

/-- The function `n ↦ 1/n²` (with `0` at `n = 0`) is summable over `ℤ`. -/
lemma summable_indicator_one_div_int_sq :
    Summable (fun n : ℤ => if n = 0 then (0 : ℝ) else 1 / (n : ℝ) ^ 2) := by
  have h : Summable (fun n : ℤ => 1 / (n : ℝ) ^ 2) :=
    summable_one_div_int_pow.mpr (by norm_num)
  -- The two sequences differ only at n = 0, where one is 0 and the other is
  -- 1 / 0 ^ 2 = 0 (Mathlib convention for division by zero).
  refine h.congr (fun n => ?_)
  by_cases hn : n = 0
  · simp [hn]
  · simp [hn]

/-- The function `n ↦ F₂ / n²` (with `0` at `n = 0`) is summable over `ℤ`. -/
lemma summable_F₂_div_int_sq (F₂ : ℝ) :
    Summable (fun n : ℤ => if n = 0 then (0 : ℝ) else F₂ / (n : ℝ) ^ 2) := by
  have h := (summable_indicator_one_div_int_sq).mul_left F₂
  refine h.congr (fun n => ?_)
  by_cases hn : n = 0
  · simp [hn]
  · simp [hn]; ring

/-- The sum of `1/n²` over `n : ℤ \ {0}` (with `0` plugged at `n = 0`) equals
`π²/3`. Combines `hasSum_zeta_two` over `ℕ` with the Int ↔ Nat splitting. -/
lemma tsum_one_div_int_sq_excluding_zero :
    ∑' n : ℤ, (if n = 0 then (0 : ℝ) else 1 / (n : ℝ) ^ 2) = Real.pi ^ 2 / 3 := by
  -- Let `f n = if n = 0 then 0 else 1/n²` over ℤ.
  -- `∑'_{n∈ℕ} f n = ∑_{k≥1} 1/k² = π²/6` (since f 0 = 0 and f k = 1/k²
  -- for k ≥ 1, matching `hasSum_zeta_two`).
  -- Similarly `∑'_{n∈ℕ} f(-n) = π²/6`.
  -- Using `HasSum.of_nat_of_neg`: `∑'_{ℤ} f n = π²/6 + π²/6 - f 0 = π²/3`.
  set f : ℤ → ℝ := fun n => if n = 0 then 0 else 1 / (n : ℝ) ^ 2 with hf_def
  -- Both nat-side sums coincide with the standard Basel sequence.
  have hzeta := _root_.hasSum_zeta_two
  -- Show: ∑'_{n∈ℕ} f n = π²/6.
  have hpos : HasSum (fun n : ℕ => f n) (Real.pi ^ 2 / 6) := by
    -- For n : ℕ, f n = (if (n : ℤ) = 0 then 0 else 1/(n:ℝ)²). The two
    -- sequences agree pointwise: when n = 0, both equal 0 (the Basel
    -- sequence has 1/0² = 0 by Mathlib convention); when n ≥ 1, both
    -- equal 1/n².
    have hcongr : ∀ n : ℕ, f n = 1 / (n : ℝ) ^ 2 := by
      intro n
      by_cases hn : n = 0
      · subst hn; simp [f]
      · have hint : ((n : ℤ) : ℝ) ≠ 0 := by exact_mod_cast hn
        have hint' : (n : ℤ) ≠ 0 := by exact_mod_cast hn
        simp only [f, hint', if_false]
        push_cast
        rfl
    convert hzeta using 1
    funext n; exact hcongr n
  -- Show: ∑'_{n∈ℕ} f(-n) = π²/6.
  have hneg : HasSum (fun n : ℕ => f (-(n : ℤ))) (Real.pi ^ 2 / 6) := by
    have hcongr : ∀ n : ℕ, f (-(n : ℤ)) = 1 / (n : ℝ) ^ 2 := by
      intro n
      by_cases hn : n = 0
      · subst hn; simp [f]
      · have hint : (-(n : ℤ)) ≠ 0 := by
          intro h; apply hn
          have : (n : ℤ) = 0 := by linarith
          exact_mod_cast this
        simp only [f, hint, if_false]
        push_cast
        rw [neg_sq]
    convert hzeta using 1
    funext n; exact hcongr n
  -- Combine via HasSum.of_nat_of_neg: total sum = π²/6 + π²/6 - f 0.
  have hcombined : HasSum f (Real.pi ^ 2 / 6 + Real.pi ^ 2 / 6 - f 0) :=
    HasSum.of_nat_of_neg hpos hneg
  have hf_zero : f 0 = 0 := by simp [f]
  rw [hf_zero, sub_zero] at hcombined
  rw [hcombined.tsum_eq]
  ring

-- ===========================================================================
-- Step 4 (Fourier-series link): the harmonic-analysis content as a theorem
-- ===========================================================================

/-- Summability of the (zero-excluded) Fourier-coefficient norms for any
`C²` 2π-periodic real function. Follows from the IBP bound `‖c_n‖ ≤ F₂/n²`
and summability of `1/n²`. -/
lemma summable_fourierCoeff_norm_of_contDiff (g : ℝ → ℝ)
    (hg : ContDiff ℝ 2 g) (hg_per : Function.Periodic g (2 * Real.pi)) :
    Summable (fun n : ℤ =>
      if n = 0 then (0 : ℝ) else
        ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal g) n‖) := by
  -- Use ‖iteratedDeriv 2 g‖_∞ as F₂.
  -- A C² function is continuous, but not necessarily globally bounded —
  -- however the IBP bound is pointwise: for each x, |iteratedDeriv 2 g x|
  -- bounds ‖c_n‖ · n². We bound by a fixed constant via continuity on the
  -- compact interval [0, 2π] combined with periodicity.
  -- Build F₂ as a sup over the fundamental period.
  have hg''_cont : Continuous (iteratedDeriv 2 g) :=
    hg.continuous_iteratedDeriv 2 (by decide)
  have hg''_per : Function.Periodic (iteratedDeriv 2 g) (2 * Real.pi) := by
    -- iteratedDeriv 2 g = deriv (deriv g)
    have h_iter2 : iteratedDeriv 2 g = deriv (deriv g) := by
      rw [iteratedDeriv_succ, iteratedDeriv_one]
    rw [h_iter2]
    exact Periodic.deriv_periodic (Periodic.deriv_periodic hg_per)
  -- The continuous periodic function has a sup over [0, 2π], which is its
  -- global sup. Use `Continuous.exists_forall_le_of_hasCompactSupport`-like
  -- argument via `IsCompact.exists_isMaxOn`.
  have hcompact : IsCompact (Set.Icc (0 : ℝ) (2 * Real.pi)) := isCompact_Icc
  have hne : (Set.Icc (0 : ℝ) (2 * Real.pi)).Nonempty :=
    Set.nonempty_Icc.mpr (by linarith [Real.pi_pos])
  obtain ⟨x₀, _hx₀_mem, hx₀_max⟩ :=
    hcompact.exists_isMaxOn hne (hg''_cont.abs.continuousOn)
  set F₂ : ℝ := |iteratedDeriv 2 g x₀| with hF₂_def
  -- F₂ bounds |iteratedDeriv 2 g| on all of ℝ.
  have hF₂_bd : ∀ u, |iteratedDeriv 2 g u| ≤ F₂ := by
    intro u
    -- Reduce u to [0, 2π] via periodicity.
    have hpos : 0 < 2 * Real.pi := zero_lt_two_pi
    obtain ⟨y, hy_mem, hu_eq⟩ := hg''_per.exists_mem_Ico₀ hpos u
    -- Note: `iteratedDeriv 2 g` is periodic; the original |·| isn't directly,
    -- but |iteratedDeriv 2 g u| = |iteratedDeriv 2 g y|.
    rw [hu_eq]
    -- y ∈ Ico 0 (2 * Real.pi) ⊆ Icc 0 (2 * Real.pi).
    have hmem : y ∈ Set.Icc (0 : ℝ) (2 * Real.pi) :=
      Set.mem_Icc.mpr ⟨hy_mem.1, hy_mem.2.le⟩
    exact hx₀_max hmem
  -- Now apply the comparison: ‖c_n‖ ≤ F₂/n² and summability of F₂/n².
  refine Summable.of_nonneg_of_le ?_ ?_ (summable_F₂_div_int_sq F₂)
  · intro n
    by_cases hn : n = 0
    · simp [hn]
    · simp [hn]
  · intro n
    by_cases hn : n = 0
    · simp [hn]
    · simp only [hn, if_false]
      exact fourierCoeff_norm_le_F₂_div_sq g hg hg_per F₂ hF₂_bd hn

/-- Summability (over ℤ) of `n ↦ hilbertMult n · c_n · exp(inu)` for any
`C²` 2π-periodic real `g`. The complex Fourier series of the Hilbert-
transformed signal converges absolutely. -/
lemma summable_hilbertMult_fourierCoeff_exp (g : ℝ → ℝ)
    (hg : ContDiff ℝ 2 g) (hg_per : Function.Periodic g (2 * Real.pi)) (u : ℝ) :
    Summable (fun n : ℤ =>
      hilbertMult n *
        fourierCoeffOn zero_lt_two_pi (gCircOfReal g) n *
        Complex.exp ((n : ℂ) * Complex.I * (u : ℂ))) := by
  -- Compare the norm pointwise to the (zero-excluded) Fourier-coefficient norms.
  apply Summable.of_norm
  refine Summable.of_nonneg_of_le ?_ ?_
    (summable_fourierCoeff_norm_of_contDiff g hg hg_per)
  · intro n; exact norm_nonneg _
  · intro n
    -- ‖hilbertMult n · c_n · exp(inu)‖ = ‖hilbertMult n‖ · ‖c_n‖ · 1.
    rw [norm_mul, norm_mul]
    have hexp_norm :
        ‖Complex.exp ((n : ℂ) * Complex.I * (u : ℂ))‖ = 1 := by
      -- (n : ℂ) * I * (u : ℂ) = ((n*u : ℝ) : ℂ) * I, which is purely imaginary.
      have h : (n : ℂ) * Complex.I * (u : ℂ) =
          (((n : ℝ) * u : ℝ) : ℂ) * Complex.I := by
        push_cast; ring
      rw [h, Complex.norm_exp_ofReal_mul_I]
    rw [hexp_norm, mul_one]
    by_cases hn : n = 0
    · simp [hilbertMult, hn]
    · simp only [hn, if_false]
      rw [norm_hilbertMult_of_ne_zero hn]
      rw [one_mul]

/-- **Fourier representation of `periodicHilbert` for C² periodic inputs.**

For `g ∈ C²(ℝ)` 2π-periodic, the periodic Hilbert transform's pointwise
modulus is bounded by the (zero-excluded) `ℓ¹` norm of the Fourier
coefficients of `g`:

```
∀ u, |periodicHilbert g u| ≤ ∑'_{n : ℤ} (if n = 0 then 0 else ‖c_n‖)
```

Proof: starting from the concrete definition
`periodicHilbert g u = Re ∑'_{n} hilbertMult n · c_n · e^{inu}`, take
`|Re ·| ≤ ‖·‖` then `‖∑'‖ ≤ ∑' ‖·‖` (via `norm_tsum_le_tsum_norm`), then
use `‖hilbertMult n‖ = 1` for `n ≠ 0`, `hilbertMult 0 = 0`, and
`‖exp(inu)‖ = 1`. -/
theorem periodicHilbert_le_tsum_fourierCoeff (g : ℝ → ℝ)
    (hg : ContDiff ℝ 2 g) (hg_per : Function.Periodic g (2 * Real.pi)) (u : ℝ) :
    |periodicHilbert g u| ≤
      ∑' n : ℤ,
        (if n = 0 then 0 else
          ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal g) n‖) := by
  -- Unfold the definition.
  set S : ℂ := ∑' n : ℤ,
      hilbertMult n *
        fourierCoeffOn zero_lt_two_pi (gCircOfReal g) n *
        Complex.exp ((n : ℂ) * Complex.I * (u : ℂ)) with hS_def
  have hPH : periodicHilbert g u = S.re := rfl
  rw [hPH]
  -- Step A: |Re S| ≤ ‖S‖.
  have hA : |S.re| ≤ ‖S‖ := Complex.abs_re_le_norm S
  -- Step B: ‖S‖ ≤ ∑' n, ‖summand n‖ (uses summability of summand norms).
  have hsum_norm :
      Summable (fun n : ℤ =>
        ‖hilbertMult n *
            fourierCoeffOn zero_lt_two_pi (gCircOfReal g) n *
            Complex.exp ((n : ℂ) * Complex.I * (u : ℂ))‖) := by
    exact (summable_hilbertMult_fourierCoeff_exp g hg hg_per u).norm
  have hB : ‖S‖ ≤
      ∑' n : ℤ,
        ‖hilbertMult n *
            fourierCoeffOn zero_lt_two_pi (gCircOfReal g) n *
            Complex.exp ((n : ℂ) * Complex.I * (u : ℂ))‖ := by
    rw [hS_def]
    exact norm_tsum_le_tsum_norm hsum_norm
  -- Step C: pointwise, each ‖summand n‖ equals (if n = 0 then 0 else ‖c_n‖).
  have hC : ∀ n : ℤ,
      ‖hilbertMult n *
          fourierCoeffOn zero_lt_two_pi (gCircOfReal g) n *
          Complex.exp ((n : ℂ) * Complex.I * (u : ℂ))‖ =
        (if n = 0 then (0 : ℝ) else
          ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal g) n‖) := by
    intro n
    rw [norm_mul, norm_mul]
    have hexp_norm :
        ‖Complex.exp ((n : ℂ) * Complex.I * (u : ℂ))‖ = 1 := by
      have h : (n : ℂ) * Complex.I * (u : ℂ) =
          (((n : ℝ) * u : ℝ) : ℂ) * Complex.I := by
        push_cast; ring
      rw [h, Complex.norm_exp_ofReal_mul_I]
    rw [hexp_norm, mul_one]
    by_cases hn : n = 0
    · simp [hilbertMult, hn]
    · simp only [hn, if_false]
      rw [norm_hilbertMult_of_ne_zero hn, one_mul]
  -- Combine: |Re S| ≤ ‖S‖ ≤ ∑' n, ‖summand n‖ = ∑' n, (if n = 0 then 0 else ‖c_n‖).
  calc |S.re|
      ≤ ‖S‖ := hA
    _ ≤ ∑' n : ℤ,
          ‖hilbertMult n *
              fourierCoeffOn zero_lt_two_pi (gCircOfReal g) n *
              Complex.exp ((n : ℂ) * Complex.I * (u : ℂ))‖ := hB
    _ = ∑' n : ℤ,
          (if n = 0 then (0 : ℝ) else
            ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal g) n‖) :=
        tsum_congr hC

-- ===========================================================================
-- Main theorem
-- ===========================================================================

/-- N3: Fourier-tail `π²/3` bound for `‖periodicHilbert g‖_∞`.

For `g ∈ C²(𝕋)` 2π-periodic with `‖g''‖_∞ ≤ F₂`, the periodic Hilbert
transform is bounded uniformly by `F₂ · π²/3`.

The constant `π²/3` is sharp: it equals `2 · π²/6 = 2 · (Σ 1/k²)`. -/
theorem fourier_tail_hilbert_bound (g : ℝ → ℝ)
    (hg : ContDiff ℝ 2 g) (hg_per : Function.Periodic g (2 * Real.pi))
    (F₂ : ℝ) (hF₂ : ∀ u, |iteratedDeriv 2 g u| ≤ F₂) :
    ∀ u, |periodicHilbert g u| ≤ F₂ * Real.pi ^ 2 / 3 := by
  -- F₂ is nonneg because it bounds a nonneg quantity.
  have hF₂_nn : 0 ≤ F₂ := le_trans (abs_nonneg _) (hF₂ 0)
  intro u
  -- Step 1: link |Hg(u)| to the tail-sum of Fourier coefficients.
  have hHsum := periodicHilbert_le_tsum_fourierCoeff g hg hg_per u
  -- Step 2: bound each coefficient by F₂/n².
  have hcoeff : ∀ n : ℤ,
      (if n = 0 then (0 : ℝ) else
        ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal g) n‖)
        ≤ (if n = 0 then 0 else F₂ / (n : ℝ) ^ 2) := by
    intro n
    by_cases hn : n = 0
    · simp [hn]
    · simp only [hn, if_false]
      exact fourierCoeff_norm_le_F₂_div_sq g hg hg_per F₂ hF₂ hn
  -- Summability of the coefficient-norms via comparison.
  have hsum_lhs : Summable (fun n : ℤ =>
      if n = 0 then (0 : ℝ) else
        ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal g) n‖) := by
    refine Summable.of_nonneg_of_le ?_ hcoeff (summable_F₂_div_int_sq F₂)
    intro n; by_cases hn : n = 0 <;> simp [hn]
  -- Step 3+5: the bounding series sums to F₂ · π²/3.
  have hsum_bound :
      ∑' n : ℤ, (if n = 0 then (0 : ℝ) else F₂ / (n : ℝ) ^ 2)
        = F₂ * Real.pi ^ 2 / 3 := by
    have h_split : ∀ n : ℤ,
        (if n = 0 then (0 : ℝ) else F₂ / (n : ℝ) ^ 2)
          = F₂ * (if n = 0 then (0 : ℝ) else 1 / (n : ℝ) ^ 2) := by
      intro n; by_cases hn : n = 0
      · simp [hn]
      · simp [hn]; ring
    calc ∑' n : ℤ, (if n = 0 then (0 : ℝ) else F₂ / (n : ℝ) ^ 2)
        = ∑' n : ℤ, F₂ * (if n = 0 then (0 : ℝ) else 1 / (n : ℝ) ^ 2) :=
          tsum_congr h_split
      _ = F₂ * ∑' n : ℤ, (if n = 0 then (0 : ℝ) else 1 / (n : ℝ) ^ 2) :=
          tsum_mul_left
      _ = F₂ * (Real.pi ^ 2 / 3) := by
          rw [tsum_one_div_int_sq_excluding_zero]
      _ = F₂ * Real.pi ^ 2 / 3 := by ring
  -- Combine: |Hg(u)| ≤ ∑' (F₂/n² bounds) = F₂ · π²/3.
  calc |periodicHilbert g u|
      ≤ ∑' n : ℤ,
          (if n = 0 then (0 : ℝ) else
            ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal g) n‖) := hHsum
    _ ≤ ∑' n : ℤ, (if n = 0 then (0 : ℝ) else F₂ / (n : ℝ) ^ 2) :=
        hsum_lhs.tsum_le_tsum hcoeff (summable_F₂_div_int_sq F₂)
    _ = F₂ * Real.pi ^ 2 / 3 := hsum_bound

-- ===========================================================================
-- Trig-mode action and additivity (formerly axioms in Definitions.lean)
-- ===========================================================================

/-- Helper: `fourier (-n) ((x : ℝ) : AddCircle (2π - 0)) = exp(-i n x)`
on the line. -/
lemma fourier_neg_coe_apply_real (n : ℤ) (x : ℝ) :
    fourier (-n) ((x : ℝ) : AddCircle (2 * Real.pi - 0)) =
      Complex.exp (-((n : ℂ) * Complex.I * (x : ℂ))) := by
  rw [show (2 * Real.pi - 0 : ℝ) = 2 * Real.pi by ring]
  rw [fourier_coe_apply]
  congr 1
  have hpi : (Real.pi : ℂ) ≠ 0 := Complex.ofReal_ne_zero.mpr Real.pi_pos.ne'
  push_cast
  field_simp

/-- Helper: the Fourier coefficient on `[0, 2π]` of `exp(i m x)` for integer
`m`. Equals `1` when `n = m`, else `0`. -/
lemma fourierCoeffOn_exp_intMul_I (m n : ℤ) :
    fourierCoeffOn zero_lt_two_pi
        (fun x : ℝ => Complex.exp ((m : ℂ) * Complex.I * (x : ℂ))) n =
      (if n = m then 1 else 0 : ℂ) := by
  rw [fourierCoeffOn_eq_integral]
  -- The integrand is `fourier(-n)(x) • exp(i m x) = exp(i (m-n) x)`.
  have h_integrand : ∀ x : ℝ,
      fourier (-n) ((x : ℝ) : AddCircle (2 * Real.pi - 0)) •
        Complex.exp ((m : ℂ) * Complex.I * (x : ℂ)) =
      Complex.exp (((m - n : ℤ) : ℂ) * Complex.I * (x : ℂ)) := by
    intro x
    rw [fourier_neg_coe_apply_real n x]
    rw [smul_eq_mul, ← Complex.exp_add]
    congr 1
    push_cast
    ring
  rw [intervalIntegral.integral_congr (g := fun x =>
      Complex.exp (((m - n : ℤ) : ℂ) * Complex.I * (x : ℂ)))
      (fun x _ => h_integrand x)]
  by_cases hmn : n = m
  · -- n = m: integrand is 1, integral is 2π, prefactor cancels.
    rw [if_pos hmn]
    -- Rewrite the integrand as the constant function 1.
    have h_mn_zero : ((m - n : ℤ) : ℂ) = 0 := by
      have : (m - n : ℤ) = 0 := by omega
      exact_mod_cast this
    have h_const : ∀ x : ℝ,
        Complex.exp (((m - n : ℤ) : ℂ) * Complex.I * (x : ℂ)) = 1 := by
      intro x
      rw [h_mn_zero, zero_mul, zero_mul, Complex.exp_zero]
    rw [intervalIntegral.integral_congr (fun x _ => h_const x)]
    rw [intervalIntegral.integral_const]
    have h2pi_ne : (2 * Real.pi : ℝ) ≠ 0 := by
      have := Real.pi_pos
      linarith
    rw [smul_smul]
    rw [show (1 / (2 * Real.pi - 0) : ℝ) * (2 * Real.pi - 0 : ℝ) = 1 from by
        rw [sub_zero]; field_simp]
    rw [one_smul]
  · -- n ≠ m: integrand is exp(i(m-n)x), nonzero exponent, integral over [0, 2π] vanishes.
    rw [if_neg hmn]
    have hmn_ne : ((m - n : ℤ) : ℂ) ≠ 0 := by
      have hint_ne : (m - n : ℤ) ≠ 0 := by
        intro h
        apply hmn
        omega
      exact_mod_cast hint_ne
    have hI : Complex.I ≠ 0 := Complex.I_ne_zero
    have hc_ne : ((m - n : ℤ) : ℂ) * Complex.I ≠ 0 := mul_ne_zero hmn_ne hI
    rw [show (fun x : ℝ =>
            Complex.exp (((m - n : ℤ) : ℂ) * Complex.I * (x : ℂ))) =
          (fun x : ℝ => Complex.exp ((((m - n : ℤ) : ℂ) * Complex.I) * (x : ℂ))) from rfl]
    rw [integral_exp_mul_complex hc_ne]
    -- Now we need (exp(c * 2π) - exp(c * 0))/c = 0.
    have hsum :
        Complex.exp (((m - n : ℤ) : ℂ) * Complex.I * ((2 * Real.pi : ℝ) : ℂ)) = 1 := by
      have h_eq :
          ((m - n : ℤ) : ℂ) * Complex.I * ((2 * Real.pi : ℝ) : ℂ) =
          ((m - n : ℤ) : ℂ) * (2 * Real.pi * Complex.I) := by
        push_cast; ring
      rw [h_eq]
      exact Complex.exp_int_mul_two_pi_mul_I (m - n)
    have h0 : ((m - n : ℤ) : ℂ) * Complex.I * ((0 : ℝ) : ℂ) = 0 := by
      simp
    rw [hsum, h0, Complex.exp_zero, sub_self, zero_div, smul_zero]


/-- Helper: `∫_0^{2π} exp(i·m·x) dx = if m = 0 then 2π else 0` for integer `m`. -/
lemma integral_exp_intMul_I_two_pi (m : ℤ) :
    ∫ x in (0 : ℝ)..(2 * Real.pi),
        Complex.exp ((m : ℂ) * Complex.I * (x : ℂ)) =
      (if m = 0 then ((2 * Real.pi : ℝ) : ℂ) else 0) := by
  have h_arg : ∀ x : ℝ,
      ((m : ℂ) * Complex.I * (x : ℂ)) =
      (((m : ℂ) * Complex.I) * (x : ℂ)) := by
    intro x; ring
  by_cases hm : m = 0
  · subst hm
    rw [if_pos rfl]
    simp only [Int.cast_zero, zero_mul]
    rw [intervalIntegral.integral_congr (g := fun _ : ℝ => (1 : ℂ))
        (fun x _ => by rw [Complex.exp_zero])]
    rw [intervalIntegral.integral_const, sub_zero]
    show (2 * Real.pi : ℝ) • (1 : ℂ) = ((2 * Real.pi : ℝ) : ℂ)
    rw [Complex.real_smul, mul_one]
  · rw [if_neg hm]
    have hI : Complex.I ≠ 0 := Complex.I_ne_zero
    have hmC : (m : ℂ) ≠ 0 := by exact_mod_cast hm
    have hc_ne : (m : ℂ) * Complex.I ≠ 0 := mul_ne_zero hmC hI
    rw [show (fun x : ℝ => Complex.exp ((m : ℂ) * Complex.I * (x : ℂ))) =
            (fun x : ℝ => Complex.exp (((m : ℂ) * Complex.I) * (x : ℂ))) from rfl]
    rw [integral_exp_mul_complex hc_ne]
    have h_2pi :
        Complex.exp ((m : ℂ) * Complex.I * ((2 * Real.pi : ℝ) : ℂ)) = 1 := by
      have h_eq :
          (m : ℂ) * Complex.I * ((2 * Real.pi : ℝ) : ℂ) =
          (m : ℂ) * (2 * Real.pi * Complex.I) := by
        push_cast; ring
      rw [h_eq]
      exact Complex.exp_int_mul_two_pi_mul_I m
    have h_0 : (m : ℂ) * Complex.I * ((0 : ℝ) : ℂ) = 0 := by simp
    rw [h_2pi, h_0, Complex.exp_zero, sub_self, zero_div]

/-- Helper: Fourier coefficient of `cos(k·)` for `k : ℕ`, `k ≥ 1`.

Equals `1/2` at `n = ±k`, `0` elsewhere. -/
lemma fourierCoeffOn_cos_real (k : ℕ) (hk : 1 ≤ k) (n : ℤ) :
    fourierCoeffOn zero_lt_two_pi
        (gCircOfReal (fun t : ℝ => Real.cos ((k : ℝ) * t))) n =
      (if n = (k : ℤ) then (1/2 : ℂ) else
        if n = -(k : ℤ) then (1/2 : ℂ) else 0) := by
  rw [fourierCoeffOn_eq_integral]
  rw [show (1 / (2 * Real.pi - 0) : ℝ) = (1 / (2 * Real.pi)) from by rw [sub_zero]]
  -- The integrand: fourier(-n)(x) • (cos(kx) : ℂ) = exp(-i n x) * cos(kx).
  -- Rewrite as (1/2) (exp(i(k-n)x) + exp(-i(k+n)x)).
  have h_integrand : ∀ x : ℝ,
      fourier (-n) ((x : ℝ) : AddCircle (2 * Real.pi - 0)) •
          gCircOfReal (fun s : ℝ => Real.cos ((k : ℝ) * s)) x =
        (1/2 : ℂ) *
          (Complex.exp ((((k : ℤ) - n : ℤ) : ℂ) * Complex.I * (x : ℂ)) +
           Complex.exp (((-((k : ℤ) + n) : ℤ) : ℂ) * Complex.I * (x : ℂ))) := by
    intro x
    rw [smul_eq_mul, fourier_neg_coe_apply_real n x]
    -- gCircOfReal cos(kx) = (cos(kx) : ℂ) = (exp(ikx) + exp(-ikx))/2.
    have h_cos : (gCircOfReal (fun s : ℝ => Real.cos ((k : ℝ) * s)) x : ℂ) =
        (Complex.exp (((k : ℤ) : ℂ) * Complex.I * (x : ℂ)) +
         Complex.exp ((-(k : ℤ) : ℂ) * Complex.I * (x : ℂ))) / 2 := by
      show ((Real.cos ((k : ℝ) * x) : ℝ) : ℂ) = _
      rw [Complex.ofReal_cos, Complex.cos]
      have h_arg : (((k : ℝ) * x : ℝ) : ℂ) = ((k : ℤ) : ℂ) * (x : ℂ) := by
        push_cast; ring
      rw [h_arg]
      have h_eq1 :
          Complex.exp ((((k : ℤ) : ℂ) * (x : ℂ)) * Complex.I) =
          Complex.exp (((k : ℤ) : ℂ) * Complex.I * (x : ℂ)) := by
        congr 1; ring
      have h_eq2 :
          Complex.exp (-(((k : ℤ) : ℂ) * (x : ℂ)) * Complex.I) =
          Complex.exp ((-(k : ℤ) : ℂ) * Complex.I * (x : ℂ)) := by
        congr 1; ring
      rw [h_eq1, h_eq2]
    rw [h_cos]
    -- Now: exp(-i n x) * ((exp(ikx) + exp(-ikx))/2) = (1/2)(exp(i(k-n)x) + exp(-i(k+n)x)).
    have hexp_add1 :
        Complex.exp (-((n : ℂ) * Complex.I * (x : ℂ))) *
            Complex.exp (((k : ℤ) : ℂ) * Complex.I * (x : ℂ)) =
        Complex.exp ((((k : ℤ) - n : ℤ) : ℂ) * Complex.I * (x : ℂ)) := by
      rw [← Complex.exp_add]
      congr 1; push_cast; ring
    have hexp_add2 :
        Complex.exp (-((n : ℂ) * Complex.I * (x : ℂ))) *
            Complex.exp ((-(k : ℤ) : ℂ) * Complex.I * (x : ℂ)) =
        Complex.exp (((-((k : ℤ) + n) : ℤ) : ℂ) * Complex.I * (x : ℂ)) := by
      rw [← Complex.exp_add]
      congr 1; push_cast; ring
    have h_distrib :
        Complex.exp (-((n : ℂ) * Complex.I * (x : ℂ))) *
          ((Complex.exp (((k : ℤ) : ℂ) * Complex.I * (x : ℂ)) +
            Complex.exp ((-(k : ℤ) : ℂ) * Complex.I * (x : ℂ))) / 2) =
        (1/2 : ℂ) *
          (Complex.exp (-((n : ℂ) * Complex.I * (x : ℂ))) *
              Complex.exp (((k : ℤ) : ℂ) * Complex.I * (x : ℂ)) +
           Complex.exp (-((n : ℂ) * Complex.I * (x : ℂ))) *
              Complex.exp ((-(k : ℤ) : ℂ) * Complex.I * (x : ℂ))) := by
      ring
    rw [h_distrib, hexp_add1, hexp_add2]
  rw [intervalIntegral.integral_congr (fun x _ => h_integrand x)]
  -- Now split the integral over the addition.
  have hcont_exp1 : Continuous
      (fun x : ℝ => Complex.exp ((((k : ℤ) - n : ℤ) : ℂ) * Complex.I * (x : ℂ))) := by
    apply Complex.continuous_exp.comp
    fun_prop
  have hcont_exp2 : Continuous
      (fun x : ℝ => Complex.exp (((-((k : ℤ) + n) : ℤ) : ℂ) * Complex.I * (x : ℂ))) := by
    apply Complex.continuous_exp.comp
    fun_prop
  have hint1 : IntervalIntegrable
      (fun x : ℝ => Complex.exp ((((k : ℤ) - n : ℤ) : ℂ) * Complex.I * (x : ℂ)))
      volume 0 (2 * Real.pi) :=
    hcont_exp1.intervalIntegrable _ _
  have hint2 : IntervalIntegrable
      (fun x : ℝ => Complex.exp (((-((k : ℤ) + n) : ℤ) : ℂ) * Complex.I * (x : ℂ)))
      volume 0 (2 * Real.pi) :=
    hcont_exp2.intervalIntegrable _ _
  -- Rewrite as integral of (1/2)*(... + ...) and split.
  have h_eq :
      ∫ x in (0 : ℝ)..(2 * Real.pi),
        (1/2 : ℂ) *
          (Complex.exp ((((k : ℤ) - n : ℤ) : ℂ) * Complex.I * (x : ℂ)) +
           Complex.exp (((-((k : ℤ) + n) : ℤ) : ℂ) * Complex.I * (x : ℂ))) =
      (1/2 : ℂ) *
        ((∫ x in (0 : ℝ)..(2 * Real.pi),
              Complex.exp ((((k : ℤ) - n : ℤ) : ℂ) * Complex.I * (x : ℂ))) +
         (∫ x in (0 : ℝ)..(2 * Real.pi),
              Complex.exp (((-((k : ℤ) + n) : ℤ) : ℂ) * Complex.I * (x : ℂ)))) := by
    have h_inner := intervalIntegral.integral_add hint1 hint2 (μ := volume) (a := (0:ℝ)) (b := 2 * Real.pi)
    rw [← h_inner]
    -- Now the goal is ∫ 1/2 * (e1+e2) = 1/2 * ∫ (e1+e2)
    exact intervalIntegral.integral_const_mul _ _
  rw [h_eq]
  rw [integral_exp_intMul_I_two_pi ((k : ℤ) - n)]
  rw [integral_exp_intMul_I_two_pi (-((k : ℤ) + n))]
  -- Now split based on n = k, n = -k.
  by_cases hnk : n = (k : ℤ)
  · -- n = k: k - n = 0, -(k + n) = -2k ≠ 0 (since k ≥ 1).
    rw [if_pos hnk]
    have h1 : ((k : ℤ) - n : ℤ) = 0 := by omega
    have h2 : (-((k : ℤ) + n) : ℤ) ≠ 0 := by
      have : (k : ℤ) ≥ 1 := by exact_mod_cast hk
      omega
    rw [if_pos h1, if_neg h2]
    have hpi : (Real.pi : ℝ) > 0 := Real.pi_pos
    have hpi_ne : (2 * Real.pi : ℝ) ≠ 0 := by linarith
    have h_smul : (1 / (2 * Real.pi) : ℝ) •
        ((1 / 2 : ℂ) * (((2 * Real.pi : ℝ) : ℂ) + 0)) = (1 / 2 : ℂ) := by
      rw [show ((2 * Real.pi : ℝ) : ℂ) + 0 = ((2 * Real.pi : ℝ) : ℂ) from by ring]
      rw [Complex.real_smul]
      push_cast
      field_simp
    exact h_smul
  · by_cases hnnegk : n = -(k : ℤ)
    · -- n = -k: k - n = 2k ≠ 0, -(k + n) = 0.
      rw [if_neg hnk, if_pos hnnegk]
      have h1 : ((k : ℤ) - n : ℤ) ≠ 0 := by
        have : (k : ℤ) ≥ 1 := by exact_mod_cast hk
        omega
      have h2 : (-((k : ℤ) + n) : ℤ) = 0 := by omega
      rw [if_neg h1, if_pos h2]
      have hpi : (Real.pi : ℝ) > 0 := Real.pi_pos
      have hpi_ne : (2 * Real.pi : ℝ) ≠ 0 := by linarith
      have h_smul : (1 / (2 * Real.pi) : ℝ) •
          ((1 / 2 : ℂ) * (0 + ((2 * Real.pi : ℝ) : ℂ))) = (1 / 2 : ℂ) := by
        rw [show (0 + ((2 * Real.pi : ℝ) : ℂ)) = ((2 * Real.pi : ℝ) : ℂ) from by ring]
        rw [Complex.real_smul]
        push_cast
        field_simp
      exact h_smul
    · -- n ≠ ±k: both integrals vanish.
      rw [if_neg hnk, if_neg hnnegk]
      have h1 : ((k : ℤ) - n : ℤ) ≠ 0 := by
        intro h
        apply hnk
        omega
      have h2 : (-((k : ℤ) + n) : ℤ) ≠ 0 := by
        intro h
        apply hnnegk
        omega
      rw [if_neg h1, if_neg h2]
      have h_smul : (1 / (2 * Real.pi) : ℝ) • ((1 / 2 : ℂ) * (0 + 0)) = (0 : ℂ) := by
        rw [show ((0 : ℂ) + 0) = (0 : ℂ) from by ring]
        rw [show ((1 / 2 : ℂ) * 0) = (0 : ℂ) from by ring]
        rw [Complex.real_smul]
        ring
      exact h_smul

/-- Helper: Fourier coefficient of `sin(k·)` for `k : ℕ`, `k ≥ 1`.

Equals `-i/2` at `n = k`, `i/2` at `n = -k`, `0` elsewhere. -/
lemma fourierCoeffOn_sin_real (k : ℕ) (hk : 1 ≤ k) (n : ℤ) :
    fourierCoeffOn zero_lt_two_pi
        (gCircOfReal (fun t : ℝ => Real.sin ((k : ℝ) * t))) n =
      (if n = (k : ℤ) then -Complex.I/2 else
        if n = -(k : ℤ) then Complex.I/2 else 0) := by
  rw [fourierCoeffOn_eq_integral]
  rw [show (1 / (2 * Real.pi - 0) : ℝ) = (1 / (2 * Real.pi)) from by rw [sub_zero]]
  -- Use sin(kx) = (exp(ikx) - exp(-ikx))/(2i) = -i/2 (exp(ikx) - exp(-ikx)).
  -- Integrand: fourier(-n)(x) • (sin(kx) : ℂ) = exp(-i n x) * sin(kx)
  --          = -i/2 (exp(i(k-n)x) - exp(-i(k+n)x)).
  have h_integrand : ∀ x : ℝ,
      fourier (-n) ((x : ℝ) : AddCircle (2 * Real.pi - 0)) •
          gCircOfReal (fun s : ℝ => Real.sin ((k : ℝ) * s)) x =
        (-Complex.I/2 : ℂ) *
          (Complex.exp ((((k : ℤ) - n : ℤ) : ℂ) * Complex.I * (x : ℂ)) -
           Complex.exp (((-((k : ℤ) + n) : ℤ) : ℂ) * Complex.I * (x : ℂ))) := by
    intro x
    rw [smul_eq_mul, fourier_neg_coe_apply_real n x]
    -- gCircOfReal sin(kx) = (sin(kx) : ℂ) = (exp(-ikx) - exp(ikx)) * I / 2
    --                    = -i/2 (exp(ikx) - exp(-ikx)).
    have h_sin : (gCircOfReal (fun s : ℝ => Real.sin ((k : ℝ) * s)) x : ℂ) =
        (-Complex.I/2 : ℂ) *
          (Complex.exp (((k : ℤ) : ℂ) * Complex.I * (x : ℂ)) -
           Complex.exp ((-(k : ℤ) : ℂ) * Complex.I * (x : ℂ))) := by
      show ((Real.sin ((k : ℝ) * x) : ℝ) : ℂ) = _
      rw [Complex.ofReal_sin, Complex.sin]
      have h_arg : (((k : ℝ) * x : ℝ) : ℂ) = ((k : ℤ) : ℂ) * (x : ℂ) := by
        push_cast; ring
      rw [h_arg]
      have h_eq1 :
          Complex.exp ((-(((k : ℤ) : ℂ) * (x : ℂ))) * Complex.I) =
          Complex.exp ((-(k : ℤ) : ℂ) * Complex.I * (x : ℂ)) := by
        congr 1; ring
      have h_eq2 :
          Complex.exp ((((k : ℤ) : ℂ) * (x : ℂ)) * Complex.I) =
          Complex.exp (((k : ℤ) : ℂ) * Complex.I * (x : ℂ)) := by
        congr 1; ring
      rw [h_eq1, h_eq2]
      have hI2 : (2 : ℂ) * Complex.I ≠ 0 := by apply mul_ne_zero <;> norm_num
      field_simp
      ring
    rw [h_sin]
    -- Now: exp(-i n x) * (-i/2 (exp(ikx) - exp(-ikx)))
    --   = -i/2 * (exp(-i n x) * exp(ikx) - exp(-i n x) * exp(-ikx))
    --   = -i/2 * (exp(i(k-n)x) - exp(-i(k+n)x)).
    have hexp_add1 :
        Complex.exp (-((n : ℂ) * Complex.I * (x : ℂ))) *
            Complex.exp (((k : ℤ) : ℂ) * Complex.I * (x : ℂ)) =
        Complex.exp ((((k : ℤ) - n : ℤ) : ℂ) * Complex.I * (x : ℂ)) := by
      rw [← Complex.exp_add]
      congr 1; push_cast; ring
    have hexp_add2 :
        Complex.exp (-((n : ℂ) * Complex.I * (x : ℂ))) *
            Complex.exp ((-(k : ℤ) : ℂ) * Complex.I * (x : ℂ)) =
        Complex.exp (((-((k : ℤ) + n) : ℤ) : ℂ) * Complex.I * (x : ℂ)) := by
      rw [← Complex.exp_add]
      congr 1; push_cast; ring
    have h_distrib :
        Complex.exp (-((n : ℂ) * Complex.I * (x : ℂ))) *
          ((-Complex.I/2 : ℂ) *
            (Complex.exp (((k : ℤ) : ℂ) * Complex.I * (x : ℂ)) -
             Complex.exp ((-(k : ℤ) : ℂ) * Complex.I * (x : ℂ)))) =
        (-Complex.I/2 : ℂ) *
          (Complex.exp (-((n : ℂ) * Complex.I * (x : ℂ))) *
              Complex.exp (((k : ℤ) : ℂ) * Complex.I * (x : ℂ)) -
           Complex.exp (-((n : ℂ) * Complex.I * (x : ℂ))) *
              Complex.exp ((-(k : ℤ) : ℂ) * Complex.I * (x : ℂ))) := by
      ring
    rw [h_distrib, hexp_add1, hexp_add2]
  rw [intervalIntegral.integral_congr (fun x _ => h_integrand x)]
  -- Now split the integral over the subtraction.
  have hcont_exp1 : Continuous
      (fun x : ℝ => Complex.exp ((((k : ℤ) - n : ℤ) : ℂ) * Complex.I * (x : ℂ))) := by
    apply Complex.continuous_exp.comp
    fun_prop
  have hcont_exp2 : Continuous
      (fun x : ℝ => Complex.exp (((-((k : ℤ) + n) : ℤ) : ℂ) * Complex.I * (x : ℂ))) := by
    apply Complex.continuous_exp.comp
    fun_prop
  have hint1 : IntervalIntegrable
      (fun x : ℝ => Complex.exp ((((k : ℤ) - n : ℤ) : ℂ) * Complex.I * (x : ℂ)))
      volume 0 (2 * Real.pi) :=
    hcont_exp1.intervalIntegrable _ _
  have hint2 : IntervalIntegrable
      (fun x : ℝ => Complex.exp (((-((k : ℤ) + n) : ℤ) : ℂ) * Complex.I * (x : ℂ)))
      volume 0 (2 * Real.pi) :=
    hcont_exp2.intervalIntegrable _ _
  have h_eq :
      ∫ x in (0 : ℝ)..(2 * Real.pi),
        (-Complex.I/2 : ℂ) *
          (Complex.exp ((((k : ℤ) - n : ℤ) : ℂ) * Complex.I * (x : ℂ)) -
           Complex.exp (((-((k : ℤ) + n) : ℤ) : ℂ) * Complex.I * (x : ℂ))) =
      (-Complex.I/2 : ℂ) *
        ((∫ x in (0 : ℝ)..(2 * Real.pi),
              Complex.exp ((((k : ℤ) - n : ℤ) : ℂ) * Complex.I * (x : ℂ))) -
         (∫ x in (0 : ℝ)..(2 * Real.pi),
              Complex.exp (((-((k : ℤ) + n) : ℤ) : ℂ) * Complex.I * (x : ℂ)))) := by
    have h_inner := intervalIntegral.integral_sub hint1 hint2 (μ := volume) (a := (0:ℝ)) (b := 2 * Real.pi)
    rw [← h_inner]
    exact intervalIntegral.integral_const_mul _ _
  rw [h_eq]
  rw [integral_exp_intMul_I_two_pi ((k : ℤ) - n)]
  rw [integral_exp_intMul_I_two_pi (-((k : ℤ) + n))]
  by_cases hnk : n = (k : ℤ)
  · rw [if_pos hnk]
    have h1 : ((k : ℤ) - n : ℤ) = 0 := by omega
    have h2 : (-((k : ℤ) + n) : ℤ) ≠ 0 := by
      have : (k : ℤ) ≥ 1 := by exact_mod_cast hk
      omega
    rw [if_pos h1, if_neg h2]
    have hpi : (Real.pi : ℝ) > 0 := Real.pi_pos
    have hpi_ne : (2 * Real.pi : ℝ) ≠ 0 := by linarith
    have h_smul : (1 / (2 * Real.pi) : ℝ) •
        ((-Complex.I / 2 : ℂ) * (((2 * Real.pi : ℝ) : ℂ) - 0)) =
        (-Complex.I / 2 : ℂ) := by
      rw [show (((2 * Real.pi : ℝ) : ℂ) - 0) = ((2 * Real.pi : ℝ) : ℂ) from by ring]
      rw [Complex.real_smul]
      push_cast
      field_simp
    exact h_smul
  · by_cases hnnegk : n = -(k : ℤ)
    · rw [if_neg hnk, if_pos hnnegk]
      have h1 : ((k : ℤ) - n : ℤ) ≠ 0 := by
        have : (k : ℤ) ≥ 1 := by exact_mod_cast hk
        omega
      have h2 : (-((k : ℤ) + n) : ℤ) = 0 := by omega
      rw [if_neg h1, if_pos h2]
      have hpi : (Real.pi : ℝ) > 0 := Real.pi_pos
      have hpi_ne : (2 * Real.pi : ℝ) ≠ 0 := by linarith
      have h_smul : (1 / (2 * Real.pi) : ℝ) •
          ((-Complex.I / 2 : ℂ) * (0 - ((2 * Real.pi : ℝ) : ℂ))) =
          (Complex.I / 2 : ℂ) := by
        rw [show ((0 : ℂ) - ((2 * Real.pi : ℝ) : ℂ)) = -((2 * Real.pi : ℝ) : ℂ) from by ring]
        rw [Complex.real_smul]
        push_cast
        field_simp
      exact h_smul
    · rw [if_neg hnk, if_neg hnnegk]
      have h1 : ((k : ℤ) - n : ℤ) ≠ 0 := by
        intro h; apply hnk; omega
      have h2 : (-((k : ℤ) + n) : ℤ) ≠ 0 := by
        intro h; apply hnnegk; omega
      rw [if_neg h1, if_neg h2]
      have h_smul : (1 / (2 * Real.pi) : ℝ) • ((-Complex.I / 2 : ℂ) * (0 - 0)) = (0 : ℂ) := by
        rw [show ((0 : ℂ) - 0) = (0 : ℂ) from by ring]
        rw [show ((-Complex.I / 2 : ℂ) * 0) = (0 : ℂ) from by ring]
        rw [Complex.real_smul]
        ring
      exact h_smul



/-- `periodicHilbert` is additive in its argument (pointwise) for `C²`
2π-periodic real functions.

Proved as a theorem from the concrete Fourier-series definition. The
hypothesis `ContDiff ℝ 2 f, ContDiff ℝ 2 g, Function.Periodic f (2π),
Function.Periodic g (2π)` is needed to invoke
`summable_hilbertMult_fourierCoeff_exp` so that we can split the
multiplier tsum via `Summable.tsum_add`. Every input the certified chain
feeds to `periodicHilbert_add` is `C²` periodic (in particular,
`cos(k·) + sin(k·)`), so this hypothesis is not restrictive. -/
theorem periodicHilbert_add (f g : ℝ → ℝ)
    (hf : ContDiff ℝ 2 f) (hg : ContDiff ℝ 2 g)
    (hf_per : Function.Periodic f (2 * Real.pi))
    (hg_per : Function.Periodic g (2 * Real.pi)) :
    periodicHilbert (fun u => f u + g u) =
      fun u => periodicHilbert f u + periodicHilbert g u := by
  funext u
  -- Step 1: pointwise additivity of the Fourier coefficients.
  have hf_cont : Continuous f := hf.continuous
  have hg_cont : Continuous g := hg.continuous
  have hfC_cont : Continuous (gCircOfReal f) :=
    Complex.continuous_ofReal.comp hf_cont
  have hgC_cont : Continuous (gCircOfReal g) :=
    Complex.continuous_ofReal.comp hg_cont
  have h_coeff : ∀ n : ℤ,
      fourierCoeffOn zero_lt_two_pi (gCircOfReal (fun v => f v + g v)) n =
        fourierCoeffOn zero_lt_two_pi (gCircOfReal f) n +
          fourierCoeffOn zero_lt_two_pi (gCircOfReal g) n := by
    intro n
    rw [fourierCoeffOn_eq_integral, fourierCoeffOn_eq_integral,
        fourierCoeffOn_eq_integral]
    -- Rewrite the integrand using gCircOfReal_add and distribute the smul.
    have h_distrib :
        (fun x : ℝ =>
            fourier (-n) ((x : ℝ) : AddCircle (2 * Real.pi - 0)) •
              gCircOfReal (fun v => f v + g v) x) =
        (fun x : ℝ =>
            fourier (-n) ((x : ℝ) : AddCircle (2 * Real.pi - 0)) •
              gCircOfReal f x +
            fourier (-n) ((x : ℝ) : AddCircle (2 * Real.pi - 0)) •
              gCircOfReal g x) := by
      funext x
      have hgcirc : gCircOfReal (fun v => f v + g v) x =
          gCircOfReal f x + gCircOfReal g x := by
        simp [gCircOfReal, Complex.ofReal_add]
      rw [hgcirc, smul_add]
    rw [h_distrib]
    have hfourier_cont : Continuous
        (fun x : ℝ => fourier (-n) ((x : ℝ) : AddCircle (2 * Real.pi - 0))) :=
      (map_continuous (fourier (-n))).comp (AddCircle.continuous_mk' _)
    have hf_int :
        IntervalIntegrable
          (fun x : ℝ =>
            fourier (-n) ((x : ℝ) : AddCircle (2 * Real.pi - 0)) •
              gCircOfReal f x) volume 0 (2 * Real.pi) :=
      (hfourier_cont.smul hfC_cont).intervalIntegrable _ _
    have hg_int :
        IntervalIntegrable
          (fun x : ℝ =>
            fourier (-n) ((x : ℝ) : AddCircle (2 * Real.pi - 0)) •
              gCircOfReal g x) volume 0 (2 * Real.pi) :=
      (hfourier_cont.smul hgC_cont).intervalIntegrable _ _
    rw [intervalIntegral.integral_add hf_int hg_int]
    rw [smul_add]
  -- Step 2: split the tsum.
  set termF : ℤ → ℂ := fun n =>
    hilbertMult n *
      fourierCoeffOn zero_lt_two_pi (gCircOfReal f) n *
      Complex.exp ((n : ℂ) * Complex.I * (u : ℂ)) with htermF_def
  set termG : ℤ → ℂ := fun n =>
    hilbertMult n *
      fourierCoeffOn zero_lt_two_pi (gCircOfReal g) n *
      Complex.exp ((n : ℂ) * Complex.I * (u : ℂ)) with htermG_def
  set termFG : ℤ → ℂ := fun n =>
    hilbertMult n *
      fourierCoeffOn zero_lt_two_pi (gCircOfReal (fun v => f v + g v)) n *
      Complex.exp ((n : ℂ) * Complex.I * (u : ℂ)) with htermFG_def
  have h_termFG : ∀ n, termFG n = termF n + termG n := by
    intro n
    simp only [htermFG_def, htermF_def, htermG_def]
    rw [h_coeff n]
    ring
  -- Summability of termF and termG (from C² + periodicity).
  have hsumF : Summable termF :=
    summable_hilbertMult_fourierCoeff_exp f hf hf_per u
  have hsumG : Summable termG :=
    summable_hilbertMult_fourierCoeff_exp g hg hg_per u
  -- Conclude via tsum_congr + Summable.tsum_add.
  change Complex.re (∑' n, termFG n) =
      Complex.re (∑' n, termF n) + Complex.re (∑' n, termG n)
  have h_eq : (∑' n, termFG n) = (∑' n, termF n) + (∑' n, termG n) := by
    rw [← Summable.tsum_add hsumF hsumG]
    exact tsum_congr h_termFG
  rw [h_eq]
  exact Complex.add_re _ _

-- ===========================================================================
-- periodicHilbert_cos and periodicHilbert_sin via Fourier-series
-- ===========================================================================

/-- `periodicHilbert (cos(k·)) = sin(k·)` for integer `k ≥ 1`.

Proved as a theorem from the concrete Fourier-series definition:
the Fourier coefficients of `cos(k·)` are `1/2` at `n = ±k` and `0` elsewhere
(see `fourierCoeffOn_cos_real`), so the Hilbert-multiplier tsum collapses to
two terms `hilbertMult k · (1/2) · e^{iku} + hilbertMult (-k) · (1/2) · e^{-iku}
= -i/2 · e^{iku} + i/2 · e^{-iku} = sin(ku)`. -/
theorem periodicHilbert_cos (k : ℕ) (hk : 1 ≤ k) :
    periodicHilbert (fun u => Real.cos ((k : ℝ) * u)) =
      fun u => Real.sin ((k : ℝ) * u) := by
  funext u
  -- Reduce to the tsum on the LHS.
  change Complex.re
        (∑' n : ℤ, hilbertMult n *
          fourierCoeffOn zero_lt_two_pi
            (gCircOfReal (fun v => Real.cos ((k : ℝ) * v))) n *
          Complex.exp ((n : ℂ) * Complex.I * (u : ℂ)))
      = Real.sin ((k : ℝ) * u)
  -- Use fourierCoeffOn_cos_real: c_n = 1/2 at n = ±k, 0 elsewhere.
  -- Then the summand is non-zero only at n = ±k.
  set f : ℤ → ℂ := fun n =>
    hilbertMult n *
      fourierCoeffOn zero_lt_two_pi
        (gCircOfReal (fun v => Real.cos ((k : ℝ) * v))) n *
      Complex.exp ((n : ℂ) * Complex.I * (u : ℂ)) with hf_def
  -- Reduce: f n = 0 outside {k, -k} (using c_n = 0 there).
  have hf_zero_outside : ∀ n : ℤ, n ∉ ({(k : ℤ), -(k : ℤ)} : Finset ℤ) → f n = 0 := by
    intro n hn
    simp only [hf_def]
    simp only [Finset.mem_insert, Finset.mem_singleton] at hn
    push_neg at hn
    obtain ⟨hnk, hnnegk⟩ := hn
    rw [fourierCoeffOn_cos_real k hk n, if_neg hnk, if_neg hnnegk]
    ring
  -- tsum reduces to the finite sum.
  have h_tsum : (∑' n : ℤ, f n) = ∑ n ∈ ({(k : ℤ), -(k : ℤ)} : Finset ℤ), f n := by
    apply tsum_eq_sum (s := ({(k : ℤ), -(k : ℤ)} : Finset ℤ))
    exact fun b hb => hf_zero_outside b hb
  rw [h_tsum]
  -- Evaluate the finite sum.
  have hk1 : (k : ℤ) ≥ 1 := by exact_mod_cast hk
  have hk_ne_neg : (k : ℤ) ≠ -(k : ℤ) := by omega
  rw [show ({(k : ℤ), -(k : ℤ)} : Finset ℤ) = insert (k : ℤ) {-(k : ℤ)} from rfl]
  rw [Finset.sum_insert (by simp [hk_ne_neg])]
  rw [Finset.sum_singleton]
  -- f k = hilbertMult k * (1/2) * exp(iku), f (-k) = hilbertMult (-k) * (1/2) * exp(-iku).
  have h_fk :
      f (k : ℤ) = (-Complex.I / 2) * Complex.exp (((k : ℤ) : ℂ) * Complex.I * (u : ℂ)) := by
    simp only [hf_def]
    rw [fourierCoeffOn_cos_real k hk (k : ℤ), if_pos rfl]
    have hk_pos : (k : ℤ) > 0 := by omega
    have hk_ne : (k : ℤ) ≠ 0 := by omega
    have hhk : hilbertMult (k : ℤ) = -Complex.I := by
      unfold hilbertMult
      rw [if_neg hk_ne]
      rw [Int.sign_eq_one_iff_pos.mpr hk_pos]
      push_cast
      ring
    rw [hhk]
    ring
  have h_fnk :
      f (-(k : ℤ)) = (Complex.I / 2) * Complex.exp ((-(k : ℤ) : ℂ) * Complex.I * (u : ℂ)) := by
    simp only [hf_def]
    rw [fourierCoeffOn_cos_real k hk (-(k : ℤ))]
    have hne_kk : (-(k : ℤ)) ≠ (k : ℤ) := by omega
    rw [if_neg hne_kk, if_pos rfl]
    have hnk_neg : (-(k : ℤ)) < 0 := by omega
    have hne0 : (-(k : ℤ)) ≠ 0 := by omega
    have hhk : hilbertMult (-(k : ℤ)) = Complex.I := by
      unfold hilbertMult
      rw [if_neg hne0]
      rw [Int.sign_eq_neg_one_iff_neg.mpr hnk_neg]
      push_cast
      ring
    rw [hhk]
    have h_exp_eq :
        Complex.exp ((((-(k : ℤ)) : ℤ) : ℂ) * Complex.I * (u : ℂ)) =
        Complex.exp (((-(k : ℤ) : ℂ)) * Complex.I * (u : ℂ)) := by
      congr 1; push_cast; ring
    rw [h_exp_eq]
    ring
  rw [h_fk, h_fnk]
  -- Now: Re((-I/2) exp(iku) + (I/2) exp(-iku)) = sin(ku).
  -- (-I/2) exp(iku) + (I/2) exp(-iku) = (I/2)(exp(-iku) - exp(iku)) = sin(ku)
  --   since sin(ku) = (exp(iku) - exp(-iku)) / (2i) = -((exp(-iku) - exp(iku))/(2i))
  --     = (exp(-iku) - exp(iku)) * I / 2 (multiplying num & denom by -i: 1/(2i) = -i/2, and the
  --       numerator becomes -(...))
  -- Actually: (-I/2) exp(iku) + (I/2) exp(-iku) = (-I/2)(exp(iku) - exp(-iku))
  -- And sin(ku) = (exp(i ku) - exp(-i ku))/(2 I), so (-I/2)(exp - exp) = (-I/2) * 2I * sin(ku) = sin(ku).
  have h_sin :
      ((-Complex.I / 2) * Complex.exp (((k : ℤ) : ℂ) * Complex.I * (u : ℂ)) +
       (Complex.I / 2) * Complex.exp ((-(k : ℤ) : ℂ) * Complex.I * (u : ℂ))) =
      (Real.sin ((k : ℝ) * u) : ℂ) := by
    rw [Complex.ofReal_sin]
    have h_arg_eq : (((k : ℝ) * u : ℝ) : ℂ) = ((k : ℤ) : ℂ) * (u : ℂ) := by
      push_cast; ring
    rw [h_arg_eq, Complex.sin]
    have h_eq1 :
        Complex.exp ((((k : ℤ) : ℂ) * (u : ℂ)) * Complex.I) =
        Complex.exp (((k : ℤ) : ℂ) * Complex.I * (u : ℂ)) := by
      congr 1; ring
    have h_eq2 :
        Complex.exp ((-(((k : ℤ) : ℂ) * (u : ℂ))) * Complex.I) =
        Complex.exp ((-(k : ℤ) : ℂ) * Complex.I * (u : ℂ)) := by
      congr 1; ring
    rw [h_eq1, h_eq2]
    have h2 : (2 : ℂ) ≠ 0 := by norm_num
    have hI : Complex.I ≠ 0 := Complex.I_ne_zero
    field_simp
    ring
  rw [h_sin]
  -- Re of a real is the real.
  rw [Complex.ofReal_re]

/-- `periodicHilbert (sin(k·)) = -cos(k·)` for integer `k ≥ 1`.

Proved as a theorem from the concrete Fourier-series definition:
the Fourier coefficients of `sin(k·)` are `-i/2` at `n = k`, `i/2` at `n = -k`,
and `0` elsewhere (see `fourierCoeffOn_sin_real`), so the Hilbert-multiplier
tsum collapses to two terms `(-i)(-i/2) e^{iku} + (i)(i/2) e^{-iku}
= -1/2 e^{iku} - 1/2 e^{-iku} = -cos(ku)`. -/
theorem periodicHilbert_sin (k : ℕ) (hk : 1 ≤ k) :
    periodicHilbert (fun u => Real.sin ((k : ℝ) * u)) =
      fun u => -Real.cos ((k : ℝ) * u) := by
  funext u
  change Complex.re
        (∑' n : ℤ, hilbertMult n *
          fourierCoeffOn zero_lt_two_pi
            (gCircOfReal (fun v => Real.sin ((k : ℝ) * v))) n *
          Complex.exp ((n : ℂ) * Complex.I * (u : ℂ)))
      = -Real.cos ((k : ℝ) * u)
  set f : ℤ → ℂ := fun n =>
    hilbertMult n *
      fourierCoeffOn zero_lt_two_pi
        (gCircOfReal (fun v => Real.sin ((k : ℝ) * v))) n *
      Complex.exp ((n : ℂ) * Complex.I * (u : ℂ)) with hf_def
  have hf_zero_outside : ∀ n : ℤ, n ∉ ({(k : ℤ), -(k : ℤ)} : Finset ℤ) → f n = 0 := by
    intro n hn
    simp only [hf_def]
    simp only [Finset.mem_insert, Finset.mem_singleton] at hn
    push_neg at hn
    obtain ⟨hnk, hnnegk⟩ := hn
    rw [fourierCoeffOn_sin_real k hk n, if_neg hnk, if_neg hnnegk]
    ring
  have h_tsum : (∑' n : ℤ, f n) = ∑ n ∈ ({(k : ℤ), -(k : ℤ)} : Finset ℤ), f n := by
    apply tsum_eq_sum (s := ({(k : ℤ), -(k : ℤ)} : Finset ℤ))
    exact fun b hb => hf_zero_outside b hb
  rw [h_tsum]
  have hk1 : (k : ℤ) ≥ 1 := by exact_mod_cast hk
  have hk_ne_neg : (k : ℤ) ≠ -(k : ℤ) := by omega
  rw [show ({(k : ℤ), -(k : ℤ)} : Finset ℤ) = insert (k : ℤ) {-(k : ℤ)} from rfl]
  rw [Finset.sum_insert (by simp [hk_ne_neg])]
  rw [Finset.sum_singleton]
  -- f k = (-I)(-I/2) exp(iku) = (I²/2) exp(iku) = (-1/2) exp(iku).
  -- f (-k) = (I)(I/2) exp(-iku) = (I²/2) exp(-iku) = (-1/2) exp(-iku).
  have h_fk :
      f (k : ℤ) = (-1/2 : ℂ) * Complex.exp (((k : ℤ) : ℂ) * Complex.I * (u : ℂ)) := by
    simp only [hf_def]
    rw [fourierCoeffOn_sin_real k hk (k : ℤ), if_pos rfl]
    have hk_pos : (k : ℤ) > 0 := by omega
    have hk_ne : (k : ℤ) ≠ 0 := by omega
    have hhk : hilbertMult (k : ℤ) = -Complex.I := by
      unfold hilbertMult
      rw [if_neg hk_ne]
      rw [Int.sign_eq_one_iff_pos.mpr hk_pos]
      push_cast
      ring
    rw [hhk]
    have hI : Complex.I ^ 2 = -1 := Complex.I_sq
    -- (-I) * (-I/2) = I²/2 = -1/2.
    have hprod : (-Complex.I) * (-Complex.I / 2) = (-1/2 : ℂ) := by
      have h_sq : (-Complex.I) * (-Complex.I) = Complex.I * Complex.I := by ring
      rw [show (-Complex.I) * (-Complex.I / 2) = (-Complex.I * -Complex.I) / 2 from by ring]
      rw [h_sq, ← sq, hI]
    rw [show (-Complex.I) * (-Complex.I / 2) *
            Complex.exp (((k : ℤ) : ℂ) * Complex.I * (u : ℂ)) =
            ((-Complex.I) * (-Complex.I / 2)) *
              Complex.exp (((k : ℤ) : ℂ) * Complex.I * (u : ℂ)) from by ring]
    rw [hprod]
  have h_fnk :
      f (-(k : ℤ)) = (-1/2 : ℂ) * Complex.exp ((-(k : ℤ) : ℂ) * Complex.I * (u : ℂ)) := by
    simp only [hf_def]
    rw [fourierCoeffOn_sin_real k hk (-(k : ℤ))]
    have hne_kk : (-(k : ℤ)) ≠ (k : ℤ) := by omega
    rw [if_neg hne_kk, if_pos rfl]
    have hnk_neg : (-(k : ℤ)) < 0 := by omega
    have hne0 : (-(k : ℤ)) ≠ 0 := by omega
    have hhk : hilbertMult (-(k : ℤ)) = Complex.I := by
      unfold hilbertMult
      rw [if_neg hne0]
      rw [Int.sign_eq_neg_one_iff_neg.mpr hnk_neg]
      push_cast
      ring
    rw [hhk]
    have hI : Complex.I ^ 2 = -1 := Complex.I_sq
    have hprod : Complex.I * (Complex.I / 2) = (-1/2 : ℂ) := by
      rw [show Complex.I * (Complex.I / 2) = (Complex.I * Complex.I) / 2 from by ring]
      rw [← sq, hI]
    have h_exp_eq :
        Complex.exp ((((-(k : ℤ)) : ℤ) : ℂ) * Complex.I * (u : ℂ)) =
        Complex.exp (((-(k : ℤ) : ℂ)) * Complex.I * (u : ℂ)) := by
      congr 1; push_cast; ring
    rw [h_exp_eq]
    rw [show Complex.I * (Complex.I / 2) *
            Complex.exp ((-(k : ℤ) : ℂ) * Complex.I * (u : ℂ)) =
            (Complex.I * (Complex.I / 2)) *
              Complex.exp ((-(k : ℤ) : ℂ) * Complex.I * (u : ℂ)) from by ring]
    rw [hprod]
  rw [h_fk, h_fnk]
  have h_cos :
      ((-1/2 : ℂ) * Complex.exp (((k : ℤ) : ℂ) * Complex.I * (u : ℂ)) +
       (-1/2 : ℂ) * Complex.exp ((-(k : ℤ) : ℂ) * Complex.I * (u : ℂ))) =
      ((-Real.cos ((k : ℝ) * u) : ℝ) : ℂ) := by
    rw [show ((-Real.cos ((k : ℝ) * u) : ℝ) : ℂ) = -((Real.cos ((k : ℝ) * u) : ℝ) : ℂ) from by
        push_cast; ring]
    rw [Complex.ofReal_cos]
    have h_arg_eq : (((k : ℝ) * u : ℝ) : ℂ) = ((k : ℤ) : ℂ) * (u : ℂ) := by
      push_cast; ring
    rw [h_arg_eq, Complex.cos]
    have h_eq1 :
        Complex.exp ((((k : ℤ) : ℂ) * (u : ℂ)) * Complex.I) =
        Complex.exp (((k : ℤ) : ℂ) * Complex.I * (u : ℂ)) := by
      congr 1; ring
    have h_eq2 :
        Complex.exp ((-(((k : ℤ) : ℂ) * (u : ℂ))) * Complex.I) =
        Complex.exp ((-(k : ℤ) : ℂ) * Complex.I * (u : ℂ)) := by
      congr 1; ring
    rw [h_eq1, h_eq2]
    ring
  rw [h_cos]
  rw [Complex.ofReal_re]

-- ===========================================================================
-- periodicHilbert commutes with deriv on C³ periodic functions
-- ===========================================================================

/-- **`periodicHilbert` commutes with `deriv` on C³ periodic functions.**

The proof differentiates the Fourier multiplier series
`∑' n, hilbertMult n · c_n(g) · exp(inu)` term-by-term using
`hasDerivAt_tsum`.  The bound on the term-wise derivatives is
`‖d/du term n‖ ≤ |n| · ‖c_n(g)‖ ≤ ‖c_n(deriv g)‖ ≤ F₂(deriv g) / n²`, where
the middle step is one IBP and the last is two IBPs.  Both IBPs need
periodicity, so two derivative levels of `g` must be periodic — this needs
`ContDiff ℝ 3 g`.  Once the term-by-term differentiation is justified, the
IBP identity `c_n(deriv g) = (in) · c_n(g)` rearranges the differentiated
sum back into the multiplier series for `deriv g`. -/
theorem periodicHilbert_commutes_deriv (g : ℝ → ℝ)
    (hg : ContDiff ℝ 3 g) (hg_per : Function.Periodic g (2 * Real.pi)) :
    deriv (periodicHilbert g) = periodicHilbert (deriv g) := by
  funext u
  -- C¹/C²/C³ shorthand for g.
  have hg1 : ContDiff ℝ 1 g := hg.of_le (by norm_num)
  have hg2 : ContDiff ℝ 2 g := hg.of_le (by norm_num)
  -- deriv g is C² (from g being C³), and periodic with the same period.
  have hg'_cd2 : ContDiff ℝ 2 (deriv g) := hg.deriv'
  have hg'_per : Function.Periodic (deriv g) (2 * Real.pi) :=
    Periodic.deriv_periodic hg_per
  -- Define the term-wise summand `term n : ℝ → ℂ` and its derivative.
  let cG : ℤ → ℂ := fun n => fourierCoeffOn zero_lt_two_pi (gCircOfReal g) n
  let cG' : ℤ → ℂ := fun n => fourierCoeffOn zero_lt_two_pi (gCircOfReal (deriv g)) n
  let term : ℤ → ℝ → ℂ := fun n u =>
    hilbertMult n * cG n * Complex.exp ((n : ℂ) * Complex.I * (u : ℂ))
  let dterm : ℤ → ℝ → ℂ := fun n u =>
    hilbertMult n * cG n * ((n : ℂ) * Complex.I) *
      Complex.exp ((n : ℂ) * Complex.I * (u : ℂ))
  -- Step 1: term-wise `HasDerivAt`.
  have h_term_deriv : ∀ n (y : ℝ), HasDerivAt (term n) (dterm n y) y := by
    intro n y
    -- The derivative of u ↦ exp((n*I) * u) (viewed via ofReal) is (n*I) · exp((n*I) u).
    have hExp : HasDerivAt (fun u : ℝ =>
        Complex.exp ((n : ℂ) * Complex.I * (u : ℂ))) (((n : ℂ) * Complex.I) *
          Complex.exp ((n : ℂ) * Complex.I * (y : ℂ))) y := by
      -- exp(c*z) has derivative c*exp(c*z); restrict to ℝ via comp_ofReal.
      have hinner : HasDerivAt (fun z : ℂ => (n : ℂ) * Complex.I * z)
          ((n : ℂ) * Complex.I) (y : ℂ) := by
        simpa using (hasDerivAt_id ((y : ℝ) : ℂ)).const_mul ((n : ℂ) * Complex.I)
      have hcomp :
          HasDerivAt (fun z : ℂ => Complex.exp ((n : ℂ) * Complex.I * z))
            (Complex.exp ((n : ℂ) * Complex.I * (y : ℂ)) * ((n : ℂ) * Complex.I))
            (y : ℂ) :=
        (Complex.hasDerivAt_exp ((n : ℂ) * Complex.I * (y : ℂ))).comp _ hinner
      have h1 : HasDerivAt (fun z : ℂ => Complex.exp ((n : ℂ) * Complex.I * z))
          (((n : ℂ) * Complex.I) *
            Complex.exp ((n : ℂ) * Complex.I * (y : ℂ))) (y : ℂ) := by
        convert hcomp using 1
        ring
      exact h1.comp_ofReal
    -- term n u = (hilbertMult n * cG n) * exp((n*I) u); differentiate the constant factor in.
    have := hExp.const_mul (hilbertMult n * cG n)
    -- Massage to match `dterm n y`.
    convert this using 1
    simp [dterm]
    ring
  -- Step 2: uniform bound on dterm.
  -- Choose F₂ as a global sup of |iteratedDeriv 2 (deriv g)| = |g'''|.
  have hg'''_cont : Continuous (iteratedDeriv 2 (deriv g)) :=
    hg'_cd2.continuous_iteratedDeriv 2 (by decide)
  have hg'''_per : Function.Periodic (iteratedDeriv 2 (deriv g)) (2 * Real.pi) := by
    -- iteratedDeriv 2 (deriv g) = deriv (deriv (deriv g))
    have h_iter2 : iteratedDeriv 2 (deriv g) = deriv (deriv (deriv g)) := by
      rw [iteratedDeriv_succ, iteratedDeriv_one]
    rw [h_iter2]
    exact Periodic.deriv_periodic (Periodic.deriv_periodic hg'_per)
  have hcompact : IsCompact (Set.Icc (0 : ℝ) (2 * Real.pi)) := isCompact_Icc
  have hne : (Set.Icc (0 : ℝ) (2 * Real.pi)).Nonempty :=
    Set.nonempty_Icc.mpr (by linarith [Real.pi_pos])
  obtain ⟨x₀, _hx₀_mem, hx₀_max⟩ :=
    hcompact.exists_isMaxOn hne (hg'''_cont.abs.continuousOn)
  let F : ℝ := |iteratedDeriv 2 (deriv g) x₀|
  have hF_bd : ∀ x, |iteratedDeriv 2 (deriv g) x| ≤ F := by
    intro x
    have hpos : 0 < 2 * Real.pi := zero_lt_two_pi
    obtain ⟨y, hy_mem, hu_eq⟩ := hg'''_per.exists_mem_Ico₀ hpos x
    rw [hu_eq]
    have hmem : y ∈ Set.Icc (0 : ℝ) (2 * Real.pi) :=
      Set.mem_Icc.mpr ⟨hy_mem.1, hy_mem.2.le⟩
    exact hx₀_max hmem
  -- Bound: ‖dterm n y‖ ≤ v n where v n = (if n = 0 then 0 else F / n^2).
  let v : ℤ → ℝ := fun n => if n = 0 then 0 else F / (n : ℝ) ^ 2
  have hv_sum : Summable v := summable_F₂_div_int_sq F
  have h_dterm_bd : ∀ n (y : ℝ), ‖dterm n y‖ ≤ v n := by
    intro n y
    by_cases hn : n = 0
    · -- n = 0: hilbertMult 0 = 0, summand is 0.
      simp [dterm, v, hn, hilbertMult]
    · -- n ≠ 0: use IBP twice.  ‖dterm n y‖ = |n| · ‖c_n(g)‖ ≤ ‖c_n(g')‖ ≤ F/n².
      simp only [v, hn, if_false]
      -- Compute ‖dterm n y‖.
      have hexp_norm :
          ‖Complex.exp ((n : ℂ) * Complex.I * (y : ℂ))‖ = 1 := by
        have h : (n : ℂ) * Complex.I * (y : ℂ) =
            (((n : ℝ) * y : ℝ) : ℂ) * Complex.I := by
          push_cast; ring
        rw [h, Complex.norm_exp_ofReal_mul_I]
      have h1 : ‖dterm n y‖ =
          ‖hilbertMult n‖ * ‖cG n‖ * (|(n : ℝ)|) := by
        simp only [dterm]
        rw [norm_mul, norm_mul, norm_mul, hexp_norm, mul_one]
        rw [show ‖((n : ℂ) * Complex.I)‖ = |(n : ℝ)| by
          rw [norm_mul, Complex.norm_intCast, Complex.norm_I, mul_one]]
      rw [h1, norm_hilbertMult_of_ne_zero hn, one_mul]
      -- Now: ‖cG n‖ · |n| ≤ F / n².
      -- By IBP, ‖cG n‖ ≤ (1/|n|) · ‖cG' n‖, hence |n| · ‖cG n‖ ≤ ‖cG' n‖.
      have hibp :
          ‖cG n‖ ≤ (1 / |(n : ℝ)|) * ‖cG' n‖ :=
        fourierCoeff_norm_le_inv_n_mul_fourierCoeff_deriv g hg1 hg_per hn
      have hN_pos : (0 : ℝ) < |(n : ℝ)| := by
        have : ((n : ℝ)) ≠ 0 := by exact_mod_cast hn
        exact abs_pos.mpr this
      have h_step1 : ‖cG n‖ * |(n : ℝ)| ≤ ‖cG' n‖ := by
        have := mul_le_mul_of_nonneg_right hibp (le_of_lt hN_pos)
        rw [mul_comm (1 / |(n : ℝ)|) (‖cG' n‖), mul_assoc, one_div_mul_cancel hN_pos.ne',
            mul_one] at this
        exact this
      -- Bound ‖cG' n‖ ≤ F/n² via fourierCoeff_norm_le_F₂_div_sq applied to deriv g.
      have h_step2 : ‖cG' n‖ ≤ F / (n : ℝ) ^ 2 :=
        fourierCoeff_norm_le_F₂_div_sq (deriv g) hg'_cd2 hg'_per F hF_bd hn
      linarith
  -- Step 3: summability of term n y₀ at y₀ := 0.
  have h_term_sum0 : Summable (fun n => term n (0 : ℝ)) := by
    -- This is exactly `summable_hilbertMult_fourierCoeff_exp` at u = 0.
    have h := summable_hilbertMult_fourierCoeff_exp g hg2 hg_per 0
    -- Convert the form.
    refine h.congr ?_
    intro n
    rfl
  -- Step 4: differentiate the tsum term-by-term.
  have h_S : HasDerivAt (fun y : ℝ => ∑' n, term n y)
      (∑' n, dterm n u) u :=
    hasDerivAt_tsum (g := term) (g' := dterm) (u := v)
      hv_sum h_term_deriv h_dterm_bd h_term_sum0 u
  -- Step 5: compose with `Complex.reCLM` to get a real-valued HasDerivAt.
  have h_Re : HasDerivAt (fun y : ℝ => (∑' n, term n y).re)
      ((∑' n, dterm n u).re) u := by
    have hCLM : HasFDerivAt (fun z : ℂ => z.re) (Complex.reCLM : ℂ →L[ℝ] ℝ)
        (∑' n, term n u) := Complex.reCLM.hasFDerivAt
    have := hCLM.comp_hasDerivAt u h_S
    -- (Complex.reCLM (∑' n, dterm n u)) = (∑' n, dterm n u).re
    simpa [Complex.reCLM_apply, Function.comp] using this
  -- Step 6: deduce deriv (periodicHilbert g) u = (∑' n, dterm n u).re.
  have h_deriv_eq : deriv (periodicHilbert g) u = (∑' n, dterm n u).re := by
    have h_pH : periodicHilbert g = fun y : ℝ => (∑' n, term n y).re := rfl
    rw [h_pH]
    exact h_Re.deriv
  rw [h_deriv_eq]
  -- Step 7: identify (∑' n, dterm n u).re with periodicHilbert (deriv g) u
  -- via the IBP identity dterm n u = hilbertMult n · cG' n · exp(inu).
  have h_dterm_eq : ∀ n,
      dterm n u =
        hilbertMult n * cG' n *
          Complex.exp ((n : ℂ) * Complex.I * (u : ℂ)) := by
    intro n
    by_cases hn : n = 0
    · simp [dterm, hn, hilbertMult]
    · -- For n ≠ 0: dterm n u = hilbertMult n * cG n * (n*I) * exp(inu)
      --              = hilbertMult n * ((n*I) * cG n) * exp(inu)
      --              = hilbertMult n * cG' n * exp(inu) by IBP identity.
      simp only [dterm]
      have hibp : cG' n = ((n : ℂ) * Complex.I) * cG n :=
        fourierCoeffOn_deriv_eq g hg1 hg_per hn
      rw [hibp]
      ring
  have h_tsum_dterm :
      (∑' n, dterm n u) =
        ∑' n, hilbertMult n * cG' n *
          Complex.exp ((n : ℂ) * Complex.I * (u : ℂ)) :=
    tsum_congr h_dterm_eq
  rw [h_tsum_dterm]
  -- This is exactly periodicHilbert (deriv g) u by definition.
  rfl

end WaveCAP

-- Confirm: both N3 endpoints depend only on `propext, Classical.choice, Quot.sound`
-- (no WaveCAP-specific axioms — `periodicHilbert_le_tsum_fourierCoeff` is now a
-- theorem proved from the concrete Fourier-series definition of `periodicHilbert`).
-- `periodicHilbert_smul` and `periodicHilbert_const` were also upgraded to
-- theorems (using `fourierCoeffOn.const_smul` and the IBP formula respectively).
#print axioms WaveCAP.periodicHilbert_le_tsum_fourierCoeff
#print axioms WaveCAP.fourier_tail_hilbert_bound
#print axioms WaveCAP.periodicHilbert_smul
#print axioms WaveCAP.periodicHilbert_const
#print axioms WaveCAP.periodicHilbert_add
#print axioms WaveCAP.periodicHilbert_cos
#print axioms WaveCAP.periodicHilbert_sin
#print axioms WaveCAP.periodicHilbert_commutes_deriv
