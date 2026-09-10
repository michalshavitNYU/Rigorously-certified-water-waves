import WaveCAP.HilbertCommutesDeriv

/-!
# Sharper smoothness sub-lemmas for `periodicHilbert` via Cauchy–Schwarz

This file provides the `+1`-loss sub-lemmas used by `hilbert_contDiff`:

* `periodicHilbert_continuous_of_contDiff_one` — `Hg` continuous from `g ∈ C¹`.
* `periodicHilbert_differentiable_of_contDiff_two` — `Hg` differentiable from
  `g ∈ C²`, with derivative the multiplier series.
* `iteratedDeriv_periodicHilbert_super_tight` — `iteratedDeriv n (Hg) = H(D^n g)`
  from `g ∈ C^(n+1)`.

The original versions in `HilbertSmoothness.lean` paid `+2`/`+3`/`+3` because
the dominating bound `|n|^m · ‖c_n(g)‖` was estimated as `F_{m+2}/n²`, which
needs IBP twice on `D^m g`, i.e., `D^m g ∈ C²`, i.e., `g ∈ C^(m+2)`.

The sharpening uses Cauchy–Schwarz on Fourier-coefficient L²-sums:

  Σ_{n ≠ 0} |n|^k · ‖c_n(g)‖
    ≤ √(Σ |n|^{2(k+1)} ‖c_n(g)‖²) · √(Σ_{n ≠ 0} 1/n²)
    = √(Σ ‖c_n(D^{k+1} g)‖²) · √(π²/3)
    = √(‖D^{k+1} g‖²_{L²} / (2π)) · √(π²/3)  (Parseval)

which is finite as soon as `D^{k+1} g ∈ L²(0, 2π]`, i.e., `g ∈ C^{k+1}`
(continuous periodic ⇒ bounded ⇒ L²). This is one fewer derivative than
the `g ∈ C^{k+2}` that absolute-convergence-via-`F_{k+2}/n²` would need.

The trade is that we lose explicit control of the sup-bound constant —
the Cauchy–Schwarz bound is in terms of the integral L²-norm rather than
a sup-norm — but the qualitative `Summable` statement is what
`continuous_tsum` and `hasDerivAt_tsum` consume, so this is exactly
sufficient for our purposes.
-/

namespace WaveCAP

open Real Complex MeasureTheory

/-! ## Parseval bridge -/

/-- For a continuous function `g : ℝ → ℝ`, the complex cast `gCircOfReal g`
is in `L²` on `(0, 2π]` (a continuous function on a compact-measure set is
trivially `L²`). -/
lemma gCircOfReal_memLp_two_of_continuous {g : ℝ → ℝ} (hg_cont : Continuous g) :
    MemLp (gCircOfReal g) 2 (volume.restrict (Set.Ioc 0 (2 * Real.pi))) := by
  haveI : Fact (volume (Set.Ioc 0 (2 * Real.pi)) < ⊤) := by
    refine ⟨?_⟩
    rw [Real.volume_Ioc]; exact ENNReal.ofReal_lt_top
  -- Pull a uniform sup-bound from compactness of `[0, 2π]`.
  obtain ⟨C, hC⟩ : ∃ C, ∀ t ∈ Set.Icc (0 : ℝ) (2 * Real.pi), |g t| ≤ C := by
    have hcompact : IsCompact (Set.Icc (0 : ℝ) (2 * Real.pi)) := isCompact_Icc
    have hne : (Set.Icc (0 : ℝ) (2 * Real.pi)).Nonempty :=
      Set.nonempty_Icc.mpr (by linarith [Real.pi_pos])
    obtain ⟨x₀, _, hx₀_max⟩ := hcompact.exists_isMaxOn hne hg_cont.abs.continuousOn
    exact ⟨|g x₀|, fun t ht => hx₀_max ht⟩
  have hg_complex : Continuous (gCircOfReal g) := Complex.continuous_ofReal.comp hg_cont
  apply MemLp.of_bound hg_complex.aestronglyMeasurable C
  filter_upwards [ae_restrict_mem measurableSet_Ioc] with t ht
  unfold gCircOfReal
  rw [Complex.norm_real]
  exact hC t ⟨le_of_lt ht.1, ht.2⟩

/-- **Parseval** for periodic continuous `g`: the squared Fourier coefficients
of `g` (viewed via `gCircOfReal`) are summable over `ℤ`. -/
lemma fourierCoeff_sq_summable_of_continuous {g : ℝ → ℝ} (hg_cont : Continuous g) :
    Summable (fun n : ℤ => ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal g) n‖ ^ 2) :=
  (hasSum_sq_fourierCoeffOn zero_lt_two_pi
    (gCircOfReal_memLp_two_of_continuous hg_cont)).summable

/-! ## Iterated IBP for Fourier coefficients -/

/-- **Iterated IBP**: For `g ∈ C^k` 2π-periodic and `n ≠ 0`,
`c_n(D^k g) = (in)^k · c_n(g)`. -/
lemma fourierCoeffOn_iteratedDeriv_eq (g : ℝ → ℝ) (k : ℕ)
    (hg : ContDiff ℝ k g) (hg_per : Function.Periodic g (2 * Real.pi))
    {n : ℤ} (hn : n ≠ 0) :
    fourierCoeffOn zero_lt_two_pi (gCircOfReal (iteratedDeriv k g)) n
      = ((n : ℂ) * Complex.I) ^ k *
        fourierCoeffOn zero_lt_two_pi (gCircOfReal g) n := by
  induction k with
  | zero => simp [iteratedDeriv_zero]
  | succ k ih =>
    rw [iteratedDeriv_succ]
    -- For the single IBP step we need `iteratedDeriv k g ∈ C¹`.
    have hg_iter_cd1 : ContDiff ℝ 1 (iteratedDeriv k g) := by
      have h_eq : iteratedDeriv k g = (deriv^[k]) g := iteratedDeriv_eq_iterate
      rw [h_eq]
      apply ContDiff.iterate_deriv' 1 k (f := g)
      apply hg.of_le
      have hle : (1 + k : ℕ) ≤ ((k + 1) : ℕ) := by omega
      have h1 : ((1 + k : ℕ) : ℕ∞) ≤ ((k + 1 : ℕ) : ℕ∞) := ENat.coe_le_coe.mpr hle
      have h2 : (((1 + k : ℕ) : ℕ∞) : WithTop ℕ∞) ≤ (((k + 1 : ℕ) : ℕ∞) : WithTop ℕ∞) :=
        WithTop.coe_le_coe.mpr h1
      simpa using h2
    have hg_iter_per : Function.Periodic (iteratedDeriv k g) (2 * Real.pi) :=
      WaveCAP.Periodic.iteratedDeriv_periodic hg_per k
    rw [fourierCoeffOn_deriv_eq (iteratedDeriv k g) hg_iter_cd1 hg_iter_per hn]
    -- IH at level `k` needs `g ∈ C^k`.
    have hg_ih : ContDiff ℝ k g := by
      apply hg.of_le
      have hle : (k : ℕ) ≤ ((k + 1) : ℕ) := Nat.le_succ k
      have h1 : ((k : ℕ) : ℕ∞) ≤ ((k + 1 : ℕ) : ℕ∞) := ENat.coe_le_coe.mpr hle
      have h2 : (((k : ℕ) : ℕ∞) : WithTop ℕ∞) ≤ (((k + 1 : ℕ) : ℕ∞) : WithTop ℕ∞) :=
        WithTop.coe_le_coe.mpr h1
      simpa using h2
    rw [ih hg_ih]
    ring

/-! ## L² Fourier-coefficient bound for `g ∈ C^p` -/

/-- For `g ∈ C^p` periodic, `Σ_n |n|^{2p} ‖c_n(g)‖²` is summable. -/
lemma sq_pow_mul_fourierCoeff_summable {g : ℝ → ℝ} (p : ℕ)
    (hg : ContDiff ℝ p g) (hg_per : Function.Periodic g (2 * Real.pi)) :
    Summable (fun n : ℤ => |(n : ℝ)| ^ (2 * p) *
      ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal g) n‖ ^ 2) := by
  have hg_deriv_p_cont : Continuous (iteratedDeriv p g) := by
    apply hg.continuous_iteratedDeriv p
    exact le_refl _
  have h_par := fourierCoeff_sq_summable_of_continuous hg_deriv_p_cont
  refine h_par.of_nonneg_of_le ?_ ?_
  · intro n; positivity
  · intro n
    by_cases hn : n = 0
    · simp only [hn, Int.cast_zero, abs_zero]
      by_cases hp : p = 0
      · simp [hp]
      · rw [zero_pow (by omega), zero_mul]
        positivity
    · have h_ibp := fourierCoeffOn_iteratedDeriv_eq g p hg hg_per hn
      rw [h_ibp, norm_mul, norm_pow, norm_mul, Complex.norm_intCast, Complex.norm_I, mul_one,
          mul_pow]
      rw [show |(n : ℝ)| ^ (2 * p) = (|(n : ℝ)| ^ p) ^ 2 by rw [mul_comm, pow_mul]]

/-! ## Cauchy–Schwarz: `Σ |n|^k ‖c_n‖` summable from `g ∈ C^(k+1)` -/

/-- **Cauchy–Schwarz sharpening**: For `g ∈ C^(k+1)` periodic,
`Σ_{n ≠ 0} |n|^k · ‖c_n(g)‖` is summable. This is the key lemma that drops
the regularity threshold by one derivative compared to the F₂/n² absolute
bound. -/
lemma summable_pow_mul_fourierCoeff_of_contDiff_succ {g : ℝ → ℝ} (k : ℕ)
    (hg : ContDiff ℝ (k + 1) g) (hg_per : Function.Periodic g (2 * Real.pi)) :
    Summable (fun n : ℤ => if n = 0 then 0 else
        |(n : ℝ)| ^ k * ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal g) n‖) := by
  -- The Parseval L²-bound at level k+1.
  have h_sq := sq_pow_mul_fourierCoeff_summable (k + 1) hg hg_per
  -- Setup for Hölder/C-S.
  set a : ℤ → ℝ := fun n => ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal g) n‖ with ha_def
  have ha_nn : ∀ n, 0 ≤ a n := fun n => norm_nonneg _
  set f : ℤ → ℝ := fun n => |(n : ℝ)| ^ (k + 1) * a n with hf_def
  set g_seq : ℤ → ℝ := fun n => if n = 0 then 0 else 1 / |(n : ℝ)| with hg_seq_def
  have hpq : (2 : ℝ).HolderConjugate 2 := ⟨by norm_num, by norm_num, by norm_num⟩
  have hf_nn : ∀ n, 0 ≤ f n := fun n => mul_nonneg (by positivity) (ha_nn n)
  have hg_seq_nn : ∀ n, 0 ≤ g_seq n := by
    intro n
    by_cases hn : n = 0
    · simp [g_seq, hn]
    · simp only [g_seq, if_neg hn]; positivity
  -- `f ^ (2 : ℕ)` summable directly from h_sq, then transport to `^ (2 : ℝ)`.
  have hf_sq_sum_nat : Summable (fun n => f n ^ (2 : ℕ)) := by
    refine h_sq.congr (fun n => ?_)
    simp only [f, mul_pow]
    rw [show |(n : ℝ)| ^ (2 * (k + 1)) = (|(n : ℝ)| ^ (k + 1)) ^ 2 by
      rw [mul_comm, pow_mul]]
  have hf_sq_sum : Summable (fun n => f n ^ (2 : ℝ)) := by
    refine hf_sq_sum_nat.congr (fun n => ?_)
    rw [show ((2 : ℝ)) = ((2 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
  -- `g_seq ^ (2 : ℝ)` summable: it is the indicator `1/n²`.
  have hg_seq_sq_sum : Summable (fun n => g_seq n ^ (2 : ℝ)) := by
    refine summable_indicator_one_div_int_sq.congr (fun n => ?_)
    rw [show ((2 : ℝ)) = ((2 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
    by_cases hn : n = 0
    · simp [g_seq, hn]
    · simp only [g_seq, if_neg hn]
      rw [div_pow, one_pow]
      rw [show |(n : ℝ)| ^ 2 = (n : ℝ) ^ 2 from sq_abs _]
  -- Hölder (p = q = 2) gives summability of f · g_seq.
  have h_main : Summable (fun n => f n * g_seq n) :=
    summable_mul_of_Lp_Lq_of_nonneg hpq hf_nn hg_seq_nn hf_sq_sum hg_seq_sq_sum
  -- Identify f · g_seq with the desired sequence.
  refine h_main.congr (fun n => ?_)
  by_cases hn : n = 0
  · simp [f, g_seq, hn]
  · simp only [f, g_seq, if_neg hn]
    have h_abs_pos : (0 : ℝ) < |(n : ℝ)| := by
      have : (n : ℝ) ≠ 0 := by exact_mod_cast hn
      exact abs_pos.mpr this
    have h_pow : |(n : ℝ)| ^ (k + 1) = |(n : ℝ)| ^ k * |(n : ℝ)| := by ring
    rw [h_pow, show a n = ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal g) n‖ from rfl]
    field_simp

/-! ## Sharper sub-lemmas for `hilbert_contDiff` -/

/-- **Sharper summability**: For `g ∈ C¹` periodic, the multiplier Fourier
series `n ↦ hilbertMult n · c_n(g) · exp(inu)` is absolutely summable.

This is the C¹ analogue of `summable_hilbertMult_fourierCoeff_exp` (which
needed `C²`). The improvement comes from `summable_pow_mul_fourierCoeff_of_contDiff_succ`
at `k = 0`, which gives `Σ_{n ≠ 0} ‖c_n(g)‖` summable directly from `g ∈ C¹`. -/
lemma summable_hilbertMult_fourierCoeff_exp_sharper {g : ℝ → ℝ}
    (hg : ContDiff ℝ 1 g) (hg_per : Function.Periodic g (2 * Real.pi)) (u : ℝ) :
    Summable (fun n : ℤ =>
      hilbertMult n *
        fourierCoeffOn zero_lt_two_pi (gCircOfReal g) n *
        Complex.exp ((n : ℂ) * Complex.I * (u : ℂ))) := by
  apply Summable.of_norm
  -- Bound by `if n = 0 then 0 else ‖c_n(g)‖`, summable by Cauchy–Schwarz at k = 0.
  have h_dom : Summable (fun n : ℤ => if n = 0 then 0 else
      |(n : ℝ)| ^ (0 : ℕ) * ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal g) n‖) :=
    summable_pow_mul_fourierCoeff_of_contDiff_succ 0 hg hg_per
  have h_dom' : Summable (fun n : ℤ => if n = 0 then (0 : ℝ) else
      ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal g) n‖) := by
    refine h_dom.congr (fun n => ?_)
    by_cases hn : n = 0 <;> simp [hn]
  refine h_dom'.of_nonneg_of_le ?_ ?_
  · intro n; exact norm_nonneg _
  · intro n
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

/-- **Continuity sub-lemma (sharper).** For `g ∈ C¹(ℝ)` 2π-periodic, the
periodic Hilbert transform `periodicHilbert g` is continuous.

Proof: `continuous_tsum` applied to the multiplier Fourier series. The
uniform bound `‖summand n u‖ ≤ ‖c_n(g)‖` (independent of `u`) is summable
by `summable_pow_mul_fourierCoeff_of_contDiff_succ` at `k = 0`. -/
theorem periodicHilbert_continuous_of_contDiff_one
    {g : ℝ → ℝ} (hg : ContDiff ℝ 1 g)
    (hg_per : Function.Periodic g (2 * Real.pi)) :
    Continuous (periodicHilbert g) := by
  let cG : ℤ → ℂ := fun n => fourierCoeffOn zero_lt_two_pi (gCircOfReal g) n
  let term : ℤ → ℝ → ℂ := fun n u =>
    hilbertMult n * cG n * Complex.exp ((n : ℂ) * Complex.I * (u : ℂ))
  have h_term_cont : ∀ n, Continuous (term n) := by
    intro n
    have h_exp : Continuous (fun u : ℝ => Complex.exp ((n : ℂ) * Complex.I * (u : ℂ))) := by
      refine Complex.continuous_exp.comp ?_
      exact (continuous_const.mul Complex.continuous_ofReal)
    exact (continuous_const.mul h_exp)
  -- Dominating sequence via Cauchy–Schwarz at k = 0.
  let v : ℤ → ℝ := fun n => if n = 0 then 0 else ‖cG n‖
  have hv_sum : Summable v := by
    have h_dom : Summable (fun n : ℤ => if n = 0 then 0 else
        |(n : ℝ)| ^ (0 : ℕ) * ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal g) n‖) :=
      summable_pow_mul_fourierCoeff_of_contDiff_succ 0 hg hg_per
    refine h_dom.congr (fun n => ?_)
    by_cases hn : n = 0 <;> simp [v, cG, hn]
  -- Pointwise bound.
  have h_term_bd : ∀ n (u : ℝ), ‖term n u‖ ≤ v n := by
    intro n u
    by_cases hn : n = 0
    · simp [term, v, hn, hilbertMult]
    · simp only [v, hn, if_false]
      have hexp_norm :
          ‖Complex.exp ((n : ℂ) * Complex.I * (u : ℂ))‖ = 1 := by
        have h : (n : ℂ) * Complex.I * (u : ℂ) =
            (((n : ℝ) * u : ℝ) : ℂ) * Complex.I := by
          push_cast; ring
        rw [h, Complex.norm_exp_ofReal_mul_I]
      simp only [term]
      rw [norm_mul, norm_mul, hexp_norm, mul_one,
          norm_hilbertMult_of_ne_zero hn, one_mul]
  have h_tsum_cont : Continuous (fun u : ℝ => ∑' n : ℤ, term n u) :=
    continuous_tsum h_term_cont hv_sum h_term_bd
  have h_pH : periodicHilbert g = fun u => (∑' n, term n u).re := rfl
  rw [h_pH]
  exact Complex.continuous_re.comp h_tsum_cont

/-- **Sharper single-step commute.** For `g ∈ C²(ℝ)` 2π-periodic,
`deriv (H g) = H (deriv g)`.

The original `periodicHilbert_commutes_deriv` needed `g ∈ C³` because it
bounded `|n| · ‖c_n(g)‖` as `F/n²` (two IBPs on `deriv g`). Here we use
`summable_pow_mul_fourierCoeff_of_contDiff_succ` at `k = 1`, which gives
the `|n| · ‖c_n(g)‖` summability directly from `g ∈ C²` via Cauchy–Schwarz. -/
theorem periodicHilbert_commutes_deriv_sharper (g : ℝ → ℝ)
    (hg : ContDiff ℝ 2 g) (hg_per : Function.Periodic g (2 * Real.pi)) :
    deriv (periodicHilbert g) = periodicHilbert (deriv g) := by
  funext u
  have hg1 : ContDiff ℝ 1 g := hg.of_le (by norm_num)
  have hg'_cd1 : ContDiff ℝ 1 (deriv g) := hg.deriv'
  have hg'_per : Function.Periodic (deriv g) (2 * Real.pi) :=
    WaveCAP.Periodic.deriv_periodic hg_per
  let cG : ℤ → ℂ := fun n => fourierCoeffOn zero_lt_two_pi (gCircOfReal g) n
  let cG' : ℤ → ℂ := fun n => fourierCoeffOn zero_lt_two_pi (gCircOfReal (deriv g)) n
  let term : ℤ → ℝ → ℂ := fun n u =>
    hilbertMult n * cG n * Complex.exp ((n : ℂ) * Complex.I * (u : ℂ))
  let dterm : ℤ → ℝ → ℂ := fun n u =>
    hilbertMult n * cG n * ((n : ℂ) * Complex.I) *
      Complex.exp ((n : ℂ) * Complex.I * (u : ℂ))
  -- Step 1: term-wise HasDerivAt.
  have h_term_deriv : ∀ n (y : ℝ), HasDerivAt (term n) (dterm n y) y := by
    intro n y
    have hExp : HasDerivAt (fun u : ℝ =>
        Complex.exp ((n : ℂ) * Complex.I * (u : ℂ))) (((n : ℂ) * Complex.I) *
          Complex.exp ((n : ℂ) * Complex.I * (y : ℂ))) y := by
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
    have := hExp.const_mul (hilbertMult n * cG n)
    convert this using 1
    simp [dterm]
    ring
  -- Step 2: uniform bound on dterm via Cauchy–Schwarz at k = 1.
  let v : ℤ → ℝ := fun n => if n = 0 then 0 else |(n : ℝ)| * ‖cG n‖
  have hv_sum : Summable v := by
    have h_dom := summable_pow_mul_fourierCoeff_of_contDiff_succ 1 hg hg_per
    refine h_dom.congr (fun n => ?_)
    by_cases hn : n = 0
    · simp [v, hn]
    · simp only [v, cG, hn, if_false]
      ring
  have h_dterm_bd : ∀ n (y : ℝ), ‖dterm n y‖ ≤ v n := by
    intro n y
    by_cases hn : n = 0
    · simp [dterm, v, hn, hilbertMult]
    · simp only [v, hn, if_false]
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
      rw [h1, norm_hilbertMult_of_ne_zero hn, one_mul, mul_comm]
  -- Step 3: summability of term n 0.
  have h_term_sum0 : Summable (fun n => term n (0 : ℝ)) := by
    have h := summable_hilbertMult_fourierCoeff_exp_sharper hg1 hg_per 0
    refine h.congr ?_
    intro n
    rfl
  -- Step 4: term-by-term differentiation.
  have h_S : HasDerivAt (fun y : ℝ => ∑' n, term n y)
      (∑' n, dterm n u) u :=
    hasDerivAt_tsum (g := term) (g' := dterm) (u := v)
      hv_sum h_term_deriv h_dterm_bd h_term_sum0 u
  -- Step 5: compose with Complex.reCLM.
  have h_Re : HasDerivAt (fun y : ℝ => (∑' n, term n y).re)
      ((∑' n, dterm n u).re) u := by
    have hCLM : HasFDerivAt (fun z : ℂ => z.re) (Complex.reCLM : ℂ →L[ℝ] ℝ)
        (∑' n, term n u) := Complex.reCLM.hasFDerivAt
    have := hCLM.comp_hasDerivAt u h_S
    simpa [Complex.reCLM_apply, Function.comp] using this
  -- Step 6: deduce deriv (H g) u = (∑' n, dterm n u).re.
  have h_deriv_eq : deriv (periodicHilbert g) u = (∑' n, dterm n u).re := by
    have h_pH : periodicHilbert g = fun y : ℝ => (∑' n, term n y).re := rfl
    rw [h_pH]
    exact h_Re.deriv
  rw [h_deriv_eq]
  -- Step 7: identify dterm n u with the multiplier series of `deriv g`.
  have h_dterm_eq : ∀ n,
      dterm n u =
        hilbertMult n * cG' n *
          Complex.exp ((n : ℂ) * Complex.I * (u : ℂ)) := by
    intro n
    by_cases hn : n = 0
    · simp [dterm, hn, hilbertMult]
    · simp only [dterm]
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
  rfl

/-- **Super-tightened iterated form**: `H ∘ ∂_u^n = ∂_u^n ∘ H` for
`C^{n+1}` periodic `g`. Improves the `+2` of
`iteratedDeriv_periodicHilbert_tight` to `+1` by using the sharper
single-step commute `periodicHilbert_commutes_deriv_sharper`. -/
theorem iteratedDeriv_periodicHilbert_super_tight (n : ℕ) (g : ℝ → ℝ)
    (hg : ContDiff ℝ (n + 1) g) (hg_per : Function.Periodic g (2 * Real.pi)) :
    iteratedDeriv n (periodicHilbert g) = periodicHilbert (iteratedDeriv n g) := by
  induction n with
  | zero => simp [iteratedDeriv_zero]
  | succ n ih =>
    -- IH needs ContDiff ℝ (n + 1) g.  Have ContDiff ℝ ((n + 1) + 1) g.
    have hg_ih : ContDiff ℝ (n + 1) g := by
      apply hg.of_le
      have hle : (n + 1 : ℕ) ≤ ((n + 1) + 1 : ℕ) := by omega
      have h1 : ((n + 1 : ℕ) : ℕ∞) ≤ (((n + 1) + 1 : ℕ) : ℕ∞) :=
        ENat.coe_le_coe.mpr hle
      have h2 : (((n + 1 : ℕ) : ℕ∞) : WithTop ℕ∞) ≤ ((((n + 1) + 1 : ℕ) : ℕ∞) : WithTop ℕ∞) :=
        WithTop.coe_le_coe.mpr h1
      simpa using h2
    have ih_eq := ih hg_ih
    rw [iteratedDeriv_succ]
    rw [ih_eq]
    -- Single-step at the top needs `iteratedDeriv n g ∈ C²`,
    -- i.e., `g ∈ C^(n+2)`. Available from `g ∈ C^((n+1)+1) = C^(n+2)`. Exactly tight.
    have h_iter_cd : ContDiff ℝ 2 (iteratedDeriv n g) := by
      have h_eq : iteratedDeriv n g = (deriv^[n]) g := iteratedDeriv_eq_iterate
      rw [h_eq]
      apply ContDiff.iterate_deriv' 2 n (f := g)
      apply hg.of_le
      have hle : (2 + n : ℕ) ≤ ((n + 1) + 1 : ℕ) := by omega
      have h1 : ((2 + n : ℕ) : ℕ∞) ≤ (((n + 1) + 1 : ℕ) : ℕ∞) :=
        ENat.coe_le_coe.mpr hle
      have h2 : (((2 + n : ℕ) : ℕ∞) : WithTop ℕ∞) ≤ ((((n + 1) + 1 : ℕ) : ℕ∞) : WithTop ℕ∞) :=
        WithTop.coe_le_coe.mpr h1
      simpa using h2
    have h_iter_per : Function.Periodic (iteratedDeriv n g) (2 * Real.pi) :=
      WaveCAP.Periodic.iteratedDeriv_periodic hg_per n
    rw [periodicHilbert_commutes_deriv_sharper (iteratedDeriv n g) h_iter_cd h_iter_per]
    rw [← iteratedDeriv_succ]

/-- **Differentiability sub-lemma (sharper).** For `g ∈ C²(ℝ)` 2π-periodic,
`periodicHilbert g` is differentiable.

Proof: as a corollary of `periodicHilbert_commutes_deriv_sharper`, whose
underlying `HasDerivAt` we re-extract. We bypass having to recreate the
proof by simply noting that `periodicHilbert_commutes_deriv_sharper`
implies the existence of a derivative at every point. -/
theorem periodicHilbert_differentiable_of_contDiff_two
    {g : ℝ → ℝ} (hg : ContDiff ℝ 2 g)
    (hg_per : Function.Periodic g (2 * Real.pi)) :
    Differentiable ℝ (periodicHilbert g) := by
  intro u
  have hg1 : ContDiff ℝ 1 g := hg.of_le (by norm_num)
  have hg'_cd1 : ContDiff ℝ 1 (deriv g) := hg.deriv'
  have hg'_per : Function.Periodic (deriv g) (2 * Real.pi) :=
    WaveCAP.Periodic.deriv_periodic hg_per
  -- Recreate the HasDerivAt from periodicHilbert_commutes_deriv_sharper.
  let cG : ℤ → ℂ := fun n => fourierCoeffOn zero_lt_two_pi (gCircOfReal g) n
  let term : ℤ → ℝ → ℂ := fun n u =>
    hilbertMult n * cG n * Complex.exp ((n : ℂ) * Complex.I * (u : ℂ))
  let dterm : ℤ → ℝ → ℂ := fun n u =>
    hilbertMult n * cG n * ((n : ℂ) * Complex.I) *
      Complex.exp ((n : ℂ) * Complex.I * (u : ℂ))
  -- Step 1: term-wise HasDerivAt (copied from periodicHilbert_commutes_deriv_sharper).
  have h_term_deriv : ∀ n (y : ℝ), HasDerivAt (term n) (dterm n y) y := by
    intro n y
    have hExp : HasDerivAt (fun u : ℝ =>
        Complex.exp ((n : ℂ) * Complex.I * (u : ℂ))) (((n : ℂ) * Complex.I) *
          Complex.exp ((n : ℂ) * Complex.I * (y : ℂ))) y := by
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
    have := hExp.const_mul (hilbertMult n * cG n)
    convert this using 1
    simp [dterm]
    ring
  -- Step 2: uniform bound on dterm via Cauchy–Schwarz at k = 1.
  let v : ℤ → ℝ := fun n => if n = 0 then 0 else |(n : ℝ)| * ‖cG n‖
  have hv_sum : Summable v := by
    have h_dom := summable_pow_mul_fourierCoeff_of_contDiff_succ 1 hg hg_per
    refine h_dom.congr (fun n => ?_)
    by_cases hn : n = 0
    · simp [v, hn]
    · simp only [v, cG, hn, if_false]
      ring
  have h_dterm_bd : ∀ n (y : ℝ), ‖dterm n y‖ ≤ v n := by
    intro n y
    by_cases hn : n = 0
    · simp [dterm, v, hn, hilbertMult]
    · simp only [v, hn, if_false]
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
      rw [h1, norm_hilbertMult_of_ne_zero hn, one_mul, mul_comm]
  -- Step 3: summability of term n 0 (from sharper C¹ summability).
  have h_term_sum0 : Summable (fun n => term n (0 : ℝ)) := by
    have h := summable_hilbertMult_fourierCoeff_exp_sharper hg1 hg_per 0
    refine h.congr ?_
    intro n
    rfl
  -- Step 4: term-by-term differentiation.
  have h_S : HasDerivAt (fun y : ℝ => ∑' n, term n y)
      (∑' n, dterm n u) u :=
    hasDerivAt_tsum (g := term) (g' := dterm) (u := v)
      hv_sum h_term_deriv h_dterm_bd h_term_sum0 u
  -- Step 5: compose with Complex.reCLM to get a real HasDerivAt.
  have h_Re : HasDerivAt (fun y : ℝ => (∑' n, term n y).re)
      ((∑' n, dterm n u).re) u := by
    have hCLM : HasFDerivAt (fun z : ℂ => z.re) (Complex.reCLM : ℂ →L[ℝ] ℝ)
        (∑' n, term n u) := Complex.reCLM.hasFDerivAt
    have := hCLM.comp_hasDerivAt u h_S
    simpa [Complex.reCLM_apply, Function.comp] using this
  -- Conclude differentiability at u.
  have h_pH : periodicHilbert g = fun y : ℝ => (∑' n, term n y).re := rfl
  rw [h_pH]
  exact h_Re.differentiableAt

end WaveCAP

#print axioms WaveCAP.gCircOfReal_memLp_two_of_continuous
#print axioms WaveCAP.fourierCoeff_sq_summable_of_continuous
#print axioms WaveCAP.fourierCoeffOn_iteratedDeriv_eq
#print axioms WaveCAP.sq_pow_mul_fourierCoeff_summable
#print axioms WaveCAP.summable_pow_mul_fourierCoeff_of_contDiff_succ
#print axioms WaveCAP.summable_hilbertMult_fourierCoeff_exp_sharper
#print axioms WaveCAP.periodicHilbert_continuous_of_contDiff_one
#print axioms WaveCAP.periodicHilbert_commutes_deriv_sharper
#print axioms WaveCAP.iteratedDeriv_periodicHilbert_super_tight
#print axioms WaveCAP.periodicHilbert_differentiable_of_contDiff_two
