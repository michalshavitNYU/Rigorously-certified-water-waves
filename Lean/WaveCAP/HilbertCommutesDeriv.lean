import WaveCAP.HilbertOnTrigPoly

/-!
# `H` commutes with `∂_u` on `C³` periodic functions

The mode-by-mode action of `periodicHilbert` (cosine ↔ sine swap) makes
`H ∘ ∂_u = ∂_u ∘ H` formally clear on trig polynomials.  This file extends
that to general `C³` 2π-periodic functions via term-by-term differentiation
of the multiplier Fourier series — see `periodicHilbert_commutes_deriv`
in `FourierTailHilbert.lean`.

It also derives the iterated form `H ∘ ∂_u^n = ∂_u^n ∘ H` (used implicitly
by `fft_hilbert_series` when it iterates over derivative levels).

Cross-reference: `BabenkoRigorous/src/FourierSeries.jl` +
`BabenkoRigorous/src/FFTHilbert.jl`.
-/

namespace WaveCAP

open Real

/-- Re-export: `H` commutes with `deriv` on `C³` periodic functions. -/
theorem deriv_periodicHilbert (g : ℝ → ℝ)
    (hg : ContDiff ℝ 3 g) (hg_per : Function.Periodic g (2 * Real.pi)) :
    deriv (periodicHilbert g) = periodicHilbert (deriv g) :=
  periodicHilbert_commutes_deriv g hg hg_per

/-- `iteratedDeriv n g` is `(2π)`-periodic when `g` is. -/
lemma Periodic.iteratedDeriv_periodic {g : ℝ → ℝ} {T : ℝ}
    (hg_per : Function.Periodic g T) (n : ℕ) :
    Function.Periodic (iteratedDeriv n g) T := by
  induction n with
  | zero => simpa [iteratedDeriv_zero] using hg_per
  | succ n ih =>
    rw [iteratedDeriv_succ]
    exact WaveCAP.Periodic.deriv_periodic ih

/-- Iterated form: `H ∘ ∂_u^n = ∂_u^n ∘ H` for `C^{n+3}` periodic `g`.

This is the form used by the Julia `fft_hilbert_series` routine when it
applies `H` derivative-level-by-derivative-level: the `m`-th `sup_bound` of
`H[G]` is bounded via the `m+2`-th `sup_bound` of `G` (one Fourier-tail step
per derivative level).

This historical form pays `+3` for backwards compatibility with downstream
call sites (`FFTHilbertCorrectness`). The tightened `+2` version,
`iteratedDeriv_periodicHilbert_tight`, is below and is what
`hilbert_contDiff` actually uses.

Proof by induction on `n` using the single-step
`periodicHilbert_commutes_deriv` applied at each level. -/
theorem iteratedDeriv_periodicHilbert (n : ℕ) (g : ℝ → ℝ)
    (hg : ContDiff ℝ (n + 3) g) (hg_per : Function.Periodic g (2 * Real.pi)) :
    iteratedDeriv n (periodicHilbert g) = periodicHilbert (iteratedDeriv n g) := by
  induction n with
  | zero => simp [iteratedDeriv_zero]
  | succ n ih =>
    -- IH needs ContDiff ℝ (n + 3) g.  Have ContDiff ℝ ((n + 1) + 3) g.
    have hg_ih : ContDiff ℝ (n + 3) g := by
      apply hg.of_le
      have hle : (n + 3 : ℕ) ≤ ((n + 1) + 3 : ℕ) := by omega
      have h1 : ((n + 3 : ℕ) : ℕ∞) ≤ (((n + 1) + 3 : ℕ) : ℕ∞) :=
        ENat.coe_le_coe.mpr hle
      have h2 : (((n + 3 : ℕ) : ℕ∞) : WithTop ℕ∞) ≤ ((((n + 1) + 3 : ℕ) : ℕ∞) : WithTop ℕ∞) :=
        WithTop.coe_le_coe.mpr h1
      simpa using h2
    have ih_eq := ih hg_ih
    -- iteratedDeriv (n+1) (periodicHilbert g) = deriv (iteratedDeriv n (periodicHilbert g))
    rw [iteratedDeriv_succ]
    rw [ih_eq]
    -- Now: deriv (periodicHilbert (iteratedDeriv n g)) = periodicHilbert (iteratedDeriv (n+1) g)
    -- Need: ContDiff ℝ 3 (iteratedDeriv n g) and periodicity.
    have h_iter_cd : ContDiff ℝ 3 (iteratedDeriv n g) := by
      have h_eq : iteratedDeriv n g = (deriv^[n]) g := iteratedDeriv_eq_iterate
      rw [h_eq]
      apply ContDiff.iterate_deriv' 3 n (f := g)
      apply hg.of_le
      have hle : (3 + n : ℕ) ≤ ((n + 1) + 3 : ℕ) := by omega
      have h1 : ((3 + n : ℕ) : ℕ∞) ≤ (((n + 1) + 3 : ℕ) : ℕ∞) :=
        ENat.coe_le_coe.mpr hle
      have h2 : (((3 + n : ℕ) : ℕ∞) : WithTop ℕ∞) ≤ ((((n + 1) + 3 : ℕ) : ℕ∞) : WithTop ℕ∞) :=
        WithTop.coe_le_coe.mpr h1
      simpa using h2
    have h_iter_per : Function.Periodic (iteratedDeriv n g) (2 * Real.pi) :=
      Periodic.iteratedDeriv_periodic hg_per n
    rw [periodicHilbert_commutes_deriv (iteratedDeriv n g) h_iter_cd h_iter_per]
    -- deriv (iteratedDeriv n g) = iteratedDeriv (n+1) g
    rw [← iteratedDeriv_succ]

/-- **Tightened iterated form**: `H ∘ ∂_u^n = ∂_u^n ∘ H` for `C^{n+2}`
periodic `g`. This is the sharp form: in the inductive proof, the IH for
level `n` needs `g ∈ C^(n+2)` (available from `g ∈ C^((n+1)+2) = C^(n+3)`),
and the single-step at the top needs `iteratedDeriv n g ∈ C^3`, i.e.,
`g ∈ C^(n+3)`. So `+2` is optimal under this strategy. Used by
`hilbert_contDiff`. -/
theorem iteratedDeriv_periodicHilbert_tight (n : ℕ) (g : ℝ → ℝ)
    (hg : ContDiff ℝ (n + 2) g) (hg_per : Function.Periodic g (2 * Real.pi)) :
    iteratedDeriv n (periodicHilbert g) = periodicHilbert (iteratedDeriv n g) := by
  induction n with
  | zero => simp [iteratedDeriv_zero]
  | succ n ih =>
    -- IH needs ContDiff ℝ (n + 2) g.  Have ContDiff ℝ ((n + 1) + 2) g.
    have hg_ih : ContDiff ℝ (n + 2) g := by
      apply hg.of_le
      have hle : (n + 2 : ℕ) ≤ ((n + 1) + 2 : ℕ) := by omega
      have h1 : ((n + 2 : ℕ) : ℕ∞) ≤ (((n + 1) + 2 : ℕ) : ℕ∞) :=
        ENat.coe_le_coe.mpr hle
      have h2 : (((n + 2 : ℕ) : ℕ∞) : WithTop ℕ∞) ≤ ((((n + 1) + 2 : ℕ) : ℕ∞) : WithTop ℕ∞) :=
        WithTop.coe_le_coe.mpr h1
      simpa using h2
    have ih_eq := ih hg_ih
    rw [iteratedDeriv_succ]
    rw [ih_eq]
    -- Single-step at the top needs `iteratedDeriv n g ∈ C^3`,
    -- i.e., `g ∈ C^(n+3)`. Available from `g ∈ C^((n+1)+2) = C^(n+3)`. Exactly tight.
    have h_iter_cd : ContDiff ℝ 3 (iteratedDeriv n g) := by
      have h_eq : iteratedDeriv n g = (deriv^[n]) g := iteratedDeriv_eq_iterate
      rw [h_eq]
      apply ContDiff.iterate_deriv' 3 n (f := g)
      apply hg.of_le
      have hle : (3 + n : ℕ) ≤ ((n + 1) + 2 : ℕ) := by omega
      have h1 : ((3 + n : ℕ) : ℕ∞) ≤ (((n + 1) + 2 : ℕ) : ℕ∞) :=
        ENat.coe_le_coe.mpr hle
      have h2 : (((3 + n : ℕ) : ℕ∞) : WithTop ℕ∞) ≤ ((((n + 1) + 2 : ℕ) : ℕ∞) : WithTop ℕ∞) :=
        WithTop.coe_le_coe.mpr h1
      simpa using h2
    have h_iter_per : Function.Periodic (iteratedDeriv n g) (2 * Real.pi) :=
      Periodic.iteratedDeriv_periodic hg_per n
    rw [periodicHilbert_commutes_deriv (iteratedDeriv n g) h_iter_cd h_iter_per]
    rw [← iteratedDeriv_succ]

end WaveCAP

#print axioms WaveCAP.deriv_periodicHilbert
#print axioms WaveCAP.iteratedDeriv_periodicHilbert
#print axioms WaveCAP.iteratedDeriv_periodicHilbert_tight
