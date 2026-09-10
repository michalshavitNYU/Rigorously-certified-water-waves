import WaveCAP.HilbertSharper

/-!
# `periodicHilbert` preserves `C^k` smoothness on 2π-periodic functions

This file proves `hilbert_contDiff`: if `G` is `C^(k+1)` and 2π-periodic, then
`periodicHilbert G` is `C^k`.

## Statement

```
theorem hilbert_contDiff
    {k : ℕ} {G : ℝ → ℝ}
    (hG : ContDiff ℝ (k + 1) G)
    (hG_per : Function.Periodic G (2 * Real.pi)) :
    ContDiff ℝ k (periodicHilbert G)
```

## Why `+1`?

The proof goes through the Fourier multiplier picture
`(H g)^∧(n) = -i sign(n) · ĝ(n)`. Three ingredients (all sharpened by
`HilbertSharper.lean` via the Cauchy–Schwarz Fourier-coefficient bound):

* `iteratedDeriv_periodicHilbert_super_tight n g` gives
  `iteratedDeriv m (H G) = H (iteratedDeriv m G)` under `G ∈ C^(m+1)`. The
  single-step `periodicHilbert_commutes_deriv_sharper` needs `g ∈ C²` (one
  fewer than the `C³` of `periodicHilbert_commutes_deriv`).

* For each `m < k`, we need `H (D^m G)` differentiable, which uses
  `periodicHilbert_differentiable_of_contDiff_two` applied to `D^m G ∈ C²`,
  i.e., `G ∈ C^(m+2) ⊆ C^(k+1)` (since `m + 1 ≤ k`).

* For the top level `m = k`, we need just continuity of `H (D^k G)`, which uses
  `periodicHilbert_continuous_of_contDiff_one` applied to `D^k G ∈ C¹`, i.e.,
  `G ∈ C^(k+1)`. This is the only place the top-level hypothesis bites.

The split `differentiable for m < k, continuous for m = k` is exactly what
`contDiff_nat_iff_iteratedDeriv` requires — using
`contDiff_of_differentiable_iteratedDeriv` (which demands differentiability
even at `m = k`) is what forced `+3` historically.

## Strategy

1. **Continuity sub-lemma** (in `HilbertSharper`):
   `periodicHilbert_continuous_of_contDiff_one` — for `C¹` periodic `g`,
   `periodicHilbert g` is continuous, via `continuous_tsum` with the
   Cauchy–Schwarz `Summable (n ↦ ‖c_n(g)‖)` (zero-excluded).

2. **Differentiability sub-lemma** (in `HilbertSharper`):
   `periodicHilbert_differentiable_of_contDiff_two` — for `C²` periodic
   `g`, `periodicHilbert g` is differentiable, via `hasDerivAt_tsum` with
   the Cauchy–Schwarz `Summable (n ↦ |n| · ‖c_n(g)‖)` (zero-excluded).

3. **Main theorem**. Use `contDiff_nat_iff_iteratedDeriv` to reduce
   `ContDiff k (periodicHilbert G)` to continuity of `iteratedDeriv m`
   for each `m ≤ k` and differentiability for each `m < k`. Use the
   super-tightened `iteratedDeriv_periodicHilbert_super_tight` to swap
   `iteratedDeriv m ∘ H = H ∘ iteratedDeriv m`. Apply sub-lemmas 1 and 2
   per level.
-/

namespace WaveCAP

open Real Complex

/-- Backwards-compatibility corollary: the original `C²` continuity
sub-lemma. Now a one-line consequence of the sharper `C¹` version. -/
theorem periodicHilbert_continuous_of_contDiff_two
    {g : ℝ → ℝ} (hg : ContDiff ℝ 2 g)
    (hg_per : Function.Periodic g (2 * Real.pi)) :
    Continuous (periodicHilbert g) :=
  periodicHilbert_continuous_of_contDiff_one (hg.of_le (by norm_num)) hg_per

/-- Backwards-compatibility corollary: the original `C³` differentiability
sub-lemma. Now a one-line consequence of the sharper `C²` version. -/
theorem periodicHilbert_differentiable_of_contDiff_three
    {g : ℝ → ℝ} (hg : ContDiff ℝ 3 g)
    (hg_per : Function.Periodic g (2 * Real.pi)) :
    Differentiable ℝ (periodicHilbert g) :=
  periodicHilbert_differentiable_of_contDiff_two (hg.of_le (by norm_num)) hg_per

/-- **Main theorem (smoothness preservation).**

For `G ∈ C^(k+1)(ℝ)` 2π-periodic, the periodic Hilbert transform of `G` is
in `C^k(ℝ)`.

The loss of one derivative is sharper than the historical `+2` realized by
`F_2/n²` absolute bounds. The improvement comes from Cauchy–Schwarz on
Fourier coefficients: `Σ_{n ≠ 0} |n|^k · ‖c_n(g)‖ ≤ √(Σ|n|^{2(k+1)} ‖c_n‖²)
· √(Σ 1/n²)` is finite as soon as `g ∈ C^(k+1)` (so `D^{k+1} g ∈ L²`).

Proof strategy: split the smoothness requirement into
* continuity of `iteratedDeriv m (H G)` for each `m ≤ k`, and
* differentiability of `iteratedDeriv m (H G)` for each `m < k`.

Each iterated derivative is rewritten via the (super-tightened)
`iteratedDeriv_periodicHilbert_super_tight`, which requires `G ∈ C^(m+1)` —
available from `G ∈ C^(k+1)` since `m ≤ k`. Then continuity of `H(D^m G)`
uses `periodicHilbert_continuous_of_contDiff_one` (`D^m G ∈ C¹`, available
from `G ∈ C^(m+1)`), and differentiability uses
`periodicHilbert_differentiable_of_contDiff_two` (`D^m G ∈ C²`, available
from `G ∈ C^(m+2) ⊆ C^(k+1)` since `m + 1 ≤ k`). -/
theorem hilbert_contDiff
    {k : ℕ} {G : ℝ → ℝ}
    (hG : ContDiff ℝ (k + 1 : ℕ) G)
    (hG_per : Function.Periodic G (2 * Real.pi)) :
    ContDiff ℝ k (periodicHilbert G) := by
  -- Use the natural-index version: continuity for m ≤ k AND differentiability for m < k.
  rw [contDiff_nat_iff_iteratedDeriv]
  refine ⟨?_, ?_⟩
  · -- Continuity of `iteratedDeriv m (periodicHilbert G)` for `m ≤ k`.
    intro m hm_le
    -- Rewrite via the iterated-commute identity (super-tightened to `+1`).
    have hG_for_commute : ContDiff ℝ (m + 1 : ℕ) G := by
      apply hG.of_le
      have hle : (m + 1 : ℕ) ≤ (k + 1 : ℕ) := by omega
      have h1 : ((m + 1 : ℕ) : ℕ∞) ≤ ((k + 1 : ℕ) : ℕ∞) :=
        ENat.coe_le_coe.mpr hle
      have h2 : (((m + 1 : ℕ) : ℕ∞) : WithTop ℕ∞) ≤ (((k + 1 : ℕ) : ℕ∞) : WithTop ℕ∞) :=
        WithTop.coe_le_coe.mpr h1
      simpa using h2
    have h_iter_commute :
        iteratedDeriv m (periodicHilbert G) = periodicHilbert (iteratedDeriv m G) :=
      iteratedDeriv_periodicHilbert_super_tight m G hG_for_commute hG_per
    rw [h_iter_commute]
    -- Continuity sub-lemma: `iteratedDeriv m G ∈ C¹`, available from `G ∈ C^(m+1)`.
    have hG_iter_cd1 : ContDiff ℝ 1 (iteratedDeriv m G) := by
      have h_eq : iteratedDeriv m G = (deriv^[m]) G := iteratedDeriv_eq_iterate
      rw [h_eq]
      apply ContDiff.iterate_deriv' 1 m (f := G)
      apply hG.of_le
      have hle : (1 + m : ℕ) ≤ (k + 1 : ℕ) := by omega
      have h1 : ((1 + m : ℕ) : ℕ∞) ≤ ((k + 1 : ℕ) : ℕ∞) :=
        ENat.coe_le_coe.mpr hle
      have h2 : (((1 + m : ℕ) : ℕ∞) : WithTop ℕ∞) ≤ (((k + 1 : ℕ) : ℕ∞) : WithTop ℕ∞) :=
        WithTop.coe_le_coe.mpr h1
      simpa using h2
    have hG_iter_per : Function.Periodic (iteratedDeriv m G) (2 * Real.pi) :=
      WaveCAP.Periodic.iteratedDeriv_periodic hG_per m
    exact periodicHilbert_continuous_of_contDiff_one hG_iter_cd1 hG_iter_per
  · -- Differentiability of `iteratedDeriv m (periodicHilbert G)` for `m < k`.
    intro m hm_lt
    -- From `m < k` we have `2 + m ≤ k + 1` (omega).  This is exactly what the
    -- differentiability sub-lemma needs (it requires `iteratedDeriv m G ∈ C²`).
    -- Rewrite via the iterated-commute identity (super-tightened to `+1`).
    have hG_for_commute : ContDiff ℝ (m + 1 : ℕ) G := by
      apply hG.of_le
      have hle : (m + 1 : ℕ) ≤ (k + 1 : ℕ) := by omega
      have h1 : ((m + 1 : ℕ) : ℕ∞) ≤ ((k + 1 : ℕ) : ℕ∞) :=
        ENat.coe_le_coe.mpr hle
      have h2 : (((m + 1 : ℕ) : ℕ∞) : WithTop ℕ∞) ≤ (((k + 1 : ℕ) : ℕ∞) : WithTop ℕ∞) :=
        WithTop.coe_le_coe.mpr h1
      simpa using h2
    have h_iter_commute :
        iteratedDeriv m (periodicHilbert G) = periodicHilbert (iteratedDeriv m G) :=
      iteratedDeriv_periodicHilbert_super_tight m G hG_for_commute hG_per
    rw [h_iter_commute]
    -- Differentiability sub-lemma: `iteratedDeriv m G ∈ C²`, available from
    -- `G ∈ C^(m+2) ⊆ C^(k+1)` since `m + 1 ≤ k`.
    have hG_iter_cd2 : ContDiff ℝ 2 (iteratedDeriv m G) := by
      have h_eq : iteratedDeriv m G = (deriv^[m]) G := iteratedDeriv_eq_iterate
      rw [h_eq]
      apply ContDiff.iterate_deriv' 2 m (f := G)
      apply hG.of_le
      have hle : (2 + m : ℕ) ≤ (k + 1 : ℕ) := by omega
      have h1 : ((2 + m : ℕ) : ℕ∞) ≤ ((k + 1 : ℕ) : ℕ∞) :=
        ENat.coe_le_coe.mpr hle
      have h2 : (((2 + m : ℕ) : ℕ∞) : WithTop ℕ∞) ≤ (((k + 1 : ℕ) : ℕ∞) : WithTop ℕ∞) :=
        WithTop.coe_le_coe.mpr h1
      simpa using h2
    have hG_iter_per : Function.Periodic (iteratedDeriv m G) (2 * Real.pi) :=
      WaveCAP.Periodic.iteratedDeriv_periodic hG_per m
    exact periodicHilbert_differentiable_of_contDiff_two hG_iter_cd2 hG_iter_per

end WaveCAP

#print axioms WaveCAP.periodicHilbert_continuous_of_contDiff_two
#print axioms WaveCAP.periodicHilbert_differentiable_of_contDiff_three
#print axioms WaveCAP.hilbert_contDiff
