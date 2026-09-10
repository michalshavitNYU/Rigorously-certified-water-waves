import WaveCAP.Common

/-!
# WaveCAP definitions: y, y_u, H[y_u], D, G, R (the Babenko residual)

This file defines the mathematical objects of the Babenko water-wave equation
on the torus `𝕋 = ℝ / 2πℤ`. All functions are 2π-periodic; `y` is supplied as a
finite even cosine series.

```
y(u)      = a_1 + Σ_{k=1}^{N-1} a_{k+1} cos(k u)
y_u(u)    = -Σ_{k=1}^{N-1} k a_{k+1} sin(k u)
H[y_u](u) =  Σ_{k=1}^{N-1} k a_{k+1} cos(k u)     (Path A — mode-by-mode H)
D(u)      = √(y_u(u)² + (1 + H[y_u](u))²)
G(u)      = (1 + H[y_u](u)) / D(u)
R(u)      = μ H[y_u] - y - y H[y_u] - H[y · y_u]
            + β ∂_u(y_u/D) - β H[∂_u G]
```

The Hilbert transform `periodicHilbert` is given a concrete definition here
as the real part of a Fourier-series formula
`Re ∑'_{n} m(n) · ĝ(n) · e^{i n u}` where the multiplier
`m(n) = -i · sgn n` has unit modulus for `n ≠ 0` and `m(0) = 0`.
The essential properties (linearity, action on trig modes, derivative
commutation) are all proved as theorems from this definition — see
`periodicHilbert_smul`/`periodicHilbert_const` below and
`periodicHilbert_add`/`periodicHilbert_cos`/`periodicHilbert_sin` in
`FourierTailHilbert.lean`. The bridging identity
`periodicHilbert (yu a) = Hyu a` is proved in `HilbertOnTrigPoly.lean`
(`periodicHilbert_yu_eq_Hyu`).

Cross-reference: `BabenkoRigorous/src/{FourierSeries,ResidualBound}.jl` —
the Julia residual implementation uses identical algebraic structure with
Arb interval arithmetic.
-/

namespace WaveCAP

open Real Complex MeasureTheory

-- ===========================================================================
-- Helpers (shared with FourierTailHilbert.lean)
-- ===========================================================================

/-- Cast a real-valued function to complex-valued; used to feed `g` to
`fourierCoeffOn`. -/
noncomputable def gCircOfReal (g : ℝ → ℝ) : ℝ → ℂ := fun t => (g t : ℂ)

/-- Two-pi positivity, kept as a named lemma for reuse. -/
lemma zero_lt_two_pi : (0 : ℝ) < 2 * Real.pi := by
  have h := Real.pi_pos
  linarith

/-- The Hilbert-multiplier: `m(n) = -i · sgn n` (and `m(0) = 0`).

For `n ≠ 0` we have `‖m n‖ = 1`; this is the only quantitative property of the
multiplier used in the Fourier-tail bound. -/
noncomputable def hilbertMult (n : ℤ) : ℂ :=
  if n = 0 then 0 else (-Complex.I) * (Int.sign n : ℂ)

lemma hilbertMult_zero : hilbertMult 0 = 0 := by simp [hilbertMult]

lemma norm_hilbertMult_of_ne_zero {n : ℤ} (hn : n ≠ 0) :
    ‖hilbertMult n‖ = 1 := by
  unfold hilbertMult
  simp only [hn, if_false]
  rw [norm_mul, norm_neg, Complex.norm_I, one_mul]
  -- ‖((Int.sign n : ℤ) : ℂ)‖ = 1 because sign n ∈ {-1, 1}
  rcases lt_or_gt_of_ne hn with hn_neg | hn_pos
  · rw [Int.sign_eq_neg_one_iff_neg.mpr hn_neg]
    simp
  · rw [Int.sign_eq_one_iff_pos.mpr hn_pos]
    simp

lemma norm_hilbertMult_le_one (n : ℤ) : ‖hilbertMult n‖ ≤ 1 := by
  by_cases hn : n = 0
  · simp [hilbertMult, hn]
  · exact (norm_hilbertMult_of_ne_zero hn).le

-- ===========================================================================
-- Periodic Hilbert transform
-- ===========================================================================

/-- The 2π-periodic Hilbert transform.

Defined as the real part of the Fourier-series multiplier formula
```
periodicHilbert g (u) = Re ∑'_{n : ℤ} (-i · sgn n) · ĝ(n) · e^{i n u}
```
which on a real-valued `C²` 2π-periodic
`g(u) = ∑ (a_k cos(ku) + b_k sin(ku))` simplifies to
`(periodicHilbert g)(u) = ∑ (a_k sin(ku) - b_k cos(ku))` (so `H[cos(ku)] = sin(ku)`
and `H[sin(ku)] = -cos(ku)` for k ≥ 1, and `H[1] = 0`).

For non-summable Fourier coefficients (i.e. `g` not `C²` enough) the
`tsum` returns `0` by Mathlib convention; that is fine for our use case (we
only invoke `periodicHilbert` on `C²` periodic inputs in the certified
chain). -/
noncomputable def periodicHilbert (g : ℝ → ℝ) : ℝ → ℝ := fun u =>
  Complex.re
    (∑' n : ℤ,
      hilbertMult n *
        fourierCoeffOn zero_lt_two_pi (gCircOfReal g) n *
        Complex.exp ((n : ℂ) * Complex.I * (u : ℂ)))

-- `periodicHilbert_add`, `periodicHilbert_cos`, `periodicHilbert_sin` are
-- proved as theorems in `FourierTailHilbert.lean` (which has access to the
-- summability lemma `summable_hilbertMult_fourierCoeff_exp` needed for the
-- additivity proof, and to `fourierCoeffOn_of_hasDerivAt` for the trig-mode
-- computations).

/-- `periodicHilbert` is `ℝ`-linear in its argument (pointwise scalar
multiplication).

Proved as a theorem from the concrete Fourier-series definition using
`fourierCoeffOn.const_smul` (which is *unconditional*: non-integrable inputs
have both sides equal to `0`). -/
theorem periodicHilbert_smul (c : ℝ) (f : ℝ → ℝ) :
    periodicHilbert (fun u => c * f u) = fun u => c * periodicHilbert f u := by
  funext u
  -- Show gCircOfReal (fun u => c * f u) = (c : ℂ) • gCircOfReal f.
  have h_gcirc :
      (gCircOfReal (fun v => c * f v) : ℝ → ℂ) = (c : ℂ) • (gCircOfReal f) := by
    funext v
    simp only [gCircOfReal, Pi.smul_apply, smul_eq_mul, Complex.ofReal_mul]
  -- Hence the Fourier coefficients factor: c_n(c · f) = c · c_n(f).
  have h_coeff : ∀ n : ℤ,
      fourierCoeffOn zero_lt_two_pi (gCircOfReal (fun v => c * f v)) n =
        (c : ℂ) * fourierCoeffOn zero_lt_two_pi (gCircOfReal f) n := by
    intro n
    rw [h_gcirc]
    rw [fourierCoeffOn.const_smul (gCircOfReal f) (c : ℂ) n zero_lt_two_pi]
    rw [smul_eq_mul]
  -- Plug into the periodicHilbert formula.
  change Complex.re
        (∑' n : ℤ, hilbertMult n *
          fourierCoeffOn zero_lt_two_pi (gCircOfReal (fun v => c * f v)) n *
          Complex.exp ((n : ℂ) * Complex.I * (u : ℂ)))
      = c * Complex.re
        (∑' n : ℤ, hilbertMult n *
          fourierCoeffOn zero_lt_two_pi (gCircOfReal f) n *
          Complex.exp ((n : ℂ) * Complex.I * (u : ℂ)))
  -- Factor (c : ℂ) out of the tsum.
  have h_eq : (∑' n : ℤ, hilbertMult n *
        fourierCoeffOn zero_lt_two_pi (gCircOfReal (fun v => c * f v)) n *
        Complex.exp ((n : ℂ) * Complex.I * (u : ℂ)))
      = (c : ℂ) *
        ∑' n : ℤ, hilbertMult n *
          fourierCoeffOn zero_lt_two_pi (gCircOfReal f) n *
          Complex.exp ((n : ℂ) * Complex.I * (u : ℂ)) := by
    rw [← tsum_mul_left]
    refine tsum_congr ?_
    intro n
    rw [h_coeff n]
    ring
  rw [h_eq]
  -- Re((c : ℂ) * z) = c * Re z because c is real.
  exact Complex.re_ofReal_mul c _

/-- `periodicHilbert` of any constant function is zero.

Proved as a theorem from the concrete Fourier-series definition: for every
`n ≠ 0`, `fourierCoeffOn h (fun _ => (c : ℂ)) n = 0` by the IBP formula
`fourierCoeffOn_of_hasDerivAt` applied with `f' = 0` (a constant function
has zero derivative, and `f(2π) - f(0) = 0`). For `n = 0` the multiplier
`hilbertMult 0 = 0`. Hence every term of the Hilbert tsum is `0`, the tsum
is `0`, and `Re 0 = 0`. -/
theorem periodicHilbert_const (c : ℝ) :
    periodicHilbert (fun _ => c) = fun _ => 0 := by
  funext u
  -- Show every term of the Fourier-multiplier tsum is zero.
  have h_term : ∀ n : ℤ,
      hilbertMult n *
        fourierCoeffOn zero_lt_two_pi (gCircOfReal (fun _ : ℝ => c)) n *
        Complex.exp ((n : ℂ) * Complex.I * (u : ℂ)) = 0 := by
    intro n
    by_cases hn : n = 0
    · -- At n = 0, hilbertMult 0 = 0.
      simp [hilbertMult, hn]
    · -- At n ≠ 0, the Fourier coefficient of a constant is zero (via IBP with f' = 0).
      have h_coeff :
          fourierCoeffOn zero_lt_two_pi (gCircOfReal (fun _ : ℝ => c)) n = 0 := by
        -- Apply fourierCoeffOn_of_hasDerivAt with f = const, f' = 0.
        have hderiv : ∀ x : ℝ, x ∈ Set.uIcc (0 : ℝ) (2 * Real.pi) →
            HasDerivAt (gCircOfReal (fun _ : ℝ => c)) 0 x := by
          intro x _
          -- gCircOfReal (fun _ => c) = fun _ => (c : ℂ), a constant function.
          have : (gCircOfReal (fun _ : ℝ => c)) = fun _ : ℝ => (c : ℂ) := rfl
          rw [this]
          exact hasDerivAt_const x ((c : ℂ))
        have hint : IntervalIntegrable (fun _ : ℝ => (0 : ℂ)) volume 0 (2 * Real.pi) :=
          intervalIntegrable_const
        have hIBP := fourierCoeffOn_of_hasDerivAt zero_lt_two_pi hn hderiv hint
        -- Substitute into the IBP formula.
        -- f(2π) = f(0) = c, so f(2π) - f(0) = 0.
        -- fourierCoeffOn (fun _ => 0) n = 0 by direct integral.
        have h_boundary : (gCircOfReal (fun _ : ℝ => c)) (2 * Real.pi) -
            (gCircOfReal (fun _ : ℝ => c)) 0 = 0 := by
          simp [gCircOfReal]
        have h_zero_coeff :
            fourierCoeffOn zero_lt_two_pi (fun _ : ℝ => (0 : ℂ)) n = 0 := by
          rw [fourierCoeffOn_eq_integral]
          simp
        rw [hIBP, h_boundary, h_zero_coeff]
        ring
      rw [h_coeff]
      ring
  -- Hence the entire tsum is zero (all terms zero ⇒ tsum is zero).
  change Complex.re
      (∑' n : ℤ, hilbertMult n *
          fourierCoeffOn zero_lt_two_pi (gCircOfReal (fun _ : ℝ => c)) n *
          Complex.exp ((n : ℂ) * Complex.I * (u : ℂ))) = 0
  have h_tsum : (∑' n : ℤ, hilbertMult n *
        fourierCoeffOn zero_lt_two_pi (gCircOfReal (fun _ : ℝ => c)) n *
        Complex.exp ((n : ℂ) * Complex.I * (u : ℂ))) = 0 := by
    rw [show (0 : ℂ) = ∑' _ : ℤ, (0 : ℂ) from tsum_zero.symm]
    exact tsum_congr h_term
  rw [h_tsum]
  simp

-- `periodicHilbert_commutes_deriv` is proved in `FourierTailHilbert.lean`
-- (as a theorem from the concrete Fourier-series definition).  It is
-- re-exported in `HilbertCommutesDeriv.lean`.

-- ===========================================================================
-- Babenko-side definitions: y, y_u, D, G, R
-- ===========================================================================

variable {N : ℕ} (a : Fin N → ℝ) (μ β : ℝ)

/-- The water-wave profile `y(u)` as a finite even cosine series:
`y(u) = a 0 + Σ_{k=1}^{N-1} a k · cos(k u)`. -/
noncomputable def y (u : ℝ) : ℝ :=
  ∑ k : Fin N, a k * Real.cos ((k : ℝ) * u)

/-- The derivative `y_u(u) = -Σ_{k=1}^{N-1} k · a k · sin(k u)`. -/
noncomputable def yu (u : ℝ) : ℝ :=
  -∑ k : Fin N, (k : ℝ) * a k * Real.sin ((k : ℝ) * u)

/-- The Hilbert-transformed derivative `H[y_u](u) = Σ k · a k · cos(k u)`
(Path A — explicit mode-by-mode action of `periodicHilbert`). -/
noncomputable def Hyu (u : ℝ) : ℝ :=
  ∑ k : Fin N, (k : ℝ) * a k * Real.cos ((k : ℝ) * u)

/-- The Babenko denominator `D(u) = √(y_u² + (1 + H[y_u])²)`. -/
noncomputable def D (u : ℝ) : ℝ :=
  Real.sqrt ((yu a u) ^ 2 + (1 + Hyu a u) ^ 2)

/-- The auxiliary `G(u) = (1 + H[y_u]) / D(u)`. -/
noncomputable def G (u : ℝ) : ℝ :=
  (1 + Hyu a u) / D a u

/-- The Babenko residual:
`R(u) = μ H[y_u] − y − y · H[y_u] − H[y · y_u] + β ∂_u(y_u/D) − β H[∂_u G]`. -/
noncomputable def R (u : ℝ) : ℝ :=
  μ * Hyu a u
    - y a u
    - y a u * Hyu a u
    - periodicHilbert (fun v => y a v * yu a v) u
    + β * deriv (fun v => yu a v / D a v) u
    - β * periodicHilbert (deriv (G a)) u

end WaveCAP
