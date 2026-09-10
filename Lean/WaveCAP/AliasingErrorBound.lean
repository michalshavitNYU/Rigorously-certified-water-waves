import WaveCAP.FourierTailHilbert

/-!
# N4 — DFT-based periodic Hilbert aliasing remainder

```
For G ∈ C^p(𝕋) 2π-periodic with ‖G^(p)‖_∞ ≤ Cp, p ≥ 2, N ≥ 4 even:
  ∀ j : Fin N,
    |dftHilbert N G (fftGridPoint N j) − periodicHilbert G (fftGridPoint N j)|
       ≤ 4 Cp / ((p−1)(N/2 − 1)^(p−1)).
```

`dftHilbert N G` is the **sampling-based DFT Hilbert transform**: it samples
`G` at the `N` grid points `u_j = 2πj/N`, takes the discrete Fourier
transform, multiplies by the discrete Hilbert multiplier (`-i` on positive
modes `1..N/2-1`, `+i` on negative modes `N/2+1..N-1`, `0` at `0` and at
the Nyquist mode `N/2`), and inverts the DFT. This matches the operator
implemented in `BabenkoRigorous/src/FFTHilbert.jl`.

It differs from `periodicHilbert G` (continuous Fourier multiplier series)
by an *aliasing* error which we control rigorously via the periodic
Poisson summation identity on the torus:
`Ĝ_k = ∑'_m c_{k + mN}(G)` where `Ĝ_k = dftCoeff N G k` is the discrete
Fourier coefficient. Combined with a per-mode multiplier-disagreement
bound of `2` for aliased modes and `0` for principal modes
`|n| < N/2`, this gives the closed-form remainder above.

## Proof outline

1. **Sampling DFT definition**: `dftHilbert N G u = Re ∑_{k ∈ Fin N}
   M_k · Ĝ_k · exp(i k u)` where `M_k = hilbertMultDiscrete N k` and
   `Ĝ_k = dftCoeff N G k`.

2. **Periodic Poisson identity** (`dftCoeff_eq_aliased_sum`): for `C²`
   2π-periodic `G`, `Ĝ_k = ∑'_m c_{k + mN}(G)`.

3. **Mode collapse at grid points**: at `u_j = 2πj/N`, the discrete
   exponential `exp(i k u_j)` and the periodic Poisson identity collapse
   the difference `dftHilbert N G u_j - periodicHilbert G u_j` to a
   tsum over aliased modes `|n| ≥ N/2`.

4. **Multiplier disagreement** (`hilbertMult_disagreement_le_two`):
   `‖hilbertMultDiscrete N (n % N) - hilbertMult n‖ ≤ 2`, with equality
   to `0` whenever `|n| < N/2`. So aliased modes pick up a multiplier
   gap of at most `2`.

5. **Real-Fourier symmetry** (`norm_fourierCoeff_neg_eq`): the two-sided
   tail `|n| ≥ N/2` collapses to twice the one-sided tail `n ≥ N/2`.
   Combined with step 4: `|...| ≤ 4 · ∑'_{m ≥ N/2} ‖c_m‖`.

6. **Fourier-coefficient decay** (`fourierCoeff_norm_le_Cp_div_pow`):
   `‖c_n(g)‖ ≤ Cp / |n|^p` for `n ≠ 0`, `g ∈ C^p` periodic with
   `‖g^{(p)}‖_∞ ≤ Cp`, `p ≥ 2`.

7. **Integral test** (`tsum_inv_pow_tail_le`):
   `∑'_{k : ℕ} 1/(k + M + 1 : ℝ)^p ≤ M^{1-p}/(p-1)`.

8. **Combine**: chain (3–7) with `M = N/2 - 1`.

## Trust-chain status

The theorem `dft_hilbert_aliasing_bound` is proved with **zero `WaveCAP`
axioms**. The bound holds for the genuine sampling-DFT operator at grid
points (rather than the older spectral-truncation idealization);
`truncatedSpectralHilbert` is retained as auxiliary infrastructure
(used in `periodicHilbert_sub_truncatedSpectralHilbert_eq` and
`truncatedSpectralHilbert_diff_le_fourierTail`) for the multiplier-error
analysis.
-/

namespace WaveCAP

open Real Complex MeasureTheory intervalIntegral Set
open scoped Real Topology

-- ===========================================================================
-- Step 1: Spectral truncation (auxiliary) and the sampling-DFT Hilbert
-- ===========================================================================

/-- **Auxiliary: spectral truncation** of `periodicHilbert G`.

Restricts the continuous Fourier-multiplier series to modes `|n| < N/2`:
```
truncatedSpectralHilbert N G u = Re ∑'_n (if |n| < N/2
   then hilbertMult n · c_n(G) · e^{i n u} else 0).
```

This operator is *not* what the Julia FFT routine computes; it is an
intermediate idealization used in the multiplier-disagreement step of the
sampling-DFT aliasing proof. The genuine sampling-DFT operator is
`dftHilbert` below (matching `BabenkoRigorous/src/FFTHilbert.jl`).
-/
noncomputable def truncatedSpectralHilbert (N : ℕ) (G : ℝ → ℝ) : ℝ → ℝ := fun u =>
  Complex.re
    (∑' n : ℤ,
      (if (|n| : ℤ) < ((N / 2 : ℕ) : ℤ) then
        hilbertMult n *
          fourierCoeffOn zero_lt_two_pi (gCircOfReal G) n *
          Complex.exp ((n : ℂ) * Complex.I * (u : ℂ))
       else 0))

/-- The discrete Fourier coefficient of `G` on the `N`-point grid
`u_j = 2π·j/N`:
```
dftCoeff N G k = (1/N) · ∑_{j ∈ Fin N} G(u_j) · exp(-2πi k j / N).
```
By the periodic Poisson identity (`dftCoeff_eq_aliased_sum`),
`dftCoeff N G k = ∑'_m c_{k + m·N}(G)` for `C²` 2π-periodic `G`. -/
noncomputable def dftCoeff (N : ℕ) (G : ℝ → ℝ) (k : ℤ) : ℂ :=
  (1 / (N : ℂ)) * ∑ j : Fin N,
    (G (2 * Real.pi * (j.val : ℝ) / N) : ℂ) *
    Complex.exp (-(2 * Real.pi * Complex.I * (k : ℂ) * (j.val : ℂ) / (N : ℂ)))

/-- **Discrete Hilbert multiplier** on `Fin N` (extended `ℤ`-periodically
in the input by reduction modulo `N`). Returns:
* `-i` on principal positive modes `k ∈ {1, …, N/2 − 1}`,
* `+i` on principal negative modes `k ∈ {N/2 + 1, …, N − 1}`,
* `0` at `k = 0` and at the Nyquist mode `k = N/2`.

Matches Julia's `H_k` convention in
`BabenkoRigorous/src/FFTHilbert.jl:42`. -/
noncomputable def hilbertMultDiscrete (N : ℕ) (k : ℤ) : ℂ :=
  let r := k % (N : ℤ)
  if r = 0 then 0
  else if r = ((N / 2 : ℕ) : ℤ) then 0
  else if r < ((N / 2 : ℕ) : ℤ) then -Complex.I
  else Complex.I

/-- **Sampling-based DFT Hilbert transform** (multiplier form):
```
dftHilbert N G u = Re ∑_{k ∈ Fin N}
   hilbertMultDiscrete N k · dftCoeff N G k · exp(i k u).
```
At grid points `u_j = 2π j / N`, this matches the cot-kernel circular
convolution of `BabenkoRigorous/src/FFTHilbert.jl` (the underlying
Julia computation). -/
noncomputable def dftHilbert (N : ℕ) (G : ℝ → ℝ) : ℝ → ℝ := fun u =>
  Complex.re
    (∑ k : Fin N,
      hilbertMultDiscrete N (k.val : ℤ) *
        dftCoeff N G (k.val : ℤ) *
        Complex.exp ((k.val : ℂ) * Complex.I * (u : ℂ)))

/-- The Julia `u_j = 2π · j / N_fft` grid point. -/
noncomputable def fftGridPoint (N : ℕ) (j : Fin N) : ℝ :=
  2 * Real.pi * (j.val : ℝ) / (N : ℝ)

-- ===========================================================================
-- Stage 2: Periodic Poisson identity on 𝕋
-- ===========================================================================

/-- **Discrete orthogonality of grid exponentials.**

For integer `r` and natural `N ≥ 1`:
```
∑_{j ∈ Fin N} exp(2πi r j / N) = if (N : ℤ) ∣ r then (N : ℂ) else 0.
```
This is the standard "N-th roots of unity sum" identity: when `N ∣ r`,
each exponential is `1` and the sum is `N`; otherwise it is a finite
geometric series with ratio `exp(2πi r / N) ≠ 1` summing to `0`. -/
lemma sum_exp_two_pi_I_grid_eq (N : ℕ) (hN : 0 < N) (r : ℤ) :
    ∑ j : Fin N, Complex.exp (2 * Real.pi * Complex.I * (r : ℂ) * (j.val : ℂ) / (N : ℂ))
      = if (N : ℤ) ∣ r then (N : ℂ) else 0 := by
  have hN_ne : (N : ℂ) ≠ 0 := Nat.cast_ne_zero.mpr hN.ne'
  -- Set z := exp(2πi r / N).
  set z : ℂ := Complex.exp (2 * Real.pi * Complex.I * (r : ℂ) / (N : ℂ)) with hz_def
  -- Each summand is z^j (using exp((j) * a) = exp(a)^j for j : ℕ).
  have h_summand : ∀ j : Fin N,
      Complex.exp (2 * Real.pi * Complex.I * (r : ℂ) * (j.val : ℂ) / (N : ℂ))
        = z ^ j.val := by
    intro j
    rw [hz_def, ← Complex.exp_nat_mul]
    congr 1
    field_simp
  -- Rewrite the sum as ∑ j ∈ Fin N, z^j.val = ∑ k ∈ range N, z^k.
  have h_finset_eq : ∑ j : Fin N,
      Complex.exp (2 * Real.pi * Complex.I * (r : ℂ) * (j.val : ℂ) / (N : ℂ))
        = ∑ k ∈ Finset.range N, z ^ k := by
    rw [Finset.sum_congr rfl (fun j _ => h_summand j)]
    -- ∑ j : Fin N, z^j.val = ∑ k ∈ range N, z^k
    rw [Fin.sum_univ_eq_sum_range]
  rw [h_finset_eq]
  by_cases h_dvd : (N : ℤ) ∣ r
  · -- N ∣ r ⇒ z = 1 ⇒ each term = 1 ⇒ sum = N.
    rw [if_pos h_dvd]
    have hz_eq_one : z = 1 := by
      obtain ⟨m, hm⟩ := h_dvd
      rw [hz_def]
      -- 2π · r / N = 2π · m, so exp = exp(2πi · m) = 1.
      have h_arg : 2 * Real.pi * Complex.I * (r : ℂ) / (N : ℂ)
          = (m : ℂ) * (2 * Real.pi * Complex.I) := by
        have hr_eq : (r : ℂ) = (N : ℂ) * (m : ℂ) := by exact_mod_cast hm
        rw [hr_eq]
        field_simp
      rw [h_arg]
      exact Complex.exp_int_mul_two_pi_mul_I m
    rw [hz_eq_one]
    rw [Finset.sum_congr rfl (fun k _ => by rw [one_pow])]
    rw [Finset.sum_const, Finset.card_range]
    simp
  · -- N ∤ r ⇒ z ≠ 1 ⇒ apply geom_sum_eq ⇒ numerator = z^N - 1 = exp(2πi r) - 1 = 0.
    rw [if_neg h_dvd]
    have hz_ne_one : z ≠ 1 := by
      intro hz_eq_one
      apply h_dvd
      -- z = exp(2πi r / N) = 1 implies 2πi r / N = 2πi m for some m, i.e. r = N m.
      rw [hz_def] at hz_eq_one
      rw [Complex.exp_eq_one_iff] at hz_eq_one
      obtain ⟨n, hn⟩ := hz_eq_one
      -- 2π·i·r/N = n · 2π·i ⇒ r/N = n ⇒ r = n·N.
      have h_pi_ne : (2 * Real.pi * Complex.I : ℂ) ≠ 0 := by
        have hpi := Real.pi_pos
        have hI := Complex.I_ne_zero
        have h2pi : (2 * Real.pi : ℂ) ≠ 0 := by
          have h2pi_real : (2 * Real.pi : ℝ) ≠ 0 := by linarith
          exact_mod_cast h2pi_real
        exact mul_ne_zero h2pi hI
      -- From 2πi·r/N = n·(2πi), cancel 2πi.
      have hmul : (2 * Real.pi * Complex.I) * ((r : ℂ) / (N : ℂ))
          = (2 * Real.pi * Complex.I) * (n : ℂ) := by
        have h_lhs : (2 * Real.pi * Complex.I) * ((r : ℂ) / (N : ℂ))
            = 2 * Real.pi * Complex.I * (r : ℂ) / (N : ℂ) := by ring
        rw [h_lhs, hn]
        ring
      have hdiv_eq : (r : ℂ) / (N : ℂ) = (n : ℂ) :=
        mul_left_cancel₀ h_pi_ne hmul
      have hr_eq_C : (r : ℂ) = (N : ℂ) * (n : ℂ) := by
        rw [← hdiv_eq]
        field_simp
      -- Cast back to ℤ.
      have hr_eq : r = N * n := by exact_mod_cast hr_eq_C
      exact ⟨n, hr_eq⟩
    -- Now apply geom_sum_eq.
    rw [geom_sum_eq hz_ne_one]
    -- z^N = exp(2πi r) = 1
    have hzN : z ^ N = 1 := by
      rw [hz_def]
      rw [← Complex.exp_nat_mul]
      have h_arg : (N : ℂ) * (2 * Real.pi * Complex.I * (r : ℂ) / (N : ℂ))
          = (r : ℂ) * (2 * Real.pi * Complex.I) := by
        field_simp
      rw [h_arg]
      exact Complex.exp_int_mul_two_pi_mul_I r
    rw [hzN]
    simp

set_option maxHeartbeats 800000 in
/-- **Periodic Poisson summation identity on 𝕋.**

For a continuous 2π-periodic function `G : ℝ → ℝ` whose Fourier coefficients
are absolutely summable, the discrete `N`-point Fourier coefficient at index
`k : ℤ` equals the aliased sum of continuous Fourier coefficients:
```
dftCoeff N G k = ∑'_{m : ℤ} c_{k + m·N}(G).
```
This is the standard periodic Poisson identity: sampling on the `N`-point
grid aliases together exactly those continuous modes whose index differs by
a multiple of `N`. -/
lemma dftCoeff_eq_aliased_sum
    (N : ℕ) (hN : 0 < N)
    (G : ℝ → ℝ)
    (hG_cont : Continuous G)
    (hG_per : Function.Periodic G (2 * Real.pi))
    (h_summ : Summable (fun n : ℤ =>
        ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal G) n‖))
    (k : ℤ) :
    dftCoeff N G k
      = ∑' m : ℤ, fourierCoeffOn zero_lt_two_pi (gCircOfReal G)
          (k + m * (N : ℤ)) := by
  -- Local notation for the continuous Fourier coefficient.
  set c : ℤ → ℂ := fun n => fourierCoeffOn zero_lt_two_pi (gCircOfReal G) n with hc_def
  -- N as a nonzero complex.
  have hN_ne : (N : ℂ) ≠ 0 := Nat.cast_ne_zero.mpr hN.ne'
  -- Step 1: Use T := 2π - 0 throughout to align with `fourierCoeffOn`'s def.
  -- (The proof of positivity used by `fourierCoeffOn` is `linarith.. : 0 < 2π - 0`,
  -- not `zero_lt_two_pi`. We avoid mismatch by working over `AddCircle T`.)
  set T : ℝ := 2 * Real.pi - 0 with hT_def
  have hT_pos : 0 < T := by rw [hT_def]; rw [sub_zero]; exact zero_lt_two_pi
  haveI hFact : Fact (0 < T) := ⟨hT_pos⟩
  -- Step 2: Build the continuous lift G_circ on AddCircle T.
  have h_endpoint : (gCircOfReal G) 0 = (gCircOfReal G) (0 + T) := by
    change (G 0 : ℂ) = (G (0 + T) : ℂ)
    rw [hT_def, show (0 + (2 * Real.pi - 0) : ℝ) = (0 + 2 * Real.pi) from by ring,
        hG_per 0]
  have h_gcirc_cont : Continuous (gCircOfReal G) :=
    Complex.continuous_ofReal.comp hG_cont
  have h_gcirc_contOn : ContinuousOn (gCircOfReal G) (Set.Icc 0 (0 + T)) :=
    h_gcirc_cont.continuousOn
  have h_lift_cont : Continuous (AddCircle.liftIoc T 0 (gCircOfReal G)) :=
    AddCircle.liftIoc_continuous h_endpoint h_gcirc_contOn
  set G_circ : C(AddCircle T, ℂ) :=
    ⟨AddCircle.liftIoc T 0 (gCircOfReal G), h_lift_cont⟩ with hG_circ_def
  -- Step 3: Identify fourierCoeff G_circ.toFun with our `c` (now by `rfl`).
  have h_fourier_coeff : ∀ n : ℤ, fourierCoeff G_circ.toFun n = c n := by
    intro n
    rfl
  -- Step 4: Summability of fourierCoeff G_circ implies pointwise convergence.
  have h_fourierCoeff_summ : Summable (fourierCoeff G_circ.toFun) := by
    refine Summable.of_norm ?_
    refine h_summ.congr (fun n => ?_)
    rw [h_fourier_coeff n]
  -- The Mathlib pointwise convergence theorem.
  have h_pointwise : ∀ x : AddCircle T,
      HasSum (fun n : ℤ => fourierCoeff G_circ.toFun n • fourier n x)
        (G_circ.toFun x) :=
    has_pointwise_sum_fourier_series_of_summable h_fourierCoeff_summ
  -- Step 5: Express G(u_j) = ∑'_n c_n · exp(i n u_j) at each grid point.
  -- The Mathlib basis function `fourier n ((u_j : ℝ) : AddCircle (2π))` equals
  -- `exp(2πi n u_j / (2π)) = exp(i n u_j) = exp(2πi nj/N)`.
  have h_lift_at_grid : ∀ j : Fin N,
      G_circ.toFun ((2 * Real.pi * (j.val : ℝ) / N : ℝ) : AddCircle T)
        = (G (2 * Real.pi * (j.val : ℝ) / N) : ℂ) := by
    intro j
    change (AddCircle.liftIoc T 0 (gCircOfReal G))
        ((2 * Real.pi * (j.val : ℝ) / N : ℝ) : AddCircle T)
      = (G (2 * Real.pi * (j.val : ℝ) / N) : ℂ)
    by_cases hj : j.val = 0
    · -- u_j = 0. The circle image of 0 is the same as the image of T = 2π.
      rw [hj]
      simp only [Nat.cast_zero, mul_zero, zero_div]
      have h_circ_eq : ((0 : ℝ) : AddCircle T)
          = ((T : ℝ) : AddCircle T) := by
        -- Use coe_add_period: ((x + p : 𝕜) : AddCircle p) = x.
        -- So ((0 + T : ℝ) : AddCircle T) = (0 : ℝ : AddCircle T).
        -- Hence ((T : ℝ) : AddCircle T) = ((0 + T : ℝ) : AddCircle T) = ((0 : ℝ) : AddCircle T).
        have h_period : ((0 + T : ℝ) : AddCircle T) = ((0 : ℝ) : AddCircle T) :=
          AddCircle.coe_add_period (p := T) 0
        have h_simp : ((0 + T : ℝ) : AddCircle T) = ((T : ℝ) : AddCircle T) := by
          congr 1
          ring
        rw [← h_simp, h_period]
      rw [h_circ_eq]
      rw [AddCircle.liftIoc_coe_apply (p := T) (a := 0)
          (f := gCircOfReal G) (x := T)
          (by rw [zero_add]; exact ⟨hT_pos, le_refl _⟩)]
      change (G T : ℂ) = (G 0 : ℂ)
      have hT_eq : T = 2 * Real.pi := by rw [hT_def]; ring
      rw [hT_eq]
      have hp := hG_per 0
      rw [zero_add] at hp
      rw [hp]
    · -- u_j ∈ (0, 2π) for j ∈ {1, ..., N-1}.
      have hj_pos : 0 < j.val := Nat.pos_of_ne_zero hj
      have hj_lt : j.val < N := j.isLt
      have hu_pos : 0 < 2 * Real.pi * (j.val : ℝ) / N := by
        have hj_nat : (0 : ℝ) < (j.val : ℝ) := by exact_mod_cast hj_pos
        have hN_real : (0 : ℝ) < (N : ℝ) := by exact_mod_cast hN
        positivity
      have hu_le : 2 * Real.pi * (j.val : ℝ) / N ≤ 2 * Real.pi := by
        have hN_real : (0 : ℝ) < (N : ℝ) := by exact_mod_cast hN
        have hj_le : (j.val : ℝ) ≤ (N : ℝ) := by
          have : j.val ≤ N := j.isLt.le
          exact_mod_cast this
        rw [div_le_iff₀ hN_real]
        nlinarith [zero_lt_two_pi]
      rw [AddCircle.liftIoc_coe_apply (a := 0) (p := T)
          (f := gCircOfReal G) (x := 2 * Real.pi * (j.val : ℝ) / N)
          (by rw [zero_add]
              refine ⟨hu_pos, ?_⟩
              rw [hT_def, sub_zero]; exact hu_le)]
      rfl
  -- Step 6: Identify fourier n at grid point u_j with exp(2πi n j / N).
  have h_fourier_at_grid : ∀ (n : ℤ) (j : Fin N),
      fourier n ((2 * Real.pi * (j.val : ℝ) / N : ℝ) : AddCircle T)
        = Complex.exp (2 * Real.pi * Complex.I * (n : ℂ) * (j.val : ℂ) / (N : ℂ)) := by
    intro n j
    rw [fourier_coe_apply]
    -- fourier_coe_apply: fourier n (↑x : AddCircle T) = exp(2πi n x / T).
    -- T = 2π (after `sub_zero` simplification).
    congr 1
    rw [hT_def]
    rw [show ((2 * Real.pi - 0 : ℝ) : ℂ) = (2 * Real.pi : ℂ) by push_cast; ring]
    push_cast
    field_simp
  -- Step 7: Express G(u_j) = ∑'_n c_n · exp(2πi n j / N) at each grid point.
  have h_G_at_grid : ∀ j : Fin N,
      (G (2 * Real.pi * (j.val : ℝ) / N) : ℂ)
        = ∑' n : ℤ, c n
              * Complex.exp (2 * Real.pi * Complex.I * (n : ℂ) * (j.val : ℂ) / (N : ℂ)) := by
    intro j
    have h := h_pointwise ((2 * Real.pi * (j.val : ℝ) / N : ℝ) : AddCircle T)
    have h_eq_tsum : ∑' n, fourierCoeff G_circ.toFun n
        • fourier n (((2 * Real.pi * (j.val : ℝ) / N : ℝ) : AddCircle T))
      = G_circ.toFun ((2 * Real.pi * (j.val : ℝ) / N : ℝ) : AddCircle T) :=
      h.tsum_eq
    rw [h_lift_at_grid j] at h_eq_tsum
    rw [← h_eq_tsum]
    apply tsum_congr
    intro n
    rw [h_fourier_coeff n, h_fourier_at_grid n j, smul_eq_mul]
  -- Step 8: Substitute into dftCoeff N G k.
  -- dftCoeff N G k = (1/N) · ∑_j (G u_j : ℂ) · exp(-2πi k j / N)
  --                = (1/N) · ∑_j (∑'_n c_n · exp(2πi nj/N)) · exp(-2πi k j / N)
  --                = (1/N) · ∑_j ∑'_n c_n · exp(2πi (n-k) j / N)
  -- Interchange (Summable.tsum_finsetSum applied with negated indices because the
  -- finset is the inner sum):
  --                = (1/N) · ∑'_n c_n · (∑_j exp(2πi (n-k) j / N))
  -- By discrete orthogonality, ∑_j exp(2πi (n-k) j / N) = N · 𝟙[N ∣ n - k].
  --                = ∑'_n c_n · 𝟙[N ∣ n - k]
  --                = ∑'_m c_{k+mN}   (reindex)
  have h_dft_unfold : dftCoeff N G k
      = (1 / (N : ℂ)) * ∑ j : Fin N,
          (G (2 * Real.pi * (j.val : ℝ) / N) : ℂ) *
          Complex.exp (-(2 * Real.pi * Complex.I * (k : ℂ) * (j.val : ℂ) / (N : ℂ))) := rfl
  rw [h_dft_unfold]
  -- Substitute G(u_j) by its Fourier series:
  have h_substitute :
      ∑ j : Fin N,
          (G (2 * Real.pi * (j.val : ℝ) / N) : ℂ) *
          Complex.exp (-(2 * Real.pi * Complex.I * (k : ℂ) * (j.val : ℂ) / (N : ℂ)))
        = ∑ j : Fin N,
          (∑' n : ℤ, c n *
              Complex.exp (2 * Real.pi * Complex.I * (n : ℂ) * (j.val : ℂ) / (N : ℂ))) *
          Complex.exp (-(2 * Real.pi * Complex.I * (k : ℂ) * (j.val : ℂ) / (N : ℂ))) := by
    apply Finset.sum_congr rfl
    intro j _
    rw [h_G_at_grid j]
  rw [h_substitute]
  -- Combine into a single tsum per j:
  have h_combine : ∀ j : Fin N,
      (∑' n : ℤ, c n *
          Complex.exp (2 * Real.pi * Complex.I * (n : ℂ) * (j.val : ℂ) / (N : ℂ))) *
      Complex.exp (-(2 * Real.pi * Complex.I * (k : ℂ) * (j.val : ℂ) / (N : ℂ)))
      = ∑' n : ℤ, c n *
          Complex.exp (2 * Real.pi * Complex.I * ((n - k : ℤ) : ℂ) * (j.val : ℂ) / (N : ℂ)) := by
    intro j
    rw [← tsum_mul_right]
    apply tsum_congr
    intro n
    rw [mul_assoc, ← Complex.exp_add]
    congr 2
    push_cast
    ring
  rw [Finset.sum_congr rfl (fun j _ => h_combine j)]
  -- Step 9: Interchange. The inner tsums are over n : ℤ, the outer sum over j : Fin N.
  -- Each inner tsum is summable: we have ∑' c_n * exp(...) where ‖exp‖ = 1, so summable.
  -- Apply Summable.tsum_finsetSum: ∑' n, ∑ j, f n j = ∑ j, ∑' n, f n j.
  -- We're going from outer-Finset, inner-tsum to outer-tsum, inner-Finset.
  have h_per_n_summ : ∀ j : Fin N,
      Summable (fun n : ℤ => c n *
          Complex.exp (2 * Real.pi * Complex.I * ((n - k : ℤ) : ℂ) * (j.val : ℂ) / (N : ℂ))) := by
    intro j
    -- The norm of each summand is ‖c n‖ · 1 = ‖c n‖, which is summable.
    apply Summable.of_norm
    refine h_summ.congr (fun n => ?_)
    show ‖c n‖ = ‖c n *
        Complex.exp (2 * Real.pi * Complex.I * ((n - k : ℤ) : ℂ) * (j.val : ℂ) / (N : ℂ))‖
    rw [norm_mul]
    have h_exp_norm :
        ‖Complex.exp (2 * Real.pi * Complex.I * ((n - k : ℤ) : ℂ) * (j.val : ℂ) / (N : ℂ))‖ = 1 := by
      have h_arg_real : 2 * Real.pi * Complex.I * ((n - k : ℤ) : ℂ) * (j.val : ℂ) / (N : ℂ)
          = (((2 * Real.pi * ((n - k : ℤ) : ℝ) * (j.val : ℝ) / (N : ℝ)) : ℝ) : ℂ) * Complex.I := by
        push_cast; ring
      rw [h_arg_real, Complex.norm_exp_ofReal_mul_I]
    rw [h_exp_norm, mul_one]
  -- Apply tsum_finsetSum reversed: ∑ j, ∑' n, f n j = ∑' n, ∑ j, f n j.
  rw [show (∑ j : Fin N, ∑' n : ℤ, c n
              * Complex.exp (2 * Real.pi * Complex.I * ((n - k : ℤ) : ℂ) * (j.val : ℂ) / (N : ℂ)))
      = ∑' n : ℤ, ∑ j : Fin N, c n
              * Complex.exp (2 * Real.pi * Complex.I * ((n - k : ℤ) : ℂ) * (j.val : ℂ) / (N : ℂ)) from
    (Summable.tsum_finsetSum (fun j _ => h_per_n_summ j)).symm]
  -- Pull out c n from the j-sum (since it doesn't depend on j):
  have h_pull_cn : ∀ n : ℤ,
      ∑ j : Fin N, c n *
          Complex.exp (2 * Real.pi * Complex.I * ((n - k : ℤ) : ℂ) * (j.val : ℂ) / (N : ℂ))
      = c n * ∑ j : Fin N,
          Complex.exp (2 * Real.pi * Complex.I * ((n - k : ℤ) : ℂ) * (j.val : ℂ) / (N : ℂ)) :=
    fun n => Finset.mul_sum _ _ _ |>.symm
  rw [tsum_congr h_pull_cn]
  -- Apply discrete orthogonality:
  have h_orth : ∀ n : ℤ,
      ∑ j : Fin N, Complex.exp (2 * Real.pi * Complex.I * ((n - k : ℤ) : ℂ) * (j.val : ℂ) / (N : ℂ))
      = if (N : ℤ) ∣ (n - k) then (N : ℂ) else 0 := by
    intro n
    exact sum_exp_two_pi_I_grid_eq N hN (n - k)
  rw [tsum_congr (fun n => by rw [h_orth n])]
  -- Step 10: (1/N) · ∑'_n c_n · (if N ∣ (n - k) then N else 0)
  --        = ∑'_n c_n · (if N ∣ (n - k) then 1 else 0)
  --        = ∑'_n if N ∣ (n - k) then c_n else 0
  have h_simplify : ∀ n : ℤ,
      c n * (if (N : ℤ) ∣ (n - k) then (N : ℂ) else 0)
      = (if (N : ℤ) ∣ (n - k) then c n * (N : ℂ) else 0) := by
    intro n
    by_cases h : (N : ℤ) ∣ (n - k)
    · rw [if_pos h, if_pos h]
    · rw [if_neg h, if_neg h, mul_zero]
  rw [tsum_congr h_simplify]
  -- Factor out the constant N inside the tsum:
  have h_factor_N : ∀ n : ℤ,
      (if (N : ℤ) ∣ (n - k) then c n * (N : ℂ) else 0)
      = (N : ℂ) * (if (N : ℤ) ∣ (n - k) then c n else 0) := by
    intro n
    by_cases h : (N : ℤ) ∣ (n - k)
    · rw [if_pos h, if_pos h, mul_comm]
    · rw [if_neg h, if_neg h, mul_zero]
  rw [tsum_congr h_factor_N, tsum_mul_left]
  -- Now we have (1/N) · N · ∑'_n (if N ∣ (n-k) then c n else 0) = ∑'_n (if ...).
  rw [show (1 / (N : ℂ)) * ((N : ℂ) * ∑' n : ℤ,
        (if (N : ℤ) ∣ (n - k) then c n else 0))
      = ∑' n : ℤ, (if (N : ℤ) ∣ (n - k) then c n else 0) from by
    field_simp]
  -- Step 11: Reindex via the bijection ℤ → ℤ, m ↦ k + m·N.
  -- The image of this bijection is exactly {n : ℤ | N ∣ (n - k)} (as a set), but
  -- since we have indicator-form, we directly reindex.
  -- We show: ∑' n, (if N ∣ (n - k) then c n else 0) = ∑' m, c (k + m · N).
  -- Use HasSum.tsum_eq applied to a constructed HasSum.
  -- The cleanest path: ∑'_m c (k + m·N) is a tsum over ℤ. Apply Equiv.tsum_eq
  -- with the bijection ℤ ≃ ℤ defined by m ↦ k + m·N? But that's not bijective
  -- onto ℤ — it's bijective onto {n : ℤ | N ∣ (n - k)}.
  -- Use the subtype trick: tsum over subtype = tsum over indicator.
  -- ∑' n, (if N ∣ (n - k) then c n else 0)
  --   = ∑' n, ({n | N ∣ (n - k)} : Set ℤ).indicator c n   (by definition of indicator)
  --   = ∑' (n : {n | N ∣ (n - k)}), c n.val   (tsum_subtype reverse)
  -- Then bijection from ℤ to {n | N ∣ (n - k)} via m ↦ ⟨k + m·N, _⟩ closes it.
  symm
  -- Goal: ∑' m, c (k + m·N) = ∑' n, (if N ∣ (n - k) then c n else 0).
  -- We'll show HasSum (fun m : ℤ => c (k + m·N)) (∑' n, ...).
  -- The equivalence:
  have hN_ne_int : (N : ℤ) ≠ 0 := by
    have : 0 < (N : ℤ) := Int.natCast_pos.mpr hN
    omega
  -- Build the equivalence ℤ ≃ {n : ℤ // (N : ℤ) ∣ (n - k)}, m ↦ ⟨k + N·m, _⟩.
  -- Note: in Lean's `Int.Dvd`, `(N : ℤ) ∣ r` unfolds to `∃ c, r = N * c`.
  set e : ℤ ≃ {n : ℤ // (N : ℤ) ∣ (n - k)} :=
    { toFun := fun m => ⟨k + (N : ℤ) * m, by
        refine ⟨m, ?_⟩
        ring⟩
      invFun := fun nh => (nh.val - k) / (N : ℤ)
      left_inv := fun m => by
        change (k + (N : ℤ) * m - k) / (N : ℤ) = m
        have h : k + (N : ℤ) * m - k = (N : ℤ) * m := by ring
        rw [h, Int.mul_ediv_cancel_left m hN_ne_int]
      right_inv := fun nh => by
        obtain ⟨n, hn⟩ := nh
        obtain ⟨m, hm⟩ := hn
        apply Subtype.ext
        change k + (N : ℤ) * ((n - k) / (N : ℤ)) = n
        have h_div : (n - k) / (N : ℤ) = m := by
          rw [hm]
          exact Int.mul_ediv_cancel_left m hN_ne_int
        rw [h_div]
        linarith }
    with he_def
  -- Now compute ∑'_n (if ...) via tsum_subtype.
  have h_to_subtype :
      ∑' n : ℤ, (if (N : ℤ) ∣ (n - k) then c n else 0)
      = ∑' nh : {n : ℤ // (N : ℤ) ∣ (n - k)}, c nh.val := by
    have h_eq : (fun n : ℤ => if (N : ℤ) ∣ (n - k) then c n else 0)
          = ({n : ℤ | (N : ℤ) ∣ (n - k)}).indicator c := by
      funext n
      by_cases h : (N : ℤ) ∣ (n - k)
      · rw [if_pos h]
        rw [Set.indicator_of_mem (show n ∈ {n : ℤ | (N : ℤ) ∣ (n - k)} from h)]
      · rw [if_neg h]
        rw [Set.indicator_of_notMem (show n ∉ {n : ℤ | (N : ℤ) ∣ (n - k)} from h)]
    rw [h_eq]
    exact (tsum_subtype _ c).symm
  rw [h_to_subtype]
  -- Reindex via e: ∑' m, c (e m).val = ∑' nh, c nh.val.
  -- The Equiv.tsum_eq says ∑' c, f (e c) = ∑' b, f b. Apply with f = (fun nh => c nh.val).
  -- Then ∑' m : ℤ, (fun nh => c nh.val) (e m) = ∑' nh, c nh.val.
  -- We need to identify (fun nh => c nh.val) (e m) with c (k + m·N).
  have h_compose : ∀ m : ℤ, (fun nh : {n : ℤ // (N : ℤ) ∣ (n - k)} => c nh.val) (e m)
      = c (k + m * (N : ℤ)) := by
    intro m
    show c ((e m).val) = c (k + m * (N : ℤ))
    have h_eq : (e m).val = k + (N : ℤ) * m := rfl
    rw [h_eq]
    congr 1
    ring
  -- Equiv.tsum_eq : ∑' c, f (e c) = ∑' b, f b
  -- So ∑' m, (fun nh => c nh.val) (e m) = ∑' nh, c nh.val.
  have h_tsum_via_e :
      ∑' m : ℤ, (fun nh : {n : ℤ // (N : ℤ) ∣ (n - k)} => c nh.val) (e m)
        = ∑' nh : {n : ℤ // (N : ℤ) ∣ (n - k)}, c nh.val :=
    e.tsum_eq (fun nh => c nh.val)
  -- Rewrite both sides.
  calc ∑' m : ℤ, c (k + m * (N : ℤ))
      = ∑' m : ℤ, (fun nh : {n : ℤ // (N : ℤ) ∣ (n - k)} => c nh.val) (e m) := by
        refine tsum_congr (fun m => ?_)
        rw [h_compose m]
    _ = ∑' nh : {n : ℤ // (N : ℤ) ∣ (n - k)}, c nh.val := h_tsum_via_e

-- ===========================================================================
-- Stage 3: Multiplier-disagreement bound + aliasing bound at grid points
-- ===========================================================================

/-- **Multiplier-disagreement bound** for the discrete vs continuous Hilbert
multipliers. The mismatch `‖M^d_N(n mod N) − M(n)‖` is at most `2`
for every `n : ℤ` (and zero on principal modes `|n| < N/2`). -/
lemma hilbertMult_disagreement_le_two
    (N : ℕ) (n : ℤ) :
    ‖hilbertMultDiscrete N n - hilbertMult n‖ ≤ 2 := by
  -- Both multipliers have norm at most 1.
  have h1 : ‖hilbertMultDiscrete N n‖ ≤ 1 := by
    set r := n % (N : ℤ) with hr_def
    show ‖(if r = 0 then (0 : ℂ)
           else if r = ((N / 2 : ℕ) : ℤ) then 0
           else if r < ((N / 2 : ℕ) : ℤ) then -Complex.I
           else Complex.I)‖ ≤ 1
    split_ifs <;> simp [Complex.norm_I]
  have h2 : ‖hilbertMult n‖ ≤ 1 := norm_hilbertMult_le_one n
  calc ‖hilbertMultDiscrete N n - hilbertMult n‖
      ≤ ‖hilbertMultDiscrete N n‖ + ‖hilbertMult n‖ := norm_sub_le _ _
    _ ≤ 1 + 1 := by linarith
    _ = 2 := by norm_num

/-- A simpler formula at grid points: `exp(2πi · (k + m·N) · j / N)
= exp(2πi · k · j / N) · exp(2πi · m · j) = exp(2πi · k · j / N)`. -/
lemma exp_two_pi_I_grid_aliased
    (N : ℕ) (hN : 0 < N) (n m : ℤ) (j : Fin N) :
    Complex.exp (2 * Real.pi * Complex.I * ((n + m * (N : ℤ) : ℤ) : ℂ) * (j.val : ℂ) / (N : ℂ))
      = Complex.exp (2 * Real.pi * Complex.I * (n : ℂ) * (j.val : ℂ) / (N : ℂ)) := by
  have hN_ne : (N : ℂ) ≠ 0 := Nat.cast_ne_zero.mpr hN.ne'
  -- Compute: exp(2πi (n + mN) j / N) = exp(2πi n j/N + 2πi m j)
  --        = exp(2πi n j/N) · exp(2πi m j) = exp(2πi n j/N) · 1.
  rw [show 2 * Real.pi * Complex.I * ((n + m * (N : ℤ) : ℤ) : ℂ) * (j.val : ℂ) / (N : ℂ)
        = 2 * Real.pi * Complex.I * (n : ℂ) * (j.val : ℂ) / (N : ℂ)
          + (m * j.val : ℤ) * (2 * Real.pi * Complex.I) from by push_cast; field_simp]
  rw [Complex.exp_add, Complex.exp_int_mul_two_pi_mul_I, mul_one]

/-- The cast-normalized variant of `exp_two_pi_I_grid_aliased`, with
`Complex` argument `k + m·N` (push-cast form, where `N` is cast through ℤ).
Useful inside `tsum_congr` when Lean has already pushed casts inward. -/
lemma exp_two_pi_I_grid_aliased_pushed
    (N : ℕ) (hN : 0 < N) (k m : ℤ) (j : Fin N) :
    Complex.exp (2 * Real.pi * Complex.I *
                  ((k : ℂ) + (m : ℂ) * ((N : ℤ) : ℂ)) *
                  (j.val : ℂ) / (N : ℂ))
      = Complex.exp (2 * Real.pi * Complex.I * (k : ℂ) * (j.val : ℂ) / (N : ℂ)) := by
  have h := exp_two_pi_I_grid_aliased N hN k m j
  have h_cast : ((k + m * (N : ℤ) : ℤ) : ℂ) = (k : ℂ) + (m : ℂ) * ((N : ℤ) : ℂ) := by
    push_cast; ring
  rw [h_cast] at h
  exact h

/-- **Principal-mode multiplier agreement.** For modes `|n| < N/2`, the
discrete and continuous Hilbert multipliers agree exactly.

(For `n = 0`, both are `0`. For `0 < n < N/2`, both are `-i`. For
`-(N/2) < n < 0`, both are `+i` — since `(-i) · sign(n) = (-i)·(-1) = i`.) -/
lemma hilbertMult_disagreement_eq_zero_of_principal
    (N : ℕ) (hN_even : Even N) (hN : 4 ≤ N) (n : ℤ)
    (hn : |n| < ((N / 2 : ℕ) : ℤ)) :
    hilbertMultDiscrete N n - hilbertMult n = 0 := by
  have hNpos : 0 < N := by omega
  have hN_half_pos : 0 < (N / 2 : ℕ) := by omega
  have hN_int : (N : ℤ) ≠ 0 := by exact_mod_cast hNpos.ne'
  rw [sub_eq_zero]
  -- Show `hilbertMultDiscrete N n = hilbertMult n`.
  unfold hilbertMultDiscrete hilbertMult
  by_cases hn0 : n = 0
  · subst hn0
    simp
  · have hnabs_pos : 0 < |n| := abs_pos.mpr hn0
    rcases lt_or_gt_of_ne hn0 with hn_neg | hn_pos
    · -- n < 0: -(N/2) < n < 0.
      have h_neg_lt : -((N / 2 : ℕ) : ℤ) < n := by
        have h := abs_of_neg hn_neg
        linarith [h]
      have h_n_neg : n < 0 := hn_neg
      have h_lt_N : n < (N : ℤ) := by
        have : ((N / 2 : ℕ) : ℤ) ≤ (N : ℤ) := by
          have : N / 2 ≤ N := Nat.div_le_self _ _
          exact_mod_cast this
        linarith
      have h_neg_N : -(N : ℤ) < n := by
        have h_half_le : ((N / 2 : ℕ) : ℤ) ≤ (N : ℤ) := by
          have : N / 2 ≤ N := Nat.div_le_self _ _
          exact_mod_cast this
        linarith
      -- n mod N = n + N.
      have h_mod : n % (N : ℤ) = n + (N : ℤ) := by
        have h_emod_pos : 0 ≤ n + (N : ℤ) := by linarith
        have h_emod_lt : n + (N : ℤ) < (N : ℤ) := by linarith
        have h_eq : (n + (N : ℤ)) % (N : ℤ) = n % (N : ℤ) := by
          have := Int.add_mul_emod_self_left n (N : ℤ) 1
          rw [mul_one] at this
          exact this
        rw [← h_eq, Int.emod_eq_of_lt h_emod_pos h_emod_lt]
      rw [h_mod]
      have h_npos : 0 < n + (N : ℤ) := by linarith
      have h_nN_gt : ((N / 2 : ℕ) : ℤ) < n + (N : ℤ) := by
        have h_abs_eq := abs_of_neg hn_neg
        -- |n| = -n < N/2 ⇒ N - N/2 < n + N.
        -- For N even, N = 2·(N/2), so N - N/2 = N/2.
        rcases hN_even with ⟨k, hk⟩
        have hk_eq : (N : ℕ) / 2 = k := by omega
        have hk_Z : (N : ℤ) = 2 * (k : ℤ) := by rw [hk]; push_cast; ring
        have hN_half_Z : ((N / 2 : ℕ) : ℤ) = (k : ℤ) := by rw [hk_eq]
        rw [hN_half_Z]
        rw [hN_half_Z] at hn
        rw [h_abs_eq] at hn
        linarith
      have h_nN_lt : n + (N : ℤ) < (N : ℤ) := by linarith
      have h_mod_ne_zero : ¬ (n + (N : ℤ) = 0) := by linarith
      have h_mod_ne_half : ¬ (n + (N : ℤ) = ((N / 2 : ℕ) : ℤ)) := by linarith
      have h_mod_lt : ¬ (n + (N : ℤ) < ((N / 2 : ℕ) : ℤ)) := by linarith
      rw [if_neg h_mod_ne_zero, if_neg h_mod_ne_half, if_neg h_mod_lt]
      -- LHS: i. RHS: -i · sign(n) = -i · (-1) = i.
      have hsign : Int.sign n = -1 := Int.sign_eq_neg_one_iff_neg.mpr hn_neg
      rw [if_neg hn0, hsign]
      push_cast
      ring
    · -- n > 0: 0 < n < N/2.
      have h_n_pos : 0 < n := hn_pos
      have h_abs : |n| = n := abs_of_pos h_n_pos
      have h_n_lt : n < ((N / 2 : ℕ) : ℤ) := by rw [h_abs] at hn; exact hn
      have h_lt_N : n < (N : ℤ) := by
        have : ((N / 2 : ℕ) : ℤ) ≤ (N : ℤ) := by
          have : N / 2 ≤ N := Nat.div_le_self _ _
          exact_mod_cast this
        linarith
      have h_mod : n % (N : ℤ) = n := Int.emod_eq_of_lt h_n_pos.le h_lt_N
      rw [h_mod]
      have h_mod_ne_zero : ¬ (n = 0) := h_n_pos.ne'
      have h_mod_ne_half : ¬ (n = ((N / 2 : ℕ) : ℤ)) := by linarith
      rw [if_neg h_mod_ne_zero, if_neg h_mod_ne_half, if_pos h_n_lt]
      -- LHS: -i. RHS: -i · sign(n) = -i · 1 = -i.
      have hsign : Int.sign n = 1 := Int.sign_eq_one_iff_pos.mpr h_n_pos
      rw [if_neg h_mod_ne_zero, hsign]
      push_cast
      ring

-- ===========================================================================
-- Stage 3 main theorem (`samplingDFTHilbert_diff_le_aliasingTail_grid`):
-- Aliasing bound at grid points. The proof outline is:
--   1. Re-express dftHilbert N G u_j (multiplier-form ∑_{k ∈ Fin N})
--      as ∑'_n M^d(n) c_n exp(2πi nj/N) by interchanging the finite k-sum
--      with the inner Poisson tsum and using ℤ ≃ Fin N × ℤ.
--   2. Subtract periodicHilbert G u_j = Re ∑'_n M(n) c_n exp(2πi nj/N).
--   3. Use the multiplier-disagreement lemmas (above) to collapse to a
--      bound ≤ ∑'_{|n| ≥ N/2} 2‖c_n‖.
--   4. Use norm_fourierCoeff_neg_eq to collapse to 4 · ∑'_{m ≥ N/2} ‖c_m‖.
--
-- This theorem is fully proved later in this file
-- (`samplingDFTHilbert_diff_le_aliasingTail_grid`), following exactly
-- this outline; the auxiliary multiplier-disagreement lemmas above feed
-- into it. (An earlier comment here claimed the proof was deferred —
-- stale; corrected 2026-07-03.)

/-- **Fourier-coefficient decay for `C^p` functions (proved by induction on `p`).**

For `G ∈ C^p(ℝ)` 2π-periodic with `‖G^{(p)}‖_∞ ≤ Cp` and `p ≥ 2`, the `n`-th
Fourier coefficient satisfies
```
‖c_n(G)‖ ≤ Cp / |n|^p   for every  n ≠ 0.
```

**Proof.** Induction on `p` starting at `p = 2` (`Nat.le_induction`),
with the inductive hypothesis quantified over the function and the bound.
- Base `p = 2`: direct application of `WaveCAP.fourierCoeff_norm_le_F₂_div_sq`.
- Step `p → p + 1`: a single IBP step
  (`WaveCAP.fourierCoeff_norm_le_inv_n_mul_fourierCoeff_deriv`) gives
  `‖c_n(G)‖ ≤ (1/|n|) · ‖c_n(G')‖`. Applied with `G' = deriv G`
  (which is `C^p` periodic with `iteratedDeriv p (deriv G) = iteratedDeriv (p+1) G`
  by `iteratedDeriv_succ'`), the inductive hypothesis yields
  `‖c_n(G')‖ ≤ Cp / |n|^p`, hence `‖c_n(G)‖ ≤ Cp / |n|^(p+1)`. -/
theorem fourierCoeff_norm_le_Cp_div_pow
    (p : ℕ) (hp : 2 ≤ p) (G : ℝ → ℝ)
    (hG : ContDiff ℝ p G) (hG_per : Function.Periodic G (2 * Real.pi))
    (Cp : ℝ) (hCp : ∀ u, |iteratedDeriv p G u| ≤ Cp)
    {n : ℤ} (hn : n ≠ 0) :
    ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal G) n‖ ≤ Cp / |(n : ℝ)| ^ p := by
  -- We prove the strengthened statement (∀-quantified over G, Cp) by induction
  -- on p, then specialise.
  suffices hSuff : ∀ q : ℕ, ∀ G : ℝ → ℝ,
      ContDiff ℝ (q + 2) G → Function.Periodic G (2 * Real.pi) →
      ∀ Cp : ℝ, (∀ u, |iteratedDeriv (q + 2) G u| ≤ Cp) →
        ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal G) n‖
          ≤ Cp / |(n : ℝ)| ^ (q + 2) by
    -- Specialise to q = p - 2.
    obtain ⟨q, rfl⟩ : ∃ q : ℕ, p = q + 2 := ⟨p - 2, by omega⟩
    exact hSuff q G hG hG_per Cp hCp
  -- The induction on q : ℕ.
  intro q
  induction q with
  | zero =>
      -- q = 0, so p = 2: invoke fourierCoeff_norm_le_F₂_div_sq.
      intro G hG hG_per Cp hCp
      have h := fourierCoeff_norm_le_F₂_div_sq G hG hG_per Cp hCp hn
      -- `(n : ℝ) ^ 2 = |(n : ℝ)| ^ 2`.
      have hpow : (n : ℝ) ^ 2 = |(n : ℝ)| ^ (0 + 2) := by
        rw [zero_add]; exact (sq_abs (n : ℝ)).symm
      rw [hpow] at h
      exact h
  | succ q ih =>
      -- q+1, so p = (q+1)+2 = (q+2)+1. Apply one IBP step + IH.
      intro G hG hG_per Cp hCp
      -- Smoothness reduction: G ∈ C^((q+1)+2) ⟹ G ∈ C^1 and deriv G ∈ C^(q+2).
      -- Reshape: ContDiff ℝ ((q + 1 + 2) : ℕ) G ⟹ ContDiff ℝ (((q + 2) : ℕ) + 1) G.
      -- We need the form `(_ + 1)` in WithTop ℕ∞ so we can apply `.one_of_succ` and `.deriv'`.
      set m : WithTop ℕ∞ := ((q + 2 : ℕ) : WithTop ℕ∞) with hm_def
      have hG_qp1 : ContDiff ℝ (m + 1) G := by
        have : ContDiff ℝ ((q + 1 + 2 : ℕ) : WithTop ℕ∞) G := hG
        have heq : ((q + 1 + 2 : ℕ) : WithTop ℕ∞) = m + 1 := by
          rw [hm_def]; push_cast; ring
        rw [heq] at this
        exact this
      -- C^1 ⊂ C^(succ) by ContDiff.one_of_succ.
      have hG_C1 : ContDiff ℝ 1 G := hG_qp1.one_of_succ
      -- ContDiff.deriv' : ContDiff (n+1) f → ContDiff n (deriv f).
      have hG'_ContDiff : ContDiff ℝ (q + 2 : ℕ) (deriv G) := by
        have := hG_qp1.deriv'
        -- this : ContDiff ℝ m (deriv G), and `m = ((q+2 : ℕ) : WithTop ℕ∞)`.
        rw [hm_def] at this
        exact_mod_cast this
      -- Periodicity of deriv G.
      have hG'_per : Function.Periodic (deriv G) (2 * Real.pi) :=
        Periodic.deriv_periodic hG_per
      -- The ((q+1)+2)-th iterated derivative of G equals (q+2)-th of deriv G.
      have hiter_eq : iteratedDeriv ((q + 1) + 2) G = iteratedDeriv (q + 2) (deriv G) := by
        have h_rw : ((q + 1 : ℕ) + 2 : ℕ) = ((q + 2 : ℕ) + 1 : ℕ) := by omega
        rw [h_rw]
        exact iteratedDeriv_succ'
      have hCp_deriv : ∀ u, |iteratedDeriv (q + 2) (deriv G) u| ≤ Cp := by
        intro u
        have := hCp u
        rw [hiter_eq] at this
        exact this
      -- Apply the single-IBP step.
      have hibp := fourierCoeff_norm_le_inv_n_mul_fourierCoeff_deriv
        G hG_C1 hG_per hn
      -- Apply IH at q to G' = deriv G.
      have hIH := ih (deriv G) hG'_ContDiff hG'_per Cp hCp_deriv
      -- Combine: ‖c_n(G)‖ ≤ (1/|n|) · (Cp/|n|^(q+2)) = Cp/|n|^(q+3).
      have habs_pos : (0 : ℝ) < |(n : ℝ)| := by
        rw [abs_pos]; exact_mod_cast hn
      have habs_nn : (0 : ℝ) ≤ |(n : ℝ)| := habs_pos.le
      have hCp_nn : (0 : ℝ) ≤ Cp := le_trans (abs_nonneg _) (hCp 0)
      calc ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal G) n‖
          ≤ (1 / |(n : ℝ)|) *
                ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal (deriv G)) n‖ := hibp
        _ ≤ (1 / |(n : ℝ)|) * (Cp / |(n : ℝ)| ^ (q + 2)) := by
              apply mul_le_mul_of_nonneg_left hIH
              positivity
        _ = Cp / |(n : ℝ)| ^ (q + 1 + 2) := by
              -- 1/|n| · Cp/|n|^(q+2) = Cp / |n|^(q+3).
              have hne : |(n : ℝ)| ≠ 0 := habs_pos.ne'
              have h_pow : |(n : ℝ)| ^ (q + 1 + 2) = |(n : ℝ)| ^ (q + 2) * |(n : ℝ)| := by
                rw [show q + 1 + 2 = (q + 2) + 1 from by omega, pow_succ]
              rw [h_pow]
              field_simp

-- ===========================================================================
-- Step 2a: Real-Fourier symmetry  ‖c_{-n}(G)‖ = ‖c_n(G)‖
-- ===========================================================================

/-- **Real-Fourier symmetry.** For a real-valued function `G : ℝ → ℝ` viewed
as a complex-valued integrand via `gCircOfReal`, the Fourier coefficients
satisfy `c_{-n}(G) = conj(c_n(G))`. Hence `‖c_{-n}(G)‖ = ‖c_n(G)‖`.

The proof: `c_n(G) = (1/(2π)) · ∫_0^{2π} exp(-i n x) · G(x) dx`. Conjugating
flips the sign of the imaginary exponent, and `G(x)` is real so `conj G(x) =
G(x)`; the result is `c_{-n}(G)`. -/
lemma norm_fourierCoeff_neg_eq (G : ℝ → ℝ) (n : ℤ) :
    ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal G) (-n)‖
      = ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal G) n‖ := by
  -- We show `c_{-n}(G) = conj(c_n(G))` then take norms.
  have hkey :
      fourierCoeffOn zero_lt_two_pi (gCircOfReal G) (-n)
        = starRingEnd ℂ (fourierCoeffOn zero_lt_two_pi (gCircOfReal G) n) := by
    -- Compute both via the integral formula and identify them.
    -- LHS = (1/(2π)) • ∫_0^{2π} fourier(--n)(x) • G(x) dx
    -- RHS = conj ((1/(2π)) • ∫_0^{2π} fourier(-n)(x) • G(x) dx)
    --     = (1/(2π)) • conj (∫_0^{2π} fourier(-n)(x) • G(x) dx) (real prefactor)
    --     = (1/(2π)) • ∫_0^{2π} conj (fourier(-n)(x) • G(x)) dx (conj-integral)
    --     = (1/(2π)) • ∫_0^{2π} fourier(--n)(x) • G(x) dx (integrand identity)
    rw [fourierCoeffOn_eq_integral, fourierCoeffOn_eq_integral]
    -- Strategy: factor out the real prefactor 1/(2π) from both smul; rewrite RHS so
    -- conj is just on the integral, then identify integrands.
    -- Convert smul to multiplication explicitly using `Complex.real_smul`.
    change ((1 / (2 * Real.pi - 0) : ℝ) : ℂ) *
            ∫ x in (0:ℝ)..(2 * Real.pi),
              fourier (-(-n)) ((x : ℝ) : AddCircle (2 * Real.pi - 0)) • gCircOfReal G x =
        starRingEnd ℂ
          (((1 / (2 * Real.pi - 0) : ℝ) : ℂ) *
            ∫ x in (0:ℝ)..(2 * Real.pi),
              fourier (-n) ((x : ℝ) : AddCircle (2 * Real.pi - 0)) • gCircOfReal G x)
    rw [map_mul]
    -- conj(pref) = pref.
    rw [show starRingEnd ℂ (((1 / (2 * Real.pi - 0) : ℝ) : ℂ))
          = ((1 / (2 * Real.pi - 0) : ℝ) : ℂ) from Complex.conj_ofReal _]
    congr 1
    -- Move conj inside the integral. The interval integral is `∫_Ioc a b - ∫_Ioc b a`
    -- (Bochner integrals); conj distributes through both via MeasureTheory.integral_conj.
    have hconj_int : ∀ (f : ℝ → ℂ),
        starRingEnd ℂ (∫ x in (0:ℝ)..(2*Real.pi), f x)
          = ∫ x in (0:ℝ)..(2*Real.pi), starRingEnd ℂ (f x) := by
      intro f
      unfold intervalIntegral
      rw [map_sub]
      congr 1
      · exact (integral_conj (f := f)
            (μ := MeasureTheory.volume.restrict (Set.Ioc 0 (2 * Real.pi)))).symm
      · exact (integral_conj (f := f)
            (μ := MeasureTheory.volume.restrict (Set.Ioc (2 * Real.pi) 0))).symm
    rw [hconj_int]
    apply intervalIntegral.integral_congr
    intro x _
    -- Goal (after beta reduction): fourier(--n)(x) • G(x) = conj (fourier(-n)(x) • G(x)).
    change fourier (-(-n)) ((x : ℝ) : AddCircle (2 * Real.pi - 0)) • gCircOfReal G x =
        starRingEnd ℂ (fourier (-n) ((x : ℝ) : AddCircle (2 * Real.pi - 0)) • gCircOfReal G x)
    rw [smul_eq_mul, smul_eq_mul, map_mul]
    -- conj (fourier(-n)(x)) = fourier(--n)(x); conj (G(x) : ℂ) = G(x) since G real.
    have hpos : fourier (-(-n)) ((x : ℝ) : AddCircle (2 * Real.pi - 0)) =
        Complex.exp (((n : ℂ) * Complex.I * (x : ℂ))) := by
      have h := fourier_neg_coe_apply_real (-n) x
      rw [h]
      congr 1
      push_cast
      ring
    have hneg : fourier (-n) ((x : ℝ) : AddCircle (2 * Real.pi - 0)) =
        Complex.exp (-((n : ℂ) * Complex.I * (x : ℂ))) :=
      fourier_neg_coe_apply_real n x
    rw [hpos, hneg]
    -- Goal: exp(i n x) * G(x) = conj(exp(-i n x)) * conj(G(x)).
    have hconj_exp :
        starRingEnd ℂ (Complex.exp (-((n : ℂ) * Complex.I * (x : ℂ))))
          = Complex.exp (((n : ℂ) * Complex.I * (x : ℂ))) := by
      rw [← Complex.exp_conj]
      congr 1
      rw [map_neg, map_mul, map_mul, Complex.conj_I]
      have hnconj : starRingEnd ℂ ((n : ℂ)) = (n : ℂ) := by
        rw [show ((n : ℂ)) = ((n : ℝ) : ℂ) from by push_cast; rfl]
        exact Complex.conj_ofReal _
      have hxconj : starRingEnd ℂ ((x : ℂ)) = (x : ℂ) := Complex.conj_ofReal _
      rw [hnconj, hxconj]
      ring
    have hconj_real :
        starRingEnd ℂ (gCircOfReal G x) = gCircOfReal G x := by
      change starRingEnd ℂ ((G x : ℂ)) = ((G x : ℂ))
      exact Complex.conj_ofReal _
    rw [hconj_exp, hconj_real]
  rw [hkey]
  exact RCLike.norm_conj _

-- ===========================================================================
-- Step 2b: Spectral-truncation difference equals the high-mode tail
-- ===========================================================================

/-- **Mode-wise difference between the spectral truncation and the full
Hilbert series.** For `g ∈ C²(ℝ)` 2π-periodic, the difference
`periodicHilbert g u − truncatedSpectralHilbert N g u` equals the missing
high-mode tail:
```
periodicHilbert g u − truncatedSpectralHilbert N g u
  = Re ∑'_{n : ℤ} (if |n| < N/2 then 0
                    else hilbertMult n · c_n(g) · e^{i n u}).
```
This follows from subtracting the two Fourier-series formulas and using
summability of both. -/
lemma periodicHilbert_sub_truncatedSpectralHilbert_eq
    (g : ℝ → ℝ) (hg : ContDiff ℝ 2 g) (hg_per : Function.Periodic g (2 * Real.pi))
    (N : ℕ) (u : ℝ) :
    periodicHilbert g u - truncatedSpectralHilbert N g u =
      Complex.re
        (∑' n : ℤ,
          (if (|n| : ℤ) < ((N / 2 : ℕ) : ℤ) then (0 : ℂ)
           else hilbertMult n *
                  fourierCoeffOn zero_lt_two_pi (gCircOfReal g) n *
                  Complex.exp ((n : ℂ) * Complex.I * (u : ℂ)))) := by
  -- The full series for periodicHilbert.
  have hsumm := summable_hilbertMult_fourierCoeff_exp g hg hg_per u
  -- Truncation summability: truncation = full minus low-mode complement,
  -- and the low-mode complement (with the `if` switched) is also summable
  -- since each term equals 0 outside a finite set.
  -- Strategy: split full = low-mode + high-mode; both are summable; periodicHilbert
  -- = Re full, dftHilbert = Re low-mode; difference = Re high-mode.
  -- Define the low-mode and high-mode parts. We use the *exact* if-condition
  -- shape `(|n| : ℤ) < ((N / 2 : ℕ) : ℤ)` to match `dftHilbert`'s definition.
  set fullTerm : ℤ → ℂ := fun n =>
      hilbertMult n *
        fourierCoeffOn zero_lt_two_pi (gCircOfReal g) n *
        Complex.exp ((n : ℂ) * Complex.I * (u : ℂ)) with hfullTerm_def
  set lowTerm : ℤ → ℂ := fun n =>
      if (|n| : ℤ) < ((N / 2 : ℕ) : ℤ) then fullTerm n else 0 with hlowTerm_def
  set highTerm : ℤ → ℂ := fun n =>
      if (|n| : ℤ) < ((N / 2 : ℕ) : ℤ) then 0 else fullTerm n with hhighTerm_def
  have hsplit : ∀ n, fullTerm n = lowTerm n + highTerm n := by
    intro n
    show fullTerm n = (if (|n| : ℤ) < ((N / 2 : ℕ) : ℤ) then fullTerm n else 0)
                    + (if (|n| : ℤ) < ((N / 2 : ℕ) : ℤ) then 0 else fullTerm n)
    by_cases hcond : (|n| : ℤ) < ((N / 2 : ℕ) : ℤ)
    · rw [if_pos hcond, if_pos hcond]; ring
    · rw [if_neg hcond, if_neg hcond]; ring
  -- Pointwise norm bound on lowTerm.
  have hlow_norm_le : ∀ n, ‖lowTerm n‖ ≤ ‖fullTerm n‖ := by
    intro n
    show ‖if (|n| : ℤ) < ((N / 2 : ℕ) : ℤ) then fullTerm n else 0‖ ≤ ‖fullTerm n‖
    by_cases hcond : (|n| : ℤ) < ((N / 2 : ℕ) : ℤ)
    · rw [if_pos hcond]
    · rw [if_neg hcond]; rw [norm_zero]; exact norm_nonneg _
  -- Summability of low and high parts.
  have hlow_summable : Summable lowTerm := by
    apply Summable.of_norm
    exact Summable.of_nonneg_of_le (fun n => norm_nonneg _) hlow_norm_le hsumm.norm
  have hhigh_summable : Summable highTerm := by
    -- highTerm n = fullTerm n - lowTerm n.
    have h := hsumm.sub hlow_summable
    refine h.congr (fun n => ?_)
    have hsp := hsplit n
    -- fullTerm n = lowTerm n + highTerm n ⇒ fullTerm n - lowTerm n = highTerm n.
    show fullTerm n - lowTerm n = highTerm n
    linear_combination hsp
  -- ∑' fullTerm = ∑' lowTerm + ∑' highTerm.
  have htsum_split :
      ∑' n : ℤ, fullTerm n = (∑' n : ℤ, lowTerm n) + (∑' n : ℤ, highTerm n) := by
    rw [← Summable.tsum_add hlow_summable hhigh_summable]
    exact tsum_congr hsplit
  -- periodicHilbert g u = Re (∑' fullTerm).
  have hPH_eq : periodicHilbert g u = Complex.re (∑' n : ℤ, fullTerm n) := rfl
  -- truncatedSpectralHilbert N g u = Re (∑' lowTerm).
  have hDFT_eq : truncatedSpectralHilbert N g u = Complex.re (∑' n : ℤ, lowTerm n) := rfl
  rw [hPH_eq, hDFT_eq, htsum_split]
  rw [Complex.add_re]
  ring

-- ===========================================================================
-- Step 2c: The packaging theorem (formerly the axiom)
-- ===========================================================================

/-- **Spectral-truncation tail bound (auxiliary).**

For `G ∈ C^p(ℝ)` 2π-periodic and `N ≥ 4` even, the spectral truncation
`truncatedSpectralHilbert N G` agrees with the true periodic Hilbert
transform `periodicHilbert G` up to a one-sided tail of `4 ×` the
Fourier coefficient norms with index `m ≥ N/2`:
```
∀ u, |truncatedSpectralHilbert N G u − periodicHilbert G u|
    ≤ 4 · ∑'_{k : ℕ} ‖c_{N/2 + k}(G)‖.
```

**Proof.** Subtract the two Fourier-series formulas to isolate the
high-mode tail. Take `|Re ·| ≤ ‖·‖` and `‖∑'‖ ≤ ∑' ‖·‖`. Each
high-mode term has norm `‖hilbertMult n‖ · ‖c_n(G)‖ · ‖e^{inu}‖
≤ ‖c_n(G)‖`. The sum over `|n| ≥ N/2` collapses to twice the one-sided
sum over `n ≥ N/2` by real-Fourier symmetry `‖c_{-n}‖ = ‖c_n‖`
(`norm_fourierCoeff_neg_eq`), yielding `2 · ∑'_{m ≥ N/2} ‖c_m(G)‖`.
The factor `4 ≥ 2` matches the bound the genuine sampling-DFT operator
satisfies (Stage 3 of the M4 chain). -/
theorem truncatedSpectralHilbert_diff_le_fourierTail
    (p : ℕ) (hp : 2 ≤ p)
    (N : ℕ) (hN : 4 ≤ N) (hN_even : Even N)
    (G : ℝ → ℝ)
    (hG : ContDiff ℝ p G) (hG_per : Function.Periodic G (2 * Real.pi)) :
    ∀ u, |truncatedSpectralHilbert N G u - periodicHilbert G u|
        ≤ 4 * ∑' k : ℕ,
              ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal G)
                  ((N / 2 : ℕ) + k : ℤ)‖ := by
  -- Reduce ContDiff p ⇒ ContDiff 2 (since p ≥ 2).
  intro u
  have hG_C2 : ContDiff ℝ 2 G := by
    -- ContDiff ℝ p G ⇒ ContDiff ℝ 2 G via `ContDiff.of_le` (since 2 ≤ p in ℕ ⊆ WithTop ℕ∞).
    obtain ⟨q, rfl⟩ : ∃ q : ℕ, p = q + 2 := ⟨p - 2, by omega⟩
    -- For p = q + 2, ContDiff ℝ (q + 2) G ⇒ ContDiff ℝ 2 G via `.of_le`.
    have hG' : ContDiff ℝ (q + 2 : ℕ) G := hG
    -- Both 2 and q+2 cast to WithTop ℕ∞; the inequality 2 ≤ q+2 holds in ℕ.
    have hle_nat : (2 : ℕ) ≤ q + 2 := Nat.le_add_left _ _
    have hcast_top : ((2 : ℕ) : WithTop ℕ∞) ≤ ((q + 2 : ℕ) : WithTop ℕ∞) := by
      have := WithTop.coe_le_coe.mpr (ENat.coe_le_coe.mpr hle_nat)
      convert this
    have hG2 : ContDiff ℝ ((2 : ℕ) : WithTop ℕ∞) G := hG'.of_le hcast_top
    -- Now reduce ContDiff ℝ ((2 : ℕ) : WithTop ℕ∞) G ⇒ ContDiff ℝ 2 G.
    have hnorm : ((2 : ℕ) : WithTop ℕ∞) = (2 : WithTop ℕ∞) := by norm_cast
    rw [hnorm] at hG2
    exact hG2
  -- N/2 ≥ 2 (since N ≥ 4).
  have hN_half : (2 : ℕ) ≤ N / 2 := by omega
  -- Abbreviation for c_n(G).
  set cG : ℤ → ℂ := fun n =>
      fourierCoeffOn zero_lt_two_pi (gCircOfReal G) n with hcG
  -- Step 1: difference equals the high-mode tail (Re).
  have hdiff := periodicHilbert_sub_truncatedSpectralHilbert_eq G hG_C2 hG_per N u
  have habs_eq :
      |truncatedSpectralHilbert N G u - periodicHilbert G u|
        = |periodicHilbert G u - truncatedSpectralHilbert N G u| := by
    rw [abs_sub_comm]
  rw [habs_eq, hdiff]
  -- Step 2: |Re S| ≤ ‖S‖.
  set S : ℂ := ∑' n : ℤ,
      (if (|n| : ℤ) < ((N / 2 : ℕ) : ℤ) then (0 : ℂ)
       else hilbertMult n * cG n *
              Complex.exp ((n : ℂ) * Complex.I * (u : ℂ))) with hS_def
  have hA : |S.re| ≤ ‖S‖ := Complex.abs_re_le_norm S
  -- Step 3: ‖S‖ ≤ ∑' ‖high-mode summand‖.
  -- Summability of the high-mode summand norms.
  have hsumm_full := summable_hilbertMult_fourierCoeff_exp G hG_C2 hG_per u
  have hsumm_high : Summable (fun n : ℤ =>
      if (|n| : ℤ) < ((N / 2 : ℕ) : ℤ) then (0 : ℂ)
      else hilbertMult n * cG n *
              Complex.exp ((n : ℂ) * Complex.I * (u : ℂ))) := by
    apply Summable.of_norm
    refine Summable.of_nonneg_of_le ?_ ?_ hsumm_full.norm
    · intro n; exact norm_nonneg _
    · intro n
      by_cases hcond : (|n| : ℤ) < ((N / 2 : ℕ) : ℤ)
      · rw [if_pos hcond, norm_zero]; exact norm_nonneg _
      · rw [if_neg hcond]
  have hsumm_norm := hsumm_high.norm
  have hB : ‖S‖ ≤ ∑' n : ℤ,
      ‖(if (|n| : ℤ) < ((N / 2 : ℕ) : ℤ) then (0 : ℂ)
        else hilbertMult n * cG n *
                Complex.exp ((n : ℂ) * Complex.I * (u : ℂ)))‖ := by
    rw [hS_def]
    exact norm_tsum_le_tsum_norm hsumm_norm
  -- Step 4: simplify the summand norms.
  have hnorm_eq : ∀ n : ℤ,
      ‖(if (|n| : ℤ) < ((N / 2 : ℕ) : ℤ) then (0 : ℂ)
        else hilbertMult n * cG n *
                Complex.exp ((n : ℂ) * Complex.I * (u : ℂ)))‖ =
      (if (|n| : ℤ) < ((N / 2 : ℕ) : ℤ) then (0 : ℝ) else ‖cG n‖) := by
    intro n
    by_cases hcond : (|n| : ℤ) < ((N / 2 : ℕ) : ℤ)
    · rw [if_pos hcond, if_pos hcond, norm_zero]
    · rw [if_neg hcond, if_neg hcond]
      rw [norm_mul, norm_mul]
      have hexp_norm :
          ‖Complex.exp ((n : ℂ) * Complex.I * (u : ℂ))‖ = 1 := by
        have h : (n : ℂ) * Complex.I * (u : ℂ) =
            (((n : ℝ) * u : ℝ) : ℂ) * Complex.I := by
          push_cast; ring
        rw [h, Complex.norm_exp_ofReal_mul_I]
      rw [hexp_norm, mul_one]
      -- For high modes (|n| ≥ N/2), n ≠ 0 (since N/2 ≥ 2 > 0).
      have hn_ne : n ≠ 0 := by
        intro hn0
        apply hcond
        subst hn0
        have : (|(0 : ℤ)| : ℤ) = 0 := by simp
        rw [this]
        have hN_half_pos : (0 : ℤ) < ((N / 2 : ℕ) : ℤ) := by
          exact_mod_cast (by omega : 0 < N / 2)
        linarith
      rw [norm_hilbertMult_of_ne_zero hn_ne, one_mul]
  -- Step 5: convert tsum to a one-sided sum via real-Fourier symmetry.
  -- ∑'_n (if |n| < N/2 then 0 else ‖c_n‖) = 2 · ∑'_{m ≥ N/2 (as ℕ-indexed)} ‖c_m‖.
  -- Split the ℤ-sum into ℕ + ℕ-of-negatives.
  set tail_ℕ : ℕ → ℝ := fun m =>
      ‖cG ((N / 2 : ℕ) + m : ℤ)‖ with htail_def
  set thunk : ℤ → ℝ := fun n =>
      (if (|n| : ℤ) < ((N / 2 : ℕ) : ℤ) then (0 : ℝ) else ‖cG n‖) with hthunk_def
  -- Summability of thunk.
  have hsumm_thunk : Summable thunk := by
    refine hsumm_norm.congr (fun n => ?_)
    exact hnorm_eq n
  -- Show: ∑'_n thunk n = 2 · ∑'_m tail_ℕ m.
  -- Split via `HasSum.of_nat_of_neg` (assuming summability).
  -- For n ≥ 0 (n : ℤ as ℕ-cast), `thunk (n : ℤ) = if n < N/2 then 0 else ‖c_n‖`.
  -- For n = N/2 + m (m ≥ 0), `thunk` = `‖c_{N/2 + m}‖`.
  -- For n ≤ -1 (n = -(m+1)), `thunk (-(m+1)) = if m+1 < N/2 then 0 else ‖c_{-(m+1)}‖`.
  -- Use real-Fourier symmetry: `‖c_{-(m+1)}‖ = ‖c_{m+1}‖`. So the negatives sum equals
  -- positives sum minus the n = 0 term (which is 0 since 0 < N/2 ≥ 2). Hence
  -- ∑'_ℤ thunk = 2 · ∑'_{n : ℕ} (if n < N/2 then 0 else ‖c_n‖).
  -- The one-sided-tail in the goal is ∑'_m ‖c_{N/2 + m}‖, which equals
  -- ∑'_{n : ℕ} (if n < N/2 then 0 else ‖c_n‖) by shift (n = N/2 + m).
  -- We package this as one lemma:
  have hkey : ∑' n : ℤ, thunk n = 2 * ∑' m : ℕ, tail_ℕ m := by
    -- ∑_ℤ thunk = ∑_ℕ thunk(n) + ∑_ℕ thunk(-n) - thunk(0).
    -- We use HasSum.of_nat_of_neg with HasSum.tsum_eq.
    -- First, build the ℕ-sided sums.
    have hpos_thunk_nn : ∀ n : ℕ, 0 ≤ thunk (n : ℤ) := by
      intro n
      simp only [thunk]
      split_ifs <;> [rfl; exact norm_nonneg _]
    have hneg_thunk_nn : ∀ n : ℕ, 0 ≤ thunk (-(n : ℤ)) := by
      intro n
      simp only [thunk]
      split_ifs <;> [rfl; exact norm_nonneg _]
    -- The ℕ-side of the split.
    have hpos_eq : ∀ n : ℕ, thunk (n : ℤ) =
        (if (n : ℤ) < (N / 2 : ℕ) then 0 else ‖cG (n : ℤ)‖) := by
      intro n
      simp only [thunk, hthunk_def]
      have h_abs : (|(n : ℤ)| : ℤ) = (n : ℤ) := by
        rw [abs_of_nonneg (by positivity : (0:ℤ) ≤ (n:ℤ))]
      rw [h_abs]
    have hneg_eq : ∀ n : ℕ, thunk (-(n : ℤ)) =
        (if (n : ℤ) < (N / 2 : ℕ) then 0 else ‖cG (-(n : ℤ))‖) := by
      intro n
      simp only [thunk, hthunk_def]
      have h_abs : (|(-(n : ℤ))| : ℤ) = (n : ℤ) := by
        rw [abs_neg, abs_of_nonneg (by positivity : (0:ℤ) ≤ (n:ℤ))]
      rw [h_abs]
    -- The negative-side norms equal positive-side norms by symmetry.
    have hneg_eq_pos : ∀ n : ℕ, ‖cG (-(n : ℤ))‖ = ‖cG (n : ℤ)‖ := by
      intro n
      have h := norm_fourierCoeff_neg_eq G (n : ℤ)
      change ‖cG (-(n : ℤ))‖ = ‖cG (n : ℤ)‖
      exact h
    -- thunk(-n) = thunk(n) by the above.
    have hneg_eq_pos_thunk : ∀ n : ℕ, thunk (-(n : ℤ)) = thunk (n : ℤ) := by
      intro n
      rw [hneg_eq n, hpos_eq n]
      split_ifs with h
      · rfl
      · rw [hneg_eq_pos]
    -- Build ∑'_ℕ thunk (n : ℤ) — denote it Sn.
    -- Build ∑'_ℕ thunk (-(n : ℤ)) — by hneg_eq_pos_thunk equals Sn.
    -- thunk 0 = 0 since 0 < N/2 (as N/2 ≥ 2).
    have hthunk0 : thunk 0 = 0 := by
      show (if (|(0 : ℤ)| : ℤ) < ((N / 2 : ℕ) : ℤ) then (0 : ℝ) else ‖cG 0‖) = 0
      have h_abs : (|(0 : ℤ)| : ℤ) = (0 : ℤ) := by simp
      have h_lt : (0 : ℤ) < ((N / 2 : ℕ) : ℤ) := by
        exact_mod_cast (by omega : 0 < N / 2)
      rw [h_abs, if_pos h_lt]
    -- Convert summability of thunk over ℤ to the ℕ-shape via Summable.int_natFn:
    -- Actually we use HasSum.of_nat_of_neg, requiring HasSum on both nat sides.
    have hsumm_nat : Summable (fun n : ℕ => thunk (n : ℤ)) := by
      apply hsumm_thunk.comp_injective
      intro a b hab
      exact_mod_cast hab
    have hsumm_neg : Summable (fun n : ℕ => thunk (-(n : ℤ))) := by
      refine Summable.congr hsumm_nat ?_
      intro n; exact (hneg_eq_pos_thunk n).symm
    -- Apply HasSum.of_nat_of_neg.
    have hhasSum_nat := hsumm_nat.hasSum
    have hhasSum_neg := hsumm_neg.hasSum
    have hhasSum_int := HasSum.of_nat_of_neg hhasSum_nat hhasSum_neg
    -- hhasSum_int : HasSum thunk (Sn + Sn - thunk 0) where Sn = ∑' n : ℕ, thunk (n : ℤ).
    have hSn_eq : ∑' n : ℕ, thunk (-(n : ℤ)) = ∑' n : ℕ, thunk (n : ℤ) := by
      refine tsum_congr ?_
      exact hneg_eq_pos_thunk
    rw [hSn_eq] at hhasSum_int
    rw [hthunk0, sub_zero] at hhasSum_int
    have hint_tsum : ∑' n : ℤ, thunk n = ∑' n : ℕ, thunk (n : ℤ) + ∑' n : ℕ, thunk (n : ℤ) :=
      hhasSum_int.tsum_eq
    rw [hint_tsum]
    -- Now reduce ∑' n : ℕ, thunk (n : ℤ) to ∑' m : ℕ, tail_ℕ m.
    have hSn_to_tail : ∑' n : ℕ, thunk (n : ℤ) = ∑' m : ℕ, tail_ℕ m := by
      -- thunk (n : ℤ) = 0 if n < N/2 else ‖cG n‖. We split the sum at N/2.
      -- A clean approach: reindex via n = N/2 + m.
      -- summable_nat_add_iff: Summable (fun n => f (n + k)) ↔ Summable f.
      have hreindex : ∀ m : ℕ, thunk (((N / 2 : ℕ) + m : ℕ) : ℤ) = tail_ℕ m := by
        intro m
        show (if (|(((N / 2 : ℕ) + m : ℕ) : ℤ)| : ℤ) < ((N / 2 : ℕ) : ℤ) then (0 : ℝ)
              else ‖cG (((N / 2 : ℕ) + m : ℕ) : ℤ)‖)
            = ‖cG ((N / 2 : ℕ) + m : ℤ)‖
        have h_abs : (|(((N / 2 : ℕ) + m : ℕ) : ℤ)| : ℤ) = (((N / 2 : ℕ) + m : ℕ) : ℤ) := by
          rw [abs_of_nonneg]; positivity
        rw [h_abs]
        have h_ge : ¬ (((N / 2 : ℕ) + m : ℕ) : ℤ) < ((N / 2 : ℕ) : ℤ) := by
          push_neg
          have heq : (((N / 2 : ℕ) + m : ℕ) : ℤ) = ((N / 2 : ℕ) : ℤ) + ((m : ℕ) : ℤ) := by
            push_cast; ring
          rw [heq]
          have hm : (0 : ℤ) ≤ ((m : ℕ) : ℤ) := by positivity
          linarith
        rw [if_neg h_ge]
        -- Goal: ‖cG (((N/2 + m : ℕ) : ℤ))‖ = ‖cG ((N/2 : ℕ) + m : ℤ)‖
        -- Both casts equal the same integer.
        have hint_eq : (((N / 2 : ℕ) + m : ℕ) : ℤ) = ((N / 2 : ℕ) + m : ℤ) := by
          push_cast; ring
        rw [hint_eq]
      have hsumm_thunk_nat_shift :
          Summable (fun m : ℕ => thunk (((N / 2 : ℕ) + m : ℕ) : ℤ)) := by
        -- This is the shifted summable_nat_add of `fun n => thunk (n : ℤ)`.
        have := (summable_nat_add_iff (N / 2)).mpr hsumm_nat
        refine this.congr (fun m => ?_)
        congr 1
        push_cast; ring
      -- Below: ∀ n < N/2, thunk (n : ℤ) = 0. So ∑'_{ℕ} thunk(n:ℤ) = ∑'_{m : ℕ} thunk((N/2+m):ℤ).
      have hzero_below : ∀ n : ℕ, n < N / 2 → thunk (n : ℤ) = 0 := by
        intro n hn
        rw [hpos_eq n]
        have hlt : (n : ℤ) < ((N / 2 : ℕ) : ℤ) := by exact_mod_cast hn
        rw [if_pos hlt]
      -- A standard reindexing: ∑'_{n : ℕ} f n = ∑'_{n < N/2} f n + ∑'_{m : ℕ} f (N/2 + m).
      -- We compute ∑'_{n < N/2} f n = 0 since each term is 0.
      have hsplit_at_M : ∑' n : ℕ, thunk (n : ℤ) =
          (∑ n ∈ Finset.range (N / 2), thunk (n : ℤ)) +
          ∑' m : ℕ, thunk (((N / 2 : ℕ) + m : ℕ) : ℤ) := by
        rw [← Summable.sum_add_tsum_nat_add (N / 2) hsumm_nat]
        congr 1
        refine tsum_congr (fun m => ?_)
        congr 1
        push_cast
        ring
      have hfinsum_zero : ∑ n ∈ Finset.range (N / 2), thunk (n : ℤ) = 0 := by
        apply Finset.sum_eq_zero
        intro n hn
        rw [Finset.mem_range] at hn
        exact hzero_below n hn
      rw [hsplit_at_M, hfinsum_zero, zero_add]
      refine tsum_congr ?_
      exact hreindex
    rw [hSn_to_tail]
    ring
  -- Step 6: combine everything.
  -- |Re S| ≤ ‖S‖ ≤ ∑'_n ‖summand‖ = ∑'_n thunk n = 2 · ∑'_m tail_ℕ m ≤ 4 · ∑'_m tail_ℕ m.
  have hsum_eq : ∑' n : ℤ, ‖(if (|n| : ℤ) < ((N / 2 : ℕ) : ℤ) then (0 : ℂ)
                            else hilbertMult n * cG n *
                                    Complex.exp ((n : ℂ) * Complex.I * (u : ℂ)))‖
                = ∑' n : ℤ, thunk n := tsum_congr hnorm_eq
  -- ∑' tail_ℕ m ≥ 0.
  have htail_nn : (0 : ℝ) ≤ ∑' m : ℕ, tail_ℕ m := by
    apply tsum_nonneg
    intro m
    simp [tail_ℕ]
  calc |S.re|
      ≤ ‖S‖ := hA
    _ ≤ ∑' n : ℤ, ‖(if (|n| : ℤ) < ((N / 2 : ℕ) : ℤ) then (0 : ℂ)
                    else hilbertMult n * cG n *
                            Complex.exp ((n : ℂ) * Complex.I * (u : ℂ)))‖ := hB
    _ = ∑' n : ℤ, thunk n := hsum_eq
    _ = 2 * ∑' m : ℕ, tail_ℕ m := hkey
    _ ≤ 4 * ∑' m : ℕ, tail_ℕ m := by
        have : (2 : ℝ) * ∑' m : ℕ, tail_ℕ m ≤ 4 * ∑' m : ℕ, tail_ℕ m := by
          apply mul_le_mul_of_nonneg_right (by norm_num) htail_nn
        exact this

-- ===========================================================================
-- Step 3: Integral test on `∑' 1/(k + M + 1)^p`
-- ===========================================================================

/-- Antitonicity of `x ↦ x^{-p}` on `[1, ∞)` (and a fortiori on `[M, ∞)` for
`M ≥ 1`), with `p ≥ 1`. -/
lemma antitoneOn_inv_rpow {p : ℝ} (hp : 1 ≤ p) {a : ℝ} (ha : 1 ≤ a) :
    AntitoneOn (fun x : ℝ => x ^ (-p)) (Set.Ici a) := by
  intro x hx y hy hxy
  simp only [mem_Ici] at hx hy
  have hx_pos : (0 : ℝ) < x := lt_of_lt_of_le (by linarith : (0:ℝ) < 1) (le_trans ha hx)
  have hy_pos : (0 : ℝ) < y := lt_of_lt_of_le (by linarith : (0:ℝ) < 1) (le_trans ha hy)
  have hp_pos : 0 < p := by linarith
  -- x^{-p} = 1 / x^p decreasing in x for x > 0, p > 0.
  change y ^ (-p) ≤ x ^ (-p)
  rw [Real.rpow_neg hx_pos.le, Real.rpow_neg hy_pos.le]
  -- 1/y^p ≤ 1/x^p iff x^p ≤ y^p, since both are positive.
  rw [inv_le_inv₀ (Real.rpow_pos_of_pos hy_pos p) (Real.rpow_pos_of_pos hx_pos p)]
  exact Real.rpow_le_rpow hx_pos.le hxy hp_pos.le

/-- Integral evaluation: `∫_a^b x^{-p} dx = (a^{1-p} − b^{1-p})/(p−1)`
for `0 < a ≤ b` and `p > 1`. -/
lemma integral_inv_rpow_eq {p a b : ℝ} (hp : 1 < p) (ha : 0 < a) (hab : a ≤ b) :
    ∫ x in a..b, x ^ (-p) = (a ^ (1 - p) - b ^ (1 - p)) / (p - 1) := by
  have hzero_notmem : (0 : ℝ) ∉ Set.uIcc a b := by
    rw [Set.uIcc_of_le hab]
    intro hmem
    rw [Set.mem_Icc] at hmem
    linarith
  have hneg_p_ne : (-p) ≠ -1 := by linarith
  have hi := integral_rpow (a := a) (b := b) (r := -p) (Or.inr ⟨hneg_p_ne, hzero_notmem⟩)
  rw [hi]
  -- LHS = (b^(-p+1) - a^(-p+1)) / (-p+1).
  have hpm1_ne : p - 1 ≠ 0 := by linarith
  have h1mp_ne : (1 - p : ℝ) ≠ 0 := by linarith
  have heq1 : (-p + 1 : ℝ) = (1 - p) := by ring
  rw [heq1]
  -- Goal: (b^(1-p) - a^(1-p)) / (1 - p) = (a^(1-p) - b^(1-p)) / (p - 1).
  -- Multiply both sides by (1 - p) * (p - 1):
  rw [div_eq_div_iff h1mp_ne hpm1_ne]
  ring

/-- **Integral test.** For `M ≥ 1` natural and `p ≥ 2` natural:
`∑'_{k : ℕ} 1/(k + M + 1 : ℝ)^p ≤ M^{1-p}/(p-1)`.

Bound the partial sums by an integral; take a sup. -/
lemma tsum_inv_pow_tail_le {p : ℕ} (hp : 2 ≤ p) {M : ℕ} (hM : 1 ≤ M) :
    ∑' k : ℕ, 1 / ((k + M + 1 : ℕ) : ℝ) ^ p
      ≤ (M : ℝ) ^ (1 - p : ℝ) / ((p : ℝ) - 1) := by
  -- p ≥ 2 in real, so p > 1.
  have hp_pos : (0 : ℝ) < p := by
    have : (2 : ℝ) ≤ (p : ℝ) := by exact_mod_cast hp
    linarith
  have hp_ge_two_real : (2 : ℝ) ≤ (p : ℝ) := by exact_mod_cast hp
  have hp_gt_one : (1 : ℝ) < (p : ℝ) := by linarith
  have hpm1_pos : (0 : ℝ) < (p : ℝ) - 1 := by linarith
  -- M ≥ 1 in real.
  have hM_real : (1 : ℝ) ≤ (M : ℝ) := by exact_mod_cast hM
  have hM_pos : (0 : ℝ) < (M : ℝ) := by linarith
  -- M^(1-p) > 0.
  have hM_pow_pos : (0 : ℝ) < (M : ℝ) ^ (1 - p : ℝ) := Real.rpow_pos_of_pos hM_pos _
  -- Bound: M^(1-p)/(p-1) > 0.
  set C : ℝ := (M : ℝ) ^ (1 - p : ℝ) / ((p : ℝ) - 1) with hC_def
  have hC_pos : (0 : ℝ) ≤ C := by
    rw [hC_def]
    exact (div_nonneg hM_pow_pos.le hpm1_pos.le)
  -- Apply `Real.tsum_le_of_sum_range_le`.
  apply Real.tsum_le_of_sum_range_le
  · -- Nonneg term.
    intro k
    have hpos : (0 : ℝ) ≤ ((k + M + 1 : ℕ) : ℝ) ^ p := by positivity
    apply div_nonneg one_pos.le hpos
  · -- For each n, ∑_{k < n} 1/(k+M+1)^p ≤ C.
    intro n
    -- Apply AntitoneOn.sum_le_integral with f x = x^{-p}, x₀ = M, count = n.
    -- The lemma gives: ∑ i ∈ range n, f (M + (i + 1 : ℕ)) ≤ ∫_M^{M+n} f.
    have hM_le_Mn : M ≤ M + n := Nat.le_add_right _ _
    have hAnti : AntitoneOn (fun x : ℝ => x ^ (-(p : ℝ)))
        (Set.Icc (M : ℝ) ((M : ℝ) + (n : ℝ))) := by
      have hsub : Set.Icc (M : ℝ) ((M : ℝ) + (n : ℝ)) ⊆ Set.Ici (M : ℝ) :=
        fun x hx => (Set.mem_Icc.mp hx).1
      have hp_real_ge_one : (1 : ℝ) ≤ (p : ℝ) := by exact_mod_cast (by omega : 1 ≤ p)
      exact (antitoneOn_inv_rpow hp_real_ge_one hM_real).mono hsub
    have hint := hAnti.sum_le_integral (x₀ := (M : ℝ)) (a := n)
    -- hint : ∑ i ∈ range n, (M + (i+1:ℕ) : ℝ)^(-(p:ℝ)) ≤ ∫_M^{M+n} x^(-p) dx.
    -- Reindex/identify: (M + (i + 1 : ℕ) : ℝ) = ((i + M + 1 : ℕ) : ℝ) = ((k + M + 1 : ℕ) : ℝ).
    have hLHS_eq :
        ∑ i ∈ Finset.range n, ((M : ℝ) + ((i + 1 : ℕ) : ℝ)) ^ (-(p : ℝ))
          = ∑ k ∈ Finset.range n, 1 / ((k + M + 1 : ℕ) : ℝ) ^ p := by
      apply Finset.sum_congr rfl
      intro i _
      -- Need: (M + ((i+1):ℕ:ℝ))^(-(p:ℝ)) = 1 / ((k + M + 1 : ℕ) : ℝ)^p.
      have hM1pos : (0 : ℝ) < (M : ℝ) + ((i + 1 : ℕ) : ℝ) := by
        have : (0 : ℝ) ≤ ((i + 1 : ℕ) : ℝ) := by positivity
        linarith
      rw [Real.rpow_neg hM1pos.le, ← Real.rpow_natCast _ p]
      rw [one_div]
      congr 1
      push_cast
      ring
    rw [hLHS_eq] at hint
    -- Now convert the RHS integral using integral_inv_rpow_eq.
    have hM_le_Mn_real : (M : ℝ) ≤ (M : ℝ) + (n : ℝ) := by
      have : (0 : ℝ) ≤ (n : ℝ) := by positivity
      linarith
    have hint_val :
        ∫ x in (M : ℝ)..((M : ℝ) + (n : ℝ)), x ^ (-(p : ℝ))
          = ((M : ℝ) ^ (1 - (p : ℝ)) - ((M : ℝ) + (n : ℝ)) ^ (1 - (p : ℝ))) / ((p : ℝ) - 1) :=
      integral_inv_rpow_eq (p := (p : ℝ)) (a := (M : ℝ)) (b := (M : ℝ) + (n : ℝ))
        hp_gt_one hM_pos hM_le_Mn_real
    rw [hint_val] at hint
    -- Conclude: ∑ ≤ (M^{1-p} - (M+n)^{1-p})/(p-1) ≤ M^{1-p}/(p-1) = C.
    have hMn_pow_nonneg : (0 : ℝ) ≤ ((M : ℝ) + (n : ℝ)) ^ (1 - (p : ℝ)) := by
      have hMn_pos : (0 : ℝ) < (M : ℝ) + (n : ℝ) := by linarith
      exact (Real.rpow_pos_of_pos hMn_pos _).le
    have h_drop :
        ((M : ℝ) ^ (1 - (p : ℝ)) - ((M : ℝ) + (n : ℝ)) ^ (1 - (p : ℝ))) / ((p : ℝ) - 1)
          ≤ (M : ℝ) ^ (1 - (p : ℝ)) / ((p : ℝ) - 1) := by
      apply div_le_div_of_nonneg_right _ hpm1_pos.le
      linarith
    calc ∑ k ∈ Finset.range n, 1 / ((k + M + 1 : ℕ) : ℝ) ^ p
        ≤ ((M : ℝ) ^ (1 - (p : ℝ)) - ((M : ℝ) + (n : ℝ)) ^ (1 - (p : ℝ))) / ((p : ℝ) - 1) := hint
      _ ≤ (M : ℝ) ^ (1 - (p : ℝ)) / ((p : ℝ) - 1) := h_drop

-- ===========================================================================
-- Step 4: Summability + bound on the Fourier-coefficient one-sided tail
-- ===========================================================================

/-- Summability of the one-sided Fourier-coefficient tail `‖c_{N/2 + k}(G)‖`,
indexed by `k : ℕ`. Follows from `‖c_n(G)‖ ≤ Cp/|n|^p` (the intermediate
axiom) and `∑'_k 1/(k+M+1)^p < ∞` for `p ≥ 2`. -/
lemma summable_fourierCoeff_tail
    (p : ℕ) (hp : 2 ≤ p)
    (N : ℕ) (hN : 4 ≤ N) (hN_even : Even N)
    (G : ℝ → ℝ)
    (hG : ContDiff ℝ p G) (hG_per : Function.Periodic G (2 * Real.pi))
    (Cp : ℝ) (hCp : ∀ u, |iteratedDeriv p G u| ≤ Cp) :
    Summable (fun k : ℕ =>
      ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal G)
          ((N / 2 : ℕ) + k : ℤ)‖) := by
  -- N/2 ≥ 2, so (N/2 + k : ℕ) ≥ 2 > 0 for every k.
  have hN_half : (2 : ℕ) ≤ N / 2 := by omega
  -- Compare to 1/(N/2 + k)^p via the coefficient bound, then to summable 1/(k + (N/2) - 1 + 1)^p.
  -- More cleanly: bound by Cp / ((N/2 + k):ℝ)^p, which is summable.
  have hCp_nn : 0 ≤ Cp := le_trans (abs_nonneg _) (hCp 0)
  have hp_ge_one : (1 : ℕ) ≤ p := by omega
  -- Pointwise bound.
  have hbound : ∀ k : ℕ,
      ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal G)
          ((N / 2 : ℕ) + k : ℤ)‖
        ≤ Cp / ((N / 2 + k : ℕ) : ℝ) ^ p := by
    intro k
    have hne : ((N / 2 : ℕ) + k : ℤ) ≠ 0 := by
      have hpos : 0 < ((N / 2 : ℕ) + k : ℤ) := by
        have : (0 : ℤ) < (N / 2 : ℕ) := by exact_mod_cast (by omega : 0 < N / 2)
        omega
      omega
    have h := fourierCoeff_norm_le_Cp_div_pow p hp G hG hG_per Cp hCp hne
    -- `|((N/2 + k):ℤ):ℝ| = ((N/2 + k):ℕ):ℝ`
    have hpos_int : (0 : ℤ) < ((N / 2 : ℕ) + k : ℤ) := by
      have : (0 : ℤ) < (N / 2 : ℕ) := by exact_mod_cast (by omega : 0 < N / 2)
      omega
    have habs :
        |(((N / 2 : ℕ) + k : ℤ) : ℝ)| = ((N / 2 + k : ℕ) : ℝ) := by
      rw [abs_of_pos (show (0:ℝ) < (((N / 2 : ℕ) + k : ℤ) : ℝ) from by exact_mod_cast hpos_int)]
      have hint_eq : (((N / 2 : ℕ) + k : ℤ) : ℝ) = ((N / 2 + k : ℕ) : ℝ) := by
        push_cast
        rfl
      exact hint_eq
    rw [habs] at h
    exact h
  -- Summability of the bounding series via comparison to summable_one_div_nat_pow on the shift.
  -- ∑' k, Cp / (N/2 + k)^p — reindex j = N/2 + k.
  -- Use Summable.of_nonneg_of_le with comparator 1/(k + (N/2))^p (we shift the series).
  -- Direct path: use the comparison lemma with f = ‖c_{N/2+k}‖ and g = Cp/(k+N/2)^p.
  refine Summable.of_nonneg_of_le (fun k => norm_nonneg _) hbound ?_
  -- Summability of Cp / (N/2 + k)^p over k : ℕ.
  -- That's Cp * (1/(N/2+k)^p), and the latter is the tail of summable 1/n^p.
  have h_one_div_summable : Summable (fun k : ℕ => 1 / ((N / 2 + k : ℕ) : ℝ) ^ p) := by
    -- ∑'_{k} 1/(N/2 + k)^p is just a shift of ∑'_n 1/n^p, which is summable for p ≥ 2.
    have hsumm_full : Summable (fun n : ℕ => 1 / (n : ℝ) ^ p) :=
      summable_one_div_nat_pow.mpr (by omega : 1 < p)
    -- Shift by N/2.
    have := (summable_nat_add_iff (N/2)).mpr hsumm_full
    refine this.congr (fun k => ?_)
    -- (1 / ((n + N/2):ℝ)^p) when n = k corresponds to (1 / (N/2 + k))^p.
    congr 1
    push_cast
    ring
  exact h_one_div_summable.mul_left Cp |>.congr (fun k => by
    rw [mul_one_div])

/-- The bounding tail-tsum estimate: with `M = N/2 − 1` and `p ≥ 2`,
`∑'_k ‖c_{N/2 + k}(G)‖ ≤ Cp · (N/2 − 1)^{1−p}/(p−1)`. -/
lemma tsum_fourierCoeff_tail_le
    (p : ℕ) (hp : 2 ≤ p)
    (N : ℕ) (hN : 4 ≤ N) (hN_even : Even N)
    (G : ℝ → ℝ)
    (hG : ContDiff ℝ p G) (hG_per : Function.Periodic G (2 * Real.pi))
    (Cp : ℝ) (hCp : ∀ u, |iteratedDeriv p G u| ≤ Cp) :
    ∑' k : ℕ,
        ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal G)
            ((N / 2 : ℕ) + k : ℤ)‖
      ≤ Cp * ((N / 2 - 1 : ℕ) : ℝ) ^ (1 - p : ℝ) / ((p : ℝ) - 1) := by
  -- Set M = N/2 - 1 so M ≥ 1 (since N ≥ 4 ⇒ N/2 ≥ 2 ⇒ M ≥ 1).
  have hN_half : (2 : ℕ) ≤ N / 2 := by omega
  set M : ℕ := N / 2 - 1 with hM_def
  have hM_pos : 1 ≤ M := by omega
  have hM_real : (1 : ℝ) ≤ (M : ℝ) := by exact_mod_cast hM_pos
  -- Useful: N/2 = M + 1.
  have hN_half_eq : (N / 2 : ℕ) = M + 1 := by omega
  -- Bridge: ((N/2 + k):ℕ) = ((k + M + 1):ℕ).
  have hM_pow_pos : (0 : ℝ) < (M : ℝ) ^ (1 - p : ℝ) :=
    Real.rpow_pos_of_pos (by linarith : (0 : ℝ) < (M : ℝ)) _
  have hCp_nn : 0 ≤ Cp := le_trans (abs_nonneg _) (hCp 0)
  have hp_ge_one : (1 : ℕ) ≤ p := by omega
  have hp_pos_real : (0 : ℝ) < (p : ℝ) := by
    have : (2 : ℝ) ≤ (p : ℝ) := by exact_mod_cast hp
    linarith
  have hpm1_pos : (0 : ℝ) < (p : ℝ) - 1 := by
    have : (2 : ℝ) ≤ (p : ℝ) := by exact_mod_cast hp
    linarith
  -- Step 1: pointwise bound ‖c_{N/2 + k}‖ ≤ Cp / ((N/2 + k):ℕ:ℝ)^p.
  have hpointwise : ∀ k : ℕ,
      ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal G)
          ((N / 2 : ℕ) + k : ℤ)‖
        ≤ Cp / ((N / 2 + k : ℕ) : ℝ) ^ p := by
    intro k
    have hne : ((N / 2 : ℕ) + k : ℤ) ≠ 0 := by
      have hpos : 0 < ((N / 2 : ℕ) + k : ℤ) := by
        have : (0 : ℤ) < (N / 2 : ℕ) := by exact_mod_cast (by omega : 0 < N / 2)
        omega
      omega
    have h := fourierCoeff_norm_le_Cp_div_pow p hp G hG hG_per Cp hCp hne
    have hpos_int : (0 : ℤ) < ((N / 2 : ℕ) + k : ℤ) := by
      have : (0 : ℤ) < (N / 2 : ℕ) := by exact_mod_cast (by omega : 0 < N / 2)
      omega
    have habs : |(((N / 2 : ℕ) + k : ℤ) : ℝ)| = ((N / 2 + k : ℕ) : ℝ) := by
      rw [abs_of_pos (show (0:ℝ) < (((N / 2 : ℕ) + k : ℤ) : ℝ) from by exact_mod_cast hpos_int)]
      have hint_eq : (((N / 2 : ℕ) + k : ℤ) : ℝ) = ((N / 2 + k : ℕ) : ℝ) := by
        push_cast; rfl
      exact hint_eq
    rw [habs] at h
    exact h
  -- Step 2: ∑' ‖c‖ ≤ ∑' Cp / ((N/2 + k):ℝ)^p.
  have hsumm_lhs := summable_fourierCoeff_tail p hp N hN hN_even G hG hG_per Cp hCp
  have hsumm_bound : Summable (fun k : ℕ => Cp / ((N / 2 + k : ℕ) : ℝ) ^ p) := by
    -- Reindex: ∑'_k Cp / ((N/2 + k):ℕ)^p = Cp · ∑'_k 1 / ((N/2 + k):ℕ)^p.
    have h_one_div_summable : Summable (fun k : ℕ => 1 / ((N / 2 + k : ℕ) : ℝ) ^ p) := by
      have hsumm_full : Summable (fun n : ℕ => 1 / (n : ℝ) ^ p) :=
        summable_one_div_nat_pow.mpr (by omega : 1 < p)
      have := (summable_nat_add_iff (N/2)).mpr hsumm_full
      refine this.congr (fun k => ?_)
      congr 1
      push_cast; ring
    exact h_one_div_summable.mul_left Cp |>.congr (fun k => by
      rw [mul_one_div])
  -- Apply tsum_le_tsum.
  have hstep1 :
      ∑' k : ℕ,
          ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal G)
              ((N / 2 : ℕ) + k : ℤ)‖
        ≤ ∑' k : ℕ, Cp / ((N / 2 + k : ℕ) : ℝ) ^ p :=
    hsumm_lhs.tsum_le_tsum hpointwise hsumm_bound
  -- Step 3: ∑' Cp / ((N/2 + k):ℝ)^p = Cp * ∑' 1 / ((k + M + 1):ℝ)^p (since N/2 = M + 1).
  have hreindex : ∀ k : ℕ,
      Cp / ((N / 2 + k : ℕ) : ℝ) ^ p
        = Cp * (1 / ((k + M + 1 : ℕ) : ℝ) ^ p) := by
    intro k
    have hidx_eq : ((N / 2 + k : ℕ) : ℝ) = ((k + M + 1 : ℕ) : ℝ) := by
      have heq_nat : (N / 2 + k : ℕ) = (k + M + 1 : ℕ) := by omega
      exact_mod_cast heq_nat
    rw [hidx_eq, mul_one_div]
  have hstep2 :
      ∑' k : ℕ, Cp / ((N / 2 + k : ℕ) : ℝ) ^ p
        = Cp * ∑' k : ℕ, 1 / ((k + M + 1 : ℕ) : ℝ) ^ p := by
    rw [tsum_congr hreindex, tsum_mul_left]
  -- Step 4: apply tsum_inv_pow_tail_le.
  have hbound_inner : ∑' k : ℕ, 1 / ((k + M + 1 : ℕ) : ℝ) ^ p
        ≤ (M : ℝ) ^ (1 - p : ℝ) / ((p : ℝ) - 1) :=
    tsum_inv_pow_tail_le hp hM_pos
  -- Chain: ∑' ‖c‖ ≤ ∑' Cp/(N/2+k)^p = Cp * ∑' 1/(k+M+1)^p ≤ Cp * M^{1-p}/(p-1).
  calc ∑' k : ℕ,
          ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal G)
              ((N / 2 : ℕ) + k : ℤ)‖
      ≤ ∑' k : ℕ, Cp / ((N / 2 + k : ℕ) : ℝ) ^ p := hstep1
    _ = Cp * ∑' k : ℕ, 1 / ((k + M + 1 : ℕ) : ℝ) ^ p := hstep2
    _ ≤ Cp * ((M : ℝ) ^ (1 - p : ℝ) / ((p : ℝ) - 1)) :=
        mul_le_mul_of_nonneg_left hbound_inner hCp_nn
    _ = Cp * (M : ℝ) ^ (1 - p : ℝ) / ((p : ℝ) - 1) := by ring

-- ===========================================================================
-- Step 5: Main theorem
-- ===========================================================================

/-- **Spectral-truncation aliasing bound (auxiliary).**

For `G ∈ C^p(𝕋)` 2π-periodic with `‖G^(p)‖_∞ ≤ Cp`, `p ≥ 2`, `N ≥ 4` even,
the spectral-truncation `truncatedSpectralHilbert N G` agrees with
`periodicHilbert G` uniformly up to the explicit closed-form remainder
`4 Cp / ((p−1)(N/2 − 1)^{p−1})`.

This is the auxiliary "truncation-only" bound; the headline
`dft_hilbert_aliasing_bound` strengthens it to the genuine sampling-DFT
operator at grid points using the periodic Poisson identity. -/
theorem truncatedSpectralHilbert_aliasing_bound
    (p : ℕ) (hp : 2 ≤ p)
    (N : ℕ) (hN : 4 ≤ N) (hN_even : Even N)
    (G : ℝ → ℝ)
    (hG : ContDiff ℝ p G) (hG_per : Function.Periodic G (2 * Real.pi))
    (Cp : ℝ) (hCp : ∀ u, |iteratedDeriv p G u| ≤ Cp) :
    ∀ u, |truncatedSpectralHilbert N G u - periodicHilbert G u|
        ≤ 4 * Cp / ((p - 1 : ℝ) * ((N / 2 - 1 : ℝ) ^ (p - 1))) := by
  intro u
  -- Useful arithmetic facts.
  have hN_half : (2 : ℕ) ≤ N / 2 := by omega
  have hN_half_minus_one : 1 ≤ N / 2 - 1 := by omega
  have hM_real : (1 : ℝ) ≤ ((N / 2 - 1 : ℕ) : ℝ) := by exact_mod_cast hN_half_minus_one
  have hM_pos_real : (0 : ℝ) < ((N / 2 - 1 : ℕ) : ℝ) := by linarith
  have hCp_nn : 0 ≤ Cp := le_trans (abs_nonneg _) (hCp 0)
  have hp_ge_two_real : (2 : ℝ) ≤ (p : ℝ) := by exact_mod_cast hp
  have hp_gt_one : (1 : ℝ) < (p : ℝ) := by linarith
  have hpm1_pos : (0 : ℝ) < (p : ℝ) - 1 := by linarith
  -- Step 1: pointwise modulus ≤ 4 · tail of Fourier coefficients.
  have h1 :=
    truncatedSpectralHilbert_diff_le_fourierTail p hp N hN hN_even G hG hG_per u
  -- Step 2+3: bound the Fourier-coefficient tail.
  have h2 :=
    tsum_fourierCoeff_tail_le p hp N hN hN_even G hG hG_per Cp hCp
  -- Combine: 4 * tail ≤ 4 * (Cp · M^{1-p}/(p-1)).
  have hmult :
      4 * ∑' k : ℕ,
            ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal G)
                ((N / 2 : ℕ) + k : ℤ)‖
        ≤ 4 * (Cp * ((N / 2 - 1 : ℕ) : ℝ) ^ (1 - p : ℝ) / ((p : ℝ) - 1)) := by
    have : 4 * ∑' k : ℕ,
            ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal G)
                ((N / 2 : ℕ) + k : ℤ)‖
        ≤ 4 * (Cp * ((N / 2 - 1 : ℕ) : ℝ) ^ (1 - p : ℝ) / ((p : ℝ) - 1)) :=
      mul_le_mul_of_nonneg_left h2 (by norm_num : (0 : ℝ) ≤ 4)
    exact this
  -- Now express `M^{1-p:ℝ}` as `1/M^{p-1}` (since M > 0, p ≥ 2 ⇒ p - 1 ≥ 1 in ℕ).
  -- And the natural-cast `(N/2 - 1 : ℕ):ℝ = (N/2 - 1 : ℝ)` for N ≥ 2.
  have hM_cast :
      ((N / 2 - 1 : ℕ) : ℝ) = ((N : ℝ) / 2 - 1) := by
    have hN_even_nat : 2 ∣ N := hN_even.two_dvd
    rcases hN_even_nat with ⟨k, hk⟩
    -- N = 2k. N/2 = k. So (N/2 - 1 : ℕ) = k - 1.
    -- And ((N : ℝ)/2 - 1) = k - 1.
    have hk_ge_two : 2 ≤ k := by omega
    have hk_ge_one : 1 ≤ k := by omega
    have hN_div : N / 2 = k := by rw [hk]; omega
    rw [hN_div]
    have hk_real : ((k - 1 : ℕ) : ℝ) = (k : ℝ) - 1 := by
      rw [Nat.cast_sub hk_ge_one]
      simp
    rw [hk_real, hk]
    push_cast
    ring
  -- M^{1-p:ℝ} = M^{-(p-1):ℝ} = (M^{p-1})⁻¹ = 1 / M^{p-1}.
  have hpm1_nat : 1 ≤ p - 1 := by omega
  have hp_ge_one : 1 ≤ p := by omega
  have hpm1_real_eq : ((p - 1 : ℕ) : ℝ) = (p : ℝ) - 1 := by
    rw [Nat.cast_sub hp_ge_one]
    simp
  have hM_rpow_eq :
      ((N / 2 - 1 : ℕ) : ℝ) ^ (1 - (p : ℝ))
        = 1 / (((N : ℝ) / 2 - 1) ^ (p - 1)) := by
    rw [hM_cast]
    have hM2_pos : (0 : ℝ) < ((N : ℝ) / 2 - 1) := by
      rw [← hM_cast]; exact hM_pos_real
    have hrw_exp : (1 : ℝ) - (p : ℝ) = -((p : ℝ) - 1) := by ring
    rw [hrw_exp, Real.rpow_neg hM2_pos.le]
    rw [show ((p : ℝ) - 1) = ((p - 1 : ℕ) : ℝ) from hpm1_real_eq.symm]
    rw [Real.rpow_natCast]
    rw [one_div]
  -- Chain the inequalities.
  calc |truncatedSpectralHilbert N G u - periodicHilbert G u|
      ≤ 4 * ∑' k : ℕ,
              ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal G)
                  ((N / 2 : ℕ) + k : ℤ)‖ := h1
    _ ≤ 4 * (Cp * ((N / 2 - 1 : ℕ) : ℝ) ^ (1 - p : ℝ) / ((p : ℝ) - 1)) := hmult
    _ = 4 * Cp * ((N / 2 - 1 : ℕ) : ℝ) ^ (1 - p : ℝ) / ((p : ℝ) - 1) := by ring
    _ = 4 * Cp * (1 / (((N : ℝ) / 2 - 1) ^ (p - 1))) / ((p : ℝ) - 1) := by rw [hM_rpow_eq]
    _ = 4 * Cp / ((p - 1 : ℝ) * (((N : ℝ) / 2 - 1) ^ (p - 1))) := by
        field_simp

-- ===========================================================================
-- Stage 3: sampling-DFT aliasing bound at grid points
-- ===========================================================================

/-- **Alias**: `samplingDFTHilbert` is exactly `dftHilbert`. We introduce this
name to align with the Stage 3 theorem statement and the broader naming
convention "sampling-DFT Hilbert"; `dftHilbert` is the implementation
matching the Julia `BabenkoRigorous/src/FFTHilbert.jl` operator. -/
noncomputable def samplingDFTHilbert (N : ℕ) (G : ℝ → ℝ) : ℝ → ℝ :=
  dftHilbert N G

/-- **Index equivalence**: `ℤ ≃ Fin N × ℤ` via the Euclidean
quotient-remainder decomposition `n ↦ (n % N, n / N)`. The inverse is
`(k, m) ↦ k.val + m·N`. This is the bijection used to reindex the
`tsum ℤ` form of `samplingDFTHilbert` into the `Fin N`-finite sum × inner
`tsum ℤ` form (Poisson identity). -/
def Int.equivQuotMod (N : ℕ) (hN : 0 < N) : ℤ ≃ Fin N × ℤ where
  toFun n := (⟨(n % (N : ℤ)).toNat, by
    have h1 : 0 ≤ n % (N : ℤ) := Int.emod_nonneg _ (by exact_mod_cast hN.ne')
    have h2 : n % (N : ℤ) < (N : ℤ) := Int.emod_lt_of_pos _ (by exact_mod_cast hN)
    have htoNat : ((n % (N : ℤ)).toNat : ℤ) = n % (N : ℤ) := Int.toNat_of_nonneg h1
    have hlt : ((n % (N : ℤ)).toNat : ℤ) < (N : ℤ) := by rw [htoNat]; exact h2
    exact_mod_cast hlt⟩, n / (N : ℤ))
  invFun p := (p.1.val : ℤ) + p.2 * (N : ℤ)
  left_inv n := by
    simp only
    have h1 : 0 ≤ n % (N : ℤ) := Int.emod_nonneg _ (by exact_mod_cast hN.ne')
    have htoNat : ((n % (N : ℤ)).toNat : ℤ) = n % (N : ℤ) := Int.toNat_of_nonneg h1
    rw [htoNat]
    exact Int.emod_add_ediv_mul n N
  right_inv := fun ⟨k, m⟩ => by
    simp only
    have hN' : (N : ℤ) ≠ 0 := by exact_mod_cast hN.ne'
    have hk_lt : k.val < N := k.isLt
    have hmod : ((k.val : ℤ) + m * (N : ℤ)) % (N : ℤ) = (k.val : ℤ) := by
      rw [Int.add_mul_emod_self_right]
      exact Int.emod_eq_of_lt (by exact_mod_cast Nat.zero_le k.val)
        (by exact_mod_cast hk_lt)
    have hdiv : ((k.val : ℤ) + m * (N : ℤ)) / (N : ℤ) = m := by
      rw [Int.add_mul_ediv_right _ _ hN']
      have hk_div : (k.val : ℤ) / (N : ℤ) = 0 :=
        Int.ediv_eq_zero_of_lt (by exact_mod_cast Nat.zero_le k.val)
          (by exact_mod_cast hk_lt)
      rw [hk_div, zero_add]
    refine Prod.ext ?_ ?_
    · apply Fin.ext
      show ((((k.val : ℤ) + m * (N : ℤ)) % (N : ℤ)).toNat : ℕ) = k.val
      rw [hmod]
      simp
    · exact hdiv

/-- The exponential `exp(i · n · u_j)` at grid points `u_j = 2π j / N`
equals `exp(2π i · n · j / N)`. -/
lemma exp_grid_eq (N : ℕ) (n : ℂ) (j : Fin N) :
    Complex.exp (n * Complex.I * (fftGridPoint N j : ℂ))
      = Complex.exp (2 * Real.pi * Complex.I * n * (j.val : ℂ) / (N : ℂ)) := by
  congr 1
  push_cast [fftGridPoint]
  ring

/-- Summability of `‖c_n(G)‖` (over all `n : ℤ`, including `n = 0`) for `G ∈ C²`
2π-periodic. The companion `summable_fourierCoeff_norm_of_contDiff` returns the
zero-excluded form; here we package the un-excluded form. -/
lemma summable_fourierCoeff_norm_full_of_contDiff (g : ℝ → ℝ)
    (hg : ContDiff ℝ 2 g) (hg_per : Function.Periodic g (2 * Real.pi)) :
    Summable (fun n : ℤ => ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal g) n‖) := by
  set f0 : ℤ → ℝ := fun n =>
      if n = 0 then (0 : ℝ) else ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal g) n‖
  set f1 : ℤ → ℝ := fun n => ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal g) n‖
  have hf0_summ : Summable f0 := summable_fourierCoeff_norm_of_contDiff g hg hg_per
  -- f1 = f0 + (if n = 0 then ‖c_0‖ else 0); the second part is a singleton-supported
  -- finitely supported function, so summable.
  set delta : ℤ → ℝ := fun n =>
      if n = 0 then ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal g) 0‖ else 0
  have h_delta_summ : Summable delta := by
    refine summable_of_ne_finset_zero (s := ({0} : Finset ℤ)) ?_
    intro n hn
    have : n ≠ 0 := by
      intro h_eq
      apply hn
      rw [Finset.mem_singleton]
      exact h_eq
    simp [delta, this]
  have h_eq : ∀ n : ℤ, f1 n = f0 n + delta n := by
    intro n
    by_cases hn : n = 0
    · subst hn; simp [f0, f1, delta]
    · simp [f0, f1, delta, hn]
  exact (hf0_summ.add h_delta_summ).congr (fun n => (h_eq n).symm)

set_option maxHeartbeats 1200000 in
/-- **Helper: tsum-of-products form of `samplingDFTHilbert` at grid points.**

For `G ∈ C²(ℝ)` 2π-periodic with absolutely summable Fourier coefficients,
the sampling-DFT Hilbert at `u_j` admits the `tsum ℤ` representation
```
samplingDFTHilbert N G u_j = Re ∑'_{n : ℤ}
    hilbertMultDiscrete N n · c_n(G) · exp(2π i n j / N).
```

**Proof.** Substitute the periodic Poisson identity
`dftCoeff N G k = ∑'_m c_{k + m·N}` into each Finset summand, then collapse
the double sum via `ℤ ≃ Fin N × ℤ` using `n = k.val + m·N` (so
`k = n mod N` and `m = n / N`), using `hilbertMultDiscrete` periodicity
(`n mod N = (n + m·N) mod N`) and `exp_two_pi_I_grid_aliased` to align the
basis exponentials. -/
lemma samplingDFTHilbert_eq_tsum_int_grid
    (N : ℕ) (hN : 0 < N)
    (G : ℝ → ℝ)
    (hG_cont : Continuous G)
    (hG_per : Function.Periodic G (2 * Real.pi))
    (h_summ : Summable (fun n : ℤ =>
        ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal G) n‖))
    (j : Fin N) :
    samplingDFTHilbert N G (fftGridPoint N j)
      = Complex.re
          (∑' n : ℤ,
            hilbertMultDiscrete N n *
              fourierCoeffOn zero_lt_two_pi (gCircOfReal G) n *
              Complex.exp (2 * Real.pi * Complex.I * (n : ℂ) * (j.val : ℂ) / (N : ℂ))) := by
  set c : ℤ → ℂ := fun n => fourierCoeffOn zero_lt_two_pi (gCircOfReal G) n with hc_def
  -- Step 1: unfold `samplingDFTHilbert` to `dftHilbert` and simplify the exp at grid points.
  show dftHilbert N G (fftGridPoint N j) = _
  -- Step 2: write the goal LHS as a Finset sum with the grid-form exponential
  -- using the lifted-Int form `(↑(↑↑k : ℤ) : ℂ)` consistently (matches the cast
  -- pattern of `exp_two_pi_I_grid_aliased`).
  have h_dft_unfold : dftHilbert N G (fftGridPoint N j) =
      Complex.re
        (∑ k : Fin N,
          hilbertMultDiscrete N (k.val : ℤ) *
            dftCoeff N G (k.val : ℤ) *
            Complex.exp (2 * Real.pi * Complex.I * ((k.val : ℤ) : ℂ) *
              (j.val : ℂ) / (N : ℂ))) := by
    unfold dftHilbert
    congr 1
    refine Finset.sum_congr rfl (fun k _ => ?_)
    rw [exp_grid_eq N (k.val : ℂ) j]
    -- Match (k.val : ℂ) with ((k.val : ℤ) : ℂ) via cast normalization.
    push_cast
    rfl
  rw [h_dft_unfold]
  congr 1
  -- Step 3: substitute Poisson identity in each term: dftCoeff k = ∑' m, c (k + m·N).
  -- Per term equality (using push-cast form to align with Lean's auto-normalization).
  have h_per_term : ∀ k : Fin N,
      hilbertMultDiscrete N (k.val : ℤ) *
          dftCoeff N G (k.val : ℤ) *
          Complex.exp (2 * Real.pi * Complex.I * ((k.val : ℤ) : ℂ) *
            (j.val : ℂ) / (N : ℂ))
        = ∑' m : ℤ,
            (hilbertMultDiscrete N ((k.val : ℤ) + m * (N : ℤ)) *
                c ((k.val : ℤ) + m * (N : ℤ)) *
                Complex.exp (2 * Real.pi * Complex.I *
                  (((k.val : ℤ) : ℂ) + (m : ℂ) * ((N : ℤ) : ℂ)) *
                    (j.val : ℂ) / (N : ℂ))) := by
    intro k
    -- Inner each term: pulls out M^d (constant in m) and exp at index k.val (constant in m).
    -- Rewrite `dftCoeff k = ∑' m, c (k + mN)` and pull out the constants via tsum_mul_left/right.
    rw [dftCoeff_eq_aliased_sum N hN G hG_cont hG_per h_summ (k.val : ℤ)]
    -- LHS now: M^d(k.val) * (∑' m, c (k.val + m*N)) * exp(2πi k j / N).
    -- Use the fact that hilbertMultDiscrete N ((k.val) + m * N) = hilbertMultDiscrete N k.val
    -- and exp(2πi (k.val + mN) j / N) = exp(2πi k.val j / N) (grid aliasing) to pull
    -- those expressions inside.
    set M : ℂ := hilbertMultDiscrete N (k.val : ℤ) with hM_def
    set E : ℂ := Complex.exp (2 * Real.pi * Complex.I * ((k.val : ℤ) : ℂ) *
                                (j.val : ℂ) / (N : ℂ))
      with hE_def
    -- Identify the (constant-in-m) shifted forms.
    have h_M_inv : ∀ m : ℤ,
        hilbertMultDiscrete N ((k.val : ℤ) + m * (N : ℤ)) = M := by
      intro m
      rw [hM_def]
      unfold hilbertMultDiscrete
      have hmod_eq : ((k.val : ℤ) + m * (N : ℤ)) % (N : ℤ) = (k.val : ℤ) % (N : ℤ) := by
        rw [Int.add_mul_emod_self_right]
      rw [hmod_eq]
    -- Collapse per-term: the body of the tsum equals M * c (k + mN) * E.
    -- Per-term equality at m: each summand equals M * c(k + mN) * E.
    have h_inner : ∀ m : ℤ,
        hilbertMultDiscrete N ((k.val : ℤ) + m * (N : ℤ)) *
              c ((k.val : ℤ) + m * (N : ℤ)) *
              Complex.exp (2 * Real.pi * Complex.I *
                (((k.val : ℤ) : ℂ) + (m : ℂ) * ((N : ℤ) : ℂ)) *
                  (j.val : ℂ) / (N : ℂ))
          = M * c ((k.val : ℤ) + m * (N : ℤ)) * E := by
      intro m
      rw [h_M_inv m]
      congr 1
      rw [hE_def]
      exact exp_two_pi_I_grid_aliased_pushed N hN (k.val : ℤ) m j
    -- Step: rewrite the tsum body via h_inner.
    have h_tsum_eq : ∑' m : ℤ,
            (hilbertMultDiscrete N ((k.val : ℤ) + m * (N : ℤ)) *
                c ((k.val : ℤ) + m * (N : ℤ)) *
                Complex.exp (2 * Real.pi * Complex.I *
                  (((k.val : ℤ) + m * (N : ℤ)) : ℂ) * (j.val : ℂ) / (N : ℂ)))
        = ∑' m : ℤ, M * c ((k.val : ℤ) + m * (N : ℤ)) * E :=
      tsum_congr h_inner
    rw [h_tsum_eq]
    -- ∑' m, M * c (...) * E = M * (∑' m, c (...)) * E (constants in/out of tsum).
    have h_pull_M : ∑' m : ℤ, M * c ((k.val : ℤ) + m * (N : ℤ)) * E
        = ∑' m : ℤ, M * (c ((k.val : ℤ) + m * (N : ℤ)) * E) := by
      refine tsum_congr (fun m => ?_); ring
    rw [h_pull_M, tsum_mul_left]
    have h_pull_E : ∑' m : ℤ, c ((k.val : ℤ) + m * (N : ℤ)) * E
        = (∑' m : ℤ, c ((k.val : ℤ) + m * (N : ℤ))) * E := tsum_mul_right
    rw [h_pull_E]
    -- LHS = M * (∑' m, c (k + mN)) * E. RHS = M * ((∑' m, c (k + mN)) * E).
    ring
  rw [Finset.sum_congr rfl (fun k _ => h_per_term k)]
  -- Step 4: collapse ∑ k ∑' m via the equivalence ℤ ≃ Fin N × ℤ.
  -- The double sum equals ∑' p : Fin N × ℤ, [inner].
  -- And ∑' p : Fin N × ℤ, [inner] = ∑' n : ℤ, [n-expression] via the equiv.
  set F : ℤ → ℂ := fun n =>
      hilbertMultDiscrete N n * c n *
        Complex.exp (2 * Real.pi * Complex.I * (n : ℂ) * (j.val : ℂ) / (N : ℂ)) with hF_def
  -- Summability of F.
  have h_F_summ : Summable F := by
    apply Summable.of_norm
    refine Summable.of_nonneg_of_le ?_ ?_ h_summ
    · intro n; exact norm_nonneg _
    · intro n
      show ‖F n‖ ≤ ‖c n‖
      rw [hF_def]
      rw [norm_mul, norm_mul]
      have h_exp_norm :
          ‖Complex.exp (2 * Real.pi * Complex.I * (n : ℂ) * (j.val : ℂ) / (N : ℂ))‖ = 1 := by
        have h_arg : 2 * Real.pi * Complex.I * (n : ℂ) * (j.val : ℂ) / (N : ℂ)
            = (((2 * Real.pi * (n : ℝ) * (j.val : ℝ) / (N : ℝ)) : ℝ) : ℂ) * Complex.I := by
          push_cast; ring
        rw [h_arg, Complex.norm_exp_ofReal_mul_I]
      rw [h_exp_norm, mul_one]
      have h_M_le : ‖hilbertMultDiscrete N n‖ ≤ 1 := by
        set r := n % (N : ℤ) with hr_def
        show ‖(if r = 0 then (0 : ℂ)
               else if r = ((N / 2 : ℕ) : ℤ) then 0
               else if r < ((N / 2 : ℕ) : ℤ) then -Complex.I
               else Complex.I)‖ ≤ 1
        split_ifs <;> simp [Complex.norm_I]
      have h_cn_nn : 0 ≤ ‖c n‖ := norm_nonneg _
      calc ‖hilbertMultDiscrete N n‖ * ‖c n‖
          ≤ 1 * ‖c n‖ := mul_le_mul_of_nonneg_right h_M_le h_cn_nn
        _ = ‖c n‖ := one_mul _
  -- Use Equiv.tsum_eq: ∑' n : ℤ, F n = ∑' p : Fin N × ℤ, F ((Int.equivQuotMod N hN).symm p).
  -- Note (Int.equivQuotMod N hN).symm p = p.1.val + p.2 * N.
  have h_equiv_unfold : ∀ p : Fin N × ℤ,
      F ((Int.equivQuotMod N hN).symm p)
        = hilbertMultDiscrete N ((p.1.val : ℤ) + p.2 * (N : ℤ)) *
            c ((p.1.val : ℤ) + p.2 * (N : ℤ)) *
            Complex.exp (2 * Real.pi * Complex.I *
              (((p.1.val : ℤ) : ℂ) + (p.2 : ℂ) * ((N : ℤ) : ℂ)) *
                (j.val : ℂ) / (N : ℂ)) := by
    intro p
    -- (Int.equivQuotMod N hN).symm p unfolds (per `invFun`) to (p.1.val : ℤ) + p.2 * (N : ℤ).
    show F ((p.1.val : ℤ) + p.2 * (N : ℤ)) = _
    -- F evaluated yields the lifted-Int cast form `((... : ℤ) : ℂ)`; the RHS
    -- already has the push-cast form. Convert via push_cast.
    rw [hF_def]
    congr 1
    push_cast
    ring
  -- ∑' n : ℤ, F n = ∑' p, F ((equiv.symm) p), by Equiv.tsum_eq.
  have h_eq_via_equiv : ∑' n : ℤ, F n
      = ∑' p : Fin N × ℤ, F ((Int.equivQuotMod N hN).symm p) :=
    ((Int.equivQuotMod N hN).symm.tsum_eq F).symm
  rw [h_eq_via_equiv]
  -- Simplify each (equiv.symm) application.
  rw [tsum_congr h_equiv_unfold]
  -- Summability on Fin N × ℤ (push-cast form to match `h_equiv_unfold`).
  have h_prod_summ : Summable (fun p : Fin N × ℤ =>
      hilbertMultDiscrete N ((p.1.val : ℤ) + p.2 * (N : ℤ)) *
        c ((p.1.val : ℤ) + p.2 * (N : ℤ)) *
        Complex.exp (2 * Real.pi * Complex.I *
          (((p.1.val : ℤ) : ℂ) + (p.2 : ℂ) * ((N : ℤ) : ℂ)) *
            (j.val : ℂ) / (N : ℂ))) := by
    have h_sm : Summable (fun p : Fin N × ℤ => F ((Int.equivQuotMod N hN).symm p)) :=
      (Equiv.summable_iff (Int.equivQuotMod N hN).symm).mpr h_F_summ
    refine h_sm.congr (fun p => ?_)
    exact h_equiv_unfold p
  -- Apply tsum_prod' to split: ∑' p : Fin N × ℤ, g p = ∑ k : Fin N, ∑' m : ℤ, g (k, m).
  rw [h_prod_summ.tsum_prod]
  -- Replace ∑' (k : Fin N), ... with ∑ k : Fin N, ... (Fintype).
  rw [tsum_fintype]

/-- **Multiplier-disagreement form of the difference at grid points.**

The pointwise difference at `u_j` equals the real part of a `tsum ℤ` of the
multiplier-disagreement `(M^d_N(n) - M(n))` times the Fourier coefficient
and the basis exponential. This is the bridge from "sampling-DFT formula"
to "aliasing tail bound". -/
lemma samplingDFTHilbert_sub_periodicHilbert_grid_eq
    (N : ℕ) (hN : 0 < N)
    (G : ℝ → ℝ)
    (hG : ContDiff ℝ 2 G)
    (hG_per : Function.Periodic G (2 * Real.pi))
    (j : Fin N) :
    samplingDFTHilbert N G (fftGridPoint N j) - periodicHilbert G (fftGridPoint N j)
      = Complex.re
          (∑' n : ℤ,
            (hilbertMultDiscrete N n - hilbertMult n) *
              fourierCoeffOn zero_lt_two_pi (gCircOfReal G) n *
              Complex.exp (2 * Real.pi * Complex.I * (n : ℂ) * (j.val : ℂ) / (N : ℂ))) := by
  have hG_cont : Continuous G := hG.continuous
  have h_summ : Summable (fun n : ℤ =>
      ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal G) n‖) :=
    summable_fourierCoeff_norm_full_of_contDiff G hG hG_per
  -- Step 1: samplingDFTHilbert in tsum form.
  have h_dft :=
    samplingDFTHilbert_eq_tsum_int_grid N hN G hG_cont hG_per h_summ j
  -- Step 2: periodicHilbert in tsum form at u_j (with the grid-aligned exp).
  have h_PH : periodicHilbert G (fftGridPoint N j)
      = Complex.re
          (∑' n : ℤ,
            hilbertMult n *
              fourierCoeffOn zero_lt_two_pi (gCircOfReal G) n *
              Complex.exp (2 * Real.pi * Complex.I * (n : ℂ) * (j.val : ℂ) / (N : ℂ))) := by
    show Complex.re
        (∑' n : ℤ, hilbertMult n *
          fourierCoeffOn zero_lt_two_pi (gCircOfReal G) n *
          Complex.exp ((n : ℂ) * Complex.I * (fftGridPoint N j : ℂ))) = _
    congr 1
    refine tsum_congr (fun n => ?_)
    rw [exp_grid_eq N n j]
  rw [h_dft, h_PH]
  -- Step 3: subtract the Re's and combine the tsums.
  set c : ℤ → ℂ := fun n => fourierCoeffOn zero_lt_two_pi (gCircOfReal G) n with hc_def
  set E : ℤ → ℂ := fun n =>
      Complex.exp (2 * Real.pi * Complex.I * (n : ℂ) * (j.val : ℂ) / (N : ℂ)) with hE_def
  -- Summability of the two individual tsums (multiplier · c · E).
  have h_norm_E : ∀ n : ℤ, ‖E n‖ = 1 := by
    intro n
    show ‖Complex.exp (2 * Real.pi * Complex.I * (n : ℂ) * (j.val : ℂ) / (N : ℂ))‖ = 1
    have h_arg : 2 * Real.pi * Complex.I * (n : ℂ) * (j.val : ℂ) / (N : ℂ)
        = (((2 * Real.pi * (n : ℝ) * (j.val : ℝ) / (N : ℝ)) : ℝ) : ℂ) * Complex.I := by
      push_cast; ring
    rw [h_arg, Complex.norm_exp_ofReal_mul_I]
  have h_Md_sum : Summable (fun n : ℤ => hilbertMultDiscrete N n * c n * E n) := by
    apply Summable.of_norm
    refine Summable.of_nonneg_of_le ?_ ?_ h_summ
    · intro n; exact norm_nonneg _
    · intro n
      rw [norm_mul, norm_mul, h_norm_E n, mul_one]
      have h_M_le : ‖hilbertMultDiscrete N n‖ ≤ 1 := by
        set r := n % (N : ℤ) with hr_def
        show ‖(if r = 0 then (0 : ℂ)
               else if r = ((N / 2 : ℕ) : ℤ) then 0
               else if r < ((N / 2 : ℕ) : ℤ) then -Complex.I
               else Complex.I)‖ ≤ 1
        split_ifs <;> simp [Complex.norm_I]
      calc ‖hilbertMultDiscrete N n‖ * ‖c n‖
          ≤ 1 * ‖c n‖ := mul_le_mul_of_nonneg_right h_M_le (norm_nonneg _)
        _ = ‖c n‖ := one_mul _
  have h_M_sum : Summable (fun n : ℤ => hilbertMult n * c n * E n) := by
    apply Summable.of_norm
    refine Summable.of_nonneg_of_le ?_ ?_ h_summ
    · intro n; exact norm_nonneg _
    · intro n
      rw [norm_mul, norm_mul, h_norm_E n, mul_one]
      calc ‖hilbertMult n‖ * ‖c n‖
          ≤ 1 * ‖c n‖ := mul_le_mul_of_nonneg_right (norm_hilbertMult_le_one n) (norm_nonneg _)
        _ = ‖c n‖ := one_mul _
  -- Now: Re X - Re Y = Re (X - Y), and X - Y = ∑' n, (M^d(n) - M(n)) * c n * E n.
  rw [← Complex.sub_re]
  congr 1
  rw [← Summable.tsum_sub h_Md_sum h_M_sum]
  refine tsum_congr (fun n => ?_)
  show hilbertMultDiscrete N n * c n * E n - hilbertMult n * c n * E n
      = (hilbertMultDiscrete N n - hilbertMult n) * c n * E n
  ring

/-- **Stage 3 helper**: term-wise multiplier-disagreement collapse to one-sided tail.

Given the absolute summability of `‖c_n(G)‖`, the two-sided tsum of
`(if |n| < N/2 then 0 else 2·‖c_n‖)` collapses to `4 · ∑'_k ‖c_{N/2+k}‖`
via real-Fourier symmetry. -/
lemma tsum_disagreement_bound_collapses (G : ℝ → ℝ)
    (N : ℕ) (hN_even : Even N) (hN : 4 ≤ N)
    (h_summ : Summable (fun n : ℤ =>
        ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal G) n‖)) :
    ∑' n : ℤ,
        (if (|n| : ℤ) < ((N / 2 : ℕ) : ℤ) then (0 : ℝ)
         else 2 * ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal G) n‖)
      ≤ 4 * ∑' k : ℕ,
            ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal G)
                ((N / 2 : ℕ) + k : ℤ)‖ := by
  set c : ℤ → ℂ := fun n => fourierCoeffOn zero_lt_two_pi (gCircOfReal G) n with hc_def
  -- N/2 ≥ 2 ⇒ N/2 > 0.
  have hN_half : (2 : ℕ) ≤ N / 2 := by omega
  -- Define the disagreement bound function.
  set thunk : ℤ → ℝ := fun n =>
      (if (|n| : ℤ) < ((N / 2 : ℕ) : ℤ) then (0 : ℝ) else 2 * ‖c n‖) with hthunk_def
  -- Summability of thunk.
  have h_thunk_summ : Summable thunk := by
    refine Summable.of_nonneg_of_le ?_ ?_ (h_summ.mul_left 2)
    · intro n
      simp only [thunk]
      split_ifs <;> [rfl; positivity]
    · intro n
      simp only [thunk]
      split_ifs
      · positivity
      · rfl
  -- Bound: thunk(n) ≤ 2·‖c n‖ for all n (and thunk(0) = 0).
  -- Split: ∑' thunk = ∑'_ℕ thunk(n:ℤ) + ∑'_ℕ thunk(-n:ℤ) - thunk(0).
  -- By real-Fourier symmetry, thunk(-n:ℤ) = thunk(n:ℤ), so the integer sum is 2 · ∑'_ℕ.
  have h_thunk0 : thunk 0 = 0 := by
    show (if (|(0 : ℤ)| : ℤ) < ((N / 2 : ℕ) : ℤ) then (0 : ℝ) else 2 * ‖c 0‖) = 0
    have h_abs : (|(0 : ℤ)| : ℤ) = (0 : ℤ) := by simp
    have h_lt : (0 : ℤ) < ((N / 2 : ℕ) : ℤ) := by exact_mod_cast (by omega : 0 < N / 2)
    rw [h_abs, if_pos h_lt]
  have h_pos_eq : ∀ n : ℕ, thunk (n : ℤ) =
      (if (n : ℤ) < (N / 2 : ℕ) then 0 else 2 * ‖c (n : ℤ)‖) := by
    intro n
    simp only [thunk, hthunk_def]
    have h_abs : (|(n : ℤ)| : ℤ) = (n : ℤ) :=
      abs_of_nonneg (by positivity : (0:ℤ) ≤ (n:ℤ))
    rw [h_abs]
  have h_neg_eq : ∀ n : ℕ, thunk (-(n : ℤ)) =
      (if (n : ℤ) < (N / 2 : ℕ) then 0 else 2 * ‖c (-(n : ℤ))‖) := by
    intro n
    simp only [thunk, hthunk_def]
    have h_abs : (|(-(n : ℤ))| : ℤ) = (n : ℤ) := by
      rw [abs_neg, abs_of_nonneg (by positivity : (0:ℤ) ≤ (n:ℤ))]
    rw [h_abs]
  have h_neg_eq_pos : ∀ n : ℕ, thunk (-(n : ℤ)) = thunk (n : ℤ) := by
    intro n
    rw [h_neg_eq n, h_pos_eq n]
    split_ifs with h
    · rfl
    · have : ‖c (-(n : ℤ))‖ = ‖c (n : ℤ)‖ := by
        change ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal G) (-(n : ℤ))‖
            = ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal G) (n : ℤ)‖
        exact norm_fourierCoeff_neg_eq G (n : ℤ)
      rw [this]
  -- Sum split.
  have h_sum_nat : Summable (fun n : ℕ => thunk (n : ℤ)) := by
    apply h_thunk_summ.comp_injective
    intro a b hab
    exact_mod_cast hab
  have h_sum_neg : Summable (fun n : ℕ => thunk (-(n : ℤ))) :=
    h_sum_nat.congr (fun n => (h_neg_eq_pos n).symm)
  have h_hasSum_int := HasSum.of_nat_of_neg h_sum_nat.hasSum h_sum_neg.hasSum
  have h_Sn_eq : ∑' n : ℕ, thunk (-(n : ℤ)) = ∑' n : ℕ, thunk (n : ℤ) :=
    tsum_congr h_neg_eq_pos
  rw [h_Sn_eq] at h_hasSum_int
  rw [h_thunk0, sub_zero] at h_hasSum_int
  have h_int_eq : ∑' n : ℤ, thunk n = 2 * ∑' n : ℕ, thunk (n : ℤ) := by
    have h_eq_2sn : ∑' n : ℕ, thunk (n : ℤ) + ∑' n : ℕ, thunk (n : ℤ)
        = 2 * ∑' n : ℕ, thunk (n : ℤ) := by ring
    rw [← h_eq_2sn]
    exact h_hasSum_int.tsum_eq
  rw [h_int_eq]
  -- Now: 2 · ∑'_ℕ thunk(n) ≤ 4 · ∑'_k ‖c_{N/2+k}‖.
  -- ∑'_ℕ thunk(n) = ∑'_{n : ℕ}, 2·‖c_n‖ for n ≥ N/2 and 0 otherwise
  --              = 2 · ∑'_{n : ℕ, n ≥ N/2} ‖c_n‖
  --              = 2 · ∑'_k ‖c_{N/2+k}‖
  -- Hence the whole thing = 4 · ∑'_k ‖c_{N/2+k}‖.
  -- Tail equivalent form.
  set tail_ℕ : ℕ → ℝ := fun m => ‖c ((N / 2 : ℕ) + m : ℤ)‖ with htail_def
  have h_Sn_to_tail : ∑' n : ℕ, thunk (n : ℤ) = 2 * ∑' m : ℕ, tail_ℕ m := by
    have h_reindex : ∀ m : ℕ, thunk (((N / 2 : ℕ) + m : ℕ) : ℤ) = 2 * tail_ℕ m := by
      intro m
      rw [h_pos_eq ((N / 2 : ℕ) + m)]
      have h_ge : ¬ (((N / 2 : ℕ) + m : ℕ) : ℤ) < ((N / 2 : ℕ) : ℤ) := by
        push_neg
        have heq : (((N / 2 : ℕ) + m : ℕ) : ℤ) = ((N / 2 : ℕ) : ℤ) + ((m : ℕ) : ℤ) := by
          push_cast; ring
        rw [heq]
        have hm : (0 : ℤ) ≤ ((m : ℕ) : ℤ) := by positivity
        linarith
      rw [if_neg h_ge]
      -- Now LHS = 2 * ‖c ((N/2 + m : ℕ) : ℤ)‖. RHS = 2 * ‖c ((N/2 : ℕ) + m : ℤ)‖.
      have hint_eq : (((N / 2 : ℕ) + m : ℕ) : ℤ) = ((N / 2 : ℕ) + m : ℤ) := by
        push_cast; ring
      rw [hint_eq]
    -- Split ∑'_n thunk(n:ℤ) = (low : sum over n < N/2 = 0) + (high : reindex).
    have h_low_zero : ∀ n : ℕ, n < N / 2 → thunk (n : ℤ) = 0 := by
      intro n hn
      rw [h_pos_eq n]
      have hlt : (n : ℤ) < ((N / 2 : ℕ) : ℤ) := by exact_mod_cast hn
      rw [if_pos hlt]
    have h_split : ∑' n : ℕ, thunk (n : ℤ)
        = (∑ n ∈ Finset.range (N / 2), thunk (n : ℤ)) +
          ∑' m : ℕ, thunk (((N / 2 : ℕ) + m : ℕ) : ℤ) := by
      rw [← Summable.sum_add_tsum_nat_add (N / 2) h_sum_nat]
      congr 1
      refine tsum_congr (fun m => ?_)
      congr 1
      push_cast; ring
    have h_low_sum_zero : ∑ n ∈ Finset.range (N / 2), thunk (n : ℤ) = 0 := by
      apply Finset.sum_eq_zero
      intro n hn
      rw [Finset.mem_range] at hn
      exact h_low_zero n hn
    rw [h_split, h_low_sum_zero, zero_add]
    rw [tsum_congr h_reindex]
    rw [tsum_mul_left]
  rw [h_Sn_to_tail]
  -- 2 · (2 · ∑'_m tail) = 4 · ∑'_m tail.
  ring_nf
  rfl

/-- **Stage 3 main theorem (M4 aliasing bound at grid points).**

For `G ∈ C^p(ℝ)` 2π-periodic with `p ≥ 2` and `N ≥ 4` even, the sampling
DFT Hilbert transform at the grid points `u_j = 2π j / N` differs from the
true periodic Hilbert transform by at most `4 ×` the one-sided tail of the
Fourier coefficient norms with index `m ≥ N/2`:
```
∀ j : Fin N,
  |samplingDFTHilbert N G u_j − periodicHilbert G u_j|
    ≤ 4 · ∑'_{k : ℕ} ‖c_{N/2 + k}(G)‖.
```

**Proof.** Combine the helpers:
1. `samplingDFTHilbert_sub_periodicHilbert_grid_eq`: difference equals
   `Re ∑'_n (M^d(n) − M(n)) · c_n(G) · exp(2π i n j / N)`.
2. `|Re X| ≤ ‖X‖`, `‖∑'‖ ≤ ∑' ‖·‖` (absolute summability of the integrand
   follows from summable Fourier coefficients × ≤2 multiplier disagreement).
3. Multiplier disagreement: `‖M^d(n) − M(n)‖ ≤ 2` for all `n`, and
   `= 0` for `|n| < N/2`.
4. `tsum_disagreement_bound_collapses`: two-sided tail collapses to
   `4 × one-sided tail` via real-Fourier symmetry. -/
theorem samplingDFTHilbert_diff_le_aliasingTail_grid
    (p : ℕ) (hp : 2 ≤ p)
    (N : ℕ) (hN : 4 ≤ N) (hN_even : Even N)
    (G : ℝ → ℝ)
    (hG : ContDiff ℝ p G) (hG_per : Function.Periodic G (2 * Real.pi)) :
    ∀ j : Fin N,
      |samplingDFTHilbert N G (fftGridPoint N j) - periodicHilbert G (fftGridPoint N j)|
        ≤ 4 * ∑' k : ℕ,
            ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal G)
                ((N / 2 : ℕ) + k : ℤ)‖ := by
  intro j
  -- Reduce ContDiff p ⇒ ContDiff 2.
  have hG_C2 : ContDiff ℝ 2 G := by
    obtain ⟨q, rfl⟩ : ∃ q : ℕ, p = q + 2 := ⟨p - 2, by omega⟩
    have hG' : ContDiff ℝ (q + 2 : ℕ) G := hG
    have hle_nat : (2 : ℕ) ≤ q + 2 := Nat.le_add_left _ _
    have hcast_top : ((2 : ℕ) : WithTop ℕ∞) ≤ ((q + 2 : ℕ) : WithTop ℕ∞) := by
      have := WithTop.coe_le_coe.mpr (ENat.coe_le_coe.mpr hle_nat)
      convert this
    have hG2 : ContDiff ℝ ((2 : ℕ) : WithTop ℕ∞) G := hG'.of_le hcast_top
    have hnorm : ((2 : ℕ) : WithTop ℕ∞) = (2 : WithTop ℕ∞) := by norm_cast
    rw [hnorm] at hG2
    exact hG2
  have hNpos : 0 < N := by omega
  set c : ℤ → ℂ := fun n => fourierCoeffOn zero_lt_two_pi (gCircOfReal G) n with hc_def
  set E : ℤ → ℂ := fun n =>
      Complex.exp (2 * Real.pi * Complex.I * (n : ℂ) * (j.val : ℂ) / (N : ℂ)) with hE_def
  -- Summability of Fourier coefficient norms.
  have h_summ : Summable (fun n : ℤ => ‖c n‖) :=
    summable_fourierCoeff_norm_full_of_contDiff G hG_C2 hG_per
  -- Step 1: Identify the difference.
  have h_diff := samplingDFTHilbert_sub_periodicHilbert_grid_eq N hNpos G hG_C2 hG_per j
  rw [h_diff]
  -- Step 2: Set S = ∑' n, (M^d(n) - M(n)) * c(n) * E(n).
  set S : ℂ := ∑' n : ℤ,
      (hilbertMultDiscrete N n - hilbertMult n) * c n * E n with hS_def
  -- Step 3: |Re S| ≤ ‖S‖.
  have hA : |S.re| ≤ ‖S‖ := Complex.abs_re_le_norm S
  -- Step 4: ‖S‖ ≤ ∑' n, ‖summand n‖.
  have h_norm_E : ∀ n : ℤ, ‖E n‖ = 1 := by
    intro n
    show ‖Complex.exp (2 * Real.pi * Complex.I * (n : ℂ) * (j.val : ℂ) / (N : ℂ))‖ = 1
    have h_arg : 2 * Real.pi * Complex.I * (n : ℂ) * (j.val : ℂ) / (N : ℂ)
        = (((2 * Real.pi * (n : ℝ) * (j.val : ℝ) / (N : ℝ)) : ℝ) : ℂ) * Complex.I := by
      push_cast; ring
    rw [h_arg, Complex.norm_exp_ofReal_mul_I]
  -- The pointwise norm bound.
  have h_summand_norm : ∀ n : ℤ,
      ‖(hilbertMultDiscrete N n - hilbertMult n) * c n * E n‖
        ≤ (if (|n| : ℤ) < ((N / 2 : ℕ) : ℤ) then (0 : ℝ) else 2 * ‖c n‖) := by
    intro n
    rw [norm_mul, norm_mul, h_norm_E n, mul_one]
    by_cases hcond : (|n| : ℤ) < ((N / 2 : ℕ) : ℤ)
    · rw [if_pos hcond]
      have h_zero := hilbertMult_disagreement_eq_zero_of_principal N hN_even hN n hcond
      rw [h_zero, norm_zero, zero_mul]
      -- goal: 0 ≤ 0
    · rw [if_neg hcond]
      have h_le_2 := hilbertMult_disagreement_le_two N n
      exact mul_le_mul_of_nonneg_right h_le_2 (norm_nonneg _)
  -- Summability of (M^d(n) - M(n)) * c(n) * E(n).
  have h_S_summ : Summable (fun n : ℤ =>
      (hilbertMultDiscrete N n - hilbertMult n) * c n * E n) := by
    apply Summable.of_norm
    refine Summable.of_nonneg_of_le ?_ ?_ (h_summ.mul_left 2)
    · intro n; exact norm_nonneg _
    · intro n
      have h := h_summand_norm n
      have h_thunk_le : (if (|n| : ℤ) < ((N / 2 : ℕ) : ℤ) then (0 : ℝ) else 2 * ‖c n‖)
          ≤ 2 * ‖c n‖ := by
        split_ifs
        · positivity
        · rfl
      exact le_trans h h_thunk_le
  -- Apply norm_tsum_le_tsum_norm.
  have hB : ‖S‖ ≤ ∑' n : ℤ, ‖(hilbertMultDiscrete N n - hilbertMult n) * c n * E n‖ := by
    rw [hS_def]
    exact norm_tsum_le_tsum_norm h_S_summ.norm
  -- Bound the inner ∑' ‖·‖ by the indicator form ∑' (if ... then 0 else 2·‖c n‖).
  have h_disag_summ : Summable (fun n : ℤ =>
      (if (|n| : ℤ) < ((N / 2 : ℕ) : ℤ) then (0 : ℝ) else 2 * ‖c n‖)) := by
    apply Summable.of_nonneg_of_le ?_ ?_ (h_summ.mul_left 2)
    · intro n; split_ifs <;> [rfl; positivity]
    · intro n
      split_ifs
      · positivity
      · rfl
  have hC : ∑' n : ℤ, ‖(hilbertMultDiscrete N n - hilbertMult n) * c n * E n‖
      ≤ ∑' n : ℤ,
          (if (|n| : ℤ) < ((N / 2 : ℕ) : ℤ) then (0 : ℝ) else 2 * ‖c n‖) :=
    h_S_summ.norm.tsum_le_tsum h_summand_norm h_disag_summ
  -- Apply the disagreement collapse helper.
  have hD := tsum_disagreement_bound_collapses G N hN_even hN h_summ
  -- Chain the bounds.
  calc |S.re|
      ≤ ‖S‖ := hA
    _ ≤ ∑' n : ℤ, ‖(hilbertMultDiscrete N n - hilbertMult n) * c n * E n‖ := hB
    _ ≤ ∑' n : ℤ,
          (if (|n| : ℤ) < ((N / 2 : ℕ) : ℤ) then (0 : ℝ) else 2 * ‖c n‖) := hC
    _ ≤ 4 * ∑' k : ℕ,
            ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal G)
                ((N / 2 : ℕ) + k : ℤ)‖ := hD

-- ===========================================================================
-- Stage 5: Closed-form headline theorem for the sampling-DFT operator
-- ===========================================================================

/-- **Stage 5 headline theorem (M4 aliasing bound for the sampling DFT
operator at grid points).**

For `G ∈ C^p(ℝ)` 2π-periodic with `‖G^(p)‖_∞ ≤ Cp`, `p ≥ 2` and `N ≥ 4`
even, the sampling-DFT Hilbert operator at grid points `u_j = 2π j / N`
matches the periodic Hilbert transform up to the explicit closed-form
remainder
```
∀ j : Fin N, |dftHilbert N G u_j − periodicHilbert G u_j|
                  ≤ 4 Cp / ((p − 1)(N/2 − 1)^{p−1}).
```

This is the analog of `truncatedSpectralHilbert_aliasing_bound` but for the
genuine sampling-DFT operator implemented in
`BabenkoRigorous/src/FFTHilbert.jl` (the Julia FFT routine), reached via
the periodic Poisson identity (Stage 2) and the multiplier-disagreement
analysis (Stage 3).

**Proof.** Chain
`samplingDFTHilbert_diff_le_aliasingTail_grid` (Stage 3) with the existing
N4 machinery: `tsum_fourierCoeff_tail_le` for the closed-form bound on
`∑'_k ‖c_{N/2+k}‖`, plus the arithmetic rearrangement
`M^{1-p}/(p-1) = 1/((p-1) · M^{p-1})` with `M = N/2 − 1`.

The arithmetic massage at the end is identical to the corresponding step
in `truncatedSpectralHilbert_aliasing_bound`. -/
theorem dft_hilbert_aliasing_bound
    (p : ℕ) (hp : 2 ≤ p)
    (N : ℕ) (hN : 4 ≤ N) (hN_even : Even N)
    (G : ℝ → ℝ)
    (hG : ContDiff ℝ p G) (hG_per : Function.Periodic G (2 * Real.pi))
    (Cp : ℝ) (hCp : ∀ u, |iteratedDeriv p G u| ≤ Cp) :
    ∀ j : Fin N,
      |dftHilbert N G (fftGridPoint N j) - periodicHilbert G (fftGridPoint N j)|
        ≤ 4 * Cp / ((p - 1 : ℝ) * ((N / 2 - 1 : ℝ) ^ (p - 1))) := by
  intro j
  -- Useful arithmetic facts.
  have hN_half : (2 : ℕ) ≤ N / 2 := by omega
  have hN_half_minus_one : 1 ≤ N / 2 - 1 := by omega
  have hM_real : (1 : ℝ) ≤ ((N / 2 - 1 : ℕ) : ℝ) := by exact_mod_cast hN_half_minus_one
  have hM_pos_real : (0 : ℝ) < ((N / 2 - 1 : ℕ) : ℝ) := by linarith
  have hCp_nn : 0 ≤ Cp := le_trans (abs_nonneg _) (hCp 0)
  have hp_ge_two_real : (2 : ℝ) ≤ (p : ℝ) := by exact_mod_cast hp
  have hp_gt_one : (1 : ℝ) < (p : ℝ) := by linarith
  have hpm1_pos : (0 : ℝ) < (p : ℝ) - 1 := by linarith
  -- Step 1: pointwise modulus ≤ 4 · tail of Fourier coefficients (Stage 3).
  have h1 :=
    samplingDFTHilbert_diff_le_aliasingTail_grid p hp N hN hN_even G hG hG_per j
  -- Replace samplingDFTHilbert with dftHilbert (definitionally equal).
  change |dftHilbert N G (fftGridPoint N j) - periodicHilbert G (fftGridPoint N j)|
      ≤ 4 * ∑' k : ℕ, ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal G)
            ((N / 2 : ℕ) + k : ℤ)‖ at h1
  -- Step 2: bound the Fourier-coefficient tail (N4 machinery).
  have h2 :=
    tsum_fourierCoeff_tail_le p hp N hN hN_even G hG hG_per Cp hCp
  -- Step 3: combine.
  have hmult :
      4 * ∑' k : ℕ,
            ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal G)
                ((N / 2 : ℕ) + k : ℤ)‖
        ≤ 4 * (Cp * ((N / 2 - 1 : ℕ) : ℝ) ^ (1 - p : ℝ) / ((p : ℝ) - 1)) :=
    mul_le_mul_of_nonneg_left h2 (by norm_num : (0 : ℝ) ≤ 4)
  -- Now express `M^{1-p:ℝ}` as `1/M^{p-1}`.
  have hM_cast :
      ((N / 2 - 1 : ℕ) : ℝ) = ((N : ℝ) / 2 - 1) := by
    have hN_even_nat : 2 ∣ N := hN_even.two_dvd
    rcases hN_even_nat with ⟨k, hk⟩
    have hk_ge_two : 2 ≤ k := by omega
    have hk_ge_one : 1 ≤ k := by omega
    have hN_div : N / 2 = k := by rw [hk]; omega
    rw [hN_div]
    have hk_real : ((k - 1 : ℕ) : ℝ) = (k : ℝ) - 1 := by
      rw [Nat.cast_sub hk_ge_one]
      simp
    rw [hk_real, hk]
    push_cast
    ring
  have hpm1_nat : 1 ≤ p - 1 := by omega
  have hp_ge_one : 1 ≤ p := by omega
  have hpm1_real_eq : ((p - 1 : ℕ) : ℝ) = (p : ℝ) - 1 := by
    rw [Nat.cast_sub hp_ge_one]
    simp
  have hM_rpow_eq :
      ((N / 2 - 1 : ℕ) : ℝ) ^ (1 - (p : ℝ))
        = 1 / (((N : ℝ) / 2 - 1) ^ (p - 1)) := by
    rw [hM_cast]
    have hM2_pos : (0 : ℝ) < ((N : ℝ) / 2 - 1) := by
      rw [← hM_cast]; exact hM_pos_real
    have hrw_exp : (1 : ℝ) - (p : ℝ) = -((p : ℝ) - 1) := by ring
    rw [hrw_exp, Real.rpow_neg hM2_pos.le]
    rw [show ((p : ℝ) - 1) = ((p - 1 : ℕ) : ℝ) from hpm1_real_eq.symm]
    rw [Real.rpow_natCast]
    rw [one_div]
  -- Chain the inequalities.
  calc |dftHilbert N G (fftGridPoint N j) - periodicHilbert G (fftGridPoint N j)|
      ≤ 4 * ∑' k : ℕ,
              ‖fourierCoeffOn zero_lt_two_pi (gCircOfReal G)
                  ((N / 2 : ℕ) + k : ℤ)‖ := h1
    _ ≤ 4 * (Cp * ((N / 2 - 1 : ℕ) : ℝ) ^ (1 - p : ℝ) / ((p : ℝ) - 1)) := hmult
    _ = 4 * Cp * ((N / 2 - 1 : ℕ) : ℝ) ^ (1 - p : ℝ) / ((p : ℝ) - 1) := by ring
    _ = 4 * Cp * (1 / (((N : ℝ) / 2 - 1) ^ (p - 1))) / ((p : ℝ) - 1) := by rw [hM_rpow_eq]
    _ = 4 * Cp / ((p - 1 : ℝ) * (((N : ℝ) / 2 - 1) ^ (p - 1))) := by
        field_simp

-- ===========================================================================
-- Stage 4: cot-kernel circular convolution form of `samplingDFTHilbert`
-- ===========================================================================
--
-- This stage establishes the equivalence between Lean's multiplier-form
-- `samplingDFTHilbert` and Julia's cot-kernel circular convolution form
-- (`BabenkoRigorous/src/FFTHilbert.jl` lines 120–135). The equivalence
-- is what justifies the Julia FFT routine's computation matching the
-- Lean specification.
--
-- This is *not* required for the headline trust chain
-- (`dft_hilbert_aliasing_bound`, `fft_hilbert_correct`,
-- `fft_hilbert_series_correct`) to be axiom-free; it bridges the Lean
-- abstraction to the Julia implementation.
--
-- **Status: fully proved.** The convolution-multiplier equivalence
-- theorem `samplingDFTHilbert_eq_cot_convolution` is proved later in
-- this file, via the closed-form sine sum identity
--   `∑_{k=1}^{N/2-1} sin(2πkn/N) = (1/2) · cot(πn/N) · (sin(πn/2))² · [n odd]`
-- (and `= 0` for `n` even) — `sum_sin_grid_eq_cotKernel`. (An earlier
-- comment here claimed the proof was deferred — stale; corrected
-- 2026-07-03.)

/-- **Julia's cot-kernel** for the circular-convolution form of the
discrete Hilbert transform. Matches `BabenkoRigorous/src/FFTHilbert.jl`
lines 120–135.

```
h_n = (2/N) cot(π n / N)   for n odd in [1, N-1]
h_n = 0                    for n even (including n = 0 and n = N/2).
```

Circular convolution with this kernel reproduces the multiplier-form
`samplingDFTHilbert` at grid points — proved as the Stage-4 main theorem
`samplingDFTHilbert_eq_cot_convolution` below. -/
noncomputable def cotKernel (N : ℕ) (n : Fin N) : ℝ :=
  if (n.val) % 2 = 0 then 0
  else (2 / (N : ℝ)) * Real.cos (Real.pi * (n.val : ℝ) / N) /
       Real.sin (Real.pi * (n.val : ℝ) / N)

/-- **Closed-form sine-sum identity** for the cot-kernel.

For `N ≥ 4` even and any `n : Fin N`, the half-grid sine sum
`(2/N) · Σ_{k=1}^{N/2-1} sin(2πkn/N)` equals Julia's cot-kernel
`cotKernel N n`. This is the trig-identity backbone for the deferred
`samplingDFTHilbert_eq_cot_convolution` theorem.

The proof uses the product-to-sum identity
`2 sin β · sin(2kβ) = cos((2k-1)β) - cos((2k+1)β)` with `β = πn/N`
to telescope the sum to `cos(β) - cos((N-1)·β) = (1 - (-1)^n) cos(β)`. -/
lemma sum_sin_grid_eq_cotKernel
    {N : ℕ} (hN_pos : 0 < N) (hN_even : Even N) (hN_ge : 4 ≤ N)
    (n : Fin N) :
    (2 / (N : ℝ)) * ∑ k ∈ Finset.Ico 1 (N / 2),
        Real.sin (2 * Real.pi * (k : ℝ) * (n.val : ℝ) / N)
      = cotKernel N n := by
  -- Basic facts about N.
  have hN_real_pos : (0 : ℝ) < (N : ℝ) := by exact_mod_cast hN_pos
  have hN_real_ne : (N : ℝ) ≠ 0 := ne_of_gt hN_real_pos
  -- N/2 ≥ 2 since N ≥ 4 and N even.
  obtain ⟨m, hm⟩ := hN_even
  have hN_half : N / 2 = m := by
    rw [hm]; omega
  have hm_ge : 2 ≤ m := by omega
  have hN_half_ge : 2 ≤ N / 2 := by rw [hN_half]; exact hm_ge
  -- Case split on n.val = 0.
  by_cases hn0 : n.val = 0
  · -- All sine terms are zero.
    have hsum : ∑ k ∈ Finset.Ico 1 (N / 2),
        Real.sin (2 * Real.pi * (k : ℝ) * (n.val : ℝ) / N) = 0 := by
      apply Finset.sum_eq_zero
      intro k _
      rw [hn0]
      simp
    rw [hsum, mul_zero]
    -- cotKernel: n.val = 0 ⇒ 0 % 2 = 0 ⇒ if-branch fires, gives 0.
    unfold cotKernel
    rw [if_pos (by rw [hn0])]
  -- Case n.val ≠ 0. Set β = π·n/N.
  set β : ℝ := Real.pi * (n.val : ℝ) / N with hβ_def
  have hn_pos : 0 < n.val := Nat.pos_of_ne_zero hn0
  have hn_lt : n.val < N := n.isLt
  -- π · n / N ∈ (0, π), so sin β > 0.
  have hβ_pos : 0 < β := by
    rw [hβ_def]
    apply div_pos
    · exact mul_pos Real.pi_pos (by exact_mod_cast hn_pos)
    · exact hN_real_pos
  have hβ_lt_pi : β < Real.pi := by
    rw [hβ_def]
    rw [div_lt_iff₀ hN_real_pos]
    have h_n_lt_N : (n.val : ℝ) < (N : ℝ) := by exact_mod_cast hn_lt
    nlinarith [Real.pi_pos]
  have hsinβ_pos : 0 < Real.sin β := Real.sin_pos_of_pos_of_lt_pi hβ_pos hβ_lt_pi
  have hsinβ_ne : Real.sin β ≠ 0 := ne_of_gt hsinβ_pos
  -- Rewrite the summand: 2π·k·n/N = 2k·β.
  have h_summand_eq : ∀ k : ℕ,
      Real.sin (2 * Real.pi * (k : ℝ) * (n.val : ℝ) / N)
        = Real.sin (2 * (k : ℝ) * β) := by
    intro k
    congr 1
    rw [hβ_def]
    field_simp
  -- Let S = ∑_{k ∈ Ico 1 (N/2)} sin(2kβ).
  set S : ℝ := ∑ k ∈ Finset.Ico 1 (N / 2), Real.sin (2 * (k : ℝ) * β) with hS_def
  have h_sum_S : ∑ k ∈ Finset.Ico 1 (N / 2),
      Real.sin (2 * Real.pi * (k : ℝ) * (n.val : ℝ) / N) = S := by
    apply Finset.sum_congr rfl
    intro k _
    exact h_summand_eq k
  rw [h_sum_S]
  -- Telescope: 2 sin β · sin(2kβ) = cos((2k-1)β) - cos((2k+1)β).
  -- Reindex Ico 1 (N/2) → range (N/2 - 1) via k = j + 1.
  have h_reindex : S = ∑ j ∈ Finset.range (N / 2 - 1),
      Real.sin (2 * ((j : ℝ) + 1) * β) := by
    rw [hS_def]
    rw [Finset.sum_Ico_eq_sum_range]
    apply Finset.sum_congr rfl
    intro j _
    congr 2
    push_cast
    ring
  -- Define F(j) = cos((2j+1)β). Then cos((2k-1)β) - cos((2k+1)β) = F(j) - F(j+1) when k = j+1.
  -- Use 2 sin β · sin(2(j+1)β) = cos((2j+1)β) - cos((2j+3)β).
  have h_prod : ∀ j : ℕ,
      2 * Real.sin β * Real.sin (2 * ((j : ℝ) + 1) * β)
        = Real.cos ((2 * (j : ℝ) + 1) * β) - Real.cos ((2 * (j : ℝ) + 3) * β) := by
    intro j
    have := Real.two_mul_sin_mul_sin β (2 * ((j : ℝ) + 1) * β)
    -- cos(β - 2(j+1)β) - cos(β + 2(j+1)β)
    rw [this]
    congr 1
    · -- cos(β - 2(j+1)β) = cos(-(2j+1)β) = cos((2j+1)β)
      rw [show β - 2 * ((j : ℝ) + 1) * β = -((2 * (j : ℝ) + 1) * β) from by ring]
      exact Real.cos_neg _
    · -- cos(β + 2(j+1)β) = cos((2j+3)β)
      congr 1; ring
  -- Therefore: 2 sin β · S = ∑ (F(j) - F(j+1)) = F(0) - F(N/2 - 1), where F(j) = cos((2j+1)β).
  have h_2sinβ_S : 2 * Real.sin β * S
      = Real.cos β - Real.cos ((2 * ((N / 2 : ℕ) : ℝ) - 1) * β) := by
    rw [h_reindex]
    rw [Finset.mul_sum]
    -- Each term: 2 sin β · sin(2(j+1)β) = F(j) - F(j+1) where F(j) = cos((2j+1)β).
    have h_each : ∀ j ∈ Finset.range (N / 2 - 1),
        2 * Real.sin β * Real.sin (2 * ((j : ℝ) + 1) * β)
          = Real.cos ((2 * (j : ℝ) + 1) * β) - Real.cos ((2 * ((j : ℝ) + 1) + 1) * β) := by
      intro j _
      have := h_prod j
      rw [this]
      congr 2
      ring
    rw [Finset.sum_congr rfl h_each]
    -- Now apply telescoping. Set F : ℕ → ℝ, F j = cos((2j+1)β).
    -- We have ∑ j ∈ range M, (F j - F (j+1)) = F 0 - F M.
    set F : ℕ → ℝ := fun j => Real.cos ((2 * (j : ℝ) + 1) * β) with hF_def
    have h_telescope : ∑ j ∈ Finset.range (N / 2 - 1),
        (F j - F (j + 1)) = F 0 - F (N / 2 - 1) := by
      -- Use Finset.sum_range_sub' (or equivalent): ∑ (F j - F (j+1)) = F 0 - F n.
      -- Or: ∑ (F (j+1) - F j) = F n - F 0, then negate.
      have h := Finset.sum_range_sub F (N / 2 - 1)
      -- h : ∑ i ∈ range n, (F (i + 1) - F i) = F n - F 0
      have : ∑ j ∈ Finset.range (N / 2 - 1), (F j - F (j + 1))
          = -(∑ j ∈ Finset.range (N / 2 - 1), (F (j + 1) - F j)) := by
        rw [← Finset.sum_neg_distrib]
        apply Finset.sum_congr rfl
        intro j _; ring
      rw [this, h]
      ring
    -- Rewrite the goal using F.
    have h_goal_eq : ∑ j ∈ Finset.range (N / 2 - 1),
        (Real.cos ((2 * (j : ℝ) + 1) * β) - Real.cos ((2 * ((j : ℝ) + 1) + 1) * β))
        = ∑ j ∈ Finset.range (N / 2 - 1), (F j - F (j + 1)) := by
      apply Finset.sum_congr rfl
      intro j _
      simp only [hF_def]
      congr 2
      push_cast
      ring
    rw [h_goal_eq, h_telescope]
    -- F 0 = cos(β), F (N/2 - 1) = cos((2(N/2-1) + 1)β) = cos((2·N/2 - 1)β)
    have hF0 : F 0 = Real.cos β := by
      simp only [hF_def, Nat.cast_zero]
      congr 1; ring
    have hN2_pos : 1 ≤ N / 2 := by omega
    have h_cast : ((N / 2 - 1 : ℕ) : ℝ) = ((N / 2 : ℕ) : ℝ) - 1 := by
      rw [Nat.cast_sub hN2_pos]
      simp
    have hFM : F (N / 2 - 1) = Real.cos ((2 * ((N / 2 : ℕ) : ℝ) - 1) * β) := by
      simp only [hF_def]
      rw [h_cast]
      congr 1; ring
    rw [hF0, hFM]
  -- Now compute cos((2·(N/2) - 1)β) = cos((N-1)·β) = cos(π·n - β) = (-1)^n · cos β.
  -- N / 2 cast: since N even, N = 2·(N/2), so 2·(N/2) = N as nats.
  have h_2_half_eq_N : (2 : ℝ) * ((N / 2 : ℕ) : ℝ) = (N : ℝ) := by
    have h_nat : 2 * (N / 2) = N := by rw [hN_half]; omega
    exact_mod_cast h_nat
  rw [h_2_half_eq_N] at h_2sinβ_S
  -- Now (N - 1) * β = N · β - β = π·n - β.
  have h_Nβ : (N : ℝ) * β = Real.pi * (n.val : ℝ) := by
    rw [hβ_def]
    field_simp
  -- cos((N - 1) * β) = cos(N·β - β) = cos(π·n - β).
  have h_cos_Nm1 : Real.cos (((N : ℝ) - 1) * β) = Real.cos (Real.pi * (n.val : ℝ) - β) := by
    congr 1
    rw [sub_mul, one_mul, h_Nβ]
  -- cos(πn - β) = cos(πn)cos(β) + sin(πn)sin(β) = (-1)^n · cos β + 0.
  have h_cos_pinm_sub : Real.cos (Real.pi * (n.val : ℝ) - β)
      = (-1 : ℝ) ^ (n.val) * Real.cos β := by
    rw [Real.cos_sub]
    -- cos(πn) = (-1)^n, sin(πn) = 0
    rw [show Real.pi * (n.val : ℝ) = (n.val : ℝ) * Real.pi from by ring]
    rw [Real.cos_nat_mul_pi, Real.sin_nat_mul_pi]
    ring
  rw [h_cos_Nm1, h_cos_pinm_sub] at h_2sinβ_S
  -- So: 2 sin β · S = cos β - (-1)^n · cos β = (1 - (-1)^n) · cos β.
  -- Now case split on n.val parity.
  rcases Nat.even_or_odd n.val with hn_even | hn_odd
  · -- n.val even: (-1)^n.val = 1, so 2 sin β · S = 0, so S = 0.
    have h_pow : (-1 : ℝ) ^ (n.val) = 1 := hn_even.neg_one_pow
    rw [h_pow] at h_2sinβ_S
    -- 2 sin β · S = cos β - 1 * cos β = 0.
    have h_zero : (2 * Real.sin β) * S = (2 * Real.sin β) * 0 := by
      rw [mul_zero]
      have : 2 * Real.sin β * S = 0 := by
        rw [h_2sinβ_S]; ring
      exact this
    have hS_zero : S = 0 := by
      have h2 : (2 : ℝ) * Real.sin β ≠ 0 := mul_ne_zero two_ne_zero hsinβ_ne
      exact mul_left_cancel₀ h2 h_zero
    rw [hS_zero, mul_zero]
    -- cotKernel: n.val even ⇒ 0.
    unfold cotKernel
    rw [if_pos (Nat.even_iff.mp hn_even)]
  · -- n.val odd: (-1)^n.val = -1, so 2 sin β · S = cos β - (-1)·cos β = 2 cos β.
    have h_pow : (-1 : ℝ) ^ (n.val) = -1 := hn_odd.neg_one_pow
    rw [h_pow] at h_2sinβ_S
    -- So Real.sin β * S = Real.cos β.
    have h_sinβ_S : Real.sin β * S = Real.cos β := by
      have h2ne : (2 : ℝ) ≠ 0 := two_ne_zero
      have h_eq : 2 * (Real.sin β * S) = 2 * Real.cos β := by
        have hsub : 2 * Real.sin β * S = Real.cos β - (-1) * Real.cos β := h_2sinβ_S
        linarith [hsub]
      exact mul_left_cancel₀ h2ne h_eq
    -- So S = cos β / sin β.
    have hS_eq : S = Real.cos β / Real.sin β := by
      field_simp
      linarith [h_sinβ_S]
    rw [hS_eq]
    -- (2/N) · cos β / sin β = cotKernel.
    unfold cotKernel
    have hn_mod : n.val % 2 ≠ 0 := by
      rw [Nat.odd_iff.mp hn_odd]
      exact one_ne_zero
    rw [if_neg hn_mod]
    -- Goal: (2/N) * (cos β / sin β) = (2/N) * cos(π·n/N) / sin(π·n/N).
    rw [hβ_def]
    ring

/-- **Inner kernel sum identity.**

For `N ≥ 4` even and `n : Fin N`, the multiplier-weighted sum of grid
exponentials equals `N` times Julia's cot-kernel:
```
∑_{k : Fin N} hilbertMultDiscrete N k.val · exp(2πi · k · n / N)
  = N · cotKernel N n.
```

**Proof strategy.** The multiplier is `-i` on `[1, N/2-1]`, `+i` on
`[N/2+1, N-1]`, and `0` at `0` and `N/2`. Pair `k` and `N - k` to get
the Euler identity `(-i)e^{ix} + (i)e^{-ix} = 2 sin x` (using that
`e^{2πi n} = 1`). The result is the half-grid sine sum, which equals
`(N/2) · cotKernel N n` by `sum_sin_grid_eq_cotKernel`. -/
private lemma cot_inner_sum_eq
    {N : ℕ} (hN_pos : 0 < N) (hN_even : Even N) (hN_ge : 4 ≤ N)
    (n : Fin N) :
    ∑ k : Fin N, hilbertMultDiscrete N (k.val : ℤ) *
        Complex.exp (2 * Real.pi * Complex.I * (k.val : ℂ) * (n.val : ℂ) / (N : ℂ))
      = (N : ℂ) * ((cotKernel N n : ℝ) : ℂ) := by
  -- Basic facts about N.
  have hN_real_pos : (0 : ℝ) < (N : ℝ) := by exact_mod_cast hN_pos
  have hN_real_ne : (N : ℝ) ≠ 0 := ne_of_gt hN_real_pos
  have hN_c_ne : (N : ℂ) ≠ 0 := by exact_mod_cast hN_real_ne
  -- N/2 facts: N = 2 · (N/2), N/2 ≥ 2, so 1 ≤ N/2 ≤ N - 1.
  have hN_even' : Even N := hN_even
  obtain ⟨m, hm⟩ := hN_even'
  have hN_half_eq : N / 2 = m := by rw [hm]; omega
  have hm_ge : 2 ≤ m := by omega
  have hN_half_ge : 2 ≤ N / 2 := by rw [hN_half_eq]; exact hm_ge
  have hN_half_pos : 0 < N / 2 := by omega
  have hN_half_lt_N : N / 2 < N := by omega
  have hN_half_plus_le_N : N / 2 + 1 ≤ N := by omega
  -- Abbreviate the summand.
  set F : ℕ → ℂ := fun k => hilbertMultDiscrete N (k : ℤ) *
      Complex.exp (2 * Real.pi * Complex.I * (k : ℂ) * (n.val : ℂ) / (N : ℂ)) with hF_def
  -- Rewrite Σ over Fin N as Σ over range N.
  have h_to_range : ∑ k : Fin N, hilbertMultDiscrete N (k.val : ℤ) *
        Complex.exp (2 * Real.pi * Complex.I * (k.val : ℂ) * (n.val : ℂ) / (N : ℂ))
      = ∑ k ∈ Finset.range N, F k := by
    have : ∑ k : Fin N, hilbertMultDiscrete N (k.val : ℤ) *
        Complex.exp (2 * Real.pi * Complex.I * (k.val : ℂ) * (n.val : ℂ) / (N : ℂ))
        = ∑ k : Fin N, F k.val := rfl
    rw [this]
    exact Fin.sum_univ_eq_sum_range F N
  rw [h_to_range]
  -- Split range N = {0} ∪ Ico 1 (N/2) ∪ {N/2} ∪ Ico (N/2 + 1) N.
  -- Use a single equation: range N = {0} ∪ Ico 1 (N/2) ∪ {N/2} ∪ Ico (N/2 + 1) N.
  have h_split : Finset.range N
      = ({0} ∪ Finset.Ico 1 (N / 2)) ∪ ({N / 2} ∪ Finset.Ico (N / 2 + 1) N) := by
    ext x
    simp only [Finset.mem_union, Finset.mem_range, Finset.mem_singleton, Finset.mem_Ico]
    omega
  rw [h_split]
  -- Disjointness for the outer union.
  have h_disj_outer : Disjoint (({0} : Finset ℕ) ∪ Finset.Ico 1 (N / 2))
      (({N / 2} : Finset ℕ) ∪ Finset.Ico (N / 2 + 1) N) := by
    rw [Finset.disjoint_left]
    intro x hx1 hx2
    simp only [Finset.mem_union, Finset.mem_singleton, Finset.mem_Ico] at hx1 hx2
    omega
  rw [Finset.sum_union h_disj_outer]
  -- Disjointness for inner unions.
  have h_disj_L : Disjoint ({0} : Finset ℕ) (Finset.Ico 1 (N / 2)) := by
    rw [Finset.disjoint_singleton_left, Finset.mem_Ico]
    omega
  have h_disj_R : Disjoint ({N / 2} : Finset ℕ) (Finset.Ico (N / 2 + 1) N) := by
    rw [Finset.disjoint_singleton_left, Finset.mem_Ico]
    omega
  rw [Finset.sum_union h_disj_L, Finset.sum_union h_disj_R]
  -- Compute F 0 = 0.
  have hF_zero : F 0 = 0 := by
    rw [hF_def]
    simp only
    unfold hilbertMultDiscrete
    simp
  -- Compute F (N/2) = 0.
  have hF_half : F (N / 2) = 0 := by
    rw [hF_def]
    simp only
    unfold hilbertMultDiscrete
    -- ((N/2 : ℕ) : ℤ) % N = N/2 since 0 ≤ N/2 < N.
    have h_mod_self : ((N / 2 : ℕ) : ℤ) % (N : ℤ) = ((N / 2 : ℕ) : ℤ) := by
      apply Int.emod_eq_of_lt
      · exact_mod_cast Nat.zero_le _
      · exact_mod_cast hN_half_lt_N
    rw [h_mod_self]
    have h_ne_zero : ((N / 2 : ℕ) : ℤ) ≠ 0 := by exact_mod_cast hN_half_pos.ne'
    rw [if_neg h_ne_zero, if_pos rfl]
    ring
  -- Simplify singletons: {0} sums to F 0 and {N/2} sums to F (N/2).
  rw [Finset.sum_singleton, Finset.sum_singleton, hF_zero, hF_half, zero_add, zero_add]
  -- Now: Σ k ∈ Ico 1 (N/2), F k + Σ k ∈ Ico (N/2+1) N, F k = N · cotKernel N n.
  -- For k ∈ Ico 1 (N/2), F k = -i · exp(2πi k n / N).
  -- For k ∈ Ico (N/2+1) N, F k = +i · exp(2πi k n / N).
  -- Substitute j = N - k in the second sum.
  have h_left_F : ∀ k ∈ Finset.Ico 1 (N / 2), F k
      = (-Complex.I) * Complex.exp (2 * Real.pi * Complex.I * (k : ℂ) * (n.val : ℂ) / (N : ℂ)) := by
    intro k hk
    rw [Finset.mem_Ico] at hk
    rw [hF_def]
    simp only
    unfold hilbertMultDiscrete
    have h_mod : ((k : ℕ) : ℤ) % (N : ℤ) = ((k : ℕ) : ℤ) := by
      apply Int.emod_eq_of_lt
      · exact_mod_cast Nat.zero_le _
      · exact_mod_cast (by omega : k < N)
    rw [h_mod]
    have h_kne_zero : ((k : ℕ) : ℤ) ≠ 0 := by exact_mod_cast (by omega : k ≠ 0)
    have h_kne_half : ((k : ℕ) : ℤ) ≠ ((N / 2 : ℕ) : ℤ) := by
      exact_mod_cast (by omega : k ≠ N / 2)
    have h_klt : ((k : ℕ) : ℤ) < ((N / 2 : ℕ) : ℤ) := by exact_mod_cast (by omega : k < N / 2)
    rw [if_neg h_kne_zero, if_neg h_kne_half, if_pos h_klt]
  have h_right_F : ∀ k ∈ Finset.Ico (N / 2 + 1) N, F k
      = Complex.I * Complex.exp (2 * Real.pi * Complex.I * (k : ℂ) * (n.val : ℂ) / (N : ℂ)) := by
    intro k hk
    rw [Finset.mem_Ico] at hk
    rw [hF_def]
    simp only
    unfold hilbertMultDiscrete
    have h_mod : ((k : ℕ) : ℤ) % (N : ℤ) = ((k : ℕ) : ℤ) := by
      apply Int.emod_eq_of_lt
      · exact_mod_cast Nat.zero_le _
      · exact_mod_cast (by omega : k < N)
    rw [h_mod]
    have h_kne_zero : ((k : ℕ) : ℤ) ≠ 0 := by exact_mod_cast (by omega : k ≠ 0)
    have h_kne_half : ((k : ℕ) : ℤ) ≠ ((N / 2 : ℕ) : ℤ) := by
      exact_mod_cast (by omega : k ≠ N / 2)
    have h_kge : ¬ ((k : ℕ) : ℤ) < ((N / 2 : ℕ) : ℤ) := by
      push_neg; exact_mod_cast (by omega : N / 2 ≤ k)
    rw [if_neg h_kne_zero, if_neg h_kne_half, if_neg h_kge]
  rw [Finset.sum_congr rfl h_left_F, Finset.sum_congr rfl h_right_F]
  -- Substitute j = N - k in the right sum: ∑_{k ∈ Ico (N/2+1) N} f(k) = ∑_{j ∈ Ico 1 (N/2)} f(N - j).
  -- Use Finset.sum_bij with bij j ↦ N - j from Ico 1 (N/2) → Ico (N/2 + 1) N.
  have h_sub : ∑ k ∈ Finset.Ico (N / 2 + 1) N,
        Complex.I * Complex.exp (2 * Real.pi * Complex.I * (k : ℂ) * (n.val : ℂ) / (N : ℂ))
      = ∑ j ∈ Finset.Ico 1 (N / 2),
        Complex.I * Complex.exp (2 * Real.pi * Complex.I * ((N - j : ℕ) : ℂ) * (n.val : ℂ) / (N : ℂ)) := by
    refine Finset.sum_nbij' (fun k => N - k) (fun j => N - j) ?_ ?_ ?_ ?_ ?_
    · intro k hk
      simp only [Finset.mem_Ico] at hk ⊢
      omega
    · intro j hj
      simp only [Finset.mem_Ico] at hj ⊢
      omega
    · intro k hk
      simp only [Finset.mem_Ico] at hk
      show N - (N - k) = k
      omega
    · intro j hj
      simp only [Finset.mem_Ico] at hj
      show N - (N - j) = j
      omega
    · intro k hk
      simp only [Finset.mem_Ico] at hk
      -- Need: I * exp(2πi · k · n / N) = I * exp(2πi · (N - (N - k)) · n / N).
      have h_rw : N - (N - k) = k := by omega
      rw [h_rw]
  rw [h_sub]
  -- Combine the two sums termwise.
  rw [← Finset.sum_add_distrib]
  -- Now: ∑ j ∈ Ico 1 (N/2), [(-i) * exp(2πi j n / N) + i * exp(2πi (N - j) n / N)] = N · cotKernel.
  -- Show each term equals 2 · sin(2π j n / N).
  have h_each : ∀ j ∈ Finset.Ico 1 (N / 2),
      (-Complex.I) * Complex.exp (2 * Real.pi * Complex.I * (j : ℂ) * (n.val : ℂ) / (N : ℂ))
      + Complex.I * Complex.exp (2 * Real.pi * Complex.I * ((N - j : ℕ) : ℂ) * (n.val : ℂ) / (N : ℂ))
      = (2 : ℂ) * Real.sin (2 * Real.pi * (j : ℝ) * (n.val : ℝ) / N) := by
    intro j hj
    rw [Finset.mem_Ico] at hj
    -- Let α := 2π j n / N (as a complex number, but real).
    set α : ℝ := 2 * Real.pi * (j : ℝ) * (n.val : ℝ) / (N : ℝ) with hα_def
    -- Key: exp(2πi · (N - j) · n / N) = exp(2πi · n) · exp(-2πi · j · n / N) = exp(-i α).
    -- Because exp(2πi · n) = 1 (n : ℕ).
    have hjN : (j : ℕ) ≤ N := by omega
    have h_Nsubj : ((N - j : ℕ) : ℂ) = (N : ℂ) - (j : ℂ) := by
      rw [Nat.cast_sub hjN]
    rw [h_Nsubj]
    -- exp(2πi · (N - j) · n / N) = exp(2πi · n - 2πi · j · n / N) = exp(2πi · n) · exp(-2πi · j · n / N).
    have h_split_exp : (2 * Real.pi * Complex.I * ((N : ℂ) - (j : ℂ)) * (n.val : ℂ) / (N : ℂ))
        = (n.val : ℂ) * (2 * Real.pi * Complex.I)
          + (-(2 * Real.pi * Complex.I * (j : ℂ) * (n.val : ℂ) / (N : ℂ))) := by
      field_simp
      ring
    rw [h_split_exp, Complex.exp_add]
    -- exp((n.val : ℂ) * (2π i)) = 1.
    have h_n_exp : Complex.exp ((n.val : ℂ) * (2 * Real.pi * Complex.I)) = 1 := by
      have h_form : (n.val : ℂ) * (2 * Real.pi * Complex.I)
          = ((n.val : ℤ) : ℂ) * (2 * Real.pi * Complex.I) := by push_cast; ring
      rw [h_form]
      exact Complex.exp_int_mul_two_pi_mul_I _
    rw [h_n_exp, one_mul]
    -- Now: -i · exp(iα) + i · exp(-iα) where α (real) = 2π j n / N.
    -- exp(iα) - exp(-iα) = 2i sin α  ⇒  -i exp(iα) + i exp(-iα) = -i(exp(iα) - exp(-iα)) = -i · 2i · sin α = 2 sin α.
    -- First, write the exp arguments in the form `α · I`.
    have h_arg1 : (2 * Real.pi * Complex.I * (j : ℂ) * (n.val : ℂ) / (N : ℂ))
        = ((α : ℝ) : ℂ) * Complex.I := by
      simp only [hα_def]
      push_cast
      field_simp
    rw [h_arg1]
    -- exp(α · I) = cos α + sin α · I (Complex.exp_mul_I).
    -- exp(-α · I) = cos α - sin α · I.
    rw [Complex.exp_mul_I, show -(((α : ℝ) : ℂ) * Complex.I) = ((-α : ℝ) : ℂ) * Complex.I from by
      push_cast; ring, Complex.exp_mul_I]
    -- Convert Complex.cos / Complex.sin to Real.cos / Real.sin via ofReal.
    rw [show Complex.cos ((α : ℝ) : ℂ) = ((Real.cos α : ℝ) : ℂ) from
        (Complex.ofReal_cos α).symm,
        show Complex.sin ((α : ℝ) : ℂ) = ((Real.sin α : ℝ) : ℂ) from
        (Complex.ofReal_sin α).symm,
        show Complex.cos ((-α : ℝ) : ℂ) = ((Real.cos (-α) : ℝ) : ℂ) from
        (Complex.ofReal_cos (-α)).symm,
        show Complex.sin ((-α : ℝ) : ℂ) = ((Real.sin (-α) : ℝ) : ℂ) from
        (Complex.ofReal_sin (-α)).symm]
    rw [Real.cos_neg, Real.sin_neg]
    -- Now: -I (cos α + sin α I) + I (cos α + (-sin α) I) = 2 sin α (in ℂ).
    -- = -I cos α - sin α I I + I cos α + (- sin α) I I = -I cos α + sin α + I cos α + sin α = 2 sin α.
    push_cast
    ring_nf
    rw [Complex.I_sq]
    ring
  rw [Finset.sum_congr rfl h_each]
  -- Now: ∑ j ∈ Ico 1 (N/2), 2 · sin(2π j n / N) = N · cotKernel N n.
  -- Factor out 2:
  rw [← Finset.mul_sum]
  -- Use (2/N) · Σ sin = cotKernel, i.e. Σ sin = (N/2) · cotKernel, so 2 · Σ sin = N · cotKernel.
  have h_sin_sum : (2 / (N : ℝ)) * ∑ k ∈ Finset.Ico 1 (N / 2),
      Real.sin (2 * Real.pi * (k : ℝ) * (n.val : ℝ) / N) = cotKernel N n :=
    sum_sin_grid_eq_cotKernel hN_pos hN_even hN_ge n
  -- Multiply by N: 2 · Σ sin = N · cotKernel (over ℝ).
  have h_sin_R : (2 : ℝ) * ∑ k ∈ Finset.Ico 1 (N / 2),
      Real.sin (2 * Real.pi * (k : ℝ) * (n.val : ℝ) / N) = (N : ℝ) * cotKernel N n := by
    have h_mul : (N : ℝ) * ((2 / (N : ℝ)) * ∑ k ∈ Finset.Ico 1 (N / 2),
        Real.sin (2 * Real.pi * (k : ℝ) * (n.val : ℝ) / N))
        = (N : ℝ) * cotKernel N n := by
      rw [h_sin_sum]
    rw [← h_mul]
    rw [show (N : ℝ) * (2 / (N : ℝ) * ∑ k ∈ Finset.Ico 1 (N / 2),
            Real.sin (2 * Real.pi * (k : ℝ) * (n.val : ℝ) / N))
        = ((N : ℝ) * (2 / (N : ℝ))) * ∑ k ∈ Finset.Ico 1 (N / 2),
            Real.sin (2 * Real.pi * (k : ℝ) * (n.val : ℝ) / N) from by ring]
    rw [mul_div_cancel₀ _ hN_real_ne]
  -- Cast to complex.
  have h_cast_sum : (∑ j ∈ Finset.Ico 1 (N / 2),
        ((2 : ℂ) * (Real.sin (2 * Real.pi * (j : ℝ) * (n.val : ℝ) / N) : ℂ)))
      = ((((2 : ℝ) * ∑ k ∈ Finset.Ico 1 (N / 2),
          Real.sin (2 * Real.pi * (k : ℝ) * (n.val : ℝ) / N) : ℝ)) : ℂ) := by
    push_cast
    rw [Finset.mul_sum]
  -- The current LHS is `(2 : ℂ) * Σ_j (sin _ : ℂ)` after `mul_sum`. Let me match the form.
  rw [show (∑ j ∈ Finset.Ico 1 (N / 2),
        ((Real.sin (2 * Real.pi * (j : ℝ) * (n.val : ℝ) / N) : ℝ) : ℂ))
      = (((∑ j ∈ Finset.Ico 1 (N / 2),
          Real.sin (2 * Real.pi * (j : ℝ) * (n.val : ℝ) / N) : ℝ)) : ℂ) from by
    push_cast; rfl]
  rw [show ((2 : ℂ) * (((∑ j ∈ Finset.Ico 1 (N / 2),
          Real.sin (2 * Real.pi * (j : ℝ) * (n.val : ℝ) / N) : ℝ)) : ℂ))
      = (((2 : ℝ) * (∑ j ∈ Finset.Ico 1 (N / 2),
          Real.sin (2 * Real.pi * (j : ℝ) * (n.val : ℝ) / N) : ℝ) : ℝ) : ℂ) from by
    push_cast; ring]
  rw [h_sin_R]
  push_cast
  ring
/-- **Stage 4 main theorem.** The multiplier-form `samplingDFTHilbert` at
grid points equals Julia's cot-kernel circular convolution. -/
theorem samplingDFTHilbert_eq_cot_convolution
    {N : ℕ} (hN_pos : 0 < N) (hN_even : Even N) (hN_ge : 4 ≤ N)
    (G : ℝ → ℝ) (j : Fin N) :
    samplingDFTHilbert N G (fftGridPoint N j)
      = ∑ n : Fin N, cotKernel N n * G (fftGridPoint N (j - n)) := by
  -- Basic facts.
  have hN_real_pos : (0 : ℝ) < (N : ℝ) := by exact_mod_cast hN_pos
  have hN_real_ne : (N : ℝ) ≠ 0 := ne_of_gt hN_real_pos
  have hN_c_ne : (N : ℂ) ≠ 0 := by exact_mod_cast hN_real_ne
  haveI : NeZero N := ⟨by omega⟩
  -- Unfold samplingDFTHilbert = dftHilbert.
  show dftHilbert N G (fftGridPoint N j) = _
  unfold dftHilbert
  -- Replace exp(k.val · I · fftGridPoint N j) with exp(2πi k j / N).
  have h_exp_grid : ∀ k : Fin N,
      Complex.exp ((k.val : ℂ) * Complex.I * ((fftGridPoint N j : ℝ) : ℂ))
        = Complex.exp (2 * Real.pi * Complex.I * (k.val : ℂ) * (j.val : ℂ) / (N : ℂ)) := by
    intro k
    exact exp_grid_eq N (k.val : ℂ) j
  rw [show (∑ k : Fin N,
        hilbertMultDiscrete N (k.val : ℤ) *
          dftCoeff N G (k.val : ℤ) *
          Complex.exp ((k.val : ℂ) * Complex.I * ((fftGridPoint N j : ℝ) : ℂ)))
      = ∑ k : Fin N,
        hilbertMultDiscrete N (k.val : ℤ) *
          dftCoeff N G (k.val : ℤ) *
          Complex.exp (2 * Real.pi * Complex.I * (k.val : ℂ) * (j.val : ℂ) / (N : ℂ))
        from Finset.sum_congr rfl (fun k _ => by rw [h_exp_grid k])]
  -- Expand dftCoeff and distribute.
  -- dftCoeff N G k = (1/N) * Σ m G(u_m) * exp(-2πi k m / N)
  have h_dftCoeff_expand : ∀ k : Fin N,
      hilbertMultDiscrete N (k.val : ℤ) *
        dftCoeff N G (k.val : ℤ) *
        Complex.exp (2 * Real.pi * Complex.I * (k.val : ℂ) * (j.val : ℂ) / (N : ℂ))
      = (1 / (N : ℂ)) * ∑ m : Fin N,
          ((G (fftGridPoint N m) : ℝ) : ℂ) *
            (hilbertMultDiscrete N (k.val : ℤ) *
              Complex.exp (2 * Real.pi * Complex.I * (k.val : ℂ) *
                ((j.val : ℂ) - (m.val : ℂ)) / (N : ℂ))) := by
    intro k
    unfold dftCoeff
    -- After unfold: (M(k) * (1/N * Σ m, G(u_m) * exp(-2πi k m / N))) * exp(2πi k j / N).
    -- Note: the inner sum has cast `((k.val : ℤ) : ℂ)` while the outer exp has `(k.val : ℂ)`.
    -- Goal: equal to (1/N) * Σ m, G(u_m) * (M(k) * exp(2πi k (j-m) / N)).
    -- Strategy: rewrite both sides into a single sum over m, with term equal by ring + exp_add.
    rw [Finset.mul_sum, Finset.mul_sum, Finset.sum_mul, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro m _
    -- For each m: combine exponentials.
    have h_grid : (G (2 * Real.pi * (m.val : ℝ) / (N : ℝ)) : ℝ) = G (fftGridPoint N m) := by
      rfl
    rw [h_grid]
    -- Normalize the inner exp's k-cast to match: ((k.val : ℤ) : ℂ) = (k.val : ℂ).
    have h_cast_k : (((k.val : ℤ) : ℂ)) = ((k.val : ℂ)) := by push_cast; rfl
    rw [h_cast_k]
    -- Combine exponentials: exp(-2πi k m / N) * exp(2πi k j / N) = exp(2πi k (j-m) / N).
    have h_exp_sub : Complex.exp (-(2 * Real.pi * Complex.I * (k.val : ℂ) *
          (m.val : ℂ) / (N : ℂ))) *
        Complex.exp (2 * Real.pi * Complex.I * (k.val : ℂ) * (j.val : ℂ) / (N : ℂ))
      = Complex.exp (2 * Real.pi * Complex.I * (k.val : ℂ) *
          ((j.val : ℂ) - (m.val : ℂ)) / (N : ℂ)) := by
      rw [← Complex.exp_add]
      congr 1
      field_simp
      ring
    have h_assoc : hilbertMultDiscrete N (k.val : ℤ) *
          (1 / (N : ℂ) * ((G (fftGridPoint N m) : ℂ) *
            Complex.exp (-(2 * Real.pi * Complex.I * (k.val : ℂ) * (m.val : ℂ) / (N : ℂ))))) *
          Complex.exp (2 * Real.pi * Complex.I * (k.val : ℂ) * (j.val : ℂ) / (N : ℂ))
        = 1 / (N : ℂ) * ((G (fftGridPoint N m) : ℂ) *
          (hilbertMultDiscrete N (k.val : ℤ) *
            (Complex.exp (-(2 * Real.pi * Complex.I * (k.val : ℂ) * (m.val : ℂ) / (N : ℂ))) *
              Complex.exp (2 * Real.pi * Complex.I * (k.val : ℂ) * (j.val : ℂ) / (N : ℂ))))) := by
      ring
    rw [h_assoc, h_exp_sub]
  rw [Finset.sum_congr rfl (fun k _ => h_dftCoeff_expand k)]
  -- Now: ∑ k, (1/N) Σ m, G(u_m) * (M(k) * exp(2πi k (j - m) / N))
  -- Factor out (1/N) and swap sums.
  rw [← Finset.mul_sum]
  rw [Finset.sum_comm]
  -- Now: (1/N) * Σ m, Σ k, G(u_m) * (M(k) * exp(...))
  --    = (1/N) * Σ m, G(u_m) * Σ k, M(k) * exp(...)
  rw [show (∑ m : Fin N, ∑ k : Fin N,
        ((G (fftGridPoint N m) : ℝ) : ℂ) *
          (hilbertMultDiscrete N (k.val : ℤ) *
            Complex.exp (2 * Real.pi * Complex.I * (k.val : ℂ) *
              ((j.val : ℂ) - (m.val : ℂ)) / (N : ℂ))))
      = ∑ m : Fin N, ((G (fftGridPoint N m) : ℝ) : ℂ) *
        ∑ k : Fin N, hilbertMultDiscrete N (k.val : ℤ) *
          Complex.exp (2 * Real.pi * Complex.I * (k.val : ℂ) *
            ((j.val : ℂ) - (m.val : ℂ)) / (N : ℂ)) from by
      apply Finset.sum_congr rfl
      intro m _
      rw [Finset.mul_sum]]
  -- Inner sum: apply cot_inner_sum_eq with n = (j - m : Fin N).
  -- Use that exp(2πi k (j - m) / N) is the same as exp(2πi k ((j-m).val : Fin N) / N)
  -- because they differ by exp(2πi k · integer) = 1.
  have h_inner : ∀ m : Fin N,
      ∑ k : Fin N, hilbertMultDiscrete N (k.val : ℤ) *
          Complex.exp (2 * Real.pi * Complex.I * (k.val : ℂ) *
            ((j.val : ℂ) - (m.val : ℂ)) / (N : ℂ))
        = (N : ℂ) * ((cotKernel N (j - m) : ℝ) : ℂ) := by
    intro m
    -- exp(2πi k (j - m) / N) = exp(2πi k (j - m).val / N).
    have h_exp_rewrite : ∀ k : Fin N,
        Complex.exp (2 * Real.pi * Complex.I * (k.val : ℂ) *
            ((j.val : ℂ) - (m.val : ℂ)) / (N : ℂ))
          = Complex.exp (2 * Real.pi * Complex.I * (k.val : ℂ) *
              (((j - m).val : ℕ) : ℂ) / (N : ℂ)) := by
      intro k
      -- (j - m).val ≡ j.val - m.val mod N, i.e. (j-m).val * N + (multiple of N) = j.val - m.val
      -- So (j.val - m.val : ℤ) - ((j - m).val : ℤ) = q * N for some q : ℤ.
      -- Thus exp(2πi k ((j-m).val - (j - m).val) / N) = exp(2πi k q) = 1.
      -- Hence exp(2πi k (j.val - m.val) / N) = exp(2πi k (j - m).val / N).
      have h_cast : ((j - m : Fin N).val : ℤ) = ((j.val : ℤ) - (m.val : ℤ)) % (N : ℤ) :=
        Fin.coe_int_sub_eq_mod j m
      -- (j.val - m.val : ℤ) = (j - m).val + N * q for some q.
      -- Set diff = (j.val - m.val) - (j - m).val; then diff is a multiple of N.
      set diff : ℤ := ((j.val : ℤ) - (m.val : ℤ)) - ((j - m : Fin N).val : ℤ) with hdiff_def
      have hdiff_mod : diff % (N : ℤ) = 0 := by
        rw [hdiff_def, h_cast]
        simp [Int.sub_emod, Int.emod_emod_of_dvd]
      obtain ⟨q, hq⟩ := Int.dvd_of_emod_eq_zero hdiff_mod
      -- Now: 2πi k (j.val - m.val) / N = 2πi k (j-m).val / N + 2πi k q
      have h_diff_Z : ((j.val : ℤ) - (m.val : ℤ)) =
          ((j - m : Fin N).val : ℤ) + (N : ℤ) * q := by
        have h_eq : diff = (N : ℤ) * q := hq
        rw [hdiff_def] at h_eq
        linarith
      have h_diff_C : ((j.val : ℂ) - (m.val : ℂ)) =
          (((j - m).val : ℕ) : ℂ) + (N : ℂ) * (q : ℂ) := by
        have h_cast' : (((j.val : ℤ) - (m.val : ℤ) : ℤ) : ℂ) =
            ((((j - m : Fin N).val : ℤ) + (N : ℤ) * q : ℤ) : ℂ) := by
          exact_mod_cast h_diff_Z
        push_cast at h_cast'
        linear_combination h_cast'
      have h_split : (2 * Real.pi * Complex.I * (k.val : ℂ) *
              ((j.val : ℂ) - (m.val : ℂ)) / (N : ℂ))
          = (2 * Real.pi * Complex.I * (k.val : ℂ) *
              (((j - m).val : ℕ) : ℂ) / (N : ℂ))
            + (k.val : ℂ) * (q : ℂ) * (2 * Real.pi * Complex.I) := by
        rw [h_diff_C]
        field_simp
      rw [h_split]
      rw [Complex.exp_add]
      have h_kq_exp : Complex.exp ((k.val : ℂ) * (q : ℂ) * (2 * Real.pi * Complex.I)) = 1 := by
        have h_form : (k.val : ℂ) * (q : ℂ) * (2 * Real.pi * Complex.I)
            = (((k.val : ℤ) * q : ℤ) : ℂ) * (2 * Real.pi * Complex.I) := by
          push_cast; ring
        rw [h_form]
        exact Complex.exp_int_mul_two_pi_mul_I _
      rw [h_kq_exp, mul_one]
    rw [Finset.sum_congr rfl (fun k _ => by rw [h_exp_rewrite k])]
    exact cot_inner_sum_eq hN_pos hN_even hN_ge (j - m)
  rw [Finset.sum_congr rfl (fun m _ => by rw [h_inner m])]
  -- Now: (1/N) * Σ m, G(u_m) * (N * cotKernel N (j - m))
  --    = Σ m, G(u_m) * cotKernel N (j - m)
  rw [show (∑ m : Fin N, ((G (fftGridPoint N m) : ℝ) : ℂ) *
            ((N : ℂ) * ((cotKernel N (j - m) : ℝ) : ℂ)))
      = (N : ℂ) * ∑ m : Fin N, ((G (fftGridPoint N m) : ℝ) : ℂ) *
            ((cotKernel N (j - m) : ℝ) : ℂ) from by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro m _
    ring]
  rw [show (1 / (N : ℂ)) * ((N : ℂ) * ∑ m : Fin N, _) = ∑ m : Fin N,
        ((G (fftGridPoint N m) : ℝ) : ℂ) * ((cotKernel N (j - m) : ℝ) : ℂ) from by
    rw [← mul_assoc]
    rw [div_mul_cancel₀ 1 hN_c_ne]
    rw [one_mul]]
  -- Take Re of both sides; the sum is real.
  rw [show (∑ m : Fin N, ((G (fftGridPoint N m) : ℝ) : ℂ) *
            ((cotKernel N (j - m) : ℝ) : ℂ))
      = (((∑ m : Fin N, G (fftGridPoint N m) * cotKernel N (j - m)) : ℝ) : ℂ) from by
    rw [Complex.ofReal_sum]
    apply Finset.sum_congr rfl
    intro m _
    push_cast
    ring]
  rw [Complex.ofReal_re]
  -- Re-index n = j - m. Use Equiv.subLeft j : Fin N ≃ Fin N, n ↦ j - n.
  rw [← (Equiv.subLeft j).sum_comp
    (fun n : Fin N => cotKernel N n * G (fftGridPoint N (j - n)))]
  -- After sum_comp: Σ m, cotKernel N (j - m) * G(fftGridPoint N (j - (j - m)))
  -- Need to show j - (j - m) = m.
  apply Finset.sum_congr rfl
  intro m _
  show G (fftGridPoint N m) * cotKernel N (j - m) =
      cotKernel N (Equiv.subLeft j m) * G (fftGridPoint N (j - (Equiv.subLeft j m)))
  rw [Equiv.subLeft_apply]
  rw [show j - (j - m) = m from by abel]
  ring

end WaveCAP

-- All N4 *auxiliary* endpoints depend only on `propext, Classical.choice, Quot.sound`.
#print axioms WaveCAP.fourierCoeff_norm_le_Cp_div_pow
#print axioms WaveCAP.truncatedSpectralHilbert_diff_le_fourierTail
#print axioms WaveCAP.truncatedSpectralHilbert_aliasing_bound
#print axioms WaveCAP.sum_exp_two_pi_I_grid_eq
#print axioms WaveCAP.dftCoeff_eq_aliased_sum
#print axioms WaveCAP.samplingDFTHilbert_diff_le_aliasingTail_grid
#print axioms WaveCAP.dft_hilbert_aliasing_bound
#print axioms WaveCAP.sum_sin_grid_eq_cotKernel
#print axioms WaveCAP.samplingDFTHilbert_eq_cot_convolution
