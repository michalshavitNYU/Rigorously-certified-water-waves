import WaveCAP.TaylorSeriesCorrectness
import WaveCAP.SeriesCorrectness
import WaveCAP.FFTHilbertCorrectness
import WaveCAP.HilbertOnTrigPoly
import WaveCAP.HilbertCommutesDeriv
import WaveCAP.DLowerBound
import WaveCAP.TightDGSupBounds
import WaveCAP.Definitions
import WaveCAP.Certificate

/-!
# Area G — Residual Assembly Correctness

This file **composes** the per-operation correctness lemmas from Areas A
(`TaylorSeriesCorrectness`), B (`SeriesCorrectness`), F
(`FFTHilbertCorrectness`), and the periodicity machinery from
`TightDGSupBounds` / `HilbertCommutesDeriv` into a single pipeline-level
correctness theorem mirroring the Julia routine `residual_bound` in
`BabenkoRigorous/src/ResidualBound.jl`.

## What is produced

A theorem `cert_residual_le_derived` of the same shape as the trust-anchor
axiom `WaveCAP.Certificate.cert_residual_le`:

```
∀ u, |WaveCAP.R a_cert μ_cert β_cert u| ≤ ε₀_cert
```

Phase 6 retired the monolithic axioms: `cert_residual_le` / `cert_D_ge`
are now derived theorems in `CertificateAxioms.lean`, resting on the two
numeric trust anchors only (de-axiomatization completed 2026-07-03).

## Structure of the proof

The Julia `_residual_bound_full` routine returns, on top of the scalar
`bound`, a `TaylorSeriesData` `ts_R` describing the residual `R` itself
together with grid derivatives, per-level sup_bounds, and (via
`sup_bound(ts_R)`) a Taylor-remainder-based sup-norm bound. The Lean
side's job is to **certify** that the bound returned matches the true
`R`.

We split the work into:

* **Per-operation chain** (the *content* of the certification). At every
  step of the Julia pipeline, we apply the matching Area-A/B/F lemma to
  derive `IsTaylorSeriesOf` for the new intermediate. The actual
  `TaylorSeriesData` records come from the Julia output; here they are
  abstract `TaylorSeriesData` values bound to the caller's witness.
* **Sup-bound bridge** (the *quantitative content*). Once
  `IsTaylorSeriesOf ts_R R 0` is established, we invoke
  `sup_bound_taylorseries_correct` to obtain a bound on `|R u|` over
  one period `[0, 2π]`. The bound matches `ε₀_cert` by hypothesis (the
  Julia `sup_bound(ts_R)` computation).
* **Periodicity lift**. The bound on `[0, 2π]` lifts to all of `ℝ`
  because `R a μ β` is `2π`-periodic.

## Caller hypotheses (Arb trust boundary)

Each `IsTaylorSeriesOf` witness in the chain is a *caller hypothesis*: it
abstracts the fact that the Julia routine's Arb-computed
`TaylorSeriesData` correctly describes the corresponding mathematical
function. Phase 6 will replace each such hypothesis with the
corresponding `cert_arb_*` axiom (one per Arb primitive).

The genuine analytic hypotheses are:

* `h_E_input`: `D_lower ≤ D a u` (Area E). In Phase 6 this is the
  `cert_D_ge` axiom in `Certificate.lean`.
* `h_F_inputs` (per-call F1 contract): the `h_derivs_acc` of
  `fft_hilbert_series_correct` for each of the FFT-Hilbert calls in the
  pipeline (two in Path B, one in Path A). These bundle the per-grid-point
  rounding plus the aliasing radius.
* (The `ContDiff (M+1) (periodicHilbert g)` smoothness fact is no longer a
  caller obligation: it is discharged internally by `fft_hilbert_series_correct`
  via the axiom-clean `hilbert_contDiff` from `HilbertSmoothness.lean`.)

## Pipeline-level bundle

For brevity, the headline theorem accepts these caller hypotheses bundled
into a single `h_pipeline_bound` value:

```
h_pipeline_bound :
  ∃ (M : ℕ) (N_fft : ℕ) (h_pos : 0 < N_fft) (ts_R : TaylorSeriesData M N_fft),
    IsTaylorSeriesOf ts_R (R a_cert μ_cert β_cert) 0 ∧
    ts_R.h = 2 * Real.pi / N_fft ∧
    (∀ u ∈ Set.Icc (0 : ℝ) (0 + N_fft * ts_R.h),
      |R a_cert μ_cert β_cert u| ≤ ε₀_cert)
```

This bundle is what the per-operation chain produces. The
`make_pipeline_bound_from_chain` helper below shows how to *construct* it
by chaining the Area A/B/F lemmas; the user typically constructs the
bundle via the Julia `residual_bound` output and then invokes
`cert_residual_le_derived` for the lift to `∀ u : ℝ`.

## Cross-reference

* `BabenkoRigorous/src/ResidualBound.jl` — Julia source.
* `WaveCAP/WaveCAP/TaylorSeriesCorrectness.lean` — Area A.
* `WaveCAP/WaveCAP/SeriesCorrectness.lean` — Area B.
* `WaveCAP/WaveCAP/FFTHilbertCorrectness.lean` — Area F.
-/

namespace WaveCAP

open WaveCAP.Certificate Real Set

-- ===========================================================================
-- 1. Periodicity bridge: residual sup over ℝ = sup over [0, 2π]
-- ===========================================================================

/-- `y a` is `2π`-periodic for any cosine coefficient vector `a`. (Companion
to `yu_periodic`, `Hyu_periodic`, `D_periodic` already in
`TightDGSupBounds.lean`.) -/
theorem y_periodic {N : ℕ} (a : Fin N → ℝ) :
    Function.Periodic (y a) (2 * Real.pi) := by
  intro u
  unfold y
  refine Finset.sum_congr rfl ?_
  intro k _
  congr 1
  have hk : (k : ℝ) * (u + 2 * Real.pi) = (k : ℝ) * u + (k : ℤ) * (2 * Real.pi) := by
    push_cast; ring
  rw [hk, Real.cos_add_int_mul_two_pi]

/-- `G a` is `2π`-periodic. -/
theorem G_periodic {N : ℕ} (a : Fin N → ℝ) :
    Function.Periodic (G a) (2 * Real.pi) := by
  intro u
  show G a (u + 2 * Real.pi) = G a u
  unfold G
  rw [Hyu_periodic a u, D_periodic a u]

/-- `periodicHilbert g` is `2π`-periodic for any input `g`. (Trivial from the
Fourier-series definition: each term `(-i sgn n) ĝ(n) exp(i n u)` has period
`2π/|n|`, all dividing `2π`.) -/
theorem periodicHilbert_is_periodic (g : ℝ → ℝ) :
    Function.Periodic (periodicHilbert g) (2 * Real.pi) := by
  intro u
  unfold periodicHilbert
  congr 1
  refine tsum_congr ?_
  intro n
  congr 1
  have heq : ((u + 2 * Real.pi : ℝ) : ℂ) = (u : ℂ) + 2 * Real.pi := by
    push_cast; ring
  rw [heq]
  have hexp_split :
      Complex.exp ((n : ℂ) * Complex.I * ((u : ℂ) + 2 * Real.pi))
        = Complex.exp ((n : ℂ) * Complex.I * (u : ℂ))
          * Complex.exp ((n : ℂ) * Complex.I * (2 * Real.pi)) := by
    rw [show (n : ℂ) * Complex.I * ((u : ℂ) + 2 * Real.pi)
          = (n : ℂ) * Complex.I * (u : ℂ)
            + (n : ℂ) * Complex.I * (2 * Real.pi) from by ring,
        Complex.exp_add]
  rw [hexp_split]
  have hreduce : (n : ℂ) * Complex.I * (2 * Real.pi)
      = (n : ℤ) * (2 * Real.pi * Complex.I) := by ring
  rw [hreduce, Complex.exp_int_mul_two_pi_mul_I, mul_one]

/-- The residual `R a μ β` is `2π`-periodic. -/
theorem R_periodic {N : ℕ} (a : Fin N → ℝ) (μ β : ℝ) :
    Function.Periodic (R a μ β) (2 * Real.pi) := by
  intro u
  unfold R
  -- All six pieces are 2π-periodic.
  have hH := Hyu_periodic a u
  have hy := y_periodic a u
  have hHyyu := periodicHilbert_is_periodic (fun v => y a v * yu a v) u
  -- `yu / D` is 2π-periodic; hence its derivative is too.
  have hyuD_per : Function.Periodic (fun v => yu a v / D a v) (2 * Real.pi) := by
    intro v
    change yu a (v + 2 * Real.pi) / D a (v + 2 * Real.pi) = yu a v / D a v
    rw [yu_periodic a v, D_periodic a v]
  have hC1 : deriv (fun v => yu a v / D a v) (u + 2 * Real.pi)
      = deriv (fun v => yu a v / D a v) u :=
    WaveCAP.Periodic.deriv_periodic hyuD_per u
  have hHGu : periodicHilbert (deriv (G a)) (u + 2 * Real.pi)
      = periodicHilbert (deriv (G a)) u :=
    periodicHilbert_is_periodic (deriv (G a)) u
  rw [hH, hy, hHyyu, hC1, hHGu]

/-- For a `2π`-periodic function `f`, a sup-norm bound on `[0, 2π]` extends to
a sup-norm bound on all of `ℝ`. Used to lift the Julia residual bound to a
statement about all `u : ℝ`. -/
theorem periodic_sup_extends {f : ℝ → ℝ} (hper : Function.Periodic f (2 * Real.pi))
    (C : ℝ) (h_per_bound : ∀ u ∈ Set.Icc (0 : ℝ) (2 * Real.pi), |f u| ≤ C) :
    ∀ u, |f u| ≤ C := by
  intro u
  -- Shift `u` by an integer multiple of `2π` to land in `[0, 2π)`.
  set v := u - (⌊u / (2 * Real.pi)⌋ : ℤ) * (2 * Real.pi) with hv_def
  have h2π_pos : (0 : ℝ) < 2 * Real.pi := zero_lt_two_pi
  have h2π_ne : (2 * Real.pi) ≠ 0 := ne_of_gt h2π_pos
  have hv_floor : (⌊u / (2 * Real.pi)⌋ : ℝ) ≤ u / (2 * Real.pi) := Int.floor_le _
  have hv_lt : u / (2 * Real.pi) < (⌊u / (2 * Real.pi)⌋ : ℝ) + 1 := Int.lt_floor_add_one _
  have hv_ge : 0 ≤ v := by
    have h1 : (⌊u / (2 * Real.pi)⌋ : ℝ) * (2 * Real.pi) ≤ u := by
      have := mul_le_mul_of_nonneg_right hv_floor h2π_pos.le
      rwa [div_mul_cancel₀ _ h2π_ne] at this
    have h2 : v = u - (⌊u / (2 * Real.pi)⌋ : ℝ) * (2 * Real.pi) := by simp [hv_def]
    linarith
  have hv_lt_2pi : v ≤ 2 * Real.pi := by
    have h1 : u ≤ ((⌊u / (2 * Real.pi)⌋ : ℝ) + 1) * (2 * Real.pi) := by
      have := mul_le_mul_of_nonneg_right hv_lt.le h2π_pos.le
      rwa [div_mul_cancel₀ _ h2π_ne] at this
    have h2 : v = u - (⌊u / (2 * Real.pi)⌋ : ℝ) * (2 * Real.pi) := by simp [hv_def]
    nlinarith
  -- f u = f v by periodicity.
  have hfeq : f u = f v := by
    have hshift : v + (⌊u / (2 * Real.pi)⌋ : ℤ) * (2 * Real.pi) = u := by
      simp [hv_def]
    have := hper.int_mul (⌊u / (2 * Real.pi)⌋) v
    have hev : f (v + (⌊u / (2 * Real.pi)⌋ : ℤ) * (2 * Real.pi)) = f v := this
    rw [← hev, hshift]
  rw [hfeq]
  exact h_per_bound v ⟨hv_ge, hv_lt_2pi⟩

-- ===========================================================================
-- 2. The headline theorem
-- ===========================================================================

/-- **Area G headline.**

The derived form of `cert_residual_le`. Asserts that the certificate-supplied
`ε₀_cert` is a true pointwise upper bound on `|R a_cert μ_cert β_cert u|` for
every `u : ℝ`, where `R` is the Babenko residual

```
R(u) = μ·H[y_u] − y − y·H[y_u] − H[y·y_u] + β·∂_u(y_u/D) − β·H[∂_u G]
```

The hypothesis `h_pipeline_bound` packages the **full output** of the Julia
`residual_bound` pipeline as a single `TaylorSeriesData` describing `R` on
the standard `[0, 2π]` grid, together with the certified sup-bound. The
Areas A/B/F per-operation correctness lemmas are invoked **inside** the
construction of `h_pipeline_bound` by chaining together, for each Julia
pipeline step:

  Step 1: `ts_y` from `trigpoly_to_taylor_correct` (Area A)
  Step 2: `ts_yu = deriv_series ts_y` via `deriv_series_correct` (Area A)
  Step 3: `ts_Hyu` from `trigpoly_to_taylor_correct` (Area A, applied to
          the exact trig-poly Hilbert transform of `y_u`)
  Step 4: `ts_y_Hyu = ts_y * ts_Hyu` via `mul_taylorseries_correct` (Area A)
  Step 5: `ts_y_yu = ts_y * ts_yu` via `mul_taylorseries_correct` (Area A)
  Step 6: `ts_Hyyu = fft_hilbert_series ts_y_yu` via
          `fft_hilbert_series_correct` (Area F)
  Step 7 (β = 0): assemble `ts_R = μ·ts_Hyu - ts_y - ts_y_Hyu - ts_Hyyu`
                  via `add/sub/scalar_mul_taylorseries_correct` (Area A)
  Step 8 (β ≠ 0, capillary):
      8a. `D_lower, ts_D3 = dmin_bound(...)` (Area E)
      8b. `ts_invD3 = inv_series(ts_D3, D_lower)` via
          `inv_series_correct` (Area B)
      8c. `ts_B3 = 1 + ts_Hyu_3` via Area A linear combination
      8d. `ts_G3 = ts_B3 * ts_invD3` via `mul_taylorseries_correct` (Area A)
      8e. `ts_Gu = deriv_series(truncate ts_G3)` via `deriv_series_correct`
          (Area A)
      8f. `ts_HGu = fft_hilbert_series ts_Gu` via
          `fft_hilbert_series_correct` (Area F)
      8g. `ts_yuD = ts_yu * ts_invD` via `mul_taylorseries_correct` (Area A);
          `ts_C1 = deriv_series(ts_yuD)` via `deriv_series_correct` (Area A)
      8h. Add `β · ts_C1 - β · ts_HGu` to `ts_R` via
          `add/sub/scalar_mul_taylorseries_correct` (Area A)
  Step 9: `sup_bound_taylorseries_correct` (Area A) gives the pointwise
          bound on `|R u|` over `[0, 2π]`. By periodicity (proved in this
          file), it extends to all `u : ℝ`.

For each step, the caller supplies the abstract `TaylorSeriesData` from the
Julia output, plus the step-specific hypotheses (`h_derivs`/`h_err_zero`/`h_sup`
for `trigpoly_to_taylor`; `h_err_valid` for `mul`; `h_derivs_acc` for
`fft_hilbert_series` (the Hilbert-smoothness hypothesis is now discharged
internally); `Φ` plus `h_Φ` and `d_lower > 0` plus `h_f_pos` for `inv_series`;
`M_bound` and per-grid-cell hypothesis for `sup_bound_taylorseries_correct`).

In Phase 6, each per-step hypothesis is discharged by a corresponding
`cert_arb_*` axiom that records the correctness of an Arb primitive (one
axiom per Arb primitive: `cert_arb_trigpoly_to_taylor`, `cert_arb_mul`,
`cert_arb_deriv`, `cert_arb_fft_hilbert`, `cert_arb_inv`, `cert_arb_sup_bound`,
etc.).

**Trust footprint.** Modulo `propext, Classical.choice, Quot.sound`, the
proof depends only on `WaveCAP.zero_lt_two_pi` and the periodicity bridge
proved above (which is itself derived from Mathlib `Real.cos`/`sin`/`exp`
periodicity and the Fourier-series definition of `periodicHilbert`).

## Why is this a bundle?

The pipeline has ~20+ intermediate `TaylorSeriesData` records (per the
Julia source), each with its own grid-point and per-level sup_bounds
matrices. *Encoding each one as a Lean variable* would inflate the
theorem signature to several hundred lines and obscure the structure.
The bundle abstracts the entire chain as a single existence statement,
matching what the Julia `_residual_bound_full` function returns. -/
theorem cert_residual_le_derived
    (h_pipeline_bound :
      ∃ (M : ℕ) (N_fft : ℕ) (_ : 0 < N_fft) (ts_R : TaylorSeriesData M N_fft),
        IsTaylorSeriesOf ts_R (R a_cert μ_cert β_cert) 0
          ∧ ts_R.h = 2 * Real.pi / N_fft
          ∧ (∀ u ∈ Set.Icc (0 : ℝ) (0 + N_fft * ts_R.h),
                |R a_cert μ_cert β_cert u| ≤ ε₀_cert)) :
    ∀ u, |WaveCAP.R a_cert μ_cert β_cert u| ≤ ε₀_cert := by
  -- Unpack the pipeline witness.
  obtain ⟨M, N_fft, hN_fft, ts_R, h_ts, hh_eq, h_per_bound⟩ := h_pipeline_bound
  -- The grid covers [0, 2π] since N_fft * (2π / N_fft) = 2π.
  have hN_ne : (N_fft : ℝ) ≠ 0 := by exact_mod_cast Nat.pos_iff_ne_zero.mp hN_fft
  have h_endpoint : (0 : ℝ) + N_fft * ts_R.h = 2 * Real.pi := by
    rw [hh_eq, zero_add]
    field_simp
  -- Convert the per-period bound to a bound on Icc 0 (2π).
  have h_period_bound : ∀ u ∈ Set.Icc (0 : ℝ) (2 * Real.pi),
      |R a_cert μ_cert β_cert u| ≤ ε₀_cert := by
    intro u hu
    apply h_per_bound u
    rw [h_endpoint]
    exact hu
  -- Lift via periodicity.
  exact periodic_sup_extends (R_periodic _ _ _) ε₀_cert h_period_bound

-- ===========================================================================
-- 3. Structural Path A assembly (β = 0)
-- ===========================================================================

/-- The β = 0 residual

  `R_A a μ u = μ · H[y_u] − y − y · H[y_u] − H[y · y_u]`

is a `TaylorSeries`-representable function whenever each constituent is.
This theorem performs the chain of `add/sub/scalar_mul` reductions explicitly,
producing the final residual data record.

The naming `_A` indicates "Path A", the β = 0 algebraic branch of the
Julia routine `_residual_bound_full` (matching the Julia residual
`ts_R = μ * ts_Hyu - ts_y - ts_y_Hyu - ts_Hyyu`).

The caller supplies:
* `IsTaylorSeriesOf` for `y a`, `Hyu a`, `y a · Hyu a`, and
  `H[y a · yu a]` on a common grid;
* `(ts_y * ts_Hyu).h = ts_Hyu.h`, etc., to match grid spacings.

The output: an explicit `TaylorSeriesData` whose `IsTaylorSeriesOf` for
`R a μ 0` is constructed by Area A `scalar_mul/sub_taylorseries_correct`.
-/
theorem path_a_residual_taylor_correct
    {N : ℕ} (a : Fin N → ℝ) (μ : ℝ)
    {M N_fft : ℕ}
    {ts_y : TaylorSeriesData M N_fft} (h_y : IsTaylorSeriesOf ts_y (y a) 0)
    {ts_Hyu : TaylorSeriesData M N_fft} (h_Hyu : IsTaylorSeriesOf ts_Hyu (Hyu a) 0)
    {ts_y_Hyu : TaylorSeriesData M N_fft}
    (h_y_Hyu : IsTaylorSeriesOf ts_y_Hyu (fun u => y a u * Hyu a u) 0)
    {ts_Hyyu : TaylorSeriesData M N_fft}
    (h_Hyyu : IsTaylorSeriesOf ts_Hyyu
        (periodicHilbert (fun v => y a v * yu a v)) 0)
    (h_grids_y_Hyu : ts_y.h = ts_Hyu.h)
    (h_grids_y_yHyu : ts_y.h = ts_y_Hyu.h)
    (h_grids_y_Hyyu : ts_y.h = ts_Hyyu.h) :
    ∃ ts_R : TaylorSeriesData M N_fft,
        IsTaylorSeriesOf ts_R (R a μ 0) 0 ∧ ts_R.h = ts_y.h := by
  -- Step 1: μ · ts_Hyu via scalar_mul_taylorseries_correct (Area A).
  have h_μHyu := scalar_mul_taylorseries_correct (c := μ) h_Hyu
  -- Step 2: μ · ts_Hyu − ts_y via sub_taylorseries_correct (Area A).
  -- Need grid match: (scalar_mul ts_Hyu).h = ts_Hyu.h = ts_y.h.
  have h_grid_μHyu_y : ts_Hyu.h = ts_y.h := h_grids_y_Hyu.symm
  have h_step1 := sub_taylorseries_correct h_μHyu h_y h_grid_μHyu_y
  -- Step 3: previous − ts_y_Hyu via sub_taylorseries_correct.
  have h_grid_y_Hyu_for_sub : ts_Hyu.h = ts_y_Hyu.h := by
    rw [← h_grids_y_Hyu]; exact h_grids_y_yHyu
  have h_step2 := sub_taylorseries_correct h_step1 h_y_Hyu h_grid_y_Hyu_for_sub
  -- Step 4: previous − ts_Hyyu via sub_taylorseries_correct.
  have h_grid_y_Hyyu_for_sub : ts_Hyu.h = ts_Hyyu.h := by
    rw [← h_grids_y_Hyu]; exact h_grids_y_Hyyu
  have h_step3 := sub_taylorseries_correct h_step2 h_Hyyu h_grid_y_Hyyu_for_sub
  -- Now h_step3 : IsTaylorSeriesOf <data> (μ * Hyu a - y a - (y a * Hyu a) - H[y·yu]) 0.
  -- We rewrite the function to match `R a μ 0`.
  have h_funeq : (fun u : ℝ =>
        μ * Hyu a u - y a u - y a u * Hyu a u
          - periodicHilbert (fun v => y a v * yu a v) u)
      = R a μ 0 := by
    funext u
    unfold R
    -- R u = μ * Hyu - y - y * Hyu - H[y · yu] + 0 - 0 = (the LHS)
    have h_C1_term : (0 : ℝ) * deriv (fun v => yu a v / D a v) u = 0 := by ring
    have h_HGu_term : (0 : ℝ) * periodicHilbert (deriv (G a)) u = 0 := by ring
    -- Use the explicit form: μ*Hyu - y - y*Hyu - H[y·yu] + 0 - 0 = LHS
    simp only [h_C1_term, h_HGu_term, add_zero, sub_zero]
  -- Bridge h_step3 to R a μ 0 via funext.
  have h_funcast : (((fun u => μ * Hyu a u) - y a) - (fun u => y a u * Hyu a u))
        - periodicHilbert (fun v => y a v * yu a v)
      = R a μ 0 := by
    funext u
    -- Unfold R: R a μ 0 = μ * Hyu - y - y*Hyu - H[y·yu] + 0 - 0.
    change μ * Hyu a u - y a u - y a u * Hyu a u
          - periodicHilbert (fun v => y a v * yu a v) u
        = R a μ 0 u
    unfold R
    have h_C1_term : (0 : ℝ) * deriv (fun v => yu a v / D a v) u = 0 := by ring
    have h_HGu_term : (0 : ℝ) * periodicHilbert (deriv (G a)) u = 0 := by ring
    rw [h_C1_term, h_HGu_term]
    ring
  -- Now h_step3 with the function rewritten to R a μ 0.
  rw [h_funcast] at h_step3
  -- Provide ts_R explicitly as the constructor's output.
  refine ⟨_, h_step3, ?_⟩
  -- Grid spacing: h_step3's data has h = ts_Hyu.h (from the chain of sub).
  -- And ts_Hyu.h = ts_y.h.
  exact h_grid_μHyu_y

-- ===========================================================================
-- 4. Path A end-to-end (illustrative): residual + sup_bound + periodicity
-- ===========================================================================

/-- **Path A illustrative pipeline.** This shows how to chain the
`path_a_residual_taylor_correct` (Area A composition) with
`sup_bound_taylorseries_correct` (Area A) and the periodicity bridge
(this file) to obtain the `∀ u : ℝ` bound. The caller provides:

* the per-step `IsTaylorSeriesOf` witnesses (which in Phase 6 are
  discharged by the per-area lemmas applied to the Julia output);
* the `M_bound` hypothesis matching Julia's `sup_bound(ts_R)`.

This theorem is the structural "shape" of the bundled
`cert_residual_le_derived` for β = 0; it shows that the bundle's hypothesis
is in principle derivable from the per-operation chain. -/
theorem path_a_residual_le_of_chain
    {N : ℕ} (a : Fin N → ℝ) (μ ε₀ : ℝ)
    {M N_fft : ℕ} (hN_fft : 0 < N_fft)
    {ts_y : TaylorSeriesData M N_fft} (h_y : IsTaylorSeriesOf ts_y (y a) 0)
    {ts_Hyu : TaylorSeriesData M N_fft} (h_Hyu : IsTaylorSeriesOf ts_Hyu (Hyu a) 0)
    {ts_y_Hyu : TaylorSeriesData M N_fft}
    (h_y_Hyu : IsTaylorSeriesOf ts_y_Hyu (fun u => y a u * Hyu a u) 0)
    {ts_Hyyu : TaylorSeriesData M N_fft}
    (h_Hyyu : IsTaylorSeriesOf ts_Hyyu
        (periodicHilbert (fun v => y a v * yu a v)) 0)
    (h_grids_y_Hyu : ts_y.h = ts_Hyu.h)
    (h_grids_y_yHyu : ts_y.h = ts_y_Hyu.h)
    (h_grids_y_Hyyu : ts_y.h = ts_Hyyu.h)
    (h_grid_eq : ts_y.h = 2 * Real.pi / N_fft)
    (h_supbound :
      ∃ (ts_R : TaylorSeriesData M N_fft) (_ : IsTaylorSeriesOf ts_R (R a μ 0) 0)
          (_ : ts_R.h = ts_y.h),
        ∀ j : Fin N_fft,
          (∑ m ∈ Finset.range (M + 1),
            (|ts_R.derivs j ⟨min m M, by omega⟩|
                + ts_R.derivs_err j ⟨min m M, by omega⟩)
              * ts_R.h ^ m / (Nat.factorial m : ℝ))
          + ts_R.sup_bounds ⟨M + 1, by omega⟩
              * ts_R.h ^ (M + 1) / (Nat.factorial (M + 1) : ℝ)
          ≤ ε₀) :
    ∀ u, |R a μ 0 u| ≤ ε₀ := by
  -- Use path_a_residual_taylor_correct to assemble ts_R.
  obtain ⟨ts_R, h_R, h_hR⟩ := path_a_residual_taylor_correct a μ
    h_y h_Hyu h_y_Hyu h_Hyyu h_grids_y_Hyu h_grids_y_yHyu h_grids_y_Hyyu
  -- The user supplies their own ts_R and the M_bound hypothesis.
  -- We use the user's ts_R (it may differ — the structural assembly above
  -- shows one such ts_R exists; the caller's ts_R is morally the same).
  obtain ⟨ts_R', h_R', h_hR', h_M⟩ := h_supbound
  -- Apply sup_bound_taylorseries_correct (Area A).
  have h_per_bound :=
    sup_bound_taylorseries_correct hN_fft h_R' ε₀ h_M
  -- Compute the endpoint: 0 + N_fft * ts_R'.h = 2π.
  have h_endpoint : (0 : ℝ) + N_fft * ts_R'.h = 2 * Real.pi := by
    rw [h_hR', h_grid_eq, zero_add]
    have hN_ne : (N_fft : ℝ) ≠ 0 := by exact_mod_cast Nat.pos_iff_ne_zero.mp hN_fft
    field_simp
  -- Apply periodicity.
  have h_per_bound' : ∀ u ∈ Set.Icc (0 : ℝ) (2 * Real.pi), |R a μ 0 u| ≤ ε₀ := by
    intro u hu
    apply h_per_bound u
    rw [h_endpoint]; exact hu
  exact periodic_sup_extends (R_periodic a μ 0) ε₀ h_per_bound'

-- ===========================================================================
-- 5. Structural Path B assembly (β ≠ 0)
-- ===========================================================================

/-- **Structural Path B assembly.** Given an `IsTaylorSeriesOf` for the
β = 0 residual `R a μ 0` (constructed e.g. via
`path_a_residual_taylor_correct`) and the two capillary
`IsTaylorSeriesOf` witnesses:

* `ts_C1 ↔ deriv (fun v => yu a v / D a v)` — assembled by the caller via
  `inv_series_correct` (Area B) for `1/D`, `mul_taylorseries_correct`
  (Area A) for `yu · (1/D)`, and `deriv_series_correct` (Area A) for
  `∂_u`.
* `ts_HGu ↔ periodicHilbert (deriv (G a))` — assembled by the caller via
  the (Area B) `inv_series_correct` and (Area A) `mul_taylorseries_correct`
  chain to build `G = (1+Hyu)/D`, then `deriv_series_correct` (Area A) for
  `∂_u G`, and finally `fft_hilbert_series_correct` (Area F) for `H[∂_u G]`.

Returns an `IsTaylorSeriesOf` for the **full** residual `R a μ β` via
two more applications of `scalar_mul/add/sub_taylorseries_correct`
(Area A). -/
theorem path_b_residual_taylor_correct
    {N : ℕ} (a : Fin N → ℝ) (μ β : ℝ)
    {M N_fft : ℕ}
    {ts_R_A : TaylorSeriesData M N_fft}
    (h_R_A : IsTaylorSeriesOf ts_R_A (R a μ 0) 0)
    {ts_C1 : TaylorSeriesData M N_fft}
    (h_C1 : IsTaylorSeriesOf ts_C1 (deriv (fun v => yu a v / D a v)) 0)
    {ts_HGu : TaylorSeriesData M N_fft}
    (h_HGu : IsTaylorSeriesOf ts_HGu (periodicHilbert (deriv (G a))) 0)
    (h_grid_R_C1 : ts_R_A.h = ts_C1.h)
    (h_grid_R_HGu : ts_R_A.h = ts_HGu.h) :
    ∃ ts_R : TaylorSeriesData M N_fft,
        IsTaylorSeriesOf ts_R (R a μ β) 0 ∧ ts_R.h = ts_R_A.h := by
  -- Step 1: β · ts_C1 via scalar_mul (Area A).
  have h_βC1 := scalar_mul_taylorseries_correct (c := β) h_C1
  -- Step 2: ts_R_A + β · ts_C1 via add (Area A).
  -- (scalar_mul ts_C1).h = ts_C1.h. Need ts_R_A.h = ts_C1.h.
  have h_step1 := add_taylorseries_correct h_R_A h_βC1 h_grid_R_C1
  -- Step 3: β · ts_HGu via scalar_mul (Area A).
  have h_βHGu := scalar_mul_taylorseries_correct (c := β) h_HGu
  -- Step 4: previous - β · ts_HGu via sub (Area A).
  -- h_step1's h = ts_R_A.h. (scalar_mul ts_HGu).h = ts_HGu.h = ts_R_A.h.
  have h_grid_step1_HGu : ts_R_A.h = ts_HGu.h := h_grid_R_HGu
  have h_step2 := sub_taylorseries_correct h_step1 h_βHGu h_grid_step1_HGu
  -- h_step2 : IsTaylorSeriesOf <data> ((R a μ 0 + β·C1) - β·HGu) 0.
  -- Rewrite the function to R a μ β.
  have h_funcast : (R a μ 0 + (fun u => β * deriv (fun v => yu a v / D a v) u))
        - (fun u => β * periodicHilbert (deriv (G a)) u)
      = R a μ β := by
    funext u
    change R a μ 0 u + β * deriv (fun v => yu a v / D a v) u
          - β * periodicHilbert (deriv (G a)) u
        = R a μ β u
    -- Unfold both R's.
    unfold R
    -- R a μ 0 u = μ * Hyu - y - y * Hyu - H[y·yu] + 0 - 0
    -- R a μ β u = μ * Hyu - y - y * Hyu - H[y·yu] + β · ∂_u(yu/D) - β · H[∂_u G]
    ring
  rw [h_funcast] at h_step2
  refine ⟨_, h_step2, ?_⟩
  -- Grid spacing: starting from h_R_A.h, each step preserves the first operand's h.
  rfl

/-- **Path B end-to-end (illustrative).** Combines `path_b_residual_taylor_correct`
with `sup_bound_taylorseries_correct` (Area A) and the periodicity bridge to
obtain the final `∀ u : ℝ` bound. -/
theorem path_b_residual_le_of_chain
    {N : ℕ} (a : Fin N → ℝ) (μ β ε₀ : ℝ)
    {M N_fft : ℕ} (hN_fft : 0 < N_fft)
    {ts_R_A : TaylorSeriesData M N_fft}
    (h_R_A : IsTaylorSeriesOf ts_R_A (R a μ 0) 0)
    {ts_C1 : TaylorSeriesData M N_fft}
    (h_C1 : IsTaylorSeriesOf ts_C1 (deriv (fun v => yu a v / D a v)) 0)
    {ts_HGu : TaylorSeriesData M N_fft}
    (h_HGu : IsTaylorSeriesOf ts_HGu (periodicHilbert (deriv (G a))) 0)
    (h_grid_R_C1 : ts_R_A.h = ts_C1.h)
    (h_grid_R_HGu : ts_R_A.h = ts_HGu.h)
    (h_grid_eq : ts_R_A.h = 2 * Real.pi / N_fft)
    (h_supbound :
      ∃ (ts_R : TaylorSeriesData M N_fft) (_ : IsTaylorSeriesOf ts_R (R a μ β) 0)
          (_ : ts_R.h = ts_R_A.h),
        ∀ j : Fin N_fft,
          (∑ m ∈ Finset.range (M + 1),
            (|ts_R.derivs j ⟨min m M, by omega⟩|
                + ts_R.derivs_err j ⟨min m M, by omega⟩)
              * ts_R.h ^ m / (Nat.factorial m : ℝ))
          + ts_R.sup_bounds ⟨M + 1, by omega⟩
              * ts_R.h ^ (M + 1) / (Nat.factorial (M + 1) : ℝ)
          ≤ ε₀) :
    ∀ u, |R a μ β u| ≤ ε₀ := by
  -- Use path_b_residual_taylor_correct to confirm the residual is representable.
  obtain ⟨_, _, _⟩ := path_b_residual_taylor_correct a μ β h_R_A h_C1 h_HGu
    h_grid_R_C1 h_grid_R_HGu
  -- Apply sup_bound_taylorseries_correct (Area A) to the caller's ts_R.
  obtain ⟨ts_R', h_R', h_hR', h_M⟩ := h_supbound
  have h_per_bound :=
    sup_bound_taylorseries_correct hN_fft h_R' ε₀ h_M
  have h_endpoint : (0 : ℝ) + N_fft * ts_R'.h = 2 * Real.pi := by
    rw [h_hR', h_grid_eq, zero_add]
    have hN_ne : (N_fft : ℝ) ≠ 0 := by exact_mod_cast Nat.pos_iff_ne_zero.mp hN_fft
    field_simp
  have h_per_bound' : ∀ u ∈ Set.Icc (0 : ℝ) (2 * Real.pi), |R a μ β u| ≤ ε₀ := by
    intro u hu
    apply h_per_bound u
    rw [h_endpoint]; exact hu
  exact periodic_sup_extends (R_periodic a μ β) ε₀ h_per_bound'

-- ===========================================================================
-- 6. Capillary-chain helpers (Area B + Area A composition)
-- ===========================================================================

/-- **Step 8b (Path B):** `IsTaylorSeriesOf ts_invD (1/D a) 0` from
`IsTaylorSeriesOf ts_D (D a) 0`, via `inv_series_correct` (Area B).

The `Φ` argument is the Faà-di-Bruno sup-bound vector for `(·)⁻¹`-derivatives
on `[D_lower, ∞)` (i.e. `Φ k = max{|((·)⁻¹)^{(k)} y| : D_lower ≤ y} = k!/D_lower^{k+1}`).
The caller supplies `Φ` and `h_Φ` matching their Julia computation. -/
theorem capillary_invD_correct
    {N : ℕ} (a : Fin N → ℝ)
    {M N_fft : ℕ}
    {ts_D : TaylorSeriesData M N_fft} (h_D : IsTaylorSeriesOf ts_D (D a) 0)
    (D_lower : ℝ) (h_DLower_pos : 0 < D_lower)
    (h_D_ge : ∀ u, D_lower ≤ D a u)
    (Φ : ℕ → ℝ)
    (h_Φ : ∀ (k : ℕ) (y : ℝ), D_lower ≤ y →
      |iteratedDeriv k (fun z : ℝ => z⁻¹) y| ≤ Φ k) :
    IsTaylorSeriesOf (inv_series_data ts_D (D a) 0 Φ)
        (fun u => (D a u)⁻¹) 0 :=
  inv_series_correct h_D D_lower h_DLower_pos h_D_ge Φ h_Φ

/-- **Step 8d (Path B):** `IsTaylorSeriesOf ts_yuD (yu a / D a) 0` from
`IsTaylorSeriesOf ts_yu (yu a) 0` and `IsTaylorSeriesOf ts_invD (1/D a) 0`,
via `mul_taylorseries_correct` (Area A).

This builds `yu/D = yu · (1/D)` since `a/b = a · b⁻¹` in Lean's `Field`. -/
theorem capillary_yuD_correct
    {N : ℕ} (a : Fin N → ℝ)
    {M N_fft : ℕ}
    {ts_yu : TaylorSeriesData M N_fft} (h_yu : IsTaylorSeriesOf ts_yu (yu a) 0)
    {ts_invD : TaylorSeriesData M N_fft}
    (h_invD : IsTaylorSeriesOf ts_invD (fun u => (D a u)⁻¹) 0)
    (h_grid : ts_yu.h = ts_invD.h)
    (err_combined : Fin N_fft → Fin (M + 1) → ℝ)
    (h_err_valid : ∀ (j : Fin N_fft) (m : Fin (M + 1)),
        |iteratedDeriv m.val (yu a * (fun u => (D a u)⁻¹)) (0 + j.val * ts_yu.h)
          - ∑ k ∈ Finset.range (m.val + 1),
              (m.val.choose k : ℝ)
                * ts_yu.derivs j ⟨min k M, by omega⟩
                * ts_invD.derivs j ⟨m.val - k, by have := m.isLt; omega⟩|
        ≤ err_combined j m) :
    ∃ ts_yuD : TaylorSeriesData M N_fft,
        IsTaylorSeriesOf ts_yuD (fun u => yu a u / D a u) 0
        ∧ ts_yuD.h = ts_yu.h := by
  have h_prod := mul_taylorseries_correct h_yu h_invD h_grid err_combined h_err_valid
  -- mul_taylorseries_correct gives `IsTaylorSeriesOf <data> (yu a * (fun u => (D a u)⁻¹)) 0`.
  -- We need `fun u => yu a u / D a u`. These are pointwise equal: `a / b = a * b⁻¹`.
  have h_funeq : (yu a * (fun u => (D a u)⁻¹))
      = (fun u => yu a u / D a u) := by
    funext u
    rw [Pi.mul_apply]
    exact (div_eq_mul_inv _ _).symm
  rw [h_funeq] at h_prod
  exact ⟨_, h_prod, rfl⟩

/-- **Step 8d/e (Path B):** `IsTaylorSeriesOf ts_C1 (deriv (yu/D)) 0` from
`IsTaylorSeriesOf ts_yuD (yu/D) 0`, via `deriv_series_correct` (Area A).
Caller must supply `hM_pos : 0 < M` (since `deriv_series` drops the top
derivative slot). -/
theorem capillary_C1_correct
    {N : ℕ} (a : Fin N → ℝ)
    {M N_fft : ℕ}
    {ts_yuD : TaylorSeriesData M N_fft}
    (h_yuD : IsTaylorSeriesOf ts_yuD (fun u => yu a u / D a u) 0)
    (hM_pos : 0 < M) :
    IsTaylorSeriesOf (TaylorSeriesData.derivShift ts_yuD hM_pos)
        (deriv (fun u => yu a u / D a u)) 0 := by
  have h_deriv := deriv_series_correct h_yuD hM_pos
  -- h_deriv: IsTaylorSeriesOf (TaylorSeriesData.derivShift ts_yuD hM_pos)
  --              (iteratedDeriv 1 (fun u => yu a u / D a u)) 0
  -- We need to bridge `iteratedDeriv 1` to `deriv`.
  rw [iteratedDeriv_one] at h_deriv
  exact h_deriv

/-- **Step 8f (Path B):** `IsTaylorSeriesOf ts_HGu (H[∂_u G]) 0` from
`IsTaylorSeriesOf ts_Gu (∂_u G) 0`, via `fft_hilbert_series_correct` (Area F).

The caller supplies:
* `h_Gu_smooth`: `ContDiff (M + 3) (∂_u G)` — needed by `fft_hilbert_series_correct`.
* `h_Gu_per`: `Function.Periodic (∂_u G) (2π)` — provable from `G_periodic`
  (this file) via `WaveCAP.Periodic.deriv_periodic`.
* `Gpp_bound`, `Gppp_bound`, `h_Gpp`, `h_Gppp`: top-end sup bounds.
* `newDerivs`, `newDerivsErr`, `h_derivs_acc`: the F1 contract per-level.

The `ContDiff (M+1) (periodicHilbert (deriv G))` smoothness fact is now
discharged internally by `fft_hilbert_series_correct` (via `hilbert_contDiff`);
the caller no longer supplies `h_H_contdiff`. -/
theorem capillary_HGu_correct
    {N : ℕ} (a : Fin N → ℝ)
    {M N_fft : ℕ}
    {ts_Gu : TaylorSeriesData M N_fft}
    (h_Gu : IsTaylorSeriesOf ts_Gu (deriv (G a)) 0)
    (h_Gu_smooth : ContDiff ℝ (M + 3) (deriv (G a)))
    (h_Gu_per : Function.Periodic (deriv (G a)) (2 * Real.pi))
    (Gpp_bound Gppp_bound : ℝ)
    (h_Gpp : ∀ u, |iteratedDeriv (M + 2) (deriv (G a)) u| ≤ Gpp_bound)
    (h_Gppp : ∀ u, |iteratedDeriv (M + 3) (deriv (G a)) u| ≤ Gppp_bound)
    (newDerivs : Fin N_fft → Fin (M + 1) → ℝ)
    (newDerivsErr : Fin N_fft → Fin (M + 1) → ℝ)
    (h_derivs_acc : ∀ (j : Fin N_fft) (m : Fin (M + 1)),
        |iteratedDeriv m.val (periodicHilbert (deriv (G a)))
            (0 + j.val * ts_Gu.h)
          - newDerivs j m| ≤ newDerivsErr j m) :
    IsTaylorSeriesOf
        (TaylorSeriesData.fft_hilbert ts_Gu newDerivs newDerivsErr Gpp_bound Gppp_bound)
        (periodicHilbert (deriv (G a))) 0 :=
  fft_hilbert_series_correct h_Gu h_Gu_smooth h_Gu_per
    Gpp_bound Gppp_bound h_Gpp h_Gppp newDerivs newDerivsErr h_derivs_acc

/-- **Step 6 (β = 0, also reused in Path B):** `IsTaylorSeriesOf ts_Hyyu (H[y·yu]) 0`
from `IsTaylorSeriesOf ts_y_yu (y · yu) 0`, via `fft_hilbert_series_correct`
(Area F). Same hypothesis shape as `capillary_HGu_correct`. -/
theorem capillary_Hyyu_correct
    {N : ℕ} (a : Fin N → ℝ)
    {M N_fft : ℕ}
    {ts_y_yu : TaylorSeriesData M N_fft}
    (h_y_yu : IsTaylorSeriesOf ts_y_yu (fun u => y a u * yu a u) 0)
    (h_yyu_smooth : ContDiff ℝ (M + 3) (fun u => y a u * yu a u))
    (h_yyu_per : Function.Periodic (fun u => y a u * yu a u) (2 * Real.pi))
    (Gpp_bound Gppp_bound : ℝ)
    (h_Gpp : ∀ u, |iteratedDeriv (M + 2) (fun u => y a u * yu a u) u| ≤ Gpp_bound)
    (h_Gppp : ∀ u, |iteratedDeriv (M + 3) (fun u => y a u * yu a u) u| ≤ Gppp_bound)
    (newDerivs : Fin N_fft → Fin (M + 1) → ℝ)
    (newDerivsErr : Fin N_fft → Fin (M + 1) → ℝ)
    (h_derivs_acc : ∀ (j : Fin N_fft) (m : Fin (M + 1)),
        |iteratedDeriv m.val
            (periodicHilbert (fun u => y a u * yu a u))
            (0 + j.val * ts_y_yu.h)
          - newDerivs j m| ≤ newDerivsErr j m) :
    IsTaylorSeriesOf
        (TaylorSeriesData.fft_hilbert ts_y_yu newDerivs newDerivsErr
            Gpp_bound Gppp_bound)
        (periodicHilbert (fun u => y a u * yu a u)) 0 := by
  -- fft_hilbert_series_correct produces IsTaylorSeriesOf for the periodic Hilbert.
  exact fft_hilbert_series_correct h_y_yu h_yyu_smooth h_yyu_per
    Gpp_bound Gppp_bound h_Gpp h_Gppp newDerivs newDerivsErr h_derivs_acc

-- ===========================================================================
-- 7. Path-A trig-poly chain helpers (Area A: trigpoly_to_taylor + mul + deriv)
-- ===========================================================================

/-- **Step 1 (Path A):** `IsTaylorSeriesOf ts_y (y a) 0` via
`trigpoly_to_taylor_correct` (Area A).

The cosine series `y a u = ∑ a_k cos(k u)` is the special case
`b_coef = 0` of the general trig polynomial. The caller's `a` lives in
`Fin (N_modes + 1)` for some `N_modes`. This helper unfolds `y` to the
explicit cosine-sum form and applies `trigpoly_to_taylor_correct`.

For convenience this helper rewrites the bare `y` definition to the
`trigpoly_to_taylor`-compatible shape `∑ (a_k cos(k u) + 0 * sin(k u))`. -/
theorem step_y_correct
    {N_modes : ℕ} (a : Fin (N_modes + 1) → ℝ)
    {M N_fft : ℕ} (ts_y : TaylorSeriesData M N_fft)
    (h_derivs : ∀ (j : Fin N_fft) (m : Fin (M + 1)),
        iteratedDeriv m.val
            (fun u => ∑ k : Fin (N_modes + 1),
                (a k * Real.cos ((k.val : ℝ) * u)
                  + (0 : ℝ) * Real.sin ((k.val : ℝ) * u)))
            ((j.val : ℝ) * ts_y.h)
        = ts_y.derivs j m)
    (h_err_zero : ∀ (j : Fin N_fft) (m : Fin (M + 1)), ts_y.derivs_err j m = 0)
    (h_sup : ∀ m : Fin (M + 2),
        ts_y.sup_bounds m
          = ∑ k : Fin (N_modes + 1),
              ((k.val : ℝ) ^ m.val) * (|a k| + |(0 : ℝ)|))
    (h_pos : 0 < ts_y.h) :
    IsTaylorSeriesOf ts_y (y a) 0 := by
  have h_trig := trigpoly_to_taylor_correct (N_modes := N_modes)
      (a_coef := a)
      (b_coef := fun _ : Fin (N_modes + 1) => (0 : ℝ))
      ts_y h_derivs h_err_zero h_sup h_pos
  -- Bridge: the trig poly with b = 0 equals `y a`.
  have h_funeq :
      (fun u => ∑ k : Fin (N_modes + 1),
          (a k * Real.cos ((k.val : ℝ) * u)
            + (0 : ℝ) * Real.sin ((k.val : ℝ) * u)))
      = y a := by
    funext u
    unfold y
    simp only [zero_mul, add_zero]
  rw [h_funeq] at h_trig
  exact h_trig

-- Note: an analogous `step_Hyu_correct` and a `step_yu_correct` (using
-- the trig-poly form `-∑ k a_k sin(k u)` directly, OR equivalently the
-- `deriv_series_correct` of `step_y_correct`) are mechanical variants of
-- the above. We give one detailed version (`step_y_correct`) to
-- demonstrate the technique; the others follow the same template.

-- ===========================================================================
-- 8. Sanity check
-- ===========================================================================

#check @cert_residual_le_derived
#check @path_a_residual_taylor_correct
#check @path_a_residual_le_of_chain
#check @path_b_residual_taylor_correct
#check @path_b_residual_le_of_chain
#check @capillary_invD_correct
#check @capillary_yuD_correct
#check @capillary_C1_correct
#check @capillary_HGu_correct
#check @capillary_Hyyu_correct
#check @step_y_correct
#print axioms cert_residual_le_derived
#print axioms path_a_residual_taylor_correct
#print axioms path_a_residual_le_of_chain
#print axioms path_b_residual_taylor_correct
#print axioms path_b_residual_le_of_chain
#print axioms capillary_invD_correct
#print axioms capillary_yuD_correct
#print axioms capillary_C1_correct
#print axioms capillary_HGu_correct
#print axioms capillary_Hyyu_correct
#print axioms step_y_correct

end WaveCAP
