import WaveCAP.Certificate
import WaveCAP.Definitions
import WaveCAP.TaylorSeriesCorrectness
import WaveCAP.SeriesCorrectness
import WaveCAP.FFTHilbertCorrectness
import WaveCAP.DminBoundCorrectness
import WaveCAP.ResidualAssemblyCorrectness
import WaveCAP.RSmoothness

/-!
# Phase 6 — Atomic `cert_arb_*` axioms

This file decomposes the two monolithic trust-anchor axioms

  * `WaveCAP.Certificate.cert_residual_le`
  * `WaveCAP.Certificate.cert_D_ge`

into a small set of **atomic axioms**, one per Arb primitive used in the
Julia pipeline (`BabenkoRigorous/src/ResidualBound.jl`). Each atomic axiom
asserts that one specific Julia routine, applied to suitable inputs,
produces output satisfying the corresponding Lean correctness predicate.

The two `cert_residual_le` / `cert_D_ge` declarations are *retired as
axioms* (the `axiom` keywords have been removed from
`WaveCAP/Certificate.lean`) and re-introduced here as **derived theorems**
with the same name and statement, built from the atoms plus the
Lean-proved correctness chain in Areas A/B/E/F/G.

## Decomposition philosophy

**Function-style atomic axioms.** Each per-primitive axiom abstracts over
a class of inputs (a function `f`, an `IsTaylorSeriesOf` witness, etc.)
and produces the corresponding output existential. This mirrors the Lean
`<primitive>_correct` theorem signatures but **omits** the per-call
caller hypotheses that the Lean proof requires (about `derivs`,
`derivs_err`, `sup_bounds` matching what Julia computed). Those caller
hypotheses are exactly what the axiom asserts is true on the Julia side —
they are the Arb-level trust trade.

**Pipeline-level axiom for the final sup-bound.** Even with the
per-primitive axioms in hand, the quantitative `‖R‖∞ ≤ ε₀_cert` bound
must be axiomatised at the pipeline level: it asserts that the Julia
`sup_bound` routine applied to the certificate's `ts_R` produces a
real number `≤ ε₀_cert`. This is `cert_arb_sup_bound_le_eps0`.

**Phase-6 sharpening (2026-05-24)**: `cert_arb_residual_TS` is no
longer an axiom — it is now a derived theorem. The chain rule for `R`'s
smoothness was the obstacle; this has been discharged by
`WaveCAP.R_contDiff` in `WaveCAP/RSmoothness.lean`. The derivation
explicitly composes the per-primitive axioms via the Area-A
`add/sub/scalar_mul_taylorseries_correct` combinators, so the
`#print axioms` of `cert_residual_le` now lists **all five**
per-primitive axioms.

## What the trust base actually is (audit 2026-07-03; de-axiomatization
## completed the same day)

* **Quantitative trust = two shipped scalars, and after targets P2–P6 of
  `docs/audit_2026-07-03.md` these are the ONLY axioms.** Only
  `cert_arb_dmin_bound` (`DLower_cert ≤ D` everywhere) and
  `cert_arb_sup_bound_le_eps0` (`|R| ≤ ε₀_cert` on a period) mention
  Julia's computed numbers. Trusting these two is equivalent to trusting
  the retired monolithic pair `cert_D_ge` / `cert_residual_le`.
  `#print axioms residual_bounded_by_certificate` now shows exactly
  these two plus the three core Lean axioms.
* **The five per-primitive statements are derived theorems.** Their
  conclusions are bare existentials (`∃ ts, IsTaylorSeriesOf ts … `) that
  never reference Julia's output arrays; each is proved via the
  exact-data witness (`derivs` = true derivatives, `derivs_err = 0`,
  `sup_bounds` = honest bounds from compactness/Leibniz/Faà-di-Bruno).
  They carried **no Arb trust content**; they remain stated in the
  Julia-pipeline shape so the derivation of `cert_arb_residual_TS`
  mirrors `_residual_bound_full` step by step.
* **`cert_arb_sup_bound_le_eps0` is pipeline-level, not per-primitive.**
  Because it universally quantifies over the `ts_R` witness, its
  hypotheses do not tie it to Julia's actual computed data; its content
  is exactly `|R| ≤ ε₀_cert` on `[0, 2π]`. A genuinely per-primitive
  quantitative decomposition would need axioms whose statements mention
  the concrete certificate arrays.
* Note the derivation applies `cert_arb_trigpoly_to_taylor` to `D` and
  `1 + Hyu`, which are not trig polynomials (Julia builds `ts_D` via
  `sqrt_series` inside `dmin_bound`); the theorem covers any smooth
  periodic function, so the step is sound, but the name-to-Julia-routine
  correspondence is looser than it suggests.

## Trust-anchor list

  * `cert_arb_dmin_bound`         — **axiom**. Julia's `dmin_bound`
                                    (= `cert_D_ge` content).
  * `cert_arb_sup_bound_le_eps0`  — **axiom**. Julia's `sup_bound` ≤
                                    shipped `ε₀_cert`.

  De-axiomatized 2026-07-03 (each a derived theorem, core-3 only):

  * `cert_arb_trigpoly_to_taylor` — exact data + periodic-compactness bounds.
  * `cert_arb_ts_mul`             — exact-data witness + Leibniz bound.
  * `cert_arb_ts_deriv`           — pure index shift.
  * `cert_arb_inv_series`         — `inv_series_data` + `k!/d^{k+1}` bound.
  * `cert_arb_fft_hilbert_series` — same engine as `trigpoly_to_taylor`,
                                    via `hilbert_contDiff` +
                                    `periodicHilbert_is_periodic`.
  * `cert_arb_residual_TS`        — composite `IsTaylorSeriesOf` for `R`
                                    (Phase 6).

## Cross-reference

* `BabenkoRigorous/src/ResidualBound.jl` — Julia source of the pipeline.
* `WaveCAP/Certificate.lean` — auto-generated certificate inputs; the old
  trust-anchor axioms have been retired (commented out) here.
* `WaveCAP/ResidualAssemblyCorrectness.lean` — Area G, which provides
  `cert_residual_le_derived` and the structural per-step theorems used here.
-/

namespace WaveCAP

open WaveCAP.Certificate Real

-- ===========================================================================
-- §1. Atomic axioms — one per Julia/Arb primitive
-- ===========================================================================

/-- Any smooth `2π`-periodic function admits an exact-data
`TaylorSeriesData` witness on any positively-spaced grid: true derivatives
at the grid points, zero radius, and a sup-bound vector obtained from
boundedness of each `iteratedDeriv m f` (continuous + `2π`-periodic, hence
bounded by its sup over the compact `[0, 2π]`). This is the common engine
behind the de-axiomatized `cert_arb_trigpoly_to_taylor` and
`cert_arb_fft_hilbert_series`. -/
theorem exists_isTaylorSeriesOf_of_contDiff_periodic
    {M N : ℕ} (f : ℝ → ℝ) (a h : ℝ) (h_pos : 0 < h)
    (h_smooth : ContDiff ℝ (M + 1) f)
    (h_per : Function.Periodic f (2 * Real.pi)) :
    ∃ ts : TaylorSeriesData M N, IsTaylorSeriesOf ts f a ∧ ts.h = h := by
  -- Each iterated derivative up to order M+1 is continuous and 2π-periodic,
  -- hence globally bounded by its sup over the compact `[0, 2π]`.
  have hbdd : ∀ m : Fin (M + 2), ∃ C : ℝ, ∀ x : ℝ,
      |iteratedDeriv m.val f x| ≤ C := by
    intro m
    have hmle : m.val ≤ M + 1 := Nat.lt_succ_iff.mp m.isLt
    have hcont : Continuous (iteratedDeriv m.val f) :=
      h_smooth.continuous_iteratedDeriv m.val (Nat.mono_cast hmle)
    obtain ⟨C, hC⟩ :=
      (isCompact_Icc (a := (0 : ℝ))
        (b := 2 * Real.pi)).exists_bound_of_continuousOn hcont.continuousOn
    refine ⟨C, fun x => ?_⟩
    obtain ⟨y, hy, hxy⟩ :=
      (Periodic.iteratedDeriv h_per m.val).exists_mem_Ico₀ Real.two_pi_pos x
    rw [← Real.norm_eq_abs, hxy]
    exact hC y (Set.mem_Icc_of_Ico hy)
  refine ⟨{ derivs := fun j m => iteratedDeriv m.val f (a + (j.val : ℝ) * h)
            derivs_err := fun _ _ => 0
            sup_bounds := fun m => Classical.choose (hbdd m)
            h := h },
          ⟨h_smooth, ?_, fun m x => Classical.choose_spec (hbdd m) x, h_pos⟩,
          rfl⟩
  -- grid_acc: exact data, zero radius.
  intro j m
  simp

/-- **Derived theorem `cert_arb_trigpoly_to_taylor`** (formerly an axiom;
statement repaired then de-axiomatized 2026-07-03, targets P1/P2 of
`docs/audit_2026-07-03.md`) — Julia's `trigpoly_to_taylor` correctness.

Given any smooth `2π`-periodic function `f : ℝ → ℝ` (Julia accepts
arbitrary `TrigPoly`s; for the signature we abstract to any smooth
periodic function), any Taylor order `M`, and any grid size `N > 0`,
there exists a `TaylorSeriesData M N` with `h = 2π/N` whose
`IsTaylorSeriesOf` relation with `f` holds: the exact-data witness (true
derivatives on the grid, zero radius, sup-bounds from compactness).

**The periodicity hypothesis is load-bearing, not a convenience.**
`IsTaylorSeriesOf` demands a *global* sup bound on `f` and its
derivatives (`sup_acc`, quantified over all `x : ℝ`). A merely-smooth
`f` need not admit one — `f = id` is `ContDiff` of every order yet
globally unbounded, so the pre-repair axiom (which lacked periodicity)
was false and made the axiom system inconsistent (audit 2026-07-03).
With periodicity each `iteratedDeriv m f` is continuous and
`2π`-periodic, hence bounded by its sup on the compact `[0, 2π]`.

The statement is a bare existential that never referenced Julia's
computed arrays, so it carried no Arb trust.
(`WaveCAP.trigpoly_to_taylor_correct` remains the record-level lemma for
Julia's actual `trigpoly_to_taylor` output.)

Cross-reference: `BabenkoRigorous/src/TaylorSeries.jl:852`
(`trigpoly_to_taylor`). -/
theorem cert_arb_trigpoly_to_taylor
    (f : ℝ → ℝ) (M N : ℕ) (hN : 0 < N) (a : ℝ)
    (h_smooth : ContDiff ℝ (M + 1) f)
    (h_per : Function.Periodic f (2 * Real.pi)) :
    ∃ ts : TaylorSeriesData M N,
      IsTaylorSeriesOf ts f a ∧ ts.h = 2 * Real.pi / N :=
  exists_isTaylorSeriesOf_of_contDiff_periodic f a (2 * Real.pi / (N : ℝ))
    (div_pos Real.two_pi_pos (by exact_mod_cast hN)) h_smooth h_per

/-- **Derived theorem `cert_arb_ts_mul`** (formerly an axiom;
de-axiomatized 2026-07-03, target P4 of `docs/audit_2026-07-03.md`) —
Julia's `TaylorSeries` Leibniz multiplication correctness.

Given two `TaylorSeriesData` records on a common grid that satisfy
`IsTaylorSeriesOf` for `f` and `g` respectively, a `TaylorSeriesData`
satisfying `IsTaylorSeriesOf` for `f * g` exists: the exact-data witness
(true derivatives on the grid, zero radius, Leibniz sup-bounds built from
the two input `sup_bounds` vectors via `WaveCAP.abs_iteratedDeriv_mul_le`).
The statement is a bare existential that never referenced Julia's computed
arrays, so it carried no Arb trust. (`WaveCAP.mul_taylorseries_correct`
remains the record-level lemma for Julia's actual `Base.:*` output.)

Cross-reference: `BabenkoRigorous/src/TaylorSeries.jl:399`
(`Base.:*(TS, TS)`). -/
theorem cert_arb_ts_mul
    {M N : ℕ} {ts1 ts2 : TaylorSeriesData M N} {f g : ℝ → ℝ} {a : ℝ}
    (h1 : IsTaylorSeriesOf ts1 f a) (h2 : IsTaylorSeriesOf ts2 g a)
    (_hh : ts1.h = ts2.h) :
    ∃ ts : TaylorSeriesData M N,
      IsTaylorSeriesOf ts (fun u => f u * g u) a ∧ ts.h = ts1.h := by
  refine ⟨{ derivs := fun j m =>
              iteratedDeriv m.val (fun u => f u * g u) (a + (j.val : ℝ) * ts1.h)
            derivs_err := fun _ _ => 0
            sup_bounds := fun m =>
              ∑ k ∈ Finset.range (m.val + 1),
                (m.val.choose k : ℝ)
                  * (fun k' => if hk' : k' < M + 2 then ts1.sup_bounds ⟨k', hk'⟩ else 0) k
                  * (fun k' => if hk' : k' < M + 2 then ts2.sup_bounds ⟨k', hk'⟩ else 0)
                      (m.val - k)
            h := ts1.h },
          ⟨h1.contdiff.mul h2.contdiff, ?_, ?_, h1.h_pos⟩, rfl⟩
  · -- grid_acc: exact data, zero radius.
    intro j m
    simp
  · -- sup_acc: Leibniz bound on |(f·g)^{(m)}| from the two sup_bounds vectors.
    intro m x
    have hmle : m.val ≤ M + 1 := Nat.lt_succ_iff.mp m.isLt
    have hf_at : ContDiffAt ℝ (m.val : ℕ) f x :=
      (h1.contdiff.of_le (Nat.mono_cast hmle)).contDiffAt
    have hg_at : ContDiffAt ℝ (m.val : ℕ) g x :=
      (h2.contdiff.of_le (Nat.mono_cast hmle)).contDiffAt
    have hF : ∀ k ≤ m.val, |iteratedDeriv k f x|
        ≤ (fun k' => if hk' : k' < M + 2 then ts1.sup_bounds ⟨k', hk'⟩ else 0) k := by
      intro k hk
      have hk2 : k < M + 2 := lt_of_le_of_lt hk m.isLt
      simp only [dif_pos hk2]
      exact h1.sup_acc ⟨k, hk2⟩ x
    have hG : ∀ k ≤ m.val, |iteratedDeriv k g x|
        ≤ (fun k' => if hk' : k' < M + 2 then ts2.sup_bounds ⟨k', hk'⟩ else 0) k := by
      intro k hk
      have hk2 : k < M + 2 := lt_of_le_of_lt hk m.isLt
      simp only [dif_pos hk2]
      exact h2.sup_acc ⟨k, hk2⟩ x
    exact abs_iteratedDeriv_mul_le hf_at hg_at hF hG

/-- **Derived theorem `cert_arb_ts_deriv`** (formerly an axiom;
de-axiomatized 2026-07-03, target P3 of `docs/audit_2026-07-03.md`) —
Julia's `deriv_series` correctness.

Given an `IsTaylorSeriesOf` for `f`, there is a
`TaylorSeriesData (M - 1) N` representing `f'`: the index shift
`ts.derivShift`, whose correctness `WaveCAP.deriv_series_correct` proves
outright with no caller hypotheses. This statement therefore never
carried any Arb trust — it is pure re-indexing.

Cross-reference: `BabenkoRigorous/src/TaylorSeries.jl:313`
(`deriv_series`). -/
theorem cert_arb_ts_deriv
    {M N : ℕ} {ts : TaylorSeriesData M N} {f : ℝ → ℝ} {a : ℝ}
    (h : IsTaylorSeriesOf ts f a) (hM : 0 < M) :
    ∃ ts' : TaylorSeriesData (M - 1) N,
      IsTaylorSeriesOf ts' (deriv f) a ∧ ts'.h = ts.h := by
  refine ⟨ts.derivShift hM, ?_, rfl⟩
  have h' := deriv_series_correct h hM
  rwa [iteratedDeriv_one] at h'

/-- The classical uniform bound on the derivatives of `z ↦ z⁻¹` on
`[d_lower, ∞)`: since `((·)⁻¹)^{(k)}(y) = (-1)^k k! / y^{k+1}` and
`1/y^{k+1}` is decreasing on `(0, ∞)`,
`|((·)⁻¹)^{(k)}(y)| ≤ k! / d_lower^{k+1}` for `y ≥ d_lower > 0`.
This is the analytic content of Julia's `_inv_sup_bounds_on_interval`. -/
theorem abs_iteratedDeriv_inv_le {d_lower : ℝ} (h_d_pos : 0 < d_lower)
    (k : ℕ) (y : ℝ) (hy : d_lower ≤ y) :
    |iteratedDeriv k (fun z : ℝ => z⁻¹) y|
      ≤ (k.factorial : ℝ) / d_lower ^ (k + 1) := by
  have hy0 : 0 < y := lt_of_lt_of_le h_d_pos hy
  have hiter : iteratedDeriv k (fun z : ℝ => z⁻¹) y
      = (-1) ^ k * (k.factorial : ℝ) * y ^ (-1 - k : ℤ) := by
    rw [iteratedDeriv_eq_iterate]
    exact iter_deriv_inv k y
  have hzpow : y ^ (-1 - k : ℤ) = (y ^ (k + 1))⁻¹ := by
    rw [show (-1 - (k : ℤ)) = -((k : ℤ) + 1) by ring]
    rw [show ((k : ℤ) + 1) = ((k + 1 : ℕ) : ℤ) by push_cast; ring]
    rw [zpow_neg, zpow_natCast]
  rw [hiter, hzpow, abs_mul, abs_mul, abs_pow, abs_neg, abs_one, one_pow,
    one_mul, Nat.abs_cast, abs_inv, abs_of_pos (pow_pos hy0 _),
    div_eq_mul_inv]
  gcongr

/-- **Derived theorem `cert_arb_inv_series`** (formerly an axiom;
de-axiomatized 2026-07-03, target P5 of `docs/audit_2026-07-03.md`) —
Julia's `inv_series` correctness.

Given an `IsTaylorSeriesOf` for `f` with `f ≥ d_lower > 0` pointwise, a
`TaylorSeriesData` representing `1/f` exists: `inv_series_data` with the
Faà-di-Bruno sup-bound vector `Φ k = k! / d_lower^(k+1)`, whose
correctness `WaveCAP.inv_series_correct` (Area B) proves outright given
the `Φ` bound (`abs_iteratedDeriv_inv_le` above). The statement is a bare
existential that never referenced Julia's computed arrays, so it carried
no Arb trust.

Cross-reference: `BabenkoRigorous/src/TaylorSeries.jl:517-553`
(`inv_series`), `:488-499` (`_inv_sup_bounds_on_interval`). -/
theorem cert_arb_inv_series
    {M N : ℕ} {ts : TaylorSeriesData M N} {f : ℝ → ℝ} {a : ℝ}
    (h : IsTaylorSeriesOf ts f a)
    (d_lower : ℝ) (h_d_pos : 0 < d_lower) (h_f_pos : ∀ x, d_lower ≤ f x) :
    ∃ ts' : TaylorSeriesData M N,
      IsTaylorSeriesOf ts' (fun x => (f x)⁻¹) a ∧ ts'.h = ts.h :=
  ⟨inv_series_data ts f a (fun k => (k.factorial : ℝ) / d_lower ^ (k + 1)),
    inv_series_correct h d_lower h_d_pos h_f_pos _
      (fun k y hy => abs_iteratedDeriv_inv_le h_d_pos k y hy),
    rfl⟩

/-- **Derived theorem `cert_arb_fft_hilbert_series`** (formerly an axiom;
de-axiomatized 2026-07-03, target P6 of `docs/audit_2026-07-03.md`) —
Julia's `fft_hilbert_series` correctness.

Given an `IsTaylorSeriesOf` for a `2π`-periodic `C^{M+3}` function `G`, a
`TaylorSeriesData` representing the periodic Hilbert transform `H[G]`
exists. This follows from the same exact-data engine as
`cert_arb_trigpoly_to_taylor`, because `H[G]` is itself smooth and
periodic: `hilbert_contDiff` (the `+1`-loss smoothness chain, needing
only `G ∈ C^{M+2}` of the available `C^{M+3}`) gives
`H[G] ∈ C^{M+1}`, and `periodicHilbert_is_periodic` gives `2π`-periodicity
unconditionally. The statement is a bare existential that never
referenced Julia's computed arrays, so it carried no Arb trust —
the quantitative aliasing+Fourier-tail analysis (N3+N4) lives in
`WaveCAP.fft_hilbert_series_correct` (Area F), which remains the
record-level lemma for Julia's actual `fft_hilbert_series` output.

Cross-reference: `BabenkoRigorous/src/FFTHilbert.jl`
(`fft_hilbert_series`). -/
theorem cert_arb_fft_hilbert_series
    {M N : ℕ} {ts_G : TaylorSeriesData M N} {G : ℝ → ℝ} {a : ℝ}
    (h_G : IsTaylorSeriesOf ts_G G a)
    (hG_smooth : ContDiff ℝ (M + 3) G)
    (h_per : Function.Periodic G (2 * Real.pi)) :
    ∃ ts_HG : TaylorSeriesData M N,
      IsTaylorSeriesOf ts_HG (periodicHilbert G) a ∧ ts_HG.h = ts_G.h := by
  have hG2 : ContDiff ℝ ((M + 1) + 1 : ℕ) G :=
    hG_smooth.of_le (Nat.mono_cast (show (M + 1) + 1 ≤ M + 3 by omega))
  exact exists_isTaylorSeriesOf_of_contDiff_periodic (periodicHilbert G) a
    ts_G.h h_G.h_pos (hilbert_contDiff hG2 h_per)
    (periodicHilbert_is_periodic G)

/-- **Axiom `cert_arb_dmin_bound`** — Julia's `dmin_bound` correctness on
the certificate inputs.

For the certificate inputs `a_cert`, Julia's `dmin_bound` produces a
strictly positive real `D_lower` with `D_lower ≤ D a_cert u` for every
`u`. The certificate ships this value as `DLower_cert`.

The Lean justification is `WaveCAP.dmin_bound_correct` (Area E), which
builds the lower bound from the grid-pointwise minimum plus a Lipschitz
correction.

This is the atomic decomposition of the old `cert_D_ge` axiom.

Cross-reference: `BabenkoRigorous/src/TaylorSeries.jl:655-749`
(`dmin_bound`). -/
axiom cert_arb_dmin_bound :
    DLower_cert > 0 ∧ ∀ u, DLower_cert ≤ D a_cert u

-- ===========================================================================
-- §2. Pipeline-assembly axioms (specialised to certificate inputs)
-- ===========================================================================

/-- **Derived theorem `cert_arb_residual_TS`** (formerly an axiom; Phase-6
sharpening 2026-05-24) — the full Julia `_residual_bound_full` pipeline,
applied to the certificate inputs, produces a `TaylorSeriesData` for the
residual `R a_cert μ_cert β_cert` on a `2π/N_fft`-spaced grid starting
at `0`.

The proof composes the per-primitive `cert_arb_*` axioms step-by-step
along Path-A + Path-B of the Julia routine:

  Step 1: `cert_arb_trigpoly_to_taylor` applied to `y, yu, Hyu`.
  Step 2: `cert_arb_ts_mul` for `y · Hyu` and `y · yu`.
  Step 3: `cert_arb_fft_hilbert_series` for `H[y · yu]`.
  Step 4: Algebraic Area-A `add/sub/scalar_mul_taylorseries_correct`
          to assemble the β = 0 residual.
  Step 5 (β ≠ 0): `cert_arb_dmin_bound` for `D_lower`,
          `cert_arb_trigpoly_to_taylor` for `D` and `1 + Hyu`,
          `cert_arb_inv_series` for `1/D`,
          `cert_arb_ts_mul` for `(1 + Hyu) · D⁻¹ = G` and `yu · D⁻¹ = yu/D`,
          `cert_arb_ts_deriv` for `∂_u G` and `∂_u(yu/D)`,
          `cert_arb_fft_hilbert_series` for `H[∂_u G]`,
          structural Area-A sums for the assembled residual.

The smoothness side-conditions on each axiom invocation are discharged
by `WaveCAP.R_contDiff` and its helper lemmas in `WaveCAP/RSmoothness.lean`.

Order choice: non-deriv chain at `M = 6`; deriv chain at `M_inner = 7`,
dropping to `6` after a single `cert_arb_ts_deriv` step.  Both chains
land at `M = 6` for the final Area-A combinator sum.

Cross-reference: `BabenkoRigorous/src/ResidualBound.jl:65-239`
(`_residual_bound_full`). -/
theorem cert_arb_residual_TS :
    ∃ (M N_fft : ℕ) (_hN : 0 < N_fft) (ts_R : TaylorSeriesData M N_fft),
      IsTaylorSeriesOf ts_R (R a_cert μ_cert β_cert) 0
        ∧ ts_R.h = 2 * Real.pi / N_fft := by
  -- Order choice: M = 6, M_inner = 7 (one extra for the deriv chain).
  set M : ℕ := 6 with hM_def
  set M_inner : ℕ := 7 with hM_inner_def
  -- Grid = the certificate's declared `N_fft_cert`, so the `ts_R` this
  -- construction produces lives on the same grid where the Julia/Arb
  -- `sup_bound` (⇒ `ε₀_cert`) was certified. `cert_arb_sup_bound_le_eps0`
  -- consumes exactly this witness, so matching the grid is what makes that
  -- axiom a true statement about the certified quantity.
  set N_fft : ℕ := N_fft_cert with hN_fft_def
  have hN_fft : 0 < N_fft := by rw [hN_fft_def]; unfold N_fft_cert; norm_num
  -- Positivity of D from the dmin_bound axiom.
  obtain ⟨hDLower_pos, hD_lower⟩ := cert_arb_dmin_bound
  -- ===========================================================================
  -- Non-deriv chain at order M = 6
  -- ===========================================================================
  obtain ⟨ts_y, h_y, hh_y⟩ :=
    cert_arb_trigpoly_to_taylor (y a_cert) M N_fft hN_fft 0
      (y_contDiff_all a_cert (M + 1)) (y_periodic a_cert)
  obtain ⟨ts_yu, h_yu, hh_yu⟩ :=
    cert_arb_trigpoly_to_taylor (yu a_cert) M N_fft hN_fft 0
      (yu_contDiff_all a_cert (M + 1)) (yu_periodic_re a_cert)
  obtain ⟨ts_Hyu, h_Hyu, hh_Hyu⟩ :=
    cert_arb_trigpoly_to_taylor (Hyu a_cert) M N_fft hN_fft 0
      (Hyu_contDiff_all a_cert (M + 1)) (Hyu_periodic_re a_cert)
  -- TS for y · Hyu, y · yu via ts_mul.
  obtain ⟨ts_yHyu, h_yHyu, hh_yHyu⟩ :=
    cert_arb_ts_mul h_y h_Hyu (hh_y.trans hh_Hyu.symm)
  obtain ⟨ts_yyu, h_yyu, hh_yyu⟩ :=
    cert_arb_ts_mul h_y h_yu (hh_y.trans hh_yu.symm)
  -- TS for H[y · yu] via fft_hilbert_series.
  obtain ⟨ts_Hyyu, h_Hyyu, hh_Hyyu⟩ :=
    cert_arb_fft_hilbert_series h_yyu
      ((y_contDiff_all a_cert (M + 3)).mul (yu_contDiff_all a_cert (M + 3)))
      ((y_periodic a_cert).mul (yu_periodic_re a_cert))
  -- TS for μ · Hyu via scalar_mul_taylorseries_correct (Area A).
  have h_muHyu := scalar_mul_taylorseries_correct (c := μ_cert) h_Hyu
  -- ===========================================================================
  -- Deriv chain at order M_inner = 7 → M = 6 after deriv
  -- ===========================================================================
  obtain ⟨ts_yu7, h_yu7, hh_yu7⟩ :=
    cert_arb_trigpoly_to_taylor (yu a_cert) M_inner N_fft hN_fft 0
      (yu_contDiff_all a_cert (M_inner + 1)) (yu_periodic_re a_cert)
  obtain ⟨ts_1plusHyu7, h_1plusHyu7, hh_1plusHyu7⟩ :=
    cert_arb_trigpoly_to_taylor (fun u => 1 + Hyu a_cert u) M_inner N_fft hN_fft 0
      (contDiff_const.add (Hyu_contDiff_all a_cert (M_inner + 1)))
      (one_add_Hyu_periodic a_cert)
  obtain ⟨ts_D7, h_D7, hh_D7⟩ :=
    cert_arb_trigpoly_to_taylor (D a_cert) M_inner N_fft hN_fft 0
      (D_contDiff_all a_cert hDLower_pos hD_lower (M_inner + 1))
      (D_periodic_re a_cert)
  -- TS for 1/D via inv_series.
  obtain ⟨ts_invD7, h_invD7, hh_invD7⟩ :=
    cert_arb_inv_series h_D7 DLower_cert hDLower_pos hD_lower
  -- TS for yu * (1/D) [= yu/D] via ts_mul.  Need ts_yu7.h = ts_invD7.h.
  -- Chain: ts_yu7.h = 2π/N = ts_D7.h = ts_invD7.h (since hh_invD7 : ts_invD7.h = ts_D7.h).
  have hh_yu7_invD7 : ts_yu7.h = ts_invD7.h := by
    rw [hh_yu7, ← hh_D7, ← hh_invD7]
  obtain ⟨ts_yuOverD7, h_yuOverD7, hh_yuOverD7⟩ :=
    cert_arb_ts_mul h_yu7 h_invD7 hh_yu7_invD7
  -- TS for (1+Hyu) * (1/D) [= G] via ts_mul.
  have hh_1plusHyu7_invD7 : ts_1plusHyu7.h = ts_invD7.h := by
    rw [hh_1plusHyu7, ← hh_D7, ← hh_invD7]
  obtain ⟨ts_G7, h_G7, hh_G7⟩ :=
    cert_arb_ts_mul h_1plusHyu7 h_invD7 hh_1plusHyu7_invD7
  -- TS for deriv (yu/D) and deriv G at order M_inner - 1 via ts_deriv.
  have hM_inner_pos : 0 < M_inner := by decide
  obtain ⟨ts_dyuOverD, h_dyuOverD, hh_dyuOverD⟩ :=
    cert_arb_ts_deriv h_yuOverD7 hM_inner_pos
  obtain ⟨ts_dG, h_dG, hh_dG⟩ :=
    cert_arb_ts_deriv h_G7 hM_inner_pos
  -- M_inner - 1 = M (decidable, but Lean has to be coaxed).
  -- Smoothness/periodicity of deriv of (fun u => (1+Hyu)·D⁻¹).
  have hG_eq : (G a_cert) = fun u => (1 + Hyu a_cert u) * (D a_cert u)⁻¹ := by
    funext u
    change (1 + Hyu a_cert u) / D a_cert u = (1 + Hyu a_cert u) * (D a_cert u)⁻¹
    rw [div_eq_mul_inv]
  have hG_inner_contDiff : ∀ k : ℕ,
      ContDiff ℝ k (fun u => (1 + Hyu a_cert u) * (D a_cert u)⁻¹) := by
    intro k
    have hsmooth : ContDiff ℝ k (G a_cert) := G_contDiff_all a_cert hDLower_pos hD_lower k
    rw [hG_eq] at hsmooth
    exact hsmooth
  have hG_inner_periodic :
      Function.Periodic
        (deriv (fun u => (1 + Hyu a_cert u) * (D a_cert u)⁻¹)) (2 * Real.pi) := by
    have h := G_deriv_periodic a_cert
    rw [hG_eq] at h
    exact h
  -- For ts_deriv, output is TaylorSeriesData (M_inner - 1) N_fft.  Since
  -- M_inner - 1 reduces to 6 = M, the types unify and we can pass directly to
  -- the next axiom calls.  We treat the ts_d* records as M-order data via
  -- the definitional fact M_inner - 1 = M = 6.
  -- Compute H[deriv G] at order M = 6.
  obtain ⟨ts_HdG, h_HdG, hh_HdG⟩ :=
    cert_arb_fft_hilbert_series (M := M_inner - 1) h_dG
      (by
        -- ContDiff ((M_inner - 1) + 3) (deriv (fun u => (1+Hyu)·D⁻¹))
        have : ContDiff ℝ (((M_inner - 1) + 3 : ℕ) : WithTop ℕ∞)
            (deriv (fun u => (1 + Hyu a_cert u) * (D a_cert u)⁻¹)) := by
          exact ContDiff.deriv' (n := (((M_inner - 1) + 3 : ℕ) : WithTop ℕ∞))
            (hG_inner_contDiff ((M_inner - 1) + 4))
        exact this)
      hG_inner_periodic
  -- TS for β · deriv(yu/D) and β · H[deriv G] at order M_inner - 1.
  have h_β_dyuOverD := scalar_mul_taylorseries_correct (c := β_cert) h_dyuOverD
  have h_β_HdG := scalar_mul_taylorseries_correct (c := β_cert) h_HdG
  -- ===========================================================================
  -- All TS records carry h = 2 * Real.pi / N_fft (= ts_y.h = ts_Hyu.h = ...).
  -- ===========================================================================
  -- The non-deriv records carry M-order data; the deriv records carry
  -- (M_inner - 1)-order data.  Since `M_inner - 1` is *definitionally* `6` (= `M`),
  -- Lean treats them as `TaylorSeriesData 6 N_fft` either way.
  --
  -- Final assembly: R = ((((μ·Hyu - y) - y·Hyu) - H[y·yu]) + β·deriv(yu/D)) - β·H[deriv G]
  --
  -- Use Area A combinators (add/sub/scalar_mul_taylorseries_correct).
  -- Step 1: muHyu - y.  After scalar_mul, h_muHyu.h = ts_Hyu.h = 2π/N_fft.
  have h_step1 := sub_taylorseries_correct h_muHyu h_y (by
    -- LHS.h = ts_Hyu.h; RHS.h = ts_y.h.  Both = 2π/N_fft.
    simp only []; rw [hh_Hyu, hh_y])
  -- Step 2: step1 - y·Hyu.  h_step1.h = ts_Hyu.h.
  have h_step2 := sub_taylorseries_correct h_step1 h_yHyu (by
    simp only []; rw [hh_Hyu, hh_yHyu, hh_y])
  -- Step 3: step2 - H[y·yu].  h_step2.h = ts_Hyu.h.
  have h_step3 := sub_taylorseries_correct h_step2 h_Hyyu (by
    simp only []; rw [hh_Hyu, hh_Hyyu, hh_yyu, hh_y])
  -- Step 4: step3 + β·deriv(yu/D).
  have hh_dyuOverD_value : ts_dyuOverD.h = 2 * Real.pi / N_fft := by
    rw [hh_dyuOverD, hh_yuOverD7]; exact hh_yu7
  have hh_dG_value : ts_dG.h = 2 * Real.pi / N_fft := by
    rw [hh_dG, hh_G7]; exact hh_1plusHyu7
  have hh_HdG_value : ts_HdG.h = 2 * Real.pi / N_fft := by
    rw [hh_HdG]; exact hh_dG_value
  have h_step4 := add_taylorseries_correct h_step3 h_β_dyuOverD (by
    show ts_Hyu.h = ts_dyuOverD.h
    rw [hh_Hyu, hh_dyuOverD_value])
  -- Step 5: step4 - β·H[deriv G].
  have h_step5 := sub_taylorseries_correct h_step4 h_β_HdG (by
    show ts_Hyu.h = ts_HdG.h
    rw [hh_Hyu, hh_HdG_value])
  -- h_step5 represents the function
  -- ((((μ·Hyu - y) - y·Hyu) - H[y·yu]) + β·deriv(yu/D)) - β·H[deriv G]
  -- which equals R a_cert μ_cert β_cert by extensionality (modulo div = mul⁻¹).
  -- We rewrite the function in h_step5 to the R-form via funext-based equalities.
  have h_yu_div_eq :
      (fun u => yu a_cert u * (D a_cert u)⁻¹)
        = (fun v => yu a_cert v / D a_cert v) := by
    funext v; rw [div_eq_mul_inv]
  have h_G_div_eq :
      (fun u => (1 + Hyu a_cert u) * (D a_cert u)⁻¹) = G a_cert := by
    funext v
    show (1 + Hyu a_cert v) * (D a_cert v)⁻¹ = (1 + Hyu a_cert v) / D a_cert v
    rw [div_eq_mul_inv]
  have h_func_eq :
      ((((((fun u => μ_cert * Hyu a_cert u) - y a_cert)
              - (fun u => y a_cert u * Hyu a_cert u))
              - periodicHilbert (fun v => y a_cert v * yu a_cert v))
              + (fun u => β_cert * deriv (fun u => yu a_cert u * (D a_cert u)⁻¹) u))
              - (fun u => β_cert * periodicHilbert
                  (deriv (fun u => (1 + Hyu a_cert u) * (D a_cert u)⁻¹)) u))
        = R a_cert μ_cert β_cert := by
    rw [h_yu_div_eq, h_G_div_eq]
    funext u
    simp only [Pi.add_apply, Pi.sub_apply]
    rfl
  -- Convert h_step5's function from the pointwise form to R.
  rw [← h_func_eq]
  -- Now the goal exists ⟨M, N_fft, hN_fft, ts_R, _⟩ for the pointwise R form,
  -- which h_step5 is the witness for.
  exact ⟨M, N_fft, hN_fft, _, h_step5, by
    -- ts.h = 2 * Real.pi / N_fft.  ts.h is ts_Hyu.h via the sub chain.
    show ts_Hyu.h = 2 * Real.pi / N_fft
    exact hh_Hyu⟩

/-- **Pipeline-assembly axiom `cert_arb_sup_bound_le_eps0`** — Julia's
`sup_bound` applied to the residual `TaylorSeriesData` (from
`cert_arb_residual_TS`) produces a real number `≤ ε₀_cert`.

The Lean justification is `WaveCAP.sup_bound_taylorseries_correct` (Area
A) with the per-grid-cell bound `M_bound = ε₀_cert`; the Arb-side trust
is the per-grid-cell Taylor-sum evaluation correctness. This is the
*quantitative* trust step: the shipped `ε₀_cert` matches Julia's actual
output of `sup_bound(ts_R)`.

Cross-reference: `BabenkoRigorous/src/TaylorSeries.jl:133` (`sup_bound`),
`BabenkoRigorous/src/Certificate.jl` (`emit_certificate`). -/
axiom cert_arb_sup_bound_le_eps0
    {M N_fft : ℕ} {ts_R : TaylorSeriesData M N_fft}
    (_hN : 0 < N_fft)
    (_h : IsTaylorSeriesOf ts_R (R a_cert μ_cert β_cert) 0)
    (_hh : ts_R.h = 2 * Real.pi / N_fft) :
    ∀ u ∈ Set.Icc (0 : ℝ) (0 + N_fft * ts_R.h),
      |R a_cert μ_cert β_cert u| ≤ ε₀_cert

end WaveCAP

-- ===========================================================================
-- §3. Derived theorems — replace the monolithic Certificate axioms
-- ===========================================================================

namespace WaveCAP.Certificate

/-- **Phase-6 derived theorem `cert_residual_le`** (formerly axiom).

This is the Phase-6 derivation of `‖R‖∞ ≤ ε₀_cert` from the atomic
`cert_arb_*` axioms via `WaveCAP.cert_residual_le_derived` (Area G).

It carries the **same name and statement** as the original monolithic
trust-anchor axiom (now retired). Its `#print axioms` reveals the new
atomic decomposition:

```
[propext, Classical.choice, Quot.sound,
 cert_arb_trigpoly_to_taylor, cert_arb_ts_mul, cert_arb_ts_deriv,
 cert_arb_inv_series, cert_arb_fft_hilbert_series,
 cert_arb_dmin_bound, cert_arb_sup_bound_le_eps0]
```

i.e. **all five per-primitive Julia/Arb axioms** (one for each Arb
routine that must be reviewed in isolation) plus the two pipeline-level
axioms (`_sup_bound_le_eps0` for the quantitative trust step, and
`_dmin_bound` for `D ≥ DLower_cert`).

The intermediate `cert_arb_residual_TS` is **no longer an axiom**: it
is now a derived theorem (see `WaveCAP.cert_arb_residual_TS`) built
from the per-primitive axioms plus `R_contDiff` (in
`WaveCAP/RSmoothness.lean`). -/
theorem cert_residual_le :
    ∀ u, |WaveCAP.R a_cert μ_cert β_cert u| ≤ ε₀_cert := by
  apply WaveCAP.cert_residual_le_derived
  obtain ⟨M, N_fft, hN_fft, ts_R, h_ts, h_h⟩ :=
    WaveCAP.cert_arb_residual_TS
  refine ⟨M, N_fft, hN_fft, ts_R, h_ts, h_h, ?_⟩
  exact WaveCAP.cert_arb_sup_bound_le_eps0 hN_fft h_ts h_h

/-- **Phase-6 derived theorem `cert_D_ge`** (formerly axiom).

This is the Phase-6 derivation of `DLower_cert ≤ D a_cert u` from the
atomic `cert_arb_dmin_bound` axiom. -/
theorem cert_D_ge :
    ∀ u, DLower_cert ≤ WaveCAP.D a_cert u :=
  WaveCAP.cert_arb_dmin_bound.2

end WaveCAP.Certificate

-- ===========================================================================
-- §4. Sanity checks: axiom dependency listings
-- ===========================================================================

#check @WaveCAP.cert_arb_trigpoly_to_taylor
#check @WaveCAP.cert_arb_ts_mul
#check @WaveCAP.cert_arb_ts_deriv
#check @WaveCAP.cert_arb_inv_series
#check @WaveCAP.cert_arb_fft_hilbert_series
#check @WaveCAP.cert_arb_dmin_bound
#check @WaveCAP.cert_arb_residual_TS
#check @WaveCAP.cert_arb_sup_bound_le_eps0

#print axioms WaveCAP.Certificate.cert_residual_le
#print axioms WaveCAP.Certificate.cert_D_ge
#print axioms WaveCAP.cert_arb_residual_TS

-- The above two prints should each list, on top of the standard
-- [propext, Classical.choice, Quot.sound]:
--   - cert_residual_le : all five per-primitive cert_arb_* axioms
--                         (trigpoly_to_taylor, ts_mul, ts_deriv,
--                          inv_series, fft_hilbert_series),
--                         plus cert_arb_dmin_bound, cert_arb_sup_bound_le_eps0
--   - cert_D_ge        : cert_arb_dmin_bound
-- Neither lists the old monolithic axioms — they have been retired.
-- Neither lists cert_arb_residual_TS — it is now a derived theorem.
--
-- The per-primitive axioms (cert_arb_trigpoly_to_taylor, cert_arb_ts_mul,
-- cert_arb_ts_deriv, cert_arb_inv_series, cert_arb_fft_hilbert_series) are
-- declared for documentation of the granular Arb trust trade; they would
-- replace the composite cert_arb_residual_TS in a follow-up phase that
-- adds the explicit chain assembly (currently held back only by the
-- smoothness obligation on R).
