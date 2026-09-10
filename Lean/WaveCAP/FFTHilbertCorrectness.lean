import WaveCAP.TaylorSeriesCorrectness
import WaveCAP.AliasingErrorBound
import WaveCAP.FourierTailHilbert
import WaveCAP.HilbertCommutesDeriv
import WaveCAP.HilbertOnTrigPoly
import WaveCAP.HilbertSmoothness

/-!
# Area F — FFT-based discrete Hilbert routines (correctness)

This file justifies the Julia routines

  `fft_hilbert(G_vals, N_fft, p, Gp_sup; prec)`
  `fft_hilbert_series(ts_G, N_fft; Gpp_bound, Gppp_bound, prec)`

found in `BabenkoRigorous/src/FFTHilbert.jl`. They compute the discrete
Hilbert transform of a function sampled on a uniform grid on the torus
`𝕋 = ℝ/2πℤ`, together with an aliasing/tail error bound on the difference
from the true continuous `periodicHilbert G`.

## Outline

### F1 — `fft_hilbert_correct`

The Julia routine returns a vector `HG_vals : Vector{Arb}` of N_fft balls
together with an explicit aliasing radius `ε_alias`. Mathematically, the
contract is that each ball encloses `periodicHilbert G (2π·j/N_fft)`. In
the Lean abstraction:

* The `derivs` slot of a `TaylorSeriesData` at derivative level `m` is the
  Arb *midpoint* of `H[G^{(m)}](u_j)` (a real number, not a ball).
* The `derivs_err` slot is the Arb *radius* (a non-negative real number)
  that upper-bounds the distance from the midpoint to the true value.

Step F1 splits the radius into two parts: a "rounding" part coming from
input ball radii (propagated through Arb's `mul!`), and a "model" part
`aliasingBound N p Gp_sup` coming from the truncation/aliasing residual.
The rounding part is opaque to the formal proof (it is the trust boundary
of Arb's certified arithmetic); the model part is `4 Cp / ((p − 1)(N/2 − 1)^{p−1})`
and is bounded by the N4 headline theorem `dft_hilbert_aliasing_bound`,
which controls the sampling-DFT operator `dftHilbert` at the grid points
`u_j = 2πj/N` directly (via the periodic Poisson identity).

### F2 — `fft_hilbert_series_correct`

Wraps F1 over derivative levels `m = 0, ..., M`, plus output `sup_bounds`
via the N3 theorem `fourier_tail_hilbert_bound` (`‖H[g]‖_∞ ≤ ‖g''‖_∞ π²/3`).
At level `m` the output's `sup_bounds m` equals `(input sup at level m+2) · π²/3`,
which matches the Julia formula

```
new_sup_bounds[m+1] = ts_G.sup_bounds[m+3] * π²/3   for m = 0..M-1
new_sup_bounds[M+1] = Gpp_bound * π²/3              for m = M
new_sup_bounds[M+2] = Gppp_bound * π²/3             for m = M+1
```

Per-level correctness uses
`iteratedDeriv_periodicHilbert : iteratedDeriv n (H g) = H (iteratedDeriv n g)`
plus F1 to bound `|H[G^{(m)}](u_j) - dftHilbert N (G^{(m)}) u_j|`. We then
demand the Julia caller supply a per-level radius (the `derivs_err` slot)
that absorbs both the per-input rounding *and* the aliasing radius;
correctness reduces to a per-level triangle inequality plus a single
invocation of `dft_hilbert_aliasing_bound`.

## What is proved

* `aliasingBound` — closed-form formula `4 Cp / ((p−1)(N/2−1)^{p−1})` matching
  Julia's `ε_alias`.
* `fft_hilbert_correct` — abstract correctness predicate: any caller-supplied
  discrete output `out` whose deviation from `dftHilbert N G` is bounded by
  `inputErr` satisfies `|out j - periodicHilbert G (2π·j/N)| ≤
  inputErr + aliasingBound N p Gp_sup`.
* `TaylorSeriesData.fft_hilbert` — top-level Lean constructor matching the
  Julia output shape.
* `fft_hilbert_series_correct` — wraps F1 + N3 across derivative levels.

The file contains **no `sorry`**. The single genuine boundary is:
  * The per-grid-point `newDerivs`/`newDerivsErr` data is an *input* to
    `fft_hilbert_series_correct` (the caller supplies the Arb-certified
    values + radii produced by Julia's `Arblib.mul!` FFT pipeline). The
    bound `|iteratedDeriv m (H G) (a + j h) − newDerivs j m| ≤ newDerivsErr j m`
    is the F1 contract per derivative level, embedded as the
    `h_derivs_acc` hypothesis.

The `ContDiff (M+1) (periodicHilbert G)` smoothness fact is now discharged
internally by `hilbert_contDiff` (proved axiom-clean in
`HilbertSmoothness.lean`) from `ContDiff (M+2) G` (which follows from the
existing `ContDiff (M+4) G` hypothesis) and periodicity.
-/

namespace WaveCAP

open Real Set

-- ===========================================================================
-- F-helper. `iteratedDeriv` composes additively.
-- ===========================================================================

/-- `iteratedDeriv` composes additively: `D^a (D^b f) = D^(a+b) f`. -/
lemma iteratedDeriv_iteratedDeriv (a b : ℕ) (f : ℝ → ℝ) :
    iteratedDeriv a (iteratedDeriv b f) = iteratedDeriv (a + b) f := by
  rw [iteratedDeriv_eq_iterate, iteratedDeriv_eq_iterate, iteratedDeriv_eq_iterate]
  exact (Function.iterate_add_apply deriv a b f).symm

-- ===========================================================================
-- F0. Aliasing bound formula (matches Julia ε_alias)
-- ===========================================================================

/-- The closed-form aliasing bound used by Julia `fft_hilbert`:
`ε_alias = 4 · Gp_sup / ((p − 1) · (N/2 − 1)^(p − 1))`.

Matches the formula on line 183 of `BabenkoRigorous/src/FFTHilbert.jl`. -/
noncomputable def aliasingBound (N p : ℕ) (Gp_sup : ℝ) : ℝ :=
  4 * Gp_sup / ((p - 1 : ℝ) * ((N / 2 - 1 : ℝ) ^ (p - 1)))

/-- `aliasingBound` is nonnegative when `Gp_sup ≥ 0`, `p ≥ 2`, `N ≥ 4`. -/
lemma aliasingBound_nonneg {N p : ℕ} {Gp_sup : ℝ}
    (hp : 2 ≤ p) (hN : 4 ≤ N) (hGp_sup : 0 ≤ Gp_sup) :
    0 ≤ aliasingBound N p Gp_sup := by
  unfold aliasingBound
  have hpm1_pos : (0 : ℝ) < (p - 1 : ℝ) := by
    have : (2 : ℝ) ≤ (p : ℝ) := by exact_mod_cast hp
    linarith
  have hN_half : 1 ≤ N / 2 - 1 := by omega
  have hN_half_real : (1 : ℝ) ≤ ((N / 2 - 1 : ℕ) : ℝ) := by exact_mod_cast hN_half
  -- ((N/2 - 1 : ℕ) : ℝ) = ((N : ℝ)/2 - 1) under N even, but we don't need that here.
  -- The expression `((N / 2 - 1 : ℝ))` here is implicit-coerced from the ℕ
  -- arithmetic.  We bound the denominator positive via the cast inequality.
  have hN_real : (0 : ℝ) < ((N / 2 - 1 : ℝ)) := by
    -- (N / 2 - 1 : ℝ) involves ℕ.sub which casts via Nat.cast; this expression
    -- without explicit ℕ qualification is parsed as `((N : ℝ) / 2) - 1`. With
    -- N ≥ 4, that quantity is ≥ 1 > 0.
    have hN_real_ge : (4 : ℝ) ≤ (N : ℝ) := by exact_mod_cast hN
    linarith
  have hpow_pos : (0 : ℝ) < ((N / 2 - 1 : ℝ)) ^ (p - 1) := pow_pos hN_real _
  have hden_pos : (0 : ℝ) < (p - 1 : ℝ) * ((N / 2 - 1 : ℝ) ^ (p - 1)) :=
    mul_pos hpm1_pos hpow_pos
  have hnum_nn : (0 : ℝ) ≤ 4 * Gp_sup := by linarith
  exact div_nonneg hnum_nn hden_pos.le

-- ===========================================================================
-- F1. fft_hilbert_correct
-- ===========================================================================

/-- **F1 — Correctness of `fft_hilbert`.**

The Julia routine `fft_hilbert(G_vals, N_fft, p, Gp_sup)` returns a vector
of Arb balls `HG_vals[1..N_fft]` and a scalar `ε_alias`. Mathematically each
ball encloses the *continuous* periodic Hilbert transform of `G` evaluated
at the corresponding grid point: for every `j`,

```
|HG_vals[j] − periodicHilbert G (u_j)| ≤ inputErr + aliasingBound N p Gp_sup
```

where `inputErr` absorbs (a) the propagated input Arb radii in `G_vals` and
(b) the FFT rounding from `Arblib.mul!`. Both are tracked by Arb interval
arithmetic and are formally opaque to this Lean proof; the *analytic*
content is the aliasing residual `aliasingBound`, which is the closed-form
bound proved by `dft_hilbert_aliasing_bound` (N4).

Concretely: any caller-supplied function `out : Fin N → ℝ` whose pointwise
distance from `dftHilbert N G u_j` is bounded by `inputErr j` satisfies the
above bound, where the analytic content (the aliasing bound) comes from the
N4 theorem `dft_hilbert_aliasing_bound`.

**Proof.** Triangle inequality on
`|out j − H G u_j| ≤ |out j − dftHilbert N G u_j| + |dftHilbert N G u_j − H G u_j|`,
with the first term bounded by the hypothesis and the second by
`dft_hilbert_aliasing_bound`. -/
theorem fft_hilbert_correct
    {N : ℕ} (hN : 4 ≤ N) (hN_even : Even N)
    {G : ℝ → ℝ}
    (p : ℕ) (hp : 2 ≤ p)
    (hG : ContDiff ℝ p G) (hG_per : Function.Periodic G (2 * Real.pi))
    (Gp_sup : ℝ) (hGp : ∀ u, |iteratedDeriv p G u| ≤ Gp_sup)
    (out : Fin N → ℝ)
    (inputErr : Fin N → ℝ)
    (h_out : ∀ j : Fin N, |out j - dftHilbert N G (fftGridPoint N j)| ≤ inputErr j) :
    ∀ j : Fin N,
      |out j - periodicHilbert G (fftGridPoint N j)|
        ≤ inputErr j + aliasingBound N p Gp_sup := by
  intro j
  -- The N4 packaging theorem at grid points (Stage 5 headline:
  -- `dft_hilbert_aliasing_bound`): the sampling-DFT operator differs from
  -- `periodicHilbert G` at every grid point by at most the aliasing bound.
  have h_alias : |dftHilbert N G (fftGridPoint N j)
        - periodicHilbert G (fftGridPoint N j)| ≤ aliasingBound N p Gp_sup := by
    unfold aliasingBound
    exact dft_hilbert_aliasing_bound p hp N hN hN_even G hG hG_per Gp_sup hGp j
  -- Triangle inequality.
  have h_in := h_out j
  have h_tri :
      |out j - periodicHilbert G (fftGridPoint N j)|
        ≤ |out j - dftHilbert N G (fftGridPoint N j)|
          + |dftHilbert N G (fftGridPoint N j) - periodicHilbert G (fftGridPoint N j)| := by
    have h_eq :
        out j - periodicHilbert G (fftGridPoint N j)
          = (out j - dftHilbert N G (fftGridPoint N j))
            + (dftHilbert N G (fftGridPoint N j) - periodicHilbert G (fftGridPoint N j)) := by
      ring
    rw [h_eq]
    exact abs_add_le _ _
  linarith

-- ===========================================================================
-- F2. fft_hilbert_series_correct
-- ===========================================================================

/-- **The `TaylorSeriesData.fft_hilbert` constructor.**

Given the abstract per-grid-point per-level data (`derivs`, `derivs_err`)
that the Julia routine produces, and the analytic top-end bounds `Gpp_bound`
and `Gppp_bound` on `‖G^{(M+2)}‖_∞` and `‖G^{(M+3)}‖_∞`, package a new
`TaylorSeriesData` whose `sup_bounds` match the Julia formula

```
new_sup_bounds[m+1] = ts_G.sup_bounds[m+3] · π²/3   for m = 0..M-1
new_sup_bounds[M+1] = Gpp_bound · π²/3              for m = M
new_sup_bounds[M+2] = Gppp_bound · π²/3             for m = M+1
```

The N4 aliasing radius is the responsibility of the caller's `derivs_err`
slot (it embeds the per-level `aliasingBound N (p_m) (B_m)` plus the
propagated Arb rounding); the model-side bound on each `sup_bounds`
entry is exactly two derivatives of N3 applied to `G^{(m)}`. -/
noncomputable def TaylorSeriesData.fft_hilbert
    {M N : ℕ} (ts : TaylorSeriesData M N)
    (newDerivs : Fin N → Fin (M + 1) → ℝ)
    (newDerivsErr : Fin N → Fin (M + 1) → ℝ)
    (Gpp_bound Gppp_bound : ℝ) : TaylorSeriesData M N where
  derivs     := newDerivs
  derivs_err := newDerivsErr
  sup_bounds := fun m =>
    -- The Julia formula:
    --   m ∈ 0..M-1  →  ts.sup_bounds[m+3] · π²/3   (m+2 ≤ M+1, an in-bounds slot of ts.sup_bounds)
    --   m = M       →  Gpp_bound · π²/3
    --   m = M+1     →  Gppp_bound · π²/3
    -- We pivot the case-split on `m.val + 2 ≤ M + 1` so the M = 0 case
    -- (where the first branch is vacuous) cleanly skips it.
    if h : m.val + 2 ≤ M + 1 then
      ts.sup_bounds ⟨m.val + 2, by omega⟩ * Real.pi ^ 2 / 3
    else if hM : m.val = M then
      Gpp_bound * Real.pi ^ 2 / 3
    else
      Gppp_bound * Real.pi ^ 2 / 3
  h          := ts.h

/-- The Julia routine's chosen smoothness order at derivative level `m` of
`fft_hilbert_series`: `natural_p_m = M + 1 - m` if that is `≥ 2`,
otherwise `2` (the boundary case, using `Gpp_bound`).
Matches the Julia function-body branches at line ~256. -/
def fftHilbertSeriesSmoothness (M m : ℕ) : ℕ :=
  if M + 1 - m ≥ 2 then M + 1 - m else 2

/-- The Julia routine's chosen sup-bound at derivative level `m`:
`F_top = ts.sup_bounds[M+2]` if `m ≤ M-1`, otherwise `Gpp_bound`.
Matches the Julia function-body branches at line ~256. -/
noncomputable def fftHilbertSeriesBound {M N : ℕ}
    (ts : TaylorSeriesData M N) (Gpp_bound : ℝ) (m : ℕ) : ℝ :=
  if M + 1 - m ≥ 2 then ts.sup_bounds ⟨M + 1, by omega⟩ else Gpp_bound

/-- **F2 — Correctness of `fft_hilbert_series`.**

Given an `IsTaylorSeriesOf ts_G G a` with `G` 2π-periodic and `C^{M + 3}`
(so all derivative levels we use stay `C^{≥ 2}` for `dft_hilbert_aliasing_bound`,
and the super-tight `iteratedDeriv_periodicHilbert_super_tight` commute holds),
and given analytic bounds `Gpp_bound`, `Gppp_bound` on `‖G^{(M+2)}‖_∞` and
`‖G^{(M+3)}‖_∞`, the output of `TaylorSeriesData.fft_hilbert` is a valid
`IsTaylorSeriesOf … (periodicHilbert G) a`. The Hilbert-transform smoothness
hypothesis `ContDiff (M+1) (periodicHilbert G)` is discharged internally
via `hilbert_contDiff` (no longer a caller obligation). The caller still
supplies:

* `h_derivs_err` — a per-level bound on
  `|newDerivs j m − H[G^{(m)}](a + j·h)|`, which the Julia caller obtains
  as `(Arb input radius) + aliasingBound N p_m B_m` (with `p_m, B_m` per
  the Julia branches above);
* `Gpp_bound`/`Gppp_bound` upper-bound the corresponding iterated derivatives.

The `sup_bounds` slot is *automatically* discharged by `fourier_tail_hilbert_bound`
(N3) applied per derivative level after using
`iteratedDeriv_periodicHilbert : iteratedDeriv n (H g) = H (iteratedDeriv n g)`.

**Note on grid point alignment.** The Julia routine uses the grid
`u_j = 2π·j/N`, hence `a = 0` and `ts.h = 2π/N` in the Julia call. We
parametrise the theorem with `a` and `ts.h` so that callers can invoke it
in the WaveCAP residual chain where the same TaylorSeries machinery is used
with different `a` / `h`. -/
theorem fft_hilbert_series_correct
    {M N : ℕ} {ts_G : TaylorSeriesData M N} {G : ℝ → ℝ} {a : ℝ}
    (h_G : IsTaylorSeriesOf ts_G G a)
    (hG_smooth : ContDiff ℝ (M + 3) G)
    (h_per : Function.Periodic G (2 * Real.pi))
    (Gpp_bound Gppp_bound : ℝ)
    (h_Gpp : ∀ u, |iteratedDeriv (M + 2) G u| ≤ Gpp_bound)
    (h_Gppp : ∀ u, |iteratedDeriv (M + 3) G u| ≤ Gppp_bound)
    (newDerivs : Fin N → Fin (M + 1) → ℝ)
    (newDerivsErr : Fin N → Fin (M + 1) → ℝ)
    (h_derivs_acc : ∀ (j : Fin N) (m : Fin (M + 1)),
        |iteratedDeriv m.val (periodicHilbert G) (a + j.val * ts_G.h)
            - newDerivs j m| ≤ newDerivsErr j m) :
    IsTaylorSeriesOf
        (TaylorSeriesData.fft_hilbert ts_G newDerivs newDerivsErr Gpp_bound Gppp_bound)
        (periodicHilbert G) a := by
  -- Derive `ContDiff ℝ (M+1) (periodicHilbert G)` internally via the
  -- axiom-clean `hilbert_contDiff` (HilbertSmoothness.lean): it requires
  -- `ContDiff ℝ (M+2) G`, available as `hG_smooth.of_le` from `ContDiff ℝ (M+3) G`.
  have hG_for_H : ContDiff ℝ ((M + 1) + 1 : ℕ) G := by
    apply hG_smooth.of_le
    have hle : (M + 1) + 1 ≤ M + 3 := by omega
    have h1 : (((M + 1) + 1 : ℕ) : ℕ∞) ≤ ((M + 3 : ℕ) : ℕ∞) :=
      ENat.coe_le_coe.mpr hle
    have h2 : ((((M + 1) + 1 : ℕ) : ℕ∞) : WithTop ℕ∞)
          ≤ (((M + 3 : ℕ) : ℕ∞) : WithTop ℕ∞) :=
      WithTop.coe_le_coe.mpr h1
    simpa using h2
  have h_H_contdiff : ContDiff ℝ (M + 1 : ℕ) (periodicHilbert G) :=
    hilbert_contDiff hG_for_H h_per
  refine ⟨?_, ?_, ?_, ?_⟩
  · -- contdiff (M + 1) (periodicHilbert G) — derived above via `hilbert_contDiff`.
    -- The HilbertSmoothness module provides the axiom-clean derivation
    -- `ContDiff (k+1) G → Periodic G (2π) → ContDiff k (periodicHilbert G)`.
    exact h_H_contdiff
  · -- grid_acc.
    intro j m
    -- The hypothesis `h_derivs_acc` is exactly the conclusion modulo the
    -- definitional `derivs` / `derivs_err` projection through the constructor.
    change |iteratedDeriv m.val (periodicHilbert G) (a + j.val * ts_G.h)
            - newDerivs j m| ≤ newDerivsErr j m
    exact h_derivs_acc j m
  · -- sup_acc — the analytic π²/3 bound per level.
    intro m x
    -- iteratedDeriv m.val (periodicHilbert G) x = periodicHilbert (iteratedDeriv m.val G) x
    -- (from HilbertSharper; super-tight: requires ContDiff (m + 1) G).
    have hm_le_M2 : m.val ≤ M + 1 := Nat.lt_succ_iff.mp m.isLt
    have hG_iter_cd : ContDiff ℝ (m.val + 1) G := by
      apply hG_smooth.of_le
      have hle : m.val + 1 ≤ M + 3 := by omega
      have h1 : ((m.val + 1 : ℕ) : ℕ∞) ≤ ((M + 3 : ℕ) : ℕ∞) :=
        ENat.coe_le_coe.mpr hle
      have h2 : (((m.val + 1 : ℕ) : ℕ∞) : WithTop ℕ∞)
            ≤ (((M + 3 : ℕ) : ℕ∞) : WithTop ℕ∞) :=
        WithTop.coe_le_coe.mpr h1
      simpa using h2
    have h_commute :=
      iteratedDeriv_periodicHilbert_super_tight m.val G hG_iter_cd h_per
    rw [h_commute]
    -- iteratedDeriv m.val G is C² periodic with sup-bound determined by the
    -- (m + 2)-nd derivative bound on G.
    have hiter_C2 : ContDiff ℝ 2 (iteratedDeriv m.val G) := by
      -- iteratedDeriv m.val G = deriv^[m.val] G; C² of the iterate needs C^(m+2) of G.
      have h_eq : iteratedDeriv m.val G = (deriv^[m.val]) G := iteratedDeriv_eq_iterate
      rw [h_eq]
      apply ContDiff.iterate_deriv' 2 m.val (f := G)
      apply hG_smooth.of_le
      have hle : 2 + m.val ≤ M + 3 := by omega
      have h1 : ((2 + m.val : ℕ) : ℕ∞) ≤ ((M + 3 : ℕ) : ℕ∞) :=
        ENat.coe_le_coe.mpr hle
      have h2 : (((2 + m.val : ℕ) : ℕ∞) : WithTop ℕ∞)
            ≤ (((M + 3 : ℕ) : ℕ∞) : WithTop ℕ∞) :=
        WithTop.coe_le_coe.mpr h1
      simpa using h2
    have hiter_per : Function.Periodic (iteratedDeriv m.val G) (2 * Real.pi) :=
      Periodic.iteratedDeriv_periodic h_per m.val
    -- Choose the (m+2)-derivative bound depending on level.
    -- Goal RHS via the constructor projection:
    --   (TaylorSeriesData.fft_hilbert …).sup_bounds m
    --     = (per-level bound) · π² / 3.
    change |periodicHilbert (iteratedDeriv m.val G) x|
          ≤ (TaylorSeriesData.fft_hilbert ts_G newDerivs newDerivsErr
              Gpp_bound Gppp_bound).sup_bounds m
    -- Case-split on which π²/3 bound the sup_bounds projection selects.
    by_cases hlo : m.val + 2 ≤ M + 1
    · -- Lower-level case: sup_bounds m = ts_G.sup_bounds ⟨m.val + 2, _⟩ · π² / 3.
      -- ‖iteratedDeriv 2 (iteratedDeriv m.val G)‖_∞ = ‖iteratedDeriv (m.val + 2) G‖_∞
      --   ≤ ts_G.sup_bounds ⟨m.val + 2, _⟩.
      have h_iter_eq : iteratedDeriv 2 (iteratedDeriv m.val G)
          = iteratedDeriv (m.val + 2) G := by
        rw [iteratedDeriv_iteratedDeriv 2 m.val G]
        congr 1; omega
      have h_bound2 : ∀ u, |iteratedDeriv 2 (iteratedDeriv m.val G) u|
          ≤ ts_G.sup_bounds ⟨m.val + 2, by omega⟩ := by
        intro u
        rw [h_iter_eq]
        exact h_G.sup_acc ⟨m.val + 2, by omega⟩ u
      have h_N3 := fourier_tail_hilbert_bound (iteratedDeriv m.val G) hiter_C2 hiter_per
          (ts_G.sup_bounds ⟨m.val + 2, by omega⟩) h_bound2 x
      -- Identify the constructor's sup_bounds slot.
      have h_select :
          (TaylorSeriesData.fft_hilbert ts_G newDerivs newDerivsErr
              Gpp_bound Gppp_bound).sup_bounds m
            = ts_G.sup_bounds ⟨m.val + 2, by omega⟩ * Real.pi ^ 2 / 3 := by
        change (if h : m.val + 2 ≤ M + 1 then
                ts_G.sup_bounds ⟨m.val + 2, by omega⟩ * Real.pi ^ 2 / 3
              else if hM : m.val = M then
                Gpp_bound * Real.pi ^ 2 / 3
              else
                Gppp_bound * Real.pi ^ 2 / 3) =
              ts_G.sup_bounds ⟨m.val + 2, by omega⟩ * Real.pi ^ 2 / 3
        rw [dif_pos hlo]
      rw [h_select]
      exact h_N3
    · -- Upper-level case: m.val ∈ {M, M+1}. Two sub-cases.
      by_cases hM : m.val = M
      · -- m.val = M: bound by Gpp_bound · π²/3.
        have h_iter_eq : iteratedDeriv 2 (iteratedDeriv m.val G)
            = iteratedDeriv (m.val + 2) G := by
          rw [iteratedDeriv_iteratedDeriv 2 m.val G]
          congr 1; omega
        have h_bound2 : ∀ u, |iteratedDeriv 2 (iteratedDeriv m.val G) u|
            ≤ Gpp_bound := by
          intro u
          rw [h_iter_eq, hM]
          exact h_Gpp u
        have h_N3 := fourier_tail_hilbert_bound (iteratedDeriv m.val G) hiter_C2
            hiter_per Gpp_bound h_bound2 x
        have h_select :
            (TaylorSeriesData.fft_hilbert ts_G newDerivs newDerivsErr
                Gpp_bound Gppp_bound).sup_bounds m
              = Gpp_bound * Real.pi ^ 2 / 3 := by
          change (if h : m.val + 2 ≤ M + 1 then
                  ts_G.sup_bounds ⟨m.val + 2, by omega⟩ * Real.pi ^ 2 / 3
                else if hM' : m.val = M then
                  Gpp_bound * Real.pi ^ 2 / 3
                else
                  Gppp_bound * Real.pi ^ 2 / 3) =
                Gpp_bound * Real.pi ^ 2 / 3
          rw [dif_neg hlo, dif_pos hM]
        rw [h_select]
        exact h_N3
      · -- m.val ≠ M and m.val > M - 1, so m.val = M + 1.
        have hMp1 : m.val = M + 1 := by omega
        have h_iter_eq : iteratedDeriv 2 (iteratedDeriv m.val G)
            = iteratedDeriv (m.val + 2) G := by
          rw [iteratedDeriv_iteratedDeriv 2 m.val G]
          congr 1; omega
        have h_bound2 : ∀ u, |iteratedDeriv 2 (iteratedDeriv m.val G) u|
            ≤ Gppp_bound := by
          intro u
          rw [h_iter_eq, hMp1]
          have h3 : M + 1 + 2 = M + 3 := by omega
          rw [h3]
          exact h_Gppp u
        have h_N3 := fourier_tail_hilbert_bound (iteratedDeriv m.val G) hiter_C2
            hiter_per Gppp_bound h_bound2 x
        have h_select :
            (TaylorSeriesData.fft_hilbert ts_G newDerivs newDerivsErr
                Gpp_bound Gppp_bound).sup_bounds m
              = Gppp_bound * Real.pi ^ 2 / 3 := by
          change (if h : m.val + 2 ≤ M + 1 then
                  ts_G.sup_bounds ⟨m.val + 2, by omega⟩ * Real.pi ^ 2 / 3
                else if hM' : m.val = M then
                  Gpp_bound * Real.pi ^ 2 / 3
                else
                  Gppp_bound * Real.pi ^ 2 / 3) =
                Gppp_bound * Real.pi ^ 2 / 3
          rw [dif_neg hlo, dif_neg hM]
        rw [h_select]
        exact h_N3
  · -- h_pos
    exact h_G.h_pos

-- ===========================================================================
-- Sanity checks
-- ===========================================================================

#check @aliasingBound
#check @fft_hilbert_correct
#check @TaylorSeriesData.fft_hilbert
#check @fft_hilbert_series_correct

end WaveCAP

-- Both F1 and F2 endpoints depend only on `propext, Classical.choice, Quot.sound`
-- (no new WaveCAP-specific axioms beyond what N3/N4 already use).
#print axioms WaveCAP.aliasingBound_nonneg
#print axioms WaveCAP.fft_hilbert_correct
#print axioms WaveCAP.fft_hilbert_series_correct
