import WaveCAP.Common
import WaveCAP.FaaDiBrunoBound
import WaveCAP.TaylorSeriesCorrectness

/-!
# Area B — Correctness of `sqrt_series` and `inv_series` constructions

This file justifies the two analytic-function compositions in Julia:

  * `BabenkoRigorous/src/TaylorSeries.jl:606-644` (`sqrt_series`)
  * `BabenkoRigorous/src/TaylorSeries.jl:517-553` (`inv_series`)

Both take a `TaylorSeries` for some real function `f` and produce a
`TaylorSeries` for `sqrt(f)` (resp. `1/f`), assuming `f` stays strictly
positive (resp. nonzero) on the relevant domain.

**Modelling note.** Julia computes per-grid derivatives by a recurrence
specific to `sqrt` and `1/x` (see Julia source for the closed-form recursions
based on `g^2 = f` and `g · f = 1`). The recurrence's *true* (real-arithmetic)
output is, by construction, the iterated derivative of the composite —
the recurrence is just the chain rule unrolled. Arb interval arithmetic adds
quantified roundoff to each step.

In Lean we sidestep the recurrence and define `sqrt_series_data` / `inv_series_data`
in a "spec" form: the `derivs` field carries the *true* iterated derivatives of
`sqrt ∘ f` / `(·)⁻¹ ∘ f` at the grid points, with zero `derivs_err`. The
sup-bound vector is supplied via a `Φ`-parameter (the bound on
`|iteratedDeriv k (sqrt | inv) y|` for `y ≥ d_lower`) and the Faà di Bruno
product expansion from `FaaDiBrunoBound.abs_iteratedDeriv_comp_le`.

This is exactly the correctness content we need at this layer: the
intersection of the Lean `TaylorSeries` semantics and the Julia output is
"these are the iterated derivatives of the composite, bounded by Faà di
Bruno". The Arb roundoff is captured by `derivs_err`/`sup_bounds` slack in
the *concrete* Julia output, which the Lean `IsTaylorSeriesOf` predicate
already tolerates termwise.

**Proof status:**
  * `sqrt_series_correct` — fully proven (delegates contdiff to
    `ContDiff.sqrt`, grid_acc trivially by construction, sup_acc via
    `abs_iteratedDeriv_comp_le`).
  * `inv_series_correct` — fully proven (delegates contdiff to
    `ContDiff.inv`, grid_acc trivially by construction, sup_acc via
    `abs_iteratedDeriv_comp_le`).
-/

namespace WaveCAP

open Real Set

-- ===========================================================================
-- 1. Faà di Bruno sup_bounds vector (shared between sqrt_series and inv_series)
-- ===========================================================================

/-- The Faà di Bruno sup-bound vector for the composite `g ∘ f`, in the form
required by `IsTaylorSeriesOf.sup_acc`:

  `faaDiBrunoSupBound Φ F m = ∑ c : OrderedFinpartition m, Φ c.length * ∏ j, F (c.partSize j)`

where `Φ` is the per-derivative bound for `g` and `F` is the per-derivative
bound for `f`. This is the universal (non-truncated) form; the concrete
TaylorSeries records use it with `Φ k = sup_bounds_outer (min k (M+1))` etc.

Mirrors the Julia `_faa_di_bruno_derivs` routine — see
`BabenkoRigorous/src/TaylorSeries.jl`. -/
noncomputable def faaDiBrunoSupBound (Φ F : ℕ → ℝ) (m : ℕ) : ℝ :=
  ∑ c : OrderedFinpartition m, Φ c.length * ∏ j, F (c.partSize j)

-- ===========================================================================
-- 2. sqrt_series — TaylorSeries for sqrt(f)
-- ===========================================================================

/-- The "spec" version of Julia's `sqrt_series` output, as a
`TaylorSeriesData M N`. Carries:
  * `derivs j m` = true `iteratedDeriv m (sqrt ∘ f) (a + j·h)`,
  * `derivs_err j m` = 0,
  * `sup_bounds m` = Faà di Bruno bound `faaDiBrunoSupBound Φ F m`,
where `Φ` is a per-derivative bound on `sqrt`-derivatives (typically
`_sqrt_sup_bounds_on_interval d_lower² S_upper M+1 prec` in Julia notation),
and `F` is the input sup-bound vector `ts.sup_bounds`.

The `derivs` field is `noncomputable` (refers to iteratedDeriv). The Julia
side actually computes via the chain-rule recurrence and feeds Arb roundoff
into `derivs_err`; here we model the *exact* output.

Cross-reference: `BabenkoRigorous/src/TaylorSeries.jl:606-644` (`sqrt_series`). -/
noncomputable def sqrt_series_data
    {M N : ℕ} (ts : TaylorSeriesData M N) (f : ℝ → ℝ) (a : ℝ)
    (Φ : ℕ → ℝ) : TaylorSeriesData M N where
  derivs     := fun j m =>
    iteratedDeriv m.val (fun x => Real.sqrt (f x)) (a + (j.val : ℝ) * ts.h)
  derivs_err := fun _ _ => 0
  sup_bounds := fun m =>
    faaDiBrunoSupBound Φ (fun k => ts.sup_bounds ⟨min k (M + 1), by omega⟩) m.val
  h          := ts.h

/-- Correctness of `sqrt_series_data`. If `ts` describes `f` on a grid with
spacing `h`, and `f > 0` everywhere (witness `d_lower > 0` with
`d_lower ≤ f x`), and `Φ k` is a uniform sup-bound on `|iteratedDeriv k Real.sqrt y|`
for `y ≥ d_lower`, then `sqrt_series_data ts f a Φ` describes `sqrt ∘ f`.

The `Φ` hypothesis is the natural shape of `_sqrt_sup_bounds_on_interval`:
the Julia routine produces a vector with entry `k` an upper bound on
`max{|sqrt^{(k)}(y)| : d_lower ≤ y}`. The Lean theorem statement is
parametric in this bound (the user supplies the actual closed form when
instantiating; this keeps the proof one level higher than the
`(2k-3)!! / 2^k · d_lower^{1/2-k}` arithmetic).

Cross-reference: `BabenkoRigorous/src/TaylorSeries.jl:606-644` (`sqrt_series`),
`:571-590` (`_sqrt_sup_bounds_on_interval`). -/
theorem sqrt_series_correct
    {M N : ℕ} {ts : TaylorSeriesData M N} {f : ℝ → ℝ} {a : ℝ}
    (h : IsTaylorSeriesOf ts f a)
    (d_lower : ℝ) (h_d_pos : 0 < d_lower)
    (h_f_pos : ∀ x, d_lower ≤ f x)
    (Φ : ℕ → ℝ)
    (h_Φ : ∀ (k : ℕ) (y : ℝ), d_lower ≤ y →
      |iteratedDeriv k (fun z : ℝ => Real.sqrt z) y| ≤ Φ k) :
    IsTaylorSeriesOf (sqrt_series_data ts f a Φ)
      (fun x => Real.sqrt (f x)) a := by
  -- Auxiliary: f is strictly positive (in particular f x ≠ 0) everywhere.
  have h_f_ne_zero : ∀ x, f x ≠ 0 := fun x => by
    have h1 : d_lower ≤ f x := h_f_pos x
    have h2 : 0 < f x := lt_of_lt_of_le h_d_pos h1
    exact ne_of_gt h2
  refine ⟨?_, ?_, ?_, ?_⟩
  · -- contdiff: ContDiff (M+1) (sqrt ∘ f) follows from ContDiff (M+1) f and
    -- f x ≠ 0 (avoids the non-differentiability of sqrt at 0).
    exact h.contdiff.sqrt h_f_ne_zero
  · -- grid_acc: the new derivs *are* the true iterated derivatives, so the
    -- difference is 0 ≤ 0 = derivs_err.
    intro j m
    change |iteratedDeriv m.val (fun x => Real.sqrt (f x))
            (a + (j.val : ℝ) * ts.h)
          - iteratedDeriv m.val (fun x => Real.sqrt (f x))
              (a + (j.val : ℝ) * ts.h)|
        ≤ 0
    simp
  · -- sup_acc: bound |iteratedDeriv m.val (sqrt ∘ f) x| via Faà di Bruno.
    intro m x
    -- We need to bridge between `iteratedDeriv m.val (fun x => Real.sqrt (f x))`
    -- and `iteratedDeriv m.val ((Real.sqrt) ∘ f)`. These are equal by funext
    -- of the inner expression — they are definitionally `(Real.sqrt) ∘ f`.
    have hm_le : m.val ≤ M + 1 := Nat.lt_succ_iff.mp m.isLt
    -- ContDiff of sqrt at f x (needs f x ≠ 0).
    have hsqrt_at : ContDiffAt ℝ m.val (fun z : ℝ => Real.sqrt z) (f x) := by
      have := Real.contDiffAt_sqrt (h_f_ne_zero x) (n := (m.val : WithTop ℕ∞))
      simpa using this
    have hf_at : ContDiffAt ℝ m.val f x :=
      h.contdiff.contDiffAt.of_le (Nat.mono_cast hm_le)
    -- Pointwise bound on |iteratedDeriv k sqrt (f x)| via h_Φ + h_f_pos.
    have hΦ_at : ∀ k ≤ m.val,
        |iteratedDeriv k (fun z : ℝ => Real.sqrt z) (f x)| ≤ Φ k := by
      intro k _
      exact h_Φ k (f x) (h_f_pos x)
    -- Pointwise bound on |iteratedDeriv k f x| via ts.sup_bounds (truncated at M+1).
    have hF_at : ∀ k ≤ m.val,
        |iteratedDeriv k f x| ≤
          ts.sup_bounds ⟨min k (M + 1), by omega⟩ := by
      intro k hk
      have hk_le : k ≤ M + 1 := le_trans hk hm_le
      have hmin : min k (M + 1) = k := min_eq_left hk_le
      have hsup := h.sup_acc ⟨min k (M + 1), by omega⟩ x
      simpa [hmin] using hsup
    -- Apply FaaDiBrunoBound.abs_iteratedDeriv_comp_le.
    have hbound :
        |iteratedDeriv m.val ((fun z : ℝ => Real.sqrt z) ∘ f) x| ≤
          ∑ c : OrderedFinpartition m.val,
            Φ c.length *
              ∏ j, ts.sup_bounds ⟨min (c.partSize j) (M + 1), by omega⟩ :=
      abs_iteratedDeriv_comp_le hsqrt_at hf_at hΦ_at hF_at
    -- Bridge: (sqrt ∘ f) x = sqrt (f x).
    have h_comp_eq :
        ((fun z : ℝ => Real.sqrt z) ∘ f) = (fun x => Real.sqrt (f x)) := rfl
    rw [h_comp_eq] at hbound
    -- The sup_bounds entry m of sqrt_series_data is exactly this sum.
    change |iteratedDeriv m.val (fun x => Real.sqrt (f x)) x| ≤
        faaDiBrunoSupBound Φ
          (fun k => ts.sup_bounds ⟨min k (M + 1), by omega⟩) m.val
    unfold faaDiBrunoSupBound
    exact hbound
  · -- h_pos: unchanged from input ts.
    exact h.h_pos

-- ===========================================================================
-- 3. inv_series — TaylorSeries for 1/f
-- ===========================================================================

/-- The "spec" version of Julia's `inv_series` output, as a
`TaylorSeriesData M N`. Same shape as `sqrt_series_data` but with
`Real.sqrt` replaced by `(·)⁻¹`.

Cross-reference: `BabenkoRigorous/src/TaylorSeries.jl:517-553` (`inv_series`). -/
noncomputable def inv_series_data
    {M N : ℕ} (ts : TaylorSeriesData M N) (f : ℝ → ℝ) (a : ℝ)
    (Φ : ℕ → ℝ) : TaylorSeriesData M N where
  derivs     := fun j m =>
    iteratedDeriv m.val (fun x => (f x)⁻¹) (a + (j.val : ℝ) * ts.h)
  derivs_err := fun _ _ => 0
  sup_bounds := fun m =>
    faaDiBrunoSupBound Φ (fun k => ts.sup_bounds ⟨min k (M + 1), by omega⟩) m.val
  h          := ts.h

/-- Correctness of `inv_series_data`. If `ts` describes `f` on a grid, and
`f ≥ d_lower > 0` everywhere, and `Φ k` is a uniform sup-bound on
`|iteratedDeriv k (·)⁻¹ y|` for `y ≥ d_lower`, then `inv_series_data ts f a Φ`
describes `(f ·)⁻¹`.

The `Φ` hypothesis is the natural shape of `_inv_sup_bounds_on_interval`:
the Julia routine produces a vector with entry `k` an upper bound on
`max{|((·)⁻¹)^{(k)}(y)| : d_lower ≤ y} = k! / d_lower^{k+1}` (since
`((·)⁻¹)^{(k)}(y) = (-1)^k k! / y^{k+1}` and `1/y^{k+1}` is decreasing
in `y` on `(0, ∞)`).

Cross-reference: `BabenkoRigorous/src/TaylorSeries.jl:517-553` (`inv_series`),
`:488-499` (`_inv_sup_bounds_on_interval`). -/
theorem inv_series_correct
    {M N : ℕ} {ts : TaylorSeriesData M N} {f : ℝ → ℝ} {a : ℝ}
    (h : IsTaylorSeriesOf ts f a)
    (d_lower : ℝ) (h_d_pos : 0 < d_lower)
    (h_f_pos : ∀ x, d_lower ≤ f x)
    (Φ : ℕ → ℝ)
    (h_Φ : ∀ (k : ℕ) (y : ℝ), d_lower ≤ y →
      |iteratedDeriv k (fun z : ℝ => z⁻¹) y| ≤ Φ k) :
    IsTaylorSeriesOf (inv_series_data ts f a Φ)
      (fun x => (f x)⁻¹) a := by
  -- Auxiliary: f is strictly positive (in particular f x ≠ 0) everywhere.
  have h_f_ne_zero : ∀ x, f x ≠ 0 := fun x => by
    have h1 : d_lower ≤ f x := h_f_pos x
    have h2 : 0 < f x := lt_of_lt_of_le h_d_pos h1
    exact ne_of_gt h2
  refine ⟨?_, ?_, ?_, ?_⟩
  · -- contdiff: ContDiff (M+1) (1/f) follows from ContDiff (M+1) f and f x ≠ 0.
    exact h.contdiff.inv h_f_ne_zero
  · -- grid_acc: same as sqrt_series, the new derivs *are* the true iterated
    -- derivatives, so difference is 0 ≤ 0 = derivs_err.
    intro j m
    change |iteratedDeriv m.val (fun x => (f x)⁻¹)
            (a + (j.val : ℝ) * ts.h)
          - iteratedDeriv m.val (fun x => (f x)⁻¹)
              (a + (j.val : ℝ) * ts.h)|
        ≤ 0
    simp
  · -- sup_acc: bound |iteratedDeriv m.val ((·)⁻¹ ∘ f) x| via Faà di Bruno.
    intro m x
    have hm_le : m.val ≤ M + 1 := Nat.lt_succ_iff.mp m.isLt
    -- ContDiff of (·)⁻¹ at f x (needs f x ≠ 0).
    have hinv_at : ContDiffAt ℝ m.val (fun z : ℝ => z⁻¹) (f x) := by
      exact contDiffAt_inv ℝ (h_f_ne_zero x)
    have hf_at : ContDiffAt ℝ m.val f x :=
      h.contdiff.contDiffAt.of_le (Nat.mono_cast hm_le)
    -- Pointwise bound on |iteratedDeriv k (·)⁻¹ (f x)| via h_Φ + h_f_pos.
    have hΦ_at : ∀ k ≤ m.val,
        |iteratedDeriv k (fun z : ℝ => z⁻¹) (f x)| ≤ Φ k := by
      intro k _
      exact h_Φ k (f x) (h_f_pos x)
    -- Pointwise bound on |iteratedDeriv k f x| via ts.sup_bounds.
    have hF_at : ∀ k ≤ m.val,
        |iteratedDeriv k f x| ≤
          ts.sup_bounds ⟨min k (M + 1), by omega⟩ := by
      intro k hk
      have hk_le : k ≤ M + 1 := le_trans hk hm_le
      have hmin : min k (M + 1) = k := min_eq_left hk_le
      have hsup := h.sup_acc ⟨min k (M + 1), by omega⟩ x
      simpa [hmin] using hsup
    -- Apply FaaDiBrunoBound.abs_iteratedDeriv_comp_le.
    have hbound :
        |iteratedDeriv m.val ((fun z : ℝ => z⁻¹) ∘ f) x| ≤
          ∑ c : OrderedFinpartition m.val,
            Φ c.length *
              ∏ j, ts.sup_bounds ⟨min (c.partSize j) (M + 1), by omega⟩ :=
      abs_iteratedDeriv_comp_le hinv_at hf_at hΦ_at hF_at
    -- Bridge: ((·)⁻¹ ∘ f) x = (f x)⁻¹.
    have h_comp_eq :
        ((fun z : ℝ => z⁻¹) ∘ f) = (fun x => (f x)⁻¹) := rfl
    rw [h_comp_eq] at hbound
    change |iteratedDeriv m.val (fun x => (f x)⁻¹) x| ≤
        faaDiBrunoSupBound Φ
          (fun k => ts.sup_bounds ⟨min k (M + 1), by omega⟩) m.val
    unfold faaDiBrunoSupBound
    exact hbound
  · -- h_pos: unchanged from input ts.
    exact h.h_pos

end WaveCAP

#check @WaveCAP.sqrt_series_correct
#check @WaveCAP.inv_series_correct

#print axioms WaveCAP.sqrt_series_correct
#print axioms WaveCAP.inv_series_correct
