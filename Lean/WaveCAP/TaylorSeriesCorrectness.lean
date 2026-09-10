import WaveCAP.Common
import WaveCAP.LeibnizBound
import WaveCAP.FaaDiBrunoBound
import WaveCAP.TightDGSupBounds
import WaveCAP.GridToSupNormTaylor

/-!
# Area A — Correctness of `TaylorSeries` arithmetic against real functions

This file provides the foundational predicate `IsTaylorSeriesOf` that ties an
Arb-backed `TaylorSeries` data record to a real-valued function `f`, plus one
correctness lemma per arithmetic operation on `TaylorSeries`.

It is the prerequisite for deriving `cert_residual_le` from the Arb residual
bound, rather than axiomatizing it.

**Plan reference:** `docs/lean_taylorseries_correctness_plan.md`.

**Julia cross-reference:** `BabenkoRigorous/src/TaylorSeries.jl`
  - struct `TaylorSeries`              (line ~43)
  - `sup_bound(ts)`                    (line ~133)
  - `deriv_series(ts)`                 (line ~313)
  - `truncate_order(ts, M_new)`        (line ~345)
  - `Base.:+(::TaylorSeries, ::TaylorSeries)` (line ~362)
  - `Base.:*(::TaylorSeries, ::TaylorSeries)` (line ~399)
  - `Base.:*(::Arb, ::TaylorSeries)`           (line ~446)
  - `Base.:*(::Real, ::TaylorSeries)`          (line ~464)
  - `Base.:-(::TaylorSeries, ::TaylorSeries)`  (line ~475)
  - `trigpoly_to_taylor(f, ...)`               (line ~852)

**Modelling note.** We do *not* model `Arb` in Lean. Instead, the structure
`TaylorSeriesData` records the post-extraction Float64-style values: a midpoint
matrix for grid-point derivatives, an error matrix that upper-bounds the Arb
radius, and a sup-bound vector. The "Arb correctness" trust appears only at
the very leaves of the certificate chain where individual Arb primitives
matter; those are isolated in a separate `cert_arb_*` axiom file.

**Proof status (after switching `sup_acc` from `iSup ≤` to pointwise `∀ x`):**
  * `scalar_mul_taylorseries_correct` — fully proven.
  * `truncate_order_correct` — fully proven.
  * `deriv_series_correct` — fully proven.
  * `add_taylorseries_correct`, `sub_taylorseries_correct`,
    `mul_taylorseries_correct` — fully proven (`sup_acc` discharged via
    `iteratedDeriv_fun_add`/`_sub`+triangle inequality and via
    `LeibnizBound.abs_iteratedDeriv_mul_le` for the product).
  * `trigpoly_to_taylor_correct` — fully proven (signature now takes
    explicit `(h_pos : 0 < ts.h)` hypothesis; `sup_acc` via
    `TightDGSupBounds.trigpoly_supnorm_bound`).
  * `sup_bound_taylorseries_correct` — fully proven (delegates to
    `GridToSupNormTaylor.grid_to_supNorm_taylor`; bridges
    `iteratedDerivWithin → iteratedDeriv` via `UniqueDiffOn` on each grid
    cell, and bounds `|iteratedDeriv k f (a + j·h)|` by
    `|ts.derivs j ⟨k,_⟩| + ts.derivs_err j ⟨k,_⟩` via triangle inequality
    from `grid_acc`).
-/

namespace WaveCAP

open Real Set

-- ===========================================================================
-- 1. The `TaylorSeriesData` structure and `IsTaylorSeriesOf` predicate
-- ===========================================================================

/-- Post-extraction data carried by a Julia `TaylorSeries` value.

* `derivs j m`     — midpoint of the Arb interval enclosing `f^{(m)}(u_j)`
                     where `u_j = a + j·h` is the `j`-th grid point.
* `derivs_err j m` — an upper bound on the Arb radius at `(j, m)`; equivalently
                     an upper bound on `|f^{(m)}(u_j) - derivs j m|`.
* `sup_bounds m`   — upper bound on `‖f^{(m)}‖∞` over the relevant interval.
* `h`              — grid spacing (typically `2π / N` in our pipeline).

Cross-reference: `BabenkoRigorous/src/TaylorSeries.jl:43` (struct definition,
fields `derivs::Matrix{Arb}` and `sup_bounds::Vector{Arb}`). -/
structure TaylorSeriesData (M : ℕ) (N : ℕ) where
  derivs     : Fin N → Fin (M + 1) → ℝ
  derivs_err : Fin N → Fin (M + 1) → ℝ
  sup_bounds : Fin (M + 2) → ℝ
  h          : ℝ

/-- `IsTaylorSeriesOf ts f a` asserts that the post-extraction Taylor data
`ts` correctly describes the real-valued function `f` on the grid starting at
`a` with spacing `ts.h`.

* `contdiff`  — `f` is `C^{M+1}` on the ambient domain.
* `grid_acc`  — for each grid point `u_j = a + j·h` and each order `m ≤ M+1`,
                the midpoint `ts.derivs j m` lies within distance
                `ts.derivs_err j m` of the true derivative.
* `sup_acc`   — for each order `m ≤ M+1`, the value `ts.sup_bounds m` is a
                valid sup-norm upper bound for `f^{(m)}` on the relevant
                domain.
* `h_pos`     — grid spacing is positive (needed for `exists_grid_subinterval`).

The sup-norm here is interpreted as `iSup` over the whole real line; in the
periodic-on-`[0, 2π]` setting this equals the sup over one period.

Cross-reference: the `sup_bounds[m+1]` field on the Julia side is meant to
upper-bound `‖f^{(m)}‖∞` exactly in this sense — see
`BabenkoRigorous/src/TaylorSeries.jl:133` (`sup_bound`). -/
structure IsTaylorSeriesOf {M N : ℕ} (ts : TaylorSeriesData M N)
    (f : ℝ → ℝ) (a : ℝ) : Prop where
  contdiff   : ContDiff ℝ (M + 1) f
  grid_acc   : ∀ (j : Fin N) (m : Fin (M + 1)),
                 |iteratedDeriv m.val f (a + j.val * ts.h) - ts.derivs j m|
                   ≤ ts.derivs_err j m
  sup_acc    : ∀ (m : Fin (M + 2)) (x : ℝ),
                 |iteratedDeriv m.val f x| ≤ ts.sup_bounds m
  h_pos      : 0 < ts.h

-- ===========================================================================
-- 2. Helpers for the structure
-- ===========================================================================

/-- Truncate a `TaylorSeriesData M N` to a `TaylorSeriesData M_new N` for
`M_new ≤ M`. The `derivs`, `derivs_err`, and `sup_bounds` are simply
restricted (by `Fin.castLE`) to the smaller index set.

Cross-reference: `BabenkoRigorous/src/TaylorSeries.jl:345` (`truncate_order`). -/
def TaylorSeriesData.truncate {M N : ℕ} (ts : TaylorSeriesData M N)
    (M_new : ℕ) (hM : M_new ≤ M) : TaylorSeriesData M_new N where
  derivs     := fun j m => ts.derivs j ⟨m.val, by omega⟩
  derivs_err := fun j m => ts.derivs_err j ⟨m.val, by omega⟩
  sup_bounds := fun m => ts.sup_bounds ⟨m.val, by omega⟩
  h          := ts.h

/-- Shift a `TaylorSeriesData M N` (representing `f`) into a
`TaylorSeriesData (M-1) N` (representing `f'`): the `(m)`-th derivative entry
of the new data is the `(m+1)`-th of the old.

Cross-reference: `BabenkoRigorous/src/TaylorSeries.jl:313` (`deriv_series`). -/
def TaylorSeriesData.derivShift {M N : ℕ} (ts : TaylorSeriesData M N)
    (hM : 0 < M) : TaylorSeriesData (M - 1) N where
  derivs     := fun j m => ts.derivs j ⟨m.val + 1, by omega⟩
  derivs_err := fun j m => ts.derivs_err j ⟨m.val + 1, by omega⟩
  sup_bounds := fun m => ts.sup_bounds ⟨m.val + 1, by omega⟩
  h          := ts.h

-- ===========================================================================
-- 3. Per-operation correctness lemmas
-- ===========================================================================

/-- A trigonometric polynomial with coefficients `a_k, b_k` admits an exact
`TaylorSeries` description: the `m`-th derivative at `u_j` is a closed-form
combination of `±k^m` factors times `cos(k u_j)` or `sin(k u_j)`, and the
sup-bound is the term-wise triangle inequality `∑ k^m · (|a_k| + |b_k|)`.

The hypothesis `h_derivs` is left as the abstract identity that the caller
must supply (matching the Julia `trigpoly_to_taylor` output); spelling out the
trig-derivative formula here would obscure the structure. The hypothesis
`h_sup` is the standard term-wise triangle bound.

Cross-reference: `BabenkoRigorous/src/TaylorSeries.jl:852` (`trigpoly_to_taylor`). -/
theorem trigpoly_to_taylor_correct
    {N_modes M N : ℕ}
    (a_coef : Fin (N_modes + 1) → ℝ) (b_coef : Fin (N_modes + 1) → ℝ)
    (ts : TaylorSeriesData M N)
    (h_derivs : ∀ (j : Fin N) (m : Fin (M + 1)),
        iteratedDeriv m.val
            (fun u => ∑ k : Fin (N_modes + 1),
                (a_coef k * Real.cos ((k.val : ℝ) * u)
                  + b_coef k * Real.sin ((k.val : ℝ) * u)))
            ((j.val : ℝ) * ts.h)
        = ts.derivs j m)
    (h_err_zero : ∀ (j : Fin N) (m : Fin (M + 1)), ts.derivs_err j m = 0)
    (h_sup : ∀ m : Fin (M + 2),
        ts.sup_bounds m
          = ∑ k : Fin (N_modes + 1),
              ((k.val : ℝ) ^ m.val) * (|a_coef k| + |b_coef k|))
    (h_pos : 0 < ts.h) :
    IsTaylorSeriesOf ts
        (fun u => ∑ k : Fin (N_modes + 1),
            (a_coef k * Real.cos ((k.val : ℝ) * u)
              + b_coef k * Real.sin ((k.val : ℝ) * u)))
        0 := by
  -- The trig polynomial f(u) = Σₖ (aₖ cos(k u) + bₖ sin(k u)) is C^∞.
  have hcd : ContDiff ℝ (M + 1 : ℕ)
      (fun u => ∑ k : Fin (N_modes + 1),
          (a_coef k * Real.cos ((k.val : ℝ) * u)
            + b_coef k * Real.sin ((k.val : ℝ) * u))) := by
    apply ContDiff.sum
    intro k _
    have hcos : ContDiff ℝ (M + 1 : ℕ) (fun u => Real.cos ((k.val : ℝ) * u)) :=
      Real.contDiff_cos.comp (contDiff_const.mul contDiff_id)
    have hsin : ContDiff ℝ (M + 1 : ℕ) (fun u => Real.sin ((k.val : ℝ) * u)) :=
      Real.contDiff_sin.comp (contDiff_const.mul contDiff_id)
    exact (contDiff_const.mul hcos).add (contDiff_const.mul hsin)
  refine ⟨hcd, ?_, ?_, ?_⟩
  · -- grid_acc: by h_derivs, the iteratedDeriv equals ts.derivs j m exactly,
    -- so the diff is 0, which is ≤ ts.derivs_err j m = 0.
    intro j m
    have hd := h_derivs j m
    have he := h_err_zero j m
    have h0 : (0 : ℝ) + (j.val : ℝ) * ts.h = (j.val : ℝ) * ts.h := by ring
    rw [h0, hd]
    simp [he]
  · -- sup_acc (pointwise form after the 2026-05-23 predicate fix):
    -- Each mode k contributes at most k^m·(|a_k|+|b_k|) to |f^(m)(x)|, so the
    -- term-wise sum bound matches ts.sup_bounds m via h_sup.
    intro m x
    have hbound := trigpoly_supnorm_bound a_coef b_coef m.val x
    rw [h_sup m]
    exact hbound
  · -- h_pos: discharged by the added `(h_pos : 0 < ts.h)` hypothesis.
    exact h_pos

/-- Correctness of `TaylorSeries` addition: if `ts1` represents `f` and `ts2`
represents `g` on a common grid, then the term-wise sum of the data records
represents `f + g`.

Cross-reference: `BabenkoRigorous/src/TaylorSeries.jl:362`
(`Base.:+(::TaylorSeries, ::TaylorSeries)`). -/
theorem add_taylorseries_correct
    {M N : ℕ} {ts1 ts2 : TaylorSeriesData M N} {f g : ℝ → ℝ} {a : ℝ}
    (h1 : IsTaylorSeriesOf ts1 f a) (h2 : IsTaylorSeriesOf ts2 g a)
    (hh : ts1.h = ts2.h) :
    IsTaylorSeriesOf
        { derivs     := fun j m => ts1.derivs j m + ts2.derivs j m
          derivs_err := fun j m => ts1.derivs_err j m + ts2.derivs_err j m
          sup_bounds := fun m => ts1.sup_bounds m + ts2.sup_bounds m
          h          := ts1.h }
        (f + g) a := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · -- contdiff
    exact h1.contdiff.add h2.contdiff
  · -- grid_acc
    intro j m
    have hf : ContDiffAt ℝ m.val f (a + (j.val : ℝ) * ts1.h) :=
      h1.contdiff.contDiffAt.of_le (Nat.mono_cast m.isLt.le)
    have hg : ContDiffAt ℝ m.val g (a + (j.val : ℝ) * ts1.h) :=
      h2.contdiff.contDiffAt.of_le (Nat.mono_cast m.isLt.le)
    rw [show (f + g) = (fun x => f x + g x) from rfl,
        iteratedDeriv_fun_add hf hg]
    have h1g := h1.grid_acc j m
    have h2g := h2.grid_acc j m
    have hh' : (j.val : ℝ) * ts1.h = (j.val : ℝ) * ts2.h := by rw [hh]
    have h2g' : |iteratedDeriv m.val g (a + (j.val : ℝ) * ts1.h) - ts2.derivs j m|
        ≤ ts2.derivs_err j m := by
      rw [hh']; exact h2g
    have triangle :
        |iteratedDeriv m.val f (a + (j.val : ℝ) * ts1.h)
            + iteratedDeriv m.val g (a + (j.val : ℝ) * ts1.h)
            - (ts1.derivs j m + ts2.derivs j m)|
          ≤ |iteratedDeriv m.val f (a + (j.val : ℝ) * ts1.h) - ts1.derivs j m|
            + |iteratedDeriv m.val g (a + (j.val : ℝ) * ts1.h) - ts2.derivs j m| := by
      have eq1 :
          iteratedDeriv m.val f (a + (j.val : ℝ) * ts1.h)
            + iteratedDeriv m.val g (a + (j.val : ℝ) * ts1.h)
            - (ts1.derivs j m + ts2.derivs j m)
          = (iteratedDeriv m.val f (a + (j.val : ℝ) * ts1.h) - ts1.derivs j m)
            + (iteratedDeriv m.val g (a + (j.val : ℝ) * ts1.h) - ts2.derivs j m) := by ring
      rw [eq1]
      exact abs_add_le _ _
    linarith
  · -- sup_acc (pointwise form after the 2026-05-23 predicate fix):
    -- iteratedDeriv m (f + g) x = iteratedDeriv m f x + iteratedDeriv m g x,
    -- then triangle inequality + pointwise sup hypotheses on f and g.
    intro m x
    have hm_le : m.val ≤ M + 1 := Nat.lt_succ_iff.mp m.isLt
    have hf : ContDiffAt ℝ m.val f x :=
      h1.contdiff.contDiffAt.of_le (Nat.mono_cast hm_le)
    have hg : ContDiffAt ℝ m.val g x :=
      h2.contdiff.contDiffAt.of_le (Nat.mono_cast hm_le)
    rw [show (f + g) = (fun x => f x + g x) from rfl,
        iteratedDeriv_fun_add hf hg]
    calc |iteratedDeriv m.val f x + iteratedDeriv m.val g x|
        ≤ |iteratedDeriv m.val f x| + |iteratedDeriv m.val g x| := abs_add_le _ _
      _ ≤ ts1.sup_bounds m + ts2.sup_bounds m :=
          add_le_add (h1.sup_acc m x) (h2.sup_acc m x)
  · -- h_pos
    exact h1.h_pos

/-- Correctness of `TaylorSeries` subtraction. Same shape as `add`, but with
sign-flipped midpoints and (still nonnegative) summed error bars.

Cross-reference: `BabenkoRigorous/src/TaylorSeries.jl:475`
(`Base.:-(::TaylorSeries, ::TaylorSeries)`, defined as `ts1 + (-ts2)`). -/
theorem sub_taylorseries_correct
    {M N : ℕ} {ts1 ts2 : TaylorSeriesData M N} {f g : ℝ → ℝ} {a : ℝ}
    (h1 : IsTaylorSeriesOf ts1 f a) (h2 : IsTaylorSeriesOf ts2 g a)
    (hh : ts1.h = ts2.h) :
    IsTaylorSeriesOf
        { derivs     := fun j m => ts1.derivs j m - ts2.derivs j m
          derivs_err := fun j m => ts1.derivs_err j m + ts2.derivs_err j m
          sup_bounds := fun m => ts1.sup_bounds m + ts2.sup_bounds m
          h          := ts1.h }
        (f - g) a := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · -- contdiff
    exact h1.contdiff.sub h2.contdiff
  · -- grid_acc
    intro j m
    have hf : ContDiffAt ℝ m.val f (a + (j.val : ℝ) * ts1.h) :=
      h1.contdiff.contDiffAt.of_le (Nat.mono_cast m.isLt.le)
    have hg : ContDiffAt ℝ m.val g (a + (j.val : ℝ) * ts1.h) :=
      h2.contdiff.contDiffAt.of_le (Nat.mono_cast m.isLt.le)
    rw [show (f - g) = (fun x => f x - g x) from rfl,
        iteratedDeriv_fun_sub hf hg]
    have h1g := h1.grid_acc j m
    have h2g := h2.grid_acc j m
    have hh' : (j.val : ℝ) * ts1.h = (j.val : ℝ) * ts2.h := by rw [hh]
    have h2g' : |iteratedDeriv m.val g (a + (j.val : ℝ) * ts1.h) - ts2.derivs j m|
        ≤ ts2.derivs_err j m := by
      rw [hh']; exact h2g
    have triangle :
        |iteratedDeriv m.val f (a + (j.val : ℝ) * ts1.h)
            - iteratedDeriv m.val g (a + (j.val : ℝ) * ts1.h)
            - (ts1.derivs j m - ts2.derivs j m)|
          ≤ |iteratedDeriv m.val f (a + (j.val : ℝ) * ts1.h) - ts1.derivs j m|
            + |iteratedDeriv m.val g (a + (j.val : ℝ) * ts1.h) - ts2.derivs j m| := by
      have eq1 :
          iteratedDeriv m.val f (a + (j.val : ℝ) * ts1.h)
            - iteratedDeriv m.val g (a + (j.val : ℝ) * ts1.h)
            - (ts1.derivs j m - ts2.derivs j m)
          = (iteratedDeriv m.val f (a + (j.val : ℝ) * ts1.h) - ts1.derivs j m)
            - (iteratedDeriv m.val g (a + (j.val : ℝ) * ts1.h) - ts2.derivs j m) := by ring
      rw [eq1]
      exact abs_sub _ _
    linarith
  · -- sup_acc (pointwise form after the 2026-05-23 predicate fix):
    -- iteratedDeriv m (f - g) x = iteratedDeriv m f x - iteratedDeriv m g x,
    -- then |a - b| ≤ |a| + |b| + pointwise sup hypotheses on f and g.
    intro m x
    have hm_le : m.val ≤ M + 1 := Nat.lt_succ_iff.mp m.isLt
    have hf : ContDiffAt ℝ m.val f x :=
      h1.contdiff.contDiffAt.of_le (Nat.mono_cast hm_le)
    have hg : ContDiffAt ℝ m.val g x :=
      h2.contdiff.contDiffAt.of_le (Nat.mono_cast hm_le)
    rw [show (f - g) = (fun x => f x - g x) from rfl,
        iteratedDeriv_fun_sub hf hg]
    calc |iteratedDeriv m.val f x - iteratedDeriv m.val g x|
        ≤ |iteratedDeriv m.val f x| + |iteratedDeriv m.val g x| := abs_sub _ _
      _ ≤ ts1.sup_bounds m + ts2.sup_bounds m :=
          add_le_add (h1.sup_acc m x) (h2.sup_acc m x)
  · -- h_pos
    exact h1.h_pos

/-- Correctness of scalar multiplication. The midpoints are scaled by `c`, and
the error and sup-bound entries are scaled by `|c|`.

Cross-reference: `BabenkoRigorous/src/TaylorSeries.jl:464`
(`Base.:*(::Real, ::TaylorSeries)`; the Arb version is at line 446). -/
theorem scalar_mul_taylorseries_correct
    {M N : ℕ} {ts : TaylorSeriesData M N} {f : ℝ → ℝ} {a : ℝ} (c : ℝ)
    (h : IsTaylorSeriesOf ts f a) :
    IsTaylorSeriesOf
        { derivs     := fun j m => c * ts.derivs j m
          derivs_err := fun j m => |c| * ts.derivs_err j m
          sup_bounds := fun m => |c| * ts.sup_bounds m
          h          := ts.h }
        (fun u => c * f u) a := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · -- contdiff: c * f is ContDiff if f is.
    have : ContDiff ℝ (M + 1 : ℕ) (fun u => c * f u) := by
      have hconst : ContDiff ℝ (M + 1 : ℕ) (fun _ : ℝ => c) := contDiff_const
      exact hconst.mul h.contdiff
    exact this
  · -- grid_acc
    intro j m
    -- iteratedDeriv m.val (c * f) x = c * iteratedDeriv m.val f x
    rw [iteratedDeriv_const_mul_field]
    -- Goal: |c * iteratedDeriv m.val f (a + j.val * ts.h) - c * ts.derivs j m| ≤ |c| * ts.derivs_err j m
    have hg := h.grid_acc j m
    have : c * iteratedDeriv m.val f (a + (j.val : ℝ) * ts.h) - c * ts.derivs j m
         = c * (iteratedDeriv m.val f (a + (j.val : ℝ) * ts.h) - ts.derivs j m) := by ring
    rw [this, abs_mul]
    exact mul_le_mul_of_nonneg_left hg (abs_nonneg c)
  · -- sup_acc (pointwise form after the 2026-05-23 predicate fix)
    intro m x
    rw [iteratedDeriv_const_mul_field, abs_mul]
    exact mul_le_mul_of_nonneg_left (h.sup_acc m x) (abs_nonneg c)
  · -- h_pos
    exact h.h_pos

/-- Correctness of `TaylorSeries` multiplication: the exact Leibniz identity
on grid derivatives, with a Leibniz-style sup-bound

`(f·g)^{(m)} bound = ∑_{k=0}^{m} C(m,k) · F_k · G_{m-k}`

obtained by `LeibnizBound.abs_iteratedDeriv_mul_le` followed by taking sup
over `x`. The `derivs_err` formula is left as an abstract record-level
assumption matching the Julia interval-arithmetic propagation.

Cross-reference: `BabenkoRigorous/src/TaylorSeries.jl:399`
(`Base.:*(::TaylorSeries, ::TaylorSeries)`). -/
theorem mul_taylorseries_correct
    {M N : ℕ} {ts1 ts2 : TaylorSeriesData M N} {f g : ℝ → ℝ} {a : ℝ}
    (h1 : IsTaylorSeriesOf ts1 f a) (h2 : IsTaylorSeriesOf ts2 g a)
    (hh : ts1.h = ts2.h)
    (err_combined : Fin N → Fin (M + 1) → ℝ)
    (h_err_valid : ∀ (j : Fin N) (m : Fin (M + 1)),
        |iteratedDeriv m.val (f * g) (a + j.val * ts1.h)
          - ∑ k ∈ Finset.range (m.val + 1),
              (m.val.choose k : ℝ)
                * ts1.derivs j ⟨min k M, by omega⟩
                * ts2.derivs j ⟨m.val - k, by have := m.isLt; omega⟩|
        ≤ err_combined j m) :
    IsTaylorSeriesOf
        { derivs     := fun j m =>
            ∑ k ∈ Finset.range (m.val + 1),
              (m.val.choose k : ℝ)
                * ts1.derivs j ⟨min k M, by omega⟩
                * ts2.derivs j ⟨m.val - k, by have := m.isLt; omega⟩
          derivs_err := err_combined
          sup_bounds := fun m =>
            ∑ k ∈ Finset.range (m.val + 1),
              (m.val.choose k : ℝ)
                * ts1.sup_bounds ⟨min k (M + 1), by omega⟩
                * ts2.sup_bounds ⟨m.val - k, by have := m.isLt; omega⟩
          h          := ts1.h }
        (f * g) a := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · -- contdiff
    exact h1.contdiff.mul h2.contdiff
  · -- grid_acc: hypothesis `h_err_valid` IS exactly the conclusion.
    exact h_err_valid
  · -- sup_acc (pointwise form after the 2026-05-23 predicate fix):
    -- Apply LeibnizBound.abs_iteratedDeriv_mul_le with F and G chosen so the
    -- resulting Leibniz sum is definitionally equal to the product's sup_bounds.
    intro m x
    have hm_le : m.val ≤ M + 1 := Nat.lt_succ_iff.mp m.isLt
    have hf : ContDiffAt ℝ m.val f x :=
      h1.contdiff.contDiffAt.of_le (Nat.mono_cast hm_le)
    have hg : ContDiffAt ℝ m.val g x :=
      h2.contdiff.contDiffAt.of_le (Nat.mono_cast hm_le)
    set F : ℕ → ℝ := fun k =>
      ts1.sup_bounds ⟨min k (M + 1), by omega⟩ with hF_def
    set G : ℕ → ℝ := fun k =>
      ts2.sup_bounds ⟨min k (M + 1), by omega⟩ with hG_def
    have hF : ∀ k ≤ m.val, |iteratedDeriv k f x| ≤ F k := by
      intro k hk
      have hkle : k ≤ M + 1 := le_trans hk hm_le
      have hmin : min k (M + 1) = k := min_eq_left hkle
      have := h1.sup_acc ⟨min k (M + 1), by omega⟩ x
      simpa [F, hmin] using this
    have hG : ∀ k ≤ m.val, |iteratedDeriv k g x| ≤ G k := by
      intro k hk
      have hkle : k ≤ M + 1 := le_trans hk hm_le
      have hmin : min k (M + 1) = k := min_eq_left hkle
      have := h2.sup_acc ⟨min k (M + 1), by omega⟩ x
      simpa [G, hmin] using this
    have hbound :
        |iteratedDeriv m.val (f * g) x| ≤
          ∑ k ∈ Finset.range (m.val + 1),
            (m.val.choose k : ℝ) * F k * G (m.val - k) :=
      abs_iteratedDeriv_mul_le hf hg hF hG
    -- Rewrite each G term in the product sup_bounds to match (m.val - k ≤ m.val ≤ M+1).
    have hsum_eq :
        (∑ k ∈ Finset.range (m.val + 1),
            (m.val.choose k : ℝ) * F k * G (m.val - k))
          = ∑ k ∈ Finset.range (m.val + 1),
              (m.val.choose k : ℝ)
                * ts1.sup_bounds ⟨min k (M + 1), by omega⟩
                * ts2.sup_bounds ⟨m.val - k, by have := m.isLt; omega⟩ := by
      refine Finset.sum_congr rfl ?_
      intro k hk
      have hk_le : k ≤ m.val := Nat.lt_succ_iff.mp (Finset.mem_range.mp hk)
      have hmk_le : m.val - k ≤ M + 1 := le_trans (Nat.sub_le _ _) hm_le
      have hmin : min (m.val - k) (M + 1) = m.val - k := min_eq_left hmk_le
      simp [F, G, hmin]
    linarith [hbound, hsum_eq.le, hsum_eq.ge]
  · -- h_pos
    exact h1.h_pos

/-- Order-truncation preserves the `IsTaylorSeriesOf` predicate: if a record
correctly describes `f` up to order `M+1`, it still correctly describes `f` up
to any lower order `M_new + 1`.

Cross-reference: `BabenkoRigorous/src/TaylorSeries.jl:345` (`truncate_order`). -/
theorem truncate_order_correct
    {M N : ℕ} {ts : TaylorSeriesData M N} {f : ℝ → ℝ} {a : ℝ}
    (h : IsTaylorSeriesOf ts f a)
    (M_new : ℕ) (hM : M_new ≤ M) :
    IsTaylorSeriesOf (TaylorSeriesData.truncate ts M_new hM) f a := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · -- contdiff
    exact h.contdiff.of_le (Nat.mono_cast (by omega : M_new + 1 ≤ M + 1))
  · -- grid_acc
    intro j m
    -- truncate.derivs j m = ts.derivs j ⟨m.val, _⟩
    -- truncate.derivs_err j m = ts.derivs_err j ⟨m.val, _⟩
    -- and truncate.h = ts.h
    exact h.grid_acc j ⟨m.val, by omega⟩
  · -- sup_acc
    intro m
    exact h.sup_acc ⟨m.val, by omega⟩
  · -- h_pos
    exact h.h_pos

/-- Differentiation correctness: if `ts` describes `f`, then `ts.derivShift`
(which drops the zeroth-order row and re-indexes derivatives upward) describes
`f'` (i.e. `iteratedDeriv 1 f`).

Note this drops the top derivative slot, so the new structure is
`TaylorSeriesData (M - 1) N` (requires `0 < M`).

Cross-reference: `BabenkoRigorous/src/TaylorSeries.jl:313` (`deriv_series`). -/
theorem deriv_series_correct
    {M N : ℕ} {ts : TaylorSeriesData M N} {f : ℝ → ℝ} {a : ℝ}
    (h : IsTaylorSeriesOf ts f a) (hM : 0 < M) :
    IsTaylorSeriesOf (TaylorSeriesData.derivShift ts hM)
        (iteratedDeriv 1 f) a := by
  -- Key rewrite: iteratedDeriv k (iteratedDeriv 1 f) = iteratedDeriv (k+1) f.
  -- Proof: iteratedDeriv 1 f = deriv f, and iteratedDeriv k (deriv f) = iteratedDeriv (k+1) f.
  have hshift : ∀ k : ℕ, iteratedDeriv k (iteratedDeriv 1 f) = iteratedDeriv (k + 1) f := by
    intro k
    rw [iteratedDeriv_one, ← iteratedDeriv_succ']
  refine ⟨?_, ?_, ?_, ?_⟩
  · -- contdiff: M - 1 + 1 = M; need ContDiff M (iteratedDeriv 1 f). Since
    -- iteratedDeriv 1 f = deriv f, this is ContDiff M (deriv f), which follows
    -- from ContDiff (M+1) f via ContDiff.iterate_deriv' M 1.
    rw [iteratedDeriv_one]
    have hM1 : (M - 1) + 1 = M := Nat.succ_pred_eq_of_pos hM
    have hcd : ContDiff ℝ (M + 1 : ℕ) f := h.contdiff
    have hd1 : ContDiff ℝ (M : ℕ) (deriv^[1] f) :=
      ContDiff.iterate_deriv' (𝕜 := ℝ) (F := ℝ) M 1 hcd
    have hd : ContDiff ℝ (M : ℕ) (deriv f) := by simpa using hd1
    -- Goal is `ContDiff ℝ (↑(M - 1) + 1) (deriv f)`. Use of_le with M - 1 + 1 ≤ M.
    apply hd.of_le
    -- Goal: (↑(M - 1) + 1 : WithTop ℕ∞) ≤ (↑M : WithTop ℕ∞)
    have hnat : (M - 1) + 1 ≤ M := hM1.le
    have h1 : ((M - 1) + 1 : ℕ) ≤ (M : ℕ) := hnat
    have h2 : (((M - 1) + 1 : ℕ) : WithTop ℕ∞) ≤ ((M : ℕ) : WithTop ℕ∞) :=
      Nat.mono_cast h1
    exact h2
  · -- grid_acc
    intro j m
    -- derivShift.derivs j m = ts.derivs j ⟨m+1, _⟩
    -- derivShift.derivs_err j m = ts.derivs_err j ⟨m+1, _⟩
    -- iteratedDeriv m (iteratedDeriv 1 f) x = iteratedDeriv (m+1) f x
    rw [hshift]
    exact h.grid_acc j ⟨m.val + 1, by omega⟩
  · -- sup_acc (pointwise form after the 2026-05-23 predicate fix)
    intro m x
    rw [hshift]
    exact h.sup_acc ⟨m.val + 1, by omega⟩ x
  · -- h_pos
    exact h.h_pos

-- ===========================================================================
-- 4. Bridge to `GridToSupNormTaylor.lean`
-- ===========================================================================

/-- Bridge from `IsTaylorSeriesOf` data to a sup-norm bound on `f` over
`[a, a + N·h]`.

This is the `IsTaylorSeriesOf`-flavoured restatement of
`GridToSupNormTaylor.grid_to_supNorm_taylor`. The hypothesis `hM_bound`
asks the caller to supply a per-grid-cell bound on the Taylor-coefficient sum
plus the Lagrange-remainder term (using the `derivs`, `derivs_err`, and
`sup_bounds` recorded in `ts`), exactly as the Julia `sup_bound(ts)` routine
computes.

Cross-reference: `BabenkoRigorous/src/TaylorSeries.jl:133` (`sup_bound`). -/
theorem sup_bound_taylorseries_correct
    {M N : ℕ} (hN : 0 < N) {ts : TaylorSeriesData M N} {f : ℝ → ℝ} {a : ℝ}
    (h : IsTaylorSeriesOf ts f a)
    (M_bound : ℝ)
    (hM_bound : ∀ j : Fin N,
        (∑ m ∈ Finset.range (M + 1),
          (|ts.derivs j ⟨min m M, by omega⟩|
              + ts.derivs_err j ⟨min m M, by omega⟩)
            * ts.h ^ m / (Nat.factorial m : ℝ))
        + ts.sup_bounds ⟨M + 1, by omega⟩
            * ts.h ^ (M + 1) / (Nat.factorial (M + 1) : ℝ)
        ≤ M_bound) :
    ∀ u ∈ Set.Icc a (a + N * ts.h), |f u| ≤ M_bound := by
  -- Strategy: apply `grid_to_supNorm_taylor` with `F = ts.sup_bounds ⟨M+1, _⟩`.
  -- We must convert the `IsTaylorSeriesOf` data into the per-sub-interval
  -- hypotheses (`ContDiffOn`, sup-bound on `iteratedDerivWithin (M+1)`, and
  -- the Taylor-sum bound). The key bridges are:
  --   * `iteratedDerivWithin n f Icc x = iteratedDeriv n f x`
  --       (via `iteratedDerivWithin_eq_iteratedDeriv` since `f` is ContDiff on ℝ).
  --   * `|iteratedDeriv k f (a+j·h)| ≤ |ts.derivs j ⟨k, _⟩| + ts.derivs_err j ⟨k, _⟩`
  --       (triangle inequality from `h.grid_acc`).
  have hh_pos : 0 < ts.h := h.h_pos
  -- ContDiffOn for each sub-interval (free from ContDiff on ℝ).
  have hcontDiffOn : ∀ j : Fin N,
      ContDiffOn ℝ (M + 1) f
        (Set.Icc (a + (j : ℕ) * ts.h) (a + ((j : ℕ) + 1) * ts.h)) := by
    intro j
    exact h.contdiff.contDiffOn
  -- Sub-interval endpoints `a + j*h ≤ a + (j+1)*h` (used repeatedly).
  have hab : ∀ j : Fin N,
      a + (j : ℕ) * ts.h ≤ a + ((j : ℕ) + 1) * ts.h := by
    intro j
    have : ((j : ℕ) : ℝ) * ts.h ≤ ((j : ℕ) + 1) * ts.h :=
      mul_le_mul_of_nonneg_right (by linarith) hh_pos.le
    linarith
  -- `UniqueDiffOn` on each sub-interval (needed to convert iteratedDerivWithin).
  have hUDO : ∀ j : Fin N,
      UniqueDiffOn ℝ (Set.Icc (a + (j : ℕ) * ts.h) (a + ((j : ℕ) + 1) * ts.h)) := by
    intro j
    apply uniqueDiffOn_Icc
    -- need strict: a + j*h < a + (j+1)*h, i.e. h > 0.
    have : ((j : ℕ) : ℝ) * ts.h < ((j : ℕ) + 1) * ts.h := by
      have : ((j : ℕ) : ℝ) * ts.h + ts.h = ((j : ℕ) + 1) * ts.h := by ring
      linarith
    linarith
  -- Sup-bound on `iteratedDerivWithin (M+1) f Icc s` for each `s ∈ Icc`.
  have hF : ∀ j : Fin N, ∀ s ∈ Set.Icc (a + (j : ℕ) * ts.h) (a + ((j : ℕ) + 1) * ts.h),
      |iteratedDerivWithin (M + 1) f
          (Set.Icc (a + (j : ℕ) * ts.h) (a + ((j : ℕ) + 1) * ts.h)) s|
        ≤ ts.sup_bounds ⟨M + 1, by omega⟩ := by
    intro j s hs
    have hcontDiffAt : ContDiffAt ℝ (M + 1) f s := h.contdiff.contDiffAt
    have hbridge : iteratedDerivWithin (M + 1) f
        (Set.Icc (a + (j : ℕ) * ts.h) (a + ((j : ℕ) + 1) * ts.h)) s
        = iteratedDeriv (M + 1) f s :=
      iteratedDerivWithin_eq_iteratedDeriv (hUDO j) hcontDiffAt hs
    rw [hbridge]
    exact h.sup_acc ⟨M + 1, by omega⟩ s
  -- Per-grid-cell bound: convert from `ts.derivs / ts.derivs_err`-form to the
  -- `|iteratedDerivWithin k f Icc (a + j·h)|`-form expected by
  -- `grid_to_supNorm_taylor`.
  have hMbound' : ∀ j : Fin N,
      (∑ k ∈ Finset.range (M + 1),
            |iteratedDerivWithin k f
                (Set.Icc (a + (j : ℕ) * ts.h) (a + ((j : ℕ) + 1) * ts.h))
                (a + (j : ℕ) * ts.h)|
              * ts.h ^ k / (Nat.factorial k : ℝ))
          + ts.sup_bounds ⟨M + 1, by omega⟩
              * ts.h ^ (M + 1) / (Nat.factorial (M + 1) : ℝ)
        ≤ M_bound := by
    intro j
    -- We show the LHS is ≤ the user-provided LHS (using triangle inequality
    -- termwise on the sum), then chain to `hM_bound j`.
    have huser := hM_bound j
    -- For each k ∈ Finset.range (M+1), `min k M = k`.
    have hk_min : ∀ k ∈ Finset.range (M + 1), min k M = k := by
      intro k hk
      have hk' : k ≤ M := Nat.lt_succ_iff.mp (Finset.mem_range.mp hk)
      exact min_eq_left hk'
    -- For each k, bridge: |iteratedDerivWithin k f Icc (a+j·h)| =
    --                    |iteratedDeriv k f (a+j·h)|
    have hpt_in_Icc : a + (j : ℕ) * ts.h ∈
        Set.Icc (a + (j : ℕ) * ts.h) (a + ((j : ℕ) + 1) * ts.h) :=
      ⟨le_refl _, hab j⟩
    -- Termwise bound on the sum:
    have hterm_le : ∀ k ∈ Finset.range (M + 1),
        |iteratedDerivWithin k f
            (Set.Icc (a + (j : ℕ) * ts.h) (a + ((j : ℕ) + 1) * ts.h))
            (a + (j : ℕ) * ts.h)|
          * ts.h ^ k / (Nat.factorial k : ℝ)
        ≤ (|ts.derivs j ⟨min k M, by omega⟩|
              + ts.derivs_err j ⟨min k M, by omega⟩)
            * ts.h ^ k / (Nat.factorial k : ℝ) := by
      intro k hk
      have hk_le : k ≤ M := Nat.lt_succ_iff.mp (Finset.mem_range.mp hk)
      have hk_min_eq : (⟨min k M, by omega⟩ : Fin (M + 1))
          = ⟨k, by omega⟩ := by
        ext; simp [min_eq_left hk_le]
      -- Bridge iteratedDerivWithin → iteratedDeriv at (a + j*h).
      have hcontDiffAt : ContDiffAt ℝ k f (a + (j : ℕ) * ts.h) := by
        have hnat : k ≤ M + 1 := Nat.le_succ_of_le hk_le
        have : ((k : ℕ) : WithTop ℕ∞) ≤ ((M + 1 : ℕ) : WithTop ℕ∞) :=
          Nat.mono_cast hnat
        exact h.contdiff.contDiffAt.of_le this
      have hbridge : iteratedDerivWithin k f
          (Set.Icc (a + (j : ℕ) * ts.h) (a + ((j : ℕ) + 1) * ts.h))
          (a + (j : ℕ) * ts.h)
          = iteratedDeriv k f (a + (j : ℕ) * ts.h) :=
        iteratedDerivWithin_eq_iteratedDeriv (hUDO j) hcontDiffAt hpt_in_Icc
      rw [hbridge, hk_min_eq]
      -- Triangle-inequality bound on |iteratedDeriv k f (a+j·h)|.
      have hgrid := h.grid_acc j ⟨k, by omega⟩
      have htri :
          |iteratedDeriv k f (a + (j : ℕ) * ts.h)|
            ≤ |ts.derivs j ⟨k, by omega⟩| + ts.derivs_err j ⟨k, by omega⟩ := by
        have : |iteratedDeriv k f (a + (j : ℕ) * ts.h)|
            ≤ |iteratedDeriv k f (a + (j : ℕ) * ts.h) - ts.derivs j ⟨k, by omega⟩|
              + |ts.derivs j ⟨k, by omega⟩| := by
          have := abs_sub_abs_le_abs_sub
                    (iteratedDeriv k f (a + (j : ℕ) * ts.h))
                    (ts.derivs j ⟨k, by omega⟩)
          linarith [abs_nonneg (ts.derivs j ⟨k, by omega⟩),
                    abs_nonneg (iteratedDeriv k f (a + (j : ℕ) * ts.h)
                                - ts.derivs j ⟨k, by omega⟩)]
        linarith
      -- Multiply both sides by `ts.h ^ k / k!` (nonneg).
      have hpow_nonneg : (0 : ℝ) ≤ ts.h ^ k := pow_nonneg hh_pos.le _
      have hfact_pos : (0 : ℝ) < (Nat.factorial k : ℝ) := by
        exact_mod_cast Nat.factorial_pos _
      have hcoef_nonneg : (0 : ℝ) ≤ ts.h ^ k / (Nat.factorial k : ℝ) :=
        div_nonneg hpow_nonneg hfact_pos.le
      have := mul_le_mul_of_nonneg_right htri hcoef_nonneg
      -- Rewrite `a*(b/c) = a*b/c`.
      simp only [mul_div_assoc'] at this ⊢
      exact this
    have hsum_le :
        (∑ k ∈ Finset.range (M + 1),
            |iteratedDerivWithin k f
                (Set.Icc (a + (j : ℕ) * ts.h) (a + ((j : ℕ) + 1) * ts.h))
                (a + (j : ℕ) * ts.h)|
              * ts.h ^ k / (Nat.factorial k : ℝ))
        ≤ ∑ k ∈ Finset.range (M + 1),
            (|ts.derivs j ⟨min k M, by omega⟩|
              + ts.derivs_err j ⟨min k M, by omega⟩)
              * ts.h ^ k / (Nat.factorial k : ℝ) :=
      Finset.sum_le_sum hterm_le
    linarith
  -- Apply the headline grid wrapper.
  exact grid_to_supNorm_taylor hN hh_pos hcontDiffOn hF hMbound'

-- ===========================================================================
-- Sanity checks
-- ===========================================================================

#check @TaylorSeriesData
#check @IsTaylorSeriesOf
#check @trigpoly_to_taylor_correct
#check @add_taylorseries_correct
#check @sub_taylorseries_correct
#check @scalar_mul_taylorseries_correct
#check @mul_taylorseries_correct
#check @truncate_order_correct
#check @deriv_series_correct
#check @sup_bound_taylorseries_correct

end WaveCAP
