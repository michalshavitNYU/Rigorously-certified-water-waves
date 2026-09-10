# Plan: `TightenSupBoundsLemma.lean` (Area C)

**Goal.** Justify Julia's `tighten_sup_bounds!` algorithm: replacing
loose formula-based `sup_bounds[m+1]` values with grid+Lipschitz
recursion that walks down from a (potentially loose) top bound.

**File location:** `WaveCAP/WaveCAP/TightenSupBoundsLemma.lean`.
**Imports:** `WaveCAP.Common`, `WaveCAP.TaylorSeriesCorrectness` (Area A).
**Cross-reference Julia:** `BabenkoRigorous/src/TaylorSeries.jl:259–290`.

---

## 1. Single-step lemma

The core inequality: if `f^(m+1)` is bounded by a real `B`, then for any
grid spacing `h > 0`, the sup-norm of `f^(m)` is bounded by the worst-case
grid value plus `B·h`.

```lean
theorem grid_lipschitz_sup_bound_step
    {f : ℝ → ℝ} {m : ℕ}
    (hf : ContDiff ℝ (m + 1) f)
    {h B : ℝ} (hh : 0 < h) (hB : 0 ≤ B)
    (hBound : ∀ u, |iteratedDeriv (m + 1) f u| ≤ B)
    {a : ℝ} {N : ℕ} (hN : 0 < N)
    (grid_max : ℝ)
    (hgrid : ∀ j : Fin N, |iteratedDeriv m f (a + j.val * h)| ≤ grid_max)
    : ∀ u ∈ Set.Icc a (a + (N - 1) * h + h), |iteratedDeriv m f u| ≤ grid_max + B * h
```

**Proof outline (for later):** Pick `u`, find the nearest grid point
`u_j`, apply MVT: `|f^(m)(u) - f^(m)(u_j)| ≤ B · |u - u_j| ≤ B · h`.
Then `|f^(m)(u)| ≤ |f^(m)(u_j)| + B·h ≤ grid_max + B·h`.

The conservative `h` (rather than the optimal `h/2`) matches Julia's
implementation choice and the existing pattern in `dmin_bound`'s
`d_lower_init` formula.

---

## 2. Recursion lemma

Iterating the single-step lemma from `m = M` down to `m = 0`, with
each level using the just-computed bound at level `m+1` as the input
`B` for level `m`.

```lean
def tightenSupBoundsRecursion
    (top_bound : ℝ)          -- B_{M+1}, possibly very loose (Faà di Bruno etc.)
    (M : ℕ) (h : ℝ)
    (grid_max : Fin (M + 1) → ℝ)  -- grid_max[m] = max_j |f^(m)(u_j)| (the "tight" part)
    : Fin (M + 2) → ℝ
  | ⟨M + 1, _⟩ => top_bound
  | ⟨m, _⟩     =>
      grid_max ⟨m, by omega⟩ + tightenSupBoundsRecursion top_bound M h grid_max ⟨m + 1, by omega⟩ * h

theorem tighten_sup_bounds_correct
    {f : ℝ → ℝ} (M : ℕ) {h : ℝ} (hh : 0 < h)
    (hf : ContDiff ℝ (M + 1) f)
    {top_bound : ℝ} (h_top : ∀ u, |iteratedDeriv (M + 1) f u| ≤ top_bound)
    {a : ℝ} {N : ℕ} (hN : 0 < N)
    {grid_max : Fin (M + 1) → ℝ}
    (h_grid : ∀ (m : Fin (M + 1)) (j : Fin N),
        |iteratedDeriv m.val f (a + j.val * h)| ≤ grid_max m)
    : ∀ (m : Fin (M + 1)) (u : ℝ),
        u ∈ Set.Icc a (a + (N - 1) * h + h) →
          |iteratedDeriv m.val f u|
            ≤ tightenSupBoundsRecursion top_bound M h grid_max ⟨m.val, by omega⟩ := sorry
```

**Proof outline:** Induct on `m` walking down from `M`. Base case
(`m = M`) uses `grid_lipschitz_sup_bound_step` with `B = top_bound`.
Inductive step uses it with `B = tightenSupBoundsRecursion top_bound M h grid_max ⟨m+1, _⟩`,
where the IH gives that this value really is an upper bound on
`iteratedDeriv (m+1) f`.

---

## 3. Bridge to `TaylorSeriesData`

The Julia `tighten_sup_bounds!` operates on a `TaylorSeries` struct;
it reads `ts.derivs[j, m+1]` (giving grid values) and `ts.sup_bounds[m+2]`
(the loose top), and writes new `ts.sup_bounds[m+1]` for `m = 0..M`.

```lean
theorem tighten_sup_bounds_taylorseries_correct
    {M N : ℕ} {ts : TaylorSeriesData M N} {f : ℝ → ℝ} {a : ℝ}
    (h : IsTaylorSeriesOf ts f a)
    (let ts' := TaylorSeriesData.tighten ts)
    : IsTaylorSeriesOf ts' f a := sorry
```

where `TaylorSeriesData.tighten` is the constructive function:

```lean
def TaylorSeriesData.tighten {M N : ℕ} (ts : TaylorSeriesData M N) : TaylorSeriesData M N :=
  let new_sup_bounds (m : Fin (M + 2)) : ℝ :=
    if h : m.val = M + 1 then ts.sup_bounds m
    else
      let m_lt : m.val ≤ M := by omega
      let grid_max : ℝ := ⨆ (j : Fin N), |ts.derivs j ⟨m.val, by omega⟩| + ts.derivs_err j ⟨m.val, by omega⟩
      min (ts.sup_bounds m) (grid_max + new_sup_bounds ⟨m.val + 1, by omega⟩ * ts.h)
  ⟨ts.derivs, ts.derivs_err, new_sup_bounds, ts.h⟩
```

(Note the `min` mirrors the Julia code's "only adopt if strictly tighter
than the existing entry" check.)

**Proof outline.** The `IsTaylorSeriesOf` requirements:
- `derivs`/`derivs_err`/`h` unchanged → those clauses transfer.
- `sup_bounds[M+1]` unchanged → top clause transfers.
- `sup_bounds[m]` for `m ≤ M`: bounded by the `min`, so still ≥ truth.
  Direction-1 (`ts.sup_bounds m`): hypothesis of input `h.sup_acc`.
  Direction-2 (grid+Lipschitz): apply `tighten_sup_bounds_correct`
  combined with `h.grid_acc` and the induction hypothesis on the
  just-computed `new_sup_bounds ⟨m.val + 1, _⟩`.

---

## 4. Counts and effort estimate

- 1 step lemma (~30 lines statement, ~50–80 lines proof).
- 1 recursion definition + correctness theorem (~80 lines statement, ~150 proof).
- 1 bridge to `TaylorSeriesData.tighten` (~60 lines statement, ~150 proof).

**Total drafting:** ~150 lines of Lean signatures + `sorry`.
**Proof effort estimate:** ~400–500 lines.

---

## 5. Dependencies

- `TaylorSeriesCorrectness.lean` (Area A) — for `IsTaylorSeriesOf` and
  `TaylorSeriesData`.
- Mathlib's MVT (`mean_value_theorem_for_iteratedDeriv` or equivalent).
- Mathlib's `Finset.sup'` for grid maxes.

---

## 6. Notes on the `min` semantic

The Julia code only adopts a tighter `new_sb` if `ubound(new_sb) < ubound(existing)`.
The Lean side captures this as `min (ts.sup_bounds m) (grid_max + ...)`. Both
are valid upper bounds on `‖f^(m)‖∞`, so `min` of two valid upper bounds is
still valid. This trivially gives the strictly-tighter property.
