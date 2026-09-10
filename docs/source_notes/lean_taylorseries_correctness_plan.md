# Plan: `TaylorSeriesCorrectness.lean` (Area A)

**Goal.** Provide the foundational predicate that ties an Arb-backed
`TaylorSeries` to a real-valued function, plus one correctness lemma per
arithmetic operation on `TaylorSeries`. This is the prerequisite for
deriving `cert_residual_le` instead of axiomatizing it.

**File location:** `WaveCAP/WaveCAP/TaylorSeriesCorrectness.lean`.
**Imports:** `WaveCAP.Common`, `WaveCAP.LeibnizBound`, `WaveCAP.FaaDiBrunoBound`.
**Cross-reference Julia:** `BabenkoRigorous/src/TaylorSeries.jl` (struct + ops at
lines ~30, ~252, ~338, ~355, ~392, ~439, ~510, ~599).

---

## 1. The `IsTaylorSeriesOf` predicate

A `TaylorSeries` carries a `derivs[j, m+1]` matrix and a
`sup_bounds[m+1]` vector. We model this in Lean as a structure with
abstract real-valued fields, then state the predicate.

```lean
structure TaylorSeriesData (M : ℕ) (N : ℕ) where
  derivs     : Fin N → Fin (M + 1) → ℝ      -- derivs[j, m] = (Arb midpoint) of f^(m)(u_j)
  derivs_err : Fin N → Fin (M + 1) → ℝ      -- ≥ Arb radius of derivs[j, m]
  sup_bounds : Fin (M + 2) → ℝ              -- sup_bounds[m] ≥ ‖f^(m)‖∞
  h          : ℝ                              -- grid spacing = 2π / N

structure IsTaylorSeriesOf {M N : ℕ} (ts : TaylorSeriesData M N)
                            (f : ℝ → ℝ) (a : ℝ) : Prop where
  contdiff   : ContDiff ℝ (M + 1) f
  grid_acc   : ∀ (j : Fin N) (m : Fin (M + 1)),
                 |iteratedDeriv m.val f (a + j.val * ts.h) - ts.derivs j m|
                   ≤ ts.derivs_err j m
  sup_acc    : ∀ (m : Fin (M + 2)),
                 ‖iteratedDeriv m.val f‖∞ ≤ ts.sup_bounds m
  h_pos      : 0 < ts.h
```

(The `‖·‖∞` notation needs `Periodic` or `BoundedOn` context — for the
periodic-on-[0,2π] setting, use `Periodic.essSup` or define a local
`periodic_sup` notation.)

**Note on Arb modelling.** We don't model `Arb` as such in Lean; we just
model the *post-extraction* values (midpoint as `ℝ`, ubound of radius as
`ℝ`). The "Arb correctness" trust appears only at the very leaves where
specific Arb-primitive correctness is needed — those become atomic
`cert_arb_*` axioms in a separate file.

---

## 2. Per-operation correctness lemmas

For each Julia operation, a lemma of the shape:
"If inputs are `IsTaylorSeriesOf` real functions, the output is
`IsTaylorSeriesOf` the corresponding combination, with the operation's
sup-bound rule."

### 2.1 `trigpoly_to_taylor`

```lean
theorem trigpoly_to_taylor_correct
    {N_modes M N : ℕ}
    (a : Fin (N_modes + 1) → ℝ) (b : Fin (N_modes + 1) → ℝ)
    (ts : TaylorSeriesData M N)
    (h_derivs : ∀ j m, ts.derivs j m =
        ∑ k : Fin (N_modes + 1), (k.val^m.val : ℝ) *
          (a k * Real.cos (k.val * (j.val * ts.h)) + b k * Real.sin (k.val * (j.val * ts.h)))
        * trigSignFactor m.val k.val) -- (concrete derivative formula; spelled out in proof)
    (h_sup   : ∀ m : Fin (M + 2),
        ts.sup_bounds m = ∑ k : Fin (N_modes + 1), (k.val^m.val : ℝ) * (|a k| + |b k|)) :
    IsTaylorSeriesOf ts (fun u => ∑ k : Fin (N_modes + 1),
        a k * Real.cos (k.val * u) + b k * Real.sin (k.val * u)) 0 := sorry
```

The sup-bound part is the triangle inequality on derivative coefficients.

### 2.2 Linear combinations (`+`, `-`, scalar `*`)

```lean
theorem add_taylorseries_correct
    {M N : ℕ} {ts1 ts2 : TaylorSeriesData M N} {f g : ℝ → ℝ} {a : ℝ}
    (h1 : IsTaylorSeriesOf ts1 f a) (h2 : IsTaylorSeriesOf ts2 g a)
    (hh : ts1.h = ts2.h)
    : IsTaylorSeriesOf
        ⟨fun j m => ts1.derivs j m + ts2.derivs j m,
         fun j m => ts1.derivs_err j m + ts2.derivs_err j m,
         fun m => ts1.sup_bounds m + ts2.sup_bounds m,
         ts1.h⟩
        (f + g) a := sorry

theorem scalar_mul_taylorseries_correct
    {M N : ℕ} {ts : TaylorSeriesData M N} {f : ℝ → ℝ} {a : ℝ} (c : ℝ)
    (h : IsTaylorSeriesOf ts f a)
    : IsTaylorSeriesOf
        ⟨fun j m => c * ts.derivs j m,
         fun j m => |c| * ts.derivs_err j m,
         fun m => |c| * ts.sup_bounds m,
         ts.h⟩
        (c • f) a := sorry
```

### 2.3 Leibniz multiplication

```lean
theorem mul_taylorseries_correct
    {M N : ℕ} {ts1 ts2 : TaylorSeriesData M N} {f g : ℝ → ℝ} {a : ℝ}
    (h1 : IsTaylorSeriesOf ts1 f a) (h2 : IsTaylorSeriesOf ts2 g a)
    (hh : ts1.h = ts2.h)
    : IsTaylorSeriesOf
        ⟨fun j m => ∑ k ∈ Finset.range (m.val + 1),
                     Nat.choose m.val k * ts1.derivs j ⟨k, sorry⟩ * ts2.derivs j ⟨m.val - k, sorry⟩,
         /- derivs_err formula -/,
         fun m => ∑ k ∈ Finset.range (m.val + 1),
                   Nat.choose m.val k * ts1.sup_bounds ⟨k, sorry⟩ * ts2.sup_bounds ⟨m.val - k, sorry⟩,
         ts1.h⟩
        (f * g) a := sorry
```

Uses `LeibnizBound.lean`'s `abs_iteratedDeriv_mul_le` for the sup-bound
piece; the per-grid piece is the exact Leibniz identity on real
derivatives.

### 2.4 `truncate_order` and `deriv_series`

```lean
theorem truncate_order_correct
    {M N : ℕ} {ts : TaylorSeriesData M N} {f : ℝ → ℝ} {a : ℝ}
    (h : IsTaylorSeriesOf ts f a)
    (M_new : ℕ) (hM : M_new ≤ M)
    : IsTaylorSeriesOf (TaylorSeriesData.truncate ts M_new hM) f a := sorry

theorem deriv_series_correct
    {M N : ℕ} {ts : TaylorSeriesData M N} {f : ℝ → ℝ} {a : ℝ}
    (h : IsTaylorSeriesOf ts f a) (hM : 0 < M)
    : IsTaylorSeriesOf (TaylorSeriesData.derivShift ts hM)
        (iteratedDeriv 1 f) a := sorry
```

---

## 3. Bridge to `Definitions.lean`

The downstream consumer wants to claim
`IsTaylorSeriesOf ts_R (R a μ β) 0`. The composition will go:

```
trigpoly_to_taylor_correct  ⊢ IsTaylorSeriesOf ts_y_h (WaveCAP.y a)
trigpoly_to_taylor_correct  ⊢ IsTaylorSeriesOf ts_yu_h (WaveCAP.yu a)
trigpoly_to_taylor_correct  ⊢ IsTaylorSeriesOf ts_Hyu (WaveCAP.Hyu a)
mul_taylorseries_correct    ⊢ IsTaylorSeriesOf ts_y_yu_h (fun u => y a u * yu a u)
[fft_hilbert_series_correct ⊢ IsTaylorSeriesOf ts_Hyyu (periodicHilbert (· * yu a ·))] — Phase 4
...
add/scalar_mul              ⊢ IsTaylorSeriesOf ts_R (R a μ β)
sup_bound_correct           ⊢ ‖R a μ β‖∞ ≤ result
```

The final `sup_bound_correct` lemma is essentially what
`GridToSupNormTaylor.lean`'s `grid_to_supNorm_taylor` already does, but
will need re-statement in the `IsTaylorSeriesOf` framework:

```lean
theorem sup_bound_taylorseries_correct
    {M N : ℕ} {ts : TaylorSeriesData M N} {f : ℝ → ℝ} {a : ℝ}
    (h : IsTaylorSeriesOf ts f a)
    (M_bound : ℝ)
    (hM_bound : ∀ j : Fin N,
        (∑ m ∈ Finset.range (M + 1),
          (|ts.derivs j ⟨m, sorry⟩| + ts.derivs_err j ⟨m, sorry⟩)
            * ts.h ^ m / (Nat.factorial m : ℝ))
        + ts.sup_bounds ⟨M + 1, sorry⟩ * ts.h ^ (M + 1) / (Nat.factorial (M + 1) : ℝ)
        ≤ M_bound)
    : ∀ u ∈ Set.Icc a (a + N * ts.h), |f u| ≤ M_bound := sorry
```

---

## 4. Counts and effort estimate

- 1 structure + 1 predicate definition (~50 lines).
- ~7 correctness lemmas (10–60 lines each statement, proofs deferred).
- 1 sup-bound theorem (already analogous to existing `grid_to_supNorm_taylor`).

**Total drafting:** ~300 lines of Lean signatures + `sorry`.
**Proof effort estimate:** ~1500–2500 lines once the predicate is right.

---

## 5. Dependencies

- `LeibnizBound.lean` (for `*` sup-bound proof)
- `FaaDiBrunoBound.lean` (for composition sup-bound proofs in Phase 3)
- Mathlib `iteratedDeriv`, `Periodic`, `ContDiff`.
