# Plan: `TightDGSupBounds.lean` (Area D)

**Goal.** Justify Julia's `tight_DG_sup_bounds`: algebraic recurrences
for `‖D^(m)‖∞` and `‖G^(m)‖∞` derived from `D² = A² + B²` and `G·D = B`,
bypassing the Faà di Bruno chain that loses ~10¹⁰× for high-mode input.

**File location:** `WaveCAP/WaveCAP/TightDGSupBounds.lean`.
**Imports:** `WaveCAP.Common`, `WaveCAP.Definitions`, `WaveCAP.LeibnizBound`,
`WaveCAP.DLowerBound`.
**Cross-reference Julia:** `BabenkoRigorous/src/TaylorSeries.jl:752–847`,
`FourierSeries.jl:240`.

---

## 1. Foundational identities

### 1.1 `D² = A² + B²` (where `A = y_u`, `B = 1 + H[y_u]`)

This is by definition from `Definitions.lean`:

```lean
namespace WaveCAP

variable {N : ℕ} (a : Fin N → ℝ)

theorem D_squared_eq (u : ℝ) :
    D a u ^ 2 = (yu a u) ^ 2 + (1 + Hyu a u) ^ 2 := by
  unfold D
  rw [Real.sq_sqrt]
  positivity
```

Trivially provable today — included for completeness and for forward
use in the recurrences.

### 1.2 `G · D = 1 + H[y_u]`

```lean
theorem G_mul_D_eq {DLower : ℝ} (hDLower : 0 < DLower)
    (hcert : ∀ u, DLower ≤ D a u)
    (u : ℝ) : G a u * D a u = 1 + Hyu a u := by
  unfold G
  field_simp
  exact (lt_of_lt_of_le hDLower (hcert u)).ne'
```

---

## 2. Leibniz on `D · D = D²` to derive ‖D^(m)‖

The m-th derivative of the LHS `D · D` (by Leibniz) is
`Σ_{k=0..m} C(m,k) · D^(k) · D^(m-k)`. The k=0 and k=m terms are both
`D · D^(m)`. So:

```
2·D·D^(m) = (D²)^(m) − Σ_{k=1..m-1} C(m,k)·D^(k)·D^(m-k)
```

### 2.1 The Leibniz identity (real-valued, point-wise)

```lean
theorem iteratedDeriv_D_sq_Leibniz
    {DLower : ℝ} (hD : 0 < DLower) (hcert : ∀ u, DLower ≤ D a u)
    (m : ℕ) (u : ℝ) :
    iteratedDeriv m (fun v => D a v * D a v) u
      = ∑ k ∈ Finset.range (m + 1),
          (Nat.choose m k : ℝ) *
            iteratedDeriv k (D a) u *
            iteratedDeriv (m - k) (D a) u := by
  -- direct application of `iteratedDeriv_mul`
  sorry
```

(Needs sufficient smoothness of `D a` — follows from `D > 0` and
analyticity of `A`, `B`. The smoothness hypothesis can be derived from
`hcert` + a `ContDiff` claim on `A`, `B`.)

### 2.2 Solving for `D^(m)` and bounding sup-norm

```lean
theorem D_sup_bound_recurrence
    {DLower : ℝ} (hD : 0 < DLower) (hcert : ∀ u, DLower ≤ D a u)
    {D_sup : ℕ → ℝ}
    (h_base : D_sup 0 = Real.sqrt (sSup (Set.range (fun u => D a u * D a u))))
    (h_rec : ∀ m, 0 < m → D_sup m =
        (sSup (Set.range (fun u => |iteratedDeriv m (fun v => D a v * D a v) u|))
         + ∑ k ∈ Finset.range m \ {0}, (Nat.choose m k : ℝ) * D_sup k * D_sup (m - k))
        / (2 * DLower))
    (m : ℕ) :
    ∀ u, |iteratedDeriv m (D a) u| ≤ D_sup m := by
  -- induct on m; base m=0 trivial via h_base
  -- inductive: rearrange `iteratedDeriv_D_sq_Leibniz` and apply IH
  sorry
```

The recurrence matches Julia's `D_sup[m+1] = [‖(D²)^(m)‖∞ + Σ C·D_sup·D_sup] / (2·D_lower)`.

---

## 3. Leibniz on `G · D = B` to derive ‖G^(m)‖

Same shape as §2, applied to `G · D = 1 + Hyu`:

```lean
theorem iteratedDeriv_G_mul_D_Leibniz
    {DLower : ℝ} (hD : 0 < DLower) (hcert : ∀ u, DLower ≤ D a u)
    (m : ℕ) (u : ℝ) :
    iteratedDeriv m (fun v => G a v * D a v) u
      = ∑ k ∈ Finset.range (m + 1),
          (Nat.choose m k : ℝ) *
            iteratedDeriv k (G a) u *
            iteratedDeriv (m - k) (D a) u := sorry

theorem G_sup_bound_recurrence
    {DLower : ℝ} (hD : 0 < DLower) (hcert : ∀ u, DLower ≤ D a u)
    {D_sup G_sup : ℕ → ℝ}
    (h_D_sup_valid : ∀ m u, |iteratedDeriv m (D a) u| ≤ D_sup m)
    (h_B_sup : ∀ m, sSup (Set.range (fun u => |iteratedDeriv m (fun v => 1 + Hyu a v) u|)) ≤ B_sup m)
    (h_base : G_sup 0 = B_sup 0 / DLower)
    (h_rec : ∀ m, 0 < m → G_sup m =
        (B_sup m + ∑ k ∈ Finset.range m, (Nat.choose m k : ℝ) * G_sup k * D_sup (m - k))
        / DLower)
    (m : ℕ) :
    ∀ u, |iteratedDeriv m (G a) u| ≤ G_sup m := by
  -- same induction shape as D_sup_bound_recurrence
  sorry
```

---

## 4. Bridge to the trig-poly sup-bounds

`B_sup m` and `‖(D²)^(m)‖∞` in the recurrences are computed in Julia
via `supnorm_bound` on TrigPolys (`(D²)^(m)`, `B^(m)`). The TrigPoly
arithmetic is exact — `supnorm_bound` is `Σ k^m |c_k|` of the explicit
coefficients.

Lean doesn't need a `TrigPoly` type for this; we just need:

```lean
theorem trigpoly_supnorm_bound
    {N_modes : ℕ} (a b : Fin (N_modes + 1) → ℝ) (m : ℕ) :
    ∀ u, |iteratedDeriv m
        (fun v => ∑ k : Fin (N_modes + 1), a k * Real.cos (k.val * v) + b k * Real.sin (k.val * v)) u|
      ≤ ∑ k : Fin (N_modes + 1), (k.val^m : ℝ) * (|a k| + |b k|) := by
  -- iteratedDeriv of cos(k·) and sin(k·) cycles through factors of k
  -- and then triangle inequality
  sorry
```

This applied to `D² = (A·A + B·B)` with explicit coefficients (computed
in Julia, stated in Lean as the `D²` function with those coefficients)
gives the `‖(D²)^(m)‖∞ ≤ Σ k^m |c_k|` we need.

The coefficients of `D² = A² + B²` themselves come from the
trig-poly product, which Lean handles via the standard product-to-sum
formulas (no `TrigPoly` struct needed; just real-valued identities):

```lean
theorem cos_mul_cos (k j : ℕ) (u : ℝ) :
    Real.cos (k * u) * Real.cos (j * u) =
      (Real.cos ((k + j) * u) + Real.cos ((|k - j|) * u)) / 2 := by
  sorry  -- standard product-to-sum, in Mathlib likely

theorem sin_mul_sin (k j : ℕ) (u : ℝ) :
    Real.sin (k * u) * Real.sin (j * u) =
      (Real.cos ((|k - j|) * u) - Real.cos ((k + j) * u)) / 2 := by
  sorry

theorem sin_mul_cos (k j : ℕ) (u : ℝ) :
    Real.sin (k * u) * Real.cos (j * u) =
      (Real.sin ((k + j) * u) + Real.sin ((k - j) * u)) / 2 := by
  sorry
```

Most of these should already be in Mathlib as `Real.cos_mul_cos`,
`Real.sin_mul_sin`, etc. — confirm before drafting.

---

## 5. Counts and effort estimate

- 2 algebraic identities (`D_squared_eq`, `G_mul_D_eq`) — ~30 lines, trivial.
- 2 Leibniz identities (`iteratedDeriv_D_sq_Leibniz`, `iteratedDeriv_G_mul_D_Leibniz`)
  — ~60 lines each statement, proofs follow from Mathlib's `iteratedDeriv_mul`.
- 2 sup-bound recurrence theorems (`D_sup_bound_recurrence`, `G_sup_bound_recurrence`)
  — ~80 lines each statement, ~200 lines proof.
- 1 trigpoly sup-norm lemma — ~30 lines, ~80 lines proof.
- 3 product-to-sum identities — likely already in Mathlib; otherwise ~30 lines.

**Total drafting:** ~250 lines of Lean signatures + `sorry`.
**Proof effort estimate:** ~600–800 lines.

---

## 6. Dependencies

- `Definitions.lean` (D, G defs).
- `LeibnizBound.lean` (sup-bound shape for Leibniz).
- Mathlib `iteratedDeriv_mul`, `Real.cos_mul_cos`, etc.

---

## 7. Note on `min(chain, algebraic)`

Julia picks `min(chain_bound, algebraic_bound)` for `Gpp_bound_Gu` and
`Gppp_bound_Gu`. On the Lean side this is trivial: a `min` of two valid
upper bounds is still a valid upper bound. No new lemma needed —
captured by transitivity of `≤`.
