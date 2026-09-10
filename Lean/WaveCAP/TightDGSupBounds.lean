import WaveCAP.Common
import WaveCAP.Definitions
import WaveCAP.LeibnizBound
import WaveCAP.DLowerBound

/-!
# Area D — Tight sup-bound recurrences for `D^(m)` and `G^(m)`

This file justifies the Julia routine `tight_DG_sup_bounds` in
`BabenkoRigorous/src/TaylorSeries.jl` (lines ~752–847), which derives sup-norm
bounds on the iterated derivatives of `D` and `G` from the algebraic
identities

```
D(u)² = y_u(u)² + (1 + H[y_u](u))²
G(u) · D(u) = 1 + H[y_u](u)
```

rather than from a Faà di Bruno chain through `√·` and `1/·`. The algebraic
route bypasses a ~10¹⁰× loss that the chain approach incurs for high-mode
inputs, because each Leibniz convolution step preserves polynomial structure
exactly instead of compounding analytic-function compositions.

For each `m`, differentiating `D · D = D²` (resp. `G · D = 1 + H[y_u]`) `m`
times by Leibniz and solving for the highest-order term yields a recurrence
expressing `‖D^(m)‖∞` (resp. `‖G^(m)‖∞`) in terms of:

  - `‖(D²)^(m)‖∞` (resp. `‖(1 + H[y_u])^(m)‖∞`), bounded directly from the
    trig-poly coefficients via `trigpoly_supnorm_bound`;
  - lower-order sup-bounds `‖D^(k)‖∞` (resp. `‖G^(k)‖∞`) for `k < m`,
    supplied by the induction hypothesis;
  - the certified lower bound `DLower ≤ D(u)` from `DLowerBound.lean` /
    `Certificate.lean`.

Cross-reference: `BabenkoRigorous/src/TaylorSeries.jl` `tight_DG_sup_bounds`
(lines 752–847); `BabenkoRigorous/src/FourierSeries.jl:240` (`supnorm_bound`).

Plan document: `docs/lean_tight_dg_sup_bounds_plan.md`.
-/

namespace WaveCAP

open Real

variable {N : ℕ} (a : Fin N → ℝ)

-- ===========================================================================
-- §1. Foundational algebraic identities
-- ===========================================================================

/-- `D(u)² = y_u(u)² + (1 + H[y_u](u))²`. Immediate from the definition of `D`
as `√(y_u² + (1 + H[y_u])²)`.

Julia: `BabenkoRigorous/src/TaylorSeries.jl:787` (`tight_DG_sup_bounds`,
algebraic identity used to derive the `D` recurrence). -/
theorem D_squared_eq (u : ℝ) :
    D a u ^ 2 = (yu a u) ^ 2 + (1 + Hyu a u) ^ 2 := by
  unfold D
  rw [Real.sq_sqrt]
  positivity

/-- `G(u) · D(u) = 1 + H[y_u](u)`. Requires `D(u) ≠ 0`, supplied by the
certified positive lower bound `DLower ≤ D(u)`.

Julia: `BabenkoRigorous/src/TaylorSeries.jl:787` (`tight_DG_sup_bounds`,
algebraic identity used to derive the `G` recurrence). -/
theorem G_mul_D_eq
    {DLower : ℝ} (hDLower : 0 < DLower)
    (hcert : ∀ u, DLower ≤ D a u)
    (u : ℝ) : G a u * D a u = 1 + Hyu a u := by
  unfold G
  have hDpos : 0 < D a u := lt_of_lt_of_le hDLower (hcert u)
  field_simp

-- ===========================================================================
-- §2. Leibniz expansion of `D · D` and the `D^(m)` sup-bound recurrence
-- ===========================================================================

/-- `y_u` is a finite sine series, hence `C^∞`. -/
theorem yu_contDiff {n : ℕ∞} : ContDiff ℝ n (yu a) := by
  unfold yu
  -- `yu = - Σ_k (k * a_k) * sin (k * ·)`.
  apply ContDiff.neg
  apply ContDiff.sum
  intro k _
  have hsin : ContDiff ℝ n (fun u : ℝ => Real.sin ((k : ℝ) * u)) :=
    Real.contDiff_sin.comp (contDiff_const.mul contDiff_id)
  exact contDiff_const.mul hsin

/-- `H[y_u]` is a finite cosine series (Path A), hence `C^∞`. -/
theorem Hyu_contDiff {n : ℕ∞} : ContDiff ℝ n (Hyu a) := by
  unfold Hyu
  apply ContDiff.sum
  intro k _
  have hcos : ContDiff ℝ n (fun u : ℝ => Real.cos ((k : ℝ) * u)) :=
    Real.contDiff_cos.comp (contDiff_const.mul contDiff_id)
  exact contDiff_const.mul hcos

/-- The argument of the `√` in `D` is the polynomial `y_u² + (1+H[y_u])²`,
which is `C^∞` (since `y_u` and `H[y_u]` are). -/
theorem D_sq_arg_contDiff {n : ℕ∞} :
    ContDiff ℝ n (fun u => (yu a u)^2 + (1 + Hyu a u)^2) := by
  have hyu2 : ContDiff ℝ n (fun u => (yu a u)^2) := (yu_contDiff a).pow 2
  have hH : ContDiff ℝ n (fun u => 1 + Hyu a u) :=
    contDiff_const.add (Hyu_contDiff a)
  exact hyu2.add (hH.pow 2)

/-- `D` is `C^∞`, given a positive lower bound `DLower ≤ D`. The hypothesis
ensures `(D a u)^2 = y_u² + (1 + H[y_u])^2 ≥ DLower^2 > 0`, so the argument
of `√` never vanishes, and `Real.sqrt` is smooth on the positive reals. -/
theorem D_contdiff
    {DLower : ℝ} (hD : 0 < DLower) (hcert : ∀ u, DLower ≤ D a u)
    {n : ℕ∞} : ContDiff ℝ n (D a) := by
  -- D u = √((yu)² + (1 + Hyu)²). Show the inner is never zero.
  have h_ne : ∀ u, ((yu a u)^2 + (1 + Hyu a u)^2) ≠ 0 := by
    intro u
    -- The inner equals (D a u)^2 by D_squared_eq.
    have h_eq : (yu a u)^2 + (1 + Hyu a u)^2 = (D a u)^2 := (D_squared_eq a u).symm
    rw [h_eq]
    have hpos : 0 < D a u := lt_of_lt_of_le hD (hcert u)
    exact pow_ne_zero 2 (ne_of_gt hpos)
  -- D a u = √(...) literally by definition.
  have h_def : (D a) = fun u => Real.sqrt ((yu a u)^2 + (1 + Hyu a u)^2) := by
    funext u; rfl
  rw [h_def]
  exact (D_sq_arg_contDiff a).sqrt h_ne

/-- `G = (1 + H[y_u]) / D` is `C^∞`, given a positive lower bound on `D`. -/
theorem G_contdiff
    {DLower : ℝ} (hD : 0 < DLower) (hcert : ∀ u, DLower ≤ D a u)
    {n : ℕ∞} : ContDiff ℝ n (G a) := by
  have h_def : (G a) = fun u => (1 + Hyu a u) / D a u := by
    funext u; rfl
  rw [h_def]
  have hnum : ContDiff ℝ n (fun u => 1 + Hyu a u) :=
    contDiff_const.add (Hyu_contDiff a)
  have hden : ContDiff ℝ n (D a) := D_contdiff a hD hcert
  have hne : ∀ u, D a u ≠ 0 := by
    intro u; exact ne_of_gt (lt_of_lt_of_le hD (hcert u))
  exact hnum.div hden hne

-- ---------------------------------------------------------------------------
-- Periodicity helpers (for deriving BddAbove on derivative ranges)
-- ---------------------------------------------------------------------------

/-- The derivative of a periodic function is periodic with the same period.
This is the key ingredient for showing `iteratedDeriv m (D · D)` has bounded
range. -/
lemma Periodic.deriv_eq {f : ℝ → ℝ} {c : ℝ} (hp : Function.Periodic f c) :
    Function.Periodic (deriv f) c := by
  intro x
  -- By Periodic, (fun y => f (y + c)) = f.
  have h_fun_eq : (fun y => f (y + c)) = f := Function.Periodic.funext hp
  -- (deriv f)(x + c) = deriv f (x + c).
  -- deriv_comp_add_const : deriv (fun y => f (y + c)) x = deriv f (x + c).
  -- So deriv f (x + c) = deriv (fun y => f (y + c)) x. Rewrite by h_fun_eq.
  change deriv f (x + c) = deriv f x
  rw [← deriv_comp_add_const f c, h_fun_eq]

/-- Iterated derivative of a periodic function is periodic. -/
lemma Periodic.iteratedDeriv {f : ℝ → ℝ} {c : ℝ} (hp : Function.Periodic f c)
    (m : ℕ) : Function.Periodic (_root_.iteratedDeriv m f) c := by
  induction m with
  | zero =>
    simp only [iteratedDeriv_zero]; exact hp
  | succ n IH =>
    rw [iteratedDeriv_succ]
    exact Periodic.deriv_eq IH

/-- Helper: `cos(k * (x + 2π)) = cos(k * x)` for natural `k`. -/
lemma cos_nat_mul_periodic_eq (k : ℕ) (x : ℝ) :
    Real.cos ((k : ℝ) * (x + 2 * Real.pi)) = Real.cos ((k : ℝ) * x) := by
  have heq : (k : ℝ) * (x + 2 * Real.pi) = (k : ℝ) * x + k * (2 * Real.pi) := by ring
  rw [heq]
  exact Real.cos_periodic.nat_mul k ((k : ℝ) * x)

/-- Helper: `sin(k * (x + 2π)) = sin(k * x)` for natural `k`. -/
lemma sin_nat_mul_periodic_eq (k : ℕ) (x : ℝ) :
    Real.sin ((k : ℝ) * (x + 2 * Real.pi)) = Real.sin ((k : ℝ) * x) := by
  have heq : (k : ℝ) * (x + 2 * Real.pi) = (k : ℝ) * x + k * (2 * Real.pi) := by ring
  rw [heq]
  exact Real.sin_periodic.nat_mul k ((k : ℝ) * x)

/-- `y_u` is `2π`-periodic. -/
lemma yu_periodic : Function.Periodic (yu a) (2 * Real.pi) := by
  intro x
  unfold yu
  congr 1
  apply Finset.sum_congr rfl
  intro k _
  rw [sin_nat_mul_periodic_eq (k : ℕ) x]

/-- `H[y_u]` is `2π`-periodic. -/
lemma Hyu_periodic : Function.Periodic (Hyu a) (2 * Real.pi) := by
  intro x
  unfold Hyu
  apply Finset.sum_congr rfl
  intro k _
  rw [cos_nat_mul_periodic_eq (k : ℕ) x]

/-- `D` is `2π`-periodic. -/
lemma D_periodic : Function.Periodic (D a) (2 * Real.pi) := by
  intro x
  unfold D
  rw [yu_periodic a x, Hyu_periodic a x]

/-- `D · D` is `2π`-periodic. -/
lemma D_mul_D_periodic : Function.Periodic (fun u => D a u * D a u) (2 * Real.pi) := by
  intro x
  change D a (x + 2 * Real.pi) * D a (x + 2 * Real.pi) = D a x * D a x
  rw [D_periodic a x]

/-- For any natural `m`, `iteratedDeriv m (D · D)` is `2π`-periodic, hence its
range over `ℝ` is bounded, hence `BddAbove` on the absolute value. -/
lemma bddAbove_abs_iteratedDeriv_DD
    {DLower : ℝ} (hD : 0 < DLower) (hcert : ∀ u, DLower ≤ D a u)
    (m : ℕ) :
    BddAbove (Set.range (fun u => |iteratedDeriv m (fun v => D a v * D a v) u|)) := by
  -- iteratedDeriv m (D·D) is periodic.
  have hper : Function.Periodic (iteratedDeriv m (fun v => D a v * D a v))
                (2 * Real.pi) :=
    Periodic.iteratedDeriv (D_mul_D_periodic a) m
  -- D · D is C^m.
  have h_DD_cd : ContDiff ℝ (m : ℕ) (fun v => D a v * D a v) := by
    have hDcd : ContDiff ℝ (m : ℕ) (D a) := D_contdiff a hD hcert
    exact hDcd.mul hDcd
  -- iteratedDeriv m (D · D) is continuous.
  have h_cont : Continuous (iteratedDeriv m (fun v => D a v * D a v)) :=
    h_DD_cd.continuous_iteratedDeriv' m
  -- 2π ≠ 0.
  have hpi : (2 * Real.pi) ≠ 0 := by
    have := Real.pi_pos; linarith
  -- Continuous + periodic ⇒ compact range.
  have h_compact : IsCompact (Set.range (iteratedDeriv m (fun v => D a v * D a v))) :=
    hper.compact_of_continuous hpi h_cont
  -- abs ∘ iteratedDeriv m (D·D) is continuous; hence its range is the image
  -- of the compact range under abs, which is compact.
  have h_compact_abs : IsCompact
        (Set.range (fun u => |iteratedDeriv m (fun v => D a v * D a v) u|)) := by
    have h_range_eq : Set.range (fun u => |iteratedDeriv m (fun v => D a v * D a v) u|)
        = (fun x => |x|) '' (Set.range (iteratedDeriv m (fun v => D a v * D a v))) := by
      ext y
      constructor
      · rintro ⟨u, hu⟩
        exact ⟨iteratedDeriv m (fun v => D a v * D a v) u, ⟨u, rfl⟩, hu⟩
      · rintro ⟨x, ⟨u, rfl⟩, rfl⟩
        exact ⟨u, rfl⟩
    rw [h_range_eq]
    exact h_compact.image _root_.continuous_abs
  exact h_compact_abs.bddAbove

/-- The range of `D · D` is `BddAbove`. -/
lemma bddAbove_DD
    {DLower : ℝ} (hD : 0 < DLower) (hcert : ∀ u, DLower ≤ D a u) :
    BddAbove (Set.range (fun u => D a u * D a u)) := by
  have hper : Function.Periodic (fun u => D a u * D a u) (2 * Real.pi) :=
    D_mul_D_periodic a
  have h_cont : Continuous (fun u => D a u * D a u) := by
    have hDcd : ContDiff ℝ (1 : ℕ∞) (D a) := D_contdiff a hD hcert
    exact (hDcd.mul hDcd).continuous
  have hpi : (2 * Real.pi) ≠ 0 := by
    have := Real.pi_pos; linarith
  exact (hper.isBounded_of_continuous hpi h_cont).bddAbove

/-- `1 + H[y_u]` is `2π`-periodic. -/
lemma one_add_Hyu_periodic : Function.Periodic (fun v => 1 + Hyu a v) (2 * Real.pi) := by
  intro x
  show 1 + Hyu a (x + 2 * Real.pi) = 1 + Hyu a x
  rw [Hyu_periodic a x]

/-- For any natural `m`, `iteratedDeriv m (1 + H[y_u])` has `BddAbove` range. -/
lemma bddAbove_abs_iteratedDeriv_one_add_Hyu (m : ℕ) :
    BddAbove (Set.range (fun u => |iteratedDeriv m (fun v => 1 + Hyu a v) u|)) := by
  -- iteratedDeriv m (1 + Hyu) is periodic.
  have hper : Function.Periodic (iteratedDeriv m (fun v => 1 + Hyu a v))
                (2 * Real.pi) :=
    Periodic.iteratedDeriv (one_add_Hyu_periodic a) m
  -- 1 + Hyu is C^m.
  have h_cd : ContDiff ℝ (m : ℕ) (fun v => 1 + Hyu a v) :=
    contDiff_const.add (Hyu_contDiff a)
  -- iteratedDeriv m (1 + Hyu) is continuous.
  have h_cont : Continuous (iteratedDeriv m (fun v => 1 + Hyu a v)) :=
    h_cd.continuous_iteratedDeriv' m
  -- 2π ≠ 0.
  have hpi : (2 * Real.pi) ≠ 0 := by
    have := Real.pi_pos; linarith
  -- Continuous + periodic ⇒ compact range.
  have h_compact : IsCompact (Set.range (iteratedDeriv m (fun v => 1 + Hyu a v))) :=
    hper.compact_of_continuous hpi h_cont
  -- abs preserves compactness.
  have h_compact_abs : IsCompact
        (Set.range (fun u => |iteratedDeriv m (fun v => 1 + Hyu a v) u|)) := by
    have h_range_eq : Set.range (fun u => |iteratedDeriv m (fun v => 1 + Hyu a v) u|)
        = (fun x => |x|) '' (Set.range (iteratedDeriv m (fun v => 1 + Hyu a v))) := by
      ext y
      constructor
      · rintro ⟨u, hu⟩
        exact ⟨iteratedDeriv m (fun v => 1 + Hyu a v) u, ⟨u, rfl⟩, hu⟩
      · rintro ⟨x, ⟨u, rfl⟩, rfl⟩
        exact ⟨u, rfl⟩
    rw [h_range_eq]
    exact h_compact.image _root_.continuous_abs
  exact h_compact_abs.bddAbove

/-- The `m`-th derivative of `D · D` expanded by the Leibniz rule:
`(D·D)^(m) = Σ_{k=0..m} C(m,k) · D^(k) · D^(m-k)`. Requires smoothness of `D`,
which follows from `D(u) ≥ DLower > 0` together with analyticity of
`y_u` and `1 + H[y_u]`.

Julia: `BabenkoRigorous/src/TaylorSeries.jl:787` (`tight_DG_sup_bounds`,
Leibniz step on `D² = D · D`). -/
theorem iteratedDeriv_D_sq_Leibniz
    {DLower : ℝ} (hD : 0 < DLower) (hcert : ∀ u, DLower ≤ D a u)
    (m : ℕ) (u : ℝ) :
    iteratedDeriv m (fun v => D a v * D a v) u
      = ∑ k ∈ Finset.range (m + 1),
          (Nat.choose m k : ℝ) *
            iteratedDeriv k (D a) u *
            iteratedDeriv (m - k) (D a) u := by
  have hD_cd : ContDiffAt ℝ (m : ℕ∞) (D a) u := (D_contdiff a hD hcert).contDiffAt
  exact iteratedDeriv_fun_mul hD_cd hD_cd

/-- Sup-norm bound recurrence for `‖D^(m)‖∞`.

Rearranging `iteratedDeriv_D_sq_Leibniz` and isolating the two `k = 0` and
`k = m` terms (each equal to `D · D^(m)`) gives

```
2 · D(u) · D^(m)(u) = (D²)^(m)(u) − Σ_{k=1..m-1} C(m,k) · D^(k)(u) · D^(m-k)(u).
```

Taking absolute values, applying the triangle inequality and the certified
lower bound `D(u) ≥ DLower`, and inducting on `m`, yields the recurrence
implemented in Julia.

Julia: `BabenkoRigorous/src/TaylorSeries.jl:787` (`tight_DG_sup_bounds`,
`D_sup[m+1] = [‖(D²)^(m)‖∞ + Σ C·D_sup·D_sup] / (2·D_lower)`). -/
theorem D_sup_bound_recurrence
    {DLower : ℝ} (hD : 0 < DLower) (hcert : ∀ u, DLower ≤ D a u)
    {D_sup : ℕ → ℝ}
    (h_base : D_sup 0 = Real.sqrt (sSup (Set.range (fun u => D a u * D a u))))
    (h_rec : ∀ m, 0 < m → D_sup m =
        (sSup (Set.range (fun u => |iteratedDeriv m (fun v => D a v * D a v) u|))
         + ∑ k ∈ Finset.range m \ {0},
             (Nat.choose m k : ℝ) * D_sup k * D_sup (m - k))
        / (2 * DLower))
    (m : ℕ) :
    ∀ u, |iteratedDeriv m (D a) u| ≤ D_sup m := by
  -- Strong induction on m.
  induction m using Nat.strong_induction_on with
  | _ m IH =>
  intro u
  rcases Nat.eq_zero_or_pos m with hm0 | hm_pos
  · -- Base case: m = 0.
    subst hm0
    -- Goal: |iteratedDeriv 0 (D a) u| ≤ D_sup 0.
    rw [iteratedDeriv_zero]
    have hDu_pos : 0 < D a u := lt_of_lt_of_le hD (hcert u)
    have hDu_nn : 0 ≤ D a u := le_of_lt hDu_pos
    -- |D a u| = D a u.
    rw [abs_of_nonneg hDu_nn]
    -- D a u = √(D a u * D a u) since D a u ≥ 0.
    have hD_sq : D a u = Real.sqrt (D a u * D a u) := by
      rw [show D a u * D a u = (D a u)^2 by ring]
      rw [Real.sqrt_sq hDu_nn]
    rw [hD_sq]
    -- √(D a u * D a u) ≤ √(sSup (range D·D)) iff D a u * D a u ≤ sSup ...
    rw [h_base]
    apply Real.sqrt_le_sqrt
    -- D a u * D a u ∈ range (D·D), apply le_csSup with BddAbove.
    exact le_csSup (bddAbove_DD a hD hcert) ⟨u, rfl⟩
  · -- Inductive step: m > 0.
    -- IH: for all k < m, ∀ u', |iteratedDeriv k (D a) u'| ≤ D_sup k.
    have h_DD_eq := iteratedDeriv_D_sq_Leibniz a hD hcert m u
    -- Split the sum into k=0, k=m, and 1..m-1.
    -- iteratedDeriv 0 (D a) u = D a u, iteratedDeriv (m-0) (D a) u = iteratedDeriv m (D a) u.
    -- k=0: choose m 0 = 1; term = 1 * D a u * iteratedDeriv m (D a) u = D a u * iteratedDeriv m (D a) u.
    -- k=m: choose m m = 1; term = 1 * iteratedDeriv m (D a) u * D a u = iteratedDeriv m (D a) u * D a u.
    -- Sum of k=0 and k=m = 2 * D a u * iteratedDeriv m (D a) u.
    have hm_in : m ∈ Finset.range (m + 1) := by
      rw [Finset.mem_range]; omega
    have h0_in : (0 : ℕ) ∈ Finset.range (m + 1) := by
      rw [Finset.mem_range]; omega
    have hm_ne_0 : m ≠ 0 := Nat.pos_iff_ne_zero.mp hm_pos
    -- Set:
    --   T := Σ_{k ∈ range (m+1)} C(m,k) · D^(k) u · D^(m-k) u
    --   T0 := the k=0 term = D u · D^(m) u
    --   Tm := the k=m term = D^(m) u · D u
    --   Tmid := Σ_{k ∈ range m \ {0}} C(m,k) · D^(k) u · D^(m-k) u
    -- So T = T0 + Tm + Tmid = 2·D u · D^(m) u + Tmid.
    set T : ℝ :=
      ∑ k ∈ Finset.range (m + 1),
        (Nat.choose m k : ℝ) * iteratedDeriv k (D a) u *
          iteratedDeriv (m - k) (D a) u with hT_def
    -- Rewrite T by pulling out k=0 and k=m.
    -- range m \ {0} = (range m).erase 0
    have h_erase_eq : Finset.range m \ {0} = (Finset.range m).erase 0 := by
      ext x
      simp [Finset.mem_erase, Finset.mem_sdiff]
      tauto
    have h_0_in_range : (0 : ℕ) ∈ Finset.range m := by
      rw [Finset.mem_range]; exact hm_pos
    have h_T_split : T = 2 * D a u * iteratedDeriv m (D a) u
        + ∑ k ∈ Finset.range m \ {0},
            (Nat.choose m k : ℝ) * iteratedDeriv k (D a) u *
              iteratedDeriv (m - k) (D a) u := by
      -- Rewrite range m \ {0} = (range m).erase 0 on the RHS first.
      rw [h_erase_eq]
      -- Now manipulate T: range (m+1) = insert m (range m) (because m ∉ range m).
      have h_range_succ_eq :
          Finset.range (m + 1) = insert m (Finset.range m) :=
        (Finset.range_succ).symm ▸ rfl
      have h_m_not_mem : m ∉ Finset.range m := by simp
      have h_choose_m : (Nat.choose m m : ℝ) = 1 := by
        rw [Nat.choose_self]; simp
      have h_mm0 : m - m = 0 := Nat.sub_self m
      have h_choose_0 : (Nat.choose m 0 : ℝ) = 1 := by
        rw [Nat.choose_zero_right]; simp
      have h_m0 : m - 0 = m := Nat.sub_zero m
      -- Pull out the k = m term from T explicitly.
      have hT_step1 : T = (Nat.choose m m : ℝ) * iteratedDeriv m (D a) u *
                            iteratedDeriv (m - m) (D a) u
                        + ∑ k ∈ Finset.range m,
                          (Nat.choose m k : ℝ) * iteratedDeriv k (D a) u *
                            iteratedDeriv (m - k) (D a) u := by
        rw [hT_def, h_range_succ_eq, Finset.sum_insert h_m_not_mem]
      rw [hT_step1]
      -- Pull out the k = 0 term from the sum over range m.
      have h_range_m_split :
          (Finset.range m) = insert 0 ((Finset.range m).erase 0) := by
        exact (Finset.insert_erase h_0_in_range).symm
      have h_sum_range_m_split :
          ∑ k ∈ Finset.range m,
            (Nat.choose m k : ℝ) * iteratedDeriv k (D a) u *
              iteratedDeriv (m - k) (D a) u
          = (Nat.choose m 0 : ℝ) * iteratedDeriv 0 (D a) u *
              iteratedDeriv (m - 0) (D a) u
            + ∑ k ∈ (Finset.range m).erase 0,
                (Nat.choose m k : ℝ) * iteratedDeriv k (D a) u *
                  iteratedDeriv (m - k) (D a) u := by
        nth_rewrite 1 [h_range_m_split]
        rw [Finset.sum_insert (Finset.notMem_erase _ _)]
      rw [h_sum_range_m_split]
      -- Now simplify the boundary terms.
      rw [h_choose_m, h_mm0, h_choose_0, h_m0]
      simp only [iteratedDeriv_zero]
      ring
    -- Now we have: iteratedDeriv m (D·D) u = T = 2·D u·D^(m) u + Tmid.
    -- Rearrange: 2·D u·D^(m) u = iteratedDeriv m (D·D) u - Tmid.
    -- |2·D u·D^(m) u| ≤ |iteratedDeriv m (D·D) u| + |Tmid|.
    have h_eq_T : iteratedDeriv m (fun v => D a v * D a v) u = T := h_DD_eq
    have h_iso : 2 * D a u * iteratedDeriv m (D a) u
        = iteratedDeriv m (fun v => D a v * D a v) u
            - ∑ k ∈ Finset.range m \ {0},
                (Nat.choose m k : ℝ) * iteratedDeriv k (D a) u *
                  iteratedDeriv (m - k) (D a) u := by
      rw [h_eq_T, h_T_split]; ring
    -- Take absolute values.
    have h_2D_pos : 0 < 2 * D a u := by
      have hDu_pos : 0 < D a u := lt_of_lt_of_le hD (hcert u)
      linarith
    have h_2D_nn : 0 ≤ 2 * D a u := le_of_lt h_2D_pos
    have h_2DLower_pos : 0 < 2 * DLower := by linarith
    -- |2·D u·D^(m) u| = 2·D u·|D^(m) u| since 2·D u ≥ 0.
    have h_abs_lhs : |2 * D a u * iteratedDeriv m (D a) u|
        = 2 * D a u * |iteratedDeriv m (D a) u| := by
      rw [abs_mul, abs_of_nonneg h_2D_nn]
    -- Bound the middle sum.
    -- For each k ∈ range m \ {0}: k ∈ [1, m-1], so k < m and m - k < m. Apply IH.
    have h_mid_bd : ∀ k ∈ Finset.range m \ {0},
        |(Nat.choose m k : ℝ) * iteratedDeriv k (D a) u *
          iteratedDeriv (m - k) (D a) u|
        ≤ (Nat.choose m k : ℝ) * D_sup k * D_sup (m - k) := by
      intro k hk_mem
      rw [Finset.mem_sdiff, Finset.mem_range, Finset.mem_singleton] at hk_mem
      obtain ⟨hk_lt, hk_ne⟩ := hk_mem
      have hk_pos : 0 < k := Nat.pos_of_ne_zero hk_ne
      have hmk_lt : m - k < m := by omega
      have h_IH_k : |iteratedDeriv k (D a) u| ≤ D_sup k := IH k hk_lt u
      have h_IH_mk : |iteratedDeriv (m - k) (D a) u| ≤ D_sup (m - k) := IH (m - k) hmk_lt u
      have h_choose_nn : (0 : ℝ) ≤ (Nat.choose m k : ℝ) := Nat.cast_nonneg _
      have h_Dsupk_nn : 0 ≤ D_sup k := le_trans (abs_nonneg _) h_IH_k
      have h_Dsupmk_nn : 0 ≤ D_sup (m - k) := le_trans (abs_nonneg _) h_IH_mk
      calc |(Nat.choose m k : ℝ) * iteratedDeriv k (D a) u *
            iteratedDeriv (m - k) (D a) u|
          = (Nat.choose m k : ℝ) * |iteratedDeriv k (D a) u| *
            |iteratedDeriv (m - k) (D a) u| := by
            rw [abs_mul, abs_mul, abs_of_nonneg h_choose_nn]
        _ ≤ (Nat.choose m k : ℝ) * D_sup k * D_sup (m - k) := by
            gcongr
    -- Triangle inequality on |Σ ...| ≤ Σ |...|.
    have h_mid_abs_sum :
        |∑ k ∈ Finset.range m \ {0},
            (Nat.choose m k : ℝ) * iteratedDeriv k (D a) u *
              iteratedDeriv (m - k) (D a) u|
        ≤ ∑ k ∈ Finset.range m \ {0},
              (Nat.choose m k : ℝ) * D_sup k * D_sup (m - k) :=
      le_trans (Finset.abs_sum_le_sum_abs _ _) (Finset.sum_le_sum h_mid_bd)
    -- Sum bound: |LHS - RHS| ≤ |LHS| + |RHS|.
    have h_combined :
        2 * D a u * |iteratedDeriv m (D a) u|
        ≤ |iteratedDeriv m (fun v => D a v * D a v) u|
            + ∑ k ∈ Finset.range m \ {0},
                (Nat.choose m k : ℝ) * D_sup k * D_sup (m - k) := by
      calc 2 * D a u * |iteratedDeriv m (D a) u|
          = |2 * D a u * iteratedDeriv m (D a) u| := h_abs_lhs.symm
        _ = |iteratedDeriv m (fun v => D a v * D a v) u
                - ∑ k ∈ Finset.range m \ {0},
                    (Nat.choose m k : ℝ) * iteratedDeriv k (D a) u *
                      iteratedDeriv (m - k) (D a) u| := by rw [h_iso]
        _ ≤ |iteratedDeriv m (fun v => D a v * D a v) u|
            + |∑ k ∈ Finset.range m \ {0},
                (Nat.choose m k : ℝ) * iteratedDeriv k (D a) u *
                  iteratedDeriv (m - k) (D a) u| :=
            abs_sub _ _
        _ ≤ |iteratedDeriv m (fun v => D a v * D a v) u|
            + ∑ k ∈ Finset.range m \ {0},
                (Nat.choose m k : ℝ) * D_sup k * D_sup (m - k) := by
            linarith [h_mid_abs_sum]
    -- Now bound |iteratedDeriv m (D·D) u| ≤ sSup (range ...) using BddAbove.
    have h_iter_sSup : |iteratedDeriv m (fun v => D a v * D a v) u|
        ≤ sSup (Set.range (fun u' => |iteratedDeriv m (fun v => D a v * D a v) u'|)) :=
      le_csSup (bddAbove_abs_iteratedDeriv_DD a hD hcert m) ⟨u, rfl⟩
    -- Combine: 2·D u·|D^(m) u| ≤ sSup + ∑.
    have h_2D_abs_bd :
        2 * D a u * |iteratedDeriv m (D a) u|
        ≤ sSup (Set.range (fun u' => |iteratedDeriv m (fun v => D a v * D a v) u'|))
          + ∑ k ∈ Finset.range m \ {0},
              (Nat.choose m k : ℝ) * D_sup k * D_sup (m - k) := by
      linarith
    -- Use 2 * DLower ≤ 2 * D u to get 2·DLower·|D^(m) u| ≤ ... .
    have h_DLower_le_D : 2 * DLower ≤ 2 * D a u := by
      have := hcert u; linarith
    have h_2DLower_abs_bd :
        2 * DLower * |iteratedDeriv m (D a) u|
        ≤ sSup (Set.range (fun u' => |iteratedDeriv m (fun v => D a v * D a v) u'|))
          + ∑ k ∈ Finset.range m \ {0},
              (Nat.choose m k : ℝ) * D_sup k * D_sup (m - k) := by
      have habs_nn : 0 ≤ |iteratedDeriv m (D a) u| := abs_nonneg _
      have h1 : 2 * DLower * |iteratedDeriv m (D a) u|
                  ≤ 2 * D a u * |iteratedDeriv m (D a) u| :=
        mul_le_mul_of_nonneg_right h_DLower_le_D habs_nn
      linarith
    -- Divide by 2 * DLower > 0 to get the bound.
    rw [h_rec m hm_pos]
    rw [le_div_iff₀ h_2DLower_pos]
    -- Goal: |D^(m) u| * (2 * DLower) ≤ sSup + ∑.
    -- From h_2DLower_abs_bd: 2 * DLower * |D^(m) u| ≤ sSup + ∑.
    have hreorder : |iteratedDeriv m (D a) u| * (2 * DLower)
                  = 2 * DLower * |iteratedDeriv m (D a) u| := by ring
    rw [hreorder]
    exact h_2DLower_abs_bd

-- ===========================================================================
-- §3. Leibniz expansion of `G · D` and the `G^(m)` sup-bound recurrence
-- ===========================================================================

/-- The `m`-th derivative of `G · D` expanded by the Leibniz rule:
`(G·D)^(m) = Σ_{k=0..m} C(m,k) · G^(k) · D^(m-k)`. Smoothness of `G` follows
from `D ≥ DLower > 0` and smoothness of `1 + H[y_u]`.

Julia: `BabenkoRigorous/src/TaylorSeries.jl:787` (`tight_DG_sup_bounds`,
Leibniz step on `G · D = 1 + H[y_u]`). -/
theorem iteratedDeriv_G_mul_D_Leibniz
    {DLower : ℝ} (hD : 0 < DLower) (hcert : ∀ u, DLower ≤ D a u)
    (m : ℕ) (u : ℝ) :
    iteratedDeriv m (fun v => G a v * D a v) u
      = ∑ k ∈ Finset.range (m + 1),
          (Nat.choose m k : ℝ) *
            iteratedDeriv k (G a) u *
            iteratedDeriv (m - k) (D a) u := by
  have hG_cd : ContDiffAt ℝ (m : ℕ∞) (G a) u := (G_contdiff a hD hcert).contDiffAt
  have hD_cd : ContDiffAt ℝ (m : ℕ∞) (D a) u := (D_contdiff a hD hcert).contDiffAt
  exact iteratedDeriv_fun_mul hG_cd hD_cd

/-- Sup-norm bound recurrence for `‖G^(m)‖∞`.

From `G · D = 1 + H[y_u]` and `iteratedDeriv_G_mul_D_Leibniz`, isolating the
`k = m` term (equal to `G^(m) · D`) gives

```
D(u) · G^(m)(u) = (1 + H[y_u])^(m)(u) − Σ_{k=0..m-1} C(m,k) · G^(k)(u) · D^(m-k)(u).
```

Combined with `D(u) ≥ DLower`, `|G^(k)(u)| ≤ G_sup k` for `k < m` (IH), and
`|D^(m-k)(u)| ≤ D_sup (m-k)` (from `D_sup_bound_recurrence`), this yields the
Julia recurrence.

Julia: `BabenkoRigorous/src/TaylorSeries.jl:787` (`tight_DG_sup_bounds`,
`G_sup[m+1] = [‖B^(m)‖∞ + Σ C·G_sup·D_sup] / D_lower`). -/
theorem G_sup_bound_recurrence
    {DLower : ℝ} (hD : 0 < DLower) (hcert : ∀ u, DLower ≤ D a u)
    {D_sup G_sup B_sup : ℕ → ℝ}
    (h_D_sup_valid : ∀ m u, |iteratedDeriv m (D a) u| ≤ D_sup m)
    (h_B_sup : ∀ m,
        sSup (Set.range (fun u => |iteratedDeriv m (fun v => 1 + Hyu a v) u|))
          ≤ B_sup m)
    (h_base : G_sup 0 = B_sup 0 / DLower)
    (h_rec : ∀ m, 0 < m → G_sup m =
        (B_sup m + ∑ k ∈ Finset.range m,
            (Nat.choose m k : ℝ) * G_sup k * D_sup (m - k))
        / DLower)
    (m : ℕ) :
    ∀ u, |iteratedDeriv m (G a) u| ≤ G_sup m := by
  induction m using Nat.strong_induction_on with
  | _ m IH =>
  intro u
  -- First the key identity: G u * D u = 1 + Hyu u everywhere (D > 0).
  have hG_D_eq : ∀ v, G a v * D a v = 1 + Hyu a v :=
    fun v => G_mul_D_eq a hD hcert v
  -- The two functions u ↦ G u * D u and u ↦ 1 + Hyu u are pointwise-equal,
  -- so their iterated derivatives match.
  have h_iter_eq : iteratedDeriv m (fun v => G a v * D a v) u
                = iteratedDeriv m (fun v => 1 + Hyu a v) u := by
    have h_funeq : (fun v => G a v * D a v) = (fun v => 1 + Hyu a v) := by
      funext v; exact hG_D_eq v
    rw [h_funeq]
  -- Pointwise bound on |1+Hyu|^(m) via BddAbove + le_csSup → h_B_sup.
  have h_B_bound : |iteratedDeriv m (fun v => 1 + Hyu a v) u| ≤ B_sup m := by
    have hle_sSup : |iteratedDeriv m (fun v => 1 + Hyu a v) u|
        ≤ sSup (Set.range (fun u' => |iteratedDeriv m (fun v => 1 + Hyu a v) u'|)) :=
      le_csSup (bddAbove_abs_iteratedDeriv_one_add_Hyu a m) ⟨u, rfl⟩
    exact le_trans hle_sSup (h_B_sup m)
  rcases Nat.eq_zero_or_pos m with hm0 | hm_pos
  · -- Base case m = 0: |G u| ≤ G_sup 0 = B_sup 0 / DLower.
    subst hm0
    rw [iteratedDeriv_zero]
    -- G u = (1 + Hyu u) / D u.
    have h_Gu_eq : G a u = (1 + Hyu a u) / D a u := G_eq_div u
    rw [h_Gu_eq, abs_div]
    have hDu_pos : 0 < D a u := lt_of_lt_of_le hD (hcert u)
    rw [abs_of_pos hDu_pos]
    rw [h_base]
    -- Goal: |1 + Hyu u| / D u ≤ B_sup 0 / DLower.
    -- Chain: |1 + Hyu u| ≤ B_sup 0 (from h_B_bound for m=0, where iteratedDeriv 0 (1+Hyu) = 1+Hyu).
    have h_abs_nn : 0 ≤ |1 + Hyu a u| := abs_nonneg _
    -- h_B_bound (after iteratedDeriv_zero simp) says |1+Hyu u| ≤ B_sup 0.
    have h_one_Hyu_le : |1 + Hyu a u| ≤ B_sup 0 := by
      have := h_B_bound
      rw [iteratedDeriv_zero] at this
      exact this
    -- D u ≥ DLower > 0 ⇒ 1/D u ≤ 1/DLower.
    have h_div_le : |1 + Hyu a u| / D a u ≤ |1 + Hyu a u| / DLower := by
      apply div_le_div_of_nonneg_left h_abs_nn hD (hcert u)
    -- |1+Hyu u| / DLower ≤ B_sup 0 / DLower (numerator bound).
    have h_div_le2 : |1 + Hyu a u| / DLower ≤ B_sup 0 / DLower := by
      apply div_le_div_of_nonneg_right h_one_Hyu_le (le_of_lt hD)
    exact le_trans h_div_le h_div_le2
  · -- Inductive step m > 0:
    -- iteratedDeriv m (G·D) u = (1 + Hyu)^(m) u.
    -- By Leibniz: (G·D)^(m) u = Σ_{k=0..m} C(m,k) · G^(k) u · D^(m-k) u
    --   = G^(m) u · D u + Σ_{k=0..m-1} C(m,k) · G^(k) u · D^(m-k) u.
    -- So G^(m) u · D u = (1+Hyu)^(m) u - Σ_{k=0..m-1} ...
    have h_leib := iteratedDeriv_G_mul_D_Leibniz a hD hcert m u
    -- Split the m+1-sum: range (m+1) = insert m (range m). The m-summand is
    -- C(m,m) · G^(m) u · D^0 u = G^(m) u · D u.
    have h_range_succ_eq :
        Finset.range (m + 1) = insert m (Finset.range m) :=
      (Finset.range_succ).symm ▸ rfl
    have h_m_not_mem : m ∉ Finset.range m := by simp
    have h_choose_m : (Nat.choose m m : ℝ) = 1 := by
      rw [Nat.choose_self]; simp
    have h_mm0 : m - m = 0 := Nat.sub_self m
    have h_split_sum :
        ∑ k ∈ Finset.range (m + 1),
          (Nat.choose m k : ℝ) * iteratedDeriv k (G a) u *
            iteratedDeriv (m - k) (D a) u
        = iteratedDeriv m (G a) u * D a u
          + ∑ k ∈ Finset.range m,
              (Nat.choose m k : ℝ) * iteratedDeriv k (G a) u *
                iteratedDeriv (m - k) (D a) u := by
      rw [h_range_succ_eq, Finset.sum_insert h_m_not_mem, h_choose_m, h_mm0]
      simp only [iteratedDeriv_zero]
      ring
    -- Hence: iteratedDeriv m (G·D) u = G^(m) u · D u + Σ_{k=0..m-1}.
    have h_iso : iteratedDeriv m (G a) u * D a u
        = iteratedDeriv m (fun v => 1 + Hyu a v) u
            - ∑ k ∈ Finset.range m,
                (Nat.choose m k : ℝ) * iteratedDeriv k (G a) u *
                  iteratedDeriv (m - k) (D a) u := by
      rw [← h_iter_eq, h_leib, h_split_sum]
      ring
    -- Bound the middle sum.
    have h_mid_bd : ∀ k ∈ Finset.range m,
        |(Nat.choose m k : ℝ) * iteratedDeriv k (G a) u *
          iteratedDeriv (m - k) (D a) u|
        ≤ (Nat.choose m k : ℝ) * G_sup k * D_sup (m - k) := by
      intro k hk_mem
      rw [Finset.mem_range] at hk_mem
      have h_IH_k : |iteratedDeriv k (G a) u| ≤ G_sup k := IH k hk_mem u
      have h_IH_mk : |iteratedDeriv (m - k) (D a) u| ≤ D_sup (m - k) :=
        h_D_sup_valid (m - k) u
      have h_choose_nn : (0 : ℝ) ≤ (Nat.choose m k : ℝ) := Nat.cast_nonneg _
      have h_Gsupk_nn : 0 ≤ G_sup k := le_trans (abs_nonneg _) h_IH_k
      have h_Dsupmk_nn : 0 ≤ D_sup (m - k) := le_trans (abs_nonneg _) h_IH_mk
      calc |(Nat.choose m k : ℝ) * iteratedDeriv k (G a) u *
            iteratedDeriv (m - k) (D a) u|
          = (Nat.choose m k : ℝ) * |iteratedDeriv k (G a) u| *
            |iteratedDeriv (m - k) (D a) u| := by
            rw [abs_mul, abs_mul, abs_of_nonneg h_choose_nn]
        _ ≤ (Nat.choose m k : ℝ) * G_sup k * D_sup (m - k) := by gcongr
    have h_mid_sum_abs :
        |∑ k ∈ Finset.range m,
            (Nat.choose m k : ℝ) * iteratedDeriv k (G a) u *
              iteratedDeriv (m - k) (D a) u|
        ≤ ∑ k ∈ Finset.range m,
            (Nat.choose m k : ℝ) * G_sup k * D_sup (m - k) :=
      le_trans (Finset.abs_sum_le_sum_abs _ _) (Finset.sum_le_sum h_mid_bd)
    -- Take absolute values: |G^(m) u| · D u = |G^(m) u · D u| ≤ ...
    have hDu_pos : 0 < D a u := lt_of_lt_of_le hD (hcert u)
    have hDu_nn : 0 ≤ D a u := le_of_lt hDu_pos
    have h_abs_iso : |iteratedDeriv m (G a) u| * D a u
        = |iteratedDeriv m (G a) u * D a u| := by
      rw [abs_mul, abs_of_nonneg hDu_nn]
    have h_combined :
        |iteratedDeriv m (G a) u| * D a u
        ≤ |iteratedDeriv m (fun v => 1 + Hyu a v) u|
          + ∑ k ∈ Finset.range m,
              (Nat.choose m k : ℝ) * G_sup k * D_sup (m - k) := by
      calc |iteratedDeriv m (G a) u| * D a u
          = |iteratedDeriv m (G a) u * D a u| := h_abs_iso
        _ = |iteratedDeriv m (fun v => 1 + Hyu a v) u
              - ∑ k ∈ Finset.range m,
                  (Nat.choose m k : ℝ) * iteratedDeriv k (G a) u *
                    iteratedDeriv (m - k) (D a) u| := by rw [h_iso]
        _ ≤ |iteratedDeriv m (fun v => 1 + Hyu a v) u|
            + |∑ k ∈ Finset.range m,
                  (Nat.choose m k : ℝ) * iteratedDeriv k (G a) u *
                    iteratedDeriv (m - k) (D a) u| := abs_sub _ _
        _ ≤ |iteratedDeriv m (fun v => 1 + Hyu a v) u|
            + ∑ k ∈ Finset.range m,
                (Nat.choose m k : ℝ) * G_sup k * D_sup (m - k) := by
            linarith [h_mid_sum_abs]
    -- Replace |(1+Hyu)^(m) u| by B_sup m via h_B_bound.
    have h_B_combined :
        |iteratedDeriv m (G a) u| * D a u
        ≤ B_sup m + ∑ k ∈ Finset.range m,
              (Nat.choose m k : ℝ) * G_sup k * D_sup (m - k) := by
      linarith [h_B_bound]
    -- Use D u ≥ DLower to get |G^(m) u| * DLower ≤ ... .
    have h_abs_nn : 0 ≤ |iteratedDeriv m (G a) u| := abs_nonneg _
    have h_DL_le : |iteratedDeriv m (G a) u| * DLower
                 ≤ |iteratedDeriv m (G a) u| * D a u :=
      mul_le_mul_of_nonneg_left (hcert u) h_abs_nn
    have h_DL_combined :
        |iteratedDeriv m (G a) u| * DLower
        ≤ B_sup m + ∑ k ∈ Finset.range m,
              (Nat.choose m k : ℝ) * G_sup k * D_sup (m - k) := by
      linarith
    -- Divide by DLower > 0.
    rw [h_rec m hm_pos]
    rw [le_div_iff₀ hD]
    exact h_DL_combined

-- ===========================================================================
-- §4. Bridge: trig-poly sup-norm bound
-- ===========================================================================

/-- Sup-norm bound on the `m`-th derivative of a finite cosine/sine trig
polynomial.

For `f(u) = Σ_k (a_k · cos(k·u) + b_k · sin(k·u))`, the `m`-th derivative
of each mode contributes a factor of `k^m` (with sign / cos↔sin swap depending
on `m mod 4`), so by the triangle inequality
`|f^(m)(u)| ≤ Σ_k k^m · (|a_k| + |b_k|)` uniformly in `u`.

This is the Lean counterpart of Julia's `supnorm_bound` for `TrigPoly`
objects, used to bound the `‖(D²)^(m)‖∞` and `‖(1 + H[y_u])^(m)‖∞` terms in
the recurrences above.

Julia: `BabenkoRigorous/src/FourierSeries.jl:240` (`supnorm_bound`);
`BabenkoRigorous/src/TaylorSeries.jl:787` (`tight_DG_sup_bounds`, called for
`(D²)^(m)` and `B^(m)`). -/
theorem trigpoly_supnorm_bound
    {N_modes : ℕ} (a b : Fin (N_modes + 1) → ℝ) (m : ℕ) :
    ∀ u, |iteratedDeriv m
        (fun v => ∑ k : Fin (N_modes + 1),
            (a k * Real.cos ((k.val : ℝ) * v) + b k * Real.sin ((k.val : ℝ) * v))) u|
      ≤ ∑ k : Fin (N_modes + 1), ((k.val : ℝ) ^ m) * (|a k| + |b k|) := by
  intro u
  -- Smoothness of each mode k: cos(k·) and sin(k·) compose linearly with cos/sin.
  have h_cos_comp : ∀ k : Fin (N_modes + 1),
      ContDiff ℝ (m : ℕ∞) (fun v : ℝ => Real.cos ((k.val : ℝ) * v)) := by
    intro k
    exact Real.contDiff_cos.comp (contDiff_const.mul contDiff_id)
  have h_sin_comp : ∀ k : Fin (N_modes + 1),
      ContDiff ℝ (m : ℕ∞) (fun v : ℝ => Real.sin ((k.val : ℝ) * v)) := by
    intro k
    exact Real.contDiff_sin.comp (contDiff_const.mul contDiff_id)
  -- Per-mode ContDiffAt at u.
  have h_mode_cd : ∀ k : Fin (N_modes + 1),
      ContDiffAt ℝ (m : ℕ∞)
        (fun v => a k * Real.cos ((k.val : ℝ) * v) +
                  b k * Real.sin ((k.val : ℝ) * v)) u := by
    intro k
    exact ((contDiff_const.mul (h_cos_comp k)).add
            (contDiff_const.mul (h_sin_comp k))).contDiffAt
  -- Step 1: distribute iteratedDeriv through the sum.
  rw [iteratedDeriv_fun_sum (fun k _ => h_mode_cd k)]
  -- Step 2: each summand is a_k * cos(k·) + b_k * sin(k·).
  have h_term : ∀ k : Fin (N_modes + 1),
      iteratedDeriv m
        (fun v => a k * Real.cos ((k.val : ℝ) * v) +
                  b k * Real.sin ((k.val : ℝ) * v)) u
      = a k * ((k.val : ℝ) ^ m * iteratedDeriv m Real.cos ((k.val : ℝ) * u)) +
        b k * ((k.val : ℝ) ^ m * iteratedDeriv m Real.sin ((k.val : ℝ) * u)) := by
    intro k
    have hcos_cd : ContDiffAt ℝ (m : ℕ∞)
        (fun v : ℝ => Real.cos ((k.val : ℝ) * v)) u := (h_cos_comp k).contDiffAt
    have hsin_cd : ContDiffAt ℝ (m : ℕ∞)
        (fun v : ℝ => Real.sin ((k.val : ℝ) * v)) u := (h_sin_comp k).contDiffAt
    -- Split the addition.
    rw [iteratedDeriv_fun_add
          (((contDiff_const.mul (h_cos_comp k))).contDiffAt)
          (((contDiff_const.mul (h_sin_comp k))).contDiffAt)]
    -- Pull out the constants a k and b k.
    rw [iteratedDeriv_const_mul (a k) hcos_cd,
        iteratedDeriv_const_mul (b k) hsin_cd]
    -- iteratedDeriv m (cos (k * ·)) u = k^m * iteratedDeriv m cos (k*u)
    rw [show (fun v => Real.cos ((k.val : ℝ) * v))
            = (fun v => Real.cos ((k.val : ℝ) * v)) from rfl]
    rw [iteratedDeriv_comp_const_mul Real.contDiff_cos (k.val : ℝ),
        iteratedDeriv_comp_const_mul Real.contDiff_sin (k.val : ℝ)]
  -- Apply h_term to each summand.
  simp_rw [h_term]
  -- Step 3: triangle inequality |∑ ... | ≤ ∑ | ... |.
  refine le_trans (Finset.abs_sum_le_sum_abs _ _) ?_
  -- Step 4: bound each term by k^m · (|a k| + |b k|).
  apply Finset.sum_le_sum
  intro k _
  have h_cos_abs : |iteratedDeriv m Real.cos ((k.val : ℝ) * u)| ≤ 1 :=
    Real.abs_iteratedDeriv_cos_le_one m _
  have h_sin_abs : |iteratedDeriv m Real.sin ((k.val : ℝ) * u)| ≤ 1 :=
    Real.abs_iteratedDeriv_sin_le_one m _
  have h_km_nn : (0 : ℝ) ≤ (k.val : ℝ) ^ m := pow_nonneg (Nat.cast_nonneg _) m
  -- Bound |a_k * (k^m * cos^(m)(ku)) + b_k * (k^m * sin^(m)(ku))| via triangle.
  calc |a k * ((k.val : ℝ) ^ m * iteratedDeriv m Real.cos ((k.val : ℝ) * u)) +
        b k * ((k.val : ℝ) ^ m * iteratedDeriv m Real.sin ((k.val : ℝ) * u))|
      ≤ |a k * ((k.val : ℝ) ^ m * iteratedDeriv m Real.cos ((k.val : ℝ) * u))|
          + |b k * ((k.val : ℝ) ^ m * iteratedDeriv m Real.sin ((k.val : ℝ) * u))| :=
        abs_add_le _ _
    _ = |a k| * ((k.val : ℝ) ^ m * |iteratedDeriv m Real.cos ((k.val : ℝ) * u)|)
          + |b k| * ((k.val : ℝ) ^ m * |iteratedDeriv m Real.sin ((k.val : ℝ) * u)|) := by
        rw [abs_mul, abs_mul, abs_mul, abs_mul, abs_of_nonneg h_km_nn]
    _ ≤ |a k| * ((k.val : ℝ) ^ m * 1) + |b k| * ((k.val : ℝ) ^ m * 1) := by
        gcongr
    _ = (k.val : ℝ) ^ m * (|a k| + |b k|) := by ring

end WaveCAP

-- ===========================================================================
-- Headline sanity checks
-- ===========================================================================

#check @WaveCAP.D_squared_eq
#check @WaveCAP.G_mul_D_eq
#check @WaveCAP.iteratedDeriv_D_sq_Leibniz
#check @WaveCAP.D_sup_bound_recurrence
#check @WaveCAP.iteratedDeriv_G_mul_D_Leibniz
#check @WaveCAP.G_sup_bound_recurrence
#check @WaveCAP.trigpoly_supnorm_bound
