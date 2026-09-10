import WaveCAP.FourierTailHilbert

/-!
# Path-A: action of the periodic Hilbert transform on trig polynomials

The fundamental facts:

```
H[1]        = 0
H[cos(k·)]  = sin(k·)      (k ≥ 1)
H[sin(k·)]  = -cos(k·)     (k ≥ 1)
```

are proved as theorems in `FourierTailHilbert.lean` from the concrete
Fourier-series definition of `periodicHilbert`.  Here we expose them via
mathlib-style theorem names and derive the linearity/mode-by-mode action
the Julia code uses through `BabenkoRigorous/src/FourierSeries.jl`
`hilbert(::TrigPoly)`.

The wave-cap residual contains `H[y_u]` (a finite cosine series, evaluated
mode-by-mode via these theorems) and `H[y · y_u]` (a finite trig product,
expanded via product-to-sum identities and again applied mode-by-mode).
-/

namespace WaveCAP

open Real

/-- Mode-by-mode action of `periodicHilbert` on a single cosine. -/
theorem periodicHilbert_cos_apply (k : ℕ) (hk : 1 ≤ k) (u : ℝ) :
    periodicHilbert (fun v => Real.cos ((k : ℝ) * v)) u =
      Real.sin ((k : ℝ) * u) := by
  rw [periodicHilbert_cos k hk]

/-- Mode-by-mode action of `periodicHilbert` on a single sine. -/
theorem periodicHilbert_sin_apply (k : ℕ) (hk : 1 ≤ k) (u : ℝ) :
    periodicHilbert (fun v => Real.sin ((k : ℝ) * v)) u =
      -Real.cos ((k : ℝ) * u) := by
  rw [periodicHilbert_sin k hk]

/-- `H[cos(k·)] + H[sin(k·)] = sin(k·) − cos(k·)` (sanity check for combined
linearity + mode-action theorems). -/
theorem periodicHilbert_cos_add_sin (k : ℕ) (hk : 1 ≤ k) (u : ℝ) :
    periodicHilbert (fun v => Real.cos ((k : ℝ) * v) + Real.sin ((k : ℝ) * v)) u =
      Real.sin ((k : ℝ) * u) - Real.cos ((k : ℝ) * u) := by
  -- Smoothness and periodicity of cos(k·) and sin(k·).
  have hcos_cd : ContDiff ℝ 2 (fun v : ℝ => Real.cos ((k : ℝ) * v)) := by
    have h1 : ContDiff ℝ 2 (fun v : ℝ => ((k : ℝ) * v)) :=
      (contDiff_const.mul contDiff_id)
    exact Real.contDiff_cos.comp h1
  have hsin_cd : ContDiff ℝ 2 (fun v : ℝ => Real.sin ((k : ℝ) * v)) := by
    have h1 : ContDiff ℝ 2 (fun v : ℝ => ((k : ℝ) * v)) :=
      (contDiff_const.mul contDiff_id)
    exact Real.contDiff_sin.comp h1
  have hcos_per : Function.Periodic (fun v : ℝ => Real.cos ((k : ℝ) * v))
      (2 * Real.pi) := by
    intro v
    show Real.cos ((k : ℝ) * (v + 2 * Real.pi)) = Real.cos ((k : ℝ) * v)
    have : (k : ℝ) * (v + 2 * Real.pi) = (k : ℝ) * v + (k : ℤ) * (2 * Real.pi) := by
      push_cast; ring
    rw [this, Real.cos_add_int_mul_two_pi]
  have hsin_per : Function.Periodic (fun v : ℝ => Real.sin ((k : ℝ) * v))
      (2 * Real.pi) := by
    intro v
    show Real.sin ((k : ℝ) * (v + 2 * Real.pi)) = Real.sin ((k : ℝ) * v)
    have : (k : ℝ) * (v + 2 * Real.pi) = (k : ℝ) * v + (k : ℤ) * (2 * Real.pi) := by
      push_cast; ring
    rw [this, Real.sin_add_int_mul_two_pi]
  rw [periodicHilbert_add _ _ hcos_cd hsin_cd hcos_per hsin_per,
      periodicHilbert_cos k hk, periodicHilbert_sin k hk]
  ring

/-- `periodicHilbert` commutes with finite sums of `C²` `2π`-periodic
functions (iterated `periodicHilbert_add`). -/
theorem periodicHilbert_finsetSum {ι : Type*} (s : Finset ι) (f : ι → ℝ → ℝ)
    (hcd : ∀ i ∈ s, ContDiff ℝ 2 (f i))
    (hper : ∀ i ∈ s, Function.Periodic (f i) (2 * Real.pi)) :
    periodicHilbert (fun u => ∑ i ∈ s, f i u) =
      fun u => ∑ i ∈ s, periodicHilbert (f i) u := by
  induction s using Finset.cons_induction with
  | empty =>
    simpa using periodicHilbert_const 0
  | cons i s hi ih =>
    have hcd_tail : ∀ j ∈ s, ContDiff ℝ 2 (f j) :=
      fun j hj => hcd j (Finset.mem_cons_of_mem hj)
    have hper_tail : ∀ j ∈ s, Function.Periodic (f j) (2 * Real.pi) :=
      fun j hj => hper j (Finset.mem_cons_of_mem hj)
    have hsum_cd : ContDiff ℝ 2 (fun u => ∑ j ∈ s, f j u) :=
      ContDiff.sum fun j hj => hcd_tail j hj
    have hsum_per : Function.Periodic (fun u => ∑ j ∈ s, f j u)
        (2 * Real.pi) := fun x =>
      Finset.sum_congr rfl fun j hj => hper_tail j hj x
    have hstep := periodicHilbert_add (f i) (fun u => ∑ j ∈ s, f j u)
      (hcd i (Finset.mem_cons_self ..)) hsum_cd
      (hper i (Finset.mem_cons_self ..)) hsum_per
    funext u
    simp only [Finset.sum_cons]
    calc periodicHilbert (fun u => f i u + ∑ j ∈ s, f j u) u
        = periodicHilbert (f i) u
            + periodicHilbert (fun u => ∑ j ∈ s, f j u) u := by
          rw [hstep]
      _ = periodicHilbert (f i) u + ∑ j ∈ s, periodicHilbert (f j) u := by
          rw [ih hcd_tail hper_tail]

/-- **Bridging lemma (audit 2026-07-03, Q4)**: the hand-expanded cosine sum
`Hyu` in `Definitions.lean` really is the periodic Hilbert transform of the
sine series `yu`. This closes the definitional gap between the residual `R`
as formalized (which uses `Hyu` in five places, including inside `D` and
`G`) and the Babenko residual with `H[y_u]` everywhere. -/
theorem periodicHilbert_yu_eq_Hyu {N : ℕ} (a : Fin N → ℝ) :
    periodicHilbert (yu a) = Hyu a := by
  -- Rewrite `yu` as a finite sum of scaled sines.
  have hyu_eq : yu a =
      fun u => ∑ k : Fin N,
        (-(((k : ℕ) : ℝ) * a k)) * Real.sin (((k : ℕ) : ℝ) * u) := by
    funext u
    unfold yu
    rw [← Finset.sum_neg_distrib]
    exact Finset.sum_congr rfl fun k _ => by ring
  rw [hyu_eq,
    periodicHilbert_finsetSum (ι := Fin N) Finset.univ
      (fun (k : Fin N) u =>
        (-(((k : ℕ) : ℝ) * a k)) * Real.sin (((k : ℕ) : ℝ) * u))
      (fun k _ => contDiff_const.mul
        (Real.contDiff_sin.comp (contDiff_const.mul contDiff_id)))
      (fun k _ x => by
        show (-(((k : ℕ) : ℝ) * a k))
              * Real.sin (((k : ℕ) : ℝ) * (x + 2 * Real.pi))
            = (-(((k : ℕ) : ℝ) * a k)) * Real.sin (((k : ℕ) : ℝ) * x)
        have harg : ((k : ℕ) : ℝ) * (x + 2 * Real.pi)
            = ((k : ℕ) : ℝ) * x + ((k : ℕ) : ℤ) * (2 * Real.pi) := by
          push_cast; ring
        rw [harg, Real.sin_add_int_mul_two_pi])]
  funext u
  unfold Hyu
  refine Finset.sum_congr rfl fun k _ => ?_
  -- Per-mode: `c_k · H[sin(k·)] = k · a k · cos(k·)`, with the `k = 0`
  -- mode vanishing on both sides.
  simp only [periodicHilbert_smul]
  rcases Nat.eq_zero_or_pos (k : ℕ) with hk0 | hk1
  · simp [hk0]
  · rw [periodicHilbert_sin_apply (k : ℕ) hk1 u]
    ring

end WaveCAP

#print axioms WaveCAP.periodicHilbert_cos_apply
#print axioms WaveCAP.periodicHilbert_yu_eq_Hyu
