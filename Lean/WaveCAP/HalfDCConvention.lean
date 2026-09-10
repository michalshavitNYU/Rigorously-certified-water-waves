import WaveCAP.Definitions

/-!
# Adapter for the Julia half-DC cosine convention

`BabenkoRigorous` stores a cosine series as

`a[0] / 2 + sum_{k >= 1} a[k] cos(k u)`,

whereas `WaveCAP.y` interprets its coefficient vector as

`sum_{k >= 0} a[k] cos(k u)`.

The adapter below changes only the zeroth coefficient before it is supplied
to `WaveCAP.y`. It lets new generated certificates follow Julia's convention
without changing `Definitions.lean` or any existing certificate.
-/

namespace WaveCAP

/-- Convert coefficients stored with Julia's `a[0] / 2` convention to the
coefficient vector expected by `WaveCAP.y`. -/
noncomputable def halfDCCoeffs {N : Nat} (a : Fin N -> Real) : Fin N -> Real := fun k =>
  if k.val = 0 then a k / 2 else a k

/-- The profile represented by a Julia/BabenkoRigorous coefficient vector. -/
noncomputable def yHalfDC {N : Nat} (a : Fin N -> Real) : Real -> Real :=
  y (halfDCCoeffs a)

/-- The Babenko residual represented by a Julia/BabenkoRigorous coefficient
vector. All existing residual definitions and theorems remain unchanged. -/
noncomputable def RHalfDC {N : Nat} (a : Fin N -> Real) (mu beta : Real) :
    Real -> Real :=
  R (halfDCCoeffs a) mu beta

theorem halfDCCoeffs_zero {N : Nat} (a : Fin N -> Real) (hN : 0 < N) :
    halfDCCoeffs a (Fin.mk 0 hN) = a (Fin.mk 0 hN) / 2 := by
  simp [halfDCCoeffs]

theorem halfDCCoeffs_of_ne_zero {N : Nat} (a : Fin N -> Real) (k : Fin N)
    (hk : k.val ≠ 0) :
    halfDCCoeffs a k = a k := by
  simp [halfDCCoeffs, hk]

/-- Pointwise expansion of `yHalfDC`, matching `tp_eval` in
`BabenkoRigorous/src/FourierSeries.jl`. -/
theorem yHalfDC_apply {N : Nat} (a : Fin N -> Real) (u : Real) :
    yHalfDC a u =
      Finset.sum Finset.univ fun k : Fin N =>
        (if k.val = 0 then a k / 2 else a k) * Real.cos ((k : Real) * u) := by
  rfl

/-- Halving the DC coefficient does not change the derivative series. -/
theorem yu_halfDCCoeffs {N : Nat} (a : Fin N -> Real) :
    yu (halfDCCoeffs a) = yu a := by
  funext u
  unfold yu
  congr 1
  apply Finset.sum_congr rfl
  intro k _
  by_cases hk : k.val = 0
  · simp [halfDCCoeffs, hk]
  · simp [halfDCCoeffs, hk]

/-- Halving the DC coefficient does not change `H[y_u]`. -/
theorem Hyu_halfDCCoeffs {N : Nat} (a : Fin N -> Real) :
    Hyu (halfDCCoeffs a) = Hyu a := by
  funext u
  unfold Hyu
  apply Finset.sum_congr rfl
  intro k _
  by_cases hk : k.val = 0
  · simp [halfDCCoeffs, hk]
  · simp [halfDCCoeffs, hk]

/-- The denominator `D` is invariant under the DC-convention conversion. -/
theorem D_halfDCCoeffs {N : Nat} (a : Fin N -> Real) :
    D (halfDCCoeffs a) = D a := by
  funext u
  unfold D
  rw [yu_halfDCCoeffs, Hyu_halfDCCoeffs]

/-- The auxiliary capillary quantity `G` is invariant under the
DC-convention conversion. -/
theorem G_halfDCCoeffs {N : Nat} (a : Fin N -> Real) :
    G (halfDCCoeffs a) = G a := by
  funext u
  unfold G
  rw [Hyu_halfDCCoeffs, D_halfDCCoeffs]

end WaveCAP
