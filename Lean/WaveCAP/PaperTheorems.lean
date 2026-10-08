import WaveCAP.PaperCertificateAxioms
import WaveCAP.HalfDCConvention

/-!
# Phase-4 N=150 residual theorem with the Julia half-DC convention

This is a parallel headline module for the two-million-point certificate. It
leaves `Main.lean` and the older certificate untouched.
-/

namespace WaveCAP

open PaperCertificate

/-- Exact headline theorem using the Arb upper endpoint serialized in the
generated certificate. -/
theorem residual_bounded_by_phase4_n150_half_dc_certificate :
    ∀ u, |R a_cert μ_cert β_cert u| ≤ ε₀_cert :=
  cert_residual_le

/-- Paper-friendly corollary: the rigorous residual is below
`1.78 * 10^-13`. -/
theorem residual_bounded_by_phase4_n150_half_dc_decimal :
    ∀ u, |R a_cert μ_cert β_cert u| ≤ (178 : ℝ) / (10 : ℝ) ^ 15 := by
  intro u
  calc
    |R a_cert μ_cert β_cert u| ≤ ε₀_cert :=
      residual_bounded_by_phase4_n150_half_dc_certificate u
    _ ≤ (178 : ℝ) / (10 : ℝ) ^ 15 := by
      norm_num [ε₀_cert, ε₀_cert_ℚ]

/-- Paper-friendly denominator corollary: `D` is uniformly above
`0.3446588`. -/
theorem D_bounded_below_by_phase4_n150_half_dc_decimal :
    ∀ u, (3446588 : ℝ) / (10 : ℝ) ^ 7 ≤ D a_cert u := by
  intro u
  calc
    (3446588 : ℝ) / (10 : ℝ) ^ 7 ≤ DLower_cert := by
      norm_num [DLower_cert, DLower_cert_ℚ]
    _ ≤ D a_cert u := cert_D_ge u

/-- The emitted zeroth Lean coefficient is exactly one half of the source
Float64 DC coefficient. -/
theorem phase4_n150_dc_coefficient_is_source_half :
    a_cert_ℚ ⟨0, by decide⟩ =
      (((-7413908600616953 : ℚ) / 36028797018963968) / 2) := by
  norm_num [a_cert_ℚ]

#print axioms residual_bounded_by_phase4_n150_half_dc_certificate
#print axioms residual_bounded_by_phase4_n150_half_dc_decimal
#print axioms D_bounded_below_by_phase4_n150_half_dc_decimal

end WaveCAP
