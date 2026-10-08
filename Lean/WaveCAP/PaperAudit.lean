import WaveCAP.PaperTheorems

/-!
# Audit of the phase-4 N=150 half-DC certificate

This module changes no definitions and adds no assumptions. It records the
certificate-facing declarations whose statements and axiom dependencies are
reported in the paper and supplementary material.
-/

namespace WaveCAP

open PaperCertificate

/-- The manuscript's Babenko operator in the normalization `g = 1`, with
`mu = c^2` and `beta = kappa`. -/
noncomputable def paperBabenkoGOne {N : ℕ} (a : Fin N → ℝ) (mu beta u : ℝ) : ℝ :=
  -mu * Hyu a u
    + y a u
    + y a u * Hyu a u
    + periodicHilbert (fun v => y a v * yu a v) u
    - beta * deriv (fun v => yu a v / D a v) u
    + beta * periodicHilbert (deriv (G a)) u

/-- Lean's residual uses the opposite sign from the manuscript's Babenko
operator. -/
theorem R_eq_neg_paperBabenkoGOne {N : ℕ} (a : Fin N → ℝ) (mu beta u : ℝ) :
    R a mu beta u = -paperBabenkoGOne a mu beta u := by
  unfold R paperBabenkoGOne
  ring

/-- The sign convention has no effect on the certified absolute residual. -/
theorem abs_R_eq_abs_paperBabenkoGOne {N : ℕ}
    (a : Fin N → ℝ) (mu beta u : ℝ) :
    |R a mu beta u| = |paperBabenkoGOne a mu beta u| := by
  rw [R_eq_neg_paperBabenkoGOne]
  exact abs_neg _

/-- Exact headline certificate stated for the manuscript's Babenko operator. -/
theorem paperBabenkoGOne_bounded_by_phase4_n150_half_dc_certificate :
    ∀ u, |paperBabenkoGOne a_cert μ_cert β_cert u| ≤ ε₀_cert := by
  intro u
  rw [← abs_R_eq_abs_paperBabenkoGOne]
  exact residual_bounded_by_phase4_n150_half_dc_certificate u

/-- Decimal corollary stated for the manuscript's Babenko operator. -/
theorem paperBabenkoGOne_bounded_by_phase4_n150_half_dc_decimal :
    ∀ u, |paperBabenkoGOne a_cert μ_cert β_cert u| ≤
      (178 : ℝ) / (10 : ℝ) ^ 15 := by
  intro u
  rw [← abs_R_eq_abs_paperBabenkoGOne]
  exact residual_bounded_by_phase4_n150_half_dc_decimal u

#check @halfDCCoeffs_zero
#check @halfDCCoeffs_of_ne_zero
#check @yHalfDC_apply
#check @yu_halfDCCoeffs
#check @Hyu_halfDCCoeffs
#check @D_halfDCCoeffs
#check @G_halfDCCoeffs

#check cert_arb_dmin_bound
#check cert_arb_sup_bound_le_eps0
#check cert_D_ge
#check cert_residual_le

#check residual_bounded_by_phase4_n150_half_dc_certificate
#check residual_bounded_by_phase4_n150_half_dc_decimal
#check D_bounded_below_by_phase4_n150_half_dc_decimal
#check phase4_n150_dc_coefficient_is_source_half
#check R_eq_neg_paperBabenkoGOne
#check abs_R_eq_abs_paperBabenkoGOne
#check paperBabenkoGOne_bounded_by_phase4_n150_half_dc_certificate
#check paperBabenkoGOne_bounded_by_phase4_n150_half_dc_decimal

#print axioms halfDCCoeffs_zero
#print axioms halfDCCoeffs_of_ne_zero
#print axioms yHalfDC_apply
#print axioms yu_halfDCCoeffs
#print axioms Hyu_halfDCCoeffs
#print axioms D_halfDCCoeffs
#print axioms G_halfDCCoeffs
#print axioms cert_D_ge
#print axioms cert_residual_le
#print axioms residual_bounded_by_phase4_n150_half_dc_certificate
#print axioms residual_bounded_by_phase4_n150_half_dc_decimal
#print axioms D_bounded_below_by_phase4_n150_half_dc_decimal
#print axioms phase4_n150_dc_coefficient_is_source_half
#print axioms R_eq_neg_paperBabenkoGOne
#print axioms abs_R_eq_abs_paperBabenkoGOne
#print axioms paperBabenkoGOne_bounded_by_phase4_n150_half_dc_certificate
#print axioms paperBabenkoGOne_bounded_by_phase4_n150_half_dc_decimal

end WaveCAP
