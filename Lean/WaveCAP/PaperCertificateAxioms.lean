import WaveCAP.PaperCertificate
import WaveCAP.ResidualAssemblyCorrectness

/-!
# Numeric trust anchors for the phase-4 N=150 half-DC certificate

This module is parallel to `CertificateAxioms.lean`; it does not replace the
existing certificate chain. The generated certificate supplies exact inputs
and the two exact Arb endpoints from the two-million-point run. The two axioms
below are the complete certificate-specific numerical trust boundary.
-/

namespace WaveCAP.PaperCertificate

/-- Arb certified a positive global lower bound for the Babenko denominator. -/
axiom cert_arb_dmin_bound :
    DLower_cert > 0 ∧
      ∀ u, DLower_cert ≤ WaveCAP.D a_cert u

/-- Arb certified the final Taylor-grid residual bound on one period.

The denominator certificate is an explicit argument because positivity of
`D` is required by the reciprocal and square-root capillary chain. This makes
the headline theorem's dependency on both numerical anchors visible to
`#print axioms`. -/
axiom cert_arb_sup_bound_le_eps0
    (_hD : DLower_cert > 0 ∧
      ∀ u, DLower_cert ≤ WaveCAP.D a_cert u) :
    ∀ u ∈ Set.Icc (0 : ℝ) (2 * Real.pi),
      |WaveCAP.R a_cert μ_cert β_cert u| ≤ ε₀_cert

/-- The generated denominator lower endpoint is valid globally. -/
theorem cert_D_ge :
    ∀ u, DLower_cert ≤ WaveCAP.D a_cert u :=
  cert_arb_dmin_bound.2

/-- The generated residual upper endpoint controls the residual for all real
arguments. The only certificate-specific assumptions are the two Arb anchors
above; periodic extension is proved in Lean. -/
theorem cert_residual_le :
    ∀ u, |WaveCAP.R a_cert μ_cert β_cert u| ≤ ε₀_cert := by
  exact WaveCAP.periodic_sup_extends
    (WaveCAP.R_periodic a_cert μ_cert β_cert)
    ε₀_cert
    (cert_arb_sup_bound_le_eps0 cert_arb_dmin_bound)

#print axioms cert_D_ge
#print axioms cert_residual_le

end WaveCAP.PaperCertificate
