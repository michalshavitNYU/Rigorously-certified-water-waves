# Lean proof design for the certified water-wave residual

**Publication repository:** Rigorously certified water waves

**Updated:** 2026-10-07

This document describes the implemented formalization accompanying the paper.
It replaces the historical design previously stored at this path. Paths below
use the publication layout: `Lean/` contains the Lean project and `Julia/`
contains the validated numerical code. The Lean namespace remains `WaveCAP`.
The source snapshot and directory mapping are recorded in
[PROVENANCE.md](../../PROVENANCE.md).

## 1. Result and division of work

The computer-assisted proof certifies a fixed, degree-150 cosine polynomial
obtained from the machine-learning calculation. For this candidate, the reported
bounds are

$$
\|\mathcal B_{\mu,\beta}(y_N)\|_{L^\infty(\mathbb R)}
\leq 1.78\times10^{-13},
\qquad
D(u)\geq 0.3446588>0\quad\text{for every }u\in\mathbb R.
$$

Julia/Arblib establishes the numerical inequalities using interval arithmetic,
including bounds for rounding, Fourier aliasing and truncation, and variation
between grid points. The production calculation uses 2,000,000 grid points,
Taylor order 6, and 128-bit Arb precision. Lean supplies a reusable formal
library for the analytic estimates and checks the deductions from the numerical
certificate to the stated global bounds.

The implementation has two complementary parts:

- **Reusable analysis.** Lean defines the Babenko residual and proves the
  Hilbert-transform identities, derivative estimates, Taylor-remainder results,
  data-record contracts, and periodicity arguments used in the proof framework.
- **The publication certificate.** Exact rational data specify the candidate,
  equation parameters, and certified endpoints. Two explicit numerical
  hypotheses record the final Arb conclusions. Lean derives the all-real
  residual estimate, checks the reported decimal bounds by exact arithmetic,
  and verifies the correspondence with the paper's notation.

The certified statement concerns the residual of this explicit polynomial.
An error bound relative to an exact solution, or an existence theorem for a
nearby solution, would require additional stability or inverse estimates.

## 2. Publication entry points

All files in this table are under `Lean/WaveCAP/`.

| File | Role |
| --- | --- |
| [Definitions.lean](../../Lean/WaveCAP/Definitions.lean) | Defines the Fourier Hilbert transform, profile, denominator, capillary quotient, and residual. |
| [CertificatePhase4ScalarN150Phase420260827HalfDC.lean](../../Lean/WaveCAP/CertificatePhase4ScalarN150Phase420260827HalfDC.lean) | Stores the 151 exact coefficients, exact parameters, grid size, Taylor order, and two exact endpoints. |
| [CertificatePhase4ScalarN150Phase420260827HalfDCAxioms.lean](../../Lean/WaveCAP/CertificatePhase4ScalarN150Phase420260827HalfDCAxioms.lean) | Declares the two numerical hypotheses and proves the global residual bound by periodic extension. |
| [MainPhase4ScalarN150Phase420260827HalfDC.lean](../../Lean/WaveCAP/MainPhase4ScalarN150Phase420260827HalfDC.lean) | States the exact result, decimal corollaries, and the certificate's zeroth-coefficient identity. |
| [HalfDCConvention.lean](../../Lean/WaveCAP/HalfDCConvention.lean) | Proves the generic conversion between half-DC and full-DC coefficient conventions. |
| [AuditPhase4ScalarN150HalfDC.lean](../../Lean/WaveCAP/AuditPhase4ScalarN150HalfDC.lean) | Proves the paper-sign bridge and prints the relevant theorem statements and axiom dependencies. |

The older `Main.lean`, `Certificate.lean`, and `CertificateAxioms.lean` remain in
the source snapshot. The dedicated half-DC files above identify the result
reported in the paper.

## 3. Mathematical objects and conventions

### Finite profile and coefficient indexing

The paper and the numerical input use

$$
y_N(u)=\frac{a_0}{2}+\sum_{k=1}^{150}a_k\cos(ku).
$$

Lean's `y` evaluates a full constant coefficient. For a vector
`b : Fin 151 → ℝ`, it represents

$$
y(b;u)=\sum_{k=0}^{150}b_k\cos(ku),
\qquad b_0=a_0/2,\quad b_k=a_k\ (k\geq1).
$$

Thus `N_cert = 151` is the coefficient count; the degree is 150. The generated
publication certificate already stores the converted constant coefficient.
`HalfDCConvention.lean` proves the generic adapter and the invariance of the
profile derivative, its Hilbert transform, `D`, and `G`. The dedicated main
module also proves the exact zeroth-entry identity for this certificate.

### Periodic Hilbert transform

`Definitions.lean` defines the transform by its Fourier multiplier:

$$
H[f](u)=\operatorname{Re}\sum_{n\in\mathbb Z}
  \bigl(-i\,\operatorname{sgn}(n)\bigr)\widehat f_n e^{inu},
\qquad
\widehat f_n=\frac1{2\pi}\int_0^{2\pi}f(v)e^{-inv}\,dv.
$$

The multiplier at zero is zero. On the smooth periodic inputs used in the
analysis, the formal convergence results justify this Fourier representation.
For positive integer modes,

$$
H[\cos(ku)]=\sin(ku),\qquad
H[\sin(ku)]=-\cos(ku),\qquad H[1]=0.
$$

Consequently,

$$
y_u(u)=-\sum_{k=1}^{150}k a_k\sin(ku),
\qquad
H_y(u):=H[y_u](u)=\sum_{k=1}^{150}k a_k\cos(ku).
$$

The Lean names are `yu`, `Hyu`, and `periodicHilbert`. The bridge theorem is
`periodicHilbert_yu_eq_Hyu`.

### Denominator, capillary quotient, and residual

The remaining definitions are

$$
D=\sqrt{y_u^2+(1+H_y)^2},\qquad G=\frac{1+H_y}{D},
$$

and, in Lean's sign convention,

$$
R=\mu H_y-y-yH_y-H[yy_u]
  +\beta\,\partial_u\!\left(\frac{y_u}{D}\right)
  -\beta H[\partial_uG].
$$

For the paper's normalization $g=1$, Lean proves
$R=-\mathcal B_{\mu,\beta}(y_N)$ in
`R_eq_neg_paperBabenkoGOne`, and equality of the absolute values in
`abs_R_eq_abs_paperBabenkoGOne`. Therefore the same residual bound applies to
both conventions.

The equation parameters are the exact rationals

$$
\mu=\frac{1538048895284245}{1125899906842624},\qquad
\beta=\frac{2233821443972785}{9007199254740992}.
$$

They are approximately $1.3660618372350841$ and $0.248004$, respectively.
The latter is the coefficient multiplying the capillary terms; the training
parameter $0.498$ is its square-root parameter, as used in the numerical input
convention. The exact certificate values identify the equation being certified.

## 4. Reusable analytic library

The following modules prove generic results under their stated regularity,
positivity, derivative-bound, and data-accuracy hypotheses. Their statements
can be used for candidates other than the publication profile.

- **Hilbert transform on finite sums.**
  [HilbertOnTrigPoly.lean](../../Lean/WaveCAP/HilbertOnTrigPoly.lean)
  proves the finite-mode calculation (Path A), including `H[yu] = Hyu`.
  [HilbertCommutesDeriv.lean](../../Lean/WaveCAP/HilbertCommutesDeriv.lean)
  proves derivative commutation under its regularity and periodicity hypotheses.
  [FourierTailHilbert.lean](../../Lean/WaveCAP/FourierTailHilbert.lean)
  proves Fourier decay and the estimate
  $|H[f](u)|\leq(\pi^2/3)\|f''\|_\infty$ for the stated smooth periodic inputs.

- **Sampled transform and aliasing (Path B).**
  [AliasingErrorBound.lean](../../Lean/WaveCAP/AliasingErrorBound.lean)
  treats the DFT of sampled values, proves its aliasing estimate, and establishes
  the discrete cotangent-kernel identity. For even $N\geq4$, integer $p\geq2$,
  a $C^p$ periodic input, and $|f^{(p)}|\leq C_p$, the sampled-Hilbert error at
  grid points is bounded by
  $4C_p/((p-1)(N/2-1)^{p-1})$.
  [FFTHilbertCorrectness.lean](../../Lean/WaveCAP/FFTHilbertCorrectness.lean)
  expresses the accuracy contract for the transform and its lift to Taylor
  records. The numerical grid-error hypotheses include the enclosure errors
  associated with a particular transform evaluation.

- **Derivative calculus.**
  [LeibnizBound.lean](../../Lean/WaveCAP/LeibnizBound.lean) and
  [FaaDiBrunoBound.lean](../../Lean/WaveCAP/FaaDiBrunoBound.lean)
  prove product and composition derivative bounds. These supply the analytic
  estimates for polynomial products and the nonlinear capillary operations.

- **Taylor records and continuous bounds.**
  [TaylorSeriesCorrectness.lean](../../Lean/WaveCAP/TaylorSeriesCorrectness.lean)
  defines `TaylorSeriesData` and `IsTaylorSeriesOf`, and proves correctness
  results for operations including addition, subtraction, scalar multiplication,
  multiplication, differentiation, and order truncation, given their data-error
  hypotheses.
  [TaylorRemainder.lean](../../Lean/WaveCAP/TaylorRemainder.lean) proves the
  local remainder estimate, while
  [GridToSupNormTaylor.lean](../../Lean/WaveCAP/GridToSupNormTaylor.lean)
  promotes cell bounds to a bound on the full interval.
  [TightenSupBoundsLemma.lean](../../Lean/WaveCAP/TightenSupBoundsLemma.lean)
  proves the grid-based tightening estimates and preservation of the record
  invariant under its formal tightening construction.

- **Reciprocals, square roots, and the denominator.**
  [SeriesCorrectness.lean](../../Lean/WaveCAP/SeriesCorrectness.lean)
  constructs mathematical Taylor records for reciprocal and square-root
  compositions and proves their semantic correctness, using positivity and
  supplied outer-derivative bounds. These are exact-derivative specification
  records; the Julia interval recurrences implement the numerical calculation.
  [DLowerBound.lean](../../Lean/WaveCAP/DLowerBound.lean) develops consequences
  of a positive denominator bound.
  [DminBoundCorrectness.lean](../../Lean/WaveCAP/DminBoundCorrectness.lean)
  formalizes lower-bound estimates and their refinement from grid information
  under its stated positivity and bound hypotheses.
  [TightDGSupBounds.lean](../../Lean/WaveCAP/TightDGSupBounds.lean)
  provides the derivative estimates for `D` and `G`.

- **Smoothness and residual assembly.**
  [HilbertSmoothness.lean](../../Lean/WaveCAP/HilbertSmoothness.lean),
  [PipelineSmoothness.lean](../../Lean/WaveCAP/PipelineSmoothness.lean), and
  [RSmoothness.lean](../../Lean/WaveCAP/RSmoothness.lean)
  supply smoothness results for the transform and nonlinear residual chain.
  [ResidualAssemblyCorrectness.lean](../../Lean/WaveCAP/ResidualAssemblyCorrectness.lean)
  composes the finite-mode and sampled-transform components under their
  contracts. It also proves `R_periodic` and `periodic_sup_extends`, which
  extend a one-period residual bound to every real argument.

## 5. Taylor data and the grid-to-continuum estimate

For Taylor order $M$ and grid size $N$, a `TaylorSeriesData M N` record contains:

| Field | Meaning |
| --- | --- |
| `derivs j k` | Real approximation to $f^{(k)}(u_j)$ for $0\leq k\leq M$. |
| `derivs_err j k` | Upper bound for the absolute error in that approximation. |
| `sup_bounds k` | Global bound for $|f^{(k)}|$, for $0\leq k\leq M+1$. |
| `h` | Positive grid spacing. |

`IsTaylorSeriesOf ts f a` requires $C^{M+1}$ regularity, the stated grid
accuracy, valid derivative bounds, and positive spacing, with
$u_j=a+jh$. The data are mathematical real-valued records; validity is expressed
by a proposition proved or supplied alongside them.

Writing $c_{jk}$ for `derivs j k`, $e_{jk}$ for `derivs_err j k`, and $F$ for a
bound on $|f^{(M+1)}|$, the resulting estimate has the form

$$
|f(u)|\leq
\max_{0\leq j<N}\left\{
\sum_{k=0}^{M} (|c_{jk}|+e_{jk})\frac{h^k}{k!}
+F\frac{h^{M+1}}{(M+1)!}\right\},
\qquad u\in[a,a+Nh].
$$

The formal grid theorem states this using an upper bound for every cell's
Taylor sum. For one period, $a=0$ and $h=2\pi/N$.
`sup_bound_taylorseries_correct` connects the record invariant with this
continuous bound. The production choice $M=6$ uses derivatives through order
6 at grid points and a seventh-derivative remainder bound.

The exported $\varepsilon_0$ is the final continuous residual upper bound,
already including these corrections. No extra Lipschitz correction is added
to it by the publication theorem.

## 6. Exact certificate and numerical interface

The generated certificate contains exact rational coefficients and parameters,
`N_fft_cert = 2000000`, `M_order_cert = 6`, and the exact endpoints

```text
epsilon0_cert =
  66570650447636094343693612891855023841 /
  374144419156711147060143317175368453031918731001856

DLower_cert =
  30024018999569627600479976468787845960585 /
  87112285931760246646623899502532662132736
```

The source identifier for the first endpoint is `ε₀_cert`, with rational
version `ε₀_cert_ℚ`. The corresponding denominator identifiers are
`DLower_cert` and `DLower_cert_ℚ`. The coefficient and parameter definitions
likewise have rational values and real-valued views. There is no Lipschitz
constant field in this certificate.

The two numerical conclusions supplied by Julia/Arb are declared in the
dedicated axioms module:

1. `cert_arb_dmin_bound` supplies
   $D_{\mathrm{lower}}>0$ and
   $D(u)\geq D_{\mathrm{lower}}$ for every real $u$.
2. `cert_arb_sup_bound_le_eps0`, with that denominator assertion as an explicit
   argument, supplies $|R(u)|\leq\varepsilon_0$ on $[0,2\pi]$.

These two propositions are the numerical trust boundary: Julia/Arb establishes
the enclosures, and Lean checks deductions from them. Their validity as
numerical inequalities is supplied by the validated computation. Exact rational
serialization fixes the values used by Lean without an additional rounding
step.

The reusable library describes the analytic and data-record contracts of the
full calculation. For this publication instance, the interface imports the
final denominator and continuous residual bounds; it does not serialize the
intermediate Taylor records or replay the two-million-point calculation in
Lean. This is the intended division between validated computation and formal
mathematical deduction.

The [archived run evidence](../../Julia/results/phase4_scalar_n150_20260827_2m_20260829/README.md)
contains the input, logs, generated outputs, and archive provenance. See the
[Julia guide](../../Julia/README.md) for numerical reproduction commands.

## 7. The publication proof

The dedicated proof proceeds as follows:

1. Fix `a_cert`, `μ_cert`, and `β_cert` from the exact generated certificate.
2. Obtain the global denominator bound from `cert_arb_dmin_bound`.
3. Apply `cert_arb_sup_bound_le_eps0` with that denominator assertion to obtain
   the residual bound on one period.
4. Use `R_periodic` and `periodic_sup_extends` to obtain the bound for all
   $u\in\mathbb R$.
5. Use exact rational comparisons, checked by `norm_num`, to obtain
   $\varepsilon_0\leq178/10^{15}$ and
   $3446588/10^7\leq D_{\mathrm{lower}}$.
6. Apply the proved sign identity to express the residual bound for the
   paper's Babenko operator.

The main theorem names are:

```text
WaveCAP.residual_bounded_by_phase4_n150_half_dc_certificate
WaveCAP.residual_bounded_by_phase4_n150_half_dc_decimal
WaveCAP.D_bounded_below_by_phase4_n150_half_dc_decimal
WaveCAP.paperBabenkoGOne_bounded_by_phase4_n150_half_dc_certificate
WaveCAP.paperBabenkoGOne_bounded_by_phase4_n150_half_dc_decimal
```

The first three are in the dedicated main module; the last two are in the
audit module.

## 8. Proof status and axiom audit

The implemented proof sources have no `sorry` or `admit` placeholders. In
particular, the aliasing results and Fourier arguments are implemented proofs;
there are no pending Mathlib stubs in the publication proof.

The audit module prints the dependencies of the exact and decimal residual
theorems. They are:

```text
propext
Classical.choice
Quot.sound
WaveCAP.CertificatePhase4ScalarN150Phase420260827HalfDC.cert_arb_dmin_bound
WaveCAP.CertificatePhase4ScalarN150Phase420260827HalfDC.cert_arb_sup_bound_le_eps0
```

The first three are Lean's standard logical axioms. The remaining two are the
numerical interface described above. There is no `sorryAx` dependency. The
denominator theorem uses the denominator hypothesis and the standard logical
axioms; the coefficient-convention and residual-sign identities use only the
standard logical axioms.

For this documentation update, all 32 publication Lean source files and the
three pinned build files were compared byte-for-byte with the source checkout
used for the successful direct audit on 2026-10-07. The publication copy matches
that audited source. This documentation change introduces no changes to the
proofs, certificate data, or numerical computation.

## 9. Reproducing the Lean checks

The pinned toolchain is `leanprover/lean4:v4.29.0-rc6`, with Mathlib
`v4.29.0-rc6` and the dependency resolution in `Lean/lake-manifest.json`.
With Elan available, run from the repository root:

```sh
cd Lean
lake exe cache get
lake build
lake build WaveCAP.MainPhase4ScalarN150Phase420260827HalfDC
lake build WaveCAP.AuditPhase4ScalarN150HalfDC
```

Elan selects the local pinned toolchain. Include the two explicit publication
targets: the default library target alone does not select the dedicated
half-DC certificate audit. To print the audit declarations and dependencies
again after the dependencies are built, run from `Lean/`:

```sh
lake env lean WaveCAP/AuditPhase4ScalarN150HalfDC.lean
```

A successful audit checks the theorem statements and prints the dependency
lists above. Reproducing the archived numerical calculation is a separate
Julia task described in the [Julia guide](../../Julia/README.md).
