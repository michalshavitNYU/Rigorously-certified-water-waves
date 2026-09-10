-- Umbrella import for the WaveCAP Lean project.
-- Forces `lake build` to compile every leaf module.

import WaveCAP.Common
import WaveCAP.Foundations

-- Definitions and Path-A facts on trig polynomials
import WaveCAP.Definitions
import WaveCAP.HilbertOnTrigPoly
import WaveCAP.HilbertCommutesDeriv
import WaveCAP.HilbertSmoothness  -- closes h_H_contdiff gap

-- Universal analytic lemmas (N1–N5)
import WaveCAP.FaaDiBrunoBound       -- N1
import WaveCAP.LeibnizBound          -- N2
import WaveCAP.FourierTailHilbert    -- N3
import WaveCAP.AliasingErrorBound    -- N4
import WaveCAP.TaylorRemainder       -- N5 (pointwise core, port from CCF)
import WaveCAP.GridToSupNormTaylor   -- N5 (grid-max wrapper)

-- Bounds on G = (1 + H[y_u]) / D
import WaveCAP.DLowerBound

-- Per-algorithm correctness lemmas (Areas A, C, D from
-- docs/lean_roadmap_2026-05-23.md) — all sorry-free.
import WaveCAP.TaylorSeriesCorrectness
import WaveCAP.TightenSupBoundsLemma
import WaveCAP.TightDGSupBounds
import WaveCAP.SeriesCorrectness     -- Area B
import WaveCAP.DminBoundCorrectness  -- Area E
import WaveCAP.FFTHilbertCorrectness  -- Area F
import WaveCAP.ResidualAssemblyCorrectness  -- Area G
import WaveCAP.PipelineSmoothness  -- closes Cat-2 hoists (chain rule, deriv bounds, periodicity)
import WaveCAP.RSmoothness  -- C^k smoothness of the Babenko residual R

-- Certificate (Julia-generated; Lean trust boundary)
import WaveCAP.Certificate
import WaveCAP.CertificateAxioms  -- Phase 6: atomic cert_arb_* axioms

-- Main theorem
import WaveCAP.Main
