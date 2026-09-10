import Mathlib

/-!
# Common imports and notation for WaveCAP

We use the `Mathlib` umbrella so individual files don't have to chase import
paths that change between Mathlib versions. The Mathlib build cache makes this
essentially free; the only cost is incremental rebuild time on our own files,
which is negligible at this size.

Patterned after `ccf-lean-poc/CCFLeanPOC/Common.lean`.
-/

namespace WaveCAP

open Real intervalIntegral MeasureTheory
open scoped Real ENNReal NNReal Topology

end WaveCAP
