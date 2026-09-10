"""
BabenkoRigorous.jl — Rigorous interval-arithmetic package for the Babenko water-wave equation.

Milestone M1: Foundational trig-polynomial arithmetic layer using Arblib.jl.
Milestone M2: Taylor series at grid points with certified sup/inf bounds.
Milestone M3: Certified TaylorSeries for G(u) = (1 + H[y_u]) / D.
Milestone M4: DFT-based periodic Hilbert transform with certified aliasing error.
Milestone M5: Certified upper bound on ‖R‖_∞ for the Babenko residual (β=0).
Milestone M6: Full gravity-capillary residual (β ≠ 0).
Milestone M7: Emit a Lean-readable certificate (.lean) from the pipeline output.
"""
module BabenkoRigorous

using Arblib

include("FourierSeries.jl")
include("TaylorSeries.jl")
include("GDerivatives.jl")
include("FFTHilbert.jl")
include("ResidualBound.jl")
include("Certificate.jl")
include("ShardStore.jl")
include("Sharded.jl")

end  # module
