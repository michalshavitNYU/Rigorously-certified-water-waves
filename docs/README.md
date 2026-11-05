# Mathematical and implementation documentation

## Current Lean status

The [Lean verification status](../Lean/README.md#verification-status) records
the successful 2026-09-09 publication audit: no `sorryAx` dependency, exactly
two certificate-specific numerical axioms in addition to Lean's three
standard logical axioms, and no end-to-end Lean verification of the Crunchy2
interval computation. The two numerical conclusions remain assumptions in
Lean; sorry-free does not mean independently verified numerical data.

## Historical source notes

`source_notes/` preserves the upstream tracked implementation notes, audits,
and development plans. Their historical bodies retain their original dates,
paths, and status descriptions. Current-status notices have been added to the
Lean roadmap and [proof design](../Lean/lean_design.md). Older claims of seven numerical axioms or
unfinished `sorry` proofs do not describe the current publication result;
illustrative `sorry` stubs in Markdown are not compiled Lean proofs. Use the
current source and the dedicated half-DC audit when interpreting those plans.

Package-specific Julia reports are also retained under `Julia/docs/`.
See [PROVENANCE.md](../PROVENANCE.md) for the source commit and directory mapping.
