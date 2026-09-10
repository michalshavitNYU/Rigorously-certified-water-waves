# Mathematical and implementation documentation

## Current Lean status

The [Lean verification status](../Lean/README.md#verification-status) records
the successful 2026-09-09 publication audit: no `sorryAx` dependency, exactly
two certificate-specific numerical axioms in addition to Lean's three
standard logical axioms, and no end-to-end Lean verification of the Crunchy2
interval computation. The two numerical conclusions remain assumptions in
Lean; sorry-free does not mean independently verified numerical data.

## Current standalone note

[`y_hilbert_yu_interval_estimate.tex`](y_hilbert_yu_interval_estimate.tex) is a
self-contained, 11-page treatment of the estimate of `y H[y_u]`. The
[compiled PDF](y_hilbert_yu_interval_estimate.pdf) includes:

- Fourier conventions and the exact candidate.
- Interval derivative records and coefficient-based bounds.
- Leibniz products and the seventh-order Taylor remainder.
- The role of global bound tightening and its lack of effect on this term's
  final additive use.
- Julia code correspondences and the sharded production route.
- Ordinary mathematical statements and proofs corresponding to Lean results.
- The two numerical assumptions of the dedicated certificate.

Compile from this directory using a standard TeX installation:

```sh
pdflatex -interaction=nonstopmode -halt-on-error y_hilbert_yu_interval_estimate.tex
pdflatex -interaction=nonstopmode -halt-on-error y_hilbert_yu_interval_estimate.tex
```

The note was prepared against the original source layout. In its references,
`wave-cap/BabenkoRigorous/src/` now corresponds to `Julia/src/`, and
`wave-cap/WaveCAP/WaveCAP/` corresponds to `Lean/WaveCAP/`.

## Historical source notes

`source_notes/` preserves the upstream tracked implementation notes, audits,
and development plans. Their historical bodies retain their original dates,
paths, and status descriptions. Current-status notices have been added to the
Lean roadmap and proof design. Older claims of seven numerical axioms or
unfinished `sorry` proofs do not describe the current publication result;
illustrative `sorry` stubs in Markdown are not compiled Lean proofs. Use the
current source and the dedicated half-DC audit when interpreting those plans.

Package-specific Julia reports are also retained under `Julia/docs/`.
See [PROVENANCE.md](../PROVENANCE.md) for the source commit and directory mapping.
