# Mathematical and implementation documentation

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
and development plans. They retain their original dates, paths, and status
descriptions. Use the current source and the dedicated half-DC certificate
when interpreting historical plans or claims about earlier milestones.

Package-specific Julia reports are also retained under `Julia/docs/`.
See [PROVENANCE.md](../PROVENANCE.md) for the source commit and directory mapping.
