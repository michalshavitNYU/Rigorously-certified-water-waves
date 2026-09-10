# Source provenance

This publication repository was initialized on September 9, 2026, from the
local checkout of [tristanbuckmaster/wave-cap](https://github.com/tristanbuckmaster/wave-cap).
The source checkout was at commit:

```text
a674ca319e704d891585208c9c1602b655ef1470
```

The initial import preserves the current computational sources, tests, Lean
modules, dependency pins, coefficient inputs, and tracked run archives. The
Julia package retains its existing name, UUID, and author metadata. The Lean
library retains the `WaveCAP` module namespace.

| Original location | Publication location |
| --- | --- |
| `WaveCAP/` | `Lean/` |
| `BabenkoRigorous/` | `Julia/` |
| Tracked `docs/` notes | `docs/source_notes/` |
| Newly prepared `y_hilbert_yu_interval_estimate.tex` | `docs/y_hilbert_yu_interval_estimate.tex` |

The independent mathematical note and its checked PDF were prepared in the
working session after the upstream commit. The original repository history,
dependency caches, unfinished parameter-sweep outputs, and machine-local files
are not imported. The source checkout is left in place.

Original source comments and historical notes may refer to `BabenkoRigorous`,
`WaveCAP`, or a `wave-cap/` parent directory. Use the mapping above when resolving
such references. Existing development workflows embedded in the old Lean
subdirectory were not carried over as publication automation.

## Archived numerical evidence

The two archives in
`Julia/results/phase4_scalar_n150_20260827_2m_20260829/` are preserved unchanged:

```text
4d65357a64a0ccfbf0ed67f8dd0e1970f0ceb549a8cbf647d8490e4ac48abb07  phase4_scalar_n150_20260827_2m_results_20260829.tgz
ee8fe5db1ea3bbd6e37d0888fdadc00c9c9e1d6ce81cb9d46ae3e4db8ffcdf43  wave-cap-half-dc-alignment-20260830.tgz
```

The numerical run archive is also expanded in that directory under `run/`.
Its original file paths and contents are preserved; the archived intermediate
working-directory subset contains only the generated certificate and two
scalar dumps, not the large per-grid shard arrays. The half-DC conversion
archive remains separate from this historical numerical output.

The mathematical note distinguishes the named Lean theorems from their
numerical hypotheses and records the difference between the current formal
tightening construction and Julia's scalar recursion. Those distinctions also
apply to this imported source snapshot.
