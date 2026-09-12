# I=4 Fusion Breeding-Blanket Verification Instance

Small-instance ($I=4$ sectors, $K=4$ candidate material recipes, $4^4=256$
total configurations) toroidal breeding-blanket neutronics model, built as
the brute-force-checkable verification case for RQ3 of an NSF Engineering
Research Initiation proposal on surrogate-screened combinatorial search for
PDE-constrained design problems.

See [`report.pdf`](report.pdf) (source: [`report.tex`](report.tex)) for the
full technical report: software stack, geometry-construction history
(including several CAD/DAGMC failure modes and how they were resolved),
materials, source/tally definitions, closed-form cost model, and the
complete 256-configuration enumeration results with the resulting 25-point
true Pareto frontier.

## Repository structure

```
i4-fusion-blanket-verification/
  README.md
  report.tex, report.pdf          <- technical report (this repo's main artifact)
  code/
    i4_case_v3.py                 <- single-configuration reference script
    run_full_enumeration.py       <- full 256-configuration sweep
    i4_blanket_v3.h5m             <- DAGMC geometry file (shared across all 256 runs)
  results/
    call_0000.json ... call_0255.json   <- one file per configuration
    summary.json                        <- all 256 records + Pareto frontier
```

## Reproducing the results

Requires OpenMC 0.15.3 (DAGMC build), Paramak, CadQuery, cad_to_dagmc, and
the ENDF/B-VIII.0 nuclear data library. See Section 2 ("Software Stack") of
the report for exact installation commands.

```bash
mamba activate fusion-env
cd code
python run_full_enumeration.py
```

Any individual reported TBR value can be checked directly against its
`results/call_NNNN.json` file without re-running anything.

## Status

Pipeline verified end to end; full brute-force Pareto frontier computed for
the small instance. Geometry is currently a simplified solid (non-hollow)
PbLi torus segment, not a full multi-layer WCLL-style blanket -- see the
report's Discussion (Section 10.2) for the specific simplifications on
which expert feedback is requested before scaling to the full $I=80$ case.

## Citation / related work

Companion case study (nuclear fuel assembly design):
https://github.com/s-shim/arr-dragon5-bwr10x10
