# Result completion status

The committed results are the complete revised study, produced after the
service-delivery boundary correction:

- 45 of 45 main-study runs (3 scenarios x 3 controllers x 5 seeds);
- 150 of 150 demand-scale sweep runs;
- 300 Monte Carlo controller runs (50 parameter draws per scenario x 2 controllers;
  reduced from the 500-draw design);
- 3 of 3 perfect-forecast ablation runs.

Main and ablation runs were executed locally; sweep and Monte Carlo runs were
executed on GitHub Actions and merged with `.github/scripts/ci_study.py merge`.
Tables and figures were regenerated with `study.aggregate_tables()` and
`python figures.py`.
