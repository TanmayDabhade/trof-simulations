# Result completion status

The committed result files are an **audited validation subset**, not the final
paper dataset. They contain the full 30-day Scenario A, seed 0 run for MPC,
tuned heuristic, and naive heuristic, plus the parameter and causal-forecast
tables for all three scenarios.

Completed at this checkpoint:

- 3 of 45 main-study runs;
- 0 of 150 demand-scale sweep runs;
- 0 of 3,000 Monte Carlo controller runs;
- 0 of 3 perfect-forecast ablation runs.

Accordingly, single-seed standard deviations in `main_results.csv` are blank,
and figures 3 and 4 are not present. The existing numbers are produced by the
running model and pass the energy audit, but must not be quoted as the final
mean ± SD results. Run `python study.py` repeatedly (optionally with
`--budget-seconds`) to resume from `run_registry.csv`, then run
`python figures.py` after the matrix completes.
