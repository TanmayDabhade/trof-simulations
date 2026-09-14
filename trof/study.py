"""Resumable multi-run study driver and paper-table generation."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .components import PlantParameters, component_parameter_table
from .run import RunSpec, run_closed_loop
from .scenarios import SCENARIOS, forecast_error_table, generate_scenario


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TIMESERIES = RESULTS / "timeseries"
REGISTRY = RESULTS / "run_registry.csv"
FAILED_RUNS = RESULTS / "failed_runs.csv"


def _load_registry() -> pd.DataFrame:
    return pd.read_csv(REGISTRY) if REGISTRY.exists() else pd.DataFrame()


def _append_registry(summary: dict[str, object]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([summary])
    row.to_csv(REGISTRY, mode="a", header=not REGISTRY.exists(), index=False)


def _parameter_draws(count: int, seed: int = 20260903) -> list[PlantParameters]:
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(count):
        base = PlantParameters()
        draws.append(replace(
            base,
            capture_fraction=float(rng.uniform(0.55, 0.80)),
            heat_pump=replace(base.heat_pump, second_law_efficiency=float(rng.uniform(0.40, 0.55))),
            absorption=replace(base.absorption, nominal_cop_at_30c=float(rng.uniform(0.60, 0.80))),
            orc=replace(base.orc, second_law_efficiency=float(rng.uniform(0.45, 0.60))),
            store=replace(base.store, standing_loss_per_day=float(rng.uniform(0.005, 0.03))),
            mechanical_chiller=replace(base.mechanical_chiller, nominal_cop=float(rng.uniform(4.5, 6.5))),
        ))
    return draws


def build_run_queue(min_seeds: int, mc_draws: int, days: int, phases: set[str]) -> list[tuple[RunSpec, PlantParameters]]:
    queue: list[tuple[RunSpec, PlantParameters]] = []
    base = PlantParameters()
    if "main" in phases:
        for scenario in SCENARIOS:
            for controller in ("mpc", "tuned", "naive"):
                for seed in range(min_seeds):
                    queue.append((RunSpec(scenario, controller, seed, days=days, study_group="main"), base))
    if "sweep" in phases:
        for scenario in SCENARIOS:
            for scale in (0.5, 1.0, 2.0, 3.0, 5.0):
                for controller in ("mpc", "tuned"):
                    for seed in range(min_seeds):
                        queue.append((RunSpec(scenario, controller, seed, days=days, demand_scale=scale, study_group="sweep"), base))
    if "mc" in phases:
        for draw, params in enumerate(_parameter_draws(mc_draws)):
            for scenario_index, scenario in enumerate(SCENARIOS):
                simulation_seed = 100_000 + 10 * draw + scenario_index
                for controller in ("mpc", "tuned"):
                    queue.append((RunSpec(scenario, controller, simulation_seed, days=days, study_group="monte_carlo", parameter_draw=draw), params))
    if "ablation" in phases:
        for scenario in SCENARIOS:
            queue.append((RunSpec(scenario, "mpc", 0, days=days, perfect_forecast=True, study_group="perfect_forecast"), base))
    return queue


def aggregate_tables() -> None:
    registry = _load_registry()
    if registry.empty:
        return
    main = registry[registry.study_group == "main"].copy()
    metrics = [
        "recovery_fraction_it", "recovery_fraction_captured", "dhw_delivered_mwh",
        "process_delivered_mwh", "absorption_heat_mwh", "absorption_cooling_mwh",
        "orc_electric_mwh", "unmet_demand_percent", "cooling_electric_saving_percent",
        "fallback_demand_percent", "compressor_mwh", "net_cost_inr",
    ]
    if not main.empty:
        grouped = main.groupby(["scenario", "city", "controller"])[metrics].agg(["mean", "std"]).reset_index()
        grouped.columns = ["_".join(filter(None, map(str, col))).rstrip("_") for col in grouped.columns]
        grouped.to_csv(RESULTS / "main_results.csv", index=False)

        comparisons = []
        for (scenario, seed), group in main.groupby(["scenario", "seed"]):
            costs = group.set_index("controller")["net_cost_inr"]
            if "mpc" not in costs:
                continue
            for baseline in ("tuned", "naive"):
                if baseline in costs and costs[baseline] != 0:
                    comparisons.append({
                        "scenario": scenario,
                        "seed": seed,
                        "baseline": baseline,
                        "mpc_improvement_percent": 100.0 * (costs[baseline] - costs["mpc"]) / costs[baseline],
                    })
        comparison_runs = pd.DataFrame(comparisons)
        if not comparison_runs.empty:
            comparison_runs.to_csv(RESULTS / "controller_comparison_runs.csv", index=False)
            comparison_runs.groupby(["scenario", "baseline"])["mpc_improvement_percent"].agg(["mean", "std"]).reset_index().to_csv(
                RESULTS / "controller_comparison.csv", index=False
            )

        audits = []
        for scenario in SCENARIOS:
            subset = main[(main.scenario == scenario) & (main.controller == "mpc")]
            if subset.empty:
                continue
            for _, row in subset.iterrows():
                closure = (
                    row.captured_mwh + row.compressor_mwh
                    - row.dhw_hp_mwh - row.process_hp_mwh
                    - row.absorption_heat_mwh - row.orc_heat_mwh
                    - row.store_charge_commanded_mwh
                    - row.rejected_mwh - row.buffer_net_change_mwh - row.buffer_standing_loss_mwh
                )
                store_closure = (
                    row.store_charge_mwh - row.store_discharge_mwh
                    - row.store_net_change_mwh - row.store_standing_loss_mwh
                )
                service_closure = (
                    row.dhw_hp_mwh + row.process_hp_mwh + row.store_charge_commanded_mwh + row.store_discharge_mwh
                    - row.dhw_delivered_mwh - row.process_delivered_mwh - row.store_charge_mwh - row.dumped_heat_mwh
                )
                audits.append({
                    "scenario": scenario, "seed": row.seed,
                    "captured_mwh": row.captured_mwh, "absorbed_mwh": row.absorbed_mwh,
                    "compressor_mwh": row.compressor_mwh,
                    "dhw_delivered_mwh": row.dhw_delivered_mwh,
                    "process_delivered_mwh": row.process_delivered_mwh,
                    "dhw_hp_mwh": row.dhw_hp_mwh,
                    "process_hp_mwh": row.process_hp_mwh,
                    "absorption_heat_mwh": row.absorption_heat_mwh,
                    "orc_heat_mwh": row.orc_heat_mwh,
                    "store_net_change_mwh": row.store_net_change_mwh,
                    "store_charge_mwh": row.store_charge_mwh,
                    "store_discharge_mwh": row.store_discharge_mwh,
                    "dumped_heat_mwh": row.dumped_heat_mwh,
                    "absorption_cooling_mwh": row.absorption_cooling_mwh,
                    "surplus_cooling_mwh": row.surplus_cooling_mwh,
                    "it_heat_mwh": row.it_heat_mwh if "it_heat_mwh" in row else float("nan"),
                    "store_standing_loss_mwh": row.store_standing_loss_mwh,
                    "buffer_net_change_mwh": row.buffer_net_change_mwh,
                    "buffer_standing_loss_mwh": row.buffer_standing_loss_mwh,
                    "rejected_mwh": row.rejected_mwh,
                    "closure_residual_mwh": closure,
                    "store_closure_residual_mwh": store_closure,
                    "service_closure_residual_mwh": service_closure,
                    "max_timestep_residual_kw": row.max_balance_residual_kw,
                })
        pd.DataFrame(audits).to_csv(RESULTS / "energy_balance_audit.csv", index=False)

    sweep = registry[registry.study_group == "sweep"].copy()
    if not sweep.empty:
        sweep.to_csv(RESULTS / "demand_scale_sweep_runs.csv", index=False)
        rows = []
        for (scenario, scale, seed), group in sweep.groupby(["scenario", "demand_scale", "seed"]):
            by_controller = group.set_index("controller")
            if {"mpc", "tuned"}.issubset(by_controller.index):
                tuned_cost = float(by_controller.loc["tuned", "net_cost_inr"])
                rows.append({
                    "scenario": scenario, "demand_scale": scale, "seed": seed,
                    "mpc_recovery_fraction_it": float(by_controller.loc["mpc", "recovery_fraction_it"]),
                    "tuned_recovery_fraction_it": float(by_controller.loc["tuned", "recovery_fraction_it"]),
                    "mpc_improvement_percent": 100.0 * (tuned_cost - float(by_controller.loc["mpc", "net_cost_inr"])) / tuned_cost,
                })
        sweep_pairs = pd.DataFrame(rows)
        if not sweep_pairs.empty:
            sweep_pairs.groupby(["scenario", "demand_scale"]).agg(
                recovery_fraction_it_mean=("mpc_recovery_fraction_it", "mean"),
                recovery_fraction_it_sd=("mpc_recovery_fraction_it", "std"),
                mpc_improvement_percent_mean=("mpc_improvement_percent", "mean"),
                mpc_improvement_percent_sd=("mpc_improvement_percent", "std"),
            ).reset_index().to_csv(RESULTS / "demand_scale_sweep.csv", index=False)

    mc = registry[registry.study_group == "monte_carlo"].copy()
    if not mc.empty:
        pairs = []
        for (scenario, draw), group in mc.groupby(["scenario", "parameter_draw"]):
            by_controller = group.set_index("controller")
            if {"mpc", "tuned"}.issubset(by_controller.index):
                tuned = float(by_controller.loc["tuned", "net_cost_inr"])
                pairs.append({
                    "scenario": scenario, "parameter_draw": draw,
                    "mpc_improvement_percent": 100.0 * (tuned - float(by_controller.loc["mpc", "net_cost_inr"])) / tuned,
                    "max_balance_residual_kw": max(float(by_controller.loc["mpc", "max_balance_residual_kw"]), float(by_controller.loc["tuned", "max_balance_residual_kw"])),
                })
        mc_pairs = pd.DataFrame(pairs)
        if not mc_pairs.empty:
            mc_pairs.to_csv(RESULTS / "monte_carlo_runs.csv", index=False)
            mc_pairs.groupby("scenario")["mpc_improvement_percent"].describe(percentiles=[0.05, 0.5, 0.95]).reset_index().to_csv(
                RESULTS / "monte_carlo_summary.csv", index=False
            )

    ablation = registry[registry.study_group == "perfect_forecast"].copy()
    if not ablation.empty:
        ablation.to_csv(RESULTS / "perfect_forecast_ablation.csv", index=False)


def write_static_tables(days: int = 30) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    component_parameter_table().to_csv(RESULTS / "component_parameters.csv", index=False)
    forecast_rows = []
    for scenario in SCENARIOS:
        data = generate_scenario(scenario, seed=0, days=days)
        table = forecast_error_table(data)
        table.insert(0, "scenario", scenario)
        forecast_rows.append(table)
    pd.concat(forecast_rows, ignore_index=True).to_csv(RESULTS / "forecast_errors.csv", index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-seconds", type=float, default=None, help="Stop starting new runs after this wall-clock budget; completed runs remain saved.")
    parser.add_argument("--min-seeds", type=int, default=5)
    parser.add_argument("--mc-draws", type=int, default=500)
    parser.add_argument("--days", type=int, default=30, help="30 is the paper design; smaller values are for smoke tests only.")
    parser.add_argument("--phases", default="main,sweep,mc,ablation", help="Comma-separated subset of main,sweep,mc,ablation")
    parser.add_argument("--mpc-horizon", type=int, default=48)
    parser.add_argument("--solver-time-limit", type=float, default=60.0)
    parser.add_argument("--no-timeseries", action="store_true", help="Do not retain per-step files (residual maxima remain in summaries).")
    parser.add_argument("--shard-index", type=int, default=0, help="Run only this shard of the not-yet-completed queue (for parallel CI).")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--continue-on-error", action="store_true", help="Record a failed run in failed_runs.csv and continue instead of stopping.")
    args = parser.parse_args(argv)
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must be in [0, --shard-count)")
    phases = set(args.phases.split(","))
    unknown = phases - {"main", "sweep", "mc", "ablation"}
    if unknown:
        parser.error(f"unknown phases: {sorted(unknown)}")
    TIMESERIES.mkdir(parents=True, exist_ok=True)
    write_static_tables(days=args.days)
    completed = set(_load_registry().get("run_id", pd.Series(dtype=str)).astype(str))
    queue = build_run_queue(args.min_seeds, args.mc_draws, args.days, phases)
    if args.shard_count > 1:
        pending = [item for item in queue if item[0].run_id not in completed]
        # Finish the core design before Monte Carlo so a time budget cuts only MC draws.
        pending.sort(key=lambda item: item[0].study_group == "monte_carlo")
        queue = pending[args.shard_index::args.shard_count]
    start = time.monotonic()
    for index, (spec, params) in enumerate(queue, start=1):
        if spec.run_id in completed:
            continue
        if args.budget_seconds is not None and time.monotonic() - start >= args.budget_seconds:
            break
        print(f"[{index}/{len(queue)}] {spec.run_id}", flush=True)
        keep_ts = not args.no_timeseries and spec.study_group in {"main", "perfect_forecast"}
        path = TIMESERIES / f"{spec.run_id}.csv" if keep_ts else None
        try:
            _, summary = run_closed_loop(
                spec, params=params, save_path=path, mpc_horizon=args.mpc_horizon,
                mpc_solver_time_limit_s=args.solver_time_limit,
            )
        except (ValueError, RuntimeError) as error:
            if not args.continue_on_error:
                raise
            print(f"FAILED {spec.run_id}: {type(error).__name__}: {error}", flush=True)
            pd.DataFrame([{"run_id": spec.run_id, "error": f"{type(error).__name__}: {error}"}]).to_csv(
                FAILED_RUNS, mode="a", header=not FAILED_RUNS.exists(), index=False
            )
            continue
        summary["parameters_json"] = json.dumps({
            "capture_fraction": params.capture_fraction,
            "hp_eta2": params.heat_pump.second_law_efficiency,
            "absorption_nominal_cop": params.absorption.nominal_cop_at_30c,
            "orc_eta2": params.orc.second_law_efficiency,
            "store_loss_per_day": params.store.standing_loss_per_day,
            "mechanical_chiller_nominal_cop": params.mechanical_chiller.nominal_cop,
        }, sort_keys=True)
        _append_registry(summary)
        completed.add(spec.run_id)
        aggregate_tables()
    aggregate_tables()
    print(f"Completed {len(completed)} registered runs. Re-run the same command to resume.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
