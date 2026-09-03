"""Generate all paper figures from saved CSV outputs only."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TIMESERIES = RESULTS / "timeseries"
FIGURES = ROOT / "figures"


def _style() -> None:
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.constrained_layout.use": True})


def _save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.jpg", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _main_timeseries() -> dict[str, pd.DataFrame]:
    files = list(TIMESERIES.glob("main__A__*__seed0__days*__scale1__perfect0__drawnone.csv"))
    output = {}
    for path in files:
        frame = pd.read_csv(path, parse_dates=["timestamp"])
        output[str(frame.iloc[0].controller)] = frame
    return output


def dispatch_day() -> bool:
    frames = _main_timeseries()
    if not {"mpc", "tuned", "naive"}.issubset(frames):
        return False
    fig, ax = plt.subplots(figsize=(7.1, 3.6))
    colours = {"mpc": "#1565c0", "tuned": "#2e7d32", "naive": "#ef6c00"}
    day_slice = slice(7 * 96, 8 * 96)
    for name in ("mpc", "tuned", "naive"):
        frame = frames[name].iloc[day_slice]
        hours = frame.timestamp.dt.hour + frame.timestamp.dt.minute / 60.0
        ax.plot(hours, frame.hp_delivered_kw, label=name.upper(), lw=1.5, color=colours[name])
    frame = frames["mpc"].iloc[day_slice]
    community = frame.dhw_demand_kw + frame.process_demand_kw + frame.cooling_demand_kw
    hours = frame.timestamp.dt.hour + frame.timestamp.dt.minute / 60.0
    ax.plot(hours, community, label="Community demand (thermal + cooling)", color="#424242", ls="--", lw=1.2)
    ax.set(xlabel="Hour", ylabel="Power (kW)", title="Representative 24-hour dispatch — Scenario A, seed 0")
    ax.set_xlim(0, 24)
    ax.legend(ncol=2, frameon=False)
    _save(fig, "01_dispatch_day")
    return True


def energy_flow() -> bool:
    path = RESULTS / "energy_balance_audit.csv"
    if not path.exists():
        return False
    data = pd.read_csv(path).groupby("scenario").mean(numeric_only=True)
    if data.empty:
        return False
    scenarios = list(data.index)
    x = np.arange(len(scenarios))
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8))
    absorbed = data.absorbed_mwh.to_numpy()
    rejected = data.rejected_mwh.to_numpy()
    buffer_terms = (data.buffer_net_change_mwh + data.buffer_standing_loss_mwh).to_numpy()
    axes[0].bar(x, absorbed, label="Absorbed")
    axes[0].bar(x, rejected, bottom=absorbed, label="Rejected")
    axes[0].bar(x, buffer_terms, bottom=absorbed + rejected, label="Buffer change + loss")
    axes[0].plot(x, data.captured_mwh, "k_", ms=18, mew=2, label="Captured")
    axes[0].set(title="Source-side flow", ylabel="Energy over 30 days (MWh)", xticks=x, xticklabels=scenarios)
    axes[0].legend(frameon=False, fontsize=8)
    stacks = [("DHW", data.dhw_hp_mwh.to_numpy()), ("Process", data.process_hp_mwh.to_numpy()),
              ("Absorption", data.absorption_heat_mwh.to_numpy()), ("ORC", data.orc_heat_mwh.to_numpy()),
              ("Store net + loss", (data.store_net_change_mwh + data.store_standing_loss_mwh).to_numpy())]
    bottom = np.zeros(len(x))
    for label, values in stacks:
        axes[1].bar(x, values, bottom=bottom, label=label)
        bottom += values
    axes[1].set(title="Upgraded-heat destinations", ylabel="Energy over 30 days (MWh)", xticks=x, xticklabels=scenarios)
    axes[1].legend(frameon=False, fontsize=8)
    _save(fig, "02_energy_flow")
    return True


def demand_sweep() -> bool:
    path = RESULTS / "demand_scale_sweep.csv"
    if not path.exists():
        return False
    data = pd.read_csv(path)
    if data.empty:
        return False
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 6.0), sharex=True)
    for scenario, group in data.groupby("scenario"):
        group = group.sort_values("demand_scale")
        axes[0].errorbar(group.demand_scale, 100 * group.recovery_fraction_it_mean,
                         yerr=100 * group.recovery_fraction_it_sd.fillna(0), marker="o", label=f"Scenario {scenario}")
        axes[1].errorbar(group.demand_scale, group.mpc_improvement_percent_mean,
                         yerr=group.mpc_improvement_percent_sd.fillna(0), marker="o", label=f"Scenario {scenario}")
    axes[0].set(ylabel="Recovered IT heat (%)", title="Community demand-scale sweep")
    axes[1].set(xlabel="Demand multiplier", ylabel="MPC improvement vs tuned (%)")
    axes[0].legend(frameon=False, ncol=3)
    _save(fig, "03_demand_scale")
    return True


def monte_carlo() -> bool:
    path = RESULTS / "monte_carlo_runs.csv"
    deterministic = RESULTS / "controller_comparison.csv"
    if not path.exists():
        return False
    data = pd.read_csv(path)
    det = pd.read_csv(deterministic) if deterministic.exists() else pd.DataFrame()
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0), sharey=True)
    for ax, scenario in zip(axes, ("A", "B", "C"), strict=True):
        values = data.loc[data.scenario == scenario, "mpc_improvement_percent"]
        ax.hist(values, bins=25, color="#5c6bc0", alpha=0.85)
        marker = det[(det.scenario == scenario) & (det.baseline == "tuned")]
        if not marker.empty:
            ax.axvline(float(marker.iloc[0]["mean"]), color="#c62828", lw=1.6, label="Deterministic")
        ax.set(title=f"Scenario {scenario}", xlabel="MPC improvement (%)")
    axes[0].set_ylabel("Draws")
    axes[-1].legend(frameon=False)
    _save(fig, "04_monte_carlo")
    return True


def forecast_error() -> bool:
    path = RESULTS / "forecast_errors.csv"
    if not path.exists():
        return False
    data = pd.read_csv(path)
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0), sharey=True)
    labels = {"dhw_demand_kw": "DHW", "process_demand_kw": "Process", "cooling_demand_kw": "Cooling"}
    for ax, scenario in zip(axes, ("A", "B", "C"), strict=True):
        subset = data[data.scenario == scenario]
        for series, group in subset.groupby("series"):
            ax.plot(group.horizon_hours, 100 * group.nmae, marker="o", label=labels.get(series, series))
        ax.set(title=f"Scenario {scenario}", xlabel="Forecast horizon (h)")
    axes[0].set_ylabel("Normalised MAE (%)")
    axes[-1].legend(frameon=False)
    _save(fig, "05_forecast_error")
    return True


def store_soc() -> bool:
    frames = _main_timeseries()
    if "mpc" not in frames:
        return False
    frame = frames["mpc"].iloc[: 7 * 96].copy()
    x = np.arange(len(frame)) / 96.0
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot(x, 100 * frame.store_soc, color="#1565c0", lw=1.3)
    outage = frame.grid_available.to_numpy() < 0.5
    starts = np.flatnonzero(outage & ~np.r_[False, outage[:-1]])
    ends = np.flatnonzero(outage & ~np.r_[outage[1:], False]) + 1
    for start, end in zip(starts, ends, strict=True):
        ax.axvspan(start / 96.0, end / 96.0, color="#ef5350", alpha=0.13)
    ax.set(xlabel="Day", ylabel="State of charge (%)", title="MPC thermal-store operation; outages shaded")
    _save(fig, "06_store_soc")
    return True


def balance_residual() -> bool:
    frames = _main_timeseries()
    if "mpc" not in frames:
        return False
    frame = frames["mpc"]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    for column, label in (("source_balance_residual_kw", "Source"), ("hp_balance_residual_kw", "Heat pump"),
                          ("store_balance_residual_kw", "Store"), ("buffer_balance_residual_kw", "Buffer")):
        ax.plot(np.arange(len(frame)) / 96.0, frame[column], lw=0.8, label=label)
    ax.axhline(1e-9, color="black", ls=":", lw=0.8)
    ax.axhline(-1e-9, color="black", ls=":", lw=0.8)
    ax.set(xlabel="Day", ylabel="Balance residual (kW)", title="Timestep energy-balance audit")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(frameon=False, ncol=4)
    _save(fig, "07_balance_residual")
    return True


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    _style()
    builders = [dispatch_day, energy_flow, demand_sweep, monte_carlo, forecast_error, store_soc, balance_residual]
    generated, skipped = [], []
    for index, builder in enumerate(builders, start=1):
        (generated if builder() else skipped).append(index)
    print(f"Generated figures: {generated}")
    if skipped:
        print(f"Skipped figures without completed prerequisite runs: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
