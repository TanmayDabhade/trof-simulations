"""Synthetic scenario generation and strictly causal seasonal-naive forecasts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


STEPS_PER_HOUR = 4
STEPS_PER_DAY = 24 * STEPS_PER_HOUR


@dataclass(frozen=True)
class ScenarioConfig:
    key: str
    city: str
    season: str
    ambient_mean_c: float
    ambient_swing_c: float
    it_utilisation: float
    dhw_peak_kw: float
    process_peak_kw: float
    cooling_peak_kw: float
    grid_hours_per_day: float
    outage_windows_h: tuple[tuple[float, float], ...]


SCENARIOS: dict[str, ScenarioConfig] = {
    "A": ScenarioConfig("A", "Nagpur", "peak summer (May)", 34.0, 8.0, 0.75, 430.0, 300.0, 760.0, 16.0, ((11.0, 14.0), (18.5, 23.5))),
    "B": ScenarioConfig("B", "Bhopal", "monsoon, grid stress (Aug)", 26.5, 4.5, 0.62, 620.0, 240.0, 380.0, 12.0, ((6.0, 9.0), (11.0, 14.0), (17.5, 23.5))),
    # The source brief states 18 grid h/day but gives only a 19:00-23:00
    # outage (4 h). The executable model follows the explicit window: 20 h/day.
    "C": ScenarioConfig("C", "Coimbatore", "winter, industrial (Jan)", 24.0, 7.0, 0.70, 780.0, 420.0, 300.0, 20.0, ((19.0, 23.0),)),
}


def _ar1(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    values = np.zeros(n, dtype=float)
    innovations = rng.normal(0.0, sigma, n)
    for i in range(1, n):
        values[i] = phi * values[i - 1] + innovations[i]
    return values


def _periodic_gaussian(hour: np.ndarray, centre: float, width: float) -> np.ndarray:
    distance = np.minimum(np.abs(hour - centre), 24.0 - np.abs(hour - centre))
    return np.exp(-0.5 * (distance / width) ** 2)


def _scheduled_grid(config: ScenarioConfig, hours: np.ndarray) -> np.ndarray:
    available = np.ones_like(hours, dtype=float)
    for start, end in config.outage_windows_h:
        available[(hours >= start) & (hours < end)] = 0.0
    return available


def generate_scenario(
    config: ScenarioConfig | str,
    seed: int,
    days: int = 30,
    demand_scale: float = 1.0,
    capture_fraction: float = 0.70,
) -> pd.DataFrame:
    """Generate one realised scenario.

    The shapes and noise processes are design assumptions, not measurements.
    Outage timing jitter is realised independently of the published roster.
    """
    if isinstance(config, str):
        config = SCENARIOS[config]
    rng = np.random.default_rng(seed)
    n = days * STEPS_PER_DAY
    idx = pd.date_range("2026-01-05", periods=n, freq="15min")
    step = np.arange(n)
    hour = (step % STEPS_PER_DAY) / STEPS_PER_HOUR
    day = step // STEPS_PER_DAY
    weekday = day % 7

    # Minimum occurs close to 05:30; peak-to-trough equals ambient_swing_c.
    ambient = config.ambient_mean_c - 0.5 * config.ambient_swing_c * np.cos(2.0 * np.pi * (hour - 5.5) / 24.0)
    ambient += _ar1(rng, n, phi=0.985, sigma=0.13)

    interactive = 0.86 + 0.14 * _periodic_gaussian(hour, 14.0, 4.2)
    it_noise = _ar1(rng, n, phi=0.80, sigma=0.007)
    it_kw = 5000.0 * config.it_utilisation * np.clip(interactive + it_noise, 0.72, 1.08)
    site_captured_kw = capture_fraction * it_kw

    dhw_shape = 0.12 + 0.88 * np.maximum(
        _periodic_gaussian(hour, 7.0, 1.25),
        0.92 * _periodic_gaussian(hour, 19.5, 1.55),
    )
    dhw_noise = _ar1(rng, n, phi=0.70, sigma=0.018)
    dhw_kw = demand_scale * config.dhw_peak_kw * np.clip(dhw_shape + dhw_noise, 0.05, None)

    working = (weekday <= 5) & (hour >= 8.0) & (hour < 18.0)
    saturday_factor = np.where(weekday == 5, 0.5, 1.0)
    process_shape = np.where(working, saturday_factor * (0.88 + 0.12 * np.sin(np.pi * (hour - 8.0) / 10.0)), 0.0)
    process_noise = _ar1(rng, n, phi=0.65, sigma=0.012)
    process_kw = demand_scale * config.process_peak_kw * np.clip(process_shape + process_noise, 0.0, None)

    afternoon = _periodic_gaussian(hour, 15.0, 3.3)
    ambient_factor = np.clip(0.55 + 0.055 * (ambient - config.ambient_mean_c), 0.18, 1.15)
    cooling_shape = np.clip(0.12 + 0.88 * afternoon, 0.0, 1.0) * ambient_factor
    cooling_noise = _ar1(rng, n, phi=0.72, sigma=0.015)
    cooling_kw = demand_scale * config.cooling_peak_kw * np.clip(cooling_shape + cooling_noise, 0.0, None)

    scheduled_grid = _scheduled_grid(config, hour)
    realised_grid = np.ones(n, dtype=float)
    for d in range(days):
        jitter = rng.normal(0.0, 0.35, len(config.outage_windows_h))
        mask_day = day == d
        for (start, end), shift in zip(config.outage_windows_h, jitter, strict=True):
            realised_grid[mask_day & (hour >= start + shift) & (hour < end + shift)] = 0.0

    return pd.DataFrame({
        "timestamp": idx,
        "step": step,
        "hour": hour,
        "ambient_c": ambient,
        "it_kw": it_kw,
        "site_captured_kw": site_captured_kw,
        "dhw_demand_kw": dhw_kw,
        "process_demand_kw": process_kw,
        "cooling_demand_kw": cooling_kw,
        "grid_available": realised_grid,
        "grid_roster_available": scheduled_grid,
    })


class CausalSeasonalNaive:
    """Same-slot previous-day forecast using observations strictly before t."""

    weights = np.array([0.55, 0.25, 0.12, 0.05, 0.02, 0.01], dtype=float)

    def __init__(self, frame: pd.DataFrame, columns: tuple[str, ...] | None = None):
        self.frame = frame
        self.columns = columns or ("ambient_c", "it_kw", "dhw_demand_kw", "process_demand_kw", "cooling_demand_kw")

    def _one(self, column: str, origin: int, target: int) -> float:
        values = self.frame[column].to_numpy()
        candidates: list[float] = []
        candidate_weights: list[float] = []
        for day_back, weight in enumerate(self.weights, start=1):
            index = target - day_back * STEPS_PER_DAY
            if 0 <= index < origin:  # The strict causality condition.
                candidates.append(float(values[index]))
                candidate_weights.append(float(weight))
        if candidates:
            w = np.asarray(candidate_weights)
            return float(np.dot(candidates, w) / w.sum())
        if origin <= 0:
            # Static scenario design values only; never read realised values.
            defaults = {
                "ambient_c": 30.0,
                "it_kw": 3500.0,
                "dhw_demand_kw": 200.0,
                "process_demand_kw": 100.0,
                "cooling_demand_kw": 250.0,
            }
            return defaults[column]
        return float(np.mean(values[:origin]))

    def forecast(self, origin: int, horizon: int = 48) -> pd.DataFrame:
        rows: dict[str, list[float]] = {column: [] for column in self.columns}
        n = len(self.frame)
        for offset in range(horizon):
            target = min(origin + offset, n - 1)
            for column in self.columns:
                rows[column].append(self._one(column, origin, target))
        # The published grid roster is intentionally known; realised jitter is not.
        roster = self.frame["grid_roster_available"].to_numpy()
        rows["grid_available"] = [float(roster[min(origin + h, n - 1)]) for h in range(horizon)]
        rows["step"] = [min(origin + h, n - 1) for h in range(horizon)]
        return pd.DataFrame(rows)


def forecast_error_table(frame: pd.DataFrame, max_origins: int | None = None) -> pd.DataFrame:
    """Compute causal forecast MAE/NMAE at the requested paper horizons."""
    forecaster = CausalSeasonalNaive(frame)
    mappings = {0.0: 0, 1.0: 4, 3.0: 12, 6.0: 24, 12.0: 48}
    columns = ("dhw_demand_kw", "process_demand_kw", "cooling_demand_kw")
    origins = range(len(frame))
    if max_origins is not None:
        origins = range(min(len(frame), max_origins))
    errors: dict[tuple[str, float], list[float]] = {(c, h): [] for c in columns for h in mappings}
    for origin in origins:
        fc = forecaster.forecast(origin, 49)
        for horizon_h, offset in mappings.items():
            target = origin + offset
            if target >= len(frame):
                continue
            for column in columns:
                errors[(column, horizon_h)].append(abs(float(fc.iloc[offset][column]) - float(frame.iloc[target][column])))
    rows = []
    for column in columns:
        normaliser = max(float(frame[column].mean()), 1e-12)
        for horizon_h in mappings:
            mae = float(np.mean(errors[(column, horizon_h)]))
            rows.append({"series": column, "horizon_hours": horizon_h, "mae_kw": mae, "nmae": mae / normaliser})
    return pd.DataFrame(rows)
