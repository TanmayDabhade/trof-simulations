"""Closed-loop execution for one scenario/controller/seed combination."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .components import PlantParameters
from .dispatch import Controller, build_controller
from .plant import PlantState, advance_plant
from .scenarios import CausalSeasonalNaive, SCENARIOS, ScenarioConfig, generate_scenario


TIMESERIES_COLUMNS = [
    "controller", "timestamp", "step", "hour", "ambient_c", "it_kw",
    "dhw_demand_kw", "process_demand_kw", "cooling_demand_kw",
    "grid_available", "grid_roster_available", "site_captured_kw", "captured_kw",
    "hp_absorbed_kw", "hp_compressor_kw", "hp_delivered_kw", "hp_units_on",
    "dhw_heat_kw", "process_heat_kw", "absorption_heat_kw", "orc_heat_kw",
    "store_charge_kw", "store_discharge_kw", "store_energy_kwh", "store_soc",
    "store_standing_loss_kw", "buffer_charge_kw", "buffer_discharge_kw",
    "buffer_energy_kwh", "buffer_standing_loss_kw", "rejected_kw",
    "absorption_cooling_kw", "orc_electric_kw", "fallback_dhw_kw",
    "fallback_process_kw", "fallback_cooling_kw", "baseline_rejection_electric_kw",
    "actual_rejection_electric_kw", "cooling_electric_saving_kw", "grid_electric_kw",
    "diesel_electric_kw", "net_cost_inr", "source_balance_residual_kw",
    "hp_balance_residual_kw", "store_balance_residual_kw", "buffer_balance_residual_kw",
    "dhw_served_kw", "process_served_kw", "cooling_served_kw", "store_discharge_commanded_kw", "store_charge_commanded_kw",
    "dumped_heat_kw", "surplus_cooling_kw", "service_balance_residual_kw",
]


@dataclass(frozen=True)
class RunSpec:
    scenario: str
    controller: str
    seed: int
    days: int = 30
    demand_scale: float = 1.0
    perfect_forecast: bool = False
    study_group: str = "main"
    parameter_draw: int | None = None

    @property
    def run_id(self) -> str:
        draw = "none" if self.parameter_draw is None else str(self.parameter_draw)
        return (
            f"{self.study_group}__{self.scenario}__{self.controller}__seed{self.seed}"
            f"__days{self.days}__scale{self.demand_scale:g}__perfect{int(self.perfect_forecast)}__draw{draw}"
        )


def _perfect_forecast(frame: pd.DataFrame, origin: int, horizon: int) -> pd.DataFrame:
    """Return a copy, never a view or the same array as realised data."""
    indices = np.minimum(np.arange(origin, origin + horizon), len(frame) - 1)
    columns = ["ambient_c", "it_kw", "dhw_demand_kw", "process_demand_kw", "cooling_demand_kw"]
    result = frame.iloc[indices][columns].reset_index(drop=True).copy(deep=True)
    result["grid_available"] = frame.iloc[indices]["grid_available"].to_numpy(copy=True)
    result["step"] = indices.copy()
    return result


def run_closed_loop(
    spec: RunSpec,
    params: PlantParameters | None = None,
    scenario_data: pd.DataFrame | None = None,
    controller: Controller | None = None,
    save_path: Path | None = None,
    mpc_horizon: int = 48,
    mpc_solver_time_limit_s: float | None = None,
) -> tuple[pd.DataFrame, dict[str, float | str | int | bool]]:
    p = params or PlantParameters()
    p.validate()
    config: ScenarioConfig = SCENARIOS[spec.scenario]
    frame = scenario_data if scenario_data is not None else generate_scenario(
        config, spec.seed, days=spec.days, demand_scale=spec.demand_scale, capture_fraction=p.capture_fraction
    )
    if controller is None:
        kwargs = {"horizon": mpc_horizon, "solver_time_limit_s": mpc_solver_time_limit_s} if spec.controller == "mpc" else {}
        controller = build_controller(spec.controller, p, **kwargs)
    forecaster = CausalSeasonalNaive(frame)
    state = PlantState.initial(p)
    initial_store = state.store_energy_kwh
    initial_buffer = state.buffer_energy_kwh
    rows: list[dict[str, object]] = []

    for step, realised in frame.iterrows():
        forecast = _perfect_forecast(frame, step, mpc_horizon) if spec.perfect_forecast else forecaster.forecast(step, mpc_horizon)
        # Guard the no-lookahead rule structurally, not by convention.
        if not spec.perfect_forecast:
            realised_arrays = [frame[c].to_numpy() for c in forecaster.columns]
            forecast_arrays = [forecast[c].to_numpy() for c in forecaster.columns]
            if any(np.shares_memory(a, b) for a in realised_arrays for b in forecast_arrays):
                raise RuntimeError("forecast shares memory with realised data; study is invalid")
        action = controller.act(step, state, forecast, realised)
        realised_dict = {k: float(realised[k]) for k in (
            "ambient_c", "site_captured_kw", "dhw_demand_kw", "process_demand_kw",
            "cooling_demand_kw", "grid_available",
        )}
        state, physical = advance_plant(state, action, realised_dict, p)
        rows.append({
            "run_id": spec.run_id,
            "study_group": spec.study_group,
            "scenario": spec.scenario,
            "city": config.city,
            "controller": spec.controller,
            "seed": spec.seed,
            "demand_scale": spec.demand_scale,
            "perfect_forecast": spec.perfect_forecast,
            "timestamp": realised["timestamp"],
            "step": step,
            "hour": realised["hour"],
            "ambient_c": realised["ambient_c"],
            "it_kw": realised["it_kw"],
            "dhw_demand_kw": realised["dhw_demand_kw"],
            "process_demand_kw": realised["process_demand_kw"],
            "cooling_demand_kw": realised["cooling_demand_kw"],
            "grid_available": realised["grid_available"],
            "grid_roster_available": realised["grid_roster_available"],
            "forecast_ambient_c": forecast.iloc[0]["ambient_c"],
            "forecast_it_kw": forecast.iloc[0]["it_kw"],
            "forecast_dhw_kw": forecast.iloc[0]["dhw_demand_kw"],
            "forecast_process_kw": forecast.iloc[0]["process_demand_kw"],
            "forecast_cooling_kw": forecast.iloc[0]["cooling_demand_kw"],
            **physical,
        })

    result = pd.DataFrame(rows)
    summary = summarise_run(result, spec, p, initial_store, initial_buffer)
    summary["mpc_solves"] = getattr(controller, "solves", 0)
    summary["mpc_time_limited_solves"] = getattr(controller, "time_limited_solves", 0)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        temp = save_path.with_suffix(".tmp")
        result[TIMESERIES_COLUMNS].to_csv(temp, index=False, float_format="%.7g")
        temp.replace(save_path)
    return result, summary


def summarise_run(
    result: pd.DataFrame,
    spec: RunSpec,
    params: PlantParameters,
    initial_store_kwh: float,
    initial_buffer_kwh: float,
) -> dict[str, float | str | int | bool]:
    dt = params.timestep_hours
    energy = lambda column: float(result[column].sum() * dt)
    it_kwh = energy("it_kw")
    site_captured_kwh = energy("site_captured_kw")
    absorbed_kwh = energy("hp_absorbed_kw")
    thermal_demand_kwh = energy("dhw_demand_kw") + energy("process_demand_kw")
    cooling_demand_kwh = energy("cooling_demand_kw")
    unmet_equivalent_kwh = energy("unmet_dhw_kw") + energy("unmet_process_kw") + energy("unmet_cooling_kw")
    fallback_equivalent_kwh = energy("fallback_dhw_kw") + energy("fallback_process_kw") + energy("fallback_cooling_kw")
    total_demand_kwh = thermal_demand_kwh + cooling_demand_kwh
    baseline_rejection_electric_kwh = energy("baseline_rejection_electric_kw")
    cooling_saving_kwh = energy("cooling_electric_saving_kw")
    max_residual = float(result[[
        "source_balance_residual_kw", "hp_balance_residual_kw", "store_balance_residual_kw", "buffer_balance_residual_kw",
        "service_balance_residual_kw",
    ]].abs().to_numpy().max())
    return {
        "run_id": spec.run_id,
        "study_group": spec.study_group,
        "scenario": spec.scenario,
        "city": SCENARIOS[spec.scenario].city,
        "controller": spec.controller,
        "seed": spec.seed,
        "days": spec.days,
        "demand_scale": spec.demand_scale,
        "perfect_forecast": spec.perfect_forecast,
        "parameter_draw": -1 if spec.parameter_draw is None else spec.parameter_draw,
        "recovery_fraction_it": absorbed_kwh / max(it_kwh, 1e-12),
        "recovery_fraction_captured": absorbed_kwh / max(site_captured_kwh, 1e-12),
        "it_heat_mwh": it_kwh / 1000.0,
        "captured_mwh": site_captured_kwh / 1000.0,
        "absorbed_mwh": absorbed_kwh / 1000.0,
        "compressor_mwh": energy("hp_compressor_kw") / 1000.0,
        "dhw_delivered_mwh": energy("dhw_served_kw") / 1000.0,
        "process_delivered_mwh": energy("process_served_kw") / 1000.0,
        "dhw_hp_mwh": energy("dhw_heat_kw") / 1000.0,
        "process_hp_mwh": energy("process_heat_kw") / 1000.0,
        "absorption_heat_mwh": energy("absorption_heat_kw") / 1000.0,
        "absorption_cooling_mwh": energy("cooling_served_kw") / 1000.0,
        "orc_heat_mwh": energy("orc_heat_kw") / 1000.0,
        "orc_electric_mwh": energy("orc_electric_kw") / 1000.0,
        "store_charge_mwh": energy("store_charge_kw") / 1000.0,
        "store_charge_commanded_mwh": energy("store_charge_commanded_kw") / 1000.0,
        "store_discharge_mwh": energy("store_discharge_kw") / 1000.0,
        "store_discharge_dhw_mwh": energy("store_discharge_dhw_kw") / 1000.0,
        "store_discharge_process_mwh": energy("store_discharge_process_kw") / 1000.0,
        "dumped_heat_mwh": energy("dumped_heat_kw") / 1000.0,
        "surplus_cooling_mwh": energy("surplus_cooling_kw") / 1000.0,
        "store_net_change_mwh": (float(result.iloc[-1]["store_energy_kwh"]) - initial_store_kwh) / 1000.0,
        "store_standing_loss_mwh": energy("store_standing_loss_kw") / 1000.0,
        "buffer_net_change_mwh": (float(result.iloc[-1]["buffer_energy_kwh"]) - initial_buffer_kwh) / 1000.0,
        "buffer_standing_loss_mwh": energy("buffer_standing_loss_kw") / 1000.0,
        "rejected_mwh": energy("rejected_kw") / 1000.0,
        "unmet_demand_percent": 100.0 * unmet_equivalent_kwh / max(total_demand_kwh, 1e-12),
        "fallback_demand_percent": 100.0 * fallback_equivalent_kwh / max(total_demand_kwh, 1e-12),
        "cooling_electric_saving_percent": 100.0 * cooling_saving_kwh / max(baseline_rejection_electric_kwh, 1e-12),
        "net_cost_inr": float(result["net_cost_inr"].sum()),
        "max_source_residual_kw": float(result["source_balance_residual_kw"].abs().max()),
        "max_hp_residual_kw": float(result["hp_balance_residual_kw"].abs().max()),
        "max_store_residual_kw": float(result["store_balance_residual_kw"].abs().max()),
        "max_buffer_residual_kw": float(result["buffer_balance_residual_kw"].abs().max()),
        "max_service_residual_kw": float(result["service_balance_residual_kw"].abs().max()),
        "max_balance_residual_kw": max_residual,
    }
