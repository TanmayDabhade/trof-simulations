"""MPC and pathway-aware heuristic controllers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
import pulp

from .components import PlantParameters
from .plant import DispatchAction, PlantState, action_absorbed_and_work, infer_units_on


class Controller(Protocol):
    name: str

    def act(self, step: int, state: PlantState, forecast: pd.DataFrame, realised_now: pd.Series) -> DispatchAction: ...


def _source_credit_per_absorbed_kwh(
    ambient_c: float, captured_kw: float, p: PlantParameters, electricity_tariff: float | None = None
) -> float:
    tariff = p.tariffs.grid_inr_per_kwh if electricity_tariff is None else electricity_tariff
    if ambient_c <= p.dry_cooler.loop_return_c - p.dry_cooler.approach_k and captured_kw <= p.dry_cooler.capacity_kw:
        return tariff * p.dry_cooler.fan_fraction
    return tariff / p.mechanical_chiller.cop(ambient_c)


def _greedy_add(action: DispatchAction, field: str, requested_kw: float, remaining_output_kw: float, remaining_source_kw: float, sink: str, p: PlantParameters) -> tuple[float, float]:
    factor = 1.0 - 1.0 / p.heat_pump.cop(p.sink_temperatures_c[sink])
    limit = min(max(0.0, requested_kw), remaining_output_kw, remaining_source_kw / max(factor, 1e-12))
    setattr(action, field, getattr(action, field) + limit)
    return remaining_output_kw - limit, remaining_source_kw - limit * factor


def _make_dispatch_feasible(action: DispatchAction, state: PlantState, p: PlantParameters, raw_capture_kw: float) -> DispatchAction:
    """Meet modular turndown by adding useful store charge, else drop tiny output."""
    delivered = action.hp_delivered_kw
    if delivered <= 1e-10:
        action.hp_units_on = 0
        return action
    units = infer_units_on(delivered, p)
    minimum = units * p.heat_pump.min_turndown * p.heat_pump.rated_output_kw_per_unit
    if delivered < minimum:
        if action.store_discharge_kw > 1e-10:
            return DispatchAction(
                store_discharge_dhw_kw=action.store_discharge_dhw_kw,
                store_discharge_process_kw=action.store_discharge_process_kw,
            )
        room_kw = max(0.0, (p.store.max_soc * p.store.capacity_kwh - state.store_energy_kwh) / p.timestep_hours)
        extra = min(minimum - delivered, p.store.max_charge_kw - action.store_charge_kw, room_kw)
        # Check source absorption for the extra charge.
        current_abs, _ = action_absorbed_and_work(action, p)
        factor = 1.0 - 1.0 / p.heat_pump.cop(p.sink_temperatures_c["store"])
        extra = min(extra, max(0.0, raw_capture_kw + p.buffer.max_discharge_kw - current_abs) / factor)
        action.store_charge_kw += extra
        delivered = action.hp_delivered_kw
        if delivered < minimum - 1e-8:
            return DispatchAction()
    action.hp_units_on = infer_units_on(action.hp_delivered_kw, p)
    return action


@dataclass
class HeuristicController:
    params: PlantParameters
    tuned: bool = True

    @property
    def name(self) -> str:
        return "tuned" if self.tuned else "naive"

    def act(self, step: int, state: PlantState, forecast: pd.DataFrame, realised_now: pd.Series) -> DispatchAction:
        p = self.params
        row = forecast.iloc[0]
        ambient = float(row["ambient_c"])
        grid = float(row["grid_available"]) >= 0.5
        tariff = p.tariffs.grid_inr_per_kwh if grid else p.tariffs.diesel_inr_per_kwh
        raw_capture = p.capture_fraction * float(row["it_kw"])
        action = DispatchAction()

        dhw_need = max(0.0, float(row["dhw_demand_kw"]))
        process_need = max(0.0, float(row["process_demand_kw"]))
        store_loss_kw = p.store.standing_loss_kw(state.store_energy_kwh, p.timestep_hours)
        available_store_kw = min(
            p.store.max_discharge_kw,
            max(0.0, (state.store_energy_kwh - p.store.min_soc * p.store.capacity_kwh) / p.timestep_hours - store_loss_kw),
        )
        if (not self.tuned) or (not grid):
            action.store_discharge_dhw_kw = min(dhw_need, available_store_kw)
            dhw_need -= action.store_discharge_dhw_kw
            available_store_kw -= action.store_discharge_dhw_kw
            action.store_discharge_process_kw = min(process_need, available_store_kw)
            process_need -= action.store_discharge_process_kw

        remaining_output = p.heat_pump.rated_output_kw
        remaining_source = raw_capture + min(p.buffer.max_discharge_kw, state.buffer_energy_kwh / p.timestep_hours)
        source_credit = _source_credit_per_absorbed_kwh(ambient, raw_capture, p, tariff)

        # SOC limits apply after standing loss. At the lower bound, a small
        # maintenance charge is required even if no discretionary charging is
        # economic; reserve it first so saturated demand cannot crowd it out.
        if action.store_discharge_kw <= 1e-10:
            lower_bound = p.store.min_soc * p.store.capacity_kwh
            minimum_charge = max(
                0.0,
                p.store.standing_loss_kw(state.store_energy_kwh, p.timestep_hours)
                - (state.store_energy_kwh - lower_bound) / p.timestep_hours,
            )
            remaining_output, remaining_source = _greedy_add(
                action, "store_charge_kw", minimum_charge, remaining_output, remaining_source, "store", p
            )

        pathways: list[tuple[str, str, float, str, float]] = []
        # tuple field, sink, requested heat input, label, net value per heat input
        for field, sink, requested in (
            ("dhw_heat_kw", "dhw", dhw_need),
            ("process_heat_kw", "process", process_need),
        ):
            cop = p.heat_pump.cop(p.sink_temperatures_c[sink])
            absorbed_factor = 1.0 - 1.0 / cop
            value = p.tariffs.displaced_lpg_inr_per_kwh_th + source_credit * absorbed_factor - tariff / cop
            pathways.append((field, sink, requested, sink, value))

        absorption_cop = p.absorption.cop(ambient)
        absorption_request = min(p.absorption.rated_cooling_kw, float(row["cooling_demand_kw"])) / max(absorption_cop, 1e-12)
        hp_cop_abs = p.heat_pump.cop(p.sink_temperatures_c["absorption"])
        abs_factor = 1.0 - 1.0 / hp_cop_abs
        absorption_value = tariff * absorption_cop / p.mechanical_chiller.cop(ambient) + source_credit * abs_factor - tariff / hp_cop_abs
        pathways.append(("absorption_heat_kw", "absorption", absorption_request, "absorption", absorption_value))

        orc_eff = p.orc.efficiency(ambient)
        hp_cop_orc = p.heat_pump.cop(p.sink_temperatures_c["orc"])
        orc_factor = 1.0 - 1.0 / hp_cop_orc
        orc_value = p.tariffs.orc_export_inr_per_kwh * orc_eff + source_credit * orc_factor - tariff / hp_cop_orc

        if self.tuned:
            pathways.sort(key=lambda item: item[4], reverse=True)
        for field, sink, request, _label, value in pathways:
            if self.tuned and value <= 0.0:
                continue
            remaining_output, remaining_source = _greedy_add(action, field, request, remaining_output, remaining_source, sink, p)

        # Store only when it has plausible grid-to-outage arbitrage value. The
        # naive controller charges whenever capacity remains, matching fixed priority.
        room_kw = min(
            p.store.max_charge_kw,
            max(0.0, (p.store.max_soc * p.store.capacity_kwh - state.store_energy_kwh) / p.timestep_hours),
        ) - action.store_charge_kw
        should_charge = (not self.tuned) or (grid and state.store_energy_kwh < 0.65 * p.store.capacity_kwh)
        if should_charge and action.store_discharge_kw <= 1e-10:
            remaining_output, remaining_source = _greedy_add(action, "store_charge_kw", room_kw, remaining_output, remaining_source, "store", p)

        if (not self.tuned) or orc_value > 0.0:
            remaining_output, remaining_source = _greedy_add(action, "orc_heat_kw", p.orc.max_heat_input_kw, remaining_output, remaining_source, "orc", p)

        return _make_dispatch_feasible(action, state, p, raw_capture)


class MPCController:
    name = "mpc"

    def __init__(
        self,
        params: PlantParameters,
        horizon: int = 48,
        replan_steps: int = 2,
        solver_time_limit_s: float | None = 60.0,
        gap_abs_inr: float = 50.0,
    ):
        self.params = params
        self.horizon = horizon
        self.replan_steps = replan_steps
        self.solver_time_limit_s = solver_time_limit_s
        # CBC stops once the incumbent is within this many INR of the best
        # bound for the 12 h horizon objective. Proving exact optimality took
        # minutes on some steps; 50 INR is ~0.03% of a day's operating cost.
        self.gap_abs_inr = gap_abs_inr
        self.solves = 0
        self.time_limited_solves = 0
        self._plan: dict[int, DispatchAction] = {}

    def act(self, step: int, state: PlantState, forecast: pd.DataFrame, realised_now: pd.Series) -> DispatchAction:
        if step in self._plan:
            return self._plan.pop(step)
        actions = self._optimise(state, forecast.iloc[: self.horizon])
        self._plan = {step + i: action for i, action in enumerate(actions[: self.replan_steps])}
        return self._plan.pop(step)

    def _optimise(self, state: PlantState, forecast: pd.DataFrame) -> list[DispatchAction]:
        p = self.params
        n = len(forecast)
        model = pulp.LpProblem("trof_mpc", pulp.LpMinimize)
        paths = ("dhw", "process", "absorption", "orc", "store")
        q = pulp.LpVariable.dicts("q", (range(n), paths), lowBound=0.0)
        units_on = pulp.LpVariable.dicts("units_on", range(n), lowBound=0, upBound=p.heat_pump.units, cat="Integer")
        discharge_dhw = pulp.LpVariable.dicts("discharge_dhw", range(n), lowBound=0.0)
        discharge_process = pulp.LpVariable.dicts("discharge_process", range(n), lowBound=0.0)
        store_mode = pulp.LpVariable.dicts("store_charge_mode", range(n), cat="Binary")
        store_e = pulp.LpVariable.dicts(
            "store_e", range(n + 1),
            lowBound=p.store.min_soc * p.store.capacity_kwh,
            upBound=p.store.max_soc * p.store.capacity_kwh,
        )
        unmet_dhw = pulp.LpVariable.dicts("unmet_dhw", range(n), lowBound=0.0)
        unmet_process = pulp.LpVariable.dicts("unmet_process", range(n), lowBound=0.0)
        unmet_cooling = pulp.LpVariable.dicts("unmet_cooling", range(n), lowBound=0.0)
        grid_power = pulp.LpVariable.dicts("grid_power", range(n), lowBound=0.0)
        diesel_power = pulp.LpVariable.dicts("diesel_power", range(n), lowBound=0.0)
        # Normalize sub-micro-kWh CBC drift at a bound. This changes only the
        # optimizer's initial condition; physical balances retain the raw state.
        initial_store = min(
            p.store.max_soc * p.store.capacity_kwh,
            max(p.store.min_soc * p.store.capacity_kwh, state.store_energy_kwh),
        )
        model += store_e[0] == initial_store
        objective: list[pulp.LpAffineExpression] = []
        dt = p.timestep_hours
        store_loss_rate = (1.0 - (1.0 - p.store.standing_loss_per_day) ** (dt / 24.0)) / dt

        for h, (_, row) in enumerate(forecast.iterrows()):
            ambient = float(row["ambient_c"])
            cops = {path: p.heat_pump.cop(p.sink_temperatures_c[path]) for path in paths}
            absorbed = pulp.lpSum(q[h][path] * (1.0 - 1.0 / cops[path]) for path in paths)
            compressor = pulp.lpSum(q[h][path] / cops[path] for path in paths)
            delivered = pulp.lpSum(q[h][path] for path in paths)
            # Identical modules share load freely, so n modules on can deliver
            # any output in [n*turndown*rated, n*rated]. An integer module
            # count is therefore equivalent to per-module binaries and avoids
            # their symmetric branching.
            model += delivered <= p.heat_pump.rated_output_kw_per_unit * units_on[h]
            model += delivered >= p.heat_pump.min_turndown * p.heat_pump.rated_output_kw_per_unit * units_on[h]
            capture = p.capture_fraction * float(row["it_kw"])
            model += absorbed <= capture
            model += q[h]["orc"] <= p.orc.max_heat_input_kw
            abs_cop = p.absorption.cop(ambient)
            model += q[h]["absorption"] * abs_cop <= p.absorption.rated_cooling_kw
            model += q[h]["dhw"] + discharge_dhw[h] + unmet_dhw[h] == float(row["dhw_demand_kw"])
            model += q[h]["process"] + discharge_process[h] + unmet_process[h] == float(row["process_demand_kw"])
            model += q[h]["absorption"] * abs_cop + unmet_cooling[h] == float(row["cooling_demand_kw"])
            total_discharge = discharge_dhw[h] + discharge_process[h]
            model += q[h]["store"] <= p.store.max_charge_kw * store_mode[h]
            model += total_discharge <= p.store.max_discharge_kw * (1.0 - store_mode[h])
            model += store_e[h + 1] == store_e[h] + dt * (q[h]["store"] - total_discharge - store_loss_rate * store_e[h])

            model += grid_power[h] + diesel_power[h] == compressor
            roster_grid = float(row["grid_available"])
            model += grid_power[h] <= roster_grid * p.heat_pump.rated_output_kw
            model += diesel_power[h] <= (1.0 - roster_grid) * p.diesel_generator_limit_kw

            period_tariff = p.tariffs.grid_inr_per_kwh if roster_grid >= 0.5 else p.tariffs.diesel_inr_per_kwh
            source_credit = _source_credit_per_absorbed_kwh(ambient, capture, p, period_tariff)
            cooling_fallback_electric = unmet_cooling[h] / p.mechanical_chiller.cop(ambient)
            orc_power = q[h]["orc"] * p.orc.efficiency(ambient)
            objective.append(dt * (
                p.tariffs.grid_inr_per_kwh * grid_power[h]
                + p.tariffs.diesel_inr_per_kwh * diesel_power[h]
                + p.tariffs.displaced_lpg_inr_per_kwh_th * (unmet_dhw[h] + unmet_process[h])
                + period_tariff * cooling_fallback_electric
                - source_credit * absorbed
                - p.tariffs.orc_export_inr_per_kwh * orc_power
            ))

        # Modest terminal value prevents horizon-end dumping while valuing useful heat.
        objective.append(-0.25 * p.tariffs.displaced_lpg_inr_per_kwh_th * store_e[n])
        model += pulp.lpSum(objective)
        solver = pulp.PULP_CBC_CMD(msg=False, threads=1, timeLimit=self.solver_time_limit_s, gapAbs=self.gap_abs_inr)
        model.solve(solver)
        # PuLP reports "Optimal" even when CBC stops on the time limit, so the
        # solution status is checked directly: 1 = within gap, 2 = feasible
        # incumbent at the time limit (counted and reported), else failure.
        if model.sol_status not in (pulp.LpSolutionOptimal, pulp.LpSolutionIntegerFeasible):
            raise RuntimeError(f"MPC solve failed: solution status {model.sol_status}")
        self.solves += 1
        if model.sol_status == pulp.LpSolutionIntegerFeasible:
            self.time_limited_solves += 1

        actions: list[DispatchAction] = []
        for h in range(n):
            # CBC integrality tolerance can leave a mode binary at ~1e-7, which
            # permits a sub-milliwatt flow on the excluded side; round the mode.
            charging = (pulp.value(store_mode[h]) or 0.0) > 0.5
            action = DispatchAction(
                dhw_heat_kw=max(0.0, pulp.value(q[h]["dhw"]) or 0.0),
                process_heat_kw=max(0.0, pulp.value(q[h]["process"]) or 0.0),
                absorption_heat_kw=max(0.0, pulp.value(q[h]["absorption"]) or 0.0),
                orc_heat_kw=max(0.0, pulp.value(q[h]["orc"]) or 0.0),
                store_charge_kw=max(0.0, pulp.value(q[h]["store"]) or 0.0) if charging else 0.0,
                store_discharge_dhw_kw=0.0 if charging else max(0.0, pulp.value(discharge_dhw[h]) or 0.0),
                store_discharge_process_kw=0.0 if charging else max(0.0, pulp.value(discharge_process[h]) or 0.0),
                hp_units_on=int(round(pulp.value(units_on[h]) or 0.0)),
            )
            actions.append(action)
        return actions


def build_controller(name: str, params: PlantParameters, **kwargs: object) -> Controller:
    if name == "mpc":
        return MPCController(params, **kwargs)
    if name == "tuned":
        return HeuristicController(params, tuned=True)
    if name == "naive":
        return HeuristicController(params, tuned=False)
    raise ValueError(f"unknown controller: {name}")
