"""Single-timestep plant advance with mandatory energy-balance assertions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil

from .components import PlantParameters


@dataclass
class PlantState:
    store_energy_kwh: float
    buffer_energy_kwh: float

    @classmethod
    def initial(cls, params: PlantParameters) -> "PlantState":
        return cls(
            store_energy_kwh=0.50 * params.store.capacity_kwh,
            buffer_energy_kwh=0.50 * params.buffer.capacity_kwh,
        )


@dataclass
class DispatchAction:
    dhw_heat_kw: float = 0.0
    process_heat_kw: float = 0.0
    absorption_heat_kw: float = 0.0
    orc_heat_kw: float = 0.0
    store_charge_kw: float = 0.0
    store_discharge_dhw_kw: float = 0.0
    store_discharge_process_kw: float = 0.0
    hp_units_on: int = 0

    @property
    def hp_delivered_kw(self) -> float:
        return self.dhw_heat_kw + self.process_heat_kw + self.absorption_heat_kw + self.orc_heat_kw + self.store_charge_kw

    @property
    def store_discharge_kw(self) -> float:
        return self.store_discharge_dhw_kw + self.store_discharge_process_kw


def action_absorbed_and_work(action: DispatchAction, params: PlantParameters) -> tuple[float, float]:
    absorbed = 0.0
    work = 0.0
    for name, delivered in (
        ("dhw", action.dhw_heat_kw),
        ("process", action.process_heat_kw),
        ("absorption", action.absorption_heat_kw),
        ("orc", action.orc_heat_kw),
        ("store", action.store_charge_kw),
    ):
        sink = params.sink_temperatures_c[name]
        absorbed += params.heat_pump.absorbed_kw(delivered, sink)
        work += params.heat_pump.compressor_kw(delivered, sink)
    return absorbed, work


def infer_units_on(delivered_kw: float, params: PlantParameters) -> int:
    if delivered_kw <= 1e-12:
        return 0
    return min(params.heat_pump.units, max(1, ceil(delivered_kw / params.heat_pump.rated_output_kw_per_unit - 1e-12)))


def _assert_residual(name: str, residual: float, tolerance: float) -> None:
    if abs(residual) > tolerance:
        raise RuntimeError(f"{name} energy balance failed: residual={residual:.16g} kW")


def advance_plant(
    state: PlantState,
    action: DispatchAction,
    realised: dict[str, float],
    params: PlantParameters,
) -> tuple[PlantState, dict[str, float]]:
    """Advance the physical plant by one 15-minute interval.

    The capture-loop buffer is upstream of the required source-side accounting
    boundary. ``captured_kw`` is therefore gross capture plus buffer discharge
    minus buffer charge, and is asserted to equal HP absorption plus rejection.
    ``site_captured_kw`` remains the gross recovered stream used in denominator
    reporting. Buffer terms are separately logged for full audit closure.
    """
    params.validate()
    dt = params.timestep_hours
    tol = params.balance_tolerance_kw
    constraint_tol = 1e-4  # CBC feasibility tolerance; balance assertions remain 1e-9.
    action = replace(action, **{
        key: max(0.0, float(getattr(action, key)))
        for key in (
            "dhw_heat_kw", "process_heat_kw", "absorption_heat_kw", "orc_heat_kw",
            "store_charge_kw", "store_discharge_dhw_kw", "store_discharge_process_kw",
        )
    })

    # The absorption chiller's local capacity control throttles generator heat
    # to its rating at the realised ambient. Dispatch is sized on forecast
    # ambient, so a warmer-than-forecast step would otherwise overfeed it.
    absorption_limit_kw = params.absorption.rated_cooling_kw / max(params.absorption.cop(float(realised["ambient_c"])), 1e-12)
    if action.absorption_heat_kw > absorption_limit_kw:
        action = replace(action, absorption_heat_kw=absorption_limit_kw)

    delivered = action.hp_delivered_kw
    units_on = int(action.hp_units_on)
    if units_on < 0 or units_on > params.heat_pump.units:
        raise ValueError("invalid number of heat-pump units")
    if delivered > tol:
        if units_on == 0:
            raise ValueError("positive heat-pump delivery requires a unit to be on")
        minimum = units_on * params.heat_pump.min_turndown * params.heat_pump.rated_output_kw_per_unit
        maximum = units_on * params.heat_pump.rated_output_kw_per_unit
        if delivered < minimum - constraint_tol or delivered > maximum + constraint_tol:
            raise ValueError(f"heat-pump output {delivered} is outside [{minimum}, {maximum}] for {units_on} units")
    elif units_on != 0:
        raise ValueError("heat-pump units cannot be on at zero output")

    if action.absorption_heat_kw * params.absorption.cop(float(realised["ambient_c"])) > params.absorption.rated_cooling_kw + constraint_tol:
        raise ValueError("absorption chiller rating exceeded")
    if action.orc_heat_kw > params.orc.max_heat_input_kw + constraint_tol:
        raise ValueError("ORC heat-input rating exceeded")
    if action.store_charge_kw > params.store.max_charge_kw + constraint_tol or action.store_discharge_kw > params.store.max_discharge_kw + constraint_tol:
        raise ValueError("thermal store power rating exceeded")
    if action.store_charge_kw > tol and action.store_discharge_kw > tol:
        raise ValueError("simultaneous store charge and discharge is forbidden")

    absorbed_kw, compressor_kw = action_absorbed_and_work(action, params)
    hp_residual_kw = delivered - absorbed_kw - compressor_kw
    _assert_residual("heat_pump", hp_residual_kw, tol)

    # Service-delivery boundary. Dispatch is sized on forecast demand, so the
    # store discharge valve passes only the realised shortfall left after
    # direct heat-pump supply; the undischarged energy stays in the store.
    # Heat-pump heat above realised demand cannot reach a sink and is logged
    # as dumped rather than silently dropped.
    dhw_demand_kw = float(realised["dhw_demand_kw"])
    process_demand_kw = float(realised["process_demand_kw"])
    store_discharge_dhw_kw = min(action.store_discharge_dhw_kw, max(0.0, dhw_demand_kw - action.dhw_heat_kw))
    store_discharge_process_kw = min(action.store_discharge_process_kw, max(0.0, process_demand_kw - action.process_heat_kw))
    store_discharge_kw = store_discharge_dhw_kw + store_discharge_process_kw

    # Main thermal store: losses are evaluated from the beginning-of-step state.
    store_loss_kw = params.store.standing_loss_kw(state.store_energy_kwh, dt)
    next_store = state.store_energy_kwh + dt * (action.store_charge_kw - store_discharge_kw - store_loss_kw)
    store_min = params.store.min_soc * params.store.capacity_kwh
    store_max = params.store.max_soc * params.store.capacity_kwh
    if next_store < store_min - constraint_tol or next_store > store_max + constraint_tol:
        raise ValueError("thermal store SOC bound exceeded")
    store_residual_kw = (next_store - state.store_energy_kwh) / dt - action.store_charge_kw + store_discharge_kw + store_loss_kw
    _assert_residual("store", store_residual_kw, tol)

    # Capture buffer moves only the excess/deficit around the source boundary.
    raw_capture_kw = max(0.0, float(realised["site_captured_kw"]))
    buffer_loss_kw = params.buffer.standing_loss_kw(state.buffer_energy_kwh, dt)
    post_loss_buffer = max(0.0, state.buffer_energy_kwh - dt * buffer_loss_kw)
    buffer_min = params.buffer.min_soc * params.buffer.capacity_kwh
    buffer_max = params.buffer.max_soc * params.buffer.capacity_kwh
    buffer_charge_kw = 0.0
    buffer_discharge_kw = 0.0
    if absorbed_kw <= raw_capture_kw:
        surplus = raw_capture_kw - absorbed_kw
        room_kw = max(0.0, (buffer_max - post_loss_buffer) / dt)
        buffer_charge_kw = min(surplus, params.buffer.max_charge_kw, room_kw)
    else:
        need = absorbed_kw - raw_capture_kw
        available_kw = max(0.0, (post_loss_buffer - buffer_min) / dt)
        buffer_discharge_kw = min(need, params.buffer.max_discharge_kw, available_kw)
        if buffer_discharge_kw < need - tol:
            raise ValueError("heat-pump absorption exceeds capture loop availability")
    next_buffer = post_loss_buffer + dt * (buffer_charge_kw - buffer_discharge_kw)
    buffer_residual_kw = (next_buffer - state.buffer_energy_kwh) / dt - buffer_charge_kw + buffer_discharge_kw + buffer_loss_kw
    _assert_residual("buffer", buffer_residual_kw, tol)

    captured_boundary_kw = raw_capture_kw + buffer_discharge_kw - buffer_charge_kw
    rejected_kw = max(0.0, captured_boundary_kw - absorbed_kw)
    source_residual_kw = captured_boundary_kw - absorbed_kw - rejected_kw
    _assert_residual("source", source_residual_kw, tol)

    ambient = float(realised["ambient_c"])
    absorption_cooling_kw = params.absorption.cooling_kw(action.absorption_heat_kw, ambient)
    orc_electric_kw = params.orc.net_power_kw(action.orc_heat_kw, ambient)
    dhw_served_kw = min(dhw_demand_kw, action.dhw_heat_kw + store_discharge_dhw_kw)
    process_served_kw = min(process_demand_kw, action.process_heat_kw + store_discharge_process_kw)
    cooling_served_kw = min(float(realised["cooling_demand_kw"]), absorption_cooling_kw)
    dumped_heat_kw = max(0.0, action.dhw_heat_kw - dhw_demand_kw) + max(0.0, action.process_heat_kw - process_demand_kw)
    surplus_cooling_kw = absorption_cooling_kw - cooling_served_kw
    service_residual_kw = (
        action.dhw_heat_kw + action.process_heat_kw + store_discharge_kw
        - dhw_served_kw - process_served_kw - dumped_heat_kw
    )
    _assert_residual("service", service_residual_kw, tol)
    fallback_dhw_kw = max(0.0, float(realised["dhw_demand_kw"]) - dhw_served_kw)
    fallback_process_kw = max(0.0, float(realised["process_demand_kw"]) - process_served_kw)
    fallback_cooling_kw = max(0.0, float(realised["cooling_demand_kw"]) - cooling_served_kw)
    # The counterfactual LPG boilers and mechanical chiller have sufficient
    # capacity, so demand not met by TROF is fallback-supplied, not shed.
    unmet_dhw_kw = 0.0
    unmet_process_kw = 0.0
    unmet_cooling_kw = 0.0

    # Counterfactual and actual data-centre rejection electricity.
    baseline_dry_kw = params.dry_cooler.duty_kw(raw_capture_kw, ambient)
    baseline_mech_heat_kw = max(0.0, raw_capture_kw - baseline_dry_kw)
    baseline_rejection_electric_kw = baseline_dry_kw * params.dry_cooler.fan_fraction + baseline_mech_heat_kw / params.mechanical_chiller.cop(ambient)
    actual_dry_kw = params.dry_cooler.duty_kw(rejected_kw, ambient)
    actual_mech_heat_kw = max(0.0, rejected_kw - actual_dry_kw)
    actual_rejection_electric_kw = actual_dry_kw * params.dry_cooler.fan_fraction + actual_mech_heat_kw / params.mechanical_chiller.cop(ambient)
    cooling_electric_saving_kw = baseline_rejection_electric_kw - actual_rejection_electric_kw

    community_cooling_electric_kw = fallback_cooling_kw / params.mechanical_chiller.cop(ambient)
    gross_electric_load_kw = compressor_kw + actual_rejection_electric_kw + community_cooling_electric_kw
    grid_available = float(realised["grid_available"]) >= 0.5
    grid_electric_kw = gross_electric_load_kw if grid_available else 0.0
    diesel_electric_kw = 0.0 if grid_available else gross_electric_load_kw
    if diesel_electric_kw > params.diesel_generator_limit_kw + tol:
        raise ValueError("diesel generator capacity exceeded")
    electric_cost_inr_per_h = (
        grid_electric_kw * params.tariffs.grid_inr_per_kwh
        + diesel_electric_kw * params.tariffs.diesel_inr_per_kwh
    )
    fallback_lpg_kw = fallback_dhw_kw + fallback_process_kw
    fallback_cost_inr_per_h = fallback_lpg_kw * params.tariffs.displaced_lpg_inr_per_kwh_th
    orc_credit_inr_per_h = orc_electric_kw * params.tariffs.orc_export_inr_per_kwh
    net_cost_inr = dt * (electric_cost_inr_per_h + fallback_cost_inr_per_h - orc_credit_inr_per_h)

    row = {
        "site_captured_kw": raw_capture_kw,
        "captured_kw": captured_boundary_kw,
        "hp_absorbed_kw": absorbed_kw,
        "hp_compressor_kw": compressor_kw,
        "hp_delivered_kw": delivered,
        "hp_units_on": units_on,
        "dhw_heat_kw": action.dhw_heat_kw,
        "process_heat_kw": action.process_heat_kw,
        "absorption_heat_kw": action.absorption_heat_kw,
        "orc_heat_kw": action.orc_heat_kw,
        "store_charge_kw": action.store_charge_kw,
        "store_discharge_kw": store_discharge_kw,
        "store_discharge_dhw_kw": store_discharge_dhw_kw,
        "store_discharge_process_kw": store_discharge_process_kw,
        "store_discharge_commanded_kw": action.store_discharge_kw,
        "dumped_heat_kw": dumped_heat_kw,
        "surplus_cooling_kw": surplus_cooling_kw,
        "store_energy_kwh": next_store,
        "store_soc": next_store / params.store.capacity_kwh,
        "store_standing_loss_kw": store_loss_kw,
        "buffer_charge_kw": buffer_charge_kw,
        "buffer_discharge_kw": buffer_discharge_kw,
        "buffer_energy_kwh": next_buffer,
        "buffer_standing_loss_kw": buffer_loss_kw,
        "rejected_kw": rejected_kw,
        "absorption_cooling_kw": absorption_cooling_kw,
        "orc_electric_kw": orc_electric_kw,
        "dhw_served_kw": dhw_served_kw,
        "process_served_kw": process_served_kw,
        "cooling_served_kw": cooling_served_kw,
        "unmet_dhw_kw": unmet_dhw_kw,
        "unmet_process_kw": unmet_process_kw,
        "unmet_cooling_kw": unmet_cooling_kw,
        "fallback_dhw_kw": fallback_dhw_kw,
        "fallback_process_kw": fallback_process_kw,
        "fallback_cooling_kw": fallback_cooling_kw,
        "baseline_rejection_electric_kw": baseline_rejection_electric_kw,
        "actual_rejection_electric_kw": actual_rejection_electric_kw,
        "cooling_electric_saving_kw": cooling_electric_saving_kw,
        "community_cooling_electric_kw": community_cooling_electric_kw,
        "grid_electric_kw": grid_electric_kw,
        "diesel_electric_kw": diesel_electric_kw,
        "net_cost_inr": net_cost_inr,
        "source_balance_residual_kw": source_residual_kw,
        "hp_balance_residual_kw": hp_residual_kw,
        "store_balance_residual_kw": store_residual_kw,
        "buffer_balance_residual_kw": buffer_residual_kw,
        "service_balance_residual_kw": service_residual_kw,
    }
    return PlantState(next_store, next_buffer), row
