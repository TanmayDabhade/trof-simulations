"""Quasi-steady physical component models used by TROF.

All power values are kW, energy values are kWh, temperatures supplied to the
public methods are degrees Celsius, and prices are INR/kWh unless noted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Iterable

import numpy as np
import pandas as pd


KELVIN_OFFSET = 273.15  # SI temperature conversion, exact by definition.


@dataclass(frozen=True)
class HeatPumpBank:
    # Four 500 kW_th modules are the study design assumption.
    units: int = 4
    rated_output_kw_per_unit: float = 500.0
    min_turndown: float = 0.30
    # Arpagaus et al. (2018), Energy 152:985-1010: 0.40-0.55.
    second_law_efficiency: float = 0.45
    # Manufacturer catalogues generally do not claim HTHP COP above 6.
    cop_cap: float = 6.0
    # Study design: evaporator approach below the 35 C source return.
    evaporator_approach_k: float = 5.0
    # Study design: condenser approach above the required sink temperature.
    condenser_approach_k: float = 5.0
    source_inlet_c: float = 45.0

    @property
    def rated_output_kw(self) -> float:
        return self.units * self.rated_output_kw_per_unit

    def cop(self, sink_temperature_c: float) -> float:
        # The evaporator receives the 45 C capture-loop supply; the separate
        # dry-cooler threshold is referenced to the 35 C loop return.
        t_evap = self.source_inlet_c - self.evaporator_approach_k + KELVIN_OFFSET
        t_cond = sink_temperature_c + self.condenser_approach_k + KELVIN_OFFSET
        if t_cond <= t_evap:
            return self.cop_cap
        value = self.second_law_efficiency * t_cond / (t_cond - t_evap)
        return float(min(value, self.cop_cap))

    def compressor_kw(self, delivered_kw: float, sink_temperature_c: float) -> float:
        return max(0.0, delivered_kw) / self.cop(sink_temperature_c)

    def absorbed_kw(self, delivered_kw: float, sink_temperature_c: float) -> float:
        delivered_kw = max(0.0, delivered_kw)
        return delivered_kw * (1.0 - 1.0 / self.cop(sink_temperature_c))


@dataclass(frozen=True)
class AbsorptionChiller:
    # Study design rating; Kim & Infante Ferreira (2008) covers this class.
    rated_cooling_kw: float = 700.0
    # Kim & Infante Ferreira (2008), Int J Refrig 31:3-15: COP 0.6-0.8.
    nominal_cop_at_30c: float = 0.72
    # Study assumption introduced to represent hot-ambient performance loss.
    ambient_derate_per_k: float = 0.008
    # Broad physically admissible clamp around the cited operating range.
    cop_min: float = 0.45
    cop_max: float = 0.80
    # Kim & Infante Ferreira (2008): single-effect generator range 75-95 C.
    minimum_generator_c: float = 75.0

    def cop(self, ambient_c: float, generator_inlet_c: float = 85.0) -> float:
        if generator_inlet_c < self.minimum_generator_c:
            return 0.0
        value = self.nominal_cop_at_30c - self.ambient_derate_per_k * (ambient_c - 30.0)
        return float(np.clip(value, self.cop_min, self.cop_max))

    def cooling_kw(self, heat_input_kw: float, ambient_c: float) -> float:
        return min(self.rated_cooling_kw, max(0.0, heat_input_kw) * self.cop(ambient_c))


@dataclass(frozen=True)
class OrganicRankineCycle:
    # Study design size.
    max_heat_input_kw: float = 800.0
    # Quoilin et al. (2013), Renew Sustain Energy Rev 22:168-186 reports
    # roughly 0.45-0.60 of Carnot for small low-temperature ORCs.
    second_law_efficiency: float = 0.55
    # Quoilin et al. (2013): pumps/fans commonly consume 5-15% of gross.
    parasitic_fraction: float = 0.12
    # Study design for an air-cooled condenser.
    condenser_approach_k: float = 15.0
    evaporator_c: float = 95.0

    def efficiency(self, ambient_c: float, evaporator_c: float | None = None) -> float:
        t_evap = (self.evaporator_c if evaporator_c is None else evaporator_c) + KELVIN_OFFSET
        t_cond = ambient_c + self.condenser_approach_k + KELVIN_OFFSET
        if t_evap <= t_cond:
            return 0.0
        carnot = 1.0 - t_cond / t_evap
        return max(0.0, self.second_law_efficiency * carnot * (1.0 - self.parasitic_fraction))

    def net_power_kw(self, heat_input_kw: float, ambient_c: float) -> float:
        return min(max(0.0, heat_input_kw), self.max_heat_input_kw) * self.efficiency(ambient_c)


@dataclass(frozen=True)
class ThermalStore:
    # Study design values, comparable to commercial stratified hot-water tanks.
    capacity_kwh: float = 2000.0
    max_charge_kw: float = 500.0
    max_discharge_kw: float = 500.0
    min_soc: float = 0.10
    max_soc: float = 0.95
    # IEA ECES Annex 30 hot-water tank studies report about 0.5-3%/day.
    standing_loss_per_day: float = 0.015

    def standing_loss_kw(self, energy_kwh: float, dt_hours: float) -> float:
        retained = (1.0 - self.standing_loss_per_day) ** (dt_hours / 24.0)
        return max(0.0, energy_kwh) * (1.0 - retained) / dt_hours


@dataclass(frozen=True)
class BufferTank:
    # Study design for a short-duration capture-loop hydraulic buffer.
    capacity_kwh: float = 250.0
    max_charge_kw: float = 500.0
    max_discharge_kw: float = 500.0
    min_soc: float = 0.0
    max_soc: float = 1.0
    # IEA ECES Annex 30 range for small, relatively high-area tanks.
    standing_loss_per_day: float = 0.03

    def standing_loss_kw(self, energy_kwh: float, dt_hours: float) -> float:
        retained = (1.0 - self.standing_loss_per_day) ** (dt_hours / 24.0)
        return max(0.0, energy_kwh) * (1.0 - retained) / dt_hours


@dataclass(frozen=True)
class MechanicalChiller:
    # ASHRAE Handbook HVAC Systems and Equipment (2020), water-cooled
    # centrifugal systems: full-load COP is commonly about 4.5-6.5.
    nominal_cop: float = 5.5
    # Study assumption for hot-ambient condenser derating.
    ambient_derate_per_k: float = 0.020
    # ASHRAE operational lower bound used for extreme ambient conditions.
    minimum_cop: float = 2.8

    def cop(self, ambient_c: float) -> float:
        return max(self.nominal_cop * (1.0 - self.ambient_derate_per_k * (ambient_c - 25.0)), self.minimum_cop)


@dataclass(frozen=True)
class DryCooler:
    # Study design based on a packaged data-centre dry-cooler bank.
    capacity_kw: float = 4000.0
    # ASHRAE liquid-cooling guidance: free cooling below loop return approach.
    approach_k: float = 10.0
    loop_return_c: float = 35.0
    # Eurovent dry-cooler catalogue range is approximately 1-3% of heat duty.
    fan_fraction: float = 0.018

    def duty_kw(self, heat_rejection_kw: float, ambient_c: float) -> float:
        if ambient_c > self.loop_return_c - self.approach_k:
            return 0.0
        return min(max(0.0, heat_rejection_kw), self.capacity_kw)


@dataclass(frozen=True)
class Tariffs:
    # Maharashtra/Madhya Pradesh/Tamil Nadu HT industrial tariffs cluster near
    # INR 7-9/kWh; INR 8 is the stated cross-site design assumption.
    grid_inr_per_kwh: float = 8.0
    # Indian diesel generation estimates commonly span INR 18-25/kWh.
    diesel_inr_per_kwh: float = 22.0
    # 75 INR/kg LPG and 80% boiler efficiency, stated study assumption.
    displaced_lpg_inr_per_kwh_th: float = 6.9
    # Indian distributed renewable export tariffs span roughly INR 3-5/kWh.
    orc_export_inr_per_kwh: float = 4.5


@dataclass(frozen=True)
class PlantParameters:
    # Facility and capture fractions are study design assumptions.
    it_capacity_kw: float = 5000.0
    capture_fraction: float = 0.70
    timestep_hours: float = 0.25
    # Published outage rosters imply zero grid import; diesel remains available.
    grid_import_limit_outage_kw: float = 0.0
    # Study design for the standby generator plant.
    diesel_generator_limit_kw: float = 2500.0
    balance_tolerance_kw: float = 1e-9
    sink_temperatures_c: dict[str, float] = field(default_factory=lambda: {
        "dhw": 70.0,
        "process": 85.0,
        "absorption": 85.0,
        "orc": 95.0,
        "store": 70.0,
    })
    heat_pump: HeatPumpBank = field(default_factory=HeatPumpBank)
    absorption: AbsorptionChiller = field(default_factory=AbsorptionChiller)
    orc: OrganicRankineCycle = field(default_factory=OrganicRankineCycle)
    store: ThermalStore = field(default_factory=ThermalStore)
    buffer: BufferTank = field(default_factory=BufferTank)
    mechanical_chiller: MechanicalChiller = field(default_factory=MechanicalChiller)
    dry_cooler: DryCooler = field(default_factory=DryCooler)
    tariffs: Tariffs = field(default_factory=Tariffs)

    def validate(self) -> None:
        if not 0.55 <= self.capture_fraction <= 0.80:
            raise ValueError("capture_fraction must remain inside the stated 0.55-0.80 sensitivity range")
        if not 0.40 <= self.heat_pump.second_law_efficiency <= 0.55:
            raise ValueError("heat-pump second-law efficiency is outside the cited range")
        if not 0.60 <= self.absorption.nominal_cop_at_30c <= 0.80:
            raise ValueError("absorption nominal COP is outside the cited range")
        if not 0.45 <= self.orc.second_law_efficiency <= 0.60:
            raise ValueError("ORC second-law efficiency is outside the cited range")
        if not 0.005 <= self.store.standing_loss_per_day <= 0.03:
            raise ValueError("store loss is outside the sensitivity range")
        if not 4.5 <= self.mechanical_chiller.nominal_cop <= 6.5:
            raise ValueError("mechanical-chiller nominal COP is outside the cited range")


def component_parameter_table(params: PlantParameters | None = None) -> pd.DataFrame:
    """Return the paper-ready parameter appendix with source text."""
    p = params or PlantParameters()
    rows: list[dict[str, object]] = []

    def add(component: str, parameter: str, value: object, unit: str, source: str) -> None:
        rows.append({"component": component, "parameter": parameter, "value": value, "unit": unit, "source": source})

    add("Site", "IT capacity", p.it_capacity_kw, "kW", "Study design: 5 MW colocation facility")
    add("Site", "Capture fraction", p.capture_fraction, "fraction", "Study design sensitivity range 0.55-0.80")
    add("Heat pump", "Units", p.heat_pump.units, "count", "Study design")
    add("Heat pump", "Rated output per unit", p.heat_pump.rated_output_kw_per_unit, "kW_th", "Study design")
    add("Heat pump", "Minimum turndown", p.heat_pump.min_turndown, "fraction", "Study design")
    add("Heat pump", "Second-law efficiency", p.heat_pump.second_law_efficiency, "fraction", "Arpagaus et al. (2018), Energy 152:985-1010")
    add("Heat pump", "COP cap", p.heat_pump.cop_cap, "-", "Commercial HTHP manufacturer catalogue convention")
    add("Absorption chiller", "Rated cooling", p.absorption.rated_cooling_kw, "kW_cool", "Study design")
    add("Absorption chiller", "Nominal COP", p.absorption.nominal_cop_at_30c, "-", "Kim & Infante Ferreira (2008), Int J Refrig 31:3-15")
    add("ORC", "Maximum heat input", p.orc.max_heat_input_kw, "kW_th", "Study design")
    add("ORC", "Second-law efficiency", p.orc.second_law_efficiency, "fraction Carnot", "Quoilin et al. (2013), RSER 22:168-186")
    add("ORC", "Parasitic fraction", p.orc.parasitic_fraction, "fraction gross", "Quoilin et al. (2013), RSER 22:168-186")
    add("Thermal store", "Capacity", p.store.capacity_kwh, "kWh", "Study design")
    add("Thermal store", "Standing loss", p.store.standing_loss_per_day, "fraction/day", "IEA ECES Annex 30 hot-water storage range")
    add("Buffer tank", "Capacity", p.buffer.capacity_kwh, "kWh", "Study design")
    add("Buffer tank", "Standing loss", p.buffer.standing_loss_per_day, "fraction/day", "IEA ECES Annex 30 small-tank range")
    add("Mechanical chiller", "Nominal COP", p.mechanical_chiller.nominal_cop, "-", "ASHRAE Handbook HVAC Systems and Equipment (2020)")
    add("Dry cooler", "Capacity", p.dry_cooler.capacity_kw, "kW", "Study design")
    add("Dry cooler", "Fan fraction", p.dry_cooler.fan_fraction, "fraction duty", "Eurovent dry-cooler catalogue range")
    add("Tariff", "Grid HT industrial", p.tariffs.grid_inr_per_kwh, "INR/kWh", "Cross-site tariff design assumption")
    add("Tariff", "Diesel generation", p.tariffs.diesel_inr_per_kwh, "INR/kWh", "Indian diesel-generation cost range")
    add("Tariff", "Displaced LPG", p.tariffs.displaced_lpg_inr_per_kwh_th, "INR/kWh_th", "75 INR/kg LPG at 80% boiler efficiency")
    add("Tariff", "ORC export", p.tariffs.orc_export_inr_per_kwh, "INR/kWh", "Indian distributed export tariff range")
    return pd.DataFrame(rows)
