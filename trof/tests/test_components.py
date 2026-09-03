from dataclasses import replace

import pytest

from trof.components import AbsorptionChiller, HeatPumpBank, OrganicRankineCycle, PlantParameters


def test_heat_pump_cop_falls_as_lift_rises() -> None:
    hp = HeatPumpBank()
    assert hp.cop(70.0) > hp.cop(85.0) > hp.cop(95.0)
    assert hp.cop(70.0) <= hp.cop_cap


def test_absorption_cop_falls_as_ambient_rises() -> None:
    chiller = AbsorptionChiller()
    assert chiller.cop(25.0) > chiller.cop(35.0) > chiller.cop(45.0)
    assert chiller.cop(30.0, generator_inlet_c=74.9) == 0.0
    assert chiller.cop(25.0) <= chiller.cop_max


def test_orc_zero_when_evaporator_is_not_hotter_than_condenser() -> None:
    orc = OrganicRankineCycle()
    assert orc.efficiency(ambient_c=30.0, evaporator_c=40.0) == 0.0
    assert 0.0 < orc.efficiency(ambient_c=30.0) < 1.0


def test_parameter_ranges_are_enforced() -> None:
    base = PlantParameters()
    with pytest.raises(ValueError):
        replace(base, heat_pump=replace(base.heat_pump, second_law_efficiency=0.7)).validate()
