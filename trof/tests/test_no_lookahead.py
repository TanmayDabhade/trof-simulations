from dataclasses import asdict

import numpy as np

from trof.components import PlantParameters
from trof.dispatch import HeuristicController
from trof.plant import PlantState, advance_plant
from trof.scenarios import CausalSeasonalNaive, SCENARIOS, generate_scenario


def _actions_until(frame, cutoff: int):
    params = PlantParameters()
    controller = HeuristicController(params, tuned=True)
    forecaster = CausalSeasonalNaive(frame)
    state = PlantState.initial(params)
    actions = []
    for step in range(cutoff + 1):
        realised = frame.iloc[step]
        forecast = forecaster.forecast(step, 48)
        action = controller.act(step, state, forecast, realised)
        actions.append(asdict(action))
        state, _ = advance_plant(state, action, {
            key: float(realised[key]) for key in (
                "ambient_c", "site_captured_kw", "dhw_demand_kw", "process_demand_kw",
                "cooling_demand_kw", "grid_available",
            )
        }, params)
    return actions


def test_future_realised_perturbations_cannot_change_past_actions() -> None:
    cutoff = 110
    original = generate_scenario("B", seed=7, days=3)
    perturbed = original.copy(deep=True)
    future = perturbed.index > cutoff
    for column in ("ambient_c", "it_kw", "site_captured_kw", "dhw_demand_kw", "process_demand_kw", "cooling_demand_kw"):
        perturbed.loc[future, column] = perturbed.loc[future, column] * 9.0 + 137.0
    perturbed.loc[future, "grid_available"] = 1.0 - perturbed.loc[future, "grid_available"]
    assert _actions_until(original, cutoff) == _actions_until(perturbed, cutoff)


def test_forecast_never_shares_realised_memory() -> None:
    frame = generate_scenario("C", seed=2, days=2)
    forecast = CausalSeasonalNaive(frame).forecast(50, 48)
    assert not np.shares_memory(frame["dhw_demand_kw"].to_numpy(), forecast["dhw_demand_kw"].to_numpy())


def test_published_rosters_match_modelled_grid_hours() -> None:
    for key, config in SCENARIOS.items():
        frame = generate_scenario(key, seed=1, days=1)
        assert frame.grid_roster_available.sum() / 4.0 == config.grid_hours_per_day
