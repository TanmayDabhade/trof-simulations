import numpy as np

from trof.components import PlantParameters
from trof.plant import DispatchAction, PlantState, advance_plant, infer_units_on
from trof.run import RunSpec, run_closed_loop


RESIDUALS = [
    "source_balance_residual_kw",
    "hp_balance_residual_kw",
    "store_balance_residual_kw",
    "buffer_balance_residual_kw",
]


def test_balances_hold_across_full_thirty_day_run() -> None:
    result, summary = run_closed_loop(RunSpec("A", "tuned", seed=19, days=30))
    assert len(result) == 2880
    assert result[RESIDUALS].abs().to_numpy().max() <= 1e-9
    assert summary["max_balance_residual_kw"] <= 1e-9


def test_balances_hold_under_randomised_valid_inputs() -> None:
    rng = np.random.default_rng(93)
    params = PlantParameters()
    state = PlantState.initial(params)
    for _ in range(250):
        output = float(rng.uniform(150.0, 500.0))
        action = DispatchAction(dhw_heat_kw=output, hp_units_on=infer_units_on(output, params))
        state, row = advance_plant(state, action, {
            "ambient_c": float(rng.uniform(18.0, 44.0)),
            "site_captured_kw": float(rng.uniform(2500.0, 3500.0)),
            "dhw_demand_kw": output,
            "process_demand_kw": 0.0,
            "cooling_demand_kw": 0.0,
            "grid_available": float(rng.integers(0, 2)),
        }, params)
        assert max(abs(row[column]) for column in RESIDUALS) <= 1e-9
