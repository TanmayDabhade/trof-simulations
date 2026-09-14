"""CI helpers: count pending runs and merge shard registries into results/."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

from trof import study


def pending(mc_draws: int, phases: str) -> int:
    completed = set(study._load_registry().get("run_id", pd.Series(dtype=str)).astype(str))
    queue = study.build_run_queue(5, mc_draws, 30, set(phases.split(",")))
    return sum(spec.run_id not in completed for spec, _ in queue)


def merge(shard_root: Path) -> None:
    frames = [study._load_registry()]
    for registry in sorted(shard_root.rglob("run_registry.csv")):
        frames.append(pd.read_csv(registry))
        timeseries = registry.parent / "timeseries"
        if timeseries.is_dir():
            study.TIMESERIES.mkdir(parents=True, exist_ok=True)
            for file in timeseries.glob("*.csv"):
                shutil.copy2(file, study.TIMESERIES / file.name)
    failures = [pd.read_csv(path) for path in sorted(shard_root.rglob("failed_runs.csv"))]
    if failures:
        pd.concat(failures, ignore_index=True).drop_duplicates("run_id").to_csv(study.FAILED_RUNS, index=False)
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return
    merged = pd.concat(frames, ignore_index=True).drop_duplicates("run_id", keep="first")
    merged.to_csv(study.REGISTRY, index=False)
    study.aggregate_tables()


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "pending":
        print(pending(int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "main,sweep,mc,ablation"))
    elif command == "merge":
        merge(Path(sys.argv[2]))
    else:
        raise SystemExit(f"unknown command: {command}")
