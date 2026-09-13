"""CI helpers: count pending runs and merge shard registries into results/."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

from trof import study


def pending() -> int:
    completed = set(study._load_registry().get("run_id", pd.Series(dtype=str)).astype(str))
    queue = study.build_run_queue(5, 500, 30, {"main", "sweep", "mc", "ablation"})
    return sum(spec.run_id not in completed for spec, _ in queue)


def merge(shard_root: Path) -> None:
    frames = [study._load_registry()]
    for shard in sorted(shard_root.glob("*")):
        registry = shard / "run_registry.csv"
        if registry.exists():
            frames.append(pd.read_csv(registry))
        timeseries = shard / "timeseries"
        if timeseries.is_dir():
            study.TIMESERIES.mkdir(parents=True, exist_ok=True)
            for file in timeseries.glob("*.csv"):
                shutil.copy2(file, study.TIMESERIES / file.name)
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return
    merged = pd.concat(frames, ignore_index=True).drop_duplicates("run_id", keep="first")
    merged.to_csv(study.REGISTRY, index=False)
    study.aggregate_tables()


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "pending":
        print(pending())
    elif command == "merge":
        merge(Path(sys.argv[2]))
    else:
        raise SystemExit(f"unknown command: {command}")
