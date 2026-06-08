"""Result records and CSV export helpers for DCOP experiments."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class RawRunRow:
    """One cost observation for one run iteration."""

    simulator: str
    algorithm: str
    problem_index: int
    problem_seed: int
    iteration: int
    cost: int


@dataclass
class SummaryRow:
    """One summary row for a completed algorithm/simulator/problem run."""

    simulator: str
    algorithm: str
    problem_index: int
    problem_seed: int
    initial_cost: int
    final_cost: int
    best_cost: int
    total_messages: int
    runtime_seconds: float


@dataclass
class AverageCostRow:
    """Average cost at one iteration for one algorithm and simulator."""

    simulator: str
    algorithm: str
    iteration: int
    average_cost: float


@dataclass
class ExperimentOutput:
    """All tabular outputs from a full experiment run."""

    raw_rows: list[RawRunRow]
    summary_rows: list[SummaryRow]
    average_rows: list[AverageCostRow]
    output_dir: Path


def write_rows_csv(path: Path, rows: list[object]) -> None:
    """Write dataclass rows to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}.")

    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_experiment_csvs(output: ExperimentOutput) -> dict[str, Path]:
    """Write raw, summary, and average cost CSV files."""

    paths = {
        "raw": output.output_dir / "raw_runs.csv",
        "summary": output.output_dir / "summary.csv",
        "average": output.output_dir / "average_costs.csv",
    }
    write_rows_csv(paths["raw"], output.raw_rows)
    write_rows_csv(paths["summary"], output.summary_rows)
    write_rows_csv(paths["average"], output.average_rows)
    paths.update(write_simulator_average_csvs(output))
    return paths


def write_simulator_average_csvs(output: ExperimentOutput) -> dict[str, Path]:
    """Write one wide average-cost CSV per simulator for plotting/report use."""

    simulator_labels = {
        "sync": "synchronous",
        "async": "asynchronous",
    }
    paths: dict[str, Path] = {}

    for simulator_name, simulator_label in simulator_labels.items():
        rows = [row for row in output.average_rows if row.simulator == simulator_name]
        if not rows:
            continue

        algorithms = sorted({row.algorithm for row in rows})
        iterations = sorted({row.iteration for row in rows})
        values = {
            (row.iteration, row.algorithm): row.average_cost
            for row in rows
        }
        path = output.output_dir / f"{simulator_label}_average_cost.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["iteration", *algorithms])
            for iteration in iterations:
                writer.writerow(
                    [
                        iteration,
                        *[
                            values.get((iteration, algorithm), "")
                            for algorithm in algorithms
                        ],
                    ]
                )
        paths[f"{simulator_name}_average_wide"] = path

    return paths
