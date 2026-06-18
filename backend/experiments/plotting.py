"""Plot generation for DCOP experiment report graphs."""

from __future__ import annotations

from pathlib import Path

from backend.experiments.results import AverageCostRow


SIMULATOR_PLOT_NAMES = {
    "sync": "synchronous_average_cost.png",
    "async": "asynchronous_average_cost.png",
}

# Plot order and colors chosen to match the reference example graphs
# (matplotlib default cycle applied to DSA-C, MGM, MGM-2, DMS in that order).
ALGORITHM_PLOT_ORDER = ("dsa-c", "mgm", "mgm2", "dms")
ALGORITHM_COLORS = {
    "dsa-c": "#1f77b4",  # blue
    "mgm": "#ff7f0e",  # orange
    "mgm2": "#2ca02c",  # green
    "dms": "#d62728",  # red
}
ALGORITHM_MARKERS = {
    "dsa-c": "o",  # circle
    "dms": "s",  # square
    "mgm": "^",  # triangle
    "mgm2": "*",  # star
}


def write_average_cost_plots(
    average_rows: list[AverageCostRow],
    output_dir: Path,
    simulators: tuple[str, ...],
) -> dict[str, Path]:
    """Write one average-cost PNG plot per selected simulator."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "matplotlib is not installed; CSV files were written, but plots were skipped. "
            "Install matplotlib or run with --no-plots."
        ) from error

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    for simulator_name in simulators:
        rows_for_simulator = [
            row for row in average_rows if row.simulator == simulator_name
        ]
        if not rows_for_simulator:
            continue

        by_algorithm: dict[str, list[AverageCostRow]] = {}
        for row in rows_for_simulator:
            by_algorithm.setdefault(row.algorithm, []).append(row)

        plt.figure(figsize=(10, 6))
        # Iterate in the reference order first, then any remaining algorithms.
        ordered_names = [
            name for name in ALGORITHM_PLOT_ORDER if name in by_algorithm
        ] + [name for name in sorted(by_algorithm) if name not in ALGORITHM_PLOT_ORDER]
        for algorithm_name in ordered_names:
            algorithm_rows = sorted(
                by_algorithm[algorithm_name],
                key=lambda row: row.iteration,
            )
            plt.plot(
                [row.iteration for row in algorithm_rows],
                [row.average_cost for row in algorithm_rows],
                label=algorithm_name,
                color=ALGORITHM_COLORS.get(algorithm_name),
                marker=ALGORITHM_MARKERS.get(algorithm_name),
                markersize=5,
            )

        title = (
            "Synchronous Average Solution Cost"
            if simulator_name == "sync"
            else "Asynchronous Average Solution Cost"
        )
        plt.title(title)
        plt.xlabel("Iteration")
        plt.ylabel("Average Solution Cost")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = output_dir / SIMULATOR_PLOT_NAMES[simulator_name]
        plt.savefig(output_path, dpi=150)
        plt.close()
        paths[simulator_name] = output_path

    return paths
