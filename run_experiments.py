"""Minimal CLI runner for DCOP smoke experiments."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from backend.config import (
    DEFAULT_AGENTS,
    DEFAULT_CONSTRAINT_PROBABILITY,
    DEFAULT_DMS_DAMPING,
    DEFAULT_DOMAIN_SIZE,
    DEFAULT_DSA_PROBABILITY,
    DEFAULT_ITERATIONS,
    DEFAULT_MAX_COST,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PLOT_INTERVAL,
    DEFAULT_PROBLEMS,
    DEFAULT_SEED,
)
from backend.algorithms.dms import DMSAlgorithm
from backend.algorithms.dsa_c import DSACAlgorithm
from backend.algorithms.mgm import MGMAlgorithm
from backend.algorithms.mgm2 import MGM2Algorithm
from backend.dcop.cost import calculate_global_cost
from backend.dcop.generator import generate_random_dcop
from backend.experiments.plotting import write_average_cost_plots
from backend.experiments.results import write_experiment_csvs
from backend.experiments.runner import (
    SUPPORTED_ALGORITHMS,
    SUPPORTED_SIMULATORS,
    ExperimentConfig,
    parse_name_list,
    run_full_experiment,
)
from backend.simulators.asynchronous import AsynchronousSimulator
from backend.simulators.synchronous import SynchronousSimulator


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run one generated DCOP problem with a selected algorithm."
    )
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--agents", type=int, default=DEFAULT_AGENTS)
    parser.add_argument("--domain-size", type=int, default=DEFAULT_DOMAIN_SIZE)
    parser.add_argument("--max-cost", type=int, default=DEFAULT_MAX_COST)
    parser.add_argument(
        "--constraint-probability",
        type=float,
        default=DEFAULT_CONSTRAINT_PROBABILITY,
    )
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dsa-probability", type=float, default=DEFAULT_DSA_PROBABILITY)
    parser.add_argument(
        "--algorithm",
        choices=("dsa-c", "mgm", "mgm2", "dms"),
        default="dsa-c",
    )
    parser.add_argument("--simulator", choices=("sync", "async"), default="sync")
    parser.add_argument("--dms-damping", type=float, default=DEFAULT_DMS_DAMPING)
    parser.add_argument("--problems", type=int, default=DEFAULT_PROBLEMS)
    parser.add_argument("--algorithms", default="dsa-c,mgm,mgm2,dms")
    parser.add_argument("--simulators", default="sync,async")
    parser.add_argument("--plot-interval", type=int, default=DEFAULT_PLOT_INTERVAL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--output",
        default=None,
        help="Path for iteration,cost CSV output.",
    )
    return parser.parse_args()


def build_algorithm(
    args: argparse.Namespace,
) -> DMSAlgorithm | DSACAlgorithm | MGMAlgorithm | MGM2Algorithm:
    """Build the selected algorithm."""

    if args.algorithm == "dsa-c":
        return DSACAlgorithm(
            probability=args.dsa_probability,
            seed=args.seed,
        )
    if args.algorithm == "mgm":
        return MGMAlgorithm(seed=args.seed)
    if args.algorithm == "mgm2":
        return MGM2Algorithm(seed=args.seed)
    if args.algorithm == "dms":
        return DMSAlgorithm(
            damping=args.dms_damping,
            seed=args.seed,
        )

    raise ValueError(f"Unsupported algorithm: {args.algorithm}.")


def write_cost_history_csv(path: Path, cost_history: list[int]) -> None:
    """Write one cost value per iteration."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["iteration", "cost"])
        for index, cost in enumerate(cost_history, start=1):
            writer.writerow([index, cost])


def main() -> None:
    """Run smoke or full experiment mode."""

    args = parse_args()
    if args.mode == "full":
        run_full_mode(args)
    else:
        run_smoke_mode(args)


def run_smoke_mode(args: argparse.Namespace) -> None:
    """Run a single algorithm/simulator smoke experiment."""

    problem = generate_random_dcop(
        num_agents=args.agents,
        constraint_probability=args.constraint_probability,
        domain_size=args.domain_size,
        max_cost=args.max_cost,
        seed=args.seed,
    )
    initial_cost = calculate_global_cost(problem, problem.initial_assignment)

    algorithm = build_algorithm(args)
    simulator_class = (
        SynchronousSimulator if args.simulator == "sync" else AsynchronousSimulator
    )
    simulator = simulator_class(
        problem=problem,
        algorithm=algorithm,
        iterations=args.iterations,
        seed=args.seed,
    )
    result = simulator.run()
    final_cost = result.cost_history[-1]
    best_cost = min(result.cost_history)

    algorithm_slug = args.algorithm.replace("-", "_")
    output_path = Path(args.output or f"results/{algorithm_slug}_{args.simulator}_smoke.csv")
    write_cost_history_csv(output_path, result.cost_history)

    print("Problem summary")
    print(f"  agents: {problem.num_agents}")
    print(f"  domain size: {problem.domain_size}")
    print(f"  constraints: {len(problem.constraints)}")
    print(f"  initial cost: {initial_cost}")
    print("Run summary")
    print(f"  algorithm: {args.algorithm}")
    print(f"  simulator: {args.simulator}")
    print(f"  final cost: {final_cost}")
    print(f"  best cost: {best_cost}")
    print(f"  total messages: {result.total_messages}")
    print(f"  runtime seconds: {result.runtime_seconds:.6f}")
    print(f"  csv: {output_path}")


def run_full_mode(args: argparse.Namespace) -> None:
    """Run a full multi-problem experiment batch."""

    algorithms = parse_name_list(args.algorithms, SUPPORTED_ALGORITHMS, "algorithms")
    simulators = parse_name_list(args.simulators, SUPPORTED_SIMULATORS, "simulators")
    config = ExperimentConfig(
        problems=args.problems,
        agents=args.agents,
        domain_size=args.domain_size,
        max_cost=args.max_cost,
        constraint_probability=args.constraint_probability,
        iterations=args.iterations,
        seed=args.seed,
        algorithms=algorithms,
        simulators=simulators,
        dsa_probability=args.dsa_probability,
        dms_damping=args.dms_damping,
        plot_interval=args.plot_interval,
        output_dir=Path(args.output_dir),
        progress_enabled=not args.no_progress,
    )

    print("Full experiment configuration")
    print(f"  problems: {config.problems}")
    print(f"  agents: {config.agents}")
    print(f"  domain size: {config.domain_size}")
    print(f"  max cost: {config.max_cost}")
    print(f"  constraint probability: {config.constraint_probability}")
    print(f"  iterations: {config.iterations}")
    print(f"  plot interval: {config.plot_interval}")
    print(f"  seed: {config.seed}")
    print(f"  algorithms: {', '.join(config.algorithms)}")
    print(f"  simulators: {', '.join(config.simulators)}")

    output = run_full_experiment(config)
    csv_paths = write_experiment_csvs(output)
    print("CSV outputs")
    for label, path in csv_paths.items():
        print(f"  {label}: {path}")

    if args.no_plots:
        print("Plots skipped because --no-plots was passed.")
    else:
        try:
            plot_paths = write_average_cost_plots(
                average_rows=output.average_rows,
                output_dir=config.output_dir,
                simulators=config.simulators,
            )
        except RuntimeError as error:
            print(error)
        else:
            print("Plot outputs")
            for simulator_name, path in plot_paths.items():
                print(f"  {simulator_name}: {path}")

    print("Run summary")
    print(f"  raw rows: {len(output.raw_rows)}")
    print(f"  summary rows: {len(output.summary_rows)}")
    print(f"  average rows: {len(output.average_rows)}")


if __name__ == "__main__":
    main()
