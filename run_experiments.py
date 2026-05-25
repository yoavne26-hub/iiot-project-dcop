"""Minimal CLI runner for synchronous DSA-C smoke experiments."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from backend.algorithms.dsa_c import DSACAlgorithm
from backend.dcop.cost import calculate_global_cost
from backend.dcop.generator import generate_random_dcop
from backend.simulators.synchronous import SynchronousSimulator


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run one generated DCOP problem with synchronous DSA-C."
    )
    parser.add_argument("--agents", type=int, default=50)
    parser.add_argument("--domain-size", type=int, default=10)
    parser.add_argument("--max-cost", type=int, default=100)
    parser.add_argument("--constraint-probability", type=float, default=0.3)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dsa-probability", type=float, default=0.75)
    parser.add_argument(
        "--output",
        default="results/dsa_c_sync_smoke.csv",
        help="Path for iteration,cost CSV output.",
    )
    return parser.parse_args()


def write_cost_history_csv(path: Path, cost_history: list[int]) -> None:
    """Write one cost value per iteration."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["iteration", "cost"])
        for index, cost in enumerate(cost_history, start=1):
            writer.writerow([index, cost])


def main() -> None:
    """Run a single synchronous DSA-C experiment."""

    args = parse_args()
    problem = generate_random_dcop(
        num_agents=args.agents,
        constraint_probability=args.constraint_probability,
        domain_size=args.domain_size,
        max_cost=args.max_cost,
        seed=args.seed,
    )
    initial_cost = calculate_global_cost(problem, problem.initial_assignment)

    algorithm = DSACAlgorithm(
        probability=args.dsa_probability,
        seed=args.seed,
    )
    simulator = SynchronousSimulator(
        problem=problem,
        algorithm=algorithm,
        iterations=args.iterations,
        seed=args.seed,
    )
    result = simulator.run()
    final_cost = result.cost_history[-1]
    best_cost = min(result.cost_history)

    output_path = Path(args.output)
    write_cost_history_csv(output_path, result.cost_history)

    print("Problem summary")
    print(f"  agents: {problem.num_agents}")
    print(f"  domain size: {problem.domain_size}")
    print(f"  constraints: {len(problem.constraints)}")
    print(f"  initial cost: {initial_cost}")
    print("Run summary")
    print(f"  final cost: {final_cost}")
    print(f"  best cost: {best_cost}")
    print(f"  total messages: {result.total_messages}")
    print(f"  runtime seconds: {result.runtime_seconds:.6f}")
    print(f"  csv: {output_path}")


if __name__ == "__main__":
    main()
