"""Full experiment orchestration for DCOP algorithms."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

from backend.algorithms.dms import DMSAlgorithm
from backend.algorithms.dsa_c import DSACAlgorithm
from backend.algorithms.mgm import MGMAlgorithm
from backend.algorithms.mgm2 import MGM2Algorithm
from backend.dcop.cost import calculate_global_cost
from backend.dcop.generator import generate_random_dcop
from backend.dcop.problem import DCOPProblem
from backend.experiments.results import (
    AverageCostRow,
    ExperimentOutput,
    RawRunRow,
    SummaryRow,
)
from backend.simulators.asynchronous import AsynchronousSimulator
from backend.simulators.synchronous import SynchronousSimulator


SUPPORTED_ALGORITHMS = ("dsa-c", "mgm", "mgm2", "dms")
SUPPORTED_SIMULATORS = ("sync", "async")


@dataclass
class ExperimentConfig:
    """Configuration for a full DCOP experiment batch."""

    problems: int
    agents: int
    domain_size: int
    max_cost: int
    constraint_probability: float
    iterations: int
    seed: int
    algorithms: tuple[str, ...]
    simulators: tuple[str, ...]
    dsa_probability: float
    dms_damping: float
    output_dir: Path


def stable_run_seed(
    master_seed: int,
    problem_index: int,
    algorithm: str,
    simulator: str,
) -> int:
    """Return a deterministic seed that does not depend on Python hash randomization."""

    key = f"{master_seed}:{problem_index}:{algorithm}:{simulator}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def build_algorithm(
    algorithm_name: str,
    dsa_probability: float,
    dms_damping: float,
    seed: int,
) -> DMSAlgorithm | DSACAlgorithm | MGMAlgorithm | MGM2Algorithm:
    """Build one algorithm instance for one run."""

    if algorithm_name == "dsa-c":
        return DSACAlgorithm(probability=dsa_probability, seed=seed)
    if algorithm_name == "mgm":
        return MGMAlgorithm(seed=seed)
    if algorithm_name == "mgm2":
        return MGM2Algorithm(seed=seed)
    if algorithm_name == "dms":
        return DMSAlgorithm(damping=dms_damping, seed=seed)

    raise ValueError(f"Unsupported algorithm: {algorithm_name}.")


def build_simulator(
    simulator_name: str,
    problem: DCOPProblem,
    algorithm: DMSAlgorithm | DSACAlgorithm | MGMAlgorithm | MGM2Algorithm,
    iterations: int,
    seed: int,
) -> SynchronousSimulator | AsynchronousSimulator:
    """Build one simulator instance for one run."""

    if simulator_name == "sync":
        return SynchronousSimulator(
            problem=problem,
            algorithm=algorithm,
            iterations=iterations,
            seed=seed,
        )
    if simulator_name == "async":
        return AsynchronousSimulator(
            problem=problem,
            algorithm=algorithm,
            iterations=iterations,
            seed=seed,
        )

    raise ValueError(f"Unsupported simulator: {simulator_name}.")


def parse_name_list(raw_value: str, supported: tuple[str, ...], label: str) -> tuple[str, ...]:
    """Parse a comma-separated CLI list and validate supported names."""

    names = tuple(name.strip() for name in raw_value.split(",") if name.strip())
    if not names:
        raise ValueError(f"{label} must contain at least one value.")

    invalid = [name for name in names if name not in supported]
    if invalid:
        raise ValueError(
            f"Unsupported {label}: {invalid}. Supported values: {', '.join(supported)}."
        )

    return names


def generate_problem_set(config: ExperimentConfig) -> list[tuple[int, DCOPProblem]]:
    """Generate all problems once so every algorithm and simulator reuses them."""

    generated: list[tuple[int, DCOPProblem]] = []
    for problem_index in range(config.problems):
        problem_seed = config.seed + problem_index
        problem = generate_random_dcop(
            num_agents=config.agents,
            constraint_probability=config.constraint_probability,
            domain_size=config.domain_size,
            max_cost=config.max_cost,
            seed=problem_seed,
        )
        generated.append((problem_seed, problem))

    return generated


def run_full_experiment(config: ExperimentConfig) -> ExperimentOutput:
    """Run all selected algorithms and simulators over the generated problem set."""

    if config.problems <= 0:
        raise ValueError("problems must be greater than 0.")

    problems = generate_problem_set(config)
    raw_rows: list[RawRunRow] = []
    summary_rows: list[SummaryRow] = []
    average_accumulator: dict[tuple[str, str], list[float]] = {}
    average_counts: dict[tuple[str, str], int] = {}

    for simulator_name in config.simulators:
        for algorithm_name in config.algorithms:
            accumulator_key = (simulator_name, algorithm_name)
            average_accumulator[accumulator_key] = [0.0] * config.iterations
            average_counts[accumulator_key] = 0

            for problem_index, (problem_seed, problem) in enumerate(problems):
                run_seed = stable_run_seed(
                    master_seed=config.seed,
                    problem_index=problem_index,
                    algorithm=algorithm_name,
                    simulator=simulator_name,
                )
                algorithm = build_algorithm(
                    algorithm_name=algorithm_name,
                    dsa_probability=config.dsa_probability,
                    dms_damping=config.dms_damping,
                    seed=run_seed,
                )
                simulator = build_simulator(
                    simulator_name=simulator_name,
                    problem=problem,
                    algorithm=algorithm,
                    iterations=config.iterations,
                    seed=run_seed,
                )
                result = simulator.run()
                initial_cost = calculate_global_cost(problem, problem.initial_assignment)

                for iteration, cost in enumerate(result.cost_history, start=1):
                    raw_rows.append(
                        RawRunRow(
                            simulator=simulator_name,
                            algorithm=algorithm_name,
                            problem_index=problem_index,
                            problem_seed=problem_seed,
                            iteration=iteration,
                            cost=cost,
                        )
                    )
                    average_accumulator[accumulator_key][iteration - 1] += cost

                average_counts[accumulator_key] += 1
                summary_rows.append(
                    SummaryRow(
                        simulator=simulator_name,
                        algorithm=algorithm_name,
                        problem_index=problem_index,
                        problem_seed=problem_seed,
                        initial_cost=initial_cost,
                        final_cost=result.cost_history[-1],
                        best_cost=min(result.cost_history),
                        total_messages=result.total_messages,
                        runtime_seconds=result.runtime_seconds,
                    )
                )

    average_rows = _build_average_rows(average_accumulator, average_counts)
    return ExperimentOutput(
        raw_rows=raw_rows,
        summary_rows=summary_rows,
        average_rows=average_rows,
        output_dir=config.output_dir,
    )


def _build_average_rows(
    average_accumulator: dict[tuple[str, str], list[float]],
    average_counts: dict[tuple[str, str], int],
) -> list[AverageCostRow]:
    """Build average-cost rows from accumulated iteration sums."""

    rows: list[AverageCostRow] = []

    for simulator_name, algorithm_name in sorted(average_accumulator):
        count = average_counts[(simulator_name, algorithm_name)]
        for iteration, total_cost in enumerate(
            average_accumulator[(simulator_name, algorithm_name)],
            start=1,
        ):
            rows.append(
                AverageCostRow(
                    simulator=simulator_name,
                    algorithm=algorithm_name,
                    iteration=iteration,
                    average_cost=total_cost / count,
                )
            )

    return rows
