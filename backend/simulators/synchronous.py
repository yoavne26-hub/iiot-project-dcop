"""Synchronous simulator for DCOP algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
import time

from backend.algorithms.base import DistributedAlgorithm
from backend.dcop.cost import calculate_global_cost
from backend.dcop.problem import DCOPProblem


@dataclass
class SimulationRunResult:
    """Result returned by a synchronous DCOP simulation run."""

    simulator: str
    algorithm: str
    iterations: int
    cost_history: list[int]
    final_assignment: dict[int, int]
    total_messages: int
    runtime_seconds: float
    metadata: dict[str, object] = field(default_factory=dict)


class SynchronousSimulator:
    """Run a distributed algorithm in fixed synchronous iterations."""

    def __init__(
        self,
        problem: DCOPProblem,
        algorithm: DistributedAlgorithm,
        iterations: int,
        seed: int | None = None,
    ) -> None:
        if iterations <= 0:
            raise ValueError("iterations must be greater than 0.")

        self.problem = problem
        self.algorithm = algorithm
        self.iterations = iterations
        self.seed = seed

    def run(self) -> SimulationRunResult:
        """Run the configured algorithm and collect cost history."""

        started_at = time.perf_counter()
        self.algorithm.initialize(
            self.problem,
            self.problem.initial_assignment,
            seed=self.seed,
        )

        cost_history: list[int] = []
        total_messages = 0

        for _ in range(self.iterations):
            step_result = self.algorithm.run_synchronous_iteration()
            assignment = self.algorithm.get_assignment()
            cost_history.append(calculate_global_cost(self.problem, assignment))
            total_messages += step_result.messages_sent

        runtime_seconds = time.perf_counter() - started_at

        return SimulationRunResult(
            simulator="synchronous",
            algorithm=self.algorithm.name,
            iterations=self.iterations,
            cost_history=cost_history,
            final_assignment=self.algorithm.get_assignment(),
            total_messages=total_messages,
            runtime_seconds=runtime_seconds,
            metadata={
                "seed": self.seed,
            },
        )
