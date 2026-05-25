"""DSA-C implementation for binary DCOPs."""

from __future__ import annotations

import random

from backend.algorithms.base import AlgorithmStepResult, DistributedAlgorithm
from backend.dcop.cost import calculate_local_cost
from backend.dcop.problem import DCOPProblem


class DSACAlgorithm(DistributedAlgorithm):
    """Distributed Stochastic Algorithm, variant C."""

    name = "dsa-c"

    def __init__(self, probability: float = 0.75, seed: int | None = None) -> None:
        if not 0 <= probability <= 1:
            raise ValueError("probability must be between 0 and 1.")

        self.probability = probability
        self.seed = seed
        self._rng = random.Random(seed)
        self._problem: DCOPProblem | None = None
        self._assignment: dict[int, int] = {}

    def initialize(
        self,
        problem: DCOPProblem,
        assignment: dict[int, int],
        seed: int | None = None,
    ) -> None:
        """Initialize DSA-C state from the provided assignment."""

        problem.validate_assignment(assignment)
        rng_seed = self.seed if seed is None else seed

        self._problem = problem
        self._assignment = dict(assignment)
        self._rng = random.Random(rng_seed)

    def run_synchronous_iteration(self) -> AlgorithmStepResult:
        """Run one simultaneous DSA-C update over all agents."""

        if self._problem is None:
            raise RuntimeError("DSACAlgorithm must be initialized before running.")

        starting_assignment = dict(self._assignment)
        selected_changes: dict[int, int] = {}

        for agent_id in range(self._problem.num_agents):
            current_value = starting_assignment[agent_id]
            current_cost = calculate_local_cost(
                self._problem,
                starting_assignment,
                agent_id,
            )
            best_value = current_value
            best_cost = current_cost

            for candidate_value in range(self._problem.domain_size):
                candidate_cost = calculate_local_cost(
                    self._problem,
                    starting_assignment,
                    agent_id,
                    value=candidate_value,
                )
                if candidate_cost < best_cost:
                    best_value = candidate_value
                    best_cost = candidate_cost

            if best_cost < current_cost and self._rng.random() < self.probability:
                selected_changes[agent_id] = best_value

        if selected_changes:
            updated_assignment = dict(starting_assignment)
            updated_assignment.update(selected_changes)
            self._assignment = updated_assignment

        messages_sent = sum(
            len(neighbors) for neighbors in self._problem.neighbors.values()
        )
        return AlgorithmStepResult(
            changed_agents=set(selected_changes),
            messages_sent=messages_sent,
            metadata={
                "candidate_changes": len(selected_changes),
            },
        )

    def get_assignment(self) -> dict[int, int]:
        """Return a copy of the current assignment."""

        return dict(self._assignment)
