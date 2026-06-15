"""DSA-C implementation for binary DCOPs."""

from __future__ import annotations

import random

from backend.algorithms.base import (
    AlgorithmMessage,
    AlgorithmStepResult,
    DistributedAlgorithm,
)
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
        self._local_views: dict[int, dict[int, int]] = {}

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
        self._local_views = {
            agent_id: {} for agent_id in range(problem.num_agents)
        }
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

    def initial_async_messages(self) -> list[AlgorithmMessage]:
        """Broadcast every agent's initial value to its neighbors."""

        problem = self._require_problem()
        messages: list[AlgorithmMessage] = []

        for agent_id in range(problem.num_agents):
            messages.extend(self._build_value_messages(agent_id))

        return messages

    def handle_async_message(self, message: AlgorithmMessage) -> AlgorithmStepResult:
        """Update a receiver's local view from one neighbor value message."""

        problem = self._require_problem()
        if message.kind != "value":
            raise ValueError(f"Unsupported DSA-C message kind: {message.kind}.")
        if message.receiver not in self._local_views:
            raise ValueError(f"Invalid message receiver: {message.receiver}.")
        if message.sender not in problem.neighbors[message.receiver]:
            raise ValueError(
                f"Agent {message.sender} is not a neighbor of agent {message.receiver}."
            )

        value = message.payload.get("value")
        if not isinstance(value, int) or not 0 <= value < problem.domain_size:
            raise ValueError("DSA-C value messages must contain an in-domain integer value.")

        self._local_views[message.receiver][message.sender] = value
        return AlgorithmStepResult(changed_agents=set(), messages_sent=0)

    def on_async_activation(self, agent_id: int) -> AlgorithmStepResult:
        """Run one asynchronous DSA-C decision for a single agent."""

        problem = self._require_problem()
        if not 0 <= agent_id < problem.num_agents:
            raise ValueError(f"agent_id must be in 0..{problem.num_agents - 1}.")

        current_value = self._assignment[agent_id]
        current_cost = self._calculate_async_local_cost(agent_id, current_value)
        best_value = current_value
        best_cost = current_cost

        for candidate_value in range(problem.domain_size):
            candidate_cost = self._calculate_async_local_cost(agent_id, candidate_value)
            if candidate_cost < best_cost:
                best_value = candidate_value
                best_cost = candidate_cost

        changed_agents: set[int] = set()
        if best_cost < current_cost and self._rng.random() < self.probability:
            self._assignment[agent_id] = best_value
            changed_agents.add(agent_id)

        messages = self._build_value_messages(agent_id)
        return AlgorithmStepResult(
            changed_agents=changed_agents,
            messages_sent=len(messages),
            metadata={
                "messages": messages,
            },
        )

    def _require_problem(self) -> DCOPProblem:
        """Return the initialized problem or fail clearly."""

        if self._problem is None:
            raise RuntimeError("DSACAlgorithm must be initialized before running.")
        return self._problem

    def _build_value_messages(self, agent_id: int) -> list[AlgorithmMessage]:
        """Build current-value messages from one agent to all neighbors."""

        problem = self._require_problem()
        value = self._assignment[agent_id]
        return [
            AlgorithmMessage(
                sender=agent_id,
                receiver=neighbor_id,
                kind="value",
                payload={
                    "value": value,
                },
            )
            for neighbor_id in problem.neighbors[agent_id]
        ]

    def _calculate_async_local_cost(self, agent_id: int, value: int) -> int:
        """Calculate local cost using the agent's current neighbor view."""

        problem = self._require_problem()
        total = 0

        for constraint in problem.constraints.values():
            if agent_id == constraint.agent_a:
                neighbor_id = constraint.agent_b
                neighbor_value = self._known_neighbor_value(agent_id, neighbor_id)
                total += constraint.cost(value, neighbor_value)
            elif agent_id == constraint.agent_b:
                neighbor_id = constraint.agent_a
                neighbor_value = self._known_neighbor_value(agent_id, neighbor_id)
                total += constraint.cost(neighbor_value, value)

        return total

    def _known_neighbor_value(self, agent_id: int, neighbor_id: int) -> int:
        """Return a locally known neighbor value, falling back to current assignment."""

        return self._local_views[agent_id].get(neighbor_id, self._assignment[neighbor_id])
