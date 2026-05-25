"""Core data model for Distributed Constraint Optimization Problems."""

from __future__ import annotations

from dataclasses import dataclass


def _normalize_constraint_key(agent_a: int, agent_b: int) -> tuple[int, int]:
    """Return the normalized key for an undirected binary constraint."""

    return (min(agent_a, agent_b), max(agent_a, agent_b))


def _transpose_matrix(costs: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    """Return a transposed immutable cost matrix."""

    return tuple(tuple(row[index] for row in costs) for index in range(len(costs[0])))


@dataclass
class BinaryConstraint:
    """A binary cost table between two agents."""

    agent_a: int
    agent_b: int
    costs: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if self.agent_a == self.agent_b:
            raise ValueError("A binary constraint must connect two different agents.")
        if self.agent_a < 0 or self.agent_b < 0:
            raise ValueError("Constraint agents must be non-negative integers.")

        normalized_costs = self._validate_cost_matrix(self.costs)
        if self.agent_a > self.agent_b:
            original_a = self.agent_a
            object.__setattr__(self, "agent_a", self.agent_b)
            object.__setattr__(self, "agent_b", original_a)
            object.__setattr__(self, "costs", _transpose_matrix(normalized_costs))
        else:
            object.__setattr__(self, "costs", normalized_costs)

    @staticmethod
    def _validate_cost_matrix(costs: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
        """Validate and normalize a square cost matrix."""

        if not costs:
            raise ValueError("Constraint cost matrix must not be empty.")

        normalized = tuple(tuple(row) for row in costs)
        row_count = len(normalized)

        for row in normalized:
            if len(row) != row_count:
                raise ValueError("Constraint cost matrix must be square.")
            if any(not isinstance(cost, int) for cost in row):
                raise ValueError("Constraint costs must be integers.")

        return normalized

    def key(self) -> tuple[int, int]:
        """Return the normalized constraint key."""

        return self.agent_a, self.agent_b

    def cost(self, value_a: int, value_b: int) -> int:
        """Return the cost for values ordered as this constraint's agents."""

        return self.costs[value_a][value_b]


@dataclass
class DCOPProblem:
    """A complete DCOP instance with one variable per agent."""

    num_agents: int
    domain_size: int
    constraints: dict[tuple[int, int], BinaryConstraint]
    initial_assignment: dict[int, int]
    seed: int | None
    constraint_probability: float
    max_cost: int

    def __post_init__(self) -> None:
        if self.num_agents <= 0:
            raise ValueError("num_agents must be greater than 0.")
        if self.domain_size <= 0:
            raise ValueError("domain_size must be greater than 0.")
        if not 0 <= self.constraint_probability <= 1:
            raise ValueError("constraint_probability must be between 0 and 1.")
        if self.max_cost < 1:
            raise ValueError("max_cost must be at least 1.")

        self._validate_constraints()
        self.validate_assignment(self.initial_assignment)

    @property
    def domains(self) -> dict[int, tuple[int, ...]]:
        """Return each agent's finite domain."""

        domain = tuple(range(self.domain_size))
        return {agent_id: domain for agent_id in range(self.num_agents)}

    @property
    def neighbors(self) -> dict[int, tuple[int, ...]]:
        """Return sorted neighbors derived from the binary constraints."""

        neighbor_sets: dict[int, set[int]] = {
            agent_id: set() for agent_id in range(self.num_agents)
        }
        for agent_a, agent_b in self.constraints:
            neighbor_sets[agent_a].add(agent_b)
            neighbor_sets[agent_b].add(agent_a)

        return {
            agent_id: tuple(sorted(neighbors))
            for agent_id, neighbors in neighbor_sets.items()
        }

    def validate_assignment(self, assignment: dict[int, int]) -> None:
        """Validate that an assignment has exactly one in-domain value per agent."""

        expected_agents = set(range(self.num_agents))
        assigned_agents = set(assignment)
        if assigned_agents != expected_agents:
            missing = sorted(expected_agents - assigned_agents)
            extra = sorted(assigned_agents - expected_agents)
            raise ValueError(
                f"Assignment must contain exactly one value for every agent. "
                f"Missing: {missing}; extra: {extra}."
            )

        for agent_id, value in assignment.items():
            if not isinstance(value, int):
                raise ValueError(f"Assignment value for agent {agent_id} must be an integer.")
            if not 0 <= value < self.domain_size:
                raise ValueError(
                    f"Assignment value for agent {agent_id} must be in "
                    f"0..{self.domain_size - 1}."
                )

    def _validate_constraints(self) -> None:
        """Validate constraint keys, endpoints, and matrix dimensions."""

        for key, constraint in self.constraints.items():
            if len(key) != 2:
                raise ValueError("Constraint keys must contain exactly two agent IDs.")

            agent_a, agent_b = key
            if key != _normalize_constraint_key(agent_a, agent_b):
                raise ValueError(f"Constraint key {key} is not normalized.")
            if constraint.key() != key:
                raise ValueError(
                    f"Constraint stored at key {key} has internal key {constraint.key()}."
                )
            if not 0 <= agent_a < self.num_agents or not 0 <= agent_b < self.num_agents:
                raise ValueError(f"Constraint key {key} refers to an invalid agent.")
            if len(constraint.costs) != self.domain_size:
                raise ValueError(f"Constraint {key} has an invalid row count.")
            if any(len(row) != self.domain_size for row in constraint.costs):
                raise ValueError(f"Constraint {key} has an invalid column count.")
