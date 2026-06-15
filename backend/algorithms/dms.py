"""Damped min-sum implementation for binary DCOPs."""

from __future__ import annotations

from backend.algorithms.base import (
    AlgorithmMessage,
    AlgorithmStepResult,
    DistributedAlgorithm,
)
from backend.dcop.problem import BinaryConstraint, DCOPProblem


MessageVector = tuple[float, ...]


class DMSAlgorithm(DistributedAlgorithm):
    """Max-sum with damping, implemented as min-sum for minimization."""

    name = "dms"

    def __init__(self, damping: float = 0.9, seed: int | None = None) -> None:
        if not 0 <= damping < 1:
            raise ValueError("damping must be greater than or equal to 0 and less than 1.")

        self.damping = damping
        self.seed = seed
        self._problem: DCOPProblem | None = None
        self._assignment: dict[int, int] = {}
        self._variable_to_factor: dict[tuple[int, tuple[int, int]], MessageVector] = {}
        self._factor_to_variable: dict[tuple[tuple[int, int], int], MessageVector] = {}

    def initialize(
        self,
        problem: DCOPProblem,
        assignment: dict[int, int],
        seed: int | None = None,
    ) -> None:
        """Initialize DMS messages and assignment."""

        problem.validate_assignment(assignment)
        self._problem = problem
        self._assignment = dict(assignment)

        zero_vector = self._zero_vector(problem.domain_size)
        self._variable_to_factor = {}
        self._factor_to_variable = {}

        for edge_key in sorted(problem.constraints):
            agent_a, agent_b = edge_key
            self._variable_to_factor[(agent_a, edge_key)] = zero_vector
            self._variable_to_factor[(agent_b, edge_key)] = zero_vector
            self._factor_to_variable[(edge_key, agent_a)] = zero_vector
            self._factor_to_variable[(edge_key, agent_b)] = zero_vector

    def run_synchronous_iteration(self) -> AlgorithmStepResult:
        """Run one synchronous damped min-sum iteration."""

        problem = self._require_problem()
        new_variable_to_factor = self._compute_all_variable_to_factor()
        new_factor_to_variable = self._compute_all_factor_to_variable(new_variable_to_factor)

        self._variable_to_factor = {
            key: self._damp_and_normalize(self._variable_to_factor[key], message)
            for key, message in new_variable_to_factor.items()
        }
        self._factor_to_variable = {
            key: self._damp_and_normalize(self._factor_to_variable[key], message)
            for key, message in new_factor_to_variable.items()
        }

        changed_agents = self._update_assignment_from_beliefs()
        edge_count = len(problem.constraints)
        return AlgorithmStepResult(
            changed_agents=changed_agents,
            messages_sent=4 * edge_count,
            metadata={
                "edge_count": edge_count,
            },
        )

    def get_assignment(self) -> dict[int, int]:
        """Return a copy of the current assignment."""

        return dict(self._assignment)

    def initial_async_messages(self) -> list[AlgorithmMessage]:
        """Seed the async simulator with zero vector messages on every edge."""

        problem = self._require_problem()
        messages: list[AlgorithmMessage] = []
        for edge_key in sorted(problem.constraints):
            agent_a, agent_b = edge_key
            messages.append(self._build_message(agent_a, agent_b, edge_key, "variable_to_factor"))
            messages.append(self._build_message(agent_b, agent_a, edge_key, "variable_to_factor"))

        return messages

    def handle_async_message(self, message: AlgorithmMessage) -> AlgorithmStepResult:
        """Apply one async DMS vector message."""

        problem = self._require_problem()
        edge_key = self._parse_edge_key(message.payload.get("edge"))
        vector = self._parse_vector(message.payload.get("vector"))

        if edge_key not in problem.constraints:
            raise ValueError(f"Unknown DMS edge key: {edge_key}.")
        if message.sender not in edge_key or message.receiver not in edge_key:
            raise ValueError("DMS messages must stay on the constraint edge.")

        kind = message.kind
        if kind == "variable_to_factor":
            self._variable_to_factor[(message.sender, edge_key)] = vector
        elif kind == "factor_to_variable":
            self._factor_to_variable[(edge_key, message.receiver)] = vector
        else:
            raise ValueError(f"Unsupported DMS message kind: {kind}.")

        return AlgorithmStepResult(changed_agents=set(), messages_sent=0)

    def on_async_activation(self, agent_id: int) -> AlgorithmStepResult:
        """Run one deterministic async DMS activation for an agent."""

        problem = self._require_problem()
        if not 0 <= agent_id < problem.num_agents:
            raise ValueError(f"agent_id must be in 0..{problem.num_agents - 1}.")

        messages: list[AlgorithmMessage] = []
        incident_edges = [
            (min(agent_id, neighbor_id), max(agent_id, neighbor_id))
            for neighbor_id in problem.neighbors[agent_id]
        ]

        for edge_key in incident_edges:
            variable_key = (agent_id, edge_key)
            new_variable_message = self._compute_variable_to_factor(agent_id, edge_key)
            self._variable_to_factor[variable_key] = self._damp_and_normalize(
                self._variable_to_factor[variable_key],
                new_variable_message,
            )
            messages.append(
                self._build_message(
                    sender=agent_id,
                    receiver=self._other_agent(edge_key, agent_id),
                    edge_key=edge_key,
                    kind="variable_to_factor",
                )
            )

        for edge_key in incident_edges:
            constraint = problem.constraints[edge_key]
            for target_agent in edge_key:
                factor_key = (edge_key, target_agent)
                new_factor_message = self._compute_factor_to_variable(
                    constraint,
                    target_agent,
                    self._variable_to_factor,
                )
                self._factor_to_variable[factor_key] = self._damp_and_normalize(
                    self._factor_to_variable[factor_key],
                    new_factor_message,
                )
                messages.append(
                    self._build_message(
                        sender=self._other_agent(edge_key, target_agent),
                        receiver=target_agent,
                        edge_key=edge_key,
                        kind="factor_to_variable",
                    )
                )

        changed_agents = self._update_single_assignment_from_belief(agent_id)
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
            raise RuntimeError("DMSAlgorithm must be initialized before running.")
        return self._problem

    def _compute_all_variable_to_factor(
        self,
    ) -> dict[tuple[int, tuple[int, int]], MessageVector]:
        """Compute all variable-to-factor messages from current factor messages."""

        problem = self._require_problem()
        messages: dict[tuple[int, tuple[int, int]], MessageVector] = {}

        for edge_key in sorted(problem.constraints):
            for agent_id in edge_key:
                messages[(agent_id, edge_key)] = self._compute_variable_to_factor(
                    agent_id,
                    edge_key,
                )

        return messages

    def _compute_all_factor_to_variable(
        self,
        variable_to_factor: dict[tuple[int, tuple[int, int]], MessageVector],
    ) -> dict[tuple[tuple[int, int], int], MessageVector]:
        """Compute all factor-to-variable messages from variable messages."""

        problem = self._require_problem()
        messages: dict[tuple[tuple[int, int], int], MessageVector] = {}

        for edge_key, constraint in sorted(problem.constraints.items()):
            for target_agent in edge_key:
                messages[(edge_key, target_agent)] = self._compute_factor_to_variable(
                    constraint,
                    target_agent,
                    variable_to_factor,
                )

        return messages

    def _compute_variable_to_factor(
        self,
        agent_id: int,
        target_edge: tuple[int, int],
    ) -> MessageVector:
        """Compute one variable-to-factor message."""

        problem = self._require_problem()
        values: list[float] = []

        for value in range(problem.domain_size):
            total = 0.0
            for neighbor_id in problem.neighbors[agent_id]:
                edge_key = (min(agent_id, neighbor_id), max(agent_id, neighbor_id))
                if edge_key == target_edge:
                    continue
                total += self._factor_to_variable[(edge_key, agent_id)][value]
            values.append(total)

        return tuple(values)

    def _compute_factor_to_variable(
        self,
        constraint: BinaryConstraint,
        target_agent: int,
        variable_to_factor: dict[tuple[int, tuple[int, int]], MessageVector],
    ) -> MessageVector:
        """Compute one factor-to-variable min-sum message."""

        problem = self._require_problem()
        edge_key = constraint.key()
        other_agent = self._other_agent(edge_key, target_agent)
        other_message = variable_to_factor[(other_agent, edge_key)]
        values: list[float] = []

        for target_value in range(problem.domain_size):
            best_cost: float | None = None
            for other_value in range(problem.domain_size):
                if target_agent == constraint.agent_a:
                    cost = constraint.cost(target_value, other_value)
                else:
                    cost = constraint.cost(other_value, target_value)
                candidate_cost = cost + other_message[other_value]
                if best_cost is None or candidate_cost < best_cost:
                    best_cost = candidate_cost
            values.append(float(best_cost if best_cost is not None else 0.0))

        return tuple(values)

    def _update_assignment_from_beliefs(self) -> set[int]:
        """Update variable assignments from current beliefs."""

        problem = self._require_problem()
        changed_agents: set[int] = set()
        updated_assignment = dict(self._assignment)

        for agent_id in range(problem.num_agents):
            belief = self._belief(agent_id)
            best_value = min(range(problem.domain_size), key=lambda value: belief[value])
            if updated_assignment[agent_id] != best_value:
                updated_assignment[agent_id] = best_value
                changed_agents.add(agent_id)

        self._assignment = updated_assignment
        return changed_agents

    def _update_single_assignment_from_belief(self, agent_id: int) -> set[int]:
        """Update only one variable assignment from its current belief."""

        problem = self._require_problem()
        belief = self._belief(agent_id)
        best_value = min(range(problem.domain_size), key=lambda value: belief[value])

        if self._assignment[agent_id] == best_value:
            return set()

        self._assignment[agent_id] = best_value
        return {agent_id}

    def _belief(self, agent_id: int) -> MessageVector:
        """Return the belief vector for one variable."""

        problem = self._require_problem()
        values: list[float] = []

        for value in range(problem.domain_size):
            total = 0.0
            for neighbor_id in problem.neighbors[agent_id]:
                edge_key = (min(agent_id, neighbor_id), max(agent_id, neighbor_id))
                total += self._factor_to_variable[(edge_key, agent_id)][value]
            values.append(total)

        return tuple(values)

    def _damp_and_normalize(
        self,
        old_message: MessageVector,
        new_message: MessageVector,
    ) -> MessageVector:
        """Apply damping and subtract the minimum entry."""

        damped = tuple(
            self.damping * old_value + (1 - self.damping) * new_value
            for old_value, new_value in zip(old_message, new_message)
        )
        minimum = min(damped, default=0.0)
        return tuple(value - minimum for value in damped)

    def _build_message(
        self,
        sender: int,
        receiver: int,
        edge_key: tuple[int, int],
        kind: str,
    ) -> AlgorithmMessage:
        """Build a serializable async DMS vector message."""

        if kind == "variable_to_factor":
            vector = self._variable_to_factor[(sender, edge_key)]
        elif kind == "factor_to_variable":
            vector = self._factor_to_variable[(edge_key, receiver)]
        else:
            raise ValueError(f"Unsupported DMS message kind: {kind}.")

        return AlgorithmMessage(
            sender=sender,
            receiver=receiver,
            kind=kind,
            payload={
                "edge": edge_key,
                "vector": vector,
            },
        )

    def _parse_edge_key(self, value: object) -> tuple[int, int]:
        """Validate and return an edge key from a message payload."""

        if (
            not isinstance(value, (tuple, list))
            or len(value) != 2
            or not all(isinstance(agent_id, int) for agent_id in value)
        ):
            raise ValueError("DMS message edge must contain two integer agent IDs.")
        return min(value), max(value)

    def _parse_vector(self, value: object) -> MessageVector:
        """Validate and return a message vector from a message payload."""

        problem = self._require_problem()
        if (
            not isinstance(value, (tuple, list))
            or len(value) != problem.domain_size
            or not all(isinstance(entry, (int, float)) for entry in value)
        ):
            raise ValueError("DMS message vector must match the domain size.")
        return tuple(float(entry) for entry in value)

    @staticmethod
    def _zero_vector(domain_size: int) -> MessageVector:
        """Return a zero message vector for the domain."""

        return tuple(0.0 for _ in range(domain_size))

    @staticmethod
    def _other_agent(edge_key: tuple[int, int], agent_id: int) -> int:
        """Return the other endpoint of an edge."""

        if agent_id == edge_key[0]:
            return edge_key[1]
        if agent_id == edge_key[1]:
            return edge_key[0]
        raise ValueError(f"Agent {agent_id} is not in edge {edge_key}.")
