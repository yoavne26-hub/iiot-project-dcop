"""Damped min-sum implementation for binary DCOPs."""

from __future__ import annotations

from backend.algorithms.base import AlgorithmStepResult, DistributedAlgorithm
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

        # Variable -> factor messages, computed from the previous round's factor
        # messages, then damped in place.
        new_variable_to_factor = self._compute_all_variable_to_factor()
        self._variable_to_factor = {
            key: self._damp(self._variable_to_factor[key], message)
            for key, message in new_variable_to_factor.items()
        }

        # Factor -> variable messages, computed from the just-damped variable
        # messages (not the raw ones), then damped. Feeding the damped variable
        # messages into the factor update matches the reference implementation and
        # is what makes min-sum converge well; using the raw messages here leaves
        # DMS stuck at a noticeably worse solution.
        new_factor_to_variable = self._compute_all_factor_to_variable(self._variable_to_factor)
        self._factor_to_variable = {
            key: self._damp(self._factor_to_variable[key], message)
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

        return self._normalize(tuple(values))

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

        return self._normalize(tuple(values))

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

    def _damp(
        self,
        old_message: MessageVector,
        new_message: MessageVector,
    ) -> MessageVector:
        """Blend the previous message with the freshly computed one.

        The freshly computed message is already normalized (its minimum entry is
        zero) before it reaches here, matching the reference implementation which
        normalizes each computed message and then damps without renormalizing.
        """

        return tuple(
            (1 - self.damping) * old_value + self.damping * new_value
            for old_value, new_value in zip(old_message, new_message)
        )

    @staticmethod
    def _normalize(message: MessageVector) -> MessageVector:
        """Shift a message so its minimum entry is zero."""

        minimum = min(message, default=0.0)
        return tuple(value - minimum for value in message)

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
