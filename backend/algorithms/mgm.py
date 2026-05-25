"""MGM implementation for binary DCOPs."""

from __future__ import annotations

from dataclasses import dataclass

from backend.algorithms.base import (
    AlgorithmMessage,
    AlgorithmStepResult,
    DistributedAlgorithm,
)
from backend.dcop.cost import calculate_local_cost
from backend.dcop.problem import DCOPProblem


@dataclass(frozen=True)
class _GainProposal:
    """One agent's proposed MGM move."""

    gain: int
    value: int


class MGMAlgorithm(DistributedAlgorithm):
    """Maximum Gain Message algorithm."""

    name = "mgm"

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed
        self._problem: DCOPProblem | None = None
        self._assignment: dict[int, int] = {}
        self._local_views: dict[int, dict[int, int]] = {}
        self._neighbor_gains: dict[int, dict[int, int]] = {}
        self._own_gains: dict[int, int] = {}

    def initialize(
        self,
        problem: DCOPProblem,
        assignment: dict[int, int],
        seed: int | None = None,
    ) -> None:
        """Initialize MGM state from the provided assignment."""

        problem.validate_assignment(assignment)
        self._problem = problem
        self._assignment = dict(assignment)
        self._local_views = {
            agent_id: {} for agent_id in range(problem.num_agents)
        }
        self._neighbor_gains = {
            agent_id: {} for agent_id in range(problem.num_agents)
        }
        self._own_gains = {
            agent_id: 0 for agent_id in range(problem.num_agents)
        }

    def run_synchronous_iteration(self) -> AlgorithmStepResult:
        """Run one two-phase synchronous MGM iteration."""

        problem = self._require_problem()
        starting_assignment = dict(self._assignment)
        proposals: dict[int, _GainProposal] = {}

        for agent_id in range(problem.num_agents):
            proposals[agent_id] = self._compute_sync_proposal(
                agent_id,
                starting_assignment,
            )

        winning_changes: dict[int, int] = {}
        for agent_id, proposal in proposals.items():
            if proposal.gain <= 0:
                continue

            if self._wins_gain_comparison(
                agent_id,
                proposal.gain,
                {
                    neighbor_id: proposals[neighbor_id].gain
                    for neighbor_id in problem.neighbors[agent_id]
                },
            ):
                winning_changes[agent_id] = proposal.value

        if winning_changes:
            updated_assignment = dict(starting_assignment)
            updated_assignment.update(winning_changes)
            self._assignment = updated_assignment

        sum_degrees = sum(len(neighbors) for neighbors in problem.neighbors.values())
        return AlgorithmStepResult(
            changed_agents=set(winning_changes),
            messages_sent=2 * sum_degrees,
            metadata={
                "positive_gains": sum(
                    1 for proposal in proposals.values() if proposal.gain > 0
                ),
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
        """Handle an asynchronous value or gain message."""

        problem = self._require_problem()
        if message.receiver not in range(problem.num_agents):
            raise ValueError(f"Invalid message receiver: {message.receiver}.")
        if message.sender not in problem.neighbors[message.receiver]:
            raise ValueError(
                f"Agent {message.sender} is not a neighbor of agent {message.receiver}."
            )

        if message.kind == "value":
            value = message.payload.get("value")
            if not isinstance(value, int) or not 0 <= value < problem.domain_size:
                raise ValueError("MGM value messages must contain an in-domain integer value.")
            self._local_views[message.receiver][message.sender] = value
            return AlgorithmStepResult(changed_agents=set(), messages_sent=0)

        if message.kind == "gain":
            gain = message.payload.get("gain")
            if not isinstance(gain, int) or gain < 0:
                raise ValueError("MGM gain messages must contain a non-negative integer gain.")
            self._neighbor_gains[message.receiver][message.sender] = gain
            return AlgorithmStepResult(changed_agents=set(), messages_sent=0)

        raise ValueError(f"Unsupported MGM message kind: {message.kind}.")

    def on_async_activation(self, agent_id: int) -> AlgorithmStepResult:
        """Run one message-driven MGM activation for a single agent."""

        problem = self._require_problem()
        if not 0 <= agent_id < problem.num_agents:
            raise ValueError(f"agent_id must be in 0..{problem.num_agents - 1}.")

        proposal = self._compute_async_proposal(agent_id)
        self._own_gains[agent_id] = proposal.gain

        messages = self._build_gain_messages(agent_id, proposal.gain)
        changed_agents: set[int] = set()

        neighbor_gains = {
            neighbor_id: self._neighbor_gains[agent_id].get(neighbor_id, 0)
            for neighbor_id in problem.neighbors[agent_id]
        }
        if proposal.gain > 0 and self._wins_gain_comparison(
            agent_id,
            proposal.gain,
            neighbor_gains,
        ):
            self._assignment[agent_id] = proposal.value
            changed_agents.add(agent_id)
            messages.extend(self._build_value_messages(agent_id))

        return AlgorithmStepResult(
            changed_agents=changed_agents,
            messages_sent=len(messages),
            metadata={
                "messages": messages,
                "gain": proposal.gain,
            },
        )

    def _require_problem(self) -> DCOPProblem:
        """Return the initialized problem or fail clearly."""

        if self._problem is None:
            raise RuntimeError("MGMAlgorithm must be initialized before running.")
        return self._problem

    def _compute_sync_proposal(
        self,
        agent_id: int,
        assignment: dict[int, int],
    ) -> _GainProposal:
        """Compute an MGM gain proposal from a synchronous assignment snapshot."""

        problem = self._require_problem()
        current_value = assignment[agent_id]
        current_cost = calculate_local_cost(problem, assignment, agent_id)
        best_value = current_value
        best_cost = current_cost

        for candidate_value in range(problem.domain_size):
            candidate_cost = calculate_local_cost(
                problem,
                assignment,
                agent_id,
                value=candidate_value,
            )
            if candidate_cost < best_cost:
                best_value = candidate_value
                best_cost = candidate_cost

        gain = current_cost - best_cost
        if gain <= 0:
            return _GainProposal(gain=0, value=current_value)

        return _GainProposal(gain=gain, value=best_value)

    def _compute_async_proposal(self, agent_id: int) -> _GainProposal:
        """Compute an MGM gain proposal from local async neighbor views."""

        problem = self._require_problem()
        current_value = self._assignment[agent_id]
        current_cost = self._calculate_async_local_cost(agent_id, current_value)
        best_value = current_value
        best_cost = current_cost

        for candidate_value in range(problem.domain_size):
            candidate_cost = self._calculate_async_local_cost(agent_id, candidate_value)
            if candidate_cost < best_cost:
                best_value = candidate_value
                best_cost = candidate_cost

        gain = current_cost - best_cost
        if gain <= 0:
            return _GainProposal(gain=0, value=current_value)

        return _GainProposal(gain=gain, value=best_value)

    @staticmethod
    def _wins_gain_comparison(
        agent_id: int,
        gain: int,
        neighbor_gains: dict[int, int],
    ) -> bool:
        """Return True if an agent wins MGM gain comparison against neighbors."""

        for neighbor_id, neighbor_gain in neighbor_gains.items():
            if gain < neighbor_gain:
                return False
            if gain == neighbor_gain and agent_id > neighbor_id:
                return False

        return True

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

    def _build_gain_messages(self, agent_id: int, gain: int) -> list[AlgorithmMessage]:
        """Build gain messages from one agent to all neighbors."""

        problem = self._require_problem()
        return [
            AlgorithmMessage(
                sender=agent_id,
                receiver=neighbor_id,
                kind="gain",
                payload={
                    "gain": gain,
                },
            )
            for neighbor_id in problem.neighbors[agent_id]
        ]

    def _calculate_async_local_cost(self, agent_id: int, value: int) -> int:
        """Calculate local cost using the agent's latest neighbor value view."""

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
