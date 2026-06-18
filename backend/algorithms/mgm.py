"""MGM implementation for binary DCOPs."""

from __future__ import annotations

from dataclasses import dataclass

from backend.algorithms.base import AlgorithmStepResult, DistributedAlgorithm
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
