"""MGM-2 implementation for binary DCOPs."""

from __future__ import annotations

import random
from dataclasses import dataclass

from backend.algorithms.base import AlgorithmStepResult, DistributedAlgorithm
from backend.dcop.cost import calculate_local_cost
from backend.dcop.problem import BinaryConstraint, DCOPProblem


DEFAULT_OFFER_PROBABILITY = 0.50


@dataclass(frozen=True)
class _MoveCandidate:
    """One MGM-2 candidate move."""

    kind: str
    agents: tuple[int, ...]
    values: dict[int, int]
    gain: int


@dataclass
class _Group:
    """A committed MGM-2 move group (a singleton or an accepted pair)."""

    members: tuple[int, ...]
    gain: int
    values: dict[int, int]

    @property
    def leader(self) -> int:
        """Return the group leader, defined as the lowest member id."""

        return min(self.members)


class MGM2Algorithm(DistributedAlgorithm):
    """Randomized MGM-2 with offering, pair coordination, and damping-free gains."""

    name = "mgm2"

    def __init__(
        self,
        offer_probability: float = DEFAULT_OFFER_PROBABILITY,
        seed: int | None = None,
    ) -> None:
        if not 0 <= offer_probability <= 1:
            raise ValueError("offer_probability must be between 0 and 1.")

        self.offer_probability = offer_probability
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
        """Initialize MGM-2 state from the provided assignment."""

        problem.validate_assignment(assignment)
        rng_seed = self.seed if seed is None else seed

        self._problem = problem
        self._assignment = dict(assignment)
        self._rng = random.Random(rng_seed)

    def run_synchronous_iteration(self) -> AlgorithmStepResult:
        """Run one randomized synchronous MGM-2 iteration.

        Mirrors the reference algorithm: agents offer to a random neighbor with
        ``offer_probability``; each non-offering receiver accepts its best-gain
        proposal to form a pair; unpaired agents fall back to a singleton 1-opt
        move; every improving group commits when it has the highest gain among
        neighboring groups, with ties broken by the lower group leader id.
        """

        problem = self._require_problem()
        starting_assignment = dict(self._assignment)

        proposed_to = self._collect_offers(problem)
        groups = self._build_groups(problem, starting_assignment, proposed_to)
        group_of = {
            member: group for group in groups for member in group.members
        }
        winning_groups = self._select_winning_groups(problem, groups, group_of)

        changed_agents: set[int] = set()
        if winning_groups:
            updated_assignment = dict(starting_assignment)
            for group in winning_groups:
                for agent_id, value in group.values.items():
                    if updated_assignment[agent_id] != value:
                        changed_agents.add(agent_id)
                        updated_assignment[agent_id] = value
            self._assignment = updated_assignment

        sum_degrees = sum(len(neighbors) for neighbors in problem.neighbors.values())
        return AlgorithmStepResult(
            changed_agents=changed_agents,
            messages_sent=3 * sum_degrees,
            metadata={
                "groups": len(groups),
                "winning_groups": len(winning_groups),
                "pairs": sum(1 for group in groups if len(group.members) == 2),
            },
        )

    def get_assignment(self) -> dict[int, int]:
        """Return a copy of the current assignment."""

        return dict(self._assignment)

    def _require_problem(self) -> DCOPProblem:
        """Return the initialized problem or fail clearly."""

        if self._problem is None:
            raise RuntimeError("MGM2Algorithm must be initialized before running.")
        return self._problem

    def _collect_offers(self, problem: DCOPProblem) -> dict[int, int]:
        """Pick, per offering agent, the random neighbor it proposes to."""

        proposed_to: dict[int, int] = {}
        for agent_id in range(problem.num_agents):
            neighbors = problem.neighbors[agent_id]
            if neighbors and self._rng.random() < self.offer_probability:
                proposed_to[agent_id] = self._rng.choice(neighbors)

        return proposed_to

    def _build_groups(
        self,
        problem: DCOPProblem,
        assignment: dict[int, int],
        proposed_to: dict[int, int],
    ) -> list[_Group]:
        """Form accepted pair groups and singleton groups for the rest."""

        proposals_by_receiver: dict[int, list[int]] = {
            agent_id: [] for agent_id in range(problem.num_agents)
        }
        for proposer, receiver in proposed_to.items():
            proposals_by_receiver[receiver].append(proposer)

        groups: list[_Group] = []
        paired: set[int] = set()

        # Only non-offering agents accept proposals; an offering agent rejects.
        for receiver in range(problem.num_agents):
            if receiver in proposed_to:
                continue

            incoming = proposals_by_receiver[receiver]
            best_proposer: int | None = None
            best_candidate: _MoveCandidate | None = None
            best_score: tuple[int, int] = (-1, 0)
            for proposer in incoming:
                constraint = problem.constraints[
                    (min(proposer, receiver), max(proposer, receiver))
                ]
                pair_candidate = self._compute_sync_pair_candidate(constraint, assignment)
                # Highest gain wins; ties break to the lower proposer id.
                score = (pair_candidate.gain, -proposer)
                if best_proposer is None or score > best_score:
                    best_score = score
                    best_proposer = proposer
                    best_candidate = pair_candidate

            if best_candidate is None:
                continue

            members = tuple(sorted((best_proposer, receiver)))
            groups.append(
                _Group(
                    members=members,
                    gain=best_candidate.gain,
                    values=dict(best_candidate.values),
                )
            )
            paired.add(best_proposer)
            paired.add(receiver)

        # Every agent without a partner competes with a singleton 1-opt move.
        for agent_id in range(problem.num_agents):
            if agent_id in paired:
                continue
            single = self._compute_sync_single_candidate(agent_id, assignment)
            groups.append(
                _Group(
                    members=(agent_id,),
                    gain=single.gain,
                    values=dict(single.values),
                )
            )

        return groups

    def _select_winning_groups(
        self,
        problem: DCOPProblem,
        groups: list[_Group],
        group_of: dict[int, _Group],
    ) -> list[_Group]:
        """Select improving groups that dominate their neighboring groups."""

        winning: list[_Group] = []
        for group in groups:
            if group.gain <= 0:
                continue

            members = set(group.members)
            neighbor_agents: set[int] = set()
            for member in members:
                neighbor_agents.update(problem.neighbors[member])
            neighbor_agents.difference_update(members)

            wins = True
            for neighbor in neighbor_agents:
                neighbor_group = group_of[neighbor]
                if neighbor_group is group:
                    continue
                if neighbor_group.gain > group.gain:
                    wins = False
                    break
                if neighbor_group.gain == group.gain and neighbor_group.leader < group.leader:
                    wins = False
                    break

            if wins:
                winning.append(group)

        return winning

    def _compute_sync_single_candidate(
        self,
        agent_id: int,
        assignment: dict[int, int],
    ) -> _MoveCandidate:
        """Compute one snapshot-based single-agent candidate."""

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
            gain = 0
            best_value = current_value

        return _MoveCandidate(
            kind="single",
            agents=(agent_id,),
            values={agent_id: best_value},
            gain=gain,
        )

    def _compute_sync_pair_candidate(
        self,
        constraint: BinaryConstraint,
        assignment: dict[int, int],
    ) -> _MoveCandidate:
        """Compute one snapshot-based pair candidate for a constraint edge."""

        problem = self._require_problem()
        agent_a, agent_b = constraint.key()
        current_values = {
            agent_a: assignment[agent_a],
            agent_b: assignment[agent_b],
        }
        current_cost = self._affected_cost(
            agents=(agent_a, agent_b),
            assignment=assignment,
            replacement_values={},
        )
        best_values = dict(current_values)
        best_cost = current_cost

        for value_a in range(problem.domain_size):
            for value_b in range(problem.domain_size):
                if value_a == current_values[agent_a] and value_b == current_values[agent_b]:
                    continue

                candidate_values = {
                    agent_a: value_a,
                    agent_b: value_b,
                }
                candidate_cost = self._affected_cost(
                    agents=(agent_a, agent_b),
                    assignment=assignment,
                    replacement_values=candidate_values,
                )
                if candidate_cost < best_cost:
                    best_values = candidate_values
                    best_cost = candidate_cost

        gain = current_cost - best_cost
        if gain <= 0:
            gain = 0
            best_values = current_values

        return _MoveCandidate(
            kind="pair",
            agents=(agent_a, agent_b),
            values=best_values,
            gain=gain,
        )

    def _affected_cost(
        self,
        agents: tuple[int, ...],
        assignment: dict[int, int],
        replacement_values: dict[int, int],
    ) -> int:
        """Return cost of all constraints touching any agent, counted once."""

        problem = self._require_problem()
        affected_agents = set(agents)
        total = 0

        for constraint in problem.constraints.values():
            if constraint.agent_a not in affected_agents and constraint.agent_b not in affected_agents:
                continue

            value_a = replacement_values.get(
                constraint.agent_a,
                assignment[constraint.agent_a],
            )
            value_b = replacement_values.get(
                constraint.agent_b,
                assignment[constraint.agent_b],
            )
            total += constraint.cost(value_a, value_b)

        return total
