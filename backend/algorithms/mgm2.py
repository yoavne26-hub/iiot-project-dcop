"""MGM-2 implementation for binary DCOPs."""

from __future__ import annotations

from dataclasses import dataclass

from backend.algorithms.base import (
    AlgorithmMessage,
    AlgorithmStepResult,
    DistributedAlgorithm,
)
from backend.dcop.cost import calculate_local_cost
from backend.dcop.problem import BinaryConstraint, DCOPProblem


@dataclass(frozen=True)
class _MoveCandidate:
    """One MGM-2 candidate move."""

    kind: str
    agents: tuple[int, ...]
    values: dict[int, int]
    gain: int


class MGM2Algorithm(DistributedAlgorithm):
    """Deterministic practical MGM-2 algorithm."""

    name = "mgm2"

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed
        self._problem: DCOPProblem | None = None
        self._assignment: dict[int, int] = {}
        self._local_views: dict[int, dict[int, int]] = {}
        self._neighbor_candidates: dict[int, dict[int, _MoveCandidate]] = {}

    def initialize(
        self,
        problem: DCOPProblem,
        assignment: dict[int, int],
        seed: int | None = None,
    ) -> None:
        """Initialize MGM-2 state from the provided assignment."""

        problem.validate_assignment(assignment)
        self._problem = problem
        self._assignment = dict(assignment)
        self._local_views = {
            agent_id: {} for agent_id in range(problem.num_agents)
        }
        self._neighbor_candidates = {
            agent_id: {} for agent_id in range(problem.num_agents)
        }

    def run_synchronous_iteration(self) -> AlgorithmStepResult:
        """Run one synchronous MGM-2 iteration."""

        problem = self._require_problem()
        starting_assignment = dict(self._assignment)
        candidates: list[_MoveCandidate] = []

        for agent_id in range(problem.num_agents):
            single_candidate = self._compute_sync_single_candidate(
                agent_id,
                starting_assignment,
            )
            if single_candidate.gain > 0:
                candidates.append(single_candidate)

        for constraint in problem.constraints.values():
            pair_candidate = self._compute_sync_pair_candidate(
                constraint,
                starting_assignment,
            )
            if pair_candidate.gain > 0:
                candidates.append(pair_candidate)

        selected_moves = self._select_non_conflicting_moves(candidates)
        changed_agents: set[int] = set()
        if selected_moves:
            updated_assignment = dict(starting_assignment)
            for move in selected_moves:
                for agent_id, value in move.values.items():
                    if updated_assignment[agent_id] != value:
                        changed_agents.add(agent_id)
                        updated_assignment[agent_id] = value
            self._assignment = updated_assignment

        sum_degrees = sum(len(neighbors) for neighbors in problem.neighbors.values())
        return AlgorithmStepResult(
            changed_agents=changed_agents,
            messages_sent=3 * sum_degrees,
            metadata={
                "candidate_moves": len(candidates),
                "selected_moves": len(selected_moves),
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
        """Handle asynchronous value and MGM-2 candidate messages."""

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
                raise ValueError("MGM-2 value messages must contain an in-domain integer value.")
            self._local_views[message.receiver][message.sender] = value
            return AlgorithmStepResult(changed_agents=set(), messages_sent=0)

        if message.kind == "mgm2_candidate":
            candidate = self._candidate_from_payload(message.payload)
            self._neighbor_candidates[message.receiver][message.sender] = candidate
            return AlgorithmStepResult(changed_agents=set(), messages_sent=0)

        raise ValueError(f"Unsupported MGM-2 message kind: {message.kind}.")

    def on_async_activation(self, agent_id: int) -> AlgorithmStepResult:
        """Run one practical asynchronous MGM-2 activation."""

        problem = self._require_problem()
        if not 0 <= agent_id < problem.num_agents:
            raise ValueError(f"agent_id must be in 0..{problem.num_agents - 1}.")

        candidate = self._best_async_candidate_for_agent(agent_id)
        messages = self._build_candidate_messages(agent_id, candidate)
        changed_agents: set[int] = set()

        if candidate.gain > 0 and self._wins_against_known_neighbor_candidates(
            candidate,
            agent_id,
        ):
            for changed_agent, value in candidate.values.items():
                if self._assignment[changed_agent] != value:
                    self._assignment[changed_agent] = value
                    changed_agents.add(changed_agent)
            for changed_agent in sorted(changed_agents):
                messages.extend(self._build_value_messages(changed_agent))

        return AlgorithmStepResult(
            changed_agents=changed_agents,
            messages_sent=len(messages),
            metadata={
                "messages": messages,
                "candidate_gain": candidate.gain,
                "candidate_kind": candidate.kind,
            },
        )

    def _require_problem(self) -> DCOPProblem:
        """Return the initialized problem or fail clearly."""

        if self._problem is None:
            raise RuntimeError("MGM2Algorithm must be initialized before running.")
        return self._problem

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

    def _best_async_candidate_for_agent(self, agent_id: int) -> _MoveCandidate:
        """Return the best locally visible candidate involving one agent."""

        problem = self._require_problem()
        assignment = self._assignment_with_local_view(agent_id)
        candidates = [self._compute_sync_single_candidate(agent_id, assignment)]

        for neighbor_id in problem.neighbors[agent_id]:
            constraint = problem.constraints[(min(agent_id, neighbor_id), max(agent_id, neighbor_id))]
            candidates.append(self._compute_sync_pair_candidate(constraint, assignment))

        return min(candidates, key=self._priority_key)

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

    def _select_non_conflicting_moves(
        self,
        candidates: list[_MoveCandidate],
    ) -> list[_MoveCandidate]:
        """Select non-overlapping candidate moves by deterministic priority."""

        selected: list[_MoveCandidate] = []
        used_agents: set[int] = set()

        for candidate in sorted(candidates, key=self._priority_key):
            if any(agent_id in used_agents for agent_id in candidate.agents):
                continue
            selected.append(candidate)
            used_agents.update(candidate.agents)

        return selected

    @staticmethod
    def _priority_key(candidate: _MoveCandidate) -> tuple[int, int, tuple[int, ...]]:
        """Sort candidates by higher gain, pair before single, then lower IDs."""

        pair_priority = 0 if candidate.kind == "pair" else 1
        return (-candidate.gain, pair_priority, candidate.agents)

    def _wins_against_known_neighbor_candidates(
        self,
        candidate: _MoveCandidate,
        owner_id: int,
    ) -> bool:
        """Return True if a candidate beats known overlapping neighbor candidates."""

        known_candidates = self._neighbor_candidates[owner_id].values()
        candidate_agents = set(candidate.agents)
        for neighbor_candidate in known_candidates:
            if candidate_agents.isdisjoint(neighbor_candidate.agents):
                continue
            if self._priority_key(neighbor_candidate) < self._priority_key(candidate):
                return False

        return True

    def _assignment_with_local_view(self, agent_id: int) -> dict[int, int]:
        """Return current assignment with one agent's neighbor values overlaid."""

        assignment = dict(self._assignment)
        assignment.update(self._local_views[agent_id])
        return assignment

    def _build_value_messages(self, agent_id: int) -> list[AlgorithmMessage]:
        """Build current-value messages from one agent to all neighbors."""

        problem = self._require_problem()
        return [
            AlgorithmMessage(
                sender=agent_id,
                receiver=neighbor_id,
                kind="value",
                payload={
                    "value": self._assignment[agent_id],
                },
            )
            for neighbor_id in problem.neighbors[agent_id]
        ]

    def _build_candidate_messages(
        self,
        agent_id: int,
        candidate: _MoveCandidate,
    ) -> list[AlgorithmMessage]:
        """Build candidate messages from one agent to all neighbors."""

        problem = self._require_problem()
        return [
            AlgorithmMessage(
                sender=agent_id,
                receiver=neighbor_id,
                kind="mgm2_candidate",
                payload=self._candidate_to_payload(candidate),
            )
            for neighbor_id in problem.neighbors[agent_id]
        ]

    @staticmethod
    def _candidate_to_payload(candidate: _MoveCandidate) -> dict[str, object]:
        """Convert a candidate move to a message payload."""

        return {
            "kind": candidate.kind,
            "agents": list(candidate.agents),
            "values": dict(candidate.values),
            "gain": candidate.gain,
        }

    @staticmethod
    def _candidate_from_payload(payload: dict[str, object]) -> _MoveCandidate:
        """Convert a message payload back to a candidate move."""

        kind = payload.get("kind")
        agents = payload.get("agents")
        values = payload.get("values")
        gain = payload.get("gain")

        if kind not in {"single", "pair"}:
            raise ValueError("MGM-2 candidate payload has invalid kind.")
        if not isinstance(agents, list) or any(not isinstance(agent, int) for agent in agents):
            raise ValueError("MGM-2 candidate payload has invalid agents.")
        if not isinstance(values, dict):
            raise ValueError("MGM-2 candidate payload has invalid values.")
        if not isinstance(gain, int) or gain < 0:
            raise ValueError("MGM-2 candidate payload has invalid gain.")

        normalized_values: dict[int, int] = {}
        for agent_id, value in values.items():
            if not isinstance(agent_id, int) or not isinstance(value, int):
                raise ValueError("MGM-2 candidate payload values must map int to int.")
            normalized_values[agent_id] = value

        return _MoveCandidate(
            kind=kind,
            agents=tuple(agents),
            values=normalized_values,
            gain=gain,
        )
