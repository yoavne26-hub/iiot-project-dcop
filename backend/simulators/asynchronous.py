"""Asynchronous simulator for DCOP algorithms.

Each agent is its own thread that owns all of its state (current value, the latest
known value/message from every neighbor, and algorithm-specific bookkeeping). The
only shared state is a global value registry and the per-agent logical-step
counters, each guarded by a short-lived lock. Agents communicate exclusively
through per-agent mailboxes (one ``queue.Queue`` each).

This mirrors the reference design in ``examples/exampleCode.py`` and the
assignment clarifications: an agent performs one local step whenever it has
received at least one message, using the latest message available from every
neighbor (section 3 / 6.1); the run stops when the slowest agent reaches the
configured number of local steps (section 5); the horizontal axis is the
Lamport-style minimum local step across agents (section 4).

There is deliberately no global "algorithm lock": because every agent owns its
state and only writes its own value into the shared registry, agents run truly
concurrently. An earlier design funneled all computation through one shared
algorithm object and a single lock, which serialized everything and could
livelock the more expensive algorithms.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import queue
import random
import threading
import time

import numpy as np

from backend.algorithms.base import DistributedAlgorithm
from backend.dcop.cost import calculate_global_cost
from backend.dcop.problem import DCOPProblem
from backend.simulators.synchronous import SimulationRunResult


@dataclass(frozen=True)
class _Msg:
    """A message passed between asynchronous agents."""

    sender: int
    kind: str
    payload: dict


# ================================================================
# Local cost / move helpers (pure functions over the agent's view)
# ================================================================
def _local_cost(
    agent_id: int,
    value: int,
    known: dict[int, int],
    problem: DCOPProblem,
    incident: list[tuple[int, int]],
) -> int:
    """Cost of one agent's incident constraints given its neighbor view."""

    total = 0
    for edge in incident:
        constraint = problem.constraints[edge]
        if constraint.agent_a == agent_id:
            total += constraint.cost(value, known[constraint.agent_b])
        else:
            total += constraint.cost(known[constraint.agent_a], value)
    return total


def _best_single(
    agent_id: int,
    value: int,
    known: dict[int, int],
    problem: DCOPProblem,
    incident: list[tuple[int, int]],
) -> tuple[int, int, int, int]:
    """Best single-agent value (ties break to the lower-index value).

    Returns ``(best_value, gain, current_local, best_local)``. ``gain`` is
    positive when the move strictly improves cost; equal-cost lower-index moves
    are reported with gain 0 (this is what lets DSA-C make plateau moves).
    """

    current_local = _local_cost(agent_id, value, known, problem, incident)
    best_value, best_cost = value, current_local
    for candidate in range(problem.domain_size):
        candidate_cost = _local_cost(agent_id, candidate, known, problem, incident)
        if candidate_cost < best_cost or (
            candidate_cost == best_cost and candidate < best_value
        ):
            best_value, best_cost = candidate, candidate_cost
    return best_value, current_local - best_cost, current_local, best_cost


def _affected_edges(first: int, second: int, problem: DCOPProblem) -> set[tuple[int, int]]:
    """All constraint edges incident to either agent (each counted once)."""

    edges: set[tuple[int, int]] = set()
    for agent in (first, second):
        for neighbor in problem.neighbors[agent]:
            edges.add((min(agent, neighbor), max(agent, neighbor)))
    return edges


def _best_pair_values(
    first: int,
    second: int,
    known: dict[int, int],
    problem: DCOPProblem,
) -> tuple[int, int, int]:
    """Best joint values for a pair (a true 2-opt move over affected edges)."""

    edges = _affected_edges(first, second, problem)

    def pair_cost(value_first: int, value_second: int) -> int:
        total = 0
        for i, j in edges:
            vi = value_first if i == first else value_second if i == second else known[i]
            vj = value_first if j == first else value_second if j == second else known[j]
            total += problem.constraints[(i, j)].cost(vi, vj)
        return total

    current_first, current_second = known[first], known[second]
    current_cost = pair_cost(current_first, current_second)
    best_first, best_second, best_cost = current_first, current_second, current_cost
    for value_first in range(problem.domain_size):
        for value_second in range(problem.domain_size):
            cost = pair_cost(value_first, value_second)
            if cost < best_cost or (
                cost == best_cost and (value_first, value_second) < (best_first, best_second)
            ):
                best_first, best_second, best_cost = value_first, value_second, cost
    return best_first, best_second, current_cost - best_cost


def _normalize(vector: np.ndarray) -> np.ndarray:
    """Shift a message vector so its minimum entry is zero."""

    return vector - float(np.min(vector))


# ================================================================
# Shared world
# ================================================================
class _AsyncWorld:
    """Shared state across asynchronous agent threads."""

    def __init__(self, problem: DCOPProblem, max_steps: int) -> None:
        self.problem = problem
        self.max_steps = max_steps
        self.num_agents = problem.num_agents
        self.values: dict[int, int] = dict(problem.initial_assignment)
        self.values_lock = threading.Lock()
        self.clock_lock = threading.Lock()
        self.local_steps: dict[int, int] = {i: 0 for i in range(self.num_agents)}
        self.stop_event = threading.Event()
        self.queues: dict[int, queue.Queue] = {
            i: queue.Queue() for i in range(self.num_agents)
        }
        self.start_barrier = (
            threading.Barrier(self.num_agents) if self.num_agents > 0 else None
        )

    def snapshot(self) -> dict[int, int]:
        with self.values_lock:
            return dict(self.values)

    def set_value(self, agent_id: int, value: int) -> None:
        with self.values_lock:
            self.values[agent_id] = int(value)

    def update_step(self, agent_id: int, step: int) -> None:
        with self.clock_lock:
            self.local_steps[agent_id] = int(step)
            if min(self.local_steps.values()) >= self.max_steps:
                self.stop_event.set()

    def min_step(self) -> int:
        with self.clock_lock:
            return min(self.local_steps.values()) if self.local_steps else self.max_steps


# ================================================================
# Per-agent thread
# ================================================================
class _AsyncAgent(threading.Thread):
    """One DCOP agent running asynchronously in its own thread."""

    def __init__(
        self,
        agent_id: int,
        world: _AsyncWorld,
        algorithm_name: str,
        rng_seed: int,
        dsa_probability: float,
        offer_probability: float,
        dms_lambda: float,
    ) -> None:
        super().__init__(daemon=True, name=f"DCOPAsyncAgent-{agent_id}")
        self.agent_id = agent_id
        self.world = world
        self.problem = world.problem
        self.algorithm_name = algorithm_name
        self.dsa_probability = dsa_probability
        self.offer_probability = offer_probability
        self.dms_lambda = dms_lambda

        self.neighbors = list(self.problem.neighbors[agent_id])
        self.domain_size = self.problem.domain_size
        self.value = self.problem.initial_assignment[agent_id]
        self.known: dict[int, int] = dict(self.problem.initial_assignment)
        self.inbox = world.queues[agent_id]
        self.rng = random.Random(rng_seed)
        self.local_step = 0
        self.messages_sent = 0
        self.assignment_changes = 0
        self.incident = [
            (min(agent_id, n), max(agent_id, n)) for n in self.neighbors
        ]

        # MGM state.
        self.best_value = self.value
        self.gain = 0
        self.neighbor_gains: dict[int, int] = {n: 0 for n in self.neighbors}

        # MGM-2 state.
        self.proposed_to: int | None = None
        self.received_proposals: list[_Msg] = []
        self.received_accepts: list[_Msg] = []
        self.neighbor_group_messages: dict[int, dict] = {}
        self.group_id: tuple[int, ...] = (agent_id,)
        self.group_gain = 0
        self.group_candidate_values: dict[int, int] = {agent_id: self.value}

        # DMS state (owns the function node for every edge where it has the lower id).
        self.owned_edges = [edge for edge in self.incident if min(edge) == agent_id]
        self.dms_matrices = {
            edge: np.array(self.problem.constraints[edge].costs, dtype=float)
            for edge in self.incident
        }
        self.q_old = {edge: np.zeros(self.domain_size) for edge in self.incident}
        self.r_incoming = {edge: np.zeros(self.domain_size) for edge in self.incident}
        self.owned_q_messages = {
            edge: {edge[0]: np.zeros(self.domain_size), edge[1]: np.zeros(self.domain_size)}
            for edge in self.owned_edges
        }
        self.owned_r_old = {
            (edge, endpoint): np.zeros(self.domain_size)
            for edge in self.owned_edges
            for endpoint in edge
        }

    # ---- messaging -------------------------------------------------
    def send(self, receiver: int, kind: str, payload: dict) -> None:
        if self.world.stop_event.is_set():
            return
        self.world.queues[receiver].put(_Msg(self.agent_id, kind, payload))
        self.messages_sent += 1

    def broadcast_value(self) -> None:
        for neighbor in self.neighbors:
            self.send(neighbor, "value", {"value": self.value})

    def set_value(self, new_value: int) -> None:
        new_value = int(new_value)
        if new_value != self.value:
            self.assignment_changes += 1
        self.value = new_value
        self.known[self.agent_id] = self.value
        self.world.set_value(self.agent_id, self.value)

    # ---- run loop --------------------------------------------------
    def run(self) -> None:
        if self.world.start_barrier is not None:
            self.world.start_barrier.wait()

        self.broadcast_value()

        if not self.neighbors:
            self._run_isolated()
            return

        while not self.world.stop_event.is_set():
            try:
                first = self.inbox.get(timeout=0.05)
            except queue.Empty:
                continue

            messages = [first]
            while True:
                try:
                    messages.append(self.inbox.get_nowait())
                except queue.Empty:
                    break

            self._process(messages)
            self._step()
            self.local_step += 1
            self.world.update_step(self.agent_id, self.local_step)

    def _run_isolated(self) -> None:
        # No neighbors means no messages will ever arrive; advance the clock so the
        # global minimum-step stopping rule can still complete.
        while not self.world.stop_event.is_set() and self.local_step < self.world.max_steps:
            self.local_step += 1
            self.world.update_step(self.agent_id, self.local_step)
            time.sleep(0.0001)

    # ---- message handling -----------------------------------------
    def _process(self, messages: list[_Msg]) -> None:
        for message in messages:
            kind = message.kind
            if kind == "value":
                self.known[message.sender] = int(message.payload["value"])
            elif kind == "gain":
                self.neighbor_gains[message.sender] = message.payload["gain"]
            elif kind == "propose":
                self.received_proposals.append(message)
            elif kind == "accept":
                self.received_accepts.append(message)
            elif kind == "mgm2_gain":
                self.neighbor_group_messages[message.sender] = dict(message.payload)
            elif kind == "dms_q":
                edge = tuple(message.payload["edge"])
                if edge in self.owned_q_messages:
                    self.owned_q_messages[edge][message.sender] = np.array(
                        message.payload["vector"], dtype=float
                    )
            elif kind == "dms_r":
                edge = tuple(message.payload["edge"])
                if edge in self.r_incoming:
                    self.r_incoming[edge] = np.array(message.payload["vector"], dtype=float)
            # "no_propose" and "reject" carry no state.

    # ---- algorithm dispatch ---------------------------------------
    def _step(self) -> None:
        if self.algorithm_name == "dsa-c":
            self._step_dsa_c()
        elif self.algorithm_name == "mgm":
            self._step_mgm()
        elif self.algorithm_name == "mgm2":
            self._step_mgm2()
        elif self.algorithm_name == "dms":
            self._step_dms()
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm_name}")

    def _step_dsa_c(self) -> None:
        best_value, _, current_local, best_local = _best_single(
            self.agent_id, self.value, self.known, self.problem, self.incident
        )
        if best_local <= current_local and self.rng.random() < self.dsa_probability:
            self.set_value(best_value)
        self.broadcast_value()

    def _step_mgm(self) -> None:
        phase = self.local_step % 2
        if phase == 0:
            self.broadcast_value()
            best_value, gain, _, _ = _best_single(
                self.agent_id, self.value, self.known, self.problem, self.incident
            )
            self.best_value, self.gain = best_value, gain
            for neighbor in self.neighbors:
                self.send(neighbor, "gain", {"gain": gain, "best_value": best_value})
        else:
            should_change = self.gain > 0
            for neighbor in self.neighbors:
                neighbor_gain = self.neighbor_gains.get(neighbor, 0)
                if neighbor_gain > self.gain:
                    should_change = False
                    break
                if neighbor_gain == self.gain and neighbor < self.agent_id:
                    should_change = False
                    break
            if should_change:
                self.set_value(self.best_value)
            self.broadcast_value()

    def _step_mgm2(self) -> None:
        phase = self.local_step % 5

        if phase == 0:
            self.received_proposals.clear()
            self.received_accepts.clear()
            self.neighbor_group_messages.clear()
            self.group_id = (self.agent_id,)
            self.group_gain = 0
            self.group_candidate_values = {self.agent_id: self.value}
            self.proposed_to = None

            if self.neighbors and self.rng.random() < self.offer_probability:
                self.proposed_to = self.rng.choice(self.neighbors)
                for neighbor in self.neighbors:
                    if neighbor == self.proposed_to:
                        self.send(neighbor, "propose", {"value": self.value})
                    else:
                        self.send(neighbor, "no_propose", {})
            else:
                for neighbor in self.neighbors:
                    self.send(neighbor, "no_propose", {})

        elif phase == 1:
            if self.proposed_to is not None:
                for proposal in self.received_proposals:
                    self.send(proposal.sender, "reject", {})
                self.broadcast_value()
                return
            if not self.received_proposals:
                self.broadcast_value()
                return

            snapshot = dict(self.known)
            snapshot[self.agent_id] = self.value
            best_message: _Msg | None = None
            best_score = (-float("inf"), 10**9)
            for proposal in self.received_proposals:
                proposer = proposal.sender
                snapshot.setdefault(proposer, int(proposal.payload.get("value", snapshot.get(proposer, 0))))
                _, _, pair_gain = _best_pair_values(proposer, self.agent_id, snapshot, self.problem)
                score = (float(pair_gain), -proposer)
                if score > best_score:
                    best_score, best_message = score, proposal

            if best_message is None:
                return

            proposer = best_message.sender
            first_value, second_value, pair_gain = _best_pair_values(
                proposer, self.agent_id, snapshot, self.problem
            )
            self.group_id = tuple(sorted((proposer, self.agent_id)))
            self.group_gain = pair_gain
            self.group_candidate_values = {proposer: first_value, self.agent_id: second_value}
            self.send(
                proposer,
                "accept",
                {
                    "group_id": self.group_id,
                    "gain": pair_gain,
                    "candidate_values": self.group_candidate_values,
                },
            )
            for proposal in self.received_proposals:
                if proposal.sender != proposer:
                    self.send(proposal.sender, "reject", {})

        elif phase == 2:
            accepted = False
            for accept in self.received_accepts:
                group_id = tuple(accept.payload["group_id"])
                if self.proposed_to in group_id and self.agent_id in group_id:
                    self.group_id = group_id
                    self.group_gain = accept.payload["gain"]
                    self.group_candidate_values = {
                        int(k): int(v) for k, v in accept.payload["candidate_values"].items()
                    }
                    accepted = True
                    break

            if not accepted and self.group_id == (self.agent_id,):
                best_value, gain, _, _ = _best_single(
                    self.agent_id, self.value, self.known, self.problem, self.incident
                )
                self.group_gain = gain
                self.group_candidate_values = {self.agent_id: best_value}

            for neighbor in self.neighbors:
                self.send(
                    neighbor,
                    "mgm2_gain",
                    {"gain": self.group_gain, "leader": min(self.group_id)},
                )

        elif phase == 3:
            if self.group_gain <= 0:
                self.broadcast_value()
                return

            leader = min(self.group_id)
            wins = True
            for neighbor in self.neighbors:
                payload = self.neighbor_group_messages.get(neighbor)
                if not payload:
                    continue
                neighbor_gain = payload.get("gain", 0)
                neighbor_leader = int(payload.get("leader", neighbor))
                if neighbor_gain > self.group_gain:
                    wins = False
                    break
                if neighbor_gain == self.group_gain and neighbor_leader < leader:
                    wins = False
                    break

            if wins and self.agent_id in self.group_candidate_values:
                self.set_value(self.group_candidate_values[self.agent_id])
            self.broadcast_value()

        else:
            self.broadcast_value()

    def _step_dms(self) -> None:
        lam = self.dms_lambda

        # Variable -> function Q messages.
        for edge in self.incident:
            calculated = np.zeros(self.domain_size)
            for other_edge, r_vector in self.r_incoming.items():
                if other_edge != edge:
                    calculated += r_vector
            calculated = _normalize(calculated)
            q_vector = (1.0 - lam) * self.q_old[edge] + lam * calculated
            self.q_old[edge] = q_vector

            owner = min(edge)
            if owner == self.agent_id:
                self.owned_q_messages[edge][self.agent_id] = q_vector.copy()
            else:
                self.send(owner, "dms_q", {"edge": edge, "vector": q_vector.tolist()})

        # Owned function nodes compute and send R messages.
        for edge in self.owned_edges:
            i, j = edge
            matrix = self.dms_matrices[edge]
            q_i = self.owned_q_messages[edge][i]
            q_j = self.owned_q_messages[edge][j]
            calculated_to_i = _normalize(np.min(matrix + q_j.reshape(1, -1), axis=1))
            calculated_to_j = _normalize(np.min(matrix + q_i.reshape(-1, 1), axis=0))

            for endpoint, calculated in ((i, calculated_to_i), (j, calculated_to_j)):
                r_vector = (1.0 - lam) * self.owned_r_old[(edge, endpoint)] + lam * calculated
                self.owned_r_old[(edge, endpoint)] = r_vector
                if endpoint == self.agent_id:
                    self.r_incoming[edge] = r_vector.copy()
                else:
                    self.send(endpoint, "dms_r", {"edge": edge, "vector": r_vector.tolist()})

        belief = np.zeros(self.domain_size)
        for r_vector in self.r_incoming.values():
            belief += r_vector
        self.set_value(int(np.argmin(belief)))
        self.broadcast_value()


# ================================================================
# Simulator
# ================================================================
class AsynchronousSimulator:
    """Run a distributed algorithm with one thread + one mailbox per agent."""

    def __init__(
        self,
        problem: DCOPProblem,
        algorithm: DistributedAlgorithm,
        iterations: int,
        seed: int | None = None,
        progress_callback: Callable[[int], None] | None = None,
    ) -> None:
        if iterations <= 0:
            raise ValueError("iterations must be greater than 0.")

        self.problem = problem
        self.algorithm = algorithm
        self.iterations = iterations
        self.seed = seed
        self.progress_callback = progress_callback

        # Read per-algorithm parameters off the configured algorithm object so the
        # experiment runner can keep building a single algorithm instance per run.
        self.algorithm_name = algorithm.name
        self.dsa_probability = float(getattr(algorithm, "probability", 0.70))
        self.offer_probability = float(getattr(algorithm, "offer_probability", 0.50))
        self.dms_lambda = float(getattr(algorithm, "damping", 0.90))

    def run(self) -> SimulationRunResult:
        started_at = time.perf_counter()
        world = _AsyncWorld(self.problem, self.iterations)
        initial_cost = calculate_global_cost(self.problem, world.snapshot())
        base_seed = self.seed if self.seed is not None else 0

        agents = [
            _AsyncAgent(
                agent_id=i,
                world=world,
                algorithm_name=self.algorithm_name,
                rng_seed=base_seed + 7919 * i,
                dsa_probability=self.dsa_probability,
                offer_probability=self.offer_probability,
                dms_lambda=self.dms_lambda,
            )
            for i in range(self.problem.num_agents)
        ]
        for agent in agents:
            agent.start()

        # Sample one global-cost point per Lamport min-step, until every agent has
        # reached the requested number of local steps.
        cost_history: list[int] = []
        next_step_to_sample = 1
        while next_step_to_sample <= self.iterations:
            current_min = world.min_step()
            while next_step_to_sample <= min(current_min, self.iterations):
                cost_history.append(calculate_global_cost(self.problem, world.snapshot()))
                if self.progress_callback is not None:
                    self.progress_callback(len(cost_history))
                next_step_to_sample += 1
            if world.stop_event.is_set():
                break
            time.sleep(0.001)

        world.stop_event.set()
        for agent in agents:
            agent.join(timeout=1.0)

        while len(cost_history) < self.iterations:
            cost_history.append(calculate_global_cost(self.problem, world.snapshot()))

        runtime_seconds = time.perf_counter() - started_at
        total_messages = sum(agent.messages_sent for agent in agents)
        total_changes = sum(agent.assignment_changes for agent in agents)
        final_assignment = world.snapshot()

        return SimulationRunResult(
            simulator="async",
            algorithm=self.algorithm_name,
            iterations=self.iterations,
            cost_history=cost_history,
            final_assignment=final_assignment,
            total_messages=total_messages,
            runtime_seconds=runtime_seconds,
            metadata={
                "seed": self.seed,
                "messages": total_messages,
                "assignment_changes": total_changes,
                "initial_cost": initial_cost,
                "min_local_step": world.min_step(),
                "async_iteration_definition": (
                    "one local step per agent activation; the global step is the "
                    "minimum local step across agents; the run stops when that "
                    "minimum reaches the iteration limit"
                ),
            },
        )
