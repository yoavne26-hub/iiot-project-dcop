"""
IIOT / Distributed Algorithms - Assignment 2
Random DCOP simulation with DSA-C, MGM, MGM-2 and DMS (Max-Sum + Damping)

This file is intentionally written as a single runnable module, while keeping the
same architectural spirit as Assignment 1:
    * Agent classes with internal state
    * Synchronous simulator with message passing phases
    * Asynchronous simulator with one Thread and one Queue per agent
    * networkx graph for neighbor relations
    * logical clocks in the asynchronous simulator
    * clean run functions, statistics and matplotlib graphs

Dependencies allowed by the assignment:
    networkx, numpy, matplotlib, threading, queue, random, time, copy/dataclasses

Run full experiment:
    python dcop_assignment2_full.py

Run a quick smoke test:
    python dcop_assignment2_full.py --quick-test
"""

from __future__ import annotations

import argparse
import copy
import shutil
import subprocess
import sys
import queue
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# ================================================================
# Constants required by the assignment
# ================================================================
DEFAULT_NUM_PROBLEMS = 50
DEFAULT_NUM_AGENTS = 50
DEFAULT_DOMAIN_SIZE = 10
DEFAULT_CONSTRAINT_PROBABILITY = 0.30
DEFAULT_MAX_COST = 100
DEFAULT_MAX_ITERATIONS = 5000
DEFAULT_SAMPLE_EVERY = 5

DSA_C_P = 0.70
MGM2_OFFER_PROBABILITY = 0.50
DMS_LAMBDA = 0.90

ALGORITHM_NAMES = ("DSA-C", "MGM", "MGM-2", "DMS")


# ================================================================
# Optional sleep prevention for long experiments
# ================================================================
class SleepPreventer:
    """Prevent the computer from sleeping while a long experiment is running.

    The implementation uses only standard-library Python plus operating-system
    commands/APIs:
        * Windows: SetThreadExecutionState through ctypes.
        * macOS: the built-in caffeinate command.
        * Linux: systemd-inhibit when available.

    If the current OS does not expose a supported sleep-inhibit mechanism, the
    class fails safely and the experiment still runs normally.
    """

    def __init__(self, enabled: bool = True, keep_display_on: bool = False) -> None:
        self.enabled = enabled
        self.keep_display_on = keep_display_on
        self._process: Optional[subprocess.Popen] = None
        self._windows_active = False
        self._status = "disabled"

    def __enter__(self) -> "SleepPreventer":
        if not self.enabled:
            print("Sleep prevention: disabled by user flag.")
            return self

        if sys.platform.startswith("win"):
            self._enable_windows()
        elif sys.platform == "darwin":
            self._enable_macos()
        elif sys.platform.startswith("linux"):
            self._enable_linux()
        else:
            self._status = "unsupported"
            print("Sleep prevention: unsupported OS; continuing without inhibition.")

        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
            print("Sleep prevention: released.")

        if self._windows_active:
            try:
                import ctypes

                ES_CONTINUOUS = 0x80000000
                ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
                print("Sleep prevention: released.")
            except Exception as error:
                print(f"Sleep prevention: failed to release Windows state ({error}).")
            finally:
                self._windows_active = False

    def _enable_windows(self) -> None:
        try:
            import ctypes

            ES_CONTINUOUS = 0x80000000
            ES_SYSTEM_REQUIRED = 0x00000001
            ES_DISPLAY_REQUIRED = 0x00000002

            flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED
            if self.keep_display_on:
                flags |= ES_DISPLAY_REQUIRED

            result = ctypes.windll.kernel32.SetThreadExecutionState(flags)
            if result == 0:
                self._status = "failed"
                print("Sleep prevention: Windows API call failed; continuing normally.")
            else:
                self._windows_active = True
                self._status = "active-windows"
                print("Sleep prevention: active via Windows SetThreadExecutionState.")
        except Exception as error:
            self._status = "failed"
            print(f"Sleep prevention: Windows setup failed ({error}); continuing normally.")

    def _enable_macos(self) -> None:
        caffeinate_path = shutil.which("caffeinate")
        if caffeinate_path is None:
            self._status = "unavailable"
            print("Sleep prevention: caffeinate not found; continuing normally.")
            return

        command = [caffeinate_path, "-i", "-s"]
        if self.keep_display_on:
            command.append("-d")

        try:
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._status = "active-macos"
            print("Sleep prevention: active via caffeinate.")
        except Exception as error:
            self._process = None
            self._status = "failed"
            print(f"Sleep prevention: caffeinate setup failed ({error}); continuing normally.")

    def _enable_linux(self) -> None:
        inhibit_path = shutil.which("systemd-inhibit")
        if inhibit_path is None:
            self._status = "unavailable"
            print("Sleep prevention: systemd-inhibit not found; continuing normally.")
            return

        command = [
            inhibit_path,
            "--what=sleep:idle",
            "--who=DCOP simulator",
            "--why=Long IIOT assignment experiment is running",
            "--mode=block",
            "sleep",
            "infinity",
        ]

        try:
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._status = "active-linux"
            print("Sleep prevention: active via systemd-inhibit.")
        except Exception as error:
            self._process = None
            self._status = "failed"
            print(f"Sleep prevention: systemd-inhibit setup failed ({error}); continuing normally.")


# ================================================================
# DCOP representation
# ================================================================
@dataclass(frozen=True)
class Message:
    """A simple message object used by both simulators."""

    sender: int
    receiver: int
    msg_type: str
    payload: Dict[str, Any]
    clock: int = 0
    iteration: int = 0


@dataclass
class Constraint:
    """
    Binary constraint between two agents.

    The matrix is stored in canonical order (agent_i < agent_j):
        cost_matrix[value_of_agent_i, value_of_agent_j] -> cost
    """

    agent_i: int
    agent_j: int
    cost_matrix: np.ndarray

    def __post_init__(self) -> None:
        if self.agent_i > self.agent_j:
            self.agent_i, self.agent_j = self.agent_j, self.agent_i
            self.cost_matrix = self.cost_matrix.T.copy()

    @property
    def edge(self) -> Tuple[int, int]:
        return (self.agent_i, self.agent_j)

    def cost(self, value_i: int, value_j: int) -> int:
        return int(self.cost_matrix[value_i, value_j])

    def cost_for(self, agent_id: int, agent_value: int, other_value: int) -> int:
        if agent_id == self.agent_i:
            return int(self.cost_matrix[agent_value, other_value])
        if agent_id == self.agent_j:
            return int(self.cost_matrix[other_value, agent_value])
        raise ValueError(f"Agent {agent_id} is not part of constraint {self.edge}")


@dataclass
class DCOPProblem:
    """A generated DCOP instance: graph, domains, constraints and initial assignment."""

    num_agents: int
    domain_size: int
    graph: nx.Graph
    constraints: Dict[Tuple[int, int], Constraint]
    initial_assignment: Dict[int, int]
    seed: int

    @property
    def domain(self) -> List[int]:
        return list(range(self.domain_size))

    def neighbors(self, agent_id: int) -> List[int]:
        return list(self.graph.neighbors(agent_id))

    def get_constraint(self, a: int, b: int) -> Constraint:
        return self.constraints[tuple(sorted((a, b)))]

    def has_constraint(self, a: int, b: int) -> bool:
        return tuple(sorted((a, b))) in self.constraints

    def constraint_cost(self, a: int, value_a: int, b: int, value_b: int) -> int:
        return self.get_constraint(a, b).cost_for(a, value_a, value_b)

    def total_cost(self, assignment: Dict[int, int]) -> int:
        """Global cost. Every binary constraint is counted exactly once."""
        total = 0
        for (i, j), constraint in self.constraints.items():
            total += constraint.cost(assignment[i], assignment[j])
        return int(total)

    def local_cost(
        self,
        agent_id: int,
        value: Optional[int] = None,
        known_assignments: Optional[Dict[int, int]] = None,
    ) -> int:
        """Cost of one agent against its neighbors, using the latest known assignments."""
        if known_assignments is None:
            known_assignments = self.initial_assignment
        if value is None:
            value = known_assignments[agent_id]

        total = 0
        for neighbor in self.neighbors(agent_id):
            if neighbor not in known_assignments:
                continue
            total += self.constraint_cost(agent_id, value, neighbor, known_assignments[neighbor])
        return int(total)

    def best_single_value(
        self,
        agent_id: int,
        current_value: int,
        known_assignments: Dict[int, int],
    ) -> Tuple[int, int, int, int]:
        """
        Return best alternative for a single agent.

        Returns:
            (best_value, gain, current_local_cost, best_local_cost)

        gain is positive when the move improves the cost.
        """
        current_local = self.local_cost(agent_id, current_value, known_assignments)
        best_value = current_value
        best_cost = current_local

        # Tie-breaking by lower value gives deterministic behavior when costs are equal.
        for candidate_value in self.domain:
            candidate_cost = self.local_cost(agent_id, candidate_value, known_assignments)
            if candidate_cost < best_cost or (
                candidate_cost == best_cost and candidate_value < best_value
            ):
                best_value = candidate_value
                best_cost = candidate_cost

        gain = current_local - best_cost
        return best_value, int(gain), int(current_local), int(best_cost)

    def pair_delta_cost(
        self,
        first: int,
        second: int,
        first_new_value: int,
        second_new_value: int,
        known_assignments: Dict[int, int],
    ) -> Tuple[int, int, int]:
        """
        Compute the 2-opt gain for a pair of agents.

        Only constraints touching first or second can change. Constraints are still
        counted once, including the constraint between the two paired agents.
        """
        affected_edges = set()
        for agent_id in (first, second):
            for neighbor in self.neighbors(agent_id):
                affected_edges.add(tuple(sorted((agent_id, neighbor))))

        old_cost = 0
        new_cost = 0
        for edge in affected_edges:
            i, j = edge
            old_i = known_assignments[i]
            old_j = known_assignments[j]
            new_i = first_new_value if i == first else second_new_value if i == second else old_i
            new_j = first_new_value if j == first else second_new_value if j == second else old_j
            constraint = self.constraints[edge]
            old_cost += constraint.cost(old_i, old_j)
            new_cost += constraint.cost(new_i, new_j)

        return int(old_cost - new_cost), int(old_cost), int(new_cost)

    def best_pair_values(
        self,
        first: int,
        second: int,
        known_assignments: Dict[int, int],
    ) -> Tuple[int, int, int]:
        """Return (best_first_value, best_second_value, pair_gain) for a true 2-opt move."""
        current_first = known_assignments[first]
        current_second = known_assignments[second]
        best_first = current_first
        best_second = current_second
        best_gain, _, _ = self.pair_delta_cost(
            first, second, current_first, current_second, known_assignments
        )

        for value_first in self.domain:
            for value_second in self.domain:
                gain, _, _ = self.pair_delta_cost(
                    first, second, value_first, value_second, known_assignments
                )
                if gain > best_gain:
                    best_gain = gain
                    best_first = value_first
                    best_second = value_second
                elif gain == best_gain:
                    # Deterministic tie-breaking over joint assignments.
                    if (value_first, value_second) < (best_first, best_second):
                        best_first = value_first
                        best_second = value_second

        return best_first, best_second, int(best_gain)

    def clone(self) -> "DCOPProblem":
        """Return a safe copy. Constraint matrices are copied because algorithms should not share mutable state."""
        copied_constraints = {
            edge: Constraint(c.agent_i, c.agent_j, c.cost_matrix.copy())
            for edge, c in self.constraints.items()
        }
        return DCOPProblem(
            num_agents=self.num_agents,
            domain_size=self.domain_size,
            graph=self.graph.copy(),
            constraints=copied_constraints,
            initial_assignment=dict(self.initial_assignment),
            seed=self.seed,
        )

    @staticmethod
    def generate_random(
        num_agents: int,
        constraint_probability: float,
        domain_size: int,
        max_cost: int,
        seed: int,
        ensure_at_least_one_edge: bool = True,
    ) -> "DCOPProblem":
        """
        Generate a random binary DCOP problem.

        The graph follows the requested Erdos-Renyi model with probability p.
        If the generated graph has zero edges, we add one edge so asynchronous
        simulations do not get stuck only because no messages can ever arrive.
        """
        py_rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)

        graph = nx.Graph()
        graph.add_nodes_from(range(num_agents))

        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                if py_rng.random() < constraint_probability:
                    graph.add_edge(i, j)

        if ensure_at_least_one_edge and num_agents > 1 and graph.number_of_edges() == 0:
            graph.add_edge(0, 1)

        constraints: Dict[Tuple[int, int], Constraint] = {}
        for i, j in graph.edges():
            matrix = np_rng.integers(1, max_cost + 1, size=(domain_size, domain_size), dtype=np.int64)
            edge = tuple(sorted((i, j)))
            constraints[edge] = Constraint(edge[0], edge[1], matrix)

        initial_assignment = {agent_id: py_rng.randrange(domain_size) for agent_id in range(num_agents)}

        return DCOPProblem(
            num_agents=num_agents,
            domain_size=domain_size,
            graph=graph,
            constraints=constraints,
            initial_assignment=initial_assignment,
            seed=seed,
        )


# ================================================================
# Agent classes
# ================================================================
class VariableAgent:
    """A DCOP variable owned by one distributed agent."""

    def __init__(self, agent_id: int, problem: DCOPProblem, rng_seed: int):
        self.agent_id = agent_id
        self.problem = problem
        self.neighbors = problem.neighbors(agent_id)
        self.domain = problem.domain
        self.value = problem.initial_assignment[agent_id]
        self.known_assignments: Dict[int, int] = dict(problem.initial_assignment)
        self.inbox: List[Message] = []
        self.messages_sent = 0
        self.assignment_changes = 0
        self.logical_round = 0
        self.rng = random.Random(rng_seed)

        # State used by MGM / MGM-2.
        self.best_value = self.value
        self.gain = 0
        self.neighbor_gains: Dict[int, float] = {n: 0.0 for n in self.neighbors}
        self.group_id: Tuple[int, ...] = (self.agent_id,)
        self.group_gain = 0.0
        self.group_candidate_values: Dict[int, int] = {self.agent_id: self.value}
        self.neighbor_group_messages: Dict[int, Dict[str, Any]] = {}
        self.proposed_to: Optional[int] = None
        self.received_proposals: List[Message] = []
        self.received_accepts: List[Message] = []

    def receive_message(self, message: Message) -> None:
        self.inbox.append(message)

    def drain_inbox(self) -> List[Message]:
        messages = self.inbox
        self.inbox = []
        return messages

    def process_messages(self, messages: Iterable[Message]) -> None:
        for message in messages:
            if message.msg_type == "ASSIGNMENT":
                self.known_assignments[message.sender] = int(message.payload["value"])
            elif message.msg_type == "GAIN":
                self.neighbor_gains[message.sender] = float(message.payload["gain"])
            elif message.msg_type == "MGM2_GAIN":
                self.neighbor_group_messages[message.sender] = dict(message.payload)
            elif message.msg_type == "PROPOSE":
                self.received_proposals.append(message)
            elif message.msg_type == "ACCEPT":
                self.received_accepts.append(message)
            elif message.msg_type == "REJECT":
                # Rejection is informative but does not require state beyond clearing old accepts.
                pass

    def set_value(self, new_value: int) -> bool:
        if new_value != self.value:
            self.value = int(new_value)
            self.known_assignments[self.agent_id] = self.value
            self.assignment_changes += 1
            return True
        self.known_assignments[self.agent_id] = self.value
        return False

    def assignment_message(self, receiver: int, iteration: int = 0) -> Message:
        return Message(
            sender=self.agent_id,
            receiver=receiver,
            msg_type="ASSIGNMENT",
            payload={"value": self.value},
            clock=self.logical_round,
            iteration=iteration,
        )

    def gain_message(self, receiver: int, iteration: int = 0) -> Message:
        return Message(
            sender=self.agent_id,
            receiver=receiver,
            msg_type="GAIN",
            payload={"gain": self.gain, "best_value": self.best_value},
            clock=self.logical_round,
            iteration=iteration,
        )

    def mgm2_gain_message(self, receiver: int, iteration: int = 0) -> Message:
        return Message(
            sender=self.agent_id,
            receiver=receiver,
            msg_type="MGM2_GAIN",
            payload={
                "gain": self.group_gain,
                "group_id": self.group_id,
                "members": list(self.group_id),
                "leader": min(self.group_id),
            },
            clock=self.logical_round,
            iteration=iteration,
        )

    def compute_best_single_move(self) -> Tuple[int, int]:
        best_value, gain, _, _ = self.problem.best_single_value(
            self.agent_id, self.value, self.known_assignments
        )
        self.best_value = best_value
        self.gain = gain
        return best_value, gain


# ================================================================
# Synchronous simulator
# ================================================================
@dataclass
class SimulationResult:
    algorithm: str
    simulator: str
    costs: List[int]
    initial_cost: int
    final_cost: int
    messages_sent: int
    assignment_changes: int
    runtime_seconds: float
    converged_at: Optional[int] = None


class SynchronousSimulator:
    """
    Round-based simulator.

    The simulator is deliberately not written as one monolithic algorithmic loop.
    Each VariableAgent holds its own local state, inbox, latest neighbor values and
    algorithm state. The simulator only delivers messages and calls agent methods
    in logical phases, similar to a distributed synchronous round/barrier model.
    """

    def __init__(self, problem: DCOPProblem, max_iterations: int, rng_seed: int):
        self.problem = problem.clone()
        self.max_iterations = max_iterations
        self.rng_seed = rng_seed
        self.agents: Dict[int, VariableAgent] = {
            agent_id: VariableAgent(agent_id, self.problem, rng_seed + 1009 * agent_id)
            for agent_id in range(self.problem.num_agents)
        }
        self.extra_messages_sent = 0
        self._dms_state: Optional[DMSState] = None

    def current_assignment(self) -> Dict[int, int]:
        return {agent_id: agent.value for agent_id, agent in self.agents.items()}

    def deliver_messages(self, messages: Iterable[Message]) -> None:
        for message in messages:
            self.agents[message.receiver].receive_message(message)
            self.agents[message.sender].messages_sent += 1

    def broadcast_assignments(self, iteration: int) -> None:
        messages: List[Message] = []
        for agent in self.agents.values():
            for neighbor in agent.neighbors:
                messages.append(agent.assignment_message(neighbor, iteration))
        self.deliver_messages(messages)
        for agent in self.agents.values():
            agent.process_messages(agent.drain_inbox())

    def broadcast_gains(self, iteration: int) -> None:
        messages: List[Message] = []
        for agent in self.agents.values():
            for neighbor in agent.neighbors:
                messages.append(agent.gain_message(neighbor, iteration))
        self.deliver_messages(messages)
        for agent in self.agents.values():
            agent.process_messages(agent.drain_inbox())

    def broadcast_mgm2_gains(self, iteration: int) -> None:
        messages: List[Message] = []
        for agent in self.agents.values():
            for neighbor in agent.neighbors:
                messages.append(agent.mgm2_gain_message(neighbor, iteration))
        self.deliver_messages(messages)
        for agent in self.agents.values():
            agent.process_messages(agent.drain_inbox())

    def run(self, algorithm_name: str) -> SimulationResult:
        started = time.time()
        initial_cost = self.problem.total_cost(self.current_assignment())
        costs: List[int] = []
        converged_at: Optional[int] = None

        for iteration in range(1, self.max_iterations + 1):
            if algorithm_name == "DSA-C":
                converged = self._iteration_dsa_c(iteration)
            elif algorithm_name == "MGM":
                converged = self._iteration_mgm(iteration)
            elif algorithm_name == "MGM-2":
                converged = self._iteration_mgm2(iteration)
            elif algorithm_name == "DMS":
                converged = self._iteration_dms(iteration)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm_name}")

            costs.append(self.problem.total_cost(self.current_assignment()))

            if converged and converged_at is None:
                converged_at = iteration
                # The graph must still have values up to max_iterations.
                costs.extend([costs[-1]] * (self.max_iterations - iteration))
                break

        runtime = time.time() - started
        total_messages = sum(agent.messages_sent for agent in self.agents.values()) + self.extra_messages_sent
        total_changes = sum(agent.assignment_changes for agent in self.agents.values())
        final_cost = costs[-1] if costs else initial_cost
        return SimulationResult(
            algorithm=algorithm_name,
            simulator="Synchronous",
            costs=costs,
            initial_cost=initial_cost,
            final_cost=final_cost,
            messages_sent=total_messages,
            assignment_changes=total_changes,
            runtime_seconds=runtime,
            converged_at=converged_at,
        )

    def _iteration_dsa_c(self, iteration: int) -> bool:
        self.broadcast_assignments(iteration)
        for agent in self.agents.values():
            best_value, _, current_local, best_local = self.problem.best_single_value(
                agent.agent_id, agent.value, agent.known_assignments
            )
            # DSA-C: move to the best value if it improves OR equals the current local cost,
            # with probability p. There is no extra condition such as current_cost != 0.
            if best_local <= current_local and agent.rng.random() < DSA_C_P:
                agent.set_value(best_value)
            agent.logical_round += 1
        return False  # Stochastic DSA-C is normally run until max iterations.

    def _iteration_mgm(self, iteration: int) -> bool:
        self.broadcast_assignments(iteration)

        any_positive_gain = False
        for agent in self.agents.values():
            _, gain = agent.compute_best_single_move()
            if gain > 0:
                any_positive_gain = True

        self.broadcast_gains(iteration)

        for agent in self.agents.values():
            should_change = agent.gain > 0
            for neighbor in agent.neighbors:
                neighbor_gain = agent.neighbor_gains.get(neighbor, 0.0)
                if neighbor_gain > agent.gain:
                    should_change = False
                    break
                if neighbor_gain == agent.gain and neighbor < agent.agent_id:
                    should_change = False
                    break
            if should_change:
                agent.set_value(agent.best_value)
            agent.logical_round += 1

        # MGM converges to a 1-opt local optimum when no agent has a positive gain.
        return not any_positive_gain

    def _iteration_mgm2(self, iteration: int) -> bool:
        self.broadcast_assignments(iteration)
        rng = random.Random(self.rng_seed + 30011 * iteration)

        # Clear round-local MGM-2 state.
        for agent in self.agents.values():
            agent.group_id = (agent.agent_id,)
            agent.group_gain = 0.0
            agent.group_candidate_values = {agent.agent_id: agent.value}
            agent.neighbor_group_messages.clear()
            agent.received_proposals.clear()
            agent.received_accepts.clear()
            agent.proposed_to = None

        # Step 1: decide whether to offer. If not offering, explicitly send NO_PROPOSE.
        proposal_messages: List[Message] = []
        for agent in self.agents.values():
            if agent.neighbors and rng.random() < MGM2_OFFER_PROBABILITY:
                chosen_neighbor = rng.choice(agent.neighbors)
                agent.proposed_to = chosen_neighbor
                for neighbor in agent.neighbors:
                    if neighbor == chosen_neighbor:
                        proposal_messages.append(
                            Message(
                                sender=agent.agent_id,
                                receiver=neighbor,
                                msg_type="PROPOSE",
                                payload={
                                    "value": agent.value,
                                    "domain": list(agent.domain),
                                    "known_assignments": dict(agent.known_assignments),
                                    # Constraint matrices are globally known in this simulator.
                                    # The payload is kept explicit to match the assignment wording.
                                    "constraints": [tuple(sorted((agent.agent_id, n))) for n in agent.neighbors],
                                },
                                clock=agent.logical_round,
                                iteration=iteration,
                            )
                        )
                    else:
                        proposal_messages.append(
                            Message(agent.agent_id, neighbor, "NO_PROPOSE", {}, agent.logical_round, iteration)
                        )
            else:
                for neighbor in agent.neighbors:
                    proposal_messages.append(
                        Message(agent.agent_id, neighbor, "NO_PROPOSE", {}, agent.logical_round, iteration)
                    )

        self.deliver_messages(proposal_messages)
        for agent in self.agents.values():
            agent.process_messages(agent.drain_inbox())

        # Step 2: non-offering agents choose one proposal. We choose the proposal with the
        # largest computed pair gain, tie-broken by lower proposer index.
        pair_groups: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for receiver in self.agents.values():
            if receiver.proposed_to is not None:
                continue  # A proposer does not accept proposals from others.
            if not receiver.received_proposals:
                continue

            best_offer: Optional[Message] = None
            best_tuple: Tuple[float, int] = (-float("inf"), 10**9)
            for offer in receiver.received_proposals:
                proposer = offer.sender
                first, second = proposer, receiver.agent_id
                best_first, best_second, pair_gain = self.problem.best_pair_values(
                    first, second, self.current_assignment()
                )
                score = (pair_gain, -proposer)
                if score > best_tuple:
                    best_tuple = score
                    best_offer = offer

            if best_offer is None:
                continue

            proposer = best_offer.sender
            first, second = proposer, receiver.agent_id
            best_first, best_second, pair_gain = self.problem.best_pair_values(
                first, second, self.current_assignment()
            )
            group_id = tuple(sorted((first, second)))
            candidate_values = {first: best_first, second: best_second}
            pair_groups[group_id] = {
                "gain": pair_gain,
                "members": group_id,
                "candidate_values": candidate_values,
                "leader": min(group_id),
            }

        paired_agents = {member for group in pair_groups for member in group}

        # Non-paired agents participate as singleton MGM moves. This is a standard practical
        # MGM-2 choice: the algorithm still performs true 2-opt for accepted pairs, while agents
        # without a partner can compete with a normal 1-opt gain instead of idling forever.
        all_groups: Dict[Tuple[int, ...], Dict[str, Any]] = dict(pair_groups)
        for agent in self.agents.values():
            if agent.agent_id in paired_agents:
                continue
            best_value, gain = agent.compute_best_single_move()
            all_groups[(agent.agent_id,)] = {
                "gain": gain,
                "members": (agent.agent_id,),
                "candidate_values": {agent.agent_id: best_value},
                "leader": agent.agent_id,
            }

        for group_id, group in all_groups.items():
            for member in group["members"]:
                agent = self.agents[member]
                agent.group_id = tuple(group["members"])
                agent.group_gain = float(group["gain"])
                agent.group_candidate_values = dict(group["candidate_values"])

        self.broadcast_mgm2_gains(iteration)

        # Step 4: choose locally maximal improving groups. Tie-breaking: lower leader index wins.
        winning_groups: List[Tuple[int, ...]] = []
        for group_id, group in all_groups.items():
            gain = float(group["gain"])
            if gain <= 0:
                continue
            leader = int(group["leader"])
            members = set(group["members"])
            wins = True

            neighbor_agents = set()
            for member in members:
                neighbor_agents.update(self.agents[member].neighbors)
            neighbor_agents.difference_update(members)

            for neighbor in neighbor_agents:
                neighbor_group = self.agents[neighbor].group_id
                if neighbor_group == group_id:
                    continue
                neighbor_gain = self.agents[neighbor].group_gain
                neighbor_leader = min(neighbor_group)
                if neighbor_gain > gain:
                    wins = False
                    break
                if neighbor_gain == gain and neighbor_leader < leader:
                    wins = False
                    break

            if wins:
                winning_groups.append(group_id)

        changed_any = False
        for group_id in winning_groups:
            candidate_values = all_groups[group_id]["candidate_values"]
            for member, new_value in candidate_values.items():
                changed_any = self.agents[member].set_value(new_value) or changed_any

        for agent in self.agents.values():
            agent.logical_round += 1

        # Because offers are random, absence of a positive chosen group in one iteration is not a
        # proof of global MGM-2 convergence. Therefore the full max_iterations are used.
        return False

    def _iteration_dms(self, iteration: int) -> bool:
        if self._dms_state is None:
            self._dms_state = DMSState(self.problem, DMS_LAMBDA)
        changed = self._dms_state.sync_iteration(self.agents)
        self.extra_messages_sent += self._dms_state.last_iteration_messages
        for agent in self.agents.values():
            agent.logical_round += 1
        return self._dms_state.stable_rounds >= self._dms_state.required_stable_rounds


# ================================================================
# DMS / Min-Sum with damping
# ================================================================
class DMSState:
    """
    State for Max-Sum + Damping.

    The assignment is a cost-minimization DCOP, so this implementation uses the
    equivalent Min-Sum formulation instead of negating all costs into utilities.
    Therefore variable beliefs are sums of R cost messages and each variable
    chooses the value with the minimum belief. This is equivalent to Max-Sum over
    negative utilities, and it keeps the code aligned with the cost matrices.
    """

    def __init__(self, problem: DCOPProblem, damping_lambda: float):
        self.problem = problem
        self.damping_lambda = damping_lambda
        self.edges = sorted(problem.constraints.keys())
        self.incident_edges: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(problem.num_agents)}
        for edge in self.edges:
            i, j = edge
            self.incident_edges[i].append(edge)
            self.incident_edges[j].append(edge)

        domain_size = problem.domain_size
        zeros = lambda: np.zeros(domain_size, dtype=float)
        self.q_messages: Dict[Tuple[int, Tuple[int, int]], np.ndarray] = {}
        self.r_messages: Dict[Tuple[Tuple[int, int], int], np.ndarray] = {}
        for edge in self.edges:
            i, j = edge
            self.q_messages[(i, edge)] = zeros()
            self.q_messages[(j, edge)] = zeros()
            self.r_messages[(edge, i)] = zeros()
            self.r_messages[(edge, j)] = zeros()

        self.last_assignment: Optional[Dict[int, int]] = None
        self.stable_rounds = 0
        self.required_stable_rounds = 25
        self.last_iteration_messages = 0

    @staticmethod
    def normalize(message: np.ndarray) -> np.ndarray:
        """Normalize a DMS message by shifting its minimum entry to 0.

        The assignment asks for a normalization where every message vector has
        at least one zero entry after normalization. Subtracting the minimum
        value preserves all pairwise differences in the vector, so the Min-Sum
        decision is unchanged while message magnitudes remain bounded.
        """
        return message - float(np.min(message))

    def sync_iteration(self, agents: Dict[int, VariableAgent]) -> bool:
        lam = self.damping_lambda
        self.last_iteration_messages = 0

        # Variable -> function Q messages.
        new_q: Dict[Tuple[int, Tuple[int, int]], np.ndarray] = {}
        for edge in self.edges:
            i, j = edge
            for variable in (i, j):
                incoming_sum = np.zeros(self.problem.domain_size, dtype=float)
                for other_edge in self.incident_edges[variable]:
                    if other_edge == edge:
                        continue
                    incoming_sum += self.r_messages[(other_edge, variable)]
                calculated = self.normalize(incoming_sum)
                old = self.q_messages[(variable, edge)]
                new_q[(variable, edge)] = (1.0 - lam) * old + lam * calculated
                self.last_iteration_messages += 1
        self.q_messages.update(new_q)

        # Function -> variable R messages.
        new_r: Dict[Tuple[Tuple[int, int], int], np.ndarray] = {}
        for edge in self.edges:
            i, j = edge
            matrix = self.problem.constraints[edge].cost_matrix.astype(float)
            q_i = self.q_messages[(i, edge)]
            q_j = self.q_messages[(j, edge)]

            # R_{f->i}(x_i) = min_{x_j} cost(x_i,x_j) + Q_{j->f}(x_j)
            calculated_to_i = np.min(matrix + q_j.reshape(1, -1), axis=1)
            calculated_to_j = np.min(matrix + q_i.reshape(-1, 1), axis=0)
            calculated_to_i = self.normalize(calculated_to_i)
            calculated_to_j = self.normalize(calculated_to_j)

            old_i = self.r_messages[(edge, i)]
            old_j = self.r_messages[(edge, j)]
            new_r[(edge, i)] = (1.0 - lam) * old_i + lam * calculated_to_i
            new_r[(edge, j)] = (1.0 - lam) * old_j + lam * calculated_to_j
            self.last_iteration_messages += 2
        self.r_messages.update(new_r)

        changed = False
        current_assignment: Dict[int, int] = {}
        for agent_id, agent in agents.items():
            belief = np.zeros(self.problem.domain_size, dtype=float)
            for edge in self.incident_edges[agent_id]:
                belief += self.r_messages[(edge, agent_id)]
            new_value = int(np.argmin(belief))
            if agent.set_value(new_value):
                changed = True
            current_assignment[agent_id] = agent.value

        if self.last_assignment == current_assignment:
            self.stable_rounds += 1
        else:
            self.stable_rounds = 0
        self.last_assignment = current_assignment
        return changed


# ================================================================
# Asynchronous simulator with Thread + Queue per agent
# ================================================================
class RestrictedMailbox:
    """A limited mailbox view given to one asynchronous agent.

    The actual queues are stored only inside AsyncWorld.  An agent receives this
    wrapper instead of the full queues dictionary, so it can read only its own
    inbox and can send messages only to itself or to graph neighbors.  This keeps
    the simulator closer to a distributed message-passing model and prevents
    accidental access to non-neighbor mailboxes.
    """

    def __init__(self, owner_id: int, allowed_receivers: Iterable[int], queues: Dict[int, queue.Queue]):
        self.owner_id = owner_id
        self._allowed_receivers = set(allowed_receivers) | {owner_id}
        self._own_queue = queues[owner_id]
        self._neighbor_outboxes = {receiver: queues[receiver] for receiver in self._allowed_receivers}

    def put(self, receiver: int, message: Message) -> None:
        if receiver not in self._allowed_receivers:
            raise PermissionError(
                f"Agent {self.owner_id} is not allowed to send directly to non-neighbor agent {receiver}"
            )
        self._neighbor_outboxes[receiver].put(message)

    def get(self, timeout: Optional[float] = None) -> Message:
        return self._own_queue.get(timeout=timeout)

    def get_nowait(self) -> Message:
        return self._own_queue.get_nowait()


class AsyncWorld:
    """Shared state between asynchronous agent threads."""

    def __init__(self, problem: DCOPProblem, max_steps: int):
        self.problem = problem
        self.max_steps = max_steps
        self.values: Dict[int, int] = dict(problem.initial_assignment)
        self.values_lock = threading.Lock()
        self.clock_lock = threading.Lock()
        self.clocks: Dict[int, int] = {i: 0 for i in range(problem.num_agents)}
        self.stop_event = threading.Event()
        # Private raw queues. Agents do not receive this dictionary directly;
        # they get a RestrictedMailbox containing only their own inbox and
        # neighbor outboxes.
        self._queues: Dict[int, queue.Queue] = {i: queue.Queue() for i in range(problem.num_agents)}
        self.start_barrier = threading.Barrier(problem.num_agents) if problem.num_agents > 0 else None

    def mailbox_for(self, agent_id: int) -> RestrictedMailbox:
        return RestrictedMailbox(agent_id, self.problem.neighbors(agent_id), self._queues)

    def snapshot_assignment(self) -> Dict[int, int]:
        with self.values_lock:
            return dict(self.values)

    def update_value(self, agent_id: int, value: int) -> bool:
        with self.values_lock:
            changed = self.values.get(agent_id) != value
            self.values[agent_id] = int(value)
            return changed

    def update_clock(self, agent_id: int, clock: int) -> None:
        with self.clock_lock:
            self.clocks[agent_id] = int(clock)
            if min(self.clocks.values()) >= self.max_steps:
                self.stop_event.set()

    def min_clock(self) -> int:
        with self.clock_lock:
            return min(self.clocks.values()) if self.clocks else self.max_steps


class AsyncVariableAgent(threading.Thread):
    """Threaded asynchronous DCOP agent."""

    def __init__(
        self,
        agent_id: int,
        algorithm_name: str,
        world: AsyncWorld,
        rng_seed: int,
        dms_lambda: float = DMS_LAMBDA,
    ):
        super().__init__()
        self.daemon = True
        self.agent_id = agent_id
        self.algorithm_name = algorithm_name
        self.world = world
        self.mailbox = world.mailbox_for(agent_id)
        self.problem = world.problem
        self.neighbors = self.problem.neighbors(agent_id)
        self.domain = self.problem.domain
        self.value = self.problem.initial_assignment[agent_id]
        self.known_assignments = dict(self.problem.initial_assignment)
        self.logical_round = 0
        self.rng = random.Random(rng_seed)
        self.messages_sent = 0
        self.assignment_changes = 0

        # MGM state.
        self.best_value = self.value
        self.gain = 0.0
        self.neighbor_gains: Dict[int, float] = {n: 0.0 for n in self.neighbors}

        # MGM-2 state.
        self.proposed_to: Optional[int] = None
        self.received_proposals: List[Message] = []
        self.received_accepts: List[Message] = []
        self.group_id: Tuple[int, ...] = (self.agent_id,)
        self.group_gain: float = 0.0
        self.group_candidate_values: Dict[int, int] = {self.agent_id: self.value}
        self.neighbor_group_messages: Dict[int, Dict[str, Any]] = {}

        # DMS state. This agent owns function nodes for edges where it has the lower id.
        self.dms_lambda = dms_lambda
        self.incident_edges = [tuple(sorted((self.agent_id, n))) for n in self.neighbors]
        self.owned_edges = [edge for edge in self.incident_edges if min(edge) == self.agent_id]
        self.q_old: Dict[Tuple[int, int], np.ndarray] = {
            edge: np.zeros(self.problem.domain_size, dtype=float) for edge in self.incident_edges
        }
        self.r_incoming: Dict[Tuple[int, int], np.ndarray] = {
            edge: np.zeros(self.problem.domain_size, dtype=float) for edge in self.incident_edges
        }
        self.owned_q_messages: Dict[Tuple[int, int], Dict[int, np.ndarray]] = {
            edge: {
                edge[0]: np.zeros(self.problem.domain_size, dtype=float),
                edge[1]: np.zeros(self.problem.domain_size, dtype=float),
            }
            for edge in self.owned_edges
        }
        self.owned_r_old: Dict[Tuple[Tuple[int, int], int], np.ndarray] = {}
        for edge in self.owned_edges:
            for endpoint in edge:
                self.owned_r_old[(edge, endpoint)] = np.zeros(self.problem.domain_size, dtype=float)

    @staticmethod
    def normalize(message: np.ndarray) -> np.ndarray:
        """Normalize a DMS message by shifting its minimum entry to 0.

        The assignment asks for a normalization where every message vector has
        at least one zero entry after normalization. Subtracting the minimum
        value preserves all pairwise differences in the vector, so the Min-Sum
        decision is unchanged while message magnitudes remain bounded.
        """
        return message - float(np.min(message))

    def send(self, receiver: int, msg_type: str, payload: Dict[str, Any]) -> None:
        if self.world.stop_event.is_set():
            return
        message = Message(
            sender=self.agent_id,
            receiver=receiver,
            msg_type=msg_type,
            payload=payload,
            clock=self.logical_round,
            iteration=self.logical_round,
        )
        self.mailbox.put(receiver, message)
        self.messages_sent += 1

    def broadcast_assignment(self) -> None:
        for neighbor in self.neighbors:
            self.send(neighbor, "ASSIGNMENT", {"value": self.value})

    def process_messages(self, messages: Sequence[Message]) -> None:
        for message in messages:
            if message.msg_type == "ASSIGNMENT":
                self.known_assignments[message.sender] = int(message.payload["value"])
            elif message.msg_type == "GAIN":
                self.neighbor_gains[message.sender] = float(message.payload["gain"])
            elif message.msg_type == "PROPOSE":
                self.received_proposals.append(message)
            elif message.msg_type == "ACCEPT":
                self.received_accepts.append(message)
            elif message.msg_type == "MGM2_GAIN":
                self.neighbor_group_messages[message.sender] = dict(message.payload)
            elif message.msg_type == "DMS_Q":
                edge = tuple(message.payload["edge"])
                if edge in self.owned_q_messages:
                    self.owned_q_messages[edge][message.sender] = np.array(message.payload["vector"], dtype=float)
            elif message.msg_type == "DMS_R":
                edge = tuple(message.payload["edge"])
                if edge in self.r_incoming:
                    self.r_incoming[edge] = np.array(message.payload["vector"], dtype=float)

    def set_value(self, new_value: int) -> None:
        new_value = int(new_value)
        if new_value != self.value:
            self.assignment_changes += 1
        self.value = new_value
        self.known_assignments[self.agent_id] = self.value
        self.world.update_value(self.agent_id, self.value)

    def compute_best_single_move(self) -> Tuple[int, float]:
        best_value, gain, _, _ = self.problem.best_single_value(
            self.agent_id, self.value, self.known_assignments
        )
        self.best_value = best_value
        self.gain = float(gain)
        return best_value, float(gain)

    def run(self) -> None:
        if self.world.start_barrier is not None:
            self.world.start_barrier.wait()

        # Initial assignment broadcast, same as Assignment 1's initial table broadcast.
        self.broadcast_assignment()

        if not self.neighbors:
            self._run_isolated_agent()
            return

        while not self.world.stop_event.is_set():
            try:
                first_message = self.mailbox.get(timeout=0.05)
            except queue.Empty:
                continue

            messages = [first_message]
            while True:
                try:
                    messages.append(self.mailbox.get_nowait())
                except queue.Empty:
                    break

            self.process_messages(messages)
            self._algorithm_step()
            self.logical_round += 1
            self.world.update_clock(self.agent_id, self.logical_round)

    def _run_isolated_agent(self) -> None:
        # An isolated agent has no constraints and no incoming messages. It advances its
        # logical clock so the asynchronous global min-clock stopping rule can still finish.
        while not self.world.stop_event.is_set() and self.logical_round < self.world.max_steps:
            self.logical_round += 1
            self.world.update_clock(self.agent_id, self.logical_round)
            time.sleep(0.0001)

    def _algorithm_step(self) -> None:
        if self.algorithm_name == "DSA-C":
            self._step_dsa_c()
        elif self.algorithm_name == "MGM":
            self._step_mgm()
        elif self.algorithm_name == "MGM-2":
            self._step_mgm2()
        elif self.algorithm_name == "DMS":
            self._step_dms()
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm_name}")

    def _step_dsa_c(self) -> None:
        best_value, _, current_local, best_local = self.problem.best_single_value(
            self.agent_id, self.value, self.known_assignments
        )
        if best_local <= current_local and self.rng.random() < DSA_C_P:
            self.set_value(best_value)
        self.broadcast_assignment()

    def _step_mgm(self) -> None:
        phase = self.logical_round % 2
        if phase == 0:
            self.broadcast_assignment()
            self.compute_best_single_move()
            for neighbor in self.neighbors:
                self.send(neighbor, "GAIN", {"gain": self.gain, "best_value": self.best_value})
        else:
            should_change = self.gain > 0
            for neighbor in self.neighbors:
                neighbor_gain = self.neighbor_gains.get(neighbor, 0.0)
                if neighbor_gain > self.gain:
                    should_change = False
                    break
                if neighbor_gain == self.gain and neighbor < self.agent_id:
                    should_change = False
                    break
            if should_change:
                self.set_value(self.best_value)
            self.broadcast_assignment()

    def _step_mgm2(self) -> None:
        phase = self.logical_round % 5

        if phase == 0:
            self.received_proposals.clear()
            self.received_accepts.clear()
            self.neighbor_group_messages.clear()
            self.group_id = (self.agent_id,)
            self.group_gain = 0.0
            self.group_candidate_values = {self.agent_id: self.value}
            self.proposed_to = None

            if self.neighbors and self.rng.random() < MGM2_OFFER_PROBABILITY:
                self.proposed_to = self.rng.choice(self.neighbors)
                for neighbor in self.neighbors:
                    if neighbor == self.proposed_to:
                        self.send(
                            neighbor,
                            "PROPOSE",
                            {
                                "value": self.value,
                                "domain": list(self.domain),
                                "known_assignments": dict(self.known_assignments),
                                "constraints": [tuple(sorted((self.agent_id, n))) for n in self.neighbors],
                            },
                        )
                    else:
                        self.send(neighbor, "NO_PROPOSE", {})
            else:
                for neighbor in self.neighbors:
                    self.send(neighbor, "NO_PROPOSE", {})

        elif phase == 1:
            if self.proposed_to is not None:
                # A proposer does not accept other proposals in this cycle.
                for proposal in self.received_proposals:
                    self.send(proposal.sender, "REJECT", {})
                # Keep the asynchronous queues alive even when no proposal is accepted.
                self.broadcast_assignment()
                return

            if not self.received_proposals:
                # Keep the asynchronous queues alive; an async agent only steps after messages.
                self.broadcast_assignment()
                return

            # Pick the proposal with the largest locally computed pair gain.
            best_message: Optional[Message] = None
            best_score: Tuple[float, int] = (-float("inf"), 10**9)
            snapshot = self.known_assignments.copy()
            snapshot[self.agent_id] = self.value
            for proposal in self.received_proposals:
                proposer = proposal.sender
                if proposer not in snapshot:
                    snapshot[proposer] = int(proposal.payload.get("value", self.problem.initial_assignment[proposer]))
                proposer_value, receiver_value, pair_gain = self.problem.best_pair_values(
                    proposer, self.agent_id, snapshot
                )
                score = (float(pair_gain), -proposer)
                if score > best_score:
                    best_score = score
                    best_message = proposal

            if best_message is None:
                return

            proposer = best_message.sender
            first_value, second_value, pair_gain = self.problem.best_pair_values(
                proposer, self.agent_id, snapshot
            )
            group_id = tuple(sorted((proposer, self.agent_id)))
            candidate_values = {proposer: first_value, self.agent_id: second_value}
            self.group_id = group_id
            self.group_gain = float(pair_gain)
            self.group_candidate_values = candidate_values

            self.send(
                proposer,
                "ACCEPT",
                {
                    "group_id": group_id,
                    "gain": float(pair_gain),
                    "candidate_values": candidate_values,
                    "receiver": self.agent_id,
                },
            )
            for proposal in self.received_proposals:
                if proposal.sender != proposer:
                    self.send(proposal.sender, "REJECT", {})

        elif phase == 2:
            accepted = False
            for accept in self.received_accepts:
                payload = accept.payload
                group_id = tuple(payload["group_id"])
                if self.proposed_to in group_id and self.agent_id in group_id:
                    self.group_id = group_id
                    self.group_gain = float(payload["gain"])
                    self.group_candidate_values = {
                        int(k): int(v) for k, v in payload["candidate_values"].items()
                    }
                    accepted = True
                    break

            if not accepted and self.group_id == (self.agent_id,):
                best_value, gain = self.compute_best_single_move()
                self.group_gain = float(gain)
                self.group_candidate_values = {self.agent_id: best_value}

            for neighbor in self.neighbors:
                self.send(
                    neighbor,
                    "MGM2_GAIN",
                    {
                        "gain": self.group_gain,
                        "group_id": self.group_id,
                        "members": list(self.group_id),
                        "leader": min(self.group_id),
                    },
                )

        elif phase == 3:
            if self.group_gain <= 0:
                self.broadcast_assignment()
                return

            leader = min(self.group_id)
            wins = True
            for neighbor in self.neighbors:
                payload = self.neighbor_group_messages.get(neighbor)
                if not payload:
                    continue
                neighbor_gain = float(payload.get("gain", 0.0))
                neighbor_leader = int(payload.get("leader", neighbor))
                if neighbor_gain > self.group_gain:
                    wins = False
                    break
                if neighbor_gain == self.group_gain and neighbor_leader < leader:
                    wins = False
                    break

            if wins and self.agent_id in self.group_candidate_values:
                self.set_value(self.group_candidate_values[self.agent_id])
            self.broadcast_assignment()

        else:
            self.broadcast_assignment()

    def _step_dms(self) -> None:
        lam = self.dms_lambda

        # Variable -> function Q messages.
        for edge in self.incident_edges:
            calculated = np.zeros(self.problem.domain_size, dtype=float)
            for other_edge, r_vector in self.r_incoming.items():
                if other_edge != edge:
                    calculated += r_vector
            calculated = self.normalize(calculated)
            old = self.q_old[edge]
            q_vector = (1.0 - lam) * old + lam * calculated
            self.q_old[edge] = q_vector

            owner = min(edge)
            if owner == self.agent_id:
                self.owned_q_messages[edge][self.agent_id] = q_vector.copy()
            else:
                self.send(owner, "DMS_Q", {"edge": edge, "vector": q_vector.tolist()})

        # Owned function nodes compute and send R messages.
        for edge in self.owned_edges:
            i, j = edge
            matrix = self.problem.constraints[edge].cost_matrix.astype(float)
            q_i = self.owned_q_messages[edge][i]
            q_j = self.owned_q_messages[edge][j]

            calculated_to_i = np.min(matrix + q_j.reshape(1, -1), axis=1)
            calculated_to_j = np.min(matrix + q_i.reshape(-1, 1), axis=0)
            calculated_to_i = self.normalize(calculated_to_i)
            calculated_to_j = self.normalize(calculated_to_j)

            for endpoint, calculated in ((i, calculated_to_i), (j, calculated_to_j)):
                old = self.owned_r_old[(edge, endpoint)]
                r_vector = (1.0 - lam) * old + lam * calculated
                self.owned_r_old[(edge, endpoint)] = r_vector
                if endpoint == self.agent_id:
                    self.r_incoming[edge] = r_vector.copy()
                else:
                    self.send(endpoint, "DMS_R", {"edge": edge, "vector": r_vector.tolist()})

        # Belief and assignment update. Min-Sum chooses the minimum cost belief.
        belief = np.zeros(self.problem.domain_size, dtype=float)
        for r_vector in self.r_incoming.values():
            belief += r_vector
        new_value = int(np.argmin(belief))
        self.set_value(new_value)
        self.broadcast_assignment()


class AsynchronousSimulator:
    """
    Asynchronous simulator.

    Each agent is a Thread. Each agent has a Queue. An agent performs a local step
    only after receiving at least one message and uses the latest message/state it
    has from each neighbor. The global plot clock is Lamport-style: samples are
    taken when the minimum logical clock across all agents reaches the sample.
    The run stops when that minimum logical clock reaches max_steps.
    """

    def __init__(self, problem: DCOPProblem, max_steps: int, rng_seed: int):
        self.problem = problem.clone()
        self.max_steps = max_steps
        self.rng_seed = rng_seed

    def run(self, algorithm_name: str) -> SimulationResult:
        started = time.time()
        world = AsyncWorld(self.problem, self.max_steps)
        initial_cost = self.problem.total_cost(world.snapshot_assignment())
        agents = [
            AsyncVariableAgent(
                agent_id=i,
                algorithm_name=algorithm_name,
                world=world,
                rng_seed=self.rng_seed + 7919 * i,
            )
            for i in range(self.problem.num_agents)
        ]

        for agent in agents:
            agent.start()

        costs_by_clock: List[int] = []
        next_clock_to_sample = 1

        # Monitor min logical clock. This is the requested clock synchronization for plotting.
        while next_clock_to_sample <= self.max_steps:
            current_min_clock = world.min_clock()
            while next_clock_to_sample <= min(current_min_clock, self.max_steps):
                costs_by_clock.append(self.problem.total_cost(world.snapshot_assignment()))
                next_clock_to_sample += 1
            if world.stop_event.is_set():
                break
            time.sleep(0.001)

        world.stop_event.set()
        for agent in agents:
            agent.join(timeout=1.0)

        while len(costs_by_clock) < self.max_steps:
            costs_by_clock.append(self.problem.total_cost(world.snapshot_assignment()))

        runtime = time.time() - started
        total_messages = sum(agent.messages_sent for agent in agents)
        total_changes = sum(agent.assignment_changes for agent in agents)
        final_cost = costs_by_clock[-1] if costs_by_clock else initial_cost

        return SimulationResult(
            algorithm=algorithm_name,
            simulator="Asynchronous",
            costs=costs_by_clock,
            initial_cost=initial_cost,
            final_cost=final_cost,
            messages_sent=total_messages,
            assignment_changes=total_changes,
            runtime_seconds=runtime,
            converged_at=None,  # Async runs stop by min Lamport clock as required.
        )


# ================================================================
# Experiment and plotting utilities
# ================================================================
@dataclass
class AggregateStats:
    algorithm: str
    simulator: str
    mean_initial_cost: float
    mean_final_cost: float
    mean_messages: float
    mean_assignment_changes: float
    total_runtime_seconds: float
    converged_count: int
    mean_converged_at: Optional[float]


def make_problem_suite(
    num_problems: int,
    num_agents: int,
    domain_size: int,
    constraint_probability: float,
    max_constraint_cost: int,
    base_seed: int,
) -> List[DCOPProblem]:
    return [
        DCOPProblem.generate_random(
            num_agents=num_agents,
            constraint_probability=constraint_probability,
            domain_size=domain_size,
            max_cost=max_constraint_cost,
            seed=base_seed + problem_index,
        )
        for problem_index in range(num_problems)
    ]


def average_cost_curves(results: List[SimulationResult], max_iterations: int) -> Dict[str, np.ndarray]:
    curves: Dict[str, List[List[int]]] = {name: [] for name in ALGORITHM_NAMES}
    for result in results:
        padded_costs = list(result.costs)
        if len(padded_costs) < max_iterations:
            padded_costs.extend([padded_costs[-1]] * (max_iterations - len(padded_costs)))
        curves[result.algorithm].append(padded_costs[:max_iterations])

    averaged: Dict[str, np.ndarray] = {}
    for name, curve_list in curves.items():
        if not curve_list:
            averaged[name] = np.zeros(max_iterations, dtype=float)
        else:
            averaged[name] = np.mean(np.array(curve_list, dtype=float), axis=0)
    return averaged


def aggregate_stats(results: List[SimulationResult]) -> List[AggregateStats]:
    stats: List[AggregateStats] = []
    grouped: Dict[Tuple[str, str], List[SimulationResult]] = {}
    for result in results:
        grouped.setdefault((result.simulator, result.algorithm), []).append(result)

    for (simulator, algorithm), group in sorted(grouped.items()):
        converged_values = [r.converged_at for r in group if r.converged_at is not None]
        stats.append(
            AggregateStats(
                algorithm=algorithm,
                simulator=simulator,
                mean_initial_cost=float(np.mean([r.initial_cost for r in group])),
                mean_final_cost=float(np.mean([r.final_cost for r in group])),
                mean_messages=float(np.mean([r.messages_sent for r in group])),
                mean_assignment_changes=float(np.mean([r.assignment_changes for r in group])),
                total_runtime_seconds=float(np.sum([r.runtime_seconds for r in group])),
                converged_count=len(converged_values),
                mean_converged_at=float(np.mean(converged_values)) if converged_values else None,
            )
        )
    return stats


def plot_average_curves(
    curves: Dict[str, np.ndarray],
    title: str,
    output_path: str,
    sample_every: int,
) -> None:
    plt.figure(figsize=(11, 7))
    max_iterations = len(next(iter(curves.values()))) if curves else 0
    x_values = np.arange(1, max_iterations + 1)
    sampled = np.arange(0, max_iterations, sample_every)

    for algorithm_name in ALGORITHM_NAMES:
        y_values = curves[algorithm_name]
        plt.plot(x_values[sampled], y_values[sampled], marker="o", markersize=3, label=algorithm_name)

    plt.xlabel("Iteration / synchronized Lamport min-clock step")
    plt.ylabel("Average solution cost over problems")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def progress_bar(percent: float, width: int = 24) -> str:
    """Return a simple ASCII progress bar for terminal output.

    This avoids non-standard dependencies such as tqdm and keeps every printed
    experiment row self-contained: the row shows both a bar and the exact
    completion percentage.
    """
    clamped = max(0.0, min(100.0, percent))
    filled = int(round((clamped / 100.0) * width))
    return "#" * filled + "-" * (width - filled)


def print_summary(stats: List[AggregateStats]) -> None:
    print("\n==================== EXPERIMENT SUMMARY ====================")
    header = (
        f"{'Simulator':<14} {'Algorithm':<8} {'Initial':>12} {'Final':>12} "
        f"{'Messages':>12} {'Changes':>12} {'Runtime(s)':>12} {'Converged':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in stats:
        convergence_text = (
            f"{row.converged_count} @ {row.mean_converged_at:.1f}"
            if row.mean_converged_at is not None
            else str(row.converged_count)
        )
        print(
            f"{row.simulator:<14} {row.algorithm:<8} {row.mean_initial_cost:>12.2f} "
            f"{row.mean_final_cost:>12.2f} {row.mean_messages:>12.2f} "
            f"{row.mean_assignment_changes:>12.2f} {row.total_runtime_seconds:>12.2f} "
            f"{convergence_text:>12}"
        )
    print("============================================================\n")


def run_experiment(
    num_problems: int = DEFAULT_NUM_PROBLEMS,
    num_agents: int = DEFAULT_NUM_AGENTS,
    domain_size: int = DEFAULT_DOMAIN_SIZE,
    constraint_probability: float = DEFAULT_CONSTRAINT_PROBABILITY,
    max_constraint_cost: int = DEFAULT_MAX_COST,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    sample_every: int = DEFAULT_SAMPLE_EVERY,
    base_seed: int = 2026,
    run_async: bool = True,
    run_sync: bool = True,
) -> Tuple[List[SimulationResult], Dict[str, Dict[str, np.ndarray]]]:
    """
    Run all algorithms on the same generated problem suite.

    The same DCOPProblem objects, including the same initial assignments, are used
    for every algorithm and both simulators. Each simulator clones a problem before
    running to avoid accidental cross-run mutation.
    """
    problems = make_problem_suite(
        num_problems=num_problems,
        num_agents=num_agents,
        domain_size=domain_size,
        constraint_probability=constraint_probability,
        max_constraint_cost=max_constraint_cost,
        base_seed=base_seed,
    )

    all_results: List[SimulationResult] = []

    active_simulators = []
    if run_sync:
        active_simulators.append("Synchronous")
    if run_async:
        active_simulators.append("Asynchronous")
    total_runs = len(active_simulators) * len(ALGORITHM_NAMES) * num_problems
    completed_runs = 0

    for simulator_name in ("Synchronous", "Asynchronous"):
        if simulator_name == "Synchronous" and not run_sync:
            continue
        if simulator_name == "Asynchronous" and not run_async:
            continue

        print(f"\n--- Running {simulator_name} simulator ---")
        for algorithm_name in ALGORITHM_NAMES:
            print(f"Algorithm: {algorithm_name}")
            for problem_index, problem in enumerate(problems):
                # Deterministic seed per simulator/algorithm/problem.
                sim_offset = 100000 if simulator_name == "Synchronous" else 200000
                alg_offset = 10000 * ALGORITHM_NAMES.index(algorithm_name)
                run_seed = base_seed + sim_offset + alg_offset + problem_index

                if simulator_name == "Synchronous":
                    simulator = SynchronousSimulator(problem, max_iterations, run_seed)
                else:
                    simulator = AsynchronousSimulator(problem, max_iterations, run_seed)

                result = simulator.run(algorithm_name)
                all_results.append(result)
                completed_runs += 1

                algorithm_percent = 100.0 * (problem_index + 1) / max(1, num_problems)
                total_percent = 100.0 * completed_runs / max(1, total_runs)
                print(
                    f"  [{progress_bar(total_percent)}] total={total_percent:6.2f}% | "
                    f"algorithm={algorithm_percent:6.2f}% | "
                    f"problem {problem_index + 1:>2}/{num_problems}: "
                    f"initial={result.initial_cost}, final={result.final_cost}, "
                    f"messages={result.messages_sent}, changes={result.assignment_changes}, "
                    f"time={result.runtime_seconds:.2f}s"
                )

    curves_by_simulator: Dict[str, Dict[str, np.ndarray]] = {}
    for simulator_name in ("Synchronous", "Asynchronous"):
        simulator_results = [r for r in all_results if r.simulator == simulator_name]
        if simulator_results:
            curves_by_simulator[simulator_name] = average_cost_curves(simulator_results, max_iterations)

    if "Synchronous" in curves_by_simulator:
        plot_average_curves(
            curves_by_simulator["Synchronous"],
            "Synchronous DCOP algorithms - average solution cost",
            "sync_results.png",
            sample_every,
        )
        print("Saved sync_results.png")

    if "Asynchronous" in curves_by_simulator:
        plot_average_curves(
            curves_by_simulator["Asynchronous"],
            "Asynchronous DCOP algorithms - average solution cost",
            "async_results.png",
            sample_every,
        )
        print("Saved async_results.png")

    print_summary(aggregate_stats(all_results))
    return all_results, curves_by_simulator


# ================================================================
# Validation helpers
# ================================================================
def smoke_test() -> None:
    """Small correctness test requested in the prompt: 5 agents and 10 iterations."""
    print("\nRunning smoke test: 5 agents, 10 iterations, all algorithms, both simulators")
    results, curves = run_experiment(
        num_problems=2,
        num_agents=5,
        domain_size=3,
        constraint_probability=0.6,
        max_constraint_cost=20,
        max_iterations=10,
        sample_every=1,
        base_seed=12345,
        run_async=True,
        run_sync=True,
    )

    expected_results = 2 * 2 * len(ALGORITHM_NAMES)
    assert len(results) == expected_results, f"Expected {expected_results} results, got {len(results)}"
    for result in results:
        assert len(result.costs) == 10, f"{result.algorithm}/{result.simulator} did not return 10 costs"
        assert all(isinstance(cost, (int, np.integer)) for cost in result.costs), "Costs must be integers"

    assert "Synchronous" in curves, "Missing synchronous curves"
    assert "Asynchronous" in curves, "Missing asynchronous curves"
    print("Smoke test passed. Graph files were generated: sync_results.png, async_results.png")


# ================================================================
# Main entry point
# ================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Assignment 2 DCOP simulator")
    parser.add_argument("--quick-test", action="store_true", help="Run a small smoke test instead of the full experiment")
    parser.add_argument("--problems", type=int, default=DEFAULT_NUM_PROBLEMS, help="Number of random problems")
    parser.add_argument("--iterations", type=int, default=DEFAULT_MAX_ITERATIONS, help="Iterations / min-clock steps")
    parser.add_argument("--seed", type=int, default=2026, help="Base seed for reproducibility")
    parser.add_argument("--allow-sleep", action="store_true", help="Do not prevent the computer from sleeping during the run")
    parser.add_argument("--keep-display-on", action="store_true", help="Also try to keep the display awake, not only the system")
    args = parser.parse_args()

    with SleepPreventer(enabled=not args.allow_sleep, keep_display_on=args.keep_display_on):
        if args.quick_test:
            smoke_test()
            return

        run_experiment(
            num_problems=args.problems,
            num_agents=DEFAULT_NUM_AGENTS,
            domain_size=DEFAULT_DOMAIN_SIZE,
            constraint_probability=DEFAULT_CONSTRAINT_PROBABILITY,
            max_constraint_cost=DEFAULT_MAX_COST,
            max_iterations=args.iterations,
            sample_every=DEFAULT_SAMPLE_EVERY,
            base_seed=args.seed,
            run_async=True,
            run_sync=True,
        )


if __name__ == "__main__":
    main()
