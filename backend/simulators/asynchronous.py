"""Asynchronous simulator for DCOP algorithms."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import queue
import random
import threading
import time

from backend.algorithms.base import (
    AlgorithmMessage,
    AlgorithmStepResult,
    DistributedAlgorithm,
)
from backend.dcop.cost import calculate_global_cost
from backend.dcop.problem import DCOPProblem
from backend.simulators.synchronous import SimulationRunResult


@dataclass(frozen=True)
class _AsyncMessageEvent:
    """One asynchronous algorithm message delivered to a receiver inbox."""

    message: AlgorithmMessage


@dataclass(frozen=True)
class _AsyncStopEvent:
    """Worker shutdown sentinel."""


_AsyncEvent = _AsyncMessageEvent | _AsyncStopEvent
_HEARTBEAT_KIND = "__heartbeat__"


class _AsyncAgentWorker(threading.Thread):
    """Worker thread that owns one agent inbox."""

    def __init__(self, agent_id: int, simulator: "AsynchronousSimulator") -> None:
        super().__init__(name=f"DCOPAsyncAgent-{agent_id}", daemon=True)
        self.agent_id = agent_id
        self.simulator = simulator
        self.inbox: queue.Queue[_AsyncEvent] = queue.Queue()

    def run(self) -> None:
        """Process queued activations and messages until stopped."""

        while True:
            event = self.inbox.get()
            try:
                if isinstance(event, _AsyncStopEvent) or self.simulator._stop_event.is_set():
                    return
                self.simulator._process_worker_event(self.agent_id, event)
            except Exception as error:
                self.simulator._record_worker_error(self.agent_id, event, error)
                return


class AsynchronousSimulator:
    """Run a distributed algorithm through independent worker inboxes."""

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
        self._message_backlog_limit = max(1, 2 * problem.num_agents)

        self._scheduler_rng = random.Random(seed)
        self._algorithm_lock = threading.Lock()
        self._state_condition = threading.Condition(threading.RLock())
        self._stop_event = threading.Event()
        self._worker_error: tuple[int, _AsyncEvent, Exception] | None = None
        self._scheduler_thread: threading.Thread | None = None

        self._workers = {
            agent_id: _AsyncAgentWorker(agent_id=agent_id, simulator=self)
            for agent_id in range(problem.num_agents)
        }

        self._lamport_clocks = {
            agent_id: 0 for agent_id in range(problem.num_agents)
        }
        self._sender_sequences = {
            agent_id: 0 for agent_id in range(problem.num_agents)
        }
        self._latest_sequence_by_receiver = {
            agent_id: {} for agent_id in range(problem.num_agents)
        }
        self._local_clocks = {
            agent_id: 0 for agent_id in range(problem.num_agents)
        }

        self._processed_async_events = 0
        self._pending_messages = 0
        self._heartbeat_messages_injected = 0
        self._heartbeat_messages_delivered = 0
        self._messages_delivered = 0
        self._stale_messages_ignored = 0

    def run(self) -> SimulationRunResult:
        """Run asynchronous measurement checkpoints and collect global costs."""

        started_at = time.perf_counter()
        cost_history: list[int] = []
        self._start_workers()

        try:
            with self._algorithm_lock:
                self.algorithm.initialize(
                    self.problem,
                    self.problem.initial_assignment,
                    seed=self.seed,
                )
                initial_messages = self.algorithm.initial_async_messages()

            self._enqueue_messages(initial_messages)
            self._start_scheduler()

            # Async iterations use the course clarification's global progress
            # definition: each agent owns a local logical clock, and the global
            # async step is the minimum local clock across all agents.
            while len(cost_history) < self.iterations:
                with self._state_condition:
                    self._state_condition.wait_for(
                        lambda: (
                            self._global_async_step() > len(cost_history)
                            or self._worker_error is not None
                        )
                    )
                    self._raise_worker_error_if_needed()

                    while (
                        self._global_async_step() > len(cost_history)
                        and len(cost_history) < self.iterations
                    ):
                        with self._algorithm_lock:
                            assignment = self.algorithm.get_assignment()
                        cost_history.append(
                            calculate_global_cost(self.problem, assignment)
                        )
                        if self.progress_callback is not None:
                            self.progress_callback(len(cost_history))

            with self._algorithm_lock:
                final_assignment = self.algorithm.get_assignment()

            runtime_seconds = time.perf_counter() - started_at
            return SimulationRunResult(
                simulator="async",
                algorithm=self.algorithm.name,
                iterations=self.iterations,
                cost_history=cost_history,
                final_assignment=final_assignment,
                total_messages=self._messages_delivered,
                runtime_seconds=runtime_seconds,
                metadata=self._metadata(),
            )
        finally:
            self._stop_event.set()
            with self._state_condition:
                self._state_condition.notify_all()
            self._stop_scheduler()
            self._stop_workers()

    def _start_workers(self) -> None:
        """Start all agent workers."""

        for worker in self._workers.values():
            worker.start()

    def _stop_workers(self) -> None:
        """Stop all agent workers cleanly."""

        for worker in self._workers.values():
            worker.inbox.put(_AsyncStopEvent())
        for worker in self._workers.values():
            worker.join()

    def _start_scheduler(self) -> None:
        """Start the idle heartbeat message scheduler."""

        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            name="DCOPAsyncHeartbeatScheduler",
            daemon=True,
        )
        self._scheduler_thread.start()

    def _stop_scheduler(self) -> None:
        """Stop the idle heartbeat scheduler."""

        if self._scheduler_thread is not None:
            self._scheduler_thread.join()

    def _scheduler_loop(self) -> None:
        """Inject heartbeat messages only when no algorithm messages are pending."""

        while not self._stop_event.is_set():
            with self._state_condition:
                self._state_condition.wait_for(
                    lambda: (
                        self._stop_event.is_set()
                        or self._worker_error is not None
                        or (
                            self._pending_messages == 0
                            and self._global_async_step() < self.iterations
                        )
                    )
                )
                if self._stop_event.is_set() or self._worker_error is not None:
                    return

                candidates = [
                    agent_id
                    for agent_id, clock in self._local_clocks.items()
                    if clock < self.iterations and self.problem.neighbors[agent_id]
                ]
                if not candidates:
                    self._worker_error = (
                        -1,
                        _AsyncStopEvent(),
                        RuntimeError(
                            "Asynchronous simulation cannot progress because no "
                            "agent below the iteration target has a neighbor."
                        ),
                    )
                    self._stop_event.set()
                    self._state_condition.notify_all()
                    return

                min_clock = min(self._local_clocks[agent_id] for agent_id in candidates)
                lagging_agents = [
                    agent_id
                    for agent_id in candidates
                    if self._local_clocks[agent_id] == min_clock
                ]
                receiver = self._scheduler_rng.choice(lagging_agents)
                sender = self._scheduler_rng.choice(self.problem.neighbors[receiver])

            self._enqueue_heartbeat_message(sender=sender, receiver=receiver)

    def _process_worker_event(self, agent_id: int, event: _AsyncEvent) -> None:
        """Process one event from one worker inbox."""

        if isinstance(event, _AsyncMessageEvent):
            message_result = self._handle_message(event.message)
            messages: list[AlgorithmMessage] = []
            if message_result is not None:
                messages.extend(self._extract_messages(message_result))
                activation_result = self._handle_activation(event.message.receiver)
                messages.extend(self._extract_messages(activation_result))
            self._enqueue_messages(messages)
            self._mark_event_processed()
            return

        raise ValueError(f"Unsupported async event for agent {agent_id}: {event!r}.")

    def _handle_activation(self, agent_id: int) -> AlgorithmStepResult:
        """Run one algorithm activation for one agent."""

        self._advance_internal_clock(agent_id)
        with self._algorithm_lock:
            return self.algorithm.on_async_activation(agent_id)

    def _handle_message(
        self,
        message: AlgorithmMessage,
    ) -> AlgorithmStepResult | None:
        """Deliver one message unless it is stale."""

        self._update_receiver_clock(message)
        if message.kind == _HEARTBEAT_KIND:
            with self._state_condition:
                self._messages_delivered += 1
                self._heartbeat_messages_delivered += 1
            return AlgorithmStepResult(changed_agents=set(), messages_sent=0)

        if self._is_stale_message(message):
            return None

        with self._state_condition:
            self._messages_delivered += 1

        with self._algorithm_lock:
            return self.algorithm.handle_async_message(message)

    def _enqueue_messages(self, messages: list[AlgorithmMessage]) -> None:
        """Stamp and enqueue outgoing algorithm messages."""

        if self._stop_event.is_set():
            return

        for message in messages:
            stamped = self._stamp_message(message)
            if stamped.receiver not in self._workers:
                raise ValueError(f"Invalid async message receiver: {stamped.receiver}.")
            self._enqueue_message_event(stamped)

    def _enqueue_heartbeat_message(self, sender: int, receiver: int) -> None:
        """Enqueue a simulator heartbeat message that can trigger one activation."""

        message = AlgorithmMessage(
            sender=sender,
            receiver=receiver,
            kind=_HEARTBEAT_KIND,
            payload={},
        )
        stamped = self._stamp_message(message)
        with self._state_condition:
            self._heartbeat_messages_injected += 1
        self._enqueue_message_event(stamped)

    def _enqueue_message_event(self, message: AlgorithmMessage) -> None:
        """Place one message event in a receiver inbox and count it as pending."""

        with self._state_condition:
            self._pending_messages += 1
        self._workers[message.receiver].inbox.put(_AsyncMessageEvent(message))

    def _stamp_message(self, message: AlgorithmMessage) -> AlgorithmMessage:
        """Attach simulator-owned Lamport and sender-sequence metadata."""

        if message.sender not in self._workers:
            raise ValueError(f"Invalid async message sender: {message.sender}.")

        with self._state_condition:
            self._lamport_clocks[message.sender] += 1
            self._sender_sequences[message.sender] += 1
            return AlgorithmMessage(
                sender=message.sender,
                receiver=message.receiver,
                kind=message.kind,
                payload=dict(message.payload),
                lamport_time=self._lamport_clocks[message.sender],
                sender_sequence=self._sender_sequences[message.sender],
            )

    def _advance_internal_clock(self, agent_id: int) -> None:
        """Advance an agent Lamport clock for an internal activation event."""

        with self._state_condition:
            self._lamport_clocks[agent_id] += 1
            self._local_clocks[agent_id] += 1

    def _update_receiver_clock(self, message: AlgorithmMessage) -> None:
        """Apply Lamport receive-clock update for one message."""

        with self._state_condition:
            current = self._lamport_clocks[message.receiver]
            self._lamport_clocks[message.receiver] = (
                max(current, message.lamport_time) + 1
            )

    def _is_stale_message(self, message: AlgorithmMessage) -> bool:
        """Return True and count a message if its sender sequence is stale."""

        with self._state_condition:
            latest_by_sender = self._latest_sequence_by_receiver[message.receiver]
            latest = latest_by_sender.get(message.sender, 0)
            if message.sender_sequence <= latest:
                self._stale_messages_ignored += 1
                return True

            latest_by_sender[message.sender] = message.sender_sequence
            return False

    def _mark_event_processed(self) -> None:
        """Record one completed async event and notify waiters."""

        with self._state_condition:
            self._processed_async_events += 1
            self._pending_messages = max(0, self._pending_messages - 1)
            self._state_condition.notify_all()

    def _record_worker_error(
        self,
        agent_id: int,
        event: _AsyncEvent,
        error: Exception,
    ) -> None:
        """Store the first worker exception and wake controller threads."""

        with self._state_condition:
            if self._worker_error is None:
                self._worker_error = (agent_id, event, error)
            self._stop_event.set()
            self._state_condition.notify_all()

    def _raise_worker_error_if_needed(self) -> None:
        """Raise a clear error if any worker failed."""

        if self._worker_error is None:
            return

        agent_id, event, error = self._worker_error
        raise RuntimeError(
            f"Asynchronous worker {agent_id} failed while processing "
            f"{self._describe_event(event)}."
        ) from error

    def _metadata(self) -> dict[str, object]:
        """Return async execution metadata for diagnostics and reports."""

        with self._state_condition:
            lamport_max = max(self._lamport_clocks.values(), default=0)
            return {
                "seed": self.seed,
                "processed_async_events": self._processed_async_events,
                "messages_delivered": self._messages_delivered,
                "messages": self._messages_delivered,
                "heartbeat_messages_injected": self._heartbeat_messages_injected,
                "heartbeat_messages_delivered": self._heartbeat_messages_delivered,
                "lamport_max": lamport_max,
                "global_async_step": self._global_async_step(),
                "min_local_clock": self._global_async_step(),
                "local_clocks": dict(self._local_clocks),
                "async_iteration_definition": (
                    "global_async_step = min(agent.local_clock for all agents); "
                    "run stops when global_async_step reaches the configured iteration limit"
                ),
                "stale_messages_ignored": self._stale_messages_ignored,
                "message_backlog_limit": self._message_backlog_limit,
            }

    def _global_async_step(self) -> int:
        """Return the Lamport-style global async step used for measurement."""

        return min(self._local_clocks.values(), default=0)

    @staticmethod
    def _describe_event(event: _AsyncEvent) -> str:
        """Return a concise event description for worker failure messages."""

        if isinstance(event, _AsyncMessageEvent):
            message = event.message
            return (
                f"message event kind={message.kind!r} "
                f"sender={message.sender} receiver={message.receiver} "
                f"sequence={message.sender_sequence}"
            )
        return type(event).__name__

    @staticmethod
    def _extract_messages(step_result: AlgorithmStepResult) -> list[AlgorithmMessage]:
        """Extract generated messages from algorithm result metadata."""

        raw_messages = step_result.metadata.get("messages", [])
        if not isinstance(raw_messages, list):
            raise ValueError("Algorithm result metadata field 'messages' must be a list.")
        if any(not isinstance(message, AlgorithmMessage) for message in raw_messages):
            raise ValueError("Algorithm result metadata contains a non-message item.")

        return list(raw_messages)
