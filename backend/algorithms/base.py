"""Shared interfaces for DCOP algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from backend.dcop.problem import DCOPProblem


@dataclass
class AlgorithmStepResult:
    """Result produced by one algorithm iteration."""

    changed_agents: set[int]
    messages_sent: int = 0
    metadata: dict[str, object] = field(default_factory=dict)


class DistributedAlgorithm(ABC):
    """Base class for distributed DCOP algorithms."""

    name: str

    @abstractmethod
    def initialize(
        self,
        problem: DCOPProblem,
        assignment: dict[int, int],
        seed: int | None = None,
    ) -> None:
        """Initialize algorithm state for a problem and starting assignment."""

    @abstractmethod
    def run_synchronous_iteration(self) -> AlgorithmStepResult:
        """Run one synchronous iteration."""

    @abstractmethod
    def get_assignment(self) -> dict[int, int]:
        """Return the algorithm's current assignment."""
