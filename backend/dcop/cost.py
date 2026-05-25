"""Cost evaluation utilities for DCOP assignments."""

from __future__ import annotations

from backend.dcop.problem import DCOPProblem


def calculate_global_cost(problem: DCOPProblem, assignment: dict[int, int]) -> int:
    """Return the sum of all active binary constraint costs counted once."""

    problem.validate_assignment(assignment)

    total = 0
    for constraint in problem.constraints.values():
        total += constraint.cost(
            assignment[constraint.agent_a],
            assignment[constraint.agent_b],
        )

    return total


def calculate_local_cost(
    problem: DCOPProblem,
    assignment: dict[int, int],
    agent_id: int,
    value: int | None = None,
) -> int:
    """Return the cost contribution of constraints touching one agent."""

    if not 0 <= agent_id < problem.num_agents:
        raise ValueError(f"agent_id must be in 0..{problem.num_agents - 1}.")

    effective_assignment = dict(assignment)
    if value is not None:
        effective_assignment[agent_id] = value
    problem.validate_assignment(effective_assignment)

    total = 0
    for constraint in problem.constraints.values():
        if agent_id not in constraint.key():
            continue
        total += constraint.cost(
            effective_assignment[constraint.agent_a],
            effective_assignment[constraint.agent_b],
        )

    return total
