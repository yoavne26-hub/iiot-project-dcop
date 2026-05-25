"""Random DCOP problem generation."""

from __future__ import annotations

import random

from backend.dcop.problem import BinaryConstraint, DCOPProblem


def generate_random_dcop(
    num_agents: int,
    constraint_probability: float,
    domain_size: int,
    max_cost: int,
    seed: int | None = None,
) -> DCOPProblem:
    """Generate a random binary DCOP with one variable per agent."""

    if num_agents <= 0:
        raise ValueError("num_agents must be greater than 0.")
    if not 0 <= constraint_probability <= 1:
        raise ValueError("constraint_probability must be between 0 and 1.")
    if domain_size <= 0:
        raise ValueError("domain_size must be greater than 0.")
    if max_cost < 1:
        raise ValueError("max_cost must be at least 1.")

    rng = random.Random(seed)
    constraints: dict[tuple[int, int], BinaryConstraint] = {}

    for agent_a in range(num_agents):
        for agent_b in range(agent_a + 1, num_agents):
            if rng.random() >= constraint_probability:
                continue

            costs = tuple(
                tuple(rng.randint(1, max_cost) for _ in range(domain_size))
                for _ in range(domain_size)
            )
            constraint = BinaryConstraint(
                agent_a=agent_a,
                agent_b=agent_b,
                costs=costs,
            )
            constraints[constraint.key()] = constraint

    initial_assignment = {
        agent_id: rng.randrange(domain_size) for agent_id in range(num_agents)
    }

    return DCOPProblem(
        num_agents=num_agents,
        domain_size=domain_size,
        constraints=constraints,
        initial_assignment=initial_assignment,
        seed=seed,
        constraint_probability=constraint_probability,
        max_cost=max_cost,
    )
