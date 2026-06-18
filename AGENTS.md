# AGENTS.md

## Project Context

This repository is for Assignment 2 in an IIOT course. It was created from the Assignment 1 project, which already had synchronous and asynchronous distributed-agent simulators.

Assignment 2 changes the main goal: instead of the Assignment 1 routing-table update algorithm, this project must solve Distributed Constraint Optimization Problems (DCOPs) using incomplete distributed algorithms.

The final system must support:

* A random DCOP problem generator.
* Synchronous simulation.
* Asynchronous simulation, where agents operate through message passing and threaded/asynchronous behavior.
* Four algorithms:

  * DSA-C
  * MGM
  * MGM-2
  * Max-sum + Damping, also called DMS.
* A CLI experiment runner.
* CSV result files.
* Two graph images for the report.

## Assignment Requirements

The experiment configuration for the final report must support:

* 50 generated DCOP problems.
* The same 50 problems must be used for every algorithm and both simulators.
* 50 agents.
* One variable per agent.
* Domain size 10.
* Binary constraints.
* Constraint costs between 1 and 100.
* 1000 iterations.
* DSA-C probability: 0.70.
* DMS damping factor: 0.9.
* Constraint probability must be a user input.
* Seeds must control both problem generation and initial assignments.

The global solution cost is the sum of all active binary constraint costs, counted once per constraint edge. Lower cost is better.

## Main Development Priority

Build the project in this order:

1. DCOP data model.
2. Random DCOP generator.
3. Cost evaluator.
4. Base algorithm interface.
5. Synchronous simulator.
6. DSA-C implementation.
7. CLI experiment runner for one algorithm on the synchronous simulator.
8. Asynchronous simulator.
9. DSA-C on both simulators.
10. MGM.
11. MGM-2.
12. DMS.
13. CSV exports.
14. Plot generation.
15. Documentation cleanup.

## Expected Commands

The project should support:

```bash
python main.py
```

for CLI experiments.

## Architecture Guidelines

Keep the design modular.

The DCOP problem model should not depend on any specific algorithm.

The algorithms should not directly depend on plotting.

The simulators should control timing and message delivery.

The algorithms should control local decision logic.

The experiment runner should control:

* generated problem seeds,
* selected algorithms,
* selected simulators,
* number of iterations,
* output paths,
* aggregation of average costs.

## Recommended Folder Structure

Use this target structure unless there is a strong reason to change it:

```text
backend/
  config.py
  dcop/
    __init__.py
    problem.py
    generator.py
    cost.py
  algorithms/
    __init__.py
    base.py
    dsa_c.py
    mgm.py
    mgm2.py
    dms.py
  simulators/
    __init__.py
    synchronous.py
    asynchronous.py
  experiments/
    __init__.py
    runner.py
    results.py
    plotting.py

run_experiments.py
main.py
requirements.txt
README.md
AGENTS.md
```

## Coding Style

Use Python 3.10+.

Prefer clear dataclasses for problem definitions and results.

Use type hints.

Use readable names.

Keep functions small.

Avoid hidden global mutable state.

Use deterministic random generation through explicit `random.Random(seed)` objects.

Avoid importing heavy external libraries unless needed. Matplotlib is acceptable for plots. Standard library should be preferred.

## Reproducibility Rules

All experiments must be reproducible.

The same generated problem and initial assignment must be reused across:

* all algorithms,
* both simulators,
* repeated experiment runs with the same seed.

Do not generate a new initial assignment separately for each algorithm unless explicitly requested. Initial assignments are part of the generated problem instance.

## Result Tracking

For every run, track the global solution cost at every iteration from 1 to the requested iteration limit.

For final report graphs:

* X-axis: iteration number from 1 to 1000.
* Y-axis: average solution cost over the 50 problems.
* One graph for synchronous simulator.
* One graph for asynchronous simulator.
* Each graph should include lines for DSA-C, MGM, MGM-2, and DMS.

## Asynchronous Simulator Notes

Use agents, messages, and asynchronous/threaded behavior.

However, the output must still be compatible with the experiment runner:

* cost per iteration/step,
* total messages,
* final assignment,
* runtime metrics if available.

When possible, keep asynchronous behavior controlled enough for reproducible comparison through seeds.

## Do Not Do

Do not hard-code final experiment parameters in the algorithms.

Do not count constraint costs twice.

Do not use random global state when a seed object can be passed.

Do not implement all algorithms in one large file.

Do not optimize prematurely before correctness is verified.

## First Milestone

The first meaningful milestone is:

A CLI command can generate one DCOP problem, run DSA-C on the synchronous simulator for a small number of iterations, and print/save the cost per iteration.

Only after that milestone should the asynchronous simulator be connected.
