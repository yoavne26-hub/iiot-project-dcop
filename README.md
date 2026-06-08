# IIOT Assignment 2 DCOP Experiments

This project runs Distributed Constraint Optimization Problem (DCOP) experiments for IIOT Assignment 2. It generates random binary DCOP instances, solves the same generated problems with multiple incomplete distributed algorithms, and writes average solution-cost CSV files and graphs.

Lower global solution cost is better. The global cost is the sum of all active binary constraint costs, counted once per constraint edge.

## What It Runs

The experiment runner supports four algorithms:

- DSA-C
- MGM
- MGM-2
- DMS, implemented as damped min-sum / max-sum style message passing

Each algorithm runs with both simulators:

- synchronous simulator
- asynchronous simulator

The default full experiment uses:

- 50 generated DCOP problems
- 50 agents
- one variable per agent
- domain size 10
- binary constraints
- constraint costs from 1 to 100
- 1000 iterations
- DSA-C probability 0.75
- DMS damping lambda 0.9
- plot and CSV sampling every 5 iterations

The same generated problem and initial assignment are reused for every algorithm and both simulators. Seeds control problem generation, initial assignments, algorithm randomness, and asynchronous scheduling.

## Install

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

External dependencies are matplotlib for PNG graph output and Rich for terminal progress bars.

## Run

Run the default full experiment:

```powershell
python main.py
```

This shows Rich progress bars for overall problem completion and the current algorithm/simulator run.

Equivalent direct command:

```powershell
python run_experiments.py
```

For a faster smoke run:

```powershell
python main.py --mode smoke --agents 10 --iterations 50 --algorithm dsa-c --simulator sync
```

Useful full-run options:

```powershell
python main.py --constraint-probability 0.3
python main.py --problems 50 --agents 50 --iterations 1000
python main.py --algorithms dsa-c,mgm,mgm2,dms --simulators sync,async
python main.py --output-dir results
python main.py --no-progress
python main.py --no-plots
```

Example with progress enabled:

```powershell
python main.py --mode full --problems 50 --agents 50 --iterations 1000
```

Example with progress disabled for CI or logs:

```powershell
python main.py --mode full --problems 50 --agents 50 --iterations 1000 --no-progress
```

## Outputs

Outputs are written under `results/` by default:

- `results/synchronous_average_cost.png`
- `results/asynchronous_average_cost.png`
- `results/synchronous_average_cost.csv`
- `results/asynchronous_average_cost.csv`
- `results/raw_runs.csv`
- `results/summary.csv`
- `results/average_costs.csv`

The synchronous and asynchronous average CSV files are wide tables with one row per recorded iteration and one column per algorithm. The graph files plot the same average solution cost data.

## Synchronous Measurement

The synchronous simulator runs fixed synchronized iterations. In each iteration, the selected algorithm computes from a shared assignment snapshot or synchronized message phase and then applies the iteration result. A global solution cost is recorded after each iteration; the exported average graph data keeps every 5th iteration by default.

## Asynchronous Measurement

The asynchronous simulator uses worker threads, per-agent inboxes, algorithm messages, Lamport metadata, and per-agent local logical clocks.

Each agent maintains the latest known message/value from its neighbors. An agent activation uses the latest available neighbor information and does not wait for all neighbors to send a new message.

Every time an agent performs an asynchronous activation, its local clock increments by 1. The global asynchronous step is:

```text
global_async_step = min(agent.local_clock for all agents)
```

The asynchronous run stops only when `global_async_step >= 1000` for the default experiment. It does not stop merely because queues are temporarily empty; the scheduler continues activations so all agents can reach the requested local step count.

## Project Layout

```text
backend/
  config.py
  dcop/
    problem.py
    generator.py
    cost.py
  algorithms/
    base.py
    dsa_c.py
    mgm.py
    mgm2.py
    dms.py
  simulators/
    synchronous.py
    asynchronous.py
  experiments/
    progress.py
    runner.py
    results.py
    plotting.py

main.py
run_experiments.py
requirements.txt
```
