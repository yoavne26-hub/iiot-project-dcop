# IIoT Assignment 2 — Distributed Constraint Optimization: A Research Analysis

This repository implements and compares four **incomplete distributed algorithms**
for solving randomly generated **Distributed Constraint Optimization Problems
(DCOPs)**, under both a **synchronous** and an **asynchronous** execution model.

This document is a self-contained research write-up: it defines the problem,
explains *how and why* each algorithm works, presents the experimental results as
graphs, analyses each graph, and explains *why we see the behaviour we see*. It
also validates the implementation against a reference implementation on identical
problem instances.

> **TL;DR.** The three *local-search* algorithms (DSA‑C, MGM, MGM‑2) converge
> within ~30 iterations to a good local optimum (≈ 37 % cost reduction at 50
> agents). The *inference* algorithm (DMS / damped min‑sum) **oscillates for the
> entire run and never settles** — because the random DCOP factor graphs are
> densely cyclic, where belief propagation has no convergence guarantee. Crucially,
> DMS's *best-seen* assignment is competitive (≈ 34–38 %); it simply drifts away
> from it. On identical instances our backend matches the reference implementation
> (MGM and DMS bit‑for‑bit; DSA‑C and MGM‑2 within stochastic noise).

---

## 1. The DCOP problem

A **DCOP** is defined by:

- A set of **agents** `0..N-1`, each owning exactly one **variable**.
- A finite **domain** `{0, 1, …, D-1}` of values each variable can take.
- A set of **binary constraints**. A constraint between agents *i* and *j* is a
  `D×D` integer **cost matrix** `C_ij`, where `C_ij[a][b]` is the cost incurred
  when agent *i* picks value *a* and agent *j* picks value *b*.

The **global cost** of an assignment is the sum of the costs of every constraint,
counted once per edge:

```
cost(assignment) = Σ over edges (i,j)  C_ij[ assignment[i] ][ assignment[j] ]
```

The goal is to **minimise** the global cost. **Lower is better.**

### A tiny worked example

Two agents `A`, `B`, domain `{0,1}`, one constraint:

```
          B=0   B=1
   A=0  [   5  ,  2  ]
   A=1  [   8  ,  1  ]
```

- Assignment `A=0, B=0` → cost `5`.
- Assignment `A=0, B=1` → cost `2`.
- Assignment `A=1, B=1` → cost `1`  ← optimal here.

With one binary constraint this is trivial; with 50 agents and a dense web of
conflicting constraints, finding the global optimum is **NP‑hard**, which is why
we use *incomplete* algorithms that find good solutions quickly rather than
provably optimal ones.

---

## 2. Experimental setup

Problems are generated randomly and reproducibly (`backend/dcop/generator.py`):
an Erdős–Rényi graph where each pair of agents shares a constraint with
probability `p`, and every constraint gets a random `D×D` cost matrix with entries
in `[1, max_cost]`. The same generated problems (and the same initial assignment)
are reused across all algorithms and both simulators, so comparisons are fair.

The graphs in this report come from the **full simulation run**:

| Parameter | Value |
|---|---|
| Problems (averaged) | 50 |
| Agents | 50 |
| Domain size | 10 |
| Constraint probability `p` | 0.30 (≈ 367 edges, mean degree ≈ 15) |
| Constraint cost range | 1 – 100 |
| Iterations / async steps | 500 |
| DSA‑C move probability | 0.70 |
| DMS damping λ | 0.90 |
| Mean initial cost | 18,552 |

Each curve is the **average global cost over the 5 problems** at each iteration.

---

## 3. The four algorithms

### 3.1 DSA‑C — Distributed Stochastic Algorithm (variant C)

**How it works.** Every iteration, *all* agents act simultaneously. Each agent
looks at its neighbours' latest values, computes the value that minimises its own
local cost, and — **with probability `p = 0.70`** — switches to it. Variant **C**
accepts a move when the new local cost is *less than or equal* to the current one
(it allows equal‑cost "plateau" moves), which distinguishes it from variant A
(strict improvement only).

**Why it works.** It is parallel randomised hill‑climbing. The probability `p` is
the crucial ingredient: if every agent always moved greedily and simultaneously,
two neighbours would react to each other's *stale* values and overshoot, causing
oscillation. Moving only with probability `p` decorrelates neighbours and lets the
system settle. Equal‑cost moves let it drift across plateaus and escape shallow
traps.

**What to expect.** A fast drop to a local optimum, then a flat plateau. Cheap per
iteration; only value messages are exchanged.

### 3.2 MGM — Maximum Gain Messages

**How it works.** Two phases per iteration. (1) Each agent computes its **gain**
(how much it could reduce its local cost by moving) and sends it to its
neighbours. (2) An agent actually moves **only if its gain is strictly the largest
in its neighbourhood** (ties broken by lower agent id).

**Why it works.** Because two neighbours can never move in the same round, the
global cost is **monotonically non‑increasing** — MGM never thrashes. It provably
converges to a **1‑opt local optimum** (no single agent can improve). It is
deterministic.

**What to expect.** A smooth, monotone descent to a plateau, often reached in very
few iterations. Its weakness is that it gets stuck at the first 1‑opt optimum.

### 3.3 MGM‑2 — coordinated pair moves

**How it works.** Extends MGM with **2‑opt** moves. With probability 0.5 an agent
*offers* to a random neighbour; the receiver evaluates the best *joint* move for
the pair and accepts/rejects (propose → accept/reject handshake). Pairs and
unpaired singletons then compete by gain (ties broken by the lower group leader),
and the locally‑dominant improving groups commit together.

**Why it works.** Some local optima trap any single‑agent (1‑opt) algorithm: no
agent can improve alone, but a *pair* changing together can. By coordinating pair
moves, MGM‑2 escapes a class of optima that MGM and DSA‑C cannot, so it can reach
a (slightly) lower cost — at the price of the most communication (the handshake).

**What to expect.** Fast descent like MGM, to the **lowest plateau** of the four,
but with the **most messages**.

### 3.4 DMS — Max‑Sum with damping (implemented as Min‑Sum)

**How it works.** A belief‑propagation / inference method on the **factor graph**
(variable nodes + one factor node per constraint). Two message types flow along
each edge: variable→factor (`Q`) and factor→variable (`R`). Messages are vectors
over the domain; `R` is computed by minimising the constraint cost plus the
incoming `Q` (min‑sum). **Damping** (`λ = 0.9`) blends each new message with the
previous one to suppress oscillation. Each agent forms a **belief** = sum of
incoming `R` messages and picks the value with minimum belief.

**Why it works — and why it struggles here.** On a **tree‑structured** factor
graph, max‑sum computes the exact optimum. But random DCOPs at `p = 0.30` are
**densely cyclic ("loopy")**, and on loopy graphs min‑/max‑sum has **no
convergence guarantee**: messages circulate around cycles, beliefs oscillate, and
the decoded assignment keeps changing. Damping reduces but does not eliminate
this. The result is a **noisy cost curve that never settles**.

**What to expect.** A jagged, oscillating curve sitting well above the
local‑search trio.

---

## 4. The two simulators

**Synchronous** (`backend/simulators/synchronous.py`). Global rounds with
barriers: every agent reads a consistent snapshot, all act, the round advances.
Deterministic for a given seed → smooth, repeatable curves. The x‑axis is the
iteration number.

**Asynchronous** (`backend/simulators/asynchronous.py`). One **thread + mailbox
per agent**; each agent owns its state and acts whenever it has received at least
one message, using the **latest** message from every neighbour (an `agent_view`).
There is no global clock, so progress is measured by a **Lamport‑style minimum
local step** across all agents, and a run stops when the slowest agent reaches the
step limit. Because thread scheduling is non‑deterministic, async curves are
**noisier** than sync, even though final quality is comparable.

---

## 5. Results — synchronous

![Synchronous average cost](report/figures/backend_sync.png)

**What the graph shows.** The three local‑search algorithms (blue = DSA‑C ●,
orange = MGM ▲, green = MGM‑2 ★) collapse from 18,552 to ≈ 11,500–11,750 within
**~30 iterations** and then stay perfectly flat for the remaining ~470 iterations.
DMS (red ■) drops less and **oscillates between ~14,000 and ~16,300 for the entire
run** — it never converges.

**Average results over the 5 problems (synchronous, 500 iterations):**

| algorithm | final cost | drop | best‑seen | messages | runtime |
|---|--:|--:|--:|--:|--:|
| dsa‑c | 11,634 | 37.3 % | 11,634 | 361,200 | 34 s |
| mgm   | 11,738 | 36.7 % | 11,738 | 722,400 | 34 s |
| mgm2  | 11,541 | **37.8 %** | 11,541 | 1,083,600 | 70 s |
| dms   | 14,795 | 20.2 % | **12,201 (34.2 %)** | 722,400 | 1,014 s |

**Why we see this.**

- **Local search wins on these instances.** DSA‑C / MGM / MGM‑2 exploit locality
  and reach a good 1‑opt/2‑opt optimum within ~30 iterations; the remaining
  iterations add nothing. MGM‑2 reaches the lowest plateau because its pair moves
  escape optima the single‑agent methods cannot.
- **DMS oscillates and never settles** — the loopy‑graph behaviour of §3.4. But
  its *best‑seen* cost (12,201, a 34.2 % reduction) is close to local search: DMS
  *does* find good assignments, it just drifts away from them (see §7).
- **Communication cost** scales with the message pattern: DSA‑C sends one value
  message per neighbour (cheapest), MGM adds gains (2×), DMS exchanges Q/R vectors,
  and MGM‑2's propose/accept handshake is the most expensive (3×). Because the
  synchronous simulator runs the full iteration count (no early stop), these scale
  linearly with iterations.
- **Runtime** is dominated by the synchronous DMS (1,014 s). It is the only
  pure‑Python algorithm; the async DMS (and the reference) are numpy‑vectorised and
  ~30× faster. This is why a full 50‑agent / 1000‑iteration run currently takes
  ~7.4 h — see §10.

---

## 6. Results — asynchronous

![Asynchronous average cost](report/figures/backend_async.png)

**Average results over the 5 problems (asynchronous, 500 steps):**

| algorithm | final cost | drop | best‑seen | runtime |
|---|--:|--:|--:|--:|
| dsa‑c | 11,762 | 36.6 % | 11,762 | 11 s |
| mgm   | 11,784 | 36.5 % | 11,784 | 8 s |
| mgm2  | 11,713 | 36.9 % | 11,488 | 10 s |
| dms   | 14,129 | 23.8 % | **11,566 (37.7 %)** | 36 s |

**Why we see this.** The qualitative picture matches the synchronous case — local
search converges fast and low, DMS oscillates high — confirming the behaviour is a
property of the *algorithms*, not the execution model. The async curves are
**jaggier**, because each agent acts on whatever messages have arrived so far (not
a clean global snapshot) and thread timing varies run to run. DMS is the noisiest
of all; its *best‑seen* assignment (11,566, 37.7 %) actually **matches local
search**, again showing that DMS's poor *final* number is an artefact of its
oscillation, not of solution quality.

> **Engineering note.** Reaching this required rewriting the async simulator to a
> true per‑agent‑thread design (each agent owns its state; no global lock). An
> earlier event‑driven design that re‑decided on *every* message made local search
> thrash to a much worse solution (~15 % drop) and could livelock MGM‑2.

---

## 7. Why DMS oscillates (the key insight)

It is tempting to read the graphs as "DMS is a worse algorithm," but the precise
statement is: **min‑sum is exact on trees and unreliable on dense cyclic graphs.**
Our random DCOPs at `p = 0.30` have ~367 edges over 50 nodes — far from a tree,
full of short cycles. Messages propagate around those cycles and reinforce
themselves, so beliefs oscillate instead of fixing a consistent optimum. Damping
(`λ = 0.9`) is exactly the standard remedy and it *helps* (without it the
oscillation is worse), but it cannot make a loopy graph behave like a tree.

The **best‑seen** columns in §5/§6 are the clearest evidence: across its
oscillation DMS *visits* assignments as good as ~34–38 % reduction — competitive
with local search — but the decode‑every‑step rule reports wherever it happens to
be at the final step. An *anytime* rule (keep the best assignment ever seen) would
make DMS competitive. Local search has no such issue: it only ever makes locally
improving moves, so its final state *is* its best state.

---

## 8. Implementation validation against the reference

Absolute costs cannot be compared directly between our backend and the reference
implementation, because the two use **different random number generators** and
therefore generate **different problem instances**. To validate correctness we
instead ran **both implementations on identical instances** (backend‑generated
problems converted into the reference's representation) in a dedicated matched
experiment (30 agents, 200 iterations) and overlaid the curves.

![Synchronous, identical instances](report/figures/same_instance_sync.png)

The coloured (backend) and black‑dashed (reference) curves lie on top of each other:

| algorithm | backend drop | reference drop | note |
|---|--:|--:|---|
| dsa‑c | 46.8 % | 46.0 % | within noise (backend slightly better) |
| mgm   | 44.0 % | 44.0 % | **identical** (MGM is deterministic) |
| mgm2  | 46.1 % | 47.8 % | within stochastic noise |
| dms   | 25.5 % | 25.5 % | **identical** |

![Asynchronous, identical instances](report/figures/same_instance_async.png)

In the asynchronous case the curves again occupy the same band; remaining
differences are thread‑scheduling / stochastic noise. **Conclusion: on equal
footing the two implementations are equivalent.**

---

## 9. Summary

- **Algorithm ranking on dense random DCOPs (50 agents):** MGM‑2 ≈ DSA‑C ≈ MGM
  (≈ 37 % reduction, converged by ~30 iterations) ≫ DMS by *final* cost (20–24 %,
  oscillating) — though DMS's *best‑seen* is competitive (~34–38 %).
- **MGM** is the smoothest and stable but most easily trapped; **MGM‑2** reaches
  the lowest cost but is the most communication‑hungry; **DSA‑C** is a good cheap
  middle ground.
- **DMS** is the wrong tool for densely‑cyclic instances — its theory only
  guarantees optimality on trees — but it would benefit greatly from an anytime
  (best‑so‑far) decode rule.
- **Synchronous vs asynchronous** make little difference to final quality; async
  is just noisier.
- **Our backend is validated**: on identical instances it matches the reference
  implementation (MGM and DMS exactly; DSA‑C/MGM‑2 within noise).

---

## 10. Reproducing the experiments

The full simulation in this report (5 problems, 50 agents, 500 iterations):

```bash
python main.py --mode full --problems 5 --agents 50 --iterations 500 \
  --algorithms dsa-c,mgm,mgm2,dms --simulators sync,async \
  --output-dir results
```

Default full experiment (50 problems, 50 agents, 1000 iterations):

```bash
python main.py
```

Useful flags: `--constraint-probability`, `--seed`, `--no-progress`, `--no-plots`,
`--output-dir`. Outputs are average‑cost CSVs and the PNG graphs shown above.

> Note: the default 50/50/1000 run currently takes ~7.4 h, dominated by the
> pure‑Python synchronous DMS; vectorising it with numpy (as the async DMS already
> is) would bring it to ~1.5 h.

### Project layout

```
backend/
  dcop/         problem model, random generator, cost evaluation
  algorithms/   dsa_c.py, mgm.py, mgm2.py, dms.py (+ base.py)
  simulators/   synchronous.py, asynchronous.py
  experiments/  runner.py, results.py, plotting.py, progress.py
report/figures/ graphs used in this analysis
main.py         CLI entry point
```
