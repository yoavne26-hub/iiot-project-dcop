# Assignment 2 - Clarifications and Additional Implementation Requirements

This document adds clarifications to Assignment 2 following the discussion held in class. The clarifications below are a direct continuation of the assignment requirements and should therefore be treated as part of the implementation, execution, measurement, and reporting requirements.

## 1. DMS Parameter

In the implementation of the `Max-sum + Damping (DMS)` algorithm, use the following value:

```text
lambda = 0.9
```

That is, wherever the damping mechanism is implemented in DMS, the damping parameter must be fixed at `0.9`. Comparisons with other lambda values are not required unless they are used separately for internal testing only.

## 2. Presenting the Average Cost in the Graphs

The average cost of each algorithm may be presented in the graphs once every five iterations.

Instead of displaying a measurement point for every individual iteration, points may be shown at:

```text
0, 5, 10, 15, ..., 1000
```

The graph should still represent the algorithm's progress throughout the run, but in a cleaner and more readable form. The same sampling method must be used consistently across all algorithms and both simulators.

## 3. Asynchronous Execution of the Algorithms

During the asynchronous execution of each algorithm, agents do not wait to receive messages from all of their neighbors.

Each agent performs an iteration whenever it receives at least one new message. During the computation, the agent uses the most recent information currently available from every neighbor. Specifically:

- If a new message has arrived from a certain neighbor, the latest message from that neighbor is used.
- If no new message has arrived from another neighbor, the agent continues using the most recently known message from that neighbor.
- The agent does not need to wait for every neighbor to send a message in the same round.

In practice, each agent should maintain an `agent_view`, or a similar data structure, containing the latest message received from every neighbor.

## 4. Presenting the Asynchronous Execution Graph

To present the graph of the asynchronous execution, use the clock synchronization mechanism studied in Lamport's second lecture.

The horizontal axis of the asynchronous graph does not represent ordinary global iterations, because an asynchronous system has no shared global iteration. Instead, it should represent the number of local steps, meaning the number of unsynchronized iterations performed by the agents.

Therefore, each agent should maintain a local step counter during execution:

```text
local_step_i
```

This counter is incremented whenever the agent performs a computation after receiving at least one message.

When sampling the solution cost for the graph, use logical-clock synchronization, or an equivalent mechanism that allows the local progress of the agents to be compared, rather than relying on a regular global iteration counter.

## 5. Stopping Condition for the Asynchronous Execution

The asynchronous execution must stop when the agent with the lowest clock value reaches 1000 steps.

If each agent `Ai` has a local step counter `local_step_i`, the stopping condition is:

```text
min(local_step_1, local_step_2, ..., local_step_n) >= 1000
```

It is not sufficient for only one agent, or only some of the agents, to reach 1000 steps. The execution ends only after the least-advanced agent has also reached 1000 steps. This provides a fairer comparison between different asynchronous runs and between the algorithms.

## 6. Recommended Implementation Consequences

To satisfy these clarifications, the implementation should include the following components.

### 6.1 Store the Latest Message from Every Neighbor

Each agent should maintain a data structure such as:

```python
last_messages = {
    neighbor_id: last_message_from_neighbor
}
```

Whenever a new message arrives, only the entry corresponding to the sending neighbor is updated.

### 6.2 Triggering an Asynchronous Iteration

During asynchronous execution, an agent performs a step whenever at least one new message has been received:

```text
if received_at_least_one_message:
    update_local_view()
    perform_algorithm_step()
    local_step += 1
    send_messages_to_neighbors()
```

### 6.3 Measuring the Average Cost

The average solution cost must be measured for each algorithm over the same 50 problem instances and in the same way for all algorithms.

If graph points are presented once every five iterations, the same rule must be applied to all algorithms and to both simulators.

### 6.4 Fair Comparison Between Algorithms

To maintain a fair comparison, make sure to use:

- The same 50 problem instances for all algorithms.
- The same problem instances in both the synchronous and asynchronous simulators.
- A controlled random seed so that the results can be reproduced.
- The same initial assignment when comparing different versions of an algorithm.

## 7. Short Text That May Be Included in the Report

Following the clarifications provided in class, DMS was implemented with `lambda = 0.9`. During asynchronous execution, each agent performs a computation step whenever it receives at least one message, while using the latest message received from every neighbor. Because an asynchronous system has no shared global iteration, the asynchronous graphs were measured using a Lamport-style clock synchronization mechanism based on the number of local steps performed by the agents. The stopping condition was defined so that execution ends only when the agent with the lowest number of steps reaches 1000 steps. For graph readability, the average cost is displayed once every five iterations or steps.
