# Agents Lab: Goal-Based Warehouse Navigation Agent

Standard library only (Python 3).

```bash
python warehouse_agent.py        # prints results; saved copy in results/results.txt
```

| File | Purpose |
|---|---|
| `warehouse_agent.py` | `WarehouseAgent` (state, goal, action model, A\* decision component), runner, map scaler, independent BFS check |
| `results/results.txt` | Output for the three scenarios below |

The warehouse map (21 x 7):

```
#####################
#S....#............G#
#.##....##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################
```

---

## Task 1: Understanding the problem

1. **Environment:** a 2-D grid warehouse (static, fully observable, deterministic, single agent). `#` cells are shelving that cannot be crossed.
2. **Goal:** reach the dispatch area `G` from the start `S`, collision-free.
3. **Actions:** Up, Down, Left, Right, each moving one square.
4. **Information the agent must maintain:** its current position, the position of the goal, the map (which cells are obstacles) and, while planning, which positions it has already explored and the cheapest way found to reach each.
5. **Why goal-based rather than simple reflex?** A reflex agent maps the current percept straight to an action ("if wall on the right, turn up") and has no notion of where it is trying to get. Here, a good move depends on the *future*: a move that looks blocked or that points away from `G` may be part of the only route. The agent therefore keeps an explicit goal and a model of what its actions do, and it chooses actions by considering the sequence of their consequences (search).

**Think About It: a warehouse twice as large.** I doubled the map in each direction (`scaled()` in the code) and ran both A\* and plain BFS. The path length went from 20 to 40 moves; A\* expanded 50 states and BFS 237 (original: A\* 21, BFS 59). Search is still appropriate, since the space is small and fully known, but the cost grows roughly with the number of cells (x4) and blind search grows fastest. Additional difficulties for a real warehouse: much larger frontiers and memory, many equal-cost ties, moving obstacles or other vehicles (plans become stale, so replanning is needed), several packages/goals, and non-uniform costs (turns, congested aisles). Larger warehouses call for hierarchical planning, precomputed distances, or incremental methods (e.g. D\* Lite).

## Task 2: Design of the agent

| Component | In this agent |
|---|---|
| Environment | the grid of characters (`self.grid`) |
| Current state | the agent's position `(row, col)` (`self.position`) |
| Goal | position of `G` (`self.goal`, tested by `goal_test`) |
| Available actions | Up, Down, Left, Right (`MOVES`); validity given by `result()` (returns `None` on collision) |
| Decision-making component | `decide()`: A\* search with Manhattan-distance heuristic; returns an action sequence |

```
             +---------------------------------------------+
             |               GOAL-BASED AGENT              |
 percept     |   +--------------+     +------------------+ |
 (position,  |   | current state|     |  goal: reach G   | |
  map)  --->-|-->| (row, col)   |     +---------+--------+ |
             |   +------+-------+               |          |
             |          |   +-------------------v-------+  |
             |          +-->| model: what does each     |  |
             |              | action do? (result())     |  |
             |              +-------------+-------------+  |
             |                            |                |
             |              +-------------v-------------+  |
             |              | decision making: A* search|  |
             |              | -> best action sequence   |  |
             |              +-------------+-------------+  |
             +----------------------------|----------------+
                                          | actions (Up/Down/Left/Right)
                                          v
                              +-----------------------+
                              |  ENVIRONMENT (warehouse)|
                              +-----------------------+
```

## Task 3: Prompt engineering

**Prompt used**

```
Write a well-documented Python program implementing a goal-based agent for the warehouse
navigation problem below.
[map]
The program should
- represent the warehouse as a two-dimensional grid;
- determine a collision-free path from S to G;
- avoid all obstacles;
- print either the path found or a suitable message if no path exists;
- explain the search algorithm that has been chosen and why it is appropriate.
Structure it as an agent with an explicit current state, goal, model of the actions and a
decision-making step. Use only the Python standard library.
```

**Questions**

1. **Did the LLM generate a working program on the first attempt?** The agent itself (`WarehouseAgent`) ran correctly on the first run and found a 20-move path. One helper I added for the "twice as large" extension did not work at first: the map scaler copied `S` and `G` into 2x2 blocks, giving several starts and goals. I noticed this by printing the rendered map and fixed `scaled()` to keep `S` and `G` single cells.
2. **How could the prompt be improved?** By stating the input format (rows of equal length, one `S` and one `G`), the required outputs (path, length, states expanded), the no-path behaviour, and the allowed libraries, and by asking for a self-check (validate the path).
3. **What algorithm was chosen?** A\* search with the Manhattan-distance heuristic.
4. **Why?** All moves cost 1 and only 4 directions are allowed, so Manhattan distance never over-estimates the remaining cost (admissible); A\* is then guaranteed to return a shortest path and typically expands fewer states than BFS. The state space is small and fully known, so its memory cost is acceptable.

## Testing

| Scenario | Result |
|---|---|
| Original warehouse | Path found: **20 moves**, **21** states expanded. The move sequence is executed in the environment with a collision check. |
| Independent check | A plain flood-fill BFS (separate code) gives the same shortest length, 20, expanding 59 states. |
| Unreachable goal (`#S#G#`) | Prints `No path exists from S to G.` (1 state expanded). |
| Twice-as-large warehouse | 40 moves, A\* expanded 50 states; independent BFS gives 40 and expands 237. |

Path on the original warehouse (`*`):

```
#####################
#S****#************G#
#.##.***##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################
```

## Reflection on LLM-assisted development

- **Strengths:** it turns a clear specification into working structured code quickly (heap-based A\*, path reconstruction, agent class), and explains its algorithm choice.
- **Limitations:** plausible output is not proof of correctness. Every claim above (optimal length, correct no-path behaviour, scaling) was checked with an independent method rather than accepted, and the one defect (the map scaler) was only found by looking at the output.
- **What I designed and tested myself:** the agent decomposition (Task 2), the tests, and the independent BFS check. **What the LLM supplied:** the implementation.
