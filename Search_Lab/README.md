# Search Lab: Warehouse Navigation with A*

Standard library only (Python 3). Run everything with:

```bash
python run_search_lab.py        # writes results/results.txt
```

| File | Purpose |
|---|---|
| `search.py` | Grid, A\* (with pluggable heuristic), BFS, independent path validator, independent flood-fill ground truth |
| `run_search_lab.py` | All tests (Tasks 3, 5, 6) and the randomised check |
| `results/results.txt` | Full output of the run |

---

## Task 0: The search problem P = (S, A, T, s0, G, c)

| Component | Specification |
|---|---|
| State S | A grid position (row, col) of a free cell (`.`, `S` or `G`) |
| Actions A | Up, Down, Left, Right |
| Transition T | T((r,c), a) = (r,c) + delta(a) if that cell is inside the map and not `#`; otherwise the action is invalid |
| Initial state s0 | Position of `S`, which is (1, 1) |
| Goal G | {position of `G`} = {(7, 15)} |
| Cost c | 1 per move |

**(a)** Only the robot's position (row, col). The walls never change, so they belong to the environment, not the state.
**(b)** An action is invalid if it would leave the map or move into a `#` cell.
**(c)** Yes. Each action has exactly one outcome, the environment is fully observable and static.
**(d)** A sequence of valid actions that starts at `S` and ends at `G`. It is optimal if it has the fewest moves (cost = length).

## Task 1: Design of the agent

1. **State in Python:** a tuple `(row, col)`.
2. **Warehouse:** a list of strings (`Grid.rows`), parsed once. `S` and `G` are located while parsing.
3. **Valid actions:** `Grid.successors(pos)` tries the four deltas and yields only the ones that stay inside the map and hit a non-`#` cell.
4. **Goal recognition:** `Grid.is_goal(pos)` is `pos == goal`.
5. **Frontier contents:** a heap of `(f, tie-break h, counter, g, state)`. A `parent` dict stores the predecessor of each state, and a `g` dict stores the best known cost-to-come.
6. **Path reconstruction:** follow `parent` pointers from the goal back to `S` and reverse.

**Reported at termination:** whether a solution was found, the path, its length and the number of states expanded.

## Task 2: Prompt used with the LLM

```
I am implementing a simple goal-based search agent in Python.
The environment is a grid represented by an ASCII map. The agent starts at S and must
reach G. The symbols # represent obstacles and . represents free cells. The agent can
move up, down, left, or right, and every movement has cost 1.
Implement A* search. Use Manhattan distance as the heuristic
h(n) = |x - x_G| + |y - y_G|.
The program should:
- represent grid positions as states;
- maintain an appropriate frontier;
- calculate g(n), h(n) and f(n);
- avoid repeatedly expanding the same state;
- reconstruct the path when the goal is reached;
- report the path and its length;
- report the number of states expanded.
Keep the implementation simple and explain the main components of the code.
Use only the Python standard library.
```

Follow-up prompt (Task 5): *"Add a BFS version of the same agent on the same Grid class and report the same measures."*
Follow-up prompt (Task 6): *"Let `astar` take the heuristic as a parameter so I can swap Manhattan for other functions."*

## Task 3: Tests

| Test | Result |
|---|---|
| **1. Original warehouse** | Path found, **length 40**, **64 states expanded**. Path is valid (checked by an independent validator) and equals the independent flood-fill shortest length (40). |
| **2. Trivial** (`#SG##`) | Found, length 1, 2 expanded. |
| **3. No solution** (goal walled in) | A\* and BFS both report `found=False` after expanding the 9 reachable cells. No infinite loop. |
| **4. Alternative paths** | Two routes exist; A\* returned a length-6 path, equal to the independent shortest length (6). |
| **Extra: randomised** | 300 random 8x12 maps (178 solvable): A\* and BFS agree with the independent ground truth on solvability and on shortest length in **all 300** cases. |

The A\* path on the original warehouse (`*` marks the route):

```
#################
#S****#*********#
#.###*#*#######*#
#...#*#*******#*#
###.#*#######*#*#
#...#*********#*#
#.###########.#*#
#.............#G#
#################
```

## Task 4: Where each concept appears in the code

| Concept | Where |
|---|---|
| State | a `(row, col)` tuple; used as dict key in `g`, `parent`, `closed` |
| Action | the `ACTIONS` dict (Up/Down/Left/Right deltas) |
| Transition | `Grid.successors`, which computes `nxt = pos + delta` and checks `is_free` |
| Goal test | `Grid.is_goal`, called when a node is popped in `astar` |
| g(n) | the `g` dict and the `gcost` / `new_g` variables |
| h(n) | the heuristic function passed as `h` (default `manhattan`) |
| f(n) | first element of every heap entry: `new_g + hn` |
| Frontier | `frontier`, a binary heap (`heapq`) |
| Visited states | the `closed` set (plus the `g` dict for discovered states) |
| Path reconstruction | `reconstruct(parent, state)` |

**(a)** A binary min-heap (`heapq`).
**(b)** By popping the entry with the smallest f = g + h; ties are broken by smaller h, then insertion order.
**(c)** In the `manhattan` function, called when a successor is pushed (and once for the start state).
**(d)** Yes: `new_g + hn` is computed and stored as the first element of the heap entry.
**(e)** A state is only expanded once: popped states go in `closed` and stale duplicates are skipped. A neighbour is only re-pushed if a cheaper `g` was found.

## Task 5: BFS vs A\*

**Original warehouse**

| Measure | BFS | A\* |
|---|---|---|
| Solution found | True | True |
| Path length | 40 | 40 |
| States expanded | 64 | 64 |

**Supplementary: 10x20 room with one shelf (S top-left, G bottom-right)**

| Measure | BFS | A\* (Manhattan) |
|---|---|---|
| Solution found | True | True |
| Path length | 28 | 28 |
| States expanded | 156 | 29 |

**(a)** Yes, both found a solution. **(b)** Yes, same length (40, and 28), as expected because both are optimal for unit costs.
**(c)** On the original warehouse, neither: both expanded all 64 free cells. On the shelf room A\* expanded far fewer (29 vs 156).
**(d)** The lab's warehouse is a single winding corridor, so there is only one route and no heuristic can prune it; the goal is reached only after every corridor cell has been visited. In the open room, Manhattan distance guides A\* along the diagonal towards G so cells far from the direct route are never expanded. A\* only saves work when the heuristic can tell good directions from bad ones.

Tie-breaking also matters. In the shelf room, A\* with FIFO tie-breaking expanded 120 states; breaking ties in favour of the smaller h expanded 29. Many nodes share the same f, so the order among ties decides how much is explored.

## Task 6: Heuristic investigation

**Original warehouse** (true shortest = 40): every heuristic found length 40 and expanded 64 states (the single corridor gives no room to differ).

**Shelf room** (true shortest = 28):

| Heuristic | Path length | Expanded | Optimal? |
|---|---|---|---|
| Manhattan | 28 | 29 | yes |
| h = 0 | 28 | 156 | yes |
| Euclidean | 28 | 101 | yes |
| 2 x Manhattan | 28 | 29 | yes |

**Small maze designed so over-estimating hurts** (true shortest = 10):

| Heuristic | Path length | Expanded | Optimal? |
|---|---|---|---|
| Manhattan | 10 | 17 | yes |
| h = 0 | 10 | 21 | yes |
| Euclidean | 10 | 14 | yes |
| 2 x Manhattan | 10 | 18 | yes |
| 5 x Manhattan | 14 | 15 | **no** |

**Interpretation.**
- **h = 0** is admissible but uninformative: A\* turns into uniform-cost search and expands the most states (156 in the shelf room).
- **Euclidean** is admissible (never above the true cost) but weaker than Manhattan on a 4-connected grid because it underestimates more; it expanded 101 states in the shelf room against 29 for Manhattan.
- **Manhattan** is admissible and the tightest of these for 4-directional moves.
- **Multiplied heuristics** are no longer admissible. 2 x Manhattan was still optimal on all of our maps, but that is not guaranteed; 5 x Manhattan was too aggressive and returned a path of length 14 instead of 10, i.e. it traded optimality for expanding fewer states. A heuristic that over-estimates pushes the search greedily towards the goal and can commit to a route before a cheaper one is discovered.

## Task 7: Evaluation of the LLM-generated agent

1. **Correct immediately:** the overall structure (heap-based frontier, `g`/`parent` dicts, closed set, path reconstruction). It solved the original warehouse optimally on the first run.
2. **Bugs / design problems found:** (i) FIFO tie-breaking made A\* expand 120 states in the shelf room instead of 29, so the tie-break was changed to prefer smaller h. (ii) In my results script, a variable name (`truth`) was reused inside the random-test loop, which made the "Optimal?" column of the heuristic table falsely say `False` for correct paths.
3. **How discovered:** (i) from the expanded-state counts in the room experiment; (ii) from noticing that the table said `False` although the lengths equalled the printed shortest length.
4. **Unfamiliar terms:** stale heap entries ("lazy deletion") and the tie-break counter needed to stop `heapq` comparing states.
5. **Modified:** yes, the tie-breaking change and the parameterised heuristic.
6. **Most useful tests:** Test 3 (no solution) and the independent flood-fill ground truth, because they check behaviour that a plausible-looking path cannot show.
7. **Could I have trusted it without testing?** No. The output looked right, but it took the experiments to expose the tie-breaking cost and the invalid table.
8. **Learned about A\*:** the heuristic only helps if the map gives it something to prune; on a single-corridor map A\* and BFS are the same amount of work; and admissibility is what buys optimality.

**Designed by me:** the problem formulation, the design in Task 1, the tests and the experiments. **Suggested by the LLM:** the code. **Accepted:** the heap/closed-set structure. **Changed:** tie-breaking, heuristic parameter. **Tested:** everything above.

## Final reflection

1. **Why formulate the problem first?** The formulation (states, actions, transitions, goal, costs) is the specification of the algorithm's input. If it is unclear, the code will be, too, and there is nothing to test it against. It also separates the problem (what is being solved) from the solution (how).
2. **In what sense is A\* informed?** It uses problem-specific knowledge in the form of h(n), an estimate of the remaining cost, to order the frontier by f = g + h. BFS only uses the distance already travelled.
3. **Why does the heuristic matter?** It decides both efficiency and correctness: a tight admissible heuristic gives optimal paths with few expansions (Manhattan: 29 vs 156 for h = 0), while an over-estimating one can return sub-optimal paths (5 x Manhattan: 14 vs 10).
4. **What did the LLM contribute?** A first working implementation and the boilerplate (heap handling, path reconstruction), which frees time for design and testing. It did not supply the problem understanding or the validation.
5. **What could go wrong if code is accepted untested?** Plausible but wrong answers: a non-shortest path, an infinite loop on unsolvable maps, or silently wasteful search (as with tie-breaking). None of these is visible from a single successful run.
