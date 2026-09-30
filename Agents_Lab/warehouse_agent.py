"""Goal-based warehouse navigation agent (standard library only).

Agent architecture (see README for the block diagram):
    percept -> update state -> [goal + model of actions] -> choose action sequence -> act

The decision-making component is A* search with the Manhattan-distance heuristic.
It is appropriate because
  * every move costs 1 and moves are only up/down/left/right, so Manhattan distance
    never over-estimates the remaining cost (admissible) and A* returns a SHORTEST path;
  * it expands far fewer states than blind search when the goal is in a known direction;
  * the state space is small and fully observable, so exhaustive search is affordable.
"""
import heapq
import sys

WAREHOUSE = """\
#####################
#S....#............G#
#.##....##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################"""

MOVES = {"Up": (-1, 0), "Down": (1, 0), "Left": (0, -1), "Right": (0, 1)}


class WarehouseAgent:
    """A goal-based agent: it holds an explicit goal and a model of what its actions do."""

    def __init__(self, layout):
        # ---- environment (2-D grid) --------------------------------------
        self.grid = [list(row) for row in layout.strip("\n").split("\n")]
        assert len({len(r) for r in self.grid}) == 1, "rows must have equal length"
        self.start = self.goal = None
        for r, row in enumerate(self.grid):
            for c, ch in enumerate(row):
                if ch == "S":
                    self.start = (r, c)
                elif ch == "G":
                    self.goal = (r, c)
        assert self.start and self.goal, "layout needs S and G"
        # ---- agent's internal state and goal -----------------------------
        self.position = self.start        # current state
        self.plan = None
        self.states_expanded = 0

    # ---- model of the world: which actions are valid, what they do -------
    def free(self, pos):
        r, c = pos
        return 0 <= r < len(self.grid) and 0 <= c < len(self.grid[0]) and self.grid[r][c] != "#"

    def result(self, pos, action):
        dr, dc = MOVES[action]
        nxt = (pos[0] + dr, pos[1] + dc)
        return nxt if self.free(nxt) else None    # None = collision, action invalid

    def goal_test(self, pos):
        return pos == self.goal

    def heuristic(self, pos):
        return abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1])

    # ---- decision making: A* search over action sequences ----------------
    def decide(self):
        """Return a list of actions from the current position to the goal, or None."""
        counter = 0
        frontier = [(self.heuristic(self.position), self.heuristic(self.position), counter, 0, self.position)]
        g = {self.position: 0}
        parent = {self.position: None}      # state -> (previous state, action)
        closed = set()
        self.states_expanded = 0
        while frontier:
            _, _, _, cost, state = heapq.heappop(frontier)
            if state in closed:
                continue
            closed.add(state)
            self.states_expanded += 1
            if self.goal_test(state):
                actions = []
                while parent[state] is not None:
                    state, act = parent[state][0], parent[state][1]
                    actions.append(act)
                self.plan = actions[::-1]
                return self.plan
            for action in MOVES:
                nxt = self.result(state, action)
                if nxt is None:
                    continue
                new_cost = cost + 1
                if nxt not in g or new_cost < g[nxt]:
                    g[nxt] = new_cost
                    parent[nxt] = (state, action)
                    counter += 1
                    h = self.heuristic(nxt)
                    heapq.heappush(frontier, (new_cost + h, h, counter, new_cost, nxt))
        self.plan = None
        return None

    # ---- acting -----------------------------------------------------------
    def execute(self, plan):
        """Carry out the plan in the environment, checking every move for collisions."""
        pos, trace = self.position, [self.position]
        for act in plan:
            nxt = self.result(pos, act)
            if nxt is None:
                raise RuntimeError(f"collision at {pos} doing {act}")
            pos = nxt
            trace.append(pos)
        self.position = pos
        return trace

    def render(self, trace):
        cells = [row[:] for row in self.grid]
        for r, c in trace[1:-1]:
            cells[r][c] = "*"
        return "\n".join("".join(row) for row in cells)


def run(layout, title):
    print(f"--- {title} ---")
    agent = WarehouseAgent(layout)
    plan = agent.decide()
    if plan is None:
        print("No path exists from S to G.")
        print(f"States expanded before giving up: {agent.states_expanded}\n")
        return None
    trace = agent.execute(plan)
    assert agent.goal_test(agent.position), "plan did not reach the goal"
    print(agent.render(trace))
    print(f"Path length : {len(plan)} moves")
    print(f"Moves       : {' '.join(plan)}")
    print(f"Expanded    : {agent.states_expanded} states\n")
    return plan


def scaled(layout, k):
    """Warehouse k times larger in each direction.

    Walls and free cells become k x k blocks; S and G stay single cells
    (placed in the top-left of their block).
    """
    rows = layout.strip("\n").split("\n")
    plain = [row.replace("S", ".").replace("G", ".") for row in rows]
    big = []
    for row in plain:
        line = "".join(ch * k for ch in row)
        big.extend([line] * k)
    grid = [list(r) for r in big]
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch in "SG":
                grid[r * k][c * k] = ch
    return "\n".join("".join(r) for r in grid)


def flood_fill_shortest(layout):
    """Independent ground truth: plain breadth-first distances, no A* code reused."""
    from collections import deque
    agent = WarehouseAgent(layout)
    dist, queue, expanded = {agent.start: 0}, deque([agent.start]), 0
    while queue:
        cur = queue.popleft()
        expanded += 1
        if cur == agent.goal:
            return dist[cur], expanded
        for dr, dc in MOVES.values():
            nxt = (cur[0] + dr, cur[1] + dc)
            if agent.free(nxt) and nxt not in dist:
                dist[nxt] = dist[cur] + 1
                queue.append(nxt)
    return None, expanded


if __name__ == "__main__":
    original = run(WAREHOUSE, "Original warehouse")
    truth, bfs_exp = flood_fill_shortest(WAREHOUSE)
    print(f"Independent BFS shortest path = {truth} moves (agent found {len(original)}); "
          f"BFS expanded {bfs_exp} states\n")
    # A wall between S and G to test the no-path branch.
    run("#####\n#S#G#\n#####", "Unreachable goal")
    big = scaled(WAREHOUSE, 2)
    big_plan = run(big, "Warehouse twice as large in each direction")
    truth2, bfs_exp2 = flood_fill_shortest(big)
    print(f"Independent BFS shortest path = {truth2} moves (agent found {len(big_plan)}); "
          f"BFS expanded {bfs_exp2} states")
