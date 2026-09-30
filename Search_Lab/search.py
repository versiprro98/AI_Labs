"""Search lab: grid navigation with A* and BFS (standard library only).

Search problem P = (S, A, T, s0, G, c)
  S  : free grid cells (row, col)
  A  : Up, Down, Left, Right
  T  : (r, c) + delta if the target cell is inside the grid and not '#'
  s0 : position of 'S'
  G  : {position of 'G'}
  c  : 1 per move
"""
import heapq
import math
from collections import deque

WAREHOUSE = """\
#################
#S....#.........#
#.###.#.#######.#
#...#.#.......#.#
###.#.#######.#.#
#...#.........#.#
#.###########.#.#
#.............#G#
#################"""

ACTIONS = {"Up": (-1, 0), "Down": (1, 0), "Left": (0, -1), "Right": (0, 1)}


class Grid:
    def __init__(self, text):
        self.rows = [line for line in text.strip("\n").split("\n")]
        width = {len(r) for r in self.rows}
        assert len(width) == 1, "map rows must have equal length"
        self.start = self.goal = None
        for r, row in enumerate(self.rows):
            for c, ch in enumerate(row):
                if ch == "S":
                    self.start = (r, c)
                elif ch == "G":
                    self.goal = (r, c)
        assert self.start and self.goal, "map needs S and G"

    def is_free(self, pos):
        r, c = pos
        return 0 <= r < len(self.rows) and 0 <= c < len(self.rows[0]) and self.rows[r][c] != "#"

    def successors(self, pos):
        """Yield (action_name, next_state, cost) for every VALID action."""
        for name, (dr, dc) in ACTIONS.items():
            nxt = (pos[0] + dr, pos[1] + dc)
            if self.is_free(nxt):
                yield name, nxt, 1

    def is_goal(self, pos):
        return pos == self.goal

    def render(self, path=None):
        cells = [list(r) for r in self.rows]
        for (r, c) in (path or [])[1:-1]:
            cells[r][c] = "*"
        return "\n".join("".join(r) for r in cells)


# ---- heuristics -----------------------------------------------------------
def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def zero(a, b):
    return 0

def double_manhattan(a, b):
    return 2 * manhattan(a, b)


def reconstruct(parent, state):
    path = [state]
    while parent[state] is not None:
        state = parent[state]
        path.append(state)
    return path[::-1]


def astar(grid, h=manhattan, tie_break="h"):
    """A*: expand the frontier node with smallest f(n) = g(n) + h(n).

    frontier : binary heap of (f, tie, counter, g, state).
               tie = h(n) when tie_break == "h" (prefer nodes closer to the goal),
               tie = 0 when tie_break == "fifo" (ties broken by insertion order).
    g        : best known cost-to-come for every discovered state
    closed   : states already expanded (never expanded twice)
    Returns dict(found, path, length, expanded).
    """
    start, goal = grid.start, grid.goal
    counter = 0
    frontier = [(h(start, goal), h(start, goal) if tie_break == "h" else 0, counter, 0, start)]
    g = {start: 0}
    parent = {start: None}
    closed = set()
    expanded = 0
    while frontier:
        f, _, _, gcost, state = heapq.heappop(frontier)
        if state in closed:
            continue                      # stale heap entry
        closed.add(state)
        expanded += 1
        if grid.is_goal(state):           # goal test on expansion => optimal for admissible h
            path = reconstruct(parent, state)
            return dict(found=True, path=path, length=len(path) - 1, expanded=expanded)
        for _, nxt, cost in grid.successors(state):
            new_g = gcost + cost
            if nxt not in g or new_g < g[nxt]:
                g[nxt] = new_g
                parent[nxt] = state
                counter += 1
                hn = h(nxt, goal)
                heapq.heappush(frontier, (new_g + hn, hn if tie_break == "h" else 0, counter, new_g, nxt))
    return dict(found=False, path=None, length=None, expanded=expanded)


def bfs(grid):
    """Breadth-first search (uniform cost 1 => shortest path)."""
    start = grid.start
    frontier = deque([start])
    parent = {start: None}
    expanded = 0
    while frontier:
        state = frontier.popleft()
        expanded += 1
        if grid.is_goal(state):
            path = reconstruct(parent, state)
            return dict(found=True, path=path, length=len(path) - 1, expanded=expanded)
        for _, nxt, _ in grid.successors(state):
            if nxt not in parent:
                parent[nxt] = state
                frontier.append(nxt)
    return dict(found=False, path=None, length=None, expanded=expanded)


# ---- independent checks ---------------------------------------------------
def path_is_valid(grid, path):
    """Independent validator: starts at S, ends at G, unit steps, no walls."""
    if not path or path[0] != grid.start or path[-1] != grid.goal:
        return False
    for a, b in zip(path, path[1:]):
        if manhattan(a, b) != 1 or not grid.is_free(b):
            return False
    return True


def true_shortest_length(grid):
    """Ground truth by plain distance flood fill (no reuse of astar/bfs code)."""
    dist = {grid.start: 0}
    queue = deque([grid.start])
    while queue:
        cur = queue.popleft()
        for dr, dc in ACTIONS.values():
            nxt = (cur[0] + dr, cur[1] + dc)
            if grid.is_free(nxt) and nxt not in dist:
                dist[nxt] = dist[cur] + 1
                queue.append(nxt)
    return dist.get(grid.goal)


if __name__ == "__main__":
    g = Grid(WAREHOUSE)
    res = astar(g)
    print(g.render(res["path"]))
    print(res["length"], res["expanded"])
