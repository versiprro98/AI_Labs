"""Runs Tasks 3, 5 and 6 of the search lab and writes results/results.txt."""
import random
from search import (Grid, WAREHOUSE, astar, bfs, manhattan, euclidean, zero,
                    double_manhattan, path_is_valid, true_shortest_length)

log = []
def out(s=""):
    print(s)
    log.append(s)

def show(name, grid, res):
    out(f"{name}: found={res['found']}, length={res['length']}, expanded={res['expanded']}, "
        f"valid={path_is_valid(grid, res['path']) if res['found'] else 'n/a'}")

# ---- Task 3, Test 1: original warehouse -----------------------------------
g = Grid(WAREHOUSE)
res = astar(g)
out("=== Test 1: original warehouse (A*, Manhattan) ===")
out(g.render(res["path"]))
out("Path (row,col): " + " -> ".join(map(str, res["path"])))
show("A*", g, res)
truth = true_shortest_length(g)
out(f"Independent flood-fill shortest length = {truth}; A* optimal: {res['length'] == truth}")

# ---- Test 2: trivial ------------------------------------------------------
t2 = Grid("#####\n#SG##\n#####")
r2 = astar(t2)
out("\n=== Test 2: goal adjacent to start ===")
show("A*", t2, r2)
out(f"Path: {r2['path']}")

# ---- Test 3: no solution --------------------------------------------------
# lab map: the goal is walled in
t3 = Grid("#######\n#S....#\n###.###\n#...#G#\n#######")
r3 = astar(t3)
b3 = bfs(t3)
out("\n=== Test 3: no solution ===")
show("A*", t3, r3)
show("BFS", t3, b3)
out(f"Independent ground truth: {true_shortest_length(t3)} (None = unreachable)")

# ---- Test 4: alternative paths -------------------------------------------
t4 = Grid("#######\n#S...##\n#.##.##\n#....G#\n#######")
r4 = astar(t4)
out("\n=== Test 4: two or more S->G paths ===")
out(t4.render(r4["path"]))
show("A*", t4, r4)
out(f"Independent shortest length = {true_shortest_length(t4)}; A* returned a shortest path: "
    f"{r4['length'] == true_shortest_length(t4)}")

# ---- extra: randomised property test --------------------------------------
random.seed(0)
bad, n_solvable = 0, 0
for _ in range(300):
    h, w = 8, 12
    cells = [["#" if random.random() < 0.28 else "." for _ in range(w)] for _ in range(h)]
    cells[0][0], cells[h - 1][w - 1] = "S", "G"
    grid = Grid("\n".join("".join(r) for r in cells))
    tl = true_shortest_length(grid)
    ra, rb = astar(grid), bfs(grid)
    if tl is None:
        bad += (ra["found"] or rb["found"])
    else:
        n_solvable += 1
        bad += not (ra["found"] and rb["found"] and ra["length"] == tl == rb["length"]
                    and path_is_valid(grid, ra["path"]))
out(f"\n=== Randomised test: 300 random 8x12 maps ({n_solvable} solvable) ===")
out(f"Disagreements with independent ground truth (A* and BFS): {bad}")

# ---- Task 5: BFS vs A* ----------------------------------------------------
ra, rb = astar(g), bfs(g)
out("\n=== Task 5: BFS vs A* on the original warehouse ===")
out("| Measure | BFS | A* |")
out("|---|---|---|")
out(f"| Solution found | {rb['found']} | {ra['found']} |")
out(f"| Path length | {rb['length']} | {ra['length']} |")
out(f"| States expanded | {rb['expanded']} | {ra['expanded']} |")
free = sum(ch != "#" for row in g.rows for ch in row)
out(f"(free cells in the map: {free})")

# ---- supplementary: an open room, where a heuristic can help --------------
rows = ["#" * 22]
for i in range(1, 11):
    line = ["#"] + ["."] * 20 + ["#"]
    if 1 <= i <= 8:
        line[11] = "#"          # vertical shelf with a gap at the bottom
    rows.append("".join(line))
rows.append("#" * 22)
rows[1] = rows[1][:1] + "S" + rows[1][2:]
rows[10] = rows[10][:20] + "G" + rows[10][21:]
open_room = Grid("\n".join(rows))
ro_a, ro_b = astar(open_room), bfs(open_room)
tro = true_shortest_length(open_room)
out("\n=== Supplementary: 10 x 20 open room with one shelf (S top-left, G bottom-right) ===")
out(open_room.render(ro_a["path"]))
out("| Measure | BFS | A* (Manhattan) |")
out("|---|---|---|")
out(f"| Solution found | {ro_b['found']} | {ro_a['found']} |")
out(f"| Path length | {ro_b['length']} | {ro_a['length']} |")
out(f"| States expanded | {ro_b['expanded']} | {ro_a['expanded']} |")
ro_fifo = astar(open_room, manhattan, tie_break="fifo")
out(f"A* with FIFO tie-breaking (ties in f broken by insertion order): expanded={ro_fifo['expanded']}, length={ro_fifo['length']}")
out(f"(true shortest = {tro}; free cells = {sum(ch != '#' for row in open_room.rows for ch in row)})")

# ---- Task 6: heuristics ---------------------------------------------------
out("\n=== Task 6: heuristic investigation (original warehouse) ===")
out("| Heuristic | Found | Path length | Expanded | Optimal? |")
out("|---|---|---|---|---|")
for name, hfun in [("Manhattan", manhattan), ("h = 0 (Dijkstra/UCS)", zero),
                   ("Euclidean", euclidean), ("2 x Manhattan", double_manhattan)]:
    r = astar(g, hfun)
    out(f"| {name} | {r['found']} | {r['length']} | {r['expanded']} | {r['length'] == truth} |")

out("\nSame four heuristics on the open room (true shortest = %d):" % tro)
out("| Heuristic | Path length | Expanded | Optimal? |")
out("|---|---|---|---|")
for name, hfun in [("Manhattan", manhattan), ("h = 0", zero), ("Euclidean", euclidean),
                   ("2 x Manhattan", double_manhattan)]:
    r = astar(open_room, hfun)
    out(f"| {name} | {r['length']} | {r['expanded']} | {r['length'] == tro} |")

# heuristic overestimation on a map where it matters
tricky = Grid("""\
#########
#S......#
#.#####.#
#.#...#.#
#.#.#.#.#
#...#..G#
#########""")
tt = true_shortest_length(tricky)
out(f"\nSecond map designed so that overestimating hurts (true shortest = {tt}):")
out("| Heuristic | Path length | Expanded | Optimal? |")
out("|---|---|---|---|")
for name, hfun in [("Manhattan", manhattan), ("h = 0", zero), ("Euclidean", euclidean),
                   ("2 x Manhattan", double_manhattan), ("5 x Manhattan", lambda a, b: 5 * manhattan(a, b))]:
    r = astar(tricky, hfun)
    out(f"| {name} | {r['length']} | {r['expanded']} | {r['length'] == tt} |")

with open("results/results.txt", "w") as f:
    f.write("\n".join(log) + "\n")
