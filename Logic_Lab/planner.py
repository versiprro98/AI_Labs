"""STRIPS-style planner: logic (applicability + effects) + breadth-first search.

State  : frozenset of ground atoms, e.g. {"At(Robot,A)", "At(Package,A)"}
Action : name, positive/negative preconditions, positive/negative effects.
"""
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Action:
    name: str
    pos_pre: frozenset = frozenset()
    neg_pre: frozenset = frozenset()
    pos_eff: frozenset = frozenset()
    neg_eff: frozenset = frozenset()


def fs(*atoms):
    return frozenset(atoms)


# ---- logical component ------------------------------------------------------
def applicable(state, action):
    """S |= Preconditions(a): every positive precondition holds, no negative one does."""
    return action.pos_pre <= state and not (action.neg_pre & state)


def apply_action(state, action):
    """S' = Apply(S, a): 1) remove negative effects, 2) add positive effects."""
    return frozenset((state - action.neg_eff) | action.pos_eff)


# ---- search component -------------------------------------------------------
def bfs_plan(initial, goal, actions):
    """Return (plan, states) or (None, None). Goal is a set of atoms that must all hold."""
    initial = frozenset(initial)
    goal = frozenset(goal)
    if goal <= initial:
        return [], [initial]
    frontier = deque([initial])
    parent = {initial: None}                 # state -> (previous state, action)
    while frontier:
        state = frontier.popleft()
        for act in actions:
            if not applicable(state, act):   # logic decides what is possible
                continue
            nxt = apply_action(state, act)
            if nxt in parent:
                continue                     # already reached this state
            parent[nxt] = (state, act)
            if goal <= nxt:                  # S_n |= G
                plan, states = [], [nxt]
                cur = nxt
                while parent[cur] is not None:
                    prev, a = parent[cur]
                    plan.append(a)
                    states.append(prev)
                    cur = prev
                return plan[::-1], states[::-1]
            frontier.append(nxt)             # search decides what to try
    return None, None


# ---- the warehouse domain ---------------------------------------------------
LOCS = ["A", "B", "C"]
EDGES = [("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")]


def warehouse_actions(include_pickup=True, include_drop=True):
    acts = []
    for x, y in EDGES:
        acts.append(Action(f"Move({x},{y})",
                           pos_pre=fs(f"At(Robot,{x})"),
                           pos_eff=fs(f"At(Robot,{y})"),
                           neg_eff=fs(f"At(Robot,{x})")))
    for l in LOCS:
        if include_pickup:
            acts.append(Action(f"PickUp(Package,{l})",
                               pos_pre=fs(f"At(Robot,{l})", f"At(Package,{l})"),
                               pos_eff=fs("Holding(Package)"),
                               neg_eff=fs(f"At(Package,{l})")))
        if include_drop:
            acts.append(Action(f"Drop(Package,{l})",
                               pos_pre=fs(f"At(Robot,{l})", "Holding(Package)"),
                               pos_eff=fs(f"At(Package,{l})"),
                               neg_eff=fs("Holding(Package)")))
    return acts


INITIAL = fs("At(Robot,A)", "At(Package,A)")
GOAL = fs("At(Package,C)")


# ---- independent validator (does NOT use applicable/apply_action) ----------
def validate_plan_independently(plan_names, initial, goal):
    """Re-simulate the plan with hand-written domain rules; return (ok, explanation)."""
    robot = next(a[len("At(Robot,"):-1] for a in initial if a.startswith("At(Robot,"))
    pkg_at = next((a[len("At(Package,"):-1] for a in initial if a.startswith("At(Package,")), None)
    holding = "Holding(Package)" in initial
    edges = set(EDGES)
    for step in plan_names:
        kind, args = step.split("(")
        args = args.rstrip(")").split(",")
        if kind == "Move":
            x, y = args
            if robot != x or (x, y) not in edges:
                return False, f"{step}: robot is at {robot} / edge missing"
            robot = y
        elif kind == "PickUp":
            loc = args[1]
            if robot != loc or pkg_at != loc:
                return False, f"{step}: robot at {robot}, package at {pkg_at}"
            pkg_at, holding = None, True
        elif kind == "Drop":
            loc = args[1]
            if robot != loc or not holding:
                return False, f"{step}: robot at {robot}, holding={holding}"
            pkg_at, holding = loc, False
        else:
            return False, f"unknown action {step}"
    reached = f"At(Package,{pkg_at})" if pkg_at else None
    ok = all((g == reached) or (g == "Holding(Package)" and holding) or
             (g == f"At(Robot,{robot})") for g in goal)
    return ok, f"final: robot at {robot}, package at {pkg_at}, holding={holding}"


def fmt(state):
    return ", ".join(sorted(state))


def report(title, initial, goal, actions):
    print(f"\n=== {title} ===")
    print("Initial state:", fmt(initial))
    print("Goal         :", fmt(goal))
    plan, states = bfs_plan(initial, goal, actions)
    if plan is None:
        print("Plan found   : No plan found")
        return None, None
    names = [a.name for a in plan]
    print("Plan found   :", " -> ".join(names) if names else "(empty plan, goal already true)")
    for i, s in enumerate(states):
        label = "S0" if i == 0 else f"S{i} after {names[i-1]}"
        print(f"  {label}: {fmt(s)}")
    ok, why = validate_plan_independently(names, initial, goal)
    print(f"Independent validation: valid={ok} ({why})")
    return names, ok


if __name__ == "__main__":
    full = warehouse_actions()
    report("Test A: solvable warehouse problem", INITIAL, GOAL, full)
    report("Test B: PickUp removed (impossible)", INITIAL, GOAL, warehouse_actions(include_pickup=False))
    only_moves = warehouse_actions(include_pickup=False, include_drop=False)
    report("Test C1: only Move actions, goal = robot at C (robot-only goal)", INITIAL, fs("At(Robot,C)"), only_moves)
    report("Test C2: only Move actions, goal = package at C (must not confuse with robot at C)", INITIAL, GOAL, only_moves)
    report("Test D: goal already satisfied", fs("At(Robot,C)", "At(Package,C)"), GOAL, full)
    # The sequence hinted at on the lab sheet: Move(A,B), PickUp(Package,B), Move(B,C), Drop(Package,C)
    print("\n=== Test E: the plan sketched on the lab sheet ===")
    sketch = ["Move(A,B)", "PickUp(Package,B)", "Move(B,C)", "Drop(Package,C)"]
    by_name = {a.name: a for a in full}
    state = INITIAL
    for step in sketch:
        act = by_name[step]
        if not applicable(state, act):
            missing = sorted(act.pos_pre - state)
            print(f"{step}: NOT applicable in {{{fmt(state)}}}; missing preconditions: {missing}")
            break
        state = apply_action(state, act)
        print(f"{step}: ok -> {{{fmt(state)}}}")
    ok, why = validate_plan_independently(sketch, INITIAL, GOAL)
    print(f"Independent validation: valid={ok} ({why})")
