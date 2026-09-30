# Logic Lab: Logical Reasoning for Planning

Standard library only (Python 3). Central idea: **Logic + Search = Planning**.

```bash
python planner.py        # runs Tests A-E; saved copy in results/results.txt
```

| File | Purpose |
|---|---|
| `planner.py` | STRIPS-style planner: `applicable` and `apply_action` (logic), `bfs_plan` (search), independent plan validator, all tests |
| `planner.pl` | Prolog file for the optional Prolog tasks (6-8) |
| `results/results.txt` | Output of the Python tests |

---

## Task 0: The planning problem

**(a) Initial state** I = { At(Robot,A), At(Package,A) }
**(b) Goal** G = { At(Package,C) }
**(c) Actions:** Move(X,Y) for the connections A-B, B-A, B-C, C-B; PickUp(Package,L) and Drop(Package,L) for L in {A, B, C}.

**(d) Preconditions and effects**

| Action | Preconditions | Negative effects | Positive effects |
|---|---|---|---|
| Move(X,Y) | At(Robot,X) | At(Robot,X) | At(Robot,Y) |
| PickUp(Package,L) | At(Robot,L), At(Package,L) | At(Package,L) | Holding(Package) |
| Drop(Package,L) | At(Robot,L), Holding(Package) | Holding(Package) | At(Package,L) |

**Which actions are initially applicable?**
- **PickUp(Package,A): applicable.** Its preconditions At(Robot,A) and At(Package,A) are both in I.
- **Drop(Package,C): not applicable.** It needs At(Robot,C) (the robot is at A) and Holding(Package) (it is not holding anything), and neither holds in I.

**Think About It.** Being in the list of actions is not enough: applicability means *all* preconditions are satisfied in the current state, S |= Preconditions(a). That check is where logic enters the planner.

## Task 1: Plan constructed by hand

| State | Facts |
|---|---|
| S0 | At(Robot,A), At(Package,A) |
| S1 after PickUp(Package,A) | At(Robot,A), Holding(Package) |
| S2 after Move(A,B) | At(Robot,B), Holding(Package) |
| S3 after Move(B,C) | At(Robot,C), Holding(Package) |
| S4 after Drop(Package,C) | At(Robot,C), At(Package,C) |

S4 |= G, so this is a valid plan of 4 actions.

**Note on the sketch printed in the lab sheet.** The sheet suggests the reasoning Move(A,B), PickUp(Package,B), Move(B,C), Drop(Package,C). That is *not* a valid plan: after Move(A,B) the package is still at A, so PickUp(Package,B) has an unsatisfied precondition, At(Package,B). This is a good illustration of "a plan that looks reasonable is not necessarily valid". The planner rejects it (Test E below).

## Task 2: Prompt used with the LLM

```
I want to implement a simple planning agent in Python.
Represent a state as a set of logical propositions.
Each action should contain: a name; positive preconditions; negative preconditions;
positive effects; negative effects.
An action is applicable if all of its preconditions are satisfied by the current state.
When an action is applied: 1. remove its negative effects from the state; 2. add its
positive effects to the state.
Use breadth-first search to find a sequence of actions that achieves a specified goal.
The program should also: detect when no plan exists; print the resulting sequence of
actions; print the states reached after each action.
Explain the implementation and identify any assumptions you make.
```

**Think About It: where the specification appears in the program**

| Idea | Location |
|---|---|
| Preconditions ("when is an action applicable?") | `applicable(state, action)`: `action.pos_pre <= state and not (action.neg_pre & state)` |
| Effects ("how does the state change?") | `apply_action(state, action)`: `(state - neg_eff) | pos_eff`, i.e. remove first, then add |
| Goal ("when does planning terminate?") | `if goal <= nxt:` in `bfs_plan` (and the `goal <= initial` shortcut) |
| BFS ("how are alternatives explored?") | the `deque` frontier (FIFO) and the `parent` dict, which also prevents revisiting states |

**Assumptions.** Actions are deterministic; states are sets of ground atoms; the goal is a conjunction of positive atoms; atoms not in the state are false (closed-world assumption).

## Task 3: Test results

| Test | Initial state | Goal | Plan found? | Plan | Valid? |
|---|---|---|---|---|---|
| **A**: solvable | At(Robot,A), At(Package,A) | At(Package,C) | Yes | PickUp(Package,A) → Move(A,B) → Move(B,C) → Drop(Package,C) | Yes (independent validator agrees) |
| **B**: PickUp removed | same | At(Package,C) | **No plan found** | n/a | n/a (correctly no plan; nothing invented) |
| **C1**: only Move actions | same | At(Robot,C) | Yes | Move(A,B) → Move(B,C) | Yes; the robot reaches C, package stays at A |
| **C2**: only Move actions | same | At(Package,C) | **No plan found** | n/a | Correct: the planner does *not* treat "robot at C" as "package at C" |
| **D**: goal already true | At(Robot,C), At(Package,C) | At(Package,C) | Yes | empty plan | Yes |
| **E**: sketch from lab sheet | same as A | At(Package,C) | n/a | Move(A,B), PickUp(Package,B), … | **Invalid**: PickUp(Package,B) needs At(Package,B) |

"Independent validator" = `validate_plan_independently`, a separately written function that replays the plan using hard-coded domain rules and does not call `applicable` / `apply_action`. It agreed with the planner on every test.

## Task 4: Logic and search

The missing box is: **"Action is applicable? (S |= Preconditions(a)); if not, discard it."** The full chain is:

Current state → check action preconditions → *action is applicable* → generate successor state S' = Apply(S, a) → search over alternatives → goal?

In my own words: logic answers "which actions are possible here, and what will the world look like afterwards?"; search answers "of all the possible action sequences, which should I try next and when do I stop?". Logic supplies the state-transition rules, and BFS uses them to explore the space of reachable states until one satisfies the goal. Logic determines what is possible; search determines what to try.

## Task 5 (optional): Can the LLM verify its own plan?

**Which should I trust more: (a) the LLM's explanation or (b) the independently executed state transitions?** **(b).** A generated explanation is more text produced by the same process that produced the plan; it can be fluent and wrong (as the sheet's own sketch shows, a natural-sounding plan can contain an inapplicable step). Executing the transitions with code separate from the generator, and checking each precondition against the actual state, produces evidence that can fail. *A generated explanation is not the same as an independent verification.*

## Reflection questions

1. **Why specify preconditions and effects first?** They define the problem's semantics. With them written down, the LLM implements a known specification, and I can check the code against it (e.g. locate `applicable` and `apply_action`), rather than trusting whatever domain model it invents.
2. **Error if preconditions were not checked:** the planner could produce Move(A,C), teleporting the robot over a non-existent edge, or Drop(Package,C) while the robot is at A or is not holding the package, and "achieve" the goal with an impossible plan.
3. **Why is a plan that "looks reasonable" not necessarily valid?** Validity depends on every step's preconditions holding in the state reached at that step. The lab-sheet sketch reads naturally but fails at PickUp(Package,B) since the package is still at A.
4. **What did the LLM contribute?** The implementation: the action class, the applicability and effect functions, the BFS with parent pointers, and the state printing.
5. **What did I verify independently?** That preconditions are checked, that effects remove-then-add, that the plan reaches the goal (by replaying it in a separately written validator), that "no plan" is reported when PickUp is removed, that robot-at-C is not confused with package-at-C, and that the lab-sheet sketch is invalid.
6. **Where is logical reasoning used?** In deciding applicability (S |= Preconditions(a)), in computing successor states from effects, and in testing whether the goal holds (S |= G).
7. **How is planning related to search?** A planning problem is a search problem whose states are logical states, whose actions are the applicable operators, and whose goal test is S |= G. BFS here is the same algorithm as in the previous module; only the transition function is defined logically.

## Optional extension: Prolog as a logical verifier

`planner.pl` contains the facts and rules. **SWI-Prolog was not available in the environment where I prepared this, so the file has not been executed. The answers below are worked out by hand from the rules; please run the queries in `swipl` and record the actual output before you submit.**

### Task 6: Prolog as a plan verifier
- `?- can_move(a,b).` → `true` because the fact `connected(a,b)` exists and `can_move(X,Y) :- connected(X,Y)` matches it with X=a, Y=b.
- `?- can_move(a,c).` → `false`; there is no fact `connected(a,c)` and no other rule that could derive `can_move(a,c)`. (Prolog answers `false` for "cannot be proved from this knowledge base", not "impossible in the world".)
- The rule `can_move(X,Y) :- connected(X,Y).` corresponds to the implication Connected(X,Y) → CanMove(X,Y), universally quantified over X and Y. Prolog's `:-` reads "if", with the head as the conclusion.

### Task 7: Using Prolog to check a proposed plan
- `valid_move(a,b)` → true; `valid_move(b,c)` → true; `valid_move(a,c)` → false.
- **Challenge:** if the Python planner proposed Move(a,c), Prolog reports `valid_move(a,c)` as `false`: the warehouse knowledge does not support that action. I added `valid_path/2` to check a whole sequence: `?- valid_path(a,[b,c]).` is true and `?- valid_path(a,[c]).` is false.

### Task 8: Prolog and logical reasoning
The query `?- reduce_speed.` succeeds because: `wet_road` is a fact; `slippery :- wet_road.` lets Prolog derive `slippery`; `reduce_speed :- slippery.` lets it derive `reduce_speed`. As implications:

**wet_road ⇒ (wet_road → slippery) ⇒ (slippery → reduce_speed) ⇒ reduce_speed.**

### 7.2 Reflection
1. A **fact** is an unconditional statement (`connected(a,b).`); a **rule** has a head that holds *if* its body holds (`can_move(X,Y) :- connected(X,Y).`).
2. A **query** asks whether a statement follows from (can be proved from) the facts and rules; `true` means it can be derived, `false` means it cannot.
3. A Prolog program that encodes the domain independently can check the Python plan step by step against the stated warehouse knowledge, using different code and a different formalism from the generator.
4. An **independent verifier** does not share the generator's mistakes; if an LLM-generated plan uses an unsupported move, the verifier catches it. This is the *generate → independent verification* architecture.

## Reflection on LLM use

The LLM generated the planner from a precise specification. I did not accept it on the strength of its output: I mapped each specified idea to code (table above), wrote a separate validator with hard-coded domain rules, and ran six test cases, including the impossible-problem and irrelevant-action cases the lab asks for. The most instructive finding was that the plan sketched on the lab sheet fails the precondition check even though it reads plausibly. The LLM's contribution was implementation speed; the specification, the tests and the judgement about validity remain the engineer's responsibility.
