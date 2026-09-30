% planner.pl -- Prolog as an independent logical checker (optional part of the lab).
% Run with SWI-Prolog:   swipl planner.pl     then type the queries below.

% ---- Task 6: warehouse facts and rule ------------------------------------
connected(a,b).
connected(b,a).
connected(b,c).
connected(c,b).

can_move(X,Y) :-
    connected(X,Y).

% ?- can_move(a,b).   true.
% ?- can_move(a,c).   false.

% ---- Task 7: checking a proposed plan ------------------------------------
valid_move(X,Y) :-
    connected(X,Y).

% ?- valid_move(a,b).   true.
% ?- valid_move(b,c).   true.
% ?- valid_move(a,c).   false.   (the Python planner's Move(a,c) is NOT supported)

% Check a whole sequence of moves, threading the robot's position through it.
valid_path(_, []).
valid_path(From, [To|Rest]) :-
    valid_move(From, To),
    valid_path(To, Rest).

% ?- valid_path(a, [b,c]).   true.
% ?- valid_path(a, [c]).     false.

% ---- Task 8: chain of rules ----------------------------------------------
wet_road.

slippery :-
    wet_road.

reduce_speed :-
    slippery.

% ?- reduce_speed.   true.
% Reasoning: wet_road  =>  (wet_road -> slippery)  =>  (slippery -> reduce_speed)  =>  reduce_speed
