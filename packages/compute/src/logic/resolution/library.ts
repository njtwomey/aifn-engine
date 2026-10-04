/**
 * The library predicates every program can call, written in Prolog. A program's own definition of the same
 * name/arity replaces the library's. Solution orders match SWI-Prolog's `lists` and `between/3`.
 */
export const LIBRARY_SOURCE = `% member(X, L): X is an element of the list L.
member(X, [X|_]).
member(X, [_|T]) :- member(X, T).

% append(A, B, C): C is the list A followed by B.
append([], L, L).
append([H|T], L, [H|R]) :- append(T, L, R).

% select(X, L, R): R is L with one occurrence of X removed.
select(X, [X|T], T).
select(X, [H|T], [H|R]) :- select(X, T, R).

% permutation(L, P): P is a permutation of L.
permutation([], []).
permutation(L, [H|T]) :- select(H, L, R), permutation(R, T).

% reverse(L, R): R is L reversed (with an accumulator).
reverse(L, R) :- reverse(L, [], R).
reverse([], A, A).
reverse([H|T], A, R) :- reverse(T, [H|A], R).

% length(L, N): the list L has N elements.
length([], 0).
length([_|T], N) :- length(T, M), N is M + 1.

% last(L, X): X is the last element of L.
last([X], X).
last([_|T], X) :- last(T, X).

% sum_list(L, S): S is the sum of the numbers in L.
sum_list([], 0).
sum_list([H|T], S) :- sum_list(T, S0), S is S0 + H.

% between(L, H, X): X is an integer with L =< X =< H, counting up.
between(L, H, L) :- L =< H.
between(L, H, X) :- L < H, L1 is L + 1, between(L1, H, X).
`
