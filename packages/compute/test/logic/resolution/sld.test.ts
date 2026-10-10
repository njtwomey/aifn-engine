/**
 * The Prolog engine (aifn-compute/logic/resolution) on classic programs. Expected answers and their order are standard
 * Prolog's (SWI-Prolog 9 gives the same lists for these programs; SWI-Prolog is not installed here, so the answers
 * are written out by hand and checked against brute force where that is possible).
 */
import { describe, expect, it } from 'vitest'
import { trace } from 'aifn-compute/foundation/trace'
import { prologProgram, sldSteps, sldTree, solveQuery } from 'aifn-compute/logic/resolution'

const FAMILY = `
parent(tom, bob).  parent(tom, liz).  parent(bob, ann).
parent(bob, pat).  parent(pat, jim).
male(tom). male(bob). male(jim).
female(liz). female(ann). female(pat).
grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
sibling(X, Y) :- parent(P, X), parent(P, Y), X \\= Y.
`

const answers = (program: string, query: string, options = {}) => solveQuery(program, query, options).answers

describe('solveQuery: classic programs', () => {
  it('answers family queries in clause order', () => {
    expect(answers(FAMILY, 'grandparent(tom, X)')).toEqual(['X = ann', 'X = pat'])
    expect(answers(FAMILY, 'grandparent(X, Y)')).toEqual(['X = tom, Y = ann', 'X = tom, Y = pat', 'X = bob, Y = jim'])
    expect(answers(FAMILY, 'ancestor(tom, X)')).toEqual(['X = bob', 'X = liz', 'X = ann', 'X = pat', 'X = jim'])
    expect(answers(FAMILY, 'sibling(ann, X)')).toEqual(['X = pat'])
    expect(answers(FAMILY, 'ancestor(jim, X)')).toEqual([])
    expect(answers(FAMILY, 'parent(tom, bob)')).toEqual(['true'])
  })

  it('runs append, member and reverse (naive and accumulator)', () => {
    expect(answers('', 'append(X, Y, [1, 2])')).toEqual([
      'X = [], Y = [1, 2]',
      'X = [1], Y = [2]',
      'X = [1, 2], Y = []',
    ])
    expect(answers('', 'member(X, [a, b, c])')).toEqual(['X = a', 'X = b', 'X = c'])
    const naive = `
      nrev([], []).
      nrev([H|T], R) :- nrev(T, RT), append(RT, [H], R).
    `
    expect(answers(naive, 'nrev([1, 2, 3, 4], R)')).toEqual(['R = [4, 3, 2, 1]'])
    expect(answers('', 'reverse([1, 2, 3], R)')).toEqual(['R = [3, 2, 1]'])
    // Naive reverse makes O(n²) inferences, the accumulator O(n).
    const steps = (program: string, q: string) => solveQuery(program, q).steps
    const list = `[${Array.from({ length: 12 }, (_, i) => i).join(', ')}]`
    expect(steps(naive, `nrev(${list}, R)`)).toBeGreaterThan(3 * steps('', `reverse(${list}, R)`))
    expect(answers('', 'length([a, b, c], N)')).toEqual(['N = 3'])
    expect(answers('', 'between(1, 3, X)')).toEqual(['X = 1', 'X = 2', 'X = 3'])
    expect(answers('', 'permutation([1, 2, 3], P)')).toEqual([
      'P = [1, 2, 3]',
      'P = [1, 3, 2]',
      'P = [2, 1, 3]',
      'P = [2, 3, 1]',
      'P = [3, 1, 2]',
      'P = [3, 2, 1]',
    ])
  })

  it('solves n-queens (4 and 5) with the known solution counts and order', () => {
    const queens = `
      queens(N, Qs) :- numlist(1, N, Ns), permutation(Ns, Qs), safe(Qs).
      numlist(L, H, []) :- L > H.
      numlist(L, H, [L|T]) :- L =< H, L1 is L + 1, numlist(L1, H, T).
      safe([]).
      safe([Q|Qs]) :- no_attack(Q, Qs, 1), safe(Qs).
      no_attack(_, [], _).
      no_attack(Q, [Q1|Qs], D) :- Q =\\= Q1 + D, Q =\\= Q1 - D, D1 is D + 1, no_attack(Q, Qs, D1).
    `
    expect(answers(queens, 'queens(4, Qs)')).toEqual(['Qs = [2, 4, 1, 3]', 'Qs = [3, 1, 4, 2]'])
    // Brute force: the 10 solutions of 5 queens, in lexicographic order (permutation/2 enumerates in that order).
    const brute: string[] = []
    const perm = (rest: number[], acc: number[]) => {
      if (!rest.length) {
        if (acc.every((q, i) => acc.every((r, j) => j <= i || Math.abs(q - r) !== j - i)))
          brute.push(`Qs = [${acc.join(', ')}]`)
        return
      }
      rest.forEach((x, i) => perm([...rest.slice(0, i), ...rest.slice(i + 1)], [...acc, x]))
    }
    perm([1, 2, 3, 4, 5], [])
    expect(answers(queens, 'queens(5, Qs)')).toEqual(brute)
    expect(brute.length).toBe(10)
  })

  it('evaluates arithmetic and comparisons', () => {
    expect(answers('', 'X is 7 // 2 + 7 mod 3 * 2 - 2 ** 3')).toEqual(['X = -3'])
    expect(answers('', 'X is 7 / 2')).toEqual(['X = 3.5'])
    expect(answers('', 'X is -7 mod 3')).toEqual(['X = 2'])
    expect(answers('', '1 + 2 =:= 3, 2 < 3, 3 >= 3, 1 =\\= 2')).toEqual(['true'])
    expect(answers('', '2 > 3')).toEqual([])
    const fact = `
      fact(0, 1).
      fact(N, F) :- N > 0, N1 is N - 1, fact(N1, F1), F is N * F1.
    `
    expect(answers(fact, 'fact(6, F)')).toEqual(['F = 720'])
  })

  it('implements cut, negation as failure, if-then-else, disjunction and findall', () => {
    const max = `
      max(X, Y, X) :- X >= Y, !.
      max(_, Y, Y).
    `
    expect(answers(max, 'max(3, 2, M)')).toEqual(['M = 3'])
    expect(answers(max, 'max(2, 3, M)')).toEqual(['M = 3'])
    // Cut commits to the first member and removes the clauses after it.
    const first = `
      first(X, [X|_]) :- !.
      t(a). t(b). t(c).
      firstt(X) :- t(X), !.
    `
    expect(answers(first, 'firstt(X)')).toEqual(['X = a'])
    expect(answers(first, 't(X), !')).toEqual(['X = a'])
    // Cut is local to the clause: the outer member still backtracks.
    expect(answers(first, 'member(X, [1, 2]), firstt(Y)')).toEqual(['X = 1, Y = a', 'X = 2, Y = a'])
    expect(answers(FAMILY, '\\+ parent(jim, _)')).toEqual(['true'])
    expect(answers(FAMILY, 'male(X), \\+ parent(X, _)')).toEqual(['X = jim'])
    expect(answers('', '( 1 > 2 -> X = a ; X = b )')).toEqual(['X = b'])
    expect(answers('', '( member(X, [1, 2, 3]), X > 1 -> Y = X ; Y = none )')).toEqual(['X = 2, Y = 2'])
    expect(answers('', '( X = 1 ; X = 2 )')).toEqual(['X = 1', 'X = 2'])
    expect(answers(FAMILY, 'findall(C, parent(tom, C), L)')).toEqual(['L = [bob, liz]'])
    expect(answers(FAMILY, 'findall(X-Y, grandparent(X, Y), L), length(L, N)')[0]).toContain('N = 3')
    expect(answers('', 'findall(X, fail, L)')).toEqual(['L = []'])
  })

  it('stops at the step and depth limits with a message, and reports errors', () => {
    const loop = 'p(X) :- p(X).'
    const r = solveQuery(loop, 'p(a)', { maxSteps: 50 })
    expect(r.stopped).toBe('steps')
    expect(r.message).toMatch(/step limit of 50/)
    const d = solveQuery(loop, 'p(a)', { maxDepth: 20 })
    expect(d.stopped).toBe('exhausted')
    expect(d.depthLimited).toBe(true)
    expect(d.message).toMatch(/depth limit/)
    expect(solveQuery('', 'foo(1)').message).toMatch(/unknown procedure foo\/1/)
    expect(solveQuery('', 'X is Y + 1').message).toMatch(/not sufficiently instantiated/)
    expect(solveQuery('', 'member(X, [a, b, c])', { maxSolutions: 2 }).answers).toEqual(['X = a', 'X = b'])
  })

  it('raises division by zero for /, //, mod and rem alike', () => {
    for (const op of ['/', '//', 'mod', 'rem'])
      expect(solveQuery('', `X is 7 ${op} 0`).message, op).toMatch(/division by zero/)
    expect(answers('', 'X is -7 rem 2, Y is -7 mod 2')).toEqual(['X = -1, Y = 1'])
  })
})

describe('sldSteps: the search tree', () => {
  it('records the SLD tree with successes, failures and cut-pruned branches', () => {
    const program = prologProgram(`
      t(a). t(b). t(c).
      firstt(X) :- t(X), !.
      q(X) :- t(X), X \\= a.
    `)
    const run = trace(sldSteps(program, 'firstt(X)'), undefined, 100, { keep: 'all' })
    const tree = sldTree(run.steps)
    expect(tree[0].note).toBe('the query')
    expect(tree.filter((n) => n.status === 'success').length).toBe(1)
    // t(b) and t(c) were never tried: the cut removed them.
    const pruned = tree.filter((n) => n.status === 'pruned')
    expect(pruned.map((n) => n.clause)).toEqual([1, 2])
    const final = run.steps[run.steps.length - 1]
    expect(final.stopped).toBe('exhausted')

    const q = trace(sldSteps(program, 'q(X)'), undefined, 100, { keep: 'all' })
    const qt = sldTree(q.steps)
    expect(qt.filter((n) => n.status === 'failure').length).toBe(1)
    expect(qt.filter((n) => n.status === 'success').length).toBe(2)
    // Every node but the root has a parent created before it.
    for (const n of qt) if (n.parent !== null) expect(qt[n.parent].created).toBeLessThanOrEqual(n.created)
    expect(q.steps.at(-1)!.solutions.length).toBe(2)
  })

  it('shows each resolution step with its clause and unifier', () => {
    const program = prologProgram('p(X, b) :- q(X).  q(a).')
    const run = trace(sldSteps(program, 'p(Z, W)'), undefined, 20, { keep: 'all' })
    const first = run.steps[1].event
    expect(first.kind).toBe('resolve')
    if (first.kind === 'resolve') {
      expect(first.clause).toBe(0)
      expect(first.unifier.length).toBe(2)
    }
    expect(run.steps.at(-1)!.solutions[0].bindings.map((b) => b.name)).toEqual(['Z', 'W'])
  })
})

// Einstein's (zebra) puzzle: a classic search of ~39k steps with a unique answer (the German owns the fish).
const EINSTEIN = `% Einstein's puzzle. Five houses in a row, each with a colour, an owner's
% nationality, a drink, a cigar brand and a pet: h(Colour, Nation, Drink, Smoke, Pet).
% Who owns the fish?
right_of(X, Y, [Y, X | _]).
right_of(X, Y, [_ | T]) :- right_of(X, Y, T).
next_to(X, Y, L) :- right_of(X, Y, L).
next_to(X, Y, L) :- right_of(Y, X, L).

houses(Hs) :-
    Hs = [h(_, norwegian, _, _, _), _, h(_, _, milk, _, _), _, _],  % clues 9 and 8
    member(h(red, brit, _, _, _), Hs),                              % 1
    member(h(_, swede, _, _, dog), Hs),                             % 2
    member(h(_, dane, tea, _, _), Hs),                              % 3
    right_of(h(white, _, _, _, _), h(green, _, _, _, _), Hs),       % 4: green just left of white
    member(h(green, _, coffee, _, _), Hs),                          % 5
    member(h(_, _, _, pallmall, birds), Hs),                        % 6
    member(h(yellow, _, _, dunhill, _), Hs),                        % 7
    next_to(h(_, _, _, blends, _), h(_, _, _, _, cats), Hs),        % 10
    next_to(h(_, _, _, _, horse), h(_, _, _, dunhill, _), Hs),      % 11
    member(h(_, _, beer, bluemasters, _), Hs),                      % 12
    member(h(_, german, _, prince, _), Hs),                         % 13
    next_to(h(_, norwegian, _, _, _), h(blue, _, _, _, _), Hs),     % 14
    next_to(h(_, _, _, blends, _), h(_, _, water, _, _), Hs),       % 15
    member(h(_, _, _, _, fish), Hs).

fish_owner(Who) :- houses(Hs), member(h(_, Who, _, _, fish), Hs).
`

describe("Einstein's puzzle", () => {
  it('has one solution, in which the German owns the fish', () => {
    const r = solveQuery(EINSTEIN, 'fish_owner(Who)', { maxSteps: 400_000, maxDepth: 2000 })
    expect(r.stopped).toBe('exhausted')
    expect(r.answers).toEqual(['Who = german'])
  })
  it('places every house', () => {
    const r = solveQuery(EINSTEIN, 'houses(Hs)', { maxSteps: 400_000, maxDepth: 2000 })
    expect(r.answers).toEqual([
      'Hs = [h(yellow, norwegian, water, dunhill, cats), h(blue, dane, tea, blends, horse), h(red, brit, milk, pallmall, birds), h(green, german, coffee, prince, fish), h(white, swede, beer, bluemasters, dog)]',
    ])
  })
})
