/**
 * Plotkin's generality traced (aifn-compute/logic/induction `generalise.ts`): the subsumption search state by state
 * against `thetaSubsumes`, the lgg of terms comparison by comparison, the clause lgg pair by pair, and Golem's relative
 * lgg.
 */
import { describe, expect, it } from 'vitest'
import {
  clauseLgg,
  clauseLggTrace,
  headLinked,
  predicateOf,
  reduceClause,
  relativeLgg,
  renameApart,
  targetLiterals,
  termLgg,
  termLggTraced,
  thetaSubsumes,
  thetaSubsumption,
  thetaSubsumptionTrace,
  type HornClause,
  type LggStep,
} from 'aifn-compute/logic/induction'
import {
  clauseToString,
  compound,
  parseProgram,
  parseTerm,
  termToString,
  termVariables,
  variable,
  atom,
} from 'aifn-compute/logic/terms'

const clause = (s: string): HornClause => {
  const c = parseProgram(s).clauses[0]
  return { head: c.head, body: c.body }
}
const last = <T>(xs: readonly T[]) => xs[xs.length - 1]

const PAIRS: [string, string][] = [
  [
    'grandparent(X, Z) :- parent(X, Y), parent(Y, Z).',
    'grandparent(ann, dee) :- parent(ann, bob), parent(bob, dee), female(dee).',
  ],
  ['p(X) :- q(X, Y), q(X, Z).', 'p(X) :- q(X, Y).'],
  ['p(X) :- q(X, Y).', 'p(X) :- q(X, Y), q(X, Z).'],
  ['p(X) :- q(X, Y), r(Y).', 'p(a) :- q(a, b), q(a, c), r(c).'],
  ['p(X) :- q(X, Y), r(Y).', 'p(a) :- q(a, b), q(a, c), r(d).'],
  ['p(f(X)) :- p(X).', 'p(f(f(Y))) :- p(Y).'],
  ['p(X, Y) :- q(X, Y), q(Y, X).', 'p(a, b) :- q(a, b), q(b, a).'],
  ['p(X, X).', 'p(a, b).'],
  ['s(X) :- t(X).', 'p(a) :- t(a).'],
  ['daughter(X, Y) :- female(X), parent(Y, X).', 'daughter(X, Y) :- parent(Y, X), female(X), female(Y).'],
]

describe('thetaSubsumptionTrace', () => {
  it.each(PAIRS)('ends as thetaSubsumes says, in either order: %s onto %s', (a, b) => {
    const c = clause(a)
    const d = clause(b)
    const expected = thetaSubsumes(c, d)
    expect(last(thetaSubsumptionTrace(c, d)).event).toBe(expected ? 'subsumes' : 'fails')
    expect(last(thetaSubsumptionTrace(c, d, { order: 'constrained' })).event).toBe(expected ? 'subsumes' : 'fails')
  })

  it('gives the known answers', () => {
    expect(PAIRS.map(([a, b]) => thetaSubsumes(clause(a), clause(b)))).toEqual([
      true,
      true,
      true,
      true,
      false,
      false,
      true,
      false,
      false,
      true,
    ])
  })

  it('pair 4 clashes and backtracks', () => {
    const steps = thetaSubsumptionTrace(clause(PAIRS[3][0]), clause(PAIRS[3][1]))
    expect(steps.map((s) => s.event)).toEqual([
      'start',
      'match',
      'match',
      'clash',
      'backtrack',
      'match',
      'match',
      'subsumes',
    ])
    expect(last(steps).assignment).toEqual([0, 2, 3])
    expect(steps[3].note).toBe('r(Y) does not match r(c) under the bindings so far.')
  })

  it('a triangle fails to map into a symmetric 4-cycle in 228 states, and maps into a 3-cycle', () => {
    const tri = clause('cyclic :- e(X, Y), e(Y, Z), e(Z, X).')
    const four = clause('cyclic :- e(a, b), e(b, a), e(b, c), e(c, b), e(c, d), e(d, c), e(d, a), e(a, d).')
    const steps = thetaSubsumptionTrace(tri, four)
    expect(steps).toHaveLength(228)
    expect(last(steps).event).toBe('fails')
    const three = thetaSubsumptionTrace(tri, clause('cyclic :- e(a, b), e(b, c), e(c, a).'))
    expect(three).toHaveLength(9)
    expect(last(three).event).toBe('subsumes')
  })

  it('stops at the step limit with a failure', () => {
    const tri = clause('cyclic :- e(X, Y), e(Y, Z), e(Z, X).')
    const four = clause('cyclic :- e(a, b), e(b, a), e(b, c), e(c, b), e(c, d), e(d, c), e(d, a), e(a, d).')
    const steps = thetaSubsumptionTrace(tri, four, { maxSteps: 50 })
    expect(steps).toHaveLength(51)
    expect(last(steps).note).toBe('Stopped after 50 steps.')
  })

  it('renames D apart and numbers the targets from the head', () => {
    const c = clause('p(X) :- q(X, Y).')
    const d = clause('p(X) :- q(X, X).')
    const ids = termVariables(...targetLiterals(c, d)).map((v) => v.id)
    expect(Math.min(...ids)).toBeGreaterThan(1)
    expect(termToString(renameApart(c, d).head)).toBe('p(X)')
    expect(predicateOf(parseTerm('q(a, b)'))).toBe('q/2')
    expect(predicateOf(parseTerm('cyclic'))).toBe('cyclic/0')
    const theta = thetaSubsumption(c, clause('p(a) :- q(a, b), r(b).'))!
    expect(theta.map((b) => `${b.variable.name} = ${termToString(b.value)}`)).toEqual(['X = a', 'Y = b'])
  })
})

describe('the lgg traced', () => {
  it('records each comparison', () => {
    const steps: LggStep[] = []
    const g = termLggTraced(parseTerm('p(f(a, b), g(a), b)'), parseTerm('p(f(c, d), g(c), e)'), undefined, steps)
    expect(termToString(g)).toBe('p(f(A, B), g(A), C)')
    expect(steps.map((s) => s.action)).toEqual(['descend', 'descend', 'new', 'new', 'descend', 'reuse', 'new'])
    expect(steps[5].path).toEqual([1, 0])
  })

  it('gives the classic term lggs', () => {
    const lgg = (a: string, b: string) => termToString(termLgg(parseTerm(a), parseTerm(b)))
    expect(lgg('f(a, g(a))', 'f(b, g(b))')).toBe('f(A, g(A))')
    expect(lgg('f(a, b)', 'f(b, a)')).toBe('f(A, B)')
    expect(lgg('parent(ann, bob)', 'parent(cid, fay)')).toBe('parent(A, B)')
    expect(lgg('[1, 2, 3]', '[4, 2, 5]')).toBe(termToString(parseTerm('[A, 2, B]')))
  })

  it('keeps a variable paired with itself, with new variables numbered above it', () => {
    const X = variable('X', 0)
    const g = termLgg(compound('f', [X, atom('a')]), compound('f', [X, atom('b')]))
    expect(termToString(g)).toBe('f(X, A)')
    const [x, a] = termVariables(g)
    expect(x.id).toBe(0)
    expect(a.id).toBe(1)
  })

  it('generalises clauses pair by pair, as clauseLgg', () => {
    const c = clause('daughter(mary, ann) :- female(mary), female(ann), parent(ann, mary).')
    const d = clause('daughter(eve, tom) :- female(eve), parent(tom, eve).')
    const out = clauseLggTrace(c, d)!
    expect(out.steps).toHaveLength(4)
    expect(clauseToString(out.clause)).toBe('daughter(A, B) :- female(A), female(C), parent(B, A).')
    expect(clauseToString(reduceClause(out.clause))).toBe('daughter(A, B) :- female(A), parent(B, A).')
    expect(clauseToString(clauseLgg(c, d)!)).toBe(clauseToString(out.clause))
    expect(clauseLggTrace(c, clause('son(a, b).'))).toBeNull()
  })
})

describe('relativeLgg', () => {
  it("finds Golem's daughter clause: 25 raw literals, 7 ground, 13 unlinked, 5 linked, 2 after reduction", () => {
    const B = parseProgram(
      'parent(ann, mary). parent(ann, tom). parent(tom, eve). parent(tom, ian). female(ann). female(mary). female(eve).',
    ).clauses.map((c) => c.head)
    const r = relativeLgg(parseTerm('daughter(mary, ann)'), parseTerm('daughter(eve, tom)'), B)!
    expect(r.raw.body).toHaveLength(25)
    expect(r.ground).toHaveLength(7)
    expect(r.unlinked).toHaveLength(13)
    expect(r.linked.body).toHaveLength(5)
    expect(r.steps).toHaveLength(26)
    expect(clauseToString(r.clause)).toBe('daughter(A, B) :- parent(B, A), female(A).')
  })

  it('links literals to the head through shared variables', () => {
    const { linked, unlinked } = headLinked(clause('p(X) :- q(X, Y), r(Y), s(Z).'))
    expect(linked.map((t) => termToString(t))).toEqual(['q(X, Y)', 'r(Y)'])
    expect(unlinked.map((t) => termToString(t))).toEqual(['s(Z)'])
  })
})
