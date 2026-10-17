/**
 * Forward chaining (aifn-compute/logic/resolution `leastHerbrandModel`): the least model of a family program equals the
 * answers SLD resolution gives, round by round as $T_P$ grows; the Herbrand base and error reporting.
 */
import { describe, expect, it } from 'vitest'
import { herbrandBaseSize, leastHerbrandModel, solveQuery } from 'aifn-compute/logic/resolution'
import { parseProgram } from 'aifn-compute/logic/terms'

const FAMILY = `
parent(ann, bob). parent(bob, cid). parent(bob, dee). parent(cid, eve).
grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
sibling(X, Y) :- parent(P, X), parent(P, Y), X \\= Y.
`

const sorted = (xs: readonly string[]) => [...xs].sort()

describe('leastHerbrandModel', () => {
  const m = leastHerbrandModel(parseProgram(FAMILY).clauses)

  it('reaches the fixpoint with 17 facts in 4 rounds (4, 9, 3, 1)', () => {
    expect(m.error).toBeNull()
    expect(m.fixpoint).toBe(true)
    expect(m.rounds).toBe(4)
    expect(m.facts).toHaveLength(17)
    const perRound = [1, 2, 3, 4].map((r) => m.facts.filter((f) => f.round === r).length)
    expect(perRound).toEqual([4, 9, 3, 1])
  })

  it.each(['grandparent(A, B)', 'ancestor(A, B)', 'sibling(A, B)'])('equals the SLD answers of %s', (q) => {
    const name = q.split('(')[0]
    const sld = solveQuery(FAMILY, q).solutions.map(
      (s) => `${name}(${s.bindings.map((b) => (b.value.kind === 'atom' ? b.value.name : '?')).join(', ')})`,
    )
    const fwd = m.facts.filter((f) => f.text.startsWith(name + '(')).map((f) => f.text)
    expect(sorted(fwd)).toEqual(sorted(sld))
  })

  it('records each derivation: premises are earlier facts matching the body', () => {
    const f = m.facts.find((x) => x.text === 'grandparent(ann, cid)')!
    expect(f.round).toBe(2)
    expect(f.premises.map((i) => m.facts[i].text)).toEqual(['parent(ann, bob)', 'parent(bob, cid)'])
  })

  it('closes a 4-cycle in 5 rounds with all 16 path atoms', () => {
    const r = leastHerbrandModel(
      parseProgram(`
        edge(a, b). edge(b, c). edge(c, d). edge(d, a).
        path(X, Y) :- edge(X, Y).
        path(X, Z) :- edge(X, Y), path(Y, Z).`).clauses,
    )
    expect(r.fixpoint).toBe(true)
    expect(r.rounds).toBe(5)
    expect(r.facts.filter((f) => f.text.startsWith('path(')).length).toBe(16)
  })

  it('reports an unsafe rule, a non-ground fact and a round limit instead of throwing', () => {
    expect(leastHerbrandModel(parseProgram('p(a). q(X, Y) :- p(X).').clauses).error).not.toBeNull()
    expect(leastHerbrandModel(parseProgram('p(X).').clauses).error).not.toBeNull()
    expect(leastHerbrandModel(parseProgram('p(a). q(X) :- p(X), \\+ r(X).').clauses).error).toMatch(/negation/)
    const limited = leastHerbrandModel(parseProgram(FAMILY).clauses, { maxRounds: 2 })
    expect(limited.fixpoint).toBe(false)
    expect(limited.error).toMatch(/no fixpoint/)
  })
})

describe('herbrandBaseSize', () => {
  it('counts 4 binary predicates over 5 constants: 100 atoms', () => {
    const hb = herbrandBaseSize(parseProgram(FAMILY).clauses)
    expect(hb.size).toBe(100)
    expect(sorted(hb.constants)).toEqual(['ann', 'bob', 'cid', 'dee', 'eve'])
    expect(hb.predicates).toHaveLength(4)
  })
})
