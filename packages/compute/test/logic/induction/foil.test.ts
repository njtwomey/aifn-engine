/**
 * FOIL and Plotkin's generality (aifn-compute/logic/induction) on classic tasks: `daughter` (Lavrač and Džeroski 1994, §2),
 * `grandparent`, and recursive `ancestor` from a family tree. Each learned program is run on the Prolog engine and
 * must give exactly the target relation.
 */
import { describe, expect, it } from 'vitest'
import { trace } from 'aifn-compute/foundation/trace'
import { solveQuery } from 'aifn-compute/logic/resolution'
import {
  clauseLgg,
  foil,
  foilGain,
  foilProblem,
  foilRefinements,
  foilSteps,
  reduceClause,
  termLgg,
  thetaSubsumes,
} from 'aifn-compute/logic/induction'
import { clauseToString, parseProgram, parseTerm, termToString } from 'aifn-compute/logic/terms'

const PARENTS = `
parent(ann, mary). parent(ann, tom). parent(tom, eve). parent(tom, ian).
female(ann). female(mary). female(eve).
`

// A three-generation family: 10 people.
const FAMILY = `
parent(george, bob). parent(george, liz). parent(mum, bob). parent(mum, liz).
parent(bob, ann). parent(bob, pat). parent(liz, kim). parent(pat, jim). parent(pat, sue). parent(kim, tim).
`

/** The pairs (x, y) a program makes `pred(x, y)` true for, sorted. */
function relation(program: string, pred: string): string[] {
  return [...new Set(solveQuery(program, `${pred}(X, Y)`).answers)].sort()
}

describe('foil', () => {
  it('computes FOIL gain', () => {
    expect(foilGain(2, 2, 2, 1, 2)).toBeCloseTo(2 * (Math.log2(2 / 3) - Math.log2(1 / 2)), 12)
    expect(foilGain(3, 3, 0, 1, 0)).toBe(-Infinity)
  })

  it('learns daughter(X, Y) :- female(X), parent(Y, X) (Lavrač and Džeroski)', () => {
    // With only the two given negatives, parent(C, B) ("B has a parent") has the highest gain: it covers one positive
    // and no negative, so FOIL needs two clauses. Under the closed world it finds the textbook definition.
    const given = foil(
      foilProblem(PARENTS, 'daughter(mary, ann). daughter(eve, tom).', 'daughter(tom, ann). daughter(eve, ann).'),
    )
    expect(given.clauses[0]).toBe('daughter(A, B) :- parent(C, B).')
    expect(given.uncovered).toEqual([])
    const problem = foilProblem(PARENTS, 'daughter(mary, ann). daughter(eve, tom).')
    const result = foil(problem)
    expect(result.clauses.length).toBe(1)
    const body = parseProgram(result.program)
      .clauses[0].body.map((g) => termToString(g))
      .sort()
    expect(body).toEqual(['female(A)', 'parent(B, A)'])
    expect(result.uncovered).toEqual([])
  })

  it('learns grandparent from a family tree under the closed world', () => {
    const truth = relation(FAMILY + 'gp(X, Z) :- parent(X, Y), parent(Y, Z).', 'gp')
    const positives = truth.map((a) => a.replace(/X = (\w+), Y = (\w+)/, 'grandparent($1, $2).')).join(' ')
    const result = foil(foilProblem(FAMILY, positives))
    expect(result.clauses).toEqual(['grandparent(A, B) :- parent(A, C), parent(C, B).'])
    expect(relation(FAMILY + result.program, 'grandparent')).toEqual(truth)
  })

  it('learns the recursive ancestor relation', () => {
    const truth = relation(FAMILY + 'anc(X, Y) :- parent(X, Y).\nanc(X, Y) :- parent(X, Z), anc(Z, Y).', 'anc')
    const positives = truth.map((a) => a.replace(/X = (\w+), Y = (\w+)/, 'ancestor($1, $2).')).join(' ')
    const result = foil(foilProblem(FAMILY, positives))
    expect(result.uncovered).toEqual([])
    // The learned program, run by the Prolog engine, gives exactly the ancestor relation.
    expect(relation(FAMILY + result.program, 'ancestor')).toEqual(truth)
    expect(result.clauses).toEqual([
      'ancestor(A, B) :- parent(A, B).',
      'ancestor(A, B) :- parent(A, C), ancestor(C, B).',
    ])
  })

  it('steps: each step adds the best literal or commits a clause, with every candidate scored', () => {
    const problem = foilProblem(PARENTS, 'daughter(mary, ann). daughter(eve, tom).')
    const run = trace(foilSteps(problem), undefined, 20, { keep: 'all' })
    const states = run.steps
    expect(states[0].currentText).toBe('daughter(A, B).')
    expect(states[0].coveredNegatives.length).toBe(23)
    const s1 = states[1]
    expect(s1.event.kind).toBe('literal')
    expect(s1.candidates.length).toBeGreaterThan(5)
    expect(s1.candidates[0].gain).toBeGreaterThanOrEqual(s1.candidates[s1.candidates.length - 1].gain)
    const last = states[states.length - 1]
    expect(last.terminated).toBe(true)
    expect(last.clauses.length).toBe(1)
    // The refinement operator generates each literal once.
    const root = { body: [], variables: 2, pos: [], neg: [], p0: 0, n0: 0, t: 0, gain: 0 }
    const texts = foilRefinements(problem, root).map((n) => JSON.stringify(n.body))
    expect(new Set(texts).size).toBe(texts.length)
  })

  it('reports when no literal helps', () => {
    // Positives and negatives indistinguishable from the background.
    const result = foil(foilProblem('likes(a, b). likes(b, a).', 'friend(a, b).', 'friend(b, a).'))
    expect(result.uncovered).toEqual([0])
    expect(result.stopped).toMatch(/no literal has positive gain/)
  })
})

describe('generality', () => {
  const clause = (s: string) => parseProgram(s).clauses[0]
  it('computes the lgg of terms (Plotkin)', () => {
    expect(termToString(termLgg(parseTerm('f(a, g(a))'), parseTerm('f(b, g(b))')))).toBe('f(A, g(A))')
    expect(termToString(termLgg(parseTerm('p(a, b, a)'), parseTerm('p(c, d, c)')))).toBe('p(A, B, A)')
    expect(termToString(termLgg(parseTerm('[1, 2]'), parseTerm('[3]')))).toBe('[A|B]')
  })
  it('decides θ-subsumption', () => {
    expect(thetaSubsumes(clause('p(X) :- q(X, Y).'), clause('p(a) :- q(a, b), r(b).'))).toBe(true)
    expect(thetaSubsumes(clause('p(a) :- q(a, b), r(b).'), clause('p(X) :- q(X, Y).'))).toBe(false)
    expect(thetaSubsumes(clause('p(X) :- q(X, X).'), clause('p(a) :- q(a, b).'))).toBe(false)
    expect(thetaSubsumes(clause('p(X) :- q(X, Y), q(Y, X).'), clause('p(a) :- q(a, a).'))).toBe(true)
  })
  it('generalises two clauses to one that θ-subsumes both, and reduces it', () => {
    const c = clause('daughter(mary, ann) :- female(mary), parent(ann, mary), parent(ann, tom).')
    const d = clause('daughter(eve, tom) :- female(eve), parent(tom, eve), parent(tom, ian).')
    const g = clauseLgg(c, d)!
    expect(thetaSubsumes(g, c)).toBe(true)
    expect(thetaSubsumes(g, d)).toBe(true)
    const r = reduceClause(g)
    expect(thetaSubsumes(r, g) && thetaSubsumes(g, r)).toBe(true)
    expect(clauseToString(r)).toBe('daughter(A, B) :- female(A), parent(B, A).')
  })
})
