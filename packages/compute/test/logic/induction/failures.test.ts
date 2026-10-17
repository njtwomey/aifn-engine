/**
 * Learning from failures (aifn-compute/logic/induction `learnFromFailures`): last/2 and ancestor/2, against plain
 * generate-and-test; the constraints are sound and the solution is the smallest.
 */
import { describe, expect, it } from 'vitest'
import { DomainError } from 'aifn-compute/foundation/errors'
import { prologProgram, solveQuery } from 'aifn-compute/logic/resolution'
import {
  learnFromFailures,
  lffClauses,
  lffPrograms,
  lffSubsumption,
  type LffBias,
  type LffProblem,
  type LffResult,
} from 'aifn-compute/logic/induction'

const LAST: LffProblem = {
  background: 'head([H|_], H). tail([_|T], T). empty([]).',
  target: { name: 'last', arity: 2, types: ['list', 'element'] },
  body: [
    { name: 'head', arity: 2, types: ['list', 'element'] },
    { name: 'tail', arity: 2, types: ['list', 'list'] },
    { name: 'empty', arity: 1, types: ['list'] },
  ],
  positives: ['last([a, b, c], c)', 'last([b, a], a)', 'last([c], c)', 'last([a, c, b, a], a)'],
  negatives: ['last([a, b, c], a)', 'last([a, b, c], b)', 'last([b, a], b)', 'last([a, c, b, a], c)'],
}
const LAST_BIAS: LffBias = { maxVars: 3, maxBody: 3, maxClauses: 2 }
const ANCESTOR: LffProblem = {
  background:
    'parent(ann, bob). parent(ann, cid). parent(bob, dee). parent(bob, eve). parent(cid, fay). parent(eve, gus).',
  target: { name: 'ancestor', arity: 2 },
  body: [{ name: 'parent', arity: 2 }],
  positives: ['ancestor(ann, bob)', 'ancestor(ann, dee)', 'ancestor(ann, gus)', 'ancestor(cid, fay)'],
  negatives: ['ancestor(bob, ann)', 'ancestor(dee, bob)', 'ancestor(cid, gus)'],
}
const OFF = { generalisation: false, specialisation: false, elimination: false, maxTests: 5000 }

const solutionText = (r: LffResult) => r.programs[r.solution].clauses.map((i) => r.clauses[i].text)

describe('learnFromFailures', () => {
  const popper = learnFromFailures(LAST, LAST_BIAS)

  it('last/2: 57 clauses, 1220 programs; a 7-literal solution after 55 tests against 255', () => {
    expect(popper.clauses).toHaveLength(57)
    expect(popper.programs).toHaveLength(1220)
    expect(popper.programs[popper.solution].size).toBe(7)
    expect(popper.steps).toHaveLength(55)
    expect(solutionText(popper)).toEqual([
      'last(A, B) :- tail(A, C), last(C, B).',
      'last(A, B) :- head(A, B), tail(A, C), empty(C).',
    ])
    const plain = learnFromFailures(LAST, LAST_BIAS, OFF)
    expect(plain.steps).toHaveLength(255)
    expect(plain.programs[plain.solution].size).toBe(7)
    expect(popper.enumerateTests).toBe(255)
  })

  it('the constraints are sound', () => {
    const proves = (pi: number, examples: readonly string[]) => {
      const text = `${LAST.background}\n${popper.programs[pi].clauses.map((i) => popper.clauses[i].text).join('\n')}`
      const p = prologProgram(text)
      const test = (e: string) => solveQuery(p, e, { maxSolutions: 1, maxSteps: 2000, maxDepth: 40 }).solutions.length
      return examples.map((e) => test(e) > 0)
    }
    const by = (c: string) => popper.prunedBy.flatMap((x, i) => (x === c ? [i] : []))
    expect(by('generalisation')).toHaveLength(123)
    expect(by('specialisation')).toHaveLength(421)
    expect(by('elimination')).toHaveLength(199)
    // A sample of each kind, to keep the test fast.
    for (const pi of by('generalisation').slice(0, 20)) expect(proves(pi, LAST.negatives).some(Boolean)).toBe(true)
    for (const pi of by('specialisation').slice(0, 20)) expect(proves(pi, LAST.positives).every(Boolean)).toBe(false)
    for (const pi of by('elimination').slice(0, 20)) {
      const solved = proves(pi, LAST.positives).every(Boolean) && !proves(pi, LAST.negatives).some(Boolean)
      expect(solved && popper.programs[pi].size <= 7).toBe(false)
    }
  })

  it('ancestor/2: 44 clauses, 497 programs; 21 tests against 35', () => {
    const r = learnFromFailures(ANCESTOR, { maxVars: 3, maxBody: 2, maxClauses: 2 })
    expect(r.clauses).toHaveLength(44)
    expect(r.programs).toHaveLength(497)
    expect(r.steps).toHaveLength(21)
    expect(r.enumerateTests).toBe(35)
    expect(solutionText(r)).toEqual([
      'ancestor(A, B) :- parent(A, B).',
      'ancestor(A, B) :- ancestor(A, C), ancestor(C, B).',
    ])
  })

  it('builds the space from its parts, and refuses too few variables', () => {
    const clauses = lffClauses(ANCESTOR, { maxVars: 3, maxBody: 2 })
    const subsumes = lffSubsumption(ANCESTOR.target, clauses)
    expect(subsumes.every((row, i) => row[i])).toBe(true)
    expect(lffPrograms(clauses, subsumes, 2)).toHaveLength(497)
    expect(() => lffClauses(ANCESTOR, { maxVars: 1 })).toThrow(DomainError)
  })
})
