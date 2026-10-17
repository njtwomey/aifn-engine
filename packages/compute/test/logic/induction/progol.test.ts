/**
 * Mode-directed inverse entailment (aifn-compute/logic/induction `bottomClause`, `bottomClauseSearch`): the bottom
 * clause of daughter(cid, ann) worked by hand in the Progol note, and the search below it, checked by SLD resolution.
 */
import { describe, expect, it } from 'vitest'
import { DomainError } from 'aifn-compute/foundation/errors'
import { solveQuery } from 'aifn-compute/logic/resolution'
import { parseProgram, parseTerm, termToString } from 'aifn-compute/logic/terms'
import { bottomClause, bottomClauseSearch, parseMode, subsetText } from 'aifn-compute/logic/induction'

const FACTS = parseProgram(
  'parent(ann, bob). parent(ann, cid). parent(bob, dee). parent(bob, eve). parent(cid, fay). parent(eve, gus). ' +
    'female(ann). female(cid). female(dee). female(eve). female(fay).',
).clauses.map((c) => c.head)
const POS = ['daughter(cid, ann)', 'daughter(dee, bob)', 'daughter(eve, bob)', 'daughter(fay, cid)'].map(parseTerm)
const NEG = ['daughter(bob, ann)', 'daughter(gus, eve)', 'daughter(ann, cid)', 'daughter(dee, ann)'].map(parseTerm)
const MODES = ['female(+person)', 'parent(+person, -person)', 'parent(-person, +person)'].map(parseMode)
const HEAD = parseMode('daughter(+person, +person)')

describe('bottom clauses', () => {
  const bc = bottomClause(FACTS, POS[0], HEAD, MODES, 1)

  it('saturates daughter(cid, ann) at depth 1 into the five hand-worked literals', () => {
    expect(
      subsetText(
        bc,
        bc.literals.map((_, i) => i),
      ),
    ).toBe('daughter(A, B) :- female(B), female(A), parent(B, C), parent(B, A), parent(A, D).')
    expect(bc.constants.slice(0, 2)).toEqual(['cid', 'ann'])
  })

  it('every literal is a fact of the background under the variable-to-constant map', () => {
    const facts = new Set(FACTS.map((f) => termToString(f)))
    for (const l of bc.literals) {
      let text = termToString(l.literal)
      bc.names.forEach((n, i) => (text = text.replace(new RegExp(`\\b${n}\\b`, 'g'), bc.constants[i])))
      expect(facts.has(text)).toBe(true)
    }
  })

  it('searches 24 clauses and returns daughter(A, B) :- female(A), parent(B, A) with score 2', () => {
    const steps = bottomClauseSearch(bc, FACTS, POS, NEG)
    expect(steps).toHaveLength(24)
    const best = steps[steps.length - 1].best!
    expect(best.text).toBe('daughter(A, B) :- female(A), parent(B, A).')
    expect(best.score).toBe(2)
    expect(best.positives).toEqual([0, 1, 2, 3])
    expect(best.negatives).toEqual([])
    const prog = `${FACTS.map((f) => `${termToString(f)}.`).join(' ')} ${best.text}`
    for (const e of POS) expect(solveQuery(prog, termToString(e)).solutions.length).toBeGreaterThan(0)
    for (const e of NEG) expect(solveQuery(prog, termToString(e)).solutions).toHaveLength(0)
  })

  it('reads modes and refuses malformed ones', () => {
    expect(parseMode('parent(+person, -person)')).toEqual({
      predicate: 'parent',
      args: [
        { io: '+', type: 'person' },
        { io: '-', type: 'person' },
      ],
    })
    expect(() => parseMode('parent')).toThrow(DomainError)
    expect(() => parseMode('parent(person)')).toThrow(DomainError)
  })
})
