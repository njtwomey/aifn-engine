/**
 * Martelli and Montanari's unification step by step (aifn-compute/logic/terms `unificationTrace`) against the engine's
 * `unify`: the same pairs fail, and otherwise both unifiers give variants of one common instance.
 */
import { describe, expect, it } from 'vitest'
import {
  applySubstitution,
  matchTerm,
  parseQuery,
  renameVariables,
  substitutionToString,
  termsEqual,
  unificationTrace,
  unify,
  type Term,
} from 'aifn-compute/logic/terms'

const PAIRS = [
  'f(X, g(Y)), f(a, g(b))',
  'p(X, [Y | T]), p(1, [2, 3])',
  'f(X, X), f(a, b)',
  'f(X, Y, Z), f(g(Y), h(Z), a)',
  'grandparent(ann, Z), grandparent(X1, Z1)',
  'X, f(X)',
  'f(X, g(X)), f(Y, Y)',
  'p(f(X), Y), p(Y, f(a))',
  'k(X, Y), k(Y, X)',
]

const goals = (text: string) => parseQuery(text).goals as [Term, Term]

describe('unificationTrace', () => {
  it.each(PAIRS)('agrees with unify (occurs check on): %s', (text) => {
    const [a, b] = goals(text)
    const steps = unificationTrace(a, b, { occursCheck: true })
    const final = steps[steps.length - 1]
    const engine = unify(a, b, { occursCheck: true })
    expect(final.status === 'failed').toBe(engine === null)
    if (engine === null) return
    expect(final.rule).toBe('done')
    const ia = applySubstitution(a, final.solved)
    const ib = applySubstitution(b, final.solved)
    const ea = applySubstitution(a, engine)
    expect(termsEqual(ia, ib)).toBe(true)
    expect(matchTerm(renameVariables(ia, 500), ea)).not.toBeNull()
    expect(matchTerm(renameVariables(ea, 500), ia)).not.toBeNull()
  })

  it('fails on a clash and on the occurs check', () => {
    const clash = unificationTrace(...goals('f(X, X), f(a, b)'))
    expect(clash[clash.length - 1].rule).toBe('conflict')
    const cyc = unificationTrace(...goals('X, f(X)'), { occursCheck: true })
    expect(cyc.map((s) => s.rule)).toEqual(['start', 'occurs'])
    expect(cyc[1].status).toBe('failed')
  })

  it('binds X to f(X) without the occurs check (the default, as unify)', () => {
    const cyc = unificationTrace(...goals('X, f(X)'))
    expect(cyc.map((s) => s.rule)).toEqual(['start', 'eliminate', 'done'])
    expect(cyc[cyc.length - 1].status).toBe('unified')
  })

  it('traces the worked example to a solved form', () => {
    const steps = unificationTrace(...goals('p(f(X), Y, b), p(Z, g(Z), W)'))
    expect(steps[0].rule).toBe('start')
    expect(steps[0].equations).toHaveLength(1)
    const final = steps[steps.length - 1]
    expect(final.status).toBe('unified')
    expect(substitutionToString(final.solved)).toBe('{Z = f(X), Y = g(f(X)), W = b}')
    expect(steps.map((s) => s.rule)).toEqual([
      'start',
      'decompose',
      'swap',
      'eliminate',
      'eliminate',
      'swap',
      'eliminate',
      'done',
    ])
  })
})
