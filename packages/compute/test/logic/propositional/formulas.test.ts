/**
 * Propositional formulas (aifn-compute/logic/propositional): models and entailment by truth table on the sprinkler
 * knowledge base, truth-table order, and printing with precedence and right-associative implication.
 */
import { describe, expect, it } from 'vitest'
import { DomainError } from 'aifn-compute/foundation/errors'
import {
  atomsOf,
  conjoin,
  disj,
  entails,
  evaluateFormula,
  formulaText,
  iff,
  implies,
  interpretations,
  modelsOf,
  negate,
  prop,
  truth,
} from 'aifn-compute/logic/propositional'

const rain = prop('rain')
const wet = prop('wet')
const sprinkler = prop('sprinkler')
const KB = [implies(rain, wet), implies(sprinkler, wet), disj(rain, sprinkler)]
const ATOMS = ['rain', 'sprinkler', 'wet']

describe('entailment by enumeration', () => {
  it('KB has 3 of 8 models and entails wet', () => {
    const e = entails(KB, wet, ATOMS)
    expect(e.models).toHaveLength(3)
    expect(e.entailed).toBe(true)
    expect(e.counterModels).toEqual([])
  })

  it('KB does not entail rain: one counter-model, rain false and sprinkler true', () => {
    const r = entails(KB, rain, ATOMS)
    expect(r.entailed).toBe(false)
    expect(r.counterModels).toHaveLength(1)
    const I = interpretations(ATOMS)[r.counterModels[0]]
    expect(I).toEqual({ rain: false, sprinkler: true, wet: true })
  })

  it('{rain -> wet, wet} does not entail rain (affirming the consequent)', () => {
    expect(entails([KB[0], wet], rain, ATOMS).entailed).toBe(false)
  })

  it('rain -> wet has 6 models over three atoms; an unsatisfiable KB entails anything', () => {
    expect(modelsOf([KB[0]], ATOMS)).toHaveLength(6)
    expect(entails([rain, negate(rain)], prop('anything')).entailed).toBe(true)
  })
})

describe('interpretations and evaluation', () => {
  it('lists 8 rows, the first all true, the first atom varying slowest', () => {
    const rows = interpretations(ATOMS)
    expect(rows).toHaveLength(8)
    expect(rows[0]).toEqual({ rain: true, sprinkler: true, wet: true })
    expect(rows[1]).toEqual({ rain: true, sprinkler: true, wet: false })
    expect(rows[4].rain).toBe(false)
  })

  it('evaluates every connective and refuses a missing atom or too many atoms', () => {
    expect(evaluateFormula(iff(rain, wet), { rain: false, wet: false })).toBe(true)
    expect(evaluateFormula(conjoin(), {})).toBe(true)
    expect(evaluateFormula(disj(), {})).toBe(false)
    expect(evaluateFormula(truth(false), {})).toBe(false)
    expect(() => evaluateFormula(rain, {})).toThrow(DomainError)
    expect(() => interpretations(Array.from({ length: 21 }, (_, i) => `p${i}`))).toThrow(DomainError)
  })

  it('collects atoms in order of first occurrence', () => {
    expect(atomsOf(...KB)).toEqual(['rain', 'wet', 'sprinkler'])
  })
})

describe('formulaText', () => {
  it('brackets only where precedence needs it', () => {
    const a = prop('a')
    const b = prop('b')
    const c = prop('c')
    expect(formulaText(implies(conjoin(a, disj(b, c)), negate(conjoin(a, b))))).toBe('a ∧ (b ∨ c) → ¬(a ∧ b)')
    expect(formulaText(implies(implies(a, b), c))).toBe('(a → b) → c')
    expect(formulaText(implies(a, implies(b, c)))).toBe('a → b → c')
    expect(formulaText(negate(a), 'tex')).toBe('\\neg \\text{a}')
  })
})
