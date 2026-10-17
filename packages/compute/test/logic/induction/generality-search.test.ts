/**
 * Top-down and bottom-up search of the generality order (aifn-compute/logic/induction `topDownSearch`,
 * `bottomUpSearch`) on three binary attributes.
 */
import { describe, expect, it } from 'vitest'
import {
  bottomUpSearch,
  conjunctionText,
  topDownSearch,
  type AttributeSpace,
  type LabelledInstance,
} from 'aifn-compute/logic/induction'

const BINARY: AttributeSpace = {
  names: ['a', 'b', 'c'],
  values: [
    ['0', '1'],
    ['0', '1'],
    ['0', '1'],
  ],
}
const EX: LabelledInstance[] = [
  { x: [1, 0, 0], positive: true },
  { x: [1, 1, 0], positive: true },
  { x: [0, 1, 0], positive: false },
]
const last = <T>(xs: readonly T[]) => xs[xs.length - 1]

describe('generality search', () => {
  it('top-down returns a = 1, scoring the refinements of the top 2, 1, 0, 1', () => {
    const td = topDownSearch(BINARY, EX)
    expect(conjunctionText(last(td).hypothesis, BINARY, 'logic')).toBe('a = 1')
    expect(td[0].candidates.map((c) => c.score)).toEqual([2, 1, 0, 1])
    expect(td[0].candidates.map((c) => conjunctionText(c.hypothesis, BINARY, 'logic'))).toEqual([
      'a = 1',
      'b = 0',
      'b = 1',
      'c = 0',
    ])
  })

  it('bottom-up returns a = 1 and c = 0', () => {
    const bu = bottomUpSearch(EX)
    expect(bu[0].hypothesis).toBeNull()
    expect(conjunctionText(last(bu).hypothesis, BINARY, 'logic')).toBe('a = 1 ∧ c = 0')
  })

  it('with 111 negative, top-down meets bottom-up at a = 1 and c = 0', () => {
    const td = topDownSearch(BINARY, [...EX, { x: [1, 1, 1], positive: false }])
    expect(conjunctionText(last(td).hypothesis, BINARY, 'logic')).toBe('a = 1 ∧ c = 0')
  })

  it('stops at the step limit', () => {
    const td = topDownSearch(BINARY, [...EX, { x: [1, 1, 1], positive: false }], { maxSteps: 1 })
    expect(last(td).note).toBe('step limit reached')
  })
})
