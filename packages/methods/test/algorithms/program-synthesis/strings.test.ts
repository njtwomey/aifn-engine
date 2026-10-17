/**
 * String programs from examples (aifn-methods/algorithms/program-synthesis): the version-space counts equal brute-force
 * enumeration of the paths, and the cheapest program generalises to a new input.
 */
import { describe, expect, it } from 'vitest'
import {
  atomText,
  countByLength,
  evalAtom,
  exampleVersionSpace,
  intersectVersionSpace,
  runStringProgram,
  stringTokens,
  topPrograms,
  versionSpace,
  type StringAtom,
} from 'aifn-methods/algorithms/program-synthesis'
import { DomainError } from 'aifn-compute/foundation/errors'

const E1 = { input: 'jane doe', output: 'Doe, J.' }
const E2 = { input: 'alan turing', output: 'Turing, A.' }
const total = (counts: readonly number[]) => counts.reduce((s, x) => s + x, 0)

describe('string program synthesis', () => {
  it('one example: 216 programs, equal to the enumerated paths, each mapping the input to the output', () => {
    const space = exampleVersionSpace(E1)
    const all: StringAtom[][] = []
    const grow = (i: number, acc: StringAtom[]) => {
      if (i === E1.output.length) {
        all.push(acc)
        return
      }
      for (const e of space.edges.filter((x) => x.from === i))
        for (const k of e.atoms) grow(e.to, [...acc, space.atoms.get(k)!])
    }
    grow(0, [])
    expect(total(countByLength(space))).toBe(216)
    expect(all).toHaveLength(216)
    expect(all.every((p) => runStringProgram(p, E1.input) === E1.output)).toBe(true)

    const both = intersectVersionSpace(space, E2)
    const brute = all.filter((p) => runStringProgram(p, E2.input) === E2.output).length
    expect(total(countByLength(both))).toBe(8)
    expect(brute).toBe(8)
    expect(total(countByLength(versionSpace([E1, E2])))).toBe(8)
  })

  it('the cheapest program maps ada lovelace to Lovelace, A.', () => {
    const [best] = topPrograms(versionSpace([E1, E2]), 1, 'fewest-atoms')
    expect(runStringProgram(best.atoms, 'ada lovelace')).toBe('Lovelace, A.')
    expect(best.atoms.map(atomText)).toEqual(['title(token 2)', '", "', 'upper(initial(token 1))', '"."'])
  })

  it('evaluates atoms and reports an atom undefined on an input', () => {
    expect(evalAtom({ kind: 'token', index: -1, transform: 'title' }, 'grace brewster hopper')).toBe('Hopper')
    expect(evalAtom({ kind: 'initial', index: 2, transform: 'upper' }, 'grace brewster hopper')).toBe('B')
    expect(evalAtom({ kind: 'token', index: 4, transform: 'id' }, 'grace brewster hopper')).toBeNull()
    expect(runStringProgram([{ kind: 'token', index: 3, transform: 'id' }], 'ada lovelace')).toBeNull()
    expect(stringTokens('grace brewster-hopper, 1906')).toEqual(['grace', 'brewster', 'hopper', '1906'])
  })

  it('an inconsistent example empties the space', () => {
    const space = versionSpace([E1, { input: 'alan turing', output: 'Zzz' }])
    expect(space.end).toBe(-1)
    expect(countByLength(space)).toEqual([])
    expect(topPrograms(space, 3, 'fewest-constants')).toEqual([])
    expect(() => versionSpace([])).toThrow(DomainError)
  })
})
