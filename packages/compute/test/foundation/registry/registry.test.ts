import { describe, expect, it } from 'vitest'
import { define, entries, isEntry } from 'aifn-compute/foundation/registry'
import { AifnError } from 'aifn-compute/foundation/errors'
import { auroc } from 'aifn-compute/learning/metrics'

const info = (key: string) =>
  ({ kind: 'metric', key, module: 'test/registry', name: key, stability: 'stable' }) as const

describe('define, isEntry and entries', () => {
  it('define attaches frozen metadata and returns the value itself', () => {
    const f = (x: number) => x + 1
    const e = define(info('plus-one'), f)
    expect(e).toBe(f)
    expect(e(1)).toBe(2)
    expect(e.info.key).toBe('plus-one')
    expect(Object.isFrozen(e.info)).toBe(true)
  })

  it('isEntry recognises entries, of a kind when asked', () => {
    const e = define(info('a'), { value: 1 })
    expect(isEntry(e)).toBe(true)
    expect(isEntry(e, 'metric')).toBe(true)
    expect(isEntry(e, 'loss')).toBe(false)
    for (const x of [null, 1, 'x', {}, { info: null }, { info: { kind: 3 } }]) expect(isEntry(x)).toBe(false)
    // A library entry.
    expect(isEntry(auroc, 'metric')).toBe(true)
  })

  it('entries collects one kind from namespaces, keyed, in definition order, frozen', () => {
    const a = define(info('a'), () => 1)
    const b = define(info('b'), () => 2)
    const loss = define({ ...info('c'), kind: 'loss' } as const, () => 3)
    const table = entries('metric', { a, loss }, { b, other: 5 })
    expect(Object.keys(table)).toEqual(['a', 'b'])
    expect(table.a).toBe(a)
    expect(Object.isFrozen(table)).toBe(true)
    // The same value under two names is one entry; two values under one key are an error.
    expect(Object.keys(entries('metric', { a, again: a }))).toEqual(['a'])
    const clash = define(info('a'), () => 4)
    expect(() => entries('metric', { a, clash })).toThrow(AifnError)
  })
})
