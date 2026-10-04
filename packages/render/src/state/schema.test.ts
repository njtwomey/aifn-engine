import { describe, expect, it } from 'vitest'
import { choice, coerce, isCase, number, slider, variants, when, type VariantsDef } from './schema'
import { fromEntries, initialRaw, setAt, toEntries, typedValues } from './store'

describe('isCase', () => {
  it('tells a { label, params } case from a bare record of fields', () => {
    expect(isCase({ label: 'xor', params: { a: slider(0, 1, 0.5) } })).toBe(true)
    expect(isCase({ a: slider(0, 1, 0.5) })).toBe(false)
  })
  it('keeps a case whose parameters include a field named kind', () => {
    expect(isCase({ label: 'xor', params: { kind: number(1), noise: slider(0, 1, 0.1) } })).toBe(true)
  })
  it('a bare record with a field named kind is still bare', () => {
    expect(isCase({ kind: number(1) })).toBe(false)
  })
})

describe('variants with a kind field', () => {
  const def = variants({
    xor: { label: 'xor', params: { kind: slider(0, 3, 1, { step: 1 }), noise: slider(0, 1, 0.1) } },
    moons: { noise: slider(0, 1, 0.2) },
  }) as VariantsDef
  const schema = { base: def }
  it('builds each case once', () => {
    expect(def.specs.xor.label).toBe('xor')
    expect(Object.keys(def.specs.xor.params)).toEqual(['kind', 'noise'])
    expect(def.specs.moons.label).toBe('moons')
  })
  it('sets, reads and round-trips the case values through the URL form', () => {
    let r = initialRaw(schema)
    r = setAt(schema, r, ['base', 'kind'], 2)
    const v = typedValues(schema, r) as { base: { key: string; values: Record<string, number> } }
    expect(v.base.key).toBe('xor')
    expect(v.base.values.kind).toBe(2)
    const entries = toEntries(schema, r)
    expect(entries).toEqual([['base.kind', '2']])
    const back = fromEntries(schema, entries)
    expect(back.dropped).toEqual([])
    expect(toEntries(schema, back.raw)).toEqual(entries)
  })
})

describe('coerce and URL decoding snap to the step grid', () => {
  const schema = { alpha: slider(0.02, 10, 0.4, { step: 0.05 }), n: number(5, { min: 1, step: 1 }) }
  it('keeps an initial value on the grid of multiples', () => {
    expect(coerce(schema.alpha, 0.4)).toBe(0.4)
    expect((typedValues(schema, initialRaw(schema)) as { alpha: number }).alpha).toBe(0.4)
  })
  it('snaps a URL value to the grid and clamps an out-of-range one, reporting it', () => {
    const d = fromEntries(schema, [
      ['alpha', '0.43'],
      ['n', '1e6'],
    ])
    const v = typedValues(schema, d.raw) as { alpha: number; n: number }
    expect(v.alpha).toBe(0.45)
    expect(v.n).toBe(1e6)
    // Out of range is clamped (clamp is silent); only unknown and inactive keys are dropped and reported.
    const out = fromEntries({ alpha: schema.alpha }, [['alpha', '50']])
    expect(out.dropped).toEqual([])
    expect((typedValues({ alpha: schema.alpha }, out.raw) as { alpha: number }).alpha).toBe(10)
  })
})

describe('clampReport drops', () => {
  const schema = {
    shape: choice(['plain', 'shifted']),
    shift: slider(-2, 2, 0, { when: when('shape', 'shifted') }),
  }
  it('drops keys of inactive fields and unknown keys, and keeps active ones', () => {
    const inactive = fromEntries(schema, [['shift', '1']])
    expect(inactive.dropped).toEqual(['shift'])
    const active = fromEntries(schema, [
      ['shape', 'shifted'],
      ['shift', '1'],
      ['other', '3'],
    ])
    expect(active.dropped).toEqual(['other'])
    expect((typedValues(schema, active.raw) as { shift: number }).shift).toBe(1)
  })
})
