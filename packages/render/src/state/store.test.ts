import { describe, expect, it } from 'vitest'
import { choice, row, setting, slider, toggle, variants } from './schema'
import { fromEntries, initialRaw, setAt, toEntries, typedValues } from './store'

const schema = {
  input: variants(
    { normal: { mu: slider(-3, 3, 0), sd: slider(0.1, 3, 1) }, gamma: { shape: slider(0.5, 8, 2) } },
    { label: '1 · input' },
  ),
  reveal: row('reveal', { jacobian: toggle(false, 'Jacobian'), draws: choice([1000, 5000, 20000]) }),
  log: setting(false),
  x0: slider(-4, 4, 0.5, { onChart: true }),
}

describe('figure state store', () => {
  it('starts at the initial values, typed', () => {
    const v = typedValues(schema, initialRaw(schema)) as Record<string, any>
    expect(v.input.key).toBe('normal')
    expect(v.input.values).toEqual({ mu: 0, sd: 1 })
    expect(v.reveal).toEqual({ jacobian: false, draws: 1000 })
    expect(v.x0).toBe(0.5)
  })
  it('sets by path, clamps, and remembers each case', () => {
    let r = initialRaw(schema)
    r = setAt(schema, r, ['input', 'mu'], 9)
    r = setAt(schema, r, ['input'], 'gamma')
    r = setAt(schema, r, ['input', 'shape'], 3)
    r = setAt(schema, r, ['input'], 'normal')
    const v = typedValues(schema, r) as Record<string, any>
    expect(v.input.values.mu).toBe(3)
    r = setAt(schema, r, ['input'], 'gamma')
    expect((typedValues(schema, r) as any).input.values.shape).toBe(3)
  })
  it('round-trips through the URL form with non-defaults only', () => {
    let r = initialRaw(schema)
    expect(toEntries(schema, r)).toEqual([])
    r = setAt(schema, r, ['input'], 'gamma')
    r = setAt(schema, r, ['input', 'shape'], 3)
    r = setAt(schema, r, ['reveal', 'draws'], 5000)
    r = setAt(schema, r, ['x0'], -1)
    const entries = toEntries(schema, r)
    expect(entries).toEqual([
      ['input', 'gamma'],
      ['input.shape', '3'],
      ['reveal.draws', '5000'],
      ['x0', '-1'],
    ])
    const back = fromEntries(schema, entries)
    expect(back.dropped).toEqual([])
    expect(toEntries(schema, back.raw)).toEqual(entries)
  })
  it('reports unknown, inactive and unparseable keys', () => {
    const d = fromEntries(schema, [
      ['input.shape', '2'],
      ['nope', '1'],
      ['x0', 'abc'],
      ['x0', '99'],
    ])
    expect(d.dropped).toContain('nope')
    expect(d.dropped).toContain('x0')
    expect(d.dropped.some((k) => k.includes('shape'))).toBe(true)
    expect((typedValues(schema, d.raw) as any).x0).toBe(4)
  })
})
