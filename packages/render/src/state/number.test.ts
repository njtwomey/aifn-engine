import { describe, expect, it } from 'vitest'
import {
  checkNumber,
  clampNumber,
  formatNumberValue,
  numberBounds,
  parseNumber,
  stepNumber,
  validateNumber,
} from './number'
import { float, fromSpace, int, number, toSpace, type NumberDef } from './schema'
import { fromEntries, setAt, initialRaw, typedValues } from './store'

describe('parseNumber', () => {
  it('reads decimals, exponents and a typographic minus', () => {
    expect(parseNumber(' 3e-4 ')).toBe(3e-4)
    expect(parseNumber('−2.5')).toBe(-2.5)
    expect(parseNumber('.5')).toBe(0.5)
    expect(parseNumber('1E3')).toBe(1000)
  })
  it('rejects empty, partial, hex and infinite text', () => {
    for (const t of ['', ' ', '1e', 'abc', '0x10', 'Infinity', '1,000', '--1']) expect(parseNumber(t)).toBeNull()
  })
})

describe('validateNumber', () => {
  it('checks the int type', () => {
    expect(validateNumber({ type: 'int' }, 2.5)).toBe('must be an integer')
    expect(validateNumber({ type: 'int' }, 3)).toBeNull()
    expect(validateNumber({}, 2.5)).toBeNull()
  })
  it('checks strict and inclusive bounds', () => {
    expect(validateNumber({ gt: 0 }, 0)).toBe('must be > 0')
    expect(validateNumber({ ge: 0 }, 0)).toBeNull()
    expect(validateNumber({ lt: 1 }, 1)).toBe('must be < 1')
    expect(validateNumber({ le: 5000 }, 5000)).toBeNull()
    expect(validateNumber({ le: 5000 }, 5001)).toBe('must be ≤ 5000')
    expect(validateNumber({ ge: 1 }, 0)).toBe('must be ≥ 1')
  })
  it('treats min and max as ge and le', () => {
    expect(validateNumber({ min: 1, max: 3 }, 1)).toBeNull()
    expect(validateNumber({ min: 1, max: 3 }, 4)).toBe('must be ≤ 3')
  })
  it('checkNumber gives the value or the reason', () => {
    expect(checkNumber({ gt: 0 }, '1e-3')).toEqual({ value: 1e-3 })
    expect(checkNumber({ gt: 0 }, '-1')).toEqual({ error: 'must be > 0' })
    expect(checkNumber({ gt: 0 }, 'x')).toEqual({ error: 'not a number' })
  })
})

describe('numberBounds', () => {
  it('takes the tighter bound of each pair', () => {
    expect(numberBounds({ gt: 0, ge: -1 })).toMatchObject({ lower: 0, lowerStrict: true })
    expect(numberBounds({ gt: 0, ge: 1 })).toMatchObject({ lower: 1, lowerStrict: false })
  })
  it('turns an int’s strict bounds into inclusive integers', () => {
    expect(numberBounds({ type: 'int', gt: 0, lt: 10 })).toEqual({
      lower: 1,
      lowerStrict: false,
      upper: 9,
      upperStrict: false,
    })
  })
})

describe('clampNumber', () => {
  it('clamps inclusive bounds and rounds ints', () => {
    expect(clampNumber({ type: 'int', ge: 1, le: 10 }, 12.7)).toBe(10)
    expect(clampNumber({ type: 'int', ge: 1 }, 2.6)).toBe(3)
  })
  it('stops just inside a strict bound', () => {
    const y = clampNumber({ gt: 0 }, -1)
    expect(y > 0).toBe(true)
    expect(validateNumber({ gt: 0 }, y)).toBeNull()
  })
})

describe('stepNumber', () => {
  const lr = { gt: 0, scale: 'log10' } as const
  it('steps a log10 field by half decades along the grid', () => {
    expect(stepNumber(lr, 1e-3, 1)).toBeCloseTo(3.16227766e-3, 10)
    expect(stepNumber(lr, stepNumber(lr, 1e-3, 1)!, 1)).toBe(1e-2)
    expect(stepNumber(lr, 1e-3, -1)).toBeCloseTo(3.16227766e-4, 11)
    expect(stepNumber(lr, 2e-3, 1)).toBeCloseTo(3.16227766e-3, 10)
  })
  it('moves a whole decade with Shift, and by a given step in decades', () => {
    expect(stepNumber(lr, 1e-3, 1, true)).toBe(1e-2)
    expect(stepNumber({ ...lr, step: 1 / 3 }, 1e-3, 1)).toBeCloseTo(10 ** (-3 + 1 / 3), 12)
  })
  it('clamps at inclusive bounds and reports no move there', () => {
    expect(stepNumber({ type: 'int', ge: 1, le: 5 }, 5, 1)).toBeNull()
    expect(stepNumber({ ge: 0, step: 0.3 }, 0.2, -1)).toBe(0)
    expect(stepNumber({ type: 'int', ge: 1 }, 1, -1)).toBeNull()
  })
  it('stops inside a strict bound', () => {
    expect(stepNumber({ gt: 0 }, 0.5, -1)).toBeNull()
    expect(stepNumber({ lt: 1, step: 0.25 }, 0.5, 1)).toBe(0.75)
    expect(stepNumber({ lt: 1, step: 0.25 }, 0.75, 1)).toBeNull()
  })
  it('moves an int on a log scale by at least one', () => {
    expect(stepNumber({ type: 'int', ge: 1, scale: 'log10' }, 1, 1)).toBe(3)
    expect(stepNumber({ type: 'int', ge: 1, scale: 'log10' }, 3, 1)).toBe(10)
  })
})

describe('formatNumberValue', () => {
  it('uses compact exponents on a log10 scale', () => {
    expect(formatNumberValue({ scale: 'log10' }, 1e-3)).toBe('1e-3')
    expect(formatNumberValue({ scale: 'log10' }, 3.16227766e-4)).toBe('3.16e-4')
    expect(formatNumberValue({ scale: 'log10' }, 0.0316)).toBe('0.0316')
    expect(formatNumberValue({ scale: 'log10' }, 2e5)).toBe('2e5')
    expect(formatNumberValue({ type: 'int' }, 5000)).toBe('5000')
  })
})

describe('builders', () => {
  it('float and int set the type', () => {
    expect(float(1e-3, { gt: 0, scale: 'log10' })).toMatchObject({ kind: 'number', type: 'float', gt: 0 })
    expect(int(200, { ge: 1, le: 5000 })).toMatchObject({ kind: 'number', type: 'int', initial: 200 })
  })
})

describe('number fields in the store and the URL', () => {
  const schema = {
    lr: float(1e-3, { gt: 0, scale: 'log10' }),
    episodes: int(200, { ge: 1, le: 10000, suggestions: [100, 500] }),
    seed: number(1, { min: 0, max: 9999, step: 1 }),
  }
  it('rejects invalid URL values and reports them', () => {
    const d = fromEntries(schema, [
      ['lr', '0'],
      ['episodes', '2.5'],
      ['seed', '1e6'],
    ])
    expect(d.dropped).toEqual(['lr', 'episodes', 'seed'])
    expect(typedValues(schema, d.raw)).toEqual({ lr: 1e-3, episodes: 200, seed: 1 })
  })
  it('keeps valid URL values exactly', () => {
    const d = fromEntries(schema, [
      ['lr', '3e-4'],
      ['episodes', '10000'],
      ['seed', '0'],
    ])
    expect(d.dropped).toEqual([])
    expect(typedValues(schema, d.raw)).toEqual({ lr: 3e-4, episodes: 10000, seed: 0 })
  })
  it('clamps programmatic sets (the buttons’ rule)', () => {
    let r = initialRaw(schema)
    r = setAt(schema, r, ['episodes'], 20000.4)
    r = setAt(schema, r, ['lr'], -1)
    const v = typedValues(schema, r) as { episodes: number; lr: number }
    expect(v.episodes).toBe(10000)
    expect(v.lr > 0).toBe(true)
  })
})

describe('Space mapping', () => {
  it('maps an int to an int dimension with integer bounds', () => {
    const s = toSpace({ n: int(5, { gt: 0, le: 100 }) })
    expect(s.dims.n).toMatchObject({ type: 'int', min: 1, max: 100, default: 5 })
  })
  it('maps a log10 float with a strict bound to a log real just inside it', () => {
    const s = toSpace({ lr: float(1e-3, { gt: 0, lt: 1, scale: 'log10' }) })
    const d = s.dims.lr as { type: string; min: number; max: number; scale?: string }
    expect(d.type).toBe('real')
    expect(d.scale).toBe('log')
    expect(d.min > 0 && d.min < 1e-300).toBe(true)
    expect(d.max < 1 && d.max > 0.999999).toBe(true)
  })
  it('round-trips the type and bounds through fromSpace for a wide int and a log real', () => {
    const s = toSpace({ episodes: int(200, { ge: 1, le: 10000 }), lr: float(0.1, { ge: 1e-4, le: 1, scale: 'log10' }) })
    const back = fromSpace(s) as Record<string, NumberDef>
    expect(back.episodes).toMatchObject({ kind: 'number', type: 'int', ge: 1, le: 10000, initial: 200 })
    expect(back.lr).toMatchObject({ kind: 'number', type: 'float', ge: 1e-4, le: 1, scale: 'log10', initial: 0.1 })
  })
})
