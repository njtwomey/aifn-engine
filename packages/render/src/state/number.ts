/**
 * A typed number field's rules, as pure functions: its type (`float` or `int`), bounds (`gt`, `ge`, `lt`, `le`; `min`
 * and `max` are aliases of `ge` and `le`), scale (`linear` or `log10`) and step. One definition serves the field's
 * validation, its − and + buttons and ↑/↓ keys, the URL decoder and the conversion to an aifn `Space`.
 *
 *   float(1e-3, { gt: 0, scale: 'log10', suggestions: [1e-4, 3e-4, 1e-3] })
 *   int(200, { ge: 1, le: 5000 })
 */
import { formatField } from './step'

export type NumberType = 'float' | 'int'
export type NumberScale = 'linear' | 'log10'
export type SpacingMode = 'linear' | 'log10' | 'lin' | 'log'

export type NumberOptions = {
  /** `int` accepts only integers (default `float`). */
  type?: NumberType
  /** Strictly greater than. */
  gt?: number
  /** Greater than or equal to. */
  ge?: number
  /** Strictly less than. */
  lt?: number
  /** Less than or equal to. */
  le?: number
  /** An alias of `ge`. */
  min?: number
  /** An alias of `le`. */
  max?: number
  /** Scale: linear or log10. */
  scale?: NumberScale
  /** Spacing mode: 'lin' | 'log' | 'linear' | 'log10'. */
  spacing?: SpacingMode
  /** Increment mode alias: 'lin' | 'log' | 'linear' | 'log10'. */
  increment?: SpacingMode
  /** Points per decade on log scale (e.g. 2 for 1, 3, 10, 30; 3 for 1, 2, 5, 10; 1 for 1, 10, 100). */
  points_per_decade?: number
  /** Points per decade (on log scale) or step size (on lin scale, e.g. points=10 increments by 10). */
  points?: number
  /**
   * Linear: the step of − and + and of ↑/↓ (default 1; × 10 with Shift). Log10: the step in decades. Values are not
   * snapped to it.
   */
  step?: number
  /** Common values, one click away in a menu beside the field. If omitted, sensible defaults are generated. */
  suggestions?: readonly number[]
  /** When true or 'value-is-log' or 'value-is-real', shows log and real value in header, e.g. -2 (0.01). */
  logTransform?: boolean | 'value-is-log' | 'value-is-real'
  /** Custom header readout value or formatter function. */
  headerValue?: string | ((v: number) => string)
}

/** Whether the options specify logarithmic scaling or spacing. */
export function isLogScale(o: NumberOptions): boolean {
  return (
    o.scale === 'log10' ||
    o.spacing === 'log' ||
    o.spacing === 'log10' ||
    o.increment === 'log' ||
    o.increment === 'log10'
  )
}

/** Standard mantissas for points-per-decade sequences (e.g. 2 -> [1, 3], 3 -> [1, 2, 5]). */
export function logSeriesMantissas(pointsPerDecade: number): number[] {
  if (pointsPerDecade <= 1) return [1]
  if (pointsPerDecade === 2) return [1, 3]
  if (pointsPerDecade === 3) return [1, 2, 5]
  if (pointsPerDecade === 5) return [1, 1.5, 2.5, 4, 6.3]
  return Array.from({ length: pointsPerDecade }, (_, i) => Number((10 ** (i / pointsPerDecade)).toPrecision(2)))
}

/** The bounds as one interval: each end finite or infinite, strict or not. An int's bounds are inclusive integers. */
export type Bounds = { lower: number; lowerStrict: boolean; upper: number; upperStrict: boolean }

export function numberBounds(o: NumberOptions): Bounds {
  const ge = Math.max(o.ge ?? -Infinity, o.min ?? -Infinity)
  const le = Math.min(o.le ?? Infinity, o.max ?? Infinity)
  const gt = o.gt ?? -Infinity
  const lt = o.lt ?? Infinity
  // The tighter of each pair; at a tie the strict one.
  let lower = gt >= ge ? gt : ge
  let lowerStrict = o.gt !== undefined && gt >= ge
  let upper = lt <= le ? lt : le
  let upperStrict = o.lt !== undefined && lt <= le
  if (o.type === 'int') {
    if (Number.isFinite(lower)) [lower, lowerStrict] = [lowerStrict ? Math.floor(lower) + 1 : Math.ceil(lower), false]
    if (Number.isFinite(upper)) [upper, upperStrict] = [upperStrict ? Math.ceil(upper) - 1 : Math.floor(upper), false]
  }
  return { lower, lowerStrict, upper, upperStrict }
}

/** The text of a typed number: decimal or exponent notation, an optional sign ("−" too); null otherwise. */
export function parseNumber(text: string): number | null {
  const t = text.trim().replace(/^[−–]/, '-')
  if (!/^[+-]?(\d+\.?\d*|\.\d+)(e[+-]?\d+)?$/i.test(t)) return null
  const x = Number(t)
  return Number.isFinite(x) ? x : null
}

/** Why `x` is not a valid value ("must be an integer", "must be > 0"), or null when it is. */
export function validateNumber(o: NumberOptions, x: number): string | null {
  if (!Number.isFinite(x)) return 'not a number'
  if (o.type === 'int' && !Number.isInteger(x)) return 'must be an integer'
  const f = (v: number) => formatNumberValue({ scale: o.scale }, v)
  if (o.gt !== undefined && !(x > o.gt)) return `must be > ${f(o.gt)}`
  const ge = Math.max(o.ge ?? -Infinity, o.min ?? -Infinity)
  if (!(x >= ge)) return `must be ≥ ${f(ge)}`
  if (o.lt !== undefined && !(x < o.lt)) return `must be < ${f(o.lt)}`
  const le = Math.min(o.le ?? Infinity, o.max ?? Infinity)
  if (!(x <= le)) return `must be ≤ ${f(le)}`
  return null
}

/** A draft's value, or why it cannot be committed. */
export function checkNumber(o: NumberOptions, text: string): { value: number; error?: undefined } | { error: string } {
  const x = parseNumber(text)
  if (x === null) return { error: 'not a number' }
  const error = validateNumber(o, x)
  return error ? { error } : { value: x }
}

/** The smallest double above `x` (near enough: one ulp for |x| ≥ 2⁻¹⁰²²), for a strict bound. */
export const nextUp = (x: number) => (x === 0 ? Number.MIN_VALUE : x + Math.abs(x) * Number.EPSILON)
export const nextDown = (x: number) => -nextUp(-x)

/**
 * `x` made valid by clamping (what the buttons, handles and programmatic sets do; a typed draft is validated
 * instead): an int is rounded and a strict bound clamps to just inside it. There is no step grid: a valid typed value
 * is kept as typed, and the buttons step from it.
 */
export function clampNumber(o: NumberOptions, x: number): number {
  const b = numberBounds(o)
  const lo = b.lowerStrict ? nextUp(b.lower) : b.lower
  const hi = b.upperStrict ? nextDown(b.upper) : b.upper
  return Math.min(Math.max(o.type === 'int' ? Math.round(x) : x, lo), hi)
}

const clean = (x: number) => Number(x.toPrecision(12))

/**
 * The next value from `x` in direction `dir` (− or + button, ↓ or ↑ key; `big` with Shift), clamped to the bounds; null
 * when there is no move (at an inclusive bound, or the step would cross a strict one: the buttons stop just inside).
 * Log10 steps along the grid 10^(k·step), so 2e-3 goes up to 3.16e-3 rather than 6.32e-3.
 */
export function stepNumber(o: NumberOptions, x: number, dir: 1 | -1, big = false): number | null {
  let y: number
  const isLog = isLogScale(o)
  if (isLog) {
    if (!(x > 0)) return null
    const ppd = o.points_per_decade ?? (isLog && o.points !== undefined && o.points <= 10 ? o.points : undefined)
    if (ppd !== undefined) {
      if (big) {
        y = clean(dir > 0 ? x * 10 : x / 10)
      } else {
        const mantissas = logSeriesMantissas(ppd)
        const k = Math.floor(Math.log10(x))
        const candidates: number[] = []
        for (let j = k - 2; j <= k + 2; j++) {
          for (const m of mantissas) {
            candidates.push(clean(m * 10 ** j))
          }
        }
        candidates.sort((a, b) => a - b)
        if (dir > 0) {
          const next = candidates.find((c) => c > x * (1 + 1e-6))
          y = next ?? clean(x * 10)
        } else {
          const prev = [...candidates].reverse().find((c) => c < x * (1 - 1e-6))
          y = prev ?? clean(x / 10)
        }
      }
      if (o.type === 'int') y = Math.round(y)
    } else {
      const d = big ? 1 : (o.step ?? 0.5)
      const k = Math.log10(x) / d
      let j = dir > 0 ? Math.floor(k + 1e-9) + 1 : Math.ceil(k - 1e-9) - 1
      y = clean(10 ** (j * d))
      // An int takes the next grid point that rounds to a different integer (3 → 10, not 3 → 3.16 → 3).
      if (o.type === 'int') {
        while (Math.round(y) === x && j > -50 && j < 50) y = clean(10 ** ((j += dir) * d))
        y = Math.round(y)
      }
    }
  } else {
    const linStep =
      o.step ?? (o.increment === 'lin' && o.points ? o.points : o.spacing === 'lin' && o.points ? o.points : 1)
    y = clean(x + dir * linStep * (big ? 10 : 1))
    if (o.type === 'int') y = Math.round(y)
  }
  const b = numberBounds(o)
  if (b.lowerStrict && y <= b.lower) return null
  if (b.upperStrict && y >= b.upper) return null
  const next = clampNumber(o, y)
  return next === x ? null : next
}

/**
 * A value as the field shows it: an int in full; log10 in compact exponent form outside [0.01, 10⁴) (1e-3, 3.16e-4,
 * 2e5), three significant figures inside; otherwise as `formatField`.
 */
export function formatNumberValue(o: NumberOptions, x: number): string {
  if (!Number.isFinite(x)) return formatField(x)
  if (o.type === 'int') return String(x)
  if (!isLogScale(o)) return formatField(x)
  const a = Math.abs(x)
  if (a === 0) return '0'
  if (a >= 0.01 && a < 1e4) return String(Number(x.toPrecision(3)))
  const [m, e] = x.toExponential(2).split('e')
  return `${String(Number(m))}e${e.replace('+', '')}`
}

/** Generate sensible suggestions across bounds for quick selection. */
export function defaultSuggestions(o: NumberOptions): number[] {
  const isLog = isLogScale(o)
  const bounds = numberBounds(o)
  const lo = Number.isFinite(bounds.lower) ? bounds.lower : isLog ? 1e-3 : 0
  const hi = Number.isFinite(bounds.upper) ? bounds.upper : isLog ? 1e3 : 100
  const ppd = o.points_per_decade ?? (isLog && o.points ? o.points : 2)

  if (isLog) {
    if (lo <= 0 || hi <= lo) return [0.01, 0.1, 1, 10, 100]
    const mantissas = logSeriesMantissas(ppd)
    const kMin = Math.floor(Math.log10(lo))
    const kMax = Math.ceil(Math.log10(hi))
    const all: number[] = []
    for (let k = kMin; k <= kMax; k++) {
      for (const m of mantissas) {
        const v = clean(m * 10 ** k)
        if (v >= lo && v <= hi) all.push(v)
      }
    }
    if (all.length <= 8) return all
    const step = Math.ceil(all.length / 7)
    const picked = all.filter((_, i) => i % step === 0 || i === all.length - 1)
    return Array.from(new Set(picked))
  } else {
    if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) return [1, 5, 10, 20, 50, 100]
    const step = o.step ?? (o.increment === 'lin' && o.points ? o.points : (hi - lo) / 5)
    const list: number[] = []
    for (let v = lo; v <= hi + step * 1e-6; v += step) {
      list.push(clean(o.type === 'int' ? Math.round(v) : v))
      if (list.length >= 10) break
    }
    return Array.from(new Set(list))
  }
}

/** Format the header display value: e.g. -2 (0.01) when log transform is active. */
export function formatHeaderValue(o: NumberOptions, value: number): string | null {
  if (typeof o.headerValue === 'function') return o.headerValue(value)
  if (typeof o.headerValue === 'string') return o.headerValue
  if (o.logTransform === false) return null

  const isLog = isLogScale(o)
  const logTransform = o.logTransform

  if (logTransform === true) {
    if (isLog) {
      if (value > 0) {
        const logVal = clean(Math.round(Math.log10(value) * 100) / 100)
        const realStr = formatNumberValue(o, value)
        return `${logVal} (${realStr})`
      }
    } else {
      const real = 10 ** value
      const realStr =
        real >= 0.001 && real < 1e4 ? String(Number(real.toPrecision(3))) : formatNumberValue({ scale: 'log10' }, real)
      return `${clean(value)} (${realStr})`
    }
  }

  if (logTransform === 'value-is-log') {
    const real = 10 ** value
    const realStr =
      real >= 0.001 && real < 1e4 ? String(Number(real.toPrecision(3))) : formatNumberValue({ scale: 'log10' }, real)
    return `${clean(value)} (${realStr})`
  }

  if (logTransform === 'value-is-real' || (isLog && logTransform !== undefined)) {
    if (value > 0) {
      const logVal = clean(Math.round(Math.log10(value) * 100) / 100)
      const realStr = formatNumberValue(o, value)
      return `${logVal} (${realStr})`
    }
  }

  return null
}
