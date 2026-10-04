/**
 * Domains of observations and actions (`Domain`, docs/aifn-gym.md §3): constructors (`discreteDomain`, `boxDomain`)
 * and the helpers every agent and rollout needs: `domainContains`, `clipToDomain` (the nearest point of the domain),
 * `domainSize` (the number of values, `Infinity` for a box), `domainDimension` (the numbers in one value) and a
 * uniform draw `sampleDomain`.
 */

import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import type { BoxDomain, DiscreteDomain, Domain, Shape, Size } from 'aifn-compute/foundation/contracts'
import { integers, uniform, type Stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'

/** The discrete domain 0 … n − 1, with optional display names (one per value). */
export function discreteDomain(n: Size, names?: readonly string[]): DiscreteDomain {
  if (!(Number.isInteger(n) && n >= 1))
    throw new DomainError('discreteDomain', `discreteDomain: n must be a positive integer, got ${n}`)
  if (names && names.length !== n)
    throw new DomainError('discreteDomain', `discreteDomain: ${names.length} names for ${n} values`)
  return names ? { kind: 'discrete', n, names } : { kind: 'discrete', n }
}

/**
 * The box [low, high] of arrays of `shape` (default: a vector as long as the bounds). A number for a bound applies to
 * every element; arrays are flat, row-major. Bounds may be infinite but need low ≤ high.
 *
 * @example boxDomain([-2], [2]) // a torque in [−2, 2]
 */
export function boxDomain(
  low: number | readonly number[],
  high: number | readonly number[],
  { shape, names }: { shape?: Shape; names?: readonly string[] } = {},
): BoxDomain {
  const fromBounds = Array.isArray(low) ? low.length : Array.isArray(high) ? high.length : undefined
  const sh: Shape = shape ?? (fromBounds === undefined ? [] : [fromBounds])
  const size = sh.reduce((a, b) => a * b, 1)
  const expand = (b: number | readonly number[], which: string) => {
    if (typeof b === 'number') return Array.from({ length: size }, () => b)
    if (b.length !== size)
      throw new ShapeError('boxDomain', `boxDomain: ${which} has ${b.length} values for shape [${sh.join(', ')}]`)
    return [...b]
  }
  const lo = expand(low, 'low')
  const hi = expand(high, 'high')
  lo.forEach((l, i) => {
    if (!(l <= hi[i])) throw new DomainError('boxDomain', `boxDomain: needs low ≤ high, got [${l}, ${hi[i]}] at ${i}`)
  })
  if (names && names.length !== size)
    throw new DomainError('boxDomain', `boxDomain: ${names.length} names for ${size} elements`)
  return names ? { kind: 'box', low: lo, high: hi, shape: sh, names } : { kind: 'box', low: lo, high: hi, shape: sh }
}

/** The elements of a box value as a flat array, or null when x is not numeric or has the wrong length. */
function boxValues(domain: BoxDomain, x: unknown): ArrayLike<number> | null {
  const v = typeof x === 'number' ? [x] : x
  if (!(v instanceof Float64Array || Array.isArray(v))) return null
  return v.length === domain.low.length ? (v as ArrayLike<number>) : null
}

/** True when `x` is a value of the domain: an integer in range, or an array of the box's size inside its bounds. */
export function domainContains(domain: Domain, x: unknown): boolean {
  if (domain.kind === 'discrete') return typeof x === 'number' && Number.isInteger(x) && x >= 0 && x < domain.n
  const v = boxValues(domain, x)
  if (!v) return false
  for (let i = 0; i < v.length; i++)
    if (typeof v[i] !== 'number' || !(v[i] >= domain.low[i] && v[i] <= domain.high[i])) return false
  return true
}

/**
 * The nearest value of the domain (Gymnasium's action clipping): a box clips each element to its bounds; a discrete
 * domain rounds and clips to 0 … n − 1.
 */
export function clipToDomain(domain: DiscreteDomain, x: number): number
export function clipToDomain(domain: BoxDomain, x: ArrayLike<number>): Float64Array
export function clipToDomain(domain: Domain, x: number | ArrayLike<number>): number | Float64Array
export function clipToDomain(domain: Domain, x: number | ArrayLike<number>): number | Float64Array {
  if (domain.kind === 'discrete') {
    if (typeof x !== 'number') throw new ShapeError('clipToDomain', 'clipToDomain: a discrete domain takes a number')
    return Math.min(domain.n - 1, Math.max(0, Math.round(x)))
  }
  const v = boxValues(domain, x)
  if (!v) throw new ShapeError('clipToDomain', `clipToDomain: expected ${domain.low.length} values`)
  return Float64Array.from(v, (e, i) => Math.min(domain.high[i], Math.max(domain.low[i], e)))
}

/** The number of values of a domain: n for a discrete domain, `Infinity` for a box (continuous). */
export function domainSize(domain: Domain): Size {
  return domain.kind === 'discrete' ? domain.n : Infinity
}

/** The numbers in one value: 1 for a discrete domain, the product of the shape for a box. */
export function domainDimension(domain: Domain): Size {
  return domain.kind === 'discrete' ? 1 : domain.low.length
}

/**
 * A uniform draw from the domain: an integer in 0 … n − 1, or a flat array uniform on the box (whose bounds must then
 * be finite).
 */
export function sampleDomain(s: Stream, domain: DiscreteDomain): number
export function sampleDomain(s: Stream, domain: BoxDomain): Float64Array
export function sampleDomain(s: Stream, domain: Domain): number | Float64Array
export function sampleDomain(s: Stream, domain: Domain): number | Float64Array {
  if (domain.kind === 'discrete') return integers(s, domain.n)
  const n = domain.low.length
  domain.low.forEach((lo, i) => {
    if (!(Number.isFinite(lo) && Number.isFinite(domain.high[i])))
      throw new DomainError('sampleDomain', `sampleDomain: element ${i} of the box is unbounded`)
  })
  // One block of n draws, so the elements are independent.
  const u = toFlat(uniform(s, 0, 1, { shape: [n] }))
  const out = Float64Array.from(u, (v, i) => domain.low[i] + (domain.high[i] - domain.low[i]) * v)
  return out
}
