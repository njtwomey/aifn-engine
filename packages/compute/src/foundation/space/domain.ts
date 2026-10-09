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

/**
 * The discrete domain $\{0, \dots, n - 1\}$, with optional display names (one per value). Throws `DomainError` when
 * `n` is not a positive integer or the names do not number `n`.
 *
 * @param n The number of values, a positive integer.
 * @param names A display name for each value, in order (value `i` is `names[i]`). Left out, the domain has none.
 * @returns The domain `{ kind: 'discrete', n }`, with `names` when given.
 *
 * @example Three moves with names
 * const moves = discreteDomain(3, ['left', 'stay', 'right'])
 * print('domain:', moves)
 * print('values:', domainSize(moves))
 */
export function discreteDomain(n: Size, names?: readonly string[]): DiscreteDomain {
  if (!(Number.isInteger(n) && n >= 1))
    throw new DomainError('discreteDomain', `discreteDomain: n must be a positive integer, got ${n}`)
  if (names && names.length !== n)
    throw new DomainError('discreteDomain', `discreteDomain: ${names.length} names for ${n} values`)
  return names ? { kind: 'discrete', n, names } : { kind: 'discrete', n }
}

/**
 * The box $[\text{low}, \text{high}]$ of arrays of `shape` (default: a vector as long as the bounds, or a scalar
 * when both bounds are numbers). A number for a bound applies to every element; arrays are flat, row-major. Bounds may
 * be infinite but need $\text{low} \le \text{high}$ elementwise, or `DomainError` is thrown; a bound array of the
 * wrong length throws `ShapeError`.
 *
 * @param low The lower bounds: a number for every element, or a flat row-major array with one per element.
 * @param high The upper bounds, in the same form as `low`.
 * @param options The shape of a value and the display names of its elements; both optional.
 * @param options.shape The shape of one value, whose elements the bounds list in row-major order. Left out, it is
 *   `[k]` for a bound array of length $k$ (`low`'s when both are arrays), or `[]` when both bounds are numbers.
 * @param options.names A display name per element, in row-major order; there must be as many as the elements.
 * @returns The domain `{ kind: 'box', low, high, shape }` with both bounds expanded to flat arrays of one value per
 *   element, and `names` when given.
 *
 * @example A torque in an interval
 * print(boxDomain([-2], [2]))
 *
 * @example One bound for every element of a matrix
 * print(boxDomain(-1, 1, { shape: [2, 2] }))
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

/**
 * The elements of a box value as a flat array, or null when `x` is not numeric or has the wrong length.
 *
 * @param domain The box the value should belong to; only its number of elements is read.
 * @param x A candidate value: a number (taken as one element), a plain array or a `Float64Array`.
 * @returns `x` itself as a list of numbers (a number wrapped in an array), or null when it is of another type or does
 *   not have one entry per element of the box.
 */
function boxValues(domain: BoxDomain, x: unknown): ArrayLike<number> | null {
  const v = typeof x === 'number' ? [x] : x
  if (!(v instanceof Float64Array || Array.isArray(v))) return null
  return v.length === domain.low.length ? (v as ArrayLike<number>) : null
}

/**
 * True when `x` is a value of the domain: an integer in range, or an array of the box's size inside its bounds.
 *
 * @param domain The domain to test against.
 * @param x Any value. For a discrete domain it must be an integer in $\{0, \dots, n - 1\}$; for a box, a plain array
 *   or `Float64Array` with one number per element (or a number, for a box of one element), each within its bounds.
 * @returns Whether `x` belongs to the domain; never throws.
 *
 * @example Membership in a discrete domain and a box
 * const moves = discreteDomain(3)
 * print('2 in moves:', domainContains(moves, 2))
 * print('3 in moves:', domainContains(moves, 3))
 * print('1.5 in moves:', domainContains(moves, 1.5))
 * const torque = boxDomain([-2], [2])
 * print('[0.5] in torque:', domainContains(torque, [0.5]))
 * print('[3] in torque:', domainContains(torque, [3]))
 */
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
 * domain rounds and clips to $\{0, \dots, n - 1\}$. Throws `ShapeError` when `x` is not a number for a discrete
 * domain, or does not have one number per element of a box.
 *
 * @param domain The domain to clip into.
 * @param x The value to clip: a number for a discrete domain; for a box, an array with one number per element (or a
 *   number, for a box of one element). It is not modified.
 * @returns The clipped value: a number for a discrete domain, a new `Float64Array` for a box.
 *
 * @example Clip an action into a box and a discrete domain
 * print('box:', clipToDomain(boxDomain([-2, 0], [2, 1]), [3, -1]))
 * print('discrete, 4.6:', clipToDomain(discreteDomain(3), 4.6))
 * print('discrete, 0.6:', clipToDomain(discreteDomain(3), 0.6))
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

/**
 * The number of values of a domain: $n$ for a discrete domain, `Infinity` for a box (continuous).
 *
 * @param domain The domain.
 * @returns The number of distinct values.
 *
 * @example Finite and continuous
 * print('discrete:', domainSize(discreteDomain(4)))
 * print('box:', domainSize(boxDomain([-1], [1])))
 */
export function domainSize(domain: Domain): Size {
  return domain.kind === 'discrete' ? domain.n : Infinity
}

/**
 * The numbers in one value: 1 for a discrete domain, the product of the shape for a box.
 *
 * @param domain The domain.
 * @returns How many numbers make up one value (the length of a flattened box value).
 *
 * @example One number per discrete value, one per element of a box
 * print('discrete:', domainDimension(discreteDomain(4)))
 * print('2 x 2 box:', domainDimension(boxDomain(-1, 1, { shape: [2, 2] })))
 */
export function domainDimension(domain: Domain): Size {
  return domain.kind === 'discrete' ? 1 : domain.low.length
}

/**
 * A uniform draw from the domain: an integer in $\{0, \dots, n - 1\}$, or a flat array uniform on the box (whose
 * bounds must then be finite, or `DomainError` is thrown).
 *
 * @param s The stream to draw from.
 * @param domain The domain to draw from.
 * @returns A number for a discrete domain; a new `Float64Array` of one draw per element, in row-major order, for a box.
 *
 * @example Random actions
 * print('discrete:', sampleDomain(stream(0), discreteDomain(3)))
 * print('box:', sampleDomain(stream(0), boxDomain([-2, 0], [2, 1])))
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
