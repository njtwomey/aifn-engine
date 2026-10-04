/**
 * Internal helpers shared by the distribution families: shapes, raw (untraced) evaluation, masks, a few elementwise
 * primitives that `aifn-compute/numerics/special` does not have, numerical inversion of cdfs, and the builder that turns a family's
 * specification into a `Univariate` object.
 */

import { standardNormals, units, type Stream } from 'aifn-compute/foundation/random'
import { DomainError, NotDifferentiableError } from 'aifn-compute/foundation/errors'
import {
  add,
  broadcastShapes,
  broadcastTo,
  div,
  elementwise,
  fromData,
  mul,
  square,
  isTensor,
  isTraced,
  item,
  shapeOfValue,
  sqrt,
  exp,
  log,
  log1p,
  neg,
  sub,
  sum,
  toFlat,
  unwrap,
  where,
  type Raw,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import type { ExponentialFamily, SampleOptions, Support, Univariate } from './types'

// ── Shapes and raw values ────────────────────────────────────────────────────────────────────────────────────────────

/** The broadcast shape of the parameters: the batch shape. */
export function batchShapeOf(...params: Value[]): number[] {
  return broadcastShapes(...params.map(shapeOfValue))
}

/** `v` broadcast to its broadcast shape with `others` (a batch mean when only the scale is batched, say). */
export function atBatch(v: Value, ...others: Value[]): Value {
  const batch = batchShapeOf(v, ...others)
  const shape = shapeOfValue(v)
  if (shape.length === batch.length && shape.every((d, k) => d === batch[k])) return v
  return broadcastTo(v, batch)
}

/** The untraced value of `v`; a traced value is an error naming `where` (the operation has no derivative). */
export function raw(v: Value, where: string): Raw {
  if (isTraced(v)) throw new NotDifferentiableError(where, `${where}: not differentiable (a traced value was passed)`)
  return v
}

/** Sum the last `k` axes of v (every axis, to a number, when k reaches the rank): reduces event axes. */
export function sumLastAxes(v: Value, k: number): Value {
  const rank = shapeOfValue(v).length
  if (k === 0) return v
  if (k >= rank) return sum(v)
  return sum(
    v,
    Array.from({ length: k }, (_, j) => rank - k + j),
  )
}

/** Sum over the last axis (to a number for a vector): reduces one event axis. */
export function sumLast(v: Value): Value {
  return sumLastAxes(v, 1)
}

/** True when any of the values is traced. */
export function anyTraced(...vs: Value[]): boolean {
  return vs.some(isTraced)
}

/** A number, or a tensor as a number when it has one element; for scalar-only code paths. */
export function scalarOf(v: Raw): number {
  return typeof v === 'number' ? v : item(v)
}

/**
 * Apply a scalar function elementwise to broadcast raw values (a number when all are numbers). For masks and for
 * quantities computed without derivatives (entropies by summation, cdfs by quadrature).
 */
export function rawMap(values: readonly Value[], f: (...xs: number[]) => number): Raw {
  const rs = values.map((v) => unwrap(v))
  if (rs.every((r) => typeof r === 'number')) return f(...(rs as number[]))
  const shape = broadcastShapes(...rs.map((r) => (typeof r === 'number' ? [] : r.shape)))
  const flats = rs.map((r) => (typeof r === 'number' ? null : toFlat(broadcastTo(r, shape))))
  const n = shape.reduce((a, b) => a * b, 1)
  const out = new Float64Array(n)
  const args = new Array<number>(rs.length)
  for (let k = 0; k < n; k++) {
    for (let j = 0; j < rs.length; j++) args[j] = flats[j] ? flats[j]![k] : (rs[j] as number)
    out[k] = f(...args)
  }
  return fromData(out, shape)
}

/** Like `rawMap`, but refuses traced inputs: for quantities with no derivative rule. */
export function rawOnly(where: string, values: readonly Value[], f: (...xs: number[]) => number): Raw {
  values.forEach((v) => raw(v, where))
  return rawMap(values, f)
}

// ── Masks ────────────────────────────────────────────────────────────────────────────────────────────────────────────

/** A mask (1 where true) of a raw predicate over broadcast values. */
export function mask(values: readonly Value[], test: (...xs: number[]) => boolean): Raw {
  return rawMap(values, (...xs) => (test(...xs) ? 1 : 0))
}

/**
 * Replace `x` by `safe` where `valid` is 0, before evaluating an expression that is only defined on the valid set. The
 * result is then masked again with `outside`. Guarding the argument (not only the result) keeps the derivative finite:
 * a NaN or ∞ in the unused branch of `where` would otherwise turn a zero cotangent into NaN.
 */
export function guard(x: Value, valid: Raw, safe: number): Value {
  if (valid === 1) return x
  return where(valid, x, safe)
}

/** `expr` where `valid`, and `fill` elsewhere. */
export function outside(valid: Raw, expr: Value, fill: number): Value {
  if (valid === 1) return expr
  return where(valid, expr, fill)
}

/** Is x an integer? */
export const isInteger = (x: number): boolean => Number.isFinite(x) && Math.floor(x) === x

// ── Primitives not in aifn-compute/special ───────────────────────────────────────────────────────────────────────────────────

/** arctan x, elementwise; d/dx = 1/(1 + x²). */
export const atan = elementwise({
  id: 'probability/distributions/atan',
  f: Math.atan,
  derivative: [(x) => div(1, add(1, square(x)))],
  doc: { summary: 'The arctangent.' },
})

/**
 * The standard Cauchy quantile tan(π(p − ½)), written as −1/tan(πp) below ½ and 1/tan(π(1 − p)) above (1 − p is exact
 * there), so that it keeps its relative accuracy as p → 0 or 1 (tan(π(p − ½)) saturates near −1.6e16 once p − ½
 * rounds to −½); its derivative is π(1 + y²).
 */
export const standardCauchyQuantile = elementwise({
  id: 'probability/distributions/standardCauchyQuantile',
  f: (p) => {
    if (!(p >= 0 && p <= 1)) return NaN
    if (p === 0.5) return 0
    return p < 0.5 ? -1 / Math.tan(Math.PI * p) : 1 / Math.tan(Math.PI * (1 - p))
  },
  derivative: [(_p, y) => mul(Math.PI, add(1, square(y)))],
  doc: { summary: 'The standard Cauchy quantile.' },
  test: { domain: { lo: 0.05, hi: 0.95 } },
})

/**
 * The standard Cauchy cdf 1/2 + arctan(z)/π, computed as arctan(−1/z)/π for z < 0 so that the lower tail keeps its
 * relative accuracy; its derivative is the density 1/(π(1 + z²)).
 */
export const standardCauchyCdf = elementwise({
  id: 'probability/distributions/standardCauchyCdf',
  f: (z) => (z < 0 ? Math.atan(-1 / z) / Math.PI : 0.5 + Math.atan(z) / Math.PI),
  derivative: [(z) => div(1 / Math.PI, add(1, square(z)))],
  doc: { summary: 'The standard Cauchy cdf.' },
})

/** The Euler–Mascheroni constant γ. */
export const EULER_GAMMA = 0.5772156649015329

export const LOG_2PI = Math.log(2 * Math.PI)

// ── Numerical inversion of a cdf ─────────────────────────────────────────────────────────────────────────────────────

/** The bounds of a univariate support as raw values (−∞ and ∞ for the real line). */
export function supportBounds(support: Support): [Value, Value] {
  if (support.type === 'interval' || support.type === 'integers' || support.type === 'circle')
    return [support.lower, support.upper]
  return [-Infinity, Infinity]
}

/**
 * The quantile of a univariate distribution by bracketing and bisection, elementwise over the broadcast of p and the
 * batch. Continuous: bisection to adjacent doubles (or a relative width of 4ε). Discrete: the smallest integer k with
 * cdf(k) ≥ p. p = 0 and p = 1 give the support's ends; p outside [0, 1] gives NaN. Not differentiable.
 *
 * With a `survival` function, probabilities above ½ invert it at 1 − p (exact there) instead of the cdf, so upper
 * quantiles keep the relative accuracy of the survival function rather than the absolute accuracy ε of a cdf near 1.
 */
export function invertCdf(
  name: string,
  cdf: (x: Tensor) => Value,
  p: Value,
  batchShape: readonly number[],
  support: Support,
  discrete: boolean,
  survival?: (x: Tensor) => Value,
): Raw {
  return invert(name, 'quantile', cdf, survival, p, batchShape, support, discrete)
}

/**
 * The inverse survival function by bracketing and bisection: the smallest x with survival(x) ≤ q, elementwise over the
 * broadcast of q and the batch. q = 1 and q = 0 give the support's ends. Not differentiable.
 */
export function invertSurvival(
  name: string,
  survival: (x: Tensor) => Value,
  q: Value,
  batchShape: readonly number[],
  support: Support,
  discrete: boolean,
): Raw {
  return invert(name, 'isf', undefined, survival, q, batchShape, support, discrete)
}

/**
 * Shared bisection. Each element inverts one tail: the cdf (reached when cdf(x) ≥ t) or the survival function
 * (reached when survival(x) ≤ t), keeping `lo` unreached and `hi` reached.
 */
function invert(
  name: string,
  what: 'quantile' | 'isf',
  cdf: ((x: Tensor) => Value) | undefined,
  survival: ((x: Tensor) => Value) | undefined,
  p: Value,
  batchShape: readonly number[],
  support: Support,
  discrete: boolean,
): Raw {
  const where_ = `${name}.${what}`
  const pr = raw(p, where_)
  const [lower, upper] = supportBounds(support).map((b) => raw(b, where_))
  const shape = broadcastShapes(shapeOfValue(pr), batchShape)
  const n = shape.reduce((a, b) => a * b, 1)
  const at = (v: Raw) =>
    typeof v === 'number' ? new Float64Array(n).fill(v) : Float64Array.from(toFlat(broadcastTo(v, shape)))
  const ps = at(pr)
  const L = at(lower)
  const U = at(upper)
  // Per element: the target, on the cdf (p) or on the survival side (1 − p for a quantile, q for isf).
  const target = new Float64Array(n)
  const upperTail = new Uint8Array(n)
  const out = new Float64Array(n)
  const active: boolean[] = []
  const lo = new Float64Array(n)
  const hi = new Float64Array(n)
  for (let k = 0; k < n; k++) {
    const q = ps[k]
    active[k] = false
    if (!(q >= 0 && q <= 1)) {
      out[k] = NaN
      continue
    }
    if (what === 'quantile' ? q === 0 : q === 1) out[k] = L[k]
    else if (what === 'quantile' ? q === 1 : q === 0) out[k] = U[k]
    else {
      active[k] = true
      const onSurvival = what === 'isf' || (survival !== undefined && !discrete && q > 0.5)
      upperTail[k] = onSurvival ? 1 : 0
      target[k] = onSurvival && what === 'quantile' ? 1 - q : q
      // Start from the finite end(s) of the support, or around 0 on the real line.
      lo[k] = Number.isFinite(L[k]) ? (discrete ? L[k] - 1 : L[k]) : Number.isFinite(U[k]) ? U[k] - 1 : -1
      hi[k] = Number.isFinite(U[k]) ? U[k] : Number.isFinite(L[k]) ? L[k] + 1 : 1
    }
  }
  const values = (f: ((x: Tensor) => Value) | undefined, xs: Float64Array) => {
    if (!f) return null
    const c = unwrap(f(fromData(xs, shape)))
    return typeof c === 'number' ? new Float64Array(n).fill(c) : Float64Array.from(toFlat(broadcastTo(c, shape)))
  }
  const anyLower = upperTail.some((u, k) => active[k] && u === 0)
  const anyUpper = upperTail.some((u, k) => active[k] && u === 1)
  // reached[k] is true when x has reached the target in element k's tail.
  const reached = (xs: Float64Array) => {
    const c = anyLower ? values(cdf, xs) : null
    const s = anyUpper ? values(survival, xs) : null
    return (k: number) => (upperTail[k] ? s![k] <= target[k] : c![k] >= target[k])
  }
  // Expand the bracket until lo is unreached and hi reached; an infinite end is never evaluated.
  for (let it = 0, step = 1; it < 1100; it++, step *= 2) {
    const rl = reached(lo)
    const rh = reached(hi)
    let moved = false
    for (let k = 0; k < n; k++) {
      if (!active[k]) continue
      if (rl(k) && lo[k] > L[k]) {
        lo[k] = Math.max(L[k], lo[k] - step)
        moved = true
      }
      if (!rh(k) && hi[k] < U[k]) {
        hi[k] = Math.min(U[k], hi[k] + step)
        moved = true
      }
    }
    if (!moved) break
  }
  // Bisection, keeping lo unreached and hi reached.
  for (let it = 0; it < 3000; it++) {
    const mid = Float64Array.from(lo)
    const live = new Uint8Array(n)
    let open = false
    for (let k = 0; k < n; k++) {
      if (!active[k]) continue
      if (discrete) {
        if (hi[k] - lo[k] <= 1) continue
        mid[k] = Math.floor((lo[k] + hi[k]) / 2)
      } else {
        const m = lo[k] + (hi[k] - lo[k]) / 2
        const width = 4 * Number.EPSILON * Math.max(Math.abs(lo[k]), Math.abs(hi[k]))
        if (m <= lo[k] || m >= hi[k] || hi[k] - lo[k] <= width) continue
        mid[k] = m
      }
      live[k] = 1
      open = true
    }
    if (!open) break
    const rm = reached(mid)
    for (let k = 0; k < n; k++) {
      if (!live[k]) continue
      if (rm(k)) hi[k] = mid[k]
      else lo[k] = mid[k]
    }
  }
  for (let k = 0; k < n; k++) if (active[k]) out[k] = discrete ? hi[k] : lo[k] + (hi[k] - lo[k]) / 2
  return shape.length === 0 && typeof pr === 'number' ? out[0] : fromData(out, shape)
}

/**
 * log p for a probability p whose complement q = 1 − p is computed separately: log p where p ≤ ½, and log1p(−q)
 * above, so that the result keeps its relative accuracy at both ends (log p alone gives 0 once p rounds to 1).
 * `logP`, when given, replaces log p in the lower branch (a log-space form that does not underflow). Differentiable
 * through whichever branch is used.
 */
export function logFromTails(p: Value, q: Value, logP?: Value): Value {
  const low = mask([p], (v) => !(v > 0.5))
  const high = mask([p], (v) => v > 0.5)
  const lower = logP ?? log(guard(p, low, 0.25))
  if (low === 1) return lower
  if (high === 1) return log1p(neg(q))
  return where(low, lower, log1p(neg(guard(q, high, 0.25))))
}

// ── The builder ──────────────────────────────────────────────────────────────────────────────────────────────────────

/** A family's specification; everything else a `Univariate` needs is derived from it. */
export type UnivariateSpec = {
  name: string
  params: Record<string, Value>
  /** Default: the parameters' broadcast shape (families with a parameter vector per member override it). */
  batchShape?: readonly number[]
  support: Support
  discrete?: boolean
  logProb(x: Value): Value
  cdf(x: Value): Value
  /** Default `logFromTails(cdf, survival)`: log cdf where cdf ≤ ½, log1p(−survival) above. */
  logcdf?(x: Value): Value
  /** Default 1 − cdf. Give it whenever it can be computed directly: the log tails and upper quantiles rely on it. */
  survival?(x: Value): Value
  /** Default `logFromTails(survival, cdf)`. */
  logSurvival?(x: Value): Value
  /**
   * Default: numerical inversion of the cdf, or of the survival function above p = ½ when one is given (not
   * differentiable).
   */
  quantile?(p: Value): Value
  /** The inverse survival function. Default: numerical inversion of the survival function (not differentiable). */
  isf?(q: Value): Value
  /**
   * Draws of the full shape `[...sampleShape, ...batchShape]`, from raw parameters. Optional when `rsample` is given:
   * the draw is then `rsample` on the raw parameters (the same values).
   */
  sample?(s: Stream, shape: number[]): Tensor
  /**
   * A pathwise (reparameterised) draw written with primitives, so it is differentiable in the parameters: of the full
   * shape, or a number (or traced number) when `scalar` is true. Absent where no pathwise draw exists.
   */
  rsample?(s: Stream, shape: number[], scalar: boolean): Value
  mean(): Value
  variance(): Value
  entropy(): Value
  mode(): Value
  expFamily?: ExponentialFamily
}

/** Draws of a batch: the sample shape and the batch shape, and whether to return a number. */
export function drawShape(
  batchShape: readonly number[],
  options?: SampleOptions,
): { shape: number[]; scalar: boolean } {
  const shape = [...(options?.shape ?? []), ...batchShape]
  return { shape, scalar: options?.shape === undefined && batchShape.length === 0 }
}

/** Build a `Univariate` from a specification. */
export function univariate<P extends Value>(spec: UnivariateSpec): Univariate<P> {
  const batchShape = spec.batchShape ?? batchShapeOf(...Object.values(spec.params))
  const discrete = spec.discrete ?? false
  const survival = (x: Value) => (spec.survival ? spec.survival(x) : sub(1, spec.cdf(x)))
  const self = {
    kind: 'distribution' as const,
    name: spec.name,
    params: spec.params,
    batchShape,
    eventShape: [] as const,
    support: spec.support,
    discrete,
    expFamily: spec.expFamily,
    logProb: (x: Value) => spec.logProb(x),
    prob: (x: Value) => exp(spec.logProb(x)),
    cdf: (x: Value) => spec.cdf(x),
    logcdf: (x: Value) => {
      if (spec.logcdf) return spec.logcdf(x)
      // Without a direct survival function, 1 − cdf carries no more information than the cdf.
      return spec.survival ? logFromTails(spec.cdf(x), spec.survival(x)) : log(spec.cdf(x))
    },
    survival,
    logSurvival: (x: Value) => {
      if (spec.logSurvival) return spec.logSurvival(x)
      return spec.survival ? logFromTails(spec.survival(x), spec.cdf(x)) : log1p(neg(spec.cdf(x)))
    },
    quantile: (p: Value) =>
      spec.quantile
        ? spec.quantile(p)
        : invertCdf(spec.name, (x) => spec.cdf(x), p, batchShape, spec.support, discrete, spec.survival),
    isf: (q: Value) =>
      spec.isf ? spec.isf(q) : invertSurvival(spec.name, survival, q, batchShape, spec.support, discrete),
    sample: (s: Stream, options?: SampleOptions) => {
      for (const [k, v] of Object.entries(spec.params)) raw(v, `${spec.name}.sample (parameter ${k})`)
      const { shape, scalar } = drawShape(batchShape, options)
      if (spec.sample) {
        const t = spec.sample(s, shape)
        return scalar ? item(t) : t
      }
      const r = unwrap(spec.rsample!(s, shape, scalar))
      return scalar || !isTensor(r) ? r : broadcastTo(r, shape)
    },
    ...(spec.rsample
      ? {
          rsample: (s: Stream, options?: SampleOptions) => {
            const { shape, scalar } = drawShape(batchShape, options)
            return spec.rsample!(s, shape, scalar)
          },
        }
      : {}),
    mean: () => spec.mean(),
    variance: () => spec.variance(),
    stddev: () => sqrt(spec.variance()),
    entropy: () => spec.entropy(),
    mode: () => spec.mode(),
  }
  return self as unknown as Univariate<P>
}

/** Base variates of a draw: a number when `scalar`, else a tensor of `shape` filled row-major. */
function variates(values: Float64Array, shape: number[], scalar: boolean): number | Tensor {
  return scalar ? values[0] : fromData(values, shape)
}

/** Uniforms in [0, 1) for a draw of `shape` (a number when `scalar`), from one block of the stream. */
export function unitsOf(s: Stream, shape: number[], scalar: boolean): number | Tensor {
  return variates(units(s, scalar ? 1 : shape.reduce((a, b) => a * b, 1)), shape, scalar)
}

/** Standard normals for a draw of `shape` (a number when `scalar`), from one block of the stream. */
export function normalsOf(s: Stream, shape: number[], scalar: boolean): number | Tensor {
  return variates(standardNormals(s, scalar ? 1 : shape.reduce((a, b) => a * b, 1)), shape, scalar)
}

/**
 * A location–scale pathwise draw loc + scale · z with z standard normal (Kingma and Welling 2014, "Auto-encoding
 * variational Bayes", §2.4): differentiable in loc and scale.
 */
export function locationScale(s: Stream, shape: number[], scalar: boolean, loc: Value, scale: Value): Value {
  return add(loc, mul(scale, normalsOf(s, shape, scalar)))
}

/**
 * Inverse-transform sampling as a pathwise draw: uniforms of the draw's shape mapped through a quantile function
 * written with primitives, so the draw is differentiable in the parameters the quantile uses (Devroye 1986,
 * "Non-Uniform Random Variate Generation", §2.1).
 */
export function inverseTransform(s: Stream, shape: number[], scalar: boolean, quantile: (u: Value) => Value): Value {
  return quantile(unitsOf(s, shape, scalar))
}

/**
 * Check a parameter's raw values (traced values are checked through their underlying value): throws a `DomainError`
 * naming the family, the parameter and the requirement when any element fails `test`.
 */
export function check(family: string, label: string, v: Value, test: (x: number) => boolean, what: string): void {
  const r = unwrap(v)
  const values = typeof r === 'number' ? [r] : toFlat(r)
  for (const x of values) if (!test(x)) throw new DomainError(family, `${family}: ${label} must be ${what}, got ${x}`)
}
