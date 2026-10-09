/**
 * Internal helpers shared by the distribution families: shapes, raw (untraced) evaluation, masks, a few elementwise
 * primitives that `aifn-compute/numerics/special` does not have, numerical inversion of cdfs, base variates for draws,
 * parameter checks, and `univariate`, the builder that turns a family's specification into a `Univariate` object.
 *
 * Values are numbers, tensors or traced values throughout. A raw value is one that is not traced; masks are raw
 * (computed without derivatives) and select between branches with `where`, and an argument is guarded before an
 * expression defined only on part of its range, so that the unused branch never puts NaN into a derivative.
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

/**
 * The broadcast shape of the parameters: the batch shape.
 *
 * @param params The parameters, numbers, tensors or traced values.
 * @returns Their broadcast shape (NumPy rules), `[]` when every parameter is a number.
 */
export function batchShapeOf(...params: Value[]): number[] {
  return broadcastShapes(...params.map(shapeOfValue))
}

/**
 * `v` broadcast to its broadcast shape with `others` (a batch mean when only the scale is batched, say).
 *
 * @param v The value to broadcast.
 * @param others The values whose shapes it is broadcast with (the other parameters); they are not read otherwise.
 * @returns `v` itself when it already has the broadcast shape, else `v` broadcast to it.
 */
export function atBatch(v: Value, ...others: Value[]): Value {
  const batch = batchShapeOf(v, ...others)
  const shape = shapeOfValue(v)
  if (shape.length === batch.length && shape.every((d, k) => d === batch[k])) return v
  return broadcastTo(v, batch)
}

/**
 * The untraced value of `v`; a traced value is an error naming `where` (the operation has no derivative).
 *
 * @param v The value: a number or tensor is returned as it is; a traced value throws a `NotDifferentiableError`.
 * @param where The caller's name for the error message, such as `'Gamma.quantile'`.
 * @returns `v`, as a raw value.
 */
export function raw(v: Value, where: string): Raw {
  if (isTraced(v)) throw new NotDifferentiableError(where, `${where}: not differentiable (a traced value was passed)`)
  return v
}

/**
 * Sum the last `k` axes of `v` (every axis, to a number, when `k` reaches the rank): reduces event axes.
 *
 * @param v The value to sum.
 * @param k The number of trailing axes to sum over; 0 returns `v` unchanged.
 * @returns The sum, with the leading axes of `v` (the batch axes) kept.
 */
export function sumLastAxes(v: Value, k: number): Value {
  const rank = shapeOfValue(v).length
  if (k === 0) return v
  if (k >= rank) return sum(v)
  return sum(
    v,
    Array.from({ length: k }, (_, j) => rank - k + j),
  )
}

/**
 * Sum over the last axis (to a number for a vector): reduces one event axis.
 *
 * @param v The value to sum; a number is returned as it is.
 * @returns The sum over the last axis of `v`.
 */
export function sumLast(v: Value): Value {
  return sumLastAxes(v, 1)
}

/**
 * True when any of the values is traced.
 *
 * @param vs The values to test.
 * @returns True when at least one is traced (false for none).
 */
export function anyTraced(...vs: Value[]): boolean {
  return vs.some(isTraced)
}

/**
 * A number, or a tensor as a number when it has one element; for scalar-only code paths.
 *
 * @param v A number, or a tensor of one element (read with `item`, which throws for any other size).
 * @returns The number.
 */
export function scalarOf(v: Raw): number {
  return typeof v === 'number' ? v : item(v)
}

/**
 * Apply a scalar function elementwise to broadcast raw values (a number when all are numbers). For masks and for
 * quantities computed without derivatives (entropies by summation, cdfs by quadrature).
 *
 * @param values The arguments, broadcast together; traced values are unwrapped to their underlying values (no
 *   derivative is recorded).
 * @param f The scalar function, called once per element with one number from each of `values`, in order.
 * @returns The number `f(...values)` when every value is a number, else a tensor of the broadcast shape.
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

/**
 * Like `rawMap`, but refuses traced inputs: for quantities with no derivative rule.
 *
 * @param where The caller's name for the error message, thrown as a `NotDifferentiableError` when any of `values` is
 *   traced.
 * @param values The arguments, broadcast together.
 * @param f The scalar function, called once per element with one number from each of `values`, in order.
 * @returns As `rawMap`: a number when every value is a number, else a tensor of the broadcast shape.
 */
export function rawOnly(where: string, values: readonly Value[], f: (...xs: number[]) => number): Raw {
  values.forEach((v) => raw(v, where))
  return rawMap(values, f)
}

// ── Masks ────────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * A mask (1 where true, 0 elsewhere) of a raw predicate over broadcast values.
 *
 * @param values The values the predicate reads, broadcast together; traced values are unwrapped.
 * @param test The predicate, called once per element with one number from each of `values`, in order.
 * @returns The number 1 or 0 when every value is a number, else a tensor of 1s and 0s of the broadcast shape.
 */
export function mask(values: readonly Value[], test: (...xs: number[]) => boolean): Raw {
  return rawMap(values, (...xs) => (test(...xs) ? 1 : 0))
}

/**
 * Replace `x` by `safe` where `valid` is 0, before evaluating an expression that is only defined on the valid set. The
 * result is then masked again with `outside`. Guarding the argument (not only the result) keeps the derivative finite:
 * a NaN or $\infty$ in the unused branch of `where` would otherwise turn a zero cotangent into NaN.
 *
 * @param x The argument to guard.
 * @param valid The mask of the valid set, as `mask` returns it; the number 1 returns `x` unchanged.
 * @param safe A value inside the expression's domain, used where `valid` is 0.
 * @returns `x` where `valid` is 1 and `safe` elsewhere, differentiable in `x` on the valid set.
 */
export function guard(x: Value, valid: Raw, safe: number): Value {
  if (valid === 1) return x
  return where(valid, x, safe)
}

/**
 * `expr` where `valid`, and `fill` elsewhere.
 *
 * @param valid The mask of the valid set, as `mask` returns it; the number 1 returns `expr` unchanged.
 * @param expr The value on the valid set.
 * @param fill The value elsewhere, such as $-\infty$ for a log-density outside the support.
 * @returns `expr` with its invalid elements replaced by `fill`.
 */
export function outside(valid: Raw, expr: Value, fill: number): Value {
  if (valid === 1) return expr
  return where(valid, expr, fill)
}

/**
 * Is `x` an integer?
 *
 * @param x The number to test.
 * @returns True when `x` is finite and has no fractional part.
 */
export const isInteger = (x: number): boolean => Number.isFinite(x) && Math.floor(x) === x

// ── Primitives not in aifn-compute/special ───────────────────────────────────────────────────────────────────────────────────

/** $\arctan x$, elementwise; its derivative is $1/(1 + x^2)$. */
export const atan = elementwise({
  id: 'probability/distributions/atan',
  f: Math.atan,
  derivative: [(x) => div(1, add(1, square(x)))],
  doc: { summary: 'The arctangent.' },
})

/**
 * The standard Cauchy quantile $\tan(\pi(p - 1/2))$, elementwise, written as $-1/\tan(\pi p)$ below $1/2$ and
 * $1/\tan(\pi(1 - p))$ above ($1 - p$ is exact there), so that it keeps its relative accuracy as $p \to 0$ or 1
 * ($\tan(\pi(p - 1/2))$ saturates near $-1.6 \times 10^{16}$ once $p - 1/2$ rounds to $-1/2$); its derivative is
 * $\pi(1 + y^2)$ for the quantile $y$. NaN for $p$ outside $[0, 1]$.
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
 * The standard Cauchy cdf $1/2 + \arctan(z)/\pi$, elementwise, computed as $\arctan(-1/z)/\pi$ for $z < 0$ so that the
 * lower tail keeps its relative accuracy; its derivative is the density $1/(\pi(1 + z^2))$.
 */
export const standardCauchyCdf = elementwise({
  id: 'probability/distributions/standardCauchyCdf',
  f: (z) => (z < 0 ? Math.atan(-1 / z) / Math.PI : 0.5 + Math.atan(z) / Math.PI),
  derivative: [(z) => div(1 / Math.PI, add(1, square(z)))],
  doc: { summary: 'The standard Cauchy cdf.' },
})

/** The Euler–Mascheroni constant $\gamma$. */
export const EULER_GAMMA = 0.5772156649015329

/** $\log 2\pi$, the constant of the normal log-density. */
export const LOG_2PI = Math.log(2 * Math.PI)

// ── Numerical inversion of a cdf ─────────────────────────────────────────────────────────────────────────────────────

/**
 * The bounds of a univariate support ($-\infty$ and $\infty$ for the real line).
 *
 * @param support The support, as a distribution's `support` holds it.
 * @returns The lower and upper bounds of an interval, integer or circle support (as stored, possibly tensors or traced
 *   values), else $[-\infty, \infty]$.
 */
export function supportBounds(support: Support): [Value, Value] {
  if (support.type === 'interval' || support.type === 'integers' || support.type === 'circle')
    return [support.lower, support.upper]
  return [-Infinity, Infinity]
}

/**
 * The quantile of a univariate distribution by bracketing and bisection, elementwise over the broadcast of $p$ and the
 * batch. Continuous: bisection to adjacent doubles (or a relative width of $4\varepsilon$). Discrete: the smallest
 * integer $k$ with $F(k) \ge p$. $p = 0$ and $p = 1$ give the support's ends; $p$ outside $[0, 1]$ gives NaN. Not
 * differentiable.
 *
 * For a continuous distribution with a `survival` function, probabilities above $1/2$ invert it at $1 - p$ (exact
 * there) instead of the cdf, so upper quantiles keep the relative accuracy of the survival function rather than the
 * absolute accuracy $\varepsilon$ of a cdf near 1.
 *
 * @param name The family's name, for error messages (as `name.quantile`).
 * @param cdf The cdf, called with a tensor of the broadcast shape of `p` and the batch and returning values of (or
 *   broadcastable to) that shape.
 * @param p The probabilities; a traced value throws a `NotDifferentiableError`.
 * @param batchShape The distribution's batch shape, broadcast with the shape of `p`.
 * @param support The support, whose ends start the bracket and are returned for $p = 0$ and $p = 1$.
 * @param discrete Whether the support is the integers (the search is then over integers).
 * @param survival The survival function, called like `cdf`; when left out, every quantile inverts the cdf.
 * @returns The quantiles: a number when `p` is a number and the batch shape is `[]`, else a tensor of the broadcast
 *   shape.
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
 * The inverse survival function by bracketing and bisection: the smallest $x$ with $S(x) \le q$, elementwise over the
 * broadcast of $q$ and the batch. $q = 1$ and $q = 0$ give the support's ends; $q$ outside $[0, 1]$ gives NaN. Not
 * differentiable.
 *
 * @param name The family's name, for error messages (as `name.isf`).
 * @param survival The survival function $S$, called with a tensor of the broadcast shape of `q` and the batch.
 * @param q The upper-tail probabilities; a traced value throws a `NotDifferentiableError`.
 * @param batchShape The distribution's batch shape, broadcast with the shape of `q`.
 * @param support The support, whose ends start the bracket and are returned for $q = 1$ and $q = 0$.
 * @param discrete Whether the support is the integers (the search is then over integers).
 * @returns The points: a number when `q` is a number and the batch shape is `[]`, else a tensor of the broadcast shape.
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
 * Shared bisection of `invertCdf` and `invertSurvival`. Each element inverts one tail: the cdf (reached when
 * $F(x) \ge t$) or the survival function (reached when $S(x) \le t$), keeping `lo` unreached and `hi` reached. The
 * bracket starts at the support's finite ends (or around 0 on the real line) and doubles its step outwards.
 *
 * @param name The family's name; errors name `name.what`.
 * @param what Which inverse: `'quantile'` (of $p$, by the cdf, or by the survival function above $1/2$) or `'isf'`
 *   (of $q$, by the survival function).
 * @param cdf The cdf; needed for `'quantile'`, unused for `'isf'`.
 * @param survival The survival function; needed for `'isf'`, optional for `'quantile'`.
 * @param p The probabilities to invert; a traced value throws a `NotDifferentiableError`.
 * @param batchShape The distribution's batch shape, broadcast with the shape of `p`.
 * @param support The support, whose ends start the bracket and are the results at probabilities 0 and 1.
 * @param discrete Whether the support is the integers (the search is then over integers, and only the cdf is used for
 *   a quantile).
 * @returns The inverses: a number when `p` is a number and the batch shape is `[]`, else a tensor of the broadcast
 *   shape.
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
 * $\log p$ for a probability $p$ whose complement $q = 1 - p$ is computed separately: $\log p$ where $p \le 1/2$, and
 * $\operatorname{log1p}(-q)$ above, so that the result keeps its relative accuracy at both ends ($\log p$ alone gives 0
 * once $p$ rounds to 1). Differentiable through whichever branch is used.
 *
 * @param p The probability $p$, which selects the branch elementwise.
 * @param q Its complement $1 - p$, computed directly (a survival function for a cdf, say).
 * @param logP A log-space form of $\log p$ that does not underflow, used in the lower branch instead of $\log p$;
 *   when left out, $\log p$ is taken.
 * @returns $\log p$, with the broadcast shape of the inputs.
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

/**
 * A family's specification, as `univariate` takes it; everything else a `Univariate` needs is derived from it. Each
 * function takes values of any shape broadcastable with the batch and is written with primitives, so that it is
 * differentiable where its primitives are.
 */
export type UnivariateSpec = {
  /** The family's name, such as `'Normal'`; it names the family in error messages. */
  name: string
  /** The parameters by name, as given (numbers, tensors or traced values); public as the distribution's `params`. */
  params: Record<string, Value>
  /** Default: the parameters' broadcast shape (families with a parameter vector per member override it). */
  batchShape?: readonly number[]
  /** The support, whose ends also bound the numerical inversion of the cdf. */
  support: Support
  /** True for a distribution on the integers (default false). */
  discrete?: boolean
  /** The log-density (or log-mass) at `x`, $-\infty$ outside the support. */
  logProb(x: Value): Value
  /** The cdf $F(x) = P(X \le x)$. */
  cdf(x: Value): Value
  /**
   * The log cdf. Default `logFromTails(cdf, survival)`: $\log F$ where $F \le 1/2$ and
   * $\operatorname{log1p}(-S)$ above, or $\log F$ alone when there is no `survival`.
   */
  logcdf?(x: Value): Value
  /**
   * The survival function $S(x) = 1 - F(x)$. Default $1 - F$. Give it whenever it can be computed directly: the log
   * tails and upper quantiles rely on it.
   */
  survival?(x: Value): Value
  /**
   * The log survival function. Default `logFromTails(survival, cdf)`, or $\operatorname{log1p}(-F)$ without
   * `survival`.
   */
  logSurvival?(x: Value): Value
  /**
   * The quantile $F^{-1}(p)$. Default: numerical inversion of the cdf, or of the survival function above $p = 1/2$
   * when one is given (not differentiable).
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
  /** The mean, of the batch shape where the family has one; NaN or $\infty$ where it does not exist. */
  mean(): Value
  /** The variance; `stddev` is derived from it. */
  variance(): Value
  /** The differential entropy (the Shannon entropy for a discrete family), in nats. */
  entropy(): Value
  /** The mode, or NaN where it is not unique or not defined. */
  mode(): Value
  /** The exponential-family structure, for families that are one. */
  expFamily?: ExponentialFamily
}

/**
 * Draws of a batch: the sample shape and the batch shape, and whether to return a number.
 *
 * @param batchShape The distribution's batch shape.
 * @param options The `sample` options; `options.shape` is the sample shape, prepended to the batch shape.
 * @returns `shape`, the full shape `[...options.shape, ...batchShape]`, and `scalar`, true when no sample shape was
 *   given and the batch shape is `[]` (one draw, returned as a number).
 */
export function drawShape(
  batchShape: readonly number[],
  options?: SampleOptions,
): { shape: number[]; scalar: boolean } {
  const shape = [...(options?.shape ?? []), ...batchShape]
  return { shape, scalar: options?.shape === undefined && batchShape.length === 0 }
}

/**
 * Build a `Univariate` from a specification: the defaults of the specification's optional functions are filled in,
 * `prob` is $e^{\text{logProb}}$, `stddev` the square root of `variance`, and `sample` checks that no parameter is
 * traced (throwing a `NotDifferentiableError` naming it) before drawing with `spec.sample`, or with `spec.rsample` on
 * the raw parameters. `rsample` is present only when the specification has one. The event shape is `[]`.
 *
 * @param spec The family's specification. Its functions are called afresh on every method call, never cached.
 * @returns The distribution, with batch shape `spec.batchShape`, or the broadcast shape of the parameters.
 *
 * @example A family from its log-density and cdf
 * // A batch of two exponentials, with no quantile given: it is found by bisection on the cdf.
 * const rate = tensor([1, 2])
 * const d = univariate({
 *   name: 'MyExponential',
 *   params: { rate },
 *   support: { type: 'interval', lower: 0, upper: Infinity },
 *   logProb: (x) => sub(log(rate), mul(rate, x)),
 *   cdf: (x) => neg(expm1(neg(mul(rate, x)))),
 *   mean: () => div(1, rate),
 *   variance: () => div(1, square(rate)),
 *   entropy: () => sub(1, log(rate)),
 *   mode: () => mul(0, rate),
 * })
 * print('batch shape:', d.batchShape)
 * print('p(1) =', d.prob(1))
 * print('median =', d.quantile(0.5), ' log(2) / rate =', div(Math.LN2, rate))
 * print('stddev =', d.stddev())
 */
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

/**
 * Base variates of a draw: a number when `scalar`, else a tensor of `shape` filled row-major.
 *
 * @param values The variates, at least one when `scalar` and exactly the size of `shape` otherwise; not copied.
 * @param shape The draw's full shape.
 * @param scalar Whether to return the first variate as a number.
 * @returns The variates as a number or a tensor.
 */
function variates(values: Float64Array, shape: number[], scalar: boolean): number | Tensor {
  return scalar ? values[0] : fromData(values, shape)
}

/**
 * Uniforms in $[0, 1)$ for a draw of `shape` (a number when `scalar`), from one block of the stream.
 *
 * @param s The stream to draw from; it is advanced.
 * @param shape The draw's full shape.
 * @param scalar Whether to draw one uniform and return it as a number.
 * @returns The uniforms.
 */
export function unitsOf(s: Stream, shape: number[], scalar: boolean): number | Tensor {
  return variates(units(s, scalar ? 1 : shape.reduce((a, b) => a * b, 1)), shape, scalar)
}

/**
 * Standard normals for a draw of `shape` (a number when `scalar`), from one block of the stream.
 *
 * @param s The stream to draw from; it is advanced.
 * @param shape The draw's full shape.
 * @param scalar Whether to draw one normal and return it as a number.
 * @returns The standard normals.
 */
export function normalsOf(s: Stream, shape: number[], scalar: boolean): number | Tensor {
  return variates(standardNormals(s, scalar ? 1 : shape.reduce((a, b) => a * b, 1)), shape, scalar)
}

/**
 * A location–scale pathwise draw $\mu + \sigma z$ with $z$ standard normal (Kingma and Welling 2014, "Auto-encoding
 * variational Bayes", §2.4): differentiable in $\mu$ and $\sigma$.
 *
 * @param s The stream to draw from; it is advanced.
 * @param shape The draw's full shape (sample shape, then batch shape).
 * @param scalar Whether to draw one value (a number, or a traced number).
 * @param loc The location $\mu$, broadcast with the draw.
 * @param scale The scale $\sigma$, broadcast with the draw.
 * @returns The draw, traced when `loc` or `scale` is.
 */
export function locationScale(s: Stream, shape: number[], scalar: boolean, loc: Value, scale: Value): Value {
  return add(loc, mul(scale, normalsOf(s, shape, scalar)))
}

/**
 * Inverse-transform sampling as a pathwise draw: uniforms of the draw's shape mapped through a quantile function
 * written with primitives, so the draw is differentiable in the parameters the quantile uses (Devroye 1986,
 * "Non-Uniform Random Variate Generation", §2.1).
 *
 * @param s The stream to draw from; it is advanced.
 * @param shape The draw's full shape (sample shape, then batch shape).
 * @param scalar Whether to draw one value (a number, or a traced number).
 * @param quantile The quantile function, applied to the uniforms.
 * @returns The draw.
 */
export function inverseTransform(s: Stream, shape: number[], scalar: boolean, quantile: (u: Value) => Value): Value {
  return quantile(unitsOf(s, shape, scalar))
}

/**
 * Check a parameter's raw values (traced values are checked through their underlying value): throws a `DomainError`
 * naming the family, the parameter and the requirement when any element fails `test`.
 *
 * @param family The family's name, which starts the error message.
 * @param label The parameter's name in the message (or an expression of parameters, such as the width of a
 *   uniform's interval).
 * @param v The parameter, every element of which is tested.
 * @param test The requirement on one element.
 * @param what The requirement in words for the message, such as `'positive'`.
 */
export function check(family: string, label: string, v: Value, test: (x: number) => boolean, what: string): void {
  const r = unwrap(v)
  const values = typeof r === 'number' ? [r] : toFlat(r)
  for (const x of values) if (!test(x)) throw new DomainError(family, `${family}: ${label} must be ${what}, got ${x}`)
}
