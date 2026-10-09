/**
 * Maps of the real line for pushing distributions forward: intervals with open and closed ends, monotone bijectors
 * (each with its domain, codomain, inverse and $\log \lvert f'(x) \rvert$), and many-to-one maps given by monotone
 * branches ($y = x^2$). `imageOf` computes the image of an interval under a map, which is how `Transformed` and
 * `Pushforward` find their supports and report a base whose support does not fit the map's domain.
 *
 * The scalar bijectors act elementwise on numbers, tensors and traced values. Two maps act on vectors along the last
 * axis (`eventRank: 1`): `orderedBijector` takes $\reals^K$ onto strictly increasing vectors (ordinal thresholds), and
 * `affineCouplingBijector` is the coupling layer of a normalising flow. Every forward map, inverse and log-Jacobian is
 * a composition of tensor primitives, so all of them are differentiable.
 */

import {
  logSigmoid,
  logit,
  normalCdf,
  normalLogPdf,
  normalQuantile,
  sigmoid,
  softplus,
  logExpm1,
} from 'aifn-compute/numerics/special'
import {
  abs,
  add,
  div,
  exp,
  isTensor,
  log,
  log1p,
  mul,
  neg,
  pow,
  sqrt,
  square,
  concat,
  matmul,
  shapeOfValue,
  slice,
  sub,
  sum,
  tanh,
  fromData,
  toFlat,
  unwrap,
  type Value,
} from 'aifn-compute/foundation/tensor'
import type { Bijector, Interval, Scalar, Support } from 'aifn-compute/foundation/contracts'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

// Types defined once, in `aifn-compute/foundation/contracts`.
export type { Interval, Bijector } from 'aifn-compute/foundation/contracts'

// ── Intervals ────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The interval from `lower` to `upper`, with its ends open or closed as the brackets say. An infinite end is open
 * whatever the brackets say. The ends are not checked: `lower` above `upper` gives an empty interval.
 *
 * @param lower The lower end, a number (may be $-\infty$).
 * @param upper The upper end, a number (may be $\infty$).
 * @param ends Which ends are closed: `'[]'` both, `'()'` neither, `'[)'` the lower only, `'(]'` the upper only.
 * @returns The `Interval`, with `lowerOpen` and `upperOpen` set from the brackets and the infinite ends.
 *
 * @example Brackets, and an infinite end
 * print('closed:', formatInterval(interval(0, 1)))
 * print('half-open:', formatInterval(interval(0, 1, '[)')))
 * print('[0, Infinity]:', formatInterval(interval(0, Infinity)))
 */
export function interval(lower: number, upper: number, ends: '[]' | '()' | '[)' | '(]' = '[]'): Interval {
  return {
    lower,
    upper,
    lowerOpen: ends[0] === '(' || !Number.isFinite(lower),
    upperOpen: ends[1] === ')' || !Number.isFinite(upper),
  }
}

/** The real line $\reals = (-\infty, \infty)$. */
export const REALS: Interval = interval(-Infinity, Infinity)
/** The positive half-line $(0, \infty)$, open at 0. */
export const POSITIVE: Interval = interval(0, Infinity, '()')
/** The open unit interval $(0, 1)$. */
export const UNIT_INTERVAL: Interval = interval(0, 1, '()')

/**
 * The smallest or the largest raw value of a bound: a number is itself, a batch of bounds (a tensor) gives its
 * outermost entry, and a traced bound is read through to its value.
 *
 * @param v The bound: a number, a tensor or a traced value.
 * @param which `'min'` for the smallest entry, `'max'` for the largest.
 * @returns That entry, as a number (NaN when the raw value is neither a number nor a tensor).
 */
function extreme(v: Value, which: 'min' | 'max'): number {
  const r = unwrap(v)
  const values = typeof r === 'number' ? [r] : isTensor(r) ? toFlat(r) : [NaN]
  return which === 'min' ? Math.min(...values) : Math.max(...values)
}

/**
 * A univariate support as an interval: `real` is $\reals$; `interval`, `integers` and `circle` take their bounds (the
 * outermost of a batch), closed at finite ends unless an `interval` support marks them open. Other supports (simplex,
 * vectors, matrices) throw a `DomainError`.
 *
 * @param support The support of a univariate distribution, as its `support` field holds it.
 * @returns The interval from its smallest lower bound to its largest upper bound.
 *
 * @example A batch of uniform supports, and the integers
 * print(formatInterval(supportInterval({ type: 'interval', lower: tensor([0, -1]), upper: tensor([1, 2]) })))
 * print(formatInterval(supportInterval({ type: 'integers', lower: 0, upper: Infinity })))
 */
export function supportInterval(support: Support): Interval {
  if (support.type === 'real') return REALS
  if (support.type === 'interval' || support.type === 'integers' || support.type === 'circle') {
    const lower = extreme(support.lower, 'min')
    const upper = extreme(support.upper, 'max')
    const open = support.type === 'interval' ? support : { lowerOpen: false, upperOpen: false }
    return {
      lower,
      upper,
      lowerOpen: !!open.lowerOpen || !Number.isFinite(lower),
      upperOpen: !!open.upperOpen || !Number.isFinite(upper),
    }
  }
  throw new DomainError(
    'supportInterval',
    `supportInterval: a ${support.type} support is not an interval of the real line`,
  )
}

/**
 * An interval as a `Support`: $\reals$ as `real`, anything else as `interval` with its open ends marked.
 *
 * @param i The interval.
 * @returns The support: `{ type: 'real' }` for the whole line, otherwise `{ type: 'interval', ... }`.
 *
 * @example The real line and the unit interval
 * print('REALS:', intervalSupport(REALS))
 * print('UNIT_INTERVAL:', intervalSupport(UNIT_INTERVAL))
 */
export function intervalSupport(i: Interval): Support {
  if (i.lower === -Infinity && i.upper === Infinity) return { type: 'real' }
  return { type: 'interval', lower: i.lower, upper: i.upper, lowerOpen: i.lowerOpen, upperOpen: i.upperOpen }
}

/**
 * One end of an interval as plain text, to four significant digits, with the infinity sign for an infinite end and the
 * typeset minus sign (U+2212) for a negative one.
 *
 * @param v The end.
 * @returns The text, e.g. `'0.3679'`, `'∞'`.
 */
const formatEnd = (v: number) =>
  v === Infinity ? '∞' : v === -Infinity ? '−∞' : String(Number(v.toPrecision(4))).replace('-', '−')

/**
 * Plain text for an interval, its ends to four significant digits and in brackets that say which are open. The whole
 * line is written as the double-struck R. Used in error messages and labels.
 *
 * @param i The interval.
 * @returns The text, with Unicode infinity and minus signs.
 *
 * @example Three intervals
 * print(formatInterval(REALS))
 * print(formatInterval(POSITIVE))
 * print(formatInterval(interval(Math.exp(-1), Math.E)))
 */
export function formatInterval(i: Interval): string {
  if (i.lower === -Infinity && i.upper === Infinity) return 'ℝ'
  return `${i.lowerOpen ? '(' : '['}${formatEnd(i.lower)}, ${formatEnd(i.upper)}${i.upperOpen ? ')' : ']'}`
}

/**
 * A point strictly inside an interval, e.g. to stand in for values outside it: the midpoint of a bounded interval, one
 * unit inside the finite end of a half-line, and 0 for the whole line.
 *
 * @param i The interval (not checked to be non-empty).
 * @returns The point.
 *
 * @example Bounded, half-line and whole line
 * print('[2, 4]:', supportInteriorPoint(interval(2, 4)))
 * print('(0, Infinity):', supportInteriorPoint(POSITIVE))
 * print('REALS:', supportInteriorPoint(REALS))
 */
export function supportInteriorPoint(i: Interval): Scalar {
  const { lower: a, upper: b } = i
  if (Number.isFinite(a) && Number.isFinite(b)) return (a + b) / 2
  if (Number.isFinite(a)) return a + 1
  if (Number.isFinite(b)) return b - 1
  return 0
}

/**
 * Whether `inner` lies inside `outer`. With `endpoints: 'ignore'` (the default, for continuous distributions) a closed
 * end of `inner` may sit on an open end of `outer`: a single point has probability zero, so $[0, \infty)$ counts as
 * inside $(0, \infty)$. With `endpoints: 'strict'` (for mass functions) it does not.
 *
 * @param inner The interval to test.
 * @param outer The interval it should lie in.
 * @param options How shared ends are judged.
 * @param options.endpoints `'ignore'` compares the ends' values only; `'strict'` also requires a shared end that is
 *   open in `outer` to be open in `inner`.
 * @returns True when every point of `inner` (up to its ends, under `'ignore'`) is in `outer`.
 *
 * @example A closed end on an open one
 * const halfLine = interval(0, Infinity, '[)')
 * print('ignore:', intervalInside(halfLine, POSITIVE))
 * print('strict:', intervalInside(halfLine, POSITIVE, { endpoints: 'strict' }))
 */
export function intervalInside(
  inner: Interval,
  outer: Interval,
  { endpoints = 'ignore' }: { endpoints?: 'ignore' | 'strict' } = {},
): boolean {
  if (inner.lower < outer.lower || inner.upper > outer.upper) return false
  if (endpoints === 'ignore') return true
  const lowerOk = inner.lower > outer.lower || !outer.lowerOpen || inner.lowerOpen
  const upperOk = inner.upper < outer.upper || !outer.upperOpen || inner.upperOpen
  return lowerOk && upperOk
}

/**
 * The intersection of two intervals: an end is open when it is open in the interval it comes from (in either, when
 * both share it).
 *
 * @param a One interval.
 * @param b The other.
 * @returns The intersection, or `null` when it is empty (including a single point that one of them leaves out).
 */
function intersect(a: Interval, b: Interval): Interval | null {
  const lower = Math.max(a.lower, b.lower)
  const upper = Math.min(a.upper, b.upper)
  const lowerOpen = (a.lower === lower && a.lowerOpen) || (b.lower === lower && b.lowerOpen)
  const upperOpen = (a.upper === upper && a.upperOpen) || (b.upper === upper && b.upperOpen)
  if (lower > upper || (lower === upper && (lowerOpen || upperOpen))) return null
  return { lower, upper, lowerOpen, upperOpen }
}

/**
 * The smallest interval holding every interval (their union, when they overlap). An end is open only when every part
 * that reaches it has it open.
 *
 * @param parts The intervals, at least one (an empty list gives the empty interval from $\infty$ to $-\infty$).
 * @returns Their hull.
 */
function hull(parts: readonly Interval[]): Interval {
  const lower = Math.min(...parts.map((p) => p.lower))
  const upper = Math.max(...parts.map((p) => p.upper))
  return {
    lower,
    upper,
    lowerOpen: parts.every((p) => p.lower !== lower || p.lowerOpen),
    upperOpen: parts.every((p) => p.upper !== upper || p.upperOpen),
  }
}

// ── Bijectors ────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * $y = e^x$, from $\reals$ to $(0, \infty)$, with $\log \lvert dy/dx \rvert = x$. Pushes a normal forward to a
 * log-normal.
 *
 * @example Forward, inverse and log-Jacobian at a point
 * print('forward(1):', expBijector.forward(1))
 * print('inverse(e):', expBijector.inverse(Math.E))
 * print('log |dy/dx| at 1:', expBijector.logAbsDetJacobian(1))
 */
export const expBijector: Bijector = {
  name: 'exp',
  forward: exp,
  inverse: log,
  logAbsDetJacobian: (x) => x,
  increasing: true,
  domain: REALS,
  codomain: POSITIVE,
}

/**
 * $y = \log x$, from $(0, \infty)$ to $\reals$, with $\log \lvert dy/dx \rvert = -\log x$. Pushes a log-normal back to
 * a normal.
 *
 * @example Forward, inverse and log-Jacobian at a point
 * print('forward(e):', logBijector.forward(Math.E))
 * print('inverse(0):', logBijector.inverse(0))
 * print('log |dy/dx| at e:', logBijector.logAbsDetJacobian(Math.E))
 */
export const logBijector: Bijector = {
  name: 'log',
  forward: log,
  inverse: exp,
  logAbsDetJacobian: (x) => neg(log(x)),
  increasing: true,
  domain: POSITIVE,
  codomain: REALS,
}

/**
 * $y = \sigma(x) = 1/(1 + e^{-x})$, from $\reals$ to $(0, 1)$ (the logit-normal and the like). The inverse is the
 * logit, and $\log \lvert dy/dx \rvert = \log\sigma(x) + \log\sigma(-x)$, computed from log-sigmoids so that it
 * stays finite in the tails.
 *
 * @example Forward, inverse and log-Jacobian at 0
 * print('forward(0):', sigmoidBijector.forward(0))
 * print('inverse(0.5):', sigmoidBijector.inverse(0.5))
 * print('log |dy/dx| at 0 (log 1/4):', sigmoidBijector.logAbsDetJacobian(0))
 */
export const sigmoidBijector: Bijector = {
  name: 'sigmoid',
  forward: sigmoid,
  inverse: logit,
  logAbsDetJacobian: (x) => add(logSigmoid(x), logSigmoid(neg(x))),
  increasing: true,
  domain: REALS,
  codomain: UNIT_INTERVAL,
}

/**
 * $y = \tanh x$, from $\reals$ to $(-1, 1)$. The inverse is
 * $\operatorname{artanh} y = \tfrac{1}{2}(\log(1 + y) - \log(1 - y))$, each term by `log1p`;
 * $\log \lvert dy/dx \rvert = \log(1 - \tanh^2 x)$ is computed as $2(\log 2 - x - \operatorname{softplus}(-2x))$,
 * which stays finite in the tails where $1 - \tanh^2 x$ underflows.
 *
 * @example Forward, inverse and log-Jacobian at a point, and far in the tail
 * print('forward(0.5):', tanhBijector.forward(0.5))
 * print('inverse(tanh 0.5):', tanhBijector.inverse(Math.tanh(0.5)))
 * print('log |dy/dx| at 0:', tanhBijector.logAbsDetJacobian(0))
 * print('log |dy/dx| at 400:', tanhBijector.logAbsDetJacobian(400))
 */
export const tanhBijector: Bijector = {
  name: 'tanh',
  forward: tanh,
  inverse: (y) => mul(0.5, sub(log1p(y), log1p(neg(y)))),
  logAbsDetJacobian: (x) => mul(2, sub(sub(Math.LN2, x), softplus(mul(-2, x)))),
  increasing: true,
  domain: REALS,
  codomain: interval(-1, 1, '()'),
}

/**
 * $y = \operatorname{softplus} x = \log(1 + e^x)$, from $\reals$ to $(0, \infty)$; the inverse is $\log(e^y - 1)$ and
 * $dy/dx = \sigma(x)$, so $\log \lvert dy/dx \rvert = \log\sigma(x)$.
 *
 * @example Forward, inverse and log-Jacobian at 0
 * print('forward(0) (log 2):', softplusBijector.forward(0))
 * print('inverse(log 2):', softplusBijector.inverse(Math.LN2))
 * print('log |dy/dx| at 0 (log 1/2):', softplusBijector.logAbsDetJacobian(0))
 */
export const softplusBijector: Bijector = {
  name: 'softplus',
  forward: softplus,
  inverse: logExpm1,
  logAbsDetJacobian: logSigmoid,
  increasing: true,
  domain: REALS,
  codomain: POSITIVE,
}

/**
 * $y = \Phi(x)$, the standard normal cdf, from $\reals$ to $(0, 1)$; the inverse is the probit $\Phi^{-1}$ and
 * $dy/dx = \phi(x)$, the standard normal density. Pushes $\Gauss(0, 1)$ forward to $\Unif(0, 1)$ (the probability
 * integral transform). Its `name` is the Unicode capital phi.
 *
 * @example Forward, inverse and log-Jacobian at a point
 * print('forward(1.96):', normalCdfBijector.forward(1.96))
 * print('inverse(0.975):', normalCdfBijector.inverse(0.975))
 * print('log |dy/dx| at 0 (log of 1/sqrt(2 pi)):', normalCdfBijector.logAbsDetJacobian(0))
 */
export const normalCdfBijector: Bijector = {
  name: 'Φ',
  forward: normalCdf,
  inverse: normalQuantile,
  logAbsDetJacobian: normalLogPdf,
  increasing: true,
  domain: REALS,
  codomain: UNIT_INTERVAL,
}

/**
 * $y = a + bx$ ($a$ = `loc`, $b$ = `scale`), from $\reals$ to $\reals$, with
 * $\log \lvert dy/dx \rvert = \log \lvert b \rvert$ at every point. Increasing for $b > 0$ and decreasing for $b < 0$.
 * Throws a `DomainError` when the scale is zero or not finite.
 *
 * @param loc The shift $a$: a number, or a tensor that broadcasts against the input (a batch of shifts).
 * @param scale The scale $b$, a finite non-zero number (not a tensor, so that the direction of the map is known).
 * @returns The bijector.
 *
 * @example A negative scale reverses the order
 * const f = affineBijector(1, -2)
 * print('forward(3):', f.forward(3))
 * print('inverse(-5):', f.inverse(-5))
 * print('log |dy/dx| (log 2):', f.logAbsDetJacobian(3))
 * print('increasing:', f.increasing)
 */
export function affineBijector(loc: Value, scale: number): Bijector {
  if (!(scale !== 0 && Number.isFinite(scale)))
    throw new DomainError('affineBijector', 'affineBijector: scale must be finite and non-zero')
  return {
    name: 'affine',
    forward: (x) => add(loc, mul(scale, x)),
    inverse: (y) => div(sub(y, loc), scale),
    logAbsDetJacobian: (x) => add(mul(0, x), Math.log(Math.abs(scale))),
    increasing: scale > 0,
    domain: REALS,
    codomain: REALS,
  }
}

/**
 * $y = x^p$ from $(0, \infty)$ onto itself, for a finite non-zero power $p$ (increasing for $p > 0$, decreasing for
 * $p < 0$); the inverse is $y^{1/p}$ and $\log \lvert dy/dx \rvert = \log \lvert p \rvert + (p - 1) \log x$. Throws a
 * `DomainError` when $p$ is zero or not finite.
 *
 * @param p The power $p$.
 * @returns The bijector, named `power p`.
 *
 * @example The square on the positive half-line
 * const f = powerBijector(2)
 * print('forward(3):', f.forward(3))
 * print('inverse(9):', f.inverse(9))
 * print('log |dy/dx| at 3 (log 6):', f.logAbsDetJacobian(3))
 */
export function powerBijector(p: number): Bijector {
  if (!(p !== 0 && Number.isFinite(p)))
    throw new DomainError('powerBijector', 'powerBijector: the power must be finite and non-zero')
  return {
    name: `power ${p}`,
    forward: (x) => pow(x, p),
    inverse: (y) => pow(y, 1 / p),
    logAbsDetJacobian: (x) => add(Math.log(Math.abs(p)), mul(p - 1, log(x))),
    increasing: p > 0,
    domain: POSITIVE,
    codomain: POSITIVE,
  }
}

/**
 * The composition that applies `bijectors` in order, first to last: $y = f_n(\dots f_1(x))$. The log-Jacobians add
 * along the chain, each taken at the point that bijector receives; the chain is increasing when an even number of its
 * bijectors decrease. The domain is the first's, and the codomain the image of the first's codomain through the rest.
 * Throws a `DomainError` when there are no bijectors, or when one's image does not fit the next one's domain.
 *
 * @param bijectors The bijectors, in the order they are applied (at least one).
 * @returns The composed bijector, named after its parts (`affine then sigmoid`).
 *
 * @example A sigmoid with temperature 2
 * const f = chainBijectors(affineBijector(0, 1 / 2), sigmoidBijector)
 * print('name:', f.name)
 * print('forward(2) (sigmoid of 1):', f.forward(2))
 * print('inverse back:', f.inverse(f.forward(2)))
 * print('log |dy/dx| at 2:', f.logAbsDetJacobian(2))
 * print('codomain:', formatInterval(f.codomain))
 */
export function chainBijectors(...bijectors: Bijector[]): Bijector {
  if (bijectors.length === 0) throw new DomainError('chainBijectors', 'chainBijectors: needs at least one bijector')
  let codomain = bijectors[0].codomain
  for (const b of bijectors.slice(1)) codomain = imageOf(b, codomain, { where: 'chainBijectors' })
  return {
    name: bijectors.map((b) => b.name).join(' then '),
    forward: (x) => bijectors.reduce((v, b) => b.forward(v), x),
    inverse: (y) => bijectors.reduceRight((v, b) => b.inverse(v), y),
    logAbsDetJacobian: (x) => {
      let total: Value = 0
      let v = x
      for (const b of bijectors) {
        total = add(total, b.logAbsDetJacobian(v))
        v = b.forward(v)
      }
      return total
    },
    increasing: bijectors.filter((b) => !b.increasing).length % 2 === 0,
    domain: bijectors[0].domain,
    codomain,
  }
}

// ── Ordered vectors ──────────────────────────────────────────────────────────────────────────────────────────────────

/** Options of `orderedBijector`. */
export type OrderedOptions = {
  /**
   * The map from an unconstrained increment to a positive gap: `exp` (default, as Stan) or `softplus` (gaps grow
   * linearly rather than exponentially in the unconstrained value).
   */
  gap?: 'exp' | 'softplus'
}

/**
 * The $K \times K$ upper-triangular matrix of ones $\Umat$: $\zvec\Umat$ is the cumulative sum of $\zvec$ along its
 * last axis.
 *
 * @param k The size $K$.
 * @returns $\Umat$ as a $K \times K$ tensor.
 */
function cumulativeSumMatrix(k: number): Value {
  const u = new Float64Array(k * k)
  for (let i = 0; i < k; i++) for (let j = i; j < k; j++) u[i * k + j] = 1
  return fromData(u, [k, k])
}

/**
 * The entries `start` to `stop - 1` of the last axis of `v`, whatever its rank (every other axis is kept whole).
 *
 * @param v A value of rank at least 1.
 * @param start The first index kept.
 * @param stop One past the last index kept.
 * @returns The slice, of the same rank as `v`.
 */
function lastAxis(v: Value, start: number, stop: number): Value {
  const rank = shapeOfValue(v).length
  return slice(v, ...Array.from({ length: rank - 1 }, () => null), [start, stop])
}

/**
 * The ordered bijector: $\reals^K$ onto strictly increasing vectors $y_1 < y_2 < \dots < y_K$, applied along the last
 * axis (Stan Development Team, "Stan Reference Manual", §10.6 "Ordered vector"). $y_1 = x_1$ and
 * $y_k = y_{k-1} + g(x_k)$ for $k \ge 2$, with $g = \exp$ (default) or softplus. The inverse is $x_1 = y_1$,
 * $x_k = g^{-1}(y_k - y_{k-1})$; the Jacobian is triangular, so
 * $\log \lvert \det \Jmat \rvert = \sum_{k \ge 2} \log g'(x_k)$ ($\sum_{k \ge 2} x_k$ for $\exp$), summed over the last
 * axis (a number for one vector, one value per row of a batch). Everything is a composition of primitives, so
 * thresholds built this way are differentiable. The `Bijector` contract describes scalar maps: here `domain` and
 * `codomain` are the coordinates' range $\reals$, `increasing` says each $y_k$ increases in every $x_j$, and
 * `eventRank: 1` marks the vector event. A scalar input throws a `ShapeError`; a vector of length 1 passes through
 * unchanged.
 *
 * @param options How the gaps are made.
 * @param options.gap The map $g$ from an unconstrained value to a positive gap (see `OrderedOptions`).
 * @returns The bijector, named `ordered (exp)` or `ordered (softplus)`.
 *
 * @example Gaps of 1 and 2
 * const f = orderedBijector()
 * const y = f.forward(tensor([-1, 0, Math.LN2]))
 * print('forward:', y)
 * print('inverse:', f.inverse(y))
 * print('log |det J| (log 2):', f.logAbsDetJacobian(tensor([-1, 0, Math.LN2])))
 *
 * @example Softplus gaps, on a batch of two vectors
 * const f = orderedBijector({ gap: 'softplus' })
 * print('forward:', f.forward(tensor([[0, 0], [1, 5]])))
 * print('log |det J| per row:', f.logAbsDetJacobian(tensor([[0, 0], [1, 5]])))
 */
export function orderedBijector({ gap = 'exp' }: OrderedOptions = {}): Bijector & { readonly eventRank: 1 } {
  const positive = gap === 'exp' ? exp : softplus
  const positiveInverse = gap === 'exp' ? log : logExpm1
  const logDerivative = gap === 'exp' ? (x: Value) => x : logSigmoid
  const width = (v: Value, where: string) => {
    const shape = shapeOfValue(v)
    if (shape.length === 0)
      throw new ShapeError('orderedBijector', `orderedBijector: ${where} needs a vector (rank ≥ 1)`)
    return shape[shape.length - 1]
  }
  return {
    name: `ordered (${gap})`,
    eventRank: 1,
    forward: (x) => {
      const k = width(x, 'forward')
      if (k === 1) return x
      const steps = concat([lastAxis(x, 0, 1), positive(lastAxis(x, 1, k))], -1)
      return matmul(steps, cumulativeSumMatrix(k))
    },
    inverse: (y) => {
      const k = width(y, 'inverse')
      if (k === 1) return y
      const gaps = sub(lastAxis(y, 1, k), lastAxis(y, 0, k - 1))
      return concat([lastAxis(y, 0, 1), positiveInverse(gaps)], -1)
    },
    logAbsDetJacobian: (x) => {
      const k = width(x, 'logAbsDetJacobian')
      if (k === 1) return sum(mul(0, x), -1)
      return sum(logDerivative(lastAxis(x, 1, k)), -1)
    },
    increasing: true,
    domain: REALS,
    codomain: REALS,
  }
}

/**
 * What a coupling layer's conditioner returns from the kept coordinates: `shift`, the shift $\tvec$ of the moved
 * coordinates, and `logScale`, their log-scale $\svec$ (left out for NICE's additive coupling). Each has the input's
 * shape or broadcasts to it; entries on the kept coordinates are ignored.
 */
export type CouplingParameters = { shift: Value; logScale?: Value }

/**
 * The affine coupling bijector of RealNVP (Dinh, Sohl-Dickstein and Bengio, 2017), along the last axis: with a 0/1
 * `mask` $\mvec$ of length $D$, the coordinates where $m_j = 1$ pass through, and the others are scaled and shifted by
 * functions of them,
 * $\yvec = \mvec \odot \xvec + (1 - \mvec) \odot (\xvec \odot \exp \svec + \tvec)$ with
 * $(\svec, \tvec)$ = `conditioner`$(\mvec \odot \xvec)$. The Jacobian is triangular, so
 * $\log \lvert \det \Jmat \rvert = \sum_j (1 - m_j) s_j$; the inverse needs no inverse of the conditioner,
 * $\xvec = \mvec \odot \yvec + (1 - \mvec) \odot (\yvec - \tvec) \odot \exp(-\svec)$, because
 * $\mvec \odot \yvec = \mvec \odot \xvec$. Without a log-scale the layer is NICE's additive coupling (Dinh, Krueger
 * and Bengio, 2015), with $\log \lvert \det \Jmat \rvert = 0$. The conditioner is any function of values (a neural
 * network's apply, closed over its parameters), so the map is differentiable in $\xvec$ and in the conditioner's
 * parameters. Works on one vector of length $D$ or a batch of shape $[n, D]$; any other last axis throws a
 * `ShapeError`.
 *
 * @param mask The 0/1 mask $\mvec$, one entry per coordinate: 1 keeps the coordinate (and feeds it to the
 *   conditioner), 0 moves it.
 * @param conditioner The function from $\mvec \odot \xvec$ (the input with its moved coordinates zeroed, same shape)
 *   to the shift and log-scale of the moved coordinates (see `CouplingParameters`). Its output at a moved coordinate
 *   is meant to depend on the kept ones: the input's own entry there is always 0.
 * @returns The bijector, with `eventRank: 1`.
 *
 * @example The second coordinate shifted and scaled by the first
 * const f = affineCouplingBijector([1, 0], (xm) => ({ shift: sum(xm, -1, true), logScale: sum(xm, -1, true) }))
 * const y = f.forward(tensor([1, 2]))
 * print('forward (1, 2e + 1):', y)
 * print('inverse:', f.inverse(y))
 * print('log |det J| (s = 1):', f.logAbsDetJacobian(tensor([1, 2])))
 */
export function affineCouplingBijector(
  mask: readonly number[],
  conditioner: (masked: Value) => CouplingParameters,
): Bijector & { readonly eventRank: 1 } {
  const m = fromData(Float64Array.from(mask), [mask.length])
  const free = fromData(
    Float64Array.from(mask, (v) => 1 - v),
    [mask.length],
  )
  const check = (v: Value, where: string) => {
    const shape = shapeOfValue(v)
    if (shape.length === 0 || shape[shape.length - 1] !== mask.length)
      throw new ShapeError(
        'affineCouplingBijector',
        `affineCouplingBijector: ${where} needs a last axis of length ${mask.length}`,
      )
  }
  const params = (kept: Value) => {
    const { shift, logScale } = conditioner(kept)
    return { shift: mul(free, shift), logScale: logScale === undefined ? null : mul(free, logScale) }
  }
  return {
    name: 'affine coupling',
    eventRank: 1,
    forward: (x) => {
      check(x, 'forward')
      const kept = mul(m, x)
      const { shift, logScale } = params(kept)
      const moved = logScale === null ? mul(free, x) : mul(mul(free, x), exp(logScale))
      return add(kept, add(moved, shift))
    },
    inverse: (y) => {
      check(y, 'inverse')
      const kept = mul(m, y)
      const { shift, logScale } = params(kept)
      const moved = mul(free, sub(y, shift))
      return add(kept, logScale === null ? moved : mul(moved, exp(neg(logScale))))
    },
    logAbsDetJacobian: (x) => {
      check(x, 'logAbsDetJacobian')
      const { logScale } = params(mul(m, x))
      return sum(logScale === null ? mul(0, x) : logScale, -1)
    },
    increasing: true,
    domain: REALS,
    codomain: REALS,
  }
}

// ── Many-to-one maps ─────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * One monotone piece of a many-to-one map: on `domain` the map is a bijection with this inverse and
 * $\log \lvert f'(x) \rvert$.
 */
export type Branch = {
  /** The piece of the map's domain this branch covers. */
  domain: Interval
  /** The preimage $x$ in `domain` of a value $y$ in the branch's image. */
  inverse(y: Value): Value
  /** $\log \lvert f'(x) \rvert$ at a point $x$ of `domain`. */
  logAbsDetJacobian(x: Value): Value
  /** Whether the map increases on `domain` (false: it decreases). */
  increasing: boolean
}

/**
 * A map $y = f(x)$ whose domain is covered by monotone branches (overlapping at most at their ends). A value $y$ then
 * has one preimage per branch whose image holds it, and the pushforward density sums over them.
 */
export type ManyToOneMap = {
  /** A readable name, e.g. `square`. */
  name: string
  /** $f(x)$, elementwise. */
  forward(x: Value): Value
  /** The monotone pieces, which together cover `domain`. */
  branches: readonly Branch[]
  /** Where the map is defined. */
  domain: Interval
  /** The image of `domain` under $f$. */
  codomain: Interval
}

/**
 * $y = x^2$, from $\reals$ onto $[0, \infty)$: two branches, $x = -\sqrt{y}$ on $(-\infty, 0]$ and $x = \sqrt{y}$ on
 * $[0, \infty)$, each with $\log \lvert dy/dx \rvert = \log \lvert 2x \rvert$. A standard normal pushed forward is the
 * $\chi^2_1$ distribution.
 *
 * @example A value's two preimages
 * print('forward(-3):', squareMap.forward(-3))
 * print('preimages of 9:', squareMap.branches.map((b) => b.inverse(9)))
 * print('log |dy/dx| at -3 (log 6):', squareMap.branches[0].logAbsDetJacobian(-3))
 */
export const squareMap: ManyToOneMap = {
  name: 'square',
  forward: square,
  branches: [
    {
      domain: interval(-Infinity, 0, '(]'),
      inverse: (y) => neg(sqrt(y)),
      logAbsDetJacobian: (x) => log(abs(mul(2, x))),
      increasing: false,
    },
    {
      domain: interval(0, Infinity, '[)'),
      inverse: sqrt,
      logAbsDetJacobian: (x) => log(abs(mul(2, x))),
      increasing: true,
    },
  ],
  domain: REALS,
  codomain: interval(0, Infinity, '[)'),
}

/**
 * A bijector as a many-to-one map with one branch, so that code for pushforwards handles both; a many-to-one map is
 * returned as it is.
 *
 * @param b The bijector or many-to-one map.
 * @returns The map, with the bijector's domain, inverse, log-Jacobian and direction as its single branch.
 *
 * @example The exponential as one branch
 * const m = asManyToOne(expBijector)
 * print('branches:', m.branches.length)
 * print('branch domain:', formatInterval(m.branches[0].domain))
 */
export function asManyToOne(b: Bijector | ManyToOneMap): ManyToOneMap {
  if ('branches' in b) return b
  return { name: b.name, forward: b.forward, branches: [{ ...b }], domain: b.domain, codomain: b.codomain }
}

/**
 * The image of a subinterval `x` of a monotone branch's domain, from the map's values at the ends of `x`. An end of the
 * image is open when the end of `x` is open or infinite, when the image end is infinite, or when the end of `x` sits on
 * an open end of the domain (log at 0, say): those values are approached but not attained.
 *
 * @param forward The map $f$, evaluated at the two ends of `x` (it must accept infinite arguments).
 * @param increasing Whether $f$ increases on the branch; a decreasing one swaps the ends of the image.
 * @param x The subinterval, inside `domain`.
 * @param domain The branch's domain, for its open ends.
 * @returns The image $f(x)$; a batched map gives the outermost image over the batch.
 */
function monotoneImage(forward: (x: Value) => Value, increasing: boolean, x: Interval, domain: Interval): Interval {
  const at = (v: number) => forward(v)
  const lowOpen = (a: number) =>
    x.lowerOpen || !Number.isFinite(x.lower) || !Number.isFinite(a) || (x.lower === domain.lower && domain.lowerOpen)
  const highOpen = (b: number) =>
    x.upperOpen || !Number.isFinite(x.upper) || !Number.isFinite(b) || (x.upper === domain.upper && domain.upperOpen)
  const fl = at(x.lower)
  const fu = at(x.upper)
  if (increasing) {
    const a = extreme(fl, 'min')
    const b = extreme(fu, 'max')
    return { lower: a, upper: b, lowerOpen: lowOpen(a), upperOpen: highOpen(b) }
  }
  const a = extreme(fu, 'min')
  const b = extreme(fl, 'max')
  return { lower: a, upper: b, lowerOpen: highOpen(a), upperOpen: lowOpen(b) }
}

/**
 * The image $f(x)$ of an interval under a bijector or a many-to-one map (the hull of its branches' images). Throws a
 * `DomainError` naming `where` when `x` is not inside the map's domain.
 *
 * @param map The bijector or many-to-one map $f$.
 * @param x The interval to map.
 * @param options How the domain is checked.
 * @param options.where The caller's name, for the error message.
 * @param options.endpoints How a closed end of `x` on an open end of the domain is judged (see `intervalInside`).
 * @returns The image, with its ends open where they are approached but not attained.
 *
 * @example Images of intervals
 * print('sigmoid of REALS:', formatInterval(imageOf(sigmoidBijector, REALS)))
 * print('a negative scale flips the ends:', formatInterval(imageOf(affineBijector(0, -2), interval(0, 1))))
 * print('square of [-1, 2]:', formatInterval(imageOf(squareMap, interval(-1, 2))))
 *
 * @example A base outside the domain is reported
 * try {
 *   imageOf(logBijector, interval(-1, 1), { where: 'myTransform' })
 * } catch (e) {
 *   print(e.message)
 * }
 */
export function imageOf(
  map: Bijector | ManyToOneMap,
  x: Interval,
  { where = 'imageOf', endpoints = 'ignore' }: { where?: string; endpoints?: 'ignore' | 'strict' } = {},
): Interval {
  if (!intervalInside(x, map.domain, { endpoints }))
    throw new DomainError(
      where,
      `${where}: ${formatInterval(x)} is not inside the domain ${formatInterval(map.domain)} of ${map.name}`,
    )
  return hull(branchImages(map, x).map((p) => p.image))
}

/**
 * For each branch of a map (one for a bijector) that meets the interval `x`: the piece of `x` in the branch's domain
 * and that piece's image. Branches that miss `x` are left out. Does not check that `x` is inside the domain.
 *
 * @param map The bijector or many-to-one map.
 * @param x The interval to map.
 * @returns One entry per branch that meets `x`: the `branch`, the `piece` of `x` it covers, and its `image`.
 *
 * @example The square of [-1, 2], branch by branch
 * for (const { piece, image } of branchImages(squareMap, interval(-1, 2)))
 *   print(formatInterval(piece), 'maps to', formatInterval(image))
 */
export function branchImages(
  map: Bijector | ManyToOneMap,
  x: Interval,
): { branch: Branch; piece: Interval; image: Interval }[] {
  const m = asManyToOne(map)
  return m.branches.flatMap((branch) => {
    const piece = intersect(x, branch.domain)
    return piece ? [{ branch, piece, image: monotoneImage(m.forward, branch.increasing, piece, branch.domain) }] : []
  })
}
