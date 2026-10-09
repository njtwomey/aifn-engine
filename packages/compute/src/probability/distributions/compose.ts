/**
 * Distributions built from others: finite mixtures, zero inflation, `Independent` (a batch reinterpreted as one event,
 * as in PyTorch's `Independent`), `Transformed` (the pushforward of a univariate distribution through a monotone
 * bijection, with the change-of-variables log-Jacobian) and `Pushforward` (through a many-to-one map such as $x^2$,
 * summing over preimages).
 *
 * Log-densities are written with primitives, so they stay differentiable in the parameters of the distributions they
 * are built from. Quantities without a closed form (a mixture's entropy, a transformed distribution's moments) throw
 * `AifnError` and say to estimate them from samples.
 */

import { AifnError, DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { categorical as categoricalDraws, child, units, type Stream } from 'aifn-compute/foundation/random'
import { logAddExp, logSigmoid, sigmoid } from 'aifn-compute/numerics/special'
import {
  add,
  broadcastShapes,
  broadcastTo,
  div,
  exp,
  expandDims,
  eye,
  fromData,
  get,
  isTensor,
  log,
  log1p,
  mul,
  shapeOfValue,
  slice,
  sqrt,
  square,
  sub,
  sum,
  tensor,
  toFlat,
  unwrap,
  where,
  type Raw,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import {
  asManyToOne,
  branchImages,
  imageOf,
  supportInteriorPoint,
  intervalSupport,
  supportInterval,
  type Bijector,
  type Interval,
  type ManyToOneMap,
} from 'aifn-compute/probability/bijectors'
import type { Distribution, Multivariate, SampleOptions, Support, Univariate } from './types'
import {
  check,
  guard,
  invertCdf,
  mask,
  outside,
  raw,
  rawOnly,
  scalarOf,
  sumLastAxes,
  supportBounds,
  univariate,
} from './util'

/**
 * A method that throws `AifnError` saying that `what` has no closed form and should be estimated from samples.
 *
 * @param what The method's name in the message, such as `'Mixture.entropy'`.
 * @returns The method, which takes no arguments and always throws.
 */
const noClosedForm = (what: string) => () => {
  throw new AifnError(what, `${what}: no closed form (estimate it from samples)`)
}

/**
 * The smallest or largest raw value of a bound (a batch of bounds gives the outermost).
 *
 * @param v The bound: a number or a tensor of them (read through a traced value).
 * @param which `'min'` for the smallest value, `'max'` for the largest.
 */
function extreme(v: Value, which: 'min' | 'max'): number {
  const r = unwrap(v)
  const values = typeof r === 'number' ? [r] : toFlat(r)
  return which === 'min' ? Math.min(...values) : Math.max(...values)
}

// ── Mixture ──────────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * A finite mixture $p(x) = \sum_k w_k p_k(x)$ of univariate components with weights $w_k$ (normalised here). The
 * weights may be a batch $[\dots, K]$, one weight vector per element of the batch (a mixture density network's head:
 * weights and component parameters that vary by input); the batch shape is the broadcast of the weights' and the
 * components'. `logProb` is a log-sum-exp over components (differentiable in the weights and in every component's
 * parameters), the cdf and survival function are the weighted sums, and the quantile is found numerically. The mean
 * $\sum_k w_k \mu_k$ and the variance $\sum_k w_k (\sigma_k^2 + \mu_k^2) - \mu^2$ follow from the law of total
 * variance; the entropy and mode have no closed form and throw. Draws pick a component per draw (stream
 * `child(s, 'component')`) and take that component's draw (stream `child(s, 'draws', k)`). No `rsample`: the choice
 * of component is discrete. Throws `DomainError` when there are no components, the weights' last axis is not $K$ long
 * or a weight is negative.
 *
 * @param weights The $K$ non-negative weights, one per component, on the last axis: a number array or a tensor
 *   (possibly traced), or a batch `[..., K]`.
 * @param components The $K$ component distributions; the support is the hull of theirs, discrete when all are.
 * @returns The mixture.
 *
 * @example Two normals, an equal mixture
 * const d = Mixture([0.5, 0.5], [Normal(-1, 1), Normal(1, 1)])
 * print('p(0) =', d.prob(0), ' Normal(1, 1) at 0 =', Normal(1, 1).prob(0))
 * print('mean =', d.mean(), ' variance =', d.variance())
 * print('draws =', d.sample(stream(5), { shape: [5] }))
 */
export function Mixture(weights: Value | readonly number[], components: readonly Univariate[]): Univariate {
  const w: Value = Array.isArray(weights) ? tensor(weights as number[]) : (weights as Value)
  const K = components.length
  if (K === 0) throw new DomainError('Mixture', 'Mixture: needs at least one component')
  const wShape = shapeOfValue(w)
  if (wShape.length < 1 || wShape[wShape.length - 1] !== K)
    throw new DomainError('Mixture', `Mixture: needs ${K} weights, one per component (on the last axis)`)
  check('Mixture', 'weights', w, (x) => x >= 0, 'non-negative')
  const weightBatch = wShape.slice(0, -1)
  const normalised = weightBatch.length === 0 ? div(w, sum(w)) : div(w, sum(w, -1, true))
  const leading = weightBatch.map(() => null)
  const weight = (k: number) => (weightBatch.length === 0 ? get(normalised, k) : slice(normalised, ...leading, k))
  const logWeight = (k: number) => log(weight(k))
  const fold = (f: (k: number) => Value, combine: (a: Value, b: Value) => Value): Value => {
    let acc = f(0)
    for (let k = 1; k < K; k++) acc = combine(acc, f(k))
    return acc
  }
  const weighted = (f: (c: Univariate) => Value) => fold((k) => mul(weight(k), f(components[k])), add)
  const lower = Math.min(...components.map((c) => extreme(supportBounds(c.support)[0], 'min')))
  const upper = Math.max(...components.map((c) => extreme(supportBounds(c.support)[1], 'max')))
  const discrete = components.every((c) => c.discrete)
  const support: Support =
    lower === -Infinity && upper === Infinity
      ? { type: 'real' }
      : { type: discrete ? 'integers' : 'interval', lower, upper }
  const cdf = (x: Value) => weighted((c) => c.cdf(x))
  const mean = () => weighted((c) => c.mean())
  const params: Record<string, Value> = { weights: w }
  components.forEach((c, k) => {
    for (const [name, v] of Object.entries(c.params)) params[`${k}.${name}`] = v
  })
  return univariate<Value>({
    name: 'Mixture',
    params,
    batchShape: broadcastShapes(weightBatch, ...components.map((c) => c.batchShape)),
    support,
    discrete,
    logProb: (x) => fold((k) => add(logWeight(k), components[k].logProb(x)), logAddExp),
    cdf,
    survival: (x) => weighted((c) => c.survival(x)),
    quantile: (p) => invertCdf('Mixture', cdf, p, [], support, discrete),
    sample: (s: Stream, shape: number[]) => {
      const choice = categoricalDraws(child(s, 'component'), raw(normalised, 'Mixture') as Tensor, { shape }) as Tensor
      const picks = toFlat(choice)
      const draws = components.map((c, k) => {
        // `shape` is the full draw shape (sample axes, then the batch); a component prepends its own batch.
        const t = c.sample(child(s, 'draws', k), { shape: shape.slice(0, shape.length - c.batchShape.length) })
        return toFlat(broadcastTo(t, shape))
      })
      return fromData(
        Float64Array.from(picks, (k, i) => draws[k][i]),
        shape,
      )
    },
    mean,
    variance: () =>
      sub(
        weighted((c) => add(c.variance(), square(c.mean()))),
        square(mean()),
      ),
    entropy: noClosedForm('Mixture.entropy'),
    mode: noClosedForm('Mixture.mode'),
  })
}

// ── ZeroInflated ─────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * A zero-inflated discrete distribution (Lambert, 1992): with probability $\pi$ the outcome is a structural zero,
 * otherwise it is drawn from `base`, so $\pr(0) = \pi + (1 - \pi) p(0)$ and $\pr(k) = (1 - \pi) p(k)$ for
 * $k \ne 0$. `logProb` is differentiable in $\pi$ and in the base's parameters; with $\pi = 0$ it is the base's. The
 * mean is $(1 - \pi)\mu$ and the variance $(1 - \pi)(\sigma^2 + \pi\mu^2)$, with $\mu$ and $\sigma^2$ the base's.
 * Draws pick structural zeros on stream `child(s, 'structural')` and base draws on `child(s, 'base')`. The entropy and
 * mode throw. The base must be discrete with 0 in its support (a zero-inflated Poisson, negative binomial or
 * Bernoulli), else `DomainError`.
 *
 * @param pi The probability $\pi \in [0, 1]$ of a structural zero, or `{ logits }`, its log-odds $\eta$ (stable near 0
 *   and 1: $\log \pi = \log \sigma(\eta)$). It may be a batch, broadcast against the base's batch.
 * @param base The discrete distribution of the outcomes that are not structural zeros.
 * @returns The zero-inflated distribution.
 *
 * @example A zero-inflated Poisson
 * const d = ZeroInflated(0.2, Poisson(2))
 * print('P(0) =', d.prob(0), ' 0.2 + 0.8 e^-2 =', 0.2 + 0.8 * Math.exp(-2))
 * print('P(1) =', d.prob(1))
 * print('mean =', d.mean(), ' variance =', d.variance())
 */
export function ZeroInflated(pi: Value | { logits: Value }, base: Univariate): Univariate {
  if (!base.discrete) throw new DomainError('ZeroInflated', 'ZeroInflated: the base must be discrete')
  const [lo, hi] = supportBounds(base.support)
  if (extreme(lo, 'min') > 0 || extreme(hi, 'max') < 0)
    throw new DomainError('ZeroInflated', 'ZeroInflated: 0 must be in the support of the base')
  const fromLogits = typeof pi === 'object' && pi !== null && 'logits' in pi
  const logits: Value | null = fromLogits ? (pi as { logits: Value }).logits : null
  const probs: Value = logits !== null ? sigmoid(logits) : (pi as Value)
  if (logits === null) check('ZeroInflated', 'π', probs, (x) => x >= 0 && x <= 1, 'in [0, 1]')
  const logPi = logits !== null ? logSigmoid(logits) : log(probs)
  const logKeep = logits !== null ? logSigmoid(mul(-1, logits)) : log1p(mul(-1, probs))
  const keep = sub(1, probs)
  const isZero = (x: Value) => mask([x], (v) => v === 0)
  const atLeastZero = (x: Value) => mask([x], (v) => v >= 0)
  const support: Support = base.support
  const cdf = (x: Value) => add(mul(keep, base.cdf(x)), mul(probs, atLeastZero(x)))
  const params: Record<string, Value> = logits !== null ? { logits } : { pi: probs }
  for (const [name, v] of Object.entries(base.params)) params[`base.${name}`] = v
  const batchShape = broadcastShapes(shapeOfValue(probs), base.batchShape)
  return univariate<Value>({
    name: 'ZeroInflated',
    params,
    batchShape,
    support,
    discrete: true,
    logProb: (x) => {
      const nonZero = add(logKeep, base.logProb(x))
      return where(isZero(x), logAddExp(logPi, add(logKeep, base.logProb(0))), nonZero)
    },
    cdf,
    survival: (x) => add(mul(keep, base.survival(x)), mul(probs, sub(1, atLeastZero(x)))),
    quantile: (p) => invertCdf('ZeroInflated', cdf, p, batchShape, support, true),
    sample: (s: Stream, shape: number[]) => {
      const p = toFlat(broadcastTo(fromData(Float64Array.from(flatOf(probs)), shapeOfValue(probs)), shape))
      const u = units(child(s, 'structural'), p.length)
      const draws = toFlat(
        broadcastTo(
          base.sample(child(s, 'base'), { shape: shape.slice(0, shape.length - base.batchShape.length) }) as Tensor,
          shape,
        ),
      )
      return fromData(
        Float64Array.from(draws, (d, i) => (u[i] < p[i] ? 0 : d)),
        shape,
      )
    },
    mean: () => mul(keep, base.mean()),
    variance: () => mul(keep, add(base.variance(), mul(probs, square(base.mean())))),
    entropy: noClosedForm('ZeroInflated.entropy'),
    mode: noClosedForm('ZeroInflated.mode'),
  })
}

/**
 * The raw values of a parameter as a flat array (a number gives one value). Throws for a traced value, which cannot
 * be sampled.
 *
 * @param v The parameter.
 */
function flatOf(v: Value): Float64Array {
  const r = raw(v, 'ZeroInflated.sample')
  return typeof r === 'number' ? Float64Array.of(r) : Float64Array.from(toFlat(r))
}

// ── Independent ──────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Reinterpret the last `reinterpreted` batch axes of `base` as event axes (PyTorch's `Independent`): a batch of $d$
 * independent normals becomes one distribution over vectors in $\reals^d$ with a diagonal covariance,
 * $\log p(\xvec) = \sum_i \log p_i(x_i)$. `logProb` and `entropy` sum over the reinterpreted axes; moments and draws
 * (and `rsample`, when the base has it) are the base's. `covariance()` is available for a univariate base with one
 * reinterpreted axis (a diagonal matrix) and throws otherwise.
 *
 * @param base The distribution whose batch axes become event axes.
 * @param reinterpreted How many trailing batch axes to move into the event, from 0 to the base's batch rank (else
 *   `ShapeError`).
 * @returns The distribution, with `base` and `reinterpreted` kept on it.
 *
 * @example Three independent standard normals as one vector
 * const d = Independent(Normal(tensor([0, 0, 0]), 1))
 * print('batch', d.batchShape, ' event', d.eventShape)
 * print('log p(0) =', d.logProb(tensor([0, 0, 0])), ' 3 log Normal(0, 1) at 0 =', 3 * Normal(0, 1).logProb(0))
 * print('covariance =', d.covariance())
 */
export function Independent(base: Distribution, reinterpreted = 1): Multivariate {
  const b = base.batchShape
  if (!Number.isInteger(reinterpreted) || reinterpreted < 0 || reinterpreted > b.length)
    throw new ShapeError('Independent', `Independent: can reinterpret 0 … ${b.length} batch axes, got ${reinterpreted}`)
  const batchShape = b.slice(0, b.length - reinterpreted)
  const eventShape = [...b.slice(b.length - reinterpreted), ...base.eventShape]
  const baseLogProb = base.logProb as (x: Value) => Value
  const baseEntropy = base.entropy as () => Value
  return {
    kind: 'distribution',
    name: 'Independent',
    params: base.params,
    batchShape,
    eventShape,
    support: base.support,
    discrete: base.discrete,
    base,
    reinterpreted,
    logProb: (x: Value) => sumLastAxes(baseLogProb(x), reinterpreted),
    prob: (x: Value) => exp(sumLastAxes(baseLogProb(x), reinterpreted)),
    mean: () => base.mean(),
    variance: () => base.variance(),
    stddev: () => sqrt(base.variance() as Value),
    covariance: () => {
      if (base.eventShape.length !== 0 || reinterpreted !== 1)
        throw new AifnError(
          'Independent',
          'Independent.covariance: only for a univariate base with one reinterpreted axis',
        )
      const d = eventShape[0]
      return mul(expandDims(base.variance() as Value, -1), eye(d))
    },
    entropy: () => sumLastAxes(baseEntropy(), reinterpreted),
    mode: () => base.mode(),
    sample: (s: Stream, options?: SampleOptions) => {
      const t = base.sample(s, { shape: options?.shape ?? [] })
      return typeof t === 'number' ? fromData(new Float64Array([t]), []) : t
    },
    ...(base.rsample
      ? { rsample: (s: Stream, options?: SampleOptions) => base.rsample!(s, { shape: options?.shape ?? [] }) }
      : {}),
  } as unknown as Multivariate
}

// ── Transformed ──────────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The distribution of $y = f(x)$ for $x \sim$ `base` and a monotone bijector $f$, by the change of variables
 * $\log p(y) = \log p_{\text{base}}(f^{-1}(y)) - \log \lvert f'(f^{-1}(y)) \rvert$. The support is the image of the
 * base's support under $f$ (`imageOf`), with its open ends marked, e.g. $(0, 1)$ for a normal through the sigmoid; a
 * base whose support is not inside $f$'s domain (a normal through log) is a `DomainError`. The cdf is the base's cdf
 * at $f^{-1}(y)$ (its survival function when $f$ decreases), and quantiles and draws map through $f$ (a pathwise
 * `rsample` when the base has one). Moments and entropy have no closed form in general and throw.
 *
 * @param base The univariate distribution of $x$.
 * @param bijector The monotone map $f$ with its inverse and $\log \lvert f' \rvert$, such as the bijectors of
 *   `aifn-compute/probability/bijectors`.
 * @returns The distribution of $y$.
 *
 * @example The exponential of a standard normal is the log-normal
 * // expBijector of aifn-compute/probability/bijectors, written out: Transformed(Normal(0, 1), expBijector).
 * const open = (lower, upper) => ({ lower, upper, lowerOpen: true, upperOpen: true })
 * const expBijector = { name: 'exp', forward: exp, inverse: log, logAbsDetJacobian: (x) => x, increasing: true,
 *   domain: open(-Infinity, Infinity), codomain: open(0, Infinity) }
 * const d = Transformed(Normal(0, 1), expBijector)
 * print('p(2) =', d.prob(2), ' LogNormal(0, 1) at 2 =', LogNormal(0, 1).prob(2))
 * print('median =', d.quantile(0.5), ' P(Y ≤ e) =', d.cdf(Math.E))
 */
export function Transformed(base: Univariate, bijector: Bijector): Univariate {
  // The image of the base's support; throws when that support does not fit the bijector's domain.
  const image = imageOf(bijector, supportInterval(base.support), {
    where: `Transformed(${base.name}, ${bijector.name})`,
    endpoints: base.discrete ? 'strict' : 'ignore',
  })
  const [lo, hi] = supportBounds(base.support).map((b) => raw(b, 'Transformed'))
  const ends = [unwrap(bijector.forward(lo)), unwrap(bijector.forward(hi))] as Raw[]
  const [lower, upper] = bijector.increasing ? ends : [ends[1], ends[0]]
  const real = extreme(lower, 'min') === -Infinity && extreme(upper, 'max') === Infinity
  const support: Support = real
    ? { type: 'real' }
    : base.discrete
      ? { type: 'integers', lower, upper }
      : { type: 'interval', lower, upper, lowerOpen: image.lowerOpen, upperOpen: image.upperOpen }
  const inside = (y: Value) => (real ? 1 : mask([y, lower, upper], (v, a, b) => v >= a && v <= b))
  // A point inside the codomain (the image of the base median), substituted for values outside it before inverting.
  let safe: number | undefined
  const safePoint = () => {
    if (safe === undefined) {
      const m = unwrap(bijector.forward(base.quantile(0.5)))
      const v = typeof m === 'number' ? m : isTensor(m) ? toFlat(m)[0] : NaN
      safe = Number.isFinite(v) ? v : 1
    }
    return safe
  }
  const pull = (y: Value, ok: Raw) => bijector.inverse(ok === 1 ? y : guard(y, ok, safePoint()))
  const below = (y: Value) => mask([y, lower], (v, a) => v < a)
  const direct = (y: Value, f: (x: Value) => Value, g: (x: Value) => Value, low: number, high: number) => {
    const ok = inside(y)
    const x = pull(y, ok)
    return where(below(y), low, outside(ok, bijector.increasing ? f(x) : g(x), high))
  }
  return univariate<Value>({
    name: 'Transformed',
    params: base.params,
    batchShape: base.batchShape,
    support,
    discrete: base.discrete,
    logProb: (y) => {
      const ok = inside(y)
      const x = pull(y, ok)
      return outside(ok, sub(base.logProb(x) as Value, bijector.logAbsDetJacobian(x)), -Infinity)
    },
    cdf: (y) =>
      direct(
        y,
        (x) => base.cdf(x),
        (x) => base.survival(x),
        0,
        1,
      ),
    survival: (y) =>
      direct(
        y,
        (x) => base.survival(x),
        (x) => base.cdf(x),
        1,
        0,
      ),
    quantile: (p) => bijector.forward(base.quantile(bijector.increasing ? p : sub(1, p))),
    sample: (s, shape) => {
      const t = base.sample(s, { shape: shape.slice(0, shape.length - base.batchShape.length) })
      return unwrap(bijector.forward(t)) as Tensor
    },
    // A pathwise draw exists when the base has one: f of the base's pathwise draw (differentiable through f).
    ...(base.rsample
      ? {
          rsample: (s: Stream, shape: number[], scalar: boolean) =>
            bijector.forward(
              scalar
                ? base.rsample!(s)
                : base.rsample!(s, { shape: shape.slice(0, shape.length - base.batchShape.length) }),
            ),
        }
      : {}),
    mean: noClosedForm('Transformed.mean'),
    variance: noClosedForm('Transformed.variance'),
    entropy: noClosedForm('Transformed.entropy'),
    mode: noClosedForm('Transformed.mode'),
  })
}

// ── Pushforward through a many-to-one map ───────────────────────────────────────────────────────────────────────────

/**
 * The distribution of $y = f(x)$ for a continuous, unbatched $x \sim$ `base` and a map $f$ made of monotone branches
 * (such as `squareMap`; a bijector is one branch). The density sums the change of variables over the preimages,
 * $p(y) = \sum_k p_{\text{base}}(x_k) / \lvert f'(x_k) \rvert$ with $x_k = f_k^{-1}(y)$, as a log-sum-exp
 * (differentiable in $y$ and the base's parameters). A branch contributes only where $y$ is in its image. The cdf adds
 * each branch's base mass on $\{x : f(x) \le y\}$ (not differentiable), the quantile inverts it numerically, and
 * draws are $f$ of the base's draws. The support is the image of the base's support (`imageOf`). Moments and entropy
 * throw. A discrete or batched base is a `DomainError`.
 *
 * @param base The continuous, unbatched distribution of $x$.
 * @param map The map $f$: a many-to-one map with its monotone branches, or a bijector (one branch).
 * @returns The distribution of $y$.
 *
 * @example The square of a standard normal is chi-squared with one degree of freedom
 * // squareMap of aifn-compute/probability/bijectors, written out: Pushforward(Normal(0, 1), squareMap).
 * const span = (lower, upper, lowerOpen, upperOpen) => ({ lower, upper, lowerOpen, upperOpen })
 * const logAbsDetJacobian = (x) => log(abs(mul(2, x)))
 * const branches = [
 *   { domain: span(-Infinity, 0, true, false), inverse: (y) => neg(sqrt(y)), logAbsDetJacobian, increasing: false },
 *   { domain: span(0, Infinity, false, true), inverse: sqrt, logAbsDetJacobian, increasing: true },
 * ]
 * const squareMap = { name: 'square', forward: square, branches, domain: span(-Infinity, Infinity, true, true),
 *   codomain: span(0, Infinity, false, true) }
 * const d = Pushforward(Normal(0, 1), squareMap)
 * print('p(1) =', d.prob(1), ' ChiSquare(1) at 1 =', ChiSquare(1).prob(1))
 * print('P(Y ≤ 1) =', d.cdf(1), ' ChiSquare(1) =', ChiSquare(1).cdf(1))
 */
export function Pushforward(base: Univariate, map: ManyToOneMap | Bijector): Univariate {
  const where = `Pushforward(${base.name}, ${map.name})`
  if (base.discrete) throw new DomainError(where, `${where}: needs a continuous base (branches may share an end point)`)
  if (base.batchShape.length !== 0) throw new DomainError(where, `${where}: needs an unbatched base`)
  const m = asManyToOne(map)
  const baseSet = supportInterval(base.support)
  const image = imageOf(m, baseSet, { where })
  const support = intervalSupport(image)
  // Each branch's piece of the base support, its image, and a point inside that image to guard values outside it.
  const pieces = branchImages(m, baseSet).map((p) => ({ ...p, safe: supportInteriorPoint(p.image) }))
  const inRange = (y: Value, i: Interval) => mask([y], (v) => v >= i.lower && v <= i.upper)
  const at = (f: (x: number) => Value, x: number) => scalarOf(raw(f(x), `${where}.cdf`))
  const F = (x: number) => (x === -Infinity ? 0 : x === Infinity ? 1 : at((v) => base.cdf(v), x))
  const cdf = (y: Value) =>
    rawOnly(`${where}.cdf`, [y], (v) => {
      let total = 0
      for (const { branch, piece, image: i } of pieces) {
        const { lower, upper } = piece
        const mass = F(upper) - F(lower)
        if (v >= i.upper) total += mass
        else if (v > i.lower) {
          const x = at((t) => branch.inverse(t), v)
          total += branch.increasing ? F(x) - F(lower) : F(upper) - F(x)
        }
      }
      return total
    })
  return univariate<Value>({
    name: 'Pushforward',
    params: base.params,
    batchShape: [],
    support,
    logProb: (y) => {
      const terms = pieces.map(({ branch, image: i, safe }) => {
        const ok = inRange(y, i)
        const x = branch.inverse(guard(y, ok, safe))
        return outside(ok, sub(base.logProb(x) as Value, branch.logAbsDetJacobian(x)), -Infinity)
      })
      return terms.reduce((a, b) => logAddExp(a, b))
    },
    cdf,
    quantile: (p) => invertCdf('Pushforward', cdf, p, [], support, false),
    sample: (s, shape) => unwrap(m.forward(base.sample(s, { shape }))) as Tensor,
    ...(base.rsample
      ? {
          rsample: (s: Stream, shape: number[], scalar: boolean) =>
            m.forward(scalar ? base.rsample!(s) : base.rsample!(s, { shape })),
        }
      : {}),
    mean: noClosedForm('Pushforward.mean'),
    variance: noClosedForm('Pushforward.variance'),
    entropy: noClosedForm('Pushforward.entropy'),
    mode: noClosedForm('Pushforward.mode'),
  })
}
