/**
 * Projections of a continuous univariate distribution onto the normal family under KL: the moment projection
 * $\argmin_q \KL(p \,\Vert\, q)$ (forward KL) and the information projection $\argmin_q \KL(q \,\Vert\, p)$
 * (reverse KL).
 *
 * Each is fitted by L-BFGS (`aifn-compute/optim/second-order`) over $(\mu, \log\sigma)$, with gradients from
 * `aifn-compute/foundation/autodiff` and expectations by quadrature (`aifn-compute/numerics/quadrature`), and keeps the
 * optimiser's path so a figure can show the fit moving. The forward projection matches the mean and variance of $p$
 * and covers all its mass; the reverse projection seeks a mode (Bishop, 2006, §10.1.2; Murphy, 2012, §21.2.2).
 */

import { valueAndGrad } from 'aifn-compute/foundation/autodiff'
import { Normal, type AnyUnivariate, type Distribution } from 'aifn-compute/probability/distributions'
import { lbfgs } from 'aifn-compute/optim/second-order'
import { gaussHermite, gaussLegendre } from 'aifn-compute/numerics/quadrature'
import {
  add,
  exp,
  get,
  mul,
  neg,
  sum,
  tensor,
  toFlat,
  unwrap,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { DomainError } from 'aifn-compute/foundation/errors'

/**
 * Which KL a projection minimises: `forward` $\KL(p \,\Vert\, q)$ (the moment projection) or `reverse`
 * $\KL(q \,\Vert\, p)$ (the information projection).
 */
export type KlDirection = 'forward' | 'reverse'

/**
 * A point on the optimiser's path: the normal $q = \Gauss(\text{loc}, \text{scale}^2)$ given by `loc` and `scale`,
 * and the `objective` there (as in `NormalProjection`).
 */
export type NormalFitStep = { loc: number; scale: number; objective: number }

/** The fitted normal and how it was reached. */
export type NormalProjection = {
  /** The direction of the KL minimised, as given. */
  direction: KlDirection
  /** The fitted mean $\mu$. */
  loc: number
  /** The fitted standard deviation $\sigma$. */
  scale: number
  /**
   * The minimised objective at the fit: the cross-entropy $H(p, q)$ for `forward` ($\KL(p \,\Vert\, q)$ plus the
   * constant $H(p)$), $\KL(q \,\Vert\, p)$ for `reverse` (with $\expect_q[\log p]$ by Gauss–Hermite quadrature).
   */
  objective: number
  /** Whether L-BFGS met its convergence test within `maxSteps`. */
  converged: boolean
  /** Every L-BFGS iterate, from the start to the fit. */
  path: NormalFitStep[]
}

/** Options of `normalProjection`. */
export type NormalProjectionOptions = {
  /** The starting normal, its `loc` and positive `scale` (default $\Gauss(0, 1)$). */
  init?: { loc: number; scale: number }
  /** L-BFGS steps at most (default 200). */
  maxSteps?: number
  /**
   * Gauss–Hermite nodes for $\expect_q$ (reverse; default 40) or Gauss–Legendre nodes per piece of $p$'s range
   * (forward; default 32).
   */
  nodes?: number
}

/**
 * A scalar value as a number: a number as it is, a tensor's first entry.
 *
 * @param v The value, untraced.
 * @returns The number.
 */
const scalar = (v: Value): number => {
  const r = unwrap(v)
  return typeof r === 'number' ? r : toFlat(r)[0]
}

/** Quantiles of $p$ that split its range into pieces for Gauss–Legendre, from $10^{-9}$ to $1 - 10^{-9}$. */
const PIECES = [1e-9, 1e-4, 0.01, 0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95, 0.99, 1 - 1e-4, 1 - 1e-9]

/**
 * A quadrature rule for $\expect_p$: nodes $x_j$ and weights $w_j p(x_j)$, normalised to sum to 1, with
 * $\sum_j w_j p(x_j) f(x_j) \approx \expect_p[f(X)]$. The range between the quantiles $10^{-9}$ and $1 - 10^{-9}$ is
 * cut at the quantiles in `PIECES` (duplicates and non-finite cuts dropped) and each piece gets an $n$-node
 * Gauss–Legendre rule; nodes where the density is zero or not finite are left out. The mass beyond the outer cuts is
 * ignored.
 *
 * @param p The distribution: its `quantile` and `prob` are evaluated at numbers.
 * @param n The number of Gauss–Legendre nodes per piece.
 * @returns The nodes `x` and the normalised weights `w`, as tensors of equal length.
 */
function pRule(p: AnyUnivariate, n: number): { x: Tensor; w: Tensor } {
  const cuts = [...new Set(PIECES.map((u) => scalar(p.quantile(u))).filter(Number.isFinite))].sort((a, b) => a - b)
  const xs: number[] = []
  const ws: number[] = []
  for (let i = 0; i + 1 < cuts.length; i++) {
    const { nodes, weights } = gaussLegendre(n, [cuts[i], cuts[i + 1]])
    const x = toFlat(nodes)
    const w = toFlat(weights)
    for (let j = 0; j < x.length; j++) {
      const px = scalar(p.prob(x[j]))
      if (Number.isFinite(px) && px > 0) {
        xs.push(x[j])
        ws.push(w[j] * px)
      }
    }
  }
  const total = ws.reduce((a, b) => a + b, 0)
  return { x: tensor(xs), w: tensor(ws.map((v) => v / total)) }
}

/**
 * Fits $q = \Gauss(\mu, \sigma^2)$ to a continuous univariate $p$ by minimising KL in the given direction, over
 * $(\mu, \log\sigma)$ with L-BFGS and autodiff gradients.
 *
 * - `forward`: minimises the cross-entropy $-\expect_p[\log q(X)]$, with $\expect_p$ by piecewise Gauss–Legendre over
 *   $p$'s range. The minimiser is the moment match $\mu = \expect_p[X]$, $\sigma^2 = \var_p[X]$, whatever the start:
 *   $q$ covers all of $p$'s mass.
 * - `reverse`: minimises $\KL(q \,\Vert\, p) = -H(q) - \expect_q[\log p(X)]$, with $\expect_q$ by Gauss–Hermite in
 *   the reparameterisation $X = \mu + \sqrt{2}\sigma z$. For a multimodal $p$ the minimiser depends on the start: $q$
 *   settles on one mode (mode seeking).
 *
 * Throws `DomainError` for a batched, multivariate or discrete $p$, or a non-positive initial scale. Any other
 * `direction` than `'forward'` is taken as `'reverse'`.
 *
 * @param p The distribution to project: unbatched, continuous and univariate. `forward` evaluates its `quantile` and
 *   `prob` at numbers; `reverse` evaluates its `logProb` at traced values, which must be differentiable.
 * @param direction Which KL to minimise.
 * @param options The start, the step budget and the quadrature.
 * @param options.init The starting normal, its `loc` and positive `scale`.
 * @param options.maxSteps The largest number of L-BFGS steps.
 * @param options.nodes The quadrature nodes: Gauss–Hermite nodes for `reverse` (default 40), Gauss–Legendre nodes per
 *   piece of $p$'s range for `forward` (default 32).
 * @returns The fitted normal, the objective there, whether L-BFGS converged and its path from the start.
 *
 * @example The moment projection of a uniform matches its mean and variance
 * // Uniform(0, 1), written out with the fields a forward projection reads.
 * const uniform01 = {
 *   kind: 'distribution', name: 'Uniform', params: {}, batchShape: [], eventShape: [], discrete: false,
 *   prob: (x) => (x >= 0 && x <= 1 ? 1 : 0),
 *   quantile: (u) => u,
 * }
 * const fit = normalProjection(uniform01, 'forward')
 * print('loc =', fit.loc, 'scale =', fit.scale, 'after', fit.path.length - 1, 'steps')
 * print('error in mean =', Math.abs(fit.loc - 0.5), 'in sd =', Math.abs(fit.scale - Math.sqrt(1 / 12)))
 *
 * @example The information projection of a bimodal density picks one mode
 * // ½ N(−2, 0.5²) + ½ N(2, 0.5²), its log-density written with tensor operations so that it can be differentiated.
 * const component = (x, m) => sub(mul(-2, square(sub(x, m))), Math.log(0.5 * Math.sqrt(2 * Math.PI)))
 * const bimodal = {
 *   kind: 'distribution', name: 'Mixture', params: {}, batchShape: [], eventShape: [], discrete: false,
 *   logProb: (x) => sub(logsumexp(stack([component(x, -2), component(x, 2)]), 0), Math.log(2)),
 * }
 * const fit = normalProjection(bimodal, 'reverse', { init: { loc: 1, scale: 1 } })
 * print('loc =', fit.loc, 'scale =', fit.scale)
 * print('KL(q ‖ p) =', fit.objective, 'log 2 =', Math.log(2))
 */
export function normalProjection(
  p: Distribution,
  direction: KlDirection,
  { init = { loc: 0, scale: 1 }, maxSteps = 200, nodes }: NormalProjectionOptions = {},
): NormalProjection {
  if (p.eventShape.length !== 0 || p.batchShape.length !== 0 || p.discrete)
    throw new DomainError('normalProjection', 'normalProjection: needs an unbatched continuous univariate distribution')
  const target = p as AnyUnivariate
  if (!(init.scale > 0))
    throw new DomainError('normalProjection', 'normalProjection: the initial scale must be positive')

  let f: (theta: Value) => Value
  if (direction === 'forward') {
    const { x, w } = pRule(target, nodes ?? 32)
    f = (theta) => neg(sum(mul(w, Normal(get(theta, 0), exp(get(theta, 1))).logProb(x))))
  } else {
    const { nodes: z, weights } = gaussHermite(nodes ?? 40)
    const w = mul(weights, 1 / Math.sqrt(Math.PI))
    f = (theta) => {
      const m = get(theta, 0)
      const s = exp(get(theta, 1))
      const q = Normal(m, s)
      const x = add(m, mul(mul(Math.SQRT2, s), z))
      return neg(add(q.entropy(), sum(mul(w, target.logProb(x) as Value))))
    }
  }
  const vg = valueAndGrad(f)
  const objective = (theta: Tensor) => {
    const r = vg(theta)
    return { value: scalar(r.value), grad: r.grad as Tensor }
  }
  const t = trace(lbfgs(objective), { x0: [init.loc, Math.log(init.scale)] }, maxSteps)
  const path = t.steps.map((s) => {
    const [m, logS] = toFlat(s.x)
    return { loc: m, scale: Math.exp(logS), objective: s.value }
  })
  const last = t.final
  const end = path[path.length - 1]
  return {
    direction,
    loc: end.loc,
    scale: end.scale,
    objective: end.objective,
    converged: last.converged,
    path,
  }
}
