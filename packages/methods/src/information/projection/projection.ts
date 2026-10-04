/**
 * Projections of a univariate distribution onto the normal family under KL: the moment projection (forward KL,
 * argmin_q KL(p ‖ q)) and the information projection (reverse KL, argmin_q KL(q ‖ p)), each fitted by L-BFGS with
 * gradients from `aifn-compute/foundation/autodiff`, keeping the optimiser's path.
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

/** Which KL a projection minimises: `forward` KL(p ‖ q) (moment projection) or `reverse` KL(q ‖ p) (information). */
export type KlDirection = 'forward' | 'reverse'

/** A point on the optimiser's path: q = N(loc, scale²) and the objective there. */
export type NormalFitStep = { loc: number; scale: number; objective: number }

/** The fitted normal and how it was reached. */
export type NormalProjection = {
  direction: KlDirection
  loc: number
  scale: number
  /**
   * The minimised objective at the fit: the cross-entropy H(p, q) for `forward` (KL(p ‖ q) plus the constant H(p)),
   * KL(q ‖ p) for `reverse` (with E_q[log p] by Gauss–Hermite quadrature).
   */
  objective: number
  converged: boolean
  /** Every L-BFGS iterate, from the start to the fit. */
  path: NormalFitStep[]
}

export type NormalProjectionOptions = {
  /** The starting normal (default N(0, 1)). */
  init?: { loc: number; scale: number }
  /** L-BFGS steps at most (default 200). */
  maxSteps?: number
  /** Gauss–Hermite nodes for E_q (reverse; default 40) or Gauss–Legendre nodes per piece of p's range (forward; 32). */
  nodes?: number
}

const scalar = (v: Value): number => {
  const r = unwrap(v)
  return typeof r === 'number' ? r : toFlat(r)[0]
}

/** Quantiles of p that split its range into pieces for Gauss–Legendre, from 1e-9 to 1 − 1e-9. */
const PIECES = [1e-9, 1e-4, 0.01, 0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95, 0.99, 1 - 1e-4, 1 - 1e-9]

/** Nodes xⱼ and weights wⱼ p(xⱼ) with Σ wⱼ p(xⱼ) f(xⱼ) ≈ E_p f(X), normalised to sum to 1. */
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
 * Fits q = N(μ, σ²) to a continuous univariate p by minimising KL in the given direction, over (μ, log σ) with L-BFGS
 * and autodiff gradients.
 *
 * - `forward`: minimises the cross-entropy −E_p[log q(X)], with E_p by piecewise Gauss–Legendre over p's range. The
 *   minimiser is the moment match μ = E_p[X], σ² = Var_p[X], whatever the start: q covers all of p's mass.
 * - `reverse`: minimises KL(q ‖ p) = −H(q) − E_q[log p(X)], with E_q by Gauss–Hermite in the reparameterisation
 *   X = μ + √2 σ z. For a multimodal p the minimiser depends on the start: q settles on one mode (mode seeking).
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
