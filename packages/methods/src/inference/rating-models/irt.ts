/**
 * Item response theory (Rasch, 1960; Birnbaum in Lord and Novick, 1968): person p answers item i correctly with
 * probability P = c + (1 − c) σ(aᵢ(θₚ − bᵢ)), with ability θ, difficulty b, discrimination a (1 in the one-parameter
 * Rasch model) and guessing floor c (0 here unless given). The item information aᵢ² P(1 − P) (for c = 0) is the
 * Fisher information an item carries about θ; it peaks at θ = b.
 *
 * `fitIrt` estimates the item parameters by marginal maximum likelihood (Bock and Aitkin, 1981): abilities θ ~ N(0, 1)
 * are integrated out on a quadrature grid, so the number of parameters does not grow with the persons (joint
 * maximum likelihood is inconsistent and inflates the discriminations). Weak Gaussian priors b ~ N(0, 2²) and
 * log a ~ N(0, 0.5²) keep items everyone (or no one) answers finite. The marginal log-likelihood is written with tensor
 * primitives, differentiated by autodiff and maximised by L-BFGS; abilities are then their posterior means (EAP).
 */

import { valueAndGrad } from 'aifn-compute/foundation/autodiff'
import {
  add,
  logsumexp,
  matmul,
  transpose,
  exp,
  fromData,
  mul,
  reshape,
  slice,
  square,
  sub,
  sum,
  toFlat,
  unwrap,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { logSigmoid } from 'aifn-compute/numerics/special'
import { minimize } from 'aifn-compute/optim/minimize'

/** P(correct) = c + (1 − c) σ(a(θ − b)). */
export function irtProbability(theta: number, a: number, b: number, c = 0): number {
  return c + (1 - c) / (1 + Math.exp(-a * (theta - b)))
}

/**
 * The Fisher information of one item about θ: a² (P − c)² (1 − P) / ((1 − c)² P), which is a² P(1 − P) when c = 0.
 */
export function itemInformation(theta: number, a: number, b: number, c = 0): number {
  const p = irtProbability(theta, a, b, c)
  return (a * a * (p - c) * (p - c) * (1 - p)) / ((1 - c) * (1 - c) * p)
}

/** Options of `fitIrt`. */
export interface IrtFitOptions {
  /** `1pl` (Rasch: every a = 1) or `2pl` (default `2pl`). */
  model?: '1pl' | '2pl'
  /** Prior standard deviations of θ, b and log a (1, 2, 0.5). */
  abilitySd?: number
  difficultySd?: number
  logDiscriminationSd?: number
  /** Quadrature nodes over θ (default 41). */
  nodes?: number
  /** Most L-BFGS steps (default 500). */
  maxSteps?: number
}

/** A fitted IRT model. */
export interface IrtFit {
  readonly model: '1pl' | '2pl'
  /** Abilities θ, one per person: posterior means given the fitted items (EAP). */
  readonly ability: Float64Array
  /** Difficulties b and discriminations a, one per item. */
  readonly difficulty: Float64Array
  readonly discrimination: Float64Array
  /** The marginal log-likelihood of the responses at the fit (θ integrated out; without the item priors). */
  readonly logLikelihood: number
  readonly converged: boolean
}

/**
 * Fit a 1PL or 2PL model to responses [persons, items] of 0 and 1; NaN marks an item a person did not answer.
 */
export function fitIrt(responses: Tensor, options: IrtFitOptions = {}): IrtFit {
  const { model = '2pl', abilitySd = 1, difficultySd = 2, logDiscriminationSd = 0.5, maxSteps = 500 } = options
  const nodes = options.nodes ?? 41
  const [P, I] = responses.shape
  const raw = toFlat(responses)
  const correct = new Float64Array(P * I)
  const wrong = new Float64Array(P * I)
  for (let k = 0; k < P * I; k++)
    if (Number.isFinite(raw[k])) {
      correct[k] = raw[k]
      wrong[k] = 1 - raw[k]
    }
  const C = fromData(correct, [P, I])
  const W = fromData(wrong, [P, I])
  // Quadrature over θ ~ N(0, σ²): an even grid on ±5σ with normalised normal weights.
  const grid = Float64Array.from({ length: nodes }, (_, q) => abilitySd * (-5 + (10 * q) / (nodes - 1)))
  const logWeights = Float64Array.from(grid, (t) => (-0.5 * t * t) / (abilitySd * abilitySd))
  const lz = Math.log(logWeights.reduce((s, v) => s + Math.exp(v), 0))
  for (let q = 0; q < nodes; q++) logWeights[q] -= lz
  const T = fromData(grid, [nodes, 1])
  const LW = fromData(logWeights, [1, nodes])
  const twoPl = model === '2pl'
  const dim = twoPl ? 2 * I : I

  /** log p(responses of p, θ = node q) as [P, Q], from item rows b and log a [1, I]. */
  const joint = (b: Value, logA: Value | null): Value => {
    const margin = sub(T, b)
    const z = logA ? mul(exp(logA), margin) : margin
    // [P, I] @ [I, Q]: the log-likelihood of every person's answers at every node.
    const ll = add(matmul(C, transpose(logSigmoid(z))), matmul(W, transpose(logSigmoid(mul(-1, z)))))
    return add(ll, LW)
  }
  const parts = (x: Value) => ({
    b: reshape(slice(x, [0, I]), [1, I]),
    logA: twoPl ? reshape(slice(x, [I, 2 * I]), [1, I]) : null,
  })
  const objective = (x: Value): Value => {
    const { b, logA } = parts(x)
    const marginal = sum(logsumexp(joint(b, logA), 1))
    let penalty: Value = mul(0.5 / (difficultySd * difficultySd), sum(square(b)))
    if (logA) penalty = add(penalty, mul(0.5 / (logDiscriminationSd * logDiscriminationSd), sum(square(logA))))
    return sub(penalty, marginal)
  }
  const vg = valueAndGrad(objective)
  const scalar = (v: Value) => {
    const u = unwrap(v)
    return typeof u === 'number' ? u : toFlat(u as Tensor)[0]
  }
  const result = minimize(
    (x) => {
      const { value, grad } = vg(x)
      return { value: scalar(value), grad: unwrap(grad as Value) as Tensor }
    },
    new Float64Array(dim),
    { method: 'lbfgs', maxSteps },
  )
  const x = toFlat(result.x)
  const difficulty = Float64Array.from(x.slice(0, I))
  const discrimination = twoPl ? Float64Array.from(x.slice(I, 2 * I), Math.exp) : new Float64Array(I).fill(1)
  // Abilities by their posterior means (EAP) on the grid, and the marginal log-likelihood.
  const fitted = parts(result.x)
  const J = toFlat(unwrap(joint(fitted.b, fitted.logA)) as Tensor)
  const ability = new Float64Array(P)
  let logLikelihood = 0
  for (let p = 0; p < P; p++) {
    let m = -Infinity
    for (let q = 0; q < nodes; q++) m = Math.max(m, J[p * nodes + q])
    let z = 0
    let t = 0
    for (let q = 0; q < nodes; q++) {
      const w = Math.exp(J[p * nodes + q] - m)
      z += w
      t += w * grid[q]
    }
    ability[p] = t / z
    logLikelihood += m + Math.log(z)
  }
  return { model, ability, difficulty, discrimination, logLikelihood, converged: result.converged }
}
