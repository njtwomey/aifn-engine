/**
 * Item response theory (Rasch, 1960; Birnbaum in Lord and Novick, 1968): person $p$ answers item $i$ correctly with
 * probability $P = c + (1 - c)\,\sigma(a_i(\theta_p - b_i))$, with ability $\theta$, difficulty $b$, discrimination
 * $a$ (1 in the one-parameter Rasch model) and guessing floor $c$ (0 here unless given). The item information
 * $a_i^2 P(1 - P)$ (for $c = 0$) is the Fisher information an item carries about $\theta$; it peaks at $\theta = b$.
 *
 * `fitIrt` estimates the item parameters by marginal maximum likelihood (Bock and Aitkin, 1981): abilities
 * $\theta \sim \Gauss(0, 1)$ are integrated out on a quadrature grid, so the number of parameters does not grow with
 * the persons (joint maximum likelihood is inconsistent and inflates the discriminations). Weak Gaussian priors
 * $b \sim \Gauss(0, 2^2)$ and $\log a \sim \Gauss(0, 0.5^2)$ keep items everyone (or no one) answers finite. The
 * marginal log-likelihood is written with tensor primitives, differentiated by autodiff and maximised by L-BFGS;
 * abilities are then their posterior means (EAP).
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

/**
 * The probability of a correct answer, $P = c + (1 - c)\,\sigma(a(\theta - b))$, with $\sigma$ the logistic
 * function.
 *
 * @param theta The person's ability $\theta$.
 * @param a The item's discrimination $a$: the slope of the curve at $\theta = b$ (1 in the Rasch model).
 * @param b The item's difficulty $b$: the ability at which $P$ is halfway between $c$ and 1.
 * @param c The guessing floor $c$: the probability of a correct answer at very low ability.
 * @returns The probability, in $(c, 1)$.
 *
 * @example At the item's difficulty, with and without guessing
 * print('theta = b:', irtProbability(0.5, 1.2, 0.5))
 * print('theta = b, c = 0.25:', irtProbability(0.5, 1.2, 0.5, 0.25))
 * print('theta two above b:', irtProbability(2.5, 1.2, 0.5))
 */
export function irtProbability(theta: number, a: number, b: number, c = 0): number {
  return c + (1 - c) / (1 + Math.exp(-a * (theta - b)))
}

/**
 * The Fisher information of one item about $\theta$: $a^2 (P - c)^2 (1 - P) / ((1 - c)^2 P)$, which is
 * $a^2 P(1 - P)$ when $c = 0$, with $P$ from `irtProbability`.
 *
 * @param theta The ability $\theta$ at which the information is measured.
 * @param a The item's discrimination $a$.
 * @param b The item's difficulty $b$.
 * @param c The item's guessing floor $c$.
 * @returns The information, non-negative.
 *
 * @example The information peaks at the item's difficulty
 * print('theta = -1, 0, 1, 2:', [-1, 0, 1, 2].map((t) => itemInformation(t, 1.5, 1)))
 * print('peak, a^2 / 4:', 1.5 ** 2 / 4)
 */
export function itemInformation(theta: number, a: number, b: number, c = 0): number {
  const p = irtProbability(theta, a, b, c)
  return (a * a * (p - c) * (p - c) * (1 - p)) / ((1 - c) * (1 - c) * p)
}

/** Options of `fitIrt`. */
export interface IrtFitOptions {
  /** `1pl` (Rasch: every a = 1) or `2pl` (default `2pl`). */
  model?: '1pl' | '2pl'
  /** The prior standard deviation of the ability $\theta$, which sets the quadrature grid (default 1). */
  abilitySd?: number
  /** The prior standard deviation of each difficulty $b$ (default 2). */
  difficultySd?: number
  /** The prior standard deviation of each $\log a$ (default 0.5; unused by `1pl`). */
  logDiscriminationSd?: number
  /** Quadrature nodes over $\theta$, evenly spaced on $\pm 5$ prior standard deviations (default 41). */
  nodes?: number
  /** Most L-BFGS steps (default 500). */
  maxSteps?: number
}

/** A fitted IRT model. */
export interface IrtFit {
  /** The model fitted. */
  readonly model: '1pl' | '2pl'
  /** Abilities $\theta$, one per person: posterior means given the fitted items (EAP). */
  readonly ability: Float64Array
  /** Difficulties $b$, one per item. */
  readonly difficulty: Float64Array
  /** Discriminations $a$, one per item (all 1 for `1pl`). */
  readonly discrimination: Float64Array
  /** The marginal log-likelihood of the responses at the fit ($\theta$ integrated out; without the item priors). */
  readonly logLikelihood: number
  /** Whether L-BFGS converged within `maxSteps`. */
  readonly converged: boolean
}

/**
 * Fit a 1PL or 2PL model to responses by marginal maximum a posteriori over the item parameters (L-BFGS from
 * $b = 0$, $a = 1$), then score each person by the posterior mean of $\theta$. The guessing floor is not fitted.
 *
 * @param responses The answers as a $\text{persons} \times \text{items}$ tensor of 0 (wrong) and 1 (right); NaN
 *   marks an item a person did not answer, which then contributes nothing.
 * @param options The model, the priors, the number of quadrature nodes and the step limit.
 * @returns The item parameters, the abilities, the marginal log-likelihood and whether the optimiser converged.
 *
 * @example Recover four difficulties from 100 simulated persons
 * // Rasch items of difficulty -1, 0, 1 and 2; abilities and answers drawn from stream(0) and stream(1).
 * const b = [-1, 0, 1, 2]
 * const theta = toArray(normals(stream(0), [100]))
 * const u = toArray(uniform(stream(1), 0, 1, { shape: [100, 4] }))
 * const responses = tensor(u.map((row, p) => row.map((ui, i) => (ui < irtProbability(theta[p], 1, b[i]) ? 1 : 0))))
 * const fit = fitIrt(responses, { model: '1pl' })
 * print('difficulties:', fit.difficulty)
 * // In the Rasch model a person's ability depends only on how many items they got right.
 * const right = toArray(responses).map((row) => row.reduce((n, x) => n + x, 0))
 * print('ability by number right (0 to 4):', [0, 1, 2, 3, 4].map((k) => fit.ability[right.indexOf(k)]))
 * print('converged:', fit.converged)
 *
 * @example Unanswered items are skipped
 * // Person 0 did not answer item 2.
 * const fit = fitIrt(tensor([[1, 1, NaN], [1, 0, 0], [0, 0, 0], [1, 1, 1]]), { model: '1pl' })
 * print('difficulties:', fit.difficulty)
 * print('abilities:', fit.ability)
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
