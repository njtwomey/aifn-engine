/**
 * The transductive (semi-supervised) support vector machine solved in the primal by gradient descent with
 * continuation, the $\nabla$TSVM of Chapelle and Zien (2005, §2.2 and Algorithm 1), and the unlabelled losses of
 * Collobert et al. (2006, figure 1).
 *
 * On features $\phivec(\xvec)$ of length $d$ the decision function is $f(\xvec) = \wvec^\top\phivec(\xvec) + b$
 * and the objective is
 *
 * $$
 * \frac{1}{2}\lVert \wvec \rVert^2 + C\sum_{i=1}^{\ell} \max(0, 1 - y_i f(\xvec_i))^2 + C^*\sum_{j=1}^{u}
 * \exp(-s f(\xvec_j)^2),
 * $$
 *
 * the squared hinge on the labelled points ($y_i = \pm 1$) and the symmetric sigmoid $\exp(-s t^2)$ ($s = 3$), a
 * smooth stand-in for the symmetric hinge $\max(0, 1 - \lvert t \rvert)$, on the unlabelled points. The balancing
 * constraint $\frac{1}{u}\sum_j f(\xvec_j) = r$, with $r$ the mean labelled label by default, is imposed exactly by
 * substitution: $b = r - \wvec^\top\bar\phivec_U$ with $\bar\phivec_U$ the mean unlabelled feature vector, so the
 * objective is a function of $\wvec$ alone. Each stage minimises it by L-BFGS from the previous stage's $\wvec$,
 * with $C^*_k = 2^{k - K} C^*$ for $k = 0, \dots, K$ ($K = 10$ in the paper).
 *
 * The supervised baseline is the squared-hinge SVM with a free, unpenalised bias (`balance: null`, $C^* = 0$); it
 * differs from `linearSvm` of `aifn-methods/learning/kernel-methods`, which uses the hinge and penalises its bias.
 * Chapelle and Zien run conjugate gradients on kernel PCA coordinates; any feature map can be passed here (random
 * Fourier features approximate their RBF kernel). Pure functions of plain arrays, so a worker can run them.
 */

import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { minimize } from 'aifn-compute/optim/minimize'
import { packRows, type FeatureRows } from './features'

export type { FeatureRows } from './features'

const OP = 'transductiveSvm'

/**
 * The symmetric hinge $\max(0, 1 - \lvert t \rvert)$, the TSVM's loss on an unlabelled output $t$.
 *
 * @param t The decision value $f(\xvec)$ of an unlabelled point.
 * @returns The loss, 1 at $t = 0$ and 0 for $\lvert t \rvert \ge 1$.
 *
 * @example Zero outside the margin
 * print([-2, -0.5, 0, 0.5, 2].map((t) => symmetricHinge(t)))
 */
export function symmetricHinge(t: number): number {
  return Math.max(0, 1 - Math.abs(t))
}

/**
 * The symmetric sigmoid $\exp(-s t^2)$ of Chapelle and Zien (default $s = 3$), a smooth symmetric hinge.
 *
 * @param t The decision value of an unlabelled point.
 * @param s The sharpness $s > 0$.
 * @returns The loss, 1 at $t = 0$, falling smoothly on both sides.
 *
 * @example Beside the symmetric hinge
 * for (const t of [0, 0.5, 1]) print('t =', t, ' sigmoid:', symmetricSigmoid(t), ' hinge:', symmetricHinge(t))
 */
export function symmetricSigmoid(t: number, s = 3): number {
  return Math.exp(-s * t * t)
}

/**
 * The symmetric ramp $\min(1 + s, \max(0, 1 - \lvert t \rvert))$ of Collobert et al. (default $s = -0.3$): the
 * symmetric hinge clipped to a plateau of width $2\lvert s \rvert$, the loss that CCCP-TSVM minimises.
 *
 * @param t The decision value of an unlabelled point.
 * @param s The ramp parameter $s \in (-1, 0]$; the plateau sits at height $1 + s$.
 * @returns The loss.
 *
 * @example Flat near the boundary
 * print([0, 0.2, 0.5, 1].map((t) => symmetricRamp(t)))
 */
export function symmetricRamp(t: number, s = -0.3): number {
  return Math.min(1 + s, Math.max(0, 1 - Math.abs(t)))
}

/** A transductive SVM problem on fixed features. */
export interface TransductiveSvmProblem {
  /** Features of the labelled points, $\ell$ rows of length $d$. */
  labelled: FeatureRows
  /** Labels of the labelled points, $\pm 1$. */
  labels: ArrayLike<number>
  /** Features of the unlabelled points, $u$ rows of length $d$. */
  unlabelled: FeatureRows
}

/** The weights of one objective. */
export interface TransductiveSvmWeights {
  /** The cost $C > 0$ of the labelled squared hinge. */
  C: number
  /** The cost $C^* \ge 0$ of the unlabelled symmetric sigmoid. */
  Cstar: number
  /** The sharpness $s > 0$ of $\exp(-s t^2)$ (default 3). */
  sharpness?: number
  /**
   * The target mean output $r$ on the unlabelled points (default the mean labelled label); `null` frees the bias
   * instead (no balancing constraint; the bias is not penalised).
   */
  balance?: number | null
}

/** One solution. */
export interface TransductiveSvmSolution {
  /** The cost $C^*$ it was fitted with. */
  Cstar: number
  /** The weights $\wvec$ ($d$ values). */
  weights: Float64Array
  /** The bias $b$. */
  bias: number
  /** The objective at the solution. */
  objective: number
  /** $\sum_i \max(0, 1 - y_i f(\xvec_i))^2$. */
  labelledLoss: number
  /** $\sum_j \exp(-s f(\xvec_j)^2)$. */
  unlabelledLoss: number
  /** $f(\xvec_j)$ on each unlabelled point. */
  unlabelledOutputs: Float64Array
  /** L-BFGS steps taken. */
  steps: number
  /** Whether L-BFGS reached its gradient tolerance. */
  converged: boolean
}

/**
 * The objective as a function of the parameters (the weights $\wvec$, then the bias when `balance` is `null`), with
 * its closed-form gradient, and functions that recover the bias and split the objective into its parts. With the
 * balancing constraint every feature vector is centred on the unlabelled mean, so
 * $f(\xvec) = \wvec^\top(\phivec(\xvec) - \bar\phivec_U) + r$. Throws `DomainError` for no labelled points, labels
 * other than $\pm 1$, $C \le 0$ or $C^* < 0$, or a constraint with no unlabelled points, and `ShapeError` when the
 * labels do not match the labelled rows or a row has the wrong length.
 *
 * @param problem The labelled features and $\pm 1$ labels, and the unlabelled features.
 * @param weights $C$, $C^*$, the sharpness $s$ and the balance $r$.
 * @returns `size` (the parameter count), `dim` ($d$), `value(params)` (the objective with `grad`, the form the
 *   `aifn-compute/optim/minimize` methods take), `bias(params)` (in the uncentred features) and
 *   `solution(params, steps, converged)`.
 *
 * @example The objective and its gradient at zero weights
 * const problem = { labelled: [[0], [4.2]], labels: [-1, 1], unlabelled: [[1], [2], [3]] }
 * const f = transductiveSvmObjective(problem, { C: 1, Cstar: 0.5 })
 * print('parameters:', f.size, ' at w = 0:', f.value([0]))
 * print('bias at w = 1:', f.bias([1]))
 */
export function transductiveSvmObjective(problem: TransductiveSvmProblem, weights: TransductiveSvmWeights) {
  const nl = problem.labelled.length
  const nu = problem.unlabelled.length
  if (nl === 0) throw new DomainError(OP, `${OP}: no labelled points`)
  if (problem.labels.length !== nl) throw new ShapeError(OP, `${OP}: labels do not match the labelled rows`)
  const y = Array.from(problem.labels)
  if (y.some((v) => v !== 1 && v !== -1)) throw new DomainError(OP, `${OP}: labels must be ±1`)
  if (!(weights.C > 0) || !(weights.Cstar >= 0)) throw new DomainError(OP, `${OP}: need C > 0 and C* ≥ 0`)
  const dim = problem.labelled[0].length
  const s = weights.sharpness ?? 3
  const free = weights.balance === null
  if (!free && nu === 0) throw new DomainError(OP, `${OP}: the balancing constraint needs unlabelled points`)
  const r = free ? 0 : (weights.balance ?? y.reduce((a, v) => a + v, 0) / nl)
  const xl = packRows(problem.labelled, dim, OP)
  const xu = packRows(problem.unlabelled, dim, OP)
  // With the constraint, centre every feature vector on the unlabelled mean: f(x) = w·(φ(x) − m) + r.
  const m = new Float64Array(dim)
  if (!free) {
    for (let j = 0; j < nu; j++) for (let c = 0; c < dim; c++) m[c] += xu[j * dim + c] / nu
    for (let i = 0; i < nl; i++) for (let c = 0; c < dim; c++) xl[i * dim + c] -= m[c]
    for (let j = 0; j < nu; j++) for (let c = 0; c < dim; c++) xu[j * dim + c] -= m[c]
  }
  const size = dim + (free ? 1 : 0)
  const offset = (p: ArrayLike<number>) => (free ? p[dim] : r)
  const output = (x: Float64Array, row: number, p: ArrayLike<number>) => {
    let f = offset(p)
    for (let c = 0; c < dim; c++) f += x[row * dim + c] * p[c]
    return f
  }

  const evaluate = (p: ArrayLike<number>, withGrad: boolean) => {
    const grad = withGrad ? new Float64Array(size) : null
    let reg = 0
    for (let c = 0; c < dim; c++) {
      reg += 0.5 * p[c] * p[c]
      if (grad) grad[c] = p[c]
    }
    let labelledLoss = 0
    for (let i = 0; i < nl; i++) {
      const slack = Math.max(0, 1 - y[i] * output(xl, i, p))
      labelledLoss += slack * slack
      if (grad && slack > 0) {
        const g = -2 * weights.C * slack * y[i]
        for (let c = 0; c < dim; c++) grad[c] += g * xl[i * dim + c]
        if (free) grad[dim] += g
      }
    }
    let unlabelledLoss = 0
    const outputs = new Float64Array(nu)
    for (let j = 0; j < nu; j++) {
      const f = output(xu, j, p)
      outputs[j] = f
      const e = Math.exp(-s * f * f)
      unlabelledLoss += e
      if (grad && weights.Cstar > 0) {
        const g = weights.Cstar * e * (-2 * s * f)
        for (let c = 0; c < dim; c++) grad[c] += g * xu[j * dim + c]
        if (free) grad[dim] += g
      }
    }
    const objective = reg + weights.C * labelledLoss + weights.Cstar * unlabelledLoss
    return { objective, labelledLoss, unlabelledLoss, outputs, grad }
  }

  // The bias b in terms of the original (uncentred) features.
  const bias = (p: ArrayLike<number>) => {
    let b = offset(p)
    for (let c = 0; c < dim; c++) b -= p[c] * m[c]
    return b
  }

  return {
    size,
    dim,
    value: (p: ArrayLike<number>) => {
      const e = evaluate(p, true)
      return { value: e.objective, grad: e.grad as Float64Array }
    },
    bias,
    solution: (p: ArrayLike<number>, steps: number, converged: boolean): TransductiveSvmSolution => {
      const e = evaluate(p, false)
      return {
        Cstar: weights.Cstar,
        weights: Float64Array.from({ length: dim }, (_, c) => p[c]),
        bias: bias(p),
        objective: e.objective,
        labelledLoss: e.labelledLoss,
        unlabelledLoss: e.unlabelledLoss,
        unlabelledOutputs: e.outputs,
        steps,
        converged,
      }
    },
  }
}

/**
 * Minimises one objective by L-BFGS (`minimize` of `aifn-compute/optim/minimize`) from `start` (default zeros). With
 * `balance: null` and $C^* = 0$ it is the supervised squared-hinge SVM with a free, unpenalised bias, the baseline
 * the stages are compared with. Throws as `transductiveSvmObjective` does, and `ShapeError` for a start of the wrong
 * length. Plain arrays in and out, so a worker can run it.
 *
 * @param problem The labelled features and $\pm 1$ labels, and the unlabelled features.
 * @param options The weights, the step budget `maxSteps` (default 500), the gradient `tolerance` (default
 *   $10^{-6}$) and a `start` (the parameter vector: $\wvec$, then the bias when `balance` is `null`).
 * @returns The solution.
 *
 * @example The balancing constraint holds exactly at the solution
 * const problem = { labelled: [[0], [4.2]], labels: [-1, 1], unlabelled: [[0.4], [1.2], [2], [2.4], [3.4], [3.8]] }
 * const sol = transductiveSvmFit(problem, { C: 10, Cstar: 1 })
 * print('w:', sol.weights, ' b:', sol.bias, ' threshold:', -sol.bias / sol.weights[0])
 * print('mean unlabelled output:', sol.unlabelledOutputs.reduce((a, v) => a + v, 0) / 6)
 */
export function transductiveSvmFit(
  problem: TransductiveSvmProblem,
  options: TransductiveSvmWeights & { maxSteps?: number; tolerance?: number; start?: ArrayLike<number> },
): TransductiveSvmSolution {
  const f = transductiveSvmObjective(problem, options)
  const x0 = options.start ? Array.from(options.start) : new Array(f.size).fill(0)
  if (x0.length !== f.size)
    throw new ShapeError('transductiveSvmFit', `transductiveSvmFit: start has ${x0.length} values, expected ${f.size}`)
  const result = minimize((x: Tensor) => f.value(toFlat(x)), x0, {
    method: 'lbfgs',
    maxSteps: options.maxSteps ?? 500,
    tolerance: options.tolerance ?? 1e-6,
  })
  return f.solution(toFlat(result.x), result.steps, result.converged)
}

/**
 * The continuation of $\nabla$TSVM: stage $k = 0, \dots, K$ minimises the balanced objective with
 * $C^*_k = 2^{k - K} C^*$, starting from stage $k - 1$'s weights (stage 0 from zeros). The bias is always
 * constrained here (a numeric `balance`, or the mean labelled label), since the start is the weights alone.
 *
 * @param problem The labelled features and $\pm 1$ labels, and the unlabelled features.
 * @param options $C$, the final $C^*$, the number of doublings $K$ (default 10), the sharpness, the balance, and the
 *   step budget per stage.
 * @returns The $K + 1$ solutions in order of increasing $C^*$.
 *
 * @example The threshold moves into the gap as the unlabelled weight grows
 * const line = [0.4, 0.8, 1.2, 1.6, 2, 2.4, 3.4, 3.8]
 * const problem = { labelled: [[0], [4.2]], labels: [-1, 1], unlabelled: line.map((x) => [x]) }
 * const stages = transductiveSvmStages(problem, { C: 10, Cstar: 1, doublings: 4 })
 * print('C*:', stages.map((s) => s.Cstar))
 * print('threshold:', stages.map((s) => -s.bias / s.weights[0]))
 */
export function transductiveSvmStages(
  problem: TransductiveSvmProblem,
  options: Omit<TransductiveSvmWeights, 'balance'> & {
    balance?: number
    doublings?: number
    maxSteps?: number
    tolerance?: number
  },
): TransductiveSvmSolution[] {
  const K = options.doublings ?? 10
  const out: TransductiveSvmSolution[] = []
  let start: Float64Array | undefined
  for (let k = 0; k <= K; k++) {
    const sol = transductiveSvmFit(problem, { ...options, Cstar: options.Cstar * 2 ** (k - K), start })
    out.push(sol)
    start = sol.weights
  }
  return out
}

/**
 * The decision values $f(\xvec) = \wvec^\top\phivec(\xvec) + b$ of feature rows under a solution.
 *
 * @param rows Feature rows, each of length $d$.
 * @param solution The weights and bias.
 * @returns One decision value per row; its sign is the predicted class.
 *
 * @example Decision values of a hand-made solution
 * print(transductiveSvmDecision([[0], [2.9], [4]], { weights: Float64Array.from([1]), bias: -2.9 }))
 */
export function transductiveSvmDecision(
  rows: FeatureRows,
  solution: Pick<TransductiveSvmSolution, 'weights' | 'bias'>,
): Float64Array {
  return Float64Array.from(rows, (r) => {
    let f = solution.bias
    for (let c = 0; c < solution.weights.length; c++) f += solution.weights[c] * r[c]
    return f
  })
}
