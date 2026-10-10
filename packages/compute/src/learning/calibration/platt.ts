/**
 * Platt scaling (Platt, 1999, "Probabilistic outputs for support vector machines and comparisons to regularized
 * likelihood methods"): class probabilities from a classifier's real-valued scores $f$ by a fitted sigmoid
 * $P(y = 1 \mid f) = 1/(1 + \exp(Af + B))$. $A$ and $B$ minimise the cross-entropy against regularised targets
 * $t_+ = (N_+ + 1)/(N_+ + 2)$ for the $N_+$ positives and $t_- = 1/(N_- + 2)$ for the $N_-$ negatives, which keeps the
 * fit finite on separable scores. The minimiser is Newton's method with a backtracking line search, in the numerically
 * stable form of Lin, Lin and Weng (2007, "A note on Platt's probabilistic outputs for support vector machines",
 * Machine Learning 68, Algorithm 1), as LIBSVM uses.
 */

import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { sigmoid, softplus } from 'aifn-compute/numerics/special'
import { DomainError, ShapeError } from 'aifn-compute/foundation/errors'

/** A fitted Platt sigmoid. */
export interface PlattScaling {
  /** The slope $A$ (negative when larger scores mean the positive class). */
  readonly A: number
  /** The offset $B$. */
  readonly B: number
  /** Newton iterations taken. */
  readonly iterations: number
  /** The gradient fell below `tolerance`; false when the iteration limit or a failed line search stopped it. */
  readonly converged: boolean
  /** $P(y = 1 \mid f) = 1/(1 + \exp(Af + B))$ for each of $m$ scores $f$, as a vector of $m$ values. */
  probability(scores: Tensor): Tensor
}

/** Options of `plattScaling`. */
export interface PlattOptions {
  /** Newton iterations at most (default 100). */
  maxIterations?: number
  /** Stop when both gradient components are below this (default 1e-5). */
  tolerance?: number
}

/**
 * Fit Platt's sigmoid to scores and binary labels, from $A = 0$ and $B = \log((N_- + 1)/(N_+ + 1))$. Throws
 * `ShapeError` when the scores and labels differ in length, `DomainError` when a score is not finite. Not converging
 * is reported in the result (`converged`), not thrown.
 *
 * @param scores The classifier's real-valued scores $f_i$ on $n$ held-out cases (read as a flat vector).
 * @param labels The $n$ labels: 1 is positive and anything else negative, so $\pm 1$ and 0/1 both work.
 * @param options The iteration limit and gradient tolerance of Newton's method.
 * @returns The fitted $A$ and $B$, how the fit ended, and `probability`, the map from new scores to $P(y = 1)$.
 *
 * @example Overlapping scores: larger scores are more often positive
 * const platt = plattScaling(tensor([-2, -1, 0, 1, 2, 3]), tensor([0, 0, 1, 0, 1, 1]))
 * print('A =', platt.A, 'B =', platt.B, 'converged =', platt.converged)
 * print('P(y = 1) at -2, 0, 2:', platt.probability(tensor([-2, 0, 2])))
 *
 * @example Separable scores still give a finite fit
 * // The targets 3/4 and 1/4 (two cases of each class) stop the sigmoid from becoming a step.
 * const platt = plattScaling(tensor([-2, -1, 1, 2]), tensor([-1, -1, 1, 1]))
 * print('A =', platt.A, 'B =', platt.B, 'iterations =', platt.iterations)
 * print('P(y = 1) at the training scores:', platt.probability(tensor([-2, -1, 1, 2])))
 */
export function plattScaling(scores: Tensor, labels: Tensor, options: PlattOptions = {}): PlattScaling {
  const { maxIterations = 100, tolerance = 1e-5 } = options
  const f = dense.data(scores)
  const y = dense.data(labels)
  const n = f.length
  if (y.length !== n) throw new ShapeError('plattScaling', `plattScaling: ${n} scores but ${y.length} labels`)
  if (!f.every(Number.isFinite)) throw new DomainError('plattScaling', 'plattScaling: scores must be finite')
  let positives = 0
  for (const v of y) if (v === 1) positives++
  const negatives = n - positives
  const hi = (positives + 1) / (positives + 2)
  const lo = 1 / (negatives + 2)
  const t = Float64Array.from(y, (v) => (v === 1 ? hi : lo))
  // The cross-entropy Σ tᵢzᵢ + log(1 + e^{−zᵢ}) with zᵢ = A fᵢ + B (stable for either sign of z).
  const objective = (A: number, B: number) => {
    let s = 0
    for (let i = 0; i < n; i++) {
      const z = A * f[i] + B
      s += t[i] * z + (softplus(-z) as number)
    }
    return s
  }
  let A = 0
  let B = Math.log((negatives + 1) / (positives + 1))
  let value = objective(A, B)
  let iterations = 0
  let converged = false
  for (; iterations < maxIterations; iterations++) {
    // Gradient and Hessian; the Hessian's diagonal gets 1e-12 so it stays positive definite.
    let h11 = 1e-12
    let h22 = 1e-12
    let h21 = 0
    let g1 = 0
    let g2 = 0
    for (let i = 0; i < n; i++) {
      const z = A * f[i] + B
      const p = sigmoid(-z) as number // P(y = 1)
      const d2 = p * (1 - p)
      h11 += f[i] * f[i] * d2
      h22 += d2
      h21 += f[i] * d2
      const d1 = t[i] - p
      g1 += f[i] * d1
      g2 += d1
    }
    if (Math.abs(g1) < tolerance && Math.abs(g2) < tolerance) {
      converged = true
      break
    }
    const det = h11 * h22 - h21 * h21
    const dA = -(h22 * g1 - h21 * g2) / det
    const dB = -(-h21 * g1 + h11 * g2) / det
    const slope = g1 * dA + g2 * dB
    let step = 1
    let accepted = false
    for (; step >= 1e-10; step /= 2) {
      const next = objective(A + step * dA, B + step * dB)
      if (next < value + 1e-4 * step * slope) {
        A += step * dA
        B += step * dB
        value = next
        accepted = true
        break
      }
    }
    if (!accepted) break
  }
  const [slopeA, offsetB] = [A, B]
  return {
    A: slopeA,
    B: offsetB,
    iterations,
    converged,
    probability: (s: Tensor) => {
      const v = dense.data(s)
      return fromData(
        Float64Array.from(v, (x) => sigmoid(-(slopeA * x + offsetB)) as number),
        [v.length],
      )
    },
  }
}
