/**
 * Label shift: $p(\xvec \mid y)$ is the same in both domains but the class priors differ. Two estimators of the
 * target priors from a classifier trained on the source:
 *
 * - **black-box shift estimation** (BBSE; Lipton, Wang and Smola, 2018): with $\Cmat$ the joint confusion matrix
 *   $C_{ij} = P_{\mathrm{source}}(\hat y = i, y = j)$ from held-out source data and $\muvec$ the target's
 *   distribution of predictions $\mu_i = P_{\mathrm{target}}(\hat y = i)$, the importance weights
 *   $w_j = p_{\mathrm{target}}(y = j) / p_{\mathrm{source}}(y = j)$ solve $\Cmat\wvec = \muvec$;
 * - **EM** (Saerens, Latinne and Decaestecker, 2002): alternate posteriors re-weighted by the prior ratio,
 *   $p_t(y \mid \xvec) \propto p_s(y \mid \xvec) \pi_t(y) / \pi_s(y)$, and priors $\pi_t$ as their mean over the
 *   target.
 *
 * The corrected posteriors re-weight the classifier's outputs by the estimated prior ratio. Classes are the integers
 * $0, \dots, k - 1$; labels are used as indices and not checked.
 */

import { dense, fromData, toFlat, type MatrixLike, type VectorLike } from 'aifn-compute/foundation/tensor'
import { solve } from 'aifn-compute/numerics/linalg'

/**
 * Black-box shift estimation: the importance weights $\wvec$ that solve $\Cmat\wvec = \muvec$ (clipped at 0), and the
 * target priors $\wvec \odot \pivec_{\mathrm{source}}$ renormalised to sum to 1, where $\pivec_{\mathrm{source}}$ is
 * the class frequency of `sourceTrue`. The linear solve throws when $\Cmat$ is singular (a class never predicted, or
 * never seen, on the source).
 *
 * @param sourceTrue The true labels of held-out source points, $n$ of them.
 * @param sourcePredicted The classifier's predicted labels for the same points.
 * @param targetPredicted The classifier's predicted labels for the target points.
 * @param classes The number of classes $k$.
 * @returns `weights` $\wvec$ ($k$ values), `priors` (the estimated target priors, $k$ values) and `confusion`
 *   ($\Cmat$, row-major $k \times k$, rows the predicted class and columns the true one).
 *
 * @example A classifier right 80% of the time on a balanced source predicts class 1 for 70% of the target
 * const sourceTrue = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
 * const sourcePredicted = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
 * const targetPredicted = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
 * const { weights, priors, confusion } = blackBoxShiftEstimate(sourceTrue, sourcePredicted, targetPredicted, 2)
 * print('confusion C:', confusion)
 * print('importance weights:', weights)
 * print('target priors:', priors)
 */
export function blackBoxShiftEstimate(
  sourceTrue: VectorLike,
  sourcePredicted: VectorLike,
  targetPredicted: VectorLike,
  classes: number,
): { weights: Float64Array; priors: Float64Array; confusion: Float64Array } {
  const yt = dense.toF64(sourceTrue, 'blackBoxShiftEstimate')
  const yp = dense.toF64(sourcePredicted, 'blackBoxShiftEstimate')
  const tp = dense.toF64(targetPredicted, 'blackBoxShiftEstimate')
  const k = classes
  const C = new Float64Array(k * k)
  const prior = new Float64Array(k)
  for (let i = 0; i < yt.length; i++) {
    C[yp[i] * k + yt[i]] += 1 / yt.length
    prior[yt[i]] += 1 / yt.length
  }
  const mu = new Float64Array(k)
  for (const v of tp) mu[v] += 1 / tp.length
  const w = Float64Array.from(toFlat(solve(fromData(C, [k, k]), fromData(mu, [k]))), (v) => Math.max(0, v))
  const p = Float64Array.from(w, (v, j) => v * prior[j])
  const z = p.reduce((a, b) => a + b, 0)
  return { weights: w, priors: p.map((v) => v / z), confusion: C }
}

/**
 * EM for the target priors from the source classifier's posteriors on target points: each iteration re-weights every
 * posterior by $\pi_t(y) / \pi_s(y)$, renormalises it, and takes the mean over the points as the new $\pi_t$, starting
 * from $\pi_t = \pi_s$. It stops when no prior moves by more than `tolerance`.
 *
 * @param posteriors The source posteriors $p_s(y \mid \xvec)$ on $n$ target points, $n \times k$.
 * @param sourcePriors The source priors $\pi_s$, $k$ values.
 * @param options The stopping rule.
 * @param options.maxIterations The most EM iterations.
 * @param options.tolerance Stop when the largest change of a prior is below this.
 * @returns `priors` (the estimated $\pi_t$), `iterations` (the iterations taken, or `maxIterations` when it did not
 *   stop) and `path` (the priors at the start and after each iteration).
 *
 * @example Five target points whose posteriors lean to class 1
 * const posteriors = [[0.9, 0.1], [0.6, 0.4], [0.3, 0.7], [0.2, 0.8], [0.1, 0.9]]
 * const { priors, iterations, path } = priorShiftEm(posteriors, [0.5, 0.5])
 * print('target priors:', priors, 'after', iterations, 'iterations')
 * print('the first steps:', path.slice(0, 3))
 */
export function priorShiftEm(
  posteriors: MatrixLike,
  sourcePriors: readonly number[],
  options: { maxIterations?: number; tolerance?: number } = {},
): { priors: Float64Array; iterations: number; path: Float64Array[] } {
  const { maxIterations = 500, tolerance = 1e-10 } = options
  const P = dense.toMatrixF64(posteriors, 'priorShiftEm')
  const { m: n, n: k } = P
  let pi = Float64Array.from(sourcePriors)
  const path = [pi]
  for (let it = 1; it <= maxIterations; it++) {
    const next = new Float64Array(k)
    for (let i = 0; i < n; i++) {
      let z = 0
      const row = new Float64Array(k)
      for (let c = 0; c < k; c++) z += row[c] = (P.data[i * k + c] * pi[c]) / sourcePriors[c]
      for (let c = 0; c < k; c++) next[c] += row[c] / z / n
    }
    let change = 0
    for (let c = 0; c < k; c++) change = Math.max(change, Math.abs(next[c] - pi[c]))
    pi = next
    path.push(pi)
    if (change < tolerance) return { priors: pi, iterations: it, path }
  }
  return { priors: pi, iterations: maxIterations, path }
}

/**
 * Posteriors re-weighted by the target-to-source prior ratios and renormalised:
 * $p_t(y \mid \xvec) \propto p_s(y \mid \xvec) \pi_t(y) / \pi_s(y)$.
 *
 * @param posteriors The source posteriors $p_s(y \mid \xvec)$, $n \times k$.
 * @param sourcePriors The source priors $\pi_s$, $k$ values.
 * @param targetPriors The target priors $\pi_t$, $k$ values (such as `priorShiftEm` or `blackBoxShiftEstimate`
 *   estimates).
 * @returns The corrected posteriors, row-major $n \times k$.
 *
 * @example Balanced-source posteriors moved towards a target where class 1 has prior 0.8
 * print(reweightPosteriors([[0.9, 0.1], [0.5, 0.5]], [0.5, 0.5], [0.2, 0.8]))
 */
export function reweightPosteriors(
  posteriors: MatrixLike,
  sourcePriors: readonly number[],
  targetPriors: ArrayLike<number>,
): Float64Array {
  const P = dense.toMatrixF64(posteriors, 'reweightPosteriors')
  const { m: n, n: k } = P
  const out = new Float64Array(n * k)
  for (let i = 0; i < n; i++) {
    let z = 0
    for (let c = 0; c < k; c++) z += out[i * k + c] = (P.data[i * k + c] * targetPriors[c]) / sourcePriors[c]
    for (let c = 0; c < k; c++) out[i * k + c] /= z
  }
  return out
}
