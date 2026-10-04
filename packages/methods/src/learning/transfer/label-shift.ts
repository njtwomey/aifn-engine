/**
 * Label shift: p(x | y) is the same in both domains but the class priors differ. Two estimators of the target priors
 * from a classifier trained on the source:
 *
 * - **black-box shift estimation** (BBSE; Lipton, Wang and Smola, 2018): with C the joint confusion matrix
 *   C_ij = P_source(ŷ = i, y = j) from held-out source data and μ the target's distribution of predictions
 *   μ_i = P_target(ŷ = i), the importance weights w = p_target(y)/p_source(y) solve C w = μ;
 * - **EM** (Saerens, Latinne and Decaestecker, 2002): alternate posteriors re-weighted by the prior ratio,
 *   p_t(y | x) ∝ p_s(y | x) π_t(y)/π_s(y), and priors π_t as their mean over the target.
 *
 * The corrected posteriors re-weight the classifier's outputs by the estimated prior ratio.
 */

import { dense, fromData, toFlat, type MatrixLike, type VectorLike } from 'aifn-compute/foundation/tensor'
import { solve } from 'aifn-compute/numerics/linalg'

/** BBSE: importance weights w (clipped at 0) and the target priors w ⊙ π_source. */
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

/** EM for the target priors from source posteriors on target points [n, k] and the source priors. */
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

/** Posteriors [n, k] re-weighted by target/source prior ratios and renormalised. */
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
