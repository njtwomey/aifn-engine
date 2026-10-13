/**
 * Least-angle regression and its lasso modification (Efron, Hastie, Johnstone and Tibshirani, 2004): the whole
 * piecewise-linear coefficient path, knot by knot, as scikit-learn's `lars_path`.
 *
 * From $\wvec = \mathbf{0}$, the features most correlated with the residual form the active set $\mathcal{A}$, and the
 * weights move along the equiangular direction, which keeps the active features' absolute correlations
 * $\lvert \xvec_j^\top\rvec \rvert$ tied as they fall. A knot is where an inactive feature's correlation catches up
 * and it joins, or, in the lasso modification, where an active weight reaches zero and that feature leaves. With
 * $C$ the shared correlation at a knot, its penalty is $\alpha = C / n$, the scaling of the lasso in `elasticNet`, so
 * the lasso path's knots lie on `elasticNetPath`'s curves. The direction solves $\Gmat_{\mathcal{A}}\uvec = \svec$
 * ($\Gmat = \Xmat^\top\Xmat$ on the active set, $\svec$ the signs of the correlations) by Cholesky.
 */

import { cholesky, choleskySolve } from 'aifn-compute/numerics/linalg'
import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import { DomainError } from 'aifn-compute/foundation/errors'
import { centre, gram } from './elasticNet'

/** Options of `leastAngleRegression`. */
export interface LeastAngleOptions {
  /**
   * `'lar'` (default): features only join. `'lasso'`: a weight that reaches zero leaves, which gives the lasso
   * path.
   */
  method?: 'lar' | 'lasso'
  /** Centre the data, for an unpenalised intercept (default true). */
  intercept?: boolean
  /** The most knots after the first (default 500). */
  maxSteps?: number
}

/** A change of the active set at a knot. */
export interface LeastAngleEvent {
  /** The knot, an index into `alphas`. */
  readonly knot: number
  /** The feature (column of `x`). */
  readonly feature: number
  /** `'enter'`: its weight is zero at the knot and moves after it. `'drop'`: its weight reaches zero at the knot. */
  readonly kind: 'enter' | 'drop'
}

/** The least-angle path: the knots, the weights at each, and the changes of the active set. */
export interface LeastAnglePath {
  /** The penalty $\alpha = C / n$ at each knot, decreasing from $\alpha_{\max}$ to 0 at the least-squares fit. */
  readonly alphas: number[]
  /** The weights at the knots, $d \times k$; between knots the path is linear. */
  readonly coef: Tensor
  /** $\norm{\wvec}_1$ at each knot, the usual horizontal axis of a path plot. */
  readonly l1Norm: number[]
  /** The joins and departures, in order. */
  readonly events: LeastAngleEvent[]
  /** The intercept at each knot ($\bar y - \bar{\xvec}^\top\wvec$; 0 without an intercept). */
  readonly intercept: number[]
}

/**
 * Least-angle regression (Efron, Hastie, Johnstone and Tibshirani, 2004), as scikit-learn's `lars_path`: the knots of
 * the piecewise-linear path from $\wvec = \mathbf{0}$ to the least-squares fit, one feature joining per knot. With
 * `method: 'lasso'` a weight that crosses zero is set to zero and its feature leaves the active set, and the path is
 * then exactly the lasso path; a feature that leaves may join again later. `alphas` are in scikit-learn's scaling,
 * the largest absolute correlation over $n$. On data whose active columns become linearly dependent (more features
 * than rows), the path stops at the last knot it can solve. Throws `DomainError` for an unknown `method`, and
 * `ShapeError` for mismatched shapes.
 *
 * @param x The design, $n \times d$; standardise it first, as the method compares correlations.
 * @param y The target, $n$ values.
 * @param options The method, whether to centre for an intercept, and the most knots, as `LeastAngleOptions`.
 * @returns The path: penalties, weights ($d \times k$), $\ell_1$ norms and intercepts at the knots, and the events.
 *
 * @example The order in which three features join
 * const x = normals(stream(0), [50, 3])
 * const y = tensor(toArray(x).map((r) => 4 * r[0] + 2 * r[1] + 0.5 * r[2]))
 * const path = leastAngleRegression(x, y)
 * print('alphas:', path.alphas)
 * print('events:', path.events.map((e) => `${e.kind} ${e.feature} at knot ${e.knot}`))
 * print('final weights:', toArray(path.coef).map((row) => row.at(-1)))
 *
 * @example The lasso modification drops a weight that crosses zero
 * // x2 is nearly x0 + x1, so its weight overshoots and comes back to zero.
 * const z = toArray(normals(stream(1), [60, 3]))
 * const x = tensor(z.map((r) => [r[0], r[1], r[0] + r[1] + 0.3 * r[2]]))
 * const y = tensor(z.map((r) => 2 * r[0] + 2 * r[1] - 0.5 * r[2]))
 * for (const method of ['lar', 'lasso']) {
 *   const path = leastAngleRegression(x, y, { method })
 *   print(method, path.events.map((e) => `${e.kind} ${e.feature}@${e.knot}`))
 * }
 */
export function leastAngleRegression(x: Tensor, y: Tensor, options: LeastAngleOptions = {}): LeastAnglePath {
  const { method = 'lar', intercept = true, maxSteps = 500 } = options
  if (method !== 'lar' && method !== 'lasso')
    throw new DomainError('leastAngleRegression', `leastAngleRegression: unknown method '${String(method)}'`)
  const c = centre(x, y, intercept, 'leastAngleRegression')
  const { n, d, H: G, q } = { ...c, ...gram(c) }
  // The most features that can be active: the rank of a centred n-row design is at most n − 1.
  const most = Math.min(d, intercept ? n - 1 : n)
  const w = new Float64Array(d)
  const corr = (j: number): number => {
    let s = q[j]
    for (let k = 0; k < d; k++) s -= G[j * d + k] * w[k]
    return s
  }
  const active: number[] = []
  const alphas: number[] = []
  const knots: Float64Array[] = []
  const events: LeastAngleEvent[] = []
  let C = 0
  for (let j = 0; j < d; j++) C = Math.max(C, Math.abs(q[j]))
  const record = (): void => {
    alphas.push(C / n)
    knots.push(w.slice())
  }
  record()
  // The first feature to join: the most correlated (none when the target is uncorrelated with every feature).
  const scale = C
  let joining = -1
  if (C > 0 && most > 0) joining = q.findIndex((v) => Math.abs(v) === C)
  // A feature that has just left, whose correlation still ties at the knot where it left.
  let barred = -1
  for (let step = 0; step < maxSteps; step++) {
    if (joining >= 0) {
      active.push(joining)
      events.push({ knot: alphas.length - 1, feature: joining, kind: 'enter' })
      joining = -1
    }
    const m = active.length
    if (m === 0) break
    const signs = active.map((j) => Math.sign(corr(j)) || 1)
    const GA = new Float64Array(m * m)
    for (let a = 0; a < m; a++) for (let b = 0; b < m; b++) GA[a * m + b] = G[active[a] * d + active[b]]
    let u: Float64Array
    try {
      const { L } = cholesky(fromData(GA, [m, m]), { jitter: false })
      u = Float64Array.from(dense.data(choleskySolve(L, fromData(Float64Array.from(signs), [m]))))
    } catch {
      break
    }
    let su = 0
    for (let a = 0; a < m; a++) su += signs[a] * u[a]
    if (!(su > 0)) break
    // The equiangular direction δ = A u: a step γ along it lowers every active |correlation| by Aγ.
    const A = 1 / Math.sqrt(su)
    const delta = u.map((v) => v * A)
    // (Gδ)_j: how fast each correlation changes along the direction.
    const along = new Float64Array(d)
    for (let j = 0; j < d; j++) for (let b = 0; b < m; b++) along[j] += G[j * d + active[b]] * delta[b]
    // The step to the least-squares fit on the active set, unless a feature joins or (lasso) a weight hits zero first.
    let gamma = C / A
    let next = -1
    if (m < most) {
      for (let j = 0; j < d; j++) {
        if (active.includes(j)) continue
        const cj = corr(j)
        for (const [num, den] of [
          [C - cj, A - along[j]],
          [C + cj, A + along[j]],
        ]) {
          // The feature that has just left still ties at this knot: not that tie, but a later crossing, re-enters it.
          if (j === barred && Math.abs(num) <= 1e-9 * C) continue
          const g = num / den
          if (g > 1e-12 * gamma && g < gamma) {
            gamma = g
            next = j
          }
        }
      }
    }
    let drop = -1
    if (method === 'lasso') {
      for (let a = 0; a < m; a++) {
        const wj = w[active[a]]
        if (wj === 0) continue
        const z = -wj / delta[a]
        if (z > 0 && z < gamma) {
          gamma = z
          drop = a
          next = -1
        }
      }
    }
    for (let a = 0; a < m; a++) w[active[a]] += gamma * delta[a]
    C = next < 0 && drop < 0 ? 0 : Math.max(0, C - gamma * A)
    barred = -1
    if (drop >= 0) {
      const j = active[drop]
      w[j] = 0
      active.splice(drop, 1)
      record()
      events.push({ knot: alphas.length - 1, feature: j, kind: 'drop' })
      barred = j
    } else {
      record()
      if (next < 0) break
      joining = next
    }
    if (C <= 1e-14 * scale) break
  }
  const k = alphas.length
  const coef = new Float64Array(d * k)
  const l1Norm: number[] = []
  const intercepts: number[] = []
  knots.forEach((wt, t) => {
    let l1 = 0
    let b = c.yMean
    for (let j = 0; j < d; j++) {
      coef[j * k + t] = wt[j]
      l1 += Math.abs(wt[j])
      b -= c.xMean[j] * wt[j]
    }
    l1Norm.push(l1)
    intercepts.push(b)
  })
  return { alphas, coef: fromData(coef, [d, k]), l1Norm, events, intercept: intercepts }
}
