/**
 * Backfitting (Buja, Hastie and Tibshirani, 1989, Annals of Statistics 17; Hastie and Tibshirani, 1990, "Generalized
 * Additive Models", §4.4) with local scoring for non-Gaussian families (§6.5), as a traceable algorithm over any
 * additive model η = α + Σⱼ Bⱼβⱼ with a quadratic penalty per term: each step updates the working response and weights
 * from the current linear predictor, then refits each term's penalised smoother to its partial residual in turn,
 * holding the others fixed. Shared by the generalised models; `gam` builds the term designs and penalties.
 */

import type { Status } from 'aifn-compute/foundation/contracts'
import { dense, fromData, type Tensor } from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { cholesky, choleskySolve } from 'aifn-compute/numerics/linalg'
import {
  checkLink,
  gaussianFamily,
  likelihood,
  type Family,
  type Link,
  type LinkName,
} from 'aifn-compute/probability/likelihoods'

/** The problem a backfitting run solves: one design block and one penalty per term. */
export type BackfitProblem = {
  /** Each term's design Bⱼ [n, pⱼ] (centred, so the intercept is identifiable). */
  designs: readonly Tensor[]
  /** Each term's penalty Sⱼ [pⱼ, pⱼ], smoothing parameters already applied. */
  penalties: readonly Tensor[]
  y: Tensor
  weights?: Tensor
  /** Default the Gaussian family. */
  family?: Family
  /** Default the family's default link. */
  link?: LinkName | Link
  /** Converged when the largest change of any fⱼ(xᵢ) is below tolerance × (1 + max |f|) (default 1e-8). */
  tolerance?: number
}

/** A backfitting state. */
export type BackfitState = Status & {
  /** Sweeps done. */
  t: number
  intercept: number
  /** Each term's fitted values at the data, [terms, n]. */
  contributions: Tensor
  /** Each term's coefficients βⱼ [pⱼ] (zeros at the start). */
  coefficients: Tensor[]
  /** Linear predictor η = α + Σ fⱼ, [n]. */
  eta: Tensor
  /** Deviance at η. */
  deviance: number
  /** Largest change of a term's fitted value in this sweep. */
  change: number
  /** Halvings of this sweep towards the previous state, to keep the mean inside the family's mean space. */
  halvings: number
  converged: boolean
}

/** Backfitting as an `Algorithm` (no start: α begins at the weighted mean of g(μ₀), every fⱼ at 0). */
export function backfitting(problem: BackfitProblem): Algorithm<void, BackfitState> {
  const family = problem.family ?? gaussianFamily()
  const lk = checkLink(family, problem.link, 'backfitting')
  const y = dense.data(problem.y)
  const n = y.length
  const w = problem.weights ? dense.data(problem.weights) : new Float64Array(n).fill(1)
  const tolerance = problem.tolerance ?? 1e-8
  const designs = problem.designs.map((B) => dense.data(B))
  const sizes = problem.designs.map((B) => B.shape[1])
  const penalties = problem.penalties.map((S) => dense.data(S))
  const T = designs.length
  // The deviance from η, stable where μ rounds to the edge of the mean space (`likelihood`).
  const lik = likelihood(family, lk.name)
  const deviance = (eta: Float64Array) => {
    const u = dense.data(lik.unitDeviance(problem.y, fromData(eta, [n])) as Tensor)
    return u.reduce((s, v, i) => s + w[i] * v, 0)
  }
  const stateOf = (
    alpha: number,
    f: Float64Array,
    betas: Float64Array[],
    t: number,
    change: number,
    scale: number,
  ): BackfitState => {
    const eta = new Float64Array(n).fill(alpha)
    for (let j = 0; j < T; j++) for (let i = 0; i < n; i++) eta[i] += f[j * n + i]
    const dev = deviance(eta)
    return {
      t,
      intercept: alpha,
      contributions: fromData(f, [T, n]),
      coefficients: betas.map((b) => fromData(b, [b.length])),
      eta: fromData(eta, [n]),
      deviance: dev,
      change,
      halvings: 0,
      converged: t > 0 && change < tolerance * (1 + scale),
      diverged: !Number.isFinite(dev),
    }
  }
  return {
    name: 'backfitting',
    init: () => {
      const mu0 = family.initialMean(problem.y, fromData(w, [n]))
      const eta0 = dense.data(lk.link(mu0) as Tensor)
      let alpha = 0
      let sw = 0
      for (let i = 0; i < n; i++) {
        alpha += w[i] * eta0[i]
        sw += w[i]
      }
      return stateOf(
        alpha / sw,
        new Float64Array(T * n),
        sizes.map((p) => new Float64Array(p)),
        0,
        Infinity,
        0,
      )
    },
    step: (state) => {
      const eta = dense.data(state.eta)
      const etaT = fromData(Float64Array.from(eta), [n])
      const mu = dense.data(lk.inverse(etaT) as Tensor)
      const dmu = dense.data(lk.derivative(etaT) as Tensor)
      const V = dense.data(family.variance(fromData(Float64Array.from(mu), [n])) as Tensor)
      // Working response and weights (for the Gaussian identity model, z = y and W = w).
      // An observation whose μ has rounded to the edge of the mean space carries no weight in this sweep.
      const W = Float64Array.from(dmu, (g, i) => {
        const v = (w[i] * g * g) / V[i]
        return Number.isFinite(v) ? v : 0
      })
      const z = Float64Array.from(eta, (e, i) => (W[i] > 0 ? e + (y[i] - mu[i]) / dmu[i] : e))
      const f = Float64Array.from(dense.data(state.contributions))
      const betas = state.coefficients.map((b) => Float64Array.from(dense.data(b)))
      let alpha = 0
      let sw = 0
      for (let i = 0; i < n; i++) {
        let r = z[i]
        for (let j = 0; j < T; j++) r -= f[j * n + i]
        alpha += W[i] * r
        sw += W[i]
      }
      alpha /= sw
      let change = 0
      let scale = 0
      for (let j = 0; j < T; j++) {
        const B = designs[j]
        const p = sizes[j]
        const A = Float64Array.from(penalties[j])
        const b = new Float64Array(p)
        for (let i = 0; i < n; i++) {
          let r = z[i] - alpha
          for (let q = 0; q < T; q++) if (q !== j) r -= f[q * n + i]
          for (let a = 0; a < p; a++) {
            const v = B[i * p + a] * W[i]
            if (v === 0) continue
            b[a] += v * r
            for (let c = 0; c < p; c++) A[a * p + c] += v * B[i * p + c]
          }
        }
        const beta = Float64Array.from(
          dense.data(choleskySolve(cholesky(fromData(A, [p, p])).L, fromData(b, [p])) as Tensor),
        )
        betas[j] = beta
        for (let i = 0; i < n; i++) {
          let v = 0
          for (let a = 0; a < p; a++) v += B[i * p + a] * beta[a]
          change = Math.max(change, Math.abs(v - f[j * n + i]))
          scale = Math.max(scale, Math.abs(v))
          f[j * n + i] = v
        }
      }
      // A sweep whose η leaves the mean space (a gamma model's inverse link overshooting zero) is halved towards the
      // previous state, as P-IRLS halves its steps; the change reported is the accepted one.
      const old = dense.data(state.contributions)
      const oldBetas = state.coefficients.map((q) => dense.data(q))
      let halvings = 0
      const invalid = () => {
        const etaNew = new Float64Array(n).fill(alpha)
        for (let j = 0; j < T; j++) for (let i = 0; i < n; i++) etaNew[i] += f[j * n + i]
        const muNew = lk.inverse(fromData(etaNew, [n])) as Tensor
        const valid = lk.total ? etaNew.every(Number.isFinite) : family.validMean(muNew)
        return !valid || !Number.isFinite(deviance(etaNew))
      }
      for (; halvings < 30 && invalid(); halvings++) {
        alpha = (alpha + state.intercept) / 2
        for (let k = 0; k < f.length; k++) f[k] = (f[k] + old[k]) / 2
        betas.forEach((b, j) => b.forEach((v, a) => (b[a] = (v + oldBetas[j][a]) / 2)))
        change /= 2
      }
      return { ...stateOf(alpha, f, betas, state.t + 1, change, scale), halvings }
    },
  }
}
