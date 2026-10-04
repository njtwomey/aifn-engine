/**
 * Variational Bayesian Gaussian mixture by coordinate ascent (Bishop, 2006, §10.2; Attias, 2000). The model:
 * π ~ Dir(α₀), Λₖ ~ Wishart(W₀, ν₀), μₖ | Λₖ ~ N(m₀, (β₀Λₖ)⁻¹), zₙ ~ Cat(π), xₙ | zₙ = k ~ N(μₖ, Λₖ⁻¹); the family
 * q(Z)q(π)Πₖq(μₖ, Λₖ) with q(π) = Dir(α), q(μₖ, Λₖ) = N(mₖ, (βₖΛₖ)⁻¹)W(Λₖ | Wₖ, νₖ) and responsibilities rₙₖ.
 * With a small α₀, components the data do not need are emptied (their Nₖ → 0) and E[πₖ] → 0.
 */

import { inverse, logDet } from 'aifn-compute/numerics/linalg'
import type { Status } from 'aifn-compute/foundation/contracts'
import { integers } from 'aifn-compute/foundation/random'
import { digamma, logGamma } from 'aifn-compute/numerics/special'
import {
  fromData,
  isTensor,
  tensor,
  toFlat,
  type Matrix,
  type Tensor,
  type Vector,
} from 'aifn-compute/foundation/tensor'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import type { dense } from 'aifn-compute/foundation/tensor'
import { ShapeError } from 'aifn-compute/foundation/errors'

type F64 = dense.F64

const LOG_2PI = Math.log(2 * Math.PI)
const dg = (x: number) => digamma(x) as number
const lg = (x: number) => logGamma(x) as number

/** Prior hyperparameters (Bishop's notation). Defaults are data-scaled, see `caviGaussianMixture`. */
export type MixturePrior = {
  alpha0?: number
  beta0?: number
  /** m₀ (length D). */
  mean0?: ArrayLike<number>
  /** W₀ (D×D, row-major or rows). */
  W0?: Tensor | readonly (readonly number[])[]
  nu0?: number
}

/** The state of `caviGaussianMixture`. */
export type MixtureState = Status & {
  /** Coordinate sweeps done. */
  t: number
  /** Responsibilities rₙₖ (N×K). */
  responsibilities: Matrix
  /** Nₖ = Σₙ rₙₖ. */
  counts: Vector
  /** q(π) = Dir(α) and E[πₖ] = αₖ/Σα. */
  alpha: Vector
  weights: Vector
  /** q(μₖ, Λₖ): mₖ (K×D), βₖ, Wₖ (K×D×D), νₖ. */
  means: Matrix
  beta: Vector
  W: Tensor
  nu: Vector
  /** E[Λₖ]⁻¹ = (νₖWₖ)⁻¹ (K×D×D), a point summary of each component's covariance. */
  covariances: Tensor
  elbo: number
  elboChange: number
  converged: boolean
  diverged: boolean
}

type Params = { alpha: F64; beta: F64; m: F64; W: F64; nu: F64; counts: F64 }

function toMatrix(x: Tensor | readonly (readonly number[])[] | ArrayLike<number>): { data: F64; N: number; D: number } {
  if (!isTensor(x) && typeof (x as ArrayLike<unknown>)[0] === 'number') {
    const data = Float64Array.from(x as ArrayLike<number>)
    return { data, N: data.length, D: 1 }
  }
  const t = isTensor(x) ? x : tensor(x as number[][])
  if (t.shape.length === 1) return { data: Float64Array.from(toFlat(t)), N: t.shape[0], D: 1 }
  if (t.shape.length !== 2)
    throw new ShapeError('caviGaussianMixture', 'caviGaussianMixture: data must be N×D (or a vector for D = 1)')
  return { data: Float64Array.from(toFlat(t)), N: t.shape[0], D: t.shape[1] }
}

const mat = (a: F64, D: number) => fromData(a, [D, D])
const invD = (a: F64, D: number): F64 => Float64Array.from(toFlat(inverse(mat(a, D))))
const logDetD = (a: F64, D: number): number => logDet(mat(a, D)) as number

/** (x − m)ᵀ A (x − m). */
function quad(
  x: F64 | ArrayLike<number>,
  m: ArrayLike<number>,
  A: ArrayLike<number>,
  D: number,
  xo = 0,
  mo = 0,
  Ao = 0,
) {
  let s = 0
  for (let i = 0; i < D; i++)
    for (let j = 0; j < D; j++) s += (x[xo + i] - m[mo + i]) * A[Ao + i * D + j] * (x[xo + j] - m[mo + j])
  return s
}

/** ln B(W, ν), the Wishart normaliser (Bishop, 2006, eq. B.79). */
function logWishartB(logDetW: number, nu: number, D: number) {
  let s = 0
  for (let i = 1; i <= D; i++) s += lg((nu + 1 - i) / 2)
  return -(nu / 2) * logDetW - ((nu * D) / 2) * Math.LN2 - ((D * (D - 1)) / 4) * Math.log(Math.PI) - s
}

/**
 * CAVI for the Bayesian Gaussian mixture (Bishop, 2006, §10.2.1): the M-like step updates q(π) and q(μₖ, Λₖ) from
 * the responsibilities (eqs. 10.51–10.63), the E-like step updates rₙₖ ∝ exp(E[ln πₖ] + ½E[ln|Λₖ|] − D/(2βₖ) −
 * (νₖ/2)(xₙ − mₖ)ᵀWₖ(xₙ − mₖ)) (eqs. 10.46–10.67), and the ELBO is eqs. 10.70–10.77, which never decreases.
 * Defaults: α₀ = 1/K, β₀ = 1, m₀ the data mean, ν₀ = D, W₀ = (ν₀Σ̂)⁻¹ so that E[Λ] = Σ̂⁻¹ (Σ̂ the data covariance).
 * `x` is N×D, or N numbers for D = 1. `init` places the K means at distinct data points drawn from the stream and assigns each point to its nearest.
 */
export function caviGaussianMixture(
  x: Tensor | readonly (readonly number[])[] | ArrayLike<number>,
  K: number,
  prior: MixturePrior = {},
  options: { tolerance?: number } = {},
): Algorithm<void, MixtureState> {
  const { data: X, N, D } = toMatrix(x)
  const tolerance = options.tolerance ?? 1e-8
  const dataMean = new Float64Array(D)
  for (let n = 0; n < N; n++) for (let i = 0; i < D; i++) dataMean[i] += X[n * D + i] / N
  const cov = new Float64Array(D * D)
  for (let n = 0; n < N; n++)
    for (let i = 0; i < D; i++)
      for (let j = 0; j < D; j++) cov[i * D + j] += ((X[n * D + i] - dataMean[i]) * (X[n * D + j] - dataMean[j])) / N
  const alpha0 = prior.alpha0 ?? 1 / K
  const beta0 = prior.beta0 ?? 1
  const nu0 = prior.nu0 ?? D
  const m0 = prior.mean0 ? Float64Array.from(prior.mean0) : dataMean
  const W0 = prior.W0
    ? Float64Array.from(toFlat(isTensor(prior.W0) ? prior.W0 : tensor(prior.W0 as number[][])))
    : invD(
        cov.map((v) => v * nu0),
        D,
      )
  const W0inv = invD(W0, D)
  const logDetW0 = logDetD(W0, D)

  /** q(π), q(μ, Λ) from responsibilities (Bishop eqs. 10.51–10.53, 10.58, 10.60–10.63). */
  function mStep(r: F64): Params {
    const counts = new Float64Array(K)
    const xbar = new Float64Array(K * D)
    const S = new Float64Array(K * D * D)
    for (let n = 0; n < N; n++)
      for (let k = 0; k < K; k++) {
        counts[k] += r[n * K + k]
        for (let i = 0; i < D; i++) xbar[k * D + i] += r[n * K + k] * X[n * D + i]
      }
    for (let k = 0; k < K; k++) for (let i = 0; i < D; i++) xbar[k * D + i] /= counts[k] + 1e-300
    for (let n = 0; n < N; n++)
      for (let k = 0; k < K; k++)
        for (let i = 0; i < D; i++)
          for (let j = 0; j < D; j++)
            S[(k * D + i) * D + j] += r[n * K + k] * (X[n * D + i] - xbar[k * D + i]) * (X[n * D + j] - xbar[k * D + j])
    const alpha = counts.map((c) => alpha0 + c)
    const beta = counts.map((c) => beta0 + c)
    const nu = counts.map((c) => nu0 + c)
    const m = new Float64Array(K * D)
    const W = new Float64Array(K * D * D)
    for (let k = 0; k < K; k++) {
      for (let i = 0; i < D; i++) m[k * D + i] = (beta0 * m0[i] + counts[k] * xbar[k * D + i]) / beta[k]
      const Winv = new Float64Array(D * D)
      const c = (beta0 * counts[k]) / (beta0 + counts[k])
      for (let i = 0; i < D; i++)
        for (let j = 0; j < D; j++)
          Winv[i * D + j] =
            W0inv[i * D + j] + S[(k * D + i) * D + j] + c * (xbar[k * D + i] - m0[i]) * (xbar[k * D + j] - m0[j])
      W.set(invD(Winv, D), k * D * D)
    }
    return { alpha, beta, m, W, nu, counts }
  }

  const expectations = (p: Params) => {
    const alphaHat = p.alpha.reduce((a, b) => a + b, 0)
    const logPi = p.alpha.map((a) => dg(a) - dg(alphaHat))
    const logDetW = Float64Array.from({ length: K }, (_, k) => logDetD(p.W.slice(k * D * D, (k + 1) * D * D), D))
    const logLambda = Float64Array.from({ length: K }, (_, k) => {
      let s = D * Math.LN2 + logDetW[k]
      for (let i = 1; i <= D; i++) s += dg((p.nu[k] + 1 - i) / 2)
      return s
    })
    return { alphaHat, logPi, logDetW, logLambda }
  }

  /** Responsibilities (Bishop eqs. 10.46, 10.49, 10.64–10.67). */
  function eStep(p: Params): F64 {
    const { logPi, logLambda } = expectations(p)
    const r = new Float64Array(N * K)
    for (let n = 0; n < N; n++) {
      let max = -Infinity
      for (let k = 0; k < K; k++) {
        const v =
          logPi[k] +
          0.5 * logLambda[k] -
          D / (2 * p.beta[k]) -
          (p.nu[k] / 2) * quad(X, p.m, p.W, D, n * D, k * D, k * D * D)
        r[n * K + k] = v
        if (v > max) max = v
      }
      let s = 0
      for (let k = 0; k < K; k++) s += r[n * K + k] = Math.exp(r[n * K + k] - max)
      for (let k = 0; k < K; k++) r[n * K + k] /= s
    }
    return r
  }

  /** The ELBO (Bishop eqs. 10.70–10.77), with the statistics Nₖ, x̄ₖ, Sₖ recomputed from r. */
  function elboOf(p: Params, r: F64): number {
    const { alphaHat, logPi, logDetW, logLambda } = expectations(p)
    const counts = new Float64Array(K)
    const xbar = new Float64Array(K * D)
    for (let n = 0; n < N; n++)
      for (let k = 0; k < K; k++) {
        counts[k] += r[n * K + k]
        for (let i = 0; i < D; i++) xbar[k * D + i] += r[n * K + k] * X[n * D + i]
      }
    for (let k = 0; k < K; k++) for (let i = 0; i < D; i++) xbar[k * D + i] /= counts[k] + 1e-300
    let L = 0
    for (let k = 0; k < K; k++) {
      const Wk = p.W.subarray(k * D * D, (k + 1) * D * D)
      // Tr(Sₖ Wₖ) Nₖ = Σₙ rₙₖ (xₙ − x̄ₖ)ᵀ Wₖ (xₙ − x̄ₖ).
      let trSW = 0
      for (let n = 0; n < N; n++) if (r[n * K + k] > 0) trSW += r[n * K + k] * quad(X, xbar, Wk, D, n * D, k * D)
      const dev = quad(xbar, p.m, Wk, D, k * D, k * D)
      // 10.71
      L += 0.5 * (counts[k] * (logLambda[k] - D / p.beta[k] - p.nu[k] * dev - D * LOG_2PI) - p.nu[k] * trSW)
      // 10.72
      for (let n = 0; n < N; n++) L += r[n * K + k] * logPi[k]
      // 10.74
      let trW0W = 0
      for (let i = 0; i < D; i++) for (let j = 0; j < D; j++) trW0W += W0inv[i * D + j] * Wk[j * D + i]
      L +=
        0.5 *
          (D * Math.log(beta0 / (2 * Math.PI)) +
            logLambda[k] -
            (D * beta0) / p.beta[k] -
            beta0 * p.nu[k] * quad(p.m, m0, Wk, D, k * D, 0)) +
        logWishartB(logDetW0, nu0, D) +
        ((nu0 - D - 1) / 2) * logLambda[k] -
        0.5 * p.nu[k] * trW0W
      // −10.75
      for (let n = 0; n < N; n++) {
        const v = r[n * K + k]
        if (v > 0) L -= v * Math.log(v)
      }
      // −10.77, with H[q(Λₖ)] = −ln B(Wₖ, νₖ) − (νₖ − D − 1)/2 E[ln|Λₖ|] + νₖD/2.
      const entropyW = -logWishartB(logDetW[k], p.nu[k], D) - ((p.nu[k] - D - 1) / 2) * logLambda[k] + (p.nu[k] * D) / 2
      L -= 0.5 * logLambda[k] + (D / 2) * Math.log(p.beta[k] / (2 * Math.PI)) - D / 2 - entropyW
    }
    // 10.73 and −10.76: ln C(α) = ln Γ(Σα) − Σ ln Γ(αₖ).
    const logC0 = lg(K * alpha0) - K * lg(alpha0)
    let logC = lg(alphaHat)
    for (let k = 0; k < K; k++) {
      logC -= lg(p.alpha[k])
      L += (alpha0 - 1) * logPi[k] - (p.alpha[k] - 1) * logPi[k]
    }
    return L + logC0 - logC
  }

  const toState = (t: number, p: Params, r: F64, previous: number): MixtureState => {
    const value = elboOf(p, r)
    const alphaHat = p.alpha.reduce((a, b) => a + b, 0)
    const covariances = new Float64Array(K * D * D)
    for (let k = 0; k < K; k++)
      covariances.set(
        invD(
          p.W.slice(k * D * D, (k + 1) * D * D).map((v) => v * p.nu[k]),
          D,
        ),
        k * D * D,
      )
    const change = value - previous
    return {
      t,
      responsibilities: fromData(r, [N, K]),
      counts: fromData(p.counts, [K]),
      alpha: fromData(p.alpha, [K]),
      weights: fromData(
        p.alpha.map((a) => a / alphaHat),
        [K],
      ),
      means: fromData(p.m, [K, D]),
      beta: fromData(p.beta, [K]),
      W: fromData(p.W, [K, D, D]),
      nu: fromData(p.nu, [K]),
      covariances: fromData(covariances, [K, D, D]),
      elbo: value,
      elboChange: change,
      converged: Math.abs(change) < tolerance * Math.max(1, Math.abs(value)),
      diverged: !Number.isFinite(value),
    }
  }

  return {
    name: 'cavi-gaussian-mixture',
    init: (_start, st) => {
      // K distinct data points as centres (partial Fisher–Yates), then hard nearest-centre responsibilities.
      const order = Int32Array.from({ length: N }, (_, i) => i)
      for (let k = 0; k < Math.min(K, N); k++) {
        const j = k + integers(st, N - k)
        ;[order[k], order[j]] = [order[j], order[k]]
      }
      const r0 = new Float64Array(N * K)
      for (let n = 0; n < N; n++) {
        let best = 0
        let bestD = Infinity
        for (let k = 0; k < K; k++) {
          const c = order[k % N]
          let dist = 0
          for (let i = 0; i < D; i++) dist += (X[n * D + i] - X[c * D + i]) ** 2
          if (dist < bestD) [best, bestD] = [k, dist]
        }
        r0[n * K + best] = 1
      }
      const p = mStep(r0)
      return toState(0, p, eStep(p), NaN)
    },
    step: (s) => {
      const p = mStep(Float64Array.from(toFlat(s.responsibilities)))
      return toState(s.t + 1, p, eStep(p), s.elbo)
    },
  }
}

/**
 * The predictive density p(x̂ | X) of the variational mixture (Bishop, 2006, eq. 10.81): a mixture of Student t
 * distributions, (1/α̂) Σₖ αₖ St(x̂ | mₖ, Lₖ, νₖ + 1 − D) with precision Lₖ = ((νₖ + 1 − D)βₖ/(1 + βₖ))Wₖ. `points`
 * is M×D (or a vector of M points when D = 1); returns the M densities, and each component's weighted density (K×M).
 */
export function mixturePredictiveDensity(
  state: MixtureState,
  points: Tensor | readonly (readonly number[])[] | ArrayLike<number>,
): { density: Vector; components: Matrix } {
  const K = state.alpha.shape[0]
  const D = state.means.shape[1]
  const P = isTensor(points)
    ? Float64Array.from(toFlat(points))
    : typeof (points as ArrayLike<unknown>)[0] === 'number'
      ? Float64Array.from(points as ArrayLike<number>)
      : Float64Array.from((points as readonly (readonly number[])[]).flat())
  const M = P.length / D
  const alpha = toFlat(state.alpha)
  const alphaHat = alpha.reduce((a, b) => a + b, 0)
  const beta = toFlat(state.beta)
  const nu = toFlat(state.nu)
  const m = toFlat(state.means)
  const W = Float64Array.from(toFlat(state.W))
  const comps = new Float64Array(K * M)
  const density = new Float64Array(M)
  for (let k = 0; k < K; k++) {
    const dof = nu[k] + 1 - D
    const c = (dof * beta[k]) / (1 + beta[k])
    const Lk = W.subarray(k * D * D, (k + 1) * D * D).map((v) => v * c)
    const logDetL = logDetD(Float64Array.from(Lk), D)
    const logNorm = lg((dof + D) / 2) - lg(dof / 2) + 0.5 * logDetL - (D / 2) * Math.log(dof * Math.PI)
    for (let j = 0; j < M; j++) {
      const q = quad(P, m, Lk, D, j * D, k * D)
      const v = (alpha[k] / alphaHat) * Math.exp(logNorm - ((dof + D) / 2) * Math.log1p(q / dof))
      comps[k * M + j] = v
      density[j] += v
    }
  }
  return { density: fromData(density, [M]), components: fromData(comps, [K, M]) }
}
